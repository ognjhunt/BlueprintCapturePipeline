"""NuRec-first swappable asset orchestration entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .blueprintpipeline_runner import (
    BlueprintPipelineRunner,
    BlueprintPipelineRunnerConfig,
    required_env_from_command_result,
)
from .capture_bridge import CaptureDescriptor
from .common import (
    PipelineError,
    StageError,
    ensure_dir,
    has_nonempty_file,
    infer_storage_root_from_scene_path,
    optional_read_json,
    parse_bool,
    parse_gs_uri,
    read_json,
    relative_scene_path,
    resolve_gs_uri_to_path,
    to_pipeline_prefix,
    utc_now_iso,
    write_json,
)
from .interactive_reconciliation import (
    find_required_articulation_failures,
    reconcile_interactive_results,
)
from .ios_manifest import load_object_index, load_raw_manifest, resolve_object_index_uri
from .manifest_builder import build_scene_artifacts
from .nurec_worker_client import NurecWorkerClient, NurecWorkerConfig
from .quality_gates import AdvancedQualityGateConfig, run_advanced_quality_gates
from .retrieval_fallback import enforce_hard_fail_if_unresolved, run_retrieval_fallback
from .runtime_preflight import enforce_preflight, validate_runtime_preflight
from .sam3d_assets import materialize_candidate_assets, materialize_scene_shell_assets
from .swap_candidates import build_swap_candidates_payload


@dataclass(frozen=True)
class OrchestratorConfig:
    gcs_root: Path = Path(os.getenv("GCS_ROOT", "/mnt/gcs"))
    blueprintpipeline_root: Path = Path(
        os.getenv("BLUEPRINTPIPELINE_ROOT", "/opt/BlueprintPipeline")
    )
    expected_blueprintpipeline_commit: str = os.getenv("BLUEPRINTPIPELINE_COMMIT_HASH", "")
    fail_on_commit_mismatch: bool = parse_bool(
        os.getenv("FAIL_ON_BLUEPRINTPIPELINE_COMMIT_MISMATCH"), default=True
    )
    nurec_timeout_seconds: int = int(os.getenv("NUREC_TIMEOUT_SECONDS", "14400") or "14400")
    nurec_poll_seconds: int = int(os.getenv("NUREC_POLL_SECONDS", "20") or "20")
    nurec_worker_mode: str = (os.getenv("NUREC_WORKER_MODE", "local_worker") or "local_worker").strip()
    nurec_worker_command: str = os.getenv("NUREC_WORKER_COMMAND", "").strip()
    runtime_preflight_enabled: bool = parse_bool(
        os.getenv("RUNTIME_PREFLIGHT_ENABLED"),
        default=True,
    )
    swap_policy_path: str = os.getenv("SWAP_POLICY_CONFIG_PATH", "").strip()
    generation_provider_chain: str = os.getenv(
        "TEXT_ASSET_GENERATION_PROVIDER_CHAIN", "sam3d,hunyuan3d"
    ).strip()
    image_conditioned_generation: bool = parse_bool(
        os.getenv("IMAGE_CONDITIONED_GENERATION_ENABLED"), default=True
    )
    crop_cleanup_provider: str = (
        os.getenv("CROP_CLEANUP_PROVIDER", "skip").strip().lower()
    )
    advanced_quality_config: AdvancedQualityGateConfig = field(
        default_factory=AdvancedQualityGateConfig
    )
    interactive_extra_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class _Gate:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


def _default_qa_report_uri(descriptor_uri: str) -> str:
    parsed = parse_gs_uri(descriptor_uri)
    if parsed.key.endswith("capture_descriptor.json"):
        qa_key = parsed.key[: -len("capture_descriptor.json")] + "qa_report.json"
    else:
        qa_key = f"{parsed.key.rstrip('/')}/qa_report.json"
    return f"gs://{parsed.bucket}/{qa_key}"


def _scene_prefix(scene_id: str) -> str:
    return f"scenes/{scene_id}"


def _assets_prefix(scene_id: str) -> str:
    return f"scenes/{scene_id}/assets"


def _layout_prefix(scene_id: str) -> str:
    return f"scenes/{scene_id}/layout"


def _seg_prefix(scene_id: str) -> str:
    return f"scenes/{scene_id}/seg"


def _usd_prefix(scene_id: str) -> str:
    return f"scenes/{scene_id}/usd"


def _required_articulation_ids(candidates: List[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    for candidate in candidates:
        articulation = (
            candidate.get("articulation")
            if isinstance(candidate.get("articulation"), Mapping)
            else {}
        )
        if bool(articulation.get("required", False)):
            out.append(str(candidate.get("object_id")))
    return out


def _asset_dir_for_candidate(candidate: Mapping[str, Any]) -> str:
    object_id = str(candidate.get("object_id"))
    return str(candidate.get("asset_dir") or f"obj_{object_id}")


def _validate_swap_assets(
    *,
    storage_root: Path,
    assets_prefix: str,
    candidates: List[Mapping[str, Any]],
) -> tuple[bool, str]:
    missing: List[str] = []
    for candidate in candidates:
        object_id = str(candidate.get("object_id"))
        asset_dir_name = _asset_dir_for_candidate(candidate)
        model_path = storage_root / assets_prefix / asset_dir_name / "model.usd"
        metadata_path = storage_root / assets_prefix / asset_dir_name / "metadata.json"
        if not has_nonempty_file(model_path) or not has_nonempty_file(metadata_path):
            missing.append(object_id)

    if missing:
        return False, f"missing assets for object IDs: {', '.join(missing)}"
    return True, "all swap assets materialized"


def _read_interactive_results(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise StageError("interactive", f"interactive results missing at {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise StageError("interactive", f"invalid interactive results payload type at {path}")
    return payload


def _synthesize_interactive_results_from_failure(
    *,
    scene_id: str,
    failed_required_ids: List[str],
    reason: str,
) -> Dict[str, Any]:
    objects = [
        {
            "id": obj_id,
            "status": "error",
            "backend": "interactive_failed_marker",
            "required_articulation": True,
            "is_articulated": False,
            "joint_count": 0,
        }
        for obj_id in sorted(set(failed_required_ids))
    ]
    return {
        "scene_id": scene_id,
        "objects": objects,
        "total_objects": len(objects),
        "ok_count": 0,
        "error_count": len(objects),
        "fallback_count": 0,
        "articulated_count": 0,
        "source": "interactive_failed_marker",
        "failure_reason": reason,
    }


def _write_pipeline_failure(
    *,
    pipeline_dir: Path,
    descriptor_uri: str,
    stage: str,
    error: Exception,
    gates: List[_Gate],
    debug: Optional[Mapping[str, Any]] = None,
) -> None:
    ensure_dir(pipeline_dir)
    payload: Dict[str, Any] = {
        "schema_version": "v1",
        "status": "failed",
        "stage": stage,
        "descriptor_uri": descriptor_uri,
        "error": str(error),
        "failed_at": utc_now_iso(),
        "gates": [gate.to_dict() for gate in gates],
        "traceback": traceback.format_exc(limit=20),
    }
    if debug:
        payload["debug"] = dict(debug)
    write_json(pipeline_dir / ".swap_pipeline_failed.json", payload)


def run_swap_pipeline(
    *,
    descriptor_gcs_uri: str,
    config: Optional[OrchestratorConfig] = None,
    nurec_client: Optional[NurecWorkerClient] = None,
    blueprint_runner: Optional[BlueprintPipelineRunner] = None,
) -> Dict[str, Any]:
    """Run the NuRec-first swappable-asset pipeline for one capture descriptor."""

    cfg = config or OrchestratorConfig()
    gates: List[_Gate] = []
    stage = "intake"
    debug: Dict[str, Any] = {}

    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root)
    storage_root = infer_storage_root_from_scene_path(descriptor_path)

    descriptor = CaptureDescriptor.from_file(descriptor_path)
    parsed_uri = parse_gs_uri(descriptor_gcs_uri)
    bucket = parsed_uri.bucket

    scene_id = descriptor.scene_id
    capture_id = descriptor.capture_id

    pipeline_prefix = to_pipeline_prefix(scene_id, capture_id)
    pipeline_dir = storage_root / pipeline_prefix
    ensure_dir(pipeline_dir)

    assets_prefix = _assets_prefix(scene_id)
    layout_prefix = _layout_prefix(scene_id)
    seg_prefix = _seg_prefix(scene_id)
    usd_prefix = _usd_prefix(scene_id)

    nurec_outputs: Dict[str, Any] = {}
    swap_candidates_payload: Dict[str, Any] = {}
    interactive_results_payload: Dict[str, Any] = {}
    fallback_payload: Dict[str, Any] = {}
    advanced_quality_report: Dict[str, Any] = {}
    runtime_preflight_report: Dict[str, Any] = {}

    try:
        # ------------------------------------------------------------------
        # Stage 0: runtime preflight
        # ------------------------------------------------------------------
        stage = "runtime_preflight"
        if cfg.runtime_preflight_enabled:
            checks = validate_runtime_preflight(
                gcs_root=storage_root,
                blueprintpipeline_root=cfg.blueprintpipeline_root,
                generation_provider_chain=cfg.generation_provider_chain,
                swap_policy_path=cfg.swap_policy_path,
                nurec_worker_mode=cfg.nurec_worker_mode,
                nurec_worker_command=cfg.nurec_worker_command,
                advanced_quality_gates_enabled=cfg.advanced_quality_config.enabled,
            )
            has_failures = any(not check.passed for check in checks)
            runtime_preflight_report = {
                "schema_version": "v1",
                "status": "failed" if has_failures else "passed",
                "generated_at": utc_now_iso(),
                "checks": [check.to_dict() for check in checks],
            }
            write_json(pipeline_dir / "runtime_preflight_report.json", runtime_preflight_report)
            enforce_preflight(checks)
            gates.append(_Gate("runtime_preflight_gate", True, "runtime preflight passed"))
        else:
            runtime_preflight_report = {
                "schema_version": "v1",
                "status": "skipped",
                "generated_at": utc_now_iso(),
                "detail": "runtime preflight disabled by configuration",
            }
            write_json(pipeline_dir / "runtime_preflight_report.json", runtime_preflight_report)
            gates.append(
                _Gate("runtime_preflight_gate", True, "runtime preflight skipped by configuration")
            )

        # ------------------------------------------------------------------
        # Stage A: intake
        # ------------------------------------------------------------------
        stage = "intake"
        qa_report_uri = descriptor.qa_report_uri or _default_qa_report_uri(descriptor_gcs_uri)
        qa_report_path = resolve_gs_uri_to_path(qa_report_uri, storage_root)
        qa_report = read_json(qa_report_path)
        qa_status = str(qa_report.get("status") or "").strip().lower()
        if qa_status != "passed":
            raise StageError("intake", f"qa_report status must be 'passed', got '{qa_status}'")

        manifest = load_raw_manifest(descriptor.raw_prefix_uri, gcs_root=storage_root)
        object_index_uri = resolve_object_index_uri(descriptor.raw_prefix_uri, manifest)
        if not object_index_uri:
            raise StageError("intake", "manifest missing object_point_cloud_index")

        object_index_entries = load_object_index(object_index_uri, gcs_root=storage_root)

        gates.append(_Gate("intake_gate", True, "qa passed and descriptor parsed"))

        # ------------------------------------------------------------------
        # Stage B: NuRec reconstruction
        # ------------------------------------------------------------------
        stage = "nurec"
        if nurec_client is None:
            nurec_client = NurecWorkerClient(
                storage_root=storage_root,
                bucket=bucket,
                pipeline_prefix=pipeline_prefix,
                config=NurecWorkerConfig(
                    timeout_seconds=cfg.nurec_timeout_seconds,
                    poll_interval_seconds=cfg.nurec_poll_seconds,
                    worker_mode=cfg.nurec_worker_mode,
                    worker_command=cfg.nurec_worker_command,
                ),
            )
        nurec_outputs = nurec_client.run(
            descriptor=descriptor,
            descriptor_uri=descriptor_gcs_uri,
            object_index_uri=object_index_uri,
        )

        has_required_nurec = bool(
            nurec_outputs
            and isinstance(nurec_outputs.get("artifacts"), Mapping)
            and nurec_outputs["artifacts"].get("visual_usdz")
            and nurec_outputs["artifacts"].get("collision_mesh_ply")
        )
        gates.append(
            _Gate(
                "nurec_gate",
                has_required_nurec,
                "required NuRec artifacts found" if has_required_nurec else "required NuRec artifacts missing",
            )
        )
        if not has_required_nurec:
            raise StageError("nurec", "required NuRec artifacts missing from nurec_outputs")

        # ------------------------------------------------------------------
        # Stage C: swap candidate selection
        # ------------------------------------------------------------------
        stage = "swap_candidates"
        swap_candidates_payload = build_swap_candidates_payload(
            descriptor=descriptor,
            object_index_entries=object_index_entries,
            policy_path=cfg.swap_policy_path or None,
        )
        write_json(pipeline_dir / "swap_candidates.json", swap_candidates_payload)
        swap_candidates = swap_candidates_payload.get("candidates")
        if not isinstance(swap_candidates, list):
            raise StageError("swap_candidates", "invalid swap_candidates payload")

        # Conditionally strip or clean up reference crops
        if not cfg.image_conditioned_generation:
            for cand in swap_candidates:
                cand.pop("reference_crop", None)
                cand.pop("all_crops", None)
        elif cfg.crop_cleanup_provider != "skip":
            from .reference_image_utils import cleanup_crop_with_vlm
            for cand in swap_candidates:
                ref_crop = cand.get("reference_crop")
                if ref_crop and Path(str(ref_crop)).is_file():
                    crop_path = Path(str(ref_crop))
                    cleaned_path = crop_path.parent / f"{crop_path.stem}_cleaned.png"
                    result = cleanup_crop_with_vlm(
                        crop_path, cleaned_path, provider=cfg.crop_cleanup_provider
                    )
                    if result is not None and result != crop_path:
                        cand["reference_crop"] = str(result)

        # ------------------------------------------------------------------
        # Stage D: SAM3D-first materialization
        # ------------------------------------------------------------------
        stage = "sam3d"
        if blueprint_runner is None:
            blueprint_runner = BlueprintPipelineRunner(
                BlueprintPipelineRunnerConfig(
                    blueprintpipeline_root=cfg.blueprintpipeline_root,
                    gcs_root=storage_root,
                    bucket=bucket,
                )
            )

        if cfg.expected_blueprintpipeline_commit:
            try:
                blueprint_runner.ensure_expected_commit(cfg.expected_blueprintpipeline_commit)
            except StageError:
                if cfg.fail_on_commit_mismatch:
                    raise
                gates.append(_Gate("blueprintpipeline_commit", False, "commit mismatch ignored by config"))

        shell_assets = materialize_scene_shell_assets(
            storage_root=storage_root,
            assets_prefix=assets_prefix,
            nurec_outputs=nurec_outputs,
            swap_candidates=swap_candidates,
        )
        sam3d_report = materialize_candidate_assets(
            runner=blueprint_runner,
            storage_root=storage_root,
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            room_type=descriptor.environment_type_hint or "unknown",
            swap_candidates=swap_candidates,
            generation_provider_chain=cfg.generation_provider_chain,
        )

        swap_execution_report = {
            "schema_version": "v1",
            "scene_id": scene_id,
            "capture_id": capture_id,
            "policy": "sam3d_first",
            "generated_at": utc_now_iso(),
            "scene_shell": shell_assets,
            "sam3d": sam3d_report,
        }
        write_json(pipeline_dir / "swap_execution_report.json", swap_execution_report)

        swap_gate_ok, swap_gate_detail = _validate_swap_assets(
            storage_root=storage_root,
            assets_prefix=assets_prefix,
            candidates=swap_candidates,
        )
        gates.append(_Gate("swap_gate", swap_gate_ok, swap_gate_detail))
        if not swap_gate_ok:
            raise StageError("sam3d", swap_gate_detail)

        # ------------------------------------------------------------------
        # Stage E: manifest/layout synthesis
        # ------------------------------------------------------------------
        stage = "manifest"
        artifact_paths = build_scene_artifacts(
            storage_root=storage_root,
            scene_id=scene_id,
            capture_id=capture_id,
            descriptor=descriptor,
            descriptor_uri=descriptor_gcs_uri,
            nurec_outputs=nurec_outputs,
            swap_candidates=swap_candidates,
            assets_prefix=assets_prefix,
            layout_prefix=layout_prefix,
            seg_prefix=seg_prefix,
        )

        # ------------------------------------------------------------------
        # Stage F: interactive articulation validation
        # ------------------------------------------------------------------
        stage = "interactive"
        interactive_result = blueprint_runner.run_interactive_job(
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            regen3d_prefix=assets_prefix,
            extra_env=cfg.interactive_extra_env,
        )
        debug["interactive_job"] = required_env_from_command_result(interactive_result)

        required_ids = _required_articulation_ids(swap_candidates)
        interactive_results_path = storage_root / assets_prefix / "interactive" / "interactive_results.json"
        if interactive_results_path.is_file():
            interactive_results_payload = _read_interactive_results(interactive_results_path)
        else:
            failure_marker_path = storage_root / assets_prefix / ".interactive_failed"
            failure_marker = optional_read_json(failure_marker_path) or {}
            failure_reason = ""
            failure_required_ids: List[str] = []
            if isinstance(failure_marker, Mapping):
                failure_reason = str(failure_marker.get("reason") or "").strip()
                details = (
                    failure_marker.get("details")
                    if isinstance(failure_marker.get("details"), Mapping)
                    else {}
                )
                raw_required = details.get("required_objects") if isinstance(details, Mapping) else []
                if isinstance(raw_required, list):
                    failure_required_ids = [str(value) for value in raw_required if str(value).strip()]

            if interactive_result.return_code != 0 and failure_reason == "required_articulation_unmet":
                synthesized_failed_ids = failure_required_ids or required_ids
                interactive_results_payload = _synthesize_interactive_results_from_failure(
                    scene_id=scene_id,
                    failed_required_ids=synthesized_failed_ids,
                    reason=failure_reason,
                )
                write_json(interactive_results_path, interactive_results_payload)
                debug["interactive_synthesized_results"] = {
                    "reason": failure_reason,
                    "failure_marker": str(failure_marker_path),
                    "failed_required_ids": synthesized_failed_ids,
                }
            else:
                raise StageError(
                    "interactive",
                    (
                        "interactive results missing at "
                        f"{interactive_results_path}; return_code={interactive_result.return_code}; "
                        f"failure_reason={failure_reason or 'unknown'}"
                    ),
                )

        failed_required_ids = find_required_articulation_failures(
            interactive_results=interactive_results_payload,
            required_object_ids=required_ids,
        )

        # ------------------------------------------------------------------
        # Stage G: retrieval fallback
        # ------------------------------------------------------------------
        articulation_gate_passed = True
        articulation_detail = "all required articulated objects passed interactive validation"
        if failed_required_ids:
            stage = "retrieval_fallback"
            failed_candidates = [
                candidate
                for candidate in swap_candidates
                if str(candidate.get("object_id")) in set(failed_required_ids)
            ]

            fallback_payload = run_retrieval_fallback(
                runner=blueprint_runner,
                storage_root=storage_root,
                scene_id=scene_id,
                assets_prefix=assets_prefix,
                room_type=descriptor.environment_type_hint or "unknown",
                failed_candidates=failed_candidates,
            )
            write_json(pipeline_dir / "retrieval_fallback_report.json", fallback_payload)

            enforce_hard_fail_if_unresolved(fallback_payload)

            resolved_ids = fallback_payload.get("resolved_ids", [])
            if not isinstance(resolved_ids, list):
                resolved_ids = []
            interactive_results_payload = reconcile_interactive_results(
                interactive_results=interactive_results_payload,
                resolved_object_ids=resolved_ids,
            )
            write_json(interactive_results_path, interactive_results_payload)
            articulation_detail = (
                f"fallback resolved required articulation IDs: {', '.join(str(v) for v in resolved_ids)}"
                if resolved_ids
                else "fallback executed with no resolved IDs"
            )

        gates.append(_Gate("articulation_gate", articulation_gate_passed, articulation_detail))

        # ------------------------------------------------------------------
        # Stage H: simready + USD assembly
        # ------------------------------------------------------------------
        stage = "assembly"
        simready_result = blueprint_runner.run_simready_job(
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            layout_prefix=layout_prefix,
            usd_prefix=usd_prefix,
        )
        debug["simready_job"] = required_env_from_command_result(simready_result)
        if simready_result.return_code != 0:
            raise StageError(
                "simready",
                f"prepare_simready_assets.py failed with code {simready_result.return_code}",
            )

        usd_result = blueprint_runner.run_usd_assembly_job(
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            layout_prefix=layout_prefix,
            usd_prefix=usd_prefix,
        )
        debug["usd_assembly_job"] = required_env_from_command_result(usd_result)
        if usd_result.return_code != 0:
            raise StageError("usd_assembly", f"assemble_scene.py failed with code {usd_result.return_code}")

        scene_usda_path = storage_root / usd_prefix / "scene.usda"
        assembly_gate_ok = has_nonempty_file(scene_usda_path)
        gates.append(
            _Gate(
                "assembly_gate",
                assembly_gate_ok,
                "scene.usda exists" if assembly_gate_ok else f"missing {scene_usda_path}",
            )
        )
        if not assembly_gate_ok:
            raise StageError("usd_assembly", f"scene.usda missing at {scene_usda_path}")

        # ------------------------------------------------------------------
        # Stage I: quality + completion
        # ------------------------------------------------------------------
        stage = "quality_gates"
        advanced_quality_report = run_advanced_quality_gates(
            storage_root=storage_root,
            assets_prefix=assets_prefix,
            nurec_outputs=nurec_outputs,
            config=cfg.advanced_quality_config,
        )
        write_json(pipeline_dir / "advanced_quality_report.json", advanced_quality_report)
        advanced_status = str(advanced_quality_report.get("status") or "").strip().lower()
        advanced_ok = advanced_status in {"passed", "skipped"}
        gates.append(
            _Gate(
                "advanced_quality_gate",
                advanced_ok,
                f"advanced quality status={advanced_status or 'unknown'}",
            )
        )
        if not advanced_ok:
            raise StageError(
                "quality_gates",
                f"advanced quality gates failed: {advanced_quality_report.get('gates', [])}",
            )

        stage = "completion"
        quality_report = {
            "schema_version": "v1",
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "passed",
            "generated_at": utc_now_iso(),
            "gates": [gate.to_dict() for gate in gates],
            "artifacts": {
                "descriptor_uri": descriptor_gcs_uri,
                "qa_report_uri": qa_report_uri,
                "runtime_preflight_report": f"gs://{bucket}/{pipeline_prefix}/runtime_preflight_report.json",
                "nurec_outputs": f"gs://{bucket}/{pipeline_prefix}/nurec_outputs.json",
                "swap_candidates": f"gs://{bucket}/{pipeline_prefix}/swap_candidates.json",
                "swap_execution_report": f"gs://{bucket}/{pipeline_prefix}/swap_execution_report.json",
                "advanced_quality_report": f"gs://{bucket}/{pipeline_prefix}/advanced_quality_report.json",
                "manifest": f"gs://{bucket}/{relative_scene_path(artifact_paths['manifest_path'], storage_root)}",
                "layout": f"gs://{bucket}/{relative_scene_path(artifact_paths['layout_path'], storage_root)}",
                "inventory": f"gs://{bucket}/{relative_scene_path(artifact_paths['inventory_path'], storage_root)}",
                "scene_usda": f"gs://{bucket}/{relative_scene_path(scene_usda_path, storage_root)}",
            },
        }
        write_json(pipeline_dir / "swap_quality_report.json", quality_report)

        completion_payload = {
            "schema_version": "v1",
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "completed",
            "completed_at": utc_now_iso(),
            "quality_report": f"gs://{bucket}/{pipeline_prefix}/swap_quality_report.json",
        }
        write_json(pipeline_dir / ".swap_pipeline_complete", completion_payload)

        return {
            "status": "completed",
            "scene_id": scene_id,
            "capture_id": capture_id,
            "pipeline_prefix": pipeline_prefix,
            "quality_report": quality_report,
        }

    except Exception as exc:
        if isinstance(exc, StageError):
            stage = exc.stage
        _write_pipeline_failure(
            pipeline_dir=pipeline_dir,
            descriptor_uri=descriptor_gcs_uri,
            stage=stage,
            error=exc,
            gates=gates,
            debug=debug,
        )

        quality_report = {
            "schema_version": "v1",
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "failed",
            "generated_at": utc_now_iso(),
            "failed_stage": stage,
            "error": str(exc),
            "gates": [gate.to_dict() for gate in gates],
        }
        write_json(pipeline_dir / "swap_quality_report.json", quality_report)
        raise


def _startup_checks() -> List[str]:
    """Quick startup sanity checks before accepting work. Returns list of errors."""
    errors: List[str] = []
    cfg = OrchestratorConfig()

    if not cfg.gcs_root.exists():
        errors.append(
            f"GCS_ROOT={cfg.gcs_root} does not exist. "
            "Mount your GCS bucket (gcsfuse, GCS FUSE CSI, or symlink) or set GCS_ROOT."
        )
    if not cfg.blueprintpipeline_root.exists():
        errors.append(
            f"BLUEPRINTPIPELINE_ROOT={cfg.blueprintpipeline_root} does not exist. "
            "Clone https://github.com/ognjhunt/BlueprintPipeline.git to that path "
            "or set BLUEPRINTPIPELINE_ROOT to the correct location."
        )
    elif not (cfg.blueprintpipeline_root / "tools" / "source_pipeline" / "adapter.py").is_file():
        errors.append(
            f"BLUEPRINTPIPELINE_ROOT={cfg.blueprintpipeline_root} exists but is missing "
            "required scripts (tools/source_pipeline/adapter.py). Is the repo complete?"
        )

    # Check critical API credentials
    provider_chain = cfg.generation_provider_chain.lower()
    if "sam3d" in provider_chain:
        host = os.getenv("TEXT_SAM3D_API_HOST") or os.getenv("SAM3D_API_HOST") or ""
        key = os.getenv("TEXT_SAM3D_API_KEY") or os.getenv("SAM3D_API_KEY") or ""
        if not host.strip() or not key.strip():
            errors.append(
                "SAM3D credentials missing. Set TEXT_SAM3D_API_HOST and TEXT_SAM3D_API_KEY "
                "(or SAM3D_API_HOST / SAM3D_API_KEY)."
            )

    # Check NuRec worker config
    nurec_cmd = (os.getenv("NUREC_PIPELINE_COMMAND") or "").strip()
    nurec_skip = (os.getenv("NUREC_SKIP_PIPELINE_COMMAND") or "").strip().lower() in {"1", "true", "yes"}
    if cfg.nurec_worker_mode == "local_worker" and not nurec_cmd and not nurec_skip:
        errors.append(
            "NUREC_PIPELINE_COMMAND is not set. Example:\n"
            '  export NUREC_PIPELINE_COMMAND="python3 /app/scripts/nurec_shim.py '
            '--job-spec {JOB_SPEC_PATH} --output-dir {NUREC_OUTPUT_DIR} '
            '--raw-prefix {RAW_PREFIX_URI}"\n'
            "Or set NUREC_SKIP_PIPELINE_COMMAND=true if NuRec artifacts are pre-generated."
        )

    return errors


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run NuRec-first swappable asset pipeline")
    parser.add_argument(
        "--descriptor-gcs-uri",
        required=True,
        help="gs:// URI for capture_descriptor.json",
    )
    parser.add_argument(
        "--skip-startup-checks",
        action="store_true",
        help="Skip early environment validation (preflight still runs)",
    )
    args = parser.parse_args(argv)

    if not args.skip_startup_checks:
        errors = _startup_checks()
        if errors:
            print("[swap-orchestrator] STARTUP FAILED — environment not ready:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            return 1

    try:
        run_swap_pipeline(descriptor_gcs_uri=args.descriptor_gcs_uri)
    except PipelineError as exc:
        print(f"[swap-orchestrator] FAILED: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - safety net
        print(f"[swap-orchestrator] FAILED (unexpected): {exc}")
        return 1

    print("[swap-orchestrator] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
