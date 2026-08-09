"""Seal an observed AuraFusion360 InteriorGS execution without self-admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_aura_adapter import SCHEMA_VERSION as ADAPTER_SCHEMA_VERSION


SCHEMA_VERSION = "adp009b_aurafusion360_execution_receipt.v1"
RUNTIME_SCHEMA_VERSION = "adp_aura_interiorgs_result.v1"
RUN_SCHEMA_VERSION = "adp_aura_interiorgs_vast_run.v1"
REQUIRED_STAGES = (
    "reference_lama",
    "train",
    "render",
    "remove",
    "sam2_masks",
    "inpaint_init",
    "sdedit",
    "inpaint_finetune",
)


class AuraExecutionReceiptError(ValueError):
    """A deterministic Aura execution-evidence validation failure."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: str | Path, roots: Sequence[Path], code: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise AuraExecutionReceiptError([code])
    return resolved


def _read_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraExecutionReceiptError([code]) from exc
    if not isinstance(value, dict):
        raise AuraExecutionReceiptError([code])
    return value


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_record(
    record: Mapping[str, Any], *, root: Path, code: str
) -> tuple[Path, dict[str, Any]]:
    canonical_relative = str(record.get("relative_path") or "")
    legacy_relative = str(record.get("path") or "")
    if canonical_relative and legacy_relative:
        raise AuraExecutionReceiptError([code])
    relative = canonical_relative or legacy_relative
    path = (root / relative).resolve()
    if not relative or (path != root and root not in path.parents):
        raise AuraExecutionReceiptError([code])
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise AuraExecutionReceiptError([code])
    return path, _file_record(path, root)


def materialize_aura_execution_receipt(
    *,
    adapter_receipt_path: str | Path,
    runtime_result_path: str | Path,
    run_result_path: str | Path,
    evidence_root: str | Path,
    repo_root: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Derive a candidate receipt from actual runtime files and provider teardown."""

    evidence = Path(evidence_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    if not evidence.is_dir() or not repo.is_dir():
        raise AuraExecutionReceiptError(["aurafusion360_execution_allowlisted_root_missing"])
    roots = (evidence, repo)
    adapter_path = _under(
        adapter_receipt_path, roots, "aurafusion360_adapter_receipt_outside_allowlisted_roots"
    )
    runtime_path = _under(
        runtime_result_path, (evidence,), "aurafusion360_runtime_result_outside_evidence_root"
    )
    run_path = _under(
        run_result_path, (evidence,), "aurafusion360_run_result_outside_evidence_root"
    )
    adapter = _read_object(adapter_path, "aurafusion360_adapter_receipt_invalid")
    runtime = _read_object(runtime_path, "aurafusion360_runtime_result_invalid")
    run = _read_object(run_path, "aurafusion360_run_result_invalid")

    if (
        adapter.get("schema_version") != ADAPTER_SCHEMA_VERSION
        or adapter.get("status") != "prepared_unexecuted"
        or canonical_digest(adapter, digest_field="receipt_digest")
        != adapter.get("receipt_digest")
    ):
        raise AuraExecutionReceiptError(["aurafusion360_adapter_receipt_invalid"])
    if (
        runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION
        or runtime.get("status") != "completed"
        or runtime.get("blockers")
    ):
        raise AuraExecutionReceiptError(["aurafusion360_runtime_not_completed"])
    if (
        run.get("schema_version") != RUN_SCHEMA_VERSION
        or run.get("status") != "completed"
        or run.get("blockers")
    ):
        raise AuraExecutionReceiptError(["aurafusion360_provider_run_not_completed"])
    if (
        run.get("continuing_spend_from_this_run") is not False
        or run.get("all_staged_objects_absent") is not True
    ):
        raise AuraExecutionReceiptError(["aurafusion360_provider_zero_not_proven"])
    run_execution_path = Path(str(run.get("execution_result_path") or "")).resolve()
    if run_execution_path != runtime_path:
        raise AuraExecutionReceiptError(["aurafusion360_runtime_result_binding_mismatch"])

    scene = adapter.get("scene") or {}
    source = adapter.get("source") or {}
    camera_ids = sorted(
        Path(str(row.get("relative_path") or "")).stem
        for row in (adapter.get("artifacts") or [])
        if str(row.get("relative_path") or "").startswith("data/Other-360/")
        and "/images/" in str(row.get("relative_path") or "")
        and str(row.get("relative_path") or "").endswith(".png")
    )
    reference_camera_id = str(scene.get("reference_camera_id") or "")
    configured_reference_index = scene.get("reference_camera_index")
    reference_binding_observed = bool(
        camera_ids
        and len(camera_ids) == int(scene.get("camera_count") or 0)
        and reference_camera_id in camera_ids
        and isinstance(configured_reference_index, int)
    )
    expected_reference_index = (
        camera_ids.index(reference_camera_id) if reference_binding_observed else None
    )
    reference_binding_valid = (
        configured_reference_index == expected_reference_index
        if reference_binding_observed
        else None
    )
    if (
        runtime.get("scene_id") != scene.get("publisher_scene_id")
        or runtime.get("target_instance_id") != scene.get("target_instance_id")
    ):
        raise AuraExecutionReceiptError(["aurafusion360_execution_scene_or_target_mismatch"])
    if (
        runtime.get("source_commit") != source.get("commit")
        or runtime.get("source_tree") != source.get("tree")
        or runtime.get("source_modified") is not False
        or (runtime.get("source_identity_before") or {}).get("matches") is not True
        or (runtime.get("source_identity_after") or {}).get("matches") is not True
    ):
        raise AuraExecutionReceiptError(["aurafusion360_execution_source_identity_invalid"])

    observed_workflow = runtime.get("workflow") or []
    stages = [str(row.get("stage") or "") for row in observed_workflow]
    if stages != list(REQUIRED_STAGES) or any(
        row.get("returncode") != 0 or row.get("timed_out") is not False
        for row in observed_workflow
    ):
        raise AuraExecutionReceiptError(["aurafusion360_execution_workflow_incomplete"])
    flags = (
        "reference_generation_executed",
        "training_executed",
        "removal_executed",
        "inpaint_init_executed",
        "inpaint_finetune_executed",
    )
    if any(runtime.get(name) is not True for name in flags):
        raise AuraExecutionReceiptError(["aurafusion360_execution_flags_incomplete"])

    artifact_root = runtime_path.parent
    _, final_point_cloud = _verify_record(
        runtime.get("final_point_cloud") or {},
        root=artifact_root,
        code="aurafusion360_final_point_cloud_changed",
    )
    final_point_cloud["vertex_count"] = (runtime.get("final_point_cloud") or {}).get(
        "vertex_count"
    )
    if not isinstance(final_point_cloud["vertex_count"], int) or final_point_cloud[
        "vertex_count"
    ] <= 0:
        raise AuraExecutionReceiptError(["aurafusion360_final_point_cloud_count_invalid"])
    frames: list[dict[str, Any]] = []
    runtime_frames = runtime.get("final_frames") or []
    if len(runtime_frames) != int(scene.get("camera_count") or 0) or not runtime_frames:
        raise AuraExecutionReceiptError(["aurafusion360_final_frame_set_incomplete"])
    for camera_id, record in zip(camera_ids, runtime_frames, strict=True):
        _, verified = _verify_record(
            record,
            root=artifact_root,
            code="aurafusion360_final_frame_changed",
        )
        verified["camera_id"] = camera_id
        frames.append(verified)

    intermediate_sets = runtime.get("intermediate_frame_sets") or {}
    intermediate_stage_artifacts_retained = (
        runtime.get("stage_localization_evidence_retained") is True
        and isinstance(intermediate_sets, Mapping)
        and set(intermediate_sets) == {"inpaint_init_renders", "sdedit_images"}
        and all(
            isinstance(records, list) and len(records) == len(camera_ids)
            for records in intermediate_sets.values()
        )
    )
    verified_intermediate_sets: dict[str, list[dict[str, Any]]] = {}
    if intermediate_stage_artifacts_retained:
        for role, records in sorted(intermediate_sets.items()):
            verified_intermediate_sets[role] = [
                _verify_record(
                    record,
                    root=artifact_root,
                    code="aurafusion360_intermediate_frame_changed",
                )[1]
                for record in records
            ]

    logs: list[dict[str, Any]] = []
    for row in observed_workflow:
        log_name = str(row.get("log") or "")
        log_path = (artifact_root / log_name).resolve()
        if (
            not log_name
            or (log_path != artifact_root and artifact_root not in log_path.parents)
            or not log_path.is_file()
            or _sha256(log_path) != row.get("stdout_stderr_sha256")
        ):
            raise AuraExecutionReceiptError(["aurafusion360_stage_log_changed"])
        logs.append({"stage": row["stage"], **_file_record(log_path, artifact_root)})

    teardown_path = _under(
        str(run.get("teardown_manifest_path") or ""),
        (evidence,),
        "aurafusion360_teardown_manifest_outside_evidence_root",
    )
    teardown = _read_object(teardown_path, "aurafusion360_teardown_manifest_invalid")
    if teardown.get("continuing_spend_from_this_run") is not False:
        raise AuraExecutionReceiptError(["aurafusion360_teardown_provider_zero_not_proven"])

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "executed_candidate",
        "scene": {
            "publisher_scene_id": scene.get("publisher_scene_id"),
            "target_instance_id": scene.get("target_instance_id"),
            "target_semantic_label": scene.get("target_semantic_label"),
            "camera_count": scene.get("camera_count"),
            "input_receipt_digest": scene.get("input_receipt_digest"),
        },
        "source": source,
        "prepared_adapter": {
            "path": str(adapter_path),
            "size_bytes": adapter_path.stat().st_size,
            "sha256": _sha256(adapter_path),
            "receipt_digest": adapter.get("receipt_digest"),
        },
        "provider_run": {
            "path": str(run_path),
            "size_bytes": run_path.stat().st_size,
            "sha256": _sha256(run_path),
            "estimated_cost_usd": run.get("estimated_cost_usd"),
            "hard_cap_usd": run.get("hard_cap_usd"),
            "hard_ttl_seconds": run.get("hard_ttl_seconds"),
            "retry_cap": run.get("retry_cap"),
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "teardown_manifest": {
                "path": str(teardown_path),
                "size_bytes": teardown_path.stat().st_size,
                "sha256": _sha256(teardown_path),
            },
        },
        "execution": {
            "runtime_result": {
                "path": str(runtime_path),
                "size_bytes": runtime_path.stat().st_size,
                "sha256": _sha256(runtime_path),
            },
            "required_stages": list(REQUIRED_STAGES),
            "stage_logs": logs,
            "final_point_cloud": final_point_cloud,
            "final_frames": frames,
            "intermediate_frame_sets": verified_intermediate_sets,
            "depth_anything3_used": runtime.get("depth_anything3_used") is True,
        },
        "quality": {
            "status": "not_admitted",
            "hidden_background_truth_available": False,
            "quantitative_locality_measurement_observed": False,
            "human_visual_review_observed": False,
            "intermediate_stage_artifacts_retained": (
                intermediate_stage_artifacts_retained
            ),
            "runtime_reference_camera_binding_observed": reference_binding_observed,
            "runtime_reference_camera_binding_valid": reference_binding_valid,
            "reference_camera_id": reference_camera_id or None,
            "configured_reference_index": configured_reference_index,
            "expected_runtime_sorted_reference_index": expected_reference_index,
        },
        "claim_boundary": {
            "released_method_execution_completed": True,
            "successful_inpainting_admitted": False,
            "publisher_splat_edited_in_place": False,
            "source_collider_removed": False,
            "simready_replacement_inserted": False,
            "output_claim_ceiling": "visual_candidate_only",
        },
        "blockers": [
            (
                "aurafusion360_runtime_reference_camera_binding_mismatch"
                if reference_binding_valid is False
                else (
                    "aurafusion360_interiorgs_quality_admission_missing"
                    if intermediate_stage_artifacts_retained
                    else "aurafusion360_stage_localization_evidence_missing"
                )
            )
        ],
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if receipt_output is not None:
        output = _under(
            receipt_output, (repo,), "aurafusion360_execution_receipt_output_outside_repo"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-receipt", required=True)
    parser.add_argument("--runtime-result", required=True)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--receipt-output", required=True)
    args = parser.parse_args(argv)
    receipt = materialize_aura_execution_receipt(
        adapter_receipt_path=args.adapter_receipt,
        runtime_result_path=args.runtime_result,
        run_result_path=args.run_result,
        evidence_root=args.evidence_root,
        repo_root=args.repo_root,
        receipt_output=args.receipt_output,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
