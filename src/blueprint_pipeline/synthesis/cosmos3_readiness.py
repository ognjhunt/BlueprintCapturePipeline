"""Cosmos 3 feasibility checks for capture-grounded Blueprint site worlds.

This module is intentionally local-only. It reads existing capture, geometry,
site-reference, and Cosmos Predict artifacts, then reports what Blueprint can
claim before any model download, GPU runner, or live provider job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ..common import ensure_dir, read_json, utc_now_iso, write_json
from ..local_capture import resolve_local_capture_context

COSMOS3_READINESS_SCHEMA_VERSION = "cosmos3_capture_grounded_readiness.v1"

COSMOS3_SOURCE_URLS = {
    "research_page": "https://research.nvidia.com/labs/cosmos-lab/cosmos3/",
    "technical_report": "https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf",
    "github": "https://github.com/NVIDIA/cosmos",
}

COSMOS3_STACK_MAPPING = {
    "reasoner": {
        "blueprint_role": "site understanding and capture-quality review",
        "required_blueprint_evidence": [
            "raw walkthrough media",
            "capture provenance",
            "rights and privacy lineage",
            "keyframes or video clips",
        ],
        "claim_boundary": "reasoner output is advisory and cannot replace capture truth",
    },
    "generator": {
        "blueprint_role": "site-conditioned future-frame and synthetic-video generation",
        "required_blueprint_evidence": [
            "site reference database",
            "pose or local geometry",
            "intrinsics",
            "privacy-safe conditioning media",
            "held-out real revisits for validation",
        ],
        "claim_boundary": "generated clips are derived artifacts, not ground truth",
    },
    "world_action": {
        "blueprint_role": "future action-conditioned rollout layer",
        "required_blueprint_evidence": [
            "embodiment-specific action logs",
            "teleoperation or policy traces",
            "held-out action validation",
        ],
        "claim_boundary": "video-only walkthroughs do not prove robot-action readiness",
    },
}


def evaluate_cosmos3_capture_readiness(
    capture_root: str | Path,
    *,
    site_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate local Cosmos 3 data readiness for one staged capture."""

    context = resolve_local_capture_context(capture_root)
    descriptor = _read_optional_json(context.descriptor_path)
    raw_manifest = _merge_raw_sidecars(
        _read_optional_json(context.raw_root / "manifest.json"),
        context.raw_root,
    )
    resolved_site_id = site_id or _resolve_site_id(descriptor=descriptor, raw_manifest=raw_manifest)
    site_root = _resolve_site_reference_root(context=context, site_id=resolved_site_id)
    geometry_summary = _read_optional_json(context.pipeline_root / "geometry" / "geometry_summary.json")
    site_manifest = _read_optional_json(site_root / "site_reference_manifest.json") if site_root else {}
    training_export = _read_optional_json(context.pipeline_root / "cosmos_training_export" / "manifest.json")
    benchmark_manifests = _read_benchmark_manifests(context.pipeline_root)

    raw_check = _raw_capture_check(context=context, descriptor=descriptor, raw_manifest=raw_manifest)
    geometry_check = _geometry_check(geometry_summary)
    site_reference_check = _site_reference_check(site_root=site_root, site_manifest=site_manifest)
    training_export_check = _training_export_check(training_export)
    benchmark_check = _benchmark_check(benchmark_manifests)
    validation_check = _validation_check(
        context=context,
        descriptor=descriptor,
        raw_manifest=raw_manifest,
        benchmark_check=benchmark_check,
    )
    action_check = _action_evidence_check(context=context, descriptor=descriptor, raw_manifest=raw_manifest)

    stack_checks = {
        "raw_capture_contract": raw_check,
        "geometry_lane": geometry_check,
        "site_reference_database": site_reference_check,
        "cosmos_predict25_export": training_export_check,
        "cosmos_predict25_benchmark": benchmark_check,
        "held_out_validation": validation_check,
        "action_evidence": action_check,
    }

    capabilities = _capability_readiness(stack_checks)
    proof_gaps = _proof_gaps(stack_checks)
    sim_boundary = _simulation_boundary(capabilities=capabilities, stack_checks=stack_checks)

    payload = {
        "schema_version": COSMOS3_READINESS_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "claim_policy": "capture_grounded_local_preflight_only",
        "provider_jobs_called": False,
        "model_download_required": False,
        "cosmos3_source_urls": dict(COSMOS3_SOURCE_URLS),
        "cosmos3_stack_mapping": COSMOS3_STACK_MAPPING,
        "capture": {
            "storage_root": str(context.storage_root),
            "bucket": context.bucket,
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "site_id": resolved_site_id,
            "capture_root": str(context.capture_root),
        },
        "stack_checks": stack_checks,
        "capabilities": capabilities,
        "simulation_boundary": sim_boundary,
        "proof_gaps": proof_gaps,
        "today_stackable_work": _today_stackable_work(proof_gaps=proof_gaps),
        "public_claims": _public_claims(capabilities=capabilities),
    }
    return payload


def write_cosmos3_capture_readiness(
    capture_root: str | Path,
    *,
    site_id: Optional[str] = None,
    output_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Write JSON and Markdown readiness artifacts for a capture."""

    payload = evaluate_cosmos3_capture_readiness(capture_root, site_id=site_id)
    context = resolve_local_capture_context(capture_root)
    report_root = Path(output_root).expanduser().resolve() if output_root else (
        context.pipeline_root / "cosmos3_readiness"
    )
    ensure_dir(report_root)
    json_path = report_root / "cosmos3_capture_grounded_readiness.json"
    markdown_path = report_root / "cosmos3_capture_grounded_readiness.md"
    write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    payload["artifact_paths"] = {
        "json": str(json_path.resolve()),
        "markdown": str(markdown_path.resolve()),
    }
    write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _raw_capture_check(
    *,
    context,
    descriptor: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    evidence: Dict[str, Any] = {
        "capture_descriptor": str(context.descriptor_path) if context.descriptor_path.is_file() else None,
        "raw_manifest": str(context.raw_root / "manifest.json")
        if (context.raw_root / "manifest.json").is_file()
        else None,
        "capture_upload_complete": str(context.raw_complete_path)
        if context.raw_complete_path.is_file()
        else None,
        "raw_video": _raw_video_path(context=context, descriptor=descriptor, raw_manifest=raw_manifest),
        "privacy_safe_video": _privacy_safe_video_path(context=context, descriptor=descriptor),
        "poses": _existing_path(context.raw_root / "arkit" / "poses.jsonl"),
        "intrinsics": _existing_path(context.raw_root / "arkit" / "intrinsics.json"),
        "depth_dir": _existing_path(context.raw_root / "arkit" / "depth"),
    }
    if not evidence["capture_descriptor"] and not evidence["raw_manifest"]:
        blockers.append("missing_capture_descriptor_or_raw_manifest")
    if not evidence["capture_upload_complete"]:
        blockers.append("missing_capture_upload_complete")
    if not evidence["raw_video"]:
        blockers.append("missing_raw_walkthrough_video")
    if not _derived_scene_generation_allowed(descriptor=descriptor, raw_manifest=raw_manifest):
        blockers.append("derived_scene_generation_not_allowed")
    if not evidence["privacy_safe_video"]:
        warnings.append("privacy_safe_conditioning_video_not_found")
    if not evidence["poses"]:
        warnings.append("raw_pose_stream_not_found")
    if not evidence["intrinsics"]:
        warnings.append("raw_intrinsics_not_found")
    if not evidence["depth_dir"]:
        warnings.append("raw_depth_dir_not_found")
    state = "ready" if not blockers and not warnings else "degraded" if not blockers else "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "warnings": warnings,
        "evidence": evidence,
        "world_model_candidate": _world_model_candidate(descriptor=descriptor, raw_manifest=raw_manifest),
        "claim_boundary": "raw capture, rights, privacy, timestamps, poses, and device metadata remain authoritative",
    }


def _geometry_check(geometry_summary: Mapping[str, Any]) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not geometry_summary:
        blockers.append("missing_geometry_summary")
    geometry_source = str(geometry_summary.get("geometry_source") or "").strip()
    fallback_used = bool(geometry_summary.get("fallback_used"))
    live_ready = (
        geometry_source == "video_to_world"
        and bool(geometry_summary.get("geometry_live_ready"))
        and not fallback_used
    )
    local_reference_ready = (
        geometry_source == "local_sfm"
        and bool(geometry_summary.get("contract_ready_for_world_model"))
        and bool(geometry_summary.get("intrinsics_available"))
        and int(geometry_summary.get("pose_track_count") or 0) > 0
        and not fallback_used
    )
    if fallback_used:
        blockers.append("fallback_geometry_not_allowed_for_cosmos3_grounding")
    if geometry_summary and not live_ready and not local_reference_ready:
        blockers.append("geometry_not_reference_ready")
    if local_reference_ready and not live_ready:
        warnings.append("local_sfm_reference_media_only_not_provider_native_geometry")
    state = "ready" if live_ready else "degraded" if local_reference_ready else "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "warnings": warnings,
        "geometry_source": geometry_source or None,
        "fallback_used": fallback_used,
        "geometry_live_ready": live_ready,
        "local_reference_ready": local_reference_ready,
        "site_frame_available": bool(geometry_summary.get("site_frame_available")),
        "scale_resolved": bool(geometry_summary.get("scale_resolved")),
        "pose_track_count": int(geometry_summary.get("pose_track_count") or 0),
        "claim_boundary": "local geometry can ground retrieval, but only non-fallback video_to_world proof can mark provider-native geometry live",
    }


def _site_reference_check(
    *,
    site_root: Optional[Path],
    site_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    site_index_path = site_root / "site_reference_index.jsonl" if site_root else None
    if site_root is None:
        blockers.append("missing_site_id")
    elif not site_manifest:
        blockers.append("missing_site_reference_manifest")
    elif not site_index_path or not site_index_path.is_file():
        blockers.append("missing_site_reference_index")

    reference_frames = int(site_manifest.get("total_reference_frames") or 0)
    capture_count = int(site_manifest.get("capture_count") or 0)
    manifest_readiness = (
        dict(site_manifest.get("readiness") or {})
        if isinstance(site_manifest.get("readiness"), Mapping)
        else {}
    )
    if site_manifest and reference_frames <= 0:
        blockers.append("no_site_reference_frames")
    if site_manifest and capture_count <= 0:
        blockers.append("no_site_reference_captures")
    if site_manifest and not bool(site_manifest.get("site_frame_established")):
        warnings.append("site_frame_not_established")
    for blocker in _string_list(manifest_readiness.get("blockers")):
        if blocker not in warnings and blocker not in blockers:
            warnings.append(blocker)
    state = "ready" if not blockers and not warnings else "degraded" if not blockers else "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "warnings": warnings,
        "site_root": str(site_root) if site_root else None,
        "site_reference_manifest": str(site_root / "site_reference_manifest.json")
        if site_root and (site_root / "site_reference_manifest.json").is_file()
        else None,
        "site_reference_index": str(site_index_path) if site_index_path and site_index_path.is_file() else None,
        "total_reference_frames": reference_frames,
        "capture_count": capture_count,
        "site_frame_established": bool(site_manifest.get("site_frame_established")),
        "claim_boundary": "site reference DB is derived support evidence and must not expose dense rows to WebApp",
    }


def _training_export_check(training_export: Mapping[str, Any]) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not training_export:
        blockers.append("missing_cosmos_training_export_manifest")
    elif str(training_export.get("status") or "") != "ready":
        blockers.append(str(training_export.get("reason") or "cosmos_training_export_not_ready"))
    paired_count = int(training_export.get("paired_example_count") or 0)
    if training_export and paired_count <= 0:
        blockers.append("no_cosmos_paired_examples")
    val_count = int(training_export.get("val_count") or 0)
    if training_export and val_count <= 0:
        warnings.append("no_validation_split_examples")
    state = "ready" if not blockers and not warnings else "degraded" if not blockers else "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "warnings": warnings,
        "model_family": str(training_export.get("model_family") or "nvidia/Cosmos-Predict2.5-2B"),
        "source_mode": training_export.get("source_mode"),
        "paired_example_count": paired_count,
        "val_count": val_count,
        "trainer_config_path": training_export.get("trainer_config_path"),
        "inference_backend_shape_path": training_export.get("inference_backend_shape_path"),
        "claim_boundary": "Cosmos Predict 2.5 export is a reusable adapter substrate, not proof that Cosmos 3 has run",
    }


def _benchmark_check(benchmark_manifests: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    present = sorted(benchmark_manifests.keys())
    if not present:
        warnings.append("missing_cosmos_benchmark_manifest")
    for name, manifest in benchmark_manifests.items():
        reason = str(manifest.get("reason") or "").strip()
        if reason in {"cosmos_runtime_unavailable", "missing_cosmos_runtime_package"}:
            warnings.append(f"{name}:runtime_unavailable")
    state = "ready" if present and not blockers and not warnings else "degraded" if not blockers else "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "warnings": warnings,
        "benchmark_manifests": {
            name: _safe_manifest_summary(manifest)
            for name, manifest in sorted(benchmark_manifests.items())
        },
        "claim_boundary": "benchmark manifests may define local eval intent, but live/model proof requires a real Cosmos runtime run",
    }


def _validation_check(
    *,
    context,
    descriptor: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    benchmark_check: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    held_out_paths = [
        context.pipeline_root / "evaluation_prep" / "held_out_revisits.json",
        context.pipeline_root / "evaluation_prep" / "held_out_routes.json",
        context.raw_root / "held_out_revisits.json",
        context.raw_root / "revisit_manifest.json",
    ]
    existing_held_out = [str(path) for path in held_out_paths if path.is_file()]
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    manifest_topology = raw_manifest.get("capture_topology") if isinstance(raw_manifest.get("capture_topology"), Mapping) else {}
    descriptor_topology = metadata.get("capture_topology") if isinstance(metadata.get("capture_topology"), Mapping) else {}
    pass_count = _first_int(
        manifest_topology.get("pass_count") if isinstance(manifest_topology, Mapping) else None,
        descriptor_topology.get("pass_count") if isinstance(descriptor_topology, Mapping) else None,
    )
    if not existing_held_out and (pass_count or 0) < 2:
        blockers.append("missing_held_out_revisit_or_second_pass")
    if benchmark_check.get("state") == "degraded":
        warnings.append("cosmos_benchmark_plan_missing_or_runtime_unavailable")
    state = "ready" if not blockers and not warnings else "degraded" if not blockers else "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "warnings": warnings,
        "held_out_paths": existing_held_out,
        "pass_count": pass_count,
        "claim_boundary": "Cosmos outputs need held-out real capture checks before replacing sim-ready twin work for site review",
    }


def _action_evidence_check(
    *,
    context,
    descriptor: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    action_paths = [
        context.raw_root / "action_labels.jsonl",
        context.raw_root / "teleop_actions.jsonl",
        context.raw_root / "robot_policy_logs.jsonl",
        context.pipeline_root / "evaluation_prep" / "action_eval_manifest.json",
    ]
    existing = [str(path) for path in action_paths if path.is_file()]
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    manifest_actions = raw_manifest.get("action_evidence")
    descriptor_actions = metadata.get("action_evidence") if isinstance(metadata, Mapping) else None
    if not existing and not manifest_actions and not descriptor_actions:
        blockers.append("missing_action_or_teleoperation_evidence")
    state = "ready" if not blockers and not warnings else "degraded" if not blockers else "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "warnings": warnings,
        "action_evidence_paths": existing,
        "claim_boundary": "Cosmos 3 action/policy modes require action-labeled evidence, not video-only walkthroughs",
    }


def _capability_readiness(stack_checks: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    raw_ok = stack_checks["raw_capture_contract"]["state"] in {"ready", "degraded"}
    geometry_ok = stack_checks["geometry_lane"]["state"] in {"ready", "degraded"}
    site_ref_ok = stack_checks["site_reference_database"]["state"] in {"ready", "degraded"}
    export_ok = stack_checks["cosmos_predict25_export"]["state"] in {"ready", "degraded"}
    validation_ok = stack_checks["held_out_validation"]["state"] in {"ready", "degraded"}
    action_ok = stack_checks["action_evidence"]["state"] in {"ready", "degraded"}

    return {
        "reasoner_site_understanding": _capability(
            raw_ok,
            "Cosmos 3 Reasoner can be evaluated as an advisory capture/site understanding layer",
            "blocked until raw capture, provenance, rights, and privacy evidence exist",
        ),
        "generator_site_conditioning": _capability(
            raw_ok and geometry_ok and site_ref_ok and export_ok,
            "Cosmos 3 Generator can be treated as a candidate site-conditioned visual generation layer",
            "blocked until raw capture, geometry/reference media, site reference DB, and export substrate exist",
        ),
        "evaluator_site_consistency": _capability(
            raw_ok and site_ref_ok and validation_ok,
            "Cosmos 3 style evaluation can compare generated clips against held-out real site evidence",
            "blocked until held-out revisits or second-pass validation evidence exists",
        ),
        "world_action_policy": _capability(
            raw_ok and action_ok,
            "Cosmos 3 action modes can be investigated for embodiment-specific policy/eval work",
            "blocked until action-labeled or teleoperation evidence exists",
        ),
    }


def _capability(condition: bool, ready_detail: str, blocked_detail: str) -> Dict[str, Any]:
    return {
        "state": "data_ready" if condition else "blocked",
        "detail": ready_detail if condition else blocked_detail,
        "runtime_ready": False,
        "claim_boundary": "data readiness only; no Cosmos 3 runtime or provider proof",
    }


def _simulation_boundary(
    *,
    capabilities: Mapping[str, Mapping[str, Any]],
    stack_checks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    generator_ready = capabilities["generator_site_conditioning"]["state"] == "data_ready"
    evaluator_ready = capabilities["evaluator_site_consistency"]["state"] == "data_ready"
    action_ready = capabilities["world_action_policy"]["state"] == "data_ready"
    geometry_live_ready = bool(stack_checks["geometry_lane"].get("geometry_live_ready"))
    return {
        "full_sim_ready_digital_twin_avoidable_now": False,
        "visual_site_review_without_full_digital_twin": generator_ready and evaluator_ready,
        "perception_synthetic_data_without_full_digital_twin": generator_ready and evaluator_ready,
        "robot_action_or_collision_eval_without_sim_ready_twin": (
            generator_ready and evaluator_ready and action_ready and geometry_live_ready
        ),
        "blocked_claims": [
            "Generated outputs are ground truth",
            "one walkthrough creates a persistent exact simulator",
            "video-only capture is enough for action-policy evaluation",
            "fallback or local-only geometry proves live provider-native world-model readiness",
        ],
        "use_case_boundary": {
            "can_reduce_sim_ready_twin_need_for": [
                "visual site review",
                "future-frame prediction experiments",
                "site-conditioned synthetic perception data",
                "capture-quality and consistency evaluation",
            ],
            "still_needs_sim_or_stronger_evidence_for": [
                "collision/contact clearance",
                "manipulation",
                "door/drawer interaction",
                "safety-critical policy validation",
                "embodiment-specific action training",
            ],
        },
    }


def _proof_gaps(stack_checks: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    gaps: list[Dict[str, Any]] = []
    for check_name, check in stack_checks.items():
        for blocker in _string_list(check.get("blockers")):
            gaps.append(_gap(check_name=check_name, code=blocker, severity="blocker"))
        for warning in _string_list(check.get("warnings")):
            gaps.append(_gap(check_name=check_name, code=warning, severity="warning"))
    return gaps


def _gap(*, check_name: str, code: str, severity: str) -> Dict[str, Any]:
    next_action = {
        "missing_geometry_summary": "run local geometry lane before Cosmos readiness: python3 scripts/run_geometry_lane.py --capture-root <capture> --provider local_sfm --model local-sfm-offline",
        "fallback_geometry_not_allowed_for_cosmos3_grounding": "rerun geometry with local_sfm or provider-native video_to_world proof; do not use fallback geometry for Cosmos grounding claims",
        "missing_site_reference_manifest": "run retrieval_index lane after geometry/reference media is present",
        "missing_site_reference_index": "run retrieval_index lane to materialize sites/{site_id}/reference_memory/site_reference_index.jsonl",
        "missing_cosmos_training_export_manifest": "run the Cosmos Predict export lane before any model download",
        "missing_held_out_revisit_or_second_pass": "capture or stage a held-out revisit/second pass and write evaluation_prep/held_out_revisits.json",
        "missing_action_or_teleoperation_evidence": "keep action/policy claims blocked until action labels or teleoperation logs exist",
    }.get(code, "inspect the named stack check and add source-owned proof before promoting the claim")
    return {
        "check": check_name,
        "code": code,
        "severity": severity,
        "next_action": next_action,
    }


def _today_stackable_work(*, proof_gaps: Iterable[Mapping[str, Any]]) -> list[Dict[str, str]]:
    gap_codes = {str(gap.get("code") or "") for gap in proof_gaps}
    work: list[Dict[str, str]] = []
    if "missing_geometry_summary" in gap_codes:
        work.append(
            {
                "lane": "provider_native_geometry",
                "safe_command": (
                    "VIDEO_TO_WORLD_URL=<provider-url> VIDEO_TO_WORLD_RUNNER_TOKEN=<token> "
                    "python3 scripts/run_geometry_lane.py --capture-root <capture> "
                    "--provider video_to_world --model video_to_world-default"
                ),
                "success_criteria": (
                    "geometry_summary.json exists with geometry_source=video_to_world, "
                    "fallback_used=false, provider_native_result=true, and geometry_live_ready=true"
                ),
            }
        )
    if "missing_site_reference_manifest" in gap_codes or "missing_site_reference_index" in gap_codes:
        work.append(
            {
                "lane": "site_reference_database",
                "safe_command": "run retrieval_index lane on a capture with stable site_id and reference-indexable geometry",
                "success_criteria": "site_reference_manifest.json and site_reference_index.jsonl exist under sites/{site_id}/reference_memory/",
            }
        )
    if "missing_cosmos_training_export_manifest" in gap_codes:
        work.append(
            {
                "lane": "cosmos_predict25_export",
                "safe_command": "run export_cosmos_training_substrate() locally on the staged capture",
                "success_criteria": "pipeline/cosmos_training_export/manifest.json is ready with paired examples",
            }
        )
    if not work:
        work.append(
            {
                "lane": "cosmos3_runtime_not_today_by_default",
                "safe_command": "do not download or run Cosmos 3 until the local readiness report has no blocker gaps",
                "success_criteria": "local preflight stays reusable without provider credentials or GPU jobs",
            }
        )
    return work


def _public_claims(capabilities: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    any_data_ready = any(
        check.get("state") == "data_ready" for check in capabilities.values()
    )
    return {
        "allowed_internal_claim": (
            "Cosmos 3 is a candidate reasoner/generator/evaluator layer for capture-grounded site packages"
            if any_data_ready
            else "Cosmos 3 feasibility is still blocked on local evidence gaps"
        ),
        "public_display_ready_claim": "Do not make a public Cosmos 3 capability claim from this report alone.",
        "blocked_public_claims": [
            "Blueprint runs Cosmos 3 live",
            "Blueprint can replace digital twins for all robot evaluation",
            "Generated outputs are ground truth",
            "A single walkthrough creates an exact persistent simulator",
        ],
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    capture = payload.get("capture") if isinstance(payload.get("capture"), Mapping) else {}
    stack_checks = payload.get("stack_checks") if isinstance(payload.get("stack_checks"), Mapping) else {}
    capabilities = payload.get("capabilities") if isinstance(payload.get("capabilities"), Mapping) else {}
    sim_boundary = (
        payload.get("simulation_boundary")
        if isinstance(payload.get("simulation_boundary"), Mapping)
        else {}
    )
    lines = [
        "# Cosmos 3 Capture-Grounded Readiness",
        "",
        f"Generated: {payload.get('generated_at')}",
        f"Capture: {capture.get('scene_id')}/{capture.get('capture_id')}",
        f"Site: {capture.get('site_id') or 'missing'}",
        "",
        "## Stack Checks",
        "",
    ]
    for name, check in stack_checks.items():
        if not isinstance(check, Mapping):
            continue
        lines.append(f"- {name}: {check.get('state')}")
        for blocker in _string_list(check.get("blockers")):
            lines.append(f"  - blocker: {blocker}")
        for warning in _string_list(check.get("warnings")):
            lines.append(f"  - warning: {warning}")
    lines.extend(["", "## Capabilities", ""])
    for name, capability in capabilities.items():
        if isinstance(capability, Mapping):
            lines.append(f"- {name}: {capability.get('state')} - {capability.get('detail')}")
    lines.extend(
        [
            "",
            "## Simulation Boundary",
            "",
            f"- full_sim_ready_digital_twin_avoidable_now: {sim_boundary.get('full_sim_ready_digital_twin_avoidable_now')}",
            f"- visual_site_review_without_full_digital_twin: {sim_boundary.get('visual_site_review_without_full_digital_twin')}",
            f"- robot_action_or_collision_eval_without_sim_ready_twin: {sim_boundary.get('robot_action_or_collision_eval_without_sim_ready_twin')}",
            "",
            "Generated Cosmos outputs remain derived support artifacts. Raw capture, rights, privacy, poses, depth, intrinsics, and held-out revisits remain the evidence authority.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_benchmark_manifests(pipeline_root: Path) -> Dict[str, Dict[str, Any]]:
    candidates = {
        "cosmos_single_capture_smoke": pipeline_root
        / "cosmos_single_capture_smoke"
        / "cosmos_single_capture_smoke_manifest.json",
        "cosmos_zero_shot_validation": pipeline_root
        / "cosmos_zero_shot_validation"
        / "cosmos_zero_shot_benchmark.json",
    }
    manifests: Dict[str, Dict[str, Any]] = {}
    for name, path in candidates.items():
        payload = _read_optional_json(path)
        if payload:
            manifests[name] = payload
    return manifests


def _safe_manifest_summary(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": manifest.get("status"),
        "reason": manifest.get("reason"),
        "benchmark_family": manifest.get("benchmark_family"),
        "validation_example_count": manifest.get("validation_example_count"),
        "runtime_probe": manifest.get("runtime_probe"),
    }


def _resolve_site_reference_root(*, context, site_id: Optional[str]) -> Optional[Path]:
    if not site_id:
        return None
    candidates = [
        context.storage_root / context.bucket / "sites" / site_id / "reference_memory",
        context.storage_root / "sites" / site_id / "reference_memory",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_site_id(
    *,
    descriptor: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
) -> Optional[str]:
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    candidates = [
        descriptor.get("site_id"),
        raw_manifest.get("site_id"),
    ]
    for payload in (metadata, descriptor, raw_manifest):
        if not isinstance(payload, Mapping):
            continue
        identity = payload.get("site_identity")
        if isinstance(identity, Mapping):
            candidates.append(identity.get("site_id"))
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _merge_raw_sidecars(raw_manifest: Mapping[str, Any], raw_root: Path) -> Dict[str, Any]:
    merged = dict(raw_manifest)
    for key, filename in {
        "site_identity": "site_identity.json",
        "capture_topology": "capture_topology.json",
        "capture_mode": "capture_mode.json",
        "route_anchors": "route_anchors.json",
        "checkpoint_events": "checkpoint_events.json",
        "relocalization_events": "relocalization_events.json",
    }.items():
        if isinstance(merged.get(key), Mapping):
            continue
        sidecar = _read_optional_json(raw_root / filename)
        if sidecar:
            merged[key] = sidecar
    return merged


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _raw_video_path(*, context, descriptor: Mapping[str, Any], raw_manifest: Mapping[str, Any]) -> Optional[str]:
    for path in (context.raw_root / "walkthrough.mov", context.raw_root / "walkthrough.mp4"):
        if path.is_file() and path.stat().st_size > 0:
            return str(path)
    for value in (
        raw_manifest.get("video_uri"),
        raw_manifest.get("raw_video_uri"),
        descriptor.get("raw_video_uri"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return None


def _privacy_safe_video_path(*, context, descriptor: Mapping[str, Any]) -> Optional[str]:
    for path in (
        context.pipeline_root / "privacy" / "final_walkthrough.mov",
        context.pipeline_root / "privacy" / "final_walkthrough.mp4",
        context.pipeline_root / "privacy_safe" / "final_walkthrough.mov",
        context.pipeline_root / "privacy_safe" / "final_walkthrough.mp4",
    ):
        if path.is_file() and path.stat().st_size > 0:
            return str(path)
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    for value in (
        descriptor.get("privacy_processed_video_uri"),
        descriptor.get("world_model_video_uri"),
        metadata.get("privacy_processed_video_uri") if isinstance(metadata, Mapping) else None,
        metadata.get("world_model_video_uri") if isinstance(metadata, Mapping) else None,
    ):
        text = str(value or "").strip()
        if text:
            return text
    return None


def _derived_scene_generation_allowed(
    *,
    descriptor: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
) -> bool:
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    candidates = [
        descriptor.get("capture_rights"),
        raw_manifest.get("capture_rights"),
        metadata.get("capture_rights") if isinstance(metadata, Mapping) else None,
        metadata.get("rights_lineage") if isinstance(metadata, Mapping) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key in (
            "derived_scene_generation_allowed",
            "derived_generation_allowed",
            "world_model_generation_allowed",
        ):
            if key in candidate:
                return bool(candidate.get(key))
    return False


def _world_model_candidate(
    *,
    descriptor: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
) -> bool:
    quality = descriptor.get("quality") if isinstance(descriptor.get("quality"), Mapping) else {}
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    scene_memory = metadata.get("scene_memory_capture") if isinstance(metadata, Mapping) else {}
    return any(
        value is True
        for value in (
            descriptor.get("world_model_candidate"),
            quality.get("world_model_candidate") if isinstance(quality, Mapping) else None,
            scene_memory.get("world_model_candidate") if isinstance(scene_memory, Mapping) else None,
            raw_manifest.get("world_model_candidate"),
        )
    )


def _existing_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    out: list[str] = []
    for item in values:
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out
