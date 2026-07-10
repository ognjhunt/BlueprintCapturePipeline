"""Qualification-first pipeline with optional downstream derived artifacts."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .capture_enrichment_llm import build_capture_enrichment_runner
from .alpha_readiness import write_alpha_readiness_summary, write_pipeline_sync_result
from .canonical_site_package import write_blueprint_canonical_site_package
from .common import (
    MAXIMUM_HIDDEN_ZONE_BOUND,
    PipelineError,
    StageError,
    ensure_dir,
    ensure_local_uri_path,
    infer_storage_root_from_scene_path,
    parse_bool,
    parse_gs_uri,
    read_json,
    relative_scene_path,
    resolve_gs_uri_to_path,
    to_pipeline_prefix,
    utc_now_iso,
    write_json,
    write_text,
)
from .geometry_stage import build_geometry_stage_contract
from .industrial_ontology import classify_industrial_entity, derive_capture_plan_tags, industrial_tags_for_label
from .ios_manifest import IOSManifest, load_object_index, load_raw_manifest, resolve_object_index_uri
from .launch_bundle import build_buyer_trust_score, build_launch_qualification_bundle
from .launch_proof_policy import (
    production_forces_false,
    production_forces_true,
    production_launch_mode,
    relative_artifact_checksum,
)
from .object_index_stage import ensure_object_index_stage
from .object_index_artifacts import resolve_current_object_index_artifacts
from .privacy_processing import run_privacy_postprocess
from .provider_preview import run_preview_provider
from .proof_contracts import build_rights_provenance_review
from .runtime_layer_grounding import build_presentation_variance_policy, with_grounding_fields
from .scene_semantics import infer_capture_fidelity_review
from .task_targets import infer_task_targets, write_task_targets
from .webapp_sync import (
    WebappSyncError,
    derive_webapp_opportunity_state,
    derive_webapp_qualification_state,
    sync_webapp_pipeline_attachment,
)
from .world_model_policy import (
    WorldModelPolicy,
    build_output_linkage,
    build_presentation_derivation_policy,
    build_provenance_record,
)


@dataclass
class QualificationGate:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _webapp_sync_failure_requires_stage_failure() -> bool:
    return (
        production_forces_true("PIPELINE_SYNC_REQUIRED", default=False)
        or bool(str(os.getenv("PIPELINE_SYNC_WEBAPP_URL") or "").strip())
        or bool(str(os.getenv("PIPELINE_SYNC_TOKEN") or "").strip())
    )


def _local_file_pointer(path: Path) -> Dict[str, Any]:
    pointer: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return pointer
    stat = path.stat()
    pointer["size_bytes"] = int(stat.st_size)
    pointer["modified_at"] = utc_now_iso()
    return pointer


def _default_qa_report_uri(descriptor_uri: str) -> str:
    parsed = parse_gs_uri(descriptor_uri)
    if parsed.key.endswith("capture_descriptor.json"):
        qa_key = parsed.key[: -len("capture_descriptor.json")] + "qa_report.json"
    else:
        qa_key = f"{parsed.key.rstrip('/')}/qa_report.json"
    return f"gs://{parsed.bucket}/{qa_key}"


def _safe_path_exists(uri: Optional[str], storage_root: Path) -> bool:
    if not uri:
        return False
    try:
        return resolve_gs_uri_to_path(uri, storage_root).exists()
    except Exception:
        return False


def _try_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return read_json(path)
    except Exception:
        return None


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _presentation_demo_ui_payload() -> Dict[str, Optional[str]]:
    return {
        "ui_base_url": str(os.getenv("BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL") or "").strip() or None,
        "public_ui_base_url": str(os.getenv("BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL") or "").strip() or None,
    }


def _capture_orientation_payload(descriptor: CaptureDescriptor) -> Dict[str, Any]:
    raw = (
        descriptor.capture_orientation
        if isinstance(descriptor.capture_orientation, Mapping)
        else {}
    )
    payload = dict(raw)
    encoded_size = payload.get("encoded_size") if isinstance(payload.get("encoded_size"), Mapping) else {}
    display_size = payload.get("display_size") if isinstance(payload.get("display_size"), Mapping) else {}
    payload.setdefault("encoded_width", int(encoded_size.get("width") or 0))
    payload.setdefault("encoded_height", int(encoded_size.get("height") or 0))
    payload.setdefault("declared_capture_width", int(display_size.get("width") or 0))
    payload.setdefault("declared_capture_height", int(display_size.get("height") or 0))
    payload.setdefault("display_rotation_degrees", int(payload.get("rotation_degrees") or 0))
    payload.setdefault("normalization_applied", bool(payload.get("display_rotation_degrees") or 0))
    payload.setdefault("display_orientation", "unknown")
    payload.setdefault("rotation_degrees", 0)
    payload.setdefault("display_size", {})
    payload.setdefault("encoded_size", {})
    payload.setdefault("source", "inferred")
    payload.setdefault("preserve_original_display_orientation", True)
    return payload


def _artifact_pointer(path: Path, *, bucket: str, storage_root: Path) -> Dict[str, Any]:
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "uri": f"gs://{bucket}/{relative_scene_path(path, storage_root)}",
    }


def _presentation_primary_asset(
    *,
    pipeline_dir: Path,
    bucket: str,
    storage_root: Path,
) -> Optional[Dict[str, Any]]:
    advanced_path = pipeline_dir / "advanced_geometry" / "3dgs_compressed.ply"
    if advanced_path.is_file():
        payload = _artifact_pointer(advanced_path, bucket=bucket, storage_root=storage_root)
        payload["source_name"] = "advanced_geometry_3dgs"
        return payload
    return None


def _presentation_supporting_assets(
    *,
    pipeline_dir: Path,
    bucket: str,
    storage_root: Path,
) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for path in (
        pipeline_dir / "advanced_geometry" / "advanced_geometry_bundle.json",
    ):
        if path.is_file():
            assets.append(_artifact_pointer(path, bucket=bucket, storage_root=storage_root))
    return assets


def _canonical_world_model_payload(
    *,
    pipeline_dir: Path,
    bucket: str,
    storage_root: Path,
    capture_orientation: Mapping[str, Any],
) -> Dict[str, Any]:
    primary_asset = _presentation_primary_asset(
        pipeline_dir=pipeline_dir,
        bucket=bucket,
        storage_root=storage_root,
    )
    supporting_assets = _presentation_supporting_assets(
        pipeline_dir=pipeline_dir,
        bucket=bucket,
        storage_root=storage_root,
    )
    status = "ready" if primary_asset is not None else "missing"
    return {
        "world_model_backend": "site_world_runtime",
        "primary_runtime_backend": "site_world_runtime",
        "scene_representation": "advanced_geometry_3dgs" if primary_asset is not None else "pending_world_model_service",
        "render_source": "canonical_world_model" if primary_asset is not None else "pending_world_model_service",
        "fallback_mode": "none",
        "evidence_mode": "full_capture_persistent_scene",
        "primary_render_asset_role": "authoritative_runtime_render_asset",
        "renderer_backend": "site_world_runtime" if primary_asset is not None else None,
        "bundle_type": "site_world_runtime_video_world_model_v1" if primary_asset is not None else None,
        "status": status,
        "primary_asset_path": str(primary_asset.get("path") or "") if primary_asset else "",
        "primary_asset_uri": str(primary_asset.get("uri") or "") if primary_asset else "",
        "primary_asset_source": str(primary_asset.get("source_name") or "") if primary_asset else "",
        "supporting_assets": supporting_assets,
        "orientation": dict(capture_orientation),
    }


def _presentation_bundle_status(
    *,
    emit_presentation: bool,
    primary_asset: Optional[Mapping[str, Any]],
    render_inputs: Mapping[str, Any],
) -> str:
    if not emit_presentation:
        return "disabled"
    missing_inputs = _string_list(render_inputs.get("missing_inputs"))
    if primary_asset and not missing_inputs:
        return "ready"
    if primary_asset:
        return "partial"
    return "missing"


def _presentation_quality_summary(
    *,
    primary_asset: Optional[Mapping[str, Any]],
    supporting_assets: List[Mapping[str, Any]],
    render_inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    missing_inputs = _string_list(render_inputs.get("missing_inputs"))
    return {
        "primary_asset_present": primary_asset is not None,
        "supporting_asset_count": len(supporting_assets),
        "required_input_count": len(_string_list(render_inputs.get("required_inputs"))),
        "available_input_count": len(_string_list(render_inputs.get("available_inputs"))),
        "missing_input_count": len(missing_inputs),
        "missing_inputs": missing_inputs,
    }


def _presentation_camera_behavior(capture_orientation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "primary_mode": "pose_driven_preview",
        "supported_modes": [
            "pose_driven_preview",
            "canonical_anchor_jump",
            "bounded_lookaround",
        ],
        "viewpoint_frame": "canonical_site_world",
        "pose_inputs": ["raw_video", "arkit_poses", "arkit_intrinsics"],
        "allow_pose_extrapolation": False,
        "preserve_capture_display_orientation": bool(
            capture_orientation.get("preserve_original_display_orientation", True)
        ),
    }


def _presentation_render_inputs(
    *,
    descriptor: CaptureDescriptor,
    scene_memory_manifest_uri: str,
    conditioning_bundle_uri: str,
    preview_simulation_manifest_uri: str,
    geometry_conditioning: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    geometry = geometry_conditioning if isinstance(geometry_conditioning, Mapping) else {}
    privacy_video_uri = descriptor.preferred_world_model_video_uri
    payload = {
        "privacy_processed_video_uri": privacy_video_uri,
        "keyframe_uri": descriptor.keyframe_uri,
        "arkit_poses_uri": descriptor.arkit_poses_uri,
        "arkit_intrinsics_uri": descriptor.arkit_intrinsics_uri,
        "arkit_frames_uri": descriptor.arkit_frames_uri,
        "arkit_depth_prefix_uri": descriptor.arkit_depth_prefix_uri,
        "arkit_confidence_prefix_uri": descriptor.arkit_confidence_prefix_uri,
        "geometry_manifest_uri": geometry.get("geometry_manifest_uri"),
        "geometry_summary_uri": geometry.get("geometry_summary_uri"),
        "geometry_poses_uri": geometry.get("camera_poses_uri"),
        "geometry_intrinsics_uri": geometry.get("camera_intrinsics_uri"),
        "geometry_depth_manifest_uri": geometry.get("depth_manifest_uri"),
        "geometry_confidence_manifest_uri": geometry.get("confidence_manifest_uri"),
        "scene_memory_manifest_uri": scene_memory_manifest_uri,
        "conditioning_bundle_uri": conditioning_bundle_uri,
        "preview_simulation_manifest_uri": preview_simulation_manifest_uri,
        "protected_regions_manifest_uri": None,
        "object_geometry_manifest_uri": None,
        "site_world_spec_uri": None,
    }
    required_inputs = [
        "privacy_processed_video_uri",
        "scene_memory_manifest_uri",
        "conditioning_bundle_uri",
    ]
    if descriptor.arkit_poses_uri:
        required_inputs.extend(["arkit_poses_uri", "arkit_intrinsics_uri", "arkit_depth_prefix_uri"])
    elif geometry.get("camera_poses_uri"):
        required_inputs.extend(
            ["geometry_poses_uri", "geometry_intrinsics_uri", "geometry_depth_manifest_uri"]
        )
    available_inputs = [key for key, value in payload.items() if str(value or "").strip()]
    missing_inputs = [key for key in required_inputs if key not in available_inputs]
    payload["required_inputs"] = required_inputs
    payload["available_inputs"] = available_inputs
    payload["missing_inputs"] = missing_inputs
    return payload


def _presentation_demo_readiness(
    *,
    render_inputs: Mapping[str, Any],
    ui_payload: Mapping[str, Optional[str]],
) -> Dict[str, Any]:
    blockers = [f"missing_render_input:{key}" for key in _string_list(render_inputs.get("missing_inputs"))]
    if not str(ui_payload.get("ui_base_url") or "").strip() and not str(
        ui_payload.get("public_ui_base_url") or ""
    ).strip():
        blockers.append("missing_demo_ui_base_url")
    readiness_state = "ready" if not blockers else "blocked"
    return {
        "readiness_state": readiness_state,
        "blockers": blockers,
        "warnings": [],
        "ui_configured": readiness_state == "ready" or "missing_demo_ui_base_url" not in blockers,
    }


def _has_structured_intake_from_metadata(
    metadata: Mapping[str, Any],
    *,
    intake_packet_uri: Optional[str] = None,
) -> bool:
    task_statement = str(metadata.get("task_statement") or "").strip()
    workflow_context = str(metadata.get("workflow_context") or "").strip()
    task_zone = metadata.get("task_zone") if isinstance(metadata.get("task_zone"), Mapping) else {}
    zone_label = str(task_zone.get("label") or "").strip()
    success_criteria = _string_list(metadata.get("success_criteria"))
    return bool((task_statement or workflow_context) and (zone_label or intake_packet_uri) and success_criteria)


def _has_structured_intake(descriptor: CaptureDescriptor) -> bool:
    metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    return _has_structured_intake_from_metadata(
        metadata,
        intake_packet_uri=descriptor.intake_packet_uri,
    )


_GENERIC_TASK_PHRASES = {
    "pick and place",
    "pick-and-place",
    "pick place",
    "walkthrough",
    "walk through",
    "scan",
    "mapping",
    "inspection",
}

_TASK_OBJECT_KEYWORDS = {
    "tote": ("tote", "bin", "box", "container", "crate"),
    "shelf": ("shelf", "rack", "cabinet"),
    "drawer": ("drawer",),
    "door": ("door", "gate"),
    "pallet": ("pallet",),
    "panel": ("panel", "breaker", "switch", "valve"),
    "laundry": ("washer", "dryer", "hamper"),
    "bedroom": ("bed", "dresser", "closet"),
    "kitchen": ("fridge", "refrigerator", "microwave", "oven", "dishwasher", "sink"),
}

_INDUSTRIAL_ENVIRONMENTS = {"warehouse", "industrial_unknown", "manufacturing", "fulfillment", "brownfield_site"}
_RESIDENTIAL_ENVIRONMENTS = {"default", "bedroom", "kitchen"}


def _try_read_optional_json_uri(uri: Optional[str], storage_root: Path) -> Optional[Dict[str, Any]]:
    if not uri:
        return None
    try:
        path = resolve_gs_uri_to_path(uri, storage_root)
    except Exception:
        return None
    return _try_read_json(path)


def _object_index_runtime_blockers(capture_root: Path) -> List[str]:
    resolved = resolve_current_object_index_artifacts(capture_root)
    build_report_path = str(resolved.get("build_report_path") or "").strip()
    build_report = _try_read_json(Path(build_report_path)) if build_report_path else {}
    runtime_preflight = build_report.get("runtime_preflight") if isinstance(build_report.get("runtime_preflight"), Mapping) else {}
    preflight_backends = runtime_preflight.get("backends") if isinstance(runtime_preflight.get("backends"), Mapping) else {}
    backend_summary = build_report.get("backend_summary") if isinstance(build_report.get("backend_summary"), Mapping) else {}
    providers = backend_summary.get("providers") if isinstance(backend_summary.get("providers"), list) else []
    blockers: List[str] = []
    for provider in providers:
        if not isinstance(provider, Mapping):
            continue
        backend = str(provider.get("backend") or "").strip()
        reason = str(provider.get("reason") or "").strip()
        if not backend or not reason:
            continue
        support_level = "required"
        preflight_entry = preflight_backends.get(backend)
        if isinstance(preflight_entry, Mapping):
            support_level = str(preflight_entry.get("support_level") or "required").strip().lower() or "required"
        if support_level == "optional":
            continue
        if any(
            token in reason.lower()
            for token in ("missing", "not_installed", "weights_missing", "failed_to_launch", "ultralytics_missing")
        ):
            blocker = f"object_index_backend:{backend}:{reason}"
            if blocker not in blockers:
                blockers.append(blocker)
    return blockers


def _object_index_exception_blocker(context: str, exc: BaseException) -> str:
    message = str(exc).strip().replace("\n", " ")[:240] or "no_message"
    return f"object_index_stage:{context}:{type(exc).__name__}:{message}"


def _append_unique(items: List[str], item: str) -> None:
    if item and item not in items:
        items.append(item)


def _extend_unique(items: List[str], new_items: Iterable[str]) -> None:
    for item in new_items:
        _append_unique(items, str(item))


def _task_hypothesis_string(raw: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = raw.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _task_hypothesis_list(raw: Mapping[str, Any], *keys: str) -> List[str]:
    for key in keys:
        value = raw.get(key)
        items = _string_list(value)
        if items:
            return items
    return []


def _normalize_task_hypothesis_source(raw: Mapping[str, Any], descriptor: CaptureDescriptor) -> str:
    source = _task_hypothesis_string(raw, "source")
    if source:
        return source
    intake_source = (
        descriptor.metadata.get("intake_source")
        if isinstance(descriptor.metadata, Mapping)
        else None
    )
    source = str(intake_source or "").strip()
    if source:
        return source
    return "authoritative_intake" if descriptor.intake_packet_uri else "unknown"


def _build_task_hypothesis_seed(
    *,
    descriptor: CaptureDescriptor,
    raw_task_hypothesis: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    raw = dict(raw_task_hypothesis) if isinstance(raw_task_hypothesis, Mapping) else {}
    has_raw_hypothesis = bool(raw)

    workflow_name = _task_hypothesis_string(raw, "workflow_name", "workflowName")
    if not workflow_name and not has_raw_hypothesis:
        workflow_name = str(metadata.get("task_statement") or metadata.get("workflow_context") or "").strip()

    task_steps = _task_hypothesis_list(raw, "task_steps", "taskSteps")
    if not task_steps and not has_raw_hypothesis:
        task_steps = _string_list(metadata.get("workflow_decomposition")) or _string_list(metadata.get("workflow_context"))

    zone = _task_hypothesis_string(raw, "zone")
    if not zone and not has_raw_hypothesis:
        task_zone = metadata.get("task_zone") if isinstance(metadata.get("task_zone"), Mapping) else {}
        zone = str(task_zone.get("label") or "").strip()

    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "workflow_name": workflow_name,
        "task_steps": task_steps,
        "target_kpi": _task_hypothesis_string(raw, "target_kpi", "targetKPI") or (
            _string_list(metadata.get("success_criteria"))[:1][0]
            if (not has_raw_hypothesis and _string_list(metadata.get("success_criteria")))
            else ""
        ),
        "zone": zone,
        "owner": _task_hypothesis_string(raw, "owner") or (str(metadata.get("owner") or "").strip() if not has_raw_hypothesis else ""),
        "confidence": _safe_float(raw.get("confidence"), 1.0 if descriptor.intake_packet_uri else 0.0),
        "source": _normalize_task_hypothesis_source(raw, descriptor),
        "model": _task_hypothesis_string(raw, "model"),
        "fps": int(_safe_float(raw.get("fps"), 0.0)) if raw.get("fps") is not None else None,
        "warnings": _task_hypothesis_list(raw, "warnings"),
        "status": _task_hypothesis_string(raw, "status") or "accepted",
    }


def _task_hypothesis_object_matches(text: str, object_labels: List[str]) -> List[str]:
    lowered = text.lower()
    matches: List[str] = []
    for semantic_key, keywords in _TASK_OBJECT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            if any(keyword in label for label in object_labels for keyword in keywords):
                matches.append(semantic_key)
    return matches


def _task_hypothesis_is_generic(workflow_name: str, task_steps: List[str]) -> bool:
    lowered = " ".join([workflow_name, *task_steps]).strip().lower()
    return any(phrase in lowered for phrase in _GENERIC_TASK_PHRASES)


def _task_hypothesis_environment_contradictions(
    *,
    text: str,
    environment_hint: str,
) -> List[str]:
    contradictions: List[str] = []
    lowered = text.lower()
    if environment_hint in _INDUSTRIAL_ENVIRONMENTS and any(keyword in lowered for keyword in ("bedroom", "closet", "laundry", "washer", "dryer")):
        contradictions.append("The inferred task sounds residential while the capture is tagged as industrial.")
    if environment_hint in _RESIDENTIAL_ENVIRONMENTS and any(keyword in lowered for keyword in ("pallet", "forklift", "dock", "tote", "aisle")):
        contradictions.append("The inferred task sounds industrial while the capture is tagged as residential or generic.")
    return contradictions


def _build_task_hypothesis_report(
    *,
    descriptor: CaptureDescriptor,
    raw_task_hypothesis: Optional[Mapping[str, Any]],
    object_index_entries: List[Mapping[str, Any]],
    task_targets_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    hypothesis = _build_task_hypothesis_seed(
        descriptor=descriptor,
        raw_task_hypothesis=raw_task_hypothesis,
    )
    workflow_name = str(hypothesis.get("workflow_name") or "").strip()
    task_steps = _string_list(hypothesis.get("task_steps"))
    target_kpi = str(hypothesis.get("target_kpi") or "").strip()
    zone = str(hypothesis.get("zone") or "").strip()
    owner = str(hypothesis.get("owner") or "").strip()
    source = str(hypothesis.get("source") or "unknown")
    confidence = _safe_float(hypothesis.get("confidence"), 0.0)
    warnings = _string_list(hypothesis.get("warnings"))

    object_labels = [
        str(entry.get("label") or entry.get("name") or "").strip().lower()
        for entry in object_index_entries
        if isinstance(entry, Mapping) and str(entry.get("label") or entry.get("name") or "").strip()
    ]
    grounded_matches = _task_hypothesis_object_matches(
        " ".join([workflow_name, *task_steps]),
        object_labels,
    )
    target_ids = _string_list(task_targets_payload.get("target_object_ids"))
    has_object_grounding = bool(grounded_matches or target_ids)
    contradictions: List[str] = []
    if source == "ai_inferred":
        contradictions = _task_hypothesis_environment_contradictions(
            text=" ".join([workflow_name, *task_steps]),
            environment_hint=descriptor.environment_type_hint or "default",
        )
    if source == "ai_inferred" and str(hypothesis.get("status") or "").strip() == "rejected":
        contradictions.append("The app-side AI task hypothesis was rejected before qualification.")
    generic_task = _task_hypothesis_is_generic(workflow_name, task_steps)
    if generic_task and not has_object_grounding:
        warnings.append("Task remains generic without grounded objects or a specific task zone.")
    if not zone:
        warnings.append("No task zone was grounded from the current evidence.")

    if contradictions:
        status = "contradicted"
    elif source == "ai_inferred":
        if confidence >= 0.8 and not generic_task and (has_object_grounding or zone):
            status = "accepted"
        elif confidence >= 0.6 and (has_object_grounding or zone):
            status = "accepted_with_warnings"
        else:
            status = "needs_confirmation"
    else:
        status = "accepted_with_warnings" if warnings else "accepted"

    normalized = {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "task_hypothesis_status": status,
        "workflow_name": workflow_name,
        "task_steps": task_steps,
        "target_kpi": target_kpi or None,
        "zone": zone or None,
        "owner": owner or None,
        "confidence": round(confidence, 4),
        "source": source,
        "model": hypothesis.get("model"),
        "fps": hypothesis.get("fps"),
        "warnings": warnings,
        "contradictions": contradictions,
        "generic_task": generic_task,
        "grounded_object_labels": grounded_matches,
        "grounded_target_ids": target_ids,
    }
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "task_hypothesis_status": status,
        "source": source,
        "confidence": round(confidence, 4),
        "raw_task_hypothesis": hypothesis,
        "normalized_task_hypothesis": normalized,
        "warnings": warnings,
        "contradictions": contradictions,
        "evidence_summary": {
            "object_index_count": len(object_index_entries),
            "grounded_object_labels": grounded_matches,
            "grounded_target_ids": target_ids,
            "environment_hint": descriptor.environment_type_hint or "default",
        },
    }


def _effective_task_metadata(
    descriptor: CaptureDescriptor,
    *,
    task_hypothesis_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    base = dict(descriptor.metadata) if isinstance(descriptor.metadata, Mapping) else {}
    report = task_hypothesis_report if isinstance(task_hypothesis_report, Mapping) else {}
    normalized = report.get("normalized_task_hypothesis") if isinstance(report.get("normalized_task_hypothesis"), Mapping) else {}
    status = str(report.get("task_hypothesis_status") or "").strip()
    if status not in {"accepted", "accepted_with_warnings"}:
        base["task_hypothesis_status"] = status or None
        if normalized:
            base["task_hypothesis_confidence"] = normalized.get("confidence")
            base["task_hypothesis_source"] = normalized.get("source")
        return base

    workflow_name = str(normalized.get("workflow_name") or "").strip()
    task_steps = _string_list(normalized.get("task_steps"))
    target_kpi = str(normalized.get("target_kpi") or "").strip()
    zone = str(normalized.get("zone") or "").strip()
    owner = str(normalized.get("owner") or "").strip()
    warnings = _string_list(normalized.get("warnings"))

    if workflow_name:
        base["task_statement"] = workflow_name
    if task_steps:
        base["workflow_context"] = " | ".join(task_steps)
        base["workflow_decomposition"] = task_steps
    if target_kpi:
        base["success_criteria"] = [target_kpi]
    elif not _string_list(base.get("success_criteria")) and workflow_name:
        base["success_criteria"] = [f"Confirm whether '{workflow_name}' is ready for downstream review."]
    if zone:
        base["task_zone"] = {"label": zone}
    if owner:
        base["owner"] = owner
    base["task_hypothesis_status"] = status
    base["task_hypothesis_confidence"] = normalized.get("confidence")
    base["task_hypothesis_source"] = normalized.get("source")
    if warnings:
        base["task_hypothesis_warnings"] = warnings
    return base



def _modality_supports_metric_automation(descriptor: CaptureDescriptor) -> bool:
    if descriptor.evidence_tier == "qualified_metric_capture":
        return True
    scaffolding_validation = (
        descriptor.scaffolding_validation
        if isinstance(descriptor.scaffolding_validation, Mapping)
        else {}
    )
    if (
        descriptor.evidence_tier == "video_with_validated_scaffolding"
        and bool(scaffolding_validation.get("validated_metric_bundle"))
    ):
        return True
    return False


def _relative_path_from(base_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, base_dir)).as_posix()


def _normalize_local_path(path_value: Any, *, base_dir: Path) -> Optional[str]:
    raw = str(path_value or "").strip()
    if not raw or raw.startswith("gs://"):
        return None
    candidate = Path(raw)
    resolved = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
    return _relative_path_from(base_dir, resolved)


def _metadata_scene_package_path(metadata: Mapping[str, Any], *, base_dir: Path) -> Optional[str]:
    scene_package = (
        metadata.get("scene_package")
        if isinstance(metadata.get("scene_package"), Mapping)
        else {}
    )
    for value in (
        scene_package.get("scene_package_path"),
        scene_package.get("root_path"),
        scene_package.get("bundle_path"),
        metadata.get("scene_package_path"),
    ):
        normalized = _normalize_local_path(value, base_dir=base_dir)
        if normalized:
            return normalized
    return None


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def attach_handoff_package_paths(
    handoff_payload: Mapping[str, Any],
    *,
    pipeline_dir: Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(handoff_payload)
    handoff_dir = pipeline_dir
    advanced_dir = pipeline_dir / "advanced_geometry"
    bundle_manifest = advanced_dir / "advanced_geometry_bundle.json"

    if bundle_manifest.is_file():
        geometry_package: Dict[str, Any] = {"bundle_path": _relative_path_from(handoff_dir, advanced_dir)}
        optional_files = {
            "ply_path": advanced_dir / "3dgs_compressed.ply",
            "labels_path": advanced_dir / "labels.json",
            "structure_path": advanced_dir / "structure.json",
            "holi_spatial_grounding_path": advanced_dir / "holi_spatial_grounding.json",
            "task_hints_path": advanced_dir / "task_targets.synthetic.json",
        }
        for key, path in optional_files.items():
            if path.is_file():
                geometry_package[key] = _relative_path_from(handoff_dir, path)
        payload["geometry_package"] = geometry_package
    else:
        payload.pop("geometry_package", None)

    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    scene_memory_manifest = pipeline_dir / "scene_memory" / "scene_memory_manifest.json"
    if scene_memory_manifest.is_file():
        payload["scene_memory_package"] = {
            "bundle_path": _relative_path_from(handoff_dir, scene_memory_manifest.parent),
            "scene_memory_manifest_path": _relative_path_from(handoff_dir, scene_memory_manifest),
            "scene_memory_readiness_path": _relative_path_from(
                handoff_dir,
                scene_memory_manifest.parent / "scene_memory_readiness.json",
            ),
            "conditioning_bundle_path": _relative_path_from(
                handoff_dir,
                scene_memory_manifest.parent / "conditioning_bundle.json",
            ),
            "preview_simulation_manifest_path": _relative_path_from(
                handoff_dir,
                pipeline_dir / "preview_simulation" / "preview_simulation_manifest.json",
            ),
            "gen3c_adapter_manifest_path": _relative_path_from(
                handoff_dir,
                scene_memory_manifest.parent / "adapter_manifests" / "gen3c.json",
            ),
            "site_world_runtime_adapter_manifest_path": _relative_path_from(
                handoff_dir,
                scene_memory_manifest.parent / "adapter_manifests" / "site_world_runtime.json",
            ),
            "cosmos_transfer_adapter_manifest_path": _relative_path_from(
                handoff_dir,
                scene_memory_manifest.parent / "adapter_manifests" / "cosmos_transfer.json",
            ),
            **(
                {
                    "presentation_bundle_path": _relative_path_from(
                        handoff_dir,
                        pipeline_dir / "presentation_world" / "presentation_bundle.json",
                    )
                }
                if (pipeline_dir / "presentation_world" / "presentation_bundle.json").is_file()
                else {}
            ),
            **(
                {
                    "presentation_world_manifest_path": _relative_path_from(
                        handoff_dir,
                        pipeline_dir / "presentation_world" / "presentation_world_manifest.json",
                    )
                }
                if (pipeline_dir / "presentation_world" / "presentation_world_manifest.json").is_file()
                else {}
            ),
            **(
                {
                    "runtime_demo_manifest_path": _relative_path_from(
                        handoff_dir,
                        pipeline_dir / "presentation_world" / "runtime_demo_manifest.json",
                    )
                }
                if (pipeline_dir / "presentation_world" / "runtime_demo_manifest.json").is_file()
                else {}
            ),
        }
    else:
        payload.pop("scene_memory_package", None)

    scene_package_path = _metadata_scene_package_path(metadata_mapping, base_dir=handoff_dir)
    if scene_package_path:
        payload["scene_package"] = {"scene_package_path": scene_package_path}
    else:
        payload.pop("scene_package", None)

    return payload


def _capture_rights(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    raw = metadata.get("capture_rights") if isinstance(metadata.get("capture_rights"), Mapping) else {}
    return {
        "derived_scene_generation_allowed": parse_bool(
            raw.get("derived_scene_generation_allowed"),
            default=False,
        ),
        "data_licensing_allowed": parse_bool(
            raw.get("data_licensing_allowed"),
            default=False,
        ),
        "capture_contributor_payout_eligible": parse_bool(
            raw.get("capture_contributor_payout_eligible"),
            default=False,
        ),
        "consent_status": str(raw.get("consent_status") or "unknown"),
        "permission_document_uri": str(raw.get("permission_document_uri") or "").strip() or None,
        "consent_scope": _string_list(raw.get("consent_scope")),
        "commercialization_terms": _mapping(
            raw.get("commercialization_terms")
            or raw.get("commercializationTerms")
            or raw.get("commercial_terms")
            or raw.get("commercialTerms")
        ),
        "operator_revenue_terms": _mapping(
            raw.get("operator_revenue_terms")
            or raw.get("operatorRevenueTerms")
            or raw.get("revenue_share_terms")
            or raw.get("revenueShareTerms")
        ),
        "exclusivity_terms": _mapping(
            raw.get("exclusivity_terms") or raw.get("exclusivityTerms")
        ),
    }


def _worldlabs_derived_rights_allowed(*, metadata: Mapping[str, Any]) -> bool:
    """Whether the capture is rights-cleared for derived scene generation.

    PIPE-04: the WorldLabs preview input video is a derived, reviewer-facing
    transformation of the capture, so it must not be generated unless
    ``derived_scene_generation_allowed`` is set — mirroring scene-memory readiness
    (see the ``derived_scene_rights`` gate).
    """
    rights = _capture_rights(metadata if isinstance(metadata, Mapping) else {})
    return bool(rights["derived_scene_generation_allowed"])


def _scene_memory_capture_summary(
    descriptor: CaptureDescriptor,
    *,
    geometry_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    capture_summary = (
        metadata.get("scene_memory_capture")
        if isinstance(metadata.get("scene_memory_capture"), Mapping)
        else {}
    )
    quality = descriptor.quality if isinstance(descriptor.quality, Mapping) else {}
    return {
        "continuity_score": float(capture_summary.get("continuity_score", 0.0) or 0.0),
        "lighting_consistency": str(capture_summary.get("lighting_consistency") or "unknown"),
        "dynamic_object_density": str(capture_summary.get("dynamic_object_density") or "unknown"),
        "sensor_availability": (
            dict(capture_summary.get("sensor_availability"))
            if isinstance(capture_summary.get("sensor_availability"), Mapping)
            else {
                "arkit_poses": descriptor.arkit_poses_uri is not None,
                "arkit_intrinsics": descriptor.arkit_intrinsics_uri is not None,
                "arkit_depth": descriptor.arkit_depth_prefix_uri is not None,
                "arkit_confidence": descriptor.arkit_confidence_prefix_uri is not None,
                "depth_conditioning": bool(descriptor.depth_conditioning),
            }
        ),
        "operator_notes": _string_list(capture_summary.get("operator_notes")),
        "world_model_candidate": bool(
            capture_summary.get("world_model_candidate", quality.get("world_model_candidate"))
        ),
        "world_model_candidate_reasoning": list(capture_summary.get("world_model_candidate_reasoning") or []),
        "site_identity": (
            dict(metadata.get("site_identity"))
            if isinstance(metadata.get("site_identity"), Mapping)
            else None
        ),
        "capture_topology": (
            dict(metadata.get("capture_topology"))
            if isinstance(metadata.get("capture_topology"), Mapping)
            else None
        ),
        "capture_mode": (
            dict(metadata.get("capture_mode"))
            if isinstance(metadata.get("capture_mode"), Mapping)
            else None
        ),
        "geometry_summary": (
            {
                "status": str(geometry_summary.get("status") or "missing"),
                "ready_for_world_model": bool(geometry_summary.get("ready_for_world_model")),
                "scale_status": str(
                    ((geometry_summary.get("scale_assessment") or {}) if isinstance(geometry_summary.get("scale_assessment"), Mapping) else {}).get("status")
                    or "missing"
                ),
            }
            if isinstance(geometry_summary, Mapping)
            else {}
        ),
    }


def _build_scene_memory_readiness(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    geometry_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    capture_summary = _scene_memory_capture_summary(
        descriptor,
        geometry_summary=geometry_summary,
    )
    completeness_status = str(scorecard.get("completeness_status") or "unknown")
    metric_ready = bool(qualification_record.get("metric_ready"))
    readiness_state = str(qualification_record.get("readiness_state") or "not_ready_yet")
    rights = _capture_rights(descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {})
    geometry_ready = bool(
        isinstance(geometry_summary, Mapping) and geometry_summary.get("ready_for_world_model")
    )
    status = (
        "ready"
        if (
            readiness_state == "ready"
            and capture_summary["world_model_candidate"]
            and completeness_status == "sufficient"
            and bool(descriptor.raw_video_uri)
            and rights["derived_scene_generation_allowed"]
        )
        else "needs_more_evidence"
    )
    return {
        "schema_version": "v1",
        "lane": "scene_memory",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "status": status,
        "derived_only": True,
        "authoritative_source": "qualification_and_capture_evidence",
        "world_model_candidate": capture_summary["world_model_candidate"],
        "rights": rights,
        "capture_summary": capture_summary,
        "qualification_alignment": {
            "readiness_state": readiness_state,
            "metric_ready": metric_ready,
            "completeness_status": completeness_status,
        },
        "gates": [
            {
                "name": "raw_video_present",
                "passed": bool(descriptor.raw_video_uri),
                "detail": "raw walkthrough video available" if descriptor.raw_video_uri else "raw walkthrough video missing",
            },
            {
                "name": "qualification_completeness",
                "passed": completeness_status == "sufficient",
                "detail": f"qualification completeness is {completeness_status}",
            },
            {
                "name": "derived_scene_rights",
                "passed": rights["derived_scene_generation_allowed"],
                "detail": "capture is rights-cleared for derived scene generation"
                if rights["derived_scene_generation_allowed"]
                else "capture rights do not permit derived scene generation",
            },
            {
                "name": "explicit_conditioning_available",
                "passed": metric_ready or descriptor.arkit_poses_uri is not None or geometry_ready,
                "detail": "explicit conditioning is available from metric geometry, ARKit poses, or advisory geometry lane"
                if (metric_ready or descriptor.arkit_poses_uri is not None or geometry_ready)
                else "scene memory will rely on monocular-only conditioning",
            },
        ],
    }


def _scene_memory_derived_assets(
    scene_memory_artifacts: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    assets: Dict[str, Dict[str, Any]] = {}
    if scene_memory_artifacts.get("scene_memory_manifest_uri"):
        scene_memory_status = str(scene_memory_artifacts.get("scene_memory_status") or "needs_more_evidence")
        preview_status = str(scene_memory_artifacts.get("preview_simulation_status") or "review_required")
        assets.update(
            {
                "scene_memory": {
                    "status": scene_memory_status,
                    "manifest_uri": scene_memory_artifacts.get("scene_memory_manifest_uri"),
                    "artifact_uri": scene_memory_artifacts.get("conditioning_bundle_uri"),
                },
                "preview_simulation": {
                    "status": preview_status,
                    "manifest_uri": scene_memory_artifacts.get("preview_simulation_manifest_uri"),
                    "artifact_uri": scene_memory_artifacts.get("preview_simulation_manifest_uri"),
                },
            }
        )
    if scene_memory_artifacts.get("presentation_world_manifest_uri"):
        assets["presentation_world"] = {
            "status": "available",
            "manifest_uri": scene_memory_artifacts.get("presentation_world_manifest_uri"),
            "artifact_uri": scene_memory_artifacts.get("presentation_bundle_uri"),
        }
    if scene_memory_artifacts.get("geometry_summary_uri"):
        assets["geometry_conditioning"] = {
            "status": "available"
            if scene_memory_artifacts.get("geometry_summary_uri")
            else "missing",
            "manifest_uri": scene_memory_artifacts.get("geometry_manifest_uri"),
            "artifact_uri": scene_memory_artifacts.get("geometry_summary_uri"),
        }
    return assets


def _write_scene_memory_bundle(
    *,
    storage_root: Path,
    bucket: str,
    pipeline_prefix: str,
    pipeline_dir: Path,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    geometry_artifacts: Optional[Mapping[str, Any]] = None,
    depth_conditioning: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    privacy_video_uri = descriptor.preferred_world_model_video_uri
    if not privacy_video_uri:
        raise StageError(
            "scene_memory",
            "privacy_processed_video_required_for_derived_artifacts",
        )
    policy = WorldModelPolicy.from_env()
    geometry_conditioning = (
        dict(geometry_artifacts) if isinstance(geometry_artifacts, Mapping) else {}
    )
    effective_depth_conditioning = (
        dict(depth_conditioning)
        if isinstance(depth_conditioning, Mapping)
        else dict(descriptor.depth_conditioning)
        if isinstance(descriptor.depth_conditioning, Mapping)
        else {}
    )
    geometry_summary = (
        geometry_conditioning.get("summary")
        if isinstance(geometry_conditioning.get("summary"), Mapping)
        else {}
    )
    scene_memory_dir = pipeline_dir / "scene_memory"
    adapter_dir = scene_memory_dir / "adapter_manifests"
    preview_dir = pipeline_dir / "preview_simulation"
    presentation_dir = pipeline_dir / "presentation_world"
    ensure_dir(scene_memory_dir)
    ensure_dir(adapter_dir)
    ensure_dir(preview_dir)
    ensure_dir(presentation_dir)

    scene_memory_manifest_uri = f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'scene_memory_manifest.json', storage_root)}"
    scene_memory_readiness_uri = f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'scene_memory_readiness.json', storage_root)}"
    conditioning_bundle_uri = f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'conditioning_bundle.json', storage_root)}"
    preview_simulation_manifest_uri = f"gs://{bucket}/{relative_scene_path(preview_dir / 'preview_simulation_manifest.json', storage_root)}"
    presentation_bundle_uri = f"gs://{bucket}/{relative_scene_path(presentation_dir / 'presentation_bundle.json', storage_root)}"
    presentation_world_manifest_uri = f"gs://{bucket}/{relative_scene_path(presentation_dir / 'presentation_world_manifest.json', storage_root)}"
    runtime_demo_manifest_uri = f"gs://{bucket}/{relative_scene_path(presentation_dir / 'runtime_demo_manifest.json', storage_root)}"
    capture_orientation = _capture_orientation_payload(descriptor)
    presentation_variance_policy = build_presentation_variance_policy()
    derivation_policy = build_presentation_derivation_policy(
        policy=policy,
        variance_policy=presentation_variance_policy,
    )

    readiness_payload = _build_scene_memory_readiness(
        descriptor=descriptor,
        scorecard=scorecard,
        qualification_record=qualification_record,
        geometry_summary=geometry_summary,
    )
    write_json(scene_memory_dir / "scene_memory_readiness.json", readiness_payload)

    advanced_dir = pipeline_dir / "advanced_geometry"
    explicit_conditioning: Dict[str, Any] = {}
    for name, rel_path in {
        "advanced_geometry_bundle_uri": "advanced_geometry_bundle.json",
        "compressed_geometry_uri": "3dgs_compressed.ply",
        "labels_uri": "labels.json",
        "structure_uri": "structure.json",
        "task_hints_uri": "task_targets.synthetic.json",
    }.items():
        path = advanced_dir / rel_path
        if path.is_file():
            explicit_conditioning[name] = f"gs://{bucket}/{relative_scene_path(path, storage_root)}"
    if geometry_conditioning.get("geometry_manifest_uri"):
        explicit_conditioning.update(
            {
                "geometry_manifest_uri": geometry_conditioning.get("geometry_manifest_uri"),
                "geometry_summary_uri": geometry_conditioning.get("geometry_summary_uri"),
                "geometry_poses_uri": geometry_conditioning.get("camera_poses_uri"),
                "geometry_intrinsics_uri": geometry_conditioning.get("camera_intrinsics_uri"),
                "geometry_depth_manifest_uri": geometry_conditioning.get("depth_manifest_uri"),
                "geometry_confidence_manifest_uri": geometry_conditioning.get("confidence_manifest_uri"),
            }
        )
    if effective_depth_conditioning:
        explicit_conditioning.update(
            {
                "depth_conditioning_source": effective_depth_conditioning.get("source"),
                "depth_conditioning_depth_manifest_uri": effective_depth_conditioning.get("depth_manifest_uri"),
                "depth_conditioning_confidence_manifest_uri": effective_depth_conditioning.get("confidence_manifest_uri"),
                "depth_conditioning_depth_prefix_uri": effective_depth_conditioning.get("depth_prefix_uri"),
                "depth_conditioning_confidence_prefix_uri": effective_depth_conditioning.get("confidence_prefix_uri"),
            }
        )

    conditioning_provenance = build_provenance_record(
        grounding_level="observed",
        evidence_sources=[
            privacy_video_uri,
            descriptor.frames_index_uri,
            descriptor.arkit_poses_uri,
            descriptor.arkit_intrinsics_uri,
            descriptor.arkit_depth_prefix_uri,
            geometry_conditioning.get("geometry_summary_uri"),
            geometry_conditioning.get("camera_poses_uri"),
            geometry_conditioning.get("depth_manifest_uri"),
            effective_depth_conditioning.get("depth_manifest_uri"),
            effective_depth_conditioning.get("confidence_manifest_uri"),
            effective_depth_conditioning.get("depth_prefix_uri"),
            effective_depth_conditioning.get("confidence_prefix_uri"),
        ],
        observation_coverage={
            "capture_modality": descriptor.capture_modality,
            "has_explicit_conditioning": bool(explicit_conditioning),
        },
        confidence=qualification_record.get("confidence"),
        canonical_truth=True,
        presentation_only=False,
    )
    canonical_world_model = _canonical_world_model_payload(
        pipeline_dir=pipeline_dir,
        bucket=bucket,
        storage_root=storage_root,
        capture_orientation=capture_orientation,
    )
    runtime_render_source = (
        "site_world_runtime_full_capture"
        if privacy_video_uri and descriptor.arkit_poses_uri and descriptor.arkit_intrinsics_uri
        else str(canonical_world_model.get("render_source") or "unavailable")
    )
    scene_representation = (
        "site_world_runtime_video_world_model_v1"
        if runtime_render_source == "site_world_runtime_full_capture"
        else str(canonical_world_model.get("scene_representation") or "unavailable")
    )
    conditioning_bundle = with_grounding_fields({
        "schema_version": "v1",
        "lane": "scene_memory",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "privacy_processed_video_uri": privacy_video_uri,
        "world_model_video_uri": privacy_video_uri,
        "frames_index_uri": descriptor.frames_index_uri,
        "keyframe_uri": descriptor.keyframe_uri,
        "arkit": {
            "poses_uri": descriptor.arkit_poses_uri,
            "intrinsics_uri": descriptor.arkit_intrinsics_uri,
            "depth_prefix_uri": descriptor.arkit_depth_prefix_uri,
            "confidence_prefix_uri": descriptor.arkit_confidence_prefix_uri,
        },
        "depth_conditioning": dict(effective_depth_conditioning),
        "geometry": {
            "manifest_uri": geometry_conditioning.get("geometry_manifest_uri"),
            "summary_uri": geometry_conditioning.get("geometry_summary_uri"),
            "poses_uri": geometry_conditioning.get("camera_poses_uri"),
            "intrinsics_uri": geometry_conditioning.get("camera_intrinsics_uri"),
            "depth_manifest_uri": geometry_conditioning.get("depth_manifest_uri"),
            "confidence_manifest_uri": geometry_conditioning.get("confidence_manifest_uri"),
            "summary": dict(geometry_summary) if isinstance(geometry_summary, Mapping) else {},
        },
        "explicit_conditioning": explicit_conditioning,
        "qualification_artifacts": {
            "qualification_record_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "readiness_decision_uri": f"gs://{bucket}/{pipeline_prefix}/readiness_decision.json",
            "geometry_evidence_uri": f"gs://{bucket}/{pipeline_prefix}/geometry_evidence.json",
            "opportunity_handoff_uri": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
        },
        "output_policy": {
            "derived_only": True,
            "authoritative_record": "qualification_record.json",
            "generated_outputs_cannot_override_readiness": True,
        },
        "capture_orientation": capture_orientation,
        "geometry_conditioning": {
            "manifest_uri": geometry_conditioning.get("geometry_manifest_uri"),
            "summary_uri": geometry_conditioning.get("geometry_summary_uri"),
            "summary": dict(geometry_summary) if isinstance(geometry_summary, Mapping) else {},
        },
        "primary_runtime_backend": "site_world_runtime",
        "canonical_world_model": canonical_world_model,
        "runtime_render_source": runtime_render_source,
        "fallback_mode": "arkit_rgbd_last_resort",
        "scene_representation": scene_representation,
        "world_model_policy": policy.to_dict(),
        "canonical_output": build_output_linkage(
            policy=policy,
            canonical_artifact_uri=conditioning_bundle_uri,
            presentation_artifact_uri=presentation_bundle_uri if policy.emit_presentation else None,
            authoritative_record=True,
        ),
        "canonical_package_version": None,
        "provenance": conditioning_provenance,
    }, provenance=conditioning_provenance)
    write_json(scene_memory_dir / "conditioning_bundle.json", conditioning_bundle)

    scene_memory_provenance = build_provenance_record(
        grounding_level="observed",
        evidence_sources=[
            conditioning_bundle_uri,
            f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            f"gs://{bucket}/{pipeline_prefix}/readiness_decision.json",
        ],
        observation_coverage={"scene_memory_status": readiness_payload["status"]},
        confidence=qualification_record.get("confidence"),
        canonical_truth=True,
        presentation_only=False,
    )
    scene_memory_manifest = with_grounding_fields({
        "schema_version": "v1",
        "lane": "scene_memory",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "scene_memory_readiness_uri": scene_memory_readiness_uri,
        "conditioning_bundle_uri": conditioning_bundle_uri,
        "authoritative_artifacts": {
            "qualification_record_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "qualification_brief_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_brief.json",
            "readiness_decision_uri": f"gs://{bucket}/{pipeline_prefix}/readiness_decision.json",
            "human_actions_required_uri": f"gs://{bucket}/{pipeline_prefix}/human_actions_required.json",
        },
        "rights": readiness_payload["rights"],
        "capture_orientation": capture_orientation,
        "geometry_conditioning": {
            "manifest_uri": geometry_conditioning.get("geometry_manifest_uri"),
            "summary_uri": geometry_conditioning.get("geometry_summary_uri"),
            "poses_uri": geometry_conditioning.get("camera_poses_uri"),
            "intrinsics_uri": geometry_conditioning.get("camera_intrinsics_uri"),
            "depth_manifest_uri": geometry_conditioning.get("depth_manifest_uri"),
            "confidence_manifest_uri": geometry_conditioning.get("confidence_manifest_uri"),
            "summary": dict(geometry_summary) if isinstance(geometry_summary, Mapping) else {},
        },
        "depth_conditioning": dict(effective_depth_conditioning),
        "primary_runtime_backend": "site_world_runtime",
        "canonical_world_model": canonical_world_model,
        "runtime_render_source": runtime_render_source,
        "fallback_mode": "arkit_rgbd_last_resort",
        "scene_representation": scene_representation,
        "world_model_policy": policy.to_dict(),
        "canonical_output": build_output_linkage(
            policy=policy,
            canonical_artifact_uri=scene_memory_manifest_uri,
            presentation_artifact_uri=presentation_bundle_uri if policy.emit_presentation else None,
            authoritative_record=True,
        ),
        "canonical_package_version": None,
        "provenance": scene_memory_provenance,
    }, provenance=scene_memory_provenance)
    write_json(scene_memory_dir / "scene_memory_manifest.json", scene_memory_manifest)

    adapter_specs = {
        "gen3c": {
            "family": "GEN3C",
            "preferred_conditioning": ["rgb_video", "camera_poses", "depth", "explicit_geometry"],
            "required_conditioning": ["camera_poses", "intrinsics", "depth_or_explicit_geometry"],
            "execution_mode": "remote_service",
            "reconstruction_backend_name": "gen3c",
            "service_contract_version": "stage1_world_model_remote_v1",
            "normalized_output_contract": [
                "export_last.usdz",
                "nvblox_mesh.ply",
                "visual_mesh.glb",
                "mesh_manifest.json",
                "occupancy.bin",
                "object_point_cloud_index.json",
                "capture_quality_report.json",
            ],
            "status": "available_stage1_remote",
        },
        "site_world_runtime": {
            "family": "Site world runtime",
            "preferred_conditioning": ["rgb_video", "camera_trajectory", "feed_forward_4d_reconstruction"],
            "required_conditioning": ["rgb_video"],
            "execution_mode": "local_gpu_runtime",
            "reconstruction_backend_name": "site_world_runtime",
            "service_contract_version": "stage1_world_model_local_v1",
            "normalized_output_contract": [
                "export_last.usdz",
                "nvblox_mesh.ply",
                "visual_mesh.glb",
                "mesh_manifest.json",
                "occupancy.bin",
                "object_point_cloud_index.json",
                "capture_quality_report.json",
            ],
            "status": "available_stage1_local",
        },
        "cosmos_transfer": {
            "family": "Cosmos Transfer",
            "preferred_conditioning": ["rgb_video", "depth", "segmentation", "edge"],
            "required_conditioning": ["depth", "segmentation", "edge"],
            "execution_mode": "planned_phase3",
            "reconstruction_backend_name": "cosmos_transfer",
            "service_contract_version": "reserved_phase3",
            "normalized_output_contract": [
                "export_last.usdz",
                "nvblox_mesh.ply",
                "visual_mesh.glb",
                "mesh_manifest.json",
                "occupancy.bin",
                "object_point_cloud_index.json",
                "capture_quality_report.json",
            ],
            "status": "planned_phase3",
        },
    }
    adapter_artifacts: Dict[str, str] = {}
    for adapter_id, spec in adapter_specs.items():
        payload = {
            "schema_version": "v1",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "adapter_id": adapter_id,
            "family": spec["family"],
            "generated_at": utc_now_iso(),
            "conditioning_bundle_uri": f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'conditioning_bundle.json', storage_root)}",
            "preferred_conditioning": spec["preferred_conditioning"],
            "required_conditioning": spec["required_conditioning"],
            "execution_mode": spec["execution_mode"],
            "reconstruction_backend_name": spec["reconstruction_backend_name"],
            "service_contract_version": spec["service_contract_version"],
            "normalized_output_contract": spec["normalized_output_contract"],
            "status": spec["status"],
            "derived_only": True,
            "output_policy": policy.to_dict(),
            "grounding_requirements": {
                "preserve_capture_backed_truth": True,
                "provenance_required": policy.provenance_required,
                "canonical_incomplete_ok": policy.canonical_incomplete_ok,
            },
            "presentation_allowed": True,
            "canonical_write_allowed": False,
            "authoritative_record": False,
            "canonical_artifact_uri": scene_memory_manifest_uri,
            "presentation_artifact_uri": presentation_world_manifest_uri if policy.emit_presentation else None,
            "derivation_mode": policy.output_policy,
        }
        path = adapter_dir / f"{adapter_id}.json"
        write_json(path, payload)
        adapter_artifacts[f"{adapter_id}_adapter_manifest_uri"] = f"gs://{bucket}/{relative_scene_path(path, storage_root)}"

    preview_manifest = {
        "schema_version": "v1",
        "lane": "preview_simulation",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "status": "prep_ready" if readiness_payload["status"] == "ready" else "review_required",
        "scene_memory_manifest_uri": scene_memory_manifest_uri,
        "supported_backends": ["gen3c", "site_world_runtime", "cosmos_transfer"],
        "note": "Low-volume preview generation only. High-volume synthetic frames and datasets belong in BlueprintValidation.",
        "world_model_policy": policy.to_dict(),
        "canonical_artifact_uri": scene_memory_manifest_uri,
        "presentation_artifact_uri": preview_simulation_manifest_uri,
        "derivation_mode": policy.allow_generative_completion,
        "authoritative_record": False,
    }
    write_json(preview_dir / "preview_simulation_manifest.json", preview_manifest)

    render_inputs = _presentation_render_inputs(
        descriptor=descriptor,
        scene_memory_manifest_uri=scene_memory_manifest_uri,
        conditioning_bundle_uri=conditioning_bundle_uri,
        preview_simulation_manifest_uri=preview_simulation_manifest_uri,
        geometry_conditioning=geometry_conditioning,
    )
    primary_asset = _presentation_primary_asset(
        pipeline_dir=pipeline_dir,
        bucket=bucket,
        storage_root=storage_root,
    )
    supporting_assets = _presentation_supporting_assets(
        pipeline_dir=pipeline_dir,
        bucket=bucket,
        storage_root=storage_root,
    )
    bundle_status = _presentation_bundle_status(
        emit_presentation=policy.emit_presentation,
        primary_asset=primary_asset,
        render_inputs=render_inputs,
    )
    canonical_source = {
        "scene_memory_manifest_uri": scene_memory_manifest_uri,
        "conditioning_bundle_uri": conditioning_bundle_uri,
        "preview_simulation_manifest_uri": preview_simulation_manifest_uri,
        "canonical_package_uri": None,
        "canonical_package_version": None,
        "authoritative_source": "canonical_site_world",
    }
    camera_behavior = _presentation_camera_behavior(capture_orientation)
    presentation_provenance = build_provenance_record(
        grounding_level="generated" if policy.emit_presentation else "reconstructed",
        evidence_sources=[scene_memory_manifest_uri, conditioning_bundle_uri],
        observation_coverage={"presentation_enabled": policy.emit_presentation},
        confidence=qualification_record.get("confidence"),
        canonical_truth=False,
        presentation_only=True,
    )
    presentation_quality_summary = _presentation_quality_summary(
        primary_asset=primary_asset,
        supporting_assets=supporting_assets,
        render_inputs=render_inputs,
    )
    if isinstance(primary_asset, Mapping):
        authoritative_runtime_render_manifest = with_grounding_fields({
            "schema_version": "v1",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "generated_at": utc_now_iso(),
            "status": "ready",
            "world_model_backend": "site_world_runtime",
            "scene_representation": "advanced_geometry_3dgs",
            "render_source": "canonical_world_model",
            "fallback_mode": "none",
            "renderer_backend": "site_world_runtime",
            "bundle_type": "site_world_runtime_video_world_model_v1",
            "primary_asset_uri": primary_asset.get("uri"),
            "primary_asset_path": primary_asset.get("path"),
            "primary_asset_source": primary_asset.get("source_name"),
            "supporting_assets": supporting_assets,
            "orientation": capture_orientation,
            "provenance": presentation_provenance,
        }, provenance=presentation_provenance)
        write_json(
            presentation_dir / "authoritative_runtime_render_manifest.json",
            authoritative_runtime_render_manifest,
        )
    presentation_bundle = with_grounding_fields({
        "schema_version": "v1",
        "lane": "presentation_world_bundle",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "status": bundle_status if policy.emit_presentation else "disabled",
        "bundle_type": "gsplat_scene_v1",
        "renderer_backend": "gsplat",
        "authoritative_record": False,
        "canonical_artifact_uri": scene_memory_manifest_uri,
        "presentation_artifact_uri": presentation_bundle_uri,
        "canonical_source": canonical_source,
        "derivation_policy": derivation_policy,
        "capture_orientation": capture_orientation,
        "orientation": capture_orientation,
        "camera_behavior": camera_behavior,
        "render_inputs": render_inputs,
        "required_dependencies": {
            "privacy_processed_video_uri": privacy_video_uri,
            "arkit_poses_uri": descriptor.arkit_poses_uri,
            "arkit_intrinsics_uri": descriptor.arkit_intrinsics_uri,
            "arkit_depth_prefix_uri": descriptor.arkit_depth_prefix_uri,
            "arkit_confidence_prefix_uri": descriptor.arkit_confidence_prefix_uri,
            "capture_orientation": capture_orientation,
        },
        "primary_asset_uri": primary_asset.get("uri") if isinstance(primary_asset, Mapping) else None,
        "primary_asset_path": primary_asset.get("path") if isinstance(primary_asset, Mapping) else None,
        "supporting_assets": supporting_assets,
        "fallback_policy": "canonical_only",
        "quality_summary": presentation_quality_summary,
        "runtime_contract": {
            "runtime_demo_manifest_uri": runtime_demo_manifest_uri if policy.emit_presentation else None,
            "interactive_demo_type": "canonical_grounded_site_world",
            "consumer_contract": {
                "supported_consumers": ["BlueprintValidation", "Blueprint-WebApp"],
                "launch_readiness_field": "interactive_demo.readiness_state",
                "legacy_url_fields": ["ui_base_url", "public_ui_base_url"],
            },
        },
        "world_model_policy": policy.to_dict(),
        "canonical_package_version": None,
        "canonical_package_uri": None,
        "provenance": presentation_provenance,
    }, provenance=presentation_provenance)
    write_json(presentation_dir / "presentation_bundle.json", presentation_bundle)

    demo_ui_payload = _presentation_demo_ui_payload()
    interactive_demo = _presentation_demo_readiness(
        render_inputs=render_inputs,
        ui_payload=demo_ui_payload,
    )
    demo_status = (
        "demo_ready"
        if policy.emit_presentation and interactive_demo["readiness_state"] == "ready"
        else bundle_status
        if policy.emit_presentation
        else "disabled"
    )
    presentation_manifest = with_grounding_fields({
        "schema_version": "v1",
        "lane": "presentation_world",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "status": demo_status,
        "bundle_type": "gsplat_scene_v1",
        "renderer_backend": "gsplat",
        "canonical_artifact_uri": scene_memory_manifest_uri,
        "presentation_artifact_uri": presentation_bundle_uri,
        "presentation_bundle_uri": presentation_bundle_uri if policy.emit_presentation else None,
        "runtime_demo_manifest_uri": runtime_demo_manifest_uri if policy.emit_presentation else None,
        "preview_simulation_manifest_uri": preview_simulation_manifest_uri if policy.emit_presentation else None,
        "canonical_source": canonical_source,
        "derivation_policy": derivation_policy,
        "capture_orientation": capture_orientation,
        "orientation": capture_orientation,
        "orientation_summary": capture_orientation,
        "primary_asset_uri": primary_asset.get("uri") if isinstance(primary_asset, Mapping) else None,
        "primary_asset_path": primary_asset.get("path") if isinstance(primary_asset, Mapping) else None,
        "supporting_assets": supporting_assets,
        "fallback_policy": "canonical_only",
        "quality_summary": presentation_quality_summary,
        "readiness": {
            "bundle_status": bundle_status if policy.emit_presentation else "disabled",
            "interactive_demo_readiness": interactive_demo["readiness_state"] if policy.emit_presentation else "disabled",
            "blockers": list(interactive_demo.get("blockers") or []),
        },
        "derivation_mode": policy.allow_generative_completion,
        "authoritative_record": False,
        "world_model_policy": policy.to_dict(),
        "canonical_package_version": None,
        "canonical_package_uri": None,
        "provenance": presentation_provenance,
    }, provenance=presentation_provenance)
    write_json(presentation_dir / "presentation_world_manifest.json", presentation_manifest)

    runtime_demo_provenance = build_provenance_record(
        grounding_level="generated" if policy.emit_presentation else "reconstructed",
        evidence_sources=[scene_memory_manifest_uri, presentation_bundle_uri],
        observation_coverage={"presentation_enabled": policy.emit_presentation},
        confidence=qualification_record.get("confidence"),
        canonical_truth=False,
        presentation_only=True,
    )
    runtime_demo_manifest = with_grounding_fields({
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "status": demo_status,
        "primary_runtime_backend": "site_world_runtime",
        "canonical_world_model": canonical_world_model,
        "runtime_render_source": runtime_render_source,
        "fallback_mode": "arkit_rgbd_last_resort",
        "scene_representation": scene_representation,
        "canonical_artifact_uri": scene_memory_manifest_uri,
        "presentation_artifact_uri": runtime_demo_manifest_uri,
        "presentation_bundle_uri": presentation_bundle_uri if policy.emit_presentation else None,
        "presentation_world_manifest_uri": presentation_world_manifest_uri,
        "presentation_manifest_uri": presentation_world_manifest_uri,
        "preview_simulation_manifest_uri": preview_simulation_manifest_uri,
        "canonical_source": canonical_source,
        "derivation_policy": derivation_policy,
        "capture_orientation": capture_orientation,
        "orientation": capture_orientation,
        "bundle_type": "gsplat_scene_v1",
        "renderer_backend": "gsplat",
        "bundle_status": bundle_status if policy.emit_presentation else "disabled",
        "fallback_policy": "canonical_only",
        "interactive_demo": {
            "readiness_state": interactive_demo["readiness_state"] if policy.emit_presentation else "disabled",
            "blockers": list(interactive_demo.get("blockers") or []),
            "warnings": list(interactive_demo.get("warnings") or []),
            "camera_behavior": camera_behavior,
            "render_inputs": render_inputs,
            "consumer_contract": {
                "site_world_mode": "interactive_presentation_world",
                "presentation_bundle_uri_field": "presentation_bundle_uri",
                "legacy_url_fields": ["ui_base_url", "public_ui_base_url"],
            },
        },
        "derivation_mode": policy.allow_generative_completion,
        "authoritative_record": False,
        "world_model_policy": policy.to_dict(),
        "canonical_package_version": None,
        "canonical_package_uri": None,
        **demo_ui_payload,
        "provenance": runtime_demo_provenance,
    }, provenance=runtime_demo_provenance)
    write_json(presentation_dir / "runtime_demo_manifest.json", runtime_demo_manifest)

    return {
        "scene_memory_manifest_uri": scene_memory_manifest_uri,
        "scene_memory_readiness_uri": scene_memory_readiness_uri,
        "conditioning_bundle_uri": conditioning_bundle_uri,
        "preview_simulation_manifest_uri": preview_simulation_manifest_uri,
        "presentation_bundle_uri": presentation_bundle_uri,
        "presentation_world_manifest_uri": presentation_world_manifest_uri,
        "runtime_demo_manifest_uri": runtime_demo_manifest_uri,
        "scene_memory_status": readiness_payload["status"],
        "preview_simulation_status": preview_manifest["status"],
        "geometry_manifest_uri": geometry_conditioning.get("geometry_manifest_uri"),
        "geometry_summary_uri": geometry_conditioning.get("geometry_summary_uri"),
        "geometry_poses_uri": geometry_conditioning.get("camera_poses_uri"),
        "geometry_intrinsics_uri": geometry_conditioning.get("camera_intrinsics_uri"),
        "geometry_depth_manifest_uri": geometry_conditioning.get("depth_manifest_uri"),
        "geometry_confidence_manifest_uri": geometry_conditioning.get("confidence_manifest_uri"),
        "depth_conditioning": dict(effective_depth_conditioning),
        **adapter_artifacts,
    }


def _disabled_task_targets(scene_id: str, capture_id: str, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": utc_now_iso(),
        "inference_mode": "disabled",
        "video_analysis": {"external_inference": {"status": "skipped", "reason": reason}},
        "manipulation_candidates": [],
        "articulation_hints": [],
        "navigation_hints": [],
        "target_object_ids": [],
        "articulation_required_ids": [],
        "tasks": [],
    }


def _build_runtime_preflight_report(
    *,
    descriptor_path: Path,
    qa_report_path: Path,
    manifest_path: Optional[Path],
    object_index_path: Optional[Path],
    gcs_root: Path,
) -> Dict[str, Any]:
    checks = [
        QualificationGate(
            "gcs_root",
            gcs_root.exists() and gcs_root.is_dir(),
            f"found {gcs_root}" if gcs_root.exists() and gcs_root.is_dir() else f"missing {gcs_root}",
        ),
        QualificationGate(
            "descriptor_access",
            descriptor_path.is_file(),
            f"found {descriptor_path}" if descriptor_path.is_file() else f"missing {descriptor_path}",
        ),
        QualificationGate(
            "qa_report_access",
            qa_report_path.is_file(),
            f"found {qa_report_path}" if qa_report_path.is_file() else f"missing {qa_report_path}",
        ),
        QualificationGate(
            "raw_manifest_access",
            bool(manifest_path and manifest_path.is_file()),
            f"found {manifest_path}" if manifest_path and manifest_path.is_file() else "missing raw manifest",
        ),
        QualificationGate(
            "object_index_access",
            bool(object_index_path and object_index_path.is_file()),
            f"found {object_index_path}" if object_index_path and object_index_path.is_file() else "missing object index",
        ),
    ]
    status = "passed" if all(check.passed for check in checks) else "degraded"
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "status": status,
        "generated_at": utc_now_iso(),
        "checks": [check.to_dict() for check in checks],
    }


def _build_presentation_demo_preflight_report() -> Dict[str, Any]:
    ui_payload = _presentation_demo_ui_payload()
    ui_base_url = str(ui_payload.get("ui_base_url") or "").strip()
    public_ui_base_url = str(ui_payload.get("public_ui_base_url") or "").strip()
    checks = [
        QualificationGate(
            "ui_base_url",
            bool(ui_base_url),
            ui_base_url or "missing BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL",
        ),
        QualificationGate(
            "public_ui_base_url",
            bool(public_ui_base_url),
            public_ui_base_url or "missing BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL",
        ),
    ]
    status = "passed" if any(check.passed for check in checks) else "failed"
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "status": status,
        "generated_at": utc_now_iso(),
        "ui_base_url": ui_base_url or None,
        "public_ui_base_url": public_ui_base_url or None,
        "checks": [check.to_dict() for check in checks],
    }


def _build_capture_package_manifest(
    *,
    descriptor: CaptureDescriptor,
    descriptor_uri: str,
    qa_report_uri: str,
    manifest_uri: Optional[str],
    object_index_uri: Optional[str],
    task_hypothesis_uri: Optional[str],
    storage_root: Path,
    object_index_entries: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    evidence_items = [
        {"name": "descriptor", "uri": descriptor_uri, "exists": _safe_path_exists(descriptor_uri, storage_root)},
        {"name": "qa_report", "uri": qa_report_uri, "exists": _safe_path_exists(qa_report_uri, storage_root)},
        {"name": "raw_manifest", "uri": manifest_uri, "exists": _safe_path_exists(manifest_uri, storage_root)},
        {
            "name": "object_index",
            "uri": object_index_uri,
            "exists": _safe_path_exists(object_index_uri, storage_root),
            "object_count": len(object_index_entries),
        },
        {
            "name": "frames_index",
            "uri": descriptor.frames_index_uri,
            "exists": _safe_path_exists(descriptor.frames_index_uri, storage_root),
        },
        {
            "name": "raw_video",
            "uri": descriptor.raw_video_uri,
            "exists": _safe_path_exists(descriptor.raw_video_uri, storage_root),
        },
        {
            "name": "keyframe",
            "uri": descriptor.keyframe_uri,
            "exists": _safe_path_exists(descriptor.keyframe_uri, storage_root),
        },
        {
            "name": "arkit_poses",
            "uri": descriptor.arkit_poses_uri,
            "exists": _safe_path_exists(descriptor.arkit_poses_uri, storage_root),
        },
        {
            "name": "arkit_intrinsics",
            "uri": descriptor.arkit_intrinsics_uri,
            "exists": _safe_path_exists(descriptor.arkit_intrinsics_uri, storage_root),
        },
        {
            "name": "task_hypothesis",
            "uri": task_hypothesis_uri,
            "exists": _safe_path_exists(task_hypothesis_uri, storage_root),
        },
    ]
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "requested_lanes": list(descriptor.requested_lanes),
        "descriptor_uri": descriptor_uri,
        "qa_report_uri": qa_report_uri,
        "raw_prefix_uri": descriptor.raw_prefix_uri,
        "raw_manifest_uri": manifest_uri,
        "object_index_uri": object_index_uri,
        "task_hypothesis_uri": task_hypothesis_uri,
        "evidence_items": evidence_items,
        "counts": {
            "object_index_count": len(object_index_entries),
            "manipulation_candidates": len(descriptor.manipulation_candidates),
            "articulation_hints": len(descriptor.articulation_hints),
        },
    }


def _build_completeness_scorecard(
    *,
    descriptor: CaptureDescriptor,
    qa_report: Mapping[str, Any],
    manifest: Optional[IOSManifest],
    object_index_uri: Optional[str],
    object_index_entries: List[Mapping[str, Any]],
    metadata_override: Optional[Mapping[str, Any]] = None,
    task_hypothesis_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    qa_status = str(qa_report.get("status") or "missing").strip().lower()
    metadata = metadata_override if isinstance(metadata_override, Mapping) else descriptor.metadata
    structured_intake = _has_structured_intake_from_metadata(
        metadata if isinstance(metadata, Mapping) else {},
        intake_packet_uri=descriptor.intake_packet_uri,
    )
    metric_ready = _modality_supports_metric_automation(descriptor)
    calibration_sufficient = descriptor.capture_modality not in {"glasses_plus_scaffolding", "android_plus_scaffolding"} or bool(descriptor.calibration_assets)
    task_hypothesis_status = str(task_hypothesis_report.get("task_hypothesis_status") or "").strip() if isinstance(task_hypothesis_report, Mapping) else ""
    checks = [
        QualificationGate(
            "qa_report_present",
            qa_status != "missing",
            f"qa_status={qa_status}",
        ),
        QualificationGate(
            "qa_usable",
            qa_status == "passed",
            "qa passed" if qa_status == "passed" else f"qa status is {qa_status}",
        ),
        QualificationGate(
            "raw_manifest_present",
            manifest is not None,
            "raw manifest resolved" if manifest is not None else "raw manifest missing",
        ),
        QualificationGate(
            "object_index_present",
            bool(object_index_uri),
            "object index resolved" if object_index_uri else "object index missing",
        ),
        QualificationGate(
            "object_index_populated",
            len(object_index_entries) > 0,
            f"{len(object_index_entries)} indexed objects",
        ),
        QualificationGate(
            "task_evidence_present",
            bool(descriptor.frames_index_uri or descriptor.raw_video_uri or descriptor.keyframe_uri),
            "capture evidence URIs present"
            if descriptor.frames_index_uri or descriptor.raw_video_uri or descriptor.keyframe_uri
            else "missing capture evidence URIs",
        ),
        QualificationGate(
            "structured_intake_present",
            structured_intake,
            "workflow, zone, and success criteria are present"
            if structured_intake
            else "missing workflow, zone, or success criteria",
        ),
        QualificationGate(
            "task_hypothesis_verified",
            task_hypothesis_status in {"", "accepted", "accepted_with_warnings"},
            "task hypothesis accepted"
            if task_hypothesis_status in {"", "accepted", "accepted_with_warnings"}
            else f"task hypothesis status is {task_hypothesis_status}",
        ),
        QualificationGate(
            "metric_capture_supported",
            metric_ready,
            "capture modality supports geometry-backed automation"
            if metric_ready
            else f"capture modality {descriptor.capture_modality} is not metric-ready",
        ),
        QualificationGate(
            "calibration_sufficient",
            calibration_sufficient,
            "calibration/scaffolding artifacts are sufficient"
            if calibration_sufficient
            else "scaffolding capture is missing calibration assets",
        ),
    ]
    follow_ups = [check.detail for check in checks if not check.passed]
    passed_count = sum(1 for check in checks if check.passed)
    score = round(passed_count / float(len(checks) or 1), 4)
    completeness_status = "sufficient" if all(check.passed for check in checks[1:]) else "need_more_evidence"
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "qa_status": qa_status,
        "capture_modality": descriptor.capture_modality,
        "completeness_status": completeness_status,
        "task_hypothesis_status": task_hypothesis_status or None,
        "score": score,
        "checks": [check.to_dict() for check in checks],
        "follow_ups": follow_ups,
    }


def _build_task_scope_record(
    *,
    descriptor: CaptureDescriptor,
    task_targets_payload: Mapping[str, Any],
    completeness_status: str,
    metadata_override: Optional[Mapping[str, Any]] = None,
    object_index_runtime_blockers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    target_object_ids = [
        str(value)
        for value in task_targets_payload.get("target_object_ids", [])
        if str(value).strip()
    ]
    articulation_required_ids = [
        str(value)
        for value in task_targets_payload.get("articulation_required_ids", [])
        if str(value).strip()
    ]
    tasks = [
        dict(item)
        for item in task_targets_payload.get("tasks", [])
        if isinstance(item, Mapping)
    ]
    metadata = metadata_override if isinstance(metadata_override, Mapping) else (
        descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    )
    assumptions = [
        str(item).strip()
        for item in metadata.get("assumptions", [])
        if str(item).strip()
    ] if isinstance(metadata.get("assumptions"), list) else []
    blockers = [
        str(item).strip()
        for item in metadata.get("known_blockers", [])
        if str(item).strip()
    ] if isinstance(metadata.get("known_blockers"), list) else []
    if not target_object_ids and not tasks:
        blockers.append("Task zone is not yet well scoped from the available evidence.")
    if completeness_status != "sufficient":
        blockers.append("Additional evidence is required before scope can be locked.")
    for blocker in object_index_runtime_blockers or []:
        if blocker not in blockers:
            blockers.append(blocker)
    success_criteria = [
        str(item).strip()
        for item in metadata.get("success_criteria", [])
        if str(item).strip()
    ] if isinstance(metadata.get("success_criteria"), list) else []
    if not success_criteria:
        success_criteria = ["Identify the task zone, key objects, and blockers well enough for buyer review."]
    scope_status = "scoped" if target_object_ids or tasks else "needs_clarification"
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "scope_status": scope_status,
        "task_statement": str(metadata.get("task_statement") or metadata.get("workflow_context") or "unknown_task"),
        "buyer_type": str(metadata.get("buyer_type") or "unknown"),
        "task_zone": metadata.get("task_zone") if isinstance(metadata.get("task_zone"), Mapping) else {},
        "adjacent_workflow_context": str(metadata.get("workflow_context") or "unknown"),
        "target_object_ids": target_object_ids,
        "articulation_required_ids": articulation_required_ids,
        "tasks": tasks,
        "manipulation_candidates": [
            dict(item)
            for item in task_targets_payload.get("manipulation_candidates", [])
            if isinstance(item, Mapping)
        ],
        "articulation_hints": [
            dict(item)
            for item in task_targets_payload.get("articulation_hints", [])
            if isinstance(item, Mapping)
        ],
        "assumptions": assumptions,
        "blockers": blockers,
        "success_criteria": success_criteria,
        "task_hypothesis_status": metadata.get("task_hypothesis_status"),
        "task_hypothesis_confidence": metadata.get("task_hypothesis_confidence"),
    }


def _score_bucket(value: float, detail: str) -> Dict[str, Any]:
    bounded = max(0.0, min(1.0, float(value)))
    return {"score": round(bounded, 4), "detail": detail}


def _build_qualification_record(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    object_index_entries: List[Mapping[str, Any]],
    object_index_runtime_blockers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    completeness_status = str(scorecard.get("completeness_status") or "need_more_evidence")
    qa_status = str(scorecard.get("qa_status") or "missing")
    target_object_ids = scope_record.get("target_object_ids", [])
    articulation_required_ids = scope_record.get("articulation_required_ids", [])
    scope_status = str(scope_record.get("scope_status") or "needs_clarification")
    task_hypothesis_status = str(scope_record.get("task_hypothesis_status") or "").strip()
    metric_ready = _modality_supports_metric_automation(descriptor)
    route_widths = []
    target_distances = []
    grouped = _group_objects_by_entity_type(object_index_entries)
    route_objects = (
        grouped.get("aisle", [])
        + grouped.get("threshold", [])
        + grouped.get("door_type", [])
        + grouped.get("forklift_lane", [])
        + grouped.get("traffic_zone", [])
    )
    for entry in route_objects:
        width = _measure_width(_entry_extents(entry))
        if width > 0.0:
            route_widths.append(width)
    zone_center = _zone_center(
        scope_record.get("task_zone") if isinstance(scope_record.get("task_zone"), Mapping) else {},
        object_index_entries,
    )
    for entry in object_index_entries:
        entry_id = str(entry.get("id") or entry.get("object_id") or "").strip()
        if entry_id and entry_id in target_object_ids:
            target_distances.append(_distance(zone_center, _entry_center(entry)))
    measured_route_width = min(route_widths) if route_widths else None
    max_target_reach = max(target_distances) if target_distances else None

    privacy_restricted = bool(descriptor.metadata.get("privacy_restrictions")) if isinstance(descriptor.metadata, Mapping) else False
    safety_concerns = descriptor.metadata.get("safety_concerns") if isinstance(descriptor.metadata, Mapping) else []
    safety_count = len(safety_concerns) if isinstance(safety_concerns, list) else 0

    rubric = {
        "physical_access": _score_bucket(
            0.9
            if metric_ready and measured_route_width is not None and measured_route_width >= _GENERIC_CAPABILITY_ENVELOPE["minimum_path_width_m"]
            else 0.6
            if metric_ready
            else 0.15,
            (
                f"Measured route width {round(measured_route_width, 4)} m supports clearance checks."
                if metric_ready and measured_route_width is not None
                else "Metric geometry is incomplete for route clearance checks."
            ),
        ),
        "task_repeatability": _score_bucket(
            0.82 if target_object_ids else 0.35,
            "Task targets were inferred from the capture package."
            if target_object_ids
            else "No stable task targets were inferred from the current evidence.",
        ),
        "environmental_conditions": _score_bucket(
            0.82 if qa_status == "passed" else 0.25,
            "Capture QA passed."
            if qa_status == "passed"
            else "Capture QA did not pass, so the environment evidence is unreliable.",
        ),
        "safety_process_constraints": _score_bucket(
            0.45 if safety_count else 0.75,
            "Safety or process constraints were flagged in metadata."
            if safety_count
            else "No explicit safety blockers were supplied in metadata.",
        ),
        "integration_friction": _score_bucket(
            0.5 if articulation_required_ids else 0.78,
            "Articulated targets suggest extra integration complexity."
            if articulation_required_ids
            else "No articulated manipulation requirement was inferred.",
        ),
        "evidence_completeness": _score_bucket(
            min(
                float(scorecard.get("score") or 0.0),
                0.5 if descriptor.evidence_tier == "pre_screen_video" else 1.0,
            ),
            f"Completeness status is {completeness_status} for evidence_tier={descriptor.evidence_tier}.",
        ),
    }

    risks: List[Dict[str, Any]] = []
    if completeness_status != "sufficient":
        risks.append(
            {
                "id": "need_more_evidence",
                "severity": "high",
                "category": "evidence",
                "detail": "Additional capture evidence is required before the opportunity can be routed.",
            }
        )
    if scope_status != "scoped":
        risks.append(
            {
                "id": "scope_ambiguity",
                "severity": "medium",
                "category": "scoping",
                "detail": "Task scope remains ambiguous from the available capture evidence.",
            }
        )
    if task_hypothesis_status == "needs_confirmation":
        risks.append(
            {
                "id": "task_hypothesis_needs_confirmation",
                "severity": "medium",
                "category": "scoping",
                "detail": "AI task hypothesis needs confirmation before the workflow can be trusted.",
            }
        )
    if task_hypothesis_status == "contradicted":
        risks.append(
            {
                "id": "task_hypothesis_contradicted",
                "severity": "high",
                "category": "scoping",
                "detail": "AI task hypothesis contradicts the current capture evidence.",
            }
        )
    if articulation_required_ids:
        risks.append(
            {
                "id": "articulation_complexity",
                "severity": "medium",
                "category": "integration",
                "detail": "Articulated targets indicate a more complex manipulation environment.",
            }
        )
    if descriptor.evidence_tier == "pre_screen_video":
        risks.append(
            {
                "id": "non_metric_capture",
                "severity": "high",
                "category": "geometry",
                "detail": "Pre-screen capture is not sufficient for geometry-backed readiness automation.",
            }
        )
    elif descriptor.capture_modality in {"glasses_plus_scaffolding", "android_plus_scaffolding"} and not bool(descriptor.scaffolding_validation.get("validated_metric_bundle")):
        risks.append(
            {
                "id": "missing_validated_scaffolding",
                "severity": "high",
                "category": "geometry",
                "detail": "Video scaffolding lacks validated scale and pose coverage required for metric checks.",
            }
        )
    if metric_ready and measured_route_width is not None and measured_route_width < _GENERIC_CAPABILITY_ENVELOPE["minimum_path_width_m"]:
        risks.append(
            {
                "id": "route_clearance_risk",
                "severity": "high",
                "category": "geometry",
                "detail": f"Measured minimum route width {round(measured_route_width, 4)} m is below the generic clearance threshold.",
            }
        )
    if metric_ready and max_target_reach is not None and max_target_reach > _GENERIC_CAPABILITY_ENVELOPE["maximum_target_reach_distance_m"]:
        risks.append(
            {
                "id": "reach_risk",
                "severity": "medium",
                "category": "task_fit",
                "detail": f"Inferred target reach distance {round(max_target_reach, 4)} m exceeds the bounded pilot envelope.",
            }
        )
    if privacy_restricted:
        risks.append(
            {
                "id": "privacy_restrictions",
                "severity": "medium",
                "category": "access",
                "detail": "Privacy restrictions may hide decision-critical areas.",
            }
        )
    if object_index_runtime_blockers:
        risks.append(
            {
                "id": "object_index_runtime_missing",
                "severity": "high",
                "category": "runtime",
                "detail": "; ".join(object_index_runtime_blockers),
            }
        )

    blockers = [
        str(item).strip()
        for item in scope_record.get("blockers", [])
        if str(item).strip()
    ]

    rubric_scores = [float(item.get("score") or 0.0) for item in rubric.values()]
    confidence = round(sum(rubric_scores) / float(len(rubric_scores) or 1), 4)
    if scope_status == "scoped":
        confidence = round(min(1.0, confidence + 0.05), 4)
    if task_hypothesis_status == "accepted_with_warnings":
        confidence = round(max(0.0, confidence - 0.08), 4)
    elif task_hypothesis_status == "needs_confirmation":
        confidence = round(max(0.0, confidence - 0.12), 4)
    elif task_hypothesis_status == "contradicted":
        confidence = round(max(0.0, confidence - 0.2), 4)

    if completeness_status != "sufficient":
        readiness_state = "not_ready_yet"
    elif confidence >= 0.8 and not any(risk["severity"] == "high" for risk in risks):
        readiness_state = "ready"
    else:
        readiness_state = "risky"

    advanced_geometry_recommended = metric_ready and completeness_status == "sufficient" and bool(target_object_ids or articulation_required_ids)

    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "readiness_state": readiness_state,
        "capture_modality": descriptor.capture_modality,
        "evidence_tier": descriptor.evidence_tier,
        "confidence": confidence,
        "advanced_geometry_recommended": advanced_geometry_recommended,
        "rubric": rubric,
        "measurements": {
            "minimum_route_width_m": round(measured_route_width, 4) if measured_route_width is not None else None,
            "maximum_target_reach_m": round(max_target_reach, 4) if max_target_reach is not None else None,
        },
        "risks": risks,
        "blockers": blockers,
        "escalation": {
            "required": readiness_state != "ready" or advanced_geometry_recommended,
            "reason": "Need more evidence" if readiness_state == "not_ready_yet" else "Advanced geometry is recommended",
        },
    }


def _build_qualification_brief(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
) -> Dict[str, Any]:
    readiness_state = str(qualification_record.get("readiness_state") or "not_ready_yet")
    completeness_status = str(scorecard.get("completeness_status") or "need_more_evidence")
    risks = qualification_record.get("risks", [])
    risk_summaries = [str(item.get("detail") or "") for item in risks if isinstance(item, Mapping)]
    next_steps: List[str] = []
    if completeness_status != "sufficient":
        next_steps.append("Collect another capture pass that closes evidence gaps called out in the QA scorecard.")
    if str(scope_record.get("scope_status") or "") != "scoped":
        next_steps.append("Clarify the task zone, workflow, and target objects before routing.")
    if bool(qualification_record.get("advanced_geometry_recommended")):
        next_steps.append("Escalate to the advanced geometry lane for richer object-localized geometry outputs.")
    if not next_steps:
        next_steps.append("Route the opportunity handoff to deployment, process, and safety reviewers.")
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "headline": f"{descriptor.scene_id} is {readiness_state.replace('_', ' ')} for qualification routing.",
        "readiness_state": readiness_state,
        "completeness_status": completeness_status,
        "confidence": float(qualification_record.get("confidence") or 0.0),
        "task_statement": scope_record.get("task_statement"),
        "top_risks": risk_summaries[:3],
        "next_steps": next_steps,
    }


def _build_opportunity_handoff(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    brief: Mapping[str, Any],
    config: Any,
    pipeline_dir: Path,
    metadata_override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata_override if isinstance(metadata_override, Mapping) else (
        descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    )
    readiness_state = str(qualification_record.get("readiness_state") or "not_ready_yet")
    completeness_status = str(scorecard.get("completeness_status") or "need_more_evidence")
    confidence = float(qualification_record.get("confidence") or 0.0)
    match_ready = completeness_status == "sufficient" and readiness_state == "ready" and confidence >= 0.8
    recommended_lane = (
        "advanced_geometry"
        if bool(qualification_record.get("advanced_geometry_recommended"))
        else "scene_memory"
        if completeness_status == "sufficient"
        else "qualification"
    )
    site_submission_id = str(
        metadata.get("site_submission_id")
        or getattr(descriptor, "site_submission_id", "")
        or ""
    ).strip()
    buyer_request_id = str(
        metadata.get("buyer_request_id")
        or getattr(descriptor, "buyer_request_id", "")
        or ""
    ).strip()
    capture_job_id = str(
        metadata.get("capture_job_id")
        or getattr(descriptor, "capture_job_id", "")
        or ""
    ).strip()
    upstream_link_blockers = [
        blocker
        for blocker, value in (
            ("missing_site_submission_id", site_submission_id),
            ("missing_buyer_request_id", buyer_request_id),
            ("missing_capture_job_id", capture_job_id),
        )
        if not value
    ]
    opportunity_id = str(metadata.get("opportunity_id") or "").strip() or site_submission_id or descriptor.scene_id
    task_statement = (
        str(metadata.get("task_statement") or "").strip()
        or str(metadata.get("workflow_context") or "").strip()
        or f"Qualification review for {descriptor.scene_id}/{descriptor.capture_id}"
    )
    operating_constraints = _string_list(metadata.get("operating_constraints"))
    if not operating_constraints:
        operating_hours = str(metadata.get("operating_hours") or "").strip()
        if operating_hours:
            operating_constraints = [operating_hours]
    if not operating_constraints:
        operating_constraints = ["Not provided in intake metadata"]

    privacy_security_constraints = _string_list(metadata.get("privacy_restrictions"))
    privacy_security_constraints.extend(
        value
        for value in _string_list(metadata.get("security_restrictions"))
        if value not in privacy_security_constraints
    )
    if not privacy_security_constraints:
        privacy_security_constraints = ["Not provided in intake metadata"]

    known_blockers = _string_list(scope_record.get("blockers"))
    if not known_blockers:
        known_blockers = _string_list(metadata.get("known_blockers"))
    if not known_blockers:
        known_blockers = ["No known blockers supplied"]

    task_zone = scope_record.get("task_zone") if isinstance(scope_record.get("task_zone"), Mapping) else {}
    target_object_ids = _string_list(scope_record.get("target_object_ids"))
    in_scope_zone: Any
    if task_zone:
        in_scope_zone = dict(task_zone)
    elif target_object_ids:
        in_scope_zone = target_object_ids
    else:
        in_scope_zone = descriptor.environment_type_hint or descriptor.scene_id

    success_criteria = _string_list(scope_record.get("success_criteria"))
    target_robot_team = (
        metadata.get("target_robot_team")
        if isinstance(metadata.get("target_robot_team"), Mapping)
        else {}
    )
    handoff = {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "site_submission_id": site_submission_id,
        "buyer_request_id": buyer_request_id,
        "capture_job_id": capture_job_id,
        "upstream_link_truth_state": "verified" if not upstream_link_blockers else "blocked_missing_upstream_ids",
        "upstream_link_blockers": upstream_link_blockers,
        "opportunity_id": opportunity_id,
        "qualification_state": readiness_state,
        "downstream_evaluation_eligibility": readiness_state == "ready",
        "operator_approved_summary": (
            str(metadata.get("operator_approved_summary") or "").strip()
            or str(brief.get("headline") or "").strip()
        ),
        "scoped_task_definition": {
            "task_id": str(metadata.get("task_id") or "").strip() or opportunity_id,
            "scoped_task_statement": task_statement,
            "success_criteria": success_criteria,
            "in_scope_zone": in_scope_zone,
        },
        "site_constraints": {
            "operating_constraints": operating_constraints,
            "privacy_security_constraints": privacy_security_constraints,
            "known_blockers": known_blockers,
        },
        "match_ready": match_ready,
        "recommended_lane": recommended_lane,
        "readiness_state": readiness_state,
        "confidence": confidence,
        "risks": qualification_record.get("risks", []),
        "qualification_focus": "neutral_site_readiness",
        "task_hypothesis_status": metadata.get("task_hypothesis_status"),
        "task_hypothesis_confidence": metadata.get("task_hypothesis_confidence"),
    }
    if target_robot_team or metadata.get("robot_platform") or getattr(config, "robot_type", None):
        robot_platform = (
            str(target_robot_team.get("robot_platform") or "").strip()
            or str(metadata.get("robot_platform") or "").strip()
            or str(getattr(config, "robot_type", "") or "").strip()
        )
        if robot_platform:
            handoff["target_robot_team"] = {
                "team_name_or_id": (
                    str(target_robot_team.get("team_name_or_id") or "").strip()
                    or str(metadata.get("team_name_or_id") or "").strip()
                    or "named_robot_team_required"
                ),
                "robot_platform": robot_platform,
                "embodiment_notes": (
                    str(target_robot_team.get("embodiment_notes") or "").strip()
                    or str(metadata.get("embodiment_notes") or "").strip()
                    or f"Explicit downstream evaluation target for {robot_platform}"
                ),
            }
    return attach_handoff_package_paths(handoff, pipeline_dir=pipeline_dir, metadata=metadata)


def _build_pipeline_summary(
    *,
    bucket: str,
    descriptor_uri: str,
    qa_report_uri: str,
    object_index_uri: Optional[str],
    pipeline_prefix: str,
    pipeline_dir: Path,
    task_targets_payload: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "generated_at": utc_now_iso(),
        "pipeline_prefix": pipeline_prefix,
        "source_uris": {
            "descriptor_uri": descriptor_uri,
            "qa_report_uri": qa_report_uri,
            "object_index_uri": object_index_uri,
            "task_targets_uri": f"gs://{bucket}/{pipeline_prefix}/task_targets.json",
            "site_intake_uri": f"gs://{bucket}/{pipeline_prefix}/site_intake.json",
            "capture_package_manifest_uri": f"gs://{bucket}/{pipeline_prefix}/capture_package_manifest.json",
            "capture_qa_scorecard_uri": f"gs://{bucket}/{pipeline_prefix}/capture_qa_scorecard.json",
            "task_scope_record_uri": f"gs://{bucket}/{pipeline_prefix}/task_scope_record.json",
            "qualification_record_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "qualification_brief_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_brief.json",
            "task_hypothesis_report_uri": f"gs://{bucket}/{pipeline_prefix}/task_hypothesis_report.json",
            "normalized_task_hypothesis_uri": f"gs://{bucket}/{pipeline_prefix}/normalized_task_hypothesis.json",
            "opportunity_handoff_uri": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
            "runtime_preflight_report_uri": f"gs://{bucket}/{pipeline_prefix}/runtime_preflight_report.json",
            "presentation_demo_preflight_report_uri": f"gs://{bucket}/{pipeline_prefix}/presentation_demo_preflight_report.json",
            "human_actions_required_uri": f"gs://{bucket}/{pipeline_prefix}/human_actions_required.json",
            "qualification_quality_report_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_quality_report.json",
        },
        "source_files": {
            "runtime_preflight_report": _local_file_pointer(pipeline_dir / "runtime_preflight_report.json"),
            "presentation_demo_preflight_report": _local_file_pointer(pipeline_dir / "presentation_demo_preflight_report.json"),
            "task_targets": _local_file_pointer(pipeline_dir / "task_targets.json"),
            "site_intake": _local_file_pointer(pipeline_dir / "site_intake.json"),
            "capture_package_manifest": _local_file_pointer(pipeline_dir / "capture_package_manifest.json"),
            "capture_qa_scorecard": _local_file_pointer(pipeline_dir / "capture_qa_scorecard.json"),
            "task_hypothesis_report": _local_file_pointer(pipeline_dir / "task_hypothesis_report.json"),
            "normalized_task_hypothesis": _local_file_pointer(pipeline_dir / "normalized_task_hypothesis.json"),
            "task_scope_record": _local_file_pointer(pipeline_dir / "task_scope_record.json"),
            "qualification_record": _local_file_pointer(pipeline_dir / "qualification_record.json"),
            "qualification_brief": _local_file_pointer(pipeline_dir / "qualification_brief.json"),
            "human_actions_required": _local_file_pointer(pipeline_dir / "human_actions_required.json"),
            "opportunity_handoff": _local_file_pointer(pipeline_dir / "opportunity_handoff.json"),
            "qualification_quality_report": _local_file_pointer(pipeline_dir / "qualification_quality_report.json"),
        },
        "metrics": {
            "completeness_status": str(scorecard.get("completeness_status") or "need_more_evidence"),
            "readiness_state": str(qualification_record.get("readiness_state") or "not_ready_yet"),
            "confidence": float(qualification_record.get("confidence") or 0.0),
            "task_target_count": len(task_targets_payload.get("target_object_ids", []))
            if isinstance(task_targets_payload.get("target_object_ids"), list)
            else 0,
            "risk_count": len(qualification_record.get("risks", []))
            if isinstance(qualification_record.get("risks"), list)
            else 0,
            "object_index_count": len(task_targets_payload.get("object_index_entries", []))
            if isinstance(task_targets_payload.get("object_index_entries"), list)
            else 0,
        },
    }


def _empty_downstream_artifacts() -> Dict[str, Any]:
    return {
        "scene_memory_manifest_uri": None,
        "scene_memory_readiness_uri": None,
        "conditioning_bundle_uri": None,
        "preview_simulation_manifest_uri": None,
        "presentation_bundle_uri": None,
        "presentation_world_manifest_uri": None,
        "runtime_demo_manifest_uri": None,
        "gen3c_adapter_manifest_uri": None,
        "site_world_runtime_adapter_manifest_uri": None,
        "cosmos_transfer_adapter_manifest_uri": None,
        "scene_memory_status": "not_requested",
        "preview_simulation_status": "not_requested",
    }


def _read_geometry_summary(pipeline_dir: Path) -> Dict[str, Any]:
    path = pipeline_dir / "geometry" / "geometry_summary.json"
    payload = _try_read_json(path)
    return payload if isinstance(payload, Mapping) else {}


def _geometry_artifacts(
    *,
    pipeline_dir: Path,
    bucket: str,
    pipeline_prefix: str,
) -> Dict[str, Any]:
    geometry_dir = pipeline_dir / "geometry"
    summary = _read_geometry_summary(pipeline_dir)
    summary_uri = (
        f"gs://{bucket}/{pipeline_prefix}/geometry/geometry_summary.json"
        if (geometry_dir / "geometry_summary.json").is_file()
        else None
    )
    manifest_uri = (
        f"gs://{bucket}/{pipeline_prefix}/geometry/geometry_manifest.json"
        if (geometry_dir / "geometry_manifest.json").is_file()
        else None
    )
    result = {
        "geometry_manifest_uri": manifest_uri,
        "geometry_summary_uri": summary_uri,
        "camera_poses_uri": (
            f"gs://{bucket}/{pipeline_prefix}/geometry/camera/poses.jsonl"
            if (geometry_dir / "camera" / "poses.jsonl").is_file()
            else None
        ),
        "camera_intrinsics_uri": (
            f"gs://{bucket}/{pipeline_prefix}/geometry/camera/intrinsics.json"
            if (geometry_dir / "camera" / "intrinsics.json").is_file()
            else None
        ),
        "depth_manifest_uri": (
            f"gs://{bucket}/{pipeline_prefix}/geometry/depth/depth_manifest.json"
            if (geometry_dir / "depth" / "depth_manifest.json").is_file()
            else None
        ),
        "confidence_manifest_uri": (
            f"gs://{bucket}/{pipeline_prefix}/geometry/confidence/confidence_manifest.json"
            if (geometry_dir / "confidence" / "confidence_manifest.json").is_file()
            else None
        ),
        "status": str(summary.get("status") or "missing"),
        "ready_for_world_model": bool(summary.get("ready_for_world_model")),
        "contract_ready_for_world_model": bool(summary.get("contract_ready_for_world_model")),
        "internal_fallback_ready": bool(summary.get("internal_fallback_ready")),
        "geometry_live_ready": bool(summary.get("geometry_live_ready")),
        "external_market_ready": bool(summary.get("external_market_ready")),
        "site_faithful_market_ready": bool(summary.get("site_faithful_market_ready")),
        "provider_native_result": bool(summary.get("provider_native_result")),
        "geometry_source": str(summary.get("geometry_source") or "missing"),
        "fallback_used": bool(summary.get("fallback_used")),
        "fallback_kind": summary.get("fallback_kind"),
        "launch_blockers": list(summary.get("launch_blockers") or []),
        "canonical_frame_id": summary.get("canonical_frame_id"),
        "scale_status": str(
            ((summary.get("scale_assessment") or {}) if isinstance(summary.get("scale_assessment"), Mapping) else {}).get("status")
            or "missing"
        ),
        "pose_coverage": float(
            ((summary.get("deliverables") or {}) if isinstance(summary.get("deliverables"), Mapping) else {}).get("pose_coverage")
            or 0.0
        ),
        "confidence_coverage": float(
            ((summary.get("deliverables") or {}) if isinstance(summary.get("deliverables"), Mapping) else {}).get("confidence_coverage")
            or 0.0
        ),
        "depth_coverage": float(
            ((summary.get("deliverables") or {}) if isinstance(summary.get("deliverables"), Mapping) else {}).get("depth_coverage")
            or 0.0
        ),
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }
    return result


def _geometry_advisory_payload(geometry_artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    summary = geometry_artifacts.get("summary") if isinstance(geometry_artifacts.get("summary"), Mapping) else {}
    return {
        "status": str(geometry_artifacts.get("status") or "missing"),
        "ready_for_world_model": bool(geometry_artifacts.get("ready_for_world_model")),
        "contract_ready_for_world_model": bool(geometry_artifacts.get("contract_ready_for_world_model")),
        "internal_fallback_ready": bool(geometry_artifacts.get("internal_fallback_ready")),
        "geometry_live_ready": bool(geometry_artifacts.get("geometry_live_ready")),
        "external_market_ready": bool(geometry_artifacts.get("external_market_ready")),
        "site_faithful_market_ready": bool(geometry_artifacts.get("site_faithful_market_ready")),
        "provider_native_result": bool(geometry_artifacts.get("provider_native_result")),
        "geometry_source": str(geometry_artifacts.get("geometry_source") or "missing"),
        "fallback_used": bool(geometry_artifacts.get("fallback_used")),
        "fallback_kind": geometry_artifacts.get("fallback_kind"),
        "launch_blockers": list(geometry_artifacts.get("launch_blockers") or []),
        "scale_status": str(geometry_artifacts.get("scale_status") or "missing"),
        "pose_coverage": float(geometry_artifacts.get("pose_coverage") or 0.0),
        "confidence_coverage": float(geometry_artifacts.get("confidence_coverage") or 0.0),
        "depth_coverage": float(geometry_artifacts.get("depth_coverage") or 0.0),
        "geometry_summary_uri": geometry_artifacts.get("geometry_summary_uri"),
        "geometry_manifest_uri": geometry_artifacts.get("geometry_manifest_uri"),
        "warnings": list(
            ((summary.get("provider") or {}) if isinstance(summary.get("provider"), Mapping) else {}).get("warnings")
            or []
        ),
    }


def _should_run_default_geometry_stage(descriptor: CaptureDescriptor) -> bool:
    if descriptor.capture_source == "iphone" and descriptor.arkit_poses_uri:
        return False
    capture_rights = (
        descriptor.metadata.get("capture_rights")
        if isinstance(descriptor.metadata.get("capture_rights"), Mapping)
        else {}
    )
    return bool(capture_rights.get("derived_scene_generation_allowed", False))


def _requested_downstream_lanes(
    *,
    descriptor: CaptureDescriptor,
    requested_lanes: Optional[List[str]] = None,
) -> List[str]:
    lanes: List[str] = []
    explicit = {str(value or "").strip().lower() for value in (requested_lanes or []) if str(value or "").strip()}
    requested_outputs = {str(value or "").strip().lower() for value in descriptor.requested_outputs if str(value or "").strip()}

    if "scene_memory" in explicit or "evaluation_prep" in explicit:
        lanes.append("scene_memory")
    if requested_outputs.intersection({"managed_tuning", "data_licensing", "deeper_evaluation"}):
        if "scene_memory" not in lanes:
            lanes.append("scene_memory")
    if "evaluation_prep" in explicit or "deeper_evaluation" in requested_outputs:
        lanes.append("evaluation_prep")
    if requested_outputs.intersection({"robot_eval_dataset", "task_evaluation_run"}):
        if "evaluation_prep" not in lanes:
            lanes.append("evaluation_prep")
    return lanes


def _rights_review_required_use_classes(
    *,
    descriptor: CaptureDescriptor,
    requested_lanes: Optional[List[str]] = None,
) -> List[str]:
    """Map requested product lanes to consent use classes.

    Plain qualification can clear from a documented rights packet. Downstream
    generated, robot-eval, training, and licensing products need an explicit use
    class; a location-only consent scope is not enough for those products.
    """
    explicit = {
        str(value or "").strip().lower()
        for value in (requested_lanes or [])
        if str(value or "").strip()
    }
    outputs = {
        str(value or "").strip().lower()
        for value in descriptor.requested_outputs
        if str(value or "").strip()
    }
    required: List[str] = []

    def add(use_class: str) -> None:
        if use_class not in required:
            required.append(use_class)

    if explicit.intersection({"scene_memory", "evaluation_prep"}) or outputs.intersection(
        {
            "preview",
            "preview_simulation",
            "scene_memory",
            "evaluation_prep",
            "managed_tuning",
            "data_licensing",
            "deeper_evaluation",
        }
    ):
        add("derived_generation")
    if explicit.intersection({"evaluation_prep"}) or outputs.intersection(
        {"robot_eval_dataset", "task_evaluation_run", "deeper_evaluation"}
    ):
        add("robot_evaluation")
    if outputs.intersection(
        {"managed_tuning", "post_training_data_package", "training_dataset", "data_licensing"}
    ):
        add("model_training")
    if outputs.intersection({"data_licensing"}):
        add("data_licensing")
    return required


# Privacy pipeline statuses that completed some privacy processing. Delivery
# gates remove fallback/local-proof statuses from this set because those are
# review inputs, not verified removal.
_PRIVACY_POSTPROCESS_CLEARED_STATUSES = frozenset(
    {
        "no_people_detected",
        "person_removed",
        "face_anonymized_fallback",
        "full_frame_redacted_local_proof",
    }
)


def _privacy_postprocess_gate(*, privacy_status: str, delivery_run: bool) -> QualificationGate:
    """Build the privacy_postprocess_gate.

    PIPE-03: for a *delivery* run (one that will build buyer/reviewer-facing
    downstream artifacts), ``not_run`` is NON-passing — privacy must actually have
    executed and cleared. For non-delivery / local flows, ``not_run`` remains
    acceptable so existing test/local pipelines keep working. ``failed_closed`` never
    passes.
    """
    status = str(privacy_status or "").strip().lower()
    passing_statuses = set(_PRIVACY_POSTPROCESS_CLEARED_STATUSES)
    fallback_statuses = {
        "face_anonymized_fallback",
        "full_frame_redacted_local_proof",
    }
    if delivery_run:
        passing_statuses.difference_update(fallback_statuses)
    if not delivery_run:
        passing_statuses.add("not_run")
    passed = status in passing_statuses
    detail = f"privacy_status={privacy_status}"
    if not passed and status == "not_run" and delivery_run:
        detail = (
            f"privacy_status={privacy_status}; delivery runs require privacy post-processing "
            "to run and clear (enable PRIVACY_PIPELINE_ENABLED / production launch mode)"
        )
    elif not passed and status in fallback_statuses and delivery_run:
        detail = (
            f"privacy_status={privacy_status}; delivery runs require verified privacy "
            "removal, not fallback/local redaction proof"
        )
    return QualificationGate("privacy_postprocess_gate", passed, detail)


def _present_artifacts(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }


def _resolve_optional_uri_to_path(uri: Optional[str], storage_root: Path) -> Optional[Path]:
    if not uri:
        return None
    try:
        return ensure_local_uri_path(uri, gcs_root=storage_root, scratch_dir=storage_root / ".downloads")
    except Exception:
        return None


def _ffprobe_video_metrics(video_path: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "default=noprint_wrappers=1:nokey=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise StageError("worldlabs_input_prep", f"ffprobe_failed:{proc.stderr[-400:].strip()}")
    metrics: Dict[str, Any] = {"duration_seconds": 0.0, "size_bytes": 0}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition("=")
        if key == "duration":
            try:
                metrics["duration_seconds"] = float(value)
            except ValueError:
                metrics["duration_seconds"] = 0.0
        elif key == "size":
            try:
                metrics["size_bytes"] = int(float(value))
            except ValueError:
                metrics["size_bytes"] = 0
    return metrics


def _allow_raw_worldlabs_bypass(
    *,
    descriptor: CaptureDescriptor,
    privacy_processing: Mapping[str, Any],
) -> bool:
    if production_launch_mode():
        return False
    metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    for raw in (
        metadata.get("allow_raw_worldlabs_bypass"),
        metadata.get("allowRawWorldlabsBypass"),
        privacy_processing.get("allow_raw_worldlabs_bypass"),
        os.getenv("BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS"),
    ):
        if raw is not None:
            return parse_bool(raw, default=False)
    return False


def _worldlabs_input_labeling(
    *,
    source_id: str,
    privacy_status: Optional[str],
    bypass_allowed: bool,
) -> Dict[str, Any]:
    raw_bypass_used = source_id == "raw_video_uri"
    privacy_safe_input = source_id in {
        "worldlabs_input_video_uri",
        "world_model_video_uri",
        "privacy_processed_video_uri",
    }
    if raw_bypass_used:
        warnings = [
            "Non-production preview only.",
            "Input came from the unredacted raw walkthrough video.",
            "Do not present this preview as privacy-safe buyer media.",
        ]
        review_state = "non_production_unredacted_raw_preview"
    else:
        warnings = []
        review_state = "standard_privacy_safe_preview"
    return {
        "privacy_safe_input": privacy_safe_input,
        "raw_video_bypass_allowed": bool(bypass_allowed),
        "raw_video_bypass_used": raw_bypass_used,
        "unredacted_input": raw_bypass_used,
        "non_production": raw_bypass_used,
        "review_state": review_state,
        "privacy_status": privacy_status,
        "warnings": warnings,
    }


def _worldlabs_transcode_attempts(
    *,
    input_duration: float,
) -> List[Dict[str, Any]]:
    defaults = [
        {"clip_seconds": 30.0, "max_width": 1280, "crf": 28, "maxrate": "4M", "bufsize": "8M", "audio_bitrate": "96k"},
        {"clip_seconds": 24.0, "max_width": 1280, "crf": 30, "maxrate": "3M", "bufsize": "6M", "audio_bitrate": "96k"},
        {"clip_seconds": 20.0, "max_width": 960, "crf": 32, "maxrate": "2M", "bufsize": "4M", "audio_bitrate": "64k"},
    ]
    attempts: List[Dict[str, Any]] = []
    for item in defaults:
        clip_seconds = float(item["clip_seconds"])
        actual_clip = min(clip_seconds, input_duration) if input_duration > 0 else clip_seconds
        trim_applied = input_duration > actual_clip if input_duration > 0 else False
        start_seconds = max(0.0, (input_duration - actual_clip) / 2.0) if trim_applied else 0.0
        end_seconds = min(input_duration, start_seconds + actual_clip) if input_duration > 0 else actual_clip
        attempts.append(
            {
                **item,
                "actual_clip_seconds": actual_clip,
                "trim_applied": trim_applied,
                "clip_start_seconds": start_seconds,
                "clip_end_seconds": end_seconds,
            }
        )
    return attempts


def _worldlabs_source_candidate(
    *,
    descriptor: CaptureDescriptor,
    privacy_processing: Mapping[str, Any],
) -> Dict[str, Any]:
    privacy_status = str(privacy_processing.get("status") or descriptor.privacy_status or "").strip().lower()
    metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    worldlabs_input_video_uri = str(metadata.get("worldlabs_input_video_uri") or "").strip()
    bypass_allowed = production_forces_false(
        "BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS",
        default=_allow_raw_worldlabs_bypass(descriptor=descriptor, privacy_processing=privacy_processing),
    )
    candidates = [
        {
            "source_id": "worldlabs_input_video_uri",
            "uri": worldlabs_input_video_uri,
            "eligible": bool(worldlabs_input_video_uri),
        },
        {
            "source_id": "world_model_video_uri",
            "uri": descriptor.world_model_video_uri,
            "eligible": bool(descriptor.world_model_video_uri),
        },
        {
            "source_id": "privacy_processed_video_uri",
            "uri": descriptor.privacy_processed_video_uri,
            "eligible": bool(descriptor.privacy_processed_video_uri),
        },
        {
            "source_id": "raw_video_uri",
            "uri": descriptor.raw_video_uri,
            "eligible": bool(bypass_allowed and descriptor.raw_video_uri),
        },
    ]
    selected = next((item for item in candidates if item["eligible"] and item["uri"]), None)
    return {
        "privacy_status": privacy_status or None,
        "raw_video_bypass_allowed": bypass_allowed,
        "selected": selected,
        "candidates": [
            {
                "source_id": item["source_id"],
                "uri": item["uri"],
                "eligible": item["eligible"],
            }
            for item in candidates
            if item["uri"]
        ],
    }


def _prepare_worldlabs_input_video(
    *,
    descriptor: CaptureDescriptor,
    privacy_processing: Mapping[str, Any],
    storage_root: Path,
    pipeline_dir: Path,
    bucket: str,
) -> Dict[str, Any]:
    artifact_dir = pipeline_dir / "worldlabs_input"
    ensure_dir(artifact_dir)
    manifest_path = artifact_dir / "worldlabs_input_manifest.json"
    audit_path = pipeline_dir / "worldlabs_input_audit.json"
    output_path = artifact_dir / "worldlabs_input.mp4"
    output_uri = f"gs://{bucket}/{relative_scene_path(output_path, storage_root)}"
    manifest_uri = f"gs://{bucket}/{relative_scene_path(manifest_path, storage_root)}"
    audit_uri = f"gs://{bucket}/{relative_scene_path(audit_path, storage_root)}"
    max_duration_seconds = 30.0
    max_size_bytes = 100_000_000

    selection = _worldlabs_source_candidate(descriptor=descriptor, privacy_processing=privacy_processing)
    selected = selection.get("selected") if isinstance(selection, Mapping) else None
    source_uri = str(selected.get("uri") or "").strip() if isinstance(selected, Mapping) else ""
    source_id = str(selected.get("source_id") or "").strip() if isinstance(selected, Mapping) else ""
    input_labeling = _worldlabs_input_labeling(
        source_id=source_id,
        privacy_status=selection.get("privacy_status") if isinstance(selection, Mapping) else None,
        bypass_allowed=bool(selection.get("raw_video_bypass_allowed")) if isinstance(selection, Mapping) else False,
    )
    if not source_uri:
        payload = {
            "schema_version": "v1",
            "status": "blocked",
            "generated_at": utc_now_iso(),
            "reason": "no_worldlabs_source_video",
            "selected_video_source_id": None,
            "selected_video_uri": None,
            "video_candidates": selection.get("candidates") if isinstance(selection, Mapping) else [],
            "input_labeling": input_labeling,
            "output_video_uri": None,
        }
        write_json(manifest_path, payload)
        write_json(
            audit_path,
            {
                "schema_version": "v1",
                "status": "blocked",
                "generated_at": utc_now_iso(),
                "reason": "no_worldlabs_source_video",
                "privacy_safe_input": False,
                "selected_video_source_id": None,
                "selected_video_uri": None,
                "output_video_uri": None,
                "source_manifest_uri": None,
            },
        )
        return {
            "status": "blocked",
            "manifest_uri": manifest_uri,
            "audit_uri": audit_uri,
            "output_video_uri": None,
            "manifest_path": str(manifest_path),
            "audit_path": str(audit_path),
            "output_path": None,
            "input_labeling": input_labeling,
        }

    source_path = _resolve_optional_uri_to_path(source_uri, storage_root)
    if not source_path or not source_path.is_file():
        source_manifest_uri = str(privacy_processing.get("privacy_manifest_uri") or "").strip() or None
        source_is_final_walkthrough = source_uri.rstrip("/").endswith(
            "/privacy/final_walkthrough.mov"
        ) or source_uri.rstrip("/").endswith("/privacy/final_walkthrough.mp4")
        derived_from_final_walkthrough = bool(
            source_is_final_walkthrough
            or str(privacy_processing.get("privacy_processed_video_uri") or "").strip().rstrip("/")
            == source_uri.rstrip("/")
            or str(privacy_processing.get("world_model_video_uri") or "").strip().rstrip("/")
            == source_uri.rstrip("/")
        )
        payload = {
            "schema_version": "v1",
            "status": "blocked",
            "generated_at": utc_now_iso(),
            "reason": "source_video_missing",
            "selected_video_source_id": source_id or None,
            "selected_video_uri": source_uri,
            "video_candidates": selection.get("candidates") if isinstance(selection, Mapping) else [],
            "input_labeling": input_labeling,
            "output_video_uri": None,
        }
        audit_payload = {
            "schema_version": "v1",
            "status": "blocked",
            "generated_at": payload["generated_at"],
            "reason": "source_video_missing",
            "privacy_safe_input": False,
            "selected_video_source_id": source_id or None,
            "selected_video_uri": source_uri,
            "source_manifest_uri": source_manifest_uri,
            "source_is_final_walkthrough": source_is_final_walkthrough,
            "derivative_of_final_walkthrough": derived_from_final_walkthrough,
            "output_video_uri": None,
            "output_manifest_uri": manifest_uri,
            "input_labeling": input_labeling,
        }
        write_json(manifest_path, payload)
        write_json(audit_path, audit_payload)
        if production_launch_mode():
            raise StageError("worldlabs_input_prep", f"source_video_missing:{source_uri}")
        return {
            "status": "blocked",
            "manifest_uri": manifest_uri,
            "audit_uri": audit_uri,
            "output_video_uri": None,
            "manifest_path": str(manifest_path),
            "audit_path": str(audit_path),
            "output_path": None,
            "input_labeling": input_labeling,
            "audit_payload": audit_payload,
        }

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise StageError("worldlabs_input_prep", "ffmpeg_not_found")

    input_metrics = _ffprobe_video_metrics(source_path)
    input_duration = float(input_metrics.get("duration_seconds") or 0.0)
    input_size_bytes = int(input_metrics.get("size_bytes") or 0)
    attempts = _worldlabs_transcode_attempts(input_duration=input_duration)
    output_metrics: Dict[str, Any] = {}
    selected_attempt: Dict[str, Any] = {}
    attempt_reports: List[Dict[str, Any]] = []
    compliant = False
    last_error = ""
    for attempt_index, attempt in enumerate(attempts, start=1):
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{float(attempt['clip_start_seconds']):.3f}",
                "-i",
                str(source_path),
                "-t",
                f"{float(attempt['actual_clip_seconds']):.3f}",
                "-vf",
                f"scale='min({int(attempt['max_width'])},iw)':-2",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(int(attempt["crf"])),
                "-maxrate",
                str(attempt["maxrate"]),
                "-bufsize",
                str(attempt["bufsize"]),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-b:a",
                str(attempt["audio_bitrate"]),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not output_path.is_file():
            last_error = proc.stderr[-400:].strip()
            attempt_reports.append(
                {
                    "attempt_index": attempt_index,
                    **attempt,
                    "status": "failed",
                    "ffmpeg_error": last_error or "ffmpeg_failed",
                }
            )
            continue
        output_metrics = _ffprobe_video_metrics(output_path)
        output_duration = float(output_metrics.get("duration_seconds") or 0.0)
        output_size_bytes = int(output_metrics.get("size_bytes") or 0)
        compliant = output_duration <= max_duration_seconds + 0.25 and output_size_bytes <= max_size_bytes
        selected_attempt = dict(attempt)
        attempt_reports.append(
            {
                "attempt_index": attempt_index,
                **attempt,
                "status": "ready" if compliant else "review_required",
                "output_duration_seconds": round(output_duration, 3),
                "output_size_bytes": output_size_bytes,
                "duration_ok": output_duration <= max_duration_seconds + 0.25,
                "size_ok": output_size_bytes <= max_size_bytes,
            }
        )
        if compliant:
            break
    if not selected_attempt:
        raise StageError("worldlabs_input_prep", f"ffmpeg_failed:{last_error or 'unknown'}")

    output_duration = float(output_metrics.get("duration_seconds") or 0.0)
    output_size_bytes = int(output_metrics.get("size_bytes") or 0)
    source_manifest_uri = str(privacy_processing.get("privacy_manifest_uri") or "").strip() or None
    source_is_final_walkthrough = source_uri.rstrip("/").endswith("/privacy/final_walkthrough.mov") or source_uri.rstrip("/").endswith("/privacy/final_walkthrough.mp4")
    derived_from_final_walkthrough = bool(
        source_is_final_walkthrough
        or str(privacy_processing.get("privacy_processed_video_uri") or "").strip().rstrip("/") == source_uri.rstrip("/")
        or str(privacy_processing.get("world_model_video_uri") or "").strip().rstrip("/") == source_uri.rstrip("/")
    )
    source_checksum = relative_artifact_checksum(source_path)
    output_checksum = relative_artifact_checksum(output_path)
    privacy_safe_input = bool(input_labeling.get("privacy_safe_input")) and not bool(input_labeling.get("raw_video_bypass_used")) and derived_from_final_walkthrough
    audit_payload = {
        "schema_version": "v1",
        "status": "ready" if privacy_safe_input and compliant else "blocked" if production_launch_mode() else "review_required",
        "generated_at": utc_now_iso(),
        "selected_video_source_id": source_id or None,
        "selected_video_uri": source_uri,
        "source_manifest_uri": source_manifest_uri,
        "source_checksum_sha256": source_checksum,
        "source_is_final_walkthrough": source_is_final_walkthrough,
        "derivative_of_final_walkthrough": derived_from_final_walkthrough,
        "privacy_safe_input": privacy_safe_input,
        "raw_video_bypass_used": bool(input_labeling.get("raw_video_bypass_used")),
        "output_video_uri": output_uri,
        "output_manifest_uri": manifest_uri,
        "output_checksum_sha256": output_checksum,
        "input_labeling": input_labeling,
    }
    if production_launch_mode() and not privacy_safe_input:
        write_json(audit_path, audit_payload)
        raise StageError("worldlabs_input_prep", "production_worldlabs_input_not_privacy_safe")
    payload = {
        "schema_version": "v1",
        "status": "ready" if compliant else "review_required",
        "generated_at": utc_now_iso(),
        "selected_video_source_id": source_id or None,
        "selected_video_uri": source_uri,
        "video_candidates": selection.get("candidates") if isinstance(selection, Mapping) else [],
        "input_labeling": input_labeling,
        "input_metrics": {
            "duration_seconds": round(input_duration, 3),
            "size_bytes": input_size_bytes,
        },
        "selection_policy": {
            "max_duration_seconds": max_duration_seconds,
            "max_size_bytes": max_size_bytes,
            "window_strategy": "center_window",
            "transcode_strategy": "iterative_trim_and_compress",
            "selected_attempt": {
                "clip_seconds": round(float(selected_attempt.get("actual_clip_seconds") or 0.0), 3),
                "trim_applied": bool(selected_attempt.get("trim_applied")),
                "clip_start_seconds": round(float(selected_attempt.get("clip_start_seconds") or 0.0), 3),
                "clip_end_seconds": round(float(selected_attempt.get("clip_end_seconds") or 0.0), 3),
                "max_width": int(selected_attempt.get("max_width") or 0),
                "crf": int(selected_attempt.get("crf") or 0),
                "maxrate": selected_attempt.get("maxrate"),
                "bufsize": selected_attempt.get("bufsize"),
                "audio_bitrate": selected_attempt.get("audio_bitrate"),
            },
            "attempts": attempt_reports,
        },
        "output_metrics": {
            "duration_seconds": round(output_duration, 3),
            "size_bytes": output_size_bytes,
            "format": "mp4",
        },
        "compliance": {
            "duration_ok": output_duration <= max_duration_seconds + 0.25,
            "size_ok": output_size_bytes <= max_size_bytes,
            "ready_for_worldlabs": compliant,
        },
        "output_video_uri": output_uri,
        "output_video_path": str(output_path),
        "input_audit_uri": audit_uri,
        "input_audit_path": str(audit_path),
        "input_checksum_sha256": source_checksum,
        "output_checksum_sha256": output_checksum,
    }
    write_json(manifest_path, payload)
    write_json(audit_path, audit_payload)
    return {
        "status": payload["status"],
        "manifest_uri": manifest_uri,
        "audit_uri": audit_uri,
        "output_video_uri": output_uri,
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
        "output_path": str(output_path),
        "payload": payload,
        "audit_payload": audit_payload,
        "input_labeling": input_labeling,
    }


def _build_world_model_fit_summary(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    capture_fidelity_review: Mapping[str, Any],
    privacy_processing: Mapping[str, Any],
    metadata: Mapping[str, Any],
    geometry_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rights = _capture_rights(metadata)
    review_status = str(capture_fidelity_review.get("status") or "failed").strip().lower()
    review_scores = capture_fidelity_review.get("scores") if isinstance(capture_fidelity_review.get("scores"), Mapping) else {}
    assessments = capture_fidelity_review.get("assessments") if isinstance(capture_fidelity_review.get("assessments"), Mapping) else {}
    findings = capture_fidelity_review.get("findings") if isinstance(capture_fidelity_review.get("findings"), Mapping) else {}
    privacy_status = str(privacy_processing.get("status") or "not_run").strip().lower()
    reasons: List[str] = []
    fit_status = "review_required"

    if review_status != "succeeded":
        reasons.append("Gemini multimodal review is required before alpha scoring can complete.")
        fit_status = "review_required"
    if not rights["derived_scene_generation_allowed"]:
        reasons.append("Capture rights do not yet allow derived scene generation.")
        fit_status = "not_permitted"
    if str(scorecard.get("completeness_status") or "") != "sufficient":
        reasons.append("Capture evidence is still incomplete for downstream world-model work.")
        fit_status = "review_required"
    if descriptor.evidence_tier == "pre_screen_video":
        reasons.append("Capture remains pre-screen video only and is not yet world-model ready.")
        fit_status = "review_required"
    if privacy_status == "failed_closed":
        reasons.append("Privacy post-processing failed closed, so buyer-safe world-model media cannot be published.")
        fit_status = "review_required"
    elif privacy_status == "not_run":
        reasons.append("Privacy post-processing has not completed for world-model media.")
        fit_status = "review_required"

    coverage_score = float(review_scores.get("coverage") or 0.0)
    world_model_fitness = float(review_scores.get("world_model_fitness") or 0.0)
    if (
        review_status == "succeeded"
        and rights["derived_scene_generation_allowed"]
        and privacy_status
        in {
            "no_people_detected",
            "person_removed",
            "face_anonymized_fallback",
            "full_frame_redacted_local_proof",
        }
    ):
        if coverage_score >= 0.7 and world_model_fitness >= 0.72 and descriptor.evidence_tier != "pre_screen_video":
            fit_status = "good_candidate"
        elif fit_status != "not_permitted":
            fit_status = "review_required"
    if coverage_score < 0.7:
        reasons.append("Gemini review found missing views or weak coverage for scene reconstruction.")
    for key, message in (
        ("blur", "Gemini review found blur levels that may limit world-model quality."),
        ("lighting", "Gemini review found lighting instability that may reduce world-model quality."),
        ("motion_speed", "Gemini review found camera speed or pacing issues that reduce usable evidence."),
        ("task_zone_completeness", "Gemini review found incomplete capture of the task-relevant zone."),
        ("depth_and_spatial_conditioning", "Gemini review found weak depth or spatial conditioning for world-model generation."),
    ):
        assessment = assessments.get(key)
        if isinstance(assessment, Mapping) and str(assessment.get("status") or "").strip() in {"poor", "review_required"}:
            reasons.append(message)
    if _string_list(findings.get("blur_observations")):
        reasons.append("Gemini review found blur or motion clarity issues that may limit reconstruction quality.")
    if _string_list(findings.get("lighting_observations")):
        reasons.append("Gemini review found lighting instability that may reduce world-model quality.")
    if _string_list(findings.get("occlusion_observations")):
        reasons.append("Gemini review found occlusions or hidden zones that need another pass.")

    geometry = geometry_summary if isinstance(geometry_summary, Mapping) else {}
    geometry_scale = geometry.get("scale_assessment") if isinstance(geometry.get("scale_assessment"), Mapping) else {}
    geometry_deliverables = geometry.get("deliverables") if isinstance(geometry.get("deliverables"), Mapping) else {}
    advisory_geometry = {
        "status": str(geometry.get("status") or "missing"),
        "ready_for_world_model": bool(geometry.get("ready_for_world_model")),
        "contract_ready_for_world_model": bool(geometry.get("contract_ready_for_world_model")),
        "internal_fallback_ready": bool(geometry.get("internal_fallback_ready")),
        "geometry_live_ready": bool(geometry.get("geometry_live_ready")),
        "external_market_ready": bool(geometry.get("external_market_ready")),
        "site_faithful_market_ready": bool(geometry.get("site_faithful_market_ready")),
        "provider_native_result": bool(geometry.get("provider_native_result")),
        "geometry_source": str(geometry.get("geometry_source") or "missing"),
        "fallback_used": bool(geometry.get("fallback_used")),
        "fallback_kind": geometry.get("fallback_kind"),
        "launch_blockers": list(geometry.get("launch_blockers") or []),
        "scale_status": str(geometry_scale.get("status") or "missing"),
        "pose_coverage": float(geometry_deliverables.get("pose_coverage") or 0.0),
        "confidence_coverage": float(geometry_deliverables.get("confidence_coverage") or 0.0),
        "depth_coverage": float(geometry_deliverables.get("depth_coverage") or 0.0),
    }
    if geometry and advisory_geometry["status"] == "completed" and advisory_geometry["ready_for_world_model"]:
        reasons.append(
            "Advisory geometry conditioning is available for downstream world-model work."
        )

    return {
        "schema_version": "v1",
        "status": fit_status,
        "confidence": capture_fidelity_review.get("confidence"),
        "world_model_fitness_score": round(world_model_fitness, 4),
        "coverage_score": round(coverage_score, 4),
        "readiness_state": qualification_record.get("readiness_state"),
        "derived_scene_generation_allowed": rights["derived_scene_generation_allowed"],
        "privacy_status": privacy_status,
        "privacy_mode": privacy_processing.get("mode"),
        "world_model_video_uri": descriptor.preferred_world_model_video_uri,
        "advisory_geometry": advisory_geometry,
        "assessment_statuses": {
            key: (value.get("status") if isinstance(value, Mapping) else None)
            for key, value in assessments.items()
        },
        "reasons": reasons,
        "recommended_next_step": "scene_memory" if fit_status == "good_candidate" else "recapture_or_review",
    }


def _build_capturer_payout_recommendation(
    *,
    descriptor: CaptureDescriptor,
    capture_fidelity_review: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    rights = _capture_rights(metadata)
    review_status = str(capture_fidelity_review.get("status") or "failed").strip().lower()
    scores = capture_fidelity_review.get("scores") if isinstance(capture_fidelity_review.get("scores"), Mapping) else {}
    bonus_signals = capture_fidelity_review.get("bonus_signals") if isinstance(capture_fidelity_review.get("bonus_signals"), Mapping) else {}
    assessments = capture_fidelity_review.get("assessments") if isinstance(capture_fidelity_review.get("assessments"), Mapping) else {}
    payout_quality = float(scores.get("payout_quality") or 0.0)
    confidence = float(capture_fidelity_review.get("confidence") or 0.0)
    base_payout_cents = int(descriptor.quoted_payout_cents or 4500)
    reasons: List[str] = []
    recommendation_status = "review_required"
    recommended_payout_cents: Optional[int] = None
    bonus_breakdown: List[Dict[str, Any]] = []

    def _bonus_score(key: str) -> float:
        raw = bonus_signals.get(key)
        if isinstance(raw, Mapping):
            try:
                return max(0.0, min(1.0, float(raw.get("score") or 0.0)))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _bonus_reason(key: str, fallback: str) -> str:
        raw = bonus_signals.get(key)
        if isinstance(raw, Mapping):
            text = str(raw.get("reason") or "").strip()
            if text:
                return text
        return fallback

    if not rights["capture_contributor_payout_eligible"]:
        reasons.append("Capture is not yet marked payout-eligible in the source rights metadata.")
    if str(rights.get("consent_status") or "unknown").strip().lower() not in {"documented", "policy_only"}:
        reasons.append("Consent status is not yet strong enough for an automated payout recommendation.")
    if review_status != "succeeded":
        reasons.append("Gemini multimodal quality review is incomplete.")

    if not reasons:
        multiplier = 1.0
        bonus_specs = [
            ("complete_coverage", "complete_coverage_bonus", 0.25, "Gemini reviewed zone coverage for the whole task area."),
            ("multi_pass", "multi_pass_bonus", 0.50, "Gemini reviewed whether the capture revisited areas from multiple angles."),
            ("lidar_depth", "lidar_depth_bonus", 1.00, "Gemini reviewed depth and spatial-conditioning quality."),
            ("steady_walkthrough", "steady_walkthrough_bonus", 0.20, "Gemini reviewed pacing, steadiness, and rescan behavior."),
        ]
        for signal_key, label, max_bonus, fallback_reason in bonus_specs:
            signal_score = _bonus_score(signal_key)
            bonus_fraction = round(signal_score * max_bonus, 4)
            bonus_cents = int(round(base_payout_cents * bonus_fraction / 100.0) * 100)
            bonus_breakdown.append(
                {
                    "id": label,
                    "label": label.replace("_", " "),
                    "score": round(signal_score, 4),
                    "max_bonus_percent": round(max_bonus * 100, 2),
                    "awarded_bonus_percent": round(bonus_fraction * 100, 2),
                    "awarded_bonus_cents": bonus_cents,
                    "reason": _bonus_reason(signal_key, fallback_reason),
                }
            )
            multiplier += bonus_fraction
        multiplier *= 0.8 + (payout_quality * 0.4)
        for key in ("blur", "lighting", "motion_speed", "coverage_completeness"):
            assessment = assessments.get(key)
            if isinstance(assessment, Mapping) and str(assessment.get("status") or "").strip() == "poor":
                multiplier = min(multiplier, 1.0)
                reasons.append("Poor raw-video quality reduces the payout recommendation back toward baseline.")
                break
        if confidence < 0.65:
            multiplier = min(multiplier, 1.0)
            reasons.append("Low Gemini confidence caps the payout recommendation at the baseline rate.")
        recommended_payout_cents = int(round(base_payout_cents * multiplier / 100.0) * 100)
        if multiplier >= 1.08:
            recommendation_status = "bonus"
        elif multiplier <= 0.92:
            recommendation_status = "discount"
        else:
            recommendation_status = "baseline"

    return {
        "schema_version": "v1",
        "status": recommendation_status,
        # Explicit eligibility decision: True only when rights metadata marks the
        # contributor payout-eligible, consent is strong enough, and the quality
        # review succeeded so a recommendation was actually computed. Downstream
        # gates must read this field instead of inferring eligibility from the
        # presence of a quote or recommended amount.
        "eligible_for_payout": recommended_payout_cents is not None,
        "base_payout_cents": base_payout_cents,
        "recommended_payout_cents": recommended_payout_cents,
        "confidence": capture_fidelity_review.get("confidence"),
        "payout_quality_score": round(payout_quality, 4),
        "bonus_breakdown": bonus_breakdown,
        "reasons": reasons,
        "final_authority": "webapp_ops_review",
    }


def _build_provenance_summary(
    *,
    descriptor_uri: str,
    qa_report_uri: str,
    pipeline_prefix: str,
    bucket: str,
    capture_fidelity_review: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    provenance = build_provenance_record(
        grounding_level="observed",
        evidence_sources=[
            descriptor_uri,
            qa_report_uri,
            f"gs://{bucket}/{pipeline_prefix}/task_scope_record.json",
            f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
        ],
        observation_coverage={
            "gemini_review_status": capture_fidelity_review.get("status"),
            "consent_status": _capture_rights(metadata).get("consent_status"),
        },
        confidence=capture_fidelity_review.get("confidence"),
        canonical_truth=True,
        presentation_only=False,
        extra={
            "provider_name": capture_fidelity_review.get("provider_name"),
            "provider_model": capture_fidelity_review.get("provider_model"),
        },
    )
    return {
        "schema_version": "v1",
        "status": "grounded",
        "record": provenance,
    }


def _apply_capture_fidelity_to_qualification(
    *,
    qualification_record: Mapping[str, Any],
    capture_fidelity_review: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = dict(qualification_record)
    risks = [
        dict(item)
        for item in qualification_record.get("risks", [])
        if isinstance(item, Mapping)
    ]
    review_status = str(capture_fidelity_review.get("status") or "failed").strip().lower()
    scores = capture_fidelity_review.get("scores") if isinstance(capture_fidelity_review.get("scores"), Mapping) else {}
    assessments = capture_fidelity_review.get("assessments") if isinstance(capture_fidelity_review.get("assessments"), Mapping) else {}
    findings = capture_fidelity_review.get("findings") if isinstance(capture_fidelity_review.get("findings"), Mapping) else {}
    readiness_state = str(payload.get("readiness_state") or "not_ready_yet")
    confidence = float(payload.get("confidence") or 0.0)

    if review_status != "succeeded":
        risks.append(
            {
                "id": "gemini_multimodal_review_missing",
                "severity": "high",
                "category": "fidelity",
                "detail": "Gemini multimodal capture review is required before alpha scoring can complete.",
            }
        )
        readiness_state = "not_ready_yet"
        confidence = min(confidence, 0.45)
    else:
        coverage = float(scores.get("coverage") or 0.0)
        world_model_fitness = float(scores.get("world_model_fitness") or 0.0)
        task_understanding = float(scores.get("task_understanding") or 0.0)
        confidence = round(min(confidence, (confidence + float(capture_fidelity_review.get("confidence") or 0.0) + coverage + task_understanding) / 4.0), 4)
        if coverage < 0.7:
            risks.append(
                {
                    "id": "gemini_missing_views",
                    "severity": "high",
                    "category": "coverage",
                    "detail": "Gemini review found missing or weakly covered views in the capture.",
                }
            )
            readiness_state = "not_ready_yet"
        elif world_model_fitness < 0.72 and readiness_state == "ready":
            risks.append(
                {
                    "id": "gemini_world_model_review_required",
                    "severity": "medium",
                    "category": "fidelity",
                    "detail": "Gemini review found that world-model suitability still needs review.",
                }
            )
            readiness_state = "risky"
        for key, risk_id, detail in (
            ("blur", "gemini_blur_detected", "Gemini review found blur that may reduce world-model quality."),
            ("lighting", "gemini_lighting_instability", "Gemini review found unstable or poor lighting in the walkthrough."),
            ("motion_speed", "gemini_excessive_motion_speed", "Gemini review found camera speed or pacing issues in the walkthrough."),
            ("coverage_completeness", "gemini_incomplete_scene_coverage", "Gemini review found incomplete scene coverage."),
            ("task_zone_completeness", "gemini_incomplete_task_zone", "Gemini review found incomplete task-zone coverage."),
            ("occlusion_and_hidden_zone", "gemini_hidden_zone_risk", "Gemini review found occlusions or hidden-zone risks."),
            ("depth_and_spatial_conditioning", "gemini_weak_spatial_conditioning", "Gemini review found weak depth or spatial conditioning."),
        ):
            assessment = assessments.get(key)
            if not isinstance(assessment, Mapping):
                continue
            status = str(assessment.get("status") or "").strip()
            if status not in {"poor", "review_required"}:
                continue
            severity = "high" if key in {"coverage_completeness", "task_zone_completeness", "depth_and_spatial_conditioning"} else "medium"
            risks.append(
                {
                    "id": risk_id,
                    "severity": severity,
                    "category": "quality",
                    "detail": detail,
                }
            )
            if status == "poor":
                readiness_state = "not_ready_yet"
            elif readiness_state == "ready":
                readiness_state = "risky"
        if _string_list(findings.get("blur_observations")):
            risks.append(
                {
                    "id": "gemini_blur_or_motion",
                    "severity": "medium",
                    "category": "quality",
                    "detail": "Gemini review found blur or motion stability issues in the walkthrough.",
                }
            )
            if readiness_state == "ready":
                readiness_state = "risky"

    if not _capture_rights(metadata)["derived_scene_generation_allowed"]:
        payload["advanced_geometry_recommended"] = False

    payload["risks"] = risks
    payload["readiness_state"] = readiness_state
    payload["confidence"] = round(confidence, 4)
    payload["alpha_scoring_status"] = review_status
    return payload


_GENERIC_CAPABILITY_ENVELOPE = {
    "minimum_path_width_m": 0.95,
    "preferred_path_width_m": 1.15,
    "maximum_threshold_height_m": 0.04,
    "maximum_target_reach_distance_m": 1.1,
    "maximum_workcell_span_m": 2.5,
    "maximum_hidden_zone_bound": MAXIMUM_HIDDEN_ZONE_BOUND,
    "maximum_uncertainty_score": 0.3,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _entry_center(entry: Mapping[str, Any]) -> List[float]:
    box = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else {}
    center = box.get("center") if isinstance(box.get("center"), list) else [0.0, 0.0, 0.0]
    return [_safe_float(center[idx] if idx < len(center) else 0.0, 0.0) for idx in range(3)]


def _entry_extents(entry: Mapping[str, Any]) -> List[float]:
    box = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else {}
    extents = box.get("extents") if isinstance(box.get("extents"), list) else [0.0, 0.0, 0.0]
    values = [_safe_float(extents[idx] if idx < len(extents) else 0.0, 0.0) for idx in range(3)]
    return [max(0.0, value) for value in values]


def _distance(a: List[float], b: List[float]) -> float:
    return sum((a[idx] - b[idx]) ** 2 for idx in range(3)) ** 0.5


def _measure_width(extents: List[float]) -> float:
    planar = [value for value in extents[:2] if value > 0.0]
    if not planar:
        return 0.0
    return min(planar)


def _zone_center(task_zone: Mapping[str, Any], object_index_entries: List[Mapping[str, Any]]) -> List[float]:
    if isinstance(task_zone.get("center"), list):
        center = task_zone.get("center")
        return [_safe_float(center[idx] if idx < len(center) else 0.0, 0.0) for idx in range(3)]
    if object_index_entries:
        centers = [_entry_center(entry) for entry in object_index_entries if isinstance(entry, Mapping)]
        if centers:
            return [
                sum(center[idx] for center in centers) / float(len(centers))
                for idx in range(3)
            ]
    return [0.0, 0.0, 0.0]


def _group_objects_by_entity_type(object_index_entries: List[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for entry in object_index_entries:
        if not isinstance(entry, Mapping):
            continue
        label = str(entry.get("label") or entry.get("name") or "object").strip()
        entity = classify_industrial_entity(label)
        grouped.setdefault(entity.entity_type, []).append(entry)
    return grouped


def _build_scene_graph(
    *,
    descriptor: CaptureDescriptor,
    scope_record: Mapping[str, Any],
    object_index_entries: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    for entry in object_index_entries:
        if not isinstance(entry, Mapping):
            continue
        node_id = str(entry.get("id") or entry.get("object_id") or "").strip()
        if not node_id:
            continue
        label = str(entry.get("label") or entry.get("name") or "object").strip()
        obb = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else {}
        entity = classify_industrial_entity(label)
        nodes.append(
            {
                "id": node_id,
                "type": "object",
                "label": label,
                "category": entity.entity_type,
                "tags": industrial_tags_for_label(label),
                "geometry": dict(obb),
                "center_m": _entry_center(entry),
                "extents_m": _entry_extents(entry),
                "ontology": entity.to_dict(),
            }
        )
    task_zone = scope_record.get("task_zone") if isinstance(scope_record.get("task_zone"), Mapping) else {}
    zone_center = _zone_center(task_zone, object_index_entries)
    if task_zone:
        nodes.append(
            {
                "id": "task_zone",
                "type": "zone",
                "label": str(task_zone.get("label") or "task_zone"),
                "attributes": dict(task_zone),
                "center_m": zone_center,
                "tags": ["task_zone", *derive_capture_plan_tags([task_zone.get("label")])],
            }
        )
    metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    for system in _string_list(metadata.get("adjacent_systems")):
        nodes.append(
            {
                "id": f"system:{system}",
                "type": "system",
                "label": system,
                "tags": ["adjacent_system"],
            }
        )
    for plan_tag in derive_capture_plan_tags(descriptor.coverage_plan):
        nodes.append(
            {
                "id": f"capture_plan:{plan_tag}",
                "type": "capture_plan_hint",
                "label": plan_tag,
                "tags": ["capture_plan_hint", plan_tag],
            }
        )
    edges: List[Dict[str, Any]] = []
    for object_id in _string_list(scope_record.get("target_object_ids")):
        edges.append({"source": "task_zone", "target": object_id, "relation": "contains_target"})
    for node in nodes:
        if not isinstance(node, Mapping):  # pragma: no cover - nodes are constructed as mappings above.
            continue
        if node.get("id") == "task_zone":
            continue
        ontology = node.get("ontology") if isinstance(node.get("ontology"), Mapping) else {}
        if ontology.get("hazard_relevant"):
            edges.append({"source": node.get("id"), "target": "task_zone", "relation": "hazard_near_task"})
        if ontology.get("route_relevant"):
            edges.append({"source": "task_zone", "target": node.get("id"), "relation": "route_context"})
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "nodes": nodes,
        "edges": edges,
    }


def _build_route_graph(
    *,
    descriptor: CaptureDescriptor,
    scene_graph: Mapping[str, Any],
) -> Dict[str, Any]:
    graph_nodes = [
        dict(node)
        for node in scene_graph.get("nodes", [])
        if isinstance(node, Mapping)
    ]
    route_context = []
    for node in graph_nodes:
        ontology = node.get("ontology") if isinstance(node.get("ontology"), Mapping) else {}
        if ontology.get("route_relevant"):
            route_context.append(node)
    route_context.sort(key=lambda item: _distance(item.get("center_m", [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]))
    nodes = [{"id": "entry", "type": "waypoint", "label": "capture_entry", "center_m": [0.0, 0.0, 0.0]}]
    for node in route_context:
        nodes.append(
            {
                "id": f"route:{node.get('id')}",
                "type": "route_context",
                "label": node.get("label"),
                "entity_type": node.get("category"),
                "center_m": node.get("center_m"),
                "width_m": _measure_width(node.get("extents_m", [0.0, 0.0, 0.0])),
            }
        )
    if any(node.get("id") == "task_zone" for node in graph_nodes if isinstance(node, Mapping)):
        task_zone = next(node for node in graph_nodes if node.get("id") == "task_zone")
        nodes.append({"id": "task_zone", "type": "waypoint", "label": "task_zone", "center_m": task_zone.get("center_m", [0.0, 0.0, 0.0])})
    handoff_nodes = [node for node in graph_nodes if node.get("category") == "handoff_point"]
    if handoff_nodes:
        for idx, node in enumerate(handoff_nodes):
            nodes.append(
                {
                    "id": f"handoff:{idx}",
                    "type": "handoff",
                    "label": node.get("label"),
                    "center_m": node.get("center_m", [0.0, 0.0, 0.0]),
                }
            )
    elif any(node.get("id") == "task_zone" for node in graph_nodes if isinstance(node, Mapping)):
        nodes.append({"id": "handoff", "type": "handoff", "label": "handoff", "center_m": [0.0, 0.0, 0.0]})

    edges: List[Dict[str, Any]] = []
    for idx in range(len(nodes) - 1):
        source = nodes[idx]
        target = nodes[idx + 1]
        source_center = source.get("center_m", [0.0, 0.0, 0.0])
        target_center = target.get("center_m", [0.0, 0.0, 0.0])
        edges.append(
            {
                "source": source["id"],
                "target": target["id"],
                "status": "measured" if idx > 0 else "candidate",
                "distance_m": round(_distance(source_center, target_center), 4),
                "constraining_width_m": round(
                    min(
                        [
                            _safe_float(source.get("width_m"), 99.0),
                            _safe_float(target.get("width_m"), 99.0),
                        ]
                    ),
                    4,
                ),
            }
        )
    if not any(node.get("id") == "task_zone" for node in graph_nodes if isinstance(node, Mapping)):
        edges = []
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "nodes": nodes,
        "edges": edges,
    }


def _build_geometry_evidence(
    *,
    descriptor: CaptureDescriptor,
    qa_report: Mapping[str, Any],
    object_index_entries: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    grouped = _group_objects_by_entity_type(object_index_entries)
    task_objects = grouped.get("tote", []) + grouped.get("pallet_zone", []) + grouped.get("rack", []) + grouped.get("handoff_point", [])
    route_objects = grouped.get("aisle", []) + grouped.get("threshold", []) + grouped.get("door_type", []) + grouped.get("forklift_lane", []) + grouped.get("traffic_zone", [])
    path_widths = [_measure_width(_entry_extents(entry)) for entry in route_objects if _measure_width(_entry_extents(entry)) > 0.0]
    target_distances = []
    if task_objects:
        zone_center = _zone_center({}, task_objects)
        target_distances = [_distance(zone_center, _entry_center(entry)) for entry in task_objects]
    all_centers = [_entry_center(entry) for entry in object_index_entries if isinstance(entry, Mapping)]
    if all_centers:
        mins = [min(center[idx] for center in all_centers) for idx in range(3)]
        maxs = [max(center[idx] for center in all_centers) for idx in range(3)]
        workcell_span = max(maxs[idx] - mins[idx] for idx in range(3))
    else:
        workcell_span = 0.0
    scaffolding_validation = (
        descriptor.scaffolding_validation
        if isinstance(descriptor.scaffolding_validation, Mapping)
        else {}
    )
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "capture_modality": descriptor.capture_modality,
        "evidence_tier": descriptor.evidence_tier,
        "metric_ready": _modality_supports_metric_automation(descriptor),
        "object_count": len(object_index_entries),
        "uncertainty_score": float(qa_report.get("uncertainty_score") or 0.0),
        "hidden_zone_score": float(qa_report.get("hidden_zone_score") or 0.0),
        "hidden_zone_bound": float(
            qa_report.get("hidden_zone_bound")
            or scaffolding_validation.get("hidden_zone_bound")
            or (0.25 if _modality_supports_metric_automation(descriptor) else 1.0)
        ),
        "scaffolding_used": list(descriptor.scaffolding_used),
        "calibration_assets": list(descriptor.calibration_assets),
        "measured_route_width_m": round(min(path_widths), 4) if path_widths else None,
        "target_reach_distance_m": round(max(target_distances), 4) if target_distances else None,
        "workcell_span_m": round(float(workcell_span), 4),
        "route_entity_counts": {key: len(value) for key, value in grouped.items() if value},
        "validated_pose_coverage": float(scaffolding_validation.get("validated_pose_coverage") or 0.0),
        "validated_scale_m": scaffolding_validation.get("validated_scale_m"),
    }


def _build_capability_checks(
    *,
    descriptor: CaptureDescriptor,
    geometry_evidence: Mapping[str, Any],
    route_graph: Mapping[str, Any],
    scope_record: Mapping[str, Any],
) -> Dict[str, Any]:
    metric_ready = bool(geometry_evidence.get("metric_ready"))
    target_count = len(_string_list(scope_record.get("target_object_ids")))
    route_edges = route_graph.get("edges") if isinstance(route_graph.get("edges"), list) else []
    measured_route_width = geometry_evidence.get("measured_route_width_m")
    target_reach_distance = geometry_evidence.get("target_reach_distance_m")
    workcell_span = _safe_float(geometry_evidence.get("workcell_span_m"), 0.0)
    hidden_zone_bound = _safe_float(geometry_evidence.get("hidden_zone_bound"), 1.0)
    uncertainty_score = _safe_float(geometry_evidence.get("uncertainty_score"), 1.0)

    def _status_for_threshold(value: Any, *, maximum: float | None = None, minimum: float | None = None) -> str:
        if value is None or not metric_ready:
            return "needs_more_evidence"
        numeric = _safe_float(value, 0.0)
        if maximum is not None and numeric > maximum:
            return "blocked"
        if minimum is not None and numeric < minimum:
            return "blocked"
        return "pass"

    checks = [
        {
            "id": "clearance_precheck",
            "status": _status_for_threshold(
                measured_route_width,
                minimum=_GENERIC_CAPABILITY_ENVELOPE["minimum_path_width_m"],
            )
            if route_edges
            else "needs_more_evidence",
            "detail": (
                f"Measured minimum route width is {measured_route_width} m."
                if measured_route_width is not None
                else "Route width is not yet measured."
            ),
        },
        {
            "id": "reach_envelope_precheck",
            "status": _status_for_threshold(
                target_reach_distance,
                maximum=_GENERIC_CAPABILITY_ENVELOPE["maximum_target_reach_distance_m"],
            )
            if target_count
            else "needs_more_evidence",
            "detail": (
                f"Maximum inferred target reach distance is {target_reach_distance} m."
                if target_reach_distance is not None
                else "Target geometry is insufficient for reach checks."
            ),
        },
        {
            "id": "workcell_occupancy_analysis",
            "status": _status_for_threshold(
                workcell_span,
                maximum=_GENERIC_CAPABILITY_ENVELOPE["maximum_workcell_span_m"],
            )
            if target_count
            else "needs_more_evidence",
            "detail": f"Estimated workcell span is {round(workcell_span, 4)} m.",
        },
        {
            "id": "choke_point_detection",
            "status": (
                "blocked"
                if measured_route_width is not None and measured_route_width < _GENERIC_CAPABILITY_ENVELOPE["preferred_path_width_m"]
                else "pass"
            )
            if metric_ready and route_edges
            else "needs_more_evidence",
            "detail": (
                f"Preferred route width threshold is {_GENERIC_CAPABILITY_ENVELOPE['preferred_path_width_m']} m; measured {measured_route_width} m."
                if measured_route_width is not None
                else "Route graph is not yet strong enough for choke-point detection."
            ),
        },
        {
            "id": "occlusion_analysis",
            "status": _status_for_threshold(
                hidden_zone_bound,
                maximum=_GENERIC_CAPABILITY_ENVELOPE["maximum_hidden_zone_bound"],
            ),
            "detail": f"Hidden-zone bound is {round(hidden_zone_bound, 4)}.",
        },
        {
            "id": "candidate_charger_placement",
            "status": "pass" if metric_ready else "needs_more_evidence",
            "detail": "Scene geometry is available for charger placement hypotheses." if metric_ready else "Candidate charger placement requires stronger metric context.",
        },
        {
            "id": "route_viability_hypotheses",
            "status": (
                "pass"
                if metric_ready and route_edges and uncertainty_score <= _GENERIC_CAPABILITY_ENVELOPE["maximum_uncertainty_score"]
                else "blocked"
            )
            if metric_ready and route_edges
            else "needs_more_evidence",
            "detail": f"Uncertainty score is {round(uncertainty_score, 4)}.",
        },
        {
            "id": "scenario_batches_in_simulation",
            "status": "pass" if metric_ready and len(descriptor.requested_lanes) > 1 else "needs_more_evidence",
            "detail": "Capture is eligible for scene-memory derivation and explicit conditioning batches." if metric_ready and len(descriptor.requested_lanes) > 1 else "Scene-memory derivation requires stronger metric or conditioning outputs.",
        },
        {
            "id": "coexistence_fit",
            "status": (
                "blocked"
                if hidden_zone_bound > _GENERIC_CAPABILITY_ENVELOPE["maximum_hidden_zone_bound"]
                else "pass"
            )
            if metric_ready
            else "needs_more_evidence",
            "detail": "Shared-space coexistence is gated by hidden-zone coverage and traffic visibility.",
        },
    ]
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "capability_envelope": dict(_GENERIC_CAPABILITY_ENVELOPE),
        "checks": checks,
    }


def _build_blocker_register(
    *,
    descriptor: CaptureDescriptor,
    qualification_record: Mapping[str, Any],
    capability_checks: Mapping[str, Any],
    geometry_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for risk in qualification_record.get("risks", []):
        if not isinstance(risk, Mapping):
            continue
        entries.append(
            {
                "id": str(risk.get("id") or f"risk_{len(entries)}"),
                "severity": str(risk.get("severity") or "medium"),
                "category": str(risk.get("category") or "general"),
                "detail": str(risk.get("detail") or ""),
                "evidence": {
                    "capture_modality": descriptor.capture_modality,
                    "evidence_tier": descriptor.evidence_tier,
                    "uncertainty_score": float(geometry_evidence.get("uncertainty_score") or 0.0),
                },
            }
        )
    for check in capability_checks.get("checks", []):
        if not isinstance(check, Mapping) or check.get("status") == "pass":
            continue
        entries.append(
            {
                "id": str(check.get("id") or f"check_{len(entries)}"),
                "severity": "high" if check.get("status") == "blocked" else "medium",
                "category": "automation_gap",
                "detail": str(check.get("detail") or ""),
                "evidence": {
                    "source_check": str(check.get("id") or ""),
                    "status": str(check.get("status") or "needs_more_evidence"),
                },
            }
        )
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "entries": entries,
    }


def _build_readiness_decision(
    *,
    descriptor: CaptureDescriptor,
    qualification_record: Mapping[str, Any],
    blocker_register: Mapping[str, Any],
    capability_checks: Mapping[str, Any],
    geometry_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers = blocker_register.get("entries", []) if isinstance(blocker_register.get("entries"), list) else []
    unresolved_high_blockers = [
        entry for entry in blockers
        if isinstance(entry, Mapping) and str(entry.get("severity") or "").lower() == "high"
    ]
    uncertainty_score = float(geometry_evidence.get("uncertainty_score") or 0.0)
    hidden_zone_bound = float(geometry_evidence.get("hidden_zone_bound") or 1.0)
    human_review_required = True
    remediation = [entry.get("detail") for entry in blockers[:5] if isinstance(entry, Mapping)]
    status = str(qualification_record.get("readiness_state") or "not_ready_yet")
    if unresolved_high_blockers or uncertainty_score > _GENERIC_CAPABILITY_ENVELOPE["maximum_uncertainty_score"]:
        status = "not_ready_yet"
    elif status == "ready" and hidden_zone_bound > _GENERIC_CAPABILITY_ENVELOPE["maximum_hidden_zone_bound"]:
        status = "risky"
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "status": status,
        "confidence": qualification_record.get("confidence"),
        "capture_modality": descriptor.capture_modality,
        "evidence_tier": descriptor.evidence_tier,
        "human_review_required": human_review_required,
        "human_review_scope": [
            "workflow boundary confirmation",
            "safety and non-routine review",
            "hidden conditions and restricted zones",
            "final readiness signoff",
        ],
        "evidence_gaps": [item.get("detail") for item in blockers if isinstance(item, Mapping)],
        "blockers": blockers,
        "capability_checks": capability_checks.get("checks", []),
        "remediation": remediation,
    }


def _render_readiness_report(
    *,
    descriptor: CaptureDescriptor,
    readiness_decision: Mapping[str, Any],
    blocker_register: Mapping[str, Any],
    human_actions_required: Optional[Mapping[str, Any]] = None,
    task_hypothesis_report: Optional[Mapping[str, Any]] = None,
) -> str:
    lines = [
        f"# Readiness Report: {descriptor.scene_id}/{descriptor.capture_id}",
        "",
        f"- Status: `{readiness_decision.get('status', 'not_ready_yet')}`",
        f"- Confidence: `{readiness_decision.get('confidence', 0.0)}`",
        f"- Capture modality: `{descriptor.capture_modality}`",
        f"- Evidence tier: `{descriptor.evidence_tier}`",
        f"- Human review required: `{bool(readiness_decision.get('human_review_required'))}`",
        f"- Task hypothesis status: `{str(task_hypothesis_report.get('task_hypothesis_status') or 'not_available') if isinstance(task_hypothesis_report, Mapping) else 'not_available'}`",
        "",
        "## Review Scope",
    ]
    review_scope = readiness_decision.get("human_review_scope") if isinstance(readiness_decision.get("human_review_scope"), list) else []
    if not review_scope:
        lines.append("- Final human signoff remains required.")
    else:
        for item in review_scope:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
        "## Blockers",
        ]
    )
    blockers = blocker_register.get("entries", []) if isinstance(blocker_register.get("entries"), list) else []
    if not blockers:
        lines.append("- None recorded")
    else:
        for blocker in blockers[:10]:
            if not isinstance(blocker, Mapping):
                continue
            lines.append(
                f"- [{blocker.get('severity', 'medium')}] {blocker.get('detail', '')}"
            )
    lines.extend(["", "## Human Actions Required"])
    actions = (
        human_actions_required.get("actions", [])
        if isinstance(human_actions_required, Mapping)
        else []
    )
    if not actions:
        lines.append("- Final human signoff remains required.")
    else:
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            lines.append(f"- {action.get('action', '')}")
    return "\n".join(lines) + "\n"


def _build_human_actions_required(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    readiness_decision: Mapping[str, Any],
    blocker_register: Mapping[str, Any],
    geometry_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = [
        {
            "action": "Confirm workflow boundary and success criteria.",
            "required": True,
            "owner": "human_reviewer",
            "reason": "Workflow fit remains a human accountability boundary.",
        },
        {
            "action": "Confirm the in-scope zone and accountable site owner.",
            "required": True,
            "owner": "human_reviewer",
            "reason": "Task ownership and zone boundaries are required for deployment routing.",
        },
        {
            "action": "Review non-routine modes and safety/EHS constraints.",
            "required": True,
            "owner": "human_reviewer",
            "reason": "Safety, recovery, and non-routine modes are not auto-approved.",
        },
        {
            "action": "Confirm hidden or restricted areas were adequately captured.",
            "required": True,
            "owner": "human_reviewer",
            "reason": "Privacy restrictions and hidden zones can hide decision-critical evidence.",
        },
        {
            "action": "Approve recapture when evidence is incomplete.",
            "required": str(scorecard.get("completeness_status") or "need_more_evidence") != "sufficient",
            "owner": "human_reviewer",
            "reason": "Incomplete capture evidence must be accepted or recaptured by a human.",
        },
        {
            "action": "Make the final readiness signoff.",
            "required": True,
            "owner": "human_reviewer",
            "reason": f"Pipeline status is {readiness_decision.get('status', 'not_ready_yet')}.",
        },
        {
            "action": "Choose the OEM, integrator, or target robot platform for downstream evaluation.",
            "required": True,
            "owner": "human_reviewer",
            "reason": "The repo does not auto-select downstream deployment targets.",
        },
    ]
    blocker_details = [
        str(item.get("detail") or "").strip()
        for item in blocker_register.get("entries", [])
        if isinstance(item, Mapping)
    ]
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "readiness_state": readiness_decision.get("status"),
        "completeness_status": scorecard.get("completeness_status"),
        "capture_modality": descriptor.capture_modality,
        "evidence_tier": descriptor.evidence_tier,
        "hidden_zone_bound": geometry_evidence.get("hidden_zone_bound"),
        "risk_count": len(qualification_record.get("risks", []))
        if isinstance(qualification_record.get("risks"), list)
        else 0,
        "blocker_details": blocker_details[:10],
        "actions": actions,
    }


def _llm_weakness_payload(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    readiness_decision: Mapping[str, Any],
    blocker_register: Mapping[str, Any],
    human_actions_required: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "capture_modality": descriptor.capture_modality,
        "evidence_tier": descriptor.evidence_tier,
        "task_statement": descriptor.metadata.get("task_statement"),
        "workflow_context": descriptor.metadata.get("workflow_context"),
        "task_zone": descriptor.metadata.get("task_zone"),
        "completeness_status": scorecard.get("completeness_status"),
        "scope_status": scope_record.get("scope_status"),
        "readiness_decision": dict(readiness_decision),
        "blocker_register": dict(blocker_register),
        "human_actions_required": dict(human_actions_required),
    }


def _llm_recapture_payload(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    blocker_register: Mapping[str, Any],
    human_actions_required: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "task_statement": descriptor.metadata.get("task_statement"),
        "workflow_context": descriptor.metadata.get("workflow_context"),
        "task_zone": descriptor.metadata.get("task_zone"),
        "capture_modality": descriptor.capture_modality,
        "scorecard_follow_ups": list(scorecard.get("follow_ups", [])) if isinstance(scorecard.get("follow_ups"), list) else [],
        "blocker_details": list(human_actions_required.get("blocker_details", [])) if isinstance(human_actions_required.get("blocker_details"), list) else [],
        "scope_blockers": list(scope_record.get("blockers", [])) if isinstance(scope_record.get("blockers"), list) else [],
        "blocker_register": dict(blocker_register),
    }


def _write_failure(
    *,
    pipeline_dir: Path,
    descriptor_uri: str,
    stage: str,
    error: Exception,
    gates: List[QualificationGate],
) -> None:
    payload = {
        "schema_version": "v1",
        "lane": "qualification",
        "status": "failed",
        "stage": stage,
        "descriptor_uri": descriptor_uri,
        "error": str(error),
        "failed_at": utc_now_iso(),
        "gates": [gate.to_dict() for gate in gates],
    }
    write_json(pipeline_dir / ".qualification_pipeline_failed.json", payload)
    write_json(pipeline_dir / ".swap_pipeline_failed.json", payload)


def run_qualification_pipeline(
    *,
    descriptor_gcs_uri: str,
    config: Any,
    requested_lanes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    stage = "intake"
    gates: List[QualificationGate] = []
    scene_id = "unknown_scene"
    capture_id = "unknown_capture"
    pipeline_prefix = "_pipeline_failures"
    pipeline_dir = config.gcs_root / pipeline_prefix
    ensure_dir(pipeline_dir)

    try:
        parsed_uri = parse_gs_uri(descriptor_gcs_uri)
        bucket = parsed_uri.bucket
        descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, config.gcs_root)
        storage_root = infer_storage_root_from_scene_path(descriptor_path)
        capture_root = descriptor_path.parent
        descriptor = CaptureDescriptor.from_file(descriptor_path)
        scene_id = descriptor.scene_id
        capture_id = descriptor.capture_id
        pipeline_prefix = to_pipeline_prefix(scene_id, capture_id)
        pipeline_dir = storage_root / pipeline_prefix
        ensure_dir(pipeline_dir)
        downstream_requested_lanes = _requested_downstream_lanes(
            descriptor=descriptor,
            requested_lanes=requested_lanes,
        )
        rights_required_use_classes = _rights_review_required_use_classes(
            descriptor=descriptor,
            requested_lanes=downstream_requested_lanes,
        )

        qa_report_uri = descriptor.qa_report_uri or _default_qa_report_uri(descriptor_gcs_uri)
        qa_report_path = resolve_gs_uri_to_path(qa_report_uri, storage_root)
        qa_report = _try_read_json(qa_report_path) or {"schema_version": "v1", "status": "missing"}

        manifest_uri = None
        manifest: Optional[IOSManifest] = None
        manifest_path: Optional[Path] = None
        object_index_uri: Optional[str] = None
        object_index_path: Optional[Path] = None
        object_index_entries: List[Mapping[str, Any]] = []
        grounding_payload: Optional[Dict[str, Any]] = None
        raw_task_hypothesis: Optional[Dict[str, Any]] = None
        object_index_runtime_blockers: List[str] = []

        try:
            manifest = load_raw_manifest(descriptor.raw_prefix_uri, gcs_root=storage_root)
            manifest_uri = f"{descriptor.raw_prefix_uri.rstrip('/')}/manifest.json"
            manifest_path = resolve_gs_uri_to_path(manifest_uri, storage_root)
            try:
                stage_result = ensure_object_index_stage(
                    capture_root=descriptor_path.parent,
                    force_rebuild=parse_bool(os.getenv("OBJECT_INDEX_FORCE_REBUILD"), default=False),
                )
            except Exception as exc:  # noqa: BLE001 - qualification must degrade but record the blocker
                _append_unique(
                    object_index_runtime_blockers,
                    _object_index_exception_blocker("ensure_object_index_stage", exc),
                )
                stage_result = {}
            if isinstance(stage_result.get("grounding_payload"), Mapping):
                grounding_payload = dict(stage_result["grounding_payload"])
            try:
                stage_current_usable_count = int(stage_result.get("current_usable_object_count"))
            except (TypeError, ValueError):
                stage_current_usable_count = None
            if stage_current_usable_count is not None and stage_current_usable_count <= 0:
                # A blocked immutable rerun explicitly clears stale descriptor/raw fallbacks.
                object_index_uri = None
            else:
                object_index_uri = (
                    str(stage_result.get("object_index_uri") or "").strip()
                    or str(descriptor.object_index_uri or "").strip()
                    or resolve_object_index_uri(descriptor.raw_prefix_uri, manifest)
                )
            if object_index_uri:
                object_index_path = resolve_gs_uri_to_path(object_index_uri, storage_root)
                object_index_entries = load_object_index(object_index_uri, gcs_root=storage_root)
            _extend_unique(
                object_index_runtime_blockers,
                _object_index_runtime_blockers(descriptor_path.parent),
            )
        except Exception as exc:  # noqa: BLE001 - keep qualification running, but do not hide the crash
            manifest = None
            object_index_uri = None
            object_index_entries = []
            grounding_payload = None
            object_index_runtime_blockers = _object_index_runtime_blockers(descriptor_path.parent)
            _append_unique(
                object_index_runtime_blockers,
                _object_index_exception_blocker("object_index_load", exc),
            )

        raw_task_hypothesis = _try_read_optional_json_uri(
            descriptor.task_hypothesis_uri,
            storage_root,
        )

        stage = "runtime_preflight"
        if getattr(config, "runtime_preflight_enabled", True):
            runtime_preflight_report = _build_runtime_preflight_report(
                descriptor_path=descriptor_path,
                qa_report_path=qa_report_path,
                manifest_path=manifest_path,
                object_index_path=object_index_path,
                gcs_root=storage_root,
            )
        else:
            runtime_preflight_report = {
                "schema_version": "v1",
                "lane": "qualification",
                "status": "skipped",
                "generated_at": utc_now_iso(),
                "detail": "runtime preflight disabled by configuration",
            }
        write_json(pipeline_dir / "runtime_preflight_report.json", runtime_preflight_report)
        gates.append(
            QualificationGate(
                "runtime_preflight_gate",
                runtime_preflight_report.get("status") in {"passed", "skipped", "degraded"},
                f"status={runtime_preflight_report.get('status')}",
            )
        )

        stage = "presentation_demo_preflight"
        presentation_demo_preflight_report = _build_presentation_demo_preflight_report()
        write_json(pipeline_dir / "presentation_demo_preflight_report.json", presentation_demo_preflight_report)

        stage = "capture_package_manifest"
        capture_package_manifest = _build_capture_package_manifest(
            descriptor=descriptor,
            descriptor_uri=descriptor_gcs_uri,
            qa_report_uri=qa_report_uri,
            manifest_uri=manifest_uri,
            object_index_uri=object_index_uri,
            task_hypothesis_uri=descriptor.task_hypothesis_uri,
            storage_root=storage_root,
            object_index_entries=object_index_entries,
        )
        write_json(pipeline_dir / "capture_package_manifest.json", capture_package_manifest)

        stage = "task_targets"
        if manifest is not None and object_index_uri:
            task_targets_payload = infer_task_targets(
                descriptor=descriptor,
                manifest=manifest,
                object_index_entries=object_index_entries,
                object_index_uri=object_index_uri,
                storage_root=storage_root,
                grounding_payload=grounding_payload,
                max_targets=max(1, int(getattr(config, "task_target_max_objects", 24) or 24)),
            )
        else:
            task_targets_payload = _disabled_task_targets(
                descriptor.scene_id,
                descriptor.capture_id,
                "missing_manifest_or_object_index",
            )
        task_targets_with_index = dict(task_targets_payload)
        task_targets_with_index["object_index_entries"] = [dict(item) for item in object_index_entries]
        write_task_targets(pipeline_dir / "task_targets.json", task_targets_with_index)

        stage = "task_hypothesis_verification"
        task_hypothesis_report = _build_task_hypothesis_report(
            descriptor=descriptor,
            raw_task_hypothesis=raw_task_hypothesis,
            object_index_entries=object_index_entries,
            task_targets_payload=task_targets_payload,
        )
        normalized_task_hypothesis = (
            dict(task_hypothesis_report.get("normalized_task_hypothesis"))
            if isinstance(task_hypothesis_report.get("normalized_task_hypothesis"), Mapping)
            else {}
        )
        effective_metadata = _effective_task_metadata(
            descriptor,
            task_hypothesis_report=task_hypothesis_report,
        )
        write_json(pipeline_dir / "task_hypothesis_report.json", task_hypothesis_report)
        write_json(pipeline_dir / "normalized_task_hypothesis.json", normalized_task_hypothesis)
        gates.append(
            QualificationGate(
                "task_hypothesis_gate",
                str(task_hypothesis_report.get("task_hypothesis_status") or "accepted")
                in {"accepted", "accepted_with_warnings"},
                f"task_hypothesis_status={task_hypothesis_report.get('task_hypothesis_status')}",
            )
        )

        stage = "intake"
        site_intake = {
            "schema_version": "v1",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "generated_at": utc_now_iso(),
            "descriptor": descriptor.to_dict(),
            "descriptor_uri": descriptor_gcs_uri,
            "qa_report_uri": qa_report_uri,
            "task_hypothesis_report_uri": f"gs://{bucket}/{pipeline_prefix}/task_hypothesis_report.json",
            "normalized_task_hypothesis_uri": f"gs://{bucket}/{pipeline_prefix}/normalized_task_hypothesis.json",
            "site_identity": {
                "scene_id": descriptor.scene_id,
                "capture_id": descriptor.capture_id,
                "environment_type_hint": descriptor.environment_type_hint or "unknown",
                "capture_modality": descriptor.capture_modality,
            },
            "task_context": {
                "buyer_type": effective_metadata.get("buyer_type"),
                "task_statement": effective_metadata.get("task_statement"),
                "workflow_context": effective_metadata.get("workflow_context"),
                "operating_hours": effective_metadata.get("operating_hours"),
                "workflow_decomposition": _string_list(effective_metadata.get("workflow_decomposition"))
                or _string_list(effective_metadata.get("workflow_context")),
                "task_zone": effective_metadata.get("task_zone") if isinstance(effective_metadata.get("task_zone"), Mapping) else {},
                "success_criteria": _string_list(effective_metadata.get("success_criteria")),
                "owner": effective_metadata.get("owner"),
                "adjacent_systems": _string_list(effective_metadata.get("adjacent_systems")),
                "non_routine_modes": _string_list(effective_metadata.get("non_routine_modes")),
                "people_traffic_notes": _string_list(effective_metadata.get("people_traffic_notes")),
                "task_hypothesis_status": effective_metadata.get("task_hypothesis_status"),
                "task_hypothesis_confidence": effective_metadata.get("task_hypothesis_confidence"),
                "task_hypothesis_source": effective_metadata.get("task_hypothesis_source"),
                "task_hypothesis_warnings": _string_list(effective_metadata.get("task_hypothesis_warnings")),
            },
            "constraints": {
                "privacy_restrictions": effective_metadata.get("privacy_restrictions"),
                "security_restrictions": effective_metadata.get("security_restrictions"),
                "known_blockers": effective_metadata.get("known_blockers") if isinstance(effective_metadata, Mapping) else [],
                "safety_concerns": effective_metadata.get("safety_concerns") if isinstance(effective_metadata, Mapping) else [],
                "capture_restrictions": _string_list(effective_metadata.get("capture_restrictions")),
            },
            "capture_plan": {
                "scaffolding_used": list(descriptor.scaffolding_used),
                "coverage_plan": list(descriptor.coverage_plan),
                "calibration_assets": list(descriptor.calibration_assets),
                "uncertainty_priors": dict(descriptor.uncertainty_priors),
            },
            "capture_rights": _capture_rights(
                effective_metadata if isinstance(effective_metadata, Mapping) else {}
            ),
            "task_hypothesis": dict(raw_task_hypothesis) if isinstance(raw_task_hypothesis, Mapping) else {},
        }
        write_json(pipeline_dir / "site_intake.json", site_intake)
        gates.append(QualificationGate("intake_gate", True, "descriptor parsed and intake record written"))

        stage = "completeness"
        scorecard = _build_completeness_scorecard(
            descriptor=descriptor,
            qa_report=qa_report,
            manifest=manifest,
            object_index_uri=object_index_uri,
            object_index_entries=object_index_entries,
            metadata_override=effective_metadata,
            task_hypothesis_report=task_hypothesis_report,
        )
        write_json(pipeline_dir / "capture_qa_scorecard.json", scorecard)
        gates.append(
            QualificationGate(
                "completeness_gate",
                str(scorecard.get("completeness_status")) == "sufficient",
                f"completeness_status={scorecard.get('completeness_status')}",
            )
        )

        stage = "scoping"
        scope_record = _build_task_scope_record(
            descriptor=descriptor,
            task_targets_payload=task_targets_payload,
            completeness_status=str(scorecard.get("completeness_status") or "need_more_evidence"),
            metadata_override=effective_metadata,
            object_index_runtime_blockers=object_index_runtime_blockers,
        )
        write_json(pipeline_dir / "task_scope_record.json", scope_record)
        gates.append(
            QualificationGate(
                "scoping_gate",
                str(scope_record.get("scope_status")) == "scoped",
                f"scope_status={scope_record.get('scope_status')}",
            )
        )

        stage = "qualification"
        qualification_record = _build_qualification_record(
            descriptor=descriptor,
            scorecard=scorecard,
            scope_record=scope_record,
            object_index_entries=object_index_entries,
            object_index_runtime_blockers=object_index_runtime_blockers,
        )
        stage = "gemini_capture_review"
        raw_video_path = _resolve_optional_uri_to_path(descriptor.raw_video_uri, storage_root)
        capture_fidelity_review = infer_capture_fidelity_review(
            capture_root=descriptor_path.parent,
            raw_video_path=raw_video_path,
            keyframe_path=_resolve_optional_uri_to_path(descriptor.keyframe_uri, storage_root),
            descriptor=descriptor.to_dict(),
            qa_report=qa_report,
            task_hypothesis_report=task_hypothesis_report,
            capture_context={
                "capture_rights": _capture_rights(effective_metadata if isinstance(effective_metadata, Mapping) else {}),
                "capture_orientation": descriptor.capture_orientation,
                "requested_outputs": list(descriptor.requested_outputs),
                "quoted_payout_cents": descriptor.quoted_payout_cents,
                "site_submission_id": descriptor.site_submission_id,
                "buyer_request_id": descriptor.buyer_request_id,
                "capture_job_id": descriptor.capture_job_id,
                "metadata": dict(effective_metadata) if isinstance(effective_metadata, Mapping) else {},
            },
            timeout_sec=int(getattr(config, "gemini_timeout_seconds", 45) or 45),
        )
        write_json(pipeline_dir / "gemini_capture_fidelity_review.json", capture_fidelity_review)
        gates.append(
            QualificationGate(
                "gemini_capture_review_gate",
                str(capture_fidelity_review.get("status") or "").strip().lower() == "succeeded",
                f"status={capture_fidelity_review.get('status')}",
            )
        )
        qualification_record = _apply_capture_fidelity_to_qualification(
            qualification_record=qualification_record,
            capture_fidelity_review=capture_fidelity_review,
            metadata=effective_metadata if isinstance(effective_metadata, Mapping) else {},
        )
        stage = "privacy_postprocess"
        privacy_processing = run_privacy_postprocess(
            bucket=bucket,
            scene_id=descriptor.scene_id,
            capture_id=descriptor.capture_id,
            capture_root=capture_root,
            pipeline_dir=pipeline_dir,
            raw_video_path=raw_video_path,
        )
        # PIPE-03: a "delivery run" builds buyer/reviewer-facing downstream artifacts
        # (scene_memory / evaluation_prep lanes). For those, privacy post-processing
        # must actually have run and cleared — ``not_run`` is NON-passing. Non-delivery
        # / local flows keep passing on ``not_run``.
        privacy_delivery_run = bool(downstream_requested_lanes) or production_launch_mode()
        gates.append(
            _privacy_postprocess_gate(
                privacy_status=str(privacy_processing.get("status") or ""),
                delivery_run=privacy_delivery_run,
            )
        )
        descriptor_payload = descriptor.to_dict()
        descriptor_payload["privacy_processed_video_uri"] = privacy_processing.get("privacy_processed_video_uri")
        descriptor_payload["world_model_video_uri"] = privacy_processing.get("world_model_video_uri")
        descriptor_payload["privacy_status"] = privacy_processing.get("status")
        descriptor_payload["privacy_mode"] = privacy_processing.get("mode")
        descriptor_payload["privacy_manifest_uri"] = privacy_processing.get("privacy_manifest_uri")
        descriptor_payload["depth_conditioning"] = (
            dict(privacy_processing.get("depth_conditioning"))
            if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
            else dict(descriptor.depth_conditioning)
            if isinstance(descriptor.depth_conditioning, Mapping)
            else {}
        )
        preview_requested_for_worldlabs = any(
            str(value or "").strip().lower() in {"preview_simulation", "preview"}
            for value in descriptor.requested_outputs
        )
        # PIPE-04: the WorldLabs preview input video is a derived, reviewer-facing
        # transformation of the capture. It must not be generated unless the capture is
        # rights-cleared for derived scene generation — mirroring scene-memory readiness.
        worldlabs_derived_rights_allowed = _worldlabs_derived_rights_allowed(
            metadata=descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
        )
        privacy_descriptor = CaptureDescriptor.from_dict(descriptor_payload)
        if preview_requested_for_worldlabs and worldlabs_derived_rights_allowed:
            worldlabs_input = _prepare_worldlabs_input_video(
                descriptor=privacy_descriptor,
                privacy_processing=privacy_processing,
                storage_root=storage_root,
                pipeline_dir=pipeline_dir,
                bucket=bucket,
            )
        elif preview_requested_for_worldlabs and not worldlabs_derived_rights_allowed:
            worldlabs_input = {
                "status": "blocked",
                "reason": "rights_not_cleared_for_derived_scene_generation",
                "manifest_uri": None,
                "output_video_uri": None,
            }
        else:
            worldlabs_input = {
                "status": "not_requested",
                "manifest_uri": None,
                "output_video_uri": None,
            }
        metadata_payload = dict(descriptor_payload.get("metadata") or {})
        metadata_payload["privacy_processing"] = {
            "status": privacy_processing.get("status"),
            "mode": privacy_processing.get("mode"),
            "fallback_used": bool(privacy_processing.get("fallback_used")),
            "raw_retained": bool(privacy_processing.get("raw_retained")),
            "fail_closed": bool(privacy_processing.get("fail_closed")),
            "people_detected": int(privacy_processing.get("people_detected") or 0),
            "people_removed": int(privacy_processing.get("people_removed") or 0),
            "face_anonymized_segments": _string_list(privacy_processing.get("face_anonymized_segments")),
            "privacy_manifest_uri": privacy_processing.get("privacy_manifest_uri"),
            "privacy_verification_report_uri": privacy_processing.get("privacy_verification_report_uri"),
            "depth_source": privacy_processing.get("depth_source"),
            "depth_conditioning": (
                dict(privacy_processing.get("depth_conditioning"))
                if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
                else {}
            ),
        }
        scene_memory_capture = (
            dict(metadata_payload.get("scene_memory_capture"))
            if isinstance(metadata_payload.get("scene_memory_capture"), Mapping)
            else {}
        )
        sensor_availability = (
            dict(scene_memory_capture.get("sensor_availability"))
            if isinstance(scene_memory_capture.get("sensor_availability"), Mapping)
            else {}
        )
        sensor_availability["depth_conditioning"] = bool(privacy_processing.get("depth_conditioning"))
        scene_memory_capture["sensor_availability"] = sensor_availability
        metadata_payload["scene_memory_capture"] = scene_memory_capture
        metadata_payload["worldlabs_input_video_uri"] = worldlabs_input.get("output_video_uri")
        metadata_payload["worldlabs_input_manifest_uri"] = worldlabs_input.get("manifest_uri")
        metadata_payload["worldlabs_input_audit_uri"] = worldlabs_input.get("audit_uri")
        metadata_payload["worldlabs_input_audit"] = (
            dict(worldlabs_input.get("audit_payload"))
            if isinstance(worldlabs_input.get("audit_payload"), Mapping)
            else {}
        )
        metadata_payload["worldlabs_input_status"] = worldlabs_input.get("status")
        metadata_payload["worldlabs_input_labeling"] = (
            dict(worldlabs_input.get("input_labeling"))
            if isinstance(worldlabs_input.get("input_labeling"), Mapping)
            else {}
        )
        descriptor_payload["metadata"] = metadata_payload
        write_json(descriptor_path, descriptor_payload)
        descriptor = CaptureDescriptor.from_dict(descriptor_payload)
        if _should_run_default_geometry_stage(descriptor):
            build_geometry_stage_contract(capture_root)
            descriptor = CaptureDescriptor.from_file(descriptor_path)
        qualification_brief = _build_qualification_brief(
            descriptor=descriptor,
            scorecard=scorecard,
            scope_record=scope_record,
            qualification_record=qualification_record,
        )
        scene_graph = _build_scene_graph(
            descriptor=descriptor,
            scope_record=scope_record,
            object_index_entries=object_index_entries,
        )
        route_graph = _build_route_graph(
            descriptor=descriptor,
            scene_graph=scene_graph,
        )
        geometry_evidence = _build_geometry_evidence(
            descriptor=descriptor,
            qa_report=qa_report,
            object_index_entries=object_index_entries,
        )
        capability_checks = _build_capability_checks(
            descriptor=descriptor,
            geometry_evidence=geometry_evidence,
            route_graph=route_graph,
            scope_record=scope_record,
        )
        blocker_register = _build_blocker_register(
            descriptor=descriptor,
            qualification_record=qualification_record,
            capability_checks=capability_checks,
            geometry_evidence=geometry_evidence,
        )
        readiness_decision = _build_readiness_decision(
            descriptor=descriptor,
            qualification_record=qualification_record,
            blocker_register=blocker_register,
            capability_checks=capability_checks,
            geometry_evidence=geometry_evidence,
        )
        human_actions_required = _build_human_actions_required(
            descriptor=descriptor,
            scorecard=scorecard,
            qualification_record=qualification_record,
            readiness_decision=readiness_decision,
            blocker_register=blocker_register,
            geometry_evidence=geometry_evidence,
        )
        enrichment_runner = build_capture_enrichment_runner(repo_root=Path(__file__).resolve().parents[2])
        weakness_summary = (
            enrichment_runner(
                "qualification_weakness_summarizer",
                _llm_weakness_payload(
                    descriptor=descriptor,
                    scorecard=scorecard,
                    scope_record=scope_record,
                    readiness_decision=readiness_decision,
                    blocker_register=blocker_register,
                    human_actions_required=human_actions_required,
                ),
            )
            if enrichment_runner is not None
            else None
        )
        recapture_instructions = (
            enrichment_runner(
                "recapture_instruction_writer",
                _llm_recapture_payload(
                    descriptor=descriptor,
                    scorecard=scorecard,
                    scope_record=scope_record,
                    blocker_register=blocker_register,
                    human_actions_required=human_actions_required,
                ),
            )
            if enrichment_runner is not None
            else None
        )
        if isinstance(recapture_instructions, Mapping):
            human_actions_required["llm_recapture_instructions"] = list(
                recapture_instructions.get("instructions", [])
            ) if isinstance(recapture_instructions.get("instructions"), list) else []
        qualification_record["readiness_state"] = readiness_decision.get("status")
        geometry_artifacts = _geometry_artifacts(
            pipeline_dir=pipeline_dir,
            bucket=bucket,
            pipeline_prefix=pipeline_prefix,
        )
        world_model_fit_summary = _build_world_model_fit_summary(
            descriptor=descriptor,
            scorecard=scorecard,
            qualification_record=qualification_record,
            capture_fidelity_review=capture_fidelity_review,
            privacy_processing=privacy_processing,
            metadata=effective_metadata if isinstance(effective_metadata, Mapping) else {},
            geometry_summary=geometry_artifacts.get("summary") if isinstance(geometry_artifacts.get("summary"), Mapping) else {},
        )
        capturer_payout_recommendation = _build_capturer_payout_recommendation(
            descriptor=descriptor,
            capture_fidelity_review=capture_fidelity_review,
            metadata=effective_metadata if isinstance(effective_metadata, Mapping) else {},
        )
        provenance_summary = _build_provenance_summary(
            descriptor_uri=descriptor_gcs_uri,
            qa_report_uri=qa_report_uri,
            pipeline_prefix=pipeline_prefix,
            bucket=bucket,
            capture_fidelity_review=capture_fidelity_review,
            metadata=effective_metadata if isinstance(effective_metadata, Mapping) else {},
        )
        opportunity_handoff = _build_opportunity_handoff(
            descriptor=descriptor,
            scorecard=scorecard,
            scope_record=scope_record,
            qualification_record=qualification_record,
            brief=qualification_brief,
            config=config,
            pipeline_dir=pipeline_dir,
            metadata_override=effective_metadata,
        )
        opportunity_handoff["evidence_bundle"] = {
            "scene_graph_uri": f"gs://{bucket}/{pipeline_prefix}/scene_graph.json",
            "route_graph_uri": f"gs://{bucket}/{pipeline_prefix}/route_graph.json",
            "geometry_evidence_uri": f"gs://{bucket}/{pipeline_prefix}/geometry_evidence.json",
            "capability_checks_uri": f"gs://{bucket}/{pipeline_prefix}/capability_checks.json",
            "blocker_register_uri": f"gs://{bucket}/{pipeline_prefix}/blocker_register.json",
            "readiness_decision_uri": f"gs://{bucket}/{pipeline_prefix}/readiness_decision.json",
            "task_hypothesis_report_uri": f"gs://{bucket}/{pipeline_prefix}/task_hypothesis_report.json",
            "normalized_task_hypothesis_uri": f"gs://{bucket}/{pipeline_prefix}/normalized_task_hypothesis.json",
            "geometry_summary_uri": geometry_artifacts.get("geometry_summary_uri"),
            "geometry_manifest_uri": geometry_artifacts.get("geometry_manifest_uri"),
        }
        opportunity_handoff["readiness_state"] = readiness_decision.get("status")
        opportunity_handoff["qualification_state"] = readiness_decision.get("status")
        opportunity_handoff["downstream_evaluation_eligibility"] = readiness_decision.get("status") == "ready"
        opportunity_handoff["match_ready"] = bool(
            readiness_decision.get("status") == "ready"
            and opportunity_handoff.get("downstream_evaluation_eligibility") is True
        )
        write_json(pipeline_dir / "qualification_record.json", qualification_record)
        write_json(pipeline_dir / "qualification_brief.json", qualification_brief)
        write_json(pipeline_dir / "scene_graph.json", scene_graph)
        write_json(pipeline_dir / "route_graph.json", route_graph)
        write_json(pipeline_dir / "geometry_evidence.json", geometry_evidence)
        write_json(pipeline_dir / "capability_checks.json", capability_checks)
        write_json(pipeline_dir / "blocker_register.json", blocker_register)
        write_json(pipeline_dir / "readiness_decision.json", readiness_decision)
        write_json(pipeline_dir / "human_actions_required.json", human_actions_required)
        write_json(pipeline_dir / "world_model_fit_summary.json", world_model_fit_summary)
        write_json(pipeline_dir / "capturer_payout_recommendation.json", capturer_payout_recommendation)
        write_json(pipeline_dir / "provenance_summary.json", provenance_summary)
        if isinstance(weakness_summary, Mapping):
            write_json(pipeline_dir / "qualification_weakness_summary.json", dict(weakness_summary))
        if isinstance(recapture_instructions, Mapping):
            write_json(pipeline_dir / "recapture_instructions.json", dict(recapture_instructions))
        write_text(
            pipeline_dir / "readiness_report.md",
            _render_readiness_report(
                descriptor=descriptor,
                readiness_decision=readiness_decision,
                blocker_register=blocker_register,
                human_actions_required=human_actions_required,
                task_hypothesis_report=task_hypothesis_report,
            ),
        )
        write_json(pipeline_dir / "opportunity_handoff.json", opportunity_handoff)
        gates.append(
            QualificationGate(
                "qualification_gate",
                True,
                f"readiness_state={qualification_record.get('readiness_state')}",
            )
        )

        stage = "completion"
        pipeline_summary = _build_pipeline_summary(
            bucket=bucket,
            descriptor_uri=descriptor_gcs_uri,
            qa_report_uri=qa_report_uri,
            object_index_uri=object_index_uri,
            pipeline_prefix=pipeline_prefix,
            pipeline_dir=pipeline_dir,
            task_targets_payload=task_targets_with_index,
            scorecard=scorecard,
            qualification_record=qualification_record,
        )
        write_json(pipeline_dir / "pipeline_summary.json", pipeline_summary)
        privacy_world_model_ready = str(privacy_processing.get("status") or "").strip().lower() in {
            "no_people_detected",
            "person_removed",
            "face_anonymized_fallback",
            "full_frame_redacted_local_proof",
        }
        scene_memory_artifacts = (
            _write_scene_memory_bundle(
                storage_root=storage_root,
                bucket=bucket,
                pipeline_prefix=pipeline_prefix,
                pipeline_dir=pipeline_dir,
                descriptor=descriptor,
                scorecard=scorecard,
                qualification_record=qualification_record,
                geometry_artifacts=geometry_artifacts,
                depth_conditioning=privacy_processing.get("depth_conditioning"),
            )
            if "scene_memory" in downstream_requested_lanes and privacy_world_model_ready
            else _empty_downstream_artifacts()
        )
        opportunity_handoff = attach_handoff_package_paths(
            opportunity_handoff,
            pipeline_dir=pipeline_dir,
            metadata=descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {},
        )
        write_json(pipeline_dir / "opportunity_handoff.json", opportunity_handoff)

        quality_report = {
            "schema_version": "v1",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "status": (
                "passed"
                if (
                    str(capture_fidelity_review.get("status") or "").strip().lower() == "succeeded"
                    and str(privacy_processing.get("status") or "").strip().lower() != "failed_closed"
                )
                else "blocked"
            ),
            "generated_at": utc_now_iso(),
            "readiness_state": readiness_decision.get("status"),
            "completeness_status": scorecard.get("completeness_status"),
            "gates": [gate.to_dict() for gate in gates],
            "artifacts": _present_artifacts({
                "descriptor_uri": descriptor_gcs_uri,
                "qa_report_uri": qa_report_uri,
                "task_targets": f"gs://{bucket}/{pipeline_prefix}/task_targets.json",
                "runtime_preflight_report": f"gs://{bucket}/{pipeline_prefix}/runtime_preflight_report.json",
                "presentation_demo_preflight_report": f"gs://{bucket}/{pipeline_prefix}/presentation_demo_preflight_report.json",
                "site_intake": f"gs://{bucket}/{pipeline_prefix}/site_intake.json",
                "capture_package_manifest": f"gs://{bucket}/{pipeline_prefix}/capture_package_manifest.json",
                "capture_qa_scorecard": f"gs://{bucket}/{pipeline_prefix}/capture_qa_scorecard.json",
                "task_hypothesis_report": f"gs://{bucket}/{pipeline_prefix}/task_hypothesis_report.json",
                "normalized_task_hypothesis": f"gs://{bucket}/{pipeline_prefix}/normalized_task_hypothesis.json",
                "task_scope_record": f"gs://{bucket}/{pipeline_prefix}/task_scope_record.json",
                "qualification_record": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
                "qualification_brief": f"gs://{bucket}/{pipeline_prefix}/qualification_brief.json",
                "scene_graph": f"gs://{bucket}/{pipeline_prefix}/scene_graph.json",
                "route_graph": f"gs://{bucket}/{pipeline_prefix}/route_graph.json",
                "geometry_evidence": f"gs://{bucket}/{pipeline_prefix}/geometry_evidence.json",
                "capability_checks": f"gs://{bucket}/{pipeline_prefix}/capability_checks.json",
                "blocker_register": f"gs://{bucket}/{pipeline_prefix}/blocker_register.json",
                "readiness_decision": f"gs://{bucket}/{pipeline_prefix}/readiness_decision.json",
                "human_actions_required": f"gs://{bucket}/{pipeline_prefix}/human_actions_required.json",
                "gemini_capture_fidelity_review": f"gs://{bucket}/{pipeline_prefix}/gemini_capture_fidelity_review.json",
                "privacy_processing_manifest": f"gs://{bucket}/{pipeline_prefix}/privacy_processing_manifest.json",
                "privacy_verification_report": f"gs://{bucket}/{pipeline_prefix}/privacy_verification_report.json",
                "privacy_depth_manifest": (
                    (privacy_processing.get("depth_conditioning") or {}).get("depth_manifest_uri")
                    if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
                    else None
                ),
                "privacy_confidence_manifest": (
                    (privacy_processing.get("depth_conditioning") or {}).get("confidence_manifest_uri")
                    if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
                    else None
                ),
                "world_model_fit_summary": f"gs://{bucket}/{pipeline_prefix}/world_model_fit_summary.json",
                "capturer_payout_recommendation": f"gs://{bucket}/{pipeline_prefix}/capturer_payout_recommendation.json",
                "provenance_summary": f"gs://{bucket}/{pipeline_prefix}/provenance_summary.json",
                "geometry_manifest": geometry_artifacts.get("geometry_manifest_uri"),
                "geometry_summary": geometry_artifacts.get("geometry_summary_uri"),
                **(
                    {"qualification_weakness_summary": f"gs://{bucket}/{pipeline_prefix}/qualification_weakness_summary.json"}
                    if isinstance(weakness_summary, Mapping)
                    else {}
                ),
                **(
                    {"recapture_instructions": f"gs://{bucket}/{pipeline_prefix}/recapture_instructions.json"}
                    if isinstance(recapture_instructions, Mapping)
                    else {}
                ),
                "readiness_report": f"gs://{bucket}/{pipeline_prefix}/readiness_report.md",
                "opportunity_handoff": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
                "pipeline_summary": f"gs://{bucket}/{pipeline_prefix}/pipeline_summary.json",
                "scene_memory_manifest": scene_memory_artifacts["scene_memory_manifest_uri"],
                "scene_memory_readiness": scene_memory_artifacts["scene_memory_readiness_uri"],
                "conditioning_bundle": scene_memory_artifacts["conditioning_bundle_uri"],
                "preview_simulation_manifest": scene_memory_artifacts["preview_simulation_manifest_uri"],
                "presentation_bundle": scene_memory_artifacts["presentation_bundle_uri"],
                "presentation_world_manifest": scene_memory_artifacts["presentation_world_manifest_uri"],
                "runtime_demo_manifest": scene_memory_artifacts["runtime_demo_manifest_uri"],
                "gen3c_adapter_manifest": scene_memory_artifacts["gen3c_adapter_manifest_uri"],
                "site_world_runtime_adapter_manifest": scene_memory_artifacts["site_world_runtime_adapter_manifest_uri"],
                "cosmos_transfer_adapter_manifest": scene_memory_artifacts["cosmos_transfer_adapter_manifest_uri"],
                **(
                    {"privacy_processed_video": str(privacy_processing.get("privacy_processed_video_uri"))}
                    if privacy_processing.get("privacy_processed_video_uri")
                    else {}
                ),
            }),
        }
        write_json(pipeline_dir / "qualification_quality_report.json", quality_report)
        write_json(pipeline_dir / "swap_quality_report.json", quality_report)
        rights_provenance_review = build_rights_provenance_review(
            rights_summary=site_intake.get("capture_rights")
            if isinstance(site_intake.get("capture_rights"), Mapping)
            else {},
            privacy_processing=privacy_processing,
            provenance_summary=provenance_summary,
            site_identity=(
                effective_metadata.get("site_identity")
                if isinstance(effective_metadata, Mapping)
                and isinstance(effective_metadata.get("site_identity"), Mapping)
                else {}
            ),
            adjacent_systems=_string_list(
                effective_metadata.get("adjacent_systems")
                if isinstance(effective_metadata, Mapping)
                else None
            ),
            artifact_uris={
                "rights_and_compliance_summary_uri": f"gs://{bucket}/{pipeline_prefix}/rights_and_compliance_summary.json",
                "privacy_processing_manifest_uri": f"gs://{bucket}/{pipeline_prefix}/privacy_processing_manifest.json",
                "provenance_summary_uri": f"gs://{bucket}/{pipeline_prefix}/provenance_summary.json",
            },
            required_use_classes=rights_required_use_classes,
        )
        write_json(pipeline_dir / "rights_provenance_review.json", rights_provenance_review)
        site_package_result = write_blueprint_canonical_site_package(
            descriptor=descriptor.to_dict(),
            capture_root=capture_root,
            pipeline_dir=pipeline_dir,
            bucket=bucket,
            storage_root=storage_root,
            pipeline_prefix=pipeline_prefix,
            descriptor_uri=descriptor_gcs_uri,
            qa_report_uri=qa_report_uri,
            qa_report=qa_report,
            privacy_processing=privacy_processing,
            worldlabs_input=worldlabs_input,
            geometry_artifacts=geometry_artifacts,
            provenance_summary=provenance_summary,
            rights_provenance_review=rights_provenance_review,
            site_intake=site_intake,
            scorecard=scorecard,
            qualification_record=qualification_record,
            task_targets_payload=task_targets_with_index,
            task_hypothesis_report=task_hypothesis_report,
            object_index_entries=object_index_entries,
        )
        descriptor_payload = descriptor.to_dict()
        metadata_payload = dict(descriptor_payload.get("metadata") or {})
        metadata_payload["canonical_site_package_uri"] = site_package_result["canonical_site_package_uri"]
        metadata_payload["provider_adapter_inputs"] = dict(site_package_result["provider_adapter_input_uris"])
        descriptor_payload["metadata"] = metadata_payload
        write_json(descriptor_path, descriptor_payload)
        descriptor = CaptureDescriptor.from_dict(descriptor_payload)

        qualification_state = derive_webapp_qualification_state(
            readiness_state=qualification_record.get("readiness_state"),
            completeness_status=scorecard.get("completeness_status"),
        )
        opportunity_state = derive_webapp_opportunity_state(
            qualification_state=qualification_state,
        )

        preview_provider_name = str(os.getenv("BLUEPRINT_PREVIEW_PROVIDER") or "world_labs").strip()
        requested_outputs = set(descriptor.requested_outputs or [])
        preview_requested = "preview_simulation" in requested_outputs or "preview" in requested_outputs
        preview_input_ready = bool(worldlabs_input.get("output_video_uri")) and str(worldlabs_input.get("status") or "").strip().lower() == "ready"
        provider_run = (
            run_preview_provider(
                provider_name=preview_provider_name,
                descriptor=descriptor.to_dict(),
                capture_root=capture_root,
                pipeline_dir=pipeline_dir,
                provider_adapter_input=site_package_result["provider_adapter_inputs"].get(
                    "world_labs_marble"
                ),
            )
            if preview_requested and preview_input_ready
            else {
                "schema_version": "v1",
                "provider_name": None,
                "provider_model": None,
                "provider_run_id": "",
                "status": "failed" if preview_requested and not preview_input_ready else "not_requested",
                "preview_manifest_uri": str(pipeline_dir / "preview_manifest.json"),
                "artifact_uris": {},
                "cost_usd": None,
                "latency_ms": None,
                "labeling": (
                    dict(worldlabs_input.get("input_labeling"))
                    if isinstance(worldlabs_input.get("input_labeling"), Mapping)
                    else {}
                ),
                "failure_reason": (
                    f"worldlabs_input_status:{worldlabs_input.get('status')}"
                    if preview_requested and not preview_input_ready
                    else None
                ),
                "canonical_site_package_uri": site_package_result["canonical_site_package_uri"],
                "provider_adapter_input_uri": site_package_result["provider_adapter_input_uris"].get(
                    "world_labs_marble"
                ),
                "adapter_input_status": site_package_result["provider_adapter_inputs"]
                .get("world_labs_marble", {})
                .get("status"),
                "adapter_input_blockers": site_package_result["provider_adapter_inputs"]
                .get("world_labs_marble", {})
                .get("blockers", []),
                "provenance": {"canonical": False, "derived": True},
            }
        )
        worldlabs_request_manifest_path = pipeline_dir / "worldlabs_request_manifest.json"
        worldlabs_request_manifest_uri = (
            f"gs://{bucket}/{pipeline_prefix}/worldlabs_request_manifest.json"
            if worldlabs_request_manifest_path.is_file()
            else None
        )
        worldlabs_operation_manifest_uri = (
            f"gs://{bucket}/{pipeline_prefix}/worldlabs_operation_manifest.json"
            if (pipeline_dir / "worldlabs_operation_manifest.json").is_file()
            else None
        )
        worldlabs_world_manifest_uri = (
            f"gs://{bucket}/{pipeline_prefix}/worldlabs_world_manifest.json"
            if (pipeline_dir / "worldlabs_world_manifest.json").is_file()
            else None
        )
        if worldlabs_request_manifest_uri:
            descriptor_payload = descriptor.to_dict()
            metadata_payload = dict(descriptor_payload.get("metadata") or {})
            metadata_payload["worldlabs_request_manifest_uri"] = worldlabs_request_manifest_uri
            descriptor_payload["metadata"] = metadata_payload
            write_json(descriptor_path, descriptor_payload)
            descriptor = CaptureDescriptor.from_dict(descriptor_payload)
        if not preview_requested or not preview_input_ready:
            write_json(pipeline_dir / "provider_run_manifest.json", provider_run)
            write_json(
                pipeline_dir / "preview_manifest.json",
                {
                    "schema_version": "v1",
                    "status": provider_run["status"],
                    "generated_at": utc_now_iso(),
                    "failure_reason": provider_run.get("failure_reason"),
                    "labeling": dict(provider_run.get("labeling") or {}),
                },
            )
        buyer_trust_score = build_buyer_trust_score(
            descriptor=descriptor.to_dict(),
            qualification_record=qualification_record,
            scorecard=scorecard,
            metadata=effective_metadata if isinstance(effective_metadata, Mapping) else {},
            provider_status=str(provider_run.get("status") or "not_requested"),
            fidelity_review=capture_fidelity_review,
        )
        launch_bundle = build_launch_qualification_bundle(
            descriptor=descriptor.to_dict(),
            qualification_record=qualification_record,
            scorecard=scorecard,
            readiness_decision=readiness_decision,
            site_intake=site_intake,
            buyer_trust_score=buyer_trust_score,
            provider_run=provider_run,
            privacy_processing=privacy_processing,
            fidelity_review=capture_fidelity_review,
            world_model_fit_summary=world_model_fit_summary,
            capturer_payout_recommendation=capturer_payout_recommendation,
            provenance_summary=provenance_summary,
        )
        write_json(pipeline_dir / "qualification_summary.json", launch_bundle["qualification_summary"])
        write_json(pipeline_dir / "capture_quality_summary.json", launch_bundle["capture_quality_summary"])
        write_json(pipeline_dir / "rights_and_compliance_summary.json", launch_bundle["rights_and_compliance_summary"])
        write_json(pipeline_dir / "buyer_trust_score.json", launch_bundle["buyer_trust_score"])
        write_json(pipeline_dir / "recapture_requirements.json", launch_bundle["recapture_requirements"])
        write_json(pipeline_dir / "provider_preview_status.json", launch_bundle["provider_preview_status"])
        rights_provenance_review = build_rights_provenance_review(
            rights_summary=launch_bundle["rights_and_compliance_summary"],
            privacy_processing=privacy_processing,
            provenance_summary=provenance_summary,
            site_identity=(
                effective_metadata.get("site_identity")
                if isinstance(effective_metadata, Mapping)
                and isinstance(effective_metadata.get("site_identity"), Mapping)
                else {}
            ),
            adjacent_systems=_string_list(
                effective_metadata.get("adjacent_systems")
                if isinstance(effective_metadata, Mapping)
                else None
            ),
            artifact_uris={
                "rights_and_compliance_summary_uri": f"gs://{bucket}/{pipeline_prefix}/rights_and_compliance_summary.json",
                "privacy_processing_manifest_uri": f"gs://{bucket}/{pipeline_prefix}/privacy_processing_manifest.json",
                "provenance_summary_uri": f"gs://{bucket}/{pipeline_prefix}/provenance_summary.json",
            },
            required_use_classes=rights_required_use_classes,
        )
        write_json(pipeline_dir / "rights_provenance_review.json", rights_provenance_review)

        completion_payload = _present_artifacts({
            "schema_version": "v1",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "site_submission_id": opportunity_handoff.get("site_submission_id"),
            "status": "completed",
            "completed_at": utc_now_iso(),
            "qualification_state": qualification_state,
            "opportunity_state": opportunity_state,
            "alpha_scoring_status": capture_fidelity_review.get("status"),
            "privacy_status": privacy_processing.get("status"),
            "quality_report": f"gs://{bucket}/{pipeline_prefix}/qualification_quality_report.json",
            "pipeline_summary": f"gs://{bucket}/{pipeline_prefix}/pipeline_summary.json",
            "qualification_record": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "opportunity_handoff": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
            "gemini_capture_fidelity_review": f"gs://{bucket}/{pipeline_prefix}/gemini_capture_fidelity_review.json",
            "privacy_processing_manifest": f"gs://{bucket}/{pipeline_prefix}/privacy_processing_manifest.json",
            "privacy_verification_report": f"gs://{bucket}/{pipeline_prefix}/privacy_verification_report.json",
            "privacy_depth_manifest": (
                (privacy_processing.get("depth_conditioning") or {}).get("depth_manifest_uri")
                if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
                else None
            ),
            "privacy_confidence_manifest": (
                (privacy_processing.get("depth_conditioning") or {}).get("confidence_manifest_uri")
                if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
                else None
            ),
            "world_model_fit_summary": f"gs://{bucket}/{pipeline_prefix}/world_model_fit_summary.json",
            "capturer_payout_recommendation": f"gs://{bucket}/{pipeline_prefix}/capturer_payout_recommendation.json",
            "provenance_summary": f"gs://{bucket}/{pipeline_prefix}/provenance_summary.json",
            "geometry_manifest": geometry_artifacts.get("geometry_manifest_uri"),
            "geometry_summary": geometry_artifacts.get("geometry_summary_uri"),
            "provider_run_manifest": f"gs://{bucket}/{pipeline_prefix}/provider_run_manifest.json",
            "preview_manifest": f"gs://{bucket}/{pipeline_prefix}/preview_manifest.json",
            "worldlabs_request_manifest": worldlabs_request_manifest_uri,
            "worldlabs_operation_manifest": worldlabs_operation_manifest_uri,
            "worldlabs_world_manifest": worldlabs_world_manifest_uri,
            "worldlabs_input_manifest": worldlabs_input.get("manifest_uri"),
            "worldlabs_input_audit": worldlabs_input.get("audit_uri"),
            "worldlabs_input_video": worldlabs_input.get("output_video_uri"),
            "canonical_site_package": site_package_result["canonical_site_package_uri"],
            "world_labs_marble_adapter_input": site_package_result["provider_adapter_input_uris"].get(
                "world_labs_marble"
            ),
            **(
                {"privacy_processed_video": str(privacy_processing.get("privacy_processed_video_uri"))}
                if privacy_processing.get("privacy_processed_video_uri")
                else {}
            ),
        })
        write_json(pipeline_dir / ".qualification_pipeline_complete", completion_payload)
        write_json(pipeline_dir / ".swap_pipeline_complete", completion_payload)
        webapp_sync_artifacts = _present_artifacts({
            "readiness_decision_uri": quality_report["artifacts"].get("readiness_decision"),
            "readiness_report_uri": quality_report["artifacts"].get("readiness_report"),
            "qualification_quality_report_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_quality_report.json",
            "qualification_summary_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_summary.json",
            "capture_quality_summary_uri": f"gs://{bucket}/{pipeline_prefix}/capture_quality_summary.json",
            "rights_and_compliance_summary_uri": f"gs://{bucket}/{pipeline_prefix}/rights_and_compliance_summary.json",
            "buyer_trust_score_uri": f"gs://{bucket}/{pipeline_prefix}/buyer_trust_score.json",
            "world_model_fit_summary_uri": f"gs://{bucket}/{pipeline_prefix}/world_model_fit_summary.json",
            "capturer_payout_recommendation_uri": f"gs://{bucket}/{pipeline_prefix}/capturer_payout_recommendation.json",
            "recapture_requirements_uri": f"gs://{bucket}/{pipeline_prefix}/recapture_requirements.json",
            "provider_preview_status_uri": f"gs://{bucket}/{pipeline_prefix}/provider_preview_status.json",
            "privacy_processing_manifest_uri": f"gs://{bucket}/{pipeline_prefix}/privacy_processing_manifest.json",
            "privacy_verification_report_uri": f"gs://{bucket}/{pipeline_prefix}/privacy_verification_report.json",
            "provenance_summary_uri": f"gs://{bucket}/{pipeline_prefix}/provenance_summary.json",
            "rights_provenance_review_uri": f"gs://{bucket}/{pipeline_prefix}/rights_provenance_review.json",
            "gemini_capture_fidelity_review_uri": f"gs://{bucket}/{pipeline_prefix}/gemini_capture_fidelity_review.json",
            "provider_run_manifest_uri": f"gs://{bucket}/{pipeline_prefix}/provider_run_manifest.json",
            "preview_manifest_uri": f"gs://{bucket}/{pipeline_prefix}/preview_manifest.json",
            "worldlabs_request_manifest_uri": worldlabs_request_manifest_uri,
            "worldlabs_operation_manifest_uri": worldlabs_operation_manifest_uri,
            "worldlabs_world_manifest_uri": worldlabs_world_manifest_uri,
            "worldlabs_input_manifest_uri": worldlabs_input.get("manifest_uri"),
            "worldlabs_input_audit_uri": worldlabs_input.get("audit_uri"),
            "worldlabs_input_video_uri": worldlabs_input.get("output_video_uri"),
            "canonical_site_package_uri": site_package_result["canonical_site_package_uri"],
            "world_labs_marble_adapter_input_uri": site_package_result["provider_adapter_input_uris"].get(
                "world_labs_marble"
            ),
            "privacy_depth_manifest_uri": (
                (privacy_processing.get("depth_conditioning") or {}).get("depth_manifest_uri")
                if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
                else None
            ),
            "privacy_confidence_manifest_uri": (
                (privacy_processing.get("depth_conditioning") or {}).get("confidence_manifest_uri")
                if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
                else None
            ),
            "geometry_manifest_uri": geometry_artifacts.get("geometry_manifest_uri"),
            "geometry_summary_uri": geometry_artifacts.get("geometry_summary_uri"),
            **(
                {
                    "privacy_processed_video_uri": str(privacy_processing.get("privacy_processed_video_uri")),
                    "world_model_video_uri": str(privacy_processing.get("world_model_video_uri")),
                }
                if privacy_world_model_ready and privacy_processing.get("privacy_processed_video_uri")
                else {}
            ),
            "opportunity_handoff_uri": quality_report["artifacts"].get("opportunity_handoff"),
            "human_actions_required_uri": quality_report["artifacts"].get("human_actions_required"),
            "agent_review_bundle_uri": f"gs://{bucket}/{pipeline_prefix}/agent_review_bundle.json",
            "agent_readiness_memo_uri": f"gs://{bucket}/{pipeline_prefix}/agent_readiness_memo.md",
        })
        webapp_evaluation_readiness = {
            "qualification_state": qualification_state,
            "opportunity_state": opportunity_state,
            "alpha_scoring_status": capture_fidelity_review.get("status"),
            "buyer_trust_score": buyer_trust_score,
            "qualification_summary": launch_bundle["qualification_summary"],
            "capture_quality_summary": launch_bundle["capture_quality_summary"],
            "rights_and_compliance": launch_bundle["rights_and_compliance_summary"],
            "privacy_processing": {
                "status": privacy_processing.get("status"),
                "mode": privacy_processing.get("mode"),
                "fallback_used": bool(privacy_processing.get("fallback_used")),
                "people_detected": int(privacy_processing.get("people_detected") or 0),
                "people_removed": int(privacy_processing.get("people_removed") or 0),
                "face_anonymized_segments": _string_list(privacy_processing.get("face_anonymized_segments")),
                "raw_retained": bool(privacy_processing.get("raw_retained")),
                "fail_closed": bool(privacy_processing.get("fail_closed")),
                "depth_source": privacy_processing.get("depth_source"),
                "depth_conditioning": (
                    dict(privacy_processing.get("depth_conditioning"))
                    if isinstance(privacy_processing.get("depth_conditioning"), Mapping)
                    else {}
                ),
            },
            "missing_evidence": launch_bundle["recapture_requirements"]["missing_evidence"],
            "recapture_required": launch_bundle["recapture_requirements"]["required"],
            "recapture_recommendations": launch_bundle["recapture_requirements"].get("recommendations"),
            "preview_status": launch_bundle["preview_status"],
            "provider_run": provider_run,
            "provider_preview_labeling": (
                dict(provider_run.get("labeling"))
                if isinstance(provider_run.get("labeling"), Mapping)
                else {}
            ),
            "world_model_fit_summary": world_model_fit_summary,
            "capturer_payout_recommendation": capturer_payout_recommendation,
            "provenance_summary": provenance_summary,
            "rights_provenance_review": rights_provenance_review,
            "canonical_site_package": site_package_result["canonical_site_package"],
            "provider_adapter_inputs": site_package_result["provider_adapter_inputs"],
            "advisory_geometry": _geometry_advisory_payload(geometry_artifacts),
        }
        webapp_sync_result_path = pipeline_dir / "webapp_sync_result.json"
        try:
            site_submission_id = str(
                descriptor.site_submission_id
                or opportunity_handoff.get("site_submission_id")
                or ""
            ).strip()
            buyer_request_id = str(
                descriptor.buyer_request_id
                or opportunity_handoff.get("buyer_request_id")
                or ""
            ).strip()
            capture_job_id = str(
                descriptor.capture_job_id
                or opportunity_handoff.get("capture_job_id")
                or ""
            ).strip()
            # Project package, proof-path, and review readiness truth back into the
            # WebApp control plane. This module still carries legacy "qualification"
            # naming, but the sync is part of the capture-first package lifecycle.
            webapp_sync_result = sync_webapp_pipeline_attachment(
                site_submission_id=site_submission_id,
                request_id=site_submission_id,
                buyer_request_id=buyer_request_id,
                capture_job_id=capture_job_id,
                scene_id=descriptor.scene_id,
                capture_id=descriptor.capture_id,
                pipeline_prefix=pipeline_prefix,
                qualification_state=qualification_state,
                opportunity_state=opportunity_state,
                authoritative_state_update=True,
                artifacts=webapp_sync_artifacts,
                derived_assets=_scene_memory_derived_assets(scene_memory_artifacts),
                evaluation_readiness=webapp_evaluation_readiness,
                # Rights are authoritative continuously: pass the capture root so
                # the delivery-time consent-takedown gate re-reads consent live and
                # blocks the sync on an open revocation (without this the gate is inert).
                capture_root=capture_root,
            )
        except (WebappSyncError, ValueError) as exc:
            webapp_sync_result = {
                "status": "failed",
                "reason": str(exc),
                "blocker": "webapp_sync_requires_upstream_request_job_bootstrap",
            }
            if _webapp_sync_failure_requires_stage_failure():
                write_json(webapp_sync_result_path, webapp_sync_result)
                raise StageError("webapp_sync", str(exc)) from exc
        if webapp_sync_result is None:
            webapp_sync_result = {"status": "skipped", "reason": "sync_returned_none"}
        write_pipeline_sync_result(
            pipeline_root=pipeline_dir,
            stage="qualification",
            result=webapp_sync_result,
        )
        write_alpha_readiness_summary(capture_root=capture_root)

        return {
            "status": "completed",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "pipeline_prefix": pipeline_prefix,
            "readiness_state": qualification_record.get("readiness_state"),
            "completeness_status": scorecard.get("completeness_status"),
            "match_ready": opportunity_handoff.get("match_ready"),
            "webapp_sync_result_uri": f"gs://{bucket}/{pipeline_prefix}/webapp_sync_result.json",
        }

    except Exception as exc:
        if isinstance(exc, StageError):
            stage = exc.stage
        ensure_dir(pipeline_dir)
        try:
            _write_failure(
                pipeline_dir=pipeline_dir,
                descriptor_uri=descriptor_gcs_uri,
                stage=stage,
                error=exc,
                gates=gates,
            )
            failure_quality_report = {
                "schema_version": "v1",
                "lane": "qualification",
                "scene_id": scene_id,
                "capture_id": capture_id,
                "status": "failed",
                "generated_at": utc_now_iso(),
                "failed_stage": stage,
                "error": str(exc),
                "gates": [gate.to_dict() for gate in gates],
            }
            write_json(pipeline_dir / "qualification_quality_report.json", failure_quality_report)
            write_json(pipeline_dir / "swap_quality_report.json", failure_quality_report)
        except Exception:
            pass
        raise PipelineError(str(exc)) from exc
