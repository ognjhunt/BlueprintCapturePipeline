"""Legacy compatibility artifact builders layered on the site-world pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .capture_enrichment_llm import build_capture_enrichment_runner
from .common import (
    PipelineError,
    StageError,
    ensure_dir,
    has_nonempty_file,
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
from .industrial_ontology import classify_industrial_entity, derive_capture_plan_tags, industrial_tags_for_label
from .ios_manifest import IOSManifest, load_object_index, load_raw_manifest, resolve_object_index_uri
from .object_index_stage import ensure_object_index_stage
from .task_targets import infer_task_targets, write_task_targets
from .webapp_sync import (
    derive_webapp_opportunity_state,
    derive_webapp_qualification_state,
    sync_webapp_pipeline_attachment,
)


@dataclass
class QualificationGate:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


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
        descriptor.evidence_tier == "glasses_with_validated_scaffolding"
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
            "neoverse_adapter_manifest_path": _relative_path_from(
                handoff_dir,
                scene_memory_manifest.parent / "adapter_manifests" / "neoverse.json",
            ),
            "cosmos_transfer_adapter_manifest_path": _relative_path_from(
                handoff_dir,
                scene_memory_manifest.parent / "adapter_manifests" / "cosmos_transfer.json",
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
        "derived_scene_generation_allowed": bool(raw.get("derived_scene_generation_allowed", True)),
        "data_licensing_allowed": bool(raw.get("data_licensing_allowed", False)),
        "capture_contributor_payout_eligible": bool(raw.get("capture_contributor_payout_eligible", False)),
    }


def _scene_memory_capture_summary(descriptor: CaptureDescriptor) -> Dict[str, Any]:
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
            }
        ),
        "operator_notes": _string_list(capture_summary.get("operator_notes")),
        "world_model_candidate": bool(
            capture_summary.get("world_model_candidate", quality.get("world_model_candidate"))
        ),
    }


def _build_scene_memory_readiness(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
) -> Dict[str, Any]:
    capture_summary = _scene_memory_capture_summary(descriptor)
    completeness_status = str(scorecard.get("completeness_status") or "unknown")
    metric_ready = bool(qualification_record.get("metric_ready"))
    readiness_state = str(qualification_record.get("readiness_state") or "not_ready_yet")
    rights = _capture_rights(descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {})
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
                "passed": metric_ready or descriptor.arkit_poses_uri is not None,
                "detail": "explicit conditioning is available from metric geometry or ARKit poses"
                if (metric_ready or descriptor.arkit_poses_uri is not None)
                else "scene memory will rely on monocular-only conditioning",
            },
        ],
    }


def _scene_memory_derived_assets(
    scene_memory_artifacts: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    scene_memory_status = str(scene_memory_artifacts.get("scene_memory_status") or "needs_more_evidence")
    preview_status = str(scene_memory_artifacts.get("preview_simulation_status") or "review_required")
    return {
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


def _write_scene_memory_bundle(
    *,
    storage_root: Path,
    bucket: str,
    pipeline_prefix: str,
    pipeline_dir: Path,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
) -> Dict[str, Any]:
    scene_memory_dir = pipeline_dir / "scene_memory"
    adapter_dir = scene_memory_dir / "adapter_manifests"
    preview_dir = pipeline_dir / "preview_simulation"
    ensure_dir(scene_memory_dir)
    ensure_dir(adapter_dir)
    ensure_dir(preview_dir)

    readiness_payload = _build_scene_memory_readiness(
        descriptor=descriptor,
        scorecard=scorecard,
        qualification_record=qualification_record,
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

    conditioning_bundle = {
        "schema_version": "v1",
        "lane": "scene_memory",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "raw_video_uri": descriptor.raw_video_uri,
        "frames_index_uri": descriptor.frames_index_uri,
        "keyframe_uri": descriptor.keyframe_uri,
        "arkit": {
            "poses_uri": descriptor.arkit_poses_uri,
            "intrinsics_uri": descriptor.arkit_intrinsics_uri,
            "depth_prefix_uri": descriptor.arkit_depth_prefix_uri,
            "confidence_prefix_uri": descriptor.arkit_confidence_prefix_uri,
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
    }
    write_json(scene_memory_dir / "conditioning_bundle.json", conditioning_bundle)

    scene_memory_manifest = {
        "schema_version": "v1",
        "lane": "scene_memory",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "scene_memory_readiness_uri": f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'scene_memory_readiness.json', storage_root)}",
        "conditioning_bundle_uri": f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'conditioning_bundle.json', storage_root)}",
        "authoritative_artifacts": {
            "qualification_record_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "qualification_brief_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_brief.json",
            "readiness_decision_uri": f"gs://{bucket}/{pipeline_prefix}/readiness_decision.json",
            "human_actions_required_uri": f"gs://{bucket}/{pipeline_prefix}/human_actions_required.json",
        },
        "rights": readiness_payload["rights"],
    }
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
        "neoverse": {
            "family": "NeoVerse",
            "preferred_conditioning": ["rgb_video", "camera_trajectory", "feed_forward_4d_reconstruction"],
            "required_conditioning": ["rgb_video"],
            "execution_mode": "local_gpu_runtime",
            "reconstruction_backend_name": "neoverse",
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
        "scene_memory_manifest_uri": f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'scene_memory_manifest.json', storage_root)}",
        "supported_backends": ["gen3c", "neoverse", "cosmos_transfer"],
        "note": "Low-volume preview generation only. High-volume synthetic frames and datasets belong in BlueprintValidation.",
    }
    write_json(preview_dir / "preview_simulation_manifest.json", preview_manifest)

    return {
        "scene_memory_manifest_uri": f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'scene_memory_manifest.json', storage_root)}",
        "scene_memory_readiness_uri": f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'scene_memory_readiness.json', storage_root)}",
        "conditioning_bundle_uri": f"gs://{bucket}/{relative_scene_path(scene_memory_dir / 'conditioning_bundle.json', storage_root)}",
        "preview_simulation_manifest_uri": f"gs://{bucket}/{relative_scene_path(preview_dir / 'preview_simulation_manifest.json', storage_root)}",
        "scene_memory_status": readiness_payload["status"],
        "preview_simulation_status": preview_manifest["status"],
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
    calibration_sufficient = descriptor.capture_modality != "glasses_plus_scaffolding" or bool(descriptor.calibration_assets)
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
    elif descriptor.capture_modality == "glasses_plus_scaffolding" and not bool(descriptor.scaffolding_validation.get("validated_metric_bundle")):
        risks.append(
            {
                "id": "missing_validated_scaffolding",
                "severity": "high",
                "category": "geometry",
                "detail": "Glasses scaffolding lacks validated scale and pose coverage required for metric checks.",
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
    site_submission_id = str(metadata.get("site_submission_id") or "").strip() or (
        f"{descriptor.scene_id}:{descriptor.capture_id}"
    )
    opportunity_id = str(metadata.get("opportunity_id") or "").strip() or site_submission_id
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
            "human_actions_required_uri": f"gs://{bucket}/{pipeline_prefix}/human_actions_required.json",
            "qualification_quality_report_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_quality_report.json",
        },
        "source_files": {
            "runtime_preflight_report": _local_file_pointer(pipeline_dir / "runtime_preflight_report.json"),
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


_GENERIC_CAPABILITY_ENVELOPE = {
    "minimum_path_width_m": 0.95,
    "preferred_path_width_m": 1.15,
    "maximum_threshold_height_m": 0.04,
    "maximum_target_reach_distance_m": 1.1,
    "maximum_workcell_span_m": 2.5,
    "maximum_hidden_zone_bound": 0.35,
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
        if not isinstance(node, Mapping):
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
        descriptor = CaptureDescriptor.from_file(descriptor_path)
        scene_id = descriptor.scene_id
        capture_id = descriptor.capture_id
        pipeline_prefix = to_pipeline_prefix(scene_id, capture_id)
        pipeline_dir = storage_root / pipeline_prefix
        ensure_dir(pipeline_dir)

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

        try:
            manifest = load_raw_manifest(descriptor.raw_prefix_uri, gcs_root=storage_root)
            manifest_uri = f"{descriptor.raw_prefix_uri.rstrip('/')}/manifest.json"
            manifest_path = resolve_gs_uri_to_path(manifest_uri, storage_root)
            try:
                stage_result = ensure_object_index_stage(
                    capture_root=descriptor_path.parent,
                    force_rebuild=parse_bool(os.getenv("OBJECT_INDEX_FORCE_REBUILD"), default=False),
                )
            except Exception:
                stage_result = {}
            if isinstance(stage_result.get("grounding_payload"), Mapping):
                grounding_payload = dict(stage_result["grounding_payload"])
            object_index_uri = (
                str(stage_result.get("object_index_uri") or "").strip()
                or str(descriptor.object_index_uri or "").strip()
                or resolve_object_index_uri(descriptor.raw_prefix_uri, manifest)
            )
            if object_index_uri:
                object_index_path = resolve_gs_uri_to_path(object_index_uri, storage_root)
                object_index_entries = load_object_index(object_index_uri, gcs_root=storage_root)
        except Exception:
            manifest = None
            object_index_uri = None
            object_index_entries = []
            grounding_payload = None

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
        )
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
        scene_memory_artifacts = _write_scene_memory_bundle(
            storage_root=storage_root,
            bucket=bucket,
            pipeline_prefix=pipeline_prefix,
            pipeline_dir=pipeline_dir,
            descriptor=descriptor,
            scorecard=scorecard,
            qualification_record=qualification_record,
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
            "status": "passed",
            "generated_at": utc_now_iso(),
            "readiness_state": readiness_decision.get("status"),
            "completeness_status": scorecard.get("completeness_status"),
            "gates": [gate.to_dict() for gate in gates],
            "artifacts": {
                "descriptor_uri": descriptor_gcs_uri,
                "qa_report_uri": qa_report_uri,
                "task_targets": f"gs://{bucket}/{pipeline_prefix}/task_targets.json",
                "runtime_preflight_report": f"gs://{bucket}/{pipeline_prefix}/runtime_preflight_report.json",
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
                "gen3c_adapter_manifest": scene_memory_artifacts["gen3c_adapter_manifest_uri"],
                "neoverse_adapter_manifest": scene_memory_artifacts["neoverse_adapter_manifest_uri"],
                "cosmos_transfer_adapter_manifest": scene_memory_artifacts["cosmos_transfer_adapter_manifest_uri"],
            },
        }
        write_json(pipeline_dir / "qualification_quality_report.json", quality_report)
        write_json(pipeline_dir / "swap_quality_report.json", quality_report)

        qualification_state = derive_webapp_qualification_state(
            readiness_state=qualification_record.get("readiness_state"),
            completeness_status=scorecard.get("completeness_status"),
        )
        opportunity_state = derive_webapp_opportunity_state(
            qualification_state=qualification_state,
        )

        completion_payload = {
            "schema_version": "v1",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "site_submission_id": opportunity_handoff.get("site_submission_id"),
            "status": "completed",
            "completed_at": utc_now_iso(),
            "qualification_state": qualification_state,
            "opportunity_state": opportunity_state,
            "quality_report": f"gs://{bucket}/{pipeline_prefix}/qualification_quality_report.json",
            "pipeline_summary": f"gs://{bucket}/{pipeline_prefix}/pipeline_summary.json",
            "qualification_record": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "opportunity_handoff": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
            "scene_memory_manifest": scene_memory_artifacts["scene_memory_manifest_uri"],
            "preview_simulation_manifest": scene_memory_artifacts["preview_simulation_manifest_uri"],
        }
        write_json(pipeline_dir / ".qualification_pipeline_complete", completion_payload)
        write_json(pipeline_dir / ".swap_pipeline_complete", completion_payload)
        sync_webapp_pipeline_attachment(
            site_submission_id=opportunity_handoff.get("site_submission_id"),
            request_id=opportunity_handoff.get("site_submission_id"),
            scene_id=descriptor.scene_id,
            capture_id=descriptor.capture_id,
            pipeline_prefix=pipeline_prefix,
            qualification_state=qualification_state,
            opportunity_state=opportunity_state,
            artifacts={
                "readiness_decision_uri": quality_report["artifacts"]["readiness_decision"],
                "readiness_report_uri": quality_report["artifacts"]["readiness_report"],
                "qualification_quality_report_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_quality_report.json",
                "opportunity_handoff_uri": quality_report["artifacts"]["opportunity_handoff"],
                "human_actions_required_uri": quality_report["artifacts"]["human_actions_required"],
                "agent_review_bundle_uri": f"gs://{bucket}/{pipeline_prefix}/agent_review_bundle.json",
                "agent_readiness_memo_uri": f"gs://{bucket}/{pipeline_prefix}/agent_readiness_memo.md",
                "scene_memory_manifest_uri": scene_memory_artifacts["scene_memory_manifest_uri"],
                "scene_memory_readiness_uri": scene_memory_artifacts["scene_memory_readiness_uri"],
                "conditioning_bundle_uri": scene_memory_artifacts["conditioning_bundle_uri"],
                "preview_simulation_manifest_uri": scene_memory_artifacts["preview_simulation_manifest_uri"],
                "gen3c_adapter_manifest_uri": scene_memory_artifacts["gen3c_adapter_manifest_uri"],
                "neoverse_adapter_manifest_uri": scene_memory_artifacts["neoverse_adapter_manifest_uri"],
                "cosmos_transfer_adapter_manifest_uri": scene_memory_artifacts["cosmos_transfer_adapter_manifest_uri"],
            },
            derived_assets=_scene_memory_derived_assets(scene_memory_artifacts),
        )

        return {
            "status": "completed",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "pipeline_prefix": pipeline_prefix,
            "readiness_state": qualification_record.get("readiness_state"),
            "completeness_status": scorecard.get("completeness_status"),
            "match_ready": opportunity_handoff.get("match_ready"),
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
