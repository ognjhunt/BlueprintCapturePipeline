"""Helpers for runtime-layer grounding policy packaging."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


PROTECTED_OBSERVED_THRESHOLD = 0.85
PROTECTED_RECONSTRUCTED_THRESHOLD = 0.80
EDITABLE_LOW_CONFIDENCE_THRESHOLD = 0.65
TASK_CRITICAL_OVERRIDE_THRESHOLD = 0.70
TASK_CRITICAL_DILATION_PX = 3
DEGRADED_EDITABLE_RATIO_THRESHOLD = 0.40
LOCK_VIOLATION_RETRY_BUDGET = 1


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def normalized_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def grounding_fields_from_provenance(provenance: Mapping[str, Any] | None) -> Dict[str, Any]:
    provenance = dict(provenance or {})
    return {
        "grounding_level": str(provenance.get("grounding_level") or "").strip() or "generated",
        "confidence": _safe_float(provenance.get("confidence")),
        "evidence_sources": [
            str(item).strip()
            for item in provenance.get("evidence_sources", [])
            if str(item).strip()
        ]
        if isinstance(provenance.get("evidence_sources"), list)
        else [],
        "observation_coverage": (
            dict(provenance.get("observation_coverage") or {})
            if isinstance(provenance.get("observation_coverage"), Mapping)
            else {}
        ),
        "canonical_truth": bool(provenance.get("canonical_truth")),
        "presentation_only": bool(provenance.get("presentation_only")),
    }


def with_grounding_fields(
    payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    result = dict(payload)
    if provenance is None and isinstance(result.get("provenance"), Mapping):
        provenance = dict(result.get("provenance") or {})
    result.update(grounding_fields_from_provenance(provenance))
    if isinstance(extra, Mapping):
        result.update(dict(extra))
    return result


def task_critical_object_ids(task_entries: Sequence[Mapping[str, Any]]) -> set[str]:
    critical: set[str] = set()
    for task in task_entries:
        if not isinstance(task, Mapping):
            continue
        for key in ("target_object_ids", "articulation_required_ids"):
            values = task.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                text = str(item or "").strip()
                if text:
                    critical.add(text)
    return critical


def classify_region(
    *,
    grounding_level: Any,
    confidence: Any,
    task_critical: bool = False,
    provenance_present: bool = True,
) -> str:
    if not provenance_present:
        return "editable"

    level = str(grounding_level or "").strip().lower()
    score = _safe_float(confidence)
    if not level:
        return "editable"
    if score is None:
        return "editable"
    if score < EDITABLE_LOW_CONFIDENCE_THRESHOLD:
        return "editable"
    if task_critical and score >= TASK_CRITICAL_OVERRIDE_THRESHOLD:
        return "locked"
    if level == "observed":
        return "locked" if score >= PROTECTED_OBSERVED_THRESHOLD else "editable"
    if level == "reconstructed":
        if score >= PROTECTED_RECONSTRUCTED_THRESHOLD:
            return "locked"
        if score >= EDITABLE_LOW_CONFIDENCE_THRESHOLD:
            return "uncertain"
        return "editable"
    if level in {"inferred", "generated"}:
        return "editable"
    return "editable"


def build_protected_regions_manifest(
    *,
    scene_id: str,
    capture_id: str,
    object_geometry_manifest: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    tasks = (
        task_anchor_manifest.get("tasks")
        if isinstance(task_anchor_manifest.get("tasks"), list)
        else []
    )
    critical_ids = task_critical_object_ids([task for task in tasks if isinstance(task, Mapping)])
    objects = (
        object_geometry_manifest.get("objects")
        if isinstance(object_geometry_manifest.get("objects"), list)
        else []
    )
    object_geometry_status = str(object_geometry_manifest.get("status") or "").strip().lower()
    regions = []
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        object_id = str(item.get("object_id") or "").strip()
        provenance = item.get("provenance") if isinstance(item.get("provenance"), Mapping) else {}
        fields = grounding_fields_from_provenance(provenance)
        task_critical = bool(item.get("task_critical")) or object_id in critical_ids
        classification = classify_region(
            grounding_level=fields["grounding_level"],
            confidence=fields["confidence"],
            task_critical=task_critical,
            provenance_present=bool(provenance),
        )
        regions.append(
            {
                "region_id": f"object:{object_id or 'unknown'}",
                "region_type": "object",
                "object_id": object_id or None,
                "label": str(item.get("label") or "").strip() or None,
                "classification": classification,
                "task_critical": task_critical,
                "confidence_thresholds": {
                    "protected_observed": PROTECTED_OBSERVED_THRESHOLD,
                    "protected_reconstructed": PROTECTED_RECONSTRUCTED_THRESHOLD,
                    "editable_low_confidence": EDITABLE_LOW_CONFIDENCE_THRESHOLD,
                    "task_critical_override": TASK_CRITICAL_OVERRIDE_THRESHOLD,
                },
                "geometry_refs": {
                    "placement_bbox": dict(item.get("placement_bbox") or {})
                    if isinstance(item.get("placement_bbox"), Mapping)
                    else {},
                    "mesh_glb_path": str(item.get("mesh_glb_path") or "").strip() or None,
                    "collision_hulls": [
                        str(hull.get("path") or "").strip()
                        for hull in item.get("collision_hulls", [])
                        if isinstance(hull, Mapping) and str(hull.get("path") or "").strip()
                    ],
                },
                "mask_refs": [
                    {
                        "view_id": str(mask.get("view_id") or "").strip(),
                        "mask_path": str(mask.get("mask_path") or "").strip(),
                        "image_path": str(mask.get("image_path") or "").strip(),
                    }
                    for mask in item.get("visual_replacement_masks", [])
                    if isinstance(mask, Mapping) and str(mask.get("mask_path") or "").strip()
                ],
                "coverage_refs": {
                    "selected_views": [
                        {
                            "view_id": str(view.get("view_id") or "").strip(),
                            "image_path": str(view.get("image_path") or "").strip(),
                            "mask_path": str(view.get("mask_path") or "").strip(),
                            "source_mode": str(view.get("source_mode") or "").strip(),
                        }
                        for view in item.get("selected_views", [])
                        if isinstance(view, Mapping)
                    ],
                },
                **fields,
            }
        )
    grounding_status = "grounded"
    ungrounded_reason: Optional[str] = None
    if object_geometry_status in {"empty_object_index", "missing_object_index"}:
        grounding_status = "ungrounded"
        ungrounded_reason = object_geometry_status
    elif not regions:
        grounding_status = "ungrounded"
        ungrounded_reason = "no_grounded_regions"
    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "grounding_status": grounding_status,
        "ungrounded_reason": ungrounded_reason,
        "region_count": len(regions),
        "thresholds": {
            "protected_observed": PROTECTED_OBSERVED_THRESHOLD,
            "protected_reconstructed": PROTECTED_RECONSTRUCTED_THRESHOLD,
            "editable_low_confidence": EDITABLE_LOW_CONFIDENCE_THRESHOLD,
            "task_critical_override": TASK_CRITICAL_OVERRIDE_THRESHOLD,
            "task_critical_dilation_px": TASK_CRITICAL_DILATION_PX,
            "unknown_is_editable": True,
            "ungrounded_default_mode": "canonical_only",
            "unsafe_editable_override_flag": "unsafe_allow_blocked_site_world",
        },
        "regions": regions,
    }


def build_canonical_render_policy() -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "compositing_mode": "runtime_layer_grounded",
        "lock_rules": {
            "hard_rule": "presentation runtime must never modify locked regions",
            "unknown_is_editable": True,
            "task_critical_dilation_px": TASK_CRITICAL_DILATION_PX,
        },
        "uncertain_region_policy": {
            "mode": "canonical_preferred",
            "edit_allowed_only_when": "geometry_consistent_with_canonical_depth_or_silhouette",
        },
        "fallback_behavior": {
            "on_locked_region_violation": "canonical_only",
            "retry_budget": LOCK_VIOLATION_RETRY_BUDGET,
            "rejection_mode": "retry_then_fallback",
        },
        "presentation_quality": {
            "degraded_when_editable_ratio_gt": DEGRADED_EDITABLE_RATIO_THRESHOLD,
            "degraded_label": "editable_ratio_gt_0.40",
        },
    }


def build_presentation_variance_policy() -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "allowed_variable_inputs": [
            "trajectory",
            "camera_path",
            "prompt",
            "presentation_model",
            "style_settings",
        ],
        "allowed_editable_region_classes": [
            "inferred",
            "generated",
            "missing_geometry",
            "low_confidence_reprojection_failure",
            "uncaptured_backsides",
            "uncaptured_interiors",
            "low_confidence_hole_fill",
        ],
        "forbidden_changes": [
            "protected_object_placement",
            "protected_geometry_silhouette",
            "protected_texture_or_material_identity",
            "task_critical_fixture_drift_when_reliably_captured",
        ],
    }


def compute_canonical_package_version(
    *,
    scene_memory_manifest: Mapping[str, Any],
    conditioning_bundle: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
    site_world_spec: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    canonical_render_policy: Mapping[str, Any],
    presentation_variance_policy: Mapping[str, Any],
) -> str:
    normalized_spec = dict(site_world_spec)
    normalized_spec.pop("canonical_package_version", None)
    digest = hashlib.sha256()
    for payload in (
        scene_memory_manifest,
        conditioning_bundle,
        object_geometry_manifest,
        task_anchor_manifest,
        normalized_spec,
        protected_regions_manifest,
        canonical_render_policy,
        presentation_variance_policy,
    ):
        digest.update(normalized_json_bytes(payload))
    return digest.hexdigest()
