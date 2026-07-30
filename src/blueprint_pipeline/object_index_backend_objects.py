"""Normalize provider object candidates without upgrading their evidence authority."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence


_DEFAULT_BOX_EXTENTS = [0.45, 0.45, 0.45]
_MIN_BOX_EXTENT = 0.02


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validated_metric_geometry_evidence(
    cluster: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only explicitly calibrated metric observations from a 2D cluster."""

    allowed_methods = {
        "calibrated_depth_ray",
        "multiview_triangulation",
        "validated_provider_metric_reconstruction",
    }
    evidence: List[Dict[str, Any]] = []
    for item in cluster:
        center = item.get("world_center")
        extents = item.get("world_extents")
        details = item.get("metric_geometry_evidence")
        if not (
            isinstance(center, list)
            and len(center) == 3
            and isinstance(extents, list)
            and len(extents) == 3
            and isinstance(details, Mapping)
        ):
            continue
        try:
            center_values = [float(value) for value in center]
            extent_values = [float(value) for value in extents]
            uncertainty_m = float(details.get("translation_uncertainty_m"))
            reprojection_error_px = float(details.get("reprojection_error_px"))
        except (TypeError, ValueError):
            continue
        method = str(details.get("method") or "").strip().lower()
        calibration_ref = str(details.get("camera_calibration_ref") or "").strip()
        finite_values = [
            *center_values,
            *extent_values,
            uncertainty_m,
            reprojection_error_px,
        ]
        if (
            method not in allowed_methods
            or not calibration_ref
            or not all(math.isfinite(value) for value in finite_values)
            or any(value <= 0.0 for value in extent_values)
            or uncertainty_m < 0.0
            or uncertainty_m > 0.25
            or reprojection_error_px < 0.0
            or reprojection_error_px > 5.0
        ):
            continue
        evidence.append(
            {
                "method": method,
                "camera_calibration_ref": calibration_ref,
                "translation_uncertainty_m": uncertainty_m,
                "reprojection_error_px": reprojection_error_px,
                "frame_index": _safe_int(item.get("frame_index"), -1),
                "world_center": center_values,
                "world_extents": extent_values,
            }
        )
    return evidence


def normalize_existing_objects(
    payload: Mapping[str, Any],
    *,
    backend_name: str,
) -> List[Dict[str, Any]]:
    """Normalize backend rows as candidates and independently qualify metric geometry."""

    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        return []
    objects: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or item.get("name") or "").strip()
        if not label:
            continue
        bbox = item.get("boundingBox") if isinstance(item.get("boundingBox"), Mapping) else {}
        center = (
            bbox.get("center")
            if isinstance(bbox.get("center"), list)
            else [float(index), 0.0, 0.0]
        )
        extents = (
            bbox.get("extents")
            if isinstance(bbox.get("extents"), list)
            else list(_DEFAULT_BOX_EXTENTS)
        )
        center_values = [
            _safe_float(center[idx] if idx < len(center) else 0.0, 0.0)
            for idx in range(3)
        ]
        extent_values = [
            max(
                _MIN_BOX_EXTENT,
                _safe_float(extents[idx] if idx < len(extents) else 0.25, 0.25),
            )
            for idx in range(3)
        ]
        metric_geometry_evidence = []
        if backend_name != "splat_analyzer":
            metric_geometry_evidence = validated_metric_geometry_evidence(
                [
                    {
                        "frame_index": -1,
                        "world_center": center_values,
                        "world_extents": extent_values,
                        "metric_geometry_evidence": item.get("metric_geometry_evidence"),
                    }
                ]
            )
        metric_placement_ready = bool(metric_geometry_evidence)
        provenance = (
            dict(item.get("provenance"))
            if isinstance(item.get("provenance"), Mapping)
            else {}
        )
        provenance.update(
            {
                "source_backend": backend_name,
                "canonical_truth": metric_placement_ready,
                "presentation_only": not metric_placement_ready,
                "metric_placement_ready": metric_placement_ready,
                "backend_output_does_not_self_qualify": True,
            }
        )
        objects.append(
            {
                "id": str(item.get("id") or item.get("object_id") or f"obj_{index + 1:04d}"),
                "label": label,
                "boundingBox": {
                    "center": center_values,
                    "extents": extent_values,
                    "axes": (
                        bbox.get("axes")
                        if isinstance(bbox.get("axes"), list)
                        else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    ),
                    "orientationQuaternion": (
                        bbox.get("orientationQuaternion")
                        if isinstance(bbox.get("orientationQuaternion"), list)
                        else [1.0, 0.0, 0.0, 0.0]
                    ),
                    "kind": (
                        "rough_axis_aligned_interaction_volume_candidate"
                        if backend_name == "splat_analyzer"
                        else str(bbox.get("kind") or "provider_box_candidate")
                    ),
                },
                "bounding_box_role": (
                    "canonical_metric_placement"
                    if metric_placement_ready
                    else "visualization_only_rough_interaction_volume"
                ),
                "metric_placement_ready": metric_placement_ready,
                "physics_ready": False,
                "geometry_source_class": (
                    "validated_metric_observation"
                    if metric_placement_ready
                    else "unqualified_provider_box_proxy"
                ),
                "metric_geometry_evidence": metric_geometry_evidence,
                "mean_confidence": _safe_float(
                    item.get("mean_confidence"),
                    _safe_float(item.get("confidence"), 0.0),
                ),
                "n_total_detections": _safe_int(item.get("n_total_detections"), 1),
                "n_frame_detections": _safe_int(item.get("n_frame_detections"), 1),
                "reference_crop": str(item.get("reference_crop") or "").strip(),
                "all_crops": (
                    [str(value).strip() for value in item.get("all_crops", []) if str(value).strip()]
                    if isinstance(item.get("all_crops"), list)
                    else []
                ),
                "task_relevance": (
                    dict(item.get("task_relevance"))
                    if isinstance(item.get("task_relevance"), Mapping)
                    else {}
                ),
                "articulation_hints": (
                    dict(item.get("articulation_hints"))
                    if isinstance(item.get("articulation_hints"), Mapping)
                    else {}
                ),
                "evidence_frames": (
                    list(item.get("evidence_frames"))
                    if isinstance(item.get("evidence_frames"), list)
                    else []
                ),
                "source_prompts": (
                    list(item.get("source_prompts"))
                    if isinstance(item.get("source_prompts"), list)
                    else []
                ),
                "provenance": provenance,
                "mean_box_px": (
                    dict(item.get("mean_box_px"))
                    if isinstance(item.get("mean_box_px"), Mapping)
                    else {}
                ),
            }
        )
    return objects
