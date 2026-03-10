"""Task-target inference and task-aware swap candidate selection."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .capture_bridge import CaptureDescriptor
from .common import (
    join_gs_uri,
    read_json_any,
    resolve_gs_uri_to_path,
    try_parse_float,
    utc_now_iso,
    write_json,
)
from .ios_manifest import IOSManifest
from .swap_candidates import build_swap_candidates_payload

_SELECTION_MODES = {"explicit_only", "hybrid", "policy_only"}
_SOURCE_PRIORITY = {
    "descriptor": 3,
    "grounding_backend": 2,
    "video_inference": 2,
    "legacy_object_index": 1,
}

_DEFAULT_CLASS_CAPS: Dict[str, int] = {
    # Empty by default — we rely on spatial deduplication to merge duplicate
    # detections of the same physical object rather than arbitrarily capping
    # classes.  Override via SWAP_PER_CLASS_MAX_COUNTS_JSON env var if needed.
}
_RESIDENTIAL_ENVIRONMENTS = {"bedroom", "kitchen"}
_RESIDENTIAL_DEFAULT_CLASS_CAPS: Dict[str, int] = {
    "door": 4,
    "drawer": 8,
    "cabinet": 8,
    "box": 10,
}

_SEMANTIC_LABEL_BUCKETS = {
    "door": (
        "door",
        "docking_door",
        "rolling_door",
        "pantry_door",
        "cabinet_door",
        "appliance_door",
    ),
    "drawer": ("drawer",),
    "cabinet": ("cabinet", "cupboard", "closet", "locker", "wardrobe"),
    "box": ("box", "package", "parcel", "carton", "container", "crate", "tote", "bin", "shipment"),
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _normalized_text(*parts: Any) -> str:
    return " ".join(str(part).strip().lower() for part in parts if str(part).strip())


def _candidate_id(entry: Mapping[str, Any]) -> str:
    for key in ("instance_id", "id", "object_id", "uuid"):
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _candidate_label(entry: Mapping[str, Any]) -> str:
    for key in ("label", "name", "class_name", "category"):
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "object"


def _semantic_label_bucket(label: Any) -> str:
    text = str(label or "").strip().lower()
    if not text:
        return "object"
    for bucket, tokens in _SEMANTIC_LABEL_BUCKETS.items():
        if any(token in text for token in tokens):
            return bucket
    return text


def _mean_box_area_px(entry: Mapping[str, Any]) -> float:
    raw = entry.get("mean_box_px")
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    if isinstance(raw, Mapping):
        area = _safe_float(raw.get("area"), 0.0)
        if area > 0.0:
            return area
        width = _safe_float(raw.get("width"), 0.0)
        height = _safe_float(raw.get("height"), 0.0)
        return max(0.0, width * height)
    if isinstance(raw, list) and len(raw) >= 2:
        width = _safe_float(raw[0], 0.0)
        height = _safe_float(raw[1], 0.0)
        return max(0.0, width * height)
    return 0.0


def _object_salience_score(entry: Mapping[str, Any]) -> float:
    detections = max(
        _safe_int(entry.get("n_total_detections"), 0),
        _safe_int(entry.get("n_frame_detections"), 0),
    )
    confidence = _safe_float(entry.get("mean_confidence"), _safe_float(entry.get("confidence"), 0.0))
    area = _mean_box_area_px(entry)
    crops = len(entry.get("all_crops")) if isinstance(entry.get("all_crops"), list) else 0

    det_score = _clamp01(detections / 20.0)
    conf_score = _clamp01(confidence)
    area_score = _clamp01(area / 30000.0)
    crop_score = _clamp01(crops / 4.0)
    return (det_score + conf_score + area_score + crop_score) / 4.0


def _entry_center_extents(entry: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
    raw_box = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else None
    if raw_box is None:
        raw_box = entry.get("obb") if isinstance(entry.get("obb"), Mapping) else {}

    center_raw = raw_box.get("center") if isinstance(raw_box.get("center"), list) else [0.0, 0.0, 0.0]
    extents_raw = raw_box.get("extents") if isinstance(raw_box.get("extents"), list) else [0.25, 0.25, 0.25]

    center = [try_parse_float(center_raw[idx] if idx < len(center_raw) else 0.0, 0.0) for idx in range(3)]
    extents = [
        max(0.02, try_parse_float(extents_raw[idx] if idx < len(extents_raw) else 0.25, 0.25))
        for idx in range(3)
    ]
    return center, extents


def _entry_bounds(entry: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
    center, extents = _entry_center_extents(entry)
    half = [max(0.01, value * 0.5) for value in extents]
    mins = [center[idx] - half[idx] for idx in range(3)]
    maxs = [center[idx] + half[idx] for idx in range(3)]
    return mins, maxs


def _obb_iou3d(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    a_min, a_max = _entry_bounds(a)
    b_min, b_max = _entry_bounds(b)
    overlap = [
        max(0.0, min(a_max[idx], b_max[idx]) - max(a_min[idx], b_min[idx]))
        for idx in range(3)
    ]
    inter = overlap[0] * overlap[1] * overlap[2]
    if inter <= 0.0:
        return 0.0
    vol_a = max(1e-9, (a_max[0] - a_min[0]) * (a_max[1] - a_min[1]) * (a_max[2] - a_min[2]))
    vol_b = max(1e-9, (b_max[0] - b_min[0]) * (b_max[1] - b_min[1]) * (b_max[2] - b_min[2]))
    union = max(1e-9, vol_a + vol_b - inter)
    return inter / union


def _center_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ac, _ = _entry_center_extents(a)
    bc, _ = _entry_center_extents(b)
    return math.sqrt(sum((ac[idx] - bc[idx]) ** 2 for idx in range(3)))


def _diag_extent(entry: Mapping[str, Any]) -> float:
    _, extents = _entry_center_extents(entry)
    return math.sqrt(sum(value * value for value in extents))


def _labels_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    label_a = _candidate_label(a).strip().lower()
    label_b = _candidate_label(b).strip().lower()
    if label_a == label_b:
        return True
    return _semantic_label_bucket(label_a) == _semantic_label_bucket(label_b)


def _merge_cluster_entries(cluster_entries: List[Mapping[str, Any]]) -> Dict[str, Any]:
    representative = max(cluster_entries, key=_object_salience_score)
    merged = dict(representative)

    mins_list: List[List[float]] = []
    maxs_list: List[List[float]] = []
    all_crops: List[str] = []
    ids: List[str] = []
    total_detections = 0
    total_frame_detections = 0
    weighted_conf_num = 0.0
    weighted_conf_den = 0.0
    max_conf = 0.0

    for entry in cluster_entries:
        mins, maxs = _entry_bounds(entry)
        mins_list.append(mins)
        maxs_list.append(maxs)

        obj_id = _candidate_id(entry)
        if obj_id and obj_id not in ids:
            ids.append(obj_id)

        ref = entry.get("reference_crop")
        if isinstance(ref, str):
            text = ref.strip()
            if text and text not in all_crops:
                all_crops.append(text)

        raw_crops = entry.get("all_crops")
        if isinstance(raw_crops, list):
            for value in raw_crops:
                text = str(value).strip()
                if text and text not in all_crops:
                    all_crops.append(text)

        detections = max(
            _safe_int(entry.get("n_total_detections"), 0),
            _safe_int(entry.get("n_frame_detections"), 0),
            1,
        )
        total_detections += _safe_int(entry.get("n_total_detections"), 0)
        total_frame_detections += _safe_int(entry.get("n_frame_detections"), 0)

        confidence = _safe_float(
            entry.get("mean_confidence"),
            _safe_float(entry.get("confidence"), 0.0),
        )
        weighted_conf_num += confidence * float(detections)
        weighted_conf_den += float(detections)
        max_conf = max(max_conf, confidence)

    if mins_list and maxs_list:
        global_min = [min(item[idx] for item in mins_list) for idx in range(3)]
        global_max = [max(item[idx] for item in maxs_list) for idx in range(3)]
        merged_center = [(global_min[idx] + global_max[idx]) * 0.5 for idx in range(3)]
        merged_extents = [max(0.02, global_max[idx] - global_min[idx]) for idx in range(3)]

        box_key = "boundingBox" if "boundingBox" in merged else "obb" if "obb" in merged else "boundingBox"
        box = dict(merged.get(box_key)) if isinstance(merged.get(box_key), Mapping) else {}
        box["center"] = merged_center
        box["extents"] = merged_extents
        if not isinstance(box.get("axes"), list) or len(box.get("axes")) < 3:
            box["axes"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if not isinstance(box.get("orientationQuaternion"), list) or len(box.get("orientationQuaternion")) < 4:
            box["orientationQuaternion"] = [1.0, 0.0, 0.0, 0.0]
        merged[box_key] = box

    if all_crops:
        merged["all_crops"] = all_crops
        if not str(merged.get("reference_crop") or "").strip():
            merged["reference_crop"] = all_crops[0]
    if ids:
        merged["merged_object_ids"] = sorted(ids)
        merged["merge_count"] = len(ids)

    if total_detections > 0:
        merged["n_total_detections"] = total_detections
    if total_frame_detections > 0:
        merged["n_frame_detections"] = total_frame_detections
    if weighted_conf_den > 0.0:
        merged["mean_confidence"] = weighted_conf_num / weighted_conf_den
    if max_conf > 0.0:
        merged["confidence"] = max_conf

    return merged


def _dedupe_object_index_entries(
    object_index_entries: List[Mapping[str, Any]],
    *,
    iou_threshold: float,
    center_ratio: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
    center_ratio = max(0.0, float(center_ratio))

    candidates = [dict(entry) for entry in object_index_entries if isinstance(entry, Mapping)]
    ranked = sorted(candidates, key=_object_salience_score, reverse=True)
    clusters: List[Dict[str, Any]] = []

    for entry in ranked:
        placed = False
        for cluster in clusters:
            representative = cluster["representative"]
            if not _labels_match(entry, representative):
                continue

            iou = _obb_iou3d(entry, representative)
            distance = _center_distance(entry, representative)
            distance_threshold = max(0.05, center_ratio * max(0.1, min(_diag_extent(entry), _diag_extent(representative))))

            if iou >= iou_threshold or distance <= distance_threshold:
                cluster["members"].append(entry)
                rep_score = _object_salience_score(representative)
                entry_score = _object_salience_score(entry)
                if entry_score > rep_score:
                    cluster["representative"] = entry
                placed = True
                break

        if not placed:
            clusters.append({"representative": entry, "members": [entry]})

    merged_entries: List[Dict[str, Any]] = []
    merged_clusters = 0
    for cluster in clusters:
        members = cluster["members"]
        if len(members) > 1:
            merged_clusters += 1
        merged_entries.append(_merge_cluster_entries(members))

    merged_entries.sort(key=_object_salience_score, reverse=True)
    report = {
        "original_count": len(candidates),
        "deduped_count": len(merged_entries),
        "merged_clusters": merged_clusters,
        "merged_entries_removed": len(candidates) - len(merged_entries),
        "iou_threshold": iou_threshold,
        "center_ratio": center_ratio,
    }
    return merged_entries, report


def _parse_caps_mapping(raw_caps: Mapping[str, Any]) -> Dict[str, int]:
    parsed: Dict[str, int] = {}
    for key, value in raw_caps.items():
        label = str(key).strip().lower()
        cap = _safe_int(value, 0)
        if label and cap > 0:
            parsed[label] = cap
    return parsed


def _parse_per_class_caps(
    override_caps: Optional[Mapping[str, int]] = None,
) -> Tuple[Dict[str, int], str]:
    if override_caps is not None:
        return _parse_caps_mapping(override_caps), "argument_override"

    json_raw = (os.getenv("SWAP_PER_CLASS_MAX_COUNTS_JSON") or "").strip()
    if json_raw:
        try:
            payload = json.loads(json_raw)
            if isinstance(payload, Mapping):
                parsed = _parse_caps_mapping(payload)
                if parsed:
                    return parsed, "env_json_override"
        except Exception:
            pass

    kv_raw = (os.getenv("SWAP_PER_CLASS_MAX_COUNTS") or "").strip()
    if kv_raw:
        parsed: Dict[str, int] = {}
        for token in kv_raw.split(","):
            text = token.strip()
            if not text:
                continue
            if ":" in text:
                key, value = text.split(":", 1)
            elif "=" in text:
                key, value = text.split("=", 1)
            else:
                continue
            label = key.strip().lower()
            cap = _safe_int(value, 0)
            if label and cap > 0:
                parsed[label] = cap
        if parsed:
            return parsed, "env_kv_override"

    return dict(_DEFAULT_CLASS_CAPS), "default"


def _descriptor_environment_hints(descriptor: CaptureDescriptor) -> List[str]:
    hints: List[str] = []
    for raw in [descriptor.environment_type_hint, *(descriptor.swap_focus or [])]:
        hint = str(raw or "").strip().lower()
        if hint and hint not in hints:
            hints.append(hint)
    return hints


def _resolve_default_class_caps_for_descriptor(
    descriptor: CaptureDescriptor,
) -> Tuple[Dict[str, int], str]:
    hints = _descriptor_environment_hints(descriptor)
    for hint in hints:
        if hint in _RESIDENTIAL_ENVIRONMENTS:
            return dict(_RESIDENTIAL_DEFAULT_CLASS_CAPS), hint
    return dict(_DEFAULT_CLASS_CAPS), hints[0] if hints else "industrial_unknown"


def _resolve_per_class_caps(
    *,
    descriptor: CaptureDescriptor,
    override_caps: Optional[Mapping[str, int]],
    explicit_object_ids: set[str],
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    parsed_caps, source = _parse_per_class_caps(override_caps)
    if source == "default" and not parsed_caps:
        parsed_caps, inferred_env = _resolve_default_class_caps_for_descriptor(descriptor)
        if parsed_caps:
            source = f"environment_default:{inferred_env}"

    diagnostics = {
        "source": source,
        "explicit_bypass_mode": "descriptor_and_external_object_ids_only",
        "explicit_object_id_count": len(explicit_object_ids),
        "environment_hints": _descriptor_environment_hints(descriptor),
    }
    if not parsed_caps:
        diagnostics["reason"] = "no per-class cap override and no environment default caps applied"
    else:
        diagnostics["reason"] = "per-class caps applied before candidate selection"
    return parsed_caps, diagnostics


def _entry_has_explicit_object_id(entry: Mapping[str, Any], explicit_object_ids: set[str]) -> bool:
    if not explicit_object_ids:
        return False
    ids = [str(entry.get("id") or "").strip()]
    merged_ids = entry.get("merged_object_ids")
    if isinstance(merged_ids, list):
        ids.extend(str(value).strip() for value in merged_ids)
    return any(obj_id and obj_id in explicit_object_ids for obj_id in ids)


def _entry_detection_counts(entry: Mapping[str, Any]) -> Tuple[int, int]:
    frame_detections = max(0, _safe_int(entry.get("n_frame_detections"), 0))
    total_detections = max(0, _safe_int(entry.get("n_total_detections"), 0))
    return frame_detections, total_detections


def _apply_detection_support_filter(
    entries: List[Dict[str, Any]],
    *,
    min_frame_detections: int,
    min_total_detections: int,
    explicit_object_ids: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    min_frame_detections = max(0, int(min_frame_detections))
    min_total_detections = max(0, int(min_total_detections))
    enabled = min_frame_detections > 0 or min_total_detections > 0
    if not enabled:
        return entries, {
            "enabled": False,
            "min_frame_detections": min_frame_detections,
            "min_total_detections": min_total_detections,
            "input_count": len(entries),
            "kept_count": len(entries),
            "dropped_count": 0,
            "dropped_low_support_count": 0,
            "explicit_override_kept_count": 0,
        }

    kept: List[Dict[str, Any]] = []
    dropped_low_support_count = 0
    explicit_override_kept_count = 0
    for entry in entries:
        has_frame_key = "n_frame_detections" in entry
        has_total_key = "n_total_detections" in entry
        frame_detections, total_detections = _entry_detection_counts(entry)
        has_detection_counts = has_frame_key or has_total_key
        if not has_detection_counts:
            kept.append(entry)
            continue

        frame_ok = True
        if min_frame_detections > 0 and has_frame_key:
            frame_ok = frame_detections >= min_frame_detections

        total_ok = True
        if min_total_detections > 0 and has_total_key:
            total_ok = total_detections >= min_total_detections

        support_ok = frame_ok and total_ok
        explicit = _entry_has_explicit_object_id(entry, explicit_object_ids)
        if support_ok or explicit:
            kept.append(entry)
            if explicit and not support_ok:
                explicit_override_kept_count += 1
            continue

        dropped_low_support_count += 1

    return kept, {
        "enabled": True,
        "min_frame_detections": min_frame_detections,
        "min_total_detections": min_total_detections,
        "input_count": len(entries),
        "kept_count": len(kept),
        "dropped_count": len(entries) - len(kept),
        "dropped_low_support_count": dropped_low_support_count,
        "explicit_override_kept_count": explicit_override_kept_count,
    }


def _apply_per_class_caps(
    entries: List[Dict[str, Any]],
    *,
    class_caps: Mapping[str, int],
    explicit_object_ids: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not class_caps:
        return entries, {
            "enabled": False,
            "caps": {},
            "input_count": len(entries),
            "kept_count": len(entries),
            "dropped_count": 0,
            "dropped_by_label": {},
        }

    counts: Dict[str, int] = {}
    kept: List[Dict[str, Any]] = []
    dropped_by_label: Dict[str, int] = {}
    kept_by_label: Dict[str, int] = {}
    explicit_bypass_kept_count = 0

    ranked = sorted(
        entries,
        key=lambda item: (
            _entry_has_explicit_object_id(item, explicit_object_ids),
            _object_salience_score(item),
        ),
        reverse=True,
    )
    for entry in ranked:
        label = _semantic_label_bucket(_candidate_label(entry))
        cap = _safe_int(class_caps.get(label), 0)
        explicit = _entry_has_explicit_object_id(entry, explicit_object_ids)

        if cap > 0 and counts.get(label, 0) >= cap and not explicit:
            dropped_by_label[label] = dropped_by_label.get(label, 0) + 1
            continue

        kept.append(entry)
        counts[label] = counts.get(label, 0) + 1
        kept_by_label[label] = kept_by_label.get(label, 0) + 1
        if cap > 0 and explicit and counts.get(label, 0) > cap:
            explicit_bypass_kept_count += 1

    report = {
        "enabled": True,
        "caps": {str(k): int(v) for k, v in class_caps.items()},
        "input_count": len(entries),
        "kept_count": len(kept),
        "dropped_count": len(entries) - len(kept),
        "dropped_by_label": dropped_by_label,
        "kept_by_label": kept_by_label,
        "explicit_bypass_kept_count": explicit_bypass_kept_count,
    }
    return kept, report


def _normalize_hint_entry(
    entry: Mapping[str, Any],
    *,
    source: str,
    role: str,
    default_confidence: float,
) -> Dict[str, Any]:
    instance_id = _candidate_id(entry)
    label = _candidate_label(entry)
    confidence = _safe_float(entry.get("confidence"), default_confidence)
    reason = str(entry.get("reason") or entry.get("rationale") or "").strip()
    return {
        "instance_id": instance_id,
        "label": label,
        "confidence": _clamp01(confidence),
        "source": source,
        "role": role,
        "reason": reason,
        "category": str(entry.get("category") or role),
        "boundingBox": (
            dict(entry.get("boundingBox"))
            if isinstance(entry.get("boundingBox"), Mapping)
            else dict(entry.get("obb"))
            if isinstance(entry.get("obb"), Mapping)
            else None
        ),
    }


def _normalize_hint_list(
    entries: Iterable[Any],
    *,
    source: str,
    role: str,
    default_confidence: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for value in entries:
        if isinstance(value, Mapping):
            out.append(
                _normalize_hint_entry(
                    value,
                    source=source,
                    role=role,
                    default_confidence=default_confidence,
                )
            )
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            out.append(
                {
                    "instance_id": text,
                    "label": "object",
                    "confidence": default_confidence,
                    "source": source,
                    "role": role,
                    "reason": "",
                }
            )
    return out


def _dedupe_hint_entries(entries: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        instance_id = str(entry.get("instance_id") or "").strip()
        label = str(entry.get("label") or "object").strip()
        key = instance_id or f"label:{label.lower()}"
        if not key:
            continue

        existing = by_key.get(key)
        source = str(entry.get("source") or "grounding_backend").strip() or "grounding_backend"
        score = (
            _SOURCE_PRIORITY.get(source, 0),
            _safe_float(entry.get("confidence"), 0.0),
        )
        if existing is None:
            by_key[key] = dict(entry)
            continue

        existing_source = str(existing.get("source") or "grounding_backend").strip() or "grounding_backend"
        existing_score = (
            _SOURCE_PRIORITY.get(existing_source, 0),
            _safe_float(existing.get("confidence"), 0.0),
        )
        if score > existing_score:
            merged = dict(entry)
        else:
            merged = dict(existing)

        reasons: List[str] = []
        for reason in (existing.get("reason"), entry.get("reason")):
            text = str(reason or "").strip()
            if text and text not in reasons:
                reasons.append(text)
        merged["reason"] = "; ".join(reasons)
        by_key[key] = merged

    return sorted(
        (dict(value) for value in by_key.values()),
        key=lambda item: (
            -_safe_float(item.get("confidence"), 0.0),
            str(item.get("instance_id") or item.get("label") or ""),
        ),
    )


def _descriptor_target_entries(
    descriptor: CaptureDescriptor,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manip = _normalize_hint_list(
        descriptor.manipulation_candidates,
        source="descriptor",
        role="manipulation",
        default_confidence=1.0,
    )
    artic = _normalize_hint_list(
        descriptor.articulation_hints,
        source="descriptor",
        role="articulation",
        default_confidence=1.0,
    )
    return manip, artic


def _grounding_objects(
    grounding_payload: Optional[Mapping[str, Any]],
    fallback_entries: List[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    if isinstance(grounding_payload, Mapping):
        grounded = grounding_payload.get("grounded_objects")
        if isinstance(grounded, list):
            return [item for item in grounded if isinstance(item, Mapping)]
    return [item for item in fallback_entries if isinstance(item, Mapping)]


def _grounding_lookup(
    grounding_payload: Optional[Mapping[str, Any]],
    fallback_entries: List[Mapping[str, Any]],
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, List[Mapping[str, Any]]]]:
    by_id: Dict[str, Mapping[str, Any]] = {}
    by_label: Dict[str, List[Mapping[str, Any]]] = {}
    for entry in _grounding_objects(grounding_payload, fallback_entries):
        obj_id = _candidate_id(entry)
        if obj_id and obj_id not in by_id:
            by_id[obj_id] = entry
        label = _semantic_label_bucket(_candidate_label(entry))
        if label:
            by_label.setdefault(label, []).append(entry)
    return by_id, by_label


def _normalize_bbox(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else None
    if raw is None and isinstance(entry.get("obb"), Mapping):
        raw = entry.get("obb")
    if not isinstance(raw, Mapping):
        return None
    center = raw.get("center") if isinstance(raw.get("center"), list) else None
    extents = raw.get("extents") if isinstance(raw.get("extents"), list) else None
    if not isinstance(center, list) or not isinstance(extents, list):
        return None
    axes = raw.get("axes") if isinstance(raw.get("axes"), list) else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    quat = (
        raw.get("orientationQuaternion")
        if isinstance(raw.get("orientationQuaternion"), list)
        else [1.0, 0.0, 0.0, 0.0]
    )
    return {
        "center": [try_parse_float(center[idx] if idx < len(center) else 0.0, 0.0) for idx in range(3)],
        "extents": [max(0.02, try_parse_float(extents[idx] if idx < len(extents) else 0.25, 0.25)) for idx in range(3)],
        "axes": axes,
        "orientationQuaternion": [try_parse_float(quat[idx] if idx < len(quat) else 0.0, 0.0) for idx in range(4)],
    }


def _enrich_hints_with_grounding(
    hints: List[Dict[str, Any]],
    *,
    grounding_payload: Optional[Mapping[str, Any]],
    fallback_entries: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_id, by_label = _grounding_lookup(grounding_payload, fallback_entries)
    enriched: List[Dict[str, Any]] = []
    for hint in hints:
        item = dict(hint)
        obj_id = str(item.get("instance_id") or "").strip()
        label_bucket = _semantic_label_bucket(item.get("label"))

        grounded: Optional[Mapping[str, Any]] = None
        if obj_id:
            grounded = by_id.get(obj_id)
        if grounded is None and label_bucket:
            options = by_label.get(label_bucket, [])
            if len(options) == 1:
                grounded = options[0]

        if grounded is not None:
            grounded_id = _candidate_id(grounded)
            grounded_label = _candidate_label(grounded)
            if grounded_id and not obj_id:
                item["instance_id"] = grounded_id
            if str(item.get("label") or "").strip().lower() in {"", "object", "unknown"} and grounded_label:
                item["label"] = grounded_label
            bbox = _normalize_bbox(grounded)
            if bbox is not None:
                item["boundingBox"] = bbox
            item["confidence"] = max(
                _safe_float(item.get("confidence"), 0.0),
                _safe_float(grounded.get("confidence"), _safe_float(grounded.get("mean_confidence"), 0.0)),
            )
        enriched.append(item)
    return enriched


def _resolve_video_uri_and_path(
    *,
    descriptor: CaptureDescriptor,
    manifest: IOSManifest,
    storage_root: Path,
) -> Tuple[str, Optional[Path]]:
    candidate_uris: List[str] = []
    for uri in (
        descriptor.raw_video_uri,
        manifest.video_uri,
        join_gs_uri(descriptor.raw_prefix_uri, "video.mp4"),
    ):
        text = str(uri or "").strip()
        if text and text not in candidate_uris:
            candidate_uris.append(text)

    for uri in candidate_uris:
        if uri.startswith("gs://"):
            path = resolve_gs_uri_to_path(uri, storage_root)
            if path.exists():
                return uri, path
        else:
            path = Path(uri)
            if path.exists():
                return uri, path
    return candidate_uris[0] if candidate_uris else "", None


def _run_external_video_task_inference(
    *,
    descriptor: CaptureDescriptor,
    object_index_uri: str,
    storage_root: Path,
    video_uri: str,
    video_path: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    command_template = (os.getenv("TASK_INFERENCE_COMMAND") or "").strip()
    if not command_template:
        return {}, {"status": "skipped", "reason": "TASK_INFERENCE_COMMAND not set"}
    if video_path is None:
        return {}, {"status": "skipped", "reason": "video file not found", "video_uri": video_uri}

    object_index_path = resolve_gs_uri_to_path(object_index_uri, storage_root)
    if not object_index_path.exists():
        return {}, {"status": "skipped", "reason": "object index path not found"}

    timeout_seconds = _safe_int(os.getenv("TASK_INFERENCE_TIMEOUT_SECONDS"), 240)
    if timeout_seconds <= 0:
        timeout_seconds = 240

    fd, tmp_name = tempfile.mkstemp(prefix="task_targets_", suffix=".json")
    os.close(fd)
    tmp_output = Path(tmp_name)
    try:
        substitutions = {
            "VIDEO_PATH": str(video_path),
            "VIDEO_URI": video_uri,
            "OBJECT_INDEX_PATH": str(object_index_path),
            "OUTPUT_JSON": str(tmp_output),
            "SCENE_ID": descriptor.scene_id,
            "CAPTURE_ID": descriptor.capture_id,
            "ENV_HINT": str(descriptor.environment_type_hint or ""),
        }
        rendered = command_template
        for key, value in substitutions.items():
            rendered = rendered.replace("{" + key + "}", value)

        try:
            command = shlex.split(rendered)
        except ValueError as exc:
            return {}, {"status": "failed", "reason": f"invalid command template: {exc}"}
        if not command:
            return {}, {"status": "failed", "reason": "empty command"}

        try:
            proc = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {}, {"status": "failed", "reason": f"command timeout after {timeout_seconds}s"}
        except Exception as exc:  # pragma: no cover - subprocess runtime edge
            return {}, {"status": "failed", "reason": f"command execution failed: {exc}"}

        report: Dict[str, Any] = {
            "status": "ok" if proc.returncode == 0 else "failed",
            "return_code": proc.returncode,
            "command": command,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "timeout_seconds": timeout_seconds,
        }
        if proc.returncode != 0:
            return {}, report
        if not tmp_output.exists() or tmp_output.stat().st_size == 0:
            report["status"] = "failed"
            report["reason"] = "output JSON missing"
            return {}, report

        payload = read_json_any(tmp_output)
        if isinstance(payload, Mapping):
            return dict(payload), report
        report["status"] = "failed"
        report["reason"] = f"unexpected output type: {type(payload).__name__}"
        return {}, report
    finally:
        try:
            tmp_output.unlink(missing_ok=True)
        except Exception:
            pass


def infer_task_targets(
    *,
    descriptor: CaptureDescriptor,
    manifest: IOSManifest,
    object_index_entries: List[Mapping[str, Any]],
    object_index_uri: str,
    storage_root: Path,
    grounding_payload: Optional[Mapping[str, Any]] = None,
    max_targets: int = 24,
) -> Dict[str, Any]:
    """Infer task-relevant targets from explicit hints + grounding signals."""

    max_targets = max(1, int(max_targets or 1))
    dedupe_iou = _safe_float(
        os.getenv("SWAP_DEDUPE_IOU_THRESHOLD"),
        0.2,
    )
    dedupe_center_ratio = _safe_float(
        os.getenv("SWAP_DEDUPE_CENTER_RATIO"),
        0.45,
    )
    preprocessed_entries, dedupe_summary = _dedupe_object_index_entries(
        object_index_entries,
        iou_threshold=dedupe_iou,
        center_ratio=dedupe_center_ratio,
    )

    desc_manip, desc_artic = _descriptor_target_entries(descriptor)

    video_uri, video_path = _resolve_video_uri_and_path(
        descriptor=descriptor,
        manifest=manifest,
        storage_root=storage_root,
    )

    external_payload = (
        dict(grounding_payload)
        if isinstance(grounding_payload, Mapping)
        else {}
    )
    external_report = (
        dict(external_payload.get("backend_report"))
        if isinstance(external_payload.get("backend_report"), Mapping)
        else {
            "status": str(external_payload.get("backend_status") or "skipped"),
            "backend": str(external_payload.get("backend") or ""),
            "reason": "no_grounding_backend" if not external_payload else "",
        }
    )
    ext_manip = _normalize_hint_list(
        external_payload.get("manipulation_candidates", []),
        source="grounding_backend",
        role="manipulation",
        default_confidence=0.7,
    )
    ext_artic = _normalize_hint_list(
        external_payload.get("articulation_hints", []),
        source="grounding_backend",
        role="articulation",
        default_confidence=0.75,
    )
    ext_nav = _normalize_hint_list(
        external_payload.get("navigation_hints", []),
        source="grounding_backend",
        role="navigation",
        default_confidence=0.65,
    )

    manip_entries = _enrich_hints_with_grounding(
        _dedupe_hint_entries([*desc_manip, *ext_manip])[:max_targets],
        grounding_payload=grounding_payload,
        fallback_entries=preprocessed_entries,
    )
    artic_entries = _enrich_hints_with_grounding(
        _dedupe_hint_entries([*desc_artic, *ext_artic])[:max_targets],
        grounding_payload=grounding_payload,
        fallback_entries=preprocessed_entries,
    )
    nav_entries = _enrich_hints_with_grounding(
        _dedupe_hint_entries(ext_nav)[:max_targets],
        grounding_payload=grounding_payload,
        fallback_entries=preprocessed_entries,
    )

    target_ids = [
        str(item.get("instance_id"))
        for item in manip_entries
        if str(item.get("instance_id") or "").strip()
    ]
    articulation_ids = [
        str(item.get("instance_id"))
        for item in artic_entries
        if str(item.get("instance_id") or "").strip()
    ]
    explicit_target_ids = sorted(
        {
            str(item.get("instance_id"))
            for item in [*desc_manip, *ext_manip, *desc_artic, *ext_artic]
            if str(item.get("instance_id") or "").strip()
        }
    )
    explicit_articulation_ids = sorted(
        {
            str(item.get("instance_id"))
            for item in [*desc_artic, *ext_artic]
            if str(item.get("instance_id") or "").strip()
        }
    )
    explicit_labels = sorted(
        {
            _semantic_label_bucket(str(item.get("label") or "").strip().lower())
            for item in [*desc_manip, *ext_manip, *desc_artic, *ext_artic]
            if str(item.get("label") or "").strip().lower() not in {"", "object", "unknown"}
        }
    )

    tasks_payload = external_payload.get("tasks")
    tasks: List[Dict[str, Any]]
    if isinstance(tasks_payload, list):
        tasks = [dict(item) for item in tasks_payload if isinstance(item, Mapping)]
    else:
        tasks = []

    if ext_manip or ext_artic or ext_nav or tasks:
        inference_mode = "descriptor+external" if (desc_manip or desc_artic) else "external"
    elif desc_manip or desc_artic:
        inference_mode = "descriptor_only"
    else:
        inference_mode = "empty"

    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "inference_mode": inference_mode,
        "video_analysis": {
            "video_uri": video_uri,
            "video_path": str(video_path) if video_path is not None else "",
            "external_inference": external_report,
        },
        "index_preprocessing": {
            "dedupe": dedupe_summary,
        },
        "manipulation_candidates": manip_entries,
        "articulation_hints": artic_entries,
        "navigation_hints": nav_entries,
        "target_object_ids": sorted(set(target_ids)),
        "articulation_required_ids": sorted(set(articulation_ids)),
        "explicit_target_object_ids": explicit_target_ids,
        "explicit_articulation_required_ids": explicit_articulation_ids,
        "explicit_target_labels": explicit_labels,
        "explicit_articulation_labels": explicit_labels,
        "tasks": tasks,
    }


def _merge_descriptor_with_task_targets(
    descriptor: CaptureDescriptor,
    task_targets: Optional[Mapping[str, Any]],
) -> CaptureDescriptor:
    """Merge task-target entries into the descriptor for candidate selection."""
    if not isinstance(task_targets, Mapping):
        return descriptor

    desc_manip = [dict(item) for item in descriptor.manipulation_candidates]
    desc_artic = [dict(item) for item in descriptor.articulation_hints]

    def _merge(
        base: List[Dict[str, Any]],
        incoming: Iterable[Any],
        *,
        role: str,
    ) -> List[Dict[str, Any]]:
        merged = _dedupe_hint_entries(
            [
                *_normalize_hint_list(base, source="descriptor", role=role, default_confidence=1.0),
                *_normalize_hint_list(
                    incoming,
                    source="grounding_backend",
                    role=role,
                    default_confidence=0.75,
                ),
            ]
        )
        out: List[Dict[str, Any]] = []
        for entry in merged:
            instance_id = str(entry.get("instance_id") or "").strip()
            label = str(entry.get("label") or "").strip()
            payload: Dict[str, Any] = {}
            if instance_id:
                payload["instance_id"] = instance_id
            if label:
                payload["label"] = label
            if payload:
                out.append(payload)
        return out

    manip = _merge(
        desc_manip,
        task_targets.get("manipulation_candidates", []),
        role="manipulation",
    )
    artic = _merge(
        desc_artic,
        task_targets.get("articulation_hints", []),
        role="articulation",
    )

    # Only ingest from explicit ID lists (descriptor/external, not heuristic).
    for obj_id in task_targets.get("explicit_target_object_ids", []):
        text = str(obj_id).strip()
        if text and text not in {str(item.get("instance_id") or "") for item in manip}:
            manip.append({"instance_id": text, "label": "object"})
    for obj_id in task_targets.get("explicit_articulation_required_ids", []):
        text = str(obj_id).strip()
        if text and text not in {str(item.get("instance_id") or "") for item in artic}:
            artic.append({"instance_id": text, "label": "object"})

    return replace(
        descriptor,
        manipulation_candidates=manip,
        articulation_hints=artic,
    )


def _normalize_selection_mode(mode: str) -> str:
    candidate = str(mode or "").strip().lower()
    return candidate if candidate in _SELECTION_MODES else "hybrid"


def _extract_explicit_sets(
    descriptor: CaptureDescriptor,
    task_targets: Optional[Mapping[str, Any]],
) -> Dict[str, set[str]]:
    descriptor_obj_ids: set[str] = set()
    task_obj_ids: set[str] = set()
    descriptor_labels: set[str] = set()
    task_labels: set[str] = set()

    for entry in descriptor.manipulation_candidates + descriptor.articulation_hints:
        if not isinstance(entry, Mapping):
            continue
        obj_id = _candidate_id(entry)
        label = _candidate_label(entry).strip().lower()
        if obj_id:
            descriptor_obj_ids.add(obj_id)
        if label and label not in {"object", "unknown"}:
            descriptor_labels.add(_semantic_label_bucket(label))

    if isinstance(task_targets, Mapping):
        inference_mode = str(task_targets.get("inference_mode") or "").strip().lower()

        for key in ("manipulation_candidates", "articulation_hints"):
            for raw in task_targets.get(key, []):
                if isinstance(raw, Mapping):
                    obj_id = _candidate_id(raw)
                    label = _candidate_label(raw).strip().lower()
                    if obj_id:
                        task_obj_ids.add(obj_id)
                    if label and label not in {"object", "unknown"}:
                        task_labels.add(_semantic_label_bucket(label))
                elif isinstance(raw, str):
                    text = raw.strip()
                    if text:
                        task_obj_ids.add(text)

        explicit_list_keys: List[str] = []
        for key in ("explicit_target_object_ids", "explicit_articulation_required_ids"):
            if isinstance(task_targets.get(key), list):
                explicit_list_keys.append(key)
        if not explicit_list_keys and inference_mode and inference_mode != "heuristic":
            explicit_list_keys = ["target_object_ids", "articulation_required_ids"]

        for key in explicit_list_keys:
            for obj_id in task_targets.get(key, []):
                text = str(obj_id).strip()
                if text:
                    task_obj_ids.add(text)

        for key in ("explicit_target_labels", "explicit_articulation_labels"):
            for raw_label in task_targets.get(key, []):
                label = str(raw_label or "").strip().lower()
                if label and label not in {"object", "unknown"}:
                    task_labels.add(_semantic_label_bucket(label))

    return {
        "descriptor_obj_ids": descriptor_obj_ids,
        "task_obj_ids": task_obj_ids,
        "descriptor_labels": descriptor_labels,
        "task_labels": task_labels,
    }


def _candidate_rank_score(
    candidate: Mapping[str, Any],
    object_entry: Optional[Mapping[str, Any]],
    *,
    explicit: bool,
) -> float:
    score = 0.0
    if explicit:
        score += 2.0
    articulation = candidate.get("articulation") if isinstance(candidate.get("articulation"), Mapping) else {}
    if bool(articulation.get("required", False)):
        score += 1.5

    if object_entry is not None:
        score += _object_salience_score(object_entry) * 1.5
        detections = max(
            _safe_int(object_entry.get("n_total_detections"), 0),
            _safe_int(object_entry.get("n_frame_detections"), 0),
        )
        score += min(1.0, detections / 20.0)
        confidence = _safe_float(
            object_entry.get("mean_confidence"),
            _safe_float(object_entry.get("confidence"), 0.0),
        )
        score += _clamp01(confidence)
    else:
        score += 0.5

    return round(score, 6)


def _apply_ranked_cap(
    candidates: List[Dict[str, Any]],
    *,
    max_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_candidates <= 0:
        summary = {
            "pre_cap_count": len(candidates),
            "selected_count": len(candidates),
            "dropped_count": 0,
            "max_candidates": 0,
            "reserved_articulated_count": sum(
                1
                for item in candidates
                if isinstance(item.get("articulation"), Mapping)
                and bool(item["articulation"].get("required"))
            ),
            "cap_overridden_by_required": False,
        }
        return candidates, summary

    required = [
        item
        for item in candidates
        if isinstance(item.get("articulation"), Mapping)
        and bool(item["articulation"].get("required"))
    ]
    optional = [item for item in candidates if item not in required]

    selected: List[Dict[str, Any]] = list(required)
    remaining = max_candidates - len(selected)
    if remaining > 0:
        selected.extend(optional[:remaining])
    cap_overridden = len(required) > max_candidates

    summary = {
        "pre_cap_count": len(candidates),
        "selected_count": len(selected),
        "dropped_count": len(candidates) - len(selected),
        "max_candidates": max_candidates,
        "reserved_articulated_count": len(required),
        "cap_overridden_by_required": cap_overridden,
    }
    return selected, summary


def build_task_aware_swap_candidates_payload(
    *,
    descriptor: CaptureDescriptor,
    object_index_entries: List[Mapping[str, Any]],
    policy_path: Optional[str] = None,
    task_targets: Optional[Mapping[str, Any]] = None,
    selection_mode: str = "hybrid",
    max_candidates: int = 24,
    per_class_caps: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Build swap candidates using task-aware explicit targets + ranked capping."""

    mode = _normalize_selection_mode(selection_mode)
    max_candidates = int(max_candidates or 0)

    if mode == "policy_only":
        descriptor_for_policy = replace(descriptor, manipulation_candidates=[], articulation_hints=[])
        descriptor_for_selection = descriptor_for_policy
    else:
        descriptor_for_selection = _merge_descriptor_with_task_targets(descriptor, task_targets)

    if mode == "policy_only":
        explicit_sets = {
            "descriptor_obj_ids": set(),
            "task_obj_ids": set(),
            "descriptor_labels": set(),
            "task_labels": set(),
        }
    else:
        explicit_sets = _extract_explicit_sets(descriptor, task_targets)
    descriptor_obj_ids = explicit_sets["descriptor_obj_ids"]
    task_obj_ids = explicit_sets["task_obj_ids"]
    descriptor_labels = explicit_sets["descriptor_labels"]
    task_labels = explicit_sets["task_labels"]
    all_explicit_ids = descriptor_obj_ids | task_obj_ids
    all_explicit_labels = descriptor_labels | task_labels

    dedupe_iou = _safe_float(os.getenv("SWAP_DEDUPE_IOU_THRESHOLD"), 0.2)
    dedupe_center_ratio = _safe_float(os.getenv("SWAP_DEDUPE_CENTER_RATIO"), 0.45)
    deduped_entries, dedupe_summary = _dedupe_object_index_entries(
        object_index_entries,
        iou_threshold=dedupe_iou,
        center_ratio=dedupe_center_ratio,
    )

    min_frame_detections = _safe_int(os.getenv("SWAP_MIN_FRAME_DETECTIONS"), 0)
    min_total_detections = _safe_int(os.getenv("SWAP_MIN_TOTAL_DETECTIONS"), 2)
    supported_entries, support_summary = _apply_detection_support_filter(
        deduped_entries,
        min_frame_detections=min_frame_detections,
        min_total_detections=min_total_detections,
        explicit_object_ids=all_explicit_ids,
    )

    class_caps, class_cap_diagnostics = _resolve_per_class_caps(
        descriptor=descriptor,
        override_caps=per_class_caps,
        explicit_object_ids=all_explicit_ids,
    )
    capped_entries, class_cap_summary = _apply_per_class_caps(
        supported_entries,
        class_caps=class_caps,
        explicit_object_ids=all_explicit_ids,
    )
    class_cap_summary["diagnostics"] = class_cap_diagnostics

    base_payload = build_swap_candidates_payload(
        descriptor=descriptor_for_selection,
        object_index_entries=capped_entries,
        policy_path=policy_path,
    )

    entry_by_id: Dict[str, Mapping[str, Any]] = {}
    for entry in capped_entries:
        if not isinstance(entry, Mapping):
            continue
        obj_id = _candidate_id(entry)
        if obj_id and obj_id not in entry_by_id:
            entry_by_id[obj_id] = entry

    candidates_raw = base_payload.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValueError("invalid swap candidates payload")

    annotated: List[Dict[str, Any]] = []
    for item in candidates_raw:
        if not isinstance(item, Mapping):
            continue
        candidate = dict(item)
        obj_id = str(candidate.get("object_id") or "").strip()
        label_text = _normalized_text(candidate.get("label"))
        label_bucket = _semantic_label_bucket(label_text)
        source_entry = entry_by_id.get(obj_id)

        explicit_by_id = obj_id in all_explicit_ids
        if source_entry is not None and not explicit_by_id:
            explicit_by_id = _entry_has_explicit_object_id(source_entry, all_explicit_ids)
        explicit_by_label = label_bucket in all_explicit_labels
        explicit = explicit_by_id or explicit_by_label

        selected_by = "policy"
        if explicit:
            if obj_id in task_obj_ids or label_bucket in task_labels:
                selected_by = "task_targets"
            elif obj_id in descriptor_obj_ids or label_bucket in descriptor_labels:
                selected_by = "descriptor"
            else:
                selected_by = "explicit"

        score = _candidate_rank_score(candidate, source_entry, explicit=explicit)
        selection_meta = {
            "explicit": explicit,
            "selected_by": selected_by,
            "rank_score": score,
        }
        candidate["selection"] = selection_meta
        annotated.append(candidate)

    if mode == "explicit_only":
        annotated = [item for item in annotated if bool(item.get("selection", {}).get("explicit"))]
    elif mode == "hybrid":
        explicit_candidates = [
            item for item in annotated if bool(item.get("selection", {}).get("explicit"))
        ]
        if explicit_candidates:
            if max_candidates > 0:
                policy_candidates = [
                    item for item in annotated if not bool(item.get("selection", {}).get("explicit"))
                ]
                slots = max(0, max_candidates - len(explicit_candidates))
                annotated = explicit_candidates + policy_candidates[:slots]
            else:
                annotated = explicit_candidates

    annotated.sort(
        key=lambda item: (
            bool(
                isinstance(item.get("articulation"), Mapping)
                and bool(item["articulation"].get("required"))
            ),
            bool(item.get("selection", {}).get("explicit")),
            _safe_float(item.get("selection", {}).get("rank_score"), 0.0),
        ),
        reverse=True,
    )

    selected, cap_summary = _apply_ranked_cap(annotated, max_candidates=max_candidates)
    out = dict(base_payload)
    out["candidates"] = selected
    out["selection_mode"] = mode
    out["max_image3d_candidates"] = max_candidates
    out["selection_summary"] = cap_summary
    out["selection_summary"]["explicit_count"] = sum(
        1 for item in selected if bool(item.get("selection", {}).get("explicit"))
    )
    out["selection_summary"]["policy_backfill_count"] = sum(
        1 for item in selected if str(item.get("selection", {}).get("selected_by")) == "policy"
    )
    out["task_targets_attached"] = bool(task_targets)
    out["task_target_object_ids"] = sorted(all_explicit_ids)
    out["index_preprocessing"] = {
        "dedupe": dedupe_summary,
        "detection_support": support_summary,
        "class_caps": class_cap_summary,
    }
    return out


def write_task_targets(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist task-target payload for pipeline observability."""
    write_json(path, payload)
