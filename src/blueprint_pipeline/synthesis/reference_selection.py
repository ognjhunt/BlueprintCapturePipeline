"""Shared reference filtering and target/reference decoupling policy."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..common import utc_now_iso

DEFAULT_REFERENCE_SELECTION_POLICY: Dict[str, Any] = {
    "policy_version": "v1",
    "selection_mode": "ranked_decoupled_reference_filtering",
    "target_reference_decoupling_mode": "temporal_gap_with_pose_and_anchor_reranking",
    "min_temporal_gap_sec": 0.5,
    "preferred_temporal_gap_sec": 1.5,
    "max_temporal_window_sec": 12.0,
    "min_frame_index_gap": 2,
    "near_duplicate_pose_distance_m": 0.12,
    "max_pose_distance_m": 3.5,
    "checkpoint_window_sec": 2.0,
    "anchor_density_scale": 2.0,
    "anchor_observation_scale": 3.0,
    "weights": {
        "pose_distance": 0.40,
        "temporal_gap": 0.25,
        "anchor_richness": 0.12,
        "route_anchor_density": 0.08,
        "checkpoint_proximity": 0.07,
        "capture_confidence": 0.04,
        "geometry_grounding": 0.04,
    },
}

LEGACY_REFERENCE_SELECTION_POLICY: Dict[str, Any] = {
    "policy_version": "legacy_v0",
    "selection_mode": "legacy_temporal_nearest",
    "target_reference_decoupling_mode": "none",
}


def build_reference_selection_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    k: int,
    selection_name: str,
    policy: Optional[Mapping[str, Any]] = None,
    max_targets: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_policy = resolve_reference_selection_policy(policy)
    entries: List[Dict[str, Any]] = []
    total_rejected_near_duplicates = 0
    aggregate_rejections: Dict[str, int] = {}

    for target_index, _record in enumerate(records):
        selection = select_references_for_target(
            records=records,
            target_index=target_index,
            k=k,
            policy=resolved_policy,
        )
        total_rejected_near_duplicates += int(selection["rejected_near_duplicate_count"])
        for reason, count in dict(selection["rejected_counts"]).items():
            aggregate_rejections[reason] = aggregate_rejections.get(reason, 0) + int(count)
        if int(selection["selected_count"]) <= 0:
            continue
        entries.append(selection)
        if max_targets is not None and len(entries) >= max_targets:
            break

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "selection_name": selection_name,
        "policy": resolved_policy,
        "record_count": len(records),
        "selected_target_count": len(entries),
        "skipped_target_count": max(0, len(records) - len(entries)),
        "rejected_near_duplicate_count": total_rejected_near_duplicates,
        "aggregate_rejected_counts": aggregate_rejections,
        "entries": entries,
    }


def build_legacy_reference_selection_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    k: int,
    selection_name: str,
    max_targets: Optional[int] = None,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    aggregate_rejections: Dict[str, int] = {}

    for target_index, _record in enumerate(records):
        selection = _legacy_select_references_for_target(
            records=records,
            target_index=target_index,
            k=k,
        )
        for reason, count in dict(selection["rejected_counts"]).items():
            aggregate_rejections[reason] = aggregate_rejections.get(reason, 0) + int(count)
        if int(selection["selected_count"]) <= 0:
            continue
        entries.append(selection)
        if max_targets is not None and len(entries) >= max_targets:
            break

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "selection_name": selection_name,
        "policy": dict(LEGACY_REFERENCE_SELECTION_POLICY),
        "record_count": len(records),
        "selected_target_count": len(entries),
        "skipped_target_count": max(0, len(records) - len(entries)),
        "rejected_near_duplicate_count": 0,
        "aggregate_rejected_counts": aggregate_rejections,
        "entries": entries,
    }


def build_reference_selection_comparison(
    *,
    current_manifest: Mapping[str, Any],
    legacy_manifest: Mapping[str, Any],
    selection_name: str,
) -> Dict[str, Any]:
    current_entries = {
        str(item.get("target_frame_id") or ""): dict(item)
        for item in list(current_manifest.get("entries") or [])
        if str(item.get("target_frame_id") or "").strip()
    }
    legacy_entries = {
        str(item.get("target_frame_id") or ""): dict(item)
        for item in list(legacy_manifest.get("entries") or [])
        if str(item.get("target_frame_id") or "").strip()
    }
    overlapping_target_ids = sorted(set(current_entries) & set(legacy_entries))

    current_primary = [_primary_reference_metrics(current_entries[target_id]) for target_id in overlapping_target_ids]
    legacy_primary = [_primary_reference_metrics(legacy_entries[target_id]) for target_id in overlapping_target_ids]
    changed_primary_reference_count = sum(
        1
        for target_id in overlapping_target_ids
        if current_entries[target_id].get("selected_reference_ids", [None])[0:1]
        != legacy_entries[target_id].get("selected_reference_ids", [None])[0:1]
    )

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "selection_name": selection_name,
        "current_policy": dict(current_manifest.get("policy") or {}),
        "legacy_policy": dict(legacy_manifest.get("policy") or {}),
        "current_selected_target_count": int(current_manifest.get("selected_target_count") or 0),
        "legacy_selected_target_count": int(legacy_manifest.get("selected_target_count") or 0),
        "selected_target_count_delta": int(current_manifest.get("selected_target_count") or 0)
        - int(legacy_manifest.get("selected_target_count") or 0),
        "overlapping_target_count": len(overlapping_target_ids),
        "changed_primary_reference_count": changed_primary_reference_count,
        "current_rejected_near_duplicate_count": int(current_manifest.get("rejected_near_duplicate_count") or 0),
        "legacy_rejected_near_duplicate_count": int(legacy_manifest.get("rejected_near_duplicate_count") or 0),
        "rejected_near_duplicate_delta": int(current_manifest.get("rejected_near_duplicate_count") or 0)
        - int(legacy_manifest.get("rejected_near_duplicate_count") or 0),
        "quality_metrics": {
            "primary_temporal_gap_sec": _metric_delta(current_primary, legacy_primary, "temporal_gap_sec"),
            "primary_pose_distance_m": _metric_delta(current_primary, legacy_primary, "pose_distance_m"),
            "primary_anchor_observation_count": _metric_delta(current_primary, legacy_primary, "anchor_observation_count"),
            "primary_route_anchor_density": _metric_delta(current_primary, legacy_primary, "route_anchor_density"),
            "primary_capture_confidence": _metric_delta(current_primary, legacy_primary, "capture_confidence"),
            "primary_geometry_grounding_quality": _metric_delta(current_primary, legacy_primary, "geometry_grounding_quality"),
        },
    }


def resolve_reference_selection_policy(
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = {
        key: (dict(value) if isinstance(value, Mapping) else value)
        for key, value in DEFAULT_REFERENCE_SELECTION_POLICY.items()
    }
    if not isinstance(policy, Mapping):
        return resolved
    for key, value in policy.items():
        if key == "weights" and isinstance(value, Mapping):
            weights = dict(resolved.get("weights") or {})
            for weight_key, weight_value in value.items():
                weights[str(weight_key)] = float(weight_value)
            resolved["weights"] = weights
            continue
        resolved[str(key)] = value
    return resolved


def select_references_for_target(
    *,
    records: Sequence[Mapping[str, Any]],
    target_index: int,
    k: int,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_policy = resolve_reference_selection_policy(policy)
    if target_index < 0 or target_index >= len(records):
        raise IndexError(f"target_index out of bounds: {target_index}")

    target = records[target_index]
    target_frame_id = str(target.get("frame_id") or "").strip()
    target_frame_uri = str(target.get("frame_uri") or "").strip() or None
    target_time = _optional_float(target.get("t_capture_sec"))
    target_frame_index = _frame_index(target)
    target_pose = _pose_matrix(target)

    scored_candidates: List[Dict[str, Any]] = []
    rejected_counts: Dict[str, int] = {}
    rejected_near_duplicates = 0

    for candidate_index, candidate in enumerate(records):
        rejection_reason = _candidate_rejection_reason(
            target=target,
            target_index=target_index,
            target_time=target_time,
            target_frame_index=target_frame_index,
            target_pose=target_pose,
            candidate=candidate,
            candidate_index=candidate_index,
            policy=resolved_policy,
        )
        if rejection_reason is not None:
            rejected_counts[rejection_reason] = rejected_counts.get(rejection_reason, 0) + 1
            if rejection_reason == "near_duplicate":
                rejected_near_duplicates += 1
            continue
        scored_candidates.append(
            _score_candidate(
                target=target,
                target_time=target_time,
                target_frame_index=target_frame_index,
                target_pose=target_pose,
                candidate=candidate,
                candidate_index=candidate_index,
                policy=resolved_policy,
            )
        )

    scored_candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["anchor_observation_count"]),
            -float(item["temporal_gap_sec"] if item["temporal_gap_sec"] is not None else -1.0),
            str(item["reference_frame_id"] or ""),
        )
    )
    selected = scored_candidates[: max(1, int(k))]

    return {
        "target_index": target_index,
        "target_frame_id": target_frame_id,
        "target_frame_uri": target_frame_uri,
        "target_t_capture_sec": target_time,
        "target_frame_index": target_frame_index,
        "selected_count": len(selected),
        "candidate_count": len(scored_candidates),
        "rejected_counts": rejected_counts,
        "rejected_near_duplicate_count": rejected_near_duplicates,
        "selected_reference_ids": [item["reference_id"] for item in selected if item["reference_id"]],
        "selected_reference_frame_ids": [item["reference_frame_id"] for item in selected if item["reference_frame_id"]],
        "selected_reference_frame_uris": [item["reference_frame_uri"] for item in selected if item["reference_frame_uri"]],
        "selected_references": selected,
        "decoupling": {
            "mode": str(resolved_policy.get("target_reference_decoupling_mode") or "unknown"),
            "min_temporal_gap_sec": float(resolved_policy.get("min_temporal_gap_sec") or 0.0),
            "min_frame_index_gap": int(resolved_policy.get("min_frame_index_gap") or 0),
            "near_duplicate_pose_distance_m": float(resolved_policy.get("near_duplicate_pose_distance_m") or 0.0),
            "max_pose_distance_m": float(resolved_policy.get("max_pose_distance_m") or 0.0),
        },
    }


def _legacy_select_references_for_target(
    *,
    records: Sequence[Mapping[str, Any]],
    target_index: int,
    k: int,
) -> Dict[str, Any]:
    target = records[target_index]
    target_frame_id = str(target.get("frame_id") or "").strip()
    target_frame_uri = str(target.get("frame_uri") or "").strip() or None
    target_time = _optional_float(target.get("t_capture_sec"))
    target_frame_index = _frame_index(target)
    target_pose = _pose_matrix(target)

    rejected_counts: Dict[str, int] = {}
    candidates: List[Dict[str, Any]] = []
    for candidate_index, candidate in enumerate(records):
        if candidate_index == target_index:
            rejected_counts["same_target"] = rejected_counts.get("same_target", 0) + 1
            continue
        frame_uri = str(candidate.get("frame_uri") or "").strip()
        if not frame_uri:
            rejected_counts["missing_frame_uri"] = rejected_counts.get("missing_frame_uri", 0) + 1
            continue
        signal_bundle = _signal_bundle(candidate)
        candidates.append(
            {
                "candidate_index": candidate_index,
                "reference_id": str(candidate.get("reference_id") or "").strip() or None,
                "reference_frame_id": str(candidate.get("frame_id") or "").strip() or None,
                "reference_frame_uri": frame_uri,
                "reference_embedding_uri": str(candidate.get("embedding_uri") or "").strip() or None,
                "score": None,
                "temporal_gap_sec": round(_temporal_gap_sec(target_time, candidate) or 0.0, 4),
                "frame_gap": _frame_gap(target_frame_index, candidate),
                "pose_distance_m": round(_pose_distance_m(target_pose, candidate) or 0.0, 4),
                "anchor_observation_count": signal_bundle["anchor_observation_count"],
                "route_anchor_density": signal_bundle["route_anchor_density"],
                "checkpoint_proximity_sec": signal_bundle["checkpoint_proximity_sec"],
                "capture_confidence": signal_bundle["capture_confidence"],
                "geometry_grounding_quality": signal_bundle["geometry_grounding_quality"],
                "score_breakdown": {"legacy_temporal_rank": True},
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["temporal_gap_sec"] if item["temporal_gap_sec"] is not None else 0.0),
            int(item["frame_gap"] if item["frame_gap"] is not None else 0),
            str(item["reference_frame_id"] or ""),
        )
    )
    selected = candidates[: max(1, int(k))]
    return {
        "target_index": target_index,
        "target_frame_id": target_frame_id,
        "target_frame_uri": target_frame_uri,
        "target_t_capture_sec": target_time,
        "target_frame_index": target_frame_index,
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "rejected_counts": rejected_counts,
        "rejected_near_duplicate_count": 0,
        "selected_reference_ids": [item["reference_id"] for item in selected if item["reference_id"]],
        "selected_reference_frame_ids": [item["reference_frame_id"] for item in selected if item["reference_frame_id"]],
        "selected_reference_frame_uris": [item["reference_frame_uri"] for item in selected if item["reference_frame_uri"]],
        "selected_references": selected,
        "decoupling": {
            "mode": "none",
            "min_temporal_gap_sec": 0.0,
            "min_frame_index_gap": 0,
            "near_duplicate_pose_distance_m": 0.0,
            "max_pose_distance_m": None,
        },
    }


def _candidate_rejection_reason(
    *,
    target: Mapping[str, Any],
    target_index: int,
    target_time: Optional[float],
    target_frame_index: Optional[int],
    target_pose: Optional[np.ndarray],
    candidate: Mapping[str, Any],
    candidate_index: int,
    policy: Mapping[str, Any],
) -> Optional[str]:
    if candidate_index == target_index:
        return "same_target"
    frame_uri = str(candidate.get("frame_uri") or "").strip()
    if not frame_uri:
        return "missing_frame_uri"
    if (
        frame_uri
        and frame_uri == str(target.get("frame_uri") or "").strip()
        or (
            str(candidate.get("frame_id") or "").strip()
            and str(candidate.get("frame_id") or "").strip() == str(target.get("frame_id") or "").strip()
        )
    ):
        return "duplicate_identity"

    temporal_gap = _temporal_gap_sec(target_time, candidate)
    frame_gap = _frame_gap(target_frame_index, candidate)
    pose_distance = _pose_distance_m(target_pose, candidate)

    min_temporal_gap_sec = float(policy.get("min_temporal_gap_sec") or 0.0)
    min_frame_index_gap = int(policy.get("min_frame_index_gap") or 0)
    near_duplicate_pose_distance_m = float(policy.get("near_duplicate_pose_distance_m") or 0.0)
    max_temporal_window_sec = float(policy.get("max_temporal_window_sec") or 0.0)
    max_pose_distance_m = float(policy.get("max_pose_distance_m") or 0.0)

    temporal_too_close = temporal_gap is not None and temporal_gap < min_temporal_gap_sec
    frame_too_close = frame_gap is not None and frame_gap < min_frame_index_gap
    pose_too_close = pose_distance is not None and pose_distance < near_duplicate_pose_distance_m

    if temporal_too_close:
        return "near_duplicate"
    if frame_too_close and (temporal_gap is None or temporal_gap < (min_temporal_gap_sec * 2.0)):
        return "near_duplicate"
    if pose_too_close and (temporal_gap is None or temporal_gap < max(min_temporal_gap_sec * 2.0, 1.0)):
        return "near_duplicate"
    if max_temporal_window_sec > 0 and temporal_gap is not None and temporal_gap > max_temporal_window_sec:
        return "outside_temporal_window"
    if max_pose_distance_m > 0 and pose_distance is not None and pose_distance > max_pose_distance_m:
        return "outside_pose_window"
    return None


def _score_candidate(
    *,
    target: Mapping[str, Any],
    target_time: Optional[float],
    target_frame_index: Optional[int],
    target_pose: Optional[np.ndarray],
    candidate: Mapping[str, Any],
    candidate_index: int,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    weights = dict(policy.get("weights") or {})
    temporal_gap = _temporal_gap_sec(target_time, candidate)
    frame_gap = _frame_gap(target_frame_index, candidate)
    pose_distance = _pose_distance_m(target_pose, candidate)
    signal_bundle = _signal_bundle(candidate)

    checkpoint_window_sec = float(policy.get("checkpoint_window_sec") or 1.0)
    max_pose_distance_m = float(policy.get("max_pose_distance_m") or 1.0)
    preferred_temporal_gap_sec = float(policy.get("preferred_temporal_gap_sec") or 1.0)
    anchor_observation_scale = float(policy.get("anchor_observation_scale") or 1.0)
    anchor_density_scale = float(policy.get("anchor_density_scale") or 1.0)

    pose_score = 0.25 if pose_distance is None else _clamp01(1.0 - (pose_distance / max(max_pose_distance_m, 1e-6)))
    temporal_score = (
        0.25
        if temporal_gap is None
        else _clamp01(temporal_gap / max(preferred_temporal_gap_sec, 1e-6))
    )
    anchor_score = _clamp01(signal_bundle["anchor_observation_count"] / max(anchor_observation_scale, 1e-6))
    route_density_score = _clamp01(signal_bundle["route_anchor_density"] / max(anchor_density_scale, 1e-6))
    checkpoint_score = (
        0.0
        if signal_bundle["checkpoint_proximity_sec"] is None
        else _clamp01(1.0 - (signal_bundle["checkpoint_proximity_sec"] / max(checkpoint_window_sec, 1e-6)))
    )
    capture_confidence_score = _clamp01(signal_bundle["capture_confidence"])
    geometry_grounding_score = _clamp01(signal_bundle["geometry_grounding_quality"])

    score = (
        float(weights.get("pose_distance") or 0.0) * pose_score
        + float(weights.get("temporal_gap") or 0.0) * temporal_score
        + float(weights.get("anchor_richness") or 0.0) * anchor_score
        + float(weights.get("route_anchor_density") or 0.0) * route_density_score
        + float(weights.get("checkpoint_proximity") or 0.0) * checkpoint_score
        + float(weights.get("capture_confidence") or 0.0) * capture_confidence_score
        + float(weights.get("geometry_grounding") or 0.0) * geometry_grounding_score
    )

    return {
        "candidate_index": candidate_index,
        "reference_id": str(candidate.get("reference_id") or "").strip() or None,
        "reference_frame_id": str(candidate.get("frame_id") or "").strip() or None,
        "reference_frame_uri": str(candidate.get("frame_uri") or "").strip() or None,
        "reference_embedding_uri": str(candidate.get("embedding_uri") or "").strip() or None,
        "score": round(score, 6),
        "temporal_gap_sec": round(temporal_gap, 4) if temporal_gap is not None else None,
        "frame_gap": frame_gap,
        "pose_distance_m": round(pose_distance, 4) if pose_distance is not None else None,
        "anchor_observation_count": signal_bundle["anchor_observation_count"],
        "route_anchor_density": signal_bundle["route_anchor_density"],
        "checkpoint_proximity_sec": signal_bundle["checkpoint_proximity_sec"],
        "capture_confidence": signal_bundle["capture_confidence"],
        "geometry_grounding_quality": signal_bundle["geometry_grounding_quality"],
        "score_breakdown": {
            "pose_distance": round(pose_score, 6),
            "temporal_gap": round(temporal_score, 6),
            "anchor_richness": round(anchor_score, 6),
            "route_anchor_density": round(route_density_score, 6),
            "checkpoint_proximity": round(checkpoint_score, 6),
            "capture_confidence": round(capture_confidence_score, 6),
            "geometry_grounding": round(geometry_grounding_score, 6),
        },
    }


def _signal_bundle(record: Mapping[str, Any]) -> Dict[str, Any]:
    retrieval_signals = (
        dict(record.get("retrieval_signals") or {})
        if isinstance(record.get("retrieval_signals"), Mapping)
        else {}
    )
    quality = dict(record.get("quality") or {}) if isinstance(record.get("quality"), Mapping) else {}

    anchor_ids = _anchor_ids(record.get("anchor_observations"))
    anchor_observation_count = int(
        retrieval_signals.get("anchor_observation_count")
        or retrieval_signals.get("anchor_richness")
        or len(anchor_ids)
    )
    route_anchor_density = float(retrieval_signals.get("route_anchor_density") or 0.0)
    checkpoint_proximity_sec = _optional_float(retrieval_signals.get("checkpoint_proximity_sec"))

    capture_confidence = _optional_float(retrieval_signals.get("capture_confidence"))
    if capture_confidence is None:
        pose_confidence = _optional_float(retrieval_signals.get("pose_confidence"))
        if pose_confidence is None:
            pose_confidence = _optional_float(quality.get("pose_confidence"))
        sharpness_score = _optional_float(quality.get("sharpness_score"))
        sharpness_confidence = (
            0.75
            if sharpness_score is None
            else _clamp01(sharpness_score / 120.0)
        )
        capture_confidence = round(
            (
                _clamp01(pose_confidence if pose_confidence is not None else 0.75)
                + sharpness_confidence
                + _world_mapping_confidence(quality.get("world_mapping_status"))
            )
            / 3.0,
            4,
        )

    geometry_grounding_quality = _optional_float(retrieval_signals.get("geometry_grounding_quality"))
    if geometry_grounding_quality is None:
        geometry_available = bool(str(record.get("depth_uri") or "").strip() or str(record.get("confidence_uri") or "").strip())
        geometry_grounding_quality = 1.0 if geometry_available else 0.5

    return {
        "anchor_observation_count": anchor_observation_count,
        "route_anchor_density": round(route_anchor_density, 4),
        "checkpoint_proximity_sec": round(checkpoint_proximity_sec, 4) if checkpoint_proximity_sec is not None else None,
        "capture_confidence": round(float(capture_confidence), 4),
        "geometry_grounding_quality": round(float(geometry_grounding_quality), 4),
    }


def _primary_reference_metrics(entry: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    selected = list(entry.get("selected_references") or [])
    if not selected:
        return {
            "temporal_gap_sec": None,
            "pose_distance_m": None,
            "anchor_observation_count": None,
            "route_anchor_density": None,
            "capture_confidence": None,
            "geometry_grounding_quality": None,
        }
    primary = dict(selected[0])
    return {
        "temporal_gap_sec": _optional_float(primary.get("temporal_gap_sec")),
        "pose_distance_m": _optional_float(primary.get("pose_distance_m")),
        "anchor_observation_count": _optional_float(primary.get("anchor_observation_count")),
        "route_anchor_density": _optional_float(primary.get("route_anchor_density")),
        "capture_confidence": _optional_float(primary.get("capture_confidence")),
        "geometry_grounding_quality": _optional_float(primary.get("geometry_grounding_quality")),
    }


def _metric_delta(
    current_items: Sequence[Mapping[str, Optional[float]]],
    legacy_items: Sequence[Mapping[str, Optional[float]]],
    key: str,
) -> Dict[str, Optional[float]]:
    current_values = [_optional_float(item.get(key)) for item in current_items]
    legacy_values = [_optional_float(item.get(key)) for item in legacy_items]
    current_avg = _mean([value for value in current_values if value is not None])
    legacy_avg = _mean([value for value in legacy_values if value is not None])
    return {
        "current_avg": current_avg,
        "legacy_avg": legacy_avg,
        "delta": (
            round(current_avg - legacy_avg, 4)
            if current_avg is not None and legacy_avg is not None
            else None
        ),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / float(len(values)), 4)


def _anchor_ids(raw_value: Any) -> List[str]:
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for item in raw_value:
        if isinstance(item, Mapping):
            text = str(item.get("anchor_id") or item.get("anchorId") or "").strip()
        else:
            text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _temporal_gap_sec(target_time: Optional[float], candidate: Mapping[str, Any]) -> Optional[float]:
    candidate_time = _optional_float(candidate.get("t_capture_sec"))
    if target_time is None or candidate_time is None:
        return None
    return abs(candidate_time - target_time)


def _frame_gap(target_frame_index: Optional[int], candidate: Mapping[str, Any]) -> Optional[int]:
    candidate_frame_index = _frame_index(candidate)
    if target_frame_index is None or candidate_frame_index is None:
        return None
    return abs(candidate_frame_index - target_frame_index)


def _frame_index(record: Mapping[str, Any]) -> Optional[int]:
    explicit = record.get("frame_index")
    parsed = _optional_int(explicit)
    if parsed is not None:
        return parsed
    frame_id = str(record.get("frame_id") or "").strip()
    match = re.search(r"(\d+)$", frame_id)
    if match is None:
        return None
    return int(match.group(1))


def _pose_distance_m(target_pose: Optional[np.ndarray], candidate: Mapping[str, Any]) -> Optional[float]:
    candidate_pose = _pose_matrix(candidate)
    if target_pose is None or candidate_pose is None:
        return None
    return float(np.linalg.norm(target_pose[:3, 3] - candidate_pose[:3, 3]))


def _pose_matrix(record: Mapping[str, Any]) -> Optional[np.ndarray]:
    raw = record.get("T_world_camera")
    if raw is None:
        return None
    pose = np.array(raw, dtype=np.float32)
    if pose.ndim == 1 and pose.size == 16:
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        return None
    return pose


def _world_mapping_confidence(status: Any) -> float:
    text = str(status or "").strip().lower()
    if text in {"mapped", "extending"}:
        return 1.0
    if text in {"limited", "limited_tracking"}:
        return 0.65
    if text:
        return 0.5
    return 0.75


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
