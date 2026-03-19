"""Synthetic trajectory augmentation derived from dense real pose tracks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..common import utc_now_iso

DEFAULT_TRAJECTORY_AUGMENTATION_POLICY: Dict[str, Any] = {
    "policy_version": "v1",
    "augmentation_mode": "pose_only_local_trajectory_augmentation",
    "target_support_mode": "real_target_with_synthetic_waypoints",
    "min_observed_context_frames": 3,
    "context_radius_frames": 2,
    "max_observed_gap_sec": 1.25,
    "max_observed_gap_m": 0.85,
    "min_midpoint_gap_sec": 0.3,
    "min_midpoint_gap_m": 0.2,
    "max_synthetic_waypoints_per_target": 3,
    "rotation_mode": "anchor_target_rotation",
}


def build_synthetic_trajectory_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    selection_entries: Sequence[Mapping[str, Any]],
    augmentation_name: str,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_policy = resolve_trajectory_augmentation_policy(policy)
    entries: List[Dict[str, Any]] = []
    skipped_sparse_context_count = 0
    synthetic_waypoint_count = 0

    sorted_records = sorted(
        [dict(record) for record in records],
        key=lambda item: (
            _optional_float(item.get("t_capture_sec")) if _optional_float(item.get("t_capture_sec")) is not None else math.inf,
            _frame_index(item) if _frame_index(item) is not None else math.inf,
        ),
    )

    for selection in selection_entries:
        target_frame_id = str(selection.get("target_frame_id") or "").strip()
        target_index = _find_sorted_index(sorted_records, target_frame_id, int(selection.get("target_index") or -1))
        if target_index is None:
            continue
        entry = _augment_target(
            sorted_records=sorted_records,
            target_sorted_index=target_index,
            selection=selection,
            policy=resolved_policy,
        )
        if entry["status"] != "augmented":
            skipped_sparse_context_count += 1
        synthetic_waypoint_count += int(entry.get("synthetic_waypoint_count") or 0)
        entries.append(entry)

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "augmentation_name": augmentation_name,
        "policy": resolved_policy,
        "record_count": len(records),
        "target_count": len(selection_entries),
        "augmented_target_count": sum(1 for entry in entries if entry.get("status") == "augmented"),
        "skipped_sparse_context_count": skipped_sparse_context_count,
        "synthetic_waypoint_count": synthetic_waypoint_count,
        "entries": entries,
    }


def resolve_trajectory_augmentation_policy(
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = dict(DEFAULT_TRAJECTORY_AUGMENTATION_POLICY)
    if not isinstance(policy, Mapping):
        return resolved
    for key, value in policy.items():
        resolved[str(key)] = value
    return resolved


def _augment_target(
    *,
    sorted_records: Sequence[Mapping[str, Any]],
    target_sorted_index: int,
    selection: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    target = sorted_records[target_sorted_index]
    context_radius_frames = int(policy.get("context_radius_frames") or 2)
    min_observed_context_frames = int(policy.get("min_observed_context_frames") or 3)
    max_observed_gap_sec = float(policy.get("max_observed_gap_sec") or 1.25)
    max_observed_gap_m = float(policy.get("max_observed_gap_m") or 0.85)
    min_midpoint_gap_sec = float(policy.get("min_midpoint_gap_sec") or 0.3)
    min_midpoint_gap_m = float(policy.get("min_midpoint_gap_m") or 0.2)
    max_synthetic_waypoints_per_target = int(policy.get("max_synthetic_waypoints_per_target") or 3)

    start = max(0, target_sorted_index - context_radius_frames)
    end = min(len(sorted_records), target_sorted_index + context_radius_frames + 1)
    observed_records = [dict(item) for item in sorted_records[start:end]]
    observed_frame_ids = [str(item.get("frame_id") or "").strip() for item in observed_records if str(item.get("frame_id") or "").strip()]
    target_frame_id = str(target.get("frame_id") or "").strip()
    target_time = _optional_float(target.get("t_capture_sec"))

    if len(observed_records) < min_observed_context_frames:
        return _skipped_entry(
            target=target,
            selection=selection,
            observed_frame_ids=observed_frame_ids,
            reason="insufficient_context_frames",
        )

    if _has_sparse_gaps(observed_records, max_gap_sec=max_observed_gap_sec, max_gap_m=max_observed_gap_m):
        return _skipped_entry(
            target=target,
            selection=selection,
            observed_frame_ids=observed_frame_ids,
            reason="insufficient_context_density",
        )

    synthetic_waypoints: List[Dict[str, Any]] = []
    for left, right in zip(observed_records, observed_records[1:]):
        if len(synthetic_waypoints) >= max_synthetic_waypoints_per_target:
            break
        left_time = _optional_float(left.get("t_capture_sec"))
        right_time = _optional_float(right.get("t_capture_sec"))
        left_pose = _pose_matrix(left)
        right_pose = _pose_matrix(right)
        pose_gap = _pose_distance(left_pose, right_pose)
        time_gap = (
            abs(right_time - left_time)
            if left_time is not None and right_time is not None
            else None
        )
        if left_pose is None or right_pose is None or time_gap is None or pose_gap is None:
            continue
        if time_gap < min_midpoint_gap_sec and pose_gap < min_midpoint_gap_m:
            continue
        midpoint_time = round((left_time + right_time) / 2.0, 4)
        midpoint_pose = _interpolate_pose(left_pose, right_pose)
        if midpoint_pose is None:
            continue
        synthetic_waypoints.append(
            {
                "waypoint_id": f"{target_frame_id or 'target'}-synthetic-{len(synthetic_waypoints):02d}",
                "source_mode": "synthetic_trajectory_interp",
                "t_capture_sec": midpoint_time,
                "relative_to_target_sec": round(midpoint_time - (target_time or midpoint_time), 4),
                "T_world_camera": midpoint_pose.tolist(),
                "supporting_frame_ids": [
                    str(left.get("frame_id") or "").strip() or None,
                    str(right.get("frame_id") or "").strip() or None,
                ],
                "supporting_reference_ids": [
                    str(left.get("reference_id") or "").strip() or None,
                    str(right.get("reference_id") or "").strip() or None,
                ],
                "pose_gap_m": round(pose_gap, 4),
                "time_gap_sec": round(time_gap, 4),
            }
        )

    if not synthetic_waypoints:
        return _skipped_entry(
            target=target,
            selection=selection,
            observed_frame_ids=observed_frame_ids,
            reason="no_augmented_midpoints",
        )

    return {
        "target_frame_id": target_frame_id,
        "target_reference_decoupling_mode": (
            (selection.get("decoupling") or {})
            if isinstance(selection.get("decoupling"), Mapping)
            else {}
        ).get("mode"),
        "trajectory_context_id": f"trajectory-{target_frame_id or target_sorted_index}",
        "status": "augmented",
        "reason": None,
        "observed_context_frame_ids": observed_frame_ids,
        "observed_context_count": len(observed_frame_ids),
        "synthetic_waypoint_count": len(synthetic_waypoints),
        "synthetic_waypoint_ids": [item["waypoint_id"] for item in synthetic_waypoints],
        "selected_reference_ids": list(selection.get("selected_reference_ids") or []),
        "selected_reference_frame_ids": list(selection.get("selected_reference_frame_ids") or []),
        "synthetic_waypoints": synthetic_waypoints,
    }


def _skipped_entry(
    *,
    target: Mapping[str, Any],
    selection: Mapping[str, Any],
    observed_frame_ids: Sequence[str],
    reason: str,
) -> Dict[str, Any]:
    target_frame_id = str(target.get("frame_id") or "").strip()
    return {
        "target_frame_id": target_frame_id,
        "target_reference_decoupling_mode": (
            (selection.get("decoupling") or {})
            if isinstance(selection.get("decoupling"), Mapping)
            else {}
        ).get("mode"),
        "trajectory_context_id": f"trajectory-{target_frame_id or 'unknown'}",
        "status": "skipped",
        "reason": reason,
        "observed_context_frame_ids": list(observed_frame_ids),
        "observed_context_count": len(observed_frame_ids),
        "synthetic_waypoint_count": 0,
        "synthetic_waypoint_ids": [],
        "selected_reference_ids": list(selection.get("selected_reference_ids") or []),
        "selected_reference_frame_ids": list(selection.get("selected_reference_frame_ids") or []),
        "synthetic_waypoints": [],
    }


def _find_sorted_index(
    sorted_records: Sequence[Mapping[str, Any]],
    target_frame_id: str,
    fallback_index: int,
) -> Optional[int]:
    for index, record in enumerate(sorted_records):
        if str(record.get("frame_id") or "").strip() == target_frame_id:
            return index
    if 0 <= fallback_index < len(sorted_records):
        return fallback_index
    return None


def _has_sparse_gaps(
    observed_records: Sequence[Mapping[str, Any]],
    *,
    max_gap_sec: float,
    max_gap_m: float,
) -> bool:
    for left, right in zip(observed_records, observed_records[1:]):
        left_time = _optional_float(left.get("t_capture_sec"))
        right_time = _optional_float(right.get("t_capture_sec"))
        time_gap = abs(right_time - left_time) if left_time is not None and right_time is not None else None
        pose_gap = _pose_distance(_pose_matrix(left), _pose_matrix(right))
        if time_gap is None or pose_gap is None:
            return True
        if time_gap > max_gap_sec or pose_gap > max_gap_m:
            return True
    return False


def _interpolate_pose(left_pose: np.ndarray, right_pose: np.ndarray) -> Optional[np.ndarray]:
    if left_pose.shape != (4, 4) or right_pose.shape != (4, 4):
        return None
    midpoint = np.array(left_pose, dtype=np.float32)
    midpoint[:3, 3] = (left_pose[:3, 3] + right_pose[:3, 3]) / 2.0
    midpoint[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return midpoint


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


def _pose_distance(left_pose: Optional[np.ndarray], right_pose: Optional[np.ndarray]) -> Optional[float]:
    if left_pose is None or right_pose is None:
        return None
    return float(np.linalg.norm(left_pose[:3, 3] - right_pose[:3, 3]))


def _frame_index(record: Mapping[str, Any]) -> Optional[int]:
    value = record.get("frame_index")
    if value is not None and value != "":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    frame_id = str(record.get("frame_id") or "").strip()
    digits = "".join(ch for ch in frame_id if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
