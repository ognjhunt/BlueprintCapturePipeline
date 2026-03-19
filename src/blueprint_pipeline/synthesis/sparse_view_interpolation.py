"""Sparse-view interpolation support derived from real target/reference poses."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..common import utc_now_iso

DEFAULT_SPARSE_VIEW_INTERPOLATION_POLICY: Dict[str, Any] = {
    "policy_version": "v1",
    "interpolation_mode": "pose_only_sparse_view_interpolation",
    "target_support_mode": "real_target_plus_interpolated_support_views",
    "require_sparse_context": True,
    "min_temporal_gap_sec": 0.4,
    "max_temporal_gap_sec": 8.0,
    "min_pose_gap_m": 0.25,
    "max_pose_gap_m": 2.5,
    "interpolation_fractions": [0.3333, 0.6667],
    "max_interpolated_views_per_target": 2,
    "rotation_mode": "target_rotation_anchor",
}


def build_sparse_view_interpolation_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    selection_entries: Sequence[Mapping[str, Any]],
    trajectory_entries: Sequence[Mapping[str, Any]],
    interpolation_name: str,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_policy = resolve_sparse_view_interpolation_policy(policy)
    record_by_frame_id = {
        str(record.get("frame_id") or "").strip(): dict(record)
        for record in records
        if str(record.get("frame_id") or "").strip()
    }
    trajectory_by_frame_id = {
        str(entry.get("target_frame_id") or "").strip(): dict(entry)
        for entry in trajectory_entries
        if str(entry.get("target_frame_id") or "").strip()
    }

    entries: List[Dict[str, Any]] = []
    interpolated_target_count = 0
    interpolated_view_count = 0
    skipped_sparse_target_count = 0

    for selection in selection_entries:
        target_frame_id = str(selection.get("target_frame_id") or "").strip()
        target_record = record_by_frame_id.get(target_frame_id)
        if target_record is None:
            continue
        trajectory_entry = trajectory_by_frame_id.get(target_frame_id, {})
        entry = _interpolate_sparse_target(
            target_record=target_record,
            selection=selection,
            trajectory_entry=trajectory_entry,
            record_by_frame_id=record_by_frame_id,
            policy=resolved_policy,
        )
        if entry["status"] == "interpolated":
            interpolated_target_count += 1
            interpolated_view_count += int(entry.get("interpolated_view_count") or 0)
        else:
            skipped_sparse_target_count += 1
        entries.append(entry)

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "interpolation_name": interpolation_name,
        "policy": resolved_policy,
        "record_count": len(records),
        "target_count": len(selection_entries),
        "interpolated_target_count": interpolated_target_count,
        "skipped_sparse_target_count": skipped_sparse_target_count,
        "interpolated_view_count": interpolated_view_count,
        "entries": entries,
    }


def resolve_sparse_view_interpolation_policy(
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = dict(DEFAULT_SPARSE_VIEW_INTERPOLATION_POLICY)
    if not isinstance(policy, Mapping):
        return resolved
    for key, value in policy.items():
        resolved[str(key)] = value
    return resolved


def _interpolate_sparse_target(
    *,
    target_record: Mapping[str, Any],
    selection: Mapping[str, Any],
    trajectory_entry: Mapping[str, Any],
    record_by_frame_id: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    target_frame_id = str(target_record.get("frame_id") or "").strip()
    target_pose = _pose_matrix(target_record)
    target_time = _optional_float(target_record.get("t_capture_sec"))
    selected_reference_frame_ids = list(selection.get("selected_reference_frame_ids") or [])
    primary_reference_frame_id = str(selected_reference_frame_ids[0] or "").strip() if selected_reference_frame_ids else ""
    primary_reference = record_by_frame_id.get(primary_reference_frame_id)

    if bool(policy.get("require_sparse_context", True)):
        trajectory_status = str(trajectory_entry.get("status") or "").strip().lower()
        if trajectory_status == "augmented":
            return _skipped_entry(
                target_frame_id=target_frame_id,
                selection=selection,
                reason="local_density_already_sufficient",
            )
        if trajectory_status and trajectory_status != "skipped":
            return _skipped_entry(
                target_frame_id=target_frame_id,
                selection=selection,
                reason="trajectory_context_unavailable",
            )

    if primary_reference is None:
        return _skipped_entry(
            target_frame_id=target_frame_id,
            selection=selection,
            reason="missing_primary_reference",
        )

    reference_pose = _pose_matrix(primary_reference)
    reference_time = _optional_float(primary_reference.get("t_capture_sec"))
    pose_gap = _pose_distance(target_pose, reference_pose)
    temporal_gap = abs(reference_time - target_time) if reference_time is not None and target_time is not None else None

    min_pose_gap_m = float(policy.get("min_pose_gap_m") or 0.25)
    max_pose_gap_m = float(policy.get("max_pose_gap_m") or 2.5)
    min_temporal_gap_sec = float(policy.get("min_temporal_gap_sec") or 0.4)
    max_temporal_gap_sec = float(policy.get("max_temporal_gap_sec") or 8.0)

    if target_pose is None or reference_pose is None or pose_gap is None or temporal_gap is None:
        return _skipped_entry(
            target_frame_id=target_frame_id,
            selection=selection,
            reason="missing_pose_or_time_support",
        )
    if pose_gap < min_pose_gap_m or temporal_gap < min_temporal_gap_sec:
        return _skipped_entry(
            target_frame_id=target_frame_id,
            selection=selection,
            reason="support_gap_too_small",
        )
    if pose_gap > max_pose_gap_m or temporal_gap > max_temporal_gap_sec:
        return _skipped_entry(
            target_frame_id=target_frame_id,
            selection=selection,
            reason="support_gap_too_large",
        )

    fractions = list(policy.get("interpolation_fractions") or [])
    max_views = int(policy.get("max_interpolated_views_per_target") or len(fractions) or 0)
    interpolation_views: List[Dict[str, Any]] = []
    for raw_fraction in fractions[:max_views]:
        try:
            fraction = float(raw_fraction)
        except (TypeError, ValueError):
            continue
        if fraction <= 0.0 or fraction >= 1.0:
            continue
        pose = _interpolated_pose(
            target_pose=target_pose,
            reference_pose=reference_pose,
            fraction=fraction,
        )
        interpolated_time = round(target_time + ((reference_time - target_time) * fraction), 4)
        interpolation_views.append(
            {
                "view_id": f"{target_frame_id}-interp-{len(interpolation_views):02d}",
                "source_mode": "sparse_view_interpolation",
                "interpolation_fraction": round(fraction, 4),
                "t_capture_sec": interpolated_time,
                "relative_to_target_sec": round(interpolated_time - target_time, 4),
                "T_world_camera": pose.tolist(),
                "supporting_frame_ids": [target_frame_id, primary_reference_frame_id],
                "supporting_reference_ids": [
                    str(target_record.get("reference_id") or "").strip() or None,
                    str(primary_reference.get("reference_id") or "").strip() or None,
                ],
                "pose_gap_m": round(pose_gap, 4),
                "temporal_gap_sec": round(temporal_gap, 4),
            }
        )

    if not interpolation_views:
        return _skipped_entry(
            target_frame_id=target_frame_id,
            selection=selection,
            reason="no_interpolation_fractions",
        )

    return {
        "target_frame_id": target_frame_id,
        "target_reference_decoupling_mode": (
            (selection.get("decoupling") or {})
            if isinstance(selection.get("decoupling"), Mapping)
            else {}
        ).get("mode"),
        "interpolation_context_id": f"sparse-interp-{target_frame_id}",
        "status": "interpolated",
        "reason": None,
        "primary_reference_frame_id": primary_reference_frame_id,
        "selected_reference_ids": list(selection.get("selected_reference_ids") or []),
        "selected_reference_frame_ids": list(selection.get("selected_reference_frame_ids") or []),
        "interpolated_view_count": len(interpolation_views),
        "interpolated_view_ids": [item["view_id"] for item in interpolation_views],
        "interpolated_views": interpolation_views,
    }


def _skipped_entry(
    *,
    target_frame_id: str,
    selection: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "target_frame_id": target_frame_id,
        "target_reference_decoupling_mode": (
            (selection.get("decoupling") or {})
            if isinstance(selection.get("decoupling"), Mapping)
            else {}
        ).get("mode"),
        "interpolation_context_id": f"sparse-interp-{target_frame_id or 'unknown'}",
        "status": "skipped",
        "reason": reason,
        "primary_reference_frame_id": (
            list(selection.get("selected_reference_frame_ids") or [None])[0]
            if list(selection.get("selected_reference_frame_ids") or [])
            else None
        ),
        "selected_reference_ids": list(selection.get("selected_reference_ids") or []),
        "selected_reference_frame_ids": list(selection.get("selected_reference_frame_ids") or []),
        "interpolated_view_count": 0,
        "interpolated_view_ids": [],
        "interpolated_views": [],
    }


def _interpolated_pose(
    *,
    target_pose: np.ndarray,
    reference_pose: np.ndarray,
    fraction: float,
) -> np.ndarray:
    pose = np.array(target_pose, dtype=np.float32)
    pose[:3, 3] = target_pose[:3, 3] + ((reference_pose[:3, 3] - target_pose[:3, 3]) * fraction)
    pose[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return pose


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


def _pose_distance(target_pose: Optional[np.ndarray], reference_pose: Optional[np.ndarray]) -> Optional[float]:
    if target_pose is None or reference_pose is None:
        return None
    return float(np.linalg.norm(target_pose[:3, 3] - reference_pose[:3, 3]))


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
