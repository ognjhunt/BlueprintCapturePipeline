"""Project registered robot kinematic landmarks into a calibrated camera."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .camera_geometry_validation import validate_camera_calibration
from .common import ensure_dir, write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "robot_skeleton_projection.v1"
TRACE_SCHEMA_VERSION = "robot_skeleton_projection_frame.v1"


def _finite_point(value: Any) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("robot_skeleton_landmark_must_be_finite_xyz")
    return point


def _validate_segments(
    segments: Sequence[Sequence[str]], landmark_ids: set[str]
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, segment in enumerate(segments):
        if len(segment) != 2:
            raise ValueError(f"robot_skeleton_segment_must_have_two_landmarks:{index}")
        start, end = str(segment[0]), str(segment[1])
        if not start or not end or start == end:
            raise ValueError(f"robot_skeleton_segment_invalid:{index}")
        if start not in landmark_ids or end not in landmark_ids:
            raise ValueError(f"robot_skeleton_segment_landmark_missing:{index}")
        normalized.append({"from": start, "to": end})
    return normalized


def build_projected_robot_skeleton_trace(
    *,
    landmark_frames: Sequence[Mapping[str, Sequence[float]]],
    segments: Sequence[Sequence[str]],
    camera_calibration: Mapping[str, Any],
    embodiment: str,
    episode_id: str,
    output_dir: str | Path,
    require_reprojection_error: bool = True,
) -> dict[str, Any]:
    """Create OSCAR-compatible projected landmarks from kinematic state only.

    ``landmark_frames`` must already be in the calibration reference frame and
    uses meters.  No RGB frame, physical future observation, or task outcome is
    consumed by this function.
    """

    if not landmark_frames:
        raise ValueError("robot_skeleton_landmark_frames_missing")
    if not embodiment.strip() or not episode_id.strip():
        raise ValueError("robot_skeleton_identity_missing")
    optical_convention = str(camera_calibration.get("optical_convention") or "").lower()
    if optical_convention not in {"opencv", "x_right_y_down_z_forward"}:
        raise ValueError("camera_optical_convention_must_be_opencv")
    calibration = validate_camera_calibration(
        camera_calibration,
        require_extrinsics=True,
        require_frame_metadata=True,
        require_translation_units=True,
        require_reprojection_error=require_reprojection_error,
    )
    if not calibration["projection_ready"]:
        raise ValueError(
            "camera_calibration_not_projection_ready:"
            + ",".join(str(item) for item in calibration["blockers"])
        )
    intrinsics = calibration["intrinsics"]
    camera_from_reference = np.asarray(
        calibration["camera_from_reference"], dtype=np.float64
    )
    first_ids = {str(key) for key in landmark_frames[0]}
    if not first_ids:
        raise ValueError("robot_skeleton_landmarks_missing")
    normalized_segments = _validate_segments(segments, first_ids)
    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    trace_path = output / "projected_robot_skeleton_trace.jsonl"
    rows: list[dict[str, Any]] = []
    total_projected = 0
    total_out_of_view = 0
    for frame_index, frame in enumerate(landmark_frames):
        frame_ids = {str(key) for key in frame}
        if frame_ids != first_ids:
            raise ValueError(f"robot_skeleton_landmark_identity_drift:{frame_index}")
        landmarks: list[dict[str, Any]] = []
        projected_count = 0
        out_of_view_count = 0
        for landmark_id in sorted(first_ids):
            reference_point = _finite_point(frame[landmark_id])
            homogeneous = np.concatenate((reference_point, np.asarray([1.0])))
            camera_point = (camera_from_reference @ homogeneous)[:3]
            z = float(camera_point[2])
            positive_depth = z > 1e-9
            u = (
                float(intrinsics["fx"] * camera_point[0] / z + intrinsics["cx"])
                if positive_depth
                else None
            )
            v = (
                float(intrinsics["fy"] * camera_point[1] / z + intrinsics["cy"])
                if positive_depth
                else None
            )
            in_view = bool(
                positive_depth
                and u is not None
                and v is not None
                and 0.0 <= u < int(intrinsics["width"])
                and 0.0 <= v < int(intrinsics["height"])
            )
            projected_count += int(in_view)
            out_of_view_count += int(not in_view)
            landmarks.append(
                {
                    "landmark_id": landmark_id,
                    "reference_position_m": reference_point.tolist(),
                    "camera_position_m": camera_point.tolist(),
                    "image_projection": {
                        "available": in_view,
                        "u_px": u,
                        "v_px": v,
                        "positive_depth": positive_depth,
                        "in_image_bounds": in_view,
                    },
                }
            )
        total_projected += projected_count
        total_out_of_view += out_of_view_count
        row: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "episode_id": episode_id,
            "frame_index": frame_index,
            "embodiment": embodiment,
            "reference_frame": calibration["reference_frame"],
            "camera_frame": calibration["camera_frame"],
            "landmarks": landmarks,
            "segments": normalized_segments,
            "projected_landmark_count": projected_count,
            "out_of_view_landmark_count": out_of_view_count,
        }
        row["frame_sha256"] = canonical_sha256(row)
        rows.append(row)
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "episode_id": episode_id,
        "embodiment": embodiment,
        "frame_count": len(rows),
        "landmark_ids": sorted(first_ids),
        "segments": normalized_segments,
        "total_projected_landmarks": total_projected,
        "total_out_of_view_landmarks": total_out_of_view,
        "all_frames_have_projected_landmark": all(
            int(row["projected_landmark_count"]) > 0 for row in rows
        ),
        "camera_calibration": calibration,
        "camera_calibration_sha256": canonical_sha256(dict(camera_calibration)),
        "trace_path": str(trace_path),
        "trace_sha256": file_sha256(trace_path),
        "provenance": {
            "kinematic_landmarks_only": True,
            "physical_future_observation_used": False,
            "task_outcome_accessed": False,
            "generated_wam_frames_used": False,
        },
        "claim_boundary": (
            "camera-aligned intended-motion conditioning only; not world prediction or "
            "physical robot evidence"
        ),
    }
    if not manifest["all_frames_have_projected_landmark"]:
        raise ValueError("robot_skeleton_all_landmarks_out_of_view_in_one_or_more_frames")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output / "projected_robot_skeleton_manifest.json", manifest)
    return manifest


__all__ = ["SCHEMA_VERSION", "build_projected_robot_skeleton_trace"]
