"""Preregistered wrist-approach capture for the ADP-009D observation gate.

At the canonical reset pose the wrist camera does not see the approved can: the
can sits 63.8 degrees off the optical axis and projects roughly 925 pixels above
the frame, against a 28.4 degree vertical half field of view.  No small
perturbation reaches it, so wrist observability can only be established by
moving the arm toward the object.

This module owns the parts of that motion that do not require Isaac: the
preregistered end-effector waypoints, the world-to-base frame conversion the
differential IK controller expects, and the visibility gate applied to whatever
frames the motion produced.  Keeping them here makes them testable without a
GPU and keeps the runtime free of unreviewable arithmetic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_digest(value: Mapping[str, Any], *, digest_field: str | None = None) -> str:
    """Self-contained twin of ``decision_evidence_contracts.canonical_digest``.

    This module is copied verbatim into the provider bundle and imported by the
    in-container runtime, so it must not depend on the wider package.  Parity
    with the repository contract is pinned by a test.
    """

    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

APPROACH_CAPTURE_SCHEMA_VERSION = "adp009d_wrist_approach_capture.v1"

CAN_AXIS_XY_M = (3.4681748, -3.3100837)
SUPPORT_HEIGHT_M = 0.5264650138348479

# End-effector standoff heights above the support plane, descending.  They stay
# well clear of the can (observed top at ~0.169 m above support) so the approach
# cannot knock it over before the wrist has seen it.
APPROACH_STANDOFF_HEIGHTS_M = (0.34, 0.28, 0.24)
# Tool pointing straight down, in Isaac Lab (w, x, y, z) order.
APPROACH_TOOL_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
APPROACH_STEPS_PER_WAYPOINT = 40
# Differential IK solves for the whole remaining error each step.  Commanding
# that directly as an absolute joint target lets the arm swing through the
# object: an unclamped run displaced the approved can by 3.42 m and tilted it
# 119 degrees.  Joint targets therefore move at most this far per step.
APPROACH_MAX_JOINT_STEP_RAD = 0.03
# The approach must observe the object, never move it.  Exceeding this aborts.
APPROACH_MAX_OBJECT_DISPLACEMENT_M = 0.01
BLOCKER_APPROACH_DISTURBED_OBJECT = "wrist_approach_disturbed_approved_task_object"
# Frame indices reserved for approach captures, after the 40-frame hold capture.
APPROACH_CAPTURE_FRAME_BASE = 100

BLOCKER_WRIST_NEVER_SAW_OBJECT = "wrist_approach_never_observed_approved_task_object"
BLOCKER_APPROACH_IK_FAILED = "wrist_approach_differential_ik_failed"
MIN_WRIST_OBJECT_PIXELS = 200


class ApproachCaptureError(ValueError):
    """Stable fail-closed approach-capture contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(e) for e in errors if str(e))))
        super().__init__(";".join(self.errors))


def approach_waypoints_world() -> list[dict[str, Any]]:
    """Preregistered end-effector waypoints above the sealed can axis."""

    waypoints: list[dict[str, Any]] = []
    for index, standoff in enumerate(APPROACH_STANDOFF_HEIGHTS_M):
        waypoints.append(
            {
                "waypoint_index": index,
                "position_world_m": [
                    CAN_AXIS_XY_M[0],
                    CAN_AXIS_XY_M[1],
                    SUPPORT_HEIGHT_M + float(standoff),
                ],
                "quaternion_wxyz": list(APPROACH_TOOL_QUAT_WXYZ),
                "standoff_above_support_m": float(standoff),
                "capture_frame_index": APPROACH_CAPTURE_FRAME_BASE + index,
                "steps": APPROACH_STEPS_PER_WAYPOINT,
            }
        )
    return waypoints


def _quat_conjugate(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in quaternion)
    return (w, -x, -y, -z)


def _quat_multiply(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = (float(v) for v in left)
    rw, rx, ry, rz = (float(v) for v in right)
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_rotate(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in quaternion)
    vx, vy, vz = (float(v) for v in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def pose_world_to_base(
    *,
    position_world: Sequence[float],
    quaternion_world_wxyz: Sequence[float],
    base_position_world: Sequence[float],
    base_quaternion_world_wxyz: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Express a world pose in the robot base frame, as the IK controller expects."""

    base_inverse = _quat_conjugate(base_quaternion_world_wxyz)
    delta = [
        float(position_world[index]) - float(base_position_world[index])
        for index in range(3)
    ]
    position_base = list(_quat_rotate(base_inverse, delta))
    quaternion_base = list(_quat_multiply(base_inverse, quaternion_world_wxyz))
    return position_base, quaternion_base


def summarize_wrist_approach_capture(
    *,
    captured_frames: Sequence[Mapping[str, Any]],
    approved_task_object_label: str = "approved_can",
    ik_succeeded: bool = True,
    object_displacement_m: float = 0.0,
    min_object_pixels: int = MIN_WRIST_OBJECT_PIXELS,
) -> dict[str, Any]:
    """Gate wrist observability over the frames the approach actually produced."""

    rows: list[dict[str, Any]] = []
    best = 0
    for frame in captured_frames:
        if str(frame.get("camera_id")) != "wrist_camera":
            continue
        semantic = frame.get("semantic_segmentation") or {}
        labels = (semantic.get("id_to_labels") or {}).get("idToLabels") or {}
        counts = semantic.get("pixel_counts_by_id") or {}
        observed = 0
        for identifier, entry in labels.items():
            label = entry.get("class") if isinstance(entry, Mapping) else entry
            if label == approved_task_object_label:
                observed += int(counts.get(str(identifier), 0) or 0)
        best = max(best, observed)
        rows.append(
            {
                "frame_index": frame.get("frame_index"),
                "approved_task_object_pixel_count": observed,
            }
        )

    blockers: list[str] = []
    if not ik_succeeded:
        blockers.append(BLOCKER_APPROACH_IK_FAILED)
    if best < int(min_object_pixels):
        blockers.append(BLOCKER_WRIST_NEVER_SAW_OBJECT)
    if float(object_displacement_m) > APPROACH_MAX_OBJECT_DISPLACEMENT_M:
        blockers.append(BLOCKER_APPROACH_DISTURBED_OBJECT)

    report: dict[str, Any] = {
        "schema_version": APPROACH_CAPTURE_SCHEMA_VERSION,
        "status": "observed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "differential_ik_succeeded": bool(ik_succeeded),
        "waypoints": approach_waypoints_world(),
        "wrist_frames": rows,
        "max_approved_task_object_pixel_count": best,
        "object_displacement_m": float(object_displacement_m),
        "max_object_displacement_allowed_m": APPROACH_MAX_OBJECT_DISPLACEMENT_M,
        "max_joint_step_rad": APPROACH_MAX_JOINT_STEP_RAD,
        "min_approved_task_object_pixel_count_required": int(min_object_pixels),
        "reset_pose_object_off_axis_deg": 63.8,
        "reset_pose_vertical_half_fov_deg": 28.4,
        "candidate_policy_queried": False,
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


__all__ = [
    "APPROACH_CAPTURE_FRAME_BASE",
    "APPROACH_MAX_JOINT_STEP_RAD",
    "APPROACH_MAX_OBJECT_DISPLACEMENT_M",
    "BLOCKER_APPROACH_DISTURBED_OBJECT",
    "APPROACH_CAPTURE_SCHEMA_VERSION",
    "APPROACH_STANDOFF_HEIGHTS_M",
    "APPROACH_STEPS_PER_WAYPOINT",
    "APPROACH_TOOL_QUAT_WXYZ",
    "ApproachCaptureError",
    "BLOCKER_APPROACH_IK_FAILED",
    "BLOCKER_WRIST_NEVER_SAW_OBJECT",
    "approach_waypoints_world",
    "pose_world_to_base",
    "summarize_wrist_approach_capture",
]
