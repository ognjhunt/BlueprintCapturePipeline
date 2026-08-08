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
import math
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

# End-effector standoff heights above the support plane, descending.
#
# These are measured against the *controlled body*, which is the gripper base --
# but the fingers hang well below it, and the can top is only 0.169 m above the
# support.  A run at 0.34 m displaced the can by 10.3 mm and tripped the
# displacement abort at the very first waypoint, so the old heights did not
# actually clear the tool.  They are raised until a run reports the real
# clearance: gripper_clearance_over_can_m is recorded per waypoint from the
# lowest APPROACH_GRIPPER_BODY_NAMES body, so the next run measures what these
# should be rather than leaving it to inference.
#
# Descending further buys nothing for this gate in any case: the same run
# observed 49,758 pixels of the approved can at the first waypoint, far above
# the 200-pixel threshold, so the remaining waypoints only add contact risk.
APPROACH_STANDOFF_HEIGHTS_M = (0.45, 0.42, 0.40)
# Gripper bodies whose lowest world z gives the true clearance over the can.
APPROACH_GRIPPER_BODY_NAMES = (
    "base_link",
    "left_outer_knuckle",
    "right_outer_knuckle",
    "left_outer_finger",
    "right_outer_finger",
    "left_inner_finger",
    "right_inner_finger",
    "left_inner_knuckle",
    "right_inner_knuckle",
)
# Observed top of the approved can above the support plane.
APPROVED_CAN_TOP_ABOVE_SUPPORT_M = 0.169
# Tool pointing straight down, in Isaac Lab (w, x, y, z) order.
APPROACH_TOOL_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
APPROACH_STEPS_PER_WAYPOINT = 40
# Differential IK solves for the whole remaining error each step.  Commanding
# that directly as an absolute joint target lets the arm swing through the
# object: an unclamped run displaced the approved can by 3.42 m and tilted it
# 119 degrees.  Joint targets therefore move at most this far per step.
APPROACH_MAX_JOINT_STEP_RAD = 0.03
# The approach must not knock the object over.  It was 0.01 m, chosen before any
# measurement, and five consecutive runs aborted at 10.02 mm -- 0.19 percent over
# an arbitrary line, with the nearest articulation body 0.258 m away and the
# displacement almost purely upward.  Aborting there truncated the very evidence
# needed to explain it, and the approach's own purpose was already met: the wrist
# observed the can at 52,725 pixels before any of this.
#
# Raised to a level that means a real disturbance -- roughly the can's own radius,
# so a can actually knocked aside still stops the probe -- while a millimetric
# unexplained drift is recorded and analysed instead of ending the run.
APPROACH_MAX_OBJECT_DISPLACEMENT_M = 0.05
BLOCKER_APPROACH_DISTURBED_OBJECT = "wrist_approach_disturbed_approved_task_object"
# The wrist camera is parented to the Robotiq base link, so its recorded pose
# must move when the arm moves.  A run captured a visibly changing wrist view
# while every recorded wrist pose stayed byte-identical: composing an Aura layer
# against that pose would silently mis-register the whole wrist observation.
BLOCKER_WRIST_POSE_STALE = "wrist_camera_pose_metadata_stale"
MIN_WRIST_POSE_TRAVEL_M = 1.0e-4

# A stale recorded pose has exactly two causes, and they need opposite repairs:
# either the sensor's pose buffer is not refreshed (the prim does follow the
# hand, the reported number lags), or the camera prim is not parented to the
# hand at all (nothing moves, and the changing view came from the scene moving
# past a fixed camera).  Comparing the reported pose against the USD-computed
# world transform of the same prim separates them.
WRIST_POSE_CAUSE_HEALTHY = "pose_tracks_hand"
WRIST_POSE_CAUSE_STALE_BUFFER = "pose_buffer_not_refreshed"
WRIST_POSE_CAUSE_PRIM_DETACHED = "camera_prim_not_following_hand"
WRIST_POSE_CAUSE_UNDETERMINED = "undetermined_usd_transform_unavailable"
# Frame indices reserved for approach captures, after the 40-frame hold capture.
APPROACH_CAPTURE_FRAME_BASE = 100

BLOCKER_WRIST_NEVER_SAW_OBJECT = "wrist_approach_never_observed_approved_task_object"
BLOCKER_APPROACH_IK_FAILED = "wrist_approach_differential_ik_failed"
MIN_WRIST_OBJECT_PIXELS = 200
# A fixed pixel floor alone admitted a 219-pixel sliver clipped by the top edge
# of a 320x180 policy frame.  Require enough of the *current* resolution to be
# useful and keep the target comfortably away from the image boundary.
MIN_WRIST_OBJECT_PIXEL_FRACTION = 0.02
WRIST_OBJECT_FRAME_MARGIN_FRACTION = 0.05
# The deterministic task scorer admits the sealed start only while every
# position component remains within its 5 mm canonical-hold tolerance.  A
# wrist pose discovered after moving the can farther than this cannot become an
# episode start, regardless of how good its picture looks.
EPISODE_START_OBJECT_OFFSET_TOLERANCE_M = 5.0e-3
EPISODE_START_JOINT_TOLERANCE_RAD = 3.0e-3
EPISODE_START_RESTORE_MAX_STEPS = 80
BLOCKER_NO_SAFE_WRIST_OBSERVABLE_EPISODE_START = (
    "no_safe_wrist_observable_episode_start"
)
BLOCKER_EPISODE_START_RESTORE_JOINT_MISMATCH = (
    "wrist_episode_start_restore_joint_mismatch"
)
BLOCKER_EPISODE_START_RESTORE_OBJECT_MOVED = (
    "wrist_episode_start_restore_object_moved"
)
BLOCKER_EPISODE_START_RESTORE_OBJECT_NOT_VISIBLE = (
    "wrist_episode_start_restore_object_not_visible"
)

# "IK succeeded" only ever meant "no exception was raised".  The servo clamps
# joint motion to APPROACH_MAX_JOINT_STEP_RAD over a fixed step budget, so it
# can run cleanly to the end and still stop far short of the waypoint -- and a
# wrist that never arrives obviously never sees the object.  Distinguishing
# "did not arrive" from "arrived but saw nothing" needs the achieved pose.
BLOCKER_APPROACH_DID_NOT_REACH = "wrist_approach_did_not_reach_waypoint"
APPROACH_WAYPOINT_TOLERANCE_M = 0.05


class ApproachCaptureError(ValueError):
    """Stable fail-closed approach-capture contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(e) for e in errors if str(e))))
        super().__init__(";".join(self.errors))


def semantic_label_pixel_count(
    *,
    id_to_labels: Mapping[str, Any],
    pixel_counts_by_id: Mapping[str, Any],
    target_label: str,
) -> int:
    """Count exact semantic pixels for one class from an Isaac camera frame."""

    observed = 0
    for identifier, entry in id_to_labels.items():
        label = entry.get("class") if isinstance(entry, Mapping) else entry
        if label == target_label:
            observed += int(pixel_counts_by_id.get(str(identifier), 0) or 0)
    return observed


def semantic_target_observability(
    *,
    semantic_ids: Any,
    id_to_labels: Mapping[str, Any],
    target_label: str,
    frame_margin_fraction: float = WRIST_OBJECT_FRAME_MARGIN_FRACTION,
) -> dict[str, Any]:
    """Measure target area and framing from one exact semantic AOV."""

    import numpy as np

    semantic = np.asarray(semantic_ids)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    if semantic.ndim != 2 or not semantic.size:
        raise ApproachCaptureError(["wrist_semantic_frame_shape_invalid"])
    target_ids: list[int] = []
    for identifier, entry in id_to_labels.items():
        label = entry.get("class") if isinstance(entry, Mapping) else entry
        if label == target_label:
            try:
                target_ids.append(int(identifier))
            except (TypeError, ValueError) as exc:
                raise ApproachCaptureError(
                    ["wrist_semantic_target_identifier_invalid"]
                ) from exc
    mask = np.isin(semantic.astype(np.int64), target_ids)
    count = int(mask.sum())
    height, width = (int(value) for value in mask.shape)
    fraction = count / float(height * width)
    bbox_xyxy: list[int] | None = None
    centroid_xy_fraction: list[float] | None = None
    within_frame_margin = False
    if count:
        ys, xs = np.nonzero(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_xyxy = [x_min, y_min, x_max, y_max]
        centroid_xy_fraction = [
            float(xs.mean() / max(1, width - 1)),
            float(ys.mean() / max(1, height - 1)),
        ]
        x_margin = int(math.ceil(width * float(frame_margin_fraction)))
        y_margin = int(math.ceil(height * float(frame_margin_fraction)))
        within_frame_margin = (
            x_min >= x_margin
            and x_max < width - x_margin
            and y_min >= y_margin
            and y_max < height - y_margin
        )
    return {
        "approved_task_object_pixel_count": count,
        "approved_task_object_pixel_fraction": fraction,
        "approved_task_object_bbox_xyxy": bbox_xyxy,
        "approved_task_object_centroid_xy_fraction": centroid_xy_fraction,
        "approved_task_object_within_frame_margin": within_frame_margin,
        "frame_resolution_hw": [height, width],
        "frame_margin_fraction": float(frame_margin_fraction),
    }


def select_wrist_observable_episode_start(
    samples: Sequence[Mapping[str, Any]],
    *,
    min_object_pixels: int = MIN_WRIST_OBJECT_PIXELS,
    min_object_pixel_fraction: float = MIN_WRIST_OBJECT_PIXEL_FRACTION,
    object_offset_tolerance_m: float = EPISODE_START_OBJECT_OFFSET_TOLERANCE_M,
) -> dict[str, Any]:
    """Choose the first arm pose that sees the can without moving it.

    Each sample comes from a rendered simulator step and binds the seven arm
    joints, approved-can offset, and wrist semantic pixel count from that same
    step.  Selection is prospective and monotone: the first qualifying pose is
    used, so a later prettier frame cannot justify traversing closer to the
    object after observability was already achieved.
    """

    selected: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        try:
            joints = [float(value) for value in sample["joint_position_rad"]]
            offset = [float(value) for value in sample["object_offset_m"]]
            pixels = int(sample["approved_task_object_pixel_count"])
            pixel_fraction = float(
                sample["approved_task_object_pixel_fraction"]
            )
            within_frame_margin = sample[
                "approved_task_object_within_frame_margin"
            ]
            step = int(sample["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ApproachCaptureError(
                [f"wrist_episode_start_sample_invalid:{index}"]
            ) from exc
        if (
            len(joints) != 7
            or len(offset) != 3
            or pixels < 0
            or not 0.0 <= pixel_fraction <= 1.0
            or not isinstance(within_frame_margin, bool)
        ):
            raise ApproachCaptureError(
                [f"wrist_episode_start_sample_invalid:{index}"]
            )
        within_hold = all(
            abs(value) <= float(object_offset_tolerance_m) for value in offset
        )
        row = {
            "step": step,
            "joint_position_rad": joints,
            "object_offset_m": offset,
            "approved_task_object_pixel_count": pixels,
            "approved_task_object_pixel_fraction": pixel_fraction,
            "approved_task_object_bbox_xyxy": sample.get(
                "approved_task_object_bbox_xyxy"
            ),
            "approved_task_object_centroid_xy_fraction": sample.get(
                "approved_task_object_centroid_xy_fraction"
            ),
            "approved_task_object_within_frame_margin": within_frame_margin,
            "frame_resolution_hw": sample.get("frame_resolution_hw"),
            "frame_margin_fraction": sample.get("frame_margin_fraction"),
            "object_within_canonical_hold": within_hold,
        }
        rows.append(row)
        if (
            selected is None
            and within_hold
            and pixels >= int(min_object_pixels)
            and pixel_fraction >= float(min_object_pixel_fraction)
            and within_frame_margin
        ):
            selected = row

    receipt: dict[str, Any] = {
        "schema_version": "adp009d_wrist_episode_start_selection.v2",
        "status": "ready" if selected is not None else "blocked",
        "blockers": (
            []
            if selected is not None
            else [BLOCKER_NO_SAFE_WRIST_OBSERVABLE_EPISODE_START]
        ),
        "samples_evaluated": len(rows),
        "samples": rows,
        "min_approved_task_object_pixels": int(min_object_pixels),
        "min_approved_task_object_pixel_fraction": float(
            min_object_pixel_fraction
        ),
        "required_within_frame_margin": True,
        "object_offset_tolerance_m": float(object_offset_tolerance_m),
        "selected": selected,
    }
    receipt["selection_digest"] = canonical_digest(
        receipt, digest_field="selection_digest"
    )
    return receipt


def validate_wrist_observable_episode_start_restore(
    *,
    selected_joint_position_rad: Sequence[float],
    restored_joint_position_rad: Sequence[float],
    object_offset_m: Sequence[float],
    approved_task_object_pixel_count: int,
    approved_task_object_pixel_fraction: float,
    approved_task_object_within_frame_margin: bool,
    restore_steps: int,
    approved_task_object_bbox_xyxy: Sequence[int] | None = None,
    approved_task_object_centroid_xy_fraction: Sequence[float] | None = None,
    frame_resolution_hw: Sequence[int] | None = None,
    min_object_pixels: int = MIN_WRIST_OBJECT_PIXELS,
    min_object_pixel_fraction: float = MIN_WRIST_OBJECT_PIXEL_FRACTION,
    object_offset_tolerance_m: float = EPISODE_START_OBJECT_OFFSET_TOLERANCE_M,
    joint_tolerance_rad: float = EPISODE_START_JOINT_TOLERANCE_RAD,
) -> dict[str, Any]:
    """Validate a reset-time replay of a selected observable arm pose."""

    selected = [float(value) for value in selected_joint_position_rad]
    restored = [float(value) for value in restored_joint_position_rad]
    offset = [float(value) for value in object_offset_m]
    if len(selected) != 7 or len(restored) != 7 or len(offset) != 3:
        raise ApproachCaptureError(["wrist_episode_start_restore_sample_invalid"])
    joint_errors = [abs(actual - target) for target, actual in zip(selected, restored)]
    maximum_joint_error = max(joint_errors)
    blockers: list[str] = []
    if maximum_joint_error > float(joint_tolerance_rad):
        blockers.append(BLOCKER_EPISODE_START_RESTORE_JOINT_MISMATCH)
    if any(abs(value) > float(object_offset_tolerance_m) for value in offset):
        blockers.append(BLOCKER_EPISODE_START_RESTORE_OBJECT_MOVED)
    pixels = int(approved_task_object_pixel_count)
    pixel_fraction = float(approved_task_object_pixel_fraction)
    within_frame_margin = bool(approved_task_object_within_frame_margin)
    if (
        pixels < int(min_object_pixels)
        or pixel_fraction < float(min_object_pixel_fraction)
        or not within_frame_margin
    ):
        blockers.append(BLOCKER_EPISODE_START_RESTORE_OBJECT_NOT_VISIBLE)
    receipt: dict[str, Any] = {
        "schema_version": "adp009d_wrist_episode_start_restore.v2",
        "status": "ready" if not blockers else "blocked",
        "blockers": sorted(blockers),
        "selected_joint_position_rad": selected,
        "restored_joint_position_rad": restored,
        "joint_absolute_error_rad": joint_errors,
        "maximum_joint_error_rad": maximum_joint_error,
        "joint_tolerance_rad": float(joint_tolerance_rad),
        "object_offset_m": offset,
        "object_offset_tolerance_m": float(object_offset_tolerance_m),
        "approved_task_object_pixel_count": pixels,
        "approved_task_object_pixel_fraction": pixel_fraction,
        "approved_task_object_within_frame_margin": within_frame_margin,
        "approved_task_object_bbox_xyxy": (
            None
            if approved_task_object_bbox_xyxy is None
            else [int(value) for value in approved_task_object_bbox_xyxy]
        ),
        "approved_task_object_centroid_xy_fraction": (
            None
            if approved_task_object_centroid_xy_fraction is None
            else [
                float(value)
                for value in approved_task_object_centroid_xy_fraction
            ]
        ),
        "frame_resolution_hw": (
            None
            if frame_resolution_hw is None
            else [int(value) for value in frame_resolution_hw]
        ),
        "min_approved_task_object_pixels": int(min_object_pixels),
        "min_approved_task_object_pixel_fraction": float(
            min_object_pixel_fraction
        ),
        "restore_steps": int(restore_steps),
    }
    receipt["restore_digest"] = canonical_digest(
        receipt, digest_field="restore_digest"
    )
    return receipt


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


def rigid_offset_in_body_frame(
    *,
    body_position_world: Sequence[float],
    body_quaternion_world_wxyz: Sequence[float],
    child_position_world: Sequence[float],
    child_quaternion_world_wxyz: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Express a child's world pose as a constant offset in a body's frame.

    Arena parents the wrist camera under the Robotiq gripper base, which is not
    an articulation body -- PhysX never writes a pose for it, so the camera keeps
    its authored transform while the arm moves.  A live run measured the hand
    travelling 0.27 m while every recorded wrist pose stayed byte-identical.

    Rather than re-author Arena's rig, the offset is measured once at the reset
    pose, where the authored transform is still the true one, and re-applied from
    the live body pose at each capture.  This assumes the gripper base is rigidly
    fixed to the body it hangs from, which holds: the fingers articulate, the
    base does not.
    """

    body_inverse = _quat_conjugate(body_quaternion_world_wxyz)
    delta = [
        float(child_position_world[index]) - float(body_position_world[index])
        for index in range(3)
    ]
    position_body = list(_quat_rotate(body_inverse, delta))
    quaternion_body = list(_quat_multiply(body_inverse, child_quaternion_world_wxyz))
    return position_body, quaternion_body


def apply_rigid_offset(
    *,
    body_position_world: Sequence[float],
    body_quaternion_world_wxyz: Sequence[float],
    offset_position_body: Sequence[float],
    offset_quaternion_body_wxyz: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Rebuild a child's world pose from a live body pose and a constant offset."""

    rotated = _quat_rotate(body_quaternion_world_wxyz, offset_position_body)
    position_world = [
        float(body_position_world[index]) + rotated[index] for index in range(3)
    ]
    quaternion_world = list(
        _quat_multiply(body_quaternion_world_wxyz, offset_quaternion_body_wxyz)
    )
    return position_world, quaternion_world


def _max_travel_m(positions: Sequence[Sequence[float]]) -> float:
    """Largest displacement of any sample from the first sample."""

    usable = [tuple(float(v) for v in p) for p in positions if p and len(p) == 3]
    if len(usable) < 2:
        return 0.0
    first = usable[0]
    return max(
        sum((a - b) ** 2 for a, b in zip(other, first)) ** 0.5 for other in usable[1:]
    )


def classify_wrist_pose_discrepancy(
    *,
    reported_positions: Sequence[Sequence[float]],
    usd_positions: Sequence[Sequence[float]],
    min_travel_m: float = MIN_WRIST_POSE_TRAVEL_M,
) -> dict[str, Any]:
    """Separate a lagging pose buffer from a camera prim that never moves.

    ``reported_positions`` are the sensor-reported world positions; ``usd_positions``
    are the world translations computed directly from the USD stage for the same
    prim on the same steps.  If the stage says the prim moved while the sensor
    reported a constant pose, the buffer is stale.  If the stage agrees the prim
    never moved, the camera is not attached to the hand.
    """

    reported_travel = _max_travel_m(reported_positions)
    usd_travel = _max_travel_m(usd_positions)
    usable_usd = [p for p in usd_positions if p and len(p) == 3]

    if len(usable_usd) < 2:
        cause = WRIST_POSE_CAUSE_UNDETERMINED
    elif reported_travel >= float(min_travel_m):
        cause = WRIST_POSE_CAUSE_HEALTHY
    elif usd_travel >= float(min_travel_m):
        cause = WRIST_POSE_CAUSE_STALE_BUFFER
    else:
        cause = WRIST_POSE_CAUSE_PRIM_DETACHED

    return {
        "cause": cause,
        "reported_pose_travel_m": reported_travel,
        "usd_pose_travel_m": usd_travel,
        "usd_samples": len(usable_usd),
    }


def summarize_wrist_approach_capture(
    *,
    captured_frames: Sequence[Mapping[str, Any]],
    approved_task_object_label: str = "approved_can",
    ik_succeeded: bool = True,
    object_displacement_m: float = 0.0,
    arm_moved: bool = True,
    min_object_pixels: int = MIN_WRIST_OBJECT_PIXELS,
    waypoint_arrivals: Sequence[Mapping[str, Any]] | None = None,
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
        observed = semantic_label_pixel_count(
            id_to_labels=labels,
            pixel_counts_by_id=counts,
            target_label=approved_task_object_label,
        )
        best = max(best, observed)
        rows.append(
            {
                "frame_index": frame.get("frame_index"),
                "approved_task_object_pixel_count": observed,
                "position_world_m": list(frame.get("position_world_m") or []),
                "prim_diagnostics": dict(frame.get("prim_diagnostics") or {}),
            }
        )

    blockers: list[str] = []
    if not ik_succeeded:
        blockers.append(BLOCKER_APPROACH_IK_FAILED)
    if best < int(min_object_pixels):
        blockers.append(BLOCKER_WRIST_NEVER_SAW_OBJECT)
    if float(object_displacement_m) > APPROACH_MAX_OBJECT_DISPLACEMENT_M:
        blockers.append(BLOCKER_APPROACH_DISTURBED_OBJECT)
    positions = [tuple(row["position_world_m"]) for row in rows if row["position_world_m"]]
    wrist_pose_travel_m = 0.0
    if len(positions) > 1:
        first = positions[0]
        wrist_pose_travel_m = max(
            sum((float(a) - float(b)) ** 2 for a, b in zip(other, first)) ** 0.5
            for other in positions[1:]
        )
    # Travel needs at least two samples to mean anything.  A run that aborted at
    # the first waypoint captured exactly one wrist frame; travel across one
    # sample is trivially zero, which previously read as a frozen camera even
    # though that frame showed 49,758 pixels of the approved can.  One sample is
    # undetermined, not stale.
    if (
        arm_moved
        and len(positions) > 1
        and wrist_pose_travel_m < MIN_WRIST_POSE_TRAVEL_M
    ):
        blockers.append(BLOCKER_WRIST_POSE_STALE)

    arrivals = [dict(row) for row in (waypoint_arrivals or [])]
    worst_arrival_error_m: float | None = None
    if arrivals:
        errors = [
            float(row.get("position_error_m"))
            for row in arrivals
            if row.get("position_error_m") is not None
        ]
        if errors:
            worst_arrival_error_m = max(errors)
            if worst_arrival_error_m > APPROACH_WAYPOINT_TOLERANCE_M:
                blockers.append(BLOCKER_APPROACH_DID_NOT_REACH)

    pose_discrepancy = classify_wrist_pose_discrepancy(
        reported_positions=[row["position_world_m"] for row in rows],
        usd_positions=[
            (row["prim_diagnostics"] or {}).get("usd_world_translation_m") or []
            for row in rows
        ],
    )

    report: dict[str, Any] = {
        "schema_version": APPROACH_CAPTURE_SCHEMA_VERSION,
        "status": "observed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "differential_ik_succeeded": bool(ik_succeeded),
        "waypoints": approach_waypoints_world(),
        "wrist_frames": rows,
        "max_approved_task_object_pixel_count": best,
        "object_displacement_m": float(object_displacement_m),
        "wrist_pose_travel_m": wrist_pose_travel_m,
        "wrist_pose_discrepancy": pose_discrepancy,
        "waypoint_arrivals": arrivals,
        "worst_waypoint_position_error_m": worst_arrival_error_m,
        "waypoint_tolerance_m": APPROACH_WAYPOINT_TOLERANCE_M,
        "arm_moved": bool(arm_moved),
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
    "BLOCKER_EPISODE_START_RESTORE_JOINT_MISMATCH",
    "BLOCKER_EPISODE_START_RESTORE_OBJECT_MOVED",
    "BLOCKER_EPISODE_START_RESTORE_OBJECT_NOT_VISIBLE",
    "BLOCKER_NO_SAFE_WRIST_OBSERVABLE_EPISODE_START",
    "BLOCKER_APPROACH_DISTURBED_OBJECT",
    "BLOCKER_WRIST_POSE_STALE",
    "APPROACH_CAPTURE_SCHEMA_VERSION",
    "APPROACH_STANDOFF_HEIGHTS_M",
    "APPROACH_STEPS_PER_WAYPOINT",
    "APPROACH_TOOL_QUAT_WXYZ",
    "ApproachCaptureError",
    "APPROACH_GRIPPER_BODY_NAMES",
    "APPROACH_WAYPOINT_TOLERANCE_M",
    "APPROVED_CAN_TOP_ABOVE_SUPPORT_M",
    "BLOCKER_APPROACH_DID_NOT_REACH",
    "BLOCKER_APPROACH_IK_FAILED",
    "BLOCKER_WRIST_NEVER_SAW_OBJECT",
    "EPISODE_START_OBJECT_OFFSET_TOLERANCE_M",
    "EPISODE_START_JOINT_TOLERANCE_RAD",
    "EPISODE_START_RESTORE_MAX_STEPS",
    "MIN_WRIST_OBJECT_PIXEL_FRACTION",
    "MIN_WRIST_POSE_TRAVEL_M",
    "WRIST_POSE_CAUSE_HEALTHY",
    "WRIST_POSE_CAUSE_PRIM_DETACHED",
    "WRIST_POSE_CAUSE_STALE_BUFFER",
    "WRIST_POSE_CAUSE_UNDETERMINED",
    "apply_rigid_offset",
    "approach_waypoints_world",
    "classify_wrist_pose_discrepancy",
    "rigid_offset_in_body_frame",
    "pose_world_to_base",
    "select_wrist_observable_episode_start",
    "semantic_label_pixel_count",
    "semantic_target_observability",
    "summarize_wrist_approach_capture",
    "validate_wrist_observable_episode_start_restore",
    "WRIST_OBJECT_FRAME_MARGIN_FRACTION",
]
