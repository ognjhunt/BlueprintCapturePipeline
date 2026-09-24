"""Pinned HumanoidArena semantic-v3 policy input and reference-action boundary.

These 40 policy values are whole-body references for SONIC. They are never
Isaac joint targets: a measured controller result must cross that later seam.
The source layout is HumanoidArena commit 68479287a784a69be9ce6ad739311d2f11f75ef9.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .gear_sonic_joint_order_contract import PROTOCOL_V4_BODY_JOINT_NAMES


CANONICAL_BODY_JOINT_NAMES_29 = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
STATE_WIDTH = 64
ACTION_WIDTH = 40
SOURCE_COMMIT = "68479287a784a69be9ce6ad739311d2f11f75ef9"

if set(CANONICAL_BODY_JOINT_NAMES_29) != set(PROTOCOL_V4_BODY_JOINT_NAMES):
    raise RuntimeError("g1_humanoidarena_joint_inventory_drift")


def _vector(value: Any, width: int, code: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(code)
    if any(isinstance(item, bool) for item in value):
        raise ValueError(code)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(code) from exc
    if len(result) != width or not all(math.isfinite(item) for item in result):
        raise ValueError(code)
    return result


def _quat_xyzw(value: Any) -> tuple[float, float, float, float]:
    x, y, z, w = _vector(value, 4, "g1_root_quaternion_invalid")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(norm - 1.0) > 1e-5:
        raise ValueError("g1_root_quaternion_invalid")
    return (w, x, y, z)


def _multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = a
    v, i, j, k = b
    return (
        w * v - x * i - y * j - z * k,
        w * i + x * v + y * k - z * j,
        w * j - x * k + y * v + z * i,
        w * k + x * j - y * i + z * v,
    )


def _named_joint_vector(value: Any, code: str) -> list[float]:
    if not isinstance(value, Mapping) or set(value) != set(CANONICAL_BODY_JOINT_NAMES_29):
        raise ValueError(code)
    try:
        return _vector([value[name] for name in CANONICAL_BODY_JOINT_NAMES_29], 29, code)
    except KeyError as exc:
        raise ValueError(code) from exc


def build_semantic_v3_state(
    *,
    initial_root_orientation_xyzw: Sequence[float],
    current_root_orientation_xyzw: Sequence[float],
    body_joint_positions_rad: Mapping[str, float],
    body_joint_velocities_rad_s: Mapping[str, float],
) -> list[float]:
    """Return the exact 64-wide heading-canonical observation.state vector."""

    initial = _quat_xyzw(initial_root_orientation_xyzw)
    current = _quat_xyzw(current_root_orientation_xyzw)
    w, x, y, z = initial
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    heading_inverse = (math.cos(yaw / 2), 0.0, 0.0, -math.sin(yaw / 2))
    q0, q1, q2, q3 = _multiply(heading_inverse, current)
    # First two rotation-matrix columns, flattened by row, match upstream
    # quat_to_rot6d_wxyz rather than a column-major SONIC encoder layout.
    rotation6 = [
        1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3),
        2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3),
        2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1),
    ]
    positions = _named_joint_vector(body_joint_positions_rad, "g1_semantic_state_joint_positions_invalid")
    velocities = _named_joint_vector(body_joint_velocities_rad_s, "g1_semantic_state_joint_velocities_invalid")
    state = [*rotation6, *positions, *velocities]
    if len(state) != STATE_WIDTH:
        raise AssertionError("g1_semantic_state_width_drift")
    return state


def parse_semantic_v3_action(value: Sequence[float]) -> dict[str, Any]:
    """Validate a policy reference without interpreting it as motor commands."""

    action = _vector(value, ACTION_WIDTH, "g1_semantic_action_shape_or_value_invalid")
    first = (action[3], action[5], action[7])
    second = (action[4], action[6], action[8])
    first_norm = math.sqrt(sum(item * item for item in first))
    second_norm = math.sqrt(sum(item * item for item in second))
    cross_norm = math.sqrt(sum(item * item for item in (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )))
    if min(first_norm, second_norm, cross_norm) < 1e-6:
        raise ValueError("g1_semantic_action_rotation_invalid")
    hands = action[38:40]
    if any(hand < 0.0 or hand > 1.0 for hand in hands):
        raise ValueError("g1_semantic_action_hand_range_invalid")
    return {
        "schema_version": "g1_humanoidarena_semantic_v3_action.v1",
        "source_commit": SOURCE_COMMIT,
        "action_interface": "humanoidarena_semantic_v3",
        "root_ref_base_local_xy_delta": action[0:2],
        "root_z": action[2],
        "root_ref_rot6d": action[3:9],
        "body_joint_reference_rad": dict(zip(CANONICAL_BODY_JOINT_NAMES_29, action[9:38], strict=True)),
        "left_right_hand_binary_reference": hands,
        "requires_sonic_controller": True,
    }
