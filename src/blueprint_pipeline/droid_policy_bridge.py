"""Fail-closed OpenPI DROID observation/action bridge for simulator loops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DROID_CONTROL_HZ = 15
DROID_INNER_CONTROL_HZ = 1000
DROID_OPEN_LOOP_HORIZON = 8
DROID_ACTION_CHUNK_SHAPE = (10, 8)
DROID_MAX_JOINT_DELTA_RAD = 0.2
DROID_SOURCE_REVISION = "33ae6a67274f36d2e29525b86f23a56616ef43a7"
OPENPI_SOURCE_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def validate_droid_observation(observation: Mapping[str, Any]) -> list[str]:
    import numpy as np

    blockers: list[str] = []
    for key in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
        image = np.asarray(observation.get(key))
        if image.shape != (224, 224, 3):
            blockers.append(f"invalid_image_shape:{key}")
        elif image.dtype != np.uint8:
            blockers.append(f"invalid_image_dtype:{key}")
    joints = np.asarray(observation.get("observation/joint_position"), dtype=float)
    gripper = np.asarray(observation.get("observation/gripper_position"), dtype=float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        blockers.append("invalid_joint_position")
    if gripper.shape != (1,) or not np.isfinite(gripper).all():
        blockers.append("invalid_gripper_position")
    if not str(observation.get("prompt") or "").strip():
        blockers.append("missing_prompt")
    return blockers


def validate_droid_action_chunk(actions: Any, *, expected_rows: int = 10) -> list[str]:
    import numpy as np

    chunk = np.asarray(actions, dtype=float)
    blockers: list[str] = []
    if chunk.shape != (int(expected_rows), 8):
        blockers.append("invalid_action_chunk_shape")
    elif not np.isfinite(chunk).all():
        blockers.append("nonfinite_action_chunk")
    return blockers


def droid_joint_position_action_to_mujoco_targets(
    action: Sequence[float],
    *,
    joint_limits: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Map an OpenPI absolute DROID joint-position action into MuJoCo targets."""
    import numpy as np

    values = np.asarray(action, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("action_must_be_finite_shape_8")
    if limits.shape != (7, 2) or not np.isfinite(limits).all():
        raise ValueError("joint_limits_must_be_finite_shape_7x2")
    target = np.clip(values[:7], limits[:, 0], limits[:, 1])
    gripper = float(np.clip(values[7], 0.0, 1.0))
    return {
        "action": [float(value) for value in values],
        "joint_position_target_rad": [float(value) for value in target],
        "gripper_position_target_m": 0.0 if gripper > 0.5 else 0.04,
        "joint_limit_clamped": bool(np.any(np.abs(target - values[:7]) > 1e-12)),
        "control_hz": DROID_CONTROL_HZ,
    }


def droid_action_to_mujoco_targets(
    action: Sequence[float],
    *,
    current_joint_position: Sequence[float],
    joint_limits: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Mirror the public DROID joint-velocity runtime's normalized action mapping.

    DROID clips the eight outputs to [-1, 1], maps each arm dimension to at most
    0.2 rad of joint delta, and binarizes gripper position at 0.5.  DROID gripper
    position is 0=open and 1=closed; MuJoCo's Panda finger actuator uses 0.04 m
    for open and 0.0 m for closed.
    """
    import numpy as np

    values = np.asarray(action, dtype=float)
    current = np.asarray(current_joint_position, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("action_must_be_finite_shape_8")
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError("current_joint_position_must_be_finite_shape_7")
    if limits.shape != (7, 2) or not np.isfinite(limits).all():
        raise ValueError("joint_limits_must_be_finite_shape_7x2")
    clipped = np.clip(values, -1.0, 1.0)
    unclamped_target = current + clipped[:7] * DROID_MAX_JOINT_DELTA_RAD
    target = np.clip(unclamped_target, limits[:, 0], limits[:, 1])
    return {
        "clipped_action": [float(value) for value in clipped],
        "joint_position_target_rad": [float(value) for value in target],
        "gripper_position_target_m": 0.0 if float(clipped[7]) > 0.5 else 0.04,
        "joint_limit_clamped": bool(np.any(np.abs(target - unclamped_target) > 1e-12)),
        "control_hz": DROID_CONTROL_HZ,
    }


__all__ = [
    "DROID_ACTION_CHUNK_SHAPE",
    "DROID_CONTROL_HZ",
    "DROID_INNER_CONTROL_HZ",
    "DROID_MAX_JOINT_DELTA_RAD",
    "DROID_OPEN_LOOP_HORIZON",
    "DROID_SOURCE_REVISION",
    "OPENPI_SOURCE_REVISION",
    "droid_joint_position_action_to_mujoco_targets",
    "droid_action_to_mujoco_targets",
    "validate_droid_action_chunk",
    "validate_droid_observation",
]
