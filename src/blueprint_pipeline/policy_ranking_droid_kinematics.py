"""Fail-closed DROID joint-command integration for prospective WAM rollouts.

This is a kinematic conditioning adapter, not a dynamics simulator.  It mirrors
the public DROID/OpenPI evaluation path closely enough to turn one 8-D policy
action chunk into a bounded Franka joint/gripper trajectory for camera and OSCAR
skeleton rendering.  Contact, torque, grasp, and physical success remain outside
its claim boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "droid_joint_kinematic_trajectory.v1"
DROID_SOURCE_REVISION = "33ae6a67274f36d2e29525b86f23a56616ef43a7"
OPENPI_SOURCE_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
CONTROL_HZ = 15
MAX_JOINT_DELTA_RAD = 0.2
MAX_ACTION_HORIZON = 15
RESET_JOINTS = np.asarray(
    [0.0, -math.pi / 5.0, 0.0, -4.0 * math.pi / 5.0, 0.0, 3.0 * math.pi / 5.0, 0.0],
    dtype=np.float64,
)
JOINT_LIMITS_RAD = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float64,
)


def _finite_vector(value: Sequence[float], width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"invalid_{name}")
    return array


def integrate_joint_velocity_chunk(
    actions: Sequence[Sequence[float]],
    *,
    initial_joint_position: Sequence[float] = RESET_JOINTS,
    initial_gripper_position: float = 0.0,
) -> dict[str, Any]:
    """Integrate one OpenPI DROID action chunk into kinematic joint states.

    The public OpenPI DROID loop clips each action dimension to ``[-1, 1]``,
    binarizes gripper position at ``0.5``, and sends joint-velocity commands to
    DROID.  DROID converts normalized joint velocity to a per-command joint
    delta using ``max_joint_delta = 0.2``.  We preserve that discrete conversion
    rather than incorrectly multiplying the normalized command by ``1/15``.
    """

    action_array = np.asarray(actions, dtype=np.float64)
    if (
        action_array.ndim != 2
        or action_array.shape[1] != 8
        or not 1 <= action_array.shape[0] <= MAX_ACTION_HORIZON
        or not np.isfinite(action_array).all()
    ):
        raise ValueError("invalid_action_chunk")
    joints = _finite_vector(initial_joint_position, 7, "initial_joint_position")
    if np.any(joints < JOINT_LIMITS_RAD[:, 0]) or np.any(joints > JOINT_LIMITS_RAD[:, 1]):
        raise ValueError("initial_joint_position_out_of_range")
    gripper = float(initial_gripper_position)
    if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
        raise ValueError("invalid_initial_gripper_position")

    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "joint_position_rad": joints.tolist(),
            "gripper_position": gripper,
            "joint_limit_clipped": [],
        }
    ]
    action_clip_count = 0
    joint_limit_clip_count = 0
    for step, raw in enumerate(action_array, start=1):
        clipped = np.clip(raw, -1.0, 1.0)
        action_clip_count += int(np.count_nonzero(clipped != raw))
        proposed = joints + clipped[:7] * MAX_JOINT_DELTA_RAD
        bounded = np.clip(proposed, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        clipped_joints = np.flatnonzero(bounded != proposed).astype(int).tolist()
        joint_limit_clip_count += len(clipped_joints)
        joints = bounded
        gripper = float(clipped[7] > 0.5)
        rows.append(
            {
                "step": step,
                "normalized_joint_velocity_command": clipped[:7].tolist(),
                "joint_position_rad": joints.tolist(),
                "gripper_position": gripper,
                "joint_limit_clipped": clipped_joints,
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "droid_source_revision": DROID_SOURCE_REVISION,
        "openpi_source_revision": OPENPI_SOURCE_REVISION,
        "control_hz": CONTROL_HZ,
        "max_joint_delta_rad_per_command": MAX_JOINT_DELTA_RAD,
        "action_dimension": 8,
        "action_step_count": int(action_array.shape[0]),
        "action_value_clip_count": action_clip_count,
        "joint_limit_clip_count": joint_limit_clip_count,
        "states": rows,
        "blockers": [],
        "claim_boundary": {
            "kinematic_skeleton_conditioning_supported": True,
            "dynamics_or_contact_simulated": False,
            "task_success_scored": False,
            "physical_execution_proven": False,
        },
    }
    result["trajectory_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON with actions and optional initial state")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("input_must_be_object")
    result = integrate_joint_velocity_chunk(
        payload.get("actions", []),
        initial_joint_position=payload.get("initial_joint_position", RESET_JOINTS),
        initial_gripper_position=float(payload.get("initial_gripper_position", 0.0)),
    )
    write_json(Path(args.output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
