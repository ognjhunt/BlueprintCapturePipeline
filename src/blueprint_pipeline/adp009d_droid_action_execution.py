"""Execute DROID policy action chunks in the ADP-009D Isaac environment.

The interfaces line up more closely than they had any right to.  Arena's
``droid_abs_joint_pos`` action space is 8-dimensional -- seven absolute arm
joint positions plus one gripper command -- which is exactly a DROID action row.
And the environment's ``sim.dt = 1/120`` with ``decimation = 8`` means one
``env.step()`` advances 1/15 s, exactly DROID's 15 Hz control rate.  So one
policy action row is one environment step, with no resampling.

What does *not* line up for free is the gripper.  DROID encodes it as a scalar
in [0, 1] where above 0.5 means closed; Arena's eighth action dimension has its
own convention, and guessing it would silently invert every grasp.  The
convention is therefore a required, measured input rather than a default: see
``GripperConvention`` and the probe contract it documents.

This module is pure arithmetic so it can be tested without a GPU.  It never
queries a policy and never steps a simulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

ACTION_EXECUTION_SCHEMA_VERSION = "adp009d_droid_action_execution.v1"

# DROID's published control contract.
DROID_CONTROL_HZ = 15
DROID_OPEN_LOOP_HORIZON = 8
DROID_ACTION_WIDTH = 8
ARM_JOINT_COUNT = 7

# The ADP-009D environment's own timing, from the runtime configuration.
ISAAC_SIM_DT_SECONDS = 1.0 / 120.0
ISAAC_DECIMATION = 8
ISAAC_ACTION_DIM = 8

BLOCKER_CHUNK_SHAPE = "droid_action_chunk_shape_invalid"
BLOCKER_CHUNK_NONFINITE = "droid_action_chunk_nonfinite"
BLOCKER_HORIZON_UNAVAILABLE = "droid_action_chunk_shorter_than_open_loop_horizon"
BLOCKER_CONTROL_RATE_MISMATCH = "isaac_step_rate_does_not_match_droid_control_hz"
BLOCKER_GRIPPER_CONVENTION_UNMEASURED = "isaac_gripper_convention_unmeasured"


class DroidActionExecutionError(ValueError):
    """Fail-closed DROID action execution contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


@dataclass(frozen=True)
class GripperConvention:
    """How Arena's eighth action dimension encodes open and closed.

    Both values must come from a probe that commanded each and observed the
    resulting finger joint travel.  There is no default: an inverted convention
    would turn every commanded grasp into a release, and the resulting eval
    would look like a policy failure rather than a harness bug.
    """

    closed_command: float
    open_command: float
    measured_by_probe: bool = False

    def command_for(self, droid_gripper: float) -> float:
        # DROID: scalar in [0, 1], above 0.5 means closed.
        return self.closed_command if float(droid_gripper) > 0.5 else self.open_command


def isaac_steps_per_droid_action(
    *,
    sim_dt_seconds: float = ISAAC_SIM_DT_SECONDS,
    decimation: int = ISAAC_DECIMATION,
    control_hz: int = DROID_CONTROL_HZ,
) -> int:
    """Environment steps per policy action, refusing any non-integer ratio.

    A fractional ratio would mean the policy's actions and the simulator's
    timeline drift apart, so it fails closed rather than rounding.
    """

    step_seconds = float(sim_dt_seconds) * int(decimation)
    action_seconds = 1.0 / float(control_hz)
    ratio = action_seconds / step_seconds
    nearest = round(ratio)
    if nearest < 1 or abs(ratio - nearest) > 1e-9:
        raise DroidActionExecutionError(
            [f"{BLOCKER_CONTROL_RATE_MISMATCH}:ratio={ratio!r}"]
        )
    return int(nearest)


def validate_action_chunk(chunk: Any, *, horizon: int = DROID_OPEN_LOOP_HORIZON) -> Any:
    """Validate a policy's action chunk before any of it reaches the simulator."""

    import numpy as np

    values = np.asarray(chunk, dtype=float)
    errors: list[str] = []
    if values.ndim != 2 or values.shape[1] != DROID_ACTION_WIDTH:
        raise DroidActionExecutionError(
            [f"{BLOCKER_CHUNK_SHAPE}:{tuple(values.shape)}"]
        )
    if not np.isfinite(values).all():
        errors.append(BLOCKER_CHUNK_NONFINITE)
    if values.shape[0] < int(horizon):
        errors.append(f"{BLOCKER_HORIZON_UNAVAILABLE}:{values.shape[0]}<{horizon}")
    if errors:
        raise DroidActionExecutionError(errors)
    return values


def droid_row_to_isaac_action(
    row: Sequence[float],
    *,
    joint_limits: Sequence[Sequence[float]],
    gripper: GripperConvention,
) -> dict[str, Any]:
    """Convert one DROID action row into Arena's 8-dimensional action vector."""

    import numpy as np

    if not gripper.measured_by_probe:
        raise DroidActionExecutionError([BLOCKER_GRIPPER_CONVENTION_UNMEASURED])

    values = np.asarray(row, dtype=float)
    if values.shape != (DROID_ACTION_WIDTH,) or not np.isfinite(values).all():
        raise DroidActionExecutionError(
            [f"{BLOCKER_CHUNK_SHAPE}:{tuple(values.shape)}"]
        )
    limits = np.asarray(joint_limits, dtype=float)
    if limits.shape != (ARM_JOINT_COUNT, 2) or not np.isfinite(limits).all():
        raise DroidActionExecutionError(["isaac_joint_limits_invalid"])

    target = np.clip(values[:ARM_JOINT_COUNT], limits[:, 0], limits[:, 1])
    clamped = bool(np.any(np.abs(target - values[:ARM_JOINT_COUNT]) > 1e-12))
    action = np.zeros(ISAAC_ACTION_DIM, dtype=float)
    action[:ARM_JOINT_COUNT] = target
    action[ARM_JOINT_COUNT] = gripper.command_for(values[ARM_JOINT_COUNT])
    return {
        "isaac_action": [float(v) for v in action],
        "joint_position_target_rad": [float(v) for v in target],
        "joint_limit_clamped": clamped,
        "droid_gripper_scalar": float(values[ARM_JOINT_COUNT]),
        "gripper_closed": bool(float(values[ARM_JOINT_COUNT]) > 0.5),
    }


def plan_chunk_execution(
    chunk: Any,
    *,
    joint_limits: Sequence[Sequence[float]],
    gripper: GripperConvention,
    horizon: int = DROID_OPEN_LOOP_HORIZON,
) -> dict[str, Any]:
    """Turn a validated chunk into the exact per-step actions to execute.

    Only the first ``horizon`` rows are executed; DROID's open-loop horizon is
    shorter than the chunk a policy returns, and executing the tail would run
    the arm on predictions the policy expected to have superseded.
    """

    values = validate_action_chunk(chunk, horizon=horizon)
    steps_per_action = isaac_steps_per_droid_action()
    rows = [
        droid_row_to_isaac_action(
            values[index], joint_limits=joint_limits, gripper=gripper
        )
        for index in range(int(horizon))
    ]
    return {
        "schema_version": ACTION_EXECUTION_SCHEMA_VERSION,
        "chunk_shape": [int(values.shape[0]), int(values.shape[1])],
        "executed_rows": int(horizon),
        "discarded_rows": int(values.shape[0]) - int(horizon),
        "isaac_steps_per_action": steps_per_action,
        "control_hz": DROID_CONTROL_HZ,
        "environment_step_seconds": ISAAC_SIM_DT_SECONDS * ISAAC_DECIMATION,
        "actions": rows,
        "any_joint_limit_clamped": any(row["joint_limit_clamped"] for row in rows),
        "candidate_policy_queried": True,
    }


def build_gripper_convention_probe_request() -> dict[str, Any]:
    """Describe the probe that must measure the gripper convention.

    Recorded rather than executed here: the measurement needs a simulator, and
    the point of this contract is that the convention is never assumed.
    """

    return {
        "schema_version": ACTION_EXECUTION_SCHEMA_VERSION,
        "purpose": "measure_isaac_eighth_action_dimension_gripper_convention",
        "method": (
            "command each candidate value on action dimension 7 with the arm "
            "held at the canonical pose, step until the finger joints settle, "
            "and record finger joint travel for each"
        ),
        "candidate_commands": [0.0, 1.0],
        "observed_joint_names": [
            "finger_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ],
        "decision_rule": (
            "closed_command is whichever value reduces finger separation; "
            "an ambiguous or zero-travel result fails closed rather than "
            "defaulting, because an inverted convention turns every commanded "
            "grasp into a release"
        ),
    }


__all__ = [
    "ACTION_EXECUTION_SCHEMA_VERSION",
    "ARM_JOINT_COUNT",
    "DROID_ACTION_WIDTH",
    "DROID_CONTROL_HZ",
    "DROID_OPEN_LOOP_HORIZON",
    "DroidActionExecutionError",
    "GripperConvention",
    "build_gripper_convention_probe_request",
    "droid_row_to_isaac_action",
    "isaac_steps_per_droid_action",
    "plan_chunk_execution",
    "validate_action_chunk",
]
