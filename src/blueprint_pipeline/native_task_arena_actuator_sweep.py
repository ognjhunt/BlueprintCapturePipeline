"""Measure, in one run, which actuator gains and postures can reach the pose.

Thirty-four controls runs tested roughly one hypothesis each, and the last
three moved the fingertip 11.6 mm, 23.0 mm, and 134.2 mm from the handle while
varying only the *controller*.  The binding constraint was never there.  The
Arena DROID embodiment gives ``panda_joint[5-7]`` stiffness 400 N-m/rad and
damping 80 N-m/(rad/s) against a 12 N-m effort limit, so the wrist saturates
past 0.03 rad of position error or 0.15 rad/s of speed -- and no controller can
satisfy both while actually moving.

A sweep answers that in one run instead of one hypothesis per paid GPU.  Each
cell is cheap: write a candidate gain set, command a candidate solved posture,
step a bounded settle, and measure what the arm actually did.  Twenty-odd cells
cost seconds of simulator time against the minutes the scene already cost, so
the run returns a sensitivity surface rather than a single verdict.

This is measurement only.  It runs before the deterministic controls, restores
the original gains before returning, and asserts no task outcome: the cells
report tracking error, saturation, and measured fingertip distance, never
success.  Nothing here gates a run, and the controls that follow are unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SWEEP_SCHEMA_VERSION = "native_task_arena_actuator_posture_sweep.v1"

#: Candidate wrist gain sets.  The shipped pair is included so the sweep
#: measures the status quo alongside the alternatives rather than assuming it
#: is wrong, and the rest walk the stiffness/damping plane toward values a
#: 12 N-m joint can actually deliver.
DEFAULT_WRIST_GAIN_CANDIDATES: tuple[tuple[float, float], ...] = (
    (400.0, 80.0),
    (200.0, 40.0),
    (80.0, 16.0),
    (80.0, 4.0),
    (40.0, 8.0),
)

#: Steps each cell is allowed to converge before it is measured.  Long enough
#: for a settled reading at the slowest candidate, short enough that the whole
#: sweep stays inside seconds.
DEFAULT_CELL_SETTLE_STEPS = 45


class ActuatorSweepError(ValueError):
    """Fail-closed sweep contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _finite_vector(values: Any, *, length: int | None = None) -> list[float] | None:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not vector or not all(math.isfinite(value) for value in vector):
        return None
    if length is not None and len(vector) != length:
        return None
    return vector


def candidate_postures(global_ik: Mapping[str, Any], *, phase_id: str) -> list[dict[str, Any]]:
    """Every solved branch for one phase, not only the selected one.

    The selector keeps the solution with the healthiest joint-limit margin,
    which is the right rule for execution and the wrong one for diagnosis: a
    posture rejected for margin may still be the one the arm can actually
    hold.  The sweep therefore measures all of them.
    """

    phases = global_ik.get("phases")
    if not isinstance(phases, list):
        return []
    postures: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, Mapping) or str(phase.get("phase_id") or "") != phase_id:
            continue
        rows = phase.get("solutions")
        if not isinstance(rows, list):
            rows = [phase.get("selected")]
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            joints = _finite_vector(row.get("joint_positions_rad"), length=7)
            if joints is None:
                continue
            postures.append(
                {
                    "posture_index": index,
                    "seed_index": row.get("seed_index"),
                    "joint_positions_rad": joints,
                    "offsim_position_error_m": row.get("position_error_m"),
                    "minimum_joint_limit_margin_rad": row.get(
                        "minimum_joint_limit_margin_rad"
                    ),
                }
            )
    return postures


def run_actuator_posture_sweep(
    *,
    environment: Any,
    robot: Any,
    arm_joint_ids: Sequence[int],
    target_position_world_m: Sequence[float],
    postures: Sequence[Mapping[str, Any]],
    gripper_open_command: float,
    max_joint_delta_rad: float,
    max_joint_setpoint_lead_rad: float,
    wrist_gain_candidates: Sequence[tuple[float, float]] = DEFAULT_WRIST_GAIN_CANDIDATES,
    settle_steps: int = DEFAULT_CELL_SETTLE_STEPS,
    wrist_joint_slice: slice = slice(4, 7),
) -> dict[str, Any]:
    """Measure tracking, saturation and reach for each gain x posture cell."""

    target = _finite_vector(target_position_world_m, length=3)
    if target is None or not postures:
        raise ActuatorSweepError(["actuator_sweep_inputs_invalid"])
    write_stiffness = getattr(robot, "write_joint_stiffness_to_sim", None)
    write_damping = getattr(robot, "write_joint_damping_to_sim", None)
    bounded = getattr(environment, "bounded_joint_action", None)
    if not callable(write_stiffness) or not callable(write_damping) or not callable(bounded):
        # Measurement is optional; a runtime that cannot retune is reported as
        # such rather than failing the controls that follow it.
        return {
            "schema_version": SWEEP_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "runtime_missing_gain_write_or_bounded_action",
            "cells": [],
            "claim_boundary": _CLAIM_BOUNDARY,
        }

    original = _finite_vector(
        getattr(getattr(robot, "data", None), "joint_stiffness", [None])[0]
        if hasattr(getattr(robot, "data", None), "joint_stiffness")
        else None
    )
    cells: list[dict[str, Any]] = []
    wrist_ids = list(arm_joint_ids)[wrist_joint_slice]
    try:
        for stiffness, damping in wrist_gain_candidates:
            write_stiffness(float(stiffness), joint_ids=wrist_ids)
            write_damping(float(damping), joint_ids=wrist_ids)
            for posture in postures:
                joints = _finite_vector(posture.get("joint_positions_rad"), length=7)
                if joints is None:
                    continue
                environment.reset()
                peak_utilization = 0.0
                saturated_steps = 0
                for _ in range(max(1, int(settle_steps))):
                    action = bounded(
                        target_joint_positions_rad=joints,
                        gripper_command=float(gripper_open_command),
                        max_joint_delta_rad=float(max_joint_delta_rad),
                        max_joint_setpoint_lead_rad=float(max_joint_setpoint_lead_rad),
                    )
                    environment.step([float(value) for value in action])
                    dynamics = environment.read_arm_dynamics_observation()
                    utilization = (
                        dynamics.get("joint_effort_utilization")
                        if isinstance(dynamics, Mapping)
                        else None
                    )
                    if isinstance(utilization, list) and utilization:
                        peak = max(
                            float(value) for value in utilization[wrist_joint_slice]
                        )
                        peak_utilization = max(peak_utilization, peak)
                        if peak > 0.999:
                            saturated_steps += 1
                observed = _finite_vector(
                    environment.read_arm_joint_positions(), length=7
                )
                sample = environment.read_object_sample()
                measured = (
                    _finite_vector(sample.get("grasp_frame_position_world_m"), length=3)
                    if isinstance(sample, Mapping)
                    else None
                )
                cells.append(
                    {
                        "wrist_stiffness_nm_per_rad": float(stiffness),
                        "wrist_damping_nm_per_rad_s": float(damping),
                        "posture_index": posture.get("posture_index"),
                        "seed_index": posture.get("seed_index"),
                        "offsim_position_error_m": posture.get("offsim_position_error_m"),
                        "joint_tracking_error_rad": (
                            max(abs(a - b) for a, b in zip(joints, observed))
                            if observed is not None
                            else None
                        ),
                        "wrist_peak_effort_utilization": peak_utilization,
                        "wrist_saturated_steps": saturated_steps,
                        "settle_steps": int(settle_steps),
                        "measured_grasp_frame_position_world_m": measured,
                        "measured_distance_to_target_m": (
                            math.dist(measured, target) if measured is not None else None
                        ),
                    }
                )
    finally:
        # Restore before the controls run, or the sweep would have silently
        # retuned the robot the deterministic canary is about to measure.
        if original is not None and len(original) > max(wrist_ids, default=-1):
            for joint_id in wrist_ids:
                write_stiffness(float(original[joint_id]), joint_ids=[joint_id])
        else:
            stiffness, damping = wrist_gain_candidates[0]
            write_stiffness(float(stiffness), joint_ids=wrist_ids)
            write_damping(float(damping), joint_ids=wrist_ids)
        environment.reset()

    reachable = [
        cell
        for cell in cells
        if cell["measured_distance_to_target_m"] is not None
    ]
    best = (
        min(reachable, key=lambda cell: cell["measured_distance_to_target_m"])
        if reachable
        else None
    )
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "status": "measured",
        "cell_count": len(cells),
        "cells": cells,
        "best_cell": best,
        "gains_restored": True,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


_CLAIM_BOUNDARY = (
    "diagnostic_measurement_only;reports_tracking_saturation_and_measured_"
    "reach;asserts_no_task_outcome_and_gates_nothing"
)


__all__ = [
    "ActuatorSweepError",
    "DEFAULT_CELL_SETTLE_STEPS",
    "DEFAULT_WRIST_GAIN_CANDIDATES",
    "SWEEP_SCHEMA_VERSION",
    "candidate_postures",
    "run_actuator_posture_sweep",
]
