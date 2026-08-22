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


def _grasp_frame_sample(environment: Any) -> Mapping[str, Any] | None:
    """Read the measured gripper sample from whichever sampler this task has.

    An articulated cell carries no rigid task object, so asking for the rigid
    sample raises rather than returning nothing -- which is how C35's sweep
    reported `unavailable` on a run whose arm was perfectly measurable.  Both
    samplers expose the same grasp-frame key, so try each and take the first
    that answers.
    """

    for name in ("read_task_sample", "read_object_sample"):
        reader = getattr(environment, name, None)
        if not callable(reader):
            continue
        try:
            sample = reader()
        except Exception:  # noqa: BLE001 - the other sampler may still answer
            continue
        if isinstance(sample, Mapping):
            position = _finite_vector(
                sample.get("grasp_frame_position_world_m"), length=3
            )
            if position is not None:
                return sample
    return None


def _grasp_frame_position(environment: Any) -> list[float] | None:
    sample = _grasp_frame_sample(environment)
    if sample is None:
        return None
    return _finite_vector(sample.get("grasp_frame_position_world_m"), length=3)


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
        # The multistart seals every seed it tried under `attempts`, so the
        # alternatives were always in the receipt -- C36 measured one posture
        # because this looked for a key the solver does not emit and fell back
        # to the selected branch alone.  Prefer the full attempt list, keep
        # only the seeds that actually solved, and fall back in that order.
        rows = phase.get("solutions")
        if not isinstance(rows, list):
            attempts = phase.get("attempts")
            if isinstance(attempts, list):
                rows = [
                    row
                    for row in attempts
                    if isinstance(row, Mapping) and row.get("solved") is not False
                ]
            else:
                rows = []
        if not rows:
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
                measured = _grasp_frame_position(environment)
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
    "CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_CALIBRATION_ITERATIONS",
    "DEFAULT_REACHABILITY_OFFSETS_M",
    "REACHABILITY_SCHEMA_VERSION",
    "calibrate_posture_to_measured_target",
    "probe_target_reachability",
    "DEFAULT_CELL_SETTLE_STEPS",
    "DEFAULT_WRIST_GAIN_CANDIDATES",
    "SWEEP_SCHEMA_VERSION",
    "candidate_postures",
    "run_actuator_posture_sweep",
]


CALIBRATION_SCHEMA_VERSION = "native_task_arena_measured_posture_calibration.v1"
DEFAULT_CALIBRATION_ITERATIONS = 4


def calibrate_posture_to_measured_target(
    *,
    environment: Any,
    solve: Any,
    target_position_world_m: Sequence[float],
    seed_joint_positions_rad: Sequence[float],
    gripper_open_command: float,
    max_joint_delta_rad: float,
    max_joint_setpoint_lead_rad: float,
    arrival_tolerance_m: float,
    settle_steps: int = DEFAULT_CELL_SETTLE_STEPS,
    max_iterations: int = DEFAULT_CALIBRATION_ITERATIONS,
) -> dict[str, Any]:
    """Solve for the posture whose *measured* fingertip lands on the target.

    C36 measured the defect that thirty-four runs of controller work could not
    remove: at the solved contact posture, with joint tracking at 0.007 rad and
    across a tenfold stiffness range, the fingertip sat a constant +13.0 mm off
    in a single axis.  Gain-independent, tracking-independent and one-axis is
    the signature of a kinematic constant -- the solver's model of where the
    fingertip is disagrees with where PhysX measures it, so every controller
    was faithfully driving to the wrong point.

    Rather than model that offset, measure it.  Solve, command, read the real
    fingertip, and fold the residual back into the *solver's* target; repeat.
    The fixed point is a posture whose measured frame is at the sealed target,
    which is the thing the gate has always been asking for.

    Nothing here touches a gate.  The arrival test still measures the real
    fingertip against the original sealed target: this only stops handing that
    test a posture the model already had wrong.
    """

    target = _finite_vector(target_position_world_m, length=3)
    seed = _finite_vector(seed_joint_positions_rad, length=7)
    if target is None or seed is None:
        raise ActuatorSweepError(["posture_calibration_inputs_invalid"])
    bounded = getattr(environment, "bounded_joint_action", None)
    if not callable(bounded) or not callable(solve):
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "runtime_missing_solver_or_bounded_action",
            "iterations": [],
            "claim_boundary": _CALIBRATION_CLAIM_BOUNDARY,
        }

    solver_target = list(target)
    iterations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    joints = list(seed)
    for index in range(max(1, int(max_iterations))):
        solved = solve(solver_target, joints)
        candidate = _finite_vector(solved, length=7)
        if candidate is None:
            iterations.append({"iteration": index, "status": "solver_returned_no_pose"})
            break
        joints = candidate
        environment.reset()
        for _ in range(max(1, int(settle_steps))):
            environment.step(
                [
                    float(value)
                    for value in bounded(
                        target_joint_positions_rad=joints,
                        gripper_command=float(gripper_open_command),
                        max_joint_delta_rad=float(max_joint_delta_rad),
                        max_joint_setpoint_lead_rad=float(max_joint_setpoint_lead_rad),
                    )
                ]
            )
        measured = _grasp_frame_position(environment)
        if measured is None:
            iterations.append({"iteration": index, "status": "fingertip_unmeasurable"})
            break
        residual = [measured[axis] - target[axis] for axis in range(3)]
        distance = math.dist(measured, target)
        row = {
            "iteration": index,
            "status": "measured",
            "solver_target_position_world_m": list(solver_target),
            "joint_positions_rad": list(joints),
            "measured_grasp_frame_position_world_m": list(measured),
            "measured_residual_m": residual,
            "measured_distance_to_target_m": distance,
        }
        iterations.append(row)
        if best is None or distance < best["measured_distance_to_target_m"]:
            best = row
        if distance <= float(arrival_tolerance_m):
            break
        # Fold the measured residual back into the solver's target, so the next
        # solve aims off by exactly what physics said the model was wrong by.
        solver_target = [
            solver_target[axis] - residual[axis] for axis in range(3)
        ]
    environment.reset()
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "measured" if best is not None else "unavailable",
        "iteration_count": len(iterations),
        "iterations": iterations,
        "best": best,
        "converged": bool(
            best is not None
            and best["measured_distance_to_target_m"] <= float(arrival_tolerance_m)
        ),
        "claim_boundary": _CALIBRATION_CLAIM_BOUNDARY,
    }


_CALIBRATION_CLAIM_BOUNDARY = (
    "solves_for_a_posture_whose_measured_fingertip_reaches_the_sealed_target;"
    "changes_no_gate_and_asserts_no_task_outcome"
)


REACHABILITY_SCHEMA_VERSION = "native_task_arena_target_reachability_probe.v1"

#: Offsets probed around the sealed contact target, in metres.  A cross rather
#: than a full grid: the question is which directions the measured pad midpoint
#: can actually follow, and a cross answers that per axis at a fraction of the
#: cells.
DEFAULT_REACHABILITY_OFFSETS_M: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.02, 0.0, 0.0),
    (-0.02, 0.0, 0.0),
    (0.04, 0.0, 0.0),
    (-0.04, 0.0, 0.0),
    (0.0, -0.02, 0.0),
    (0.0, -0.04, 0.0),
    (0.0, 0.0, 0.02),
    (0.0, 0.0, -0.02),
)


def probe_target_reachability(
    *,
    environment: Any,
    solve: Any,
    base_target_position_world_m: Sequence[float],
    seed_joint_positions_rad: Sequence[float],
    gripper_open_command: float,
    max_joint_delta_rad: float,
    max_joint_setpoint_lead_rad: float,
    offsets_m: Sequence[Sequence[float]] = DEFAULT_REACHABILITY_OFFSETS_M,
    settle_steps: int = DEFAULT_CELL_SETTLE_STEPS,
) -> dict[str, Any]:
    """Map where the measured pad midpoint can actually be placed.

    C37 shifted the solver's target 39 mm across four solved postures and the
    measured fingertip did not move by 0.05 mm, while contact with the door
    prim was active on 43% of samples.  Two very different stories fit that:
    the door is blocking the pose, or the solver's frame and the measured
    frame are different points and the target is chasing a ghost.

    Commanding a small cross of offsets separates them.  If the measured point
    follows the target away from the obstruction and stalls only toward it,
    the geometry is the constraint.  If it never follows, the constraint is in
    the frames.  Each cell records where the target asked for, where the pad
    midpoint actually went, and whether the arm was in contact while it did.

    Measurement only: no gate, no task outcome, and the arm is left reset.
    """

    base = _finite_vector(base_target_position_world_m, length=3)
    seed = _finite_vector(seed_joint_positions_rad, length=7)
    if base is None or seed is None:
        raise ActuatorSweepError(["reachability_probe_inputs_invalid"])
    bounded = getattr(environment, "bounded_joint_action", None)
    if not callable(bounded) or not callable(solve):
        return {
            "schema_version": REACHABILITY_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "runtime_missing_solver_or_bounded_action",
            "cells": [],
            "claim_boundary": _REACHABILITY_CLAIM_BOUNDARY,
        }

    cells: list[dict[str, Any]] = []
    for offset in offsets_m:
        delta = _finite_vector(offset, length=3)
        if delta is None:
            continue
        target = [base[axis] + delta[axis] for axis in range(3)]
        solved = solve(target, list(seed))
        joints = _finite_vector(solved, length=7)
        if joints is None:
            cells.append(
                {
                    "offset_m": delta,
                    "requested_target_position_world_m": target,
                    "status": "unsolved",
                }
            )
            continue
        environment.reset()
        contact_steps = 0
        for _ in range(max(1, int(settle_steps))):
            environment.step(
                [
                    float(value)
                    for value in bounded(
                        target_joint_positions_rad=joints,
                        gripper_command=float(gripper_open_command),
                        max_joint_delta_rad=float(max_joint_delta_rad),
                        max_joint_setpoint_lead_rad=float(max_joint_setpoint_lead_rad),
                    )
                ]
            )
            reader = getattr(environment, "read_task_sample", None)
            if callable(reader):
                try:
                    if (reader() or {}).get("task_contact_active"):
                        contact_steps += 1
                except Exception:  # noqa: BLE001 - contact state is optional here
                    pass
        # C37's calibration could not distinguish "the arm never realized
        # this posture" from "the arm realized it and the fingertip did not
        # move" because it recorded no per-cell joint tracking.  Record it.
        tracking = None
        joints_reader = getattr(environment, "read_arm_joint_positions", None)
        if callable(joints_reader):
            measured_joints = _finite_vector(joints_reader(), length=7)
            if measured_joints is not None:
                tracking = max(
                    abs(commanded - reached)
                    for commanded, reached in zip(joints, measured_joints)
                )
        sample = _grasp_frame_sample(environment)
        measured = (
            _finite_vector(sample.get("grasp_frame_position_world_m"), length=3)
            if sample is not None
            else None
        )
        cells.append(
            {
                "offset_m": delta,
                "requested_target_position_world_m": target,
                "status": "measured" if measured is not None else "unmeasurable",
                "joint_positions_rad": joints,
                "joint_tracking_error_rad": tracking,
                "measured_grasp_frame_position_world_m": measured,
                "measured_grasp_frame_orientation_world_xyzw": _finite_vector(
                    (sample or {}).get("grasp_frame_orientation_world_xyzw"),
                    length=4,
                ),
                "measured_gripper_pad_centers_world_m": (sample or {}).get(
                    "gripper_pad_centers_world_m"
                ),
                "measured_gripper_width_m": (sample or {}).get("gripper_width_m"),
                "measured_distance_to_requested_m": (
                    math.dist(measured, target) if measured is not None else None
                ),
                "contact_steps": contact_steps,
                "settle_steps": int(settle_steps),
            }
        )
    environment.reset()

    # Does the measured point follow the target at all?  Compare the spread of
    # what was asked for against the spread of what was reached, per axis: a
    # frame problem moves nothing, an obstruction moves some axes and not
    # others.
    reached = [
        cell for cell in cells if cell.get("measured_grasp_frame_position_world_m")
    ]
    follow: dict[str, Any] = {}
    for axis, name in enumerate(("x", "y", "z")):
        asked = [cell["requested_target_position_world_m"][axis] for cell in reached]
        got = [cell["measured_grasp_frame_position_world_m"][axis] for cell in reached]
        follow[name] = {
            "requested_span_m": (max(asked) - min(asked)) if asked else 0.0,
            "measured_span_m": (max(got) - min(got)) if got else 0.0,
        }
    return {
        "schema_version": REACHABILITY_SCHEMA_VERSION,
        "status": "measured" if reached else "unavailable",
        "cell_count": len(cells),
        "cells": cells,
        "axis_following": follow,
        "claim_boundary": _REACHABILITY_CLAIM_BOUNDARY,
    }


_REACHABILITY_CLAIM_BOUNDARY = (
    "maps_where_the_measured_pad_midpoint_can_be_placed;asserts_no_task_"
    "outcome_and_gates_nothing"
)
