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
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SWEEP_SCHEMA_VERSION = "native_task_arena_actuator_posture_sweep.v1"
CLOSE_POSTURE_SWEEP_SCHEMA_VERSION = (
    "native_task_arena_contact_close_posture_sweep.v1"
)
CONTACT_ACQUISITION_SWEEP_SCHEMA_VERSION = (
    "native_task_arena_contact_acquisition_sweep.v1"
)

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

# The authored close pose is the deepest point.  Search progressively farther
# behind it, where an open jaw can clear the door face before closing.  Five
# values on each of the approach, jaw, and lateral axes represent 125 physical
# cells in one loaded scene instead of 125 provider launches.
DEFAULT_CONTACT_APPROACH_OFFSETS_M: tuple[float, ...] = (
    -0.020,
    -0.015,
    -0.010,
    -0.005,
    0.0,
)
DEFAULT_CONTACT_TRANSVERSE_OFFSETS_M: tuple[float, ...] = (
    -0.012,
    -0.006,
    0.0,
    0.006,
    0.012,
)


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


def _quaternion_angle_xyzw(left: Sequence[float], right: Sequence[float]) -> float:
    a = _finite_vector(left, length=4)
    b = _finite_vector(right, length=4)
    if a is None or b is None:
        return math.inf
    a_norm = math.sqrt(sum(value * value for value in a))
    b_norm = math.sqrt(sum(value * value for value in b))
    if a_norm <= 0.0 or b_norm <= 0.0:
        return math.inf
    dot = abs(sum(x * y for x, y in zip(a, b)) / (a_norm * b_norm))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _task_pad_forces_n(sample: Mapping[str, Any] | None) -> dict[str, float]:
    native = sample.get("native_readback") if isinstance(sample, Mapping) else None
    instances = (
        (native.get("contact_sensor_instance_readback") or {}).get(
            "task_robot_contact"
        )
        if isinstance(native, Mapping)
        else None
    )
    peaks: dict[str, float] = {}
    for instance in instances or []:
        if not isinstance(instance, Mapping):
            continue
        for force in instance.get("nonzero_filter_forces") or []:
            if not isinstance(force, Mapping):
                continue
            path = str(force.get("filter_prim_path_expr") or "")
            side = next(
                (
                    name
                    for name in ("left_inner_finger", "right_inner_finger")
                    if name in path
                ),
                None,
            )
            if side is None:
                continue
            try:
                magnitude = float(force.get("force_magnitude_n") or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(magnitude) and magnitude >= 0.0:
                peaks[side] = max(peaks.get(side, 0.0), magnitude)
    return peaks


def _unit_vector(values: Any) -> list[float] | None:
    vector = _finite_vector(values, length=3)
    if vector is None:
        return None
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1.0e-9:
        return None
    return [value / norm for value in vector]


def _center_first(values: Sequence[float]) -> list[float]:
    return sorted((float(value) for value in values), key=lambda value: (abs(value), value))


def run_contact_acquisition_sweep(
    *,
    environment: Any,
    authored_target_position_world_m: Sequence[float],
    command_target_position_world_m: Sequence[float] | None = None,
    target_orientation_world_xyzw: Sequence[float],
    preposition_joint_positions_rad: Sequence[float],
    approach_axis_world: Sequence[float],
    jaw_axis_world: Sequence[float],
    lateral_axis_world: Sequence[float],
    gripper_open_command: Any,
    gripper_closed_command: Any,
    max_joint_delta_rad: Any,
    max_joint_setpoint_lead_rad: Any,
    arrival_tolerance_m: Any,
    orientation_tolerance_rad: Any,
    bilateral_contact_minimum_force_n: Any,
    approach_offsets_m: Sequence[float] = DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    jaw_offsets_m: Sequence[float] = DEFAULT_CONTACT_TRANSVERSE_OFFSETS_M,
    lateral_offsets_m: Sequence[float] = DEFAULT_CONTACT_TRANSVERSE_OFFSETS_M,
    preposition_steps: int = 30,
    advance_steps: int = 30,
    close_steps: int = 18,
    open_contact_trigger_required_steps: int = 1,
    open_contact_trigger_position_multiplier: Any = 2.0,
    bilateral_stability_steps: int = 2,
    stop_after_admitted_cells: int = 1,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Find a real bilateral grasp by separating open advance from closure.

    C53 held the qualified open pose while asking the gripper to close: the
    pose gate passed, but only the left finger touched.  C63 changed both arm
    posture and gripper state at once: the jaws reached their closed width
    before the arm reached the handle.  This sweep isolates those operations.
    The authored target remains the closed-gripper arrival authority.  The
    optional command target preserves the measured open-to-closed linkage
    compensation, so the open gripper starts behind that finish line instead
    of discarding the compensation before closure.  Every cell starts from the
    same measured open posture, advances while open,
    freezes the *reached* arm joints at the first task-pad contact or first
    pose-gate arrival, and only then closes.  If neither event occurs, the full
    bounded advance is still executed before closure.  Triggering closure does
    not admit a cell; the unchanged post-close pose, force, and stability gates
    remain decisive.

    The sweep records numeric telemetry only.  It does not grade the task and
    it does not weaken the final episode's native bilateral-contact gate.
    """

    target = _finite_vector(authored_target_position_world_m, length=3)
    command_target = _finite_vector(
        (
            authored_target_position_world_m
            if command_target_position_world_m is None
            else command_target_position_world_m
        ),
        length=3,
    )
    orientation = _finite_vector(target_orientation_world_xyzw, length=4)
    preposition = _finite_vector(preposition_joint_positions_rad, length=7)
    approach = _unit_vector(approach_axis_world)
    jaw = _unit_vector(jaw_axis_world)
    lateral = _unit_vector(lateral_axis_world)
    bounded = getattr(environment, "bounded_joint_action", None)
    pose_action = getattr(environment, "scripted_action_for_pose", None)
    if (
        target is None
        or command_target is None
        or orientation is None
        or preposition is None
        or approach is None
        or jaw is None
        or lateral is None
    ):
        raise ActuatorSweepError(["contact_acquisition_sweep_inputs_invalid"])
    if not callable(bounded) or not callable(pose_action):
        return {
            "schema_version": CONTACT_ACQUISITION_SWEEP_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "runtime_missing_joint_or_pose_action",
            "cells": [],
        }
    try:
        open_command = float(gripper_open_command)
        closed_command = float(gripper_closed_command)
        joint_delta = float(max_joint_delta_rad)
        setpoint_lead = float(max_joint_setpoint_lead_rad)
        arrival_tolerance = float(arrival_tolerance_m)
        orientation_tolerance = float(orientation_tolerance_rad)
        contact_threshold = float(bilateral_contact_minimum_force_n)
        open_trigger_required = int(open_contact_trigger_required_steps)
        open_trigger_position_multiplier = float(
            open_contact_trigger_position_multiplier
        )
        stability_required = int(bilateral_stability_steps)
        stop_after = int(stop_after_admitted_cells)
    except (TypeError, ValueError) as exc:
        raise ActuatorSweepError(
            ["contact_acquisition_sweep_scalar_inputs_invalid"]
        ) from exc
    scalars = (
        open_command,
        closed_command,
        joint_delta,
        setpoint_lead,
        arrival_tolerance,
        orientation_tolerance,
        contact_threshold,
        open_trigger_position_multiplier,
    )
    if (
        not all(math.isfinite(value) for value in scalars)
        or open_trigger_required < 1
        or open_trigger_position_multiplier < 1.0
        or stability_required < 1
        or stop_after < 1
    ):
        raise ActuatorSweepError(
            ["contact_acquisition_sweep_scalar_inputs_invalid"]
        )

    offsets: list[tuple[float, float, float]] = []
    try:
        for approach_offset in _center_first(approach_offsets_m):
            for jaw_offset in _center_first(jaw_offsets_m):
                for lateral_offset in _center_first(lateral_offsets_m):
                    if not all(
                        math.isfinite(value)
                        for value in (
                            approach_offset,
                            jaw_offset,
                            lateral_offset,
                        )
                    ):
                        raise ValueError("non-finite offset")
                    offsets.append(
                        (approach_offset, jaw_offset, lateral_offset)
                    )
    except (TypeError, ValueError) as exc:
        raise ActuatorSweepError(
            ["contact_acquisition_sweep_offsets_invalid"]
        ) from exc
    if not offsets:
        raise ActuatorSweepError(["contact_acquisition_sweep_offsets_invalid"])

    def _bounded_action(joints: Sequence[float], gripper: float) -> list[float]:
        return bounded(
            target_joint_positions_rad=list(joints),
            gripper_command=gripper,
            max_joint_delta_rad=joint_delta,
            max_joint_setpoint_lead_rad=setpoint_lead,
        )

    cells: list[dict[str, Any]] = []
    admitted_count = 0
    for cell_index, (approach_offset, jaw_offset, lateral_offset) in enumerate(
        offsets
    ):
        candidate_target = [
            target[axis]
            + approach_offset * approach[axis]
            + jaw_offset * jaw[axis]
            + lateral_offset * lateral[axis]
            for axis in range(3)
        ]
        candidate_command_target = [
            command_target[axis]
            + approach_offset * approach[axis]
            + jaw_offset * jaw[axis]
            + lateral_offset * lateral[axis]
            for axis in range(3)
        ]
        stage = "reset"
        try:
            environment.reset()
            stage = "preposition"
            for _ in range(max(1, int(preposition_steps))):
                environment.step(_bounded_action(preposition, open_command))

            stage = "open_advance"
            advance_sample: Mapping[str, Any] | None = None
            peak_advance_forces: dict[str, float] = {}
            open_contact_consecutive = 0
            maximum_consecutive_open_contact = 0
            open_bilateral_consecutive = 0
            maximum_consecutive_open_bilateral = 0
            open_contact_triggered = False
            open_pose_gate_triggered = False
            open_contact_trigger_step = None
            open_contact_trigger_forces: dict[str, float] = {}
            open_advance_trigger_reasons: list[str] = []
            executed_advance_steps = 0
            for advance_step_index in range(max(1, int(advance_steps))):
                action = pose_action(
                    target_position_world_m=candidate_command_target,
                    target_quaternion_world_xyzw=orientation,
                    gripper_command=open_command,
                    max_joint_delta_rad=joint_delta,
                    max_joint_setpoint_lead_rad=setpoint_lead,
                )
                environment.step(action)
                executed_advance_steps += 1
                advance_sample = _grasp_frame_sample(environment)
                advance_forces = _task_pad_forces_n(advance_sample)
                step_position = _finite_vector(
                    (advance_sample or {}).get(
                        "grasp_frame_position_world_m"
                    ),
                    length=3,
                )
                step_orientation = _finite_vector(
                    (advance_sample or {}).get(
                        "grasp_frame_orientation_world_xyzw"
                    ),
                    length=4,
                )
                pose_gate_reached = bool(
                    step_position is not None
                    and math.dist(step_position, candidate_command_target)
                    <= arrival_tolerance
                    and step_orientation is not None
                    and _quaternion_angle_xyzw(step_orientation, orientation)
                    <= orientation_tolerance
                )
                contact_trigger_pose_guard = bool(
                    step_position is not None
                    and math.dist(step_position, candidate_command_target)
                    <= arrival_tolerance * open_trigger_position_multiplier
                    and step_orientation is not None
                    and _quaternion_angle_xyzw(
                        step_orientation, orientation
                    )
                    <= orientation_tolerance
                )
                for side, magnitude in advance_forces.items():
                    peak_advance_forces[side] = max(
                        peak_advance_forces.get(side, 0.0), magnitude
                    )
                any_open_contact = contact_trigger_pose_guard and any(
                    advance_forces.get(side, 0.0) >= contact_threshold
                    for side in ("left_inner_finger", "right_inner_finger")
                )
                bilateral_open_contact = all(
                    advance_forces.get(side, 0.0) >= contact_threshold
                    for side in ("left_inner_finger", "right_inner_finger")
                )
                open_contact_consecutive = (
                    open_contact_consecutive + 1
                    if any_open_contact
                    else 0
                )
                maximum_consecutive_open_contact = max(
                    maximum_consecutive_open_contact,
                    open_contact_consecutive,
                )
                open_bilateral_consecutive = (
                    open_bilateral_consecutive + 1
                    if bilateral_open_contact
                    else 0
                )
                maximum_consecutive_open_bilateral = max(
                    maximum_consecutive_open_bilateral,
                    open_bilateral_consecutive,
                )
                if open_contact_consecutive >= open_trigger_required:
                    open_contact_triggered = True
                    open_contact_trigger_step = advance_step_index
                    open_contact_trigger_forces = dict(advance_forces)
                    open_advance_trigger_reasons.append(
                        "task_pad_contact"
                    )
                if pose_gate_reached:
                    open_pose_gate_triggered = True
                    open_advance_trigger_reasons.append("pose_gate")
                if open_contact_triggered or open_pose_gate_triggered:
                    break

            reached_open = _finite_vector(
                environment.read_arm_joint_positions(), length=7
            )
            advance_position = _finite_vector(
                (advance_sample or {}).get("grasp_frame_position_world_m"),
                length=3,
            )
            advance_orientation = _finite_vector(
                (advance_sample or {}).get(
                    "grasp_frame_orientation_world_xyzw"
                ),
                length=4,
            )
            if reached_open is None or advance_position is None:
                raise RuntimeError("open_advance_readback_missing")

            stage = "close_hold"
            consecutive_bilateral = 0
            maximum_consecutive_bilateral = 0
            peak_close_forces: dict[str, float] = {}
            close_gripper_widths: list[float] = []
            first_bilateral_close_step = None
            last_bilateral_close_step = None
            close_phase_gate_triggered = False
            executed_close_steps = 0
            terminal_sample: Mapping[str, Any] | None = None
            for close_step_index in range(max(1, int(close_steps))):
                environment.step(_bounded_action(reached_open, closed_command))
                executed_close_steps += 1
                terminal_sample = _grasp_frame_sample(environment)
                measured_width = (terminal_sample or {}).get(
                    "gripper_width_m"
                )
                if isinstance(measured_width, (int, float)) and math.isfinite(
                    float(measured_width)
                ):
                    close_gripper_widths.append(float(measured_width))
                forces = _task_pad_forces_n(terminal_sample)
                for side, magnitude in forces.items():
                    peak_close_forces[side] = max(
                        peak_close_forces.get(side, 0.0), magnitude
                    )
                bilateral = all(
                    forces.get(side, 0.0) >= contact_threshold
                    for side in ("left_inner_finger", "right_inner_finger")
                )
                consecutive_bilateral = (
                    consecutive_bilateral + 1 if bilateral else 0
                )
                if bilateral:
                    if first_bilateral_close_step is None:
                        first_bilateral_close_step = close_step_index
                    last_bilateral_close_step = close_step_index
                maximum_consecutive_bilateral = max(
                    maximum_consecutive_bilateral,
                    consecutive_bilateral,
                )
                close_position = _finite_vector(
                    (terminal_sample or {}).get(
                        "grasp_frame_position_world_m"
                    ),
                    length=3,
                )
                close_orientation = _finite_vector(
                    (terminal_sample or {}).get(
                        "grasp_frame_orientation_world_xyzw"
                    ),
                    length=4,
                )
                close_pose_gate = bool(
                    close_position is not None
                    and math.dist(close_position, candidate_target)
                    <= arrival_tolerance
                    and close_orientation is not None
                    and _quaternion_angle_xyzw(
                        close_orientation, orientation
                    )
                    <= orientation_tolerance
                )
                if (
                    close_pose_gate
                    and consecutive_bilateral >= stability_required
                ):
                    close_phase_gate_triggered = True
                    break

            terminal_position = _finite_vector(
                (terminal_sample or {}).get("grasp_frame_position_world_m"),
                length=3,
            )
            terminal_orientation = _finite_vector(
                (terminal_sample or {}).get(
                    "grasp_frame_orientation_world_xyzw"
                ),
                length=4,
            )
            terminal_forces = _task_pad_forces_n(terminal_sample)
            terminal_joints = _finite_vector(
                environment.read_arm_joint_positions(), length=7
            )
            terminal_bilateral = all(
                terminal_forces.get(side, 0.0) >= contact_threshold
                for side in ("left_inner_finger", "right_inner_finger")
            )
            orientation_error = (
                _quaternion_angle_xyzw(terminal_orientation, orientation)
                if terminal_orientation is not None
                else None
            )
            candidate_distance = (
                math.dist(terminal_position, candidate_target)
                if terminal_position is not None
                else None
            )
            admitted = bool(
                terminal_position is not None
                and candidate_distance is not None
                and candidate_distance <= arrival_tolerance
                and terminal_bilateral
                and maximum_consecutive_bilateral >= stability_required
                and orientation_error is not None
                and orientation_error <= orientation_tolerance
            )
            if admitted:
                admitted_count += 1
            cells.append(
                {
                    "cell_index": cell_index,
                    "approach_offset_m": approach_offset,
                    "jaw_offset_m": jaw_offset,
                    "lateral_offset_m": lateral_offset,
                    "candidate_target_position_world_m": candidate_target,
                    "candidate_command_target_position_world_m": (
                        candidate_command_target
                    ),
                    "reached_open_joint_positions_rad": reached_open,
                    "executed_open_advance_steps": executed_advance_steps,
                    "open_contact_trigger_required_steps": (
                        open_trigger_required
                    ),
                    "open_contact_trigger_position_tolerance_m": (
                        arrival_tolerance
                        * open_trigger_position_multiplier
                    ),
                    "open_contact_triggered": open_contact_triggered,
                    "open_pose_gate_triggered": open_pose_gate_triggered,
                    "open_advance_trigger_reasons": (
                        open_advance_trigger_reasons
                    ),
                    "open_contact_trigger_step_index": (
                        open_contact_trigger_step
                    ),
                    "open_contact_trigger_pad_forces_n": (
                        open_contact_trigger_forces
                    ),
                    "maximum_consecutive_open_contact_steps": (
                        maximum_consecutive_open_contact
                    ),
                    "maximum_consecutive_open_bilateral_steps": (
                        maximum_consecutive_open_bilateral
                    ),
                    "advance_position_world_m": advance_position,
                    "advance_position_error_m": math.dist(
                        advance_position, candidate_command_target
                    ),
                    "advance_distance_to_arrival_target_m": math.dist(
                        advance_position, candidate_target
                    ),
                    "advance_orientation_error_rad": (
                        _quaternion_angle_xyzw(
                            advance_orientation, orientation
                        )
                        if advance_orientation is not None
                        else None
                    ),
                    "peak_open_advance_pad_forces_n": peak_advance_forces,
                    "terminal_open_advance_pad_forces_n": (
                        _task_pad_forces_n(advance_sample)
                    ),
                    "terminal_open_advance_gripper_width_m": (
                        (advance_sample or {}).get("gripper_width_m")
                    ),
                    "terminal_position_world_m": terminal_position,
                    "terminal_distance_to_candidate_target_m": (
                        candidate_distance
                    ),
                    "terminal_distance_to_authored_target_m": (
                        math.dist(terminal_position, target)
                        if terminal_position is not None
                        else None
                    ),
                    "terminal_orientation_error_rad": orientation_error,
                    "terminal_task_contact_pad_forces_n": terminal_forces,
                    "terminal_reached_joint_positions_rad": terminal_joints,
                    "terminal_maximum_joint_drift_from_frozen_rad": (
                        max(
                            abs(terminal - frozen)
                            for terminal, frozen in zip(
                                terminal_joints, reached_open
                            )
                        )
                        if terminal_joints is not None
                        else None
                    ),
                    "terminal_grasp_frame_shift_from_open_m": (
                        math.dist(terminal_position, advance_position)
                        if terminal_position is not None
                        else None
                    ),
                    "peak_close_pad_forces_n": peak_close_forces,
                    "commanded_close_gripper_value": closed_command,
                    "executed_close_steps": executed_close_steps,
                    "close_phase_gate_triggered": (
                        close_phase_gate_triggered
                    ),
                    "minimum_close_gripper_width_m": (
                        min(close_gripper_widths)
                        if close_gripper_widths
                        else None
                    ),
                    "maximum_close_gripper_width_m": (
                        max(close_gripper_widths)
                        if close_gripper_widths
                        else None
                    ),
                    "first_bilateral_close_step_index": (
                        first_bilateral_close_step
                    ),
                    "last_bilateral_close_step_index": (
                        last_bilateral_close_step
                    ),
                    "maximum_consecutive_bilateral_steps": (
                        maximum_consecutive_bilateral
                    ),
                    "terminal_bilateral_task_contact_active": (
                        terminal_bilateral
                    ),
                    "terminal_gripper_width_m": (
                        (terminal_sample or {}).get("gripper_width_m")
                    ),
                    "admitted": admitted,
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate one physical cell
            cells.append(
                {
                    "cell_index": cell_index,
                    "approach_offset_m": approach_offset,
                    "jaw_offset_m": jaw_offset,
                    "lateral_offset_m": lateral_offset,
                    "candidate_target_position_world_m": candidate_target,
                    "candidate_command_target_position_world_m": (
                        candidate_command_target
                    ),
                    "status": "cell_error",
                    "error": f"{stage}:{type(exc).__name__}:{exc}",
                    "admitted": False,
                }
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "schema_version": CONTACT_ACQUISITION_SWEEP_SCHEMA_VERSION,
                    "status": "running",
                    "represented_cell_count": len(offsets),
                    "executed_cell_count": len(cells),
                    "early_stop_after_admitted_cells": stop_after,
                    "admitted_cell_count": admitted_count,
                    "cells": list(cells),
                    "last_cell": dict(cells[-1]),
                    "claim_boundary": (
                        "incremental_numeric_measurement_only;not_task_success"
                    ),
                }
            )
        if admitted_count >= stop_after:
            break

    admitted_cells = [cell for cell in cells if cell.get("admitted") is True]
    measurable = [
        cell
        for cell in cells
        if cell.get("terminal_distance_to_authored_target_m") is not None
    ]
    best = min(
        admitted_cells or measurable,
        key=lambda cell: (
            cell.get("admitted") is not True,
            -int(cell.get("maximum_consecutive_bilateral_steps") or 0),
            float(cell.get("terminal_distance_to_authored_target_m") or math.inf),
            abs(float(cell.get("approach_offset_m") or 0.0)),
            abs(float(cell.get("jaw_offset_m") or 0.0))
            + abs(float(cell.get("lateral_offset_m") or 0.0)),
        ),
        default=None,
    )
    cleanup_error = None
    try:
        environment.reset()
    except Exception as exc:  # noqa: BLE001 - preserve completed measurements
        cleanup_error = f"final_reset:{type(exc).__name__}:{exc}"
    report = {
        "schema_version": CONTACT_ACQUISITION_SWEEP_SCHEMA_VERSION,
        "status": "measured" if cells else "unavailable",
        "represented_cell_count": len(offsets),
        "executed_cell_count": len(cells),
        "early_stop_after_admitted_cells": stop_after,
        "admitted_cell_count": len(admitted_cells),
        "cells": cells,
        "best_cell": best,
        "claim_boundary": (
            "numeric_reset_isolated_open_advance_then_close_hold_search;"
            "only_the_following_deterministic_controls_episode_asserts_task_"
            "success"
        ),
    }
    if cleanup_error is not None:
        report["cleanup_error"] = cleanup_error
    if progress_callback is not None:
        progress_callback(report)
    return report


def run_contact_close_posture_sweep(
    *,
    environment: Any,
    target_position_world_m: Sequence[float],
    target_orientation_world_xyzw: Sequence[float],
    postures: Sequence[Mapping[str, Any]],
    preposition_joint_positions_rad: Sequence[float],
    gripper_open_command: Any,
    gripper_closed_command: Any,
    max_joint_delta_rad: Any,
    max_joint_setpoint_lead_rad: Any,
    arrival_tolerance_m: Any,
    orientation_tolerance_rad: Any,
    bilateral_contact_minimum_force_n: Any,
    preposition_steps: int = DEFAULT_CELL_SETTLE_STEPS,
    settle_steps: int = DEFAULT_CELL_SETTLE_STEPS,
) -> dict[str, Any]:
    """Physically measure every solved close branch before choosing one.

    Off-sim continuity can prefer a mathematically short chain whose TCP is
    extremely sensitive to a tiny tracking residual.  Each cell starts from
    the same measured open posture, closes on one solved branch, and seals the
    full selected -> commanded -> reached -> FK -> measured -> pad-force chain.
    """

    target = _finite_vector(target_position_world_m, length=3)
    orientation = _finite_vector(target_orientation_world_xyzw, length=4)
    preposition = _finite_vector(preposition_joint_positions_rad, length=7)
    bounded = getattr(environment, "bounded_joint_action", None)
    if target is None or orientation is None or preposition is None or not postures:
        raise ActuatorSweepError(["contact_close_posture_sweep_inputs_invalid"])
    if not callable(bounded):
        return {
            "schema_version": CLOSE_POSTURE_SWEEP_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "runtime_missing_bounded_action",
            "cells": [],
        }
    try:
        open_command = float(gripper_open_command)
        closed_command = float(gripper_closed_command)
        joint_delta = float(max_joint_delta_rad)
        setpoint_lead = float(max_joint_setpoint_lead_rad)
        position_tolerance = float(arrival_tolerance_m)
        orientation_tolerance = float(orientation_tolerance_rad)
        contact_threshold = float(bilateral_contact_minimum_force_n)
    except (TypeError, ValueError) as exc:
        raise ActuatorSweepError(
            ["contact_close_posture_sweep_scalar_inputs_invalid"]
        ) from exc
    if not all(
        math.isfinite(value)
        for value in (
            open_command,
            closed_command,
            joint_delta,
            setpoint_lead,
            position_tolerance,
            orientation_tolerance,
            contact_threshold,
        )
    ):
        raise ActuatorSweepError(
            ["contact_close_posture_sweep_scalar_inputs_invalid"]
        )

    def _command(joints: Sequence[float], gripper: float) -> None:
        action = bounded(
            target_joint_positions_rad=list(joints),
            gripper_command=gripper,
            max_joint_delta_rad=joint_delta,
            max_joint_setpoint_lead_rad=setpoint_lead,
        )
        # The bounded-action seam already validates and canonicalizes every
        # action component.  Converting it a second time here hid C57's failing
        # cell behind an unlocated ``float(None)`` error.
        environment.step(action)

    cells: list[dict[str, Any]] = []
    for cell_index, posture in enumerate(postures):
        commanded = _finite_vector(posture.get("joint_positions_rad"), length=7)
        if commanded is None:
            continue
        stage = "reset"
        try:
            environment.reset()
            stage = "preposition"
            for _ in range(max(1, int(preposition_steps))):
                _command(preposition, open_command)
            bilateral_steps = 0
            peak_pad_forces: dict[str, float] = {}
            terminal_sample: Mapping[str, Any] | None = None
            stage = "close"
            for _ in range(max(1, int(settle_steps))):
                _command(commanded, closed_command)
                terminal_sample = _grasp_frame_sample(environment)
                forces = _task_pad_forces_n(terminal_sample)
                for side, magnitude in forces.items():
                    peak_pad_forces[side] = max(
                        peak_pad_forces.get(side, 0.0), magnitude
                    )
                if all(
                    forces.get(side, 0.0) >= contact_threshold
                    for side in ("left_inner_finger", "right_inner_finger")
                ):
                    bilateral_steps += 1
            stage = "terminal_readback"
            reached = _finite_vector(
                environment.read_arm_joint_positions(), length=7
            )
            terminal_sample = terminal_sample or _grasp_frame_sample(environment)
            # C58 proved that isolating only reset/step/readback is not enough:
            # a later conversion or derived measurement can still erase every
            # branch through the caller's outer diagnostic catch.  Keep the
            # complete cell lifecycle inside the same isolation boundary.
            stage = "measurement"
            measured = _finite_vector(
                (terminal_sample or {}).get("grasp_frame_position_world_m"),
                length=3,
            )
            measured_orientation = _finite_vector(
                (terminal_sample or {}).get(
                    "grasp_frame_orientation_world_xyzw"
                ),
                length=4,
            )
            terminal_forces = _task_pad_forces_n(terminal_sample)
            predictor = getattr(
                environment, "predict_grasp_frame_pose_world", None
            )
            predicted = None
            if reached is not None and callable(predictor):
                try:
                    predicted = _finite_vector(
                        predictor(reached, gripper_command=closed_command),
                        length=7,
                    )
                except Exception:  # noqa: BLE001 - retain an explicit FK gap
                    predicted = None
            distance = (
                math.dist(measured, target) if measured is not None else None
            )
            orientation_error = (
                _quaternion_angle_xyzw(measured_orientation, orientation)
                if measured_orientation is not None
                else None
            )
            admitted = bool(
                distance is not None
                and distance <= position_tolerance
                and orientation_error is not None
                and orientation_error <= orientation_tolerance
                and all(
                    terminal_forces.get(side, 0.0) >= contact_threshold
                    for side in ("left_inner_finger", "right_inner_finger")
                )
            )
            cells.append(
                {
                    **{
                        key: posture.get(key)
                        for key in (
                            "posture_index",
                            "seed_index",
                            "offsim_position_error_m",
                            "minimum_joint_limit_margin_rad",
                        )
                    },
                    "commanded_joint_positions_rad": commanded,
                    "reached_joint_positions_rad": reached,
                    "commanded_to_reached_joint_l2_rad": (
                        math.sqrt(
                            sum(
                                (a - b) ** 2
                                for a, b in zip(commanded, reached)
                            )
                        )
                        if reached is not None
                        else None
                    ),
                    "predicted_grasp_frame_pose_world": predicted,
                    "fk_to_measured_tcp_error_m": (
                        math.dist(predicted[:3], measured)
                        if predicted is not None and measured is not None
                        else None
                    ),
                    "measured_grasp_frame_position_world_m": measured,
                    "measured_grasp_frame_orientation_world_xyzw": (
                        measured_orientation
                    ),
                    "measured_distance_to_target_m": distance,
                    "measured_orientation_error_rad": orientation_error,
                    "terminal_task_contact_pad_forces_n": terminal_forces,
                    "peak_task_contact_pad_forces_n": peak_pad_forces,
                    "bilateral_contact_steps": bilateral_steps,
                    "admitted": admitted,
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate one physical branch
            cells.append(
                {
                    "posture_index": posture.get("posture_index"),
                    "seed_index": posture.get("seed_index"),
                    "commanded_joint_positions_rad": commanded,
                    "status": "cell_error",
                    "error": f"{stage}:{type(exc).__name__}:{exc}",
                    "admitted": False,
                    "measured_distance_to_target_m": None,
                }
            )
            continue
    admitted_cells = [cell for cell in cells if cell["admitted"]]
    measurable = [
        cell for cell in cells if cell["measured_distance_to_target_m"] is not None
    ]
    best = min(
        admitted_cells or measurable,
        key=lambda cell: (
            cell["measured_distance_to_target_m"],
            cell.get("commanded_to_reached_joint_l2_rad") or math.inf,
        ),
        default=None,
    )
    cleanup_error = None
    try:
        environment.reset()
    except Exception as exc:  # noqa: BLE001 - preserve completed measurements
        cleanup_error = f"final_reset:{type(exc).__name__}:{exc}"
    report = {
        "schema_version": CLOSE_POSTURE_SWEEP_SCHEMA_VERSION,
        "status": "measured" if cells else "unavailable",
        "cell_count": len(cells),
        "cells": cells,
        "best_cell": best,
        "admitted_cell_count": len(admitted_cells),
        "claim_boundary": (
            "physics_measures_each_solved_close_branch;only_the_following_"
            "deterministic_controls_episode_asserts_task_success"
        ),
    }
    if cleanup_error is not None:
        report["cleanup_error"] = cleanup_error
    return report


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
                    "predicted_grasp_frame_position_world_m": row.get(
                        "predicted_grasp_frame_position_world_m"
                    ),
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
                predicted = _finite_vector(
                    posture.get("predicted_grasp_frame_position_world_m"), length=3
                )
                cells.append(
                    {
                        "wrist_stiffness_nm_per_rad": float(stiffness),
                        "wrist_damping_nm_per_rad_s": float(damping),
                        "posture_index": posture.get("posture_index"),
                        "seed_index": posture.get("seed_index"),
                        "offsim_position_error_m": posture.get("offsim_position_error_m"),
                        # What the solver believed, minus what physics did, at
                        # the same joints.  Gains, branch, posture and
                        # obstruction are all ruled out by measurement as
                        # causes of the recurring ~13 mm; this is what is left,
                        # and it has been inferred by subtracting two error
                        # magnitudes rather than measured as a vector.
                        "predicted_grasp_frame_position_world_m": predicted,
                        "measured_minus_model_m": (
                            [measured[axis] - predicted[axis] for axis in range(3)]
                            if measured is not None and predicted is not None
                            else None
                        ),
                        "measured_minus_model_distance_m": (
                            math.dist(measured, predicted)
                            if measured is not None and predicted is not None
                            else None
                        ),
                        "joint_tracking_error_rad": (
                            max(abs(a - b) for a, b in zip(joints, observed))
                            if observed is not None
                            else None
                        ),
                        # C43 left one binary unsettled for want of this: the
                        # solver moved its predicted fingertip 1.90 mm across
                        # four postures and physics moved 0.24 mm, a slope of
                        # -0.88 that eats every correction the calibration
                        # makes.  Either the arm does not differentiate the
                        # commands, or it does and the two frames disagree.
                        # The worst single joint cannot tell those apart; the
                        # whole vector can.
                        "commanded_joint_positions_rad": list(joints),
                        "measured_joint_positions_rad": (
                            list(observed) if observed is not None else None
                        ),
                        "joint_tracking_residual_rad": (
                            [a - b for a, b in zip(joints, observed)]
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
    "run_contact_close_posture_sweep",
    "CLOSE_POSTURE_SWEEP_SCHEMA_VERSION",
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
    preposition_target_position_world_m: Sequence[float] | None = None,
    preposition_settle_steps: int = DEFAULT_CELL_SETTLE_STEPS,
    abort_contact_force_n: float | None = None,
    stop_after_first_contact_cell: bool = False,
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

    preposition_target = (
        _finite_vector(preposition_target_position_world_m, length=3)
        if preposition_target_position_world_m is not None
        else None
    )
    if (
        preposition_target_position_world_m is not None
        and preposition_target is None
    ):
        raise ActuatorSweepError(["reachability_probe_preposition_invalid"])
    try:
        abort_force = (
            None
            if abort_contact_force_n is None
            else float(abort_contact_force_n)
        )
    except (TypeError, ValueError) as exc:
        raise ActuatorSweepError(
            ["reachability_probe_abort_force_invalid"]
        ) from exc
    if abort_force is not None and (
        not math.isfinite(abort_force) or abort_force <= 0.0
    ):
        raise ActuatorSweepError(["reachability_probe_abort_force_invalid"])
    preposition_joints = (
        _finite_vector(solve(preposition_target, list(seed)), length=7)
        if preposition_target is not None
        else None
    )
    if preposition_target is not None and preposition_joints is None:
        return {
            "schema_version": REACHABILITY_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "preposition_unsolved",
            "cells": [],
            "claim_boundary": _REACHABILITY_CLAIM_BOUNDARY,
        }

    def _command(joints: Sequence[float]) -> None:
        environment.step(
            [
                float(value)
                for value in bounded(
                    target_joint_positions_rad=joints,
                    gripper_command=float(gripper_open_command),
                    max_joint_delta_rad=float(max_joint_delta_rad),
                    max_joint_setpoint_lead_rad=float(
                        max_joint_setpoint_lead_rad
                    ),
                )
            ]
        )

    def _contact_measurement() -> tuple[bool, float, dict[str, float]]:
        reader = getattr(environment, "read_task_sample", None)
        if not callable(reader):
            return False, 0.0, {}
        try:
            sample = reader() or {}
        except Exception:  # noqa: BLE001 - diagnostic contact is optional
            return False, 0.0, {}
        active = sample.get("task_contact_active") is True
        try:
            peak = float(sample.get("task_robot_contact_peak_force_n") or 0.0)
        except (TypeError, ValueError):
            peak = 0.0
        pad_forces: dict[str, float] = {}
        native = sample.get("native_readback")
        instances = (
            (native.get("contact_sensor_instance_readback") or {}).get(
                "task_robot_contact"
            )
            if isinstance(native, Mapping)
            else None
        )
        for instance in instances or []:
            if not isinstance(instance, Mapping):
                continue
            for force in instance.get("nonzero_filter_forces") or []:
                if not isinstance(force, Mapping):
                    continue
                path = str(force.get("filter_prim_path_expr") or "")
                side = next(
                    (
                        name
                        for name in (
                            "left_inner_finger",
                            "right_inner_finger",
                        )
                        if name in path
                    ),
                    path or "unattributed",
                )
                try:
                    magnitude = float(force.get("force_magnitude_n") or 0.0)
                except (TypeError, ValueError):
                    continue
                pad_forces[side] = max(
                    pad_forces.get(side, 0.0), magnitude
                )
        return active, peak, pad_forces

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
        if preposition_joints is not None:
            for _ in range(max(1, int(preposition_settle_steps))):
                _command(preposition_joints)
        contact_steps = 0
        peak_contact_force_n = 0.0
        peak_pad_contact_forces_n: dict[str, float] = {}
        aborted_on_contact_force = False
        executed_steps = 0
        for _ in range(max(1, int(settle_steps))):
            _command(joints)
            executed_steps += 1
            active, peak_force, pad_forces = _contact_measurement()
            if active:
                contact_steps += 1
            peak_contact_force_n = max(peak_contact_force_n, peak_force)
            for side, magnitude in pad_forces.items():
                peak_pad_contact_forces_n[side] = max(
                    peak_pad_contact_forces_n.get(side, 0.0), magnitude
                )
            if abort_force is not None and peak_force >= abort_force:
                aborted_on_contact_force = True
                break
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
                "peak_task_contact_force_n": peak_contact_force_n,
                "peak_pad_contact_forces_n": peak_pad_contact_forces_n,
                "aborted_on_contact_force": aborted_on_contact_force,
                "executed_steps": executed_steps,
                "settle_steps": int(settle_steps),
            }
        )
        if stop_after_first_contact_cell and contact_steps > 0:
            break
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
        "preposition_target_position_world_m": preposition_target,
        "preposition_joint_positions_rad": preposition_joints,
        "abort_contact_force_n": abort_force,
        "stop_after_first_contact_cell": bool(stop_after_first_contact_cell),
        "stopped_after_first_contact_cell": bool(
            stop_after_first_contact_cell
            and cells
            and int(cells[-1].get("contact_steps") or 0) > 0
        ),
        "cells": cells,
        "axis_following": follow,
        "claim_boundary": _REACHABILITY_CLAIM_BOUNDARY,
    }


_REACHABILITY_CLAIM_BOUNDARY = (
    "maps_where_the_measured_pad_midpoint_can_be_placed;asserts_no_task_"
    "outcome_and_gates_nothing"
)
