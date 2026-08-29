"""Deterministic CPU feasibility for task-authored gripper orientations.

A robot base pose fixes more than which tool positions the effector can reach:
it fixes which tool *orientations* the arm can slew to inside a phase's step
budget.  The world-frame rest grasp orientation is ``base * rest_grasp_base``,
so the required rotation to an authored tool pose is a function of the base
orientation and the robot profile alone.  That makes it pure quaternion algebra
-- decidable locally, in microseconds, before any provider allocation.

This module exists because the analytic placement gates scored position and
support only.  Eleven paid Arena allocations on scene 839873 were each accepted
analytically and then rejected natively with ``native_task_phase_ik_unreached``
on every phase, across base poses spanning 57.9%-93.6% of arm span, because the
authored planar-push orientation demanded a 180 degree wrist slew that no base
*position* could shorten.  Base *yaw* can: the same task needs only 90 degrees
at yaw 0.  Both facts are computed here.

Robot-specific numbers (the rest grasp frame, the achievable slew rate) live on
:class:`~blueprint_pipeline.scene_placement.robot_profile.RobotProfile`, so this
generalizes to any embodiment, site, and task without edit.

Stdlib only: safe to ship inside provider bundles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "task_evaluation_robot_placement_orientation_feasibility.v1"

#: Blocker emitted when a phase's authored orientation cannot be slewed to
#: inside its step budget.  Suffixed with ``:<phase_id>``.
SLEW_BUDGET_BLOCKER = "robot_placement_orientation_slew_exceeds_phase_budget"

#: Default number of yaw samples for :func:`solve_base_yaw_for_orientation`.
#: 720 samples is 0.5 degree resolution, far finer than any achievable base
#: placement tolerance.
DEFAULT_YAW_SAMPLE_COUNT = 720


class RobotPlacementOrientationError(ValueError):
    """The supplied orientation contract cannot support a placement decision."""


def _quaternion(value: Any, *, field: str) -> tuple[float, float, float, float]:
    """Validate and normalize an XYZW quaternion, failing closed."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RobotPlacementOrientationError(
            f"robot_placement_orientation_quaternion_invalid:{field}"
        )
    if len(value) != 4:
        raise RobotPlacementOrientationError(
            f"robot_placement_orientation_quaternion_invalid:{field}"
        )
    try:
        parts = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RobotPlacementOrientationError(
            f"robot_placement_orientation_quaternion_invalid:{field}"
        ) from exc
    if not all(math.isfinite(item) for item in parts):
        raise RobotPlacementOrientationError(
            f"robot_placement_orientation_quaternion_invalid:{field}"
        )
    norm = math.sqrt(sum(item * item for item in parts))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RobotPlacementOrientationError(
            f"robot_placement_orientation_quaternion_invalid:{field}"
        )
    x, y, z, w = (item / norm for item in parts)
    return (x, y, z, w)


def quaternion_multiply_xyzw(
    left: Sequence[float], right: Sequence[float]
) -> list[float]:
    """Hamilton product of two XYZW quaternions."""

    x1, y1, z1, w1 = _quaternion(left, field="left")
    x2, y2, z2, w2 = _quaternion(right, field="right")
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def quaternion_angle_rad(left: Sequence[float], right: Sequence[float]) -> float:
    """Shortest geodesic angle between two XYZW orientations.

    ``q`` and ``-q`` denote the same rotation, so the dot product is taken in
    absolute value: a sign flip must never read as a pi difference.
    """

    a = _quaternion(left, field="left")
    b = _quaternion(right, field="right")
    dot = abs(sum(i * j for i, j in zip(a, b)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def world_rest_grasp_orientation_xyzw(
    *,
    base_orientation_xyzw: Sequence[float],
    rest_grasp_orientation_base_xyzw: Sequence[float],
) -> list[float]:
    """World orientation of the grasp frame at the robot's reset joint pose."""

    return quaternion_multiply_xyzw(
        _quaternion(base_orientation_xyzw, field="base_orientation_xyzw"),
        _quaternion(
            rest_grasp_orientation_base_xyzw,
            field="rest_grasp_orientation_base_xyzw",
        ),
    )


def required_orientation_slew_rad(
    *,
    base_orientation_xyzw: Sequence[float],
    rest_grasp_orientation_base_xyzw: Sequence[float],
    target_orientation_xyzw: Sequence[float],
) -> float:
    """Wrist rotation the arm must achieve to reach one authored tool pose."""

    world_rest = world_rest_grasp_orientation_xyzw(
        base_orientation_xyzw=base_orientation_xyzw,
        rest_grasp_orientation_base_xyzw=rest_grasp_orientation_base_xyzw,
    )
    target = _quaternion(target_orientation_xyzw, field="target_orientation_xyzw")
    return quaternion_angle_rad(world_rest, target)


def _budget(maximum_steps_per_phase: Any, orientation_slew_rad_per_step: Any) -> tuple[int, float]:
    try:
        steps = int(maximum_steps_per_phase)
        rate = float(orientation_slew_rad_per_step)
    except (TypeError, ValueError) as exc:
        raise RobotPlacementOrientationError(
            "robot_placement_orientation_budget_invalid"
        ) from exc
    if steps <= 0 or not math.isfinite(rate) or rate <= 0.0:
        raise RobotPlacementOrientationError(
            "robot_placement_orientation_budget_invalid"
        )
    return steps, rate


def _phase_orientation(phase: Any, index: int) -> tuple[str, tuple[float, float, float, float]]:
    if not isinstance(phase, Mapping):
        raise RobotPlacementOrientationError(
            "robot_placement_orientation_phase_invalid"
        )
    phase_id = str(phase.get("phase_id") or f"phase_{index:02d}")
    if "orientation_world_xyzw" not in phase:
        raise RobotPlacementOrientationError(
            f"robot_placement_orientation_phase_missing_orientation:{phase_id}"
        )
    quaternion = _quaternion(
        phase.get("orientation_world_xyzw"), field=f"phase:{phase_id}"
    )
    return phase_id, quaternion


def evaluate_orientation_slew_feasibility(
    *,
    base_orientation_xyzw: Sequence[float],
    rest_grasp_orientation_base_xyzw: Sequence[float],
    phases: Sequence[Mapping[str, Any]],
    maximum_steps_per_phase: int,
    orientation_slew_rad_per_step: float,
) -> dict[str, Any]:
    """Decide, on CPU, whether every authored tool orientation is slewable.

    The report carries per-phase required rotation, the step count that implies,
    and the fraction of the phase budget it consumes -- a gradient the placement
    agent can rank candidates by, not merely a pass/fail bit.
    """

    steps, rate = _budget(maximum_steps_per_phase, orientation_slew_rad_per_step)
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        raise RobotPlacementOrientationError(
            "robot_placement_orientation_phase_invalid"
        )
    if not phases:
        raise RobotPlacementOrientationError(
            "robot_placement_orientation_phase_plan_empty"
        )

    base = _quaternion(base_orientation_xyzw, field="base_orientation_xyzw")
    rest = _quaternion(
        rest_grasp_orientation_base_xyzw, field="rest_grasp_orientation_base_xyzw"
    )

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    worst_slew = 0.0
    worst_phase_id: str | None = None

    for index, phase in enumerate(phases):
        phase_id, target = _phase_orientation(phase, index)
        slew = required_orientation_slew_rad(
            base_orientation_xyzw=base,
            rest_grasp_orientation_base_xyzw=rest,
            target_orientation_xyzw=target,
        )
        required_steps = int(math.ceil(slew / rate))
        feasible = required_steps <= steps
        rows.append(
            {
                "phase_id": phase_id,
                "required_slew_rad": slew,
                "required_steps": required_steps,
                "step_budget": steps,
                "budget_utilization": required_steps / steps,
                "feasible": feasible,
            }
        )
        if not feasible:
            blockers.append(f"{SLEW_BUDGET_BLOCKER}:{phase_id}")
        if slew > worst_slew:
            worst_slew = slew
            worst_phase_id = phase_id

    return {
        "schema_version": SCHEMA_VERSION,
        "feasible": not blockers,
        "blockers": blockers,
        "phases": rows,
        "worst_phase_id": worst_phase_id,
        "worst_required_slew_rad": worst_slew,
        "orientation_slew_rad_per_step": rate,
        "maximum_steps_per_phase": steps,
    }


def solve_base_yaw_for_orientation(
    *,
    rest_grasp_orientation_base_xyzw: Sequence[float],
    phases: Sequence[Mapping[str, Any]],
    maximum_steps_per_phase: int,
    orientation_slew_rad_per_step: float,
    yaw_sample_count: int = DEFAULT_YAW_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Choose the base yaw that minimizes the worst authored-phase slew.

    Base yaw rotates the whole arm workspace, so it is the one placement degree
    of freedom that changes the required wrist rotation.  Callers must still
    intersect the returned yaw with reach and facing constraints -- this reports
    the orientation axis of that joint problem, it does not decide placement on
    its own.
    """

    steps, rate = _budget(maximum_steps_per_phase, orientation_slew_rad_per_step)
    try:
        samples = int(yaw_sample_count)
    except (TypeError, ValueError) as exc:
        raise RobotPlacementOrientationError(
            "robot_placement_orientation_yaw_sample_count_invalid"
        ) from exc
    if samples < 1:
        raise RobotPlacementOrientationError(
            "robot_placement_orientation_yaw_sample_count_invalid"
        )

    rest = _quaternion(
        rest_grasp_orientation_base_xyzw, field="rest_grasp_orientation_base_xyzw"
    )
    # Validate the plan once so a malformed phase fails before the sweep.
    targets = [_phase_orientation(phase, index) for index, phase in enumerate(phases)]
    if not targets:
        raise RobotPlacementOrientationError(
            "robot_placement_orientation_phase_plan_empty"
        )

    best_yaw = 0.0
    best_worst = math.inf
    feasible_yaws: list[float] = []

    for sample in range(samples):
        yaw = 2.0 * math.pi * sample / samples
        base = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        worst = 0.0
        for _phase_id, target in targets:
            slew = required_orientation_slew_rad(
                base_orientation_xyzw=base,
                rest_grasp_orientation_base_xyzw=rest,
                target_orientation_xyzw=target,
            )
            if slew > worst:
                worst = slew
        if worst < best_worst:
            best_worst = worst
            best_yaw = yaw
        if math.ceil(worst / rate) <= steps:
            feasible_yaws.append(yaw)

    feasible = bool(feasible_yaws)
    blockers: list[str] = []
    if not feasible:
        worst_phase_id = max(
            targets,
            key=lambda item: required_orientation_slew_rad(
                base_orientation_xyzw=(
                    0.0,
                    0.0,
                    math.sin(best_yaw / 2.0),
                    math.cos(best_yaw / 2.0),
                ),
                rest_grasp_orientation_base_xyzw=rest,
                target_orientation_xyzw=item[1],
            ),
        )[0]
        blockers.append(f"{SLEW_BUDGET_BLOCKER}:{worst_phase_id}")

    return {
        "schema_version": SCHEMA_VERSION,
        "feasible": feasible,
        "blockers": blockers,
        "best_yaw_rad": best_yaw,
        "best_worst_slew_rad": best_worst,
        "best_worst_required_steps": int(math.ceil(best_worst / rate)),
        "step_budget": steps,
        "feasible_yaw_count": len(feasible_yaws),
        "yaw_sample_count": samples,
    }
