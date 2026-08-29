"""Digest-bound native trajectory context for task-aware robot placement.

Robot base placement is not a point-reach problem.  A fixed-base arm must reach
every authored tool pose in the construction plan with the authored gripper
orientation.  This module projects that immutable native plan into the compact
context shown to the placement agent without granting the model authority over
the plan or its thresholds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_robot_placement_trajectory.v1"
NATIVE_RIGID_PLAN_SCHEMA_VERSION = "native_rigid_construction_phase_plan.v1"


class RobotPlacementTrajectoryError(ValueError):
    """The supplied native construction path is not an immutable trajectory."""


def _optional_positive_int(value: object) -> int | None:
    """Project a positive integer budget, or None when the plan omits it."""

    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _vector(value: object, length: int, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise RobotPlacementTrajectoryError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RobotPlacementTrajectoryError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise RobotPlacementTrajectoryError(blocker)
    return result


def placement_trajectory_from_native_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and project one exact native rigid construction phase plan."""

    plan = json.loads(json.dumps(dict(value), allow_nan=False))
    phases = plan.get("phases")
    if (
        plan.get("schema_version") != NATIVE_RIGID_PLAN_SCHEMA_VERSION
        or not isinstance(phases, list)
        or not 1 <= len(phases) <= 64
        or plan.get("phase_count") != len(phases)
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
    ):
        raise RobotPlacementTrajectoryError(
            "robot_placement_native_trajectory_plan_invalid"
        )

    compact_phases: list[dict[str, Any]] = []
    phase_ids: set[str] = set()
    for raw in phases:
        if not isinstance(raw, Mapping):
            raise RobotPlacementTrajectoryError(
                "robot_placement_native_trajectory_phase_invalid"
            )
        phase_id = str(raw.get("phase_id") or "")
        gate_ids = raw.get("gate_ids")
        if (
            not phase_id
            or phase_id in phase_ids
            or not isinstance(gate_ids, list)
            or any(not str(item) for item in gate_ids)
        ):
            raise RobotPlacementTrajectoryError(
                "robot_placement_native_trajectory_phase_invalid"
            )
        phase_ids.add(phase_id)
        compact_phases.append(
            {
                "phase_id": phase_id,
                "position_world_m": _vector(
                    raw.get("position_world_m"),
                    3,
                    blocker="robot_placement_native_trajectory_phase_invalid",
                ),
                "orientation_world_xyzw": _vector(
                    raw.get("orientation_world_xyzw"),
                    4,
                    blocker="robot_placement_native_trajectory_phase_invalid",
                ),
                "gripper_state": str(raw.get("gripper_state") or ""),
                "gate_ids": [str(item) for item in gate_ids],
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_plan_schema_version": plan["schema_version"],
        "source_plan_digest": plan["plan_digest"],
        "task_kind": str(plan.get("task_kind") or ""),
        "manipulation_strategy": str(plan.get("manipulation_strategy") or ""),
        "arrival_tolerance_m": float(
            (plan.get("execution_parameters") or {}).get("arrival_tolerance_m")
        ),
        "arrival_orientation_tolerance_rad": float(
            (plan.get("execution_parameters") or {}).get(
                "arrival_orientation_tolerance_rad"
            )
        ),
        # The per-phase control-step budget bounds how far the wrist can slew
        # inside a phase, so it is part of what makes an authored tool
        # orientation reachable from a given base pose.  Projected here so the
        # analytic placement gate can decide that before any provider spend.
        # Optional: a plan that does not declare one simply cannot be screened
        # for slew feasibility analytically, and stays native-gated.
        "maximum_steps_per_phase": _optional_positive_int(
            (plan.get("execution_parameters") or {}).get("maximum_steps_per_phase")
        ),
        "phases": compact_phases,
        "model_may_modify_trajectory": False,
        "native_ik_and_collision_readback_required_for_every_phase": True,
        "trajectory_digest": "",
    }
    if (
        not math.isfinite(result["arrival_tolerance_m"])
        or result["arrival_tolerance_m"] <= 0
        or not math.isfinite(result["arrival_orientation_tolerance_rad"])
        or result["arrival_orientation_tolerance_rad"] <= 0
    ):
        raise RobotPlacementTrajectoryError(
            "robot_placement_native_trajectory_threshold_invalid"
        )
    result["trajectory_digest"] = canonical_digest(
        result, digest_field="trajectory_digest"
    )
    return result


def placement_trajectory_from_native_result(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the exact phase plan retained by a prior native construction."""

    result = json.loads(json.dumps(dict(value), allow_nan=False))
    if (
        result.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
        or not isinstance(result.get("construction_phase_plan"), Mapping)
    ):
        raise RobotPlacementTrajectoryError(
            "robot_placement_native_construction_result_invalid"
        )
    return placement_trajectory_from_native_plan(result["construction_phase_plan"])


def validate_robot_placement_trajectory(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate an already projected placement trajectory."""

    trajectory = json.loads(json.dumps(dict(value), allow_nan=False))
    if (
        trajectory.get("schema_version") != SCHEMA_VERSION
        or trajectory.get("model_may_modify_trajectory") is not False
        or trajectory.get("native_ik_and_collision_readback_required_for_every_phase")
        is not True
        or trajectory.get("trajectory_digest")
        != canonical_digest(trajectory, digest_field="trajectory_digest")
    ):
        raise RobotPlacementTrajectoryError("robot_placement_trajectory_invalid")
    phases = trajectory.get("phases")
    if not isinstance(phases, list) or not phases:
        raise RobotPlacementTrajectoryError("robot_placement_trajectory_invalid")
    return trajectory


__all__ = [
    "RobotPlacementTrajectoryError",
    "placement_trajectory_from_native_plan",
    "placement_trajectory_from_native_result",
    "validate_robot_placement_trajectory",
]
