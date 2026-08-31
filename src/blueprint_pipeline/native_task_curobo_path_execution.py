"""Small native-worker helpers for exact cuRobo joint waypoint paths."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def validated_solver_joint_sequence(
    phase: Mapping[str, Any], *, arm_joint_names: Sequence[str]
) -> list[list[float]]:
    raw = phase.get("solver_joint_waypoint_sequence_rad")
    if raw is None:
        return []
    names = [str(value) for value in arm_joint_names]
    if (
        phase.get("solver_path_execution_required") is not True
        or not isinstance(raw, list)
        or not raw
    ):
        raise RuntimeError("native_task_curobo_entry_sequence_invalid")
    sequence: list[list[float]] = []
    for waypoint in raw:
        if not isinstance(waypoint, Mapping) or set(waypoint) != set(names):
            raise RuntimeError("native_task_curobo_entry_sequence_invalid")
        values = [float(waypoint[name]) for name in names]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("native_task_curobo_entry_sequence_invalid")
        sequence.append(values)
    return sequence


def solver_command_target(
    sequence: Sequence[Sequence[float]],
    *,
    waypoint_index: int,
    fallback: Sequence[float] | None,
) -> list[float] | None:
    if sequence:
        return [
            float(value)
            for value in sequence[min(waypoint_index, len(sequence) - 1)]
        ]
    return [float(value) for value in fallback] if fallback is not None else None


def advance_solver_waypoint(
    sequence: Sequence[Sequence[float]],
    *,
    waypoint_index: int,
    measured_joint_positions_rad: Sequence[float],
    tolerance_rad: float,
    diagnostic: dict[str, Any],
) -> int:
    if waypoint_index >= len(sequence):
        return waypoint_index
    error = max(
        abs(float(actual) - float(expected))
        for actual, expected in zip(
            measured_joint_positions_rad,
            sequence[waypoint_index],
            strict=True,
        )
    )
    diagnostic["curobo_solver_waypoint_index"] = waypoint_index
    diagnostic["curobo_solver_waypoint_joint_error_rad"] = error
    return waypoint_index + 1 if error <= float(tolerance_rad) else waypoint_index


def solver_path_result_fields(
    sequence: Sequence[Sequence[float]], *, waypoint_index: int
) -> dict[str, Any]:
    return {
        "curobo_solver_path_waypoint_count": len(sequence),
        "curobo_solver_path_waypoint_count_reached": waypoint_index,
        "curobo_solver_path_completed": waypoint_index >= len(sequence),
    }


__all__ = [
    "advance_solver_waypoint",
    "solver_command_target",
    "solver_path_result_fields",
    "validated_solver_joint_sequence",
]
