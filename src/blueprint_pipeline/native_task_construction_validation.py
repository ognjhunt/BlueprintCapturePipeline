"""Shared fail-closed validation for native construction plans."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


class NativeTaskConstructionPlanError(ValueError):
    """Stable pre-native failures for task-neutral construction planning."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def finite_vector(value: Any, *, length: int, error: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionPlanError([error]) from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise NativeTaskConstructionPlanError([error])
    return result


def positive(value: Any, *, error: str, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionPlanError([error]) from exc
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        raise NativeTaskConstructionPlanError([error])
    return result


def construction_total_step_budget(
    *,
    maximum_action_steps: Any,
    settle_window_samples: Any,
    minimum_required_steps: int,
    invalid_error: str,
    infeasible_error: str,
) -> int:
    """Reserve controls settle steps before native construction can run."""

    values = (maximum_action_steps, settle_window_samples)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise NativeTaskConstructionPlanError([invalid_error])
    available = maximum_action_steps - settle_window_samples
    if available < minimum_required_steps:
        raise NativeTaskConstructionPlanError([infeasible_error])
    return available
