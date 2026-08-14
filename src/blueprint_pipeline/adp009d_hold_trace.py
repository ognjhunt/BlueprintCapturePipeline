"""Classify what a canonical-hold arm-pose trace actually shows.

A single post-warmup pose snapshot can only say *that* the arm failed to hold
its commanded pose.  It cannot say *why*: an arm still falling through a
transient and an arm parked at a stable wrong pose produce the same number.
Those two have different causes and different fixes, so the distinction has to
survive into the retained receipt rather than being re-derived by eye.

This module is deliberately free of Isaac, torch, and USD imports so the
classification runs in the hermetic fast lane.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

ARM_JOINT_NAMES: tuple[str, ...] = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)

ARM_JOINT_COUNT = len(ARM_JOINT_NAMES)

HOLD_TRACE_SCHEMA_VERSION = "adp009d_arm_hold_trace.v1"


class HoldTraceError(RuntimeError):
    """A typed, fail-closed hold-trace rejection."""


def _arm_slice(values: Any, *, blocker: str) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise HoldTraceError(blocker)
    if len(values) < ARM_JOINT_COUNT:
        raise HoldTraceError(blocker)
    try:
        converted = [float(item) for item in list(values)[:ARM_JOINT_COUNT]]
    except (TypeError, ValueError) as exc:  # non-numeric entries are not a trace
        raise HoldTraceError(blocker) from exc
    if not all(math.isfinite(item) for item in converted):
        raise HoldTraceError(f"{blocker}_nonfinite")
    return converted


def _segment_means(errors: Sequence[float]) -> tuple[float, float]:
    """Mean error over the first and last third of the trace."""

    third = max(1, len(errors) // 3)
    head = errors[:third]
    tail = errors[-third:]
    return sum(head) / len(head), sum(tail) / len(tail)


def _first_env_row(value: Any, to_list: Callable[[Any], Iterable[Any]]) -> list[float] | None:
    """Convert the whole simulator-native array, then take the first environment.

    Conversion happens before indexing because the backends differ in what they
    hand back — a warp array does not index like a torch tensor.
    """

    if value is None:
        return None
    rows = list(to_list(value))
    if not rows:
        return None
    return [float(item) for item in rows[0]]


def extract_arm_sample(
    robot: Any,
    *,
    step_index: int,
    to_list: Callable[[Any], Iterable[Any]],
) -> dict[str, Any] | None:
    """Read one trace row off a live articulation, or return ``None``.

    Instrumentation must never be able to fail a run that would otherwise
    succeed, so every backend difference here degrades to a missing sample
    rather than an exception.  ``to_list`` is injected so the caller owns the
    tensor conversion and this module stays importable without torch.
    """

    try:
        data = getattr(robot, "data", None)
        positions = _first_env_row(getattr(data, "joint_pos", None), to_list)
        if positions is None or len(positions) < ARM_JOINT_COUNT:
            return None
        torque: list[float] | None = None
        for attribute in ("applied_torque", "computed_torque"):
            candidate = _first_env_row(getattr(data, attribute, None), to_list)
            if candidate is not None and len(candidate) >= ARM_JOINT_COUNT:
                torque = candidate[:ARM_JOINT_COUNT]
                break
        return {
            "step_index": int(step_index),
            "joint_positions_rad": positions[:ARM_JOINT_COUNT],
            "applied_torque_nm": torque,
        }
    except Exception:  # noqa: BLE001 - a trace gap must never fail the run
        return None


def extract_arm_effort_limits(
    robot: Any,
    *,
    to_list: Callable[[Any], Iterable[Any]],
) -> list[float] | None:
    """Read the arm's effort limits, preferring the simulated field.

    Isaac Lab keeps the legacy ``effort_limit`` for backwards compatibility
    while Newton reads only the ``_sim`` counterpart, so the simulated field is
    the one that describes what the solver will actually enforce.
    """

    try:
        data = getattr(robot, "data", None)
        for attribute in ("joint_effort_limits_sim", "joint_effort_limits"):
            limits = _first_env_row(getattr(data, attribute, None), to_list)
            if limits is not None and len(limits) >= ARM_JOINT_COUNT:
                return limits[:ARM_JOINT_COUNT]
        return None
    except Exception:  # noqa: BLE001 - a missing limit must never fail the run
        return None


def _summarize_torque(
    torque_by_step: list[list[float] | None],
    *,
    effort_limits_nm: Sequence[float] | None,
    saturation_fraction: float,
    tail_step_fraction: float,
) -> dict[str, Any]:
    """Report torque utilization, and say plainly when it cannot be judged."""

    summary: dict[str, Any] = {
        "available": False,
        "unavailable_reason": "applied_torque_not_retained",
        "saturation_fraction_threshold": float(saturation_fraction),
        "effort_limits_nm": None,
        "final_applied_torque_nm": None,
        "final_utilization_fraction": None,
        "saturated_joint_names": [],
        "saturated_joint_indices": [],
    }
    if any(row is None for row in torque_by_step):
        return summary

    summary["available"] = True
    summary["final_applied_torque_nm"] = torque_by_step[-1]
    if effort_limits_nm is None:
        summary["unavailable_reason"] = "effort_limits_not_supplied"
        return summary

    limits = _arm_slice(
        effort_limits_nm, blocker="adp009d_hold_trace_effort_limit_width_invalid"
    )
    summary["unavailable_reason"] = None
    summary["effort_limits_nm"] = limits
    summary["final_utilization_fraction"] = [
        abs(torque_by_step[-1][index]) / limits[index] if limits[index] > 0.0 else None
        for index in range(ARM_JOINT_COUNT)
    ]

    third = max(1, len(torque_by_step) // 3)
    tail = torque_by_step[-third:]
    for index in range(ARM_JOINT_COUNT):
        limit = limits[index]
        if limit <= 0.0:
            continue
        pinned = sum(
            1 for row in tail if abs(row[index]) >= saturation_fraction * limit
        )
        if pinned / len(tail) >= tail_step_fraction:
            summary["saturated_joint_indices"].append(index)
            summary["saturated_joint_names"].append(ARM_JOINT_NAMES[index])
    return summary


def classify_arm_hold_trace(
    samples: Iterable[dict[str, Any]],
    *,
    requested_joint_positions_rad: Sequence[float],
    tolerance_rad: float,
    effort_limits_nm: Sequence[float] | None = None,
    saturation_fraction: float = 0.99,
    tail_step_fraction: float = 0.5,
) -> dict[str, Any]:
    """Summarize a per-step canonical-hold trace into a typed failure mode.

    ``samples`` rows carry ``joint_positions_rad``; anything wider than the
    seven arm joints (a full articulation state, say) is sliced, because the
    canonical hold contract is defined over the arm only.
    """

    rows = list(samples)
    if not rows:
        raise HoldTraceError("adp009d_hold_trace_empty")
    if not math.isfinite(float(tolerance_rad)) or float(tolerance_rad) <= 0.0:
        raise HoldTraceError("adp009d_hold_trace_tolerance_invalid")
    if (
        not math.isfinite(float(saturation_fraction))
        or not 0.0 < float(saturation_fraction) <= 1.0
        or not math.isfinite(float(tail_step_fraction))
        or not 0.0 < float(tail_step_fraction) <= 1.0
    ):
        raise HoldTraceError("adp009d_hold_trace_saturation_threshold_invalid")

    target = _arm_slice(
        requested_joint_positions_rad,
        blocker="adp009d_hold_trace_target_width_invalid",
    )

    errors_by_step: list[float] = []
    final_absolute_error: list[float] = []
    torque_by_step: list[list[float] | None] = []
    for row in rows:
        if not isinstance(row, dict):
            raise HoldTraceError("adp009d_hold_trace_sample_invalid")
        positions = _arm_slice(
            row.get("joint_positions_rad"),
            blocker="adp009d_hold_trace_joint_width_invalid",
        )
        absolute_error = [abs(positions[i] - target[i]) for i in range(ARM_JOINT_COUNT)]
        errors_by_step.append(max(absolute_error))
        final_absolute_error = absolute_error
        applied = row.get("applied_torque_nm")
        torque_by_step.append(
            None
            if applied is None
            else _arm_slice(
                applied, blocker="adp009d_hold_trace_torque_width_invalid"
            )
        )

    final_maximum_error = errors_by_step[-1]
    worst_index = max(
        range(ARM_JOINT_COUNT), key=lambda index: final_absolute_error[index]
    )

    head_mean, tail_mean = _segment_means(errors_by_step)
    # A change only counts as real when it clears both a relative floor and the
    # tolerance itself, so tolerance-scale jitter never reads as a trend.
    margin = max(0.1 * head_mean, tolerance_rad)
    if tail_mean > head_mean + margin:
        convergence = "diverging"
    elif tail_mean < head_mean - margin:
        convergence = "converging"
    else:
        convergence = "settled"

    torque = _summarize_torque(
        torque_by_step,
        effort_limits_nm=effort_limits_nm,
        saturation_fraction=saturation_fraction,
        tail_step_fraction=tail_step_fraction,
    )

    if final_maximum_error <= tolerance_rad:
        failure_mode = "within_tolerance"
    elif worst_index in torque["saturated_joint_indices"]:
        # The joint carrying the error had nothing left to give: the limit is
        # the cause, whatever the position curve happens to look like.
        failure_mode = "effort_saturated"
    elif convergence == "settled":
        failure_mode = "settled_offset"
    else:
        failure_mode = "unconverged_transient"

    return {
        "torque": torque,
        "schema_version": HOLD_TRACE_SCHEMA_VERSION,
        "sample_count": len(rows),
        "joint_names": list(ARM_JOINT_NAMES),
        "tolerance_rad": float(tolerance_rad),
        "requested_joint_positions_rad": target,
        "final_joint_absolute_error_rad": final_absolute_error,
        "maximum_error_rad_by_step": errors_by_step,
        "final_maximum_error_rad": final_maximum_error,
        "worst_joint_index": worst_index,
        "worst_joint_name": ARM_JOINT_NAMES[worst_index],
        "head_mean_error_rad": head_mean,
        "tail_mean_error_rad": tail_mean,
        "convergence": convergence,
        "hold_failure_mode": failure_mode,
    }
