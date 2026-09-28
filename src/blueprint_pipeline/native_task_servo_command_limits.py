"""Validate sealed native joint-command limits shared by runtime workers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def servo_command_limits(
    execution_parameters: Mapping[str, Any],
) -> dict[str, float]:
    limits: dict[str, float] = {}
    for field in (
        "max_joint_delta_rad",
        "max_joint_setpoint_lead_rad",
        "velocity_feedforward_scale",
    ):
        try:
            value = float(execution_parameters[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"native_task_construction_servo_command_limit_missing:{field}"
            ) from exc
        floor = 0.0 if field == "velocity_feedforward_scale" else None
        if (
            not math.isfinite(value)
            or (floor is None and value <= 0.0)
            or (floor is not None and not 0.0 <= value <= 1.0)
        ):
            raise RuntimeError(
                f"native_task_construction_servo_command_limit_invalid:{field}"
            )
        limits[field] = value
    if limits["max_joint_setpoint_lead_rad"] < limits["max_joint_delta_rad"]:
        raise RuntimeError(
            "native_task_construction_servo_command_limit_invalid:"
            "max_joint_setpoint_lead_rad"
        )
    return limits


__all__ = ["servo_command_limits"]
