"""Task-neutral Franka absolute-action and grasp-frame transforms."""

from __future__ import annotations

import math
from collections.abc import Sequence


class NativeFrankaActionMathError(ValueError):
    """Stable failures before a native Franka action reaches the simulator."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def bounded_absolute_joint_setpoint(
    *,
    measured_joint_positions_rad: Sequence[float],
    desired_joint_positions_rad: Sequence[float],
    previous_commanded_joint_positions_rad: Sequence[float],
    max_command_slew_per_step_rad: float,
    max_setpoint_lead_rad: float,
) -> list[float]:
    """Bound one absolute joint target by command slew and measured-state lead."""

    try:
        measured = [float(value) for value in measured_joint_positions_rad]
        desired = [float(value) for value in desired_joint_positions_rad]
        previous = [float(value) for value in previous_commanded_joint_positions_rad]
        max_slew = float(max_command_slew_per_step_rad)
        max_lead = float(max_setpoint_lead_rad)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_joint_setpoint_contract_invalid"]
        ) from exc
    if (
        not measured
        or len(measured) != len(desired)
        or len(measured) != len(previous)
        or not all(math.isfinite(value) for value in (*measured, *desired, *previous))
        or not math.isfinite(max_slew)
        or not math.isfinite(max_lead)
        or max_slew <= 0.0
        or max_lead < max_slew
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_joint_setpoint_contract_invalid"]
        )
    command: list[float] = []
    for measured_value, desired_value, previous_value in zip(
        measured, desired, previous, strict=True
    ):
        lower = max(previous_value - max_slew, measured_value - max_lead)
        upper = min(previous_value + max_slew, measured_value + max_lead)
        if lower > upper + 1.0e-12:
            raise NativeFrankaActionMathError(
                ["native_franka_joint_setpoint_constraints_infeasible"]
            )
        command.append(min(max(desired_value, lower), upper))
    return command


def controlled_body_pose_for_grasp_frame_target(
    *,
    current_body_position_world_m: Sequence[float],
    current_body_quaternion_world_xyzw: Sequence[float],
    current_grasp_frame_position_world_m: Sequence[float],
    target_grasp_frame_position_world_m: Sequence[float],
    target_body_quaternion_world_xyzw: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Resolve the controlled-body pose placing a measured grasp frame at target."""

    try:
        body = [float(value) for value in current_body_position_world_m]
        quaternion = [float(value) for value in current_body_quaternion_world_xyzw]
        grasp = [float(value) for value in current_grasp_frame_position_world_m]
        target = [float(value) for value in target_grasp_frame_position_world_m]
        target_quaternion = [
            float(value) for value in target_body_quaternion_world_xyzw
        ]
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_grasp_frame_transform_invalid"]
        ) from exc
    if (
        len(body) != 3
        or len(quaternion) != 4
        or len(grasp) != 3
        or len(target) != 3
        or len(target_quaternion) != 4
        or not all(
            math.isfinite(value)
            for value in (*body, *quaternion, *grasp, *target, *target_quaternion)
        )
        or abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0) > 1.0e-5
        or abs(
            math.sqrt(sum(value * value for value in target_quaternion)) - 1.0
        )
        > 1.0e-5
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_grasp_frame_transform_invalid"]
        )

    def rotate(q: Sequence[float], vector: Sequence[float]) -> list[float]:
        x, y, z, w = q
        vx, vy, vz = vector
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ]

    body_to_grasp_world = [grasp[index] - body[index] for index in range(3)]
    body_to_grasp_local = rotate(
        [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
        body_to_grasp_world,
    )
    target_body_to_grasp_world = rotate(target_quaternion, body_to_grasp_local)
    target_body = [
        target[index] - target_body_to_grasp_world[index] for index in range(3)
    ]
    return target_body, target_quaternion


__all__ = [
    "NativeFrankaActionMathError",
    "bounded_absolute_joint_setpoint",
    "controlled_body_pose_for_grasp_frame_target",
]
