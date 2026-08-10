"""Task-neutral Franka absolute-action and grasp-frame transforms."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


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


def resolve_gripper_command_endpoints(
    *,
    tool_point_separations_m: Mapping[str, float],
    minimum_travel_m: float = 1.0e-3,
    equivalent_endpoint_tolerance_m: float = 5.0e-4,
) -> dict[str, object]:
    """Resolve open/closed commands from measured semantic pad separation.

    The Arena DROID action uses ``0``/``1`` while Isaac Lab's generic binary
    action uses negative/positive values.  The caller therefore probes a
    bounded candidate set and this function selects endpoints from measured
    pad travel.  Equivalent open commands are resolved deterministically to
    the smallest-magnitude value so a threshold boundary cannot vary with
    sub-millimetre simulation noise.
    """

    try:
        rows = sorted(
            (
                (float(command), float(separation))
                for command, separation in tool_point_separations_m.items()
            ),
            key=lambda row: row[0],
        )
        minimum_travel = float(minimum_travel_m)
        endpoint_tolerance = float(equivalent_endpoint_tolerance_m)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_gripper_endpoint_measurement_invalid"]
        ) from exc
    if (
        len(rows) < 2
        or len({command for command, _ in rows}) != len(rows)
        or not all(math.isfinite(value) for row in rows for value in row)
        or not math.isfinite(minimum_travel)
        or minimum_travel <= 0.0
        or not math.isfinite(endpoint_tolerance)
        or endpoint_tolerance < 0.0
        or endpoint_tolerance >= minimum_travel
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_gripper_endpoint_measurement_invalid"]
        )
    minimum = min(separation for _, separation in rows)
    maximum = max(separation for _, separation in rows)
    travel = maximum - minimum
    if travel < minimum_travel:
        return {
            "status": "ambiguous",
            "closed_command": None,
            "open_command": None,
            "separation_travel_m": travel,
            "minimum_travel_m": minimum_travel,
            "equivalent_endpoint_tolerance_m": endpoint_tolerance,
            "blockers": ["native_task_gripper_convention_travel_below_floor"],
        }

    closed_candidates = [
        command
        for command, separation in rows
        if separation <= minimum + endpoint_tolerance
    ]
    open_candidates = [
        command
        for command, separation in rows
        if separation >= maximum - endpoint_tolerance
    ]

    def preferred(commands: Sequence[float]) -> float:
        return min(commands, key=lambda value: (abs(value), -value))

    return {
        "status": "measured",
        "closed_command": preferred(closed_candidates),
        "open_command": preferred(open_candidates),
        "closed_command_candidates": closed_candidates,
        "open_command_candidates": open_candidates,
        "closed_tool_point_separation_m": minimum,
        "open_tool_point_separation_m": maximum,
        "separation_travel_m": travel,
        "minimum_travel_m": minimum_travel,
        "equivalent_endpoint_tolerance_m": endpoint_tolerance,
        "blockers": [],
    }


__all__ = [
    "NativeFrankaActionMathError",
    "bounded_absolute_joint_setpoint",
    "controlled_body_pose_for_grasp_frame_target",
    "resolve_gripper_command_endpoints",
]
