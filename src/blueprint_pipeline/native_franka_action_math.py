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


def joint_velocity_feedforward_rad_s(
    *,
    commanded_joint_positions_rad: Sequence[float],
    previous_commanded_joint_positions_rad: Sequence[float],
    control_period_seconds: float,
    scale: float = 1.0,
) -> list[float]:
    """Return the velocity the commanded setpoint is advancing at.

    Isaac Lab's implicit actuator is a PD whose torque is
    ``stiffness * (pos_target - pos) + damping * (vel_target - vel)``.  A
    position-only command leaves ``vel_target`` at zero, so the damping term
    becomes pure braking proportional to whatever speed we asked the joint to
    reach -- the arm settles where the two terms cancel, at
    ``(stiffness / damping) * position_error``, using almost none of its
    available torque.  Declaring the velocity we intend cancels that braking
    while tracking.

    The feedforward is the *commanded* advance rate, not a target speed, so it
    falls to zero the moment the setpoint stops advancing and the damping term
    returns to damping the joint at rest.
    """

    try:
        commanded = [float(value) for value in commanded_joint_positions_rad]
        previous = [float(value) for value in previous_commanded_joint_positions_rad]
        period = float(control_period_seconds)
        gain = float(scale)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_velocity_feedforward_contract_invalid"]
        ) from exc
    if (
        not commanded
        or len(commanded) != len(previous)
        or not all(math.isfinite(value) for value in (*commanded, *previous))
        or not math.isfinite(period)
        or period <= 0.0
        or not math.isfinite(gain)
        or gain < 0.0
        or gain > 1.0
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_velocity_feedforward_contract_invalid"]
        )
    return [
        gain * (command - prior) / period
        for command, prior in zip(commanded, previous, strict=True)
    ]


def implicit_pd_torque_terms(
    *,
    commanded_joint_positions_rad: Sequence[float],
    measured_joint_positions_rad: Sequence[float],
    commanded_joint_velocities_rad_s: Sequence[float],
    measured_joint_velocities_rad_s: Sequence[float],
    joint_stiffness: Sequence[float],
    joint_damping: Sequence[float],
) -> dict[str, list[float]]:
    """Split the implicit-actuator torque into its two competing terms.

    ``applied_torque`` alone is not interpretable: on a gravity-disabled arm at
    steady state the stiffness and damping terms cancel, so a correctly
    configured actuator reads near zero and looks indistinguishable from one
    whose gains were never applied.  Recording both terms and the measured
    velocity separately removes that ambiguity.
    """

    try:
        commanded = [float(value) for value in commanded_joint_positions_rad]
        measured = [float(value) for value in measured_joint_positions_rad]
        velocity_target = [
            float(value) for value in commanded_joint_velocities_rad_s
        ]
        velocity = [float(value) for value in measured_joint_velocities_rad_s]
        stiffness = [float(value) for value in joint_stiffness]
        damping = [float(value) for value in joint_damping]
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_pd_torque_terms_contract_invalid"]
        ) from exc
    lengths = {
        len(commanded),
        len(measured),
        len(velocity_target),
        len(velocity),
        len(stiffness),
        len(damping),
    }
    if len(lengths) != 1 or not commanded:
        raise NativeFrankaActionMathError(
            ["native_franka_pd_torque_terms_contract_invalid"]
        )
    stiffness_term = [
        gain * (command - state)
        for gain, command, state in zip(stiffness, commanded, measured, strict=True)
    ]
    damping_term = [
        gain * (command - state)
        for gain, command, state in zip(
            damping, velocity_target, velocity, strict=True
        )
    ]
    return {
        "stiffness_term_n_m": stiffness_term,
        "damping_term_n_m": damping_term,
        "predicted_torque_n_m": [
            first + second
            for first, second in zip(stiffness_term, damping_term, strict=True)
        ],
    }


IDENTITY_QUATERNION_TOLERANCE = 1.0e-6


def is_unauthored_identity_quaternion_xyzw(
    value: Sequence[float] | None, *, tolerance: float = IDENTITY_QUATERNION_TOLERANCE
) -> bool:
    """Return whether an orientation is the identity placeholder.

    This slot holds the rotation that aligns a contact frame with the gripper
    frame.  Isaac Lab's own Franka handle-grasp reference authors exactly such a
    rotation and says so:

        # cabinet_env_cfg.py, drawer_handle_top FrameCfg
        offset=OffsetCfg(pos=(0.305, 0.0, 0.01),
                         rot=(0.5, -0.5, -0.5, 0.5))  # align with end-effector frame

    ``OffsetCfg.rot`` is documented as "(x, y, z, w) w.r.t. the parent frame",
    applied as ``frame_world = prim_world * offset`` -- the same composition this
    repository performs.  That reference rotation is 120 degrees; the state
    machine then grasps with an *identity* offset on top of that already-aligned
    frame (``transform_multiply(handle_grasp_offset, handle_pose)``).

    So identity here does not mean "aligned", it means the alignment was never
    authored.  On this arm that costs exactly the reference's 120 degrees: the
    Franka hand rests at ``(0.5, 0.5, 0.5, 0.5)``, and an identity target
    commands a 120 degree rotation unrelated to the task.  Near that error the
    differential IK alternates between solution branches and the arm thrashes
    into its joint limits at saturated torque while the end-effector never
    approaches the target.

    A genuinely intended identity alignment is indistinguishable from an
    unauthored one, so identity is always read as "not authored".
    """

    if value is None:
        return True
    try:
        quaternion = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if len(quaternion) != 4 or not all(math.isfinite(item) for item in quaternion):
        return False
    limit = abs(float(tolerance))
    # Both signs represent the same rotation.
    return all(abs(item) <= limit for item in quaternion[:3]) and (
        abs(abs(quaternion[3]) - 1.0) <= limit
    )


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
