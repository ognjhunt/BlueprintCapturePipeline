"""Task-neutral Franka absolute-action and grasp-frame transforms."""

from __future__ import annotations

import math
from collections.abc import Sequence


class NativeFrankaActionMathError(ValueError):
    """Stable failures before a native Franka action reaches the simulator."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def bounded_cartesian_pose_target(
    *,
    current_position_world_m: Sequence[float],
    current_quaternion_world_xyzw: Sequence[float],
    target_position_world_m: Sequence[float],
    target_quaternion_world_xyzw: Sequence[float],
    max_translation_step_m: float,
    max_orientation_step_rad: float,
) -> tuple[list[float], list[float]]:
    """Return the next local pose target on the shortest Cartesian path.

    Isaac Lab's differential IK controller applies the complete Cartesian pose
    error in one Jacobian solve and does not impose joint limits. That is safe
    for the small deltas used by its teleoperation examples, but a large
    absolute orientation change can produce a multi-radian joint update before
    the existing joint-space slew limiter gets a chance to act. Interpolating
    the pose first keeps the Jacobian linearisation local; the joint-space
    limits remain an independent second bound.
    """

    try:
        current_position = [float(value) for value in current_position_world_m]
        target_position = [float(value) for value in target_position_world_m]
        current_quaternion = [
            float(value) for value in current_quaternion_world_xyzw
        ]
        target_quaternion = [
            float(value) for value in target_quaternion_world_xyzw
        ]
        translation_limit = float(max_translation_step_m)
        orientation_limit = float(max_orientation_step_rad)
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_cartesian_pose_step_invalid"]
        ) from exc
    if not (
        len(current_position) == len(target_position) == 3
        and len(current_quaternion) == len(target_quaternion) == 4
        and all(
            math.isfinite(value)
            for row in (
                current_position,
                target_position,
                current_quaternion,
                target_quaternion,
            )
            for value in row
        )
        and math.isfinite(translation_limit)
        and translation_limit > 0.0
        and math.isfinite(orientation_limit)
        and 0.0 < orientation_limit <= math.pi
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_cartesian_pose_step_invalid"]
        )

    def normalize(quaternion: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise NativeFrankaActionMathError(
                ["native_franka_cartesian_pose_step_invalid"]
            )
        return [value / norm for value in quaternion]

    delta = [
        target_position[index] - current_position[index] for index in range(3)
    ]
    distance = math.sqrt(sum(value * value for value in delta))
    position_fraction = min(1.0, translation_limit / max(distance, 1.0e-12))
    next_position = [
        current_position[index] + position_fraction * delta[index]
        for index in range(3)
    ]

    start = normalize(current_quaternion)
    end = normalize(target_quaternion)
    dot = sum(left * right for left, right in zip(start, end, strict=True))
    if dot < 0.0:
        end = [-value for value in end]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    angle = 2.0 * math.acos(dot)
    orientation_fraction = min(1.0, orientation_limit / max(angle, 1.0e-12))
    if dot > 0.9995:
        next_quaternion = [
            left + orientation_fraction * (right - left)
            for left, right in zip(start, end, strict=True)
        ]
    else:
        half_angle = math.acos(dot)
        denominator = math.sin(half_angle)
        left_weight = math.sin((1.0 - orientation_fraction) * half_angle) / denominator
        right_weight = math.sin(orientation_fraction * half_angle) / denominator
        next_quaternion = [
            left_weight * left + right_weight * right
            for left, right in zip(start, end, strict=True)
        ]
    return next_position, normalize(next_quaternion)


def bounded_absolute_joint_setpoint(
    *,
    measured_joint_positions_rad: Sequence[float],
    desired_joint_positions_rad: Sequence[float],
    previous_commanded_joint_positions_rad: Sequence[float],
    max_command_slew_per_step_rad: float,
    max_setpoint_lead_rad: float,
    max_setpoint_lead_rad_per_joint: Sequence[float] | None = None,
) -> list[float]:
    """Bound one absolute joint target by command slew and measured-state lead.

    ``max_setpoint_lead_rad_per_joint`` additionally caps each joint by what
    that actuator can actually pull -- ``effort_limit / stiffness`` -- so the
    command never asks for torque the joint will simply clip.  It only ever
    tightens the scalar lead.
    """

    try:
        measured = [float(value) for value in measured_joint_positions_rad]
        desired = [float(value) for value in desired_joint_positions_rad]
        previous = [float(value) for value in previous_commanded_joint_positions_rad]
        max_slew = float(max_command_slew_per_step_rad)
        max_lead = float(max_setpoint_lead_rad)
        per_joint_lead = (
            None
            if max_setpoint_lead_rad_per_joint is None
            else [float(value) for value in max_setpoint_lead_rad_per_joint]
        )
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_joint_setpoint_contract_invalid"]
        ) from exc
    if per_joint_lead is not None and (
        len(per_joint_lead) != len(measured)
        or not all(
            math.isfinite(value) and value > 0.0 for value in per_joint_lead
        )
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_joint_setpoint_contract_invalid"]
        )
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
    for index, (measured_value, desired_value, previous_value) in enumerate(
        zip(measured, desired, previous, strict=True)
    ):
        joint_lead = max_lead
        if per_joint_lead is not None:
            # Never loosen the caller's bound, only tighten it to feasibility,
            # and never below one slew step or the command could not advance.
            joint_lead = max(
                min(joint_lead, per_joint_lead[index]), max_slew
            )
        lower = max(previous_value - max_slew, measured_value - joint_lead)
        upper = min(previous_value + max_slew, measured_value + joint_lead)
        if lower > upper + 1.0e-12:
            raise NativeFrankaActionMathError(
                ["native_franka_joint_setpoint_constraints_infeasible"]
            )
        command.append(min(max(desired_value, lower), upper))
    return command


def clip_joint_positions_to_limits(
    *,
    desired_joint_positions_rad: Sequence[float],
    lower_joint_position_limits_rad: Sequence[float],
    upper_joint_position_limits_rad: Sequence[float],
) -> list[float]:
    """Clip a local IK solution to the articulation's measured soft limits."""

    try:
        desired = [float(value) for value in desired_joint_positions_rad]
        lower = [float(value) for value in lower_joint_position_limits_rad]
        upper = [float(value) for value in upper_joint_position_limits_rad]
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_joint_position_limits_invalid"]
        ) from exc
    if not (
        desired
        and len(desired) == len(lower) == len(upper)
        and all(math.isfinite(value) for value in (*desired, *lower, *upper))
        and all(low < high for low, high in zip(lower, upper, strict=True))
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_joint_position_limits_invalid"]
        )
    return [
        min(max(value, low), high)
        for value, low, high in zip(desired, lower, upper, strict=True)
    ]


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


def controlled_body_pose_for_rigid_grasp_frame_target(
    *,
    current_body_position_world_m: Sequence[float],
    current_body_quaternion_world_xyzw: Sequence[float],
    current_grasp_frame_position_world_m: Sequence[float],
    current_grasp_frame_quaternion_world_xyzw: Sequence[float],
    target_grasp_frame_position_world_m: Sequence[float],
    target_grasp_frame_quaternion_world_xyzw: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Place a rigidly offset grasp/TCP frame at a desired world pose."""

    try:
        body_position = [float(value) for value in current_body_position_world_m]
        grasp_position = [
            float(value) for value in current_grasp_frame_position_world_m
        ]
        target_position = [
            float(value) for value in target_grasp_frame_position_world_m
        ]
        body_quaternion = [
            float(value) for value in current_body_quaternion_world_xyzw
        ]
        grasp_quaternion = [
            float(value) for value in current_grasp_frame_quaternion_world_xyzw
        ]
        target_grasp_quaternion = [
            float(value) for value in target_grasp_frame_quaternion_world_xyzw
        ]
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_rigid_grasp_frame_transform_invalid"]
        ) from exc
    vectors = (body_position, grasp_position, target_position)
    quaternions = (body_quaternion, grasp_quaternion, target_grasp_quaternion)
    if not (
        all(len(vector) == 3 for vector in vectors)
        and all(len(quaternion) == 4 for quaternion in quaternions)
        and all(
            math.isfinite(value)
            for row in (*vectors, *quaternions)
            for value in row
        )
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_rigid_grasp_frame_transform_invalid"]
        )

    def normalize(quaternion: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise NativeFrankaActionMathError(
                ["native_franka_rigid_grasp_frame_transform_invalid"]
            )
        return [value / norm for value in quaternion]

    def multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]

    def inverse(quaternion: Sequence[float]) -> list[float]:
        x, y, z, w = normalize(quaternion)
        return [-x, -y, -z, w]

    def rotate(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
        x, y, z, w = normalize(quaternion)
        vx, vy, vz = vector
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ]

    body_quaternion = normalize(body_quaternion)
    grasp_quaternion = normalize(grasp_quaternion)
    target_grasp_quaternion = normalize(target_grasp_quaternion)
    body_to_grasp_position = rotate(
        inverse(body_quaternion),
        [grasp_position[index] - body_position[index] for index in range(3)],
    )
    body_to_grasp_quaternion = normalize(
        multiply(inverse(body_quaternion), grasp_quaternion)
    )
    target_body_quaternion = normalize(
        multiply(target_grasp_quaternion, inverse(body_to_grasp_quaternion))
    )
    target_body_to_grasp_world = rotate(
        target_body_quaternion, body_to_grasp_position
    )
    target_body_position = [
        target_position[index] - target_body_to_grasp_world[index]
        for index in range(3)
    ]
    return target_body_position, target_body_quaternion


GRASP_AXIS_DEGENERACY_TOLERANCE = 1.0e-6


def grasp_orientation_contact_xyzw(
    *,
    approach_axis: Sequence[float],
    jaw_axis: Sequence[float],
) -> list[float]:
    """Build the contact-frame -> gripper-frame rotation for a parallel jaw.

    This is the slot ``gripper_orientation_contact_xyzw`` holds, and it is the
    same slot Isaac Lab's Franka cabinet reference fills with
    ``rot=(0.5, -0.5, -0.5, 0.5)  # align with end-effector frame``.  Three
    independent places in that reference state the same axis convention:

      * ``cabinet_env_cfg.py`` authors the rotation itself, and
        ``open_cabinet_sm.py`` then drives ``des_ee_pose = handle_pose`` with
        purely translational grasp offsets, so the offset frame *is* the
        commanded end-effector frame;
      * ``franka_cabinet_env.py`` names the axes --
        ``gripper_forward_axis = +Z`` pairs with ``drawer_inward_axis = -X``,
        and ``gripper_up_axis = +Y`` pairs with ``drawer_up_axis = +Z``;
      * ``mdp/rewards.py::align_ee_handle`` restates it in prose: "the z
        direction of the gripper should be close to the -x direction of the
        handle and the x direction of the gripper should be close to the -y
        direction of the handle".

    So the gripper frame is fully determined by two axes:

      * ``+Z`` is the approach axis -- the direction the hand travels *into*
        the feature.  ``franka_cabinet_env.py`` offsets its grasp point by
        ``[0, 0.04, 0]``, the finger-open distance, which is why
      * ``+Y`` is the jaw separation axis, and
      * ``+X`` completes the right-handed frame.

    ``jaw_axis`` only fixes the roll about the approach, so it is orthogonalised
    against ``approach_axis`` rather than required to be exactly perpendicular;
    measured geometry rarely is.  When the two axes are parallel the roll is
    undefined and no frame exists, so this fails closed instead of picking one.
    """

    try:
        approach = [float(value) for value in approach_axis]
        jaw = [float(value) for value in jaw_axis]
    except (TypeError, ValueError) as exc:
        raise NativeFrankaActionMathError(
            ["native_franka_grasp_orientation_axes_invalid"]
        ) from exc
    if (
        len(approach) != 3
        or len(jaw) != 3
        or not all(math.isfinite(value) for value in (*approach, *jaw))
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_grasp_orientation_axes_invalid"]
        )
    approach_norm = math.sqrt(sum(value * value for value in approach))
    jaw_norm = math.sqrt(sum(value * value for value in jaw))
    if (
        approach_norm <= GRASP_AXIS_DEGENERACY_TOLERANCE
        or jaw_norm <= GRASP_AXIS_DEGENERACY_TOLERANCE
    ):
        raise NativeFrankaActionMathError(
            ["native_franka_grasp_orientation_axes_invalid"]
        )
    ee_z = [value / approach_norm for value in approach]
    jaw_unit = [value / jaw_norm for value in jaw]
    projection = sum(a * b for a, b in zip(ee_z, jaw_unit, strict=True))
    if abs(projection) >= 1.0 - GRASP_AXIS_DEGENERACY_TOLERANCE:
        # Parallel approach and jaw axes cannot span a grasp frame: the roll
        # about the approach is unconstrained and ee_x collapses to zero.
        raise NativeFrankaActionMathError(
            ["native_franka_grasp_orientation_axes_degenerate"]
        )
    residual = [
        value - projection * axis for value, axis in zip(jaw_unit, ee_z, strict=True)
    ]
    residual_norm = math.sqrt(sum(value * value for value in residual))
    ee_y = [value / residual_norm for value in residual]
    ee_x = [
        ee_y[1] * ee_z[2] - ee_y[2] * ee_z[1],
        ee_y[2] * ee_z[0] - ee_y[0] * ee_z[2],
        ee_y[0] * ee_z[1] - ee_y[1] * ee_z[0],
    ]
    # Columns are the gripper axes expressed in contact-frame coordinates,
    # which is what ``contact_world * offset`` consumes.
    rotation = [
        [ee_x[0], ee_y[0], ee_z[0]],
        [ee_x[1], ee_y[1], ee_z[1]],
        [ee_x[2], ee_y[2], ee_z[2]],
    ]
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            (rotation[2][1] - rotation[1][2]) / scale,
            (rotation[0][2] - rotation[2][0]) / scale,
            (rotation[1][0] - rotation[0][1]) / scale,
            0.25 * scale,
        ]
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = (
            math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        )
        quaternion = [
            0.25 * scale,
            (rotation[0][1] + rotation[1][0]) / scale,
            (rotation[0][2] + rotation[2][0]) / scale,
            (rotation[2][1] - rotation[1][2]) / scale,
        ]
    elif rotation[1][1] > rotation[2][2]:
        scale = (
            math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        )
        quaternion = [
            (rotation[0][1] + rotation[1][0]) / scale,
            0.25 * scale,
            (rotation[1][2] + rotation[2][1]) / scale,
            (rotation[0][2] - rotation[2][0]) / scale,
        ]
    else:
        scale = (
            math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        )
        quaternion = [
            (rotation[0][2] + rotation[2][0]) / scale,
            (rotation[1][2] + rotation[2][1]) / scale,
            0.25 * scale,
            (rotation[1][0] - rotation[0][1]) / scale,
        ]
    norm = math.sqrt(sum(value * value for value in quaternion))
    quaternion = [value / norm for value in quaternion]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    return quaternion


__all__ = [
    "GRASP_AXIS_DEGENERACY_TOLERANCE",
    "NativeFrankaActionMathError",
    "bounded_absolute_joint_setpoint",
    "bounded_cartesian_pose_target",
    "clip_joint_positions_to_limits",
    "controlled_body_pose_for_grasp_frame_target",
    "controlled_body_pose_for_rigid_grasp_frame_target",
    "grasp_orientation_contact_xyzw",
    "is_unauthored_identity_quaternion_xyzw",
]
