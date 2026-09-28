from __future__ import annotations

import pytest

from blueprint_pipeline.adp009d_isaac_episode_adapter import (
    bounded_absolute_joint_setpoint as original_bounded_setpoint,
)
from blueprint_pipeline.adp009d_isaac_episode_adapter import (
    controlled_body_pose_for_grasp_frame_target as original_grasp_target,
)
from blueprint_pipeline.native_franka_action_math import (
    NativeFrankaActionMathError,
    bounded_absolute_joint_setpoint,
    bounded_cartesian_pose_target,
    clip_joint_positions_to_limits,
    controlled_body_pose_for_grasp_frame_target,
    controlled_body_pose_for_rigid_grasp_frame_target,
)


def _quaternion_angle(left, right) -> float:
    import math

    dot = abs(sum(a * b for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, dot))


def test_cartesian_pose_target_limits_translation_and_rotation_before_dls() -> None:
    import math

    position, quaternion = bounded_cartesian_pose_target(
        current_position_world_m=[0.0, 0.0, 0.0],
        current_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        target_position_world_m=[0.0, 0.0, 1.0],
        target_quaternion_world_xyzw=[0.0, 0.0, 1.0, 0.0],
        max_translation_step_m=0.02,
        max_orientation_step_rad=0.10,
    )

    assert position == pytest.approx([0.0, 0.0, 0.02])
    assert _quaternion_angle([0.0, 0.0, 0.0, 1.0], quaternion) == pytest.approx(
        0.10
    )
    assert math.sqrt(sum(value * value for value in quaternion)) == pytest.approx(
        1.0
    )


def test_cartesian_pose_target_uses_shortest_equivalent_quaternion_path() -> None:
    position, quaternion = bounded_cartesian_pose_target(
        current_position_world_m=[1.0, 2.0, 3.0],
        current_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        target_position_world_m=[1.0, 2.0, 3.0],
        target_quaternion_world_xyzw=[0.0, 0.0, 0.0, -1.0],
        max_translation_step_m=0.02,
        max_orientation_step_rad=0.10,
    )

    assert position == [1.0, 2.0, 3.0]
    assert quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_local_ik_solution_is_clipped_to_measured_joint_limits() -> None:
    assert clip_joint_positions_to_limits(
        desired_joint_positions_rad=[-4.0, 0.5, 8.0],
        lower_joint_position_limits_rad=[-2.0, -1.0, -3.0],
        upper_joint_position_limits_rad=[2.0, 1.0, 3.0],
    ) == [-2.0, 0.5, 3.0]


def test_scene_neutral_joint_setpoint_matches_original_rigid_fixture() -> None:
    kwargs = {
        "measured_joint_positions_rad": [0.0, -0.2],
        "desired_joint_positions_rad": [0.5, -0.5],
        "previous_commanded_joint_positions_rad": [0.02, -0.21],
        "max_command_slew_per_step_rad": 0.03,
        "max_setpoint_lead_rad": 0.20,
    }
    assert bounded_absolute_joint_setpoint(**kwargs) == original_bounded_setpoint(
        **kwargs
    )


def test_scene_neutral_grasp_transform_matches_articulated_fixture() -> None:
    kwargs = {
        "current_body_position_world_m": [1.7, 1.8, 1.1],
        "current_body_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        "current_grasp_frame_position_world_m": [1.72, 1.8, 1.04],
        "target_grasp_frame_position_world_m": [2.0941762, 1.8068521, 1.022997],
        "target_body_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    assert controlled_body_pose_for_grasp_frame_target(
        **kwargs
    ) == original_grasp_target(**kwargs)


def test_rigid_grasp_transform_solves_translation_and_coupler_rotation() -> None:
    root_half = 2.0**-0.5
    position, quaternion = controlled_body_pose_for_rigid_grasp_frame_target(
        current_body_position_world_m=[1.0, 2.0, 3.0],
        current_body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        current_grasp_frame_position_world_m=[1.0, 2.0, 3.2],
        current_grasp_frame_quaternion_world_xyzw=[root_half, 0.0, 0.0, root_half],
        target_grasp_frame_position_world_m=[4.0, 5.0, 6.0],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    assert position == pytest.approx([4.0, 4.8, 6.0])
    assert quaternion == pytest.approx([-root_half, 0.0, 0.0, root_half])


def test_scene_neutral_action_math_fails_closed() -> None:
    with pytest.raises(NativeFrankaActionMathError) as excinfo:
        bounded_absolute_joint_setpoint(
            measured_joint_positions_rad=[0.0],
            desired_joint_positions_rad=[1.0],
            previous_commanded_joint_positions_rad=[5.0],
            max_command_slew_per_step_rad=0.03,
            max_setpoint_lead_rad=0.2,
        )
    assert excinfo.value.errors == (
        "native_franka_joint_setpoint_constraints_infeasible",
    )


def test_setpoint_lead_is_bounded_by_what_the_actuator_can_pull() -> None:
    """C30's root cause: commands past the actuator's stiffness-limited lead.

    An implicit PD actuator produces at most ``effort_limit`` newton-metres,
    so a setpoint further than ``effort_limit / stiffness`` ahead of the
    measured joint asks for torque the joint clips.  The Arena DROID
    embodiment gives panda_joint[5-7] stiffness 400 against a 12 N-m limit --
    saturation beyond 0.03 rad -- while the plans commanded a 0.2 rad lead.
    Joint 6 sat pinned at 12.000 N-m demanding up to 160 while drifting
    0.241 rad off target.
    """

    from blueprint_pipeline.native_franka_action_math import (
        bounded_absolute_joint_setpoint,
    )

    measured = [0.0] * 7
    desired = [1.0] * 7
    previous = [0.0] * 7
    # Shoulder joints: 87 / 400 = 0.2175 rad.  Wrist: 12 / 400 = 0.03 rad.
    feasible = [0.2175] * 4 + [0.03] * 3

    command = bounded_absolute_joint_setpoint(
        measured_joint_positions_rad=measured,
        desired_joint_positions_rad=desired,
        previous_commanded_joint_positions_rad=previous,
        max_command_slew_per_step_rad=0.01,
        max_setpoint_lead_rad=0.2,
        max_setpoint_lead_rad_per_joint=feasible,
    )

    # Slew still bounds a single step, and the per-joint lead never loosens it.
    assert command == pytest.approx([0.01] * 7)

    # Once a joint stalls and the command has run ahead of it, the wrist is
    # pinned at its feasible lead -- the point past which it would only clip --
    # while the shoulder, with seven times the effort budget, keeps advancing.
    stalled = bounded_absolute_joint_setpoint(
        measured_joint_positions_rad=[0.0] * 7,
        desired_joint_positions_rad=[1.0] * 7,
        previous_commanded_joint_positions_rad=[0.04] * 7,
        max_command_slew_per_step_rad=0.01,
        max_setpoint_lead_rad=0.2,
        max_setpoint_lead_rad_per_joint=feasible,
    )
    assert stalled[:4] == pytest.approx([0.05] * 4)
    assert stalled[4:] == pytest.approx([0.03] * 3)


def test_per_joint_lead_never_falls_below_one_slew_step() -> None:
    """A lead under the slew would leave the command unable to advance."""

    from blueprint_pipeline.native_franka_action_math import (
        bounded_absolute_joint_setpoint,
    )

    command = bounded_absolute_joint_setpoint(
        measured_joint_positions_rad=[0.0],
        desired_joint_positions_rad=[1.0],
        previous_commanded_joint_positions_rad=[0.0],
        max_command_slew_per_step_rad=0.02,
        max_setpoint_lead_rad=0.2,
        max_setpoint_lead_rad_per_joint=[0.001],
    )

    assert command == pytest.approx([0.02])


def test_a_malformed_per_joint_lead_fails_closed() -> None:
    from blueprint_pipeline.native_franka_action_math import (
        NativeFrankaActionMathError,
        bounded_absolute_joint_setpoint,
    )

    for bad in ([0.03, 0.03], [0.0], [float("nan")], [-0.1]):
        with pytest.raises(NativeFrankaActionMathError):
            bounded_absolute_joint_setpoint(
                measured_joint_positions_rad=[0.0],
                desired_joint_positions_rad=[1.0],
                previous_commanded_joint_positions_rad=[0.0],
                max_command_slew_per_step_rad=0.01,
                max_setpoint_lead_rad=0.2,
                max_setpoint_lead_rad_per_joint=bad,
            )


def test_a_saturated_joint_is_pulled_back_rather_than_refusing() -> None:
    """C31 died at startup on an empty intersection of the two windows.

    A joint already in saturation carries a previous command further out
    than it can pull, so intersecting the slew window with the feasible-lead
    window is empty.  The lead is therefore a clamp around the measured
    joint, not another window bound: the command is pulled back toward
    reality -- by more than one slew step if that is what it takes -- which
    is how a saturated joint stops being commanded past its own reach.
    """

    from blueprint_pipeline.native_franka_action_math import (
        bounded_absolute_joint_setpoint,
    )

    command = bounded_absolute_joint_setpoint(
        measured_joint_positions_rad=[0.0],
        desired_joint_positions_rad=[1.0],
        # A tenth of a radian of accumulated command -- inside the caller's
        # scalar lead, far outside this joint's 0.015 rad reach.
        previous_commanded_joint_positions_rad=[0.1],
        max_command_slew_per_step_rad=0.01,
        max_setpoint_lead_rad=0.2,
        max_setpoint_lead_rad_per_joint=[0.015],
    )

    assert command == pytest.approx([0.015])


def test_stale_command_outside_scalar_lead_recovers_with_actuator_limit() -> None:
    """A native joint jump rebases only when an actuator-safe lead exists."""

    command = bounded_absolute_joint_setpoint(
        measured_joint_positions_rad=[0.0],
        desired_joint_positions_rad=[1.0],
        previous_commanded_joint_positions_rad=[5.0],
        max_command_slew_per_step_rad=0.03,
        max_setpoint_lead_rad=0.2,
        max_setpoint_lead_rad_per_joint=[0.015],
    )

    # The stale setpoint is discarded toward reality, and the new setpoint is
    # still capped at the actuator-safe lead (at least one slew step).
    assert command == pytest.approx([0.03])
