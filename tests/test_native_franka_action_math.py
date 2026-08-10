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
    controlled_body_pose_for_grasp_frame_target,
    resolve_gripper_command_endpoints,
)


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


@pytest.mark.parametrize(
    ("separations", "closed", "opened"),
    [
        # Original Arena DROID / Robotiq fixture: 0 and -1 open, 1 closes.
        ({"-1.0": 0.0831, "0.0": 0.0832, "1.0": 0.001}, 1.0, 0.0),
        # Generic Isaac binary action: -1 closes, 0 and 1 open.
        ({"-1.0": 0.001, "0.0": 0.0831, "1.0": 0.0832}, -1.0, 0.0),
    ],
)
def test_semantic_pad_travel_resolves_both_arena_binary_conventions(
    separations: dict[str, float], closed: float, opened: float
) -> None:
    result = resolve_gripper_command_endpoints(
        tool_point_separations_m=separations
    )

    assert result["status"] == "measured"
    assert result["closed_command"] == closed
    assert result["open_command"] == opened
    assert result["separation_travel_m"] == pytest.approx(0.0822)


def test_indistinguishable_gripper_commands_stay_ambiguous() -> None:
    result = resolve_gripper_command_endpoints(
        tool_point_separations_m={"-1.0": 0.08, "0.0": 0.0802, "1.0": 0.0801}
    )

    assert result["status"] == "ambiguous"
    assert result["blockers"] == [
        "native_task_gripper_convention_travel_below_floor"
    ]
