from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.native_franka_pose_servo import (
    NativeFrankaDifferentialIkServo,
    NativeFrankaPoseServoError,
    PINK_CONFIGURATION_LIMIT_MARGIN_RAD,
    PINK_INTEGRATION_DT_SECONDS,
    PINK_ORIENTATION_COST,
    PINK_POSITION_COST,
    PINK_POSTURE_COST,
    contract_xyzw_to_native_xyzw,
    contract_xyzw_to_pink_wxyz,
    deterministic_pink_joint_seeds,
    native_xyzw_to_contract_xyzw,
    pink_configuration_joint_positions,
    resolve_native_franka_pose_binding,
)


def test_grasp_tcp_interpolates_between_measured_gripper_endpoints() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._body_to_grasp_position = [9.0, 9.0, 9.0]
    servo._body_to_grasp_positions_by_command = {
        0.0: [0.10, 0.00, 0.02],
        1.0: [0.10, 0.02, 0.00],
    }
    servo._last_gripper_command = 0.0

    assert servo._grasp_position_for_command(0.0) == pytest.approx(
        [0.10, 0.00, 0.02]
    )
    assert servo._grasp_position_for_command(1.0) == pytest.approx(
        [0.10, 0.02, 0.00]
    )
    assert servo._grasp_position_for_command(0.5) == pytest.approx(
        [0.10, 0.01, 0.01]
    )


def test_live_pad_readback_uses_moving_finger_bodies_and_local_offsets() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo.binding = {
        "controlled_body_name": "base_link",
        "controlled_body_index": 0,
    }
    servo._finger_body_indices = {"left": 1, "right": 2}
    servo._pad_center_offsets_in_finger_body = {
        "left": [0.01, 0.0, 0.0],
        "right": [-0.01, 0.0, 0.0],
    }
    servo._robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pose_w=np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                dtype=float,
            )
        )
    )
    servo._to_torch = lambda value: value

    readback = servo.current_gripper_pad_readback()

    assert readback["measured"]["finger_body_positions_world_m"] == {
        "left": [-0.05, 0.0, 0.0],
        "right": [0.05, 0.0, 0.0],
    }
    assert readback["measured"]["pad_centers_world_m"]["left"] == pytest.approx(
        [-0.04, 0.0, 0.0]
    )
    assert readback["measured"]["pad_centers_world_m"]["right"] == pytest.approx(
        [0.04, 0.0, 0.0]
    )
    assert readback["measured"]["pad_midpoint_world_m"] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert readback["measured"]["pad_separation_m"] == pytest.approx(0.08)


def _bodies() -> list[str]:
    return [
        *[f"panda_link{index}" for index in range(9)],
        "panda_hand",
        "base_link",
        "left_inner_finger",
        "right_inner_finger",
    ]


def test_fixed_base_binding_uses_the_physx_jacobian_row_offset() -> None:
    result = resolve_native_franka_pose_binding(
        body_names=_bodies(),
        joint_names=[f"panda_joint{index}" for index in range(1, 8)],
        fixed_base=True,
    )

    assert result["controlled_body_name"] == "panda_hand"
    assert result["controlled_body_index"] == 9
    assert result["jacobian_body_index"] == 8
    assert result["finger_body_indices"] == [11, 12]


def test_missing_semantic_finger_fails_instead_of_guessing_last_bodies() -> None:
    with pytest.raises(NativeFrankaPoseServoError) as excinfo:
        resolve_native_franka_pose_binding(
            body_names=[name for name in _bodies() if name != "left_inner_finger"],
            joint_names=[f"panda_joint{index}" for index in range(1, 8)],
            fixed_base=True,
        )

    assert excinfo.value.errors == (
        "native_franka_pose_servo_finger_body_missing:left_inner_finger",
    )


def test_canned_beverage_joint_order_cannot_hide_a_wrong_robot_binding() -> None:
    with pytest.raises(NativeFrankaPoseServoError) as excinfo:
        resolve_native_franka_pose_binding(
            body_names=_bodies(),
            joint_names=["legacy_joint"] * 7,
            fixed_base=True,
        )

    assert excinfo.value.errors == (
        "native_franka_pose_servo_arm_joint_binding_invalid",
    )


def test_nonidentity_native_quaternion_preserves_beta2_xyzw_order() -> None:
    native_xyzw = [0.5, 0.5, -0.5, 0.5]

    contract_xyzw = native_xyzw_to_contract_xyzw(native_xyzw)

    assert contract_xyzw == pytest.approx(native_xyzw)
    assert contract_xyzw_to_native_xyzw(contract_xyzw) == pytest.approx(
        native_xyzw
    )


def test_pink_spatial_state_boundary_converts_xyzw_to_wxyz() -> None:
    assert contract_xyzw_to_pink_wxyz([0.1, -0.2, 0.3, 0.9]) == pytest.approx(
        [0.9233805169, 0.1025978352, -0.2051956704, 0.3077935056]
    )


def test_pose_servo_uses_pinned_pink_manipulation_weights() -> None:
    assert PINK_POSITION_COST == 5.0
    assert PINK_ORIENTATION_COST == 1.0
    assert PINK_POSTURE_COST == 5.0e-3
    assert PINK_INTEGRATION_DT_SECONDS == pytest.approx(1.0 / 20.0)


def test_pink_configuration_clamps_float_roundtrip_just_inside_limits() -> None:
    result = pink_configuration_joint_positions(
        measured_joint_positions_rad=[0.0, 3.7525010108947754],
        lower_joint_position_limits_rad=[-2.0, -0.0175],
        upper_joint_position_limits_rad=[2.0, 3.7525],
    )

    assert result[0] == 0.0
    assert result[1] == pytest.approx(
        3.7525 - PINK_CONFIGURATION_LIMIT_MARGIN_RAD
    )
    assert -0.0175 < result[1] < 3.7525


def test_pink_configuration_limit_contract_refuses_collapsed_margin() -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_pink_configuration_limits_invalid",
    ):
        pink_configuration_joint_positions(
            measured_joint_positions_rad=[0.0],
            lower_joint_position_limits_rad=[-1.0e-6],
            upper_joint_position_limits_rad=[1.0e-6],
        )


def test_global_pink_seeds_are_deterministic_bounded_and_preferred_first() -> None:
    lower = [-2.0] * 7
    upper = [2.0] * 7
    preferred = [[0.0] * 7, [1.0] * 7]

    first = deterministic_pink_joint_seeds(
        lower_joint_position_limits_rad=lower,
        upper_joint_position_limits_rad=upper,
        preferred_seeds=preferred,
        seed_count=12,
    )
    second = deterministic_pink_joint_seeds(
        lower_joint_position_limits_rad=lower,
        upper_joint_position_limits_rad=upper,
        preferred_seeds=preferred,
        seed_count=12,
    )

    assert first == second
    assert first[:2] == preferred
    assert len(first) == 12
    assert len({tuple(row) for row in first}) == 12
    assert all(
        -2.0 < value < 2.0 for row in first for value in row
    )


def test_global_pink_seeds_reject_invalid_joint_boxes() -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_global_seeds_invalid",
    ):
        deterministic_pink_joint_seeds(
            lower_joint_position_limits_rad=[-1.0] * 6,
            upper_joint_position_limits_rad=[1.0] * 6,
        )


def test_multistart_prefers_continuous_solved_configuration() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        joints = list(kwargs["seed_joint_positions_rad"])
        return {
            "solved": True,
            "joint_positions_rad": joints,
            "position_error_m": 0.001,
            "orientation_error_rad": 0.01,
            "iterations": 4,
        }

    servo.solve_grasp_target_from_joint_seed = solve
    result = servo.solve_grasp_target_multistart(
        target_position_world_m=[0.5, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[[1.0] * 7, [0.2] * 7],
        reference_joint_positions_rad=[0.0] * 7,
        seed_count=2,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
    )

    assert result["solved"] is True
    assert all(call["position_tolerance_m"] == 0.005 for call in calls)
    assert all(call["orientation_tolerance_rad"] == 0.08 for call in calls)
    assert result["position_tolerance_m"] == 0.005
    assert result["orientation_tolerance_rad"] == 0.08
    assert result["selected"]["seed_index"] == 1
    assert result["selected"]["joint_positions_rad"] == pytest.approx(
        [0.2] * 7
    )
    assert result["selected"]["joint_space_reference_distance_rad"] == pytest.approx(
        (7 * 0.2**2) ** 0.5
    )


def test_multistart_preserves_closest_whole_arm_branch_before_maximum_delta() -> None:
    """A tiny max-joint advantage must not select much larger total motion."""

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    servo.solve_grasp_target_from_joint_seed = lambda **kwargs: {
        "solved": True,
        "joint_positions_rad": list(kwargs["seed_joint_positions_rad"]),
        "position_error_m": 0.001,
        "orientation_error_rad": 0.01,
        "iterations": 4,
    }

    result = servo.solve_grasp_target_multistart(
        target_position_world_m=[0.5, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[
            [0.90] * 7,
            [0.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        reference_joint_positions_rad=[0.0] * 7,
        seed_count=2,
    )

    assert result["attempts"][0]["maximum_reference_joint_delta_rad"] < result[
        "attempts"
    ][1]["maximum_reference_joint_delta_rad"]
    assert result["selected"]["seed_index"] == 1
    assert result["selected"]["joint_space_reference_distance_rad"] == pytest.approx(
        0.91
    )


def test_global_pink_iteration_clamps_integrated_float_overshoot() -> None:
    """The next Pink iteration must never receive its own out-of-limit output."""

    class _Rotation:
        @staticmethod
        def from_matrix(_matrix):
            return SimpleNamespace(as_quat=lambda: np.array([0.0, 0.0, 0.0, 1.0]))

    class _Configuration:
        def __init__(self, controller):
            self._controller = controller

        def get_transform_frame_to_world(self, _frame):
            return SimpleNamespace(
                translation=np.array(
                    [1.0, 0.0, 0.0]
                    if self._controller.forward_calls == 1
                    else [0.0, 0.0, 0.0]
                ),
                rotation=np.eye(3),
            )

    class _Controller:
        def __init__(self):
            self.forward_calls = 0
            self._configuration = _Configuration(self)

        def reset(self, _state, _setpoint, _time):
            return True

        def forward(self, state, _setpoint, _time):
            self.forward_calls += 1
            if self.forward_calls == 2:
                assert state[0] == pytest.approx(
                    1.0 - PINK_CONFIGURATION_LIMIT_MARGIN_RAD
                )
                assert -1.0 < state[0] < 1.0
            return [1.0008, *([0.0] * 6)]

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-1.0] * 7
    servo._joint_position_upper = [1.0] * 7
    servo._pink_time_seconds = 0.0
    servo._pink_controller = _Controller()
    servo._Rotation = _Rotation
    servo._pink_hand_target_for_grasp_world = lambda **_kwargs: (
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    servo._pink_setpoint = lambda **_kwargs: object()
    servo._pink_state_from_joint_positions = lambda joints: list(joints)
    servo._joint_positions_from_pink_state = lambda desired: list(desired)
    servo._reset_pink_controller = lambda: None

    result = servo.solve_grasp_target_from_joint_seed(
        target_position_world_m=[0.0, 0.0, 0.0],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        seed_joint_positions_rad=[0.0] * 7,
        maximum_iterations=2,
    )

    assert result["solved"] is True
    assert result["iterations"] == 2
    assert result["iteration_feedback_clamp_count"] == 2
    assert result["maximum_iteration_feedback_clamp_rad"] == pytest.approx(
        0.0008 + PINK_CONFIGURATION_LIMIT_MARGIN_RAD
    )


def test_global_joint_solution_replay_stays_under_native_command_bounds() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    servo._last_command = None
    servo._last_gripper_command = 0.0
    servo._control_period_seconds = 1.0 / 15.0
    servo.read_arm_joint_positions = lambda: [0.0] * 7
    observed_velocity = []
    servo._write_joint_velocity_target = observed_velocity.extend

    action, diagnostic = servo.action_for_joint_target(
        target_joint_positions_rad=[1.0] * 7,
        gripper_command=0.25,
        max_joint_delta_rad=0.05,
        max_joint_setpoint_lead_rad=0.2,
        velocity_feedforward_scale=1.0,
    )

    assert action[:7] == pytest.approx([0.05] * 7)
    assert action[7] == 0.25
    assert observed_velocity == pytest.approx([0.75] * 7)
    assert diagnostic["bounded_joint_positions_rad"] == pytest.approx(
        [0.05] * 7
    )
    assert diagnostic["ik_backend"].endswith("_multistart_replay")


def test_multistart_avoids_a_joint_limit_solution_before_continuity() -> None:
    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._joint_position_lower = [-2.0] * 7
    servo._joint_position_upper = [2.0] * 7
    servo.solve_grasp_target_from_joint_seed = lambda **kwargs: {
        "solved": True,
        "joint_positions_rad": list(kwargs["seed_joint_positions_rad"]),
        "position_error_m": 0.001,
        "orientation_error_rad": 0.01,
        "iterations": 4,
    }

    result = servo.solve_grasp_target_multistart(
        target_position_world_m=[0.5, 0.0, 0.4],
        target_grasp_frame_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_seeds=[[1.99] * 7, [0.5] * 7],
        reference_joint_positions_rad=[1.99] * 7,
        seed_count=2,
    )

    assert result["selected"]["seed_index"] == 1
    assert result["selected"]["minimum_joint_limit_margin_rad"] > 0.05


def test_pose_servo_uses_pink_limits_and_posture_not_plain_dls() -> None:
    import inspect

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    source = inspect.getsource(NativeFrankaDifferentialIkServo)
    assert "PinkIKController" in source
    assert 'load_pink_supported_robot("franka")' in source
    assert 'tool_frame="panda_hand"' in source
    assert "dt=PINK_INTEGRATION_DT_SECONDS" in source
    assert "self._pink_time_seconds += PINK_INTEGRATION_DT_SECONDS" in source
    assert "DifferentialIKController" not in source
    assert "pink_hand_pose_at_binding" in source
    assert "current_grasp_frame_pose_world" in source
    reset_source = inspect.getsource(
        NativeFrankaDifferentialIkServo.reset_command_state
    )
    assert "_reset_pink_controller" not in reset_source


def test_pink_setpoint_combines_cartesian_target_and_preferred_posture() -> None:
    class _RobotState:
        def __init__(self, **kwargs):
            self.joints = kwargs.get("joints")
            self.sites = kwargs.get("sites")

    class _SpatialState:
        @staticmethod
        def from_name(**kwargs):
            return kwargs

    servo = object.__new__(NativeFrankaDifferentialIkServo)
    servo._np = np
    servo._wp = SimpleNamespace(
        float32="float32",
        from_numpy=lambda value, **_kwargs: value,
    )
    servo._mg = SimpleNamespace(RobotState=_RobotState, SpatialState=_SpatialState)
    posture = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    posture_joints = object()
    servo._pink_state_from_joint_positions = lambda values: SimpleNamespace(
        joints=(posture_joints if list(values) == posture else None)
    )

    setpoint = servo._pink_setpoint(
        target_position_base=[0.4, 0.1, 0.2],
        target_quaternion_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        preferred_posture_joint_positions_rad=posture,
    )

    assert setpoint.joints is posture_joints
    assert setpoint.sites["spatial_space"] == ["panda_hand"]
    assert setpoint.sites["positions"][1].tolist()[0] == pytest.approx(
        [0.4, 0.1, 0.2]
    )
    assert setpoint.sites["orientations"][1].tolist()[0] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )


@pytest.mark.parametrize("value", ([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]))
def test_quaternion_boundary_rejects_zero_or_wrong_length(value) -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_quaternion_invalid",
    ):
        native_xyzw_to_contract_xyzw(value)
