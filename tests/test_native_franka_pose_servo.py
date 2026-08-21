from __future__ import annotations

import pytest

from blueprint_pipeline.native_franka_pose_servo import (
    NativeFrankaPoseServoError,
    PINK_CONFIGURATION_LIMIT_MARGIN_RAD,
    PINK_ORIENTATION_COST,
    PINK_POSITION_COST,
    PINK_POSTURE_COST,
    contract_xyzw_to_native_xyzw,
    contract_xyzw_to_pink_wxyz,
    native_xyzw_to_contract_xyzw,
    pink_configuration_joint_positions,
    resolve_native_franka_pose_binding,
)


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


def test_pose_servo_uses_pink_limits_and_posture_not_plain_dls() -> None:
    import inspect

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    source = inspect.getsource(NativeFrankaDifferentialIkServo)
    assert "PinkIKController" in source
    assert 'load_pink_supported_robot("franka")' in source
    assert 'tool_frame="panda_hand"' in source
    assert "DifferentialIKController" not in source
    assert "pink_hand_pose_at_binding" in source
    assert "current_grasp_frame_pose_world" in source
    reset_source = inspect.getsource(
        NativeFrankaDifferentialIkServo.reset_command_state
    )
    assert "_reset_pink_controller" not in reset_source


@pytest.mark.parametrize("value", ([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]))
def test_quaternion_boundary_rejects_zero_or_wrong_length(value) -> None:
    with pytest.raises(
        NativeFrankaPoseServoError,
        match="native_franka_pose_servo_quaternion_invalid",
    ):
        native_xyzw_to_contract_xyzw(value)
