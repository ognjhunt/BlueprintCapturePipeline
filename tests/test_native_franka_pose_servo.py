from __future__ import annotations

import pytest

from blueprint_pipeline.native_franka_pose_servo import (
    NativeFrankaPoseServoError,
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
