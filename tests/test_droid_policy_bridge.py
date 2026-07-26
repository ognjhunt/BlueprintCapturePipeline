import numpy as np
import pytest

from blueprint_pipeline.droid_policy_bridge import (
    DROID_CONTROL_HZ,
    DROID_INNER_CONTROL_HZ,
    droid_joint_position_action_to_mujoco_targets,
    droid_action_to_mujoco_targets,
    validate_droid_action_chunk,
    validate_droid_observation,
)


def _observation() -> dict:
    return {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": np.asarray([1.0]),
        "prompt": "Pick up the can and place it inside the marked tray.",
    }


def test_openpi_droid_contract_accepts_exact_observation_and_chunk() -> None:
    assert DROID_CONTROL_HZ == 15
    assert DROID_INNER_CONTROL_HZ == 1000
    assert validate_droid_observation(_observation()) == []
    assert validate_droid_action_chunk(np.zeros((10, 8))) == []


def test_openpi_droid_contract_rejects_camera_and_action_shape_drift() -> None:
    observation = _observation()
    observation["observation/wrist_image_left"] = np.zeros((480, 640, 3), dtype=np.uint8)
    assert validate_droid_observation(observation) == [
        "invalid_image_shape:observation/wrist_image_left"
    ]
    assert validate_droid_action_chunk(np.zeros((8, 8))) == ["invalid_action_chunk_shape"]


def test_joint_velocity_mapping_matches_public_droid_runtime() -> None:
    result = droid_action_to_mujoco_targets(
        [2.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.7],
        current_joint_position=[0.0] * 7,
        joint_limits=[[-1.0, 1.0]] * 7,
    )
    assert result["joint_position_target_rad"][:3] == pytest.approx([0.2, -0.2, 0.1])
    assert result["gripper_position_target_m"] == 0.0
    assert result["clipped_action"][:2] == [1.0, -1.0]


def test_gripper_mapping_matches_droid_zero_open_one_closed_convention() -> None:
    open_result = droid_action_to_mujoco_targets(
        [0.0] * 8,
        current_joint_position=[0.0] * 7,
        joint_limits=[[-1.0, 1.0]] * 7,
    )
    closed_result = droid_action_to_mujoco_targets(
        [0.0] * 7 + [1.0],
        current_joint_position=[0.0] * 7,
        joint_limits=[[-1.0, 1.0]] * 7,
    )
    assert open_result["gripper_position_target_m"] == 0.04
    assert closed_result["gripper_position_target_m"] == 0.0


def test_absolute_joint_position_mapping_preserves_radians_and_gripper() -> None:
    result = droid_joint_position_action_to_mujoco_targets(
        [0.1, -0.2, 0.3, -1.0, 0.5, 1.5, -0.7, 1.0],
        joint_limits=[[-2.0, 2.0]] * 7,
    )
    assert result["joint_position_target_rad"] == pytest.approx(
        [0.1, -0.2, 0.3, -1.0, 0.5, 1.5, -0.7]
    )
    assert result["gripper_position_target_m"] == 0.0
    assert result["joint_limit_clamped"] is False


def test_joint_velocity_mapping_clamps_model_limits_and_fails_on_nonfinite() -> None:
    result = droid_action_to_mujoco_targets(
        [1.0] * 8,
        current_joint_position=[0.95] * 7,
        joint_limits=[[-1.0, 1.0]] * 7,
    )
    assert result["joint_limit_clamped"] is True
    assert result["joint_position_target_rad"] == pytest.approx([1.0] * 7)
    with pytest.raises(ValueError, match="finite_shape_8"):
        droid_action_to_mujoco_targets(
            [float("nan")] * 8,
            current_joint_position=[0.0] * 7,
            joint_limits=[[-1.0, 1.0]] * 7,
        )
