from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.droid_oscar_closed_loop_adapter import WAM_SOURCE_VIEW_PATHS
from blueprint_pipeline.droid_oscar_skeleton_conditioning import (
    AbsoluteJointOscarConditioningBuilder,
    OSCAR_NUM_FRAMES,
    build_absolute_joint_skeleton_trajectory,
    encode_fk_reliability_actions_10d,
    validate_camera_calibration,
)
from blueprint_pipeline.droid_policy_bridge import DROID_ROBOARENA_CONCAT_VIEWS


CURRENT_JOINTS = np.asarray([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0])


def _action() -> np.ndarray:
    action = np.repeat(CURRENT_JOINTS[None, :], 16, axis=0)
    action[:, 0] = np.linspace(0.01, 0.16, 16)
    gripper = np.linspace(0.0, 1.0, 16)[:, None]
    return np.hstack((action, gripper))


def _static_calibration(width: int = 64, height: int = 48) -> dict[str, Any]:
    return {
        "image_size": [width, height],
        "camera_is_per_frame": False,
        "camera_intrinsic": [[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]],
        "camera_extrinsic": np.eye(4).tolist(),
    }


def test_absolute_chunk_expands_to_81_frames_and_advances_exact_integer_prefix() -> None:
    action = _action()
    result = build_absolute_joint_skeleton_trajectory(
        action,
        current_joint_position=CURRENT_JOINTS,
        current_gripper_position=[0.0],
        executed_prefix_steps=8,
    )

    assert result["joint_angles_array"].shape == (OSCAR_NUM_FRAMES, 7)
    assert result["gripper_openness_array"].shape == (OSCAR_NUM_FRAMES,)
    assert result["joint_angles_array"][0] == pytest.approx(CURRENT_JOINTS)
    assert result["joint_angles_array"][1] == pytest.approx(action[0, :7])
    assert result["joint_angles_array"][16] == pytest.approx(action[15, :7])
    assert result["joint_angles_array"][-1] == pytest.approx(action[15, :7])
    assert result["next_joint_position"] == pytest.approx(action[7, :7])
    assert result["next_gripper_position"] == pytest.approx(action[7, 7:8])
    assert result["joint_targets_clamped"] is False


@pytest.mark.parametrize("prefix", [2.4, True, 0, 17])
def test_absolute_chunk_rejects_fractional_or_out_of_range_prefix(prefix: Any) -> None:
    with pytest.raises(ValueError, match="executed_prefix"):
        build_absolute_joint_skeleton_trajectory(
            _action(),
            current_joint_position=CURRENT_JOINTS,
            current_gripper_position=[0.0],
            executed_prefix_steps=prefix,
        )


def test_absolute_chunk_fails_closed_instead_of_clamping() -> None:
    action = _action()
    action[3, 0] = 9.0
    with pytest.raises(ValueError, match="absolute_joint_target_out_of_range"):
        build_absolute_joint_skeleton_trajectory(
            action,
            current_joint_position=CURRENT_JOINTS,
            current_gripper_position=[0.0],
            executed_prefix_steps=8,
        )


def test_fk_trace_encodes_identity_rot6d_and_translation_deltas() -> None:
    joints = np.vstack((CURRENT_JOINTS, _action()[:, :7]))
    gripper = np.concatenate(([0.0], _action()[:, 7]))

    def fk(row: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[0, 3] = row[0]
        return pose

    actions = encode_fk_reliability_actions_10d(joints, gripper, forward_kinematics=fk)
    assert actions.shape == (16, 10)
    assert actions[:, 3:9] == pytest.approx(np.tile([1, 0, 0, 0, 1, 0], (16, 1)))
    assert actions[0, 0] == pytest.approx(0.01)
    assert actions[-1, 9] == pytest.approx(1.0)


def test_calibration_requires_explicit_valid_world_to_camera_geometry() -> None:
    valid = validate_camera_calibration(_static_calibration())
    assert valid["guessed"] is False
    assert len(valid["camera_calibration_sha256"]) == 64

    missing_mode = _static_calibration()
    del missing_mode["camera_is_per_frame"]
    with pytest.raises(ValueError, match="camera_is_per_frame_missing"):
        validate_camera_calibration(missing_mode)

    bad_rotation = _static_calibration()
    bad_rotation["camera_extrinsic"][0][0] = 2.0
    with pytest.raises(ValueError, match="world_to_camera_se3"):
        validate_camera_calibration(bad_rotation)


def test_dynamic_wrist_calibration_requires_every_oscar_frame() -> None:
    dynamic = _static_calibration()
    dynamic["camera_is_per_frame"] = True
    dynamic["camera_extrinsic"] = np.repeat(np.eye(4)[None, :, :], OSCAR_NUM_FRAMES, axis=0)
    result = validate_camera_calibration(dynamic)
    assert result["camera_extrinsic_array"].shape == (OSCAR_NUM_FRAMES, 4, 4)

    dynamic["camera_extrinsic"] = dynamic["camera_extrinsic"][:-1]
    with pytest.raises(ValueError, match="camera_extrinsic_shape_invalid"):
        validate_camera_calibration(dynamic)


def test_builder_requires_full_resolution_sources_and_retains_attribution(tmp_path: Path) -> None:
    source_paths = {}
    for index, view in enumerate(DROID_ROBOARENA_CONCAT_VIEWS):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (64, 48), color=(10 + index,) * 3).save(path)
        source_paths[view] = path
    observation = {
        **{view: np.zeros((224, 224, 3), dtype=np.uint8) for view in DROID_ROBOARENA_CONCAT_VIEWS},
        "observation/joint_position": CURRENT_JOINTS,
        "observation/gripper_position": np.asarray([0.0]),
        "prompt": "Pick up the bottle.",
        WAM_SOURCE_VIEW_PATHS: source_paths,
    }

    def fk(row: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[0, 3] = row[0]
        return pose

    render_calls = []

    def renderer(**kwargs: Any) -> Path:
        render_calls.append(kwargs)
        kwargs["output_path"].write_bytes(b"skeleton")
        return kwargs["output_path"]

    builder = AbsoluteJointOscarConditioningBuilder(
        calibration_provider=lambda **_kwargs: _static_calibration(),
        forward_kinematics=fk,
        skeleton_renderer=renderer,
    )
    result = builder(
        observation=observation,
        policy_action=_action(),
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path / "conditioning",
    )

    assert set(result["views"]) == set(DROID_ROBOARENA_CONCAT_VIEWS)
    assert len(render_calls) == 3
    assert all(call["joint_angles"].shape == (OSCAR_NUM_FRAMES, 7) for call in render_calls)
    assert result["reliability_actions_10d"].shape == (16, 10)
    assert result["evidence"]["physical_future_observation_used"] is False
    assert result["evidence"]["joint_targets_clamped"] is False

    del observation[WAM_SOURCE_VIEW_PATHS]
    with pytest.raises(ValueError, match="full_resolution_wam_source_views_required"):
        builder(
            observation=observation,
            policy_action=_action(),
            executed_prefix_steps=8,
            query_index=0,
            output_dir=tmp_path / "missing",
        )
