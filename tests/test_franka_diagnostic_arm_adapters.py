from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from blueprint_pipeline.ctrl_world_droid_action_adapter import (
    FrankaCtrlWorldJointPositionAdapter,
    validate_ctrl_world_runtime_assets,
)
from blueprint_pipeline.droid_oscar_closed_loop_adapter import (
    EXTERIOR_VIEW,
    WRIST_VIEW,
    DroidOscarSkeletonTransitionAdapter,
    SkeletonOnlyIntendedMotionArm,
)
from blueprint_pipeline.franka_droid_skeleton_conditioning import (
    FrankaDroidSkeletonConditioningBuilder,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256
from blueprint_pipeline.vast_provider_adapter import _probe_shell_script


class _FakeModel:
    jnt_range = np.asarray([[-3.0, 3.0]] * 7, dtype=float)


class _FakeData:
    def __init__(self, _model: Any) -> None:
        self.qpos = np.zeros(16, dtype=float)
        self.xpos = np.zeros((11, 3), dtype=float)
        self.xmat = np.repeat(np.eye(3, dtype=float)[None, :, :], 11, axis=0).reshape(11, 9)


class _FakeMujoco:
    mjtObj = SimpleNamespace(mjOBJ_BODY=1)
    names = {
        name: index
        for index, name in enumerate(
            (
                "link0",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
                "hand",
                "left_finger",
                "right_finger",
            )
        )
    }

    @staticmethod
    def MjData(model: Any) -> _FakeData:
        return _FakeData(model)

    @classmethod
    def mj_name2id(cls, _model: Any, _kind: Any, name: str) -> int:
        return cls.names.get(name, -1)

    @staticmethod
    def mj_forward(_model: Any, data: _FakeData) -> None:
        base_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.333],
                [0.0, 0.0, 0.333],
                [-0.22, 0.0, 0.56],
                [-0.16, 0.0, 0.61],
                [0.22, 0.0, 0.70],
                [0.22, 0.0, 0.70],
                [0.31, 0.0, 0.70],
                [0.31, 0.0, 0.59],
                [0.31, -0.04, 0.53],
                [0.31, 0.04, 0.53],
            ],
            dtype=float,
        )
        displacement = np.asarray([0.08 * data.qpos[0], 0.02 * data.qpos[1], 0.0])
        data.xpos[:] = base_points + displacement


def _runtime() -> dict[str, Any]:
    return {
        "mujoco": _FakeMujoco,
        "model": _FakeModel(),
        "ids": {"hand": _FakeMujoco.names["hand"]},
    }


def _camera_contract() -> dict[str, Any]:
    return {
        "schema_version": "captured_site_policy_observation_cameras.v1",
        "scene_id": "interiorgs_0787_841244_franka_can_to_tray_v1",
        "cameras": [],
    }


def _observation() -> dict[str, Any]:
    return {
        EXTERIOR_VIEW: np.full((224, 224, 3), 40, dtype=np.uint8),
        WRIST_VIEW: np.full((224, 224, 3), 50, dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=float),
        "observation/gripper_position": np.zeros(1, dtype=float),
        "prompt": "Pick up the spray can and place it inside the marked tray.",
    }


def test_franka_builder_materializes_camera_aligned_multiview_skeletons(tmp_path: Path) -> None:
    action = np.zeros((10, 8), dtype=float)
    action[:, 0] = np.linspace(0.0, 0.6, 10)
    builder = FrankaDroidSkeletonConditioningBuilder(
        runtime=_runtime(), camera_contract=_camera_contract(), num_frames=17
    )
    built = builder(
        observation=_observation(),
        policy_action=action,
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path,
    )

    assert set(built["views"]) == {EXTERIOR_VIEW, WRIST_VIEW}
    assert built["reliability_actions_10d"].shape == (17, 10)
    assert built["evidence"]["dynamic_wrist_camera_recomputed_each_frame"] is True
    assert built["evidence"]["task_physics_stepped"] is False
    for view in built["views"].values():
        assert Path(view["first_frame_path"]).is_file()
        assert Path(view["skeleton_video_path"]).stat().st_size > 0
        assert len(view["camera_calibration_sha256"]) == 64


def test_skeleton_only_arm_uses_conditioning_media_without_world_credit(tmp_path: Path) -> None:
    action = np.zeros((10, 8), dtype=float)
    builder = FrankaDroidSkeletonConditioningBuilder(
        runtime=_runtime(), camera_contract=_camera_contract(), num_frames=17
    )
    adapter = DroidOscarSkeletonTransitionAdapter(
        conditioning_builder=builder, action_chunk_rows=10
    )
    prepared = adapter.prepare_transition(
        observation=_observation(),
        policy_action=action,
        task_prompt=_observation()["prompt"],
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path / "conditioning",
    )
    prediction = SkeletonOnlyIntendedMotionArm().predict(
        prepared["wam_request"], output_dir=tmp_path / "prediction"
    )

    assert prediction["world_consequence_credit"] is False
    assert prediction["physical_future_observation_used"] is False
    assert set(prediction["generated_view_frames"]) == {EXTERIOR_VIEW, WRIST_VIEW}
    assert all(Path(path).is_file() for path in prediction["generated_view_frames"].values())


def test_ctrl_world_joint_position_adapter_uses_fk_and_emits_exact_7d_shape() -> None:
    action = np.zeros((10, 8), dtype=float)
    action[:, 0] = np.linspace(0.0, 0.9, 10)
    result = FrankaCtrlWorldJointPositionAdapter(_runtime()).adapt(
        policy_action=action,
        history_cartesian_pose_7d=np.zeros((6, 7), dtype=float),
    )

    assert result["action_conditioning_7d"].shape == (16, 7)
    assert result["official_ctrl_world_learned_action_adapter_used"] is False
    assert result["conversion"] == "deterministic_pinned_franka_forward_kinematics"
    assert result["physical_future_observation_used"] is False
    assert len(result["conditioning_sha256"]) == 64


def test_ctrl_world_asset_admission_distinguishes_world_model_from_adapter(tmp_path: Path) -> None:
    action_adapter = tmp_path / "action-adapter.pth"
    action_adapter.write_bytes(b"small learned action adapter")
    result = validate_ctrl_world_runtime_assets(
        world_model_checkpoint=tmp_path / "missing-world-model.pt",
        expected_world_model_sha256="0" * 64,
        action_adapter_checkpoint=action_adapter,
        expected_action_adapter_sha256=file_sha256(action_adapter),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["ctrl_world_model_checkpoint_missing"]
    assert result["action_adapter_sha256"] == file_sha256(action_adapter)
    assert result["world_model_and_action_adapter_are_distinct_assets"] is True


def test_wam_provider_script_uploads_small_terminal_diagnostic_before_full_archive() -> None:
    script = _probe_shell_script(
        "https://heartbeat.invalid",
        enable_blueprint_bundle=True,
        provider_bundle_kind="wam",
    )

    assert "provider_entrypoint_diagnostic.json" in script
    assert "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_DIAGNOSTIC_WRITTEN" in script
    assert "BLUEPRINT_VAST_PROVIDER_EARLY_DIAGNOSTIC_UPLOAD_OK" in script
    assert script.index("BLUEPRINT_VAST_PROVIDER_EARLY_DIAGNOSTIC_UPLOAD_OK") < script.index(
        "BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN"
    )
