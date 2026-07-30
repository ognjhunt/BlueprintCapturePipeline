from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.ctrl_world_droid_action_adapter import (
    CTRL_WORLD_FUTURE_FRAME_INDICES,
    CtrlWorldReleasedJointVelocityAdapter,
)
from blueprint_pipeline.droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_STATE_HISTORY,
    CTRL_WORLD_VIEW_HISTORY_PATHS,
    DroidCtrlWorldCurrentReferenceTransitionAdapter,
)
from blueprint_pipeline.droid_policy_bridge import (
    DROID_EXTERIOR_VIEW_1,
    DROID_EXTERIOR_VIEW_2,
    DROID_WRIST_VIEW,
)
from blueprint_pipeline.policy_wam_closed_loop import (
    ClosedLoopConfig,
    run_policy_wam_closed_loop,
)


def _dynamics(current_joint: np.ndarray, joint_velocity: np.ndarray) -> np.ndarray:
    assert current_joint.shape == (7,)
    assert joint_velocity.shape == (15, 7)
    return current_joint[None, :] + np.cumsum(joint_velocity, axis=0) * 0.01


def _fk(joint_position: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = joint_position[:3]
    return transform


def _released_adapter() -> CtrlWorldReleasedJointVelocityAdapter:
    return CtrlWorldReleasedJointVelocityAdapter(
        dynamics_adapter=_dynamics,
        forward_kinematics=_fk,
        gripper_max=0.75,
    )


def _policy_preprocessor(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.uint8)


def _transition_adapter() -> DroidCtrlWorldCurrentReferenceTransitionAdapter:
    return DroidCtrlWorldCurrentReferenceTransitionAdapter(
        action_adapter=_released_adapter(),
        openpi_config_name="pi05_droid",
        action_chunk_rows=15,
        policy_image_preprocessor=_policy_preprocessor,
    )


def _observation() -> dict[str, Any]:
    return {
        DROID_EXTERIOR_VIEW_1: np.zeros((224, 224, 3), dtype=np.uint8),
        DROID_EXTERIOR_VIEW_2: np.ones((224, 224, 3), dtype=np.uint8),
        DROID_WRIST_VIEW: np.full((224, 224, 3), 2, dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float64),
        "observation/gripper_position": np.zeros(1, dtype=np.float64),
        "blueprint/ctrl_world_cartesian_pose_7d": np.zeros(7, dtype=np.float64),
        "prompt": "Pick up the blue block and place it in the white plate.",
    }


def test_released_velocity_adapter_preserves_native_chunk_and_public_state_rule() -> None:
    action = np.zeros((15, 8), dtype=np.float64)
    action[:, 0] = 1.0
    action[:, 7] = np.linspace(0.0, 1.0, 15)
    result = _released_adapter().adapt(
        policy_action=action,
        current_joint_position=np.zeros(7),
        current_gripper_position=np.zeros(1),
        history_cartesian_pose_7d=np.zeros((6, 7)),
        source_action_space="joint_velocity_plus_gripper_position",
    )

    assert result["native_policy_action"].shape == (15, 8)
    assert np.array_equal(result["native_policy_action"], action)
    assert result["action_conditioning_7d"].shape == (11, 7)
    assert result["reliability_actions_10d"].shape == (5, 10)
    assert result["future_frame_indices"] == list(CTRL_WORLD_FUTURE_FRAME_INDICES)
    assert result["next_joint_position"][0] == pytest.approx(0.08)
    assert result["next_gripper_position"][0] <= 0.75
    assert result["official_ctrl_world_learned_action_adapter_used"] is True
    assert result["physical_future_observation_used"] is False


def test_released_velocity_adapter_repeats_last_row_for_ten_row_policies() -> None:
    action = np.zeros((10, 8), dtype=np.float64)
    action[-1, 1] = 0.5
    result = _released_adapter().adapt(
        policy_action=action,
        current_joint_position=np.zeros(7),
        current_gripper_position=np.zeros(1),
        history_cartesian_pose_7d=np.zeros((6, 7)),
        source_action_space="joint_velocity_plus_gripper_position",
    )

    assert result["ten_row_padding_rule"] == "repeat_final_row_to_15"
    assert result["native_policy_action_shape"] == [10, 8]


def test_released_velocity_adapter_rejects_policy_droid_absolute_positions() -> None:
    with pytest.raises(ValueError, match="requires_joint_velocity_policy"):
        _released_adapter().adapt(
            policy_action=np.zeros((15, 8)),
            current_joint_position=np.zeros(7),
            current_gripper_position=np.zeros(1),
            history_cartesian_pose_7d=np.zeros((6, 7)),
            source_action_space="absolute_joint_position_plus_gripper_position",
        )


def test_transition_adapter_freezes_three_view_history_and_native_output(
    tmp_path: Path,
) -> None:
    adapter = _transition_adapter()
    prepared = adapter.prepare_transition(
        observation=_observation(),
        policy_action=np.zeros((15, 8)),
        task_prompt=_observation()["prompt"],
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path,
    )

    assert prepared["wam_request"]["view_order"] == list(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert prepared["wam_request"]["action_conditioning_shape"] == [11, 7]
    assert prepared["wam_request"]["executed_prefix_seconds"] == pytest.approx(8 / 15)
    assert Path(prepared["native_policy_action_path"]).is_file()
    assert len(prepared["native_policy_action_sha256"]) == 64
    assert prepared["openpi_config_name_internal_only"] == "pi05_droid"
    assert all(
        len(rows) == 6
        for rows in prepared["wam_request"]["selected_history_views"].values()
    )


def test_current_reference_runs_same_policy_on_wam_generated_three_view_observations(
    tmp_path: Path,
) -> None:
    seen_external_means: list[int] = []

    class Policy:
        policy_id = "frozen_pi05_droid_fixture"

        def infer(self, observation: dict[str, Any]) -> np.ndarray:
            seen_external_means.append(int(np.mean(observation[DROID_EXTERIOR_VIEW_1])))
            return np.zeros((15, 8), dtype=np.float64)

    class Wam:
        arm_id = "ctrl_world_current_reference_fixture"

        def predict(self, request: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
            assert request["view_order"] == list(CTRL_WORLD_RELEASED_VIEW_ORDER)
            assert not any("policy" in str(key).lower() for key in request)
            query_index = int(request["query_index"])
            sequences: dict[str, list[str]] = {}
            for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
                view_dir = output_dir / f"generated_{view_index}"
                view_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for frame_index in range(5):
                    path = view_dir / f"frame_{frame_index}.png"
                    Image.new(
                        "RGB",
                        (32, 32),
                        color=((query_index + 1) * 10 + view_index,) * 3,
                    ).save(path)
                    paths.append(str(path))
                sequences[view_id] = paths
            return {"generated_view_frame_sequences": sequences}

    class Gate:
        gate_id = "fixture_reliability_pass"

        def assess(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["prepared_transition"]["reliability_actions_10d"].shape == (
                5,
                10,
            )
            return {"abstain": False, "reasons": []}

    class Terminal:
        criterion_id = "fixture_two_query_terminal"

        def assess(
            self, *, observation: dict[str, Any], query_index: int
        ) -> dict[str, Any]:
            assert CTRL_WORLD_VIEW_HISTORY_PATHS in observation
            assert CTRL_WORLD_STATE_HISTORY in observation
            return {"terminal": query_index == 1, "reason": "fixture_terminal"}

    result = run_policy_wam_closed_loop(
        initial_observation=_observation(),
        policy_client=Policy(),
        wam_arm=Wam(),
        transition_adapter=_transition_adapter(),
        reliability_gate=Gate(),
        terminal_criterion=Terminal(),
        config=ClosedLoopConfig(
            task_prompt=_observation()["prompt"],
            executed_prefix_steps=8,
            max_policy_queries=3,
            execution_mode="engineering_smoke",
        ),
        output_dir=tmp_path / "loop",
    )

    assert result["status"] == "completed"
    assert result["policy_call_count"] == 2
    assert result["wam_call_count"] == 2
    assert seen_external_means == [0, 11]
    assert (tmp_path / "loop/transition_0000/native_policy_action.npy").is_file()
    assert (tmp_path / "loop/transition_0001/native_policy_action.npy").is_file()


def test_openpi_config_binds_native_action_rows() -> None:
    with pytest.raises(ValueError, match="action_rows_mismatch"):
        DroidCtrlWorldCurrentReferenceTransitionAdapter(
            action_adapter=_released_adapter(),
            openpi_config_name="pi05_droid",
            action_chunk_rows=10,
        )
