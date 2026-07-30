from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.ctrl_world_droid_action_adapter import (
    FrankaCtrlWorldJointPositionAdapter,
)
from blueprint_pipeline.droid_ctrl_world_joint_position_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_STATE_HISTORY,
    CTRL_WORLD_VIEW_HISTORY_PATHS,
    DroidCtrlWorldJointPositionTransitionAdapter,
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


class _Model:
    jnt_range = np.asarray([[-3.0, 3.0]] * 7, dtype=float)


class _Data:
    def __init__(self, _model: Any) -> None:
        self.qpos = np.zeros(16, dtype=float)
        self.xpos = np.zeros((1, 3), dtype=float)
        self.xmat = np.eye(3, dtype=float).reshape(1, 9)


class _Mujoco:
    @staticmethod
    def MjData(model: Any) -> _Data:
        return _Data(model)

    @staticmethod
    def mj_forward(_model: Any, data: _Data) -> None:
        data.xpos[0] = data.qpos[:3]


def _adapter(rows: int = 15) -> DroidCtrlWorldJointPositionTransitionAdapter:
    runtime = {
        "model": _Model(),
        "mujoco": _Mujoco,
        "ids": {"hand": 0},
    }
    return DroidCtrlWorldJointPositionTransitionAdapter(
        action_adapter=FrankaCtrlWorldJointPositionAdapter(runtime),
        action_chunk_rows=rows,
    )


def _observation() -> dict[str, Any]:
    return {
        DROID_EXTERIOR_VIEW_1: np.zeros((224, 224, 3), dtype=np.uint8),
        DROID_EXTERIOR_VIEW_2: np.ones((224, 224, 3), dtype=np.uint8),
        DROID_WRIST_VIEW: np.full((224, 224, 3), 2, dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float64),
        "observation/gripper_position": np.zeros(1, dtype=np.float64),
        "prompt": "Pick up the spray can and place it inside the marked tray.",
    }


def test_transition_builds_exact_three_view_11x7_request(tmp_path: Path) -> None:
    action = np.zeros((15, 8), dtype=np.float64)
    action[:, 0] = np.linspace(0.0, 1.4, 15)
    prepared = _adapter().prepare_transition(
        observation=_observation(),
        policy_action=action,
        task_prompt=_observation()["prompt"],
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path,
    )

    request = prepared["wam_request"]
    assert request["view_order"] == list(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert request["action_conditioning_shape"] == [11, 7]
    assert request["action_conditioning_7d"].shape == (11, 7)
    assert prepared["reliability_actions_10d"].shape == (5, 10)
    assert prepared["next_joint_position"][0] == pytest.approx(0.7)
    assert Path(prepared["native_policy_action_path"]).is_file()
    assert len(prepared["native_policy_action_sha256"]) == 64
    assert all(len(rows) == 6 for rows in request["selected_history_views"].values())
    for rows in request["selected_history_views"].values():
        with Image.open(rows[0]["path"]) as image:
            assert image.size == (320, 192)


def test_transition_rejects_missing_second_exterior_view(tmp_path: Path) -> None:
    observation = _observation()
    observation.pop(DROID_EXTERIOR_VIEW_2)
    with pytest.raises(ValueError, match="observation_invalid"):
        _adapter().prepare_transition(
            observation=observation,
            policy_action=np.zeros((15, 8)),
            task_prompt=observation["prompt"],
            executed_prefix_steps=8,
            query_index=0,
            output_dir=tmp_path,
        )


def test_joint_position_ctrl_world_requeries_same_policy_on_generated_views(
    tmp_path: Path,
) -> None:
    seen_external_means: list[int] = []

    class Policy:
        policy_id = "pi05_droid_jointpos_polaris_fixture"

        def infer(self, observation: dict[str, Any]) -> np.ndarray:
            seen_external_means.append(int(np.mean(observation[DROID_EXTERIOR_VIEW_1])))
            return np.zeros((15, 8), dtype=np.float64)

    class Wam:
        arm_id = "blueprint_ctrl_world_joint_position_reference_fixture"

        def predict(self, request: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
            assert request["action_conditioning_7d"].shape == (11, 7)
            query_index = int(request["query_index"])
            sequences: dict[str, list[str]] = {}
            for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
                paths: list[str] = []
                for frame_index in range(5):
                    path = output_dir / f"view_{view_index}" / f"frame_{frame_index}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new(
                        "RGB",
                        (320, 192),
                        color=((query_index + 1) * 10 + view_index,) * 3,
                    ).save(path)
                    paths.append(str(path))
                sequences[view_id] = paths
            return {"generated_view_frame_sequences": sequences}

    class Gate:
        gate_id = "fixture_ctrl_world_reliability"

        def assess(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["prepared_transition"]["reliability_actions_10d"].shape == (5, 10)
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
        transition_adapter=_adapter(),
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
