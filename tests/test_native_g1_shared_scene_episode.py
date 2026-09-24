from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from blueprint_pipeline.native_g1_shared_scene_episode import run_g1_shared_scene_episode


class _Scene:
    plan = {
        "robot": {"robot_id": "unitree_g1"},
        "scenario": {"seed": 19},
        "task_spec": {"prompt": "pick the box"},
        "plan_digest": "sha256:" + "a" * 64,
    }

    def __init__(self) -> None:
        self.step = 0
        self.seed = None

    def reset(self, *, seed: int) -> None:
        self.seed = seed
        self.step = 0

    def read_policy_inputs(self) -> dict:
        return {
            "front_rgb": np.full((480, 640, 3), self.step, dtype=np.uint8),
            "observation_state": [0.0] * 64,
            "sensor_freshness": {"step_index": self.step, "sensor_frame_index": self.step + 1},
        }

    def step_controller_targets(self, targets: dict) -> dict:
        assert targets == {"joint": float(self.step + 1)}
        self.step += 1
        return {"step_index": self.step}

    def read_review_inputs(self) -> dict:
        return {
            "head_rgb": np.full((480, 640, 3), self.step, dtype=np.uint8),
            "overview_rgb": np.full((16, 16, 3), self.step, dtype=np.uint8),
            "sensor_freshness": {"head": self.step, "overview": self.step},
        }


class _Policy:
    def __init__(self) -> None:
        self.seed = None
        self.queries = 0

    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def infer_chunk(self, *, front_rgb: np.ndarray, observation_state: list, task: str) -> list:
        assert front_rgb.shape == (480, 640, 3)
        assert len(observation_state) == 64
        assert task == "pick the box"
        self.queries += 1
        return [[0.0] * 40, [0.0] * 40]


class _Bridge:
    def __init__(self) -> None:
        self.calls = 0

    def targets_for_action(self, action: list) -> dict:
        assert len(action) == 40
        self.calls += 1
        return {"joint": float(self.calls)}


@pytest.mark.parametrize("candidate", [
    "humanoidarena_dp_g1_dex3_sonic",
    "humanoidarena_pi05_g1_dex3_sonic",
])
def test_candidates_use_same_scene_episode_and_retain_each_frame(
    candidate: str, tmp_path: Path
) -> None:
    scene, policy, bridge = _Scene(), _Policy(), _Bridge()

    trace = run_g1_shared_scene_episode(
        environment=scene,
        policy_client=policy,
        sonic_bridge=bridge,
        candidate_id=candidate,
        task_prompt="pick the box",
        max_steps=3,
        output_dir=tmp_path,
        read_task_sample=lambda: {"object_z_m": float(scene.step)},
    )
    assert scene.seed == policy.seed == 19
    assert trace["scene_plan_digest"] == scene.plan["plan_digest"]
    assert trace["scene_step_count"] == 3
    assert trace["policy_query_count"] == 2
    assert trace["claim_ceiling"] == "simulator_only_unscored"
    assert [row["task_sample"]["object_z_m"] for row in trace["steps"]] == [1.0, 2.0, 3.0]
    frames = [row["policy_input_frame"] for row in trace["queries"]]
    frames += [frame for row in trace["steps"] for frame in row["review_frames"].values()]
    assert len(frames) == 8
    assert all((tmp_path / frame["relative_path"]).is_file() for frame in frames)
    assert all(frame["png_sha256"].startswith("sha256:") for frame in frames)
    assert trace["trace_digest"].startswith("sha256:")


def test_unknown_candidate_never_queries_policy(tmp_path: Path) -> None:
    policy = _Policy()
    with pytest.raises(ValueError, match="configuration_invalid"):
        run_g1_shared_scene_episode(
            environment=_Scene(), policy_client=policy, sonic_bridge=_Bridge(),
            candidate_id="unknown", task_prompt="pick the box", max_steps=1,
            output_dir=tmp_path, read_task_sample=lambda: {},
        )
    assert policy.queries == 0
