from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from blueprint_pipeline.native_g1_shared_scene_episode import (
    run_g1_built_scene_policy_episode,
    run_g1_shared_scene_episode,
)


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

    def read_observation_metadata(self, roles: tuple[str, ...]) -> dict:
        def calibration(role: str) -> dict:
            width, height = (640, 480) if role == "head" else (16, 16)
            return {
                "camera_model": "pinhole",
                "intrinsic_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "world_from_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "resolution": [width, height],
                "near_m": 0.01,
                "far_m": 20.0,
            }
        return {
            "timestamp_ns": self.step * 20_000_000,
            "simulation_time_s": self.step / 50.0,
            "calibrations": {role: calibration(role) for role in roles},
            "source_devices": {role: "cpu" for role in roles},
            "synchronizations": {
                role: {"host_bytes_ready": True, "method": "fixture"} for role in roles
            },
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
        read_task_sample=lambda: {"step_index": scene.step, "object_z_m": float(scene.step)},
    )
    assert scene.seed == policy.seed == 19
    assert trace["scene_plan_digest"] == scene.plan["plan_digest"]
    assert trace["scene_step_count"] == 3
    assert trace["policy_query_count"] == 2
    assert trace["initial_task_sample"] == {"step_index": 0, "object_z_m": 0.0}
    assert trace["claim_ceiling"] == "simulator_only_unscored"
    assert [row["task_sample"]["object_z_m"] for row in trace["steps"]] == [1.0, 2.0, 3.0]
    frames = [row["policy_input_frame"] for row in trace["queries"]]
    frames += [frame for row in trace["steps"] for frame in row["review_frames"].values()]
    frames += list(trace["terminal_observation"]["views"].values())
    assert len(frames) == 10
    assert all((tmp_path / frame["relative_path"]).is_file() for frame in frames)
    assert all(frame["png_sha256"].startswith("sha256:") for frame in frames)
    assert trace["trace_digest"].startswith("sha256:")
    assert trace["visual_evidence"]["status"] == "complete"
    assert set(trace["visual_evidence"]["videos"]) == {"head", "overview"}
    assert all(
        (tmp_path / row["relative_path"]).is_file()
        for row in trace["visual_evidence"]["videos"].values()
    )


def test_unknown_candidate_never_queries_policy(tmp_path: Path) -> None:
    policy = _Policy()
    with pytest.raises(ValueError, match="configuration_invalid"):
        run_g1_shared_scene_episode(
            environment=_Scene(), policy_client=policy, sonic_bridge=_Bridge(),
            candidate_id="unknown", task_prompt="pick the box", max_steps=1,
            output_dir=tmp_path, read_task_sample=lambda: {},
        )
    assert policy.queries == 0


def test_built_g1_scene_uses_same_task_samples_and_scorer(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import adp_task_scoring
    from blueprint_pipeline import native_g1_joint_episode_environment as g1_environment
    from blueprint_pipeline import native_g1_run_preflight
    from blueprint_pipeline import native_task_arena_readback

    scene, policy, bridge = _Scene(), _Policy(), _Bridge()
    scene.plan = {
        **scene.plan, "task_kind": "rigid_pick_place",
        "task_spec": {"prompt": "pick the box", "task_kind": "rigid_pick_place"},
    }
    monkeypatch.setattr(
        native_g1_run_preflight, "preflight_g1_shared_scene_run",
        lambda **kwargs: {
            "status": "staged_inputs_verified",
            "robot_id": "unitree_g1",
            "candidate_id": "humanoidarena_dp_g1_dex3_sonic",
            "scene_plan_digest": scene.plan["plan_digest"],
            "policy_role": "manipulation",
        },
    )
    monkeypatch.setattr(
        g1_environment, "NativeG1JointEpisodeEnvironment", lambda **kwargs: scene
    )
    monkeypatch.setattr(
        native_task_arena_readback, "NativeRigidTaskArenaReadback",
        lambda built: type("Readback", (), {"read_task_sample": lambda self: {"object_z_m": float(scene.step)}})(),
    )
    scene.read_state = lambda: {"step_index": scene.step}
    captured = {}

    def score(*, task_spec, samples):
        captured["task_spec"] = task_spec
        captured["samples"] = samples
        return {"status": "scored", "outcome": "failure"}

    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec", score)
    result = run_g1_built_scene_policy_episode(
        built=type("Built", (), {"plan": scene.plan})(),
        policy_client=policy, sonic_bridge=bridge,
        candidate_id="humanoidarena_dp_g1_dex3_sonic",
        max_steps=1, output_dir=tmp_path, preflight_inputs={},
        to_tensor=lambda value: value, make_action_tensor=lambda value, **kwargs: value,
    )
    assert [sample["step_index"] for sample in captured["samples"]] == [0, 1]
    assert [sample["object_z_m"] for sample in captured["samples"]] == [0.0, 1.0]
    assert captured["task_spec"] is scene.plan["task_spec"]
    assert result["score"] == {"status": "scored", "outcome": "failure"}
    assert result["ranking_eligible"] is False
    assert result["policy_runtime_identity_verified"] is False
    assert (tmp_path / "native_g1_shared_scene_episode_trace.v1.json").is_file()
    assert (tmp_path / "native_g1_built_scene_policy_episode.v1.json").is_file()
