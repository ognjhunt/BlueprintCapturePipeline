from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from blueprint_pipeline.core.security_controls import BoundedHttpResponse
from blueprint_pipeline.native_g1_shared_scene_episode import (
    run_g1_built_scene_policy_episode,
    run_g1_shared_scene_episode,
    team_policy_candidate_id,
)
from blueprint_pipeline.native_g1_team_policy_jsonl_client import NativeG1TeamPolicyJsonlClient
from blueprint_pipeline.native_g1_team_policy_https_client import NativeG1TeamPolicyHttpsClient
from tests.test_native_g1_team_policy_jsonl_client import _close, _process
from tests.test_team_policy_delivery_profile import OWNER, _profile, _setup


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


@pytest.mark.parametrize(
    "candidate",
    [
        "humanoidarena_dp_g1_dex3_sonic",
        "humanoidarena_pi05_g1_dex3_sonic",
    ],
)
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
            environment=_Scene(),
            policy_client=policy,
            sonic_bridge=_Bridge(),
            candidate_id="unknown",
            task_prompt="pick the box",
            max_steps=1,
            output_dir=tmp_path,
            read_task_sample=lambda: {},
        )
    assert policy.queries == 0


def test_team_policy_runs_same_scene_and_retains_profile_bound_media(tmp_path: Path) -> None:
    profile_digest = "sha256:" + "b" * 64
    candidate = team_policy_candidate_id(profile_digest)
    scene, policy, bridge = _Scene(), _Policy(), _Bridge()
    policy.profile_digest = profile_digest
    trace = run_g1_shared_scene_episode(
        environment=scene,
        policy_client=policy,
        sonic_bridge=bridge,
        candidate_id=candidate,
        task_prompt="pick the box",
        max_steps=2,
        output_dir=tmp_path,
        read_task_sample=lambda: {"step_index": scene.step},
        team_policy_profile_digest=profile_digest,
        team_objective_id="task_success",
    )
    assert policy.queries == 1
    assert trace["team_policy_profile_digest"] == profile_digest
    assert trace["team_objective_id"] == "task_success"
    assert trace["scene_step_count"] == 2
    assert trace["visual_evidence"]["status"] == "complete"
    assert all(
        (tmp_path / row["relative_path"]).is_file()
        for row in trace["visual_evidence"]["videos"].values()
    )


def test_team_policy_identity_mismatch_refuses_before_scene_reset(tmp_path: Path) -> None:
    scene, policy = _Scene(), _Policy()
    with pytest.raises(ValueError, match="configuration_invalid"):
        run_g1_shared_scene_episode(
            environment=scene,
            policy_client=policy,
            sonic_bridge=_Bridge(),
            candidate_id="team_policy_" + "a" * 64,
            task_prompt="pick the box",
            max_steps=1,
            output_dir=tmp_path,
            read_task_sample=lambda: {"step_index": scene.step},
            team_policy_profile_digest="sha256:" + "b" * 64,
            team_objective_id="task_success",
        )
    assert scene.seed is None
    assert policy.queries == 0


def test_bound_team_process_receives_real_scene_observation(tmp_path: Path) -> None:
    profile_digest = "sha256:" + "c" * 64
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    process = _process("{'ok': True, 'action_chunk': " + repr([action]) + "}")
    try:
        client = NativeG1TeamPolicyJsonlClient(
            process, timeout_seconds=2, profile_digest=profile_digest
        )
        scene = _Scene()
        trace = run_g1_shared_scene_episode(
            environment=scene,
            policy_client=client,
            sonic_bridge=_Bridge(),
            candidate_id=team_policy_candidate_id(profile_digest),
            task_prompt="pick the box",
            max_steps=1,
            output_dir=tmp_path,
            read_task_sample=lambda: {"step_index": scene.step},
            team_policy_profile_digest=profile_digest,
            team_objective_id="task_success",
        )
        assert client.candidate_policy_queried
        assert trace["policy_query_count"] == 1
        assert trace["steps"][0]["semantic_action"] == action
        assert trace["visual_evidence"]["status"] == "complete"
    finally:
        _close(process)


def test_bound_team_endpoint_receives_real_scene_observation(tmp_path: Path) -> None:
    setup = _setup()
    profile = _profile(
        setup,
        {
            "mode": "authenticated_endpoint",
            "endpoint_url": "https://policy.example.org/v1/action",
            "auth_secret_ref": "secretref:team/policy",
            "timeout_ms": 5000,
        },
    )
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    requests = []

    def fetcher(url, **options):
        request = json.loads(options["data"])
        requests.append(request)
        response = {
            "protocol": request["protocol"],
            "profile_digest": request["profile_digest"],
            "request_id": request["request_id"],
        }
        response.update({"ok": True} if request["kind"] == "reset" else {"action_chunk": [action]})
        return BoundedHttpResponse(
            body=json.dumps(response).encode(),
            status=200,
            content_type="application/json",
            final_url=url,
        )

    robot = setup["robot_presets"][0]
    client = NativeG1TeamPolicyHttpsClient(
        profile=profile,
        expected_owner=OWNER,
        expected_setup_digest=setup["setup_digest"],
        expected_interface={
            "robot_preset_id": robot["robot_preset_id"],
            "embodiment_id": robot["embodiment_id"],
            "observation_schema_id": robot["observation_schema"]["schema_id"],
            "action_schema_id": robot["action_schema"]["schema_id"],
        },
        approved_origin="https://policy.example.org",
        resolved_secret_ref="secretref:team/policy",
        credential="private-token",
        fetcher=fetcher,
    )
    scene = _Scene()
    trace = run_g1_shared_scene_episode(
        environment=scene,
        policy_client=client,
        sonic_bridge=_Bridge(),
        candidate_id=team_policy_candidate_id(profile["profile_digest"]),
        task_prompt="pick the box",
        max_steps=1,
        output_dir=tmp_path,
        read_task_sample=lambda: {"step_index": scene.step},
        team_policy_profile_digest=profile["profile_digest"],
        team_objective_id="task_success",
    )
    assert [request["kind"] for request in requests] == ["reset", "infer"]
    assert requests[1]["profile_digest"] == profile["profile_digest"]
    assert trace["policy_query_count"] == 1
    assert trace["visual_evidence"]["status"] == "complete"


def test_built_g1_scene_uses_same_task_samples_and_scorer(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import adp_task_scoring
    from blueprint_pipeline import native_g1_joint_episode_environment as g1_environment
    from blueprint_pipeline import native_g1_run_preflight
    from blueprint_pipeline import native_task_arena_readback

    scene, policy, bridge = _Scene(), _Policy(), _Bridge()
    scene.plan = {
        **scene.plan,
        "task_kind": "rigid_pick_place",
        "task_spec": {"prompt": "pick the box", "task_kind": "rigid_pick_place"},
    }
    monkeypatch.setattr(
        native_g1_run_preflight,
        "preflight_g1_shared_scene_run",
        lambda **kwargs: {
            "status": "staged_inputs_verified",
            "robot_id": "unitree_g1",
            "candidate_id": "humanoidarena_dp_g1_dex3_sonic",
            "scene_plan_digest": scene.plan["plan_digest"],
            "policy_role": "manipulation",
        },
    )
    monkeypatch.setattr(g1_environment, "NativeG1JointEpisodeEnvironment", lambda **kwargs: scene)
    monkeypatch.setattr(
        native_task_arena_readback,
        "NativeRigidTaskArenaReadback",
        lambda built: type(
            "Readback", (), {"read_task_sample": lambda self: {"object_z_m": float(scene.step)}}
        )(),
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
        policy_client=policy,
        sonic_bridge=bridge,
        candidate_id="humanoidarena_dp_g1_dex3_sonic",
        max_steps=1,
        output_dir=tmp_path,
        preflight_inputs={},
        to_tensor=lambda value: value,
        make_action_tensor=lambda value, **kwargs: value,
    )
    assert [sample["step_index"] for sample in captured["samples"]] == [0, 1]
    assert [sample["object_z_m"] for sample in captured["samples"]] == [0.0, 1.0]
    assert captured["task_spec"] is scene.plan["task_spec"]
    assert result["score"] == {"status": "scored", "outcome": "failure"}
    assert result["ranking_eligible"] is False
    assert result["policy_runtime_identity_verified"] is False
    assert (tmp_path / "native_g1_shared_scene_episode_trace.v1.json").is_file()
    assert (tmp_path / "native_g1_built_scene_policy_episode.v1.json").is_file()


def test_navigation_candidate_uses_same_scene_and_measured_goal_score(
    tmp_path: Path, monkeypatch
) -> None:
    from blueprint_pipeline import native_g1_joint_episode_environment as g1_environment
    from blueprint_pipeline import native_g1_run_preflight
    from blueprint_pipeline.native_g1_navigation_goal import PUBLISHED_TASK_INSTRUCTION

    candidate = "humanoidarena_dp_g1_dex3_sonic_vision_navi"
    scene, bridge = _Scene(), _Bridge()
    scene.plan = {
        **scene.plan,
        "task_kind": "rigid_pick_place",
        "task_spec": {
            "task_kind": "rigid_pick_place",
            "prompt": "pick the box",
            "visible_target_marker": {
                "schema_version": "native_task_target_marker.v1",
                "shape": "flat_green_disc",
                "non_colliding": True,
                "surface_position_world_m": [0.5, 0.0, 0.7],
                "radius_m": 0.06,
            },
            "g1_navigation_goal": {
                "schema_version": "native_g1_navigation_goal.v1",
                "center_world_m": [2.0, 0.0, 0.0],
                "acceptance_radius_m": 0.3,
                "max_root_height_drift_m": 0.2,
                "settle_window_samples": 2,
                "task_instruction": PUBLISHED_TASK_INSTRUCTION,
                "visible_target_marker": {
                    "schema_version": "native_task_target_marker.v1",
                    "shape": "flat_yellow_disc",
                    "non_colliding": True,
                    "surface_position_world_m": [2.0, 0.0, 0.0],
                    "radius_m": 0.4,
                },
            },
        },
    }
    scene.read_state = lambda: {
        "step_index": scene.step,
        "root_position_world_m": [
            0.0 if scene.step == 0 else 1.8 + 0.1 * (scene.step - 1),
            0.0,
            0.85,
        ],
    }

    class NavigationPolicy(_Policy):
        def infer_chunk(self, *, front_rgb, observation_state, task):
            assert task == PUBLISHED_TASK_INSTRUCTION
            self.queries += 1
            return [[0.0] * 40, [0.0] * 40]

    monkeypatch.setattr(
        native_g1_run_preflight,
        "preflight_g1_shared_scene_run",
        lambda **kwargs: {
            "status": "staged_inputs_verified",
            "robot_id": "unitree_g1",
            "candidate_id": candidate,
            "scene_plan_digest": scene.plan["plan_digest"],
            "policy_role": "movement_navigation",
        },
    )
    monkeypatch.setattr(g1_environment, "NativeG1JointEpisodeEnvironment", lambda **kwargs: scene)
    result = run_g1_built_scene_policy_episode(
        built=type("Built", (), {"plan": scene.plan})(),
        policy_client=NavigationPolicy(),
        sonic_bridge=bridge,
        candidate_id=candidate,
        max_steps=2,
        output_dir=tmp_path,
        preflight_inputs={},
        to_tensor=lambda value: value,
        make_action_tensor=lambda value, **kwargs: value,
    )
    assert result["evaluation_task_kind"] == "g1_navigation_goal"
    assert result["score"]["outcome"] == "success"
    assert result["score"]["obstacle_clearance_scored"] is False
    assert result["ranking_eligible"] is False
