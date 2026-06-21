from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from blueprint_pipeline import mujoco_g1_wam_vla_policy_endpoint_eval as lane


def test_wam_vla_policy_endpoint_discovery_matrix_and_file_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert lane._repo_root().name == "BlueprintCapturePipeline"
    assert lane._safe_id(" Door / Table ") == "door_table"
    assert lane._safe_id("", fallback="fallback") == "fallback"
    assert lane._number(True, 4.0) == 4.0
    assert lane._number("2.5") == 2.5
    assert lane._number("bad", None) is None
    assert lane._mapping({"a": 1}) == {"a": 1}
    assert lane._mapping([]) == {}
    assert lane._redact({"api_key": "secret", "items": [{"token": "abc", "ok": 1}]}) == {
        "api_key": "<redacted>",
        "items": [{"token": "<redacted>", "ok": 1}],
    }
    monkeypatch.setenv("BLUEPRINT_WAM_VLA_TEST_BOOL", "yes")
    assert lane._env_truthy("BLUEPRINT_WAM_VLA_TEST_BOOL")

    rows_path = tmp_path / "rows.jsonl"
    lane._write_jsonl(rows_path, [{"b": 2, "a": 1}])
    assert rows_path.read_text(encoding="utf-8").strip() == '{"a": 1, "b": 2}'
    assert len(lane._utc_timestamp_for_path()) == len("20260102T030405Z")
    quat = lane._yaw_quat(math.pi / 2)
    assert lane._yaw_from_quat(quat) == pytest.approx(math.pi / 2)
    assert lane._episode_frame_steps(
        steps_per_episode=100,
        render_frame_count=0,
        video_frame_stride_steps=1,
    ) == (list(range(100)), "full_episode_stride", 1)
    assert lane.DEFAULT_VIDEO_FRAME_STRIDE_STEPS == 8
    assert lane.DEFAULT_REVIEW_VIDEO_FPS == 60
    assert lane.DEFAULT_CONTROLLER_BACKEND == "auto"
    assert set(lane.CONTROLLER_BACKENDS) == {"auto", "freejoint_proxy", "unitree_rl_gym"}
    assert lane._episode_frame_steps(
        steps_per_episode=100,
        render_frame_count=5,
        video_frame_stride_steps=25,
    ) == ([0, 25, 50, 75, 99], "fixed_sample_count", 25)
    assert lane._video_output_fps(requested_fps=0, timestep=0.002, stride_steps=20) == 25
    assert lane._video_output_fps(requested_fps=0, timestep=0.002, stride_steps=1) == 500
    assert lane._video_output_fps(requested_fps=12, timestep=0.002, stride_steps=20) == 12
    slow_timing = lane._video_timing_contract(
        requested_fps=60,
        encoded_fps=60,
        timestep=0.002,
        stride_steps=1,
        physics_frame_count=3000,
        encoded_frame_count=3000,
    )
    assert slow_timing["video_playback_may_look_slow_motion"] is True
    assert slow_timing["playback_time_scale_vs_sim"] == pytest.approx(8.333333)
    realtime_timing = lane._video_timing_contract(
        requested_fps=0,
        encoded_fps=500,
        timestep=0.002,
        stride_steps=1,
        physics_frame_count=3000,
        encoded_frame_count=3000,
    )
    assert realtime_timing["fps_zero_used_for_sim_time_playback"] is True
    assert realtime_timing["video_playback_may_look_slow_motion"] is False
    assert lane._derive_health_url("http://127.0.0.1:8765/policy/action") == (
        "http://127.0.0.1:8765/health"
    )
    assert lane._probe_endpoint_health(endpoint_row={"endpoint_url": ""}, timeout_seconds=0.01)[
        "blockers"
    ] == ["blocked_missing_policy_endpoint"]

    class FakeHealthResponse:
        status = 200

        def __enter__(self) -> "FakeHealthResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"ok","token":"secret"}'

    monkeypatch.setattr(lane.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeHealthResponse())
    health_probe = lane._probe_endpoint_health(
        endpoint_row={"endpoint_url": "https://policy.example.test/policy/action"},
        timeout_seconds=0.01,
    )
    assert health_probe["status"] == "completed"
    assert health_probe["health_url"] == "https://policy.example.test/health"
    assert health_probe["health_payload_redacted"]["token"] == "<redacted>"
    assert lane.build_policy_model_truth_boundary(generated_at="now")[
        "reference_command_policy_is_not_real_wam_vla"
    ] is True
    candidate_matrix = lane.build_policy_model_candidate_matrix(generated_at="now")
    assert {row["id"] for row in candidate_matrix["candidates"]} >= {
        "oscar_wam",
        "cosmos_wam",
        "openvla_policy",
        "unitree_g1_policy",
        "command_policy",
    }
    navigation = lane.discover_realistic_navigation_policy(generated_at="now")
    assert navigation["pre_execution_discovery_only"] is True
    assert navigation["final_execution_truth_artifact"] == "controller_truth_boundary.json"
    assert navigation["execution_truth_fields"] == "deferred_to_controller_truth_boundary_json"
    assert navigation["realistic_navigation_policy_used"] is None
    assert navigation["freejoint_proxy_used"] is None
    assert navigation["official_unitree_controller_used"] is None
    assert navigation["balanced_walking_controller_proven"] is None
    assert navigation["claim_boundary"]["online_source_discovery_is_not_controller_execution_proof"] is True
    assert {row["name"] for row in navigation["official_online_candidates"]} >= {
        "unitree_rl_gym",
        "unitree_rl_lab",
        "unitree_mujoco",
        "unitree_lerobot",
        "lerobot_unitree_g1",
    }
    unitree_root = tmp_path / "unitree_rl_gym"
    for path in lane._unitree_rl_gym_required_files(unitree_root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_UNITREE_RL_GYM_ROOT", str(unitree_root))
    navigation = lane.discover_realistic_navigation_policy(generated_at="now")
    assert navigation["status"] == "candidate_available_for_endpoint_controller_selection"
    assert navigation["unitree_policy_root_ready_for_sidecar_execution"] is True
    assert navigation["unitree_policy_root_ready_for_same_scene_endpoint_controller"] is True
    assert navigation["same_scene_endpoint_controller_can_be_selected_with"] == "--controller-backend unitree_rl_gym"
    assert navigation["candidate_root_envs"][1]["required_files_present"] is True
    assert "blocked_missing_realistic_g1_navigation_policy" not in navigation["blockers"]
    monkeypatch.delenv("BLUEPRINT_UNITREE_RL_GYM_ROOT")
    monkeypatch.setenv("BLUEPRINT_REALISTIC_G1_POLICY_COMMAND", "'unterminated")
    malformed_navigation = lane.discover_realistic_navigation_policy(generated_at="now")
    assert malformed_navigation["candidate_command_envs"][0]["configured"] is True
    assert malformed_navigation["candidate_command_envs"][0]["available"] is False
    monkeypatch.delenv("BLUEPRINT_REALISTIC_G1_POLICY_COMMAND")

    discovery, runtime, auth, probe = lane.discover_policy_runtime(generated_at="now")
    assert discovery["status"] == "blocked_missing_policy_endpoint"
    assert runtime["fixture_reference_policy"]["available"] is True
    assert auth["file_based_secrets_only"] is True
    assert probe["blockers"] == ["blocked_missing_policy_endpoint"]
    assert lane.selected_endpoint(discovery) is None

    monkeypatch.setenv("WAM_POLICY_ENDPOINT_URL", "https://policy.example.test/infer")
    no_auth_discovery, _, _, _ = lane.discover_policy_runtime(generated_at="now")
    assert "blocked_missing_policy_auth_token_file" in no_auth_discovery["blockers"]
    token_file = tmp_path / "token.txt"
    token_file.write_text("tok", encoding="utf-8")
    monkeypatch.setenv("WAM_POLICY_AUTH_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("BLUEPRINT_WAM_PROVIDER_COMMAND", "python wam.py")
    discovery, runtime, auth, probe = lane.discover_policy_runtime(generated_at="now")
    assert discovery["status"] == "endpoint_ready"
    assert lane.selected_endpoint(discovery)["runtime"] == "wam"
    assert auth["status"] == "auth_ready"
    assert runtime["provider_command_contracts"][0]["value_redacted"] == "<configured>"
    assert probe["status"] == "configured_for_per_observation_calls"

    assert lane._observation_schema()["$id"] == lane.OBSERVATION_SCHEMA_ID
    assert lane._action_schema()["$id"] == lane.ACTION_SCHEMA_ID
    matrix = lane.build_scenario_eval_matrix(
        job_id="job",
        generated_at="now",
        task_filter=["inspect_target"],
        spawn_filter=["doorway"],
        max_tasks=1,
        max_spawns=1,
    )
    assert matrix["status"] == "completed"
    assert matrix["scenario_eval_run_count"] == 1
    assert matrix["runs"][0]["scenario_eval_run_id"] == "job__doorway__inspect_target"
    empty = lane.build_scenario_eval_matrix(
        job_id="job", generated_at="now", task_filter=["missing"], spawn_filter=["missing"]
    )
    assert empty["status"] == "blocked_empty_matrix"

    scene_manifest = lane._write_scene_xml(
        g1_xml=tmp_path / "g1.xml",
        output_xml=tmp_path / "scene" / "eval.xml",
    )
    assert scene_manifest["status"] == "completed"
    assert (tmp_path / "scene" / "eval.xml").is_file()


def test_official_unitree_controller_sidecar_passes_command_vector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import unitree_g1_policy_execution as unitree

    unitree_root = tmp_path / "unitree_rl_gym"
    for path in lane._unitree_rl_gym_required_files(unitree_root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_build(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {
            "status": "completed",
            "policy_id": "unitree_rl_gym_g1_pretrain_motion",
            "metrics": {"command_xyz": kwargs["command_xyz"]},
            "proof_boundary": {"non_default_policy_execution_trace_proven": True},
        }

    monkeypatch.setattr(unitree, "build_unitree_g1_policy_execution", fake_build)

    manifest = lane._run_official_unitree_controller_sidecar(
        job_dir=tmp_path / "job",
        job_id="job-sidecar",
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
        max_steps=5,
        command_xyz=[0.25, 0.05, -0.1],
    )

    assert manifest["status"] == "completed"
    assert manifest["official_unitree_controller_used"] is True
    assert manifest["command_xyz"] == [0.25, 0.05, -0.1]
    assert calls[0]["command_xyz"] == [0.25, 0.05, -0.1]

    command_rows = lane._unitree_command_rows_from_endpoint_actions(
        [
            {
                "episode_id": "episode",
                "scenario_eval_run_id": "run",
                "task_id": "approach_target",
                "spawn_id": "doorway",
                "step": 0,
                "sim_time_s": 0.0,
                "source": "endpoint_policy",
                "normalized_action": {
                    "normalization_status": "accepted",
                    "action_type": "base_velocity",
                    "vx_mps": 0.3,
                    "vy_mps": 0.1,
                    "yaw_rate_rad_s": -0.2,
                },
                "rejected": False,
            }
        ]
    )
    assert lane._representative_unitree_command(command_rows) == [0.3, 0.1, -0.2]
    assert command_rows[0]["raw_endpoint_command_xyz"] == [0.3, 0.1, -0.2]
    assert command_rows[0]["controller_command_xyz"] == [0.3, 0.1, -0.2]
    assert command_rows[0]["controller_command_clamped"] is False
    replay = lane._run_unitree_controller_replay_from_endpoint_actions(
        job_dir=tmp_path / "job",
        job_id="job-sidecar",
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
        max_steps=5,
        command_rows=command_rows,
    )
    assert replay["status"] == "completed"
    assert replay["endpoint_action_trace_bound_to_unitree_command_stream"] is True
    assert replay["representative_endpoint_command_xyz"] == [0.3, 0.1, -0.2]
    assert calls[1]["command_xyz"] == [0.3, 0.1, -0.2]
    bridge = lane.build_unitree_controller_bridge_manifest(
        generated_at="now",
        command_rows=command_rows,
        official_controller_sidecar=manifest,
        endpoint_replay=replay,
    )
    assert bridge["endpoint_action_trace_bound_to_unitree_command_stream"] is True
    assert bridge["same_scene_controller_backend_integrated"] is False
    assert "blocked_same_scene_unitree_controller_bridge_not_integrated" in bridge["blockers"]
    same_scene_bridge = lane.build_unitree_controller_bridge_manifest(
        generated_at="now",
        command_rows=command_rows,
        official_controller_sidecar={},
        endpoint_replay={},
        same_scene_controller={
            "status": "completed",
            "same_scene_controller_backend_integrated": True,
            "official_unitree_controller_used": True,
            "balanced_walking_controller_proven": True,
        },
    )
    assert same_scene_bridge["status"] == "completed"
    assert same_scene_bridge["same_scene_controller_backend_integrated"] is True
    assert same_scene_bridge["realistic_navigation_policy_used_for_endpoint_rollouts"] is True
    assert same_scene_bridge["freejoint_proxy_used_for_endpoint_rollouts"] is False
    assert "blocked_same_scene_unitree_controller_bridge_not_integrated" not in same_scene_bridge["blockers"]


def test_unitree_controller_safe_command_clamps_endpoint_actions() -> None:
    safe = lane._unitree_controller_safe_command_from_values(0.9, -0.5, 2.0)

    assert safe["raw_endpoint_command_xyz"] == [0.9, -0.5, 2.0]
    assert safe["controller_command_xyz"] == [0.35, -0.12, 0.45]
    assert safe["command_xyz"] == [0.35, -0.12, 0.45]
    assert safe["controller_command_clamped"] is True
    assert safe["controller_command_limits"]["max_forward_velocity_mps"] == 0.35

    command_rows = lane._unitree_command_rows_from_endpoint_actions(
        [
            {
                "episode_id": "episode",
                "scenario_eval_run_id": "run",
                "task_id": "approach_target",
                "spawn_id": "doorway",
                "step": 0,
                "sim_time_s": 0.0,
                "source": "endpoint_policy",
                "normalized_action": {
                    "normalization_status": "accepted",
                    "action_type": "base_velocity",
                    "vx_mps": 0.9,
                    "vy_mps": -0.5,
                    "yaw_rate_rad_s": 2.0,
                },
                "rejected": False,
            }
        ]
    )

    assert lane._representative_unitree_command(command_rows) == [0.35, -0.12, 0.45]
    assert command_rows[0]["raw_endpoint_command_xyz"] == [0.9, -0.5, 2.0]
    assert command_rows[0]["controller_command_clamped"] is True


def test_wam_vla_policy_action_normalization_endpoint_and_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = {
        "episode_id": "episode",
        "scenario_eval_run_id": "run",
        "task_id": "approach_target",
        "spawn_id": "doorway",
        "step_index": 0,
        "base_pose": {"position": [0.0, 0.0, 0.79], "yaw_rad": 0.0},
        "route_task_state": {"target_pose": [0.5, 0.0, 0.79], "target_error_m": 0.5},
        "object_state": {"position": [0.2, 0.0, 0.27]},
    }
    rejected_action, rejected = lane.normalize_policy_action(
        raw_payload="bad", observation=observation, source="unit"
    )
    assert rejected_action["normalization_status"] == "rejected"
    assert rejected["reason"] == "policy_action_not_mapping"
    for raw_payload, reason in [
        ({"action": {"action_type": "base_velocity"}}, "base_velocity_missing_numeric_linear_velocity"),
        ({"action": {"action_type": "heading_yaw"}}, "heading_yaw_missing_numeric_target_yaw"),
        ({"action": {"action_type": "waypoint", "waypoint": "bad"}}, "waypoint_missing_xy"),
        ({"action": {"action_type": "waypoint", "waypoint": ["bad", 0]}}, "waypoint_contains_non_numeric_value"),
        ({"action": {"action_type": "unknown"}}, "unsupported_policy_action_type"),
    ]:
        normalized, rejected = lane.normalize_policy_action(
            raw_payload=raw_payload, observation=observation, source="unit"
        )
        assert normalized["action_type"] == "stop"
        assert rejected["reason"] == reason

    accepted_payloads = [
        {"policy_action": {"action_type": "base_velocity", "linear_velocity_mps": 99, "lateral_velocity_mps": -99, "yaw_rate_rad_s": 99}},
        {"decision": {"action": {"action_type": "heading_yaw", "target_yaw_rad": math.pi}}},
        {"action": {"action_type": "waypoint", "waypoint": [9, 0, 0.8]}},
        {"action": {"action_type": "stop"}},
        {"action": {"action_type": "inspect_look", "yaw_rate_rad_s": 9}},
        {"action": {"action_type": "look"}},
        {"action": {"action_type": "manipulation_contact", "waypoint": [0.3, 0.0]}},
        {"action": {"action_type": "manipulation_contact"}},
    ]
    for raw_payload in accepted_payloads:
        normalized, rejected = lane.normalize_policy_action(
            raw_payload=raw_payload, observation=observation, source="unit"
        )
        assert rejected is None
        assert normalized["normalization_status"] == "accepted"

    blocked_observation = {**observation, "task_id": "inspect_target", "spawn_id": "blocked_or_occluded"}
    assert lane._fixture_policy_action(observation=blocked_observation)["fixture_intent"]
    for task_id in [
        "inspect_target",
        "contact_or_push_light_object",
        "stop_at_goal_and_report",
        "route_around_obstruction",
        "approach_target",
    ]:
        action = lane._fixture_policy_action(observation={**observation, "task_id": task_id})
        assert "action" in action
    assert lane._fixture_policy_action(
        observation={
            **observation,
            "task_id": "stop_at_goal_and_report",
            "route_task_state": {"target_pose": [0.1, 0.0, 0.79], "target_error_m": 0.1},
        }
    )["action"]["action_type"] == "stop"

    assert lane._read_token(None) is None
    assert lane._read_token(str(tmp_path / "missing")) is None
    token_file = tmp_path / "token.txt"
    token_file.write_text("secret-token\n", encoding="utf-8")
    assert lane._read_token(str(token_file)) == "secret-token"
    assert lane._call_endpoint_action(
        endpoint_row=None, observation=observation, timeout_seconds=0.01
    )[1]["status"] == "blocked_missing_policy_endpoint"
    assert lane._call_endpoint_action(
        endpoint_row={"endpoint_url": "https://x", "auth_token_file_path": str(tmp_path / "missing"), "runtime": "wam"},
        observation=observation,
        timeout_seconds=0.01,
    )[1]["status"] == "blocked_missing_policy_endpoint_or_auth"

    class FakeResponse:
        status = 202

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"action":{"action_type":"stop"},"meta":1}'

    monkeypatch.setattr(lane.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    payload, meta = lane._call_endpoint_action(
        endpoint_row={"endpoint_url": "https://x", "auth_token_file_path": str(token_file), "runtime": "wam"},
        observation=observation,
        timeout_seconds=0.01,
    )
    assert payload["action"]["action_type"] == "stop"
    assert meta["http_status"] == 202
    monkeypatch.setattr(
        lane.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(lane.urllib.error.URLError("down")),
    )
    assert lane._call_endpoint_action(
        endpoint_row={"endpoint_url": "https://x", "auth_token_file_path": str(token_file), "runtime": "wam"},
        observation=observation,
        timeout_seconds=0.01,
    )[1]["status"] == "failed"

    assert lane._counts_by_key(
        [
            {"task_id": "a", "success": True},
            {"task_id": "a", "success": False},
            {"task_id": "b", "status": "blocked"},
        ],
        "task_id",
    ) == [
        {"id": "a", "attempted": 2, "passed": 1, "failed": 1, "blocked": 0},
        {"id": "b", "attempted": 1, "passed": 0, "failed": 0, "blocked": 1},
    ]
    run = {"task_id": "blocked", "spawn_id": "blocked_or_occluded"}
    status, success, labels, metrics = lane._score_attempt(
        run={**run, "task_id": "inspect_target"},
        final_error_m=9,
        final_speed_mps=9,
        fall_count=1,
        unsafe_collision_count=1,
        object_contact_count=0,
        object_displacement_m=0,
        rejected_action_count=1,
        action_types=[],
    )
    assert status == "blocked"
    assert not success
    assert "blocked_rejected_policy_action" in labels
    assert metrics["blocked_reason"] == "fixture_malformed_action_preserved"
    _, _, rejected_labels, _ = lane._score_attempt(
        run={"task_id": "approach_target", "spawn_id": "doorway"},
        final_error_m=0.1,
        final_speed_mps=0.1,
        fall_count=0,
        unsafe_collision_count=0,
        object_contact_count=0,
        object_displacement_m=0,
        rejected_action_count=1,
        action_types=["waypoint"],
    )
    assert "failure_policy_action_rejected" in rejected_labels
    cases = [
        ("inspect_target", ["inspect_look"], 9, 0, 0, True),
        ("approach_target", ["waypoint"], 0.1, 0, 0, True),
        ("route_around_obstruction", ["waypoint"], 0.44, 0, 0, True),
        ("contact_or_push_light_object", ["manipulation_contact"], 9, 1, 0.02, True),
        ("stop_at_goal_and_report", ["stop"], 0.2, 0, 0, True),
        ("stop_at_goal_and_report", ["waypoint"], 0.2, 0, 0, False),
        ("unknown", ["waypoint"], 0.1, 0, 0, True),
    ]
    for task_id, actions, final_error, object_contacts, object_motion, expected_success in cases:
        _, success, labels, _ = lane._score_attempt(
            run={"task_id": task_id, "spawn_id": "doorway"},
            final_error_m=final_error,
            final_speed_mps=0.01 if "stop" in actions else 0.5,
            fall_count=0,
            unsafe_collision_count=0,
            object_contact_count=object_contacts,
            object_displacement_m=object_motion,
            rejected_action_count=0,
            action_types=actions,
        )
        assert success is expected_success
        assert isinstance(labels, list)


def test_wam_vla_contact_observation_camera_and_media_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = SimpleNamespace(
        nq=14,
        nv=12,
        nu=2,
        actuator_trnid=[[0], [-1]],
        jnt_qposadr=[0, 7],
        jnt_dofadr=[0, 6],
        geom_bodyid=[0, 1],
    )
    data = SimpleNamespace(
        qpos=[0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0, 0.36, -0.65, 0.27, 1.0, 0.0, 0.0, 0.0],
        qvel=[0.0] * 12,
        ctrl=[0.0, 0.0],
        time=0.25,
        ncon=1,
        contact=[SimpleNamespace(geom1=0, geom2=1, dist=-0.01, pos=[1.0, 2.0, 3.0])],
    )
    lane._set_joint_position_holds(model, data)
    assert data.ctrl[0] == data.qpos[0]

    class FakeCamera:
        def __init__(self) -> None:
            self.type = None
            self.lookat = [0.0, 0.0, 0.0]
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0

    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_GEOM=1, mjOBJ_BODY=2),
        mjtCamera=SimpleNamespace(mjCAMERA_FREE=3),
        MjvCamera=FakeCamera,
        mj_id2name=lambda _model, obj, idx: {
            (1, 0): "blueprint_reference_floor",
            (1, 1): "blueprint_light_object_geom",
            (2, 0): "left_ankle_roll_link",
            (2, 1): "blueprint_light_object",
        }.get((obj, idx)),
        mj_contactForce=lambda _model, _data, _index, force: force.__setitem__(0, 1.25),
    )
    contacts = lane._contact_records(model, data, fake_mujoco)
    assert contacts[0]["floor_contact"] is True
    assert contacts[0]["object_contact"] is True
    assert contacts[0]["left_foot_contact"] is True
    data.ncon = 3
    data.contact = [
        SimpleNamespace(geom1=0, geom2=1, dist=-0.01, pos=[1.0, 2.0, 3.0]),
        SimpleNamespace(geom1=0, geom2=1, dist=-0.02, pos=[1.0, 2.0, 3.0]),
        SimpleNamespace(geom1=0, geom2=1, dist=-0.03, pos=[1.0, 2.0, 3.0]),
    ]
    capped_state = lane._contact_state(
        model,
        data,
        fake_mujoco,
        contact_metadata=lane._build_contact_metadata(model, fake_mujoco),
        include_force=False,
        record_limit=1,
    )
    assert capped_state["contact_count"] == 3
    assert capped_state["object_contact_count"] == 3
    assert capped_state["record_count"] == 1
    assert capped_state["records_truncated"] is True
    assert capped_state["dropped_record_count"] == 2
    data.ncon = 1
    data.contact = [SimpleNamespace(geom1=0, geom2=1, dist=-0.01, pos=[1.0, 2.0, 3.0])]
    failing_mujoco = SimpleNamespace(
        **{key: getattr(fake_mujoco, key) for key in ("mjtObj", "MjvCamera", "mjtCamera", "mj_id2name")},
        mj_contactForce=lambda *_args: (_ for _ in ()).throw(RuntimeError("no force")),
    )
    assert lane._contact_records(model, data, failing_mujoco)[0]["contact_force_6d"] == [0.0] * 6

    assert lane._object_pose(data, None) == {"available": False}
    assert lane._object_pose(data, 7)["object_id"] == "blueprint_light_object"
    packet = lane._build_observation_packet(
        model=model,
        data=data,
        root_qpos=0,
        root_dof=0,
        object_qpos=7,
        run={
            "episode_id": "episode",
            "scenario_id": "scenario",
            "scenario_eval_run_id": "run",
            "task_id": "task",
            "spawn_id": "spawn",
            "target_pose": [0.5, 0.0, 0.79],
            "route_waypoints": [[0.5, 0.0]],
            "task_prompt": "Go",
        },
        step=1,
        contacts=contacts,
        contact_summary={"contact_count": 7, "object_contact_count": 3},
        mujoco_version="fake",
    )
    assert packet["contact_state"]["contact_count"] == 7
    assert packet["contact_state"]["object_contact_count"] == 3
    assert packet["route_task_state"]["target_error_m"] == 0.5
    assert lane._camera_for(fake_mujoco, "overhead", [0, 0, 1], 0).elevation == -89.0
    assert lane._camera_for(fake_mujoco, "robot_follow", [0, 0, 1], 0).azimuth == 180.0
    assert lane._camera_for(fake_mujoco, "third_person", [0, 0, 1], 0).distance == 3.2
    assert lane._episode_frame_steps(
        steps_per_episode=5, render_frame_count=0, video_frame_stride_steps=2
    ) == ([0, 2, 4], "full_episode_stride", 2)
    assert lane._episode_frame_steps(
        steps_per_episode=6, render_frame_count=0, video_frame_stride_steps=4
    ) == ([0, 4, 5], "full_episode_stride", 4)
    assert lane._video_output_fps(requested_fps=12, timestep=0.05, stride_steps=3) == 12
    assert lane._video_output_fps(requested_fps=0, timestep=0.05, stride_steps=4) == 5

    monkeypatch.setattr(lane.shutil, "which", lambda _name: None)
    assert lane._write_video_from_frames(frames_dir=tmp_path, output_path=tmp_path / "out.mp4", fps=12)["status"] == "blocked"
    assert lane._ffprobe_video(tmp_path / "missing.mp4")["status"] == "not_checked"

    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if "ffprobe" in command[0]:
            return SimpleNamespace(returncode=0, stdout='{"streams":[{"nb_frames":"3","duration":"1.5","width":640,"height":360}]}', stderr="")
        Path(command[-1]).write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lane.shutil, "which", fake_which)
    monkeypatch.setattr(lane.subprocess, "run", fake_run)
    assert lane._ffprobe_video(tmp_path / "still-missing.mp4")["reason"] == "missing_video"
    video = lane._write_video_from_frames(frames_dir=tmp_path, output_path=tmp_path / "out.mp4", fps=12)
    assert video["status"] == "complete"
    assert lane._ffprobe_video(tmp_path / "out.mp4")["frame_count"] == 3
    monkeypatch.setattr(
        lane.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad"),
    )
    assert lane._ffprobe_video(tmp_path / "out.mp4")["status"] == "blocked"


def test_wam_vla_lane_runs_with_fake_mujoco_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeVec(list[float]):
        def __setitem__(self, key: object, value: object) -> None:
            if isinstance(key, slice) and isinstance(value, (int, float)):
                indices = range(*key.indices(len(self)))
                for index in indices:
                    super().__setitem__(index, float(value))
                return
            super().__setitem__(key, value)  # type: ignore[index]

    state = {"contact": False, "fall": False, "jump": False}

    class FakeModel:
        nq = 14
        nv = 12
        nu = 1
        opt = SimpleNamespace(timestep=0.05)
        actuator_trnid = [[0]]
        jnt_qposadr = [0, 7]
        jnt_dofadr = [0, 6]
        geom_bodyid = [0, 1]

        def __init__(self) -> None:
            self.qpos0 = FakeVec([0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0, 0.36, -0.65, 0.27, 1.0, 0.0, 0.0, 0.0])
            self.key_qpos = [self.qpos0.copy()]

        @classmethod
        def from_xml_path(cls, _path: str) -> "FakeModel":
            return cls()

    class FakeData:
        def __init__(self, model: FakeModel) -> None:
            self.qpos = FakeVec(model.qpos0.copy())
            self.qvel = FakeVec([0.0] * model.nv)
            self.ctrl = FakeVec([0.0] * model.nu)
            self.time = 0.0
            self.ncon = 0
            self.contact: list[object] = []

    def fake_mj_name2id(_model: FakeModel, obj: int, name: str) -> int:
        if name == "floating_base_joint":
            return 0
        if name == "blueprint_light_object_freejoint":
            return 1
        if name == "stand":
            return 0
        return -1

    def fake_step(model: FakeModel, data: FakeData) -> None:
        if state["jump"]:
            data.qpos[0] += 2.0
        else:
            data.qpos[0] += data.qvel[0] * model.opt.timestep
        data.qpos[1] += data.qvel[1] * model.opt.timestep
        if state["fall"]:
            data.qpos[2] = 0.1
        if state["contact"]:
            data.ncon = 1
            data.contact = [SimpleNamespace(geom1=0, geom2=1, dist=-0.01, pos=[0.0, 0.0, 0.0])]
        else:
            data.ncon = 0
            data.contact = []
        data.time += model.opt.timestep

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.__version__ = "fake-1.0"
    fake_mujoco.__file__ = "fake_mujoco.py"
    fake_mujoco.MjModel = FakeModel
    fake_mujoco.MjData = FakeData
    fake_mujoco.mjtObj = SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_KEY=2, mjOBJ_GEOM=3, mjOBJ_BODY=4)
    fake_mujoco.mjtCamera = SimpleNamespace(mjCAMERA_FREE=5)
    fake_mujoco.MjvCamera = lambda: SimpleNamespace(
        type=None,
        lookat=[0.0, 0.0, 0.0],
        distance=0.0,
        azimuth=0.0,
        elevation=0.0,
    )
    fake_mujoco.mj_name2id = fake_mj_name2id
    fake_mujoco.mj_forward = lambda _model, _data: None
    fake_mujoco.mj_step = fake_step
    fake_mujoco.mj_id2name = lambda _model, obj, idx: {
        (3, 0): "blueprint_reference_floor",
        (3, 1): "blueprint_light_object_geom",
        (4, 0): "floor_body",
        (4, 1): "blueprint_light_object",
    }.get((obj, idx))
    fake_mujoco.mj_contactForce = lambda _model, _data, _index, force: force.__setitem__(0, 0.5)
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    g1_root = tmp_path / "g1"
    g1_root.mkdir()
    (g1_root / "g1.xml").write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setattr(lane, "_resolve_g1_model_root", lambda **_kwargs: g1_root)
    def fake_write_g1_xml(_src: Path, dst: Path) -> None:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("<mujoco/>", encoding="utf-8")

    monkeypatch.setattr(lane, "_write_g1_xml_with_absolute_meshes", fake_write_g1_xml)
    monkeypatch.setattr(
        lane,
        "_asset_source_manifest",
        lambda _root: {"asset_source": "fake", "asset_source_sha256": "abc"},
    )

    summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "job",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=2,
        policy_interval_steps=1,
        render=False,
        generated_at="now",
    )
    assert summary["status"] == "completed"
    assert summary["attempted_episode_count"] == 1
    assert summary["fixture_policy_used"] is True
    assert (tmp_path / "job" / "normalized_attempt_trace.json").is_file()
    assert (tmp_path / "job" / "policy_model_candidate_matrix.json").is_file()
    assert (tmp_path / "job" / "policy_model_truth_boundary.json").is_file()
    assert (tmp_path / "job" / "policy_endpoint_runtime_manifest.json").is_file()
    assert (tmp_path / "job" / "policy_endpoint_invocation_trace.jsonl").is_file()
    assert (tmp_path / "job" / "realistic_navigation_policy_discovery.json").is_file()
    assert (tmp_path / "job" / "unitree_endpoint_action_command_stream.json").is_file()
    assert (tmp_path / "job" / "unitree_controller_bridge_manifest.json").is_file()

    generated_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_root=tmp_path / "generated-jobs",
        max_tasks=0,
        max_spawns=0,
        render=False,
        generated_at="now",
    )
    assert generated_summary["status"] == "blocked"

    original_name_lookup = fake_mujoco.mj_name2id
    fake_mujoco.mj_name2id = lambda *_args: -1
    with pytest.raises(RuntimeError, match="floating_base_joint"):
        lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
            job_dir=tmp_path / "missing-root-joint",
            max_tasks=0,
            max_spawns=0,
            render=False,
            generated_at="now",
        )
    fake_mujoco.mj_name2id = original_name_lookup

    token_file = tmp_path / "endpoint-token.txt"
    token_file.write_text("token", encoding="utf-8")
    monkeypatch.setenv("WAM_POLICY_ENDPOINT_URL", "https://policy.example.test/infer")
    monkeypatch.setenv("WAM_POLICY_AUTH_TOKEN_FILE", str(token_file))
    original_endpoint_call = lane._call_endpoint_action
    monkeypatch.setattr(
        lane,
        "_call_endpoint_action",
        lambda **_kwargs: (
            {"action": {"action_type": "stop"}, "policy_id": "endpoint"},
            {"status": "completed", "endpoint_invoked": True},
        ),
    )
    endpoint_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "endpoint-job",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=1,
        policy_interval_steps=1,
        render=False,
        generated_at="now",
    )
    assert endpoint_summary["endpoint_policy_used"] is True
    assert endpoint_summary["fixture_policy_used"] is False
    assert endpoint_summary["endpoint_invocation_count"] == 1
    endpoint_attempts = json.loads(
        (tmp_path / "endpoint-job" / "normalized_attempt_trace.json").read_text(
            encoding="utf-8"
        )
    )["attempts"]
    assert endpoint_attempts[0]["claim_boundary"]["endpoint_policy_plumbing_proven"] is True
    assert endpoint_attempts[0]["claim_boundary"]["real_wam_vla_policy_proven"] is False
    endpoint_policy_summary = json.loads(
        (tmp_path / "endpoint-job" / "policy_evaluation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert endpoint_policy_summary["wam_vla_runtime_proven"] is False
    monkeypatch.setattr(
        lane,
        "_call_endpoint_action",
        lambda **_kwargs: (
            None,
            {"status": "failed", "endpoint_invoked": True, "blockers": ["policy_endpoint_call_failed"]},
        ),
    )
    endpoint_failure_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "endpoint-failure-job",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=1,
        policy_interval_steps=1,
        render=False,
        generated_at="now",
    )
    assert endpoint_failure_summary["fixture_policy_used"] is False
    assert endpoint_failure_summary["endpoint_invocation_count"] == 1
    monkeypatch.setattr(lane, "_call_endpoint_action", original_endpoint_call)
    monkeypatch.delenv("WAM_POLICY_ENDPOINT_URL")
    monkeypatch.delenv("WAM_POLICY_AUTH_TOKEN_FILE")

    original_step = fake_mujoco.mj_step
    fake_mujoco.mj_step = lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
            job_dir=tmp_path / "interrupted-job",
            max_tasks=1,
            max_spawns=1,
            steps_per_episode=1,
            policy_interval_steps=1,
            render=False,
            generated_at="now",
        )
    interruption = json.loads(
        (tmp_path / "interrupted-job" / "run_interruption_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert interruption["status"] == "interrupted"
    controller_truth = json.loads(
        (tmp_path / "interrupted-job" / "controller_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert controller_truth["status"] == "interrupted_before_final_execution_truth"
    assert controller_truth["realistic_navigation_policy_used"] is None
    fake_mujoco.mj_step = original_step

    rejected_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "rejected-job",
        task_filter=["inspect_target"],
        spawn_filter=["blocked_or_occluded"],
        steps_per_episode=1,
        policy_interval_steps=1,
        render=False,
        generated_at="now",
    )
    assert rejected_summary["blocked_episode_count"] == 1

    state["fall"] = True
    fall_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "fall-job",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=3,
        policy_interval_steps=1,
        render=False,
        generated_at="now",
    )
    assert fall_summary["failed_episode_count"] == 1
    state["fall"] = False

    state["contact"] = True
    state["jump"] = True
    contact_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "contact-jump-job",
        task_filter=["contact_or_push_light_object"],
        max_spawns=1,
        steps_per_episode=2,
        policy_interval_steps=1,
        render=False,
        generated_at="now",
    )
    assert contact_summary["collision_dynamics_validated"] is True
    continuity = json.loads((tmp_path / "contact-jump-job" / "root_motion_continuity_report.json").read_text(encoding="utf-8"))
    assert continuity["discontinuity_count"] >= 1
    manipulation_report = json.loads(
        (tmp_path / "contact-jump-job" / "manipulation_endpoint_task_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert manipulation_report["hand_end_effector_policy_used"] is False
    assert manipulation_report["manipulation_endpoint_path_used"] is False
    assert "blocked_dexterous_hand_policy_not_integrated" in manipulation_report["blockers"]
    assert "blocked_real_vla_model_not_configured" in manipulation_report["blockers"]
    state["contact"] = False
    state["jump"] = False

    class FakeRenderer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False

        def update_scene(self, *_args: object, **_kwargs: object) -> None:
            return None

        def render(self) -> list[list[list[int]]]:
            return [[[0, 0, 0]]]

        def close(self) -> None:
            self.closed = True

    class FakeImage:
        @staticmethod
        def fromarray(_frame: object) -> "FakeImage":
            return FakeImage()

        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"png")

    fake_pil = ModuleType("PIL")
    fake_pil.Image = FakeImage
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    fake_mujoco.Renderer = FakeRenderer
    monkeypatch.setattr(
        lane,
        "_write_video_from_frames",
        lambda **kwargs: {"path": str(kwargs["output_path"]), "status": "complete", "size_bytes": 3},
    )
    monkeypatch.setattr(
        lane,
        "_ffprobe_video",
        lambda path: {"path": str(path), "status": "complete", "duration_s": 1.0, "frame_count": 1},
    )
    render_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "render-job",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=2,
        policy_interval_steps=1,
        render=True,
        render_frame_count=2,
        generated_at="now",
    )
    assert render_summary["artifact_paths"]["video_generation_status"].endswith("video_generation_status.json")
    video_status = json.loads(
        (tmp_path / "render-job" / "video_generation_status.json").read_text(encoding="utf-8")
    )
    assert "every_sim_step_captured_for_selected_review_videos" in video_status["render_contract"]
    assert "playback_timing" in video_status["videos"][0]
    assert video_status["videos"][0]["video_fps"] == lane.DEFAULT_REVIEW_VIDEO_FPS
    assert video_status["videos"][0]["playback_timing"]["fps_zero_used_for_sim_time_playback"] is False
    assert (
        json.loads(
            (tmp_path / "render-job" / "review_video_selection_manifest.json").read_text(
                encoding="utf-8"
            )
        )["selected_review_video_count"]
        >= 1
    )

    class SparseFakeImage:
        @staticmethod
        def fromarray(_frame: object) -> "SparseFakeImage":
            return SparseFakeImage()

        def save(self, path: Path) -> None:
            if Path(path).name == "frame_0000.png":
                Path(path).write_bytes(b"png")

    fake_pil.Image = SparseFakeImage
    terminal_hold_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "render-terminal-hold-job",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=2,
        policy_interval_steps=1,
        render=True,
        render_frame_count=2,
        fps=0,
        generated_at="now",
    )
    video_status = json.loads(
        (tmp_path / "render-terminal-hold-job" / "video_generation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert terminal_hold_summary["status"] == "completed"
    assert video_status["videos"][0]["playback_timing"]["fps_zero_used_for_sim_time_playback"] is True
    assert video_status["videos"][0]["terminal_frame_hold_count"] == 0
    assert video_status["videos"][0]["missing_terminal_frame_count"] == 1
    assert video_status["videos"][0]["review_video_stops_at_terminal_failure"] is True
    assert video_status["videos"][0]["full_episode_video"] is False
    held_frame_path = (
        tmp_path
        / "render-terminal-hold-job"
        / "mujoco_frames"
        / video_status["videos"][0]["episode_id"]
        / video_status["videos"][0]["camera"]
        / "frame_0001.png"
    )
    assert not held_frame_path.is_file()

    terminal_hold_enabled_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "render-terminal-hold-enabled-job",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=2,
        policy_interval_steps=1,
        render=True,
        render_frame_count=2,
        extend_terminal_frame_for_review=True,
        generated_at="now",
    )
    video_status = json.loads(
        (
            tmp_path / "render-terminal-hold-enabled-job" / "video_generation_status.json"
        ).read_text(encoding="utf-8")
    )
    assert terminal_hold_enabled_summary["status"] == "completed"
    assert video_status["videos"][0]["terminal_frame_hold_count"] == 1
    assert video_status["videos"][0]["terminal_frame_extended_for_review"] is True
    assert (
        tmp_path
        / "render-terminal-hold-enabled-job"
        / "mujoco_frames"
        / video_status["videos"][0]["episode_id"]
        / video_status["videos"][0]["camera"]
        / "frame_0001.png"
    ).is_file()

    fake_pil.Image = FakeImage
    one_frame_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "render-one-frame-job",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=1,
        policy_interval_steps=1,
        render=True,
        render_frame_count=1,
        generated_at="now",
    )
    assert one_frame_summary["status"] == "completed"

    cli_calls: list[dict[str, object]] = []

    def fake_cli_run(**kwargs: object) -> dict[str, object]:
        cli_calls.append(kwargs)
        return {"status": "completed", "job_dir": "job", "attempted_episode_count": 1}

    monkeypatch.setattr(lane, "run_mujoco_g1_wam_vla_policy_endpoint_eval", fake_cli_run)
    assert lane.main(["--job-dir", str(tmp_path / "cli"), "--skip-render", "--max-tasks", "1"]) == 0
    assert '"status": "completed"' in capsys.readouterr().out
    assert cli_calls[-1]["controller_backend"] == "auto"
    assert cli_calls[-1]["video_frame_stride_steps"] == lane.DEFAULT_VIDEO_FRAME_STRIDE_STEPS
    assert cli_calls[-1]["fps"] == lane.DEFAULT_REVIEW_VIDEO_FPS
    assert (
        lane.main(
            [
                "--job-dir",
                str(tmp_path / "cli-every-step"),
                "--skip-render",
                "--max-tasks",
                "1",
                "--capture-every-sim-step-review-video",
            ]
        )
        == 0
    )
    assert cli_calls[-1]["video_frame_stride_steps"] == 1
    assert cli_calls[-1]["fps"] == 0
    assert (
        lane.main(
            [
                "--job-dir",
                str(tmp_path / "cli-every-step-fixed-fps"),
                "--skip-render",
                "--max-tasks",
                "1",
                "--capture-every-sim-step-review-video",
                "--fps",
                "60",
            ]
        )
        == 0
    )
    assert cli_calls[-1]["video_frame_stride_steps"] == 1
    assert cli_calls[-1]["fps"] == 60
    monkeypatch.setattr(
        lane,
        "run_mujoco_g1_wam_vla_policy_endpoint_eval",
        lambda **_kwargs: {"status": "blocked", "job_dir": "job"},
    )
    assert lane.main(["--job-dir", str(tmp_path / "cli-blocked")]) == 1
