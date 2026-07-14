from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from blueprint_pipeline import mujoco_g1_wam_vla_policy_endpoint_eval as lane


pytestmark = [pytest.mark.slow, pytest.mark.integration]


_UNITREE_POLICY_ENV_VARS = (
    "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
    "BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",
    "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
    "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
    "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_HOST",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_PORT",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
    "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
    "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
)

_WAM_RUNTIME_ENV_VARS = (
    "BLUEPRINT_OSCAR_WAM_COMMAND",
    "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
    "BLUEPRINT_COSMOS_WAM_COMMAND",
    "BLUEPRINT_COSMOS_WAM_PROVIDER_COMMAND",
    "BLUEPRINT_COSMOS_WAM_CHECKPOINT",
    "BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER",
    "BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_KIND",
    "BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_COMMAND",
    "BLUEPRINT_ALLOW_WAM_PERCEPTION_HARNESS_EXTERNAL_BACKEND",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _configure_fake_live_wam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, name: str = "fake_live_wam"
) -> Path:
    command = tmp_path / f"{name}.py"
    command.write_text(
        """
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_WAM_ROLLOUT_INPUT"]).read_text(encoding="utf-8"))
target = Path(request["requested_output"]["next_observation_frame_path"])
source = Path(request["source_policy_observation_frame_path"])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(source.read_bytes())
payload = {
    "schema_version": "fake_live_wam_generation.v1",
    "status": "completed",
    "wam_evaluator_backend": "oscar_wam",
    "action_conditioned_generation_ran": True,
    "generated_next_observation_frame_path": str(target),
}
Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / f"{name}_checkpoint"
    checkpoint.mkdir()
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_COMMAND", f"{sys.executable} {command}")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv(lane.WAM_GENERATION_COMMAND_GATE_ENV, "true")
    return command


def test_wam_perception_harness_backend_config_uses_env_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _WAM_RUNTIME_ENV_VARS)
    assert lane._wam_perception_harness_backend_config() == {
        "backend_kind": "fixture",
        "backend_command": None,
        "allow_external_backend": None,
        "env_gate": "BLUEPRINT_ALLOW_WAM_PERCEPTION_HARNESS_EXTERNAL_BACKEND",
        "command_env": "BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_COMMAND",
        "configured_for_external_backend": False,
    }

    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_KIND", "sam3")
    monkeypatch.setenv(
        "BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_COMMAND",
        "python run_backend.py --token $HF_TOKEN",
    )

    config = lane._wam_perception_harness_backend_config()
    assert config["backend_kind"] == "sam3"
    assert config["backend_command"] == "python run_backend.py --token $HF_TOKEN"
    assert config["configured_for_external_backend"] is True


def test_wam_vla_policy_endpoint_discovery_matrix_and_file_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch, _UNITREE_POLICY_ENV_VARS)
    _clear_env(monkeypatch, _WAM_RUNTIME_ENV_VARS)
    repo_root = lane._repo_root()
    assert (repo_root / "pyproject.toml").is_file()
    assert (repo_root / "src" / "blueprint_pipeline").is_dir()
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

    class FakeCamera:
        def __init__(self) -> None:
            self.type = None
            self.lookat = [0.0, 0.0, 0.0]
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0

    fake_mujoco = SimpleNamespace(
        MjvCamera=FakeCamera,
        mjtCamera=SimpleNamespace(mjCAMERA_FREE="free"),
    )
    head_pov = lane._camera_for(fake_mujoco, "head_pov", [1.0, 2.0, 0.79], 0.0)
    assert head_pov.type == "free"
    assert head_pov.distance == pytest.approx(1.15)
    assert head_pov.azimuth == pytest.approx(180.0)
    assert head_pov.lookat[0] > 1.0
    assert head_pov.lookat[2] > 1.9
    assert lane._episode_frame_steps(
        steps_per_episode=100,
        render_frame_count=0,
        video_frame_stride_steps=1,
    ) == (list(range(100)), "full_episode_stride", 1)
    assert lane.DEFAULT_VIDEO_FRAME_STRIDE_STEPS == 8
    assert lane.DEFAULT_REVIEW_VIDEO_FPS == 60
    assert lane.DEFAULT_WAM_LOOP_STEP_COUNT == 12
    default_sampling = lane._review_video_sampling_contract(
        fps=60,
        timestep=0.002,
        video_frame_stride_steps=8,
        render_frame_count=0,
        extend_terminal_frame_for_review=False,
    )
    assert default_sampling["sampling_mode"] == "nominal_realtime_stride_review"
    assert default_sampling["nominal_realtime_review_mp4"] is True
    assert default_sampling["captures_every_mujoco_step"] is False
    assert default_sampling["sample_every_n_sim_steps"] == 8
    assert "slow-motion" in default_sampling["why_not_every_frame_by_default"]
    every_step_fixed_fps = lane._review_video_sampling_contract(
        fps=60,
        timestep=0.002,
        video_frame_stride_steps=1,
        render_frame_count=0,
        extend_terminal_frame_for_review=False,
    )
    assert (
        every_step_fixed_fps["sampling_mode"]
        == "every_sim_step_fixed_fps_debug_slow_motion"
    )
    assert every_step_fixed_fps["every_frame_at_fixed_60fps_is_debug_slow_motion"] is True
    assert "head_pov" in lane.AVAILABLE_VIDEO_CAMERAS
    assert "robot_pov" in lane.AVAILABLE_VIDEO_CAMERAS
    assert "torso_pov" in lane.AVAILABLE_VIDEO_CAMERAS
    assert lane.DEFAULT_VIDEO_CAMERAS == ("head_pov", "torso_pov")
    assert "third_person" in lane.DIAGNOSTIC_VIDEO_CAMERAS
    assert lane.PREFERRED_G1_POLICY_OBSERVATION_MJCF == "g1_with_hands.xml"
    assert lane.DEFAULT_CONTROLLER_BACKEND == "auto"
    assert lane.DEFAULT_UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ABS == 0.5
    assert lane._unitree_rl_gym_position_target_action_clip_abs() == 0.5
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
    manipulation_discovery = lane.discover_unitree_manipulation_policy(generated_at="now")
    assert manipulation_discovery["status"] == "blocked_missing_hand_policy_runtime"
    assert manipulation_discovery["unitree_hand_manipulation_policy_used"] is False
    assert manipulation_discovery["current_mujoco_manipulation_policy_kind"] == "contact_trace_proxy_only"
    assert "blocked_dexterous_hand_policy_not_integrated" in manipulation_discovery["blockers"]
    manipulation_command = tmp_path / "unitree-manipulation-command"
    manipulation_command.write_text("#!/bin/sh\n", encoding="utf-8")
    manipulation_checkpoint = tmp_path / "unitree-hand-policy.pt"
    manipulation_checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        str(manipulation_command),
    )
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
        str(manipulation_checkpoint),
    )
    ready_manipulation = lane.discover_unitree_manipulation_policy(generated_at="now")
    assert ready_manipulation["status"] == "candidate_ready"
    assert ready_manipulation["ready_candidate_count"] == 1
    assert ready_manipulation["selected_candidate_id"] == "unitree_lerobot_g1_dex"
    assert ready_manipulation["unitree_hand_manipulation_policy_used"] is False
    assert ready_manipulation["can_claim_vla_or_dexterous_manipulation"] is False
    monkeypatch.delenv("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND")
    monkeypatch.delenv("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT")
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
    unitree_rl_gym_env_row = next(
        row
        for row in navigation["candidate_root_envs"]
        if row["env"] == "BLUEPRINT_UNITREE_RL_GYM_ROOT"
    )
    assert unitree_rl_gym_env_row["required_files_present"] is True
    assert "blocked_missing_realistic_g1_navigation_policy" not in navigation["blockers"]
    monkeypatch.delenv("BLUEPRINT_UNITREE_RL_GYM_ROOT")
    monkeypatch.setenv("BLUEPRINT_REALISTIC_G1_POLICY_COMMAND", "'unterminated")
    malformed_navigation = lane.discover_realistic_navigation_policy(generated_at="now")
    assert malformed_navigation["candidate_command_envs"][0]["configured"] is True
    assert malformed_navigation["candidate_command_envs"][0]["available"] is False
    monkeypatch.delenv("BLUEPRINT_REALISTIC_G1_POLICY_COMMAND")
    for name in (
        "BLUEPRINT_UNITREE_LEROBOT_ROOT",
        "BLUEPRINT_UNITREE_LEROBOT_PYTHON",
        "BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

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
    g1_variant_root = tmp_path / "g1_variant"
    (g1_variant_root / "assets").mkdir(parents=True)
    (g1_variant_root / "g1.xml").write_text("<mujoco/>", encoding="utf-8")
    fallback_xml, fallback_selection = lane._select_g1_policy_observation_mjcf(g1_variant_root)
    assert fallback_xml.name == "g1.xml"
    assert fallback_selection["hands_capable_g1_mjcf_selected"] is False
    (g1_variant_root / "g1_with_hands.xml").write_text("<mujoco/>", encoding="utf-8")
    for mesh_name in fallback_selection["hand_mesh_probe_names"]:
        (g1_variant_root / "assets" / mesh_name).write_text("mesh", encoding="utf-8")
    hands_xml, hands_selection = lane._select_g1_policy_observation_mjcf(g1_variant_root)
    assert hands_xml.name == "g1_with_hands.xml"
    assert hands_selection["hands_capable_g1_mjcf_available"] is True
    assert hands_selection["hands_capable_g1_mjcf_selected"] is True
    assert hands_selection["hand_meshes_present"] is True
    assert (
        hands_selection["claim_boundary"][
            "hands_capable_mjcf_does_not_prove_dexterous_hand_policy_execution"
        ]
        is True
    )
    g1_with_torso = tmp_path / "g1_with_torso.xml"
    g1_with_torso.write_text(
        '<mujoco model="g1"><worldbody><body name="torso_link"/></worldbody></mujoco>',
        encoding="utf-8",
    )
    camera_manifest = lane._add_g1_fixed_egocentric_cameras(g1_with_torso)
    camera_xml = g1_with_torso.read_text(encoding="utf-8")
    assert camera_manifest["status"] == "completed"
    assert "blueprint_g1_head_pov" in camera_xml
    assert "blueprint_g1_torso_pov" in camera_xml
    assert camera_manifest["truth_boundary"]["camera_mounted_in_mujoco_g1_mjcf"] is True


def test_policy_action_model_command_execution_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch, _UNITREE_POLICY_ENV_VARS)
    job_dir = tmp_path / "policy-action-command"
    job_dir.mkdir()
    blocked = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=False,
        timeout_seconds=1,
    )
    assert blocked["status"] == "blocked"
    assert blocked["policy_action_model_command_ran"] is False
    assert "missing_cli_allow_policy_action_model_command_run" in blocked["blockers"]

    command = tmp_path / "unitree_policy_command.py"
    command.write_text(
        """
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_POLICY_ACTION_INPUT"]).read_text(encoding="utf-8"))
assert request["schema_version"] == "policy_action_model_command_input.v1"
frame_path = request["observation"]["camera_frame_path"]
assert frame_path
assert Path(frame_path).is_file()
assert request["observation"]["visual_observation"]["available"] is True
payload = {
    "schema_version": "unitree_g1_policy_command_adapter.v1",
    "status": "completed",
    "unitree_policy_action_command_ran": True,
    "normalized_action": {
        "action_type": "waypoint",
        "target_waypoint": [0.5, 0.0],
    },
}
Path(os.environ["BLUEPRINT_POLICY_ACTION_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    (frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")
    checkpoint = tmp_path / "unitree-g1-policy.pt"
    checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_POLICY_COMMAND", f"{sys.executable} {command}")
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    completed = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=5,
    )

    assert completed["status"] == "completed"
    assert completed["policy_action_model_command_ran"] is True
    assert completed["openvla_policy_action_command_ran"] is False
    assert completed["openvla_model_executed"] is False
    assert completed["openvla_predict_action_invoked"] is False
    assert completed["unitree_policy_action_command_ran"] is True
    assert completed["selected_candidate_id"] == "unitree_g1_policy"
    assert completed["action_payload_redacted"]["action_type"] == "waypoint"
    assert (job_dir / "policy_action_model_command_discovery.json").is_file()
    assert (job_dir / "policy_action_model_command_execution.json").is_file()


def test_policy_action_model_timeout_writes_blocked_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

    job_dir = tmp_path / "policy-action-timeout"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    (frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")
    command = tmp_path / "slow_policy_command.py"
    command.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
    checkpoint = tmp_path / "unitree-g1-policy.pt"
    checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_POLICY_COMMAND", f"{sys.executable} {command}")
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    execution = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=0.1,
    )

    output = json.loads(
        (job_dir / "policy_action_model_command_output.json").read_text(encoding="utf-8")
    )
    assert execution["status"] == "blocked"
    assert execution["selected_candidate_id"] == "unitree_g1_policy"
    assert execution["policy_action_model_command_ran"] is False
    assert "policy_action_model_command_failed:TimeoutExpired" in execution["blockers"]
    assert output["status"] == "blocked"
    assert output["policy_action_model_command_ran"] is False
    assert output["action_payload_present"] is False
    assert "policy_action_model_command_failed:TimeoutExpired" in output["blockers"]
    assert output["claim_boundary"]["blocked_output_is_not_model_proof"] is True


def test_policy_action_model_blocks_one_shot_vast_launcher_for_repeated_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch, _UNITREE_POLICY_ENV_VARS)
    monkeypatch.delenv("BLUEPRINT_ALLOW_PROVIDER_LAUNCH_PER_POLICY_INFERENCE", raising=False)
    job_dir = tmp_path / "policy-action-vast-one-shot"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    (frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")
    checkpoint = tmp_path / "finetuned_n17_unitree_g1_sonic"
    checkpoint.mkdir()
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        f"{sys.executable} -m blueprint_pipeline.unitree_groot_n17_sonic_vast_policy_command",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "nvidia/GEAR-SONIC")
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    discovery = lane.discover_policy_action_model_commands(generated_at="now")
    row = next(
        item
        for item in discovery["candidates"]
        if item["candidate_id"] == "unitree_groot_n17_sonic_policy"
    )

    assert discovery["status"] == "blocked_missing_unitree_policy_action_model_command"
    assert discovery["selected_candidate_id"] == "unitree_groot_n17_sonic_policy"
    assert row["ready_for_policy_action_command"] is False
    assert row["policy_worker_invocation_kind"] == "one_shot_provider_launcher"
    assert row["provider_instance_launch_per_inference"] is True
    assert row["repeated_policy_loop_allowed"] is False
    assert (
        "one_shot_provider_launcher_not_allowed_for_repeated_policy_loop"
        in row["blockers"]
    )

    execution = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=5,
    )

    assert execution["status"] == "blocked"
    assert execution["policy_action_model_command_ran"] is False
    assert Path(execution["provider_worker_contract_path"]).is_file()
    assert (
        "one_shot_provider_launcher_not_allowed_for_repeated_policy_loop"
        in execution["blockers"]
    )
    provider_contract = json.loads(
        Path(execution["provider_worker_contract_path"]).read_text(encoding="utf-8")
    )
    assert (
        provider_contract["policy_command_classification"]["invocation_kind"]
        == "one_shot_provider_launcher"
    )
    assert (
        provider_contract["policy_command_classification"][
            "provider_instance_launch_per_inference"
        ]
        is True
    )


def test_policy_action_model_accepts_provider_worker_http_command_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _UNITREE_POLICY_ENV_VARS)
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        f"{sys.executable} -m blueprint_pipeline.provider_worker_policy_command_adapter",
    )
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",
        "provider-worker/unitree-g1-policy",
    )

    discovery = lane.discover_policy_action_model_commands(generated_at="now")
    row = next(
        item
        for item in discovery["candidates"]
        if item["candidate_id"] == "unitree_g1_policy"
    )

    assert discovery["status"] == "ready"
    assert discovery["selected_candidate_id"] == "unitree_g1_policy"
    assert row["ready_for_policy_action_command"] is True
    assert row["policy_worker_invocation_kind"] == "persistent_backend_client_command"
    assert row["repeated_policy_loop_allowed"] is True
    assert row["provider_instance_launch_per_inference"] is False
    assert row["checkpoint_reference_kind"] == "repo_id"


def test_policy_action_model_command_timeout_respects_gpu_scale_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(lane.POLICY_ACTION_MODEL_COMMAND_TIMEOUT_ENV, raising=False)
    assert lane._policy_action_model_command_timeout_seconds(8.0) == 1800.0
    assert lane._policy_action_model_command_timeout_seconds(120.0) == 120.0

    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_TIMEOUT_ENV, "2400")
    assert lane._policy_action_model_command_timeout_seconds(8.0) == 2400.0


def test_policy_action_model_input_uses_scene_wam_episode_packet(tmp_path: Path) -> None:
    job_dir = tmp_path / "scene-policy-action"
    job_dir.mkdir()
    frame = job_dir / "rendered_observations" / "kitchen_head_pov.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"fake-jpeg")
    observation = {
        "schema_version": "scene_wam_policy_initial_observation.v1",
        "task_id": "turn_on_sink_handle",
        "target_object_id": "Sink054_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {
            "available": True,
            "camera_frame_path": str(frame),
            "camera_id": "head_pov",
        },
        "state": {"target_object_id": "Sink054_handle"},
    }
    observation_path = job_dir / "initial_policy_observation.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    packet_path = job_dir / "scene_wam_policy_episode_packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "scene_wam_policy_episode_packet.v1",
                "task_id": "turn_on_sink_handle",
                "target_object_id": "Sink054_handle",
                "initial_policy_observation_path": str(observation_path),
                "initial_policy_observation_frame_path": str(frame),
            }
        ),
        encoding="utf-8",
    )

    sample = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)

    assert sample["scene_wam_policy_episode_packet_path"] == str(packet_path)
    assert sample["observation"]["task_id"] == "turn_on_sink_handle"
    assert sample["observation"]["target_object_id"] == "Sink054_handle"
    assert sample["observation"]["camera_frame_path"] == str(frame)
    assert sample["observation"]["visual_observation"]["available"] is True
    assert "sink handle" in sample["task_prompt"]
    assert sample["claim_boundary"]["task_specific_finetuning_required_for_admission"] is False
    (job_dir / "policy_action_model_command_input.json").write_text(
        json.dumps(sample),
        encoding="utf-8",
    )
    scene_task = lane._policy_action_scene_task(job_dir)
    final_question, success_field = lane._final_success_question_for_scene_task(scene_task)
    assert final_question == "Did the sink handle end up turned on?"
    assert success_field == "sink_handle_turned_on"


def test_policy_action_model_input_preserves_capture_derived_pov_boundary(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "scene-policy-action-capture-derived"
    job_dir.mkdir()
    frame = job_dir / "capture_derived_robot_pov_synthesis" / "candidate.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"fake-jpeg")
    observation = {
        "schema_version": "scene_wam_policy_initial_observation.v1",
        "task_id": "turn_on_sink_handle",
        "target_object_id": "Sink054_handle",
        "camera_frame_path": str(frame),
        "capture_derived_robot_pov_frame_path": str(frame),
        "visual_observation": {
            "available": True,
            "camera_frame_path": str(frame),
            "camera_id": "head_pov",
            "capture_derived_robot_pov_synthesis_used": True,
            "synthesized_or_splatted_outputs_are_not_raw_capture_truth": True,
        },
    }
    observation_path = job_dir / "initial_policy_observation.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    packet_path = job_dir / "scene_wam_policy_episode_packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "scene_wam_policy_episode_packet.v1",
                "task_id": "turn_on_sink_handle",
                "target_object_id": "Sink054_handle",
                "initial_policy_observation_path": str(observation_path),
                "initial_policy_observation_frame_path": str(frame),
                "capture_derived_robot_pov_synthesis_used": True,
            }
        ),
        encoding="utf-8",
    )

    sample = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)

    assert sample["observation"]["visual_observation"]["available"] is True
    assert sample["observation"]["visual_observation"]["simulated_camera_view"] is False
    assert (
        sample["observation"]["visual_observation"]["capture_derived_robot_pov_synthesis_used"]
        is True
    )
    assert sample["claim_boundary"]["visual_frame_is_simulated_mujoco_policy_observation"] is False
    assert sample["claim_boundary"]["visual_frame_is_capture_derived_synthetic_robot_pov"] is True
    assert sample["claim_boundary"]["visual_frame_is_raw_capture_truth"] is False


def test_policy_action_model_input_routes_external_photoreal_frame_into_visual_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "scene-policy-action-ext"
    job_dir.mkdir()
    frame = tmp_path / "isaac_render" / "kitchen_head_pov.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"fake-isaac-rtx-png")
    monkeypatch.setenv(lane.EXTERNAL_PHOTOREAL_OBSERVATION_FRAME_ENV, str(frame))
    monkeypatch.setenv(
        lane.EXTERNAL_PHOTOREAL_OBSERVATION_SOURCE_ENV,
        "isaac_splat_nurec_render",
    )

    sample = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)
    observation = sample["observation"]
    visual = observation["visual_observation"]
    assert observation["camera_frame_path"] == str(frame.resolve())
    assert visual["camera_frame_path"] == str(frame.resolve())
    assert visual["external_photoreal_observation_used"] is True
    assert visual["photoreal_observation_source"] == "isaac_splat_nurec_render"
    assert visual["simulated_camera_view"] is False
    assert visual["physical_robot_sensor_proof"] is False
    assert visual["synthesized_or_splatted_outputs_are_not_raw_capture_truth"] is True
    assert sample["claim_boundary"]["visual_frame_is_external_photoreal_handoff"] is True
    assert sample["claim_boundary"]["visual_frame_is_simulated_mujoco_policy_observation"] is False
    assert sample["claim_boundary"]["visual_frame_is_raw_capture_truth"] is False
    assert sample["claim_boundary"]["mujoco_owns_physics_external_lane_owns_pixels"] is True

    monkeypatch.delenv(lane.EXTERNAL_PHOTOREAL_OBSERVATION_FRAME_ENV)
    monkeypatch.delenv(lane.EXTERNAL_PHOTOREAL_OBSERVATION_SOURCE_ENV)
    base = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)
    assert not base["observation"]["visual_observation"].get(
        "external_photoreal_observation_used"
    )
    assert base["observation"]["state"] == observation["state"]
    assert base["observation"]["unitree_g1_sonic_state"] == observation["unitree_g1_sonic_state"]


def test_scene_packet_policy_action_model_input_routes_external_photoreal_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "scene-policy-action-scene-ext"
    job_dir.mkdir()
    frame = job_dir / "rendered_observations" / "kitchen_head_pov.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"fake-jpeg")
    observation = {
        "schema_version": "scene_wam_policy_initial_observation.v1",
        "task_id": "turn_on_sink_handle",
        "target_object_id": "Sink054_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {
            "available": True,
            "camera_frame_path": str(frame),
            "camera_id": "head_pov",
        },
        "state": {"target_object_id": "Sink054_handle"},
    }
    observation_path = job_dir / "initial_policy_observation.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    packet_path = job_dir / "scene_wam_policy_episode_packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "scene_wam_policy_episode_packet.v1",
                "task_id": "turn_on_sink_handle",
                "target_object_id": "Sink054_handle",
                "initial_policy_observation_path": str(observation_path),
                "initial_policy_observation_frame_path": str(frame),
            }
        ),
        encoding="utf-8",
    )
    base = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)

    external = tmp_path / "isaac_render" / "head_pov.png"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"fake-isaac-rtx-png")
    monkeypatch.setenv(lane.EXTERNAL_PHOTOREAL_OBSERVATION_FRAME_ENV, str(external))
    monkeypatch.setenv(
        lane.EXTERNAL_PHOTOREAL_OBSERVATION_SOURCE_ENV,
        "isaac_splat_nurec_render",
    )
    sample = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)
    observation = sample["observation"]
    visual = observation["visual_observation"]
    assert sample["scene_wam_policy_episode_packet_path"] == str(packet_path)
    assert observation["task_id"] == "turn_on_sink_handle"
    assert observation["target_object_id"] == "Sink054_handle"
    assert observation["state"] == base["observation"]["state"]
    assert observation["unitree_g1_sonic_state"] == base["observation"]["unitree_g1_sonic_state"]
    assert observation["camera_frame_path"] == str(external.resolve())
    assert visual["external_photoreal_observation_used"] is True
    assert visual["photoreal_observation_source"] == "isaac_splat_nurec_render"
    assert visual["simulated_camera_view"] is False
    assert sample["claim_boundary"]["visual_frame_is_external_photoreal_handoff"] is True
    assert sample["claim_boundary"]["visual_frame_is_simulated_mujoco_policy_observation"] is False
    assert sample["claim_boundary"]["visual_frame_is_raw_capture_truth"] is False

    monkeypatch.delenv(lane.EXTERNAL_PHOTOREAL_OBSERVATION_FRAME_ENV)
    monkeypatch.delenv(lane.EXTERNAL_PHOTOREAL_OBSERVATION_SOURCE_ENV)
    fallback = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)
    assert fallback["observation"]["camera_frame_path"] == str(frame)
    assert not fallback["observation"]["visual_observation"].get(
        "external_photoreal_observation_used"
    )


def test_external_photoreal_initial_observation_relabels_wam_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch, _UNITREE_POLICY_ENV_VARS)
    _clear_env(monkeypatch, _WAM_RUNTIME_ENV_VARS)
    job_dir = tmp_path / "external-photoreal-loop"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    (frame_dir / "step_000000.jpg").write_bytes(b"mujoco-jpeg")
    external = tmp_path / "wam_generator" / "head_pov.png"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"external-photoreal")
    monkeypatch.setenv(lane.EXTERNAL_PHOTOREAL_OBSERVATION_FRAME_ENV, str(external))
    monkeypatch.setenv(lane.EXTERNAL_PHOTOREAL_OBSERVATION_SOURCE_ENV, "wam_generator")

    command = tmp_path / "unitree_lerobot_policy_command.py"
    command.write_text(
        """
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_POLICY_ACTION_INPUT"]).read_text(encoding="utf-8"))
frame = request["observation"].get("camera_frame_path")
assert Path(frame).read_bytes() == b"external-photoreal"
payload = {
    "schema_version": "unitree_lerobot_policy_command_adapter.v1",
    "status": "completed",
    "policy_id": "unitree_lerobot_g1_policy",
    "unitree_lerobot_policy_action_command_ran": True,
    "action_chunk": [0.1, 0.2, 0.3],
}
Path(os.environ["BLUEPRINT_POLICY_ACTION_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "unitree-lerobot-hand-policy.pt"
    checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        f"{sys.executable} {command}",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    execution = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=15,
    )
    assert execution["status"] == "completed"
    _configure_fake_live_wam(tmp_path, monkeypatch, name="fake_external_photoreal_wam")
    loop = lane.run_robot_policy_wam_closed_loop_attempt(
        job_dir=job_dir,
        generated_at="now",
        policy_action_model_command_execution=execution,
        loop_step_count=2,
        timeout_seconds=15,
    )

    trace_path = job_dir / "robot_policy_wam_closed_loop" / "robot_policy_wam_loop_trace.jsonl"
    trace = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert loop["status"] == "completed"
    assert trace[0]["observation_source"] == "initial_external_photoreal_observation"
    assert trace[0]["photoreal_observation_source"] == "wam_generator"
    assert trace[0]["policy_observation_frame_path"] == str(external.resolve())
    assert any(row["observation_source"] == "wam_generated_next_observation" for row in trace[1:])


def test_unitree_lerobot_policy_action_model_can_drive_wam_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    job_dir = tmp_path / "unitree-lerobot-loop"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    (frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")
    command = tmp_path / "unitree_lerobot_policy_command.py"
    command.write_text(
        """
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_POLICY_ACTION_INPUT"]).read_text(encoding="utf-8"))
observation = request["observation"]
frame = observation.get("camera_frame_path") or observation["visual_observation"]["camera_frame_path"]
assert Path(frame).is_file()
payload = {
    "schema_version": "unitree_lerobot_policy_command_adapter.v1",
    "status": "completed",
    "policy_id": "unitree_lerobot_g1_policy",
    "unitree_lerobot_policy_action_command_ran": True,
    "action_chunk": [0.1, 0.2, 0.3],
}
Path(os.environ["BLUEPRINT_POLICY_ACTION_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "unitree-lerobot-hand-policy.pt"
    checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        f"{sys.executable} {command}",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    execution = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=15,
    )
    assert execution["status"] == "completed"
    assert execution["selected_candidate_id"] == "unitree_lerobot_policy"
    assert execution["unitree_policy_action_command_ran"] is True
    assert execution["unitree_lerobot_policy_action_command_ran"] is True
    assert execution["unitree_manipulation_policy_action_command_ran"] is True
    assert execution["action_payload_redacted"]["unitree_action_chunk_present"] is True

    _configure_fake_live_wam(tmp_path, monkeypatch, name="fake_lerobot_live_wam")
    loop = lane.run_robot_policy_wam_closed_loop_attempt(
        job_dir=job_dir,
        generated_at="now",
        policy_action_model_command_execution=execution,
        loop_step_count=3,
        timeout_seconds=15,
    )

    assert loop["status"] == "completed"
    assert loop["wam_evaluator_in_control_loop"] is True
    assert loop["policy_observes_wam_generated_next_observation"] is True
    assert loop["unitree_policy_action_command_ran"] is True
    assert loop["unitree_lerobot_policy_action_command_ran"] is True
    assert loop["repeated_policy_calls_count"] >= 2
    assert loop["generated_next_observation_count"] >= 1
    assert loop["live_wam_generation_command_ran"] is True
    assert loop["action_conditioned_generation_ran"] is True
    assert loop["live_wam_generation_success_count"] >= 1
    assert loop["structural_wam_generation_count"] == 0
    assert (job_dir / "robot_policy_wam_closed_loop" / "robot_policy_wam_loop_trace.jsonl").is_file()
    assert (
        job_dir / "robot_policy_wam_closed_loop" / "wam_generated_next_observations.jsonl"
    ).is_file()
    assert (
        job_dir / "robot_policy_wam_closed_loop" / "wam_generation_command_discovery.json"
    ).is_file()
    assert (
        job_dir / "robot_policy_wam_closed_loop" / "wam_generation_command_execution.json"
    ).is_file()
    assert (
        job_dir / "robot_policy_wam_closed_loop" / "wam_generation_command_output.json"
    ).is_file()


def test_unitree_policy_wam_loop_uses_default_action_skeleton_generation_without_live_wam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_OSCAR_WAM_COMMAND",
        "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND",
        "BLUEPRINT_COSMOS_WAM_COMMAND",
        "BLUEPRINT_COSMOS_WAM_PROVIDER_COMMAND",
        "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND",
        "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
        "BLUEPRINT_COSMOS_WAM_CHECKPOINT",
        "BLUEPRINT_COSMOS3_WAM_CHECKPOINT",
        lane.WAM_GENERATION_COMMAND_GATE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    job_dir = tmp_path / "unitree-default-wam-loop"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    source_frame = frame_dir / "step_000000.jpg"
    source_frame.write_bytes(b"fake-jpeg")
    lane._write_jsonl(
        job_dir / "g1_projected_skeleton_trace.jsonl",
        [
            {
                "schema_version": lane.G1_PROJECTED_SKELETON_SCHEMA_ID,
                "status": "completed",
                "episode_id": "episode_0001",
                "scenario_eval_run_id": "scenario-1",
                "step": 0,
                "camera_id": "head_pov",
                "camera_frame_path": str(source_frame),
                "projected_landmark_count": 4,
                "landmarks": [
                    {
                        "landmark_id": "left_shoulder",
                        "image_projection": {"available": True, "u_px": 220, "v_px": 190},
                    },
                    {
                        "landmark_id": "left_elbow",
                        "image_projection": {"available": True, "u_px": 248, "v_px": 245},
                    },
                    {
                        "landmark_id": "left_wrist",
                        "image_projection": {"available": True, "u_px": 280, "v_px": 296},
                    },
                    {
                        "landmark_id": "left_hand",
                        "image_projection": {"available": True, "u_px": 310, "v_px": 318},
                    },
                ],
                "segments": [
                    {"from": "left_shoulder", "to": "left_elbow"},
                    {"from": "left_elbow", "to": "left_wrist"},
                    {"from": "left_wrist", "to": "left_hand"},
                ],
            }
        ],
    )
    command = tmp_path / "unitree_lerobot_policy_command.py"
    command.write_text(
        """
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_POLICY_ACTION_INPUT"]).read_text(encoding="utf-8"))
observation = request["observation"]
frame = observation.get("camera_frame_path") or observation["visual_observation"]["camera_frame_path"]
assert Path(frame).is_file()
payload = {
    "schema_version": "unitree_lerobot_policy_command_adapter.v1",
    "status": "completed",
    "policy_id": "unitree_lerobot_g1_policy",
    "unitree_lerobot_policy_action_command_ran": True,
    "action_type": "joint_delta_chunk",
    "action_chunk": [0.1, -0.05, 0.2, 0.0, 0.03],
}
Path(os.environ["BLUEPRINT_POLICY_ACTION_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "unitree-lerobot-hand-policy.pt"
    checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        f"{sys.executable} {command}",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    execution = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=15,
    )
    assert execution["status"] == "completed"

    loop = lane.run_robot_policy_wam_closed_loop_attempt(
        job_dir=job_dir,
        generated_at="now",
        policy_action_model_command_execution=execution,
        loop_step_count=3,
        timeout_seconds=15,
    )

    assert loop["status"] == "completed"
    assert loop["policy_observes_wam_generated_next_observation"] is True
    assert loop["live_wam_generation_command_ran"] is False
    assert loop["action_conditioned_generation_ran"] is True
    assert loop["default_local_wam_generator_used"] is True
    assert loop["default_wam_generation_success_count"] == 2
    assert loop["live_wam_generation_success_count"] == 0
    assert loop["structural_wam_generation_count"] == 0
    assert "blocked_live_wam_generation_command_not_run" not in loop["blockers"]
    assert "blocked_wam_action_conditioned_generation_not_run" not in loop["blockers"]

    generated_rows = [
        json.loads(line)
        for line in Path(loop["generated_next_observation_trace"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(generated_rows) == 2
    assert generated_rows[0]["default_local_wam_generator_used"] is True
    assert generated_rows[0]["action_conditioning"]["numeric_action_value_count"] >= 5
    assert (
        generated_rows[0]["skeleton_conditioning"]["projected_skeleton_trace_used"]
        is True
    )
    generated_frame = Path(generated_rows[0]["generated_next_observation_frame_path"])
    assert generated_frame.is_file()
    assert generated_frame.read_bytes() != b"fake-jpeg"
    generated_video = Path(generated_rows[0]["generated_next_observation_video_path"])
    assert generated_video.is_file()
    assert generated_video.stat().st_size > 0
    assert "wam_derived_observation" in generated_rows[0]
    assert Path(loop["wam_derived_observation_manifest"]).is_file()
    assert Path(loop["wam_derived_observation_bundle"]).is_file()
    assert Path(loop["wam_perception_harness_checks"]).is_file()
    assert Path(loop["wam_policy_observation_adapter_report"]).is_file()
    assert Path(loop["wam_perception_harness_validation_report"]).is_file()
    assert Path(loop["wam_false_success_reduction_metrics"]).is_file()
    assert Path(loop["wam_perception_harness_review_report"]).is_file()
    assert loop["wam_derived_observation_step_count"] == 2
    assert loop["wam_derived_observation_early_termination_recommended"] is False
    assert loop["wam_perception_harness_backend_config"]["backend_kind"] == "fixture"
    assert (
        loop["wam_perception_harness_backend_config"]["raw_credentials_written_to_artifacts"]
        is False
    )
    assert loop["wam_perception_harness_validation_status"] == "not_requested"
    adapter_report = json.loads(
        Path(loop["wam_policy_observation_adapter_report"]).read_text(encoding="utf-8")
    )
    latest_adapter = adapter_report["latest_policy_adapter_report"]
    assert "objects" in latest_adapter["fields_withheld_due_to_contract"]
    assert "depth_estimates" not in latest_adapter["fields_withheld_due_to_contract"]
    assert "depth_estimates" in latest_adapter["fields_supplied_to_policy"]
    assert "depth_estimates" in latest_adapter["adapted_policy_observation"]
    assert "objects" not in latest_adapter["adapted_policy_observation"]
    harness_checks = json.loads(Path(loop["wam_perception_harness_checks"]).read_text(encoding="utf-8"))
    assert harness_checks["forward_inverse_consistency_proven"] is False

    command_execution = json.loads(
        (job_dir / "robot_policy_wam_closed_loop" / "wam_generation_command_execution.json")
        .read_text(encoding="utf-8")
    )
    command_output = json.loads(
        (job_dir / "robot_policy_wam_closed_loop" / "wam_generation_command_output.json")
        .read_text(encoding="utf-8")
    )
    assert command_execution["status"] == "completed"
    assert command_execution["command_ran_count"] == 0
    assert command_execution["default_wam_generation_success_count"] == 2
    assert command_output["status"] == "completed"
    assert command_output["default_local_wam_generator_used"] is True
    assert command_output["learned_oscar_or_cosmos_model_ran"] is False
    assert command_output["outputs"][0]["video_materialization"]["status"] == "completed"
    assert Path(
        command_output["outputs"][0]["video_materialization"][
            "generated_video_segment_path"
        ]
    ).is_file()
    assert (
        command_output["claim_boundary"]["default_local_outputs_are_support_evidence_only"]
        is True
    )


def test_unitree_groot_n17_sonic_policy_action_model_can_drive_wam_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_HOST",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_PORT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_TOKEN_FILE",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT",
    ):
        monkeypatch.delenv(name, raising=False)
    job_dir = tmp_path / "unitree-groot-loop"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    (frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")
    command = tmp_path / "unitree_groot_n17_sonic_policy_command.py"
    command.write_text(
        """
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_POLICY_ACTION_INPUT"]).read_text(encoding="utf-8"))
observation = request["observation"]
frame = observation.get("camera_frame_path") or observation["visual_observation"]["camera_frame_path"]
assert Path(frame).is_file()
payload = {
    "schema_version": "unitree_groot_n17_sonic_policy_command_adapter.v1",
    "status": "completed",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "unitree_groot_n17_sonic_policy_action_command_ran": True,
    "action_chunk": [0.1] * 78,
}
Path(os.environ["BLUEPRINT_POLICY_ACTION_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        f"{sys.executable} {command}",
    )
    n17_checkpoint = tmp_path / "finetuned_n17_unitree_g1_sonic"
    n17_checkpoint.mkdir()
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", str(n17_checkpoint))
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "nvidia/GEAR-SONIC")
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    execution = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=5,
    )
    assert execution["status"] == "completed"
    assert execution["selected_candidate_id"] == "unitree_groot_n17_sonic_policy"
    assert execution["unitree_policy_action_command_ran"] is True
    assert execution["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert execution["unitree_manipulation_policy_action_command_ran"] is True
    assert execution["action_payload_redacted"]["unitree_action_chunk_present"] is True

    _configure_fake_live_wam(tmp_path, monkeypatch, name="fake_groot_live_wam")
    loop = lane.run_robot_policy_wam_closed_loop_attempt(
        job_dir=job_dir,
        generated_at="now",
        policy_action_model_command_execution=execution,
        timeout_seconds=5,
    )

    assert loop["status"] == "completed"
    assert loop["requested_loop_step_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT
    assert loop["wam_evaluator_in_control_loop"] is True
    assert loop["policy_observes_wam_generated_next_observation"] is True
    assert loop["unitree_policy_action_command_ran"] is True
    assert loop["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert loop["repeated_policy_calls_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT
    assert loop["fresh_policy_action_call_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT
    assert loop["structural_policy_action_response_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT
    assert loop["provider_output_replay_used"] is False
    assert loop["generated_next_observation_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT - 1
    assert loop["live_wam_generation_success_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT - 1
    assert loop["structural_wam_generation_count"] == 0
    assert loop["action_conditioned_generation_ran"] is True
    assert loop["side_by_side_transition_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT - 1
    side_by_side_manifest = json.loads(
        Path(loop["side_by_side_trace_manifest"]).read_text(encoding="utf-8")
    )
    assert side_by_side_manifest["transition_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT - 1
    assert Path(side_by_side_manifest["trace_html_path"]).is_file()
    assert Path(loop["side_by_side_trace_html_path"]).is_file()
    side_by_side_html = Path(loop["side_by_side_trace_html_path"]).read_text(encoding="utf-8")
    assert "Robot Policy WAM Side By Side Trace" in side_by_side_html
    assert "Transition 1" in side_by_side_html
    side_by_side_rows = [
        json.loads(line)
        for line in Path(loop["side_by_side_trace_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(side_by_side_rows) == lane.DEFAULT_WAM_LOOP_STEP_COUNT - 1
    assert side_by_side_rows[0]["policy_pov_frame_path"].endswith("step_000000.jpg")
    assert side_by_side_rows[0]["policy_action_summary"]["action_chunk_length"] == 78
    assert side_by_side_rows[0]["wam_generated_next_observation_frame_path"].endswith(
        "wam_generated_next_observation_step_0001.jpg"
    )
    assert side_by_side_rows[0]["next_policy_call_status"] == "completed"


def test_unitree_groot_n17_sonic_replay_cannot_satisfy_wam_loop_fresh_policy_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch, _WAM_RUNTIME_ENV_VARS)
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        lane.GROOT_ROOT_ENV,
        lane.WBC_ROOT_ENV,
        lane.N17_CHECKPOINT_ENV,
        lane.SONIC_CHECKPOINT_ENV,
        lane.GROOT_POLICY_COMMAND_ENV,
        lane.POLICY_SERVER_URL_ENV,
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_HOST",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_PORT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_TOKEN_FILE",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_HF_TOKEN_FILE",
        lane.SIM2SIM_COMMAND_ENV,
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT",
    ):
        monkeypatch.delenv(name, raising=False)
    job_dir = tmp_path / "unitree-groot-replay-loop"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    (frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")
    provider_output = tmp_path / "groot_provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "schema_version": "unitree_groot_n17_sonic_policy_command_adapter.v1",
                "status": "completed",
                "policy_id": "unitree_groot_n17_sonic_policy",
                "unitree_groot_n17_sonic_policy_action_command_ran": True,
                "action": {
                    "action_type": "unitree_g1_sonic_action_chunk",
                    "action_chunk": [0.1] * 78,
                },
            }
        ),
        encoding="utf-8",
    )
    n17_checkpoint = tmp_path / "finetuned_n17_unitree_g1_sonic"
    n17_checkpoint.mkdir()
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        f"{sys.executable} -m blueprint_pipeline.unitree_groot_n17_sonic_policy_command_adapter",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", str(n17_checkpoint))
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "nvidia/GEAR-SONIC")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT",
        str(provider_output),
    )
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    execution = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=5,
    )

    assert execution["status"] == "completed"
    assert execution["policy_action_model_command_ran"] is True
    assert execution["provider_output_replay_used"] is True
    assert execution["fresh_policy_action_model_executed_this_invocation"] is False
    assert execution["unitree_policy_action_command_ran"] is False

    loop = lane.run_robot_policy_wam_closed_loop_attempt(
        job_dir=job_dir,
        generated_at="now",
        policy_action_model_command_execution=execution,
        loop_step_count=3,
        timeout_seconds=5,
    )

    assert loop["status"] == "blocked"
    assert loop["wam_evaluator_in_control_loop"] is False
    assert loop["policy_observes_wam_generated_next_observation"] is False
    assert loop["unitree_policy_action_command_ran"] is False
    assert loop["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert loop["repeated_policy_calls_count"] == 0
    assert loop["fresh_policy_action_call_count"] == 0
    assert loop["structural_policy_action_response_count"] >= 2
    assert loop["provider_output_replay_action_response_count"] >= 2
    assert loop["provider_output_replay_used"] is True
    assert "blocked_repeated_fresh_unitree_policy_calls_not_proven" in loop["blockers"]


def test_unifolm_repo_id_checkpoints_are_configured_policy_action_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    command = tmp_path / "unitree_unifolm_vla_command"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND", str(command))
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
        "unitreerobotics/UnifoLM-VLA-Base",
    )
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
        "unitreerobotics/UnifoLM-VLM-Base",
    )

    discovery = lane.discover_policy_action_model_commands(generated_at="now")
    row = next(
        item for item in discovery["candidates"] if item["candidate_id"] == "unitree_unifolm_vla_policy"
    )

    assert discovery["selected_candidate_id"] == "unitree_unifolm_vla_policy"
    assert row["ready_for_policy_action_command"] is True
    assert row["checkpoint_configured"] is True
    assert row["checkpoint_exists"] is False
    assert row["checkpoint_reference_kind"] == "repo_id"
    assert row["extra_required_checkpoints"][0]["checkpoint_reference_kind"] == "repo_id"


def test_groot_sonic_repo_id_checkpoints_are_configured_policy_action_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    command = tmp_path / "unitree_groot_n17_sonic_command"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", str(command))
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "blueprint-test/GR00T-N1.7-UNITREE-G1-SONIC-Finetuned",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "nvidia/GEAR-SONIC")

    discovery = lane.discover_policy_action_model_commands(generated_at="now")
    row = next(
        item
        for item in discovery["candidates"]
        if item["candidate_id"] == "unitree_groot_n17_sonic_policy"
    )

    assert discovery["selected_candidate_id"] == "unitree_groot_n17_sonic_policy"
    assert row["ready_for_policy_action_command"] is True
    assert row["checkpoint_configured"] is True
    assert row["checkpoint_exists"] is False
    assert row["checkpoint_reference_kind"] == "repo_id"
    assert row["extra_required_checkpoints"] == []
    assert row["optional_checkpoints"][0]["checkpoint_reference_kind"] == "repo_id"
    assert row["optional_checkpoints"][0]["required_for_policy_action_admission"] is False


def test_groot_sonic_base_n17_repo_id_uses_default_experimental_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    command = tmp_path / "unitree_groot_n17_sonic_command"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", str(command))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", "nvidia/GR00T-N1.7-3B")
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "nvidia/GEAR-SONIC")

    discovery = lane.discover_policy_action_model_commands(generated_at="now")
    row = next(
        item
        for item in discovery["candidates"]
        if item["candidate_id"] == "unitree_groot_n17_sonic_policy"
    )

    assert discovery["selected_candidate_id"] == "unitree_groot_n17_sonic_policy"
    assert discovery["selected_candidate_ready_for_policy_action_command"] is True
    assert discovery["status"] == "ready"
    assert discovery["blockers"] == []
    assert row["ready_for_policy_action_command"] is True
    assert row["checkpoint_known_base_model_without_unitree_g1_sonic_support"] is True
    assert row["checkpoint_default_applied"] is True
    assert (
        row["checkpoint_path"]
        == lane.DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT
    )
    assert row["task_specific_finetuning_required_for_admission"] is False
    assert row["trusted_for_production"] is False
    assert row["checkpoint_provenance"]["trusted_for_production"] is False
    assert (
        "blocked_BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT_points_to_base_GR00T_N17_without_UNITREE_G1_SONIC_support"
        not in row["blockers"]
    )


def test_groot_sonic_candidate_uses_builtin_command_when_command_env_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    groot_root = tmp_path / "Isaac-GR00T"
    groot_root.mkdir()
    wbc_root = tmp_path / "GR00T-WholeBodyControl"
    wbc_root.mkdir()
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT", str(groot_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT", str(wbc_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", "nvidia/GR00T-N1.7-3B")
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "nvidia/GEAR-SONIC")

    discovery = lane.discover_policy_action_model_commands(generated_at="now")
    row = next(
        item
        for item in discovery["candidates"]
        if item["candidate_id"] == "unitree_groot_n17_sonic_policy"
    )

    assert discovery["status"] == "ready"
    assert discovery["selected_candidate_id"] == "unitree_groot_n17_sonic_policy"
    assert discovery["selected_candidate_ready_for_policy_action_command"] is True
    assert row["command_from_default"] is True
    assert row["command_available"] is True
    assert row["checkpoint_default_applied"] is True
    assert row["checkpoint_path"] == lane.DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT
    assert row["task_specific_finetuning_required_for_admission"] is False


def test_policy_action_model_requires_egocentric_frame_and_ignores_openvla_as_g1_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    job_dir = tmp_path / "policy-action-negative"
    third_person_frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "third_person"
    third_person_frame_dir.mkdir(parents=True)
    (third_person_frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")

    sample = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)

    assert sample["observation"]["visual_observation"]["available"] is False
    assert sample["observation"]["visual_observation"]["camera_id"] is None
    assert sample["observation"]["visual_observation"]["first_person_policy_observation_candidate"] is False
    assert "policy_observation_frame_not_captured" in sample["observation"]["visual_observation"]["blockers"]
    assert sorted(sample["observation"]["unitree_g1_sonic_state"]) == [
        "left_arm",
        "left_hand",
        "left_leg",
        "projected_gravity",
        "right_arm",
        "right_hand",
        "right_leg",
        "waist",
    ]
    assert sample["claim_boundary"]["unitree_g1_sonic_state_is_simulated_observation"] is True
    assert sample["claim_boundary"]["unitree_g1_sonic_state_is_contract_probe"] is True
    assert lane._policy_action_model_frame_candidates(job_dir) == []

    head_frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    head_frame_dir.mkdir(parents=True)
    (head_frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")
    command = tmp_path / "generic_policy_command.py"
    command.write_text(
        """
import json
import os
from pathlib import Path

payload = {
    "schema_version": "policy_action_model_command_output.v1",
    "status": "completed",
    "normalized_action": {"action_type": "waypoint", "target_waypoint": [0.5, 0.0]},
}
Path(os.environ["BLUEPRINT_POLICY_ACTION_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OPENVLA_POLICY_COMMAND", f"{sys.executable} {command}")
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    result = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=5,
    )

    assert result["status"] == "blocked"
    assert result["policy_action_model_command_ran"] is False
    assert result["openvla_policy_action_command_ran"] is False
    assert result.get("openvla_model_executed", False) is False
    assert "blocked_missing_unitree_specific_policy_action_model_command" in result["blockers"]
    blocked_output = json.loads(
        (job_dir / "policy_action_model_command_output.json").read_text(encoding="utf-8")
    )
    assert blocked_output["status"] == "blocked"
    assert blocked_output["action_payload_present"] is False
    assert blocked_output["unitree_policy_action_command_ran"] is False
    assert blocked_output["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert blocked_output["claim_boundary"]["non_ranking_operational_claim_proven"] is False
    discovery = result["discovery"]
    assert discovery["selection_policy"] == "unitree_specific_policy_candidates_only"
    assert discovery["selected_candidate_id"] is None
    openvla_rows = [
        row
        for row in discovery["generic_policy_comparison_candidates"]
        if row["candidate_id"] == "openvla_policy"
    ]
    assert openvla_rows
    assert openvla_rows[0]["command_configured"] is True
    assert openvla_rows[0]["ready_for_policy_action_command"] is False


def test_policy_action_model_input_prefers_captured_unitree_g1_sonic_state(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "policy-action-with-sonic-state"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    frame = frame_dir / "step_000000.jpg"
    frame.write_bytes(b"fake-jpeg")
    state = {
        "left_leg": [1.0] * 6,
        "right_leg": [2.0] * 6,
        "waist": [3.0] * 3,
        "left_arm": [4.0] * 7,
        "right_arm": [5.0] * 7,
        "left_hand": [6.0] * 7,
        "right_hand": [7.0] * 7,
        "projected_gravity": [0.0, 0.0, -1.0],
    }
    (job_dir / "policy_visual_observation_trace.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "policy_visual_observation.v1",
                "available": True,
                "episode_id": "episode_0001",
                "scenario_eval_run_id": "run-1",
                "task_id": "contact_or_push_light_object",
                "step": 0,
                "camera_frame_path": str(frame),
                "unitree_g1_sonic_state": state,
                "unitree_g1_sonic_state_source": "simulated_mujoco_qpos_joint_groups",
                "unitree_g1_sonic_state_metadata": {
                    "complete": True,
                    "missing_joint_names": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = lane._sample_policy_action_model_input(generated_at="now", job_dir=job_dir)

    assert sample["observation"]["unitree_g1_sonic_state"] == state
    assert (
        sample["observation"]["unitree_g1_sonic_state_source"]
        == "simulated_mujoco_qpos_joint_groups"
    )
    assert sample["observation"]["unitree_g1_sonic_state_metadata"]["step"] == 0
    assert sample["claim_boundary"]["unitree_g1_sonic_state_is_contract_probe"] is False
    assert sample["claim_boundary"]["unitree_g1_sonic_state_derived_from_mujoco_qpos"] is True


def test_policy_action_model_execution_preserves_blocked_output_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_OPENVLA_POLICY_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    job_dir = tmp_path / "policy-action-blocked-output"
    frame_dir = job_dir / "policy_observation_frames" / "episode_0001" / "head_pov"
    frame_dir.mkdir(parents=True)
    (frame_dir / "step_000000.jpg").write_bytes(b"fake-jpeg")
    command = tmp_path / "blocked_groot_command.py"
    command.write_text(
        """
import json
import os
from pathlib import Path

payload = {
    "schema_version": "unitree_groot_n17_sonic_policy_server_command.v1",
    "status": "blocked",
    "selected_candidate_id": "unitree_groot_n17_sonic_policy",
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "unitree_policy_action_command_ran": False,
    "unitree_specific_manipulation_candidate_ran": False,
    "blockers": ["set_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL_to_running_gr00t_policy_server"],
}
Path(os.environ["BLUEPRINT_POLICY_ACTION_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
raise SystemExit(2)
""".strip(),
        encoding="utf-8",
    )
    groot_root = tmp_path / "Isaac-GR00T"
    groot_root.mkdir()
    wbc_root = tmp_path / "GR00T-WholeBodyControl"
    wbc_root.mkdir()
    sonic_checkpoint = tmp_path / "sonic"
    sonic_checkpoint.mkdir()
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        f"{sys.executable} {command}",
    )
    n17_checkpoint = tmp_path / "finetuned_n17_unitree_g1_sonic"
    n17_checkpoint.mkdir()
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", str(n17_checkpoint))
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", str(sonic_checkpoint))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT", str(groot_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT", str(wbc_root))
    monkeypatch.setenv(lane.POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "true")

    result = lane.run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at="now",
        allow_policy_action_model_command_run=True,
        timeout_seconds=5,
    )

    assert result["status"] == "blocked"
    assert result["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert "policy_action_model_command_nonzero_exit" in result["blockers"]
    assert (
        "set_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL_to_running_gr00t_policy_server"
        in result["blockers"]
    )


def test_policy_runtime_records_vla_provider_output_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "WAM_POLICY_ENDPOINT_URL",
        "WAM_POLICY_AUTH_TOKEN_FILE",
        "TEAM_POLICY_ENDPOINT_URL",
        "TEAM_POLICY_AUTH_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    token_file = tmp_path / "token.txt"
    token_file.write_text("tok", encoding="utf-8")
    provider_output = tmp_path / "openvla_policy_provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "openvla_model_executed": True,
                "openvla_policy_action_command_ran": True,
                "action": {"action_type": "stop"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLA_POLICY_ENDPOINT_URL", "http://127.0.0.1:8768/policy/action")
    monkeypatch.setenv("VLA_POLICY_AUTH_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("BLUEPRINT_OPENVLA_POLICY_COMMAND", "python -m blueprint_pipeline.openvla_policy_command_adapter")
    monkeypatch.setenv("BLUEPRINT_OPENVLA_PROVIDER_OUTPUT", str(provider_output))

    discovery, runtime, auth, _probe = lane.discover_policy_runtime(generated_at="now")
    selected = lane.selected_endpoint(discovery)

    assert selected is not None
    assert selected["runtime"] == "vla"
    assert selected["model_command_configured"] is True
    assert selected["model_provenance_recorded"] is True
    assert selected["model_provenance_kind"] == "provider_output_replay"
    assert selected["model_provider_output_path"] == str(provider_output)
    assert auth["raw_token_values_persisted"] is False
    assert runtime["endpoint_runtimes"][1]["model_provenance_recorded"] is True


def test_policy_endpoint_boundary_manifest_fixture_credentials_and_real_trace_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "WAM_POLICY_ENDPOINT_URL",
        "WAM_POLICY_AUTH_TOKEN_FILE",
        "VLA_POLICY_ENDPOINT_URL",
        "VLA_POLICY_AUTH_TOKEN_FILE",
        "TEAM_POLICY_ENDPOINT_URL",
        "TEAM_POLICY_AUTH_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)

    discovery, _runtime, _auth, _probe = lane.discover_policy_runtime(generated_at="now")
    fixture_boundary = lane.build_policy_endpoint_boundary_manifest(
        generated_at="now",
        endpoint_discovery=discovery,
        selected_runtime=lane.selected_endpoint(discovery),
        fixture_policy_used=True,
        policy_execution_manifest_path=tmp_path / "missing_policy_execution_manifest.json",
    )

    assert fixture_boundary["status"] == "fixture_boundary_only"
    assert fixture_boundary["endpoint_integration_skipped"] is True
    assert fixture_boundary["fixture_policy_used"] is True
    assert fixture_boundary["robot_policy_execution_proven"] is False
    assert (
        "blocked_fixture_policy_is_not_robot_policy_execution_proof"
        in fixture_boundary["blockers"]
    )
    assert (
        fixture_boundary["claim_boundary"][
            "fixture_policy_is_not_robot_policy_execution"
        ]
        is True
    )

    monkeypatch.setenv("VLA_POLICY_ENDPOINT_URL", "http://127.0.0.1:8768/policy/action")
    no_auth_discovery, _runtime, _auth, _probe = lane.discover_policy_runtime(
        generated_at="now"
    )
    no_auth_boundary = lane.build_policy_endpoint_boundary_manifest(
        generated_at="now",
        endpoint_discovery=no_auth_discovery,
        selected_runtime=lane.selected_endpoint(no_auth_discovery),
        fixture_policy_used=True,
    )

    assert no_auth_boundary["status"] == "decision_needed_missing_policy_credentials"
    assert no_auth_boundary["endpoint_integration_status"] == "configured_missing_credentials"
    assert no_auth_boundary["missing_credentials_decision_needed"] is True
    assert no_auth_boundary["robot_policy_execution_proven"] is False
    assert "blocked_missing_policy_auth_token_file" in no_auth_boundary["blockers"]
    assert "decision_needed_policy_auth_token_file" in no_auth_boundary["blockers"]
    assert (
        no_auth_boundary["claim_boundary"]["missing_credentials_do_not_upgrade_proof"]
        is True
    )

    trace_path = tmp_path / "policy_execution_trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_policy_execution_trace.v1",
                "robot_policy_execution_proven": True,
                "attempt_count": 1,
                "attempts": [{"scenario_eval_run_id": "run-1"}],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "policy_execution_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_policy_execution_manifest.v1",
                "status": "completed",
                "env_BLUEPRINT_ALLOW_POLICY_EXECUTION": True,
                "allow_policy_execution_flag": True,
                "robot_policy_execution_proven": True,
                "robot_team_policy_execution_proven": True,
                "default_test_policy_execution_proven": False,
                "attempt_count": 1,
                "policy_execution_trace_path": "policy_execution_trace.json",
            }
        ),
        encoding="utf-8",
    )
    real_trace_boundary = lane.build_policy_endpoint_boundary_manifest(
        generated_at="now",
        endpoint_discovery=no_auth_discovery,
        selected_runtime=lane.selected_endpoint(no_auth_discovery),
        fixture_policy_used=True,
        policy_execution_manifest_path=manifest_path,
    )

    assert real_trace_boundary["status"] == "completed_robot_policy_execution_trace_proven"
    assert real_trace_boundary["robot_policy_execution_proven"] is True
    assert (
        real_trace_boundary["robot_policy_execution_proof_source"]
        == "gated_policy_execution_manifest"
    )
    assert real_trace_boundary["real_trace_gate"]["gated_real_trace_exists"] is True
    assert real_trace_boundary["claim_boundary"]["robot_policy_execution_proven"] is True
    assert (
        real_trace_boundary["claim_boundary"][
            "endpoint_invocation_is_not_robot_policy_execution"
        ]
        is True
    )
    assert (
        real_trace_boundary["claim_boundary"]["endpoint_setup_is_not_real_world_success"]
        is True
    )
    assert (
        real_trace_boundary["claim_boundary"]["endpoint_setup_is_not_safety_validation"]
        is True
    )
    assert (
        real_trace_boundary["claim_boundary"][
            "endpoint_setup_is_not_deployment_approval"
        ]
        is True
    )


def test_unitree_endpoint_policy_response_summary_keeps_replay_separate() -> None:
    replay_summary = lane._unitree_endpoint_policy_response_summary(
        [
            {
                "policy_id": "unitree_unifolm_vla_policy_provider_replay",
                "provider_output_replay_used": True,
                "action": {
                    "action_type": "manipulation_contact",
                    "unitree_unifolm_action_chunk_present": True,
                },
                "claim_boundary": {
                    "provider_output_replay_used": True,
                    "unitree_hand_manipulation_policy_used": False,
                },
            }
        ]
    )

    assert replay_summary["unitree_endpoint_hand_policy_output_observed"] is True
    assert replay_summary["unitree_endpoint_provider_output_replay_used"] is True
    assert replay_summary["unitree_endpoint_action_chunk_used"] is True
    assert replay_summary["unitree_endpoint_fresh_policy_action_command_ran"] is False
    assert replay_summary["unitree_endpoint_hand_policy_used"] is False
    assert replay_summary["g1_robot_policy_selected_family"] is None
    assert replay_summary["unitree_hand_manipulation_policy_scope"] is None
    assert replay_summary["openvla_selected_as_g1_robot_policy"] is False
    assert replay_summary["wam_rollout_selected_as_g1_robot_policy"] is False

    fresh_summary = lane._unitree_endpoint_policy_response_summary(
        [
            {
                "policy_id": "unitree_unifolm_vla_policy",
                "unitree_unifolm_policy_action_command_ran": True,
                "action": {
                    "action_type": "manipulation_contact",
                    "unitree_unifolm_action_chunk_present": True,
                },
                "claim_boundary": {
                    "provider_output_replay_used": False,
                    "unitree_hand_manipulation_policy_used": True,
                },
            }
        ]
    )

    assert fresh_summary["unitree_endpoint_hand_policy_output_observed"] is True
    assert fresh_summary["unitree_endpoint_provider_output_replay_used"] is False
    assert fresh_summary["unitree_endpoint_fresh_policy_action_command_ran"] is True
    assert fresh_summary["unitree_endpoint_hand_policy_used"] is True
    assert fresh_summary["g1_robot_policy_selected_family"] == "unitree_native_hand_policy_endpoint"
    assert fresh_summary["unitree_hand_manipulation_policy_scope"] == "endpoint_action_command"
    assert fresh_summary["openvla_selected_as_g1_robot_policy"] is False
    assert fresh_summary["wam_rollout_selected_as_g1_robot_policy"] is False


def test_policy_action_provider_replay_propagates_to_final_truth() -> None:
    assert (
        lane._policy_action_provider_output_replay_used(
            policy_action_model_command_execution={"provider_output_replay_used": True},
            robot_policy_wam_closed_loop_attempt={"provider_output_replay_used": False},
        )
        is True
    )
    assert (
        lane._policy_action_provider_output_replay_used(
            policy_action_model_command_execution={"provider_output_replay_used": False},
            robot_policy_wam_closed_loop_attempt={"provider_output_replay_used": True},
        )
        is True
    )
    assert (
        lane._policy_action_provider_output_replay_used(
            policy_action_model_command_execution={},
            robot_policy_wam_closed_loop_attempt={},
        )
        is False
    )


def test_policy_endpoint_runtime_manifest_exposes_unitree_replay_boundary() -> None:
    replay_summary = lane._unitree_endpoint_policy_response_summary(
        [
            {
                "policy_id": "unitree_unifolm_vla_policy_provider_replay",
                "action": {
                    "action_type": "manipulation_contact",
                    "unitree_unifolm_action_chunk_present": True,
                },
                "claim_boundary": {"provider_output_replay_used": True},
            }
        ]
    )

    manifest = lane.build_policy_endpoint_runtime_manifest(
        generated_at="now",
        selected_runtime={"runtime": "team", "endpoint_env": "TEAM_POLICY_ENDPOINT_URL"},
        endpoint_policy_used=True,
        fixture_policy_used=False,
        endpoint_invocation_count=4,
        endpoint_valid_action_count=4,
        rejected_policy_action_count=0,
        unitree_endpoint_policy_summary=replay_summary,
    )

    assert manifest["endpoint_policy_used"] is True
    assert manifest["fixture_policy_used"] is False
    assert manifest["unitree_endpoint_hand_policy_output_observed"] is True
    assert manifest["unitree_endpoint_hand_policy_used"] is False
    assert manifest["unitree_endpoint_provider_output_replay_used"] is True
    assert manifest["unitree_endpoint_fresh_policy_action_command_ran"] is False
    assert manifest["unitree_endpoint_action_chunk_used"] is True
    assert manifest["g1_robot_policy_selected_family"] is None
    assert manifest["openvla_selected_as_g1_robot_policy"] is False
    assert manifest["wam_rollout_selected_as_g1_robot_policy"] is False
    assert (
        manifest["claim_boundary"][
            "unitree_endpoint_provider_replay_is_not_fresh_hand_policy_inference"
        ]
        is True
    )


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
                    "controller_vx_mps": 0.2,
                    "controller_vy_mps": -0.05,
                    "yaw_rate_rad_s": -0.2,
                },
                "rejected": False,
            }
        ]
    )
    assert lane._representative_unitree_command(command_rows) == [0.18, -0.04, -0.2]
    assert command_rows[0]["raw_endpoint_command_xyz"] == [0.2, -0.05, -0.2]
    assert command_rows[0]["controller_command_xyz"] == [0.18, -0.04, -0.2]
    assert command_rows[0]["world_velocity_xy_mps"] == [0.3, 0.1]
    assert command_rows[0]["controller_velocity_xy_mps"] == [0.2, -0.05]
    assert command_rows[0]["controller_command_clamped"] is True
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
    assert replay["representative_endpoint_command_xyz"] == [0.18, -0.04, -0.2]
    assert calls[1]["command_xyz"] == [0.18, -0.04, -0.2]
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
    assert safe["controller_command_xyz"] == [0.18, -0.04, 0.2]
    assert safe["command_xyz"] == [0.18, -0.04, 0.2]
    assert safe["controller_command_clamped"] is True
    assert safe["controller_command_limits"]["max_forward_velocity_mps"] == 0.18

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

    assert lane._representative_unitree_command(command_rows) == [0.18, -0.04, 0.2]
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
    low_speed_contact, rejected = lane.normalize_policy_action(
        raw_payload={
            "action": {
                "action_type": "manipulation_contact",
                "waypoint": [0.3, 0.0],
                "approach_speed_mps": 0.1,
            }
        },
        observation=observation,
        source="unit",
    )
    assert rejected is None
    assert low_speed_contact["vx_mps"] == pytest.approx(0.1)
    assert abs(low_speed_contact["vy_mps"]) < 1e-9
    assert low_speed_contact["velocity_frame"] == "world_xy"
    assert low_speed_contact["controller_vx_mps"] == pytest.approx(0.1)
    assert abs(low_speed_contact["controller_vy_mps"]) < 1e-9

    yawed_world_velocity, rejected = lane.normalize_policy_action(
        raw_payload={
            "action": {
                "action_type": "base_velocity",
                "velocity_frame": "world_xy",
                "linear_velocity_mps": 0.1,
                "lateral_velocity_mps": 0.0,
            }
        },
        observation={
            **observation,
            "base_pose": {"position": [0.0, 0.0, 0.79], "yaw_rad": math.pi / 2},
        },
        source="unit",
    )
    assert rejected is None
    assert yawed_world_velocity["vx_mps"] == pytest.approx(0.1)
    assert yawed_world_velocity["velocity_frame"] == "world_xy"
    assert yawed_world_velocity["controller_vx_mps"] == pytest.approx(0.0, abs=1e-6)
    assert yawed_world_velocity["controller_vy_mps"] == pytest.approx(-0.1)

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


def test_policy_visual_observation_depth_pass_writes_npy_and_restores_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    schema = lane._declared_policy_observation_schema_for_wam_loop(None)
    assert schema["supports_depth"] is True
    assert "depth" in schema["modalities"]
    assert "depth_frame_path" in schema["fields"]
    assert schema["claim_boundary"]["mujoco_render_pass_depth_co_registered_with_rgb"] is True

    monkeypatch.setattr(lane, "_has_fixed_camera", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(lane, "_camera_for_render", lambda *_args, **_kwargs: "fake-camera")
    monkeypatch.setattr(
        lane,
        "_build_unitree_g1_sonic_state_from_mujoco",
        lambda **_kwargs: (
            {"joint_positions": [0.0]},
            {"complete": True},
        ),
    )

    class FakeImageModule:
        @staticmethod
        def fromarray(_frame: object) -> "FakeImageModule":
            return FakeImageModule()

        def save(self, path: Path, **_kwargs: object) -> None:
            Path(path).write_bytes(b"jpg")

    class DepthRenderer:
        def __init__(self) -> None:
            self.depth_mode = False
            self.toggles: list[bool] = []
            self.update_calls: list[tuple[bool, object]] = []

        def enable_depth_rendering(self, _model: object, flag: bool) -> None:
            self.depth_mode = bool(flag)
            self.toggles.append(bool(flag))

        def update_scene(self, _data: object, *, camera: object) -> None:
            self.update_calls.append((self.depth_mode, camera))

        def render(self):
            if self.depth_mode:
                return np.array([[1.0, 1.5], [2.0, 3.0]], dtype=np.float32)
            return np.zeros((2, 2, 3), dtype=np.uint8)

    renderer = DepthRenderer()
    row = lane._capture_policy_visual_observation(
        job_dir=tmp_path,
        renderer=renderer,
        image_module=FakeImageModule,
        mujoco_module=SimpleNamespace(),
        model=object(),
        data=SimpleNamespace(),
        run={
            "episode_id": "episode_0001",
            "scenario_eval_run_id": "scenario-run-1",
            "task_id": "task-1",
            "spawn_id": "spawn-1",
        },
        step=3,
        camera_id="head_pov",
        root_position=(0.0, 0.0, 0.79),
        yaw=0.0,
        root_qpos=0,
    )

    assert row["available"] is True
    assert row["depth_available"] is True
    assert row["depth_is_render_pass"] is True
    assert row["depth_encoding"] == "npy_float32_meters"
    assert row["depth_frame_path"] is not None
    depth_path = Path(row["depth_frame_path"])
    assert depth_path.exists()
    depth = np.load(depth_path)
    assert depth.shape == (row["image_height"], row["image_width"])
    assert row["depth_max_m"] > row["depth_min_m"]
    assert renderer.update_calls == [(False, "fake-camera"), (True, "fake-camera")]
    assert renderer.toggles[-1] is False
    assert (
        row["claim_boundary"][
            "mujoco_render_pass_depth_is_simulator_geometry_not_physical_sensor"
        ]
        is True
    )

    class RgbOnlyRenderer:
        def update_scene(self, _data: object, *, camera: object) -> None:
            return None

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    no_depth_row = lane._capture_policy_visual_observation(
        job_dir=tmp_path,
        renderer=RgbOnlyRenderer(),
        image_module=FakeImageModule,
        mujoco_module=SimpleNamespace(),
        model=object(),
        data=SimpleNamespace(),
        run={
            "episode_id": "episode_0002",
            "scenario_eval_run_id": "scenario-run-2",
            "task_id": "task-2",
            "spawn_id": "spawn-2",
        },
        step=4,
        camera_id="head_pov",
        root_position=(0.0, 0.0, 0.79),
        yaw=0.0,
        root_qpos=0,
    )

    assert no_depth_row["available"] is True
    assert no_depth_row["depth_available"] is False
    assert no_depth_row["depth_frame_path"] is None
    assert "policy_observation_depth_pass_unavailable" in no_depth_row["blockers"]


def test_mujoco_segmentation_render_pass_maps_geom_ids_to_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    monkeypatch.setattr(lane, "_has_fixed_camera", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(lane, "_camera_for_render", lambda *_args, **_kwargs: "fake-camera")
    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_GEOM=1, mjOBJ_BODY=2),
        mj_id2name=lambda _model, obj, idx: {
            (1, 0): "floor_geom",
            (1, 1): "target_geom",
            (2, 0): "floor_body",
            (2, 1): "target_body",
        }.get((obj, idx)),
    )
    model = SimpleNamespace(ngeom=2, geom_bodyid=[0, 1])
    contact_metadata = lane._build_contact_metadata(model, fake_mujoco)

    class FakeImageModule:
        @staticmethod
        def fromarray(_frame: object) -> "FakeImageModule":
            return FakeImageModule()

        def save(self, path: Path, **_kwargs: object) -> None:
            Path(path).write_bytes(b"mask")

    class SegFakeRenderer:
        def __init__(self) -> None:
            self.segmentation_enabled = False
            self.disable_called = False
            self.update_calls: list[tuple[bool, object]] = []

        def enable_segmentation_rendering(self) -> None:
            self.segmentation_enabled = True

        def disable_segmentation_rendering(self) -> None:
            self.disable_called = True
            self.segmentation_enabled = False

        def update_scene(self, _data: object, *, camera: object) -> None:
            self.update_calls.append((self.segmentation_enabled, camera))

        def render(self):
            assert self.segmentation_enabled
            geom_type = fake_mujoco.mjtObj.mjOBJ_GEOM
            body_type = fake_mujoco.mjtObj.mjOBJ_BODY
            return np.array(
                [
                    [[0, geom_type], [1, geom_type], [1, geom_type]],
                    [[1, geom_type], [0, body_type], [-1, geom_type]],
                ],
                dtype=np.int32,
            )

    renderer = SegFakeRenderer()
    row = lane._capture_segmentation_observation(
        job_dir=tmp_path,
        renderer=renderer,
        image_module=FakeImageModule,
        mujoco_module=fake_mujoco,
        model=model,
        data=SimpleNamespace(),
        run={
            "episode_id": "episode_0001",
            "scenario_eval_run_id": "scenario-run-1",
            "task_id": "task-1",
            "spawn_id": "spawn-1",
        },
        step=7,
        camera_id="head_pov",
        root_position=(0.0, 0.0, 0.79),
        yaw=0.0,
        contact_metadata=contact_metadata,
    )

    assert row["available"] is True
    assert renderer.disable_called is True
    assert renderer.segmentation_enabled is False
    assert renderer.update_calls == [(True, "fake-camera")]
    assert row["instance_count"] == 2
    by_geom = {instance["geom_id"]: instance for instance in row["instances"]}
    assert by_geom[1]["geom_name"] == "target_geom"
    assert by_geom[1]["body_name"] == "target_body"
    assert by_geom[1]["pixel_count"] == 3
    assert by_geom[0]["pixel_count"] == 1
    assert row["segmentation_mask_path"] is not None
    assert Path(row["segmentation_mask_path"]).is_file()
    assert (
        row["claim_boundary"]["mujoco_segmentation_is_diagnostic_not_default_policy_input"]
        is True
    )
    assert row["claim_boundary"]["segmentation_is_mujoco_evidence_not_isaac_evidence"] is True
    assert lane._declared_policy_observation_schema_for_wam_loop(None)["supports_masks"] is False

    unsupported = lane._capture_segmentation_observation(
        job_dir=tmp_path,
        renderer=object(),
        image_module=FakeImageModule,
        mujoco_module=fake_mujoco,
        model=model,
        data=SimpleNamespace(),
        run={"episode_id": "episode_0002"},
        step=8,
        camera_id="head_pov",
        root_position=(0.0, 0.0, 0.79),
        yaw=0.0,
        contact_metadata=contact_metadata,
    )
    assert unsupported["available"] is False
    assert "policy_segmentation_unsupported_renderer" in unsupported["blockers"]


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
    pose_joint_names = list(lane.EGOCENTRIC_UPPER_BODY_OBSERVATION_POSE)
    pose_model = SimpleNamespace(
        jnt_qposadr=list(range(len(pose_joint_names))),
        jnt_dofadr=list(range(len(pose_joint_names))),
    )
    pose_data = SimpleNamespace(
        qpos=[0.0] * len(pose_joint_names),
        qvel=[1.0] * len(pose_joint_names),
    )
    pose_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_JOINT=99),
        mj_name2id=lambda _model, _obj, name: pose_joint_names.index(name)
        if name in pose_joint_names
        else -1,
    )
    pose_manifest = lane._apply_egocentric_upper_body_observation_pose(
        model=pose_model,
        mujoco_module=pose_mujoco,
        data=pose_data,
        generated_at="now",
    )
    assert pose_manifest["status"] == "completed"
    assert pose_manifest["hand_end_effector_policy_used"] is False
    assert pose_manifest["claim_boundary"][
        "upper_body_pose_is_support_framing_not_hand_policy_execution"
    ] is True
    assert pose_data.qpos[0] == pytest.approx(
        lane.EGOCENTRIC_UPPER_BODY_OBSERVATION_POSE[pose_joint_names[0]]
    )
    assert pose_data.qvel[0] == 0.0

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
    assert lane._camera_for(fake_mujoco, "torso_pov", [0, 0, 1], 0).distance == pytest.approx(1.05)
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


def test_g1_projected_skeleton_trace_row_uses_mujoco_body_projection() -> None:
    class FakeModel:
        cam_fovy = [75.0]

    class FakeData:
        time = 0.25
        xpos = [
            [0.0, 0.1, 1.0],
            [0.0, 0.05, 0.9],
            [0.18, 0.03, 0.78],
            [0.24, 0.03, 0.76],
            [0.0, -0.1, 1.0],
            [0.0, -0.05, 0.9],
            [0.18, -0.03, 0.78],
            [0.24, -0.03, 0.76],
        ]
        xmat = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(8)]
        cam_xpos = [[0.0, 0.0, 1.1]]
        cam_xmat = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]

    ids = {
        "blueprint_g1_head_pov": 0,
        "left_shoulder_pitch_link": 0,
        "left_elbow_link": 1,
        "left_wrist_yaw_link": 2,
        "right_shoulder_pitch_link": 4,
        "right_elbow_link": 5,
        "right_wrist_yaw_link": 6,
    }
    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_BODY=1, mjOBJ_CAMERA=2),
        mj_name2id=lambda _model, _obj, name: ids.get(name, -1),
    )

    row = lane._build_g1_projected_skeleton_trace_row(
        mujoco_module=fake_mujoco,
        model=FakeModel(),
        data=FakeData(),
        run={
            "episode_id": "episode_0001",
            "scenario_eval_run_id": "run",
            "task_id": "inspect",
            "spawn_id": "doorway",
        },
        step=4,
        visual_observation={
            "available": True,
            "camera_id": "head_pov",
            "camera_frame_path": "/tmp/frame.jpg",
            "image_width": 640,
            "image_height": 480,
        },
    )

    assert row["status"] == "completed"
    assert row["episode_id"] == "episode_0001"
    assert row["camera_id"] == "head_pov"
    assert row["available_landmark_count"] == 8
    assert row["projected_landmark_count"] == 8
    assert row["claim_boundary"]["uses_unitree_g1_mujoco_body_transforms"] is True
    assert row["claim_boundary"]["not_hand_drawn_stick_figure"] is True
    assert row["claim_boundary"]["not_physical_robot_sensor_proof"] is True
    left_hand = next(
        landmark for landmark in row["landmarks"] if landmark["landmark_id"] == "left_hand"
    )
    assert left_hand["image_projection"]["available"] is True
    assert left_hand["image_projection"]["inside_image"] is True

    manifest = lane._g1_projected_skeleton_manifest(
        generated_at="now", rows=[row], output_path=Path("trace.jsonl")
    )
    assert manifest["status"] == "completed"
    assert manifest["projectable_row_count"] == 1
    assert manifest["claim_boundary"][
        "simulated_g1_arm_hand_state_available_for_wam_conditioning"
    ] is True
    assert manifest["claim_boundary"]["not_physical_robot_sensor_proof"] is True


def test_wam_vla_lane_runs_with_fake_mujoco_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in (
        "BLUEPRINT_UNITREE_RL_GYM_ROOT",
        "BLUEPRINT_UNITREE_G1_POLICY_ROOT",
        "BLUEPRINT_UNITREE_G1_POLICY_SOURCE_ROOT",
        "BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_ROOT",
        "BLUEPRINT_UNITREE_LEROBOT_PYTHON",
        "BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

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
        nq = 20
        nv = 18
        nu = 1
        opt = SimpleNamespace(timestep=0.05)
        actuator_trnid = [[0]]
        jnt_qposadr = [0, 7, 14, 15, 16, 17, 18, 19]
        jnt_dofadr = [0, 6, 12, 13, 14, 15, 16, 17]
        geom_bodyid = [0, 1]

        def __init__(self) -> None:
            self.qpos0 = FakeVec(
                [
                    0.0,
                    0.0,
                    0.79,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.36,
                    -0.65,
                    0.27,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                + [0.0] * 6
            )
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
        ids = {
            "floating_base_joint": 0,
            "blueprint_light_object_freejoint": 1,
            "left_shoulder_pitch_joint": 2,
            "right_shoulder_pitch_joint": 3,
            "left_shoulder_roll_joint": 4,
            "right_shoulder_roll_joint": 5,
            "left_elbow_joint": 6,
            "right_elbow_joint": 7,
            "stand": 0,
        }
        if name in ids:
            return ids[name]
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
    (g1_root / "g1_with_hands.xml").write_text("<mujoco/>", encoding="utf-8")
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
    assert summary["unitree_lower_body_locomotion_policy_used"] is False
    assert summary["unitree_hand_manipulation_policy_used"] is False
    assert summary["g1_robot_policy_selected_family"] is None
    assert summary["unitree_hand_manipulation_policy_scope"] is None
    assert summary["openvla_selected_as_g1_robot_policy"] is False
    assert summary["wam_rollout_selected_as_g1_robot_policy"] is False
    assert summary["unitree_endpoint_hand_policy_output_observed"] is False
    assert summary["unitree_endpoint_hand_policy_used"] is False
    assert summary["unitree_endpoint_provider_output_replay_used"] is False
    assert summary["unitree_endpoint_fresh_policy_action_command_ran"] is False
    assert summary["unitree_lerobot_or_isaaclab_manipulation_policy_used"] is False
    assert summary["manipulation_policy_kind"] == "contact_trace_proxy_only"
    assert (tmp_path / "job" / "normalized_attempt_trace.json").is_file()
    assert (tmp_path / "job" / "policy_model_candidate_matrix.json").is_file()
    assert (tmp_path / "job" / "policy_model_truth_boundary.json").is_file()
    assert (tmp_path / "job" / "policy_endpoint_runtime_manifest.json").is_file()
    assert (tmp_path / "job" / "policy_endpoint_boundary_manifest.json").is_file()
    assert (tmp_path / "job" / "policy_endpoint_invocation_trace.jsonl").is_file()
    assert (tmp_path / "job" / "realistic_navigation_policy_discovery.json").is_file()
    assert (tmp_path / "job" / "unitree_g1_manipulation_policy_discovery.json").is_file()
    runtime_manifest = json.loads(
        (tmp_path / "job" / "policy_endpoint_runtime_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime_manifest["unitree_endpoint_hand_policy_output_observed"] is False
    assert runtime_manifest["unitree_endpoint_hand_policy_used"] is False
    assert runtime_manifest["unitree_endpoint_provider_output_replay_used"] is False
    assert runtime_manifest["openvla_selected_as_g1_robot_policy"] is False
    assert runtime_manifest["wam_rollout_selected_as_g1_robot_policy"] is False
    endpoint_boundary = json.loads(
        (tmp_path / "job" / "policy_endpoint_boundary_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert endpoint_boundary["fixture_policy_used"] is True
    assert endpoint_boundary["robot_policy_execution_proven"] is False
    assert endpoint_boundary["claim_boundary"]["endpoint_setup_is_not_safety_validation"] is True
    assert (
        endpoint_boundary["claim_boundary"]["endpoint_setup_is_not_deployment_approval"]
        is True
    )
    asset_manifest = json.loads(
        (tmp_path / "job" / "unitree_g1_mujoco_asset_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert asset_manifest["resolved_g1_xml"].endswith("g1_with_hands.xml")
    assert asset_manifest["hands_capable_g1_mjcf_selected"] is True
    assert (
        asset_manifest["g1_mjcf_selection"]["claim_boundary"][
            "hands_capable_mjcf_does_not_prove_dexterous_hand_policy_execution"
        ]
        is True
    )
    assert (tmp_path / "job" / "unitree_endpoint_action_command_stream.json").is_file()
    assert (tmp_path / "job" / "unitree_controller_bridge_manifest.json").is_file()
    assert (tmp_path / "job" / "robot_policy_wam_loop_manifest.json").is_file()
    assert (tmp_path / "job" / "policy_action_model_command_output.json").is_file()
    assert (tmp_path / "job" / "final_success_judge.json").is_file()
    assert (tmp_path / "job" / "claim_boundary.json").is_file()
    assert (tmp_path / "job" / "g1_projected_skeleton_trace.jsonl").is_file()
    assert (tmp_path / "job" / "g1_projected_skeleton_manifest.json").is_file()
    projected_manifest = json.loads(
        (tmp_path / "job" / "g1_projected_skeleton_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert projected_manifest["claim_boundary"][
        "derived_from_unitree_g1_mujoco_body_transforms"
    ] is True
    assert projected_manifest["claim_boundary"]["not_physical_robot_sensor_proof"] is True
    assert summary["artifact_paths"]["g1_projected_skeleton_trace_jsonl"].endswith(
        "g1_projected_skeleton_trace.jsonl"
    )
    assert summary["unitree_g1_manipulation_policy_discovery"] == str(
        tmp_path / "job" / "unitree_g1_manipulation_policy_discovery.json"
    )
    loop_manifest = json.loads(
        (tmp_path / "job" / "robot_policy_wam_loop_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert loop_manifest["actual_loop_mode"] == (
        "mujoco_policy_endpoint_execution_with_offline_wam_trace_package"
    )
    assert loop_manifest["robot_loaded_in_scene"] is True
    assert loop_manifest["wam_evaluator_in_control_loop"] is False
    assert loop_manifest["policy_observes_wam_generated_next_observation"] is False
    assert summary["artifact_paths"]["robot_policy_wam_loop_manifest"].endswith(
        "robot_policy_wam_loop_manifest.json"
    )
    assert summary["artifact_paths"]["policy_action_model_command_output"].endswith(
        "policy_action_model_command_output.json"
    )
    assert summary["artifact_paths"]["final_success_judge"].endswith("final_success_judge.json")
    final_success_judge = json.loads(
        (tmp_path / "job" / "final_success_judge.json").read_text(encoding="utf-8")
    )
    assert final_success_judge["final_question"] == (
        "Did the object/tote end up correctly placed?"
    )
    assert final_success_judge["answer"] == "not_proven"
    assert final_success_judge["object_or_tote_correctly_placed"] is False
    assert final_success_judge["claim_boundary"]["generated_world_rank_fidelity_result_proven"] is False
    claim_boundary = json.loads(
        (tmp_path / "job" / "claim_boundary.json").read_text(encoding="utf-8")
    )
    assert claim_boundary["groot_n17_sonic_is_candidate_not_proven_unless_action_command_runs"]
    assert claim_boundary["generated_world_rank_fidelity_result_proven"] is False
    controller_truth = json.loads(
        (tmp_path / "job" / "controller_truth_boundary.json").read_text(encoding="utf-8")
    )
    assert controller_truth["unitree_lower_body_locomotion_policy_used"] is False
    assert controller_truth["unitree_hand_manipulation_policy_used"] is False
    assert controller_truth["unitree_lerobot_or_isaaclab_manipulation_policy_used"] is False
    assert controller_truth["manipulation_policy_kind"] == "contact_trace_proxy_only"
    manipulation_discovery = json.loads(
        (tmp_path / "job" / "unitree_g1_manipulation_policy_discovery.json").read_text(
            encoding="utf-8"
        )
    )
    assert manipulation_discovery["status"] == "blocked_missing_hand_policy_runtime"

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
    endpoint_boundary = json.loads(
        (tmp_path / "endpoint-job" / "policy_endpoint_boundary_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert endpoint_boundary["endpoint_policy_used"] is True
    assert endpoint_boundary["robot_policy_execution_proven"] is False
    assert (
        endpoint_boundary["status"]
        == "endpoint_integration_configured_not_robot_policy_execution"
    )
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
            self.segmentation_enabled = False

        def enable_segmentation_rendering(self) -> None:
            self.segmentation_enabled = True

        def disable_segmentation_rendering(self) -> None:
            self.segmentation_enabled = False

        def update_scene(self, *_args: object, **_kwargs: object) -> None:
            return None

        def render(self) -> list[list[list[int]]]:
            if self.segmentation_enabled:
                return [[[1, fake_mujoco.mjtObj.mjOBJ_GEOM]]]
            return [[[0, 0, 0]]]

        def close(self) -> None:
            self.closed = True

    class FakeImage:
        @staticmethod
        def fromarray(_frame: object) -> "FakeImage":
            return FakeImage()

        def save(self, path: Path, **_kwargs: object) -> None:
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
    assert render_summary["mujoco_segmentation_diagnostic_available"] is True
    assert render_summary["segmentation_backend"] == "mujoco_renderer_native"
    assert render_summary["policy_segmentation_observation_available_count"] >= 1
    segmentation_manifest = json.loads(
        (tmp_path / "render-job" / "policy_segmentation_observations.json").read_text(
            encoding="utf-8"
        )
    )
    assert segmentation_manifest["status"] == "completed"
    assert segmentation_manifest["available_observation_count"] >= 1
    assert (
        segmentation_manifest["claim_boundary"][
            "mujoco_segmentation_is_diagnostic_not_default_policy_input"
        ]
        is True
    )
    visual_trace_rows = [
        json.loads(line)
        for line in (tmp_path / "render-job" / "policy_visual_observation_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert visual_trace_rows
    assert visual_trace_rows[0]["segmentation_observation"]["available"] is True
    assert (
        render_summary["artifact_paths"]["policy_segmentation_observation_manifest"].endswith(
            "policy_segmentation_observations.json"
        )
    )
    assert render_summary["artifact_paths"]["video_generation_status"].endswith("video_generation_status.json")
    video_status = json.loads(
        (tmp_path / "render-job" / "video_generation_status.json").read_text(encoding="utf-8")
    )
    assert "every_sim_step_captured_for_selected_review_videos" in video_status["render_contract"]
    assert video_status["render_contract"]["review_video_sampling"][
        "sampling_mode"
    ] == "fixed_sample_count_review"
    assert (
        video_status["render_contract"]["egocentric_upper_body_observation_pose"][
            "hand_end_effector_policy_used"
        ]
        is False
    )
    assert (
        video_status["render_contract"]["egocentric_upper_body_observation_pose"][
            "claim_boundary"
        ]["upper_body_pose_is_support_framing_not_hand_policy_execution"]
        is True
    )
    assert video_status["render_contract"]["nominal_realtime_review_mp4"] is False
    assert "playback_timing" in video_status["videos"][0]
    assert video_status["videos"][0]["video_fps"] == lane.DEFAULT_REVIEW_VIDEO_FPS
    assert video_status["videos"][0]["playback_timing"]["fps_zero_used_for_sim_time_playback"] is False
    assert video_status["videos"][0]["review_video_sampling_mode"] == "fixed_sample_count_review"
    assert video_status["videos"][0]["nominal_realtime_review_mp4"] is False
    assert video_status["generated_video_review_validation_count"] == len(
        lane.DEFAULT_VIDEO_CAMERAS
    )
    assert video_status["generated_videos_decode_valid_for_review"] is False
    assert video_status["generated_rollout_visual_smoke_status"] == "blocked_visual_probe_failed"
    assert video_status["generated_rollout_visually_useful_for_success_review"] is False
    assert video_status["videos"][0]["decode_valid_for_review"] is False
    assert video_status["videos"][0]["generated_video_review_validation"]["blockers"] == [
        "generated_video_missing"
    ]
    assert (tmp_path / "render-job" / "generated_rollout_visual_smoke.json").is_file()
    unitree_review = json.loads(
        (tmp_path / "render-job" / "unitree_generated_rollout_review_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert unitree_review["status"] == "not_applicable_no_unitree_policy_rollout"
    video_review_status = json.loads(
        (tmp_path / "render-job" / "video_review_status.json").read_text(encoding="utf-8")
    )
    assert video_review_status["status"] == "not_applicable_no_unitree_policy_rollout"
    assert video_review_status["claim_boundary"]["video_review_is_not_task_success_proof"] is True
    selection_manifest = json.loads(
        (tmp_path / "render-job" / "review_video_selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection_manifest["selected_review_video_count"] >= 1
    assert selection_manifest["selection_policy"]["default_camera"] == "head_pov"
    assert (
        selection_manifest["selection_policy"]["egocentric_upper_body_observation_pose"][
            "hands_or_end_effectors_expected_in_egocentric_torso_view"
        ]
        is True
    )
    assert (
        selection_manifest["selection_policy"][
            "third_person_overview_is_diagnostic_not_policy_observation"
        ]
        is True
    )
    assert selection_manifest["generated_rollout_visually_useful_for_success_review"] is False
    selected_video = selection_manifest["selected_review_videos"][0]
    assert selected_video["scenario_eval_run_id"] == "render-job__doorway__inspect_target"
    assert selected_video["task_id"] == "inspect_target"
    assert selected_video["spawn_id"] == "doorway"
    assert selected_video["camera"] == "head_pov"
    assert selected_video["first_person_policy_observation_candidate"] is True
    assert selected_video["hands_or_end_effectors_expected_due_to_observation_pose"] is True
    assert selected_video["decode_valid_for_review"] is False
    assert selected_video["review_video_sampling_mode"] == "fixed_sample_count_review"
    assert selected_video["nominal_realtime_review_mp4"] is False

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
    assert cli_calls[-1]["wam_loop_step_count"] == lane.DEFAULT_WAM_LOOP_STEP_COUNT
    assert (
        lane.main(
            [
                "--job-dir",
                str(tmp_path / "cli-groot-policy"),
                "--skip-render",
                "--max-tasks",
                "1",
                "--policy-lane",
                "unitree_groot_n17_sonic_policy",
            ]
        )
        == 0
    )
    assert cli_calls[-1]["policy_lane"] == "unitree_groot_n17_sonic_policy"
    assert (
        lane.main(
            [
                "--job-dir",
                str(tmp_path / "cli-long-wam-loop"),
                "--skip-render",
                "--max-tasks",
                "1",
                "--wam-loop-step-count",
                "20",
            ]
        )
        == 0
    )
    assert cli_calls[-1]["wam_loop_step_count"] == 20
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
