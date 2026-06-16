from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from blueprint_pipeline.manipulation_task_stack import (
    build_manipulation_object_contract,
    build_manipulation_task_stack,
)
from blueprint_pipeline.manipulation_physics_simulator_command import (
    run_mujoco_manipulation_physics,
)
from blueprint_pipeline.lucky_g1_reference_adapter import run_lucky_g1_reference_adapter
from blueprint_pipeline.robot_eval_execution import build_policy_execution_bundle


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_lucky_g1_reference_adapter_fails_closed_without_assets(tmp_path: Path) -> None:
    result = run_lucky_g1_reference_adapter(
        capture_root=tmp_path,
        lucky_root=tmp_path / "missing-lucky",
        fetch_if_missing=False,
    )

    assert result["status"] == "blocked"
    assert result["official_lucky_walker_reacher_policy_assets_executed"] is False
    assert result["claim_boundary"]["official_lucky_pick_place_physics_validated"] is False
    assert Path(result["output_path"]).is_file()


def test_tote_template_builds_default_policy_eval_stack(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")

    result = build_manipulation_task_stack(
        capture_root=tmp_path,
        object_pose=[2.0, 4.0, 0.16, 0.0],
        return_pose=[0.0, 0.0, 0.793, 0.0],
    )

    assert result["status"] == "complete"
    assert result["default_policy_available"] is True
    assert result["simulator_physics_execution_proven"] is True
    assert result["grasp_physics_validated"] is True
    assert result["carry_physics_validated"] is True
    assert result["manipulation_capable_g1_model_loaded"] is True
    assert result["controller_drove_actuators"] is True
    assert result["g1_reference_manipulation_physics_executed"] is True
    assert result["official_lucky_pick_place_physics_validated"] is False
    contracts = _read(result["artifacts"]["manipulation_object_contracts"])
    contract = contracts["contracts"][0]
    assert contract["object_class"] == "tote"
    assert contract["contract_ready_for_scored_manipulation"] is True
    assert Path(contract["asset"]["uri"]).name == "mujoco_tote_asset.xml"
    assert Path(contract["asset"]["uri"]).is_file()
    trace = _read(result["artifacts"]["default_manipulation_policy_trace"])
    assert trace["status"] == "completed_reference_trace"
    assert trace["robot_team_policy_execution_proven"] is False
    physics = _read(result["artifacts"]["manipulation_physics_output"])
    assert physics["status"] == "complete"
    assert physics["simulator_physics_execution_proven"] is True
    assert physics["grasp_physics_validated"] is True
    assert physics["manipulation_capable_g1_model_loaded"] is True
    assert physics["g1_arm_gripper_actuators_exposed"] is True
    assert physics["g1_head_camera_available"] is True
    assert physics["g1_wrist_camera_available"] is True
    assert physics["controller_drove_actuators"] is True
    assert physics["g1_reference_manipulation_physics_executed"] is True
    assert physics["release_contact_geoms_disabled_after_open"] is True
    assert physics["post_release_xy_angular_velocity_stabilized"] is True
    assert physics["claim_boundary"]["contact_only_dexterous_hand_grasp_validated"] is False
    assert Path(physics["artifacts"]["mujoco_tote_object_asset"]).is_file()
    assert Path(physics["artifacts"]["mujoco_tote_object_asset_manifest"]).is_file()
    assert Path(physics["artifacts"]["mujoco_tote_visual_mesh"]).is_file()
    assert Path(physics["artifacts"]["mujoco_g1_manipulation_model_manifest"]).is_file()
    assert Path(physics["artifacts"]["manipulation_overview_video"]).is_file()
    assert Path(physics["artifacts"]["manipulation_video_manifest"]).is_file()
    model_manifest = _read(physics["artifacts"]["mujoco_g1_manipulation_model_manifest"])
    assert model_manifest["manipulation_capable_g1_model_loaded"] is True
    assert model_manifest["controlled_joint_count"] >= 9
    assert {camera["name"] for camera in model_manifest["cameras"]} == {
        "g1_head_camera",
        "g1_right_wrist_camera",
    }
    tiers = _read(result["artifacts"]["manipulation_policy_tier_matrix"])
    assert [tier["tier_id"] for tier in tiers["tiers"]] == [
        "default_phase_policy",
        "lucky_g1_reference_or_blueprint_physics",
        "team_policy_endpoint_or_vla_adapter",
    ]
    assert tiers["tiers"][1]["simulator_physics_execution_proven"] is True
    assert tiers["tiers"][1]["controller_drove_actuators"] is True
    assert tiers["tiers"][1]["g1_reference_manipulation_physics_executed"] is True
    assert [tier["ready"] for tier in tiers["tiers"]] == [True, True, False]
    assert tiers["tiers"][1]["official_lucky_pick_place_physics_validated"] is False
    assert [phase["phase_id"] for phase in trace["phase_trace"]] == [
        "navigate_to_object",
        "pregrasp_stance",
        "reach",
        "close_grip",
        "lift",
        "verify_grasp",
        "carry_to_return_pose",
        "place",
        "release",
        "verify_placement",
    ]


def test_unknown_object_without_contract_fails_closed(tmp_path: Path) -> None:
    result = build_manipulation_task_stack(
        capture_root=tmp_path,
        object_class="random_mesh",
        object_asset_path=str(tmp_path / "random.usd"),
        allow_template_inference=False,
    )

    assert result["status"] == "blocked"
    assert "manipulation_affordances_missing" in result["blockers"]
    assert "manipulation_success_thresholds_missing" in result["blockers"]
    assert "manipulation_physics_simulator_not_complete" in result["blockers"]
    report = _read(result["artifacts"]["manipulation_eval_report"])
    assert report["contract_ready"] is False
    assert report["claim_boundary"]["grasp_physics_validated"] is False


def test_manipulation_physics_command_proves_grasp_carry_place(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    result = run_mujoco_manipulation_physics(
        capture_root=tmp_path,
        output_dir=tmp_path / "physics",
        object_pose=[0.25, 1.2, 0.16, 0.0],
        return_pose=[0.25, 0.1, 0.793, 0.0],
    )

    assert result["status"] == "complete"
    assert result["simulator_physics_execution_proven"] is True
    assert result["grasp_physics_validated"] is True
    assert result["carry_physics_validated"] is True
    assert result["placement_physics_validated"] is True
    assert result["manipulation_capable_g1_model_loaded"] is True
    assert result["controller_drove_actuators"] is True
    assert result["g1_reference_manipulation_physics_executed"] is True
    assert result["release_contact_geoms_disabled_after_open"] is True
    assert result["post_release_xy_angular_velocity_stabilized"] is True
    assert result["contact_only_dexterous_hand_grasp_validated"] is False
    assert Path(result["artifacts"]["manipulation_physics_trace"]).is_file()
    assert Path(result["artifacts"]["manipulation_contact_manifest"]).is_file()
    assert Path(result["artifacts"]["manipulation_overview_video"]).is_file()
    assert Path(result["artifacts"]["mujoco_tote_object_asset"]).is_file()
    assert Path(result["artifacts"]["mujoco_tote_visual_mesh"]).is_file()
    assert Path(result["artifacts"]["mujoco_g1_manipulation_model_manifest"]).is_file()
    first_trace = json.loads(
        Path(result["artifacts"]["manipulation_physics_trace"])
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert "end_effector_pose_xyz" in first_trace
    assert "controller_target" in first_trace
    assert "actuator_controls" in first_trace
    assert "joint_state" in first_trace
    assert "base_pose_xy_yaw" in first_trace
    assert "grip_contact_force_proxy_n" in first_trace
    assert "drop_event" in first_trace
    assert "tilt_event" in first_trace
    assert "slip_event" in first_trace


def test_team_endpoint_adapter_and_inline_contract_are_preserved(tmp_path: Path) -> None:
    contract = build_manipulation_object_contract(
        object_id="custom_tote",
        object_asset_path="simready://custom_tote.usd",
        object_pose=[1.0, 2.0, 0.16, 0.0],
    )
    job_request = {
        "schema_version": "robot_eval_job_request.v1",
        "manipulation_task": {
            "task_id": "carry_custom_tote",
            "object_id": "custom_tote",
            "object_contract": contract,
            "team_policy_endpoint": "https://example.test/policy",
            "instruction": "Pick the custom tote and return it to start.",
        },
    }
    request_path = tmp_path / "job_request.json"
    request_path.write_text(json.dumps(job_request), encoding="utf-8")

    result = build_manipulation_task_stack(
        capture_root=tmp_path,
        job_request_path=request_path,
        run_physics_sim=False,
    )

    assert result["team_policy_endpoint_configured"] is True
    adapter = _read(result["artifacts"]["manipulation_policy_adapter_contract"])
    endpoint_mode = next(
        mode for mode in adapter["policy_submission_modes"] if mode["mode"] == "policy_api_endpoint"
    )
    assert endpoint_mode["enabled"] is True
    assert endpoint_mode["endpoint_url"] == "https://example.test/policy"
    task = _read(result["artifacts"]["manipulation_task_request"])
    assert task["task_id"] == "carry_custom_tote"
    assert task["object_id"] == "custom_tote"


def test_default_manipulation_policy_runs_through_policy_execution_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    result = build_policy_execution_bundle(
        capture_root=tmp_path,
        job_dir=job_dir,
        job_request={
            "schema_version": "robot_eval_job_request.v1",
            "default_test_policy": {
                "policy_kind": "mobile_manipulation_pick_carry_place",
                "task_id": "carry_tote_home",
                "object_id": "simready_tote_001",
                "object_class": "tote",
            },
        },
        observation_manifest={
            "schema_version": "robot_pov_observation_manifest.v1",
            "observations": [
                {
                    "observation_id": "obs-1",
                    "scenario_id": "scenario-tote",
                    "scenario_eval_run_id": "run-tote-1",
                    "task_id": "carry_tote_home",
                }
            ],
        },
        allow_policy_execution=True,
        generated_at="2026-06-14T00:00:00+00:00",
    )

    manifest = result["manifest"]
    trace = result["trace"]
    assert manifest["default_test_policy_execution_proven"] is True
    assert manifest["robot_team_policy_execution_proven"] is False
    high_level = manifest["modality_results"]["high_level_skill_trace"]
    assert high_level["default_test_policy"] is True
    assert high_level["robot_team_policy_execution_proven"] is False
    attempt = trace["attempts"][0]
    assert attempt["policy_kind"] == "mobile_manipulation_pick_carry_place"
    assert attempt["target"] == "simready_tote_001"
    assert attempt["metrics"]["default_manipulation_policy"] is True
    assert attempt["metrics"]["simulator_physics_execution_proven"] is False
    assert (job_dir / "policy_execution_manifest.json").is_file()
    assert (job_dir / "policy_execution_trace.json").is_file()


def test_team_manipulation_policy_endpoint_runs_through_policy_execution_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            _ = self.rfile.read(int(self.headers.get("content-length", "0")))
            payload = {
                "attempts": [
                    {
                        "attempt_id": "team-manip-1",
                        "scenario_eval_run_id": "run-tote-1",
                        "task_id": "carry_tote_home",
                        "policy_id": "team-vla-policy",
                        "policy_kind": "mobile_manipulation_pick_carry_place",
                        "status": "completed",
                        "success": True,
                        "actions": [
                            {"action": "navigate_to_object", "status": "completed"},
                            {"action": "lift", "status": "completed"},
                            {"action": "carry_to_return_pose", "status": "completed"},
                            {"action": "place", "status": "completed"},
                        ],
                        "metrics": {
                            "grasp_physics_validated": True,
                            "carry_physics_validated": True,
                        },
                    }
                ]
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/policy"
        job_dir = tmp_path / "job-endpoint"
        job_dir.mkdir()
        result = build_policy_execution_bundle(
            capture_root=tmp_path,
            job_dir=job_dir,
            job_request={
                "schema_version": "robot_eval_job_request.v1",
                "policy_package": {
                    "policy_api_endpoint": {
                        "endpoint_url": endpoint,
                        "policy_id": "team-vla-policy",
                    }
                },
            },
            observation_manifest={
                "schema_version": "robot_pov_observation_manifest.v1",
                "observations": [
                    {
                        "observation_id": "obs-1",
                        "scenario_id": "scenario-tote",
                        "scenario_eval_run_id": "run-tote-1",
                        "task_id": "carry_tote_home",
                    }
                ],
            },
            allow_policy_execution=True,
            generated_at="2026-06-14T00:00:00+00:00",
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    manifest = result["manifest"]
    assert manifest["robot_team_policy_execution_proven"] is True
    endpoint_result = manifest["modality_results"]["policy_api_endpoint"]
    assert endpoint_result["robot_team_policy_execution_proven"] is True
    attempt = result["trace"]["attempts"][0]
    assert attempt["policy_kind"] == "mobile_manipulation_pick_carry_place"
    assert attempt["metrics"]["grasp_physics_validated"] is True
