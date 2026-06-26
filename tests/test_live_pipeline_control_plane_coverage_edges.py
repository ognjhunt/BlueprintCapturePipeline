from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import blueprint_pipeline.live_pipeline_control_plane as lcp
from blueprint_pipeline.common import write_json


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    root.mkdir(parents=True)
    write_json(root / "capture_descriptor.json", {"site_submission_id": "site-1"})
    return root


def test_basic_helper_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)

    monkeypatch.setenv(lcp.CONTROL_PLANE_TIMEOUT_SECONDS_ENV, "bad")
    assert lcp._env_int(lcp.CONTROL_PLANE_TIMEOUT_SECONDS_ENV, 12) == 12
    monkeypatch.setenv(lcp.CONTROL_PLANE_TIMEOUT_SECONDS_ENV, "-1")
    assert lcp._env_int(lcp.CONTROL_PLANE_TIMEOUT_SECONDS_ENV, 12) == 12
    monkeypatch.setenv(lcp.CONTROL_PLANE_TIMEOUT_SECONDS_ENV, "7")
    assert lcp._env_int(lcp.CONTROL_PLANE_TIMEOUT_SECONDS_ENV, 12) == 7

    monkeypatch.setenv("ROBOT_EVAL_JOB_DEFAULT_SIMULATOR_COMMAND", "run explicit")
    assert lcp._mujoco_beta_simulator_command(capture_root) == "run explicit"
    monkeypatch.delenv("ROBOT_EVAL_JOB_DEFAULT_SIMULATOR_COMMAND", raising=False)
    assert lcp._mujoco_beta_simulator_command(None) == ""
    assert lcp._mujoco_beta_simulator_command(capture_root) == ""
    monkeypatch.setenv(lcp.MUJOCO_ALLOW_FETCH_G1_ASSETS_ENV, "true")
    monkeypatch.setenv(lcp.MUJOCO_BETA_SKIP_RENDER_ENV, "true")
    command = lcp._mujoco_beta_simulator_command(capture_root)
    assert "--allow-fetch-g1-assets" in command
    assert "--skip-render-frames" in command

    assert lcp._count(True) == 0
    assert lcp._count("4") == 4
    assert lcp._count("nope") == 0
    assert lcp._output_path(capture_root, None).name == "live_pipeline_control_plane_manifest.json"
    assert lcp._output_path(None, None).name == "live_pipeline_control_plane_manifest.json"
    assert lcp._agent_adapter_from_mode("fake", allow_live_operator=False).__class__.__name__ == "FakeRobotEvalJobAgentAdapter"
    assert lcp._agent_adapter_from_mode("agents-sdk", allow_live_operator=False).__class__.__name__ == "AgentsSdkRobotEvalJobAdapter"
    assert lcp._manifest_leaks_secret({"safe": "value"}, []) is False


def test_simulator_command_parsing_and_status_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lcp.ISAAC_LAB_ARENA_COMMAND_ENV, "isaac run")
    assert lcp._parse_simulator_commands(["", "mujoco=python run"]) == {
        "mujoco": "python run",
        "isaac_lab_arena": "isaac run",
    }
    with pytest.raises(ValueError):
        lcp._parse_simulator_commands(["missing-separator"])

    assert (
        lcp._overall_status(
            capture_root=Path("/tmp/capture"),
            inbox={"status": "not_configured"},
            setup_manifest={"status": "ready_for_live_external_execution"},
        )
        == "ready_for_live_external_execution"
    )
    assert (
        lcp._overall_status(
            capture_root=Path("/tmp/capture"),
            inbox={"status": "not_configured"},
            setup_manifest={"status": "local_ready_live_external_blocked"},
        )
        == "local_ready_live_external_blocked"
    )
    assert (
        lcp._overall_status(
            capture_root=Path("/tmp/capture"),
            inbox={"status": "not_configured"},
            setup_manifest={"status": "unknown"},
        )
        == "blocked"
    )
    assert lcp._setup_section_ready({"sections": {"bad": "not-a-mapping"}}, "bad") is False
    assert lcp._input_packet_status(required_inputs=[], enablement_inputs=[{"id": "gate"}]) == (
        "core_external_inputs_ready_enablement_missing"
    )
    assert lcp._input_packet_status(required_inputs=[], enablement_inputs=[]) == "all_external_inputs_configured"


def test_next_inputs_field_sources_and_policy_modality_edges() -> None:
    next_inputs = lcp._control_plane_next_inputs_needed(
        capture_root=Path("/tmp/capture"),
        job_request_inbox=Path("/tmp/inbox"),
        setup_manifest={"sections": {}},
        webapp_upstream_truth_ready=True,
        real_robot_pov_ready=True,
        live_closure_evidence_ready=True,
        deployment_outcomes_ready=True,
        deployment_prediction_match_keys_ready=False,
        deployment_owner_evidence_ready=False,
        policy_package_ready=True,
        followup_request_queues={"ready": True, "queues": ["skip", {"safe_processing_command": "process inbox"}]},
    )
    assert not any("exact prediction join keys" in item for item in next_inputs)
    assert "process inbox" in next_inputs

    owner_inputs = lcp._control_plane_next_inputs_needed(
        capture_root=Path("/tmp/capture"),
        job_request_inbox=Path("/tmp/inbox"),
        setup_manifest={"sections": {}},
        webapp_upstream_truth_ready=True,
        real_robot_pov_ready=True,
        live_closure_evidence_ready=True,
        deployment_outcomes_ready=True,
        deployment_prediction_match_keys_ready=True,
        deployment_owner_evidence_ready=False,
        policy_package_ready=True,
    )
    assert not any("owner evidence" in item for item in owner_inputs)

    payload = {"owner_system": {"request_id": "owner-request"}}
    assert lcp._field_value_from_sources(payload, "request_id", [{}]) == "owner-request"
    assert lcp._field_value_from_sources({}, "request_id", [{}]) is None

    assert lcp._policy_modality_missing_inputs("policy_api_endpoint", {"endpoint_url": "ftp://bad"}) == [
        "policy_package.policy_api_endpoint.endpoint_url"
    ]
    assert set(lcp._policy_modality_missing_inputs("docker_container", {})) == {
        "policy_package.docker_container.image_ref",
        "policy_package.docker_container.digest",
    }
    assert lcp._policy_modality_missing_inputs("recorded_action_trace", {}) == [
        "policy_package.recorded_action_trace.trace_manifest_uri",
        "policy_package.recorded_action_trace.timestamp_alignment",
    ]
    assert lcp._policy_modality_missing_inputs("high_level_skill_trace", {}) == [
        "policy_package.high_level_skill_trace.ordered_skill_sequence"
    ]
    assert lcp._policy_modality_missing_inputs("teleop_demo", {}) == [
        "policy_package.teleop_demo.demo_artifact_uri",
        "policy_package.teleop_demo.rights_privacy_attestation",
    ]
    assert lcp._policy_modality_missing_inputs("sim_controller_plugin", {}) == [
        "policy_package.sim_controller_plugin.simulator_framework",
        "policy_package.sim_controller_plugin.plugin_uri",
    ]


def test_followup_request_queue_edges(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    jobs_dir = capture_root / "pipeline" / "robot_eval_jobs"
    bad_json = jobs_dir / "job-bad-json" / "real_world_validation_followup_request_queue.json"
    bad_json.parent.mkdir(parents=True)
    bad_json.write_text("{", encoding="utf-8")
    list_queue = jobs_dir / "job-list" / "real_world_validation_followup_request_queue.json"
    list_queue.parent.mkdir(parents=True)
    list_queue.write_text("[]", encoding="utf-8")
    write_json(
        jobs_dir / "job-missing-inbox" / "real_world_validation_followup_request_queue.json",
        {
            "schema_version": lcp.REAL_WORLD_VALIDATION_FOLLOWUP_REQUEST_QUEUE_SCHEMA_VERSION,
            "status": "ready_for_inbox_processing",
            "inbox_dir": str(tmp_path / "missing-inbox"),
            "queued_request_count": 1,
            "queued_request_paths": [str(tmp_path / "missing-request.json")],
        },
    )
    write_json(
        jobs_dir / "job-empty" / "real_world_validation_followup_request_queue.json",
        {
            "schema_version": lcp.REAL_WORLD_VALIDATION_FOLLOWUP_REQUEST_QUEUE_SCHEMA_VERSION,
            "status": "ready_for_inbox_processing",
            "queued_request_count": 0,
        },
    )

    result = lcp._real_world_validation_followup_request_queues(capture_root)
    assert result["status"] == "blocked"
    assert "followup_request_queue_read_failed:JSONDecodeError" in result["blockers"]
    assert "followup_request_queue_not_json_object" in result["blockers"]
    assert "followup_request_queue_schema_mismatch" in result["blockers"]
    assert "followup_request_queue_inbox_dir_missing" in result["blockers"]
    assert "followup_request_queue_request_file_missing" in result["blockers"]
    assert "followup_request_queue_inbox_missing" in result["blockers"]
    assert "followup_request_queue_empty" in result["blockers"]

    quiet_root = tmp_path / "quiet" / "storage" / "bucket" / "scenes" / "scene" / "captures" / "capture"
    write_json(
        quiet_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-waiting"
        / "real_world_validation_followup_request_queue.json",
        {
            "schema_version": lcp.REAL_WORLD_VALIDATION_FOLLOWUP_REQUEST_QUEUE_SCHEMA_VERSION,
            "status": "waiting_for_actuals",
        },
    )
    assert lcp._real_world_validation_followup_request_queues(quiet_root)["status"] == "no_followup_requests_queued"


def test_webapp_payload_and_inbox_truth_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    assert lcp._request_from_webapp_payload({"queue_contract": lcp.WEBAPP_JOB_REQUEST_QUEUE_CONTRACT}) is None
    top_level = {"schema_version": lcp.WEBAPP_JOB_REQUEST_SCHEMA_VERSION, "job_id": "job-1"}
    assert lcp._request_from_webapp_payload(top_level) == top_level
    assert lcp._request_from_webapp_payload({"schema_version": "other"}) is None
    assert lcp._path_matches_configured_capture_root(None, capture_root) is False

    class _BadPath:
        def resolve(self) -> Path:
            raise RuntimeError("bad path")

    monkeypatch.setattr(lcp, "Path", lambda _value: _BadPath())
    assert lcp._path_matches_configured_capture_root("bad", capture_root) is False

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / ".hidden.json").write_text("{}", encoding="utf-8")
    (inbox / "invalid.json").write_text("{", encoding="utf-8")
    (inbox / "list.json").write_text("[]", encoding="utf-8")
    write_json(inbox / "other.json", {"schema_version": "other"})
    write_json(
        inbox / "candidate.json",
        {
            "schema_version": lcp.WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
            "job_id": "candidate",
            "site_package": {"capture_root": str(tmp_path / "other-capture")},
            "source": {"site_submission_id": "site-1"},
        },
    )

    truth = lcp._webapp_job_request_inbox_truth(inbox_path=inbox, capture_root=capture_root)
    assert truth["invalid_json_count"] == 1
    assert truth["status"] == "blocked"
    assert "no_job_request_matches_configured_capture_root" in truth["blockers"]
    assert "job_request_missing_required_webapp_ids" in truth["blockers"]


def test_staged_inputs_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "control" / "manifest.json"
    configured = tmp_path / "configured-staged.json"
    monkeypatch.setenv(lcp.STAGED_INPUTS_ENV, str(configured))
    assert lcp._staged_inputs_path(output) == configured.resolve()

    invalid = tmp_path / "invalid-staged.json"
    invalid.write_text("{", encoding="utf-8")
    assert lcp._load_staged_inputs(invalid, capture_root=None)["blockers"] == [
        "staged_inputs_read_failed:JSONDecodeError"
    ]
    configured.write_text("[]", encoding="utf-8")
    assert lcp._load_staged_inputs(configured, capture_root=None)["blockers"] == [
        "staged_inputs_not_json_object"
    ]

    capture_root = _capture_root(tmp_path)
    write_json(
        configured,
        {
            "schema_version": "wrong",
            "configured_capture_root": str(tmp_path / "other-root"),
            "arena_results": {"ready": True, "arena_results_dir": str(tmp_path / "missing-arena")},
            "webapp_request": {"staged": True, "target_path": str(tmp_path / "missing-webapp.json")},
            "live_closure_evidence": {"ready": True, "target_path": str(tmp_path / "missing-closure.json")},
            "deployment_outcomes": {
                "ready": True,
                "target_path": str(tmp_path / "missing-outcomes.json"),
                "records_ready_for_calibration": True,
                "owner_evidence_ready": True,
            },
            "policy_package": {"ready": True, "target_path": str(tmp_path / "missing-policy.json")},
            "real_robot_pov": {"ready": True, "target_path": str(tmp_path / "missing-pov.json")},
        },
    )
    blocked = lcp._load_staged_inputs(configured, capture_root=capture_root)
    assert blocked["status"] == "blocked"
    assert set(blocked["blockers"]) >= {
        "staged_inputs_schema_mismatch",
        "staged_inputs_capture_root_mismatch",
        "staged_arena_results_dir_missing",
        "staged_webapp_request_missing",
        "staged_live_closure_evidence_missing",
        "staged_policy_package_missing",
    }
    assert set(blocked["diagnostic_blockers"]) >= {
        "staged_deployment_outcomes_missing",
        "staged_real_robot_pov_missing",
    }

    write_json(configured, {"schema_version": lcp.LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION})
    assert lcp._load_staged_inputs(configured, capture_root=capture_root)["status"] == "empty"


def _packet_base(tmp_path: Path) -> dict[str, Any]:
    return {
        "generated_at": "2026-06-20T00:00:00Z",
        "capture_root": _capture_root(tmp_path),
        "job_request_inbox": tmp_path / "inbox",
        "package_dir": tmp_path / "package",
        "arena_results_dir": tmp_path / "arena",
        "output_path": tmp_path / "control" / "manifest.json",
        "setup_manifest_path": tmp_path / "setup.json",
        "setup_manifest": {"sections": {"webapp_upstream_truth": {"ready": True}, "real_arena_execution": {"ready": True}}},
        "inbox_run": {"manifest_path": "inbox.json"},
        "webapp_inbox_truth": {"ready": True, "accepted_policy_package_request_count": 1},
        "followup_request_queues": {"status": "none", "queues": []},
    }


def test_external_packet_prediction_and_owner_evidence_branches(tmp_path: Path) -> None:
    base = _packet_base(tmp_path)
    prediction_packet = lcp._build_external_input_packet(
        **base,
        staged_inputs={
            "status": "ready",
            "deployment_outcomes_ready": True,
            "deployment_outcomes_prediction_match_keys_ready": False,
            "deployment_outcomes_owner_evidence_ready": False,
            "policy_package_ready": True,
            "real_robot_pov_ready": True,
            "live_closure_evidence_ready": True,
        },
    )
    prediction_ids = {item["id"] for item in prediction_packet["required_inputs"]}
    assert "predicted_vs_actual_exact_match_keys" not in prediction_ids
    assert "real_world_deployment_outcome_owner_evidence" not in prediction_ids

    owner_packet = lcp._build_external_input_packet(
        **base,
        staged_inputs={
            "status": "ready",
            "deployment_outcomes_ready": True,
            "deployment_outcomes_prediction_match_keys_ready": True,
            "deployment_outcomes_owner_evidence_ready": False,
            "policy_package_ready": True,
            "real_robot_pov_ready": True,
            "live_closure_evidence_ready": True,
        },
    )
    owner_ids = {item["id"] for item in owner_packet["required_inputs"]}
    assert "predicted_vs_actual_exact_match_keys" not in owner_ids
    assert "real_world_deployment_outcome_owner_evidence" not in owner_ids


def test_external_input_packet_markdown_edges() -> None:
    packet = {
        "schema_version": lcp.LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION,
        "status": "waiting",
        "generated_at": "2026-06-20T00:00:00Z",
        "configured_paths": {"capture_root": "/tmp/capture"},
        "real_world_validation_followup_request_queues": {
            "status": "blocked",
            "ready_queue_count": 0,
            "queued_request_count": 1,
            "proof_boundary": "draft only",
            "queues": [
                "skip",
                {
                    "job_id": "job-1",
                    "status": "blocked",
                    "inbox_dir": "/tmp/inbox",
                    "safe_processing_command": "process",
                    "blockers": ["missing"],
                },
            ],
        },
        "required_inputs": [
            "skip",
            {
                "id": "webapp",
                "status": "blocked",
                "missing_fields": ["request_id"],
                "current_blockers": ["missing_request_id"],
                "blocker_packet": {
                    "owner": "webapp",
                    "safe_proof_command": "prove",
                    "retry_condition": "ready",
                    "disallowed_workaround": "placeholder",
                },
            },
        ],
        "enablement_inputs": [
            "skip",
            {
                "id": "delivery",
                "status": "blocked",
                "current_blockers": ["disabled"],
                "blocker_packet": {
                    "owner": "ops",
                    "safe_proof_command": "enable",
                    "retry_condition": "configured",
                    "disallowed_workaround": "manual claim",
                },
            },
        ],
    }
    markdown = lcp._external_input_packet_markdown(packet)
    assert "Safe processing command" in markdown
    assert "Missing fields" in markdown
    assert "Disallowed workaround" in markdown

    empty = {**packet, "real_world_validation_followup_request_queues": {}, "required_inputs": [], "enablement_inputs": []}
    assert lcp._external_input_packet_markdown(empty).count("- None.") == 3


def test_control_plane_uses_staged_arena_and_missing_inbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    arena_dir = tmp_path / "arena-results"
    arena_dir.mkdir()
    output = tmp_path / "control" / "manifest.json"
    staged_path = tmp_path / "staged.json"
    write_json(
        staged_path,
        {
            "schema_version": lcp.LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION,
            "arena_results": {"ready": True, "arena_results_dir": str(arena_dir)},
        },
    )
    monkeypatch.setenv(lcp.STAGED_INPUTS_ENV, str(staged_path))

    result = lcp.run_live_pipeline_control_plane(
        capture_root=capture_root,
        load_local_env=False,
        output_path=output,
    )

    assert result["staged_inputs"]["arena_results_ready"] is True
    assert result["inbox_run"]["blockers"] == ["missing_job_request_inbox"]


def test_main_forwards_arguments_and_prints_blockers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"output_path": str(tmp_path / "manifest.json"), "status": "blocked", "blockers": ["missing"]}

    monkeypatch.setattr(lcp, "run_live_pipeline_control_plane", _fake_run)
    assert (
        lcp.main(
            [
                "--capture-root",
                "capture",
                "--job-request-inbox",
                "inbox",
                "--package-dir",
                "package",
                "--arena-results-dir",
                "arena",
                "--simulator-audit-command",
                "sim",
                "--vision-labeling-command",
                "vision",
                "--delivery-command",
                "deliver",
                "--no-process-inbox",
                "--no-load-env-files",
                "--allow-digitalocean-read",
                "--digitalocean-token-env",
                "DO_TOKEN",
                "--digitalocean-droplet-name",
                "droplet",
                "--digitalocean-droplet-ip",
                "127.0.0.1",
                "--agent-mode",
                "fake",
                "--allow-live-agent-operator",
                "--provisioner",
                "local",
                "--simulator",
                "mujoco",
                "--allow-gpu-provisioning",
                "--allow-simulator-execution",
                "--allow-simulator",
                "mujoco",
                "--simulator-command",
                "mujoco=run",
                "--allow-cpu-simulator-preflight",
                "--cpu-preflight-backend",
                "mujoco",
                "--cpu-preflight-smoke-steps",
                "3",
                "--allow-cpu-preflight-render",
                "--allow-training",
                "--training-command",
                "train",
                "--timeout-seconds",
                "9",
                "--budget-usd",
                "1.5",
                "--arena-scenario-count",
                "12",
                "--arena-shard-size",
                "4",
                "--arena-num-envs",
                "2",
                "--arena-retry-budget",
                "1",
                "--allow-rollout-vision-labeling",
                "--allow-delivery-upload",
                "--arena-operator-mode",
                "fake",
                "--allow-live-agents-sdk",
                "--allow-live-codex-sdk",
                "--output-path",
                "out.json",
            ]
        )
        == 0
    )

    assert captured["process_inbox"] is False
    assert captured["load_local_env"] is False
    assert captured["allowed_simulators"] == ["mujoco"]
    assert captured["simulator_commands"] == ["mujoco=run"]
    assert captured["cpu_preflight_backends"] == ["mujoco"]
    assert captured["budget_usd"] == 1.5
    output = capsys.readouterr().out
    assert "status=blocked" in output
    assert "blockers=1" in output
