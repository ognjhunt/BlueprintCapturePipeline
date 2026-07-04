from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline.g1_endpoint_reference_adapter import adapter_manifest, build_response
from blueprint_pipeline.policy_endpoint_token_setup import create_team_policy_endpoint_token
from blueprint_pipeline import wam_vla_policy_endpoint_server as endpoint_server
from blueprint_pipeline import wam_vla_policy_endpoint_setup as endpoint_setup
from blueprint_pipeline.wam_vla_policy_endpoint_server import create_app, run_policy_command
from blueprint_pipeline.wam_vla_policy_endpoint_setup import build_wam_vla_policy_endpoint_setup


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _write_openvla_provider_smoke_job(job_dir: Path) -> Path:
    (job_dir / "openvla_provider_output").mkdir(parents=True)
    action = {"action_type": "waypoint", "waypoint": [0.36, -0.65, 0.79]}
    summary = {
        "status": "completed",
        "openvla_model_executed": True,
        "openvla_policy_action_command_ran": True,
        "action": action,
        "blockers": [],
    }
    output = {
        **summary,
        "schema_version": "openvla_policy_provider_output.v1",
        "openvla_model_loaded": True,
        "openvla_predict_action_invoked": True,
    }
    (job_dir / "openvla_policy_provider_smoke_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (
        job_dir / "openvla_provider_output" / "openvla_policy_provider_output.json"
    ).write_text(
        json.dumps(output),
        encoding="utf-8",
    )
    return job_dir


def test_wam_vla_policy_endpoint_setup_writes_contracts(tmp_path: Path) -> None:
    assert len(endpoint_setup._timestamp()) == len("20260620T010203Z")
    assert endpoint_setup._repo_root().name == "BlueprintCapturePipeline"
    summary = build_wam_vla_policy_endpoint_setup(
        output_dir=tmp_path / "setup",
        generated_at="now",
    )
    assert summary["status"] == "completed"
    contract = json.loads(Path(summary["artifacts"]["contract"]).read_text(encoding="utf-8"))
    options = json.loads(Path(summary["artifacts"]["options"]).read_text(encoding="utf-8"))
    assert contract["auth"]["raw_tokens_written_to_artifacts"] is False
    assert contract["http_contract"]["canonical"]["readyz"]["path"] == "/readyz"
    assert contract["http_contract"]["canonical"]["infer"]["path"] == "/infer"
    assert contract["http_contract"]["legacy_compatibility"]["policy_action"]["path"] == "/policy/action"
    assert "TEAM_POLICY_WORKER_URL" in contract["evaluator_envs"]["team"]
    assert "TEAM_POLICY_ENDPOINT_URL" in contract["evaluator_envs"]["team"]
    assert {row["id"] for row in options["options"]} >= {
        "openvla",
        "cosmos_predict_2_5",
        "unitree_rl_gym",
    }
    assert Path(summary["artifacts"]["env_template"]).is_file()
    assert Path(summary["artifacts"]["runbook"]).is_file()
    assert Path(summary["artifacts"]["policy_model_runnable_env"]).is_file()
    assert Path(summary["artifacts"]["policy_model_runnable_env_manifest"]).is_file()
    endpoint_boundary = json.loads(
        Path(summary["artifacts"]["policy_endpoint_boundary_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    assert endpoint_boundary["status"] == "endpoint_setup_boundary_only"
    assert endpoint_boundary["endpoint_setup_configured"] is True
    assert endpoint_boundary["robot_policy_execution_proven"] is False
    assert (
        endpoint_boundary["claim_boundary"][
            "endpoint_setup_is_not_robot_policy_execution"
        ]
        is True
    )
    assert (
        endpoint_boundary["claim_boundary"]["endpoint_setup_is_not_safety_validation"]
        is True
    )
    assert (
        endpoint_boundary["claim_boundary"]["endpoint_setup_is_not_deployment_approval"]
        is True
    )
    candidate_matrix = json.loads(
        Path(summary["artifacts"]["policy_model_candidate_matrix"]).read_text(encoding="utf-8")
    )
    assert {row["id"] for row in candidate_matrix["candidates"]} >= {
        "oscar_wam",
        "cosmos_wam",
        "openvla_policy",
        "unitree_g1_policy",
        "command_policy",
    }
    openvla_candidate = next(
        row for row in candidate_matrix["candidates"] if row["id"] == "openvla_policy"
    )
    assert (
        openvla_candidate["default_adapter_command"]
        == "blueprint-openvla-policy-command-adapter"
    )
    assert (
        openvla_candidate["current_repo_support"]
        == "implemented_command_adapter_requires_runtime_checkpoint_and_visual_frame"
    )
    truth = json.loads(
        Path(summary["artifacts"]["policy_model_truth_boundary"]).read_text(encoding="utf-8")
    )
    assert truth["reference_command_policy_is_not_real_wam_vla"] is True
    readiness = json.loads(
        Path(summary["artifacts"]["policy_model_endpoint_readiness_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    assert readiness["http_endpoint_wrapper_available"] is True
    assert readiness["claim_boundary"]["endpoint_creation_is_not_model_execution_proof"] is True
    assert {row["candidate_id"] for row in readiness["candidates"]} >= {
        "oscar_wam",
        "cosmos_wam",
        "openvla_policy",
        "unitree_g1_policy",
    }
    creation_plan = json.loads(
        Path(summary["artifacts"]["policy_model_endpoint_creation_plan"]).read_text(
            encoding="utf-8"
        )
    )
    assert creation_plan["http_wrapper_binary_available"] is True
    layer_summary = creation_plan["readiness_layer_summary"]
    assert layer_summary["reference_endpoint_wrapper_ready"] is True
    assert layer_summary["reference_endpoint_real_model_claim_allowed"] is False
    assert layer_summary["wam_rollout_provider_ready"] is False
    assert layer_summary["vla_manipulation_policy_ready_candidate_count"] == 0
    assert layer_summary["closed_loop_wam_policy_endpoint_ready"] is False
    assert (
        "blocked_closed_loop_wam_policy_requery_not_yet_proven"
        in layer_summary["closed_loop_wam_policy_endpoint_blockers"]
    )
    assert layer_summary["claim_boundary"][
        "wam_rollout_provider_ready_is_not_robot_policy_ready"
    ] is True
    assert creation_plan["can_create_real_model_endpoint_now"] is False
    assert creation_plan["minimum_user_supplied_inputs"]
    assert "HTTP endpoint without a runnable command" in " ".join(
        creation_plan["why_cannot_just_create_missing_model_endpoints"]
    )
    adapter = json.loads(
        Path(summary["artifacts"]["policy_command_adapter_manifest"]).read_text(encoding="utf-8")
    )
    assert (
        adapter["default_reference_adapter_command"]
        == endpoint_server.BUILTIN_REFERENCE_ADAPTER_COMMAND
    )
    assert adapter["fallback_reference_adapter_subprocess_command"].endswith(
        "src/blueprint_pipeline/g1_endpoint_reference_adapter.py"
    )
    assert adapter["default_reference_adapter_invocation_mode"] == "in_process_builtin"
    assert (
        adapter["console_script_reference_adapter_command"]
        == "blueprint-g1-endpoint-reference-adapter"
    )
    assert adapter["openvla_policy_adapter_command"] == "blueprint-openvla-policy-command-adapter"
    assert (
        adapter["provider_worker_policy_adapter_command"]
        == "blueprint-provider-worker-policy-command-adapter"
    )
    assert adapter["provider_worker_policy_adapter_contract"][
        "requires_readyz_before_infer"
    ] is True


def test_wam_vla_policy_endpoint_setup_imports_openvla_provider_smoke_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_job = _write_openvla_provider_smoke_job(
        tmp_path / "robot_eval_jobs" / "openvla_policy_provider_smoke_20260622T092432Z"
    )
    monkeypatch.setenv("BLUEPRINT_OPENVLA_PROVIDER_SMOKE_JOB_DIR", str(provider_job))
    monkeypatch.setattr(endpoint_setup, "_repo_root", lambda: tmp_path)

    summary = build_wam_vla_policy_endpoint_setup(
        output_dir=tmp_path / "setup-openvla-proof",
        generated_at="now",
    )

    candidate_matrix = json.loads(
        Path(summary["artifacts"]["policy_model_candidate_matrix"]).read_text(encoding="utf-8")
    )
    openvla_candidate = next(
        row for row in candidate_matrix["candidates"] if row["id"] == "openvla_policy"
    )
    assert openvla_candidate["provider_smoke_completed"] is True
    assert openvla_candidate["openvla_model_executed"] is True
    assert openvla_candidate["openvla_policy_action_command_ran"] is False
    assert openvla_candidate["openvla_policy_action_command_imported"] is True
    assert openvla_candidate["last_provider_action"]["action_type"] == "waypoint"
    assert openvla_candidate["endpoint_closed_loop_policy_proven"] is False
    assert openvla_candidate["unitree_g1_dexterous_manipulation_proven"] is False

    truth = json.loads(
        Path(summary["artifacts"]["policy_model_truth_boundary"]).read_text(encoding="utf-8")
    )
    assert truth["openvla_provider_smoke_model_executed"] is True
    assert truth["policy_action_model_command_ran"] is False
    assert truth["openvla_policy_action_command_ran"] is False
    assert truth["policy_action_model_provider_smoke_imported"] is True
    assert truth["openvla_policy_action_command_imported"] is True
    assert (
        truth["openvla_provider_smoke_is_not_closed_loop_endpoint_or_dexterous_manipulation"]
        is True
    )


def test_policy_model_runnable_env_detects_oscar_replay_artifacts(tmp_path: Path) -> None:
    checkpoint = (
        tmp_path
        / "robot_eval_jobs"
        / "wam_model_runtime_bootstrap_oscar_20260621T025044Z"
        / "runtime_sources"
        / "oscar_wam"
        / "checkpoint"
    )
    checkpoint.mkdir(parents=True)
    provider_dir = (
        tmp_path
        / "robot_eval_jobs"
        / "oscar_wam_hands_pov_first_person_passthrough_fresh_vast_49f_20260622T065451Z"
        / "oscar_wam_provider_command_workspace"
        / "vast_provider_run"
    )
    provider_dir.mkdir(parents=True)
    with zipfile.ZipFile(provider_dir / "vast_provider_runtime_output.zip", "w") as archive:
        archive.writestr(
            "wam_runtime_result.json",
            json.dumps(
                {
                    "status": "completed",
                    "learned_wam_model_ran": True,
                    "truth_boundary": {"generated_video_is_model_output": True},
                }
            ),
        )

    metadata, env_text = endpoint_setup.build_policy_model_runnable_env_artifact(
        repo_root=tmp_path,
        generated_at="now",
    )

    assert metadata["status"] == "ready"
    assert metadata["oscar_replay_provider_ready"] is True
    assert metadata["oscar_fresh_provider_command_ready"] is True
    assert metadata["oscar_checkpoint_path"] == str(checkpoint)
    assert metadata["oscar_completed_provider_job_dir"] == str(provider_dir)
    assert "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL=true" in env_text
    assert "BLUEPRINT_OSCAR_WAM_PROVIDER_COMPLETED_JOB_DIR" in env_text
    assert (
        metadata["claim_boundary"]["replay_completed_provider_output_is_not_fresh_model_run"]
        is True
    )


def test_wam_vla_policy_endpoint_setup_main_prints_summary(
    tmp_path: Path,
    capsys,
) -> None:
    assert endpoint_setup.main(["--output-dir", str(tmp_path / "setup-main")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "completed"
    assert printed["output_dir"].endswith("setup-main")


def test_team_policy_endpoint_token_helper_and_reference_adapter(tmp_path: Path) -> None:
    token_file = tmp_path / "team_policy_endpoint_token.txt"
    summary = create_team_policy_endpoint_token(token_file=token_file, generated_at="now")
    assert summary["status"] == "created"
    assert summary["file_mode_octal"] == "0o600"
    assert summary["raw_token_written_to_stdout"] is False
    assert summary["raw_token_hash_written_to_artifacts"] is False
    assert token_file.read_text(encoding="utf-8").strip()
    second = create_team_policy_endpoint_token(token_file=token_file, generated_at="now")
    assert second["status"] == "already_present"

    manifest = adapter_manifest()
    assert manifest["policy_id"] == "blueprint_g1_endpoint_reference_adapter"
    assert manifest["fixture_policy_used"] is False
    assert set(manifest["supported_action_types"]) >= {
        "waypoint",
        "base_velocity",
        "stop",
        "inspect_look",
        "manipulation_contact",
    }
    base_observation = {
        "task_id": "contact_or_push_light_object",
        "step_index": 0,
        "object_state": {"position": [0.3, -0.6, 0.27]},
        "route_task_state": {"target_pose": [0.5, 0.0, 0.79], "target_error_m": 0.5},
    }
    response = build_response({"observation": base_observation})
    assert response["policy_id"] == "blueprint_g1_endpoint_reference_adapter"
    assert response["action"]["action_type"] == "manipulation_contact"
    assert response["adapter_metadata"]["real_wam_vla_model"] is False


def test_wam_vla_policy_endpoint_server_wraps_command_with_file_token(tmp_path: Path) -> None:
    assert endpoint_server._read_token(None) is None
    assert endpoint_server._read_token(str(tmp_path / "missing-token.txt")) is None
    assert endpoint_server._redact({"api_key": "secret", "items": [{"token": "abc"}]}) == {
        "api_key": "<redacted>",
        "items": [{"token": "<redacted>"}],
    }
    endpoint_server._check_auth(authorization=None, token_file=None)

    command_path = tmp_path / "policy_command.py"
    command_path.write_text(
        """
import json
import sys

payload = json.loads(sys.stdin.read())
assert payload["observation"]["task_id"] == "approach_target"
print(json.dumps({
    "policy_id": "unit_test_policy",
    "action": {"action_type": "waypoint", "waypoint": [0.5, 0.0, 0.79]},
}))
""".strip(),
        encoding="utf-8",
    )
    command = f"{sys.executable} {command_path}"
    response, meta = run_policy_command(
        command=command,
        payload={"observation": {"task_id": "approach_target"}},
        timeout_seconds=2.0,
    )
    assert response["action"]["action_type"] == "waypoint"
    assert meta["command_exit_code"] == 0
    assert meta["policy_adapter_invocation_mode"] == "subprocess"
    assert meta["subprocess_spawned"] is True

    builtin_response, builtin_meta = run_policy_command(
        command=endpoint_server.BUILTIN_REFERENCE_ADAPTER_COMMAND,
        payload={
            "observation": {
                "task_id": "approach_target",
                "step_index": 0,
                "base_pose": {"position": [0.0, 0.0, 0.79], "yaw_rad": 0.0},
                "route_task_state": {"target_pose": [1.0, 0.0, 0.79], "target_error_m": 1.0},
            }
        },
        timeout_seconds=0.001,
    )
    assert builtin_response["policy_id"] == "blueprint_g1_endpoint_reference_adapter"
    assert builtin_response["action"]["action_type"] == "base_velocity"
    assert builtin_meta["policy_adapter_invocation_mode"] == "in_process_builtin"
    assert builtin_meta["subprocess_spawned"] is False

    token_file = tmp_path / "token.txt"
    token_file.write_text("secret-token\n", encoding="utf-8")
    app = create_app(
        policy_command=command,
        auth_token_file=str(token_file),
        timeout_seconds=5.0,
    )
    client = TestClient(app)
    legacy_health = client.get("/health").json()
    assert legacy_health["policy_command_configured"] is True
    assert legacy_health["canonical_http_contract"]["readyz"] == "/readyz"
    assert legacy_health["canonical_http_contract"]["infer"] == "/infer"
    assert client.get("/healthz").json()["policy_command_configured"] is True
    ready = client.get("/readyz").json()
    assert ready["model_ready"] is True
    assert ready["ready_for_inference"] is True
    unauthorized = client.post(
        "/policy/action",
        json={"observation": {"task_id": "approach_target"}},
    )
    assert unauthorized.status_code == 401
    forbidden = client.post(
        "/policy/action",
        headers={"authorization": "Bearer wrong-token"},
        json={"observation": {"task_id": "approach_target"}},
    )
    assert forbidden.status_code == 403
    ok = client.post(
        "/policy/action",
        headers={"authorization": "Bearer secret-token"},
        json={"observation": {"task_id": "approach_target"}},
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["policy_id"] == "unit_test_policy"
    assert payload["action"]["action_type"] == "waypoint"
    assert payload["endpoint_metadata"]["raw_token_values_returned"] is False
    infer_ok = client.post(
        "/infer",
        headers={"authorization": "Bearer secret-token"},
        json={"observation": {"task_id": "approach_target"}},
    )
    assert infer_ok.status_code == 200
    assert infer_ok.json()["action"]["action_type"] == "waypoint"
    shutdown = client.post(
        "/shutdown",
        headers={"authorization": "Bearer secret-token"},
    )
    assert shutdown.status_code == 200
    assert shutdown.json()["provider_adapter_must_record_teardown"] is True
    assert client.get("/readyz").json()["ready_for_inference"] is False
    blocked_after_shutdown = client.post(
        "/infer",
        headers={"authorization": "Bearer secret-token"},
        json={"observation": {"task_id": "approach_target"}},
    )
    assert blocked_after_shutdown.status_code == 503

    builtin_client = TestClient(
        create_app(
            policy_command=endpoint_server.BUILTIN_REFERENCE_ADAPTER_COMMAND,
            auth_token_file=None,
            timeout_seconds=0.001,
        )
    )
    builtin_health = builtin_client.get("/health").json()
    assert builtin_health["policy_adapter_invocation_mode"] == "in_process_builtin"
    assert builtin_health["subprocess_spawned_per_request"] is False
    assert builtin_client.get("/readyz").json()["ready_for_inference"] is True
    builtin_ok = builtin_client.post(
        "/policy/action",
        json={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "object_state": {"position": [0.3, -0.6, 0.27]},
            }
        },
    )
    assert builtin_ok.status_code == 200
    builtin_payload = builtin_ok.json()
    assert builtin_payload["policy_id"] == "blueprint_g1_endpoint_reference_adapter"
    assert builtin_payload["action"]["action_type"] == "manipulation_contact"
    assert (
        builtin_payload["endpoint_metadata"]["policy_adapter_invocation_mode"]
        == "in_process_builtin"
    )
    assert builtin_payload["endpoint_metadata"]["subprocess_spawned"] is False
    builtin_infer = builtin_client.post(
        "/infer",
        json={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "object_state": {"position": [0.3, -0.6, 0.27]},
            }
        },
    )
    assert builtin_infer.status_code == 200
    assert builtin_infer.json()["policy_id"] == "blueprint_g1_endpoint_reference_adapter"

    with pytest.raises(RuntimeError, match="missing_policy_command"):
        run_policy_command(command="", payload={}, timeout_seconds=0.1)

    fail_command = tmp_path / "fail_policy.py"
    fail_command.write_text(
        "import sys\nsys.stderr.write('secret stderr')\nsys.exit(2)\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="policy_command_failed"):
        run_policy_command(
            command=f"{sys.executable} {fail_command}",
            payload={"observation": {}},
            timeout_seconds=2.0,
        )

    list_command = tmp_path / "list_policy.py"
    list_command.write_text("print('[]')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stdout_not_json_object"):
        run_policy_command(
            command=f"{sys.executable} {list_command}",
            payload={"observation": {}},
            timeout_seconds=2.0,
        )

    missing_command_client = TestClient(create_app(policy_command="", auth_token_file=None))
    assert missing_command_client.get("/health").json()["status"] == "blocked_missing_policy_command"
    missing_observation = missing_command_client.post("/policy/action", json={})
    assert missing_observation.status_code == 422
    no_command = missing_command_client.post(
        "/policy/action",
        json={"observation": {"task_id": "approach_target"}},
    )
    assert no_command.status_code == 503

    fail_client = TestClient(
        create_app(policy_command=f"{sys.executable} {fail_command}", auth_token_file=None)
    )
    assert fail_client.post(
        "/policy/action",
        json={"observation": {"task_id": "approach_target"}},
    ).status_code == 502

    no_action_command = tmp_path / "no_action_policy.py"
    no_action_command.write_text("print('{\"policy_id\":\"missing-action\"}')\n", encoding="utf-8")
    no_action_client = TestClient(
        create_app(policy_command=f"{sys.executable} {no_action_command}", auth_token_file=None)
    )
    assert no_action_client.post(
        "/policy/action",
        json={"observation": {"task_id": "approach_target"}},
    ).status_code == 502


def test_wam_vla_policy_endpoint_server_main_uses_uvicorn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(app, *, host: str, port: int) -> None:
        calls.append({"app": app, "host": host, "port": port})

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))

    assert endpoint_server.main(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--policy-command",
            "python policy.py",
            "--auth-token-file",
            str(tmp_path / "token.txt"),
            "--timeout-seconds",
            "1.5",
        ]
    ) == 0
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 9999
