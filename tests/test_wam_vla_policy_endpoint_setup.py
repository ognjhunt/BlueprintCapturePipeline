from __future__ import annotations

import json
import sys
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
    assert "TEAM_POLICY_ENDPOINT_URL" in contract["evaluator_envs"]["team"]
    assert {row["id"] for row in options["options"]} >= {
        "openvla",
        "cosmos_predict_2_5",
        "unitree_rl_gym",
    }
    assert Path(summary["artifacts"]["env_template"]).is_file()
    assert Path(summary["artifacts"]["runbook"]).is_file()
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
    assert creation_plan["can_create_real_model_endpoint_now"] is False
    assert creation_plan["minimum_user_supplied_inputs"]
    assert "HTTP endpoint without a runnable command" in " ".join(
        creation_plan["why_cannot_just_create_missing_model_endpoints"]
    )
    adapter = json.loads(
        Path(summary["artifacts"]["policy_command_adapter_manifest"]).read_text(encoding="utf-8")
    )
    assert adapter["default_reference_adapter_command"] == "blueprint-g1-endpoint-reference-adapter"


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

    token_file = tmp_path / "token.txt"
    token_file.write_text("secret-token\n", encoding="utf-8")
    app = create_app(
        policy_command=command,
        auth_token_file=str(token_file),
        timeout_seconds=2.0,
    )
    client = TestClient(app)
    assert client.get("/health").json()["policy_command_configured"] is True
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
