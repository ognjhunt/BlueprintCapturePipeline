from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import lambda_provider_adapter as adapter
from blueprint_pipeline.lambda_provider_adapter import (
    DEFAULT_LAMBDA_API_KEY_FILE,
    LAMBDA_API_GATE_ENV,
    LAMBDA_API_KEY_ENV,
    LAMBDA_API_KEY_FILE_ENV,
    LAMBDA_INSTANCE_IDS_ENV,
    main as lambda_adapter_main,
    run_lambda_provider_adapter,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ready_lambda_request(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "lambda-adapter-job-1",
            "provider": "lambda_cloud",
            "status": "request_manifest_ready",
            "operation": "enqueue_lambda_cloud_worker",
            "provider_request_shape": {
                "operation": "enqueue_lambda_cloud_worker",
                "image": {
                    "configured_image_ref": (
                        "registry.example/blueprint/isaac-eval-worker:2026-06-12"
                    ),
                    "configured_image_ref_is_versioned": True,
                    "configured_image_ref_fetchable_by_provider": True,
                },
                "command": (
                    "blueprint-run-robot-eval-worker --manifest "
                    "${BLUEPRINT_EVAL_MANIFEST_URI}"
                ),
                "inputs": {
                    "manifest_uri": (
                        "r2://blueprint-artifacts/jobs/lambda-adapter-job-1/"
                        "worker_manifest.json"
                    ),
                    "manifest_uri_fetchable_by_provider": True,
                    "capture_root_bundle_uri": (
                        "r2://blueprint-artifacts/jobs/lambda-adapter-job-1/"
                        "capture_root.tar"
                    ),
                    "capture_root_bundle_uri_fetchable_by_provider": True,
                    "artifact_output_uri": (
                        "r2://blueprint-artifacts/jobs/lambda-adapter-job-1/out/"
                    ),
                    "artifact_output_uri_required": True,
                    "artifact_output_uri_provider_writable": True,
                    "artifact_output_write_auth_contract_ready": True,
                    "artifact_output_write_auth": {
                        "write_auth_contract_ready": True,
                    },
                },
                "local_sim_only_prerequisite": {
                    "status": "passed",
                    "local_sim_only_evidence_clean": True,
                    "blockers": [],
                },
                "limits": {
                    "hard_timeout_seconds": 600,
                    "idle_timeout_seconds": 60,
                    "external_watchdog_ttl_seconds": 900,
                    "external_watchdog_owner": "provider_launcher_or_owner_control_plane",
                    "max_active_workers": 1,
                    "requested_budget_usd": 0.25,
                    "idle_shutdown_required": True,
                },
                "artifact_finalizer": {
                    "upload_before_shutdown_required": True,
                },
                "environment": {
                    "secret_values_in_artifact": False,
                },
            },
        },
    )
    return path


@pytest.fixture(autouse=True)
def _clear_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LAMBDA_API_KEY_ENV, raising=False)
    monkeypatch.delenv(LAMBDA_API_KEY_FILE_ENV, raising=False)
    monkeypatch.delenv(LAMBDA_API_GATE_ENV, raising=False)
    monkeypatch.delenv(LAMBDA_INSTANCE_IDS_ENV, raising=False)


def test_read_api_key_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LAMBDA_API_KEY_ENV, "secret_env_value")
    key, meta = adapter._read_lambda_api_key()
    assert key == "secret_env_value"
    assert meta["api_key_configured"] is True
    assert meta["api_key_source"] == LAMBDA_API_KEY_ENV


def test_read_api_key_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "lambda_api_key"
    key_file.write_text("secret_file_value\n", encoding="utf-8")
    monkeypatch.setenv(LAMBDA_API_KEY_FILE_ENV, str(key_file))
    key, meta = adapter._read_lambda_api_key()
    assert key == "secret_file_value"
    assert meta["api_key_configured"] is True
    assert meta["api_key_source"] == LAMBDA_API_KEY_FILE_ENV


def test_default_key_file_path_matches_secrets_convention() -> None:
    assert DEFAULT_LAMBDA_API_KEY_FILE == "~/.blueprint-secrets/lambda_api_key"


def test_dry_run_is_consumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "lambda_api_key"
    key_file.write_text("secret_file_value\n", encoding="utf-8")
    monkeypatch.setenv(LAMBDA_API_KEY_FILE_ENV, str(key_file))
    request_path = _ready_lambda_request(tmp_path / "request.json")
    output_path = tmp_path / "lambda_provider_adapter_result.json"

    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=output_path,
        mode="dry-run",
        region_name="us-west-1",
        instance_type_name="gpu_1x_a10",
        ssh_key_name="blueprint-key",
    )

    assert result["status"] == "dry_run_ready"
    assert result["blockers"] == []
    assert result["api_call_performed"] is False
    assert result["api_key_readiness"]["api_key_configured"] is True
    assert result["raw_api_key_stored"] is False
    assert result["lambda_request"]["body"]["user_data"] == "<redacted:user_data>"
    readiness = _read_json(tmp_path / "lambda_provider_readiness_manifest.json")
    assert readiness["status"] == "ready_for_explicit_paid_provider_attempt"
    assert readiness["lambda_launch_contract"]["ssh_key_names_required_count"] == 1
    endpoint = _read_json(tmp_path / "provider_worker_endpoint_manifest.json")
    assert endpoint["provider"] == "lambda_cloud"
    persisted = _read_json(output_path)
    assert persisted["status"] == "dry_run_ready"
    assert "secret_file_value" not in json.dumps(persisted)
    assert "blueprint-run-robot-eval-worker" not in json.dumps(persisted)


def test_dry_run_can_resolve_signed_url_files_without_persisting_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_file = tmp_path / "lambda_api_key"
    key_file.write_text("secret_file_value\n", encoding="utf-8")
    monkeypatch.setenv(LAMBDA_API_KEY_FILE_ENV, str(key_file))
    bundle_url_file = tmp_path / "provider_bundle_url.txt"
    output_url_file = tmp_path / "provider_output_put_url.txt"
    bundle_url_file.write_text(
        "https://object-store.example/bundle.zip?X-Amz-Signature=bundle-secret\n",
        encoding="utf-8",
    )
    kitchen_url_file = tmp_path / "kitchen_bundle_url.txt"
    kitchen_url_file.write_text(
        "https://object-store.example/kitchen.zip?X-Amz-Signature=kitchen-secret\n",
        encoding="utf-8",
    )
    output_url_file.write_text(
        "https://object-store.example/output.zip?X-Amz-Signature=put-secret\n",
        encoding="utf-8",
    )
    request_path = _ready_lambda_request(tmp_path / "request.json")
    request = _read_json(request_path)
    inputs = request["provider_request_shape"]["inputs"]  # type: ignore[index]
    inputs.pop("manifest_uri")  # type: ignore[union-attr]
    inputs.pop("capture_root_bundle_uri")  # type: ignore[union-attr]
    inputs["manifest_uri_file"] = str(bundle_url_file)  # type: ignore[index]
    inputs["capture_root_bundle_uri_file"] = str(bundle_url_file)  # type: ignore[index]
    inputs["artifact_output_uri_required"] = False  # type: ignore[index]
    inputs["artifact_output_signed_put_url_file"] = str(output_url_file)  # type: ignore[index]
    environment = request["provider_request_shape"]["environment"]  # type: ignore[index]
    environment["plaintext_env_var_names"] = ["KITCHEN_BUNDLE_URL"]  # type: ignore[index]
    environment["plaintext_env_value_files"] = {  # type: ignore[index]
        "KITCHEN_BUNDLE_URL": str(kitchen_url_file)
    }
    _write_json(request_path, request)

    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "out.json",
        mode="dry-run",
        region_name="us-west-1",
        instance_type_name="gpu_1x_a10",
        ssh_key_name="blueprint-key",
    )

    assert result["status"] == "dry_run_ready"
    assert result["request_summary"]["manifest_uri_present"] is True
    assert result["request_summary"]["capture_root_bundle_uri_present"] is True
    persisted = json.dumps(_read_json(tmp_path / "out.json"))
    assert "bundle-secret" not in persisted
    assert "put-secret" not in persisted
    assert "kitchen-secret" not in persisted
    assert "X-Amz-Signature" not in persisted


def test_live_mode_blocks_without_explicit_api_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "lambda_api_key"
    key_file.write_text("secret_file_value\n", encoding="utf-8")
    monkeypatch.setenv(LAMBDA_API_KEY_FILE_ENV, str(key_file))
    request_path = _ready_lambda_request(tmp_path / "request.json")

    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "out.json",
        mode="launch-instance",
        region_name="us-west-1",
        instance_type_name="gpu_1x_a10",
        ssh_key_name="blueprint-key",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "lambda_api_gate_blocked"
    assert f"missing_env_{LAMBDA_API_GATE_ENV}" in result["blockers"]
    assert "missing_cli_allow_lambda_api_call" in result["blockers"]
    assert result["api_call_performed"] is False
    assert result["lambda_side_effects_may_have_occurred"] is False


def test_wrong_provider_is_blocked(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    _write_json(
        request_path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "provider": "runpod",
            "status": "request_manifest_ready",
            "provider_request_shape": {"inputs": {}},
        },
    )
    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "out.json",
    )
    assert result["status"] == "blocked"
    assert "provider_launch_request_not_lambda_cloud" in result["blockers"]
    assert "missing_provider_worker_manifest_uri" in result["blockers"]


def test_invalid_json_request_is_blocked(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("[]", encoding="utf-8")
    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "out.json",
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["invalid_provider_launch_request_json"]


def test_launch_instance_submits_with_explicit_gates_and_redacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_file = tmp_path / "lambda_api_key"
    key_file.write_text("secret_file_value\n", encoding="utf-8")
    monkeypatch.setenv(LAMBDA_API_KEY_FILE_ENV, str(key_file))
    monkeypatch.setenv(LAMBDA_API_GATE_ENV, "true")
    request_path = _ready_lambda_request(tmp_path / "request.json")
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"data":{"instance_ids":["lambda-instance-1"]}}'

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers.get("Authorization")
        captured["user_agent"] = (
            request.headers.get("User-agent") or request.headers.get("User-Agent")
        )
        return FakeResponse()

    monkeypatch.setattr(adapter.urllib.request, "urlopen", fake_urlopen)

    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "out.json",
        mode="launch-instance",
        allow_lambda_api_call=True,
        region_name="us-west-1",
        instance_type_name="gpu_1x_a10",
        ssh_key_name="blueprint-key",
    )

    assert result["status"] == "submitted"
    assert result["api_call_performed"] is True
    assert result["lambda_side_effects_may_have_occurred"] is True
    assert result["provider_job_submitted"] is True
    assert result["lambda_instance_ids"] == ["lambda-instance-1"]
    assert captured["url"] == "https://cloud.lambda.ai/api/v1/instance-operations/launch"
    assert captured["authorization"] == "Bearer secret_file_value"
    assert captured["user_agent"] == "curl/8.7.1"
    assert captured["body"]["region_name"] == "us-west-1"  # type: ignore[index]
    assert captured["body"]["ssh_key_names"] == ["blueprint-key"]  # type: ignore[index]
    assert captured["body"]["user_data"].startswith("#!/usr/bin/env bash")  # type: ignore[index]
    assert 'DOCKER_CMD="sudo docker"' in captured["body"]["user_data"]  # type: ignore[operator]
    assert "--entrypoint bash" in captured["body"]["user_data"]  # type: ignore[operator]
    assert "-v /workspace:/workspace" in captured["body"]["user_data"]  # type: ignore[operator]
    assert " bash -lc " not in captured["body"]["user_data"]  # type: ignore[operator]
    assert " -lc 'blueprint-run-robot-eval-worker" in captured["body"]["user_data"]  # type: ignore[operator]
    persisted = _read_json(tmp_path / "out.json")
    payload = json.dumps(persisted)
    assert "secret_file_value" not in payload
    assert "#!/usr/bin/env bash" not in payload
    assert persisted["lambda_request"]["body"]["user_data"] == "<redacted:user_data>"


def test_terminate_instances_writes_teardown_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_file = tmp_path / "lambda_api_key"
    key_file.write_text("secret_file_value\n", encoding="utf-8")
    monkeypatch.setenv(LAMBDA_API_KEY_FILE_ENV, str(key_file))
    monkeypatch.setenv(LAMBDA_API_GATE_ENV, "true")
    request_path = _ready_lambda_request(tmp_path / "request.json")
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"data":{"terminated_instances":[{"id":"lambda-instance-1","status":"terminating"}]}}'

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(adapter.urllib.request, "urlopen", fake_urlopen)

    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "out.json",
        mode="terminate-instances",
        allow_lambda_api_call=True,
        instance_ids=["lambda-instance-1"],
    )

    assert result["status"] == "termination_requested"
    assert result["provider_teardown_requested"] is True
    assert captured["url"] == "https://cloud.lambda.ai/api/v1/instance-operations/terminate"
    assert captured["body"] == {"instance_ids": ["lambda-instance-1"]}
    teardown = _read_json(tmp_path / "lambda_provider_teardown_manifest.json")
    assert teardown["status"] == "termination_requested"
    assert teardown["continuing_spend_requires_followup_list_instances"] is True


def test_cli_main_dry_run_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_file = tmp_path / "lambda_api_key"
    key_file.write_text("secret_file_value\n", encoding="utf-8")
    monkeypatch.setenv(LAMBDA_API_KEY_FILE_ENV, str(key_file))
    request_path = _ready_lambda_request(tmp_path / "request.json")
    output_path = tmp_path / "out.json"

    exit_code = lambda_adapter_main(
        [
            "--provider-launch-request",
            str(request_path),
            "--output-path",
            str(output_path),
            "--mode",
            "dry-run",
            "--lambda-region-name",
            "us-west-1",
            "--lambda-instance-type-name",
            "gpu_1x_a10",
            "--lambda-ssh-key-name",
            "blueprint-key",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "status=dry_run_ready" in captured
