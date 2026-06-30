from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import lambda_provider_adapter as adapter
from blueprint_pipeline.lambda_provider_adapter import (
    DEFAULT_LAMBDA_API_KEY_FILE,
    LAMBDA_API_KEY_ENV,
    LAMBDA_API_KEY_FILE_ENV,
    LIVE_LAUNCH_NOT_IMPLEMENTED_BLOCKER,
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
                },
                "inputs": {
                    "manifest_uri": (
                        "r2://blueprint-artifacts/jobs/lambda-adapter-job-1/"
                        "worker_manifest.json"
                    ),
                    "capture_root_bundle_uri": (
                        "r2://blueprint-artifacts/jobs/lambda-adapter-job-1/"
                        "capture_root.tar"
                    ),
                    "artifact_output_uri": (
                        "r2://blueprint-artifacts/jobs/lambda-adapter-job-1/out/"
                    ),
                    "artifact_output_uri_required": True,
                },
            },
        },
    )
    return path


@pytest.fixture(autouse=True)
def _clear_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LAMBDA_API_KEY_ENV, raising=False)
    monkeypatch.delenv(LAMBDA_API_KEY_FILE_ENV, raising=False)


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
    )

    assert result["status"] == "dry_run_ready"
    assert result["blockers"] == []
    assert result["api_call_performed"] is False
    assert result["live_launch_supported"] is False
    assert result["api_key_readiness"]["api_key_configured"] is True
    assert result["raw_api_key_stored"] is False
    persisted = _read_json(output_path)
    assert persisted["status"] == "dry_run_ready"
    assert "secret_file_value" not in json.dumps(persisted)


def test_live_mode_blocked_not_implemented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "lambda_api_key"
    key_file.write_text("secret_file_value\n", encoding="utf-8")
    monkeypatch.setenv(LAMBDA_API_KEY_FILE_ENV, str(key_file))
    request_path = _ready_lambda_request(tmp_path / "request.json")

    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "out.json",
        mode="allocate",
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [LIVE_LAUNCH_NOT_IMPLEMENTED_BLOCKER]
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
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "status=dry_run_ready" in captured
