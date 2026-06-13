from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.runpod_live_execution_proof import (
    RUNPOD_GPU_LAUNCH_GATE_ENV,
    collect_runpod_live_execution_proof,
    main,
)
from blueprint_pipeline.runpod_provider_adapter import (
    RUNPOD_API_GATE_ENV,
    RUNPOD_API_KEY_ENV,
    RUNPOD_API_KEY_FILE_ENV,
    RUNPOD_CONFIG_FILE_ENV,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _provider_launch_request(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "runpod-proof-job",
            "provider": "runpod",
            "status": "request_manifest_ready",
        },
    )
    return path


def test_runpod_live_execution_proof_blocks_without_gates(tmp_path: Path, monkeypatch) -> None:
    request_path = _provider_launch_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.delenv(RUNPOD_API_GATE_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_GPU_LAUNCH_GATE_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_CONFIG_FILE_ENV, str(tmp_path / "missing-config.toml"))

    result = collect_runpod_live_execution_proof(
        provider_launch_request_path=request_path,
        allow_runpod_api_call=False,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert result["runpod_side_effects_may_have_occurred"] is False
    assert f"missing_env_{RUNPOD_API_GATE_ENV}" in result["blockers"]
    assert f"missing_env_{RUNPOD_GPU_LAUNCH_GATE_ENV}" in result["blockers"]
    assert (
        f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}_or_{RUNPOD_CONFIG_FILE_ENV}"
        in result["blockers"]
    )


def test_runpod_live_execution_proof_lists_stops_and_redacts_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _provider_launch_request(tmp_path / "gpu_provider_launch_request.json")
    adapter_result = tmp_path / "runpod_provider_adapter_result.live.json"
    _write_json(
        adapter_result,
        {
            "schema_version": "runpod_provider_adapter_result.v1",
            "job_id": "runpod-proof-job",
            "runpod_response": {"id": "pod-123"},
        },
    )
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_GPU_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")
    calls: list[dict[str, object]] = []

    class FakeResponse:
        status = 200

        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.header_items()),
                "body": body,
            }
        )
        if request.full_url.endswith("/pods/pod-123/stop"):
            return FakeResponse(b'{"id":"pod-123","desiredStatus":"STOPPED"}')
        if len([call for call in calls if str(call["url"]).endswith("/pods")]) == 1:
            return FakeResponse(
                b'[{"id":"pod-123","desiredStatus":"RUNNING"}]'
            )
        return FakeResponse(
            b'[{"id":"pod-123","desiredStatus":"STOPPED"}]'
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = collect_runpod_live_execution_proof(
        provider_launch_request_path=request_path,
        adapter_result_path=adapter_result,
        output_path=tmp_path / "runpod_live_execution_proof.json",
        stop_pod=True,
        allow_runpod_api_call=True,
    )

    persisted = (tmp_path / "runpod_live_execution_proof.json").read_text(encoding="utf-8")
    assert result["status"] == "runpod_live_proof_collected"
    assert result["active_pod_count_before"] == 1
    assert result["active_pod_count_after"] == 0
    assert result["pod_stop_performed"] is True
    assert result["shutdown_or_termination_proof"] is True
    assert result["production_runpod_worker_execution_proven"] is False
    assert calls[0]["url"] == "https://rest.runpod.io/v1/pods"
    assert calls[1]["url"] == "https://rest.runpod.io/v1/pods/pod-123/stop"
    assert calls[2]["url"] == "https://rest.runpod.io/v1/pods"
    assert "secret-runpod-key" not in persisted


def test_runpod_live_execution_proof_combines_runtime_manifest_for_production_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _provider_launch_request(tmp_path / "gpu_provider_launch_request.json")
    adapter_result = tmp_path / "runpod_provider_adapter_result.live.json"
    _write_json(
        adapter_result,
        {
            "schema_version": "runpod_provider_adapter_result.v1",
            "job_id": "runpod-proof-job",
            "status": "submitted",
            "api_call_performed": True,
            "provider_job_submitted": True,
            "runpod_response": {"id": "pod-123"},
        },
    )
    runtime_manifest = tmp_path / "worker_runtime_manifest.json"
    _write_json(
        runtime_manifest,
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "job_id": "runpod-proof-job",
            "status": "completed",
            "blockers": [],
            "job_status": "simulator_command_completed",
            "job_blockers": [],
            "runtime_preflight_status": "passed",
            "runtime_preflight_blockers": [],
            "startup_architecture_audit_status": "passed",
            "startup_architecture_blockers": [],
            "scenario_eval_matrix_status": "completed",
            "simulator_service_status": "completed",
            "evaluation_status": "completed",
            "simulator_execution_proven": True,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "signed_put_runtime_manifest_upload": {"status": "completed"},
        },
    )
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_GPU_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")

    class FakeResponse:
        status = 200

        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return self.body

    calls: list[str] = []

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        calls.append(request.full_url)
        if request.full_url.endswith("/pods/pod-123/stop"):
            return FakeResponse(b'{"id":"pod-123","desiredStatus":"STOPPED"}')
        if len([url for url in calls if url.endswith("/pods")]) == 1:
            return FakeResponse(b'[{"id":"pod-123","desiredStatus":"RUNNING"}]')
        return FakeResponse(b'[{"id":"pod-123","desiredStatus":"STOPPED"}]')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = collect_runpod_live_execution_proof(
        provider_launch_request_path=request_path,
        adapter_result_path=adapter_result,
        runtime_manifest_path=runtime_manifest,
        output_path=tmp_path / "runpod_live_execution_proof.json",
        stop_pod=True,
        allow_runpod_api_call=True,
    )

    persisted = (tmp_path / "runpod_live_execution_proof.json").read_text(encoding="utf-8")
    assert result["status"] == "runpod_live_proof_collected"
    assert result["runtime_manifest_worker_completed"] is True
    assert result["shutdown_or_termination_proof"] is True
    assert result["production_runpod_worker_execution_proven"] is True
    assert result["simulator_execution_proven"] is True
    assert result["robot_readiness_proven"] is False
    assert result["public_claim_upgrade_allowed"] is False
    assert "secret-runpod-key" not in persisted


def test_runpod_live_execution_proof_cli_blocks_without_gates(tmp_path: Path, capsys) -> None:
    request_path = _provider_launch_request(tmp_path / "gpu_provider_launch_request.json")

    exit_code = main(["--provider-launch-request", str(request_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "status=blocked" in captured.out


def test_runpod_live_execution_proof_accepts_api_key_file_without_persisting_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _provider_launch_request(tmp_path / "gpu_provider_launch_request.json")
    api_key_file = tmp_path / "runpod.key"
    api_key_file.write_text("secret-runpod-key-from-file\n", encoding="utf-8")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_GPU_LAUNCH_GATE_ENV, "true")
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_API_KEY_FILE_ENV, str(api_key_file))
    calls: list[dict[str, object]] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b"[]"

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        calls.append({"headers": dict(request.header_items())})
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = collect_runpod_live_execution_proof(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "runpod_live_execution_proof.json",
        allow_runpod_api_call=True,
    )

    persisted = (tmp_path / "runpod_live_execution_proof.json").read_text(encoding="utf-8")
    assert result["status"] == "runpod_live_proof_collected"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-runpod-key-from-file"  # type: ignore[index]
    assert "secret-runpod-key-from-file" not in persisted


def test_runpod_live_execution_proof_accepts_runpod_config_without_persisting_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _provider_launch_request(tmp_path / "gpu_provider_launch_request.json")
    config_file = tmp_path / "config.toml"
    config_file.write_text('[default]\napi_key = "secret-runpod-key-from-config"\n', encoding="utf-8")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_GPU_LAUNCH_GATE_ENV, "true")
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_CONFIG_FILE_ENV, str(config_file))
    calls: list[dict[str, object]] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b"[]"

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        calls.append({"headers": dict(request.header_items())})
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = collect_runpod_live_execution_proof(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "runpod_live_execution_proof.json",
        allow_runpod_api_call=True,
    )

    persisted = (tmp_path / "runpod_live_execution_proof.json").read_text(encoding="utf-8")
    assert result["status"] == "runpod_live_proof_collected"
    assert result["api_key_source"] == RUNPOD_CONFIG_FILE_ENV
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-runpod-key-from-config"  # type: ignore[index]
    assert "secret-runpod-key-from-config" not in persisted
