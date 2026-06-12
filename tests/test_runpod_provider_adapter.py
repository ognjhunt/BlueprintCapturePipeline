from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

from blueprint_pipeline.runpod_provider_adapter import (
    RUNPOD_API_GATE_ENV,
    RUNPOD_API_KEY_ENV,
    RUNPOD_ENDPOINT_ID_ENV,
    main as runpod_adapter_main,
    run_runpod_provider_adapter,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ready_runpod_request(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "runpod-adapter-job-1",
            "provider": "runpod",
            "status": "request_manifest_ready",
            "operation": "enqueue_runpod_serverless_or_on_demand_worker",
            "provider_request_shape": {
                "api_payload_is_provider_adapter_template": True,
                "api_payload_values_are_redacted": True,
                "operation": "enqueue_runpod_serverless_or_on_demand_worker",
                "image": {
                    "configured_image_ref": (
                        "registry.example/blueprint/isaac-eval-worker:2026-06-12"
                    ),
                    "configured_image_ref_is_versioned": True,
                },
                "command": (
                    "blueprint-run-robot-eval-worker "
                    "--manifest ${BLUEPRINT_EVAL_MANIFEST_URI}"
                ),
                "environment": {
                    "secret_env_var_names": ["RUNPOD_API_KEY"],
                    "secret_values_in_artifact": False,
                },
                "inputs": {
                    "manifest_uri_required_for_provider": True,
                    "manifest_uri": (
                        "r2://blueprint-artifacts/jobs/runpod-adapter-job-1/"
                        "worker_manifest.json"
                    ),
                    "manifest_uri_fetchable_by_provider": True,
                    "artifact_output_uri_required": True,
                    "artifact_output_uri": (
                        "r2://blueprint-artifacts/jobs/runpod-adapter-job-1"
                    ),
                },
                "gpu": {
                    "preferred_gpu_class": "NVIDIA RTX A6000",
                    "disallowed_gpu_classes": ["A100", "H100"],
                },
                "limits": {
                    "max_active_workers": 1,
                    "hard_timeout_seconds": 120,
                    "idle_timeout_seconds": 60,
                    "external_watchdog_ttl_seconds": 180,
                },
            },
        },
    )
    return path


def test_runpod_adapter_dry_run_writes_serverless_and_pod_shapes(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    persisted = _read_json(tmp_path / "runpod_provider_adapter_result.json")
    assert result["status"] == "dry_run_ready"
    assert result["api_call_performed"] is False
    assert result["secret_values_in_artifact"] is False
    assert result["raw_api_key_stored"] is False
    cost_policy = result["cost_control_policy"]
    assert cost_policy["hard_timeout_seconds"] == 120  # type: ignore[index]
    assert cost_policy["idle_timeout_seconds"] == 60  # type: ignore[index]
    assert cost_policy["external_watchdog_ttl_seconds"] == 180  # type: ignore[index]
    assert cost_policy["max_active_workers"] == 1  # type: ignore[index]
    assert cost_policy["serverless_endpoint_controls"][  # type: ignore[index]
        "idle_timeout_set_by_run_request"
    ] is False
    assert cost_policy["serverless_endpoint_controls"][  # type: ignore[index]
        "endpoint_level_settings_required"
    ] == [
        "active_workers",
        "max_workers",
        "idle_timeout",
        "execution_timeout",
        "job_ttl",
    ]
    assert cost_policy["on_demand_pod_controls"][  # type: ignore[index]
        "external_watchdog_or_owner_terminator_required"
    ] is True
    serverless = result["runpod_request"]["serverless_run"]  # type: ignore[index]
    assert serverless["url"] == "https://api.runpod.ai/v2/endpoint-123/run"
    assert serverless["body"]["input"]["worker_manifest_uri"].startswith("r2://")  # type: ignore[index]
    assert serverless["body"]["input"]["cost_control_policy"][  # type: ignore[index]
        "serverless_idle_timeout_requires_endpoint_setting"
    ] is True
    assert serverless["body"]["policy"]["executionTimeout"] == 120000  # type: ignore[index]
    assert serverless["body"]["policy"]["ttl"] == 180000  # type: ignore[index]
    pod_input = result["runpod_request"]["on_demand_pod"]["variables"]["input"]  # type: ignore[index]
    assert pod_input["gpuTypeId"] == "NVIDIA RTX A6000"
    assert pod_input["imageName"].endswith(":2026-06-12")
    assert {"key": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"} in pod_input["env"]
    assert {
        "key": "BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS",
        "value": "180",
    } in pod_input["env"]
    assert "RUNPOD_API_KEY" not in json.dumps(persisted)


def test_runpod_adapter_blocks_missing_cost_control_limits(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["limits"] = {}  # type: ignore[index]
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "runpod_request_not_launchable"
    assert "missing_provider_hard_timeout_seconds" in result["blockers"]
    assert "missing_provider_idle_timeout_seconds" in result["blockers"]
    assert "missing_provider_external_watchdog_ttl_seconds" in result["blockers"]
    assert "missing_provider_max_active_workers" in result["blockers"]


def test_runpod_adapter_blocks_live_serverless_without_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.delenv(RUNPOD_API_GATE_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_ENDPOINT_ID_ENV, "endpoint-123")

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="serverless-run",
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert f"missing_env_{RUNPOD_API_GATE_ENV}" in result["blockers"]
    assert "missing_cli_allow_runpod_api_call" in result["blockers"]
    assert f"missing_env_{RUNPOD_API_KEY_ENV}" in result["blockers"]


def test_runpod_adapter_submits_serverless_run_with_redacted_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        raise HTTPError(
            request.full_url,
            401,
            "unauthorized secret-runpod-key",
            hdrs=None,
            fp=SimpleNamespace(read=lambda: b"bad secret-runpod-key"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="serverless-run",
        allow_runpod_api_call=True,
        endpoint_id="endpoint-123",
    )

    persisted = (tmp_path / "runpod_provider_adapter_result.json").read_text(
        encoding="utf-8"
    )
    assert captured["url"] == "https://api.runpod.ai/v2/endpoint-123/run"
    assert captured["headers"]["Authorization"] == "Bearer secret-runpod-key"  # type: ignore[index]
    assert captured["body"]["input"]["job_id"] == "runpod-adapter-job-1"  # type: ignore[index]
    assert result["status"] == "failed"
    assert result["api_call_performed"] is True
    assert result["runpod_side_effects_may_have_occurred"] is True
    assert "<redacted:RUNPOD_API_KEY>" in result["runpod_error"]
    assert "secret-runpod-key" not in persisted


def test_runpod_adapter_submits_on_demand_pod_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")

    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b'{"data":{"podFindAndDeployOnDemand":{"id":"pod-123"}}}'

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
        pod_name="blueprint-test-pod",
    )

    assert captured["url"] == "https://api.runpod.io/graphql"
    assert "podFindAndDeployOnDemand" in captured["body"]["query"]  # type: ignore[index]
    pod_input = captured["body"]["variables"]["input"]  # type: ignore[index]
    assert pod_input["name"] == "blueprint-test-pod"
    assert pod_input["imageName"].endswith(":2026-06-12")
    assert pod_input["gpuTypeId"] == "NVIDIA RTX A6000"
    assert {"key": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"} in pod_input["env"]
    assert result["status"] == "submitted"
    assert result["provider_job_submitted"] is True
    assert result["provider_allocation_proven"] is False
    assert result["simulator_execution_proven"] is False


def test_runpod_adapter_cli_defaults_to_dry_run(
    tmp_path: Path,
    capsys,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")

    exit_code = runpod_adapter_main(
        [
            "--provider-launch-request",
            str(request_path),
            "--endpoint-id",
            "endpoint-123",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status=dry_run_ready" in captured.out
    assert (tmp_path / "runpod_provider_adapter_result.json").is_file()


def test_runpod_adapter_requires_endpoint_for_serverless(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="serverless-run",
    )

    assert result["status"] == "blocked"
    assert f"missing_env_{RUNPOD_ENDPOINT_ID_ENV}" in result["blockers"]
