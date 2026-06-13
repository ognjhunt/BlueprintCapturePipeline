from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

from blueprint_pipeline.runpod_provider_adapter import (
    RUNPOD_API_GATE_ENV,
    RUNPOD_API_KEY_ENV,
    RUNPOD_API_KEY_FILE_ENV,
    RUNPOD_CONFIG_FILE_ENV,
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
                    "configured_image_ref_fetchable_by_provider": True,
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
    pod_input = result["runpod_request"]["on_demand_pod"]["body"]  # type: ignore[index]
    assert pod_input["gpuTypeIds"] == ["NVIDIA RTX A6000"]
    assert pod_input["imageName"].endswith(":2026-06-12")
    assert pod_input["env"]["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert pod_input["env"]["BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS"] == "180"
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


def test_runpod_adapter_blocks_unfetchable_worker_image_ref(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["image"][  # type: ignore[index]
        "configured_image_ref_fetchable_by_provider"
    ] = False
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    assert result["status"] == "blocked"
    assert "prebuilt_worker_image_ref_not_provider_fetchable" in result["blockers"]


def test_runpod_adapter_uses_provider_gpu_priority_and_cache_env(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["gpu"]["provider_gpu_priority"] = [  # type: ignore[index]
        "NVIDIA L4",
        "NVIDIA RTX A4000",
    ]
    request["provider_request_shape"]["cache"] = {  # type: ignore[index]
        "paths": {
            "mujoco_assets": "/cache/mujoco",
            "policy_files": "/cache/policies",
            "converted_scenes": "/cache/scenes",
            "worker_deps": "/cache/deps",
        }
    }
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    pod = result["runpod_request"]["on_demand_pod"]["body"]  # type: ignore[index]
    assert pod["gpuTypeIds"] == ["NVIDIA L4"]
    assert pod["blueprintGpuTypePriority"] == ["NVIDIA L4", "NVIDIA RTX A4000"]
    assert pod["env"]["BLUEPRINT_MUJOCO_ASSET_CACHE"] == "/cache/mujoco"
    assert pod["env"]["BLUEPRINT_POLICY_CACHE"] == "/cache/policies"
    assert pod["env"]["BLUEPRINT_CONVERTED_SCENE_CACHE"] == "/cache/scenes"
    assert pod["env"]["BLUEPRINT_WORKER_DEPS_CACHE"] == "/cache/deps"


def test_runpod_adapter_blocks_unwritable_artifact_output_uri(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["inputs"]["artifact_output_uri"] = (  # type: ignore[index]
        "https://storage.example/output"
    )
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    assert result["status"] == "blocked"
    assert "provider_artifact_output_uri_not_writable" in result["blockers"]


def test_runpod_adapter_blocks_provider_input_setup_blockers(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["status"] = "blocked_provider_input_setup"
    request["provider_input_setup"] = {
        "status": "prepared_with_external_blockers",
        "blockers": ["upload_failed:Forbidden"],
    }
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "runpod_request_not_launchable"
    assert "provider_launch_request_not_ready" in result["blockers"]
    assert "provider_input_setup_blocked" in result["blockers"]
    assert "upload_failed:Forbidden" in result["blockers"]


def test_runpod_adapter_blocks_live_serverless_without_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.delenv(RUNPOD_API_GATE_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_CONFIG_FILE_ENV, str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv(RUNPOD_ENDPOINT_ID_ENV, "endpoint-123")

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="serverless-run",
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert f"missing_env_{RUNPOD_API_GATE_ENV}" in result["blockers"]
    assert "missing_cli_allow_runpod_api_call" in result["blockers"]
    assert (
        f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}_or_{RUNPOD_CONFIG_FILE_ENV}"
        in result["blockers"]
    )


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
    request = _read_json(request_path)
    signed_manifest_url = (
        "https://storage.googleapis.com/blueprint/worker.json?"
        "x-goog-signature=manifest-secret-signature&x-goog-date=20260612"
    )
    signed_capture_bundle_url = (
        "https://storage.googleapis.com/blueprint/capture-root.zip?"
        "x-goog-signature=bundle-secret-signature&x-goog-date=20260612"
    )
    request["provider_request_shape"]["inputs"][  # type: ignore[index]
        "manifest_uri"
    ] = signed_manifest_url
    request["provider_request_shape"]["inputs"][  # type: ignore[index]
        "capture_root_bundle_uri"
    ] = signed_capture_bundle_url
    request["provider_request_shape"]["runtime_preflight"] = {  # type: ignore[index]
        "simulator": "mujoco"
    }
    _write_json(request_path, request)
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    signed_put_url = (
        "https://storage.googleapis.com/blueprint/runtime.json?"
        "x-goog-signature=put-secret-signature&x-goog-date=20260612"
    )
    monkeypatch.setenv(
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
        signed_put_url,
    )

    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "pod-123",
                    "imageName": "registry.example/blueprint/isaac-eval-worker:2026-06-12",
                    "env": {
                        "BLUEPRINT_EVAL_MANIFEST_URI": signed_manifest_url,
                        "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI": signed_capture_bundle_url,
                        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": signed_put_url,
                    },
                }
            ).encode("utf-8")

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

    assert captured["url"] == "https://rest.runpod.io/v1/pods"
    pod_input = captured["body"]  # type: ignore[assignment]
    assert pod_input["name"] == "blueprint-test-pod"
    assert pod_input["imageName"].endswith(":2026-06-12")
    assert pod_input["gpuTypeIds"] == ["NVIDIA RTX A6000"]
    assert pod_input["env"]["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert pod_input["env"]["BLUEPRINT_ALLOW_GPU_PROVISIONING"] == "true"
    assert pod_input["env"]["BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"] == "true"
    assert pod_input["env"]["BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME"] == "true"
    assert pod_input["env"]["BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"].endswith(
        ":2026-06-12"
    )
    assert pod_input["env"]["BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF"].endswith(
        ":2026-06-12"
    )
    assert pod_input["env"]["BLUEPRINT_EVAL_MANIFEST_URI"] == signed_manifest_url
    assert pod_input["env"]["BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI"] == signed_capture_bundle_url
    assert (
        pod_input["env"]["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"]
        == signed_put_url
    )
    assert result["status"] == "submitted"
    assert result["provider_job_submitted"] is True
    assert result["provider_allocation_proven"] is False
    assert result["simulator_execution_proven"] is False
    persisted = (tmp_path / "runpod_provider_adapter_result.json").read_text(
        encoding="utf-8"
    )
    assert "manifest-secret-signature" not in persisted
    assert "bundle-secret-signature" not in persisted
    assert "put-secret-signature" not in persisted
    redacted_env = result["runpod_request"]["body"]["env"]  # type: ignore[index]
    assert (
        redacted_env["BLUEPRINT_EVAL_MANIFEST_URI"]
        == (
            "https://storage.googleapis.com/blueprint/worker.json?"
            "x-goog-signature=<redacted:signed-url-signature>&x-goog-date=20260612"
        )
    )
    assert redacted_env["BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI"] == (
        "https://storage.googleapis.com/blueprint/capture-root.zip?"
        "x-goog-signature=<redacted:signed-url-signature>&x-goog-date=20260612"
    )
    assert (
        redacted_env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"]
        == "<redacted:signed-url>"
    )
    response_env = result["runpod_response"]["env"]  # type: ignore[index]
    assert response_env["BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI"] == (
        "https://storage.googleapis.com/blueprint/capture-root.zip?"
        "x-goog-signature=<redacted:signed-url-signature>&x-goog-date=20260612"
    )
    assert (
        response_env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"]
        == "<redacted:signed-url>"
    )


def test_runpod_adapter_accepts_api_key_file_without_persisting_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    api_key_file = tmp_path / "runpod.key"
    api_key_file.write_text("secret-runpod-key-from-file\n", encoding="utf-8")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_API_KEY_FILE_ENV, str(api_key_file))

    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b'{"id":"pod-file"}'

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
        pod_name="blueprint-test-pod",
    )

    persisted = (tmp_path / "runpod_provider_adapter_result.json").read_text(
        encoding="utf-8"
    )
    assert captured["headers"]["Authorization"] == "Bearer secret-runpod-key-from-file"  # type: ignore[index]
    assert result["status"] == "submitted"
    assert "secret-runpod-key-from-file" not in persisted


def test_runpod_adapter_accepts_runpod_config_without_persisting_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    config_file = tmp_path / "config.toml"
    config_file.write_text('[default]\napi_key = "secret-runpod-key-from-config"\n', encoding="utf-8")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_CONFIG_FILE_ENV, str(config_file))

    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b'{"id":"pod-config"}'

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
        pod_name="blueprint-test-pod",
    )

    persisted = (tmp_path / "runpod_provider_adapter_result.json").read_text(
        encoding="utf-8"
    )
    assert captured["headers"]["Authorization"] == "Bearer secret-runpod-key-from-config"  # type: ignore[index]
    assert result["status"] == "submitted"
    assert result["api_key_source"] == RUNPOD_CONFIG_FILE_ENV
    assert "secret-runpod-key-from-config" not in persisted


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
