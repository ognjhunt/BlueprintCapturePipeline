from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from blueprint_pipeline import runpod_provider_adapter as adapter
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.runpod_provider_adapter import (
    RUNPOD_API_GATE_ENV,
    RUNPOD_API_KEY_ENV,
    RUNPOD_API_KEY_FILE_ENV,
    RUNPOD_CONFIG_FILE_ENV,
    RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV,
    RUNPOD_ENDPOINT_ID_ENV,
    RUNPOD_EXISTING_POD_ID_ENV,
    main as runpod_adapter_main,
    run_runpod_provider_adapter,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paid_grant() -> PaidResourceAdmissionGrant:
    admission = build_paid_lane_admission(
        resource_class="runpod_provider_adapter",
        blockers=[],
    )
    return require_paid_resource_admission(
        admission,
        resource_class="runpod_provider_adapter",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def test_runpod_http_transport_rejects_non_runpod_origins_before_network() -> None:
    with pytest.raises(
        adapter.safe_outbound_http.SafeOutboundHttpError,
        match="outbound_http_host_not_allowed",
    ):
        adapter._http_json(
            url="https://rest.runpod.io.evil.example/v1/pods",
            payload={},
            api_key="secret-runpod-key",
            timeout_seconds=5,
        )


def test_runpod_http_policy_allows_configured_https_rest_origin(monkeypatch) -> None:
    monkeypatch.setattr(
        adapter, "RUNPOD_REST_API_BASE", "https://runpod-proxy.example/v1"
    )

    policy = adapter._runpod_provider_api_policy()

    assert "runpod-proxy.example" in policy.allowed_hosts


@pytest.mark.parametrize(
    "base",
    (
        "http://runpod-proxy.example/v1",
        "https://user:secret@runpod-proxy.example/v1",
        "runpod-proxy.example/v1",
    ),
)
def test_runpod_http_policy_rejects_unsafe_configured_rest_origin(
    monkeypatch, base: str
) -> None:
    monkeypatch.setattr(adapter, "RUNPOD_REST_API_BASE", base)

    with pytest.raises(
        ValueError, match="RUNPOD_REST_API_BASE_must_be_credential_free_https_origin"
    ):
        adapter._runpod_provider_api_policy()


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
                    "customer_visible_secret_values_allowed": False,
                },
                "inputs": {
                    "manifest_uri_required_for_provider": True,
                    "manifest_uri": (
                        "r2://blueprint-artifacts/jobs/runpod-adapter-job-1/"
                        "worker_manifest.json"
                    ),
                    "manifest_uri_fetchable_by_provider": True,
                    "capture_root_bundle_uri_required_for_provider": True,
                    "capture_root_bundle_uri": (
                        "r2://blueprint-artifacts/jobs/runpod-adapter-job-1/"
                        "capture-root.zip"
                    ),
                    "capture_root_bundle_uri_fetchable_by_provider": True,
                    "artifact_output_uri_required": True,
                    "artifact_output_uri": (
                        "r2://blueprint-artifacts/jobs/runpod-adapter-job-1"
                    ),
                    "artifact_output_uri_scheme": "r2",
                    "artifact_output_uri_provider_writable": True,
                    "artifact_output_write_auth_contract_ready": True,
                    "artifact_output_write_auth": {
                        "write_auth_contract_ready": True,
                        "secret_values_in_artifact": False,
                    },
                },
                "gpu": {
                    "preferred_gpu_class": "NVIDIA RTX A6000",
                    "disallowed_gpu_classes": ["A100", "H100"],
                },
                "limits": {
                    "max_active_workers": 1,
                    "requested_budget_usd": 0.25,
                    "hard_timeout_seconds": 120,
                    "idle_timeout_seconds": 60,
                    "startup_artifact_watchdog_required": True,
                    "startup_artifact_timeout_seconds": 90,
                    "idle_shutdown_required": True,
                    "external_watchdog_ttl_required": True,
                    "external_watchdog_ttl_seconds": 180,
                    "external_watchdog_owner": "provider_launcher_or_owner_control_plane",
                    "scale_to_zero_default": True,
                },
                "artifact_finalizer": {
                    "upload_before_shutdown_required": True,
                    "record_actual_gpu_time_required": True,
                },
                "local_sim_only_prerequisite": {
                    "schema_version": "robot_eval_provider_local_sim_only_prerequisite.v1",
                    "required_before_provider_spend": True,
                    "status": "passed",
                    "source_artifact": "robot_team_grade_eval_closure_manifest.json",
                    "local_sim_only_evidence_clean": True,
                    "sim_only_beta_core_complete": True,
                    "sim_only_beta_blocked_requirement_ids": [],
                    "blockers": [],
                    "claim_boundary": {
                        "provider_spend_requires_local_sim_only_evidence_clean": True,
                        "local_sim_only_clean_does_not_prove_remote_provider_execution": True,
                        "local_sim_only_clean_does_not_prove_launch_approval": True,
                    },
                },
            },
        },
    )
    return path


def test_runpod_adapter_dry_run_writes_serverless_and_pod_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_FILE_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_CONFIG_FILE_ENV, raising=False)
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
    endpoint_manifest_path = Path(result["provider_worker_endpoint_manifest_path"])
    endpoint_manifest = _read_json(endpoint_manifest_path)
    assert endpoint_manifest_path == tmp_path / "provider_worker_endpoint_manifest.json"
    assert endpoint_manifest["provider"] == "runpod"
    assert endpoint_manifest["provider_mode"] == "serverless-run"
    assert endpoint_manifest["worker_invocation_grain"] == "evaluation_job_provider_submission"
    assert endpoint_manifest["direct_policy_infer_from_local_loop_allowed"] is False
    assert endpoint_manifest["known_endpoint"]["serverless_endpoint_id_present"] is True
    assert (
        endpoint_manifest["consumer_env_contract"]["worker_url_env"]
        == "BLUEPRINT_PROVIDER_POLICY_WORKER_URL"
    )
    assert endpoint_manifest == result["provider_worker_endpoint_manifest"]
    readiness_path = Path(result["provider_readiness_manifest_path"])
    readiness = _read_json(readiness_path)
    assert readiness_path == tmp_path / "runpod_provider_readiness_manifest.json"
    assert readiness["schema_version"] == "runpod_provider_readiness_manifest.v1"
    assert readiness["status"] == "ready_for_explicit_paid_provider_attempt"
    assert readiness["api_call_performed"] is False
    assert readiness["live_provider_call_authorized"] is False
    assert readiness["spend_limits"]["requested_budget_usd"] == 0.25  # type: ignore[index]
    assert readiness["spend_limits"]["bounded_single_worker_attempt"] is True  # type: ignore[index]
    assert readiness["spend_limits"]["startup_artifact_timeout_seconds"] == 90  # type: ignore[index]
    assert readiness["image_startup_diagnostic"]["large_image_pull_risk"] is False  # type: ignore[index]
    assert readiness["image_startup_diagnostic"]["metadata_present"] is False  # type: ignore[index]
    provider_inputs = readiness["provider_inputs"]  # type: ignore[index]
    assert provider_inputs["manifest_uri_present"] is True
    assert provider_inputs["manifest_uri_fetchable_by_provider"] is True
    assert provider_inputs["capture_root_bundle_uri_present"] is True
    assert (
        provider_inputs["capture_root_bundle_uri_fetchable_by_provider"]
        is True
    )
    assert (
        readiness["artifact_output"]["artifact_output_uri_provider_writable"]  # type: ignore[index]
        is True
    )
    assert (
        readiness["artifact_output"][  # type: ignore[index]
            "artifact_output_uri_scheme_provider_writable"
        ]
        is True
    )
    assert (
        readiness["watchdog_and_teardown"]["idle_shutdown_required"]  # type: ignore[index]
        is True
    )
    assert (
        readiness["watchdog_and_teardown"][  # type: ignore[index]
            "startup_artifact_watchdog_required"
        ]
        is True
    )
    assert readiness["watchdog_and_teardown"][  # type: ignore[index]
        "startup_artifact_timeout_seconds"
    ] == 90
    assert readiness["watchdog_and_teardown"][  # type: ignore[index]
        "external_watchdog_ttl_exceeds_hard_timeout"
    ] is True
    assert (
        readiness["watchdog_and_teardown"][  # type: ignore[index]
            "upload_before_shutdown_required"
        ]
        is True
    )
    assert (
        readiness["no_secret_artifact_policy"]["secret_values_in_artifact"]  # type: ignore[index]
        is False
    )
    claim_boundary = readiness["claim_boundary"]  # type: ignore[index]
    assert claim_boundary["optional_provider_runtime_evidence_only"] is True
    assert claim_boundary[
        "not_sim_only_launch_proof_until_artifacts_imported_and_reviewed"
    ] is True
    assert readiness == result["provider_readiness_manifest"]
    cost_policy = result["cost_control_policy"]
    assert cost_policy["hard_timeout_seconds"] == 120  # type: ignore[index]
    assert cost_policy["idle_timeout_seconds"] == 60  # type: ignore[index]
    assert cost_policy["startup_artifact_timeout_seconds"] == 90  # type: ignore[index]
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
    assert cost_policy["on_demand_pod_controls"][  # type: ignore[index]
        "startup_artifact_watchdog_required"
    ] is True
    assert result["image_startup_diagnostic"]["large_image_pull_risk"] is False  # type: ignore[index]
    serverless = result["runpod_request"]["serverless_run"]  # type: ignore[index]
    assert serverless["url"] == "https://api.runpod.ai/v2/endpoint-123/run"
    assert serverless["body"]["input"][  # type: ignore[index]
        "worker_manifest_uri"
    ].startswith("r2://")
    assert serverless["body"]["input"]["capture_root_bundle_uri"].endswith(  # type: ignore[index]
        "capture-root.zip"
    )
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


def test_runpod_pod_payload_attaches_existing_volume_and_cuda_dc_constraints(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "request.json")
    request = _read_json(request_path)
    shape = request["provider_request_shape"]
    assert isinstance(shape, dict)
    shape.update(
        {
            "network_volume_id": "volume-1",
            "data_center_id": "US-TX-3",
            "allowed_cuda_versions": ["12.6"],
        }
    )
    _write_json(request_path, request)
    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path, mode="dry-run"
    )
    body = result["runpod_request"]["on_demand_pod"]["body"]
    assert body["networkVolumeId"] == "volume-1"
    assert body["dataCenterIds"] == ["US-TX-3"]
    assert body["allowedCudaVersions"] == ["12.6"]


def test_runpod_adapter_blocks_when_prelaunch_spend_guard_is_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_FILE_ENV, raising=False)
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["prelaunch_spend_guard"] = {
        "schema_version": "robot_eval_provider_prelaunch_spend_guard.v1",
        "required_before_provider_launch": True,
        "status": "blocked",
        "can_launch": False,
        "blockers": ["prelaunch_local_sim_only_prerequisite_not_passed"],
    }
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    assert result["status"] == "blocked"
    assert "provider_prelaunch_spend_guard_not_passed" in result["blockers"]
    assert "prelaunch_local_sim_only_prerequisite_not_passed" in result["blockers"]
    readiness = _read_json(tmp_path / "runpod_provider_readiness_manifest.json")
    assert readiness["status"] == "blocked_before_paid_provider_attempt"


def test_runpod_adapter_forwards_docker_entrypoint_to_pod_payload(tmp_path: Path) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["docker_entrypoint"] = ["bash"]  # type: ignore[index]
    request["provider_request_shape"]["docker_start_cmd"] = [  # type: ignore[index]
        "-lc",
        "echo provider-heartbeat",
    ]
    request["provider_request_shape"]["command"] = "echo provider-heartbeat"  # type: ignore[index]
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    pod = result["runpod_request"]["on_demand_pod"]["body"]  # type: ignore[index]
    assert pod["dockerEntrypoint"] == ["bash"]
    assert pod["dockerStartCmd"] == ["-lc", "echo provider-heartbeat"]


def test_runpod_adapter_forwards_allowed_secret_env_with_redaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    secret_value = "secret-gcp-json-b64"
    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_FORWARD_SECRET_ENV_VARS",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON_B64",
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_JSON_B64", secret_value)
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")

    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return None

        def read(self) -> bytes:
            return json.dumps({"id": "pod-secret-env"}).encode()

    def fake_urlopen(request, timeout, policy):  # type: ignore[no-untyped-def]
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy", fake_urlopen
    )

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
        paid_resource_admission_grant=_paid_grant(),
    )
    persisted = _read_json(tmp_path / "runpod_provider_adapter_result.json")

    outbound = captured["body"]  # type: ignore[assignment]
    persisted_pod = persisted["runpod_request"]["body"]  # type: ignore[index]
    assert outbound["env"]["GOOGLE_APPLICATION_CREDENTIALS_JSON_B64"] == secret_value  # type: ignore[index]
    assert persisted_pod["env"]["GOOGLE_APPLICATION_CREDENTIALS_JSON_B64"] == (  # type: ignore[index]
        "<redacted:secret-env>"
    )
    assert result["status"] == "submitted"
    assert secret_value not in json.dumps(persisted)


def test_runpod_adapter_blocks_missing_shared_admission_before_api_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")

    def api_must_not_run(*_args, **_kwargs):
        raise AssertionError("RunPod mutation reached without shared admission")

    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy",
        api_must_not_run,
    )
    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert "runpod_provider_shared_admission_missing_or_invalid" in result["blockers"]


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


def test_runpod_adapter_blocks_missing_worker_image_without_internal_error(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["image"]["configured_image_ref"] = ""  # type: ignore[index]
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "runpod_request_not_launchable"
    assert "missing_provider_worker_image_ref" in result["blockers"]


def test_runpod_adapter_blocks_missing_or_unfetchable_capture_bundle_uri(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["inputs"][  # type: ignore[index]
        "capture_root_bundle_uri"
    ] = ""
    request["provider_request_shape"]["inputs"][  # type: ignore[index]
        "capture_root_bundle_uri_fetchable_by_provider"
    ] = False
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )
    readiness = _read_json(tmp_path / "runpod_provider_readiness_manifest.json")

    assert result["status"] == "blocked"
    assert "missing_provider_capture_root_bundle_uri" in result["blockers"]
    assert "provider_capture_root_bundle_uri_not_fetchable" in result["blockers"]
    assert readiness["provider_inputs"]["capture_root_bundle_uri_present"] is False
    assert (
        readiness["provider_inputs"]["capture_root_bundle_uri_fetchable_by_provider"]
        is False
    )


def test_runpod_adapter_blocks_missing_or_blocked_local_sim_only_prerequisite(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    del request["provider_request_shape"]["local_sim_only_prerequisite"]  # type: ignore[index]
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    assert result["status"] == "blocked"
    assert "missing_local_sim_only_provider_prerequisite" in result["blockers"]

    request["provider_request_shape"]["local_sim_only_prerequisite"] = {  # type: ignore[index]
        "schema_version": "robot_eval_provider_local_sim_only_prerequisite.v1",
        "required_before_provider_spend": True,
        "status": "blocked",
        "source_artifact": "robot_team_grade_eval_closure_manifest.json",
        "local_sim_only_evidence_clean": False,
        "sim_only_beta_core_complete": False,
        "sim_only_beta_blocked_requirement_ids": ["failure_diagnosis"],
        "blockers": [
            "local_sim_only_evidence_not_clean",
            "sim_only_beta_requirement_failure_diagnosis_not_complete",
        ],
    }
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )
    readiness = _read_json(tmp_path / "runpod_provider_readiness_manifest.json")

    assert result["status"] == "blocked"
    assert "local_sim_only_provider_prerequisite_not_passed" in result["blockers"]
    assert "local_sim_only_evidence_not_clean" in result["blockers"]
    assert readiness["local_sim_only_prerequisite"]["status"] == "blocked"
    assert (
        readiness["local_sim_only_prerequisite"]["local_sim_only_evidence_clean"]
        is False
    )


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
            "groot_oscar_models": (
                "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1"
            ),
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
    assert "blueprintGpuTypePriority" not in pod
    assert pod["env"]["BLUEPRINT_MUJOCO_ASSET_CACHE"] == "/cache/mujoco"
    assert pod["env"]["BLUEPRINT_POLICY_CACHE"] == "/cache/policies"
    assert pod["env"]["BLUEPRINT_CONVERTED_SCENE_CACHE"] == "/cache/scenes"
    assert pod["env"]["BLUEPRINT_WORKER_DEPS_CACHE"] == "/cache/deps"
    assert pod["env"]["BLUEPRINT_GROOT_OSCAR_MODEL_CACHE"] == (
        "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1"
    )


def test_runpod_adapter_forwards_declared_plaintext_env_values(tmp_path: Path) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["environment"].update(  # type: ignore[index]
        {
            "plaintext_env_var_names": ["ACCEPT_EULA", "PRIVACY_CONSENT"],
            "plaintext_env_values": {
                "ACCEPT_EULA": "Y",
                "PRIVACY_CONSENT": "Y",
                "UNDECLARED_ENV": "ignored",
            },
            "secret_env_var_names": ["RUNPOD_API_KEY"],
        }
    )
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    pod = result["runpod_request"]["on_demand_pod"]["body"]  # type: ignore[index]
    assert pod["env"]["ACCEPT_EULA"] == "Y"
    assert pod["env"]["PRIVACY_CONSENT"] == "Y"
    assert "UNDECLARED_ENV" not in pod["env"]
    assert "RUNPOD_API_KEY" not in json.dumps(pod)


def test_runpod_adapter_forwards_container_registry_auth_id_without_secret(
    tmp_path: Path, monkeypatch
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["image"][  # type: ignore[index]
        "configured_image_ref"
    ] = "nvcr.io/nvidia/isaac-sim:5.1.0"
    monkeypatch.setenv(RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV, "registry-auth-123")
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    pod = result["runpod_request"]["on_demand_pod"]["body"]  # type: ignore[index]
    assert pod["imageName"] == "nvcr.io/nvidia/isaac-sim:5.1.0"
    assert pod["containerRegistryAuthId"] == "registry-auth-123"
    assert "NGC_API_KEY" not in json.dumps(pod)


def test_runpod_adapter_supports_signed_put_only_runtime_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["inputs"][  # type: ignore[index]
        "artifact_output_uri_required"
    ] = False
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    assert result["status"] == "blocked"
    assert "missing_runtime_manifest_signed_put_url_for_artifact_output_optional" in result[
        "blockers"
    ]

    monkeypatch.setenv(
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
        "https://storage.googleapis.com/blueprint/runtime.json?"
        + ("x-goog-" + "signature=put-secret"),
    )
    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )

    assert result["status"] == "dry_run_ready"
    pod = result["runpod_request"]["on_demand_pod"]["body"]  # type: ignore[index]
    assert "BLUEPRINT_ARTIFACT_OUTPUT_URI" not in pod["env"]
    assert (
        pod["env"]["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"]
        == "<redacted:signed-url>"
    )


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


def test_runpod_adapter_blocks_local_artifact_output_for_required_remote_runtime(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["inputs"]["artifact_output_uri"] = (  # type: ignore[index]
        "file:///tmp/blueprint-provider-output"
    )
    request["provider_request_shape"]["inputs"][  # type: ignore[index]
        "artifact_output_uri_scheme"
    ] = "file"
    request["provider_request_shape"]["inputs"][  # type: ignore[index]
        "artifact_output_uri_provider_writable"
    ] = True
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )
    readiness = _read_json(tmp_path / "runpod_provider_readiness_manifest.json")

    assert result["status"] == "blocked"
    assert "provider_artifact_output_uri_not_writable" in result["blockers"]
    assert readiness["artifact_output"]["artifact_output_uri_scheme"] == "file"
    assert (
        readiness["artifact_output"]["artifact_output_uri_scheme_provider_writable"]
        is False
    )


def test_runpod_adapter_blocks_artifact_output_uri_not_marked_writable(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["inputs"][  # type: ignore[index]
        "artifact_output_uri_provider_writable"
    ] = False
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="dry-run",
        endpoint_id="endpoint-123",
    )
    readiness = _read_json(tmp_path / "runpod_provider_readiness_manifest.json")

    assert result["status"] == "blocked"
    assert "provider_artifact_output_uri_not_marked_writable" in result["blockers"]
    assert readiness["status"] == "blocked_before_paid_provider_attempt"
    assert "provider_artifact_output_uri_not_marked_writable" in readiness["blockers"]


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

    def fake_urlopen(request, timeout, policy):  # type: ignore[no-untyped-def]
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

    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy", fake_urlopen
    )

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="serverless-run",
        allow_runpod_api_call=True,
        endpoint_id="endpoint-123",
        paid_resource_admission_grant=_paid_grant(),
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
    signed_url_signature_param = "x-goog-" + "signature="
    signed_manifest_url = (
        "https://storage.googleapis.com/blueprint/worker.json?"
        f"{signed_url_signature_param}manifest-secret-signature&x-goog-date=20260612"
    )
    signed_capture_bundle_url = (
        "https://storage.googleapis.com/blueprint/capture-root.zip?"
        f"{signed_url_signature_param}bundle-secret-signature&x-goog-date=20260612"
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
        f"{signed_url_signature_param}put-secret-signature&x-goog-date=20260612"
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

    def fake_urlopen(request, timeout, policy):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy", fake_urlopen
    )

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
        pod_name="blueprint-test-pod",
        paid_resource_admission_grant=_paid_grant(),
    )

    assert captured["url"] == "https://rest.runpod.io/v1/pods"
    pod_input = captured["body"]  # type: ignore[assignment]
    assert pod_input["name"] == "blueprint-test-pod"
    assert pod_input["imageName"].endswith(":2026-06-12")
    assert pod_input["gpuTypeIds"] == ["NVIDIA RTX A6000"]
    assert "blueprintGpuTypePriority" not in pod_input
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
    assert signed_url_signature_param not in persisted
    redacted_env = result["runpod_request"]["body"]["env"]  # type: ignore[index]
    assert (
        redacted_env["BLUEPRINT_EVAL_MANIFEST_URI"]
        == (
            "https://storage.googleapis.com/blueprint/worker.json?"
            "x-goog-redacted-signature-param=<redacted:signed-url-signature>"
            "&x-goog-date=20260612"
        )
    )
    assert redacted_env["BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI"] == (
        "https://storage.googleapis.com/blueprint/capture-root.zip?"
        "x-goog-redacted-signature-param=<redacted:signed-url-signature>"
        "&x-goog-date=20260612"
    )
    assert (
        redacted_env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"]
        == "<redacted:signed-url>"
    )
    response_env = result["runpod_response"]["env"]  # type: ignore[index]
    assert response_env["BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI"] == (
        "https://storage.googleapis.com/blueprint/capture-root.zip?"
        "x-goog-redacted-signature-param=<redacted:signed-url-signature>"
        "&x-goog-date=20260612"
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

    def fake_urlopen(request, timeout, policy):  # type: ignore[no-untyped-def]
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy", fake_urlopen
    )

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
        pod_name="blueprint-test-pod",
        paid_resource_admission_grant=_paid_grant(),
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

    def fake_urlopen(request, timeout, policy):  # type: ignore[no-untyped-def]
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy", fake_urlopen
    )

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
        pod_name="blueprint-test-pod",
        paid_resource_admission_grant=_paid_grant(),
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


def test_runpod_adapter_helper_edges_and_config_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert adapter._number(True) is None
    assert adapter._number("12.5") == 12.5
    assert adapter._number("not-a-number") is None
    assert adapter._bool("yes") is True
    assert adapter._bool("off") is False
    assert adapter._bool("maybe") is None
    assert adapter._string_list("one") == ["one"]
    assert adapter._redact_runtime_value(
        ("https://example.test/file?x-goog-signature=secret", {"MY_TOKEN": "secret"})
    ) == [
        "https://example.test/file?x-goog-redacted-signature-param=<redacted:signed-url-signature>",
        {"MY_TOKEN": "<redacted:secret-env>"},
    ]

    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_API_KEY_FILE_ENV, str(tmp_path / "missing-key"))
    key, meta = adapter._read_runpod_api_key()
    assert key == ""
    assert meta["api_key_file_read_error"] == "FileNotFoundError"

    bad_config = tmp_path / "bad-runpod.toml"
    bad_config.write_text("[default\n", encoding="utf-8")
    monkeypatch.delenv(RUNPOD_API_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_CONFIG_FILE_ENV, str(bad_config))
    key, meta = adapter._read_runpod_api_key()
    assert key == ""
    assert meta["api_key_config_file_read_error"] == "TOMLDecodeError"


def test_runpod_adapter_import_remains_safe_without_toml_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[default]\napi_key = "must-not-be-read"\n', encoding="utf-8")
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_API_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_CONFIG_FILE_ENV, str(config_file))
    monkeypatch.setattr(adapter, "tomllib", None)

    key, meta = adapter._read_runpod_api_key()

    assert key == ""
    assert meta["api_key_config_file_read_error"] == "TOMLParserUnavailable"


def test_runpod_adapter_pod_env_filters_plaintext_and_forwarded_secret_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["runtime_preflight"] = {"simulator": "isaac_sim"}  # type: ignore[index]
    request["provider_request_shape"]["environment"] = {  # type: ignore[index]
        "secret_env_var_names": ["SKIP_SECRET"],
        "plaintext_env_var_names": ["ALLOWED", "EMPTY", "SIGNED", "SKIP_SECRET"],
        "plaintext_env_values": {
            "ALLOWED": "value",
            "EMPTY": "",
            "SIGNED": "https://example.test/file?x-goog-signature=secret",
            "SKIP_SECRET": "secret",
            "IGNORED": "not-allowed",
        },
        "secret_values_in_artifact": False,
    }
    monkeypatch.setenv(adapter.RUNPOD_FORWARD_SECRET_ENV_VARS_ENV, "NOTSENSITIVE,MY_TOKEN")
    monkeypatch.setenv("NOTSENSITIVE", "visible")
    monkeypatch.setenv("MY_TOKEN", "secret-token")

    env = {item["key"]: item["value"] for item in adapter._pod_env(request)}

    assert env["ALLOWED"] == "value"
    assert env["MY_TOKEN"] == "secret-token"
    assert env["BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF"].endswith(":2026-06-12")
    assert "EMPTY" not in env
    assert "SIGNED" not in env
    assert "SKIP_SECRET" not in env
    assert "IGNORED" not in env
    assert "NOTSENSITIVE" not in env


def test_runpod_adapter_pod_payload_uses_entrypoint_with_command_start(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    request["provider_request_shape"]["docker_entrypoint"] = ["bash", "-lc"]  # type: ignore[index]
    request["provider_request_shape"]["command"] = "python worker.py --once"  # type: ignore[index]

    pod = adapter._pod_payload(request)["body"]

    assert pod["dockerEntrypoint"] == ["bash", "-lc"]
    assert pod["dockerStartCmd"] == ["python worker.py --once"]


def test_runpod_adapter_request_blocker_variants() -> None:
    bad_request = {
        "schema_version": "bad",
        "provider": "vast",
        "status": "draft",
        "provider_request_shape": {
            "image": {"configured_image_ref": ""},
            "inputs": {
                "manifest_uri": "",
                "manifest_uri_fetchable_by_provider": False,
                "artifact_output_uri": "",
                "artifact_output_uri_required": True,
            },
            "limits": {
                "hard_timeout_seconds": 120,
                "idle_timeout_seconds": 60,
                "external_watchdog_ttl_seconds": 120,
                "max_active_workers": 1,
            },
            "environment": {"secret_values_in_artifact": True},
        },
    }

    blockers = adapter._request_blockers(
        request=bad_request,
        mode="on-demand-pod",
        endpoint_id="",
    )

    assert blockers == [
        "invalid_provider_launch_request_schema",
        "provider_launch_request_not_runpod",
        "provider_launch_request_not_ready",
        "missing_provider_worker_image_ref",
        "missing_provider_worker_manifest_uri",
        "provider_worker_manifest_uri_not_fetchable",
        "missing_provider_capture_root_bundle_uri",
        "provider_capture_root_bundle_uri_not_fetchable",
        "missing_provider_artifact_output_uri",
        "provider_external_watchdog_ttl_must_exceed_hard_timeout",
        "missing_provider_requested_budget_usd",
        "provider_idle_shutdown_not_required",
        "provider_external_watchdog_owner_missing",
        "provider_artifact_upload_before_shutdown_not_required",
        "provider_launch_request_secret_values_in_artifact",
    ]
    unversioned = {
        "schema_version": "robot_eval_gpu_provider_launch_request.v1",
        "provider": "runpod",
        "status": "request_manifest_ready",
        "provider_request_shape": {
            "image": {
                "configured_image_ref": "registry/worker:latest",
                "configured_image_ref_is_versioned": False,
            },
            "inputs": {
                "manifest_uri": "r2://bucket/worker.json",
                "manifest_uri_fetchable_by_provider": True,
                "capture_root_bundle_uri": "r2://bucket/capture-root.zip",
                "capture_root_bundle_uri_fetchable_by_provider": True,
                "artifact_output_uri": "r2://bucket/artifacts",
                "artifact_output_uri_provider_writable": True,
                "artifact_output_write_auth_contract_ready": True,
            },
            "limits": {
                "hard_timeout_seconds": 120,
                "idle_timeout_seconds": 60,
                "idle_shutdown_required": True,
                "external_watchdog_ttl_required": True,
                "external_watchdog_ttl_seconds": 180,
                "external_watchdog_owner": "provider_launcher_or_owner_control_plane",
                "max_active_workers": 1,
                "requested_budget_usd": 0.25,
            },
            "artifact_finalizer": {"upload_before_shutdown_required": True},
            "local_sim_only_prerequisite": {
                "required_before_provider_spend": True,
                "status": "passed",
                "local_sim_only_evidence_clean": True,
                "blockers": [],
            },
            "environment": {"secret_values_in_artifact": False},
        },
    }
    assert adapter._request_blockers(
        request=unversioned,
        mode="on-demand-pod",
        endpoint_id="",
    ) == ["prebuilt_worker_image_ref_not_versioned"]


def test_runpod_adapter_blocks_invalid_json_auto_mode_and_unsupported_mode(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[]", encoding="utf-8")
    invalid = run_runpod_provider_adapter(provider_launch_request_path=invalid_path)
    assert invalid["status"] == "blocked"
    assert invalid["blockers"] == ["invalid_provider_launch_request_json"]

    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    auto = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "auto.json",
        mode="auto",
        endpoint_id="endpoint-123",
    )
    assert auto["mode"] == "serverless-run"
    assert auto["reason"] == "runpod_api_gate_blocked"

    unsupported = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "unsupported.json",
        mode="unexpected",
    )
    assert unsupported["status"] == "blocked"
    assert unsupported["blockers"] == ["unsupported_runpod_adapter_mode:unexpected"]


def test_runpod_adapter_empty_http_response_is_submitted_with_empty_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")

    class FakeResponse:
        status = 202

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def read(self) -> bytes:
            return b""

    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy",
        lambda request, timeout, policy: FakeResponse(),
    )

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        mode="on-demand-pod",
        allow_runpod_api_call=True,
        paid_resource_admission_grant=_paid_grant(),
    )

    assert result["status"] == "submitted"
    assert result["http_status_code"] == 202
    assert result["runpod_response"] == {}


def test_runpod_adapter_surfaces_large_worker_image_startup_risk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    image = request["provider_request_shape"]["image"]  # type: ignore[index]
    assert isinstance(image, dict)
    image["image_size_diagnostic"] = {
        "total_compressed_size_bytes": 13_500_000_000,
        "layers": [
            {"digest": "sha256:small", "size": 120_000_000},
            {"digest": "sha256:isaac-layer", "size": 10_600_000_000},
        ],
    }
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "large-image.json",
        mode="dry-run",
    )

    diagnostic = result["image_startup_diagnostic"]
    assert diagnostic["metadata_present"] is True  # type: ignore[index]
    assert diagnostic["large_image_pull_risk"] is True  # type: ignore[index]
    assert diagnostic["total_compressed_size_bytes"] == 13_500_000_000  # type: ignore[index]
    assert diagnostic["largest_layer_size_bytes"] == 10_600_000_000  # type: ignore[index]
    assert diagnostic["same_image_canary_recommended"] is True  # type: ignore[index]
    assert diagnostic["warm_existing_pod_mode_available"] is True  # type: ignore[index]
    assert diagnostic["diagnostic_blocker_if_canary_times_out"] == (  # type: ignore[index]
        "prebuilt_isaac_image_layer_pull_exceeded_watchdog"
    )
    readiness = _read_json(tmp_path / "runpod_provider_readiness_manifest.json")
    assert readiness["image_startup_diagnostic"] == diagnostic

    blocked = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "large-image-live.json",
        mode="on-demand-pod",
    )

    assert blocked["status"] == "blocked"
    assert (
        "large_worker_image_requires_canary_or_warm_provider"
        in blocked["blockers"]
    )

    monkeypatch.setenv(adapter.RUNPOD_ALLOW_LARGE_IMAGE_FRESH_START_ENV, "true")
    override = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "large-image-live-override.json",
        mode="on-demand-pod",
    )
    assert "large_worker_image_requires_canary_or_warm_provider" not in override[
        "request_blockers"
    ]


def test_runpod_adapter_plaintext_env_does_not_override_core_provider_inputs(
    tmp_path: Path,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    provider_shape = request["provider_request_shape"]  # type: ignore[index]
    assert isinstance(provider_shape, dict)
    environment = provider_shape["environment"]
    assert isinstance(environment, dict)
    environment["plaintext_env_var_names"] = [
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI",
        "BLUEPRINT_ISAAC_PROVIDER_PYTHON",
    ]
    environment["plaintext_env_values"] = {
        "BLUEPRINT_EVAL_MANIFEST_URI": "<stage bundle zip to https/gs/s3/r2>",
        "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI": "<stage bundle zip to https/gs/s3/r2>",
        "BLUEPRINT_ISAAC_PROVIDER_PYTHON": "/isaac-sim/python.sh",
    }
    _write_json(request_path, request)

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "plaintext-env.json",
        mode="dry-run",
    )

    body = result["runpod_request"]["on_demand_pod"]["body"]  # type: ignore[index]
    env = body["env"]
    assert env["BLUEPRINT_EVAL_MANIFEST_URI"].endswith("worker_manifest.json")
    assert env["BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI"].endswith("capture-root.zip")
    assert env["BLUEPRINT_ISAAC_PROVIDER_PYTHON"] == "/isaac-sim/python.sh"
    assert "<stage bundle zip to https/gs/s3/r2>" not in json.dumps(result)


def test_runpod_adapter_updates_and_starts_existing_pod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")
    monkeypatch.delenv(RUNPOD_EXISTING_POD_ID_ENV, raising=False)
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, status: int, body: dict[str, object]) -> None:
            self.status = status
            self.body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps(self.body).encode("utf-8")

    def fake_urlopen(request, timeout, policy):  # type: ignore[no-untyped-def]
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "body": body,
            }
        )
        if request.full_url.endswith("/pods/pod-123/update"):
            return FakeResponse(200, {"id": "pod-123", "desiredStatus": "EXITED"})
        if request.full_url.endswith("/pods/pod-123/start"):
            return FakeResponse(200, {"id": "pod-123", "desiredStatus": "RUNNING"})
        raise AssertionError(request.full_url)

    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy", fake_urlopen
    )

    result = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "existing.json",
        mode="existing-pod-start",
        existing_pod_id="pod-123",
        pod_name="reused-pod",
        allow_runpod_api_call=True,
        paid_resource_admission_grant=_paid_grant(),
    )

    assert result["status"] == "submitted"
    assert result["provider_job_submitted"] is True
    assert result["runpod_response"]["id"] == "pod-123"  # type: ignore[index]
    assert result["runpod_response"]["update_http_status_code"] == 200  # type: ignore[index]
    assert result["runpod_response"]["start_http_status_code"] == 200  # type: ignore[index]
    assert [call["url"] for call in calls] == [
        "https://rest.runpod.io/v1/pods/pod-123/update",
        "https://rest.runpod.io/v1/pods/pod-123/start",
    ]
    update_body = calls[0]["body"]
    assert isinstance(update_body, dict)
    assert update_body["name"] == "reused-pod"
    assert update_body["imageName"].endswith(":2026-06-12")
    assert update_body["dockerStartCmd"] == []
    assert update_body["env"]["BLUEPRINT_ROBOT_EVAL_JOB_ID"] == "runpod-adapter-job-1"
    assert "gpuTypeIds" not in update_body
    assert "gpuCount" not in update_body
    assert "computeType" not in update_body
    persisted = _read_json(tmp_path / "existing.json")
    assert "secret-runpod-key" not in json.dumps(persisted)


def test_runpod_adapter_builds_image_startup_canary_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.delenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", raising=False)

    missing_signed_put = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "canary-missing-signed-put.json",
        mode=adapter.RUNPOD_IMAGE_STARTUP_CANARY_MODE,
    )

    assert missing_signed_put["status"] == "blocked"
    assert (
        "missing_runtime_manifest_signed_put_url_for_image_startup_canary"
        in missing_signed_put["blockers"]
    )

    signed_put = "https://example.test/upload?x-goog-signature=secret-signature"
    monkeypatch.setenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", signed_put)
    monkeypatch.setenv(
        adapter.RUNPOD_IMAGE_STARTUP_CANARY_HOLD_SECONDS_ENV,
        "300",
    )
    request = _read_json(request_path)
    provider_shape = request["provider_request_shape"]  # type: ignore[index]
    assert isinstance(provider_shape, dict)
    provider_shape.pop("local_sim_only_prerequisite")
    _write_json(request_path, request)
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")
    shaped = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "canary-shaped.json",
        mode=adapter.RUNPOD_IMAGE_STARTUP_CANARY_MODE,
        pod_name="canary-pod",
    )
    persisted = (tmp_path / "canary-shaped.json").read_text(encoding="utf-8")

    assert shaped["status"] == "dry_run_ready"
    assert shaped["reason"] == "runpod_request_shape_validated_without_api_call"
    assert shaped["request_blockers"] == []
    body = shaped["runpod_request"]["body"]  # type: ignore[index]
    assert body["name"] == "canary-pod"
    assert body["dockerEntrypoint"] == ["bash"]
    assert body["dockerStartCmd"][0] == "-lc"
    assert "runpod_image_startup_canary.v1" in body["dockerStartCmd"][1]
    assert 'PYTHON_BIN="$(command -v python3 || command -v python || true)"' in body[
        "dockerStartCmd"
    ][1]
    assert 'PYTHON_BIN="/isaac-sim/python.sh"' in body["dockerStartCmd"][1]
    assert 'RESOLVED_PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"' in body[
        "dockerStartCmd"
    ][1]
    assert '"python3_path": shutil.which("python3")' in body["dockerStartCmd"][1]
    assert '"curl_path": shutil.which("curl")' in body["dockerStartCmd"][1]
    assert '"python_executable": sys.executable' in body["dockerStartCmd"][1]
    assert body["env"]["BLUEPRINT_RUNPOD_IMAGE_STARTUP_CANARY"] == "true"
    assert body["env"]["BLUEPRINT_CANARY_POST_UPLOAD_SLEEP_SECONDS"] == "300"
    assert body["env"]["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"] == (
        "<redacted:signed-url>"
    )
    assert "secret-signature" not in persisted
    assert "secret-runpod-key" not in persisted

    provider_shape["docker_entrypoint"] = [
        "/opt/blueprint/thin_release_entrypoint.sh"
    ]
    _write_json(request_path, request)
    thin = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "thin-canary-shaped.json",
        mode=adapter.RUNPOD_IMAGE_STARTUP_CANARY_MODE,
        pod_name="thin-canary-pod",
    )
    thin_body = thin["runpod_request"]["body"]  # type: ignore[index]
    assert thin_body["dockerEntrypoint"] == [
        "/opt/blueprint/thin_release_entrypoint.sh"
    ]
    assert thin_body["dockerStartCmd"][:2] == ["bash", "-lc"]


def test_runpod_adapter_builds_only_fixed_strict_policy_smoke_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    request = _read_json(request_path)
    provider_shape = request["provider_request_shape"]  # type: ignore[index]
    assert isinstance(provider_shape, dict)
    provider_shape["command"] = "echo caller-controlled-command-must-not-run"
    provider_shape["docker_entrypoint"] = [
        "/opt/blueprint/thin_release_entrypoint.sh"
    ]
    provider_shape.pop("local_sim_only_prerequisite", None)
    _write_json(request_path, request)
    monkeypatch.setenv(
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
        "https://example.test/upload?x-goog-signature=secret-signature",
    )
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")

    shaped = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "strict-policy-smoke-shaped.json",
        mode=adapter.RUNPOD_STRICT_POLICY_SMOKE_MODE,
        pod_name="strict-policy-smoke-pod",
    )

    assert shaped["status"] == "dry_run_ready"
    assert shaped["request_blockers"] == []
    body = shaped["runpod_request"]["body"]  # type: ignore[index]
    assert body["dockerEntrypoint"] == [
        "/opt/blueprint/thin_release_entrypoint.sh"
    ]
    assert body["dockerStartCmd"][:2] == ["bash", "-lc"]
    command = body["dockerStartCmd"][2]
    assert "caller-controlled-command-must-not-run" not in command
    assert "groot_oscar_runpod_strict_policy_smoke.v1" in command
    assert '"requested_action_count": 3' in command
    assert "while len(actions) < 3" in command
    assert '"physical_robot_control_performed": False' in command
    python_source = command.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(python_source, "strict-policy-smoke", "exec")

    def provider_call_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("provider call reached without shared admission grant")

    monkeypatch.setattr(adapter, "_http_json", provider_call_must_not_run)
    blocked = run_runpod_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "strict-policy-smoke-no-grant.json",
        mode=adapter.RUNPOD_STRICT_POLICY_SMOKE_MODE,
        pod_name="strict-policy-smoke-pod",
        allow_runpod_api_call=True,
    )
    assert blocked["status"] == "blocked"
    assert "runpod_provider_shared_admission_missing_or_invalid" in blocked["blockers"]
    assert blocked["api_call_performed"] is False


def test_runpod_adapter_main_errors_and_env_request_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(adapter.PROVIDER_LAUNCH_REQUEST_ENV, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        runpod_adapter_main([])
    assert excinfo.value.code == 2

    request_path = _ready_runpod_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.setenv(adapter.PROVIDER_LAUNCH_REQUEST_ENV, str(request_path))

    def fake_runpod_provider_adapter(**kwargs: object) -> dict[str, object]:
        assert kwargs["provider_launch_request_path"] == request_path
        return {
            "output_path": str(tmp_path / "result.json"),
            "status": "blocked",
            "mode": kwargs["mode"],
            "blockers": ["blocked-for-test"],
        }

    monkeypatch.setattr(adapter, "run_runpod_provider_adapter", fake_runpod_provider_adapter)
    exit_code = runpod_adapter_main(["--mode", "dry-run"])

    assert exit_code == 1
    assert "blockers=blocked-for-test" in capsys.readouterr().out
