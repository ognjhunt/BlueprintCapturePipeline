from __future__ import annotations

import pytest

from blueprint_pipeline.production_gpu_worker_agent import (
    WorkerEvidenceError,
    _read_token,
    build_worker_registration_payload,
    run_worker_agent,
)


IMAGE = "docker.io/blueprint/worker@sha256:" + "a" * 64


def _evidence() -> tuple[dict, dict, dict]:
    return (
        {
            "schema_version": "production_gpu_host_boot_evidence.v1",
            "host_image_id": "runpod-secure-l40s-active-worker-v1",
            "actual_gpu_model": "NVIDIA L40S",
            "checks": {
                "host_image_booted": True,
                "nvidia_driver_ready": True,
                "container_runtime_ready": True,
            },
        },
        {
            "schema_version": "production_gpu_cache_evidence.v1",
            "worker_image_ref": IMAGE,
            "model_manifest_digest": "sha256:" + "b" * 64,
            "checks": {"worker_image_cached": True, "models_cached_offline": True},
        },
        {
            "schema_version": "production_gpu_warm_serve_ready.v2",
            "status": "serving",
            "launch_session_id": "session-1",
            "worker_image_ref": IMAGE,
            "checks": {
                "isaac_renderer_warm": True,
                "kitchen_scene_loaded": True,
                "policy_endpoint_ready": True,
                "worker_healthcheck_passed": True,
            },
        },
    )


def _payload() -> dict:
    host, cache, warm = _evidence()
    return build_worker_registration_payload(
        worker_id="runpod-l40s-1",
        provider="runpod",
        host_image_id="runpod-secure-l40s-active-worker-v1",
        worker_image_ref=IMAGE,
        gpu_family="NVIDIA L40S",
        endpoint_ref="https://worker.example.internal/jobs",
        launch_session_id="session-1",
        host_evidence=host,
        cache_evidence=cache,
        warm_evidence=warm,
    )


def test_registration_requires_all_nine_independent_checks() -> None:
    payload = _payload()

    assert set(payload["readiness"]) == {
        "host_image_booted",
        "nvidia_driver_ready",
        "container_runtime_ready",
        "worker_image_cached",
        "models_cached_offline",
        "isaac_renderer_warm",
        "kitchen_scene_loaded",
        "policy_endpoint_ready",
        "worker_healthcheck_passed",
    }
    assert all(payload["readiness"].values())
    assert payload["agent_evidence"]["customer_command_executed"] is False


@pytest.mark.parametrize(
    ("layer", "field", "error"),
    [
        (0, "nvidia_driver_ready", "host_checks_incomplete:nvidia_driver_ready"),
        (1, "models_cached_offline", "cache_checks_incomplete:models_cached_offline"),
        (2, "policy_endpoint_ready", "warm_checks_incomplete:policy_endpoint_ready"),
    ],
)
def test_registration_fails_closed_for_each_evidence_layer(layer: int, field: str, error: str) -> None:
    records = list(_evidence())
    records[layer]["checks"][field] = False

    with pytest.raises(WorkerEvidenceError, match=error):
        build_worker_registration_payload(
            worker_id="worker-1",
            provider="runpod",
            host_image_id="runpod-secure-l40s-active-worker-v1",
            worker_image_ref=IMAGE,
            gpu_family="NVIDIA L40S",
            endpoint_ref="https://worker.example.internal/jobs",
            launch_session_id="session-1",
            host_evidence=records[0],
            cache_evidence=records[1],
            warm_evidence=records[2],
        )


def test_registration_rejects_stale_session_and_release() -> None:
    host, cache, warm = _evidence()
    warm["launch_session_id"] = "old-session"
    with pytest.raises(WorkerEvidenceError, match="warm_evidence_launch_session_mismatch"):
        build_worker_registration_payload(
            worker_id="worker-1", provider="runpod",
            host_image_id="runpod-secure-l40s-active-worker-v1",
            worker_image_ref=IMAGE, gpu_family="NVIDIA L40S",
            endpoint_ref="https://worker.example.internal/jobs",
            launch_session_id="session-1", host_evidence=host,
            cache_evidence=cache, warm_evidence=warm,
        )


def test_agent_registers_once_without_provisioning_or_customer_command() -> None:
    calls: list[tuple[str, str, dict, str]] = []

    def sender(base: str, path: str, payload: dict, token: str) -> dict:
        calls.append((base, path, payload, token))
        return {"ready_for_customer_binding": True}

    result = run_worker_agent(
        registration_payload=_payload(),
        pool_base_url="https://pool.example.internal",
        token="secret-token",
        once=True,
        sender=sender,
    )

    assert result == {"status": "registered", "worker_id": "runpod-l40s-1"}
    assert [call[1] for call in calls] == ["/v1/workers/ready"]
    assert calls[0][2]["agent_evidence"]["customer_command_executed"] is False


def test_pool_token_symlink_is_rejected(tmp_path) -> None:
    target = tmp_path / "token"
    target.write_text("x" * 32)
    target.chmod(0o600)
    link = tmp_path / "token-link"
    link.symlink_to(target)

    with pytest.raises(WorkerEvidenceError, match="pool_token_file_invalid"):
        _read_token(link)
