from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline.production_gpu_worker_pool import (
    REQUIRED_READY_CHECKS,
    ProductionGpuWorkerPool,
    WorkerLeaseConflict,
    build_production_startup_readiness,
    create_production_gpu_worker_pool_app,
)


IMAGE = "registry.example/blueprint-worker@sha256:" + "a" * 64
HOST = "projects/blueprint/global/images/blueprint-g4-host-20260714"
GPU = "g4-rtx-pro-6000"
READY = {name: True for name in REQUIRED_READY_CHECKS}


def _register(pool: ProductionGpuWorkerPool, worker_id: str = "worker-1") -> dict:
    return pool.register_ready_worker(
        worker_id=worker_id,
        provider="gcp",
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        endpoint_ref=f"private-worker://{worker_id}",
        readiness=READY,
    )


def test_customer_bind_uses_ready_exact_release_without_provider_call(tmp_path: Path) -> None:
    pool = ProductionGpuWorkerPool(tmp_path / "pool.sqlite")
    registered = _register(pool)

    binding = pool.bind_customer_job(
        job_id="customer-job-1",
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
    )

    assert binding["status"] == "bound_to_ready_worker"
    assert binding["worker_id"] == "worker-1"
    assert binding["release_fingerprint"] == registered["release_fingerprint"]
    assert binding["customer_request_provider_calls"] == 0
    assert binding["cold_provisioning_started_in_request_path"] is False
    assert binding["bind_latency_ms"] >= 0


def test_no_ready_worker_queues_and_emits_async_scale_request(tmp_path: Path) -> None:
    pool = ProductionGpuWorkerPool(tmp_path / "pool.sqlite")

    first = pool.bind_customer_job(
        job_id="customer-job-1",
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
    )
    second = pool.bind_customer_job(
        job_id="customer-job-2",
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
    )

    assert first["status"] == "queued_waiting_for_warm_worker"
    assert first["customer_request_provider_calls"] == 0
    assert first["cold_provisioning_started_in_request_path"] is False
    assert second["scale_request_id"] == first["scale_request_id"]
    assert pool.snapshot()["pending_scale_requests"] == 1


def test_registration_fails_until_every_expensive_readiness_phase_passes(tmp_path: Path) -> None:
    pool = ProductionGpuWorkerPool(tmp_path / "pool.sqlite")
    readiness = dict(READY)
    readiness["isaac_renderer_warm"] = False
    readiness["policy_endpoint_ready"] = False

    with pytest.raises(ValueError, match="worker_readiness_incomplete:isaac_renderer_warm,policy_endpoint_ready"):
        pool.register_ready_worker(
            worker_id="worker-1",
            provider="gcp",
            host_image_id=HOST,
            worker_image_ref=IMAGE,
            gpu_family=GPU,
            endpoint_ref="private-worker://worker-1",
            readiness=readiness,
        )


def test_worker_release_requires_active_job_token_and_quarantines_unhealthy(tmp_path: Path) -> None:
    pool = ProductionGpuWorkerPool(tmp_path / "pool.sqlite")
    _register(pool)
    binding = pool.bind_customer_job(
        job_id="customer-job-1", host_image_id=HOST, worker_image_ref=IMAGE, gpu_family=GPU
    )

    with pytest.raises(WorkerLeaseConflict, match="active_worker_job_lease_required"):
        pool.release_worker(
            worker_id="worker-1",
            job_id="customer-job-1",
            lease_token="gwl_" + "0" * 64,
            healthy=True,
        )
    released = pool.release_worker(
        worker_id="worker-1",
        job_id="customer-job-1",
        lease_token=binding["lease_token"],
        healthy=False,
    )
    assert released["state"] == "quarantined"
    assert pool.snapshot()["worker_counts"]["quarantined"] == 1


def test_provider_teardown_quarantines_worker_before_it_can_bind(tmp_path: Path) -> None:
    pool = ProductionGpuWorkerPool(tmp_path / "pool.sqlite")
    _register(pool)

    result = pool.quarantine_worker(
        worker_id="worker-1", reason="watchdog_provider_teardown"
    )
    queued = pool.bind_customer_job(
        job_id="customer-job-1",
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
    )

    assert result["state"] == "quarantined"
    assert result["ready_for_customer_binding"] is False
    assert queued["status"] == "queued_waiting_for_warm_worker"


def test_stale_heartbeat_and_expired_lease_fail_closed(tmp_path: Path) -> None:
    now = [1000.0]
    pool = ProductionGpuWorkerPool(tmp_path / "pool.sqlite", clock=lambda: now[0])
    _register(pool)
    now[0] += 60

    queued = pool.bind_customer_job(
        job_id="customer-job-1",
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        heartbeat_ttl_seconds=45,
    )
    assert queued["status"] == "queued_waiting_for_warm_worker"
    assert pool.snapshot()["worker_counts"]["quarantined"] == 1


def test_min_ready_reconciliation_only_writes_async_demand(tmp_path: Path) -> None:
    pool = ProductionGpuWorkerPool(tmp_path / "pool.sqlite")
    result = pool.reconcile_min_ready(
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        min_ready_workers=2,
    )
    assert result["deficit"] == 2
    assert result["provider_calls_performed"] == 0
    assert result["autoscaler_must_process_asynchronously"] is True


def test_scale_demand_has_one_atomic_autoscaler_owner_and_retry(tmp_path: Path) -> None:
    pool = ProductionGpuWorkerPool(tmp_path / "pool.sqlite")
    pool.reconcile_min_ready(
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        min_ready_workers=2,
    )
    claimed = pool.claim_scale_request(autoscaler_id="autoscaler-1")
    assert claimed is not None
    assert claimed["requested_ready_workers"] == 2
    assert claimed["provider_mutation_authorized_by_customer_request"] is False
    assert pool.claim_scale_request(autoscaler_id="autoscaler-2") is None

    _register(pool, "worker-1")
    assert pool.snapshot()["pending_scale_requests"] == 1
    assert pool.claim_scale_request(autoscaler_id="autoscaler-2") is None
    _register(pool, "worker-2")
    assert pool.snapshot()["pending_scale_requests"] == 0

    # A separate deficit can be returned to the queue after a retryable
    # provider failure, but only by its active autoscaler lease owner.
    pool.reconcile_min_ready(
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        min_ready_workers=3,
    )
    retry_claim = pool.claim_scale_request(autoscaler_id="autoscaler-1")
    assert retry_claim is not None
    retried = pool.release_scale_request(
        scale_request_id=retry_claim["scale_request_id"],
        scale_token=retry_claim["scale_token"],
        retryable=True,
    )
    assert retried["status"] == "pending"
    assert pool.claim_scale_request(autoscaler_id="autoscaler-2") is not None


def test_atomic_database_update_prevents_double_binding_one_worker(tmp_path: Path) -> None:
    path = tmp_path / "pool.sqlite"
    pool = ProductionGpuWorkerPool(path)
    _register(pool)

    first = pool.bind_customer_job(
        job_id="customer-job-1", host_image_id=HOST, worker_image_ref=IMAGE, gpu_family=GPU
    )
    second = pool.bind_customer_job(
        job_id="customer-job-2", host_image_id=HOST, worker_image_ref=IMAGE, gpu_family=GPU
    )
    assert first["status"] == "bound_to_ready_worker"
    assert second["status"] == "queued_waiting_for_warm_worker"
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM production_gpu_workers WHERE state='leased'"
        ).fetchone()[0] == 1


def test_readiness_gate_keeps_local_contract_separate_from_live_launch_proof() -> None:
    local = build_production_startup_readiness(
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        min_ready_workers=1,
        bind_slo_seconds=10,
        cold_replenishment_slo_seconds=180,
    )
    assert local["status"] == "local_contract_ready_live_proof_required"
    assert "live_evidence_missing:warm_bind_p95" in local["blockers"]
    assert local["claim_boundary"]["campaign_cold_start_is_release_engineering_evidence_only"] is True

    live = build_production_startup_readiness(
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        min_ready_workers=1,
        bind_slo_seconds=10,
        cold_replenishment_slo_seconds=180,
        live_evidence={
            "release_fingerprint": local["release_fingerprint"],
            "baked_host_image_verified": True,
            "worker_image_cached_on_host_verified": True,
            "all_required_ready_checks_observed": True,
            "ready_worker_count": 2,
            "current_capacity_deployed": True,
            "current_provider_live_worker_count": 2,
            "warm_bind_p95_seconds": 2.5,
            "cold_replenishment_p95_seconds": 120,
            "customer_request_provider_calls": 0,
            "async_scale_replenishment_proven": True,
            "rollback_drill_passed": True,
            "provider_inventory_confirmed": True,
            "teardown_and_absence_confirmed": True,
        },
    )
    assert live["status"] == "customer_launch_ready"
    assert live["blockers"] == []


def test_async_replenishment_may_be_longer_than_customer_bind_slo() -> None:
    result = build_production_startup_readiness(
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        min_ready_workers=2,
        bind_slo_seconds=10,
        cold_replenishment_slo_seconds=1800,
    )

    assert "cold_replenishment_slo_must_be_at_most_1800_seconds" not in result["blockers"]
    assert result["startup_targets"]["cold_replenishment_is_outside_customer_request_slo"] is True


def test_private_api_binds_only_after_authenticated_worker_registration(tmp_path: Path) -> None:
    app = create_production_gpu_worker_pool_app(
        database_path=tmp_path / "pool.sqlite", auth_token="t" * 32
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + "t" * 32}
    payload = {
        "worker_id": "worker-1",
        "provider": "gcp",
        "host_image_id": HOST,
        "worker_image_ref": IMAGE,
        "gpu_family": GPU,
        "endpoint_ref": "private-worker://worker-1",
        "readiness": READY,
    }
    assert client.post("/v1/workers/ready", json=payload).status_code == 401
    assert client.post("/v1/workers/ready", json=payload, headers=headers).status_code == 200
    response = client.post(
        "/v1/customer-jobs/bind",
        json={
            "job_id": "customer-job-1",
            "host_image_id": HOST,
            "worker_image_ref": IMAGE,
            "gpu_family": GPU,
        },
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "bound_to_ready_worker"


def test_private_worker_agent_heartbeat_continues_during_customer_lease(
    tmp_path: Path,
) -> None:
    token = "t" * 32
    app = create_production_gpu_worker_pool_app(
        database_path=tmp_path / "pool.sqlite", auth_token=token
    )
    pool = app.state.pool
    _register(pool, "worker-agent-heartbeat")
    pool.bind_customer_job(
        job_id="customer-job-1",
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
    )

    response = TestClient(app).post(
        "/v1/workers/worker-agent-heartbeat/heartbeat",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["heartbeat_recorded"] is True
    assert response.json()["state"] == "leased"
