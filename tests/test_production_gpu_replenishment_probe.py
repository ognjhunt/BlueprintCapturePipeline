from blueprint_pipeline.production_gpu_replenishment_probe import (
    build_replenishment_probe,
    run_replenishment_cycle,
)
from blueprint_pipeline.production_gpu_worker_pool import release_fingerprint


IMAGE = "docker.io/blueprint/worker@sha256:" + "a" * 64
HOST = "runpod-host-v1"
GPU = "runpod-secure-l40s-preferred-a40-fallback"


def test_cycle_proves_customer_queues_before_async_worker_arrives() -> None:
    now = {"value": 0.0}
    calls = {"bind": 0}

    def sender(_base, path, _payload, _token):
        if path == "/v1/customer-jobs/bind":
            calls["bind"] += 1
            if calls["bind"] == 1:
                return {
                    "status": "queued_waiting_for_warm_worker",
                    "scale_request_id": "scale-1",
                    "customer_request_provider_calls": 0,
                }
            return {
                "status": "bound_to_ready_worker",
                "worker_id": "pod-1",
                "lease_token": "gwl_" + "b" * 64,
            }
        return {"state": "ready"}

    result = run_replenishment_cycle(
        pool_base_url="https://pool.example.internal",
        token="x" * 32,
        host_image_id=HOST,
        worker_image_ref=IMAGE,
        gpu_family=GPU,
        timeout_seconds=60,
        poll_interval_seconds=5,
        sender=sender,
        clock=lambda: now["value"],
        sleeper=lambda seconds: now.__setitem__("value", now["value"] + seconds),
    )

    assert result["status"] == "passed"
    assert result["replenishment_seconds"] == 5
    assert result["provisioning_occurred_asynchronously"] is True


def test_aggregate_requires_three_exact_release_cycles_for_p95() -> None:
    fingerprint = release_fingerprint(
        host_image_id=HOST, worker_image_ref=IMAGE, gpu_family=GPU
    )
    cycle = {
        "schema_version": "production_gpu_replenishment_cycle.v1",
        "status": "passed",
        "release_fingerprint": fingerprint,
        "replenishment_seconds": 500,
        "customer_request_provider_calls": 0,
        "provisioning_occurred_asynchronously": True,
    }
    blocked = build_replenishment_probe(
        cycles=[cycle], release_fingerprint_value=fingerprint
    )
    passed = build_replenishment_probe(
        cycles=[cycle, {**cycle, "replenishment_seconds": 600}, {**cycle, "replenishment_seconds": 700}],
        release_fingerprint_value=fingerprint,
    )

    assert blocked["status"] == "failed"
    assert "replenishment_sample_count_insufficient" in blocked["blockers"]
    assert passed["status"] == "passed"
    assert passed["cold_replenishment_p95_seconds"] == 690
