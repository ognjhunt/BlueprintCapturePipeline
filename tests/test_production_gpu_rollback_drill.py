from blueprint_pipeline.production_gpu_rollback_drill import run_rollback_drill


IMAGE = "docker.io/blueprint/worker@sha256:" + "a" * 64


def test_rollback_quarantines_candidate_and_routes_to_other_ready_worker() -> None:
    calls: list[str] = []

    def sender(_base, path, payload, _token):
        calls.append(path)
        if path.endswith("/quarantine"):
            return {"state": "quarantined"}
        if path == "/v1/customer-jobs/bind":
            return {
                "status": "bound_to_ready_worker",
                "worker_id": "pod-2",
                "lease_token": "gwl_" + "b" * 64,
                "customer_request_provider_calls": 0,
            }
        return {"state": "ready"}

    result = run_rollback_drill(
        pool_base_url="https://pool.example.internal",
        token="x" * 32,
        candidate_worker_id="pod-1",
        host_image_id="runpod-host-v1",
        worker_image_ref=IMAGE,
        gpu_family="runpod-secure-l40s-preferred-a40-fallback",
        sender=sender,
    )

    assert result["status"] == "passed"
    assert result["candidate_quarantined_before_provider_teardown"] is True
    assert result["customer_bind_routed_to_quarantined_worker"] is False
    assert calls[0] == "/v1/workers/pod-1/quarantine"


def test_rollback_fails_if_no_alternate_warm_worker_is_ready() -> None:
    def sender(_base, path, _payload, _token):
        if path.endswith("/quarantine"):
            return {"state": "quarantined"}
        return {"status": "queued_waiting_for_warm_worker", "customer_request_provider_calls": 0}

    result = run_rollback_drill(
        pool_base_url="https://pool.example.internal",
        token="x" * 32,
        candidate_worker_id="pod-1",
        host_image_id="runpod-host-v1",
        worker_image_ref=IMAGE,
        gpu_family="runpod-secure-l40s-preferred-a40-fallback",
        sender=sender,
    )

    assert result["status"] == "failed"
    assert "alternate_warm_worker_not_available" in result["blockers"]
