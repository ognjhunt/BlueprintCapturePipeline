from blueprint_pipeline import groot_oscar_runpod_serverless_watchdog as watchdog


def test_teardown_deletes_endpoint_before_template_and_proves_absence(monkeypatch):
    calls = []
    inventories = {
        "endpoints": [
            {"id": "endpoint-1", "name": "blueprint-groot-oscar-serverless-test-endpoint"}
        ],
        "templates": [
            {"id": "template-1", "name": "blueprint-groot-oscar-serverless-test-template"}
        ],
    }

    def fake_call(method, path, body, *, key, timeout):
        calls.append((method, path, body, key, timeout))
        kind = path.strip("/").split("/", 1)[0]
        if method == "GET":
            return 200, list(inventories[kind])
        resource_id = path.rsplit("/", 1)[-1]
        inventories[kind] = [row for row in inventories[kind] if row["id"] != resource_id]
        return 204, {}

    monkeypatch.setattr(watchdog, "_runpod_call", fake_call)
    proof = watchdog.teardown_matching_resources(
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
        api_key="private",
        request_reason="test",
        clock=lambda: 100.0,
    )

    assert proof["status"] == "PASS"
    assert proof["provider_absence"]["billing_compute_stopped"] is True
    delete_paths = [path for method, path, *_ in calls if method == "DELETE"]
    assert delete_paths == ["/endpoints/endpoint-1", "/templates/template-1"]
    assert "private" not in str(proof)


def test_teardown_blocks_when_inventory_cannot_prove_absence(monkeypatch):
    monkeypatch.setattr(
        watchdog,
        "_runpod_call",
        lambda *args, **kwargs: (503, {"error": "unavailable"}),
    )

    proof = watchdog.teardown_matching_resources(
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
        api_key="private",
        request_reason="test",
    )

    assert proof["status"] == "BLOCKED"
    assert proof["provider_absence"]["billing_compute_stopped"] is False


def test_ambiguous_create_is_charged_only_when_provider_saw_endpoint(monkeypatch):
    settlements = []

    class Budget:
        def __init__(self, *_args, **_kwargs):
            pass

        def settle(self, **kwargs):
            settlements.append(kwargs)
            return {"ledger_status": "settled"}

    monkeypatch.setattr(watchdog, "ProductionGpuCampaignBudget", Budget)
    monkeypatch.setattr(watchdog.time, "time", lambda: 110.0)
    state = {
        "endpoint_create_requested_at_epoch": 100.0,
        "campaign_budget": {
            "ledger_path": "unused",
            "reservation_id": "reservation-1",
            "reserved_gpu_seconds": 100,
            "max_hourly_rate_usd": 1.0,
            "initial_spent_usd": 0.0,
            "initial_used_gpu_seconds": 0,
            "total_spend_cap_usd": 20.0,
            "combined_gpu_wall_cap_seconds": 21_000,
        },
    }
    present = watchdog._settle_budget(
        state,
        {
            "status": "PASS",
            "reason": "test",
            "pre_teardown": {"matching_endpoint_count": 1},
        },
    )
    assert present["measurement"] == ("endpoint_request_wall_clock_provider_presence_confirmed")
    assert settlements[-1]["charged_gpu_seconds"] == 10

    absent = watchdog._settle_budget(
        state,
        {
            "status": "PASS",
            "reason": "test",
            "pre_teardown": {"matching_endpoint_count": 0},
        },
    )
    assert absent["measurement"] == "endpoint_wall_clock"
    assert settlements[-1]["charged_gpu_seconds"] == 0


def test_queue_only_job_settles_zero_gpu_time(monkeypatch):
    settlements = []

    class Budget:
        def __init__(self, *_args, **_kwargs):
            pass

        def settle(self, **kwargs):
            settlements.append(kwargs)
            return {"ledger_status": "settled"}

    monkeypatch.setattr(watchdog, "ProductionGpuCampaignBudget", Budget)
    monkeypatch.setattr(watchdog.time, "time", lambda: 1_307.0)
    result = watchdog._settle_budget(
        {
            "endpoint_allocated_at_epoch": 100.0,
            "serverless_job_execution": {
                "worker_execution_observed": False,
                "provider_job_status": "IN_QUEUE",
                "poll_result_status": "WALL_TIMEOUT",
                "execution_time_ms": None,
            },
            "campaign_budget": {
                "ledger_path": "unused",
                "reservation_id": "reservation-queue-only",
                "reserved_gpu_seconds": 5_215,
                "max_hourly_rate_usd": 1.75,
                "initial_spent_usd": 0.0,
                "initial_used_gpu_seconds": 0,
                "total_spend_cap_usd": 20.0,
                "combined_gpu_wall_cap_seconds": 21_000,
            },
        },
        {"status": "PASS", "reason": "queue_timeout"},
    )

    assert result["measurement"] == "provider_job_queue_only_no_worker_execution"
    assert settlements[-1]["charged_gpu_seconds"] == 0
    assert settlements[-1]["charged_usd"] == 0
