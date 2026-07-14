from blueprint_pipeline.groot_oscar_runpod_preflight import collect_runpod_preflight


def test_read_only_preflight_binds_volume_capacity_inventory_and_watchdog() -> None:
    result = collect_runpod_preflight(
        network_volume_id="volume-1",
        model_cache_path="/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1",
        gpu_type_id="NVIDIA A40",
        required_cuda_version="12.6",
        name_prefix="blueprint-groot-oscar-canary-",
        watchdog={
            "schema_version": "production_gpu_warm_watchdog.v1",
            "status": "armed",
            "independent_process": True,
            "pid": 1,
            "deadline_epoch": 2800.0,
        },
        max_spend_usd=1.0,
        paid_mutation_authorized=True,
        volume_getter=lambda _id: (
            200,
            {"id": "volume-1", "dataCenterId": "US-TX-3", "size": 50},
        ),
        capacity_probe=lambda _request: {
            "status": "available",
            "viable_gpu_types": [
                {
                    "gpu_type_id": "NVIDIA A40",
                    "capacity_confidence": "advisory",
                    "available_gpu_counts": [1],
                    "on_demand_price_usd_per_hour": 0.44,
                }
            ],
        },
        inventory_probe=lambda _prefix: {
            "api_confirmed": True,
            "live_resource_count": 0,
        },
        clock=lambda: 1000.0,
    )
    assert result["status"] == "verified"
    assert result["provider_mutations_performed"] == 0
    assert result["volume"]["data_center_id"] == "US-TX-3"
    assert result["runtime"]["launch_constraints"] == {
        "dataCenterIds": ["US-TX-3"],
        "allowedCudaVersions": ["12.6"],
    }
    assert result["spend"]["watchdog_armed_before_allocation"] is True


def test_preflight_rejects_unknown_volume_capacity_inventory_and_watchdog() -> None:
    result = collect_runpod_preflight(
        network_volume_id="missing",
        model_cache_path="/workspace/models",
        gpu_type_id="NVIDIA A40",
        required_cuda_version="12.6",
        name_prefix="blueprint-groot-oscar-canary-",
        watchdog={},
        max_spend_usd=1.0,
        paid_mutation_authorized=True,
        volume_getter=lambda _id: (404, {}),
        capacity_probe=lambda _request: {"status": "blocked"},
        inventory_probe=lambda _prefix: {"api_confirmed": False},
        clock=lambda: 1000.0,
    )
    assert result["status"] == "blocked"
    assert "runpod_network_volume_provider_id_mismatch" in result["blockers"]
    assert "runpod_gpu_capacity_not_provider_verified" in result["blockers"]
    assert "runpod_preallocation_inventory_not_zero" in result["blockers"]
    assert "runpod_teardown_watchdog_not_armed_before_allocation" in result["blockers"]
