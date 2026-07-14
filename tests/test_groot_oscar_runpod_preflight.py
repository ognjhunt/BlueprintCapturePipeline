import os

from blueprint_pipeline.groot_oscar_runpod_preflight import collect_runpod_preflight


def test_read_only_preflight_binds_volume_capacity_inventory_and_watchdog() -> None:
    result = collect_runpod_preflight(
        network_volume_id="volume-1",
        model_cache_path="/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1",
        gpu_type_id="NVIDIA A40",
        required_cuda_version="12.6",
        name_prefix="blueprint-groot-oscar-canary-",
        watchdog={
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "armed",
            "independent_process": True,
            "pid": 1,
            "deadline_epoch": 2800.0,
            "pod_name_prefix": "blueprint-groot-oscar-canary-",
        },
        max_spend_usd=1.0,
        paid_mutation_authorized=True,
        volume_getter=lambda _id: (
            200,
            {"id": "volume-1", "dataCenterId": "US-TX-3", "size": 50},
        ),
        capacity_probe=lambda request: {
            "status": "available",
            "viable_gpu_types": [
                {
                    "gpu_type_id": "NVIDIA A40",
                    "capacity_confidence": "advisory",
                    "capacity_data_center_id": request["dataCenterIds"][0],
                    "capacity_allowed_cuda_versions": request[
                        "allowedCudaVersions"
                    ],
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
        process_argv_probe=lambda _pid: (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--pod-name-prefix",
            "blueprint-groot-oscar-canary-",
            "--deadline-epoch",
            "2800.0",
        ),
    )
    assert result["status"] == "verified"
    assert result["provider_mutations_performed"] == 0
    assert result["volume"]["data_center_id"] == "US-TX-3"
    assert result["runtime"]["launch_constraints"] == {
        "dataCenterIds": ["US-TX-3"],
        "allowedCudaVersions": ["12.6"],
    }
    assert result["spend"]["watchdog_armed_before_allocation"] is True
    assert result["spend"]["watchdog_process_identity_verified"] is True


def test_preflight_rejects_unrelated_live_process_as_watchdog() -> None:
    result = collect_runpod_preflight(
        network_volume_id="volume-1",
        model_cache_path="/workspace/models",
        gpu_type_id="NVIDIA A40",
        required_cuda_version="12.6",
        name_prefix="blueprint-groot-oscar-canary-",
        watchdog={
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "armed",
            "independent_process": True,
            "pid": os.getpid(),
            "deadline_epoch": 2800.0,
            "pod_name_prefix": "blueprint-groot-oscar-canary-",
        },
        max_spend_usd=1.0,
        paid_mutation_authorized=True,
        volume_getter=lambda _id: (
            200,
            {"id": "volume-1", "dataCenterId": "US-TX-3", "size": 50},
        ),
        capacity_probe=lambda _request: {"status": "available", "viable_gpu_types": []},
        inventory_probe=lambda _prefix: {"api_confirmed": True, "live_resource_count": 0},
        clock=lambda: 1000.0,
        process_argv_probe=lambda _pid: ("python", "unrelated.py"),
    )
    assert result["status"] == "blocked"
    assert result["spend"]["watchdog_process_identity_verified"] is False
    assert "runpod_teardown_watchdog_not_armed_before_allocation" in result["blockers"]


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
