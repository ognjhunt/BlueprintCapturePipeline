import os

import pytest

import blueprint_pipeline.groot_oscar_runpod_preflight as preflight_module

build_model_volume_watchdog_handoff_evidence = (
    preflight_module.build_model_volume_watchdog_handoff_evidence
)
collect_runpod_preflight = preflight_module.collect_runpod_preflight

MODEL_VOLUME_WATCHDOG_STATE = "/tmp/model-volume/watchdog_state.json"


def _model_volume_handoff(*, deadline_epoch: float = 4000.0) -> dict:
    return {
        "schema_version": "groot_oscar_model_volume_watchdog_handoff.v1",
        "status": "volume_ready_watchdog_retained",
        "volume_id": "volume-1",
        "preparation_pod_absence_confirmed": True,
        "volume_presence_confirmed": True,
        "teardown_owner": "independent_model_volume_watchdog",
        "watchdog_pid": os.getpid(),
        "watchdog_state_path": MODEL_VOLUME_WATCHDOG_STATE,
        "watchdog_deadline_epoch": deadline_epoch,
        "next_owner_must_arm_before_transfer": True,
    }


def _watchdog_argv(pid: int, *, canary_deadline_epoch: float) -> tuple[str, ...]:
    if pid == os.getpid():
        return (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_model_volume",
            "watchdog",
            "--state",
            MODEL_VOLUME_WATCHDOG_STATE,
        )
    return (
        "python",
        "-m",
        "blueprint_pipeline.groot_oscar_runpod_watchdog",
        "--pod-name-prefix",
        "blueprint-groot-oscar-canary-",
        "--deadline-epoch",
        str(canary_deadline_epoch),
    )


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
        model_volume_watchdog_handoff=_model_volume_handoff(),
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
                    "available_gpu_counts": [],
                    "single_gpu_offer_requested": True,
                    "single_gpu_offer_available": True,
                    "on_demand_price_usd_per_hour": 0.44,
                }
            ],
        },
        inventory_probe=lambda _prefix: {
            "api_confirmed": True,
            "live_resource_count": 0,
        },
        clock=lambda: 1000.0,
        process_argv_probe=lambda pid: _watchdog_argv(
            pid, canary_deadline_epoch=2800.0
        ),
    )
    assert result["status"] == "verified"
    assert result["provider_mutations_performed"] == 0
    assert result["volume"]["data_center_id"] == "US-TX-3"
    assert result["runtime"]["launch_constraints"] == {
        "dataCenterIds": ["US-TX-3"],
        "allowedCudaVersions": ["12.6"],
    }
    assert result["runtime"]["single_gpu_available"] is True
    assert result["spend"]["watchdog_armed_before_allocation"] is True
    assert result["spend"]["watchdog_process_identity_verified"] is True
    assert result["model_volume_watchdog_handoff"]["status"] == "verified"
    assert (
        result["model_volume_watchdog_handoff"][
            "watchdog_process_identity_verified"
        ]
        is True
    )
    refreshed_handoff = build_model_volume_watchdog_handoff_evidence(
        handoff=result["model_volume_watchdog_handoff"],
        network_volume_id="volume-1",
        canary_watchdog_deadline_epoch=2800.0,
        clock=lambda: 1001.0,
        process_argv_probe=lambda pid: _watchdog_argv(
            pid, canary_deadline_epoch=2800.0
        ),
    )
    assert refreshed_handoff["status"] == "verified"


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
        model_volume_watchdog_handoff=_model_volume_handoff(),
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
    assert "model_volume_watchdog_process_identity_invalid" in result["blockers"]


def test_preflight_rejects_unknown_volume_capacity_inventory_and_watchdog() -> None:
    result = collect_runpod_preflight(
        network_volume_id="missing",
        model_cache_path="/workspace/models",
        gpu_type_id="NVIDIA A40",
        required_cuda_version="12.6",
        name_prefix="blueprint-groot-oscar-canary-",
        watchdog={},
        model_volume_watchdog_handoff={},
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


def test_preflight_rejects_model_volume_watchdog_deadline_before_canary() -> None:
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
            "pid": 1,
            "deadline_epoch": 1900.0,
            "pod_name_prefix": "blueprint-groot-oscar-canary-",
        },
        model_volume_watchdog_handoff=_model_volume_handoff(deadline_epoch=1950.0),
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
                    "capacity_allowed_cuda_versions": request["allowedCudaVersions"],
                    "single_gpu_offer_requested": True,
                    "single_gpu_offer_available": True,
                    "on_demand_price_usd_per_hour": 0.44,
                }
            ],
        },
        inventory_probe=lambda _prefix: {"api_confirmed": True, "live_resource_count": 0},
        clock=lambda: 1000.0,
        process_argv_probe=lambda pid: _watchdog_argv(
            pid, canary_deadline_epoch=1900.0
        ),
    )
    assert result["status"] == "blocked"
    assert "model_volume_watchdog_ttl_does_not_cover_canary" in result["blockers"]


def test_model_volume_handoff_rejects_dead_watchdog_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_pid = 4242
    handoff = _model_volume_handoff()
    handoff["watchdog_pid"] = model_pid

    def kill(pid: int, _signal: int) -> None:
        if pid == model_pid:
            raise ProcessLookupError

    monkeypatch.setattr(preflight_module.os, "kill", kill)
    result = build_model_volume_watchdog_handoff_evidence(
        handoff=handoff,
        network_volume_id="volume-1",
        canary_watchdog_deadline_epoch=2800.0,
        clock=lambda: 1000.0,
        process_argv_probe=lambda pid: _watchdog_argv(
            pid, canary_deadline_epoch=2800.0
        ),
    )

    assert result["status"] == "blocked"
    assert result["watchdog_process_identity_verified"] is False
    assert "model_volume_watchdog_process_not_alive" in result["blockers"]
