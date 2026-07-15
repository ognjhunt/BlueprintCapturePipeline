import json
import os

from blueprint_pipeline.groot_oscar_runpod_canary import (
    _finalize_adapter_allocation,
    prepare_canary_launch,
    refresh_runpod_preflight,
    run_canary,
)


DIGEST = "docker.io/example/release@sha256:" + "a" * 64


def test_submitted_canary_without_pod_id_fails_closed(tmp_path) -> None:
    output = tmp_path / "adapter.json"

    result = _finalize_adapter_allocation(
        adapter={"status": "submitted", "runpod_response": {}},
        adapter_output=output,
        pod_name="blueprint-groot-oscar-canary-test",
        release_image_ref=DIGEST,
    )

    assert result["status"] == "failed"
    assert result["blockers"] == ["runpod_canary_pod_id_missing"]
    assert result["provider_allocation_ambiguous"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    allocation = json.loads(
        (tmp_path / "warm_serve_pod.json").read_text(encoding="utf-8")
    )
    assert allocation["status"] == "allocation_ambiguous"
    assert allocation["pod_id"] is None


def _request() -> dict:
    return {
        "provider_request_shape": {
            "image": {"configured_image_ref": DIGEST},
            "gpu": {"preferred_gpu_type_id": "NVIDIA A40"},
        }
    }


def _preflight() -> dict:
    return {
        "status": "verified",
        "volume": {
            "provider": "runpod",
            "provider_api_verified": True,
            "id": "volume-1",
            "data_center_id": "US-TX-3",
            "size_bytes": 50 * 1024**3,
            "model_cache_path": "/workspace/models",
        },
        "runtime": {
            "provider": "runpod",
            "provider_api_verified": True,
            "data_center_id": "US-TX-3",
            "capacity_data_center_id": "US-TX-3",
            "capacity_allowed_cuda_versions": ["12.6"],
            "gpu_type_id": "NVIDIA A40",
            "capacity_confidence": "advisory",
            "single_gpu_available": True,
            "required_cuda_version": "12.6",
            "allowed_cuda_versions": ["12.6"],
            "on_demand_price_usd_per_hour": 0.44,
            "warm_worker_only": True,
            "provider_inventory_verified_zero": True,
        },
        "spend": {
            "paid_mutation_authorized": True,
            "max_spend_usd": 1.0,
            "hard_ttl_seconds": 900,
            "one_resource_limit": True,
            "independent_teardown_watchdog": True,
            "watchdog_armed_before_allocation": True,
            "watchdog_pid": os.getpid(),
            "watchdog_deadline_epoch": 1900.0,
            "watchdog_pod_name_prefix": "blueprint-groot-oscar-canary-",
        },
    }


def test_canary_preparation_binds_exact_admitted_tuple_into_request() -> None:
    result = prepare_canary_launch(
        request=_request(),
        release={
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        model_cache={
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        preflight=_preflight(),
    )
    assert result["status"] == "admitted"
    shape = result["bound_request"]["provider_request_shape"]
    assert shape["network_volume_id"] == "volume-1"
    assert shape["data_center_id"] == "US-TX-3"
    assert shape["allowed_cuda_versions"] == ["12.6"]
    assert shape["cache"]["paths"]["groot_oscar_models"] == "/workspace/models"
    assert shape["docker_entrypoint"] == [
        "/opt/blueprint/thin_release_entrypoint.sh"
    ]
    assert shape["environment"]["plaintext_env_values"][
        "BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST"
    ] == "sha256:" + "b" * 64


def test_canary_preparation_rejects_tag_or_different_digest() -> None:
    request = _request()
    request["provider_request_shape"]["image"]["configured_image_ref"] = "example/release:v1"
    result = prepare_canary_launch(
        request=request,
        release={
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        model_cache={
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        preflight=_preflight(),
    )
    assert result["status"] == "blocked"
    assert "runpod_request_release_image_differs_from_admission" in result["blockers"]


def test_canary_preparation_rejects_different_model_cache_path() -> None:
    request = _request()
    request["provider_request_shape"]["cache"] = {
        "paths": {"groot_oscar_models": "/workspace/other-cache"}
    }
    result = prepare_canary_launch(
        request=request,
        release={
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        model_cache={
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        preflight=_preflight(),
    )
    assert result["status"] == "blocked"
    assert "runpod_request_model_cache_path_differs_from_admission" in result["blockers"]


def test_execute_refresh_rechecks_every_mutable_provider_fact() -> None:
    observed: dict[str, object] = {}

    def volume_getter(volume_id: str):
        observed["volume_id"] = volume_id
        return 200, {"id": volume_id, "dataCenterId": "US-TX-3", "size": 50}

    def capacity_probe(request):
        observed["capacity_request"] = request
        return {
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
                    "single_gpu_offer_requested": True,
                    "single_gpu_offer_available": True,
                    "on_demand_price_usd_per_hour": 0.44,
                }
            ],
        }

    def inventory_probe(prefix: str):
        observed["prefix"] = prefix
        return {"api_confirmed": True, "live_resource_count": 0}

    result = refresh_runpod_preflight(
        preflight=_preflight(),
        volume_getter=volume_getter,
        capacity_probe=capacity_probe,
        inventory_probe=inventory_probe,
        clock=lambda: 1000.0,
        process_argv_probe=lambda _pid: (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--pod-name-prefix",
            "blueprint-groot-oscar-canary-",
            "--deadline-epoch",
            "1900.0",
        ),
    )
    assert result["status"] == "verified"
    assert result["observed_at_epoch"] == 1000.0
    assert observed["volume_id"] == "volume-1"
    assert observed["capacity_request"] == {
        "cloudType": "SECURE",
        "gpuTypeIds": ["NVIDIA A40"],
        "dataCenterIds": ["US-TX-3"],
        "allowedCudaVersions": ["12.6"],
        "requires_rtx": True,
    }
    assert observed["prefix"] == "blueprint-groot-oscar-canary-"


def test_execute_blocks_before_adapter_when_launch_refresh_fails(
    tmp_path, monkeypatch
) -> None:
    class Provider:
        def _key(self):
            return "test-key"

        def capacity_preflight(self, _request):
            return {"status": "blocked", "viable_gpu_types": []}

        def billable_inventory(self, *, name_prefix):
            del name_prefix
            return {"api_confirmed": False, "live_resource_count": 0}

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.get_render_provider",
        lambda _name: Provider(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary._runpod_call",
        lambda *args, **kwargs: (404, {}),
    )

    def adapter_must_not_run(**_kwargs):
        raise AssertionError("provider adapter reached after blocked launch refresh")

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.run_runpod_provider_adapter",
        adapter_must_not_run,
    )
    payloads = {
        "request.json": _request(),
        "release.json": {
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        "models.json": {
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        "preflight.json": _preflight(),
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    result = run_canary(
        provider_launch_request=tmp_path / "request.json",
        release_evidence=tmp_path / "release.json",
        model_cache_evidence=tmp_path / "models.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        pod_name="blueprint-groot-oscar-canary-test",
        execute=True,
    )
    assert result["status"] == "blocked"
    assert "runpod_preflight_bundle_not_verified" in result["blockers"]
    refresh = json.loads(
        (tmp_path / "runpod_preflight_launch_refresh.json").read_text(
            encoding="utf-8"
        )
    )
    assert refresh["status"] == "blocked"
    assert refresh["provider_mutations_performed"] == 0
