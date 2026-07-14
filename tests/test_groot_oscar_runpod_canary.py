from blueprint_pipeline.groot_oscar_runpod_canary import prepare_canary_launch


DIGEST = "docker.io/example/release@sha256:" + "a" * 64


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
            "checks": {"models_cached_offline": True},
        },
        preflight=_preflight(),
    )
    assert result["status"] == "admitted"
    shape = result["bound_request"]["provider_request_shape"]
    assert shape["network_volume_id"] == "volume-1"
    assert shape["data_center_id"] == "US-TX-3"
    assert shape["allowed_cuda_versions"] == ["12.6"]


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
            "checks": {"models_cached_offline": True},
        },
        preflight=_preflight(),
    )
    assert result["status"] == "blocked"
    assert "runpod_request_release_image_differs_from_admission" in result["blockers"]
