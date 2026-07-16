from __future__ import annotations

from blueprint_pipeline.groot_oscar_runpod_carrier_volume import (
    CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    DEFAULT_MODEL_CACHE_ROOT,
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    DEFAULT_RUNTIME_ROOT,
    RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
    RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
)
from blueprint_pipeline.groot_oscar_runpod_persistent_carrier import (
    PERSISTENT_CARRIER_IMAGE_REF,
    PERSISTENT_CARRIER_PROBE_KIND,
    prepare_persistent_carrier_launch,
)


RELEASE_REF = "docker.io/blueprint/release@sha256:" + "1" * 64
CARRIER_REF = PERSISTENT_CARRIER_IMAGE_REF


def test_persistent_carrier_uses_published_system_complete_image() -> None:
    assert CARRIER_REF == (
        "docker.io/nijelhunt/blueprint-groot-oscar-carrier@sha256:"
        "d8f7e8c92e87cebc6ae0c15ebc94d5a083811ed3f63d2b2047a235e80b2de42d"
    )


def _carrier() -> dict:
    return {
        "schema_version": CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
        "status": "verified",
        "carrier_image_ref": CARRIER_REF,
        "network_volume": {"id": "volume123", "data_center_id": "EUR-IS-1", "size_gib": 120},
        "runtime_bundle": {
            "manifest_schema_version": RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "source_release_image_ref": RELEASE_REF,
            "root": DEFAULT_RUNTIME_ROOT,
            "archive_path": DEFAULT_RUNTIME_ARCHIVE_PATH,
            "manifest_path": DEFAULT_RUNTIME_MANIFEST_PATH,
            "archive_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
        },
        "runtime_source_release": {
            "schema_version": RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
            "status": "verified",
            "release_image_ref": RELEASE_REF,
            "source_commit": "a" * 40,
            "thin_release_contract_sha256": "6" * 64,
            "models_externalized": True,
        },
        "model_cache": {
            "status": "verified",
            "root": DEFAULT_MODEL_CACHE_ROOT,
            "manifest_sha256": "5" * 64,
            "manifest_digest": "sha256:" + "7" * 64,
        },
        "s3_transfer_verification": {
            "upload_completed": True,
            "full_redownload_sha256_verified": True,
            "provider_volume_id": "volume123",
            "data_center_id": "EUR-IS-1",
        },
    }


def _release() -> dict:
    return {
        "resolved_digest_ref": RELEASE_REF,
        "thin_release_contract": {"status": "passed"},
        "runnable_platform": "linux/amd64",
        "required_cuda_version": "12.8",
        "required_cuda_version_source": "image_config_env:BLUEPRINT_REQUIRED_CUDA_VERSION",
    }


def _model() -> dict:
    return {
        "schema_version": "groot_oscar_external_model_cache_verification.v2",
        "status": "passed",
        "model_manifest_digest": "sha256:" + "6" * 64,
        "cache_root": DEFAULT_MODEL_CACHE_ROOT,
        "provider_volume_id": "volume123",
        "verified_size_bytes": 10 * 1024**3,
        "checks": {"models_cached_offline": True},
    }


def _preflight(gpu: str = "NVIDIA A40") -> dict:
    return {
        "status": "verified",
        "volume": {
            "provider": "runpod",
            "provider_api_verified": True,
            "id": "volume123",
            "data_center_id": "EUR-IS-1",
            "size_bytes": 120 * 1024**3,
            "model_cache_path": DEFAULT_MODEL_CACHE_ROOT,
        },
        "runtime": {
            "provider": "runpod",
            "data_center_id": "EUR-IS-1",
            "capacity_data_center_id": "EUR-IS-1",
            "gpu_type_id": gpu,
            "provider_api_verified": True,
            "capacity_confidence": "advisory",
            "single_gpu_available": True,
            "required_cuda_version": "12.8",
            "allowed_cuda_versions": ["12.8"],
            "capacity_allowed_cuda_versions": ["12.8"],
            "warm_worker_only": True,
            "provider_inventory_verified_zero": True,
            "on_demand_price_usd_per_hour": 0.74,
        },
        "spend": {
            "paid_mutation_authorized": True,
            "max_spend_usd": 4.0,
            "hard_ttl_seconds": 18_600,
            "one_resource_limit": True,
            "independent_teardown_watchdog": True,
            "watchdog_armed_before_allocation": True,
        },
    }


def test_persistent_carrier_binds_exact_campaign_and_small_image() -> None:
    result = prepare_persistent_carrier_launch(
        request={"provider_request_shape": {}},
        release=_release(),
        model_cache=_model(),
        preflight=_preflight(),
        carrier_volume_admission=_carrier(),
        loop_step_count=5,
        max_wait_seconds=18_000,
    )

    assert result["status"] == "admitted"
    admission = result["admission"]
    assert admission["probe_kind"] == PERSISTENT_CARRIER_PROBE_KIND
    assert admission["campaign_contract"]["policy_call_count"] == 5
    assert admission["campaign_contract"]["learned_wam_generation_count"] == 4
    assert admission["campaign_contract"]["h100_allowed"] is False
    shape = result["bound_request"]["provider_request_shape"]
    assert shape["image"]["configured_image_ref"] == CARRIER_REF
    assert shape["network_volume_id"] == "volume123"
    assert shape["gpu"]["container_disk_in_gb"] == 240
    assert shape["gpu"]["volume_in_gb"] == 120
    assert shape["persistent_campaign"]["same_pod_required"] is True
    assert shape["claim_boundary"]["does_not_prove_semantic_task_success"] is True


def test_persistent_carrier_rejects_h100_wrong_loop_and_short_watchdog() -> None:
    preflight = _preflight("NVIDIA H100 PCIe")
    preflight["spend"]["hard_ttl_seconds"] = 480
    carrier = _carrier()
    carrier["carrier_image_ref"] = (
        "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:" + "2" * 64
    )
    result = prepare_persistent_carrier_launch(
        request={"provider_request_shape": {}},
        release=_release(),
        model_cache=_model(),
        preflight=preflight,
        carrier_volume_admission=carrier,
        loop_step_count=4,
        max_wait_seconds=480,
    )

    assert result["status"] == "blocked"
    assert "persistent_carrier_h100_disallowed" in result["blockers"]
    assert "persistent_carrier_exact_image_digest_mismatch" in result["blockers"]
    assert "persistent_carrier_requires_exactly_five_policy_calls" in result["blockers"]
    assert "persistent_carrier_requires_18000_second_run_bound" in result["blockers"]
    assert "persistent_carrier_watchdog_below_18600_seconds" in result["blockers"]
