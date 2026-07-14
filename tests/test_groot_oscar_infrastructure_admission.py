from blueprint_pipeline.groot_oscar_infrastructure_admission import (
    MIN_BUILD_FREE_BYTES,
    build_build_plane_admission,
    build_cpu_build_execution_admission,
    build_digitalocean_cpu_builder_profile_evidence,
    build_live_machine_capability_evidence,
    build_runpod_serve_plane_admission,
)


def test_live_machine_capability_requires_direct_observation() -> None:
    requested = build_live_machine_capability_evidence(
        {
            "observation_source": "requested_configuration",
            "system": "Linux",
            "architecture": "x86_64",
            "mount_path": "/",
            "free_bytes": 200 * 1024**3,
            "docker_cli_present": True,
            "docker_daemon_responding": True,
            "docker_buildx_available": True,
            "builder_ready_marker": True,
        }
    )
    assert requested["status"] == "blocked"
    assert "live_machine_observation_source_invalid" in requested["blockers"]


def test_cpu_build_execution_requires_verified_live_machine() -> None:
    live = build_live_machine_capability_evidence(
        {
            "observation_source": "live_machine_probe",
            "system": "Linux",
            "architecture": "x86_64",
            "mount_path": "/",
            "free_bytes": 200 * 1024**3,
            "docker_cli_present": True,
            "docker_daemon_responding": True,
            "docker_buildx_available": True,
            "builder_ready_marker": True,
        }
    )
    result = build_cpu_build_execution_admission(
        allocation_admission={
            "schema_version": "groot_oscar_build_plane_admission.v1",
            "status": "admitted",
        },
        live_machine=live,
    )
    assert result["status"] == "admitted"


COMMIT = "a" * 40
DIGEST_REF = "docker.io/example/release@sha256:" + "b" * 64
MANIFEST_DIGEST = "sha256:" + "c" * 64


def _packet() -> dict:
    return {
        "status": "ready",
        "source_commit": COMMIT,
        "source_worktree_dirty": False,
        "provider_launch_performed_by_packet": False,
    }


def _builder(provider: str = "digitalocean") -> dict:
    return {
        "provider": provider,
        "purpose": "image_build",
        "platform": "linux/amd64",
        "docker_daemon_verified": True,
        "docker_buildx_verified": True,
        "free_disk_bytes": MIN_BUILD_FREE_BYTES,
        "registry_push_auth_file_verified": True,
        "independent_teardown_watchdog": True,
        "ssh_host_key_sha256": "SHA256:" + "d" * 43,
        "ssh_host_key_independently_verified": True,
        "ssh_host_key_verification_method": "launch_bound_generated_host_key",
        "expected_source_commit": COMMIT,
    }


def _spend() -> dict:
    return {
        "paid_mutation_authorized": True,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 7200,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
    }


def test_cpu_build_execution_requires_verified_live_machine_evidence() -> None:
    allocation = build_build_plane_admission(packet=_packet(), builder=_builder(), spend=_spend())
    live = build_live_machine_capability_evidence(
        {
            "observation_source": "live_machine_probe",
            "system": "Linux",
            "architecture": "x86_64",
            "mount_path": "/",
            "free_bytes": MIN_BUILD_FREE_BYTES,
            "docker_cli_present": True,
            "docker_daemon_responding": True,
            "docker_buildx_available": True,
            "builder_ready_marker": True,
        }
    )
    assert (
        build_cpu_build_execution_admission(allocation_admission=allocation, live_machine=live)[
            "status"
        ]
        == "admitted"
    )
    catalog = {**live, "status": "blocked", "blockers": ["catalog_only"]}
    blocked = build_cpu_build_execution_admission(
        allocation_admission=allocation, live_machine=catalog
    )
    assert blocked["status"] == "blocked"
    assert "cpu_builder_live_capability_not_verified" in blocked["blockers"]


def test_live_machine_capability_waits_for_builder_initialization_marker() -> None:
    live = build_live_machine_capability_evidence(
        {
            "observation_source": "live_machine_probe",
            "system": "Linux",
            "architecture": "x86_64",
            "mount_path": "/",
            "free_bytes": MIN_BUILD_FREE_BYTES,
            "docker_cli_present": True,
            "docker_daemon_responding": True,
            "docker_buildx_available": True,
            "builder_ready_marker": False,
        }
    )
    assert live["status"] == "blocked"
    assert "live_machine_builder_initialization_incomplete" in live["blockers"]


def _serve_spend() -> dict:
    return {
        **_spend(),
        "hard_ttl_seconds": 1800,
        "watchdog_armed_before_allocation": True,
    }


def test_build_plane_admits_known_native_docker_builder() -> None:
    result = build_build_plane_admission(packet=_packet(), builder=_builder(), spend=_spend())
    assert result["status"] == "admitted"
    assert result["blockers"] == []
    assert result["checks"]["free_disk_at_least_120_gib"] is True


def test_build_plane_refuses_runpod_even_when_claimed_capabilities_are_true() -> None:
    result = build_build_plane_admission(
        packet=_packet(), builder=_builder("runpod"), spend=_spend()
    )
    assert result["status"] == "blocked"
    assert "runpod_pods_are_serve_plane_not_image_build_plane" in result["blockers"]


def test_build_plane_refuses_tofu_and_insufficient_disk() -> None:
    builder = _builder()
    builder.update(
        {
            "free_disk_bytes": MIN_BUILD_FREE_BYTES - 1,
            "ssh_host_key_independently_verified": False,
            "ssh_host_key_verification_method": "accept-new",
        }
    )
    result = build_build_plane_admission(packet=_packet(), builder=builder, spend=_spend())
    assert "builder_free_disk_below_120_gib" in result["blockers"]
    assert "builder_ssh_host_key_not_independently_verified" in result["blockers"]
    assert "builder_ssh_host_key_verification_method_unsafe" in result["blockers"]


def test_known_digitalocean_cpu_builder_profile_requires_live_catalog_match() -> None:
    size = {
        "slug": "s-8vcpu-16gb-amd",
        "available": True,
        "disk": 320,
        "vcpus": 8,
        "memory": 16384,
        "price_hourly": 0.16667,
        "regions": ["sfo3"],
    }
    result = build_digitalocean_cpu_builder_profile_evidence(
        size=size, region="sfo3", observed_live_builders=0
    )
    assert result["status"] == "verified"
    assert result["observed"]["disk_gb"] == 320


def test_known_digitalocean_cpu_builder_profile_blocks_drift_and_overlap() -> None:
    size = {
        "slug": "s-8vcpu-16gb-amd",
        "available": True,
        "disk": 100,
        "vcpus": 8,
        "memory": 16384,
        "price_hourly": 0.20,
        "regions": ["nyc3"],
    }
    result = build_digitalocean_cpu_builder_profile_evidence(
        size=size, region="sfo3", observed_live_builders=1
    )
    assert "digitalocean_builder_disk_below_profile" in result["blockers"]
    assert "digitalocean_builder_hourly_rate_above_profile" in result["blockers"]
    assert "digitalocean_builder_region_not_available" in result["blockers"]
    assert "digitalocean_builder_one_resource_limit_not_clear" in result["blockers"]


def _release() -> dict:
    return {
        "resolved_digest_ref": DIGEST_REF,
        "thin_release_contract_status": "passed",
        "runnable_platform": "linux/amd64",
        "required_cuda_version": "12.6",
        "required_cuda_version_source": (
            "image_config_env:BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION"
        ),
    }


def _models() -> dict:
    return {
        "status": "passed",
        "model_manifest_digest": MANIFEST_DIGEST,
        "checks": {"models_cached_offline": True},
    }


def _volume() -> dict:
    return {
        "provider": "runpod",
        "provider_api_verified": True,
        "id": "volume-1",
        "data_center_id": "US-TX-3",
        "size_bytes": 50 * 1024**3,
        "model_cache_path": "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1",
    }


def _runtime() -> dict:
    return {
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
    }


def test_runpod_serve_plane_admits_only_published_volume_ready_worker() -> None:
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=_models(),
        volume=_volume(),
        runtime=_runtime(),
        spend=_serve_spend(),
    )
    assert result["status"] == "admitted"
    assert result["blockers"] == []


def test_runpod_serve_plane_blocks_missing_volume_and_cold_start() -> None:
    volume = _volume()
    volume.update({"id": "", "model_cache_path": "/models/cache"})
    runtime = _runtime()
    runtime["warm_worker_only"] = False
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=_models(),
        volume=volume,
        runtime=runtime,
        spend=_serve_spend(),
    )
    assert "runpod_network_volume_id_missing" in result["blockers"]
    assert "runpod_model_cache_path_must_be_under_workspace" in result["blockers"]
    assert "runpod_customer_cold_start_disallowed" in result["blockers"]


def test_runpod_serve_plane_rejects_operator_asserted_cuda_version() -> None:
    release = _release()
    release["required_cuda_version_source"] = "operator_input"
    result = build_runpod_serve_plane_admission(
        release=release,
        model_cache=_models(),
        volume=_volume(),
        runtime=_runtime(),
        spend=_serve_spend(),
    )
    assert result["status"] == "blocked"
    assert "runpod_release_cuda_not_registry_config_verified" in result["blockers"]


def test_runpod_serve_plane_blocks_unknown_capacity_cuda_and_unarmed_watchdog() -> None:
    runtime = _runtime()
    runtime.update(
        {
            "provider_api_verified": False,
            "capacity_confidence": "unknown",
            "single_gpu_available": False,
            "allowed_cuda_versions": [],
        }
    )
    spend = _serve_spend()
    spend["watchdog_armed_before_allocation"] = False
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=_models(),
        volume=_volume(),
        runtime=runtime,
        spend=spend,
    )
    assert "runpod_gpu_capacity_not_provider_verified" in result["blockers"]
    assert "runpod_single_gpu_availability_unknown" in result["blockers"]
    assert "runpod_cuda_compatibility_not_bound" in result["blockers"]
    assert "runpod_teardown_watchdog_not_armed_before_allocation" in result["blockers"]


def test_runpod_serve_plane_rejects_cuda_constraint_different_from_release() -> None:
    runtime = _runtime()
    runtime.update({"required_cuda_version": "12.8", "allowed_cuda_versions": ["12.8"]})
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=_models(),
        volume=_volume(),
        runtime=runtime,
        spend=_serve_spend(),
    )
    assert "runpod_cuda_version_differs_from_release" in result["blockers"]
