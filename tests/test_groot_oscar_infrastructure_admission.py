import hashlib
import io
import json
import tarfile
from pathlib import Path

from blueprint_pipeline.groot_oscar_carrier_remote_build_packet import (
    PACKET_DIRNAME,
    render_remote_build_script,
)
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


def test_cpu_build_execution_rejects_explicit_packet_kind_mismatch() -> None:
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
            "checks": {"packet_kind": "model_cache_s3"},
        },
        live_machine=live,
    )
    assert result["status"] == "blocked"
    assert "cpu_builder_live_capability_packet_kind_mismatch" in result["blockers"]


def test_runtime_bundle_execution_requires_live_docker_on_model_cache_builder() -> None:
    live = build_live_machine_capability_evidence(
        {
            "observation_source": "live_machine_probe",
            "system": "Linux",
            "architecture": "x86_64",
            "mount_path": "/",
            "free_bytes": 200 * 1024**3,
            "docker_cli_present": False,
            "docker_daemon_responding": False,
            "python3_available": True,
            "python_version": "3.12",
            "python_venv_available": True,
            "dns_resolution_verified": True,
            "outbound_https_verified": True,
            "s3_endpoint_host": "s3api-eur-is-1.runpod.io",
            "builder_ready_marker": True,
        },
        packet_kind="model_cache_s3",
        expected_s3_endpoint_host="s3api-eur-is-1.runpod.io",
    )
    result = build_cpu_build_execution_admission(
        allocation_admission={
            "schema_version": "groot_oscar_build_plane_admission.v1",
            "status": "admitted",
            "checks": {"packet_kind": "model_cache_s3"},
        },
        live_machine=live,
        runtime_bundle_requested=True,
    )

    assert result["status"] == "blocked"
    assert "cpu_builder_runtime_bundle_docker_cli_missing" in result["blockers"]
    assert "cpu_builder_runtime_bundle_docker_daemon_unavailable" in result["blockers"]


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


def _carrier_packet(
    tmp_path: Path, *, tamper_script: bool = False, executable_script: bool = True
) -> dict:
    image_ref = "docker.io/example/carrier:versioned"
    base_ref = "docker.io/example/base@sha256:" + "a" * 64
    dockerfile = b"ARG ISAAC_CARRIER_BASE\nFROM ${ISAAC_CARRIER_BASE}\n"
    dockerfile_sha256 = hashlib.sha256(dockerfile).hexdigest()
    script = render_remote_build_script(
        image_ref=image_ref,
        base_image_ref=base_ref,
        source_commit=COMMIT,
        dockerfile_sha256=dockerfile_sha256,
    ).encode()
    if tamper_script:
        script += b"\necho unbound-command\n"
    payloads = {
        f"{PACKET_DIRNAME}/README.md": b"carrier packet\n",
        f"{PACKET_DIRNAME}/context/Dockerfile": dockerfile,
        f"{PACKET_DIRNAME}/remote_build_groot_oscar_carrier.sh": script,
    }
    member_digests = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    tarball = tmp_path / "carrier.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        for name in sorted(payloads):
            info = tarfile.TarInfo(name)
            info.size = len(payloads[name])
            info.mode = 0o755 if name.endswith(".sh") and executable_script else 0o644
            archive.addfile(info, io.BytesIO(payloads[name]))
    return {
        **_packet(),
        "schema_version": "groot_oscar_carrier_remote_build_packet.v1",
        "packet_kind": "carrier_image",
        "carrier_image_ref": image_ref,
        "carrier_base_image_ref": base_ref,
        "carrier_dockerfile_sha256": dockerfile_sha256,
        "tarball_path": str(tarball),
        "tarball_sha256": hashlib.sha256(tarball.read_bytes()).hexdigest(),
        "archive_members": sorted(payloads),
        "archive_member_sha256": member_digests,
        "archive_member_manifest_sha256": hashlib.sha256(
            json.dumps(member_digests, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
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


def test_build_plane_admits_typed_carrier_image_packet(tmp_path: Path) -> None:
    packet = _carrier_packet(tmp_path)
    result = build_build_plane_admission(packet=packet, builder=_builder(), spend=_spend())
    assert result["status"] == "admitted"
    assert result["checks"]["packet_kind"] == "carrier_image"

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
        },
        packet_kind="carrier_image",
    )
    assert live["status"] == "verified"


def test_build_plane_rejects_unbound_carrier_script_before_allocation(
    tmp_path: Path,
) -> None:
    packet = _carrier_packet(tmp_path, tamper_script=True)
    result = build_build_plane_admission(packet=packet, builder=_builder(), spend=_spend())
    assert result["status"] == "blocked"
    assert "builder_carrier_archive_script_binding_mismatch" in result["blockers"]


def test_build_plane_rejects_nonexecutable_carrier_script_before_allocation(
    tmp_path: Path,
) -> None:
    packet = _carrier_packet(tmp_path, executable_script=False)
    result = build_build_plane_admission(packet=packet, builder=_builder(), spend=_spend())
    assert result["status"] == "blocked"
    assert "builder_carrier_archive_script_not_executable" in result["blockers"]


def test_build_plane_rejects_malformed_carrier_packet_before_allocation() -> None:
    packet = {
        **_packet(),
        "schema_version": "groot_oscar_carrier_remote_build_packet.v0",
        "packet_kind": "carrier_image",
        "carrier_image_ref": "docker.io/example/carrier",
        "carrier_base_image_ref": "docker.io/example/base:latest",
        "carrier_dockerfile_sha256": "bad",
    }
    result = build_build_plane_admission(packet=packet, builder=_builder(), spend=_spend())
    assert result["status"] == "blocked"
    assert "builder_carrier_packet_schema_invalid" in result["blockers"]
    assert "builder_carrier_image_ref_not_versioned" in result["blockers"]
    assert "builder_carrier_base_image_not_digest_pinned" in result["blockers"]
    assert "builder_carrier_dockerfile_sha256_invalid" in result["blockers"]


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


def test_build_plane_rejects_unknown_packet_kind() -> None:
    packet = {**_packet(), "packet_kind": "arbitrary_script"}
    result = build_build_plane_admission(packet=packet, builder=_builder(), spend=_spend())
    assert result["status"] == "blocked"
    assert "builder_packet_kind_unsupported" in result["blockers"]


def test_model_cache_build_plane_requires_python_contract_not_docker() -> None:
    packet = {
        **_packet(),
        "packet_kind": "model_cache_s3",
        "data_center_id": "US-WA-1",
    }
    builder = {
        **_builder(),
        "purpose": "model_cache_s3",
        "docker_daemon_verified": False,
        "docker_buildx_verified": False,
        "registry_push_auth_file_verified": False,
        "python_runtime_verified": True,
        "python_version": "3.12",
        "dependency_lock_verified": True,
        "dependency_wheelhouse_verified": True,
        "dns_resolution_verified": True,
        "outbound_https_verified": True,
        "s3_endpoint_host": "s3api-us-wa-1.runpod.io",
    }
    result = build_build_plane_admission(packet=packet, builder=builder, spend=_spend())
    assert result["status"] == "admitted"
    assert result["checks"]["execution_runtime_ready"] is True

    builder["dependency_wheelhouse_verified"] = False
    blocked = build_build_plane_admission(packet=packet, builder=builder, spend=_spend())
    assert blocked["status"] == "blocked"
    assert "builder_dependency_wheelhouse_not_verified" in blocked["blockers"]


def test_model_cache_live_capability_does_not_require_docker() -> None:
    observation = {
        "observation_source": "live_machine_probe",
        "system": "Linux",
        "architecture": "x86_64",
        "mount_path": "/",
        "free_bytes": MIN_BUILD_FREE_BYTES,
        "docker_cli_present": False,
        "docker_daemon_responding": False,
        "docker_buildx_available": False,
        "python3_available": True,
        "python_version": "3.12",
        "python_venv_available": True,
        "dns_resolution_verified": True,
        "outbound_https_verified": True,
        "s3_endpoint_host": "s3api-us-wa-1.runpod.io",
        "builder_ready_marker": True,
    }
    result = build_live_machine_capability_evidence(
        observation,
        packet_kind="model_cache_s3",
        expected_s3_endpoint_host="s3api-us-wa-1.runpod.io",
    )
    assert result["status"] == "verified"

    blocked = build_live_machine_capability_evidence(
        {**observation, "python_venv_available": False},
        packet_kind="model_cache_s3",
        expected_s3_endpoint_host="s3api-us-wa-1.runpod.io",
    )
    assert "live_machine_python_venv_missing" in blocked["blockers"]


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
        "source_commit": COMMIT,
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
        "schema_version": "groot_oscar_external_model_cache_verification.v2",
        "status": "passed",
        "model_manifest_digest": MANIFEST_DIGEST,
        "cache_root": "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1",
        "provider_volume_id": "volume-1",
        "verified_size_bytes": 20 * 1024**3,
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
    }


def test_runpod_serve_plane_admits_only_published_volume_ready_worker() -> None:
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=_models(),
        volume=_volume(),
        runtime=_runtime(),
        spend=_serve_spend(),
        expected_source_commit=COMMIT,
    )
    assert result["status"] == "admitted"
    assert result["blockers"] == []


def test_runpod_serve_plane_rejects_release_from_different_source_commit() -> None:
    release = _release()
    release["source_commit"] = "b" * 40
    result = build_runpod_serve_plane_admission(
        release=release,
        model_cache=_models(),
        volume=_volume(),
        runtime=_runtime(),
        spend=_serve_spend(),
        expected_source_commit=COMMIT,
    )
    assert result["status"] == "blocked"
    assert result["source_bound"] is False
    assert "runpod_release_source_commit_mismatch" in result["blockers"]


def test_runpod_serve_plane_rejects_legacy_weak_model_verification() -> None:
    models = _models()
    models["schema_version"] = "groot_oscar_external_model_cache_verification.v1"
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=models,
        volume=_volume(),
        runtime=_runtime(),
        spend=_serve_spend(),
        expected_source_commit=COMMIT,
    )
    assert result["status"] == "blocked"
    assert "runpod_model_cache_verification_schema_invalid" in result["blockers"]


def test_runpod_serve_plane_binds_cache_verification_to_volume_and_path() -> None:
    models = _models()
    models.update({"provider_volume_id": "other-volume", "cache_root": "/workspace/other"})
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=models,
        volume=_volume(),
        runtime=_runtime(),
        spend=_serve_spend(),
        expected_source_commit=COMMIT,
    )
    assert result["status"] == "blocked"
    assert "runpod_model_cache_verification_volume_mismatch" in result["blockers"]
    assert "runpod_model_cache_verification_path_mismatch" in result["blockers"]


def test_runpod_serve_plane_rejects_volume_smaller_than_verified_cache() -> None:
    models = _models()
    models["verified_size_bytes"] = 60 * 1024**3
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=models,
        volume=_volume(),
        runtime=_runtime(),
        spend=_serve_spend(),
        expected_source_commit=COMMIT,
    )
    assert result["status"] == "blocked"
    assert "runpod_network_volume_smaller_than_verified_model_cache" in result["blockers"]


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
        expected_source_commit=COMMIT,
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
        expected_source_commit=COMMIT,
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
        expected_source_commit=COMMIT,
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
        expected_source_commit=COMMIT,
    )
    assert "runpod_cuda_version_differs_from_release" in result["blockers"]


def test_runpod_serve_plane_rejects_capacity_from_different_data_center() -> None:
    runtime = _runtime()
    runtime["capacity_data_center_id"] = "EU-RO-1"
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=_models(),
        volume=_volume(),
        runtime=runtime,
        spend=_serve_spend(),
        expected_source_commit=COMMIT,
    )
    assert result["status"] == "blocked"
    assert "runpod_gpu_capacity_not_verified_in_volume_data_center" in result["blockers"]


def test_runpod_serve_plane_rejects_capacity_not_filtered_for_cuda() -> None:
    runtime = _runtime()
    runtime["capacity_allowed_cuda_versions"] = ["12.5"]
    result = build_runpod_serve_plane_admission(
        release=_release(),
        model_cache=_models(),
        volume=_volume(),
        runtime=runtime,
        spend=_serve_spend(),
        expected_source_commit=COMMIT,
    )
    assert result["status"] == "blocked"
    assert "runpod_gpu_capacity_not_verified_for_cuda_version" in result["blockers"]
