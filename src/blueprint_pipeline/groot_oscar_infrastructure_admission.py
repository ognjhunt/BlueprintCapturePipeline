"""Fail-closed infrastructure admission for thin-image build and serve planes.

The durable GR00T + OSCAR layout has two intentionally different execution
planes:

* a native linux/amd64 Docker build host with ample local disk; and
* a RunPod GPU serving pod that consumes already-published image digests and an
  already-populated model volume.

RunPod serving allocations are never used to discover image-builder
capabilities.  This module is pure so admission can run before any paid API
mutation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .common import write_json

BUILD_SCHEMA_VERSION = "groot_oscar_build_plane_admission.v1"
SERVE_SCHEMA_VERSION = "groot_oscar_runpod_serve_plane_admission.v1"
LIVE_MACHINE_SCHEMA_VERSION = "blueprint.live_machine_capability.v1"
CPU_BUILD_EXECUTION_SCHEMA_VERSION = "blueprint.cpu_build_execution_admission.v1"
MIN_BUILD_FREE_BYTES = 120 * 1024**3
MAX_BUILD_TTL_SECONDS = 2 * 60 * 60
MAX_CANARY_TTL_SECONDS = 30 * 60
MIN_MODEL_VOLUME_BYTES = 30 * 1024**3
DIGITALOCEAN_CPU_BUILDER_PROFILE = {
    "profile_id": "digitalocean-s-8vcpu-16gb-amd-ubuntu-24-04-v1",
    "provider": "digitalocean",
    "size_slug": "s-8vcpu-16gb-amd",
    "image_slug": "ubuntu-24-04-x64",
    "minimum_vcpus": 8,
    "minimum_memory_mb": 16384,
    "minimum_disk_gb": 320,
    "maximum_known_hourly_rate_usd": 0.16667,
    "provisioning_contract": {
        "docker_packages": ["docker.io", "docker-buildx"],
        "registry_auth": "file_mounted_after_ssh",
        "source_packet": "checksum_verified_archive",
        "host_key": "launch_bound_generated_ed25519",
    },
}
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")
_DIGEST_REF = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_SSH_HOST_KEY_SHA256 = re.compile(r"\ASHA256:[A-Za-z0-9+/]{43}\Z")


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def build_live_machine_capability_evidence(
    observation: Mapping[str, Any],
    *,
    minimum_free_bytes: int = MIN_BUILD_FREE_BYTES,
    required_architecture: str = "x86_64",
) -> dict[str, Any]:
    """Validate facts measured by a probe running on the machine itself.

    Catalog rows and requested VM configuration are deliberately not accepted
    here.  The caller must pass the direct output of the live-machine probe.
    """

    blockers: list[str] = []
    if observation.get("observation_source") != "live_machine_probe":
        blockers.append("live_machine_observation_source_invalid")
    if observation.get("system") != "Linux":
        blockers.append("live_machine_linux_not_verified")
    architecture = _string(observation.get("architecture"))
    if architecture != required_architecture:
        blockers.append("live_machine_architecture_mismatch")
    mount_path = _string(observation.get("mount_path"))
    if not mount_path.startswith("/"):
        blockers.append("live_machine_mount_path_invalid")
    free_bytes = observation.get("free_bytes")
    if type(free_bytes) is not int or free_bytes < minimum_free_bytes:
        blockers.append("live_machine_free_space_below_minimum")
    if observation.get("docker_cli_present") is not True:
        blockers.append("live_machine_docker_cli_missing")
    if observation.get("docker_daemon_responding") is not True:
        blockers.append("live_machine_docker_daemon_unavailable")
    if observation.get("docker_buildx_available") is not True:
        blockers.append("live_machine_docker_buildx_unavailable")
    if observation.get("builder_ready_marker") is not True:
        blockers.append("live_machine_builder_initialization_incomplete")
    return {
        "schema_version": LIVE_MACHINE_SCHEMA_VERSION,
        "status": "verified" if not blockers else "blocked",
        "blockers": blockers,
        "observation_source": observation.get("observation_source"),
        "system": observation.get("system"),
        "architecture": architecture or None,
        "mount_path": mount_path or None,
        "free_bytes": free_bytes,
        "docker_cli_present": observation.get("docker_cli_present"),
        "docker_daemon_responding": observation.get("docker_daemon_responding"),
        "docker_buildx_available": observation.get("docker_buildx_available"),
        "builder_ready_marker": observation.get("builder_ready_marker"),
        "minimum_free_bytes": minimum_free_bytes,
        "required_architecture": required_architecture,
        "claim_boundary": {
            "requested_configuration_is_not_capability_evidence": True,
            "provider_catalog_is_not_live_machine_evidence": True,
        },
    }


def build_cpu_build_execution_admission(
    *,
    allocation_admission: Mapping[str, Any],
    live_machine: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit build execution only after allocation and a live host probe."""

    blockers: list[str] = []
    if allocation_admission.get("schema_version") != BUILD_SCHEMA_VERSION:
        blockers.append("cpu_builder_allocation_admission_schema_invalid")
    if allocation_admission.get("status") != "admitted":
        blockers.append("cpu_builder_allocation_not_admitted")
    if live_machine.get("schema_version") != LIVE_MACHINE_SCHEMA_VERSION:
        blockers.append("cpu_builder_live_capability_schema_invalid")
    if live_machine.get("status") != "verified":
        blockers.append("cpu_builder_live_capability_not_verified")
    blockers.extend(f"live_capability:{item}" for item in live_machine.get("blockers", []))
    return {
        "schema_version": CPU_BUILD_EXECUTION_SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "allocation_admission_schema_version": allocation_admission.get("schema_version"),
        "live_machine_capability": dict(live_machine),
        "claim_boundary": {
            "allocation_admission_is_not_build_execution_admission": True,
            "live_probe_is_required_after_allocation": True,
        },
    }


def build_digitalocean_cpu_builder_profile_evidence(
    *, size: Mapping[str, Any], region: str, observed_live_builders: int
) -> dict[str, Any]:
    """Validate the known 320 GB CPU builder against a live DO size row."""

    profile = DIGITALOCEAN_CPU_BUILDER_PROFILE
    blockers: list[str] = []
    if size.get("slug") != profile["size_slug"]:
        blockers.append("digitalocean_builder_size_slug_mismatch")
    if size.get("available") is not True:
        blockers.append("digitalocean_builder_size_not_available")
    if type(size.get("disk")) is not int or size["disk"] < profile["minimum_disk_gb"]:
        blockers.append("digitalocean_builder_disk_below_profile")
    if type(size.get("vcpus")) is not int or size["vcpus"] < profile["minimum_vcpus"]:
        blockers.append("digitalocean_builder_vcpus_below_profile")
    if type(size.get("memory")) is not int or size["memory"] < profile["minimum_memory_mb"]:
        blockers.append("digitalocean_builder_memory_below_profile")
    hourly = size.get("price_hourly")
    if not _positive_number(hourly) or hourly > profile["maximum_known_hourly_rate_usd"]:
        blockers.append("digitalocean_builder_hourly_rate_above_profile")
    regions = size.get("regions") if isinstance(size.get("regions"), list) else []
    if not region or region not in regions:
        blockers.append("digitalocean_builder_region_not_available")
    if type(observed_live_builders) is not int or observed_live_builders != 0:
        blockers.append("digitalocean_builder_one_resource_limit_not_clear")
    return {
        "schema_version": "groot_oscar_digitalocean_builder_profile_evidence.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": blockers,
        "profile": profile,
        "selected_region": region or None,
        "observed": {
            "available": size.get("available"),
            "disk_gb": size.get("disk"),
            "vcpus": size.get("vcpus"),
            "memory_mb": size.get("memory"),
            "price_hourly_usd": hourly,
            "live_builder_count": observed_live_builders,
        },
        "claim_boundary": {
            "catalog_verification_is_not_droplet_allocation": True,
            "catalog_verification_is_not_docker_runtime_probe": True,
            "catalog_verification_is_not_image_build": True,
        },
    }


def build_runpod_network_volume_evidence(
    *, provider_payload: Mapping[str, Any], expected_volume_id: str, model_cache_path: str
) -> dict[str, Any]:
    """Normalize the authoritative RunPod ``GET /networkvolumes/{id}`` row."""

    observed_id = _string(provider_payload.get("id"))
    data_center_id = _string(provider_payload.get("dataCenterId"))
    size_gib = provider_payload.get("size")
    blockers: list[str] = []
    if not expected_volume_id or observed_id != expected_volume_id:
        blockers.append("runpod_network_volume_provider_id_mismatch")
    if not data_center_id:
        blockers.append("runpod_network_volume_data_center_missing")
    if type(size_gib) is not int or size_gib <= 0:
        blockers.append("runpod_network_volume_size_invalid")
    size_bytes = size_gib * 1024**3 if type(size_gib) is int and size_gib > 0 else None
    return {
        "schema_version": "groot_oscar_runpod_network_volume_evidence.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": blockers,
        "provider": "runpod",
        "provider_api_verified": not blockers,
        "id": observed_id or None,
        "data_center_id": data_center_id or None,
        "size_bytes": size_bytes,
        "model_cache_path": model_cache_path,
        "raw_provider_response_recorded": False,
    }


def build_runpod_gpu_runtime_evidence(
    *,
    capacity: Mapping[str, Any],
    gpu_type_id: str,
    data_center_id: str,
    required_cuda_version: str,
    provider_inventory_verified_zero: bool,
) -> dict[str, Any]:
    """Bind a selected GPU to a provider capacity row and CUDA/DC constraints."""

    rows = capacity.get("viable_gpu_types")
    rows = rows if isinstance(rows, list) else []
    selected = next(
        (
            row
            for row in rows
            if isinstance(row, Mapping) and _string(row.get("gpu_type_id")) == gpu_type_id
        ),
        {},
    )
    counts = selected.get("available_gpu_counts")
    counts = counts if isinstance(counts, list) else []
    confidence = _string(selected.get("capacity_confidence"))
    single_available = confidence == "advisory" and 1 in counts
    provider_verified = capacity.get("status") == "available" and bool(selected)
    return {
        "schema_version": "groot_oscar_runpod_gpu_runtime_evidence.v1",
        "provider": "runpod",
        "provider_api_verified": provider_verified,
        "data_center_id": data_center_id,
        "gpu_type_id": gpu_type_id,
        "capacity_confidence": confidence or "unknown",
        "single_gpu_available": single_available,
        "on_demand_price_usd_per_hour": selected.get("on_demand_price_usd_per_hour"),
        "required_cuda_version": required_cuda_version,
        "allowed_cuda_versions": [required_cuda_version] if required_cuda_version else [],
        "warm_worker_only": True,
        "provider_inventory_verified_zero": provider_inventory_verified_zero,
        "launch_constraints": {
            "dataCenterIds": [data_center_id] if data_center_id else [],
            "allowedCudaVersions": [required_cuda_version] if required_cuda_version else [],
        },
        "claim_boundary": {
            "capacity_snapshot_is_not_reservation": True,
            "provider_create_response_is_authoritative": True,
        },
    }


def build_build_plane_admission(
    *,
    packet: Mapping[str, Any],
    builder: Mapping[str, Any],
    spend: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit a known Docker builder before a provider allocation is created."""

    blockers: list[str] = []
    provider = _string(builder.get("provider")).lower()
    if provider in {"runpod", "runpod_pod", "runpod-pod"}:
        blockers.append("runpod_pods_are_serve_plane_not_image_build_plane")
    if _string(builder.get("purpose")) != "image_build":
        blockers.append("builder_purpose_must_be_image_build")
    if _string(builder.get("platform")) != "linux/amd64":
        blockers.append("builder_native_linux_amd64_not_verified")
    if builder.get("docker_daemon_verified") is not True:
        blockers.append("builder_docker_daemon_not_verified")
    if builder.get("docker_buildx_verified") is not True:
        blockers.append("builder_docker_buildx_not_verified")
    free_bytes = builder.get("free_disk_bytes")
    if type(free_bytes) is not int or free_bytes < MIN_BUILD_FREE_BYTES:
        blockers.append("builder_free_disk_below_120_gib")
    if builder.get("registry_push_auth_file_verified") is not True:
        blockers.append("builder_file_based_registry_push_auth_not_verified")
    if builder.get("independent_teardown_watchdog") is not True:
        blockers.append("builder_independent_teardown_watchdog_missing")

    if provider not in {"local", "github_actions"}:
        fingerprint = _string(builder.get("ssh_host_key_sha256"))
        if not _SSH_HOST_KEY_SHA256.fullmatch(fingerprint):
            blockers.append("builder_ssh_host_key_fingerprint_missing")
        if builder.get("ssh_host_key_independently_verified") is not True:
            blockers.append("builder_ssh_host_key_not_independently_verified")
        if _string(builder.get("ssh_host_key_verification_method")) in {
            "",
            "trust_on_first_use",
            "accept-new",
            "known_hosts_delete_and_retry",
        }:
            blockers.append("builder_ssh_host_key_verification_method_unsafe")

    packet_commit = _string(packet.get("source_commit"))
    if packet.get("status") != "ready":
        blockers.append("remote_build_packet_not_ready")
    if not _COMMIT.fullmatch(packet_commit):
        blockers.append("remote_build_packet_source_commit_invalid")
    if packet.get("source_worktree_dirty") is not False:
        blockers.append("remote_build_packet_source_not_clean")
    if packet.get("provider_launch_performed_by_packet") is not False:
        blockers.append("remote_build_packet_provider_boundary_invalid")
    if builder.get("expected_source_commit") != packet_commit:
        blockers.append("builder_source_commit_mismatch")

    if spend.get("paid_mutation_authorized") is not True:
        blockers.append("builder_paid_mutation_not_authorized")
    max_spend = spend.get("max_spend_usd")
    if not _positive_number(max_spend):
        blockers.append("builder_max_spend_usd_missing")
    ttl = spend.get("hard_ttl_seconds")
    if type(ttl) is not int or ttl <= 0 or ttl > MAX_BUILD_TTL_SECONDS:
        blockers.append("builder_hard_ttl_must_be_at_most_two_hours")
    if spend.get("one_resource_limit") is not True:
        blockers.append("builder_one_resource_limit_missing")

    checks = {
        "runpod_excluded_from_build_plane": provider not in {"runpod", "runpod_pod", "runpod-pod"},
        "native_linux_amd64": _string(builder.get("platform")) == "linux/amd64",
        "docker_buildx_ready": builder.get("docker_daemon_verified") is True
        and builder.get("docker_buildx_verified") is True,
        "free_disk_at_least_120_gib": type(free_bytes) is int
        and free_bytes >= MIN_BUILD_FREE_BYTES,
        "source_bound": builder.get("expected_source_commit") == packet_commit,
        "paid_envelope_bound": spend.get("paid_mutation_authorized") is True
        and _positive_number(max_spend)
        and type(ttl) is int
        and 0 < ttl <= MAX_BUILD_TTL_SECONDS
        and spend.get("one_resource_limit") is True,
    }
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "provider": provider or None,
        "source_commit": packet_commit or None,
        "checks": checks,
        "limits": {
            "minimum_builder_free_bytes": MIN_BUILD_FREE_BYTES,
            "maximum_builder_ttl_seconds": MAX_BUILD_TTL_SECONDS,
            "max_spend_usd": max_spend,
        },
        "claim_boundary": {
            "admission_is_not_image_build": True,
            "admission_is_not_registry_publish_proof": True,
            "runpod_is_reserved_for_digest_pinned_serving": True,
        },
    }


def build_runpod_serve_plane_admission(
    *,
    release: Mapping[str, Any],
    model_cache: Mapping[str, Any],
    volume: Mapping[str, Any],
    runtime: Mapping[str, Any],
    spend: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit a RunPod warm worker only from published/cache-ready evidence."""

    blockers: list[str] = []
    release_ref = _string(release.get("resolved_digest_ref") or release.get("release_image_ref"))
    if not _DIGEST_REF.fullmatch(release_ref):
        blockers.append("runpod_release_image_not_digest_pinned")
    thin_contract = release.get("thin_release_contract")
    thin_status = (
        thin_contract.get("status")
        if isinstance(thin_contract, Mapping)
        else release.get("thin_release_contract_status")
    )
    if thin_status != "passed":
        blockers.append("runpod_thin_release_contract_not_passed")
    if release.get("runnable_platform") != "linux/amd64":
        blockers.append("runpod_release_platform_not_linux_amd64")

    manifest_digest = _string(model_cache.get("model_manifest_digest"))
    if (
        model_cache.get("schema_version")
        != "groot_oscar_external_model_cache_verification.v2"
    ):
        blockers.append("runpod_model_cache_verification_schema_invalid")
    if model_cache.get("status") != "passed":
        blockers.append("runpod_external_model_cache_not_verified")
    if not _SHA256.fullmatch(manifest_digest):
        blockers.append("runpod_model_manifest_digest_invalid")
    if model_cache.get("checks", {}).get("models_cached_offline") is not True:
        blockers.append("runpod_models_not_verified_offline")

    volume_id = _string(volume.get("id"))
    volume_dc = _string(volume.get("data_center_id"))
    volume_bytes = volume.get("size_bytes")
    if volume.get("provider") != "runpod" or volume.get("provider_api_verified") is not True:
        blockers.append("runpod_network_volume_not_provider_verified")
    if not volume_id:
        blockers.append("runpod_network_volume_id_missing")
    if not volume_dc:
        blockers.append("runpod_network_volume_data_center_missing")
    if type(volume_bytes) is not int or volume_bytes < MIN_MODEL_VOLUME_BYTES:
        blockers.append("runpod_network_volume_below_30_gib")
    cache_path = _string(volume.get("model_cache_path"))
    if not cache_path.startswith("/workspace/"):
        blockers.append("runpod_model_cache_path_must_be_under_workspace")

    if runtime.get("provider") != "runpod":
        blockers.append("runpod_serve_provider_invalid")
    if runtime.get("data_center_id") != volume_dc:
        blockers.append("runpod_gpu_and_network_volume_data_center_mismatch")
    if not _string(runtime.get("gpu_type_id")):
        blockers.append("runpod_gpu_type_not_selected")
    if runtime.get("provider_api_verified") is not True:
        blockers.append("runpod_gpu_capacity_not_provider_verified")
    if runtime.get("capacity_confidence") != "advisory":
        blockers.append("runpod_single_gpu_availability_unknown")
    if runtime.get("single_gpu_available") is not True:
        blockers.append("runpod_single_gpu_not_available")
    required_cuda = _string(runtime.get("required_cuda_version"))
    release_cuda = _string(release.get("required_cuda_version"))
    release_cuda_source = _string(release.get("required_cuda_version_source"))
    allowed_cuda = runtime.get("allowed_cuda_versions")
    allowed_cuda = allowed_cuda if isinstance(allowed_cuda, list) else []
    if not required_cuda or required_cuda not in allowed_cuda:
        blockers.append("runpod_cuda_compatibility_not_bound")
    if not release_cuda or release_cuda != required_cuda:
        blockers.append("runpod_cuda_version_differs_from_release")
    if not release_cuda_source.startswith("image_config_env:"):
        blockers.append("runpod_release_cuda_not_registry_config_verified")
    if runtime.get("warm_worker_only") is not True:
        blockers.append("runpod_customer_cold_start_disallowed")
    if runtime.get("provider_inventory_verified_zero") is not True:
        blockers.append("runpod_preallocation_inventory_not_zero")

    if spend.get("paid_mutation_authorized") is not True:
        blockers.append("runpod_paid_mutation_not_authorized")
    if not _positive_number(spend.get("max_spend_usd")):
        blockers.append("runpod_max_spend_usd_missing")
    ttl = spend.get("hard_ttl_seconds")
    if type(ttl) is not int or ttl <= 0 or ttl > MAX_CANARY_TTL_SECONDS:
        blockers.append("runpod_hard_ttl_must_be_at_most_30_minutes")
    if spend.get("one_resource_limit") is not True:
        blockers.append("runpod_one_resource_limit_missing")
    if spend.get("independent_teardown_watchdog") is not True:
        blockers.append("runpod_independent_teardown_watchdog_missing")
    if spend.get("watchdog_armed_before_allocation") is not True:
        blockers.append("runpod_teardown_watchdog_not_armed_before_allocation")
    hourly_rate = runtime.get("on_demand_price_usd_per_hour")
    max_spend = spend.get("max_spend_usd")
    if not _positive_number(hourly_rate):
        blockers.append("runpod_hourly_rate_missing")
    elif (
        _positive_number(max_spend)
        and type(ttl) is int
        and ttl > 0
        and float(hourly_rate) * ttl / 3600 > float(max_spend)
    ):
        blockers.append("runpod_ttl_cost_exceeds_max_spend")

    return {
        "schema_version": SERVE_SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "release_image_ref": release_ref or None,
        "model_manifest_digest": manifest_digest or None,
        "network_volume_id": volume_id or None,
        "data_center_id": volume_dc or None,
        "gpu_type_id": _string(runtime.get("gpu_type_id")) or None,
        "required_cuda_version": required_cuda or None,
        "limits": {
            "maximum_canary_ttl_seconds": MAX_CANARY_TTL_SECONDS,
            "hard_ttl_seconds": ttl,
            "max_spend_usd": max_spend,
            "watchdog_pod_name_prefix": spend.get("watchdog_pod_name_prefix"),
        },
        "claim_boundary": {
            "admission_is_not_provider_startup": True,
            "admission_is_not_warm_readiness": True,
            "admission_is_not_task_success": True,
        },
    }


def _load(path: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--packet", required=True)
    build.add_argument("--builder", required=True)
    build.add_argument("--spend", required=True)
    build.add_argument("--out", required=True)
    serve = subparsers.add_parser("runpod-serve")
    serve.add_argument("--release", required=True)
    serve.add_argument("--model-cache", required=True)
    serve.add_argument("--volume", required=True)
    serve.add_argument("--runtime", required=True)
    serve.add_argument("--spend", required=True)
    serve.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_build_plane_admission(
            packet=_load(args.packet),
            builder=_load(args.builder),
            spend=_load(args.spend),
        )
    else:
        result = build_runpod_serve_plane_admission(
            release=_load(args.release),
            model_cache=_load(args.model_cache),
            volume=_load(args.volume),
            runtime=_load(args.runtime),
            spend=_load(args.spend),
        )
    write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "admitted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
