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
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .groot_oscar_carrier_remote_build_packet import (
    PACKET_DIRNAME as CARRIER_PACKET_DIRNAME,
    render_remote_build_script,
)
from .openpi_policy_ranking_remote_build_packet import (
    PACKET_KIND as OPENPI_POLICY_RANKING_PACKET_KIND,
    SCHEMA_VERSION as OPENPI_POLICY_RANKING_PACKET_SCHEMA,
    validate_openpi_policy_ranking_archive,
)
from .isaac_worker_remote_build_packet import (
    PACKET_KIND as ISAAC_WORKER_PACKET_KIND,
    SCHEMA_VERSION as ISAAC_WORKER_PACKET_SCHEMA,
    validate_isaac_worker_archive,
)
from .reconstruction_worker_build_packet import (
    PACKET_KIND as RECONSTRUCTION_WORKER_PACKET_KIND,
    REMOTE_PACKET_SCHEMA_VERSION as RECONSTRUCTION_WORKER_PACKET_SCHEMA,
    validate_reconstruction_worker_archive,
)


# RunPod's create API returned this authoritative network-volume-capable set on
# 2026-07-14. The general datacenter catalog includes additional GPU locations
# that reject network-volume creation, so all volume/S3 paths share this one
# fail-closed provider-derived set until it is deliberately refreshed.
RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS = frozenset(
    {
        "AP-IN-2",
        "AP-JP-1",
        "CA-MTL-3",
        "CA-MTL-4",
        "EU-CZ-1",
        "EU-FR-1",
        "EU-NL-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-IS-3",
        "EUR-NO-1",
        "EUR-NO-2",
        "US-CA-2",
        "US-IL-1",
        "US-MO-2",
        "US-NC-1",
        "US-NE-1",
        "US-TX-3",
        "US-WA-1",
    }
)
RUNPOD_S3_DATA_CENTER_IDS = frozenset(
    {
        "EU-CZ-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-NO-1",
        "US-CA-2",
        "US-GA-2",
        "US-IL-1",
        "US-KS-2",
        "US-MD-1",
        "US-MO-1",
        "US-MO-2",
        "US-NC-1",
        "US-NC-2",
        "US-NE-1",
        "US-WA-1",
    }
)
RUNPOD_S3_VOLUME_DATA_CENTER_IDS = RUNPOD_S3_DATA_CENTER_IDS & RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS

BUILD_SCHEMA_VERSION = "groot_oscar_build_plane_admission.v1"
SERVE_SCHEMA_VERSION = "groot_oscar_runpod_serve_plane_admission.v2"
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
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_IMAGE_TAG = re.compile(r"\A[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_SSH_HOST_KEY_SHA256 = re.compile(r"\ASHA256:[A-Za-z0-9+/]{43}\Z")


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _versioned_image_ref(value: Any) -> bool:
    ref = _string(value)
    leaf = ref.rsplit("/", 1)[-1]
    name, separator, tag = leaf.rpartition(":")
    return bool(
        name
        and separator
        and _IMAGE_TAG.fullmatch(tag)
        and tag not in {"latest", "dev", "test", "local"}
        and "@" not in ref
        and not any(char.isspace() for char in ref)
    )


def _safe_archive_member(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def validate_carrier_image_archive(packet: Mapping[str, Any]) -> list[str]:
    """Bind the exact carrier archive and executable before paid allocation."""

    blockers: list[str] = []
    expected_names = sorted(
        (
            f"{CARRIER_PACKET_DIRNAME}/README.md",
            f"{CARRIER_PACKET_DIRNAME}/context/Dockerfile",
            f"{CARRIER_PACKET_DIRNAME}/remote_build_groot_oscar_carrier.sh",
        )
    )
    declared_names = packet.get("archive_members")
    declared_names = declared_names if isinstance(declared_names, list) else []
    declared_digests = packet.get("archive_member_sha256")
    declared_digests = declared_digests if isinstance(declared_digests, Mapping) else {}
    if declared_names != expected_names:
        blockers.append("builder_carrier_archive_member_contract_invalid")
    if sorted(declared_digests) != expected_names or any(
        not _HEX64.fullmatch(_string(value)) for value in declared_digests.values()
    ):
        blockers.append("builder_carrier_archive_digest_contract_invalid")
    manifest_digest = hashlib.sha256(
        json.dumps(dict(declared_digests), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if packet.get("archive_member_manifest_sha256") != manifest_digest:
        blockers.append("builder_carrier_archive_member_manifest_mismatch")

    path = Path(_string(packet.get("tarball_path"))).expanduser().resolve()
    declared_tarball = _string(packet.get("tarball_sha256"))
    if not _HEX64.fullmatch(declared_tarball):
        blockers.append("builder_carrier_tarball_sha256_invalid")
    payloads: dict[str, bytes] = {}
    if not path.is_file():
        blockers.append("builder_carrier_archive_missing")
    else:
        observed_tarball = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_tarball != declared_tarball:
            blockers.append("builder_carrier_archive_tarball_mismatch")
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getmembers()
                names = [member.name for member in members]
                if len(names) != len(set(names)):
                    blockers.append("builder_carrier_archive_duplicate_member")
                if any(not _safe_archive_member(name) for name in names):
                    blockers.append("builder_carrier_archive_unsafe_path")
                if any(not member.isfile() for member in members):
                    blockers.append("builder_carrier_archive_nonregular_member")
                script_member = next(
                    (
                        member
                        for member in members
                        if member.name
                        == f"{CARRIER_PACKET_DIRNAME}/remote_build_groot_oscar_carrier.sh"
                    ),
                    None,
                )
                if script_member is None or script_member.mode & 0o111 == 0:
                    blockers.append("builder_carrier_archive_script_not_executable")
                if sorted(names) != expected_names or names != declared_names:
                    blockers.append("builder_carrier_archive_inventory_mismatch")
                if not blockers:
                    for member in members:
                        stream = archive.extractfile(member)
                        if stream is None:
                            blockers.append("builder_carrier_archive_member_unreadable")
                            continue
                        payloads[member.name] = stream.read()
        except (OSError, tarfile.TarError):
            blockers.append("builder_carrier_archive_unreadable")

    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != declared_digests.get(name):
            blockers.append("builder_carrier_archive_member_digest_mismatch")
    dockerfile_name = f"{CARRIER_PACKET_DIRNAME}/context/Dockerfile"
    script_name = f"{CARRIER_PACKET_DIRNAME}/remote_build_groot_oscar_carrier.sh"
    dockerfile = payloads.get(dockerfile_name)
    if dockerfile is not None and hashlib.sha256(dockerfile).hexdigest() != packet.get(
        "carrier_dockerfile_sha256"
    ):
        blockers.append("builder_carrier_archive_dockerfile_binding_mismatch")
    expected_script = render_remote_build_script(
        image_ref=_string(packet.get("carrier_image_ref")),
        base_image_ref=_string(packet.get("carrier_base_image_ref")),
        source_commit=_string(packet.get("source_commit")),
        dockerfile_sha256=_string(packet.get("carrier_dockerfile_sha256")),
    ).encode()
    if payloads.get(script_name) != expected_script:
        blockers.append("builder_carrier_archive_script_binding_mismatch")
    return sorted(set(blockers))


def build_live_machine_capability_evidence(
    observation: Mapping[str, Any],
    *,
    minimum_free_bytes: int = MIN_BUILD_FREE_BYTES,
    required_architecture: str = "x86_64",
    packet_kind: str = "thin_release",
    expected_s3_endpoint_host: str | None = None,
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
    if packet_kind not in {
        "thin_release",
        "carrier_image",
        OPENPI_POLICY_RANKING_PACKET_KIND,
        ISAAC_WORKER_PACKET_KIND,
        RECONSTRUCTION_WORKER_PACKET_KIND,
        "model_cache_s3",
    }:
        blockers.append("live_machine_packet_kind_unsupported")
    if packet_kind in {
        "thin_release",
        "carrier_image",
        OPENPI_POLICY_RANKING_PACKET_KIND,
        ISAAC_WORKER_PACKET_KIND,
        RECONSTRUCTION_WORKER_PACKET_KIND,
    }:
        if observation.get("docker_cli_present") is not True:
            blockers.append("live_machine_docker_cli_missing")
        if observation.get("docker_daemon_responding") is not True:
            blockers.append("live_machine_docker_daemon_unavailable")
        if observation.get("docker_buildx_available") is not True:
            blockers.append("live_machine_docker_buildx_unavailable")
    elif packet_kind == "model_cache_s3":
        if observation.get("python3_available") is not True:
            blockers.append("live_machine_python3_missing")
        if observation.get("python_version") != "3.12":
            blockers.append("live_machine_python_version_mismatch")
        if observation.get("python_venv_available") is not True:
            blockers.append("live_machine_python_venv_missing")
        if observation.get("dns_resolution_verified") is not True:
            blockers.append("live_machine_dns_resolution_unverified")
        if observation.get("outbound_https_verified") is not True:
            blockers.append("live_machine_outbound_https_unverified")
        if (
            not expected_s3_endpoint_host
            or observation.get("s3_endpoint_host") != expected_s3_endpoint_host
        ):
            blockers.append("live_machine_s3_endpoint_binding_mismatch")
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
        "python3_available": observation.get("python3_available"),
        "python_version": observation.get("python_version"),
        "python_venv_available": observation.get("python_venv_available"),
        "dns_resolution_verified": observation.get("dns_resolution_verified"),
        "outbound_https_verified": observation.get("outbound_https_verified"),
        "s3_endpoint_host": observation.get("s3_endpoint_host"),
        "builder_ready_marker": observation.get("builder_ready_marker"),
        "packet_kind": packet_kind,
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
    runtime_bundle_requested: bool = False,
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
    allocation_packet_kind = (
        allocation_admission.get("checks", {}).get("packet_kind")
        if isinstance(allocation_admission.get("checks"), Mapping)
        else None
    )
    if allocation_packet_kind is None:
        allocation_packet_kind = "thin_release"
    if live_machine.get("packet_kind") != allocation_packet_kind:
        blockers.append("cpu_builder_live_capability_packet_kind_mismatch")
    if runtime_bundle_requested:
        if allocation_packet_kind != "model_cache_s3":
            blockers.append("cpu_builder_runtime_bundle_packet_kind_invalid")
        if live_machine.get("docker_cli_present") is not True:
            blockers.append("cpu_builder_runtime_bundle_docker_cli_missing")
        if live_machine.get("docker_daemon_responding") is not True:
            blockers.append("cpu_builder_runtime_bundle_docker_daemon_unavailable")
    blockers.extend(f"live_capability:{item}" for item in live_machine.get("blockers", []))
    return {
        "schema_version": CPU_BUILD_EXECUTION_SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "allocation_admission_schema_version": allocation_admission.get("schema_version"),
        "live_machine_capability": dict(live_machine),
        "runtime_bundle_requested": runtime_bundle_requested,
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
    *,
    provider_payload: Mapping[str, Any],
    expected_volume_id: str,
    model_cache_path: str,
    expected_name: str | None = None,
    allocation_nonce: str | None = None,
) -> dict[str, Any]:
    """Normalize the authoritative RunPod ``GET /networkvolumes/{id}`` row."""

    observed_id = _string(provider_payload.get("id"))
    observed_name = _string(provider_payload.get("name"))
    data_center_id = _string(provider_payload.get("dataCenterId"))
    size_gib = provider_payload.get("size")
    blockers: list[str] = []
    if not expected_volume_id or observed_id != expected_volume_id:
        blockers.append("runpod_network_volume_provider_id_mismatch")
    if not data_center_id:
        blockers.append("runpod_network_volume_data_center_missing")
    if type(size_gib) is not int or size_gib <= 0:
        blockers.append("runpod_network_volume_size_invalid")
    if expected_name is not None and observed_name != expected_name:
        blockers.append("runpod_network_volume_provider_name_mismatch")
    if allocation_nonce is not None and (
        not allocation_nonce or allocation_nonce not in observed_name
    ):
        blockers.append("runpod_network_volume_allocation_nonce_mismatch")
    size_bytes = size_gib * 1024**3 if type(size_gib) is int and size_gib > 0 else None
    return {
        "schema_version": "groot_oscar_runpod_network_volume_evidence.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": blockers,
        "provider": "runpod",
        "provider_api_verified": not blockers,
        "id": observed_id or None,
        "name": observed_name or None,
        "data_center_id": data_center_id or None,
        "size_bytes": size_bytes,
        "allocation_nonce": allocation_nonce,
        "allocation_name_verified": bool(
            expected_name is not None
            and observed_name == expected_name
            and allocation_nonce
            and allocation_nonce in observed_name
        ),
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
    single_available = bool(
        confidence == "advisory"
        and selected.get("single_gpu_offer_requested") is True
        and selected.get("single_gpu_offer_available") is True
        and (not counts or 1 in counts)
    )
    capacity_data_center_id = _string(selected.get("capacity_data_center_id"))
    capacity_cuda_versions = selected.get("capacity_allowed_cuda_versions")
    capacity_cuda_versions = (
        [str(item) for item in capacity_cuda_versions]
        if isinstance(capacity_cuda_versions, list)
        else []
    )
    provider_verified = (
        capacity.get("status") == "available"
        and bool(selected)
        and bool(data_center_id)
        and capacity_data_center_id == data_center_id
        and bool(required_cuda_version)
        and required_cuda_version in capacity_cuda_versions
    )
    return {
        "schema_version": "groot_oscar_runpod_gpu_runtime_evidence.v1",
        "provider": "runpod",
        "provider_api_verified": provider_verified,
        "data_center_id": data_center_id,
        "capacity_data_center_id": capacity_data_center_id or None,
        "capacity_allowed_cuda_versions": capacity_cuda_versions,
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
    packet_kind = _string(packet.get("packet_kind")) or "thin_release"
    if packet_kind not in {
        "thin_release",
        "carrier_image",
        OPENPI_POLICY_RANKING_PACKET_KIND,
        ISAAC_WORKER_PACKET_KIND,
        RECONSTRUCTION_WORKER_PACKET_KIND,
        "model_cache_s3",
    }:
        blockers.append("builder_packet_kind_unsupported")
    expected_purpose = {
        "thin_release": "image_build",
        "carrier_image": "image_build",
        OPENPI_POLICY_RANKING_PACKET_KIND: "image_build",
        ISAAC_WORKER_PACKET_KIND: "image_build",
        RECONSTRUCTION_WORKER_PACKET_KIND: "image_build",
        "model_cache_s3": "model_cache_s3",
    }.get(packet_kind)
    if packet_kind == "carrier_image":
        if packet.get("schema_version") != "groot_oscar_carrier_remote_build_packet.v1":
            blockers.append("builder_carrier_packet_schema_invalid")
        if not _versioned_image_ref(packet.get("carrier_image_ref")):
            blockers.append("builder_carrier_image_ref_not_versioned")
        if not _DIGEST_REF.fullmatch(_string(packet.get("carrier_base_image_ref"))):
            blockers.append("builder_carrier_base_image_not_digest_pinned")
        if not _HEX64.fullmatch(_string(packet.get("carrier_dockerfile_sha256"))):
            blockers.append("builder_carrier_dockerfile_sha256_invalid")
        if not _COMMIT.fullmatch(_string(packet.get("source_commit"))):
            blockers.append("builder_carrier_source_commit_invalid")
        blockers.extend(validate_carrier_image_archive(packet))
    if packet_kind == OPENPI_POLICY_RANKING_PACKET_KIND:
        if packet.get("schema_version") != OPENPI_POLICY_RANKING_PACKET_SCHEMA:
            blockers.append("builder_openpi_packet_schema_invalid")
        if not _versioned_image_ref(packet.get("image_ref")):
            blockers.append("builder_openpi_image_ref_not_versioned")
        if not _HEX64.fullmatch(_string(packet.get("dockerfile_sha256"))):
            blockers.append("builder_openpi_dockerfile_sha256_invalid")
        if not _HEX64.fullmatch(_string(packet.get("context_manifest_sha256"))):
            blockers.append("builder_openpi_context_manifest_sha256_invalid")
        if not _COMMIT.fullmatch(_string(packet.get("source_commit"))):
            blockers.append("builder_openpi_source_commit_invalid")
        blockers.extend(validate_openpi_policy_ranking_archive(packet))
    if packet_kind == ISAAC_WORKER_PACKET_KIND:
        if packet.get("schema_version") != ISAAC_WORKER_PACKET_SCHEMA:
            blockers.append("builder_isaac_packet_schema_invalid")
        if not _versioned_image_ref(packet.get("image_ref")):
            blockers.append("builder_isaac_image_ref_not_versioned")
        if not _DIGEST_REF.fullmatch(_string(packet.get("base_image_ref"))):
            blockers.append("builder_isaac_base_image_not_digest_pinned")
        if not _HEX64.fullmatch(_string(packet.get("dockerfile_sha256"))):
            blockers.append("builder_isaac_dockerfile_sha256_invalid")
        if not _HEX64.fullmatch(_string(packet.get("context_manifest_sha256"))):
            blockers.append("builder_isaac_context_manifest_sha256_invalid")
        if not _COMMIT.fullmatch(_string(packet.get("source_commit"))):
            blockers.append("builder_isaac_source_commit_invalid")
        blockers.extend(validate_isaac_worker_archive(packet))
    if packet_kind == RECONSTRUCTION_WORKER_PACKET_KIND:
        if packet.get("schema_version") != RECONSTRUCTION_WORKER_PACKET_SCHEMA:
            blockers.append("builder_reconstruction_packet_schema_invalid")
        if not _versioned_image_ref(packet.get("image_ref")):
            blockers.append("builder_reconstruction_image_ref_not_versioned")
        for field, blocker in (
            ("dockerfile_sha256", "builder_reconstruction_dockerfile_sha256_invalid"),
            (
                "requirements_lock_sha256",
                "builder_reconstruction_requirements_lock_sha256_invalid",
            ),
            (
                "context_manifest_sha256",
                "builder_reconstruction_context_manifest_sha256_invalid",
            ),
        ):
            if not _HEX64.fullmatch(_string(packet.get(field))):
                blockers.append(blocker)
        for field, blocker in (
            (
                "worker_stack_manifest_digest",
                "builder_reconstruction_stack_manifest_digest_invalid",
            ),
            (
                "license_review_receipt_digest",
                "builder_reconstruction_license_receipt_digest_invalid",
            ),
            (
                "paid_execution_envelope_digest",
                "builder_reconstruction_paid_envelope_digest_invalid",
            ),
        ):
            if not _SHA256.fullmatch(_string(packet.get(field))):
                blockers.append(blocker)
        if not _COMMIT.fullmatch(_string(packet.get("source_commit"))):
            blockers.append("builder_reconstruction_source_commit_invalid")
        paid_envelope = packet.get("paid_execution_envelope")
        paid_envelope = paid_envelope if isinstance(paid_envelope, Mapping) else {}
        if (
            paid_envelope.get("schema_version")
            != "reconstruction_worker_paid_execution_envelope.v1"
            or paid_envelope.get("authorized_action") != "cpu-build"
            or paid_envelope.get("paid_mutation_authorized") is not True
            or paid_envelope.get("authority_issued_by_agent") is not False
            or not _string(paid_envelope.get("authority_id"))
            or paid_envelope.get("source_commit_sha") != packet.get("source_commit")
            or paid_envelope.get("worker_stack_manifest_digest")
            != packet.get("worker_stack_manifest_digest")
            or paid_envelope.get("license_inventory_digest")
            != packet.get("license_inventory_digest")
            or paid_envelope.get("license_review_receipt_digest")
            != packet.get("license_review_receipt_digest")
            or paid_envelope.get("paid_execution_envelope_digest")
            != packet.get("paid_execution_envelope_digest")
            or paid_envelope.get("paid_execution_envelope_digest")
            != canonical_digest(
                paid_envelope, digest_field="paid_execution_envelope_digest"
            )
        ):
            blockers.append("builder_reconstruction_paid_envelope_invalid")
        blockers.extend(validate_reconstruction_worker_archive(packet))
    if provider in {"runpod", "runpod_pod", "runpod-pod"}:
        blockers.append("runpod_pods_are_serve_plane_not_image_build_plane")
    if expected_purpose is not None and _string(builder.get("purpose")) != expected_purpose:
        blockers.append("builder_purpose_does_not_match_packet_kind")
    if _string(builder.get("platform")) != "linux/amd64":
        blockers.append("builder_native_linux_amd64_not_verified")
    if packet_kind in {
        "thin_release",
        "carrier_image",
        OPENPI_POLICY_RANKING_PACKET_KIND,
        ISAAC_WORKER_PACKET_KIND,
        RECONSTRUCTION_WORKER_PACKET_KIND,
    }:
        if builder.get("docker_daemon_verified") is not True:
            blockers.append("builder_docker_daemon_not_verified")
        if builder.get("docker_buildx_verified") is not True:
            blockers.append("builder_docker_buildx_not_verified")
    elif packet_kind == "model_cache_s3":
        packet_data_center = _string(packet.get("data_center_id"))
        if packet_data_center not in RUNPOD_S3_VOLUME_DATA_CENTER_IDS:
            blockers.append("builder_model_cache_s3_data_center_unsupported")
        if builder.get("python_runtime_verified") is not True:
            blockers.append("builder_python_runtime_not_verified")
        if builder.get("python_version") != "3.12":
            blockers.append("builder_python_version_mismatch")
        if builder.get("dependency_lock_verified") is not True:
            blockers.append("builder_dependency_lock_not_verified")
        if builder.get("dependency_wheelhouse_verified") is not True:
            blockers.append("builder_dependency_wheelhouse_not_verified")
        if builder.get("dns_resolution_verified") is not True:
            blockers.append("builder_dns_resolution_not_verified")
        if builder.get("outbound_https_verified") is not True:
            blockers.append("builder_outbound_https_not_verified")
        expected_s3_host = "s3api-" + packet_data_center.lower() + ".runpod.io"
        if builder.get("s3_endpoint_host") != expected_s3_host:
            blockers.append("builder_s3_endpoint_binding_mismatch")
    free_bytes = builder.get("free_disk_bytes")
    if type(free_bytes) is not int or free_bytes < MIN_BUILD_FREE_BYTES:
        blockers.append("builder_free_disk_below_120_gib")
    if (
        packet_kind
        in {
            "thin_release",
            "carrier_image",
            OPENPI_POLICY_RANKING_PACKET_KIND,
            ISAAC_WORKER_PACKET_KIND,
            RECONSTRUCTION_WORKER_PACKET_KIND,
        }
        and builder.get("registry_push_auth_file_verified") is not True
    ):
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
    if packet_kind == RECONSTRUCTION_WORKER_PACKET_KIND:
        paid_envelope = packet.get("paid_execution_envelope")
        paid_envelope = paid_envelope if isinstance(paid_envelope, Mapping) else {}
        if (
            spend.get("paid_mutation_authorized")
            != paid_envelope.get("paid_mutation_authorized")
            or spend.get("max_spend_usd") != paid_envelope.get("max_spend_usd")
            or spend.get("hard_ttl_seconds")
            != paid_envelope.get("hard_ttl_seconds")
            or spend.get("retry_cap") != paid_envelope.get("retry_cap")
            or spend.get("authority_id") != paid_envelope.get("authority_id")
            or spend.get("authority_issued_by_agent")
            != paid_envelope.get("authority_issued_by_agent")
        ):
            blockers.append("builder_reconstruction_spend_envelope_mismatch")

    checks = {
        "runpod_excluded_from_build_plane": provider not in {"runpod", "runpod_pod", "runpod-pod"},
        "native_linux_amd64": _string(builder.get("platform")) == "linux/amd64",
        "execution_runtime_ready": (
            builder.get("docker_daemon_verified") is True
            and builder.get("docker_buildx_verified") is True
            if packet_kind
            in {
                "thin_release",
                "carrier_image",
                OPENPI_POLICY_RANKING_PACKET_KIND,
                ISAAC_WORKER_PACKET_KIND,
                RECONSTRUCTION_WORKER_PACKET_KIND,
            }
            else builder.get("python_runtime_verified") is True
            and builder.get("python_version") == "3.12"
            and builder.get("dependency_lock_verified") is True
            and builder.get("dependency_wheelhouse_verified") is True
            and builder.get("dns_resolution_verified") is True
            and builder.get("outbound_https_verified") is True
            and builder.get("s3_endpoint_host")
            == "s3api-" + _string(packet.get("data_center_id")).lower() + ".runpod.io"
        ),
        "free_disk_at_least_120_gib": type(free_bytes) is int
        and free_bytes >= MIN_BUILD_FREE_BYTES,
        "source_bound": builder.get("expected_source_commit") == packet_commit,
        "packet_kind": packet_kind,
        "builder_purpose_matches_packet_kind": expected_purpose is not None
        and _string(builder.get("purpose")) == expected_purpose,
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
    expected_source_commit: str,
    maximum_ttl_seconds: int = MAX_CANARY_TTL_SECONDS,
) -> dict[str, Any]:
    """Admit a RunPod warm worker only from published/cache-ready evidence."""

    blockers: list[str] = []
    release_source_commit = _string(release.get("source_commit"))
    expected_source_commit = _string(expected_source_commit)
    if not _COMMIT.fullmatch(expected_source_commit):
        blockers.append("runpod_expected_source_commit_invalid")
    if not _COMMIT.fullmatch(release_source_commit):
        blockers.append("runpod_release_source_commit_invalid")
    elif release_source_commit != expected_source_commit:
        blockers.append("runpod_release_source_commit_mismatch")
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
    if model_cache.get("schema_version") != "groot_oscar_external_model_cache_verification.v2":
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
    verified_cache_bytes = model_cache.get("verified_size_bytes")
    if volume.get("provider") != "runpod" or volume.get("provider_api_verified") is not True:
        blockers.append("runpod_network_volume_not_provider_verified")
    if not volume_id:
        blockers.append("runpod_network_volume_id_missing")
    if not volume_dc:
        blockers.append("runpod_network_volume_data_center_missing")
    if type(volume_bytes) is not int or volume_bytes < MIN_MODEL_VOLUME_BYTES:
        blockers.append("runpod_network_volume_below_30_gib")
    if type(verified_cache_bytes) is not int or verified_cache_bytes <= 0:
        blockers.append("runpod_model_cache_verified_size_missing")
    elif type(volume_bytes) is int and verified_cache_bytes > volume_bytes:
        blockers.append("runpod_network_volume_smaller_than_verified_model_cache")
    cache_path = _string(volume.get("model_cache_path"))
    if not cache_path.startswith("/workspace/"):
        blockers.append("runpod_model_cache_path_must_be_under_workspace")
    if _string(model_cache.get("cache_root")) != cache_path:
        blockers.append("runpod_model_cache_verification_path_mismatch")
    if _string(model_cache.get("provider_volume_id")) != volume_id:
        blockers.append("runpod_model_cache_verification_volume_mismatch")

    if runtime.get("provider") != "runpod":
        blockers.append("runpod_serve_provider_invalid")
    if runtime.get("data_center_id") != volume_dc:
        blockers.append("runpod_gpu_and_network_volume_data_center_mismatch")
    if runtime.get("capacity_data_center_id") != volume_dc:
        blockers.append("runpod_gpu_capacity_not_verified_in_volume_data_center")
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
    capacity_cuda_versions = runtime.get("capacity_allowed_cuda_versions")
    capacity_cuda_versions = (
        capacity_cuda_versions if isinstance(capacity_cuda_versions, list) else []
    )
    if not required_cuda or required_cuda not in capacity_cuda_versions:
        blockers.append("runpod_gpu_capacity_not_verified_for_cuda_version")
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
    if (
        type(maximum_ttl_seconds) is not int
        or maximum_ttl_seconds <= 0
        or type(ttl) is not int
        or ttl <= 0
        or ttl > maximum_ttl_seconds
    ):
        blockers.append(
            "runpod_hard_ttl_must_be_at_most_30_minutes"
            if maximum_ttl_seconds == MAX_CANARY_TTL_SECONDS
            else "runpod_hard_ttl_exceeds_admitted_lane_maximum"
        )
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
        "source_commit": release_source_commit or None,
        "expected_source_commit": expected_source_commit or None,
        "source_bound": bool(
            _COMMIT.fullmatch(release_source_commit)
            and release_source_commit == expected_source_commit
        ),
        "release_image_ref": release_ref or None,
        "model_manifest_digest": manifest_digest or None,
        "model_cache_path": cache_path or None,
        "verified_model_cache_bytes": verified_cache_bytes,
        "network_volume_id": volume_id or None,
        "data_center_id": volume_dc or None,
        "gpu_type_id": _string(runtime.get("gpu_type_id")) or None,
        "required_cuda_version": required_cuda or None,
        "limits": {
            "maximum_canary_ttl_seconds": maximum_ttl_seconds,
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
    serve.add_argument("--expected-source-commit", required=True)
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
            expected_source_commit=args.expected_source_commit,
        )
    write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "admitted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
