"""Fail-closed sandbox plan for untrusted company policy containers.

The plan uses an isolated container runtime plus a Blueprint-owned proxy
sidecar.  The policy and proxy share an otherwise empty network namespace;
Blueprint reaches the proxy over a host-owned Unix socket.  The policy gets no
scene, capture, evidence, output, credential, Docker-socket, or host-network
mount.  A measured no-egress receipt and synthetic protocol conformance are
required before the first real observation may cross the proxy.

This module builds and validates commands and receipts.  It never executes a
container, redeems a credential, opens a network connection, grants launch
authority, or allocates a provider.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from .company_policy_container_contract_v2 import (
    CompanyPolicyContainerContractV2Error,
    validate_company_policy_container_contract_v2,
)
from .decision_evidence_contracts import cross_runtime_canonical_digest


SCHEMA_VERSION = "company_policy_sandbox_plan.v2"
QUALIFICATION_SCHEMA_VERSION = "company_policy_sandbox_qualification_receipt.v2"
SUPPORTED_RUNTIME_CLASSES = frozenset(
    {"runsc", "kata-runtime", "firecracker-containerd"}
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,191}$")
_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$")

DENIAL_PROBES = (
    "dns_resolution",
    "public_ipv4",
    "public_ipv6",
    "rfc1918",
    "host_gateway",
    "link_local",
    "cloud_metadata",
    "registry_after_pull",
    "redirect_target",
)
REACHABILITY_PROBES = (
    "blueprint_proxy_unix_socket",
    "policy_action_route_through_proxy",
)


class CompanyPolicySandboxV2Error(ValueError):
    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(sorted({str(item) for item in blockers if str(item)}))
        super().__init__(";".join(self.blockers))


def company_policy_registry_host(image: str) -> str:
    repository = image.split("@sha256:", 1)[0]
    first = repository.split("/", 1)[0].lower()
    if "." not in first and ":" not in first:
        return "docker.io"
    host = re.sub(r":\d+$", "", first)
    if (
        "." not in host
        or host == "localhost"
        or host.endswith((".localhost", ".local", ".internal"))
    ):
        raise CompanyPolicySandboxV2Error(
            ["company_policy_sandbox_registry_origin_invalid"]
        )
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return first
    raise CompanyPolicySandboxV2Error(
        ["company_policy_sandbox_registry_ip_literal_forbidden"]
    )


def _public_registry_addresses(values: Sequence[str]) -> list[str]:
    blockers: list[str] = []
    normalized: list[str] = []
    for value in values:
        try:
            address = ipaddress.ip_address(str(value))
        except ValueError:
            blockers.append("company_policy_sandbox_registry_address_invalid")
            continue
        if not address.is_global:
            blockers.append("company_policy_sandbox_registry_address_not_global")
        normalized.append(str(address))
    if not normalized:
        blockers.append("company_policy_sandbox_registry_addresses_missing")
    if blockers:
        raise CompanyPolicySandboxV2Error(blockers)
    return sorted(set(normalized))


def _exact_text(value: Any, *, blocker: str, pattern: re.Pattern[str]) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not pattern.fullmatch(text):
        raise CompanyPolicySandboxV2Error([blocker])
    return text


def build_company_policy_sandbox_plan(
    *,
    admission_receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    sandbox_attempt_id: str,
    pipeline_release_sha: str,
    worker_identity: str,
    runtime_class: str,
    blueprint_proxy_image: str,
    blueprint_proxy_contract_digest: str,
    seccomp_profile_id: str,
    seccomp_profile_digest: str,
    apparmor_profile_id: str,
    apparmor_profile_digest: str,
    registry_addresses: Sequence[str],
    allowed_registry_hosts: Sequence[str],
) -> dict[str, Any]:
    """Build an immutable command plan; do not execute any command."""

    try:
        normalized_contract = validate_company_policy_container_contract_v2(contract)
    except CompanyPolicyContainerContractV2Error as exc:
        raise CompanyPolicySandboxV2Error(exc.errors) from exc
    blockers: list[str] = []
    if admission_receipt.get("accepted") is not True or admission_receipt.get("status") != "admitted_no_spend":
        blockers.append("company_policy_sandbox_admission_not_accepted")
    for field in (
        "launch_queued",
        "launch_authority_granted",
        "provider_mutation_authorized",
        "provider_mutation_performed",
        "registry_credential_consumed",
    ):
        if admission_receipt.get(field) is not False:
            blockers.append(f"company_policy_sandbox_admission_{field}_not_false")
    contract_digest = str(normalized_contract["contract_digest"])
    if admission_receipt.get("contract_digest") != contract_digest:
        blockers.append("company_policy_sandbox_contract_digest_mismatch")
    admission_digest = str(admission_receipt.get("admission_digest") or "")
    if not _DIGEST.fullmatch(admission_digest):
        blockers.append("company_policy_sandbox_admission_digest_invalid")
    admission_id = str(admission_receipt.get("admission_id") or "")
    if not _IDENTIFIER.fullmatch(admission_id):
        blockers.append("company_policy_sandbox_admission_id_invalid")
    if runtime_class not in SUPPORTED_RUNTIME_CLASSES:
        blockers.append("company_policy_sandbox_shared_kernel_runtime_forbidden")
    if not _SHA.fullmatch(pipeline_release_sha):
        blockers.append("company_policy_sandbox_release_sha_invalid")
    for label, value in (
        ("sandbox_attempt_id", sandbox_attempt_id),
        ("worker_identity", worker_identity),
        ("seccomp_profile_id", seccomp_profile_id),
        ("apparmor_profile_id", apparmor_profile_id),
    ):
        if not _IDENTIFIER.fullmatch(value):
            blockers.append(f"company_policy_sandbox_{label}_invalid")
    for label, value in (
        ("proxy_contract_digest", blueprint_proxy_contract_digest),
        ("seccomp_profile_digest", seccomp_profile_digest),
        ("apparmor_profile_digest", apparmor_profile_digest),
    ):
        if not _DIGEST.fullmatch(value):
            blockers.append(f"company_policy_sandbox_{label}_invalid")
    if not _IMAGE.fullmatch(blueprint_proxy_image):
        blockers.append("company_policy_sandbox_proxy_image_not_digest_pinned")
    if blockers:
        raise CompanyPolicySandboxV2Error(blockers)

    image = str(normalized_contract["container"]["image"])
    registry_host = company_policy_registry_host(image)
    allowed = {str(host).strip().lower() for host in allowed_registry_hosts if str(host).strip()}
    if not allowed or registry_host not in allowed:
        raise CompanyPolicySandboxV2Error(
            ["company_policy_sandbox_registry_origin_not_allowed"]
        )
    addresses = _public_registry_addresses(registry_addresses)
    suffix = sha256(f"{admission_digest}\0{sandbox_attempt_id}".encode()).hexdigest()[:20]
    proxy_name = f"blueprint-policy-proxy-{suffix}"
    policy_name = f"company-policy-{suffix}"
    ipc_host_dir = f"/run/blueprint/company-policy/{suffix}"
    docker_config_dir = f"/run/blueprint/company-policy-registry/{suffix}"
    resources = normalized_contract["container"]["resources"]
    policy_run_argv = [
        "docker",
        "run",
        "--detach",
        "--name",
        policy_name,
        "--runtime",
        runtime_class,
        "--network",
        f"container:{proxy_name}",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        f"--security-opt=seccomp={seccomp_profile_id}",
        f"--security-opt=apparmor={apparmor_profile_id}",
        "--pids-limit",
        str(resources["pids_limit"]),
        "--cpus",
        str(resources["cpus"]),
        "--memory",
        f'{resources["memory_mib"]}m',
        "--tmpfs",
        f'/tmp:rw,noexec,nosuid,nodev,size={resources["tmpfs_mib"]}m',
        "--user",
        f'{normalized_contract["container"]["run_as_uid"]}:{normalized_contract["container"]["run_as_gid"]}',
        "--log-driver=none",
    ]
    if normalized_contract["container"]["gpu_required"]:
        policy_run_argv.extend(["--gpus", "all"])
    policy_run_argv.extend(
        [image, *[str(item) for item in normalized_contract["container"]["serve_command"]]]
    )
    proxy_run_argv = [
        "docker",
        "run",
        "--detach",
        "--name",
        proxy_name,
        "--runtime",
        runtime_class,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        f"--security-opt=seccomp={seccomp_profile_id}",
        f"--security-opt=apparmor={apparmor_profile_id}",
        "--pids-limit=64",
        "--memory=256m",
        "--cpus=1",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--mount",
        f"type=bind,src={ipc_host_dir},dst=/run/blueprint-ipc",
        "--log-driver=none",
        blueprint_proxy_image,
        "serve",
        "--unix-socket",
        "/run/blueprint-ipc/policy.sock",
        "--upstream",
        f'http://127.0.0.1:{normalized_contract["container"]["port"]}',
        "--route",
        "/v1/actions",
    ]
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned_no_spend",
        "admission_id": admission_id,
        "admission_digest": admission_digest,
        "contract_digest": contract_digest,
        "sandbox_attempt_id": sandbox_attempt_id,
        "pipeline_release_sha": pipeline_release_sha,
        "worker_identity": worker_identity,
        "runtime_class": runtime_class,
        "worker_isolation_required": "dedicated_ephemeral_worker_plus_isolated_runtime",
        "registry": {
            "host": registry_host,
            "resolved_global_addresses": addresses,
            "redirects_allowed": False,
            "pull_by_digest_only": True,
        },
        "credential_broker_request_binding": {
            "schema_version": "company_policy_registry_credential_consume.v2",
            "admission_id": admission_id,
            "admission_digest": admission_digest,
            "sandbox_attempt_id": sandbox_attempt_id,
            "pipeline_release_sha": pipeline_release_sha,
            "worker_identity": worker_identity,
            "purpose": "pull_digest_pinned_company_policy_image",
            "image": image,
        },
        "blueprint_ipc": {
            "kind": "host_unix_socket_to_proxy_then_container_loopback",
            "host_directory": ipc_host_dir,
            "policy_socket_mounted": False,
            "scene_bytes_mounted": False,
        },
        "security": {
            "network_namespace": "proxy_sidecar_network_none_shared_by_policy",
            "policy_mounts": [],
            "policy_environment_credentials": [],
            "docker_socket_mounted": False,
            "root_filesystem_read_only": True,
            "capabilities_dropped": "all",
            "no_new_privileges": True,
            "raw_container_logs_retained": False,
            "seccomp_profile_id": seccomp_profile_id,
            "seccomp_profile_digest": seccomp_profile_digest,
            "apparmor_profile_id": apparmor_profile_id,
            "apparmor_profile_digest": apparmor_profile_digest,
        },
        "images": {
            "policy": image,
            "blueprint_proxy": blueprint_proxy_image,
            "blueprint_proxy_contract_digest": blueprint_proxy_contract_digest,
        },
        "phase_order": [
            "credential_redeemed_for_exact_pull",
            "policy_image_pulled_and_digest_verified",
            "registry_credential_and_docker_config_deleted",
            "network_namespace_created",
            "blueprint_proxy_started",
            "policy_container_started",
            "no_egress_measured",
            "synthetic_conformance_completed",
            "first_observation_permitted",
        ],
        "commands": {
            "image_pull": ["docker", "--config", docker_config_dir, "pull", image],
            "proxy_run": proxy_run_argv,
            "policy_run": policy_run_argv,
            "cleanup": [
                ["docker", "rm", "--force", policy_name],
                ["docker", "rm", "--force", proxy_name],
                ["docker", "image", "rm", image],
            ],
        },
        "cleanup_targets": {
            "policy_container": policy_name,
            "proxy_container": proxy_name,
            "policy_image": image,
            "registry_config_directory": docker_config_dir,
            "ipc_directory": ipc_host_dir,
        },
        "required_denial_probes": list(DENIAL_PROBES),
        "required_reachability_probes": list(REACHABILITY_PROBES),
        "launch_authority_granted": False,
        "provider_mutation_authorized": False,
        "provider_mutation_performed": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = cross_runtime_canonical_digest(plan, digest_field="plan_digest")
    return plan


def evaluate_preobservation_sandbox_qualification(
    *, plan: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Require measured denial and conformance before observation 1."""

    blockers: list[str] = []
    expected_plan_digest = cross_runtime_canonical_digest(plan, digest_field="plan_digest")
    if plan.get("plan_digest") != expected_plan_digest:
        blockers.append("company_policy_sandbox_plan_digest_mismatch")
    if evidence.get("plan_digest") != expected_plan_digest:
        blockers.append("company_policy_sandbox_evidence_plan_mismatch")
    if evidence.get("policy_image_digest_verified") is not True:
        blockers.append("company_policy_sandbox_policy_image_not_verified")
    if evidence.get("credential_deleted_before_policy_start") is not True:
        blockers.append("company_policy_sandbox_credential_not_deleted_before_start")
    if evidence.get("docker_config_deleted_before_policy_start") is not True:
        blockers.append("company_policy_sandbox_docker_config_not_deleted_before_start")
    if evidence.get("dedicated_worker_verified") is not True:
        blockers.append("company_policy_sandbox_dedicated_worker_unverified")
    if evidence.get("isolated_runtime_verified") is not True:
        blockers.append("company_policy_sandbox_runtime_unverified")
    if evidence.get("raw_container_logs_retained") is not False:
        blockers.append("company_policy_sandbox_raw_logs_retained")
    if evidence.get("teardown_plan_armed") is not True:
        blockers.append("company_policy_sandbox_teardown_not_armed")
    probes = evidence.get("network_probes")
    if not isinstance(probes, Mapping):
        probes = {}
    for probe in DENIAL_PROBES:
        row = probes.get(probe)
        if not isinstance(row, Mapping) or row.get("status") != "denied":
            blockers.append(f"company_policy_sandbox_egress_probe_not_denied:{probe}")
        elif row.get("redirect_followed") is not False:
            blockers.append(f"company_policy_sandbox_redirect_observed:{probe}")
    for probe in REACHABILITY_PROBES:
        row = probes.get(probe)
        if not isinstance(row, Mapping) or row.get("status") != "reachable":
            blockers.append(f"company_policy_sandbox_required_path_unreachable:{probe}")
    conformance = evidence.get("synthetic_conformance")
    if (
        not isinstance(conformance, Mapping)
        or conformance.get("status") != "conformant"
        or conformance.get("contract_digest") != plan.get("contract_digest")
        or conformance.get("real_observation_used") is not False
    ):
        blockers.append("company_policy_sandbox_synthetic_conformance_invalid")
    sequence = evidence.get("completed_phase_order")
    expected_sequence = list(plan.get("phase_order") or [])[:-1]
    if sequence != expected_sequence:
        blockers.append("company_policy_sandbox_phase_order_invalid")
    if blockers:
        raise CompanyPolicySandboxV2Error(blockers)
    receipt = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "status": "qualified_before_first_observation",
        "plan_digest": expected_plan_digest,
        "admission_digest": plan["admission_digest"],
        "contract_digest": plan["contract_digest"],
        "sandbox_attempt_id": plan["sandbox_attempt_id"],
        "pipeline_release_sha": plan["pipeline_release_sha"],
        "worker_identity": plan["worker_identity"],
        "network_denial_probe_count": len(DENIAL_PROBES),
        "required_path_probe_count": len(REACHABILITY_PROBES),
        "synthetic_conformance_completed": True,
        "first_observation_permitted": True,
        "launch_authority_granted": False,
        "provider_mutation_authorized": False,
        "provider_mutation_performed": False,
        "qualification_digest": "",
    }
    receipt["qualification_digest"] = cross_runtime_canonical_digest(
        receipt, digest_field="qualification_digest"
    )
    return receipt
