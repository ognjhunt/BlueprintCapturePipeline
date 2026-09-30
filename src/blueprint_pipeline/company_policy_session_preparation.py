"""Prepare one job-bound policy sandbox session on a trusted worker.

The caller supplies an already admitted company contract and a confirmed
canonical run request.  This module only seals the inputs for the isolated
worker; it neither rents a GPU nor exposes a scene to the policy container.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from .company_policy_sandbox_v2 import (
    build_company_policy_sandbox_plan, company_policy_registry_host,
)
from .controlled_policy_configuration import canonical_request_digest
from .decision_evidence_contracts import cross_runtime_canonical_digest


_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,191}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _private_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(value, output, sort_keys=True, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())


def _private_secret(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())


def prepare_company_policy_session(
    *,
    job_request: Mapping[str, Any],
    contract: Mapping[str, Any],
    admission_receipt: Mapping[str, Any],
    root: Path,
    pipeline_release_sha: str,
    worker_identity: str,
    proxy_image: str,
    proxy_contract_digest: str,
    seccomp_profile_path: str,
    seccomp_profile_digest: str,
    apparmor_profile_source_path: str,
    apparmor_profile_digest: str,
    registry_addresses: list[str],
    allowed_registry_hosts: list[str],
    approved_by: str,
    lifetime_seconds: int = 2400,
) -> dict[str, Any]:
    """Seal a fresh, private session bound to one canonical request and image.

    Existing session directories are never reused.  A retry must inspect the
    prior process and terminal evidence before it requests a new session.
    """
    request = json.loads(json.dumps(job_request, allow_nan=False))
    job_id = request.get("job_id")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError("company_policy_session_job_id_invalid")
    if not _SHA.fullmatch(pipeline_release_sha):
        raise ValueError("company_policy_session_release_invalid")
    if not isinstance(lifetime_seconds, int) or not 60 <= lifetime_seconds <= 3600:
        raise ValueError("company_policy_session_lifetime_invalid")
    if not approved_by or not worker_identity:
        raise ValueError("company_policy_session_authority_invalid")
    normalized = validate_company_policy_container_contract_v2(contract)
    receipt = dict(admission_receipt)
    if (receipt.get("status") != "admitted_no_spend"
            or receipt.get("accepted") is not True
            or receipt.get("run_id") != job_id
            or receipt.get("contract_digest") != normalized["contract_digest"]
            or receipt.get("submission_id") != request.get("robot_profile", {}).get("robot_profile_id")
            or receipt.get("launch_authority_granted") is not False
            or receipt.get("provider_mutation_authorized") is not False):
        raise ValueError("company_policy_session_admission_binding_invalid")
    if not _DIGEST.fullmatch(proxy_contract_digest):
        raise ValueError("company_policy_session_proxy_digest_invalid")
    artifact = ((request.get("policy_package") or {}).get("docker_container") or {}).get("model_artifact")
    if not registry_addresses:
        host = company_policy_registry_host(normalized["container"]["image"])
        registry_addresses = sorted({str(address)
            for family, _, _, _, sockaddr in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            if family in {socket.AF_INET, socket.AF_INET6}
            for address in [ipaddress.ip_address(sockaddr[0])]
            if address.is_global})
        if not registry_addresses:
            raise ValueError("company_policy_session_registry_resolution_unavailable")
    plan = build_company_policy_sandbox_plan(
        admission_receipt=receipt,
        contract=normalized,
        sandbox_attempt_id="sandbox-attempt-" + hashlib.sha256(job_id.encode()).hexdigest()[:32],
        pipeline_release_sha=pipeline_release_sha,
        worker_identity=worker_identity,
        runtime_class="runsc",
        blueprint_proxy_image=proxy_image,
        blueprint_proxy_contract_digest=proxy_contract_digest,
        seccomp_profile_id="blueprint-policy-seccomp-v2",
        seccomp_profile_path=seccomp_profile_path,
        seccomp_profile_digest=seccomp_profile_digest,
        apparmor_profile_id="blueprint-policy-apparmor-v2",
        apparmor_profile_source_path=apparmor_profile_source_path,
        apparmor_profile_digest=apparmor_profile_digest,
        registry_addresses=registry_addresses,
        allowed_registry_hosts=allowed_registry_hosts,
        model_artifact=artifact,
    )
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    session = root / job_id
    session.mkdir(mode=0o700, exist_ok=False)
    key = secrets.token_urlsafe(48).encode("ascii")
    bearer = secrets.token_urlsafe(48).encode("ascii")
    key_id = "blueprint-policy-session-" + hashlib.sha256(job_id.encode()).hexdigest()[:32]
    boot = {
        "schema_version": "company_policy_worker_boot_receipt.v1",
        "source": "trusted_company_policy_worker_bootstrap",
        "status": "dedicated_ephemeral_worker_ready",
        "worker_identity": worker_identity,
        "pipeline_release_sha": pipeline_release_sha,
        "sandbox_attempt_id": plan["sandbox_attempt_id"],
        "dedicated_ephemeral_worker": True,
        "scene_bytes_present": False,
        "observation_bytes_present": False,
        "mounted_customer_input_paths": [],
        "attestation_key_id": key_id,
    }
    boot["receipt_digest"] = cross_runtime_canonical_digest(boot)
    boot["attestation_hmac_sha256"] = hmac.new(
        key, boot["receipt_digest"].encode(), hashlib.sha256
    ).hexdigest()
    authority = {
        "schema_version": "blueprint.controlled_policy_bridge_authority.v1",
        "job_id": job_id,
        "canonical_request_digest": canonical_request_digest(request),
        "plan_digest": plan["plan_digest"],
        "expires_at_iso": (datetime.now(timezone.utc) + timedelta(seconds=lifetime_seconds)).isoformat(),
        "approved_by": approved_by,
        "max_policy_calls": 128,
    }
    for name, value in (
        ("plan.json", plan),
        ("contract.json", normalized),
        ("admission-receipt.json", receipt),
        ("job-request.json", request),
        ("worker-boot-receipt.json", boot),
        ("authority.json", authority),
    ):
        _private_json(session / name, value)
    _private_secret(session / "attestation-key", key)
    _private_secret(session / "bearer-token", bearer)
    descriptor = os.open(session, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "job_id": job_id,
        "canonical_request_digest": authority["canonical_request_digest"],
        "contract_digest": normalized["contract_digest"],
        "plan_digest": plan["plan_digest"],
        "session_dir": str(session),
        "attestation_key_id": key_id,
        "expires_at_iso": authority["expires_at_iso"],
    }
