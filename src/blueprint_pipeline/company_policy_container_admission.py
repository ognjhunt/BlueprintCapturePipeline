"""Immutable, no-spend admission for company policy-container declarations.

Admission proves only that an authenticated WebApp handoff is structurally
valid, cross-bound, digest stable, secret free, and durably retained.  It does
not redeem registry credentials, create a launch profile, queue work, or grant
provider authority.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from .company_policy_container_contract_v2 import (
    CompanyPolicyContainerContractV2Error,
    validate_company_policy_container_contract_v2,
)
from .decision_evidence_contracts import cross_runtime_canonical_digest
from .common import utc_now_iso


REQUEST_SCHEMA_VERSION = "company_policy_container_admission_request.v1"
RECEIPT_SCHEMA_VERSION = "company_policy_container_admission_receipt.v1"
DEFAULT_CLAIM_CEILING = "development_only"
ADMISSION_ROOT_ENV = "BLUEPRINT_COMPANY_POLICY_CONTAINER_ADMISSION_ROOT"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_LEASE_ID = re.compile(r"^policy-registry-lease-[0-9a-f]{47}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "tenant_id",
        "run_id",
        "submission_id",
        "company_id",
        "contract_digest",
        "contract",
        "registry_credential_lease_id",
        "claim_ceiling",
        "launch_authority_granted",
        "provider_mutation_authorized",
    }
)
_SECRET_KEYS = frozenset(
    {
        "authorization",
        "authorization_header",
        "credential",
        "docker_config_json",
        "encrypted_credential",
        "password",
        "registry_password",
        "registry_secret",
        "secret",
        "token",
    }
)


class CompanyPolicyContainerAdmissionError(ValueError):
    """Fail-closed admission error with stable blocker identifiers."""

    def __init__(self, blockers: list[str] | tuple[str, ...], *, status_code: int = 422):
        self.blockers = tuple(sorted({str(item) for item in blockers if str(item)}))
        self.status_code = status_code
        super().__init__(";".join(self.blockers))


def _contains_secret_carrier(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() in _SECRET_KEYS or _contains_secret_carrier(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_carrier(item) for item in value)
    return False


def _identifier(value: Any, *, field: str, blockers: list[str]) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not _IDENTIFIER.fullmatch(text):
        blockers.append(f"company_policy_container_admission_invalid:{field}")
    return text


def validate_company_policy_container_admission_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the secret-free WebApp-to-Pipeline handoff."""

    if not isinstance(value, Mapping):
        raise CompanyPolicyContainerAdmissionError(
            ["company_policy_container_admission_invalid:not_mapping"]
        )
    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CompanyPolicyContainerAdmissionError(
            ["company_policy_container_admission_invalid:not_json"]
        ) from exc
    blockers: list[str] = []
    unknown = sorted(str(key) for key in payload if key not in _TOP_LEVEL_FIELDS)
    blockers.extend(
        f"company_policy_container_admission_invalid:unknown_field:{field}"
        for field in unknown
    )
    if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("company_policy_container_admission_invalid:schema_version")
    if _contains_secret_carrier(payload):
        blockers.append("company_policy_container_admission_secret_carrier_detected")

    tenant_id = _identifier(payload.get("tenant_id"), field="tenant_id", blockers=blockers)
    run_id = _identifier(payload.get("run_id"), field="run_id", blockers=blockers)
    submission_id = _identifier(
        payload.get("submission_id"), field="submission_id", blockers=blockers
    )
    company_id = _identifier(
        payload.get("company_id"), field="company_id", blockers=blockers
    )
    declared_digest = payload.get("contract_digest")
    if not isinstance(declared_digest, str) or not _DIGEST.fullmatch(declared_digest):
        blockers.append("company_policy_container_admission_invalid:contract_digest")

    contract_value = payload.get("contract")
    try:
        contract = validate_company_policy_container_contract_v2(contract_value)
    except CompanyPolicyContainerContractV2Error as exc:
        blockers.extend(exc.errors)
        contract = {}
    if contract:
        if contract.get("contract_digest") != declared_digest:
            blockers.append("company_policy_container_admission_contract_digest_mismatch")
        if contract.get("company_id") != company_id:
            blockers.append("company_policy_container_admission_company_id_mismatch")

    lease_value = payload.get("registry_credential_lease_id")
    lease_id = lease_value.strip() if isinstance(lease_value, str) else None
    visibility = ((contract.get("container") or {}).get("visibility") if contract else None)
    if visibility == "private":
        if not lease_id or not _LEASE_ID.fullmatch(lease_id):
            blockers.append("company_policy_container_admission_private_lease_required")
    elif visibility == "public" and lease_value is not None:
        blockers.append("company_policy_container_admission_public_lease_forbidden")

    if payload.get("claim_ceiling") != DEFAULT_CLAIM_CEILING:
        blockers.append("company_policy_container_admission_claim_ceiling_invalid")
    if payload.get("launch_authority_granted") is not False:
        blockers.append("company_policy_container_admission_launch_authority_forbidden")
    if payload.get("provider_mutation_authorized") is not False:
        blockers.append("company_policy_container_admission_provider_authority_forbidden")
    if blockers:
        raise CompanyPolicyContainerAdmissionError(blockers)
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "run_id": run_id,
        "submission_id": submission_id,
        "company_id": company_id,
        "contract_digest": declared_digest,
        "contract": contract,
        "registry_credential_lease_id": lease_id,
        "claim_ceiling": DEFAULT_CLAIM_CEILING,
        "launch_authority_granted": False,
        "provider_mutation_authorized": False,
    }


def default_company_policy_container_admission_root(work_root: Path) -> Path:
    configured = str(os.getenv(ADMISSION_ROOT_ENV) or "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else work_root.expanduser().resolve() / "company_policy_container_admissions"
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if path.read_bytes() != encoded:
            raise CompanyPolicyContainerAdmissionError(
                ["company_policy_container_admission_atomic_readback_failed"],
                status_code=500,
            )
    finally:
        temporary.unlink(missing_ok=True)


def stage_company_policy_container_admission(
    *, value: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Durably stage one admission without consuming credentials or launching."""

    request = validate_company_policy_container_admission_request(value)
    request_digest = cross_runtime_canonical_digest(request)
    identity_material = {
        "tenant_id": request["tenant_id"],
        "run_id": request["run_id"],
        "submission_id": request["submission_id"],
        "company_id": request["company_id"],
        "contract_digest": request["contract_digest"],
    }
    admission_id = "company-policy-admission-" + sha256(
        json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:40]
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    admission_root = root / admission_id
    admission_root.mkdir(mode=0o700, exist_ok=True)
    lock_path = admission_root / ".lock"
    request_path = admission_root / "admission_request.json"
    receipt_path = admission_root / "admission_receipt.json"
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if request_path.is_file() or receipt_path.is_file():
            try:
                existing_request = json.loads(request_path.read_text(encoding="utf-8"))
                existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CompanyPolicyContainerAdmissionError(
                    ["company_policy_container_admission_existing_state_invalid"],
                    status_code=409,
                ) from exc
            if existing_request != request or existing_receipt.get("request_digest") != request_digest:
                raise CompanyPolicyContainerAdmissionError(
                    ["company_policy_container_admission_idempotency_conflict"],
                    status_code=409,
                )
            return {**existing_receipt, "already_exists": True}

        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "admitted_no_spend",
            "accepted": True,
            "already_exists": False,
            "admission_id": admission_id,
            "admission_digest": cross_runtime_canonical_digest(
                {"identity": identity_material, "request_digest": request_digest}
            ),
            "request_digest": request_digest,
            "contract_digest": request["contract_digest"],
            "tenant_id": request["tenant_id"],
            "run_id": request["run_id"],
            "submission_id": request["submission_id"],
            "company_id": request["company_id"],
            "registry_credential_lease_id": request["registry_credential_lease_id"],
            "registry_credential_consumed": False,
            "profile_published": False,
            "launch_queued": False,
            "launch_authority_granted": False,
            "provider_mutation_authorized": False,
            "provider_mutation_performed": False,
            "claim_ceiling": DEFAULT_CLAIM_CEILING,
            "admitted_at_iso": utc_now_iso(),
            "proof_boundary": {
                "contract_validated": True,
                "immutable_admission_retained": True,
                "sandbox_qualified": False,
                "synthetic_conformance_completed": False,
                "policy_episode_completed": False,
            },
        }
        _atomic_write_json(request_path, request)
        _atomic_write_json(receipt_path, receipt)
        return receipt
