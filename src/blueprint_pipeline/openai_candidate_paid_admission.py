"""Fail-closed admission contract for paid OpenAI candidate execution.

This module is deliberately provider-mutation free.  The canonical paid
allocator supplies checkout identity, writes the resulting record, and issues
the opaque in-process grant only after this contract returns ``admitted``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.candidate_policy import (
    CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION,
)
from .task_evaluation_supervisor.phase2_artifacts import (
    AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
)


OPENAI_API_CANDIDATE_ADMISSION_SCHEMA_VERSION = "openai_api_candidate_allocation_admission.v1"
OPENAI_API_CANDIDATE_RESOURCE_CLASS = "openai_api_candidate"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _utc(value: Any, *, blocker: str, blockers: list[str]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        blockers.append(blocker)
        return None
    if parsed.tzinfo is None:
        blockers.append(blocker)
        return None
    return parsed.astimezone(timezone.utc)


def _finite_number(value: Any, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= minimum else None


def build_openai_api_candidate_admission(
    *,
    suite: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    candidate_id: str,
    provider_id: str,
    runtime_configuration_digest: str,
    cost_authority_binding_digest: str,
    license_attestation_digest: str,
    expected_source_commit: str,
    checkout_source_commit: str,
    checkout_clean: bool,
    maximum_execution_seconds: float,
    runtime_watchdog_enforced: bool,
    teardown_enforced: bool,
    execute_requested: bool,
    source_authority_blockers: tuple[str, ...] = (),
    admitted_at: str | None = None,
) -> dict[str, Any]:
    """Bind one paid candidate to immutable source, authority, and limits.

    The returned artifact has no proof effect.  ``admitted`` means only that
    the canonical allocator may issue an in-process capability; it does not
    mean that a provider call ran, an evaluation passed, or cost reconciled.
    """

    blockers: list[str] = [
        str(blocker).replace("gpu_canary_", "openai_candidate_")
        for blocker in source_authority_blockers
        if str(blocker).strip()
    ]
    suite_value = dict(suite)
    authorization = dict(execution_authorization)
    suite_digest = canonical_digest(
        suite_value,
        digest_field="candidate_evaluation_suite_digest",
    )
    authorization_digest = canonical_digest(
        authorization,
        digest_field="authorization_receipt_digest",
    )
    if (
        suite_value.get("schema_version") != CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION
        or suite_value.get("candidate_evaluation_suite_digest") != suite_digest
    ):
        blockers.append("openai_candidate_suite_invalid")
    if (
        authorization.get("schema_version") != AUTHORIZATION_RECEIPT_SCHEMA_VERSION
        or authorization.get("authorization_receipt_digest") != authorization_digest
    ):
        blockers.append("openai_candidate_authorization_invalid")
    if (
        authorization.get("approved") is not True
        or authorization.get("issued_by_agent") is not False
        or authorization.get("granted_tool_id") != "execute_candidate_policy_suite"
        or authorization.get("proof_effect") != "none"
        or not str(authorization.get("operator_id") or "").strip()
    ):
        blockers.append("openai_candidate_operator_authority_missing")

    specs = [
        dict(row)
        for row in suite_value.get("candidate_evaluation_run_specs") or []
        if isinstance(row, Mapping)
        and str((row.get("policy_adapter") or {}).get("policy_id") or "") == candidate_id
    ]
    if len(specs) != 1:
        blockers.append("openai_candidate_spec_not_unique")
        policy: dict[str, Any] = {}
    else:
        policy = dict(specs[0].get("policy_adapter") or {})
        metadata = dict(specs[0].get("metadata") or {})
        if (
            policy.get("runtime_configuration_digest") != runtime_configuration_digest
            or metadata.get("candidate_policy_manifest_digest")
            != policy.get("candidate_policy_manifest_digest")
            or policy.get("hidden_labels_included") is not False
        ):
            blockers.append("openai_candidate_runtime_binding_invalid")

    immutable_inputs = sorted(
        str(row) for row in authorization.get("immutable_input_digests") or []
    )
    expected_inputs = sorted(
        [suite_digest, str(suite_value.get("hidden_label_manifest_digest") or "")]
    )
    if immutable_inputs != expected_inputs:
        blockers.append("openai_candidate_authorized_inputs_mismatch")
    if sorted(str(row) for row in authorization.get("granted_action_ids") or []) != sorted(
        str((row.get("policy_adapter") or {}).get("policy_id") or "")
        for row in suite_value.get("candidate_evaluation_run_specs") or []
        if isinstance(row, Mapping)
    ):
        blockers.append("openai_candidate_actions_not_authorized")
    paid_provider_ids = {provider_id} if specs and provider_id.strip() else set()
    if provider_id not in [str(row) for row in authorization.get("granted_provider_ids") or []]:
        blockers.append("openai_candidate_provider_not_authorized")

    max_cost = _finite_number(policy.get("max_cost_usd"))
    granted_cost = _finite_number(authorization.get("granted_max_cost_usd"))
    retry_limit = policy.get("retry_limit")
    granted_retries = authorization.get("granted_retry_count")
    maximum_seconds = _finite_number(maximum_execution_seconds, minimum=0.001)
    granted_ttl = _finite_number(authorization.get("granted_ttl_seconds"), minimum=0.001)
    if max_cost is None or granted_cost is None or granted_cost < max_cost:
        blockers.append("openai_candidate_spend_envelope_insufficient")
    if (
        isinstance(retry_limit, bool)
        or not isinstance(retry_limit, int)
        or isinstance(granted_retries, bool)
        or not isinstance(granted_retries, int)
        or granted_retries < retry_limit
    ):
        blockers.append("openai_candidate_retry_envelope_insufficient")
    if maximum_seconds is None or granted_ttl is None or maximum_seconds > granted_ttl:
        blockers.append("openai_candidate_ttl_envelope_insufficient")

    now = _utc(
        admitted_at or datetime.now(timezone.utc).isoformat(),
        blocker="openai_candidate_admission_time_invalid",
        blockers=blockers,
    )
    issued = _utc(
        authorization.get("issued_at"),
        blocker="openai_candidate_authority_time_invalid",
        blockers=blockers,
    )
    expires = _utc(
        authorization.get("expires_at"),
        blocker="openai_candidate_authority_time_invalid",
        blockers=blockers,
    )
    if now is not None and issued is not None and expires is not None:
        if now < issued or now >= expires or expires <= issued:
            blockers.append("openai_candidate_authority_inactive")
        elif granted_ttl is not None and (expires - issued).total_seconds() > granted_ttl:
            blockers.append("openai_candidate_authority_ttl_invalid")

    expected_commit = str(expected_source_commit or "").strip().lower()
    checkout_commit = str(checkout_source_commit or "").strip().lower()
    if not _COMMIT.fullmatch(expected_commit) or checkout_commit != expected_commit:
        blockers.append("openai_candidate_source_commit_mismatch")
    if checkout_clean is not True:
        blockers.append("openai_candidate_checkout_not_clean")
    if not _SHA256.fullmatch(runtime_configuration_digest):
        blockers.append("openai_candidate_runtime_digest_invalid")
    if not _SHA256.fullmatch(cost_authority_binding_digest):
        blockers.append("openai_candidate_cost_authority_binding_invalid")
    if not _SHA256.fullmatch(license_attestation_digest):
        blockers.append("openai_candidate_license_attestation_invalid")
    if not provider_id.strip() or not candidate_id.strip() or not paid_provider_ids:
        blockers.append("openai_candidate_identity_invalid")
    if runtime_watchdog_enforced is not True:
        blockers.append("openai_candidate_runtime_watchdog_missing")
    if teardown_enforced is not True:
        blockers.append("openai_candidate_teardown_missing")
    if not isinstance(execute_requested, bool):
        blockers.append("openai_candidate_execute_flag_invalid")

    normalized_blockers = sorted(set(blockers))
    status = (
        "blocked" if normalized_blockers else "admitted" if execute_requested else "dry_run_ready"
    )
    value: dict[str, Any] = {
        "schema_version": OPENAI_API_CANDIDATE_ADMISSION_SCHEMA_VERSION,
        "status": status,
        "resource_class": OPENAI_API_CANDIDATE_RESOURCE_CLASS,
        "blockers": normalized_blockers,
        "candidate_id": candidate_id,
        "provider_id": provider_id,
        "candidate_evaluation_suite_digest": suite_digest,
        "authorization_receipt_digest": authorization_digest,
        "runtime_configuration_digest": runtime_configuration_digest,
        "cost_authority_binding_digest": cost_authority_binding_digest,
        "license_attestation_digest": license_attestation_digest,
        "expected_source_commit": expected_commit or None,
        "checkout_source_commit": checkout_commit or None,
        "checkout_clean": checkout_clean is True,
        "maximum_execution_seconds": maximum_seconds,
        "granted_ttl_seconds": granted_ttl,
        "candidate_max_cost_usd": max_cost,
        "granted_max_cost_usd": granted_cost,
        "candidate_retry_limit": retry_limit,
        "granted_retry_count": granted_retries,
        "runtime_watchdog_enforced": runtime_watchdog_enforced is True,
        "teardown_enforced": teardown_enforced is True,
        "persistent_provider_resource_created": False,
        "provider_mutations_performed": 0,
        "execute_requested": execute_requested,
        "candidate_reported_cost_is_authoritative": False,
        "evaluator_authority_granted": False,
        "proof_effect": "none",
        "admitted_at": now.isoformat() if now is not None else None,
    }
    value["allocation_binding_digest"] = canonical_digest(
        value,
        digest_field="allocation_binding_digest",
    )
    return value


def prepare_openai_api_candidate_admission(
    *,
    suite: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    candidate_id: str,
    provider_id: str,
    runtime_configuration_digest: str,
    cost_authority_binding_digest: str,
    license_attestation_digest: str,
    expected_source_commit: str,
    maximum_execution_seconds: float,
    runtime_watchdog_enforced: bool,
    teardown_enforced: bool,
    admission_out: str | Path,
    source_checkout_validator: Callable[..., tuple[list[str], str]],
    checkout_state_reader: Callable[[], tuple[str, bool]],
    execute: bool = False,
    experimental_branch_diagnostic: bool = False,
    admitted_at: str | None = None,
) -> dict[str, Any]:
    """Build and persist a proposal; this helper cannot issue execution authority."""

    source_blockers, checkout_commit = source_checkout_validator(
        expected_source_commit,
        allow_pushed_branch_diagnostic=experimental_branch_diagnostic,
    )
    _observed_commit, checkout_clean = checkout_state_reader()
    admission = build_openai_api_candidate_admission(
        suite=suite,
        execution_authorization=execution_authorization,
        candidate_id=candidate_id,
        provider_id=provider_id,
        runtime_configuration_digest=runtime_configuration_digest,
        cost_authority_binding_digest=cost_authority_binding_digest,
        license_attestation_digest=license_attestation_digest,
        expected_source_commit=expected_source_commit,
        checkout_source_commit=checkout_commit,
        checkout_clean=checkout_clean,
        maximum_execution_seconds=maximum_execution_seconds,
        runtime_watchdog_enforced=runtime_watchdog_enforced,
        teardown_enforced=teardown_enforced,
        execute_requested=execute,
        source_authority_blockers=tuple(source_blockers),
        admitted_at=admitted_at,
    )
    write_json(Path(admission_out).expanduser().resolve(), admission)
    return admission


def prepare_pigey_candidate_runtime_admission(
    *,
    runtime: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a Pigey admission proposal without granting execution authority."""

    return prepare_openai_api_candidate_admission(
        candidate_id=str(runtime.candidate_id),
        provider_id=str(runtime.provider_id),
        runtime_configuration_digest=str(runtime.runtime_configuration_digest),
        cost_authority_binding_digest=str(runtime.cost_authority_binding_digest),
        license_attestation_digest=str(
            runtime.license_attestation["license_attestation_digest"]
        ),
        maximum_execution_seconds=float(runtime.maximum_execution_seconds),
        runtime_watchdog_enforced=runtime.runtime_watchdog_enforced is True,
        teardown_enforced=runtime.teardown_enforced is True,
        **kwargs,
    )


__all__ = [
    "OPENAI_API_CANDIDATE_ADMISSION_SCHEMA_VERSION",
    "OPENAI_API_CANDIDATE_RESOURCE_CLASS",
    "build_openai_api_candidate_admission",
    "prepare_openai_api_candidate_admission",
    "prepare_pigey_candidate_runtime_admission",
]
