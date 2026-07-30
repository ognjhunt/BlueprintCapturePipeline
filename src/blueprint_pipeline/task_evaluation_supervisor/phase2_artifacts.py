"""Deterministic Phase 2 artifacts around non-authoritative agent proposals."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..common import write_json
from ..decision_evidence_contracts import MaintainedSiteTaskTestbed, canonical_digest
from .capture_ingress import CaptureBuildIngressError, validate_capture_build_ingress


CUSTOMER_REPORT_SCHEMA_VERSION = "task_evaluation_customer_report.v1"
CLARIFICATION_REQUEST_SCHEMA_VERSION = "task_evaluation_clarification_request.v1"
CLARIFICATION_RECEIPT_SCHEMA_VERSION = "task_evaluation_clarification_receipt.v1"
AUTHORIZATION_REQUEST_SCHEMA_VERSION = "task_evaluation_authorization_request.v1"
AUTHORIZATION_RECEIPT_SCHEMA_VERSION = "task_evaluation_authorization_receipt.v1"
TARGETED_RECAPTURE_REQUEST_SCHEMA_VERSION = "targeted_recapture_request.v1"
TARGETED_RECAPTURE_RECEIPT_SCHEMA_VERSION = "task_evaluation_targeted_recapture_receipt.v1"
RECAPTURE_REINSPECTION_SCHEMA_VERSION = "task_evaluation_recapture_reinspection.v1"
SCENARIO_PROPOSAL_SET_SCHEMA_VERSION = "task_evaluation_scenario_proposal_set.v1"
FROZEN_SCENARIO_MANIFEST_SCHEMA_VERSION = "task_evaluation_frozen_scenario_manifest.v1"
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class Phase2ArtifactError(ValueError):
    """Raised when a Phase 2 artifact would cross an authority boundary."""


def _strings(
    values: Any,
    *,
    field: str,
    minimum: int = 1,
    maximum: int = 50,
    item_maximum: int = 500,
) -> list[str]:
    if not isinstance(values, list):
        raise Phase2ArtifactError(f"{field}:must_be_list")
    normalized = sorted(
        {str(item).strip() for item in values if isinstance(item, str) and str(item).strip()}
    )
    if len(normalized) < minimum or len(normalized) > maximum:
        raise Phase2ArtifactError(f"{field}:count_out_of_range")
    if any(len(item) > item_maximum for item in normalized):
        raise Phase2ArtifactError(f"{field}:item_too_long")
    return normalized


def _require_digest(value: Any, *, field: str) -> str:
    rendered = str(value or "")
    if not _SHA256_DIGEST.fullmatch(rendered):
        raise Phase2ArtifactError(f"{field}:invalid_digest")
    return rendered


def _is_nonnegative_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _parse_time(value: Any, *, field: str) -> datetime:
    text = str(value or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise Phase2ArtifactError(f"{field}:invalid") from exc
    if parsed.tzinfo is None:
        raise Phase2ArtifactError(f"{field}:timezone_required")
    return parsed.astimezone(timezone.utc)


def _finalize(value: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    result = dict(value)
    result[digest_field] = canonical_digest(result, digest_field=digest_field)
    return result


def clarification_request(
    *,
    run_id: str,
    source_digest: str,
    questions: Sequence[str],
    blocking_fields: Sequence[str],
) -> dict[str, Any]:
    value = {
        "schema_version": CLARIFICATION_REQUEST_SCHEMA_VERSION,
        "request_id": f"{run_id}-clarification",
        "run_id": run_id,
        "source_digest": _require_digest(source_digest, field="source_digest"),
        "questions": _strings(list(questions), field="questions", maximum=20),
        "blocking_fields": _strings(
            list(blocking_fields), field="blocking_fields", maximum=30, item_maximum=100
        ),
        "status": "awaiting_customer_response",
        "agent_may_answer": False,
        "proof_effect": "none",
    }
    return validate_clarification_request(
        _finalize(value, digest_field="clarification_request_digest")
    )


def validate_clarification_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact non-authoritative customer clarification request."""

    required_fields = {
        "schema_version",
        "request_id",
        "run_id",
        "source_digest",
        "questions",
        "blocking_fields",
        "status",
        "agent_may_answer",
        "proof_effect",
        "clarification_request_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("clarification_request_fields_invalid")
    run_id = str(value.get("run_id") or "").strip()
    questions = _strings(value.get("questions"), field="questions", maximum=20)
    blocking_fields = _strings(
        value.get("blocking_fields"),
        field="blocking_fields",
        maximum=30,
        item_maximum=100,
    )
    expected = canonical_digest(value, digest_field="clarification_request_digest")
    if (
        value.get("schema_version") != CLARIFICATION_REQUEST_SCHEMA_VERSION
        or not run_id
        or value.get("request_id") != f"{run_id}-clarification"
        or value.get("clarification_request_digest") != expected
        or list(value.get("questions") or []) != questions
        or list(value.get("blocking_fields") or []) != blocking_fields
        or value.get("status") != "awaiting_customer_response"
        or value.get("agent_may_answer") is not False
        or value.get("proof_effect") != "none"
    ):
        raise Phase2ArtifactError("clarification_request_contract_invalid")
    _require_digest(value.get("source_digest"), field="source_digest")
    return dict(value)


def _validate_untrusted_response_json(value: Any, *, depth: int = 0) -> None:
    if depth > 8:
        raise Phase2ArtifactError("clarification_response_depth_exceeded")
    if isinstance(value, Mapping):
        if len(value) > 100 or any(
            not isinstance(key, str) or not key.strip() or len(key) > 200 for key in value
        ):
            raise Phase2ArtifactError("clarification_response_mapping_invalid")
        for item in value.values():
            _validate_untrusted_response_json(item, depth=depth + 1)
        return
    if isinstance(value, list):
        if len(value) > 200:
            raise Phase2ArtifactError("clarification_response_list_too_large")
        for item in value:
            _validate_untrusted_response_json(item, depth=depth + 1)
        return
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 10_000:
            raise Phase2ArtifactError("clarification_response_string_too_large")
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise Phase2ArtifactError("clarification_response_value_invalid")


def validate_clarification_receipt(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a bounded response while preserving its untrusted status."""

    required_fields = {
        "schema_version",
        "receipt_id",
        "run_id",
        "clarification_request_digest",
        "responder_id",
        "responses",
        "received_at",
        "accepted_as_customer_input",
        "requires_deterministic_contract_validation",
        "response_is_untrusted",
        "responder_identity_verified_by_supervisor",
        "proof_effect",
        "clarification_receipt_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("clarification_receipt_fields_invalid")
    responses = value.get("responses")
    if not isinstance(responses, Mapping) or not responses:
        raise Phase2ArtifactError("clarification_receipt_responses_invalid")
    _validate_untrusted_response_json(responses)
    try:
        serialized = json.dumps(responses, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise Phase2ArtifactError("clarification_receipt_responses_invalid") from exc
    if len(serialized.encode("utf-8")) > 100_000:
        raise Phase2ArtifactError("clarification_receipt_responses_too_large")
    expected = canonical_digest(value, digest_field="clarification_receipt_digest")
    _parse_time(value.get("received_at"), field="clarification_received_at")
    request_digest = _require_digest(
        value.get("clarification_request_digest"),
        field="clarification_request_digest",
    )
    if (
        value.get("schema_version") != CLARIFICATION_RECEIPT_SCHEMA_VERSION
        or value.get("clarification_receipt_digest") != expected
        or not str(value.get("run_id") or "").strip()
        or not str(value.get("responder_id") or "").strip()
        or value.get("receipt_id") != f"{value.get('run_id')}-clarification-receipt"
        or value.get("accepted_as_customer_input") is not False
        or value.get("requires_deterministic_contract_validation") is not True
        or value.get("response_is_untrusted") is not True
        or value.get("responder_identity_verified_by_supervisor") is not False
        or value.get("proof_effect") != "none"
    ):
        raise Phase2ArtifactError("clarification_receipt_contract_invalid")
    if request is not None:
        validated_request = validate_clarification_request(request)
        if (
            value.get("run_id") != validated_request["run_id"]
            or request_digest != validated_request["clarification_request_digest"]
            or value.get("receipt_id") != f"{validated_request['request_id']}-receipt"
        ):
            raise Phase2ArtifactError("clarification_receipt_request_mismatch")
    return dict(value)


def clarification_receipt(
    *,
    request: Mapping[str, Any],
    responder_id: str,
    responses: Mapping[str, Any],
    received_at: str,
) -> dict[str, Any]:
    validated_request = validate_clarification_request(request)
    expected = validated_request["clarification_request_digest"]
    if not responder_id.strip() or not received_at.strip() or not isinstance(responses, Mapping):
        raise Phase2ArtifactError("clarification_receipt_missing_fields")
    value = {
        "schema_version": CLARIFICATION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": f"{validated_request['request_id']}-receipt",
        "run_id": validated_request["run_id"],
        "clarification_request_digest": expected,
        "responder_id": responder_id.strip(),
        "responses": dict(responses),
        "received_at": received_at,
        "accepted_as_customer_input": False,
        "requires_deterministic_contract_validation": True,
        "response_is_untrusted": True,
        "responder_identity_verified_by_supervisor": False,
        "proof_effect": "none",
    }
    return validate_clarification_receipt(
        _finalize(value, digest_field="clarification_receipt_digest"),
        request=validated_request,
    )


def authorization_request(
    *,
    run_id: str,
    tool_id: str,
    reason: str,
    requested_max_cost_usd: float,
    requested_ttl_seconds: int,
    immutable_input_digests: Sequence[str],
    requested_retry_count: int = 0,
    requested_provider_ids: Sequence[str] = (),
    requested_action_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if (
        not math.isfinite(float(requested_max_cost_usd))
        or requested_max_cost_usd < 0
        or requested_ttl_seconds < 1
        or requested_retry_count < 0
    ):
        raise Phase2ArtifactError("authorization_request_envelope_invalid")
    digests = sorted(
        {
            _require_digest(value, field="immutable_input_digest")
            for value in immutable_input_digests
        }
    )
    if not tool_id.strip() or not reason.strip() or not digests:
        raise Phase2ArtifactError("authorization_request_missing_fields")
    provider_ids = _strings(
        list(requested_provider_ids),
        field="requested_provider_ids",
        minimum=0,
        maximum=20,
        item_maximum=100,
    )
    action_ids = _strings(
        list(requested_action_ids),
        field="requested_action_ids",
        minimum=0,
        maximum=50,
        item_maximum=100,
    )
    if tool_id == "execute_preauthorized_recovery" and (not provider_ids or not action_ids):
        raise Phase2ArtifactError("recovery_authorization_scope_missing")
    value = {
        "schema_version": AUTHORIZATION_REQUEST_SCHEMA_VERSION,
        "request_id": f"{run_id}-{tool_id}-authorization",
        "run_id": run_id,
        "tool_id": tool_id,
        "reason": reason[:2_000],
        "requested_max_cost_usd": float(requested_max_cost_usd),
        "requested_ttl_seconds": requested_ttl_seconds,
        "requested_retry_count": requested_retry_count,
        "immutable_input_digests": digests,
        "requested_provider_ids": provider_ids,
        "requested_action_ids": action_ids,
        "status": "awaiting_authorized_operator",
        "agent_may_approve": False,
        "proof_effect": "none",
    }
    return validate_authorization_request(
        _finalize(value, digest_field="authorization_request_digest")
    )


def validate_authorization_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact authority an agent may request but never grant."""

    required_fields = {
        "schema_version",
        "request_id",
        "run_id",
        "tool_id",
        "reason",
        "requested_max_cost_usd",
        "requested_ttl_seconds",
        "requested_retry_count",
        "immutable_input_digests",
        "requested_provider_ids",
        "requested_action_ids",
        "status",
        "agent_may_approve",
        "proof_effect",
        "authorization_request_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("authorization_request_fields_invalid")
    run_id = str(value.get("run_id") or "").strip()
    tool_id = str(value.get("tool_id") or "").strip()
    reason = str(value.get("reason") or "").strip()
    cost = value.get("requested_max_cost_usd")
    ttl = value.get("requested_ttl_seconds")
    retries = value.get("requested_retry_count")
    raw_digests = value.get("immutable_input_digests")
    if not isinstance(raw_digests, list):
        raise Phase2ArtifactError("authorization_request_input_digests_invalid")
    digests = sorted(
        {_require_digest(item, field="immutable_input_digest") for item in raw_digests}
    )
    providers = _strings(
        value.get("requested_provider_ids"),
        field="requested_provider_ids",
        minimum=0,
        maximum=20,
        item_maximum=100,
    )
    actions = _strings(
        value.get("requested_action_ids"),
        field="requested_action_ids",
        minimum=0,
        maximum=50,
        item_maximum=100,
    )
    expected = canonical_digest(value, digest_field="authorization_request_digest")
    if (
        value.get("schema_version") != AUTHORIZATION_REQUEST_SCHEMA_VERSION
        or not run_id
        or not tool_id
        or not reason
        or len(reason) > 2_000
        or value.get("request_id") != f"{run_id}-{tool_id}-authorization"
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
        or isinstance(ttl, bool)
        or not isinstance(ttl, int)
        or ttl < 1
        or isinstance(retries, bool)
        or not isinstance(retries, int)
        or retries < 0
        or not digests
        or raw_digests != digests
        or list(value.get("requested_provider_ids") or []) != providers
        or list(value.get("requested_action_ids") or []) != actions
        or (tool_id == "execute_preauthorized_recovery" and (not providers or not actions))
        or value.get("status") != "awaiting_authorized_operator"
        or value.get("agent_may_approve") is not False
        or value.get("proof_effect") != "none"
        or value.get("authorization_request_digest") != expected
    ):
        raise Phase2ArtifactError("authorization_request_contract_invalid")
    return dict(value)


def authorization_receipt(
    *,
    request: Mapping[str, Any],
    operator_id: str,
    approved: bool,
    granted_max_cost_usd: float,
    granted_ttl_seconds: int,
    granted_retry_count: int,
    issued_at: str,
    expires_at: str,
    granted_provider_ids: Sequence[str] = (),
    granted_action_ids: Sequence[str] = (),
) -> dict[str, Any]:
    validated_request = validate_authorization_request(request)
    expected = validated_request["authorization_request_digest"]
    if not operator_id.strip() or not issued_at.strip() or not expires_at.strip():
        raise Phase2ArtifactError("authorization_receipt_missing_fields")
    requested_cost = float(validated_request.get("requested_max_cost_usd") or 0.0)
    requested_ttl = int(validated_request.get("requested_ttl_seconds") or 0)
    requested_retries = int(validated_request.get("requested_retry_count") or 0)
    requested_providers = set(
        _strings(
            validated_request.get("requested_provider_ids") or [],
            field="requested_provider_ids",
            minimum=0,
            maximum=20,
            item_maximum=100,
        )
    )
    requested_actions = set(
        _strings(
            validated_request.get("requested_action_ids") or [],
            field="requested_action_ids",
            minimum=0,
            maximum=50,
            item_maximum=100,
        )
    )
    granted_providers = _strings(
        list(granted_provider_ids),
        field="granted_provider_ids",
        minimum=0,
        maximum=20,
        item_maximum=100,
    )
    granted_actions = _strings(
        list(granted_action_ids),
        field="granted_action_ids",
        minimum=0,
        maximum=50,
        item_maximum=100,
    )
    issued = _parse_time(issued_at, field="authorization_receipt_issued_at")
    expires = _parse_time(expires_at, field="authorization_receipt_expires_at")
    if (
        not math.isfinite(float(granted_max_cost_usd))
        or granted_max_cost_usd < 0
        or granted_max_cost_usd > requested_cost
        or granted_ttl_seconds < 1
        or granted_ttl_seconds > requested_ttl
        or granted_retry_count < 0
        or granted_retry_count > requested_retries
        or not set(granted_providers).issubset(requested_providers)
        or not set(granted_actions).issubset(requested_actions)
        or expires <= issued
        or (expires - issued).total_seconds() > granted_ttl_seconds
    ):
        raise Phase2ArtifactError("authorization_receipt_exceeds_request")
    if not approved and (
        granted_max_cost_usd != 0
        or granted_retry_count != 0
        or granted_providers
        or granted_actions
    ):
        raise Phase2ArtifactError("denied_authorization_cannot_grant_authority")
    if (
        approved
        and validated_request.get("tool_id") == "execute_preauthorized_recovery"
        and (not granted_providers or not granted_actions)
    ):
        raise Phase2ArtifactError("recovery_authorization_scope_missing")
    value = {
        "schema_version": AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": f"{validated_request['request_id']}-receipt",
        "run_id": validated_request["run_id"],
        "authorization_request_digest": expected,
        "operator_id": operator_id.strip(),
        "approved": bool(approved),
        "granted_tool_id": validated_request["tool_id"],
        "granted_max_cost_usd": float(granted_max_cost_usd),
        "granted_ttl_seconds": granted_ttl_seconds,
        "granted_retry_count": granted_retry_count,
        "immutable_input_digests": list(validated_request["immutable_input_digests"]),
        "granted_provider_ids": granted_providers,
        "granted_action_ids": granted_actions,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "issued_by_agent": False,
        "proof_effect": "none",
    }
    return validate_authorization_receipt(
        _finalize(value, digest_field="authorization_receipt_digest"),
        request=validated_request,
    )


def validate_authorization_receipt(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an operator receipt without turning it into agent authority."""

    required_fields = {
        "schema_version",
        "receipt_id",
        "run_id",
        "authorization_request_digest",
        "operator_id",
        "approved",
        "granted_tool_id",
        "granted_max_cost_usd",
        "granted_ttl_seconds",
        "granted_retry_count",
        "immutable_input_digests",
        "granted_provider_ids",
        "granted_action_ids",
        "issued_at",
        "expires_at",
        "issued_by_agent",
        "proof_effect",
        "authorization_receipt_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("authorization_receipt_fields_invalid")
    run_id = str(value.get("run_id") or "").strip()
    tool_id = str(value.get("granted_tool_id") or "").strip()
    cost = value.get("granted_max_cost_usd")
    ttl = value.get("granted_ttl_seconds")
    retries = value.get("granted_retry_count")
    raw_digests = value.get("immutable_input_digests")
    if not isinstance(raw_digests, list):
        raise Phase2ArtifactError("authorization_receipt_input_digests_invalid")
    digests = sorted(
        {_require_digest(item, field="immutable_input_digest") for item in raw_digests}
    )
    providers = _strings(
        value.get("granted_provider_ids"),
        field="granted_provider_ids",
        minimum=0,
        maximum=20,
        item_maximum=100,
    )
    actions = _strings(
        value.get("granted_action_ids"),
        field="granted_action_ids",
        minimum=0,
        maximum=50,
        item_maximum=100,
    )
    issued = _parse_time(value.get("issued_at"), field="authorization_receipt_issued_at")
    expires = _parse_time(value.get("expires_at"), field="authorization_receipt_expires_at")
    approved = value.get("approved")
    expected = canonical_digest(value, digest_field="authorization_receipt_digest")
    _require_digest(
        value.get("authorization_request_digest"),
        field="authorization_request_digest",
    )
    if (
        value.get("schema_version") != AUTHORIZATION_RECEIPT_SCHEMA_VERSION
        or not run_id
        or not tool_id
        or not str(value.get("operator_id") or "").strip()
        or value.get("receipt_id") != f"{run_id}-{tool_id}-authorization-receipt"
        or not isinstance(approved, bool)
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
        or isinstance(ttl, bool)
        or not isinstance(ttl, int)
        or ttl < 1
        or isinstance(retries, bool)
        or not isinstance(retries, int)
        or retries < 0
        or not digests
        or raw_digests != digests
        or list(value.get("granted_provider_ids") or []) != providers
        or list(value.get("granted_action_ids") or []) != actions
        or expires <= issued
        or (expires - issued).total_seconds() > ttl
        or value.get("issued_by_agent") is not False
        or value.get("proof_effect") != "none"
        or value.get("authorization_receipt_digest") != expected
        or (approved is False and (float(cost) != 0 or retries != 0 or providers or actions))
        or (approved is True and tool_id == "execute_preauthorized_recovery" and not providers)
        or (approved is True and tool_id == "execute_preauthorized_recovery" and not actions)
    ):
        raise Phase2ArtifactError("authorization_receipt_contract_invalid")
    if request is not None:
        validated_request = validate_authorization_request(request)
        if (
            value.get("run_id") != validated_request["run_id"]
            or value.get("granted_tool_id") != validated_request["tool_id"]
            or value.get("authorization_request_digest")
            != validated_request["authorization_request_digest"]
            or value.get("receipt_id") != f"{validated_request['request_id']}-receipt"
            or float(cost) > float(validated_request["requested_max_cost_usd"])
            or ttl > int(validated_request["requested_ttl_seconds"])
            or retries > int(validated_request["requested_retry_count"])
            or digests != validated_request["immutable_input_digests"]
            or not set(providers).issubset(validated_request["requested_provider_ids"])
            or not set(actions).issubset(validated_request["requested_action_ids"])
        ):
            raise Phase2ArtifactError("authorization_receipt_exceeds_request")
    return dict(value)


def validate_targeted_recapture_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact fail-closed request accepted by recapture ingress."""

    required_fields = {
        "schema_version",
        "request_id",
        "run_id",
        "source_digest",
        "source_type",
        "missing_evidence",
        "requested_scope",
        "full_site_recapture_requested",
        "status",
        "capture_started",
        "rights_clearance_inferred",
        "raw_capture_mutated",
        "authoritative",
        "proof_effect",
        "targeted_recapture_request_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("targeted_recapture_request_fields_invalid")
    expected = canonical_digest(value, digest_field="targeted_recapture_request_digest")
    missing_evidence = _strings(
        value.get("missing_evidence"),
        field="missing_evidence",
        maximum=50,
        item_maximum=200,
    )
    run_id = str(value.get("run_id") or "").strip()
    if (
        value.get("schema_version") != TARGETED_RECAPTURE_REQUEST_SCHEMA_VERSION
        or value.get("targeted_recapture_request_digest") != expected
        or not run_id
        or value.get("request_id") != f"{run_id}-targeted-recapture"
        or value.get("source_type") not in {"capture_build", "site_task_testbed"}
        or value.get("requested_scope") != "targeted_only"
        or value.get("full_site_recapture_requested") is not False
        or value.get("status") != "proposed_for_review"
        or value.get("capture_started") is not False
        or value.get("rights_clearance_inferred") is not False
        or value.get("raw_capture_mutated") is not False
        or value.get("authoritative") is not False
        or value.get("proof_effect") != "none"
        or list(value.get("missing_evidence") or []) != missing_evidence
    ):
        raise Phase2ArtifactError("targeted_recapture_request_contract_invalid")
    _require_digest(value.get("source_digest"), field="source_digest")
    return dict(value)


def targeted_recapture_request(
    *,
    run_id: str,
    source_digest: str,
    source_type: str,
    missing_evidence: Sequence[str],
) -> dict[str, Any]:
    if not run_id.strip() or source_type not in {"capture_build", "site_task_testbed"}:
        raise Phase2ArtifactError("targeted_recapture_request_identity_invalid")
    normalized_missing = _strings(
        list(missing_evidence),
        field="missing_evidence",
        maximum=50,
        item_maximum=200,
    )
    value = {
        "schema_version": TARGETED_RECAPTURE_REQUEST_SCHEMA_VERSION,
        "request_id": f"{run_id}-targeted-recapture",
        "run_id": run_id,
        "source_digest": _require_digest(source_digest, field="source_digest"),
        "source_type": source_type,
        "missing_evidence": normalized_missing,
        "requested_scope": "targeted_only",
        "full_site_recapture_requested": False,
        "status": "proposed_for_review",
        "capture_started": False,
        "rights_clearance_inferred": False,
        "raw_capture_mutated": False,
        "authoritative": False,
        "proof_effect": "none",
    }
    return _finalize(value, digest_field="targeted_recapture_request_digest")


def validate_targeted_recapture_receipt(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
    capture_build: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a customer-bound recapture submission without declaring resolution."""

    required_fields = {
        "schema_version",
        "receipt_id",
        "run_id",
        "targeted_recapture_request_digest",
        "source_digest",
        "recapture_build_digest",
        "submitted_by",
        "received_at",
        "requested_missing_evidence",
        "capture_build_projection_contract_validated",
        "accepted_as_authoritative_evidence",
        "original_blocker_resolution",
        "rights_clearance_inferred",
        "submitted_by_agent",
        "proof_effect",
        "targeted_recapture_receipt_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("targeted_recapture_receipt_fields_invalid")
    expected = canonical_digest(value, digest_field="targeted_recapture_receipt_digest")
    requested_missing = _strings(
        value.get("requested_missing_evidence"),
        field="requested_missing_evidence",
        maximum=50,
        item_maximum=200,
    )
    _parse_time(value.get("received_at"), field="targeted_recapture_received_at")
    if (
        value.get("schema_version") != TARGETED_RECAPTURE_RECEIPT_SCHEMA_VERSION
        or value.get("targeted_recapture_receipt_digest") != expected
        or not str(value.get("receipt_id") or "").strip()
        or not str(value.get("run_id") or "").strip()
        or not str(value.get("submitted_by") or "").strip()
        or list(value.get("requested_missing_evidence") or []) != requested_missing
        or value.get("capture_build_projection_contract_validated") is not True
        or value.get("accepted_as_authoritative_evidence") is not False
        or value.get("original_blocker_resolution") != "undetermined_pending_reinspection"
        or value.get("rights_clearance_inferred") is not False
        or value.get("submitted_by_agent") is not False
        or value.get("proof_effect") != "none"
    ):
        raise Phase2ArtifactError("targeted_recapture_receipt_contract_invalid")
    request_digest = _require_digest(
        value.get("targeted_recapture_request_digest"),
        field="targeted_recapture_request_digest",
    )
    source_digest = _require_digest(value.get("source_digest"), field="source_digest")
    recapture_digest = _require_digest(
        value.get("recapture_build_digest"), field="recapture_build_digest"
    )
    expected_receipt_id = (
        f"{value['run_id']}-targeted-recapture-receipt-"
        f"{recapture_digest.removeprefix('sha256:')[:16]}"
    )
    if value.get("receipt_id") != expected_receipt_id:
        raise Phase2ArtifactError("targeted_recapture_receipt_identity_invalid")
    if request is not None:
        validated_request = validate_targeted_recapture_request(request)
        if (
            request_digest != validated_request["targeted_recapture_request_digest"]
            or value.get("run_id") != validated_request["run_id"]
            or source_digest != validated_request["source_digest"]
            or requested_missing != validated_request["missing_evidence"]
        ):
            raise Phase2ArtifactError("targeted_recapture_receipt_request_mismatch")
        if (
            validated_request["source_type"] == "capture_build"
            and recapture_digest == source_digest
        ):
            raise Phase2ArtifactError("targeted_recapture_receipt_capture_unchanged")
    if capture_build is not None:
        try:
            validated_capture = validate_capture_build_ingress(capture_build)
        except CaptureBuildIngressError as exc:
            raise Phase2ArtifactError("targeted_recapture_receipt_capture_invalid") from exc
        capture_digest = str(validated_capture["capture_build_digest"])
        if recapture_digest != capture_digest:
            raise Phase2ArtifactError("targeted_recapture_receipt_capture_mismatch")
    return dict(value)


def targeted_recapture_receipt(
    *,
    request: Mapping[str, Any],
    capture_build: Mapping[str, Any],
    submitted_by: str,
    received_at: str,
) -> dict[str, Any]:
    validated_request = validate_targeted_recapture_request(request)
    try:
        validated_capture = validate_capture_build_ingress(capture_build)
    except CaptureBuildIngressError as exc:
        raise Phase2ArtifactError("targeted_recapture_capture_build_invalid") from exc
    capture_digest = str(validated_capture["capture_build_digest"])
    if (
        validated_request["source_type"] == "capture_build"
        and validated_request["source_digest"] == capture_digest
    ):
        raise Phase2ArtifactError("targeted_recapture_receipt_capture_unchanged")
    if not submitted_by.strip():
        raise Phase2ArtifactError("targeted_recapture_submitter_missing")
    _parse_time(received_at, field="targeted_recapture_received_at")
    value = {
        "schema_version": TARGETED_RECAPTURE_RECEIPT_SCHEMA_VERSION,
        "receipt_id": (
            f"{validated_request['request_id']}-receipt-"
            f"{capture_digest.removeprefix('sha256:')[:16]}"
        ),
        "run_id": validated_request["run_id"],
        "targeted_recapture_request_digest": validated_request["targeted_recapture_request_digest"],
        "source_digest": validated_request["source_digest"],
        "recapture_build_digest": capture_digest,
        "submitted_by": submitted_by.strip(),
        "received_at": received_at,
        "requested_missing_evidence": list(validated_request["missing_evidence"]),
        "capture_build_projection_contract_validated": True,
        "accepted_as_authoritative_evidence": False,
        "original_blocker_resolution": "undetermined_pending_reinspection",
        "rights_clearance_inferred": False,
        "submitted_by_agent": False,
        "proof_effect": "none",
    }
    receipt = _finalize(value, digest_field="targeted_recapture_receipt_digest")
    return validate_targeted_recapture_receipt(
        receipt,
        request=validated_request,
        capture_build=capture_build,
    )


def validate_recapture_reinspection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a deterministic recapture-to-testbed gap-resolution artifact."""

    required_fields = {
        "schema_version",
        "run_id",
        "source_run_id",
        "targeted_recapture_request_digest",
        "targeted_recapture_receipt_digest",
        "recapture_build_digest",
        "testbed_digest",
        "testbed_capture_build_digest",
        "testbed_bound_to_recapture",
        "predecessor_binding_required",
        "predecessor_testbed_bound",
        "requested_missing_evidence",
        "resolved_missing_evidence",
        "unresolved_missing_evidence",
        "coverage_evidence_ids",
        "coverage_source_artifact_digests",
        "status",
        "accepted_as_authoritative_evidence",
        "rights_clearance_inferred",
        "proof_effect",
        "recapture_reinspection_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("recapture_reinspection_fields_invalid")
    expected = canonical_digest(value, digest_field="recapture_reinspection_digest")
    requested = _strings(
        value.get("requested_missing_evidence"),
        field="requested_missing_evidence",
        maximum=50,
        item_maximum=200,
    )
    resolved = _strings(
        value.get("resolved_missing_evidence"),
        field="resolved_missing_evidence",
        minimum=0,
        maximum=50,
        item_maximum=200,
    )
    unresolved = _strings(
        value.get("unresolved_missing_evidence"),
        field="unresolved_missing_evidence",
        minimum=0,
        maximum=50,
        item_maximum=200,
    )
    coverage_ids = _strings(
        value.get("coverage_evidence_ids"),
        field="coverage_evidence_ids",
        minimum=0,
        maximum=200,
        item_maximum=200,
    )
    coverage_source_digests = _strings(
        value.get("coverage_source_artifact_digests"),
        field="coverage_source_artifact_digests",
        minimum=0,
        maximum=200,
        item_maximum=80,
    )
    for digest in coverage_source_digests:
        _require_digest(digest, field="coverage_source_artifact_digest")
    testbed_capture_digest = value.get("testbed_capture_build_digest")
    if testbed_capture_digest is not None:
        _require_digest(testbed_capture_digest, field="testbed_capture_build_digest")
    capture_bound = value.get("testbed_bound_to_recapture") is True
    predecessor_required = value.get("predecessor_binding_required") is True
    predecessor_bound = value.get("predecessor_testbed_bound")
    if (
        not isinstance(value.get("testbed_bound_to_recapture"), bool)
        or not isinstance(value.get("predecessor_binding_required"), bool)
        or (predecessor_bound is not None and not isinstance(predecessor_bound, bool))
    ):
        raise Phase2ArtifactError("recapture_reinspection_predecessor_binding_invalid")
    expected_status = (
        "blocked_testbed_not_bound_to_recapture"
        if not capture_bound
        else (
            "blocked_testbed_lineage_mismatch"
            if predecessor_required and predecessor_bound is not True
            else (
                "unresolved_missing_evidence"
                if unresolved
                else "resolved_by_deterministic_testbed_reinspection"
            )
        )
    )
    if (
        value.get("schema_version") != RECAPTURE_REINSPECTION_SCHEMA_VERSION
        or value.get("recapture_reinspection_digest") != expected
        or not str(value.get("run_id") or "").strip()
        or not str(value.get("source_run_id") or "").strip()
        or sorted(resolved + unresolved) != requested
        or set(resolved) & set(unresolved)
        or list(value.get("requested_missing_evidence") or []) != requested
        or list(value.get("resolved_missing_evidence") or []) != resolved
        or list(value.get("unresolved_missing_evidence") or []) != unresolved
        or list(value.get("coverage_evidence_ids") or []) != coverage_ids
        or list(value.get("coverage_source_artifact_digests") or []) != coverage_source_digests
        or capture_bound != (testbed_capture_digest == value.get("recapture_build_digest"))
        or (not predecessor_required and predecessor_bound is not None)
        or value.get("status") != expected_status
        or value.get("accepted_as_authoritative_evidence") is not False
        or value.get("rights_clearance_inferred") is not False
        or value.get("proof_effect") != "none"
    ):
        raise Phase2ArtifactError("recapture_reinspection_contract_invalid")
    for field in (
        "targeted_recapture_request_digest",
        "targeted_recapture_receipt_digest",
        "recapture_build_digest",
        "testbed_digest",
    ):
        _require_digest(value.get(field), field=field)
    return dict(value)


def recapture_reinspection(
    *,
    run_id: str,
    request: Mapping[str, Any],
    receipt: Mapping[str, Any],
    capture_build: Mapping[str, Any],
    testbed: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically decide whether a rebuilt testbed covers requested gaps."""

    if not run_id.strip():
        raise Phase2ArtifactError("recapture_reinspection_run_id_missing")
    validated_request = validate_targeted_recapture_request(request)
    validated_receipt = validate_targeted_recapture_receipt(
        receipt,
        request=validated_request,
        capture_build=capture_build,
    )
    validated_testbed = MaintainedSiteTaskTestbed.from_mapping(testbed).to_mapping()
    recapture_digest = str(validated_receipt["recapture_build_digest"])
    raw_testbed_capture_digest = (validated_testbed.get("validation_envelope") or {}).get(
        "capture_build_digest"
    )
    testbed_capture_digest = (
        str(raw_testbed_capture_digest)
        if _SHA256_DIGEST.fullmatch(str(raw_testbed_capture_digest or ""))
        else None
    )
    testbed_bound_to_recapture = testbed_capture_digest == recapture_digest
    predecessor_binding_required = validated_request["source_type"] == "site_task_testbed"
    predecessor_testbed_bound: bool | None = None
    if predecessor_binding_required:
        predecessor_testbed_bound = validated_testbed.get(
            "predecessor_testbed_digest"
        ) == validated_request["source_digest"] or validated_request["source_digest"] in set(
            validated_testbed.get("supersedes") or []
        )

    capture_artifact_digests = {
        str(row.get("sha256"))
        for row in capture_build.get("artifacts") or []
        if isinstance(row, Mapping)
    }
    coverage: dict[str, set[tuple[str, str]]] = {}
    for row in validated_testbed.get("evidence_inventory") or []:
        if not isinstance(row, Mapping):
            continue
        evidence_id = str(row.get("evidence_id") or "").strip()
        source_artifact_digest = str(row.get("source_capture_artifact_digest") or "")
        if not evidence_id or source_artifact_digest not in capture_artifact_digests:
            continue
        requirements = {evidence_id}
        addresses = row.get("addresses_recapture_requirements")
        if isinstance(addresses, list):
            requirements.update(
                str(item).strip()
                for item in addresses
                if isinstance(item, str) and str(item).strip()
            )
        for requirement in requirements:
            coverage.setdefault(requirement, set()).add((evidence_id, source_artifact_digest))
    requested = list(validated_request["missing_evidence"])
    resolved = sorted(requirement for requirement in requested if coverage.get(requirement))
    unresolved = sorted(set(requested) - set(resolved))
    coverage_ids = sorted(
        {
            evidence_id
            for requirement in resolved
            for evidence_id, _source_digest in coverage[requirement]
        }
    )
    coverage_source_digests = sorted(
        {
            source_digest
            for requirement in resolved
            for _evidence_id, source_digest in coverage[requirement]
        }
    )
    status = (
        "blocked_testbed_not_bound_to_recapture"
        if not testbed_bound_to_recapture
        else (
            "blocked_testbed_lineage_mismatch"
            if predecessor_binding_required and predecessor_testbed_bound is not True
            else (
                "unresolved_missing_evidence"
                if unresolved
                else "resolved_by_deterministic_testbed_reinspection"
            )
        )
    )
    value = {
        "schema_version": RECAPTURE_REINSPECTION_SCHEMA_VERSION,
        "run_id": run_id.strip(),
        "source_run_id": validated_receipt["run_id"],
        "targeted_recapture_request_digest": validated_request["targeted_recapture_request_digest"],
        "targeted_recapture_receipt_digest": validated_receipt["targeted_recapture_receipt_digest"],
        "recapture_build_digest": recapture_digest,
        "testbed_digest": validated_testbed["testbed_digest"],
        "testbed_capture_build_digest": testbed_capture_digest,
        "testbed_bound_to_recapture": testbed_bound_to_recapture,
        "predecessor_binding_required": predecessor_binding_required,
        "predecessor_testbed_bound": predecessor_testbed_bound,
        "requested_missing_evidence": requested,
        "resolved_missing_evidence": resolved,
        "unresolved_missing_evidence": unresolved,
        "coverage_evidence_ids": coverage_ids,
        "coverage_source_artifact_digests": coverage_source_digests,
        "status": status,
        "accepted_as_authoritative_evidence": False,
        "rights_clearance_inferred": False,
        "proof_effect": "none",
    }
    return validate_recapture_reinspection(
        _finalize(value, digest_field="recapture_reinspection_digest")
    )


def scenario_proposal_set(
    *,
    run_id: str,
    request_digest: str,
    scenarios: Sequence[Mapping[str, Any]],
    candidate_results_observed: bool,
) -> dict[str, Any]:
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise Phase2ArtifactError("scenario_run_id_missing")
    if candidate_results_observed:
        raise Phase2ArtifactError("post_result_scenario_generation_forbidden")
    if not scenarios or len(scenarios) > 100:
        raise Phase2ArtifactError("scenario_count_out_of_range")
    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
    for row in scenarios:
        if not isinstance(row, Mapping):
            raise Phase2ArtifactError("scenario_not_mapping")
        scenario_id = str(row.get("scenario_id") or "").strip()
        failure_mode = str(row.get("failure_mode") or "").strip()
        description = str(row.get("description") or "").strip()
        if not scenario_id or not failure_mode or not description:
            raise Phase2ArtifactError("scenario_missing_fields")
        if scenario_id in identities:
            raise Phase2ArtifactError("scenario_id_duplicate")
        identities.add(scenario_id)
        normalized.append(
            {
                "scenario_id": scenario_id,
                "failure_mode": failure_mode[:200],
                "description": description[:2_000],
                "success_predicate_proposed": row.get("success_predicate_proposed"),
                "hidden_label": None,
            }
        )
    value = {
        "schema_version": SCENARIO_PROPOSAL_SET_SCHEMA_VERSION,
        "proposal_set_id": f"{normalized_run_id}-scenario-proposals",
        "run_id": normalized_run_id,
        "request_digest": _require_digest(request_digest, field="request_digest"),
        "scenarios": sorted(normalized, key=lambda row: row["scenario_id"]),
        "candidate_results_observed": False,
        "frozen": False,
        "authoritative": False,
        "proof_effect": "none",
    }
    return validate_scenario_proposal_set(
        _finalize(value, digest_field="scenario_proposal_set_digest")
    )


def validate_scenario_proposal_set(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a pre-result proposal set without accepting hidden labels."""

    required_fields = {
        "schema_version",
        "proposal_set_id",
        "run_id",
        "request_digest",
        "scenarios",
        "candidate_results_observed",
        "frozen",
        "authoritative",
        "proof_effect",
        "scenario_proposal_set_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("scenario_proposal_set_fields_invalid")
    run_id = str(value.get("run_id") or "").strip()
    raw_scenarios = value.get("scenarios")
    if not isinstance(raw_scenarios, list) or not 1 <= len(raw_scenarios) <= 100:
        raise Phase2ArtifactError("scenario_count_out_of_range")
    normalized: list[dict[str, Any]] = []
    scenario_ids: set[str] = set()
    for row in raw_scenarios:
        if not isinstance(row, Mapping) or set(row) != {
            "scenario_id",
            "failure_mode",
            "description",
            "success_predicate_proposed",
            "hidden_label",
        }:
            raise Phase2ArtifactError("scenario_contract_invalid")
        scenario_id = str(row.get("scenario_id") or "").strip()
        failure_mode = str(row.get("failure_mode") or "").strip()
        description = str(row.get("description") or "").strip()
        if (
            not scenario_id
            or len(scenario_id) > 100
            or scenario_id in scenario_ids
            or not failure_mode
            or len(failure_mode) > 200
            or not description
            or len(description) > 2_000
            or row.get("hidden_label") is not None
        ):
            raise Phase2ArtifactError("scenario_contract_invalid")
        _validate_untrusted_response_json(row.get("success_predicate_proposed"))
        scenario_ids.add(scenario_id)
        normalized.append(dict(row))
    if raw_scenarios != sorted(normalized, key=lambda row: row["scenario_id"]):
        raise Phase2ArtifactError("scenario_order_invalid")
    expected = canonical_digest(value, digest_field="scenario_proposal_set_digest")
    if (
        value.get("schema_version") != SCENARIO_PROPOSAL_SET_SCHEMA_VERSION
        or not run_id
        or value.get("run_id") != run_id
        or value.get("proposal_set_id") != f"{run_id}-scenario-proposals"
        or value.get("candidate_results_observed") is not False
        or value.get("frozen") is not False
        or value.get("authoritative") is not False
        or value.get("proof_effect") != "none"
        or value.get("scenario_proposal_set_digest") != expected
    ):
        raise Phase2ArtifactError("scenario_proposal_set_contract_invalid")
    _require_digest(value.get("request_digest"), field="request_digest")
    return dict(value)


def freeze_scenario_manifest(
    *,
    proposal_set: Mapping[str, Any],
    authorization: Mapping[str, Any],
    evaluator_digest: str,
    success_predicate_digest: str,
    hidden_label_manifest_digest: str,
    frozen_at: str,
) -> dict[str, Any]:
    validated_proposal = validate_scenario_proposal_set(proposal_set)
    validated_authorization = validate_authorization_receipt(authorization)
    proposal_digest = validated_proposal["scenario_proposal_set_digest"]
    receipt_digest = validated_authorization["authorization_receipt_digest"]
    if (
        validated_authorization.get("approved") is not True
        or validated_authorization.get("issued_by_agent") is not False
    ):
        raise Phase2ArtifactError("scenario_freeze_not_operator_authorized")
    if validated_authorization.get("granted_tool_id") != "freeze_scenario_manifest":
        raise Phase2ArtifactError("scenario_freeze_wrong_authority")
    if validated_authorization.get("run_id") != validated_proposal.get("run_id"):
        raise Phase2ArtifactError("scenario_freeze_run_mismatch")
    if validated_authorization.get("immutable_input_digests") != [proposal_digest]:
        raise Phase2ArtifactError("scenario_freeze_input_not_authorized")
    frozen_time = _parse_time(frozen_at, field="scenario_frozen_at")
    issued = _parse_time(
        validated_authorization.get("issued_at"), field="scenario_authority_issued_at"
    )
    expires = _parse_time(
        validated_authorization.get("expires_at"), field="scenario_authority_expires_at"
    )
    if frozen_time < issued or frozen_time >= expires:
        raise Phase2ArtifactError("scenario_freeze_authority_inactive")
    if validated_proposal.get("candidate_results_observed") is not False:
        raise Phase2ArtifactError("post_result_scenario_freeze_forbidden")
    value = {
        "schema_version": FROZEN_SCENARIO_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"{validated_proposal['proposal_set_id']}-frozen",
        "run_id": validated_proposal["run_id"],
        "scenario_proposal_set_digest": proposal_digest,
        "scenario_ids": [row["scenario_id"] for row in validated_proposal["scenarios"]],
        "evaluator_digest": _require_digest(evaluator_digest, field="evaluator_digest"),
        "success_predicate_digest": _require_digest(
            success_predicate_digest, field="success_predicate_digest"
        ),
        "hidden_label_manifest_digest": _require_digest(
            hidden_label_manifest_digest, field="hidden_label_manifest_digest"
        ),
        "hidden_labels_included": False,
        "candidate_results_observed_before_freeze": False,
        "authorization_receipt_digest": receipt_digest,
        "frozen_at": frozen_at,
        "frozen": True,
        "proof_effect": "none",
    }
    return validate_frozen_scenario_manifest(
        _finalize(value, digest_field="frozen_scenario_manifest_digest")
    )


def validate_frozen_scenario_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable public side of a hidden-evaluation freeze."""

    required_fields = {
        "schema_version",
        "manifest_id",
        "run_id",
        "scenario_proposal_set_digest",
        "scenario_ids",
        "evaluator_digest",
        "success_predicate_digest",
        "hidden_label_manifest_digest",
        "hidden_labels_included",
        "candidate_results_observed_before_freeze",
        "authorization_receipt_digest",
        "frozen_at",
        "frozen",
        "proof_effect",
        "frozen_scenario_manifest_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("frozen_scenario_manifest_fields_invalid")
    run_id = str(value.get("run_id") or "").strip()
    scenario_ids = _strings(
        value.get("scenario_ids"),
        field="scenario_ids",
        maximum=100,
        item_maximum=100,
    )
    expected = canonical_digest(value, digest_field="frozen_scenario_manifest_digest")
    if (
        value.get("schema_version") != FROZEN_SCENARIO_MANIFEST_SCHEMA_VERSION
        or not run_id
        or value.get("run_id") != run_id
        or value.get("manifest_id") != f"{run_id}-scenario-proposals-frozen"
        or value.get("scenario_ids") != scenario_ids
        or value.get("hidden_labels_included") is not False
        or value.get("candidate_results_observed_before_freeze") is not False
        or value.get("frozen") is not True
        or value.get("proof_effect") != "none"
        or value.get("frozen_scenario_manifest_digest") != expected
    ):
        raise Phase2ArtifactError("frozen_scenario_manifest_contract_invalid")
    for field in (
        "scenario_proposal_set_digest",
        "evaluator_digest",
        "success_predicate_digest",
        "hidden_label_manifest_digest",
        "authorization_receipt_digest",
    ):
        _require_digest(value.get(field), field=field)
    _parse_time(value.get("frozen_at"), field="scenario_frozen_at")
    return dict(value)


def deterministic_customer_report(
    *,
    context: Any,
    capability_results: Sequence[Mapping[str, Any]],
    invocation_manifests: Sequence[Mapping[str, Any]],
    generated_artifact_references: Sequence[Mapping[str, Any]],
    tool_observations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    request = dict(context.decision_request or {})
    plan = dict(context.evidence_plan or {})
    decision = dict(context.decision_envelope or {})
    results = [dict(row) for row in context.evidence_results]
    claims = [dict(row) for row in request.get("claims") or [] if isinstance(row, Mapping)]
    claim_evidence: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = claim.get("claim_id")
        matching = [row for row in results if row.get("claim_id") == claim_id]
        evidence_sources = []
        for row in matching:
            profile = row.get("method_profile_snapshot") or {}
            evidence_sources.append(
                {
                    "result_digest": row.get("result_digest"),
                    "method_profile_digest": row.get("method_profile_digest"),
                    "method_id": profile.get("method_id"),
                    "method_version": profile.get("version"),
                    "method_family": profile.get("method_family"),
                    "authority_level": {
                        "authority_tier": profile.get("authority_tier"),
                        "proof_tier": profile.get("proof_tier"),
                        "self_qualified": profile.get("self_qualified"),
                    },
                    "claim_ceiling": dict(row.get("claim_ceiling") or {}),
                    "status": row.get("status"),
                    "validity": row.get("validity"),
                }
            )
        claim_evidence.append(
            {
                "claim_id": claim_id,
                "claim_type": claim.get("claim_type"),
                "evidence_result_digests": [row.get("result_digest") for row in matching],
                "evidence_statuses": [row.get("status") for row in matching],
                "evidence_sources": evidence_sources,
                "accepted_by_agent": False,
            }
        )
    methods_attempted = sorted(
        {
            str((row.get("method_profile_snapshot") or {}).get("method_id"))
            for row in results
            if isinstance(row.get("method_profile_snapshot"), Mapping)
            and (row.get("method_profile_snapshot") or {}).get("method_id")
        }
    )
    failed_methods = [
        {
            "result_digest": row.get("result_digest"),
            "status": row.get("status"),
            "failure_type": row.get("failure_type"),
        }
        for row in results
        if row.get("status") not in {"valid", "accepted", "sufficient"}
    ]
    proposals = [
        dict(proposal)
        for result in capability_results
        for proposal in result.get("proposals") or []
        if isinstance(proposal, Mapping)
    ]
    outcome = decision.get("overall_outcome") or "abstention"
    value = {
        "schema_version": CUSTOMER_REPORT_SCHEMA_VERSION,
        "run_id": context.run_id,
        "customer_original_question": context.customer_question,
        "validated_interpretation": request.get("decision_question"),
        "accepted_leaf_claims": [row.get("claim_id") for row in claims],
        "rejected_or_unresolved_interpretations": (
            [] if request else ["validated_decision_request_missing"]
        ),
        "claim_evidence": claim_evidence,
        "methods_attempted": methods_attempted,
        "failed_methods": failed_methods,
        "skipped_methods": [
            row
            for row in plan.get("claim_plans") or []
            if isinstance(row, Mapping) and row.get("status") != "planned"
        ],
        "agent_actions_and_recommendations": proposals,
        "agent_output_authoritative": False,
        "deterministic_validations": {
            "request_digest": request.get("request_digest"),
            "testbed_digest": (context.testbed or {}).get("testbed_digest"),
            "plan_digest": plan.get("plan_digest"),
            "decision_envelope_digest": decision.get("decision_envelope_digest"),
        },
        "spending_and_runtime": {
            "reported_agent_cost_usd": sum(
                float(row.get("cost_usd") or 0.0) for row in invocation_manifests
            ),
            "reserved_agent_cost_ceiling_usd": max(
                [
                    float((row.get("budget_state") or {}).get("cumulative_reserved_cost_usd") or 0.0)
                    for row in invocation_manifests
                ],
                default=0.0,
            ),
            "invocation_count": len(invocation_manifests),
            "reported_agent_duration_seconds": sum(
                float(row.get("latency_seconds") or 0.0) for row in invocation_manifests
            ),
            "reported_action_cost_usd": sum(
                float(row.get("cost_usd") or 0.0) for row in tool_observations
            ),
            "reported_action_duration_seconds": sum(
                float(row.get("duration_seconds") or 0.0) for row in tool_observations
            ),
            "tool_observation_count": len(tool_observations),
        },
        "decision": outcome,
        "partial_decision": outcome == "partial_decision",
        "abstention": outcome == "abstention",
        "uncertainty_and_evidence_ceiling": decision.get("claim_ceiling"),
        "next_experiments": [
            row.get("next_cheapest_experiment")
            for row in plan.get("claim_plans") or []
            if isinstance(row, Mapping) and row.get("next_cheapest_experiment")
        ],
        "generated_artifact_references": [dict(row) for row in generated_artifact_references],
        "blueprint_cannot_claim": sorted(
            {
                "agent_output_is_proof",
                "provider_completion_proves_scientific_claim",
                "simulation_proves_physical_success",
                "deployment_readiness_without_qualified_evidence",
                *[str(item) for item in decision.get("prohibited_claims") or []],
            }
        ),
        "proof_state_mutated_by_report": False,
    }
    return validate_customer_report(_finalize(value, digest_field="customer_report_digest"))


def validate_customer_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the customer-facing projection without granting it proof authority."""

    required_fields = {
        "schema_version",
        "run_id",
        "customer_original_question",
        "validated_interpretation",
        "accepted_leaf_claims",
        "rejected_or_unresolved_interpretations",
        "claim_evidence",
        "methods_attempted",
        "failed_methods",
        "skipped_methods",
        "agent_actions_and_recommendations",
        "agent_output_authoritative",
        "deterministic_validations",
        "spending_and_runtime",
        "decision",
        "partial_decision",
        "abstention",
        "uncertainty_and_evidence_ceiling",
        "next_experiments",
        "generated_artifact_references",
        "blueprint_cannot_claim",
        "proof_state_mutated_by_report",
        "customer_report_digest",
    }
    if set(value) != required_fields:
        raise Phase2ArtifactError("customer_report_fields_invalid")
    _validate_untrusted_response_json(value)
    try:
        serialized = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise Phase2ArtifactError("customer_report_value_invalid") from exc
    if len(serialized.encode("utf-8")) > 2_000_000:
        raise Phase2ArtifactError("customer_report_too_large")
    decision = value.get("decision")
    spending = value.get("spending_and_runtime")
    validations = value.get("deterministic_validations")
    cannot_claim = value.get("blueprint_cannot_claim")
    required_prohibitions = {
        "agent_output_is_proof",
        "provider_completion_proves_scientific_claim",
        "simulation_proves_physical_success",
        "deployment_readiness_without_qualified_evidence",
    }
    numeric_fields = (
        "reported_agent_cost_usd",
        "reserved_agent_cost_ceiling_usd",
        "invocation_count",
        "reported_agent_duration_seconds",
        "reported_action_cost_usd",
        "reported_action_duration_seconds",
        "tool_observation_count",
    )
    expected = canonical_digest(value, digest_field="customer_report_digest")
    if (
        value.get("schema_version") != CUSTOMER_REPORT_SCHEMA_VERSION
        or not str(value.get("run_id") or "").strip()
        or not str(value.get("customer_original_question") or "").strip()
        or decision not in {"decision", "partial_decision", "abstention"}
        or value.get("partial_decision") is not (decision == "partial_decision")
        or value.get("abstention") is not (decision == "abstention")
        or value.get("agent_output_authoritative") is not False
        or value.get("proof_state_mutated_by_report") is not False
        or not isinstance(spending, Mapping)
        or set(spending) != set(numeric_fields)
        or any(not _is_nonnegative_finite_number(spending.get(field)) for field in numeric_fields)
        or not isinstance(validations, Mapping)
        or set(validations)
        != {"request_digest", "testbed_digest", "plan_digest", "decision_envelope_digest"}
        or not isinstance(cannot_claim, list)
        or any(not isinstance(item, str) or not item for item in cannot_claim)
        or cannot_claim != sorted(set(cannot_claim))
        or not required_prohibitions.issubset(set(cannot_claim))
        or value.get("customer_report_digest") != expected
    ):
        raise Phase2ArtifactError("customer_report_contract_invalid")
    for field in (
        "accepted_leaf_claims",
        "rejected_or_unresolved_interpretations",
        "claim_evidence",
        "methods_attempted",
        "failed_methods",
        "skipped_methods",
        "agent_actions_and_recommendations",
        "next_experiments",
        "generated_artifact_references",
    ):
        if not isinstance(value.get(field), list):
            raise Phase2ArtifactError("customer_report_contract_invalid")
    for claim in value.get("claim_evidence") or []:
        if not isinstance(claim, Mapping) or set(claim) != {
            "claim_id",
            "claim_type",
            "evidence_result_digests",
            "evidence_statuses",
            "evidence_sources",
            "accepted_by_agent",
        }:
            raise Phase2ArtifactError("customer_report_claim_evidence_invalid")
        sources = claim.get("evidence_sources")
        if (
            claim.get("accepted_by_agent") is not False
            or not isinstance(sources, list)
            or claim.get("evidence_result_digests")
            != [source.get("result_digest") for source in sources if isinstance(source, Mapping)]
            or claim.get("evidence_statuses")
            != [source.get("status") for source in sources if isinstance(source, Mapping)]
        ):
            raise Phase2ArtifactError("customer_report_claim_evidence_invalid")
        for source in sources:
            if not isinstance(source, Mapping) or set(source) != {
                "result_digest",
                "method_profile_digest",
                "method_id",
                "method_version",
                "method_family",
                "authority_level",
                "claim_ceiling",
                "status",
                "validity",
            }:
                raise Phase2ArtifactError("customer_report_evidence_source_invalid")
            authority = source.get("authority_level")
            if (
                not isinstance(authority, Mapping)
                or set(authority) != {"authority_tier", "proof_tier", "self_qualified"}
                or not _is_nonnegative_finite_number(authority.get("authority_tier"))
                or not str(authority.get("proof_tier") or "").strip()
                or authority.get("self_qualified") is not False
                or any(
                    not str(source.get(field) or "").strip()
                    for field in ("method_id", "method_version", "method_family")
                )
                or not isinstance(source.get("claim_ceiling"), Mapping)
                or source.get("status")
                not in {
                    "valid",
                    "invalid",
                    "uncertain",
                    "contradictory",
                    "unavailable",
                    "evidence_requested",
                }
                or not isinstance(source.get("validity"), bool)
            ):
                raise Phase2ArtifactError("customer_report_evidence_source_invalid")
            _require_digest(source.get("result_digest"), field="evidence_source_result_digest")
            _require_digest(
                source.get("method_profile_digest"),
                field="evidence_source_method_profile_digest",
            )
    return dict(value)


def write_phase2_artifact(
    output_dir: str | Path,
    relative_path: str,
    value: Mapping[str, Any],
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents:
        raise Phase2ArtifactError("phase2_artifact_path_escape")
    write_json(path, value)
    return path


__all__ = [
    "AUTHORIZATION_RECEIPT_SCHEMA_VERSION",
    "AUTHORIZATION_REQUEST_SCHEMA_VERSION",
    "CLARIFICATION_RECEIPT_SCHEMA_VERSION",
    "CLARIFICATION_REQUEST_SCHEMA_VERSION",
    "CUSTOMER_REPORT_SCHEMA_VERSION",
    "FROZEN_SCENARIO_MANIFEST_SCHEMA_VERSION",
    "Phase2ArtifactError",
    "RECAPTURE_REINSPECTION_SCHEMA_VERSION",
    "SCENARIO_PROPOSAL_SET_SCHEMA_VERSION",
    "TARGETED_RECAPTURE_RECEIPT_SCHEMA_VERSION",
    "TARGETED_RECAPTURE_REQUEST_SCHEMA_VERSION",
    "authorization_receipt",
    "authorization_request",
    "clarification_receipt",
    "clarification_request",
    "deterministic_customer_report",
    "freeze_scenario_manifest",
    "recapture_reinspection",
    "scenario_proposal_set",
    "targeted_recapture_receipt",
    "targeted_recapture_request",
    "validate_targeted_recapture_receipt",
    "validate_targeted_recapture_request",
    "validate_recapture_reinspection",
    "validate_clarification_receipt",
    "validate_clarification_request",
    "validate_authorization_receipt",
    "validate_authorization_request",
    "validate_customer_report",
    "validate_frozen_scenario_manifest",
    "validate_scenario_proposal_set",
    "write_phase2_artifact",
]
