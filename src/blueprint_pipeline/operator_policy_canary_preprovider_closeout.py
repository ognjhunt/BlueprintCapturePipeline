"""Publish an operator canary refusal while preserving its original receipt bytes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("operator_preprovider_digest_invalid")
    return value


def _metadata_identifier(value: Any) -> str:
    # Match the Website registration grammar, including owner namespaces. These
    # values are metadata, never filesystem paths; shared path validation stays strict.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}", value):
        raise ValueError("operator_preprovider_metadata_identifier_invalid")
    return value


def build_operator_preprovider_publication(
    *, registration: Mapping[str, Any], closeout_bytes: bytes,
) -> dict[str, Any]:
    """Bind one original native receipt to its immutable Website registration."""
    if (
        registration.get("schema_version") != "task_evaluation_operator_policy_canary_registration.v1"
        or registration.get("registration_digest") != cross_runtime_canonical_digest(
            registration, digest_field="registration_digest"
        )
        or registration.get("run_kind") != "internal_policy_canary"
        or registration.get("claim_ceiling") != "diagnostic_policy_execution"
        or registration.get("firebase_tenant_id") is not None
    ):
        raise ValueError("operator_preprovider_registration_invalid")
    if not isinstance(closeout_bytes, bytes) or not 0 < len(closeout_bytes) <= 65536:
        raise ValueError("operator_preprovider_closeout_size_invalid")
    raw_json = closeout_bytes.decode("utf-8")
    receipt = json.loads(raw_json)
    if not isinstance(receipt, dict) or receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        raise ValueError("operator_preprovider_native_receipt_digest_mismatch")
    expected = {
        "schema_version": "operator_policy_no_allocation_closeout.v1",
        "run_id": registration.get("run_id"),
        "status": "blocked_without_provider_allocation",
        "watchdog_status": "cancelled_no_allocation",
        "vast_instance_ids": [],
    }
    flags = {
        "provider_allocation_performed": False, "provider_instance_charge_not_applicable": True,
        "provider_create_attempted": False, "vast_side_effects_may_have_occurred": False,
        "all_staged_objects_absent": True, "continuing_spend_from_this_run": False,
        "provider_zero_verified": True, "legacy_no_allocation_predicate_passed": False,
    }
    charge = receipt.get("provider_instance_charge_usd")
    if (
        set(receipt) != set(expected) | set(flags) | {
            "provider_instance_charge_usd", "inner_adapter_sha256",
            "provider_attempt_classification", "legacy_predicate_gap", "receipt_digest",
        }
        or any(receipt.get(key) != value for key, value in expected.items())
        or any(receipt.get(key) is not value for key, value in flags.items())
        or isinstance(charge, bool) or not isinstance(charge, (int, float)) or charge != 0
        or not isinstance(receipt.get("legacy_predicate_gap"), str)
        or not receipt["legacy_predicate_gap"].strip()
    ):
        raise ValueError("operator_preprovider_no_allocation_proof_invalid")
    _digest(receipt.get("inner_adapter_sha256"))
    attempt = receipt.get("provider_attempt_classification")
    if not isinstance(attempt, dict):
        raise ValueError("operator_preprovider_attempt_proof_invalid")
    if (
        set(attempt) != {
            "schema_version", "classification", "provider_bundle_started", "provider_entrypoint_started",
            "provider_output_returned", "scientific_attempt_consumed", "automatic_requeue_authorized",
            "automatic_requeue_executed", "maximum_automatic_requeues",
            "pre_execution_requeue_eligible_in_principle", "authority_required_for_next_provider_mutation",
            "blockers",
        }
        or attempt.get("schema_version") != "provider_attempt_classification.v1"
        or attempt.get("classification") != "pre_execution_provider_null"
        or any(attempt.get(key) is not False for key in (
            "provider_bundle_started", "provider_entrypoint_started", "provider_output_returned",
            "scientific_attempt_consumed", "automatic_requeue_authorized", "automatic_requeue_executed",
        ))
        or attempt.get("authority_required_for_next_provider_mutation") is not True
        or type(attempt.get("maximum_automatic_requeues")) is not int
        or attempt["maximum_automatic_requeues"] != 0
        or type(attempt.get("pre_execution_requeue_eligible_in_principle")) is not bool
    ):
        raise ValueError("operator_preprovider_attempt_proof_invalid")
    blockers = attempt.get("blockers")
    if not isinstance(blockers, list) or not 1 <= len(blockers) <= 128 or any(
        not isinstance(code, str) or not code.strip() for code in blockers
    ):
        raise ValueError("operator_preprovider_blockers_invalid")
    payload = {
        "schema_version": "task_evaluation_operator_policy_canary_preprovider_blocked.v1",
        **{key: _metadata_identifier(registration.get(key))
           for key in ("run_id", "capture_session_id", "intake_id", "team_namespace")},
        **{key: _digest(registration.get(key)) for key in ("request_digest", "configuration_digest")},
        "operator_registration_digest": _digest(registration.get("registration_digest")),
        "firebase_tenant_id": None,
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "result_status": "blocked", "provider_allocation_performed": False,
        "automatic_retry_performed": False, "blockers": blockers,
        "no_allocation_closeout": {
            "raw_json": raw_json, "size_bytes": len(closeout_bytes),
            "sha256": "sha256:" + hashlib.sha256(closeout_bytes).hexdigest(),
        },
    }
    payload["payload_digest"] = cross_runtime_canonical_digest(payload, digest_field="payload_digest")
    return payload


def sync_operator_preprovider_closeout(
    *, registration: Mapping[str, Any], closeout_bytes: bytes,
    endpoint_url: str, token: str | None = None, timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    """Send one signed attempt; require the exact terminal identity in its readback."""
    payload = build_operator_preprovider_publication(
        registration=registration, closeout_bytes=closeout_bytes,
    )
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib_request.Request(
        validated_https_sync_url(endpoint_url), data=body, method="POST",
        headers=_pipeline_sync_headers(load_pipeline_sync_token(token=token), body),
    )
    common = {"run_id": payload["run_id"], "payload_digest": payload["payload_digest"]}
    try:
        with urllib_request.urlopen(request, timeout=max(0.1, timeout_seconds)) as response:  # nosec B310
            receipt = json.loads(response.read(262144).decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, ValueError) as exc:
        return {**common, "status": "failed", "reason": type(exc).__name__}
    if not isinstance(receipt, dict) or (
        receipt.get("schema_version") != "capture_task_evaluation_operator_policy_canary_blocked_receipt.v1"
        or receipt.get("status") != "blocked"
        or receipt.get("provider_allocation_performed") is not False
        or any(receipt.get(key) != payload[key] for key in (
            "run_id", "capture_session_id", "intake_id", "operator_registration_digest",
            "request_digest", "configuration_digest", "payload_digest",
        ))
    ):
        return {**common, "status": "failed", "reason": "response_binding_mismatch"}
    notification = receipt.get("notification_delivery")
    if not isinstance(notification, dict) or (
        notification.get("terminal_state") != "blocked"
        or notification.get("run_result_digest") != payload["payload_digest"]
        or notification.get("status") not in {"accepted", "delivered", "failed"}
    ):
        return {**common, "status": "failed", "reason": "notification_binding_mismatch"}
    return {**common, "status": "succeeded", "response": receipt,
            "notification_delivery": notification}
