"""Signed, receipt-bound publication of terminal Task Evaluation Runs."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import DecisionEnvelope, EvidencePlan, canonical_digest
from .task_evaluation_result_delivery import DELIVERY_SCHEMA_VERSION
from .task_evaluation_launch_webapp_sync import (
    PipelineSyncTokenError,
    load_pipeline_sync_token,
)
from .task_evaluation_policy_run_contract import (
    TaskEvaluationPolicyRunContractError,
    validate_policy_run_result_projection,
)
from .task_evaluation_policy_canary_result import (
    TaskEvaluationPolicyCanaryResultError,
    validate_policy_canary_result,
)
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


TASK_EVALUATION_RUN_WEBAPP_URL_ENV = "PIPELINE_TASK_EVALUATION_RUN_WEBAPP_URL"
TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV = (
    "PIPELINE_TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED"
)
_TERMINAL_STATES = {"decided", "partially_decided", "abstained"}
_CANARY_TERMINAL_STATES = {"completed_unqualified", "blocked", "cancelled"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _resolved_sync_token(token: str | None) -> str:
    try:
        return load_pipeline_sync_token(token=token)
    except PipelineSyncTokenError:
        return ""


def build_task_evaluation_run_webapp_publication(
    *,
    capture_session_id: str,
    intake_id: str,
    run_id: str,
    state: str,
    evidence_plan: Mapping[str, Any],
    decision_envelope: Mapping[str, Any],
    result_delivery: Mapping[str, Any] | None = None,
    policy_run_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session = strict_identifier(
        capture_session_id, field="capture_session_id", max_length=192
    )
    intake = strict_identifier(intake_id, field="intake_id", max_length=192)
    run = strict_identifier(run_id, field="run_id", max_length=192)
    if state not in _TERMINAL_STATES:
        raise ValueError("task_evaluation_run_state_not_terminal")
    plan = EvidencePlan.from_mapping(evidence_plan).to_mapping()
    envelope = DecisionEnvelope.from_mapping(decision_envelope).to_mapping()
    expected_state = {
        "decision": "decided",
        "partial_decision": "partially_decided",
        "abstention": "abstained",
    }[envelope["overall_outcome"]]
    if state != expected_state:
        raise ValueError("task_evaluation_run_state_outcome_mismatch")
    for field in ("request_digest", "plan_digest", "testbed_digest"):
        if plan[field] != envelope[field]:
            raise ValueError(f"task_evaluation_run_{field}_binding_mismatch")
    publication = {
        "schema_version": (
            "task_evaluation_run_publication.v3"
            if policy_run_result is not None
            else "task_evaluation_run_publication.v2"
            if result_delivery is not None
            else "task_evaluation_run_publication.v1"
        ),
        "capture_session_id": session,
        "intake_id": intake,
        "run_id": run,
        "testbed_digest": plan["testbed_digest"],
        "request_digest": plan["request_digest"],
        "plan_digest": plan["plan_digest"],
        "state": state,
        "evidence_plan": plan,
        "decision_envelope": envelope,
        "proof_boundary": {
            "simulation_is_physical_success": False,
            "deployment_or_safety_approved": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }
    if result_delivery is not None:
        delivery = dict(result_delivery)
        if delivery.get("schema_version") != DELIVERY_SCHEMA_VERSION:
            raise ValueError("task_evaluation_result_delivery_schema_invalid")
        if (
            delivery.get("run_id") != run
            or delivery.get("state") != state
            or delivery.get("decision_envelope_digest")
            != envelope["decision_envelope_digest"]
        ):
            raise ValueError("task_evaluation_result_delivery_binding_mismatch")
        if delivery.get("delivery_digest") != canonical_digest(
            delivery, digest_field="delivery_digest"
        ):
            raise ValueError("task_evaluation_result_delivery_digest_mismatch")
        publication["result_delivery"] = delivery
    if policy_run_result is not None:
        if result_delivery is None:
            raise ValueError("task_evaluation_policy_run_result_delivery_missing")
        try:
            policy_result = validate_policy_run_result_projection(
                policy_run_result
            )
        except TaskEvaluationPolicyRunContractError as exc:
            raise ValueError("task_evaluation_policy_run_result_invalid") from exc
        if (
            policy_result["run_id"] != run
            or policy_result["state"] != state
            or policy_result["result_delivery_digest"]
            != publication["result_delivery"]["delivery_digest"]
        ):
            raise ValueError("task_evaluation_policy_run_result_binding_mismatch")
        publication["policy_run_result"] = policy_result
    return publication


def build_task_evaluation_policy_canary_webapp_publication(
    *,
    capture_session_id: str,
    intake_id: str,
    run_id: str,
    request_digest: str,
    configuration_digest: str,
    result_status: str,
    result_delivery: Mapping[str, Any],
    policy_canary_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build additive v4 publication without inventing a decision envelope."""

    session = strict_identifier(
        capture_session_id, field="capture_session_id", max_length=192
    )
    intake = strict_identifier(intake_id, field="intake_id", max_length=192)
    run = strict_identifier(run_id, field="run_id", max_length=192)
    if result_status not in _CANARY_TERMINAL_STATES:
        raise ValueError("task_evaluation_policy_canary_state_not_terminal")
    delivery = dict(result_delivery)
    if (
        delivery.get("schema_version") != "task_evaluation_result_delivery.v2"
        or delivery.get("run_id") != run
        or delivery.get("result_status") != result_status
        or delivery.get("claim_ceiling") != "diagnostic_policy_execution"
        or delivery.get("delivery_digest")
        != canonical_digest(delivery, digest_field="delivery_digest")
    ):
        raise ValueError("task_evaluation_policy_canary_result_delivery_invalid")
    try:
        canary = validate_policy_canary_result(policy_canary_result)
    except TaskEvaluationPolicyCanaryResultError as exc:
        raise ValueError("task_evaluation_policy_canary_result_invalid") from exc
    if (
        canary["run_id"] != run
        or canary["request_digest"] != request_digest
        or canary["configuration_digest"] != configuration_digest
        or canary["result_status"] != result_status
        or canary["result_delivery_digest"] != delivery["delivery_digest"]
    ):
        raise ValueError("task_evaluation_policy_canary_result_binding_mismatch")
    return {
        "schema_version": "task_evaluation_run_publication.v4",
        "capture_session_id": session,
        "intake_id": intake,
        "run_id": run,
        "request_digest": request_digest,
        "configuration_digest": configuration_digest,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "result_status": result_status,
        "scene_controls_status": "configured_controls_pending",
        "warning": "Controls pending — results are unqualified.",
        "result_delivery": delivery,
        "policy_canary_result": canary,
        "proof_boundary": {
            "scene_promotion_authorized": False,
            "official_policy_ranking_authorized": False,
            "winner_selection_authorized": False,
            "simulation_is_physical_success": False,
            "deployment_or_safety_approved": False,
        },
    }


def sync_task_evaluation_run_to_webapp(
    *,
    capture_session_id: str,
    intake_id: str,
    run_id: str,
    state: str,
    evidence_plan: Mapping[str, Any],
    decision_envelope: Mapping[str, Any],
    result_delivery: Mapping[str, Any] | None = None,
    policy_run_result: Mapping[str, Any] | None = None,
    endpoint_url: str | None = None,
    token: str | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.0,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    payload = build_task_evaluation_run_webapp_publication(
        capture_session_id=capture_session_id,
        intake_id=intake_id,
        run_id=run_id,
        state=state,
        evidence_plan=evidence_plan,
        decision_envelope=decision_envelope,
        result_delivery=result_delivery,
        policy_run_result=policy_run_result,
    )
    resolved_url = _text(endpoint_url) or _text(
        os.getenv(TASK_EVALUATION_RUN_WEBAPP_URL_ENV)
    )
    resolved_token = _resolved_sync_token(token)
    common = {
        "schema_version": "task_evaluation_run_webapp_sync_result.v1",
        "capture_session_id": payload["capture_session_id"],
        "intake_id": payload["intake_id"],
        "run_id": payload["run_id"],
        "state": payload["state"],
        "testbed_digest": payload["testbed_digest"],
        "request_digest": payload["request_digest"],
        "plan_digest": payload["plan_digest"],
        "decision_envelope_digest": payload["decision_envelope"][
            "decision_envelope_digest"
        ],
        "proof_boundary": payload["proof_boundary"],
    }
    if payload.get("result_delivery"):
        common["result_delivery_digest"] = payload["result_delivery"][
            "delivery_digest"
        ]
    if payload.get("policy_run_result"):
        common["policy_run_projection_digest"] = payload["policy_run_result"][
            "projection_digest"
        ]
    if not resolved_url or not resolved_token:
        return {
            **common,
            "status": "skipped",
            "reason": "sync_not_configured",
            "attempts": 0,
        }
    resolved_url = validated_https_sync_url(resolved_url)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    attempts = max(1, min(int(max_attempts), 10))
    last_reason = "sync_unknown_failure"
    for attempt in range(1, attempts + 1):
        outbound = urllib_request.Request(
            resolved_url,
            data=body,
            headers=_pipeline_sync_headers(resolved_token, body),
            method="POST",
        )
        try:
            # URL structure is validated immediately above; Bandit cannot infer that guard.
            with urllib_request.urlopen(  # nosec B310
                outbound, timeout=max(0.1, timeout_seconds)
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            last_reason = f"http_error:{exc.code}"
        except urllib_error.URLError as exc:
            last_reason = f"url_error:{exc.reason}"
        except (TimeoutError, ValueError) as exc:
            last_reason = exc.__class__.__name__.lower()
        else:
            try:
                receipt = _mapping(json.loads(raw) if raw else {})
            except json.JSONDecodeError:
                last_reason = "invalid_json"
            else:
                matches = (
                    receipt.get("schema_version")
                    == "capture_task_evaluation_run_publication_receipt.v1"
                    and receipt.get("status") == payload["state"]
                    and isinstance(receipt.get("already_exists"), bool)
                    and all(
                        receipt.get(field) == common[field]
                        for field in (
                            "capture_session_id",
                            "intake_id",
                            "run_id",
                            "testbed_digest",
                            "request_digest",
                            "plan_digest",
                            "decision_envelope_digest",
                            "proof_boundary",
                        )
                    )
                    and (
                        "result_delivery_digest" not in common
                        or receipt.get("result_delivery_digest")
                        == common["result_delivery_digest"]
                    )
                    and (
                        "policy_run_projection_digest" not in common
                        or receipt.get("policy_run_projection_digest")
                        == common["policy_run_projection_digest"]
                    )
                )
                if matches:
                    return {**common, "status": "succeeded", "attempts": attempt, "response": receipt}
                last_reason = "response_binding_mismatch"
        if attempt < attempts and retry_delay_seconds > 0:
            time.sleep(min(float(retry_delay_seconds), 5.0))
    return {**common, "status": "failed", "reason": last_reason, "attempts": attempts}


def sync_task_evaluation_policy_canary_to_webapp(
    *,
    capture_session_id: str,
    intake_id: str,
    run_id: str,
    request_digest: str,
    configuration_digest: str,
    result_status: str,
    result_delivery: Mapping[str, Any],
    policy_canary_result: Mapping[str, Any],
    endpoint_url: str | None = None,
    token: str | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.0,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Publish a canary once and require Website-owned email delivery readback."""

    payload = build_task_evaluation_policy_canary_webapp_publication(
        capture_session_id=capture_session_id,
        intake_id=intake_id,
        run_id=run_id,
        request_digest=request_digest,
        configuration_digest=configuration_digest,
        result_status=result_status,
        result_delivery=result_delivery,
        policy_canary_result=policy_canary_result,
    )
    resolved_url = _text(endpoint_url) or _text(
        os.getenv(TASK_EVALUATION_RUN_WEBAPP_URL_ENV)
    )
    resolved_token = _resolved_sync_token(token)
    common = {
        "schema_version": "task_evaluation_policy_canary_webapp_sync_result.v1",
        "capture_session_id": payload["capture_session_id"],
        "intake_id": payload["intake_id"],
        "run_id": payload["run_id"],
        "request_digest": payload["request_digest"],
        "configuration_digest": payload["configuration_digest"],
        "result_status": payload["result_status"],
        "result_delivery_digest": payload["result_delivery"]["delivery_digest"],
        "policy_canary_projection_digest": payload["policy_canary_result"][
            "projection_digest"
        ],
    }
    if not resolved_url or not resolved_token:
        return {
            **common,
            "status": "skipped",
            "reason": "sync_not_configured",
            "attempts": 0,
        }
    resolved_url = validated_https_sync_url(resolved_url)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    attempts = max(1, min(int(max_attempts), 10))
    last_reason = "sync_unknown_failure"
    for attempt in range(1, attempts + 1):
        outbound = urllib_request.Request(
            resolved_url,
            data=body,
            headers=_pipeline_sync_headers(resolved_token, body),
            method="POST",
        )
        try:
            with urllib_request.urlopen(  # nosec B310 - URL validated above
                outbound, timeout=max(0.1, timeout_seconds)
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            last_reason = f"http_error:{exc.code}"
        except urllib_error.URLError as exc:
            last_reason = f"url_error:{exc.reason}"
        except (TimeoutError, ValueError) as exc:
            last_reason = exc.__class__.__name__.lower()
        else:
            try:
                receipt = _mapping(json.loads(raw) if raw else {})
            except json.JSONDecodeError:
                last_reason = "invalid_json"
            else:
                notification = _mapping(receipt.get("notification_delivery"))
                expected_terminal = {
                    "completed_unqualified": "completed",
                    "blocked": "blocked",
                    "cancelled": "cancelled",
                }[result_status]
                matches = (
                    receipt.get("schema_version")
                    in {
                        "capture_task_evaluation_policy_canary_publication_receipt.v1",
                        "capture_task_evaluation_run_publication_receipt.v1",
                    }
                    and receipt.get("status") == result_status
                    and isinstance(receipt.get("already_exists"), bool)
                    and all(
                        receipt.get(field) == common[field]
                        for field in (
                            "capture_session_id",
                            "intake_id",
                            "run_id",
                            "request_digest",
                            "configuration_digest",
                            "result_delivery_digest",
                            "policy_canary_projection_digest",
                        )
                    )
                    and notification.get("terminal_state") == expected_terminal
                    and notification.get("status")
                    in {"accepted", "delivered", "failed"}
                    and isinstance(notification.get("attempts"), int)
                    and not isinstance(notification.get("attempts"), bool)
                    and notification["attempts"] >= 1
                    and notification.get("run_result_digest")
                    == common["policy_canary_projection_digest"]
                )
                if matches:
                    return {
                        **common,
                        "status": "succeeded",
                        "attempts": attempt,
                        "notification_delivery": notification,
                        "response": receipt,
                    }
                last_reason = "response_binding_mismatch"
        if attempt < attempts and retry_delay_seconds > 0:
            time.sleep(min(float(retry_delay_seconds), 5.0))
    return {
        **common,
        "status": "failed",
        "reason": last_reason,
        "attempts": attempts,
    }


def sync_policy_canary_preprovider_blocked_to_webapp(
    *,
    activation_id: str,
    capture_session_id: str,
    intake_id: str,
    request_digest: str,
    blockers: list[str],
    endpoint_url: str | None = None,
    token: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Publish a no-allocation terminal blocker and require email readback."""

    payload = {
        "schema_version": "task_evaluation_policy_canary_preprovider_blocked.v1",
        "activation_id": strict_identifier(
            activation_id, field="activation_id", max_length=192
        ),
        "capture_session_id": strict_identifier(
            capture_session_id, field="capture_session_id", max_length=192
        ),
        "intake_id": strict_identifier(intake_id, field="intake_id", max_length=192),
        "request_digest": request_digest,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "result_status": "blocked",
        "provider_allocation_performed": False,
        "automatic_retry_performed": False,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "payload_digest": "",
    }
    payload["payload_digest"] = canonical_digest(
        payload, digest_field="payload_digest"
    )
    resolved_url = _text(endpoint_url) or _text(
        os.getenv(TASK_EVALUATION_RUN_WEBAPP_URL_ENV)
    )
    resolved_token = _resolved_sync_token(token)
    if not resolved_url or not resolved_token:
        return {"status": "skipped", "reason": "sync_not_configured", **payload}
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    outbound = urllib_request.Request(
        validated_https_sync_url(resolved_url),
        data=body,
        headers=_pipeline_sync_headers(resolved_token, body),
        method="POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - URL validated above
            outbound, timeout=max(0.1, timeout_seconds)
        ) as response:
            receipt = _mapping(json.loads(response.read().decode("utf-8")))
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "failed", "reason": type(exc).__name__, **payload}
    notification = _mapping(receipt.get("notification_delivery"))
    if (
        receipt.get("schema_version")
        != "capture_task_evaluation_policy_canary_blocked_receipt.v1"
        or receipt.get("status") != "blocked"
        or receipt.get("activation_id") != payload["activation_id"]
        or receipt.get("request_digest") != request_digest
        or receipt.get("payload_digest") != payload["payload_digest"]
        or notification.get("terminal_state") != "blocked"
        or notification.get("status")
        not in {"accepted", "delivered", "failed"}
        or notification.get("run_result_digest") != payload["payload_digest"]
    ):
        return {"status": "failed", "reason": "response_binding_mismatch", **payload}
    return {
        "status": "succeeded",
        "payload_digest": payload["payload_digest"],
        "notification_delivery": notification,
        "response": receipt,
    }


__all__ = [
    "TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV",
    "TASK_EVALUATION_RUN_WEBAPP_URL_ENV",
    "build_task_evaluation_run_webapp_publication",
    "build_task_evaluation_policy_canary_webapp_publication",
    "sync_task_evaluation_policy_canary_to_webapp",
    "sync_policy_canary_preprovider_blocked_to_webapp",
    "sync_task_evaluation_run_to_webapp",
]
