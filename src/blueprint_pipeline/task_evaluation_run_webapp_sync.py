"""Signed, receipt-bound publication of terminal Task Evaluation Runs."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import DecisionEnvelope, EvidencePlan
from .webapp_sync import _pipeline_sync_headers


TASK_EVALUATION_RUN_WEBAPP_URL_ENV = "PIPELINE_TASK_EVALUATION_RUN_WEBAPP_URL"
TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV = (
    "PIPELINE_TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED"
)
_TERMINAL_STATES = {"decided", "partially_decided", "abstained"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def build_task_evaluation_run_webapp_publication(
    *,
    capture_session_id: str,
    intake_id: str,
    run_id: str,
    state: str,
    evidence_plan: Mapping[str, Any],
    decision_envelope: Mapping[str, Any],
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
    return {
        "schema_version": "task_evaluation_run_publication.v1",
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


def sync_task_evaluation_run_to_webapp(
    *,
    capture_session_id: str,
    intake_id: str,
    run_id: str,
    state: str,
    evidence_plan: Mapping[str, Any],
    decision_envelope: Mapping[str, Any],
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
    )
    resolved_url = _text(endpoint_url) or _text(
        os.getenv(TASK_EVALUATION_RUN_WEBAPP_URL_ENV)
    )
    resolved_token = _text(token) or _text(os.getenv("PIPELINE_SYNC_TOKEN"))
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
    if not resolved_url or not resolved_token:
        return {
            **common,
            "status": "skipped",
            "reason": "sync_not_configured",
            "attempts": 0,
        }
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
            with urllib_request.urlopen(
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
                )
                if matches:
                    return {**common, "status": "succeeded", "attempts": attempt, "response": receipt}
                last_reason = "response_binding_mismatch"
        if attempt < attempts and retry_delay_seconds > 0:
            time.sleep(min(float(retry_delay_seconds), 5.0))
    return {**common, "status": "failed", "reason": last_reason, "attempts": attempts}


__all__ = [
    "TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV",
    "TASK_EVALUATION_RUN_WEBAPP_URL_ENV",
    "build_task_evaluation_run_webapp_publication",
    "sync_task_evaluation_run_to_webapp",
]
