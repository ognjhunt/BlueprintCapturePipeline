"""Publish an immutable Capture QA result to the owning Blueprint-WebApp session."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from .decision_evidence_contracts import canonical_digest
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


CAPTURE_QA_WEBAPP_URL_ENV = "PIPELINE_CAPTURE_QA_WEBAPP_URL"
CAPTURE_QA_WEBAPP_SYNC_REQUIRED_ENV = "PIPELINE_CAPTURE_QA_WEBAPP_SYNC_REQUIRED"
_STATUS_STATE = {
    "accepted": "capture_accepted",
    "analysis_required": "validating",
    "recapture_required": "rejected_or_recapture_required",
    "rejected": "failed",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _verified_report(value: Mapping[str, Any]) -> dict[str, Any]:
    report = json.loads(json.dumps(value))
    errors: list[str] = []
    if report.get("schema_version") != "capture_qa_report.v1":
        errors.append("schema_version_mismatch")
    for field in ("intake_id", "capture_authority_profile"):
        if not _text(report.get(field)):
            errors.append(f"{field}_missing")
    for field in ("envelope_digest", "qa_report_digest"):
        value_text = _text(report.get(field))
        if not value_text.startswith("sha256:") or len(value_text) != 71:
            errors.append(f"{field}_invalid")
    expected_digest = canonical_digest(report, digest_field="qa_report_digest")
    if report.get("qa_report_digest") != expected_digest:
        errors.append("qa_report_digest_mismatch")
    status = _text(report.get("status"))
    if status not in _STATUS_STATE or report.get("state") != _STATUS_STATE.get(status):
        errors.append("status_state_mismatch")
    if not isinstance(report.get("checks"), list):
        errors.append("checks_invalid")
    recapture = report.get("recapture_plan")
    if not isinstance(recapture, list):
        errors.append("recapture_plan_invalid")
    elif status == "recapture_required" and not all(
        _text(_mapping(row).get("code"))
        and _text(_mapping(row).get("instruction"))
        and _text(_mapping(row).get("reason"))
        for row in recapture
    ):
        errors.append("recapture_instructions_invalid")
    if status == "accepted" and recapture:
        errors.append("accepted_capture_has_recapture_plan")
    ceiling = _mapping(report.get("claim_ceiling"))
    if any(
        ceiling.get(field) is not False
        for field in ("physical_task_success", "deployment_readiness", "safety_certification")
    ):
        errors.append("claim_ceiling_upgrade_forbidden")
    if report.get("comparative_policy_ranking_verdict") != "thesis_not_supported":
        errors.append("policy_ranking_verdict_mismatch")
    if errors:
        raise ValueError("capture_qa_report_invalid:" + ",".join(sorted(set(errors))))
    return report


def build_capture_qa_webapp_publication(
    *, capture_session_id: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    verified = _verified_report(report)
    session_id = _text(capture_session_id)
    if not session_id:
        raise ValueError("capture_session_id_required")
    return {
        "schema_version": "capture_qa_publication.v1",
        "capture_session_id": session_id,
        "intake_id": verified["intake_id"],
        "capture_authority_profile": verified["capture_authority_profile"],
        "envelope_digest": verified["envelope_digest"],
        "qa_report_digest": verified["qa_report_digest"],
        "status": verified["status"],
        "state": verified["state"],
        "report": verified,
        "proof_boundary": {
            "qa_is_task_success": False,
            "qa_is_physical_success": False,
            "deployment_or_safety_approved": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }


def sync_capture_qa_to_webapp(
    *,
    capture_session_id: str,
    report: Mapping[str, Any],
    endpoint_url: str | None = None,
    token: str | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.0,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    payload = build_capture_qa_webapp_publication(
        capture_session_id=capture_session_id,
        report=report,
    )
    resolved_url = _text(endpoint_url) or _text(os.getenv(CAPTURE_QA_WEBAPP_URL_ENV))
    resolved_token = _text(token) or _text(os.getenv("PIPELINE_SYNC_TOKEN"))
    common = {
        "schema_version": "capture_qa_webapp_sync_result.v1",
        "capture_session_id": payload["capture_session_id"],
        "intake_id": payload["intake_id"],
        "qa_report_digest": payload["qa_report_digest"],
        "status_value": payload["status"],
        "state": payload["state"],
        "proof_boundary": payload["proof_boundary"],
    }
    if not resolved_url or not resolved_token:
        return {**common, "status": "skipped", "reason": "sync_not_configured", "attempts": 0}
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
                    receipt.get("schema_version") == "capture_qa_publication_receipt.v1"
                    and isinstance(receipt.get("already_exists"), bool)
                    and all(
                        receipt.get(field) == payload[field]
                        for field in (
                            "capture_session_id",
                            "intake_id",
                            "capture_authority_profile",
                            "envelope_digest",
                            "qa_report_digest",
                            "status",
                            "state",
                            "proof_boundary",
                        )
                    )
                )
                if matches:
                    return {
                        **common,
                        "status": "succeeded",
                        "attempts": attempt,
                        "response": receipt,
                    }
                last_reason = "response_binding_mismatch"
        if attempt < attempts and retry_delay_seconds > 0:
            time.sleep(min(float(retry_delay_seconds), 5.0))
    return {**common, "status": "failed", "reason": last_reason, "attempts": attempts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-session-id", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--require-sync", action="store_true")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = sync_capture_qa_to_webapp(
        capture_session_id=args.capture_session_id,
        report=report,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "succeeded" or not args.require_sync else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAPTURE_QA_WEBAPP_SYNC_REQUIRED_ENV",
    "CAPTURE_QA_WEBAPP_URL_ENV",
    "build_capture_qa_webapp_publication",
    "sync_capture_qa_to_webapp",
]
