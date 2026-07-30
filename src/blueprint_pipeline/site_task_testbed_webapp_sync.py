"""Signed, receipt-bound publication of immutable testbeds to Blueprint-WebApp."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

from .decision_evidence_contracts import (
    DecisionEvidenceRequest,
    DecisionEvidenceContractError,
    MaintainedSiteTaskTestbed,
)
from .webapp_sync import _pipeline_sync_headers


TESTBED_WEBAPP_URL_ENV = "PIPELINE_TESTBED_WEBAPP_URL"
TESTBED_WEBAPP_SYNC_REQUIRED_ENV = "PIPELINE_TESTBED_WEBAPP_SYNC_REQUIRED"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def build_site_task_testbed_webapp_publication(
    *,
    capture_session_id: str,
    intake_id: str,
    approved_task_digest: str,
    testbed: Mapping[str, Any],
    decision_evidence_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        verified = MaintainedSiteTaskTestbed.from_mapping(testbed).to_mapping()
    except DecisionEvidenceContractError as exc:
        raise ValueError(f"testbed_invalid:{exc}") from exc
    session_id = _text(capture_session_id)
    expected_intake_id = _text(intake_id)
    if not session_id or not expected_intake_id:
        raise ValueError("capture_session_id_and_intake_id_required")
    sources = [
        row
        for row in verified["source_capture_bundles"]
        if _text(_mapping(row).get("bundle_id")) == expected_intake_id
    ]
    if len(sources) != 1:
        raise ValueError("testbed_intake_source_binding_mismatch")
    expected_task_digest = _text(
        _mapping(verified.get("approved_task_definition")).get("digest")
    )
    if _text(approved_task_digest) != expected_task_digest:
        raise ValueError("testbed_approved_task_binding_mismatch")
    digest = verified["testbed_digest"]
    publication = {
        "schema_version": "site_task_testbed_publication.v1",
        "capture_session_id": session_id,
        "intake_id": expected_intake_id,
        "approved_task_digest": expected_task_digest,
        "testbed_id": verified["testbed_id"],
        "version": verified["version"],
        "testbed_digest": digest,
        "artifact_reference": {
            "uri": (
                f"testbed://{verified['testbed_id']}/{verified['version']}/"
                f"{digest.removeprefix('sha256:')}.json"
            ),
            "digest": digest,
        },
        "testbed": verified,
        "status": "testbed_ready",
        "proof_boundary": verified["proof_boundary"],
    }
    if decision_evidence_request is not None:
        request = DecisionEvidenceRequest.from_mapping(
            decision_evidence_request
        ).to_mapping()
        if (
            request["testbed_id"] != verified["testbed_id"]
            or request["testbed_version"] != verified["version"]
            or request["testbed_digest"] != verified["testbed_digest"]
        ):
            raise ValueError("decision_evidence_request_testbed_binding_mismatch")
        publication["decision_evidence_request"] = request
    return publication


def sync_site_task_testbed_to_webapp(
    *,
    capture_session_id: str,
    intake_id: str,
    approved_task_digest: str,
    testbed: Mapping[str, Any],
    decision_evidence_request: Mapping[str, Any] | None = None,
    endpoint_url: str | None = None,
    token: str | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.0,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    payload = build_site_task_testbed_webapp_publication(
        capture_session_id=capture_session_id,
        intake_id=intake_id,
        approved_task_digest=approved_task_digest,
        testbed=testbed,
        decision_evidence_request=decision_evidence_request,
    )
    resolved_url = _text(endpoint_url) or _text(os.getenv(TESTBED_WEBAPP_URL_ENV))
    resolved_token = _text(token) or _text(os.getenv("PIPELINE_SYNC_TOKEN"))
    common = {
        "schema_version": "site_task_testbed_webapp_sync_result.v1",
        "capture_session_id": payload["capture_session_id"],
        "intake_id": payload["intake_id"],
        "testbed_id": payload["testbed_id"],
        "version": payload["version"],
        "testbed_digest": payload["testbed_digest"],
        "proof_boundary": payload["proof_boundary"],
    }
    if not resolved_url or not resolved_token:
        return {**common, "status": "skipped", "reason": "sync_not_configured", "attempts": 0}
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
                    == "capture_site_task_testbed_publication_receipt.v1"
                    and receipt.get("status") == "testbed_ready"
                    and isinstance(receipt.get("already_exists"), bool)
                    and all(
                        receipt.get(field) == payload[field]
                        for field in (
                            "capture_session_id",
                            "intake_id",
                            "approved_task_digest",
                            "testbed_id",
                            "version",
                            "testbed_digest",
                            "artifact_reference",
                            "proof_boundary",
                        )
                    )
                    and receipt.get("request_digest")
                    == _mapping(payload.get("decision_evidence_request")).get(
                        "request_digest"
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
    return {
        **common,
        "status": "failed",
        "reason": last_reason,
        "attempts": attempts,
    }


__all__ = [
    "TESTBED_WEBAPP_SYNC_REQUIRED_ENV",
    "TESTBED_WEBAPP_URL_ENV",
    "build_site_task_testbed_webapp_publication",
    "sync_site_task_testbed_to_webapp",
]
