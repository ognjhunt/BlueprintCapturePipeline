"""Use the existing Pipeline service signature and exact publication receipts."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit, urlunsplit

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .job_evidence import PUBLICATION_SCHEMA
from .task_evaluation_launch_webapp_sync import PipelineSyncTokenError, load_pipeline_sync_token
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


JOB_EVIDENCE_WEBAPP_URL_ENV = "PIPELINE_JOB_EVIDENCE_WEBAPP_URL"


def _endpoint() -> str:
    configured = str(os.getenv(JOB_EVIDENCE_WEBAPP_URL_ENV) or "").strip()
    if configured:
        return validated_https_sync_url(configured)
    # Reuse the configured WebApp service origin, never a public owner link.
    for name in ("PIPELINE_TASK_EVALUATION_RUN_WEBAPP_URL", "PIPELINE_TESTBED_WEBAPP_URL"):
        existing = str(os.getenv(name) or "").strip()
        if existing:
            parsed = urlsplit(validated_https_sync_url(existing))
            return urlunsplit((parsed.scheme, parsed.netloc, "/api/internal/job-evidence-publications", "", ""))
    return ""


def sync_job_evidence_to_webapp(
    publication: Mapping[str, Any], *, endpoint_url: str | None = None,
    token: str | None = None, timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    payload = dict(publication)
    if payload.get("schema_version") != PUBLICATION_SCHEMA or payload.get("publication_digest") != cross_runtime_canonical_digest(
        payload, digest_field="publication_digest"
    ):
        raise ValueError("job_evidence_publication_invalid")
    endpoint = validated_https_sync_url(endpoint_url) if endpoint_url else _endpoint()
    try:
        secret = load_pipeline_sync_token(token=token)
    except PipelineSyncTokenError:
        secret = ""
    common = {"request_id": payload["request_id"], "publication_digest": payload["publication_digest"]}
    if not endpoint or not secret:
        return {**common, "status": "skipped", "reason": "sync_not_configured"}
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    outbound = urllib_request.Request(
        endpoint, data=body, headers=_pipeline_sync_headers(secret, body), method="POST"
    )
    try:
        with urllib_request.urlopen(outbound, timeout=timeout_seconds) as response:  # nosec B310 -- HTTPS operator configuration validated above
            raw = response.read(65537)
        receipt = json.loads(raw) if len(raw) <= 65536 else {}
    except urllib_error.HTTPError as exc:
        return {**common, "status": "failed", "reason": f"http_error:{exc.code}"}
    except (OSError, ValueError):
        return {**common, "status": "failed", "reason": "publication_transport_failed"}
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != "job_evidence_publication_receipt.v1" or receipt.get("status") not in {"published", "replayed"} or any(
        receipt.get(field) != payload[field]
        for field in ("request_id", "job_id", "request_digest", "publication_digest")
    ):
        return {**common, "status": "failed", "reason": "publication_receipt_mismatch"}
    return {**common, "status": "succeeded", "receipt": dict(receipt)}
