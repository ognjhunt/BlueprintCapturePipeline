"""Read the website's confirmed task immediately before scene preparation.

The original upload manifest is never rewritten to impersonate a later
confirmation. The authenticated snapshot is retained as derived run evidence.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.parse import quote, urlsplit

from .decision_evidence_contracts import canonical_digest
from .safe_outbound_http import pinned_api_policy, request as safe_request
from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


def validate_website_task_context(
    value: Mapping[str, Any], *, request_id: str, scene_id: str, capture_id: str,
) -> dict[str, Any]:
    if value.get("schema_version") != "website_site_task_context.v1":
        raise ValueError("website_task_context_schema_invalid")
    if any(value.get(key) != expected for key, expected in (
        ("request_id", request_id), ("scene_id", scene_id), ("capture_id", capture_id),
    )):
        raise ValueError("website_task_context_identity_mismatch")
    if value.get("context_digest") != canonical_digest(value, digest_field="context_digest"):
        raise ValueError("website_task_context_digest_mismatch")
    if value.get("confirmed") is not True or not value.get("confirmed_at"):
        raise ValueError("website_task_context_not_confirmed")
    if not isinstance(value.get("description"), str) or not value["description"].strip():
        raise ValueError("website_task_context_description_missing")
    return dict(value)


def load_current_website_task_context(
    *, request_id: str, scene_id: str, capture_id: str,
) -> dict[str, Any]:
    """One authenticated bounded read; errors hold preparation without fallback."""
    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        raise ValueError("website_task_context_webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    token = load_pipeline_sync_token()
    body = json.dumps({"request_id": request_id, "scene_id": scene_id},
                      separators=(",", ":")).encode()
    response = safe_request(
        f"{origin}/api/internal/pipeline/creator-captures/{quote(capture_id, safe='')}/task-context",
        method="POST", data=body, headers=_pipeline_sync_headers(token, body),
        timeout_seconds=10, policy=pinned_api_policy(origin, max_response_bytes=100_000),
        max_response_bytes=100_000,
    )
    value = json.loads(response.body)
    if not isinstance(value, Mapping):
        raise ValueError("website_task_context_response_invalid")
    return validate_website_task_context(value, request_id=request_id,
                                         scene_id=scene_id, capture_id=capture_id)
