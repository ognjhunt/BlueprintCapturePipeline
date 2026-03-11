"""Helpers for syncing pipeline attachment metadata back into Blueprint-WebApp."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


def _string_env(name: str) -> str:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) else ""


def _int_env(name: str, default: int) -> int:
    raw = _string_env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def derive_webapp_qualification_state(*, readiness_state: object, completeness_status: object) -> str:
    normalized_readiness = str(readiness_state or "").strip().lower()
    normalized_completeness = str(completeness_status or "").strip().lower()
    if normalized_completeness and normalized_completeness != "sufficient":
        return "needs_more_evidence"
    if normalized_readiness == "ready":
        return "qualified_ready"
    if normalized_readiness == "risky":
        return "qualified_risky"
    return "not_ready_yet"


def derive_webapp_opportunity_state(*, qualification_state: object) -> str:
    normalized = str(qualification_state or "").strip().lower()
    if normalized in {"qualified_ready", "qualified_risky"}:
        return "handoff_ready"
    return "not_applicable"


def sync_webapp_pipeline_attachment(
    *,
    site_submission_id: object,
    request_id: object = None,
    scene_id: object,
    capture_id: object,
    pipeline_prefix: object,
    qualification_state: object,
    opportunity_state: object,
    artifacts: Mapping[str, Any],
    derived_assets: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    sync_url = _string_env("PIPELINE_SYNC_WEBAPP_URL")
    sync_token = _string_env("PIPELINE_SYNC_TOKEN")
    if not sync_url or not sync_token:
        return None

    payload = {
        "schema_version": "v1",
        "site_submission_id": str(site_submission_id or "").strip(),
        "request_id": str(request_id or "").strip(),
        "scene_id": str(scene_id or "").strip(),
        "capture_id": str(capture_id or "").strip(),
        "pipeline_prefix": str(pipeline_prefix or "").strip(),
        "qualification_state": str(qualification_state or "").strip(),
        "opportunity_state": str(opportunity_state or "").strip(),
        "artifacts": {str(key): value for key, value in artifacts.items() if value},
        "derived_assets": (
            {str(key): value for key, value in derived_assets.items() if value}
            if isinstance(derived_assets, Mapping)
            else {}
        ),
    }
    request = urllib_request.Request(
        sync_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Blueprint-Pipeline-Token": sync_token,
        },
        method="POST",
    )
    timeout_seconds = max(1, _int_env("PIPELINE_SYNC_TIMEOUT_SECONDS", 10))
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, ValueError):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
