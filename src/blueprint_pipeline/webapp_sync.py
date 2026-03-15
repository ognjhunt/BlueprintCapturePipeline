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


def build_webapp_pipeline_attachment_payload(
    *,
    site_submission_id: object,
    request_id: object = None,
    buyer_request_id: object = None,
    capture_job_id: object = None,
    scene_id: object,
    capture_id: object,
    pipeline_prefix: object,
    qualification_state: object,
    opportunity_state: object,
    artifacts: Mapping[str, Any],
    derived_assets: Optional[Mapping[str, Any]] = None,
    deployment_readiness: Optional[Mapping[str, Any]] = None,
    authoritative_state_update: bool = False,
) -> Dict[str, Any]:
    payload = {
        "schema_version": "v1",
        "site_submission_id": str(site_submission_id or "").strip(),
        "request_id": str(request_id or "").strip(),
        "buyer_request_id": str(buyer_request_id or "").strip(),
        "capture_job_id": str(capture_job_id or "").strip(),
        "scene_id": str(scene_id or "").strip(),
        "capture_id": str(capture_id or "").strip(),
        "pipeline_prefix": str(pipeline_prefix or "").strip(),
        "qualification_state": str(qualification_state or "").strip(),
        "opportunity_state": str(opportunity_state or "").strip(),
        "authoritative_state_update": bool(authoritative_state_update),
        "artifacts": {str(key): value for key, value in artifacts.items() if value},
        "derived_assets": (
            {str(key): value for key, value in derived_assets.items() if value}
            if isinstance(derived_assets, Mapping)
            else {}
        ),
        "deployment_readiness": (
            {str(key): value for key, value in deployment_readiness.items()}
            if isinstance(deployment_readiness, Mapping)
            else None
        ),
    }
    if not payload["site_submission_id"] and not payload["request_id"]:
        raise ValueError("site_submission_id or request_id is required")
    return payload


def sync_webapp_pipeline_attachment(
    *,
    site_submission_id: object,
    request_id: object = None,
    buyer_request_id: object = None,
    capture_job_id: object = None,
    scene_id: object,
    capture_id: object,
    pipeline_prefix: object,
    qualification_state: object,
    opportunity_state: object,
    artifacts: Mapping[str, Any],
    derived_assets: Optional[Mapping[str, Any]] = None,
    deployment_readiness: Optional[Mapping[str, Any]] = None,
    authoritative_state_update: bool = False,
) -> Optional[Dict[str, Any]]:
    sync_url = _string_env("PIPELINE_SYNC_WEBAPP_URL")
    sync_token = _string_env("PIPELINE_SYNC_TOKEN")
    if not sync_url or not sync_token:
        return None

    payload = build_webapp_pipeline_attachment_payload(
        site_submission_id=site_submission_id,
        request_id=request_id,
        buyer_request_id=buyer_request_id,
        capture_job_id=capture_job_id,
        scene_id=scene_id,
        capture_id=capture_id,
        pipeline_prefix=pipeline_prefix,
        qualification_state=qualification_state,
        opportunity_state=opportunity_state,
        artifacts=artifacts,
        derived_assets=derived_assets,
        deployment_readiness=deployment_readiness,
        authoritative_state_update=authoritative_state_update,
    )
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
