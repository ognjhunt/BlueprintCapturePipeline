"""Helpers for syncing pipeline attachment metadata back into Blueprint-WebApp."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


class WebappSyncError(RuntimeError):
    """Raised when pipeline-to-webapp sync is configured as required and fails."""


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


def _bool_env(name: str, default: bool = False) -> bool:
    raw = _string_env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


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
) -> Dict[str, Any]:
    sync_url = _string_env("PIPELINE_SYNC_WEBAPP_URL")
    sync_token = _string_env("PIPELINE_SYNC_TOKEN")
    sync_required = _bool_env("PIPELINE_SYNC_REQUIRED", default=False)
    max_attempts = max(1, _int_env("PIPELINE_SYNC_MAX_ATTEMPTS", 3))
    retry_delay_ms = max(0, _int_env("PIPELINE_SYNC_RETRY_DELAY_MS", 500))
    if not sync_url or not sync_token:
        result = {
            "status": "failed" if sync_required else "skipped",
            "reason": "sync_not_configured",
            "attempts": 0,
        }
        if sync_required:
            raise WebappSyncError("sync_not_configured")
        return result

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
    timeout_seconds = max(1, _int_env("PIPELINE_SYNC_TIMEOUT_SECONDS", 10))
    last_reason = "sync_unknown_failure"

    for attempt in range(1, max_attempts + 1):
        request = urllib_request.Request(
            sync_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Blueprint-Pipeline-Token": sync_token,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            last_reason = f"http_error:{exc.code}"
        except urllib_error.URLError as exc:
            last_reason = f"url_error:{exc.reason}"
        except (TimeoutError, ValueError) as exc:
            last_reason = exc.__class__.__name__.lower()
        else:
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                last_reason = "invalid_json"
            else:
                return {
                    "status": "succeeded",
                    "attempts": attempt,
                    "response": parsed if isinstance(parsed, dict) else {},
                }

        if attempt < max_attempts and retry_delay_ms:
            time.sleep(retry_delay_ms / 1000)

    if sync_required:
        raise WebappSyncError(last_reason)
    return {
        "status": "failed",
        "reason": last_reason,
        "attempts": max_attempts,
    }
