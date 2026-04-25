"""Helpers for syncing package and review metadata back into Blueprint-WebApp's control plane."""

from __future__ import annotations

import json
import os
import time
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from .launch_proof_policy import buyer_access_required, production_forces_false, production_forces_true


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


def _artifact_uri_checksums(artifacts: Mapping[str, Any]) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for key, value in artifacts.items():
        if not value:
            continue
        text = str(value)
        checksums[str(key)] = sha256(text.encode("utf-8")).hexdigest()
    return checksums


def _extract_webapp_response_ids(response: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "id",
        "request_id",
        "attachment_id",
        "pipeline_attachment_id",
        "listing_id",
        "site_world_id",
        "artifact_id",
        "capture_job_id",
        "buyer_artifact_id",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        value = response.get(key)
        if value:
            out[key] = value
    nested = response.get("attachment") if isinstance(response.get("attachment"), Mapping) else {}
    for key in keys:
        value = nested.get(key)
        if value and key not in out:
            out[key] = value
    return out


def _buyer_access_check_payload(response: Mapping[str, Any]) -> Dict[str, Any]:
    check_url = _string_env("PIPELINE_BUYER_ACCESS_CHECK_URL")
    check_token = _string_env("PIPELINE_BUYER_ACCESS_CHECK_TOKEN") or _string_env("PIPELINE_SYNC_TOKEN")
    if not check_url:
        return {
            "status": "blocked" if buyer_access_required() else "skipped",
            "buyer_access_checked": False,
            "reason": "buyer_access_check_not_configured",
            "blocker": "buyer_access_check_not_configured" if buyer_access_required() else None,
        }
    payload = {"webapp_response_ids": _extract_webapp_response_ids(response)}
    request = urllib_request.Request(
        check_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {check_token}"} if check_token else {}),
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=max(1, _int_env("PIPELINE_BUYER_ACCESS_TIMEOUT_SECONDS", 10))) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        return {
            "status": "blocked",
            "buyer_access_checked": False,
            "reason": f"buyer_access_check_failed:{exc}",
            "blocker": "buyer_access_check_failed",
        }
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {}
    allowed = bool(parsed.get("ok") or parsed.get("accessible") or parsed.get("buyer_accessible"))
    return {
        "status": "succeeded" if allowed else "blocked",
        "buyer_access_checked": True,
        "buyer_accessible": allowed,
        "response": parsed if isinstance(parsed, dict) else {},
        "blocker": None if allowed else "buyer_access_check_not_accessible",
    }


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
    sync_required = production_forces_true("PIPELINE_SYNC_REQUIRED", default=False)
    placeholder_sync_allowed = production_forces_false("PIPELINE_SYNC_ALLOW_PLACEHOLDER_FALLBACK", default=True)
    max_attempts = max(1, _int_env("PIPELINE_SYNC_MAX_ATTEMPTS", 3))
    retry_delay_ms = max(0, _int_env("PIPELINE_SYNC_RETRY_DELAY_MS", 500))
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
    payload["artifact_uri_checksums"] = _artifact_uri_checksums(payload.get("artifacts") or {})
    payload["placeholder_fallback_allowed"] = bool(placeholder_sync_allowed)
    if not sync_url or not sync_token:
        result = {
            "status": "failed" if sync_required else "skipped",
            "reason": "sync_not_configured",
            "attempts": 0,
            "attachment_payload": payload,
            "artifact_uri_checksums": payload["artifact_uri_checksums"],
            "webapp_response_ids": {},
            "buyer_access_check": {
                "status": "blocked" if buyer_access_required() else "skipped",
                "buyer_access_checked": False,
                "reason": "sync_not_configured",
                "blocker": "sync_not_configured" if buyer_access_required() else None,
            },
            "deployment_readiness": (
                {str(key): value for key, value in deployment_readiness.items()}
                if isinstance(deployment_readiness, Mapping)
                else None
            ),
        }
        if sync_required:
            raise WebappSyncError("sync_not_configured")
        return result

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
                response_ids = _extract_webapp_response_ids(parsed if isinstance(parsed, dict) else {})
                buyer_access_check = _buyer_access_check_payload(parsed if isinstance(parsed, dict) else {})
                return {
                    "status": "succeeded",
                    "attempts": attempt,
                    "response": parsed if isinstance(parsed, dict) else {},
                    "webapp_response_ids": response_ids,
                    "artifact_uri_checksums": payload["artifact_uri_checksums"],
                    "buyer_access_check": buyer_access_check,
                    "attachment_payload": payload,
                    "deployment_readiness": (
                        {str(key): value for key, value in deployment_readiness.items()}
                        if isinstance(deployment_readiness, Mapping)
                        else None
                    ),
                }

        if attempt < max_attempts and retry_delay_ms:
            time.sleep(retry_delay_ms / 1000)

    if sync_required:
        raise WebappSyncError(last_reason)
    return {
        "status": "failed",
        "reason": last_reason,
        "attempts": max_attempts,
        "webapp_response_ids": {},
        "artifact_uri_checksums": payload["artifact_uri_checksums"],
        "buyer_access_check": {
            "status": "blocked" if buyer_access_required() else "skipped",
            "buyer_access_checked": False,
            "reason": last_reason,
            "blocker": "sync_failed" if buyer_access_required() else None,
        },
        "attachment_payload": payload,
        "deployment_readiness": (
            {str(key): value for key, value in deployment_readiness.items()}
            if isinstance(deployment_readiness, Mapping)
            else None
        ),
    }
