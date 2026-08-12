"""Signed idempotent publication of immutable launch receipts to the WebApp."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


LAUNCH_WEBAPP_URL_ENV = "PIPELINE_TASK_EVALUATION_LAUNCH_WEBAPP_URL"
LAUNCH_SUPERVISION_WEBAPP_URL_ENV = (
    "PIPELINE_TASK_EVALUATION_LAUNCH_SUPERVISION_WEBAPP_URL"
)
LAUNCH_PROGRESS_WEBAPP_URL_ENV = "PIPELINE_TASK_EVALUATION_LAUNCH_PROGRESS_WEBAPP_URL"


def sync_launch_progress_to_webapp(
    *,
    progress: Mapping[str, Any],
    endpoint_url: str | None = None,
    token: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Publish one non-terminal progress record for an in-flight launch.

    Progress is best effort by design: a failed publish must never affect the
    run, so every failure is recorded and returned rather than raised. It also
    carries no receipt digest, because it is an observation rather than an
    immutable result.
    """

    payload = dict(progress)
    common = {
        "schema_version": "task_evaluation_launch_webapp_progress_result.v1",
        "launch_id": payload.get("launch_id"),
        "run_id": payload.get("run_id"),
        "request_digest": payload.get("request_digest"),
    }
    resolved_url = str(
        endpoint_url or os.getenv(LAUNCH_PROGRESS_WEBAPP_URL_ENV) or ""
    ).strip()
    resolved_token = str(token or os.getenv("PIPELINE_SYNC_TOKEN") or "").strip()
    if not resolved_url or not resolved_token:
        return {**common, "status": "skipped", "reason": "progress_sync_not_configured"}
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    try:
        url = validated_https_sync_url(resolved_url)
    except ValueError:
        return {**common, "status": "failed", "reason": "sync_url_invalid"}
    outbound = urllib_request.Request(
        url,
        data=body,
        headers=_pipeline_sync_headers(resolved_token, body),
        method="POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - URL is pinned to validated HTTPS
            outbound, timeout=max(0.1, timeout_seconds)
        ) as response:
            response.read()
    except urllib_error.HTTPError as exc:
        return {**common, "status": "failed", "reason": f"http_error:{exc.code}"}
    except urllib_error.URLError as exc:
        return {**common, "status": "failed", "reason": f"url_error:{exc.reason}"}
    except (TimeoutError, ValueError) as exc:
        return {**common, "status": "failed", "reason": type(exc).__name__.lower()}
    return {**common, "status": "succeeded"}


def sync_launch_receipt_to_webapp(
    *, receipt: Mapping[str, Any], endpoint_url: str | None = None,
    token: str | None = None, timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    payload = dict(receipt)
    common = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "launch_id": payload.get("launch_id"),
        "run_id": payload.get("run_id"),
        "request_digest": payload.get("request_digest"),
        "receipt_digest": payload.get("receipt_digest"),
    }
    resolved_url = str(endpoint_url or os.getenv(LAUNCH_WEBAPP_URL_ENV) or "").strip()
    resolved_token = str(token or os.getenv("PIPELINE_SYNC_TOKEN") or "").strip()
    if not resolved_url or not resolved_token:
        return {**common, "status": "skipped", "reason": "sync_not_configured"}
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    try:
        url = validated_https_sync_url(resolved_url)
    except ValueError:
        return {**common, "status": "failed", "reason": "sync_url_invalid"}
    outbound = urllib_request.Request(
        url,
        data=body,
        headers=_pipeline_sync_headers(resolved_token, body),
        method="POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - URL is pinned to validated HTTPS
            outbound, timeout=max(0.1, timeout_seconds)
        ) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        return {**common, "status": "failed", "reason": f"http_error:{exc.code}"}
    except urllib_error.URLError as exc:
        return {**common, "status": "failed", "reason": f"url_error:{exc.reason}"}
    except (TimeoutError, ValueError) as exc:
        return {**common, "status": "failed", "reason": type(exc).__name__.lower()}
    try:
        response = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {**common, "status": "failed", "reason": "invalid_json"}
    if not isinstance(response, Mapping) or any(
        response.get(field) != common[field]
        for field in ("launch_id", "run_id", "request_digest", "receipt_digest")
    ):
        return {**common, "status": "failed", "reason": "response_binding_mismatch"}
    return {**common, "status": "succeeded", "response": dict(response)}


def sync_launch_supervision_to_webapp(
    *, supervision: Mapping[str, Any], endpoint_url: str | None = None,
    token: str | None = None, timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    payload = dict(supervision)
    common = {
        "schema_version": "task_evaluation_launch_supervision_webapp_sync_result.v1",
        "snapshot_digest": payload.get("snapshot_digest"),
        "supervision_digest": payload.get("supervision_digest"),
    }
    resolved_url = str(
        endpoint_url or os.getenv(LAUNCH_SUPERVISION_WEBAPP_URL_ENV) or ""
    ).strip()
    resolved_token = str(token or os.getenv("PIPELINE_SYNC_TOKEN") or "").strip()
    if not resolved_url or not resolved_token:
        return {**common, "status": "skipped", "reason": "sync_not_configured"}
    try:
        url = validated_https_sync_url(resolved_url)
    except ValueError:
        return {**common, "status": "failed", "reason": "sync_url_invalid"}
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    outbound = urllib_request.Request(
        url,
        data=body,
        headers=_pipeline_sync_headers(resolved_token, body),
        method="POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - URL is pinned to validated HTTPS
            outbound, timeout=max(0.1, timeout_seconds)
        ) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        return {**common, "status": "failed", "reason": f"http_error:{exc.code}"}
    except urllib_error.URLError as exc:
        return {**common, "status": "failed", "reason": f"url_error:{exc.reason}"}
    except (TimeoutError, ValueError) as exc:
        return {**common, "status": "failed", "reason": type(exc).__name__.lower()}
    try:
        response = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {**common, "status": "failed", "reason": "invalid_json"}
    if not isinstance(response, Mapping) or any(
        response.get(field) != common[field]
        for field in ("snapshot_digest", "supervision_digest")
    ):
        return {**common, "status": "failed", "reason": "response_binding_mismatch"}
    return {**common, "status": "succeeded", "response": dict(response)}


__all__ = [
    "LAUNCH_SUPERVISION_WEBAPP_URL_ENV",
    "LAUNCH_WEBAPP_URL_ENV",
    "sync_launch_receipt_to_webapp",
    "sync_launch_supervision_to_webapp",
]
