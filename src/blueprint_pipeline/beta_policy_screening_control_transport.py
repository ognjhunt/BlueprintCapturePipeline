"""Existing signed controller transport for exact screening cancellation reads."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit

from .beta_policy_screening_control_contracts import validate_binding, validate_control


def request_control(*, binding: Mapping[str, Any], execution_deadline_epoch: float | None = None,
                    transport=None) -> dict[str, Any]:
    """Current source/cancel observation; this response grants no fresh dispatch."""
    from .safe_outbound_http import pinned_api_policy, request
    from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
    from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url
    binding = validate_binding(binding)
    payload = {"schema_version": "blueprint.beta_policy_screening_execution_control_request.v1", **binding}
    if execution_deadline_epoch is not None:
        if (type(execution_deadline_epoch) not in (int, float) or not math.isfinite(execution_deadline_epoch)
                or execution_deadline_epoch <= 0):
            raise ValueError("beta_screening_control_deadline_invalid")
        payload["execution_deadline_at_ms"] = math.floor(execution_deadline_epoch * 1000)
    def validated(value):
        control = validate_control(value, binding=binding)
        if (control["status"] == "current" and execution_deadline_epoch is not None
                and control["execution_deadline_at_ms"] > payload["execution_deadline_at_ms"]):
            raise ValueError("beta_screening_execution_control_deadline_extended")
        return control
    if transport is not None:
        return validated(transport(payload))
    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        raise ValueError("beta_screening_execution_control_webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    try:
        response = request(f"{origin}/api/internal/pipeline/beta-policy-screening-runs/"
            f"{quote(binding['campaign_id'], safe='')}/execution-control", method="POST", data=body,
            headers=_pipeline_sync_headers(load_pipeline_sync_token(), body), timeout_seconds=10,
            policy=pinned_api_policy(origin, max_response_bytes=100_000), max_response_bytes=100_000)
    except HTTPError as exc:
        code = exc.code
        exc.close()
        raise ValueError(f"beta_screening_execution_control_http_{code}") from None
    return validated(json.loads(response.body))
