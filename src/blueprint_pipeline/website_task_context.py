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
    value = website_webapp_request(capture_id=capture_id, operation="task-context",
                                  payload={"request_id": request_id, "scene_id": scene_id})
    return validate_website_task_context(value, request_id=request_id,
                                         scene_id=scene_id, capture_id=capture_id)


def website_webapp_request(*, capture_id: str, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Use the existing signed transport for private preparation control data."""
    if operation not in {"task-context", "scene-sponsorship", "prepared-scene", "visual-scene"}:
        raise ValueError("website_control_operation_invalid")
    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        raise ValueError("website_task_context_webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    token = load_pipeline_sync_token()
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = safe_request(
        f"{origin}/api/internal/pipeline/creator-captures/{quote(capture_id, safe='')}/{operation}",
        method="POST", data=body, headers=_pipeline_sync_headers(token, body),
        timeout_seconds=10, policy=pinned_api_policy(origin, max_response_bytes=100_000),
        max_response_bytes=100_000,
    )
    value = json.loads(response.body)
    if not isinstance(value, Mapping):
        raise ValueError("website_task_context_response_invalid")
    return dict(value)


def publish_website_visual_scene(*, descriptor: Mapping[str, Any], world: Mapping[str, Any],
                                 operation_id: str) -> dict[str, Any]:
    """Publish the first viewable world before mesh downloads or native authoring."""
    metadata = descriptor.get("metadata") or {}
    context = metadata.get("site_task_context") or {}
    if (metadata.get("capture_entry_source") != "browser_self_capture"
            or (metadata.get("clean_plate") or {}).get("privacy_verified") is not True):
        raise ValueError("website_visual_scene_preparation_missing")
    validate_website_task_context(context, request_id=context.get("request_id", ""),
                                  scene_id=descriptor["scene_id"], capture_id=descriptor["capture_id"])
    assets = world.get("assets") or {}
    imagery = assets.get("imagery") or {}
    world_id = world.get("world_id") or world.get("id")
    launch_url = world.get("world_marble_url")
    if not world_id or not launch_url or not operation_id:
        raise ValueError("website_visual_scene_not_viewable")
    payload = {"request_id": context["request_id"], "scene_id": context["scene_id"],
               "task_context_digest": context["context_digest"], "world_id": world_id,
               "operation_id": operation_id, "model": world.get("model") or "marble-1.1-plus",
               "launch_url": launch_url, "thumbnail_url": assets.get("thumbnail_url") or world.get("thumbnail_url"),
               "pano_url": imagery.get("pano_url")}
    value = website_webapp_request(capture_id=context["capture_id"], operation="visual-scene", payload=payload)
    if (value.get("world_id") != world_id or value.get("state") != "ready"
            or value.get("task_context_digest") != context["context_digest"]):
        raise ValueError("website_visual_scene_receipt_invalid")
    return value


def load_website_scene_sponsorship(*, task_context: Mapping[str, Any], now: float) -> dict[str, Any]:
    value = website_webapp_request(capture_id=task_context["capture_id"], operation="scene-sponsorship",
        payload={"request_id": task_context["request_id"], "scene_id": task_context["scene_id"]})
    from math import isfinite
    amounts = [value.get(key) for key in ("preparation_max_total_spend_usd", "upstream_max_spend_usd", "max_total_spend_usd")]
    if (value.get("schema_version") != "website_scene_sponsorship.v1" or value.get("sponsor") != "blueprint"
            or value.get("authority_digest") != canonical_digest(value, digest_field="authority_digest")
            or any(value.get(key) != task_context[key] for key in ("request_id", "scene_id", "capture_id"))
            or value.get("task_context_digest") != task_context["context_digest"]
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not isfinite(x) or x <= 0 for x in amounts)
            or amounts[1] + amounts[2] > amounts[0]
            or not isinstance(value.get("expires_at_epoch"), (int, float))
            or not now < value["expires_at_epoch"] <= now + 86400):
        raise ValueError("website_scene_sponsorship_binding_invalid")
    return value


def enqueue_website_prepared_scene(*, task_context: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    value = website_webapp_request(capture_id=task_context["capture_id"], operation="prepared-scene",
        payload={"request_id": task_context["request_id"], "scene_id": task_context["scene_id"], "request": request})
    if (value.get("request_digest") != canonical_digest(request) or value.get("state") != "forward_pending"
            or not isinstance(value.get("id"), str) or not value["id"].startswith("scene-")):
        raise ValueError("website_scene_outbox_receipt_invalid")
    return value
