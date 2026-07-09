"""Client for the dedicated video_to_world GPU service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

from .cloud_run_iam_auth import CloudRunIamAuthError, cloud_run_id_token_headers


def _runner_url() -> str:
    return str(os.getenv("VIDEO_TO_WORLD_URL") or "").strip()


def _runner_token() -> str:
    return str(os.getenv("VIDEO_TO_WORLD_RUNNER_TOKEN") or os.getenv("PRIVACY_RUNNER_TOKEN") or "").strip()


def _timeout_seconds() -> int:
    raw = str(os.getenv("VIDEO_TO_WORLD_TIMEOUT_SECONDS") or "7200").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 7200
    return max(30, value)


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = _runner_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def run_video_to_world_provider(
    *,
    video_path: Path,
    video_uri: str,
    geometry_root: Path,
    dynamic_mask_manifest_path: Path,
    dynamic_mask_manifest_uri: str,
    provider: str,
    model: str,
    execution_mode: str,
    video_probe: Mapping[str, Any],
) -> Dict[str, Any]:
    url = _runner_url()
    if not url:
        raise RuntimeError("video_to_world_runner_not_configured")

    request_payload = {
        "input_video_path": str(video_path),
        "input_video_uri": video_uri,
        "geometry_root_path": str(geometry_root),
        "geometry_root_uri": dynamic_mask_manifest_uri.rsplit("/masks/", 1)[0],
        "dynamic_mask_manifest_path": str(dynamic_mask_manifest_path),
        "dynamic_mask_manifest_uri": dynamic_mask_manifest_uri,
        "provider": provider,
        "model": model,
        "execution_mode": execution_mode,
        "video_probe": dict(video_probe),
    }
    raw = json.dumps(request_payload).encode("utf-8")
    endpoint = url.rstrip("/") + "/run"
    try:
        headers = cloud_run_id_token_headers(_headers(), url=url)
    except CloudRunIamAuthError as exc:
        raise RuntimeError(str(exc)) from exc
    req = urllib_request.Request(endpoint, data=raw, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=_timeout_seconds()) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"video_to_world_http_{exc.code}:{detail[-1000:]}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"video_to_world_unreachable:{exc.reason}") from exc

    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError("video_to_world_invalid_json") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("video_to_world_invalid_payload")
    if str(payload.get("status") or "").strip().lower() != "succeeded":
        raise RuntimeError(str(payload.get("reason") or "video_to_world_failed"))
    return payload
