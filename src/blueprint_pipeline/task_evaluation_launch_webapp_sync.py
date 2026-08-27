"""Signed idempotent publication of immutable launch receipts to the WebApp."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


LAUNCH_WEBAPP_URL_ENV = "PIPELINE_TASK_EVALUATION_LAUNCH_WEBAPP_URL"
LAUNCH_SUPERVISION_WEBAPP_URL_ENV = (
    "PIPELINE_TASK_EVALUATION_LAUNCH_SUPERVISION_WEBAPP_URL"
)
LAUNCH_PROGRESS_WEBAPP_URL_ENV = "PIPELINE_TASK_EVALUATION_LAUNCH_PROGRESS_WEBAPP_URL"
PIPELINE_SYNC_TOKEN_FILE_ENV = "PIPELINE_SYNC_TOKEN_FILE"


class PipelineSyncTokenError(RuntimeError):
    """The canonical file-backed WebApp synchronization token is unavailable."""


def load_pipeline_sync_token(
    *,
    token: str | None = None,
    token_file_path: str | Path | None = None,
    require_file: bool = False,
) -> str:
    """Resolve a sync token without exposing its bytes in errors or receipts."""

    if token is not None:
        resolved = str(token).strip()
        if not resolved:
            raise PipelineSyncTokenError("pipeline_sync_token_missing")
        if require_file:
            raise PipelineSyncTokenError("pipeline_sync_token_file_required")
        return resolved
    raw_path = str(
        token_file_path or os.getenv(PIPELINE_SYNC_TOKEN_FILE_ENV) or ""
    ).strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        descriptor = -1
        try:
            if path.is_symlink():
                raise PipelineSyncTokenError("pipeline_sync_token_file_unsafe")
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
            mode = stat.S_IMODE(metadata.st_mode)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or mode & ~0o640
                or not mode & 0o440
            ):
                raise PipelineSyncTokenError("pipeline_sync_token_file_unsafe")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read(4097)
        except PipelineSyncTokenError:
            raise
        except OSError as exc:
            raise PipelineSyncTokenError(
                "pipeline_sync_token_file_unavailable"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(payload) > 4096:
            raise PipelineSyncTokenError("pipeline_sync_token_file_unsafe")
        try:
            resolved = payload.decode("utf-8").strip()
        except UnicodeError as exc:
            raise PipelineSyncTokenError(
                "pipeline_sync_token_file_unavailable"
            ) from exc
        if not resolved:
            raise PipelineSyncTokenError("pipeline_sync_token_missing")
        return resolved
    if require_file:
        raise PipelineSyncTokenError("pipeline_sync_token_file_required")
    resolved = str(os.getenv("PIPELINE_SYNC_TOKEN") or "").strip()
    if not resolved:
        raise PipelineSyncTokenError("pipeline_sync_token_missing")
    return resolved


def _optional_pipeline_sync_token(token: str | None) -> str:
    try:
        return load_pipeline_sync_token(token=token)
    except PipelineSyncTokenError:
        return ""


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
    resolved_token = _optional_pipeline_sync_token(token)
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
        for field in ("launch_id", "run_id", "request_digest")
    ):
        return {**common, "status": "failed", "reason": "response_binding_mismatch"}
    if response.get("schema_version") != (
        "task_evaluation_launch_progress_web_sync_receipt.v1"
    ):
        return {**common, "status": "failed", "reason": "response_schema_mismatch"}
    if response.get("status") not in {"recorded", "ignored_terminal"}:
        return {**common, "status": "failed", "reason": "response_status_invalid"}
    if response.get("phase") != payload.get("phase"):
        return {**common, "status": "failed", "reason": "response_phase_mismatch"}
    return {**common, "status": "succeeded", "response": dict(response)}


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
    terminal = payload.get("terminal_evidence")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    scene_configuration = terminal.get("scene_configuration")
    scene_configuration = (
        scene_configuration if isinstance(scene_configuration, Mapping) else {}
    )
    offering = scene_configuration.get("configured_scene_offering")
    if isinstance(offering, Mapping):
        common["configured_scene_offering_digest"] = offering.get(
            "offering_digest"
        )
        common["configured_scene_offering_status"] = "launch_ready"
    resolved_url = str(endpoint_url or os.getenv(LAUNCH_WEBAPP_URL_ENV) or "").strip()
    resolved_token = _optional_pipeline_sync_token(token)
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
    if response.get("schema_version") != (
        "task_evaluation_launch_web_sync_receipt.v1"
    ):
        return {**common, "status": "failed", "reason": "response_schema_mismatch"}
    if (
        response.get("status") != payload.get("status")
        or not isinstance(response.get("already_exists"), bool)
    ):
        return {**common, "status": "failed", "reason": "response_status_mismatch"}
    if "configured_scene_offering_digest" in common and any(
        response.get(field) != common[field]
        for field in (
            "configured_scene_offering_digest",
            "configured_scene_offering_status",
        )
    ):
        return {
            **common,
            "status": "failed",
            "reason": "configured_scene_offering_binding_mismatch",
        }
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
    resolved_token = _optional_pipeline_sync_token(token)
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
    "LAUNCH_PROGRESS_WEBAPP_URL_ENV",
    "LAUNCH_SUPERVISION_WEBAPP_URL_ENV",
    "LAUNCH_WEBAPP_URL_ENV",
    "PIPELINE_SYNC_TOKEN_FILE_ENV",
    "PipelineSyncTokenError",
    "load_pipeline_sync_token",
    "sync_launch_progress_to_webapp",
    "sync_launch_receipt_to_webapp",
    "sync_launch_supervision_to_webapp",
]
