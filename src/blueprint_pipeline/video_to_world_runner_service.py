"""HTTP service wrapper for video_to_world execution."""

from __future__ import annotations

import hmac
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .logging_utils import log_event
from .video_to_world_service_runtime import execute_video_to_world_request


logger = logging.getLogger(__name__)
DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
LOCAL_PATH_ROOT_ENV = "VIDEO_TO_WORLD_RUNNER_LOCAL_PATH_ROOT"
_LOCAL_PATH_FIELDS = frozenset(
    {
        "dynamic_mask_manifest_path",
        "geometry_root_path",
        "input_video_path",
    }
)
_LOCAL_URI_FIELDS = frozenset({"input_video_uri"})


def _auth_token() -> str:
    return str(os.getenv("VIDEO_TO_WORLD_RUNNER_TOKEN") or os.getenv("PRIVACY_RUNNER_TOKEN") or "").strip()


def _require_auth_token() -> str:
    token = _auth_token()
    if not token:
        raise RuntimeError("VIDEO_TO_WORLD_RUNNER_TOKEN must be nonempty")
    return token


def _max_request_bytes() -> int:
    try:
        value = int(
            str(
                os.getenv("VIDEO_TO_WORLD_RUNNER_MAX_REQUEST_BYTES")
                or DEFAULT_MAX_REQUEST_BYTES
            )
        )
    except ValueError:
        value = DEFAULT_MAX_REQUEST_BYTES
    return max(1, min(value, 64 * 1024 * 1024))


def _validated_local_path(value: str, *, root: str) -> str:
    """Normalize an HTTP-supplied path and contain it within the configured root."""

    base_path = os.path.realpath(root)
    full_path = os.path.realpath(os.path.join(base_path, value))
    if full_path != base_path and not full_path.startswith(base_path + os.sep):
        raise ValueError("local_path_outside_allowed_root")
    return full_path


def _validated_request_body(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return an HTTP-safe body with every local filesystem path contained."""

    normalized = dict(body)
    local_values: list[tuple[str, str]] = []
    for key in _LOCAL_PATH_FIELDS:
        value = str(normalized.get(key) or "").strip()
        if value:
            local_values.append((key, value))
    for key in _LOCAL_URI_FIELDS:
        value = str(normalized.get(key) or "").strip()
        if value and not value.startswith("gs://"):
            local_values.append((key, value))
    if not local_values:
        return normalized

    root = str(os.getenv(LOCAL_PATH_ROOT_ENV) or "").strip()
    if not root:
        raise ValueError("local_path_root_not_configured")
    for key, value in local_values:
        normalized[key] = _validated_local_path(value, root=root)
    return normalized


class _Handler(BaseHTTPRequestHandler):
    server_version = "BlueprintVideoToWorldRunner/1.0"

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _authorized(self) -> bool:
        token = _auth_token()
        if not token:
            return False
        header = str(self.headers.get("Authorization") or "").strip()
        return hmac.compare_digest(
            header.encode("utf-8"),
            f"Bearer {token}".encode("utf-8"),
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/healthz"}:
            self._send_json(HTTPStatus.OK, {"status": "ok", "runner": "video_to_world"})
            log_event(
                logger,
                logging.INFO,
                "video_to_world_runner.health_checked",
                path=self.path,
                status_code=int(HTTPStatus.OK),
                runner="video_to_world",
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"status": "failed", "reason": "not_found"})
        log_event(
            logger,
            logging.WARNING,
            "video_to_world_runner.request_rejected",
            path=self.path,
            status_code=int(HTTPStatus.NOT_FOUND),
            runner="video_to_world",
            reason="not_found",
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/run", "/", "/canary"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "failed", "reason": "not_found"})
            log_event(
                logger,
                logging.WARNING,
                "video_to_world_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.NOT_FOUND),
                runner="video_to_world",
                reason="not_found",
            )
            return
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"status": "failed", "reason": "unauthorized"})
            log_event(
                logger,
                logging.WARNING,
                "video_to_world_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.UNAUTHORIZED),
                runner="video_to_world",
                reason="unauthorized",
            )
            return
        if self.path == "/canary":
            if str(self.headers.get("Content-Length") or "0").strip() not in {"", "0"}:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"status": "failed", "reason": "canary_body_forbidden"},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "authentication": "verified",
                    "runner": "video_to_world",
                    "model_execution_performed": False,
                },
            )
            log_event(
                logger,
                logging.INFO,
                "video_to_world_runner.auth_canary_passed",
                runner="video_to_world",
                model_execution_performed=False,
            )
            return
        content_length = str(self.headers.get("Content-Length") or "").strip()
        try:
            length = int(content_length or "0")
        except ValueError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "failed", "reason": "invalid_content_length"},
            )
            return
        if length < 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "failed", "reason": "invalid_content_length"},
            )
            return
        if length > _max_request_bytes():
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"status": "failed", "reason": "request_too_large"},
            )
            return
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "failed", "reason": "invalid_json"})
            log_event(
                logger,
                logging.WARNING,
                "video_to_world_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.BAD_REQUEST),
                runner="video_to_world",
                reason="invalid_json",
            )
            return
        if not isinstance(body, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "failed", "reason": "invalid_payload"})
            log_event(
                logger,
                logging.WARNING,
                "video_to_world_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.BAD_REQUEST),
                runner="video_to_world",
                reason="invalid_payload",
            )
            return
        try:
            body = _validated_request_body(body)
        except ValueError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "failed", "reason": str(exc)},
            )
            log_event(
                logger,
                logging.WARNING,
                "video_to_world_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.BAD_REQUEST),
                runner="video_to_world",
                reason=str(exc),
            )
            return
        payload = execute_video_to_world_request(body)
        status = HTTPStatus.OK if str(payload.get("status") or "").lower() == "succeeded" else HTTPStatus.BAD_GATEWAY
        self._send_json(status, payload)
        log_event(
            logger,
            logging.INFO if status == HTTPStatus.OK else logging.WARNING,
            "video_to_world_runner.request_completed",
            path=self.path,
            status_code=int(status),
            runner="video_to_world",
            result_status=payload.get("status"),
            reason=payload.get("reason"),
        )


def main() -> int:
    _require_auth_token()
    port_raw = str(os.getenv("PORT") or "8080").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 8080
    host = str(
        os.getenv("VIDEO_TO_WORLD_RUNNER_HOST")
        or ("0.0.0.0" if os.getenv("K_SERVICE") else "127.0.0.1")
    ).strip()
    server = ThreadingHTTPServer((host, port), _Handler)
    log_event(
        logger,
        logging.INFO,
        "video_to_world_runner.service_started",
        host=host,
        port=port,
        runner="video_to_world",
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
