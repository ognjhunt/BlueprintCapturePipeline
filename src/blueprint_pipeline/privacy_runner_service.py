"""HTTP service wrapper for concrete privacy backends."""

from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .logging_utils import log_event
from .privacy_service_runtime import execute_privacy_service_request


logger = logging.getLogger(__name__)


def _service_kind() -> str:
    return str(os.getenv("PRIVACY_RUNNER_KIND") or "").strip().lower()


def _auth_token() -> str:
    return str(os.getenv("PRIVACY_RUNNER_TOKEN") or "").strip()


class _Handler(BaseHTTPRequestHandler):
    server_version = "BlueprintPrivacyRunner/2.0"

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
            return True
        header = str(self.headers.get("Authorization") or "").strip()
        return header == f"Bearer {token}"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/healthz"}:
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "runner_kind": _service_kind()},
            )
            log_event(
                logger,
                logging.INFO,
                "privacy_runner.health_checked",
                path=self.path,
                status_code=int(HTTPStatus.OK),
                runner_kind=_service_kind(),
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"status": "failed", "reason": "not_found"})
        log_event(
            logger,
            logging.WARNING,
            "privacy_runner.request_rejected",
            path=self.path,
            status_code=int(HTTPStatus.NOT_FOUND),
            runner_kind=_service_kind(),
            reason="not_found",
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/", "/run"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "failed", "reason": "not_found"})
            log_event(
                logger,
                logging.WARNING,
                "privacy_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.NOT_FOUND),
                runner_kind=_service_kind(),
                reason="not_found",
            )
            return
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"status": "failed", "reason": "unauthorized"})
            log_event(
                logger,
                logging.WARNING,
                "privacy_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.UNAUTHORIZED),
                runner_kind=_service_kind(),
                reason="unauthorized",
            )
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "failed", "reason": "invalid_json"})
            log_event(
                logger,
                logging.WARNING,
                "privacy_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.BAD_REQUEST),
                runner_kind=_service_kind(),
                reason="invalid_json",
            )
            return
        if not isinstance(body, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "failed", "reason": "invalid_payload"})
            log_event(
                logger,
                logging.WARNING,
                "privacy_runner.request_rejected",
                path=self.path,
                status_code=int(HTTPStatus.BAD_REQUEST),
                runner_kind=_service_kind(),
                reason="invalid_payload",
            )
            return

        payload = execute_privacy_service_request(_service_kind(), body)
        status = HTTPStatus.OK if str(payload.get("status") or "").lower() == "succeeded" else HTTPStatus.BAD_GATEWAY
        self._send_json(status, payload)
        log_event(
            logger,
            logging.INFO if status == HTTPStatus.OK else logging.WARNING,
            "privacy_runner.request_completed",
            path=self.path,
            status_code=int(status),
            runner_kind=_service_kind(),
            result_status=payload.get("status"),
            reason=payload.get("reason"),
        )


def main() -> int:
    port_raw = str(os.getenv("PORT") or "8080").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    log_event(
        logger,
        logging.INFO,
        "privacy_runner.service_started",
        host="0.0.0.0",
        port=port,
        runner_kind=_service_kind(),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
