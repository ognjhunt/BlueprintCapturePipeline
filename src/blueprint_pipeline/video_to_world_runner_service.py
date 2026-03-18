"""HTTP service wrapper for video_to_world execution."""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .video_to_world_service_runtime import execute_video_to_world_request


def _auth_token() -> str:
    return str(os.getenv("VIDEO_TO_WORLD_RUNNER_TOKEN") or os.getenv("PRIVACY_RUNNER_TOKEN") or "").strip()


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
            return True
        header = str(self.headers.get("Authorization") or "").strip()
        return header == f"Bearer {token}"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/healthz"}:
            self._send_json(HTTPStatus.OK, {"status": "ok", "runner": "video_to_world"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"status": "failed", "reason": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/run", "/"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "failed", "reason": "not_found"})
            return
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"status": "failed", "reason": "unauthorized"})
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
            return
        if not isinstance(body, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "failed", "reason": "invalid_payload"})
            return
        payload = execute_video_to_world_request(body)
        status = HTTPStatus.OK if str(payload.get("status") or "").lower() == "succeeded" else HTTPStatus.BAD_GATEWAY
        self._send_json(status, payload)


def main() -> int:
    port_raw = str(os.getenv("PORT") or "8080").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
