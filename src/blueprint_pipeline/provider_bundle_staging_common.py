"""Provider-neutral token, URL, and local server primitives for bundle staging."""

from __future__ import annotations

import http.server
import json
import secrets
import shutil
import socketserver
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from .core.common import ensure_dir
from .secret_artifact_policy import redacted_secret_file_status


BUNDLE_ROUTE = "/bundle.zip"
OUTPUT_ROUTE = "/output.zip"
HEALTH_ROUTE = "/health"
DEFAULT_STAGING_HOST = "127.0.0.1"
DEFAULT_MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024


def read_or_create_staging_token(path: Path) -> tuple[str, dict[str, Any]]:
    """Load or create a mode-0600 staging token without returning it in metadata."""

    ensure_dir(path.parent)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        created = False
    else:
        token = secrets.token_urlsafe(32)
        path.write_text(token + "\n", encoding="utf-8")
        created = True
    path.chmod(0o600)
    mode = oct(path.stat().st_mode & 0o777)
    status = redacted_secret_file_status(
        path,
        path_source="staging_token_file",
        raw_secret_field="token_recorded_in_manifest",
    )
    status.update(
        {
            "created": created,
            "present": path.is_file(),
            "mode": mode,
            "mode_is_0600": mode == "0o600",
            "token_recorded_in_manifest": False,
        }
    )
    return token, status


def staging_url_with_token(base_url: str, route: str, token: str) -> str:
    """Attach a staging token while normalizing the provider-neutral route."""

    parsed = urlparse(base_url)
    clean_path = "/" + route.strip("/")
    query = urlencode({"token": token})
    return urlunparse((parsed.scheme, parsed.netloc, clean_path, "", query, ""))


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ProviderBundleStagingRequestHandler(http.server.BaseHTTPRequestHandler):
    """Serve one token-gated input bundle and accept one bounded output archive."""

    server_version = "BlueprintProviderBundleStaging/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None

    @property
    def _bundle_path(self) -> Path:
        return self.server.bundle_path  # type: ignore[attr-defined]

    @property
    def _output_path(self) -> Path:
        return self.server.output_path  # type: ignore[attr-defined]

    @property
    def _token(self) -> str:
        return self.server.token  # type: ignore[attr-defined]

    @property
    def _max_output_bytes(self) -> int:
        return self.server.max_output_bytes  # type: ignore[attr-defined]

    def _authorized(self) -> bool:
        token_values = parse_qs(urlparse(self.path).query).get("token") or []
        return bool(
            self._token and token_values and secrets.compare_digest(token_values[0], self._token)
        )

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(dict(payload), sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _serve_bundle(self, *, head_only: bool) -> None:
        if not self._authorized():
            self._send_json(403, {"ok": False, "error": "forbidden"})
            return
        if not self._bundle_path.is_file():
            self._send_json(404, {"ok": False, "error": "bundle_missing"})
            return
        size = self._bundle_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if not head_only:
            with self._bundle_path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)

    def do_HEAD(self) -> None:
        if urlparse(self.path).path == BUNDLE_ROUTE:
            self._serve_bundle(head_only=True)
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == HEALTH_ROUTE:
            self._send_json(
                200,
                {
                    "ok": True,
                    "bundle_present": self._bundle_path.is_file(),
                    "output_parent_present": self._output_path.parent.is_dir(),
                },
            )
            return
        if path == BUNDLE_ROUTE:
            self._serve_bundle(head_only=False)
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_PUT(self) -> None:
        if urlparse(self.path).path != OUTPUT_ROUTE:
            self._send_json(404, {"ok": False, "error": "not_found"})
            return
        if not self._authorized():
            self._send_json(403, {"ok": False, "error": "forbidden"})
            return
        content_length = int(self.headers.get("Content-Length") or "0")
        if content_length <= 0:
            self._send_json(400, {"ok": False, "error": "empty_upload"})
            return
        if content_length > self._max_output_bytes:
            self._send_json(413, {"ok": False, "error": "upload_too_large"})
            return
        ensure_dir(self._output_path.parent)
        temp_path = self._output_path.with_suffix(self._output_path.suffix + ".tmp")
        remaining = content_length
        with temp_path.open("wb") as handle:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        if remaining:
            temp_path.unlink(missing_ok=True)
            self._send_json(400, {"ok": False, "error": "short_upload"})
            return
        temp_path.replace(self._output_path)
        self._send_json(
            200,
            {
                "ok": True,
                "output_path": str(self._output_path),
                "bytes_written": content_length,
            },
        )


def create_staging_server(
    *,
    bundle_path: str | Path,
    output_path: str | Path,
    token: str,
    host: str = DEFAULT_STAGING_HOST,
    port: int = 0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> http.server.HTTPServer:
    """Build a provider-neutral local staging server without starting its thread."""

    server = _ThreadingHTTPServer((host, port), ProviderBundleStagingRequestHandler)
    server.bundle_path = Path(bundle_path).expanduser().resolve()  # type: ignore[attr-defined]
    server.output_path = Path(output_path).expanduser().resolve()  # type: ignore[attr-defined]
    server.token = token  # type: ignore[attr-defined]
    server.max_output_bytes = max_output_bytes  # type: ignore[attr-defined]
    return server
