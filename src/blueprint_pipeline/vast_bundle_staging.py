"""Token-gated staging helper for Vast Blueprint provider bundle runs.

The live Vast adapter needs a provider-fetchable bundle URL and a writable PUT
URL for the returned runtime zip. This module keeps that staging contract
separate from the paid Vast launch path so URL readiness can be proven before a
GPU instance is created.
"""

from __future__ import annotations

import argparse
import http.server
import json
import re
import secrets
import shutil
import socketserver
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse, urlunparse

from .common import ensure_dir, utc_now_iso, write_json
from .provider_bundle_staging_common import (
    BUNDLE_ROUTE,
    HEALTH_ROUTE,
    OUTPUT_ROUTE,
    read_or_create_staging_token as _read_or_create_token,
    staging_url_with_token as _url_with_token,
)
from .secret_artifact_policy import (
    redacted_secret_file_status,
    secret_path_disclosure_policy,
)


VAST_BUNDLE_STAGING_SCHEMA_VERSION = "vast_bundle_staging_manifest.v1"
VAST_BUNDLE_STAGING_SELF_TEST_SCHEMA_VERSION = "vast_bundle_staging_self_test.v1"
VAST_PUBLIC_STAGING_VERIFICATION_SCHEMA_VERSION = "vast_public_staging_verification.v1"
VAST_CLOUDFLARED_TUNNEL_SCHEMA_VERSION = "vast_cloudflared_tunnel_manifest.v1"
DEFAULT_TOKEN_FILE = "~/.blueprint-secrets/vast_bundle_staging_token"
DEFAULT_SECRET_ENV_FILE = "~/.blueprint-secrets/vast_bundle_staging_urls.env"
DEFAULT_OUTPUT_FILENAME = "vast_provider_runtime_output.zip"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_PUBLIC_VERIFY_MAX_WAIT_SECONDS = 120
DEFAULT_PUBLIC_VERIFY_RETRY_INTERVAL_SECONDS = 5.0
DEFAULT_PUBLIC_VERIFY_TIMEOUT_SECONDS = 20
REDACTED_TOKEN = "<redacted-token>"
CLOUDFLARED_URL_RE = re.compile(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com")


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _redacted_url_path(route: str) -> str:
    return f"/{route.strip('/')}?token={REDACTED_TOKEN}"


def _redact_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.query:
        return value
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            "",
            "REDACTED_QUERY",
            "",
        )
    )


def start_cloudflared_tunnel(
    *,
    job_dir: str | Path,
    local_base_url: str,
    cloudflared_path: str | Path | None = None,
    startup_timeout_seconds: float = 45.0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Start a temporary cloudflared tunnel and persist the public base URL.

    The process intentionally stays running after this function returns. Callers
    must stop the recorded pid after the provider run, and should verify the
    returned URL with ``verify_public_staging_urls`` before any paid launch.
    """

    resolved_job_dir = Path(job_dir).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    executable_hint = _string(str(cloudflared_path)) if cloudflared_path else "cloudflared"
    executable = shutil.which(executable_hint) or str(Path(executable_hint).expanduser())
    blockers: list[str] = []
    if not executable or not Path(executable).exists():
        blockers.append("cloudflared_binary_missing")
    if not _string(local_base_url):
        blockers.append("local_base_url_missing")
    elif urlparse(local_base_url).scheme not in {"http", "https"}:
        blockers.append("local_base_url_scheme_not_http")
    if blockers:
        manifest = {
            "schema_version": VAST_CLOUDFLARED_TUNNEL_SCHEMA_VERSION,
            "generated_at": generated_at or utc_now_iso(),
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "local_base_url": local_base_url,
            "cloudflared_path": executable,
            "public_base_url": None,
            "pid": None,
            "blockers": blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_cloudflared_tunnel_manifest.json", manifest)
        return manifest

    assert executable is not None
    log_path = resolved_job_dir / "vast_cloudflared_tunnel.log"
    argv = [executable, "tunnel", "--url", local_base_url, "--no-autoupdate"]
    public_base_url = ""
    observed_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603 - executable is discovered or explicit.
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    deadline = time.monotonic() + max(1.0, float(startup_timeout_seconds))
    while time.monotonic() < deadline:
        try:
            log_text = log_path.read_text(encoding="utf-8")
        except Exception:
            log_text = ""
        observed_lines = log_text.splitlines()
        match = CLOUDFLARED_URL_RE.search(log_text)
        if match:
            public_base_url = match.group(0)
            break
        if process.poll() is not None:
            blockers.append("cloudflared_process_exited_before_public_url")
            break
        time.sleep(0.25)
    if not public_base_url and not blockers:
        blockers.append("cloudflared_public_url_not_observed_before_timeout")
    if blockers:
        try:
            process.terminate()
        except Exception:
            pass
    status = "running" if public_base_url and not blockers else "blocked"
    manifest = {
        "schema_version": VAST_CLOUDFLARED_TUNNEL_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": status,
        "job_dir": str(resolved_job_dir),
        "local_base_url": local_base_url,
        "public_base_url": public_base_url or None,
        "pid": process.pid if status == "running" else None,
        "cloudflared_path": executable,
        "command": ["cloudflared", "tunnel", "--url", "<local_base_url>", "--no-autoupdate"],
        "log_path": str(log_path),
        "observed_line_count": len(observed_lines),
        "blockers": blockers,
        "cleanup_command": f"kill {process.pid}" if status == "running" else None,
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "vast_cloudflared_tunnel_manifest.json", manifest)
    return manifest


def _write_secret_env_file(
    *,
    path: Path,
    provider_bundle_url: str,
    provider_output_put_url: str,
) -> dict[str, Any]:
    ensure_dir(path.parent)
    path.write_text(
        "\n".join(
            [
                f"BLUEPRINT_EVAL_MANIFEST_URI={provider_bundle_url}",
                f"BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL={provider_output_put_url}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    status = redacted_secret_file_status(
        path,
        path_source="staging_secret_env_file",
        raw_secret_field="raw_secret_values_recorded_in_manifest",
    )
    status.update({
        "present": path.is_file(),
        "mode": oct(path.stat().st_mode & 0o777),
        "mode_is_0600": (path.stat().st_mode & 0o777) == 0o600,
        "raw_secret_values_recorded_in_manifest": False,
    })
    return status


def prepare_vast_bundle_staging(
    *,
    job_dir: str | Path,
    bundle_path: str | Path,
    public_base_url: str | None = None,
    token_file: str | Path | None = None,
    secret_env_file: str | Path | None = None,
    output_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve()
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_job_dir / DEFAULT_OUTPUT_FILENAME
    )
    resolved_token_file = (
        Path(token_file).expanduser().resolve()
        if token_file
        else Path(DEFAULT_TOKEN_FILE).expanduser().resolve()
    )
    resolved_secret_env_file = (
        Path(secret_env_file).expanduser().resolve()
        if secret_env_file
        else Path(DEFAULT_SECRET_ENV_FILE).expanduser().resolve()
    )
    ensure_dir(resolved_job_dir)
    token, token_status = _read_or_create_token(resolved_token_file)
    base_url = _string(public_base_url)
    provider_bundle_url = _url_with_token(base_url, BUNDLE_ROUTE, token) if base_url else ""
    provider_output_put_url = _url_with_token(base_url, OUTPUT_ROUTE, token) if base_url else ""
    blockers: list[str] = []
    warnings: list[str] = []
    if not resolved_bundle.is_file():
        blockers.append("provider_runtime_bundle_missing")
    if not base_url:
        blockers.append("public_base_url_missing")
    elif urlparse(base_url).scheme not in {"http", "https"}:
        blockers.append("public_base_url_scheme_not_http")
    elif urlparse(base_url).hostname in {"127.0.0.1", "localhost"}:
        warnings.append("local_base_url_not_provider_fetchable_from_vast")
    if not token:
        blockers.append("staging_token_missing")
    secret_env_status = None
    if provider_bundle_url and provider_output_put_url:
        secret_env_status = _write_secret_env_file(
            path=resolved_secret_env_file,
            provider_bundle_url=provider_bundle_url,
            provider_output_put_url=provider_output_put_url,
        )
    bundle_zip_entry_count = 0
    bundle_zip_parse_error = None
    bundle_zip_testzip_result = None
    if resolved_bundle.is_file():
        try:
            with zipfile.ZipFile(resolved_bundle) as archive:
                bundle_zip_entry_count = len(archive.namelist())
                bundle_zip_testzip_result = archive.testzip()
        except Exception as exc:
            bundle_zip_parse_error = f"{type(exc).__name__}:{str(exc)[:300]}"
            blockers.append(f"provider_runtime_bundle_zip_inspection_failed:{type(exc).__name__}")
    if bundle_zip_testzip_result is not None:
        blockers.append("provider_runtime_bundle_zip_integrity_failed")
    cloudflared_path = shutil.which("cloudflared")
    manifest = {
        "schema_version": VAST_BUNDLE_STAGING_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "ready" if not blockers else "blocked",
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(resolved_bundle),
        "bundle_present": resolved_bundle.is_file(),
        "bundle_size_bytes": resolved_bundle.stat().st_size if resolved_bundle.is_file() else 0,
        "bundle_zip_entry_count": bundle_zip_entry_count,
        "bundle_zip_parse_error": bundle_zip_parse_error,
        "bundle_zip_testzip_result": bundle_zip_testzip_result,
        "bundle_zip_integrity_passed": (
            resolved_bundle.is_file()
            and bundle_zip_parse_error is None
            and bundle_zip_testzip_result is None
        ),
        "output_path": str(resolved_output),
        "token_file": token_status,
        "secret_env_file": secret_env_status,
        "secret_artifact_policy": secret_path_disclosure_policy(),
        "base_url_redacted": _redact_url(base_url) if base_url else None,
        "bundle_url_path": _redacted_url_path(BUNDLE_ROUTE),
        "output_put_url_path": _redacted_url_path(OUTPUT_ROUTE),
        "provider_bundle_url_present": bool(provider_bundle_url),
        "provider_output_put_url_present": bool(provider_output_put_url),
        "provider_fetchable_bundle_uri_ready": bool(provider_bundle_url and not blockers),
        "provider_output_callback_ready": bool(provider_output_put_url and not blockers),
        "cloudflared_available": bool(cloudflared_path),
        "cloudflared_path": cloudflared_path,
        "suggested_local_server": {
            "host": DEFAULT_HOST,
            "bundle_route": BUNDLE_ROUTE,
            "output_route": OUTPUT_ROUTE,
            "health_route": HEALTH_ROUTE,
        },
        "privacy_boundary": (
            "Token-gated staging for a bounded provider run only. Raw tokenized URLs are "
            "written only to the chmod 600 secret env file when a base URL is supplied."
        ),
        "blockers": blockers,
        "warnings": warnings,
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "vast_bundle_staging_manifest.json", manifest)
    return manifest


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class VastBundleStagingRequestHandler(http.server.BaseHTTPRequestHandler):
    server_version = "BlueprintVastBundleStaging/1.0"

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
        parsed = urlparse(self.path)
        token_values = parse_qs(parsed.query).get("token") or []
        return bool(self._token and token_values and secrets.compare_digest(token_values[0], self._token))

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
    host: str = DEFAULT_HOST,
    port: int = 0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> http.server.HTTPServer:
    server = _ThreadingHTTPServer((host, port), VastBundleStagingRequestHandler)
    server.bundle_path = Path(bundle_path).expanduser().resolve()  # type: ignore[attr-defined]
    server.output_path = Path(output_path).expanduser().resolve()  # type: ignore[attr-defined]
    server.token = token  # type: ignore[attr-defined]
    server.max_output_bytes = max_output_bytes  # type: ignore[attr-defined]
    return server


def run_local_staging_self_test(
    *,
    job_dir: str | Path,
    bundle_path: str | Path,
    output_path: str | Path | None = None,
    token_file: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve()
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_job_dir / "vast_staging_self_test_output.zip"
    )
    token_path = (
        Path(token_file).expanduser().resolve()
        if token_file
        else Path(DEFAULT_TOKEN_FILE).expanduser().resolve()
    )
    token, token_status = _read_or_create_token(token_path)
    server = create_staging_server(
        bundle_path=resolved_bundle,
        output_path=resolved_output,
        token=token,
    )
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        health = urllib.request.urlopen(f"{base_url}{HEALTH_ROUTE}", timeout=10)
        checks.append({"label": "health", "http_status": health.status})
        bundle_url = _url_with_token(base_url, BUNDLE_ROUTE, token)
        head_request = urllib.request.Request(bundle_url, method="HEAD")
        head = urllib.request.urlopen(head_request, timeout=10)
        content_length = int(head.headers.get("Content-Length") or "0")
        checks.append(
            {
                "label": "bundle_head",
                "http_status": head.status,
                "content_length": content_length,
                "content_type": head.headers.get("Content-Type"),
            }
        )
        if resolved_bundle.is_file() and content_length != resolved_bundle.stat().st_size:
            blockers.append("bundle_head_content_length_mismatch")
        upload_bytes = b"PK\x05\x06" + (b"\x00" * 18)
        put_request = urllib.request.Request(
            _url_with_token(base_url, OUTPUT_ROUTE, token),
            data=upload_bytes,
            method="PUT",
            headers={"Content-Type": "application/zip"},
        )
        put = urllib.request.urlopen(put_request, timeout=10)
        checks.append({"label": "output_put", "http_status": put.status})
        if not resolved_output.is_file() or resolved_output.read_bytes() != upload_bytes:
            blockers.append("output_put_file_not_written")
    except Exception as exc:
        blockers.append(f"local_staging_self_test_failed:{type(exc).__name__}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    manifest = {
        "schema_version": VAST_BUNDLE_STAGING_SELF_TEST_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "passed" if not blockers else "blocked",
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(resolved_bundle),
        "output_path": str(resolved_output),
        "token_file": token_status,
        "secret_artifact_policy": secret_path_disclosure_policy(),
        "local_base_url": base_url,
        "provider_public_base_url_ready": False,
        "provider_public_base_url_blocker": "public_tunnel_not_started",
        "checks": checks,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "vast_bundle_staging_self_test.json", manifest)
    return manifest


def _probe_exception(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {"error_type": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        payload["http_status_code"] = exc.code
    elif isinstance(exc, urllib.error.URLError):
        payload["reason_type"] = type(exc.reason).__name__
    return payload


def _head_bundle_url(
    *,
    bundle_url: str,
    bundle_path: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        request = urllib.request.Request(bundle_url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(getattr(response, "status", 200))
            headers = dict(response.headers.items())
        content_length_text = headers.get("Content-Length")
        content_length = int(content_length_text) if content_length_text else None
        probe = {
            "status": "passed" if 200 <= status_code < 300 else "blocked",
            "method": "HEAD",
            "http_status_code": status_code,
            "content_type": headers.get("Content-Type"),
            "content_length": content_length,
            "expected_content_length": (
                bundle_path.stat().st_size
                if bundle_path and bundle_path.is_file()
                else None
            ),
        }
        if not (200 <= status_code < 300):
            probe["blocker"] = "provider_bundle_fetch_url_unreachable"
        elif (
            bundle_path
            and bundle_path.is_file()
            and content_length is not None
            and content_length != bundle_path.stat().st_size
        ):
            probe["status"] = "blocked"
            probe["blocker"] = "provider_bundle_fetch_url_size_mismatch"
        return probe
    except Exception as exc:
        return {
            "status": "blocked",
            "method": "HEAD",
            "blocker": "provider_bundle_fetch_url_unreachable",
            **_probe_exception(exc),
        }


def _put_output_probe(
    *,
    output_put_url: str,
    timeout_seconds: float,
    probe_zip: bytes,
) -> dict[str, Any]:
    try:
        request = urllib.request.Request(
            output_put_url,
            data=probe_zip,
            method="PUT",
            headers={"Content-Type": "application/zip"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(getattr(response, "status", 200))
            response_text = response.read().decode("utf-8", errors="replace")
        probe = {
            "status": "passed" if 200 <= status_code < 300 else "blocked",
            "method": "PUT",
            "http_status_code": status_code,
            "probe_bytes": len(probe_zip),
            "response_preview": response_text[:200],
        }
        if not (200 <= status_code < 300):
            probe["blocker"] = "provider_output_put_url_unwritable"
        return probe
    except Exception as exc:
        return {
            "status": "blocked",
            "method": "PUT",
            "probe_bytes": len(probe_zip),
            "blocker": "provider_output_put_url_unwritable",
            **_probe_exception(exc),
        }


def _cleanup_output_probe(
    *,
    output_path: Path | None,
    probe_zip: bytes,
    cleanup_output_probe: bool,
) -> dict[str, Any]:
    if not cleanup_output_probe:
        return {"status": "skipped", "reason": "cleanup_output_probe_false"}
    if output_path is None:
        return {"status": "skipped", "reason": "output_path_missing"}
    if not output_path.exists():
        return {"status": "skipped", "reason": "output_probe_file_not_present"}
    try:
        data = output_path.read_bytes()
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": "output_probe_file_read_failed",
            "error_type": type(exc).__name__,
        }
    if data != probe_zip:
        return {
            "status": "skipped",
            "reason": "output_path_does_not_match_probe_bytes",
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        }
    try:
        output_path.unlink()
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": "output_probe_file_cleanup_failed",
            "error_type": type(exc).__name__,
        }
    return {"status": "removed", "path": str(output_path), "removed_bytes": len(probe_zip)}


def verify_public_staging_urls(
    *,
    job_dir: str | Path,
    provider_bundle_url: str,
    provider_output_put_url: str,
    bundle_path: str | Path | None = None,
    output_path: str | Path | None = None,
    max_wait_seconds: int = DEFAULT_PUBLIC_VERIFY_MAX_WAIT_SECONDS,
    retry_interval_seconds: float = DEFAULT_PUBLIC_VERIFY_RETRY_INTERVAL_SECONDS,
    timeout_seconds: float = DEFAULT_PUBLIC_VERIFY_TIMEOUT_SECONDS,
    required_consecutive_successes: int = 1,
    allow_output_put_probe: bool = True,
    cleanup_output_probe: bool = True,
    require_bundle_fetch_probe: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve() if bundle_path else None
    resolved_output = Path(output_path).expanduser().resolve() if output_path else None
    ensure_dir(resolved_job_dir)
    bundle_url = _string(provider_bundle_url)
    output_put_url = _string(provider_output_put_url)
    attempts: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if not bundle_url:
        blockers.append("provider_bundle_fetch_url_missing")
    if not output_put_url:
        blockers.append("provider_output_put_url_missing")
    if blockers:
        manifest = {
            "schema_version": VAST_PUBLIC_STAGING_VERIFICATION_SCHEMA_VERSION,
            "generated_at": generated_at or utc_now_iso(),
            "completed_at": utc_now_iso(),
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "provider_bundle_url_redacted": _redact_url(bundle_url) if bundle_url else None,
            "provider_output_put_url_redacted": _redact_url(output_put_url)
            if output_put_url
            else None,
            "attempt_count": 0,
            "attempts": attempts,
            "blockers": blockers,
            "warnings": warnings,
            "output_probe_cleanup": {"status": "not_requested"},
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / "vast_public_staging_verification.json", manifest)
        return manifest

    probe_zip = b"PK\x05\x06" + (b"\x00" * 18)
    started_at = time.monotonic()
    deadline = started_at + max(0, max_wait_seconds)
    final_cleanup: dict[str, Any] = {"status": "not_requested"}
    final_status = "blocked"
    attempt_number = 0
    required_successes = max(1, int(required_consecutive_successes))
    consecutive_successes = 0
    successful_attempts = 0
    while True:
        attempt_number += 1
        if require_bundle_fetch_probe:
            bundle_probe = _head_bundle_url(
                bundle_url=bundle_url,
                bundle_path=resolved_bundle,
                timeout_seconds=timeout_seconds,
            )
        else:
            bundle_probe = {
                "status": "skipped",
                "method": "HEAD",
                "reason": "bundle_fetch_replaced_by_inline_transport",
            }
        if allow_output_put_probe and (
            bundle_probe.get("status") == "passed" or not require_bundle_fetch_probe
        ):
            output_probe = _put_output_probe(
                output_put_url=output_put_url,
                timeout_seconds=timeout_seconds,
                probe_zip=probe_zip,
            )
            if output_probe.get("status") == "passed":
                final_cleanup = _cleanup_output_probe(
                    output_path=resolved_output,
                    probe_zip=probe_zip,
                    cleanup_output_probe=cleanup_output_probe,
                )
                if final_cleanup.get("status") == "blocked":
                    output_probe = {
                        **output_probe,
                        "status": "blocked",
                        "blocker": final_cleanup.get("reason")
                        or "output_probe_cleanup_failed",
                    }
        elif allow_output_put_probe:
            output_probe = {
                "status": "blocked",
                "method": "PUT",
                "reason": "bundle_probe_must_pass_before_output_put_probe",
                "blocker": "provider_output_put_url_not_checked",
            }
        else:
            output_probe = {
                "status": "skipped",
                "method": "PUT",
                "reason": "output_put_probe_requires_explicit_allow",
            }
            warnings.append("provider_output_put_url_not_mutation_probed")
        attempt_passed = bundle_probe.get("status") == "passed" and (
            output_probe.get("status") == "passed" or not allow_output_put_probe
        )
        if not require_bundle_fetch_probe:
            attempt_passed = (
                output_probe.get("status") == "passed" or not allow_output_put_probe
            )
        if attempt_passed:
            successful_attempts += 1
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        attempts.append(
            {
                "attempt": attempt_number,
                "checked_at": utc_now_iso(),
                "bundle_probe": bundle_probe,
                "output_put_probe": output_probe,
                "output_probe_cleanup": final_cleanup,
                "passed": attempt_passed,
                "consecutive_successes_after_attempt": consecutive_successes,
            }
        )
        if consecutive_successes >= required_successes:
            final_status = "passed"
            blockers = []
            break
        now = time.monotonic()
        if now >= deadline:
            blockers = []
            bundle_blocker = bundle_probe.get("blocker")
            output_blocker = output_probe.get("blocker")
            if require_bundle_fetch_probe and isinstance(bundle_blocker, str):
                blockers.append(bundle_blocker)
            if allow_output_put_probe and isinstance(output_blocker, str):
                blockers.append(output_blocker)
            if not blockers:
                blockers.append(
                    "public_staging_url_stability_not_proven"
                    if successful_attempts
                    else "public_staging_url_verification_failed"
                )
            break
        time.sleep(min(max(0.0, retry_interval_seconds), max(0.0, deadline - now)))

    manifest = {
        "schema_version": VAST_PUBLIC_STAGING_VERIFICATION_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "completed_at": utc_now_iso(),
        "status": final_status,
        "job_dir": str(resolved_job_dir),
        "provider_bundle_url_redacted": _redact_url(bundle_url),
        "provider_output_put_url_redacted": _redact_url(output_put_url),
        "bundle_path": str(resolved_bundle) if resolved_bundle else None,
        "output_path": str(resolved_output) if resolved_output else None,
        "max_wait_seconds": max_wait_seconds,
        "retry_interval_seconds": retry_interval_seconds,
        "timeout_seconds": timeout_seconds,
        "required_consecutive_successes": required_successes,
        "require_bundle_fetch_probe": require_bundle_fetch_probe,
        "successful_attempt_count": successful_attempts,
        "final_consecutive_success_count": consecutive_successes,
        "allow_output_put_probe": allow_output_put_probe,
        "cleanup_output_probe": cleanup_output_probe,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "blockers": blockers,
        "warnings": sorted(set(warnings)),
        "output_probe_cleanup": final_cleanup,
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / "vast_public_staging_verification.json", manifest)
    return manifest


def serve_vast_bundle_staging(
    *,
    bundle_path: str | Path,
    output_path: str | Path,
    token_file: str | Path,
    host: str = DEFAULT_HOST,
    port: int = 8765,
) -> None:
    token, _status = _read_or_create_token(Path(token_file).expanduser().resolve())
    server = create_staging_server(
        bundle_path=bundle_path,
        output_path=output_path,
        token=token,
        host=host,
        port=port,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or serve Vast provider bundle staging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--job-dir", required=True)
    prepare.add_argument("--bundle-path", required=True)
    prepare.add_argument("--public-base-url")
    prepare.add_argument("--token-file")
    prepare.add_argument("--secret-env-file")
    prepare.add_argument("--output-path")

    self_test = subparsers.add_parser("self-test")
    self_test.add_argument("--job-dir", required=True)
    self_test.add_argument("--bundle-path", required=True)
    self_test.add_argument("--output-path")
    self_test.add_argument("--token-file")

    verify_public = subparsers.add_parser("verify-public")
    verify_public.add_argument("--job-dir", required=True)
    verify_public.add_argument("--bundle-path", required=True)
    verify_public.add_argument("--public-base-url", required=True)
    verify_public.add_argument("--token-file", required=True)
    verify_public.add_argument("--output-path")
    verify_public.add_argument(
        "--max-wait-seconds",
        type=int,
        default=DEFAULT_PUBLIC_VERIFY_MAX_WAIT_SECONDS,
    )
    verify_public.add_argument(
        "--retry-interval-seconds",
        type=float,
        default=DEFAULT_PUBLIC_VERIFY_RETRY_INTERVAL_SECONDS,
    )
    verify_public.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_PUBLIC_VERIFY_TIMEOUT_SECONDS,
    )
    verify_public.add_argument(
        "--required-consecutive-successes",
        type=int,
        default=1,
    )
    verify_public.add_argument("--no-output-put-probe", action="store_true")
    verify_public.add_argument("--no-cleanup-output-probe", action="store_true")

    cloudflared = subparsers.add_parser("start-cloudflared")
    cloudflared.add_argument("--job-dir", required=True)
    cloudflared.add_argument("--local-base-url", required=True)
    cloudflared.add_argument("--cloudflared-path")
    cloudflared.add_argument("--startup-timeout-seconds", type=float, default=45.0)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--bundle-path", required=True)
    serve.add_argument("--output-path", required=True)
    serve.add_argument("--token-file", required=True)
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_vast_bundle_staging(
            job_dir=args.job_dir,
            bundle_path=args.bundle_path,
            public_base_url=args.public_base_url,
            token_file=args.token_file,
            secret_env_file=args.secret_env_file,
            output_path=args.output_path,
        )
        print(f"[vast-bundle-staging] manifest={Path(args.job_dir).resolve() / 'vast_bundle_staging_manifest.json'}")
        print(f"[vast-bundle-staging] status={manifest.get('status')}")
        blockers = manifest.get("blockers") or []
        if blockers:
            print("[vast-bundle-staging] blockers=" + ",".join(str(item) for item in blockers))
        return 0 if manifest.get("status") == "ready" else 1
    if args.command == "self-test":
        manifest = run_local_staging_self_test(
            job_dir=args.job_dir,
            bundle_path=args.bundle_path,
            output_path=args.output_path,
            token_file=args.token_file,
        )
        print(f"[vast-bundle-staging] self_test={Path(args.job_dir).resolve() / 'vast_bundle_staging_self_test.json'}")
        print(f"[vast-bundle-staging] status={manifest.get('status')}")
        return 0 if manifest.get("status") == "passed" else 1
    if args.command == "verify-public":
        token, _token_status = _read_or_create_token(Path(args.token_file).expanduser().resolve())
        manifest = verify_public_staging_urls(
            job_dir=args.job_dir,
            provider_bundle_url=_url_with_token(args.public_base_url, BUNDLE_ROUTE, token),
            provider_output_put_url=_url_with_token(args.public_base_url, OUTPUT_ROUTE, token),
            bundle_path=args.bundle_path,
            output_path=args.output_path,
            max_wait_seconds=args.max_wait_seconds,
            retry_interval_seconds=args.retry_interval_seconds,
            timeout_seconds=args.timeout_seconds,
            required_consecutive_successes=args.required_consecutive_successes,
            allow_output_put_probe=not args.no_output_put_probe,
            cleanup_output_probe=not args.no_cleanup_output_probe,
        )
        print(f"[vast-bundle-staging] public_verification={Path(args.job_dir).resolve() / 'vast_public_staging_verification.json'}")
        print(f"[vast-bundle-staging] status={manifest.get('status')}")
        blockers = manifest.get("blockers") or []
        if blockers:
            print("[vast-bundle-staging] blockers=" + ",".join(str(item) for item in blockers))
        return 0 if manifest.get("status") == "passed" else 1
    if args.command == "start-cloudflared":
        manifest = start_cloudflared_tunnel(
            job_dir=args.job_dir,
            local_base_url=args.local_base_url,
            cloudflared_path=args.cloudflared_path,
            startup_timeout_seconds=args.startup_timeout_seconds,
        )
        print(f"[vast-bundle-staging] cloudflared_tunnel={Path(args.job_dir).resolve() / 'vast_cloudflared_tunnel_manifest.json'}")
        print(f"[vast-bundle-staging] status={manifest.get('status')}")
        if manifest.get("public_base_url"):
            print(f"[vast-bundle-staging] public_base_url={manifest.get('public_base_url')}")
        blockers = manifest.get("blockers") or []
        if blockers:
            print("[vast-bundle-staging] blockers=" + ",".join(str(item) for item in blockers))
        return 0 if manifest.get("status") == "running" else 1
    serve_vast_bundle_staging(
        bundle_path=args.bundle_path,
        output_path=args.output_path,
        token_file=args.token_file,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
