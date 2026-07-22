"""Provider-neutral token, URL, and local server primitives for bundle staging."""

from __future__ import annotations

import http.server
import json
import secrets
import shutil
import socketserver
import zipfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from .core.common import ensure_dir, utc_now_iso, write_json
from .secret_artifact_policy import (
    redacted_secret_file_status,
    secret_path_disclosure_policy,
)


BUNDLE_ROUTE = "/bundle.zip"
OUTPUT_ROUTE = "/output.zip"
HEALTH_ROUTE = "/health"
DEFAULT_STAGING_HOST = "127.0.0.1"
DEFAULT_MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
PROVIDER_BUNDLE_STAGING_SCHEMA_VERSION = "provider_bundle_staging_manifest.v1"
DEFAULT_PROVIDER_BUNDLE_STAGING_MANIFEST = "provider_bundle_staging_manifest.json"
DEFAULT_PROVIDER_OUTPUT_FILENAME = "provider_runtime_output.zip"
REDACTED_TOKEN = "<redacted-token>"


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


def provider_staging_urls(
    public_base_url: str,
    token_file: Path,
) -> tuple[str, str, dict[str, Any]]:
    """Build the token-gated bundle/output URL pair and redacted token metadata."""

    token, token_status = read_or_create_staging_token(token_file)
    return (
        staging_url_with_token(public_base_url, BUNDLE_ROUTE, token),
        staging_url_with_token(public_base_url, OUTPUT_ROUTE, token),
        token_status,
    )


def redacted_staging_url_path(route: str) -> str:
    """Describe a token-gated route without persisting the staging credential."""

    return f"/{route.strip('/')}?token={REDACTED_TOKEN}"


def redact_staging_url(value: str) -> str:
    """Remove the full query string from a staging URL before artifact emission."""

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


def _write_staging_secret_env_file(
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
                (
                    "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL="
                    f"{provider_output_put_url}"
                ),
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
    status.update(
        {
            "present": path.is_file(),
            "mode": oct(path.stat().st_mode & 0o777),
            "mode_is_0600": (path.stat().st_mode & 0o777) == 0o600,
            "raw_secret_values_recorded_in_manifest": False,
        }
    )
    return status


def prepare_provider_bundle_staging(
    *,
    job_dir: str | Path,
    bundle_path: str | Path,
    public_base_url: str | None,
    token_file: str | Path,
    secret_env_file: str | Path,
    output_path: str | Path | None = None,
    generated_at: str | None = None,
    schema_version: str = PROVIDER_BUNDLE_STAGING_SCHEMA_VERSION,
    manifest_filename: str = DEFAULT_PROVIDER_BUNDLE_STAGING_MANIFEST,
    default_output_filename: str = DEFAULT_PROVIDER_OUTPUT_FILENAME,
    local_base_url_warning: str = "local_base_url_not_provider_fetchable",
) -> dict[str, Any]:
    """Prepare a provider-neutral bundle/output staging contract.

    Provider-specific entrypoints may retain historical schema and artifact names by
    supplying the compatibility arguments. Tokenized URLs are emitted only to the
    mode-0600 secret environment file; the returned and persisted manifest is redacted.
    """

    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve()
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_job_dir / default_output_filename
    )
    resolved_token_file = Path(token_file).expanduser().resolve()
    resolved_secret_env_file = Path(secret_env_file).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    token, token_status = read_or_create_staging_token(resolved_token_file)
    base_url = public_base_url.strip() if isinstance(public_base_url, str) else ""
    provider_bundle_url = staging_url_with_token(base_url, BUNDLE_ROUTE, token) if base_url else ""
    provider_output_put_url = (
        staging_url_with_token(base_url, OUTPUT_ROUTE, token) if base_url else ""
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not resolved_bundle.is_file():
        blockers.append("provider_runtime_bundle_missing")
    if not base_url:
        blockers.append("public_base_url_missing")
    elif urlparse(base_url).scheme not in {"http", "https"}:
        blockers.append("public_base_url_scheme_not_http")
    elif urlparse(base_url).hostname in {"127.0.0.1", "localhost"}:
        warnings.append(local_base_url_warning)
    if not token:
        blockers.append("staging_token_missing")

    secret_env_status = None
    if provider_bundle_url and provider_output_put_url:
        secret_env_status = _write_staging_secret_env_file(
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
        "schema_version": schema_version,
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
        "base_url_redacted": redact_staging_url(base_url) if base_url else None,
        "bundle_url_path": redacted_staging_url_path(BUNDLE_ROUTE),
        "output_put_url_path": redacted_staging_url_path(OUTPUT_ROUTE),
        "provider_bundle_url_present": bool(provider_bundle_url),
        "provider_output_put_url_present": bool(provider_output_put_url),
        "provider_fetchable_bundle_uri_ready": bool(provider_bundle_url and not blockers),
        "provider_output_callback_ready": bool(provider_output_put_url and not blockers),
        "cloudflared_available": bool(cloudflared_path),
        "cloudflared_path": cloudflared_path,
        "suggested_local_server": {
            "host": DEFAULT_STAGING_HOST,
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
    write_json(resolved_job_dir / manifest_filename, manifest)
    return manifest


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
