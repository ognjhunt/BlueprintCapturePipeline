"""Token-gated staging helper for Vast Blueprint provider bundle runs.

The live Vast adapter needs a provider-fetchable bundle URL and a writable PUT
URL for the returned runtime zip. This module keeps that staging contract
separate from the paid Vast launch path so URL readiness can be proven before a
GPU instance is created.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, utc_now_iso, write_json
from .provider_bundle_staging_common import (
    BUNDLE_ROUTE,
    DEFAULT_MAX_OUTPUT_BYTES as _DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_STAGING_HOST,
    HEALTH_ROUTE,
    OUTPUT_ROUTE,
    ProviderBundleStagingRequestHandler as _ProviderBundleStagingRequestHandler,
    create_staging_server,
    prepare_provider_bundle_staging,
    redact_staging_url as _redact_url,  # noqa: F401 - compatibility export
    read_or_create_staging_token as _read_or_create_token,
    staging_url_with_token as _url_with_token,
)
from .provider_staging_verification import (
    cleanup_output_probe as _neutral_cleanup_output_probe,
    get_bundle_url as _neutral_get_bundle_url,
    head_bundle_url as _neutral_head_bundle_url,
    probe_exception as _neutral_probe_exception,
    put_output_probe as _neutral_put_output_probe,
    validated_http_staging_url,
    verify_provider_staging_urls,
)
from .secret_artifact_policy import secret_path_disclosure_policy


VAST_BUNDLE_STAGING_SCHEMA_VERSION = "vast_bundle_staging_manifest.v1"
VAST_BUNDLE_STAGING_SELF_TEST_SCHEMA_VERSION = "vast_bundle_staging_self_test.v1"
VAST_PUBLIC_STAGING_VERIFICATION_SCHEMA_VERSION = "vast_public_staging_verification.v1"
VAST_CLOUDFLARED_TUNNEL_SCHEMA_VERSION = "vast_cloudflared_tunnel_manifest.v1"
DEFAULT_TOKEN_FILE = "~/.blueprint-secrets/vast_bundle_staging_token"
DEFAULT_SECRET_ENV_FILE = "~/.blueprint-secrets/vast_bundle_staging_urls.env"
DEFAULT_OUTPUT_FILENAME = "vast_provider_runtime_output.zip"
DEFAULT_HOST = DEFAULT_STAGING_HOST
DEFAULT_MAX_OUTPUT_BYTES = _DEFAULT_MAX_OUTPUT_BYTES
VastBundleStagingRequestHandler = _ProviderBundleStagingRequestHandler
DEFAULT_PUBLIC_VERIFY_MAX_WAIT_SECONDS = 120
DEFAULT_PUBLIC_VERIFY_RETRY_INTERVAL_SECONDS = 5.0
DEFAULT_PUBLIC_VERIFY_TIMEOUT_SECONDS = 20
REDACTED_TOKEN = "<redacted-token>"
CLOUDFLARED_URL_RE = re.compile(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com")


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


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
    return prepare_provider_bundle_staging(
        job_dir=job_dir,
        bundle_path=bundle_path,
        public_base_url=public_base_url,
        token_file=resolved_token_file,
        secret_env_file=resolved_secret_env_file,
        output_path=output_path,
        generated_at=generated_at,
        schema_version=VAST_BUNDLE_STAGING_SCHEMA_VERSION,
        manifest_filename="vast_bundle_staging_manifest.json",
        default_output_filename=DEFAULT_OUTPUT_FILENAME,
        local_base_url_warning="local_base_url_not_provider_fetchable_from_vast",
    )


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
        health_url = validated_http_staging_url(f"{base_url}{HEALTH_ROUTE}")
        health = urllib.request.urlopen(health_url, timeout=10)  # nosec B310
        checks.append({"label": "health", "http_status": health.status})
        bundle_url = _url_with_token(base_url, BUNDLE_ROUTE, token)
        head_request = urllib.request.Request(validated_http_staging_url(bundle_url), method="HEAD")
        head = urllib.request.urlopen(head_request, timeout=10)  # nosec B310
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
            validated_http_staging_url(_url_with_token(base_url, OUTPUT_ROUTE, token)),
            data=upload_bytes,
            method="PUT",
            headers={"Content-Type": "application/zip"},
        )
        put = urllib.request.urlopen(put_request, timeout=10)  # nosec B310
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
    return _neutral_probe_exception(exc)


def _head_bundle_url(
    *,
    bundle_url: str,
    bundle_path: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    return _neutral_head_bundle_url(
        bundle_url=bundle_url,
        bundle_path=bundle_path,
        timeout_seconds=timeout_seconds,
    )


def _put_output_probe(
    *,
    output_put_url: str,
    timeout_seconds: float,
    probe_zip: bytes,
) -> dict[str, Any]:
    return _neutral_put_output_probe(
        output_put_url=output_put_url,
        timeout_seconds=timeout_seconds,
        probe_zip=probe_zip,
    )


def _get_bundle_url(
    *,
    bundle_url: str,
    bundle_path: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    return _neutral_get_bundle_url(
        bundle_url=bundle_url,
        bundle_path=bundle_path,
        timeout_seconds=timeout_seconds,
    )


def _cleanup_output_probe(
    *,
    output_path: Path | None,
    probe_zip: bytes,
    cleanup_output_probe: bool,
) -> dict[str, Any]:
    return _neutral_cleanup_output_probe(
        output_path=output_path,
        probe_zip=probe_zip,
        cleanup_output_probe=cleanup_output_probe,
    )


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
    bundle_probe_method: str = "HEAD",
    generated_at: str | None = None,
) -> dict[str, Any]:
    return verify_provider_staging_urls(
        job_dir=job_dir,
        provider_bundle_url=provider_bundle_url,
        provider_output_put_url=provider_output_put_url,
        bundle_path=bundle_path,
        output_path=output_path,
        max_wait_seconds=max_wait_seconds,
        retry_interval_seconds=retry_interval_seconds,
        timeout_seconds=timeout_seconds,
        required_consecutive_successes=required_consecutive_successes,
        allow_output_put_probe=allow_output_put_probe,
        cleanup_output_probe_requested=cleanup_output_probe,
        require_bundle_fetch_probe=require_bundle_fetch_probe,
        generated_at=generated_at,
        schema_version=VAST_PUBLIC_STAGING_VERIFICATION_SCHEMA_VERSION,
        manifest_filename="vast_public_staging_verification.json",
        head_probe=(
            _get_bundle_url if bundle_probe_method.strip().upper() == "GET" else _head_bundle_url
        ),
        output_probe=_put_output_probe,
        cleanup_probe=_cleanup_output_probe,
        monotonic=time.monotonic,
        sleep=time.sleep,
        now_iso=utc_now_iso,
    )


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
        print(
            f"[vast-bundle-staging] manifest={Path(args.job_dir).resolve() / 'vast_bundle_staging_manifest.json'}"
        )
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
        print(
            f"[vast-bundle-staging] self_test={Path(args.job_dir).resolve() / 'vast_bundle_staging_self_test.json'}"
        )
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
        print(
            f"[vast-bundle-staging] public_verification={Path(args.job_dir).resolve() / 'vast_public_staging_verification.json'}"
        )
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
        print(
            f"[vast-bundle-staging] cloudflared_tunnel={Path(args.job_dir).resolve() / 'vast_cloudflared_tunnel_manifest.json'}"
        )
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
