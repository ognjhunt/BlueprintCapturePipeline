"""Provider-neutral verification for public bundle and output staging URLs."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .core.common import ensure_dir, utc_now_iso, write_json
from .provider_bundle_staging_common import redact_staging_url


PROVIDER_STAGING_VERIFICATION_SCHEMA_VERSION = "provider_staging_verification.v1"
DEFAULT_PROVIDER_STAGING_VERIFICATION_MANIFEST = "provider_staging_verification.json"
DEFAULT_PUBLIC_VERIFY_MAX_WAIT_SECONDS = 120
DEFAULT_PUBLIC_VERIFY_RETRY_INTERVAL_SECONDS = 5.0
DEFAULT_PUBLIC_VERIFY_TIMEOUT_SECONDS = 20


def probe_exception(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {"error_type": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        payload["http_status_code"] = exc.code
    elif isinstance(exc, urllib.error.URLError):
        payload["reason_type"] = type(exc.reason).__name__
    return payload


def validated_http_staging_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("provider_staging_url_must_be_absolute_http_or_https")
    return value


def head_bundle_url(
    *,
    bundle_url: str,
    bundle_path: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        request = urllib.request.Request(validated_http_staging_url(bundle_url), method="HEAD")
        # The URL was restricted to absolute HTTP(S) immediately above.
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
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
                bundle_path.stat().st_size if bundle_path and bundle_path.is_file() else None
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
            **probe_exception(exc),
        }


def put_output_probe(
    *,
    output_put_url: str,
    timeout_seconds: float,
    probe_zip: bytes,
) -> dict[str, Any]:
    try:
        request = urllib.request.Request(
            validated_http_staging_url(output_put_url),
            data=probe_zip,
            method="PUT",
            headers={"Content-Type": "application/zip"},
        )
        # The URL was restricted to absolute HTTP(S) immediately above.
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
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
            **probe_exception(exc),
        }


def cleanup_output_probe(
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


def verify_provider_staging_urls(
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
    cleanup_output_probe_requested: bool = True,
    require_bundle_fetch_probe: bool = True,
    generated_at: str | None = None,
    schema_version: str = PROVIDER_STAGING_VERIFICATION_SCHEMA_VERSION,
    manifest_filename: str = DEFAULT_PROVIDER_STAGING_VERIFICATION_MANIFEST,
    head_probe: Callable[..., dict[str, Any]] = head_bundle_url,
    output_probe: Callable[..., dict[str, Any]] = put_output_probe,
    cleanup_probe: Callable[..., dict[str, Any]] = cleanup_output_probe,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now_iso: Callable[[], str] = utc_now_iso,
) -> dict[str, Any]:
    """Verify bounded public staging reads/writes without provider assumptions."""

    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve() if bundle_path else None
    resolved_output = Path(output_path).expanduser().resolve() if output_path else None
    ensure_dir(resolved_job_dir)
    bundle_url = provider_bundle_url.strip() if isinstance(provider_bundle_url, str) else ""
    output_put_url = (
        provider_output_put_url.strip() if isinstance(provider_output_put_url, str) else ""
    )
    attempts: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if not bundle_url:
        blockers.append("provider_bundle_fetch_url_missing")
    if not output_put_url:
        blockers.append("provider_output_put_url_missing")
    if blockers:
        manifest = {
            "schema_version": schema_version,
            "generated_at": generated_at or now_iso(),
            "completed_at": now_iso(),
            "status": "blocked",
            "job_dir": str(resolved_job_dir),
            "provider_bundle_url_redacted": redact_staging_url(bundle_url) if bundle_url else None,
            "provider_output_put_url_redacted": (
                redact_staging_url(output_put_url) if output_put_url else None
            ),
            "attempt_count": 0,
            "attempts": attempts,
            "blockers": blockers,
            "warnings": warnings,
            "output_probe_cleanup": {"status": "not_requested"},
            "raw_secret_values_recorded": False,
        }
        write_json(resolved_job_dir / manifest_filename, manifest)
        return manifest

    probe_zip = b"PK\x05\x06" + (b"\x00" * 18)
    deadline = monotonic() + max(0, max_wait_seconds)
    final_cleanup: dict[str, Any] = {"status": "not_requested"}
    final_status = "blocked"
    attempt_number = 0
    required_successes = max(1, int(required_consecutive_successes))
    consecutive_successes = 0
    successful_attempts = 0
    while True:
        attempt_number += 1
        if require_bundle_fetch_probe:
            bundle_probe = head_probe(
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
            output_probe_result = output_probe(
                output_put_url=output_put_url,
                timeout_seconds=timeout_seconds,
                probe_zip=probe_zip,
            )
            if output_probe_result.get("status") == "passed":
                final_cleanup = cleanup_probe(
                    output_path=resolved_output,
                    probe_zip=probe_zip,
                    cleanup_output_probe=cleanup_output_probe_requested,
                )
                if final_cleanup.get("status") == "blocked":
                    output_probe_result = {
                        **output_probe_result,
                        "status": "blocked",
                        "blocker": final_cleanup.get("reason")
                        or "output_probe_cleanup_failed",
                    }
        elif allow_output_put_probe:
            output_probe_result = {
                "status": "blocked",
                "method": "PUT",
                "reason": "bundle_probe_must_pass_before_output_put_probe",
                "blocker": "provider_output_put_url_not_checked",
            }
        else:
            output_probe_result = {
                "status": "skipped",
                "method": "PUT",
                "reason": "output_put_probe_requires_explicit_allow",
            }
            warnings.append("provider_output_put_url_not_mutation_probed")
        attempt_passed = bundle_probe.get("status") == "passed" and (
            output_probe_result.get("status") == "passed" or not allow_output_put_probe
        )
        if not require_bundle_fetch_probe:
            attempt_passed = (
                output_probe_result.get("status") == "passed" or not allow_output_put_probe
            )
        if attempt_passed:
            successful_attempts += 1
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        attempts.append(
            {
                "attempt": attempt_number,
                "checked_at": now_iso(),
                "bundle_probe": bundle_probe,
                "output_put_probe": output_probe_result,
                "output_probe_cleanup": final_cleanup,
                "passed": attempt_passed,
                "consecutive_successes_after_attempt": consecutive_successes,
            }
        )
        if consecutive_successes >= required_successes:
            final_status = "passed"
            blockers = []
            break
        now = monotonic()
        if now >= deadline:
            blockers = []
            bundle_blocker = bundle_probe.get("blocker")
            output_blocker = output_probe_result.get("blocker")
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
        sleep(min(max(0.0, retry_interval_seconds), max(0.0, deadline - now)))

    manifest = {
        "schema_version": schema_version,
        "generated_at": generated_at or now_iso(),
        "completed_at": now_iso(),
        "status": final_status,
        "job_dir": str(resolved_job_dir),
        "provider_bundle_url_redacted": redact_staging_url(bundle_url),
        "provider_output_put_url_redacted": redact_staging_url(output_put_url),
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
        "cleanup_output_probe": cleanup_output_probe_requested,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "blockers": blockers,
        "warnings": sorted(set(warnings)),
        "output_probe_cleanup": final_cleanup,
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_job_dir / manifest_filename, manifest)
    return manifest
