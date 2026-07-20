"""Stage WAM provider bundles through S3-compatible object storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

from .common import ensure_dir, utc_now_iso, write_json
from .secret_artifact_policy import (
    redacted_secret_file_status,
    secret_path_disclosure_policy,
)
from .safe_outbound_http import (
    SafeOutboundHttpError,
    presigned_transfer_policy,
    request as safe_http_request,
)


SCHEMA_VERSION = "wam_provider_object_store_staging.v1"
SIGNED_OUTPUT_ROUND_TRIP_SCHEMA_VERSION = "wam_signed_output_round_trip.v1"
SIGNED_OUTPUT_SENTINEL_BYTES = 96
SIGNED_OUTPUT_HTTP_TIMEOUT_SECONDS = 30
SIGNED_OUTPUT_MAX_RESPONSE_BYTES = 64 * 1024
DEFAULT_ACCESS_KEY_FILES = (
    "~/.blueprint-secrets/digitalocean_spaces_access_key_id",
    "~/.blueprint-secrets/runpod_s3_access_key",
    "~/.blueprint-secrets/r2_access_key_id",
    "~/.blueprint-secrets/aws_access_key_id",
)
DEFAULT_SECRET_KEY_FILES = (
    "~/.blueprint-secrets/digitalocean_spaces_secret_access_key",
    "~/.blueprint-secrets/runpod_s3_secret_key",
    "~/.blueprint-secrets/r2_secret_access_key",
    "~/.blueprint-secrets/aws_secret_access_key",
)
DEFAULT_ENDPOINT_FILES = (
    "~/.blueprint-secrets/digitalocean_spaces_endpoint_url",
    "~/.blueprint-secrets/runpod_s3_endpoint_url",
    "~/.blueprint-secrets/r2_endpoint_url",
)
DEFAULT_BUCKET_FILES = (
    "~/.blueprint-secrets/digitalocean_spaces_bucket",
    "~/.blueprint-secrets/runpod_network_volume_id",
    "~/.blueprint-secrets/r2_bucket",
    "~/.blueprint-secrets/aws_s3_bucket",
)
DEFAULT_REGION_FILES = (
    "~/.blueprint-secrets/digitalocean_spaces_region",
    "~/.blueprint-secrets/aws_region",
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _redact_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return "<redacted-url>" if value else ""
    query = "REDACTED_QUERY" if parsed.query else ""
    fragment = "REDACTED_FRAGMENT" if parsed.fragment else ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query, fragment))


def _parse_iso_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _presigned_url_expiry_metadata(generated_at: str, expiration_seconds: int) -> dict[str, Any]:
    generated_dt = _parse_iso_datetime(generated_at) or datetime.now(timezone.utc)
    expires_at = generated_dt + timedelta(seconds=int(expiration_seconds))
    return {
        "generated_at": generated_at,
        "expires_at": _iso_z(expires_at),
        "expiration_seconds": int(expiration_seconds),
        "expiry_warning": int(expiration_seconds) <= 60 * 60,
        "raw_url_values_recorded": False,
    }


def _safe_key_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return "_".join(part for part in cleaned.split("_") if part) or "wam_provider_run"


def _job_key_component(path: Path) -> str:
    parts = [part for part in path.parts[-3:] if part and part != "/"]
    return _safe_key_component("__".join(parts))


def _file_status(path: Path, *, label: str, value_present: bool = False) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    mode = oct(path.stat().st_mode & 0o777) if exists else None
    return {
        "label": label,
        "path": str(path),
        "present": is_file,
        "mode": mode,
        "mode_is_0600": mode == "0o600",
        "size_bytes": path.stat().st_size if is_file else 0,
        "mtime_ns": path.stat().st_mtime_ns if is_file else None,
        "value_present": value_present,
        "raw_secret_values_recorded": False,
    }


def _credential_file_status(
    path: Path,
    *,
    label: str,
    source: str,
    value_present: bool = False,
) -> dict[str, Any]:
    status = redacted_secret_file_status(
        path,
        path_source=source,
        raw_secret_field="raw_secret_values_recorded",
    )
    status.update(
        {
            "label": label,
            "source": source,
            "value_present": value_present,
        }
    )
    return status


def _read_first_file(
    *,
    explicit_path: str | Path | None,
    env_name: str,
    default_paths: Sequence[str],
    label: str,
    allow_env_value: bool = False,
) -> tuple[str, dict[str, Any]]:
    env_value = _string(os.getenv(env_name))
    if allow_env_value and env_value:
        return env_value, {
            "label": label,
            "source": "env_value",
            "env_name": env_name,
            "configured": True,
            "value_present": True,
            "raw_secret_values_recorded": False,
        }
    candidates: list[tuple[str, Path]] = []
    if explicit_path:
        candidates.append(("explicit_path", Path(explicit_path).expanduser()))
    env_file = _string(os.getenv(f"{env_name}_FILE"))
    if env_file:
        candidates.append((f"{env_name}_FILE", Path(env_file).expanduser()))
    for item in default_paths:
        candidates.append(("default_secret_file", Path(item).expanduser()))
    statuses: list[dict[str, Any]] = []
    for source, path in candidates:
        resolved = path.resolve()
        try:
            value = resolved.read_text(encoding="utf-8").strip() if resolved.is_file() else ""
        except OSError as exc:
            status = _credential_file_status(
                resolved,
                label=label,
                source=source,
                value_present=False,
            )
            status["read_error"] = type(exc).__name__
            statuses.append(status)
            continue
        status = _credential_file_status(
            resolved,
            label=label,
            source=source,
            value_present=bool(value),
        )
        statuses.append(status)
        if value:
            return value, {
                "label": label,
                "source": source,
                "selected_file": status,
                "configured": True,
                "candidate_files": statuses,
                "raw_secret_values_recorded": False,
            }
    return "", {
        "label": label,
        "source": "not_found",
        "configured": False,
        "candidate_files": statuses,
        "raw_secret_values_recorded": False,
    }


def _write_sensitive_file(path: Path, value: str, *, label: str) -> dict[str, Any]:
    ensure_dir(path.parent)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
    path.chmod(0o600)
    return _file_status(path, label=label, value_present=bool(value))


def signed_output_object_binding_sha256(put_url: str, get_url: str) -> str:
    """Hash the non-secret origin/path identity shared by one PUT/GET pair."""

    identities: list[str] = []
    for label, value in (("put", put_url), ("get", get_url)):
        parsed = urlparse(str(value or ""))
        if (
            parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError(f"signed_output_{label}_url_invalid")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"signed_output_{label}_url_invalid") from exc
        identity = (
            parsed.scheme.lower(),
            parsed.hostname.lower(),
            port or 443,
            parsed.path,
        )
        identities.append(json.dumps(identity, separators=(",", ":")))
    if identities[0] != identities[1]:
        raise ValueError("signed_output_put_get_object_identity_mismatch")
    return hashlib.sha256(identities[0].encode("utf-8")).hexdigest()


def _safe_transfer_exception(exc: Exception) -> dict[str, Any]:
    """Return diagnostics that cannot contain a presigned URL or query."""

    if isinstance(exc, urllib.error.HTTPError):
        return {"error_type": type(exc).__name__, "http_status_code": int(exc.code)}
    if isinstance(exc, urllib.error.URLError):
        return {
            "error_type": type(exc).__name__,
            "reason_type": type(exc.reason).__name__,
        }
    if isinstance(exc, SafeOutboundHttpError):
        return {
            "error_type": type(exc).__name__,
            "policy_error_code": str(exc).partition(":")[0],
        }
    return {"error_type": type(exc).__name__}


def _s3_absence_confirmed(client: Any, *, bucket: str, key: str) -> dict[str, Any]:
    """Prove an S3 object is absent without recording its key."""

    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        response = _mapping(getattr(exc, "response", None))
        metadata = _mapping(response.get("ResponseMetadata"))
        error = _mapping(response.get("Error"))
        status = metadata.get("HTTPStatusCode")
        code = _string(error.get("Code"))
        if status == 404 or code.lower() in {"404", "nosuchkey", "notfound"}:
            return {
                "status": "passed",
                "absence_confirmed": True,
                "http_status_code": 404,
                "raw_secret_values_recorded": False,
            }
        return {
            "status": "blocked",
            "absence_confirmed": False,
            **_safe_transfer_exception(exc),
            "raw_secret_values_recorded": False,
        }
    return {
        "status": "blocked",
        "absence_confirmed": False,
        "object_still_present": True,
        "raw_secret_values_recorded": False,
    }


def _signed_output_round_trip_preflight(
    client: Any,
    *,
    bucket: str,
    sentinel_key: str,
    expiration_seconds: int,
) -> dict[str, Any]:
    """Exercise a distinct signed PUT/GET object and prove its deletion."""

    sentinel = b"blueprint-signed-output-preflight-v1\n" + secrets.token_bytes(
        SIGNED_OUTPUT_SENTINEL_BYTES
    )
    sentinel_sha256 = hashlib.sha256(sentinel).hexdigest()
    key_sha256 = hashlib.sha256(sentinel_key.encode("utf-8")).hexdigest()
    blockers: list[str] = []
    put_probe: dict[str, Any] = {"status": "not_run"}
    get_probe: dict[str, Any] = {"status": "not_run"}
    cleanup: dict[str, Any] = {"status": "not_run", "absence_confirmed": False}
    try:
        put_url = client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket,
                "Key": sentinel_key,
                "ContentType": "application/octet-stream",
            },
            ExpiresIn=int(expiration_seconds),
            HttpMethod="PUT",
        )
        get_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": sentinel_key},
            ExpiresIn=int(expiration_seconds),
            HttpMethod="GET",
        )
        signed_output_object_binding_sha256(put_url, get_url)
        put_response = safe_http_request(
            put_url,
            method="PUT",
            data=sentinel,
            headers={"Content-Type": "application/octet-stream"},
            timeout_seconds=SIGNED_OUTPUT_HTTP_TIMEOUT_SECONDS,
            max_response_bytes=SIGNED_OUTPUT_MAX_RESPONSE_BYTES,
            policy=presigned_transfer_policy(
                put_url,
                max_response_bytes=SIGNED_OUTPUT_MAX_RESPONSE_BYTES,
            ),
        )
        put_status = int(put_response.status)
        put_probe = {
            "status": "passed" if 200 <= put_status < 300 else "blocked",
            "http_status_code": put_status,
            "sentinel_size_bytes": len(sentinel),
        }
        if put_probe["status"] != "passed":
            blockers.append("signed_output_sentinel_put_failed")
        else:
            get_response = safe_http_request(
                get_url,
                method="GET",
                timeout_seconds=SIGNED_OUTPUT_HTTP_TIMEOUT_SECONDS,
                max_response_bytes=len(sentinel) + 1,
                policy=presigned_transfer_policy(
                    get_url,
                    max_response_bytes=len(sentinel) + 1,
                ),
            )
            received_sha256 = hashlib.sha256(get_response.body).hexdigest()
            exact = get_response.body == sentinel and received_sha256 == sentinel_sha256
            get_probe = {
                "status": "passed" if int(get_response.status) == 200 and exact else "blocked",
                "http_status_code": int(get_response.status),
                "received_size_bytes": len(get_response.body),
                "received_sha256": received_sha256,
                "exact_bytes_and_sha256": exact,
            }
            if get_probe["status"] != "passed":
                blockers.append("signed_output_sentinel_get_mismatch")
    except Exception as exc:
        failed_phase = "get" if put_probe.get("status") == "passed" else "put"
        if failed_phase == "put":
            put_probe = {"status": "blocked", **_safe_transfer_exception(exc)}
            blockers.append("signed_output_sentinel_put_failed")
        else:
            get_probe = {"status": "blocked", **_safe_transfer_exception(exc)}
            blockers.append("signed_output_sentinel_get_failed")
    finally:
        try:
            delete_response = client.delete_object(Bucket=bucket, Key=sentinel_key)
            response_metadata = _mapping(_mapping(delete_response).get("ResponseMetadata"))
            delete_status = int(response_metadata.get("HTTPStatusCode") or 204)
            if not 200 <= delete_status < 300:
                cleanup = {
                    "status": "blocked",
                    "delete_http_status_code": delete_status,
                    "absence_confirmed": False,
                }
            else:
                cleanup = {
                    **_s3_absence_confirmed(
                        client,
                        bucket=bucket,
                        key=sentinel_key,
                    ),
                    "delete_http_status_code": delete_status,
                }
        except Exception as exc:
            cleanup = {
                "status": "blocked",
                "absence_confirmed": False,
                **_safe_transfer_exception(exc),
            }
        if cleanup.get("status") != "passed" or cleanup.get("absence_confirmed") is not True:
            blockers.append("signed_output_sentinel_cleanup_unverified")

    unique_blockers = sorted(set(blockers))
    return {
        "schema_version": SIGNED_OUTPUT_ROUND_TRIP_SCHEMA_VERSION,
        "status": "passed" if not unique_blockers else "blocked",
        "sentinel_key_sha256": key_sha256,
        "sentinel_sha256": sentinel_sha256,
        "sentinel_size_bytes": len(sentinel),
        "put": put_probe,
        "get": get_probe,
        "cleanup": cleanup,
        "actual_output_key_was_not_used": True,
        "blockers": unique_blockers,
        "raw_signed_urls_recorded": False,
        "raw_secret_values_recorded": False,
    }


def stage_wam_provider_bundle_object_store(
    *,
    job_dir: str | Path,
    bundle_path: str | Path,
    access_key_id_file: str | Path | None = None,
    secret_access_key_file: str | Path | None = None,
    endpoint_url: str = "",
    endpoint_url_file: str | Path | None = None,
    bucket: str = "",
    bucket_file: str | Path | None = None,
    region: str = "",
    region_file: str | Path | None = None,
    key_prefix: str = "blueprint/wam-provider",
    expiration_seconds: int = 12 * 60 * 60,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    expiry_metadata = _presigned_url_expiry_metadata(generated, expiration_seconds)
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    access_key, access_meta = _read_first_file(
        explicit_path=access_key_id_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID",
        default_paths=DEFAULT_ACCESS_KEY_FILES,
        label="object_store_access_key_id",
    )
    secret_key, secret_meta = _read_first_file(
        explicit_path=secret_access_key_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY",
        default_paths=DEFAULT_SECRET_KEY_FILES,
        label="object_store_secret_access_key",
    )
    endpoint, endpoint_meta = _read_first_file(
        explicit_path=endpoint_url_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL",
        default_paths=DEFAULT_ENDPOINT_FILES,
        label="object_store_endpoint_url",
        allow_env_value=True,
    )
    if endpoint_url:
        endpoint = endpoint_url
        endpoint_meta = {
            "label": "object_store_endpoint_url",
            "source": "cli_argument",
            "configured": True,
            "value_present": True,
            "raw_secret_values_recorded": False,
        }
    bucket_value, bucket_meta = _read_first_file(
        explicit_path=bucket_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_BUCKET",
        default_paths=DEFAULT_BUCKET_FILES,
        label="object_store_bucket",
        allow_env_value=True,
    )
    if bucket:
        bucket_value = bucket
        bucket_meta = {
            "label": "object_store_bucket",
            "source": "cli_argument",
            "configured": True,
            "value_present": True,
            "raw_secret_values_recorded": False,
        }
    region_value, region_meta = _read_first_file(
        explicit_path=region_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_REGION",
        default_paths=DEFAULT_REGION_FILES,
        label="object_store_region",
        allow_env_value=True,
    )
    if region:
        region_value = region
        region_meta = {
            "label": "object_store_region",
            "source": "cli_argument",
            "configured": True,
            "value_present": True,
            "raw_secret_values_recorded": False,
        }
    region_value = region_value or "us-east-1"
    blockers: list[str] = []
    if not resolved_bundle.is_file():
        blockers.append("wam_provider_bundle_missing")
    if not access_key:
        blockers.append("missing_object_store_access_key_id_file")
    if not secret_key:
        blockers.append("missing_object_store_secret_access_key_file")
    if not bucket_value:
        blockers.append("missing_object_store_bucket_or_network_volume_id_file")
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.client import Config  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        boto3 = None  # type: ignore[assignment]
        Config = None  # type: ignore[assignment]
        blockers.append(f"boto3_or_botocore_unavailable:{type(exc).__name__}")
    bundle_url = ""
    output_put_url = ""
    output_get_url = ""
    bundle_key = ""
    output_key = ""
    upload_detail: dict[str, Any] = {"status": "not_run"}
    output_key_absence: dict[str, Any] = {
        "status": "not_run",
        "absence_confirmed": False,
    }
    signed_output_round_trip: dict[str, Any] = {
        "schema_version": SIGNED_OUTPUT_ROUND_TRIP_SCHEMA_VERSION,
        "status": "not_run",
        "blockers": [],
        "raw_signed_urls_recorded": False,
        "raw_secret_values_recorded": False,
    }
    output_url_object_binding_sha256 = ""
    if not blockers:
        assert boto3 is not None
        assert Config is not None
        safe_prefix = key_prefix.strip("/ ") or "blueprint/wam-provider"
        job_key = _job_key_component(resolved_job_dir)
        run_nonce = secrets.token_hex(16)
        bundle_key = f"{safe_prefix}/{job_key}/{resolved_bundle.name}"
        output_key = f"{safe_prefix}/{job_key}/runpod_provider_runtime_output_{run_nonce}.zip"
        sentinel_key = f"{safe_prefix}/{job_key}/preflight/signed_output_round_trip_{run_nonce}.bin"
        client_kwargs: dict[str, Any] = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region_value,
            "config": Config(signature_version="s3v4"),
        }
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        try:
            client = boto3.client("s3", **client_kwargs)
            client.upload_file(str(resolved_bundle), bucket_value, bundle_key)
            # The paid worker output key is run-unique and must be absent before
            # its GET URL is handed to a poller.  This prevents a stale object
            # from being consumed as a fresh episode result.
            output_key_absence = _s3_absence_confirmed(
                client,
                bucket=bucket_value,
                key=output_key,
            )
            if output_key_absence.get("status") != "passed":
                blockers.append("fresh_output_key_absence_unverified")

            # Exercise the same presign/HTTP path on a distinct sentinel key.
            # The sentinel is always deleted and absence-confirmed; the actual
            # worker output key is never populated by this preflight.
            signed_output_round_trip = _signed_output_round_trip_preflight(
                client,
                bucket=bucket_value,
                sentinel_key=sentinel_key,
                expiration_seconds=int(expiration_seconds),
            )
            blockers.extend(signed_output_round_trip.get("blockers") or [])
            bundle_url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_value, "Key": bundle_key},
                ExpiresIn=int(expiration_seconds),
                HttpMethod="GET",
            )
            if not blockers:
                output_put_url = client.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": bucket_value,
                        "Key": output_key,
                        "ContentType": "application/zip",
                    },
                    ExpiresIn=int(expiration_seconds),
                    HttpMethod="PUT",
                )
                output_get_url = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_value, "Key": output_key},
                    ExpiresIn=int(expiration_seconds),
                    HttpMethod="GET",
                )
                output_url_object_binding_sha256 = signed_output_object_binding_sha256(
                    output_put_url,
                    output_get_url,
                )
            upload_detail = {
                "status": "completed" if not blockers else "blocked",
                "bucket_configured": True,
                "endpoint_configured": bool(endpoint),
                "region": region_value,
                "bundle_key": bundle_key,
                "output_key": output_key,
                "bundle_size_bytes": resolved_bundle.stat().st_size,
                "raw_secret_values_recorded": False,
            }
        except Exception as exc:
            blockers.append(f"object_store_upload_or_presign_failed:{type(exc).__name__}")
            upload_detail = {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "raw_secret_values_recorded": False,
            }
    bundle_url_file = resolved_job_dir / "provider_bundle_url.txt"
    output_put_url_file = resolved_job_dir / "provider_output_put_url.txt"
    output_get_url_file = resolved_job_dir / "provider_output_get_url.txt"
    bundle_url_file_status = (
        _write_sensitive_file(bundle_url_file, bundle_url, label="provider_bundle_url")
        if bundle_url
        else _file_status(bundle_url_file, label="provider_bundle_url")
    )
    output_put_url_file_status = (
        _write_sensitive_file(output_put_url_file, output_put_url, label="provider_output_put_url")
        if output_put_url
        else _file_status(output_put_url_file, label="provider_output_put_url")
    )
    output_get_url_file_status = (
        _write_sensitive_file(output_get_url_file, output_get_url, label="provider_output_get_url")
        if output_get_url
        else _file_status(output_get_url_file, label="provider_output_get_url")
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": (
            "completed"
            if bundle_url and output_put_url and output_get_url and not blockers
            else "blocked"
        ),
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(resolved_bundle),
        "bundle_present": resolved_bundle.is_file(),
        "bundle_size_bytes": resolved_bundle.stat().st_size if resolved_bundle.is_file() else 0,
        "object_store": {
            "access_key_id": access_meta,
            "secret_access_key": secret_meta,
            "endpoint_url": endpoint_meta,
            "bucket": bucket_meta,
            "region": region_meta,
            "key_prefix": key_prefix,
            "expiration_seconds": int(expiration_seconds),
            "expires_at": expiry_metadata["expires_at"],
            "expiry_warning": expiry_metadata["expiry_warning"],
        },
        "presigned_url_expiry": expiry_metadata,
        "upload_detail": upload_detail,
        "signed_output_round_trip": signed_output_round_trip,
        "fresh_output_key_absence": output_key_absence,
        "output_key_run_unique": bool(output_key),
        "output_url_object_binding_sha256": (output_url_object_binding_sha256 or None),
        "bundle_key": bundle_key or None,
        "output_key": output_key or None,
        "provider_bundle_url_file": bundle_url_file_status,
        "provider_output_put_url_file": output_put_url_file_status,
        "provider_output_get_url_file": output_get_url_file_status,
        "provider_bundle_url_redacted": _redact_url(bundle_url) if bundle_url else None,
        "provider_output_put_url_redacted": _redact_url(output_put_url) if output_put_url else None,
        "provider_output_get_url_redacted": _redact_url(output_get_url) if output_get_url else None,
        "runpod_create_command_template": (
            "python -m blueprint_pipeline.runpod_wam_async_runner create "
            f"--job-dir {resolved_job_dir} --bundle-path {resolved_bundle} "
            f"--provider-bundle-url-file {bundle_url_file} "
            f"--provider-output-put-url-file {output_put_url_file} "
            f"--provider-output-get-url-file {output_get_url_file} "
            "--allow-paid-runpod-launch"
        ),
        "vast_create_command_template": (
            "python -m blueprint_pipeline.vast_wam_async_runner create "
            f"--job-dir {resolved_job_dir} --bundle-path {resolved_bundle} "
            f"--provider-bundle-url-file {bundle_url_file} "
            f"--provider-output-put-url-file {output_put_url_file} "
            f"--provider-output-get-url-file {output_get_url_file} "
            "--allow-paid-vast-launch"
        ),
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
        "secret_artifact_policy": secret_path_disclosure_policy(),
    }
    write_json(resolved_job_dir / "wam_provider_object_store_staging_manifest.json", manifest)
    return manifest


def refresh_wam_provider_output_get_url(
    *,
    job_dir: str | Path,
    access_key_id_file: str | Path | None = None,
    secret_access_key_file: str | Path | None = None,
    endpoint_url: str = "",
    endpoint_url_file: str | Path | None = None,
    bucket: str = "",
    bucket_file: str | Path | None = None,
    region: str = "",
    region_file: str | Path | None = None,
    expiration_seconds: int = 6 * 24 * 60 * 60,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Refresh read access to an existing staged output without mutating it."""

    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    manifest_path = resolved_job_dir / "wam_provider_object_store_staging_manifest.json"
    blockers: list[str] = []
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = dict(value) if isinstance(value, Mapping) else {}
    except (OSError, json.JSONDecodeError):
        manifest = {}
    output_key = _string(manifest.get("output_key"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or not output_key
    ):
        blockers.append("wam_provider_output_refresh_staging_manifest_invalid")

    access_key, access_meta = _read_first_file(
        explicit_path=access_key_id_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID",
        default_paths=DEFAULT_ACCESS_KEY_FILES,
        label="object_store_access_key_id",
    )
    secret_key, secret_meta = _read_first_file(
        explicit_path=secret_access_key_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY",
        default_paths=DEFAULT_SECRET_KEY_FILES,
        label="object_store_secret_access_key",
    )
    endpoint, endpoint_meta = _read_first_file(
        explicit_path=endpoint_url_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL",
        default_paths=DEFAULT_ENDPOINT_FILES,
        label="object_store_endpoint_url",
        allow_env_value=True,
    )
    if endpoint_url:
        endpoint = endpoint_url
    bucket_value, bucket_meta = _read_first_file(
        explicit_path=bucket_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_BUCKET",
        default_paths=DEFAULT_BUCKET_FILES,
        label="object_store_bucket",
        allow_env_value=True,
    )
    if bucket:
        bucket_value = bucket
    region_value, region_meta = _read_first_file(
        explicit_path=region_file,
        env_name="BLUEPRINT_WAM_OBJECT_STORE_REGION",
        default_paths=DEFAULT_REGION_FILES,
        label="object_store_region",
        allow_env_value=True,
    )
    if region:
        region_value = region
    region_value = region_value or "us-east-1"
    if not access_key:
        blockers.append("missing_object_store_access_key_id_file")
    if not secret_key:
        blockers.append("missing_object_store_secret_access_key_file")
    if not bucket_value:
        blockers.append("missing_object_store_bucket_or_network_volume_id_file")
    if not 60 <= int(expiration_seconds) <= 7 * 24 * 60 * 60:
        blockers.append("wam_provider_output_refresh_expiration_invalid")

    refreshed_url = ""
    object_size = -1
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.client import Config  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        boto3 = None  # type: ignore[assignment]
        Config = None  # type: ignore[assignment]
        blockers.append(f"boto3_or_botocore_unavailable:{type(exc).__name__}")
    if not blockers:
        assert boto3 is not None
        assert Config is not None
        client_kwargs: dict[str, Any] = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region_value,
            "config": Config(signature_version="s3v4"),
        }
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        try:
            client = boto3.client("s3", **client_kwargs)
            observed = _mapping(client.head_object(Bucket=bucket_value, Key=output_key))
            object_size = int(observed.get("ContentLength") or -1)
            if object_size <= 0:
                raise ValueError("staged_output_object_size_invalid")
            refreshed_url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_value, "Key": output_key},
                ExpiresIn=int(expiration_seconds),
                HttpMethod="GET",
            )
            put_url = (resolved_job_dir / "provider_output_put_url.txt").read_text(
                encoding="utf-8"
            ).strip()
            signed_output_object_binding_sha256(put_url, refreshed_url)
        except Exception as exc:
            blockers.append(
                f"wam_provider_output_refresh_failed:{type(exc).__name__}"
            )
            refreshed_url = ""

    expiry = _presigned_url_expiry_metadata(generated, int(expiration_seconds))
    url_path = resolved_job_dir / "provider_output_get_url.txt"
    url_status = (
        _write_sensitive_file(
            url_path,
            refreshed_url,
            label="provider_output_get_url",
        )
        if refreshed_url and not blockers
        else _file_status(url_path, label="provider_output_get_url")
    )
    result = {
        "schema_version": "wam_provider_output_get_refresh.v1",
        "status": "completed" if refreshed_url and not blockers else "blocked",
        "generated_at": generated,
        "job_dir": str(resolved_job_dir),
        "object_size_bytes": object_size if object_size > 0 else None,
        "presigned_url_expiry": expiry,
        "provider_output_get_url_file": url_status,
        "provider_output_get_url_redacted": (
            _redact_url(refreshed_url) if refreshed_url else None
        ),
        "object_store": {
            "access_key_id": access_meta,
            "secret_access_key": secret_meta,
            "endpoint_url": endpoint_meta,
            "bucket": bucket_meta,
            "region": region_meta,
        },
        "output_object_mutated": False,
        "raw_signed_urls_recorded": False,
        "raw_secret_values_recorded": False,
        "blockers": sorted(set(blockers)),
    }
    if result["status"] == "completed":
        manifest["provider_output_get_url_file"] = url_status
        manifest["provider_output_get_url_redacted"] = _redact_url(refreshed_url)
        manifest["output_get_url_refresh"] = {
            "schema_version": result["schema_version"],
            "status": "completed",
            "generated_at": generated,
            "object_size_bytes": object_size,
            "presigned_url_expiry": expiry,
            "output_object_mutated": False,
            "raw_signed_urls_recorded": False,
        }
        write_json(manifest_path, manifest)
    write_json(
        resolved_job_dir / "wam_provider_object_store_get_refresh.json",
        result,
    )
    return result


def presign_warm_inbox_channel(
    job_dir: str | Path,
    *,
    key_prefix: str = "blueprint/isaac-g1-parity",
    expiration_seconds: int = 12 * 60 * 60,
) -> dict[str, Any]:
    """Configure the durable warm-render broker; never mint a mutable inbox key.

    The historical function name remains for call-site compatibility. Live
    execution now requires file-backed broker URL and bearer-token inputs. The
    single overwriteable object-store channel is intentionally unavailable.
    """
    del key_prefix, expiration_seconds
    resolved_job_dir = Path(job_dir)
    resolved_job_dir.mkdir(parents=True, exist_ok=True)
    generated = utc_now_iso()
    broker_url_source_text = os.getenv("BLUEPRINT_WARM_RENDER_BROKER_BASE_URL_FILE", "").strip()
    broker_token_source_text = os.getenv("BLUEPRINT_WARM_RENDER_BROKER_TOKEN_FILE", "").strip()
    broker_url_source = Path(broker_url_source_text).expanduser()
    broker_token_source = Path(broker_token_source_text).expanduser()
    blockers: list[str] = []
    broker_base_url = ""
    broker_token = ""
    if not broker_url_source_text:
        blockers.append("missing_warm_render_broker_base_url_file")
    elif not broker_url_source.is_file() or broker_url_source.is_symlink():
        blockers.append("warm_render_broker_base_url_file_invalid")
    else:
        broker_base_url = broker_url_source.read_text(encoding="utf-8").strip()
        parsed = urlparse(broker_base_url)
        local_host = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        allowed_schemes = {"http", "https"} if local_host else {"https"}
        if (
            parsed.scheme not in allowed_schemes
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            blockers.append("warm_render_broker_base_url_invalid")
    if not broker_token_source_text:
        blockers.append("missing_warm_render_broker_token_file")
    elif not broker_token_source.is_file() or broker_token_source.is_symlink():
        blockers.append("warm_render_broker_token_file_invalid")
    else:
        if broker_token_source.stat().st_mode & 0o077:
            blockers.append("warm_render_broker_token_file_permissions_too_open")
        broker_token = broker_token_source.read_text(encoding="utf-8").strip()
        if len(broker_token.encode("utf-8")) < 32:
            blockers.append("warm_render_broker_token_too_short")
    broker_url_file = resolved_job_dir / "warm_broker_base_url.txt"
    broker_token_file = resolved_job_dir / "warm_broker_token.txt"
    if broker_base_url and not blockers:
        _write_sensitive_file(
            broker_url_file,
            broker_base_url,
            label="warm_render_broker_base_url",
        )
        _write_sensitive_file(
            broker_token_file,
            broker_token,
            label="warm_render_broker_token",
        )
    return {
        "status": "completed" if not blockers else "blocked",
        "blockers": blockers,
        "generated_at": generated,
        "transport": "durable_warm_render_broker",
        "single_object_transport_enabled": False,
        "object_per_job_durable_queue_required": True,
        "server_canonical_job_ids_required": True,
        "server_idempotency_required": True,
        "broker_base_url_file": str(broker_url_file) if not blockers else None,
        "broker_token_file": str(broker_token_file) if not blockers else None,
        "raw_secret_values_recorded": False,
        "raw_url_values_recorded": False,
        "inbox_key": None,
        "warm_inbox_get_url_file": None,
        "warm_inbox_put_url_file": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--bundle-path")
    parser.add_argument("--refresh-output-get-url", action="store_true")
    parser.add_argument("--access-key-id-file")
    parser.add_argument("--secret-access-key-file")
    parser.add_argument("--endpoint-url", default="")
    parser.add_argument("--endpoint-url-file")
    parser.add_argument("--bucket", default="")
    parser.add_argument("--bucket-file")
    parser.add_argument("--region", default="")
    parser.add_argument("--region-file")
    parser.add_argument("--key-prefix", default="blueprint/wam-provider")
    parser.add_argument("--expiration-seconds", type=int, default=12 * 60 * 60)
    args = parser.parse_args(argv)
    common = {
        "job_dir": args.job_dir,
        "access_key_id_file": args.access_key_id_file,
        "secret_access_key_file": args.secret_access_key_file,
        "endpoint_url": args.endpoint_url,
        "endpoint_url_file": args.endpoint_url_file,
        "bucket": args.bucket,
        "bucket_file": args.bucket_file,
        "region": args.region,
        "region_file": args.region_file,
        "expiration_seconds": args.expiration_seconds,
    }
    if args.refresh_output_get_url:
        manifest = refresh_wam_provider_output_get_url(**common)
    else:
        if not args.bundle_path:
            parser.error("--bundle-path is required unless --refresh-output-get-url is set")
        manifest = stage_wam_provider_bundle_object_store(
            **common,
            bundle_path=args.bundle_path,
            key_prefix=args.key_prefix,
        )
    print(json.dumps(_mapping(manifest), sort_keys=True))
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
