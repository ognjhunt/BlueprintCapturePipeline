"""Stage WAM provider bundles through S3-compatible object storage."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

from .common import ensure_dir, utc_now_iso, write_json


SCHEMA_VERSION = "wam_provider_object_store_staging.v1"
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
        "value_present": value_present,
        "raw_secret_values_recorded": False,
    }


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
            status = _file_status(resolved, label=label, value_present=False)
            status.update({"source": source, "read_error": type(exc).__name__})
            statuses.append(status)
            continue
        status = _file_status(resolved, label=label, value_present=bool(value))
        status["source"] = source
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
    if not blockers:
        assert boto3 is not None
        assert Config is not None
        safe_prefix = key_prefix.strip("/ ") or "blueprint/wam-provider"
        job_key = _job_key_component(resolved_job_dir)
        bundle_key = f"{safe_prefix}/{job_key}/{resolved_bundle.name}"
        output_key = f"{safe_prefix}/{job_key}/runpod_provider_runtime_output.zip"
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
            # A fresh run must not inherit a prior run's output object at this key. The output key
            # is not run-unique, so without this the poll's GET returns a stale pre-existing object
            # instantly and falsely reports a completed fresh model run (observed: a per-step
            # provider run grabbed a 128x128 placeholder in 0.72s while the real worker never ran).
            # Clear any stale output so the GET 404s until THIS run's worker uploads its result.
            try:
                client.delete_object(Bucket=bucket_value, Key=output_key)
            except Exception:  # pragma: no cover - best-effort cleanup, never blocks staging
                pass
            client.upload_file(str(resolved_bundle), bucket_value, bundle_key)
            bundle_url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_value, "Key": bundle_key},
                ExpiresIn=int(expiration_seconds),
                HttpMethod="GET",
            )
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
            upload_detail = {
                "status": "completed",
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
        "status": "completed" if bundle_url and output_put_url and not blockers else "blocked",
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
    }
    write_json(resolved_job_dir / "wam_provider_object_store_staging_manifest.json", manifest)
    return manifest


def presign_warm_inbox_channel(
    job_dir: str | Path,
    *,
    key_prefix: str = "blueprint/isaac-g1-parity",
    expiration_seconds: int = 12 * 60 * 60,
) -> dict[str, Any]:
    """Presign a single 'warm inbox' object-store key for the persistent --serve worker job channel.

    The control plane holds the presigned PUT (writes the next job); the warm pod polls the presigned
    GET (claims jobs by monotonic seq). Results ride the EXISTING worker output channel, so only this
    one extra key is needed. Reuses the same file-based secrets as :func:`stage_wam_provider_bundle_
    object_store` and writes the URLs as sensitive files in ``job_dir``. Returns a status dict with
    redacted URLs (raw URLs only ever touch the sensitive files)."""
    resolved_job_dir = Path(job_dir)
    resolved_job_dir.mkdir(parents=True, exist_ok=True)
    generated = utc_now_iso()
    expiry_metadata = _presigned_url_expiry_metadata(generated, expiration_seconds)
    access_key, _ = _read_first_file(explicit_path=None, env_name="BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID",
                                     default_paths=DEFAULT_ACCESS_KEY_FILES, label="object_store_access_key_id")
    secret_key, _ = _read_first_file(explicit_path=None, env_name="BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY",
                                     default_paths=DEFAULT_SECRET_KEY_FILES, label="object_store_secret_access_key")
    endpoint, _ = _read_first_file(explicit_path=None, env_name="BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL",
                                   default_paths=DEFAULT_ENDPOINT_FILES, label="object_store_endpoint_url",
                                   allow_env_value=True)
    bucket_value, _ = _read_first_file(explicit_path=None, env_name="BLUEPRINT_WAM_OBJECT_STORE_BUCKET",
                                       default_paths=DEFAULT_BUCKET_FILES, label="object_store_bucket")
    region_value, _ = _read_first_file(explicit_path=None, env_name="BLUEPRINT_WAM_OBJECT_STORE_REGION",
                                       default_paths=DEFAULT_REGION_FILES, label="object_store_region",
                                       allow_env_value=True)
    region_value = region_value or "us-east-1"
    blockers: list[str] = []
    if not access_key:
        blockers.append("missing_object_store_access_key_id_file")
    if not secret_key:
        blockers.append("missing_object_store_secret_access_key_file")
    if not bucket_value:
        blockers.append("missing_object_store_bucket_or_network_volume_id_file")
    inbox_get_url = ""
    inbox_put_url = ""
    inbox_key = ""
    if not blockers:
        try:
            import boto3  # type: ignore[import-not-found]
            from botocore.client import Config  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - environment dependent
            blockers.append(f"boto3_or_botocore_unavailable:{type(exc).__name__}")
    if not blockers:
        safe_prefix = key_prefix.strip("/ ") or "blueprint/isaac-g1-parity"
        inbox_key = f"{safe_prefix}/{_job_key_component(resolved_job_dir)}/warm_inbox.json"
        client_kwargs: dict[str, Any] = {
            "aws_access_key_id": access_key, "aws_secret_access_key": secret_key,
            "region_name": region_value, "config": Config(signature_version="s3v4"),
        }
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        try:
            client = boto3.client("s3", **client_kwargs)
            client.put_object(Bucket=bucket_value, Key=inbox_key,
                              Body=json.dumps({"seq": 0}).encode(), ContentType="application/json")
            inbox_get_url = client.generate_presigned_url(
                "get_object", Params={"Bucket": bucket_value, "Key": inbox_key},
                ExpiresIn=int(expiration_seconds), HttpMethod="GET")
            inbox_put_url = client.generate_presigned_url(
                "put_object", Params={"Bucket": bucket_value, "Key": inbox_key,
                                      "ContentType": "application/json"},
                ExpiresIn=int(expiration_seconds), HttpMethod="PUT")
        except Exception as exc:
            blockers.append(f"warm_inbox_presign_failed:{type(exc).__name__}")
    get_file = resolved_job_dir / "warm_inbox_get_url.txt"
    put_file = resolved_job_dir / "warm_inbox_put_url.txt"
    if inbox_get_url:
        _write_sensitive_file(get_file, inbox_get_url, label="warm_inbox_get_url")
    if inbox_put_url:
        _write_sensitive_file(put_file, inbox_put_url, label="warm_inbox_put_url")
    return {
        "status": "completed" if (inbox_get_url and inbox_put_url and not blockers) else "blocked",
        "blockers": blockers,
        "generated_at": generated,
        "presigned_url_expiry": expiry_metadata,
        "inbox_key": inbox_key,
        "expiration_seconds": int(expiration_seconds),
        "expires_at": expiry_metadata["expires_at"],
        "expiry_warning": expiry_metadata["expiry_warning"],
        "warm_inbox_get_url_file": str(get_file) if inbox_get_url else None,
        "warm_inbox_put_url_file": str(put_file) if inbox_put_url else None,
        "warm_inbox_get_url_redacted": _redact_url(inbox_get_url) if inbox_get_url else None,
        "warm_inbox_put_url_redacted": _redact_url(inbox_put_url) if inbox_put_url else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--bundle-path", required=True)
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
    manifest = stage_wam_provider_bundle_object_store(
        job_dir=args.job_dir,
        bundle_path=args.bundle_path,
        access_key_id_file=args.access_key_id_file,
        secret_access_key_file=args.secret_access_key_file,
        endpoint_url=args.endpoint_url,
        endpoint_url_file=args.endpoint_url_file,
        bucket=args.bucket,
        bucket_file=args.bucket_file,
        region=args.region,
        region_file=args.region_file,
        key_prefix=args.key_prefix,
        expiration_seconds=args.expiration_seconds,
    )
    print(json.dumps(_mapping(manifest), sort_keys=True))
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
