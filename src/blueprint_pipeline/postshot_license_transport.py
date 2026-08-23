"""Single-use, delete-acknowledged transport for Postshot licence material."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json

SCHEMA_VERSION = "postshot_license_transport.v1"
_KEYS = frozenset({"POSTSHOT_LOGIN_EMAIL", "POSTSHOT_LOGIN_PASSWORD"})


class PostshotLicenseTransportError(RuntimeError):
    pass


def _secret(path: str | Path, *, label: str) -> str:
    source = Path(path).expanduser().resolve()
    mode = stat.S_IMODE(source.stat().st_mode)
    if not source.is_file() or source.is_symlink() or mode & 0o077:
        raise PostshotLicenseTransportError(f"{label}_file_permissions_invalid")
    value = source.read_text(encoding="utf-8").strip()
    if not value:
        raise PostshotLicenseTransportError(f"{label}_file_empty")
    return value


def _license_bytes(path: str | Path) -> bytes:
    text = _secret(path, label="postshot_license")
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip():
            values[key.strip()] = value.strip()
    if set(values) != _KEYS or not all(values.values()):
        raise PostshotLicenseTransportError("postshot_license_file_schema_invalid")
    return ("\n".join(f"{key}={values[key]}" for key in sorted(values)) + "\n").encode()


def _client() -> tuple[Any, str]:  # pragma: no cover - production credential seam
    import boto3
    from botocore.client import Config

    access = _secret(os.environ["BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE"], label="object_store_access")
    secret = _secret(os.environ["BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE"], label="object_store_secret")
    bucket = _secret(os.environ["BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE"], label="object_store_bucket")
    region_path = os.environ.get("BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE")
    endpoint_path = os.environ.get("BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE")
    kwargs: dict[str, Any] = {
        "aws_access_key_id": access,
        "aws_secret_access_key": secret,
        "region_name": _secret(region_path, label="object_store_region") if region_path else "us-east-1",
        "config": Config(signature_version="s3v4"),
    }
    if endpoint_path:
        kwargs["endpoint_url"] = _secret(endpoint_path, label="object_store_endpoint")
    return boto3.client("s3", **kwargs), bucket


def stage_postshot_license(
    *,
    job_dir: str | Path,
    license_file: str | Path,
    expiration_seconds: int,
    client: Any = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Upload one run-unique object and return signed GET/DELETE URLs in memory."""

    if not 300 <= int(expiration_seconds) <= 14_400:
        raise PostshotLicenseTransportError("postshot_license_expiration_invalid")
    root = Path(job_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    body = _license_bytes(license_file)
    if client is None:
        client, resolved_bucket = _client()
    else:
        resolved_bucket = str(bucket or "")
    if not resolved_bucket:
        raise PostshotLicenseTransportError("postshot_license_bucket_missing")
    key = f"blueprint/postshot-license/{secrets.token_hex(24)}.env"
    client.put_object(
        Bucket=resolved_bucket,
        Key=key,
        Body=body,
        ContentType="text/plain",
        ServerSideEncryption="AES256",
    )
    get_url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": resolved_bucket, "Key": key, "ResponseCacheControl": "no-store"},
        ExpiresIn=int(expiration_seconds),
        HttpMethod="GET",
    )
    delete_url = client.generate_presigned_url(
        "delete_object",
        Params={"Bucket": resolved_bucket, "Key": key},
        ExpiresIn=int(expiration_seconds),
        HttpMethod="DELETE",
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "staged",
        "object_key_sha256": hashlib.sha256(key.encode()).hexdigest(),
        "license_payload_sha256": hashlib.sha256(body).hexdigest(),
        "expires_in_seconds": int(expiration_seconds),
        "single_use_delete_ack_required": True,
        "raw_signed_urls_recorded": False,
        "raw_secret_values_recorded": False,
        "staged_at": utc_now_iso(),
    }
    write_json(root / "postshot_license_transport.json", receipt)
    return {**receipt, "get_url": get_url, "delete_url": delete_url, "_key": key, "_bucket": resolved_bucket, "_client": client}


def close_postshot_license(*, staged: dict[str, Any], job_dir: str | Path) -> dict[str, Any]:
    """Delete again idempotently and prove the one credential object is absent."""

    client = staged["_client"]
    bucket = str(staged["_bucket"])
    key = str(staged["_key"])
    client.delete_object(Bucket=bucket, Key=key)
    absent = False
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # botocore ClientError is intentionally not serialized
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        status = int(response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
        absent = status == 404 or code in {"404", "NoSuchKey", "NotFound"}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "closed" if absent else "blocked",
        "object_key_sha256": staged["object_key_sha256"],
        "object_absence_confirmed": absent,
        "raw_signed_urls_recorded": False,
        "raw_secret_values_recorded": False,
        "closed_at": utc_now_iso(),
        "blockers": [] if absent else ["postshot_license_object_absence_unverified"],
    }
    write_json(Path(job_dir) / "postshot_license_transport_close.json", receipt)
    return receipt


__all__ = ["PostshotLicenseTransportError", "close_postshot_license", "stage_postshot_license"]
