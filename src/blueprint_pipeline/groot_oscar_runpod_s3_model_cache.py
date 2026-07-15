"""Prepare and verify a RunPod network-volume model cache without a Pod.

RunPod's S3-compatible API maps objects directly to files on a network volume.
This module is the storage-only transfer boundary used by an admitted CPU
builder.  It never creates GPU compute and never records credential values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .groot_oscar_model_cache import MANIFEST_NAME, verify_model_cache


SCHEMA_VERSION = "groot_oscar_runpod_s3_model_cache.v1"
PREFLIGHT_SCHEMA_VERSION = "groot_oscar_runpod_s3_preflight.v1"
DEFAULT_REMOTE_PREFIX = ".blueprint-model-cache/blueprint-groot-oscar-v1"
SUPPORTED_DATA_CENTERS = frozenset(
    {
        "EU-CZ-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-NO-1",
        "US-CA-2",
        "US-GA-2",
        "US-IL-1",
        "US-KS-2",
        "US-MD-1",
        "US-MO-1",
        "US-MO-2",
        "US-NC-1",
        "US-NC-2",
        "US-NE-1",
        "US-WA-1",
    }
)
_SAFE_VOLUME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,127}")
_SAFE_PREFIX = re.compile(r"[A-Za-z0-9.][A-Za-z0-9._/-]{0,254}")


def endpoint_for_data_center(data_center_id: str) -> str:
    data_center = str(data_center_id or "").strip().upper()
    if data_center not in SUPPORTED_DATA_CENTERS:
        raise ValueError("runpod_s3_data_center_not_supported")
    return f"https://s3api-{data_center.lower()}.runpod.io/"


def _secret_file(path: str | Path, *, label: str) -> tuple[str, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    blockers: list[str] = []
    value = ""
    try:
        stat = resolved.stat()
        if not resolved.is_file():
            blockers.append(f"{label}_file_missing")
        elif stat.st_mode & 0o077:
            blockers.append(f"{label}_file_permissions_too_open")
        else:
            value = resolved.read_text(encoding="utf-8").strip()
            if not value:
                blockers.append(f"{label}_file_empty")
    except OSError:
        blockers.append(f"{label}_file_missing")
    return value, {
        "configured": not blockers,
        "file_backed": True,
        "permissions_private": not any("permissions" in item for item in blockers),
        "raw_value_recorded": False,
        "blockers": blockers,
    }


def _client(
    *, data_center_id: str, access_key: str, secret_key: str
) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - operational dependency
        raise RuntimeError("boto3_required_for_runpod_s3") from exc
    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=data_center_id,
        endpoint_url=endpoint_for_data_center(data_center_id),
        config=Config(retries={"mode": "standard", "max_attempts": 10}, read_timeout=7200),
    )


def preflight_runpod_s3(
    *,
    data_center_id: str,
    access_key_file: str | Path,
    secret_key_file: str | Path,
    perform_live_probe: bool = True,
    expected_volume_id: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Validate file-backed S3 credentials and optionally list visible volumes."""

    blockers: list[str] = []
    data_center = str(data_center_id or "").strip().upper()
    try:
        endpoint = endpoint_for_data_center(data_center)
    except ValueError:
        endpoint = None
        blockers.append("runpod_s3_data_center_not_supported")
    access_key, access_meta = _secret_file(
        access_key_file, label="runpod_s3_access_key"
    )
    secret_key, secret_meta = _secret_file(
        secret_key_file, label="runpod_s3_secret_key"
    )
    blockers.extend(access_meta["blockers"])
    blockers.extend(secret_meta["blockers"])
    visible_volume_count: int | None = None
    if not blockers and perform_live_probe:
        try:
            s3 = client or _client(
                data_center_id=data_center,
                access_key=access_key,
                secret_key=secret_key,
            )
            response = s3.list_buckets()
            buckets = response.get("Buckets") if isinstance(response, Mapping) else None
            if not isinstance(buckets, list):
                raise ValueError("runpod_s3_list_buckets_shape_invalid")
            visible_volume_count = len(buckets)
            if expected_volume_id:
                visible = {
                    str(row.get("Name") or "")
                    for row in buckets
                    if isinstance(row, Mapping)
                }
                if expected_volume_id not in visible:
                    raise ValueError("runpod_s3_expected_volume_not_visible")
                s3.head_bucket(Bucket=expected_volume_id)
        except Exception as exc:  # noqa: BLE001 - live capability must fail closed
            blockers.append("runpod_s3_live_credential_probe_failed")
            probe_error_type = type(exc).__name__
        else:
            probe_error_type = None
    else:
        probe_error_type = None
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "data_center_id": data_center or None,
        "endpoint_url": endpoint,
        "access_key": access_meta,
        "secret_key": secret_meta,
        "live_probe_performed": bool(perform_live_probe and not access_meta["blockers"] and not secret_meta["blockers"]),
        "live_probe_error_type": probe_error_type,
        "visible_network_volume_count": visible_volume_count,
        "expected_volume_id": expected_volume_id or None,
        "expected_volume_head_verified": bool(
            expected_volume_id and perform_live_probe and not blockers
        ),
        "gpu_compute_allocated": False,
        "raw_secret_values_recorded": False,
    }


def _remote_keys(client: Any, *, volume_id: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": volume_id, "Prefix": prefix.rstrip("/") + "/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        keys.extend(
            str(row.get("Key"))
            for row in page.get("Contents", [])
            if isinstance(row, Mapping) and row.get("Key")
        )
        if page.get("IsTruncated") is not True:
            return sorted(keys)
        next_token = str(page.get("NextContinuationToken") or "")
        if not next_token or next_token == token:
            raise RuntimeError("runpod_s3_pagination_token_invalid")
        token = next_token


def upload_and_verify_model_cache(
    *,
    cache_root: str | Path,
    verification_root: str | Path,
    volume_id: str,
    data_center_id: str,
    access_key_file: str | Path,
    secret_key_file: str | Path,
    volume_evidence: Mapping[str, Any],
    remote_prefix: str = DEFAULT_REMOTE_PREFIX,
    client: Any | None = None,
    available_bytes: int | None = None,
) -> dict[str, Any]:
    """Upload every manifest-bound file, re-download it, and hash it again."""

    volume = str(volume_id or "").strip()
    prefix = str(remote_prefix or "").strip().strip("/")
    contract_blockers: list[str] = []
    if _SAFE_VOLUME_ID.fullmatch(volume) is None:
        contract_blockers.append("runpod_s3_volume_id_invalid")
    if (
        _SAFE_PREFIX.fullmatch(prefix) is None
        or ".." in prefix.split("/")
        or "//" in prefix
    ):
        contract_blockers.append("runpod_s3_remote_prefix_invalid")
    if volume_evidence.get("schema_version") != (
        "groot_oscar_runpod_network_volume_evidence.v1"
    ) or volume_evidence.get("status") != "verified":
        contract_blockers.append("runpod_rest_volume_evidence_not_verified")
    if volume_evidence.get("id") != volume:
        contract_blockers.append("runpod_rest_volume_id_mismatch")
    if volume_evidence.get("data_center_id") != str(data_center_id).upper():
        contract_blockers.append("runpod_rest_volume_data_center_mismatch")
    if contract_blockers:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": sorted(set(contract_blockers)),
            "provider_mutations_performed": 0,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    preflight = preflight_runpod_s3(
        data_center_id=data_center_id,
        access_key_file=access_key_file,
        secret_key_file=secret_key_file,
        expected_volume_id=volume,
        client=client,
    )
    if preflight["status"] != "ready":
        return {**preflight, "schema_version": SCHEMA_VERSION}
    cache = Path(cache_root).expanduser().resolve()
    verification = Path(verification_root).expanduser().resolve()
    if (
        cache == verification
        or cache.is_relative_to(verification)
        or verification.is_relative_to(cache)
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["model_cache_verification_root_overlaps_source"],
            "provider_mutations_performed": 0,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    local = verify_model_cache(cache)
    if local["status"] != "passed":
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["local_model_cache_verification_failed", *local["blockers"]],
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
    size_bytes = volume_evidence.get("size_bytes")
    if type(size_bytes) is not int or size_bytes < int(local["verified_size_bytes"]):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["runpod_rest_volume_capacity_below_model_cache"],
            "provider_mutations_performed": 0,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    required_free = int(local["verified_size_bytes"]) + 5 * 1024**3
    ensure_dir(verification.parent)
    if available_bytes is None:
        stats = os.statvfs(verification.parent)
        available = stats.f_bavail * stats.f_frsize
    else:
        available = int(available_bytes)
    if available < required_free:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["model_cache_redownload_disk_headroom_insufficient"],
            "available_bytes": available,
            "required_free_bytes": required_free,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
    access_key, _ = _secret_file(access_key_file, label="runpod_s3_access_key")
    secret_key, _ = _secret_file(secret_key_file, label="runpod_s3_secret_key")
    s3 = client or _client(
        data_center_id=str(data_center_id).upper(),
        access_key=access_key,
        secret_key=secret_key,
    )
    manifest = json.loads((cache / MANIFEST_NAME).read_text(encoding="utf-8"))
    entries = manifest.get("files") if isinstance(manifest, Mapping) else None
    entries = entries if isinstance(entries, list) else []
    files = [
        cache / str(row.get("path") or "")
        for row in entries
        if isinstance(row, Mapping)
    ]
    manifest_path = cache / MANIFEST_NAME
    files = [*files, manifest_path]
    expected_keys = [f"{prefix}/{path.relative_to(cache).as_posix()}" for path in files]
    uploaded_keys: list[str] = []
    try:
        existing_keys = _remote_keys(s3, volume_id=volume, prefix=prefix)
        if existing_keys:
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "blocked",
                "blockers": ["runpod_s3_remote_prefix_not_empty"],
                "provider_volume_id": volume,
                "existing_remote_object_count": len(existing_keys),
                "provider_mutations_performed": 0,
                "gpu_compute_allocated": False,
                "raw_secret_values_recorded": False,
            }
        for path, key in zip(files, expected_keys, strict=True):
            s3.upload_file(str(path), volume, key)
            uploaded_keys.append(key)
        observed_keys = _remote_keys(s3, volume_id=volume, prefix=prefix)
        if observed_keys != sorted(expected_keys):
            raise RuntimeError("runpod_s3_remote_inventory_mismatch")
        if verification.exists():
            shutil.rmtree(verification)
        for key in expected_keys:
            relative = key.removeprefix(prefix + "/")
            destination = verification / relative
            ensure_dir(destination.parent)
            s3.download_file(volume, key, str(destination))
        remote = verify_model_cache(
            verification,
            expected_manifest_digest=str(local["model_manifest_digest"]),
            provider_volume_id=volume,
        )
        if remote["status"] != "passed":
            raise RuntimeError("runpod_s3_redownload_verification_failed")
    except Exception as exc:  # noqa: BLE001 - preserve secret-free terminal evidence
        for key in reversed(uploaded_keys):
            try:
                s3.delete_object(Bucket=volume, Key=key)
            except Exception:  # noqa: BLE001 - report uncertain remote state
                pass
        try:
            cleanup_verified = not _remote_keys(
                s3, volume_id=volume, prefix=prefix
            )
        except Exception:  # noqa: BLE001 - absence must be provider-observed
            cleanup_verified = False
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "blockers": [
                "runpod_s3_model_cache_transfer_or_verification_failed",
                *(
                    ["runpod_s3_partial_upload_cleanup_unverified"]
                    if not cleanup_verified
                    else []
                ),
            ],
            "error_type": type(exc).__name__,
            "provider_volume_id": volume,
            "provider_mutations_performed": len(uploaded_keys),
            "uploaded_object_count_before_failure": len(uploaded_keys),
            "partial_upload_cleanup_verified": cleanup_verified,
            "outer_volume_deletion_required": not cleanup_verified,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "blockers": [],
        "provider_volume_id": volume,
        "data_center_id": str(data_center_id).upper(),
        "endpoint_url": endpoint_for_data_center(data_center_id),
        "remote_prefix": prefix,
        "remote_object_count": len(expected_keys),
        "provider_mutations_performed": len(expected_keys),
        "storage_mutations_performed": True,
        "model_manifest_digest": local["model_manifest_digest"],
        "verified_size_bytes": remote["verified_size_bytes"],
        "verification_method": "full_s3_redownload_and_sha256_manifest_verification",
        "multipart_etag_used_as_integrity_proof": False,
        "gpu_compute_allocated": False,
        "raw_secret_values_recorded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "upload-verify"))
    parser.add_argument("--data-center-id", required=True)
    parser.add_argument("--access-key-file", required=True)
    parser.add_argument("--secret-key-file", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--verification-root")
    parser.add_argument("--volume-id")
    parser.add_argument("--volume-evidence")
    parser.add_argument("--remote-prefix", default=DEFAULT_REMOTE_PREFIX)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight_runpod_s3(
            data_center_id=args.data_center_id,
            access_key_file=args.access_key_file,
            secret_key_file=args.secret_key_file,
        )
    else:
        missing = [
            name
            for name in (
                "cache_root",
                "verification_root",
                "volume_id",
                "volume_evidence",
            )
            if not getattr(args, name)
        ]
        if missing:
            parser.error("upload-verify requires --" + ", --".join(missing))
        result = upload_and_verify_model_cache(
            cache_root=args.cache_root,
            verification_root=args.verification_root,
            volume_id=args.volume_id,
            data_center_id=args.data_center_id,
            access_key_file=args.access_key_file,
            secret_key_file=args.secret_key_file,
            volume_evidence=json.loads(
                Path(args.volume_evidence).read_text(encoding="utf-8")
            ),
            remote_prefix=args.remote_prefix,
        )
    if args.out:
        write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"ready", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
