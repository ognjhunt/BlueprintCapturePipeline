"""Prepare and verify a RunPod network-volume model cache without a Pod.

RunPod's S3-compatible API maps objects directly to files on a network volume.
This module is the storage-only transfer boundary used by an admitted CPU
builder.  It never creates GPU compute and never records credential values.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .groot_oscar_infrastructure_admission import (
    RUNPOD_S3_VOLUME_DATA_CENTER_IDS,
)
from .groot_oscar_model_cache import MANIFEST_NAME, verify_model_cache
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)


SCHEMA_VERSION = "groot_oscar_runpod_s3_model_cache.v1"
PREFLIGHT_SCHEMA_VERSION = "groot_oscar_runpod_s3_preflight.v1"
DEFAULT_REMOTE_PREFIX = ".blueprint-model-cache/blueprint-groot-oscar-v1"
REMOTE_SAFETY_HEADROOM_BYTES = 5 * 1024**3
RUNPOD_S3_MULTIPART_THRESHOLD_BYTES = 64 * 1024**2
RUNPOD_S3_MULTIPART_CHUNK_BYTES = 128 * 1024**2
RUNPOD_S3_MAX_CONCURRENCY = 1
# RunPod's S3 API has its own documented datacenter list. It is intentionally
# independent from network-volume creation capability: some volume regions do
# not expose S3, while live create evidence rejected at least one S3-listed
# region. Only the intersection is safe for storage-only preparation.
_SAFE_VOLUME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,127}")
_SAFE_PREFIX = re.compile(r"[A-Za-z0-9.][A-Za-z0-9._/-]{0,254}")
_SAFE_NONCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}")
_TRANSPORT_CAPABILITY_ISSUER = object()


class _TransportExecutionCapability:
    __slots__ = ("_consumed", "_issuer", "_lock")

    def __init__(self, issuer: object) -> None:
        self._issuer = issuer
        self._consumed = False
        self._lock = threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            if self._issuer is not _TRANSPORT_CAPABILITY_ISSUER or self._consumed:
                return False
            self._consumed = True
            return True


def _issue_transport_execution_capability(
    *,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
    remote_parent_binding: Mapping[str, Any] | None = None,
    remote_parent_capability: bytes | None = None,
    remote_packet: Mapping[str, Any] | None = None,
) -> _TransportExecutionCapability:
    """Issue from an opaque paid grant or HMAC-bound fixed remote contract."""

    if paid_resource_admission_grant is not None:
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="model_volume",
        )
        return _TransportExecutionCapability(_TRANSPORT_CAPABILITY_ISSUER)
    binding = remote_parent_binding
    capability = remote_parent_capability
    packet = remote_packet
    blockers: list[str] = []
    if not isinstance(binding, Mapping) or not isinstance(packet, Mapping):
        blockers.append("runpod_s3_remote_parent_contract_missing")
    if not isinstance(capability, bytes) or len(capability) < 32:
        blockers.append("runpod_s3_remote_parent_capability_invalid")
    if not blockers:
        assert binding is not None and packet is not None and capability is not None
        if (
            binding.get("schema_version")
            != "groot_oscar_model_cache_s3_parent_binding.v1"
            or binding.get("packet_kind") != "model_cache_s3"
            or binding.get("raw_secret_values_recorded") is not False
            or packet.get("packet_kind") != "model_cache_s3"
            or packet.get("raw_secret_values_recorded") is not False
        ):
            blockers.append("runpod_s3_remote_parent_contract_invalid")
        if hashlib.sha256(capability).hexdigest() != binding.get("capability_sha256"):
            blockers.append("runpod_s3_remote_parent_capability_digest_mismatch")
        signed = {key: value for key, value in binding.items() if key != "binding_hmac_sha256"}
        expected = hmac.new(
            capability,
            json.dumps(signed, sort_keys=True, separators=(",", ":")).encode(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(
            expected, str(binding.get("binding_hmac_sha256") or "")
        ):
            blockers.append("runpod_s3_remote_parent_binding_hmac_invalid")
        volume = packet.get("volume_evidence")
        volume = volume if isinstance(volume, Mapping) else {}
        if (
            binding.get("provider_volume_id") != volume.get("id")
            or binding.get("allocation_nonce") != packet.get("allocation_nonce")
            or volume.get("allocation_nonce") != packet.get("allocation_nonce")
        ):
            blockers.append("runpod_s3_remote_parent_volume_binding_mismatch")
    if blockers:
        raise PaidResourceAdmissionBlocked(sorted(set(blockers)))
    return _TransportExecutionCapability(_TRANSPORT_CAPABILITY_ISSUER)


def _transport_capability_valid(value: object) -> bool:
    return isinstance(value, _TransportExecutionCapability) and value.consume()


def endpoint_for_data_center(data_center_id: str) -> str:
    data_center = str(data_center_id or "").strip().upper()
    if data_center not in RUNPOD_S3_VOLUME_DATA_CENTER_IDS:
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


def _runpod_transfer_contract() -> dict[str, Any]:
    return {
        "multipart_threshold_bytes": RUNPOD_S3_MULTIPART_THRESHOLD_BYTES,
        "multipart_chunk_bytes": RUNPOD_S3_MULTIPART_CHUNK_BYTES,
        "max_concurrency": RUNPOD_S3_MAX_CONCURRENCY,
        "use_threads": False,
        "client_retry_mode": "standard",
        "client_max_attempts": 10,
    }


def _runpod_transfer_config() -> Any:
    try:
        from boto3.s3.transfer import TransferConfig
    except ImportError as exc:  # pragma: no cover - operational dependency
        raise RuntimeError("boto3_required_for_runpod_s3") from exc
    return TransferConfig(
        multipart_threshold=RUNPOD_S3_MULTIPART_THRESHOLD_BYTES,
        multipart_chunksize=RUNPOD_S3_MULTIPART_CHUNK_BYTES,
        max_concurrency=RUNPOD_S3_MAX_CONCURRENCY,
        use_threads=False,
    )


def _sanitized_s3_exception(exc: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {"error_type": type(exc).__name__}
    current: BaseException | None = exc
    visited: set[int] = set()
    for _ in range(5):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        response = getattr(current, "response", None)
        response = response if isinstance(response, Mapping) else {}
        error = response.get("Error")
        error = error if isinstance(error, Mapping) else {}
        metadata = response.get("ResponseMetadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if error.get("Code"):
            result["error_code"] = str(error["Code"])
        if getattr(current, "operation_name", None):
            result["error_operation"] = str(current.operation_name)
        if type(metadata.get("HTTPStatusCode")) is int:
            result["error_http_status"] = metadata["HTTPStatusCode"]
        if type(metadata.get("RetryAttempts")) is int:
            result["error_retry_attempts"] = metadata["RetryAttempts"]
        current = current.__cause__ or current.__context__
    return result


def _abort_visible_multipart_uploads(
    client: Any, *, volume_id: str, prefix: str
) -> dict[str, Any]:
    attempts = 0
    successes = 0
    listing_supported = True
    absence_verified = False
    try:
        response = client.list_multipart_uploads(
            Bucket=volume_id,
            Prefix=prefix.rstrip("/") + "/",
        )
        uploads = response.get("Uploads", []) if isinstance(response, Mapping) else []
        for row in uploads if isinstance(uploads, list) else []:
            if not isinstance(row, Mapping) or not row.get("Key") or not row.get("UploadId"):
                continue
            attempts += 1
            try:
                client.abort_multipart_upload(
                    Bucket=volume_id,
                    Key=str(row["Key"]),
                    UploadId=str(row["UploadId"]),
                )
                successes += 1
            except Exception:  # noqa: BLE001 - whole-volume deletion remains fail-safe
                pass
        after = client.list_multipart_uploads(
            Bucket=volume_id,
            Prefix=prefix.rstrip("/") + "/",
        )
        remaining = after.get("Uploads", []) if isinstance(after, Mapping) else []
        absence_verified = isinstance(remaining, list) and not remaining
    except Exception:  # noqa: BLE001 - provider may not implement multipart listing
        listing_supported = False
    return {
        "multipart_listing_supported": listing_supported,
        "multipart_absence_verified": absence_verified,
        "multipart_abort_attempt_count": attempts,
        "multipart_abort_success_count": successes,
    }


def _filesystem_available_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def preflight_runpod_s3(
    *,
    data_center_id: str,
    access_key_file: str | Path,
    secret_key_file: str | Path,
    perform_live_probe: bool = True,
    expected_volume_id: str | None = None,
    client: Any | None = None,
    live_probe_attempts: int = 1,
    live_probe_interval_seconds: float = 0.0,
    sleeper: Any = time.sleep,
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
    live_probe_performed = False
    if not blockers and perform_live_probe:
        attempts = max(1, min(30, int(live_probe_attempts)))
        for attempt in range(attempts):
            try:
                s3 = client or _client(
                    data_center_id=data_center,
                    access_key=access_key,
                    secret_key=secret_key,
                )
                live_probe_performed = True
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
            except Exception as exc:  # noqa: BLE001 - capability fails closed
                probe_error_type = type(exc).__name__
                if attempt + 1 < attempts:
                    sleeper(max(0.0, float(live_probe_interval_seconds)))
                    continue
                blockers.append("runpod_s3_live_credential_probe_failed")
            else:
                probe_error_type = None
                break
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
        "live_probe_performed": live_probe_performed,
        "live_probe_error_type": probe_error_type,
        "visible_network_volume_count": visible_volume_count,
        "expected_volume_id": expected_volume_id or None,
        "expected_volume_head_verified": bool(
            expected_volume_id and perform_live_probe and not blockers
        ),
        "gpu_compute_allocated": False,
        "raw_secret_values_recorded": False,
    }


def _remote_keys(client: Any, *, volume_id: str, prefix: str = "") -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        normalized_prefix = prefix.rstrip("/")
        kwargs: dict[str, Any] = {
            "Bucket": volume_id,
            "Prefix": normalized_prefix + "/" if normalized_prefix else "",
        }
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
    allocation_nonce: str,
    remote_prefix: str = DEFAULT_REMOTE_PREFIX,
    client: Any | None = None,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
    live_probe_attempts: int = 1,
    live_probe_interval_seconds: float = 0.0,
) -> dict[str, Any]:
    """Grant-gated in-process wrapper for the storage transport."""

    try:
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="model_volume",
        )
    except PaidResourceAdmissionBlocked as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [
                "runpod_s3_shared_admission_missing_or_invalid",
                *exc.blockers,
            ],
            "provider_mutations_performed": 0,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    return _upload_and_verify_model_cache_impl(
        cache_root=cache_root,
        verification_root=verification_root,
        volume_id=volume_id,
        data_center_id=data_center_id,
        access_key_file=access_key_file,
        secret_key_file=secret_key_file,
        volume_evidence=volume_evidence,
        allocation_nonce=allocation_nonce,
        remote_prefix=remote_prefix,
        client=client,
        live_probe_attempts=live_probe_attempts,
        live_probe_interval_seconds=live_probe_interval_seconds,
        execution_capability=_issue_transport_execution_capability(
            paid_resource_admission_grant=paid_resource_admission_grant
        ),
    )


def _upload_and_verify_model_cache_impl(
    *,
    cache_root: str | Path,
    verification_root: str | Path,
    volume_id: str,
    data_center_id: str,
    access_key_file: str | Path,
    secret_key_file: str | Path,
    volume_evidence: Mapping[str, Any],
    allocation_nonce: str,
    remote_prefix: str = DEFAULT_REMOTE_PREFIX,
    client: Any | None = None,
    live_probe_attempts: int = 1,
    live_probe_interval_seconds: float = 0.0,
    execution_capability: _TransportExecutionCapability | None = None,
) -> dict[str, Any]:
    """Private transport used by grant wrapper and fixed remote adapter only."""

    if not _transport_capability_valid(execution_capability):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["runpod_s3_transport_execution_capability_invalid"],
            "provider_mutations_performed": 0,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }

    volume = str(volume_id or "").strip()
    prefix = str(remote_prefix or "").strip().strip("/")
    nonce = str(allocation_nonce or "").strip()
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
    if _SAFE_NONCE.fullmatch(nonce) is None:
        contract_blockers.append("runpod_s3_allocation_nonce_invalid")
    if (
        volume_evidence.get("allocation_nonce") != nonce
        or volume_evidence.get("allocation_name_verified") is not True
        or nonce not in str(volume_evidence.get("name") or "")
    ):
        contract_blockers.append("runpod_rest_volume_allocation_identity_mismatch")
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
        live_probe_attempts=live_probe_attempts,
        live_probe_interval_seconds=live_probe_interval_seconds,
    )
    if preflight["status"] != "ready":
        return {**preflight, "schema_version": SCHEMA_VERSION}
    cache = Path(cache_root).expanduser().resolve()
    verification_parent = Path(verification_root).expanduser().resolve()
    if (
        cache == verification_parent
        or cache.is_relative_to(verification_parent)
        or verification_parent.is_relative_to(cache)
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
    required_remote = int(local["verified_size_bytes"]) + REMOTE_SAFETY_HEADROOM_BYTES
    if type(size_bytes) is not int or size_bytes < required_remote:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["runpod_rest_volume_capacity_headroom_insufficient"],
            "required_remote_capacity_bytes": required_remote,
            "provider_mutations_performed": 0,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    required_free = int(local["verified_size_bytes"]) + REMOTE_SAFETY_HEADROOM_BYTES
    ensure_dir(verification_parent)
    available = _filesystem_available_bytes(verification_parent)
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
    upload_attempt_count = 0
    upload_success_count = 0
    transfer_contract = _runpod_transfer_contract()
    transfer_config = _runpod_transfer_config()
    try:
        existing_keys = _remote_keys(s3, volume_id=volume)
        if existing_keys:
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "blocked",
                "blockers": ["runpod_s3_dedicated_volume_not_empty"],
                "provider_volume_id": volume,
                "existing_remote_object_count": len(existing_keys),
                "provider_mutations_performed": 0,
                "gpu_compute_allocated": False,
                "raw_secret_values_recorded": False,
            }
        verification = Path(
            tempfile.mkdtemp(prefix="runpod-s3-verify-", dir=verification_parent)
        )
        for path, key in zip(files, expected_keys, strict=True):
            upload_attempt_count += 1
            s3.upload_file(str(path), volume, key, Config=transfer_config)
            uploaded_keys.append(key)
            upload_success_count += 1
        observed_keys = _remote_keys(s3, volume_id=volume, prefix=prefix)
        if observed_keys != sorted(expected_keys):
            raise RuntimeError("runpod_s3_remote_inventory_mismatch")
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
        multipart_cleanup = _abort_visible_multipart_uploads(
            s3,
            volume_id=volume,
            prefix=prefix,
        )
        delete_attempt_count = 0
        delete_success_count = 0
        for key in reversed(uploaded_keys):
            delete_attempt_count += 1
            try:
                s3.delete_object(Bucket=volume, Key=key)
                delete_success_count += 1
            except Exception:  # noqa: BLE001 - report uncertain remote state
                pass
        try:
            cleanup_verified = bool(
                multipart_cleanup["multipart_listing_supported"]
                and multipart_cleanup["multipart_absence_verified"]
                and multipart_cleanup["multipart_abort_attempt_count"]
                == multipart_cleanup["multipart_abort_success_count"]
                and not _remote_keys(s3, volume_id=volume)
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
            **_sanitized_s3_exception(exc),
            "provider_volume_id": volume,
            "provider_mutations_performed": (
                upload_attempt_count
                + delete_attempt_count
                + int(multipart_cleanup["multipart_abort_attempt_count"])
            ),
            "upload_attempt_count": upload_attempt_count,
            "upload_success_count": upload_success_count,
            "uploaded_object_count_before_failure": upload_success_count,
            "upload_transfer_contract": transfer_contract,
            **multipart_cleanup,
            "cleanup_delete_attempt_count": delete_attempt_count,
            "cleanup_delete_success_count": delete_success_count,
            "partial_upload_cleanup_verified": cleanup_verified,
            "final_provider_observed_prefix_empty": cleanup_verified,
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
        "provider_mutations_performed": upload_attempt_count,
        "upload_attempt_count": upload_attempt_count,
        "upload_success_count": upload_success_count,
        "upload_transfer_contract": transfer_contract,
        "multipart_cleanup_required": False,
        "multipart_absence_verified": None,
        "multipart_abort_attempt_count": 0,
        "multipart_abort_success_count": 0,
        "cleanup_delete_attempt_count": 0,
        "cleanup_delete_success_count": 0,
        "storage_mutations_performed": True,
        "model_manifest_digest": local["model_manifest_digest"],
        "verified_size_bytes": remote["verified_size_bytes"],
        "remote_verified_file_count": remote["verified_file_count"],
        "remote_model_manifest_digest": remote["model_manifest_digest"],
        "remote_provider_volume_id": remote["provider_volume_id"],
        "remote_verification_sha256": hashlib.sha256(
            json.dumps(remote, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
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
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [
                "legacy_runpod_s3_model_cache_mutation_cli_disabled_use_paid_resource_allocator"
            ],
            "provider_mutations_performed": 0,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    if args.out:
        write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"ready", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
