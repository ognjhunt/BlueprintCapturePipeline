"""Canonical storage-only RunPod model volume prepared by one CPU builder.

This composite allocator creates no RunPod Pod. It creates one dedicated empty
network volume behind an independently armed deadline watchdog, prepares the
locked cache on the admitted DigitalOcean CPU builder, uploads and fully
redownload-verifies the cache through RunPod S3, and retains the volume only
after canary-ready verification is retrieved. Any ambiguity deletes the whole
volume and verifies provider absence.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .gpu_render_providers import _runpod_call, get_render_provider
from .groot_oscar_digitalocean_builder import (
    _host_key_material,
    _live_profile,
    _read_private_secret,
    _read_secret,
    run_builder,
)
from .groot_oscar_infrastructure_admission import (
    RUNPOD_S3_VOLUME_DATA_CENTER_IDS,
    build_build_plane_admission,
    build_runpod_network_volume_evidence,
)
from .groot_oscar_model_cache import _canonical_digest
from .groot_oscar_model_cache_s3_remote_packet import (
    prepare_remote_model_cache_packet,
)
from .groot_oscar_model_cache_wheelhouse import build_model_cache_wheelhouse
from .groot_oscar_runpod_model_volume import (
    MODEL_CACHE_PATH,
    VOLUME_NAME_PREFIX,
    WATCHDOG_HANDOFF_SCHEMA_VERSION,
    WATCHDOG_SCHEMA_VERSION,
    _delete_volume,
    _extract_id,
    _matching_resources,
    _watchdog_process_running,
)
from .groot_oscar_runpod_s3_model_cache import (
    DEFAULT_REMOTE_PREFIX,
    _client as _runpod_s3_client,
    _secret_file as _runpod_s3_secret_file,
    preflight_runpod_s3,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    load_pending_teardowns,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_provider_lane_lease import (
    acquire_paid_provider_lane_lease,
    build_paid_provider_lane_reconciliation,
    read_lease,
    read_process_argv,
    release_paid_provider_lane_lease,
    rotate_paid_provider_lane_lease_to_retention_watchdog,
    transfer_paid_provider_lane_lease_to_watchdog,
)
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission,
)
from .render_lock import _pid_is_alive


SCHEMA_VERSION = "groot_oscar_storage_model_volume_admission.v1"
RESULT_SCHEMA_VERSION = "groot_oscar_storage_model_volume_result.v1"
MIN_VOLUME_GIB = 30
MAX_VOLUME_GIB = 100
MIN_STORAGE_TTL_SECONDS = 2 * 60 * 60
MAX_STORAGE_TTL_SECONDS = 4 * 60 * 60
MIN_CACHE_RETENTION_SECONDS = 4 * 60 * 60
MAX_CACHE_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_CACHE_RETENTION_SPEND_USD = 1.0
MIN_RECONCILED_CAMPAIGN_SPEND_USD = 12.712289
BUILDER_TO_VOLUME_MARGIN_SECONDS = 35 * 60
CANARY_AND_HANDOFF_MARGIN_SECONDS = 30 * 60
MIN_LOCAL_STAGING_BYTES = 1024**3
NO_POD_PREFIX = "blueprint-storage-only-no-pod-"
PROVIDER_LANE = "groot_oscar_model_volume"
RETENTION_SCHEMA_VERSION = "groot_oscar_bounded_model_cache_retention.v1"
RETENTION_ADMISSION_SCHEMA_VERSION = (
    "groot_oscar_bounded_model_cache_retention_admission.v1"
)
RETENTION_REMOTE_VERIFICATION_SCHEMA_VERSION = (
    "groot_oscar_bounded_model_cache_remote_verification.v1"
)
RETENTION_STREAM_CHUNK_BYTES = 8 * 1024**2
RETENTION_MANIFEST_NAME = "groot_oscar_model_cache_manifest.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("storage_model_volume_json_not_object")
    return value


def _retention_watchdog_mapping(watchdog: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "watchdog_pid": watchdog.get("pid"),
        "watchdog_state_path": watchdog.get("state_path"),
        "watchdog_deadline_epoch": watchdog.get("watchdog_deadline_epoch"),
        "pod_name_prefix": watchdog.get("pod_name_prefix"),
        "volume_name": watchdog.get("volume_name"),
        "watchdog_nonce": watchdog.get("watchdog_nonce"),
    }


def _retention_source_watchdog_mapping(
    source_handoff: Mapping[str, Any],
    rotation_handoff: Mapping[str, Any],
    *,
    process_argv_probe: Any,
) -> dict[str, Any]:
    """Resolve the current retained owner after a terminal canary return."""

    source_watchdog = {
        "watchdog_pid": source_handoff.get("watchdog_pid"),
        "watchdog_state_path": source_handoff.get("watchdog_state_path"),
        "watchdog_deadline_epoch": source_handoff.get("watchdog_deadline_epoch"),
        "pod_name_prefix": source_handoff.get("pod_name_prefix"),
        "volume_name": source_handoff.get("volume_name"),
        "watchdog_nonce": source_handoff.get("watchdog_nonce"),
    }
    current_pid = rotation_handoff.get("source_owner_pid")
    if current_pid == source_watchdog.get("watchdog_pid"):
        return source_watchdog
    if type(current_pid) is not int or current_pid <= 0:
        return {"watchdog_pid": current_pid}
    argv = [str(item) for item in process_argv_probe(current_pid)]
    try:
        module_index = argv.index("-m")
    except ValueError:
        return {"watchdog_pid": current_pid}
    expected_prefix = [
        "blueprint_pipeline.groot_oscar_runpod_model_volume",
        "watchdog",
        "--state",
    ]
    if argv[module_index + 1 : module_index + 4] != expected_prefix or len(argv) != module_index + 5:
        return {"watchdog_pid": current_pid}
    state_path = Path(argv[module_index + 4]).expanduser().resolve()
    try:
        state = _load(state_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"watchdog_pid": current_pid}
    return {
        "watchdog_pid": current_pid,
        "watchdog_state_path": str(state_path),
        "watchdog_deadline_epoch": state.get("deadline_epoch"),
        "pod_name_prefix": state.get("pod_name_prefix"),
        "volume_name": state.get("volume_name"),
        "watchdog_nonce": state.get("watchdog_nonce"),
    }


def _stream_sha256(body: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = body.read(RETENTION_STREAM_CHUNK_BYTES)
        if not chunk:
            return digest.hexdigest(), size
        if not isinstance(chunk, bytes):
            raise TypeError("bounded_cache_retention_s3_body_not_bytes")
        digest.update(chunk)
        size += len(chunk)


def _verify_retained_model_cache_remote(
    *,
    volume_id: str,
    data_center_id: str,
    expected_manifest_digest: str,
    access_key_file: Path,
    secret_key_file: Path,
    client: Any | None = None,
) -> dict[str, Any]:
    """Stream and hash every retained object before rotating teardown ownership."""

    blockers: list[str] = []
    access_key, access_meta = _runpod_s3_secret_file(
        access_key_file, label="runpod_s3_access_key"
    )
    secret_key, secret_meta = _runpod_s3_secret_file(
        secret_key_file, label="runpod_s3_secret_key"
    )
    blockers.extend(access_meta.get("blockers") or [])
    blockers.extend(secret_meta.get("blockers") or [])
    s3 = client
    if not blockers and s3 is None:
        try:
            s3 = _runpod_s3_client(
                data_center_id=data_center_id,
                access_key=access_key,
                secret_key=secret_key,
            )
        except Exception as exc:  # noqa: BLE001 - secret-free evidence
            blockers.append(
                f"bounded_cache_retention_s3_client_failed:{type(exc).__name__}"
            )
    manifest: dict[str, Any] = {}
    manifest_key = f"{DEFAULT_REMOTE_PREFIX}/{RETENTION_MANIFEST_NAME}"
    if not blockers:
        try:
            response = s3.get_object(Bucket=volume_id, Key=manifest_key)
            body = response["Body"]
            try:
                raw_manifest = body.read(1024**2 + 1)
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
            if not isinstance(raw_manifest, bytes) or len(raw_manifest) > 1024**2:
                raise ValueError("retained_manifest_size_invalid")
            parsed = json.loads(raw_manifest.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("retained_manifest_not_object")
            manifest = parsed
        except Exception as exc:  # noqa: BLE001 - persist only exception class
            blockers.append(
                f"bounded_cache_retention_manifest_read_failed:{type(exc).__name__}"
            )
    entries = manifest.get("files")
    entries = entries if isinstance(entries, list) else []
    expected_rows: dict[str, tuple[int, str]] = {}
    if not blockers:
        observed_digest = str(manifest.get("manifest_digest") or "")
        if (
            observed_digest != expected_manifest_digest
            or observed_digest != _canonical_digest(manifest)
        ):
            blockers.append("bounded_cache_retention_manifest_digest_mismatch")
        for entry in entries:
            entry = entry if isinstance(entry, Mapping) else {}
            relative = str(entry.get("path") or "")
            path = PurePosixPath(relative)
            size = entry.get("size_bytes")
            digest = str(entry.get("sha256") or "")
            if not bool(
                relative
                and not path.is_absolute()
                and ".." not in path.parts
                and relative == path.as_posix()
                and relative not in expected_rows
                and type(size) is int
                and size >= 0
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ):
                blockers.append("bounded_cache_retention_manifest_entry_invalid")
                continue
            expected_rows[relative] = (size, digest)
        if (
            manifest.get("file_count") != len(expected_rows)
            or manifest.get("total_size_bytes")
            != sum(size for size, _digest in expected_rows.values())
        ):
            blockers.append("bounded_cache_retention_manifest_totals_invalid")
    verified_files = 0
    verified_bytes = 0
    if not blockers:
        try:
            remote_keys: list[str] = []
            token: str | None = None
            while True:
                request: dict[str, Any] = {
                    "Bucket": volume_id,
                    "Prefix": DEFAULT_REMOTE_PREFIX + "/",
                }
                if token:
                    request["ContinuationToken"] = token
                page = s3.list_objects_v2(**request)
                remote_keys.extend(
                    str(row.get("Key"))
                    for row in page.get("Contents", [])
                    if isinstance(row, Mapping) and row.get("Key")
                )
                if page.get("IsTruncated") is not True:
                    break
                next_token = str(page.get("NextContinuationToken") or "")
                if not next_token or next_token == token:
                    raise ValueError("retained_inventory_pagination_invalid")
                token = next_token
            expected_keys = {
                manifest_key,
                *(
                    f"{DEFAULT_REMOTE_PREFIX}/{relative}"
                    for relative in expected_rows
                ),
            }
            if set(remote_keys) != expected_keys or len(remote_keys) != len(expected_keys):
                raise ValueError("retained_inventory_mismatch")
            for relative, (expected_size, expected_digest) in sorted(
                expected_rows.items()
            ):
                response = s3.get_object(
                    Bucket=volume_id,
                    Key=f"{DEFAULT_REMOTE_PREFIX}/{relative}",
                )
                body = response["Body"]
                try:
                    observed_digest, observed_size = _stream_sha256(body)
                finally:
                    close = getattr(body, "close", None)
                    if callable(close):
                        close()
                if (
                    observed_size != expected_size
                    or observed_digest != expected_digest
                ):
                    raise ValueError("retained_object_digest_mismatch")
                verified_files += 1
                verified_bytes += observed_size
        except Exception as exc:  # noqa: BLE001 - persist only exception class
            blockers.append(
                f"bounded_cache_retention_remote_verification_failed:{type(exc).__name__}"
            )
    return {
        "schema_version": RETENTION_REMOTE_VERIFICATION_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "provider_volume_id": volume_id or None,
        "data_center_id": data_center_id or None,
        "remote_prefix": DEFAULT_REMOTE_PREFIX,
        "model_manifest_digest": manifest.get("manifest_digest"),
        "expected_model_manifest_digest": expected_manifest_digest or None,
        "verified_file_count": verified_files if not blockers else 0,
        "verified_size_bytes": verified_bytes if not blockers else 0,
        "verification_method": "full_s3_streaming_sha256_manifest_verification",
        "provider_mutations_performed": 0,
        "gpu_compute_allocated": False,
        "raw_secret_values_recorded": False,
    }


def retain_verified_model_cache(
    *,
    output_dir: Path,
    source_output_dir: Path,
    retention_ttl_seconds: int,
    storage_hourly_rate_usd: float,
    max_retention_spend_usd: float,
    campaign_spent_to_date_usd: float,
    campaign_total_spend_cap_usd: float,
    runpod_s3_access_key_file: Path,
    runpod_s3_secret_key_file: Path,
    allow_paid: bool,
    clock: Any = time.time,
    sleeper: Any = time.sleep,
    process_argv_probe: Any = read_process_argv,
    process_signaler: Any = os.kill,
) -> dict[str, Any]:
    """Rotate a verified cache into one bounded, later-canary-ready owner."""

    output = output_dir.expanduser().resolve()
    source = source_output_dir.expanduser().resolve()
    ensure_dir(output)
    result_path = output / "bounded_model_cache_retention.json"
    if result_path.exists():
        return _load(result_path)
    blockers: list[str] = []
    try:
        source_result = _load(source / "model_volume_result.json")
        volume = _load(source / "network_volume_evidence.json")
        cache = _load(source / "model_cache_verification.json")
        transport = _load(source / "model_cache_transport_result.json")
        source_handoff = _load(source / "watchdog_handoff.json")
    except (OSError, ValueError, json.JSONDecodeError):
        source_result = {}
        volume = {}
        cache = {}
        transport = {}
        source_handoff = {}
        blockers.append("bounded_cache_retention_source_evidence_unreadable")
    volume_id = str(source_result.get("volume_id") or "")
    volume_name = str(volume.get("name") or "")
    data_center_id = str(volume.get("data_center_id") or "")
    manifest_digest = str(source_result.get("model_manifest_digest") or "")
    lane_handoff = source_handoff.get("provider_lane_handoff")
    lane_handoff = lane_handoff if isinstance(lane_handoff, Mapping) else {}
    rotation_handoff: Mapping[str, Any] = lane_handoff
    lease_path_value = str(lane_handoff.get("lease_path") or "")
    if lease_path_value:
        current_lease = read_lease(
            "runpod", PROVIDER_LANE, Path(lease_path_value).parent
        )
        current_handoff = (
            current_lease.get("handoff")
            if isinstance(current_lease, Mapping)
            else None
        )
        if isinstance(current_handoff, Mapping):
            rotation_handoff = current_handoff
    binding = rotation_handoff.get("binding")
    binding = binding if isinstance(binding, Mapping) else {}
    if source_result.get("status") != "completed":
        blockers.append("bounded_cache_retention_source_not_completed")
    if not bool(
        volume.get("status") == "verified"
        and volume.get("id") == volume_id
        and volume_name
        and data_center_id in RUNPOD_S3_VOLUME_DATA_CENTER_IDS
        and type(volume.get("size_bytes")) is int
        and int(volume.get("size_bytes") or 0) > 0
    ):
        blockers.append("bounded_cache_retention_volume_evidence_invalid")
    if not bool(
        cache.get("status") == "passed"
        and cache.get("provider_volume_id") == volume_id
        and cache.get("model_manifest_digest") == manifest_digest
        and cache.get("cache_root") == MODEL_CACHE_PATH
        and cache.get("runtime_path_mapping_verified") is True
        and transport.get("status") == "completed"
        and transport.get("provider_volume_id") == volume_id
        and transport.get("model_manifest_digest") == manifest_digest
        and transport.get("multipart_absence_verified") is True
        and transport.get("multipart_cleanup_required") is False
        and transport.get("cleanup_delete_attempt_count") == 0
        and transport.get("cleanup_delete_success_count") == 0
        and transport.get("outer_volume_deletion_required") in (None, False)
    ):
        blockers.append("bounded_cache_retention_cache_verification_invalid")
    if not bool(
        source_handoff.get("schema_version") == WATCHDOG_HANDOFF_SCHEMA_VERSION
        and source_handoff.get("status") == "volume_ready_watchdog_retained"
        and source_handoff.get("volume_id") == volume_id
        and rotation_handoff.get("status")
        in {"pending_canary_acceptance", "accepted"}
        and binding.get("volume_id") == volume_id
    ):
        blockers.append("bounded_cache_retention_handoff_invalid")
    if not MIN_CACHE_RETENTION_SECONDS <= retention_ttl_seconds <= MAX_CACHE_RETENTION_SECONDS:
        blockers.append("bounded_cache_retention_ttl_out_of_bounds")
    maximum_storage_spend = (
        storage_hourly_rate_usd * retention_ttl_seconds / 3600
    )
    if not bool(
        0 < storage_hourly_rate_usd <= 0.01
        and 0 < max_retention_spend_usd <= MAX_CACHE_RETENTION_SPEND_USD
        and maximum_storage_spend <= max_retention_spend_usd
    ):
        blockers.append("bounded_cache_retention_storage_spend_invalid")
    if not bool(
        campaign_spent_to_date_usd >= MIN_RECONCILED_CAMPAIGN_SPEND_USD
        and campaign_total_spend_cap_usd == 20.0
        and campaign_spent_to_date_usd + maximum_storage_spend
        <= campaign_total_spend_cap_usd
    ):
        blockers.append("bounded_cache_retention_campaign_spend_invalid")
    if allow_paid is not True:
        blockers.append("bounded_cache_retention_paid_authorization_missing")

    provider = get_render_provider("runpod")
    key = provider._key()  # type: ignore[attr-defined]
    if not key:
        blockers.append("runpod_api_key_missing")
        live_pods: list[str] = []
        live_volumes: list[str] = []
        inventory_verified = False
    else:
        live_pods, live_volumes, inventory_verified = _matching_resources(
            key=key, pod_prefix=None, volume_prefix=None
        )
    if not inventory_verified or live_pods or live_volumes != [volume_id]:
        blockers.append("bounded_cache_retention_global_inventory_invalid")
    s3 = preflight_runpod_s3(
        data_center_id=data_center_id,
        access_key_file=runpod_s3_access_key_file,
        secret_key_file=runpod_s3_secret_key_file,
        expected_volume_id=volume_id or None,
        perform_live_probe=bool(volume_id and data_center_id),
    )
    if s3.get("status") != "ready":
        blockers.append("bounded_cache_retention_s3_visibility_unverified")
    remote_verification = _verify_retained_model_cache_remote(
        volume_id=volume_id,
        data_center_id=data_center_id,
        expected_manifest_digest=manifest_digest,
        access_key_file=runpod_s3_access_key_file,
        secret_key_file=runpod_s3_secret_key_file,
    )
    if remote_verification.get("status") != "passed":
        blockers.append("bounded_cache_retention_remote_cache_unverified")
    admission = {
        "schema_version": RETENTION_ADMISSION_SCHEMA_VERSION,
        "resource_class": "model_volume",
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "volume_id": volume_id or None,
        "data_center_id": data_center_id or None,
        "model_manifest_digest": manifest_digest or None,
        "retention_ttl_seconds": retention_ttl_seconds,
        "maximum_storage_spend_usd": maximum_storage_spend,
        "campaign_spent_to_date_usd": campaign_spent_to_date_usd,
        "campaign_total_spend_cap_usd": campaign_total_spend_cap_usd,
        "provider_inventory": {
            "api_confirmed": inventory_verified,
            "live_pod_ids": live_pods,
            "live_network_volume_ids": live_volumes,
            "whitelisted_network_volume_id": volume_id or None,
        },
        "s3_visibility": {
            "status": s3.get("status"),
            "expected_network_volume_id": volume_id or None,
        },
        "remote_cache_verification": remote_verification,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "bounded_model_cache_retention_admission.json", admission)
    try:
        require_paid_resource_admission(
            admission,
            resource_class="model_volume",
            expected_schema_version=RETENTION_ADMISSION_SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked as exc:
        result = {
            "schema_version": RETENTION_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": list(exc.blockers),
            "provider_mutations_performed": 0,
            "paid_compute_retained": False,
            "raw_secret_values_recorded": False,
        }
        write_json(result_path, result)
        return result

    deadline = float(clock()) + retention_ttl_seconds
    allocation_nonce = str(volume.get("allocation_nonce") or "")
    watchdog_process, watchdog = _arm_watchdog(
        output=output,
        deadline=deadline,
        volume_name=volume_name,
        allocation_nonce=allocation_nonce,
    )
    if not watchdog.get("armed"):
        cleanup_verified = False
        try:
            watchdog_process.terminate()
            watchdog_process.wait(timeout=10)
            cleanup_verified = not _watchdog_process_running(watchdog_process)
        except Exception:  # noqa: BLE001 - preserve fail-closed evidence
            cleanup_verified = False
        result = {
            "schema_version": RETENTION_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["bounded_cache_retention_watchdog_not_armed"],
            "retention_watchdog_cleanup_verified": cleanup_verified,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(result_path, result)
        return result
    retention_watchdog = _retention_watchdog_mapping(watchdog)
    source_watchdog = _retention_source_watchdog_mapping(
        source_handoff,
        rotation_handoff,
        process_argv_probe=process_argv_probe,
    )
    retention_binding = {
        "provider": "runpod",
        "lane": PROVIDER_LANE,
        "volume_id": volume_id,
        "pending_teardown_record": binding.get("pending_teardown_record"),
        "watchdog_nonce": watchdog.get("watchdog_nonce"),
        "watchdog_deadline_epoch": deadline,
    }
    state_path = Path(str(watchdog["state_path"]))
    try:
        write_json(
            state_path,
            {
                **_load(state_path),
                # Deadline cleanup only needs the canonical lease path and lane
                # binding.  Persist those before ownership rotates so a local
                # write failure leaves the original watchdog as exact owner.
                "provider_lane_handoff": {
                    "status": "retention_state_prepared",
                    "lease_path": rotation_handoff.get("lease_path"),
                    "binding": retention_binding,
                },
                "pending_teardown_record": binding.get("pending_teardown_record"),
                "volume_id": volume_id,
                "retention_class": "bounded_persistent_verified_model_cache",
                "model_manifest_digest": manifest_digest,
            },
        )
    except Exception as exc:  # noqa: BLE001 - source watchdog remains exact owner
        watchdog_process.terminate()
        watchdog_process.wait(timeout=10)
        result = {
            "schema_version": RETENTION_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["bounded_cache_retention_watchdog_state_write_failed"],
            "error_type": type(exc).__name__,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(result_path, result)
        return result
    try:
        rotated = rotate_paid_provider_lane_lease_to_retention_watchdog(
            rotation_handoff,
            source_watchdog=source_watchdog,
            retention_watchdog=retention_watchdog,
            expected_binding=binding,
            retention_binding=retention_binding,
            process_argv_probe=process_argv_probe,
            clock=clock,
        )
    except Exception as exc:  # noqa: BLE001 - source watchdog remains the owner
        rotated = {
            "status": "blocked",
            "blockers": ["bounded_cache_retention_lease_rotation_failed"],
            "error_type": type(exc).__name__,
        }
    if rotated.get("status") != "pending_canary_acceptance":
        watchdog_process.terminate()
        watchdog_process.wait(timeout=10)
        write_json(
            result_path,
            {
                "schema_version": RETENTION_SCHEMA_VERSION,
                "status": "blocked",
                "blockers": list(rotated.get("blockers") or []),
                "provider_mutations_performed": 0,
                "raw_secret_values_recorded": False,
            },
        )
        return _load(result_path)
    source_pid = int(source_watchdog["watchdog_pid"])
    source_argv = [str(item) for item in process_argv_probe(source_pid)]
    source_state_path = str(Path(str(source_watchdog["watchdog_state_path"])).resolve())
    canonical_suffix = [
        "-m",
        "blueprint_pipeline.groot_oscar_runpod_model_volume",
        "watchdog",
        "--state",
        source_state_path,
    ]
    source_identity_verified = any(
        source_argv[index : index + len(canonical_suffix)] == canonical_suffix
        for index in range(len(source_argv))
    )
    source_stopped = False
    termination_signal = None
    termination_error_type = None
    if source_identity_verified:
        try:
            process_signaler(source_pid, signal.SIGTERM)
            termination_signal = "SIGTERM"
            for _ in range(100):
                if not _pid_is_alive(source_pid):
                    source_stopped = True
                    break
                sleeper(0.05)
            if not source_stopped:
                process_signaler(source_pid, signal.SIGKILL)
                termination_signal = "SIGKILL"
                for _ in range(100):
                    if not _pid_is_alive(source_pid):
                        source_stopped = True
                        break
                    sleeper(0.05)
        except Exception as exc:  # noqa: BLE001 - new watchdog remains fail-safe
            termination_error_type = type(exc).__name__
    terminal = bool(source_stopped and _watchdog_process_running(watchdog_process))
    handoff = {
        "schema_version": WATCHDOG_HANDOFF_SCHEMA_VERSION,
        "status": (
            "volume_ready_watchdog_retained"
            if terminal
            else "retention_owner_transition_incomplete"
        ),
        "volume_id": volume_id,
        "volume_name": volume_name,
        "pod_name_prefix": watchdog.get("pod_name_prefix"),
        "teardown_owner": "independent_model_volume_watchdog",
        "watchdog_pid": watchdog.get("pid"),
        "watchdog_state_path": watchdog.get("state_path"),
        "watchdog_nonce": watchdog.get("watchdog_nonce"),
        "watchdog_deadline_epoch": deadline,
        "preparation_pod_absence_confirmed": True,
        "volume_presence_confirmed": True,
        "next_owner_must_arm_before_transfer": True,
        "provider_lane_handoff": rotated,
        "retention_class": "bounded_persistent_verified_model_cache",
        "model_manifest_digest": manifest_digest,
        "data_center_id": data_center_id,
        "size_bytes": volume.get("size_bytes"),
        "storage_hourly_rate_usd": storage_hourly_rate_usd,
        "maximum_retention_spend_usd": maximum_storage_spend,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "watchdog_handoff.json", handoff)
    result = {
        "schema_version": RETENTION_SCHEMA_VERSION,
        "status": "retained" if terminal else "control_plane_open",
        "blockers": [] if terminal else ["bounded_cache_retention_owner_transition_incomplete"],
        "volume_id": volume_id,
        "volume_name": volume_name,
        "data_center_id": data_center_id,
        "size_bytes": volume.get("size_bytes"),
        "model_manifest_digest": manifest_digest,
        "retention_deadline_epoch": deadline,
        "retention_ttl_seconds": retention_ttl_seconds,
        "storage_hourly_rate_usd": storage_hourly_rate_usd,
        "maximum_retention_spend_usd": maximum_storage_spend,
        "campaign_spent_to_date_usd": campaign_spent_to_date_usd,
        "campaign_total_spend_cap_usd": campaign_total_spend_cap_usd,
        "source_watchdog_pid": source_pid,
        "source_watchdog_identity_verified": source_identity_verified,
        "source_watchdog_stopped": source_stopped,
        "source_watchdog_termination_signal": termination_signal,
        "source_watchdog_termination_error_type": termination_error_type,
        "retention_watchdog_pid": watchdog.get("pid"),
        "later_canary_handoff_ready": terminal,
        "provider_mutations_performed": 0,
        "paid_compute_retained": False,
        "whitelisted_storage_resource_count": 1,
        "retention_policy": {
            "zero_paid_compute_required": True,
            "storage_resource_kind": "runpod_network_volume",
            "storage_resource_id": volume_id,
            "content_digest": manifest_digest,
            "content_mutation_policy": "no_writes_after_verification",
            "automatic_delete_at_deadline": True,
        },
        "raw_secret_values_recorded": False,
    }
    write_json(result_path, result)
    return result


def _source_identity(root: Path) -> tuple[str, str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, hashlib.sha256(b"").hexdigest(), bool(status.strip())


def _available_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def build_storage_volume_admission(
    *,
    data_center_id: str,
    volume_size_gib: int,
    storage_ttl_seconds: int,
    storage_hourly_rate_usd: float,
    max_storage_spend_usd: float,
    builder_ttl_seconds: int,
    inventory_verified_zero: bool,
    credentials_verified: bool,
    source_clean: bool,
    local_staging_bytes: int,
    paid_mutation_authorized: bool,
    watchdog_armed_before_allocation: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if data_center_id not in RUNPOD_S3_VOLUME_DATA_CENTER_IDS:
        blockers.append("storage_model_volume_data_center_not_s3_volume_capable")
    if type(volume_size_gib) is not int or not MIN_VOLUME_GIB <= volume_size_gib <= MAX_VOLUME_GIB:
        blockers.append("storage_model_volume_size_outside_30_to_100_gib")
    if (
        type(storage_ttl_seconds) is not int
        or not MIN_STORAGE_TTL_SECONDS <= storage_ttl_seconds <= MAX_STORAGE_TTL_SECONDS
    ):
        blockers.append("storage_model_volume_ttl_outside_guardrail")
    if (
        type(builder_ttl_seconds) is not int
        or builder_ttl_seconds <= 0
        or storage_ttl_seconds
        < builder_ttl_seconds
        + BUILDER_TO_VOLUME_MARGIN_SECONDS
        + CANARY_AND_HANDOFF_MARGIN_SECONDS
    ):
        blockers.append("storage_model_volume_ttl_does_not_cover_builder_and_canary")
    if (
        not isinstance(storage_hourly_rate_usd, (int, float))
        or isinstance(storage_hourly_rate_usd, bool)
        or storage_hourly_rate_usd <= 0
        or not isinstance(max_storage_spend_usd, (int, float))
        or isinstance(max_storage_spend_usd, bool)
        or max_storage_spend_usd <= 0
        or storage_hourly_rate_usd * storage_ttl_seconds / 3600
        > max_storage_spend_usd
    ):
        blockers.append("storage_model_volume_cost_exceeds_cap")
    if inventory_verified_zero is not True:
        blockers.append("storage_model_volume_preallocation_inventory_not_zero")
    if credentials_verified is not True:
        blockers.append("storage_model_volume_credentials_unverified")
    if source_clean is not True:
        blockers.append("storage_model_volume_source_not_clean")
    if type(local_staging_bytes) is not int or local_staging_bytes < MIN_LOCAL_STAGING_BYTES:
        blockers.append("storage_model_volume_local_staging_space_insufficient")
    if paid_mutation_authorized is not True:
        blockers.append("storage_model_volume_paid_mutation_not_authorized")
    if watchdog_armed_before_allocation is not True:
        blockers.append("storage_model_volume_watchdog_not_armed_before_allocation")
    return {
        "schema_version": SCHEMA_VERSION,
        "resource_class": "model_volume",
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "data_center_id": data_center_id,
        "limits": {
            "one_volume_limit": True,
            "runpod_gpu_pod_limit": 0,
            "storage_ttl_seconds": storage_ttl_seconds,
            "max_storage_spend_usd": max_storage_spend_usd,
            "builder_ttl_seconds": builder_ttl_seconds,
        },
        "raw_secret_values_recorded": False,
    }


def _arm_watchdog(
    *, output: Path, deadline: float, volume_name: str, allocation_nonce: str
) -> tuple[Any, dict[str, Any]]:
    state_path = output / "watchdog_state.json"
    pod_prefix = NO_POD_PREFIX + allocation_nonce
    watchdog_nonce = secrets.token_hex(16)
    write_json(
        state_path,
        {
            "deadline_epoch": deadline,
            "pod_name_prefix": pod_prefix,
            "volume_name": volume_name,
            "watchdog_nonce": watchdog_nonce,
        },
    )
    with (output / "watchdog.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "blueprint_pipeline.groot_oscar_runpod_model_volume",
                "watchdog",
                "--state",
                str(state_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (output / "watchdog.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    armed_path = output / "watchdog_armed.json"
    until = time.time() + 10
    armed: dict[str, Any] = {}
    while time.time() < until:
        if armed_path.is_file() and _watchdog_process_running(process):
            try:
                armed = _load(armed_path)
            except (OSError, ValueError, json.JSONDecodeError):
                armed = {}
            if (
                armed.get("schema_version") == WATCHDOG_SCHEMA_VERSION
                and armed.get("status") == "armed"
                and armed.get("pid") == process.pid
                and armed.get("watchdog_nonce") == watchdog_nonce
                and armed.get("volume_name") == volume_name
                and armed.get("pod_name_prefix") == pod_prefix
                and armed.get("deadline_epoch") == deadline
            ):
                break
        time.sleep(0.05)
    else:
        armed = {}
    return process, {
        "armed": bool(armed),
        "pid": process.pid,
        "state_path": str(state_path),
        "watchdog_nonce": watchdog_nonce,
        "pod_name_prefix": pod_prefix,
        "volume_name": volume_name,
        "watchdog_deadline_epoch": deadline,
    }


def run_storage_model_volume(
    *,
    output_dir: Path,
    repo_root: Path,
    data_center_id: str,
    volume_size_gib: int,
    storage_ttl_seconds: int,
    storage_hourly_rate_usd: float,
    max_storage_spend_usd: float,
    builder_evidence_path: Path,
    builder_spend_path: Path,
    digitalocean_token_file: Path,
    hf_token_file: Path,
    runpod_s3_access_key_file: Path,
    runpod_s3_secret_key_file: Path,
    login_private_key: Path,
    host_private_key: Path,
    ssh_key_id: int,
    region: str,
    allow_paid: bool,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    root = repo_root.expanduser().resolve()
    ensure_dir(output)
    result_path = output / "model_volume_result.json"
    if result_path.exists():
        raise ValueError("storage_model_volume_output_already_terminal")
    provider = get_render_provider("runpod")
    key = provider._key()  # type: ignore[attr-defined]
    try:
        builder_spend = _load(builder_spend_path)
        builder_evidence = _load(builder_evidence_path)
        builder_ttl = int(builder_spend["hard_ttl_seconds"])
        source_commit, source_patch, source_dirty = _source_identity(root)
        prospective_packet = {
            "packet_kind": "model_cache_s3",
            "status": "ready",
            "source_commit": source_commit,
            "source_worktree_dirty": source_dirty,
            "provider_launch_performed_by_packet": False,
            "data_center_id": data_center_id,
        }
        builder_admission = build_build_plane_admission(
            packet=prospective_packet,
            builder=builder_evidence,
            spend=builder_spend,
        )
        if builder_admission["status"] != "admitted":
            raise ValueError("storage_model_volume_builder_static_admission_blocked")
        do_token = _read_secret(digitalocean_token_file)
        _read_secret(login_private_key)
        _read_private_secret(hf_token_file)
        _read_private_secret(runpod_s3_access_key_file)
        _read_private_secret(runpod_s3_secret_key_file)
        _private, _public, host_fingerprint = _host_key_material(host_private_key)
        if host_fingerprint != builder_evidence.get("ssh_host_key_sha256"):
            raise ValueError("storage_model_volume_builder_host_key_mismatch")
        live_profile, live_builders = _live_profile(token=do_token, region=region)
        if live_profile.get("status") != "verified" or live_builders:
            raise ValueError("storage_model_volume_builder_live_profile_unverified")
        builder_maximum_cost = (
            float(live_profile["observed"]["price_hourly_usd"])
            * builder_ttl
            / 3600
        )
        if builder_maximum_cost > float(builder_spend["max_spend_usd"]):
            raise ValueError("storage_model_volume_builder_live_cost_exceeds_cap")
        s3_preflight = preflight_runpod_s3(
            data_center_id=data_center_id,
            access_key_file=runpod_s3_access_key_file,
            secret_key_file=runpod_s3_secret_key_file,
            perform_live_probe=False,
        )
        hf_token = hf_token_file.expanduser().resolve()
        hf_private = bool(
            hf_token.is_file()
            and not (hf_token.stat().st_mode & 0o077)
            and hf_token.read_text(encoding="utf-8").strip()
        )
        if not key:
            raise ValueError("runpod_api_key_missing")
        wheelhouse = build_model_cache_wheelhouse(
            lockfile_path=root / "uv.lock",
            output_dir=output / "dependency-wheelhouse",
        )
        existing_pods, existing_volumes, inventory_verified = _matching_resources(
            key=key, pod_prefix=None, volume_prefix=None
        )
    except Exception as exc:  # noqa: BLE001 - strictly pre-allocation
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked_before_allocation",
            "blockers": ["storage_model_volume_prerequisites_unavailable"],
            "error_type": type(exc).__name__,
            "provider_mutation_attempted": False,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
        write_json(result_path, result)
        return result
    allocation_nonce = secrets.token_hex(8)
    volume_name = VOLUME_NAME_PREFIX + allocation_nonce
    deadline = time.time() + storage_ttl_seconds
    reconciliation = build_paid_provider_lane_reconciliation(
        provider="runpod",
        lane=PROVIDER_LANE,
        provider_inventory={
            "api_confirmed": inventory_verified,
            "live_resource_count": len(existing_pods) + len(existing_volumes),
            "resources": [],
        },
        open_pending_teardowns=load_pending_teardowns(),
    )
    write_json(output / "provider_lane_reconciliation.json", reconciliation)
    lease = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane=PROVIDER_LANE,
        job_dir=str(output),
        ttl_seconds=storage_ttl_seconds,
        reconciliation=reconciliation,
    )
    write_json(output / "provider_lane_lease.json", lease)
    if lease.get("status") != "acquired":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked_before_allocation",
            "blockers": lease.get("blockers") or ["storage_model_volume_lane_unavailable"],
            "provider_mutation_attempted": False,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
        write_json(result_path, result)
        return result
    try:
        watch, watchdog = _arm_watchdog(
            output=output,
            deadline=deadline,
            volume_name=volume_name,
            allocation_nonce=allocation_nonce,
        )
    except Exception as exc:  # noqa: BLE001 - no provider mutation has occurred
        write_json(
            output / "watchdog_handoff.json",
            {"status": "cancelled_before_provider_allocation"},
        )
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked_before_allocation",
            "blockers": ["storage_model_volume_watchdog_start_failed"],
            "error_type": type(exc).__name__,
            "provider_mutation_attempted": False,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
        write_json(result_path, result)
        write_json(
            output / "provider_lane_release.json",
            release_paid_provider_lane_lease(
                lease,
                reason="watchdog_start_failed_before_provider_mutation",
                provider_mutation_started=False,
            ),
        )
        return result
    admission = build_storage_volume_admission(
        data_center_id=data_center_id,
        volume_size_gib=volume_size_gib,
        storage_ttl_seconds=storage_ttl_seconds,
        storage_hourly_rate_usd=storage_hourly_rate_usd,
        max_storage_spend_usd=max_storage_spend_usd,
        builder_ttl_seconds=builder_ttl,
        inventory_verified_zero=inventory_verified
        and not existing_pods
        and not existing_volumes,
        credentials_verified=s3_preflight.get("status") == "ready" and hf_private,
        source_clean=not source_dirty and wheelhouse.get("status") == "ready",
        local_staging_bytes=_available_bytes(output),
        paid_mutation_authorized=allow_paid,
        watchdog_armed_before_allocation=watchdog["armed"],
    )
    write_json(output / "model_volume_admission.json", admission)
    try:
        require_paid_resource_admission(
            admission,
            resource_class="model_volume",
            expected_schema_version=SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked:
        write_json(output / "watchdog_handoff.json", {"status": "cancelled_before_provider_allocation"})
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked_before_allocation",
            "blockers": admission["blockers"],
            "provider_mutation_attempted": False,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
        write_json(result_path, result)
        write_json(
            output / "provider_lane_release.json",
            release_paid_provider_lane_lease(
                lease,
                reason="storage_admission_blocked_before_provider_mutation",
                provider_mutation_started=False,
            ),
        )
        return result
    volume_id = ""
    success = False
    volume_teardown: dict[str, Any] = {"provider_absence_confirmed": False}
    builder_result: dict[str, Any] = {}
    lane_handoff: dict[str, Any] = {}
    error_type: str | None = None
    try:
        pending = open_pending_teardown(
            provider="runpod",
            lane=PROVIDER_LANE,
            run_id=allocation_nonce,
            resource_kind="network_volume",
            resource_name=volume_name,
            provider_location=data_center_id,
            job_dir=output,
            max_age_seconds=storage_ttl_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - no provider mutation has occurred
        write_json(
            output / "watchdog_handoff.json",
            {"status": "cancelled_before_provider_allocation"},
        )
        write_json(
            output / "provider_lane_release.json",
            release_paid_provider_lane_lease(
                lease,
                reason="pending_teardown_open_failed_before_provider_mutation",
                provider_mutation_started=False,
            ),
        )
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked_before_allocation",
            "blockers": ["storage_model_volume_pending_teardown_open_failed"],
            "error_type": type(exc).__name__,
            "provider_mutation_attempted": False,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
        write_json(result_path, result)
        return result
    write_json(output / "pending_teardown_opened.json", pending)
    try:
        locked_pods, locked_volumes, locked_inventory_verified = _matching_resources(
            key=key, pod_prefix=None, volume_prefix=None
        )
        if (
            not locked_inventory_verified
            or locked_pods
            or locked_volumes
        ):
            raise RuntimeError("storage_model_volume_inventory_changed_under_lease")
        create_http, create_response = _runpod_call(
            "POST",
            "/networkvolumes",
            {
                "dataCenterId": data_center_id,
                "name": volume_name,
                "size": volume_size_gib,
            },
            key=key,
            timeout=45,
        )
        volume_id = _extract_id(create_response if isinstance(create_response, Mapping) else {})
        if create_http not in {200, 201} or not volume_id:
            mark_pending_teardown_ambiguous(
                pending["path"],
                reason="runpod_network_volume_create_failed_or_ambiguous",
                evidence={"http_status": create_http, "volume_id_observed": bool(volume_id)},
            )
            raise RuntimeError("storage_model_volume_create_failed_or_ambiguous")
        bind_pending_teardown_instance(pending["path"], volume_id)
        get_http, volume_row = _runpod_call(
            "GET", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
        )
        volume_evidence = build_runpod_network_volume_evidence(
            provider_payload=volume_row if isinstance(volume_row, Mapping) else {},
            expected_volume_id=volume_id,
            model_cache_path=MODEL_CACHE_PATH,
            expected_name=volume_name,
            allocation_nonce=allocation_nonce,
        )
        if (
            get_http != 200
            or volume_evidence["status"] != "verified"
            or volume_evidence["data_center_id"] != data_center_id
            or volume_evidence["size_bytes"] != volume_size_gib * 1024**3
        ):
            raise RuntimeError("storage_model_volume_post_create_verification_failed")
        write_json(output / "network_volume_evidence.json", volume_evidence)
        live_s3 = preflight_runpod_s3(
            data_center_id=data_center_id,
            access_key_file=runpod_s3_access_key_file,
            secret_key_file=runpod_s3_secret_key_file,
            expected_volume_id=volume_id,
            live_probe_attempts=12,
            live_probe_interval_seconds=5,
        )
        write_json(output / "runpod_s3_live_preflight.json", live_s3)
        if live_s3["status"] != "ready":
            raise RuntimeError("storage_model_volume_s3_visibility_unverified")
        if not _watchdog_process_running(watch):
            raise RuntimeError("storage_model_volume_watchdog_exited_before_builder")
        if deadline <= time.time() + builder_ttl + BUILDER_TO_VOLUME_MARGIN_SECONDS:
            raise RuntimeError("storage_model_volume_deadline_too_near_for_builder")
        storage_handoff = {
            "schema_version": WATCHDOG_HANDOFF_SCHEMA_VERSION,
            "status": "storage_preparation_watchdog_armed",
            "volume_id": volume_id,
            "volume_name": volume_name,
            "pod_name_prefix": watchdog["pod_name_prefix"],
            "teardown_owner": "independent_model_volume_watchdog",
            "watchdog_pid": watchdog["pid"],
            "watchdog_state_path": watchdog["state_path"],
            "watchdog_nonce": watchdog["watchdog_nonce"],
            "watchdog_deadline_epoch": deadline,
            "provider_lane_lease_path": lease["path"],
            "pending_teardown_record": pending["path"],
            "raw_secret_values_recorded": False,
        }
        write_json(output / "storage_preparation_watchdog_handoff.json", storage_handoff)
        packet = prepare_remote_model_cache_packet(
            output_dir=output / "model-cache-packet",
            repo_root=root,
            source_commit=source_commit,
            source_patch_sha256=source_patch,
            source_worktree_dirty=source_dirty,
            volume_evidence=volume_evidence,
            volume_watchdog_handoff=storage_handoff,
            allocation_nonce=allocation_nonce,
            data_center_id=data_center_id,
            dependency_wheelhouse=Path(wheelhouse["wheelhouse_path"]),
            dependency_manifest_path=Path(wheelhouse["manifest_path"]),
        )
        if packet["status"] != "ready":
            raise RuntimeError("storage_model_volume_packet_not_ready")
        builder_result = run_builder(
            output_dir=output / "cpu-builder",
            packet_manifest_path=Path(packet["manifest_path"]),
            builder_evidence_path=builder_evidence_path,
            spend_path=builder_spend_path,
            token_file=digitalocean_token_file,
            docker_username_file=output / "unused-docker-username",
            docker_password_file=output / "unused-docker-password",
            login_private_key=login_private_key,
            host_private_key=host_private_key,
            ssh_key_id=ssh_key_id,
            region=region,
            allow_paid=True,
            hf_token_file=hf_token_file,
            runpod_s3_access_key_file=runpod_s3_access_key_file,
            runpod_s3_secret_key_file=runpod_s3_secret_key_file,
        )
        if (
            builder_result.get("status") != "completed"
            or builder_result.get("outer_volume_deletion_required") is not False
            or builder_result.get("provider_volume_id") != volume_id
            or not _watchdog_process_running(watch)
        ):
            raise RuntimeError("storage_model_volume_cpu_preparation_failed")
        if deadline <= time.time() + CANARY_AND_HANDOFF_MARGIN_SECONDS:
            raise RuntimeError("storage_model_volume_deadline_too_near_for_canary")
        result_dir = output / "cpu-builder/remote_results"
        canary = _load(result_dir / "external_model_cache_verification.json")
        transport = _load(result_dir / "runpod_s3_model_cache_transport_result.json")
        if (
            canary.get("status") != "passed"
            or canary.get("provider_volume_id") != volume_id
            or canary.get("cache_root") != MODEL_CACHE_PATH
            or transport.get("status") != "completed"
            or transport.get("provider_volume_id") != volume_id
            or canary.get("model_manifest_digest")
            != transport.get("model_manifest_digest")
        ):
            raise RuntimeError("storage_model_volume_canary_handoff_invalid")
        write_json(output / "model_cache_verification.json", canary)
        write_json(output / "model_cache_transport_result.json", transport)
        success = True
    except Exception as exc:  # noqa: BLE001 - whole-volume cleanup below
        error_type = type(exc).__name__
        write_json(
            output / "model_volume_error.json",
            {
                "error_type": error_type,
                "error": str(exc),
                "raw_secret_values_recorded": False,
            },
        )
    finally:
        if not success:
            candidates = [volume_id] if volume_id else []
            if not candidates:
                _pods, candidates, _verified = _matching_resources(
                    key=key,
                    pod_prefix=watchdog["pod_name_prefix"],
                    volume_prefix=volume_name,
                )
            for candidate in candidates:
                volume_teardown = _delete_volume(key=key, volume_id=candidate)
        final_pods, final_volumes, final_inventory_verified = _matching_resources(
            key=key,
            pod_prefix=watchdog["pod_name_prefix"],
            volume_prefix=volume_name,
        )
        retained = bool(
            success
            and _watchdog_process_running(watch)
            and final_inventory_verified
            and not final_pods
            and final_volumes == [volume_id]
        )
        if success and not retained:
            success = False
            for candidate in final_volumes or ([volume_id] if volume_id else []):
                volume_teardown = _delete_volume(key=key, volume_id=candidate)
            _, final_volumes, _ = _matching_resources(
                key=key,
                pod_prefix=watchdog["pod_name_prefix"],
                volume_prefix=volume_name,
            )
        if success:
            try:
                lane_handoff = transfer_paid_provider_lane_lease_to_watchdog(
                    lease,
                    watchdog_pid=int(watchdog["pid"]),
                    capability_path=output / "provider_lane_handoff.capability",
                    binding={
                        "provider": "runpod",
                        "lane": PROVIDER_LANE,
                        "volume_id": volume_id,
                        "pending_teardown_record": pending["path"],
                        "watchdog_nonce": watchdog["watchdog_nonce"],
                        "watchdog_deadline_epoch": deadline,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - teardown must still run
                lane_handoff = {
                    "status": "blocked",
                    "blockers": ["paid_provider_lane_handoff_transfer_failed"],
                    "error_type": type(exc).__name__,
                }
            if lane_handoff.get("status") != "pending_canary_acceptance":
                success = False
                error_type = "PaidProviderLaneHandoffBlocked"
                for candidate in final_volumes or ([volume_id] if volume_id else []):
                    volume_teardown = _delete_volume(key=key, volume_id=candidate)
            else:
                watchdog_state = _load(Path(watchdog["state_path"]))
                write_json(
                    Path(watchdog["state_path"]),
                    {
                        **watchdog_state,
                        "provider_lane_handoff": lane_handoff,
                        "pending_teardown_record": pending["path"],
                        "volume_id": volume_id,
                    },
                )
        if success:
            write_json(
                output / "watchdog_handoff.json",
                {
                    **storage_handoff,
                    "status": "volume_ready_watchdog_retained",
                    "volume_presence_confirmed": True,
                    "preparation_pod_absence_confirmed": True,
                    "next_owner_must_arm_before_transfer": True,
                    "provider_lane_handoff": lane_handoff,
                },
            )
        else:
            global_pods, global_volumes, global_inventory_verified = _matching_resources(
                key=key, pod_prefix=None, volume_prefix=None
            )
            terminal = bool(
                global_inventory_verified and not global_pods and not global_volumes
            )
            if terminal:
                if volume_id:
                    close_pending_teardown(
                        pending["path"],
                        {
                            "status": "PASS",
                            "provider_absence_confirmed": True,
                            "instance_id": volume_id,
                        },
                    )
                else:
                    cancel_pending_teardown(
                        pending["path"],
                        reason="provider_inventory_verified_zero_after_ambiguous_create",
                        evidence={"provider_absence_confirmed": True},
                    )
                terminal_reconciliation = build_paid_provider_lane_reconciliation(
                    provider="runpod",
                    lane=PROVIDER_LANE,
                    provider_inventory={
                        "api_confirmed": True,
                        "live_resource_count": 0,
                        "resources": [],
                    },
                    open_pending_teardowns=load_pending_teardowns(),
                )
                write_json(
                    output / "provider_lane_release.json",
                    release_paid_provider_lane_lease(
                        lease,
                        reason="storage_model_volume_failure_provider_terminal",
                        provider_mutation_started=True,
                        terminal_reconciliation=terminal_reconciliation,
                    ),
                )
            else:
                mark_pending_teardown_ambiguous(
                    pending["path"],
                    reason="storage_model_volume_failure_cleanup_unverified",
                    evidence={"provider_absence_confirmed": False},
                )
            write_json(
                output / "watchdog_handoff.json",
                {
                    "schema_version": WATCHDOG_HANDOFF_SCHEMA_VERSION,
                    "status": (
                        "failure_cleanup_provider_terminal"
                        if terminal
                        else "failure_cleanup_unverified"
                    ),
                    "volume_id": volume_id or None,
                    "failure_volume_absence_confirmed": terminal,
                    "raw_secret_values_recorded": False,
                },
            )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed" if success else "failed",
        "blockers": [] if success else ["storage_model_volume_preparation_failed"],
        "volume_id": volume_id or None,
        "data_center_id": data_center_id,
        "model_cache_path": MODEL_CACHE_PATH,
        "model_manifest_digest": builder_result.get("model_manifest_digest"),
        "storage_only": True,
        "runpod_gpu_pods_created": 0,
        "gpu_compute_allocated": False,
        "failure_volume_teardown": volume_teardown,
        "watchdog_retained_until_epoch": deadline if success else None,
        "provider_lane_lease_path": lease.get("path"),
        "pending_teardown_record": pending.get("path"),
        "provider_lane_handoff": lane_handoff,
        "maximum_storage_spend_usd": storage_hourly_rate_usd
        * storage_ttl_seconds
        / 3600,
        "maximum_builder_compute_spend_usd": builder_result.get(
            "maximum_compute_spend_usd", 0.0
        ),
        "error_type": error_type,
        "raw_secret_values_recorded": False,
    }
    write_json(result_path, result)
    return result


def launch_detached(*, output_dir: Path, run_arguments: Sequence[str]) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    if (output / "model_volume_result.json").exists():
        raise ValueError("storage_model_volume_output_already_terminal")
    lock_path = output / "supervisor.lock"
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("storage_model_volume_output_already_has_supervisor") from exc
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
        lock.write(f"created_by_pid={os.getpid()}\n")
    with (output / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "blueprint_pipeline.paid_resource_allocator",
                "model-volume-run",
                *run_arguments,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = {
        "schema_version": "groot_oscar_storage_model_volume_supervisor.v1",
        "status": "supervisor_started",
        "pid": process.pid,
        "start_new_session": True,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "supervisor_launch.json", result)
    return result
