"""Single-use authority for one bundle attached to a retained Arena GPU."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .spend_authority_consumption_root import consumption_root


AUTHORITY_SCHEMA_VERSION = "native_task_arena_warm_attempt_authority.v1"
SESSION_SCHEMA_VERSION = "native_task_arena_warm_session.v1"
CONSUMPTION_SCHEMA_VERSION = "native_task_arena_warm_authority_consumption.v1"
MINIMUM_REMAINING_SESSION_SECONDS = 900


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_mapping(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise ValueError(code)
    return dict(value)


def _session_digest(session: Mapping[str, Any]) -> str:
    value = dict(session)
    value.pop("session_digest", None)
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_native_task_arena_warm_session(
    session: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    """Verify that one Arena bundle is compatible with a still-owned session."""

    value = dict(session)
    runtime_source = prepared_bundle.get("runtime_source_packet") or {}
    now = time.time() if observed_now_epoch is None else observed_now_epoch
    deadline = value.get("watchdog_deadline_epoch")
    instance_id = value.get("instance_id")
    ssh_port = value.get("ssh_port")
    errors: list[str] = []
    if value.get("schema_version") != SESSION_SCHEMA_VERSION:
        errors.append("schema_invalid")
    if value.get("session_digest") != _session_digest(value):
        errors.append("digest_invalid")
    if value.get("status") != "ready" or value.get("continuing_spend") is not True:
        errors.append("session_not_ready")
    if value.get("provider") != "vast":
        errors.append("provider_invalid")
    if value.get("runtime_dependency_cache_ready") is not True:
        errors.append("dependency_cache_not_ready")
    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        errors.append("instance_id_invalid")
    if not isinstance(value.get("ssh_host"), str) or not value.get("ssh_host"):
        errors.append("ssh_host_invalid")
    if isinstance(ssh_port, bool) or not isinstance(ssh_port, int) or ssh_port <= 0:
        errors.append("ssh_port_invalid")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        errors.append("watchdog_deadline_invalid")
    elif float(deadline) - now < MINIMUM_REMAINING_SESSION_SECONDS:
        errors.append("watchdog_window_too_short")
    execution_mode = prepared_bundle.get("execution_mode")
    if execution_mode not in {"controls", "policy"}:
        errors.append("bundle_mode_invalid")
    if value.get("container_image") != prepared_bundle.get("container_image"):
        errors.append("container_image_mismatch")
    if value.get("runtime_dependency_packet_sha256") != runtime_source.get(
        "packet_sha256"
    ):
        errors.append("runtime_dependency_sha256_mismatch")
    if value.get("runtime_dependency_packet_size_bytes") != runtime_source.get(
        "packet_size_bytes"
    ):
        errors.append("runtime_dependency_size_mismatch")
    if errors:
        raise ValueError(
            "native_task_arena_warm_session_invalid:" + ",".join(sorted(set(errors)))
        )
    return value


def materialize_native_task_arena_warm_attempt_authority(
    *,
    warm_session_path: str | Path,
    bundle_receipt_path: str | Path,
    prepared_bundle: Mapping[str, Any],
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    output_path: str | Path,
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    """Seal one zero-allocation, zero-retry warm attachment authority."""

    session_path = Path(warm_session_path).expanduser().resolve()
    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    session = validate_native_task_arena_warm_session(
        _read_mapping(session_path, "native_task_arena_warm_session_unreadable"),
        prepared_bundle=prepared_bundle,
        observed_now_epoch=observed_now_epoch,
    )
    receipt = _read_mapping(
        receipt_path, "native_task_arena_warm_bundle_receipt_unreadable"
    )
    if (
        receipt.get("bundle_sha256") != prepared_bundle.get("bundle_sha256")
        or receipt.get("input_digest") != prepared_bundle.get("input_digest")
        or receipt.get("implementation_commit")
        != prepared_bundle.get("implementation_commit")
        or not authorization_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
    ):
        raise ValueError("native_task_arena_warm_authority_configuration_invalid")
    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": "one_shot_native_task_arena_warm_attachment",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 0,
        "maximum_automatic_retries": 0,
        "zero_retry": True,
        "warm_session": _record(session_path),
        "warm_session_digest": session["session_digest"],
        "provider_instance_id": session["instance_id"],
        "watchdog_deadline_epoch": session["watchdog_deadline_epoch"],
        "bundle_receipt": _record(receipt_path),
        "bundle_sha256": prepared_bundle.get("bundle_sha256"),
        "bundle_input_digest": prepared_bundle.get("input_digest"),
        "execution_mode": prepared_bundle.get("execution_mode"),
        "blueprint_commit": prepared_bundle.get("implementation_commit"),
        "container_image": prepared_bundle.get("container_image"),
        "runtime_dependency_packet_sha256": (
            prepared_bundle.get("runtime_source_packet") or {}
        ).get("packet_sha256"),
        "simulator_output_is_not_physical_evidence": True,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("native_task_arena_warm_authority_output_exists")
    ensure_dir(destination.parent)
    write_json(destination, authority)
    validate_native_task_arena_warm_attempt_authority(
        authority,
        warm_session=session,
        prepared_bundle=prepared_bundle,
        observed_now_epoch=observed_now_epoch,
    )
    return authority


def validate_native_task_arena_warm_attempt_authority(
    authority: Mapping[str, Any],
    *,
    warm_session: Mapping[str, Any],
    prepared_bundle: Mapping[str, Any],
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    value = dict(authority)
    session = validate_native_task_arena_warm_session(
        warm_session,
        prepared_bundle=prepared_bundle,
        observed_now_epoch=observed_now_epoch,
    )
    expected = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "purpose": "one_shot_native_task_arena_warm_attachment",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 0,
        "maximum_automatic_retries": 0,
        "zero_retry": True,
        "warm_session_digest": session.get("session_digest"),
        "provider_instance_id": session.get("instance_id"),
        "watchdog_deadline_epoch": session.get("watchdog_deadline_epoch"),
        "bundle_sha256": prepared_bundle.get("bundle_sha256"),
        "bundle_input_digest": prepared_bundle.get("input_digest"),
        "execution_mode": prepared_bundle.get("execution_mode"),
        "blueprint_commit": prepared_bundle.get("implementation_commit"),
        "container_image": prepared_bundle.get("container_image"),
        "runtime_dependency_packet_sha256": (
            prepared_bundle.get("runtime_source_packet") or {}
        ).get("packet_sha256"),
        "simulator_output_is_not_physical_evidence": True,
    }
    errors = [
        f"{key}_mismatch"
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    ]
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("digest_invalid")
    if errors:
        raise ValueError(
            "native_task_arena_warm_authority_invalid:"
            + ",".join(sorted(set(errors)))
        )
    return value


def consume_native_task_arena_warm_authority_once(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    digest = str(authority.get("authorization_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_authority_identity_invalid"],
        }
    root = consumption_root()
    payload = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION,
        "authorization_digest": digest,
        "bundle_sha256": authority.get("bundle_sha256"),
        "warm_session_digest": authority.get("warm_session_digest"),
        "consumed_at": utc_now_iso(),
        "maximum_provider_allocations": 0,
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat = root.stat()
        if root.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise OSError("insecure_root")
        destination = root / f"native-task-arena-warm-{digest[7:]}.json"
        temporary = root / f".native-task-arena-warm-{digest[7:]}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_authority_consumed"],
        }
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_authority_consumption_failed"],
        }
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "SESSION_SCHEMA_VERSION",
    "consume_native_task_arena_warm_authority_once",
    "materialize_native_task_arena_warm_attempt_authority",
    "validate_native_task_arena_warm_attempt_authority",
    "validate_native_task_arena_warm_session",
]
