"""Private, bounded anti-replay retention using the existing nonce files.

Claims and expiry reclamation share one across-process lock. Canonical claim
files remain the restart evidence; malformed or unsafe records are retained and
reported, never interpreted as expired. Maintenance is report-only by default.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import write_json

SCHEMA_VERSION = "blueprint_intake_nonce_claim.v1"
REPORT_FILE = ".nonce_retention_report"
STATE_FILE = ".nonce_retention_state"
GRACE_SECONDS = 60.0
_NAME = re.compile(r"[0-9a-f]{64}\.json")


class NonceStoreError(RuntimeError):
    """Sanitized anti-replay state failure; the request must fail closed."""


@contextmanager
def _locked(root: Path):
    descriptor = -1
    try:
        if root.is_symlink() or not root.is_dir():
            raise NonceStoreError("intake_nonce_store_unsafe")
        descriptor = os.open(root / ".nonce_retention.lock", os.O_CREAT | os.O_RDWR
                             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise NonceStoreError("intake_nonce_store_unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except OSError as exc:
        raise NonceStoreError("intake_nonce_store_unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_record(path: Path) -> tuple[dict[str, Any], os.stat_result]:
    descriptor = -1
    try:
        if path.is_symlink():
            raise NonceStoreError("intake_nonce_record_unsafe")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0))
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size > 4096):
            raise NonceStoreError("intake_nonce_record_unsafe")
        payload = os.read(descriptor, 4097)
        value = json.loads(payload.decode("utf-8"))
        if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION
                or any(not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", value[key])
                       for key in ("client_id_sha256", "nonce_sha256"))
                or any(type(value.get(key)) not in (int, float) or not math.isfinite(value[key])
                       for key in ("claimed_at_epoch", "expires_at_epoch"))
                or value["claimed_at_epoch"] < 0 or value["expires_at_epoch"] <= value["claimed_at_epoch"]):
            raise NonceStoreError("intake_nonce_record_invalid")
        return value, metadata
    except FileNotFoundError:
        raise
    except (ValueError, UnicodeError) as exc:
        raise NonceStoreError("intake_nonce_record_invalid") from exc
    except OSError as exc:
        raise NonceStoreError("intake_nonce_record_unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _clock(root: Path, now: float) -> dict[str, Any]:
    if not math.isfinite(now) or now < 0:
        raise NonceStoreError("intake_nonce_clock_invalid")
    path = root / STATE_FILE
    state = {}
    if path.exists() or path.is_symlink():
        try:
            if path.is_symlink():
                raise ValueError("unsafe")
            state = json.loads(path.read_text())
            previous = state["high_water_epoch"]
            if (state.get("schema_version") != "blueprint_intake_nonce_retention_state.v1"
                    or type(previous) not in (int, float) or not math.isfinite(previous) or previous < 0):
                raise ValueError("invalid")
            if now + GRACE_SECONDS < previous:
                raise NonceStoreError("intake_nonce_clock_rollback")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise NonceStoreError("intake_nonce_clock_state_unavailable") from exc
    else:
        previous = now
    state = {**state, "schema_version": "blueprint_intake_nonce_retention_state.v1",
             "high_water_epoch": max(now, previous)}
    write_json(path, state)
    return state


def _unlink_same(path: Path, metadata: os.stat_result) -> bool:
    current = path.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size) != (
            metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_size):
        return False
    path.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def claim(root: Path, *, client_id: str, nonce: str, now: float, max_age_seconds: float) -> bool:
    if (type(max_age_seconds) not in (int, float) or not math.isfinite(max_age_seconds)
            or max_age_seconds <= 0):
        raise NonceStoreError("intake_nonce_expiry_invalid")
    digest = hashlib.sha256(f"{client_id}\0{nonce}".encode()).hexdigest()
    path = root / (digest + ".json")
    client_digest = hashlib.sha256(client_id.encode()).hexdigest()
    nonce_digest = hashlib.sha256(nonce.encode()).hexdigest()
    with _locked(root):
        _clock(root, now)
        try:
            existing, metadata = _read_record(path)
        except FileNotFoundError:
            pass
        else:
            if existing["client_id_sha256"] != client_digest or existing["nonce_sha256"] != nonce_digest:
                raise NonceStoreError("intake_nonce_record_binding_mismatch")
            if existing["expires_at_epoch"] + GRACE_SECONDS >= now:
                return False
            if not _unlink_same(path, metadata):
                raise NonceStoreError("intake_nonce_record_changed")
        expires = now + max(max_age_seconds * 2, max_age_seconds + 60)
        if not math.isfinite(expires) or expires <= now:
            raise NonceStoreError("intake_nonce_expiry_invalid")
        payload = {"schema_version": SCHEMA_VERSION, "client_id_sha256": client_digest,
                   "nonce_sha256": nonce_digest, "claimed_at_epoch": now, "expires_at_epoch": expires}
        # Refuse unexpected same-name creation even under the common lock.
        # All claim/reclaim writers must use that lock. Partial writes stay held.
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(payload, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True


def sweep(root: Path, *, now: float, limit: int = 128, apply: bool = False) -> dict[str, Any]:
    """Inspect a bounded number of canonical records; retain all unknown state.

    Filename discovery is separate from the bounded record-inspection budget.
    The report explicitly scopes byte/count observations to the inspected
    window; it is never a claim about total used or reclaimable storage.
    """
    maximum = max(0, min(int(limit), 1000))
    with _locked(root):
        state = _clock(root, now)
        names = sorted(path.name for path in root.iterdir() if _NAME.fullmatch(path.name))
        cursor = state.get("last_filename")
        if cursor in names:
            pivot = names.index(cursor) + 1
            names = names[pivot:] + names[:pivot]
        counts = {"inspected": 0, "expired": 0, "fresh": 0, "unproven": 0,
                  "reclaimed": 0, "inspected_bytes": 0, "expired_bytes": 0}
        for name in names[:maximum]:
            counts["inspected"] += 1
            try:
                record, metadata = _read_record(root / name)
                counts["inspected_bytes"] += metadata.st_size
                if record["expires_at_epoch"] + GRACE_SECONDS < now:
                    counts["expired"] += 1
                    counts["expired_bytes"] += metadata.st_size
                    if apply:
                        counts["reclaimed"] += int(_unlink_same(root / name, metadata))
                else:
                    counts["fresh"] += 1
            except (NonceStoreError, FileNotFoundError):
                counts["unproven"] += 1
        if counts["inspected"]:
            state["last_filename"] = names[counts["inspected"] - 1]
            write_json(root / STATE_FILE, state)
        capacity = os.statvfs(root)
        result = {"schema_version": "blueprint_intake_nonce_retention_report.v1", "observed_at_epoch": now,
                  "mode": "apply" if apply else "report_only", "grace_seconds": GRACE_SECONDS,
                  "inspection_limit": maximum, "canonical_filenames_discovered": len(names),
                  "record_inspection_truncated": len(names) > counts["inspected"],
                  "free_bytes": capacity.f_bavail * capacity.f_frsize, "free_inodes": capacity.f_favail,
                  "observation_scope": "bounded_record_window", **counts}
        write_json(root / REPORT_FILE, result)
        return result
