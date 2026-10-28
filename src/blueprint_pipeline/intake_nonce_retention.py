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
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "blueprint_intake_nonce_claim.v1"
REPORT_FILE = ".nonce_retention_report"
STATE_FILE = ".nonce_retention_state"
GRACE_SECONDS = 60.0
SWEEP_TIME_BUDGET_SECONDS = 0.05
DISCOVERY_CACHE_LIMIT = 4
DISCOVERY_ENTRY_LIMIT = 2048
_NAME = re.compile(r"[0-9a-f]{64}\.json")


class NonceStoreError(RuntimeError):
    """Sanitized anti-replay state failure; the request must fail closed."""


class _DirectoryCursor:
    """One bounded-memory, process-local continuation over the existing store."""

    def __init__(self, root: Path, *, directory_fd: int):
        self.descriptor = -1
        self.iterator = None
        self.restart(root, directory_fd=directory_fd)

    def restart(self, root: Path, *, directory_fd: int) -> None:
        """Rebind an exhausted prior-tick stream to this tick's locked root."""
        self.close()
        self.descriptor = os.dup(directory_fd)
        try:
            metadata = os.fstat(self.descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise NonceStoreError("intake_nonce_store_unsafe")
            self.identity = (metadata.st_dev, metadata.st_ino)
            self.validate(root)
            # scandir(fd) owns its own directory stream; our FD anchors later
            # record reads and unlink checks even if the pathname is replaced.
            self.iterator = os.scandir(self.descriptor)
        except BaseException:
            self.close()
            raise

    def validate(self, root: Path, *, directory_fd: int | None = None) -> None:
        metadata = root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != self.identity:
            raise NonceStoreError("intake_nonce_discovery_root_changed")
        if directory_fd is not None:
            locked = os.fstat(directory_fd)
            if (locked.st_dev, locked.st_ino) != self.identity:
                raise NonceStoreError("intake_nonce_discovery_root_changed")

    def close(self) -> None:
        iterator, self.iterator = self.iterator, None
        descriptor, self.descriptor = self.descriptor, -1
        try:
            if iterator is not None:
                iterator.close()
        finally:
            if descriptor >= 0:
                os.close(descriptor)


_DISCOVERY_LOCK = threading.Lock()
_DISCOVERY_CURSORS: OrderedDict[str, _DirectoryCursor] = OrderedDict()


def _reset_discovery_cursors() -> None:
    """A child process must not advance an inherited directory stream."""
    global _DISCOVERY_LOCK
    for cursor in _DISCOVERY_CURSORS.values():
        cursor.close()
    _DISCOVERY_CURSORS.clear()
    _DISCOVERY_LOCK = threading.Lock()


os.register_at_fork(after_in_child=_reset_discovery_cursors)


@contextmanager
def _directory_cursor(root: Path, *, directory_fd: int):
    if not _DISCOVERY_LOCK.acquire(timeout=SWEEP_TIME_BUDGET_SECONDS):
        raise NonceStoreError("intake_nonce_discovery_busy")
    key = os.path.abspath(root)
    try:
        cursor = _DISCOVERY_CURSORS.pop(key, None)
        continued = cursor is not None
        if cursor is not None:
            try:
                cursor.validate(root, directory_fd=directory_fd)
            except BaseException:
                cursor.close()
                raise
        else:
            while len(_DISCOVERY_CURSORS) >= DISCOVERY_CACHE_LIMIT:
                _, evicted = _DISCOVERY_CURSORS.popitem(last=False)
                evicted.close()
            cursor = _DirectoryCursor(root, directory_fd=directory_fd)
        _DISCOVERY_CURSORS[key] = cursor
        try:
            yield cursor, continued
        except BaseException:
            _DISCOVERY_CURSORS.pop(key, None)
            cursor.close()
            raise
        if cursor.iterator is None:
            _DISCOVERY_CURSORS.pop(key, None)
    finally:
        _DISCOVERY_LOCK.release()


@contextmanager
def _locked(root: Path):
    descriptor = -1
    directory = -1
    try:
        if root.is_symlink() or not root.is_dir():
            raise NonceStoreError("intake_nonce_store_unsafe")
        directory = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        _validate_locked_root(root, directory)
        descriptor = os.open(".nonce_retention.lock", os.O_CREAT | os.O_RDWR
                             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                             | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=directory)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise NonceStoreError("intake_nonce_store_unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _validate_locked_root(root, directory)
        yield directory
        # All writes have committed to this FD while its own lock was held.
        # A renamed/replaced pathname must still refuse the operation's result.
        _validate_locked_root(root, directory)
    except OSError as exc:
        raise NonceStoreError("intake_nonce_store_unavailable") from exc
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            if directory >= 0:
                os.close(directory)


def _validate_locked_root(root: Path, directory_fd: int) -> None:
    locked = os.fstat(directory_fd)
    current = root.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(locked.st_mode) or not stat.S_ISDIR(current.st_mode)
            or (locked.st_dev, locked.st_ino) != (current.st_dev, current.st_ino)):
        raise NonceStoreError("intake_nonce_discovery_root_changed")


def _write_metadata(directory_fd: int, name: str, value: dict[str, Any]) -> None:
    """Atomically replace only private metadata in the directory being locked."""
    if name not in (STATE_FILE, REPORT_FILE):
        raise NonceStoreError("intake_nonce_metadata_invalid")
    try:
        payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise NonceStoreError("intake_nonce_metadata_invalid") from exc
    if len(payload) > 4096:
        raise NonceStoreError("intake_nonce_metadata_invalid")
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                         0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _read_private_json(path: Path, *, error_prefix: str,
                       directory_fd: int | None = None) -> tuple[dict[str, Any], os.stat_result]:
    """Refuse unsafe local state before a bounded nonblocking read."""
    descriptor = -1
    try:
        metadata = os.stat(path, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise NonceStoreError(error_prefix + "_unsafe")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0),
                             dir_fd=directory_fd)
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size > 4096):
            raise NonceStoreError(error_prefix + "_unsafe")
        payload = os.read(descriptor, 4097)
        if len(payload) > 4096:
            raise NonceStoreError(error_prefix + "_unsafe")
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise NonceStoreError(error_prefix + "_invalid")
        return value, metadata
    except FileNotFoundError:
        raise
    except (ValueError, UnicodeError) as exc:
        raise NonceStoreError(error_prefix + "_invalid") from exc
    except OSError as exc:
        raise NonceStoreError(error_prefix + "_unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_record(path: Path, *, directory_fd: int | None = None) -> tuple[dict[str, Any], os.stat_result]:
    value, metadata = _read_private_json(path, error_prefix="intake_nonce_record",
                                        directory_fd=directory_fd)
    if (value.get("schema_version") != SCHEMA_VERSION
            or any(not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", value[key])
                   for key in ("client_id_sha256", "nonce_sha256"))
            or any(type(value.get(key)) not in (int, float) or not math.isfinite(value[key])
                   for key in ("claimed_at_epoch", "expires_at_epoch"))
            or value["claimed_at_epoch"] < 0 or value["expires_at_epoch"] <= value["claimed_at_epoch"]):
        raise NonceStoreError("intake_nonce_record_invalid")
    return value, metadata


def _clock(directory_fd: int, now: float) -> dict[str, Any]:
    if not math.isfinite(now) or now < 0:
        raise NonceStoreError("intake_nonce_clock_invalid")
    path = Path(STATE_FILE)
    try:
        state, _ = _read_private_json(path, error_prefix="intake_nonce_clock_state",
                                     directory_fd=directory_fd)
    except FileNotFoundError:
        state, previous = {}, now
    else:
        try:
            previous = state["high_water_epoch"]
            if (state.get("schema_version") != "blueprint_intake_nonce_retention_state.v1"
                    or type(previous) not in (int, float) or not math.isfinite(previous) or previous < 0):
                raise ValueError("invalid")
            if now + GRACE_SECONDS < previous:
                raise NonceStoreError("intake_nonce_clock_rollback")
        except (ValueError, KeyError, TypeError) as exc:
            raise NonceStoreError("intake_nonce_clock_state_unavailable") from exc
    state = {**state, "schema_version": "blueprint_intake_nonce_retention_state.v1",
             "high_water_epoch": max(now, previous)}
    _write_metadata(directory_fd, STATE_FILE, state)
    return state


def _unlink_same(path: Path, metadata: os.stat_result, *, directory_fd: int | None = None) -> bool:
    current = os.stat(path, dir_fd=directory_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size) != (
            metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_size):
        return False
    os.unlink(path, dir_fd=directory_fd)
    if directory_fd is not None:
        os.fsync(directory_fd)
        return True
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
    path = Path(digest + ".json")
    client_digest = hashlib.sha256(client_id.encode()).hexdigest()
    nonce_digest = hashlib.sha256(nonce.encode()).hexdigest()
    with _locked(root) as directory:
        _clock(directory, now)
        try:
            existing, metadata = _read_record(path, directory_fd=directory)
        except FileNotFoundError:
            pass
        else:
            if existing["client_id_sha256"] != client_digest or existing["nonce_sha256"] != nonce_digest:
                raise NonceStoreError("intake_nonce_record_binding_mismatch")
            if existing["expires_at_epoch"] + GRACE_SECONDS >= now:
                return False
            if not _unlink_same(path, metadata, directory_fd=directory):
                raise NonceStoreError("intake_nonce_record_changed")
        expires = now + max(max_age_seconds * 2, max_age_seconds + 60)
        if not math.isfinite(expires) or expires <= now:
            raise NonceStoreError("intake_nonce_expiry_invalid")
        payload = {"schema_version": SCHEMA_VERSION, "client_id_sha256": client_digest,
                   "nonce_sha256": nonce_digest, "claimed_at_epoch": now, "expires_at_epoch": expires}
        # Refuse unexpected same-name creation even under the common lock.
        # All claim/reclaim writers must use that lock. Partial writes stay held.
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                                 0o600, dir_fd=directory)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(payload, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
        return True


def sweep(root: Path, *, now: float, limit: int = 128, apply: bool = False) -> dict[str, Any]:
    """Continue a bounded directory window, retaining all unknown state.

    Every directory entry consumes discovery budget, including unknown names.
    The process-local stream resumes on the next tick; restart or cache eviction
    starts a new survey. Neither a partial window nor EOF is a complete census
    of a directory that can change between ticks.
    """
    try:
        return _sweep_window(root, now=now, limit=limit, apply=apply)
    except BaseException:
        # Also close an existing stream when the store/clock refuses before the
        # directory context is entered, or final report persistence fails.
        with _DISCOVERY_LOCK:
            cursor = _DISCOVERY_CURSORS.pop(os.path.abspath(root), None)
            if cursor is not None:
                cursor.close()
        raise


def _sweep_window(root: Path, *, now: float, limit: int, apply: bool) -> dict[str, Any]:
    maximum = max(0, min(int(limit), 1000))
    entry_limit = min(maximum * 2 + 8, DISCOVERY_ENTRY_LIMIT) if maximum else 0
    with _locked(root) as directory:
        state = _clock(directory, now)
        counts = {"inspected": 0, "expired": 0, "fresh": 0, "unproven": 0,
                  "reclaimed": 0, "inspected_bytes": 0, "expired_bytes": 0}
        started = time.monotonic()
        deadline = started + SWEEP_TIME_BUDGET_SECONDS
        examined = 0
        pass_completed = False
        if maximum:
            with _directory_cursor(root, directory_fd=directory) as (cursor, continued):
                while examined < entry_limit and counts["inspected"] < maximum:
                    if time.monotonic() >= deadline:
                        break
                    cursor.validate(root)
                    try:
                        entry = next(cursor.iterator)
                    except StopIteration:
                        cursor.close()
                        # A cached stream can finish with most of this tick's
                        # budget unused. Scan the current locked directory once
                        # with the remaining budgets; a fresh EOF always stops.
                        if continued and time.monotonic() < deadline:
                            continued = False
                            cursor.restart(root, directory_fd=directory)
                            continue
                        pass_completed = True
                        break
                    examined += 1
                    cursor.validate(root)
                    if not _NAME.fullmatch(entry.name):
                        continue
                    counts["inspected"] += 1
                    try:
                        path = Path(entry.name)
                        record, metadata = _read_record(path, directory_fd=cursor.descriptor)
                        counts["inspected_bytes"] += metadata.st_size
                        if record["expires_at_epoch"] + GRACE_SECONDS < now:
                            counts["expired"] += 1
                            counts["expired_bytes"] += metadata.st_size
                            if apply:
                                cursor.validate(root)
                                counts["reclaimed"] += int(_unlink_same(
                                    path, metadata, directory_fd=cursor.descriptor))
                        else:
                            counts["fresh"] += 1
                    except (NonceStoreError, FileNotFoundError):
                        counts["unproven"] += 1
                cursor.validate(root)
        elapsed = time.monotonic() - started
        state.pop("last_filename", None)  # The old sorted-list cursor cannot resume a stream.
        state["discovery_continuation_scope"] = "process_lifetime"
        _write_metadata(directory, STATE_FILE, state)
        capacity = os.fstatvfs(directory)
        result = {"schema_version": "blueprint_intake_nonce_retention_report.v1", "observed_at_epoch": now,
                  "mode": "apply" if apply else "report_only", "grace_seconds": GRACE_SECONDS,
                  "inspection_limit": maximum, "canonical_filenames_discovered": counts["inspected"],
                  "discovery_entries_examined": examined, "discovery_entry_limit": entry_limit,
                  "discovery_time_budget_seconds": SWEEP_TIME_BUDGET_SECONDS,
                  "discovery_elapsed_seconds": elapsed,
                  "discovery_time_budget_reached": elapsed >= SWEEP_TIME_BUDGET_SECONDS,
                  "discovery_continuation_scope": "process_lifetime",
                  "directory_pass_completed": pass_completed, "inventory_complete": False,
                  "record_inspection_truncated": not pass_completed,
                  "free_bytes": capacity.f_bavail * capacity.f_frsize, "free_inodes": capacity.f_favail,
                  "observation_scope": "bounded_discovery_and_record_window", **counts}
        _write_metadata(directory, REPORT_FILE, result)
        return result
