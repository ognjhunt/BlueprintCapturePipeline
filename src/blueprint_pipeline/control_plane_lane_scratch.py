"""Sealed, owned leases for manually created lane scratch folders.

Creation publishes a prepared folder and its lease in one rename. Renewal and
release compare the sealed digest while holding the lane-root lock; neither
operation removes payload bytes.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "control_plane_lane_scratch.v1"
LEASE_FILE = ".lane-scratch.v1.json"
DEFAULT_ROOT = Path("/mnt/blueprint-work/lanes")
MAX_TTL_SECONDS = 14 * 86400
MAX_LEASE_BYTES = 8192
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_REASON = re.compile(r"[a-z][a-z0-9_:+.-]{0,79}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class LaneScratchError(RuntimeError):
    """The lane scratch request or its saved lease is unsafe."""


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise LaneScratchError(f"lane_scratch_{field}_invalid")
    return value


def _ttl(value: Any) -> int:
    if type(value) is not int or not 0 < value <= MAX_TTL_SECONDS:
        raise LaneScratchError("lane_scratch_ttl_invalid")
    return value


def _now(clock: Callable[[], float]) -> float:
    try:
        value = float(clock())
    except (TypeError, ValueError, OverflowError) as exc:
        raise LaneScratchError("lane_scratch_clock_invalid") from exc
    if not math.isfinite(value) or value < 0:
        raise LaneScratchError("lane_scratch_clock_invalid")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    lease = {**value, "lease_digest": ""}
    lease["lease_digest"] = canonical_digest(lease, digest_field="lease_digest")
    return lease


@contextmanager
def _locked_root(root: str | Path) -> Iterator[int]:
    path = Path(root)
    if not path.is_absolute():
        raise LaneScratchError("lane_scratch_root_not_absolute")
    try:
        root_fd = os.open(path, _DIR_FLAGS)
    except OSError as exc:
        raise LaneScratchError("lane_scratch_root_unsafe") from exc
    try:
        try:
            lock_fd = os.open(".lane-scratch.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                              0o600, dir_fd=root_fd)
        except OSError as exc:
            raise LaneScratchError("lane_scratch_lock_unsafe") from exc
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield root_fd
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(root_fd)


@contextmanager
def _lane_fd(root_fd: int, lane: str, *, create: bool) -> Iterator[int]:
    try:
        if create:
            try:
                os.mkdir(lane, 0o750, dir_fd=root_fd)
            except FileExistsError:
                pass
        lane_fd = os.open(lane, _DIR_FLAGS, dir_fd=root_fd)
    except OSError as exc:
        raise LaneScratchError("lane_scratch_lane_unsafe") from exc
    try:
        yield lane_fd
    finally:
        os.close(lane_fd)


def _write_lease(directory_fd: int, lease: Mapping[str, Any], *, replace: bool) -> None:
    payload = (json.dumps(lease, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(payload) > MAX_LEASE_BYTES:
        raise LaneScratchError("lane_scratch_lease_too_large")
    name = f".{LEASE_FILE}.{secrets.token_hex(8)}.tmp" if replace else LEASE_FILE
    try:
        file_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                          0o640, dir_fd=directory_fd)
        try:
            with os.fdopen(file_fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            os.unlink(name, dir_fd=directory_fd)
            raise
        if replace:
            os.replace(name, LEASE_FILE, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        if replace:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise LaneScratchError("lane_scratch_write_failed") from exc


def _read_lease(directory_fd: int) -> dict[str, Any]:
    try:
        file_fd = os.open(LEASE_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                          dir_fd=directory_fd)
        with os.fdopen(file_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_LEASE_BYTES:
                raise LaneScratchError("lane_scratch_lease_unsafe")
            payload = stream.read(MAX_LEASE_BYTES + 1)
        if len(payload) > MAX_LEASE_BYTES:
            raise LaneScratchError("lane_scratch_lease_unsafe")
        lease = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaneScratchError("lane_scratch_lease_unreadable") from exc
    if (not isinstance(lease, dict) or lease.get("schema_version") != SCHEMA_VERSION
            or not isinstance(lease.get("lease_digest"), str)
            or _DIGEST.fullmatch(lease["lease_digest"]) is None
            or lease["lease_digest"] != canonical_digest(lease, digest_field="lease_digest")):
        raise LaneScratchError("lane_scratch_lease_invalid")
    return lease


def create_lane_scratch(
    lane: str, name: str, *, owner: str, reason: str, class_intent: str,
    cleanup: str, ttl_seconds: int, run_ref: str | None = None,
    scene_ref: str | None = None, size_budget_bytes: int | None = None,
    root: str | Path = DEFAULT_ROOT, now: Callable[[], float] = time.time,
) -> Path:
    """Create a new scratch folder with a sealed lease in one publication."""

    lane, name, owner = _id(lane, "lane"), _id(name, "name"), _id(owner, "owner")
    if not isinstance(reason, str) or _REASON.fullmatch(reason) is None:
        raise LaneScratchError("lane_scratch_reason_invalid")
    if class_intent not in ("cache", "evidence", "scratch"):
        raise LaneScratchError("lane_scratch_class_invalid")
    if cleanup not in ("delete", "offload", "owner_review"):
        raise LaneScratchError("lane_scratch_cleanup_invalid")
    if (run_ref is None) == (scene_ref is None):
        raise LaneScratchError("lane_scratch_reference_invalid")
    reference = _id(run_ref if run_ref is not None else scene_ref, "reference")
    if class_intent == "cache" and (type(size_budget_bytes) is not int or size_budget_bytes <= 0):
        raise LaneScratchError("lane_scratch_cache_budget_invalid")
    if size_budget_bytes is not None and (type(size_budget_bytes) is not int or size_budget_bytes <= 0):
        raise LaneScratchError("lane_scratch_budget_invalid")
    ttl = _ttl(ttl_seconds)
    observed = _now(now)
    lease = _seal({
        "schema_version": SCHEMA_VERSION, "lane": lane, "name": name,
        "owner": owner, "reason": reason, "class_intent": class_intent,
        "cleanup": cleanup, "created_at_epoch": observed,
        "expires_at_epoch": observed + ttl, "released_at_epoch": None,
        "size_budget_bytes": size_budget_bytes,
        "run_ref" if run_ref is not None else "scene_ref": reference,
    })
    with _locked_root(root) as root_fd, _lane_fd(root_fd, lane, create=True) as lane_fd:
        try:
            os.stat(name, dir_fd=lane_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise LaneScratchError("lane_scratch_exists")
        staging = f".{name}.{secrets.token_hex(8)}.tmp"
        try:
            os.mkdir(staging, 0o750, dir_fd=lane_fd)
            stage_fd = os.open(staging, _DIR_FLAGS, dir_fd=lane_fd)
            try:
                _write_lease(stage_fd, lease, replace=False)
            finally:
                os.close(stage_fd)
            os.rename(staging, name, src_dir_fd=lane_fd, dst_dir_fd=lane_fd)
            os.fsync(lane_fd)
        except OSError as exc:
            raise LaneScratchError("lane_scratch_publish_failed") from exc
        finally:
            try:
                stage_fd = os.open(staging, _DIR_FLAGS, dir_fd=lane_fd)
            except FileNotFoundError:
                pass
            else:
                try:
                    os.unlink(LEASE_FILE, dir_fd=stage_fd)
                except FileNotFoundError:
                    pass
                finally:
                    os.close(stage_fd)
                os.rmdir(staging, dir_fd=lane_fd)
    return Path(root) / lane / name


def _change_lease(
    *, root: str | Path, lane: str, name: str, owner: str, expected_digest: str,
    now: Callable[[], float], ttl_seconds: int | None,
) -> dict[str, Any]:
    lane, name, owner = _id(lane, "lane"), _id(name, "name"), _id(owner, "owner")
    if not isinstance(expected_digest, str) or _DIGEST.fullmatch(expected_digest) is None:
        raise LaneScratchError("lane_scratch_digest_invalid")
    ttl = _ttl(ttl_seconds) if ttl_seconds is not None else None
    observed = _now(now)
    with _locked_root(root) as root_fd, _lane_fd(root_fd, lane, create=False) as lane_fd:
        try:
            folder_fd = os.open(name, _DIR_FLAGS, dir_fd=lane_fd)
        except OSError as exc:
            raise LaneScratchError("lane_scratch_folder_unsafe") from exc
        try:
            lease = _read_lease(folder_fd)
            if (lease.get("lane") != lane or lease.get("name") != name
                    or lease.get("owner") != owner or lease["lease_digest"] != expected_digest):
                raise LaneScratchError("lane_scratch_lease_changed")
            if lease.get("released_at_epoch") is not None:
                raise LaneScratchError("lane_scratch_already_released")
            if ttl is None:
                lease["released_at_epoch"] = observed
            else:
                lease["expires_at_epoch"] = observed + ttl
                lease["renewed_at_epoch"] = observed
            sealed = _seal(lease)
            _write_lease(folder_fd, sealed, replace=True)
            return sealed
        finally:
            os.close(folder_fd)


def renew_lane_scratch(
    *, root: str | Path = DEFAULT_ROOT, lane: str, name: str, owner: str,
    expected_digest: str, ttl_seconds: int, now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Renew a lease only if the caller saw its latest sealed version."""

    return _change_lease(root=root, lane=lane, name=name, owner=owner,
                         expected_digest=expected_digest, ttl_seconds=ttl_seconds, now=now)


def release_lane_scratch(
    *, root: str | Path = DEFAULT_ROOT, lane: str, name: str, owner: str,
    expected_digest: str, now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """End the lease; leave the folder and payload for the governed GC phase."""

    return _change_lease(root=root, lane=lane, name=name, owner=owner,
                         expected_digest=expected_digest, ttl_seconds=None, now=now)


def list_lane_scratch(
    *, root: str | Path = DEFAULT_ROOT, lane: str, limit: int = 50, offset: int = 0,
) -> dict[str, Any]:
    """Read at most one bounded page of sealed leases beneath a named lane."""

    lane = _id(lane, "lane")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise LaneScratchError("lane_scratch_limit_invalid")
    if type(offset) is not int or not 0 <= offset <= 10000:
        raise LaneScratchError("lane_scratch_offset_invalid")
    with _locked_root(root) as root_fd, _lane_fd(root_fd, lane, create=False) as lane_fd:
        names = os.listdir(lane_fd)
        if len(names) > 10000:
            raise LaneScratchError("lane_scratch_listing_too_large")
        rows: list[dict[str, Any]] = []
        skipped = 0
        for name in sorted(names):
            if _ID.fullmatch(name) is None:
                continue
            try:
                folder_fd = os.open(name, _DIR_FLAGS, dir_fd=lane_fd)
            except OSError:
                skipped += 1
                continue
            try:
                lease = _read_lease(folder_fd)
            except LaneScratchError:
                skipped += 1
                continue
            finally:
                os.close(folder_fd)
            if lease.get("lane") != lane or lease.get("name") != name:
                skipped += 1
                continue
            rows.append({key: lease.get(key) for key in (
                "name", "owner", "run_ref", "scene_ref", "class_intent", "cleanup",
                "expires_at_epoch", "released_at_epoch", "size_budget_bytes", "lease_digest",
            )})
            if len(rows) > offset + limit:
                break
        page = rows[offset:offset + limit]
        next_offset = offset + len(page) if len(rows) > offset + limit else None
        return {"lane": lane, "leases": page, "next_offset": next_offset, "skipped": skipped}


__all__ = ["DEFAULT_ROOT", "LEASE_FILE", "LaneScratchError", "create_lane_scratch",
           "renew_lane_scratch", "release_lane_scratch", "list_lane_scratch"]
