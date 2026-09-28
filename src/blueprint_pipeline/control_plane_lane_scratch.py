"""Sealed, owned leases for manually created lane scratch folders.

Creation publishes a prepared folder and its lease in one rename. Renewal and
release compare the sealed digest while holding the lane-root lock; neither
operation removes payload bytes.
"""

from __future__ import annotations

import fcntl
import ctypes
import json
import math
import os
import re
import secrets
import stat
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "control_plane_lane_scratch.v1"
LEASE_FILE = ".lane-scratch.v1.json"
DEFAULT_ROOT = Path("/mnt/blueprint-work/lanes")
MAX_TTL_SECONDS = 14 * 86400
MAX_LEASE_BYTES = 8192
CONSUMER_LIFETIME_PROTOCOL = "leased_scratch_use.v1"
_NO_PROTOCOL = object()
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


def _lease_fields_valid(lease: Mapping[str, Any]) -> bool:
    try:
        if ("consumer_lifetime_contract" in lease
                and lease["consumer_lifetime_contract"] != CONSUMER_LIFETIME_PROTOCOL):
            return False
        for field in ("lane", "name", "owner"):
            _id(lease.get(field), field)
        if not isinstance(lease.get("reason"), str) or _REASON.fullmatch(lease["reason"]) is None:
            return False
        if lease.get("class_intent") not in ("cache", "evidence", "scratch"):
            return False
        if lease.get("cleanup") not in ("delete", "offload", "owner_review"):
            return False
        if ("run_ref" in lease) == ("scene_ref" in lease):
            return False
        _id(lease.get("run_ref", lease.get("scene_ref")), "reference")
        budget = lease.get("size_budget_bytes")
        if lease["class_intent"] == "cache" and (type(budget) is not int or budget <= 0):
            return False
        if budget is not None and (type(budget) is not int or budget <= 0):
            return False
        created = float(lease["created_at_epoch"])
        renewed = float(lease.get("renewed_at_epoch", created))
        expires = float(lease["expires_at_epoch"])
        if (not all(math.isfinite(value) for value in (created, renewed, expires))
                or created < 0 or renewed < created or not 0 < expires - renewed <= MAX_TTL_SECONDS):
            return False
        released = lease.get("released_at_epoch")
        if released is not None and (not math.isfinite(float(released)) or float(released) < created):
            return False
    except (KeyError, TypeError, ValueError, OverflowError, LaneScratchError):
        return False
    return True


@contextmanager
def _locked_root_descriptor(root_fd: int) -> Iterator[int]:
    """Share the root's coordination lock with retained-descriptor consumers."""

    try:
        lock_fd = os.open(".lane-scratch.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                          0o600, dir_fd=root_fd)
    except OSError as exc:
        raise LaneScratchError("lane_scratch_lock_unsafe") from exc
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield root_fd
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


@contextmanager
def _opened_directory_path(path: Path, *, unsafe_code: str) -> Iterator[int]:
    """Retain a no-follow descriptor chain through every absolute component."""

    if not path.is_absolute() or ".." in path.parts:
        raise LaneScratchError(unsafe_code)
    with ExitStack() as descriptors:
        try:
            directory_fd = os.open(path.anchor, _DIR_FLAGS)
            descriptors.callback(os.close, directory_fd)
            for component in path.parts[1:]:
                directory_fd = os.open(component, _DIR_FLAGS, dir_fd=directory_fd)
                descriptors.callback(os.close, directory_fd)
        except OSError as exc:
            raise LaneScratchError(unsafe_code) from exc
        yield directory_fd


@contextmanager
def _locked_root(root: str | Path) -> Iterator[int]:
    path = Path(root)
    if not path.is_absolute():
        raise LaneScratchError("lane_scratch_root_not_absolute")
    with _opened_directory_path(path, unsafe_code="lane_scratch_root_unsafe") as root_fd:
        with _locked_root_descriptor(root_fd):
            yield root_fd


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


def _publish_no_replace(lane_fd: int, staging: str, name: str) -> None:
    """Atomically publish a prepared directory, refusing an existing target."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renameatx_np
        flag = 0x00000004  # RENAME_EXCL from sys/stdio.h
    elif sys.platform.startswith("linux"):
        function = libc.renameat2
        flag = 1  # RENAME_NOREPLACE from linux/fs.h
    else:
        raise LaneScratchError("lane_scratch_publish_unsupported")
    function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_uint)
    function.restype = ctypes.c_int
    if function(lane_fd, os.fsencode(staging), lane_fd, os.fsencode(name), flag) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), name)


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
            or lease["lease_digest"] != canonical_digest(lease, digest_field="lease_digest")
            or not _lease_fields_valid(lease)):
        raise LaneScratchError("lane_scratch_lease_invalid")
    return lease


def read_lane_scratch_folder(path: str | Path, *, lane: str, name: str) -> dict[str, Any]:
    """Read an exact folder's sealed lease without creating a lock or following its symlink."""

    lane, name = _id(lane, "lane"), _id(name, "name")
    with _opened_directory_path(Path(path), unsafe_code="lane_scratch_folder_unsafe") as folder_fd:
        lease = _read_lease(folder_fd)
    if lease.get("lane") != lane or lease.get("name") != name:
        raise LaneScratchError("lane_scratch_lease_mismatch")
    return lease


def _creation_lease(
    lane: str, name: str, *, owner: str, reason: str, class_intent: str,
    cleanup: str, ttl_seconds: int, run_ref: str | None = None,
    scene_ref: str | None = None, size_budget_bytes: int | None = None,
    now: Callable[[], float] = time.time,
    consumer_lifetime_contract: Any = _NO_PROTOCOL,
) -> dict[str, Any]:
    """Validate creation metadata before a constructor can mutate a directory."""

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
    if consumer_lifetime_contract is not _NO_PROTOCOL and consumer_lifetime_contract != CONSUMER_LIFETIME_PROTOCOL:
        raise LaneScratchError("lane_scratch_protocol_invalid")
    return _seal({
        "schema_version": SCHEMA_VERSION, "lane": lane, "name": name,
        "owner": owner, "reason": reason, "class_intent": class_intent,
        "cleanup": cleanup, "created_at_epoch": observed,
        "expires_at_epoch": observed + ttl, "released_at_epoch": None,
        "size_budget_bytes": size_budget_bytes,
        "run_ref" if run_ref is not None else "scene_ref": reference,
        **({"consumer_lifetime_contract": consumer_lifetime_contract}
           if consumer_lifetime_contract is not _NO_PROTOCOL else {}),
    })


def _publish_scratch_folder(
    root_fd: int, lease: Mapping[str, Any], *,
    verify_location: Callable[[int], None] | None = None,
) -> tuple[os.stat_result, os.stat_result, os.stat_result]:
    """Publish an already validated lease beneath a coordinated root descriptor."""

    lane, name = lease["lane"], lease["name"]
    with _lane_fd(root_fd, lane, create=True) as lane_fd:
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
                identity = (os.fstat(root_fd), os.fstat(lane_fd), os.fstat(stage_fd))
            finally:
                os.close(stage_fd)
            if verify_location is not None:
                verify_location(lane_fd)
            _publish_no_replace(lane_fd, staging, name)
            os.fsync(lane_fd)
            return identity
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


def create_lane_scratch(
    lane: str, name: str, *, owner: str, reason: str, class_intent: str,
    cleanup: str, ttl_seconds: int, run_ref: str | None = None,
    scene_ref: str | None = None, size_budget_bytes: int | None = None,
    root: str | Path = DEFAULT_ROOT, now: Callable[[], float] = time.time,
    consumer_lifetime_contract: Any = _NO_PROTOCOL,
    _registered_birth: Any = None,
) -> Path:
    """Create a new scratch folder with a sealed lease in one publication."""

    _refuse_registered_legacy_mutation(lane, name)
    lease = _creation_lease(lane, name, owner=owner, reason=reason, class_intent=class_intent,
                            cleanup=cleanup, ttl_seconds=ttl_seconds, run_ref=run_ref,
                            scene_ref=scene_ref, size_budget_bytes=size_budget_bytes, now=now,
                            consumer_lifetime_contract=consumer_lifetime_contract)
    if _registered_birth is not None:
        from .control_plane_lane_experiment_birth import _RegisteredBirth
        if type(_registered_birth) is not _RegisteredBirth:
            raise LaneScratchError("lane_scratch_registered_authority_required")
        return _registered_birth.publish_creation(lease)
    with _locked_root(root) as root_fd:
        _publish_scratch_folder(root_fd, lease)
    return Path(root) / lane / name


def _refuse_registered_legacy_mutation(lane, name):
    # A missing birth/authority record cannot opt this reserved name into legacy
    # creation, renewal or release. Fixed root issuance uses an unpublished stage.
    if lane == "g1" and isinstance(name, str) and re.fullmatch(r"registered-[0-9a-f]{32}", name):
        raise LaneScratchError("lane_scratch_registered_authority_required")


def _change_lease(
    *, root: str | Path, lane: str, name: str, owner: str, expected_digest: str,
    now: Callable[[], float], ttl_seconds: int | None,
) -> dict[str, Any]:
    _refuse_registered_legacy_mutation(lane, name)
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
           "renew_lane_scratch", "release_lane_scratch", "list_lane_scratch", "read_lane_scratch_folder"]
