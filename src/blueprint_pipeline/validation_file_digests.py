"""Transaction-local hashing reuse for retained large inputs, never validation verdicts.

Each outer operation starts empty. Every first read hashes all bytes. Only
regular files of at least MINIMUM_BYTES qualify; every hit reopens the file
without following symlinks and checks device, inode, link count, ownership,
mode, size and nanosecond modification and change times. The change time is
kernel-owned and cannot be restored from userland, so a write bit on a retained
artifact is not a change witness and must not disqualify it: producers retain
PLY, bundle and video artifacts as 0600/0644, and refusing to reuse their
hashes made one controller restart re-hash the same bytes once per reuse
candidate per phase (about 70 full passes, 308 GB, 55 minutes for a 4.6 GB
tree). Authority documents stay below MINIMUM_BYTES and are never cached.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import time

_CACHE: ContextVar[dict | None] = ContextVar("validation_file_digests", default=None)
_STATS: ContextVar[dict | None] = ContextVar("validation_file_digest_stats", default=None)
MINIMUM_BYTES = 1024 * 1024


def scoped_measurement(key, compute):
    """Reuse a pure measurement keyed by current input digests and parameters.

    Callers must reopen all input bytes before constructing the key. No disk
    cache survives code changes, dependency changes or the outer operation.
    Values are copied so caller mutation cannot alter subsequent validation.
    """
    cache = _CACHE.get()
    if cache is None:
        return compute()
    key = ("measurement", key)
    if key not in cache:
        cache[key] = deepcopy(compute())
    return deepcopy(cache[key])


@contextmanager
def file_digest_scope():
    """Share hashes across nested validators only for this synchronous operation."""
    if _CACHE.get() is not None:
        yield
        return
    token = _CACHE.set({})
    stats_token = _STATS.set({"files_hashed": 0, "bytes_hashed": 0, "hash_seconds": 0.0,
                              "cache_hits": 0, "bytes_reused": 0, "last_path": None})
    try:
        yield
    finally:
        _STATS.reset(stats_token)
        _CACHE.reset(token)


def digest_scope_stats() -> dict | None:
    """Operational counters for the active scope: bytes hashed versus reused.

    Telemetry for progress heartbeats only; never a validation verdict and
    never persisted as evidence. ``None`` outside any scope.
    """
    stats = _STATS.get()
    return None if stats is None else dict(stats)


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode, value.st_uid, value.st_gid,
            value.st_nlink)


def sha256_file(path: Path) -> str:
    """Hash a regular file; reuse only bytes whose full stat identity is unchanged in scope."""
    path = Path(path)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("validation_file_digest_symlink_forbidden")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("validation_file_digest_regular_file_required")
        identity = _identity(before)
        cache = _CACHE.get()
        eligible = cache is not None and before.st_size >= MINIMUM_BYTES
        digest = cache.get(identity) if eligible else None
        stats = _STATS.get()
        if digest is None:
            started = time.monotonic()
            value = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
            digest = "sha256:" + value.hexdigest()
            if stats is not None:
                stats["files_hashed"] += 1
                stats["bytes_hashed"] += before.st_size
                stats["hash_seconds"] += time.monotonic() - started
        elif stats is not None:
            stats["cache_hits"] += 1
            stats["bytes_reused"] += before.st_size
        if stats is not None:
            stats["last_path"] = str(path)
        if (identity != _identity(os.fstat(stream.fileno()))
                or identity != _identity(path.lstat())
                or any(item.is_symlink() for item in path.parents)):
            raise ValueError("validation_file_digest_changed_during_read")
        if eligible:
            cache[identity] = digest
        return digest
