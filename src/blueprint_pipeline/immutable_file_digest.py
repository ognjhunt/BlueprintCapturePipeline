"""Reuse verified read-only file digests within one publication transaction."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import stat
from time import time_ns as _now_ns
from typing import BinaryIO, Iterator


_MAX_ENTRIES = 131072
_STABLE_AGE_NS = 2_000_000_000
_cache: ContextVar[OrderedDict[tuple[int, ...], str] | None] = ContextVar(
    "immutable_file_digest_cache", default=None
)
# Inodes an earlier release's service account already read in full, keyed by
# ``sealed_identity``. Only ``readback_ledger`` admits entries.
_proven: ContextVar[dict[tuple[int, ...], str] | None] = ContextVar(
    "immutable_file_proven_digests", default=None
)


@contextmanager
def immutable_file_digest_cache() -> Iterator[None]:
    """Discard the bounded cache when this publication (including nested calls) ends."""
    if _cache.get() is not None:
        yield
        return
    token, proven = _cache.set(OrderedDict()), _proven.set({})
    try:
        yield
    finally:
        _proven.reset(proven)
        _cache.reset(token)


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_size, value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)


def sealed_identity(value: os.stat_result) -> tuple[int, ...]:
    """What a hardlink or unlink leaves unchanged: device, inode, size, mtime, mode, owner.

    Every publication links each runtime inode into several trees and unlinks
    its staging aliases, and each of those moves ``st_nlink`` and ``st_ctime``.
    """
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_mode, value.st_uid, value.st_gid)


def _sealed_by_this_process(value: os.stat_result) -> bool:
    return (os.name == "posix" and stat.S_ISREG(value.st_mode)
            and value.st_uid == os.geteuid() and value.st_ino != 0
            and not value.st_mode & 0o222)


def admit_proven_digest(value: os.stat_result, digest: str) -> bool:
    """Trust ``digest`` for this sealed inode for the rest of the publication.

    Only for an inode a service account already read every byte of, in an
    earlier release, under exactly this ``sealed_identity``. False (and nothing
    admitted) outside a publication or for an inode this process does not own
    read-only.
    """
    proven = _proven.get()
    if proven is None or not _sealed_by_this_process(value):
        return False
    proven[sealed_identity(value)] = digest
    return True


def proven_digest(value: os.stat_result) -> str | None:
    """The admitted digest of this inode, if an earlier readback proved it."""
    proven = _proven.get()
    return None if proven is None else proven.get(sealed_identity(value))


def _read_digest(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _unchanged(path: Path, stream: BinaryIO, identity: tuple[int, ...]) -> None:
    if (_identity(os.fstat(stream.fileno())) != identity
            or _identity(path.lstat()) != identity):
        raise OSError("immutable_file_changed_during_digest")


def sha256_file(path: Path) -> str:
    """Hash normally unless the same read-only inode is verified in this transaction.

    Cache hits still open the actual path and check descriptor/path identities.
    Only transaction-owned POSIX files older than the timestamp safety window
    qualify: a recent write may share a coarse filesystem timestamp with another
    write. Mutable files, unowned files and symlink leaves are read in full.
    An inode an earlier release's service-account readback proved
    (``admit_proven_digest``) is not read again while its sealed identity holds,
    however many links this publication adds or removes.
    """
    cache = _cache.get()
    with path.open("rb") as stream:
        if cache is None:
            return _read_digest(stream)
        before = os.fstat(stream.fileno())
        identity = _identity(before)
        if not _sealed_by_this_process(before) or _identity(path.lstat()) != identity:
            return _read_digest(stream)
        proven = proven_digest(before)
        if proven is not None:
            return proven
        if _now_ns() - before.st_ctime_ns < _STABLE_AGE_NS:
            return _read_digest(stream)
        cached = cache.get(identity)
        if cached is not None:
            _unchanged(path, stream, identity)
            cache.move_to_end(identity)
            return cached
        digest = _read_digest(stream)
        _unchanged(path, stream, identity)
        cache[identity] = digest
        if len(cache) > _MAX_ENTRIES:
            cache.popitem(last=False)
        return digest
