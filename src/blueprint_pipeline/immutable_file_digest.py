"""Reuse verified read-only file digests within one publication transaction."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import stat
from typing import BinaryIO, Iterator


_MAX_ENTRIES = 131072
_cache: ContextVar[OrderedDict[tuple[int, ...], str] | None] = ContextVar(
    "immutable_file_digest_cache", default=None
)


@contextmanager
def immutable_file_digest_cache() -> Iterator[None]:
    """Discard the bounded cache when this publication (including nested calls) ends."""
    if _cache.get() is not None:
        yield
        return
    token = _cache.set(OrderedDict())
    try:
        yield
    finally:
        _cache.reset(token)


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_size, value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)


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
    Write, chmod, replacement and link changes invalidate the entry. Mutable
    files and symlink leaves are always read in full. Service-account readback
    is independent of this digest helper and still reads every installed byte.
    """
    cache = _cache.get()
    with path.open("rb") as stream:
        if cache is None:
            return _read_digest(stream)
        before = os.fstat(stream.fileno())
        identity = _identity(before)
        eligible = (stat.S_ISREG(before.st_mode) and not before.st_mode & 0o222
                    and _identity(path.lstat()) == identity)
        if not eligible:
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
