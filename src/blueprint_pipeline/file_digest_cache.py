"""Reuse a retained file's sha256 while its stat fingerprint is unchanged, like git's index.

Adopting retained Atlas work re-verifies retained files against their recorded
digests on every tick. Inside ``file_digest_cache(sidecar)``, ``file_sha256``
returns the digest it last computed for a path only while the file's
(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns) are all unchanged, and hashes
the file again otherwise. Callers still compare the result with their own
retained digest, so no receipt, record or identity changes; the sidecar is a
plain cache, and deleting it only costs one re-hash of each file. Outside the
block nothing is cached.

``verify_once`` applies the same idea to a pure check of file contents (such as
decoding two images to compare their pixels): it runs once per job for exactly
the digests, code and library versions its binding names.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

SCHEMA = "file_digest_cache.v1"
SIDECAR = "file_digest_cache.json"
MAXIMUM_ENTRIES = 50_000
MAXIMUM_CHECKS = 4096
# Threat model: user space can set a file's mtime but never its ctime, which the
# kernel sets to the current time on every write, truncate, chmod, link change
# or utime (only root can move it, by setting the clock back), and a replaced
# file has a new inode; so a content change always changes the fingerprint and
# is re-hashed. A change within the timestamp granularity of the hash could
# keep the fingerprint ("racy git"), so a digest is kept only for a file whose
# ctime and mtime are this much older than the moment hashing began. The sidecar
# lives inside the job, in the same trust boundary as the files and receipts it
# describes: whoever can rewrite it can rewrite those too.
RACY_NANOSECONDS = 2_000_000_000

_active: ContextVar[_Cache | None] = ContextVar("file_digest_cache", default=None)


def _fingerprint(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")


def _valid(row: Any) -> bool:
    return (isinstance(row, list) and len(row) == 6 and all(type(value) is int for value in row[:5])
            and _is_digest(row[5]))


class _Cache:
    def __init__(self, sidecar: Path) -> None:
        self.sidecar, self.used, self.changed = sidecar, set(), False
        try:
            # Opened, not probed: a missing cache is not a retained record anyone looked for.
            with open(sidecar, "rb") as stream:
                value = json.loads(stream.read())
            entries, checks = ((value["entries"], value.get("checks", []))
                               if value.get("schema_version") == SCHEMA else ({}, []))
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            entries, checks = {}, []
        self.entries = ({key: row for key, row in entries.items() if isinstance(key, str) and _valid(row)}
                        if isinstance(entries, dict) else {})
        self.checks = dict.fromkeys(key for key in checks if _is_digest(key)) if isinstance(checks, list) else {}

    def remember(self, key: str, before: os.stat_result, after: list[int], started: int, digest: str) -> None:
        fingerprint = _fingerprint(before)
        if after == fingerprint and max(before.st_mtime_ns, before.st_ctime_ns) < started - RACY_NANOSECONDS:
            self.entries[key], self.changed = [*fingerprint, digest], True
        elif self.entries.pop(key, None) is not None:
            self.changed = True

    def save(self) -> None:
        # Unused entries stay while their file is unchanged, so a pass that stops
        # early keeps the rest; those for gone or changed files are dropped.
        for key in [key for key in self.entries if key not in self.used]:
            try:
                current = _fingerprint(os.stat(key))
            except OSError:
                current = None
            if current != self.entries[key][:5]:
                del self.entries[key]
                self.changed = True
        if len(self.entries) > MAXIMUM_ENTRIES:
            ordered = sorted(self.entries, key=lambda key: key not in self.used)
            self.entries = {key: self.entries[key] for key in ordered[:MAXIMUM_ENTRIES]}
            self.changed = True
        if len(self.checks) > MAXIMUM_CHECKS:
            self.checks = dict.fromkeys(list(self.checks)[-MAXIMUM_CHECKS:])  # the most recently passed
            self.changed = True
        if not self.changed:
            return
        payload = json.dumps({"schema_version": SCHEMA, "entries": self.entries, "checks": list(self.checks)},
                             sort_keys=True, separators=(",", ":")).encode()
        descriptor, temporary = tempfile.mkstemp(prefix="." + self.sidecar.name + ".", dir=self.sidecar.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, self.sidecar)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def file_sha256(path: Any) -> str:
    """``sha256:<hex>`` of a file's bytes; reused from the active cache only for an unchanged fingerprint."""
    name = os.fspath(path)
    cache = _active.get()
    if cache is not None:
        key = name if os.path.isabs(name) else os.path.join(os.getcwd(), name)
        cache.used.add(key)
        row = cache.entries.get(key)
        if row is not None and _fingerprint(os.stat(name)) == row[:5]:
            return row[5]
    started = time.time_ns()
    digest = hashlib.sha256()
    with open(name, "rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = _fingerprint(os.fstat(stream.fileno()))
    value = "sha256:" + digest.hexdigest()
    if cache is not None:
        cache.remember(key, before, after, started, value)
    return value


def verify_once(binding: Mapping[str, Any], check: Callable[[], None]) -> None:
    """Run ``check`` unless it already passed in this job for exactly ``binding``.

    For a pure check of file contents: ``binding`` names every input by a digest
    the caller has just verified (through ``file_sha256``) and the code and
    library versions that check them, so the same binding means the same answer.
    A check that raises is never remembered; outside a job step it always runs.
    """
    cache = _active.get()
    key = "sha256:" + hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"),
                                                allow_nan=False).encode()).hexdigest()
    if cache is not None and key in cache.checks:
        cache.checks[key] = cache.checks.pop(key)  # most recently used: kept longest under the cap
        return
    check()
    if cache is not None:
        cache.checks[key], cache.changed = None, True


@contextmanager
def file_digest_cache(sidecar: Path) -> Iterator[None]:
    """Reuse and update the digests in ``sidecar`` for the block; written atomically after it.

    Hold the job's own lock around it: the sidecar has one writer at a time. A
    sidecar that cannot be read or written is ignored, never an error.
    """
    if _active.get() is not None:
        yield  # Nested: the outer block already owns a cache.
        return
    cache = _Cache(Path(sidecar))
    token = _active.set(cache)
    try:
        yield
    finally:
        _active.reset(token)
        try:
            cache.save()
        except OSError:
            pass
