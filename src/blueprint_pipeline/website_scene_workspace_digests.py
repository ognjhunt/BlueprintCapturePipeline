"""Bounded digest inventory and cache for website scene workspace retirement."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class HashBudget:
    """Uncached bytes the current tick may still hash, shared by every scene it plans."""

    remaining_bytes: int
    hashed_bytes: int = 0
    deadline_monotonic: float = field(default_factory=lambda: time.monotonic() + 300)
    oversized_file_threshold: int = 20 * 1024**3

    def allows(self, size: int) -> bool:
        if time.monotonic() >= self.deadline_monotonic:
            return False
        # One file larger than the default budget may be hashed once per tick;
        # otherwise it would remain deferred forever. The unit runtime bounds it.
        return size <= self.remaining_bytes or (self.hashed_bytes == 0 and size > self.oversized_file_threshold)

    def spend(self, size: int) -> None:
        self.remaining_bytes -= size
        self.hashed_bytes += size


@dataclass(frozen=True)
class Digests:
    size: int
    sha256: str  # hex
    md5: str  # base64
    crc32c: str | None  # base64; None when google_crc32c is unavailable


def hash_file(path: Path) -> Digests:
    """SHA-256, MD5 and CRC32C of a regular file in one read."""

    try:
        import google_crc32c

        crc = google_crc32c.Checksum()
    except ImportError:  # pragma: no cover - google-cloud-storage depends on it
        crc = None
    sha, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError("not a regular file")
        while chunk := stream.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)
            if crc is not None:
                crc.update(chunk)
            size += len(chunk)
    encode = lambda value: base64.b64encode(value).decode("ascii")  # noqa: E731 - local digest encoder
    return Digests(size=size, sha256=sha.hexdigest(), md5=encode(md5.digest()),
                   crc32c=encode(crc.digest()) if crc is not None else None)


def cached_digests(entry: Any, identity: list[int]) -> Digests | None:
    try:
        if entry["identity"] != identity:
            return None
        digests = entry["digests"]
        return Digests(size=int(digests["size"]), sha256=str(digests["sha256"]), md5=str(digests["md5"]),
                       crc32c=digests["crc32c"] if isinstance(digests["crc32c"], str) else None)
    except (KeyError, TypeError, ValueError):
        return None


def save_inventory_cache(path: Path, *, bucket: str, scene_id: str, files: Mapping[str, Any],
                         schema: str) -> None:
    """Replace the scene's digest cache atomically; readable by its owner alone."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    document = {"schema_version": schema, "bucket": bucket, "scene_id": scene_id, "files": dict(files)}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(document, stream, sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
