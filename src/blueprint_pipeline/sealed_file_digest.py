"""Reuse one full sha256 of a large file across processes, by its inode identity.

Every step of a scene-configuration activation (paid_authority,
allocator_dry_run, live_profile, the live allocator) reopens the same 6.1 GB
provider bundle and hashed all of it: about 95 s per step on the loaded
control-plane host (2026-10-31).

After hashing a file in full, a reader records a seal beside it: the digest
plus the identity it was read under (device, inode, size, mtime_ns, ctime_ns,
owner, group and mode). A later reader trusts the digest only when the open
descriptor and the path both still have exactly that identity. A path alone is
never trusted. Any write, truncate, rename-over, chmod, chown or utime changes
the identity: ctime is set by the kernel on every such change, and userspace
cannot set it. A file changed less than ``STABLE_AGE_NS`` before the hash began
is never sealed, because a write in the same timestamp tick would not move
ctime.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
from time import time_ns
from typing import BinaryIO

from .decision_evidence_contracts import canonical_digest, canonical_json

SCHEMA_VERSION = "sealed_file_digest.v1"
STABLE_AGE_NS = 2_000_000_000
_CHUNK = 1024 * 1024


def seal_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256-seal.v1.json")


def _identity(value: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
        "st_size": value.st_size,
        "st_mtime_ns": value.st_mtime_ns,
        "st_ctime_ns": value.st_ctime_ns,
        "st_uid": value.st_uid,
        "st_gid": value.st_gid,
        "st_mode": value.st_mode,
    }


def _read_seal(path: Path, *, identity: dict[str, int]) -> str | None:
    try:
        value = json.loads(seal_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("seal_digest") != canonical_digest(value, digest_field="seal_digest")
        or value.get("path") != str(path)
        or value.get("identity") != identity
        or not str(value.get("sha256") or "").startswith("sha256:")
    ):
        return None
    return str(value["sha256"])


def _write_seal(path: Path, *, identity: dict[str, int], digest: str) -> bool:
    value = {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "identity": identity,
        "sha256": digest,
        "seal_digest": "",
    }
    value["seal_digest"] = canonical_digest(value, digest_field="seal_digest")
    target = seal_path(path)
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(6)}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
        temporary.chmod(0o644)
        os.replace(temporary, target)
        return True
    except OSError:
        temporary.unlink(missing_ok=True)
        return False


def _hash_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(_CHUNK), b""):
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sealed_sha256(path: str | Path) -> str:
    """The file's sha256, from a matching seal or from a full read that seals it."""

    resolved = Path(path).resolve()
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        identity = _identity(before)
        if not stat.S_ISREG(before.st_mode) or _identity(resolved.lstat()) != identity:
            return _hash_stream(stream)
        sealed = _read_seal(resolved, identity=identity)
        # The identity is checked again on the descriptor and the path, so a
        # seal never vouches for bytes it did not see.
        if sealed is not None and (
            _identity(os.fstat(stream.fileno())) == identity == _identity(resolved.lstat())
        ):
            return sealed
        started = time_ns()
        digest = _hash_stream(stream)
        unchanged = (
            _identity(os.fstat(stream.fileno())) == identity == _identity(resolved.lstat())
        )
        if unchanged and started - before.st_ctime_ns >= STABLE_AGE_NS:
            _write_seal(resolved, identity=identity, digest=digest)
        return digest


__all__ = ["SCHEMA_VERSION", "STABLE_AGE_NS", "seal_path", "sealed_sha256"]
