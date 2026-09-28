"""Digest of a launch-profile immutable input, reused while its exact stat identity holds."""
from __future__ import annotations

import hashlib
from pathlib import Path

IMMUTABLE_INPUT_DIGEST_VERDICT = "launch_profile_immutable_input_sha256"


def _plain_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def immutable_input_digest(path: Path) -> str:
    """Digest of one profile-bound immutable input, reused while its exact stat identity holds.

    A scene-configuration profile binds a 1.2 GB provider bundle. Publication and
    dispatch re-read it in full at every step (2026-09-13: about four minutes of
    the eleven-minute activation worker), although the bytes never move once the
    bundle step sealed them. The stored digest is returned only while the file
    keeps its device, inode, size, nanosecond mtime and kernel-owned ctime,
    ownership, mode and link count, and the hashing code is unchanged; a small
    file is always re-hashed.
    """
    from .task_evaluation_release_identity import running_release_commit
    from .validation_file_digests import sha256_file, touched_files
    from .validation_verdict_store import executed_code_identity, lookup_entry, store

    key = {"path": str(path)}
    found, stored = lookup_entry(name=IMMUTABLE_INPUT_DIGEST_VERDICT, key=key)
    if found and isinstance(stored, str):
        return stored
    try:
        with touched_files() as touched:
            digest, code = executed_code_identity(lambda: sha256_file(path), always=[__name__])
    except ValueError:  # a symlinked parent: hash the bytes the historical way, without persistence
        return _plain_sha256(path)
    try:
        release = running_release_commit()
    except (OSError, ValueError):
        release = ""
    store(name=IMMUTABLE_INPUT_DIGEST_VERDICT, key=key, files=touched, verdict=digest, source_commit=release, code=code)
    return digest
