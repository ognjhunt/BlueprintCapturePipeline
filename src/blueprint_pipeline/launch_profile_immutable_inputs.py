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
    ownership, mode and link count, and the hashing code is unchanged.

    A small file is always re-hashed, and without the store: below ``MINIMUM_BYTES``
    the store re-hashes on every lookup anyway, so an entry only added a read, a
    parse and a code proof to the same hash, once per input a profile lists.
    """
    from .task_evaluation_release_identity import running_release_commit
    from .validation_file_digests import MINIMUM_BYTES, sha256_file, touched_files
    from .validation_verdict_store import executed_code_identity, lookup_entry, store

    try:
        small = path.stat().st_size < MINIMUM_BYTES
    except OSError:
        small = False
    if small:
        try:
            return sha256_file(path)
        except ValueError:  # a symlinked parent: hash the bytes the historical way
            return _plain_sha256(path)
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


def observed_input_digest(raw_path: Path, observed: dict[str, str | None] | None = None) -> str | None:
    """Digest of a declared input path, or ``None`` when it is not a regular, non-symlink file.

    ``observed`` shares one look per path across a pass over many profiles: on
    2026-10-12 the published profiles listed 26,015 inputs that are 3,784 distinct
    files, and the catalog reconciler digested a file once per profile naming it.
    """
    key = str(raw_path)
    if observed is not None and key in observed:
        return observed[key]
    if raw_path.is_symlink() or not raw_path.is_file():
        digest = None
    else:
        digest = immutable_input_digest(raw_path.resolve())
    if observed is not None:
        observed[key] = digest
    return digest
