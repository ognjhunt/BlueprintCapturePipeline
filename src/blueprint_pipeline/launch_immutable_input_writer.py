"""Private, byte-exact input files for a bound launch profile."""

from __future__ import annotations

import filecmp
import os
import hashlib
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

#: Bytes read or written at once. A scene-configuration profile binds a 1.2 GB provider bundle; holding
#: it and its readback in memory took the dispatcher's python to 2.4 GB on an 8.3 GB host with no swap.
COPY_BLOCK_BYTES = 1 << 20


class TaskEvaluationLaunchError(ValueError):
    """Raised when a launch request or profile fails closed."""


def stage_directory_projections(
    stage_root: Path,
    projections: Mapping[Path, set[Path]],
    staged_sources: Mapping[Path, Path],
    *,
    digest_prefix: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Copy each bound source into the directory view handed to the allocator."""

    replacements: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for source_directory, included in projections.items():
        directory_key = hashlib.sha256(str(source_directory).encode("utf-8")).hexdigest()
        projection = stage_root / "directories" / directory_key
        projection.mkdir(mode=0o700, parents=True, exist_ok=True)
        projection.chmod(0o700)
        projected_inputs: list[dict[str, Any]] = []
        for source in sorted(included, key=str):
            relative_path = source.relative_to(source_directory)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise TaskEvaluationLaunchError("immutable_input_directory_projection_path_escape")
            projected = projection / relative_path
            _, written, written_size = copy_exclusive_private_file(projected, staged_sources[source])
            projected.chmod(0o600)
            readback, readback_size = file_sha256_and_size(projected)
            digest = digest_prefix + readback
            expected_digest = digest_prefix + written
            if readback_size != written_size or digest != expected_digest:
                raise TaskEvaluationLaunchError("immutable_input_directory_projection_readback_mismatch")
            projected_inputs.append({
                "source_path": str(source), "relative_path": str(relative_path),
                "projected_path": str(projected), "digest": digest,
                "size_bytes": readback_size,
            })
        replacements[str(source_directory)] = str(projection)
        rows.append({
            "source_directory": str(source_directory), "staged_directory": str(projection),
            "inputs": projected_inputs, "allocator_argv_indices": [],
        })
    return replacements, rows


def write_exclusive_private_bytes(path: Path, payload: bytes) -> bool:
    """Create one private file, allowing only byte-identical concurrent creation."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise TaskEvaluationLaunchError(f"immutable_input_staging_conflict:{path.name}")
            return False
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256_and_size(path: Path) -> tuple[str, int]:
    """The SHA-256 hex digest and byte count of a file, read in ``COPY_BLOCK_BYTES`` blocks."""

    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(COPY_BLOCK_BYTES), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def copy_exclusive_private_file(
    path: Path, source: Path, *, admit: Callable[[str, int], None] | None = None
) -> tuple[bool, str, int]:
    """``write_exclusive_private_bytes`` for a file's bytes, copied in blocks instead of held in memory.

    Returns whether this call created ``path``, and the SHA-256 hex digest and byte count of the bytes it
    wrote, which are the bytes ``path`` now holds. An existing ``path`` must hold exactly those bytes.
    ``admit(digest, size)``, when given, sees the written bytes' identity before ``path`` exists and may
    raise to refuse them: nothing is then left behind.
    """

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as stream, source.open("rb") as input_stream:
            os.fchmod(stream.fileno(), 0o600)
            for block in iter(lambda: input_stream.read(COPY_BLOCK_BYTES), b""):
                digest.update(block)
                size += len(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if admit is not None:
            admit(digest.hexdigest(), size)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if (path.is_symlink() or not path.is_file()
                    or not filecmp.cmp(path, temporary, shallow=False)):
                raise TaskEvaluationLaunchError(f"immutable_input_staging_conflict:{path.name}")
            return False, digest.hexdigest(), size
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return True, digest.hexdigest(), size
    finally:
        temporary.unlink(missing_ok=True)


def copy_and_read_back(path: Path, source: Path, expected_digest: str, refusal: str) -> tuple[str, int]:
    """Stage ``source`` at ``path`` and read it back, both in blocks; returns the digest and size read back.

    What a whole-bytes snapshot and its readback recorded, without holding either: a scene-configuration
    profile binds a 1.2 GB provider bundle, and the two took the launch dispatcher's python to 2.4 GB. The
    bytes written must have ``expected_digest`` before ``path`` exists (a source that changed after its
    digest was checked leaves nothing behind), and ``path`` must read back as them; else ``refusal``.
    """

    def admit(written: str, _size: int) -> None:
        if "sha256:" + written != expected_digest:
            raise TaskEvaluationLaunchError(refusal)

    _, written, written_size = copy_exclusive_private_file(path, source, admit=admit)
    readback, size = file_sha256_and_size(path)
    if size != written_size or readback != written:
        raise TaskEvaluationLaunchError(refusal)
    return "sha256:" + readback, size


def write_immutable_launch_record(path: Path, payload: bytes) -> bool:
    """Publish the existing launch record under its shared release lock."""
    from .control_plane_registered_reference_gate import _publisher_checkpoint
    from .task_evaluation_release_reference_lock import release_reference_lock

    with release_reference_lock(path.parents[2], exclusive=False):
        _publisher_checkpoint()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _publisher_checkpoint()
            with path.open("xb") as stream:
                _publisher_checkpoint()
                stream.write(payload)
            return True
        except FileExistsError:
            if path.read_bytes() != payload:
                raise TaskEvaluationLaunchError(f"immutable_launch_conflict:{path.name}")
            return False
