"""Private, byte-exact input files for a bound launch profile."""

from __future__ import annotations

import os
import hashlib
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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
            payload = staged_sources[source].read_bytes()
            write_exclusive_private_bytes(projected, payload)
            projected.chmod(0o600)
            readback = projected.read_bytes()
            digest = digest_prefix + hashlib.sha256(readback).hexdigest()
            expected_digest = digest_prefix + hashlib.sha256(payload).hexdigest()
            if readback != payload or digest != expected_digest:
                raise TaskEvaluationLaunchError("immutable_input_directory_projection_readback_mismatch")
            projected_inputs.append({
                "source_path": str(source), "relative_path": str(relative_path),
                "projected_path": str(projected), "digest": digest,
                "size_bytes": len(readback),
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
