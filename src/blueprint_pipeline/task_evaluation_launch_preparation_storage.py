"""Existing checkpointed exclusive queue writer without request validation imports."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .control_plane_registered_reference_gate import _publisher_checkpoint, _publisher_observation
from .task_evaluation_release_reference_lock import release_reference_lock


class TaskEvaluationLaunchPreparationQueueError(ValueError):
    """The immutable preparation request could not be staged safely."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _write_launch_preparation_record_exclusive_locked(
    path: Path, value: Mapping[str, Any]
) -> None:
    payload = _canonical_bytes(value)
    path_token = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    temporary_path = path.with_name(
        f".queue-{path_token}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = -1
    try:
        _publisher_checkpoint()
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o440,
        )
        view = memoryview(payload)
        while view:
            _publisher_checkpoint()
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable preparation queue write")
            view = view[written:]
        _publisher_checkpoint()
        os.fsync(descriptor)
        _publisher_checkpoint()
        os.fchmod(descriptor, 0o440)
        os.close(descriptor)
        descriptor = -1
        _publisher_checkpoint()
        os.link(temporary_path, path, follow_symlinks=False)
        temporary_path.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            _publisher_checkpoint()
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        raise
    except OSError as exc:
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_write_failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


@_publisher_observation
def write_launch_preparation_record_exclusive(
    path: Path, value: Mapping[str, Any]
) -> None:
    from .control_plane_registered_reference_gate import refuse_registered_references
    refuse_registered_references(value, path)
    with release_reference_lock(path.parents[2], exclusive=False):
        _write_launch_preparation_record_exclusive_locked(path, value)

