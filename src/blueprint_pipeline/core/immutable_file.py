"""Publish complete immutable bytes without replacing an existing artifact."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_file_exclusive(path: Path, payload: bytes) -> None:
    """Retain bytes before publishing the name; callers own replay validation.

    FileExistsError preserves the existing exclusive-writer contract. A failed
    write leaves the final name absent, or a complete file after publication,
    so process restart never sees a partially written immutable artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
