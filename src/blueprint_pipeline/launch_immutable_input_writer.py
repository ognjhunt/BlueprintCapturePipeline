"""Private, byte-exact input files for a bound launch profile."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class TaskEvaluationLaunchError(ValueError):
    """Raised when a launch request or profile fails closed."""


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
