"""Cross-process coordination for Task Evaluation release references.

The common control-plane state directory is durable and is never atomically
replaced, so its inode is the lock authority.  Reference publishers take a
shared lock while making a new protected binding reachable; the release reaper
takes the exclusive lock across its final rescan and every deletion.
"""

from __future__ import annotations

import fcntl
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ReleaseReferenceLockError(ValueError):
    """The release-reference coordination boundary was not trustworthy."""


@contextmanager
def release_reference_lock(
    state_root: str | Path, *, exclusive: bool
) -> Iterator[None]:
    """Lock one exact, existing, non-symlink state-root directory inode."""

    root = Path(state_root).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise ReleaseReferenceLockError("release_reference_lock_root_invalid")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise ReleaseReferenceLockError(
            "release_reference_lock_root_unavailable"
        ) from exc
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        os.close(descriptor)
        raise ReleaseReferenceLockError("release_reference_lock_root_invalid")
    try:
        fcntl.flock(
            descriptor,
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
    except OSError as exc:
        os.close(descriptor)
        raise ReleaseReferenceLockError("release_reference_lock_failed") from exc
    try:
        yield
    finally:
        os.close(descriptor)


__all__ = [
    "ReleaseReferenceLockError",
    "release_reference_lock",
]
