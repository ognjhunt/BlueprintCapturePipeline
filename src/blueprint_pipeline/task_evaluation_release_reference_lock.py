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
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ReleaseReferenceLockError(ValueError):
    """The release-reference coordination boundary was not trustworthy."""


@contextmanager
def release_reference_lock(
    state_root: str | Path,
    *,
    exclusive: bool,
    timeout_seconds: float | None = None,
    poll_seconds: float = 0.5,
) -> Iterator[None]:
    """Lock one exact, existing, non-symlink state-root directory inode.

    Without ``timeout_seconds`` the call waits as long as it takes, which is
    what publishers want.  With it, the lock is polled without blocking until
    the deadline and then refused with ``release_reference_lock_busy``, so a
    holder that never lets go cannot hang the caller.
    """

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
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        if timeout_seconds is None:
            fcntl.flock(descriptor, operation)
        else:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))
            while True:
                try:
                    fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ReleaseReferenceLockError("release_reference_lock_busy") from None
                    time.sleep(min(max(poll_seconds, 0.001), remaining))
    except ReleaseReferenceLockError:
        os.close(descriptor)
        raise
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
