"""Which launch requests in ``processing/`` a live dispatcher can still own.

``process_launch_queue`` claims a whole batch into ``processing/`` and then
dispatches it serially, so a claimed request can wait hours for its turn with
no run directory. Every dispatcher therefore holds a shared lock on one file in
the queue root for its whole call, claims and dispatch included; dispatchers
started side by side do not wait for each other.

The reconciler asks only with a non-blocking exclusive lock, which it gets only
when no dispatcher is alive, and it releases that lock as soon as it has listed
``processing/``. A dispatcher reserves a processing name with ``O_EXCL``, so no
dispatcher can later claim a name that already exists: every name listed under
the exclusive lock belongs to no live dispatcher. The reconciler never creates
the lock file, so until a dispatcher that takes this lock has run (after a
deploy, an older dispatcher may still be finishing), it treats every request as
owned.
"""

from __future__ import annotations

import fcntl
import functools
import inspect
import os
from pathlib import Path

DISPATCHER_LOCK_NAME = ".dispatcher.lock"


def holds_dispatcher_lock(queue_argument: str):
    """Hold the shared dispatcher lock of the call's queue root for the whole call."""

    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def held(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            queue = Path(bound.arguments[queue_argument]).expanduser().resolve()
            queue.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(queue / DISPATCHER_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                # Fail before claiming anything if ownership cannot be shown.
                fcntl.flock(descriptor, fcntl.LOCK_SH)
                return function(*args, **kwargs)
            finally:
                os.close(descriptor)

        return held

    return decorate


def unowned_processing_names(queue_root: str | Path) -> frozenset[str]:
    """Names in ``processing/`` that no live dispatcher owns; none while one is alive."""

    queue = Path(queue_root).expanduser().resolve()
    try:
        descriptor = os.open(queue / DISPATCHER_LOCK_NAME, os.O_RDONLY)
    except OSError:
        return frozenset()
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return frozenset()
        return frozenset(path.name for path in (queue / "processing").glob("*.json"))
    finally:
        os.close(descriptor)
