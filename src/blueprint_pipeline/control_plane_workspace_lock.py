"""Coordinate semantic workspace consumers and bundle reclamation."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path


@contextmanager
def workspace_lock(workspace, *, reclaim=False):
    workspace = Path(workspace)
    locks = workspace.parent / '.workspace-locks'
    path = locks / (workspace.name + '.lock')
    if reclaim and not path.is_file():
        yield False  # Legacy workspaces have no cooperating producer proof.
        return
    parent = workspace.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        locks.mkdir(mode=0o770)
    except FileExistsError:
        if locks.is_symlink() or not locks.is_dir():
            raise ValueError("workspace_lock_directory_unsafe")
    else:
        if os.geteuid() == 0:
            owner = parent.stat()
            os.chown(locks, owner.st_uid, owner.st_gid)
        locks.chmod(0o770)
    path = locks / (workspace.name + '.lock')
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
    except FileExistsError:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    else:
        if os.geteuid() == 0:
            owner = parent.stat()
            os.fchown(fd, owner.st_uid, owner.st_gid)
        os.fchmod(fd, 0o660)
    try:
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX | fcntl.LOCK_NB) if reclaim else fcntl.LOCK_SH)
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        os.close(fd)
