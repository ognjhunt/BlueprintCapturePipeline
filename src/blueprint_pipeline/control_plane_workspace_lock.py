"""Coordinate semantic workspace consumers and bundle reclamation."""
from contextlib import contextmanager
import fcntl
import grp
import os
from pathlib import Path
import pwd
import stat


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_LOCK_FLAGS = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def _identity(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_uid, metadata.st_gid,
            metadata.st_mode)


def _verify_directories(bindings):
    """Verify both the held inode and its name through the held parent."""
    for parent, name, descriptor, expected in bindings:
        held = os.fstat(descriptor)
        if _identity(held) != expected:
            raise ValueError("workspace_lock_directory_retargeted")
        if parent is not None:
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if _identity(named) != expected:
                raise ValueError("workspace_lock_directory_retargeted")


def _shared_owner(uid, parent):
    # The existing coordination policy admits root and the workspace's runtime
    # group. Do not assume a numeric service UID or change installed ownership.
    if uid in {0, parent.st_uid, os.geteuid()}:
        return True
    try:
        owner = pwd.getpwuid(uid)
        return (owner.pw_gid == parent.st_gid
                or owner.pw_name in grp.getgrgid(parent.st_gid).gr_mem)
    except KeyError:
        return False


def _verify_shared(metadata, parent, *, directory):
    mode = stat.S_IMODE(metadata.st_mode)
    allowed = 0o2770 if directory else 0o660
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (not expected_type(metadata.st_mode) or mode & ~allowed
            or metadata.st_gid != parent.st_gid
            or not _shared_owner(metadata.st_uid, parent)
            or (not directory and metadata.st_nlink != 1)):
        raise ValueError("workspace_lock_object_unsafe")


def _verify_lock(directory, name, descriptor, owner):
    held = os.fstat(descriptor)
    _verify_shared(held, owner, directory=False)
    named = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if (_identity(named) != _identity(held)
            or held.st_nlink != 1 or named.st_nlink != 1):
        raise ValueError("workspace_lock_file_retargeted")


@contextmanager
def workspace_lock(workspace, *, reclaim=False):
    workspace = Path(workspace)
    if not workspace.name or '..' in workspace.parts:
        raise ValueError("workspace_lock_path_unsafe")
    parent_path = Path(os.path.abspath(workspace.parent))
    name = workspace.name + '.lock'
    bindings = []
    fd = None
    locks = None
    created = False
    try:
        root = os.open(parent_path.anchor, _DIRECTORY_FLAGS)
        bindings.append((None, None, root, _identity(os.fstat(root))))
        parent = root
        for component in parent_path.parts[1:]:
            _verify_directories(bindings)
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                if reclaim:
                    yield False  # No producer proof; do not create one.
                    return
                try:
                    os.mkdir(component, dir_fd=parent)
                except FileExistsError:
                    pass
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent)
            metadata = os.fstat(child)
            bindings.append((parent, component, child, _identity(metadata)))
            # Sticky temporary roots are valid for hermetic/local workspaces;
            # an unrestricted, non-sticky writable ancestor is not.
            if metadata.st_mode & stat.S_IWOTH and not metadata.st_mode & stat.S_ISVTX:
                raise ValueError("workspace_lock_ancestor_unsafe")
            parent = child

        owner = os.fstat(parent)
        if any(not _shared_owner(os.fstat(descriptor).st_uid, owner)
               for _, _, descriptor, _ in bindings):
            raise ValueError("workspace_lock_ancestor_owner_unsafe")
        _verify_directories(bindings)
        made_directory = False
        try:
            locks = os.open('.workspace-locks', _DIRECTORY_FLAGS, dir_fd=parent)
        except FileNotFoundError:
            if reclaim:
                yield False
                return
            try:
                os.mkdir('.workspace-locks', 0o770, dir_fd=parent)
                made_directory = True
            except FileExistsError:
                pass
            locks = os.open('.workspace-locks', _DIRECTORY_FLAGS, dir_fd=parent)
        bindings.append((parent, '.workspace-locks', locks, _identity(os.fstat(locks))))
        _verify_directories(bindings)
        if made_directory:
            if os.fstat(locks).st_uid != os.geteuid():
                raise ValueError("workspace_lock_directory_owner_unsafe")
            if os.geteuid() == 0:
                os.fchown(locks, owner.st_uid, owner.st_gid)
            elif os.fstat(locks).st_gid != owner.st_gid:
                os.fchown(locks, -1, owner.st_gid)
            os.fchmod(locks, 0o770)
            bindings[-1] = (parent, '.workspace-locks', locks, _identity(os.fstat(locks)))
        _verify_shared(os.fstat(locks), owner, directory=True)
        _verify_directories(bindings)

        if reclaim:
            try:
                fd = os.open(name, _LOCK_FLAGS, dir_fd=locks)
            except FileNotFoundError:
                yield False
                return
        else:
            try:
                fd = os.open(name, _LOCK_FLAGS | os.O_CREAT | os.O_EXCL, 0o660, dir_fd=locks)
                created = True
            except FileExistsError:
                fd = os.open(name, _LOCK_FLAGS, dir_fd=locks)
        _verify_directories(bindings)
        if created:
            if os.geteuid() == 0:
                os.fchown(fd, owner.st_uid, owner.st_gid)
            elif os.fstat(fd).st_gid != owner.st_gid:
                os.fchown(fd, -1, owner.st_gid)
            os.fchmod(fd, 0o660)
        _verify_shared(os.fstat(fd), owner, directory=False)
        _verify_lock(locks, name, fd, owner)
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX | fcntl.LOCK_NB) if reclaim else fcntl.LOCK_SH)
        except BlockingIOError:
            yield False
        else:
            _verify_directories(bindings)
            _verify_lock(locks, name, fd, owner)
            yield True
    finally:
        if fd is not None:
            # Never unlink the coordination inode, including failed creation:
            # another consumer may already hold it or be waiting on its flock.
            os.close(fd)
        for _, _, descriptor, _ in reversed(bindings):
            os.close(descriptor)
