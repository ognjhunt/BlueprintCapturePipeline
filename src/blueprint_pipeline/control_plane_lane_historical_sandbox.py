"""Confine actual mutations while retaining a kernel read-only process observer.

Landlock also restricts foreign /proc inspection. A fixed observer thread is
therefore created first and independently barred by seccomp from filesystem
mutation or writes. It receives no paths, commands or process IDs. The action
thread then confines filesystem mutations to the two retained original roots.
Both restrictions are irreversible; unsupported native protection refuses.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import platform
import queue
import stat
import sys
import threading
import time

from .control_plane_lane_historical_processes import refuse_historical_process_references


class HistoricalSandboxError(ValueError):
    """Fixed native refusal without protected paths or process contents."""


def _require(value, code):
    if not value:
        raise HistoricalSandboxError('historical_generation_' + code)


def _checked(result):
    _require(result >= 0, 'sandbox_unknown')
    return result


def _readonly_observer():
    library = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int

    class Compare(ctypes.Structure):
        _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_uint),
                    ('a', ctypes.c_uint64), ('b', ctypes.c_uint64)]

    library.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Compare)]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)
    _require(context, 'sandbox_unknown')
    def deny(name, comparisons=()):
        number = library.seccomp_syscall_resolve_name(name.encode('ascii'))
        # Unknown syscalls are already denied by the fixed unit's native
        # @system-service allowlist. No new syscall is admitted here.
        if number < 0:
            return
        array = (Compare * len(comparisons))(*comparisons) if comparisons else None
        _checked(library.seccomp_rule_add_array(context, 0x00050000 | errno.EPERM,
                                               number, len(comparisons), array))
    try:
        for name in (
            'write writev pwrite64 pwritev pwritev2 copy_file_range sendfile splice tee vmsplice '
            'sendto sendmsg sendmmsg ioctl truncate ftruncate fallocate msync '
            'creat openat2 mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2 '
            'link linkat symlink symlinkat mknod mknodat chmod fchmod fchmodat fchmodat2 '
            'chown lchown fchown fchownat setxattr lsetxattr fsetxattr removexattr '
            'lremovexattr fremovexattr utime utimes futimesat utimensat '
            'mount umount2 pivot_root chroot setns unshare fsopen fsconfig fsmount '
            'move_mount open_tree mount_setattr ptrace process_vm_writev pidfd_getfd '
            'quotactl quotactl_fd execve execveat fork vfork clone clone3 io_uring_setup '
            'io_uring_enter io_uring_register io_setup io_submit bpf'
        ).split():
            deny(name)
        for name, argument in (('open', 1), ('openat', 2)):
            for mode in (os.O_WRONLY, os.O_RDWR):
                deny(name, (Compare(argument, 7, os.O_ACCMODE, mode),))
            for flag in (os.O_CREAT, os.O_TRUNC):
                deny(name, (Compare(argument, 7, flag, flag),))
        deny('mmap', (Compare(2, 7, 2, 2), Compare(3, 7, 1, 1)))
        deny('mprotect', (Compare(2, 7, 2, 2),))
        _checked(library.seccomp_load(context))
    finally:
        library.seccomp_release(context)


def _restrict_mutations(target, journal, reservation=None):
    _require(platform.machine() in ('x86_64', 'aarch64'), 'sandbox_unknown')
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    _require(_checked(library.syscall(444, 0, 0, 1)) >= 3, 'sandbox_unknown')

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    # ABI3 mediates WRITE_FILE and TRUNCATE independently. Read/execute remain
    # unrestricted so protected owner records and current references can reopen.
    handled = sum(1 << bit for bit in (1, *range(4, 15)))
    allowed = sum(1 << bit for bit in (1, 4, 5, 7, 8, 13, 14))
    ruleset = Ruleset(handled)
    descriptor = _checked(library.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0))
    try:
        identities = []
        roots = (target, journal)
        if reservation is not None:
            from blueprint_pipeline.control_plane_disk_ledger import DEFAULT_RESERVATION_ROOT
            info = os.fstat(reservation)
            named = os.stat(DEFAULT_RESERVATION_ROOT, follow_symlinks=False)
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0
                and stat.S_IMODE(info.st_mode) == 0o2770
                and (info.st_dev, info.st_ino) == (named.st_dev, named.st_ino)
                and not any('acl' in name for name in os.listxattr(reservation)), 'sandbox_unknown')
            roots += (reservation,)
        for root in roots:
            info = os.fstat(root)
            _require(stat.S_ISDIR(info.st_mode), 'sandbox_unknown')
            identities.append((info.st_dev, info.st_ino))
            rule = PathRule(allowed, root)
            _checked(library.syscall(445, descriptor, 1, ctypes.byref(rule), 0))
        _require(len(set(identities)) == len(identities), 'sandbox_unknown')
        # NoNewPrivileges is required and independently proven on the actual
        # fixed unit. A successful syscall establishes the current restriction.
        _checked(library.syscall(446, descriptor, 0))
    finally:
        os.close(descriptor)


def _verify_inherited_reads():
    """Landlock cannot revoke a regular writable FD opened before enforcement."""
    directory = os.open('/proc/self/fd', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    def snapshot():
        result = {}
        # Keep scandir's temporary descriptor alive while inspecting this
        # process's actual FD table; otherwise it appears spuriously vanished.
        with os.scandir(directory) as stream:
            for row in stream:
                _require(len(result) < 512 and row.name.isdigit(), 'sandbox_unknown')
                descriptor = int(row.name)
                info = os.fstat(descriptor)
                flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
                _require(not stat.S_ISREG(info.st_mode) or flags & os.O_ACCMODE == os.O_RDONLY,
                         'sandbox_unknown')
                result[row.name] = (info.st_dev, info.st_ino, info.st_mode, flags)
        return result
    try:
        _require(snapshot() == snapshot(), 'sandbox_unknown')
    finally:
        os.close(directory)


class HistoricalNativeSandbox:
    """One fixed immutable manifest and one bounded current scan per request."""

    def __init__(self, target, journal, manifest, *, tick, reservation=None):
        _require(sys.platform == 'linux' and os.geteuid() == 0, 'native_unavailable')
        _require(os.listdir('/proc/self/task') == [str(os.getpid())], 'sandbox_unknown')
        self.requests, self.answers = queue.Queue(1), queue.Queue(1)
        tick_lock = threading.Lock()
        def synchronized_tick():
            with tick_lock:
                return tick()
        self.tick, self.closed = synchronized_tick, False
        self.thread = threading.Thread(target=self._observe, args=(manifest,),
                                       name='historical-read-only-observer', daemon=True)
        self.thread.start()
        try:
            self._answer()
            _require(set(os.listdir('/proc/self/task')) ==
                {str(os.getpid()), str(self.thread.native_id)}, 'sandbox_unknown')
            _verify_inherited_reads()
            _restrict_mutations(target, journal, reservation)
        except BaseException:
            self.close()
            raise

    def _observe(self, manifest):
        try:
            _readonly_observer()
        except BaseException:
            self.answers.put(HistoricalSandboxError('historical_generation_sandbox_unknown'))
            return
        self.answers.put(None)
        while self.requests.get():
            try:
                refuse_historical_process_references(manifest, tick=self.tick)
            except Exception as error:
                self.answers.put(error)
            else:
                self.answers.put(None)

    def _answer(self):
        deadline = time.monotonic() + 6
        while True:
            self.tick()
            try:
                result = self.answers.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                _require(time.monotonic() < deadline and self.thread.is_alive(), 'sandbox_unknown')
                continue
            if result is not None:
                raise result
            return

    def refuse_references(self):
        _require(not self.closed and self.thread.is_alive(), 'sandbox_unknown')
        self.requests.put_nowait(True)
        self._answer()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.requests.put_nowait(False)
            except queue.Full:
                raise HistoricalSandboxError('historical_generation_sandbox_unknown') from None
            self.thread.join(timeout=6)
            _require(not self.thread.is_alive(), 'sandbox_unknown')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
