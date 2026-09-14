"""Kernel-enforced CAD isolation for Linux containers without namespace privileges.

Landlock restricts file contents and mutations; seccomp restricts networking,
process inspection and namespace escape. Neither requires CAP_SYS_ADMIN.
The trusted launcher uses only stdlib, before executing any candidate code.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys


class KernelSandboxError(RuntimeError):
    pass


def _identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "device": stat.st_dev, "inode": stat.st_ino}


def run_with_landlock(
    argv, *, roots, write_root, cwd, env, timeout, check, capture_output, text, abi_limit=None
):
    """Start a fresh isolated launcher, avoiding preexec_fn in threaded workers."""
    system = [
        Path(p)
        for p in (
            "/usr",
            "/bin",
            "/lib",
            "/lib64",
            "/etc/fonts",
            "/etc/ld.so.cache",
            "/etc/localtime",
            "/dev/urandom",
            "/dev/random",
        )
        if Path(p).exists()
    ]
    read = list(dict.fromkeys(Path(p).resolve() for p in [*system, *roots]))
    policy = {
        "read": [_identity(p) for p in read],
        "write": _identity(write_root),
        "cwd": str(cwd),
        "environment": env,
        "abi_limit": abi_limit,
    }
    python = Path("/usr/bin/python3")
    if not python.is_file():
        raise KernelSandboxError("asset_landlock_system_python_missing")
    # Never let candidate-controlled loader paths affect the trusted launcher.
    launcher_env = {
        k: v
        for k, v in env.items()
        if k not in {"PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}
    }
    command = [
        str(python),
        "-I",
        "-S",
        str(Path(__file__).resolve()),
        json.dumps(policy),
        *map(str, argv),
    ]
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=launcher_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        close_fds=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    finally:
        # Descendants cannot change their session/group after the filter loads.
        # Clean them even if the candidate exits without waiting for its children.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if not capture_output:
        for stream, value in ((sys.stdout, stdout), (sys.stderr, stderr)):
            (stream if text else stream.buffer).write(value)
        stdout = stderr = None
    result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def _checked(result, operation):
    if result < 0:
        raise KernelSandboxError(operation + ":" + os.strerror(ctypes.get_errno()))
    return result


def _restrict_files(libc, policy):
    # Linux generic syscall numbers, shared by the supported 64-bit ABIs.
    if platform.machine() not in {"x86_64", "aarch64"}:
        raise KernelSandboxError("asset_landlock_architecture_unsupported")
    libc.syscall.restype = ctypes.c_long
    abi = _checked(libc.syscall(444, 0, 0, 1), "asset_landlock_abi")
    limit = policy.get("abi_limit")
    if limit is not None:
        if type(limit) is not int or not 1 <= limit <= abi:
            raise KernelSandboxError("asset_landlock_abi_limit_invalid")
        abi = limit
    if abi < 1:
        raise KernelSandboxError("asset_landlock_abi_unavailable")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    # ABI1 (Linux5.15 providers) lacks REFER/TRUNCATE. Cross-directory
    # reparenting is implicitly denied there; seccomp closes truncation gaps.
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    rules = Ruleset(handled)
    fd = _checked(
        libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0), "asset_landlock_create"
    )
    read = (1 << 0) | (1 << 2) | (1 << 3)
    writable = handled & ~((1 << 6) | (1 << 11))  # Never create device nodes.

    def add(record, access):
        path = record["path"]
        handle = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            st = os.fstat(handle)
            if (st.st_dev, st.st_ino) != (record["device"], record["inode"]):
                raise KernelSandboxError("asset_landlock_root_identity_changed")
            import stat

            if stat.S_ISLNK(st.st_mode):
                raise KernelSandboxError("asset_landlock_symlink_root")
            if not stat.S_ISDIR(st.st_mode):
                access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            rule = PathRule(access, handle)
            _checked(libc.syscall(445, fd, 1, ctypes.byref(rule), 0), "asset_landlock_add_rule")
        finally:
            os.close(handle)

    try:
        for record in policy["read"]:
            add(record, read)
        add(policy["write"], writable)
        add(_identity("/dev/null"), (1 << 1) | (1 << 2))
        # Current-process metadata and public CPU/memory facts, not other PIDs.
        for path in (
            "/proc/self",
            "/proc/cpuinfo",
            "/proc/meminfo",
            "/proc/stat",
            "/proc/uptime",
            "/proc/version",
        ):
            if Path(path).exists():
                add(_identity(path), read)
        _checked(libc.syscall(446, fd, 0), "asset_landlock_restrict")
    finally:
        os.close(fd)
    return abi


def _restrict_syscalls(abi):
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int

    class Compare(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_uint),
            ("a", ctypes.c_uint64),
            ("b", ctypes.c_uint64),
        ]

    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Compare),
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x00050000 | errno.EPERM)  # Unknown operations deny.
    if not context:
        raise KernelSandboxError("asset_seccomp_init_failed")

    def rule(name, comparisons=(), action=0x7FFF0000):
        number = library.seccomp_syscall_resolve_name(name.encode())
        if number < 0:
            return  # Unknown calls stay denied; this cannot widen the policy.
        array = (Compare * len(comparisons))(*comparisons) if comparisons else None
        rc = library.seccomp_rule_add_array(context, action, number, len(comparisons), array)
        if rc:
            raise KernelSandboxError("asset_seccomp_rule_failed:" + name)

    try:
        # File-content operations are checked by Landlock. Metadata mutation
        # (chmod/chown/xattrs/timestamps) is deliberately NOT allowed: Landlock
        # ABI3 does not mediate it, and container UID0 may own outside files.
        allowed = (
            "read write readv writev pread64 pwrite64 preadv pwritev preadv2 pwritev2 "
            "close close_range lseek mmap mprotect munmap mremap madvise brk msync "
            "creat fstat newfstatat stat lstat statx statfs fstatfs "
            "access faccessat faccessat2 readlink readlinkat getdents getdents64 "
            "mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2 link linkat "
            "symlink symlinkat ftruncate fsync fdatasync sync_file_range "
            "fallocate copy_file_range sendfile splice tee vmsplice flock "
            "getpid gettid getuid geteuid getgid getegid getgroups getresuid "
            "getresgid getpgrp getcpu sched_yield "
            "sched_get_priority_min sched_get_priority_max "
            "getrlimit setrlimit getrusage times time clock_gettime gettimeofday "
            "clock_getres clock_nanosleep nanosleep alarm setitimer getitimer "
            "timer_create timer_settime timer_gettime timer_getoverrun timer_delete "
            "timerfd_create timerfd_settime timerfd_gettime eventfd eventfd2 "
            "pipe pipe2 dup dup2 dup3 getsockname getpeername getsockopt setsockopt "
            "sendto recvfrom sendmsg recvmsg shutdown epoll_create epoll_create1 epoll_ctl epoll_wait "
            "epoll_pwait epoll_pwait2 poll ppoll select pselect6 "
            "rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack rt_sigsuspend "
            "rt_sigtimedwait rt_sigpending signalfd signalfd4 restart_syscall "
            "set_tid_address set_robust_list rseq futex futex_waitv "
            "arch_prctl prctl getrandom uname sysinfo getcwd chdir fchdir umask "
            "memfd_create execve execveat exit exit_group wait4 waitid fork vfork"
        )
        for name in allowed.split():
            rule(name)
        # READ-only O_TRUNC is a Linux truncation loophole not mediated by
        # Landlock ABI1/2. Writable opens remain subject to WRITE_FILE rules.
        for name, index in (("open", 1), ("openat", 2)):
            rule(name, (Compare(index, 7, os.O_ACCMODE | os.O_TRUNC, 0),))
            for mode in (os.O_WRONLY, os.O_RDWR):
                rule(name, (Compare(index, 7, os.O_ACCMODE, mode),))
        if abi >= 3:
            rule("openat2")  # Its pointed-to flags cannot be filtered by seccomp.
            rule("truncate")
        # Private Unix socketpairs support asyncio without host socket access.
        rule(
            "socketpair", (Compare(0, 4, 1, 0), Compare(1, 7, 15, 1), Compare(2, 4, 0, 0))
        )  # Connected AF_UNIX streams only.
        for name in ("kill", "tgkill", "rt_sigqueueinfo", "rt_tgsigqueueinfo"):
            rule(name, (Compare(0, 4, os.getpid(), 0),))
        for name in ("prlimit64", "sched_setaffinity", "getpgid", "getsid",
                     "sched_getaffinity", "sched_getparam", "sched_getscheduler"):
            rule(name, (Compare(0, 4, 0, 0),))
            rule(name, (Compare(0, 4, os.getpid(), 0),))
        # No F_SETOWN/F_NOTIFY/F_SETLEASE signal-delivery backdoor to the parent.
        for operation in (0, 1, 2, 3, 4, 5, 6, 7, 36, 37, 38, 1030, 1031, 1032, 1033, 1034):
            rule("fcntl", (Compare(1, 4, operation, 0),))
        for operation in (0x5401, 0x5413, 0x541B, 0x5421, 0x5450, 0x5451):
            rule("ioctl", (Compare(1, 4, operation, 0),))
        # Only the known glibc thread/fork flags; no CLONE_PARENT, namespaces,
        # or future unknown flags. Exit notifications stay SIGCHLD (or none).
        forbidden_clone_flags = (~0x013D4FFF) & 0xFFFFFFFFFFFFFFFF
        for exit_signal in (0, signal.SIGCHLD):
            rule("clone", (Compare(0, 7, forbidden_clone_flags | 255, exit_signal),))
        # glibc falls back to clone, where namespace flags are checked.
        rule("clone3", action=0x00050000 | errno.ENOSYS)
        if library.seccomp_load(context):
            raise KernelSandboxError("asset_seccomp_load_failed")
    finally:
        library.seccomp_release(context)


def _clear_capabilities(libc):
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    _checked(libc.prctl(38, 1, 0, 0, 0), "asset_no_new_privileges")
    header = Header(0x20080522, 0)
    data = (Data * 2)()
    _checked(libc.capset(ctypes.byref(header), data), "asset_clear_capabilities")
    # With no-new-privileges set, exec cannot reacquire capabilities, even as UID0.


def main():
    try:
        policy = json.loads(sys.argv[1])
        argv = sys.argv[2:]
        if not argv:
            raise KernelSandboxError("asset_landlock_command_missing")
        libc = ctypes.CDLL(None, use_errno=True)
        _clear_capabilities(libc)
        abi = _restrict_files(libc, policy)
        _restrict_syscalls(abi)
        os.closerange(3, 65536)
        os.chdir(policy["cwd"])
        os.execve(argv[0], argv, policy["environment"])
    except Exception as exc:
        print("ASSET_KERNEL_SANDBOX_FAILED:" + str(exc), file=sys.stderr)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
