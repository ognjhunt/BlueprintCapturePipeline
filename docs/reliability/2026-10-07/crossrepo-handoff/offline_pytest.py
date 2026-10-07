"""Offline reliability runner: clean env + inherited kernel network denial."""

import ctypes
import errno
import os
import sys
from pathlib import Path

keep = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "PYTHONPATH") if k in os.environ}
os.environ.clear()
os.environ.update(keep)
os.environ.update(
    {
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "BLUEPRINT_WEBSITE_OBJECT_SPEC_AGENT": "0",
        "GCE_METADATA_DISABLED": "true",
    }
)
lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
lib.seccomp_release.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(0x7FFF0000)
assert ctx
for syscall in ("socket", "connect", "sendto", "sendmsg", "sendmmsg"):
    assert (
        lib.seccomp_rule_add(
            ctx, 0x50000 | errno.EPERM, lib.seccomp_syscall_resolve_name(syscall.encode()), 0
        )
        == 0
    )
assert lib.seccomp_load(ctx) == 0
lib.seccomp_release(ctx)
# Install kernel denial before importing network modules.
import socket

try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError:
    pass
else:
    raise RuntimeError("kernel network denial not active")
print("OFFLINE: network syscalls denied; environment cleared; no live credentials")
sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
# Import the test runner only after isolation.
import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
