"""Finite retained current-process procfs view for direct anonymous creation.

Linux getdents64 reads the already owned directory without a second temporary
scanner FD. Thus no inventory handle closes and becomes a false borrowed slot.
This observes kernel slots under the controlled process/thread contract; it is
not atomic protection against hostile same-UID descriptor replacement.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import sys

from . import control_plane_lane_owner_consents as owners
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _require


def _names(raw):
    """Bounded Linux dirent64 framing; unknown/truncated rows refuse."""
    _require(isinstance(raw, bytes) and 0 < len(raw) <= 8192,
             'historical_publication_descriptor_namespace_unknown')
    offset, names = 0, []
    while offset < len(raw):
        _require(len(raw) - offset >= 24, 'historical_publication_descriptor_namespace_unknown')
        size = int.from_bytes(raw[offset + 16:offset + 18], sys.byteorder)
        _require(size >= 24 and size % 8 == 0 and offset + size <= len(raw),
                 'historical_publication_descriptor_namespace_unknown')
        text = raw[offset + 19:offset + size]
        _require(b'\0' in text, 'historical_publication_descriptor_namespace_unknown')
        name = text.split(b'\0', 1)[0]
        if name not in (b'.', b'..'):
            _require(raw[offset + 18] == 10 and re.fullmatch(rb'(?:0|[1-9][0-9]{0,9})', name),
                     'historical_publication_descriptor_namespace_unknown')
            names.append(int(name))
        offset += size
    return names


class CreationProcView:
    """Directly acquired view; no transferred descriptor/path parameter."""

    def __init__(self, files):
        _require(sys.platform == 'linux' and platform.machine() in ('x86_64', 'aarch64')
                 and ctypes.sizeof(ctypes.c_long) == ctypes.sizeof(ctypes.c_void_p) == 8,
                 'historical_publication_descriptor_namespace_unknown')
        self.files = files
        try:
            self.library = ctypes.CDLL(None, use_errno=True)
            self.library.fstatfs.argtypes = [ctypes.c_int, ctypes.c_void_p]
            self.library.fstatfs.restype = ctypes.c_int
            self.library.getdents64.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
            self.library.getdents64.restype = ctypes.c_ssize_t
        except (OSError, AttributeError):
            raise OwnerTargetVersionError('historical_publication_descriptor_namespace_unknown') from None
        self.fd = files.open('/proc/self/fd', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        self.guard()

    def guard(self):
        self.files.budget.tick()
        self.files.location(self.fd)
        # Supported 64-bit Linux ABIs use a native long for f_type. The aligned
        # 256-byte buffer bounds and exceeds their complete statfs structures.
        buffer = (ctypes.c_long * 32)()
        _require(self.library.fstatfs(self.fd, ctypes.byref(buffer)) == 0 and buffer[0] == 0x9FA0,
                 'historical_publication_descriptor_namespace_unknown')
        named = os.stat('/proc/' + str(os.getpid()) + '/fd', follow_symlinks=False)
        _require(owners._metadata(named) == owners._metadata(os.fstat(self.fd)),
                 'historical_publication_descriptor_namespace_unknown')

    def slots(self):
        self.guard()
        os.lseek(self.fd, 0, os.SEEK_SET)
        seen = set()
        buffer = ctypes.create_string_buffer(8192)
        for _ in range(9):
            self.guard()
            count = self.library.getdents64(self.fd, buffer, len(buffer))
            _require(0 <= count <= len(buffer), 'historical_publication_descriptor_namespace_unknown')
            self.guard()
            if count == 0:
                _require(self.fd in seen, 'historical_publication_descriptor_namespace_unknown')
                return seen
            for number in _names(buffer.raw[:count]):
                self.files.budget.charge('entries')
                _require(len(seen) < 256 and number not in seen,
                         'historical_publication_descriptor_namespace_unknown')
                seen.add(number)
        raise OwnerTargetVersionError('historical_publication_descriptor_namespace_unknown')

    def stat(self, number):
        self.guard()
        _require(type(number) is int and 0 <= number <= 2**31 - 1,
                 'historical_publication_descriptor_namespace_unknown')
        return os.stat(str(number), dir_fd=self.fd, follow_symlinks=True)

    def close(self):
        self.files.close(self.fd)
        _require(self.fd not in self.files.owned and not self.files.unresolved,
                 'owner_target_descriptor_cleanup_failed')
