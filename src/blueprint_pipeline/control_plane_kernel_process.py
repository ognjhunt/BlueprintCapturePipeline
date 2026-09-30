"""Recognize an observed native kernel task without a user memory space.

This does not clear filesystem references. Callers still inspect every available
FD/cwd/root and retain their namespace, start-identity and complete-scan gates.
Only the known kernel no-mm environ ESRCH is classified; other errors remain
unknown. PF_KTHREAD is the kernel's 0x00200000 task flag (include/linux/sched.h).
"""
from __future__ import annotations


def _identity(raw, pid):
    prefix, separator, fields = raw.rpartition(b') ')
    values = fields.split()
    if not (separator and prefix.startswith(pid.encode('ascii') + b' (')
            and len(values) >= 20 and values[6].isdigit() and values[19].isdigit()
            and int(values[6]) & 0x00200000):
        return None
    return int(values[19])


def kernel_has_no_user_memory(read, pid):
    """The caller supplies its retained descriptor and original bounded reader."""
    try:
        start = _identity(read('stat', 16384), pid)
        if start is None:
            return False
        status = read('status', 16384)
        fields = {}
        for line in status.splitlines():
            name, colon, value = line.partition(b':')
            if colon:
                if name in fields:
                    return False
                fields[name] = value.split()
        if not (fields.get(b'Kthread') == [b'1']
                and all(fields.get(name) == [b'0'] * 4 for name in (b'Uid', b'Gid'))
                and not any(name.startswith(b'Vm') for name in fields)):
            return False
        return (read('cmdline', 1024**2) == b'' and read('maps', 1024**2) == b''
                and _identity(read('stat', 16384), pid) == start)
    except (OSError, ValueError, IndexError, UnicodeError):
        return False
