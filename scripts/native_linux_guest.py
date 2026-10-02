"""Host admission for the connected, complete-kernel disposable CI proof.

This admission is insufficient for guest acceptance. The guest's actual kernel,
rights, input identities and original native/full execution remain separate gates.
ADP-009D/day28: the hosted runner's process churn blocks scene cleanup proof.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import platform
import stat
import sys

KVM_GET_API_VERSION = 0xAE00
KVM_CREATE_VM = 0xAE01
GUEST_MEMORY_MIB = 4096
HOST_OVERHEAD_BYTES = 512 * 1024**2


class GuestFeasibilityError(ValueError):
    """A required host capability is absent or unknown; never a guest PASS."""


def _require(condition, code):
    if not condition:
        raise GuestFeasibilityError('native_guest_' + code)


def _parse_memory(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= 16384, 'host_memory_unknown')
    try:
        fields = {}
        for line in raw.decode('ascii').splitlines():
            key, _, value = line.partition(':')
            if key not in {'MemTotal', 'MemAvailable'}:
                continue
            parts = value.split()
            _require(key not in fields and len(parts) == 2 and parts[0].isdigit()
                     and parts[1] == 'kB', 'host_memory_unknown')
            fields[key] = int(parts[0]) * 1024
        _require(set(fields) == {'MemTotal', 'MemAvailable'}
                 and 0 < fields['MemAvailable'] <= fields['MemTotal'], 'host_memory_unknown')
        return fields['MemAvailable']
    except (UnicodeError, ValueError) as error:
        if isinstance(error, GuestFeasibilityError):
            raise
        raise GuestFeasibilityError('native_guest_host_memory_unknown') from None


def _available_memory():
    fd = os.open('/proc/meminfo', os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        return _parse_memory(os.read(fd, 16385))
    finally:
        os.close(fd)


def _available_disk(path):
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def _scratch(path):
    try:
        root = Path(os.environ.get('RUNNER_TEMP', ''))
        path = Path(path)
        _require(root.is_absolute() and path.is_absolute()
                 and not any(p.is_symlink() for p in (path, *path.parents, root, *root.parents)),
                 'scratch_untrusted')
        root, path = root.resolve(strict=True), path.resolve(strict=True)
        _require(root.is_dir() and path.is_dir() and path.is_relative_to(root), 'scratch_untrusted')
        return path
    except OSError:
        raise GuestFeasibilityError('native_guest_scratch_untrusted') from None


def host_preflight(scratch, *, required_disk_bytes):
    """Observe real KVM VM creation and resource availability before any boot.

The caller must account for the complete image/input/dependency disk footprint.
No guest is started and both kernel descriptors are closed before returning.
"""
    _require(os.environ.get('GITHUB_ACTIONS') == 'true', 'disposable_ci_required')
    _require(sys.platform == 'linux' and platform.machine() == 'x86_64', 'linux_x86_required')
    _require(os.geteuid() == 0, 'root_required')
    scratch = _scratch(scratch)
    _require(type(required_disk_bytes) is int and required_disk_bytes > 0, 'disk_requirement_invalid')
    cpus = os.cpu_count()
    _require(type(cpus) is int and cpus >= 2, 'host_cpu_insufficient')
    try:
        memory = _available_memory()
        disk = _available_disk(scratch)
    except OSError:
        raise GuestFeasibilityError('native_guest_host_resources_unknown') from None
    _require(memory >= GUEST_MEMORY_MIB * 1024**2 + HOST_OVERHEAD_BYTES, 'host_memory_insufficient')
    _require(disk >= required_disk_bytes, 'host_disk_insufficient')
    fd = vm = None
    try:
        fd = os.open('/dev/kvm', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        info = os.fstat(fd)
        _require(stat.S_ISCHR(info.st_mode) and info.st_uid == 0
                 and info.st_rdev == os.makedev(10, 232), 'kvm_device_untrusted')
        version = fcntl.ioctl(fd, KVM_GET_API_VERSION, 0)
        _require(version == 12, 'kvm_api_unsupported')
        vm = fcntl.ioctl(fd, KVM_CREATE_VM, 0)
        _require(type(vm) is int and vm >= 0, 'kvm_initialization_unknown')
    except OSError:
        raise GuestFeasibilityError('native_guest_kvm_initialization_unknown') from None
    finally:
        if vm is not None:
            os.close(vm)
        if fd is not None:
            os.close(fd)
    return dict(status='host_capabilities_observed', kvm_api_version=version,
                kvm_vm_created=True, available_memory_bytes=memory, available_disk_bytes=disk,
                guest_memory_mib=GUEST_MEMORY_MIB, guest_acceptance_proven=False,
                guest_security_primitives_proven=False)
