# Covers (for impacted-test selection):
#   scripts/native_linux_guest.py
"""Portable host admission boundaries, never actual guest acceptance."""
import errno
import os
import stat

import pytest

from scripts import native_linux_guest as guest


@pytest.fixture
def host(monkeypatch, tmp_path):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setattr(guest.sys, 'platform', 'linux')
    monkeypatch.setattr(guest.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(guest.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(guest.os, 'cpu_count', lambda: 2)
    monkeypatch.setattr(guest, '_available_memory', lambda: 8 * 1024**3)
    monkeypatch.setattr(guest, '_available_disk', lambda path: 12 * 1024**3)
    calls = []
    monkeypatch.setattr(guest.os, 'open', lambda *args: calls.append(('open', args)) or 41)
    monkeypatch.setattr(guest.os, 'fstat', lambda fd: type('Stat', (), {
        'st_mode': stat.S_IFCHR | 0o660, 'st_uid': 0, 'st_rdev': os.makedev(10, 232)})())
    monkeypatch.setattr(guest.fcntl, 'ioctl', lambda fd, cmd, arg: calls.append(('ioctl', fd, cmd, arg)) or (12 if cmd == guest.KVM_GET_API_VERSION else 42))
    monkeypatch.setattr(guest.os, 'close', lambda fd: calls.append(('close', fd)))
    return tmp_path, calls


def test_actual_kvm_vm_creation_and_both_descriptor_closes(host):
    root, calls = host
    result = guest.host_preflight(root, required_disk_bytes=2 * 1024**3)
    assert [call for call in calls if call[0] == 'ioctl'] == [
        ('ioctl', 41, guest.KVM_GET_API_VERSION, 0), ('ioctl', 41, guest.KVM_CREATE_VM, 0)]
    assert calls[-2:] == [('close', 42), ('close', 41)]
    assert calls[0][1] == ('/dev/kvm', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    assert result['guest_acceptance_proven'] is False
    assert result['kvm_api_version'] == 12 and result['kvm_vm_created'] is True


@pytest.mark.parametrize('change,code', [
    ('ci', 'disposable_ci_required'), ('platform', 'linux_x86_required'),
    ('uid', 'root_required'), ('memory', 'host_memory_insufficient'),
    ('disk', 'host_disk_insufficient'), ('cpu', 'host_cpu_insufficient'),
])
def test_infeasible_host_never_opens_kvm(host, monkeypatch, change, code):
    root, calls = host
    if change == 'ci':
        monkeypatch.delenv('GITHUB_ACTIONS')
    elif change == 'platform':
        monkeypatch.setattr(guest.platform, 'machine', lambda: 'arm64')
    elif change == 'uid':
        monkeypatch.setattr(guest.os, 'geteuid', lambda: 1001)
    elif change == 'memory':
        monkeypatch.setattr(guest, '_available_memory', lambda: 1024**3)
    elif change == 'disk':
        monkeypatch.setattr(guest, '_available_disk', lambda path: 1)
    else:
        monkeypatch.setattr(guest.os, 'cpu_count', lambda: 1)
    with pytest.raises(guest.GuestFeasibilityError, match=code):
        guest.host_preflight(root, required_disk_bytes=2 * 1024**3)
    assert calls == []


def test_vm_creation_error_stays_unknown_and_closes_kvm(host, monkeypatch):
    root, calls = host
    def ioctl(fd, command, argument):
        if command == guest.KVM_GET_API_VERSION:
            return 12
        raise OSError(errno.EPERM, 'PRIVATE_HOST_PATH')
    monkeypatch.setattr(guest.fcntl, 'ioctl', ioctl)
    with pytest.raises(guest.GuestFeasibilityError, match='kvm_initialization_unknown') as raised:
        guest.host_preflight(root, required_disk_bytes=2 * 1024**3)
    assert 'PRIVATE_HOST_PATH' not in str(raised.value)
    assert calls[-1] == ('close', 41)


def test_wrong_kvm_api_does_not_create_guest(host, monkeypatch):
    root, calls = host
    monkeypatch.setattr(guest.fcntl, 'ioctl', lambda *args: 11)
    with pytest.raises(guest.GuestFeasibilityError, match='kvm_api_unsupported'):
        guest.host_preflight(root, required_disk_bytes=2 * 1024**3)
    assert calls[-1] == ('close', 41)


def test_scratch_alias_and_outside_runner_temp_are_refused_before_kvm(host, tmp_path, monkeypatch):
    root, calls = host
    alias = root / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(guest.GuestFeasibilityError, match='scratch_untrusted'):
        guest.host_preflight(alias, required_disk_bytes=2 * 1024**3)
    monkeypatch.setenv('RUNNER_TEMP', str(root / 'different'))
    with pytest.raises(guest.GuestFeasibilityError, match='scratch_untrusted'):
        guest.host_preflight(root, required_disk_bytes=2 * 1024**3)
    assert calls == []


@pytest.mark.parametrize('raw', [
    b'MemTotal: 8192 kB\n',
    b'MemTotal: 8192 kB\nMemAvailable: 10 kB\nMemAvailable: 10 kB\n',
    b'MemTotal: 8192 kB\nMemAvailable: 8193 kB\n',
    b'MemTotal: 8192 kB\nMemAvailable: 10 bytes\n',
    b'MemTotal: 8192 kB\nMemAvailable: 10 kB\n' + b' ' * 16384,
])
def test_missing_ambiguous_or_unbounded_memory_never_supplies_admission(raw):
    with pytest.raises(guest.GuestFeasibilityError, match='host_memory_unknown'):
        guest._parse_memory(raw)


def test_memory_uses_available_bytes_and_does_not_treat_total_as_free():
    assert guest._parse_memory(b'MemTotal: 8192 kB\nMemFree: 1024 kB\nMemAvailable: 2048 kB\n') == 2048 * 1024
