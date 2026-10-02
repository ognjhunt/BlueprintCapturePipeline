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
    monkeypatch.setattr(guest.os, 'sched_getaffinity', lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(guest, '_cgroup_limits', lambda: (2, 8 * 1024**3), raising=False)
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


@pytest.mark.parametrize('constraint', ['affinity', 'quota', 'cgroup_memory'])
def test_effective_launch_process_constraints_cannot_use_machine_resources(host, monkeypatch, constraint):
    root, calls = host
    if constraint == 'affinity':
        monkeypatch.setattr(guest.os, 'sched_getaffinity', lambda pid: {0})
    else:
        monkeypatch.setattr(guest, '_cgroup_limits', lambda: (1, 8 * 1024**3) if constraint == 'quota' else (2, 1024**3))
    code = 'host_memory_insufficient' if constraint == 'cgroup_memory' else 'host_cpu_insufficient'
    with pytest.raises(guest.GuestFeasibilityError, match=code):
        guest.host_preflight(root, required_disk_bytes=2 * 1024**3)
    assert calls == []


def test_vm_close_failure_still_attempts_kvm_close_and_never_admits(host, monkeypatch):
    root, calls = host
    def close(fd):
        calls.append(('close', fd))
        if fd == 42:
            raise OSError(errno.EIO, 'close failed')
    monkeypatch.setattr(guest.os, 'close', close)
    with pytest.raises(OSError):
        guest.host_preflight(root, required_disk_bytes=2 * 1024**3)
    assert calls[-2:] == [('close', 42), ('close', 41)]


@pytest.fixture
def cgroup_tree(tmp_path, monkeypatch):
    root = tmp_path / 'cgroup'
    leaf = root / 'parent' / 'worker'
    leaf.mkdir(parents=True)
    monkeypatch.setattr(guest, '_cgroup_path', lambda: (root, ['parent', 'worker']))
    for directory in (leaf.parent, leaf):
        (directory / 'cpu.max').write_text('max 100000\n')
        (directory / 'memory.max').write_text('max\n')
    return root, leaf


def test_parent_cgroup_limits_apply_even_when_leaf_is_unlimited(cgroup_tree):
    root, leaf = cgroup_tree
    (leaf.parent / 'cpu.max').write_text('150000 100000\n')
    (leaf.parent / 'memory.max').write_text('5000000000\n')
    (leaf.parent / 'memory.current').write_text('2000000000\n')
    assert guest._cgroup_limits() == (1, 3000000000)
    (root / 'cpu.max').write_text('100000 100000\n')
    (root / 'memory.max').write_text('2000000000\n')
    (root / 'memory.current').write_text('1900000000\n')
    assert guest._cgroup_limits() == (1, 100000000)


@pytest.mark.parametrize('bad', ['missing', 'invalid', 'oversize', 'alias'])
def test_unknown_cgroup_limits_never_become_unlimited(cgroup_tree, bad):
    _, leaf = cgroup_tree
    path = leaf / 'cpu.max'
    if bad == 'invalid':
        path.write_text('max 0\n')
    elif bad == 'oversize':
        path.write_text('9' * 4097)
    else:
        path.unlink()
        if bad == 'alias':
            path.symlink_to(leaf.parent / 'cpu.max')
    with pytest.raises(guest.GuestFeasibilityError, match='host_cgroup_unknown'):
        guest._cgroup_limits()


@pytest.fixture
def initial_namespace(monkeypatch):
    monkeypatch.setattr(guest, '_initial_cgroup_namespace', lambda: None)


@pytest.mark.parametrize('row', ['0::/../worker', '0::/worker\n1:cpu:/other', '0::/a//b'])
def test_ambiguous_cgroup_membership_refuses(row, monkeypatch, initial_namespace):
    monkeypatch.setattr(guest, '_bounded_text', lambda path, limit=4096: row)
    with pytest.raises(guest.GuestFeasibilityError, match='host_cgroup_unknown'):
        guest._cgroup_path()


def test_cgroup_subtree_mount_does_not_hide_parent_limits(monkeypatch, initial_namespace):
    monkeypatch.setattr(guest, '_bounded_text', lambda path, limit=4096:
                        '0::/worker' if str(path) == '/proc/self/cgroup' else
                        '1 0 0:1 /hidden-parent /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
    with pytest.raises(guest.GuestFeasibilityError, match='host_cgroup_unknown'):
        guest._cgroup_path()


def test_virtual_namespace_root_cannot_claim_unlimited_kernel_ancestry(monkeypatch):
    monkeypatch.setattr(guest, '_bounded_text', lambda path, limit=4096:
                        '0::/' if str(path) == '/proc/self/cgroup' else
                        '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
    monkeypatch.setattr(guest.os, 'open', lambda *args: 41)
    monkeypatch.setattr(guest.os, 'fstat', lambda fd: type('Stat', (), {'st_ino': 12345})())
    monkeypatch.setattr(guest.fcntl, 'ioctl', lambda *args: 0x02000000)
    monkeypatch.setattr(guest.os, 'close', lambda fd: None)
    with pytest.raises(guest.GuestFeasibilityError, match='host_cgroup_namespace_unknown'):
        guest._cgroup_path()


@pytest.mark.skipif(not (guest.sys.platform == 'linux' and
                        os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1'),
                    reason='real cgroup namespace boundary requires disposable Linux; portable skip is unproven')
def test_actual_new_cgroup_namespace_is_not_initial_host_ancestry():
    import subprocess
    script = ('import importlib.util; '
              f's=importlib.util.spec_from_file_location("guest", {str(guest.__file__)!r}); '
              'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
              'm._cgroup_path()')
    result = subprocess.run(['sudo', '-n', 'unshare', '--cgroup', '--',
                             '/usr/bin/python3', '-I', '-S', '-c', script],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode != 0
    assert 'native_guest_host_cgroup_namespace_unknown' in result.stderr, result.stderr
