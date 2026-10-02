# Covers (for impacted-test selection):
#   scripts/native_linux_guest_execution.py
"""Guest orchestration boundaries; portable results are never native proof."""
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import native_linux_guest_execution as execution


def _receipt(tmp_path, *, live=False):
    # A portable double is deliberately not a native execution receipt.
    image = tmp_path / 'image'
    image.write_bytes(b'tiny private fixture')
    info = image.stat()
    command = execution.qemu_command(image, phase='test', image_fd=41)
    process = SimpleNamespace(args=command, poll=lambda: None if live else 0, wait=lambda timeout: 0)
    return execution.GuestRun(process, str(image), (info.st_dev, info.st_ino), tuple(command), 'test', 41)


def test_test_boot_has_kvm_private_disk_and_no_guest_control_channel(tmp_path):
    image = tmp_path / 'private.qcow2'
    command = execution.qemu_command(image, phase='test')
    assert command[:1] == ['/usr/bin/qemu-system-x86_64']
    assert command[command.index('-machine') + 1] == 'q35,accel=kvm'
    assert command[command.index('-cpu') + 1] == 'host'
    assert command[command.index('-m') + 1] == '4096'
    assert command[command.index('-nic') + 1] == 'none'
    assert command[command.index('-monitor') + 1] == 'none'
    assert command[command.index('-serial') + 1] == 'stdio'
    assert command[command.index('-drive') + 1] == f'file={image},format=qcow2,if=virtio,cache=none'
    assert not any('virtfs' in arg or 'socket' in arg or 'guest-agent' in arg for arg in command)


@pytest.mark.parametrize('phase', ['unknown', '', 'TEST'])
def test_no_implicit_network_or_execution_phase(phase, tmp_path):
    with pytest.raises(execution.GuestExecutionError):
        execution.qemu_command(tmp_path / 'image', phase=phase)


@pytest.mark.parametrize('image', ['relative', '/tmp/a,b', '/tmp/a\nimage'])
def test_image_cannot_inject_drive_options(image):
    with pytest.raises(execution.GuestExecutionError):
        execution.qemu_command(Path(image), phase='test')


def test_provision_network_is_available_only_in_explicit_provision_phase(tmp_path):
    command = execution.qemu_command(tmp_path / 'image', phase='provision')
    assert '-netdev' in command and 'user,id=provision' in command
    assert not any('hostfwd' in arg for arg in command)


def test_live_guest_cannot_be_opened_for_offline_evidence(tmp_path, monkeypatch):
    process = _receipt(tmp_path, live=True)
    called = []
    monkeypatch.setattr(execution.subprocess, 'run', lambda *a, **kw: called.append(a))
    with pytest.raises(execution.GuestExecutionError, match='guest_not_reaped'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence', ['native-junit.xml'], deadline_monotonic=time.monotonic() + 60)
    assert not called


@pytest.mark.parametrize('names', [['../etc/shadow'], ['a/b'], ['native-junit.xml', 'native-junit.xml'], []])
def test_unbounded_or_unapproved_evidence_selector_refuses(tmp_path, names, monkeypatch):
    process = _receipt(tmp_path)
    called = []
    monkeypatch.setattr(execution.subprocess, 'run', lambda *a, **kw: called.append(a))
    with pytest.raises(execution.GuestExecutionError, match='evidence_selection_invalid'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence', names, deadline_monotonic=time.monotonic() + 60)
    assert not called


def test_guest_evidence_alias_is_refused_without_download(tmp_path, monkeypatch):
    process = _receipt(tmp_path)
    calls = []
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, b'true\n', b'')
    monkeypatch.setattr(execution, '_bounded_command', run)
    with pytest.raises(execution.GuestExecutionError, match='evidence_alias'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence', ['native-junit.xml'], deadline_monotonic=time.monotonic() + 60)
    assert all('download' not in call for call in calls)


def test_guest_evidence_size_checked_before_download(tmp_path, monkeypatch):
    process = _receipt(tmp_path)
    calls = []
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0,
            b'false\n' if 'is-symlink' in command else (str(execution.MAX_EVIDENCE_BYTES + 1) + '\n').encode(), b'')
    monkeypatch.setattr(execution, '_bounded_command', run)
    with pytest.raises(execution.GuestExecutionError, match='evidence_size'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence', ['native-junit.xml'], deadline_monotonic=time.monotonic() + 60)
    assert all('download' not in call for call in calls)


@pytest.mark.parametrize('behavior,code', [
    ('import time; time.sleep(5)', 'phase_deadline_expired'),
    ('print("x" * 1000)', 'serial_limit'),
    ('raise SystemExit(3)', 'qemu_failed'),
    ('print("original stdout")', None),
])
def test_original_process_failure_timeout_and_output_are_reaped(tmp_path, monkeypatch, behavior, code):
    # Real tiny host subprocesses verify host lifecycle only, not a guest boot.
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    monkeypatch.setattr(execution, 'host_preflight', lambda *args, **kwargs: None)
    monkeypatch.setattr(execution, 'qemu_command', lambda *args, **kwargs: [sys.executable, '-c', behavior])
    monkeypatch.setattr(execution, 'MAX_SERIAL_BYTES', 128)
    (tmp_path / 'image').write_bytes(b'tiny private image fixture')
    processes = []
    original = execution.subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(execution.subprocess, 'Popen', popen)
    arguments = dict(phase='test', deadline_monotonic=time.monotonic() + (0.5 if 'sleep' in behavior else 3), required_disk_bytes=1)
    if code:
        with pytest.raises(execution.GuestExecutionError, match=code):
            execution.run_vm(tmp_path / 'image', tmp_path / 'serial.log', **arguments)
    else:
        result = execution.run_vm(tmp_path / 'image', tmp_path / 'serial.log', **arguments)
        assert result.process is processes[0]
        assert (tmp_path / 'serial.log').read_text() == 'original stdout\n'
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdout.closed


def test_sealed_image_refuses_symlink_before_boot_or_extraction(tmp_path):
    payload = tmp_path / 'payload'
    payload.write_bytes(b'tiny fixture')
    alias = tmp_path / 'alias'
    alias.symlink_to(payload)
    with pytest.raises(execution.GuestExecutionError, match='image_untrusted'):
        execution._sealed_image(alias)


def test_arbitrary_reaped_host_process_is_not_a_vm_disk_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    called = []
    monkeypatch.setattr(execution, '_guestfish', lambda *args: called.append(args) or 'false')
    process = SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)
    with pytest.raises(execution.GuestExecutionError, match='guest_result_unbound'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence',
                                   ['native-junit.xml'], deadline_monotonic=time.monotonic() + 60)
    assert not called


def test_bounded_external_tool_output_stops_and_reaps_original_process(monkeypatch):
    processes = []
    original = execution.subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(execution.subprocess, 'Popen', popen)
    with pytest.raises(execution.GuestExecutionError, match='tool_output_limit'):
        execution._bounded_command([sys.executable, '-c', 'print("x" * 10000)'],
                                   deadline=time.monotonic() + 2, max_output_bytes=128)
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed


def test_replaced_host_destination_never_overwrites_foreign_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    destination = tmp_path / 'evidence'
    foreign = destination / 'native-junit.xml'
    process = _receipt(tmp_path)
    def guestfish(image, arguments, deadline, **kwargs):
        if arguments[0] == 'is-symlink':
            return 'false'
        if arguments[0] == 'filesize':
            return '8'
        destination.rename(tmp_path / 'owned-evidence')
        destination.mkdir()
        foreign.write_bytes(b'foreign-original')
        selected = arguments[-1]
        if selected.startswith('/proc/self/fd/'):
            execution.os.write(int(selected.rsplit('/', 1)[1]), b'guest00\n')
        else:
            Path(selected).write_bytes(b'guest00\n')
        return ''
    monkeypatch.setattr(execution, '_guestfish', guestfish)
    with pytest.raises(execution.GuestExecutionError, match='evidence_destination_changed'):
        execution.extract_evidence(process, tmp_path / 'image', destination, ['native-junit.xml'],
                                   deadline_monotonic=time.monotonic() + 60)
    assert foreign.read_bytes() == b'foreign-original'


def test_final_extraction_after_original_deadline_never_returns_success(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    process = _receipt(tmp_path)
    observed = [time.monotonic()]
    deadline = observed[0] + 60
    monkeypatch.setattr(execution.time, 'monotonic', lambda: observed[0])
    def guestfish(image, arguments, original_deadline, **kwargs):
        assert original_deadline == deadline
        if arguments[0] == 'is-symlink':
            return 'false'
        if arguments[0] == 'filesize':
            return '8'
        selected = arguments[-1]
        if selected.startswith('/proc/self/fd/'):
            execution.os.write(int(selected.rsplit('/', 1)[1]), b'guest00\n')
        else:
            Path(selected).write_bytes(b'guest00\n')
        observed[0] = deadline + 1
        return ''
    monkeypatch.setattr(execution, '_guestfish', guestfish)
    with pytest.raises(execution.GuestExecutionError, match='phase_deadline_expired'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence', ['native-junit.xml'],
                                   deadline_monotonic=deadline)


def test_replacement_disk_cannot_use_prior_guest_result(tmp_path, monkeypatch):
    guest = _receipt(tmp_path)
    image = tmp_path / 'image'
    image.rename(tmp_path / 'original-image')
    image.write_bytes(b'foreign replacement')
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    calls = []
    monkeypatch.setattr(execution, '_guestfish', lambda *args, **kw: calls.append(args))
    with pytest.raises(execution.GuestExecutionError, match='guest_result_unbound'):
        execution.extract_evidence(guest, image, tmp_path / 'evidence', ['native-junit.xml'],
                                   deadline_monotonic=time.monotonic() + 60)
    assert calls == []
    assert image.read_bytes() == b'foreign replacement'


def test_vm_uses_preopened_disk_descriptor_instead_of_reopening_a_path(tmp_path):
    command = execution.qemu_command(tmp_path / 'image', phase='test', image_fd=41)
    assert command[command.index('-add-fd') + 1] == 'fd=41,set=1,opaque=blueprint-private'
    assert command[command.index('-drive') + 1] == 'file=/dev/fdset/1,format=qcow2,if=virtio,cache=none'


def test_setup_that_consumes_phase_deadline_never_launches_process(tmp_path, monkeypatch):
    (tmp_path / 'image').write_bytes(b'tiny private fixture')
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    monkeypatch.setattr(execution, 'host_preflight', lambda *args, **kw: None)
    observed = [time.monotonic()]
    deadline = observed[0] + 60
    monkeypatch.setattr(execution.time, 'monotonic', lambda: observed[0])
    original = execution.selectors.DefaultSelector
    class SlowSetup(original):
        def __enter__(self):
            observed[0] = deadline + 1
            return super().__enter__()
    monkeypatch.setattr(execution.selectors, 'DefaultSelector', SlowSetup)
    calls = []
    monkeypatch.setattr(execution.subprocess, 'Popen', lambda *args, **kw: calls.append(args))
    with pytest.raises(execution.GuestExecutionError, match='phase_deadline_expired'):
        execution.run_vm(tmp_path / 'image', tmp_path / 'serial.log', phase='test',
                         required_disk_bytes=1, deadline_monotonic=deadline)
    assert calls == []


def test_image_replacement_during_extraction_cannot_change_observed_disk(tmp_path, monkeypatch):
    guest = _receipt(tmp_path)
    source = tmp_path / 'image'
    original = source.read_bytes()
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    observed = []
    def guestfish(image, arguments, deadline, **kwargs):
        if not observed:
            source.rename(tmp_path / 'original')
            source.write_bytes(b'foreign replacement')
        if str(image).startswith(f'/proc/{execution.os.getpid()}/fd/'):
            observed.append(execution.os.pread(int(image.name), 100, 0))
        else:
            observed.append(Path(image).read_bytes())
        if arguments[0] == 'is-symlink':
            return 'false'
        if arguments[0] == 'filesize':
            return '8'
        execution.os.write(int(arguments[-1].rsplit('/', 1)[1]), b'guest00\n')
        return ''
    monkeypatch.setattr(execution, '_guestfish', guestfish)
    with pytest.raises(execution.GuestExecutionError, match='image_changed'):
        execution.extract_evidence(guest, source, tmp_path / 'evidence', ['native-junit.xml'],
                                   deadline_monotonic=time.monotonic() + 60)
    assert observed and all(raw == original for raw in observed)
    assert source.read_bytes() == b'foreign replacement'


def test_ancestor_alias_cannot_pass_destination_inode_equality(tmp_path, monkeypatch):
    guest = _receipt(tmp_path)
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    parent = tmp_path / 'parent'
    parent.mkdir()
    def guestfish(image, arguments, deadline, **kwargs):
        if arguments[0] == 'is-symlink':
            return 'false'
        if arguments[0] == 'filesize':
            return '8'
        moved = tmp_path / 'original-parent'
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
        execution.os.write(int(arguments[-1].rsplit('/', 1)[1]), b'guest00\n')
        return ''
    monkeypatch.setattr(execution, '_guestfish', guestfish)
    with pytest.raises(execution.GuestExecutionError, match='evidence_destination_changed'):
        execution.extract_evidence(guest, tmp_path / 'image', parent / 'evidence', ['native-junit.xml'],
                                   deadline_monotonic=time.monotonic() + 60)


def test_last_named_image_guard_cannot_return_after_phase_deadline(tmp_path, monkeypatch):
    image = tmp_path / 'image'
    image.write_bytes(b'tiny fixture')
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    monkeypatch.setattr(execution, 'host_preflight', lambda *args, **kw: None)
    monkeypatch.setattr(execution, 'qemu_command', lambda *args, **kw: [sys.executable, '-c', 'print("done")'])
    observed = [time.monotonic()]
    deadline = observed[0] + 60
    monkeypatch.setattr(execution.time, 'monotonic', lambda: observed[0])
    original = Path.lstat
    calls = [0]
    def lstat(path):
        value = original(path)
        if path == image:
            calls[0] += 1
            if calls[0] == 2:
                observed[0] = deadline + 1
        return value
    monkeypatch.setattr(Path, 'lstat', lstat)
    with pytest.raises(execution.GuestExecutionError, match='phase_deadline_expired'):
        execution.run_vm(image, tmp_path / 'serial.log', phase='test',
                         required_disk_bytes=1, deadline_monotonic=deadline)


@pytest.mark.parametrize('change', ['replace', 'rewrite', 'mode'])
def test_later_download_cannot_change_an_earlier_evidence_leaf(tmp_path, monkeypatch, change):
    guest = _receipt(tmp_path)
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    destination = tmp_path / 'evidence'
    first = destination / 'native-junit.xml'
    def guestfish(image, arguments, deadline, **kwargs):
        if arguments[0] == 'is-symlink':
            return 'false'
        if arguments[0] == 'filesize':
            return '8'
        if arguments[1].endswith('terminal.json'):
            if change == 'replace':
                first.rename(destination / 'original-xml')
                first.write_bytes(b'foreign\n')
            elif change == 'rewrite':
                first.write_bytes(b'foreign\n')
            else:
                first.chmod(0o644)
        execution.os.write(int(arguments[-1].rsplit('/', 1)[1]), b'guest00\n')
        return ''
    monkeypatch.setattr(execution, '_guestfish', guestfish)
    with pytest.raises(execution.GuestExecutionError, match='evidence_copy_invalid'):
        execution.extract_evidence(guest, tmp_path / 'image', destination,
                                   ['native-junit.xml', 'terminal.json'],
                                   deadline_monotonic=time.monotonic() + 60)


@pytest.mark.parametrize('change', ['ancestor_alias', 'leaf_replace'])
def test_serial_publication_refuses_replaced_namespace(tmp_path, monkeypatch, change):
    (tmp_path / 'image').write_bytes(b'tiny private fixture')
    parent = tmp_path / 'serial-parent'
    parent.mkdir()
    serial = parent / 'serial.log'
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    monkeypatch.setattr(execution, 'host_preflight', lambda *args, **kwargs: None)
    monkeypatch.setattr(execution, 'qemu_command', lambda *args, **kwargs: [sys.executable, '-c', 'print("done")'])
    original = execution.subprocess.Popen
    def popen(*args, **kwargs):
        if change == 'ancestor_alias':
            moved = tmp_path / 'owned-parent'
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        else:
            serial.rename(parent / 'original-serial')
            serial.write_bytes(b'foreign-original')
        return original(*args, **kwargs)
    monkeypatch.setattr(execution.subprocess, 'Popen', popen)
    with pytest.raises(execution.GuestExecutionError, match='serial_changed'):
        execution.run_vm(tmp_path / 'image', serial, phase='test',
                         required_disk_bytes=1, deadline_monotonic=time.monotonic() + 3)
    if change == 'leaf_replace':
        assert serial.read_bytes() == b'foreign-original'


@pytest.mark.parametrize('operation', ['tool', 'vm'])
def test_reaping_exception_still_closes_every_process_pipe(tmp_path, monkeypatch, operation):
    (tmp_path / 'image').write_bytes(b'tiny private fixture')
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    monkeypatch.setattr(execution, 'host_preflight', lambda *args, **kwargs: None)
    command = [sys.executable, '-c', 'print("done")']
    monkeypatch.setattr(execution, 'qemu_command', lambda *args, **kwargs: command)
    processes = []
    original = execution.subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(execution.subprocess, 'Popen', popen)
    def failed_reap(process):
        assert process.poll() is not None
        raise execution.GuestExecutionError('native_guest_guest_not_reaped')
    monkeypatch.setattr(execution, '_reap', failed_reap)
    with pytest.raises(execution.GuestExecutionError, match='guest_not_reaped'):
        if operation == 'tool':
            execution._bounded_command(command, deadline=time.monotonic() + 3)
        else:
            execution.run_vm(tmp_path / 'image', tmp_path / 'serial.log', phase='test',
                             required_disk_bytes=1, deadline_monotonic=time.monotonic() + 3)
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdout.closed
    assert processes[0].stderr is None or processes[0].stderr.closed


def test_evidence_is_still_bound_after_last_named_image_guard(tmp_path, monkeypatch):
    guest = _receipt(tmp_path)
    image = tmp_path / 'image'
    destination = tmp_path / 'evidence'
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    def guestfish(image, arguments, deadline, **kwargs):
        if arguments[0] == 'is-symlink':
            return 'false'
        if arguments[0] == 'filesize':
            return '8'
        execution.os.write(int(arguments[-1].rsplit('/', 1)[1]), b'guest00\n')
        return ''
    monkeypatch.setattr(execution, '_guestfish', guestfish)
    original = Path.lstat
    def lstat(path):
        result = original(path)
        if path == image and (destination / 'native-junit.xml').exists():
            (destination / 'native-junit.xml').write_bytes(b'foreign\n')
        return result
    monkeypatch.setattr(Path, 'lstat', lstat)
    with pytest.raises(execution.GuestExecutionError, match='evidence_copy_invalid'):
        execution.extract_evidence(guest, image, destination, ['native-junit.xml'],
                                   deadline_monotonic=time.monotonic() + 60)


@pytest.mark.parametrize('operation', ['serial', 'evidence'])
def test_ancestor_replacement_during_final_digest_is_refused(tmp_path, monkeypatch, operation):
    guest = _receipt(tmp_path)
    parent = tmp_path / 'parent'
    parent.mkdir()
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    monkeypatch.setattr(execution, 'host_preflight', lambda *args, **kwargs: None)
    if operation == 'serial':
        monkeypatch.setattr(execution, 'qemu_command', lambda *args, **kwargs: [sys.executable, '-c', 'print("done")'])
    def guestfish(image, arguments, deadline, **kwargs):
        if arguments[0] == 'is-symlink':
            return 'false'
        if arguments[0] == 'filesize':
            return '8'
        execution.os.write(int(arguments[-1].rsplit('/', 1)[1]), b'guest00\n')
        return ''
    monkeypatch.setattr(execution, '_guestfish', guestfish)
    original = execution.os.pread
    reads = [0]
    def pread(fd, size, offset):
        raw = original(fd, size, offset)
        reads[0] += 1
        if reads[0] == (2 if operation == 'serial' else 3):
            moved = tmp_path / 'owned-parent'
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        return raw
    monkeypatch.setattr(execution.os, 'pread', pread)
    with pytest.raises(execution.GuestExecutionError, match='serial_changed|evidence_destination_changed'):
        if operation == 'serial':
            execution.run_vm(tmp_path / 'image', parent / 'serial.log', phase='test',
                             required_disk_bytes=1, deadline_monotonic=time.monotonic() + 3)
        else:
            execution.extract_evidence(guest, tmp_path / 'image', parent / 'evidence', ['native-junit.xml'],
                                       deadline_monotonic=time.monotonic() + 60)
