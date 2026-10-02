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
    process = SimpleNamespace(poll=lambda: None)
    called = []
    monkeypatch.setattr(execution.subprocess, 'run', lambda *a, **kw: called.append(a))
    with pytest.raises(execution.GuestExecutionError, match='guest_not_reaped'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence', ['native-junit.xml'], deadline_monotonic=time.monotonic() + 60)
    assert not called


@pytest.mark.parametrize('names', [['../etc/shadow'], ['a/b'], ['native-junit.xml', 'native-junit.xml'], []])
def test_unbounded_or_unapproved_evidence_selector_refuses(tmp_path, names, monkeypatch):
    process = SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)
    called = []
    monkeypatch.setattr(execution.subprocess, 'run', lambda *a, **kw: called.append(a))
    with pytest.raises(execution.GuestExecutionError, match='evidence_selection_invalid'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence', names, deadline_monotonic=time.monotonic() + 60)
    assert not called


def test_guest_evidence_alias_is_refused_without_download(tmp_path, monkeypatch):
    process = SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)
    calls = []
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, 'true\n', '')
    monkeypatch.setattr(execution.subprocess, 'run', run)
    with pytest.raises(execution.GuestExecutionError, match='evidence_alias'):
        execution.extract_evidence(process, tmp_path / 'image', tmp_path / 'evidence', ['native-junit.xml'], deadline_monotonic=time.monotonic() + 60)
    assert all('download' not in call for call in calls)


def test_guest_evidence_size_checked_before_download(tmp_path, monkeypatch):
    process = SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)
    calls = []
    monkeypatch.setattr(execution, '_sealed_image', lambda path: path)
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0,
            'false\n' if 'is-symlink' in command else str(execution.MAX_EVIDENCE_BYTES + 1) + '\n', '')
    monkeypatch.setattr(execution.subprocess, 'run', run)
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
    processes = []
    original = execution.subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(execution.subprocess, 'Popen', popen)
    arguments = dict(phase='test', deadline_monotonic=time.monotonic() + 0.5, required_disk_bytes=1)
    if code:
        with pytest.raises(execution.GuestExecutionError, match=code):
            execution.run_vm(tmp_path / 'image', tmp_path / 'serial.log', **arguments)
    else:
        result = execution.run_vm(tmp_path / 'image', tmp_path / 'serial.log', **arguments)
        assert result is processes[0]
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
