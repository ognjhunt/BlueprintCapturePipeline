"""Private KVM boot and stopped-disk evidence transport for Plan11 CI.

These operations transport execution evidence; they never certify a guest or
an XML result. The connected caller must validate its original commands and
immutable inputs, guest security prerequisites and complete result coverage.
"""
from __future__ import annotations

import os
import math
from pathlib import Path
import re
import selectors
import stat
import subprocess
import time

from scripts.native_linux_guest import GUEST_MEMORY_MIB, host_preflight

MAX_SERIAL_BYTES = 64 * 1024**2
MAX_EVIDENCE_BYTES = 64 * 1024**2
EVIDENCE_ROOT = '/var/lib/blueprint-ci/evidence'
EVIDENCE_NAMES = frozenset({
    'native-junit.xml', 'terminal.json', 'inputs.json', 'dependencies.json',
    'full-test-lane-collection.txt', 'full-test-lane-planned.json',
    'full-test-lane-duration-baseline.json', 'full-test-lane-shard-plan.json',
    'full-test-lane-shard-executed.json', 'full-test-lane-shard-junit.xml',
    'full-test-lane-shard-verification.json',
})


class GuestExecutionError(ValueError):
    """A failed guest lifecycle or transport must never imply acceptance."""


def _require(condition, code):
    if not condition:
        raise GuestExecutionError('native_guest_' + code)


def _image_argument(image):
    image = Path(image)
    _require(image.is_absolute() and not any(c in str(image) for c in ',\n\r\x00'),
             'image_argument_invalid')
    return image


def qemu_command(image, *, phase):
    """No TCG fallback, shared mount, monitor, guest agent or test-phase NIC."""
    image = _image_argument(image)
    _require(phase in {'provision', 'test'}, 'phase_invalid')
    command = ['/usr/bin/qemu-system-x86_64', '-machine', 'q35,accel=kvm',
               '-cpu', 'host', '-smp', '2', '-m', str(GUEST_MEMORY_MIB),
               '-nodefaults', '-display', 'none', '-monitor', 'none',
               '-serial', 'stdio', '-no-reboot',
               '-drive', f'file={image},format=qcow2,if=virtio,cache=none']
    if phase == 'provision':
        command.extend(['-netdev', 'user,id=provision', '-device', 'virtio-net-pci,netdev=provision'])
    else:
        command.extend(['-nic', 'none'])
    return command


def _sealed_image(image):
    image = _image_argument(image)
    _require(not any(p.is_symlink() for p in (image, *image.parents)), 'image_untrusted')
    info = image.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == 0
             and not info.st_mode & 0o077, 'image_untrusted')
    return image


def _reap(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=5)
    _require(process.poll() is not None, 'guest_not_reaped')


def run_vm(image, serial_path, *, phase, deadline_monotonic, required_disk_bytes):
    """One boot under the caller's existing phase deadline; no guest polling.

    The serial pipe is observed from the host. On timeout/oversize/failure the
    VM is terminated and reaped and no result is returned for acceptance.
    """
    _require(type(deadline_monotonic) in {int, float}
             and math.isfinite(deadline_monotonic)
             and time.monotonic() < deadline_monotonic, 'phase_deadline_expired')
    image = _sealed_image(image)
    host_preflight(image.parent, required_disk_bytes=required_disk_bytes)
    _require(time.monotonic() < deadline_monotonic, 'phase_deadline_expired')
    command = qemu_command(image, phase=phase)
    serial_path = Path(serial_path)
    _require(serial_path.is_absolute() and not serial_path.exists()
             and not any(p.is_symlink() for p in (serial_path, *serial_path.parents)), 'serial_untrusted')
    process = None
    with serial_path.open('xb') as output, selectors.DefaultSelector() as selector:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True,
                env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'HOME': '/root'})
            selector.register(process.stdout, selectors.EVENT_READ)
            count = 0
            eof = False
            while not eof or process.poll() is None:
                remaining = deadline_monotonic - time.monotonic()
                _require(remaining > 0, 'phase_deadline_expired')
                for key, _ in selector.select(timeout=min(remaining, 1)):
                    raw = os.read(key.fileobj.fileno(), 65536)
                    if not raw:
                        selector.unregister(key.fileobj)
                        eof = True
                        continue
                    _require(count + len(raw) <= MAX_SERIAL_BYTES, 'serial_limit')
                    output.write(raw)
                    count += len(raw)
            process.wait(timeout=max(0.001, deadline_monotonic - time.monotonic()))
            _require(process.returncode == 0, 'qemu_failed')
            return process
        finally:
            if process is not None:
                _reap(process)
                if process.stdout is not None:
                    process.stdout.close()


def _guestfish(image, arguments, deadline):
    remaining = deadline - time.monotonic()
    _require(remaining > 0, 'phase_deadline_expired')
    result = subprocess.run(['/usr/bin/guestfish', '--ro', '-a', str(image), '-i', *arguments],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=min(60, remaining),
        env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'HOME': '/root',
             'LIBGUESTFS_BACKEND': 'direct'}, check=False)
    _require(result.returncode == 0 and len(result.stdout.encode()) <= 65536
             and len(result.stderr.encode()) <= 65536, 'offline_extraction_failed')
    return result.stdout.strip()


def extract_evidence(process, image, destination, names, *, deadline_monotonic):
    """Copy bounded original files only after terminal shutdown and VM reap.

    A missing file/alias/oversize refuses. No receipt or XML is fabricated, and
    successful transport cannot turn failed or partial test output into PASS.
    """
    _require(process.poll() is not None, 'guest_not_reaped')
    process.wait(timeout=1)
    _require(type(names) is list and 0 < len(names) <= len(EVIDENCE_NAMES)
             and all(type(name) is str and name in EVIDENCE_NAMES for name in names)
             and len(set(names)) == len(names), 'evidence_selection_invalid')
    _require(type(deadline_monotonic) in {int, float} and math.isfinite(deadline_monotonic)
             and time.monotonic() < deadline_monotonic, 'phase_deadline_expired')
    image = _sealed_image(image)
    destination = Path(destination)
    _require(destination.is_absolute()
             and not any(p.is_symlink() for p in (destination, *destination.parents)),
             'evidence_destination_untrusted')
    destination.mkdir(mode=0o700, exist_ok=False)
    # The stopped private disk cannot change between no-follow checks and copy.
    for parent in ('/var', '/var/lib', '/var/lib/blueprint-ci', EVIDENCE_ROOT):
        _require(_guestfish(image, ['is-symlink', parent], deadline_monotonic) == 'false', 'evidence_alias')
    sizes = {}
    for name in names:
        path = EVIDENCE_ROOT + '/' + name
        _require(_guestfish(image, ['is-symlink', path], deadline_monotonic) == 'false', 'evidence_alias')
        value = _guestfish(image, ['filesize', path], deadline_monotonic)
        _require(re.fullmatch(r'[0-9]{1,10}', value) is not None, 'evidence_size')
        size = int(value)
        _require(0 < size <= MAX_EVIDENCE_BYTES and sum(sizes.values()) + size <= MAX_EVIDENCE_BYTES,
                 'evidence_size')
        sizes[name] = size
    for name, size in sizes.items():
        target = destination / name
        _guestfish(image, ['download', EVIDENCE_ROOT + '/' + name, str(target)], deadline_monotonic)
        info = target.lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and info.st_size == size, 'evidence_copy_invalid')
    return sizes
