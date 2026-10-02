"""Private KVM boot and stopped-disk evidence transport for Plan11 CI.

These operations transport execution evidence; they never certify a guest or
an XML result. The connected caller must validate its original commands and
immutable inputs, guest security prerequisites and complete result coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack, contextmanager
import hashlib
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


@dataclass(frozen=True)
class GuestRun:
    process: subprocess.Popen
    image_path: str
    image_identity: tuple[int, int]
    command: tuple[str, ...]
    phase: str
    image_fd: int


def _require(condition, code):
    if not condition:
        raise GuestExecutionError('native_guest_' + code)


def _image_argument(image):
    image = Path(image)
    _require(image.is_absolute() and '..' not in image.parts
             and not any(c in str(image) for c in ',\n\r\x00'),
             'image_argument_invalid')
    return image


def qemu_command(image, *, phase, image_fd=None):
    """No TCG fallback, shared mount, monitor, guest agent or test-phase NIC."""
    image = _image_argument(image)
    _require(phase in {'provision', 'test'}, 'phase_invalid')
    command = ['/usr/bin/qemu-system-x86_64', '-machine', 'q35,accel=kvm',
               '-cpu', 'host', '-smp', '2', '-m', str(GUEST_MEMORY_MIB),
               '-nodefaults', '-display', 'none', '-monitor', 'none',
               '-serial', 'stdio', '-no-reboot']
    if image_fd is not None:
        _require(type(image_fd) is int and image_fd > 2, 'image_argument_invalid')
        command.extend(['-add-fd', f'fd={image_fd},set=1,opaque=blueprint-private'])
    selected = str(image) if image_fd is None else '/dev/fdset/1'
    command.extend(['-drive', f'file={selected},format=qcow2,if=virtio,cache=none'])
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


def _deadline(deadline):
    _require(type(deadline) in {int, float} and math.isfinite(deadline), 'phase_deadline_expired')
    remaining = deadline - time.monotonic()
    _require(remaining > 0, 'phase_deadline_expired')
    return remaining


def _bounded_command(command, *, deadline, max_output_bytes=65536, pass_fds=(), stdin=None, env=None):
    """Receive tool output under its cap and reap on every exit path."""
    process = None
    buffers = [bytearray(), bytearray()]
    with selectors.DefaultSelector() as selector:
        try:
            _deadline(deadline)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL if stdin is None else stdin,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, pass_fds=pass_fds,
                env=env if env is not None else {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'HOME': '/root',
                                                 'LIBGUESTFS_BACKEND': 'direct'})
            for index, stream in enumerate((process.stdout, process.stderr)):
                selector.register(stream, selectors.EVENT_READ, index)
            while selector.get_map():
                for key, _ in selector.select(timeout=min(_deadline(deadline), 1)):
                    raw = os.read(key.fileobj.fileno(), min(65536, max_output_bytes + 1))
                    if not raw:
                        selector.unregister(key.fileobj)
                        continue
                    target = buffers[key.data]
                    _require(len(target) + len(raw) <= max_output_bytes, 'tool_output_limit')
                    target.extend(raw)
            process.wait(timeout=_deadline(deadline))
        finally:
            if process is not None:
                try:
                    _reap(process)
                finally:
                    try:
                        process.stdout.close()
                    finally:
                        process.stderr.close()
    _deadline(deadline)
    return subprocess.CompletedProcess(command, process.returncode, bytes(buffers[0]), bytes(buffers[1]))


def run_vm(image, serial_path, *, phase, deadline_monotonic, required_disk_bytes):
    """One boot under the caller's existing phase deadline; no guest polling.

    The serial pipe is observed from the host. On timeout/oversize/failure the
    VM is terminated and reaped and no result is returned for acceptance.
    """
    _deadline(deadline_monotonic)
    image = _sealed_image(image)
    info = image.lstat()
    identity = (info.st_dev, info.st_ino)
    host_preflight(image.parent, required_disk_bytes=required_disk_bytes)
    _require(time.monotonic() < deadline_monotonic, 'phase_deadline_expired')
    command = qemu_command(image, phase=phase)
    serial_path = Path(serial_path)
    _require(serial_path.is_absolute() and not serial_path.exists()
             and not any(p.is_symlink() for p in (serial_path, *serial_path.parents)), 'serial_untrusted')
    process = None
    with ExitStack() as stack:
        image_stream = stack.enter_context(os.fdopen(
            os.open(image, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC), 'r+b', buffering=0))
        ancestry = []
        parent = _open_directory(serial_path.parent, ancestry)
        stack.callback(os.close, parent)
        output = stack.enter_context(os.fdopen(os.open(serial_path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600, dir_fd=parent), 'r+b', buffering=0))
        selector = stack.enter_context(selectors.DefaultSelector())
        opened = os.fstat(image_stream.fileno())
        _require((opened.st_dev, opened.st_ino) == identity and opened.st_uid == info.st_uid
                 and opened.st_mode == info.st_mode and opened.st_nlink == 1, 'image_changed')
        image_fd = image_stream.fileno()
        command = qemu_command(image, phase=phase, image_fd=image_fd)
        os.fchmod(output.fileno(), 0o600)
        created_serial = _file_identity(os.fstat(output.fileno()))[:6]
        serial_digest = hashlib.sha256()
        try:
            _deadline(deadline_monotonic)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True, pass_fds=(image_fd,),
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
                    serial_digest.update(raw)
                    count += len(raw)
            process.wait(timeout=max(0.001, deadline_monotonic - time.monotonic()))
            _require(process.returncode == 0, 'qemu_failed')
        finally:
            if process is not None:
                try:
                    _reap(process)
                finally:
                    if process.stdout is not None:
                        process.stdout.close()
        snapshot = _file_snapshot(output.fileno(), deadline_monotonic, 'serial_changed')
        _require(snapshot[0][:6] == created_serial and snapshot[0][6] == count
                 and snapshot[1] == serial_digest.hexdigest(), 'serial_changed')
        final = image.lstat()
        _require((final.st_dev, final.st_ino) == identity, 'image_changed')
        _check_ancestry(serial_path.parent, ancestry, 'serial_changed')
        _check_leaf(output.fileno(), parent, serial_path.name, snapshot,
                    deadline_monotonic, 'serial_changed')
        _check_ancestry(serial_path.parent, ancestry, 'serial_changed')
        _deadline(deadline_monotonic)
        return GuestRun(process, str(image), identity, tuple(command), phase, image_fd)


def _guestfish(image, arguments, deadline, *, pass_fds=()):
    _deadline(deadline)
    result = _bounded_command(['/usr/bin/guestfish', '--ro', '--format=qcow2', '-a', str(image), '-i', *arguments],
                              deadline=min(deadline, time.monotonic() + 60), pass_fds=pass_fds)
    _require(result.returncode == 0, 'offline_extraction_failed')
    return result.stdout.decode('ascii').strip()


def _directory_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)


def _open_directory(path, identities=None):
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        if identities is not None:
            identities.append(_directory_identity(os.fstat(fd)))
        for name in path.parts[1:]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            os.close(fd)
            fd = child
            if identities is not None:
                identities.append(_directory_identity(os.fstat(fd)))
        return fd
    except BaseException:
        os.close(fd)
        raise


def _check_ancestry(path, identities, code):
    current = []
    try:
        fd = _open_directory(path, current)
    except OSError:
        raise GuestExecutionError('native_guest_' + code) from None
    try:
        _require(current == identities, code)
    finally:
        os.close(fd)


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_snapshot(fd, deadline, code):
    before = os.fstat(fd)
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and 0 <= before.st_size <= MAX_EVIDENCE_BYTES, code)
    digest, offset = hashlib.sha256(), 0
    while offset < before.st_size:
        _deadline(deadline)
        raw = os.pread(fd, min(1024**2, before.st_size - offset), offset)
        _require(bool(raw), code)
        offset += len(raw)
        digest.update(raw)
    _require(_file_identity(os.fstat(fd)) == _file_identity(before), code)
    _deadline(deadline)
    return _file_identity(before), digest.hexdigest()


def _check_leaf(fd, directory, name, snapshot, deadline, code):
    try:
        named = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OSError:
        raise GuestExecutionError('native_guest_' + code) from None
    _require(_file_identity(named) == snapshot[0]
             and _file_snapshot(fd, deadline, code) == snapshot, code)
    try:
        _require(_file_identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) == snapshot[0], code)
    except OSError:
        raise GuestExecutionError('native_guest_' + code) from None


def extract_evidence(guest, image, destination, names, *, deadline_monotonic):
    """Copy bounded original files only after terminal shutdown and VM reap.

    A missing file/alias/oversize refuses. No receipt or XML is fabricated, and
    successful transport cannot turn failed or partial test output into PASS.
    """
    _require(type(guest) is GuestRun, 'guest_result_unbound')
    process = guest.process
    _require(process.poll() is not None, 'guest_not_reaped')
    process.wait(timeout=1)
    _require(type(names) is list and 0 < len(names) <= len(EVIDENCE_NAMES)
             and all(type(name) is str and name in EVIDENCE_NAMES for name in names)
             and len(set(names)) == len(names), 'evidence_selection_invalid')
    _deadline(deadline_monotonic)
    image = _sealed_image(image)
    info = image.lstat()
    _require(str(image) == guest.image_path and (info.st_dev, info.st_ino) == guest.image_identity
             and tuple(process.args) == guest.command
             == tuple(qemu_command(image, phase=guest.phase, image_fd=guest.image_fd)),
             'guest_result_unbound')
    destination = Path(destination)
    _require(destination.is_absolute()
             and not any(p.is_symlink() for p in (destination, *destination.parents)),
             'evidence_destination_untrusted')
    fd = os.open(image, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        _require((opened.st_dev, opened.st_ino) == guest.image_identity, 'image_changed')
        # Parent PID remains alive with this descriptor until every helper is
        # reaped. Backend descendants therefore resolve the same held inode.
        bound = Path(f'/proc/{os.getpid()}/fd/{fd}')
        with _extract_files(bound, destination, names, deadline_monotonic) as sizes:
            final = image.lstat()
            _require((final.st_dev, final.st_ino) == guest.image_identity, 'image_changed')
            _deadline(deadline_monotonic)
            return sizes
    finally:
        os.close(fd)


@contextmanager
def _extract_files(image, destination, names, deadline_monotonic):
    # The guest is stopped and the source descriptor is retained by the caller.
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
    ancestry = []
    parent = _open_directory(destination.parent, ancestry)
    directory = None
    leaves = []
    try:
        os.mkdir(destination.name, mode=0o700, dir_fd=parent)
        directory = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=parent)
        original = os.fstat(directory)
        ancestry.append(_directory_identity(original))
        for name, size in sizes.items():
            _deadline(deadline_monotonic)
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=directory)
            leaves.append((fd, name, None))
            before = os.fstat(fd)
            _guestfish(image, ['download', EVIDENCE_ROOT + '/' + name, f'/proc/self/fd/{fd}'],
                       deadline_monotonic, pass_fds=(fd,))
            snapshot = _file_snapshot(fd, deadline_monotonic, 'evidence_copy_invalid')
            _require(snapshot[0][:5] == _file_identity(before)[:5]
                     and snapshot[0][6] == size, 'evidence_copy_invalid')
            _check_leaf(fd, directory, name, snapshot, deadline_monotonic, 'evidence_copy_invalid')
            leaves[-1] = (fd, name, snapshot)
        yield sizes
        _check_ancestry(destination, ancestry, 'evidence_destination_changed')
        for fd, name, snapshot in leaves:
            _check_leaf(fd, directory, name, snapshot, deadline_monotonic, 'evidence_copy_invalid')
        # Digest work for a later file must not invalidate an earlier leaf.
        # This last closure reads metadata only and retains every descriptor.
        for fd, name, snapshot in leaves:
            try:
                _require(_file_identity(os.fstat(fd)) == snapshot[0]
                         == _file_identity(os.stat(name, dir_fd=directory, follow_symlinks=False)),
                         'evidence_copy_invalid')
            except OSError:
                raise GuestExecutionError('native_guest_evidence_copy_invalid') from None
        _check_ancestry(destination, ancestry, 'evidence_destination_changed')
        _deadline(deadline_monotonic)
    finally:
        for fd, _, _ in leaves:
            os.close(fd)
        if directory is not None:
            os.close(directory)
        os.close(parent)
