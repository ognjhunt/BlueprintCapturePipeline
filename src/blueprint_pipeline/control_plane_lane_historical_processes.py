"""Strict current native process channels after the historical write fence.

This bounded scan cannot replace the owner decision, future-writer fence or
queue/pin/release checks. Every unreadable, changing or unknown view refuses.
The invoking worker's held descriptors are skipped only for its actual PID.
"""
from __future__ import annotations

import os
import re
import stat
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from .control_plane_kernel_process import kernel_has_no_user_memory


class HistoricalProcessError(ValueError):
    """Fixed refusal without foreign process contents."""


def _require(value, code='process_unknown'):
    if not value:
        raise HistoricalProcessError('historical_generation_' + code)


def _process_start(raw, pid):
    prefix, separator, fields = raw.rpartition(b') ')
    values = fields.split()
    _require(separator and prefix.startswith(pid.encode('ascii') + b' (')
             and len(values) >= 20 and values[19].isdigit())
    return int(values[19])


def _mapping_inodes(raw):
    result = set()
    for line in raw.splitlines():
        fields = line.split(None, 5)
        _require(len(fields) >= 5 and re.fullmatch(b'[0-9a-f]+-[0-9a-f]+', fields[0])
            and re.fullmatch(b'[r-][w-][x-][ps]', fields[1])
            and re.fullmatch(b'[0-9a-f]+', fields[2])
            and re.fullmatch(b'[0-9a-f]+:[0-9a-f]+', fields[3]) and fields[4].isdigit())
        if int(fields[4]):
            major, minor = fields[3].split(b':')
            result.add((os.makedev(int(major, 16), int(minor, 16)), int(fields[4])))
    return result


class _Scan:
    def __init__(self, tick):
        self.tick_operation = tick
        self.started = self.last = time.monotonic()
        self.entries, self.raw_bytes = 0, 0
        self.views = {}
        self.kernel_views = {}
        self.physical_target = None

    def tick(self):
        self.tick_operation()
        current = time.monotonic()
        _require(self.last <= current < self.started + 5)
        self.last = current

    def read(self, directory, name, cap=1024**2):
        self.tick()
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory)
        try:
            _require(stat.S_ISREG(os.fstat(fd).st_mode))
            data = bytearray()
            while True:
                self.tick()
                block = os.read(fd, min(4096, cap + 1 - len(data)))
                self.raw_bytes += len(block)
                _require(len(data) + len(block) <= cap and self.raw_bytes <= 20 * 1024**2)
                if not block:
                    return bytes(data)
                data.extend(block)
        finally:
            os.close(fd)

    def names(self, fd, limit):
        result = []
        with os.scandir(fd) as stream:
            for row in stream:
                self.tick()
                self.entries += 1
                _require(len(result) < limit and self.entries <= 20000)
                result.append(row.name)
        return sorted(result)


def _namespace(directory, *, kernel=False):
    values = []
    for kind in ('pid', 'user', 'mnt'):
        try:
            values.append(os.readlink('ns/' + kind, dir_fd=directory))
        except FileNotFoundError:
            _require(kernel and kind == 'mnt')
            values.append(None)
    _require(all(value is None and kernel and kind == 'mnt' or
        isinstance(value, str) and re.fullmatch(kind + r':\[[0-9]+\]', value)
        for kind, value in zip(('pid', 'user', 'mnt'), values, strict=True)))
    return tuple(values)


def _mount_path(raw):
    # The kernel escapes these four characters in mountinfo. Unrecognized or
    # noncanonical spellings remain unknown, rather than inventing an alias.
    raw = re.sub(rb'\\(040|011|012|134)', lambda match: bytes([int(match[1], 8)]), raw)
    _require(0 < len(raw) <= 4096 and b'\\' not in raw, 'process_view_unknown')
    text = os.fsdecode(raw)
    path = Path(text)
    _require(path.is_absolute() and str(path) == text and '..' not in path.parts
        and len(path.parts) <= 32 and all(32 <= ord(char) < 127 or 127 < ord(char) < 0xD800
            or 0xDFFF < ord(char) for char in text), 'process_view_unknown')
    return path


def _mount_rows(raw, tick=lambda: None):
    _require(isinstance(raw, bytes) and 0 < len(raw) <= 1024**2, 'process_view_unknown')
    result, identities = [], set()
    for line in raw.splitlines():
        tick()
        _require(len(result) < 4096, 'process_view_unknown')
        left, separator, right = line.partition(b' - ')
        fields, tail = left.split(), right.split()
        _require(separator and len(fields) >= 6 and len(tail) == 3
            and fields[0].isdigit() and fields[1].isdigit() and fields[0] not in identities
            and re.fullmatch(rb'[0-9]{1,10}:[0-9]{1,10}', fields[2]), 'process_view_unknown')
        identities.add(fields[0])
        major, minor = fields[2].split(b':')
        result.append((os.makedev(int(major), int(minor)), _mount_path(fields[3]),
                       _mount_path(fields[4])))
    _require(result, 'process_view_unknown')
    return result


def _physical_target(raw, target, device, *, tick=lambda: None):
    candidates = [row for row in _mount_rows(raw, tick) if target.is_relative_to(row[2])]
    _require(candidates, 'process_view_unknown')
    selected = max(candidates, key=lambda row: len(row[2].parts))
    _require(selected[0] == device, 'process_view_unknown')
    return selected[1] / target.relative_to(selected[2])


def _view_routes(raw, physical, device, *, tick=lambda: None):
    routes = []
    for current, root, point in _mount_rows(raw, tick):
        if current != device:
            continue
        # The first scope excludes mounts rooted at or inside the selected
        # generation. Ancestor mounts are known only after the actual target
        # inode and root-owned rights are observed through EVERY derived route.
        _require(not root.is_relative_to(physical), 'process_view_unknown')
        if physical.is_relative_to(root):
            route = point / physical.relative_to(root)
            _require(len(routes) < 16 and len(os.fsencode(route)) <= 4096, 'process_view_unknown')
            routes.append(route)
    _require(routes, 'process_view_unknown')
    return tuple(dict.fromkeys(routes))


def _kernel_view_disjoint(raw, devices, *, tick=lambda: None):
    """A complete isolated kernel filesystem view must exclude every member device."""
    _require(devices, 'process_view_unknown')
    return all(device not in devices for device, _, _ in _mount_rows(raw, tick))


def _isolated_kernel_view(scan, directory, view, identities, observed_root):
    raw = scan.read(directory, 'mountinfo')
    key = (view, observed_root)
    _require(len(scan.views) + len(scan.kernel_views) < 16 or key in scan.kernel_views,
             'process_view_unknown')
    _require(_kernel_view_disjoint(raw, {device for device, _ in identities}, tick=scan.tick),
             'process_view_unknown')
    if key in scan.kernel_views:
        _require(scan.kernel_views[key] == raw, 'process_view_unknown')
    scan.kernel_views[key] = raw
    _require(scan.read(directory, 'mountinfo') == raw, 'process_view_unknown')
    return raw


def _known_filesystem_view(scan, directory, view, target, identities, root_identity):
    """Authenticate actual mount routes and rights, not a namespace-name waiver.

    A different service namespace may expose the same protected generation.
    Full current mount bytes derive every ancestor alias. Missing/hidden routes,
    subtree binds, remapped owner rights, links or changed mount bytes refuse.
    Current FD/cwd/maps and start/namespace checks still run after this proof.
    """
    target = Path(target)
    current = os.stat(target, follow_symlinks=False)
    _require(stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) in identities
        and current.st_uid == current.st_gid == 0 and stat.S_IMODE(current.st_mode) == 0o700,
        'process_view_unknown')
    if scan.physical_target is None:
        own = os.open('/proc/' + str(os.getpid()), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            raw = scan.read(own, 'mountinfo')
            physical = _physical_target(raw, target, current.st_dev, tick=scan.tick)
            _require(scan.read(own, 'mountinfo') == raw, 'process_view_unknown')
            scan.physical_target = physical
        finally:
            os.close(own)
    raw = scan.read(directory, 'mountinfo')
    if view not in scan.views:
        _require(len(scan.views) < 16, 'process_view_unknown')
        scan.views[view] = (raw, _view_routes(raw, scan.physical_target, current.st_dev, tick=scan.tick))
    _require(scan.views[view][0] == raw, 'process_view_unknown')
    for route in scan.views[view][1]:
        with ExitStack() as stack:
            root = os.open('root', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=directory)
            stack.callback(os.close, root)
            info = os.fstat(root)
            _require((info.st_dev, info.st_ino) == root_identity, 'process_view_unknown')
            parent = root
            for part in route.parts[1:]:
                scan.tick()
                named = os.stat(part, dir_fd=parent, follow_symlinks=False)
                descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     dir_fd=parent)
                stack.callback(os.close, descriptor)
                opened = os.fstat(descriptor)
                _require((opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid, opened.st_gid)
                    == (named.st_dev, named.st_ino, named.st_mode, named.st_uid, named.st_gid),
                    'process_view_unknown')
                parent = descriptor
            info = os.fstat(parent)
            _require((info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)
                == (current.st_dev, current.st_ino, current.st_mode, 0, 0), 'process_view_unknown')
    _require(scan.read(directory, 'mountinfo') == raw, 'process_view_unknown')


def _inspect_process(scan, directory, pid, target, identities, namespaces, host_mount, root_identity):
    """Private parser seam; native acceptance uses an actual foreign UID PID."""
    started = _process_start(scan.read(directory, 'stat', 16384), pid)
    kernel = kernel_has_no_user_memory(lambda name, cap: scan.read(directory, name, cap), pid)
    view = _namespace(directory, kernel=kernel)
    _require(view[:2] == namespaces[:2])
    observed_root = None
    try:
        info = os.stat('root', dir_fd=directory)
    except FileNotFoundError:
        _require(kernel)
    else:
        observed_root = (info.st_dev, info.st_ino)
        _require(kernel or observed_root == root_identity)
    isolated = kernel and (view[2] not in (namespaces[2], host_mount, None)
                          or observed_root not in (root_identity, None))
    kernel_mounts = None
    if isolated:
        kernel_mounts = _isolated_kernel_view(scan, directory, view[2], identities, observed_root)
    if not kernel:
        _known_filesystem_view(scan, directory, view[2], target, identities, root_identity)
    channels = set()
    for name in ('cwd', 'root'):
        scan.tick()
        try:
            info = os.stat(name, dir_fd=directory)
            path = os.readlink(name, dir_fd=directory)
        except FileNotFoundError:
            _require(kernel)
            continue
        if (info.st_dev, info.st_ino) in identities or os.fsencode(target) in os.fsencode(path):
            channels.add(name)
    for name in ('cmdline', 'environ', 'maps'):
        try:
            raw = scan.read(directory, name)
        except ProcessLookupError:
            _require(name in ('environ', 'maps') and kernel
                and kernel_has_no_user_memory(lambda name, cap: scan.read(directory, name, cap), pid))
            raw = b''
        if os.fsencode(target) in raw:
            channels.add(name)
        if name == 'maps' and _mapping_inodes(raw).intersection(identities):
            channels.add(name)
    descriptors = os.open('fd', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                          dir_fd=directory)
    try:
        names = scan.names(descriptors, 16384)
        for name in names:
            scan.tick()
            _require(re.fullmatch('[0-9]+', name))
            info = os.stat(name, dir_fd=descriptors)
            path = os.readlink(name, dir_fd=descriptors)
            if (info.st_dev, info.st_ino) in identities or os.fsencode(target) in os.fsencode(path):
                channels.add('fd')
        _require(scan.names(descriptors, 16384) == names)
    finally:
        os.close(descriptors)
    _require(_process_start(scan.read(directory, 'stat', 16384), pid) == started
             and _namespace(directory, kernel=kernel) == view
             and (not kernel or kernel_has_no_user_memory(
                 lambda name, cap: scan.read(directory, name, cap), pid)))
    if not kernel:
        _require(scan.read(directory, 'mountinfo') == scan.views[view[2]][0], 'process_view_unknown')
        current_root = os.stat('root', dir_fd=directory)
        _require((current_root.st_dev, current_root.st_ino) == root_identity, 'process_view_unknown')
    elif isolated:
        _require(scan.read(directory, 'mountinfo') == kernel_mounts, 'process_view_unknown')
        try:
            current_root = os.stat('root', dir_fd=directory)
        except FileNotFoundError:
            _require(observed_root is None, 'process_view_unknown')
        else:
            _require((current_root.st_dev, current_root.st_ino) == observed_root,
                     'process_view_unknown')
    return channels


def refuse_historical_process_references(manifest, *, tick, restore_bounds=None):
    """Fixed real /proc, same PID/user namespace, finite complete current scan."""
    _require(sys.platform == 'linux' and os.geteuid() == 0, 'native_unavailable')
    scan = _Scan(tick)
    if restore_bounds is not None:
        from .control_plane_lane_historical_fence import _members
        _members(manifest, restore_bounds=restore_bounds)
    identities = {(row['version'][0], row['version'][1]) for row in manifest['members']}
    _require(0 < len(identities) <= (restore_bounds.member_count if restore_bounds is not None else 4096))
    target = manifest['target_path']
    proc = os.open('/proc', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        own = os.open(str(os.getpid()), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                      dir_fd=proc)
        host = os.open('1', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=proc)
        try:
            namespaces, host_namespace = _namespace(own), _namespace(host)
            _require(namespaces[:2] == host_namespace[:2])
            own_root, host_root = os.stat('root', dir_fd=own), os.stat('root', dir_fd=host)
            root_identity = own_root.st_dev, own_root.st_ino
            _require(root_identity == (host_root.st_dev, host_root.st_ino))
        finally:
            os.close(host)
            os.close(own)
        names = [name for name in scan.names(proc, 10000) if name.isdigit()]
        _require(len(names) <= 4096)
        for pid in names:
            if int(pid) == os.getpid():
                continue
            scan.tick()
            directory = os.open(pid, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=proc)
            try:
                channels = _inspect_process(scan, directory, pid, target, identities, namespaces,
                                            host_namespace[2], root_identity)
                _require(not channels, 'process_reference')
            finally:
                os.close(directory)
        after = [name for name in scan.names(proc, 10000) if name.isdigit()]
        _require(after == names)
        scan.tick()
    except HistoricalProcessError:
        raise
    except (OSError, ValueError, OverflowError):
        raise HistoricalProcessError('historical_generation_process_unknown') from None
    finally:
        os.close(proc)
