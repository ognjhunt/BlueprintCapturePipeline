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


def _namespace(directory):
    values = tuple(os.readlink('ns/' + kind, dir_fd=directory) for kind in ('pid', 'user', 'mnt'))
    _require(all(re.fullmatch(kind + r':\[[0-9]+\]', value)
        for kind, value in zip(('pid', 'user', 'mnt'), values, strict=True)))
    return values


def _inspect_process(scan, directory, pid, target, identities, namespaces, host_mount, root_identity):
    """Private parser seam; native acceptance uses an actual foreign UID PID."""
    started = _process_start(scan.read(directory, 'stat', 16384), pid)
    view = _namespace(directory)
    _require(view[:2] == namespaces[:2] and view[2] in (namespaces[2], host_mount))
    info = os.stat('root', dir_fd=directory)
    _require((info.st_dev, info.st_ino) == root_identity)
    channels = set()
    for name in ('cwd', 'root'):
        scan.tick()
        info = os.stat(name, dir_fd=directory)
        path = os.readlink(name, dir_fd=directory)
        if (info.st_dev, info.st_ino) in identities or os.fsencode(target) in os.fsencode(path):
            channels.add(name)
    for name in ('cmdline', 'environ', 'maps'):
        raw = scan.read(directory, name)
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
             and _namespace(directory) == view)
    return channels


def refuse_historical_process_references(manifest, *, tick):
    """Fixed real /proc, same PID/user namespace, finite complete current scan."""
    _require(sys.platform == 'linux' and os.geteuid() == 0, 'native_unavailable')
    scan = _Scan(tick)
    identities = {(row['version'][0], row['version'][1]) for row in manifest['members']}
    _require(0 < len(identities) <= 4096)
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
        _require([name for name in scan.names(proc, 10000) if name.isdigit()] == names)
        scan.tick()
    except HistoricalProcessError:
        raise
    except (OSError, ValueError, OverflowError):
        raise HistoricalProcessError('historical_generation_process_unknown') from None
    finally:
        os.close(proc)
