"""ADP-009D/day28: one-link immutable Linux historical metadata publication.

An anonymous inode belongs to this invocation's direct creation syscall. Its
actual proc observation precedes its first descriptor observation. No transferred
FD, caller-selected source pathname, named temporary or replacement is accepted.
Unsupported kernels/filesystems refuse; actual installed-unit proof is separate.
"""
from __future__ import annotations

import os
import re
import stat
import sys

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_publication import _CAPS, _NAMES
from .control_plane_lane_historical_descriptor_birth import CreationProcView
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _require

_KINDS = frozenset({'private', 'manifest', 'event', 'historical_restore_snapshot'})


def _create(files, parent, view):
    _require(sys.platform == 'linux' and hasattr(os, 'O_TMPFILE'),
             'historical_publication_anonymous_unavailable')
    files.location(parent)
    files.slot()
    parent_info = os.fstat(parent)
    before = view.slots()
    try:
        fd = os.open('.', os.O_TMPFILE | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=parent)
    except OSError:
        raise OwnerTargetVersionError('historical_publication_anonymous_unavailable') from None
    if fd in files.owned or fd in files.probe_owned or fd in before:
        files.unresolved += 1
        raise OwnerTargetVersionError('owner_target_descriptor_collision')
    try:
        # Fixed kernel route to the inode just created, never a user source path.
        # Do not fstat this new fd before the independent proc observation.
        named = view.stat(fd)
        _require(stat.S_ISREG(named.st_mode) and named.st_uid == named.st_gid == 0
                 and stat.S_IMODE(named.st_mode) == 0o600 and named.st_nlink == 0
                 and named.st_size == 0 and named.st_dev == parent_info.st_dev,
                 'owner_target_descriptor_ownership_unproven')
    except (OSError, OwnerTargetVersionError):
        files.unresolved += 1
        # Unknown numeric ownership is never closed or adopted.
        raise OwnerTargetVersionError('owner_target_descriptor_ownership_unproven') from None
    actual = files._adopt_named(fd, named, files.owned)
    files.acquired[fd] = actual
    _require(owners._metadata(actual) == owners._metadata(named),
             'owner_target_descriptor_ownership_unproven')
    return fd


def _guard(files, parent, fd, name, size, linked, view):
    files.budget.tick()
    files.location(parent)
    files.proof(fd)
    opened = os.fstat(fd)
    _require(stat.S_ISREG(opened.st_mode) and opened.st_uid == opened.st_gid == 0
             and stat.S_IMODE(opened.st_mode) == 0o600 and opened.st_size == size
             and opened.st_nlink == int(linked), 'historical_publication_changed')
    proc = view.stat(fd)
    _require(owners._metadata(proc) == owners._metadata(opened), 'historical_publication_changed')
    if linked:
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _require(owners._metadata(named) == owners._metadata(opened), 'historical_publication_changed')
    else:
        try:
            os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OwnerTargetVersionError('experiment_publication_destination_exists')


def _publish(files, parent, name, payload, *, kind):
    _require(kind in _KINDS and isinstance(name, str) and re.fullmatch(_NAMES[kind], name)
             and isinstance(payload, bytes) and 0 < len(payload) <= _CAPS[kind],
             'experiment_publication_invalid')
    files.budget.charge('output_bytes', len(payload))
    files.location(parent)
    owners._protected(os.fstat(parent), directory=True, mode=0o700)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise OwnerTargetVersionError('experiment_publication_destination_exists')
    # Two comparisons here and one protected caller readback must be admitted
    # before creating/linking any inode, with original counters conserved.
    _require(3 * len(payload) <= min(files.raw_cap, files.budget.limits['raw_bytes'])
             - files.budget.counts['raw_bytes'], 'owner_target_resource_exhausted')
    view = CreationProcView(files)
    fd = None
    size, linked = 0, False
    try:
        fd = _create(files, parent, view)
        while size < len(payload):
            _guard(files, parent, fd, name, size, linked, view)
            count = os.write(fd, memoryview(payload)[size:])
            _require(type(count) is int and 0 < count <= len(payload) - size,
                     'historical_publication_changed')
            size += count
        _guard(files, parent, fd, name, size, linked, view)
        os.fsync(fd)
        _guard(files, parent, fd, name, size, linked, view)
        os.lseek(fd, 0, os.SEEK_SET)
        _require(files.read_bytes(fd, _CAPS[kind]) == payload, 'historical_publication_changed')
        _guard(files, parent, fd, name, size, linked, view)
        # linkat follows only this fixed proc alias of our retained creation.
        # Destination creation is atomic and no-replace; links jump from 0 to 1.
        os.link(str(fd), name, src_dir_fd=view.fd, dst_dir_fd=parent, follow_symlinks=True)
        linked = True
        _guard(files, parent, fd, name, size, linked, view)
        os.fsync(parent)
        _guard(files, parent, fd, name, size, linked, view)
        os.lseek(fd, 0, os.SEEK_SET)
        _require(files.read_bytes(fd, _CAPS[kind]) == payload, 'historical_publication_changed')
        _guard(files, parent, fd, name, size, linked, view)
        return dict(sha256=retained._digest(payload, _work_budget=files.budget), size_bytes=size)
    finally:
        # Before link, the kernel reclaims this unnamed inode on known close.
        # After link, retain the complete record even if parent fsync was refused.
        if fd is not None:
            files.close(fd)
        view.close()
        _require(fd not in files.owned and not files.unresolved,
                 'owner_target_descriptor_cleanup_failed')
