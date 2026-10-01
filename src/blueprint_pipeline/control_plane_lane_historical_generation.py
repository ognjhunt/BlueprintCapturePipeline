"""Bound original historical bytes for a distinct Plan 12h owner decision.

This observer neither adopts a registered birth nor grants execution. Installed
rights, decommission, current readers and action fencing remain separate gates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from contextlib import ExitStack
from pathlib import Path

from . import control_plane_lane_legacy_owner as legacy
from .decision_evidence_contracts import canonical_digest

MAX_MEMBERS = 4096
MAX_PAYLOAD_BYTES = 8 * 1024**3
MAX_MANIFEST_BYTES = 1024**2
MAX_SECONDS = 4 * 3600


class HistoricalGenerationError(ValueError):
    """Typed refusal without historical path or payload contents."""


def _require(value, code):
    if not value:
        raise HistoricalGenerationError('historical_generation_' + code)


def _verify_chain(chain):
    """Retain every named ancestor; exact namespace versions only in scope.

    A sibling outside the configured parent does not change this generation.
    Its ancestor must still name the original inode with unchanged rights.
    The selected parent and target retain their full original versions.
    """
    for index, (parent, name, fd, expected) in enumerate(chain):
        opened = legacy._version(os.fstat(fd))
        named = legacy._version(os.stat(name, dir_fd=parent, follow_symlinks=False))
        if index < len(chain) - 2:
            _require(opened[:5] == named[:5] == expected[:5], 'changed')
        else:
            _require(opened == named == expected, 'changed')


def _chain(path, stack):
    components = ('/', *path.parts[1:])
    result, parent = [], None
    for index, name in enumerate(components):
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _require(stat.S_ISDIR(named.st_mode), 'changed')
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=parent)
        stack.callback(os.close, fd)
        expected, opened = legacy._version(named), legacy._version(os.fstat(fd))
        _require(opened == expected if index >= len(components) - 2
                 else opened[:5] == expected[:5], 'changed')
        result.append((parent, name, fd, opened))
        parent = fd
    return result


def verify_historical_member_versions(manifest, *, tick, _restore_bounds=None):
    """Recheck the entire original namespace without rereading payload bytes.

    This closes metadata drift since hashing; it does not fence future writers
    or confer mutation authority. The caller supplies its original deadline.
    """
    rows = manifest['members']
    if _restore_bounds is not None:
        from .control_plane_lane_historical_restore_limits import RestoreObservationBounds
        _require(type(_restore_bounds) is RestoreObservationBounds, 'manifest_invalid')
    limit = _restore_bounds.member_count if _restore_bounds is not None else MAX_MEMBERS
    _require(_restore_bounds is None or manifest['target_path'] == _restore_bounds.target_path, 'manifest_invalid')
    _require(0 < len(rows) == manifest['member_count'] <= limit, 'manifest_invalid')
    expected, children = {}, {}
    for row in rows:
        tick()
        relative = row['path']
        _require(isinstance(relative, str) and len(os.fsencode(relative)) <= (1076 if _restore_bounds else 1024)
                 and (relative == '' or all(part not in ('', '.', '..')
                                           for part in relative.split('/')))
                 and len(relative.split('/')) <= (18 if _restore_bounds else 17) and relative not in expected
                 and row['kind'] in ('directory', 'file'), 'manifest_invalid')
        _require(_restore_bounds is None or _restore_bounds.accepts(relative, row['kind']), 'manifest_invalid')
        expected[relative] = row
        if relative:
            parent, _, name = relative.rpartition('/')
            children.setdefault(parent, []).append(name)
    _require('' in expected and expected['']['kind'] == 'directory'
             and all(parent in expected and expected[parent]['kind'] == 'directory'
                     for parent in children), 'manifest_invalid')

    def same(info, relative):
        tick()
        row = expected[relative]
        _require(list(legacy._version(info)) == row['version']
                 and info.st_dev == expected['']['version'][0]
                 and (stat.S_ISDIR(info.st_mode) if row['kind'] == 'directory'
                      else stat.S_ISREG(info.st_mode) and info.st_nlink == 1), 'changed')

    def names(fd):
        found = []
        with os.scandir(fd) as stream:
            for row in stream:
                tick()
                _require(len(found) < limit, 'changed')
                found.append(row.name)
        return sorted(found)

    def walk(fd, relative):
        same(os.fstat(fd), relative)
        selected = sorted(children.get(relative, []))
        _require(names(fd) == selected, 'changed')
        for name in selected:
            child_path = relative + '/' + name if relative else name
            same(os.stat(name, dir_fd=fd, follow_symlinks=False), child_path)
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            if expected[child_path]['kind'] == 'directory':
                flags |= os.O_DIRECTORY
            child = os.open(name, flags, dir_fd=fd)
            try:
                same(os.fstat(child), child_path)
                if expected[child_path]['kind'] == 'directory':
                    walk(child, child_path)
                same(os.fstat(child), child_path)
                same(os.stat(name, dir_fd=fd, follow_symlinks=False), child_path)
            finally:
                os.close(child)
        _require(names(fd) == selected, 'changed')
        for name in selected:
            same(os.stat(name, dir_fd=fd, follow_symlinks=False),
                 relative + '/' + name if relative else name)
        same(os.fstat(fd), relative)

    try:
        with ExitStack() as stack:
            target = Path(manifest['target_path'])
            legacy._absolute(target)
            chain = _chain(target, stack)
            _require(list(legacy._version(os.fstat(chain[-2][2]))) == manifest['root_version'], 'changed')
            walk(chain[-1][2], '')
            _verify_chain(chain)
            tick()
    except (OSError, legacy.LegacyOwnerError):
        raise HistoricalGenerationError('historical_generation_changed') from None


def inventory_historical_generation(path, *, allowed_roots, max_members=MAX_MEMBERS,
                                    max_payload_bytes=MAX_PAYLOAD_BYTES,
                                    max_seconds=MAX_SECONDS, monotonic=time.monotonic, _restore_bounds=None):
    """One direct configured child, retained no-follow identities and exact hashes."""
    _require(isinstance(path, (str, Path)) and isinstance(allowed_roots, (tuple, list))
             and 1 <= len(allowed_roots) <= 2, 'scope_invalid')
    target = Path(path)
    roots = tuple(Path(root) for root in allowed_roots)
    _require(len(set(roots)) == len(roots) and sum(target.parent == root for root in roots) == 1,
             'scope_invalid')
    try:
        legacy._absolute(target)
        for root in roots:
            legacy._absolute(root)
    except legacy.LegacyOwnerError:
        raise HistoricalGenerationError('historical_generation_scope_invalid') from None
    _require(type(max_members) is int and 0 < max_members <= MAX_MEMBERS
             and type(max_payload_bytes) is int and 0 < max_payload_bytes <= MAX_PAYLOAD_BYTES
             and type(max_seconds) in (int, float) and math.isfinite(max_seconds)
             and 0 < max_seconds <= MAX_SECONDS, 'options_invalid')
    encoded_limit, path_limit, depth_limit = MAX_MANIFEST_BYTES, 1024, 16
    if _restore_bounds is not None:
        from .control_plane_lane_historical_restore_limits import RestoreObservationBounds
        _require(type(_restore_bounds) is RestoreObservationBounds
            and str(target) == _restore_bounds.target_path, 'options_invalid')
        max_members = (_restore_bounds.member_count if max_members == MAX_MEMBERS
                       else min(max_members, _restore_bounds.member_count))
        max_payload_bytes = min(max_payload_bytes, _restore_bounds.payload_bytes)
        encoded_limit, path_limit, depth_limit = _restore_bounds.encoded_bytes, 1076, 17
    started = monotonic()
    _require(type(started) in (int, float) and math.isfinite(started), 'deadline')
    last = started
    members, logical, allocated, retained_bytes = [], 0, 0, 65536

    def tick():
        nonlocal last
        current = monotonic()
        _require(type(current) in (int, float) and math.isfinite(current)
                 and last <= current < started + max_seconds, 'deadline')
        last = current

    def names(fd):
        observed = []
        with os.scandir(fd) as stream:
            for row in stream:
                tick()
                _require(len(observed) < max_members, 'limit')
                _require(row.name not in ('', '.', '..') and len(os.fsencode(row.name)) <= 255,
                         'member_unsupported')
                observed.append(row.name)
        return sorted(observed)

    def add(relative, kind, info, digest):
        nonlocal retained_bytes, allocated
        tick()
        _require(len(members) < max_members and len(os.fsencode(relative)) <= path_limit, 'limit')
        _require(_restore_bounds is None or _restore_bounds.accepts(relative, kind), 'restore_snapshot_changed')
        row = dict(path=relative, kind=kind, version=list(legacy._version(info)),
                   size_bytes=info.st_size if kind == 'file' else 0, sha256=digest)
        try:
            encoded = json.dumps(row, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        except (UnicodeError, ValueError):
            raise HistoricalGenerationError('historical_generation_member_unsupported') from None
        retained_bytes += len(encoded) + 1
        _require(retained_bytes <= encoded_limit, 'limit')
        members.append(row)
        allocated += legacy.allocated_bytes(info)

    def same(fd, initial, parent=None, name=None):
        tick()
        _require(legacy._version(os.fstat(fd)) == legacy._version(initial), 'changed')
        if parent is not None:
            _require(legacy._version(os.stat(name, dir_fd=parent, follow_symlinks=False))
                     == legacy._version(initial), 'changed')

    def walk(fd, relative, depth, device):
        nonlocal logical
        tick()
        initial = os.fstat(fd)
        _require(depth <= depth_limit and initial.st_dev == device and stat.S_ISDIR(initial.st_mode),
                 'member_unsupported')
        add(relative, 'directory', initial, None)
        selected = names(fd)
        _require(len(members) + len(selected) <= max_members, 'limit')
        versions = {}
        for name in selected:
            tick()
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            versions[name] = legacy._version(info)
            kind = 'directory' if stat.S_ISDIR(info.st_mode) else 'file' if stat.S_ISREG(info.st_mode) else None
            _require(kind is not None and info.st_dev == device
                     and (kind == 'directory' or info.st_nlink == 1), 'member_unsupported')
            child_path = relative + '/' + name if relative else name
            _require(len(os.fsencode(child_path)) <= path_limit, 'limit')
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            child = os.open(name, flags | (os.O_DIRECTORY if kind == 'directory' else 0), dir_fd=fd)
            try:
                same(child, info, fd, name)
                if kind == 'directory':
                    walk(child, child_path, depth + 1, device)
                else:
                    _require(info.st_size <= max_payload_bytes - logical, 'limit')
                    logical += info.st_size
                    digest, consumed = hashlib.sha256(), 0
                    while True:
                        same(child, info, fd, name)
                        block = os.read(child, 1024**2)
                        tick()
                        if not block:
                            break
                        consumed += len(block)
                        _require(consumed <= info.st_size, 'changed')
                        digest.update(block)
                    same(child, info, fd, name)
                    _require(consumed == info.st_size, 'changed')
                    add(child_path, 'file', info, 'sha256:' + digest.hexdigest())
            finally:
                os.close(child)
        _require(names(fd) == selected, 'changed')
        for name, version in versions.items():
            tick()
            _require(legacy._version(os.stat(name, dir_fd=fd, follow_symlinks=False)) == version, 'changed')
        same(fd, initial)

    try:
        with ExitStack() as stack:
            tick()
            chain = _chain(target, stack)
            parent_fd, target_fd = chain[-2][2], chain[-1][2]
            parent_info, target_info = os.fstat(parent_fd), os.fstat(target_fd)
            _require(parent_info.st_dev == target_info.st_dev, 'member_unsupported')
            walk(target_fd, '', 0, target_info.st_dev)
            _verify_chain(chain)
            tick()
            value = dict(schema_version='control_plane_historical_generation.v1',
                target_path=str(target), parent_path=str(target.parent),
                root_identity=legacy._directory_identity(parent_info),
                target_identity=legacy._directory_identity(target_info),
                root_version=list(legacy._version(parent_info)), target_version=list(legacy._version(target_info)),
                members=members, member_count=len(members), logical_payload_bytes=logical,
                observed_allocated_bytes=allocated, execution_authorized=False)
            _require(len(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
                     <= encoded_limit - 100, 'limit')
            tick()
            value['generation_digest'] = canonical_digest(value, digest_field='generation_digest')
            tick()
            return value
    except (OSError, legacy.LegacyOwnerError):
        raise HistoricalGenerationError('historical_generation_changed') from None
