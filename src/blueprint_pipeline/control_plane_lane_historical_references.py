"""Current held historical queue, pin, active-run and release reference gates.

These tables are observations, not future-reader authority. The complete action
also requires the authenticated owner decommission, kernel access fence and
native process scan. Unknown names/bytes/rights or any drift refuse.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_legacy_owner as legacy
from . import control_plane_lane_experiment_actions as experiments
from .control_plane_lane_historical_fence import _version


def _require(value, code):
    generation._require(value, code)


def _pairs(values):
    result = {}
    for key, value in values:
        _require(key not in result, 'table_unknown')
        result[key] = value
    return result


def _mentions(value, target, budget, depth=0):
    budget.charge('values')
    _require(depth <= 32, 'table_unknown')
    if isinstance(value, str):
        _require(str(target) not in value and target.name not in value, 'table_reference')
    elif isinstance(value, dict):
        for key, child in value.items():
            _mentions(key, target, budget, depth + 1)
            _mentions(child, target, budget, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _mentions(child, target, budget, depth + 1)


def _table(root, target, budget):
    """One bounded original named tree; exact JSON bytes and full namespace."""
    snapshots = []
    def same(fd, initial, parent=None, name=None):
        budget.tick()
        _require(_version(os.fstat(fd)) == initial, 'table_unknown')
        if parent is not None:
            _require(_version(os.stat(name, dir_fd=parent, follow_symlinks=False)) == initial,
                     'table_unknown')
    def walk(fd, relative, depth):
        _require(depth <= 16, 'table_unknown')
        initial = _version(os.fstat(fd))
        names = []
        with os.scandir(fd) as entries:
            for entry in entries:
                budget.charge('entries')
                _require(len(names) < 10000, 'table_unknown')
                names.append(entry.name)
        snapshots.append((relative, 'directory', initial, sorted(names)))
        for name in sorted(names):
            budget.tick()
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            version = _version(info)
            _require(info.st_dev == root_version[0] and
                (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode) and info.st_nlink == 1),
                'table_unknown')
            child_path = relative + '/' + name if relative else name
            _require(len(os.fsencode(child_path)) <= 4096, 'table_unknown')
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
            if stat.S_ISDIR(info.st_mode):
                flags |= os.O_DIRECTORY
            child = os.open(name, flags, dir_fd=fd)
            try:
                same(child, version, fd, name)
                if stat.S_ISDIR(info.st_mode):
                    walk(child, child_path, depth + 1)
                else:
                    digest = None
                    if name.endswith('.json'):
                        _require(0 < info.st_size <= 1024**2, 'table_unknown')
                        payload = bytearray()
                        while True:
                            same(child, version, fd, name)
                            block = os.read(child, min(65536, info.st_size + 1 - len(payload)))
                            budget.charge('raw_bytes', len(block))
                            _require(len(payload) + len(block) <= info.st_size, 'table_unknown')
                            if not block:
                                break
                            payload.extend(block)
                        _require(len(payload) == info.st_size, 'table_unknown')
                        budget.preflight(payload.decode('utf-8'))
                        value = json.loads(payload, object_pairs_hook=_pairs,
                                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
                        _require(type(value) in (dict, list), 'table_unknown')
                        _mentions(value, target, budget)
                        digest = hashlib.sha256(payload).hexdigest()
                    snapshots.append((child_path, 'file', version, digest))
                same(child, version, fd, name)
            finally:
                os.close(child)
        same(fd, initial)
        return initial
    try:
        with ExitStack() as stack:
            legacy._absolute(root)
            chain = generation._chain(root, stack)
            root_version = _version(os.fstat(chain[-1][2]))
            walk(chain[-1][2], '', 0)
            generation._verify_chain(chain)
            return snapshots
    except generation.HistoricalGenerationError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise generation.HistoricalGenerationError('historical_generation_table_unknown') from None


@contextmanager
def historical_reference_fence(files, config, target, *, observed_at):
    """Keep the established state/pin directory locks through the caller effect."""
    selected = legacy._reference_settings(files, config)
    state, _ = files.parent(Path(config.control_plane_state) / '.historical-reference-probe')
    files.location(state)
    fcntl.flock(state, fcntl.LOCK_EX | fcntl.LOCK_NB)
    experiments._pin_fence(files, config, selected['pins_root'], target, observed_at)
    roots = tuple(dict.fromkeys((*selected['queue_roots'], *selected['active_run_roots'])))
    _require(0 < len(roots) <= 16, 'table_unknown')
    baseline = {str(root): _table(root, target, files.budget) for root in roots}
    release = Path(config.active_release_link)
    parent, name = files.parent(release)
    before = _version(os.stat(name, dir_fd=parent, follow_symlinks=False))
    _require(stat.S_ISLNK(before[2]), 'table_unknown')
    link = os.readlink(name, dir_fd=parent)
    _require(len(os.fsencode(link)) <= 4096, 'table_unknown')
    active = Path(link)
    legacy._absolute(active)
    _require(not (active == target or active in target.parents or target in active.parents),
             'release_reference')
    def guard():
        files.location(state)
        _require(legacy._reference_settings(files, config) == selected, 'table_unknown')
        for root in roots:
            _require(_table(root, target, files.budget) == baseline[str(root)], 'table_unknown')
        files.location(parent)
        _require(_version(os.stat(name, dir_fd=parent, follow_symlinks=False)) == before
            and os.readlink(name, dir_fd=parent) == link, 'table_unknown')
    guard()
    yield guard
    guard()
