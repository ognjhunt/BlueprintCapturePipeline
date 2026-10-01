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
from urllib.parse import unquote, urlparse

from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_legacy_owner as legacy
from . import control_plane_lane_experiment_actions as experiments
from .control_plane_lane_historical_fence import _version
from .control_plane_storage_pin_observation import observe_storage_pins


def _require(value, code):
    generation._require(value, code)


def _pairs(values):
    result = {}
    for key, value in values:
        _require(key not in result, 'table_unknown')
        result[key] = value
    return result


def _target_observation(files, target, transient, descriptor_check):
    """Original owner's selected namespace; this supplies no action grant."""
    fd, _ = files.parent(target / '.historical-reference-probe',
                         _descriptor_check=lambda additional: descriptor_check(transient + additional))
    files.proof(fd)
    files.location(fd)
    return fd


def _mentions(value, target, budget, depth=0, *, local=None):
    budget.charge('values')
    _require(depth <= 32, 'table_unknown')
    if isinstance(value, str):
        _require(str(target) not in value and target.name not in value, 'table_reference')
        # File EvidenceReference consumers decode the selected URI once before
        # opening it. An encoded current selector is still a reference; invalid
        # encoded UTF-8 is unknown. Keep the original observation's deadline.
        budget.tick()
        decoded = unquote(value, errors='strict')
        budget.tick()
        _require(str(target) not in decoded and target.name not in decoded, 'table_reference')
        if local is not None:
            local(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            _mentions(key, target, budget, depth + 1, local=local)
            _mentions(child, target, budget, depth + 1, local=local)
    elif isinstance(value, list):
        for child in value:
            _mentions(child, target, budget, depth + 1, local=local)


def _table(root, target, budget, *, _descriptor_check=None, _target_observation=None):
    """One bounded original named tree; exact JSON bytes and full namespace."""
    snapshots, external = [], []
    transient, target_fd = 0, None
    def capacity(count):
        budget.tick()
        _require(count <= 128, 'table_unknown')
        if _descriptor_check is not None:
            _descriptor_check(count)
    def acquire(path, *, directory, active):
        nonlocal transient
        budget.charge('roots')
        legacy._absolute(path)
        components = ('/', *path.parts[1:])
        parent, rows = None, []
        for index, component in enumerate(components):
            budget.charge('entries')
            named = os.stat(component, dir_fd=parent, follow_symlinks=False)
            is_directory = directory or index < len(components) - 1
            _require(stat.S_ISDIR(named.st_mode) if is_directory else
                     stat.S_ISREG(named.st_mode) and named.st_nlink == 1, 'table_unknown')
            capacity(len(chain) + active + transient + 1)
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            if is_directory:
                flags |= os.O_DIRECTORY
            fd = os.open(component, flags, dir_fd=parent)
            stack.callback(os.close, fd)
            transient += 1
            budget.tick()
            version = _version(os.fstat(fd))
            _require(version == _version(named), 'table_unknown')
            rows.append((parent, component, fd, version))
            parent = fd
        external.append(rows)
        return rows
    def local(value, active):
        nonlocal target_fd
        # Match the consumer: parse the ORIGINAL URI, then decode its path
        # exactly once. Do not resolve or follow any queued namespace alias.
        budget.tick()
        parsed = urlparse(value)
        if parsed.scheme != 'file' and not value.startswith('/'):
            return
        _require(parsed.scheme in ('', 'file') and parsed.netloc in ('', 'localhost'), 'table_unknown')
        selected = unquote(parsed.path, errors='strict')
        _require('\x00' not in selected and selected == str(Path(selected)), 'table_unknown')
        source = Path(selected)
        legacy._absolute(source)
        if target_fd is None:
            if _target_observation is None:
                target_fd = acquire(target, directory=True, active=active)[-1][2]
            else:
                budget.tick()
                target_fd = _target_observation(len(chain) + active + transient)
        budget.tick()
        target_info = os.fstat(target_fd)
        _require(stat.S_ISDIR(target_info.st_mode), 'table_unknown')
        rows = acquire(source, directory=False, active=active)
        identity = (target_info.st_dev, target_info.st_ino)
        _require(all(version[:2] != identity for _, _, _, version in rows), 'table_reference')
        budget.charge('facts', len(rows))
        snapshot = (str(source), 'local_uri', tuple(row[3] for row in rows))
        budget.retain(snapshot)
        snapshots.append(snapshot)
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
        capacity(len(chain) + depth + transient + 1)
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
            capacity(len(chain) + depth + transient + 1)
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
                        _mentions(value, target, budget, local=lambda text: local(text, depth + 1))
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
            chain = generation._chain(root, stack, _descriptor_check=capacity)
            root_version = _version(os.fstat(chain[-1][2]))
            walk(chain[-1][2], '', 0)
            generation._verify_chain(chain)
            for rows in external:
                for parent, name, fd, version in rows:
                    same(fd, version, parent, name)
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
    pins, pin_directory = experiments._pin_fence(files, config, selected['pins_root'], target, observed_at)
    roots = tuple(dict.fromkeys((*selected['queue_roots'], *selected['active_run_roots'])))
    _require(0 < len(roots) <= 16, 'table_unknown')
    def target_observation(transient):
        return _target_observation(files, target, transient, descriptor_check)
    def descriptor_check(count):
        files.budget.tick()
        _require(len(files.owned) + len(files.probe_owned) + count <= 128, 'table_unknown')
    def table(root):
        return _table(root, target, files.budget, _descriptor_check=descriptor_check,
                      _target_observation=target_observation)
    baseline = {str(root): table(root) for root in roots}
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
        files.location(pin_directory)
        _require(experiments._reference_configuration(files, config, selected['pins_root'])
                 == pins['configuration'], 'table_unknown')
        current = observe_storage_pins(str(selected['pins_root']), observed_at_epoch=observed_at,
                                      budget=files.budget, _held_root_fd=pin_directory)
        _require(current.complete and current.root_identity
            == (pins['identity']['dev'], pins['identity']['ino']), 'table_unknown')
        for row in current.rows:
            _require(not any(Path(path) == target or target in Path(path).parents
                or Path(path) in target.parents for path in row.paths), 'pin_reference')
        for root in roots:
            _require(table(root) == baseline[str(root)], 'table_unknown')
        files.location(parent)
        _require(_version(os.stat(name, dir_fd=parent, follow_symlinks=False)) == before
            and os.readlink(name, dir_fd=parent) == link, 'table_unknown')
    guard()
    yield guard
    guard()
