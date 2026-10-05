"""Prepare a protected runtime without issuing a grant or enabling cleanup.

Run this stdlib-only installer from a root-owned release, before transferring
the release checkout to the service account. Existing snapshots are immutable; an interrupted installation can resume only
its exact protected copy intent. Unknown or changed bytes are preserved and refused.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import math
import selectors
import signal
import importlib.util
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile


_OWNER = 0
_RUNTIME_ROOT = Path('/mnt/blueprint-work/scene-retirement-runtime')
_BOOT_ROOT = Path('/usr/lib/blueprint/scene-retirement-runtime')
_FREE_FLOOR = 5 * 1024**3
_ERROR = 'scene_retirement_runtime_unproven'
_MAX_FILES = 32768
_MAX_BYTES = 4 * 1024**3
# Installation includes authenticated source, SDK hashing and durable copies of
# up to 32,768 leaves. This availability bound is not a retirement/action lease.
_MAX_SECONDS = 900
_SKIP = frozenset(('__pycache__', '.git'))


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _installation_deadline(supplied=None):
    started = time.monotonic()
    _require(type(started) is float and math.isfinite(started))
    _require(supplied is None or type(supplied) is float and math.isfinite(supplied))
    deadline = started + _MAX_SECONDS if supplied is None else min(started + _MAX_SECONDS, supplied)
    _require(started <= deadline)
    return deadline


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _protected(info, *, directory, partial=False):
    _require((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
             and info.st_uid in {0, _OWNER} and not info.st_mode & 0o022)
    if not directory:
        _require(info.st_nlink == 1 and stat.S_IMODE(info.st_mode) in ((0o600,) if partial else ()) + (0o444, 0o555, 0o644, 0o755))


def _open(path, *, directory, partial=False):
    """Open every ancestry component without following links."""
    _require(path.is_absolute() and '..' not in path.parts)
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        _protected(os.fstat(fd), directory=True)
        for index, name in enumerate(path.parts[1:]):
            is_directory = index < len(path.parts) - 2 or directory
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            _protected(before, directory=is_directory, partial=partial and not is_directory)
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            child = os.open(name, flags | (os.O_DIRECTORY if is_directory else 0), dir_fd=fd)
            try:
                _require(_identity(os.fstat(child)) == _identity(before))
            except BaseException:
                os.close(child)
                raise
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_open_file(fd, path, deadline, *, output=None, _expected=None):
    """Hash an already no-follow opened leaf and verify its final path binding."""
    before = os.fstat(fd)
    _protected(before, directory=False)
    if _expected is not None:
        _require(_identity(before) == _identity(_expected))
    digest = hashlib.sha256()
    count = 0
    while True:
        _require(time.monotonic() <= deadline)
        raw = os.read(fd, 1024 * 1024)
        if not raw:
            break
        count += len(raw)
        _require(count <= before.st_size <= _MAX_BYTES)
        digest.update(raw)
        if output is not None:
            view = memoryview(raw)
            while view:
                written = os.write(output, view)
                _require(written > 0)
                view = view[written:]
    _require(count == before.st_size and _identity(os.fstat(fd)) == _identity(before)
             and _identity(path.lstat()) == _identity(before))
    return {'size': count, 'sha256': digest.hexdigest(),
            'mode': 0o755 if before.st_mode & 0o111 else 0o644}


def _read(path, deadline, *, output=None):
    fd = _open(path, directory=False)
    try:
        return _read_open_file(fd, path, deadline, output=output)
    finally:
        os.close(fd)


def _open_tree_child(parent, name, before, *, directory):
    """Bind one child to the retained protected parent, without following links."""
    _protected(before, directory=directory)
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                 | (os.O_DIRECTORY if directory else 0), dir_fd=parent)
    try:
        _require(_identity(os.fstat(fd)) == _identity(before))
        return fd
    except BaseException:
        os.close(fd)
        raise


def _tree(path, prefix, rows, sources, deadline, depth=0, _total=None,
          _directory_fd=None, _directory_before=None):
    _require(depth <= 32 and time.monotonic() <= deadline)
    # Carry one aggregate through descendants. Re-summing every previous
    # SDK leaf on each insertion made final verification quadratic in files.
    total = [sum(row['size'] for row in rows.values())] if _total is None else _total
    _require(total[0] <= _MAX_BYTES and time.monotonic() <= deadline)
    # Retain the authenticated descriptor chain through the walk. Reopening
    # every ancestry component for each SDK leaf dominated large-tree scans.
    fd = _open(path, directory=True) if _directory_fd is None else _directory_fd
    try:
        before = os.fstat(fd)
        _protected(before, directory=True)
        if _directory_before is not None:
            _require(_identity(before) == _identity(_directory_before))
        names = sorted(os.listdir(fd))
        _require(len(names) <= _MAX_FILES)
        for name in names:
            if name in _SKIP or name.endswith(('.pyc', '.pyo')):
                continue
            child = path / name
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child_fd = _open_tree_child(fd, name, info, directory=True)
                try:
                    _tree(child, prefix / name, rows, sources, deadline, depth + 1,
                          _total=total, _directory_fd=child_fd, _directory_before=info)
                finally:
                    os.close(child_fd)
            else:
                _require(len(rows) < _MAX_FILES)
                key = str(prefix / name)
                leaf_fd = _open_tree_child(fd, name, info, directory=False)
                try:
                    row = _read_open_file(leaf_fd, child, deadline, _expected=info)
                finally:
                    os.close(leaf_fd)
                total[0] += row['size'] - rows.get(key, {}).get('size', 0)
                rows[key] = row
                sources[key] = child
                _require(total[0] <= _MAX_BYTES)
        _require(_identity(os.fstat(fd)) == _identity(before)
                 and _identity(path.lstat()) == _identity(before))
        if depth == 0:
            # Reprove the complete ancestry after the walk. A renamed root,
            # replaced ancestor, symlink alias or newly writable ancestry may
            # not make descriptor-bound bytes into an accepted path snapshot.
            current = _open(path, directory=True)
            try:
                _require(_identity(os.fstat(current)) == _identity(before))
            finally:
                os.close(current)
    finally:
        if _directory_fd is None:
            os.close(fd)


def _nearest(path):
    while not path.exists():
        _require(not path.is_symlink())
        path = path.parent
    fd = _open(path, directory=True)
    os.close(fd)
    return path


def _mkdir(path):
    if path.exists() or path.is_symlink():
        fd = _open(path, directory=True)
        os.close(fd)
        return
    _mkdir(path.parent)
    path.mkdir(mode=0o755)
    parent = _open(path.parent, directory=True)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    fd = _open(path, directory=True)
    os.close(fd)


def _prefix(path, destination, expected, deadline, *, append=False):
    """Verify every existing byte before appending to a declared partial file."""
    source = _open(path, directory=False)
    fd = -1
    try:
        fd = _open(destination, directory=False, partial=True)
        source_before, before = os.fstat(source), os.fstat(fd)
        _require(before.st_size <= expected['size'] == source_before.st_size <= _MAX_BYTES)
        if append:
            os.close(fd)
            fd = -1
            fd = os.open(destination, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
            _require(_identity(os.fstat(fd)) == _identity(before))
        digest, count = hashlib.sha256(), 0
        while True:
            _require(time.monotonic() <= deadline)
            raw = os.read(source, 1024 * 1024)
            if not raw:
                break
            digest.update(raw)
            prior = min(len(raw), max(0, before.st_size - count))
            _require(os.read(fd, prior) == raw[:prior])
            if append:
                view = memoryview(raw)[prior:]
                while view:
                    written = os.write(fd, view)
                    _require(written > 0)
                    view = view[written:]
            count += len(raw)
        _require(count == source_before.st_size == expected['size']
                 and digest.hexdigest() == expected['sha256']
                 and _identity(os.fstat(source)) == _identity(source_before)
                 and _identity(path.lstat()) == _identity(source_before))
        current = os.fstat(fd)
        _require((current.st_dev, current.st_ino) == (before.st_dev, before.st_ino)
                 and _identity(destination.lstat()) == _identity(current))
        if append:
            os.fsync(fd)
            os.fchmod(fd, expected['mode'])
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(source)


def _copy(path, destination, expected, deadline):
    _mkdir(destination.parent)
    if destination.exists() or destination.is_symlink():
        known = _open(destination, directory=False, partial=True)
        try:
            info = os.fstat(known)
            complete = info.st_size == expected['size'] and stat.S_IMODE(info.st_mode) == expected['mode']
        finally:
            os.close(known)
        if complete:
            _require(_read(destination, deadline) == expected)
            return  # Never chmod or write an already immutable generation leaf.
    if not destination.exists() and not destination.is_symlink():
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        os.close(fd)
    _prefix(path, destination, expected, deadline, append=True)
    _require(_read(destination, deadline) == expected)
    parent = _open(destination.parent, directory=True)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _partial_tree(path, rows, sources, deadline, prefix=Path('.'), depth=0, _directories=None):
    """Check the entire incomplete snapshot before completing any copy."""
    _require(depth <= 32 and time.monotonic() <= deadline)
    if _directories is None:
        # Every retained directory must lead to a declared leaf. Compute those
        # exact prefixes once instead of scanning all SDK rows per directory.
        _directories = set()
        for name in rows:
            _require(time.monotonic() <= deadline)
            while '/' in name:
                name = name.rpartition('/')[0]
                _directories.add(name)
    fd = _open(path, directory=True)
    try:
        before = os.fstat(fd)
        names = sorted(os.listdir(fd))
        _require(len(names) <= _MAX_FILES)
        for name in names:
            child = path / name
            key = str(prefix / name)
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                _require(key in _directories)
                _partial_tree(child, rows, sources, deadline, prefix / name, depth + 1, _directories)
            else:
                _require(key in rows)
                _prefix(sources[key], child, rows[key], deadline)
        _require(_identity(os.fstat(fd)) == _identity(before)
                 and _identity(path.lstat()) == _identity(before))
    finally:
        os.close(fd)


def _intent(rows, boot_row, deadline):
    """An exact copy plan is installation evidence, never deletion authority."""
    raw = json.dumps({'schema': 'scene-retirement-runtime-install.v1',
                      'runtime': str(_RUNTIME_ROOT), 'rows': rows,
                      'bootstrap': boot_row}, sort_keys=True, separators=(',', ':')).encode()
    _require(len(raw) <= 16 * 1024**2)
    record = _BOOT_ROOT / 'installation.json'
    if record.exists() or record.is_symlink():
        fd = _open(record, directory=False)
        try:
            before = os.fstat(fd)
            _require(before.st_size == len(raw))
            observed = os.read(fd, len(raw) + 1)
            _require(observed == raw and _identity(os.fstat(fd)) == _identity(before)
                     and _identity(record.lstat()) == _identity(before))
        finally:
            os.close(fd)
        return
    temporary = _BOOT_ROOT / 'installation.pending.json'
    _mkdir(_BOOT_ROOT)
    if not temporary.exists() and not temporary.is_symlink():
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        os.close(fd)
    fd = _open(temporary, directory=False, partial=True)
    before = os.fstat(fd)
    os.close(fd)
    _require(before.st_size <= len(raw))
    fd = os.open(temporary, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _require(_identity(os.fstat(fd)) == _identity(before))
        _require(os.read(fd, len(raw) + 1) == raw[:before.st_size])
        view = memoryview(raw)[before.st_size:]
        while view:
            _require(time.monotonic() <= deadline)
            written = os.write(fd, view)
            _require(written > 0)
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o644)
    finally:
        os.close(fd)
    os.rename(temporary, record)
    parent = _open(_BOOT_ROOT, directory=True)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _verify(runtime, rows, boot, boot_row, deadline):
    observed, sources = {}, {}
    _tree(runtime, Path('.'), observed, sources, deadline)
    _require(observed == rows and _read(boot, deadline) == boot_row)


def dependency_root(venv):
    """Derive the SDK directory without executing a service-owned Python."""
    try:
        venv = Path(venv)
        config = venv / 'pyvenv.cfg'
        fd = _open(config, directory=False)
        try:
            before = os.fstat(fd)
            _require(0 < before.st_size <= 4096)
            raw = os.read(fd, 4097)
            _require(len(raw) == before.st_size and _identity(os.fstat(fd)) == _identity(before)
                     and _identity(config.lstat()) == _identity(before))
        finally:
            os.close(fd)
        versions = re.findall(r'^version\s*=\s*(\d+)\.(\d+)\.\d+\s*$', raw.decode('ascii'), re.MULTILINE)
        _require(len(versions) == 1 and tuple(map(int, versions[0])) == sys.version_info[:2])
        sdk = venv / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        fd = _open(sdk, directory=True)
        os.close(fd)
        return sdk
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(_ERROR) from exc


@contextmanager
def _installation_lock():
    """Only one root installer may advance the fixed copy intent."""
    _mkdir(_BOOT_ROOT)
    path = _BOOT_ROOT / 'installation.lock'
    if not path.exists() and not path.is_symlink():
        try:
            created = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                os.fsync(created)
            finally:
                os.close(created)
    fd = _open(path, directory=False, partial=True)
    try:
        before = os.fstat(fd)
        _require(before.st_size == 0 and stat.S_IMODE(before.st_mode) == 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _require(_identity(path.lstat()) == _identity(before))
        yield
        _require(_identity(os.fstat(fd)) == _identity(before)
                 and _identity(path.lstat()) == _identity(before))
    finally:
        os.close(fd)


def prepare(source, dependencies, *, _deadline=None):
    """Copy exact protected inputs; no policy, consent, generation or flag writes."""
    try:
        source, dependencies = Path(source), Path(dependencies)
        deadline = _installation_deadline(_deadline)
        rows, sources = {}, {}
        _tree(source / 'src/blueprint_pipeline', Path('src/blueprint_pipeline'), rows, sources, deadline)
        _tree(source / 'deploy/systemd', Path('deploy/systemd'), rows, sources, deadline)
        # Worker entrypoints can import repository scripts after the UID drop.
        _tree(source / 'scripts', Path('scripts'), rows, sources, deadline)
        _tree(dependencies, Path('dependencies'), rows, sources, deadline)
        boot_source = source / 'scripts/scene_retirement_continuous_bootstrap.py'
        boot_row = _read(boot_source, deadline)
        boot = _BOOT_ROOT / 'continuous_bootstrap.py'
        with _installation_lock():
            record = _BOOT_ROOT / 'installation.json'
            if boot.exists() or boot.is_symlink():
                _verify(_RUNTIME_ROOT, rows, boot, boot_row, deadline)
                return {'status': 'already_prepared', 'authority_issued': False, 'cleanup_enabled': False}
            if _RUNTIME_ROOT.exists() or _RUNTIME_ROOT.is_symlink():
                _require(record.exists() and not record.is_symlink())
                _intent(rows, boot_row, deadline)
                _partial_tree(_RUNTIME_ROOT, rows, sources, deadline)
            temporary_boot = _BOOT_ROOT / 'continuous_bootstrap.pending.py'
            if temporary_boot.exists() or temporary_boot.is_symlink():
                _require(record.exists() and not record.is_symlink())
                _intent(rows, boot_row, deadline)
                _prefix(boot_source, temporary_boot, boot_row, deadline)
            total = sum(row['size'] for row in rows.values()) + boot_row['size']
            available = os.statvfs(_nearest(_RUNTIME_ROOT.parent))
            _require(available.f_bavail * available.f_frsize >= total + _FREE_FLOOR)
            boot_available = os.statvfs(_nearest(_BOOT_ROOT.parent))
            _require(boot_available.f_bavail * boot_available.f_frsize >= boot_row['size'] + _FREE_FLOOR)
            _intent(rows, boot_row, deadline)
            _mkdir(_RUNTIME_ROOT)
            for name, row in rows.items():
                _copy(sources[name], _RUNTIME_ROOT / name, row, deadline)
            # The fixed executable appears only after all runtime copies verify.
            _copy(boot_source, temporary_boot, boot_row, deadline)
            os.rename(temporary_boot, boot)
            parent = _open(_BOOT_ROOT, directory=True)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            _verify(_RUNTIME_ROOT, rows, boot, boot_row, deadline)
            return {'status': 'prepared', 'authority_issued': False, 'cleanup_enabled': False,
                    'files': len(rows), 'bytes': total}
    except (OSError, ValueError) as exc:
        raise ValueError(_ERROR) from exc



def _record_bytes(path, deadline, cap=16 * 1024**2, *, allow_empty=False):
    fd = _open(path, directory=False)
    try:
        before = os.fstat(fd)
        _require((allow_empty or before.st_size > 0) and before.st_size <= cap)
        raw = bytearray()
        while len(raw) < before.st_size:
            _require(time.monotonic() <= deadline)
            block = os.read(fd, min(1024 * 1024, before.st_size - len(raw)))
            _require(bool(block))
            raw.extend(block)
        _require(_identity(os.fstat(fd)) == _identity(before)
                 == _identity(path.lstat()))
        return bytes(raw), before
    finally:
        os.close(fd)


def _selector(raw):
    return {'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def _encoded(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    _require(0 < len(raw) <= 16 * 1024**2)
    return raw


def _record(path, raw, deadline, *, previous=None, allow_empty=False):
    """Publish one owned immutable record or exact current CAS under installer EX."""
    _require((allow_empty or len(raw) > 0) and len(raw) <= 16 * 1024**2)
    _mkdir(path.parent)
    if path.exists() or path.is_symlink():
        observed, info = _record_bytes(path, deadline, allow_empty=allow_empty)
        if previous is None:
            _require(observed == raw)
            return
        _require(_identity(info) == _identity(previous))
    else:
        _require(previous is None)
    temporary = path.with_name(path.name + '.pending')
    if not temporary.exists() and not temporary.is_symlink():
        created = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        os.close(created)
    fd = _open(temporary, directory=False, partial=True)
    before = os.fstat(fd)
    os.close(fd)
    _require(before.st_size <= len(raw))
    fd = os.open(temporary, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _require(_identity(os.fstat(fd)) == _identity(before))
        _require(os.read(fd, len(raw) + 1) == raw[:before.st_size])
        view = memoryview(raw)[before.st_size:]
        while view:
            _require(time.monotonic() <= deadline)
            count = os.write(fd, view[:1024 * 1024])
            _require(0 < count <= len(view))
            view = view[count:]
        _require(_identity(temporary.lstat()) == _identity(os.fstat(fd)))
        os.fsync(fd)
        os.fchmod(fd, 0o644)
        final = os.fstat(fd)
        _require(_identity(temporary.lstat()) == _identity(final))
        if previous is not None:
            _require(_identity(path.lstat()) == _identity(previous))
        else:
            _require(not path.exists() and not path.is_symlink())
        os.rename(temporary, path)
        renamed = os.fstat(fd)
        # This owned rename changes ctime on Linux/macOS. All other acquired
        # fields remain exact, and the new name must identify this same FD.
        _require(_identity(path.lstat()) == _identity(renamed)
                 and _identity(renamed)[:-1] == _identity(final)[:-1])
        parent = _open(path.parent, directory=True)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        os.close(fd)
    _require(_record_bytes(path, deadline, allow_empty=allow_empty)[0] == raw)


def _refresh_inputs(source, dependencies, deadline):
    rows, sources, sdk_rows, sdk_sources = {}, {}, {}, {}
    _tree(source / 'src/blueprint_pipeline', Path('src/blueprint_pipeline'), rows, sources, deadline)
    _tree(source / 'deploy/systemd', Path('deploy/systemd'), rows, sources, deadline)
    _tree(source / 'scripts', Path('scripts'), rows, sources, deadline)
    _tree(dependencies, Path('.'), sdk_rows, sdk_sources, deadline)
    _require(len(rows) + len(sdk_rows) <= _MAX_FILES
             and sum(row['size'] for row in (*rows.values(), *sdk_rows.values())) <= _MAX_BYTES)
    return rows, sources, sdk_rows, sdk_sources


def _generation(kind, rows, sources, deadline):
    digest = hashlib.sha256(_encoded(rows)).hexdigest()
    parent = _RUNTIME_ROOT / ('generations' if kind == 'source' else 'sdk-generations')
    generation = parent / digest
    manifest = _RUNTIME_ROOT / 'manifests' / (kind + '-' + digest + '.json')
    raw = _encoded({'schema': 'scene-retirement-runtime-generation.v1', 'kind': kind,
                    'generation': digest, 'rows': rows})
    if generation.exists() or generation.is_symlink():
        _require(manifest.exists() and not manifest.is_symlink())
        _require(_record_bytes(manifest, deadline)[0] == raw)
        _partial_tree(generation, rows, sources, deadline)
    else:
        # The exact selected digest is the only generation examined. Prior
        # immutable generations may still be executing; neither enumerate nor
        # delete them and do not impose a permanent deploy-count ceiling.
        if parent.exists():
            fd = _open(parent, directory=True)
            os.close(fd)
        _record(manifest, raw, deadline)
        _mkdir(generation)
    for name, row in rows.items():
        _copy(sources[name], generation / name, row, deadline)
    observed, _ = {}, {}
    _tree(generation, Path('.'), observed, {}, deadline)
    _require(observed == rows)
    return generation, digest


def refresh(source, dependencies, *, expected_current, _deadline=None):
    """Select one verified source/SDK cohort; old generations remain immutable."""
    try:
        _require(type(expected_current) is dict and set(expected_current) == {'sha256', 'size_bytes'}
                 and type(expected_current['sha256']) is str
                 and re.fullmatch(r'sha256:[0-9a-f]{64}', expected_current['sha256'])
                 and type(expected_current['size_bytes']) is int and 0 < expected_current['size_bytes'] <= 16 * 1024**2)
        deadline = _installation_deadline(_deadline)
        source, dependencies = Path(source), Path(dependencies)
        rows, sources, sdk_rows, sdk_sources = _refresh_inputs(source, dependencies, deadline)
        boot_source = source / 'scripts/scene_retirement_continuous_bootstrap.py'
        boot_row = _read(boot_source, deadline)
        with _installation_lock():
            current_path = _BOOT_ROOT / 'CURRENT.json'
            selected_path = current_path if current_path.exists() or current_path.is_symlink() else _BOOT_ROOT / 'installation.json'
            old_raw, old_info = _record_bytes(selected_path, deadline)
            _require(_selector(old_raw) == expected_current)
            old = json.loads(old_raw)
            old_sdk = _RUNTIME_ROOT / 'dependencies'
            if selected_path == current_path:
                _require(type(old) is dict and old.get('schema') == 'scene-retirement-runtime-cohort.v1'
                         and set(old) == {'schema', 'runtime_root', 'dependencies_root', 'source_digest', 'dependency_digest', 'bootstrap', 'previous'})
                old_sdk = Path(old['dependencies_root'])
                _require(old_sdk == _RUNTIME_ROOT / 'dependencies'
                         or old_sdk.parent == _RUNTIME_ROOT / 'sdk-generations'
                         and re.fullmatch('[0-9a-f]{64}', old_sdk.name))
            else:
                _require(type(old) is dict and old.get('schema') == 'scene-retirement-runtime-install.v1'
                         and old.get('runtime') == str(_RUNTIME_ROOT))
            # Only identical protected SDK bytes can be reused, with no relink or
            # copied GB for each new source generation. Unknown SDKs refuse.
            installed_sdk = {}
            _tree(old_sdk, Path('.'), installed_sdk, {}, deadline)
            required = sum(row['size'] for row in rows.values())
            if installed_sdk != sdk_rows:
                required += sum(row['size'] for row in sdk_rows.values())
            available = os.statvfs(_nearest(_RUNTIME_ROOT.parent))
            _require(available.f_bavail * available.f_frsize >= required + _FREE_FLOOR)
            runtime, source_digest = _generation('source', rows, sources, deadline)
            if installed_sdk == sdk_rows:
                sdk = old_sdk
                dependency_digest = hashlib.sha256(_encoded(sdk_rows)).hexdigest()
            else:
                sdk, dependency_digest = _generation('sdk', sdk_rows, sdk_sources, deadline)
            boot = _BOOT_ROOT / 'continuous_bootstrap.py'
            _require(boot.exists() and not boot.is_symlink())
            # The selector binds exact bootstrap bytes. A crash in the executable
            # update interval is a refusal, never permission to mix cohorts.
            if _read(boot, deadline) != boot_row:
                staged = _BOOT_ROOT / 'continuous_bootstrap.refresh.py'
                _copy(boot_source, staged, boot_row, deadline)
                os.replace(staged, boot)
                parent = _open(_BOOT_ROOT, directory=True)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            _require(_read(boot, deadline) == boot_row)
            _require(_record_bytes(selected_path, deadline)[0] == old_raw)
            value = {'schema': 'scene-retirement-runtime-cohort.v1', 'runtime_root': str(runtime),
                     'dependencies_root': str(sdk), 'source_digest': source_digest,
                     'dependency_digest': dependency_digest, 'bootstrap': boot_row, 'previous': expected_current}
            raw = _encoded(value)
            _require(len(raw) <= 4096)
            _record(current_path, raw, deadline, previous=old_info if selected_path == current_path else None)
            return {'status': 'refreshed', 'runtime_root': str(runtime), 'dependencies_root': str(sdk),
                    'current': _selector(raw), 'authority_issued': False, 'cleanup_enabled': False}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(_ERROR) from exc



def _sdk_root():
    # Separate fixed protected input cache; initial prepare's exact runtime
    # snapshot never adopts these staging artifacts as installed source.
    return _RUNTIME_ROOT.with_name(_RUNTIME_ROOT.name + '-sdk-inputs')


def _sdk_file(path, expected, deadline):
    actual = _read(path, deadline)
    _require(actual['sha256'] == expected['hash'].removeprefix('sha256:')
             and actual['size'] == expected['size'])
    return path


def _sdk_space(path, required, deadline):
    _require(type(required) is int and 0 <= required <= _MAX_BYTES and time.monotonic() <= deadline)
    available = os.statvfs(_nearest(path))
    _require(available.f_bavail * available.f_frsize >= required + _FREE_FLOOR)


def _sdk_existing_size(path, expected_size):
    if not path.exists() and not path.is_symlink():
        return 0
    fd = _open(path, directory=False, partial=True)
    try:
        original = os.fstat(fd)
        _require(original.st_size <= expected_size)
        return original.st_size
    finally:
        os.close(fd)


def _sdk_partial(path, expected_size):
    if path.exists() or path.is_symlink():
        check = _open(path, directory=False, partial=True)
        try:
            original = os.fstat(check)
            _require(original.st_size <= expected_size)
            output = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                _require(_identity(os.fstat(output)) == _identity(original))
            except BaseException:
                os.close(output)
                raise
            return output, original
        finally:
            os.close(check)
    output = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    return output, os.fstat(output)


def _sdk_append_chunk(output, original, raw, offset, deadline):
    _require(time.monotonic() <= deadline)
    prior = min(len(raw), max(0, original.st_size - offset))
    _require(os.pread(output, prior, offset) == raw[:prior])
    current = os.fstat(output)
    _require((current.st_dev, current.st_ino) == (original.st_dev, original.st_ino))
    view = memoryview(raw)[prior:]
    os.lseek(output, offset + prior, os.SEEK_SET)
    while view:
        _require(time.monotonic() <= deadline)
        written = os.write(output, view)
        _require(written > 0)
        view = view[written:]


def _sdk_artifact(row, wheelhouse, deadline):
    url = row['url']
    parsed = urllib.parse.urlsplit(url)
    _require(parsed.scheme == 'https' and parsed.hostname == 'files.pythonhosted.org'
             and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
             and type(row['size']) is int and 0 < row['size'] <= _MAX_BYTES
             and re.fullmatch(r'sha256:[0-9a-f]{64}', row['hash']))
    name = Path(parsed.path).name
    _require(name.endswith('.whl'))
    if wheelhouse is not None:
        return _sdk_file(Path(wheelhouse) / name, row, deadline)
    directory = _sdk_root() / 'wheel-artifacts' / row['hash'][7:]
    path = directory / name
    if path.exists() or path.is_symlink():
        return _sdk_file(path, row, deadline)
    partial = directory / (name + '.pending')
    _sdk_space(directory.parent, row['size'] - _sdk_existing_size(partial, row['size']), deadline)
    _mkdir(directory)
    claim = directory / (name + '.download.json')
    value = _encoded({'schema': 'scene-retirement-sdk-download.v1', 'url': url, 'size': row['size'], 'hash': row['hash']})
    if partial.exists() or partial.is_symlink():
        _require(claim.exists() and _record_bytes(claim, deadline)[0] == value)
    _record(claim, value, deadline)
    fd, before = _sdk_partial(partial, row['size'])
    try:
        digest, count = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=min(30, max(.001, deadline - time.monotonic()))) as response:
            _require(response.status == 200 and urllib.parse.urlsplit(response.url).hostname == parsed.hostname)
            while True:
                _require(time.monotonic() <= deadline)
                chunk = response.read(min(1024 * 1024, row['size'] + 1 - count))
                if not chunk:
                    break
                count += len(chunk)
                _require(count <= row['size'])
                digest.update(chunk)
                _sdk_append_chunk(fd, before, chunk, count - len(chunk), deadline)
        _require(count == row['size'] and digest.hexdigest() == row['hash'][7:]
                 and os.fstat(fd).st_ino == before.st_ino
                 and partial.lstat().st_ino == before.st_ino)
        os.fsync(fd)
        os.fchmod(fd, 0o644)
        _require(not path.exists() and not path.is_symlink())
        os.rename(partial, path)
    finally:
        os.close(fd)
    return _sdk_file(path, row, deadline)


def _wheel_entries(path, deadline):
    """Preflight/hash wheel payload without executing installer or .pth code."""
    fd = _open(path, directory=False)
    try:
        first = os.fstat(fd)
        rows = {}
        with os.fdopen(os.dup(fd), 'rb') as retained, zipfile.ZipFile(retained) as archive:
            infos = archive.infolist()
            _require(len(infos) <= _MAX_FILES)
            for info in infos:
                _require(time.monotonic() <= deadline and not info.flag_bits & 1)
                parts = Path(info.filename).parts
                _require(parts and not info.filename.startswith('/') and '..' not in parts
                         and '\\' not in info.filename and len(parts) <= 32)
                mode = info.external_attr >> 16
                _require(not stat.S_ISLNK(mode))
                if info.is_dir():
                    continue
                name = info.filename
                if parts[0].endswith('.data'):
                    _require(len(parts) >= 3 and parts[1] in {'purelib', 'platlib', 'scripts', 'data', 'headers'})
                    if parts[1] in {'scripts', 'data', 'headers'}:
                        continue
                    name = '/'.join(parts[2:])
                _require(not name.endswith(('.pth', '.pyc', '.pyo')) and name not in rows
                         and 0 <= info.file_size <= _MAX_BYTES)
                digest, count = hashlib.sha256(), 0
                with archive.open(info) as body:
                    while True:
                        _require(time.monotonic() <= deadline)
                        raw = body.read(1024 * 1024)
                        if not raw:
                            break
                        count += len(raw)
                        _require(count <= info.file_size)
                        digest.update(raw)
                _require(count == info.file_size)
                rows[name] = {'size': count, 'sha256': digest.hexdigest(), 'mode': 0o644,
                              'archive': str(path), 'member': info.filename}
        _require(_identity(os.fstat(fd)) == _identity(first)
                 and _identity(path.lstat()) == _identity(first))
        return rows
    finally:
        os.close(fd)


@contextmanager
def _sdk_archive_reader():
    """Retain one directory index, bound to the independently reopened wheel."""
    with ExitStack() as retained:
        selected, identity, archive = None, None, None

        def read(path, fd, before):
            nonlocal selected, identity, archive
            if selected != path:
                retained.close()
                body = retained.enter_context(os.fdopen(os.dup(fd), 'rb'))
                archive = retained.enter_context(zipfile.ZipFile(body))
                selected, identity = path, _identity(before)
            _require(_identity(before) == identity)
            return archive

        yield read


def _sdk_extract(root, rows, deadline):
    _require(len(rows) <= _MAX_FILES and sum(row['size'] for row in rows.values()) <= _MAX_BYTES)
    required = 0
    for name, row in rows.items():
        target = root / name
        partial = target if 'source' in row or target.exists() or target.is_symlink() else target.with_name(target.name + '.pending')
        required += row['size'] - _sdk_existing_size(partial, row['size'])
    _sdk_space(root.parent, required, deadline)
    _mkdir(root)
    with _sdk_archive_reader() as archive_reader:
        for name, row in sorted(rows.items()):
            _require(time.monotonic() <= deadline)
            target = root / name
            _mkdir(target.parent)
            expected = {key: row[key] for key in ('size', 'sha256', 'mode')}
            if 'source' in row:
                _copy(Path(row['source']), target, expected, deadline)
                continue
            if target.exists() or target.is_symlink():
                _require(_read(target, deadline) == expected)
                continue
            _require('archive' in row)
            fd = _open(Path(row['archive']), directory=False)
            output = None
            try:
                before = os.fstat(fd)
                archive = archive_reader(row['archive'], fd, before)
                temporary = target.with_name(target.name + '.pending')
                output, original = _sdk_partial(temporary, row['size'])
                count, digest = 0, hashlib.sha256()
                with archive.open(row['member']) as body:
                    while True:
                        _require(time.monotonic() <= deadline)
                        raw = body.read(1024 * 1024)
                        if not raw:
                            break
                        count += len(raw)
                        _require(count <= row['size'])
                        digest.update(raw)
                        _sdk_append_chunk(output, original, raw, count - len(raw), deadline)
                _require(count == row['size'] and digest.hexdigest() == row['sha256']
                         and _identity(os.fstat(fd)) == _identity(before)
                         and _identity(Path(row['archive']).lstat()) == _identity(before)
                         and temporary.lstat().st_ino == original.st_ino)
                os.fsync(output)
                os.fchmod(output, 0o644)
                _require(not target.exists() and not target.is_symlink())
                os.rename(temporary, target)
                _require(_read(target, deadline) == expected)
            finally:
                if output is not None:
                    os.close(output)
                os.close(fd)


def _sdk_marker_tools(packages, wheelhouse, deadline):
    selected = [package for package in packages if package['name'] == 'packaging']
    if not selected:
        # Tiny dependency-free/pure-wheel fixtures do not require a parser.
        _require(not any('marker' in edge for package in packages for edge in package.get('dependencies', ())))
        return None
    _require(len(selected) == 1)
    wheels = [row for row in selected[0].get('wheels', ()) if row['url'].endswith('-py3-none-any.whl')]
    _require(len(wheels) == 1)
    path = _sdk_artifact(wheels[0], wheelhouse, deadline)
    rows = _wheel_entries(path, deadline)
    root = _sdk_root() / 'sdk-tools' / wheels[0]['hash'][7:]
    _sdk_extract(root, rows, deadline)
    prefix = '_blueprint_verified_sdk_packaging_' + wheels[0]['hash'][7:]
    _require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules))
    spec = importlib.util.spec_from_file_location(prefix, root / 'packaging/__init__.py',
                                                submodule_search_locations=[str(root / 'packaging')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[prefix] = module
    try:
        spec.loader.exec_module(module)
        return tuple(__import__(prefix + '.' + name, fromlist=[name]) for name in ('markers', 'tags', 'utils'))
    except BaseException:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + '.'):
                del sys.modules[name]
        raise


def _sdk_closure(packages, tools):
    roots = [row for row in packages if row['name'] == 'blueprint-capture-pipeline'
             and row.get('source') == {'editable': '.'}]
    _require(len(roots) == 1)
    selected, todo = {}, list(roots[0].get('dependencies', ()))
    # These declared CPU runtime roots are imported by actual control-plane
    # services/guards. Model/GPU operators remain in their separate workers.
    runtime = roots[0].get('optional-dependencies', {}).get('runtime')
    if runtime is not None:
        _require(type(runtime) is list)
        cpu = {'opencv-python-headless', 'build123d', 'pycollada', 'trimesh'}
        edges = [edge for edge in runtime if edge['name'] in cpu]
        _require(len(edges) <= 128 and {edge['name'] for edge in edges} == cpu)
        todo.extend(edges)
    _require(todo)
    seen_edges = 0
    while todo:
        edge = todo.pop()
        seen_edges += 1
        _require(seen_edges <= _MAX_FILES)
        if 'marker' in edge:
            _require(tools is not None)
            if not tools[0].Marker(edge['marker']).evaluate({'extra': ''}):
                continue
        candidates = [row for row in packages if row['name'] == edge['name']
                      and ('version' not in edge or edge['version'] == row['version'])
                      and ('source' not in edge or edge['source'] == row.get('source'))]
        if len(candidates) > 1 and tools is not None:
            candidates = [row for row in candidates if any(tools[0].Marker(marker).evaluate({'extra': ''})
                          for marker in row.get('resolution-markers', ()))]
        _require(len(candidates) == 1)
        row = candidates[0]
        key = (row['name'], row['version'])
        extras = edge.get('extra', [])
        extras = [extras] if type(extras) is str else extras
        _require(type(extras) is list and len(extras) <= 64
                 and all(type(extra) is str and extra for extra in extras)
                 and len(set(extras)) == len(extras))
        requested = {None, *extras}
        previous = selected.get(key, (row, set()))[1]
        fresh = requested - previous
        if not fresh:
            continue
        previous.update(requested)
        selected[key] = (row, previous)
        if None in fresh:
            todo.extend(row.get('dependencies', ()))
        for extra in fresh - {None}:
            _require(extra in row.get('optional-dependencies', {}))
            todo.extend(row['optional-dependencies'][extra])
    _require(len({key[0] for key in selected}) == len(selected))
    return [selected[key][0] for key in sorted(selected)]


def _sdk_wheel(package, tools):
    choices = []
    tags = list(tools[1].sys_tags()) if tools is not None else None
    for row in package.get('wheels', ()):
        name = Path(urllib.parse.urlsplit(row['url']).path).name
        if tools is None:
            if name.endswith('-py3-none-any.whl'):
                choices.append((0, row))
        else:
            wheel_name, version, _, wheel_tags = tools[2].parse_wheel_filename(name)
            _require(str(wheel_name) == tools[2].canonicalize_name(package['name'])
                     and str(version) == package['version'])
            ranks = [index for index, tag in enumerate(tags) if tag in wheel_tags]
            if ranks:
                choices.append((min(ranks), row))
    _require(choices)
    return sorted(choices, key=lambda item: (item[0], item[1]['url']))[0][1]


def _sdk_git_command(checkout, arguments, deadline, *, cap=1024 * 1024, raw_checkout=False, ssh=None, input_data=None):
    executable = Path('/usr/bin/git')
    parent = _open(executable.parent, directory=True)
    try:
        before = os.stat(executable.name, dir_fd=parent, follow_symlinks=False)
        _require(stat.S_ISREG(before.st_mode) and before.st_uid == 0
                 and stat.S_IMODE(before.st_mode) == 0o755 and before.st_nlink >= 1)
        binary = os.open(executable.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        try:
            _require(_identity(os.fstat(binary)) == _identity(before))
        finally:
            os.close(binary)
    finally:
        os.close(parent)
    if raw_checkout:
        # Raw Git objects are untrusted bytes until the authorized commit and
        # every blob hash are independently verified. No checkout code runs.
        _require(checkout.is_absolute() and '..' not in checkout.parts and checkout.is_dir()
                 and not checkout.is_symlink())
    else:
        fd = _open(checkout, directory=True)
        os.close(fd)
    _require(time.monotonic() <= deadline)
    # No repository hook, external filter, replace-object or mutable user
    # configuration can execute during the raw object read.
    command = [str(executable), '--no-replace-objects', '-C', str(checkout),
               '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false',
               '-c', 'safe.directory=' + str(checkout), *arguments]
    environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'HOME': '/nonexistent',
                   'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_NOSYSTEM': '1',
                   'GIT_TERMINAL_PROMPT': '0'}
    if raw_checkout:
        environment.update({'GIT_NO_LAZY_FETCH': '1', 'GIT_ALLOW_PROTOCOL': '', 'GIT_PROTOCOL_FROM_USER': '0'})
    if ssh is not None:
        environment['GIT_SSH_COMMAND'] = ssh
    _require(type(cap) is int and 0 <= cap <= _MAX_BYTES)
    invocation_deadline = min(deadline, time.monotonic()+30)
    # A private, immediately unlinked input file avoids pipe deadlocks while
    # keeping batch requests bounded. Output still uses the same capped reader.
    with ExitStack() as inputs:
        child_input = subprocess.DEVNULL
        if input_data is not None:
            _require(type(input_data) is bytes and len(input_data) <= 65536)
            child_input = inputs.enter_context(tempfile.TemporaryFile())
            child_input.write(input_data)
            child_input.seek(0)
        process = subprocess.Popen(command, stdin=child_input, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=environment, start_new_session=True)
    output, errors = bytearray(), bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, (output,cap))
            selector.register(process.stderr, selectors.EVENT_READ, (errors,4096))
            while selector.get_map():
                _require(time.monotonic() <= invocation_deadline)
                for key, _ in selector.select(min(.1,max(0,invocation_deadline-time.monotonic()))):
                    buffer, limit = key.data
                    raw = os.read(key.fd,min(65536,limit+1-len(buffer)))
                    if not raw:
                        selector.unregister(key.fileobj)
                    else:
                        _require(len(buffer)+len(raw) <= limit)
                        buffer.extend(raw)
        _require(process.wait(timeout=max(.001,invocation_deadline-time.monotonic())) == 0
                 and time.monotonic() <= deadline
                 and _identity(executable.lstat()) == _identity(before))
        return bytes(output)
    except subprocess.SubprocessError as exc:
        raise ValueError(_ERROR) from exc
    finally:
        # The direct child remains unreaped until this owned process-group kill.
        # Fixed Git/SSH children cannot survive a quota/deadline refusal.
        if process.returncode is None:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Some local OS policies deny group signals. The retained
                # direct child can still be killed/joined; this aborted
                # acquisition never publishes SDK or deployment success.
                process.kill()
        try:
            process.wait(timeout=5)
        finally:
            try:
                process.stdout.close()
            finally:
                process.stderr.close()


def _sdk_fetch_contracts(commit, deadline):
    # This fixed repository is public. No root key, mutable credential helper,
    # user configuration or prompt is needed; the caller verifies Git hashes.
    _require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit))
    root = _sdk_root() / 'git-objects' / commit
    claim = root.parent / (commit + '.claim.json')
    raw = _encoded({'schema': 'scene-retirement-contracts-source.v1', 'commit': commit,
                    'repository': 'ognjhunt/BlueprintContracts'})
    _mkdir(root.parent)
    if root.exists() or root.is_symlink():
        _require(claim.exists() and _record_bytes(claim, deadline)[0] == raw)
    _record(claim, raw, deadline)
    _mkdir(root)
    if not (root / 'HEAD').exists():
        _sdk_git_command(root, ['init', '--bare', '--quiet'], deadline)
    _sdk_git_command(root, ['-c', 'credential.helper=', 'fetch', '--quiet', '--no-tags', '--depth=1',
                           'https://github.com/ognjhunt/BlueprintContracts.git', commit], deadline)
    return root


def _authenticated_git_entries(checkout, commit, wanted, deadline, *, raw_checkout=False):
    raw = _sdk_git_command(checkout, ['cat-file', 'commit', commit], deadline, cap=65536,
                           raw_checkout=raw_checkout)
    _require(hashlib.sha1(b'commit ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == commit)
    trees = re.findall(rb'^tree ([0-9a-f]{40})$', raw, re.MULTILINE)
    _require(len(trees) == 1)
    pending, result, count, total = [('', trees[0].decode(), 0)], [], 0, 0
    while pending:
        prefix, digest, depth = pending.pop()
        _require(depth <= 32 and time.monotonic() <= deadline)
        body = _sdk_git_command(checkout, ['cat-file', 'tree', digest], deadline, cap=1024*1024,
                                raw_checkout=raw_checkout)
        total += len(body)
        _require(total <= 20*1024*1024 and hashlib.sha1(b'tree ' + str(len(body)).encode() + b'\0' + body).hexdigest() == digest)
        offset, names = 0, set()
        while offset < len(body):
            split = body.index(b' ', offset)
            ending = body.index(b'\0', split + 1)
            mode, name = body[offset:split].decode('ascii'), body[split+1:ending].decode('utf-8')
            count += 1
            _require(count <= _MAX_FILES and name not in names and name not in {'', '.', '..'}
                     and '/' not in name and '\\' not in name and ending + 21 <= len(body))
            names.add(name)
            selected = body[ending+1:ending+21].hex()
            offset = ending + 21
            path = prefix + name
            relevant = any(path == value or path.startswith(value + '/') or value.startswith(path + '/') for value in wanted)
            if not relevant:
                continue
            if mode == '40000':
                pending.append((path + '/', selected, depth + 1))
            else:
                _require(mode in {'100644', '100755'})
                result.append((mode, selected, path))
    return sorted(result, key=lambda item: item[2])


def _sdk_git_rows(package, checkout, deadline):
    source = package.get('source', {})
    match = re.fullmatch(r'https://github\.com/ognjhunt/BlueprintContracts\.git\?rev=([0-9a-f]{40})#([0-9a-f]{40})', source.get('git', ''))
    _require(package['name'] == 'blueprint-contracts' and match is not None and match[1] == match[2])
    commit = match[1]
    checkout = Path(checkout) if checkout is not None else _sdk_fetch_contracts(commit, deadline)
    items = _authenticated_git_entries(checkout, commit, ('src/blueprint_contracts', 'blueprint_contracts'), deadline)
    _require(items)
    rows = {}
    for mode, digest, name in items:
        if name.startswith('src/'):
            name = name[4:]
        _require(name.startswith('blueprint_contracts/') and '..' not in Path(name).parts
                 and name not in rows and not name.endswith(('.pth', '.pyc', '.pyo')))
        size = _sdk_git_command(checkout, ['cat-file', '-s', digest], deadline, cap=32)
        _require(size.strip().isdigit() and int(size) <= 1024 * 1024)
        body = _sdk_git_command(checkout, ['cat-file', 'blob', digest], deadline, cap=int(size))
        _require(len(body) == int(size) and hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest() == digest)
        path = _sdk_root() / 'git-inputs' / commit / name
        _mkdir(path.parent)
        _record(path, body, deadline)
        rows[name] = {'size': len(body), 'sha256': hashlib.sha256(body).hexdigest(), 'mode': 0o644,
                      'source': str(path)}
    _require('blueprint_contracts/__init__.py' in rows)
    return rows


def _sdk_toml(raw, wheelhouse, deadline):
    try:
        import tomllib
        return tomllib.loads(raw.decode())
    except ImportError:
        # Python 3.10 has no stdlib TOML parser. Bootstrap only a hash-bound
        # pure parser from this authenticated lock, never the mutable venv.
        text = raw.decode()
        _require(len(raw) <= 1024*1024 and time.monotonic() <= deadline)
        blocks = re.findall(r'(?m)^\[\[package\]\]\n(.*?)(?=^\[\[package\]\]|\Z)', text, re.S)
        blocks = [block for block in blocks if re.match(r'name = "tomli"\n', block)]
        _require(len(blocks) == 1)
        block = blocks[0]
        version = re.findall(r'(?m)^version = "([0-9.]+)"$', block)
        _require(len(version) == 1 and '\nsource = { registry = "https://pypi.org/simple" }\n' in block)
        wheels = re.findall(r'(?m)^wheels = \[(.*?)\]\s*(?=\n\[|\Z)', block, re.S)
        _require(len(wheels) == 1)
        candidates = re.findall(r'\{ url = "(https://files\.pythonhosted\.org/[^"\\\s]+-py3-none-any\.whl)", hash = "(sha256:[0-9a-f]{64})", size = ([0-9]+)(?:, upload-time = "[^"\n]+")? \}', wheels[0])
        _require(len(candidates) == 1)
        url, digest, size = candidates[0]
        _require(Path(urllib.parse.urlsplit(url).path).name == 'tomli-'+version[0]+'-py3-none-any.whl')
        selected = {'url': url, 'hash': digest, 'size': int(size)}
        path = _sdk_artifact(selected, wheelhouse, deadline)
        rows = _wheel_entries(path, deadline)
        _require('tomli/__init__.py' in rows)
        root = _sdk_root() / 'sdk-tools' / digest[7:]
        _sdk_extract(root, rows, deadline)
        prefix = '_blueprint_verified_sdk_tomli_' + digest[7:]
        _require(not any(name == prefix or name.startswith(prefix+'.') for name in sys.modules))
        spec = importlib.util.spec_from_file_location(prefix, root/'tomli/__init__.py',
            submodule_search_locations=[str(root/'tomli')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[prefix] = module
        try:
            spec.loader.exec_module(module)
            lock = module.loads(text)
            parsers = [row for row in lock['package'] if row.get('name') == 'tomli']
            _require(len(parsers) == 1 and parsers[0]['version'] == version[0]
                     and parsers[0].get('source') == {'registry': 'https://pypi.org/simple'}
                     and any(all(row.get(key) == value for key, value in selected.items())
                             for row in parsers[0].get('wheels', ()))
                     and time.monotonic() <= deadline)
            return lock
        finally:
            for name in tuple(sys.modules):
                if name == prefix or name.startswith(prefix+'.'):
                    del sys.modules[name]


def build_sdk(source, *, wheelhouse=None, contracts_checkout=None, _deadline=None):
    """Build the locked base production closure for this system ABI, no setup.py."""
    deadline = _installation_deadline(_deadline)
    try:
        source = Path(source)
        raw, _ = _record_bytes(source / 'uv.lock', deadline)
        lock = _sdk_toml(raw, wheelhouse, deadline)
        packages = lock['package']
        _require(type(packages) is list and 0 < len(packages) <= 4096)
        tools = _sdk_marker_tools(packages, wheelhouse, deadline)
        selected = _sdk_closure(packages, tools)
        rows = {}
        for package in selected:
            _require(time.monotonic() <= deadline)
            if 'git' in package.get('source', {}):
                entries = _sdk_git_rows(package, contracts_checkout, deadline)
            else:
                path = _sdk_artifact(_sdk_wheel(package, tools), wheelhouse, deadline)
                entries = _wheel_entries(path, deadline)
            for name, row in entries.items():
                _require(name not in rows)
                rows[name] = row
        manifest = {name: {key: row[key] for key in ('size', 'sha256', 'mode')}
                    for name, row in rows.items()}
        selection = {'schema': 'scene-retirement-sdk.v1', 'lock_sha256': hashlib.sha256(raw).hexdigest(),
                     'system_python_abi': f'{sys.version_info.major}.{sys.version_info.minor}',
                     'packages': [{'name': row['name'], 'version': row['version']} for row in selected],
                     'rows': manifest}
        raw_selection = _encoded(selection)
        digest = hashlib.sha256(raw_selection).hexdigest()
        root = _sdk_root() / 'sdk-inputs' / digest
        marker = _sdk_root() / 'manifests' / ('sdk-input-' + digest + '.json')
        if root.exists() or root.is_symlink():
            _require(marker.exists() and _record_bytes(marker, deadline)[0] == raw_selection)
        _mkdir(marker.parent)
        _record(marker, raw_selection, deadline)
        _sdk_extract(root, rows, deadline)
        actual, sources = {}, {}
        _tree(root, Path('.'), actual, sources, deadline)
        _require(actual == manifest)
        return {'dependencies_root': str(root), 'packages': selection['packages'],
                'system_python_abi': selection['system_python_abi'], 'authority_issued': False,
                'cleanup_enabled': False}
    except (OSError, ValueError, KeyError, TypeError, ImportError, zipfile.BadZipFile) as exc:
        raise ValueError(_ERROR) from exc


def _signed_release_blobs(source, items, deadline):
    """Acquire a small batch; authenticate each bounded blob independently."""
    _require(0 < len(items) <= 16 and len({name for _, _, name in items}) == len(items))
    request = b''.join(digest.encode('ascii') + b'\n' for _, digest, _ in items)
    checked = _sdk_git_command(source, ['cat-file', '--batch-check'], deadline,
                               cap=len(items) * 80, raw_checkout=True, input_data=request)
    lines = checked.split(b'\n')
    _require(len(lines) == len(items) + 1 and lines[-1] == b'')
    headers, sizes = [], []
    for (_, digest, name), line in zip(items, lines):
        fields = line.split(b' ')
        _require(len(fields) == 3 and fields[:2] == [digest.encode('ascii'), b'blob']
                 and fields[2].isdigit())
        size = int(fields[2])
        limit = 16 * 1024**2 if name == 'uv.lock' else 1024 * 1024
        _require(0 <= size <= limit and fields[2] == str(size).encode('ascii'))
        headers.append(line + b'\n')
        sizes.append(size)
    cap = sum(len(header) + size + 1 for header, size in zip(headers, sizes))
    body = _sdk_git_command(source, ['cat-file', '--batch'], deadline,
                            cap=cap, raw_checkout=True, input_data=request)
    offset, result = 0, {}
    for (_, digest, name), header, size in zip(items, headers, sizes):
        _require(time.monotonic() <= deadline and body[offset:offset+len(header)] == header)
        offset += len(header)
        raw = body[offset:offset+size]
        offset += size
        _require(len(raw) == size and body[offset:offset+1] == b'\n'
                 and hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + raw).hexdigest() == digest)
        offset += 1
        _require(name not in result)
        result[name] = raw
    _require(offset == len(body))
    return result


def _signed_release(source, commit, deadline):
    """Copy only authenticated Git object bytes, never mutable checkout code."""
    _require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit))
    source = Path(source)
    items = _authenticated_git_entries(source, commit, ('src/blueprint_pipeline', 'scripts',
              'deploy/systemd', 'uv.lock', 'pyproject.toml'), deadline, raw_checkout=True)
    _require(items)
    output = _encoded(items)
    root = _sdk_root() / 'release-inputs' / commit
    claim = root.parent / (commit + '.claim.json')
    selected = _encoded({'schema': 'scene-retirement-signed-release.v1', 'commit': commit,
                         'tree_sha256': hashlib.sha256(output).hexdigest()})
    _mkdir(root.parent)
    if root.exists() or root.is_symlink():
        _require(claim.exists() and _record_bytes(claim, deadline)[0] == selected)
    _record(claim, selected, deadline)
    _mkdir(root)
    total, paths = 0, set()
    for index, (mode, digest, name) in enumerate(items):
        if index % 16 == 0:
            missing = [item for item in items[index:index+16]
                       if not (root / item[2]).exists() and not (root / item[2]).is_symlink()]
            blobs = _signed_release_blobs(source, missing, deadline) if missing else {}
        _require(name not in paths)
        paths.add(name)
        # The lock is bounded data; source modules retain the tighter cap.
        blob_cap = 16 * 1024**2 if name == 'uv.lock' else 1024 * 1024
        target = root / name
        if target.exists() or target.is_symlink():
            # The authenticated tree supplies the expected Git blob ID. Prove
            # retained bytes against it instead of spawning two Git processes
            # for every already-retained source file on a bounded retry.
            body, _ = _record_bytes(target, deadline, cap=blob_cap, allow_empty=True)
        else:
            body = blobs[name]
            _require(len(body) <= blob_cap)
        total += len(body)
        _require(total <= _MAX_BYTES
                 and hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest() == digest)
        _mkdir(target.parent)
        _record(target, body, deadline, allow_empty=True)
        if mode == '100755':
            # Root-authenticated scripts are copied as data; no SDK package or
            # checkout hook is executed by preparation.
            _require(_read(target, deadline)['sha256'] == hashlib.sha256(body).hexdigest())
    _require({'uv.lock', 'src/blueprint_pipeline/__init__.py',
              'scripts/scene_retirement_continuous_bootstrap.py'} <= paths)
    return root


def _publish_installer(source, deadline):
    raw, _ = _record_bytes(source / 'scripts/install_scene_retirement_runtime.py', deadline, cap=1024*1024)
    row = _selector(raw)
    record = _BOOT_ROOT / 'runtime-installer.json'
    pending = _BOOT_ROOT / 'runtime-installer-pending.json'
    target = _BOOT_ROOT / 'runtime_installer.py'
    value = _encoded({'schema': 'scene-retirement-runtime-installer.v1', **row})
    previous_record, previous_target = None, None
    if record.exists() or record.is_symlink():
        old, previous_record = _record_bytes(record, deadline, cap=4096)
        old_value = json.loads(old)
        _require(type(old_value) is dict and set(old_value) == {'schema', 'sha256', 'size_bytes'}
                 and old_value['schema'] == 'scene-retirement-runtime-installer.v1')
        current, previous_target = _record_bytes(target, deadline, cap=1024*1024)
        if _selector(current) != {key: old_value[key] for key in ('sha256', 'size_bytes')}:
            _require(pending.exists() and _record_bytes(pending, deadline)[0] == value
                     and _selector(current) == row)
    else:
        _require(not target.exists() and not target.is_symlink()
                 or (pending.exists() and _record_bytes(pending, deadline)[0] == value
                     and _selector(_record_bytes(target, deadline)[0]) == row))
        if target.exists():
            _, previous_target = _record_bytes(target, deadline)
    old_pending = _record_bytes(pending, deadline)[1] if pending.exists() or pending.is_symlink() else None
    _record(pending, value, deadline, previous=old_pending)
    _record(target, raw, deadline, previous=previous_target)
    _record(record, value, deadline, previous=previous_record)
    _require(_record_bytes(target, deadline)[0] == raw and _record_bytes(record, deadline)[0] == value)


def _resume_initial_intent(dependencies, deadline):
    """Finish the original protected copy plan; historical source is only data."""
    raw, _ = _record_bytes(_BOOT_ROOT / 'installation.json', deadline)
    plan = json.loads(raw)
    _require(type(plan) is dict and set(plan) == {'schema', 'runtime', 'rows', 'bootstrap'}
             and plan['schema'] == 'scene-retirement-runtime-install.v1'
             and plan['runtime'] == str(_RUNTIME_ROOT) and type(plan['rows']) is dict
             and all(type(name) is str for name in plan['rows']) and _encoded(plan) == raw)
    source_rows = {name: row for name, row in plan['rows'].items() if not name.startswith('dependencies/')}
    sdk_rows = {name.removeprefix('dependencies/'): row for name, row in plan['rows'].items()
                if name.startswith('dependencies/')}
    helper = 'scripts/install_scene_retirement_runtime.py'
    _require(helper in source_rows)
    inputs = _sdk_root()
    source_parent = inputs / 'release-inputs'
    fd = _open(source_parent, directory=True)
    try:
        names = sorted(os.listdir(fd))
        _require(len(names) <= _MAX_FILES)
    finally:
        os.close(fd)
    original_source = None
    for name in names:
        _require(time.monotonic() <= deadline)
        if not re.fullmatch('[0-9a-f]{40}', name):
            continue
        candidate = source_parent / name
        try:
            if _read(candidate / helper, deadline) != source_rows[helper]:
                continue
            observed = {}
            for directory in ('src/blueprint_pipeline', 'deploy/systemd', 'scripts'):
                _tree(candidate / directory, Path(directory), observed, {}, deadline)
            if (observed == source_rows
                    and _read(candidate / 'scripts/scene_retirement_continuous_bootstrap.py', deadline) == plan['bootstrap']):
                original_source = candidate
                break
        except FileNotFoundError:
            # An unrelated retained source snapshot may itself be incomplete.
            continue
    _require(original_source is not None)

    # Prefer the current SDK. Discover older ones only by protected manifests
    # under the fixed input root, never by arbitrary external paths.
    preferred_name = 'sdk-input-' + Path(dependencies).name + '.json'
    manifest_parent = inputs / 'manifests'
    fd = _open(manifest_parent, directory=True)
    try:
        names = sorted(os.listdir(fd))
        _require(len(names) <= _MAX_FILES)
    finally:
        os.close(fd)
    names.sort(key=lambda name: name != preferred_name)
    original_sdk = None
    for name in names:
        _require(time.monotonic() <= deadline)
        match = re.fullmatch('sdk-input-([0-9a-f]{64})[.]json', name)
        if not match:
            continue
        body, _ = _record_bytes(manifest_parent / name, deadline)
        _require(hashlib.sha256(body).hexdigest() == match[1])
        manifest = json.loads(body)
        _require(type(manifest) is dict and manifest.get('schema') == 'scene-retirement-sdk.v1')
        if manifest.get('rows') == sdk_rows:
            original_sdk = inputs / 'sdk-inputs' / match[1]
            break
    _require(original_sdk is not None)
    # prepare hashes the entire source/SDK/bootstrap set and compares canonical
    # intent bytes under its lock before validating/appending any partial file.
    prepare(original_source, original_sdk, _deadline=deadline)


def prepare_deployment(source, *, source_commit, wheelhouse=None, contracts_checkout=None, _deadline=None, _progress=None):
    """Complete root snapshot and ABI SDK before callers expose service units."""
    deadline = _installation_deadline(_deadline)
    def phase(name):
        if _progress is not None:
            _progress(name)
    phase('signed_release')
    protected_source = _signed_release(source, source_commit, deadline)
    phase('build_sdk')
    sdk = build_sdk(protected_source, wheelhouse=wheelhouse, contracts_checkout=contracts_checkout, _deadline=deadline)
    _require(time.monotonic() <= deadline)
    current = _BOOT_ROOT / 'CURRENT.json'
    installed = _BOOT_ROOT / 'installation.json'
    boot = _BOOT_ROOT / 'continuous_bootstrap.py'
    if (not current.exists() and not current.is_symlink()
            and not boot.exists() and not boot.is_symlink()
            and (installed.exists() or installed.is_symlink())):
        phase('resume_initial_intent')
        _resume_initial_intent(Path(sdk['dependencies_root']), deadline)
    if current.exists() or current.is_symlink() or installed.exists() or installed.is_symlink():
        selected = current if current.exists() or current.is_symlink() else installed
        raw, _ = _record_bytes(selected, deadline)
        phase('refresh')
        result = refresh(protected_source, Path(sdk['dependencies_root']), expected_current=_selector(raw), _deadline=deadline)
    else:
        phase('prepare')
        result = prepare(protected_source, Path(sdk['dependencies_root']), _deadline=deadline)
    _require(time.monotonic() <= deadline)
    phase('publish_installer')
    _publish_installer(protected_source, deadline)
    return result | {'source_commit': source_commit, 'sdk_packages': sdk['packages'],
                     'system_python_abi': sdk['system_python_abi']}

def main(argv=None):
    _require(os.getuid() == os.geteuid() == 0 and sys.flags.isolated and sys.flags.no_site)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--source-commit')
    parser.add_argument('--deadline-monotonic', type=float)
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--contracts-checkout', type=Path)
    sdk = parser.add_mutually_exclusive_group(required=True)
    sdk.add_argument('--dependencies', type=Path)
    sdk.add_argument('--venv', type=Path)
    sdk.add_argument('--locked-sdk', action='store_true')
    arguments = parser.parse_args(argv)
    started = time.monotonic()
    def report_phase(name):
        print('scene_retirement_runtime_phase:' + name, file=sys.stderr, flush=True)
    try:
        deadline = _installation_deadline(arguments.deadline_monotonic)
        if arguments.locked_sdk:
            _require(arguments.source_commit is not None)
            result = prepare_deployment(arguments.source, source_commit=arguments.source_commit,
                wheelhouse=arguments.wheelhouse, contracts_checkout=arguments.contracts_checkout,
                _deadline=deadline, _progress=report_phase)
        else:
            report_phase('prepare')
            dependencies = dependency_root(arguments.venv) if arguments.venv else arguments.dependencies
            result = prepare(arguments.source, dependencies, _deadline=deadline)
    except Exception as exc:
        # Never forward exception text or traceback from protected paths or SDK
        # acquisition. These fixed markers are diagnostic only, never proof.
        deadline = min(started + _MAX_SECONDS, arguments.deadline_monotonic) if arguments.deadline_monotonic is not None else started + _MAX_SECONDS
        reason = ('deadline' if time.monotonic() >= deadline else
                  'io' if isinstance(exc, OSError) else
                  'validation' if isinstance(exc, (ValueError, KeyError, TypeError)) else 'unexpected')
        print('scene_retirement_runtime_failure:' + reason, file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
