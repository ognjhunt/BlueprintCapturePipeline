"""Prepare a protected runtime without issuing a grant or enabling cleanup.

Run this stdlib-only installer from a root-owned release, before transferring
the release checkout to the service account. Existing snapshots are immutable; an interrupted installation can resume only
its exact protected copy intent. Unknown or changed bytes are preserved and refused.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time


_OWNER = 0
_RUNTIME_ROOT = Path('/mnt/blueprint-work/scene-retirement-runtime')
_BOOT_ROOT = Path('/usr/lib/blueprint/scene-retirement-runtime')
_FREE_FLOOR = 5 * 1024**3
_ERROR = 'scene_retirement_runtime_unproven'
_MAX_FILES = 32768
_MAX_BYTES = 4 * 1024**3
_MAX_SECONDS = 300
_SKIP = frozenset(('__pycache__', '.git'))


def _require(value):
    if not value:
        raise ValueError(_ERROR)


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


def _read(path, deadline, *, output=None):
    fd = _open(path, directory=False)
    try:
        before = os.fstat(fd)
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
    finally:
        os.close(fd)


def _tree(path, prefix, rows, sources, deadline, depth=0):
    _require(depth <= 32 and time.monotonic() <= deadline)
    fd = _open(path, directory=True)
    try:
        before = os.fstat(fd)
        names = sorted(os.listdir(fd))
        _require(len(names) <= _MAX_FILES)
        for name in names:
            if name in _SKIP or name.endswith(('.pyc', '.pyo')):
                continue
            child = path / name
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                _protected(info, directory=True)
                _tree(child, prefix / name, rows, sources, deadline, depth + 1)
            else:
                _require(len(rows) < _MAX_FILES)
                key = str(prefix / name)
                rows[key] = _read(child, deadline)
                sources[key] = child
                _require(sum(row['size'] for row in rows.values()) <= _MAX_BYTES)
        _require(_identity(os.fstat(fd)) == _identity(before)
                 and _identity(path.lstat()) == _identity(before))
    finally:
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


def _partial_tree(path, rows, sources, deadline, prefix=Path('.'), depth=0):
    """Check the entire incomplete snapshot before completing any copy."""
    _require(depth <= 32 and time.monotonic() <= deadline)
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
                _require(any(value.startswith(key + '/') for value in rows))
                _partial_tree(child, rows, sources, deadline, prefix / name, depth + 1)
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


def prepare(source, dependencies):
    """Copy exact protected inputs; no policy, consent, generation or flag writes."""
    try:
        source, dependencies = Path(source), Path(dependencies)
        deadline = time.monotonic() + _MAX_SECONDS
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


def main(argv=None):
    _require(os.getuid() == os.geteuid() == 0 and sys.flags.isolated and sys.flags.no_site)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    sdk = parser.add_mutually_exclusive_group(required=True)
    sdk.add_argument('--dependencies', type=Path)
    sdk.add_argument('--venv', type=Path)
    arguments = parser.parse_args(argv)
    dependencies = dependency_root(arguments.venv) if arguments.venv else arguments.dependencies
    print(json.dumps(prepare(arguments.source, dependencies), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
