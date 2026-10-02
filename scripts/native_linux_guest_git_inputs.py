"""Credential-free committed Git object inputs for the private Plan11 guest.

Dirty working files and Git configuration/hooks/credential metadata are never
transported. This records an immutable input, not native execution acceptance.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack, contextmanager
import math
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import time

from scripts.native_linux_guest_execution import (GuestExecutionError, _bounded_command,
    _check_ancestry, _directory_identity, _file_identity, _open_directory)

MAX_PACK_BYTES = 2 * 1024**3
MAX_TREE_BYTES = 2 * 1024**3
MAX_TREE_FILES = 100000
MAX_METADATA_BYTES = 16 * 1024**2


class GitInputError(ValueError):
    """Exact input selection or transport could not be established."""


def _require(value, code):
    if not value:
        raise GitInputError('native_guest_git_' + code)


def _remaining(deadline):
    _require(type(deadline) in {int, float} and math.isfinite(deadline), 'deadline')
    remaining = deadline - time.monotonic()
    _require(remaining > 0, 'deadline')
    return remaining


def _environment():
    return {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_NO_LAZY_FETCH': '1',
            'GIT_ALLOW_PROTOCOL': '', 'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_REPLACE_OBJECTS': '1'}


def _command(root, arguments):
    return ['git', '--no-optional-locks', '-c', 'core.hooksPath=/dev/null',
            '-c', 'core.fsmonitor=false', '-C', str(root), *arguments]


def _git(root, arguments, deadline, *, stdin=None, git_dir=None, pass_fds=()):
    try:
        _remaining(deadline)
        env = _environment()
        if git_dir is not None:
            env.update(GIT_DIR=str(git_dir), GIT_WORK_TREE=str(root))
        result = _bounded_command(_command(root, arguments), stdin=stdin,
                                  deadline=deadline, max_output_bytes=MAX_METADATA_BYTES,
                                  env=env, pass_fds=pass_fds)
        _require(result.returncode == 0 and len(result.stdout) <= MAX_METADATA_BYTES
                 and len(result.stderr) <= 65536, 'command_failed')
        return result.stdout
    except (OSError, subprocess.TimeoutExpired, GuestExecutionError):
        raise GitInputError('native_guest_git_command_failed') from None


def _oid(value):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{40}', value), 'identity')
    return value


def _path(path):
    path = Path(path)
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'path')
    return path


def _validate_pack(fd, path, before, deadline):
    _require(_file_identity(os.fstat(fd)) == _file_identity(before)
             and _file_identity(path.lstat()) == _file_identity(before), 'pack')
    _remaining(deadline)


def _hash_open_pack(fd, path, before, deadline):
    digest, size = hashlib.sha256(), 0
    _validate_pack(fd, path, before, deadline)
    while True:
        _remaining(deadline)
        raw = os.read(fd, 1024**2)
        if not raw:
            break
        size += len(raw)
        _require(size <= MAX_PACK_BYTES, 'pack')
        digest.update(raw)
    _require(size == before.st_size, 'pack')
    _validate_pack(fd, path, before, deadline)
    return size, digest.hexdigest()


@contextmanager
def _verified_pack(path, deadline):
    _remaining(deadline)
    try:
        with ExitStack() as stack:
            ancestry = []
            parent = _open_directory(path.parent, ancestry)
            stack.callback(os.close, parent)
            fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent)
            stream = stack.enter_context(os.fdopen(fd, 'rb', buffering=0))
            before = os.fstat(fd)
            _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                     and 0 < before.st_size <= MAX_PACK_BYTES, 'pack')
            size, digest = _hash_open_pack(fd, path, before, deadline)
            stream.seek(0)
            _check_pack_namespace(path, ancestry, deadline)
            # Callers perform their last pack and namespace guards while all
            # descriptors are held. Exit only closes descriptors; it must not
            # introduce later pathname work after a caller's final closure.
            yield stream, before, size, digest, ancestry
    except OSError:
        raise GitInputError('native_guest_git_pack') from None


def _pack_hash(path, deadline):
    with _verified_pack(path, deadline) as (stream, before, size, digest, ancestry):
        _validate_pack(stream.fileno(), path, before, deadline)
        _check_pack_namespace(path, ancestry, deadline)
        return size, digest


def _check_pack_namespace(path, ancestry, deadline):
    try:
        _check_ancestry(path.parent, ancestry, 'git_pack')
    except GuestExecutionError:
        raise GitInputError('native_guest_git_pack') from None
    _remaining(deadline)


def _read_shallow(path, deadline):
    _remaining(deadline)
    try:
        with ExitStack() as stack:
            ancestry = []
            parent = _open_directory(path.parent, ancestry)
            stack.callback(os.close, parent)
            fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent)
            stack.callback(os.close, fd)
            before = os.fstat(fd)
            _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                     and 0 < before.st_size <= 1024**2, 'shallow')
            _require(_file_identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False))
                     == _file_identity(before), 'shallow')
            raw = bytearray()
            while True:
                _remaining(deadline)
                part = os.read(fd, min(65536, 1024**2 + 1 - len(raw)))
                if not part:
                    break
                raw.extend(part)
                _require(len(raw) <= 1024**2, 'shallow')
            _require(len(raw) == before.st_size
                     and _file_identity(os.fstat(fd)) == _file_identity(before)
                     == _file_identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False)), 'shallow')
            _check_ancestry(path.parent, ancestry, 'git_shallow')
            _remaining(deadline)
            return bytes(raw)
    except (OSError, GuestExecutionError):
        raise GitInputError('native_guest_git_shallow') from None


def _tree_footprint(root, commit, deadline):
    return _tree_footprint_from_rows(_git(root, ['ls-tree', '-rlz', commit], deadline))


def _tree_footprint_from_rows(raw):
    rows = raw.split(b'\0')
    count = size = 0
    for row in rows:
        if not row:
            continue
        fields, separator, name = row.partition(b'\t')
        fields = fields.split()
        _require(separator and len(fields) == 4 and fields[1] == b'blob'
                 and fields[3].isdigit() and name and not name.startswith(b'/')
                 and not any(p in {b'..', b'.git'} for p in name.split(b'/')), 'tree')
        size += int(fields[3])
        count += 1
        _require(size <= MAX_TREE_BYTES and count <= MAX_TREE_FILES, 'tree')
    _require(count > 0, 'tree')
    return size, count


def write_git_input(checkout, expected_commit, output, *, deadline_monotonic):
    checkout, output = _path(checkout), _path(output)
    commit = _oid(expected_commit)
    _require(_git(checkout, ['rev-parse', 'HEAD'], deadline_monotonic).decode().strip() == commit,
             'selection')
    tree = _oid(_git(checkout, ['rev-parse', commit + '^{tree}'], deadline_monotonic).decode().strip())
    shallow = []
    if _git(checkout, ['rev-parse', '--is-shallow-repository'], deadline_monotonic).strip() == b'true':
        name = _git(checkout, ['rev-parse', '--git-path', 'shallow'], deadline_monotonic).decode().strip()
        path = Path(name)
        path = path if path.is_absolute() else checkout / path
        raw = _read_shallow(_path(path), deadline_monotonic)
        _require(0 < len(raw) <= 1024**2, 'shallow')
        shallow = [_oid(row) for row in raw.decode('ascii').splitlines()]
        _require(len(shallow) == len(set(shallow)), 'shallow')
    size, count = _tree_footprint(checkout, commit, deadline_monotonic)
    process = None
    try:
        with ExitStack() as stack:
            ancestry = []
            parent = _open_directory(output.parent, ancestry)
            stack.callback(os.close, parent)
            fd = os.open(output.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=parent)
            stream = stack.enter_context(os.fdopen(fd, 'r+b', buffering=0))
            created = _file_identity(os.fstat(fd))[:6]
            transferred_digest = hashlib.sha256()
            with selectors.DefaultSelector() as selector:
                try:
                    _remaining(deadline_monotonic)
                    process = subprocess.Popen(_command(checkout, ['pack-objects', '--stdout', '--revs']),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        env=_environment(), close_fds=True)
                    process.stdin.write((commit + '\n').encode())
                    process.stdin.close()
                    selector.register(process.stdout, selectors.EVENT_READ, 'pack')
                    selector.register(process.stderr, selectors.EVENT_READ, 'errors')
                    transferred = errors = 0
                    while selector.get_map():
                        for key, _ in selector.select(timeout=min(1, _remaining(deadline_monotonic))):
                            raw = os.read(key.fileobj.fileno(), 65536)
                            if not raw:
                                selector.unregister(key.fileobj)
                                continue
                            if key.data == 'errors':
                                errors += len(raw)
                                _require(errors <= 65536, 'pack')
                                continue
                            transferred += len(raw)
                            _require(transferred <= MAX_PACK_BYTES, 'pack')
                            _require(stream.write(raw) == len(raw), 'pack')
                            transferred_digest.update(raw)
                    process.wait(timeout=_remaining(deadline_monotonic))
                    _require(process.returncode == 0, 'pack')
                finally:
                    if process is not None:
                        try:
                            if process.poll() is None:
                                process.kill()
                            process.wait(timeout=5)
                        finally:
                            try:
                                process.stdout.close()
                            finally:
                                try:
                                    process.stderr.close()
                                finally:
                                    if not process.stdin.closed:
                                        process.stdin.close()
            before = os.fstat(fd)
            _require(_file_identity(before)[:6] == created, 'pack')
            stream.seek(0)
            pack_size, digest = _hash_open_pack(fd, output, before, deadline_monotonic)
            _require(pack_size == transferred and digest == transferred_digest.hexdigest(), 'pack')
            _require(_git(checkout, ['rev-parse', 'HEAD'], deadline_monotonic).decode().strip() == commit, 'selection')
            _validate_pack(fd, output, before, deadline_monotonic)
            _check_pack_namespace(output, ancestry, deadline_monotonic)
            _remaining(deadline_monotonic)
            return dict(schema='native-guest-git-input.v1', commit=commit, tree=tree,
                        shallow=shallow, pack_size_bytes=pack_size, pack_sha256=digest,
                        working_tree_bytes=size, working_tree_files=count)
    except (OSError, subprocess.TimeoutExpired):
        raise GitInputError('native_guest_git_pack') from None


def restore_git_input(pack, manifest, destination, *, deadline_monotonic):
    """Reconstruct a fresh private view, never an existing checkout."""
    _require(sys.platform == 'linux', 'linux_runtime_required')
    pack, destination = _path(pack), _path(destination)
    _require(type(manifest) is dict and manifest.get('schema') == 'native-guest-git-input.v1', 'manifest')
    commit, tree = _oid(manifest.get('commit')), _oid(manifest.get('tree'))
    shallow = manifest.get('shallow')
    _require(type(shallow) is list and len(shallow) <= 25000
             and len(set(_oid(row) for row in shallow)) == len(shallow), 'shallow')
    with _verified_pack(pack, deadline_monotonic) as (stream, before, size, digest, pack_ancestry), ExitStack() as stack:
        _require((size, digest) == (manifest.get('pack_size_bytes'), manifest.get('pack_sha256')), 'pack')
        ancestry = []
        parent = _open_directory(destination.parent, ancestry)
        stack.callback(os.close, parent)
        try:
            os.mkdir(destination.name, mode=0o755, dir_fd=parent)
        except FileExistsError:
            raise GitInputError('native_guest_git_destination_exists') from None
        directory = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=parent)
        stack.callback(os.close, directory)
        ancestry.append(_directory_identity(os.fstat(directory)))
        # Git builtins receive the held private descriptors. No mutable named
        # working-tree or .git path is reopened for their writes.
        bound = Path(f'/proc/{os.getpid()}/fd/{directory}')
        git_fd = None
        git_identity = None
        shallow_fd = None
        shallow_identity = None
        def validate_destination():
            try:
                _check_ancestry(destination, ancestry, 'git_destination_changed')
                if git_fd is not None:
                    _require(_directory_identity(os.fstat(git_fd)) == git_identity
                             == _directory_identity(os.stat('.git', dir_fd=directory, follow_symlinks=False)),
                             'destination_changed')
                if shallow_fd is not None:
                    _require(_file_identity(os.fstat(shallow_fd)) == shallow_identity
                             == _file_identity(os.stat('shallow', dir_fd=git_fd, follow_symlinks=False)), 'shallow')
            except (OSError, GuestExecutionError):
                raise GitInputError('native_guest_git_destination_changed') from None
            _remaining(deadline_monotonic)
        def command(arguments, *, stdin=None):
            validate_destination()
            result = _git(bound, arguments, deadline_monotonic, stdin=stdin,
                          git_dir=None if git_fd is None else Path(f'/proc/{os.getpid()}/fd/{git_fd}'),
                          pass_fds=(directory,) if git_fd is None else (directory, git_fd))
            validate_destination()
            return result
        command(['init', '-q', '--template='])
        try:
            git_fd = os.open('.git', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                             dir_fd=directory)
        except OSError:
            raise GitInputError('native_guest_git_destination_changed') from None
        stack.callback(os.close, git_fd)
        git_identity = _directory_identity(os.fstat(git_fd))
        if shallow:
            try:
                shallow_fd = os.open('shallow', os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     0o600, dir_fd=git_fd)
            except OSError:
                raise GitInputError('native_guest_git_shallow') from None
            shallow_stream = stack.enter_context(os.fdopen(shallow_fd, 'r+b', buffering=0))
            raw = ('\n'.join(shallow) + '\n').encode('ascii')
            _require(shallow_stream.write(raw) == len(raw), 'shallow')
            os.fsync(shallow_fd)
            shallow_identity = _file_identity(os.fstat(shallow_fd))
        _validate_pack(stream.fileno(), pack, before, deadline_monotonic)
        command(['index-pack', '--stdin'], stdin=stream)
        _validate_pack(stream.fileno(), pack, before, deadline_monotonic)
        actual_tree = command(['rev-parse', commit + '^{tree}']).decode().strip()
        _require(actual_tree == tree, 'tree')
        # Recompute the complete selected tree under the same bound view.
        footprint = _tree_footprint_from_rows(command(['ls-tree', '-rlz', commit]))
        _require(type(manifest.get('working_tree_bytes')) is int
                 and type(manifest.get('working_tree_files')) is int
                 and footprint == (manifest['working_tree_bytes'], manifest['working_tree_files']), 'tree_footprint')
        command(['checkout', '--detach', commit])
        _require(command(['rev-parse', 'HEAD']).decode().strip() == commit, 'selection')
        _validate_pack(stream.fileno(), pack, before, deadline_monotonic)
        validate_destination()
        _check_pack_namespace(pack, pack_ancestry, deadline_monotonic)
        _remaining(deadline_monotonic)
        return dict(commit=commit, tree=tree, credentials_transferred=False,
                    guest_acceptance_proven=False)
