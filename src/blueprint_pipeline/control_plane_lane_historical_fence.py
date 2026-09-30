"""Internal retained generation fence; no decision or reader-clear authority.

The installed action supplies fresh owner/unit checks and durable journal
callbacks. Revocation preserves every byte; old FDs and mappings remain readers.
Only original, single-link members beneath the retained root are supported.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path


class HistoricalFenceError(ValueError):
    """Fixed refusal without original paths or bytes."""


def _require(value, code):
    if not value:
        raise HistoricalFenceError('historical_generation_' + code)


def _version(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_blocks]


def _acl(fd):
    names = os.listxattr(fd)
    _require(len(names) <= 64 and sum(len(os.fsencode(name)) for name in names) <= 4096
             and not any('acl' in name.lower() for name in names), 'acl_unknown')


def _members(manifest):
    rows = manifest.get('members')
    _require(isinstance(rows, list) and type(manifest.get('member_count')) is int
             and 0 < len(rows) == manifest['member_count'] <= 4096, 'manifest_invalid')
    selected = {}
    for row in rows:
        _require(type(row) is dict and isinstance(row.get('path'), str), 'manifest_invalid')
        relative = row['path']
        _require(len(os.fsencode(relative)) <= 1024 and (relative == '' or
            len(relative.split('/')) <= 16 and all(part not in ('', '.', '..')
              and len(os.fsencode(part)) <= 255 for part in relative.split('/')))
            and relative not in selected and row.get('kind') in ('directory', 'file')
            and isinstance(row.get('version'), list) and len(row['version']) == 10
            and all(type(value) is int for value in row['version'])
            and not row['version'][2] & (stat.S_ISUID | stat.S_ISGID), 'manifest_invalid')
        selected[relative] = row
    _require('' in selected and selected['']['kind'] == 'directory'
             and all(name == '' or name.rpartition('/')[0] in selected
                and selected[name.rpartition('/')[0]]['kind'] == 'directory' for name in selected),
             'manifest_invalid')
    return selected


class _HistoricalGenerationFence:
    """Bounded retained ancestry. Caller holds authority, deadline and journal."""

    def __init__(self, manifest, *, tick):
        _require(sys.platform == 'linux' and os.geteuid() == 0, 'native_unavailable')
        self.tick, self.stack = tick, ExitStack()
        self.rows = _members(manifest)
        self.children = {name: set() for name, row in self.rows.items() if row['kind'] == 'directory'}
        for name in self.rows:
            if name:
                self.children[name.rpartition('/')[0]].add(name.rpartition('/')[2])
        self.versions = {name: list(row['version']) for name, row in self.rows.items()}
        self.removed = set()
        self.target = Path(manifest['target_path'])
        _require(self.target.is_absolute() and str(self.target) == manifest['target_path']
            and all(part not in ('', '.', '..') for part in self.target.parts[1:])
            and len(self.target.parts) <= 32, 'manifest_invalid')
        self.chain = []
        try:
            parent = None
            components = ('/', *self.target.parts[1:])
            for index, name in enumerate(components):
                self.tick()
                fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                             dir_fd=parent)
                self.stack.callback(os.close, fd)
                info = os.fstat(fd)
                named = _version(os.stat(name, dir_fd=parent, follow_symlinks=False))
                observed = _version(info)
                _require(observed == named if index >= len(components) - 2
                         else observed[:5] == named[:5], 'fence_changed')
                self.chain.append((parent, name, fd, _version(info)))
                parent = fd
            self.root = self.chain[-1][2]
            _require(len(self.chain) >= 2 and self.chain[-2][3] == manifest['root_version']
                     and self.chain[-1][3] == self.versions[''], 'fence_changed')
            for _, _, fd, version in self.chain[:-1]:
                _require(version[3:5] == [0, 0] and not version[2] & 0o022, 'parent_unsafe')
                _acl(fd)
            fcntl.flock(self.root, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.verify()
        except BaseException:
            self.stack.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stack.close()

    def _chain_guard(self):
        for index, (parent, name, fd, version) in enumerate(self.chain):
            self.tick()
            opened = _version(os.fstat(fd))
            named = _version(os.stat(name, dir_fd=parent, follow_symlinks=False))
            if index < len(self.chain) - 2:
                _require(opened[:5] == version[:5] == named[:5], 'fence_changed')
            else:
                _require(opened == version == named, 'fence_changed')

    def _guard(self, fd, relative, parent=None, name=None):
        self.tick()
        info = os.fstat(fd)
        row = self.rows[relative]
        _require(_version(info) == self.versions[relative]
            and info.st_dev == self.versions[''][0]
            and (stat.S_ISDIR(info.st_mode) if row['kind'] == 'directory'
                 else stat.S_ISREG(info.st_mode)
                 and info.st_nlink == (0 if relative in self.removed else 1)), 'fence_changed')
        if parent is not None:
            if relative in self.removed:
                try:
                    os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    _require(False, 'fence_changed')
            else:
                _require(_version(os.stat(name, dir_fd=parent, follow_symlinks=False)) == _version(info),
                         'fence_changed')
        _acl(fd)

    @contextmanager
    def _opened(self, relative):
        self._chain_guard()
        with ExitStack() as stack:
            parent, fd = None, self.root
            self._guard(fd, '')
            prefix = ''
            held = [(fd, '', None, None)]
            for name in relative.split('/') if relative else ():
                parent = fd
                prefix = prefix + '/' + name if prefix else name
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
                if self.rows[prefix]['kind'] == 'directory':
                    flags |= os.O_DIRECTORY
                fd = os.open(name, flags, dir_fd=parent)
                stack.callback(os.close, fd)
                self._guard(fd, prefix, parent, name)
                held.append((fd, prefix, parent, name))
            def guard():
                self._chain_guard()
                for descriptor, selected, ancestor, leaf in held:
                    self._guard(descriptor, selected, ancestor, leaf)
            yield fd, guard
            guard()

    def verify(self):
        for relative, row in self.rows.items():
            if relative in self.removed:
                continue
            with self._opened(relative) as (fd, guard):
                if row['kind'] == 'directory':
                    names = set()
                    with os.scandir(fd) as entries:
                        for entry in entries:
                            self.tick()
                            _require(len(names) < 4096, 'fence_changed')
                            names.add(entry.name)
                    _require(names == self.children[relative], 'fence_changed')
                guard()

    def sync_directory(self, relative):
        """Durably reconcile the last uncertain dentry without payload effects."""
        _require(relative in self.rows and self.rows[relative]['kind'] == 'directory', 'fence_changed')
        with self._opened(relative) as (fd, guard):
            guard()
            os.fsync(fd)
            guard()

    def revoke(self, *, before_change, record, completed=frozenset(), pending=None):
        """Root first, then every original descendant; never unlink or clear FDs.

        A crash leaves a durable intent and remaining bytes. The complete worker
        must reconcile that intent and original hashes before future actions.
        """
        for relative in sorted(self.rows, key=lambda name: (name.count('/') + bool(name), name)):
            if relative in completed:
                continue
            with self._opened(relative) as (fd, guard):
                original = list(self.versions[relative])
                mode = 0o700 if self.rows[relative]['kind'] == 'directory' else 0o600
                if relative != pending:
                    record('fence_intent', dict(path=relative, version=original, uid=0, gid=0, mode=mode))
                for operation in ('chown', 'chmod'):
                    with before_change():
                        guard()
                        if operation == 'chown':
                            os.fchown(fd, 0, 0)
                        else:
                            os.fchmod(fd, mode)
                        observed = _version(os.fstat(fd))
                        expected = list(original)
                        expected[3:5] = [0, 0]
                        if operation == 'chmod':
                            expected[2] = stat.S_IFMT(original[2]) | mode
                        _require(observed[:8] == expected[:8] and observed[9] == expected[9], 'fence_changed')
                        self.versions[relative] = observed
                        if relative == '':
                            parent, name, held, _ = self.chain[-1]
                            self.chain[-1] = (parent, name, held, observed)
                        guard()
                os.fsync(fd)
                record('fenced', dict(path=relative, version=list(self.versions[relative])))
        self.verify()

    def remove_members(self, *, before_change, record, pending=None):
        """Internal delete: fresh caller-held authority, intent, exact unlink.

        Preserve the named root as a protected tombstone. Every completed row
        has an observed removal; interrupted absence is never credited here.
        """
        logical, allocated, files, directories = 0, 0, 0, 0
        selected = sorted((name for name in self.rows if name),
                          key=lambda name: (name.count('/'), name), reverse=True)
        for relative in selected:
            row = self.rows[relative]
            parent_path, _, name = relative.rpartition('/')
            with self._opened(relative) as (fd, guard):
                original = list(self.versions[relative])
                _require(original[3:5] == [0, 0]
                    and stat.S_IMODE(original[2]) == (0o700 if row['kind'] == 'directory' else 0o600),
                    'fence_changed')
                if row['kind'] == 'file':
                    digest, consumed = hashlib.sha256(), 0
                    while True:
                        guard()
                        block = os.read(fd, 1024**2)
                        self.tick()
                        if not block:
                            break
                        consumed += len(block)
                        _require(consumed <= row['size_bytes'], 'payload_changed')
                        digest.update(block)
                    _require(consumed == row['size_bytes'] and
                        'sha256:' + digest.hexdigest() == row['sha256'], 'payload_changed')
                else:
                    with os.scandir(fd) as entries:
                        _require(next(entries, None) is None, 'fence_changed')
                if relative != pending:
                    record('removal_intent', dict(path=relative, kind=row['kind'], version=original,
                        sha256=row['sha256'], size_bytes=row['size_bytes']))
                with self._opened(parent_path) as (parent, parent_guard):
                    with before_change():
                        guard()
                        parent_guard()
                        if row['kind'] == 'file':
                            os.unlink(name, dir_fd=parent)
                        else:
                            os.rmdir(name, dir_fd=parent)
                        self.removed.add(relative)
                        self.children[parent_path].remove(name)
                        current = _version(os.fstat(fd))
                        _require(current[:5] == original[:5] and current[5] == 0,
                                 'removal_uncertain')
                        if row['kind'] == 'file':
                            _require(current[6:8] == original[6:8]
                                     and current[9] == original[9], 'removal_uncertain')
                        self.versions[relative] = current
                        parent_before = self.versions[parent_path]
                        parent_after = _version(os.fstat(parent))
                        _require(parent_after[:5] == parent_before[:5]
                            and parent_after[5] == parent_before[5] - int(row['kind'] == 'directory'),
                            'removal_uncertain')
                        self.versions[parent_path] = parent_after
                        if parent_path == '':
                            ancestor, leaf, root, _ = self.chain[-1]
                            self.chain[-1] = (ancestor, leaf, root, parent_after)
                        guard()
                        parent_guard()
                        os.fsync(parent)
            # This invocation's only open leaf descriptor has now closed.
            amount = original[9] * 512
            allocated += amount
            logical += row['size_bytes']
            files += int(row['kind'] == 'file')
            directories += int(row['kind'] == 'directory')
            record('removed', dict(path=relative, kind=row['kind'], physical_identity=original[:2],
                logical_bytes=row['size_bytes'], observed_removed_allocated_bytes=amount,
                parent_path=parent_path, parent_version=list(self.versions[parent_path])))
        self.verify()
        return dict(removed_files=files, removed_directories=directories, logical_bytes=logical,
            observed_removed_allocated_bytes=allocated, uncertain_removed_allocated_bytes=0,
            tombstone_version=list(self.versions['']), root_directory_retained=True)
