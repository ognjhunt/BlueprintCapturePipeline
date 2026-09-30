"""Retained private restore stage and no-replace publication on one tombstone.

All new rows are actual observations of this operation's own syscalls. The
original historical manifest remains immutable and is never relabeled as birth.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import stat
from contextlib import contextmanager

from .control_plane_lane_historical_fence import _version
from .control_plane_lane_historical_generation import _require


class RestoreTree:
    def __init__(self, held, worker, manifest):
        self.held, self.worker, self.manifest = held, worker, manifest
        self.name = '.historical-restore-' + worker.action_id
        _require(not any(row['path'].split('/')[0] == self.name for row in manifest['members']),
                 'restore_stage_collision')

    def _parent_changed(self, fd, relative, *, delta):
        before, after = self.held.versions[relative], _version(os.fstat(fd))
        _require(after[:5] == before[:5] and after[5] == before[5] + delta
            and after[7] >= before[7] and after[8] >= before[8], 'restore_tree_changed')
        self.held.versions[relative] = after
        if relative == '':
            ancestor, leaf, root, _ = self.held.chain[-1]
            self.held.chain[-1] = (ancestor, leaf, root, after)

    def _add(self, fd, relative, row):
        version = _version(os.fstat(fd))
        _require(version[0] == self.held.versions[''][0] and version[3:5] == [0, 0]
            and version[2] == (stat.S_IFDIR | 0o700 if row['kind'] == 'directory' else stat.S_IFREG | 0o600)
            and (row['kind'] == 'directory' or version[5] == 1), 'restore_tree_changed')
        self.held.rows[relative] = dict(row, path=relative, version=version)
        self.held.versions[relative] = version
        parent, _, name = relative.rpartition('/')
        self.held.children[parent].add(name)
        if row['kind'] == 'directory':
            self.held.children[relative] = set()

    def directory(self, row):
        relative = self.name + ('/' + row['path'] if row['path'] else '')
        parent_path, _, name = relative.rpartition('/')
        self.worker.record('restore_intent', dict(phase='directory', path=row['path'], stage_path=relative))
        with self.held._opened(parent_path) as (parent, guard):
            with self.worker.mutation_authority(readers=True):
                guard()
                os.mkdir(name, mode=0o700, dir_fd=parent)
                descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
                try:
                    self._add(descriptor, relative, row)
                    self._parent_changed(parent, parent_path, delta=1)
                    os.fsync(descriptor)
                    os.fsync(parent)
                    guard()
                finally:
                    os.close(descriptor)
        self.worker.record('restore_directory', dict(path=row['path'], stage_path=relative,
            version=self.held.versions[relative], parent_path=parent_path,
            parent_version=self.held.versions[parent_path]))

    @contextmanager
    def member(self, row):
        relative = self.name + '/' + row['path']
        parent_path, _, name = relative.rpartition('/')
        self.worker.record('restore_intent', dict(phase='member', path=row['path'], stage_path=relative,
                                                 sha256=row['sha256'], size_bytes=row['size_bytes']))
        with self.held._opened(parent_path) as (parent, parent_guard):
            with self.worker.mutation_authority(readers=True):
                parent_guard()
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=parent)
                try:
                    self._add(fd, relative, row)
                    self._parent_changed(parent, parent_path, delta=0)
                    parent_guard()
                except BaseException:
                    os.close(fd)
                    raise
            written = 0
            tree = self
            class Sink:
                def write(self, payload):
                    nonlocal written
                    _require(isinstance(payload, bytes) and 0 < len(payload) <= 1024**2
                        and written + len(payload) <= row['size_bytes'], 'restore_write_invalid')
                    offset = 0
                    while offset < len(payload):
                        with tree.worker.mutation_authority(readers=True):
                            parent_guard()
                            tree.held._guard(fd, relative, parent, name)
                            before = tree.held.versions[relative]
                            count = os.write(fd, memoryview(payload)[offset:])
                            _require(type(count) is int and 0 < count <= len(payload) - offset,
                                     'restore_write_failed')
                            written += count
                            offset += count
                            after = _version(os.fstat(fd))
                            _require(after[:6] == before[:6] and after[6] == written
                                and after[7] >= before[7] and after[8] >= before[8], 'restore_tree_changed')
                            tree.held.versions[relative] = after
                            tree.held._guard(fd, relative, parent, name)
                    return len(payload)
            try:
                yield Sink()
                _require(written == row['size_bytes'], 'restore_write_failed')
                with self.worker.mutation_authority(readers=True):
                    self.held._guard(fd, relative, parent, name)
                    os.fsync(fd)
                    os.fsync(parent)
                    parent_guard()
            finally:
                os.close(fd)
        self.worker.record('restore_member', dict(path=row['path'], stage_path=relative,
            version=self.held.versions[relative], sha256=row['sha256'], size_bytes=row['size_bytes'],
            parent_path=parent_path, parent_version=self.held.versions[parent_path]))

    def verify_bytes(self, *, staged):
        self.held.verify()
        for row in self.manifest['members']:
            if row['kind'] != 'file':
                continue
            relative = self.name + '/' + row['path'] if staged else row['path']
            digest, size = hashlib.sha256(), 0
            with self.held._opened(relative) as (fd, guard):
                while True:
                    with self.worker.mutation_authority(readers=True):
                        guard()
                        block = os.read(fd, 1024**2)
                        guard()
                    if not block:
                        break
                    size += len(block)
                    _require(size <= row['size_bytes'], 'restore_payload_changed')
                    digest.update(block)
            _require(size == row['size_bytes'] and 'sha256:' + digest.hexdigest() == row['sha256'],
                     'restore_payload_changed')

    def publish(self):
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        tops = sorted(row['path'] for row in self.manifest['members'] if row['path'] and '/' not in row['path'])
        for name in tops:
            old = self.name + '/' + name
            is_directory = self.held.rows[old]['kind'] == 'directory'
            self.worker.record('restore_intent', dict(phase='publish', path=name,
                stage_version=self.held.versions[self.name], target_version=self.held.versions[''],
                member_version=self.held.versions[old]))
            with self.held._opened(self.name) as (stage, stage_guard):
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                             dir_fd=stage)
                try:
                    with self.worker.mutation_authority(readers=True):
                        stage_guard()
                        self.held._guard(fd, old, stage, name)
                        _require(rename(stage, os.fsencode(name), self.held.root, os.fsencode(name), 1) == 0,
                                 'restore_destination_exists_or_rename_failed')
                        renamed = [path for path in self.held.rows if path == old or path.startswith(old + '/')]
                        for path in renamed:
                            new = path[len(self.name) + 1:]
                            self.held.rows[new] = dict(self.held.rows.pop(path), path=new)
                            self.held.versions[new] = self.held.versions.pop(path)
                            if path in self.held.children:
                                self.held.children[new] = self.held.children.pop(path)
                        before, after = self.held.versions[name], _version(os.fstat(fd))
                        _require(after[:8] == before[:8] and after[9] == before[9] and after[8] >= before[8],
                                 'restore_tree_changed')
                        self.held.versions[name] = after
                        self.held.children[self.name].remove(name)
                        self.held.children[''].add(name)
                        self._parent_changed(stage, self.name, delta=-int(is_directory))
                        self._parent_changed(self.held.root, '', delta=int(is_directory))
                        os.fsync(stage)
                        os.fsync(self.held.root)
                        stage_guard()
                        self.held._guard(fd, name, self.held.root, name)
                finally:
                    os.close(fd)
        with self.held._opened(self.name) as (fd, guard):
            with self.worker.mutation_authority(readers=True):
                guard()
                _require(not self.held.children[self.name], 'restore_tree_changed')
                os.rmdir(self.name, dir_fd=self.held.root)
                self.held.removed.add(self.name)
                self.held.children[''].remove(self.name)
                self.held.versions[self.name] = _version(os.fstat(fd))
                self._parent_changed(self.held.root, '', delta=-1)
                os.fsync(self.held.root)
                guard()
        self.worker.record('restore_intent', dict(phase='stage_removed', target_version=self.held.versions['']))
        self.held.verify()

    def owner_rights(self):
        for row in sorted(self.manifest['members'], key=lambda row: (-row['path'].count('/'), row['path'])):
            if not row['path']:
                continue
            self._rights(row)

    def _rights(self, row):
        relative = row['path']
        version = row['version']
        self.worker.record('restore_intent', dict(phase='owner_rights', path=relative,
            version=self.held.versions[relative], uid=version[3], gid=version[4], mode=stat.S_IMODE(version[2])))
        with self.held._opened(relative) as (fd, guard):
            def effect(operation):
                _require(self.worker.operation.moment() < self.worker.selected[1]['expires_at_epoch'],
                         'restore_approval_expired')
                guard()
                before = self.held.versions[relative]
                if operation == 'chown':
                    os.fchown(fd, version[3], version[4])
                    expected = before[:3] + version[3:5]
                else:
                    os.fchmod(fd, stat.S_IMODE(version[2]))
                    expected = before[:2] + [version[2]] + before[3:5]
                after = _version(os.fstat(fd))
                _require(after[:5] == expected and after[5:8] == before[5:8]
                    and after[9] == before[9] and after[8] >= before[8], 'restore_tree_changed')
                self.held.versions[relative] = after
                if relative == '':
                    parent, leaf, root, _ = self.held.chain[-1]
                    self.held.chain[-1] = (parent, leaf, root, after)
                guard()
            if relative == '':
                # Durable restore_final already precedes this transaction.
                # Prove the private tree before granting access; after chown,
                # owner readers are expected and the private-view predicate
                # no longer applies. Keep the current authority lock and exact
                # retained root across both planned permission syscalls.
                with self.worker.mutation_authority(readers=True):
                    effect('chown')
                    effect('chmod')
            else:
                for operation in ('chown', 'chmod'):
                    with self.worker.mutation_authority(readers=True):
                        effect(operation)
            os.fsync(fd)
        if relative == '':
            self.worker.record('access_reopened',
                               dict(phase='owner_rights_observed', path=relative, version=self.held.versions[relative]))

    def reopen(self):
        self._rights(self.manifest['members'][0])
