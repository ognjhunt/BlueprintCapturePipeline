"""Planner-owned, no-follow metadata observations; never payload acquisition.

All handles belong to one invocation. Stability is observed again at the end;
neither these checks nor advisory locks provide a filesystem snapshot.
"""
from __future__ import annotations

import os
import stat as types
from pathlib import PurePosixPath

from .control_plane_reference_budget import ReferenceCollectionBudget

MAX_ANCHORS, MAX_DIRECTORIES, MAX_FDS = 4, 512, 768
MAX_RECORD_BYTES = 4 * 1024 * 1024
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class AcquisitionError(ValueError):
    """Only fixed planner evidence codes are public."""


def require(condition, code):
    if not condition:
        raise AcquisitionError('scene_lifecycle_' + code)


def path(value, budget):
    budget.tick()
    require(type(value) is str and 0 < len(value) <= 4096, 'path_invalid')
    for offset, char in enumerate(value):
        if offset % 1024 == 0:
            budget.tick()
        require(ord(char) >= 32 and ord(char) != 127 and char not in '\\<>*?[]', 'path_invalid')
    require(len(value.encode('utf-8')) <= 4096 and value.startswith('/'), 'path_invalid')
    parts = value[1:].split('/')
    require(len(parts) <= 64 and all(p not in ('', '.', '..') for p in parts), 'path_invalid')
    budget.tick()
    return value


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


class Acquisition:
    def __init__(self, budget, anchors):
        require(type(budget) is ReferenceCollectionBudget, 'budget_invalid')
        budget.tick()
        require(isinstance(anchors, (list, tuple)) and 1 <= len(anchors) <= MAX_ANCHORS,
                'anchors_invalid')
        self.budget, self.handles, self.directories = budget, {}, {}
        self.observations, self.memberships, self.anchors = {}, {}, []
        self.unproven, self.cleanup_failed = False, False
        checked = [path(p, budget) for p in anchors]
        require(len(set(checked)) == len(checked), 'anchors_invalid')
        require(not any(PurePosixPath(a).is_relative_to(PurePosixPath(b))
                        for a in checked for b in checked if a != b), 'anchors_invalid')
        try:
            for anchor in checked:
                budget.available('roots', 1)
                descriptor, info = self._directory(anchor)
                if (info.st_dev, info.st_ino) not in {i for _, i in self.anchors}:
                    budget.charge('roots')
                self.anchors.append((anchor, (info.st_dev, info.st_ino)))
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _open(self, name, flags, parent=None):
        self.budget.tick()
        require(len(self.handles) < MAX_FDS, 'descriptors_limit')
        fd = os.open(name, flags | getattr(os, 'O_CLOEXEC', 0), dir_fd=parent)
        try:
            info = os.fstat(fd)
        except BaseException:
            # An unobserved numeric token must never be adopted/closed blindly.
            self.unproven = True
            raise AcquisitionError('scene_lifecycle_descriptor_ownership_unproven') from None
        self.handles[fd] = (info.st_dev, info.st_ino)
        self.budget.tick()
        return fd, info

    def _directory(self, absolute):
        if absolute in self.directories:
            return self.directories[absolute]
        require(len(self.directories) < MAX_DIRECTORIES, 'directories_limit')
        if absolute == '/':
            fd, info = self._open('/', DIR_FLAGS)
        else:
            parent, _, name = absolute.rpartition('/')
            parent_fd, _ = self._directory(parent or '/')
            require(len(self.directories) < MAX_DIRECTORIES, 'directories_limit')
            fd, info = self._open(name, DIR_FLAGS, parent_fd)
            self._observe(absolute, parent_fd, name, info)
        self.directories[absolute] = fd, info
        return fd, info

    def _observe(self, value, parent, name, info):
        observed = parent, name, identity(info)
        require(value not in self.observations or self.observations[value] == observed,
                'metadata_changed')
        self.observations.setdefault(value, observed)

    def _parent(self, value):
        value = path(value, self.budget)
        require(any(PurePosixPath(value).is_relative_to(PurePosixPath(a)) for a, _ in self.anchors),
                'path_outside_anchor')
        parent, _, name = value.rpartition('/')
        return self._directory(parent or '/')[0], name

    def stat(self, value):
        parent, name = self._parent(value)
        self.budget.tick()
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        self.budget.tick()
        self._observe(value, parent, name, info)
        return info

    def entries(self, value):
        value = path(value, self.budget)
        require(any(PurePosixPath(value).is_relative_to(PurePosixPath(a)) for a, _ in self.anchors),
                'path_outside_anchor')
        fd, info = self._directory(value)
        self.budget.tick()
        names = []
        with os.scandir(fd) as entries:
            for entry in entries:
                self.budget.charge('entries')
                names.append(entry.name)
        self.budget.tick()
        names.sort()
        observed = fd, tuple(names), identity(info)
        require(value not in self.memberships or self.memberships[value] == observed,
                'metadata_changed')
        self.memberships.setdefault(value, observed)
        return tuple(names)

    def read_json(self, value):
        require(type(value) is str and value.endswith('.json'), 'metadata_filename_invalid')
        parent, name = self._parent(value)
        fd = None
        try:
            fd, before = self._open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, parent)
            require(types.S_ISREG(before.st_mode), 'metadata_type_invalid')
            require(before.st_size <= MAX_RECORD_BYTES, 'metadata_bytes_limit')
            pieces, count = [], 0
            while True:
                self.budget.tick()
                remaining = self.budget.limits['raw_bytes'] - self.budget.counts['raw_bytes']
                self.budget.available('raw_bytes', 1)
                request = min(65536, MAX_RECORD_BYTES - count + 1, remaining)
                require(request > 0, 'metadata_bytes_limit')
                part = os.read(fd, request)
                self.budget.charge('raw_bytes', len(part))
                if not part:
                    break
                count += len(part)
                require(count <= MAX_RECORD_BYTES, 'metadata_bytes_limit')
                pieces.append(part)
            self.budget.tick()
            after = os.fstat(fd)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(count == before.st_size and identity(before) == identity(after) == identity(named),
                    'metadata_changed')
            self._observe(value, parent, name, before)
            self.budget.available('raw_bytes', 0)
            payload = b''.join(pieces)
            self.budget.tick()
            return payload
        except OSError:
            raise AcquisitionError('scene_lifecycle_metadata_unavailable') from None
        finally:
            if fd is not None:
                self._close(fd)

    def verify(self):
        # Named root and every retained directory handle belong to the first
        # observation. Relative children alone cannot establish those identities.
        root_info = self.directories['/'][1]
        self.budget.tick()
        require(identity(os.stat('/', follow_symlinks=False)) == identity(root_info), 'metadata_changed')
        for fd, expected in self.directories.values():
            self.budget.tick()
            require(identity(os.fstat(fd)) == identity(expected), 'metadata_changed')
        for parent, name, expected in self.observations.values():
            self.budget.tick()
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(identity(info) == expected, 'metadata_changed')
        for fd, expected, directory_identity in self.memberships.values():
            self.budget.tick()
            names = []
            with os.scandir(fd) as entries:
                for entry in entries:
                    self.budget.charge('entries')
                    names.append(entry.name)
            require(tuple(sorted(names)) == expected and identity(os.fstat(fd)) == directory_identity,
                    'metadata_changed')
        self.budget.tick()
        return True

    def _close(self, fd):
        expected = self.handles.pop(fd, None)
        if expected is None:
            return
        for _ in range(2):
            try:
                current = os.fstat(fd)
                if (current.st_dev, current.st_ino) != expected:
                    self.cleanup_failed = True
                    return
                os.close(fd)
                return
            except OSError:
                continue
        self.cleanup_failed = True

    def close(self):
        for fd in tuple(reversed(self.handles)):
            self._close(fd)
        require(not self.cleanup_failed, 'descriptor_cleanup_failed')
        require(not self.unproven, 'descriptor_ownership_unproven')
