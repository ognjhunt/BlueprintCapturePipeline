"""Finite experiment metadata publication under original owned descriptors.

These observations do not make syscalls atomic against hostile same-UID threads.
No old consent publisher, pathname stream, or replacement fallback is used.
"""
from __future__ import annotations

import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_owner_target_io import _TargetFiles, _typed
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _require

_NAMES = {
    "private": r"[0-9a-f]{32}(?:\.(?:claim|creation|publication|correspondence|completed|producer-completion|completion-head|restore-intent|restore-selection|restore-pending-head|restore-head|restored-head|head-prepared|authority-pending|action|reservation|retiring-head|retired-head))?\.json",
    "manifest": r"[0-9a-f]{32}\.manifest\.json",
    "event": r"e-[0-9]{5}\.json",
    "birth": r"[0-9a-f]{32}\.birth\.json",
    "bootstrap": r"[0-9a-f]{32}\.producer-bootstrap\.json",
    "authority": r"authority-[0-9]{8}-[0-9a-f]{32}\.json",
    "head": r"HEAD\.json",
    "certificate": r"restoration-[0-9a-f]{64}\.json",
    "lease": r"\.lane-scratch\.v1\.json",
    "marker": r"\.registered-experiment\.v1\.json",
}
_MODES = dict(manifest=0o600, event=0o600, private=0o600, birth=0o640, authority=0o640, head=0o640,
              lease=0o600, marker=0o600, certificate=0o640, bootstrap=0o640)
_CAPS = dict(manifest=1048576, event=32768, private=32768, birth=32768, authority=32768, head=4096,
             lease=8192, marker=4096, certificate=8192, bootstrap=32768)


class _BirthFiles(_TargetFiles):
    def __init__(self, budget):
        super().__init__(budget)
        self.parents = {}

    def parent(self, path, *, protected=False):
        selected = retained._path(os.fspath(path), _work_budget=self.budget)
        known = self.parents.get(selected.parent)
        if known is not None:
            self.budget.tick()
            self.location(known)
            if protected:
                current = known
                for _ in range(64):
                    owners._protected(os.fstat(current), directory=True)
                    parent = self.bindings[current][0]
                    if parent is None:
                        break
                    self.proof(parent)
                    current = parent
                else:
                    raise OwnerTargetVersionError("experiment_resource_exhausted")
            return known, selected.name
        self.budget.charge("roots")
        prefix = Path("/")
        fd = self.parents.get(prefix)
        if fd is None:
            fd = self.open("/", os.O_RDONLY | os.O_DIRECTORY)
            self.parents[prefix] = fd
        self.location(fd)
        if protected:
            owners._protected(os.fstat(fd), directory=True)
        for component in selected.parts[1:-1]:
            self.budget.charge("entries")
            prefix = prefix / component
            child = self.parents.get(prefix)
            if child is None:
                child = self.open(component, os.O_RDONLY | os.O_DIRECTORY, parent=fd)
                self.parents[prefix] = child
            self.location(child)
            if protected:
                owners._protected(os.fstat(child), directory=True)
            fd = child
        return fd, selected.name

    def location(self, fd, *, cleanup=False):
        """Verify every original edge; an authorized ownership change is explicit."""
        for _ in range(64):
            if not cleanup:
                self.budget.tick()
            self.proof(fd)
            parent, name, security = self.bindings[fd]
            if parent is not None:
                self.proof(parent)
            _require(owners._security(os.fstat(fd)) == security
                     == owners._security(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                     "experiment_location_changed")
            if parent is None:
                return
            fd = parent
        raise OwnerTargetVersionError("experiment_resource_exhausted")

    def new_directory(self, parent, name):
        _require(isinstance(name, str) and re.fullmatch(r"create-[0-9a-f]{32}", name),
                 "experiment_creation_invalid")
        self.location(parent)
        os.mkdir(name, 0o700, dir_fd=parent)
        self.location(parent)
        fd = self.open(name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
        info = self.acquired[fd]
        _require(info.st_uid == info.st_gid == 0 and stat.S_IMODE(info.st_mode) == 0o700,
                 "experiment_stage_unsafe")
        return fd

    def transition_owner(self, fd, uid, gid, mode):
        self.location(fd)
        original = self.proof(fd)
        before = os.fstat(fd)
        _require(before.st_uid == before.st_gid == 0 and mode in (0o600, 0o700), "experiment_stage_unsafe")
        os.fchown(fd, uid, gid)
        _require(_typed(os.fstat(fd)) == original and os.fstat(fd).st_uid == uid
                 and os.fstat(fd).st_gid == gid, "experiment_stage_unsafe")
        parent, name, _ = self.bindings[fd]
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _require(owners._metadata(named) == owners._metadata(os.fstat(fd)), "experiment_stage_unsafe")
        self.bindings[fd] = (parent, name, owners._security(named))
        self.location(fd)
        os.fchmod(fd, mode)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _require(_typed(named) == original and owners._metadata(named) == owners._metadata(os.fstat(fd))
                 and stat.S_IMODE(named.st_mode) == mode, "experiment_stage_unsafe")
        self.bindings[fd] = (parent, name, owners._security(named))
        self.location(fd)
        os.fsync(fd)


@dataclass
class _Record:
    parent: int
    fd: int
    temp: str
    name: str
    proof: tuple
    size: int = 0
    links: int = 1
    present: bool = True
    published: bool = False
    mode: int = 0o600
    gid: int = 0
    previous: object = None


def _guard(files, state, *, cleanup=False):
    if not cleanup:
        files.budget.tick()
    files.location(state.parent, cleanup=cleanup)
    _require(files.proof(state.fd) == state.proof, "experiment_publication_failed")
    opened = os.fstat(state.fd)
    _require(opened.st_uid == 0 and opened.st_gid == state.gid and stat.S_ISREG(opened.st_mode)
             and stat.S_IMODE(opened.st_mode) == state.mode and opened.st_nlink == state.links
             and opened.st_size == state.size, "experiment_publication_failed")
    named = os.stat(state.temp if state.present else state.name, dir_fd=state.parent, follow_symlinks=False)
    _require(owners._metadata(opened) == owners._metadata(named), "experiment_publication_failed")
    if state.published:
        _require(owners._metadata(os.stat(state.name, dir_fd=state.parent, follow_symlinks=False))
                 == owners._metadata(opened), "experiment_publication_failed")
    elif state.previous is not None:
        previous = state.previous
        files.proof(previous.fd)
        _require(previous.parent == state.parent and previous.name == state.name
                 and owners._metadata(os.fstat(previous.fd)) == owners._metadata(previous.info)
                 == owners._metadata(os.stat(state.name, dir_fd=state.parent, follow_symlinks=False)),
                 "experiment_head_changed")
    else:
        try:
            os.stat(state.name, dir_fd=state.parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OwnerTargetVersionError("experiment_publication_destination_exists")


def _publish(files, parent, name, payload, *, kind, blueprint_gid=0, _expected_head=None):
    _require(kind in _NAMES and isinstance(name, str) and re.fullmatch(_NAMES[kind], name)
             and isinstance(payload, bytes) and 0 < len(payload) <= _CAPS[kind], "experiment_publication_invalid")
    _require(_expected_head is None or kind == "head" and isinstance(_expected_head, owners._Acquired)
             and _expected_head.name == "HEAD.json" and _expected_head.parent == parent,
             "experiment_publication_invalid")
    files.budget.charge("output_bytes", len(payload))
    files.location(parent)
    info = os.fstat(parent)
    _require(info.st_uid == 0 and info.st_gid == (blueprint_gid if kind in ("birth", "authority", "head", "certificate", "bootstrap") else 0)
             and stat.S_IMODE(info.st_mode) == (0o750 if kind in ("birth", "authority", "head", "certificate", "bootstrap") else 0o700),
             "experiment_publication_parent_unsafe")
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        _require(_expected_head is None, "experiment_head_changed")
    else:
        _require(_expected_head is not None, "experiment_publication_destination_exists")
        files.verify_record(_expected_head)
    temp = ".target-version-" + secrets.token_hex(16) + ".tmp"
    fd = files.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, parent=parent)
    state = _Record(parent, fd, temp, name, files.proof(fd), previous=_expected_head)
    try:
        while state.size < len(payload):
            _guard(files, state)
            count = os.write(fd, memoryview(payload)[state.size:])
            _require(type(count) is int and 0 < count <= len(payload) - state.size, "experiment_publication_failed")
            state.size += count
        if _MODES[kind] == 0o640:
            _guard(files, state)
            os.fchown(fd, 0, blueprint_gid)
            state.gid = blueprint_gid
        _guard(files, state)
        os.fchmod(fd, _MODES[kind])
        state.mode = _MODES[kind]
        _guard(files, state)
        os.fsync(fd)
        _guard(files, state)
        if _expected_head is None:
            os.link(temp, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            state.published, state.links = True, 2
            _guard(files, state)
            os.unlink(temp, dir_fd=parent)
            state.present, state.links = False, 1
        else:
            # Exact old HEAD was proved above under the real authority EX lock.
            # Immutable versions and the prepared private head survive this CAS.
            os.rename(temp, name, src_dir_fd=parent, dst_dir_fd=parent)
            state.present, state.published = False, True
            if _expected_head in files.records:
                files.records.remove(_expected_head)
        _guard(files, state)
        os.fsync(parent)
        _guard(files, state)
        check = files.open(name, os.O_RDONLY | os.O_NONBLOCK, parent=parent)
        _require(files.proof(check) == state.proof and files.read_bytes(check, max(1, _CAPS[kind])) == payload,
                 "experiment_publication_failed")
        _guard(files, state)
        files.close(check)
        _require(check not in files.owned and not files.unresolved, "experiment_publication_failed")
        return {"sha256": retained._digest(payload, _work_budget=files.budget), "size_bytes": len(payload)}
    finally:
        if state.present:
            # Cleanup never substitutes a new identity. Other known handles are
            # finalized by the outer owner even if this original cannot be proved.
            try:
                _guard(files, state, cleanup=True)
                os.unlink(temp, dir_fd=parent)
            except (OSError, ValueError):
                pass
