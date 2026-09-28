"""Guarded immutable metadata publication for the new finite report only."""
from __future__ import annotations

import os
import re
import secrets
import stat
from dataclasses import dataclass

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_owner_target_io import _typed
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _require

_REQUEST_ID = r"[0-9]{8}T[0-9]{6}Z-owner-target-version-[0-9a-f]{8}"
_CAPS = {"attestation": 32768, "report": 65536, "outcome": 8192}


@dataclass
class _PublicationState:
    parent: int
    fd: int
    temporary: str
    name: str
    parent_proof: tuple
    temp_proof: tuple
    kind: str
    mode: int
    size: int = 0
    nlink: int = 1
    linked: bool = False
    temp_present: bool = True


def _parent_guard(files, state):
    _require(files.proof(state.parent) == state.parent_proof, "owner_target_publication_failed")
    fd, depth = state.parent, 0
    while fd is not None:
        _require(depth < 64 and fd in files.bindings, "owner_target_publication_failed")
        files.proof(fd)
        parent, name, security = files.bindings[fd]
        if parent is not None:
            files.proof(parent)
        actual = os.fstat(fd)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _require(owners._security(actual) == security == owners._security(named)
                 and actual.st_uid == actual.st_gid == 0 and stat.S_ISDIR(actual.st_mode)
                 and not actual.st_mode & 0o022, "owner_target_publication_failed")
        if state.kind != "attestation":
            _require(stat.S_IMODE(actual.st_mode) == 0o755, "owner_target_publication_failed")
        elif fd == state.parent:
            _require(stat.S_IMODE(actual.st_mode) == 0o700, "owner_target_publication_failed")
        fd, depth = parent, depth + 1


def _publication_guard(files, state, *, stage, cleanup=False):
    # Every caller supplies a finite stage, never request-provided mutation flags.
    _require(stage in {"fchmod", "write", "fsync", "link", "unlink", "parent_fsync", "readback", "cleanup"},
             "owner_target_publication_failed")
    if not cleanup:
        files.budget.tick()
    _parent_guard(files, state)
    _require(files.proof(state.fd) == state.temp_proof, "owner_target_publication_failed")
    actual = os.fstat(state.fd)
    _require(_typed(actual) == state.temp_proof and actual.st_uid == actual.st_gid == 0
             and stat.S_ISREG(actual.st_mode) and stat.S_IMODE(actual.st_mode) == state.mode
             and actual.st_size == state.size and actual.st_nlink == state.nlink,
             "owner_target_publication_failed")
    name = state.temporary if state.temp_present else state.name
    named = os.stat(name, dir_fd=state.parent, follow_symlinks=False)
    _require(owners._metadata(actual) == owners._metadata(named), "owner_target_publication_failed")
    if state.linked:
        final = os.stat(state.name, dir_fd=state.parent, follow_symlinks=False)
        _require(owners._metadata(final) == owners._metadata(actual), "owner_target_publication_failed")
    else:
        try:
            os.stat(state.name, dir_fd=state.parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OwnerTargetVersionError("owner_target_publication_destination_exists")


def _cleanup_owned_publication_temp(files, state):
    if not state.temp_present:
        return
    for _ in range(2):
        try:
            _publication_guard(files, state, stage="cleanup", cleanup=True)
            os.unlink(state.temporary, dir_fd=state.parent)
            state.temp_present = False
            state.nlink -= 1
            return
        except OSError:
            continue
        except (OwnerTargetVersionError, owners.OwnerCensusConsentError):
            # Unknown/reused parent, descriptor or name is preserved, never rebound.
            return


def _publish_owned_metadata(files, parent, name, payload, *, mode, artifact_kind):
    _require(artifact_kind in _CAPS and isinstance(payload, bytes) and 0 < len(payload) <= _CAPS[artifact_kind],
             "owner_target_resource_exhausted")
    grammar = (r"[0-9a-f]{32}\.json" if artifact_kind == "attestation" else
               _REQUEST_ID + (r"\.owner-target-version\.json" if artifact_kind == "report" else r"\.outcome\.json"))
    _require(isinstance(name, str) and re.fullmatch(grammar, name)
             and mode == (0o600 if artifact_kind == "attestation" else 0o644), "owner_target_publication_failed")
    files.budget.charge("output_bytes", len(payload))
    _require(len(files.publication_states) < 2, "owner_target_resource_exhausted")
    parent_proof = files.proof(parent)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise OwnerTargetVersionError("owner_target_publication_destination_exists")
    temporary = ".target-version-" + secrets.token_hex(16) + ".tmp"
    state = None
    try:
        fd = files.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, parent=parent, mode=0o600)
        original = files.acquired[fd]
        _require(original.st_size == 0 and original.st_nlink == 1, "owner_target_publication_failed")
        state = _PublicationState(parent, fd, temporary, name, parent_proof, files.owned[fd], artifact_kind,
                                  stat.S_IMODE(original.st_mode))
        files.publication_states.append(state)
        _publication_guard(files, state, stage="fchmod")
        os.fchmod(fd, 0o600)
        state.mode = 0o600
        while state.size < len(payload):
            _publication_guard(files, state, stage="write")
            written = os.write(fd, memoryview(payload)[state.size:])
            _require(type(written) is int and 0 < written <= len(payload) - state.size,
                     "owner_target_publication_failed")
            state.size += written
        _publication_guard(files, state, stage="fchmod")
        os.fchmod(fd, mode)
        state.mode = mode
        _publication_guard(files, state, stage="fsync")
        os.fsync(fd)
        _publication_guard(files, state, stage="link")
        os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        state.linked, state.nlink = True, 2
        _publication_guard(files, state, stage="unlink")
        os.unlink(temporary, dir_fd=parent)
        state.temp_present, state.nlink = False, 1
        _publication_guard(files, state, stage="parent_fsync")
        os.fsync(parent)
        _publication_guard(files, state, stage="readback")
        check = files.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, parent=parent)
        _require(files.owned[check] == state.temp_proof, "owner_target_publication_failed")
        readback = files.read_bytes(check, _CAPS[artifact_kind])
        _publication_guard(files, state, stage="readback")
        _require(readback == payload, "owner_target_publication_failed")
        files.close(check)
        _require(check not in files.owned and not files.unresolved, "owner_target_publication_failed")
        return dict(publication_checked=True, sha256=retained._digest(payload, _work_budget=files.budget),
                    size_bytes=len(payload))
    except OSError:
        raise OwnerTargetVersionError("owner_target_publication_failed") from None
    finally:
        if state is not None:
            _cleanup_owned_publication_temp(files, state)
