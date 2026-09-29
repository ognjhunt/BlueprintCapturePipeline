"""Finite metadata provisioning; this creates no enrollment or action grant."""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

from .control_plane_lane_experiment_publication import _BirthFiles, _Record, _guard
from .control_plane_lane_owner_target_versions import _require
from .control_plane_reference_budget import ReferenceCollectionBudget


def _mode(files, fd, *, directory, mode, gid):
    files.location(fd)
    value = os.fstat(fd)
    _require(
        value.st_uid == 0
        and value.st_gid == gid
        and stat.S_IMODE(value.st_mode) == mode
        and (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode))
        and (directory or value.st_nlink == 1 and value.st_size == 0),
        "experiment_installation_unsafe",
    )


def _existing(files, path, *, directory, mode, gid):
    parent, name = files.parent(path, protected=True)
    files.location(parent)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    fd = files.open(
        name, os.O_RDONLY | (os.O_DIRECTORY if directory else os.O_NONBLOCK), parent=parent
    )
    _mode(files, fd, directory=directory, mode=mode, gid=gid)
    return fd


def _empty_lock(files, parent, name, mode, gid):
    # The fixed caller has already checked the original directory and all existing
    # state. Publish an empty original inode without replacing any existing name.
    temporary = ".target-version-" + secrets.token_hex(16) + ".tmp"
    fd = files.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, parent=parent)
    state = _Record(parent, fd, temporary, name, files.proof(fd))
    try:
        _guard(files, state)
        os.fchown(fd, 0, gid)
        state.gid = gid
        _guard(files, state)
        os.fchmod(fd, mode)
        state.mode = mode
        _guard(files, state)
        os.fsync(fd)
        _guard(files, state)
        os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        state.published, state.links = True, 2
        _guard(files, state)
        os.unlink(temporary, dir_fd=parent)
        state.present, state.links = False, 1
        _guard(files, state)
        os.fsync(parent)
        _guard(files, state)
    finally:
        if state.present:
            try:
                _guard(files, state, cleanup=True)
                os.unlink(temporary, dir_fd=parent)
            except (OSError, ValueError):
                pass


def prepare(*, installed_config_path):
    from .control_plane_lane_experiment_birth import _blueprint_identity
    from .control_plane_lane_experiment_retirement import _configuration
    from .control_plane_lane_experiment_restore import _new_directory, _owner_mode

    _require(os.geteuid() == 0, "experiment_issuer_required")
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        config = _configuration(files, installed_config_path)
        _, gid = _blueprint_identity()
        state = Path(config.state_root)
        requests = Path(config.spool_root)
        # Existing shared ancestors are prerequisites; no new root selection.
        for path in (state, requests):
            _require(
                _existing(files, path, directory=True, mode=0o755, gid=0) is not None,
                "experiment_installation_unsafe",
            )
        directories = (
            (Path(config.experiment_record_store), 0o700, 0),
            (Path(config.experiment_authority_root), 0o750, gid),
            (Path(config.needed_checkpoint_cache_record_store), 0o700, 0),
            (Path(config.needed_checkpoint_cache_registration_root), 0o755, 0),
            (Path(config.needed_checkpoint_cache_authority_root), 0o750, gid),
        )
        locks = (
            (directories[0][0] / ".experiment-authority.lock", 0o600, 0),
            (directories[1][0] / ".authority.lock", 0o640, gid),
            (directories[2][0] / ".cache-store.lock", 0o600, 0),
            (directories[4][0] / ".authority.lock", 0o640, gid),
        )
        existing = {}
        # Validate ALL present state before creating any missing state. A missing
        # finite parent implies its children are absent; symlinks never count.
        for path, mode, group in directories:
            if path.parent in existing and existing[path.parent] is None:
                existing[path] = None
            else:
                existing[path] = _existing(files, path, directory=True, mode=mode, gid=group)
        for path, mode, group in locks:
            existing[path] = (
                None
                if existing[path.parent] is None
                else _existing(files, path, directory=False, mode=mode, gid=group)
            )
        for path, mode, group in directories:
            if existing[path] is None:
                parent, name = files.parent(path, protected=True)
                fd = _new_directory(files, parent, name)
                _owner_mode(files, fd, 0, group, mode)
                _mode(files, fd, directory=True, mode=mode, gid=group)
                existing[path] = fd
        for path, mode, group in locks:
            if existing[path] is None:
                parent, name = files.parent(path, protected=True)
                _empty_lock(files, parent, name, mode, group)
        files.verify()
        return dict(
            decision="prepared",
            creation_enabled=config.experiment_creation_enabled,
            retirement_enabled=config.experiment_retirement_enabled,
            cache_creation_enabled=config.needed_checkpoint_cache_creation_enabled,
        )
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()
