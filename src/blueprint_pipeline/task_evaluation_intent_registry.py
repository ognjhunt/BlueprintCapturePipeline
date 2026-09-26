"""Atomic, serialized publication of immutable release-bound owner intents."""
from __future__ import annotations

import fcntl
import grp
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Any

from .public_scene_host_input_intake import _verified_checkout_head


class IntentRegistryError(ValueError):
    """A registry update cannot preserve current authority and historical bytes."""


def _supersession_authority(expected_commit: str) -> None:
    if _verified_checkout_head() != expected_commit:
        raise IntentRegistryError("intent_registry_execution_commit_mismatch")
    # A separate installer must see every trigger stopped. The progression
    # service itself is serialized by systemd: a path/timer wake cannot start a
    # second copy while its MainPID is publishing the successor. Admit only
    # that exact process; an external writer still needs full quiescence.
    units = ["blueprint-task-evaluation-configured-controls-progression." + suffix
             for suffix in ("service", "path", "timer")]
    current_worker = False
    for index, unit in enumerate(units):
        try:
            result = subprocess.run(  # nosec B603 B607 - fixed read-only systemctl invocation
                ["systemctl", "show", unit, "-p", "LoadState", "-p", "ActiveState", "-p", "MainPID"],
                check=True, capture_output=True, text=True, timeout=10)
            values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        except (OSError, subprocess.SubprocessError) as exc:
            raise IntentRegistryError("intent_registry_worker_quiescence_unproven") from exc
        if values.get("LoadState") != "loaded":
            raise IntentRegistryError("intent_registry_worker_quiescence_unproven")
        if index == 0 and (values.get("ActiveState") in {"active", "activating"}
                           and values.get("MainPID") == str(os.getpid())):
            current_worker = True
            continue
        if (current_worker and index > 0 and values.get("ActiveState") in {"active", "inactive"}
                and values.get("MainPID", "0") == "0"):
            continue
        if values.get("ActiveState") != "inactive" or values.get("MainPID", "0") != "0":
            raise IntentRegistryError("intent_registry_worker_quiescence_unproven")


def _read(path: Path) -> bytes:
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise IntentRegistryError("intent_registry_file_invalid")
        return stream.read()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_release_intent(*, destination: Path, payload: bytes, expected_commit: str,
        service_group: str | None, validate: Callable[[Mapping[str, Any]], Any]) -> None:
    """Validate/stage first, archive without overwriting, then switch one live name.

    Supersession requires the exact running release and stopped progression
    workers/triggers. No rollback authority is accepted by this entrypoint.
    """
    value = json.loads(payload)
    validate(value)
    if value.get("expected_production_commit") != expected_commit:
        raise IntentRegistryError("intent_registry_execution_commit_mismatch")
    group_id = grp.getgrnam(service_group).gr_gid if service_group is not None else None
    if not destination.is_absolute() or any(p.is_symlink() for p in (destination, *destination.parents)):
        raise IntentRegistryError("intent_registry_path_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    lock_path = destination.with_name("." + destination.name + ".registry.lock")
    # The lock inode is never renamed/unlinked. All registry writers use it.
    lock_fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o440)
    try:
        lock_metadata = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_metadata.st_mode):
            raise IntentRegistryError("intent_registry_lock_invalid")
        # A root-run offline install can leave this lock owned by root. The
        # service only needs group read access, so preserve its owner on reuse.
        if group_id is not None and lock_metadata.st_gid != group_id:
            os.fchown(lock_fd, -1, group_id)
        # O_CREAT's mode is filtered by umask (the host root umask is 0077).
        # Repair only when needed; a service user cannot chmod a root-owned
        # lock that already has the correct group-readable mode.
        if stat.S_IMODE(lock_metadata.st_mode) != 0o440:
            os.fchmod(lock_fd, 0o440)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        retired_target = destination.with_name(
            f"{destination.stem}.superseded-{expected_commit}.json")
        current = _read(destination) if destination.exists() or destination.is_symlink() else None
        if current == payload:
            metadata = destination.stat(follow_symlinks=False)
            if (stat.S_IMODE(metadata.st_mode) != 0o440
                    or (group_id is not None and metadata.st_gid != group_id)):
                raise IntentRegistryError("intent_registry_live_access_invalid")
            return
        if retired_target.exists() or retired_target.is_symlink():
            raise IntentRegistryError("intent_registry_retired_release_reactivation_forbidden")
        previous = None
        if current is not None:
            old = json.loads(current)
            validate(old)
            previous = old.get("expected_production_commit")
            if previous == expected_commit or not isinstance(previous, str) or not re.fullmatch(r"[0-9a-f]{40}", previous):
                raise IntentRegistryError("intent_registry_same_release_conflict")
            _supersession_authority(expected_commit)
        # The successor has complete bytes and service ownership before the
        # original registration is touched. A staging failure leaves it intact.
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".intent-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                if group_id is not None:
                    os.fchown(stream.fileno(), os.geteuid(), group_id)
                os.fchmod(stream.fileno(), 0o440)
                if _read(temporary) != payload:
                    raise IntentRegistryError("intent_registry_staging_readback_mismatch")
                if previous is not None:
                    archive = destination.with_name(f"{destination.stem}.superseded-{previous}.json")
                    try:
                        os.link(destination, archive, follow_symlinks=False)
                    except FileExistsError:
                        if _read(archive) != current:
                            raise IntentRegistryError("intent_registry_archive_conflict") from None
                    if _read(archive) != current or _read(destination) != current:
                        raise IntentRegistryError("intent_registry_current_bytes_changed")
                    _sync_directory(destination.parent)
                os.replace(temporary, destination)
                _sync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        os.close(lock_fd)
