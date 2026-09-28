"""Primitives the disk-admission ledger and its footprint history share.

Where the ledger lives, how its lock is opened, the typed refusal, and each
role's declared footprint.  Admission (``control_plane_disk_budget``) and the
measured footprints (``control_plane_disk_footprints``) both build on this
module, so neither has to import the other.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path


GIB = 1024**3
DEFAULT_RESERVATION_ROOT = Path(
    "/var/lib/blueprint/pipeline-control-plane/disk-reservations"
)
# Declared footprint per role.  Preparation and compilation reserve their exact
# miss bytes at run time (references or runtime members the content stores do
# not already hold); these values are the typical hit-path footprint the intake
# checks before accepting a submission, and the ceiling a measured footprint
# can never exceed.
ROLE_FOOTPRINT_BYTES: Mapping[str, int] = {
    "control_plane_deploy": 2 * GIB,
    "handoff_staging": 4 * GIB,
    "launch_preparation": 2 * GIB,
    "episode_compilation": 2 * GIB,
    "launch_activation": 2 * GIB,
    "launch_dispatch": 2 * GIB,
    "policy_canary_dispatch": 2 * GIB,
    "evidence_offload": 2 * GIB,
    "result_artifact_download": 256 * 1024 * 1024,
    "stage_replay": 4 * GIB,
    "semantic_pretraining": 3 * GIB,
    "cpu_prestage": 6 * GIB,
    "g1_checkpoint_cache": 32 * GIB,
    "experiment_restore": 256 * 1024 * 1024,
}
ROLE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")


class ControlPlaneDiskBudgetError(RuntimeError):
    """A write-heavy operation was refused before it mutated its output."""


def environment_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ControlPlaneDiskBudgetError(
            f"control_plane_disk_budget_configuration_invalid:{name}"
        ) from exc
    if value < 0:
        raise ControlPlaneDiskBudgetError(
            f"control_plane_disk_budget_configuration_invalid:{name}"
        )
    return value


def footprint_bytes(role: str) -> int:
    """The role's declared footprint (its ceiling), honouring the env override."""

    if role not in ROLE_FOOTPRINT_BYTES:
        raise ControlPlaneDiskBudgetError(
            f"control_plane_disk_budget_role_invalid:{role}"
        )
    name = f"BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_{role.upper()}_BYTES"
    return environment_int(name, ROLE_FOOTPRINT_BYTES[role])


# Root and the runtime account share the ledger, and the ledger directory is
# group-writable, so nothing here follows a symlink: every open refuses one and
# every mode repair goes through a descriptor this process owns.
_NO_FOLLOW = os.O_NOFOLLOW | os.O_CLOEXEC


def prepare_ledger_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o2770)
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | _NO_FOLLOW)
    except OSError as exc:
        raise ControlPlaneDiskBudgetError("control_plane_disk_budget_ledger_invalid") from exc
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid == os.geteuid() and stat.S_IMODE(metadata.st_mode) != 0o2770:
            try:
                os.fchmod(descriptor, 0o2770)
            except OSError:
                pass  # the installer owns the ledger's mode
    finally:
        os.close(descriptor)
    return root.resolve(strict=True)


def open_ledger_lock(ledger: Path, *, require_mode: bool = False) -> int:
    """Open ``<ledger>/.lock`` without following a symlink; return its descriptor.

    The lock must be a regular file.  A lock whose mode is not 0660 is repaired
    only by its owner, through the descriptor.  With ``require_mode`` (admission)
    a lock that is still not 0660 is refused, so a mode that would lock another
    account out never passes silently.
    """

    try:
        descriptor = os.open(ledger / ".lock", os.O_RDWR | os.O_CREAT | _NO_FOLLOW, 0o660)
    except OSError as exc:
        raise ControlPlaneDiskBudgetError("control_plane_disk_budget_lock_invalid") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ControlPlaneDiskBudgetError("control_plane_disk_budget_lock_invalid")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode != 0o660:
            try:
                if metadata.st_uid != os.geteuid():
                    raise PermissionError("not the lock's owner")
                os.fchmod(descriptor, 0o660)  # nosec B103 - shared root/blueprint ledger lock
            except OSError as exc:
                if require_mode:
                    raise ControlPlaneDiskBudgetError(
                        f"control_plane_disk_budget_lock_mode_invalid:{mode:04o}"
                    ) from exc
            installed = stat.S_IMODE(os.fstat(descriptor).st_mode)
            if require_mode and installed != 0o660:
                raise ControlPlaneDiskBudgetError(
                    f"control_plane_disk_budget_lock_mode_repair_failed:{installed:04o}"
                )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


__all__ = [
    "ControlPlaneDiskBudgetError",
    "DEFAULT_RESERVATION_ROOT",
    "GIB",
    "ROLE_FOOTPRINT_BYTES",
    "ROLE_NAME_RE",
    "environment_int",
    "footprint_bytes",
    "open_ledger_lock",
    "prepare_ledger_root",
]
