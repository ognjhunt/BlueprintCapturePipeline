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


def prepare_ledger_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o2770)
    try:
        root.chmod(0o2770)
    except PermissionError:
        pass
    return root.resolve(strict=True)


def open_ledger_lock(ledger: Path) -> int:
    descriptor = os.open(ledger / ".lock", os.O_RDWR | os.O_CREAT, 0o660)
    try:
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o660:
            os.fchmod(descriptor, 0o660)
    except OSError:
        pass  # the installer owns the lock's mode; admission checks it strictly
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
