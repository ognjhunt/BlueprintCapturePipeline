"""Read-only capacity roles, summary bytes and wait projections for CPU consumers."""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import control_plane_disk_budget as disk_budget
from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT


def capacity_eta(shortfall_bytes: int, *, summary: Mapping[str, Any] | None, now: float) -> dict[str, Any]:
    """Bound a capacity wait to a measured growth or reclaim plan when known."""
    unknown = {"eta_epoch": None, "eta_basis": "unknown"}
    if not isinstance(summary, Mapping):
        return unknown
    outlook = summary.get("reclaim_outlook")
    if not isinstance(outlook, Mapping):
        return unknown
    growth = outlook.get("volume_growth")
    if not isinstance(growth, str):
        return unknown
    if growth in {"planned", "applied"}:
        return {"eta_epoch": float(now) + 600, "eta_basis": "volume_growth"}
    reclaimable = outlook.get("reclaimable_bytes")
    next_reclaim = outlook.get("next_reclaim_epoch")
    for value in (reclaimable, next_reclaim):
        if value is None:
            continue
        if type(value) not in (int, float):
            return unknown
        try:
            finite = math.isfinite(value)
        except OverflowError:
            return unknown
        if not finite:
            return unknown
    if type(reclaimable) not in (int, float) or reclaimable < 0:
        return unknown
    if (isinstance(reclaimable, (int, float)) and not isinstance(reclaimable, bool)
            and reclaimable >= max(0, shortfall_bytes)
            and isinstance(next_reclaim, (int, float)) and not isinstance(next_reclaim, bool)
            and next_reclaim > now):
        return {"eta_epoch": float(next_reclaim), "eta_basis": "reclaim_scheduled"}
    return {"eta_epoch": None, "eta_basis": "operator_action_required"}



CHAIN_ROLES: tuple[str, ...] = (
    "launch_preparation",
    "episode_compilation",
    "launch_activation",
    "launch_dispatch",
    "policy_canary_dispatch",
)



def _read_attention_summary(path: Path, *, max_bytes: int = 128 * 1024) -> dict[str, Any] | None:
    """Read a small local summary; distinguish absence from malformed evidence."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError:
        return {"status": "unreadable"}
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return {"status": "unreadable"}
        raw = os.read(fd, max_bytes + 1)
    except OSError:
        return {"status": "unreadable"}
    finally:
        os.close(fd)
    if len(raw) > max_bytes:
        return {"status": "unreadable"}
    try:
        value = json.loads(raw)
    except ValueError:
        return {"status": "unreadable"}
    return value if isinstance(value, dict) else {"status": "unreadable"}



WARNING_FRACTION = 0.70


CRITICAL_FRACTION = 0.85


def chain_footprints(
    reservation_root: str | Path = DEFAULT_RESERVATION_ROOT,
) -> dict[str, dict[str, Any]]:
    """Each chain role's footprint, computed exactly as its own reservation will."""

    return disk_budget.role_footprints(CHAIN_ROLES, reservation_root=reservation_root)


def footprint_basis(footprints: Mapping[str, Mapping[str, Any]]) -> str:
    """measured_p95 if every role is measured, declared_default if none is, else mixed."""

    measured = [row.get("basis") == "measured_p95" for row in footprints.values()]
    if measured and all(measured):
        return "measured_p95"
    return "mixed" if any(measured) else "declared_default"


def measure_mount(
    mount: str | Path,
    *,
    reservation_root: str | Path = DEFAULT_RESERVATION_ROOT,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    now: float | None = None,
    pid_alive: Callable[[int], bool] = disk_budget._pid_alive,
) -> dict[str, Any]:
    """One mount's admission projection, computed with the ledger's own functions.

    The floor, the live reservations on this mount's device and each chain
    role's measured footprint are the values a reservation on this mount would
    see, so a projected refusal is a refusal intake and the workers will make.
    """

    observed = time.time() if now is None else float(now)
    path = Path(mount)
    try:
        usage = disk_usage(path)
        device = disk_budget.target_device(path)
    except OSError as exc:
        status = "absent" if isinstance(exc, FileNotFoundError) and not path.exists() else "unreadable"
        return {"mount": str(path), "status": status, "errno": exc.errno}
    try:
        floor = disk_budget.floor_bytes(int(usage.total))
        footprints = chain_footprints(reservation_root)
        critical_floor = disk_budget.floor_bytes(int(usage.total), role="control_plane_deploy")
        critical_footprints = disk_budget.role_footprints(
            disk_budget.CRITICAL_ROLES, reservation_root=reservation_root,
        )
    except disk_budget.ControlPlaneDiskBudgetError as exc:
        # The ledger refuses every reservation under this configuration too.
        return {"mount": str(path), "status": "configuration_invalid", "blocker": str(exc)}
    try:
        reserved, live = disk_budget.live_reservations(
            reservation_root, device=device, now=observed, pid_alive=pid_alive
        )
    except disk_budget.ControlPlaneDiskBudgetError as exc:
        # Reservations that cannot be read are not zero reservations.
        return {"mount": str(path), "status": "unreadable", "blocker": str(exc)}
    available = max(0, int(usage.free) - floor - reserved)
    critical_available = max(0, int(usage.free) - critical_floor - reserved)
    refused = sorted(role for role, row in footprints.items() if row["bytes"] > available)
    critical_refused = sorted(
        role for role, row in critical_footprints.items() if row["bytes"] > critical_available
    )
    used_fraction = 0.0 if not usage.total else (usage.total - usage.free) / usage.total
    if used_fraction >= CRITICAL_FRACTION or refused or critical_refused:
        level = "critical"
    elif used_fraction >= WARNING_FRACTION:
        level = "warning"
    else:
        level = "ok"
    return {
        "mount": str(path),
        "status": "measured",
        "total_bytes": int(usage.total),
        "free_bytes": int(usage.free),
        "used_fraction": round(used_fraction, 4),
        "floor_bytes": floor,
        "critical_floor_bytes": critical_floor,
        "reserved_bytes": reserved,
        "live_reservations": live,
        "available_bytes": available,
        "critical_available_bytes": critical_available,
        "refused_roles": refused,
        "critical_roles_refused": critical_refused,
        "footprints": footprints,
        "free_needed_for_one_role_bytes": floor + footprints["launch_preparation"]["bytes"],
        "free_needed_for_whole_chain_bytes": floor
        + sum(row["bytes"] for row in footprints.values()),
        "level": level,
    }


def whole_chain_admission(
    mount,
    *,
    reservation_root=DEFAULT_RESERVATION_ROOT,
    now=None,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    role_targets: Mapping[str, str | Path] | None = None,
    device_of: Callable[[Path], int] = disk_budget.target_device,
):
    """Check the complete chain workspace before a new scene attempt starts.

    Each role counts at the footprint its own reservation will hold: the measured
    p95 once the role has history, its declared ceiling until then.  Per-stage
    reservations remain authoritative during execution. This earlier gate
    prevents starting a chain that already exceeds available headroom.
    """
    measured = measure_mount(mount, reservation_root=reservation_root, disk_usage=disk_usage, now=now)
    if not Path(mount).is_dir():
        measured = {"mount": str(mount), "status": "absent"}
    footprints = measured.get("footprints")
    if not isinstance(footprints, Mapping):
        try:
            footprints = chain_footprints(reservation_root)
        except disk_budget.ControlPlaneDiskBudgetError:
            # The measurement already failed closed, so this chain cannot be
            # admitted; report the declared ceilings and wait instead of raising
            # out of the caller's progression pass.
            footprints = {
                role: {"bytes": int(disk_budget.ROLE_FOOTPRINT_BYTES[role]),
                       "basis": "declared_default", "sample_count": None}
                for role in CHAIN_ROLES
            }
    required = sum(int(row["bytes"]) for row in footprints.values())
    devices: dict[int, dict[str, Any]] = {}
    admission_error = measured.get("status") != "measured"
    try:
        configured = (disk_budget.parse_role_targets(os.getenv("BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS"))
                      if role_targets is None else {role: Path(path) for role, path in role_targets.items()})
        if any(not path.is_dir() for path in configured.values()):
            raise disk_budget.ControlPlaneDiskBudgetError(
                "control_plane_disk_budget_role_target_unavailable"
            )
        projected = disk_budget.disk_headroom(
            target_root=mount, role_targets=configured, reservation_root=reservation_root,
            disk_usage=disk_usage, now=lambda: time.time() if now is None else float(now),
            device_of=device_of,
        )
        for target in projected["targets"]:
            role = target["role"]
            if role not in CHAIN_ROLES:
                continue
            device = target["device"]
            group = devices.setdefault(device, {
                "device": device, "path": str(configured.get(role, mount)),
                "roles": [], "required_bytes": 0,
                "available_bytes": target["available_bytes"],
            })
            group["roles"].append(role)
            group["required_bytes"] += int(footprints[role]["bytes"])
    except (disk_budget.ControlPlaneDiskBudgetError, OSError, TypeError, ValueError) as exc:
        admission_error = True
        measured = {"mount": str(mount), "status": "configuration_invalid", "blocker": str(exc)}
    for group in devices.values():
        group["roles"].sort()
        group["passed"] = group["available_bytes"] >= group["required_bytes"]
    passed = not admission_error and bool(devices) and all(group["passed"] for group in devices.values())
    return {
        'schema_version': 'control_plane_whole_chain_admission.v1',
        'status': 'admitted' if passed else 'waiting_for_capacity',
        'required_workspace_bytes': required,
        'required_workspace_basis': footprint_basis(footprints),
        'footprints': footprints,
        'measurement': measured,
        'devices': sorted(devices.values(), key=lambda row: row['device']),
        'provider_mutation_performed': False,
        'reservation_granted': False,
    }
