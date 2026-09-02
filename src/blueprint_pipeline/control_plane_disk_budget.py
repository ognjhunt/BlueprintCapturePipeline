"""Fail-closed disk admission for control-plane materialization work.

The control plane has several independent workers which can all begin a large
copy at once.  Free-space sampling alone is therefore racy: each worker can
observe the same bytes.  This module serializes admission through a small
on-disk ledger and reserves the expected footprint before mutation begins.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GIB = 1024**3
DEFAULT_RESERVATION_ROOT = Path(
    "/var/lib/blueprint/pipeline-control-plane/disk-reservations"
)
DEFAULT_FLOOR_BYTES = 8 * GIB
DEFAULT_FLOOR_FRACTION = 0.05
DEFAULT_TTL_SECONDS = 2 * 60 * 60
ROLE_FOOTPRINT_BYTES: Mapping[str, int] = {
    "control_plane_deploy": 2 * GIB,
    "launch_preparation": 6 * GIB,
    "episode_compilation": 6 * GIB,
    "launch_activation": 2 * GIB,
    "launch_dispatch": 2 * GIB,
    "policy_canary_dispatch": 2 * GIB,
}
_ROLE_RE = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")


class ControlPlaneDiskBudgetError(RuntimeError):
    """A write-heavy operation was refused before it mutated its output."""


def _environment_int(name: str, default: int) -> int:
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
    if role not in ROLE_FOOTPRINT_BYTES:
        raise ControlPlaneDiskBudgetError(
            f"control_plane_disk_budget_role_invalid:{role}"
        )
    name = f"BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_{role.upper()}_BYTES"
    return _environment_int(name, ROLE_FOOTPRINT_BYTES[role])


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prepare_ledger_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o2770)
    try:
        root.chmod(0o2770)
    except PermissionError:
        pass
    return root.resolve(strict=True)


def _load_live_reservations(
    root: Path,
    *,
    device: int,
    observed_at: float,
    pid_alive: Callable[[int], bool],
) -> tuple[int, list[str]]:
    reserved = 0
    stale: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            live = (
                isinstance(value, dict)
                and int(value.get("device", -1)) == device
                and float(value.get("expires_at_epoch", 0)) > observed_at
                and pid_alive(int(value.get("pid", -1)))
            )
            amount = int(value.get("expected_bytes", -1))
            if amount < 0:
                live = False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            live = False
            amount = 0
        if live:
            reserved += amount
        else:
            stale.append(path.name)
    return reserved, stale


@dataclass
class DiskReservation:
    role: str
    expected_bytes: int
    free_bytes: int
    floor_bytes: int
    reserved_bytes: int
    available_bytes: int
    path: Path
    token: str
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.path.unlink(missing_ok=True)
        self.released = True

    def __enter__(self) -> "DiskReservation":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "control_plane_disk_reservation.v1",
            "role": self.role,
            "expected_bytes": self.expected_bytes,
            "free_bytes_at_admission": self.free_bytes,
            "floor_bytes": self.floor_bytes,
            "reserved_bytes_before_admission": self.reserved_bytes,
            "available_bytes_before_admission": self.available_bytes,
            "reservation_token": self.token,
        }


def _snapshot(
    *,
    target_root: str | Path,
    reservation_root: str | Path,
    disk_usage: Callable[[str | os.PathLike[str]], Any],
    now: Callable[[], float],
    pid_alive: Callable[[int], bool],
) -> tuple[Path, Any, int, int, list[str]]:
    target = _existing_ancestor(Path(target_root))
    usage = disk_usage(target)
    ledger = _prepare_ledger_root(Path(reservation_root).expanduser())
    device = target.stat().st_dev
    observed_at = now()
    reserved, stale = _load_live_reservations(
        ledger,
        device=device,
        observed_at=observed_at,
        pid_alive=pid_alive,
    )
    return ledger, usage, device, reserved, stale


def reserve_control_plane_disk(
    role: str,
    *,
    target_root: str | Path,
    expected_bytes: int | None = None,
    reservation_root: str | Path = DEFAULT_RESERVATION_ROOT,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    now: Callable[[], float] = time.time,
    pid_alive: Callable[[int], bool] = _pid_alive,
    evictor: Callable[[int], Any] | None = None,
) -> DiskReservation:
    """Atomically reserve disk headroom or raise a typed refusal."""

    if not _ROLE_RE.fullmatch(role) or role not in ROLE_FOOTPRINT_BYTES:
        raise ControlPlaneDiskBudgetError(
            f"control_plane_disk_budget_role_invalid:{role}"
        )
    need = footprint_bytes(role) if expected_bytes is None else expected_bytes
    if (
        not isinstance(need, int)
        or isinstance(need, bool)
        or need <= 0
        or not isinstance(ttl_seconds, int)
        or ttl_seconds <= 0
    ):
        raise ControlPlaneDiskBudgetError(
            "control_plane_disk_budget_reservation_invalid"
        )
    ledger = _prepare_ledger_root(Path(reservation_root).expanduser())
    lock_path = ledger / ".lock"
    with lock_path.open("a+b") as lock:
        os.chmod(lock_path, 0o660)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger, usage, device, reserved, stale = _snapshot(
            target_root=target_root,
            reservation_root=ledger,
            disk_usage=disk_usage,
            now=now,
            pid_alive=pid_alive,
        )
        for name in stale:
            (ledger / name).unlink(missing_ok=True)
        floor = max(
            _environment_int(
                "BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES",
                DEFAULT_FLOOR_BYTES,
            ),
            int(usage.total * DEFAULT_FLOOR_FRACTION),
        )
        available = max(0, int(usage.free) - floor - reserved)
        if need > available and evictor is not None:
            evictor(need - available)
            usage = disk_usage(_existing_ancestor(Path(target_root)))
            available = max(0, int(usage.free) - floor - reserved)
        if need > available:
            raise ControlPlaneDiskBudgetError(
                f"control_plane_disk_budget_exceeded:{role}:"
                f"need_bytes={need}:available_bytes={available}:"
                f"free_bytes={int(usage.free)}:floor_bytes={floor}:"
                f"reserved_bytes={reserved}"
            )
        token = uuid.uuid4().hex
        payload = {
            "schema_version": "control_plane_disk_reservation.v1",
            "token": token,
            "role": role,
            "pid": os.getpid(),
            "device": device,
            "expected_bytes": need,
            "created_at_epoch": now(),
            "expires_at_epoch": now() + ttl_seconds,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".reservation-", dir=ledger
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o640)
            path = ledger / f"{token}.json"
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return DiskReservation(
        role=role,
        expected_bytes=need,
        free_bytes=int(usage.free),
        floor_bytes=floor,
        reserved_bytes=reserved,
        available_bytes=available,
        path=path,
        token=token,
    )


def disk_headroom(
    *,
    target_root: str | Path,
    reservation_root: str | Path = DEFAULT_RESERVATION_ROOT,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    now: Callable[[], float] = time.time,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> dict[str, Any]:
    """Return a path-free admission projection suitable for an intake API."""

    ledger = _prepare_ledger_root(Path(reservation_root).expanduser())
    with (ledger / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        _ledger, usage, _device, reserved, _stale = _snapshot(
            target_root=target_root,
            reservation_root=ledger,
            disk_usage=disk_usage,
            now=now,
            pid_alive=pid_alive,
        )
    floor = max(
        _environment_int(
            "BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", DEFAULT_FLOOR_BYTES
        ),
        int(usage.total * DEFAULT_FLOOR_FRACTION),
    )
    available = max(0, int(usage.free) - floor - reserved)
    refused = sorted(
        role
        for role in ROLE_FOOTPRINT_BYTES
        if footprint_bytes(role) > available
    )
    status = (
        "exhausted"
        if len(refused) == len(ROLE_FOOTPRINT_BYTES)
        else "low" if refused else "ok"
    )
    return {
        "schema_version": "control_plane_disk_headroom.v1",
        "status": status,
        "free_bytes": int(usage.free),
        "floor_bytes": floor,
        "reserved_bytes": reserved,
        "available_bytes": available,
        "refused_roles": refused,
    }


__all__ = [
    "ControlPlaneDiskBudgetError",
    "DEFAULT_RESERVATION_ROOT",
    "DiskReservation",
    "ROLE_FOOTPRINT_BYTES",
    "disk_headroom",
    "footprint_bytes",
    "reserve_control_plane_disk",
]
