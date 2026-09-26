"""Fail-closed disk admission for control-plane materialization work.

The control plane has several independent workers which can all begin a large
copy at once.  Free-space sampling alone is therefore racy: each worker can
observe the same bytes.  This module serializes admission through a small
on-disk ledger and reserves the expected footprint before mutation begins.

A role's declared footprint is a ceiling, not a guess to reserve forever.  Each
reservation that measured its workspace appends what the job really wrote to
``<ledger>/history/<role>.jsonl``; once a role has enough completed samples,
admission reserves the p95 of recent samples times a headroom factor, clamped
between a small floor and the declared ceiling.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import stat
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
# Admission floors per role.  Preparation and compilation reserve their exact
# miss bytes at run time (references or runtime members the content stores do
# not already hold); these values are the typical hit-path footprint the intake
# checks before accepting a submission.
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
_ROLE_RE = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")
_OUTCOME_RE = re.compile(r"[a-z][a-z_]{1,31}\Z")

FOOTPRINT_HISTORY_DIRNAME = "history"
FOOTPRINT_SAMPLE_SCHEMA = "control_plane_disk_footprint_sample.v1"
MEASURED_MINIMUM_SAMPLES = 10
MEASURED_WINDOW = 50  # newest completed samples considered
MEASURED_HEADROOM = 1.25
MEASURED_FLOOR_BYTES = 64 * 1024**2
HISTORY_MAX_LINES = 200  # compaction keeps the newest lines
HISTORY_COMPACTION_BYTES = 64 * 1024
_SAMPLE_MAX_BYTES = 1024
# Pid liveness is the primary liveness signal; the TTL is only the backstop for a
# recycled pid.  Long roles outlive the default, so their entries must too, or
# the ledger deletes a running job's reservation as stale.
ROLE_TTL_SECONDS: Mapping[str, int] = {
    "cpu_prestage": 12 * 3600,
    "semantic_pretraining": 12 * 3600,
    "stage_replay": 6 * 3600,
    "control_plane_deploy": 4 * 3600,
}  # every other role keeps DEFAULT_TTL_SECONDS


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


def floor_bytes(total_bytes: int) -> int:
    """The admission floor, identical for the ledger, the controller and the preflight."""

    return max(
        _environment_int(
            "BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", DEFAULT_FLOOR_BYTES
        ),
        int(total_bytes * DEFAULT_FLOOR_FRACTION),
    )


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


def _entry_liveness(
    path: Path,
    *,
    device: int,
    observed_at: float,
    pid_alive: Callable[[int], bool],
) -> tuple[bool, int]:
    """(live, expected bytes) of one ledger entry: same device, unexpired, live pid."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return False, 0
        live = (
            int(value.get("device", -1)) == device
            and float(value.get("expires_at_epoch", 0)) > observed_at
            and pid_alive(int(value.get("pid", -1)))
        )
        amount = int(value.get("expected_bytes", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, 0
    if amount < 0:
        return False, 0
    return live, amount


def _load_live_reservations(
    root: Path,
    *,
    device: int,
    observed_at: float,
    pid_alive: Callable[[int], bool],
) -> tuple[int, list[str]]:
    # Only ``<token>.json`` entries in the ledger root are reservations; the
    # ``history`` directory and in-flight ``.reservation-*`` temporaries are not.
    reserved = 0
    stale: list[str] = []
    for path in sorted(root.glob("*.json")):
        live, amount = _entry_liveness(
            path, device=device, observed_at=observed_at, pid_alive=pid_alive
        )
        if live:
            reserved += amount
        else:
            stale.append(path.name)
    return reserved, stale


def live_reservations(
    reservation_root: str | Path,
    *,
    device: int,
    now: float,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> tuple[int, int]:
    """(bytes, count) of live reservations on ``device``; read-only (never deletes).

    Liveness is exactly the ledger's own: the entry's device, an unexpired TTL
    and a live pid.  A missing ledger holds no reservations.
    """

    root = Path(reservation_root).expanduser()
    reserved = 0
    count = 0
    for path in sorted(root.glob("*.json")):
        live, amount = _entry_liveness(
            path, device=device, observed_at=float(now), pid_alive=pid_alive
        )
        if live:
            reserved += amount
            count += 1
    return reserved, count


def _history_path(reservation_root: str | Path, role: str) -> Path:
    return (
        Path(reservation_root).expanduser()
        / FOOTPRINT_HISTORY_DIRNAME
        / f"{role}.jsonl"
    )


def _open_ledger_lock(ledger: Path) -> int:
    descriptor = os.open(ledger / ".lock", os.O_RDWR | os.O_CREAT, 0o660)
    try:
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o660:
            os.fchmod(descriptor, 0o660)
    except OSError:
        pass  # the installer owns the lock's mode; admission checks it strictly
    return descriptor


def _prepare_history_directory(ledger: Path) -> Path:
    history = ledger / FOOTPRINT_HISTORY_DIRNAME
    try:
        history.mkdir(mode=0o2770)
    except FileExistsError:
        pass
    else:
        # Root (deploy) and the runtime account both append here, so the new
        # directory takes the ledger's group and the ledger's setgid mode.
        try:
            group = ledger.stat().st_gid
            if history.stat().st_gid != group:
                os.chown(history, -1, group)
            history.chmod(0o2770)
        except OSError:
            pass
    if history.is_symlink() or not history.is_dir():
        raise ControlPlaneDiskBudgetError(
            "control_plane_disk_budget_history_directory_invalid"
        )
    return history


def _compact_history(ledger: Path, path: Path) -> None:
    """Keep the newest HISTORY_MAX_LINES samples, rewritten atomically under the lock."""

    lock = _open_ledger_lock(ledger)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = path.read_bytes()
        if len(data) <= HISTORY_COMPACTION_BYTES:
            return  # another writer compacted while this one waited
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            lines.pop()  # a torn line from a writer that died mid-append
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".history-", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.writelines(lines[-HISTORY_MAX_LINES:])
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o660)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        os.close(lock)


def _append_footprint_sample(ledger: Path, role: str, line: bytes) -> None:
    history = _prepare_history_directory(ledger)
    path = history / f"{role}.jsonl"
    lock = _open_ledger_lock(ledger)
    try:
        # Appenders share the lock so compaction never drops a sample that
        # lands between its read and its replace.
        fcntl.flock(lock, fcntl.LOCK_SH)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o660,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ControlPlaneDiskBudgetError(
                    "control_plane_disk_budget_history_file_invalid"
                )
            if stat.S_IMODE(metadata.st_mode) != 0o660:
                try:
                    os.fchmod(descriptor, 0o660)
                except OSError:
                    pass
            os.write(descriptor, line)
            size = os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)
    finally:
        os.close(lock)
    if size > HISTORY_COMPACTION_BYTES:
        _compact_history(ledger, path)


def record_footprint_sample(
    *,
    reservation_root: str | Path,
    role: str,
    observed_bytes: int,
    reserved_bytes: int,
    workload: str | None = None,
    outcome: str = "completed",
    duration_seconds: float | None = None,
    device: int | None = None,
    now: Callable[[], float] = time.time,
) -> bool:
    """Append one sample to <ledger>/history/<role>.jsonl. Never raises; returns False on failure.

    A lost sample is harmless: admission keeps the older samples, or the
    declared ceiling while the history is short.
    """

    try:
        if (
            not isinstance(role, str)
            or not _ROLE_RE.fullmatch(role)
            or role not in ROLE_FOOTPRINT_BYTES
            or not isinstance(outcome, str)
            or not _OUTCOME_RE.fullmatch(outcome)
            or (workload is not None
                and (not isinstance(workload, str) or not _ROLE_RE.fullmatch(workload)))
            or not isinstance(observed_bytes, int)
            or isinstance(observed_bytes, bool)
            or not isinstance(reserved_bytes, int)
            or isinstance(reserved_bytes, bool)
            or (device is not None
                and (not isinstance(device, int) or isinstance(device, bool)))
        ):
            return False
        duration = None
        if duration_seconds is not None:
            duration = round(max(0.0, float(duration_seconds)), 3)
            if not math.isfinite(duration):
                return False
        sample = {
            "schema_version": FOOTPRINT_SAMPLE_SCHEMA,
            "role": role,
            "workload": workload,
            "observed_bytes": max(0, observed_bytes),
            "reserved_bytes": max(0, reserved_bytes),
            "outcome": outcome,
            "duration_seconds": duration,
            "device": device,
            "recorded_at_epoch": round(float(now()), 3),
        }
        line = (
            json.dumps(sample, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(line) >= _SAMPLE_MAX_BYTES:
            return False
        ledger = _prepare_ledger_root(Path(reservation_root).expanduser())
        _append_footprint_sample(ledger, role, line)
    except Exception:  # never let bookkeeping break the job that was measured
        return False
    return True


def _completed_samples(reservation_root: str | Path, role: str) -> list[int]:
    """Observed bytes of the newest MEASURED_WINDOW completed samples, oldest first."""

    try:
        lines = _history_path(reservation_root, role).read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeError):
        return []
    values: list[int] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        observed = row.get("observed_bytes")
        if (
            row.get("schema_version") == FOOTPRINT_SAMPLE_SCHEMA
            and row.get("role") == role
            and row.get("outcome") == "completed"
            and type(observed) is int
            and observed >= 0
        ):
            values.append(observed)
    return values[-MEASURED_WINDOW:]


def measured_footprint(
    role: str, *, reservation_root: str | Path = DEFAULT_RESERVATION_ROOT
) -> dict[str, Any]:
    """{"role", "bytes", "basis", "sample_count", "declared_bytes", "p95_bytes"}.

    basis is "measured_p95" when at least MEASURED_MINIMUM_SAMPLES completed samples
    exist in the newest MEASURED_WINDOW, else "declared_default" (bytes == declared).
    p95 is nearest-rank: sorted values s, k = ceil(0.95 * n) - 1, p95 = s[k].
    bytes = max(MEASURED_FLOOR_BYTES, min(declared, ceil(p95 * MEASURED_HEADROOM))).
    declared = footprint_bytes(role) (env override honoured). Unreadable history -> declared.
    """

    declared = footprint_bytes(role)
    samples = _completed_samples(reservation_root, role)
    if len(samples) < MEASURED_MINIMUM_SAMPLES:
        return {
            "role": role,
            "bytes": declared,
            "basis": "declared_default",
            "sample_count": len(samples),
            "declared_bytes": declared,
            "p95_bytes": None,
        }
    ordered = sorted(samples)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "role": role,
        "bytes": max(
            MEASURED_FLOOR_BYTES,
            min(declared, math.ceil(p95 * MEASURED_HEADROOM)),
        ),
        "basis": "measured_p95",
        "sample_count": len(samples),
        "declared_bytes": declared,
        "p95_bytes": p95,
    }


def effective_footprint_bytes(
    role: str, *, reservation_root: str | Path = DEFAULT_RESERVATION_ROOT
) -> int:
    return int(measured_footprint(role, reservation_root=reservation_root)["bytes"])


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
        lock_mode = stat.S_IMODE(os.fstat(lock.fileno()).st_mode)
        if lock_mode != 0o660:
            try:
                os.chmod(  # nosec B103 - shared root/blueprint ledger lock
                    lock_path, 0o660
                )
            except OSError as exc:
                raise ControlPlaneDiskBudgetError(
                    f"control_plane_disk_budget_lock_mode_invalid:{lock_mode:04o}"
                ) from exc
            installed_mode = stat.S_IMODE(os.fstat(lock.fileno()).st_mode)
            if installed_mode != 0o660:
                raise ControlPlaneDiskBudgetError(
                    "control_plane_disk_budget_lock_mode_repair_failed:"
                    f"{installed_mode:04o}"
                )
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
        floor = floor_bytes(int(usage.total))
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
    floor = floor_bytes(int(usage.total))
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
    "FOOTPRINT_HISTORY_DIRNAME",
    "FOOTPRINT_SAMPLE_SCHEMA",
    "HISTORY_MAX_LINES",
    "MEASURED_FLOOR_BYTES",
    "MEASURED_HEADROOM",
    "MEASURED_MINIMUM_SAMPLES",
    "MEASURED_WINDOW",
    "ROLE_FOOTPRINT_BYTES",
    "ROLE_TTL_SECONDS",
    "disk_headroom",
    "effective_footprint_bytes",
    "floor_bytes",
    "footprint_bytes",
    "live_reservations",
    "measured_footprint",
    "record_footprint_sample",
    "reserve_control_plane_disk",
]
