"""Fail-closed disk admission for control-plane materialization work.

The control plane has several independent workers which can all begin a large
copy at once.  Free-space sampling alone is therefore racy: each worker can
observe the same bytes.  This module serializes admission through a small
on-disk ledger and reserves the expected footprint before mutation begins.

What each role reserves by default is its measured footprint
(``control_plane_disk_footprints``); the ledger's shared primitives live in
``control_plane_disk_ledger``.  Both are re-exported here for compatibility.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .control_plane_disk_footprints import (
    FOOTPRINT_HISTORY_DIRNAME,
    FOOTPRINT_OUTCOMES,
    FOOTPRINT_SAMPLE_SCHEMA,
    FRESH_WORKSPACE_MAX_BYTES,
    HISTORY_COMPACTION_BYTES,
    HISTORY_LOCK_WAIT_SECONDS,
    HISTORY_MAX_LINES,
    MEASURED_FLOOR_BYTES,
    MEASURED_HEADROOM,
    MEASURED_MINIMUM_SAMPLES,
    MEASURED_WINDOW,
    effective_footprint_bytes,
    measured_footprint,
    record_footprint_sample,
    role_footprints,
)
from .control_plane_disk_ledger import (
    DEFAULT_RESERVATION_ROOT,
    GIB,
    ROLE_FOOTPRINT_BYTES,
    ROLE_NAME_RE as _ROLE_RE,
    ControlPlaneDiskBudgetError,
    environment_int as _environment_int,
    footprint_bytes,
    open_ledger_lock,
    prepare_ledger_root as _prepare_ledger_root,
)
from .control_plane_disk_usage import tree_usage


DEFAULT_FLOOR_BYTES = 8 * GIB
DEFAULT_FLOOR_FRACTION = 0.05
CRITICAL_ROLES = frozenset({"control_plane_deploy"})
DEFAULT_CRITICAL_FLOOR_BYTES = GIB
DEFAULT_CRITICAL_FLOOR_FRACTION = 0.01
DEFAULT_TTL_SECONDS = 2 * 60 * 60
# Pid liveness is the primary liveness signal; the TTL is only the backstop for a
# recycled pid.  A job holds its reservation for at most its unit's
# TimeoutStartSec, so each role's entry must outlive that timeout (pinned by a
# test against deploy/systemd), or the ledger deletes a running job's
# reservation as stale.  The canary and launch dispatchers hold theirs through a
# paid run under a 5 h start timeout.
ROLE_TTL_SECONDS: Mapping[str, int] = {
    "g1_checkpoint_cache": 4 * 3600,
    "experiment_restore": 4 * 3600,
    "cpu_prestage": 12 * 3600,
    "semantic_pretraining": 12 * 3600,
    "stage_replay": 6 * 3600,
    "policy_canary_dispatch": 6 * 3600,
    "launch_dispatch": 6 * 3600,
    "scene_configuration_output": 6 * 3600,
    "control_plane_deploy": 4 * 3600,
    "evidence_offload": 4 * 3600,
}  # every other role keeps DEFAULT_TTL_SECONDS


def floor_bytes(total_bytes: int, *, role: str | None = None) -> int:
    """Keep the bulk band intact while critical work can use its protected portion."""

    if role in CRITICAL_ROLES:
        return max(
            _environment_int(
                "BLUEPRINT_CONTROL_PLANE_DISK_CRITICAL_FLOOR_BYTES",
                DEFAULT_CRITICAL_FLOOR_BYTES,
            ),
            int(total_bytes * DEFAULT_CRITICAL_FLOOR_FRACTION),
        )
    return max(
        _environment_int(
            "BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", DEFAULT_FLOOR_BYTES
        ),
        int(total_bytes * DEFAULT_FLOOR_FRACTION),
    )


def parse_role_targets(raw: str | None) -> dict[str, Path]:
    """Parse role-specific absolute roots from a comma-separated unit setting."""

    if raw is None or raw == "":
        return {}
    targets: dict[str, Path] = {}
    for item in raw.split(","):
        role, separator, value = item.partition("=")
        path = Path(value)
        if (not separator or role not in ROLE_FOOTPRINT_BYTES or role in targets
                or not value or not path.is_absolute()):
            raise ControlPlaneDiskBudgetError("control_plane_disk_budget_role_targets_invalid")
        targets[role] = path
    return targets


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def target_device(target_root: str | Path) -> int:
    """The device a reservation against ``target_root`` is recorded under.

    Admission records the device of the target's nearest existing ancestor, so
    anything that projects admission for a path derives its device the same way.
    """

    return _existing_ancestor(Path(target_root)).stat().st_dev


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


def _entry_liveness(
    path: Path,
    *,
    device: int,
    observed_at: float,
    pid_alive: Callable[[int], bool],
    strict: bool = False,
) -> tuple[bool, int, bool]:
    """(live here, expected bytes, live elsewhere) for one ledger entry.

    ``strict`` refuses an entry it cannot read (typed ledger_unreadable) rather
    than treating it as stale: a projection must never under-count reservations.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False, 0, False  # released between the listing and the read
    except OSError as exc:
        if strict:
            raise ControlPlaneDiskBudgetError("control_plane_disk_budget_ledger_unreadable") from exc
        return False, 0, False
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            return False, 0, False
        entry_device = int(value.get("device", -1))
        active = (
            entry_device >= 0
            and float(value.get("expires_at_epoch", 0)) > observed_at
            and pid_alive(int(value.get("pid", -1)))
        )
        amount = int(value.get("expected_bytes", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, 0, False
    if amount < 0:
        return False, 0, False
    return active and entry_device == device, amount, active and entry_device != device


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
    try:
        with os.scandir(root) as listing:
            names = sorted(entry.name for entry in listing if entry.name.endswith(".json"))
    except OSError as exc:
        raise ControlPlaneDiskBudgetError("control_plane_disk_budget_ledger_unreadable") from exc
    for name in names:
        live, amount, foreign_live = _entry_liveness(
            root / name, device=device, observed_at=observed_at,
            pid_alive=pid_alive, strict=True,
        )
        if live:
            reserved += amount
        elif not foreign_live:
            stale.append(name)
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
    and a live pid.  A missing ledger holds no reservations; a ledger (or entry)
    that cannot be read raises the typed ledger_unreadable, never reads as empty.
    """

    root = Path(reservation_root).expanduser()
    try:
        with os.scandir(root) as listing:
            names = sorted(entry.name for entry in listing if entry.name.endswith(".json"))
    except FileNotFoundError:
        return 0, 0
    except OSError as exc:
        raise ControlPlaneDiskBudgetError("control_plane_disk_budget_ledger_unreadable") from exc
    reserved = 0
    count = 0
    for name in names:
        live, amount, _foreign_live = _entry_liveness(
            root / name, device=device, observed_at=float(now), pid_alive=pid_alive, strict=True
        )
        if live:
            reserved += amount
            count += 1
    return reserved, count


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
    reservation_root: Path | None = None
    device: int | None = None
    footprint_basis: str = "caller_exact"
    footprint_sample_count: int | None = None
    workload: str | None = None
    workspace: Path | None = None
    baseline_bytes: int = 0
    # None until a workspace is bound or an observation arrives: only then is
    # there a measurement worth adding to the role's history.
    peak_delta_bytes: int | None = None
    # Whether the workspace was fresh when bound (absent or nearly empty); a
    # pass over an already-populated workspace resumes a job, it does not run one.
    fresh: bool = True
    # Set when any walk of the workspace hit an unreadable entry: the bytes under
    # it were not counted, so the sample cannot stand for the job's footprint.
    measurement_incomplete: bool = False
    started_at_epoch: float = 0.0
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)

    def renew(self) -> None:
        """Extend a live reservation under the admission lock without changing its bytes."""

        if self.released or self.reservation_root is None:
            raise ControlPlaneDiskBudgetError("control_plane_disk_budget_reservation_released")
        ledger = self.reservation_root
        with os.fdopen(open_ledger_lock(ledger, require_mode=True), "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ValueError("reservation is not a regular file")
                    payload = json.loads(stream.read(16 * 1024 + 1))
                if (
                    not isinstance(payload, dict)
                    or payload.get("token") != self.token
                    or payload.get("role") != self.role
                    or payload.get("pid") != os.getpid()
                    or payload.get("device") != self.device
                    or payload.get("expected_bytes") != self.expected_bytes
                ):
                    raise ValueError("reservation identity changed")
            except (OSError, ValueError, TypeError) as exc:
                raise ControlPlaneDiskBudgetError(
                    "control_plane_disk_budget_reservation_renewal_invalid"
                ) from exc
            moment = self.clock()
            if float(payload.get("expires_at_epoch", 0)) <= moment:
                raise ControlPlaneDiskBudgetError("control_plane_disk_budget_reservation_expired")
            payload["expires_at_epoch"] = moment + self.ttl_seconds
            descriptor, temporary_name = tempfile.mkstemp(prefix=".reservation-", dir=ledger)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.chmod(0o640)
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def bind_workspace(self, path: str | Path, *, fresh: bool | None = None) -> None:
        """Measure growth of ``path`` from now on (a workspace created after admission).

        ``fresh`` is inferred from the baseline unless the caller knows better:
        ``True`` right after it cleared the workspace, ``False`` when it resumes.
        """

        self.workspace = Path(path)
        self.baseline_bytes, complete = _measure_workspace(self.workspace)
        self.measurement_incomplete = self.measurement_incomplete or not complete
        self.fresh = _is_fresh(self.baseline_bytes) if fresh is None else bool(fresh)
        if self.peak_delta_bytes is None:
            self.peak_delta_bytes = 0

    def sample(self) -> int | None:
        """Fold the workspace's current growth into the peak; never raises."""

        if self.workspace is None:
            return self.peak_delta_bytes
        current, complete = _measure_workspace(self.workspace)
        if not complete:
            self.measurement_incomplete = True
            if current == 0:
                return self.peak_delta_bytes
        delta = max(0, current - self.baseline_bytes)
        self.peak_delta_bytes = max(self.peak_delta_bytes or 0, delta)
        return self.peak_delta_bytes

    def observe(self, observed_bytes: int) -> None:
        """Fold an external measurement (the deploy's staged trees) into the peak."""

        if isinstance(observed_bytes, int) and not isinstance(observed_bytes, bool):
            self.peak_delta_bytes = max(
                self.peak_delta_bytes or 0, max(0, observed_bytes)
            )

    def release(self, *, outcome: str = "completed") -> None:
        """Free the reservation and record its sample under ``outcome``.

        A caller that catches its job's failure passes ``outcome="failed"``, and
        one whose job returned a blocked result passes ``outcome="blocked"``;
        neither shapes admission.  Idempotent: the first release's outcome stands.
        """

        self._finish(outcome)

    def _finish(self, outcome: str) -> None:
        if self.released:
            return
        self.path.unlink(missing_ok=True)
        self.released = True
        if outcome not in FOOTPRINT_OUTCOMES:
            outcome = "failed"  # an unrecognized label must never count as completed
        try:
            if self.workspace is not None:
                self.sample()
            if outcome == "completed" and self.measurement_incomplete:
                outcome = "incomplete"
            if self.peak_delta_bytes is None or self.reservation_root is None:
                return
            record_footprint_sample(
                reservation_root=self.reservation_root,
                role=self.role,
                observed_bytes=self.peak_delta_bytes,
                reserved_bytes=self.expected_bytes,
                workload=self.workload,
                outcome=outcome,
                duration_seconds=max(0.0, float(self.clock()) - self.started_at_epoch),
                device=self.device,
                now=self.clock,
                baseline_bytes=self.baseline_bytes if self.workspace is not None else None,
                fresh=self.fresh,
            )
        except Exception:  # the reservation is released; a lost sample is harmless
            pass

    def __enter__(self) -> "DiskReservation":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._finish("failed" if exc_type is not None else "completed")

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
            "footprint_basis": self.footprint_basis,
            "footprint_sample_count": self.footprint_sample_count,
            "workload": self.workload,
        }


def _is_fresh(baseline_bytes: int) -> bool:
    return baseline_bytes < FRESH_WORKSPACE_MAX_BYTES


def _measure_workspace(path: Path) -> tuple[int, bool]:
    """(allocated bytes, complete) of a workspace walk; never raises.

    The walk is incomplete when any entry could not be read (its bytes were not
    counted) or when the walk itself failed.
    """

    try:
        usage = tree_usage(path)
    except Exception:  # a measurement must never break the job it measures
        return 0, False
    return usage.allocated_bytes, usage.unreadable == 0


def _snapshot(
    *,
    target_root: str | Path,
    reservation_root: str | Path,
    disk_usage: Callable[[str | os.PathLike[str]], Any],
    now: Callable[[], float],
    pid_alive: Callable[[int], bool],
    device_of: Callable[[Path], int] = target_device,
) -> tuple[Path, Any, int, int, list[str]]:
    target = _existing_ancestor(Path(target_root))
    usage = disk_usage(target)
    ledger = _prepare_ledger_root(Path(reservation_root).expanduser())
    device = device_of(target)
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
    ttl_seconds: int | None = None,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    now: Callable[[], float] = time.time,
    pid_alive: Callable[[int], bool] = _pid_alive,
    evictor: Callable[[int], Any] | None = None,
    workspace: str | Path | None = None,
    workload: str | None = None,
    fresh: bool | None = None,
    minimum_bytes: int | None = None,
    device_of: Callable[[Path], int] = target_device,
) -> DiskReservation:
    """Atomically reserve disk headroom or raise a typed refusal.

    ``expected_bytes=None`` reserves the role's measured footprint (or its
    declared ceiling while the history is short).  ``workspace`` is the per-job
    directory whose growth is recorded as the role's footprint sample when the
    reservation is released; ``target_root`` remains the tree whose filesystem
    admission is computed against.  Only a workspace that was fresh when bound
    (inferred from its baseline, or ``fresh`` from the caller) can record a
    completed sample; a resumed pass records "resumed".  ``minimum_bytes`` is
    what the job itself declares it will write: the measured footprint never
    reserves less.
    """

    if not _ROLE_RE.fullmatch(role) or role not in ROLE_FOOTPRINT_BYTES:
        raise ControlPlaneDiskBudgetError(
            f"control_plane_disk_budget_role_invalid:{role}"
        )
    if ttl_seconds is None:
        ttl_seconds = ROLE_TTL_SECONDS.get(role, DEFAULT_TTL_SECONDS)
    if workload is not None and (
        not isinstance(workload, str) or not _ROLE_RE.fullmatch(workload)
    ):
        raise ControlPlaneDiskBudgetError(
            "control_plane_disk_budget_workload_invalid"
        )
    if minimum_bytes is not None and (
        not isinstance(minimum_bytes, int) or isinstance(minimum_bytes, bool) or minimum_bytes < 0
    ):
        raise ControlPlaneDiskBudgetError(
            "control_plane_disk_budget_reservation_invalid"
        )
    if expected_bytes is None:
        measured = measured_footprint(role, reservation_root=reservation_root)
        need = measured["bytes"]
        basis = str(measured["basis"])
        sample_count: int | None = int(measured["sample_count"])
        if minimum_bytes is not None and minimum_bytes > need:
            need, basis = minimum_bytes, "declared_minimum"
    else:
        need, basis, sample_count = expected_bytes, "caller_exact", None
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
    # The baseline walk can take a while on a large workspace, so it happens
    # before the ledger lock every other worker's admission waits on.
    bound = None if workspace is None else Path(workspace).expanduser()
    baseline, baseline_complete = (0, True) if bound is None else _measure_workspace(bound)
    # A caller resuming an earlier pass says so; otherwise a nearly empty
    # workspace is fresh and one already holding bytes is not.
    workspace_fresh = _is_fresh(baseline) if fresh is None else bool(fresh)
    ledger = _prepare_ledger_root(Path(reservation_root).expanduser())
    # The lock is opened without following a symlink and must be a 0660 regular
    # file; only its owner repairs the mode, through the descriptor.
    with os.fdopen(open_ledger_lock(ledger, require_mode=True), "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger, usage, device, reserved, stale = _snapshot(
            target_root=target_root,
            reservation_root=ledger,
            disk_usage=disk_usage,
            now=now,
            pid_alive=pid_alive,
            device_of=device_of,
        )
        for name in stale:
            (ledger / name).unlink(missing_ok=True)
        floor = floor_bytes(int(usage.total), role=role)
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
        started = now()
        payload = {
            "schema_version": "control_plane_disk_reservation.v1",
            "token": token,
            "role": role,
            "pid": os.getpid(),
            "device": device,
            "expected_bytes": need,
            "created_at_epoch": started,
            "expires_at_epoch": started + ttl_seconds,
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
        reservation_root=ledger,
        device=device,
        footprint_basis=basis,
        footprint_sample_count=sample_count,
        workload=workload,
        workspace=bound,
        baseline_bytes=baseline,
        peak_delta_bytes=None if bound is None else 0,
        fresh=workspace_fresh,
        measurement_incomplete=not baseline_complete,
        started_at_epoch=float(started),
        ttl_seconds=ttl_seconds,
        clock=now,
    )


def disk_headroom(
    *,
    target_root: str | Path,
    reservation_root: str | Path = DEFAULT_RESERVATION_ROOT,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    now: Callable[[], float] = time.time,
    pid_alive: Callable[[int], bool] = _pid_alive,
    role_targets: Mapping[str, str | Path] | None = None,
    device_of: Callable[[Path], int] = target_device,
) -> dict[str, Any]:
    """Return a path-free admission projection suitable for an intake API."""

    targets_by_role: dict[str, Path] = {}
    for role, value in (role_targets or {}).items():
        path = Path(value)
        if role not in ROLE_FOOTPRINT_BYTES or not path.is_absolute():
            raise ControlPlaneDiskBudgetError("control_plane_disk_budget_role_targets_invalid")
        targets_by_role[role] = path
    ledger = _prepare_ledger_root(Path(reservation_root).expanduser())
    with os.fdopen(open_ledger_lock(ledger), "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        _ledger, usage, default_device, reserved, _stale = _snapshot(
            target_root=target_root,
            reservation_root=ledger,
            disk_usage=disk_usage,
            now=now,
            pid_alive=pid_alive,
            device_of=device_of,
        )
        devices = {default_device: (usage, reserved)}
        role_devices = dict.fromkeys(ROLE_FOOTPRINT_BYTES, default_device)
        for role, path in targets_by_role.items():
            device = device_of(_existing_ancestor(path))
            role_devices[role] = device
            if device not in devices:
                _ledger, target_usage, _device, target_reserved, _stale = _snapshot(
                    target_root=path, reservation_root=ledger, disk_usage=disk_usage,
                    now=now, pid_alive=pid_alive, device_of=device_of,
                )
                devices[device] = (target_usage, target_reserved)
    floor = floor_bytes(int(usage.total))
    available = max(0, int(usage.free) - floor - reserved)
    footprints = role_footprints(ROLE_FOOTPRINT_BYTES, reservation_root=ledger)
    targets = []
    for role, footprint in sorted(footprints.items()):
        device = role_devices[role]
        target_usage, target_reserved = devices[device]
        target_floor = floor_bytes(int(target_usage.total), role=role)
        target_available = max(0, int(target_usage.free) - target_floor - target_reserved)
        targets.append({
            "role": role, "device": device, "free_bytes": int(target_usage.free),
            "floor_bytes": target_floor, "reserved_bytes": target_reserved,
            "available_bytes": target_available, "refused": footprint["bytes"] > target_available,
        })
    refused = sorted(row["role"] for row in targets if row["refused"])
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
        "targets": targets,
        "footprints": footprints,
    }


__all__ = [
    "CRITICAL_ROLES",
    "ControlPlaneDiskBudgetError",
    "DEFAULT_CRITICAL_FLOOR_BYTES",
    "DEFAULT_CRITICAL_FLOOR_FRACTION",
    "DEFAULT_RESERVATION_ROOT",
    "DiskReservation",
    "FOOTPRINT_HISTORY_DIRNAME",
    "FOOTPRINT_OUTCOMES",
    "FOOTPRINT_SAMPLE_SCHEMA",
    "HISTORY_COMPACTION_BYTES",
    "HISTORY_LOCK_WAIT_SECONDS",
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
    "parse_role_targets",
    "record_footprint_sample",
    "reserve_control_plane_disk",
    "role_footprints",
    "target_device",
]
