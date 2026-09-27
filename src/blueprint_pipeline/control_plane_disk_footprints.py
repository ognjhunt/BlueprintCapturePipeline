"""What each disk role really wrote: its sample history and its measured footprint.

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
import stat
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .control_plane_disk_ledger import (
    DEFAULT_RESERVATION_ROOT,
    ROLE_FOOTPRINT_BYTES,
    ROLE_NAME_RE,
    ControlPlaneDiskBudgetError,
    footprint_bytes,
    open_ledger_lock,
    prepare_ledger_root,
)

# How a measured job ended.  Only a completed job measured its whole footprint,
# so only "completed" samples shape admission; "failed" means the job raised,
# "blocked" means it returned a blocked result before finishing its work,
# "resumed" means the pass started from a workspace that already held an earlier
# pass's bytes, so its growth is not the job's footprint, and "incomplete" means
# part of the workspace could not be read, so the measurement under-counts.
FOOTPRINT_OUTCOMES = frozenset({"completed", "failed", "blocked", "resumed", "incomplete"})
# A workspace is fresh when bound if it did not exist or held less than this;
# only a fresh workspace's growth can be a completed job's whole footprint.
FRESH_WORKSPACE_MAX_BYTES = 1024**2

FOOTPRINT_HISTORY_DIRNAME = "history"
FOOTPRINT_SAMPLE_SCHEMA = "control_plane_disk_footprint_sample.v1"
MEASURED_MINIMUM_SAMPLES = 10
MEASURED_WINDOW = 50  # newest completed samples considered
MEASURED_HEADROOM = 1.25
MEASURED_FLOOR_BYTES = 64 * 1024**2
HISTORY_MAX_LINES = 200  # compaction keeps the newest lines
HISTORY_COMPACTION_BYTES = 64 * 1024
# Bookkeeping never waits indefinitely on the ledger lock: a caller that already
# holds it (an evictor running under admission) would otherwise deadlock itself
# and every worker queued behind it.  Giving up only loses one sample.
HISTORY_LOCK_WAIT_SECONDS = 5.0
_SAMPLE_MAX_BYTES = 1024


def _bounded_flock(descriptor: int, operation: int) -> None:
    """Take ``operation`` on the ledger lock or raise BlockingIOError after the bound."""

    deadline = time.monotonic() + HISTORY_LOCK_WAIT_SECONDS
    while True:
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _history_path(reservation_root: str | Path, role: str) -> Path:
    return (
        Path(reservation_root).expanduser()
        / FOOTPRINT_HISTORY_DIRNAME
        / f"{role}.jsonl"
    )


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

    lock = open_ledger_lock(ledger)
    try:
        _bounded_flock(lock, fcntl.LOCK_EX)
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
    lock = open_ledger_lock(ledger)
    try:
        # Appenders share the lock so compaction never drops a sample that
        # lands between its read and its replace.
        _bounded_flock(lock, fcntl.LOCK_SH)
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
        try:
            _compact_history(ledger, path)
        except OSError:
            pass  # the sample is recorded; the next append compacts


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
    baseline_bytes: int | None = None,
    fresh: bool | None = None,
) -> bool:
    """Append one sample to <ledger>/history/<role>.jsonl. Never raises; returns False on failure.

    A lost sample is harmless: admission keeps the older samples, or the
    declared ceiling while the history is short.  A "completed" sample whose
    workspace was not fresh is recorded as "resumed", so it never counts.
    """

    try:
        if (
            not isinstance(role, str)
            or not ROLE_NAME_RE.fullmatch(role)
            or role not in ROLE_FOOTPRINT_BYTES
            or outcome not in FOOTPRINT_OUTCOMES
            or (workload is not None
                and (not isinstance(workload, str) or not ROLE_NAME_RE.fullmatch(workload)))
            or not isinstance(observed_bytes, int)
            or isinstance(observed_bytes, bool)
            or not isinstance(reserved_bytes, int)
            or isinstance(reserved_bytes, bool)
            or (device is not None
                and (not isinstance(device, int) or isinstance(device, bool)))
            or (baseline_bytes is not None
                and (not isinstance(baseline_bytes, int) or isinstance(baseline_bytes, bool)))
            or (fresh is not None and not isinstance(fresh, bool))
        ):
            return False
        if outcome == "completed" and fresh is False:
            outcome = "resumed"
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
            "baseline_bytes": None if baseline_bytes is None else max(0, baseline_bytes),
            "fresh": fresh,
        }
        line = (
            json.dumps(sample, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(line) >= _SAMPLE_MAX_BYTES:
            return False
        ledger = prepare_ledger_root(Path(reservation_root).expanduser())
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


def role_footprints(
    roles: Iterable[str], *, reservation_root: str | Path = DEFAULT_RESERVATION_ROOT
) -> dict[str, dict[str, Any]]:
    """{role: {"bytes", "basis", "sample_count"}}: what each role's next reservation holds."""

    rows: dict[str, dict[str, Any]] = {}
    for role in roles:
        measured = measured_footprint(role, reservation_root=reservation_root)
        rows[role] = {
            "bytes": int(measured["bytes"]),
            "basis": measured["basis"],
            "sample_count": measured["sample_count"],
        }
    return rows


__all__ = [
    "FOOTPRINT_HISTORY_DIRNAME",
    "FOOTPRINT_OUTCOMES",
    "FOOTPRINT_SAMPLE_SCHEMA",
    "FRESH_WORKSPACE_MAX_BYTES",
    "HISTORY_COMPACTION_BYTES",
    "HISTORY_LOCK_WAIT_SECONDS",
    "HISTORY_MAX_LINES",
    "MEASURED_FLOOR_BYTES",
    "MEASURED_HEADROOM",
    "MEASURED_MINIMUM_SAMPLES",
    "MEASURED_WINDOW",
    "effective_footprint_bytes",
    "measured_footprint",
    "record_footprint_sample",
    "role_footprints",
]
