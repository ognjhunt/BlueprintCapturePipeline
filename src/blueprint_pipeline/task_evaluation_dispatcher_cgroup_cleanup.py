"""Reap dispatcher descendants while preserving exact live TTL watchdogs.

The task-evaluation dispatcher uses ``KillMode=process`` so systemd does not
destroy an independently retained watchdog when the oneshot caller exits.
This ExecStopPost helper restores control-group hygiene by terminating every
other remaining process.  A child is preserved only when its exact command,
PID, prefix, deadline, handoff, and armed evidence all agree under the governed
launch-run root.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .groot_oscar_runpod_watchdog import EVIDENCE_NAME
from .vast_independent_watchdog_control import (
    HANDOFF_NAME,
    HANDOFF_SCHEMA,
    SYSTEMD_KILL_MODE_PROCESS_SURVIVAL,
)


SCHEMA_VERSION = "task_evaluation_dispatcher_cgroup_cleanup.v1"
MAX_CGROUP_PROCESSES = 4096
MAX_HANDOFF_FILES = 4096
MAX_CONTROL_FILE_BYTES = 128 * 1024
WATCHDOG_MODULE = "blueprint_pipeline.groot_oscar_runpod_watchdog"


class DispatcherCgroupCleanupError(ValueError):
    """The dispatcher cgroup could not be safely reconciled."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
            or metadata.st_size <= 0
            or metadata.st_size > MAX_CONTROL_FILE_BYTES
        ):
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _own_cgroup_procs_path(
    *, proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")
) -> Path:
    try:
        rows = (proc_root / "self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DispatcherCgroupCleanupError("dispatcher_cgroup_identity_unreadable") from exc
    matches = [row.split("::", 1)[1] for row in rows if row.startswith("0::")]
    if len(matches) != 1:
        raise DispatcherCgroupCleanupError("dispatcher_cgroup_identity_invalid")
    relative = Path(matches[0].lstrip("/"))
    if ".." in relative.parts:
        raise DispatcherCgroupCleanupError("dispatcher_cgroup_identity_invalid")
    return cgroup_root / relative / "cgroup.procs"


def _cgroup_pids(path: Path) -> list[int]:
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DispatcherCgroupCleanupError("dispatcher_cgroup_processes_unreadable") from exc
    pids: list[int] = []
    for row in rows:
        if not row.isascii() or not row.isdigit() or int(row) <= 0:
            raise DispatcherCgroupCleanupError("dispatcher_cgroup_processes_invalid")
        pids.append(int(row))
    if len(pids) > MAX_CGROUP_PROCESSES or len(pids) != len(set(pids)):
        raise DispatcherCgroupCleanupError("dispatcher_cgroup_processes_invalid")
    return pids


def _cmdline(proc_root: Path, pid: int) -> tuple[str, ...]:
    try:
        payload = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ()
    if len(payload) > 64 * 1024:
        return ()
    return tuple(item.decode("utf-8", "replace") for item in payload.split(b"\0") if item)


def _process_start_ticks(proc_root: Path, pid: int) -> str | None:
    try:
        payload = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing = payload.rfind(")")
    if closing < 0:
        return None
    fields = payload[closing + 1 :].strip().split()
    return fields[19] if len(fields) > 19 and fields[19].isdigit() else None


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _valid_watchdog_handoff(
    *,
    pid: int,
    state_root: Path,
    proc_root: Path,
    now_epoch: float,
) -> dict[str, Any] | None:
    argv = _cmdline(proc_root, pid)
    if not any(
        argv[index] == "-m" and argv[index + 1] == WATCHDOG_MODULE
        for index in range(len(argv) - 1)
    ):
        return None
    observed = 0
    for path in state_root.rglob(HANDOFF_NAME):
        observed += 1
        if observed > MAX_HANDOFF_FILES:
            raise DispatcherCgroupCleanupError("dispatcher_watchdog_handoff_count_exceeded")
        handoff = _read_json(path)
        if handoff.get("watchdog_pid") != pid or handoff.get("status") not in {
            "armed",
            "retained_until_hard_ttl",
        }:
            continue
        deadline = handoff.get("watchdog_deadline_epoch")
        prefix = str(handoff.get("pod_name_prefix") or "")
        out_dir = Path(str(handoff.get("watchdog_out_dir") or path.parent))
        try:
            evidence_deadline = float(
                _read_json(out_dir / EVIDENCE_NAME).get("deadline_epoch") or 0
            )
        except (TypeError, ValueError):
            evidence_deadline = 0
        if (
            not isinstance(deadline, (int, float))
            or isinstance(deadline, bool)
            or float(deadline) <= now_epoch
            or not prefix.startswith("blueprint-")
            or handoff.get("schema_version") != HANDOFF_SCHEMA
            or handoff.get("raw_secret_values_recorded") is not False
            or handoff.get("caller_exit_survival_contract")
            != SYSTEMD_KILL_MODE_PROCESS_SURVIVAL
            or out_dir.is_symlink()
            or not _inside(out_dir, state_root)
        ):
            continue
        evidence = _read_json(out_dir / EVIDENCE_NAME)
        if (
            evidence.get("status") != "armed"
            or evidence.get("independent_process") is not True
            or evidence.get("pid") != pid
            or evidence.get("provider") != "vast"
            or evidence.get("pod_name_prefix") != prefix
            or evidence_deadline != float(deadline)
            or evidence.get("raw_secret_values_recorded") is not False
        ):
            continue
        return {
            "pid": pid,
            "pod_name_prefix": prefix,
            "watchdog_deadline_epoch": float(deadline),
            "handoff_path": str(path),
            "evidence_path": str(out_dir / EVIDENCE_NAME),
        }
    return None


def cleanup_dispatcher_cgroup(
    *,
    state_root: str | Path,
    receipt_dir: str | Path,
    cgroup_procs_path: str | Path | None = None,
    proc_root: str | Path = "/proc",
    self_pid: int | None = None,
    killer: Callable[[int, int], None] = os.kill,
    process_alive: Callable[[int], bool] | None = None,
    process_identity: Callable[[Path, int], str | None] = _process_start_ticks,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Preserve exact armed watchdogs and reap every other cgroup child."""

    root = Path(state_root).expanduser().resolve()
    receipts = Path(receipt_dir).expanduser().resolve()
    proc = Path(proc_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or receipts.is_symlink():
        raise DispatcherCgroupCleanupError("dispatcher_cgroup_cleanup_root_invalid")
    procs_path = (
        Path(cgroup_procs_path).expanduser().resolve()
        if cgroup_procs_path is not None
        else _own_cgroup_procs_path(proc_root=proc)
    )
    current_pid = int(self_pid if self_pid is not None else os.getpid())
    pids = [pid for pid in _cgroup_pids(procs_path) if pid != current_pid]
    identities = {pid: process_identity(proc, pid) for pid in pids}
    now = float(clock())
    preserved: list[dict[str, Any]] = []
    reap: list[int] = []
    for pid in pids:
        watchdog = _valid_watchdog_handoff(
            pid=pid,
            state_root=root,
            proc_root=proc,
            now_epoch=now,
        )
        if watchdog is not None:
            preserved.append(watchdog)
        else:
            reap.append(pid)
    alive = process_alive or (lambda pid: _default_process_alive(pid, killer=killer))

    def same_cgroup_identity(pid: int) -> bool:
        return bool(
            identities.get(pid)
            and process_identity(proc, pid) == identities[pid]
            and pid in _cgroup_pids(procs_path)
        )

    terminated: list[int] = []
    killed: list[int] = []
    signal_failures: list[dict[str, Any]] = []
    identity_changed: list[int] = []
    for pid in reap:
        if not same_cgroup_identity(pid):
            identity_changed.append(pid)
            continue
        try:
            killer(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError as exc:
            signal_failures.append(
                {"pid": pid, "signal": "SIGTERM", "error_type": type(exc).__name__}
            )
            continue
        terminated.append(pid)
    wait_deadline = time.monotonic() + 2.0
    while time.monotonic() < wait_deadline and any(alive(pid) for pid in terminated):
        sleeper(0.05)
    for pid in terminated:
        if not alive(pid):
            continue
        if not same_cgroup_identity(pid):
            identity_changed.append(pid)
            continue
        try:
            killer(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError as exc:
            signal_failures.append(
                {"pid": pid, "signal": "SIGKILL", "error_type": type(exc).__name__}
            )
            continue
        killed.append(pid)
    final_cgroup_pids = [
        pid for pid in _cgroup_pids(procs_path) if pid != current_pid
    ]
    preserved_pids = {int(row["pid"]) for row in preserved}
    preserved_watchdog_failures: list[int] = []
    for row in preserved:
        pid = int(row["pid"])
        if (
            pid not in final_cgroup_pids
            or not alive(pid)
            or process_identity(proc, pid) != identities.get(pid)
            or _valid_watchdog_handoff(
                pid=pid,
                state_root=root,
                proc_root=proc,
                now_epoch=float(clock()),
            )
            is None
        ):
            preserved_watchdog_failures.append(pid)
    surviving_non_watchdogs = sorted(
        pid
        for pid in final_cgroup_pids
        if pid not in preserved_pids and alive(pid)
    )
    blockers = []
    if signal_failures:
        blockers.append("dispatcher_cgroup_signal_failed")
    if surviving_non_watchdogs:
        blockers.append("dispatcher_cgroup_non_watchdog_survived")
    if preserved_watchdog_failures:
        blockers.append("dispatcher_cgroup_preserved_watchdog_not_live")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "reconciled" if not blockers else "blocked",
        "generated_at": utc_now_iso(),
        "cgroup_process_count": len(pids),
        "preserved_watchdogs": sorted(preserved, key=lambda row: row["pid"]),
        "terminated_non_watchdog_pids": sorted(terminated),
        "killed_non_watchdog_pids": sorted(killed),
        "pid_identity_changed_before_signal": sorted(set(identity_changed)),
        "signal_failures": signal_failures,
        "surviving_non_watchdog_pids": surviving_non_watchdogs,
        "preserved_watchdog_failures": sorted(preserved_watchdog_failures),
        "final_cgroup_pids": sorted(final_cgroup_pids),
        "blockers": blockers,
        "watchdog_process_count": len(preserved),
        "non_watchdog_process_count": len(reap),
        "raw_process_cmdlines_recorded": False,
        "raw_secret_values_recorded": False,
    }
    ensure_dir(receipts)
    receipt_path = receipts / f"dispatcher-cgroup-cleanup-{int(now * 1_000_000)}.json"
    write_json(receipt_path, result)
    return result


def _default_process_alive(
    pid: int, *, killer: Callable[[int, int], None] = os.kill
) -> bool:
    try:
        killer(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--receipt-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = cleanup_dispatcher_cgroup(
            state_root=args.state_root,
            receipt_dir=args.receipt_dir,
        )
    except DispatcherCgroupCleanupError as exc:
        parser.error(str(exc))
    return 0 if result.get("status") == "reconciled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
