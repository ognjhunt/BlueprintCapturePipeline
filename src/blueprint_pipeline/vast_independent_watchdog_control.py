"""Arm and close an independent hard-TTL watchdog around one Vast probe."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .watchdog_owner_teardown_contract import (
    OWNER_TEARDOWN_CANCEL_NAME,
    WATCHDOG_EVIDENCE_NAME,
    write_owner_teardown_cancel_request,
)


HANDOFF_SCHEMA = "vast_independent_watchdog_handoff.v1"
SUPERSESSION_SCHEMA = "vast_independent_watchdog_supersession.v1"
SUPERSESSION_NAME = "vast_independent_watchdog_supersession.json"
HANDOFF_NAME = "vast_independent_watchdog_handoff.json"
WATCHDOG_DIR_NAME = "independent_vast_watchdog"
EVIDENCE_NAME = WATCHDOG_EVIDENCE_NAME
WARM_SERVE_MARKER_NAME = "warm_serve_pod.json"
CALLER_EXIT_SURVIVAL_ENV = "BLUEPRINT_VAST_WATCHDOG_CALLER_EXIT_SURVIVAL"
SYSTEMD_KILL_MODE_PROCESS_SURVIVAL = "systemd_dispatcher_kill_mode_process"


@dataclass(frozen=True)
class VastWatchdogHandle:
    """Private local handle plus the public fields bound into launch evidence."""

    process: subprocess.Popen[str]
    out_dir: Path
    pod_name_prefix: str
    deadline_epoch: float
    started_instance_id_path: Path
    allowed_active_instance_ids: tuple[int, ...]
    caller_exit_survival_contract: str


def _process_alive(process: subprocess.Popen[str]) -> bool:
    return process.poll() is None


def _caller_exit_survival_contract() -> str:
    declared = str(os.environ.get(CALLER_EXIT_SURVIVAL_ENV) or "").strip()
    under_systemd = bool(str(os.environ.get("INVOCATION_ID") or "").strip())
    if under_systemd:
        return (
            SYSTEMD_KILL_MODE_PROCESS_SURVIVAL
            if declared == SYSTEMD_KILL_MODE_PROCESS_SURVIVAL
            else "systemd_cgroup_survival_unproven"
        )
    return "detached_posix_session"


def _stop_terminal_watchdog_process(
    process: subprocess.Popen[str], *, wait_seconds: float = 5.0
) -> int | None:
    """Reap a terminal watchdog so KillMode=process cannot leave an orphan."""

    try:
        return process.wait(timeout=max(0.1, wait_seconds))
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=5)


def _retained_watchdog_result(
    *,
    handle: VastWatchdogHandle,
    reason: str,
    instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    alive = _process_alive(handle.process)
    survival_proven = handle.caller_exit_survival_contract in {
        "detached_posix_session",
        SYSTEMD_KILL_MODE_PROCESS_SURVIVAL,
    }
    retained = alive and survival_proven
    return {
        "schema_version": HANDOFF_SCHEMA,
        "generated_at": utc_now_iso(),
        "status": (
            "retained_until_hard_ttl"
            if retained
            else "watchdog_process_not_live"
            if not alive
            else "watchdog_caller_exit_survival_unproven"
        ),
        "reason": reason,
        "watchdog_armed_before_allocation": True,
        "watchdog_pid": handle.process.pid,
        "watchdog_deadline_epoch": handle.deadline_epoch,
        "pod_name_prefix": handle.pod_name_prefix,
        "watchdog_retention_liveness_confirmed": alive,
        "watchdog_caller_exit_survival_confirmed": survival_proven,
        "caller_exit_survival_contract": handle.caller_exit_survival_contract,
        "watchdog_evidence_path": str(handle.out_dir / EVIDENCE_NAME),
        "watchdog_out_dir": str(handle.out_dir),
        "owner_cancel_path": str(handle.out_dir / OWNER_TEARDOWN_CANCEL_NAME),
        "instance_ids": list(instance_ids),
        "provider_mutations_performed": 0,
        "blockers": (
            []
            if retained
            else ["independent_vast_watchdog_process_not_live"]
            if not alive
            else ["independent_vast_watchdog_caller_exit_survival_unproven"]
        ),
        "raw_secret_values_recorded": False,
    }


def _write_warm_serve_lease(
    *, handle: VastWatchdogHandle, instance_id: int
) -> Path:
    """Publish the fleet-reaper lease owned by this exact live watchdog."""

    marker = {
        "schema_version": "vast_independent_watchdog_warm_lease.v1",
        "status": "serving",
        "provider": "vast",
        # gpu_spend_guard uses the historical provider-neutral ``pod_id`` key.
        "pod_id": str(int(instance_id)),
        "instance_id": int(instance_id),
        "heartbeat_at": utc_now_iso(),
        "lease_expires_at": datetime.fromtimestamp(
            handle.deadline_epoch, timezone.utc
        ).isoformat(),
        "watchdog_pid": handle.process.pid,
        "watchdog_deadline_epoch": handle.deadline_epoch,
        "watchdog_out_dir": str(handle.out_dir),
        "watchdog_pod_name_prefix": handle.pod_name_prefix,
        "continuing_spend": True,
        "raw_secret_values_recorded": False,
    }
    path = handle.out_dir / WARM_SERVE_MARKER_NAME
    write_json(path, marker)
    return path


def _terminal_evidence_matches_handle(
    evidence: Mapping[str, Any],
    *,
    handle: VastWatchdogHandle,
    instance_id: str,
) -> bool:
    recorded = evidence.get("recorded_vast_instance_teardown")
    recorded = recorded if isinstance(recorded, Mapping) else {}
    inspect_attempts = recorded.get("inspect_attempts")
    lane_inventories = (
        evidence.get("initial_inventory"),
        evidence.get("final_inventory"),
    )
    global_inventories = (
        evidence.get("initial_global_inventory"),
        evidence.get("final_global_inventory"),
    )
    allowed_ids = {str(value) for value in handle.allowed_active_instance_ids}
    try:
        observed_deadline = float(evidence.get("deadline_epoch") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        evidence.get("status") == "provider_terminal"
        and evidence.get("provider") == "vast"
        and evidence.get("provider_absence_confirmed") is True
        and evidence.get("pod_name_prefix") == handle.pod_name_prefix
        and evidence.get("pid") == handle.process.pid
        and observed_deadline == handle.deadline_epoch
        and evidence.get("owner_teardown_cancel_requested") is True
        and evidence.get("owner_teardown_cancel_request_valid") is True
        and str(recorded.get("instance_id") or "") == instance_id
        and recorded.get("status") == "absent"
        and recorded.get("provider_absence_confirmed") is True
        and all(
            isinstance(row, Mapping)
            and row.get("status") == "observed"
            and row.get("provider") == "vast"
            and row.get("name_prefix") == handle.pod_name_prefix
            and row.get("api_confirmed") is True
            and row.get("live_resource_count") == 0
            and row.get("resources") == []
            for row in lane_inventories
        )
        and all(
            isinstance(row, Mapping)
            and _global_inventory_contains_only_allowed(row, allowed_ids=allowed_ids)
            for row in global_inventories
        )
        and evidence.get("raw_secret_values_recorded") is False
        and recorded.get("provider_mutations_performed") == 0
        and isinstance(inspect_attempts, list)
        and len(inspect_attempts) >= 2
        and all(
            isinstance(row, Mapping)
            and row.get("status") == "absent"
            and row.get("provider") == "vast"
            and str(row.get("instance_id") or "") == instance_id
            and row.get("http") in {200, 404, 410}
            and row.get("api_confirmed") is True
            and row.get("provider_absence_confirmed") is True
            for row in inspect_attempts
        )
    )


def _global_inventory_contains_only_allowed(
    value: Mapping[str, Any], *, allowed_ids: set[str]
) -> bool:
    resources = value.get("resources")
    count = value.get("live_resource_count")
    if (
        value.get("api_confirmed") is not True
        or value.get("status") != "observed"
        or value.get("provider") != "vast"
        or value.get("name_prefix") != ""
        or not isinstance(count, int)
        or isinstance(count, bool)
        or not isinstance(resources, list)
        or len(resources) != count
    ):
        return False
    observed = {
        str(row.get("instance_id") or "")
        for row in resources
        if isinstance(row, Mapping)
    }
    return (
        len(observed) == count
        and all(observed)
        and observed.issubset(allowed_ids)
    )


def _safe_suffix(value: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z]+", "", value)
    return (suffix or str(int(time.time())))[:24].lower()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _exact_vast_instance_live(value: Mapping[str, Any], instance_id: int) -> bool:
    return bool(
        value.get("api_confirmed") is True
        and value.get("status") == "observed"
        and str(value.get("instance_id") or "") == str(instance_id)
        and str(value.get("actual_status") or "").lower() in {"running", "active"}
    )


def arm_independent_vast_watchdog(
    *,
    job_dir: Path,
    max_live_minutes: int,
    generated_at: str,
    pod_name_prefix: str,
    startup_wait_seconds: float = 10.0,
    allowed_active_instance_ids: Sequence[int] = (),
) -> tuple[dict[str, Any], VastWatchdogHandle | None]:
    """Start a detached name-bound watchdog and prove it is armed before create."""

    out_dir = job_dir / WATCHDOG_DIR_NAME
    ensure_dir(out_dir)
    prefix_base = str(pod_name_prefix or "").strip()
    if not re.fullmatch(r"blueprint-[a-z0-9-]{1,100}-", prefix_base):
        raise ValueError("independent_vast_watchdog_prefix_invalid")
    prefix = f"{prefix_base}{_safe_suffix(generated_at)}-"
    if int(max_live_minutes) < 2:
        blocked = {
            "schema_version": HANDOFF_SCHEMA,
            "generated_at": generated_at,
            "status": "blocked",
            "independent_process": False,
            "watchdog_armed_before_allocation": False,
            "pod_name_prefix": prefix,
            "watchdog_out_dir": str(out_dir),
            "started_instance_id_path": str(out_dir / "started_vast_instance_id.txt"),
            "provider_mutations_performed": 0,
            "blockers": ["independent_vast_watchdog_ttl_too_short"],
            "raw_secret_values_recorded": False,
        }
        write_json(out_dir / EVIDENCE_NAME, blocked)
        write_json(job_dir / HANDOFF_NAME, blocked)
        return blocked, None
    started_epoch = time.time()
    caller_exit_survival = _caller_exit_survival_contract()
    deadline = started_epoch + max(1, int(max_live_minutes)) * 60
    log_path = out_dir / "watchdog.log"
    log_handle = log_path.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.groot_oscar_runpod_watchdog",
        "--out-dir",
        str(out_dir),
        "--pod-name-prefix",
        prefix,
        "--deadline-epoch",
        str(deadline),
        "--provider",
        "vast",
    ]
    allowed_ids = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    for instance_id in allowed_ids:
        command.extend(["--allowed-active-instance-id", str(instance_id)])
    try:
        process = subprocess.Popen(  # noqa: S603  # nosec B603 - fixed module and argv
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        blocked = {
            "schema_version": HANDOFF_SCHEMA,
            "generated_at": generated_at,
            "status": "blocked",
            "independent_process": False,
            "watchdog_armed_before_allocation": False,
            "pod_name_prefix": prefix,
            "watchdog_out_dir": str(out_dir),
            "started_instance_id_path": str(out_dir / "started_vast_instance_id.txt"),
            "provider_mutations_performed": 0,
            "blockers": ["independent_vast_watchdog_process_start_failed"],
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
        write_json(out_dir / EVIDENCE_NAME, blocked)
        write_json(job_dir / HANDOFF_NAME, blocked)
        return blocked, None
    finally:
        log_handle.close()
    evidence_path = out_dir / EVIDENCE_NAME
    deadline_monotonic = time.monotonic() + max(0.1, startup_wait_seconds)
    armed: dict[str, Any] = {}
    while time.monotonic() < deadline_monotonic:
        armed = _read_json(evidence_path)
        if (
            armed.get("status") == "armed"
            and armed.get("independent_process") is True
            and armed.get("pid") == process.pid
            and armed.get("pod_name_prefix") == prefix
            and armed.get("provider") == "vast"
            and process.poll() is None
        ):
            break
        if process.poll() is not None:
            break
        time.sleep(0.1)
    passed = bool(
        armed.get("status") == "armed"
        and armed.get("independent_process") is True
        and armed.get("pid") == process.pid
        and armed.get("pod_name_prefix") == prefix
        and armed.get("provider") == "vast"
        and process.poll() is None
    )
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "generated_at": generated_at,
        "status": "armed" if passed else "blocked",
        "independent_process": passed,
        "watchdog_pid": process.pid,
        "watchdog_started_epoch": started_epoch,
        "watchdog_deadline_epoch": deadline,
        "watchdog_armed_before_allocation": passed,
        "caller_exit_survival_contract": caller_exit_survival,
        "systemd_dispatcher_kill_mode_required": (
            "process" if caller_exit_survival == SYSTEMD_KILL_MODE_PROCESS_SURVIVAL else None
        ),
        "watchdog_evidence_path": str(evidence_path),
        "pod_name_prefix": prefix,
        "watchdog_out_dir": str(out_dir),
        "started_instance_id_path": str(out_dir / "started_vast_instance_id.txt"),
        "allowed_active_instance_ids": list(allowed_ids),
        "provider_mutations_performed": 0,
        "blockers": [] if passed else ["independent_vast_watchdog_not_armed"],
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / HANDOFF_NAME, handoff)
    if not passed:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        write_json(evidence_path, handoff)
        return handoff, None
    return handoff, VastWatchdogHandle(
        process=process,
        out_dir=out_dir,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        started_instance_id_path=out_dir / "started_vast_instance_id.txt",
        allowed_active_instance_ids=allowed_ids,
        caller_exit_survival_contract=caller_exit_survival,
    )


def supersede_independent_vast_watchdog(
    *,
    job_dir: Path,
    predecessor_handoff: Mapping[str, Any],
    instance_id: int,
    max_live_minutes: int,
    generated_at: str,
    pod_name_prefix: str,
    provider_factory: Any | None = None,
    predecessor_exit_wait_seconds: float = 10.0,
) -> tuple[dict[str, Any], VastWatchdogHandle | None]:
    """Transfer one live Vast instance to a later independent watchdog.

    The predecessor remains untouched unless the successor is independently
    armed, bound to the exact same instance, has a later deadline, and two
    provider inspections prove that instance is still running.  The final
    provider inspection happens after predecessor retirement so the receipt
    proves the transfer itself did not terminate the worker.
    """

    predecessor_pid = int(predecessor_handoff.get("watchdog_pid") or 0)
    predecessor_deadline = float(
        predecessor_handoff.get("watchdog_deadline_epoch") or 0.0
    )
    predecessor_out_dir = Path(
        str(predecessor_handoff.get("watchdog_out_dir") or "")
    ).expanduser()
    predecessor_instance_path = predecessor_out_dir / "started_vast_instance_id.txt"
    try:
        predecessor_instance_id = int(
            predecessor_instance_path.read_text(encoding="utf-8").strip()
        )
    except (OSError, UnicodeError, ValueError):
        predecessor_instance_id = 0
    blockers: list[str] = []
    if predecessor_handoff.get("status") not in {
        "armed",
        "retained_until_hard_ttl",
    }:
        blockers.append("predecessor_watchdog_status_invalid")
    if not _pid_alive(predecessor_pid):
        blockers.append("predecessor_watchdog_not_live")
    if predecessor_deadline <= time.time() + 60:
        blockers.append("predecessor_watchdog_deadline_too_close")
    if predecessor_instance_id != int(instance_id):
        blockers.append("predecessor_watchdog_instance_binding_invalid")
    if blockers:
        result = {
            "schema_version": SUPERSESSION_SCHEMA,
            "generated_at": generated_at,
            "status": "blocked",
            "instance_id": int(instance_id),
            "predecessor_watchdog_pid": predecessor_pid or None,
            "provider_mutations_performed": 0,
            "blockers": blockers,
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / SUPERSESSION_NAME, result)
        return result, None

    if provider_factory is None:
        from .gpu_render_providers import get_render_provider

        provider_factory = get_render_provider
    try:
        provider = provider_factory("vast")
        before = provider.inspect(str(instance_id))
    except Exception as exc:  # noqa: BLE001 - predecessor stays armed
        result = {
            "schema_version": SUPERSESSION_SCHEMA,
            "generated_at": generated_at,
            "status": "blocked",
            "instance_id": int(instance_id),
            "predecessor_watchdog_pid": predecessor_pid,
            "provider_mutations_performed": 0,
            "blockers": ["successor_watchdog_instance_live_unproven"],
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / SUPERSESSION_NAME, result)
        return result, None
    if not _exact_vast_instance_live(before, instance_id):
        result = {
            "schema_version": SUPERSESSION_SCHEMA,
            "generated_at": generated_at,
            "status": "blocked",
            "instance_id": int(instance_id),
            "predecessor_watchdog_pid": predecessor_pid,
            "provider_inspect_before": before,
            "provider_mutations_performed": 0,
            "blockers": ["successor_watchdog_instance_not_running"],
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / SUPERSESSION_NAME, result)
        return result, None

    successor_handoff, successor = arm_independent_vast_watchdog(
        job_dir=job_dir,
        max_live_minutes=max_live_minutes,
        generated_at=generated_at,
        pod_name_prefix=pod_name_prefix,
        allowed_active_instance_ids=[instance_id],
    )
    if successor is None:
        result = {
            "schema_version": SUPERSESSION_SCHEMA,
            "generated_at": generated_at,
            "status": "blocked",
            "instance_id": int(instance_id),
            "predecessor_watchdog_pid": predecessor_pid,
            "successor_watchdog": successor_handoff,
            "provider_mutations_performed": 0,
            "blockers": ["successor_watchdog_not_armed"],
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / SUPERSESSION_NAME, result)
        return result, None
    write_started_vast_instance_id(successor.started_instance_id_path, instance_id)

    try:
        successor_bound = (
            successor.deadline_epoch > predecessor_deadline
            and successor.process.poll() is None
            and successor.started_instance_id_path.read_text(encoding="utf-8").strip()
            == str(instance_id)
        )
        second = provider.inspect(str(instance_id))
    except Exception:  # noqa: BLE001 - predecessor remains armed
        successor_bound = False
        second = {}
    if not successor_bound or not _exact_vast_instance_live(second, instance_id):
        successor.process.terminate()
        try:
            successor.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            successor.process.kill()
        result = {
            "schema_version": SUPERSESSION_SCHEMA,
            "generated_at": generated_at,
            "status": "blocked",
            "instance_id": int(instance_id),
            "predecessor_watchdog_pid": predecessor_pid,
            "successor_watchdog": successor_handoff,
            "provider_inspect_before": before,
            "provider_inspect_successor_armed": second,
            "provider_mutations_performed": 0,
            "blockers": ["successor_watchdog_exact_binding_unproven"],
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / SUPERSESSION_NAME, result)
        return result, None

    os.kill(predecessor_pid, signal.SIGTERM)
    exit_deadline = time.monotonic() + max(0.1, predecessor_exit_wait_seconds)
    while time.monotonic() < exit_deadline and _pid_alive(predecessor_pid):
        time.sleep(0.1)
    predecessor_retired = not _pid_alive(predecessor_pid)
    try:
        after = provider.inspect(str(instance_id))
    except Exception:  # noqa: BLE001 - successor stays armed and owns teardown
        after = {}
    transferred = bool(
        predecessor_retired
        and successor.process.poll() is None
        and _exact_vast_instance_live(after, instance_id)
    )
    warm_serve_lease_path = None
    if transferred:
        warm_serve_lease_path = str(
            _write_warm_serve_lease(handle=successor, instance_id=instance_id)
        )
    result = {
        "schema_version": SUPERSESSION_SCHEMA,
        "generated_at": generated_at,
        "status": "superseded" if transferred else "successor_retained_transfer_unproven",
        "instance_id": int(instance_id),
        "predecessor_watchdog_pid": predecessor_pid,
        "predecessor_watchdog_deadline_epoch": predecessor_deadline,
        "predecessor_watchdog_retired": predecessor_retired,
        "successor_watchdog_pid": successor.process.pid,
        "successor_watchdog_deadline_epoch": successor.deadline_epoch,
        "successor_watchdog_out_dir": str(successor.out_dir),
        "provider_inspect_before": before,
        "provider_inspect_successor_armed": second,
        "provider_inspect_after_transfer": after,
        "provider_instance_running_after_transfer": _exact_vast_instance_live(
            after, instance_id
        ),
        "warm_serve_lease_path": warm_serve_lease_path,
        "provider_mutations_performed": 0,
        "blockers": [] if transferred else ["watchdog_supersession_transfer_unproven"],
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / SUPERSESSION_NAME, result)
    return result, successor


def close_independent_vast_watchdog(
    *,
    job_dir: Path,
    handle: VastWatchdogHandle,
    instance_ids: list[int],
    provider_teardown_completed: bool,
    provider_allocation_impossible: bool = False,
    wait_seconds: float = 300.0,
) -> dict[str, Any]:
    """Ask the watchdog to close only after owner teardown, or leave it armed."""

    if not instance_ids:
        if not provider_allocation_impossible:
            result = _retained_watchdog_result(
                handle=handle,
                reason="provider_allocation_identity_ambiguous",
            )
            write_json(job_dir / HANDOFF_NAME, result)
            return result
        handle.process.terminate()
        try:
            handle.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.process.kill()
        result = {
            "schema_version": HANDOFF_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "cancelled_no_allocation",
            "watchdog_armed_before_allocation": True,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(handle.out_dir / EVIDENCE_NAME, result)
        write_json(job_dir / HANDOFF_NAME, result)
        return result
    if not provider_teardown_completed:
        result = _retained_watchdog_result(
            handle=handle,
            reason="provider_teardown_not_completed",
            instance_ids=instance_ids,
        )
        if result.get("status") == "retained_until_hard_ttl" and instance_ids:
            result["warm_serve_lease_path"] = str(
                _write_warm_serve_lease(
                    handle=handle,
                    instance_id=int(instance_ids[-1]),
                )
            )
        write_json(job_dir / HANDOFF_NAME, result)
        return result
    instance_id = str(instance_ids[-1])
    write_owner_teardown_cancel_request(
        root=handle.out_dir,
        pod_name_prefix=handle.pod_name_prefix,
        provider_name="vast",
        instance_id=instance_id,
    )
    # The independent watchdog double-inspects the exact instance and the
    # global inventory before writing terminal evidence, which can take
    # minutes. Poll the evidence itself and return as soon as absence is
    # confirmed; waiting only for process exit misread v7/v9's correct
    # closures as unclosed.
    wait_deadline = time.monotonic() + max(0.1, wait_seconds)
    terminal = _read_json(handle.out_dir / EVIDENCE_NAME)
    while time.monotonic() < wait_deadline:
        terminal = _read_json(handle.out_dir / EVIDENCE_NAME)
        if _terminal_evidence_matches_handle(
            terminal, handle=handle, instance_id=instance_id
        ):
            break
        if handle.process.poll() is not None:
            terminal = _read_json(handle.out_dir / EVIDENCE_NAME)
            break
        time.sleep(0.2)
    terminal_valid = _terminal_evidence_matches_handle(
        terminal, handle=handle, instance_id=instance_id
    )
    status = "provider_terminal" if terminal_valid else "retained_until_hard_ttl"
    if not terminal_valid:
        result = _retained_watchdog_result(
            handle=handle,
            reason="terminal_evidence_not_exactly_bound",
            instance_ids=instance_ids,
        )
        result["provider_absence_confirmed"] = False
        write_json(job_dir / HANDOFF_NAME, result)
        return result
    process_exit_code = handle.process.poll()
    if status == "provider_terminal":
        process_exit_code = _stop_terminal_watchdog_process(handle.process)
    result = {
        "schema_version": HANDOFF_SCHEMA,
        "generated_at": utc_now_iso(),
        "status": status,
        "watchdog_armed_before_allocation": True,
        "instance_ids": instance_ids,
        "provider_absence_confirmed": terminal.get("provider_absence_confirmed") is True,
        "watchdog_process_exit_code": process_exit_code,
        "watchdog_retention_liveness_confirmed": False,
        "provider_mutations_performed": terminal.get("provider_mutations_performed", 0),
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / HANDOFF_NAME, result)
    return result


def close_independent_vast_watchdog_without_allocation(
    *,
    job_dir: Path,
    handle: VastWatchdogHandle,
    wait_seconds: float = 300.0,
) -> dict[str, Any]:
    """Close a pre-armed watchdog after proving no instance was allocated.

    This is intentionally stronger than terminating the process locally.  The
    independent watchdog performs the same double lane-prefix/global API
    inventory used after an owned teardown and retains those facts in its
    normal evidence file.  It is only valid when the provider adapter never
    attempted create and no started-instance id was published.
    """

    if handle.started_instance_id_path.exists():
        return {
            "schema_version": HANDOFF_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "retained_until_hard_ttl",
            "reason": "provider_allocation_identity_present",
            "watchdog_armed_before_allocation": True,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
    handle.process.terminate()
    try:
        handle.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        handle.process.kill()
    try:
        from .gpu_render_providers import get_render_provider

        provider = get_render_provider("vast")
        first_lane = provider.billable_inventory(name_prefix=handle.pod_name_prefix)
        first_global = provider.billable_inventory(name_prefix="")
        second_lane = provider.billable_inventory(name_prefix=handle.pod_name_prefix)
        second_global = provider.billable_inventory(name_prefix="")
    except Exception as exc:  # noqa: BLE001 - retain typed API uncertainty
        result = {
            "schema_version": HANDOFF_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "provider_zero_unverified_no_allocation",
            "watchdog_armed_before_allocation": True,
            "provider_absence_confirmed": False,
            "error_type": type(exc).__name__,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
    else:
        zero = all(
            row.get("api_confirmed") is True and row.get("live_resource_count") == 0
            for row in (first_lane, second_lane, first_global, second_global)
        )
        result = {
            "schema_version": HANDOFF_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "provider_terminal" if zero else "provider_zero_unverified_no_allocation",
            "watchdog_armed_before_allocation": True,
            "provider_absence_confirmed": zero,
            "initial_inventory": first_lane,
            "initial_global_inventory": first_global,
            "final_inventory": second_lane,
            "final_global_inventory": second_global,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
    write_json(handle.out_dir / EVIDENCE_NAME, result)
    write_json(job_dir / HANDOFF_NAME, result)
    return result


def write_started_vast_instance_id(path: str | Path, instance_id: int) -> None:
    """Atomically publish the exact created id to the already-armed watchdog."""

    resolved = Path(path).expanduser().resolve()
    ensure_dir(resolved.parent)
    temporary = resolved.with_suffix(".tmp")
    temporary.write_text(f"{int(instance_id)}\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, resolved)
