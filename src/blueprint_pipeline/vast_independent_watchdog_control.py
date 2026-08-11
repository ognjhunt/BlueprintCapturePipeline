"""Arm and close an independent hard-TTL watchdog around one Vast probe."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .watchdog_owner_teardown_contract import (
    WATCHDOG_EVIDENCE_NAME,
    write_owner_teardown_cancel_request,
)


HANDOFF_SCHEMA = "vast_independent_watchdog_handoff.v1"
HANDOFF_NAME = "vast_independent_watchdog_handoff.json"
WATCHDOG_DIR_NAME = "independent_vast_watchdog"
EVIDENCE_NAME = WATCHDOG_EVIDENCE_NAME


@dataclass(frozen=True)
class VastWatchdogHandle:
    """Private local handle plus the public fields bound into launch evidence."""

    process: subprocess.Popen[str]
    out_dir: Path
    pod_name_prefix: str
    deadline_epoch: float
    started_instance_id_path: Path


def _safe_suffix(value: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z]+", "", value)
    return (suffix or str(int(time.time())))[:24].lower()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def arm_independent_vast_watchdog(
    *,
    job_dir: Path,
    max_live_minutes: int,
    generated_at: str,
    pod_name_prefix_base: str = "blueprint-groot-oscar-canary-vast-wam-",
    startup_wait_seconds: float = 10.0,
) -> tuple[dict[str, Any], VastWatchdogHandle | None]:
    """Start a detached name-bound watchdog and prove it is armed before create."""

    out_dir = job_dir / WATCHDOG_DIR_NAME
    ensure_dir(out_dir)
    prefix = f"{pod_name_prefix_base}{_safe_suffix(generated_at)}-"
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
    deadline = time.time() + max(1, int(max_live_minutes)) * 60
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
        "watchdog_deadline_epoch": deadline,
        "watchdog_armed_before_allocation": passed,
        "pod_name_prefix": prefix,
        "watchdog_out_dir": str(out_dir),
        "started_instance_id_path": str(out_dir / "started_vast_instance_id.txt"),
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
    )


def close_independent_vast_watchdog(
    *,
    job_dir: Path,
    handle: VastWatchdogHandle,
    instance_ids: list[int],
    provider_teardown_completed: bool,
    provider_allocation_impossible: bool = False,
    wait_seconds: float = 45.0,
) -> dict[str, Any]:
    """Ask the watchdog to close only after owner teardown, or leave it armed."""

    if not instance_ids:
        if not provider_allocation_impossible:
            result = {
                "schema_version": HANDOFF_SCHEMA,
                "generated_at": utc_now_iso(),
                "status": "retained_until_hard_ttl",
                "reason": "provider_allocation_identity_ambiguous",
                "watchdog_armed_before_allocation": True,
                "provider_mutations_performed": 0,
                "raw_secret_values_recorded": False,
            }
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
        result = {
            "schema_version": HANDOFF_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "retained_until_hard_ttl",
            "watchdog_armed_before_allocation": True,
            "instance_ids": instance_ids,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / HANDOFF_NAME, result)
        return result
    instance_id = str(instance_ids[-1])
    write_owner_teardown_cancel_request(
        root=handle.out_dir,
        pod_name_prefix=handle.pod_name_prefix,
        provider_name="vast",
        instance_id=instance_id,
    )
    wait_deadline = time.monotonic() + max(0.1, wait_seconds)
    while handle.process.poll() is None and time.monotonic() < wait_deadline:
        time.sleep(0.2)
    terminal = _read_json(handle.out_dir / EVIDENCE_NAME)
    status = (
        "provider_terminal"
        if terminal.get("status") == "provider_terminal"
        and terminal.get("provider_absence_confirmed") is True
        else "retained_until_hard_ttl"
    )
    result = {
        "schema_version": HANDOFF_SCHEMA,
        "generated_at": utc_now_iso(),
        "status": status,
        "watchdog_armed_before_allocation": True,
        "instance_ids": instance_ids,
        "provider_absence_confirmed": terminal.get("provider_absence_confirmed") is True,
        "watchdog_process_exit_code": handle.process.poll(),
        "provider_mutations_performed": terminal.get("provider_mutations_performed", 0),
        "raw_secret_values_recorded": False,
    }
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
