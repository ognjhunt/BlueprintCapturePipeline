"""Fail-closed external watchdog and campaign-budget evidence validation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, cast


def external_process_alive(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def bounded_serve_supervisor_valid(
    supervisor: Mapping[str, Any],
    *,
    process_alive: Callable[[int], bool] = external_process_alive,
) -> bool:
    payload = dict(supervisor or {})
    try:
        deadline = float(payload.get("deadline_epoch") or 0)
        pid = int(payload.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    raw_path = Path(str(payload.get("evidence_path") or "")).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file() or raw_path.stat().st_size > 1024 * 1024:
        return False
    try:
        disk_value = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    disk = dict(disk_value) if isinstance(disk_value, Mapping) else {}
    return bool(
        payload.get("schema_version") == "production_gpu_warm_watchdog.v1"
        and payload.get("status") == "armed"
        and payload.get("independent_process") is True
        and deadline > time.time() + 60
        and process_alive(pid)
        and disk.get("schema_version") == payload.get("schema_version")
        and disk.get("status") == payload.get("status")
        and disk.get("pid") == pid
        and float(disk.get("deadline_epoch") or 0) == deadline
        and Path(str(disk.get("evidence_path") or "")).expanduser().resolve() == raw_path.resolve()
    )


def campaign_budget_reservation_valid(supervisor: Mapping[str, Any]) -> bool:
    payload = dict(supervisor or {})
    raw_ledger_path = str(payload.get("campaign_budget_ledger") or "").strip()
    reservation_id = str(payload.get("campaign_reservation_id") or "").strip()
    if not raw_ledger_path or not reservation_id:
        return False
    path = Path(raw_ledger_path).expanduser()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    ledger = dict(value) if isinstance(value, Mapping) else {}
    try:
        spend_cap = float(cast(Any, ledger.get("total_spend_cap_usd")))
        wall_cap = int(cast(Any, ledger.get("combined_gpu_wall_cap_seconds")))
        committed_usd = float(cast(Any, ledger.get("committed_usd")))
        committed_seconds = int(cast(Any, ledger.get("committed_gpu_seconds")))
        deadline = float(cast(Any, payload.get("deadline_epoch")))
        armed_epoch = float(cast(Any, payload.get("armed_at_epoch")))
    except (TypeError, ValueError):
        return False
    reservations = ledger.get("reservations")
    if not isinstance(reservations, list):
        return False
    matching = [dict(row) for row in reservations if isinstance(row, Mapping) and row.get("reservation_id") == reservation_id]
    if len(matching) != 1:
        return False
    reservation = matching[0]
    try:
        reserved_seconds = int(cast(Any, reservation.get("reserved_gpu_seconds")))
        reserved_usd = float(cast(Any, reservation.get("reserved_usd")))
        max_rate = float(cast(Any, reservation.get("max_hourly_rate_usd")))
    except (TypeError, ValueError):
        return False
    return bool(
        ledger.get("schema_version") == "production_gpu_campaign_budget.v1"
        and reservation.get("status") == "open"
        and 0 < spend_cap <= 20.0
        and 0 < wall_cap <= 21_000
        and 0 <= committed_usd <= spend_cap
        and 0 <= committed_seconds <= wall_cap
        and 0 < max_rate <= 3.50
        and reserved_usd <= max_rate * reserved_seconds / 3600.0 + 0.000001
        and reserved_seconds >= deadline - armed_epoch
    )
