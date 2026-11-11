"""The final Vast billed charge of a terminal scene-configuration run, or ``None``.

A settled scene-configuration row keeps its whole GPU allowance because a
runtime estimate is not a bill: the cost model measured the adapter's
estimate at 56% of billed on capture-447ae4f4 (B-03 section 9). Vast's own
per-instance charges reach the host's spend snapshot
(``daily_spend_snapshot``), but a day's charges can still post after it
ends. A charge is final for settlement (owner decision 2026-11-10) when

- the run's teardown proves every instance it created destroyed;
- the snapshot was generated at least ``FINAL_AFTER_DESTROY_SECONDS`` after
  the teardown was recorded (a destroy happens before its manifest);
- the snapshot's charge rows for each instance cover every UTC day from the
  run's start (``launch_started.json``, written before the allocator could
  create anything) to the teardown, each with a USD amount and an interval.

Anything short of that is ``None`` and the hold stays as it is. Unknown
stays unknown: a missing day is never read as $0.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

#: The cost model says to re-read Vast charges after 24 h because rows for a
#: day can still post (B-03 section 8, gap 5); doubled for settlement.
FINAL_AFTER_DESTROY_SECONDS = 48 * 3600
SNAPSHOT_FILENAME = "daily_spend_snapshot.json"
#: Where the GPU spend guard and the billing reconciler keep the export the snapshot sits beside.
HOST_BILLING_EXPORT = Path("/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/provider_billing_export.json")
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024


def _time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def snapshot_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """The spend snapshot beside the configured billing export; on the host, the installed one."""
    from .account_monthly_spend import BILLING_EXPORT_ENV, limit_required
    env = os.environ if environ is None else environ
    configured = str(env.get(BILLING_EXPORT_ENV) or "").strip()
    if configured:
        return Path(configured).parent / SNAPSHOT_FILENAME
    return HOST_BILLING_EXPORT.parent / SNAPSHOT_FILENAME if limit_required(env) else None


# Holds are derived on every read, row by row: parse one snapshot version once per process.
_CACHE: dict[str, tuple[tuple[int, int, int], dict[str, Any] | None]] = {}


def _read_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        info = path.stat()
        if info.st_size > MAX_SNAPSHOT_BYTES:
            return None
        version = (info.st_ino, info.st_mtime_ns, info.st_size)
        cached = _CACHE.get(str(path))
        if cached is not None and cached[0] == version:
            return cached[1]
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    result = value if isinstance(value, dict) else None
    _CACHE.clear()
    _CACHE[str(path)] = (version, result)
    return result


def _days(start: datetime, end: datetime) -> set[str]:
    day, last, days = start.astimezone(timezone.utc).date(), end.astimezone(timezone.utc).date(), set()
    while day <= last:
        days.add(day.isoformat())
        day += timedelta(days=1)
    return days


def _instance_charge(rows: list[Any], resource: str, life: tuple[datetime, datetime]) -> Decimal | None:
    total, covered, found = Decimal(0), set(), False
    for row in rows:
        if not (isinstance(row, Mapping) and row.get("provider") == "vast" and row.get("service") == "charges"
                and row.get("kind") == "actual" and row.get("resource") == resource):
            continue
        start, end, amount = _time(row.get("interval_start")), _time(row.get("interval_end")), row.get("amount")
        if (start is None or end is None or end <= start or row.get("currency") != "USD"
                or isinstance(amount, bool) or not isinstance(amount, (int, float))):
            return None  # one unreadable row for this instance: its total is unknown
        try:
            total += Decimal(str(amount))
        except InvalidOperation:
            return None
        covered |= _days(start, end - timedelta(microseconds=1))
        found = True
    if not found or not _days(*life) <= covered or total < 0:
        return None
    return total


def final_billed_charge(*, launch_receipt_path: Path, launch: Mapping[str, Any], teardown: Mapping[str, Any],
                        environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """``{"billed_usd", ...evidence}`` once every instance's charge is final, else ``None``."""
    from .decision_evidence_contracts import canonical_digest
    from .daily_spend_snapshot import _identity
    ids = teardown.get("vast_instance_ids")
    if (teardown.get("schema_version") != "vast_teardown_manifest.v1" or teardown.get("status") != "completed"
            or teardown.get("continuing_spend_from_this_run") is not False
            or teardown.get("runner_gpu_teardown_completed") is not True
            or not isinstance(ids, list) or not ids
            or any(isinstance(i, bool) or not isinstance(i, int) for i in ids)):
        return None
    destroyed = _time(teardown.get("generated_at"))
    started_path = Path(launch_receipt_path).parent / "launch_started.json"
    try:
        started = json.loads(started_path.read_bytes())
    except (OSError, ValueError):
        return None
    if (not isinstance(started, dict) or started.get("schema_version") != "task_evaluation_launch_started.v1"
            or started.get("launch_id") != launch.get("launch_id")
            or started.get("started_digest") != canonical_digest(started, digest_field="started_digest")):
        return None
    begun = _time(started.get("started_at"))
    path = snapshot_path(environ)
    snapshot = None if path is None else _read_snapshot(path)
    if destroyed is None or begun is None or begun > destroyed or snapshot is None:
        return None
    generated = _time(snapshot.get("generated_at"))
    rows = snapshot.get("rows")
    if (generated is None or not isinstance(rows, list)
            or (generated - destroyed).total_seconds() < FINAL_AFTER_DESTROY_SECONDS):
        return None
    total = Decimal(0)
    for instance in ids:
        charge = _instance_charge(rows, _identity(f"instance-{instance}"), (begun, destroyed))
        if charge is None:
            return None
        total += charge
    return {"billed_usd": total, "vast_instance_ids": list(ids), "snapshot_generated_at": snapshot["generated_at"],
            "snapshot_digest": snapshot.get("snapshot_digest"), "run_started_at": started["started_at"],
            "teardown_generated_at": teardown["generated_at"],
            "final_after_destroy_seconds": FINAL_AFTER_DESTROY_SECONDS}
