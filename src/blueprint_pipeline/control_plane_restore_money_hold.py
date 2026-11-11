"""Paid admission after a restore from backup: closed until the money records have caught up, then open by itself.

A backup is up to 15 minutes old. A ledger restored from it can forget a hold, an instance launched or a spend
made after the backup: a money error if new paid work were admitted against it. So a restore onto the
production root writes ``restore_hold.v1.json`` beside the spend admission lock, and the spend guard
(``scripts/gpu_spend_guard.py``) adds ``BLOCKER`` to the lock while it is there. The hold names what has to
happen after the restore, and nothing else:

- the spend guard ran with ``--reap`` after the restore, its provider inventory was complete, and every orphan it
  found was terminated (so every live instance the restored ledger forgot is now seen and counted or gone); and
- the provider billing reconciliation the guard reads is ``reconciled`` with ``generated_at`` after the restore
  (so provider-billed spend in the gap is in the totals the hard stop is judged against).

When both hold in one guard run, that run retires the marker (renamed beside it with the run's evidence) and the
lock it writes is open. No operator step clears it, and none is needed. Holds already in the restored ledger
stay: nothing here removes one, and nothing paid is retried because of a restore.

A marker that cannot be read still blocks, from its file time, and still clears by the same two conditions.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .core.common import write_json

SCHEMA = "control_plane_restore_money_hold.v1"
MARKER_NAME = "restore_hold.v1.json"
DEFAULT_MARKER = Path("/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard") / MARKER_NAME
BLOCKER = "restored_from_backup_reconciliation_pending"


def _epoch(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp() if parsed.tzinfo is not None else None


def write_hold(path: Path, *, generation: str, generation_started_at_epoch: float, restored_at_epoch: float) -> None:
    write_json(Path(path), {
        "schema_version": SCHEMA, "generation": generation,
        "generation_started_at_epoch": float(generation_started_at_epoch),
        "restored_at_epoch": float(restored_at_epoch),
        "backup_gap_seconds": max(0.0, float(restored_at_epoch) - float(generation_started_at_epoch)),
        "clears_when": ["spend_guard_reap_complete_after_restore", "billing_reconciled_after_restore"],
    })


def read_hold(path: Path) -> dict[str, Any] | None:
    """The hold at ``path``, or None when there is none. An unreadable marker is a hold from its file time."""

    path = Path(path)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    try:
        value = json.loads(path.read_bytes())
        restored = float(value["restored_at_epoch"]) if isinstance(value, dict) else None
    except (OSError, ValueError, KeyError, TypeError):
        restored = None
        value = {}
    if restored is None:
        return {"schema_version": SCHEMA, "restored_at_epoch": info.st_mtime, "marker_unreadable": True}
    return dict(value)


def evaluate(hold: Mapping[str, Any] | None, *, billing: Mapping[str, Any], reap_mode: bool, inventory_complete: bool,
             all_reap_candidates_terminated: bool, now: float) -> dict[str, Any] | None:
    """What one spend guard run makes of the hold: ``blocking`` until both conditions hold, then ``cleared``."""

    if hold is None:
        return None
    restored = float(hold["restored_at_epoch"])
    generated = _epoch(billing.get("generated_at"))
    reap_done = bool(reap_mode and inventory_complete and all_reap_candidates_terminated and now > restored)
    billing_done = bool(billing.get("status") == "reconciled" and generated is not None and generated > restored)
    waiting = [name for name, done in (("spend_guard_reap_complete_after_restore", reap_done),
                                       ("billing_reconciled_after_restore", billing_done)) if not done]
    return {"status": "cleared" if not waiting else "blocking", "blocking": bool(waiting), "waiting_on": waiting,
            "restored_at_epoch": restored, "generation": hold.get("generation"),
            "backup_gap_seconds": hold.get("backup_gap_seconds")}


def retire(path: Path, evaluation: Mapping[str, Any], *, now: float | None = None) -> Path | None:
    """Move a cleared hold aside with the run's evidence; admission is open from this run's lock on."""

    if evaluation.get("status") != "cleared":
        return None
    path = Path(path)
    observed = time.time() if now is None else float(now)
    retired = path.with_name(f"restore_hold.retired-{int(observed)}.v1.json")
    held = read_hold(path) or {}
    write_json(retired, {**held, "retired_at_epoch": observed, "retired_by": "gpu_spend_guard",
                         "evaluation": dict(evaluation)})
    path.unlink(missing_ok=True)
    return retired


__all__ = ["BLOCKER", "DEFAULT_MARKER", "MARKER_NAME", "SCHEMA", "evaluate", "read_hold", "retire", "write_hold"]
