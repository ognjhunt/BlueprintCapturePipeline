"""Page the owner once when a website preparation stops on a mismatch in Blueprint's own records.

Three preparation stops are ours to fix, never the site's (2026-11-11):

- ``website_registration_anchor_frame_missing`` and ``website_registration_anchor_frame_invalid``: the
  reconstruction's anchor does not bind to the source views Blueprint itself recorded;
- ``website_registration_scale_conflicts_declared``: the object's estimated size and the scene's
  declared scale, both Blueprint's estimates, disagree by more than a third.

Such a preparation fails, gets the shared retry policy's six timed retries (about 46 hours) and is then
left until the job's window ends it. The site is told only that the job needs our team's review, and
before this nothing told the owner: no stall check reads the preparation ledger. This check pages once
per capture and code, on the stall alarm's route. It gates nothing and writes nothing but the alarm's
memory: it reads each ledger's ``status`` and ``last_error``.

Run inside the stall alarm's tick (``control_plane_stall_alarm``). A persisting page repeats hourly
there, so the row is emitted only for ``ANNOUNCE_SECONDS`` after the tick that first saw the stop, as
``scene_budget_alert`` does.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CODE = "website_preparation_blueprint_fault"
ANNOUNCE_SECONDS = 30 * 60
LEDGER_GLOB = "*/scenes/site-*/captures/*/pipeline_job_ledger.json"
MAX_LEDGERS = 2048
MAX_LEDGER_BYTES = 1024 * 1024
FAULT_CODES = ("website_registration_anchor_frame_invalid", "website_registration_anchor_frame_missing",
               "website_registration_scale_conflicts_declared")
_FAULT = re.compile(r"\b(" + "|".join(FAULT_CODES) + r")\b")
_STOPPED = frozenset({"failed_retryable", "retryable_blocked", "failed_terminal"})


def _fault(ledger: Mapping[str, Any]) -> str | None:
    if ledger.get("status") not in _STOPPED:
        return None
    texts = [ledger.get("last_error"), *(ledger.get("retry_blockers") or [])]
    for text in texts:
        found = _FAULT.search(text) if isinstance(text, str) else None
        if found:
            return found.group(1)
    return None


def check_preparation_faults(root: Path | None, *, now: float, remembered: Mapping[str, Any] | None = None,
                             memory_out: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One page row naming each capture newly stopped on a Blueprint-side code; unreadable ledgers are counted."""
    remembered = remembered if isinstance(remembered, Mapping) else {}
    if root is None:
        return {"status": "not_configured"}, []
    if not Path(root).is_dir():
        return {"status": "absent"}, []
    paths = sorted(Path(root).glob(LEDGER_GLOB))
    truncated = len(paths) > MAX_LEDGERS
    offenders, unreadable, stopped = [], 0, 0
    for path in paths[:MAX_LEDGERS]:
        try:
            if path.is_symlink() or path.stat().st_size > MAX_LEDGER_BYTES:
                raise ValueError("ledger_unreadable")
            ledger = json.loads(path.read_text(encoding="utf-8"))
            code = _fault(ledger) if isinstance(ledger, Mapping) else None
        except (OSError, ValueError):
            unreadable += 1
            continue
        if code is None:
            continue
        stopped += 1
        key = f"{path.parent.relative_to(root)}:{code}"
        prior = remembered.get(key) if isinstance(remembered.get(key), Mapping) else {}
        since = prior.get("since")
        since = since if isinstance(since, (int, float)) and not isinstance(since, bool) else now
        if memory_out is not None:
            memory_out[key] = {"since": since}
        if now - since < ANNOUNCE_SECONDS:
            offenders.append({"target": key, "label": path.parent.name[:120], "status": ledger.get("status"),
                              "blockers": [code], "age_seconds": round(now - since)})
    section = {"status": "observed", "ledgers": min(len(paths), MAX_LEDGERS), "truncated": truncated,
               "stopped": stopped, "announced": len(offenders), "unreadable": unreadable}
    alerts: list[dict[str, Any]] = []
    if offenders:
        offenders.sort(key=lambda row: row["target"])
        alerts.append({"code": CODE, "count": len(offenders),
                       "targets": [row["target"] for row in offenders[:10]], **offenders[0]})
    if truncated:
        # More ledgers than one tick reads: what was not read is unknown, never "nothing found".
        alerts.append({"code": "stall_check_incomplete", "reason": "preparation_ledgers_truncated",
                       "count": len(paths)})
    return section, alerts


def describe(alert: Mapping[str, Any]) -> str:
    more = int(alert.get("count") or 1) - 1
    tail = f" (+{more} more)" if more > 0 else ""
    blockers = alert.get("blockers") or ["?"]
    return (f"website preparation stopped on Blueprint's own records [{blockers[0]}], ours to fix; the site "
            f"is told only that it needs our review: {alert.get('label') or alert.get('target')}{tail}")
