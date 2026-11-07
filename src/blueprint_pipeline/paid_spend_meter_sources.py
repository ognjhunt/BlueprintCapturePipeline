"""Read this host's meter sources into one per-job view (``paid_spend_meter``).

The chokepoint imports ``paid_spend_meter`` on every paid admission, so the
readers of existing ledgers live here: the beta screening ledger and the
journal, folded with the Atlas attempts the capacity tick observed. Reads
only.

    python -m blueprint_pipeline.paid_spend_meter_sources [--job <request id>]
"""

from __future__ import annotations

import json
import os
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .paid_spend_meter import (
    DEFAULT_SCREENING_QUEUE_ROOT,
    MAX_LEDGER_BYTES,
    SCREENING_QUEUE_ROOT_ENV,
    SOURCE_SCREENING,
    _epoch,
    atlas_rows,
    job_key,
    journal_rows,
    meter_root,
    meter_view,
    money,
    public_view,
    read_journal,
)


def _read_screening_ledger(queue_root: Path) -> dict[str, Any] | None:
    from .beta_policy_screening_coordinator import LEDGER_FILENAME, LEDGER_SCHEMA
    from .decision_evidence_contracts import canonical_digest

    path = Path(queue_root) / LEDGER_FILENAME
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_LEDGER_BYTES:
        raise ValueError("paid_spend_meter_screening_ledger_unreadable")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(ledger, dict) or ledger.get("schema_version") != LEDGER_SCHEMA
            or not isinstance(ledger.get("campaigns"), dict)
            or ledger.get("ledger_digest") != canonical_digest(ledger, digest_field="ledger_digest")):
        raise ValueError("paid_spend_meter_screening_ledger_invalid")
    return ledger


def screening_rows(queue_root: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Each screening attempt, read from the screening ledger without its write lock.

    The ledger is replaced atomically, so a read sees one whole revision.
    Counted as the ledger's own site budget counts it (``_budget``).
    """

    from .beta_policy_screening_settlement import settled_cost

    if queue_root is None:
        return [], []
    try:
        ledger = _read_screening_ledger(Path(queue_root))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        reason = str(exc) if str(exc).startswith("paid_spend_meter_") else "paid_spend_meter_screening_ledger_unreadable"
        return [], [reason]
    if ledger is None:
        return [], []
    rows, gaps = [], set()
    for campaign_id, campaign in sorted(ledger["campaigns"].items()):
        if not isinstance(campaign, Mapping) or not isinstance(campaign.get("candidates"), Mapping):
            gaps.add("paid_spend_meter_screening_campaign_invalid")
            continue
        for candidate_id, candidate in sorted(campaign["candidates"].items()):
            attempts = [*(candidate.get("attempt_history") or []), *([candidate["attempt"]] if candidate.get("attempt") else [])]
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    gaps.add("paid_spend_meter_screening_attempt_invalid")
                    continue
                cap = money(attempt.get("maximum_cost_usd"))
                status = attempt.get("reservation_status")
                if cap is None or status not in {"reserved", "committed", "released"}:
                    gaps.add("paid_spend_meter_screening_attempt_invalid")
                    continue
                row = {"source": SOURCE_SCREENING, "job_id": job_key(campaign.get("site_id")),
                       "meter_key": str(attempt.get("attempt_id") or f"{campaign_id}:{candidate_id}"),
                       "resource_class": "vast_provider_adapter", "provider": "vast",
                       "held_usd": None, "settled_usd": None, "settled_at_epoch": None, "status": "released"}
                if status == "reserved":
                    row.update(status="held", held_usd=cap)
                elif status == "committed":
                    settled = settled_cost(attempt)
                    settled_at = _epoch(attempt.get("settled_at"))
                    if settled is None or settled_at is None:
                        row.update(status="held", held_usd=cap)
                    else:
                        row.update(status="settled", settled_usd=settled, settled_at_epoch=settled_at)
                rows.append(row)
    return rows, sorted(gaps)


def collect_meter_view(*, now: float, world_labs: Mapping[str, Any] | None = None,
                       environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The meter for this month from every source this host records."""

    env = os.environ if environ is None else environ
    events, journal_gaps = read_journal(meter_root(env))
    queue = Path(str(env.get(SCREENING_QUEUE_ROOT_ENV) or DEFAULT_SCREENING_QUEUE_ROOT))
    screening, screening_gaps = screening_rows(queue if queue.is_dir() else None)
    gaps = [*journal_gaps, *screening_gaps]
    if world_labs is None:
        gaps.append("paid_spend_meter_atlas_attempts_unobserved")
    elif world_labs.get("complete") is not True:
        gaps.append("paid_spend_meter_atlas_attempts_partial")
    return meter_view([*journal_rows(events), *screening, *atlas_rows(world_labs)], now=now, gaps=gaps)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Print this month's per-job metered spend (reads only).")
    parser.add_argument("--job", default=None, help="show one job (WebApp request id)")
    args = parser.parse_args(argv)
    view = public_view(collect_meter_view(now=time.time()))
    if args.job:
        view["jobs"] = [row for row in view["jobs"] if row["job_id"] == job_key(args.job)]
    print(json.dumps(view, indent=2, sort_keys=True))
    return 0


__all__ = ["collect_meter_view", "screening_rows"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
