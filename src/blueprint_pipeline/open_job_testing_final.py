"""Decide, in the screening pass, which website jobs' testing is final, and record it for the storage GC.

Beta journey step: L5 team policy evaluation. The storage GC keeps an open job's
scene run on the host (``open_job_source_runs``) until the job's testing is final.
Deciding that needs the screening coordinator's own re-plan rule
(``beta_policy_screening_coordinator._replan_scope``), and the coordinator reaches
the paid allocator and the GPU providers, which the control-plane lane must never
import (``tests/test_live_pipeline_import_isolation.py``). So the screening pass,
which loads the coordinator anyway, decides it here after each tick and writes
``TESTING_RECORD_FILENAME`` beside its ledger. The record is sealed to the exact
ledger it was computed from: the GC trusts it only while the ledger's digest is
unchanged, so any later ledger write reads as "not final" until the next pass
writes a new record. Local and unpaid; nothing here moves, deletes or spends.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .open_job_source_runs import LEDGER_FILENAME, TESTING_RECORD_FILENAME, TESTING_RECORD_SCHEMA


def _final_by_intent(root: Path) -> tuple[str, dict[str, bool]]:
    from . import beta_policy_screening_coordinator as coordinator

    ledger = coordinator._ledger(root)

    def plans(campaign_id: str, digest: str) -> Mapping[str, Any]:
        return coordinator._plan(root, campaign_id, digest)

    newest: dict[str, list[Mapping[str, Any]]] = {}
    for campaign_id, state in sorted(ledger["campaigns"].items()):
        if state.get("superseded_by"):
            continue
        plan = plans(campaign_id, state["plan_digest"])
        intent_id = (plan.get("source") or {}).get("scene_intent_id")
        if isinstance(intent_id, str) and intent_id:
            newest.setdefault(intent_id, []).append(plan)
    final = {intent_id: all(_campaign_final(ledger, plan, plans) for plan in rows)
             for intent_id, rows in newest.items()}
    return str(ledger.get("ledger_digest") or ""), final


def testing_final(intent_id: str, *, screening_queue_root: str | Path) -> bool:
    """Whether the intent's screening ended for good; no campaign, or anything unreadable, is not final."""

    root = Path(screening_queue_root)
    try:
        if root.is_symlink() or not (root / LEDGER_FILENAME).is_file():
            return False
        return _final_by_intent(root)[1].get(intent_id, False)
    except Exception:  # noqa: BLE001 - unreadable testing state keeps the run; the window still bounds it
        return False


def _campaign_final(ledger: Mapping[str, Any], plan: Mapping[str, Any],
                    plans: Callable[[str, str], Mapping[str, Any]]) -> bool:
    """A campaign whose learned rows all ended and that no deploy, retry or re-plan would take up again.

    Stricter than the known-good control's own rule (``_final_without_control``), which also counts a
    row closed for money as final: such a row is still planned again after a deploy while the site has
    money left (``_replan_scope``), and that re-plan reads the scene run (capture-447ae4f4, 2026-11-11).
    The scripted reference row is left out: its plan is sealed and it never reads the run.
    """
    from . import beta_policy_screening_coordinator as coordinator
    from .beta_policy_screening_known_good_control import is_control
    from .beta_policy_screening_result import campaign_status

    state = ledger["campaigns"][plan["campaign_id"]]
    rows = {key: row for key, row in state["candidates"].items() if not is_control(row)}
    if state.get("superseded_by") or campaign_status(rows) != "completed":
        return False
    learned = {**plan, "candidates": [row for row in plan["candidates"] if not is_control(row)]}
    return coordinator._replan_scope(ledger, learned, coordinator._ANY_LATER_COMMIT, plans=plans) is None


def publish_testing_record(queue_root: str | Path) -> dict[str, Any]:
    """Write the sealed record for the ledger as it is now. A failure removes nothing and keeps every run."""

    root = Path(queue_root)
    path = root / TESTING_RECORD_FILENAME
    if root.is_symlink() or not (root / LEDGER_FILENAME).is_file():
        return {"status": "no_ledger"}
    ledger_digest, final = _final_by_intent(root)
    record = {"schema_version": TESTING_RECORD_SCHEMA, "ledger_digest": ledger_digest,
              "final_by_intent": dict(sorted(final.items())), "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return {"status": "written", "intents": len(final), "final": sum(final.values())}


__all__ = ["publish_testing_record", "testing_final"]
