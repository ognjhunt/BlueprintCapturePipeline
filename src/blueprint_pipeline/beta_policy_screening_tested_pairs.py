"""Which (policy, scene, harness, cells) pairs a job already has a finished scored result for.

Plan 1.2 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``): never test the same pair
twice. A pair is one candidate's policy as released (its ``configuration_digest``: the sealed catalog
row, which pins the policy's digest and revision), on one scene revision, under one screening harness
revision, over one set of screening cells (the digest of the row's frozen ``scenario_cells``). Two
results of the same pair are the same measurement, so the second is never paid for.

What counts as finished is what the coordinator already treats as a final scored result
(``beta_policy_screening_coordinator._result_not_counted``): a completed or closed row that scored at
least one episode. A run our harness failed (``_harness_failed``) or the rented machine failed
(``_machine_faulted``) is not finished: the coordinator's own bounded retries still apply to it. A result
made under an older harness revision has another key, so it never blocks the candidate's run under the
current one.

The ledger and the retained plans are the record; ``record_tested_pairs`` also writes the job's derived
list under ``<queue>/tested_pairs/<site_id>.json`` for operators. Nothing here moves money.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

RECORD_SCHEMA = "blueprint.beta_screening_tested_pairs.v1"
RECORD_DIRNAME = "tested_pairs"
#: Raised by the planner when every requested row is a pair the job already has a result for.
EVERY_PAIR_TESTED = "beta_screening_every_pair_already_tested"


def cells_digest(cells: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest({"cells": list(cells)})


def pair_key(*, candidate_id: str, configuration_digest: str, scene_revision_digest: str, harness_revision: int,
             cells: Sequence[Mapping[str, Any]]) -> str:
    """The identity of one measurement: the same key is the same test."""
    return canonical_digest({"candidate_id": candidate_id, "configuration_digest": configuration_digest,
                             "scene_revision_digest": scene_revision_digest,
                             "screening_harness_revision": int(harness_revision),
                             "cells_digest": cells_digest(cells)})


def plan_row_key(plan: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    """A planned row's key; ``None`` for a row that froze no cells (it never ran)."""
    from .beta_policy_screening_plan import plan_harness_revision

    cells = row.get("scenario_cells")
    if not isinstance(cells, list) or not cells:
        return None
    return pair_key(candidate_id=row["candidate_id"], configuration_digest=row["configuration_digest"],
                    scene_revision_digest=plan["scene_revision_digest"],
                    harness_revision=plan_harness_revision(plan), cells=cells)


def finished(row: Mapping[str, Any]) -> bool:
    """A ledger row holding a final scored result (never an infrastructure failure)."""
    from .beta_policy_screening_coordinator import _harness_failed, _machine_faulted
    from .beta_policy_screening_known_good_control import is_control

    summary = row.get("summary") or {}
    return (not is_control(row) and row.get("status") in {"completed", "blocked"}
            and bool(summary.get("episodes_scored")) and not _harness_failed(row) and not _machine_faulted(row))


def in_flight(row: Mapping[str, Any]) -> bool:
    """A ledger row that may still run (queued or active), in a campaign that was not superseded."""
    from .beta_policy_screening_known_good_control import is_control
    from .beta_policy_screening_result import ACTIVE_STATUSES, QUEUED_STATUSES

    return not is_control(row) and row.get("status") in QUEUED_STATUSES | ACTIVE_STATUSES


def site_pairs(queue_root: str | Path | None, site_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    """``{"finished": {key: entry}, "in_flight": {key: entry}}`` for one job, from the ledger and its plans."""
    from .beta_policy_screening_coordinator import _ledger, _lock, _plan, _root

    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
    found: dict[str, dict[str, dict[str, Any]]] = {"finished": {}, "in_flight": {}}
    for campaign_id, state in sorted(ledger["campaigns"].items(), key=lambda item: (item[1]["created_at"], item[0])):
        if state["site_id"] != site_id:
            continue
        rows = state["candidates"]
        wanted = {key for key, row in rows.items()
                  if finished(row) or in_flight(row) and not state.get("superseded_by")}
        if not wanted:
            continue
        plan = _plan(root, campaign_id, state["plan_digest"])
        for configured in plan["candidates"]:
            row = rows.get(configured["candidate_id"]) or {}
            if configured["candidate_id"] not in wanted:
                continue
            key = plan_row_key(plan, configured)
            if key is None:
                continue
            kind = "finished" if finished(row) else "in_flight"
            found[kind].setdefault(key, {
                "pair_key": key, "candidate_id": configured["candidate_id"],
                "configuration_digest": configured["configuration_digest"],
                "scene_revision_digest": plan["scene_revision_digest"],
                "screening_harness_revision": plan.get("screening_harness_revision", 0),
                "cells_digest": cells_digest(configured["scenario_cells"]), "campaign_id": campaign_id,
                "status": row.get("status"),
                **({"session_result_digest": (row.get("summary") or {}).get("session_result_digest")}
                   if kind == "finished" else {})})
    # A finished pair is finished even while another campaign still lists it.
    for key in found["finished"]:
        found["in_flight"].pop(key, None)
    return found


def finished_pair_keys(queue_root: str | Path | None, site_id: str) -> frozenset[str]:
    return frozenset(site_pairs(queue_root, site_id)["finished"])


def record_tested_pairs(queue_root: str | Path | None, site_id: str) -> dict[str, Any]:
    """Write (and return) the job's derived record of finished pairs. Idempotent; moves no money."""
    from .beta_policy_screening_coordinator import _root, _write

    root = _root(queue_root)
    pairs = site_pairs(root, site_id)
    record = {"schema_version": RECORD_SCHEMA, "site_id": site_id,
              "finished": [pairs["finished"][key] for key in sorted(pairs["finished"])],
              "in_flight": [pairs["in_flight"][key] for key in sorted(pairs["in_flight"])], "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    directory = root / RECORD_DIRNAME
    directory.mkdir(exist_ok=True, mode=0o700)
    _write(directory / f"{site_id}.json", record)
    return record


__all__ = ["EVERY_PAIR_TESTED", "RECORD_SCHEMA", "cells_digest", "finished", "finished_pair_keys", "in_flight",
           "pair_key", "plan_row_key", "record_tested_pairs", "site_pairs"]
