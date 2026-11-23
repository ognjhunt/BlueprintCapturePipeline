"""The build plan's sealed decision and dollar budget, read back, and the plan in plain words for the company's README.

Kept apart from ``website_build_plan`` so the company package (``website_company_simulation``, the control-plane lane)
reads the sealed decision without importing the build path. Reads only: never decides, never buys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .website_job_spend_cap import SPENDING_CAP_REACHED

#: The sealed job-object decision that carries a plan (``website_atlas_pipeline.JOB_OBJECT_RECORD``, v2).
DECISION_SCHEMA = "atlas_job_object_rule.v2"
#: ``website_atlas_pipeline.JOB_OBJECT_RECORD``, named here so this module imports nothing of the build path.
JOB_OBJECT_RECORD = "job_object_rule.json"
NOT_IN_VIDEO = "not_in_video"
#: The build plan's dollar budget record (``website_build_budget``, owner decision 2026-11-23), in ``budget_root``.
BUDGET_SCHEMA = "website_build_budget.v1"
BUDGET_RECORD = "build_budget.json"
#: An object's status there when the job's spending cap kept it out of the build (``website_build_budget.KEPT``).
KEPT_AT_CAP = "kept_as_filmed"
KEPT_AT_CAP_NOTE = "Kept as filmed: {name}, the job's spending cap was reached."


def budget_root(capture_root: Path) -> Path:
    """Where the build plan's dollar budget lives: beside the plan, under the Atlas job's ``build_plan``."""
    return Path(capture_root) / "pipeline" / "atlas" / "build_plan" / "budget"


def sealed_budget(capture_root: Path) -> dict[str, Any] | None:
    """The capture's build budget record when it seals itself, else None."""
    try:
        budget = json.loads((budget_root(capture_root) / BUDGET_RECORD).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(budget, Mapping) or budget.get("schema_version") != BUDGET_SCHEMA
            or budget.get("digest") != canonical_digest(budget, digest_field="digest")):
        return None
    return dict(budget)


def kept_at_cap(budget: Mapping[str, Any] | None, decided: Mapping[str, Any] | None) -> list[str]:
    """The names, in the plan's order, of the objects the job's spending cap kept as filmed; only from a budget made
    for this decision's own plan."""
    plan = ((decided or {}).get("build_plan") or {}).get("digest")
    if not budget or not plan or budget.get("build_plan_digest") != plan:
        return []
    return [str(row["name"]) for row in budget.get("objects") or [] if isinstance(row, Mapping)
            and row.get("status") == KEPT_AT_CAP and row.get("reason") == SPENDING_CAP_REACHED and row.get("name")]


def sealed_decision(capture_root: Path) -> dict[str, Any] | None:
    """The Atlas job's sealed job-object decision when it carries a plan (``DECISION_SCHEMA``), else None."""
    try:
        decided = json.loads((Path(capture_root) / "pipeline" / "atlas" / JOB_OBJECT_RECORD).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(decided, Mapping) or decided.get("schema_version") != DECISION_SCHEMA
            or decided.get("digest") != canonical_digest(decided, digest_field="digest")):
        return None
    return dict(decided)


def _names(rows: Sequence[Mapping[str, Any]]) -> str:
    named = [f"{row['name']} ({row['variety']})" if row.get("variety") else str(row["name"]) for row in rows]
    return ", ".join(named[:-1]) + (" and " if len(named) > 1 else "") + named[-1] if named else ""


def plan_notes(decided: Mapping[str, Any] | None, budget: Mapping[str, Any] | None = None) -> list[str]:
    """The plan in plain words for the company's README: what the simulation needs, which object the plan takes out
    of the scan to build on its own, which it keeps in the scan as filmed, what the video does not show. Each is the
    plan's choice, never that anything was built: the package's own notes say whether the object was prepared. Kept
    objects are named only where nothing else was erased (one object taken out alone, or the planner's removals not
    applied); on today's one-object route the editor erases what the planner asked. An object the job's spending cap
    kept out of the build (``budget``, ``website_build_budget``) is named once, with the cap's words."""
    plan = ((decided or {}).get("build_plan") or {}).get("plan") or {}
    rows = [row for row in plan.get("objects") or [] if isinstance(row, Mapping) and row.get("name")]
    if not rows:
        return []
    capped = kept_at_cap(budget, decided)
    built = (decided or {}).get("built") or {}
    taken = [row for row in rows if row["target_id"] == built.get("target_id")]
    unseen = [row for row in rows if row["target_id"] == NOT_IN_VIDEO]
    kept = ([row for row in rows if row not in taken and row not in unseen and row["role"] in ("moved_object", "other")
             and row["name"] not in capped]
            if taken or (decided or {}).get("applied") is True else [])
    notes = ["What this simulation needs, as Blueprint's model read the job: " + _names(rows) + "."]
    if taken:
        notes.append("The plan takes this out of the scan to build as its own object: " + _names(taken) + ".")
    if kept:
        notes.append("The plan keeps these in the scan as filmed, not built on their own: " + _names(kept) + ".")
    notes.extend(KEPT_AT_CAP_NOTE.format(name=name) for name in capped)
    if unseen:
        notes.append("Not in the video, so not in this simulation: " + _names(unseen) + ".")
    return notes


def build_plan_notes(capture_root: Path) -> dict[str, Any]:
    """``plan_notes`` of the capture's sealed decision and build budget, for the company package
    (``website_company_simulation``)."""
    notes = plan_notes(sealed_decision(capture_root), sealed_budget(capture_root))
    return {"plan_notes": notes} if notes else {}


__all__ = ["BUDGET_RECORD", "BUDGET_SCHEMA", "DECISION_SCHEMA", "JOB_OBJECT_RECORD", "KEPT_AT_CAP", "KEPT_AT_CAP_NOTE",
           "NOT_IN_VIDEO", "budget_root", "build_plan_notes", "kept_at_cap", "plan_notes", "sealed_budget",
           "sealed_decision"]
