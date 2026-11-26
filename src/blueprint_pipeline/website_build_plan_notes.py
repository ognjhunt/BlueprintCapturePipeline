"""The build plan's sealed decision and dollar budget, read back, and the plan in plain words for the company's README.

Kept apart from ``website_build_plan`` so the company package (``website_company_simulation``, the control-plane lane)
reads the sealed decision without importing the build path. Reads only: never decides, never buys.

The plan's names and varieties are a model's phrases from the site's own words, which may name a person or a place;
only the WebApp can redact the site's words, so the README names each object in plain object words only
(``plain_name``, ``website_object_words``): never a phrase as the plan wrote it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .website_job_spend_cap import SPENDING_CAP_REACHED
from .website_object_words import plain_phrase

#: The sealed job-object decision that carries a plan (``website_atlas_pipeline.JOB_OBJECT_RECORD``). Sealed under v3
#: (several objects per job, stage 5), it also names each further object the plan would build (``further_rows``), and
#: those it takes out of the scan with the job's object to be built on their own. A decision sealed under v2 keeps its
#: route: its further objects stay in the scan as filmed, as they always did (``PLAN_SCHEMAS``).
DECISION_SCHEMA = "atlas_job_object_rule.v3"
#: Every sealed decision that carries a plan, read alike; only ``DECISION_SCHEMA`` takes further objects out.
PLAN_SCHEMAS = ("atlas_job_object_rule.v2", DECISION_SCHEMA)
#: ``website_atlas_pipeline.JOB_OBJECT_RECORD``, named here so this module imports nothing of the build path.
JOB_OBJECT_RECORD = "job_object_rule.json"
NOT_IN_VIDEO = "not_in_video"
#: The build plan's dollar budget record (``website_build_budget``, owner decision 2026-11-23), in ``budget_root``.
BUDGET_SCHEMA = "website_build_budget.v1"
BUDGET_RECORD = "build_budget.json"
#: An object's status there when a limit on the job's paid work kept it out of the build (``website_build_budget.KEPT``).
KEPT_AT_CAP = "kept_as_filmed"
#: What the website's refusal may mean. It sends one code, ``website_scene_preparation_budget_exhausted``, for both of
#: the job's limits on paid work: the $40 dollar cap (``checkWebsiteJobCeiling``) and the native ledger's paid-request
#: count (``preparationRequestLimit``, the sponsorship's ``max_paid_attempts``). It does not say which refused, so no
#: record or sentence here claims one.
REFUSAL_CODE_MEANS_ONE_OF = ("job_dollar_cap", "paid_request_count")
#: True whichever of the two limits refused.
KEPT_AT_CAP_NOTE = "Kept as filmed: {name}, the job reached its limit on paid work."


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


def _kept_rows(budget: Mapping[str, Any] | None, decided: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    plan = ((decided or {}).get("build_plan") or {}).get("digest")
    if not budget or not plan or budget.get("build_plan_digest") != plan:
        return []
    return [row for row in budget.get("objects") or [] if isinstance(row, Mapping)
            and row.get("status") == KEPT_AT_CAP and row.get("reason") == SPENDING_CAP_REACHED and row.get("name")]


def kept_at_cap(budget: Mapping[str, Any] | None, decided: Mapping[str, Any] | None) -> list[str]:
    """The names, in the plan's order, of the objects a limit on the job's paid work kept as filmed; only from a budget
    made for this decision's own plan."""
    return [str(row["name"]) for row in _kept_rows(budget, decided)]


def kept_disclosure(budget: Mapping[str, Any] | None, decided: Mapping[str, Any] | None) -> dict[str, Any]:
    """The handoff's disclosure of those objects: their names under the machine reason, and the website's refusal code
    exactly as received with the two limits it may mean (``REFUSAL_CODE_MEANS_ONE_OF``); never which one. Empty when
    none was kept."""
    rows = _kept_rows(budget, decided)
    if not rows:
        return {}
    code = next((row["refusal_code"] for row in rows if isinstance(row.get("refusal_code"), str)), None)
    return {"kept_at_spending_cap": [str(row["name"]) for row in rows], "reason": SPENDING_CAP_REACHED,
            **({"refusal_code": code} if code else {}),
            "refusal_code_means_one_of": list(REFUSAL_CODE_MEANS_ONE_OF)}


def sealed_decision(capture_root: Path) -> dict[str, Any] | None:
    """The Atlas job's sealed job-object decision when it carries a plan (``PLAN_SCHEMAS``), else None."""
    try:
        decided = json.loads((Path(capture_root) / "pipeline" / "atlas" / JOB_OBJECT_RECORD).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(decided, Mapping) or decided.get("schema_version") not in PLAN_SCHEMAS
            or decided.get("digest") != canonical_digest(decided, digest_field="digest")):
        return None
    return dict(decided)


def further_rows(decided: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Each further object a decision sealed under ``DECISION_SCHEMA`` names (``website_build_plan.further_removal``),
    by rank: ``removed`` when it was taken out of the scan with the job's object, to be built on its own, else with its
    ``kept_in_scan_reason``. Empty for a decision sealed earlier, whose further objects all stay in the scan."""
    if not isinstance(decided, Mapping) or decided.get("schema_version") != DECISION_SCHEMA:
        return []
    return [dict(row) for row in decided.get("further") or [] if isinstance(row, Mapping)
            and isinstance(row.get("target_id"), str) and type(row.get("rank")) is int]


#: What an object is called in the README when its plan name is not in plain object words (``plain_phrase``): its
#: parametric shape, else its role in the job. Never the site's words.
SHAPE_WORDS = {"carton": "carton", "tote": "tote", "open_bin": "open bin", "tray": "tray", "cylinder": "cylinder"}
ROLE_WORDS = {"moved_object": "an object the job moves", "destination": "where an object goes",
              "fixture": "a fixture the job needs", "other": "another object"}
UNNAMED = "an object"


def plain_name(row: Mapping[str, Any]) -> str:
    """The README's name for one plan row: the plan's name only in plain object words, else its parametric shape,
    else its role (``SHAPE_WORDS``, ``ROLE_WORDS``). The plan's words are a model's phrase from the site's, which may
    name a person ("Dr. Ana Patel's sample tray"); only the WebApp can redact the site's words, so none ships as
    written (``website_object_words``)."""
    name = plain_phrase(row.get("name"))
    if name is None:
        shape = row.get("parametric_shape") if row.get("sourcing") == "parametric" else None
        name = SHAPE_WORDS.get(shape) if isinstance(shape, str) else None
    return name or ROLE_WORDS.get(str(row.get("role")), UNNAMED)


def _names(rows: Sequence[Mapping[str, Any]]) -> str:
    named = []
    for row in rows:
        variety = plain_phrase(row.get("variety"))  # dropped when it is not in plain words: the name still says it
        named.append(f"{plain_name(row)} ({variety})" if variety else plain_name(row))
    return ", ".join(named[:-1]) + (" and " if len(named) > 1 else "") + named[-1] if named else ""


def plan_names(decided: Mapping[str, Any] | None) -> dict[str, str]:
    """Each object of the sealed plan by its target, in plain object words only (``plain_name``)."""
    plan = ((decided or {}).get("build_plan") or {}).get("plan") or {}
    return {str(row["target_id"]): plain_name(row) for row in plan.get("objects") or []
            if isinstance(row, Mapping) and isinstance(row.get("target_id"), str)}


#: Why a further object of the job was left in the scan as filmed, by the reason code its preparation recorded
#: (``website_further_objects``; named here, since that module is the build path): its code up to the first ``:``, and
#: for an overlap, what it meets. Any other code is said as ``LEFT_FOR_ANOTHER_REASON``, never guessed at.
LEFT_WORDS = {"further_object_needs_artisan": "it needs a model Blueprint would generate, and none was made for it",
              "further_object_not_tracked": "Blueprint could not follow it through the video's frames",
              "further_object_scene_not_registered": "the room scan could not be lined up with the video",
              "further_object_support_not_found": "no surface under it was found in the room scan",
              "further_object_authoring_failed": "Blueprint could not make a shape of its size",
              "further_object_static_check_failed": "its file did not pass Blueprint's static check",
              "further_object_input_invalid": "the plan's entry for it could not be used",
              "website_preparation_further_objects_left_out": "the simulation's request could not take it",
              "scene_intake_task_additional_objects_strategy_invalid": "this kind of job takes no further objects"}
OVERLAP_WORDS = {"scored_object_start": "where the video shows it, it would overlap the job's object",
                 "scored_object_destination": "where the video shows it, it would overlap where the job's object goes",
                 "further_object_rank": "where the video shows it, it would overlap another of the job's objects"}
LEFT_FOR_ANOTHER_REASON = "a step that builds it could not be completed"
#: What happened to the plan's further objects (``further``, the half's preparation's record as
#: ``website_half_preparation.further_objects`` reads it).
BUILT_TOO = "Also built as {its} own object{s}, where the video shows {them}: {names}."
LEFT_IN_SCAN = "Left in the scan as filmed, not built on its own: {name}, because {why}."
#: A further object taken out of the scan before the scan was made, to be built on its own, and then not built (the
#: record's ``removed_not_built``): it is in neither the scan nor the simulation, so it is never said to be in the scan.
REMOVED_NOT_BUILT = ("Taken out of the scan to be built as its own object, but it could not be built, so it is not in "
                     "this simulation: {name}, because {why}.")
#: The further objects a decision sealed under v3 takes out of the scan, said where no record of the half's own
#: preparation says what became of them (a room package): the plan's choice, never that they were built.
ALSO_TAKEN = "The plan also takes {this} out of the scan to build as {its} own object{s}: {names}."


def left_words(code: Any) -> str:
    """Why a further object stayed in the scan as filmed, in plain words, from its reason code."""
    head, _, detail = str(code).partition(":")
    if head == "further_object_overlaps":
        return next((words for key, words in OVERLAP_WORDS.items() if detail.startswith(key)), LEFT_FOR_ANOTHER_REASON)
    return LEFT_WORDS.get(head, LEFT_FOR_ANOTHER_REASON)


def _further(rows: Sequence[Mapping[str, Any]], further: Mapping[str, Any] | None,
             capped: set[str]) -> tuple[set[str], list[str]]:
    """The plan's targets the half's preparation decided on as further objects, and what it did with each, in words:
    those it built, then each taken out of the scan that it could not build, then each it left in the scan, each with
    why, by rank; one a limit on paid work kept (``capped``) is said once, by ``KEPT_AT_CAP_NOTE``."""
    composed = [row for row in (further or {}).get("composed") or [] if isinstance(row, Mapping)]
    gone = [row for row in (further or {}).get("removed_not_built") or [] if isinstance(row, Mapping)]
    left = [row for row in (further or {}).get("left_in_scan") or [] if isinstance(row, Mapping)
            and str(row.get("target_id")) not in capped]
    names = {str(row.get("target_id")): plain_name(row) for row in rows}
    built = [names.get(str(row.get("target_id")), UNNAMED) for row in composed]
    notes = [BUILT_TOO.format(its="its" if len(built) == 1 else "their", s="" if len(built) == 1 else "s",
                              them="it" if len(built) == 1 else "them", names=_joined(built))] if built else []
    notes += [words.format(name=names.get(str(row.get("target_id")), UNNAMED), why=left_words(row.get("reason")))
              for words, found in ((REMOVED_NOT_BUILT, gone), (LEFT_IN_SCAN, left)) for row in found]
    return {str(row.get("target_id")) for row in [*composed, *gone, *left]}, notes


def _joined(names: Sequence[str]) -> str:
    return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]


def plan_notes(decided: Mapping[str, Any] | None, budget: Mapping[str, Any] | None = None,
               further: Mapping[str, Any] | None = None) -> list[str]:
    """The plan in plain words for the company's README: what the simulation needs, which object the plan takes out
    of the scan to build on its own, which it keeps in the scan as filmed, what the video does not show. Each is the
    plan's choice, never that anything was built: the package's own notes say whether the object was prepared. Kept
    objects are named only where nothing else was erased (one object taken out alone, or the planner's removals not
    applied); on today's one-object route the editor erases what the planner asked. An object a limit on the job's paid
    work kept out of the build (``budget``, ``website_build_budget``) is named once, in words true of either limit.
    Every object is named in plain object words only (``plain_name``), never as the plan wrote it.

    ``further``: what the half's own preparation did with the plan's further objects (``website_further_objects``):
    each it built is said to be built (``BUILT_TOO``), each taken out of the scan that it could not build is said to
    be in neither (``REMOVED_NOT_BUILT``), and each it left in the scan as filmed is named with why (``LEFT_IN_SCAN``),
    never among the objects the plan keeps. A decision sealed under v3 that took further objects out of the scan with
    the job's object says so (``ALSO_TAKEN``) for each no such record names: never among the objects it keeps. Without
    either the notes are as they were."""
    plan = ((decided or {}).get("build_plan") or {}).get("plan") or {}
    rows = [row for row in plan.get("objects") or [] if isinstance(row, Mapping) and row.get("name")]
    decided_on, further_notes = _further(rows, further, {str(row.get("target_id"))
                                                         for row in _kept_rows(budget, decided)})
    if not rows:
        return further_notes
    capped = kept_at_cap(budget, decided)
    by_target = {row.get("target_id"): row for row in rows}
    capped_names = [plain_name(by_target[row.get("target_id")]) if row.get("target_id") in by_target
                    else plain_phrase(row["name"]) or UNNAMED for row in _kept_rows(budget, decided)]
    built = (decided or {}).get("built") or {}
    taken = [row for row in rows if row["target_id"] == built.get("target_id")]
    removed = {row["target_id"] for row in further_rows(decided) if row.get("removed") is True} if taken else set()
    also = [row for row in rows if row["target_id"] in removed and row["target_id"] not in decided_on]
    unseen = [row for row in rows if row["target_id"] == NOT_IN_VIDEO]
    kept = ([row for row in rows if row not in taken and row not in unseen and row["role"] in ("moved_object", "other")
             and row["name"] not in capped and row["target_id"] not in (decided_on | removed)]
            if taken or (decided or {}).get("applied") is True else [])
    notes = ["What this simulation needs, as Blueprint's model read the job: " + _names(rows) + "."]
    if taken:
        notes.append("The plan takes this out of the scan to build as its own object: " + _names(taken) + ".")
    if also:
        one = len(also) == 1
        notes.append(ALSO_TAKEN.format(this="this" if one else "these", its="its" if one else "their",
                                       s="" if one else "s", names=_names(also)))
    notes += further_notes
    if kept:
        notes.append("The plan keeps these in the scan as filmed, not built on their own: " + _names(kept) + ".")
    notes.extend(KEPT_AT_CAP_NOTE.format(name=name) for name in capped_names)
    if unseen:
        notes.append("Not in the video, so not in this simulation: " + _names(unseen) + ".")
    return notes


def build_plan_notes(capture_root: Path, further: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``plan_notes`` of the capture's sealed decision and build budget, and of what the half's own preparation did
    with the plan's further objects (``further``), for the company package (``website_company_simulation``)."""
    notes = plan_notes(sealed_decision(capture_root), sealed_budget(capture_root), further)
    return {"plan_notes": notes} if notes else {}


__all__ = ["ALSO_TAKEN", "BUDGET_RECORD", "BUDGET_SCHEMA", "BUILT_TOO", "DECISION_SCHEMA", "JOB_OBJECT_RECORD",
           "KEPT_AT_CAP", "KEPT_AT_CAP_NOTE", "LEFT_FOR_ANOTHER_REASON", "LEFT_IN_SCAN", "LEFT_WORDS", "NOT_IN_VIDEO",
           "OVERLAP_WORDS", "PLAN_SCHEMAS", "REFUSAL_CODE_MEANS_ONE_OF", "REMOVED_NOT_BUILT", "ROLE_WORDS", "SHAPE_WORDS",
           "UNNAMED", "budget_root", "build_plan_notes", "further_rows", "kept_at_cap", "kept_disclosure", "left_words",
           "plain_name", "plan_names", "plan_notes", "sealed_budget", "sealed_decision"]
