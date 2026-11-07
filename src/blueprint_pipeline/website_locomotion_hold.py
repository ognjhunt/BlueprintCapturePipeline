"""The typed hold for a compiled task that no robot policy can run here yet.

Pure: it builds the hold from runnable options already computed
(``website_locomotion_task.runnable_options``), so the hold contracts
(``website_task_context_contracts``) can re-check a recorded hold without importing the
robot or policy inventory. The handoff, which computes the options, is what releases a
hold once a body or policy becomes runnable.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

HOLD_CODE = "website_task_runnable_policy_missing"


def runnable_policy_gap_from_options(*, task_context: Mapping[str, Any], requirements: Mapping[str, Any],
                                     spec_digest: str, options: Mapping[str, Any],
                                     task_semantics: Mapping[str, Any] | None = None,
                                     source_video_digest: str | None = None) -> dict[str, Any] | None:
    """The hold for these options; ``None`` when one of their policies is runnable."""
    options = dict(options)
    if dict(options.get("requirements") or {}) != dict(requirements):
        raise ValueError("website_runnable_options_requirements_mismatch")
    if any(row["runnable"] for row in options["policies"]):
        return None
    missing = sorted({item for row in options["policies"] for item in row["missing"]}) or ["policy_for_requirements"]
    bodies = ", ".join(f"{row['robot']} ({row['status']})" for row in options["embodiments"]) or "none listed"
    names = ", ".join(row["candidate_id"] for row in options["policies"]) or "none in the catalog"
    # Items a body's development-run allowance stands in for (the registry's STOOD_IN): named as such, never
    # as satisfied, and read from the recorded options so the hold contract re-derives the same text.
    stood_in = sorted({f"embodiment:{item}" for row in options["policies"]
                       for item in (row.get("development_run_allowance") or {}).get("stood_in") or {}})
    return {
        "code": HOLD_CODE, "task_family": task_context.get("task_family") or "general_task",
        "task_context_digest": task_context["context_digest"], "job_disposition": "held_for_preparation",
        "description": task_context.get("description"), "compiled_task_spec_digest": spec_digest,
        "runnable_options": options, "missing": missing,
        "reason": ("The task is compiled and its scene is kept. No robot policy can run it here yet. "
                   f"Bodies that can do it: {bodies}. Policies for them: {names}. Missing: {', '.join(missing)}."
                   + (f" {', '.join(stood_in)}: stood in by the development-run allowance (development_only)."
                      if stood_in else "")),
        "claim_boundary": "Blueprint integration gap only; no finding that robot teams cannot perform the job.",
        **({"task_semantics": dict(task_semantics), "source_video_digest": source_video_digest}
           if task_semantics is not None else {}),
    }


def navigation_intake_refusal_hold(*, task_context: Mapping[str, Any], requirements: Mapping[str, Any],
                                   spec_digest: str, refusal: str, options: Mapping[str, Any],
                                   task_semantics: Mapping[str, Any] | None = None,
                                   source_video_digest: str | None = None) -> dict[str, Any]:
    """The hold for a navigation job whose website intake was refused: held by the refusal's own name.

    ``refusal`` is the website's typed code (an unconfirmed brief is ``website_task_context_not_confirmed``,
    ``website_navigation_intake``). The options computed for the job are kept beside it, so the record
    still names every other piece the job lacks.
    """
    options = dict(options)
    if dict(options.get("requirements") or {}) != dict(requirements):
        raise ValueError("website_runnable_options_requirements_mismatch")
    if not isinstance(refusal, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,99}", refusal) or refusal == HOLD_CODE:
        raise ValueError("website_navigation_intake_refusal_invalid")
    return {
        "code": refusal, "task_family": task_context.get("task_family") or "general_task",
        "task_context_digest": task_context["context_digest"], "job_disposition": "held_for_preparation",
        "description": task_context.get("description"), "compiled_task_spec_digest": spec_digest,
        "navigation_intake": {"status": "refused", "reason": refusal}, "runnable_options": options,
        "reason": (f"The website refused this job's navigation intake: {refusal}. The task is compiled and its "
                   "scene is kept."),
        "claim_boundary": "Blueprint integration gap only; no finding that robot teams cannot perform the job.",
        **({"task_semantics": dict(task_semantics), "source_video_digest": source_video_digest}
           if task_semantics is not None else {}),
    }


#: A runnable navigation job handed to the G1 screening lane (``navigation_preparation``).
NAVIGATION_PREPARATION_SCHEMA = "website_navigation_preparation.v1"
NAVIGATION_LANE = "g1_navigation_screening"
#: The catalog's G1 navigation checkpoints (``_vision_navi`` rows): the only candidates that lane runs.
G1_NAVIGATION_SUFFIX = "_vision_navi"
NAVIGATION_SUMMARY = ("Handed to screening in simulation: the G1 navigation candidates run against the scene "
                      "kept as filmed. No result yet.")
SOURCE_FIELDS = ("scene_intent_id", "scene_intent_digest", "capture_id", "task_id",
                 "task_context_digest", "scene_revision_digest")


def g1_navigation_lane_candidates(options: Mapping[str, Any]) -> list[str] | None:
    """The runnable candidates when every runnable option is a G1 navigation checkpoint; else None.

    Any other runnable option (a body or policy with no lane here) keeps the handoff's typed
    ``website_locomotion_task_execution_route_missing``.
    """
    from .task_evaluation_g1_catalog import G1_EMBODIMENT_ID

    runnable = [row for row in options.get("policies") or () if row.get("runnable") is True]
    if not runnable or any(not str(row.get("candidate_id") or "").endswith(G1_NAVIGATION_SUFFIX)
                           or row.get("embodiment_id") != G1_EMBODIMENT_ID for row in runnable):
        return None
    return sorted(row["candidate_id"] for row in runnable)


def navigation_preparation(*, task_context: Mapping[str, Any], spec_digest: str, options: Mapping[str, Any],
                           scene_inputs: Mapping[str, Any], receipt: Mapping[str, Any],
                           task_semantics: Mapping[str, Any] | None = None,
                           source_video_digest: str | None = None) -> dict[str, Any] | None:
    """The handoff's record of a navigation job handed to the G1 screening lane; None when it is not one.

    Screening enters through the team registry's derived row (``beta_policy_screening_family_sources``),
    which reads the record and its intake receipt, not this status. It claims no result.
    """
    candidates = g1_navigation_lane_candidates(options)
    source = receipt.get("source") if isinstance(receipt, Mapping) else None
    if (candidates is None or not isinstance(source, Mapping) or set(source) != set(SOURCE_FIELDS)
            or source.get("task_context_digest") != task_context.get("context_digest")
            or source.get("scene_revision_digest") != scene_inputs.get("scene_revision_digest")):
        return None
    return {"schema_version": NAVIGATION_PREPARATION_SCHEMA, "lane": NAVIGATION_LANE,
            "claim_ceiling": "development_only", "result_claimed": False, "summary": NAVIGATION_SUMMARY,
            "task_context_digest": task_context["context_digest"], "compiled_task_spec_digest": spec_digest,
            "candidates": candidates, "runnable_options": dict(options),
            "navigation_scene_inputs": {key: scene_inputs[key] for key in ("path", "digest", "scene_revision_digest")},
            "intake_receipt_digest": receipt["receipt_digest"], "source": dict(source),
            **({"task_semantics": dict(task_semantics), "source_video_digest": source_video_digest}
               if task_semantics is not None else {})}


__all__ = ["G1_NAVIGATION_SUFFIX", "HOLD_CODE", "NAVIGATION_LANE", "NAVIGATION_PREPARATION_SCHEMA",
           "g1_navigation_lane_candidates", "navigation_intake_refusal_hold", "navigation_preparation",
           "runnable_policy_gap_from_options"]
