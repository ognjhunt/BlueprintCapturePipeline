"""Source-bound, read-only preparation findings for the existing website status.

ADP-009D/day-28 precursor: conservative geometry refusals are not an exhaustive
mechanical impossibility proof. No provider call, retry or new spending authority.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
# The capability verdict abstains only by the hard reach rule (``website_rigid_feasibility``); the scripted
# carry codes are kept for verdicts written before it.
_PATH_REFUSAL = re.compile(r"robot_rigid_pick_place_(?:unreachable|scene_strike):[a-z_]+\Z"
                           r"|robot_reach_insufficient:(?:pick|place):\d+\.\d{3}m_vs_\d+\.\d{3}m\Z")
_MISSING = frozenset({
    "website_registration_scale_conflicts_declared", "support_surface_not_found_under_subject",
    "task_destination_pose_required", "task_destination_relation_required",
    "task_destination_interior_geometry_required", "task_destination_surface_contact_required",
    "website_owner_success_criteria_translation_required", "website_assembly_whole_object_coverage_required",
    "website_synthetic_subject_start_unplaceable",
})


#: Three switches, one per website contract, because the website's readers are strict: a reason or a
#: question kind it does not know refuses the job's whole status (review 2026-11-12).
#: (i) ``NEW_REASONS_ENV``, ON unless set to ``false``: the reasons the live website already words,
#: ``website_task_object_not_observed``, ``website_capture_frames_unusable`` and ``website_task_not_named``
#: (with its waiting status). Off, each travels as the nearest reason the older website words.
NEW_REASONS_ENV = "BLUEPRINT_WEBSITE_ACCEPTS_STAGE6_RECAPTURE_REASONS"
#: (ii) OFF unless ``true``: ``website_preparation_not_possible`` (with ``cause``) and
#: ``website_preparation_retries_spent``, until the website's follow-up lists them.
ENDINGS_ENV = "BLUEPRINT_WEBSITE_ACCEPTS_PREPARATION_ENDINGS"
#: (iii) OFF unless ``true``: the place question (``task_destination_choice``) and its picture.
PLACE_QUESTION_ENV = "BLUEPRINT_WEBSITE_ACCEPTS_PLACE_QUESTION"
#: The reason sent instead while the switch is off: not enough evidence for a robot decision.
NEW_REASON_FALLBACK = {"website_task_object_not_observed": "task_evidence_incomplete",
                       "website_capture_frames_unusable": "task_evidence_incomplete"}


#: Part 6 (2026-11-12): endings that told the site "Blueprint will review" when nothing reviews. Each now
#: says what is true. ``NOT_POSSIBLE``: this release cannot prepare the job, for the cause it names (a
#: retained blocker code). The preparation stage fails on it (``website_capture_preparation``: the
#: ledger is ``failed_retryable``), so the retry lane runs it again: six timed retries, then once per
#: release (``website_preparation_retry.RETRIES_SPENT``).
#: ``RETRIES_SPENT``: a step kept failing after its timed retries; it runs again once on each new release
#: (``website_preparation_retry.spent_waiting``), or daily when the release cannot be read.
NOT_POSSIBLE = "website_preparation_not_possible"
RETRIES_SPENT = "website_preparation_retries_spent"


def _switch(name: str, *, default: bool) -> bool:
    import os
    value = os.environ.get(name, "").strip().lower()
    return default if value not in {"true", "false"} else value == "true"


def website_lists_new_reasons() -> bool:
    """Switch (i): on by default."""
    return _switch(NEW_REASONS_ENV, default=True)


def website_lists_endings() -> bool:
    """Switch (ii): off by default."""
    return _switch(ENDINGS_ENV, default=False)


def website_lists_place_question() -> bool:
    """Switch (iii): off by default."""
    return _switch(PLACE_QUESTION_ENV, default=False)


def retries_spent(ledger: dict[str, Any], *, task_context_digest: str) -> dict[str, Any] | None:
    """Status v7's outcome for a job whose timed retries are spent, while the website lists it; else None."""
    from .website_preparation_retry_record import RETRIES_SPENT as KIND, armed
    record = armed(ledger)
    if not website_lists_endings() or record is None or record.get("kind") != KIND:
        return None
    evidence = canonical_digest({"capture_id": ledger.get("capture_id"), "task_context_digest": task_context_digest,
                                 "code": RETRIES_SPENT, "source_payload_sha256": ledger.get("source_payload_sha256"),
                                 "retry_count": record.get("retry_count")})
    return {"category": "processing_failure", "reason": RETRIES_SPENT, "basis": "execution_failure",
            "robot_id": None, "evidence_digest": evidence}


def sent_recapture_reason(code: str) -> str:
    """The outcome reason the status sends for a recapture stop, honouring what the website accepts."""
    if code in NEW_REASON_FALLBACK and not website_lists_new_reasons():
        return NEW_REASON_FALLBACK[code]
    return code


def awaiting_job_sentence(ledger: dict[str, Any], *, task_context_digest: str) -> dict[str, Any] | None:
    """Status v7's outcome for a job waiting for the site to name it, while the website accepts it; else None.

    The preparation stopped, buying nothing, on ``website_task_not_named`` and retries until the job is named
    (``website_preparation_retry.AWAITING_JOB_SENTENCE``). Behind the same switch as the new reasons.
    """

    from .website_task_context_contracts import TASK_NOT_NAMED
    if (not website_lists_new_reasons()
            or ledger.get("status") != "failed_retryable"
            or not str(ledger.get("last_error") or "").endswith(TASK_NOT_NAMED)):
        return None
    evidence = canonical_digest({"capture_id": ledger.get("capture_id"), "task_context_digest": task_context_digest,
                                 "code": TASK_NOT_NAMED, "source_payload_sha256": ledger.get("source_payload_sha256")})
    return {"category": "insufficient_evidence", "reason": TASK_NOT_NAMED, "basis": "missing_evidence",
            "robot_id": None, "evidence_digest": evidence}


def _load(path: Path, root: Path, digest_field: str) -> dict[str, Any]:
    absolute = path.absolute()
    if (".." in absolute.parts or not absolute.is_relative_to(root)
            or any(item.is_symlink() for item in (absolute, *absolute.parents))
            or not absolute.is_file()):
        raise ValueError("preparation_outcome_evidence_unavailable")
    value = json.loads(absolute.read_text())
    if (not isinstance(value, dict) or not _DIGEST.fullmatch(str(value.get(digest_field, "")))
            or value[digest_field] != canonical_digest(value, digest_field=digest_field)):
        raise ValueError("preparation_outcome_evidence_invalid")
    return value


def _robot_fit_hold(root: Path, handoff: dict[str, Any], task_context_digest: str) -> dict[str, Any] | None:
    """A verified robot fit hold (``website_robot_fit_precheck``), as status v6 reads it; else None.

    v3 to v5 readers never see it: their outcome vocabulary is closed (the WebApp parses it strictly),
    and a hold is not the carry refusal or the missing evidence those name.
    """
    from .website_robot_fit_precheck_projection import SCHEMA
    from .website_task_context_contracts import retains_robot_fit_hold

    if not retains_robot_fit_hold(handoff, {"capture_id": root.name, "scene_id": root.parent.parent.name}):
        return None
    reference = handoff["robot_fit_precheck"]
    preparation = _load(Path(handoff["preparation_path"]), root, "digest")
    precheck = _load(Path(reference["path"]), root, "precheck_digest")
    if (preparation.get("binding", {}).get("task_context_digest") != task_context_digest
            or handoff.get("preparation_digest") != preparation["digest"]
            or precheck.get("schema_version") != SCHEMA or precheck.get("preparation_digest") != preparation["digest"]
            or precheck["precheck_digest"] != reference["digest"] or precheck.get("status") != "no_fit"
            or precheck.get("hold_reason") != reference["hold_reason"]
            or precheck.get("physical_world_truth_claimed") is not False):
        raise ValueError("preparation_outcome_robot_fit_binding_invalid")
    return {"category": "robot_incompatibility", "reason": "robot_fit_none",
            "basis": "conservative_pre_spend_fit_check", "robot_id": None,
            "evidence_digest": precheck["precheck_digest"]}


def read_preparation_outcome(*, capture_root: Path, task_context_digest: str,
                             state: str, robot_fit_hold: bool = False, recapture: bool = False) -> dict[str, Any] | None:
    """Only a current stopped attempt may project a verified retained finding.

    ``robot_fit_hold`` (status v6) also projects a job held before spend because no robot the
    screening catalog runs fits its task (``_robot_fit_hold``). ``recapture`` (status v7) projects a
    preparation that ended on a stop only a new video can clear (``website_recapture_required``): its
    reason is the stop's own code, and the site is asked to film again.
    """
    if state not in {"awaiting_inputs", "failed_retryable", "failed_terminal"}:
        return None
    root = capture_root.absolute()
    path = root / "pipeline" / "website_scene_preparation" / "handoff.json"
    if not path.exists():
        return None
    handoff = _load(path, root, "digest")
    if (handoff.get("schema_version") != "website_scene_handoff.v1"
            or handoff.get("capture_id") != root.name or handoff.get("scene_id") != root.parent.parent.name):
        raise ValueError("preparation_outcome_capture_mismatch")
    from .website_target_background_correction import retains_rejected_selection
    from .observed_room_contracts import retains_observed_room_limit
    context = {"capture_id": root.name, "scene_id": root.parent.parent.name,
               "context_digest": task_context_digest}
    if retains_rejected_selection(handoff, context):
        # The site said none of the objects we found is the one: the video does not show it, so ask for one
        # that does (part 6), never a review nobody makes.
        from .website_recapture_required import NOT_OBSERVED_CODE
        return {"category": "insufficient_evidence",
                "reason": sent_recapture_reason(NOT_OBSERVED_CODE) if recapture else "task_evidence_incomplete",
                "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
    if retains_observed_room_limit(handoff, context):
        # A hold written before the observed-room review became final: the Atlas lane reviews its task once
        # more (``website_atlas_preview._reopen_observed_room``) while its window and grant are live.
        return {"category": "insufficient_evidence", "reason": "task_evidence_incomplete",
                "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
    if recapture:
        from .website_recapture_required import retains_recapture_required
        if retains_recapture_required(handoff, context):
            return {"category": "insufficient_evidence", "reason": sent_recapture_reason(handoff["recapture_required"]["code"]),
                    "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
    if robot_fit_hold:
        held = _robot_fit_hold(root, handoff, task_context_digest)
        if held is not None:
            return held
    ref = handoff.get("capability_verdict")
    if isinstance(ref, dict) and ref.get("status") == "abstained":
        from .robot_arm_path_clearance import ROBOT_PROFILE_ARMS

        preparation = _load(Path(handoff["preparation_path"]), root, "digest")
        verdict = _load(Path(ref["path"]), root, "verdict_digest")
        runtime_ref = handoff.get("runtime_inputs") or {}
        runtime = _load(Path(runtime_ref["path"]), root, "digest")
        if (preparation.get("schema_version") != "website_scene_preparation.v1"
                or preparation.get("binding", {}).get("task_context_digest") != task_context_digest
                or handoff.get("preparation_digest") != preparation["digest"]
                or runtime.get("schema_version") != "website_scene_runtime_inputs.v1"
                or runtime.get("digest") != runtime_ref.get("digest")
                or runtime.get("preparation_digest") != preparation["digest"]
                or runtime.get("collision", {}).get("digest") != verdict.get("collision_sha256")
                or verdict.get("schema_version") != "website_rigid_task_capability_verdict.v1"
                or verdict.get("preparation_digest") != preparation["digest"]
                or verdict.get("verdict_digest") != ref.get("digest")
                or verdict.get("status") != "abstained"
                or verdict.get("robot_id") not in set(ROBOT_PROFILE_ARMS.values())
                or verdict.get("claim_ceiling") != "development_only"
                or verdict.get("physical_world_truth_claimed") is not False):
            raise ValueError("preparation_outcome_verdict_binding_invalid")
        blockers = verdict.get("blockers")
        if (isinstance(blockers, list) and blockers
                and all(isinstance(code, str) and _PATH_REFUSAL.fullmatch(code) for code in blockers)):
            return {"category": "robot_incompatibility", "reason": "robot_carry_path_rejected",
                    "basis": "conservative_configuration_check", "robot_id": verdict["robot_id"],
                    "evidence_digest": verdict["verdict_digest"]}
        # Failure to find a station is missing evidence, not a reach verdict.
        if isinstance(blockers, list) and blockers and all(isinstance(code, str) and code.startswith("robot_station_")
                                                         for code in blockers):
            return {"category": "insufficient_evidence", "reason": "task_evidence_incomplete",
                    "basis": "missing_evidence", "robot_id": None, "evidence_digest": verdict["verdict_digest"]}
        return None
    blockers = handoff.get("blockers")
    if isinstance(blockers, list) and blockers and all(code in _MISSING for code in blockers):
        preparation = _load(Path(handoff["preparation_path"]), root, "digest")
        if (preparation.get("schema_version") != "website_scene_preparation.v1"
                or preparation.get("binding", {}).get("task_context_digest") != task_context_digest
                or handoff.get("preparation_digest") != preparation["digest"]):
            raise ValueError("preparation_outcome_context_mismatch")
        if recapture and website_lists_endings() and len(blockers) == 1:
            # What is true: this release cannot prepare the job, and why (part 6).
            return {"category": "insufficient_evidence", "reason": NOT_POSSIBLE, "cause": blockers[0],
                    "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
        return {"category": "insufficient_evidence", "reason": "task_evidence_incomplete",
                "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
    return None
