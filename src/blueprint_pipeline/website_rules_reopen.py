"""Reopen a website job that ended on a hold the current rules no longer stop on.

Owner direction 2026-11-17: accept every job, then determine what is needed. Two changes made that true for jobs
still to run: Gemini's credits-depleted answer is a money wait, never a non-answer (#3636), and a preparation
blocker that is no wall ships the room package, flagged (#3637, with the holds of 2026-11-16 before it). A job that
ended before them on a hold they would not stop on stays as it ended: a completed stage is never run again, and its
hold names no fact a later run could bring (``website_held_handoff_replay``). This adds that fact: the rules
revision (``REVISION``), once per job (``REOPENED_FILE``).

What a reopen does (``WHAT_HAPPENS``): the preparation runs again under the current rules. It may ship the room
package, or build the job's object, which buys the object's authoring within the job's grant and its spend limits.

The reopen is the existing resume path: the held-replay fact ``rules_reopen`` (``rules_reopen``), which the door's
``selected-handoff --resume-preparation`` (``resume_held_preparation``) and, for a ledger that kept its bytes, the
drain tick (``reconcile_held_handoffs``) run once through the same claim, on the current release, from the job's
retained outputs. It never buys twice: the Atlas result, every retained Gemini, SAM and authoring receipt are read
as they stand, and a job whose prepared scene may already have gone to the website is not reopened.

Never reopened (``reopen_verdict`` names why): a withdrawn capture, a job with no live grant (expired or none), a
stop that is a wall (privacy, rights, money, the owner's closing, the job's sentence still to come), the site's
unconfirmed brief (its own consent replay), a recapture hold (its own replay), a job whose ledger did not complete
(a failing job retries by itself; an ended authority stays ended), and a hold the current rules would keep.
Read-only; it decides, it never runs anything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

#: The rules a reopened job goes on under: #3636 (Gemini's credits out is a money wait) and #3637 (a blocker that is no
#: wall ships the room package), with the holds the room package replaced on 2026-11-16. A later revision reopens again.
REVISION = "website_accept_every_job.2026-11-17"
KIND = "rules_reopen"
WHAT_HAPPENS = ("The preparation runs again on the current release from its retained outputs. It may ship the room "
                "package, or build the job's object, which buys the object's authoring within the job's grant.")
#: Kept beside the handoff when a reopen is started (``mark_reopened``): this job reopens once per rules revision.
REOPENED_FILE = "rules_reopen.json"
#: The handoff holds an earlier release retained that the current handoff ships as the room package: no task profile,
#: a reading naming no family, the mechanism mismatch, several objects, a soft object.
REOPENED_GAPS = frozenset({"website_native_task_profile_preparation_required",
                           "website_task_operation_coverage_unresolved", "website_task_profile_mechanism_mismatch",
                           "website_multi_object_task_preparation_required",
                           "website_deformable_task_object_preparation_required"})
LEDGER = "pipeline_job_ledger.json"


def _verdict(reopen: bool, reason: str, **extra: Any) -> dict[str, Any]:
    return {"reopen": reopen, "reason": reason, **extra}


def reopen_verdict(pipeline_root: Path, *, now: float | None = None) -> dict[str, Any]:
    """Whether the held job under ``pipeline_root`` reopens under the current rules, and why. Never raises."""
    try:
        return _reopen_verdict(Path(pipeline_root), now=time.time() if now is None else now)
    except Exception as exc:  # noqa: BLE001 - an unreadable record reopens nothing
        return _verdict(False, "unreadable:" + type(exc).__name__)


def _reopen_verdict(root: Path, *, now: float) -> dict[str, Any]:
    from .job_retry_policy import MONEY, PRIVACY_RIGHTS
    from .website_atlas_preview_store import capture_withdrawn
    from .website_held_handoff_replay import CONSENT_HOLDS, _grant_live, _handoff, _retained_context
    from .website_preparation_retry import AWAITING_JOB_SENTENCE, OWNER_CLOSED, classify
    capture = root.parent
    if capture_withdrawn(capture):
        return _verdict(False, "withdrawn")
    try:
        ledger = json.loads((capture / LEDGER).read_text())
    except (OSError, ValueError):
        ledger = None
    status = ledger.get("status") if isinstance(ledger, dict) else None
    if status != "completed":
        return _verdict(False, "ledger_" + str(status or "missing"))
    if reopened(root, ledger_revision=ledger.get("revision")):
        return _verdict(False, "already_reopened_under_these_rules")
    handoff, context = _handoff(root), _retained_context(root)
    if handoff is None or context is None:
        return _verdict(False, "no_retained_handoff")
    if not _grant_live(root):
        return _verdict(False, "grant_expired_or_missing")
    from .website_scene_handoff import goes_on, prepared_scene_may_exist
    if handoff.get("website_intake_outbox") or handoff.get("navigation_intake") or prepared_scene_may_exist(
            capture, task_context_digest=context.get("context_digest")):
        return _verdict(False, "intake_may_have_been_sent")  # it may be in paid preparation: never bought twice
    blockers = [str(code) for code in handoff.get("blockers") or []]
    if handoff.get("status") not in {"needs_input", "awaiting_inputs"} or not blockers:
        return _verdict(False, "not_stopped")
    walls = sorted({kind for kind in (classify(code) for code in blockers)
                    if kind in {PRIVACY_RIGHTS, MONEY, OWNER_CLOSED, AWAITING_JOB_SENTENCE}})
    if walls:
        return _verdict(False, "wall:" + ",".join(walls), stopped_on=blockers)
    if set(blockers) & CONSENT_HOLDS:
        return _verdict(False, "consent_hold", stopped_on=blockers)
    if handoff.get("recapture_required"):
        return _verdict(False, "recapture_hold", stopped_on=blockers)
    gap = handoff.get("native_task_preparation_gap")
    if isinstance(gap, Mapping) and gap.get("code") in REOPENED_GAPS:
        from .website_task_context_contracts import holds_native_task_preparation
        from .website_task_context_lineage import binding_contexts
        if any(holds_native_task_preparation(handoff, bound) for bound in binding_contexts(root, context)):
            return _verdict(True, "held_gap_reopens_may_build_object_and_buy_authoring", stopped_on=blockers,
                            what_happens=WHAT_HAPPENS)
        return _verdict(False, "hold_unverified", stopped_on=blockers)
    from .website_preparation_terminal import retained_window_end

    def window_end() -> float | None:
        return retained_window_end(capture, scene_id=str(context.get("scene_id")),
                                   capture_id=str(context.get("capture_id")))
    if goes_on(blockers, task_context=context, now=now, window_end=window_end):
        return _verdict(True, "blocker_reopens_may_build_object_and_buy_authoring", stopped_on=blockers,
                        what_happens=WHAT_HAPPENS)
    return _verdict(False, "current_rules_still_wait", stopped_on=blockers)


def reopened(pipeline_root: Path, *, ledger_revision: Any) -> bool:
    """Whether a reopen under ``REVISION`` already ran for this job: marked, and claimed (the ledger moved past the
    revision it was marked at). A mark whose run never claimed the job (a crash, a refused claim) leaves it eligible."""
    try:
        mark = json.loads((Path(pipeline_root) / "website_scene_preparation" / REOPENED_FILE).read_text())
    except (OSError, ValueError):
        return False
    return (isinstance(mark, dict) and mark.get("rules_revision") == REVISION
            and mark.get("ledger_revision") != ledger_revision)


def mark_reopened(pipeline_root: Path, *, fact: Mapping[str, Any], ledger_revision: Any) -> None:
    """Kept before the reopen runs, beside the held-replay state, so the job reopens once per rules revision."""
    if fact.get("kind") != KIND:
        return
    from .common import write_json
    write_json(Path(pipeline_root) / "website_scene_preparation" / REOPENED_FILE,
               {"schema_version": "website_rules_reopen.v1", "rules_revision": REVISION,
                "fact_digest": fact.get("fact_digest"), "ledger_revision": ledger_revision,
                "reopened_at_epoch": time.time()})


def rules_reopen(pipeline_root: Path, *, now: float | None = None) -> dict[str, Any] | None:
    """The held-replay fact (no digest yet: ``website_held_handoff_replay.arrived_fact`` adds it), or None.

    Keyed by the job and the rules revision only, so whatever the reopened run holds on next, it never reopens again
    under these rules (``reopened``).
    """
    verdict = reopen_verdict(pipeline_root, now=now)
    if not verdict["reopen"]:
        return None
    from .website_held_handoff_replay import _retained_context
    return {"kind": KIND, "rules_revision": REVISION, "reason": verdict["reason"],
            "what_happens": WHAT_HAPPENS,
            "task_context_digest": (_retained_context(Path(pipeline_root)) or {}).get("context_digest")}


def survey(storage_root: Path, *, now: float | None = None) -> list[dict[str, Any]]:
    """Every website capture under ``storage_root`` with its verdict: ids and codes only. Read-only."""
    rows = []
    for ledger in sorted(Path(storage_root).glob(f"*/scenes/site-*/captures/*/{LEDGER}")):
        capture = ledger.parent
        if not capture.name.startswith(("walkthrough-", "supplement-")):
            continue
        verdict = reopen_verdict(capture / "pipeline", now=now)
        rows.append({"scene_id": capture.parent.parent.name, "capture_id": capture.name, **verdict})
    return rows


def main(argv: list[str] | None = None) -> int:
    """``python -m blueprint_pipeline.website_rules_reopen <storage root>``: which jobs a reopen would run, and why."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("storage_root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps({"rules_revision": REVISION, "captures": survey(args.storage_root)}, indent=2, sort_keys=True))
    return 0


__all__ = ["KIND", "REOPENED_FILE", "REOPENED_GAPS", "REVISION", "WHAT_HAPPENS", "mark_reopened", "reopen_verdict",
           "reopened", "rules_reopen", "survey"]


if __name__ == "__main__":
    raise SystemExit(main())
