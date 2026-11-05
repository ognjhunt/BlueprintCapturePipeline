"""Keep the review videos of screening runs that settled before their videos were kept.

Retention (``beta_screening_review_videos.retain``) runs when a run is reconciled,
so a run that settled before it existed has no reference to kept videos, although
its promoted output archive may still hold the episodes' MP4s. Every run that used
a machine counts, whatever its verdict, so the site can be shown any of them later:
a completed row (its summary holds the reference) and a blocked row whose
reservation was committed (the row holds it). The screening worker gives each
such row one retention pass, a bounded number per tick:

- Retention reads only our own object store (or the attempt's local ZIP), after
  teardown: no provider, no paid compute. It can take minutes, so it runs
  outside the coordinator's ledger lock.
- The reference is then recorded under the lock, and only if the row is still
  the same attempt in the same state with no reference. A completed run's kept
  videos move the campaign's revision, so its publication carries them; the
  WebApp accepts a later revision of a finished campaign that only adds a
  candidate's videos. A blocked run's are kept beside the capture, unpublished.
- A failure is recorded once as ``not_retained`` with its blockers and is never
  retried. It never changes a result, a settlement or a cost.
- A withdrawn capture is never copied: retention resolves the capture through
  the same refusal every paid attempt uses.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

#: At most this many rows get a retention pass per worker tick.
PER_TICK = 1
#: The attempt's records needed to find its archive are gone or unreadable.
ATTEMPT_UNAVAILABLE = "beta_screening_review_videos_attempt_unavailable"


def _holder(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Where this row's reference lives once kept: a completed run's summary, a spent blocked run itself."""
    attempt = row.get("attempt")
    if not isinstance(attempt, Mapping) or not isinstance(attempt.get("attempt_id"), str):
        return None
    if row.get("status") == "completed":
        return row["summary"] if isinstance(row.get("summary"), Mapping) else None
    if row.get("status") == "blocked" and attempt.get("reservation_status") == "committed":
        return row  # committed only once its machine was allocated and closed out
    return None


def _due(ledger: Mapping[str, Any], maximum: int) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Rows of runs that used a machine and have no ``review_videos`` yet, oldest campaign id first."""
    due = []
    for campaign_id, state in sorted(ledger["campaigns"].items()):
        for candidate_id, row in sorted(state["candidates"].items()):
            holder = _holder(row)
            if holder is not None and "review_videos" not in holder:
                due.append((campaign_id, state["plan_digest"], candidate_id, dict(row["attempt"])))
    return due[:maximum]


def backfill_review_videos(*, queue_root: str | Path | None = None, maximum: int = PER_TICK,
                           retain: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """One retention pass for up to ``maximum`` rows of runs that used a machine and have none; never raises."""
    from . import beta_policy_screening_coordinator as coordinator
    from .beta_screening_review_videos import not_retained, valid_reference

    if retain is None:
        from .beta_screening_review_videos import retain
    try:
        root = coordinator._root(queue_root)
        with coordinator._lock(root):
            ledger = coordinator._ledger(root)
            due = [(*item, ledger["campaigns"][item[0]]["candidates"][item[2]]["status"])
                   for item in _due(ledger, maximum)]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        from .bounded_error import error_description

        return {"status": "failed", "error": error_description(exc), "attempts": []}
    results = []
    for campaign_id, plan_digest, candidate_id, attempt, row_status in due:
        try:
            plan = coordinator._plan(root, campaign_id, plan_digest)
            candidate = next(item for item in plan["candidates"] if item["candidate_id"] == candidate_id)
            attempt_dir = coordinator._attempt_path(root, attempt)
            adapter = coordinator._read(attempt_dir / "allocator_result.json")
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            reference = not_retained(ATTEMPT_UNAVAILABLE)
        else:
            reference = retain(plan=plan, candidate=candidate, prepared=attempt.get("prepared") or {},
                               adapter=adapter, attempt_dir=attempt_dir)
        reference = valid_reference(reference) or not_retained("beta_screening_review_videos_failed")
        outcome = {"campaign_id": campaign_id, "candidate_id": candidate_id, "attempt_id": attempt["attempt_id"],
                   "row_status": row_status, "status": reference["status"]}
        try:
            with coordinator._lock(root):
                ledger = coordinator._ledger(root)
                state = ledger["campaigns"].get(campaign_id) or {}
                row = (state.get("candidates") or {}).get(candidate_id) or {}
                holder = _holder(row)
                if (holder is None or "review_videos" in holder or row["attempt"]["attempt_id"] != attempt["attempt_id"]
                        or row["status"] != outcome["row_status"]):
                    outcome["status"] = "changed"
                elif row["status"] == "completed":
                    row["summary"] = {**row["summary"], "review_videos": reference}
                    if reference["status"] != "not_retained":
                        coordinator._changed(state)  # the publication now lists this candidate's videos
                    coordinator._save(root, ledger)
                else:
                    row["review_videos"] = reference  # kept for the site, never published
                    coordinator._save(root, ledger)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            from .bounded_error import error_description

            outcome.update(status="failed", error=error_description(exc))
        results.append(outcome)
    return {"status": "processed", "attempts": results}


__all__ = ["backfill_review_videos", "PER_TICK", "ATTEMPT_UNAVAILABLE"]
