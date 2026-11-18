"""Fresh real-frames-only attempts at Atlas's refined splat, so a job's look is not left as the draft.

When every refined-splat form failed (``atlas_native_splat.run_native_splat``: up to three operations
on outages and one resubmit of ``EXECUTION_FAILED`` per form), the job used to take the depth-fused
draft as its look at once. Now it waits and tries again: a new attempt root that sends the documented
real-frames-only form (``documented_minimal``) under its own intent, reservation and idempotency key.
``job_retry_policy`` times the attempts (15 min, 1 h, 3 h, 6 h, 12 h, 24 h, then daily while the
window stays open). Only these make the draft final, flagged on the job's result:

- the job's remaining allowance cannot cover another splat and the mesh after it, read when a
  failure is recorded and again before an attempt is bought (``atlas_refined_splat_retry_unaffordable``);
- a money wall refuses an attempt, or the allowance read, with nothing in flight
  (``atlas_refined_splat_retry_money_refused``): the job has a usable draft, so it takes it instead of
  waiting, and nothing more is bought for the look (the mesh after it has its own admission);
- the job's deadline ends within ``LAST_CALL_SECONDS`` of a failure, or within ``START_BY_SECONDS`` when
  an attempt is due (``atlas_refined_splat_deadline_closing``). The deadline is the job's window, read
  live (``website_preparation_terminal.retained_window_end``), and never later than its Atlas job's own
  life (``MAXIMUM_JOB_SECONDS`` from enqueue), which ends the job as ``atlas_preview_expired``; either
  alone when only one can be read. The last attempt is set at the start of that last stretch;
- a job with neither gets the six timed retries alone (``atlas_refined_splat_retries_exhausted``).

Nothing is bought twice: an attempt root that has started is always resumed from its own receipts
(polled, adopted or reconciled by the task runner), and a new attempt is recorded only once every
earlier root is proven closed with no live operation (``_closed``). A wall (money, privacy, rights)
raised by an attempt's own admission is never bypassed: privacy and rights are raised; money takes the draft
when nothing is in flight, else it is raised.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .job_retry_policy import MONEY, REFUSAL_RETRY_DELAYS_SECONDS, hard_stop_kind, refusal_retry_delay

RECORD = "refined_splat_retry.json"
SCHEMA = "atlas_refined_splat_retry.v1"
#: The stage a job waits at, with ``next_attempt_at_epoch`` the time its next attempt is due.
WAIT_STAGE = "refined_splat_retry_wait"
#: The draft is final once the job's deadline is within this of a failure
#: (``website_scene_handoff.LAST_CALL_SECONDS``); the last attempt is set no later than its start.
LAST_CALL_SECONDS = 2 * 3600
#: A due attempt is not started this close to the deadline: a splat and the mesh after it, with tick slack.
START_BY_SECONDS = 90 * 60
DRAFT_FINAL = "draft_final"
UNAFFORDABLE = "atlas_refined_splat_retry_unaffordable"
DEADLINE_CLOSING = "atlas_refined_splat_deadline_closing"
EXHAUSTED = "atlas_refined_splat_retries_exhausted"
MONEY_REFUSED = "atlas_refined_splat_retry_money_refused"
FAILED = "atlas_task_operation_failed"


def attempt_root(output_root: Path, attempt: int) -> Path:
    """The base root of fresh attempt ``attempt``; its real-frames-only form is ``<base>-minimal``."""
    return Path(output_root) / f"refined_splat-retry-{attempt}"


def _closed(base: Path) -> bool:
    """Every form root under ``base`` is unstarted, or every operation it sent failed: nothing can still finish."""
    from .atlas_native_splat import _form_roots, _provider_failed_and_exhausted, _started
    return all(not _started(root) or _provider_failed_and_exhausted(root / "native") for _, root in _form_roots(base))


def _started(base: Path) -> bool:
    from .atlas_native_splat import _form_roots, _started as started
    return any(started(root) for _, root in _form_roots(base))


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("atlas_refined_splat_retry_record_invalid")
    if not path.is_file():
        return {"schema_version": SCHEMA, "attempts": [], "final": None}
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA or not isinstance(value.get("attempts"), list):
        raise ValueError("atlas_refined_splat_retry_record_invalid")
    return value


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _summary(record: Mapping[str, Any]) -> dict[str, Any]:
    final = record.get("final") or {}
    return {"attempts": len(record["attempts"]), "final_reason": final.get("reason"),
            "deadline_epoch": final.get("deadline_epoch"), "remaining_cost_usd": final.get("remaining_cost_usd"),
            **({"refusal": final["refusal"]} if final.get("refusal") else {})}


def _money_refusal(error: BaseException) -> str | None:
    """The code of a money wall (by its type, or by its code's words when untyped); None for anything else.

    A privacy or rights wall is never money here: it stays a stop.
    """
    from .hard_stops import wall_kind
    kind = wall_kind(error)
    if kind is None and isinstance(error, ValueError):
        kind = hard_stop_kind(str(error).partition(":")[0])
    return (str(error).partition(":")[0][:120] or type(error).__name__) if kind == MONEY else None


def next_refined_splat(*, output_root: Path, original_root: Path, run_attempt: Callable[[Path], Mapping[str, Any]],
                       remaining_cost_usd: Callable[[], Any], needed_usd: float,
                       deadline: Callable[[], float | None], now: float) -> dict[str, Any]:
    """After the refined splat failed: the next attempt's result, a timed wait, or the draft made final.

    ``original_root`` is the job's first refined-splat root (its forms failed); ``run_attempt(base)``
    resumes or starts the real-frames-only form under ``base``. ``deadline()`` is the latest the job can
    finish (``website_atlas_pipeline._job_deadline``: its window, never past its Atlas job's own life), or
    None when neither can be read. Returns the attempt's own result (``completed`` or ``processing``),
    ``{"status": "processing", "stage": WAIT_STAGE, "next_attempt_at_epoch": ...}``, or
    ``{"status": DRAFT_FINAL, "retry": ...}``.
    """
    path = Path(output_root) / RECORD
    record = _read(path)
    if record.get("final"):
        return {"status": DRAFT_FINAL, "retry": _summary(record)}

    def final(reason: str, *, end: float | None = None, remaining: Any = None, refusal: str | None = None):
        record["final"] = {"reason": reason, "at_epoch": now, "deadline_epoch": end,
                           "remaining_cost_usd": _number(remaining), **({"refusal": refusal} if refusal else {})}
        write_json(path, record)
        return {"status": DRAFT_FINAL, "retry": _summary(record)}

    def covered(remaining: Any) -> bool:
        return _number(remaining) is not None and remaining >= needed_usd

    def allowance(base: Path | None):
        """The remaining allowance; a money refusal reading it takes the draft when nothing is in flight."""
        try:
            return remaining_cost_usd(), None
        except Exception as exc:  # noqa: BLE001 - only a money wall is taken here; anything else is raised
            refusal = _money_refusal(exc)
            if refusal is None or (base is not None and not _closed(base)):
                raise
            return None, refusal

    attempts = record["attempts"]
    current = attempts[-1] if attempts else None
    if current is not None and current.get("failed_at_epoch") is None:
        base = Path(output_root) / current["root"]
        if not _started(base):
            # Nothing of this attempt was sent yet: it starts only when due, early enough to finish, and inside
            # the allowance.
            if now < current["not_before_epoch"]:
                return {"status": "processing", "stage": WAIT_STAGE, "next_attempt_at_epoch": current["not_before_epoch"],
                        "attempt": current["attempt"]}
            end = deadline()
            if end is not None and now >= end - START_BY_SECONDS:
                return final(DEADLINE_CLOSING, end=end)
            remaining, refusal = allowance(base)
            if refusal is not None:
                return final(MONEY_REFUSED, end=end, refusal=refusal)
            if not covered(remaining):
                return final(UNAFFORDABLE, end=end, remaining=remaining)
        try:
            value = dict(run_attempt(base))
        except Exception as exc:  # noqa: BLE001 - each outcome below is decided, anything else raised
            # A money wall with nothing in flight takes the draft: nothing more is bought for the look. A
            # provider failure of the whole attempt, proven closed, is recorded. Anything else (a privacy or
            # rights wall, a live operation) is raised to the job's own retry policy.
            refusal = _money_refusal(exc)
            if refusal is not None and _closed(base):
                return final(MONEY_REFUSED, end=deadline(), refusal=refusal)
            if not isinstance(exc, ValueError) or str(exc) != FAILED or not _closed(base):
                raise
            current["failed_at_epoch"] = now
            write_json(path, record)
        else:
            return value
    elif current is None and not _closed(Path(original_root)):
        # An operation of the first attempt may still finish: it is resumed there, never bought again.
        raise ValueError(FAILED)
    end = deadline()
    remaining, refusal = allowance(None)
    if refusal is not None:
        return final(MONEY_REFUSED, end=end, refusal=refusal)
    if not covered(remaining):
        return final(UNAFFORDABLE, end=end, remaining=remaining)
    if end is not None and now >= end - LAST_CALL_SECONDS:
        return final(DEADLINE_CLOSING, end=end, remaining=remaining)
    delay = refusal_retry_delay(len(attempts))
    if delay is None:
        if end is None:
            return final(EXHAUSTED, remaining=remaining)
        delay = REFUSAL_RETRY_DELAYS_SECONDS[-1]  # The deadline is ahead: daily, as the schedule's last step.
    due = now + delay if end is None else min(now + delay, end - LAST_CALL_SECONDS)
    number = len(attempts) + 1
    attempts.append({"attempt": number, "root": attempt_root(Path("."), number).name, "scheduled_at_epoch": now,
                     "not_before_epoch": due, "deadline_epoch": end})
    write_json(path, record)
    return {"status": "processing", "stage": WAIT_STAGE, "next_attempt_at_epoch": due, "attempt": number}
