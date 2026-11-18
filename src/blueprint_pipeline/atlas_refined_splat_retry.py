"""Fresh real-frames-only attempts at Atlas's refined splat, so a job's look is not left as the draft.

When every refined-splat form failed (``atlas_native_splat.run_native_splat``: up to three operations
on outages and one resubmit of ``EXECUTION_FAILED`` per form), the job used to take the depth-fused
draft as its look at once. Now it waits and tries again: a new attempt root that resends the first
attempt's request under its own intent, reservation and idempotency key: the reviewed-whole form with
the same generated views its retained review admitted (``atlas_look_views``), when it had any, then
the documented real-frames-only form (``documented_minimal``).
``job_retry_policy`` times the attempts (15 min, 1 h, 3 h, 6 h, 12 h, 24 h, then daily while the
window stays open). Only these make the draft final, flagged on the job's result:

- the job's remaining allowance cannot cover another splat and the mesh after it, read when a
  failure is recorded and again before an attempt is bought (``atlas_refined_splat_retry_unaffordable``);
- the job's window, read live (``website_preparation_terminal.retained_window_end``), ends within
  ``LAST_CALL_SECONDS`` of a failure, or has ended when an attempt is due
  (``atlas_refined_splat_window_closing``); the last attempt is set no later than the start of that
  last stretch, so one is tried there;
- a job whose window cannot be read gets the six timed retries alone
  (``atlas_refined_splat_retries_exhausted``).

Nothing is bought twice: an attempt root that has started is always resumed from its own receipts
(polled, adopted or reconciled by the task runner), and a new attempt is recorded only once every
earlier root is proven closed with no live operation (``_closed``). A wall (money, privacy, rights)
raised by an attempt's own admission stays a stop: it is raised, never recorded as a failure.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS, refusal_retry_delay

RECORD = "refined_splat_retry.json"
SCHEMA = "atlas_refined_splat_retry.v1"
#: The stage a job waits at, with ``next_attempt_at_epoch`` the time its next attempt is due.
WAIT_STAGE = "refined_splat_retry_wait"
#: The draft is final once the window ends within this of a failure (``website_scene_handoff.LAST_CALL_SECONDS``).
LAST_CALL_SECONDS = 2 * 3600
DRAFT_FINAL = "draft_final"
UNAFFORDABLE = "atlas_refined_splat_retry_unaffordable"
WINDOW_CLOSING = "atlas_refined_splat_window_closing"
EXHAUSTED = "atlas_refined_splat_retries_exhausted"
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
            "window_end_epoch": final.get("window_end_epoch"), "remaining_cost_usd": final.get("remaining_cost_usd")}


def next_refined_splat(*, output_root: Path, original_root: Path, run_attempt: Callable[[Path], Mapping[str, Any]],
                       remaining_cost_usd: Callable[[], Any], needed_usd: float,
                       window_end: Callable[[], float | None], now: float) -> dict[str, Any]:
    """After the refined splat failed: the next attempt's result, a timed wait, or the draft made final.

    ``original_root`` is the job's first refined-splat root (its forms failed); ``run_attempt(base)``
    resumes or starts the real-frames-only form under ``base``. Returns the attempt's own result
    (``completed`` or ``processing``), ``{"status": "processing", "stage": WAIT_STAGE,
    "next_attempt_at_epoch": ...}``, or ``{"status": DRAFT_FINAL, "retry": ...}``.
    """
    path = Path(output_root) / RECORD
    record = _read(path)
    if record.get("final"):
        return {"status": DRAFT_FINAL, "retry": _summary(record)}

    def final(reason: str, *, end: float | None = None, remaining: Any = None) -> dict[str, Any]:
        record["final"] = {"reason": reason, "at_epoch": now, "window_end_epoch": end,
                           "remaining_cost_usd": _number(remaining)}
        write_json(path, record)
        return {"status": DRAFT_FINAL, "retry": _summary(record)}

    def covered(remaining: Any) -> bool:
        return _number(remaining) is not None and remaining >= needed_usd

    attempts = record["attempts"]
    current = attempts[-1] if attempts else None
    if current is not None and current.get("failed_at_epoch") is None:
        base = Path(output_root) / current["root"]
        started = _started(base)
        if not started:
            # Nothing of this attempt was sent yet: it starts only when due, inside the window and the allowance.
            if now < current["not_before_epoch"]:
                return {"status": "processing", "stage": WAIT_STAGE, "next_attempt_at_epoch": current["not_before_epoch"],
                        "attempt": current["attempt"]}
            end = window_end()
            if end is not None and now >= end:
                return final(WINDOW_CLOSING, end=end)
            remaining = remaining_cost_usd()
            if not covered(remaining):
                return final(UNAFFORDABLE, end=end, remaining=remaining)
        try:
            value = dict(run_attempt(base))
        except ValueError as exc:
            # Only a provider failure of the whole attempt, proven closed, is recorded; anything else, a
            # wall among them, is raised and handled by the job's own retry policy.
            if str(exc) != FAILED or not _closed(base):
                raise
            current["failed_at_epoch"] = now
            write_json(path, record)
        else:
            return value
    elif current is None and not _closed(Path(original_root)):
        # An operation of the first attempt may still finish: it is resumed there, never bought again.
        raise ValueError(FAILED)
    remaining = remaining_cost_usd()
    end = window_end()
    if not covered(remaining):
        return final(UNAFFORDABLE, end=end, remaining=remaining)
    if end is not None and now >= end - LAST_CALL_SECONDS:
        return final(WINDOW_CLOSING, end=end, remaining=remaining)
    delay = refusal_retry_delay(len(attempts))
    if delay is None:
        if end is None:
            return final(EXHAUSTED, remaining=remaining)
        delay = REFUSAL_RETRY_DELAYS_SECONDS[-1]  # The window is open: daily, as the schedule's last step.
    due = now + delay if end is None else min(now + delay, end - LAST_CALL_SECONDS)
    number = len(attempts) + 1
    attempts.append({"attempt": number, "root": attempt_root(Path("."), number).name, "scheduled_at_epoch": now,
                     "not_before_epoch": due, "window_end_epoch": end})
    write_json(path, record)
    return {"status": "processing", "stage": WAIT_STAGE, "next_attempt_at_epoch": due, "attempt": number}
