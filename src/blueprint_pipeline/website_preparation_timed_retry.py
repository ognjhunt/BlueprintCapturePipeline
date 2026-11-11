"""Run a website preparation again on a timer once the broker has stopped delivering it.

Beta journey step: L3 website scene preparation. Pub/Sub redelivers a failed handoff up to five
times and then dead-letters it; after that nothing moved the preparation but an operator or a deploy,
and it sat until its grant expired. The job retry rule (``job_retry_policy``,
``docs/architecture/job-retry-and-deadlines.md``) says every retry is timed. So the job ledger keeps
the exact handoff bytes of a website browser delivery from its claim on (``FIELD``, inside the capture
root, so retirement and withdrawal delete them with the ledger), and the listener's existing drain tick
runs them again (``website_preparation_timed_retry_drain.reconcile_timed_retries``) through the same ``process_handoff_payload`` a delivery
takes: the same owner observation, retirement, consent, context and claim checks, and every paid
admission asked again by the step itself. Only a ledger nothing else will move is driven:

- a privacy or rights refusal never (``job_retry_policy.hard_stop_kind``);
- a money refusal is asked again every ``MONEY_RETRY_SECONDS``, not counted;
- a control-plane capacity shortage, and an attempt that stopped before its claim (the WebApp or
  storage unreachable), back off as an outage (``outage_retry_delay``), not counted;
- anything else, a dead worker included, gets the six timed retries of ``refusal_retry_delay``,
  counted from the ledger's own history since the job's last progress, and at most
  ``MAXIMUM_TIMED_RETRIES`` in the ledger's life.

Each timed attempt is one history row (``STARTED_STATUS``, marked ``MARK``) written by its own claim,
before anything runs, so a process killed mid-attempt still counts. An attempt that changed nothing
(it stopped before the claim) leaves only a stamp beside the ledger (``STATE_FILE``), never bytes.
Nothing starts once the retained grant's own expiry has passed; the terminal sweep then ends the job
on the WebApp's ``consent_expired`` answer (``website_preparation_terminal``). A job that never retained
a grant stops being retried ``WEBSITE_JOB_WINDOW_MAX_SECONDS`` after its first claim.

A timed retry changes when money can be spent, never how much: it never resets ``attempt_count``,
rewrites a history row, fabricates a wait record or marks an unknown provider outcome as known.
"""
from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .job_retry_policy import (
    MONEY, PRIVACY_RIGHTS, REFUSAL_RETRY_DELAYS_SECONDS, hard_stop_kind, outage_retry_delay, refusal_retry_delay,
)

SCHEMA = "website_preparation_timed_retry.v1"
#: Job ledger field holding the exact handoff bytes a timed retry runs again.
FIELD = "timed_retry"
#: Attempt-history mark of a run a timed retry started.
MARK = "timed_retry"
#: Status of the history row a timed retry's own claim writes before the run.
STARTED_STATUS = "timed_retry_started"
STATE_FILE = "timed_retry.json"
STATE_SCHEMA = "website_preparation_timed_retry_state.v1"
#: A money refusal is asked again this often until the job's window ends; it is never counted.
MONEY_RETRY_SECONDS = 3600
#: Timed retries of unclassified failures in one ledger's life, as Atlas allows.
MAXIMUM_TIMED_RETRIES = 2 * len(REFUSAL_RETRY_DELAYS_SECONDS)
REFUSAL = "refusal"
CAPACITY = "capacity"
KINDS = frozenset({REFUSAL, MONEY, CAPACITY})
_RECORD_FIELDS = frozenset({"schema_version", "payload_base64", "source_payload_sha256", "producer_delivery_key"})
_RETRYABLE = frozenset({"failed_retryable", "retryable_blocked"})
#: History rows after which a new run of the job starts: a completed attempt, the site's reopening.
_PROGRESS = frozenset({"completed", "completed_from_output_commit", "reopened_after_terminal_authority"})


def retained_record(listener, *, payload, payload_digest, producer_delivery_key) -> dict | None:
    """The exact handoff bytes a timed retry would run, for a website browser delivery. Never raises.

    Bytes that are not the bounded handoff wire fields (``website_assessment_resume._payload_bytes``)
    are not retained, and the job behaves as before.
    """
    from .website_assessment_resume import _payload_bytes

    try:
        if producer_delivery_key is None:
            return None
        raw = _payload_bytes(listener, payload, payload_digest)
    except Exception:  # noqa: BLE001 - without the bytes the broker and the operator stay the retry owners
        return None
    return {"schema_version": SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
            "source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key}


def _record_of(ledger: Mapping[str, Any], payload_sha256, producer_delivery_key) -> dict | None:
    record = ledger.get(FIELD)
    if (type(record) is not dict or set(record) != _RECORD_FIELDS or record.get("schema_version") != SCHEMA
            or type(record.get("payload_base64")) is not str or type(payload_sha256) is not str
            or record.get("source_payload_sha256") != payload_sha256
            or record.get("producer_delivery_key") != producer_delivery_key):
        return None
    return record


def carried_record(ledger: Mapping[str, Any], *, payload_sha256, producer_delivery_key) -> dict | None:
    """The ledger's retained bytes when a claim of the same delivery cannot derive them itself."""
    return None if producer_delivery_key is None else _record_of(ledger, payload_sha256, producer_delivery_key)


def claim_matches(ledger, expected, *, payload_sha256, producer_delivery_key) -> bool:
    """Inside the claim lock: still the ledger the drain read, nothing else owns it, the same delivery."""
    from .website_reconstruction_resume import armed

    record = expected.get("record")
    return (type(expected.get("revision")) is int and ledger.get("revision") == expected["revision"]
            and expected.get("kind") in KINDS
            and ledger.get("status") in {*_RETRYABLE, "processing"}
            and type(record) is dict and ledger.get(FIELD) == record
            and _record_of(ledger, payload_sha256, producer_delivery_key) == record
            and ledger.get("source_payload_sha256") == payload_sha256
            and ledger.get("producer_delivery_key") == producer_delivery_key
            and not armed(ledger) and not isinstance(ledger.get("assessment_resume"), Mapping))


def started_row(expected: Mapping[str, Any], *, attempt_number: int, started_at: str) -> dict[str, Any]:
    """The history row a timed retry's claim commits before the run, so a killed run still counts."""
    return {"attempt_number": attempt_number, "status": STARTED_STATUS, "stage": "timed_retry",
            "started_at": started_at, MARK: {"kind": expected["kind"], "retry": expected.get("retry"),
                                              "due_at": expected.get("due_at")}}


def _history(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = ledger.get("attempt_history")
    return [row for row in history if isinstance(row, dict)] if type(history) is list else []


def _timed(row: Mapping[str, Any], kind: str | None = None) -> bool:
    mark = row.get(MARK)
    return (row.get("status") == STARTED_STATUS and type(mark) is dict
            and (mark.get("kind") in KINDS if kind is None else mark.get("kind") == kind))


def _since_progress(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows of the job's current run: after its last completed attempt or the site's correction.

    A correction of the object the robot moves starts a new run of the same job, as
    ``website_reconstruction_resume.safe_to_arm`` treats it; nothing else resets the count.
    """
    from .website_task_target_choice import CORRECTION_MARK

    marks = [index for index, row in enumerate(history)
             if row.get("status") in _PROGRESS or row.get(CORRECTION_MARK)]
    return history[marks[-1]:] if marks else history


def classify(ledger: Mapping[str, Any]) -> str:
    """What the last attempt's failure protects: privacy/rights, money, capacity, or nothing classified."""
    from .website_preparation_terminal import _budget_refused, _failure_texts, capacity_only

    if ledger.get("status") == "processing":
        return REFUSAL  # A dead worker: its outcome is unknown, so it takes the counted schedule.
    blockers, error = _failure_texts(ledger)
    kinds = {hard_stop_kind(code) for code in (error, *blockers)}
    if PRIVACY_RIGHTS in kinds:
        return PRIVACY_RIGHTS
    if _budget_refused(ledger):
        return MONEY
    # Before the money words: control_plane_disk_budget_exceeded is a disk shortage, not a spend limit.
    if capacity_only(ledger):
        return CAPACITY
    return MONEY if MONEY in kinds else REFUSAL


#: ``website_robot_fit_hold.FIELD`` and ``RECHECK_MARK``, named here so this module (which the terminal
#: sweep and the status reader reach) never imports the robot fit catalog; a test pins them together.
ROBOT_FIT_FIELD = "robot_fit_hold_replay"
ROBOT_FIT_MARK = "robot_fit_recheck"


def robot_fit_stopped_short(ledger: Mapping[str, Any]) -> bool:
    """A robot fit recheck that stopped short and keeps its bytes (``website_robot_fit_hold``'s own test)."""
    from .website_reconstruction_resume import armed

    history = _history(ledger)
    return (ledger.get("status") in _RETRYABLE and bool(history) and ROBOT_FIT_MARK in history[-1]
            and type(ledger.get(ROBOT_FIT_FIELD)) is dict
            and not armed(ledger) and not isinstance(ledger.get("assessment_resume"), Mapping))


def _owned_elsewhere(root: Path, ledger: Mapping[str, Any], robot_fit_spent=None) -> bool:
    """Whether another reconciler or wait still owes this ledger a run (so the timer never drives it too).

    ``robot_fit_spent(root, ledger)`` says whether a stopped-short robot fit recheck has used its own
    replays for the current inputs; without it (the terminal sweep) such a ledger counts as owned.
    """
    from .website_reconstruction_resume import armed

    # The assessment wait (reconcile_waiting_assessments) and the Atlas wait (reconcile_waiting_reconstructions).
    if isinstance(ledger.get("assessment_resume"), Mapping) or armed(ledger):
        return True
    pipeline = root / "pipeline"
    # The site's correction of its target (reconcile_target_choices), while it still has replays left.
    from . import website_task_target_choice as choice
    record = ledger.get(choice.REPLAY_FIELD)
    if ((pipeline / choice.QUESTION_DIR).is_dir() and ledger.get("status") in choice.CORRECTABLE_STATUSES
            and type(record) is dict and set(record) == choice._REPLAY_FIELDS
            and record.get("schema_version") == choice.REPLAY_SCHEMA):
        digest = choice.correction_pending(ledger, pipeline)
        if digest is not None and choice._replay_due(pipeline, digest, math.inf):
            return True
    # A robot fit recheck that stopped short (reconcile_robot_fit_holds), while it still has replays left.
    if robot_fit_stopped_short(ledger) and (robot_fit_spent is None or not robot_fit_spent(root, ledger)):
        return True
    # The held replay (reconcile_held_handoffs) owes only completed ledgers, which the timer never drives.
    return False


def _state_path(root: Path) -> Path:
    return Path(root) / "pipeline" / "website_scene_preparation" / STATE_FILE


def _state(root: Path) -> dict[str, Any]:
    import json

    path = _state_path(root)
    try:
        value = json.loads(path.read_text()) if not path.is_symlink() else {}
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) and value.get("schema_version") == STATE_SCHEMA else {}


def _unchanged_attempts(state: Mapping[str, Any], ledger: Mapping[str, Any]) -> int:
    """Timed attempts in a row that stopped before their claim on this very ledger revision."""
    count = state.get("unchanged_attempts")
    if (state.get("ledger_revision") != ledger.get("revision")
            or state.get("source_payload_sha256") != ledger.get("source_payload_sha256")
            or type(count) is not int or count < 1):
        return 0
    return count


def timed_plan(root: Path, ledger: Mapping[str, Any], now_epoch: float,
               robot_fit_spent=None) -> dict[str, Any] | None:
    """The timed retry this ledger is owed (its kind, retry number and due time), or None when none is.

    Read-only and never raises. The retry is dated from when the last attempt stopped: a failure or a
    blocked result when it was recorded (``updated_at``), a dead worker when its lease expired.
    """
    from .capture_delivery_ledger import ended_producer_delivery_keys
    from .website_job_window import WEBSITE_JOB_WINDOW_MAX_SECONDS
    from .website_preparation_terminal import _parse_utc, _retained_grant

    try:
        root = Path(root)
        status = ledger.get("status")
        if status in _RETRYABLE:
            stopped = _parse_utc(ledger.get("updated_at"))
        elif status == "processing" and type(ledger.get("lease_token")) is str and ledger["lease_token"]:
            stopped = _parse_utc(ledger.get("lease_expires_at"))
            if stopped is not None and stopped.timestamp() > now_epoch:
                return None  # A live lease: its owner may still be heartbeating.
        else:
            return None
        key, digest = ledger.get("producer_delivery_key"), ledger.get("source_payload_sha256")
        record = _record_of(ledger, digest, key)
        if (stopped is None or record is None or type(key) is not str
                or key in ended_producer_delivery_keys(ledger) or _owned_elsewhere(root, ledger, robot_fit_spent)):
            return None
        grant = _retained_grant(root, ledger)
        if grant is not None:
            if grant["expires_at_epoch"] <= now_epoch:
                return None  # The window ended; the terminal sweep ends the job on the WebApp's answer.
        else:
            # No grant was ever retained (the job stopped before its first one), so it has no expiry of
            # its own. The WebApp ends such a job the longest window after its video arrived and refuses
            # its first grant after that, so the clock stops asking then too, timed from the ledger's
            # first claim (``started_at``, which no later claim rewrites). Unreadable: never retried.
            started = _parse_utc(ledger.get("started_at"))
            if started is None or started.timestamp() + WEBSITE_JOB_WINDOW_MAX_SECONDS <= now_epoch:
                return None
        kind = classify(ledger)
        if kind == PRIVACY_RIGHTS:
            return None
        history = _history(ledger)
        run = _since_progress(history)
        retry = None
        if kind == REFUSAL:
            retry = sum(1 for row in run if _timed(row, REFUSAL))
            delay = (refusal_retry_delay(retry)
                     if sum(1 for row in history if _timed(row, REFUSAL)) < MAXIMUM_TIMED_RETRIES else None)
            if delay is None:
                return None  # The schedule is spent; the window ends the job.
            retry += 1
        elif kind == MONEY:
            delay = MONEY_RETRY_SECONDS
        else:
            streak = 0
            for row in reversed([row for row in run if _timed(row)]):
                if not _timed(row, CAPACITY):
                    break
                streak += 1
            delay = outage_retry_delay(streak + 1)
        due = stopped.timestamp() + delay
        state = _state(root)
        unchanged = _unchanged_attempts(state, ledger)
        attempted = state.get("attempted_at_epoch")
        if unchanged and type(attempted) in (int, float) and math.isfinite(attempted):
            # The last timed attempt changed nothing (it stopped before its claim): an outage.
            due = max(due, attempted + outage_retry_delay(unchanged))
        return {"kind": kind, "retry": retry, "due_at_epoch": due, "record": record}
    except Exception:  # noqa: BLE001 - an unreadable ledger is owed nothing by the clock
        return None


def stamp(root: Path, before: Mapping[str, Any], plan: Mapping[str, Any], now_epoch: float) -> bool:
    """Record, under the ledger lock and before any network read, that a timed attempt starts now."""
    from .common import write_json
    from .handoff_job_state import _existing_job_ledger_lock, _read_job_ledger

    with _existing_job_ledger_lock(Path(root)) as lock_state:
        if lock_state != "ledger_present" or _read_job_ledger(Path(root)) != before:
            return False
        write_json(_state_path(root), {
            "schema_version": STATE_SCHEMA, "source_payload_sha256": before.get("source_payload_sha256"),
            "producer_delivery_key": before.get("producer_delivery_key"), "ledger_revision": before.get("revision"),
            "kind": plan["kind"], "attempted_at_epoch": now_epoch,
            "unchanged_attempts": _unchanged_attempts(_state(root), before) + 1})
    return True


def iso_epoch(epoch: float) -> str:
    from datetime import datetime, timezone

    from .website_preparation_terminal import _iso

    return _iso(datetime.fromtimestamp(epoch, tz=timezone.utc))


def next_retry_at(root: Path, ledger: Mapping[str, Any], now_epoch: float) -> str | None:
    """When the owed timed retry is due (ISO), for the terminal sweep's records; None when none is owed."""
    plan = timed_plan(root, ledger, now_epoch)
    return None if plan is None else iso_epoch(plan["due_at_epoch"])


__all__ = ["CAPACITY", "FIELD", "KINDS", "MARK", "MAXIMUM_TIMED_RETRIES", "MONEY_RETRY_SECONDS", "REFUSAL",
           "SCHEMA", "STARTED_STATUS", "STATE_FILE", "carried_record", "claim_matches", "classify",
           "next_retry_at", "retained_record", "robot_fit_stopped_short", "stamp", "started_row", "timed_plan"]
