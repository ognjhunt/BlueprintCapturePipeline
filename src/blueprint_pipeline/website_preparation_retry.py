"""The host owns the retry of a website preparation that stopped for a retryable reason.

Before this module the only automatic retry was Pub/Sub's own: five deliveries
about ten minutes apart, then a dead-letter queue nothing reads. Now:

* Every claimed website attempt keeps its exact handoff bytes beside its ledger
  (``PAYLOAD_FILE``), so the host can run it again without a delivery.
* A failed attempt (``failed_retryable``) records ``preparation_retry`` in the
  job ledger: how many attempts in a row have failed and when the next one is
  due. The listener's own tick (``reconcile_preparation_retries``) runs the job
  again from the kept bytes through ``process_handoff_payload``, the same path a
  delivery takes, on the schedule of the one retry policy (``job_retry_policy``):
  an outage or a money refusal backs off from 2 minutes to an hour until the
  job's deadline (the grant's expiry ends it as a delivery would, and
  ``end_expired_preparation`` ends a silent one); a failure nobody classified
  gets six timed retries (15 min to 24 h) and is then left as it is, so a
  deterministic failure never occupies the serial listener for days. A
  ``DeterministicRefusal`` (``deterministic_refusal``: the same inputs and code
  repeat it) gets no timed retry at all: it waits, and runs again once, only
  when the running release decides its step differently.
* An attempt whose process died holding its lease (an out-of-memory kill) is
  recorded as ``failed_retryable`` once its lease has expired and its process is
  provably gone, and is then retried the same way. A live lease is never touched.

What is never retried here: an attempt another drain owns (the Atlas wait, the
assessment wait, a target-choice correction, a robot fit recheck or a held
replay), and a failure ``hard_stop_kind`` reads as privacy or rights. Authority
endings are terminal before they get here.

A re-run buys nothing twice and creates no retry capacity for a call whose
outcome is unknown: each paid call keeps its own fsynced intent and receipt and
is reserved on the WebApp (``website_gemini_receipts``, ``meta_sam31``), exactly
as for a Pub/Sub redelivery. An intent with no receipt stops the re-run before
any dispatch, except the one resend the owner allowed for SAM (2026-11-07).
Nothing here buys, reserves, releases or settles anything.

``listener`` is the running ``pubsub_handoff_listener`` module, passed in so
this lane never imports it.
"""
from __future__ import annotations

import base64
import json
import os
import re
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS
from .website_preparation_recovery import note_recovery, recovery_failures  # noqa: F401 - re-exported for the tick

FIELD = "preparation_retry"
SCHEMA = "website_preparation_retry.v1"
PAYLOAD_FILE = ".pipeline_preparation_retry_payload.json"
PAYLOAD_SCHEMA = "website_preparation_retry_payload.v1"
#: A dead process is not a failure of the job: its attempt runs again soon, so the job
#: moves within 30 minutes of the kill (15 min lease, a tick, this delay).
RECLAIM_DELAY_SECONDS = 600

# The WebApp operation in a typed refusal ("website_control_preparation-spend_http_503")
# names the call, not the reason; only what follows it classifies the failure.
_OPERATION = re.compile(r"website_control_[a-z0-9-]+_http_")
# An outage: a provider or WebApp 429/5xx, a timeout, a dropped connection, a full disk.
_OUTAGE = re.compile(r"http_(?:429|5[0-9][0-9])(?![0-9])|timeout|timed out|connection|temporar|unavailable|quota"
                     r"|no space left|disk_budget|capacity_insufficient", re.IGNORECASE)
_CHAIN_LIMIT = 16
OUTAGE, REFUSAL, DETERMINISTIC = "outage", "refusal", "deterministic"


def classify(*texts: str) -> str:
    """privacy_rights, money (``job_retry_policy.hard_stop_kind``), outage or refusal (unclassified)."""
    from .job_retry_policy import PRIVACY_RIGHTS, hard_stop_kind
    stripped = [_OPERATION.sub("website_control_http_", text) for text in texts if text]
    kinds = {hard_stop_kind(text) for text in stripped} - {None}
    if kinds:
        return PRIVACY_RIGHTS if PRIVACY_RIGHTS in kinds else kinds.pop()
    return OUTAGE if any(_OUTAGE.search(text) for text in stripped) else REFUSAL


def retry_delay(kind: str, failures: int, *, deadline: bool = True) -> int | None:
    """Seconds before the next run after ``failures`` failed attempts in a row; None: not retried.

    An outage or a money wait runs until the job's deadline, the grant's expiry. A job with no
    grant has no deadline, so every failure of it takes the six timed retries instead, and is
    then left. The silence sweep ends it once the WebApp says the job ended with no grant
    (``end_no_grant_preparation``); ``end_expired_preparation`` needs a grant.
    """
    from .job_retry_policy import MONEY, outage_retry_delay, refusal_retry_delay
    if kind in (OUTAGE, MONEY) and deadline:
        return outage_retry_delay(failures)  # The job's deadline, not a count, ends these.
    if kind in (OUTAGE, MONEY, REFUSAL):
        return refusal_retry_delay(max(0, failures - 1))
    if kind == DETERMINISTIC:
        return 0  # Gated on its key (``deterministic_waiting``), never on a timer.
    return None


def has_deadline(capture_root: Path) -> bool:
    """Whether the job holds a grant, whose expiry is its deadline (``website_task_evidence``)."""
    from .website_preparation_terminal import GRANT_FILE
    return (Path(capture_root) / GRANT_FILE).is_file()


def deterministic_waiting(record: Mapping[str, Any] | None) -> bool:
    """A retry record of a deterministic refusal that the running release would decide the same way."""
    from .deterministic_refusal import unchanged
    return isinstance(record, Mapping) and record.get("kind") == DETERMINISTIC and unchanged(
        record.get("deterministic_refusal"))


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _texts(exc: BaseException) -> list[str]:
    texts, seen, current = [], set(), exc
    for _ in range(_CHAIN_LIMIT):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        try:
            texts.append(str(current))
        except Exception:  # noqa: BLE001 - an unprintable error adds no text
            pass
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return texts


def consecutive_failures(history: Sequence[Any]) -> int:
    """Failed attempts in a row at the end of the history; an explicit wait is no failure."""
    count = 0
    for row in reversed(list(history)):
        if (not isinstance(row, Mapping) or row.get("status") != "failed_retryable"
                or any(isinstance(row.get(wait), Mapping) for wait in ("reconstruction_resume", "assessment_resume"))):
            break
        count += 1
    return count


def _owned_elsewhere(ledger: Mapping[str, Any]) -> bool:
    """Another drain already replays this attempt (its own record or its mark on the last row)."""
    from .website_assessment_resume import PENDING_CODE
    from .website_held_handoff_replay import FIELD as HELD_FIELD, MARK as HELD_MARK
    from .website_reconstruction_resume import armed
    from .website_robot_fit_hold import FIELD as RECHECK_FIELD, RECHECK_MARK
    from .website_task_target_choice import CORRECTION_MARK
    history = ledger.get("attempt_history")
    last = history[-1] if type(history) is list and history and isinstance(history[-1], Mapping) else {}
    return (armed(ledger) or any(last.get(mark) for mark in (CORRECTION_MARK, RECHECK_MARK, HELD_MARK))
            or any(isinstance(ledger.get(field), Mapping) for field in (RECHECK_FIELD, HELD_FIELD))
            or (ledger.get("last_error") == PENDING_CODE and isinstance(ledger.get("assessment_resume"), Mapping)))


def retain_payload(listener: Any, *, capture_root: Path, payload: Any, payload_digest: str,
                   producer_delivery_key: str | None) -> None:
    """Keep the exact bytes of a claimed website attempt beside its ledger. Never raises."""
    from .website_assessment_resume import _payload_bytes
    try:
        if producer_delivery_key is None:
            return
        raw = _payload_bytes(listener, payload, payload_digest)
        path = Path(capture_root) / PAYLOAD_FILE
        record = {"schema_version": PAYLOAD_SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
                  "source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key}
        if path.is_symlink() or (path.is_file() and json.loads(path.read_text(encoding="utf-8")) == record):
            return
        write_json(path, record)
    except Exception:  # noqa: BLE001 - without the bytes only Pub/Sub retries this attempt
        listener.logger.warning("pubsub_handoff.preparation_retry_payload_unretained",
                                extra={"capture_id": Path(capture_root).name})


def _payload_candidates(root: Path, ledger: Mapping[str, Any]):
    path = root / PAYLOAD_FILE
    if path.is_file() and not path.is_symlink():
        yield json.loads(path.read_text(encoding="utf-8"))
    # A job that failed before this release kept its bytes only in a wait or replay record.
    history = ledger.get("attempt_history")
    for holder in [ledger, *(history if type(history) is list else [])]:
        if isinstance(holder, Mapping):
            for value in holder.values():
                if isinstance(value, Mapping) and type(value.get("payload_base64")) is str:
                    yield value


def retained_payload(listener: Any, root: Path, ledger: Mapping[str, Any]) -> bytes:
    """The exact bytes of this ledger's own delivery, or ValueError."""
    from .website_assessment_resume import MAX_BYTES, _payload_bytes
    digest, key = ledger.get("source_payload_sha256"), ledger.get("producer_delivery_key")
    for record in _payload_candidates(Path(root), ledger):
        if (isinstance(record, Mapping) and record.get("source_payload_sha256") == digest
                and record.get("producer_delivery_key") == key and type(digest) is str and type(key) is str
                and len(record["payload_base64"]) <= 4 * ((MAX_BYTES + 2) // 3)):
            return _payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True), digest)
    raise ValueError("website_preparation_retry_payload_unavailable")


def _record(ledger: Mapping[str, Any], *, failures: int, error: str, due: datetime, kind: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "source_payload_sha256": ledger.get("source_payload_sha256"),
            "producer_delivery_key": ledger.get("producer_delivery_key"), "error": error, "kind": kind,
            "retry_count": failures, "next_attempt_at": _iso(due)}


def failure_retry(*, exc: BaseException, capture_root: Path, payload_digest: str,
                  producer_delivery_key: str | None, history: Sequence[Any], failed_at: str,
                  owned_elsewhere: bool) -> dict[str, Any] | None:
    """The retry record for a failed attempt, or None when the host does not own its retry. Never raises."""
    try:
        from .deterministic_refusal import find
        failures = max(1, consecutive_failures(history))
        kind = classify(*_texts(exc))
        deterministic = find(exc) if kind == REFUSAL else None
        if deterministic is not None:
            kind = DETERMINISTIC
        delay = retry_delay(kind, failures, deadline=has_deadline(capture_root))
        if producer_delivery_key is None or owned_elsewhere or delay is None:
            return None
        path = Path(capture_root) / PAYLOAD_FILE
        kept = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
        if kept.get("source_payload_sha256") != payload_digest or kept.get("producer_delivery_key") != producer_delivery_key:
            return None
        from .website_preparation_terminal import _parse_utc
        at = _parse_utc(failed_at) or _now(None)
        record = _record({"source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key},
                         failures=failures, error=str(exc), due=at + timedelta(seconds=delay), kind=kind)
        return record if deterministic is None else {**record, "deterministic_refusal": deterministic.record()}
    except Exception:  # noqa: BLE001 - the failure itself is recorded either way
        return None


def armed(ledger: Mapping[str, Any]) -> dict[str, Any] | None:
    """This ledger's current retry record, or None (another status, another delivery, another error)."""
    from .website_preparation_terminal import _parse_utc
    record = ledger.get(FIELD)
    if (ledger.get("status") != "failed_retryable" or not isinstance(record, Mapping)
            or record.get("schema_version") != SCHEMA or record.get("error") != ledger.get("last_error")
            or record.get("source_payload_sha256") != ledger.get("source_payload_sha256")
            or record.get("producer_delivery_key") != ledger.get("producer_delivery_key")
            or type(record.get("retry_count")) is not int or _parse_utc(record.get("next_attempt_at")) is None):
        return None
    return dict(record)


# --- a process that died holding its lease ------------------------------------------------------

def _boot_epoch() -> float | None:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def owner_gone(lease_owner: Any, acquired_at: datetime | None) -> bool:
    """True only when this host proves the lease's process is gone; anything unproven is alive."""
    if type(lease_owner) is not str or lease_owner.count(":") < 2:
        return False
    host, pid_text, _ = lease_owner.rsplit(":", 2)
    if host != socket.gethostname() or not pid_text.isdigit() or int(pid_text) <= 0:
        return False
    pid = int(pid_text)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # It exists under another user: alive.
    # A live pid may be a reused one: a process that started after the lease was taken is not its owner.
    boot = _boot_epoch()
    if boot is None or acquired_at is None:
        return False
    try:
        started_ticks = float(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return False
    return boot + started_ticks / os.sysconf("SC_CLK_TCK") > acquired_at.timestamp() + 2.0


def _reclaimed(ledger: Mapping[str, Any], now: datetime) -> dict[str, Any] | None:
    """The ledger that records a dead attempt as failed_retryable, or None to leave it as it is."""
    from .website_preparation_terminal import _parse_utc
    from .website_reconstruction_resume import LEASE_EXPIRED_ERROR, LEASE_EXPIRED_STAGE
    expires = _parse_utc(ledger.get("lease_expires_at"))
    attempt = ledger.get("attempt_count")
    if (ledger.get("status") != "processing" or expires is None or expires > now
            or type(attempt) is not int or attempt < 1 or type(ledger.get("revision")) is not int
            or any(type(ledger.get(key)) is not str or not ledger[key] for key in
                   ("lease_owner", "lease_token", "source_payload_sha256", "producer_delivery_key"))
            or not owner_gone(ledger["lease_owner"], _parse_utc(ledger.get("lease_acquired_at")))):
        return None
    at = _iso(now)
    history = [row for row in ledger.get("attempt_history") or [] if isinstance(row, Mapping)]
    # The dead attempt keeps its number; what it started has no recorded outcome.
    row = {"attempt_number": attempt, "status": "failed_retryable", "stage": LEASE_EXPIRED_STAGE,
           "started_at": ledger.get("last_attempt_started_at"), "failed_at": at,
           "error_type": "HandoffLeaseError", "error": LEASE_EXPIRED_ERROR, "outcome": "unknown",
           "recovered_expired_lease": True, "previous_lease_owner": ledger["lease_owner"],
           **{key: ledger.get(key) for key in ("lease_acquired_at", "lease_heartbeat_at", "lease_expires_at")},
           "recovered_at": at, "recovered_by": SCHEMA}
    history = [*history, row]
    failures = consecutive_failures(history)
    delay = RECLAIM_DELAY_SECONDS
    return {**ledger, "status": "failed_retryable", "updated_at": at, "last_failed_at": at,
            "last_error_type": row["error_type"], "last_error": LEASE_EXPIRED_ERROR, "attempt_history": history,
            "lease_owner": None, "lease_token": None, "lease_expires_at": None,
            FIELD: _record(ledger, failures=failures, error=LEASE_EXPIRED_ERROR, due=now + timedelta(seconds=delay),
                           kind=OUTAGE)}


# --- a failure before the lease: no ledger yet --------------------------------------------------

#: A delivery refused before its claim (the WebApp's 10-second rights check, a disk or staging
#: refusal) leaves no ledger. Its bytes wait here, under the storage root, until the host's own
#: retry reaches a claim; then the job's ledger owns the retry. Pub/Sub giving up loses nothing.
UNCLAIMED_DIR = ".website_preparation_unclaimed"
UNCLAIMED_SCHEMA = "website_preparation_unclaimed.v1"


def _result_texts(result: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(result, Mapping):
        return []
    values = [result.get(key) for key in ("status", "owner_observation_reason", "staging_reason")]
    blockers = result.get("blockers")
    return [str(value) for value in [*values, *(blockers if isinstance(blockers, list) else [])] if value]


def spool_unclaimed(listener: Any, *, storage_root: Path, payload: Any, payload_digest: str,
                    result: Mapping[str, Any] | None, now: datetime | None = None) -> None:
    """Keep a website delivery that stopped before any claim, for the host's retry. Never raises."""
    from .website_assessment_resume import _payload_bytes
    try:
        raw = _payload_bytes(listener, payload, payload_digest)
        handoff = listener.parse_handoff_payload(raw)
        root = listener._handoff_capture_root(handoff, storage_root=storage_root)
        path = Path(storage_root) / UNCLAIMED_DIR / f"{payload_digest}.json"
        if (handoff.source_membership_selector is None or (root / listener.JOB_LEDGER_FILENAME).exists()
                or path.exists() or retry_delay(classify(*_result_texts(result)), 1) is None):
            return
        current = _now(now)
        write_json(path, {"schema_version": UNCLAIMED_SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
                          "payload_sha256": payload_digest, "capture_id": handoff.capture_id, "retry_count": 1,
                          "first_seen_at": _iso(current), "last_status": (_result_texts(result) or ["exception"])[0],
                          "next_attempt_at": _iso(current + timedelta(seconds=retry_delay(
                              classify(*_result_texts(result)), 1)))})
    except Exception:  # noqa: BLE001 - Pub/Sub's own redelivery still holds the message
        listener.logger.warning("pubsub_handoff.preparation_unclaimed_unretained",
                                extra={"payload_sha256": payload_digest})


def _unclaimed(listener: Any, path: Path) -> tuple[dict[str, Any], bytes]:
    from .website_assessment_resume import _payload_bytes
    from .website_preparation_terminal import _parse_utc
    entry = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(entry, dict) or entry.get("schema_version") != UNCLAIMED_SCHEMA or path.is_symlink()
            or path.name != f"{entry.get('payload_sha256')}.json" or type(entry.get("retry_count")) is not int
            or _parse_utc(entry.get("next_attempt_at")) is None):
        raise ValueError("website_preparation_unclaimed_entry_invalid")
    return entry, _payload_bytes(listener, base64.b64decode(entry["payload_base64"], validate=True),
                                 entry["payload_sha256"])


def _redrive_unclaimed(listener: Any, path: Path, *, storage_root: Path, process_args: Mapping[str, Any],
                       now: datetime) -> dict[str, Any] | None:
    entry, raw = _unclaimed(listener, path)
    retries = entry["retry_count"] + 1
    delay = retry_delay(classify(entry.get("last_status") or ""), retries)
    write_json(path, {**entry, "retry_count": retries, "last_redrive_at": _iso(now),
                      "next_attempt_at": _iso(now + timedelta(seconds=delay or REFUSAL_RETRY_DELAYS_SECONDS[-1]))})
    handoff = listener.parse_handoff_payload(raw)
    listener.logger.warning("pubsub_handoff.preparation_unclaimed_retry", extra={
        "scene_id": handoff.scene_id, "capture_id": handoff.capture_id, "retry_count": retries})
    try:
        result = listener.process_handoff_payload(raw, storage_root=storage_root,
                                                  payload_digest=entry["payload_sha256"], **process_args)
    except Exception as error:  # noqa: BLE001 - a claimed attempt's failure is in its ledger
        result = {"status": "failed_retryable", "queue_disposition": "retryable",
                  "error": f"{type(error).__name__}: {error}"}
    root = listener._handoff_capture_root(handoff, storage_root=storage_root)
    # Claimed (its ledger now owns the retry), ended, or stopped for money, privacy or rights.
    if ((root / listener.JOB_LEDGER_FILENAME).exists() or result.get("queue_disposition") != "retryable"
            or retry_delay(classify(*_result_texts(result)), retries) is None):
        path.unlink(missing_ok=True)
    else:
        write_json(path, {**json.loads(path.read_text(encoding="utf-8")), "last_status": _result_texts(result)[0]})
    return result


# --- the tick ----------------------------------------------------------------------------------

def _adopt(listener: Any, root: Path, ledger: Mapping[str, Any], now: datetime) -> dict[str, Any] | None:
    """A job that failed before this release: its retry, when its own bytes are still kept."""
    failures = max(1, consecutive_failures(ledger.get("attempt_history") or []))
    kind = classify(str(ledger.get("last_error") or ""))
    if (ledger.get("status") != "failed_retryable" or ledger.get(FIELD) is not None or _owned_elsewhere(ledger)
            or type(ledger.get("producer_delivery_key")) is not str or retry_delay(kind, failures, deadline=has_deadline(root)) is None):
        return None
    retained_payload(listener, root, ledger)
    return {**ledger, FIELD: _record(ledger, failures=failures, error=str(ledger.get("last_error")), due=now,
                                     kind=kind)}


def _parse_utc_or_raise(value: Any) -> datetime:
    from .website_preparation_terminal import _parse_utc
    parsed = _parse_utc(value)
    if parsed is None:
        raise ValueError("website_preparation_retry_time_invalid")
    return parsed


def _due(root: Path, ledger: Mapping[str, Any], record: Mapping[str, Any]) -> datetime:
    """When the retry is due. A grant past its own expiry first leaves the silence sweep its window."""
    from .website_preparation_terminal import (
        LIVENESS_GAP_SECONDS, _grant_may_have_ended, _parse_utc, silence_seconds,
    )
    due = _parse_utc(record["next_attempt_at"])
    stopped = _parse_utc(ledger.get("updated_at"))
    if stopped is not None and _grant_may_have_ended(root, ledger, time.time()):
        due = max(due, stopped + timedelta(seconds=silence_seconds() + LIVENESS_GAP_SECONDS))
    return due


def _transition(listener: Any, root: Path, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict | None:
    """Commit ``after`` only over the exact revision read; None when the ledger moved on."""
    with listener._locked_job_ledger(root, create=False) as current:
        if current != before:
            return None
        return listener._commit_job_ledger(root, after, previous_revision=int(before["revision"]))


def _redrive(listener: Any, root: Path, *, storage_root: Path, process_args: Mapping[str, Any],
             now: datetime) -> dict[str, Any] | None:
    before = listener._read_job_ledger(root)
    record = armed(before)
    if record is None or _due(root, before, record) > now:
        return None
    raw = retained_payload(listener, root, before)
    handoff = listener.parse_handoff_payload(raw)
    if root != listener._handoff_capture_root(handoff, storage_root=storage_root):
        raise ValueError("website_preparation_retry_payload_names_another_capture")
    # Push the next attempt out first: a run refused before its claim backs off as well.
    retries = record["retry_count"] + 1
    # A run refused before its claim waits as its failure kind would; past the schedule, a day.
    delay = (retry_delay(str(record.get("kind") or REFUSAL), retries, deadline=has_deadline(root))
             or REFUSAL_RETRY_DELAYS_SECONDS[-1])
    bumped = {**before, FIELD: {**record, "retry_count": retries, "last_redrive_at": _iso(now),
                                "next_attempt_at": _iso(now + timedelta(seconds=delay))}}
    committed = _transition(listener, root, before, bumped)
    if committed is None:
        return None
    listener.logger.warning("pubsub_handoff.preparation_retry", extra={
        "scene_id": handoff.scene_id, "capture_id": handoff.capture_id, "retry_count": retries,
        "last_error": str(before.get("last_error"))[:500]})
    try:
        return listener.process_handoff_payload(raw, storage_root=storage_root,
            payload_digest=record["source_payload_sha256"], expected_revision=committed["revision"], **process_args)
    except Exception as error:  # noqa: BLE001 - the attempt's own failure is already in its ledger
        return {"status": "failed_retryable", "queue_disposition": "retryable",
                "error": f"{type(error).__name__}: {error}"}


def reconcile_preparation_retries(listener: Any, *, storage_root: Path, process_args: Mapping[str, Any],
                                  limit: int = 1, now: datetime | None = None) -> dict[str, int]:
    """Reclaim dead attempts, then run at most ``limit`` due retries, the longest-due first."""
    current = _now(now)
    storage_root = Path(storage_root)
    counts = {"reclaimed": 0, "adopted": 0, "waiting": 0, "redriven": 0, "failed": 0}
    roots = sorted(path.parent for path in storage_root.glob(f"*/scenes/site-*/captures/*/{listener.JOB_LEDGER_FILENAME}")
                   if path.parent.name.startswith(("walkthrough-", "supplement-")))
    due: list[tuple[datetime, str, Path]] = []
    for path in sorted((storage_root / UNCLAIMED_DIR).glob("*.json")):
        try:
            entry, raw = _unclaimed(listener, path)
            root = listener._handoff_capture_root(listener.parse_handoff_payload(raw), storage_root=storage_root)
            if (root / listener.JOB_LEDGER_FILENAME).exists():
                path.unlink()  # A delivery reached its claim: the job's ledger owns the retry now.
                continue
            when = _parse_utc_or_raise(entry["next_attempt_at"])
            if when > current:
                counts["waiting"] += 1
            else:
                due.append((when, "unclaimed", path))
        except Exception as error:  # noqa: BLE001 - one unreadable entry never stops the others
            counts["failed"] += 1
            note_recovery(listener, storage_root, key=f"{UNCLAIMED_DIR}/{path.name}", recovery="unclaimed_retry",
                          error=error)
    for root in roots:
        key = str(root.relative_to(storage_root))
        try:
            if (root / listener.JOB_LEDGER_FILENAME).is_symlink():
                continue
            ledger = listener._read_job_ledger(root)
            reclaimed = _reclaimed(ledger, current)
            if reclaimed is not None and (committed := _transition(listener, root, ledger, reclaimed)) is not None:
                counts["reclaimed"] += 1
                listener.logger.warning("pubsub_handoff.preparation_dead_attempt_reclaimed", extra={
                    "capture_id": root.name, "previous_lease_owner": ledger.get("lease_owner")})
                ledger = committed
            if armed(ledger) is None:
                adopted = _adopt(listener, root, ledger, current)
                if adopted is None or (committed := _transition(listener, root, ledger, adopted)) is None:
                    continue
                counts["adopted"] += 1
                ledger = committed
            when = _due(root, ledger, armed(ledger))
            if when > current or deterministic_waiting(armed(ledger)):
                counts["waiting"] += 1
            else:
                due.append((when, "ledger", root))
        except Exception as error:  # noqa: BLE001 - one unreadable capture never stops the others
            counts["failed"] += 1
            note_recovery(listener, storage_root, key=key, recovery="preparation_retry", error=error,
                          capture_id=root.name)
    for _, kind, root in sorted(due)[:max(0, min(limit, 50))]:
        key = str(root.relative_to(storage_root))
        try:
            result = (_redrive(listener, root, storage_root=storage_root, process_args=process_args, now=current)
                      if kind == "ledger" else
                      _redrive_unclaimed(listener, root, storage_root=storage_root, process_args=process_args,
                                         now=current))
        except Exception as error:  # noqa: BLE001 - one capture's retry never stops the others
            counts["failed"] += 1
            note_recovery(listener, storage_root, key=key, recovery="preparation_retry", error=error,
                          capture_id=root.name)
            continue
        if result is None:
            continue
        counts["redriven"] += 1
        if result.get("queue_disposition") == "retryable" and result.get("status") != "retryable_blocked":
            note_recovery(listener, storage_root, key=key, recovery="preparation_retry",
                          error=str(result.get("error") or result.get("status")), capture_id=root.name)
        else:
            note_recovery(listener, storage_root, key=key, recovery="preparation_retry")
    return counts
