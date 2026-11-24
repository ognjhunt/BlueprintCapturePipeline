"""Bring one internal test job's next attempt forward to now, for the operator door's ``test-job-retry-now`` request.

After a fix is deployed a test job still waits out its own retry schedule: the Atlas preview's
(``job_retry_policy.REFUSAL_RETRY_DELAYS_SECONDS``, up to a day) and the preparation retry's
(``website_preparation_retry``). A release reopens a job parked at ``needs_input`` (``release_reopened``), not one
that is processing with a retry scheduled. This changes only when that retry is due, each set to now:

* the Atlas preview job's ``next_attempt_at_epoch``, under the tick's own lock (``preparation_job_lock``);
* the preparation retry record's ``next_attempt_at``, under the capture's ledger lock, committed as the listener
  commits its own retry's new time (the next revision), so a decision the listener made on the earlier one is refused.

Nothing else on either record changes: no refusal, retry or reopen count, no wall, grant, spend or admission. The
step that then runs asks every consent, withdrawal and money admission again and counts its own retry, as it would
have when its time came.

Only an internal test job: its id is the website's ``itest-`` id, the door's own ledger (``internal_test_job_door``)
holds its submitted row, and the door's recorded result of the ``test-job`` request that row names made this job. A
site's job has none of the three. A withdrawn capture is refused. A record whose last stop reads as privacy, rights or
money is left as it is (``_wall``). A record whose step is running (its lock is held, its step mark or lease is live,
or the listener has just begun its retry) is left as it is, and the answer says so.

At most once per job per release: a retry-now that changed a record stays on the job's door ledger row with the
release, who asked and why, and a second one on the same release is refused. A release is what makes a waiting retry
worth running sooner; between releases the job's own schedule stands. One that stopped partway (killed, or a record it
could not read) counts as used, since it may have changed a record; one that changed nothing does not count.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .internal_test_job_door import HANDOFF_ROOT, LEDGER_ROOT

KIND = "test-job-retry-now"
#: The website's id for a test job (Blueprint-WebApp ``server/routes/internal-test-jobs.ts`` ``internalTestJobId``).
JOB_ID = re.compile(r"itest-[0-9a-f]{40}")
#: Where the door keeps each request's result; its ``test-job`` runner writes ``<id>.test-job.json`` there.
DOOR_RESULTS = Path("/var/lib/blueprint-operator-door/requests/results")
#: The listener's job lease (``pubsub_handoff_listener.DEFAULT_JOB_LEASE_SECONDS``). A retry the listener began less
#: than this ago and has not claimed yet may still claim: it is left to run.
REDRIVE_CLAIM_SECONDS = 900
#: The paid-resource chokepoint's own refusal, by its type or code: a money wall by its type, as the Atlas preview reads
#: it (``atlas_preparation_job_control.admission_refusal_code``); to ``hard_stop_kind`` its CamelCase name is one word.
_CHOKEPOINT = re.compile(r"PaidResourceAdmissionBlocked|paid_resource_admission_blocked")
_TEST_JOB_REQUEST = re.compile(r"[0-9]{8}T[0-9]{6}Z-test-job-[0-9a-f]{8}")
_THIS_REQUEST = re.compile(r"[0-9]{8}T[0-9]{6}Z-test-job-retry-now-[0-9a-f]{8}")
_CALLER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_CAPTURE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
CHANGED = "brought_forward"


class Refused(ValueError):
    """A typed refusal: nothing was changed."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = {} if path.is_symlink() else json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _wall(*texts: Any) -> str | None:
    """``privacy_rights`` or ``money`` when a record's stop reads as that wall, else None.

    The preparation retry's own reading (``website_preparation_retry.classify``: ``job_retry_policy.hard_stop_kind``
    once the WebApp operation's name is taken out, and the site owner's closing of the run, read as the site's own
    word), the job's spending cap, and the paid-resource chokepoint's refusal. It errs toward naming a wall: here a
    wall only leaves a record on its own schedule.
    """
    from .hard_stops import JOB_SPEND_CAP
    from .job_retry_policy import MONEY, PRIVACY_RIGHTS
    from .website_job_spend_cap import SPENDING_CAP_REACHED
    from .website_preparation_retry import OWNER_CLOSED, classify
    codes = [text for text in texts if isinstance(text, str) and text]
    kinds = {classify(code) for code in codes}
    if kinds & {PRIVACY_RIGHTS, OWNER_CLOSED}:
        return PRIVACY_RIGHTS
    if MONEY in kinds or any(code in (JOB_SPEND_CAP, SPENDING_CAP_REACHED) or _CHOKEPOINT.search(code)
                             for code in codes):
        return MONEY
    return None


def _test_job(job_id: str, *, ledger_root: Path, door_results: Path) -> tuple[Path, dict[str, Any]]:
    """The door ledger's submitted row for this job and its path, once the door's recorded request agrees."""
    found = [(path, row) for path in sorted(ledger_root.glob("*.json")) if (row := _read(path)).get("job_id") == job_id]
    if len(found) != 1 or found[0][1].get("state") != "submitted":
        raise Refused("test_job_retry_now_not_in_door_ledger")
    path, row = found[0]
    request_id, capture_id = row.get("door_request_id"), row.get("capture_id")
    recorded = (_read(door_results / f"{request_id}.test-job.json")
                if isinstance(request_id, str) and _TEST_JOB_REQUEST.fullmatch(request_id) else {})
    if (not isinstance(capture_id, str) or not _CAPTURE.fullmatch(capture_id)
            or recorded.get("door_outcome") != "submitted"
            or (recorded.get("job_id"), recorded.get("capture_id")) != (job_id, capture_id)):
        raise Refused("test_job_retry_now_not_the_doors_test_job")
    return path, row


def _capture_root(row: Mapping[str, Any], handoff_root: Path) -> Path:
    roots = [root for root in handoff_root.glob(f"*/scenes/site-{row['job_id']}/captures/{row['capture_id']}")
             if root.is_dir() and not root.is_symlink()]
    if len(roots) != 1:
        raise Refused("test_job_retry_now_capture_absent")
    return roots[0]


def _atlas(capture_root: Path, *, current: int) -> dict[str, Any]:
    """The Atlas preview job's next attempt brought to now, under the tick's own lock, or why not."""
    from .atlas_preparation_job_control import (
        STEP_IN_FLIGHT,
        PreparationJobBusy,
        preparation_job_lock,
    )
    from .common import write_json
    from .website_atlas_preview_store import JOB_FILE, read_job

    path = capture_root / "pipeline" / "atlas" / JOB_FILE
    if not path.is_file():
        return {"result": "absent"}
    try:
        with preparation_job_lock(path.parent):
            job = read_job(path)
            due = job.get("next_attempt_at_epoch")
            seen = {"state": job.get("state"), "stage": job.get("stage"), "next_attempt_at_epoch": due}
            if STEP_IN_FLIGHT in job:
                # Under the lock a mark is a step whose process died: the next tick schedules its retry itself.
                return {**seen, "result": "step_interrupted"}
            error = job.get("last_error")
            wall = _wall(job.get("reason"), job.get("last_refusal_reason"), job.get("last_transient_reason"),
                         error.get("type") if isinstance(error, Mapping) else None)
            if wall:
                return {**seen, "result": "stop_is_a_wall", "wall": wall}
            if job.get("state") != "processing" or job.get("retry_at_epoch"):
                return {**seen, "result": "not_waiting_on_a_retry"}
            if isinstance(due, bool) or not isinstance(due, (int, float)) or due <= current:
                return {**seen, "result": "already_due"}
            write_json(path, {**job, "next_attempt_at_epoch": current})
            return {**seen, "result": CHANGED, "now_epoch": current}
    except PreparationJobBusy:
        return {"result": "step_running"}


def _preparation(capture_root: Path, *, now: datetime) -> dict[str, Any]:
    """The preparation retry's next attempt brought to now, under the capture's ledger lock, or why not."""
    from .common import write_json
    from .handoff_job_state import (
        JOB_LEDGER_FILENAME,
        JOB_LEDGER_SCHEMA_VERSION,
        _existing_job_ledger_lock,
        _read_job_ledger,
    )
    from .website_preparation_retry import (
        DETERMINISTIC,
        FIELD,
        RETRIES_SPENT,
        _due,
        _iso,
        _owned_elsewhere,
        armed,
    )
    from .website_preparation_retry_record import _parse_utc

    with _existing_job_ledger_lock(capture_root) as state:
        if state != "ledger_present":
            return {"result": "absent"}
        ledger = _read_job_ledger(capture_root)
        record = armed(ledger)
        seen = {"status": ledger.get("status"), "next_attempt_at": (record or {}).get("next_attempt_at"),
                "retry_count": (record or {}).get("retry_count")}
        if ledger.get("status") == "processing":
            return {**seen, "result": "step_running"}
        wall = _wall(ledger.get("last_error"), ledger.get("terminal_code"))
        if wall:
            return {**seen, "result": "stop_is_a_wall", "wall": wall}
        if record is None or _owned_elsewhere(ledger):
            return {**seen, "result": "not_waiting_on_a_retry"}
        if record.get("kind") in (DETERMINISTIC, RETRIES_SPENT):
            return {**seen, "result": "waits_for_a_release"}  # gated on a key or a release, not on a time
        if _parse_utc(record["next_attempt_at"]) <= now:
            return {**seen, "result": "already_due"}
        started, begun = _parse_utc(ledger.get("last_attempt_started_at")), _parse_utc(record.get("last_redrive_at"))
        if (begun is not None and (started is None or started < begun)
                and (now - begun).total_seconds() < REDRIVE_CLAIM_SECONDS):
            return {**seen, "result": "step_running"}  # the listener began this retry and may still claim it
        revision = ledger.get("revision")
        if type(revision) is not int:
            return {**seen, "result": "not_waiting_on_a_retry"}
        committed = {**ledger, FIELD: {**ledger[FIELD], "next_attempt_at": _iso(now)},
                     "schema_version": JOB_LEDGER_SCHEMA_VERSION, "revision": revision + 1}
        write_json(capture_root / JOB_LEDGER_FILENAME, committed)
        try:  # when the listener will run it: a grant past its expiry first leaves the silence sweep its window
            due = _iso(_due(capture_root, committed, armed(committed)))
        except Exception:  # noqa: BLE001 - only reported; the change above is made
            due = None
        return {**seen, "result": CHANGED, "now": _iso(now), "listener_due_at": due}


def _running_release() -> str | None:
    from .website_preparation_retry import _release
    return _release()


def _validated(request: Mapping[str, Any], door_request_id: str, requested_by: str) -> tuple[str, str]:
    """The door's request, checked again here as the door checked it."""
    if not isinstance(request, Mapping) or set(request) != {"kind", "job_id", "reason"} or request["kind"] != KIND:
        raise Refused("test_job_retry_now_request_invalid")
    job_id, reason = request["job_id"], request["reason"]
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise Refused("test_job_retry_now_not_a_test_job")
    if (not isinstance(reason, str) or reason != reason.strip() or not 1 <= len(reason) <= 500
            or not reason.isprintable()):
        raise Refused("test_job_retry_now_reason_invalid")
    if not _THIS_REQUEST.fullmatch(door_request_id) or not _CALLER.fullmatch(requested_by):
        raise Refused("test_job_retry_now_request_identity_invalid")
    return job_id, reason


def run(request: Mapping[str, Any], *, door_request_id: str, requested_by: str, ledger_root: Path = LEDGER_ROOT,
        handoff_root: Path = HANDOFF_ROOT, door_results: Path = DOOR_RESULTS,
        release: Callable[[], str | None] = _running_release, now: Callable[[], float] = time.time) -> dict[str, Any]:
    from .common import write_json
    from .website_atlas_preview_store import capture_withdrawn

    job_id, reason = _validated(request, door_request_id, requested_by)
    if not (ledger_root / ".lock").is_file():
        raise Refused("test_job_retry_now_not_in_door_ledger")
    # The door ledger's own lock (``internal_test_job_door.run``): two asks for one job never both pass the bound.
    with open(ledger_root / ".lock", "a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path, row = _test_job(job_id, ledger_root=ledger_root, door_results=door_results)
        commit = release()
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise Refused("test_job_retry_now_release_unreadable")
        asks = [entry for entry in row.get("retry_now") or [] if isinstance(entry, Mapping)]
        if any(entry.get("release") == commit for entry in asks):
            raise Refused("test_job_retry_now_used_on_this_release")
        capture_root = _capture_root(row, handoff_root)
        if capture_withdrawn(capture_root):
            raise Refused("test_job_retry_now_capture_withdrawn")
        current = int(now())
        entry = {"release": commit, "door_request_id": door_request_id, "requested_by": requested_by,
                 "reason": reason, "at_epoch": current}
        # Counted before any record changes, so a process killed between them still used this release's one.
        write_json(path, {**row, "retry_now": [*asks, {**entry, "outcome": "applying"}]})
        records = {"atlas_preview": _atlas(capture_root, current=current),
                   "preparation_retry": _preparation(capture_root, now=datetime.fromtimestamp(current, timezone.utc))}
        changed = any(record["result"] == CHANGED for record in records.values())
        # Only a retry-now that changed something uses the release's one; the door's receipt keeps every ask.
        write_json(path, {**row, "retry_now": [*asks, *([{**entry, "outcome": CHANGED, "records": {
            name: record["result"] for name, record in records.items()}}] if changed else [])]})
    walls = sorted({record["wall"] for record in records.values() if record.get("wall")})
    outcome = CHANGED if changed else "refused" if walls else "unchanged"
    return {"door_outcome": outcome, **({"code": f"test_job_retry_now_last_stop_is_{walls[0]}"} if outcome == "refused"
                                        else {}),
            "job_id": job_id, "label": row.get("label"), "release": commit, "requested_by": requested_by,
            "reason": reason, "records": records}


#: The script's exit codes (deploy/operator-door/door-test-job-retry-now.sh).
EXIT = {CHANGED: 0, "unchanged": 3, "refused": 2}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--door-request-id", required=True)
    parser.add_argument("--requested-by", required=True)
    parser.add_argument("--door-results", type=Path, default=DOOR_RESULTS)
    args = parser.parse_args(argv)
    try:
        report = run(json.loads(args.request_json), door_request_id=args.door_request_id,
                     requested_by=args.requested_by, door_results=args.door_results)
    except Refused as exc:
        report = {"door_outcome": "refused", "code": str(exc)}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        text = str(exc)
        report = {"door_outcome": "failed", "code": text if re.fullmatch(r"[a-z0-9_:.-]{1,200}", text)
                  else type(exc).__name__}
    print(json.dumps(report, sort_keys=True))
    return EXIT.get(report["door_outcome"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
