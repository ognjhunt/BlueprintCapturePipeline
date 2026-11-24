"""Start one internal test job for the operator door's ``test-job`` request.

The website makes the job (``POST /api/internal/pipeline/test-jobs``, Blueprint-WebApp
``server/routes/internal-test-jobs.ts``) from footage in its test folder; this module only asks it, over the
existing signed Pipeline -> WebApp transport, and keeps one ledger row per idempotency key under
``internal-test-jobs/``. Before a new job it reads the month (``account_monthly_spend``: the report plus the paid
admissions made since it) and the jobs already in flight, under one lock. The job then goes through every paid admission a site's job does; nothing here spends,
deletes or changes a limit. The result names the job, its capture and its label.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

LEDGER_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/internal-test-jobs")
HANDOFF_ROOT = Path("/var/lib/blueprint/pubsub-handoffs")
INTENT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents")
#: The host has 8 GB of memory and no swap: two jobs' preparation and scene chains at once is the most it holds.
MAX_IN_FLIGHT = 2
#: The refusal when both slots are taken: nothing was made, so the same ask may come again under its key.
SLOTS_TAKEN = "test_job_two_in_flight"
#: A website scene grant lasts at most 7 days (AGENTS.md, scene sponsorship TTL); an older job no longer counts.
IN_FLIGHT_FOR_SECONDS = 7 * 24 * 3600
#: The account's existing alert line a new test job must stay under ("100" is the monthly limit itself; "80"
#: would keep the last fifth of the month for sites' jobs). The one place the door and the runner read it.
STOP_LINE = "100"
#: What one rigid test job is expected to spend (docs/runbooks/test-jobs.md: World Labs, upstream reading, ARTISAN).
EXPECTED_JOB_USD = Decimal(20)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
#: A typed website refusal: nothing was created, so the job holds no slot.
_WEBSITE_REFUSAL = re.compile(r"test_job_webapp_http_4[0-9]{2}(:.*)?")


class Refused(ValueError):
    """A typed refusal."""


def _row_path(root: Path, key: str) -> Path:
    return root / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + ".json")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _scene_ended(capture_id: str, intent_root: Path) -> bool:
    """Every scene intent of the capture finished (``progression.json`` ``completed``) or ended for good
    (``outcome.json`` terminal, ``task_evaluation_scene_outcome``), and there is at least one."""
    from .task_evaluation_scene_outcome import SceneOutcomeError, terminal_outcome
    from .website_preparation_revision_identity import capture_id as intent_capture

    ended = []
    for path in intent_root.glob("*/intent.json"):
        intent = _read(path)
        try:
            if intent_capture(intent["request"]) != capture_id:
                continue
            ended.append(_read(path.parent / "progression.json").get("status") == "completed"
                         or terminal_outcome(path.parent, intent) is not None)
        except (KeyError, TypeError, ValueError, SceneOutcomeError):
            continue
    return bool(ended) and all(ended)


def _retired(row: Mapping[str, Any], handoff_root: Path) -> bool:
    """Storage retention retired the job's workspace: a receipt that verifies as the retention code writes it
    records the capture as ended for good (``website_scene_workspace_retention.retired_capture_ended``).
    Retention retires only a scene whose captures all ended and that nothing refers to any more."""
    from .website_scene_workspace_retention import RETIRED_SUFFIX, retired_capture_ended

    scene_id = f"site-{row['job_id']}"
    return any(retired_capture_ended(storage_root=handoff_root, bucket=receipt.parents[1].name, scene_id=scene_id,
                                     capture_id=row["capture_id"])
               for receipt in handoff_root.glob(f"*/scenes/{scene_id}{RETIRED_SUFFIX}"))


def finished(row: Mapping[str, Any], handoff_root: Path, intent_root: Path) -> bool:
    """Not in flight: refused by the website, its zip delivered, its preparation ended for good or on a hold
    that waits outside the pipeline (nobody answers a test job), its workspace retired, or its scene chain
    finished or ended for good. A submission whose outcome is unknown keeps its slot. Read afresh on every
    check: a held capture a release reopens counts again from the moment its ledger leaves ``completed``.

    A retired workspace is read from its receipt only while the capture's folder is gone (2026-11-24: retention
    retired nav-doorway-96 and move-the-lamp on 2026-11-21 and both counted for the rest of their seven days,
    since a test job has no scene intent to end): a new delivery that stages the capture again is live again."""
    from .website_preparation_status import held_preparation

    if row.get("state") == "refused":
        return True
    if row.get("state") != "submitted":
        return False
    present = False
    for capture in handoff_root.glob(f"*/scenes/site-{row['job_id']}/captures/{row['capture_id']}"):
        present = True
        status = str(_read(capture / "pipeline_job_ledger.json").get("status") or "")
        if (_read(capture / "pipeline" / "company_simulation" / "company_simulation.json").get("delivered")
                or (status.startswith("terminal_") and status != "terminal_success")
                or held_preparation(capture) is not None):
            return True
    if not present and _retired(row, handoff_root):
        return True
    return _scene_ended(row["capture_id"], intent_root)


def in_flight(ledger_root: Path, handoff_root: Path, now: float, intent_root: Path = INTENT_ROOT) -> list[str]:
    rows = [_read(path) for path in sorted(ledger_root.glob("*.json"))]
    return [str(row.get("label")) for row in rows
            if now - float(row.get("requested_at_epoch") or 0) < IN_FLIGHT_FOR_SECONDS
            and not finished(row, handoff_root, intent_root)]


def month_refusal(environ: Mapping[str, str] | None = None) -> str | None:
    """Why the month cannot take one more test job, from the host's own spend report, or None.

    The month is counted as the paid chokepoint counts it: the report's figure plus the paid admissions
    journaled since the report was written (``account_monthly_spend.admitted_since_report``).
    """
    from . import account_monthly_spend as month

    blockers = month.admission_blockers(environ=environ)
    if blockers:
        return "test_job_month_closed:" + blockers[0]
    policy, _error, _path = month.configured_policy(environ)
    if policy is None:  # this module runs only on the host, where a limit must exist
        return "test_job_month_closed:" + month.LIMIT_NOT_CONFIGURED
    report = month.read_report(Path(policy["report_path"]))
    counted, line = month.money(report.get("counted_usd")), month.money((report.get("alert_lines_usd") or {})
                                                                          .get(STOP_LINE))
    try:
        since = month.admitted_since_report(report, environ=environ)
    except month.JournalUnreadable:
        return "test_job_month_closed:" + month.JOURNAL_UNREADABLE
    if counted is None or line is None or since is None:
        return "test_job_month_closed:" + month.REPORT_INVALID
    return "test_job_month_stop_line" if counted + since + EXPECTED_JOB_USD >= line else None


def post_test_job(body: Mapping[str, Any]) -> dict[str, Any]:
    """POST the request with the existing signed Pipeline -> WebApp transport (the window extension's)."""
    from .safe_outbound_http import pinned_api_policy, request
    from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
    from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url

    try:  # nothing is sent until the URL and the token are in hand: a failure here made no job
        configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
        if not configured:
            raise ValueError("test_job_webapp_url_missing")
        parsed = urlsplit(validated_https_sync_url(configured))
        token = load_pipeline_sync_token()
    except (ValueError, OSError, RuntimeError) as exc:
        raise Refused(f"test_job_not_sent:{exc}"[:200]) from None
    origin = f"{parsed.scheme}://{parsed.netloc}"
    data = json.dumps(dict(body), separators=(",", ":"), allow_nan=False).encode()
    try:
        response = request(f"{origin}/api/internal/pipeline/test-jobs", method="POST", data=data,
                           headers=_pipeline_sync_headers(token, data), timeout_seconds=60,
                           policy=pinned_api_policy(origin, max_response_bytes=20_000), max_response_bytes=20_000)
    except HTTPError as exc:
        try:
            code = json.loads(exc.read(4096)).get("code")
        except (ValueError, OSError, AttributeError):
            code = None
        finally:
            exc.close()
        suffix = f":{code}" if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,80}", code) else ""
        raise ValueError(f"test_job_webapp_http_{exc.code}{suffix}") from None
    return json.loads(response.body)


def _claim(path: Path, body: Mapping[str, Any], *, ledger_root: Path, handoff_root: Path, intent_root: Path,
           month: Callable[[], str | None], now: float, door_request_id: str) -> dict[str, Any]:
    """Under the ledger lock: the row to submit, after the key, the month and the slots are checked."""
    digest = "sha256:" + hashlib.sha256(body["description"].encode()).hexdigest()
    row = _read(path)
    if row and (row.get("label"), row.get("footage_id"), row.get("description_sha256")) != (
            body["label"], body["footage_id"], digest):
        raise Refused("test_job_idempotency_key_reused")
    if row.get("state") == "submitted":
        return row
    refusal = month()  # any ask that may make a job reads the month, a retry of an unknown outcome too
    if refusal:
        raise Refused(refusal)
    # An unknown outcome still inside its window already holds its slot and keeps its time; anything else is new.
    unknown = (row.get("state") in ("submitting", "submit_failed")
               and now - float(row.get("requested_at_epoch") or 0) < IN_FLIGHT_FOR_SECONDS)
    if not unknown and len(in_flight(ledger_root, handoff_root, now, intent_root)) >= MAX_IN_FLIGHT:
        raise Refused(SLOTS_TAKEN)
    row = {"schema_version": "internal_test_job_door_row.v1", "label": body["label"],
           "footage_id": body["footage_id"], "idempotency_key": body["idempotency_key"],
           "description_sha256": digest, "door_request_id": door_request_id, "state": "submitting",
           "requested_at_epoch": int(row["requested_at_epoch"]) if unknown else int(now)}
    from .common import write_json
    write_json(path, row)
    return row


def run(request: Mapping[str, Any], *, door_request_id: str, ledger_root: Path = LEDGER_ROOT,
        handoff_root: Path = HANDOFF_ROOT, intent_root: Path = INTENT_ROOT,
        transport: Callable[[Mapping[str, Any]], dict] = post_test_job,
        month: Callable[[], str | None] = month_refusal, now: Callable[[], float] = time.time) -> dict[str, Any]:
    from .common import write_json

    body = {key: request[key] for key in ("idempotency_key", "label", "footage_id", "description")}
    path = _row_path(ledger_root, body["idempotency_key"])
    ledger_root.mkdir(parents=True, exist_ok=True)
    with open(ledger_root / ".lock", "a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        row = _claim(path, body, ledger_root=ledger_root, handoff_root=handoff_root, intent_root=intent_root,
                     month=month, now=now(), door_request_id=door_request_id)
    if row["state"] != "submitted":
        try:
            answer = transport(body)
        except (OSError, ValueError, RuntimeError) as exc:
            refused = isinstance(exc, Refused) or bool(_WEBSITE_REFUSAL.fullmatch(str(exc)))
            write_json(path, {**row, "state": "refused" if refused else "submit_failed", "error": str(exc)[:200]})
            if refused:
                raise Refused(str(exc)[:200]) from None
            raise
        footage = answer.get("footage") if isinstance(answer, dict) else None
        copy_status = footage.get("status") if isinstance(footage, dict) else None
        if isinstance(copy_status, int) and copy_status >= 400:
            code = f"test_job_footage_refused:{footage.get('code')}"[:200]
            write_json(path, {**row, "state": "refused", "error": code})
            raise Refused(code)
        if (not isinstance(answer, dict) or not _ID.fullmatch(str(answer.get("job_id")))
                or not _ID.fullmatch(str(answer.get("capture_id")))):
            write_json(path, {**row, "state": "submit_failed", "error": "test_job_webapp_answer_invalid"})
            raise ValueError("test_job_webapp_answer_invalid")
        row = {**row, "state": "submitted", "job_id": answer["job_id"], "capture_id": answer["capture_id"],
               "submitted_at_epoch": int(now())}
        write_json(path, row)
    return {"door_outcome": "submitted", "job_id": row["job_id"], "capture_id": row["capture_id"],
            "label": row["label"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--door-request-id", required=True)
    args = parser.parse_args(argv)
    try:
        report, code = run(json.loads(args.request_json), door_request_id=args.door_request_id), 0
    except Refused as exc:
        report, code = {"door_outcome": "refused", "code": str(exc)}, 2
    except (ValueError, KeyError, TypeError, OSError) as exc:
        text = str(exc)
        report = {"door_outcome": "failed", "code": text if re.fullmatch(r"[a-z0-9_:.-]{1,200}", text)
                  else type(exc).__name__}
        code = 1
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
