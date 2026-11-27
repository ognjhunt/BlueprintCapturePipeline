#!/usr/bin/env python3
"""Run real internal test jobs through the operator door, at most two at a time, and report each one.

    . ~/.blueprint-cloud/env.sh
    "$BLUEPRINT_PIPELINE_PYTHON" scripts/test_jobs.py submit --matrix jobs.json -k 2 --state run.json
    "$BLUEPRINT_PIPELINE_PYTHON" scripts/test_jobs.py follow --state run.json --report report.json
    "$BLUEPRINT_PIPELINE_PYTHON" scripts/test_jobs.py report --state run.json
    "$BLUEPRINT_PIPELINE_PYTHON" scripts/test_jobs.py retry-now --state run.json --label L --reason "fix is live" [--bucket B]

A matrix row is ``{"label", "footage_id", "description"}``. A clip mix (``docs/test-footage/clip-mix-v1.json``) is a
matrix too: its rows marked ``"footage_uploaded": false`` are skipped, and a row the website finds no clip for is
named and skipped (nothing was made). Before each submit the month's counted spend (spend
plus held reservations, as ``door status`` reports it) plus one job's expected spend must stay under the account's
existing alert line the door itself enforces (``internal_test_job_door.STOP_LINE``). The door also counts the paid
admissions made since that report, which ``door status`` cannot show, so it can still refuse a job this check let
through; a refusal stops the run. Nothing is deleted; every job
is found again by its label. It needs the Pipeline's Python (its intent reader digests requests with ``rfc8785``);
docs/runbooks/test-jobs.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The door client beside this file, by path: the door's own server package is also named ``operator_door``.
_spec = importlib.util.spec_from_file_location("operator_door_client", Path(__file__).resolve().parent / "operator_door.py")
door = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(door)
from blueprint_pipeline.internal_test_job_door import (  # noqa: E402 - the door's own rule, one place
    EXPECTED_JOB_USD, IN_FLIGHT_FOR_SECONDS, INTENT_ROOT, MAX_IN_FLIGHT, SLOTS_TAKEN, STOP_LINE)
from blueprint_pipeline.website_preparation_revision_identity import capture_id as intent_capture  # noqa: E402

HANDOFFS = "/var/lib/blueprint/pubsub-handoffs"
DEPLOY_WINDOW_HOUR_UTC = 15  # AGENTS.md D-08: the daily deploy window opens at 15:00 UTC
#: The door's answer when the website finds no clip under the row's footage id (``internal-test-jobs.ts``
#: ``findTestFootage``, before any job exists): nothing was made, so the run goes on to the next row.
FOOTAGE_MISSING = "test_job_webapp_http_404:internal_test_footage_not_found"


def rows_to_submit(matrix: dict) -> list[dict]:
    """The rows to ask for, in order. A row of a clip mix (``docs/test-footage/clip-mix-v1.json``) whose clip is not
    uploaded yet (``footage_uploaded`` false, or no ``footage_id``) is named and skipped; any other row is asked."""
    rows = []
    for row in matrix["jobs"]:
        if row.get("footage_id") and row.get("footage_uploaded") is not False:
            rows.append(row)
        else:
            print(f"skipped {row['label']}: its footage is not uploaded yet", file=sys.stderr)
    return rows


def cat_json(path: str) -> dict | None:
    try:
        with door._request("GET", "/fs/read", {"path": path}) as response:
            value = json.loads(response.read())
    except (door.DoorError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def ls(path: str, match: str | None = None) -> list[dict]:
    try:
        return list(door._json("GET", "/fs/list", {"path": path, **({"match": match} if match else {})})
                    .get("entries") or [])
    except door.DoorError:
        return []


def spend_refusal(status: dict) -> str | None:
    """Why the door would refuse one more job for the month (``internal_test_job_door.month_refusal``), or None.

    From the report alone: the door adds the paid admissions made since it, so it decides, and may refuse more.
    """
    month = (status.get("spend_guard") or {}).get("account_monthly_spend") or {}
    try:
        counted, line = float(month["counted_usd"]), float(month["alert_lines_usd"][STOP_LINE])
    except (KeyError, TypeError, ValueError):
        return "monthly spend or its stop line unreadable in door status"
    if month.get("admission_allowed") is not True:
        return "paid admission is closed: " + ",".join(month.get("blockers") or ["unknown"])
    if counted + float(EXPECTED_JOB_USD) >= line:
        return f"counted ${counted:.2f} + expected ${EXPECTED_JOB_USD} reaches the ${line:.2f} stop line ({STOP_LINE}%)"
    return None


def capture_root(job: dict) -> str | None:
    for bucket in ls(HANDOFFS):
        root = f"{HANDOFFS}/{bucket['name']}/scenes/site-{job['job_id']}/captures/{job['capture_id']}"
        if ls(root):
            return root
    return None


def retired_receipt(job: dict) -> dict | None:
    """The scene's retirement receipt beside the scenes, found by its name (``<scene>.retired.v1.json``), when storage
    retention retired the job's workspace. Named only: the door does not serve the receipt (it holds credential-shaped
    fields), so it is not verified here as the door's slot rule verifies it on the host
    (``website_scene_workspace_retention.retired_capture_ended``), and nothing is said of how the job ended."""
    name = f"site-{job['job_id']}.retired.v1.json"
    for bucket in ls(HANDOFFS):
        scenes = f"{HANDOFFS}/{bucket['name']}/scenes"
        for entry in ls(scenes, match=name):
            if entry.get("name") == name and entry.get("type") == "file":
                return {"path": f"{scenes}/{name}", "mtime": entry.get("mtime")}
    return None


def _names_capture(intent: dict | None, capture_id: str) -> bool:
    try:
        return intent_capture(intent["request"]) == capture_id
    except (KeyError, TypeError, ValueError):
        return False


def job_report(job: dict, status: dict) -> dict:
    row = {key: job.get(key) for key in ("label", "footage_id", "job_id", "capture_id", "submitted_at")}
    root = capture_root(job) if job.get("job_id") else None
    times, stage, failure = {"submitted": job.get("submitted_at")}, "submitted" if job.get("job_id") else "not_submitted", None
    kind = manifest = None
    handoff: dict = {}
    retired = retired_receipt(job) if job.get("job_id") and not root else None
    if retired:
        stage, times["workspace_retired"] = "workspace_retired", retired["mtime"]
    if root:
        stage = "handed_off"
        ledger = cat_json(f"{root}/pipeline_job_ledger.json") or {}
        # The preparation's own record says what it is doing; the ledger records a designed wait (the 3D reading)
        # as ``failed_retryable`` (``website_reconstruction_resume.WAIT_ERROR``), which is not a failure.
        handoff = cat_json(f"{root}/pipeline/website_scene_preparation/handoff.json") or {}
        if ledger:
            changed = {entry["name"]: entry.get("mtime") for entry in ls(root)}
            stage, times["preparation_updated"] = f"preparation:{handoff.get('status') or ledger.get('status')}", \
                changed.get("pipeline_job_ledger.json")
            failure = ledger.get("terminal_code") or (ledger.get("status") if str(ledger.get("status")).startswith(
                "terminal_") and ledger.get("status") != "terminal_success" else None)
            if ledger.get("status") == "completed" and handoff.get("status") == "needs_input":
                # The stage ended on a hold that waits outside the pipeline: nobody answers a test job, so it
                # ends here. This mirrors the door's rule, ``website_preparation_status.held_preparation``, from
                # the two fields it can read through the door; a change to either changes both.
                failure = failure or "held:" + str((handoff.get("blockers") or ["unnamed"])[0])
        semantics = ls(f"{root}/pipeline/atlas/task_semantics")
        if semantics:
            reading = cat_json(f"{root}/pipeline/atlas/task_semantics/{semantics[0]['name']}") or {}
            kind = (reading.get("semantics") or {}).get("task_family")
        manifest = cat_json(f"{root}/pipeline/company_simulation/company_simulation.json")
        # The scene chain, by the door's rule (``_scene_ended``): ended only when every intent of the capture has.
        scenes = [(cat_json(f"{INTENT_ROOT}/{entry['name']}/progression.json") or {},
                   cat_json(f"{INTENT_ROOT}/{entry['name']}/outcome.json") or {}) for entry in ls(str(INTENT_ROOT))
                  if _names_capture(cat_json(f"{INTENT_ROOT}/{entry['name']}/intent.json"), job["capture_id"])]
        running = [progress.get("status") for progress, outcome in scenes
                   if progress.get("status") != "completed" and outcome.get("terminal") is not True]
        # A completed progression supersedes a retained terminal outcome (task_evaluation_scene_outcome).
        ended = [outcome.get("reason_code") for progress, outcome in scenes
                 if progress.get("status") != "completed" and outcome.get("terminal") is True]
        if running:
            stage = f"scene:{running[0]}"
        elif scenes and len(ended) == len(scenes):
            failure = failure or ended[0]
    if manifest:
        stage, times["package_built"] = "package_built", manifest.get("built_at_epoch")
        refused = manifest.get("refused_by_webapp")
        failure = failure or manifest.get("refused") or (refused.get("reason") if isinstance(refused, dict) else None)
        if manifest.get("delivered"):
            stage = "package_delivered"
    if not failure and stage not in ("package_delivered", "workspace_retired") \
            and time.time() - (job.get("submitted_at") or 0) > IN_FLIGHT_FOR_SECONDS:
        failure = "stalled: no package and no ending after the door's in-flight window"
    ids = {job.get("job_id"), job.get("capture_id")} - {None}
    spend = [{key: entry.get(key) for key in ("job_id", "by_provider_settled_usd", "held_usd", "settled_usd")}
             for entry in ((status.get("spend_guard") or {}).get("account_monthly_spend") or {}).get("jobs") or []
             if any(i in str(entry.get("job_id")) for i in ids)] or None  # status lists only its top jobs
    package = (manifest or {}).get("package") or {}
    return {**row, "kind": kind, "stage": stage, "times": times, "spend": spend, "first_failure": failure,
            "preparation": {key: handoff.get(key) for key in ("status", "blockers")} if handoff else None,
            "package_revision": (manifest or {}).get("simulation_revision"), "zip_digest": package.get("digest"),
            "package_files": package.get("files"), "permission": (manifest or {}).get("permission"),
            **({"retired_receipt": retired["path"]} if retired else {})}


def door_key(run_id: str, label: str) -> str:
    """The idempotency key for a row: every label the door's label pattern allows gives a valid key."""
    return f"{run_id}:{label.replace(' ', '-')}"


def done(report: dict) -> bool:
    return report["stage"] in ("package_delivered", "not_submitted", "workspace_retired") or bool(report["first_failure"])


def submit(args: argparse.Namespace, state: dict) -> None:
    rows = rows_to_submit(json.loads(Path(args.matrix).read_text()))
    for row in rows:
        if any(job["label"] == row["label"] and job.get("job_id") for job in state["jobs"]):
            continue
        while True:
            if not _wait_for_slot(args, state, row):
                return
            result = _ask(args, state, row)
            if not (result.get("door_outcome") == "refused" and result.get("code") == SLOTS_TAKEN):
                break
            # The door counts every test job, other runs' too; this run counts only its own. A slot frees as
            # those end, and the refusal made nothing, so the same row is asked again under the same key.
            print(f"waiting before {row['label']}: the door has {MAX_IN_FLIGHT} test jobs in flight", file=sys.stderr)
            time.sleep(args.poll)
        if result.get("door_outcome") == "refused" and result.get("code") == FOOTAGE_MISSING:
            print(f"skipped {row['label']}: no clip at internal-test-footage/{row['footage_id']}/", file=sys.stderr)
            continue  # nothing was made; a later submit asks it again once the clip is there
        if result.get("door_outcome") != "submitted":
            return  # a refusal (the month) or no answer: never go on to the next row blind


def _wait_for_slot(args: argparse.Namespace, state: dict, row: dict) -> bool:
    """Wait until this run has a slot outside the deploy window; False when the month stops the run."""
    while True:
        status = door._json("GET", "/status")
        busy = [job for job in state["jobs"] if job.get("job_id") and not done(job_report(job, status))]
        hour = dt.datetime.now(dt.timezone.utc).hour
        why = (f"{len(busy)} of {args.k} in flight" if len(busy) >= args.k else
               "inside the daily deploy window" if hour == DEPLOY_WINDOW_HOUR_UTC else None)
        if why is None:
            refusal = spend_refusal(status)
            if refusal:
                print(f"stopped before {row['label']}: {refusal}", file=sys.stderr)
                return False
            return True
        print(f"waiting before {row['label']}: {why}", file=sys.stderr)
        time.sleep(args.poll)


def _ask(args: argparse.Namespace, state: dict, row: dict) -> dict:
    """Ask the door for the row's job and record its answer: one row per label, since a refusal made no job."""
    body = {"kind": "test-job", "idempotency_key": door_key(args.run_id, row["label"]),
            "label": row["label"], "footage_id": row["footage_id"], "description": row["description"]}
    accepted = door._json("POST", "/requests", body=body)
    result = {}
    for _ in range(120):
        request = door._json("GET", f"/requests/{accepted['id']}")
        if door._terminal(request) is not None:
            result = cat_json(str((request.get("outcome") or {}).get("result") or "")) or {}
            break
        time.sleep(5)
    state["jobs"] = [job for job in state["jobs"] if job["label"] != row["label"]]
    state["jobs"].append({"label": row["label"], "footage_id": row["footage_id"], "door_request": accepted["id"],
                          "job_id": result.get("job_id"), "capture_id": result.get("capture_id"),
                          "outcome": result.get("door_outcome"), "code": result.get("code"),
                          "submitted_at": int(time.time())})
    Path(args.state).write_text(json.dumps(state, indent=1))
    print(f"{row['label']}: {result.get('door_outcome') or 'no result'} {result.get('code') or ''}", file=sys.stderr)
    return result


def retry_now(args: argparse.Namespace, state: dict) -> int:
    """Bring one job's scheduled retries forward to now through the door (``test-job-retry-now``): timing only."""
    job = next((job for job in state["jobs"] if job["label"] == args.label and job.get("job_id")), None)
    if job is None:
        print(f"{args.label}: no started job of that label in {args.state}", file=sys.stderr)
        return 2
    accepted = door._json("POST", "/requests", body={"kind": "test-job-retry-now", "job_id": job["job_id"],
                                                     "reason": args.reason,
                                                     "bucket": args.bucket or door.handoff_bucket()})
    for _ in range(120):
        request = door._json("GET", f"/requests/{accepted['id']}")
        verdict = door._terminal(request)
        if verdict is not None:
            print(json.dumps(cat_json(str((request.get("outcome") or {}).get("result") or "")) or request, indent=1))
            return 0 if verdict else 1
        time.sleep(5)
    print(f"{args.label}: no answer to {accepted['id']} yet; read it with operator_door.py request", file=sys.stderr)
    return 1


def report(args: argparse.Namespace, state: dict) -> list[dict]:
    status = door._json("GET", "/status")
    rows = [job_report(job, status) for job in state["jobs"]]
    for r in rows:
        spent = ("$%6.2f" % sum(sum((s.get("by_provider_settled_usd") or {}).values()) for s in r["spend"])
                 if r["spend"] else "spend not shown")
        print(f"{r['label']:<28} {r['kind'] or '-':<24} {r['stage']:<32} {spent:>15}  "
              f"{r['first_failure'] or '-':<40} {(r['zip_digest'] or '-')[:19]}")
    if args.report:
        Path(args.report).write_text(json.dumps({"generated_at": int(time.time()), "jobs": rows}, indent=1))
    return rows


def main(argv: list[str] | None = None) -> int:
    if importlib.util.find_spec("rfc8785") is None:
        print("scripts/test_jobs.py needs the Pipeline's Python: . ~/.blueprint-cloud/env.sh, then run it with "
              "\"$BLUEPRINT_PIPELINE_PYTHON\" (rfc8785 is missing here)", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("submit", "follow", "report"):
        command = commands.add_parser(name)
        command.add_argument("--state", required=True, help="this run's local record of its jobs")
        command.add_argument("--report", help="write the JSON report here")
        command.add_argument("--poll", type=float, default=300)
    commands.choices["submit"].add_argument("--matrix", required=True)
    commands.choices["submit"].add_argument("-k", type=int, choices=range(1, MAX_IN_FLIGHT + 1), default=MAX_IN_FLIGHT)
    commands.choices["submit"].add_argument("--run-id", default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d"))
    retry = commands.add_parser("retry-now", help="bring one job's scheduled retries forward to now (timing only)")
    for flag in ("--state", "--label", "--reason"):
        retry.add_argument(flag, required=True)
    retry.add_argument("--bucket", help="the handoff bucket holding the job's scene (default: the host's only one)")
    args = parser.parse_args(argv)
    path = Path(args.state)
    state = json.loads(path.read_text()) if path.exists() else {"jobs": []}
    try:
        if args.command == "retry-now":
            return retry_now(args, state)
        if args.command == "submit":
            submit(args, state)
        while True:
            rows = report(args, state)
            if args.command != "follow" or all(done(r) for r in rows):
                return 0
            time.sleep(args.poll)
    except door.DoorError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
