#!/usr/bin/env python3
"""Run real internal test jobs through the operator door, at most two at a time, and report each one.

    python3 scripts/test_jobs.py submit --matrix jobs.json -k 2 --state run.json
    python3 scripts/test_jobs.py follow --state run.json --report report.json
    python3 scripts/test_jobs.py report --state run.json

A matrix row is ``{"label", "footage_id", "description"}``. Before each submit the month's counted spend (spend
plus held reservations, as ``door status`` reports it) plus one job's expected spend must stay under the account's
existing alert line the door itself enforces (``internal_test_job_door.STOP_LINE``). Nothing is deleted; every job
is found again by its label. Standard library plus that one module's constants; docs/runbooks/test-jobs.md.
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
    EXPECTED_JOB_USD, IN_FLIGHT_FOR_SECONDS, INTENT_ROOT, MAX_IN_FLIGHT, STOP_LINE)
from blueprint_pipeline.website_preparation_revision_identity import capture_id as intent_capture  # noqa: E402

HANDOFFS = "/var/lib/blueprint/pubsub-handoffs"
DEPLOY_WINDOW_HOUR_UTC = 15  # AGENTS.md D-08: the daily deploy window opens at 15:00 UTC


def cat_json(path: str) -> dict | None:
    try:
        with door._request("GET", "/fs/read", {"path": path}) as response:
            value = json.loads(response.read())
    except (door.DoorError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def ls(path: str) -> list[dict]:
    try:
        return list(door._json("GET", "/fs/list", {"path": path}).get("entries") or [])
    except door.DoorError:
        return []


def spend_refusal(status: dict) -> str | None:
    """Why the door would refuse one more job for the month (``internal_test_job_door.month_refusal``), or None."""
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
    if root:
        stage = "handed_off"
        ledger = cat_json(f"{root}/pipeline_job_ledger.json") or {}
        if ledger:
            changed = {entry["name"]: entry.get("mtime") for entry in ls(root)}
            stage, times["preparation_updated"] = f"preparation:{ledger.get('status')}", changed.get(
                "pipeline_job_ledger.json")
            failure = ledger.get("terminal_code") or (ledger.get("status") if str(ledger.get("status")).startswith(
                "terminal_") and ledger.get("status") != "terminal_success" else None)
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
        if running:
            stage = f"scene:{running[0]}"
        elif scenes and all(outcome.get("terminal") is True for _progress, outcome in scenes):
            failure = failure or scenes[0][1].get("reason_code")
    if manifest:
        stage, times["package_built"] = "package_built", manifest.get("built_at_epoch")
        refused = manifest.get("refused_by_webapp")
        failure = failure or manifest.get("refused") or (refused.get("reason") if isinstance(refused, dict) else None)
        if manifest.get("delivered"):
            stage = "package_delivered"
    if not failure and stage != "package_delivered" and time.time() - (job.get("submitted_at") or 0) > IN_FLIGHT_FOR_SECONDS:
        failure = "stalled: no package and no ending after the door's in-flight window"
    ids = {job.get("job_id"), job.get("capture_id")} - {None}
    spend = [{key: entry.get(key) for key in ("job_id", "by_provider_settled_usd", "held_usd", "settled_usd")}
             for entry in ((status.get("spend_guard") or {}).get("account_monthly_spend") or {}).get("jobs") or []
             if any(i in str(entry.get("job_id")) for i in ids)] or None  # status lists only its top jobs
    package = (manifest or {}).get("package") or {}
    return {**row, "kind": kind, "stage": stage, "times": times, "spend": spend, "first_failure": failure,
            "package_revision": (manifest or {}).get("simulation_revision"), "zip_digest": package.get("digest"),
            "permission": (manifest or {}).get("permission")}


def door_key(run_id: str, label: str) -> str:
    """The idempotency key for a row: every label the door's label pattern allows gives a valid key."""
    return f"{run_id}:{label.replace(' ', '-')}"


def done(report: dict) -> bool:
    return report["stage"] in ("package_delivered", "not_submitted") or bool(report["first_failure"])


def submit(args: argparse.Namespace, state: dict) -> None:
    rows = json.loads(Path(args.matrix).read_text())["jobs"]
    for row in rows:
        if any(job["label"] == row["label"] and job.get("job_id") for job in state["jobs"]):
            continue
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
                    return
                break
            print(f"waiting before {row['label']}: {why}", file=sys.stderr)
            time.sleep(args.poll)
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
        state["jobs"].append({"label": row["label"], "footage_id": row["footage_id"], "door_request": accepted["id"],
                              "job_id": result.get("job_id"), "capture_id": result.get("capture_id"),
                              "outcome": result.get("door_outcome"), "code": result.get("code"),
                              "submitted_at": int(time.time())})
        Path(args.state).write_text(json.dumps(state, indent=1))
        print(f"{row['label']}: {result.get('door_outcome') or 'no result'} {result.get('code') or ''}", file=sys.stderr)
        if result.get("door_outcome") != "submitted":
            return  # a refusal (the month, two in flight) or no answer: never go on to the next row blind


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
    args = parser.parse_args(argv)
    path = Path(args.state)
    state = json.loads(path.read_text()) if path.exists() else {"jobs": []}
    try:
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
