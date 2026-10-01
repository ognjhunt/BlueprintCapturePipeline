"""Isolated Render clock/CLI. Firestore is canonical; local inputs are disposable."""
import argparse
import base64
import hashlib
import json
import os
import signal
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.daily_research.consumer import Consumer, workflow
from tools.daily_research.firestore import (
    Bridge,
    FencedProvider,
    FirestoreLedger,
    control_configuration,
)
from tools.daily_research.runner import (
    CENTRAL,
    Ledger,
    Refusal,
    Runner,
    canonical,
    configuration,
    digest,
    due_date,
    observation_seconds,
    preflight,
    read_json,
    save_bytes,
    status_summary,
)

INPUTS = {"crm_snapshot": "crm.json", "knowledge_snapshot": "knowledge.json",
          "knowledge_refresh_policy": "refresh-policy.json"}


def next_wake(now):
    local = now.astimezone(CENTRAL)
    target = local.replace(hour=7, minute=0, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target.astimezone(timezone.utc)


def configured(bridge, cache):
    control = bridge.call("control")
    cfg = control_configuration(control)
    if workflow(control) and cfg.get("research_contract_version") != 3:
        raise Refusal("automatic_workflow_requires_reviewed_v3_contract")
    manifest_path = Path(__file__).resolve().parents[2] / "manifest.json"
    if cfg["enabled"] and (not manifest_path.is_file()
            or read_json(manifest_path).get("source_commit") != control.get("source_commit")
            or not control.get("legacy_attempts_reconciled_reference")
            or control["legacy_attempts_reconciled_reference"].startswith("PENDING")):
        raise Refusal("reviewed_release_or_legacy_ledger_unverified")
    for field, name in INPUTS.items():
        cfg[field] = str(cache / name)
        try:
            save_bytes(cache / name, FirestoreLedger(bridge).read_bytes(name))
        except FileNotFoundError:
            # Recovery still observes/cancels the saved session if inputs are
            # missing. The unchanged core refuses creation/qualification later.
            pass
    return configuration(cfg)


def invoke(command, bridge, cache, *, stopped=lambda: False, day=None, decision=None, api_factory=FencedProvider):
    ledger = FirestoreLedger(bridge)
    if command == "status":
        return {"store": "firestore", "root": "blueprintDailyResearch/sites-first",
                "runs": [status_summary(row) for row in ledger.rows()]}
    cfg = configured(bridge, cache)
    if command in {"run", "reconcile"}:
        pending_qa = bridge.call("active_qa")
        if pending_qa:
            return consume_workflow(bridge, cache, stopped=stopped, day=pending_qa, api_factory=api_factory)
    if command in {"run", "preflight"} and (command == "preflight" or cfg["enabled"]):
        summary = bridge.call("summary")
        day = due_date(datetime.now(timezone.utc), cfg["first_date"])
        if command == "preflight" or (day and not summary["unfinished"] and not summary["cleanup_required"]
                                      and summary["latest_date"] != day):
            # Persist a fresh canonical CRM snapshot before the core may create.
            # Recovery bypasses this read so it can still cancel an older run.
            with ledger.lock():
                bridge.call("refresh_crm")
            save_bytes(cache / "crm.json", ledger.read_bytes("crm.json"))
    api = None if command in {"review", "receipt"} else api_factory(ledger, os.environ.get("OPENAI_API_KEY", ""))
    runner = Runner(ledger, cfg, api)
    runner.stop_requested = stopped
    if command == "preflight":
        from tools.daily_research.runner import crm_snapshot, load_knowledge_bundle
        crm_snapshot(cfg["crm_snapshot"], datetime.now(timezone.utc))
        load_knowledge_bundle(cfg, datetime.now(timezone.utc))
        return {**preflight(api, cfg.get("expected_agent_instructions_sha256")), "enabled": cfg["enabled"],
                "unresolved_runs": [row["run_key"] for row in ledger.rows() if row.get("cleanup_required")]}
    if command in {"review", "receipt", "record-cleanup"}:
        if not day or decision is None:
            raise Refusal("date_and_input_required")
        return getattr(runner, command.replace("-", "_"))(day, decision)
    result = runner.start_or_resume(allow_create=command == "run")
    deadline = time.monotonic() + observation_seconds(result, cfg, "research")
    while result["state"] in {"running", "cancel_pending", "collecting"} and time.monotonic() < deadline:
        if stopped() or bridge.call("control").get("enabled") is not True:
            result = runner.cancel_current(result["date"], "observer_interrupted_or_disabled")
        time.sleep(3)
        result = runner.start_or_resume(allow_create=False)
    if result["state"] in {"running", "collecting"}:
        result = runner.cancel_current(result["date"], "observation_deadline")
    if result["state"] in {"awaiting_review", "reviewed"} and workflow(bridge.call("control")):
        return consume_workflow(bridge, cache, stopped=stopped, day=result["date"], api_factory=api_factory)
    return status_summary(result)


def emit(value):
    print(canonical(value), flush=True)


def consume_workflow(bridge, cache, *, stopped=lambda: False, day=None, api_factory=FencedProvider):
    ledger = FirestoreLedger(bridge)
    cfg = configured(bridge, cache)
    api = api_factory(ledger, os.environ.get("OPENAI_API_KEY", ""))
    consumer = Consumer(ledger, cfg, api, stopped=stopped)
    consumer.active_day = day
    try:
        active_day = day or bridge.call("active_qa") or bridge.call("work_item")
        # work_item is a projection object, active_qa is a date string.
        if isinstance(active_day, dict):
            active_day = active_day.get("date")
        row = ledger.get(active_day) if active_day else None
        until = time.monotonic() + observation_seconds(row, cfg, "qa")
        while True:
            result = consumer.step()
            if result["state"] not in {"qa_running", "qa_input_unresolved", "qa_cancel_pending", "reviewed"}:
                return result
            if time.monotonic() >= until:
                raise Refusal("workflow_observation_deadline")
            time.sleep(3)
    finally:
        api.client.close()


def export_snapshot(bridge, day, destination):
    snapshot = bridge.call("snapshot", day=day)
    row = snapshot["row"]
    files = {kind: base64.b64decode(raw, validate=True) for kind, raw in snapshot["files"].items()}
    if "artifact" in files and hashlib.sha256(files["artifact"]).hexdigest() != row.get("raw_output_digest"):
        raise Refusal("artifact_not_downloaded_or_digest_mismatch")
    if "output" in files:
        try:
            matches = "artifact" in files and canonical(json.loads(files["output"])) == canonical(json.loads(files["artifact"]))
        except (ValueError, UnicodeError):
            matches = False
        if not matches:
            raise Refusal("output_artifact_binding_mismatch")
    if "evidence" in files and digest(json.loads(files["evidence"])) != row.get("evidence_digest"):
        raise Refusal("evidence_digest_mismatch")
    if "review" in files:
        packet = json.loads(files["review"])
        pinned = packet.pop("packet_digest", None)
        if pinned != row.get("packet_digest") or digest(packet) != pinned:
            raise Refusal("review_packet_digest_mismatch")
    for kind, field in (("qa", "artifact_digest"), ("qa-evidence", "evidence_digest")):
        if kind in files:
            actual = hashlib.sha256(files[kind]).hexdigest() if kind == "qa" else digest(json.loads(files[kind]))
            if actual != row.get("qa", {}).get(field):
                raise Refusal("agent_qa_export_digest_mismatch")
    destination = Path(destination)
    destination.mkdir(mode=0o700, exist_ok=False)
    save_bytes(destination / "status.json", canonical(row).encode())
    for kind, raw in files.items():
        save_bytes(destination / (day + "-" + kind + ".json"), raw)
    return {"state": "exported", "directory": str(destination), "missing_files": snapshot["missing_files"]}


def scheduler(stopped, *, bridge_factory=Bridge, clock=lambda: datetime.now(timezone.utc)):
    # One child bridge, with no lease held during idle ticks.
    last_signature, last_day, retry_at, bridge = None, None, None, None
    try:
        while not stopped.is_set():
            try:
                if bridge is None:
                    bridge = bridge_factory()
                control = bridge.call("control")
                cfg = configuration(control_configuration(control))
                signature = digest({key: value for key, value in control.items() if key != "lease"})
                day = due_date(clock(), cfg["first_date"])
                if signature != last_signature or day != last_day or (retry_at and clock() >= retry_at):
                    last_signature, last_day = signature, day
                    with tempfile.TemporaryDirectory(prefix="blueprint-research-") as root:
                        if workflow(control):
                            emit(consume_workflow(bridge, Path(root), stopped=stopped.is_set))
                        result = invoke("run" if cfg["enabled"] else "reconcile", bridge, Path(root), stopped=stopped.is_set)
                        if workflow(control):
                            emit(consume_workflow(bridge, Path(root), stopped=stopped.is_set))
                    emit(result)
                    retry_at = clock() + timedelta(minutes=5) if result.get("state") in {
                        "creation_unresolved", "running", "cancel_pending", "collecting"} else None
                elif workflow(control) and (not retry_at or clock() >= retry_at):
                    with tempfile.TemporaryDirectory(prefix="blueprint-research-") as root:
                        emit(consume_workflow(bridge, Path(root), stopped=stopped.is_set))
            except Exception as exc:  # noqa: BLE001 - fixed codes, never upstream exception bodies
                emit({"state": "blocked", "error": str(exc) if isinstance(exc, Refusal) else "research_runtime_unavailable"})
                retry_at = clock() + timedelta(minutes=5)
                if bridge is not None:
                    bridge.close()
                    bridge = None
            if stopped.is_set():
                break
            # Idle ticks read the small control document only. Full history is
            # read at startup, a due date/control change or bounded recovery.
            stopped.wait(min(60, max(0.01, (next_wake(clock()) - clock()).total_seconds())))
    finally:
        if bridge is not None:
            bridge.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["scheduler", "init", "configure", "publish-input", "import-state", "preflight", "run", "reconcile", "status", "review", "receipt", "record-cleanup", "export"])
    parser.add_argument("--input")
    parser.add_argument("--name", choices=list(INPUTS.values()))
    parser.add_argument("--date")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda signum, frame: stopped.set())
    previous_umask = os.umask(0o077)
    bridge = None
    try:
        if args.command == "scheduler":
            scheduler(stopped)
            return 0
        bridge = Bridge()
        ledger = FirestoreLedger(bridge)
        if args.command == "init":
            value = read_json(args.input)
            configuration(control_configuration(value))
            if value["enabled"] is not False:
                raise Refusal("firestore_init_not_disabled")
            bridge.call("init", value=value)
            result = {"state": "initialized_disabled"}
        elif args.command == "configure":
            value = read_json(args.input)
            configuration(control_configuration(value))
            with ledger.lock():
                bridge.call("configure", value=value)
            result = {"state": "control_configured", "enabled": value["enabled"]}
        elif args.command == "publish-input":
            if not args.name or not args.input:
                raise Refusal("name_and_input_required")
            raw = Path(args.input).read_bytes()
            if len(raw) > 2_000_000:
                raise Refusal("local_input_too_large")
            json.loads(raw)
            with ledger.lock():
                ledger.write_bytes(args.name, raw)
            result = {"state": "input_persisted", "name": args.name}
        elif args.command == "import-state":
            if not args.input or not (Path(args.input) / "ledger.sqlite3").is_file():
                raise Refusal("legacy_ledger_missing")
            source = Ledger(args.input)
            try:
                with ledger.lock():
                    dates = []
                    for row in source.rows():
                        for kind in ("artifact", "evidence", "output", "review"):
                            name = row["date"] + "-" + kind + ".json"
                            path = source.root / name
                            if path.is_file():
                                ledger.write_bytes(name, path.read_bytes())
                        bridge.call("import_run", row=row)
                        dates.append(row["date"])
                result = {"state": "legacy_state_imported", "dates": dates}
            finally:
                source.db.close()
        elif args.command == "export":
            if not args.output or not args.date:
                raise Refusal("date_and_output_required")
            result = export_snapshot(bridge, args.date, args.output)
        else:
            with tempfile.TemporaryDirectory(prefix="blueprint-research-") as root:
                result = invoke(args.command, bridge, Path(root), stopped=stopped.is_set,
                                day=args.date, decision=read_json(args.input) if args.input else None)
        emit(result)
        return 1 if result.get("state") in {"failed", "cancelled", "creation_unresolved", "cancel_pending"} else 0
    except Exception as exc:  # noqa: BLE001 - fixed errors only; no live provider/key value
        emit({"state": "blocked", "error": str(exc) if isinstance(exc, Refusal) else "research_runtime_unavailable"})
        return 1
    finally:
        if bridge is not None:
            bridge.close()
        os.umask(previous_umask)


if __name__ == "__main__":
    raise SystemExit(main())
