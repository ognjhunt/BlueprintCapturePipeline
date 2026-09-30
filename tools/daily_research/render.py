"""Isolated Render clock/CLI. Firestore is canonical; local inputs are disposable."""
import argparse
import json
import os
import signal
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.daily_research.firestore import Bridge, FencedProvider, FirestoreLedger, control_configuration
from tools.daily_research.runner import (
    CENTRAL,
    Refusal,
    Runner,
    canonical,
    configuration,
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
    manifest_path = Path(__file__).resolve().parents[2] / "manifest.json"
    if cfg["enabled"]:
        if (not manifest_path.is_file() or read_json(manifest_path).get("source_commit") != control.get("source_commit")
                or not control.get("legacy_attempts_reconciled_reference")
                or control["legacy_attempts_reconciled_reference"].startswith("PENDING")):
            raise Refusal("reviewed_release_or_legacy_ledger_unverified")
    for field, name in INPUTS.items():
        cfg[field] = str(cache / name)
        try:
            save_bytes(cache / name, FirestoreLedger(bridge).read_bytes(name))
        except Refusal:
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
    api = None if command in {"review", "receipt"} else api_factory(ledger, os.environ.get("OPENAI_API_KEY", ""))
    runner = Runner(ledger, cfg, api)
    runner.stop_requested = stopped
    if command == "preflight":
        from tools.daily_research.runner import crm_snapshot, load_knowledge_bundle
        crm_snapshot(cfg["crm_snapshot"], datetime.now(timezone.utc))
        load_knowledge_bundle(cfg, datetime.now(timezone.utc))
        return {**preflight(api), "enabled": cfg["enabled"], "unresolved_runs": [row["run_key"] for row in ledger.rows() if row.get("cleanup_required")]}
    if command in {"review", "receipt", "record-cleanup"}:
        if not day or decision is None:
            raise Refusal("date_and_input_required")
        return getattr(runner, command.replace("-", "_"))(day, decision)
    result = runner.start_or_resume(allow_create=command == "run")
    deadline = time.monotonic() + 300
    while result["state"] in {"running", "cancel_pending", "collecting"} and time.monotonic() < deadline:
        if stopped() or bridge.call("control").get("enabled") is not True:
            result = runner.cancel_current(result["date"], "observer_interrupted_or_disabled")
        time.sleep(3)
        result = runner.start_or_resume(allow_create=False)
    if result["state"] in {"running", "collecting"}:
        result = runner.cancel_current(result["date"], "observation_deadline")
    return status_summary(result)


def emit(value):
    print(canonical(value), flush=True)


def scheduler(stopped, *, bridge_factory=Bridge, clock=lambda: datetime.now(timezone.utc)):
    # Catch up/reconcile on startup. Every wake uses the canonical per-date
    # ledger; neither deployment overlap nor a missed wake replays paid work.
    while not stopped.is_set():
        bridge = bridge_factory()
        try:
            with tempfile.TemporaryDirectory(prefix="blueprint-research-") as root:
                enabled = bridge.call("control").get("enabled") is True
                result = invoke("run" if enabled else "reconcile", bridge, Path(root), stopped=stopped.is_set)
                emit(result)
        except Exception as exc:  # noqa: BLE001 - never expose provider/credential exception bodies
            emit({"state": "blocked", "error": str(exc) if isinstance(exc, Refusal) else "research_runtime_unavailable"})
        finally:
            bridge.close()
        if stopped.is_set():
            break
        # Recheck wall time/control each minute to survive clock changes and
        # allow an operator's pause without waiting until the next morning.
        wait = min(60, max(0.01, (next_wake(clock()) - clock()).total_seconds()))
        stopped.wait(wait)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["scheduler", "init", "publish-input", "preflight", "run", "reconcile", "status", "review", "receipt", "record-cleanup", "export"])
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
        elif args.command == "export":
            if not args.output or not args.date:
                raise Refusal("date_and_output_required")
            row = ledger.get(args.date)
            if not row:
                raise Refusal("run_missing")
            destination = Path(args.output)
            destination.mkdir(mode=0o700, exist_ok=False)
            save_bytes(destination / "status.json", canonical(row).encode())
            for kind in ("artifact", "evidence", "review"):
                name = args.date + "-" + kind + ".json"
                try:
                    raw = ledger.read_bytes(name)
                except Refusal:
                    continue
                save_bytes(destination / name, raw)
            result = {"state": "exported", "directory": str(destination)}
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
