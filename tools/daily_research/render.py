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


def configured(bridge, cache, *, allow_create=True):
    control = bridge.call("control")
    cfg = control_configuration(control)
    if not allow_create:
        # An offline repair can import a newer reviewed validator while keeping
        # the installed/intent package unchanged. Only local config is disabled.
        cfg["enabled"] = False
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
    runner.required_history = True
    if command == "preflight":
        from tools.daily_research.runner import crm_snapshot, load_knowledge_bundle
        crm_snapshot(cfg["crm_snapshot"], datetime.now(timezone.utc))
        load_knowledge_bundle(cfg, datetime.now(timezone.utc))
        if cfg["enabled"]:
            with ledger.lock():
                if not day or ledger.learning_context(day, allow_create=False) is None:
                    raise Refusal("research_learning_input_required")
        return {**preflight(api, cfg.get("expected_agent_instructions_sha256"), cfg.get("search_provider"), cfg.get("publication_profile")), "enabled": cfg["enabled"],
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
    api.stopped = stopped
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
            if row and row["state"] == "failed" and row.get("artifact_downloaded"):
                from tools.daily_research.recovery import RepairLoop
                latest = row.get("validation_repairs", [])
                active = latest and latest[-1]["state"] in {"running", "input_unresolved", "cancel_pending"}
                if not active and (not workflow(bridge.call("control")) or stopped()):
                    return {"state": "workflow_disabled", "date": row["date"]}
                row = RepairLoop(ledger, cfg, api, clock=lambda: datetime.now(timezone.utc), stopped=stopped).step(row["date"])
                if row["state"] == "awaiting_review":
                    consumer.active_day = row["date"]
                elif row.get("validation_repairs", [{}])[-1].get("state") == "no_progress":
                    return {"state": "validation_repair_blocked", "date": row["date"],
                            "error": row["validation_repairs"][-1].get("error"),
                            "feedback": row["validation_repairs"][-1].get("feedback")}
                elif stopped():
                    return {"state": "validation_repair_cancel_pending", "date": row["date"]}
                elif time.monotonic() >= until:
                    raise Refusal("workflow_observation_deadline")
                else:
                    time.sleep(3)
                    continue
            result = consumer.step()
            if result["state"] not in {"qa_running", "qa_input_unresolved", "qa_correction_input_unresolved", "qa_cancel_pending", "reviewed", "publication_running", "publication_input_unresolved"}:
                return result
            if time.monotonic() >= until:
                raise Refusal("workflow_observation_deadline")
            time.sleep(3)
    finally:
        api.client.close()


def publication_manifest(row, manifest):
    """Inspect exact company-owned plan/claim proof without granting restore authority."""
    error = "publication_manifest_binding_invalid"
    paginated = row.get("delivery", {}).get("notion", {}).get("plan", {}).get("protocol") == "notion-paginated-v1"
    if manifest is None:
        if paginated:
            raise Refusal(error)
        return None
    envelope = manifest
    try:
        if hashlib.sha256(envelope["manifest_json"].encode("utf-8")).hexdigest() != envelope["manifest_digest"]:
            raise Refusal(error)
        manifest = json.loads(envelope["manifest_json"])
        if (manifest["schema_version"] != "blueprint.research-publication-manifest.v1"
                or manifest["date"] != row["date"] or manifest["run_key"] != row["run_key"]
                or len(manifest["source_row_blob"]) != 64
                or any(c not in "0123456789abcdef" for c in manifest["source_row_blob"])):
            raise Refusal(error)
        if (hashlib.sha256(manifest["source_row_json"].encode("utf-8")).hexdigest() != manifest["source_row_blob"]
                or json.loads(manifest["source_row_json"]) != row):
            raise Refusal(error)
        plans = manifest["plans"]
        expected = {name for name, delivery in row.get("delivery", {}).items() if delivery.get("plan")}
        if set(plans) != expected:
            raise Refusal(error)
        for name, proof in plans.items():
            if (json.loads(proof["plan_json"]) != row["delivery"][name]["plan"]
                    or hashlib.sha256(proof["plan_json"].encode("utf-8")).hexdigest() != proof["plan_digest"]):
                raise Refusal(error)
        for name, claimed in manifest["publication_claimed"].items():
            if name not in plans or claimed != row["delivery"][name]["plan"]["request_digest"]:
                raise Refusal(error)
            if (row["delivery"][name]["plan"].get("protocol") == "notion-paginated-v1"
                    and not manifest["publication_batches"].get(name, {}).get("0")):
                raise Refusal(error)
        for name, claims in manifest["publication_batches"].items():
            plan = row["delivery"][name]["plan"]
            if (name != "notion" or plan["protocol"] != "notion-paginated-v1"
                    or claims and name not in manifest["publication_claimed"]):
                raise Refusal(error)
            if set(claims) != {str(number) for number in range(len(claims))}:
                raise Refusal(error)
            page_ids, authorities = [], []
            for number in range(len(claims)):
                claim, batch = claims[str(number)], plan["batches"][number]
                if (any(type(claim[field]) is not int for field in ("number", "start", "end"))
                        or any(claim[field] != batch[field] for field in ("number", "start", "end", "request_digest"))
                        or claim["plan_digest"] != plans[name]["plan_digest"]
                        or claim["request_digest"] != hashlib.sha256(batch["body_json"].encode("utf-8")).hexdigest()
                        or (number == 0 and claim["page_id"] is not None)
                        or (number > 0 and (not isinstance(claim["page_id"], str) or not claim["page_id"]))):
                    raise Refusal(error)
                if number:
                    page_ids.append(claim["page_id"])
                authorities.append(claim["workflow_authority"])
            if len(set(page_ids)) > 1 or any(authority != authorities[0] for authority in authorities):
                raise Refusal(error)
        history = manifest.get("attempt_history", {})
        for name, attempts in history.items():
            active = row["delivery"][name].get("attempt_number", 0)
            if (name != "notion" or type(active) is not int or active < 1
                    or set(attempts) != {str(n) for n in range(active + 1)}):
                raise Refusal(error)
            for number in range(active + 1):
                attempt = attempts[str(number)]
                if number:
                    previous = attempts[str(number - 1)]
                    if (attempt["prior_attempt"] != number - 1 or attempt["absence_verified"] is not True
                            or attempt["rejection_digest"] != previous["rejection"]["response_digest"]):
                        raise Refusal(error)
                if number == active:
                    if (attempt.get("archived") or attempt.get("claimed") != manifest["publication_claimed"].get(name)
                            or attempt.get("batches", {}) != manifest["publication_batches"].get(name, {})
                            or attempt["presentation_digest"] != (row["delivery"][name].get("presentation") or {}).get("decision_digest")):
                        raise Refusal(error)
                    continue
                source_raw = attempt["source_row_json"]
                source = json.loads(source_raw)
                prior_delivery = source["delivery"][name]
                prior_plan = prior_delivery["plan"]
                plan_raw = json.dumps(prior_plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                rejection = attempt["rejection"]
                response_raw = attempt["response_json"]
                response = json.loads(response_raw)
                claims = attempt.get("batches", {})
                if prior_plan.get("protocol") == "notion-paginated-v1":
                    batch = prior_plan["batches"][0]
                    if (set(claims) != {"0"} or claims["0"]["plan_digest"] != rejection["plan_digest"]
                            or any(claims["0"][field] != batch[field] for field in ("number", "start", "end", "request_digest"))
                            or claims["0"]["page_id"] is not None):
                        raise Refusal(error)
                elif claims:
                    raise Refusal(error)
                if (attempt.get("archived") is not True
                        or hashlib.sha256(source_raw.encode("utf-8")).hexdigest() != attempt["source_row_blob"]
                        or any(source[field] != row[field] for field in ("date", "run_key", "session_id"))
                        or source["qa"]["artifact_digest"] != row["qa"]["artifact_digest"]
                        or any(prior_delivery[field] != row["delivery"][name][field]
                               for field in ("payload", "payload_json", "payload_digest"))
                        or prior_delivery.get("attempt_number", 0) != number
                        or attempt["claimed"] != prior_plan["request_digest"]
                        or rejection["request_digest"] != attempt["claimed"]
                        or rejection["plan_digest"] != hashlib.sha256(plan_raw.encode("utf-8")).hexdigest()
                        or hashlib.sha256(prior_plan["body_json"].encode("utf-8")).hexdigest() != attempt["claimed"]
                        or hashlib.sha256(response_raw.encode("utf-8")).hexdigest() != rejection["response_digest"]
                        or rejection["response_blob"] != rejection["response_digest"]
                        or rejection["http_status"] != 400 or rejection["provider_code"] != "validation_error"
                        or response.get("object") != "error" or response.get("status") != 400
                        or response.get("code") != "validation_error"):
                    raise Refusal(error)
        if any(delivery.get("attempt_number", 0) and name not in history
               for name, delivery in row.get("delivery", {}).items()):
            raise Refusal(error)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, UnicodeError):
        raise Refusal(error) from None
    return envelope


def export_snapshot(bridge, day, destination):
    snapshot = bridge.call("snapshot", day=day)
    row = snapshot["row"]
    publication = publication_manifest(row, snapshot.get("publication_manifest"))
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
    if row.get("output_recovery"):
        recovery = row["output_recovery"]
        if (recovery.get("file") != day + "-recovery.json" or "recovery" not in files
                or digest(json.loads(files["recovery"])) != recovery.get("digest")
                or recovery.get("request", {}).get("raw_output_sha256") != row.get("raw_output_digest")):
            raise Refusal("output_recovery_export_binding_mismatch")
    for number, revision in enumerate(row.get("validation_repairs", []), 1):
        input_kind, artifact_kind = f"repair-{number}-input", f"repair-{number}-artifact"
        if (revision.get("number") != number or revision.get("input_file") != day + "-" + input_kind + ".json"
                or input_kind not in files or digest(json.loads(files[input_kind])) != revision.get("request_digest")):
            raise Refusal("validation_repair_export_binding_mismatch")
        if revision.get("artifact_file") and (revision["artifact_file"] != day + "-" + artifact_kind + ".json"
                or artifact_kind not in files
                or hashlib.sha256(files[artifact_kind]).hexdigest() != revision.get("artifact_digest")):
            raise Refusal("validation_repair_export_binding_mismatch")
    revisions = row.get("validation_repairs", [])
    if revisions and revisions[-1].get("state") == "validated":
        current = revisions[-1]
        if row.get("packet", {}).get("research_revision") != {
                "number": current["number"], "turn_id": current["turn_id"],
                "artifact_sha256": current["artifact_digest"], "original_artifact_sha256": row["raw_output_digest"]}:
            raise Refusal("validation_repair_export_packet_binding_mismatch")
    outcome = row.get("validation_repair_outcome")
    if outcome and row.get("packet", {}).get("research_exclusions") != {key: outcome[key] for key in (
            "revision", "turn_id", "artifact_sha256", "original_artifact_sha256", "excluded")}:
        raise Refusal("validation_repair_export_packet_binding_mismatch")
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
    if (row.get("qa", {}).get("input_file") and (row["qa"]["input_file"] != day + "-qa-input.json" or "qa-input" not in files
            or digest(json.loads(files["qa-input"])) != row["qa"].get("request_digest"))):
        raise Refusal("agent_qa_export_digest_mismatch")
    for correction in row.get("qa", {}).get("corrections", []):
        prefix = f"qa-correction-{correction['number']}"
        raw = files.get(prefix + "-input")
        if (correction["input_file"] != f"{day}-{prefix}-input.json" or raw is None
                or digest(json.loads(raw)) != correction["request_digest"]):
            raise Refusal("agent_qa_correction_export_digest_mismatch")
        for kind, field in (("artifact", "artifact_digest"), ("evidence", "evidence_digest")):
            if not correction.get(kind + "_file"):
                continue
            raw = files.get(prefix + "-" + kind)
            if raw is None or correction[kind + "_file"] != f"{day}-{prefix}-{kind}.json":
                raise Refusal("agent_qa_correction_export_digest_mismatch")
            actual = hashlib.sha256(raw).hexdigest() if kind == "artifact" else digest(json.loads(raw))
            if actual != correction.get(field):
                raise Refusal("agent_qa_correction_export_digest_mismatch")
    if row.get("qa", {}).get("corrections"):
        original = row["qa"]["corrections"][0]["previous_review"]
        if (hashlib.sha256(files.get("qa-original", b"")).hexdigest() != original["artifact_digest"]
                or digest(json.loads(files.get("qa-original-evidence", b"null"))) != original["evidence_digest"]):
            raise Refusal("agent_qa_correction_export_digest_mismatch")
    publication_phase = row.get("publication")
    if publication_phase:
        raw = files.get("publication-input")
        if (publication_phase["input_file"] != day + "-publication-input.json" or raw is None
                or digest(json.loads(raw)) != publication_phase["request_digest"]
                or publication_phase["idempotency_key"] != row["run_key"] + ":publication"):
            raise Refusal("publication_agent_export_digest_mismatch")
        if publication_phase.get("evidence_file") and (publication_phase["evidence_file"] != day + "-publication-evidence.json"
                or "publication-evidence" not in files
                or digest(json.loads(files["publication-evidence"])) != publication_phase["evidence_digest"]):
            raise Refusal("publication_agent_export_digest_mismatch")
    for cid, call in row.get("application_tool_calls", {}).items():
        if not call.get("result_file"):
            continue
        raw = files.get("tool-" + cid)
        expected_turns = ({row.get("publication", {}).get("turn_id")} if call.get("phase") == "publication" else
                          {row.get("turn_id")} if call.get("phase") == "research" else
                          {r.get("turn_id") for r in row.get("validation_repairs", [])} if call.get("phase") == "repair"
                          else {row.get("qa", {}).get("turn_id"),
                                *(c.get("turn_id") for c in row.get("qa", {}).get("corrections", [])),
                                *(c["previous_review"].get("turn_id") for c in row.get("qa", {}).get("corrections", []))})
        if (call["result_file"] != day + "-tool-" + cid + ".json" or raw is None
                or hashlib.sha256(raw).hexdigest() != call.get("result_sha256")
                or len(raw) != call.get("result_bytes") or digest(call.get("request")) != call.get("request_digest")):
            raise Refusal("research_tool_result_digest_mismatch")
        event = json.loads(raw)
        if (digest(event) != call.get("result_digest") or event.get("turn_id") not in expected_turns
                or event.get("call_id") != cid or call["request"].get("call_id") != cid
                or call["request"].get("turn_id") not in expected_turns
                or event.get("type") != "agent.session.input.tool_result"):
            raise Refusal("research_tool_result_digest_mismatch")
    destination = Path(destination)
    destination.mkdir(mode=0o700, exist_ok=False)
    save_bytes(destination / "status.json", canonical(row).encode())
    if publication is not None:
        save_bytes(destination / "publication-manifest.json", canonical(publication).encode())
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
