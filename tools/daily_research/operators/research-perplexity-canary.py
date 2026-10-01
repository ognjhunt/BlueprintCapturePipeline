"""One bounded fresh-session canary using the exact installed research package.

inspect: GETs and private plan file only. stage: private canary control/inputs.
execute: one durable create, existing research/QA/publication. reconcile: never
creates. No scheduler, normal run edits, outreach or permanent deletion.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import os
import signal
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from tools.daily_research import discovery, render, search
from tools.daily_research.adaptive_runtime import verify_process_watchdog
from tools.daily_research.consumer import Consumer
from tools.daily_research.firestore import (
    Bridge,
    FencedProvider,
    FirestoreLedger,
    control_configuration,
)
from tools.daily_research.runner import (
    Provider,
    Refusal,
    Runner,
    canonical,
    configuration,
    crm_snapshot,
    digest,
    due_date,
    instant,
    load_knowledge_bundle,
    preflight,
    read_json,
    save_bytes,
)

spec = importlib.util.spec_from_file_location("reviewed_oct2_control", Path(__file__).with_name("research-oct2-control.py"))
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)
TEST = "perplexity-fast-20261001"
ROOT = "blueprintDailyResearch/sites-first/canaries/" + TEST
DAY = "2026-10-01"
SCOPE = "one-time-fresh-research-agent-qa-canonical-publication-no-outreach"
EXPIRES = "2026-10-02T10:00:00+00:00"
MODEL_STOP = Decimal(8)


def admission(value, now=None):
    now = now or datetime.now(timezone.utc)
    if (not isinstance(value, dict) or set(value) != {"schema_version", "test_id", "authority_reference", "ceiling_usd", "scope"}
            or value["schema_version"] != "blueprint.perplexity-canary-admission.v1"
            or value["test_id"] != TEST or type(value["ceiling_usd"]) is not int or value["ceiling_usd"] != 25
            or value["scope"] != SCOPE or not isinstance(value["authority_reference"], str)
            or not value["authority_reference"].strip() or value["authority_reference"].startswith("PENDING")
            or now >= instant(EXPIRES) or due_date(now, DAY) != DAY):
        raise Refusal("canary_one_time_admission_invalid_or_expired")
    return copy.deepcopy(value)


def driver(package, destination):
    text = Path(__file__).with_suffix(".mjs").read_text()
    text = text.replace("__RESEARCH_PACKAGE_URL__", Path(package).resolve().as_uri() + "/")
    path = Path(destination) / "canary-bridge.mjs"
    path.write_text(text)
    path.chmod(0o600)
    return path


class CanaryBridge(Bridge):
    def call(self, op, **fields):
        if op == "put":
            row = fields["row"]
            if row.get("state") == "creating" and not row.get("canary"):
                binding = super().call("control")["canary"]
                body = row["create_payload"]
                # Mutate the core's SAME body object before Store and actual POST.
                # Node-only rewriting would fail to change the provider payload.
                body["input"] = ("This is the one-time Blueprint Perplexity canary; explicitly label the resulting "
                    "brief as a test. Prioritize publicly verified named decision contacts relevant to each "
                    "site/task opportunity, then role/team channels; generic contact channels are last. "
                    "Do not invent contact details or add fields outside the strict output schema. " + body["input"])
                metadata = body["metadata"]
                metadata.pop("payload_digest", None)
                metadata.update(purpose="blueprint_research_perplexity_canary", test_id=TEST,
                                canary_root=ROOT, admission_digest=binding["admission_digest"])
                metadata["payload_digest"] = digest(body)
                row["metadata"] = metadata
                row["canary"] = copy.deepcopy(binding)
        return super().call(op, **fields)


def inspect(bridge, approval, receipt, api, cache, now=None):
    now = now or datetime.now(timezone.utc)
    approved = admission(approval, now)
    origin = bridge.call("origin")
    control = origin["control"]
    row = origin["oct1_row"]
    if (control.get("enabled") is not False or control.get("config", {}).get("enabled") is not False
            or control.get("source_commit") != migration.SOURCE or origin["summary"].get("unfinished")
            or origin["summary"].get("cleanup_required") or origin.get("active_qa")
            or not row or row.get("cleanup_required") is not False or not row.get("cleanup_receipt")):
        raise Refusal("canary_daily_guard_unreconciled_or_changed")
    raw = base64.b64decode(bridge.call("origin_file", name=DAY + "-artifact.json"), validate=True)
    migration.verify_failed(row, raw)
    inputs = {"knowledge.json": base64.b64decode(bridge.call("origin_file", name="knowledge.json"), validate=True),
              "refresh-policy.json": base64.b64decode(bridge.call("origin_file", name="refresh-policy.json"), validate=True)}
    snapshot = bridge.call("read_crm")
    inputs["crm.json"] = (canonical(snapshot) + "\n").encode()
    for name, value in inputs.items():
        save_bytes(cache / name, value)
    cfg = {**control_configuration(control), "crm_snapshot": str(cache / "crm.json"),
           "knowledge_snapshot": str(cache / "knowledge.json"), "knowledge_refresh_policy": str(cache / "refresh-policy.json")}
    configuration(cfg)
    _, known = crm_snapshot(cfg["crm_snapshot"], now)
    load_knowledge_bundle(cfg, now)
    checked = preflight(api, migration.INSTRUCTIONS, search.PROFILE)
    candidate = copy.deepcopy(control)
    candidate["workflow"]["enabled"] = False
    candidate["config"].update(first_date=DAY,
        scheduler_authority_reference="one-time-explicit-canary:" + approved["authority_reference"])
    candidate["canary"] = {"test_id": TEST, "root": ROOT, "admission": approved,
        "admission_digest": digest(approved), "origin_control": copy.deepcopy(control),
        "origin_row_digest": digest(row), "origin_raw_sha256": migration.RAW,
        "origin_row_blob": origin["oct1_row_blob"],
        "expires_at": EXPIRES, "package": receipt, "research_seconds": 1200,
        "total_seconds": 1800, "model_estimate_stop_usd": "8", "hard_total_cap": False,
        "input_digests": {name: hashlib.sha256(value).hexdigest() for name, value in inputs.items()},
        "tool_definitions_digest": digest(checked["session_agent_override"]["tools"])}
    if (candidate["config"].get("max_runtime_seconds") != 1800
            or candidate["config"].get("qa_reserved_seconds") != 600
            or candidate["config"].get("search_provider") != search.PROFILE
            or candidate["config"].get("discovery_profile") != "adaptive-sites-v1"
            or candidate["config"].get("soft_target_usd") != 5
            or candidate["config"].get("recurring_budget_authority_reference") != migration.BUDGET_AUTHORITY):
        raise Refusal("canary_production_profile_not_migrated")
    candidate["enabled"] = candidate["config"]["enabled"] = candidate["workflow"]["enabled"] = True
    configuration(control_configuration(candidate))
    from tools.daily_research.consumer import workflow
    workflow(candidate)
    candidate["enabled"] = candidate["config"]["enabled"] = candidate["workflow"]["enabled"] = False
    result = {"schema_version": "blueprint.perplexity-canary-plan.v1", "package": receipt,
              "candidate": candidate, "inputs": {name: base64.b64encode(value).decode("ascii") for name, value in inputs.items()},
              "known_crm_identity_keys": len(known), "crm_values_digest": digest(snapshot["values"]),
              "provider_calls_read_only": True, "provider_mutations": 0, "firestore_writes": 0}
    result["plan_digest"] = digest(result)
    return result


def stage(bridge, plan, receipt):
    expected = copy.deepcopy(plan)
    pinned = expected.pop("plan_digest", None)
    if pinned != digest(expected) or plan.get("schema_version") != "blueprint.perplexity-canary-plan.v1" or plan.get("package") != receipt:
        raise Refusal("canary_plan_binding_invalid")
    candidate = plan["candidate"]
    admission(candidate["canary"]["admission"])
    if candidate["canary"]["admission_digest"] != digest(candidate["canary"]["admission"]):
        raise Refusal("canary_admission_binding_invalid")
    bridge.call("stage", value=candidate)
    ledger = FirestoreLedger(bridge)
    with ledger.lock():
        bridge.call("guard")
        current = bridge.call("control")
        if ledger.get(DAY):
            raise Refusal("canary_existing_intent_reconcile_only")
        if current["canary"] != candidate["canary"]:
            raise Refusal("canary_admission_already_bound")
        for name, encoded in plan["inputs"].items():
            value = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(value).hexdigest() != candidate["canary"]["input_digests"][name]:
                raise Refusal("canary_input_binding_invalid")
            ledger.write_bytes(name, value)
        candidate = copy.deepcopy(candidate)
        candidate["enabled"] = candidate["config"]["enabled"] = candidate["workflow"]["enabled"] = True
        bridge.call("configure", value=candidate)
        if migration.normalized(bridge.call("control")) != candidate:
            raise Refusal("canary_control_readback_failed")
    return {"state": "canary_staged", "test_id": TEST, "root": ROOT, "control_digest": digest(candidate),
            "provider_mutations": 0, "normal_control_changed": False}


def spend(api, row):
    turns = api.listing("turns", row["session_id"])
    usage = {"input_tokens": 0, "output_tokens": 0}
    if not turns:
        return {"known": False, "estimate_usd": None, "hard_total_cap": False}
    for turn in turns:
        if not discovery.estimated_model_cost(turn.get("usage"))["known"]:
            return {"known": False, "estimate_usd": None, "hard_total_cap": False}
        for field in usage:
            usage[field] += turn["usage"][field]
    return discovery.estimated_model_cost(usage)


class CanaryProvider(FencedProvider):
    stopped = staticmethod(lambda: False)
    clock = staticmethod(lambda: datetime.now(timezone.utc))

    def safe(self, row=None):
        self.ledger.bridge.call("guard")
        control = self.ledger.bridge.call("control")
        binding = control["canary"]
        if not row:
            admission(binding["admission"], self.clock())
            return
        if row.get("canary") != binding:
            raise Refusal("canary_admission_binding_invalid")
        estimate = spend(self, row)
        row["canary_model_estimate"] = estimate
        if not estimate["known"] or Decimal(estimate["estimate_usd"]) >= MODEL_STOP:
            raise Refusal("canary_model_usage_unknown_or_stop_threshold")

    def create(self, payload):
        self.safe()
        self.ledger.bridge.call("create_check", day=DAY, metadata=payload["metadata"])
        self.ledger.bridge.call("guard")
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        row = self.ledger.get(DAY)
        admission(control["canary"]["admission"], self.clock())
        expired = not row or (self.clock() - instant(row["started_at"])).total_seconds() >= row["research_runtime_seconds"]
        if self.stopped() or control.get("enabled") is not True or expired:
            raise Refusal("canary_stopped_or_disabled_before_create")
        return Provider.create(self, payload)

    def tool_admit(self, row, phase):
        self.safe(row)
        return super().tool_admit(row, phase)

    def qa_input(self, session_id, event, key, day, request_digest, deadline_ms):
        self.safe(self.ledger.get(day))
        self.ledger.bridge.call("qa_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        self.ledger.bridge.call("guard")
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        if (self.stopped() or control.get("enabled") is not True or control.get("workflow", {}).get("enabled") is not True
                or self.clock().timestamp() * 1000 >= deadline_ms):
            raise Refusal("canary_stopped_disabled_or_expired_before_qa")
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)


def run(bridge, cache, *, execute=False, api_factory=CanaryProvider,
        stopped=lambda: False, clock=lambda: datetime.now(timezone.utc), sleep=time.sleep):
    ledger = FirestoreLedger(bridge)
    cfg = render.configured(bridge, cache)
    control = bridge.call("control")
    if not control or control.get("canary", {}).get("test_id") != TEST:
        raise Refusal("canary_not_staged")
    existing = ledger.get(DAY)
    if not existing:
        if not execute:
            return {"state": "no_canary_intent", "provider_mutations": 0}
        admission(control["canary"]["admission"], clock())
        bridge.call("guard")
        if due_date(clock(), cfg["first_date"]) != DAY:
            raise Refusal("canary_date_scope_invalid")
    api = api_factory(ledger, os.environ.get("OPENAI_API_KEY", ""))
    api.stopped = stopped
    api.clock = clock
    runner = Runner(ledger, cfg, api, clock=clock)
    runner.stop_requested = stopped
    consumer = Consumer(ledger, cfg, api, clock=clock, stopped=stopped)
    consumer.active_day = DAY
    try:
        result = runner.start_or_resume(allow_create=execute and not existing)
        while True:
            row = ledger.get(DAY)
            if not row:
                return result
            if row["state"] in {"failed", "cancelled", "completed", "creation_unresolved"}:
                return summary(row)
            reason = "canary_interrupted" if stopped() else None
            if row.get("session_id"):
                try:
                    bridge.call("guard")
                    estimate = spend(api, row)
                    if not estimate["known"] or Decimal(estimate["estimate_usd"]) >= MODEL_STOP:
                        reason = "canary_model_usage_unknown_or_stop_threshold"
                    with ledger.lock():
                        fresh = ledger.get(DAY)
                        fresh["canary_model_estimate"] = estimate
                        ledger.put(fresh)
                        row = fresh
                except Exception:  # noqa: BLE001 - no upstream secrets
                    reason = "canary_guard_or_usage_unavailable"
            total_exhausted = (clock() - instant(row["started_at"])).total_seconds() >= row["total_runtime_seconds"]
            if total_exhausted:
                reason = "canary_total_observation_deadline"
            if row["state"] in {"running", "collecting", "cancel_pending", "creating"}:
                if reason:
                    runner.cancel_current(DAY, reason)
                result = runner.start_or_resume(allow_create=False)
            elif row["state"] in {"awaiting_review", "reviewed"}:
                if reason and row.get("qa", {}).get("state") != "validated":
                    if row.get("qa"):
                        with ledger.lock():
                            fresh = ledger.get(DAY)
                            consumer.cancel(fresh, reason)
                    return {**summary(ledger.get(DAY)), "observer_error": reason}
                # Validated terminal QA permits only canonical publication/GET
                # recovery after paid-work deadline. No new inference here.
                result = consumer.step()
                if result["state"] in {"qa_blocked", "qa_cancel_pending", "workflow_disabled", "publication_pending"}:
                    return {**summary(ledger.get(DAY)), "observer_state": result["state"]}
            else:
                raise Refusal("canary_state_invalid")
            if reason and row["state"] != "reviewed":
                return {**summary(ledger.get(DAY)), "observer_error": reason}
            sleep(3)
    finally:
        api.client.close()


def summary(row):
    result = render.status_summary(row)
    result.update(test_id=TEST, blueprint_run_id="blueprint-research-canary:" + TEST, root=ROOT,
                  row_digest=digest(row), environment_id=row.get("environment_id"),
                  root_turn_status=row.get("turn_status"), artifact_downloaded=row.get("artifact_downloaded"),
                  raw_output_sha256=row.get("raw_output_digest"), hard_total_cap_verified=False,
                  canary_model_estimate=row.get("canary_model_estimate"), normal_control_changed=False,
                  qa_turn_id=row.get("qa", {}).get("turn_id"), qa_state=row.get("qa", {}).get("state"),
                  qa_turn_status=row.get("qa", {}).get("turn_status"),
                  qa_artifact_sha256=row.get("qa", {}).get("artifact_digest"),
                  delivery={name: {"state": value.get("state"), "receipt": value.get("receipt")}
                            for name, value in row.get("delivery", {}).items()},
                  admission_digest=row.get("canary", {}).get("admission_digest"))
    return result


def record_cleanup(bridge, cache, receipt, *, api_factory=FencedProvider):
    """Record an independently approved/completed deletion; never delete here."""
    ledger = FirestoreLedger(bridge)
    cfg = render.configured(bridge, cache)
    api = api_factory(ledger, os.environ.get("OPENAI_API_KEY", ""))
    try:
        return summary(Runner(ledger, cfg, api).record_cleanup(DAY, receipt))
    finally:
        api.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "stage", "execute", "reconcile", "status", "export", "record-cleanup"])
    parser.add_argument("--package", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--plan")
    parser.add_argument("--output")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    receipt = migration.package_receipt(args.package, args.archive)
    if args.command in {"execute", "reconcile"}:
        verify_process_watchdog()
    stop = {"requested": False}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.update(requested=True))
    with tempfile.TemporaryDirectory(prefix="blueprint-perplexity-canary-") as temporary:
        cache = Path(temporary)
        bridge = CanaryBridge(script=driver(args.package, cache))
        api = None
        try:
            if args.command == "inspect":
                if not args.approval or not args.output:
                    raise Refusal("canary_required_argument_missing")
                api = Provider(os.environ.get("OPENAI_API_KEY", ""))
                result = inspect(bridge, read_json(args.approval), receipt, api, cache)
                migration.write_private(args.output, result)
                result = {k: v for k, v in result.items() if k not in {"candidate", "inputs", "package"}}
                result["state"] = "canary_read_only_admission_checked"
            elif args.command == "stage":
                if not args.plan:
                    raise Refusal("canary_required_argument_missing")
                result = stage(bridge, read_json(args.plan), receipt)
            elif args.command == "status":
                row = FirestoreLedger(bridge).get(DAY)
                result = summary(row) if row else {"state": "no_canary_intent", "provider_mutations": 0}
            elif args.command == "export":
                if not args.output:
                    raise Refusal("canary_required_argument_missing")
                result = render.export_snapshot(bridge, DAY, args.output)
            elif args.command == "record-cleanup":
                if not args.receipt:
                    raise Refusal("canary_required_argument_missing")
                result = record_cleanup(bridge, cache, read_json(args.receipt))
            else:
                result = run(bridge, cache, execute=args.command == "execute", stopped=lambda: stop["requested"])
            print(canonical(result), flush=True)
        finally:
            if api:
                api.client.close()
            bridge.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - fixed errors only
        print(canonical({"state": "blocked", "error": str(error) if isinstance(error, Refusal) else "canary_runtime_unavailable"}), flush=True)
        raise SystemExit(1) from None
