"""One Oct 1 test using the existing store, provider, collector and agent QA.

No scheduler, session creation, publication or cleanup. An explicit execute may
attempt one input only after fresh admission; reconcile never sends research.
"""
import argparse
import copy
import os
import signal
import tempfile
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from tools.daily_research import adaptive, discovery
from tools.daily_research.consumer import Consumer
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import (
    REMOTE_OUTPUT,
    Provider,
    Refusal,
    Runner,
    canonical,
    crm_snapshot,
    digest,
    instant,
    preflight,
    read_json,
    save_json,
)

TEST_ID = "adaptive-discovery-20261001"
INSTRUCTIONS = "84aeea7cec9eb6e20d0d2fba10dcb269a615174e48ed91a60ff7a6a83ca37938"
FINAL = {"failed", "cancelled", "test_qa_validated", "test_qa_blocked"}
# 1860s: research 1200 + QA 600 + 60. 2580s adds two same-session repair windows
# (2 x (300 + 60)) so a run that needs output repair still reaches QA unkilled.
WATCHDOG_BOUNDS = (b"1860s", b"2580s")


def verify_process_watchdog():
    """Render/Linux entrypoint must be a child of the exact bounded timeout."""
    try:
        with open(f"/proc/{os.getppid()}/cmdline", "rb") as parent:
            command = parent.read(8193)
        parts = command.split(b"\0")
        valid = (len(command) <= 8192 and len(parts) >= 5 and Path(os.fsdecode(parts[0])).name == "timeout"
                 and parts[1:3] == [b"--signal=TERM", b"--kill-after=60s"] and parts[3] in WATCHDOG_BOUNDS)
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise Refusal("adaptive_independent_process_watchdog_required")


class TestLedger(FirestoreLedger):
    def get(self, day):
        if day != "2026-10-01":
            raise Refusal("adaptive_date_invalid")
        return self.bridge.call("adaptive_get")

    def rows(self):
        row = self.get("2026-10-01")
        return [row] if row else []

    def put(self, row):
        self.bridge.call("adaptive_put", row=row)

    def write_bytes(self, name, value):
        import base64
        self.bridge.call("adaptive_file_put", name=name, bytes=base64.b64encode(value).decode("ascii"))

    def read_bytes(self, name):
        import base64
        return base64.b64decode(self.bridge.call("adaptive_file_get", name=name), validate=True)


class TestProvider:
    """Scope existing adapters to new turns and the unique immutable artifact."""
    def __init__(self, api, ledger, intent, clock, stopped=lambda: False):
        self.api, self.ledger, self.intent, self.clock, self.stopped = api, ledger, intent, clock, stopped

    def get(self, resource, resource_id):
        value = self.api.get(resource, resource_id)
        if resource == "session" and value.get("agent", {}).get("service_tier") != "default":
            raise Refusal("adaptive_standard_service_tier_changed")
        return value

    def listing(self, resource, session_id=None):
        if resource == "sessions" or session_id != self.intent["session_id"]:
            raise Refusal("adaptive_exact_session_only")
        result = self.api.listing(resource, session_id)
        if resource == "turns":
            return [t for t in result if t["id"] not in self.intent["baseline_turn_ids"]]
        if resource == "artifacts":
            return [{**a, "path": REMOTE_OUTPUT} if a.get("path") == self.intent["output_path"] else a
                    for a in result if a.get("path") != REMOTE_OUTPUT]
        return result

    def artifact(self, sid, aid):
        return self.api.artifact(sid, aid)

    def create(self, payload):
        raise Refusal("adaptive_session_create_forbidden")

    def input(self, phase, event, key, deadline_ms):
        if self.stopped():
            raise Refusal("adaptive_stopped_before_input")
        session = self.get("session", self.intent["session_id"])
        row = self.ledger.get("2026-10-01")
        Consumer.check_session(row, session)
        if session.get("status") != "idle" or session.get("required_actions"):
            raise Refusal("adaptive_input_requires_idle_session")
        self.ledger.bridge.call("adaptive_claim", phase=phase, request_digest=digest(event), deadline_ms=deadline_ms)
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        if self.stopped() or control.get("enabled") is not True or self.clock().timestamp() * 1000 >= deadline_ms:
            raise Refusal("adaptive_stopped_disabled_or_expired_before_input")
        self.api.api.sessions.events.create(self.intent["session_id"], events=[event], idempotency_key=key)

    def qa_input(self, sid, event, key, day, request_digest, deadline_ms):
        if sid != self.intent["session_id"] or request_digest != digest(event):
            raise Refusal("adaptive_qa_binding_invalid")
        self.input("qa", event, key, deadline_ms)

    def cancel(self, sid, key):
        self.ledger.bridge.call("assert_lease")
        if sid != self.intent["session_id"]:
            raise Refusal("adaptive_cancel_binding_invalid")
        return self.api.cancel(sid, key)


class TestConsumer(Consumer):
    def refresh_crm(self):
        snapshot = self.ledger.bridge.call("read_crm")
        self.ledger.write_json("crm.json", snapshot)
        save_json(self.config["crm_snapshot"], snapshot)
        return crm_snapshot(self.config["crm_snapshot"], self.clock())


def daily_config(cache):
    config = read_json(Path(__file__).with_name("adaptive-daily.config.example.json"))
    return {**config, "crm_snapshot": str(cache / "crm.json"), "expected_agent_instructions_sha256": INSTRUCTIONS}


def spend_observation(api, row):
    # Sum only this test's turns; reasoning is already part of output_tokens.
    turns = api.listing("turns", row["session_id"])
    return discovery.model_cost_observation(turns)


def stage(bridge, profile, source, cache, api, clock):
    origin = bridge.call("adaptive_origin")
    original = origin["row"]
    snapshot = bridge.call("read_crm")
    save_json(cache / "crm.json", snapshot)
    crm_snapshot(cache / "crm.json", clock())
    preflight(api, INSTRUCTIONS)
    session = api.get("session", original["session_id"])
    environment = api.get("environment", original["environment_id"])
    turns = api.listing("turns", original["session_id"])
    intent = adaptive.prepare(profile, original, snapshot, source, session=session, environment=environment,
                              turns=turns, now=clock())
    if intent["admission_blockers"]:
        return {"state": "admission_blocked", "admission_blockers": intent["admission_blockers"], "provider_mutations": 0}
    started = clock()
    row = {k: copy.deepcopy(original[k]) for k in (
        "date", "metadata", "session_id", "environment_id", "research_contract_version", "knowledge_context",
        "knowledge_context_digest", "refresh_policy", "refresh_policy_digest")}
    row.update(test_id=TEST_ID, test_intent=intent, daily_blob=origin["blob"],
               run_key="blueprint-adaptive:" + TEST_ID, state="research_input_unresolved", started_at=started.isoformat(),
               discovery_profile="adaptive-sites-v1", total_runtime_seconds=1800, research_runtime_seconds=1200,
               research_deadline_ms=int((started + timedelta(seconds=1200)).timestamp() * 1000),
               turn_id=None, cleanup_required=True, cancel_attempted=False, usage=None, delivery={},
               cost_status="unknown_pending_total_billing_reconciliation", crm_snapshot=snapshot)
    return row


def invoke(bridge, profile, source, cache, api, *, execute=False, clock=lambda: datetime.now(timezone.utc),
           stopped=lambda: False, sleep=time.sleep):
    ledger = TestLedger(bridge)
    with ledger.lock():
        row = ledger.get("2026-10-01")
        newly_staged = row is None
        if newly_staged:
            if not execute:
                return {"state": "no_adaptive_test", "provider_mutations": 0}
            row = stage(bridge, profile, source, cache, api, clock)
            if row["state"] == "admission_blocked":
                return row
            ledger.put(row)  # Complete original binding, intent and event before claim/POST.
            ledger.write_json("crm.json", row["crm_snapshot"])
        save_json(cache / "crm.json", row["crm_snapshot"])
        scoped = TestProvider(api, ledger, row["test_intent"], clock, stopped)
        runner = Runner(ledger, daily_config(cache), scoped, clock=clock)
        consumer = TestConsumer(ledger, runner.config, scoped, clock=clock, stopped=stopped)
        if newly_staged:
            if stopped():
                row.update(state="cancelled", error="stopped_before_input")
                ledger.put(row)
                return summary(row)
            try:
                scoped.input("research", row["test_intent"]["event"], row["test_intent"]["idempotency_key"], row["research_deadline_ms"])
                row["state"] = "running"
                ledger.put(row)
            except Exception:  # noqa: BLE001 - keep an uncertain attempt; recovery never resends
                row["error"] = "adaptive_input_reply_unresolved"
                ledger.put(row)
        observation_until = instant(row["started_at"]) + timedelta(seconds=1860)
        while row["state"] not in FINAL and clock() < observation_until:
            active = not stopped() and bridge.call("control").get("enabled") is True
            try:
                discovery.preserve_estimate(row, "model_cost_estimate", spend_observation(scoped, row))
            except Exception:  # noqa: BLE001 - unknown usage is a stop, never zero
                row["model_cost_estimate"] = {"known": False, "estimate_usd": None, "hard_total_cap": False}
            estimate = row["model_cost_estimate"]
            admitted = active and estimate["known"] and Decimal(estimate["estimate_usd"]) < 8
            if row["state"] == "awaiting_review":
                if not admitted and not row.get("qa"):
                    row.update(state="test_qa_blocked", error="adaptive_spend_unknown_exhausted_or_disabled")
                    ledger.put(row)
                    break
                if not admitted:
                    consumer.cancel(row, "adaptive_spend_unknown_exhausted_or_disabled")
                decision = consumer.qa(row)
                if decision:
                    row.update(state="test_qa_validated", test_qa_decision=decision)
                elif row["qa"]["state"] == "qa_blocked":
                    row["state"] = "test_qa_blocked"
                ledger.put(row)
            else:
                if not admitted:
                    runner.cancel(row, "adaptive_spend_unknown_exhausted_or_disabled")
                row = runner.observe(row)
            if row["state"] not in FINAL:
                sleep(3)
        if row["state"] not in FINAL:
            if row.get("qa"):
                consumer.cancel(row, "adaptive_observation_deadline")
            else:
                runner.cancel(row, "adaptive_observation_deadline")
            row["error"] = "adaptive_terminal_state_unresolved"
            ledger.put(row)
        return summary(row)


def summary(row):
    accepted = row.get("test_qa_decision", {}).get("accepted_keys", [])
    return {"test_id": TEST_ID, "state": row["state"], "error": row.get("error"),
            "session_id": row["session_id"], "turn_id": row.get("turn_id"),
            "qa_turn_id": row.get("qa", {}).get("turn_id"), "accepted_new_count": len(accepted),
            "shortfall": max(0, 10 - len(accepted)), "cleanup_required": True,
            "publication_performed": False, "hard_total_cap_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["execute", "reconcile", "status"])
    parser.add_argument("--profile", default=str(adaptive.PROFILE))
    args = parser.parse_args(argv)
    bridge = api = None
    stop = {"requested": False}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.update(requested=True))
    try:
        if args.command != "status":
            verify_process_watchdog()  # Refuse before any connector/provider action.
        manifest = read_json(Path(__file__).resolve().parents[2] / "manifest.json")
        bridge = Bridge()
        if args.command == "status":
            row = bridge.call("adaptive_get")
            result = summary(row) if row else {"state": "no_adaptive_test", "provider_mutations": 0}
        else:
            api = Provider(os.environ.get("OPENAI_API_KEY", ""))
            with tempfile.TemporaryDirectory(prefix="blueprint-adaptive-") as root:
                result = invoke(bridge, read_json(args.profile), manifest["source_commit"], Path(root), api,
                                execute=args.command == "execute", stopped=lambda: stop["requested"])
        print(canonical(result))
        return 0 if result["state"] in {"test_qa_validated", "no_adaptive_test"} else 1
    except Exception as exc:  # noqa: BLE001 - never expose upstream exception bodies/credentials
        print(canonical({"state": "blocked", "error": str(exc) if isinstance(exc, Refusal) else "adaptive_runtime_unavailable"}))
        return 1
    finally:
        if api:
            api.client.close()
        if bridge:
            bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
