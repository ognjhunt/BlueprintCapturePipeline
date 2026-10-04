"""Optional Render store. Existing Firebase Admin is used through a private pipe."""
import base64
import hashlib
import json
import os
import re
import select
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from tools.daily_research.runner import (
    AGENT,
    MODEL,
    PROJECT,
    TEMPLATE,
    Provider,
    Refusal,
    canonical,
)


class Bridge:
    def __init__(self, node="node", script=None):
        self.closed = False
        self.broken = False
        self.process = subprocess.Popen(
            [node, str(script or Path(__file__).with_name("firestore_bridge.mjs"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
            env={k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "FIREBASE_SERVICE_ACCOUNT_JSON", "NOTION_API_TOKEN", "NOTION_API_KEY", "BLUEPRINT_DAILY_RESEARCH_LEARNING_MODULE", "BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED"}},
        )

    def call(self, op, **fields):
        if self.broken or self.closed:
            raise Refusal("firestore_bridge_unavailable")
        try:
            self.process.stdin.write(canonical({"op": op, **fields}) + "\n")
            self.process.stdin.flush()
            timeout = 120 if op in {"cleanup_archive", "cleanup_archive_verify"} else 35
            if not select.select([self.process.stdout], [], [], timeout)[0]:
                self.broken = True
                self.close()
                raise Refusal("firestore_bridge_deadline")
            result = json.loads(self.process.stdout.readline())
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                raise ValueError("invalid protocol frame")
            if result.get("ok") is not True:
                code = result.get("error", "firestore_bridge_unavailable")
                raise Refusal(code if isinstance(code, str) and code.replace("_", "").isalnum() else "firestore_bridge_unavailable")
            return result["value"]
        except (OSError, ValueError, KeyError):
            self.broken = True
            self.close()
            raise Refusal("firestore_bridge_unavailable") from None

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


class FirestoreLedger:
    def __init__(self, bridge):
        self.bridge = bridge

    @contextmanager
    def lock(self):
        self.bridge.call("acquire")
        try:
            yield
        finally:
            self.bridge.call("release")

    def rows(self):
        return self.bridge.call("rows")

    def get(self, day):
        return self.bridge.call("get", day=day)

    def learning_context(self, day, *, allow_create=True):
        return self.bridge.call("learning_context", day=day, allow_create=allow_create)

    def company_history_binding(self):
        control = self.bridge.call("control")
        return control.get("learning") if isinstance(control, dict) else None

    def contact_research_context(self, day):
        return self.bridge.call("contact_research_context", day=day)

    def reconcile_contact_research(self, day):
        return self.bridge.call("contact_research_reconcile", day=day)

    def put(self, row):
        from tools.daily_research import search
        if row.get("search_provider") == search.PROFILE and len(canonical(row).encode()) > search.MAX_RECORD:
            raise Refusal("research_tool_record_resource_ceiling")
        self.bridge.call("put", row=row)

    def write_bytes(self, name, value):
        self.bridge.call("file_put", name=name, bytes=base64.b64encode(value).decode("ascii"))

    def write_json(self, name, value):
        self.write_bytes(name, (canonical(value) + "\n").encode())

    def read_bytes(self, name):
        try:
            return base64.b64decode(self.bridge.call("file_get", name=name), validate=True)
        except Refusal as exc:
            if str(exc) == "firestore_file_missing":
                raise FileNotFoundError(name) from None
            raise


class FencedProvider(Provider):
    def __init__(self, ledger, api_key):
        super().__init__(api_key)
        self.ledger = ledger

    def create(self, payload):
        self.ledger.bridge.call("create_check", day=payload["metadata"]["run_key"].split(":", 1)[1], metadata=payload["metadata"])
        return super().create(payload)

    def expansion_context(self, row, name):
        from tools.daily_research import expansion
        from tools.daily_research.exa_transport import ExaTransport, ExaTransportError
        self.tool_admit(row, "research")
        if name == expansion.START and any(previous.get("exa_expansion")
                and not previous["exa_expansion"].get("terminal_receipt") for previous in self.ledger.rows()):
            return {"unavailable_reason": "expansion_original_run_or_ack_pending_no_new_start"}
        control = self.ledger.bridge.call("control")
        allocation = control.get("exa_expansion_allocation")
        allocation_status = expansion.allocation_diagnostic(row, allocation)
        if name == expansion.START and not isinstance(allocation, dict):
            return {"unavailable_reason": "expansion_remaining_all_in_allocation_unverified",
                    "allocation_status": allocation_status}
        key = os.environ.get("EXA_API_KEY")
        if not key:
            return {"unavailable_reason": "expansion_worker_exa_binding_missing", "allocation": allocation,
                    "allocation_status": allocation_status}

        def retain(receipt):
            from tools.daily_research.runner import digest
            raw = (canonical(receipt) + "\n").encode()
            receipt_hash = digest(receipt)
            filename = row["date"] + "-exa-http-" + receipt_hash + ".json"
            claim = row.get("exa_expansion")
            if claim and receipt.get("operation") == "tools/call":
                request = json.loads(base64.b64decode(receipt["request_body_base64"], validate=True))
                if request.get("params") == {"name": "agent_run", "arguments": claim["intent"]["request"]}:
                    filename = row["date"] + "-exa-" + claim["intent_sha256"] + "-start-http.json"
            try:
                existing = self.ledger.read_bytes(filename)
            except FileNotFoundError:
                self.ledger.write_bytes(filename, raw)
            else:
                if existing != raw:
                    raise Refusal("expansion_transport_receipt_conflict")
            refs = row.setdefault("exa_transport_receipts", [])
            ref = {"file": filename, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            if ref not in refs:
                refs.append(ref)
                self.ledger.put(row)

        transport = ExaTransport(receipt_sink=retain)
        try:
            schema = transport.discover()
        except ExaTransportError:
            return {"unavailable_reason": "expansion_authenticated_catalog_unavailable"}
        return {"transport": transport, "allocation": allocation, "tool_schema": schema,
                "allocation_status": allocation_status}

    def expansion_admit(self, row, phase):
        self.tool_admit(row, phase)
        claim = row.get("exa_expansion")
        if claim and not claim.get("run_id"):
            control = self.ledger.bridge.call("control")
            if control.get("exa_expansion_allocation") != claim["intent"]["allocation"]:
                raise Refusal("expansion_allocation_changed_before_submission")

    def cancel(self, session_id, run_key):
        self.ledger.bridge.call("assert_lease")
        return super().cancel(session_id, run_key)

    def delete_session(self, session_id, day, binding_digest):
        self.cleanup_delete_phase = "preconditions"
        row = self.ledger.get(day)
        if not row or row.get("session_id") != session_id:
            raise Refusal("cleanup_session_binding_changed")
        from tools.daily_research.render import cleanup_inventory
        inventory = cleanup_inventory(self, row, contents=False)
        import hashlib
        for name in ("provider-items.json", "provider-artifacts.json"):
            expected = next((o for o in row["cleanup"]["archive"]["objects"] if o["name"].endswith("/" + name)), None)
            if not expected or expected["sha256"] != hashlib.sha256(inventory[name]).hexdigest():
                raise Refusal("cleanup_archived_inventory_changed")
        self.ledger.bridge.call("cleanup_delete_check", day=day, binding_digest=binding_digest)
        if getattr(self, "stopped", lambda: False)():
            raise Refusal("cleanup_stopped_before_delete")
        self.cleanup_delete_phase = "provider_submission"
        return super().delete_session(session_id, day, binding_digest)

    def tool_admit(self, row, phase):
        super().tool_admit(row, phase)
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        if (control.get("enabled") is not True or control.get("config", {}).get("search_provider") != row.get("search_provider")
                or row.get("mcp_profile") and control.get("config", {}).get("mcp_profile") != row["mcp_profile"]
                or row.get("expansion_profile") and control.get("config", {}).get("expansion_profile") != row["expansion_profile"]
                or phase in {"qa", "repair", "publication"} and control.get("workflow", {}).get("enabled") is not True):
            raise Refusal("research_tool_disabled_or_profile_changed")
        if (
                control.get("config", {}).get("recurring_budget_authority_reference") != row["recurring_budget_authority_reference"]
                or control.get("config", {}).get("soft_target_usd") != row["soft_target_usd"]):
            raise Refusal("research_tool_budget_authority_changed")
        if phase == "publication" and control.get("workflow") != row["publication"]["workflow_authority"]:
            raise Refusal("publication_agent_authority_changed")
        if row.get("history_profile") == "agent-history-v1" and control.get("learning") != row["history_binding"]:
            raise Refusal("company_history_authority_changed")

    def publication_input(self, session_id, event, key, day, request_digest, deadline_ms):
        from tools.daily_research.consumer import Consumer, qa_deadline
        from tools.daily_research.runner import digest
        self.publication_input_phase = "preconditions"
        row = self.ledger.get(day)
        phase = row["publication"]
        if (row.get("publication_profile") != "agent-owned-v1" or row["state"] != "reviewed"
                or row["qa"]["state"] != "validated" or session_id != row["session_id"]
                or key != row["run_key"] + ":publication" or phase["idempotency_key"] != key
                or digest(event) != phase["request_digest"] or request_digest != phase["request_digest"]
                or deadline_ms != phase["deadline_ms"] or deadline_ms != int(qa_deadline(row, {}).timestamp() * 1000)
                or json.loads(self.ledger.read_bytes(phase["input_file"])) != event):
            raise Refusal("publication_agent_input_not_admitted")
        def guard():
            session = self.get("session", session_id)
            Consumer.check_session(row, session)
            turns = self.listing("turns", session_id)
            if (session.get("status") != "idle" or session.get("required_actions")
                    or {t["id"] for t in turns} != set(phase["baseline_turn_ids"]) or any(t.get("subagent_id") for t in turns)):
                raise Refusal("publication_agent_turn_scope_changed")
            self.tool_admit(row, "publication")
            control = self.ledger.bridge.call("control")
            if (getattr(self, "stopped", lambda: False)() or control.get("workflow") != phase["workflow_authority"]
                    or getattr(self, "clock", lambda: datetime.now(timezone.utc))().timestamp() * 1000 >= deadline_ms):
                raise Refusal("publication_agent_input_not_admitted")
        guard()
        self.ledger.bridge.call("publication_input_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        guard()
        self.publication_input_phase = "provider_submission"
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)

    def qa_input(self, session_id, event, key, day, request_digest, deadline_ms):
        self.ledger.bridge.call("qa_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        if datetime.now(timezone.utc).timestamp() * 1000 >= deadline_ms:
            raise Refusal("agent_qa_total_runtime_exhausted")
        self.recovered_qa_action_guard(session_id, day, deadline_ms)
        self.qa_input_phase = "provider_submission"
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)

    def qa_correction_input(self, session_id, event, key, day, request_digest, deadline_ms, number):
        from tools.daily_research.consumer import qa_deadline
        from tools.daily_research.runner import digest
        row = self.ledger.get(day)
        current = (row.get("qa", {}).get("corrections") or [{}])[-1]
        if (number not in {1, 2} or row.get("session_id") != session_id or current.get("number") != number
                or current.get("request_digest") != request_digest or digest(event) != request_digest
                or current.get("idempotency_key") != key or key != row["run_key"] + f":qa:correction:{number}"
                or current.get("deadline_ms") != deadline_ms or int(qa_deadline(row, {}).timestamp() * 1000) != deadline_ms
                or current.get("input_file") != f"{day}-qa-correction-{number}-input.json"
                or json.loads(self.ledger.read_bytes(current["input_file"])) != event):
            raise Refusal("qa_correction_input_not_admitted")
        self.qa_correction_action_guard(day, deadline_ms)
        self.ledger.bridge.call("qa_correction_check", day=day, request_digest=request_digest,
                               deadline_ms=deadline_ms, number=number)
        self.qa_correction_action_guard(day, deadline_ms)
        self.qa_correction_input_phase = "provider_submission"
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)

    def qa_correction_action_guard(self, day, deadline_ms):
        import hashlib

        from tools.daily_research.consumer import Consumer
        row = self.ledger.get(day)
        current = row["qa"]["corrections"][-1]
        prior = current["previous_review"]
        if hashlib.sha256(self.ledger.read_bytes(prior["artifact_file"])).hexdigest() != prior["artifact_digest"]:
            raise Refusal("qa_correction_source_artifact_changed")
        session = self.get("session", row["session_id"])
        Consumer.check_session(row, session)
        turns = self.listing("turns", row["session_id"])
        if (session.get("status") != "idle" or session.get("required_actions")
                or {turn["id"] for turn in turns} != set(current["baseline_turn_ids"])
                or any(turn.get("subagent_id") or turn["status"] not in (
                       {"completed", "failed", "cancelled"} if row.get("validation_repair_outcome")
                       and turn["id"] in row["qa"]["baseline_turn_ids"] and turn["id"] != row["turn_id"] else {"completed"})
                       or turn.get("agent_id") not in (None, AGENT) or turn.get("session_id") not in (None, row["session_id"]) for turn in turns)):
            raise Refusal("qa_correction_session_scope_changed")
        self.qa_correction_final_guard(row, deadline_ms)

    def qa_correction_final_guard(self, row, deadline_ms):
        from tools.daily_research.runner import digest
        current = row["qa"]["corrections"][-1]
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        if (getattr(self, "stopped", lambda: False)() or control.get("enabled") is not True
                or control.get("workflow", {}).get("enabled") is not True
                or current["authority_reference"] != control["workflow"].get("qa_authority_reference")
                or (not row.get("qa_retry_continuation") and current["authority_reference"] != row["qa"].get("submission_binding", {}).get("authority_reference"))
                or row["qa"].get("cancel_attempted") or current["state"] != "input_unresolved"
                or row["qa"]["state"] != "qa_correction_input_unresolved"
                or digest(row["packet"]) != row["packet_digest"]
                or getattr(self, "clock", lambda: datetime.now(timezone.utc))().timestamp() * 1000 >= deadline_ms):
            raise Refusal("qa_correction_stopped_disabled_expired_or_authority_changed")
        if (row.get("search_provider") == "perplexity-fast-v1" and (control.get("config", {}).get("search_provider") != row.get("search_provider")
                or control.get("config", {}).get("recurring_budget_authority_reference") != row["recurring_budget_authority_reference"]
                or control.get("config", {}).get("soft_target_usd") != row["soft_target_usd"])):
            raise Refusal("research_tool_budget_authority_changed")

    def recovered_qa_action_guard(self, session_id, day, deadline_ms, *, origin_guard=None):
        row = self.ledger.get(day)
        if not row.get("qa_continuation"):
            return
        from tools.daily_research.consumer import Consumer
        session = self.get("session", session_id)
        Consumer.check_session(row, session)
        turns = self.listing("turns", session_id)
        if (session.get("status") != "idle" or session.get("required_actions")
                or {t["id"] for t in turns} != set(row["qa"]["baseline_turn_ids"])
                or any(t.get("subagent_id") or t["status"] != "completed" for t in turns)):
            raise Refusal("recovered_qa_session_or_turn_changed")
        if origin_guard:
            origin_guard()
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        if (getattr(self, "stopped", lambda: False)() or control.get("enabled") is not True
                or control.get("workflow", {}).get("enabled") is not True
                or getattr(self, "clock", lambda: datetime.now(timezone.utc))().timestamp() * 1000 >= deadline_ms):
            raise Refusal("recovered_qa_stopped_disabled_or_expired")

    def qa_retry_input(self, session_id, event, key, day, request_digest, deadline_ms, number):
        from tools.daily_research import qa_retry
        from tools.daily_research.consumer import qa_deadline
        from tools.daily_research.runner import digest
        row = self.ledger.get(day)
        if (not (row.get("qa_retry_continuation") or row.get("qa", {}).get("submission_binding")) or row.get("session_id") != session_id
                or key != row["run_key"] + ":qa" or digest(event) != request_digest
                or qa_retry.original_event(self.ledger, row) != event
                or int(qa_deadline(row, {}).timestamp() * 1000) != deadline_ms):
            raise Refusal("qa_retry_input_not_admitted")
        if not row.get("qa_retry_continuation"):
            qa_retry.check_submission_binding(row, deadline_ms)
        self.qa_retry_action_guard(row, deadline_ms)
        self.ledger.bridge.call("qa_retry_check", day=day, request_digest=request_digest,
                                deadline_ms=deadline_ms, number=number)
        self.qa_retry_action_guard(self.ledger.get(day), deadline_ms)
        self.qa_input_phase = "provider_submission"
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)

    def qa_retry_action_guard(self, row, deadline_ms):
        from tools.daily_research import qa_retry
        if not qa_retry.reconcile(self, self.ledger, row)["clear"]:
            raise Refusal("qa_retry_saved_work_changed")
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        if not row.get("qa_retry_continuation"):
            binding = qa_retry.check_submission_binding(row, deadline_ms)
            if binding["authority_reference"] != control.get("workflow", {}).get("qa_authority_reference"):
                raise Refusal("qa_retry_workflow_authority_changed")
        if (getattr(self, "stopped", lambda: False)() or control.get("enabled") is not True
                or control.get("workflow", {}).get("enabled") is not True
                or getattr(self, "clock", lambda: datetime.now(timezone.utc))().timestamp() * 1000 >= deadline_ms):
            raise Refusal("qa_retry_stopped_disabled_or_expired")

    def repair_input(self, session_id, event, key, day, request_digest, deadline_ms):
        # RepairLoop holds the existing fenced lease and has durably consumed
        # this revision's input claim. A restarted observer never sends it again.
        from tools.daily_research.runner import digest
        self.ledger.bridge.call("assert_lease")
        row = self.ledger.get(day)
        current = row.get("validation_repairs", [{}])[-1]
        control = self.ledger.bridge.call("control")
        if (row.get("session_id") != session_id or current.get("input_attempted") is not True
                or current.get("state") != "input_unresolved" or current.get("request_digest") != request_digest
                or key != row.get("run_key", "") + ":repair:" + str(current.get("number"))
                or digest(event) != request_digest or current.get("deadline_ms") != deadline_ms
                or control.get("enabled") is not True or control.get("workflow", {}).get("enabled") is not True
                or getattr(self, "clock", lambda: datetime.now(timezone.utc))().timestamp() * 1000 >= deadline_ms):
            raise Refusal("validation_repair_input_not_admitted")
        self.ledger.bridge.call("repair_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        self.repair_action_guard(day, deadline_ms)
        self.repair_input_phase = "provider_submission"
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)

    def repair_action_guard(self, day, deadline_ms):
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        row = self.ledger.get(day)
        request = row.get("validation_repair_authority", {}).get("request", {})
        if (getattr(self, "stopped", lambda: False)()
                or control.get("enabled") is not True or control.get("workflow", {}).get("enabled") is not True
                or getattr(self, "clock", lambda: datetime.now(timezone.utc))().timestamp() * 1000 >= deadline_ms
                or (row.get("validation_repair_authority", {}).get("kind") == "workflow"
                    and request.get("authority_reference") != control["workflow"].get("qa_authority_reference"))):
            raise Refusal("validation_repair_stopped_disabled_expired_or_authority_changed")


def control_configuration(value):
    if (not isinstance(value, dict) or value.get("schema_version") != "blueprint.research-control.v1"
            or value.get("project_id") != PROJECT or value.get("agent_id") != AGENT
            or value.get("model") != MODEL or value.get("template_id") != TEMPLATE
            or type(value.get("enabled")) is not bool):
        raise Refusal("firestore_control_binding_invalid")
    config = value.get("config")
    if not isinstance(config, dict):
        raise Refusal("firestore_control_config_missing")
    if value["enabled"] and not re.fullmatch(r"[a-f0-9]{64}", str(config.get("expected_agent_instructions_sha256", ""))):
        raise Refusal("agent_instructions_pin_required")
    return {**config, "enabled": value["enabled"]}
