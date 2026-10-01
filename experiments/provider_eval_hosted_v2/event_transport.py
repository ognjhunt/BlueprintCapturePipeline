"""Durable keys for future event sends; never key a historical unknown send."""

import re
import time

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, canonical_json
from blueprint_pipeline.agent_execution.openai_transport import AgentTransportError
from experiments.provider_eval_recovery.harness import digest, write_once, read_json


def safe_identifier(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:\[\]-]{1,256}", value) else None


def sdk_client(api_key, project_id, *, http_client=None):
    import httpx2
    from openai import OpenAI
    return OpenAI(api_key=api_key, project=project_id, base_url="https://api.openai.com/v1",
        max_retries=0, timeout=20,
        http_client=http_client or httpx2.Client(follow_redirects=False, timeout=20))


class DurableEvents:
    """Official SDK event POSTs only. Other routes keep the existing transport."""
    def __init__(self, *, fallback, journal, receipt_root, client_factory, clock=time.time, timer=time.monotonic):
        self.fallback, self.journal, self.root = fallback, journal, receipt_root
        self.client_factory, self.clock, self.timer = client_factory, clock, timer
        self.project_id = fallback.project_id

    def prepare(self, path, body, *, task=None):
        session_id = path.split("/")[-2]
        active = [s for s in self.journal.session_tasks(session_id) if s["state"] not in {"completed", "failed", "cancelled"}]
        if task is None:
            if len(active) != 1:
                raise AgentExecutionError("future_event_single_owned_task_required")
            task = active[0]
        else:
            task = self.journal.task(task.task_id)
        if self.journal.session_owner(task["task_id"])["session_id"] != session_id:
            raise AgentExecutionError("future_event_owned_session_required")
        events = body.get("events")
        if not isinstance(events, list) or not events or any(not isinstance(e, dict) for e in events):
            raise AgentExecutionError("future_event_payload_invalid")
        cancelling = all(e.get("type") == "agent.session.input.cancel" for e in events)
        if not cancelling and (task["cancel_requested"] or self.clock() >= task["task"]["deadline"]):
            raise AgentExecutionError("future_event_task_cancelled_or_expired")
        message = any(e.get("type") == "agent.session.input.message" for e in events)
        continuation = self.journal.continuation(task["task_id"])
        message_proof = self.journal.event("hosted_keyed_message_" + task["task_id"])
        if continuation and continuation["delivery_state"] != "pending" and message_proof is None:
            raise AgentExecutionError("historical_unkeyed_continuation_remains_unresolved_no_dispatch")
        slot = {"session_id": session_id, "task_id": task["task_id"], "task_digest": task["task_digest"],
            "events": [{"type": e.get("type"), "turn_id": e.get("turn_id"), "call_id": e.get("call_id")} for e in events]}
        slot_sha = digest(slot)
        payload = canonical_json(body)
        proof = {"schema": "hosted_future_event_intent.v1", "slot": slot, "payload": payload,
            "payload_sha256": digest(body), "idempotency_key": "bp-events-" + digest({"slot": slot, "payload": payload}),
            "max_transmissions": 2, "retry_window_seconds": 60, "original_key_required_for_retry": True}
        event_id = "hosted_future_event_" + slot_sha
        # Stable slot plus immutable payload rejects a changed retry before HTTP.
        self.journal.record_event(event_id, proof)
        write_once(self.root / slot_sha / "intent.json", proof)
        if message:
            self.journal.record_event("hosted_keyed_message_" + task["task_id"],
                {"intent_event_id": event_id, "payload_sha256": proof["payload_sha256"], "task_digest": task["task_digest"]})
        return slot_sha, proof

    def request(self, method, path, *, body=None, query=None):
        if method != "POST" or not re.fullmatch(r"/agents/sessions/[A-Za-z0-9_-]{1,256}/events", path):
            return self.fallback.request(method, path, body=body, query=query)
        if query:
            raise AgentExecutionError("future_event_query_not_admitted")
        slot_sha, proof = self.prepare(path, body)
        if read_json(self.root / slot_sha / "intent.json") != proof:
            raise AgentExecutionError("future_event_intent_integrity_failure")
        with self.journal.own_operation("future-event:" + slot_sha):
            return self._send(path, slot_sha, proof)

    def _send(self, path, slot_sha, proof):
        attempt = None
        for n in range(1, 3):
            sent = self.journal.event(f"hosted_future_send_{slot_sha}_{n}")
            outcome = self.journal.event(f"hosted_future_reply_{slot_sha}_{n}")
            if outcome and outcome.get("accepted") is True:
                return {}  # Already accepted input is observed, not resubmitted.
            if outcome and outcome.get("definitively_rejected") is True:
                raise AgentTransportError(outcome["code"], status=outcome["http_status"], diagnostics=outcome)
            if sent is None:
                attempt = n
                break
        if attempt is None:
            raise AgentExecutionError("future_event_two_transmissions_exhausted_reconcile_only")
        first = self.journal.event(f"hosted_future_send_{slot_sha}_1")
        if first is not None and self.clock() >= first["at"] + proof["retry_window_seconds"]:
            raise AgentExecutionError("future_event_retry_window_expired_reconcile_only")
        intent = {"intent_event_id": "hosted_future_event_" + slot_sha, "attempt": attempt,
            "at": self.clock(), "idempotency_key": proof["idempotency_key"], "payload_sha256": proof["payload_sha256"]}
        self.journal.record_event(f"hosted_future_send_{slot_sha}_{attempt}", intent)
        write_once(self.root / slot_sha / f"send_{attempt}.json", intent)
        started = self.timer()
        try:
            with self.client_factory() as client:
                reply = client.beta.agents.sessions.events.with_raw_response.create(path.split("/")[-2],
                    events=read_json_payload(proof), idempotency_key=proof["idempotency_key"])
                result = {"accepted": True, "http_status": reply.status_code,
                    "request_id": safe_identifier(reply.headers.get("x-request-id")),
                    "exception_type": None, "code": None, "definitively_rejected": False}
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if type(status) is not int or not 100 <= status <= 599:
                status = None
            error = AgentTransportError("agents_api_http_error" if status is not None else "agents_api_connection_uncertain", status=status)
            result = {"accepted": False, "http_status": status,
                "request_id": safe_identifier(getattr(exc, "request_id", None)),
                "exception_type": safe_identifier(type(exc).__name__), "code": error.code,
                "cause_type": safe_identifier(type(exc.__cause__).__name__) if exc.__cause__ is not None else None,
                "definitively_rejected": error.definitively_rejected}
        result["latency_seconds"] = max(0, self.timer() - started)
        result["intent_event_id"] = "hosted_future_event_" + slot_sha
        self.journal.record_event(f"hosted_future_reply_{slot_sha}_{attempt}", result)
        write_once(self.root / slot_sha / f"reply_{attempt}.json", result)
        if not result["accepted"]:
            raise AgentTransportError(result["code"], status=result["http_status"], diagnostics=result)
        return {}


def read_json_payload(proof):
    import json
    return json.loads(proof["payload"])["events"]
