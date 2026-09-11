"""Durable worker scheduling and verified webhook wakeups for reasoning tasks.

Queue admission never makes a model call. Provider events only wake an already
owned task; workers inspect canonical state and independently enforce authority.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Mapping, Protocol

from .contracts import AgentExecutionError, AgentTask, canonical_json
from .journal import AgentJournal, TERMINAL_STATES


class AgentRuntime(Protocol):
    runtime_id: str

    def admit(self, task: AgentTask) -> dict: ...
    def step(self, task_id: str) -> dict: ...
    def cancel(self, task_id: str) -> dict: ...
    def cleanup(self, task_id: str) -> dict: ...


class AgentTaskService:
    def __init__(
        self, *, journal: AgentJournal,
        runtime_for_task: Callable[[AgentTask], AgentRuntime],
        validate_admission: Callable[[AgentTask], None],
        clock: Callable[[], float] = time.time,
        poll_seconds: float = 5,
    ):
        if not 0.1 <= poll_seconds <= 60:
            raise ValueError("agent_service_poll_interval_invalid")
        self.journal, self.runtime_for_task = journal, runtime_for_task
        self.validate_admission, self.clock = validate_admission, clock
        self.poll_seconds = poll_seconds

    def enqueue(self, task: AgentTask) -> dict:
        task = task.snapshot()
        self.validate_admission(task)
        runtime = self.runtime_for_task(task)
        if runtime.runtime_id != task.admission.runtime:
            raise AgentExecutionError("agent_service_runtime_mismatch")
        state = runtime.admit(task)
        self.journal.wake(task.task_id, due_at=self.clock())
        return state

    def recover(self) -> int:
        """Recover persisted work even when its last webhook was never delivered."""
        return self.journal.recover_wakeups(now=self.clock())

    def tick(self) -> dict | None:
        claimed = self.journal.claim_wakeup(now=self.clock(), visibility_timeout=60)
        if claimed is None:
            return None
        task_id, attempts = claimed["task_id"], claimed["attempt"]
        state = self.journal.task(task_id)
        task = AgentTask.model_validate(state["task"])
        error = None
        try:
            runtime = self.runtime_for_task(task)
            if runtime.runtime_id != task.admission.runtime:
                raise AgentExecutionError("agent_service_runtime_mismatch")
            if state["cleanup_state"] == "pending":
                state = runtime.cleanup(task_id)
            else:
                state = runtime.step(task_id)
        except Exception as exc:
            # Persist only stable internal codes, not provider bodies or keys.
            code = str(exc) if isinstance(exc, AgentExecutionError) else "agent_worker_step_failed"
            error = code if re.fullmatch(r"[a-z0-9_]{1,160}", code) else "agent_worker_step_failed"
            state = self.journal.task(task_id)
        delay = min(60, self.poll_seconds * 2 ** min(attempts - 1, 4)) if error else self.poll_seconds
        self.journal.schedule_next(task_id, due_at=self.clock() + delay, attempt=attempts, error_code=error)
        receipt = {"schema_version": "blueprint_agent_worker_step.v1", "task_id": task_id,
                   "runtime": task.admission.runtime, "state": state["state"],
                   "cleanup_state": state["cleanup_state"], "attempt": attempts,
                   "error_code": error, "observed_at": self.clock(), "proof_effect": "none"}
        self.journal.record_event(f"worker_{task_id}_{attempts}", receipt)
        return receipt

    def cancel(self, task_id: str) -> dict:
        # Durable intent is the HTTP action; the worker owns remote cancellation.
        self.journal.request_cancel(task_id, "agent_cancel_requested")
        self.journal.wake(task_id, due_at=self.clock())
        return self.journal.task(task_id)

    def request_cleanup(self, task_id: str) -> dict:
        owner_id = self.journal.session_owner(task_id)["task_id"]
        with self.journal.own_task(owner_id):
            state = self.journal.task(task_id)
            if (state["state"] not in TERMINAL_STATES or self.journal.unsettled_operations(task_id)
                    or self.journal.successor(task_id) is not None):
                raise AgentExecutionError("agent_cleanup_not_ready")
            self.journal.cleanup_state(task_id, "pending")
            self.journal.wake(task_id, due_at=self.clock())
            return self.journal.task(task_id)

    def receive_webhook(
        self, payload: bytes, headers: Mapping[str, str], *, signing_secret: str,
    ) -> dict:
        if not signing_secret or len(payload) > 64_000:
            raise AgentExecutionError("agent_webhook_configuration_or_size_invalid")
        # Signature verification is local and does not need an API credential.
        # An explicit dummy credential avoids reading ambient secrets.
        from openai import OpenAI, InvalidWebhookSignatureError

        try:
            with OpenAI(api_key="unused-local-signature-verification", max_retries=0) as verifier:
                verifier.webhooks.verify_signature(payload=payload, headers=headers, secret=signing_secret)
        except (InvalidWebhookSignatureError, ValueError) as exc:
            raise AgentExecutionError("agent_webhook_signature_invalid") from exc
        try:
            event = json.loads(payload)
            canonical_json(event)
            event_id, event_type, session_id = event["id"], event["type"], event["data"]["id"]
            if (not isinstance(event_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,192}", event_id) is None
                    or not isinstance(session_id, str)
                    or re.fullmatch(r"[A-Za-z0-9_-]{1,256}", session_id) is None
                    or event_type not in {"agent.session.created", "agent.session.action_required",
                                          "agent.session.in_progress", "agent.session.idle", "agent.session.failed"}):
                raise ValueError("unsupported_event")
        except (ValueError, TypeError, KeyError) as exc:
            raise AgentExecutionError("agent_webhook_payload_invalid") from exc
        members = self.journal.session_tasks(session_id)
        active = [member for member in members if member["state"] not in TERMINAL_STATES]
        if len(active) > 1:
            raise AgentExecutionError("agent_webhook_session_ownership_ambiguous")
        if not members:
            return {"accepted": False, "reason": "session_not_owned"}
        is_new = self.journal.record_event("webhook_" + event_id, {
            "provider_event_id": event_id, "event_type": event_type, "session_id": session_id,
        })
        if active:
            # Duplicates can wake safely; event identity cannot enact a tool or
            # turn outcome and state is always fetched again by the worker.
            self.journal.wake(active[0]["task_id"], due_at=self.clock())
        return {"accepted": True, "duplicate": not is_new, "woken": bool(active)}
