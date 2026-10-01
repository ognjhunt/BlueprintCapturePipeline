"""One reviewed same-session follow-up after the known reporting-lag cancellation."""

import argparse
from decimal import Decimal
from importlib.metadata import version
import json
from pathlib import Path
import time

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, AgentTask, canonical_json, digest as task_digest
from blueprint_pipeline.agent_execution.journal import AgentJournal, TERMINAL_STATES
from blueprint_pipeline.agent_execution.openai_transport import AgentTransportError, OpenAIAgentsHTTP
from experiments.provider_eval_recovery.harness import Ledger, digest, exclusive, read_json, write_once
from experiments.provider_eval_recovery.live_http import existing_key
from . import soft_pilot as base
from . import retry_pilot as retry

SESSION = "sess_09118c760a810004006abdc1869c9c8194844ab4a80b431ad5"
TURN = "turn_09118c760a810004006abdc18be18081948811782f9c45e2ef"
LAG_STOP = "fresh_hosted_usage_unknown_no_further_paid_work"


def validate_recovery(root, receipt, *, clock=time.time):
    retry.validate_receipt(root, base.OWNER, digest(receipt), clock=clock, readonly_previous=True)
    monitor = retry.RetryMonitor(root, receipt, clock=clock, notify=lambda _: None)
    journal = AgentJournal(monitor.path / "agent_journal")
    proof = read_json(monitor.path / "usage_recovery.json")
    commit, code = base.code_identity()
    parent = journal.task(monitor.task_id("parallel_fast"))
    if (proof.get("source_commit") != commit or proof.get("code_sha256") != code
            or proof.get("retry_receipt_sha256") != digest(receipt) or proof.get("session_id") != SESSION
            or proof.get("prior_turn_id") != TURN or proof.get("approval") != retry.APPROVAL
            or proof.get("parent_task_digest") != parent["task_digest"]
            or parent["state"] != "cancelled" or parent["session_id"] != SESSION or parent["turn_id"] != TURN
            or journal.unsettled_operations(parent["task_id"])
            or proof.get("usage_grace_seconds") != base.USAGE_GRACE_SECONDS
            or proof.get("followup_deadline") != min(proof.get("prepared_at", 0) + base.ARM_SECONDS, receipt["expires_at"])
            or proof.get("soft_target_usd") != "2.00" or proof.get("new_session_create_allowed") is not False
            or digest(read_json(monitor.path / "retained_usage_lag_stop.json")) != proof.get("prior_stop_sha256")
            or journal.event("hosted_usage_recovery") != proof):
        raise AgentExecutionError("exact_reviewed_same_session_recovery_required")
    ledger = Ledger(Path(root) / "live_journal.jsonl", "10.00")
    n = proof.get("ledger_events")
    if (type(n) is not int or not 1 <= n <= len(ledger.events)
            or ledger.events[n - 1]["sha256"] != proof.get("ledger_head")
            or ledger.exposure < Decimal("6.221400") or any(s not in {"completed", "not_accepted"} for s in ledger.states.values())):
        raise AgentExecutionError("original_retry_ledger_not_preserved")
    return proof


def prepare(root, receipt, *, transport, clock=time.time):
    """GET-only lifecycle verification. No model, search or session creation."""
    root = Path(root).resolve()
    retry.validate_receipt(root, base.OWNER, digest(receipt), clock=clock, readonly_previous=True)
    monitor = retry.RetryMonitor(root, receipt, clock=clock, notify=lambda _: None)
    journal = AgentJournal(monitor.path / "agent_journal")
    parent = journal.task(monitor.task_id("parallel_fast"))
    if (monitor.path / "usage_recovery.json").exists():
        proof = validate_recovery(root, receipt, clock=clock)
        stopped = monitor.path / "stop.json"
        if stopped.exists() and digest(read_json(stopped)) == proof["prior_stop_sha256"]:
            stopped.unlink()
        return proof
    if (parent["state"] != "cancelled" or parent["session_id"] != SESSION or parent["turn_id"] != TURN
            or journal.unsettled_operations(parent["task_id"])):
        raise AgentExecutionError("known_cancelled_same_session_required")
    stopped = monitor.path / "stop.json"
    archived = monitor.path / "retained_usage_lag_stop.json"
    failure = read_json(stopped if stopped.exists() else archived)
    if failure.get("reason") != LAG_STOP or failure.get("soft_receipt") != digest(receipt):
        raise AgentExecutionError("only_exact_initial_usage_lag_stop_may_continue")
    runtime, old_task = base.make_runtime(root, receipt, monitor, "parallel_fast", transport=transport)
    session = transport.request("GET", "/agents/sessions/" + SESSION)
    if runtime._validate_session(old_task, session) != SESSION:
        raise AgentExecutionError("same_session_response_identity_mismatch")
    roots = runtime._root_turns(SESSION)
    if (session.get("status") != "idle" or session.get("required_actions") not in (None, [])
            or len(roots) != 1 or roots[0].get("id") != TURN or roots[0].get("status") != "cancelled"):
        raise AgentExecutionError("fresh_idle_cancelled_root_required_no_recreate")
    if not monitor.observe(parent["task_id"], session):
        raise AgentExecutionError("settled_usage_required_for_same_session_followup")
    # The original cancelled task/deadline stays immutable. The explicitly
    # authorized follow-up has one fixed five-minute deadline, not a new cohort.
    report = monitor.report()
    if Decimal(report["projected_with_remaining_search_opportunity_usd"]) >= retry.STOP or clock() >= receipt["expires_at"]:
        raise AgentExecutionError("existing_soft_target_or_approval_has_no_headroom")
    ledger = Ledger(root / "live_journal.jsonl", "10.00")
    commit, code = base.code_identity()
    prepared_at = clock()
    proof = {"schema": "hosted_same_session_usage_lag_recovery.v1", "approval": retry.APPROVAL,
        "source_commit": commit, "code_sha256": code, "retry_receipt_sha256": digest(receipt),
        "session_id": SESSION, "prior_turn_id": TURN, "parent_task_digest": parent["task_digest"],
        "prior_stop_sha256": digest(failure), "usage_grace_seconds": base.USAGE_GRACE_SECONDS,
        "soft_target_usd": "2.00", "new_session_create_allowed": False,
        "ledger_events": len(ledger.events), "ledger_head": ledger.previous,
        "aggregate_reserved_usd": str(ledger.exposure), "fresh_session": session, "fresh_roots": roots,
        "prepared_at": prepared_at, "followup_deadline": min(prepared_at + base.ARM_SECONDS, receipt["expires_at"]),
        "new_paid_calls": 0}
    write_once(archived, failure)
    journal.record_event("hosted_usage_recovery", proof)
    write_once(monitor.path / "usage_recovery.json", proof)
    if stopped.exists():
        stopped.unlink()  # Exact retained failure remains anchored, never discarded.
    validate_recovery(root, receipt, clock=clock)
    return proof


class ContinuationRuntime(base.MonitoredRuntime):
    def continue_task(self, task):
        """Reuse the journal's durable input/turn binding; never retry an unknown send."""
        self._validate_task(task)
        parent = self.journal.task(task.parent_task_id)
        proof = validate_recovery(self.monitor.root, self.monitor.receipt, clock=self.clock)
        if (parent["task_id"] != self.monitor.task_id("parallel_fast") or parent["session_id"] != proof["session_id"]
                or task.task_id != parent["task_id"] + "_usage_resume1"):
            raise AgentExecutionError("only_one_exact_same_session_followup_authorized")
        expected = followup_task(self, AgentTask.model_validate(parent["task"]), proof)
        if expected.task_digest != task.task_digest:
            raise AgentExecutionError("exact_authorized_followup_task_required")
        with self.journal.own_task(parent["task_id"]), self.journal.own_task(task.task_id):
            state = self.journal.register(task)
            if state["state"] in TERMINAL_STATES:
                return state
            intent = self.journal.continuation(task.task_id)
            if intent and intent["delivery_state"] != "pending":
                return state
            self.monitor.guard()
            session = self._request("GET", "/agents/sessions/" + SESSION)
            if self._validate_session(task, session) != SESSION:
                raise AgentExecutionError("same_session_response_identity_mismatch")
            roots = self._root_turns(SESSION)
            if (session.get("status") != "idle" or session.get("required_actions")
                    or len(roots) != 1 or roots[0].get("id") != TURN or roots[0].get("status") != "cancelled"):
                raise AgentExecutionError("same_session_followup_remote_not_settled")
            marker = {"role": "user", "content": [{"type": "input_text", "text": canonical_json({
                "blueprint_task_id": task.task_id, "blueprint_task_digest": task.task_digest,
                "context_revision": task.context_revision,
                "instruction": "Continue the same authorized case after reporting-lag cancellation."})}]}
            payload = {"events": [{"type": "agent.session.input.message", "input": [marker, *task.input]}]}
            self.journal.prepare_continuation(task.task_id, payload, [TURN])
            self.prepare_event(task, "/agents/sessions/" + SESSION + "/events", payload)
            self.journal.bind_session(task.task_id, SESSION)
            self.monitor.guard()
            if self.clock() >= task.deadline:
                self.journal.set_state(task.task_id, "cancelled", error_code="followup_deadline_expired")
                return self.journal.task(task.task_id)
            self.journal.set_state(task.task_id, "continuing")
            self.journal.continuation_delivery(task.task_id, "sent_unknown", clock=self.clock)
            try:
                self._request("POST", "/agents/sessions/" + SESSION + "/events", body=payload)
                self.journal.continuation_delivery(task.task_id, "acknowledged")
            except AgentTransportError as exc:
                if exc.definitively_rejected:
                    self.journal.continuation_delivery(task.task_id, "rejected")
                    self.journal.set_state(task.task_id, "failed", error_code="same_session_followup_rejected")
                else:
                    self.journal.set_state(task.task_id, "continuing", error_code=exc.code)
            return self.journal.task(task.task_id)


def followup_task(runtime, old_task, proof):
    """Exactly one canonical child input/configuration, including fixed deadline."""
    inputs = [{"role": "user", "content": [{"type": "input_text", "text":
        "Finish the same Chef Robotics research case using the retained files, findings and same provider tools. "
        "The previous turn was cancelled solely for a reporting-lag check. Do not repeat completed searches. "
        "Inspect relevant evidence and return the complete answer with citations and explicit unknowns."}]}]
    with runtime.journal._connect() as connection:
        used = connection.execute("SELECT COUNT(*) FROM calls WHERE task_id=?", (old_task.task_id,)).fetchone()[0]
    data = old_task.model_dump(mode="json")
    data.update(task_id=old_task.task_id + "_usage_resume1", parent_task_id=old_task.task_id, input=inputs,
        input_digests=[task_digest(inputs), task_digest(runtime.files())],
        deadline=proof["followup_deadline"],
        max_tool_calls=max(0, old_task.max_tool_calls - used))
    data["admission"]["allowed_input_digests"] = list(dict.fromkeys(
        [*data["admission"]["allowed_input_digests"], task_digest(inputs)]))
    return AgentTask.model_validate(data)


def continuation_factory(root, receipt, monitor, mode, *, transport=None):
    runtime, old_task = base.make_runtime(root, receipt, monitor, mode, transport=transport,
        runtime_class=ContinuationRuntime if mode == "parallel_fast" else base.MonitoredRuntime)
    if mode != "parallel_fast":
        return runtime, old_task
    proof = validate_recovery(root, receipt, clock=monitor.clock)
    task = followup_task(runtime, old_task, proof)
    try:
        saved = AgentTask.model_validate(runtime.journal.task(task.task_id)["task"])
        if saved.task_digest != task.task_digest:
            raise AgentExecutionError("exact_authorized_followup_task_required")
        return runtime, saved
    except AgentExecutionError as exc:
        if str(exc) != "agent_task_missing":
            raise
    return runtime, task


def execute(root, receipt, *, clock=time.time, sleep=time.sleep, factory=continuation_factory, notify=print):
    validate_recovery(root, receipt, clock=clock)
    monitor = retry.RetryMonitor(root, receipt, clock=clock, sleep=sleep, notify=notify)
    old_path = Path(root) / "protocols" / base.PROTOCOL / "soft_pilot"
    with exclusive(old_path), exclusive(monitor.path):
        return base._run_owned(root, receipt, monitor, sleep=sleep, notify=notify, factory=factory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--execution-owner-task-id", required=True)
    parser.add_argument("--retry-receipt-sha256", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-continuation", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.aggregate_root.resolve()
    receipt = retry.validate_receipt(root, args.execution_owner_task_id, args.retry_receipt_sha256, readonly_previous=True)
    if args.prepare_continuation:
        monitor = retry.RetryMonitor(root, receipt)
        with exclusive(monitor.path):
            proof = prepare(root, receipt, transport=OpenAIAgentsHTTP(api_key=existing_key("openai"), project_id=base.PROJECT))
        print(json.dumps({"recovery_sha256": digest(proof), "new_paid_calls": 0, "session_id": SESSION}))
    else:
        if version("openai") != base.SDK_VERSION or version("openai-agents") != base.AGENTS_SDK_VERSION:
            raise AgentExecutionError("install_reviewed_sdk_pair_openai_3.22.1_openai_agents_0.22.3")
        result = execute(root, receipt)
        if result["budget"]["stopped"] or len(result["outcomes"]) != 4:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
