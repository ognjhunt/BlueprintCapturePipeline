"""Continue an owned session only after its previous task has settled.

Saved user input binds a new turn to its immutable task revision. An ambiguous
send is observed and never replayed merely because a stream or request failed.
"""

from __future__ import annotations

from collections import Counter

from .contracts import AgentExecutionError, AgentTask, canonical_json
from .journal import TERMINAL_STATES
from .openai_transport import AgentTransportError


def admit_continuation(runtime, task: AgentTask):
    """Validate and reserve the one successor slot without provider traffic."""
    task = task.snapshot()
    runtime._validate_task(task)
    if task.parent_task_id is None:
        raise AgentExecutionError("agent_continuation_parent_missing")
    owner = runtime.journal.session_owner(task.parent_task_id)
    with runtime.journal.own_task(owner["task_id"]), runtime.journal.own_task(task.task_id):
        _validate_parent(runtime, task)
        runtime.validate_admission(task)
        return runtime.journal.register(task)


def _validate_parent(runtime, task):
    parent = runtime.journal.task(task.parent_task_id)
    previous = AgentTask.model_validate(parent["task"])
    if (parent["state"] not in TERMINAL_STATES or parent["session_id"] is None
            or parent["cancel_requested"] or parent["cleanup_state"] != "not_requested"
            or runtime.journal.unsettled_operations(previous.task_id)):
        raise AgentExecutionError("agent_continuation_parent_not_settled")
    for field in ("run_id", "capability", "source_commit", "model", "reasoning_effort",
                  "instructions", "output_schema", "tool_ids", "tool_digests"):
        if getattr(previous, field) != getattr(task, field):
            raise AgentExecutionError("agent_continuation_configuration_changed")
    for field in ("project_id", "runtime", "disclosure_scope", "session_retention", "trace_retention",
                  "region", "budget_policy", "authority_digest", "project_guard_receipt_digest"):
        if getattr(previous.admission, field) != getattr(task.admission, field):
            raise AgentExecutionError("agent_continuation_authority_changed")
    if not set(previous.admission.allowed_input_digests) <= set(task.admission.allowed_input_digests):
        raise AgentExecutionError("agent_continuation_prior_context_not_admitted")
    successor = runtime.journal.successor(previous.task_id)
    if successor is not None and successor != task.task_id:
        raise AgentExecutionError("agent_continuation_successor_already_registered")
    return parent, previous


def continue_task(runtime, task: AgentTask):
    from .openai_agents_api import _identifier, _input_messages

    task = task.snapshot()
    runtime._validate_task(task)
    if task.parent_task_id is None:
        raise AgentExecutionError("agent_continuation_parent_missing")
    owner = runtime.journal.session_owner(task.parent_task_id)
    with runtime.journal.own_task(owner["task_id"]), runtime.journal.own_task(task.task_id):
        parent, previous = _validate_parent(runtime, task)
        state = runtime.journal.register(task)
        if state["state"] in TERMINAL_STATES:
            return state
        intent = runtime.journal.continuation(task.task_id)
        if intent and intent["delivery_state"] != "pending":
            return state
        if state["cancel_requested"] or runtime.clock() >= task.deadline:
            runtime.journal.set_state(task.task_id, "cancelled", error_code="cancelled_before_continuation")
            return runtime.journal.task(task.task_id)
        runtime.validate_admission(task)
        session_id = _identifier(parent["session_id"])
        session = runtime._request("GET", f"/agents/sessions/{session_id}")
        runtime._validate_session(previous, session)
        roots = runtime._root_turns(session_id)
        if any(turn.get("status") not in TERMINAL_STATES for turn in roots) or session.get("required_actions"):
            raise AgentExecutionError("agent_continuation_session_not_idle")
        # Provider turn listings are newest first. Bind the complete owned
        # lineage by identity instead of depending on a listing's order.
        root_ids = [_identifier(turn.get("id")) for turn in roots]
        owned_ids = {row["turn_id"] for row in runtime.journal.lineage_tasks(previous.task_id) if row["turn_id"]}
        if (not roots or len(root_ids) != len(set(root_ids)) or set(root_ids) != owned_ids
                or parent["turn_id"] not in owned_ids):
            raise AgentExecutionError("agent_continuation_unowned_turn")
        marker = {"role": "user", "content": [{"type": "input_text", "text": canonical_json({
            "blueprint_task_id": task.task_id, "blueprint_task_digest": task.task_digest,
            "context_revision": task.context_revision,
            "instruction": "Re-read current authoritative state for this task revision before acting.",
        })}]}
        payload = {"events": [{"type": "agent.session.input.message",
                                "input": [marker, *_input_messages(task.input)]}]}
        if intent is None:
            runtime.journal.prepare_continuation(task.task_id, payload, [turn["id"] for turn in roots])
        elif intent["payload"] != payload:
            raise AgentExecutionError("agent_continuation_intent_conflict")
        runtime.journal.bind_session(task.task_id, session_id)
        runtime.validate_admission(task.snapshot())
        if runtime.clock() >= task.deadline or runtime.journal.task(task.task_id)["cancel_requested"]:
            runtime.journal.set_state(task.task_id, "cancelled", error_code="cancelled_before_continuation")
            return runtime.journal.task(task.task_id)
        runtime.journal.set_state(task.task_id, "continuing")
        # Intent precedes the POST; any subsequent crash requires saved-input
        # reconciliation, including a crash before the socket write.
        runtime.journal.continuation_delivery(task.task_id, "sent_unknown", clock=runtime.clock)
        try:
            runtime._request("POST", f"/agents/sessions/{session_id}/events", body=payload)
            runtime.journal.continuation_delivery(task.task_id, "acknowledged")
        except AgentTransportError as exc:
            if exc.definitively_rejected:
                runtime.journal.continuation_delivery(task.task_id, "rejected")
                runtime.journal.set_state(task.task_id, "failed", error_code="agent_continuation_rejected")
            else:
                runtime.journal.set_state(task.task_id, "continuing", error_code=exc.code)
        return runtime.journal.task(task.task_id)


def task_turns(runtime, task: AgentTask, session_id: str, roots):
    """Require the exact submitted user inputs on exactly one new root turn."""
    from .openai_agents_api import _identifier, _input_messages

    intent = runtime.journal.continuation(task.task_id)
    if intent is None:
        raise AgentExecutionError("agent_continuation_intent_missing")
    if intent["turn_id"]:
        selected = [turn for turn in roots if turn.get("id") == intent["turn_id"]]
        if len(selected) != 1:
            raise AgentExecutionError("agent_continuation_bound_turn_missing")
        if {turn.get("id") for turn in roots} != {*intent["previous_turns"], intent["turn_id"]}:
            raise AgentExecutionError("agent_continuation_unowned_turn")
        return selected
    candidates = [turn for turn in roots if turn.get("id") not in intent["previous_turns"]]
    if not candidates:
        return []
    if len(candidates) != 1:
        raise AgentExecutionError("agent_continuation_new_turn_ambiguous")
    turn_id = _identifier(candidates[0]["id"])
    expected_messages = intent["payload"]["events"][0]["input"]
    expected = Counter(canonical_json(message) for message in expected_messages)
    items = [item for item in runtime._list(f"/agents/sessions/{session_id}/items")
             if item.get("type") == "message" and item.get("role") == "user" and item.get("turn_id") == turn_id]
    identities = [_identifier(item.get("id")) for item in items]
    if len(set(identities)) != len(identities):
        raise AgentExecutionError("agent_continuation_duplicate_saved_input_id")
    if any(item.get("status") != "completed" for item in items):
        return []
    if any(set(item) - {"id", "type", "role", "content", "phase", "status", "turn_id"} for item in items):
        raise AgentExecutionError("agent_continuation_saved_input_shape_invalid")
    observed_messages = _input_messages([
        {"role": item["role"], "content": item.get("content")} for item in items
    ])
    observed = Counter(canonical_json(message) for message in observed_messages)
    if expected != observed:
        # The live API coalesces adjacent user messages into one saved message,
        # retaining their separate content blocks in order. Accept only that
        # exact representation; partial, reordered or extra bytes still refuse.
        coalesced = {"role": "user", "content": [part for message in expected_messages for part in message["content"]]}
        if (any(message["role"] != "user" for message in expected_messages)
                or observed_messages != [coalesced]):
            return []
    runtime.journal.continuation_delivery(task.task_id, "bound", turn_id=turn_id)
    runtime.journal.set_state(task.task_id, "running", turn_id=turn_id)
    return candidates
