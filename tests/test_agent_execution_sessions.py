"""Exercise the real durable bridge against a stateful, wire-shaped fake API."""

from __future__ import annotations

from dataclasses import replace
import json
import threading

import pytest
from pydantic import ValidationError

from blueprint_pipeline.agent_execution.contracts import (
    AgentAdmission, AgentExecutionError, AgentTask, AgentTool, ToolReconciliation, digest,
)
from blueprint_pipeline.agent_execution.journal import AgentJournal
from blueprint_pipeline.agent_execution.openai_agents_api import OpenAIAgentsRuntime
from blueprint_pipeline.agent_execution.openai_transport import AgentTransportError
from blueprint_pipeline.agent_execution.operations import AgentOperations


OUTPUT_SCHEMA = {
    "type": "object", "properties": {"answer": {"type": "integer"}},
    "required": ["answer"], "additionalProperties": False,
}
INPUT_SCHEMA = {
    "type": "object", "properties": {"value": {"type": "integer"}},
    "required": ["value"], "additionalProperties": False,
}


def make_task(tools=(), **changes):
    inputs = [{"role": "user", "content": [{"type": "input_text", "text": "Inspect evidence."}]}]
    admission = AgentAdmission(
        authority_digest=digest({"authority": 1}), authority_reference="fixture:owned-run",
        project_id="proj_fixture", runtime="openai_agents_api", disclosure_scope="fixture_only",
        allowed_input_digests=(digest(inputs),), allowed_tool_ids=tuple(t.tool_id for t in tools),
        session_retention="until_deleted", trace_retention="provider_default", region="us",
        budget_policy="project_guard_accepted_uncertainty", inference_budget_usd=1,
        project_guard_receipt_digest=digest({"guard": "fixture"}), expires_at=2000,
    )
    values = dict(
        task_id="task_fixture", run_id="run_fixture", capability="test_investigation",
        context_revision=digest({"revision": 1}), source_commit="a" * 40,
        instructions="Use the supplied tools and return the declared object.",
        model="gpt-5.6-terra", input=inputs, input_digests=(digest(inputs),),
        output_schema=json.loads(json.dumps(OUTPUT_SCHEMA)), tool_ids=tuple(t.tool_id for t in tools),
        tool_digests={t.tool_id: t.tool_digest for t in tools}, admission=admission,
        deadline=1900,
    )
    values.update(changes)
    return AgentTask(**values)


class FakeAPI:
    project_id = "proj_fixture"

    def __init__(self):
        self.calls = []
        self.session = None
        self.turn_status = "in_progress"
        self.output = {"answer": 7}
        self.actions = []
        self.tool_results = []
        self.create_uncertain = False
        self.reply_uncertain = False
        self.deleted = False
        self.delete_conflict = False
        self.paginate_items = False
        self.cancel_ignored = False
        self.completed_at = 1700

    def final_message(self):
        return {
            "type": "message", "id": "message_final", "turn_id": "turn_1",
            "role": "assistant", "phase": "final_answer", "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(self.output)}],
        }

    def request(self, method, path, *, body=None, query=None):
        self.calls.append((method, path, body, query))
        if path == "/agents/sessions":
            if method == "POST":
                self.session = {
                    "id": "session_1", "agent": body["agent"], "metadata": body["metadata"],
                    "environment": {"type": "none"}, "usage": None,
                }
                if self.create_uncertain:
                    self.create_uncertain = False
                    raise AgentTransportError("agents_api_connection_uncertain")
                return dict(self.session)
            return {"data": [self.session] if self.session else [], "has_more": False}
        if path == "/agents/sessions/session_1":
            if self.deleted:
                raise AgentTransportError("missing", status=404)
            if method == "DELETE":
                if self.delete_conflict:
                    raise AgentTransportError("busy", status=409)
                self.deleted = True
                return {"deleted": True}
            return {
                **self.session, "status": "idle" if self.turn_status == "completed" else "in_progress",
                "required_actions": list(self.actions),
            }
        if path.endswith("/turns"):
            return {
                "data": [{"id": "turn_1", "session_id": "session_1", "subagent_id": None,
                          "status": self.turn_status, "usage": None,
                          "completed_at": self.completed_at}],
                "has_more": False,
            }
        if path.endswith("/items"):
            if self.paginate_items and not query.get("after"):
                return {
                    "data": [{"type": "message", "id": "message_commentary", "turn_id": "turn_1",
                              "role": "assistant", "phase": "commentary", "status": "completed",
                              "content": [{"type": "output_text", "text": '{"answer": 999}'}]}],
                    "has_more": True, "last_id": "message_commentary",
                }
            return {"data": [self.final_message()], "has_more": False}
        if path.endswith("/events"):
            event = body["events"][0]
            if event["type"] == "agent.session.input.cancel":
                if not self.cancel_ignored:
                    self.turn_status = "cancelled"
                    self.actions = []
                return {}
            self.tool_results.append(event)
            if self.reply_uncertain:
                self.reply_uncertain = False
                raise AgentTransportError("agents_api_connection_uncertain")
            self.actions = [a for a in self.actions if a["call_id"] != event["call_id"]]
            self.turn_status = "completed"
            return {}
        raise AssertionError((method, path))


@pytest.mark.parametrize("reason", ["cancel", "deadline", "revocation"])
def test_authority_change_during_authorization_cannot_start_effect(tmp_path, reason):
    entered, release = threading.Event(), threading.Event()
    effects, errors = [], []
    authority = {"valid": True, "calls": 0, "now": 1000}
    tool = AgentTool("submit_work", "1", "Submit fixture", INPUT_SCHEMA, "external_side_effect",
                     lambda *_: effects.append(1) or {"done": True},
                     lambda *_: ToolReconciliation("not_started"))

    def authorize(*_):
        authority["calls"] += 1
        if authority["calls"] == 1:
            entered.set()
            assert release.wait(3)
            return
        if not authority["valid"]:
            raise AgentExecutionError("revoked")

    journal = AgentJournal(tmp_path)
    task = make_task((tool,))
    journal.register(task)
    operations = AgentOperations(journal, (tool,), authorize=authorize, clock=lambda: authority["now"])

    def run():
        try:
            operations.execute(task, turn_id="turn", call_id="call", name=tool.tool_id, arguments={"value": 1})
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(2)
        if reason == "cancel":
            journal.request_cancel(task.task_id, "requested")
        elif reason == "deadline":
            authority["now"] = task.deadline + 1
        else:
            authority["valid"] = False
    finally:
        release.set()
        worker.join(3)
    assert not effects
    assert len(errors) == 1


def runtime(tmp_path, *, api=None, tools=(), clock=None, authorize=None):
    api = api or FakeAPI()
    journal = AgentJournal(tmp_path)
    clock = clock or (lambda: 1000)
    operations = AgentOperations(
        journal, tuple(tools), authorize=authorize or (lambda *_: None), clock=clock,
    )
    service = OpenAIAgentsRuntime(
        transport=api, project_id="proj_fixture", journal=journal, operations=operations,
        validate_admission=lambda _: None, clock=clock,
    )
    return service, api


def action(name="measure", call_id="call_1", value=7):
    return {"type": "function_call", "turn_id": "turn_1", "call_id": call_id,
            "name": name, "arguments": {"value": value}}


def test_managed_admission_cannot_claim_strict_cost_or_stateless_retention():
    admission = make_task().admission.model_dump()
    for patch in [
        {"budget_policy": "strict_per_call"}, {"session_retention": "not_admitted"},
        {"trace_retention": "not_admitted"}, {"region": "default"},
        {"project_guard_receipt_digest": None},
    ]:
        with pytest.raises(ValidationError, match="not_admitted"):
            AgentAdmission.model_validate({**admission, **patch})


def test_payload_and_tool_identity_are_part_of_the_admitted_task():
    value = make_task().model_dump()
    value["input"][0]["content"][0]["text"] = "Different disclosure"
    with pytest.raises(ValidationError, match="payload_not_admitted"):
        AgentTask.model_validate(value)


def test_create_uses_real_beta_shape_and_never_discloses_authority(tmp_path):
    service, api = runtime(tmp_path)
    task = make_task()
    assert service.start(task)["state"] == "running"
    payload = api.calls[0][2]
    assert payload["environment"] == {"type": "none"}
    assert payload["agent"]["multi_agent"] == {"enabled": False}
    assert payload["agent"]["text"]["format"] == {"type": "json_schema", "schema": OUTPUT_SCHEMA}
    assert payload["metadata"]["blueprint_task_digest"] == task.task_digest
    assert "admission" not in payload and "max_turns" not in payload["agent"]
    service.start(task)
    assert len(api.calls) == 1


def test_creation_timeout_adopts_original_session_without_second_post(tmp_path):
    api = FakeAPI()
    api.create_uncertain = True
    service, _ = runtime(tmp_path, api=api)
    task = make_task()
    assert service.start(task)["state"] == "creation_unresolved"
    restarted, _ = runtime(tmp_path, api=api)
    assert restarted.step(task.task_id)["session_id"] == "session_1"
    assert sum(m == "POST" and p == "/agents/sessions" for m, p, *_ in api.calls) == 1


def test_missing_list_result_does_not_authorize_recreation(tmp_path):
    service, api = runtime(tmp_path)
    task = make_task()
    service.journal.register(task)
    service.journal.set_state(task.task_id, "creating")
    assert service.step(task.task_id)["state"] == "creation_unresolved"
    assert service.step(task.task_id)["state"] == "creation_unresolved"
    assert not any(method == "POST" for method, *_ in api.calls)


def test_result_is_saved_before_delivery_and_duplicate_action_does_not_repeat_tool(tmp_path):
    invocations = []
    tool = AgentTool("measure", "1", "Read the measurement", INPUT_SCHEMA, "read_only",
                     lambda args, ctx: invocations.append(ctx.operation_id) or dict(args))
    task = make_task((tool,))
    service, api = runtime(tmp_path, tools=(tool,))
    service.start(task)
    api.actions = [action()]
    api.reply_uncertain = True
    with pytest.raises(AgentTransportError):
        service.step(task.task_id)
    restarted, _ = runtime(tmp_path, api=api, tools=(tool,))
    restarted.step(task.task_id)
    result = restarted.step(task.task_id)
    assert result["state"] == "completed"
    assert result["result"]["output"] == {"answer": 7}
    assert len(invocations) == 1
    assert api.tool_results[0] == api.tool_results[1]
    assert result["result"]["runtime"] == "openai_agents_api"
    assert result["result"]["scientific_acceptance_granted"] is False
    assert result["result"]["usage"] is None
    assert result["result"]["cost_status"] == "official_reconciliation_required"


def test_interrupted_mutation_requires_authoritative_reconciliation(tmp_path):
    external = {"attempts": 0, "result": None}

    def invoke(args, ctx):
        external["attempts"] += 1
        external["result"] = dict(args)
        raise OSError("connection lost after the external system accepted the operation")

    tool = AgentTool("measure", "1", "Execute admitted operation", INPUT_SCHEMA,
                     "external_side_effect", invoke)
    task = make_task((tool,))
    service, api = runtime(tmp_path, tools=(tool,))
    service.start(task)
    api.actions = [action()]
    assert service.step(task.task_id)["state"] == "reconciling"
    assert service.step(task.task_id)["state"] == "reconciling"
    assert external["attempts"] == 1
    reconciler = replace(tool, reconcile=lambda *_: ToolReconciliation("completed", external["result"]))
    restarted, _ = runtime(tmp_path, api=api, tools=(reconciler,))
    restarted.step(task.task_id)
    assert restarted.step(task.task_id)["state"] == "completed"
    assert external["attempts"] == 1


def test_same_mutation_under_a_new_call_id_reuses_semantic_operation(tmp_path):
    invocations = []
    tool = AgentTool("measure", "1", "Write one admitted record", INPUT_SCHEMA, "idempotent_write",
                     lambda args, ctx: invocations.append(ctx.operation_id) or dict(args))
    service, api = runtime(tmp_path, tools=(tool,))
    task = make_task((tool,))
    service.start(task)
    api.actions = [action()]
    service.step(task.task_id)
    api.actions = [action(call_id="call_2")]
    api.turn_status = "waiting"
    service.step(task.task_id)
    assert len(invocations) == 1


def test_turn_completion_requires_final_message_and_complete_pagination(tmp_path):
    service, api = runtime(tmp_path)
    task = make_task()
    service.start(task)
    api.paginate_items = True
    api.turn_status = "completed"
    result = service.step(task.task_id)
    assert result["result"]["output"] == {"answer": 7}
    assert any(q and q.get("after") == "message_commentary" for *_, q in api.calls)


def test_idle_and_schema_invalid_output_never_become_success(tmp_path):
    service, api = runtime(tmp_path)
    task = make_task()
    service.start(task)
    assert service.step(task.task_id)["state"] == "running"
    api.turn_status = "completed"
    api.output = {"answer": 7, "physical_success": True}
    with pytest.raises(AgentExecutionError, match="output_schema_invalid"):
        service.step(task.task_id)
    assert service.inspect(task.task_id)["state"] != "completed"


def test_deadline_cancels_without_executing_waiting_tools_and_cleanup_reads_absence(tmp_path):
    now = [1000]
    tool = AgentTool("measure", "1", "Read", INPUT_SCHEMA, "read_only",
                     lambda *_: pytest.fail("No tool may run after the deadline"))
    task = make_task((tool,))
    service, api = runtime(tmp_path, tools=(tool,), clock=lambda: now[0])
    service.start(task)
    api.actions = [action()]
    now[0] = 1901
    assert service.step(task.task_id)["state"] == "cancelled"
    api.delete_conflict = True
    assert service.cleanup(task.task_id)["cleanup_state"] == "pending"
    api.delete_conflict = False
    assert service.cleanup(task.task_id)["cleanup_state"] == "deleted"
    assert api.calls[-1][:2] == ("GET", "/agents/sessions/session_1")


def test_tool_drift_and_changed_call_arguments_fail_before_mutation(tmp_path):
    tool = AgentTool("measure", "1", "Read", INPUT_SCHEMA, "read_only", lambda args, _: args)
    task = make_task((tool,))
    service, api = runtime(tmp_path, tools=(tool,))
    service.start(task)
    api.actions = [action()]
    service.step(task.task_id)
    api.actions = [action(value=8)]
    api.turn_status = "waiting"
    with pytest.raises(AgentExecutionError, match="call_identity_conflict"):
        service.step(task.task_id)
    changed, _ = runtime(tmp_path, api=api, tools=(replace(tool, version="2"),))
    with pytest.raises(AgentExecutionError, match="tool_definition_changed"):
        changed.step(task.task_id)


def test_task_ownership_and_immutable_identity(tmp_path):
    first = AgentJournal(tmp_path)
    second = AgentJournal(tmp_path)
    task = make_task()
    first.register(task)
    with first.own_task(task.task_id):
        with pytest.raises(AgentExecutionError, match="owned_by_another_worker"):
            with second.own_task(task.task_id):
                pytest.fail("second worker acquired the task")
    with second.own_task(task.task_id):
        pass
    with pytest.raises(AgentExecutionError, match="task_identity_conflict"):
        first.register(make_task(model="different-model"))


def test_event_replay_is_idempotent_but_conflicting_event_is_rejected(tmp_path):
    journal = AgentJournal(tmp_path)
    assert journal.record_event("evt_1", {"type": "state_changed"}) is True
    assert journal.record_event("evt_1", {"type": "state_changed"}) is False
    with pytest.raises(AgentExecutionError, match="event_identity_conflict"):
        journal.record_event("evt_1", {"type": "different"})


@pytest.mark.parametrize("field", ["input", "output_schema", "tool_digests"])
def test_nested_task_mutation_is_rejected_before_registration_or_disclosure(tmp_path, field):
    service, api = runtime(tmp_path)
    task = make_task()
    if field == "input":
        task.input[0]["content"][0]["text"] = "UNDISCLOSED SECRET"
    elif field == "output_schema":
        task.output_schema["properties"]["unreviewed"] = {"type": "string"}
    else:
        task.tool_digests["unknown_tool"] = digest("changed")
    with pytest.raises(AgentExecutionError, match="mutated_after_validation"):
        service.start(task)
    assert not api.calls
    assert not service.journal.tasks(active_only=False)


def test_same_arguments_in_distinct_tasks_do_not_alias_effects_or_results(tmp_path):
    calls = []
    tool = AgentTool("measure", "1", "Write task-scoped output", INPUT_SCHEMA, "idempotent_write",
                     lambda args, ctx: calls.append(ctx.task_id) or {"owner": ctx.task_id})
    service, _ = runtime(tmp_path, tools=(tool,))
    first, second = make_task((tool,)), make_task((tool,), task_id="task_second")
    for task in (first, second):
        service.journal.register(task)
        with service.journal.own_task(task.task_id):
            result = service.operations.execute(
                task, turn_id="turn_1", call_id="call_1", name="measure", arguments={"value": 7},
            )
        assert result["output"] == {"owner": task.task_id}
    assert calls == [first.task_id, second.task_id]


def test_cancel_during_uncertain_creation_survives_restart_and_never_executes_tools(tmp_path):
    tool = AgentTool("measure", "1", "Must not execute", INPUT_SCHEMA, "external_side_effect",
                     lambda *_: pytest.fail("Cancelled tool executed"))
    api = FakeAPI()
    api.create_uncertain = True
    service, _ = runtime(tmp_path, api=api, tools=(tool,))
    task = make_task((tool,))
    service.start(task)
    assert service.cancel(task.task_id)["cancel_requested"] == 1
    api.actions = [action()]
    restarted, _ = runtime(tmp_path, api=api, tools=(tool,))
    assert restarted.step(task.task_id)["state"] == "cancelled"
    assert not api.tool_results


def test_transport_project_mismatch_refuses_before_post(tmp_path):
    api = FakeAPI()
    api.project_id = "proj_not_admitted"
    with pytest.raises(AgentExecutionError, match="transport_project_mismatch"):
        runtime(tmp_path, api=api)
    assert not api.calls


@pytest.mark.parametrize("with_tools", [False, True])
def test_revoked_admission_cancels_before_any_further_tool_or_acceptance(tmp_path, with_tools):
    tools = (AgentTool("measure", "1", "Must not execute", INPUT_SCHEMA, "external_side_effect",
                       lambda *_: pytest.fail("Revoked tool executed")),) if with_tools else ()
    service, api = runtime(tmp_path, tools=tools)
    task = make_task(tools)
    service.start(task)

    def revoked(_):
        raise AgentExecutionError("revoked")

    service.validate_admission = revoked
    if with_tools:
        api.actions = [action()]
    else:
        api.turn_status = "completed"
        api.cancel_ignored = True
    assert service.step(task.task_id)["state"] == "cancelled"
    assert service.inspect(task.task_id)["cancel_reason"] == "agent_admission_revoked"
    assert service.cleanup(task.task_id)["cleanup_state"] == "deleted"


@pytest.mark.parametrize("mode", ["foreign", "terminal", "duplicate"])
def test_required_actions_must_belong_to_unique_active_root_turn(tmp_path, mode):
    tool = AgentTool("measure", "1", "Must not execute", INPUT_SCHEMA, "external_side_effect",
                     lambda *_: pytest.fail("Invalid-turn tool executed"))
    service, api = runtime(tmp_path, tools=(tool,))
    task = make_task((tool,))
    service.start(task)
    api.actions = [action()]
    if mode == "foreign":
        api.actions[0]["turn_id"] = "turn_unrelated"
    elif mode == "terminal":
        api.turn_status = "completed"
    else:
        api.actions *= 2
    with pytest.raises(AgentExecutionError):
        service.step(task.task_id)
    assert not api.tool_results


def test_unresolved_creation_records_deadline_cancellation_without_recreating(tmp_path):
    service, api = runtime(tmp_path, clock=lambda: 2100)
    task = make_task()
    service.journal.register(task)
    service.journal.set_state(task.task_id, "creating")
    value = service.step(task.task_id)
    assert value["state"] == "creation_unresolved"
    assert value["cancel_requested"] == 1 and value["cancel_reason"] == "agent_task_deadline"
    assert not any(m == "POST" for m, *_ in api.calls)


@pytest.mark.parametrize("completed_at,expected", [(1800, "completed"), (1950, "cancelled"),
                                                   (None, "cancelled"), (True, "cancelled")])
def test_deadline_race_requires_an_authoritative_prior_completion_time(tmp_path, completed_at, expected):
    now = [1000]
    service, api = runtime(tmp_path, clock=lambda: now[0])
    task = make_task()
    service.start(task)
    api.turn_status = "completed"
    api.completed_at = completed_at
    api.cancel_ignored = True
    now[0] = 1910
    assert service.step(task.task_id)["state"] == expected


@pytest.mark.parametrize("outcome", ["completed", "not_started", "pending"])
def test_cancellation_reconciles_uncertain_effect_without_reexecuting(tmp_path, outcome):
    invoked = []
    reconciled = []

    def invoke(args, ctx):
        invoked.append(ctx.operation_id)
        raise OSError("uncertain")

    def reconcile(args, ctx):
        assert ctx.reconciliation_only is True
        reconciled.append(ctx.operation_id)
        return ToolReconciliation(outcome, dict(args) if outcome == "completed" else None)

    tool = AgentTool("measure", "1", "Admitted effect", INPUT_SCHEMA, "external_side_effect", invoke)
    task = make_task((tool,))
    service, api = runtime(tmp_path, tools=(tool,))
    service.start(task)
    api.actions = [action()]
    assert service.step(task.task_id)["state"] == "reconciling"
    service.cancel(task.task_id)
    restarted, _ = runtime(tmp_path, api=api, tools=(replace(tool, reconcile=reconcile),))
    result = restarted.step(task.task_id)
    assert len(invoked) == 1 and len(reconciled) == 1
    if outcome == "pending":
        assert result["state"] == "reconciling"
        with pytest.raises(AgentExecutionError, match="cleanup_before_reconciliation"):
            restarted.cleanup(task.task_id)
    else:
        assert result["state"] == "cancelled"
        assert restarted.cleanup(task.task_id)["cleanup_state"] == "deleted"
