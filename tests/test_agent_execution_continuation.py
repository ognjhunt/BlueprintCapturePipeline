"""Exercise blueprint_pipeline.agent_execution.continuation through its runtime.

The public adapter delegates these lifecycle calls to that continuation module.
"""

from __future__ import annotations

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, AgentTask, digest
from blueprint_pipeline.agent_execution.openai_transport import AgentTransportError
from tests.test_agent_execution_sessions import FakeAPI, make_task, runtime


class ContinuingAPI(FakeAPI):
    def __init__(self):
        super().__init__()
        self.current_turn = "turn_1"
        self.previous_turns, self.previous_items, self.inputs = [], [], []
        self.followup_uncertain = False
        self.hide_new_input = False
        self.reject_followup_status = None

    def final_message(self):
        return {**super().final_message(), "turn_id": self.current_turn,
                "id": "final_" + self.current_turn}

    def request(self, method, path, *, body=None, query=None):
        if path.endswith("/events") and body["events"][0]["type"] == "agent.session.input.message":
            self.calls.append((method, path, body, query))
            if self.reject_followup_status is not None:
                raise AgentTransportError("rejected", status=self.reject_followup_status)
            self.previous_turns.append({"id": self.current_turn, "session_id": "session_1",
                                        "subagent_id": None, "status": self.turn_status,
                                        "completed_at": self.completed_at})
            self.previous_items.append(self.final_message())
            self.current_turn = f"turn_{len(self.previous_turns) + 1}"
            self.turn_status = "in_progress"
            self.inputs.extend({**message, "type": "message", "status": "completed",
                                "id": f"input_{self.current_turn}_{index}", "turn_id": self.current_turn}
                               for index, message in enumerate(body["events"][0]["input"]))
            if self.followup_uncertain:
                self.followup_uncertain = False
                raise AgentTransportError("agents_api_connection_uncertain")
            return {}
        value = super().request(method, path, body=body, query=query)
        if path.endswith("/turns"):
            value["data"] = [*self.previous_turns,
                             {**value["data"][0], "id": self.current_turn}]
        if path.endswith("/items"):
            value["data"] = [*self.previous_items, self.final_message(),
                             *([] if self.hide_new_input else self.inputs)]
        return value


def successor(previous, *, task_id="followup", **changes):
    values = previous.model_dump(mode="json")
    values.update(task_id=task_id, parent_task_id=previous.task_id,
                  context_revision=digest({"revision": task_id}),
                  input=[{"role": "user", "content": [{"type": "input_text", "text": "Inspect new receipt."}]}])
    values["admission"]["allowed_input_digests"].append(digest(values["input"]))
    values["admission"]["allowed_input_digests"] = list(dict.fromkeys(values["admission"]["allowed_input_digests"]))
    values["input_digests"] = [digest(values["input"])]
    values.update(changes)
    return AgentTask(**values)


def completed_parent(tmp_path):
    api = ContinuingAPI()
    run, _ = runtime(tmp_path, api=api)
    task = make_task()
    run.start(task)
    api.turn_status = "completed"
    assert run.step(task.task_id)["state"] == "completed"
    return run, api, task


def test_continue_uses_same_session_and_new_result_turn(tmp_path):
    run, api, task = completed_parent(tmp_path)
    followup = successor(task)
    state = run.continue_task(followup)
    assert state["session_id"] == "session_1"
    assert state["state"] == "continuing"
    assert run.step(followup.task_id)["turn_id"] == "turn_2"
    api.output, api.turn_status = {"answer": 9}, "completed"
    final = run.step(followup.task_id)
    assert final["result"]["output"] == {"answer": 9}
    assert final["result"]["turn_id"] == "turn_2"
    assert run.inspect(task.task_id)["result"]["output"] == {"answer": 7}
    assert sum(method == "POST" and path == "/agents/sessions" for method, path, *_ in api.calls) == 1


def test_uncertain_followup_recovers_saved_input_without_resend(tmp_path):
    run, api, task = completed_parent(tmp_path)
    followup = successor(task)
    api.followup_uncertain = True
    run.continue_task(followup)
    restarted, _ = runtime(tmp_path, api=api)
    restarted.continue_task(followup)
    assert restarted.step(followup.task_id)["turn_id"] == "turn_2"
    assert len(api.previous_turns) == 1


def test_partial_saved_input_cannot_authorize_turn_or_reuse_old_answer(tmp_path):
    run, api, task = completed_parent(tmp_path)
    followup = successor(task)
    run.continue_task(followup)
    api.hide_new_input, api.turn_status = True, "completed"
    assert run.step(followup.task_id)["state"] == "continuing"
    assert run.inspect(followup.task_id)["result"] is None
    api.hide_new_input = False
    assert run.step(followup.task_id)["result"]["turn_id"] == "turn_2"


@pytest.mark.parametrize("change", ["model", "source_commit", "tool_digests", "prior_scope"])
def test_continuation_cannot_silently_change_configuration_or_disclosure(tmp_path, change):
    run, api, task = completed_parent(tmp_path)
    values = successor(task).model_dump()
    if change == "prior_scope":
        values["admission"]["allowed_input_digests"] = values["input_digests"]
    elif change == "tool_digests":
        values["instructions"] = "Changed tool instructions"
    else:
        values[change] = "gpt-6-astra" if change == "model" else "b" * 40
    followup = AgentTask(**values)
    with pytest.raises(AgentExecutionError, match="configuration_changed|prior_context_not_admitted"):
        run.continue_task(followup)
    assert not api.previous_turns


def test_session_cannot_fork_two_owners_or_be_deleted_by_predecessor(tmp_path):
    run, api, task = completed_parent(tmp_path)
    followup = successor(task)
    run.continue_task(followup)
    with pytest.raises(AgentExecutionError, match="successor_already_registered"):
        run.continue_task(successor(task, task_id="competing"))
    with pytest.raises(AgentExecutionError, match="has_successor"):
        run.cleanup(task.task_id)
    assert not api.deleted
    api.turn_status = "completed"
    run.step(followup.task_id)
    assert run.cleanup(followup.task_id)["cleanup_state"] == "deleted"
    assert run.inspect(task.task_id)["cleanup_state"] == "deleted"


def test_unrelated_task_cannot_claim_an_owned_session(tmp_path):
    run, _, task = completed_parent(tmp_path)
    other = make_task(task_id="unrelated")
    run.journal.register(other)
    with pytest.raises(AgentExecutionError, match="already_owned"):
        run.journal.bind_session(other.task_id, run.inspect(task.task_id)["session_id"])


def test_cancelled_unbound_leaf_can_clean_retained_parent_session(tmp_path):
    run, api, task = completed_parent(tmp_path)
    followup = successor(task, deadline=900)
    assert run.continue_task(followup)["state"] == "cancelled"
    assert run.inspect(followup.task_id)["session_id"] is None
    assert run.cleanup(followup.task_id)["cleanup_state"] == "deleted"
    assert api.deleted
    assert run.inspect(task.task_id)["cleanup_state"] == "deleted"


def test_definitively_rejected_followup_is_failed_and_cleanable(tmp_path):
    run, api, task = completed_parent(tmp_path)
    api.reject_followup_status = 400
    followup = successor(task)
    assert run.continue_task(followup)["state"] == "failed"
    assert run.journal.continuation(followup.task_id)["delivery_state"] == "rejected"
    assert run.step(followup.task_id)["state"] == "failed"
    run.cleanup(followup.task_id)
    assert api.deleted
    assert not api.previous_turns


@pytest.mark.parametrize("status", ["in_progress", "incomplete", "failed"])
def test_uncompleted_input_items_cannot_authorize_the_turn(tmp_path, status):
    run, api, task = completed_parent(tmp_path)
    followup = successor(task)
    run.continue_task(followup)
    api.turn_status = "completed"
    api.inputs[0]["status"] = status
    assert run.step(followup.task_id)["state"] == "continuing"
    assert run.inspect(followup.task_id)["result"] is None


@pytest.mark.parametrize("change", ["duplicate_id", "extra_message_field", "extra_content_field"])
def test_unsupported_saved_message_shape_cannot_bind_turn(tmp_path, change):
    run, api, task = completed_parent(tmp_path)
    followup = successor(task)
    run.continue_task(followup)
    if change == "duplicate_id":
        api.inputs[1]["id"] = api.inputs[0]["id"]
    elif change == "extra_message_field":
        api.inputs[0]["unexpected"] = True
    else:
        api.inputs[0]["content"][0]["unexpected"] = True
    with pytest.raises(AgentExecutionError, match="duplicate_saved_input|shape_invalid|content_unsupported"):
        run.step(followup.task_id)
    assert run.inspect(followup.task_id)["result"] is None
