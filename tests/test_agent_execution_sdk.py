"""Run the installed SDK against HTTP fixtures; never replace Runner or audit."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
from pydantic import BaseModel, ConfigDict
import pytest

from blueprint_pipeline.agent_execution.contracts import (
    AgentAdmission, AgentExecutionError, AgentTask, AgentTool, ToolReconciliation, digest,
)
from blueprint_pipeline.agent_execution.journal import AgentJournal
from blueprint_pipeline.agent_execution.operations import AgentOperations, OperationPending
from blueprint_pipeline.agent_execution.sdk_runtime import OpenAIAgentsSDKRuntime, SDKCredential
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKInvocationBlocked


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: int


def make_task(tools=(), **changes):
    inputs = [{"role": "user", "content": [{"type": "input_text", "text": "Inspect evidence."}]}]
    admission = AgentAdmission(
        authority_digest=digest({"authority": 1}), authority_reference="fixture:owned-run",
        project_id="proj_scoped", runtime="openai_agents_sdk", disclosure_scope="fixture_only",
        allowed_input_digests=(digest(inputs),), allowed_tool_ids=tuple(t.tool_id for t in tools),
        budget_policy="strict_per_call", inference_budget_usd=10, expires_at=time.time() + 60,
    )
    values = dict(
        task_id="sdk_fixture", run_id="run_fixture", capability="test_investigation",
        context_revision=digest({"revision": 1}), source_commit="a" * 40,
        instructions="Use admitted tools and return the declared object.",
        model="gpt-5.6-terra", input=inputs, input_digests=(digest(inputs),),
        output_schema=Answer.model_json_schema(), tool_ids=tuple(t.tool_id for t in tools),
        tool_digests={t.tool_id: t.tool_digest for t in tools}, admission=admission,
        deadline=time.time() + 30, max_tool_output_bytes=1000,
    )
    values.update(changes)
    return AgentTask(**values)


def response(output=None, *, output_tokens=32, number=1):
    return httpx.Response(200, json={
        "id": f"resp_fixture_{number}", "object": "response", "created_at": int(time.time()),
        "status": "completed", "model": "gpt-5.6-terra", "parallel_tool_calls": False,
        "output": output if output is not None else [{
            "type": "message", "id": f"msg_{number}", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": '{"answer":7}', "annotations": []}],
        }],
        "usage": {"input_tokens": 100, "output_tokens": output_tokens,
                  "total_tokens": 100 + output_tokens,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    })


def setup_runtime(tmp_path, monkeypatch, handler, *, tools=(), validate=None, credential=None):
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "1")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", "/must-not-read-ambient-key")
    journal = AgentJournal(tmp_path)
    operations = AgentOperations(journal, tools, authorize=lambda *_: None)
    runtime = OpenAIAgentsSDKRuntime(
        journal=journal, operations=operations, output_models={"test_investigation": Answer},
        validate_admission=validate or (lambda _: None),
        resolve_credential=lambda _: credential or SDKCredential("proj_scoped", "fixture_key", "sk-fixture"),
        hermetic_http_transport=httpx.MockTransport(handler),
    )
    return runtime


def test_scoped_client_real_sdk_reservation_and_result(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["OpenAI-Project"] == "proj_scoped"
        assert request.headers["Authorization"] == "Bearer sk-fixture"
        payload = json.loads(request.content)
        assert payload["store"] is False
        assert payload["model"] == "gpt-5.6-terra"
        assert runtime._audit(task).manifest()["in_flight_unknown_count"] == 1
        return response()

    runtime = setup_runtime(tmp_path, monkeypatch, handler)
    task = make_task(max_model_turns=1)
    runtime.start(task)
    state = runtime.step(task.task_id)
    assert state["state"] == "completed"
    assert state["result"]["output"] == {"answer": 7}
    assert state["result"]["project_id"] == "proj_scoped"
    assert state["result"]["hermetic"] is True
    assert runtime._audit(task).manifest()["in_flight_unknown_count"] == 0
    assert len(calls) == 1
    assert "sk-fixture" not in json.dumps(state)


def test_project_mismatch_refused_before_request(tmp_path, monkeypatch):
    calls = []
    runtime = setup_runtime(tmp_path, monkeypatch, lambda r: calls.append(r),
                            credential=SDKCredential("wrong_project", "fixture_key", "sk-fixture"))
    task = make_task()
    runtime.start(task)
    with pytest.raises(AgentExecutionError, match="credential_project_mismatch"):
        runtime.step(task.task_id)
    assert not calls


def test_budget_refuses_before_provider_and_counts_instructions(tmp_path, monkeypatch):
    calls = []
    runtime = setup_runtime(tmp_path, monkeypatch, lambda r: calls.append(r))
    original = make_task(max_model_turns=1)
    values = original.model_dump()
    values["admission"]["inference_budget_usd"] = 0.00001
    task = AgentTask(**values)
    runtime.start(task)
    with pytest.raises(AgentsSDKInvocationBlocked, match="budget_ceiling"):
        runtime.step(task.task_id)
    assert not calls
    assert runtime.inspect(task.task_id)["state"] == "failed"


@pytest.mark.parametrize("reason", ["deadline", "cancel", "revocation"])
def test_inflight_abort_preserves_reservation_no_retry(tmp_path, monkeypatch, reason):
    started = threading.Event()
    authority = {"valid": True}
    calls = []

    async def handler(request):
        calls.append(request)
        started.set()
        await asyncio.Event().wait()
        return response()

    def validate(_):
        if not authority["valid"]:
            raise ValueError("revoked")

    runtime = setup_runtime(tmp_path, monkeypatch, handler, validate=validate)
    clock = [time.time()]
    runtime.clock = lambda: clock[0]
    task = make_task(deadline=time.time() + 15)
    runtime.start(task)
    errors = []

    def run():
        try:
            runtime.step(task.task_id)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    observed_start = started.wait(30)
    if not observed_start:
        runtime.cancel(task.task_id)
        worker.join(5)
    assert observed_start, [type(error).__name__ for error in errors]
    if reason == "deadline":
        clock[0] = task.deadline + 1
    if reason == "cancel":
        before = time.monotonic()
        assert runtime.cancel(task.task_id)["cancel_requested"]
        assert time.monotonic() - before < 0.5
    if reason == "revocation":
        authority["valid"] = False
    worker.join(3)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert runtime.inspect(task.task_id)["state"] == "reconciling"
    assert runtime._audit(task).manifest()["in_flight_unknown_count"] == 1
    runtime.step(task.task_id)
    assert len(calls) == 1


def test_real_tool_loop_reuses_durable_side_effect(tmp_path, monkeypatch):
    calls, effects = [], []
    tool = AgentTool("inspect_stage", "1", "Read fixture stage", {
        "type": "object", "properties": {}, "additionalProperties": False,
    }, "idempotent_write", lambda args, context: effects.append(context.operation_id) or {"count": 1},
        lambda *_: ToolReconciliation("not_started"))

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) <= 2:
            return response([{"type": "function_call", "id": f"fc_{len(calls)}",
                              "call_id": f"call_{len(calls)}", "name": "inspect_stage",
                              "arguments": "{}", "status": "completed"}], number=len(calls))
        return response(number=3)

    runtime = setup_runtime(tmp_path, monkeypatch, handler, tools=(tool,))
    task = make_task((tool,))
    runtime.start(task)
    assert runtime.step(task.task_id)["state"] == "completed"
    assert len(effects) == 1
    assert len(calls) == 3
    assert any(item.get("type") == "function_call_output" for item in calls[-1]["input"])


def test_actual_cost_overrun_cannot_be_reported_success(tmp_path, monkeypatch):
    runtime = setup_runtime(tmp_path, monkeypatch, lambda _: response(output_tokens=100_000_000))
    task = make_task(max_model_turns=1)
    runtime.start(task)
    with pytest.raises(Exception, match="exceeds_reservation"):
        runtime.step(task.task_id)
    assert runtime.inspect(task.task_id)["state"] == "reconciling"
    assert runtime._audit(task).manifest()["in_flight_unknown_count"] == 1


def test_cancellation_at_result_commit_cannot_be_accepted(tmp_path, monkeypatch):
    runtime = setup_runtime(tmp_path, monkeypatch, lambda _: response())
    task = make_task(max_model_turns=1)
    runtime.start(task)
    original = runtime.journal.record_usage

    def cancelling_usage(task_id, usage):
        original(task_id, usage)
        runtime.cancel(task_id)

    monkeypatch.setattr(runtime.journal, "record_usage", cancelling_usage)
    with pytest.raises(AgentExecutionError, match="completion_after_cancellation"):
        runtime.step(task.task_id)
    assert runtime.inspect(task.task_id)["state"] == "reconciling"
    assert runtime.inspect(task.task_id)["result"] is None


def test_cancel_during_blocking_tool_keeps_operation_owned(tmp_path, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    calls, effects = [], []

    def invoke(args, context):
        entered.set()
        assert finish.wait(5)
        effects.append(context.operation_id)
        return {"complete": True}

    tool = AgentTool("submit_fixture", "1", "Submit fixture", {
        "type": "object", "properties": {}, "additionalProperties": False,
    }, "external_side_effect", invoke, lambda *_: ToolReconciliation("not_started"))

    def handler(request):
        calls.append(request)
        return response([{"type": "function_call", "id": "fc_1", "call_id": "call_1",
                          "name": tool.tool_id, "arguments": "{}", "status": "completed"}])

    runtime = setup_runtime(tmp_path, monkeypatch, handler, tools=(tool,))
    task = make_task((tool,))
    runtime.start(task)
    errors = []

    def run():
        try:
            runtime.step(task.task_id)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        # Tool readiness includes SDK/client initialization on a cold CI host.
        # The cancellation responsiveness assertion below remains two seconds.
        assert entered.wait(15), f"tool was not reached; worker errors: {errors!r}"
        runtime.cancel(task.task_id)
        worker.join(2)
        assert not worker.is_alive()
        with pytest.raises(OperationPending, match="still_running"):
            runtime.step(task.task_id)
    finally:
        finish.set()
        worker.join(2)
    deadline = time.monotonic() + 2
    while runtime.journal.unsettled_operations(task.task_id) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(effects) == len(calls) == len(errors) == 1
    assert not runtime.journal.unsettled_operations(task.task_id)
