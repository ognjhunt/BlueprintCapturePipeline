"""ADP-009B/day-21: managed session asset tools remain bounded after restart."""
from __future__ import annotations

import json
import time

import pytest

from blueprint_pipeline.agent_execution.contracts import (
    AgentAdmission, AgentExecutionError, ToolContext, digest,
)
from blueprint_pipeline.agent_execution.journal import AgentJournal
from blueprint_pipeline.agent_execution.openai_agents_api import OpenAIAgentsRuntime
from blueprint_pipeline.agent_execution.operations import AgentOperations
from blueprint_pipeline.task_object_agents_api import (
    AgentsAPIAssetTools, asset_input, prepare_asset_task, prepare_repair_task,
)
from blueprint_pipeline.task_object_agent_tools import AssetTools
from tests.test_astra_automatic_resume import authoring_fixture  # noqa: F401
from tests.test_task_evaluation_scene_configuration_astra_driver import retained, component  # noqa: F401
from tests.test_task_object_agent_session import agent_fixture, bounded  # noqa: F401
from tests.test_agent_execution_sessions import FakeAPI


def _tools(f):
    return AgentsAPIAssetTools(request_value=f.kwargs["request_value"],
        output_root=f.kwargs["output_root"], journal_root=f.runtime / "agents-api-tools",
        cad_executor=f.kwargs["cad_executor"], blender_runner=f.kwargs["blender_runner"],
        blender_executable=f.kwargs["blender_executable"])


def _context(f, number):
    return ToolContext(run_id=f.request.run_id, task_id="future_drawer_author",
        context_revision=digest({"request": f.request.request_digest}),
        operation_id=f"operation_{number}", authority_digest=digest({"future": True}),
        deadline=time.time() + 3600)


def test_future_scene_task_is_explicitly_admitted_and_model_bound(agent_fixture):  # noqa: F811
    f = agent_fixture
    tools = _tools(f)
    inputs = asset_input(f.kwargs["request_value"])
    assert any(item["type"] == "input_image" for item in inputs[0]["content"])
    now = time.time()
    admission = AgentAdmission(authority_digest=digest({"future": True}),
        authority_reference="future-drawer:owner-admitted", project_id="proj_future",
        runtime="openai_agents_api", disclosure_scope="task_asset_source_frames_and_metric_envelope",
        allowed_input_digests=(digest(inputs),),
        allowed_tool_ids=tuple(tool.tool_id for tool in tools.tools()),
        session_retention="until_deleted", trace_retention="provider_default", region="us",
        budget_policy="project_guard_accepted_uncertainty", inference_budget_usd=1,
        project_guard_receipt_digest=digest({"guard": "future"}), expires_at=now + 3600)
    task = prepare_asset_task(request_value=f.kwargs["request_value"], tools=tools,
        admission=admission, task_id="future_drawer_author", source_commit="a" * 40,
        deadline=now + 1800)
    assert task.model == "gpt-6-sol" and task.admission.runtime == "openai_agents_api"
    assert task.tool_ids == ("observe_object", "build_cad", "render_candidate", "inspect_candidate")
    assert "independent review" in task.instructions
    with pytest.raises(AgentExecutionError, match="asset_api_admission_mismatch"):
        prepare_asset_task(request_value=f.kwargs["request_value"], tools=tools,
            admission=admission.model_copy(update={"disclosure_scope": "other"}),
            task_id="future_drawer_author", source_commit="a" * 40, deadline=now + 1800)


def test_tool_repair_restart_and_independent_review(agent_fixture):  # noqa: F811
    f = agent_fixture
    tools = _tools(f)
    handlers = {tool.tool_id: tool for tool in tools.tools()}
    brief = f.steps[0][1]
    assert handlers["observe_object"].invoke(brief, _context(f, 1))["status"] == "recorded"
    repaired = handlers["build_cad"].invoke({"program": "broken"}, _context(f, 2))
    assert repaired["status"] == "repair_needed"
    assert "unknown fillet radius" in repaired["error"]
    restarted = _tools(f)
    handlers = {tool.tool_id: tool for tool in restarted.tools()}
    assert handlers["build_cad"].reconcile({"program": "broken"}, _context(f, 2)).output == repaired
    assert handlers["build_cad"].invoke({"program": "repaired"}, _context(f, 3))["status"] == "built"
    assert handlers["render_candidate"].invoke(f.steps[3][1], _context(f, 4))["status"] == "rendered_pending_independent_review"
    inspected = handlers["inspect_candidate"].invoke({}, _context(f, 5))
    assert sum(item["type"] == "input_image" for item in inspected) == 3
    invoker, _ = bounded(f)
    state = {"state": "completed", "task": {"run_id": f.request.run_id,
        "model": "gpt-6-sol", "capability": "task_asset_authoring",
        "input_digests": [digest(asset_input(f.kwargs["request_value"]))]}}
    with pytest.raises(AgentExecutionError, match="asset_api_author_turn_not_completed"):
        restarted.review(task_state={**state, "state": "running"}, invoker=invoker)
    reviewed = restarted.review(task_state=state, invoker=invoker)
    assert reviewed["accepted"] is True
    assert _tools(f).review(task_state=state, invoker=invoker) == reviewed
    assert invoker.calls == 2


def test_unknown_cad_outcome_is_never_reexecuted(agent_fixture):  # noqa: F811
    f = agent_fixture
    tools = _tools(f)
    handlers = {tool.tool_id: tool for tool in tools.tools()}
    handlers["observe_object"].invoke(f.steps[0][1], _context(f, 1))
    called = []

    def uncertain(**kwargs):
        called.append(kwargs["program"])
        raise TimeoutError("unknown kernel outcome")

    tools.cad_executor = uncertain
    with pytest.raises(TimeoutError, match="unknown kernel outcome"):
        handlers["build_cad"].invoke({"program": "candidate"}, _context(f, 2))
    with pytest.raises(AgentExecutionError, match="asset_api_tool_outcome_unresolved"):
        _tools(f)
    assert called == ["candidate"]
    assert json.loads((f.runtime / "agents-api-tools/002-started.json").read_text())["operation_id"] == "operation_2"


def test_rejected_review_creates_same_session_repair_revision(agent_fixture, monkeypatch):  # noqa: F811
    f = agent_fixture
    tools = _tools(f)
    handlers = {tool.tool_id: tool for tool in tools.tools()}
    handlers["observe_object"].invoke(f.steps[0][1], _context(f, 1))
    handlers["build_cad"].invoke({"program": "repaired"}, _context(f, 2))
    handlers["render_candidate"].invoke(f.steps[3][1], _context(f, 3))
    inputs = asset_input(f.kwargs["request_value"])
    now = time.time()
    admission = AgentAdmission(authority_digest=digest({"future": True}),
        authority_reference="future-drawer:owner-admitted", project_id="proj_future",
        runtime="openai_agents_api", disclosure_scope="task_asset_source_frames_and_metric_envelope",
        allowed_input_digests=(digest(inputs),),
        allowed_tool_ids=tuple(tool.tool_id for tool in tools.tools()),
        session_retention="until_deleted", trace_retention="provider_default", region="us",
        budget_policy="project_guard_accepted_uncertainty", inference_budget_usd=1,
        project_guard_receipt_digest=digest({"guard": "future"}), expires_at=now + 3600)
    parent = prepare_asset_task(request_value=f.kwargs["request_value"], tools=tools,
        admission=admission, task_id="future_drawer_author", source_commit="a" * 40,
        deadline=now + 1800)
    feedback = {"accepted": False, "review": {"blockers": ["missing_handle"],
        "repair_instructions": "Add observed handle."}}
    monkeypatch.setattr(AssetTools, "independent_review", lambda self, invoker: feedback)
    reviewed = tools.review(task_state={"state": "completed", "task": parent.model_dump(mode="json")},
        invoker=object())
    assert reviewed == feedback
    repair_input = [{"role": "user", "content": [{"type": "input_text", "text":
        "Independent review rejected the candidate. Repair within the existing tool limits: "
        + json.dumps(feedback, sort_keys=True, separators=(",", ":"))}]}]
    widened = admission.model_copy(update={"allowed_input_digests": (digest(inputs), digest(repair_input))})
    repair = prepare_repair_task(previous=parent, reviewed=reviewed, tools=tools,
        admission=widened, task_id="future_drawer_repair_1", deadline=now + 1700)
    assert repair.parent_task_id == parent.task_id
    assert repair.instructions == parent.instructions and repair.model == parent.model
    with pytest.raises(AgentExecutionError, match="asset_api_repair_parent_invalid"):
        prepare_repair_task(previous=parent, reviewed={"accepted": True}, tools=tools,
            admission=widened, task_id="bad", deadline=now + 1700)


def test_wire_shaped_agents_api_dispatches_confined_asset_tool_once(agent_fixture):  # noqa: F811
    f = agent_fixture
    tools = _tools(f)
    inputs = asset_input(f.kwargs["request_value"])
    now = time.time()
    admission = AgentAdmission(authority_digest=digest({"future": True}),
        authority_reference="future-drawer:owner-admitted", project_id="proj_fixture",
        runtime="openai_agents_api", disclosure_scope="task_asset_source_frames_and_metric_envelope",
        allowed_input_digests=(digest(inputs),),
        allowed_tool_ids=tuple(tool.tool_id for tool in tools.tools()),
        session_retention="until_deleted", trace_retention="provider_default", region="us",
        budget_policy="project_guard_accepted_uncertainty", inference_budget_usd=1,
        project_guard_receipt_digest=digest({"guard": "future"}), expires_at=now + 3600)
    task = prepare_asset_task(request_value=f.kwargs["request_value"], tools=tools,
        admission=admission, task_id="future_drawer_author", source_commit="a" * 40,
        deadline=now + 1800)
    api = FakeAPI()
    journal = AgentJournal(f.runtime / "api-journal")
    operations = AgentOperations(journal, tools.tools(), authorize=lambda *_: None,
        clock=lambda: now)
    runtime = OpenAIAgentsRuntime(transport=api, project_id="proj_fixture",
        journal=journal, operations=operations, validate_admission=lambda _: None,
        clock=lambda: now)
    assert runtime.start(task)["state"] == "running"
    payload = api.calls[0][2]
    assert payload["agent"]["model"] == "gpt-6-sol"
    assert payload["environment"] == {"type": "none"}
    assert {tool["name"] for tool in payload["agent"]["tools"]} == set(task.tool_ids)
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": "call_observe",
        "name": "observe_object", "arguments": f.steps[0][1]}]
    assert runtime.step(task.task_id)["state"] == "running"
    assert api.tool_results[0]["success"] is True
    assert json.loads(api.tool_results[0]["output"])["status"] == "recorded"
    assert len(list((f.runtime / "agents-api-tools").glob("???-started.json"))) == 1
