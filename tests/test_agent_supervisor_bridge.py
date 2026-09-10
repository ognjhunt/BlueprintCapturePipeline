"""Use the real registry, bound tool handler and observation validator."""

from dataclasses import replace

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, ToolContext, digest
from blueprint_pipeline.agent_execution.journal import AgentJournal
from blueprint_pipeline.agent_execution import supervisor_bridge as bridge_module
from blueprint_pipeline.agent_execution.supervisor_bridge import (
    BoundSupervisorService, SupervisorCapabilityBridge, context_revision,
)
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import ToolRegistry


def bridge_fixture(tmp_path):
    registry = ToolRegistry.default()
    status = {"status": "waiting", "first_blocker": "sam_review_pending", "next_required_stage": "sam_review"}
    status["status_digest"] = digest(status)
    authority = default_authority_envelope(
        run_id="bridge_run", mode=AutonomyMode.EXECUTE_NON_SPEND, tool_registry=registry,
        immutable_input_digests=[status["status_digest"]],
        agent_inference_budget_usd=1, allow_agent_inference=True,
    ).to_mapping()
    current = [SupervisorContext(
        run_id="bridge_run", customer_question="Complete the admitted scene preparation.",
        authority_envelope=authority, fresh_scene_preparation_status=status,
        supervisor_output_dir=str(tmp_path / "supervisor"),
    )]
    journal = AgentJournal(tmp_path / "journal")
    bridge = SupervisorCapabilityBridge(
        capability="capture_testbed_supervisor", registry=registry, journal=journal,
        load_context=lambda _: current[0],
    )
    tools = {tool.tool_id: tool for tool in bridge.tools()}
    context = ToolContext(
        "bridge_run", "bridge_task", context_revision(current[0]), digest({"operation": 1}),
        authority["authority_digest"], 1000, tuple(authority["immutable_input_digests"]),
    )
    arguments = {"context_revision": context.context_revision,
                 "arguments": {"status_digest": status["status_digest"]}}
    return bridge, tools, context, arguments, current


def test_real_preparation_inspection_preserves_controller_identity(tmp_path):
    bridge, tools, context, arguments, _ = bridge_fixture(tmp_path)
    observation = tools["inspect_fresh_scene_preparation"].invoke(arguments, context)
    assert observation["schema_version"] == "task_evaluation_supervisor_tool_observation.v1"
    assert observation["typed_result"]["first_blocker"] == "sam_review_pending"
    assert observation["typed_result"]["digest_matches"]
    assert observation["runtime_identity"] == "blueprint_local_deterministic_non_spend"
    assert observation["proof_effect"] == "none"
    saved = bridge.journal.event("registry_observation_" + context.operation_id[7:])
    assert saved["observation"] == observation
    assert tools["inspect_fresh_scene_preparation"].reconcile(arguments, context).output == observation


def test_changed_evidence_revision_cannot_use_old_tool_context(tmp_path):
    _, tools, context, arguments, current = bridge_fixture(tmp_path)
    current[0] = replace(current[0], customer_question="Changed objective")
    with pytest.raises(AgentExecutionError, match="revision_stale"):
        tools["inspect_fresh_scene_preparation"].invoke(arguments, context)


def test_model_cannot_substitute_context_revision_or_disclosure(tmp_path):
    _, tools, context, arguments, _ = bridge_fixture(tmp_path)
    tool = tools["inspect_fresh_scene_preparation"]
    with pytest.raises(AgentExecutionError, match="revision_stale"):
        tool.invoke({**arguments, "context_revision": digest({"new": 1})}, context)
    with pytest.raises(AgentExecutionError, match="disclosure_scope"):
        tool.invoke(arguments, replace(context, allowed_input_digests=()))


def test_missing_real_materializer_is_explicitly_unavailable(tmp_path):
    _, tools, context, _, _ = bridge_fixture(tmp_path)
    result = tools["materialize_sam31_task_inputs"].invoke({
        "context_revision": context.context_revision, "arguments": {"request_digest": digest({})},
    }, context)
    assert result["status"] == "unavailable"
    assert result["proof_effect"] == "none"


def test_non_spend_registry_does_not_invent_paid_reconciliation(tmp_path):
    bridge, _, _, _, _ = bridge_fixture(tmp_path)
    bridge.capability = "runtime_failure_recovery"
    assert "execute_preauthorized_recovery" not in {tool.tool_id for tool in bridge.tools()}


def test_execution_root_is_bound_even_when_source_inputs_do_not_change(tmp_path):
    _, tools, context, arguments, current = bridge_fixture(tmp_path)
    current[0] = replace(current[0], supervisor_output_dir=str(tmp_path / "different"))
    with pytest.raises(AgentExecutionError, match="revision_stale"):
        tools["inspect_fresh_scene_preparation"].invoke(arguments, context)


def test_anonymous_executor_is_refused_and_configured_executor_is_bound(tmp_path):
    _, _, _, _, current = bridge_fixture(tmp_path)
    with pytest.raises(AgentExecutionError, match="executor_identity_missing"):
        context_revision(replace(current[0], pose_estimator=lambda _: {}))
    executor = BoundSupervisorService("a" * 40, digest({"code": 1}), digest({"configuration": 1}), lambda _: {})
    before = context_revision(replace(current[0], pose_estimator=executor))
    after = context_revision(replace(current[0], pose_estimator=replace(
        executor, configuration_digest=digest({"configuration": 2}),
    )))
    assert before != after


def test_authoritative_context_is_reloaded_after_handler_execution(tmp_path, monkeypatch):
    bridge, tools, context, arguments, current = bridge_fixture(tmp_path)
    original = bridge_module.validate_tool_observation_binding

    def changed_during_execution(*args, **kwargs):
        value = original(*args, **kwargs)
        current[0] = replace(current[0], customer_question="A newer task supersedes this one")
        return value

    monkeypatch.setattr(bridge_module, "validate_tool_observation_binding", changed_during_execution)
    with pytest.raises(AgentExecutionError, match="context_mutated_during_execution"):
        tools["inspect_fresh_scene_preparation"].invoke(arguments, context)
    assert bridge.journal.event("registry_observation_" + context.operation_id[7:]) is None
