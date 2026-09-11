"""Real production composition drives adaptive tools and seals retained output."""
from dataclasses import replace
import json
import time

import pytest

from blueprint_pipeline.agent_execution import episode_tasks as ep
from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, digest
from blueprint_pipeline.agent_execution.openai_agents_api import OpenAIAgentsRuntime
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from blueprint_pipeline.episode_interpretation import materialize_episode_interpretation_rights
from tests.test_agent_execution_sessions import FakeAPI
from tests.test_agent_production_service import fixture, write
from tests.test_episode_investigation import evidence
from tests.test_episode_interpretation import _request, _output


def setup(tmp_path):
    service, _, old_record, config_path = fixture(tmp_path)
    old = json.loads(old_record.read_text())
    old["enabled"] = old["autostart"] = False
    write(old_record, old)
    _, data, _ = evidence(tmp_path)
    request = _request(data)
    guard = {"schema_version": "blueprint_agent_project_admission_observation.v1",
        "project_id": "proj_fixture", "credential_id": "key_fixture",
        "dashboard_hard_limit_enabled": True, "observed_at": time.time() - 1,
        "expires_at": time.time() + 3600, "disclosure_scope": "rights_admitted_episode_evidence",
        "budget_policy": "project_guard_accepted_uncertainty", "session_retention": "until_deleted",
        "trace_retention": "provider_default", "provider_api_region": "us",
        "spend_limit": {"object": "project.spend_limit", "currency": "USD", "interval": "month", "threshold_amount": 3000}}
    guard_path = tmp_path / "guard.json"
    write(guard_path, guard)
    config = json.loads(config_path.read_text())
    config.update(managed_api_enabled=True, project_guard_receipt_digest=digest(guard),
        project_guard_receipt_file=str(guard_path), max_project_budget_usd=30)
    write(config_path, config)
    service = ProductionAgentService(config_path, source_commit="a" * 40)
    rights_path = tmp_path / "episode-rights.json"
    policy = {"project_id": "proj_fixture", "disclosure_scope": guard["disclosure_scope"],
        "budget_policy": guard["budget_policy"], "session_retention": "until_deleted", "trace_retention": "provider_default",
        "region": "us", "project_guard_receipt_digest": digest(guard)}
    materialize_episode_interpretation_rights(
        episode_id=request.episode_id, input_bundle_digest=request.input_receipt["input_bundle_digest"],
        identity=ep.interpreter_identity("openai_agents_api", "gpt-5.6-terra"), allowed_artifact_roles=ep.ROLES,
        external_disclosure_authorized=True, accepted_by="fixture-owner", accepted_on="2026-09-10",
        authority_reference="test-only", source_rights_admission_digest=digest({"owned_fixture": True}),
        output_path=rights_path, agent_runtime_policy=policy)
    record = ep.prepare_episode_task(service, task_id="episode_task", run_id="episode_run", request=request,
        rights_path=rights_path, owner_client_id="fixture-client", inference_budget_usd=1)
    api = FakeAPI()
    original = service.runtime_for_task
    def runtime(task):
        value = original(task)
        assert isinstance(value, OpenAIAgentsRuntime)
        value.transport = api
        return value
    service.service.runtime_for_task = runtime
    return service, record, api, data


def drive(service, record, api, data, *, inspect=True):
    task = record.task
    service.enqueue(task.task_id, "fixture-client")
    service.service.tick()
    runtime = service.service.runtime_for_task(task)
    calls = [
        ("read_episode_context", {}),
        ("read_episode_trace", {"role": "state_trace", "start_step": 0, "end_step": 20, "limit": 100}),
        ("read_episode_trace", {"role": "contact_force_trace", "start_step": 0, "end_step": 20, "limit": 100}),
        ("inspect_episode_interval", {"start_seconds": 0, "end_seconds": 1, "max_observations": 8}),
    ]
    if inspect:
        api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": "call_" + str(i),
                        "name": name, "arguments": args} for i, (name, args) in enumerate(calls)]
        runtime.step(task.task_id)
    api.actions = []
    api.output = _output(data).model_dump(mode="json")
    api.turn_status = "completed"
    runtime.step(task.task_id)
    assert service.journal.task(task.task_id)["state"] == "completed"


def test_adaptive_api_tools_terminal_collection_restart_and_score_preservation(tmp_path):
    service, record, api, data = setup(tmp_path)
    score = data["score_path"].read_bytes()
    drive(service, record, api, data)
    count = len(api.calls)
    receipt = ep.collect_episode_task(service, record.task.task_id)
    assert receipt["schema_version"] == "episode_interpretation_receipt.v2"
    assert receipt["interpreter"]["runtime"] == "openai_agents_api"
    assert receipt["prompt_digest"] == ep.PROMPT_DIGEST
    assert receipt["interpreter_execution"]["task_digest"] == record.task.task_digest
    assert data["score_path"].read_bytes() == score
    assert len(api.calls) == count
    assert any(part["type"] == "input_image" for reply in api.tool_results for part in reply["output"]
               if isinstance(reply["output"], list))
    restarted = ProductionAgentService(service.config_path, source_commit="a" * 40)
    assert ep.collect_episode_task(restarted, record.task.task_id) == receipt
    restarted.autostart()
    assert restarted.journal.event("episode_collected_" + record.task.task_digest[7:])["receipt_digest"] == receipt["receipt_digest"]


def test_terminal_output_without_required_retrieval_does_not_become_valid_interpretation(tmp_path):
    service, record, api, data = setup(tmp_path)
    drive(service, record, api, data, inspect=False)
    with pytest.raises(AgentExecutionError, match="required_inspection_incomplete"):
        ep.collect_episode_task(service, record.task.task_id)


def test_changed_rights_refused_before_provider_call(tmp_path):
    service, record, api, _ = setup(tmp_path)
    from pathlib import Path
    Path(record.episode_investigation.rights_path).write_text('{}')
    with pytest.raises(AgentExecutionError, match="rights_changed"):
        service.enqueue(record.task.task_id, "fixture-client")
    assert api.calls == []


def test_rights_removed_after_start_does_not_prevent_cancellation_and_cleanup(tmp_path):
    from pathlib import Path
    service, record, api, _ = setup(tmp_path)
    service.enqueue(record.task.task_id, "fixture-client")
    runtime = service.service.runtime_for_task(record.task)
    runtime.step(record.task.task_id)
    Path(record.episode_investigation.rights_path).unlink()
    assert runtime.step(record.task.task_id)["state"] == "cancelled"
    # Reconstruct the production runtime after restart without the source rights.
    fresh = service.runtime_for_task(record.task)
    fresh.transport = api
    assert fresh.cleanup(record.task.task_id)["cleanup_state"] == "deleted"
    assert api.deleted


def test_candidate_policy_cannot_be_its_own_interpreter(tmp_path):
    service, record, _, data = setup(tmp_path)
    with pytest.raises(AgentExecutionError, match="self_grading"):
        ep.prepare_episode_task(service, task_id="self_grade", run_id="self_grade_run",
            request=replace(_request(data), candidate_policy_id=record.task.model),
            rights_path=record.episode_investigation.rights_path, owner_client_id="fixture-client")
