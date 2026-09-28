"""Receipt changes revisit one owner without repeating unchanged work."""
import json
import time

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, digest
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from blueprint_pipeline.agent_execution.supervision import ObservationSource, SupervisionPlan, progress_plan
from tests.test_agent_execution_continuation import ContinuingAPI
from tests.test_agent_production_service import fixture, write


def setup(tmp_path):
    _, task, template_path, config_path = fixture(tmp_path)
    guard = {"schema_version": "blueprint_agent_project_admission_observation.v1",
        "project_id": "proj_fixture", "credential_id": "key_fixture", "observed_at": time.time() - 1,
        "expires_at": time.time() + 1800, "dashboard_hard_limit_enabled": True,
        "disclosure_scope": "sanitized_operations", "budget_policy": "project_guard_accepted_uncertainty",
        "session_retention": "until_deleted", "trace_retention": "provider_default", "provider_api_region": "us",
        "spend_limit": {"object": "project.spend_limit", "threshold_amount": 3000, "currency": "USD", "interval": "month"}}
    guard_path = tmp_path / "guard.json"
    write(guard_path, guard)
    config = json.loads(config_path.read_text())
    config.update(managed_api_enabled=True, project_guard_receipt_digest=digest(guard),
        project_guard_receipt_file=str(guard_path), max_project_budget_usd=30,
        supervision_store_root=str(tmp_path / "plans"))
    write(config_path, config)
    template = json.loads(template_path.read_text())
    template["autostart"] = False
    template["task"]["admission"].update(runtime="openai_agents_api", budget_policy="project_guard_accepted_uncertainty",
        session_retention="until_deleted", trace_retention="provider_default", region="us", project_guard_receipt_digest=digest(guard))
    write(template_path, template)
    source = tmp_path / "source.json"
    write(source, {"schema_version": "test_run_state.v1", "run_id": task.run_id, "status": "failed", "phase": "first"})
    plan = SupervisionPlan(schema_version="blueprint_agent_supervision_plan.v1", watch_id="watch", enabled=True,
        run_id=task.run_id, template_task_id=task.task_id, source_commit=task.source_commit,
        sources=(ObservationSource(source_id="child", path=str(source), schema_version="test_run_state.v1",
            identity_field="run_id", identity_value=task.run_id),),
        maximum_revisions=3, maximum_reserved_inference_usd=20, expires_at=time.time() + 500)
    write(tmp_path / "plans/watch.json", plan.model_dump(mode="json"))
    api = ContinuingAPI()
    service = ProductionAgentService(config_path, source_commit=task.source_commit)
    def runtime(current):
        value = service.runtime_for_task(current)
        value.transport = api
        return value
    service.service.runtime_for_task = runtime
    return service, plan, api, source


def complete(service, state, api):
    task = service.record(state["active_task_id"]).task
    runtime = service.service.runtime_for_task(task)
    runtime.step(task.task_id)
    api.output = {"disposition": "no_action", "summary": "Current revision inspected.",
                  "evidence_references": [], "next_actions": [], "uncertainty": []}
    api.turn_status = "completed"
    runtime.step(task.task_id)
    assert service.journal.task(task.task_id)["state"] == "completed"
    return task


def test_new_evidence_continues_same_session_and_unchanged_receipts_do_not_repeat(tmp_path):
    service, plan, api, source = setup(tmp_path)
    first = progress_plan(service, plan)
    assert not api.calls
    parent = complete(service, first, api)
    value = json.loads(source.read_text())
    value["generated_at"] = time.time()
    write(source, value)
    assert progress_plan(service, plan)["status"] == "waiting_for_new_evidence"
    value["phase"] = "second"
    write(source, value)
    second = progress_plan(service, plan)
    child = complete(service, second, api)
    assert child.parent_task_id == parent.task_id
    assert sum(method == "POST" and path == "/agents/sessions" for method, path, *_ in api.calls) == 1
    restarted = ProductionAgentService(service.config_path, source_commit=plan.source_commit)
    assert progress_plan(restarted, plan)["status"] == "waiting_for_new_evidence"
    value["phase"] = "third"
    write(source, value)
    assert progress_plan(restarted, plan)["status"] == "reserved_inference_limit_reached"


def test_stale_revision_cancels_before_another_owner_can_start(tmp_path):
    service, plan, api, source = setup(tmp_path)
    first = progress_plan(service, plan)
    with pytest.raises(AgentExecutionError, match="run_owned_by_persistent"):
        service.enqueue(plan.template_task_id, "fixture-client")
    task = service.record(first["active_task_id"]).task
    runtime = service.service.runtime_for_task(task)
    runtime.step(task.task_id)
    value = json.loads(source.read_text())
    value["phase"] = "second"
    write(source, value)
    assert progress_plan(service, plan)["status"] == "settling_stale_revision"
    assert runtime.step(task.task_id)["state"] == "cancelled"
    next_state = progress_plan(service, plan)
    assert next_state["active_task_id"] != task.task_id
    assert service.record(next_state["active_task_id"]).task.parent_task_id is None


def test_crash_after_reservation_recovers_same_task_without_new_budget(tmp_path, monkeypatch):
    service, plan, api, _ = setup(tmp_path)
    original = service._enqueue_record
    monkeypatch.setattr(service, "_enqueue_record", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        progress_plan(service, plan)
    monkeypatch.setattr(service, "_enqueue_record", original)
    state = progress_plan(service, plan)
    assert len(state["revisions"]) == 1 and state["reserved_inference_usd"] == 10
    assert service.journal.task(state["active_task_id"])["state"] == "queued"
    assert not api.calls


def test_supervision_retains_context_until_revoked_then_cleans_the_whole_lineage(tmp_path):
    service, plan, api, source = setup(tmp_path)
    first = progress_plan(service, plan)
    parent = complete(service, first, api)
    assert service.record(parent.task_id).cleanup_when_terminal is False
    service.autostart()
    assert service.journal.task(parent.task_id)["cleanup_state"] == "not_requested"
    value = json.loads(source.read_text())
    value["phase"] = "second"
    write(source, value)
    child = complete(service, progress_plan(service, plan), api)
    revoked = plan.model_copy(update={"enabled": False})
    write(tmp_path / "plans/watch.json", revoked.model_dump(mode="json"))
    assert progress_plan(service, revoked)["status"] == "revoked"
    assert service.journal.task(child.task_id)["cleanup_state"] == "pending"
    assert service.service.tick()["cleanup_state"] == "deleted"
    assert service.journal.task(parent.task_id)["cleanup_state"] == "deleted"
    assert service.journal.task(parent.task_id)["result"] is not None
