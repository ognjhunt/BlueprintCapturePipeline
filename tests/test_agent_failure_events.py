"""A producer-bound failed child starts one investigation across worker restarts."""
import json
from pathlib import Path
import time
import pytest

from blueprint_pipeline.agent_execution.failure_events import FailureSubscription, discover_retained_failures
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from tests.test_agent_production_service import fixture, write
from tests.test_task_evaluation_stage_replay import _queue


@pytest.mark.parametrize("managed_enabled", [False, True])
def test_new_failed_child_is_admitted_once_without_model_or_paid_execution(tmp_path, managed_enabled):
    service, _, template_path, config_path = fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["automatic_failure_investigation"] = True
    config["managed_api_enabled"] = managed_enabled
    write(config_path, config)
    template = json.loads(template_path.read_text())
    template["autostart"] = False
    write(template_path, template)
    service = ProductionAgentService(config_path, source_commit="a" * 40)
    queue, _, job = _queue(tmp_path / "saved")
    subscription = FailureSubscription(subscription_id="subscription", run_id="scene-fixture",
        parent_preparation_id=job["parent_preparation_id"], parent_request_digest=job["parent_request_digest"],
        child_queue_root=str(queue), parent_queue_root=str(tmp_path / "parents"), input_root=str(tmp_path),
        approved_roots=(str(tmp_path),), owner_client_id="fixture-client", expires_at=time.time() + 600)
    write(service.journal.root / "failure-subscriptions/subscription.json", subscription.model_dump(mode="json"))
    first = discover_retained_failures(service)
    assert len(first) == 1 and first[0]["state"] == "queued"
    task = service.journal.task(first[0]["task_id"])
    assert task["session_id"] is None and task["result"] is None
    assert task["task"]["admission"]["runtime"] == "openai_agents_sdk"
    restarted = ProductionAgentService(config_path, source_commit="a" * 40)
    assert discover_retained_failures(restarted) == first
    state = json.loads((service.journal.root / "failure-subscription-state/subscription.json").read_text())
    assert len(state["tasks"]) == 1
    assert state["tasks"][0]["reserved_inference_usd"] == 1


def test_automatic_failure_binds_only_its_preauthorized_controller_and_revokes(tmp_path):
    from tests.test_agent_controller_recovery import prepare
    from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
    service, original, binding = prepare(tmp_path)
    replay = original.stage_replays[0]
    job = json.loads(next((Path(replay.queue_root) / "failed").glob("*.json")).read_text())
    config = service.config.model_dump(mode="json")
    config.update(automatic_failure_investigation=True, automatic_recovery_bindings=[binding.model_dump(mode="json")])
    write(service.config_path, config)
    service = ProductionAgentService(service.config_path, source_commit="a" * 40)
    subscription = FailureSubscription(subscription_id="recovery-subscription", run_id=binding.intent_id,
        parent_preparation_id=job["parent_preparation_id"], parent_request_digest=job["parent_request_digest"],
        child_queue_root=replay.queue_root, parent_queue_root=replay.parent_queue_root, input_root=replay.input_root,
        approved_roots=replay.approved_roots, owner_client_id="fixture-client", expires_at=time.time() + 300)
    write(service.journal.root / "failure-subscriptions/recovery-subscription.json", subscription.model_dump(mode="json"))
    discovered = discover_retained_failures(service)
    assert discovered == [{"task_id": discovered[0]["task_id"], "state": "queued"}]
    record = service.record(discovered[0]["task_id"])
    assert record.controller_recoveries == (binding,)
    assert "request_preauthorized_scene_progression" in record.task.tool_ids
    assert service.journal.task(record.task.task_id)["session_id"] is None
    assert discover_retained_failures(service) == discovered
    service.validate_admission(record.task)
    # Config revocation survives a service restart, before a tool can dispatch.
    config["automatic_recovery_bindings"] = []
    write(service.config_path, config)
    fresh = ProductionAgentService(service.config_path, source_commit="a" * 40)
    with pytest.raises(AgentExecutionError, match="automatic_recovery_scope_revoked"):
        fresh.validate_admission(record.task)


@pytest.mark.parametrize("mismatch", ["parent", "intent", "ambiguous"])
def test_unmatched_or_ambiguous_recovery_never_acquires_action_scope(tmp_path, mismatch):
    from tests.test_agent_controller_recovery import prepare
    service, original, binding = prepare(tmp_path)
    replay = original.stage_replays[0]
    job = json.loads(next((Path(replay.queue_root) / "failed").glob("*.json")).read_text())
    changed = binding.model_dump(mode="json")
    if mismatch == "parent":
        changed["parent_request_digest"] = "sha256:" + "0" * 64
    elif mismatch == "intent":
        changed["intent_id"] = "scene-other"
    config = service.config.model_dump(mode="json")
    config.update(automatic_failure_investigation=True,
        automatic_recovery_bindings=[changed, changed] if mismatch == "ambiguous" else [changed])
    write(service.config_path, config)
    service = ProductionAgentService(service.config_path, source_commit="a" * 40)
    subscription = FailureSubscription(subscription_id="bounded", run_id=binding.intent_id,
        parent_preparation_id=job["parent_preparation_id"], parent_request_digest=job["parent_request_digest"],
        child_queue_root=replay.queue_root, parent_queue_root=replay.parent_queue_root, input_root=replay.input_root,
        approved_roots=replay.approved_roots, owner_client_id="fixture-client", expires_at=time.time() + 300)
    write(service.journal.root / "failure-subscriptions/bounded.json", subscription.model_dump(mode="json"))
    discovered = discover_retained_failures(service)
    if mismatch == "ambiguous":
        assert discovered == [{"subscription_id": "bounded", "state": "admission_refused"}]
    else:
        record = service.record(discovered[0]["task_id"])
        assert not record.controller_recoveries
        assert "request_preauthorized_scene_progression" not in record.task.tool_ids
    assert not list((service.journal.root / "controller-requests/requests").glob("*.json"))
