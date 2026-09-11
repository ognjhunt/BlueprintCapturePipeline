"""A producer-bound failed child starts one investigation across worker restarts."""
import json
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
