"""Controller releases change only after every old reasoning session is clean."""
import json

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.release import adopt_drained_config, drain
from tests.test_agent_production_service import fixture


def test_config_adoption_retains_old_config_and_refuses_a_pending_task(tmp_path):
    service, task, _, path = fixture(tmp_path)
    previous = path.read_bytes()
    service.enqueue(task.task_id, "fixture-client")
    with pytest.raises(AgentExecutionError, match="requires_drain"):
        adopt_drained_config(path, expected_commit="b" * 40)
    assert path.read_bytes() == previous


def test_queued_task_drains_without_a_model_call_and_next_release_can_adopt(tmp_path):
    service, task, _, path = fixture(tmp_path)
    service.enqueue(task.task_id, "fixture-client")
    receipt = drain(service, target_source_commit="b" * 40, timeout_seconds=10)
    assert receipt["status"] == "drained"
    state = service.journal.task(task.task_id)
    assert state["state"] == "cancelled" and state["cleanup_state"] == "deleted"
    assert service.autostart() == 0
    adoption = adopt_drained_config(path, expected_commit="b" * 40)
    assert adoption["status"] == "adopted"
    assert json.loads(path.read_text())["source_commit"] == "b" * 40
    retained = json.loads((service.journal.root / "release-adoptions" / ("b" * 40 + ".json")).read_text())
    assert retained["previous_config"]["source_commit"] == "a" * 40
    assert service.journal.task(task.task_id) == state
    from blueprint_pipeline.agent_execution.production import ProductionAgentService
    newer = ProductionAgentService(path, source_commit="b" * 40)
    assert drain(newer, target_source_commit="a" * 40)["status"] == "drained"
    assert adopt_drained_config(path, expected_commit="a" * 40)["status"] == "adopted"
    assert json.loads(path.read_text())["source_commit"] == "a" * 40
    assert service.journal.task(task.task_id) == state


def test_failed_deploy_drain_can_be_superseded_after_verified_cleanup(tmp_path):
    service, _, _, path = fixture(tmp_path)
    old_target = "b" * 40
    new_target = "c" * 40
    assert drain(service, target_source_commit=old_target)["status"] == "drained"
    assert json.loads(path.read_text())["source_commit"] == "a" * 40
    assert drain(service, target_source_commit=new_target)["status"] == "drained"
    marker = json.loads((service.journal.root / "release_drain.json").read_text())
    assert marker["target_source_commit"] == new_target
    assert adopt_drained_config(path, expected_commit=new_target)["status"] == "adopted"


def test_failed_deploy_drain_refuses_changed_prior_receipt(tmp_path):
    service, _, _, _ = fixture(tmp_path)
    old_target = "b" * 40
    assert drain(service, target_source_commit=old_target)["status"] == "drained"
    receipt_path = service.journal.root / "release-drains" / f"{old_target}.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "reconciliation_pending"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(AgentExecutionError, match="owner_conflict"):
        drain(service, target_source_commit="c" * 40)
    assert json.loads((service.journal.root / "release_drain.json").read_text())[
        "target_source_commit"
    ] == old_target
