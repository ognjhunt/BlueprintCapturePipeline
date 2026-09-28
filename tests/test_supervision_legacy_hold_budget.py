"""Legacy investigation grants remain charged to the owner's lifetime cap."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, digest
from blueprint_pipeline.agent_execution.failure_events import FailureSubscription
from blueprint_pipeline.agent_execution.journal import AgentJournal
from blueprint_pipeline.agent_execution.supervision_authority import SupervisionAllowance
from blueprint_pipeline.agent_execution.supervision_budget import failure_hold_inventory
from blueprint_pipeline.agent_execution.supervision_producer import reserve_automatic_revision


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.chmod(0o600)


@pytest.fixture
def legacy_holds(tmp_path):
    root = tmp_path.resolve()
    run_id, intent_digest = "scene-owned", "sha256:" + "a" * 64
    journal = AgentJournal(root / "journal")
    config = SimpleNamespace(task_store_root=root / "tasks", automatic_supervision_allowances=())
    service = SimpleNamespace(journal=journal, config=config)
    policy = FailureSubscription(subscription_id="preparation-original", run_id=run_id,
        parent_preparation_id="parent-original", parent_request_digest="sha256:" + "b" * 64,
        child_queue_root=str(root / "children"), parent_queue_root=str(root / "parents"),
        input_root=str(root / "inputs"), approved_roots=(str(root),), expires_at=time.time() + 3600)
    policy_path = journal.root / "failure-subscriptions" / (policy.subscription_id + ".json")
    _write(policy_path, policy.model_dump(mode="json"))
    rows = [{"task_id": f"auto-failure-original-{number}", "reserved_inference_usd": 1,
             "source_commit": "c" * 40} for number in range(3)]
    state_path = journal.root / "failure-subscription-state" / policy_path.name
    state = {"subscription_digest": digest(policy.model_dump(mode="json")), "tasks": rows}
    _write(state_path, state)
    # Two persisted grants plus one reservation whose task was never retained:
    # absence of the third task must not invent a refund.
    for row in rows[:2]:
        _write(config.task_store_root / (row["task_id"] + ".json"), {"task": {
            "task_id": row["task_id"], "run_id": run_id, "source_commit": row["source_commit"],
            "admission": {"inference_budget_usd": 1}}})
    for number in range(3):
        task_id = f"watch-original-{number}"
        journal.record_event("automatic_supervision_budget_" + intent_digest[7:] + "_" + digest(task_id)[7:],
            {"task_id": task_id, "reserved_inference_usd": 1, "maximum_revisions": 3,
             "maximum_reserved_inference_usd": 3})
    return service, run_id, intent_digest, state_path, state, policy_path


def _allow(service, run_id, intent_digest, cap):
    allowance = SupervisionAllowance(intent_id=run_id, intent_digest=intent_digest,
        maximum_revisions=cap, maximum_reserved_inference_usd=cap,
        expires_at=time.time() + 1800, authorization_reference="owner-approval")
    service.config.automatic_supervision_allowances = (allowance,)
    return SimpleNamespace(run_id=run_id, automatic_intent_digest=intent_digest,
        automatic_allowance_digest=allowance.allowance_digest, maximum_revisions=cap,
        maximum_reserved_inference_usd=cap, expires_at=time.time() + 600)


def test_six_old_holds_block_seventh_then_exact_three_revision_amendment(legacy_holds):
    service, run_id, intent_digest, state_path, _, policy_path = legacy_holds
    originals = (state_path.read_bytes(), policy_path.read_bytes())
    plan = _allow(service, run_id, intent_digest, 6)
    assert not reserve_automatic_revision(service, plan, "watch-seventh", 1)
    assert not reserve_automatic_revision(service, plan, "watch-seventh", 1)
    assert len(failure_hold_inventory(service, run_id)) == 3
    plan = _allow(service, run_id, intent_digest, 9)
    for task in ("watch-seventh", "watch-eighth", "watch-ninth"):
        assert reserve_automatic_revision(service, plan, task, 1)
        assert reserve_automatic_revision(service, plan, task, 1)
    assert not reserve_automatic_revision(service, plan, "watch-tenth", 1)
    with service.journal._connect() as connection:
        rows = connection.execute("SELECT payload_json FROM events WHERE event_id LIKE ?",
            ("automatic_supervision_budget_" + intent_digest[7:] + "_%",)).fetchall()
    values = [json.loads(row["payload_json"]) for row in rows]
    assert len(values) == len({row["task_id"] for row in values}) == 9
    assert sum(row["reserved_inference_usd"] for row in values) == 9
    assert (state_path.read_bytes(), policy_path.read_bytes()) == originals


@pytest.mark.parametrize("amount", [.5, 2, float("nan")])
def test_changed_failure_amount_cannot_reduce_or_rewrite_grant(legacy_holds, amount):
    service, run_id, _, state_path, state, _ = legacy_holds
    state["tasks"][0]["reserved_inference_usd"] = amount
    _write(state_path, state)
    with pytest.raises(AgentExecutionError, match="failure_reservation_invalid"):
        failure_hold_inventory(service, run_id)


@pytest.mark.parametrize("field", ["run_id", "source_commit", "admission"])
def test_existing_task_must_match_its_reserved_owner_release_and_grant(legacy_holds, field):
    service, run_id, _, _, state, _ = legacy_holds
    path = service.config.task_store_root / (state["tasks"][0]["task_id"] + ".json")
    record = json.loads(path.read_text())
    record["task"][field] = {"inference_budget_usd": .5} if field == "admission" else "changed"
    _write(path, record)
    with pytest.raises(AgentExecutionError, match="failure_task_grant_changed"):
        failure_hold_inventory(service, run_id)
