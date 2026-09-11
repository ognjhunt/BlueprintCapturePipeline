"""One intent cap counts all historical inference producers without refunds."""
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, digest
from blueprint_pipeline.agent_execution.failure_events import FailureSubscription
from blueprint_pipeline.agent_execution.supervision_budget import failure_hold_inventory
from blueprint_pipeline.agent_execution.supervision_producer import reserve_automatic_revision
from tests.test_agent_production_service import write
from tests.test_agent_run_supervision_producer import setup


def seed_holds(service, run_id):
    policy = FailureSubscription(subscription_id="old-failure-subscription", run_id=run_id,
        parent_preparation_id="old-parent", parent_request_digest="sha256:" + "b" * 64,
        child_queue_root="/fixture/children", parent_queue_root="/fixture/parents", input_root="/fixture/inputs",
        approved_roots=("/fixture",), per_task_budget_usd=1, expires_at=time.time() + 600)
    write(service.journal.root / "failure-subscriptions" / (policy.subscription_id + ".json"), policy.model_dump(mode="json"))
    entries = [{"task_id": "auto-failure-" + name, "reserved_inference_usd": 1.0,
                "source_commit": "a" * 40, "job_sha256": "sha256:" + name * 64} for name in ("c", "d")]
    path = service.journal.root / "failure-subscription-state" / (policy.subscription_id + ".json")
    write(path, {"subscription_digest": digest(policy.model_dump(mode="json")), "tasks": entries})
    return path, entries


def test_append_only_migration_counts_missing_task_holds_and_deduplicates_by_task(tmp_path):
    service, plan, _, _ = setup(tmp_path)
    assert reserve_automatic_revision(service, plan, "watch-existing", 1)
    path, entries = seed_holds(service, plan.run_id)
    before = path.read_bytes()
    assert not reserve_automatic_revision(service, plan, "watch-new", 1)
    assert not reserve_automatic_revision(service, plan, "watch-new", 1)
    assert path.read_bytes() == before
    prefix = "automatic_supervision_budget_" + plan.automatic_intent_digest[7:] + "_"
    with service.journal._connect() as db:
        rows = [json.loads(row["payload_json"]) for row in db.execute("SELECT payload_json FROM events WHERE event_id LIKE ?", (prefix + "%",))]
    assert len(rows) == 3 and sum(row["reserved_inference_usd"] for row in rows) == 3
    assert {row["task_id"] for row in rows} == {"watch-existing", *(row["task_id"] for row in entries)}
    # A reserved-before-task-write crash still has a conservative lifetime hold.
    assert not any((Path(service.config.task_store_root) / (entry["task_id"] + ".json")).exists() for entry in entries)


@pytest.mark.parametrize("defect", ["lower_amount", "task_grant", "subscription_digest", "task_path"])
def test_tampered_cross_producer_holds_cannot_understate_the_reserved_grant(tmp_path, defect):
    service, plan, _, _ = setup(tmp_path)
    path, entries = seed_holds(service, plan.run_id)
    state = json.loads(path.read_text())
    if defect == "lower_amount":
        state["tasks"][0]["reserved_inference_usd"] = .1
    elif defect == "subscription_digest":
        state["subscription_digest"] = "sha256:" + "0" * 64
    elif defect == "task_path":
        state["tasks"][0]["task_id"] = "auto-failure-../../outside"
    else:
        write(Path(service.config.task_store_root) / (entries[0]["task_id"] + ".json"), {"task": {
            "task_id": entries[0]["task_id"], "run_id": plan.run_id, "source_commit": "a" * 40,
            "admission": {"inference_budget_usd": .5}}})
    write(path, state)
    with pytest.raises(AgentExecutionError, match="failure_(reservation_invalid|task_grant_changed)"):
        failure_hold_inventory(service, plan.run_id)
    with service.journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE event_id LIKE 'automatic_supervision_budget_%'").fetchone()[0] == 0
