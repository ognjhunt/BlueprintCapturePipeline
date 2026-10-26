"""Fleet saturation waits; only retained zero-spend proofs permit a fresh intent."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_launch import _record
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication


def state(root):
    return coordinator._ledger(root)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]


def expire_wait(root):
    coordinator._update(root, "campaign-1", "candidate-000", lambda row:
        row["attempt"].update(retry_not_before="2000-01-01T00:00:00+00:00"))


class CapacityBackend(Backend):
    def __init__(self):
        super().__init__()
        self.busy = True
        self.inspections = 0

    def preflight_candidate_capacity(self, **kwargs):
        self.inspections += 1
        return {"capacity_available": not self.busy,
                "blockers": ["vast_paid_launch_lock_busy"] if self.busy else []}


class RaceBackend(Backend):
    def __init__(self):
        super().__init__()
        self.busy = True
        self.paths = []
        self.allocations = 0
        self.proof_override = {}

    def launch_candidate(self, **kwargs):
        self.launches += 1
        self.paths.append(kwargs["attempt_dir"])
        if not self.busy:
            self.allocations += 1
            return {"status": "running", "provider_allocation_performed": True}
        receipt = {"status": "blocked", "blockers": ["vast_paid_launch_lock_busy"],
                   "provider_mutations_performed": 0, "provider_create_attempted": False}
        path = kwargs["attempt_dir"] / "allocator_result.json"
        path.write_text(json.dumps(receipt))
        return {"status": "blocked", "blockers": ["vast_paid_launch_lock_busy"],
                "provider_allocation_performed": False, "no_allocation_proven": True,
                "provider_zero_verified": True, "no_policy_api_calls_proven": True,
                "retryable_capacity_refusal": True, "capacity_refusal_receipt": _record(path),
                **self.proof_override}


def process(root, backend):
    return coordinator.process_screening_queue(queue_root=root, executor=backend, require_publication_ack=False)


def test_saturated_fleet_waits_before_any_launch_intent_then_starts_same_attempt(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = CapacityBackend()
    assert process(tmp_path, backend)["launches_started"] == 0
    waiting = state(tmp_path)
    assert waiting["status"] == "preparing" and waiting["blockers"] == [coordinator.CAPACITY_WAIT_BLOCKER]
    assert "launch_intent_at" not in waiting["attempt"]
    assert not waiting.get("attempt_history")
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"
    assert process(tmp_path, backend)["launches_started"] == 0
    assert backend.inspections == 1 and backend.preparations == 1 and backend.launches == 0
    backend.busy = False
    expire_wait(tmp_path)
    assert process(tmp_path, backend)["launches_started"] == 1
    assert backend.launches == 1 and state(tmp_path)["attempt"]["attempt_id"] == waiting["attempt"]["attempt_id"]


def test_proven_slot_race_archives_zero_spend_intent_and_uses_one_fresh_attempt(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = RaceBackend()
    process(tmp_path, backend)
    waiting = state(tmp_path)
    assert waiting["status"] == "preparing" and waiting["blockers"] == [coordinator.CAPACITY_WAIT_BLOCKER]
    [old] = waiting["attempt_history"]
    assert old["reservation_status"] == "released" and old["no_policy_api_calls_proven"] is True
    assert old["capacity_refusal_receipt"] == _record(backend.paths[0] / "allocator_result.json")
    assert "launch_intent_at" in old and "launch_intent_at" not in waiting["attempt"]
    assert waiting["attempt"]["attempt_id"] != old["attempt_id"] and waiting["attempt"]["generation"] == 1
    assert publication(tmp_path)["budget"] == {"currency": "USD", "limit_usd": "15",
        "reserved_usd": "4", "spent_upper_bound_usd": "0", "committed_upper_bound_usd": "4"}
    process(tmp_path, backend)
    assert backend.launches == 1 and backend.allocations == 0
    backend.busy = False
    expire_wait(tmp_path)
    process(tmp_path, backend)
    assert backend.launches == 2 and backend.allocations == 1 and backend.paths[0] != backend.paths[1]
    assert state(tmp_path)["attempt_history"] == [old]
    assert publication(tmp_path)["budget"]["committed_upper_bound_usd"] == "4"


@pytest.mark.parametrize("change", [
    {"no_allocation_proven": False},
    {"provider_allocation_performed": True},
    {"no_policy_api_calls_proven": False},
    {"capacity_refusal_receipt": {"path": "/elsewhere", "sha256": "sha256:" + "a" * 64, "size_bytes": 1}},
    {"blockers": ["vast_paid_launch_lock_busy", "another_failure"]},
])
def test_capacity_message_without_exact_zero_spend_proof_never_retries(tmp_path, change):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = RaceBackend()
    backend.proof_override = change
    process(tmp_path, backend)
    row = state(tmp_path)
    assert not row.get("attempt_history") and row["attempt"]["generation"] == 0
    assert row["status"] in {"blocked", "reconciliation_required"}
    process(tmp_path, backend)
    assert backend.launches == 1


def test_allocated_attempt_can_never_be_rotated_by_a_later_capacity_refusal(tmp_path):
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = RaceBackend()
    backend.busy = False
    process(tmp_path, backend)
    attempt = state(tmp_path)["attempt"]
    backend.busy = True
    observation = backend.launch_candidate(attempt_dir=tmp_path / "attempts" / attempt["attempt_id"])
    coordinator._observe(tmp_path, value, value["candidates"][0], observation)
    row = state(tmp_path)
    assert not row.get("attempt_history") and row["attempt"]["generation"] == 0
    assert row["attempt"]["reservation_status"] == "committed"
    assert publication(tmp_path)["budget"]["spent_upper_bound_usd"] == "4"


def test_new_capacity_attempt_requires_latest_publication_ack(tmp_path):
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    queued = publication(tmp_path)
    coordinator.acknowledge_publication(value["campaign_id"], queued["publication_digest"], queue_root=tmp_path)
    backend = RaceBackend()
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    expire_wait(tmp_path)
    backend.busy = False
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.launches == 1
    waiting = publication(tmp_path)
    coordinator.acknowledge_publication(value["campaign_id"], waiting["publication_digest"], queue_root=tmp_path)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.launches == 2 and backend.allocations == 1
