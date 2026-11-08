"""A stale spend admission lock waits and reads the money check again, bounded; it never spends."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication

# 2026-11-05 06:03Z, eight minutes after a deploy restarted the lock's refresh: the allocator's
# pre-spend preflight read billing evidence older than its window and refused before any create.
STALE = {"status": "blocked", "provider_allocation_performed": False, "no_allocation_proven": True,
         "blockers": ["paid_resource_admission_has_blockers", "paid_resource_admission_not_admitted",
                      "spend_admission:spend_admission_lock_stale"]}


def _due_now(monkeypatch):
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(coordinator, "_retry_after_offer_shortage", lambda _retries: past)


def test_a_stale_spend_lock_waits_and_reads_it_again_holding_the_reservation(tmp_path):
    """Before this, the candidate closed as blocked and waited for the next deploy to be planned again."""
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend(STALE)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    row = publication(tmp_path)["candidates"][0]
    assert (row["status"], row["blockers"]) == (
        "preparing", [coordinator.CAPACITY_WAIT_BLOCKER, coordinator.CONTROL_WAIT_BLOCKER])
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "0" and budget["reserved_usd"] == "4"
    state = coordinator._ledger(coordinator._root(tmp_path))["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert state["spend_lock_retries"] == 1 and state["attempt"]["generation"] == 1
    assert state["attempt_history"][0]["terminal_reason"] == "spend_admission_lock_stale"
    # Not due yet: the next pass does not launch again.
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 1


def test_the_wait_is_bounded_and_then_keeps_the_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "SPEND_LOCK_RETRY_CAP", 2)
    _due_now(monkeypatch)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend(STALE)
    for _ in range(4):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 3
    row = publication(tmp_path)["candidates"][0]
    assert row["status"] == "blocked" and "spend_admission:spend_admission_lock_stale" in row["blockers"]
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "0" and budget["reserved_usd"] == "0"


def test_any_other_spend_blocker_or_an_allocated_run_never_waits(tmp_path):
    """A crossed spend threshold beside a stale lock is a money stop, not a wait."""
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    crossed = {**STALE, "blockers": [*STALE["blockers"], "spend_admission:spend_admission_threshold_crossed"]}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=Backend(crossed))
    allocated = {**STALE, "provider_allocation_performed": True, "provider_zero_verified": True,
                 "no_allocation_proven": False}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Backend(allocated))
    rows = publication(tmp_path)["candidates"]
    assert [row["status"] for row in rows] == ["blocked", "blocked"]
    assert all(coordinator.CONTROL_WAIT_BLOCKER not in row["blockers"] for row in rows)


# 2026-11-08 02:40Z, the library site's pi0.5 G1 row: the G1 lane's allocator words the same refusal without
# the ``spend_admission:`` prefix and with no admission codes beside it.
STALE_G1 = {"status": "blocked", "provider_allocation_performed": False, "no_allocation_proven": True,
            "blockers": ["spend_admission_lock_stale"]}


def test_the_g1_lanes_wording_of_a_stale_lock_waits_the_same_way(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend(STALE_G1)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    row = publication(tmp_path)["candidates"][0]
    assert (row["status"], row["blockers"]) == (
        "preparing", [coordinator.CAPACITY_WAIT_BLOCKER, coordinator.CONTROL_WAIT_BLOCKER])
    state = coordinator._ledger(coordinator._root(tmp_path))["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert state["spend_lock_retries"] == 1
    assert publication(tmp_path)["budget"]["spent_upper_bound_usd"] == "0"


def test_a_stale_lock_beside_another_spend_refusal_keeps_its_refusal(tmp_path):
    """Only the stale-lock wait retries: a missing provider billing total beside it is not this wait."""
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend({**STALE_G1, "blockers": ["spend_admission_lock_stale",
                                                "spend_admission_lock_required_provider_billing_missing:vast"]})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert publication(tmp_path)["candidates"][0]["status"] == "blocked"
