"""A GPU market with no qualifying offer waits and searches again, bounded; it never spends."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication

SHORTAGE = {"status": "blocked", "provider_allocation_performed": False, "no_allocation_proven": True,
            "blockers": ["native_task_arena_policy_canary_session_runtime_not_completed",
                         "no_vast_offer_meeting_minimum_driver_version", "vast_offer_selection_blocked"]}


def _due_now(monkeypatch):
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(coordinator, "_retry_after_offer_shortage", lambda _retries: past)


def test_no_qualifying_offer_waits_and_searches_again_holding_the_reservation(tmp_path):
    """2026-11-05 00:21Z: no Vast offer met the price, driver and region rules. Nothing was created;
    the candidate closed as blocked and would have waited for a deploy to be planned again."""
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend(SHORTAGE)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    row = publication(tmp_path)["candidates"][0]
    assert row["status"] == "preparing"
    assert row["blockers"] == [coordinator.CAPACITY_WAIT_BLOCKER, coordinator.OFFER_WAIT_BLOCKER]
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "0" and budget["reserved_usd"] == "4"
    ledger = coordinator._ledger(coordinator._root(tmp_path))
    state = ledger["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert state["offer_shortage_retries"] == 1 and state["attempt"]["generation"] == 1
    assert state["attempt_history"][0]["terminal_reason"] == "no_qualifying_gpu_offer"
    # Not due yet: the next pass does not search again.
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 1


def test_the_wait_is_bounded_and_then_names_the_shortage(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "OFFER_SHORTAGE_RETRY_CAP", 2)
    _due_now(monkeypatch)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend(SHORTAGE)
    for _ in range(4):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 3
    row = publication(tmp_path)["candidates"][0]
    assert (row["status"], row["blockers"]) == ("blocked", [coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER])
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "0" and budget["reserved_usd"] == "0"


def test_an_allocated_run_or_another_refusal_never_waits_for_capacity(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    other = {**SHORTAGE, "blockers": ["paid_resource_admission_not_admitted"]}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=Backend(other))
    allocated = {"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True,
                 "blockers": ["no_vast_offer_meeting_minimum_driver_version"]}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Backend(allocated))
    rows = publication(tmp_path)["candidates"]
    assert [row["status"] for row in rows] == ["blocked", "blocked"]
    assert coordinator.OFFER_WAIT_BLOCKER not in rows[0]["blockers"] + rows[1]["blockers"]
