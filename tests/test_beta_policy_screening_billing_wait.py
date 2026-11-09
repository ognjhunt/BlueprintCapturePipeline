# Covers (for impacted-test selection):
#   src/blueprint_pipeline/beta_policy_screening_coordinator.py
"""A run refused only for a missing provider billing reading waits for the reading and runs every money check again.

Owner decision 2026-11-09 about 04:32Z ("yes", relayed by the library-aisle session): the library site's G1 rows
were refused this way three times in a day (11-08 02:27Z, 21:35Z, 11-09 04:05Z), each time $0 and no provider call,
and each waited for a deploy before it was planned again.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication

MISSING = {"status": "blocked", "provider_allocation_performed": False, "no_allocation_proven": True,
           "blockers": ["spend_admission_lock_required_provider_billing_missing:vast"]}


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "paid_spend_admission_lock.json"
    monkeypatch.setenv("BLUEPRINT_PAID_SPEND_ADMISSION_LOCK_PATH", str(path))

    def write(covered):
        path.write_text(json.dumps({"billing_covered_provider_ids": covered}))
    write(["digitalocean"])
    return write


def _due_now(monkeypatch):
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(coordinator, "_retry_after_offer_shortage", lambda _retries: past)


def _state(root):
    return coordinator._ledger(coordinator._root(root))["campaigns"]["campaign-1"]["candidates"]["candidate-000"]


def _run(root, backend, times=1):
    for _ in range(times):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend)


def test_a_missing_billing_reading_waits_for_the_lock_to_cover_the_provider_then_runs_again(tmp_path, lock,
                                                                                            monkeypatch):
    _due_now(monkeypatch)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend(MISSING)
    _run(tmp_path, backend)
    row = publication(tmp_path)["candidates"][0]
    assert (row["status"], row["blockers"]) == (
        "preparing", [coordinator.CAPACITY_WAIT_BLOCKER, coordinator.CONTROL_WAIT_BLOCKER])
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "0" and budget["reserved_usd"] == "4"
    state = _state(tmp_path)
    assert state["billing_missing_retries"] == 1 and state["attempt"]["generation"] == 1
    assert state["attempt"][coordinator.BILLING_WAIT_PROVIDER_FIELD] == "vast"
    closed = state["attempt_history"][0]
    assert closed["terminal_reason"] == "spend_admission_lock_provider_billing_missing"
    assert closed["billing_missing_provider"] == "vast" and closed["reservation_status"] == "released"
    # Due, but the lock still has no Vast reading: nothing launches.
    _run(tmp_path, backend, times=2)
    assert backend.launches == 1
    # The reading is back: the next pass launches, and its preflight checks the money again.
    lock(["digitalocean", "vast"])
    backend.observation = {"status": "running"}
    _run(tmp_path, backend)
    assert backend.launches == 2
    assert not coordinator._harness_failed(_state(tmp_path))


def test_the_wait_is_bounded_by_count_and_then_keeps_the_refusal(tmp_path, lock, monkeypatch):
    monkeypatch.setattr(coordinator, "BILLING_MISSING_RETRY_CAP", 2)
    _due_now(monkeypatch)
    lock(["vast"])  # covered, yet each preflight refuses again
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend(MISSING)
    _run(tmp_path, backend, times=4)
    assert backend.launches == 3
    row = publication(tmp_path)["candidates"][0]
    assert row["status"] == "blocked"
    assert "spend_admission_lock_required_provider_billing_missing:vast" in row["blockers"]
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "0" and budget["reserved_usd"] == "0"
    state = _state(tmp_path)
    assert state["billing_missing_retries"] == 2 and len(state["attempt_history"]) == 2
    # Never a harness failure: it spends none of the candidate's harness retries.
    assert not coordinator._harness_failed(state)


def test_the_wait_is_bounded_in_time_and_then_keeps_the_refusal(tmp_path, lock, monkeypatch):
    _due_now(monkeypatch)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend(MISSING)
    _run(tmp_path, backend)
    assert backend.launches == 1
    # A day later the provider is still uncovered: the wait no longer holds the launch back, and the
    # next refusal stands instead of waiting again.
    monkeypatch.setattr(coordinator, "_billing_wait_open", lambda _row: False)
    _run(tmp_path, backend)
    assert backend.launches == 2
    row = publication(tmp_path)["candidates"][0]
    assert row["status"] == "blocked" and publication(tmp_path)["budget"]["reserved_usd"] == "0"


@pytest.mark.parametrize("observation", [
    # Another money stop beside it is a stop, not this wait.
    {**MISSING, "blockers": [*MISSING["blockers"], "spend_admission:spend_admission_threshold_crossed"]},
    {**MISSING, "blockers": [*MISSING["blockers"], "spend_admission_lock_stale"]},
    # Two providers uncovered at once is not one missing reading.
    {**MISSING, "blockers": [*MISSING["blockers"], "spend_admission_lock_required_provider_billing_missing:gcp"]},
    # A run that reached a provider never waits here.
    {**MISSING, "provider_allocation_performed": True, "provider_zero_verified": True, "no_allocation_proven": False},
])
def test_no_other_refusal_waits(tmp_path, lock, observation):
    lock(["vast"])
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    _run(tmp_path, Backend(observation))
    row = publication(tmp_path)["candidates"][0]
    assert row["status"] == "blocked" and coordinator.CONTROL_WAIT_BLOCKER not in row["blockers"]
    assert "billing_missing_retries" not in _state(tmp_path)


def test_the_admission_wrapper_codes_beside_it_still_wait():
    observation = {**MISSING, "blockers": ["paid_resource_admission_has_blockers", "paid_resource_admission_not_admitted",
                                           "spend_admission:spend_admission_lock_required_provider_billing_missing:vast"]}
    assert coordinator._billing_missing_provider(observation) == "vast"
    assert coordinator._billing_missing_provider({"blockers": ["paid_resource_admission_not_admitted"]}) is None


def test_an_unreadable_lock_reads_as_not_covered(tmp_path, monkeypatch):
    monkeypatch.delenv("BLUEPRINT_PAID_SPEND_ADMISSION_LOCK_PATH", raising=False)
    assert coordinator._billing_covers("vast") is False
    path = tmp_path / "lock.json"
    monkeypatch.setenv("BLUEPRINT_PAID_SPEND_ADMISSION_LOCK_PATH", str(path))
    assert coordinator._billing_covers("vast") is False
    path.write_text("not json")
    assert coordinator._billing_covers("vast") is False
    path.write_text(json.dumps({"billing_covered_provider_ids": "vast"}))
    assert coordinator._billing_covers("vast") is False
