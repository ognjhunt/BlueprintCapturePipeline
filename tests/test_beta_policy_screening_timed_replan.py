"""The refusal default: an unspent refusal no other rule owns is planned again by itself on the shared
schedule, at the same commit, bounded per site and candidate, and only for its own reason."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.account_monthly_spend import LIMIT_REACHED, REPORT_INVALID, REPORT_STALE
from blueprint_pipeline.beta_policy_screening_failure_scope import (
    POLICY_PROVISIONING_BLOCKER,
    TRANSIENT_PROVISIONING_BLOCKERS,
)
from blueprint_pipeline.job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication
from tests.test_beta_policy_screening_replan import NEW, OLD, successor

DAY = 24 * 3600
REFUSAL = "beta_screening_allocator_refused"
REFUSED = ("blocked", [REFUSAL])


def refused(*codes):
    return Backend({"status": "blocked", "blockers": list(codes or [REFUSAL]),
                    "provider_allocation_performed": False, "no_allocation_proven": True})


def run(root, backend, launches=2):
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend,
                                        max_launches=launches)


def ledger(root):
    return coordinator._ledger(coordinator._root(root))


def age(root, campaign_id, seconds):
    """The campaign last changed ``seconds`` ago: the close of a row whose attempt recorded none."""
    value = ledger(root)
    value["campaigns"][campaign_id]["updated_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    coordinator._save(coordinator._root(root), value)


def set_rows(root, campaign_id, rows_by_id):
    """Record each named row as ``(status, blockers)`` with nothing spent."""
    value = ledger(root)
    states = value["campaigns"][campaign_id]["candidates"]
    for candidate_id, (status, blockers) in rows_by_id.items():
        if candidate_id in states:
            states[candidate_id] = {"status": status, "blockers": list(blockers)}
    coordinator._save(coordinator._root(root), value)


def earlier_refusals(root, candidate_id, count, *, size=2):
    """``count`` earlier campaigns of the site that ended with this candidate refused the same way."""
    for index in range(count):
        campaign_id = f"earlier-{candidate_id}-{index}"
        coordinator.enqueue_screening_campaign(plan(campaign_id=campaign_id, count=size), queue_root=root)
        set_rows(root, campaign_id, {f"candidate-{n:03}": ("unavailable", ["candidate_disabled"]) for n in range(size)})
        set_rows(root, campaign_id, {candidate_id: REFUSED})


def scope(root, campaign_id, commit=OLD, **options):
    return coordinator.replan_scope(campaign_id, implementation_commit=commit, queue_root=root, **options)


def rows(root, campaign_id="campaign-1"):
    return [(row["status"], row["blockers"]) for row in publication(root, campaign_id)["candidates"]]


def test_a_refusal_no_rule_owns_is_planned_again_on_the_exact_schedule_at_most_six_times(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    current = "campaign-1"
    for index, delay in enumerate(REFUSAL_RETRY_DELAYS_SECONDS):
        run(tmp_path, refused())
        assert rows(tmp_path, current) == [REFUSED]
        age(tmp_path, current, delay - 5)
        assert scope(tmp_path, current) is None
        age(tmp_path, current, delay)
        assert scope(tmp_path, current) == ["candidate-000"]
        following = f"campaign-1-{index}"
        coordinator.supersede_campaign(current, successor(plan(), campaign_id=following, commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
        assert [row["status"] for row in publication(tmp_path, following)["candidates"]] == ["pending"]
        current = following
    # The seventh refusal is never planned again by itself; nothing was spent on the way.
    run(tmp_path, refused())
    age(tmp_path, current, 30 * DAY)
    assert scope(tmp_path, current) is None
    assert publication(tmp_path, current)["budget"]["committed_upper_bound_usd"] == "0"
    # Another campaign of the same site reads the same count; another site keeps its own.
    coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-2"), queue_root=tmp_path)
    run(tmp_path, refused())
    age(tmp_path, "campaign-2", 30 * DAY)
    assert scope(tmp_path, "campaign-2") is None
    coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-3", site_id="site-job-3"), queue_root=tmp_path)
    run(tmp_path, refused())
    age(tmp_path, "campaign-3", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-3") == ["candidate-000"]
    # A deploy still plans the exhausted one again.
    assert scope(tmp_path, current, NEW) == ["candidate-000"]


def test_the_wait_runs_from_the_rows_own_close_when_its_attempt_recorded_one(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    run(tmp_path, refused())
    value = ledger(tmp_path)
    attempt = value["campaigns"]["campaign-1"]["candidates"]["candidate-000"]["attempt"]
    attempt["closed_at"] = (datetime.now(timezone.utc) - timedelta(seconds=REFUSAL_RETRY_DELAYS_SECONDS[0])).isoformat()
    coordinator._save(coordinator._root(tmp_path), value)
    assert scope(tmp_path, "campaign-1") == ["candidate-000"]
    for stamp in ("not a time", "2000-01-01T00:00:00"):  # unreadable or naive: not yet
        attempt["closed_at"] = stamp
        value["campaigns"]["campaign-1"]["updated_at"] = stamp
        coordinator._save(coordinator._root(tmp_path), value)
        assert scope(tmp_path, "campaign-1") is None


@pytest.mark.parametrize("code", [
    coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER, coordinator.TRANSPORT_RETRY_CAP_BLOCKER,
    "beta_screening_capacity_preflight_failed", "beta_screening_preparation_failed",
    "beta_screening_static_preflight_blocked:bundle_missing", "beta_screening_not_allocated",
    "beta_screening_prelaunch_refused:oserror",
])
def test_failures_to_try_are_the_defaults_and_other_rules_codes_are_not(code):
    owned = code == coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER
    assert coordinator._timed_retry_may_touch(code) is not owned
    assert coordinator._unowned_refusal({"status": "blocked", "blockers": [code]}) is not owned


@pytest.mark.parametrize("codes", [
    # Money: the paid admission, the spend lock, billing, the account limit and its report, the job ceiling.
    ["paid_resource_admission_not_admitted"], ["spend_admission:spend_admission_lock_stale"],
    ["spend_admission_lock_stale"], ["spend_admission_lock_required_provider_billing_missing:vast"],
    [LIMIT_REACHED], [REPORT_STALE], [REPORT_INVALID], [coordinator.G1_ADMISSION_WRAPPER],
    ["beta_screening_job_ceiling_exceeded"], ["website_job_ceiling_unavailable"],
    ["beta_screening_cost_quote_invalid_or_expired"],
    # Privacy and rights.
    ["beta_screening_current_source_revoked"], ["beta_screening_authority_expired"],
    ["beta_screening_prelaunch_refused:beta_screening_current_source_revoked_before_spawn"],
    # Answers about the candidate.
    ["beta_screening_current_source_stopped"], ["beta_screening_execution_control_refused"],
    [coordinator.INFEASIBLE_BLOCKER],
    # Codes another rule owns.
    [coordinator.COMMIT_CHANGED_BLOCKER], ["g1_beta_implementation_commit_changed"],
    [coordinator.HARNESS_HOLD_BLOCKER], [coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER],
    [POLICY_PROVISIONING_BLOCKER], ["policy_canary_session_setup_failed:policy_load"],
    *[[code] for code in sorted(TRANSIENT_PROVISIONING_BLOCKERS)],
    # One protected code beside a failure to try protects the row.
    [REFUSAL, "beta_screening_cost_quote_invalid_or_expired"],
])
def test_privacy_rights_money_answers_and_other_rules_codes_are_never_touched_by_the_default(tmp_path, codes):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    set_rows(tmp_path, "campaign-1", {"candidate-000": ("blocked", codes)})
    age(tmp_path, "campaign-1", 30 * DAY)
    row = ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert not coordinator._unowned_refusal(row)
    if (codes not in ([coordinator.COMMIT_CHANGED_BLOCKER], [coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER])
            and not coordinator._machine_faulted(row)):
        # Those are re-planned by their own rules (the commit path, the GPU-offer cool-off, the
        # machine-fault bound for transient infrastructure).
        assert scope(tmp_path, "campaign-1") is None


def test_a_due_refusal_never_carries_a_sibling_whose_own_wait_has_not_passed(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    earlier_refusals(tmp_path, "candidate-001", 1)  # its next wait is an hour
    set_rows(tmp_path, "campaign-1", {"candidate-000": REFUSED, "candidate-001": REFUSED})
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-000"]
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2), commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-000"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    assert rows(tmp_path)[1] == REFUSED
    # Left behind, it is not lost: the next deploy's re-plan takes it up from the chain.
    assert scope(tmp_path, "campaign-1b", NEW) == ["candidate-000", "candidate-001"]


def test_a_due_offer_shortage_never_carries_a_refusal_and_a_due_refusal_never_carries_it(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    earlier_refusals(tmp_path, "candidate-001", 4)  # its next wait is twelve hours
    stamp = datetime.now(timezone.utc).isoformat()
    shortage = {"status": "blocked", "blockers": [coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER],
                "attempt": {"attempt_id": "screen-" + "0" * 32, "generation": 0, "maximum_cost_usd": "4",
                            "reservation_status": "released", "provider_allocation_observed": False,
                            "closed_at": stamp}}
    value = ledger(tmp_path)
    value["campaigns"]["campaign-1"]["candidates"].update({"candidate-000": shortage,
                                                           "candidate-001": {"status": "blocked",
                                                                             "blockers": [REFUSAL]}})
    coordinator._save(coordinator._root(tmp_path), value)
    # The refusal's wait has passed (its campaign last changed 12 h ago); the shortage closed just now.
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[4])
    assert scope(tmp_path, "campaign-1") == ["candidate-001"]
    # The shortage's two hours have passed; the refusal's twelve have not.
    value = ledger(tmp_path)
    value["campaigns"]["campaign-1"]["candidates"]["candidate-000"]["attempt"]["closed_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=coordinator.OFFER_SHORTAGE_REPLAN_COOLOFF_SECONDS)).isoformat()
    value["campaigns"]["campaign-1"]["updated_at"] = stamp
    coordinator._save(coordinator._root(tmp_path), value)
    assert scope(tmp_path, "campaign-1") == ["candidate-000"]


def test_rows_that_are_not_blocked_still_go_with_a_due_refusal(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=3, cost="2.00"), queue_root=tmp_path)
    run(tmp_path, refused(), launches=1)
    assert [status for status, _ in rows(tmp_path)] == ["blocked", "pending", "pending"]
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-000", "candidate-001", "candidate-002"]
    coordinator.supersede_campaign("campaign-1", successor(plan(count=3, cost="2.00"), commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    # The same row transitions as a commit re-plan: queued rows close as re-planned.
    assert rows(tmp_path) == [REFUSED, ("blocked", [coordinator.REPLANNED_BLOCKER]),
                              ("blocked", [coordinator.REPLANNED_BLOCKER])]


def test_a_closed_job_window_or_self_replan_false_leaves_the_default_out(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    run(tmp_path, refused())
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1", self_replan=False) is None
    assert scope(tmp_path, "campaign-1") == ["candidate-000"]
    assert coordinator.close_self_replan("campaign-1", queue_root=tmp_path) is True
    age(tmp_path, "campaign-1", 30 * DAY)
    assert scope(tmp_path, "campaign-1") is None
    # A deploy's re-plan is unchanged.
    assert scope(tmp_path, "campaign-1", NEW) == ["candidate-000"]


def test_the_existing_gates_hold_no_active_row_no_reservation_not_superseded(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    run(tmp_path, refused(), launches=1)
    run(tmp_path, Backend(RuntimeError("provider response lost")), launches=1)
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"
    age(tmp_path, "campaign-1", 30 * DAY)
    assert scope(tmp_path, "campaign-1") is None
    coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-3", site_id="site-job-3"), queue_root=tmp_path)
    run(tmp_path, refused())
    age(tmp_path, "campaign-3", REFUSAL_RETRY_DELAYS_SECONDS[0])
    coordinator.supersede_campaign("campaign-3", successor(plan(campaign_id="campaign-3", site_id="site-job-3"),
                                                           campaign_id="campaign-3b", commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    age(tmp_path, "campaign-3", 30 * DAY)
    assert scope(tmp_path, "campaign-3") is None


def test_a_spent_row_is_never_carried_and_supersede_refuses_it(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    run(tmp_path, Backend({"status": "blocked", "blockers": ["vast_instance_hard_ttl_expired"],
                           "provider_allocation_performed": True, "provider_zero_verified": True}), launches=1)
    run(tmp_path, refused(), launches=1)
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):  # not yet due
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}, commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-001"]
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2), commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    assert publication(tmp_path, "campaign-1b")["budget"]["spent_upper_bound_usd"] == "4"


@pytest.mark.parametrize("extra", [{}, {"timed_replan_candidate_ids": ["candidate-000"]}])
def test_an_old_ledger_loads_and_a_field_from_the_dropped_mechanism_is_ignored(tmp_path, extra):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    run(tmp_path, refused())
    value = ledger(tmp_path)
    value["campaigns"]["campaign-1"].update(extra)
    coordinator._save(coordinator._root(tmp_path), value)
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-000"]
