"""A candidate refused before any allocation is planned again on a timer, with no deploy, never spent twice."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_plan import (
    G1_BACKEND,
    g1_cells,
    validate_screening_plan,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication, seal
from tests.test_beta_policy_screening_replan import HARNESS_FAILED, NEW, OLD, successor

DAY = 24 * 3600


def refused(*codes):
    return Backend({"status": "blocked", "blockers": list(codes or ["beta_screening_allocator_refused"]),
                    "provider_allocation_performed": False, "no_allocation_proven": True})


def run(root, backend, launches=2):
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend,
                                        max_launches=launches)


def age(root, campaign_id, seconds):
    """The campaign last changed ``seconds`` ago: the clock the timed retry reads."""
    queue = coordinator._root(root)
    with coordinator._lock(queue):
        ledger = coordinator._ledger(queue)
        ledger["campaigns"][campaign_id]["updated_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
        coordinator._save(queue, ledger)


def ledger(root):
    return coordinator._ledger(coordinator._root(root))


def scope(root, campaign_id, commit=OLD):
    return coordinator.replan_scope(campaign_id, implementation_commit=commit, queue_root=root)


def rows(root, campaign_id="campaign-1"):
    return [(row["status"], row["blockers"]) for row in publication(root, campaign_id)["candidates"]]


def on_g1(value, candidate_id="candidate-001"):
    """The same plan with one candidate on the native G1 backend (its own harness)."""
    row = next(item for item in value["candidates"] if item["candidate_id"] == candidate_id)
    row["execution_spec"]["backend"] = G1_BACKEND
    row["admission_digest"] = canonical_digest(row["execution_spec"])
    row["scenario_cells"] = g1_cells(value["scenario_cells"], None)
    row["episode_count"] = len(row["scenario_cells"])
    row["cost_quote"] = seal({**row["cost_quote"], "episode_count": row["episode_count"]}, "quote_digest")
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


def test_a_refusal_before_allocation_is_planned_again_at_the_same_commit_on_the_schedule(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    run(tmp_path, refused())
    assert rows(tmp_path) == [("blocked", ["beta_screening_allocator_refused"])] * 2
    # Nothing waits for a deploy: 15 minutes after the refusal, the same commit plans it again.
    assert scope(tmp_path, "campaign-1") is None
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0] - 5)
    assert scope(tmp_path, "campaign-1") is None
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-000", "candidate-001"]
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), commit=OLD), implementation_commit=OLD,
                                   queue_root=tmp_path)
    assert ledger(tmp_path)["campaigns"]["campaign-1"][coordinator.TIMED_REPLAN_FIELD] == [
        "candidate-000", "candidate-001"]
    assert [row["status"] for row in publication(tmp_path, "campaign-1b")["candidates"]] == ["pending", "pending"]
    assert publication(tmp_path, "campaign-1b")["budget"]["committed_upper_bound_usd"] == "0"
    # The next refusal waits the schedule's next step: an hour.
    run(tmp_path, refused())
    age(tmp_path, "campaign-1b", REFUSAL_RETRY_DELAYS_SECONDS[1] - 5)
    assert scope(tmp_path, "campaign-1b") is None
    age(tmp_path, "campaign-1b", REFUSAL_RETRY_DELAYS_SECONDS[1])
    assert scope(tmp_path, "campaign-1b") == ["candidate-000", "candidate-001"]


def test_at_most_six_timed_replans_per_site_and_candidate_counted_over_every_campaign(tmp_path):
    """Invariant 1: bounded and durable; a new chain for the same site does not reset it."""
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    current = "campaign-1"
    for index, delay in enumerate(REFUSAL_RETRY_DELAYS_SECONDS):
        run(tmp_path, refused())
        age(tmp_path, current, delay)
        assert scope(tmp_path, current) == ["candidate-000"]
        following = f"campaign-1-{index}"
        coordinator.supersede_campaign(current, successor(plan(), campaign_id=following, commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
        current = following
    run(tmp_path, refused())
    assert coordinator._timed_replans(ledger(tmp_path), "site-job-1", "candidate-000") == 6
    age(tmp_path, current, 30 * DAY)
    assert scope(tmp_path, current) is None
    # Another campaign for the same site (a catalog revision) reads the same count.
    coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-2"), queue_root=tmp_path)
    run(tmp_path, refused())
    age(tmp_path, "campaign-2", 30 * DAY)
    assert scope(tmp_path, "campaign-2") is None
    # Another site keeps its own count.
    coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-3", site_id="site-job-3"), queue_root=tmp_path)
    run(tmp_path, refused())
    age(tmp_path, "campaign-3", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-3") == ["candidate-000"]
    # A deploy still re-plans the exhausted one, and records no timed re-plan for it.
    assert scope(tmp_path, current, NEW) == ["candidate-000"]
    coordinator.supersede_campaign(current, successor(plan(), campaign_id="campaign-1-new"),
                                   implementation_commit=NEW, queue_root=tmp_path)
    assert coordinator.TIMED_REPLAN_FIELD not in ledger(tmp_path)["campaigns"][current]


@pytest.mark.parametrize("recorded", ["candidate-000", [1], None, {"candidate-000": 1}])
def test_an_unreadable_timed_replan_count_reads_as_spent(tmp_path, recorded):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-0"), queue_root=tmp_path)
    run(tmp_path, refused())
    queue = coordinator._root(tmp_path)
    value = ledger(tmp_path)
    value["campaigns"]["campaign-0"][coordinator.TIMED_REPLAN_FIELD] = recorded
    coordinator._save(queue, value)
    age(tmp_path, "campaign-1", 30 * DAY)
    assert coordinator._timed_replans(ledger(tmp_path), "site-job-1", "candidate-000") == 6
    assert scope(tmp_path, "campaign-1") is None


def test_a_ledger_without_the_count_loads_as_no_timed_replans_yet(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    run(tmp_path, refused())
    assert coordinator.TIMED_REPLAN_FIELD not in ledger(tmp_path)["campaigns"]["campaign-1"]
    assert coordinator._timed_replans(ledger(tmp_path), "site-job-1", "candidate-000") == 0


def test_a_timed_replan_covers_only_unspent_rows_and_every_money_check_still_decides(tmp_path, monkeypatch):
    """Invariant 2: a spent candidate is never in the scope, supersede re-checks under the lock, and the
    successor reserves only under the site limit with the old campaign's money still counted."""
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    run(tmp_path, Backend({"status": "blocked", "blockers": ["vast_instance_hard_ttl_expired"],
                           "provider_allocation_performed": True, "provider_zero_verified": True}), launches=1)
    run(tmp_path, refused(), launches=1)
    assert rows(tmp_path) == [("blocked", ["vast_instance_hard_ttl_expired"]),
                              ("blocked", ["beta_screening_allocator_refused"])]
    # Not yet due: supersede refuses under the lock.
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}, commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-001"]
    # The spent candidate cannot be smuggled in.
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2), commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    assert ledger(tmp_path)["campaigns"]["campaign-1"][coordinator.TIMED_REPLAN_FIELD] == ["candidate-001"]
    retried = publication(tmp_path, "campaign-1b")
    assert retried["budget"]["spent_upper_bound_usd"] == "4"
    # The site limit still decides the reservation: with $4 committed and a $4 limit it never reserves.
    monkeypatch.setattr(coordinator, "SITE_LIMIT", Decimal(4))
    running = Backend({"status": "running"})
    run(tmp_path, running)
    assert running.launches == 0, rows(tmp_path, "campaign-1b")
    assert rows(tmp_path, "campaign-1b") == [
        ("deferred", [coordinator.BUDGET_EXHAUSTED_BLOCKER])]


@pytest.mark.parametrize("codes", [
    ["paid_resource_admission_not_admitted"],
    ["spend_admission:spend_admission_lock_stale"],
    ["spend_admission_lock_required_provider_billing_missing:vast"],
    ["account_monthly_spend_limit_reached"],
    ["beta_screening_job_ceiling_exceeded"],
    ["website_job_ceiling_unavailable"],
    ["beta_screening_cost_quote_invalid_or_expired"],
    ["beta_screening_current_source_revoked"],
    ["beta_screening_prelaunch_refused:beta_screening_current_source_revoked_before_spawn"],
    ["beta_screening_authority_expired"],
    ["beta_screening_current_source_stopped"],
    ["beta_screening_execution_control_refused"],
    [coordinator.COMMIT_CHANGED_BLOCKER],
    ["g1_beta_implementation_commit_changed"],
    ["beta_screening_prelaunch_refused:beta_screening_implementation_commit_changed"],
    ["beta_screening_allocator_refused", "beta_screening_cost_quote_invalid_or_expired"],
])
def test_privacy_rights_money_answers_and_the_commit_path_are_never_retried_on_a_timer(tmp_path, codes):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    # Recorded as the refusal leaves it once every wait of its own has run out: blocked, unspent.
    value = ledger(tmp_path)
    value["campaigns"]["campaign-1"]["candidates"]["candidate-000"].update(status="blocked", blockers=codes)
    coordinator._save(coordinator._root(tmp_path), value)
    age(tmp_path, "campaign-1", 30 * DAY)
    assert not coordinator._timed_replan_due(ledger(tmp_path), ledger(tmp_path)["campaigns"]["campaign-1"],
                                             plan())
    if codes != [coordinator.COMMIT_CHANGED_BLOCKER]:  # the commit path owns that one, at any commit
        assert scope(tmp_path, "campaign-1") is None


@pytest.mark.parametrize("code", [
    coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER, coordinator.TRANSPORT_RETRY_CAP_BLOCKER,
    "beta_screening_capacity_preflight_failed", "beta_screening_preparation_failed",
    "beta_screening_static_preflight_blocked:bundle_missing", "beta_screening_not_allocated",
    "beta_screening_prelaunch_refused:oserror",
])
def test_failures_to_try_are_retried_on_a_timer(code):
    assert coordinator._timed_retry_may_touch(code)


def test_a_capacity_preflight_failure_is_planned_again_after_its_wait(tmp_path):
    class PreflightFails(Backend):
        def preflight_candidate_capacity(self, **_):
            raise OSError("storage unavailable")

    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    run(tmp_path, PreflightFails())
    assert rows(tmp_path) == [("blocked", ["beta_screening_capacity_preflight_failed"])]
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-000"]


def test_no_timed_replan_while_a_row_is_active_a_reservation_is_held_or_the_campaign_was_superseded(tmp_path):
    """Invariant 3: the existing gates on re-planning stay."""
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
    # An active row alone (its reservation aside) also keeps the campaign as it is.
    coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-2", site_id="site-job-2", count=2),
                                           queue_root=tmp_path)
    value = ledger(tmp_path)
    states = value["campaigns"]["campaign-2"]["candidates"]
    states["candidate-000"].update(status="blocked", blockers=["beta_screening_allocator_refused"])
    states["candidate-001"].update(status="running")
    coordinator._save(coordinator._root(tmp_path), value)
    age(tmp_path, "campaign-2", 30 * DAY)
    assert scope(tmp_path, "campaign-2") is None


def test_an_unreadable_refusal_time_is_not_due(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    run(tmp_path, refused())
    value = ledger(tmp_path)
    for stamp in ("not a time", "2000-01-01T00:00:00"):
        value["campaigns"]["campaign-1"]["updated_at"] = stamp
        coordinator._save(coordinator._root(tmp_path), value)
        assert scope(tmp_path, "campaign-1") is None


def test_the_old_campaign_publishes_what_a_commit_replan_publishes(tmp_path):
    """Invariant 7: the same row transitions as a commit re-plan, for the WebApp's transition rules."""
    published = {}
    for commit, root in ((NEW, tmp_path / "commit"), (OLD, tmp_path / "timed")):
        coordinator.enqueue_screening_campaign(plan(count=3, cost="2.00"), queue_root=root)
        run(root, refused(), launches=1)  # candidate-000 refused; the others still pending
        age(root, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
        coordinator.supersede_campaign("campaign-1", successor(plan(count=3, cost="2.00"), commit=commit),
                                       implementation_commit=commit, queue_root=root)
        published[commit] = (rows(root), rows(root, "campaign-1b"))
    assert published[OLD] == published[NEW]
    assert published[OLD][0] == [("blocked", ["beta_screening_allocator_refused"]),
                                 ("blocked", [coordinator.REPLANNED_BLOCKER]),
                                 ("blocked", [coordinator.REPLANNED_BLOCKER])]
    assert published[OLD][1] == [("pending", [])] * 3


def test_a_harness_hold_is_never_released_by_the_timer(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=3, cost="2.00"), queue_root=tmp_path)
    run(tmp_path, refused(), launches=1)
    run(tmp_path, Backend(HARNESS_FAILED), launches=1)
    run(tmp_path, Backend({"status": "running"}), launches=1)
    assert rows(tmp_path) == [("blocked", ["beta_screening_allocator_refused"]),
                              ("blocked", ["beta_screening_native_result_missing_or_invalid"]),
                              ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])]
    age(tmp_path, "campaign-1", 30 * DAY)
    assert scope(tmp_path, "campaign-1") is None
    # A deploy releases it, as before.
    assert scope(tmp_path, "campaign-1", NEW) == ["candidate-000", "candidate-001", "candidate-002"]


def test_a_harness_failure_beside_an_unspent_row_on_its_backend_waits_for_a_deploy(tmp_path):
    """The timed successor would not carry the failure, so a sibling on its backend could pay for it."""
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    run(tmp_path, refused(), launches=1)
    run(tmp_path, Backend(HARNESS_FAILED), launches=1)
    age(tmp_path, "campaign-1", 30 * DAY)
    assert scope(tmp_path, "campaign-1") is None
    assert scope(tmp_path, "campaign-1", NEW) == ["candidate-000", "candidate-001"]


def test_a_harness_failure_on_another_backend_does_not_stop_the_timer(tmp_path):
    """A refused G1 row is retried on its timer beside a RoboLab harness failure; the failure leaves the
    chain and is re-admitted at the next deploy, as a machine-fault re-plan already does."""
    value = on_g1(plan(count=2), "candidate-000")
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, refused(), launches=1)
    run(tmp_path, Backend(HARNESS_FAILED), launches=1)
    assert rows(tmp_path) == [("blocked", ["beta_screening_allocator_refused"]),
                              ("blocked", ["beta_screening_native_result_missing_or_invalid"])]
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-000"]
    following = successor(on_g1(plan(count=2), "candidate-000"), keep={"candidate-000"}, commit=OLD)
    coordinator.supersede_campaign("campaign-1", following, implementation_commit=OLD, queue_root=tmp_path)
    assert scope(tmp_path, "campaign-1b", NEW) == ["candidate-000", "candidate-001"]


def test_reserve_holds_only_siblings_on_the_failed_backend(tmp_path):
    """Item 2: a RoboLab harness failure does not hold a G1 candidate; it reserves under every money check."""
    value = on_g1(plan(count=3, cost="2.00"))
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Backend(HARNESS_FAILED), launches=1)
    running = Backend({"status": "running"})
    run(tmp_path, running, launches=3)
    # The G1 candidate reserves and launches; the RoboLab sibling waits its turn (one at a time).
    assert rows(tmp_path) == [("blocked", ["beta_screening_native_result_missing_or_invalid"]),
                              ("running", []), ("pending", [])]
    assert running.launches == 1
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "2" and budget["reserved_usd"] == "2"
    # Once the G1 run closes, the RoboLab sibling is held: its backend's harness failed at this commit.
    run(tmp_path, Backend({"status": "blocked", "blockers": ["vast_instance_hard_ttl_expired"],
                           "provider_allocation_performed": True, "provider_zero_verified": True}), launches=3)
    assert rows(tmp_path)[1:] == [("blocked", ["vast_instance_hard_ttl_expired"]),
                                  ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])]
    assert coordinator._harness_failures(ledger(tmp_path), "site-job-1", "candidate-000") == 1
    assert coordinator._harness_failures(ledger(tmp_path), "site-job-1", "candidate-001") == 0


def test_reserve_on_another_backend_still_passes_the_site_limit(tmp_path, monkeypatch):
    value = on_g1(plan(count=2, cost="2.00"))
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Backend(HARNESS_FAILED), launches=1)
    monkeypatch.setattr(coordinator, "SITE_LIMIT", Decimal(3))
    run(tmp_path, Backend({"status": "running"}), launches=1)
    assert rows(tmp_path)[1] == ("deferred", [coordinator.BUDGET_EXHAUSTED_BLOCKER])


def test_an_unknown_backend_holds_as_before(tmp_path):
    value = plan(count=2)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Backend(HARNESS_FAILED), launches=1)
    root = coordinator._root(tmp_path)
    unknown = {**value, "candidates": [value["candidates"][0],
                                       {**value["candidates"][1], "execution_spec": {
                                           key: item for key, item in value["candidates"][1]["execution_spec"].items()
                                           if key != "backend"}}]}
    assert coordinator._reserve(root, unknown, unknown["candidates"][1]) is None
    assert rows(tmp_path)[1] == ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])
    # And the failed sibling's backend unknown, the candidate's known: held too.
    coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-2", site_id="site-job-2", count=2),
                                           queue_root=tmp_path)
    second = plan(campaign_id="campaign-2", site_id="site-job-2", count=2)
    run(tmp_path, Backend(HARNESS_FAILED), launches=1)
    failed_unknown = {**second, "candidates": [{**second["candidates"][0], "execution_spec": {}},
                                               on_g1(second)["candidates"][1]]}
    assert coordinator._reserve(root, failed_unknown, failed_unknown["candidates"][1]) is None
    assert rows(tmp_path, "campaign-2")[1] == ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])


def test_the_source_timer_replans_a_refusal_at_the_same_commit_with_no_operator_input(tmp_path, monkeypatch):
    """Invariant 6: the pass that re-plans after a deploy re-plans on the schedule; each same-commit
    successor gets its own id (it names the campaign it supersedes) and is enqueued once."""
    import json

    from blueprint_pipeline import beta_policy_screening_worker as worker
    from tests.test_beta_policy_screening_worker import prepared_source_queue

    world = prepared_source_queue(tmp_path, monkeypatch)
    root = world["screening_queue"]
    kwargs = {"queue_root": root, "packet_queue_root": world["packet_queue"]}
    [adopted] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    chain = [adopted["campaign_id"]]
    for delay in REFUSAL_RETRY_DELAYS_SECONDS[:2]:
        coordinator.process_screening_queue(queue_root=root, executor=refused(), require_publication_ack=False)
        [replay] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
        assert replay["status"] == "already_enqueued" and replay["campaign_id"] == chain[-1]
        age(root, chain[-1], delay)
        [replanned] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
        assert replanned["status"] == "enqueued" and replanned["superseded_campaign_id"] == chain[-1]
        assert replanned["campaign_id"] not in chain
        chain.append(replanned["campaign_id"])
        [replay] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
        assert replay["status"] == "already_enqueued" and replay["campaign_id"] == chain[-1]
    value = ledger(root)
    first = value["campaigns"][chain[0]]
    [refused_id] = first[coordinator.TIMED_REPLAN_FIELD]
    assert first["candidates"][refused_id] == {**first["candidates"][refused_id], "status": "blocked",
                                               "blockers": ["beta_screening_allocator_refused"]}
    assert {row["blockers"][0] for key, row in first["candidates"].items() if key != refused_id
            and row["status"] == "blocked"} <= {coordinator.REPLANNED_BLOCKER}
    # Same commit throughout, and nothing spent.
    for campaign_id in chain:
        plan_value = json.loads((root / "plans" / f"{campaign_id}.json").read_text())
        assert {row["execution_spec"]["implementation_commit"] for row in plan_value["candidates"]
                if row["status"] in {"selected", "waitlisted"}} == {"a" * 40}
    assert value["campaigns"][chain[-1]]["site_budget"]["committed_upper_bound_usd"] == "0"


def set_rows(root, campaign_id, rows_by_id):
    """Record each named row as ``(status, blockers)``, never spent; a candidate the campaign lacks is skipped."""
    value = ledger(root)
    states = value["campaigns"][campaign_id]["candidates"]
    for candidate_id, (status, blockers) in rows_by_id.items():
        if candidate_id in states:
            states[candidate_id] = {"status": status, "blockers": list(blockers)}
    coordinator._save(coordinator._root(root), value)


def spend_timed_replans(root, candidate_id, count, *, size=3):
    """Six earlier campaigns of the site each recorded one timed re-plan of ``candidate_id``."""
    for index in range(count):
        campaign_id = f"earlier-{candidate_id}-{index}"
        coordinator.enqueue_screening_campaign(plan(campaign_id=campaign_id, count=size), queue_root=root)
        set_rows(root, campaign_id, {f"candidate-{n:03}": ("unavailable", ["candidate_disabled"])
                                     for n in range(size)})
        value = ledger(root)
        value["campaigns"][campaign_id][coordinator.TIMED_REPLAN_FIELD] = [candidate_id]
        coordinator._save(coordinator._root(root), value)


REFUSED = ("blocked", ["beta_screening_allocator_refused"])


def test_rule_3_a_row_with_no_timed_replan_left_stays_behind_and_rejoins_at_the_next_commit(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=3, cost="2.00"), queue_root=tmp_path)
    spend_timed_replans(tmp_path, "candidate-000", 6)
    set_rows(tmp_path, "campaign-1", {"candidate-000": REFUSED, "candidate-001": REFUSED})
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    # The due sibling and the pending row go; the exhausted row does not.
    assert scope(tmp_path, "campaign-1") == ["candidate-001", "candidate-002"]
    coordinator.supersede_campaign("campaign-1", successor(plan(count=3, cost="2.00"),
                                                           keep={"candidate-001", "candidate-002"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    assert ledger(tmp_path)["campaigns"]["campaign-1"][coordinator.TIMED_REPLAN_FIELD] == ["candidate-001"]
    assert rows(tmp_path)[0] == REFUSED
    # However long the successor waits, the clock never carries it.
    set_rows(tmp_path, "campaign-1b", {"candidate-001": REFUSED, "candidate-002": REFUSED})
    age(tmp_path, "campaign-1b", 30 * DAY)
    assert scope(tmp_path, "campaign-1b") == ["candidate-001", "candidate-002"]
    # The next commit plans it again, once, as it would have had it stayed in the plan.
    assert scope(tmp_path, "campaign-1b", NEW) == ["candidate-001", "candidate-002", "candidate-000"]
    coordinator.supersede_campaign("campaign-1b", successor(plan(count=3, cost="2.00"), campaign_id="campaign-1c"),
                                   implementation_commit=NEW, queue_root=tmp_path)
    assert [row["status"] for row in publication(tmp_path, "campaign-1c")["candidates"]] == ["pending"] * 3
    assert scope(tmp_path, "campaign-1c", NEW) is None


@pytest.mark.parametrize("left", [["paid_resource_admission_not_admitted"], ["beta_screening_current_source_stopped"]])
def test_rules_3_and_5_a_money_or_answer_refusal_stays_behind_and_supersede_refuses_to_carry_it(tmp_path, left):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    set_rows(tmp_path, "campaign-1", {"candidate-000": ("blocked", left), "candidate-001": REFUSED})
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    assert scope(tmp_path, "campaign-1") == ["candidate-001"]
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2), commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    assert ledger(tmp_path)["campaigns"]["campaign-1"][coordinator.TIMED_REPLAN_FIELD] == ["candidate-001"]
    assert rows(tmp_path)[0] == ("blocked", left)
    # At the next commit it is planned again, as before the timer existed.
    assert scope(tmp_path, "campaign-1b", NEW) == ["candidate-001", "candidate-000"]


def test_rules_1_and_2_an_eligible_row_rides_with_a_due_sibling_and_every_ride_is_counted(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    spend_timed_replans(tmp_path, "candidate-001", 1, size=2)
    current = "campaign-1"
    for index in range(5):
        set_rows(tmp_path, current, {"candidate-000": REFUSED, "candidate-001": REFUSED})
        counts = [coordinator._timed_replans(ledger(tmp_path), "site-job-1", key)
                  for key in ("candidate-000", "candidate-001")]
        assert counts == [index, index + 1]
        # Only candidate-000's (shorter) wait has passed; candidate-001 rides with it, counted.
        age(tmp_path, current, REFUSAL_RETRY_DELAYS_SECONDS[index])
        assert coordinator._timed_replan_due(ledger(tmp_path), ledger(tmp_path)["campaigns"][current],
                                             plan(count=2)) == ["candidate-000"]
        assert scope(tmp_path, current) == ["candidate-000", "candidate-001"]
        following = f"campaign-1-{index}"
        coordinator.supersede_campaign(current, successor(plan(count=2), campaign_id=following, commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
        assert ledger(tmp_path)["campaigns"][current][coordinator.TIMED_REPLAN_FIELD] == [
            "candidate-000", "candidate-001"]
        current = following
    # candidate-001 has had six counted re-plans: it never rides again; candidate-000 goes alone.
    set_rows(tmp_path, current, {"candidate-000": REFUSED, "candidate-001": REFUSED})
    assert coordinator._timed_replans(ledger(tmp_path), "site-job-1", "candidate-001") == 6
    age(tmp_path, current, 30 * DAY)
    assert scope(tmp_path, current) == ["candidate-000"]


def test_rule_2_the_six_retry_bound_is_exact_with_two_candidates_however_their_waits_interleave(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    patterns = [(REFUSED, REFUSED), (REFUSED, ("pending", [])), (("pending", []), REFUSED),
                (REFUSED, ("waitlisted", [coordinator.WAITLIST_BLOCKER]))]
    carried = {"candidate-000": 0, "candidate-001": 0}
    current, made = "campaign-1", 0
    for round_index in range(40):
        first, second = patterns[round_index % len(patterns)]
        set_rows(tmp_path, current, {"candidate-000": first, "candidate-001": second})
        # Waits interleave: some rounds wait only the shortest step, some a month.
        age(tmp_path, current, REFUSAL_RETRY_DELAYS_SECONDS[0] if round_index % 3 else 30 * DAY)
        found = scope(tmp_path, current)
        if found is None:
            continue
        states = ledger(tmp_path)["campaigns"][current]["candidates"]
        for key in found:
            carried[key] += states[key]["status"] == "blocked"
        following = f"campaign-1-{made}"
        coordinator.supersede_campaign(current, successor(plan(count=2), campaign_id=following, keep=set(found),
                                                          commit=OLD), implementation_commit=OLD, queue_root=tmp_path)
        current, made = following, made + 1
    counts = {key: coordinator._timed_replans(ledger(tmp_path), "site-job-1", key) for key in carried}
    assert counts == carried == {"candidate-000": 6, "candidate-001": 6}


def test_rule_4_a_row_left_behind_rejoins_on_a_new_scene_half_too(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_scene_half as scene_half

    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    set_rows(tmp_path, "campaign-1", {"candidate-000": ("blocked", ["paid_resource_admission_not_admitted"]),
                                      "candidate-001": REFUSED})
    age(tmp_path, "campaign-1", REFUSAL_RETRY_DELAYS_SECONDS[0])
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    monkeypatch.setattr(scene_half, "planned_half", lambda _plan: "sha256:" + "1" * 64)
    same = coordinator.replan_scope("campaign-1b", implementation_commit=OLD, scene_half="sha256:" + "1" * 64,
                                    queue_root=tmp_path)
    assert same is None
    assert coordinator.replan_scope("campaign-1b", implementation_commit=OLD, scene_half="sha256:" + "2" * 64,
                                    queue_root=tmp_path) == ["candidate-001", "candidate-000"]
