"""A deploy between planning and preparation re-plans unprepared candidates, never spent ones."""
from __future__ import annotations

from decimal import Decimal

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication

OLD, NEW = "a" * 40, "b" * 40


class RefusesChangedCommit(Backend):
    def prepare_candidate(self, **kwargs):
        self.preparations += 1
        raise ValueError("beta_screening_implementation_commit_changed")


def successor(value, *, campaign_id="campaign-1b", keep=None, commit=NEW):
    """The same source re-planned at ``commit`` for the candidates in ``keep``."""
    rows = [row for row in value["candidates"] if keep is None or row["candidate_id"] in keep]
    for row in rows:
        row["execution_spec"]["implementation_commit"] = commit
        row["admission_digest"] = canonical_digest(row["execution_spec"])
    total = sum(float(row["max_cost_usd"]) for row in rows if row["status"] == "selected")
    value.update(campaign_id=campaign_id, candidates=rows, maximum_planned_cost_usd=f"{total:.2f}")
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


def test_queued_candidates_at_an_old_commit_are_replanned(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    scope = coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path)
    assert scope == ["candidate-000", "candidate-001"]
    replanned = successor(plan(count=2))
    outcome = coordinator.supersede_campaign("campaign-1", replanned, implementation_commit=NEW, queue_root=tmp_path)
    assert outcome["status"] == "enqueued" and outcome["superseded_campaign_id"] == "campaign-1"
    old = publication(tmp_path)
    assert [row["status"] for row in old["candidates"]] == ["blocked", "blocked"]
    assert old["candidates"][0]["blockers"] == ["beta_screening_replanned_for_new_implementation"]
    assert [row["status"] for row in publication(tmp_path, "campaign-1b")["candidates"]] == ["pending", "pending"]
    assert coordinator.latest_campaign("campaign-1", queue_root=tmp_path) == "campaign-1b"
    # Idempotent, and the superseded campaign is never re-planned again.
    again = coordinator.supersede_campaign("campaign-1", replanned, implementation_commit=NEW, queue_root=tmp_path)
    assert again["status"] == "already_enqueued"
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1b", implementation_commit=NEW, queue_root=tmp_path) is None


def test_candidates_refused_for_the_changed_commit_trigger_a_replan(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=RefusesChangedCommit(), max_launches=2)
    old = publication(tmp_path)
    assert old["status"] == "blocked" and old["budget"]["committed_upper_bound_usd"] == "0"
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]


def test_spent_candidates_are_never_replanned_and_cannot_be_smuggled_in(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    allocated = Backend({"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=allocated)
    assert publication(tmp_path)["budget"]["spent_upper_bound_usd"] == "4"
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == ["candidate-001"]
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2)), implementation_commit=NEW,
                                       queue_root=tmp_path)
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}),
                                   implementation_commit=NEW, queue_root=tmp_path)
    # The new campaign sees the money the old one committed.
    assert publication(tmp_path, "campaign-1b")["budget"]["spent_upper_bound_usd"] == "4"
    assert publication(tmp_path)["status"] == "blocked"


def test_no_replan_while_a_reservation_is_held_or_nothing_is_stale(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Backend(RuntimeError("provider response lost")))
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) is None
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}),
                                       implementation_commit=NEW, queue_root=tmp_path)
    # A finished campaign has nothing unprepared, so a deploy does not re-run it.
    done = plan(campaign_id="campaign-2", site_id="site-job-2")
    coordinator.enqueue_screening_campaign(done, queue_root=tmp_path)
    finished = Backend({"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=finished)
    assert publication(tmp_path, "campaign-2")["status"] == "blocked"
    assert coordinator.replan_scope("campaign-2", implementation_commit=NEW, queue_root=tmp_path) is None


def test_candidates_refused_before_any_allocation_are_replanned_once_after_a_deploy(tmp_path):
    """2026-11-04: every learned-policy launch was refused before allocation (no usable temporary
    directory in the detached allocator). The fix ships in a deploy; the refused rows must not stay
    blocked for good, and a row that never spent is planned again once at the new commit."""
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    refused = Backend({"status": "blocked", "blockers": ["paid_resource_admission_not_admitted"],
                       "provider_allocation_performed": False, "no_allocation_proven": True})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=refused,
                                        max_launches=2)
    old = publication(tmp_path)
    assert old["status"] == "blocked" and old["budget"]["committed_upper_bound_usd"] == "0"
    # Same commit: nothing a re-plan would change.
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2)), implementation_commit=NEW,
                                   queue_root=tmp_path)
    assert [row["status"] for row in publication(tmp_path, "campaign-1b")["candidates"]] == ["pending", "pending"]
    # Once per commit: the successor, planned at NEW, is not re-planned again at NEW.
    assert coordinator.replan_scope("campaign-1b", implementation_commit=NEW, queue_root=tmp_path) is None


HARNESS_FAILED = {"status": "blocked", "blockers": ["beta_screening_native_result_missing_or_invalid"],
                  "provider_allocation_performed": True, "provider_zero_verified": True}


def test_a_candidate_our_harness_failed_is_planned_again_after_a_deploy_within_the_limit(tmp_path, monkeypatch):
    """2026-11-04: three learned policies ran on a rented GPU and our simulation crashed before any
    episode was scored. Nothing about the policies was measured, so after the fix ships each is
    planned again after the deploy, inside the site's allowance, at most twice; the first failure
    in a campaign holds its siblings unspent."""
    monkeypatch.setattr(coordinator, "HARNESS_RETRY_LIMIT", 2)  # the bound's behavior, at a small limit
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    for _ in range(2):  # One launch per pass, as in production.
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                            executor=Backend(HARNESS_FAILED))
    old = publication(tmp_path)
    # The first failure holds the sibling unspent: the same harness would fail it the same way.
    assert [(row["status"], row["blockers"]) for row in old["candidates"]] == [
        ("blocked", ["beta_screening_native_result_missing_or_invalid"]),
        ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])]
    assert old["status"] == "blocked" and old["budget"]["spent_upper_bound_usd"] == "4"
    # No deploy, no retry: the same code would fail the same way.
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2)), implementation_commit=NEW,
                                   queue_root=tmp_path)
    retried = publication(tmp_path, "campaign-1b")
    assert [row["status"] for row in retried["candidates"]] == ["pending", "pending"]
    # The retry shares the site's allowance with the money the failed run committed.
    assert retried["budget"]["spent_upper_bound_usd"] == "4"
    for _ in range(2):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                            executor=Backend(HARNESS_FAILED))
    rows = publication(tmp_path, "campaign-1b")["candidates"]
    assert [row["blockers"] for row in rows] == [["beta_screening_native_result_missing_or_invalid"],
                                                 [coordinator.HARNESS_HOLD_BLOCKER]]
    assert publication(tmp_path, "campaign-1b")["budget"]["spent_upper_bound_usd"] == "8"
    # Same commit: no retry. Another deploy: candidate-000 gets its last retry; the held one never ran.
    assert coordinator.replan_scope("campaign-1b", implementation_commit=NEW, queue_root=tmp_path) is None
    third = "c" * 40
    assert coordinator.replan_scope("campaign-1b", implementation_commit=third, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]
    coordinator.supersede_campaign("campaign-1b", successor(plan(count=2), campaign_id="campaign-1c", commit=third),
                                   implementation_commit=third, queue_root=tmp_path)
    for _ in range(2):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                            executor=Backend(HARNESS_FAILED))
    assert publication(tmp_path, "campaign-1c")["budget"]["spent_upper_bound_usd"] == "12"
    # candidate-000 failed three times: never a fourth paid run, not even after another deploy.
    assert coordinator.replan_scope("campaign-1c", implementation_commit="d" * 40, queue_root=tmp_path) == [
        "candidate-001"]


def test_a_raised_retry_bound_admits_a_candidate_dropped_when_its_retries_ran_out(tmp_path, monkeypatch):
    """2026-11-05: pi05 and cosmos3 used every harness retry and left the site's plan chain while
    their fixes shipped; the owner then raised the bound. After the next deploy each is planned
    again, once, inside the site's allowance; a candidate with a result never is."""
    from tests.test_beta_policy_screening_waitlist import Completes

    monkeypatch.setattr(coordinator, "HARNESS_RETRY_LIMIT", 0)  # no retries yet
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Backend(HARNESS_FAILED))
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-001"]
    second = successor(plan(count=2), keep={"candidate-001"})
    coordinator.supersede_campaign("campaign-1", second, implementation_commit=NEW, queue_root=tmp_path)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Completes(second))
    assert [row["status"] for row in publication(tmp_path, "campaign-1b")["candidates"]] == ["completed"]
    third = "c" * 40
    assert coordinator.replan_scope("campaign-1b", implementation_commit=third, queue_root=tmp_path) is None

    monkeypatch.setattr(coordinator, "HARNESS_RETRY_LIMIT", 1)  # the owner raises the bound
    # Only after a deploy: the plan that dropped it froze the commit it last failed under.
    assert coordinator.replan_scope("campaign-1b", implementation_commit=NEW, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1b", implementation_commit=third, queue_root=tmp_path) == [
        "candidate-000"]
    coordinator.supersede_campaign("campaign-1b", successor(plan(count=2), campaign_id="campaign-1c",
                                                            keep={"candidate-000"}, commit=third),
                                   implementation_commit=third, queue_root=tmp_path)
    retried = publication(tmp_path, "campaign-1c")
    assert [(row["candidate_id"], row["status"]) for row in retried["candidates"]] == [("candidate-000", "pending")]
    # The failed run's money stays committed at the site, and the re-admission is once per deploy.
    assert Decimal(retried["budget"]["spent_upper_bound_usd"]) == Decimal("8")
    assert coordinator.replan_scope("campaign-1c", implementation_commit=third, queue_root=tmp_path) is None


def test_only_harness_failures_with_no_scored_episode_are_retried(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    # A provider-side stop is not our harness: it keeps its result, spent.
    stopped = {**HARNESS_FAILED, "blockers": ["vast_instance_hard_ttl_expired"]}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Backend(stopped))
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Backend(HARNESS_FAILED))
    assert [row["status"] for row in publication(tmp_path)["candidates"]] == ["blocked", "blocked"]
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-001"]
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2)), implementation_commit=NEW,
                                       queue_root=tmp_path)
    # A harness code next to any other code is not a pure harness failure.
    assert not coordinator._harness_failed({"status": "blocked", "blockers": [
        "beta_screening_native_result_missing_or_invalid", "vast_instance_hard_ttl_expired"]})
    assert not coordinator._harness_failed({"status": "blocked", "blockers": [
        "beta_screening_native_result_invalid"], "summary": {"episodes_scored": 3}})
    assert not coordinator._harness_failed({"status": "completed", "blockers": []})


@pytest.mark.parametrize("stage", ["runtime_open", "observation_gate", "policy_load"])
def test_a_named_cell_setup_failure_is_a_harness_failure_for_retry_and_sibling_hold(tmp_path, stage):
    from blueprint_pipeline.policy_canary_cell_setup_failure import SETUP_FAILURE_BLOCKERS
    code = f"policy_canary_session_setup_failed:{stage}"
    assert code in SETUP_FAILURE_BLOCKERS <= coordinator.HARNESS_FAILURE_BLOCKERS
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    for _ in range(2):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                            executor=Backend({**HARNESS_FAILED, "blockers": [code]}))
    # The named stage reaches the published row; the sibling is held unspent.
    assert [(row["status"], row["blockers"]) for row in publication(tmp_path)["candidates"]] == [
        ("blocked", [code]), ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])]
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]
    assert not coordinator._harness_failed({"status": "blocked", "blockers": [code, "vast_instance_hard_ttl_expired"]})


def test_a_campaign_retrying_a_harness_failure_runs_one_candidate_at_a_time(tmp_path):
    """2026-11-04: after a retry deploy, all three candidates launched before the first one's result
    was in, so the same harness failure would have cost each its last retry. Until a run gets
    through, the next candidate waits for the active one's result, unreserved."""
    coordinator.enqueue_screening_campaign(plan(count=3, cost="2.00"), queue_root=tmp_path)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Backend(HARNESS_FAILED))
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                        executor=Backend(HARNESS_FAILED))
    coordinator.supersede_campaign("campaign-1", successor(plan(count=3, cost="2.00")), implementation_commit=NEW,
                                   queue_root=tmp_path)
    running = Backend({"status": "running"})
    for _ in range(3):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=running)
    rows = publication(tmp_path, "campaign-1b")
    assert [row["status"] for row in rows["candidates"]] == ["running", "pending", "pending"]
    assert rows["budget"]["reserved_usd"] == "2" and running.launches == 1
    # The first one completes: the harness is proven at this commit, the rest launch as usual.
    done = Backend({"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=done)
    assert publication(tmp_path, "campaign-1b")["candidates"][0]["status"] != "running"


def test_a_campaign_with_no_harness_failure_on_record_launches_as_before(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    running = Backend({"status": "running"})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=running,
                                        max_launches=2)
    assert [row["status"] for row in publication(tmp_path)["candidates"]] == ["running", "running"]


def _all_void_summary(count, reason="simulator_failed_before_policy"):
    return {"episodes_scored": 0, "episodes_succeeded": 0, "episodes_failed": 0, "episodes_void": count,
            "success_rate": None, "session_result_digest": "sha256:" + "0" * 64,
            "episode_outcomes": [{"cell_id": f"cell-{index}", "outcome": "void", "reason": reason}
                                 for index in range(count)]}


def test_a_completed_run_whose_every_episode_was_void_is_a_harness_failure(tmp_path, monkeypatch):
    """2026-11-05: pi05 reached a loaded policy in all ten cells, the session closed cleanly, and
    every episode stopped in the simulator before the first policy query. Nothing about the policy
    was measured: it is retried after a deploy like any harness failure, and its sibling waits."""
    monkeypatch.setattr(coordinator, "candidate_result_summary",
                        lambda *_args, **_kwargs: _all_void_summary(2))
    closed = {"status": "completed", "result": {"session": "closed"},
              "provider_allocation_performed": True, "provider_zero_verified": True}
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    for _ in range(2):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                            executor=Backend(closed))
    rows = publication(tmp_path)["candidates"]
    assert [(row["status"], row["blockers"]) for row in rows] == [
        ("completed", []), ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])]
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]


def test_one_scored_episode_is_a_result_not_a_harness_failure():
    summary = _all_void_summary(3)
    scored = {**summary, "episodes_scored": 1, "episodes_void": 2, "episode_outcomes": [
        {"cell_id": "cell-0", "outcome": "failure", "reason": "task_not_completed"}, *summary["episode_outcomes"][1:]]}
    assert coordinator._harness_failed({"status": "completed", "blockers": [], "summary": summary})
    assert not coordinator._harness_failed({"status": "completed", "blockers": [], "summary": scored})
    # No outcomes at all is not evidence that every episode was void.
    assert not coordinator._harness_failed({"status": "completed", "blockers": [],
                                            "summary": {**summary, "episode_outcomes": []}})
    assert not coordinator._harness_failed({"status": "running", "blockers": [], "summary": summary})


def test_an_all_void_completion_does_not_prove_the_harness_for_its_siblings():
    void_row = {"status": "completed", "blockers": [], "summary": _all_void_summary(2)}
    state = {"site_id": "site-1", "candidates": {"a": void_row, "b": {"status": "running"},
                                                 "c": {"status": "pending"}}}
    ledger = {"campaigns": {"campaign-1": state}}
    assert coordinator._unproven_retry_in_flight(ledger, state, "c")
    proven = {**state, "candidates": {**state["candidates"], "a": {
        "status": "completed", "blockers": [], "summary": {**_all_void_summary(2), "episodes_scored": 1}}}}
    assert not coordinator._unproven_retry_in_flight({"campaigns": {"campaign-1": proven}}, proven, "c")


def test_a_budget_deferred_unspent_candidate_is_planned_again_after_a_deploy_while_money_is_left(tmp_path, monkeypatch):
    """2026-11-05: with $3.78 of $15 left every remaining row closed as budget-deferred, and no
    rule ever planned them again. After a deploy the new plan may fit a quote to what is left."""
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    root = coordinator._root(tmp_path)
    ledger = coordinator._ledger(root)
    for row in ledger["campaigns"]["campaign-1"]["candidates"].values():
        row.update(status="deferred", blockers=[coordinator.BUDGET_EXHAUSTED_BLOCKER])
    coordinator._save(root, ledger)
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]
    # No money left at the site: nothing to fit, so nothing is planned again.
    monkeypatch.setattr(coordinator, "SITE_LIMIT", coordinator.Decimal("0"))
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) is None
