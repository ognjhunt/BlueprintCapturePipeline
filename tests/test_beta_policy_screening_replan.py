"""A deploy between planning and preparation re-plans unprepared candidates, never spent ones."""
from __future__ import annotations

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


def test_a_candidate_our_harness_failed_is_planned_again_after_a_deploy_at_most_twice(tmp_path):
    """2026-11-04: three learned policies ran on a rented GPU and our simulation crashed before any
    episode was scored. Nothing about the policies was measured, so after the fix ships each is
    planned again after the deploy, inside the site's allowance, at most twice; the first failure
    in a campaign holds its siblings unspent."""
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
