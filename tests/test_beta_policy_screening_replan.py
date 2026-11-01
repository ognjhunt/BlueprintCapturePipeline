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


def successor(value, *, campaign_id="campaign-1b", keep=None):
    """The same source re-planned at ``NEW`` for the candidates in ``keep``."""
    rows = [row for row in value["candidates"] if keep is None or row["candidate_id"] in keep]
    for row in rows:
        row["execution_spec"]["implementation_commit"] = NEW
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
