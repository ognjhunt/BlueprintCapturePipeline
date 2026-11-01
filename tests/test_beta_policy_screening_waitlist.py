"""Money a funded candidate releases before spend funds the next waitlisted one."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import Backend, native_result, plan, publication


def waitlisted_plan(*, campaign_id="campaign-1", site_id="site-job-1", cost="4.00", count=4, waitlisted=1,
                    budget="15.00", expired=()):
    """``count`` frozen rows, the last ``waitlisted`` of them unfunded by the plan."""
    # Built at a nominal cost (all selected rows must fit), then repriced.
    value = plan(campaign_id=campaign_id, site_id=site_id, cost="0.01", count=count)
    for index, row in enumerate(value["candidates"]):
        quote = row["cost_quote"]
        quote["maximum_cost_usd"] = row["max_cost_usd"] = cost
        if index in expired:
            quote["valid_until"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
        if index >= count - waitlisted:
            row.update(status="waitlisted", blockers=["site_screening_budget_insufficient"])
    value["budget_usd"] = budget
    value["maximum_planned_cost_usd"] = f"{Decimal(cost) * (count - waitlisted):.2f}"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


class Completes(Backend):
    """Every launch allocates, closes and returns a valid native result."""

    def __init__(self, value, refuse=()):
        super().__init__()
        self.value, self.refuse, self.prepared_ids = value, set(refuse), []

    def prepare_candidate(self, *, candidate, **kwargs):
        self.prepared_ids.append(candidate["candidate_id"])
        if candidate["candidate_id"] in self.refuse:
            raise ValueError("beta_screening_static_preflight_blocked:no_offer")
        return super().prepare_candidate(candidate=candidate, **kwargs)

    def launch_candidate(self, *, candidate, **kwargs):
        self.launches += 1
        return {"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
                "result": native_result(self.value, candidate)}


def statuses(root, campaign_id="campaign-1"):
    return [row["status"] for row in publication(root, campaign_id)["candidates"]]


def run(root, backend, **options):
    return coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend,
                                               max_launches=10, **options)


def test_released_money_funds_the_next_waitlisted_candidate(tmp_path):
    value = waitlisted_plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    assert statuses(tmp_path) == ["pending"] * 3 + ["waitlisted"]
    assert publication(tmp_path)["status"] == "queued"
    backend = Completes(value, refuse={"candidate-000"})
    outcome = run(tmp_path, backend)
    # The first funded candidate is refused before spend; its $4 funds the waitlisted one.
    assert statuses(tmp_path) == ["blocked", "completed", "completed", "pending"]
    assert outcome["waitlist_changes"] == 1
    run(tmp_path, backend)
    assert statuses(tmp_path) == ["blocked", "completed", "completed", "completed"]
    assert backend.launches == 3
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "12" and budget["reserved_usd"] == "0"
    assert publication(tmp_path)["status"] == "completed"


def test_waitlist_closes_as_budget_deferral_only_once_nothing_can_release_money(tmp_path):
    value = waitlisted_plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Completes(value))
    row = publication(tmp_path)["candidates"][-1]
    # $12 is spent and $3 left: the waitlisted $4 can never be funded.
    assert row["status"] == "deferred" and row["blockers"] == ["beta_screening_site_budget_exhausted"]
    assert publication(tmp_path)["status"] == "completed"
    assert publication(tmp_path)["budget"]["spent_upper_bound_usd"] == "12"


def test_waitlist_never_takes_money_planned_for_a_pending_candidate(tmp_path):
    value = waitlisted_plan(cost="5.00")
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    # Nothing is reserved yet, but $15 is planned for the three pending rows.
    assert coordinator._refill_waitlist(tmp_path) == 0
    assert statuses(tmp_path) == ["pending"] * 3 + ["waitlisted"]


def test_waitlist_waits_while_another_campaign_holds_a_reservation(tmp_path):
    first = plan(cost="8.00")
    second = waitlisted_plan(campaign_id="campaign-2", cost="8.00", count=1)
    for value in (first, second):
        coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Backend(RuntimeError("provider response lost")))
    assert publication(tmp_path)["status"] == "reconciliation_required"
    assert statuses(tmp_path, "campaign-2") == ["waitlisted"]
    assert publication(tmp_path, "campaign-2")["status"] == "queued"


def test_expired_quote_closes_a_waitlisted_row_without_reserving(tmp_path):
    value = waitlisted_plan(expired=(3,))
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    assert coordinator._refill_waitlist(tmp_path) == 1
    row = publication(tmp_path)["candidates"][-1]
    assert row["status"] == "deferred" and row["blockers"] == ["beta_screening_cost_quote_expired"]


def test_refill_respects_the_plans_own_budget_below_the_site_allowance(tmp_path):
    value = waitlisted_plan(count=2, budget="4.00")
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = Completes(value)
    run(tmp_path, backend)
    # The site has $11 left, but this plan's $4 budget is spent.
    assert statuses(tmp_path) == ["completed", "deferred"]
    assert backend.launches == 1


def test_funded_row_needs_the_websites_acknowledgment_before_it_is_reserved(tmp_path):
    value = waitlisted_plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = Completes(value, refuse={"candidate-000"})

    def acknowledge():
        current = publication(tmp_path)
        assert coordinator.acknowledge_publication(value["campaign_id"], current["publication_digest"],
                                                   queue_root=tmp_path)

    acknowledge()
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, max_launches=10)
    # Only the first row passed the acknowledged revision; it was refused before spend.
    assert backend.prepared_ids == ["candidate-000"]
    assert statuses(tmp_path) == ["blocked", "pending", "pending", "pending"]
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, max_launches=10)
    assert backend.prepared_ids == ["candidate-000"]
    acknowledge()
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, max_launches=10)
    assert backend.prepared_ids == ["candidate-000", "candidate-001"]
    assert publication(tmp_path)["budget"]["spent_upper_bound_usd"] == "4"


def test_plans_from_before_the_waitlist_keep_their_deferred_rows(tmp_path):
    value = plan(count=2)
    row = value["candidates"][1]
    for key in ("admission_digest", "execution_spec", "cost_quote", "max_cost_usd", "episode_count", "scenario_cells"):
        del row[key]
    row.update(status="deferred", blockers=["site_screening_budget_insufficient"])
    value["maximum_planned_cost_usd"] = "4.00"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    value = validate_screening_plan(value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    assert statuses(tmp_path) == ["pending", "deferred"]
    assert coordinator._refill_waitlist(tmp_path) == 0
