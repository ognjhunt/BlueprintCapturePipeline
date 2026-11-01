"""Each new screening plan fits what its website job's one dollar ceiling leaves.

Owner decision 2026-10-30 (WebApp #1168): one ceiling per website job across
Atlas, upstream, native construction and screening. The coordinator reserves
screening without asking the WebApp, so the plan's budget is the stop.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from urllib.error import URLError

import pytest
from jsonschema import Draft202012Validator

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline import website_task_context
from blueprint_pipeline.beta_policy_screening_plan import (
    JOB_CEILING_DEFERRED, BetaPolicyScreeningPlanError, build_job_ceiling_screening_plan, build_screening_plan,
    job_ceiling_record, validate_screening_plan,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_plan import NOW, inputs, plan, reseal, scene

READ_AT = "2026-10-26T00:00:00+00:00"
READ_JOB_CEILING = job_ceiling.read_job_ceiling_summary
SITE = "site-request1"


def ceiling_summary(remaining=40, *, screening=0, maximum=40):
    """The WebApp's `job_ceiling` (website_job_ceiling.v1, WebApp #1168)."""
    spent = maximum - remaining
    return {"schema_version": "website_job_ceiling.v1", "maximum_cost_usd": maximum,
            "source": "retained_atlas_grant", "preparation_cost_usd": spent - screening,
            "native_construction_cost_usd": 0, "screening_cost_usd": screening,
            "reserved_or_settled_cost_usd": spent, "remaining_cost_usd": remaining}


def ceiling_plan(remaining, *, count=40, holds=Decimal(0), campaign_id="campaign1", **kwargs):
    catalog, capabilities, quotes = inputs(count, "2.00")
    record = job_ceiling_record(ceiling_summary(remaining), local_holds_usd=holds, read_at=READ_AT)
    return build_job_ceiling_screening_plan(catalog, scene(), job_ceiling=record, campaign_id=campaign_id,
                                            capabilities=capabilities, cost_quotes=quotes, now=NOW, **kwargs)


def _unbased(value):
    return {key: item for key, item in value.items()
            if key not in {"budget_basis", "website_job_ceiling", "plan_digest"}}


@pytest.mark.parametrize("remaining", [15, 15.009, 40])
def test_fifteen_dollars_or_more_left_leaves_the_plan_unchanged(remaining):
    bounded = ceiling_plan(remaining)
    assert bounded["budget_usd"] == "15.00" and bounded["maximum_planned_cost_usd"] == "14.00"
    assert _unbased(bounded) == _unbased(plan(40, "2.00"))
    assert bounded["budget_basis"] == "website_job_ceiling"
    record = bounded["website_job_ceiling"]
    assert record["remaining_cost_usd"] == f"{Decimal(str(remaining)).quantize(Decimal('0.01'), 'ROUND_FLOOR')}"
    assert record["read_at"] == READ_AT and record["summary"] == ceiling_summary(remaining)
    assert record["summary_digest"] == canonical_digest(ceiling_summary(remaining))


def test_nine_dollars_left_funds_fewer_and_defers_the_rest_for_the_job_ceiling(tmp_path):
    value = ceiling_plan(9)
    assert value["budget_usd"] == "9.00" and value["maximum_planned_cost_usd"] == "8.00"
    assert value["website_job_ceiling"]["available_usd"] == "9.00"
    selected = [row for row in value["candidates"] if row["status"] == "selected"]
    assert len(selected) == 4
    rest = [row for row in value["candidates"] if row["status"] != "selected"]
    assert len(rest) == 36
    # Each could still be funded within the $9 from money a selected candidate
    # releases before spend, so it waits; the ceiling is named as the reason.
    assert all(row["status"] == "waitlisted" and row["blockers"] == [JOB_CEILING_DEFERRED] for row in rest)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    [publication] = coordinator.pending_publications(queue_root=tmp_path)
    assert sum(row["status"] == "pending" for row in publication["candidates"]) == 4
    assert {tuple(row["blockers"]) for row in publication["candidates"] if row["status"] == "waitlisted"} == {
        (JOB_CEILING_DEFERRED,)}


def test_waitlist_refills_only_within_the_ceiling_sized_budget(tmp_path, monkeypatch):
    # The planner fixture prices on a fixed clock; its quotes are current for this run.
    monkeypatch.setattr(coordinator, "_quote_current", lambda candidate: True)
    value = ceiling_plan(9, count=8)
    selected = [row["candidate_id"] for row in value["candidates"] if row["status"] == "selected"]
    assert len(selected) == 4 and value["budget_usd"] == "9.00"
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)

    class FirstRefusedRestComplete:
        def prepare_candidate(self, *, candidate, **_):
            if candidate["candidate_id"] == selected[0]:
                raise ValueError("beta_screening_static_preflight_blocked:no_offer")
            return {"sealed": "local"}

        def launch_candidate(self, **_):
            return {"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True}

    for _ in range(3):
        coordinator.process_screening_queue(queue_root=tmp_path, executor=FirstRefusedRestComplete(),
                                            require_publication_ack=False, max_launches=10)
    [publication] = coordinator.pending_publications(queue_root=tmp_path)
    # The refused candidate's $2 funds one waitlisted row: $8 spent of the job's $9, never more.
    assert publication["budget"]["spent_upper_bound_usd"] == "8"
    statuses = [row["status"] for row in publication["candidates"]]
    assert statuses.count("waitlisted") == 0 and statuses.count("blocked") == 5
    assert {tuple(row["blockers"]) for row in publication["candidates"] if row["status"] == "deferred"} == {
        ("beta_screening_site_budget_exhausted",)}


def test_a_campaigns_waitlist_counts_against_the_ceiling_for_later_plans(tmp_path):
    # $3 left: one $2 row is selected, the rest wait within the $3 plan budget.
    earlier = ceiling_plan(3, count=4, campaign_id="campaign0")
    assert [row["status"] for row in earlier["candidates"]].count("selected") == 1
    coordinator.enqueue_screening_campaign(earlier, queue_root=tmp_path)
    # If its selected row is refused before spend, the refill can fund a waitlisted
    # one: the campaign can still spend its whole $3, so a later plan must not get it.
    assert job_ceiling.local_screening_holds(tmp_path, site_id=SITE, published_usd=Decimal(0)) == Decimal("3.00")
    later = ceiling_plan(3, count=4, campaign_id="campaign1",
                         holds=job_ceiling.local_screening_holds(tmp_path, site_id=SITE, published_usd=Decimal(0)))
    assert later["budget_usd"] == "0.00"
    assert all(row["status"] == "deferred" for row in later["candidates"])


def test_a_replan_does_not_count_the_rows_it_replaces_against_the_ceiling(tmp_path):
    catalog, capabilities, quotes = inputs(4, "2.00")
    earlier = build_screening_plan(catalog, scene(), campaign_id="campaign0", capabilities=capabilities,
                                   cost_quotes=quotes, now=NOW)
    coordinator.enqueue_screening_campaign(earlier, queue_root=tmp_path)
    assert job_ceiling.local_screening_holds(tmp_path, site_id=SITE, published_usd=Decimal(0)) == Decimal("8.00")
    # Re-planning campaign0 closes its pending rows in the same write; they must not shrink the new plan.
    assert job_ceiling.local_screening_holds(tmp_path, site_id=SITE, published_usd=Decimal(0),
                                             superseding="campaign0") == Decimal(0)


@pytest.mark.parametrize("remaining,budget", [(1.99, "1.99"), (0.004, "0.00"), (0, "0.00")])
def test_less_than_the_cheapest_quote_defers_every_candidate_and_still_publishes(tmp_path, remaining, budget):
    value = ceiling_plan(remaining)
    assert value["budget_usd"] == budget and value["maximum_planned_cost_usd"] == "0.00"
    assert all(row["status"] == "deferred" and row["blockers"] == [JOB_CEILING_DEFERRED]
               and set(row) == {"candidate_id", "display_name", "configuration_digest", "status", "blockers"}
               for row in value["candidates"])
    assert coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)["status"] == "enqueued"
    [publication] = coordinator.pending_publications(queue_root=tmp_path)
    assert publication["status"] == "blocked" and len(publication["candidates"]) == 40
    assert all(row["status"] == "deferred" and row["blockers"] == [JOB_CEILING_DEFERRED]
               for row in publication["candidates"])


def test_ceiling_only_relabels_budget_deferrals_and_keeps_other_reasons():
    value = ceiling_plan(9, count=2)
    rows = {row["candidate_id"]: row for row in value["candidates"]}
    assert {row["candidate_id"] for row in value["candidates"] if row["status"] == "selected"} == {
        "groot_n17_droid", "pi05_droid"}
    unfunded = [row for row in rows.values() if row["status"] != "selected"]
    assert unfunded and all(JOB_CEILING_DEFERRED not in row["blockers"] for row in unfunded)
    assert _unbased(value) == {**_unbased(plan()), "budget_usd": "9.00"}


def test_unpublished_and_unreserved_screening_on_the_site_counts_against_the_ceiling(tmp_path):
    catalog, capabilities, quotes = inputs(4, "2.00")
    earlier = build_screening_plan(catalog, scene(), campaign_id="campaign0", capabilities=capabilities,
                                   cost_quotes=quotes, now=NOW)
    coordinator.enqueue_screening_campaign(earlier, queue_root=tmp_path)
    assert job_ceiling.local_screening_holds(tmp_path, site_id=SITE, published_usd=Decimal(0)) == Decimal("8.00")
    # One candidate reserved since the last publication the WebApp accepted.
    candidate = next(row for row in earlier["candidates"] if row["status"] == "selected")
    with coordinator._lock(tmp_path):
        ledger = coordinator._ledger(tmp_path)
        ledger["campaigns"]["campaign0"]["candidates"][candidate["candidate_id"]].update(
            status="preparing", attempt=coordinator._new_attempt(earlier, candidate))
        coordinator._save(tmp_path, ledger)
    assert job_ceiling.local_screening_holds(tmp_path, site_id=SITE, published_usd=Decimal(0)) == Decimal("8.00")
    assert job_ceiling.local_screening_holds(tmp_path, site_id=SITE, published_usd=Decimal(2)) == Decimal("6.00")
    assert job_ceiling.local_screening_holds(tmp_path, site_id="site-other", published_usd=Decimal(0)) == 0
    value = ceiling_plan(15, holds=Decimal("8.00"))
    assert value["budget_usd"] == "7.00" and value["website_job_ceiling"]["local_screening_holds_usd"] == "8.00"
    assert sum(row["status"] == "selected" for row in value["candidates"]) == 3


@pytest.mark.parametrize("mutation", ["budget_above_available", "summary_edited", "basis_dropped",
                                      "both_dropped_zero_budget", "available_inflated"])
def test_resealed_plans_cannot_exceed_or_misstate_the_ceiling(mutation):
    value = ceiling_plan(9)
    if mutation == "budget_above_available":
        value["budget_usd"] = "10.00"
        expected = "job_ceiling_exceeded"
    elif mutation == "summary_edited":
        value["website_job_ceiling"]["summary"]["remaining_cost_usd"] = 40
        expected = "job_ceiling_invalid"
    elif mutation == "basis_dropped":
        del value["budget_basis"]
        expected = "job_ceiling_invalid"
    elif mutation == "both_dropped_zero_budget":
        value = ceiling_plan(0)
        del value["budget_basis"], value["website_job_ceiling"]
        expected = "site_budget_exceeded"
    else:
        value["website_job_ceiling"]["available_usd"] = "15.00"
        expected = "job_ceiling_invalid"
    with pytest.raises(BetaPolicyScreeningPlanError, match=expected):
        validate_screening_plan(reseal(value))


def test_machine_readable_schema_accepts_ceiling_plans_and_pairs_their_fields():
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / "docs/schemas/beta_policy_screening_plan.v1.schema.json").read_text())
    validator = Draft202012Validator(schema)
    for remaining in (40, 9, 0):
        validator.validate(ceiling_plan(remaining))
    value = ceiling_plan(9)
    del value["website_job_ceiling"]
    assert not validator.is_valid(value)


def _transport(calls, answer):
    def send(*, capture_id, operation, payload, source_kind="website"):
        calls.append({"capture_id": capture_id, "operation": operation, "payload": dict(payload)})
        if isinstance(answer, BaseException):
            raise answer
        return {"schema_version": "website_atlas_configuration.v1", "dispatch_authorized": False,
                "grant_creation_allowed": False, "task_context_digest": payload["task_context_digest"],
                "source_video_digest": payload["source_video_digest"], **answer}
    return send


def test_worker_holds_until_the_ceiling_is_readable_then_plans_within_it_once(tmp_path, monkeypatch):
    from tests.test_beta_policy_screening_worker import prepared_source_queue

    world = prepared_source_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", READ_JOB_CEILING)
    capture = tmp_path / "partition/scenes/site-job1/captures/walkthrough-job1"
    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    calls = []
    source_digest = "sha256:" + "5" * 64

    def tick(answer):
        monkeypatch.setattr(website_task_context, "website_webapp_request", _transport(calls, answer))
        [outcome] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
        return outcome

    # No retained Atlas configuration, the network, an old WebApp, a refusal: hold, never the default.
    held = [tick(ceiling_summary(9))]
    (capture / "pipeline/atlas").mkdir(parents=True)
    (capture / "pipeline/atlas/configuration.json").write_text(json.dumps({"source_video_digest": source_digest}))
    held += [tick(URLError("unreachable")), tick({}),
             tick(ValueError("website_control_atlas-configuration_http_409:website_atlas_source_consent_required"))]
    assert all(row["status"] == "blocked" and row["blockers"] == [job_ceiling.UNAVAILABLE] for row in held)
    assert not list((world["screening_queue"] / "plans").iterdir())
    assert len(calls) == 3 and all(call["operation"] == "atlas-configuration" for call in calls)
    assert calls[0]["payload"] == {"request_id": "job1", "scene_id": "site-job1", "create": False,
        "task_context_digest": world["context"]["context_digest"], "source_video_digest": source_digest}

    adopted = tick({"job_ceiling": ceiling_summary(9, screening=0)})
    assert adopted["status"] == "enqueued"
    [path] = (world["screening_queue"] / "plans").glob("*.json")
    value = validate_screening_plan(json.loads(path.read_text()))
    assert value["budget_usd"] == "9.00" and value["maximum_planned_cost_usd"] == "8.00"
    assert value["website_job_ceiling"]["summary"] == ceiling_summary(9)
    funded = [row for row in value["candidates"] if row["status"] == "selected"]
    ceiling_deferred = [row for row in value["candidates"] if row["blockers"] == [JOB_CEILING_DEFERRED]]
    assert len(funded) == 4 and len(ceiling_deferred) == 4

    # A built plan is never resized or re-read, whatever the ceiling says later.
    assert tick({"job_ceiling": ceiling_summary(40)})["status"] == "already_enqueued"
    assert len(calls) == 4 and json.loads(path.read_text()) == value
