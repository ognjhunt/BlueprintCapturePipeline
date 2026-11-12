"""Testing is not rationed below a website job's ceiling (owner decision 2026-11-11).

The owner's words: "yes replace them - $300 monthly limit, $100 fuse". A website job's plan
budgets everything its ceiling leaves after the build: no $15 plan size, no $30 site allowance.
The $4 cap per AI robot test and the $3 scripted reference check stay as they are.
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_site_limit as site_limit_module
from blueprint_pipeline.beta_policy_screening_plan import (
    JOB_CEILING_DEFERRED, build_job_ceiling_screening_plan, job_ceiling_record, validate_screening_plan,
)
from tests.test_beta_policy_screening_job_ceiling import READ_AT, SITE, ceiling_summary
from tests.test_beta_policy_screening_known_good_control import catalog_with_scripted, rerated
from tests.test_beta_policy_screening_plan import NOW, candidate_rows, inputs, scene


def switch_to(tmp_path, monkeypatch, mode):
    """Run with the website money switch (``website_job_limits``) at ``mode``."""
    from blueprint_pipeline import website_job_limits

    path = tmp_path / f"website_job_limits.{mode}.json"
    path.write_text(json.dumps({"schema_version": website_job_limits.SCHEMA, "mode": mode}))
    monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(path))


@pytest.fixture(autouse=True)
def _fuse(tmp_path, monkeypatch):
    """These tests are the 2026-11-11 model: the switch at ``fuse``."""
    switch_to(tmp_path, monkeypatch, "fuse")


def _plan(catalog, capabilities, quotes, *, maximum, build, campaign_id="campaign1"):
    record = job_ceiling_record(ceiling_summary(maximum - build, maximum=maximum), local_holds_usd=Decimal(0),
                                read_at=READ_AT)
    return build_job_ceiling_screening_plan(catalog, scene(), job_ceiling=record, campaign_id=campaign_id,
                                            capabilities=capabilities, cost_quotes=quotes, now=NOW)


def _twelve_robots():
    catalog, capabilities, quotes = catalog_with_scripted(12, maximum="4.00", scripted_maximum="4.00")
    return catalog, rerated(capabilities, "blueprint_scripted_test_arm", rate="0.80", ttl=14_400), quotes


def _split(value):
    """The learned rows in plan order, and the scripted reference row (wherever the plan puts it)."""
    learned = [row for row in value["candidates"] if not row.get("known_good_control")]
    [control_row] = [row for row in value["candidates"] if row.get("known_good_control")]
    return learned, control_row


def test_a_100_dollar_job_whose_build_took_45_funds_all_twelve_robots_and_the_reference_check():
    catalog, capabilities, quotes = _twelve_robots()
    value = _plan(catalog, capabilities, quotes, maximum=100, build=45)
    learned, control_row = _split(value)
    assert value["budget_usd"] == "55.00"
    assert [row["status"] for row in learned] == ["selected"] * 12
    assert all(row["max_cost_usd"] == "4.00" for row in learned)
    # 12 x $4 = $48 and the reference check at its own $3 cap fit the $55 the ceiling leaves.
    assert control_row["status"] == "selected" and control_row["max_cost_usd"] == "3.00"
    assert value["maximum_planned_cost_usd"] == "51.00"
    assert validate_screening_plan(value) == value


def test_a_25_dollar_job_funds_what_fits_and_waitlists_the_rest_for_its_ceiling():
    catalog, capabilities, quotes = _twelve_robots()
    value = _plan(catalog, capabilities, quotes, maximum=25, build=15)
    learned, _control = _split(value)
    assert value["budget_usd"] == "10.00"
    statuses = [row["status"] for row in learned]
    funded = statuses.count("selected")
    # Funded in plan order while the $10 lasts; every other runnable robot waits for the ceiling, never a site budget.
    assert funded >= 1 and statuses == ["selected"] * funded + ["waitlisted"] * (12 - funded)
    assert {tuple(row["blockers"]) for row in learned if row["status"] == "waitlisted"} == {(JOB_CEILING_DEFERRED,)}
    assert Decimal(value["maximum_planned_cost_usd"]) <= Decimal("10.00")


def test_a_job_with_nothing_left_after_its_build_defers_every_robot_for_its_ceiling():
    catalog, capabilities, quotes = _twelve_robots()
    value = _plan(catalog, capabilities, quotes, maximum=25, build=25)
    assert value["budget_usd"] == "0.00" and value["maximum_planned_cost_usd"] == "0.00"
    assert {(row["status"], tuple(row["blockers"])) for row in candidate_rows(value)} == {
        ("deferred", (JOB_CEILING_DEFERRED,))}


class _CommitsAtCap:
    """A launch that allocated and ended: its whole cap is committed, as funding counts an unsettled run."""

    def prepare_candidate(self, **_):
        return {"sealed": "local"}

    def launch_candidate(self, **_):
        return {"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True}


def test_a_site_commits_past_the_old_30_dollars_within_its_100_dollar_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "_quote_current", lambda candidate: True)
    catalog, capabilities, quotes = inputs(12, "4.00")
    value = _plan(catalog, capabilities, quotes, maximum=100, build=45)
    assert [row["status"] for row in candidate_rows(value)] == ["selected"] * 12
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    ledger = coordinator._ledger(coordinator._root(tmp_path))
    assert ledger["campaigns"]["campaign1"][site_limit_module.SITE_LIMIT_FIELD] == "100"
    for _ in range(3):
        coordinator.process_screening_queue(queue_root=tmp_path, executor=_CommitsAtCap(),
                                            require_publication_ack=False, max_launches=20)
    [publication] = coordinator.pending_publications(queue_root=tmp_path)
    # Twelve $4 runs, $48, all within the job's ceiling; before 2026-11-11 the site stopped at $30.
    assert publication["budget"] == {"currency": "USD", "limit_usd": "100", "committed_upper_bound_usd": "48",
                                     "spent_upper_bound_usd": "48", "reserved_usd": "0"}
    assert all(row["status"] == "blocked" for row in candidate_rows(publication))


def test_a_reservation_past_the_ceiling_waits_for_the_ceiling_not_a_site_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "_quote_current", lambda candidate: True)
    catalog, capabilities, quotes = inputs(3, "4.00")
    first = _plan(catalog, capabilities, quotes, maximum=10, build=0, campaign_id="campaign-a")
    second = _plan(catalog, capabilities, quotes, maximum=10, build=0, campaign_id="campaign-b")
    for value in (first, second):
        coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=_CommitsAtCap(),
                                        require_publication_ack=False, max_launches=20)
    rows = coordinator._ledger(coordinator._root(tmp_path))["campaigns"]
    # The site's limit is the job's $10 ceiling: campaign a commits $8, campaign b's first $4 cannot fit.
    blocked = [row for state in rows.values() for row in state["candidates"].values()
               if row["status"] in {"waitlisted", "deferred"}]
    assert blocked and all(row["blockers"] == [JOB_CEILING_DEFERRED] for row in blocked)
    publications = coordinator.pending_publications(queue_root=tmp_path)
    assert {item["budget"]["limit_usd"] for item in publications} == {"10"}
    assert max(Decimal(item["budget"]["committed_upper_bound_usd"]) for item in publications) <= Decimal(10)


def test_a_campaign_enqueued_before_the_change_keeps_the_old_site_allowance():
    ledger = {"campaigns": {"old": {"site_id": SITE}, "new": {"site_id": SITE, "site_limit_usd": "25"}}}
    assert site_limit_module.site_limit(ledger, SITE) == Decimal("30")
    assert site_limit_module.site_limit({"campaigns": {"new": ledger["campaigns"]["new"]}}, SITE) == Decimal("25")
    assert site_limit_module.site_limit({"campaigns": {}}, SITE) == Decimal("30")


def test_the_codes_the_site_limit_names_are_the_plans_own():
    from blueprint_pipeline.beta_policy_screening_plan import JOB_CEILING_BASIS

    assert site_limit_module.JOB_CEILING_BASIS == JOB_CEILING_BASIS
    assert site_limit_module.JOB_CEILING_DEFERRED == JOB_CEILING_DEFERRED


def test_the_launch_time_ceiling_check_admits_a_100_dollar_job_and_stops_one_whose_ceiling_fell(tmp_path, monkeypatch):
    """Every paid launch reads the job's ceiling again (``require_current_job_ceiling``), past the old $30."""
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling

    monkeypatch.setattr(coordinator, "_quote_current", lambda candidate: True)
    catalog, capabilities, quotes = inputs(12, "4.00")
    value = _plan(catalog, capabilities, quotes, maximum=100, build=45)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    for _ in range(3):
        coordinator.process_screening_queue(queue_root=tmp_path, executor=_CommitsAtCap(),
                                            require_publication_ack=False, max_launches=20)
    reads = []

    def summary(maximum):
        def read(**kwargs):
            reads.append(kwargs)
            # The website has seen the $45 build and none of this queue's $48 yet.
            return ceiling_summary(maximum - 45, maximum=maximum)
        return read
    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", summary(100))
    assert job_ceiling.require_current_job_ceiling(value, context={}, capture_root=tmp_path,
                                                   queue_root=tmp_path)["maximum_cost_usd"] == 100
    # The owner's row expired between plan and launch: $45 built + $48 committed passes a $60 ceiling.
    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", summary(60))
    with pytest.raises(ValueError, match=job_ceiling.EXCEEDED):
        job_ceiling.require_current_job_ceiling(value, context={}, capture_root=tmp_path, queue_root=tmp_path)
    assert len(reads) == 2


@pytest.mark.parametrize("remaining, budget, selected", [(15, "15.00", 7), (40, "40.00", 20)])
def test_the_plan_budgets_everything_the_ceiling_leaves(remaining, budget, selected):
    """At ``nested`` the same plan is $15 at most (``test_beta_policy_screening_job_ceiling``, unchanged)."""
    from tests.test_beta_policy_screening_job_ceiling import ceiling_plan

    value = ceiling_plan(remaining)
    learned = [row for row in candidate_rows(value)]
    assert value["budget_usd"] == budget and value["maximum_planned_cost_usd"] == f"{2 * selected:.2f}"
    assert [row["status"] for row in learned] == ["selected"] * selected + ["waitlisted"] * (40 - selected)
    assert {tuple(row["blockers"]) for row in learned if row["status"] == "waitlisted"} == {(JOB_CEILING_DEFERRED,)}


def test_at_nested_the_same_job_plans_as_before(tmp_path, monkeypatch):
    switch_to(tmp_path, monkeypatch, "nested")
    catalog, capabilities, quotes = _twelve_robots()
    value = _plan(catalog, capabilities, quotes, maximum=100, build=45)
    assert value["budget_usd"] == "15.00"
    assert {tuple(row["blockers"]) for row in candidate_rows(value) if row["status"] == "waitlisted"} == {
        ("site_screening_budget_insufficient",)}


def test_the_release_acceptance_takes_a_publication_limit_up_to_the_ceiling_only_at_fuse(tmp_path, monkeypatch):
    """``production_autonomy_acceptance`` (the release gate's AUT-21 sub-report) mirrors the publication contract."""
    from blueprint_pipeline import production_autonomy_acceptance as acceptance

    catalog, capabilities, quotes = _twelve_robots()
    value = _plan(catalog, capabilities, quotes, maximum=100, build=45)
    assert acceptance.publication_limit_bound(value) == Decimal("100")
    internal = {key: item for key, item in value.items() if key not in {"budget_basis", "website_job_ceiling"}}
    assert acceptance.publication_limit_bound(internal) == Decimal("15")
    switch_to(tmp_path, monkeypatch, "nested")
    assert acceptance.publication_limit_bound(value) == Decimal("15") == acceptance.PUBLICATION_LIMIT_USD
