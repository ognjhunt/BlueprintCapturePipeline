"""One site past its screening limit holds only itself, and figures above $30 are read without raising.

Before this, ``_ledger`` refused the whole screening ledger, every site's, once any one site had
committed more than the $30 allowance. A release with this change is a safe one to roll back to
from the one that lets a website job's testing run up to its own ceiling (owner decision
2026-11-11): its campaigns record their job's ceiling (``site_limit_usd``) and may commit past $30.
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_plan import job_ceiling_record, validate_screening_plan
from blueprint_pipeline.beta_policy_screening_site_limit import SITE_LIMIT_FIELD, plan_site_limit, site_limit
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import plan, publication
from tests.test_beta_policy_screening_waitlist import Completes, run, statuses
from tests.test_beta_policy_screening_unrationed import _CommitsAtCap, _plan, switch_to


def test_a_site_past_its_limit_holds_only_itself(tmp_path, monkeypatch):
    earlier = [plan(campaign_id=f"campaign-a{index}", site_id="site-a", cost="4.00", count=2) for index in (1, 2)]
    for value in earlier:
        coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
        run(tmp_path, Completes(value))
    assert publication(tmp_path, "campaign-a1")["budget"]["committed_upper_bound_usd"] == "16"
    # Site a is now past its limit, as a site would be after a rollback had it recorded no figure.
    monkeypatch.setattr(coordinator, "SITE_LIMIT", Decimal("15"))
    later = plan(campaign_id="campaign-a3", site_id="site-a", cost="1.00")
    other = plan(campaign_id="campaign-b", site_id="site-b", cost="4.00")
    for value in (later, other):
        coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Completes(other))
    # The ledger reads; site b runs; site a's new row waits by the typed reason, reserving nothing.
    assert statuses(tmp_path, "campaign-b") == ["completed"]
    row = coordinator._ledger(coordinator._root(tmp_path))["campaigns"]["campaign-a3"]["candidates"]["candidate-000"]
    assert row["status"] == "waitlisted" and not row.get("attempt")
    assert row["blockers"] == [coordinator.SITE_EXCEEDED_BLOCKER] == ["beta_screening_site_budget_exceeded"]
    # It stays held (never closed for budget) while the site is past its limit; a raised limit releases it.
    run(tmp_path, Completes(later))
    assert statuses(tmp_path, "campaign-a3") == ["waitlisted"]
    monkeypatch.setattr(coordinator, "SITE_LIMIT", Decimal("30"))
    coordinator._refill_waitlist(coordinator._root(tmp_path))
    assert statuses(tmp_path, "campaign-a3") == ["pending"]


def test_a_recorded_site_limit_above_30_and_its_commitments_read_without_raising(tmp_path, monkeypatch):
    value = plan(campaign_id="campaign-a", site_id="site-a", cost="4.00", count=2)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Completes(value))
    root = coordinator._root(tmp_path)
    ledger = coordinator._ledger(root)
    ledger["campaigns"]["campaign-a"][SITE_LIMIT_FIELD] = "100"
    coordinator._save(root, ledger)
    ledger = coordinator._ledger(root)
    # nested (shipped): the recorded figure is read without raising and counts as $30, as before.
    assert site_limit(ledger, "site-a", coordinator.SITE_LIMIT) == Decimal("30")
    switch_to(tmp_path, monkeypatch, "fuse")
    assert site_limit(ledger, "site-a", coordinator.SITE_LIMIT) == Decimal("100")
    assert coordinator._budget(ledger, "site-a")["limit_usd"] == "100"
    # Garbage in the field counts as the old allowance, never raises.
    assert site_limit({"campaigns": {"x": {"site_id": "s", SITE_LIMIT_FIELD: "nope"}}}, "s") == Decimal("30")


def test_switching_back_to_nested_holds_only_the_site_that_committed_past_30(tmp_path, monkeypatch):
    """fuse lets a $100 job commit $48; switched back to nested, that site is held and every other goes on."""
    monkeypatch.setattr(coordinator, "_quote_current", lambda candidate: True)
    switch_to(tmp_path, monkeypatch, "fuse")
    from tests.test_beta_policy_screening_plan import inputs

    catalog, capabilities, quotes = inputs(12, "4.00")
    big = _plan(catalog, capabilities, quotes, maximum=100, build=45)
    coordinator.enqueue_screening_campaign(big, queue_root=tmp_path / "q")
    for _ in range(3):
        coordinator.process_screening_queue(queue_root=tmp_path / "q", executor=_CommitsAtCap(),
                                            require_publication_ack=False, max_launches=20)
    root = coordinator._root(tmp_path / "q")
    assert coordinator._budget(coordinator._ledger(root), big["site_id"])["committed_upper_bound_usd"] == "48"
    switch_to(tmp_path, monkeypatch, "nested")
    ledger = coordinator._ledger(root)  # reads; before this change it raised for every site
    assert coordinator._budget(ledger, big["site_id"])["limit_usd"] == "30"
    held = plan(campaign_id="campaign-held", site_id=big["site_id"], cost="1.00")
    other = plan(campaign_id="campaign-other", site_id="site-other", cost="4.00")
    for value in (held, other):
        coordinator.enqueue_screening_campaign(value, queue_root=tmp_path / "q")
    run(tmp_path / "q", Completes(other))
    assert statuses(tmp_path / "q", "campaign-other") == ["completed"]
    row = coordinator._ledger(root)["campaigns"]["campaign-held"]["candidates"]["candidate-000"]
    assert row["status"] == "waitlisted" and row["blockers"] == [coordinator.SITE_EXCEEDED_BLOCKER]
    assert coordinator._budget(coordinator._ledger(root), big["site_id"])["committed_upper_bound_usd"] == "48"


def test_a_retained_ceiling_plan_budgeting_more_than_30_validates():
    from tests.test_beta_policy_screening_job_ceiling import READ_AT, ceiling_plan, ceiling_summary

    value = ceiling_plan(15)
    record = job_ceiling_record(ceiling_summary(55, maximum=100), local_holds_usd=Decimal(0), read_at=READ_AT)
    value = {**value, "budget_usd": "55.00", "website_job_ceiling": record, "plan_digest": ""}
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    assert validate_screening_plan(value) == value
    assert plan_site_limit(value, Decimal("30")) == Decimal("100")


def test_the_basis_the_site_limit_reads_is_the_plans_own():
    from blueprint_pipeline import beta_policy_screening_site_limit as site_limit_module
    from blueprint_pipeline.beta_policy_screening_plan import JOB_CEILING_BASIS

    assert site_limit_module.JOB_CEILING_BASIS == JOB_CEILING_BASIS


def test_the_switch_ships_nested_and_refuses_an_unknown_mode(tmp_path, monkeypatch):
    from blueprint_pipeline import website_job_limits

    # The committed file a release installs (``INSTALLED_CONFIG_PATH`` on the host).
    monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(website_job_limits.SHIPPED_CONFIG_PATH))
    assert website_job_limits.limits_mode() == "nested" and website_job_limits.fuse() is False
    shipped = json.loads(website_job_limits.SHIPPED_CONFIG_PATH.read_text())
    assert shipped["mode"] == "nested" and shipped["schema_version"] == website_job_limits.SCHEMA
    for body in ({"schema_version": website_job_limits.SCHEMA, "mode": "fuze"},
                 {"schema_version": website_job_limits.SCHEMA}, {"mode": "fuse"},
                 {"schema_version": website_job_limits.SCHEMA, "mode": "fuse", "extra": 1}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(body))
        monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(path))
        with pytest.raises(website_job_limits.WebsiteJobLimitsInvalid):
            website_job_limits.limits_mode()
    monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(website_job_limits.WebsiteJobLimitsInvalid):
        website_job_limits.limits_mode()


def test_door_status_says_which_mode_the_release_runs(monkeypatch):
    from blueprint_pipeline import account_monthly_spend
    from blueprint_pipeline.account_monthly_spend_report import _website_job_limits_mode

    from blueprint_pipeline import website_job_limits

    monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(website_job_limits.SHIPPED_CONFIG_PATH))
    assert _website_job_limits_mode() == "nested"
    assert account_monthly_spend.summary({"website_job_limits_mode": "nested"})["website_job_limits_mode"] == "nested"


def test_the_money_switch_is_found_exactly_as_the_account_limit_is_and_a_release_carries_it(monkeypatch):
    """Review 8d (5): the installed release's ``config/``, unless the variable names a file; never the checkout."""
    import subprocess
    from pathlib import Path

    from blueprint_pipeline import account_monthly_spend as account
    from blueprint_pipeline import website_job_limits

    repo = Path(__file__).resolve().parents[1]
    assert website_job_limits.INSTALLED_CONFIG_PATH.parent == account.INSTALLED_CONFIG_PATH.parent
    assert website_job_limits.SHIPPED_CONFIG_PATH == repo / "config" / website_job_limits.CONFIG_FILENAME
    monkeypatch.delenv(website_job_limits.CONFIG_ENV, raising=False)
    assert website_job_limits.config_path() == website_job_limits.INSTALLED_CONFIG_PATH
    assert account.config_path({}) == account.INSTALLED_CONFIG_PATH
    monkeypatch.setenv(website_job_limits.CONFIG_ENV, "/elsewhere/limits.json")
    assert website_job_limits.config_path() == Path("/elsewhere/limits.json")
    # A release is the promoted source tree: the file is committed, and the screening unit runs that tree.
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "config/" + website_job_limits.CONFIG_FILENAME],
                             cwd=repo, capture_output=True, text=True)
    assert tracked.returncode == 0
    unit = (repo / "deploy/systemd/blueprint-beta-policy-screening.service").read_text()
    assert f"BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO={website_job_limits.INSTALLED_CONFIG_PATH.parents[1]}\n" in unit


@pytest.mark.parametrize("content", [None, "{not json", '{"schema_version": "blueprint.website_job_limits.v1", "mode": "x"}'])
def test_a_missing_or_unreadable_money_switch_stops_the_queue_with_one_line_naming_the_file(
        tmp_path, monkeypatch, capsys, content):
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline import website_job_limits

    path = tmp_path / "website_job_limits.v1.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(path))
    monkeypatch.setattr(worker, "main", lambda argv=None: website_job_limits.fuse() and 0)
    assert worker.entrypoint([]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and str(path) in err and err.startswith("beta screening stopped: website_job_limits_")
