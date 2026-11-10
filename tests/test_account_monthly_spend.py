"""The account-wide monthly limit (owner decision 2026-11-07: USD 200, alerts at 50/80/100 %)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from blueprint_pipeline import account_monthly_spend as account
from blueprint_pipeline import account_monthly_spend_report as report_module
from blueprint_pipeline import paid_spend_meter as meter
from blueprint_pipeline import provider_spend_alerts as spend
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
    require_paid_resource_admission_grant,
    require_paid_resource_budget,
)

from tests.test_gpu_spend_guard import patched_guard  # noqa: F401 - the guard's canned-provider fixture
from tests.test_paid_spend_meter import write_screening_ledger

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 20, 16, 0, tzinfo=timezone.utc).timestamp()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _policy(tmp_path: Path, *, limit: str = "200", overrides: dict | None = None) -> Path:
    path = tmp_path / "account_monthly_spend_limit.v1.json"
    path.write_text(json.dumps({"schema_version": account.POLICY_SCHEMA, "monthly_limit_usd": limit,
                                "monthly_limit_overrides": overrides or {},
                                "report_path": str(tmp_path / "account" / "report.json")}))
    return path


def _receipt(audit: Path, name: str, totals: dict) -> None:
    directory = audit / name
    directory.mkdir(parents=True)
    receipt = {"schema_version": account.BILLING_RECEIPT_SCHEMA, "status": "reconciled", "provider_totals_usd": totals}
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    receipt["receipt_digest"] = "sha256:" + hashlib.sha256(body).hexdigest()
    (directory / "provider_billing_source_receipt.json").write_text(json.dumps(receipt))


def _export(path: Path, totals: dict, *, generated: float) -> Path:
    path.write_text(json.dumps({"schema_version": account.BILLING_EXPORT_SCHEMA, "generated_at": _iso(generated),
                                "currency": "USD", "scope": account.BILLING_EXPORT_SCOPE,
                                "provider_totals_usd": totals}))
    return path


def _observation(root: Path, *, openai: float, observed: float = NOW - 300) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / account.OBSERVATION_FILENAME).write_text(json.dumps({
        "schema_version": account.OBSERVATION_SCHEMA, "observed_at_epoch": observed, "providers": {
            "openai": {"status": "observed", "blockers": [], "month": "2026-11", "month_to_date_usd": openai},
            "world_labs": {"status": "observed", "blockers": [], "month": "2026-11", "month_to_date_usd": 7.2,
                           "complete": True, "charges": [
                               {"at_epoch": NOW - 1000, "usd": 5.5, "held": False, "job_id": "site-req-1",
                                "attempt": "atlas:a"},
                               {"at_epoch": NOW - 900, "usd": 1.7, "held": True, "job_id": "site-req-3",
                                "attempt": "atlas:b"}]}}}))


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A host with every input: billing export and receipts, provider readings, screening and journal."""

    audit = tmp_path / "billing-audit"
    _receipt(audit, "20261031T235000.000000Z", {"vast": 950.0, "digitalocean": 280.0})
    _receipt(audit, "20261101T001000.000000Z", {"vast": 951.0, "digitalocean": 280.5})
    export = _export(tmp_path / "provider_billing_export.json", {"vast": 1000.0, "digitalocean": 300.0},
                     generated=NOW - 600)
    root = tmp_path / "account"
    _observation(root, openai=40.0)
    write_screening_ledger(tmp_path / "queue")
    journal = tmp_path / "meter"
    meter.record_admission(job_id="req-5", meter_key="k1", resource_class="openai_api_candidate",
                           held_usd="2.00", root=journal, now=NOW - 400)
    env = {account.CONFIG_ENV: str(_policy(tmp_path)), account.ROOT_ENV: str(root),
           account.BILLING_EXPORT_ENV: str(export), account.BILLING_AUDIT_ROOT_ENV: str(audit),
           meter.METER_ROOT_ENV: str(journal), meter.SCREENING_QUEUE_ROOT_ENV: str(tmp_path / "queue")}
    for name in (*env, "BLUEPRINT_GCP_BUDGET_GUARD_CONFIG", meter.JOB_ID_ENV):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    # The chokepoint judges the report against the module clock: pin it a minute after NOW.
    monkeypatch.setattr(account, "_clock", lambda: NOW + 60)
    return {"env": env, "root": root, "tmp": tmp_path}


def _admit(resource_class: str = "vast_provider_adapter"):
    return require_paid_resource_admission(build_paid_lane_admission(resource_class=resource_class),
                                           resource_class=resource_class,
                                           expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)


# ----------------------------- the shipped configuration -----------------------------


def test_the_shipped_config_is_200_dollars_and_parses():
    assert account.SHIPPED_CONFIG_PATH == REPO / "config" / "account_monthly_spend_limit.v1.json"
    policy = account.load_policy(account.SHIPPED_CONFIG_PATH)
    assert policy["monthly_limit_usd"] == "200"
    assert policy["monthly_limit_overrides"] == {}
    assert account.effective_limit(policy, "2026-11") == Decimal("200")
    assert policy["report_path"] == str(account.DEFAULT_ROOT / account.REPORT_FILENAME)
    assert policy["policy_sha256"] == "sha256:" + hashlib.sha256(account.SHIPPED_CONFIG_PATH.read_bytes()).hexdigest()


def test_the_deploy_installs_the_config_where_the_units_run_the_source_tree():
    assert account.INSTALLED_CONFIG_PATH.name == account.SHIPPED_CONFIG_PATH.name
    for unit in ("blueprint-gpu-spend-guard.service", "blueprint-control-plane-capacity.service"):
        text = (REPO / "deploy" / "systemd" / unit).read_text()
        assert f"BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO={account.INSTALLED_CONFIG_PATH.parents[1]}\n" in text
    capacity = (REPO / "deploy" / "systemd" / "blueprint-control-plane-capacity.service").read_text()
    assert f"ReadWritePaths=-{account.DEFAULT_ROOT}\n" in capacity
    assert f"install -d -o blueprint -g blueprint -m 0755 {account.DEFAULT_ROOT}'" in capacity


# ----------------------------- unset means report only -----------------------------


def test_no_config_means_no_enforcement(tmp_path, monkeypatch):
    monkeypatch.setenv(account.CONFIG_ENV, str(tmp_path / "absent.json"))
    assert account.configured_policy() == (None, None, tmp_path / "absent.json")
    assert account.admission_blockers(now=NOW) == []
    assert _admit().resource_class == "vast_provider_adapter"
    require_paid_resource_budget(resource_class="vast_provider_adapter")


def test_off_the_host_the_installed_config_is_absent_so_nothing_is_enforced(tmp_path, monkeypatch):
    monkeypatch.delenv(account.CONFIG_ENV, raising=False)
    monkeypatch.setattr(account, "INSTALLED_CONFIG_PATH", tmp_path / "opt" / account.CONFIG_FILENAME)
    assert account.config_path() == tmp_path / "opt" / account.CONFIG_FILENAME
    assert account.admission_blockers(now=NOW) == []
    assert _admit().resource_class == "vast_provider_adapter"


def test_unset_still_reports_and_the_alert_says_no_cap_is_set(host, monkeypatch):
    monkeypatch.setenv(account.CONFIG_ENV, str(host["tmp"] / "absent.json"))
    report = report_module.refresh_report(now=NOW)
    assert report["cap_status"] == "not_set" and report["limit_usd"] is None and report["blockers"] == []
    assert report["report_path"] == str(host["root"] / account.REPORT_FILENAME)
    assert report["counted_usd"] > 0
    row = account.observe_account_spend(now=NOW + 60)
    alerts = account.account_alerts(row)
    assert alerts[0] == {"code": "account_spend_limit_not_set", "provider": "account", "window": "utc_month:2026-11",
                         "spend_usd": report["counted_usd"]}
    assert "account_spend_limit_not_set" in spend.WARN_CODES


# ----------------------------- the view -----------------------------


def test_the_month_counts_reconciled_billing_plus_held_reservations(host):
    report = report_module.refresh_report(now=NOW)
    providers = report["providers"]
    # Vast: export 1000 less the last receipt before November (950); the meter settled 1.25.
    assert providers["vast"] == {**providers["vast"], "status": "observed", "reconciled_month_usd": 50.0,
                                 "metered_settled_usd": 1.25, "counted_usd": 50.0}
    assert providers["digitalocean"]["counted_usd"] == 20.0
    assert providers["openai"]["counted_usd"] == 40.0
    # World Labs' reading already includes its held attempt, so Atlas holds are not added again.
    assert providers["world_labs"]["counted_usd"] == 7.2
    assert providers["gcp"]["status"] == "unobserved" and providers["gcp"]["counted_usd"] == 0.0
    assert report["spend_usd"] == 117.2
    # Held: screening 8.00 (reserved 4 + committed without settlement 4) and the journal's 2.00.
    assert report["held_usd"] == 10.0
    assert report["counted_usd"] == 127.2
    assert report["limit_usd"] == "200" and report["alert_lines_usd"] == {"50": 100.0, "80": 160.0, "100": 200.0}
    assert report["alert_threshold_percent"] == 50 and report["limit_reached"] is False
    assert report["admission_allowed"] is True and report["blockers"] == []
    assert report["complete"] is False and "account_monthly_spend_provider_unobserved:gcp" in report["gaps"]
    written = json.loads(Path(report["report_path"]).read_text())
    assert written["counted_usd"] == 127.2 and written["policy_sha256"].startswith("sha256:")
    assert os.stat(report["report_path"]).st_mode & 0o777 == 0o644
    jobs = {row["job_id"]: row for row in report["meter"]["jobs"]}
    assert jobs["req-1"]["metered_usd"] == round(1.25 + 8.0 + 5.5, 6)


def test_without_a_receipt_before_the_month_the_first_one_of_the_month_stands_in(tmp_path):
    audit = tmp_path / "audit"
    _receipt(audit, "20261101T001000.000000Z", {"vast": 951.0})
    export = _export(tmp_path / "export.json", {"vast": 1000.0, "digitalocean": 10.0}, generated=NOW - 60)
    readings = report_module.billing_month_readings(export, audit, now=NOW)
    assert readings["vast"]["status"] == "partial" and readings["vast"]["month_usd"] == Decimal("49.0")
    assert readings["vast"]["reason"] == "billing_month_baseline_after_month_start"
    assert readings["digitalocean"]["status"] == "baseline_unavailable" and readings["digitalocean"]["month_usd"] is None


def test_a_billing_total_below_its_baseline_counts_zero_and_says_so(tmp_path):
    audit = tmp_path / "audit"
    _receipt(audit, "20261031T235000.000000Z", {"vast": 1200.0})
    export = _export(tmp_path / "export.json", {"vast": 1000.0}, generated=NOW - 60)
    reading = report_module.billing_month_readings(export, audit, now=NOW)["vast"]
    assert reading["month_usd"] == Decimal(0) and reading["reason"] == "billing_total_below_month_baseline"


def test_last_months_export_is_not_this_months_spend(tmp_path):
    export = _export(tmp_path / "export.json", {"vast": 1000.0}, generated=NOW - 25 * 86_400)
    assert report_module.billing_month_readings(export, None, now=NOW)["vast"]["status"] == "other_month"


def test_a_stale_provider_observation_still_counts_what_it_saw(host):
    _observation(host["root"], openai=40.0, observed=NOW - 3600)
    report = report_module.refresh_report(now=NOW)
    assert report["providers"]["openai"]["status"] == "stale"
    assert report["providers"]["openai"]["counted_usd"] == 40.0
    assert "account_monthly_spend_provider_stale:openai" in report["gaps"]


# ----------------------------- enforcement -----------------------------


def test_under_the_limit_a_paid_admission_is_allowed(host):
    report = report_module.refresh_report()
    assert report["counted_usd"] < 200
    assert account.admission_blockers() == []
    assert _admit().resource_class == "vast_provider_adapter"
    require_paid_resource_budget(resource_class="vast_provider_adapter")


def test_at_the_limit_a_paid_admission_is_refused_by_name(host):
    _observation(host["root"], openai=120.0)
    report = report_module.refresh_report()
    assert report["counted_usd"] >= 200 and report["limit_reached"] is True
    assert report["alert_threshold_percent"] == 100
    assert report["blockers"] == [account.LIMIT_REACHED] and report["admission_allowed"] is False
    assert account.admission_blockers() == ["account_monthly_spend_limit_reached"]
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        _admit()
    assert refused.value.blockers == ["account_monthly_spend_limit_reached"]
    with pytest.raises(PaidResourceAdmissionBlocked) as budget:
        require_paid_resource_budget(resource_class="openai_api_candidate")
    assert budget.value.blockers == ["account_monthly_spend_limit_reached"]


def test_exactly_200_reaches_the_limit_and_just_under_does_not(host):
    policy, _error, config = account.configured_policy()
    readings = {"openai": {"status": "observed", "month_usd": Decimal("200.00")}}
    empty = meter.meter_view([], now=NOW)
    at = report_module.build_report(now=NOW, policy=policy, policy_error=None, config=config, readings=readings, meter=empty)
    assert at["limit_reached"] is True and at["blockers"] == [account.LIMIT_REACHED]
    readings["openai"]["month_usd"] = Decimal("199.99")
    under = report_module.build_report(now=NOW, policy=policy, policy_error=None, config=config, readings=readings,
                                 meter=empty)
    assert under["limit_reached"] is False and under["alert_threshold_percent"] == 80


def test_held_reservations_alone_can_reach_the_limit(host):
    policy, _error, config = account.configured_policy()
    rows = [{"source": "beta_policy_screening", "job_id": "req-1", "meter_key": f"a{index}", "status": "held",
             "held_usd": Decimal("50"), "provider": "vast"} for index in range(4)]
    report = report_module.build_report(now=NOW, policy=policy, policy_error=None, config=config, readings={},
                                  meter=meter.meter_view(rows, now=NOW))
    assert report["held_usd"] == 200.0 and report["limit_reached"] is True


def test_an_issued_grant_is_never_stopped_by_the_limit(host):
    report_module.refresh_report()
    grant = _admit()
    _observation(host["root"], openai=500.0)
    report_module.refresh_report()
    assert account.admission_blockers() == [account.LIMIT_REACHED]
    require_paid_resource_admission_grant(grant, resource_class="vast_provider_adapter")


def test_a_dated_owner_override_sets_that_months_limit(host, monkeypatch):
    monkeypatch.setenv(account.CONFIG_ENV, str(_policy(host["tmp"], overrides={"2026-11": "300"})))
    _observation(host["root"], openai=120.0)
    report = report_module.refresh_report()
    assert report["counted_usd"] >= 200
    assert report["limit_usd"] == "300" and report["limit_reached"] is False
    assert account.admission_blockers() == []


@pytest.mark.parametrize("case", ["missing", "stale", "other_month", "policy_changed", "not_set_report"])
def test_with_a_limit_set_anything_but_a_fresh_bound_report_refuses(host, monkeypatch, case):
    report = report_module.refresh_report(now=NOW)
    path = Path(report["report_path"])
    expected = account.REPORT_INVALID
    now = NOW + 60
    if case == "missing":
        path.unlink()
    elif case == "stale":
        now = NOW + account.REPORT_MAX_AGE_SECONDS + 1
        expected = account.REPORT_STALE
    elif case == "other_month":
        now = datetime(2026, 12, 1, 0, 1, tzinfo=timezone.utc).timestamp()
        expected = account.REPORT_STALE
    elif case == "policy_changed":
        monkeypatch.setenv(account.CONFIG_ENV, str(_policy(host["tmp"], limit="250")))
    else:
        path.write_text(json.dumps({**json.loads(path.read_text()), "cap_status": "not_set"}))
    assert account.admission_blockers(now=now) == [expected]


@pytest.mark.parametrize("content", ["not json", json.dumps({"schema_version": account.POLICY_SCHEMA,
                                                             "monthly_limit_usd": 200, "report_path": "/x"}),
                                     json.dumps({"schema_version": account.POLICY_SCHEMA, "monthly_limit_usd": "0",
                                                 "report_path": "/x"}),
                                     json.dumps({"schema_version": account.POLICY_SCHEMA, "monthly_limit_usd": "200",
                                                 "report_path": "relative/report.json"}),
                                     json.dumps({"schema_version": account.POLICY_SCHEMA, "monthly_limit_usd": "200",
                                                 "report_path": "/x", "monthly_limit": "999"})])
def test_a_config_that_exists_but_does_not_validate_refuses_all_paid_work(tmp_path, monkeypatch, content):
    path = tmp_path / "limit.json"
    path.write_text(content)
    monkeypatch.setenv(account.CONFIG_ENV, str(path))
    assert account.admission_blockers(now=NOW) == [account.CONFIG_INVALID]
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        _admit()
    assert refused.value.blockers == [account.CONFIG_INVALID]


def test_a_symlinked_config_is_refused(tmp_path, monkeypatch):
    target = _policy(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    monkeypatch.setenv(account.CONFIG_ENV, str(link))
    assert account.admission_blockers(now=NOW) == [account.CONFIG_INVALID]


def test_the_existing_gcp_guard_still_applies_beside_the_account_limit(host, monkeypatch):
    report_module.refresh_report()
    monkeypatch.setenv("BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", str(host["tmp"] / "absent-gcp-report.json"))
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        _admit()
    assert refused.value.blockers == ["gcp_monthly_budget_report_missing_or_invalid"]


# ----------------------------- alerts -----------------------------


def _row(host, *, openai: float) -> dict:
    # Each case starts the month afresh: within a month the count never goes down.
    (host["root"] / account.REPORT_FILENAME).unlink(missing_ok=True)
    _observation(host["root"], openai=openai)
    report_module.refresh_report(now=NOW)
    return account.observe_account_spend(now=NOW + 60)


def test_alerts_at_50_80_and_100_percent_of_200(host):
    # Everything but OpenAI counts 87.2 (vast 50, DigitalOcean 20, World Labs 7.2, held 10).
    cases = [(5.0, None), (12.8, "account_spend_month_50pct"), (72.8, "account_spend_month_80pct"),
             (112.8, "account_spend_month_limit_reached")]
    for openai, code in cases:
        alerts = account.account_alerts(_row(host, openai=openai))
        codes = [row["code"] for row in alerts if row["code"].startswith("account_spend_month")]
        assert codes == ([code] if code else []), (openai, alerts)
    lines = {row["code"]: row for row in account.account_alerts(_row(host, openai=72.8))}
    assert lines["account_spend_month_80pct"]["threshold_usd"] == 160.0
    assert lines["account_spend_month_80pct"]["limit_usd"] == 200.0
    assert "account_spend_month_50pct" in spend.WARN_CODES
    assert {"account_spend_month_80pct", "account_spend_month_limit_reached"} <= spend.PAGE_CODES
    assert spend.describe(lines["account_spend_month_80pct"]) == (
        "Account $160.00 of $200.00 in UTC month 2026-11 (80% line $160.00)")
    reached = account.account_alerts(_row(host, openai=112.8))
    assert spend.describe(reached[0]).endswith("monthly limit reached, new paid admissions refused")


def test_the_alerts_go_through_the_provider_spend_check(host, monkeypatch):
    _row(host, openai=112.8)
    observation = spend.observe_provider_spend(NOW + 60, environ={**host["env"], "OPENAI_ADMIN_API_KEY_FILE": ""})
    assert observation["providers"]["account"]["counted_usd"] == 200.0
    assert "account_spend_month_limit_reached" in [row["code"] for row in observation["alerts"]]
    summary = spend.summary_projection(observation)["providers"]["account"]
    assert summary["cap_status"] == "set" and summary["limit_reached"] is True


def test_a_stale_report_with_a_limit_set_pages_and_unset_only_warns(host, monkeypatch):
    report_module.refresh_report(now=NOW)
    late = NOW + account.REPORT_MAX_AGE_SECONDS + 60
    assert [row["code"] for row in account.account_alerts(account.observe_account_spend(now=late))] == [
        "account_spend_report_stale"]
    monkeypatch.setenv(account.CONFIG_ENV, str(host["tmp"] / "absent.json"))
    report_module.refresh_report(now=NOW)
    assert [row["code"] for row in account.account_alerts(account.observe_account_spend(now=late))] == [
        "account_spend_unverifiable"]


def test_an_invalid_config_pages(tmp_path, monkeypatch):
    path = tmp_path / "limit.json"
    path.write_text("{}")
    monkeypatch.setenv(account.CONFIG_ENV, str(path))
    row = account.observe_account_spend(now=NOW, environ={account.CONFIG_ENV: str(path)})
    assert account.account_alerts(row)[0]["code"] == "account_spend_limit_config_invalid"
    assert "account_spend_limit_config_invalid" in spend.PAGE_CODES


def test_the_capacity_check_leaves_its_readings_for_the_guard(tmp_path, monkeypatch):
    monkeypatch.setenv(account.ROOT_ENV, str(tmp_path / "account"))
    observation = {"observed_at_epoch": NOW, "providers": {
        "openai": {"status": "observed", "blockers": [], "month": "2026-11", "month_to_date_usd": 3.0,
                   "day_usd": {"2026-11-20": 1.0}},
        "world_labs": {"status": "observed", "blockers": [], "month": "2026-11", "month_to_date_usd": 1.0,
                       "charges": [], "complete": True},
        "gcp": {"observed_cost_usd": 5.0}}}
    assert account.persist_provider_observation(observation) is True
    written = json.loads((tmp_path / "account" / account.OBSERVATION_FILENAME).read_text())
    assert set(written["providers"]) == {"openai", "world_labs"}
    assert "day_usd" not in written["providers"]["openai"]
    assert os.stat(tmp_path / "account" / account.OBSERVATION_FILENAME).st_mode & 0o777 == 0o644
    assert report_module.observation_readings(written, now=NOW + 60)["openai"] == {
        "status": "observed", "month_usd": Decimal("3.0"), "observed_at": "2026-11-20T16:00:00+00:00"}


def test_the_admission_path_stays_light():
    from tests.test_live_pipeline_import_isolation import HOT_LANE_MODULES, _transitive_local_modules

    reachable = _transitive_local_modules(("paid_resource_admission",))
    assert {"account_monthly_spend", "paid_spend_meter"} <= reachable
    assert not reachable & {"account_monthly_spend_report", "paid_spend_meter_sources", "provider_spend_alerts",
                            "beta_policy_screening_coordinator", *HOT_LANE_MODULES}


def test_the_spend_guard_rewrites_the_report_and_carries_its_summary(host):
    from scripts import gpu_spend_guard as fleet

    summary = fleet._refresh_account_monthly_spend(host["env"][account.BILLING_EXPORT_ENV], now=NOW + 60)
    assert summary["cap_status"] == "set" and summary["counted_usd"] == 127.2 and summary["limit_usd"] == "200"
    assert summary["jobs"][0]["job_id"] == "req-1"
    assert account.admission_blockers() == []


def test_a_guard_run_embeds_the_account_summary_in_its_report(patched_guard, monkeypatch, capsys):  # noqa: F811
    from scripts import gpu_spend_guard as fleet

    tmp_path, _terminated = patched_guard
    monkeypatch.setenv(account.CONFIG_ENV, str(tmp_path / "absent.json"))
    report_path = tmp_path / "ops" / "latest.json"
    fleet.main(["--output-root", str(tmp_path), "--json-report", str(report_path)])
    embedded = json.loads(report_path.read_text())["account_monthly_spend"]
    assert embedded["cap_status"] == "not_set" and embedded["admission_allowed"] is True
    assert "status" not in embedded  # not {"status": "refresh_failed"}
    assert "Account monthly spend: $" in capsys.readouterr().err


# ----------------------------- stops are never blocked (review finding 1) -----------------------------


def _stop_admission(resource_class: str = "gpu_render", **extra) -> dict:
    return {**build_paid_lane_admission(resource_class=resource_class), **extra}


@pytest.mark.parametrize("state", ["limit_reached", "report_stale"])
def test_a_clear_stop_is_admitted_when_the_limit_refuses_new_spend(host, state):
    _observation(host["root"], openai=500.0)
    report_module.refresh_report(now=NOW if state == "limit_reached" else NOW - 3600)
    assert account.admission_blockers() == [account.LIMIT_REACHED if state == "limit_reached" else account.REPORT_STALE]
    grant = require_paid_resource_admission(_stop_admission(spend_effect="stop"), resource_class="gpu_render",
                                            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                            spend_effect="stop")
    assert grant.resource_class == "gpu_render"
    with pytest.raises(PaidResourceAdmissionBlocked):
        _admit("gpu_render")


@pytest.mark.parametrize("record, keyword, expected", [
    ({}, "stop", ["account_monthly_spend_limit_reached", "paid_resource_admission_spend_effect_mismatch"]),
    ({"spend_effect": "stop"}, "new_spend", ["account_monthly_spend_limit_reached"]),
    ({"spend_effect": "stop"}, "teardown", ["account_monthly_spend_limit_reached",
                                            "paid_resource_admission_spend_effect_invalid"]),
])
def test_anything_but_a_stop_declared_twice_is_checked_as_new_spend(host, record, keyword, expected):
    _observation(host["root"], openai=500.0)
    report_module.refresh_report()
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        require_paid_resource_admission(_stop_admission(**record), resource_class="gpu_render",
                                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                        spend_effect=keyword)
    assert refused.value.blockers == expected


def test_a_stop_is_still_refused_for_its_own_admission_blockers(host):
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        require_paid_resource_admission({**_stop_admission(spend_effect="stop"), "status": "blocked",
                                         "blockers": ["x"]}, resource_class="gpu_render",
                                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                        spend_effect="stop")
    assert "paid_resource_admission_not_admitted" in refused.value.blockers


def _qualification(tmp_path, action: str):
    from blueprint_pipeline.qualification_control_admission import admit_qualification_control_mutation

    manifest = {"continuing_spend": True, "watchdog_deadline_epoch": NOW + 86_400, "resource_name": "g1-session",
                "launch_session_id": "session-1"}
    inspected = {"status": "observed", "instance_id": "4242", "name": "g1-session"}
    out = tmp_path / f"control-{action}.json"
    grant = admit_qualification_control_mutation(out, manifest, inspected, "4242", action, "controller",
                                                 clock=lambda: NOW)
    return grant, json.loads(out.read_text())


def test_an_operator_can_stop_a_qualification_component_at_the_limit(host):
    _observation(host["root"], openai=500.0)
    report_module.refresh_report()
    assert account.admission_blockers() == [account.LIMIT_REACHED]
    grant, admission = _qualification(host["tmp"], "stop")
    assert grant.resource_class == "gpu_render" and admission["spend_effect"] == "stop"
    for action in ("run", "restart", "refresh", "install-checkpoint"):
        with pytest.raises(PaidResourceAdmissionBlocked) as refused:
            _qualification(host["tmp"], action)
        assert refused.value.blockers == [account.LIMIT_REACHED]


def test_a_warm_scene_closeout_is_admitted_at_the_limit_and_an_iteration_is_not(host):
    from blueprint_pipeline import task_evaluation_scene_configuration_allocator as allocator

    _observation(host["root"], openai=500.0)
    report_module.refresh_report()
    recorded = {}
    for action in ("closeout", "iterate"):
        out = host["tmp"] / f"warm-{action}.json"
        allocator._run_scene_configuration_warm_action(argparse.Namespace(
            scene_configuration_warm_action=action, scene_configuration_warm_session_root=None,
            scene_configuration_warm_closeout_receipt=None, scene_configuration_warm_iteration_authority=None,
            scene_configuration_job_dir=None, provider="vast", admission_out=str(out),
            adapter_output=str(host["tmp"] / f"warm-{action}-result.json"), execute=False))
        recorded[action] = {**json.loads(out.read_text()), "status": "admitted", "blockers": []}
    assert recorded["closeout"]["spend_effect"] == "stop" and recorded["iterate"]["spend_effect"] == "new_spend"
    require_paid_resource_admission(recorded["closeout"], resource_class="vast_provider_adapter",
                                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                    spend_effect=recorded["closeout"]["spend_effect"])
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        require_paid_resource_admission(recorded["iterate"], resource_class="vast_provider_adapter",
                                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                        spend_effect=recorded["iterate"]["spend_effect"])
    assert refused.value.blockers == [account.LIMIT_REACHED]


def test_stops_are_journaled_as_stops(host):
    grant = require_paid_resource_admission(_stop_admission(spend_effect="stop", request_id="req-7"),
                                            resource_class="gpu_render",
                                            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                            spend_effect="stop")
    assert grant is not None
    events, _gaps = meter.read_journal(Path(host["env"][meter.METER_ROOT_ENV]))
    assert [(row["job_id"], row.get("spend_effect")) for row in events][-1] == ("req-7", "stop")


# ----------------------------- a failed reading never lowers the count (review finding 2) -----------------------------


def test_a_failed_provider_reading_keeps_the_last_count_for_the_month(host):
    _observation(host["root"], openai=210.0)
    first = report_module.refresh_report(now=NOW)
    assert first["limit_reached"] is True and first["providers"]["openai"]["counted_usd"] == 210.0
    # The next capacity tick's Costs API call fails: OpenAI's month comes back null.
    observation = json.loads((host["root"] / account.OBSERVATION_FILENAME).read_text())
    observation["observed_at_epoch"] = NOW + 600
    observation["providers"]["openai"] = {"status": "unknown", "blockers": ["openai_spend_transport_failed"],
                                          "month": "2026-11", "month_to_date_usd": None}
    (host["root"] / account.OBSERVATION_FILENAME).write_text(json.dumps(observation))
    second = report_module.refresh_report(now=NOW + 660)
    openai = second["providers"]["openai"]
    assert openai["counted_usd"] == 210.0 and openai["status"] == "stale"
    assert openai["held_from_previous_report"] is True
    assert openai["observed_at"] == "2026-11-20T15:55:00+00:00"  # when the kept reading was observed
    assert second["counted_usd"] >= first["counted_usd"] and second["limit_reached"] is True
    assert "account_monthly_spend_provider_held_at_last_count:openai" in second["gaps"]
    assert account.admission_blockers(now=NOW + 700) == [account.LIMIT_REACHED]
    # A third failed tick still holds the figure and the time it was observed.
    third = report_module.refresh_report(now=NOW + 1260)
    assert third["providers"]["openai"]["counted_usd"] == 210.0
    assert third["providers"]["openai"]["observed_at"] == "2026-11-20T15:55:00+00:00"


@pytest.mark.parametrize("breakage", ["export_unreadable", "lower_partial_reading", "screening_ledger_unreadable"])
def test_no_monitoring_failure_lowers_the_counted_total(host, breakage):
    first = report_module.refresh_report(now=NOW)
    if breakage == "export_unreadable":
        Path(host["env"][account.BILLING_EXPORT_ENV]).write_text("{not json")
    elif breakage == "lower_partial_reading":
        _observation(host["root"], openai=10.0)
        observation = json.loads((host["root"] / account.OBSERVATION_FILENAME).read_text())
        observation["providers"]["openai"]["status"] = "partial"
        (host["root"] / account.OBSERVATION_FILENAME).write_text(json.dumps(observation))
    else:
        (host["tmp"] / "queue" / "screening_ledger.v1.json").write_text("{not json")
    second = report_module.refresh_report(now=NOW + 60)
    assert second["counted_usd"] >= first["counted_usd"], (first["counted_usd"], second["counted_usd"])
    assert second["complete"] is False


def test_a_new_month_starts_from_its_own_readings(host):
    _observation(host["root"], openai=210.0)
    report_module.refresh_report(now=NOW)
    december = datetime(2026, 12, 2, 12, 0, tzinfo=timezone.utc).timestamp()
    report = report_module.refresh_report(now=december)
    assert report["period"] == "2026-12"
    assert not any(row.get("held_from_previous_report") for row in report["providers"].values())


# ----------------------------- the meter CLI shows Atlas (review finding 3) -----------------------------


def test_the_meter_cli_includes_the_retained_atlas_attempts(host, capsys):
    from blueprint_pipeline import paid_spend_meter_sources as sources

    monkey_now = NOW + 60
    sources.main([], now=monkey_now)
    view = json.loads(capsys.readouterr().out)
    jobs = {row["job_id"]: row for row in view["jobs"]}
    assert jobs["req-1"]["by_source"]["atlas"]["settled_usd"] == 5.5
    assert jobs["req-3"]["by_source"]["atlas"]["held_usd"] == 1.7
    assert "paid_spend_meter_atlas_attempts_unobserved" not in view["gaps"]


# ----------------------------- the cap cannot silently not exist in production (audit 2026-11-10) -----------------------------

PRODUCTION = "BLUEPRINT_LAUNCH_PROOF_MODE"


@pytest.fixture
def no_limit_file(tmp_path, monkeypatch):
    """No limit file, and neither sign of the production control plane until a test adds one."""

    monkeypatch.setenv(account.CONFIG_ENV, str(tmp_path / "absent.json"))
    monkeypatch.delenv(PRODUCTION, raising=False)
    state = tmp_path / "var-lib-blueprint" / "pipeline-control-plane"
    monkeypatch.setattr(account, "DEFAULT_ROOT", state / "account-monthly-spend")
    return state


@pytest.mark.parametrize("sign", ["launch_mode", "state_directory"])
def test_on_the_production_control_plane_a_missing_limit_file_refuses_new_spend(no_limit_file, monkeypatch, sign):
    assert account.limit_required() is False and account.admission_blockers(now=NOW) == []
    admitted_before = _admit()
    if sign == "launch_mode":
        monkeypatch.setenv(PRODUCTION, "production")  # what the units' environment file sets
    else:
        no_limit_file.mkdir(parents=True)  # the control-plane state directory: only the host has it
    assert account.limit_required() is True
    assert account.admission_blockers(now=NOW) == ["account_monthly_spend_limit_not_configured"]
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        _admit()
    assert refused.value.blockers == [account.LIMIT_NOT_CONFIGURED]
    with pytest.raises(PaidResourceAdmissionBlocked) as budget:
        require_paid_resource_budget(resource_class="openai_api_candidate")
    assert budget.value.blockers == [account.LIMIT_NOT_CONFIGURED]
    # A stop is never blocked, and work admitted earlier keeps its grant.
    stop = require_paid_resource_admission(_stop_admission(spend_effect="stop"), resource_class="gpu_render",
                                           expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                           spend_effect="stop")
    assert stop.resource_class == "gpu_render"
    require_paid_resource_admission_grant(admitted_before, resource_class="vast_provider_adapter")


@pytest.mark.parametrize("mode", [None, "", "local", "staging", "not-production"])
def test_outside_production_a_missing_limit_file_still_means_no_limit(no_limit_file, monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv(PRODUCTION, mode)
    assert account.limit_required() is False
    assert account.configured_policy()[:2] == (None, None)
    assert account.admission_blockers(now=NOW) == []
    assert _admit().resource_class == "vast_provider_adapter"
    require_paid_resource_budget(resource_class="vast_provider_adapter")
    report = report_module.build_report(now=NOW, policy=None, policy_error=None, config=Path("/absent"), readings={},
                                        meter=meter.meter_view([], now=NOW),
                                        limit_is_required=account.limit_required())
    assert report["cap_status"] == "not_set" and report["blockers"] == [] and report["admission_allowed"] is True


def test_the_launch_mode_is_read_as_the_units_set_it(no_limit_file):
    assert account.limit_required({PRODUCTION: " Production "}) is True
    assert account.limit_required({}) is False


def test_with_the_limit_installed_production_behaves_exactly_as_before(host, monkeypatch):
    before = report_module.refresh_report(now=NOW)
    monkeypatch.setenv(PRODUCTION, "production")
    after = report_module.refresh_report(now=NOW)
    assert after == before and after["cap_status"] == "set" and after["blockers"] == []
    assert account.admission_blockers() == []
    assert _admit().resource_class == "vast_provider_adapter"
    assert [row["code"] for row in account.account_alerts(account.observe_account_spend(now=NOW + 60))] == [
        "account_spend_month_50pct", "account_spend_unverifiable"]


@pytest.mark.parametrize("content", ["not json", "{}"])
def test_an_invalid_limit_file_is_still_named_as_invalid_in_production(tmp_path, monkeypatch, content):
    path = tmp_path / "limit.json"
    path.write_text(content)
    monkeypatch.setenv(account.CONFIG_ENV, str(path))
    monkeypatch.setenv(PRODUCTION, "production")
    assert account.admission_blockers(now=NOW) == [account.CONFIG_INVALID]
    assert account.account_alerts(account.observe_account_spend(now=NOW))[0]["code"] == (
        "account_spend_limit_config_invalid")


def test_in_production_the_report_says_the_cap_is_missing_and_the_capacity_check_pages(host, monkeypatch):
    from blueprint_pipeline import control_plane_capacity_controller as capacity

    absent = str(host["tmp"] / "absent.json")
    monkeypatch.setenv(account.CONFIG_ENV, absent)
    monkeypatch.setenv(PRODUCTION, "production")
    report = report_module.refresh_report(now=NOW)
    assert report["cap_status"] == "missing" and report["limit_usd"] is None and report["limit_reached"] is False
    assert report["blockers"] == [account.LIMIT_NOT_CONFIGURED] and report["admission_allowed"] is False
    assert report["counted_usd"] == 127.2  # the month is still counted and shown
    assert json.loads(Path(report["report_path"]).read_text())["cap_status"] == "missing"

    row = account.observe_account_spend(now=NOW + 60)
    assert row["cap_status"] == "missing" and row["admission_allowed"] is False
    alerts = account.account_alerts(row)
    assert alerts == [{"code": "account_spend_limit_missing", "provider": "account", "window": "utc_month:2026-11"}]
    assert "account_spend_limit_missing" in spend.PAGE_CODES - spend.WARN_CODES
    assert capacity._severity("account_spend_limit_missing") == "page"
    assert spend.describe(alerts[0]) == (
        "Account monthly limit config missing on the control plane: every new paid admission is refused")

    # The existing route: the capacity tick's provider spend check and its door-readable summary.
    observation = spend.observe_provider_spend(NOW + 60, environ={
        **host["env"], account.CONFIG_ENV: absent, PRODUCTION: "production", "OPENAI_ADMIN_API_KEY_FILE": ""})
    assert "account_spend_limit_missing" in [row["code"] for row in observation["alerts"]]
    summary = spend.summary_projection(observation)["providers"]["account"]
    assert summary["cap_status"] == "missing" and summary["admission_allowed"] is False

    # It pages whatever state the report is in: unreadable, or stale.
    Path(report["report_path"]).unlink()
    assert [row["code"] for row in account.account_alerts(account.observe_account_spend(now=NOW + 60))] == [
        "account_spend_limit_missing"]
    # The same file state off the host only warns, as before.
    monkeypatch.delenv(PRODUCTION)
    report_module.refresh_report(now=NOW)
    assert [row["code"] for row in account.account_alerts(account.observe_account_spend(now=NOW + 60))][0] == (
        "account_spend_limit_not_set")


def test_in_production_the_spend_guard_report_carries_the_missing_cap(host, monkeypatch, capsys):
    from scripts import gpu_spend_guard as fleet

    monkeypatch.setenv(account.CONFIG_ENV, str(host["tmp"] / "absent.json"))
    monkeypatch.setenv(PRODUCTION, "production")
    summary = fleet._refresh_account_monthly_spend(host["env"][account.BILLING_EXPORT_ENV], now=NOW + 60)
    assert summary["cap_status"] == "missing" and summary["admission_allowed"] is False
    assert summary["blockers"] == [account.LIMIT_NOT_CONFIGURED] and summary["counted_usd"] == 127.2
    assert "limit file missing, new paid admissions refused" in capsys.readouterr().err


def test_the_cli_names_the_missing_cap(host, monkeypatch, capsys):
    monkeypatch.setenv(account.CONFIG_ENV, str(host["tmp"] / "absent.json"))
    monkeypatch.setenv(PRODUCTION, "production")
    assert account.main([]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["cap_status"] == "missing" and shown["admission_blockers"] == [account.LIMIT_NOT_CONFIGURED]


# ----------------------------- old unsettled holds do not shrink the month (audit 2026-11-10) -----------------------------


def test_one_unsettled_hold_from_october_and_one_from_this_month_count_only_this_months(host):
    from tests.test_paid_spend_meter import _held_for_good, _ledger

    _ledger(host["tmp"] / "queue", {
        "october": _held_for_good("screen-oct", launched="2026-10-14T09:00:00+00:00",
                                  held_at="2026-10-14T11:30:00+00:00"),
        "november": _held_for_good("screen-nov", launched="2026-11-05T09:00:00+00:00",
                                   held_at="2026-11-05T10:00:00+00:00"),
    })
    report = report_module.refresh_report(now=NOW)
    # Providers 117.20 as before; held: this month's 4.00 and the journal's 2.00. October's 4.00 is not counted.
    assert report["spend_usd"] == 117.2 and report["held_usd"] == 6.0 and report["counted_usd"] == 123.2
    assert report["held_by_source_usd"] == {"beta_policy_screening": 4.0, "paid_resource_admission": 2.0}
    assert report["old_unsettled_held_usd"] == 4.0
    assert report["meter"]["old_unsettled_holds"] == [{
        "job_id": "req-1", "source": "beta_policy_screening", "meter_key": "screen-oct", "provider": "vast",
        "held_usd": 4.0, "billable_until": "2026-10-14T11:30:00+00:00", "reason": "beta_screening_settlement_unproven"}]
    # An old hold is not a missing reading: it adds no gap of its own.
    assert not [gap for gap in report["gaps"] if "hold" in gap]
    shown = account.summary(report)
    assert shown["old_unsettled_held_usd"] == 4.0 and shown["old_unsettled_holds"][0]["meter_key"] == "screen-oct"
    # In December neither attempt can be billed any more; both are listed, neither counted.
    december = datetime(2026, 12, 2, 12, 0, tzinfo=timezone.utc).timestamp()
    later = report_module.refresh_report(now=december)
    assert later["held_by_source_usd"]["beta_policy_screening"] == 0.0 and later["old_unsettled_held_usd"] == 8.0


def test_old_holds_alone_never_reach_the_limit(host):
    policy, _error, config = account.configured_policy()
    rows = [{"source": "beta_policy_screening", "job_id": "req-1", "meter_key": f"a{index}", "status": "held",
             "held_usd": Decimal("50"), "provider": "vast",
             "billable_until_epoch": datetime(2026, 10, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()}
            for index in range(4)]
    report = report_module.build_report(now=NOW, policy=policy, policy_error=None, config=config, readings={},
                                        meter=meter.meter_view(rows, now=NOW))
    assert report["held_usd"] == 0.0 and report["counted_usd"] == 0.0 and report["limit_reached"] is False
    assert report["old_unsettled_held_usd"] == 200.0


def test_an_unreadable_screening_ledger_still_keeps_this_months_holds_not_the_old_ones(host):
    from tests.test_paid_spend_meter import _held_for_good, _ledger

    _ledger(host["tmp"] / "queue", {
        "october": _held_for_good("screen-oct", launched="2026-10-14T09:00:00+00:00",
                                  held_at="2026-10-14T11:30:00+00:00"),
        "november": _held_for_good("screen-nov", launched="2026-11-05T09:00:00+00:00",
                                   held_at="2026-11-05T10:00:00+00:00"),
    })
    first = report_module.refresh_report(now=NOW)
    (host["tmp"] / "queue" / "screening_ledger.v1.json").write_text("{not json")
    second = report_module.refresh_report(now=NOW + 60)
    assert second["held_by_source_usd"]["beta_policy_screening"] == 4.0 == first["held_by_source_usd"]["beta_policy_screening"]
    assert second["counted_usd"] == first["counted_usd"]


# ----------------------------- a carried billing reading is counted and never reported as observed -----------------------------


def _carried_export(host, *, observed: float, generated: float = NOW - 600) -> None:
    Path(host["env"][account.BILLING_EXPORT_ENV]).write_text(json.dumps({
        "schema_version": account.BILLING_EXPORT_SCHEMA, "generated_at": _iso(generated), "currency": "USD",
        "scope": account.BILLING_EXPORT_SCOPE, "provider_totals_usd": {"digitalocean": 300.0},
        "carried_forward": {"vast": {"total_usd": 1000.0, "observed_at": _iso(observed),
                                     "age_seconds": generated - observed,
                                     "failure": "provider_billing_request_failed", "failed_at": _iso(generated)}}}))


def test_a_carried_billing_reading_keeps_the_providers_month_and_says_the_call_failed(host):
    first = report_module.refresh_report(now=NOW)
    assert first["providers"]["vast"]["status"] == "observed" and first["providers"]["vast"]["counted_usd"] == 50.0
    _carried_export(host, observed=NOW - 3 * 3600)
    report = report_module.refresh_report(now=NOW + 60)
    vast = report["providers"]["vast"]
    assert vast["counted_usd"] == 50.0 and vast["reconciled_month_usd"] == 50.0
    assert vast["status"] == "carried_forward" and vast["reason"] == "billing_reading_carried_forward"
    assert vast["observed_at"] == _iso(NOW - 3 * 3600)  # the time it was really read
    assert vast["carried_forward"] == {"observed_at": _iso(NOW - 3 * 3600),
                                       "failure": "provider_billing_request_failed", "failed_at": _iso(NOW - 600)}
    assert "held_from_previous_report" not in vast
    assert "account_monthly_spend_provider_carried_forward:vast" in report["gaps"] and report["complete"] is False
    assert report["providers"]["digitalocean"]["status"] == "observed"
    assert report["counted_usd"] == first["counted_usd"]
    # Door status (the guard's summary) and the capacity warn both show it.
    shown = account.summary(report)["providers"]["vast"]
    assert shown["status"] == "carried_forward" and shown["carried_forward"]["failure"] == (
        "provider_billing_request_failed")
    assert "carried_forward" not in account.summary(report)["providers"]["digitalocean"]
    alerts = account.account_alerts(account.observe_account_spend(now=NOW + 120))
    assert "account_monthly_spend_provider_carried_forward:vast" in alerts[-1]["reason"]


def test_a_carried_reading_from_before_the_month_or_too_old_is_not_this_months_spend(host, tmp_path):
    month_start = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    export = Path(host["env"][account.BILLING_EXPORT_ENV])
    audit = Path(host["env"][account.BILLING_AUDIT_ROOT_ENV])
    # Read at 23:50 on the 31st, carried into the first hours of November: not November's spend.
    _carried_export(host, observed=month_start - 600, generated=month_start + 3600)
    early = report_module.billing_month_readings(export, audit, now=month_start + 3700)
    assert early["vast"]["status"] == "other_month" and early["vast"]["month_usd"] is None
    assert early["vast"]["carried_forward"]["failure"] == "provider_billing_request_failed"
    assert early["digitalocean"]["status"] == "observed"  # a fresh reading is judged by its own time
    # Older than the 24 hours billing evidence is accepted for, or dated ahead: ignored.
    for observed in (NOW - account.BILLING_MAX_AGE_SECONDS - 1, NOW + 3600):
        _carried_export(host, observed=observed)
        assert "vast" not in report_module.billing_month_readings(export, audit, now=NOW)
    # A fresh total always wins over a carried row for the same provider.
    payload = json.loads(export.read_text())
    payload["provider_totals_usd"]["vast"] = 1010.0
    payload["carried_forward"]["vast"]["observed_at"] = _iso(NOW - 3600)
    export.write_text(json.dumps(payload))
    fresh = report_module.billing_month_readings(export, audit, now=NOW)["vast"]
    assert fresh["status"] == "observed" and fresh["month_usd"] == Decimal("60.0") and "carried_forward" not in fresh
