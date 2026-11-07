"""The account-wide monthly limit (owner decision 2026-11-07: USD 200, alerts at 50/80/100 %)."""

from __future__ import annotations

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
    assert report_module.observation_readings(written, now=NOW + 60)["openai"] == {"status": "observed",
                                                                          "month_usd": Decimal("3.0")}


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
