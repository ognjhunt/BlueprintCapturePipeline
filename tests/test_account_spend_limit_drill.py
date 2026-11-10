"""WP-13 / G17: the $0 limit drill, the month's forecast and the ledger-to-billing reconciliation."""

import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from blueprint_pipeline import account_monthly_spend as account
from blueprint_pipeline import account_spend_limit_drill as drill
from blueprint_pipeline.account_spend_forecast import forecast
from blueprint_pipeline.account_spend_reconciliation import reconcile
from blueprint_pipeline.paid_spend_meter import month_of

SHIPPED = Path(__file__).resolve().parents[1] / "config" / account.CONFIG_FILENAME


def test_drill_walks_the_real_admission_code_through_the_limit(tmp_path):
    result = drill.run_drill(work_root=tmp_path / "drill", config=SHIPPED)
    assert result["status"] == "passed", result["failures"]
    assert result["limit_usd"] == 200.0 and result["spend_usd"] == 0
    by_percent = {step["percent"]: step for step in result["steps"]}
    assert by_percent[49]["threshold_alerts"] == [] and by_percent[49]["admission_blockers"] == []
    assert by_percent[50]["threshold_alerts"] == [{"code": "account_spend_month_50pct", "severity": "warn"}]
    assert by_percent[80]["threshold_alerts"] == [{"code": "account_spend_month_80pct", "severity": "page"}]
    assert by_percent[99]["chokepoint_blockers"] == []
    for percent in (100, 110):
        assert by_percent[percent]["threshold_alerts"] == [
            {"code": "account_spend_month_limit_reached", "severity": "page"}]
        assert by_percent[percent]["admission_blockers"] == [account.LIMIT_REACHED]
        assert by_percent[percent]["chokepoint_blockers"] == [account.LIMIT_REACHED]
    # The drill's own copy only: the shipped limit file is unchanged.
    assert json.loads(SHIPPED.read_text())["monthly_limit_overrides"] == {}


def test_a_dated_override_moves_the_refusal_to_its_amount(tmp_path):
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"monthly_limit_overrides": {month_of(time.time()): "5"}}))
    result = drill.run_drill(work_root=tmp_path / "drill", config=SHIPPED, override_file=override)
    assert result["status"] == "passed" and result["limit_usd"] == 5.0 and result["dated_override"] == "5"
    assert [step["counted_usd"] for step in result["steps"] if step["admission_blockers"]] == [5.0, 5.5]


def test_an_override_for_another_month_or_a_host_root_is_refused(tmp_path):
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"monthly_limit_overrides": {"2020-01": "5"}}))
    with pytest.raises(drill.DrillRefused):
        drill.run_drill(work_root=tmp_path / "drill", config=SHIPPED, override_file=override)
    with pytest.raises(drill.DrillRefused):
        drill.run_drill(work_root=Path("/var/lib/blueprint/drill"), config=SHIPPED)


def test_main_prints_a_passed_drill_and_removes_its_scratch(capsys):
    assert drill.main(["--config", str(SHIPPED)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "passed" and not Path(value["work_root"]).exists()


NOV_11 = datetime(2026, 11, 11, tzinfo=timezone.utc).timestamp()  # 10 days of November measured


def _forecast(spend, *, now=NOV_11, counted=None, previous=None, limit=Decimal(200)):
    return forecast(now=now, spend_usd=Decimal(spend), counted_usd=Decimal(counted if counted is not None else spend),
                    limit_usd=limit, complete=True, previous=previous)


def test_forecast_reads_a_date_not_this_month_or_unknown():
    soon = _forecast(100)  # $10 a day from $100: $200 ten days later
    assert soon["reads"] == "2026-11-21" and soon["daily_rate_usd"] == 10.0 and soon["alert"] is True
    assert _forecast(30)["reads"] == "not this month" and _forecast(30)["alert"] is False
    assert _forecast(0)["reads"] == "not this month"
    early = _forecast(150, now=datetime(2026, 11, 1, 12, tzinfo=timezone.utc).timestamp())
    assert early["reads"] == "unknown" and early["alert"] is False
    assert _forecast(250)["reads"] == "reached"
    assert _forecast(1, limit=None)["reads"] == "unknown"


def test_forecast_alert_is_raised_at_most_once_a_day():
    first = _forecast(100)
    assert first["alert"] is True and first["alert_raised_on"] == "2026-11-11"
    assert _forecast(100, previous=first)["alert"] is True  # standing, not raised again
    cleared = _forecast(30, previous=first)
    assert cleared["alert"] is False and cleared["alert_raised_on"] == "2026-11-11"
    assert _forecast(100, previous=cleared)["alert"] is False  # not twice the same day
    tomorrow = _forecast(110, now=NOV_11 + 86_400, previous=cleared)
    assert tomorrow["alert"] is True and tomorrow["alert_raised_on"] == "2026-11-12"


def test_forecast_alert_reaches_the_account_alerts():
    row = {"cap_status": "set", "period": "2026-11", "blockers": [], "complete": True, "counted_usd": 100,
           "limit_usd": "200", "alert_threshold_percent": 50, "forecast": _forecast(100)}
    codes = [alert["code"] for alert in account.account_alerts(row)]
    assert codes == ["account_spend_month_50pct", account.FORECAST_CODE]
    assert account.FORECAST_CODE in account.ACCOUNT_WARN_CODES


def test_reconciliation_tolerance_is_the_larger_of_five_dollars_or_ten_percent_and_unknown_is_not_zero():
    providers = {
        "vast": {"status": "observed", "metered_settled_usd": 96.0, "reconciled_month_usd": 100.0},
        "openai": {"status": "observed", "metered_settled_usd": 0.0, "reconciled_month_usd": 4.99},
        "digitalocean": {"status": "observed", "metered_settled_usd": 100.0, "reconciled_month_usd": 89.0},
        "world_labs": {"status": "observed", "metered_settled_usd": 300.0, "reconciled_month_usd": 280.0},
        "gcp": {"status": "unobserved", "metered_settled_usd": 3.0, "reconciled_month_usd": None,
                "reason": "gcp_budget_guard_not_configured"},
        "lambda": {"status": "metered_only", "metered_settled_usd": 2.0, "reconciled_month_usd": None},
    }
    value = reconcile(providers, period="2026-11")
    rows = value["providers"]
    assert rows["vast"]["status"] == "within_tolerance" and rows["vast"]["tolerance_usd"] == 10.0
    assert rows["openai"]["status"] == "within_tolerance" and rows["openai"]["tolerance_usd"] == 5.0
    assert rows["digitalocean"]["status"] == "outside_tolerance"
    assert rows["world_labs"]["status"] == "within_tolerance"
    assert rows["gcp"] == {"ledger_settled_usd": 3.0, "billing_month_usd": None, "status": "unknown",
                           "reason": "gcp_budget_guard_not_configured"}
    assert rows["lambda"]["status"] == "unknown"
    assert value["outside_tolerance"] == ["digitalocean"] and value["unknown"] == ["gcp", "lambda"]
    row = {"cap_status": "set", "period": "2026-11", "blockers": [], "complete": True, "counted_usd": 1,
           "limit_usd": "200", "reconciliation": value}
    alerts = account.account_alerts(row)
    assert alerts == [{"code": account.RECONCILIATION_CODE, "provider": "account", "window": "utc_month:2026-11",
                       "reason": "digitalocean"}]
    assert account.RECONCILIATION_CODE not in account.ACCOUNT_PAGE_CODES
