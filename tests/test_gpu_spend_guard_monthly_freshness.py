"""Clock replays of native GCP budget freshness across slow fleet reconciliation."""
import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from blueprint_pipeline import gcp_monthly_budget_admission as admission
from blueprint_pipeline import gcp_monthly_budget_guard as monthly
from scripts import gpu_spend_guard as fleet

START = datetime.fromisoformat("2026-10-28T01:08:34+00:00")
FINISH = datetime.fromisoformat("2026-10-28T01:12:59+00:00")
ATTEMPT = datetime.fromisoformat("2026-10-28T01:15:12+00:00")


class NativeClient:
    """Only canned native budget, IAM, inventory and cutoff responses; no network."""

    def __init__(self, policy):
        self.policy, self.enabled, self.messages, self.calls = policy, True, [], []
        self.observe("114.61")

    def observe(self, cost):
        data = {"currencyCode": "USD", "costAmount": cost, "budgetAmount": "100",
                "budgetAmountType": "SPECIFIED_AMOUNT", "costIntervalStart": "2026-10-01T07:00:00Z"}
        self.messages = [{"ackId": "fixture-ack", "message": {
            "publishTime": "2026-10-28T01:01:31.631000Z", "data": base64.b64encode(json.dumps(data).encode()).decode(),
            "attributes": {"budgetId": self.policy["budget_id"], "billingAccountId": self.policy["billing_account_id"],
                           "schemaVersion": "1.0"}}}]

    def inventory(self, project):
        assert project == self.policy["project_id"]
        return {"api_confirmed": True, "resources": [], "live_resource_count": 0}

    def call(self, method, service, path, body=None):
        self.calls.append((method, service, path))
        if service == "billingbudgets":
            return {"name": f"billingAccounts/{self.policy['billing_account_id']}/budgets/{self.policy['budget_id']}",
                    "budgetFilter": {"projects": ["projects/123456"], "calendarPeriod": "MONTH"},
                    "amount": {"specifiedAmount": {"units": "100", "currencyCode": "USD"}},
                    "notificationsRule": {"pubsubTopic": self.policy["topic"]}}
        if service == "pubsub":
            if path.endswith(":pull"):
                messages, self.messages = self.messages, []
                return {"receivedMessages": messages}
            return {}
        if service == "cloudresourcemanager":
            return {"permissions": ["resourcemanager.projects.deleteBillingAssignment"]}
        assert service == "cloudbilling"
        if method == "PUT":
            assert body == {"billingAccountName": ""}
            self.enabled = False
        return {"projectId": self.policy["project_id"], "billingEnabled": self.enabled,
                "billingAccountName": f"billingAccounts/{self.policy['billing_account_id']}"}


@pytest.fixture
def policy(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "project_id": "blueprint-test", "project_number": "123456", "billing_account_id": "ABCDEF-123456-ABCDEF",
        "budget_id": "11111111-1111-1111-1111-111111111111", "limit_usd": "100",
        "monthly_limit_overrides": {"2026-10": "125"}, "billing_shutdown_authorized": True,
        "topic": "projects/blueprint-test/topics/monthly-budget", "subscription": "projects/blueprint-test/subscriptions/monthly-budget",
        "state_path": str(tmp_path / "state.json"), "report_path": str(tmp_path / "report.json"),
    }))
    monkeypatch.setenv(admission.CONFIG_ENV, str(path))
    monkeypatch.setenv(admission.REPORT_ENV, str(tmp_path / "report.json"))
    return admission.load_policy(path)


def test_saved_clock_replay_refuses_stale_report_without_extending_five_minutes(policy):
    client = NativeClient(policy)
    original = monthly.run_guard(policy, client=client, now=START)
    assert original["admission_allowed"] is True
    assert (ATTEMPT - START).total_seconds() == 398
    assert admission.admission_blockers(now=ATTEMPT) == ["gcp_monthly_budget_admission_blocked"]
    second_age = (datetime.fromisoformat("2026-10-28T01:23:24+00:00")
                  - datetime.fromisoformat("2026-10-28T01:15:51.117614+00:00")).total_seconds()
    assert second_age == pytest.approx(452.882386)


@pytest.fixture
def slow_reconciliation(policy, tmp_path, monkeypatch):
    clock, events = {"now": START, "finish": FINISH}, []
    client = NativeClient(policy)

    def refresh():
        events.append(("monthly", clock["now"]))
        return monthly.run_guard(policy, client=client, now=clock["now"])

    def inventory(*, provider, required, **_kwargs):
        events.append((provider, clock["now"]))
        if provider == "gcp":
            clock["now"] = clock["finish"]
        return [], {"provider": provider, "required": required, "status": "succeeded", "row_count": 0,
                    "credential_present": True, "blockers": []}

    monkeypatch.setattr(monthly, "run_configured_guard", refresh)
    monkeypatch.setattr(fleet, "_read_secret", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fleet, "cloud_provider_configured", lambda provider: provider == "gcp")
    monkeypatch.setattr(fleet, "_inventory_query", inventory)
    monkeypatch.setattr(fleet, "_now", lambda: clock["now"].timestamp())
    monkeypatch.setattr(fleet, "list_process_cmdlines", lambda: [])
    monkeypatch.setattr(fleet, "find_protected_pod_ids", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(fleet, "find_expected_serve_pod_ids", lambda *_args, **_kwargs: set())
    return clock, events, client, ["--output-root", str(tmp_path)]


def test_native_refresh_after_slow_reconciliation_restores_real_admission(policy, slow_reconciliation):
    _clock, events, _client, args = slow_reconciliation
    assert fleet.main(args) == 0
    assert [event for event in events if event[0] == "monthly"] == [("monthly", START), ("monthly", FINISH)]
    assert [event[0] for event in events] == ["monthly", "runpod", "vast", "digitalocean", "gcp", "monthly"]
    report = json.loads(Path(policy["report_path"]).read_text())
    assert report["generated_at"] == FINISH.isoformat()
    assert report["observed_cost_usd"] == "114.61"
    assert report["provider_observed_at"] == "2026-10-28T01:01:31.631000+00:00"
    assert report["limit_usd"] == "125" and report["base_limit_usd"] == "100"
    assert admission.admission_blockers(now=ATTEMPT) == []
    assert admission.admission_blockers(now=FINISH + timedelta(seconds=301)) == ["gcp_monthly_budget_admission_blocked"]


@pytest.mark.parametrize("mode", ["dry-run", "reap"])
def test_completion_refresh_also_follows_orphan_handling(policy, slow_reconciliation, monkeypatch, mode):
    clock, events, _client, args = slow_reconciliation
    orphan = fleet.GpuInstance(provider="vast", id="fixture-orphan", name="fixture", state="loading", booted=False,
                               live=True, age_seconds=1000, cost_per_hr=0)
    monkeypatch.setattr(fleet, "collect_instances", lambda **_kwargs: [orphan])
    def terminate(*_args, **_kwargs):
        events.append(("terminate", clock["now"]))
        return {"status": "terminated", "http": 200}
    monkeypatch.setattr(fleet, "terminate_instance", terminate)
    assert fleet.main(args + (["--reap"] if mode == "reap" else [])) == 0
    assert [event[0] for event in events].count("monthly") == 2
    assert events[-1] == ("monthly", FINISH)
    if mode == "reap":
        assert events[-2][0] == "terminate"


def test_failed_completion_refresh_never_rewrites_a_positive_timestamp(policy, slow_reconciliation, monkeypatch, capsys):
    clock, _events, client, args = slow_reconciliation
    def refresh():
        if clock["now"] != START:
            raise RuntimeError("private provider error must not be printed")
        return monthly.run_guard(policy, client=client, now=START)
    monkeypatch.setattr(monthly, "run_configured_guard", refresh)
    assert fleet.main(args) == 0  # Other providers' independent checks still run.
    assert json.loads(Path(policy["report_path"]).read_text())["generated_at"] == START.isoformat()
    assert admission.admission_blockers(now=ATTEMPT) == ["gcp_monthly_budget_admission_blocked"]
    error = capsys.readouterr().err
    assert "RuntimeError" in error and "private provider error" not in error


@pytest.mark.parametrize("refusal", ["exhausted", "new-month"])
def test_completion_refresh_keeps_native_cutoff_and_month_rollover_closed(policy, slow_reconciliation, monkeypatch, refusal):
    clock, _events, client, args = slow_reconciliation
    def refresh():
        if clock["now"] != START and refusal == "exhausted":
            client.observe("130")
        return monthly.run_guard(policy, client=client, now=clock["now"])
    monkeypatch.setattr(monthly, "run_configured_guard", refresh)
    if refusal == "new-month":
        clock["finish"] = datetime(2026, 11, 1, 9, tzinfo=timezone.utc)
    assert fleet.main(args) == 0
    report = json.loads(Path(policy["report_path"]).read_text())
    assert report["admission_allowed"] is False
    assert admission.admission_blockers(now=clock["finish"]) == ["gcp_monthly_budget_admission_blocked"]
    if refusal == "exhausted":
        assert report["observed_cost_usd"] == "130"
        assert report["native_cutoff_verified"] is True and client.enabled is False
    else:
        assert report["period"] == "2026-11" and report["limit_usd"] == "100"
        assert report.get("observed_cost_usd") is None
