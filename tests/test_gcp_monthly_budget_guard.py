"""No-network monthly limit, native shutoff and paid-boundary regressions."""
import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from blueprint_pipeline import gcp_monthly_budget_guard as guard
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION, PaidResourceAdmissionBlocked,
    build_paid_lane_admission, require_paid_resource_admission,
    require_paid_resource_admission_grant,
)

NOW = datetime.now(timezone.utc)
LOCAL = NOW.astimezone(ZoneInfo("America/Los_Angeles"))
INTERVAL = LOCAL.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


@pytest.fixture
def policy(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "project_id": "blueprint-test", "project_number": "123456",
        "billing_account_id": "ABCDEF-123456-ABCDEF",
        "budget_id": "11111111-1111-1111-1111-111111111111", "limit_usd": "100",
        "topic": "projects/blueprint-test/topics/monthly-budget",
        "subscription": "projects/blueprint-test/subscriptions/monthly-budget",
        "state_path": str(tmp_path / "state.json"), "report_path": str(tmp_path / "report.json"),
        "billing_shutdown_authorized": False,
    }))
    monkeypatch.setenv(guard.CONFIG_ENV, str(path))
    monkeypatch.setenv(guard.REPORT_ENV, str(tmp_path / "report.json"))
    return guard.load_policy(path)


def message(policy, cost="10", *, published=NOW, interval=INTERVAL):
    payload = {"currencyCode": "USD", "costAmount": cost, "budgetAmount": "100",
               "budgetAmountType": "SPECIFIED_AMOUNT", "costIntervalStart": interval.isoformat()}
    return {"ackId": "fixture-ack", "message": {
        "data": base64.b64encode(json.dumps(payload).encode()).decode(),
        "publishTime": published.isoformat(), "attributes": {
            "budgetId": policy["budget_id"], "billingAccountId": policy["billing_account_id"], "schemaVersion": "1.0"}}}


class Client:
    def __init__(self, policy, cost="10", *, enabled=True):
        self.policy, self.enabled = policy, enabled
        self.messages = [message(policy, cost)]
        self.calls = []
        self.budget = {
            "name": f"billingAccounts/{policy['billing_account_id']}/budgets/{policy['budget_id']}",
            "budgetFilter": {"projects": ["projects/123456"], "calendarPeriod": "MONTH"},
            "amount": {"specifiedAmount": {"units": "100", "currencyCode": "USD"}},
            "notificationsRule": {"pubsubTopic": policy["topic"]},
        }

    def inventory(self, project):
        assert project == self.policy["project_id"]
        return {"api_confirmed": True, "resources": [{"name": "cpu-sandbox", "status": "TERMINATED"}]}

    def call(self, method, service, path, body=None):
        self.calls.append((method, service, path, body))
        if service == "billingbudgets":
            return self.budget
        if service == "pubsub":
            return {"receivedMessages": self.messages} if path.endswith(":pull") else {}
        if method == "PUT":
            assert body == {"billingAccountName": ""}
            self.enabled = False
        return {"projectId": self.policy["project_id"], "billingEnabled": self.enabled,
                "billingAccountName": f"billingAccounts/{self.policy['billing_account_id']}" if self.enabled else ""}


def test_fresh_under_budget_report_admits_and_records_cpu_inventory(policy):
    result = guard.run_guard(policy, client=Client(policy), now=NOW)
    assert result["admission_allowed"] is True
    assert result["inventory"]["resources"][0]["name"] == "cpu-sandbox"
    assert guard.admission_blockers(now=NOW) == []


@pytest.mark.parametrize("cost", ["100", "105.28"])
def test_exhausted_budget_disconnects_only_bound_project_and_verifies_readback(policy, cost):
    policy["billing_shutdown_authorized"] = True
    client = Client(policy, cost)
    result = guard.run_guard(policy, client=client, now=NOW)
    assert result["admission_allowed"] is False
    assert result["native_cutoff_verified"] is True
    writes = [call for call in client.calls if call[0] == "PUT"]
    assert writes == [("PUT", "cloudbilling", "/v1/projects/blueprint-test/billingInfo", {"billingAccountName": ""})]
    assert client.calls[-2][0:2] == ("GET", "cloudbilling")


def test_activation_is_separate_from_preview_and_cannot_reenable_billing(policy):
    client = Client(policy, "105")
    result = guard.run_guard(policy, client=client, now=NOW)
    assert "gcp_billing_shutdown_activation_required" in result["blockers"]
    assert not any(call[0] == "PUT" for call in client.calls)
    client = Client(policy, enabled=False)
    result = guard.run_guard(policy, client=client, now=NOW)
    assert not result["admission_allowed"]
    assert not any(call[0] == "PUT" for call in client.calls)


def test_out_of_order_lower_cost_cannot_clear_monthly_high_water(policy):
    first = guard.run_guard(policy, client=Client(policy, "110"), now=NOW)
    second = guard.run_guard(policy, client=Client(policy, "20"), now=NOW)
    assert first["observed_cost_usd"] == second["observed_cost_usd"] == "110"
    assert not second["admission_allowed"]


def test_month_rollover_waits_for_current_provider_observation(policy):
    guard.run_guard(policy, client=Client(policy, "110"), now=NOW)
    client = Client(policy, "20")
    next_month = (LOCAL.replace(day=28) + timedelta(days=5)).replace(day=2)
    result = guard.run_guard(policy, client=client, now=next_month)
    assert result["observed_cost_usd"] is None
    assert "gcp_monthly_cost_missing_or_stale" in result["blockers"]


@pytest.mark.parametrize("kind", ["missing", "stale", "future", "bad_currency", "mid_month", "nonfinite"])
def test_missing_or_invalid_cost_evidence_never_admits(policy, kind):
    client = Client(policy)
    if kind == "missing":
        client.messages = []
    elif kind in ("stale", "future"):
        client.messages = [message(policy, published=NOW + timedelta(hours=-25 if kind == "stale" else 1))]
    elif kind == "mid_month":
        client.messages = [message(policy, interval=INTERVAL + timedelta(days=1))]
    else:
        payload = json.loads(base64.b64decode(client.messages[0]["message"]["data"]))
        payload["currencyCode" if kind == "bad_currency" else "costAmount"] = "EUR" if kind == "bad_currency" else "NaN"
        client.messages[0]["message"]["data"] = base64.b64encode(json.dumps(payload).encode()).decode()
    assert guard.run_guard(policy, client=client, now=NOW)["admission_allowed"] is False


@pytest.mark.parametrize("mutation", ["project", "services", "limit", "topic"])
def test_live_budget_scope_drift_refuses_before_native_shutdown(policy, mutation):
    policy["billing_shutdown_authorized"] = True
    client = Client(policy, "110")
    if mutation == "project":
        client.budget["budgetFilter"]["projects"] = ["projects/other-project"]
    elif mutation == "services":
        client.budget["budgetFilter"]["services"] = ["services/one-only"]
    elif mutation == "limit":
        client.budget["amount"]["specifiedAmount"]["units"] = "101"
    else:
        client.budget["notificationsRule"]["pubsubTopic"] += "-other"
    result = guard.run_guard(policy, client=client, now=NOW)
    assert not result["admission_allowed"]
    assert not any(call[0] == "PUT" for call in client.calls)


def test_missing_permissions_and_corrupt_state_preserve_accounting(policy):
    state = Path(policy["state_path"])
    state.write_text("not-json: preserve")
    original = state.read_bytes()
    assert not guard.run_guard(policy, client=Client(policy), now=NOW)["admission_allowed"]
    assert state.read_bytes() == original


def test_seed_can_refuse_but_never_grant_permission(policy):
    policy["over_budget_seed"] = {"period": guard._period(NOW), "observed_cost_usd": "105.28",
                                 "source_sha256": "a" * 64, "observed_at": NOW.isoformat()}
    client = Client(policy)
    client.messages = []
    result = guard.run_guard(policy, client=client, now=NOW)
    assert not result["admission_allowed"] and result["observed_cost_usd"] == "105.28"
    policy["over_budget_seed"]["observed_cost_usd"] = "10"
    assert not guard.run_guard(policy, client=client, now=NOW)["admission_allowed"]


def test_paid_boundary_rechecks_an_already_issued_grant(policy):
    guard.run_guard(policy, client=Client(policy), now=NOW)
    grant = require_paid_resource_admission(
        build_paid_lane_admission(resource_class="provider_reconstruction_api"),
        resource_class="provider_reconstruction_api", expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    guard.run_guard(policy, client=Client(policy, "110"), now=NOW)
    with pytest.raises(PaidResourceAdmissionBlocked, match="gcp_monthly_budget_admission_blocked"):
        require_paid_resource_admission_grant(grant, resource_class="provider_reconstruction_api")


def test_expired_or_changed_policy_report_blocks_paid_admission(policy):
    guard.run_guard(policy, client=Client(policy), now=NOW)
    assert guard.admission_blockers(now=NOW + timedelta(minutes=6))
    assert guard.admission_blockers(project_id="different-project", now=NOW)
    path = Path(policy["report_path"])
    report = json.loads(path.read_text())
    report["limit_usd"] = "1000"
    path.write_text(json.dumps(report))
    assert guard.admission_blockers(now=NOW)


def test_native_api_failure_does_not_claim_cutoff_or_admission(policy):
    class Refusal(Client):
        def call(self, method, service, path, body=None):
            if method == "PUT":
                raise RuntimeError("permission_denied")
            return super().call(method, service, path, body)
    policy["billing_shutdown_authorized"] = True
    result = guard.run_guard(policy, client=Refusal(policy, "105"), now=NOW)
    assert not result["admission_allowed"] and not result["native_cutoff_verified"]


def test_overlapping_guard_cannot_race_monthly_accounting(policy):
    import fcntl

    lock = Path(policy["state_path"]).with_suffix(".lock")
    with lock.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = guard.run_guard(policy, client=Client(policy), now=NOW)
        assert result["blockers"] == ["gcp_budget_guard_already_running"]
        assert not Path(policy["state_path"]).exists()


def test_production_reconstruction_uses_project_guard_without_per_run_override(policy, monkeypatch):
    guard.run_guard(policy, client=Client(policy, "105"), now=NOW)
    monkeypatch.delenv(guard.REPORT_ENV)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", policy["project_id"])
    with pytest.raises(PaidResourceAdmissionBlocked, match="gcp_monthly_budget_admission_blocked"):
        require_paid_resource_admission(build_paid_lane_admission(resource_class="provider_reconstruction_api"),
            resource_class="provider_reconstruction_api", expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    # Other providers' allowances are not silently replaced by this GCP project cap.
    grant = require_paid_resource_admission(build_paid_lane_admission(resource_class="gpu_canary"),
        resource_class="gpu_canary", expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    assert grant.resource_class == "gpu_canary"


def test_service_preserves_operator_owned_group_accessible_lock(policy, monkeypatch):
    import os

    lock = Path(policy["state_path"]).with_suffix(".lock")
    lock.touch(mode=0o660)
    owner = lock.stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)
    monkeypatch.setattr(os, "fchmod", lambda *args: pytest.fail("service cannot chmod another owner's lock"))
    assert guard.run_guard(policy, client=Client(policy), now=NOW)["admission_allowed"]
