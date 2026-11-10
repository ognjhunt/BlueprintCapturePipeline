"""A failed provider billing call no longer erases that provider's last good reading.

Audit, 2026-11-10: one failed Vast billing call dropped Vast from the export, so
the admission lock refused every Vast launch
(``spend_admission_lock_required_provider_billing_missing:vast``) until the
billing API answered again, although the lock accepts billing evidence up to
24 hours old. The scheduled reconciler now keeps the last good reading beside
the totals, with the time it was really read and the failure of the newer call;
the spend guard applies it for as long as that reading is within the 24 hours.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from blueprint_pipeline import cloud_vm_render_providers
from blueprint_pipeline import provider_billing_reconciler as billing
from blueprint_pipeline.provider_billing_reconciler import (
    ProviderBillingReconciliationError,
    reconcile_provider_billing,
)
from blueprint_pipeline.spend_admission_lock import (
    MAX_BILLING_AGE_SECONDS,
    build_spend_admission_lock,
    validate_spend_admission_lock,
)
from scripts import gpu_spend_guard as guard
from tests.test_provider_billing_reconciler import _secrets, _Transport
from tests.test_provider_billing_refresh_interval import _write_guard_report

NOW = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
MISSING = "spend_admission_lock_required_provider_billing_missing:vast"


class _Billing(_Transport):
    """The canned provider responses, with either provider's API failing on demand."""

    def __init__(self) -> None:
        super().__init__()
        self.failing: set[str] = set()
        self.vast_amounts = (4.0, 1.5)

    def __call__(self, request, timeout: float) -> bytes:
        host = urlsplit(request.full_url).netloc
        if (host == "console.vast.ai" and "vast" in self.failing) or (
            host == "api.digitalocean.com" and "digitalocean" in self.failing
        ):
            raise ProviderBillingReconciliationError("provider_billing_request_failed")
        if host == "console.vast.ai":
            self.requests.append((request.full_url, request.headers.get("Authorization", "")))
            first = "after_token" not in request.full_url
            return json.dumps({"success": True, "results": [{"amount": self.vast_amounts[0 if first else 1]}],
                               "next_token": "page-two" if first else None}).encode()
        if host == "api.digitalocean.com" and urlsplit(request.full_url).path.endswith("/balance"):
            # The canned balance is dated in August; date it as the provider would, a little before each call.
            return json.dumps({"generated_at": self.digitalocean_generated_at, "month_to_date_usage": "2.00"}).encode()
        return super().__call__(request, timeout)


class _Host:
    """One control plane: the reconciler unit's export and receipts, read by the guard."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.secrets = _secrets(tmp_path)
        self.export = tmp_path / "gpu_spend_guard" / "provider_billing_export.json"
        self.audit = tmp_path / "gpu_spend_guard" / "billing-audit"
        self.transport = _Billing()

    def reconcile(self, at: datetime, *, carry: float | None = MAX_BILLING_AGE_SECONDS) -> dict:
        self.transport.digitalocean_generated_at = (at - timedelta(minutes=30)).isoformat()
        return reconcile_provider_billing(
            secrets_dir=self.secrets, billing_export_path=self.export, audit_root=self.audit,
            start_at="2026-01-01T00:00:00Z", now=at, transport=self.transport,
            carry_forward_max_age_seconds=carry)

    def written(self) -> dict:
        return json.loads(self.export.read_text())

    def guard(self, at: datetime, *, live_vast: bool = False) -> dict:
        instances = [guard.GpuInstance(provider="vast", id="1", name="screen", state="running", booted=True,
                                       live=True, cost_per_hr=0.4, age_seconds=60.0)] if live_vast else []
        return guard.reconcile_billing_export(billing_export_path=self.export, instances=instances,
                                              now=at.timestamp(), required=True)

    def lock_blockers(self, at: datetime, *, live_vast: bool = False) -> list[str]:
        """What a Vast launch is told by the lock the guard would write at ``at``."""
        inventory = [{"provider": name, "status": "succeeded", "required": True, "credential_present": True,
                      "row_count": 0, "blockers": []} for name in ("vast", "digitalocean")]
        lock = build_spend_admission_lock(
            fleet_budget={"status": "passed", "total_spend_usd": 1.0, "blockers": []},
            billing_reconciliation=self.guard(at, live_vast=live_vast), instances=[], reap_results=[],
            inventory_results=inventory, override_path=None, now=at)
        return validate_spend_admission_lock(lock, now=at, required_provider="vast")


@pytest.fixture
def host(tmp_path: Path) -> _Host:
    return _Host(tmp_path)


def test_a_good_run_writes_the_export_exactly_as_before(host: _Host) -> None:
    result = host.reconcile(NOW)
    assert host.written() == {
        "schema_version": "blueprint.provider_billing_export.v1", "generated_at": NOW.isoformat(),
        "currency": "USD", "scope": "blueprint_beta_100_user_cohort",
        "provider_totals_usd": {"vast": 5.5, "digitalocean": 8.0}}
    assert "carried_forward" not in result
    assert host.lock_blockers(NOW + timedelta(minutes=1)) == []


def test_a_failed_call_after_a_good_one_carries_the_last_reading_marked_with_its_age(host: _Host) -> None:
    host.reconcile(NOW)
    host.transport.failing = {"vast"}
    later = NOW + timedelta(hours=1)
    result = host.reconcile(later)

    export = host.written()
    # The totals hold only what this run read; the carried reading sits beside them, never among them.
    assert export["provider_totals_usd"] == {"digitalocean": 8.0}
    assert export["generated_at"] == later.isoformat()
    assert export["carried_forward"] == {"vast": {
        "total_usd": 5.5, "observed_at": NOW.isoformat(), "age_seconds": 3600.0,
        "failure": "provider_billing_request_failed", "failed_at": later.isoformat()}}
    # The failure is not hidden anywhere: the run, the receipt and the guard's report all name it.
    assert result["optional_provider_failures"]["vast"] == "provider_billing_request_failed"
    assert "vast" in result["uncovered_provider_ids"] and result["covered_provider_ids"] == ["digitalocean"]
    assert result["carried_forward"] == export["carried_forward"]
    receipt = json.loads(Path(result["source_receipt_path"]).read_text())
    assert receipt["provider_totals_usd"] == {"digitalocean": 8.0}  # receipts stay what was really read
    assert "carried_forward" not in receipt
    assert result["daily_spend_snapshot"]["status"] == "written"

    for live in (False, True):
        judged = host.guard(later + timedelta(minutes=1), live_vast=live)
        assert judged["status"] == "reconciled" and judged["blockers"] == []
        assert judged["provider_totals_usd"] == {"digitalocean": 8.0, "vast": 5.5}
        assert judged["carried_forward"] == {"vast": {
            "total_usd": 5.5, "observed_at": NOW.isoformat(), "age_seconds": 3660.0,
            "failure": "provider_billing_request_failed"}}
        assert host.lock_blockers(later + timedelta(minutes=1), live_vast=live) == []


def test_without_the_carried_reading_the_same_failure_refuses_every_vast_launch(host: _Host) -> None:
    """The behaviour the audit found, kept for callers that do not ask for a carry."""
    host.reconcile(NOW, carry=None)
    host.transport.failing = {"vast"}
    later = NOW + timedelta(hours=1)
    host.reconcile(later, carry=None)
    assert "carried_forward" not in host.written()
    assert host.lock_blockers(later + timedelta(minutes=1)) == [MISSING]


def test_the_age_never_restarts_however_many_calls_fail(host: _Host) -> None:
    host.reconcile(NOW)
    host.transport.failing = {"vast"}
    for hours in (1, 2, 23):
        host.reconcile(NOW + timedelta(hours=hours))
        row = host.written()["carried_forward"]["vast"]
        assert row["observed_at"] == NOW.isoformat() and row["age_seconds"] == hours * 3600.0
        assert row["total_usd"] == 5.5
    assert host.lock_blockers(NOW + timedelta(hours=23, minutes=1)) == []


def test_a_failed_call_with_no_earlier_good_reading_carries_nothing(host: _Host) -> None:
    """A provider that was never read is never made to look read."""
    host.transport.failing = {"vast"}
    result = host.reconcile(NOW)
    assert "carried_forward" not in host.written() and "carried_forward" not in result
    assert host.written()["provider_totals_usd"] == {"digitalocean": 8.0}
    after = NOW + timedelta(minutes=1)
    assert host.lock_blockers(after) == [MISSING]
    assert "provider_billing_export_missing:vast" in host.guard(after, live_vast=True)["blockers"]
    # Failing again changes nothing: there is still no reading to carry.
    host.reconcile(NOW + timedelta(hours=1))
    assert "carried_forward" not in host.written()


def test_after_24_hours_without_a_good_reading_the_existing_refusal_applies(host: _Host) -> None:
    host.reconcile(NOW)
    host.transport.failing = {"vast"}
    last = NOW + timedelta(seconds=MAX_BILLING_AGE_SECONDS)
    host.reconcile(last)
    assert host.written()["carried_forward"]["vast"]["age_seconds"] == float(MAX_BILLING_AGE_SECONDS)
    assert host.lock_blockers(last) == []

    # The guard judges the reading's own age when it reads: one second past 24 hours it stands for nothing.
    expired = host.guard(last + timedelta(seconds=1))
    assert expired["status"] == "reconciled"  # the export itself is one second old
    assert expired["provider_totals_usd"] == {"digitalocean": 8.0}
    assert expired["carried_forward_refused"] == ["vast:older_than_accepted_billing_age"]
    assert "carried_forward" not in expired
    assert host.lock_blockers(last + timedelta(seconds=1)) == [MISSING]
    assert "provider_billing_export_missing:vast" in host.guard(last + timedelta(seconds=1), live_vast=True)["blockers"]

    # The next run stops writing it, and it does not come back while the call keeps failing.
    for extra in (1, 3600):
        host.reconcile(last + timedelta(seconds=extra))
        assert "carried_forward" not in host.written()
        assert host.lock_blockers(last + timedelta(seconds=extra + 1)) == [MISSING]


def test_a_newer_successful_reading_always_replaces_the_carried_one(host: _Host) -> None:
    host.reconcile(NOW)
    host.transport.failing = {"vast"}
    host.reconcile(NOW + timedelta(hours=1))
    assert host.written()["carried_forward"]["vast"]["total_usd"] == 5.5

    host.transport.failing = set()
    host.transport.vast_amounts = (6.0, 1.5)
    recovered = NOW + timedelta(hours=2)
    host.reconcile(recovered)
    export = host.written()
    assert export["provider_totals_usd"] == {"vast": 7.5, "digitalocean": 8.0}
    assert "carried_forward" not in export

    # The next failure carries that newer reading, never the older, lower one.
    host.transport.failing = {"vast"}
    host.reconcile(NOW + timedelta(hours=3))
    assert host.written()["carried_forward"]["vast"] == {
        "total_usd": 7.5, "observed_at": recovered.isoformat(), "age_seconds": 3600.0,
        "failure": "provider_billing_request_failed", "failed_at": (NOW + timedelta(hours=3)).isoformat()}


def test_the_guard_never_lets_a_carried_row_stand_in_for_a_provider_it_read_afresh(host: _Host) -> None:
    host.reconcile(NOW)
    export = host.written()
    export["carried_forward"] = {"vast": {"total_usd": 1.0, "observed_at": (NOW - timedelta(hours=2)).isoformat(),
                                          "failure": "provider_billing_request_failed"}}
    host.export.write_text(json.dumps(export))
    host.export.chmod(0o600)
    judged = host.guard(NOW + timedelta(minutes=1))
    assert judged["provider_totals_usd"] == {"vast": 5.5, "digitalocean": 8.0}
    assert "carried_forward" not in judged and "carried_forward_refused" not in judged


@pytest.mark.parametrize("row, refused", [
    ({"total_usd": -1.0, "observed_at": "2026-11-10T08:00:00+00:00"}, "vast:total_invalid"),
    ({"total_usd": True, "observed_at": "2026-11-10T08:00:00+00:00"}, "vast:total_invalid"),
    ({"total_usd": float("inf"), "observed_at": "2026-11-10T08:00:00+00:00"}, "vast:total_invalid"),
    ({"total_usd": "NaN", "observed_at": "2026-11-10T08:00:00+00:00"}, "vast:total_invalid"),
    ({"total_usd": 5.5}, "vast:observed_at_invalid"),
    ({"total_usd": 5.5, "observed_at": "2026-11-10T10:00:00+00:00"}, "vast:observed_at_invalid"),  # an hour ahead
    ({"total_usd": 5.5, "observed_at": "2026-11-09T08:59:59+00:00"}, "vast:older_than_accepted_billing_age"),
])
def test_the_guard_refuses_a_carried_row_it_cannot_trust(host: _Host, row: dict, refused: str) -> None:
    host.transport.failing = {"vast"}
    host.reconcile(NOW)
    export = host.written()
    export["carried_forward"] = {"vast": row, "gcp": {"total_usd": 9.0, "observed_at": NOW.isoformat()},
                                 "runpod": {"total_usd": 9.0, "observed_at": NOW.isoformat()}}
    host.export.write_text(json.dumps(export))
    host.export.chmod(0o600)
    judged = host.guard(NOW)
    # Only the providers the reconciler reads from a live API can be carried at all.
    assert judged["provider_totals_usd"] == {"digitalocean": 8.0}
    assert judged["carried_forward_refused"] == [refused] and "carried_forward" not in judged
    assert host.lock_blockers(NOW) == [MISSING]


def test_the_guard_and_the_reconciler_carry_the_same_providers() -> None:
    assert guard.CARRIED_BILLING_PROVIDERS == billing.CARRY_FORWARD_PROVIDERS == ("vast", "digitalocean")


def test_a_carried_reading_is_taken_only_from_a_trustworthy_previous_export(host: _Host) -> None:
    for hour, damage in enumerate(("writable", "garbage", "bad_total")):
        host.transport.failing = set()
        host.reconcile(NOW + timedelta(hours=hour))
        export = host.written()
        if damage == "writable":
            host.export.chmod(0o620)
        elif damage == "garbage":
            host.export.write_text("{not json")
        else:
            export["provider_totals_usd"]["vast"] = -3.0
            host.export.write_text(json.dumps(export))
        host.transport.failing = {"vast"}
        host.reconcile(NOW + timedelta(hours=hour, minutes=30))
        assert "carried_forward" not in host.written(), damage
        assert host.lock_blockers(NOW + timedelta(hours=hour, minutes=31)) == [MISSING], damage


def test_a_carried_reading_dated_in_the_future_is_not_carried(host: _Host) -> None:
    host.reconcile(NOW + timedelta(hours=5))
    host.transport.failing = {"vast"}
    host.reconcile(NOW)  # the clock went back: the previous reading claims to be from the future
    assert "carried_forward" not in host.written()


def test_digitalocean_is_carried_the_same_way_and_only_the_failed_provider(host: _Host) -> None:
    host.reconcile(NOW)
    host.transport.failing = {"digitalocean"}
    host.reconcile(NOW + timedelta(hours=2))
    export = host.written()
    assert export["provider_totals_usd"] == {"vast": 5.5}
    assert list(export["carried_forward"]) == ["digitalocean"]
    assert export["carried_forward"]["digitalocean"]["total_usd"] == 8.0
    assert host.guard(NOW + timedelta(hours=2))["provider_totals_usd"] == {"vast": 5.5, "digitalocean": 8.0}


def test_a_required_provider_that_fails_still_fails_the_run_and_keeps_the_prior_export(host: _Host) -> None:
    host.reconcile(NOW)
    before = host.export.read_bytes()
    host.transport.failing = {"vast"}
    with pytest.raises(ProviderBillingReconciliationError, match="provider_billing_request_failed"):
        reconcile_provider_billing(
            secrets_dir=host.secrets, billing_export_path=host.export, audit_root=host.audit,
            start_at="2026-01-01T00:00:00Z", now=NOW + timedelta(hours=1), transport=host.transport,
            required_providers=("vast",), carry_forward_max_age_seconds=MAX_BILLING_AGE_SECONDS)
    assert host.export.read_bytes() == before


def test_when_every_call_fails_the_prior_export_stands_untouched(host: _Host) -> None:
    host.reconcile(NOW)
    before = host.export.read_bytes()
    host.transport.failing = {"vast", "digitalocean"}
    with pytest.raises(ProviderBillingReconciliationError, match="provider_billing_no_provider_covered"):
        host.reconcile(NOW + timedelta(hours=1))
    assert host.export.read_bytes() == before
    assert host.lock_blockers(NOW + timedelta(hours=1)) == []


def test_only_the_scheduled_run_asks_for_a_carry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The unit's entry point carries for exactly the age the lock accepts; a library caller gets none."""
    import inspect

    calls: list[dict] = []
    monkeypatch.setattr(billing, "reconcile_provider_billing",
                        lambda **kwargs: calls.append(kwargs) or {"status": "reconciled"})
    monkeypatch.setattr(cloud_vm_render_providers, "GCPRenderProvider",
                        type("NoInventory", (), {"inventory_projects": lambda self: []}))
    assert billing.main(["--secrets-dir", str(tmp_path / "secrets"), "--billing-export", str(tmp_path / "export.json"),
                         "--audit-root", str(tmp_path / "audit"), "--cohort-start-at", "2026-01-01T00:00:00Z"]) == 0
    assert calls[0]["carry_forward_max_age_seconds"] == MAX_BILLING_AGE_SECONDS == 24 * 60 * 60
    assert guard.reconcile_billing_export.__kwdefaults__["max_age_seconds"] == MAX_BILLING_AGE_SECONDS
    default = inspect.signature(reconcile_provider_billing).parameters["carry_forward_max_age_seconds"].default
    assert default is None


def test_a_live_provider_covered_by_a_carried_reading_keeps_the_refresh_cadence(host: _Host) -> None:
    """Without this, a failing billing API with a live instance re-ran the whole refresh every minute."""
    current = datetime.now(timezone.utc)
    host.reconcile(current - timedelta(hours=1))
    host.transport.failing = {"vast"}
    host.reconcile(current - timedelta(minutes=2))
    report = host.tmp / "guard-report.json"
    _write_guard_report(report, live=("vast",))
    retained = billing.retained_fresh_export(host.export, now=current, min_refresh_interval_seconds=600,
                                             spend_guard_report=report)
    assert retained is not None and retained["spend_guard_live_providers"] == ["vast"]
    # A live provider with neither a fresh nor a carried reading still refreshes at once.
    export = host.written()
    del export["carried_forward"]
    host.export.write_text(json.dumps(export))
    host.export.chmod(0o600)
    assert billing.retained_fresh_export(host.export, now=current, min_refresh_interval_seconds=600,
                                         spend_guard_report=report) is None
