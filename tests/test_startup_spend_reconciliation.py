"""Hermetic tests for startup spend reconciliation and the cumulative ledger."""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline import startup_spend_reconciliation as S


def test_reconciliation_separates_reserved_elapsed_and_actual():
    result = S.build_spend_reconciliation(
        provider="runpod",
        hourly_rate_usd=0.49,
        reserved_seconds=3600,
        elapsed_seconds=1800,
        phase_durations_seconds={"image_pull_no_runtime": 300.0},
    )
    assert result["schema_version"] == S.SCHEMA_VERSION
    assert result["reserved_worst_case_usd"] == pytest.approx(0.49)
    assert result["elapsed_rate_upper_bound_usd"] == pytest.approx(0.245)
    assert result["provider_reported_actual_usd"] is None
    assert result["billing_reconciliation"] == "not_configured"
    assert result["estimate_labeled_actual"] is False
    assert result["phase_durations_seconds"]["image_pull_no_runtime"] == 300.0


def test_actual_spend_requires_provider_billing_api_source():
    refused = S.build_spend_reconciliation(
        provider="runpod",
        hourly_rate_usd=1.0,
        reserved_seconds=0,
        elapsed_seconds=0,
        provider_reported_actual_usd=0.25,
        provider_reported_source="rate_times_age_estimate",
    )
    assert refused["provider_reported_actual_usd"] is None
    assert refused["provider_reported_actual_refused_reason"] == (
        "provider_reported_actual_requires_provider_billing_api_source"
    )
    assert refused["billing_reconciliation"] == "not_configured"

    accepted = S.build_spend_reconciliation(
        provider="runpod",
        hourly_rate_usd=1.0,
        reserved_seconds=0,
        elapsed_seconds=0,
        provider_reported_actual_usd=0.25,
        provider_reported_source=S.PROVIDER_BILLING_API_SOURCE,
    )
    assert accepted["provider_reported_actual_usd"] == 0.25
    assert accepted["billing_reconciliation"] == "provider_api"


def test_stopped_disk_cost_included_in_upper_bound():
    result = S.build_spend_reconciliation(
        provider="runpod",
        hourly_rate_usd=0.0,
        reserved_seconds=0,
        elapsed_seconds=0,
        stopped_disk_usd_per_hour=0.10,
        stopped_disk_seconds=7200,
    )
    assert result["standing_stopped_disk_usd_per_hour"] == 0.10
    assert result["elapsed_rate_upper_bound_usd"] == pytest.approx(0.20)


def test_runpod_cost_components_are_accounted_separately() -> None:
    result = S.build_spend_reconciliation(
        provider="runpod",
        hourly_rate_usd=1.0,
        reserved_seconds=3600,
        elapsed_seconds=3600,
        stopped_disk_seconds=3600,
        container_disk_usd_per_hour=0.1,
        persistent_volume_usd_per_hour=0.2,
        network_volume_usd_per_hour=0.3,
    )
    assert result["cost_component_upper_bounds_usd"] == {
        "compute_usd": 1.0,
        "container_disk_usd": 0.1,
        "persistent_volume_usd": 0.4,
        "network_volume_usd": 0.6,
    }
    assert result["elapsed_rate_upper_bound_usd"] == pytest.approx(2.1)


def test_negative_and_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        S.build_spend_reconciliation(
            provider="runpod", hourly_rate_usd=-1, reserved_seconds=0,
            elapsed_seconds=0,
        )
    with pytest.raises(ValueError):
        S.build_spend_reconciliation(
            provider="runpod", hourly_rate_usd="rate", reserved_seconds=0,
            elapsed_seconds=0,
        )


def test_ledger_admits_until_cap_and_counts_failed_attempts(tmp_path):
    ledger = S.CumulativeSpendLedger(tmp_path / "ledger.json", total_cap_usd=1.0)
    ledger.admit(attempt_id="a1", reserved_usd=0.4)
    ledger.settle(attempt_id="a1", elapsed_upper_bound_usd=0.3, outcome="failed")
    ledger.admit(attempt_id="a2", reserved_usd=0.4)
    # 0.3 settled (failed attempt still counts) + 0.4 open = 0.7 committed.
    assert ledger.committed_usd() == pytest.approx(0.7)
    with pytest.raises(S.SpendCapExceeded) as excinfo:
        ledger.admit(attempt_id="a3", reserved_usd=0.5)
    assert excinfo.value.admission["blocker"] == "startup_cumulative_spend_cap_exceeded"
    snapshot = ledger.snapshot()
    assert snapshot["includes_failed_attempts"] is True
    assert snapshot["attempt_count"] == 2


def test_ledger_persists_and_reloads_across_processes(tmp_path):
    path = tmp_path / "ledger.json"
    first = S.CumulativeSpendLedger(path, total_cap_usd=2.0)
    first.admit(attempt_id="a1", reserved_usd=1.5)
    second = S.CumulativeSpendLedger(path, total_cap_usd=2.0)
    assert second.committed_usd() == pytest.approx(1.5)
    with pytest.raises(S.SpendCapExceeded):
        second.admit(attempt_id="a2", reserved_usd=1.0)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == S.LEDGER_SCHEMA_VERSION
    assert payload["remaining_usd"] == pytest.approx(0.5)


def test_ledger_settle_unknown_attempt_rejected(tmp_path):
    ledger = S.CumulativeSpendLedger(tmp_path / "ledger.json", total_cap_usd=1.0)
    with pytest.raises(ValueError):
        ledger.settle(attempt_id="ghost", elapsed_upper_bound_usd=0.1, outcome="x")
