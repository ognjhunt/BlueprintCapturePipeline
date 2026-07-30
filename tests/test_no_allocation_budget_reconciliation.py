from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.no_allocation_budget_reconciliation import (
    reconcile_no_allocation_watchdog_budget,
)
from blueprint_pipeline.production_gpu_campaign_budget import ProductionGpuCampaignBudget


RESERVATION_ID = "current-reference-no-allocation-attempt"


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    ledger_path = tmp_path / "ledger.json"
    budget = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=3.986999,
        initial_used_gpu_seconds=8_445,
        combined_gpu_wall_cap_seconds=36_000,
    )
    budget.reserve(
        reservation_id=RESERVATION_ID,
        gpu_seconds=14_400,
        max_hourly_rate_usd=0.75,
    )
    watchdog = _write(
        tmp_path / "watchdog.json",
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "provider_terminal_budget_reservation_exceeded",
            "provider": "vast",
            "completed_at": "2026-07-30T20:00:00+00:00",
            "provider_absence_confirmed": True,
            "control_plane_terminal": True,
            "provider_mutations_performed": 0,
            "recorded_vast_instance": {"status": "not_recorded", "required": False},
            "pod_pending_teardown_close": {"status": "cancelled_no_allocation"},
            "provider_lane_terminal_release": {"status": "released"},
            "campaign_budget_settlement": {
                "status": "retained_open_budget_breach",
                "elapsed_gpu_seconds": 14_401,
                "reserved_gpu_seconds": 14_400,
            },
        },
    )
    provider_zero = _write(
        tmp_path / "provider-zero.json",
        {
            "schema_version": "gpu_spend_guard.v1",
            "status": "passed",
            "generated_at": "2026-07-30T20:00:01+00:00",
            "live_instance_count": 0,
            "total_burn_per_hour_usd": 0,
            "inventory_results": [{"provider": "vast", "status": "succeeded", "row_count": 0}],
        },
    )
    return ledger_path, watchdog, provider_zero


def test_reconciles_full_reservation_only_after_terminal_zero(tmp_path: Path) -> None:
    ledger, watchdog, provider_zero = _fixtures(tmp_path)
    output = tmp_path / "reconciliation.json"
    result = reconcile_no_allocation_watchdog_budget(
        watchdog_evidence=watchdog,
        provider_zero_snapshot=provider_zero,
        campaign_budget_ledger=ledger,
        reservation_id=RESERVATION_ID,
        output_path=output,
    )
    assert result["status"] == "settled_conservative_full_reservation"
    assert result["actual_attributable_provider_spend_usd"] == 0
    assert result["conservative_charged_gpu_seconds"] == 14_400
    assert result["conservative_charged_usd"] == 3.0
    state = json.loads(ledger.read_text(encoding="utf-8"))
    assert state["open_reservation_count"] == 0
    assert state["committed_gpu_seconds"] == 22_845
    assert state["committed_usd"] == 6.986999
    assert output.is_file()


def test_rejects_provider_zero_snapshot_that_predates_terminal(tmp_path: Path) -> None:
    ledger, watchdog, provider_zero = _fixtures(tmp_path)
    value = json.loads(provider_zero.read_text(encoding="utf-8"))
    value["generated_at"] = "2026-07-30T19:59:59+00:00"
    provider_zero.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="provider_zero_snapshot_predates"):
        reconcile_no_allocation_watchdog_budget(
            watchdog_evidence=watchdog,
            provider_zero_snapshot=provider_zero,
            campaign_budget_ledger=ledger,
            reservation_id=RESERVATION_ID,
            output_path=tmp_path / "reconciliation.json",
        )
    state = json.loads(ledger.read_text(encoding="utf-8"))
    assert state["open_reservation_count"] == 1
