from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.common import write_json
from blueprint_pipeline.paid_resource_allocator import _reserve_successor_campaign_budget
from blueprint_pipeline.policy_ranking_successor_campaign_budget_reconciliation import (
    PRESERVED_SETTLEMENT_NAME,
    reconcile_successor_campaign_budget_after_watchdog,
)


def _stage_job(tmp_path: Path, *, watchdog_status: str = "provider_terminal") -> Path:
    job = tmp_path / "job"
    job.mkdir()
    budget, reservation = _reserve_successor_campaign_budget(
        job_dir=job,
        authorization_path=tmp_path / "authorization.json",
        expected_source_commit="a" * 40,
        ledger_path=job / "production_campaign_budget_ledger.json",
        initial_spent_usd=2.0,
        initial_used_gpu_seconds=100,
        total_spend_cap_usd=20.0,
        wall_cap_seconds=72_000,
        reservation_seconds=4_800,
        max_hourly_rate_usd=2.05,
    )
    assert budget is not None
    write_json(
        job / "adapter_output.json",
        {
            "status": "completed",
            "provider_mutations_performed": 1,
            "continuing_spend_from_this_run": False,
            "runtime_seconds": 354.2,
            "estimated_gpu_cost_usd": 0.171192,
        },
    )
    write_json(
        job / "successor_campaign_budget_settlement.json",
        {
            "schema_version": "policy_ranking_successor_campaign_budget_settlement.v1",
            "status": "open_reservation_retained_fail_closed",
            "reservation_id": reservation["reservation"]["reservation_id"],
            "reason": "provider_terminal_or_zero_not_proven",
        },
    )
    watchdog_dir = job / "independent_vast_watchdog"
    watchdog_dir.mkdir()
    write_json(
        watchdog_dir / "groot_oscar_runpod_canary_watchdog.json",
        {
            "status": watchdog_status,
            "provider": "vast",
            "provider_absence_confirmed": watchdog_status == "provider_terminal",
            "provider_mutations_performed": 0,
            "global_inventory_within_authorized_residual_limit": True,
            "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
            "recorded_vast_instance": {"status": "recorded", "instance_id": "123"},
            "recorded_vast_instance_teardown": {
                "instance_id": "123",
                "provider_absence_confirmed": True,
            },
        },
    )
    write_json(
        job / "vast_teardown_manifest.json",
        {
            "vast_instance_ids": [123],
            "continuing_spend_from_this_run": False,
            "runner_gpu_teardown_completed": True,
        },
    )
    return job


def test_reconciles_open_reservation_after_late_watchdog_terminal(tmp_path: Path) -> None:
    job = _stage_job(tmp_path)

    result = reconcile_successor_campaign_budget_after_watchdog(job_dir=job)

    assert result["status"] == "settled"
    assert result["charged_gpu_seconds"] == 355
    assert result["charged_usd"] == 0.171192
    assert result["provider_mutations_performed"] == 0
    assert result["global_provider_zero_claimed"] is False
    assert (job / PRESERVED_SETTLEMENT_NAME).is_file()
    ledger = json.loads((job / "production_campaign_budget_ledger.json").read_text())
    assert ledger["open_reservation_count"] == 0
    assert ledger["committed_gpu_seconds"] == 455
    assert ledger["committed_usd"] == 2.171192


def test_reconciliation_fails_closed_without_terminal_watchdog(tmp_path: Path) -> None:
    job = _stage_job(tmp_path, watchdog_status="armed")

    result = reconcile_successor_campaign_budget_after_watchdog(job_dir=job)

    assert result["status"] == "blocked"
    assert result["blockers"] == ["successor_terminal_watchdog_proof_invalid"]
    ledger = json.loads((job / "production_campaign_budget_ledger.json").read_text())
    assert ledger["open_reservation_count"] == 1
    assert not (job / PRESERVED_SETTLEMENT_NAME).exists()


def test_reconciliation_is_idempotent_after_success(tmp_path: Path) -> None:
    job = _stage_job(tmp_path)
    first = reconcile_successor_campaign_budget_after_watchdog(job_dir=job)

    second = reconcile_successor_campaign_budget_after_watchdog(job_dir=job)

    assert second == first
    ledger = json.loads((job / "production_campaign_budget_ledger.json").read_text())
    assert len(ledger["reservations"]) == 1
