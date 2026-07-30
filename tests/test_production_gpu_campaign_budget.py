from __future__ import annotations

import concurrent.futures

import pytest

from blueprint_pipeline.production_gpu_campaign_budget import (
    CampaignBudgetExceeded,
    ProductionGpuCampaignBudget,
)


def _ledger(tmp_path, *, used: int = 8_815, spent: float = 3.0):
    return ProductionGpuCampaignBudget(
        tmp_path / "campaign-budget.json",
        initial_spent_usd=spent,
        initial_used_gpu_seconds=used,
        combined_gpu_wall_cap_seconds=10_980,
    )


def test_wall_time_reservation_fails_closed_against_campaign_total(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(CampaignBudgetExceeded) as excinfo:
        ledger.reserve(
            reservation_id="qualification-too-long",
            gpu_seconds=2_166,
            max_hourly_rate_usd=1.0,
        )
    assert excinfo.value.admission["blocker"] == "campaign_gpu_wall_time_cap_exceeded"
    assert ledger.snapshot()["open_reservation_count"] == 0


def test_open_reservation_retains_full_worst_case_until_settled(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    reservation = ledger.reserve(
        reservation_id="qualification-one",
        gpu_seconds=2_000,
        max_hourly_rate_usd=1.0,
    )
    assert reservation["reserved_usd"] == pytest.approx(0.555556)
    assert ledger.snapshot()["remaining_gpu_seconds"] == 165

    ledger.settle(
        reservation_id="qualification-one",
        charged_gpu_seconds=1_000,
        charged_usd=0.25,
        outcome="provider_teardown_proven",
    )
    snapshot = ledger.snapshot()
    assert snapshot["open_reservation_count"] == 0
    assert snapshot["remaining_gpu_seconds"] == 1_165


def test_existing_ledger_identity_cannot_be_reset_to_recover_budget(tmp_path) -> None:
    _ledger(tmp_path).reserve(
        reservation_id="qualification-one",
        gpu_seconds=100,
        max_hourly_rate_usd=1.0,
    )
    with pytest.raises(ValueError, match="identity_mismatch"):
        _ledger(tmp_path, used=0)


def test_duplicate_reservation_is_idempotent_but_conflict_is_rejected(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    first = ledger.reserve(
        reservation_id="qualification-one", gpu_seconds=100, max_hourly_rate_usd=1.0
    )
    assert (
        ledger.reserve(reservation_id="qualification-one", gpu_seconds=100, max_hourly_rate_usd=1.0)
        == first
    )
    with pytest.raises(ValueError, match="reservation_id_conflict"):
        ledger.reserve(reservation_id="qualification-one", gpu_seconds=101, max_hourly_rate_usd=1.0)


def test_settled_reservation_id_cannot_be_reused(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.reserve(reservation_id="qualification-one", gpu_seconds=100, max_hourly_rate_usd=1.0)
    ledger.settle(
        reservation_id="qualification-one",
        charged_gpu_seconds=0,
        charged_usd=0,
        outcome="no_allocation",
    )
    with pytest.raises(ValueError, match="reservation_id_already_settled"):
        ledger.reserve(
            reservation_id="qualification-one",
            gpu_seconds=100,
            max_hourly_rate_usd=1.0,
        )


def test_concurrent_reservations_cannot_oversubscribe_wall_cap(tmp_path) -> None:
    ledger = _ledger(tmp_path, used=10_000)

    def reserve(index: int) -> bool:
        try:
            ledger.reserve(
                reservation_id=f"qualification-{index}",
                gpu_seconds=600,
                max_hourly_rate_usd=1.0,
            )
        except CampaignBudgetExceeded:
            return False
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        admitted = list(executor.map(reserve, range(2)))
    assert sorted(admitted) == [False, True]
    assert ledger.snapshot()["committed_gpu_seconds"] == 10_600


def test_staged_canary_then_campaign_fits_reduced_combined_plan(tmp_path) -> None:
    ledger = ProductionGpuCampaignBudget(
        tmp_path / "staged-campaign-budget.json",
        initial_spent_usd=11.57,
        initial_used_gpu_seconds=11_619,
        combined_gpu_wall_cap_seconds=16_800,
    )
    ledger.reserve(
        reservation_id="startup-canary-stage",
        gpu_seconds=1_200,
        max_hourly_rate_usd=1.99,
    )
    ledger.settle(
        reservation_id="startup-canary-stage",
        charged_gpu_seconds=1_200,
        charged_usd=round(1.99 * 1_200 / 3_600, 6),
        outcome="canary_terminal",
    )
    campaign = ledger.reserve(
        reservation_id="full-campaign-stage",
        gpu_seconds=3_900,
        max_hourly_rate_usd=1.99,
    )
    assert campaign["status"] == "open"
    snapshot = ledger.snapshot()
    assert snapshot["committed_gpu_seconds"] == 16_719
    assert snapshot["remaining_gpu_seconds"] == 81


def test_ordinary_21000_second_plan_and_persistent_authority_are_bounded(
    tmp_path,
) -> None:
    ledger = ProductionGpuCampaignBudget(
        tmp_path / "authorized-21000-second-budget.json",
        initial_spent_usd=14.557003,
        initial_used_gpu_seconds=15_624,
        combined_gpu_wall_cap_seconds=21_000,
    )
    strict = ledger.reserve(
        reservation_id="strict-policy-smoke",
        gpu_seconds=480,
        max_hourly_rate_usd=3.50,
    )
    assert strict["reserved_gpu_seconds"] == 480
    with pytest.raises(CampaignBudgetExceeded):
        ledger.reserve(
            reservation_id="single-cap-sized-job",
            gpu_seconds=21_000,
            max_hourly_rate_usd=3.50,
        )

    with pytest.raises(ValueError, match="wall_cap_exceeds_authorization"):
        ProductionGpuCampaignBudget(
            tmp_path / "over-authorized-budget.json",
            initial_spent_usd=0,
            initial_used_gpu_seconds=0,
            combined_gpu_wall_cap_seconds=72_001,
        )

    persistent = ProductionGpuCampaignBudget(
        tmp_path / "persistent-authorized-budget.json",
        initial_spent_usd=14.557003,
        initial_used_gpu_seconds=15_624,
        combined_gpu_wall_cap_seconds=36_000,
    )
    reservation = persistent.reserve(
        reservation_id="persistent-policy-wam-loop",
        gpu_seconds=18_600,
        max_hourly_rate_usd=1.0,
    )
    assert reservation["reserved_gpu_seconds"] == 18_600
    assert persistent.snapshot()["remaining_gpu_seconds"] == 1_776


def test_conservative_no_allocation_charge_still_allows_one_finite_retry(
    tmp_path,
) -> None:
    ledger = ProductionGpuCampaignBudget(
        tmp_path / "real-policy-successor-budget.json",
        initial_spent_usd=6.986999,
        initial_used_gpu_seconds=22_845,
        combined_gpu_wall_cap_seconds=72_000,
    )
    retry = ledger.reserve(
        reservation_id="current-reference-policy-identity-retry",
        gpu_seconds=14_400,
        max_hourly_rate_usd=0.75,
    )
    assert retry["reserved_usd"] == 3.0
    snapshot = ledger.snapshot()
    assert snapshot["committed_gpu_seconds"] == 37_245
    assert snapshot["remaining_gpu_seconds"] == 34_755
    assert snapshot["committed_usd"] == 9.986999
    assert snapshot["remaining_usd"] == 10.013001
