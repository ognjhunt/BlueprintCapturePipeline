import json
import threading
from pathlib import Path

import pytest

from blueprint_pipeline.paid_campaign_spend_budget import (
    PaidCampaignBudgetExceeded,
    PaidCampaignSpendBudget,
)


def _budget(
    path: Path,
    *,
    initial: float = 2.0,
    cap: float = 5.0,
    authority: str = "user-authority-usd-5",
) -> PaidCampaignSpendBudget:
    return PaidCampaignSpendBudget(
        path,
        campaign_id="arm-decision-proof-v1-test",
        authority_id=authority,
        initial_spent_usd=initial,
        total_spend_cap_usd=cap,
        initial_spend_basis="digest-bound retained provider receipts",
    )


def test_campaign_budget_reserves_worst_case_and_settles_retained_cost(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path / "campaign.json")

    preview = budget.preview(reservation_id="construction-cell-0001", max_spend_usd=1.0)
    assert preview["admitted"] is True
    assert budget.snapshot()["committed_usd"] == 2.0

    reservation = budget.reserve(
        reservation_id="construction-cell-0001",
        reservation_owner_id="controller-one",
        max_spend_usd=1.0,
    )
    assert reservation["reserved_usd"] == 1.0
    assert budget.snapshot()["committed_usd"] == 3.0

    settlement = budget.settle(
        reservation_id="construction-cell-0001",
        reservation_owner_id="controller-one",
        charged_usd=0.125,
        cost_basis="provider_estimated_cost_after_teardown",
        outcome="native construction blocked",
    )
    assert settlement["charged_usd"] == 0.125
    snapshot = budget.snapshot()
    assert snapshot["committed_usd"] == 2.125
    assert snapshot["remaining_usd"] == 2.875
    assert snapshot["open_reservation_count"] == 0
    assert snapshot["snapshot_digest"].startswith("sha256:")
    assert (tmp_path / "campaign.json").stat().st_mode & 0o777 == 0o600


def test_campaign_budget_retains_observed_overrun_and_rejects_new_spend(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path / "campaign.json", initial=12.434661, cap=12.0)
    snapshot = budget.snapshot()
    assert snapshot["budget_status"] == "exhausted"
    assert snapshot["cap_overrun_usd"] == 0.434661
    preview = budget.preview(reservation_id="construction-cell-0001", max_spend_usd=0.01)
    assert preview == {
        "schema_version": "paid_campaign_spend_admission.v1",
        "campaign_id": "arm-decision-proof-v1-test",
        "authority_id": "user-authority-usd-5",
        "reservation_id": "construction-cell-0001",
        "requested_max_spend_usd": 0.01,
        "committed_usd_before": 12.434661,
        "remaining_usd_before": 0.0,
        "cap_overrun_usd_before": 0.434661,
        "total_spend_cap_usd": 12.0,
        "admitted": False,
        "blocker": "paid_campaign_total_spend_cap_exceeded",
    }
    with pytest.raises(PaidCampaignBudgetExceeded) as excinfo:
        budget.reserve(
            reservation_id="construction-cell-0001",
            reservation_owner_id="controller-one",
            max_spend_usd=0.01,
        )
    assert excinfo.value.admission["admitted"] is False
    assert budget.snapshot()["reservations"] == []


def test_campaign_budget_serializes_competing_controller_reservations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.json"
    _budget(path, initial=0.0, cap=1.0)
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def reserve(reservation_id: str) -> None:
        controller = _budget(path, initial=0.0, cap=1.0)
        barrier.wait()
        try:
            controller.reserve(
                reservation_id=reservation_id,
                reservation_owner_id=f"owner-{reservation_id}",
                max_spend_usd=0.75,
            )
        except PaidCampaignBudgetExceeded:
            outcomes.append("blocked")
        else:
            outcomes.append("reserved")

    threads = [
        threading.Thread(target=reserve, args=("controller-one",)),
        threading.Thread(target=reserve, args=("controller-two",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert sorted(outcomes) == ["blocked", "reserved"]
    snapshot = _budget(path, initial=0.0, cap=1.0).snapshot()
    assert snapshot["committed_usd"] == 0.75
    assert snapshot["open_reservation_count"] == 1


def test_campaign_budget_rejects_open_reservation_reuse_by_another_controller(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path / "campaign.json", initial=0.0, cap=2.0)
    budget.reserve(
        reservation_id="construction-cell-0001",
        reservation_owner_id="controller-one",
        max_spend_usd=1.0,
    )

    with pytest.raises(
        ValueError,
        match="paid_campaign_budget_reservation_owned_by_another_controller",
    ):
        budget.reserve(
            reservation_id="construction-cell-0001",
            reservation_owner_id="controller-two",
            max_spend_usd=1.0,
        )

    assert budget.snapshot()["committed_usd"] == 1.0


@pytest.mark.parametrize("reserved_usd", [-1.0, True, float("inf")])
def test_campaign_budget_rejects_tampered_reservation_amounts(
    tmp_path: Path, reserved_usd: object
) -> None:
    path = tmp_path / "campaign.json"
    budget = _budget(path, initial=0.0, cap=2.0)
    budget.reserve(
        reservation_id="construction-cell-0001",
        reservation_owner_id="controller-one",
        max_spend_usd=1.0,
    )
    state = json.loads(path.read_text())
    state["reservations"][0]["reserved_usd"] = reserved_usd
    path.write_text(json.dumps(state))

    with pytest.raises(
        ValueError, match="paid_campaign_budget_reservations_invalid"
    ):
        budget.snapshot()


def test_campaign_budget_identity_cannot_be_rewritten_by_another_session(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.json"
    _budget(path)
    with pytest.raises(
        ValueError,
        match="paid_campaign_budget_ledger_identity_mismatch:authority_id",
    ):
        _budget(path, authority="different-authority")
    retained = json.loads(path.read_text())
    assert retained["authority_id"] == "user-authority-usd-5"
