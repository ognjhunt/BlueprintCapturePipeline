import json
import os

from blueprint_pipeline.groot_oscar_runpod_watchdog import (
    run_watchdog,
    terminate_canary_resources,
)
from blueprint_pipeline.production_gpu_campaign_budget import ProductionGpuCampaignBudget


class _Provider:
    def __init__(self) -> None:
        self.ids = ["pod-1", "pod-2"]

    def billable_inventory(self, *, name_prefix: str) -> dict:
        assert name_prefix == "blueprint-groot-oscar-canary-attempt-"
        return {
            "api_confirmed": True,
            "live_resource_count": len(self.ids),
            "resources": [{"instance_id": item} for item in self.ids],
        }

    def terminate(self, instance_id: str) -> dict:
        self.ids.remove(instance_id)
        return {"status": "terminated"}


def test_watchdog_reaps_every_name_bound_resource_and_proves_absence() -> None:
    result = terminate_canary_resources(
        provider=_Provider(),
        pod_name_prefix="blueprint-groot-oscar-canary-attempt-",
        armed={"status": "armed"},
    )
    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert [row["instance_id"] for row in result["terminations"]] == [
        "pod-1",
        "pod-2",
    ]


def test_watchdog_inventory_error_returns_secret_safe_unverified_evidence() -> None:
    class Provider:
        def billable_inventory(self, *, name_prefix: str):
            del name_prefix
            raise TimeoutError("secret provider response")

    result = terminate_canary_resources(
        provider=Provider(),
        pod_name_prefix="blueprint-groot-oscar-canary-attempt-",
        armed={"status": "armed"},
    )
    assert result["status"] == "teardown_unverified"
    assert result["provider_absence_confirmed"] is False
    assert result["teardown_error_type"] == "TimeoutError"
    assert "secret provider response" not in json.dumps(result)


def test_watchdog_persists_provider_factory_error(tmp_path) -> None:
    def fail_provider(_name: str):
        raise TimeoutError("secret provider initialization")

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix="blueprint-groot-oscar-canary-attempt-",
        deadline_epoch=10_000_000_000.0,
        provider_factory=fail_provider,
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )
    persisted = json.loads(
        (tmp_path / "groot_oscar_runpod_canary_watchdog.json").read_text(
            encoding="utf-8"
        )
    )
    assert result == persisted
    assert persisted["status"] == "teardown_unverified"
    assert persisted["teardown_error_type"] == "TimeoutError"
    assert "secret provider initialization" not in json.dumps(persisted)


def test_watchdog_closes_pod_record_and_returns_lane_owner(
    tmp_path, monkeypatch
) -> None:
    pending_path = tmp_path / "pending.json"
    pending_path.write_text(
        json.dumps(
            {
                "status": "open",
                "provider": "runpod",
                "lane": "groot_oscar_gpu_canary",
                "resource_kind": "compute_instance",
                "resource_name": "blueprint-groot-oscar-canary-attempt-pod",
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "provider_lane_handoff_receipt.json"
    ledger_path = tmp_path / "campaign-budget.json"
    ledger = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=11.57,
        initial_used_gpu_seconds=10_815,
        combined_gpu_wall_cap_seconds=16_800,
    )
    reservation = ledger.reserve(
        reservation_id="watchdog-budget-test",
        gpu_seconds=100,
        max_hourly_rate_usd=1.99,
    )
    receipt = {
        "lease_path": str(tmp_path / "lane.lease.json"),
        "owner_pid": 222,
        "pod_pending_teardown_record": str(pending_path),
        "pod_id": "pod-1",
        "pod_name_prefix": "blueprint-groot-oscar-canary-attempt-",
        "campaign_budget": {
            "status": "reserved",
            "ledger_path": str(ledger_path),
            "reservation_id": "watchdog-budget-test",
            "reserved_at_epoch": 9_999_999_900.0,
            "reservation": reservation,
            "identity": {
                "initial_spent_usd": 11.57,
                "initial_used_gpu_seconds": 10_815,
                "total_spend_cap_usd": 20.0,
                "combined_gpu_wall_cap_seconds": 16_800,
            },
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    os.chmod(receipt_path, 0o600)

    monkeypatch.setattr(
        "blueprint_pipeline.paid_lane_guard.close_pending_teardown",
        lambda path, evidence: {
            "status": "closed",
            "path": path,
            "evidence": evidence,
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.paid_provider_lane_lease.restore_paid_provider_lane_lease_to_retained_watchdog",
        lambda observed: {
            "status": "restored",
            "restored": observed == receipt,
        },
    )

    class EmptyProvider:
        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == receipt["pod_name_prefix"]
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=receipt["pod_name_prefix"],
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: EmptyProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )

    assert result["status"] == "provider_terminal"
    assert result["control_plane_terminal"] is True
    assert result["pod_pending_teardown_close"]["status"] == "closed"
    assert result["provider_lane_owner_return"]["status"] == "restored"
    assert result["campaign_budget_settlement"]["status"] == "settled"
    assert result["campaign_budget_settlement"]["charged_gpu_seconds"] == 100

    retried = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=receipt["pod_name_prefix"],
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: EmptyProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )
    assert retried["campaign_budget_settlement"]["status"] == "settled"


def test_unverified_teardown_retains_open_campaign_reservation(tmp_path) -> None:
    ledger = ProductionGpuCampaignBudget(
        tmp_path / "campaign-budget.json",
        initial_spent_usd=11.57,
        initial_used_gpu_seconds=10_815,
        combined_gpu_wall_cap_seconds=16_800,
    )
    ledger.reserve(
        reservation_id="unverified-watchdog",
        gpu_seconds=100,
        max_hourly_rate_usd=1.99,
    )

    class UnverifiedProvider:
        def billable_inventory(self, *, name_prefix: str):
            del name_prefix
            raise TimeoutError

    result = run_watchdog(
        out_dir=tmp_path / "watchdog",
        pod_name_prefix="blueprint-groot-oscar-canary-unverified-",
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: UnverifiedProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )
    assert result["status"] == "teardown_unverified"
    assert ledger.snapshot()["open_reservation_count"] == 1
