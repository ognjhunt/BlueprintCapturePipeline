from pathlib import Path

from blueprint_pipeline.production_gpu_warm_watchdog import (
    arm_watchdog_evidence,
    run_watchdog,
    terminate_at_watchdog_boundary,
)
from blueprint_pipeline.production_gpu_campaign_budget import ProductionGpuCampaignBudget


class Provider:
    name = "runpod"

    def __init__(self) -> None:
        self.terminated: list[str] = []

    def terminate(self, instance_id: str) -> dict:
        self.terminated.append(instance_id)
        return {"status": "terminated", "http": 204}

    def inspect(self, instance_id: str) -> dict:
        return {"status": "unavailable", "http": 404, "instance_id": instance_id}

    def billable_inventory(self, *, name_prefix: str) -> dict:
        return {
            "status": "observed", "api_confirmed": True,
            "live_resource_count": 0, "resources": [], "name_prefix": name_prefix,
        }


def test_arm_writes_independent_supervisor_evidence(tmp_path: Path) -> None:
    result = arm_watchdog_evidence(
        out_dir=tmp_path, deadline_epoch=1200, pid=123, clock=lambda: 1000
    )

    assert result["status"] == "armed"
    assert result["independent_process"] is True
    assert result["pid"] == 123
    assert Path(result["evidence_path"]).is_file()


def test_arm_binds_watchdog_to_durable_campaign_reservation(tmp_path: Path) -> None:
    ledger = tmp_path / "campaign-budget.json"
    result = arm_watchdog_evidence(
        out_dir=tmp_path,
        deadline_epoch=1200,
        pid=123,
        campaign_budget_ledger=ledger,
        campaign_reservation_id="qualification-reservation-one",
        clock=lambda: 1000,
    )

    assert result["campaign_budget_ledger"] == str(ledger.resolve())
    assert result["campaign_reservation_id"] == "qualification-reservation-one"


def test_deadline_terminates_discovered_pod_and_confirms_absence(tmp_path: Path) -> None:
    provider = Provider()
    started = tmp_path / "job" / "started_pod_id.txt"
    started.parent.mkdir()
    started.write_text("pod-1", encoding="utf-8")
    arm_watchdog_evidence(out_dir=tmp_path, deadline_epoch=10_000_000_000, pid=123)

    result = terminate_at_watchdog_boundary(
        out_dir=tmp_path, provider_factory=lambda _name: provider
    )

    assert provider.terminated == ["pod-1"]
    assert result["status"] == "PASS"
    assert result["api_confirmed_absent"] is True


def test_deadline_quarantines_registered_worker_before_provider_teardown(
    tmp_path: Path,
) -> None:
    provider = Provider()
    started = tmp_path / "job" / "started_pod_id.txt"
    started.parent.mkdir()
    started.write_text("pod-1", encoding="utf-8")
    token = tmp_path / "pool-token"
    token.write_text("x" * 32, encoding="utf-8")
    token.chmod(0o600)
    arm_watchdog_evidence(
        out_dir=tmp_path,
        deadline_epoch=10_000_000_000,
        pid=123,
        pool_base_url="https://pool.example.internal",
        pool_token_file=token,
    )
    calls: list[str] = []

    result = terminate_at_watchdog_boundary(
        out_dir=tmp_path,
        provider_factory=lambda _name: provider,
        pool_sender=lambda _base, path, _payload, _token: (
            calls.append(path) or {"state": "quarantined"}
        ),
    )

    assert calls == ["/v1/workers/pod-1/quarantine"]
    assert result["status"] == "PASS"
    assert result["pool_quarantine"]["status"] == "quarantined"


def test_cancel_before_allocation_confirms_inventory_and_performs_no_provider_mutation(
    tmp_path: Path,
) -> None:
    now = {"value": 1000.0}
    (tmp_path / "production_gpu_warm_watchdog.cancel").write_text("cancel")
    provider = Provider()

    result = run_watchdog(
        out_dir=tmp_path,
        deadline_epoch=1200,
        clock=lambda: now["value"],
        sleeper=lambda _seconds: None,
        provider_factory=lambda _name: provider,
    )

    assert result["status"] == "closed_no_allocation"
    assert result["api_confirmed_absent"] is True
    assert provider.terminated == []


def test_blocked_owner_teardown_keeps_watchdog_active_until_absence_is_proven(
    tmp_path: Path,
) -> None:
    now = {"value": 1000.0}
    provider = Provider()
    (tmp_path / "warm_serve_pod.json").write_text(
        '{"status":"teardown_blocked","pod_id":"pod-1"}', encoding="utf-8"
    )

    result = run_watchdog(
        out_dir=tmp_path,
        deadline_epoch=1200,
        clock=lambda: now["value"],
        sleeper=lambda _seconds: now.update(value=1200.0),
        provider_factory=lambda _name: provider,
    )

    assert provider.terminated == ["pod-1"]
    assert result["status"] == "PASS"
    assert result["api_confirmed_absent"] is True


def test_provider_confirmed_no_allocation_releases_reserved_campaign_budget(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "campaign-budget.json"
    ledger = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=3.0,
        initial_used_gpu_seconds=8_815,
    )
    ledger.reserve(
        reservation_id="qualification-reservation-one",
        gpu_seconds=600,
        max_hourly_rate_usd=1.0,
    )
    (tmp_path / "production_gpu_warm_watchdog.cancel").write_text("cancel")

    result = run_watchdog(
        out_dir=tmp_path,
        deadline_epoch=1200,
        clock=lambda: 1000,
        sleeper=lambda _seconds: None,
        provider_factory=lambda _name: Provider(),
        campaign_budget_ledger=ledger_path,
        campaign_reservation_id="qualification-reservation-one",
    )

    assert result["campaign_budget_settlement"]["status"] == "settled"
    assert ledger.snapshot()["remaining_gpu_seconds"] == 7_985
