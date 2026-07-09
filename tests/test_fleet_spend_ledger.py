"""Tests for the platform-level fleet spend ledger and aggregate ceiling (R041).

These tests intentionally import ONLY ``blueprint_pipeline.fleet_spend_ledger``
(pure stdlib) so they run in a minimal environment without the orchestrator's
heavier deps (numpy/PIL). Determinism is guaranteed via an injected fixed clock
and a tmp-path ledger; no wall-clock or network access is used.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.fleet_spend_ledger import (
    BLOCKER_DAILY_CAP,
    BLOCKER_KILL_SWITCH,
    BLOCKER_LEDGER_UNREADABLE,
    BLOCKER_MAX_CONCURRENT_GPU,
    BLOCKER_MONTHLY_CAP,
    FleetSpendCaps,
    FleetSpendLedger,
    evaluate_fleet_spend_guard,
    resolve_ledger_path,
)


NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)


def _fixed_clock(now: datetime = NOW):
    return lambda: now


def _ledger(tmp_path: Path, now: datetime = NOW) -> FleetSpendLedger:
    return FleetSpendLedger(tmp_path / "fleet_spend_ledger.json", clock=_fixed_clock(now))


# ---------------------------------------------------------------------------
# Core required scenarios
# ---------------------------------------------------------------------------


def test_under_budget_allows(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.record_launch(job_id="job-a", estimated_usd=10.0, gpu_count=1, provider="vast")
    caps = FleetSpendCaps(
        daily_spend_usd=100.0, monthly_spend_usd=1000.0, max_concurrent_gpu=4
    )

    decision = ledger.check_budget(estimated_usd=25.0, gpu_count=1, caps=caps)

    assert decision["allowed"] is True
    assert decision["fail_closed"] is False
    assert decision["blockers"] == []
    assert decision["aggregate_ceiling_enforced"] is True
    assert decision["ledger_read"] is True
    assert decision["current"]["rolling_daily_spend_usd"] == 10.0
    assert decision["projected"]["rolling_daily_spend_usd"] == 35.0
    assert decision["remaining"]["daily_spend_usd"] == 90.0
    assert decision["remaining"]["monthly_spend_usd"] == 990.0
    assert decision["remaining"]["concurrent_gpu"] == 3


def test_over_daily_cap_blocks(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.record_launch(job_id="job-a", estimated_usd=80.0, gpu_count=1)
    caps = FleetSpendCaps(daily_spend_usd=100.0)

    # 80 already spent today + 25 pending = 105 > 100 -> fail closed.
    decision = ledger.check_budget(estimated_usd=25.0, gpu_count=1, caps=caps)

    assert decision["allowed"] is False
    assert decision["fail_closed"] is True
    assert BLOCKER_DAILY_CAP in decision["blockers"]
    assert decision["projected"]["rolling_daily_spend_usd"] == 105.0
    assert decision["remaining"]["daily_spend_usd"] == 20.0


def test_over_monthly_cap_blocks_even_when_daily_ok(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    # Spend 10 days ago: counts toward the rolling 30d window, not the 24h one.
    old = NOW - timedelta(days=10)
    ledger.record_launch(job_id="job-old", estimated_usd=900.0, gpu_count=1, now=old)
    caps = FleetSpendCaps(daily_spend_usd=10_000.0, monthly_spend_usd=1000.0)

    decision = ledger.check_budget(estimated_usd=200.0, gpu_count=1, caps=caps)

    assert decision["allowed"] is False
    assert BLOCKER_MONTHLY_CAP in decision["blockers"]
    # Daily cap is NOT breached (the old spend is outside the 24h window).
    assert BLOCKER_DAILY_CAP not in decision["blockers"]
    assert decision["current"]["rolling_daily_spend_usd"] == 0.0
    assert decision["current"]["rolling_monthly_spend_usd"] == 900.0
    assert decision["projected"]["rolling_monthly_spend_usd"] == 1100.0


def test_over_max_concurrent_gpu_blocks(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.record_launch(job_id="job-a", estimated_usd=1.0, gpu_count=2, active=True)
    ledger.record_launch(job_id="job-b", estimated_usd=1.0, gpu_count=1, active=True)
    caps = FleetSpendCaps(max_concurrent_gpu=4)

    # 3 active + 2 pending = 5 > 4 -> blocked.
    decision = ledger.check_budget(estimated_usd=1.0, gpu_count=2, caps=caps)

    assert decision["allowed"] is False
    assert BLOCKER_MAX_CONCURRENT_GPU in decision["blockers"]
    assert decision["current"]["active_gpu_pods"] == 3
    assert decision["projected"]["active_gpu_pods"] == 5
    assert decision["remaining"]["concurrent_gpu"] == 1


def test_inactive_pods_do_not_count_toward_concurrency(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.record_launch(job_id="job-a", estimated_usd=1.0, gpu_count=3, active=True)
    ledger.mark_inactive(job_id="job-a")
    caps = FleetSpendCaps(max_concurrent_gpu=4)

    decision = ledger.check_budget(estimated_usd=1.0, gpu_count=3, caps=caps)

    assert decision["current"]["active_gpu_pods"] == 0
    assert decision["allowed"] is True


def test_kill_switch_blocks_all_launches(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    # Kill switch blocks even a tiny, well-under-budget launch.
    caps = FleetSpendCaps(daily_spend_usd=1_000_000.0, kill_switch=True)

    decision = ledger.check_budget(estimated_usd=0.01, gpu_count=1, caps=caps)

    assert decision["allowed"] is False
    assert decision["fail_closed"] is True
    assert decision["kill_switch_engaged"] is True
    assert BLOCKER_KILL_SWITCH in decision["blockers"]


def test_no_caps_configured_allows_backward_compat(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    # Even a huge pending spend passes when nothing is configured.
    caps = FleetSpendCaps()

    decision = ledger.check_budget(estimated_usd=1_000_000.0, gpu_count=999, caps=caps)

    assert decision["allowed"] is True
    assert decision["fail_closed"] is False
    assert decision["aggregate_ceiling_enforced"] is False
    assert decision["blockers"] == []
    # Default-safe: the ledger is not even read when nothing is enforced.
    assert decision["ledger_read"] is False
    assert decision["current"]["rolling_daily_spend_usd"] is None
    assert decision["remaining"]["daily_spend_usd"] is None


# ---------------------------------------------------------------------------
# Fail-closed and boundary behaviour
# ---------------------------------------------------------------------------


def test_exactly_at_cap_is_allowed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.record_launch(job_id="job-a", estimated_usd=75.0, gpu_count=1)
    caps = FleetSpendCaps(daily_spend_usd=100.0)

    # 75 + 25 == 100, not > 100 -> allowed.
    decision = ledger.check_budget(estimated_usd=25.0, gpu_count=1, caps=caps)

    assert decision["allowed"] is True
    assert decision["remaining"]["daily_spend_usd"] == 25.0


def test_unreadable_ledger_fails_closed_when_caps_active(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    ledger = FleetSpendLedger(path, clock=_fixed_clock())
    caps = FleetSpendCaps(daily_spend_usd=100.0)

    decision = ledger.check_budget(estimated_usd=1.0, gpu_count=1, caps=caps)

    assert decision["allowed"] is False
    assert decision["fail_closed"] is True
    assert BLOCKER_LEDGER_UNREADABLE in decision["blockers"]
    assert "ledger_error" in decision


def test_unreadable_ledger_ignored_when_no_caps(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    ledger = FleetSpendLedger(path, clock=_fixed_clock())

    # No caps -> default-safe: the corrupt ledger is never read, launch allowed.
    decision = ledger.check_budget(estimated_usd=1.0, gpu_count=1, caps=FleetSpendCaps())

    assert decision["allowed"] is True
    assert decision["blockers"] == []


def test_actual_usd_supersedes_estimate(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.record_launch(job_id="job-a", estimated_usd=90.0, gpu_count=1)
    # Reconcile: the job actually only cost 10 USD.
    assert ledger.record_actual(job_id="job-a", actual_usd=10.0) is True
    caps = FleetSpendCaps(daily_spend_usd=100.0)

    decision = ledger.check_budget(estimated_usd=50.0, gpu_count=1, caps=caps)

    # 10 (actual) + 50 = 60 <= 100 -> allowed (would be blocked at estimate 90).
    assert decision["current"]["rolling_daily_spend_usd"] == 10.0
    assert decision["allowed"] is True


# ---------------------------------------------------------------------------
# Persistence, determinism, and env resolution
# ---------------------------------------------------------------------------


def test_record_launch_persists_json(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.record_launch(job_id="job-a", estimated_usd=12.5, gpu_count=2, provider="runpod")

    path = tmp_path / "fleet_spend_ledger.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "blueprint_fleet_spend_ledger.v1"
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert record["job_id"] == "job-a"
    assert record["estimated_usd"] == 12.5
    assert record["gpu_count"] == 2
    assert record["provider"] == "runpod"
    # Timestamp is derived from the injected clock -> deterministic.
    assert record["timestamp"] == NOW.isoformat()


def test_missing_ledger_file_is_empty_not_created(tmp_path: Path) -> None:
    path = tmp_path / "does-not-exist.json"
    ledger = FleetSpendLedger(path, clock=_fixed_clock())

    totals = ledger.totals()

    assert totals.record_count == 0
    assert totals.rolling_daily_spend_usd == 0.0
    # A read must not create the file.
    assert not path.exists()


def test_caps_from_env_parses_and_ignores_nonpositive() -> None:
    caps = FleetSpendCaps.from_env(
        {
            "BLUEPRINT_FLEET_DAILY_SPEND_USD": "250.5",
            "BLUEPRINT_FLEET_MONTHLY_SPEND_USD": "0",  # non-positive -> disabled
            "BLUEPRINT_FLEET_MAX_CONCURRENT_GPU": "8",
            "BLUEPRINT_FLEET_SPEND_KILL_SWITCH": "true",
        }
    )
    assert caps.daily_spend_usd == 250.5
    assert caps.monthly_spend_usd is None
    assert caps.max_concurrent_gpu == 8
    assert caps.kill_switch is True
    assert caps.any_enforced is True


def test_empty_env_yields_no_enforcement() -> None:
    caps = FleetSpendCaps.from_env({})
    assert caps.any_enforced is False
    assert caps.daily_spend_usd is None
    assert caps.kill_switch is False


def test_resolve_ledger_path_prefers_argument_then_env_then_default(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.json"
    assert resolve_ledger_path(explicit) == explicit
    assert resolve_ledger_path(
        None, env={"BLUEPRINT_FLEET_SPEND_LEDGER_PATH": "/tmp/from-env.json"}
    ) == Path("/tmp/from-env.json")
    assert resolve_ledger_path(None, env={}) == Path("output/fleet_spend_ledger.json")


def test_evaluate_fleet_spend_guard_default_safe_without_env(tmp_path: Path) -> None:
    # No fleet env vars -> allow and do not read the (missing) ledger.
    decision = evaluate_fleet_spend_guard(
        estimated_usd=500.0,
        gpu_count=3,
        clock=_fixed_clock(),
        env={},
        ledger_path=tmp_path / "ledger.json",
    )
    assert decision["allowed"] is True
    assert decision["aggregate_ceiling_enforced"] is False
    assert decision["ledger_read"] is False


def test_evaluate_fleet_spend_guard_blocks_via_env(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    seed = FleetSpendLedger(ledger_path, clock=_fixed_clock())
    seed.record_launch(job_id="job-a", estimated_usd=95.0, gpu_count=1)

    decision = evaluate_fleet_spend_guard(
        estimated_usd=20.0,
        gpu_count=1,
        clock=_fixed_clock(),
        env={"BLUEPRINT_FLEET_DAILY_SPEND_USD": "100"},
        ledger_path=ledger_path,
    )
    assert decision["allowed"] is False
    assert BLOCKER_DAILY_CAP in decision["blockers"]


def test_naive_clock_is_treated_as_utc(tmp_path: Path) -> None:
    naive_now = datetime(2026, 7, 9, 12, 0, 0)  # no tzinfo
    ledger = FleetSpendLedger(tmp_path / "ledger.json", clock=lambda: naive_now)
    caps = FleetSpendCaps(daily_spend_usd=100.0)

    decision = ledger.check_budget(estimated_usd=1.0, gpu_count=1, caps=caps)

    assert decision["allowed"] is True
    # generated_at is normalised to an aware UTC timestamp.
    parsed = datetime.fromisoformat(decision["generated_at"])
    assert parsed.tzinfo is not None
