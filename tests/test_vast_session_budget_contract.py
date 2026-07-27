from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.vast_session_budget_contract import (
    attempt_estimated_cost,
    attempt_runtime_seconds,
    build_vast_session_budget_guard,
    successor_session_live_limit_minutes,
)


def test_attempt_values_prefer_runtime_evidence_and_reject_nonfinite_values() -> None:
    attempt = {
        "actual_live_runtime_seconds_observed_by_adapter": 97.485577,
        "estimated_cost_usd_using_observed_rate": 0.027521,
        "observed_hourly_rate_usd": 1.0162962963,
    }
    assert attempt_runtime_seconds(attempt) == pytest.approx(97.485577)
    assert attempt_estimated_cost(attempt) == pytest.approx(0.027521)
    assert attempt_runtime_seconds({"runtime_seconds_observed_by_adapter": "nan"}) == 0.0
    assert attempt_runtime_seconds({"runtime_seconds_observed_by_adapter": True}) == 0.0
    assert attempt_estimated_cost({"estimated_cost_usd": -1}) == 0.0


def test_successor_limit_reserves_full_ttl_above_prior_runtime(tmp_path: Path) -> None:
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            {
                "attempts": [
                    {"actual_live_runtime_seconds_observed_by_adapter": 97.485577}
                ]
            }
        ),
        encoding="utf-8",
    )

    limit = successor_session_live_limit_minutes(
        budget_path=budget,
        requested_max_live_minutes=180,
    )
    guard = build_vast_session_budget_guard(
        generated_at="2026-07-27T00:00:00+00:00",
        budget_path=budget,
        session_max_live_minutes=limit["session_max_live_runtime_minutes"],
        requested_max_live_minutes=180,
        target_spend_usd=3.25,
        hard_cap_usd=6.0,
        max_hourly_rate=1.05,
    )

    assert limit["prior_live_runtime_minutes_ceiling"] == 2
    assert limit["session_max_live_runtime_minutes"] == 182
    assert guard["status"] == "passed"
    assert guard["prior_estimated_cost_usd"] == 0.0


@pytest.mark.parametrize(
    "payload",
    ["[]", '{"attempts": "not-an-array"}', '{"attempts": [1]}'],
)
def test_malformed_session_ledgers_fail_closed(tmp_path: Path, payload: str) -> None:
    budget = tmp_path / "budget.json"
    budget.write_text(payload, encoding="utf-8")

    limit = successor_session_live_limit_minutes(
        budget_path=budget,
        requested_max_live_minutes=180,
    )
    guard = build_vast_session_budget_guard(
        generated_at="2026-07-27T00:00:00+00:00",
        budget_path=budget,
        session_max_live_minutes=180,
        requested_max_live_minutes=180,
        target_spend_usd=3.25,
        hard_cap_usd=6.0,
        max_hourly_rate=1.05,
    )

    assert limit["status"] == "blocked"
    assert "session_budget_ledger_parse_failed" in limit["blockers"]
    assert guard["status"] == "blocked"
    assert "session_budget_ledger_parse_failed" in guard["blockers"]
