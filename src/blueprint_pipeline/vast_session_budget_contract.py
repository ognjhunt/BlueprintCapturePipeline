"""Pure, reusable Vast session-budget calculations for paid launch admission."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def attempt_runtime_seconds(attempt: Mapping[str, Any]) -> float:
    """Return attributable runtime using explicit evidence before cost inference."""

    for key in (
        "runtime_seconds_observed_by_adapter",
        "actual_live_runtime_seconds_observed_by_adapter",
        "runtime_seconds_estimated_from_teardown_artifact_mtime",
    ):
        value = _number(attempt.get(key))
        if value is not None:
            return max(0.0, value)
    cost = _number(
        attempt.get("estimated_cost_usd_using_observed_rate")
        if attempt.get("estimated_cost_usd_using_observed_rate") is not None
        else attempt.get("estimated_cost_usd")
    )
    hourly = _number(attempt.get("observed_hourly_rate_usd")) or _number(
        attempt.get("selected_hourly_rate_usd")
    )
    if cost is not None and hourly and hourly > 0:
        return max(0.0, cost * 3600.0 / hourly)
    return 0.0


def attempt_estimated_cost(attempt: Mapping[str, Any]) -> float:
    """Return the non-negative estimated cost recorded for one attempt."""

    for key in ("estimated_cost_usd_using_observed_rate", "estimated_cost_usd"):
        value = _number(attempt.get(key))
        if value is not None:
            return max(0.0, value)
    return 0.0


def _load_attempts(budget_path: Path) -> tuple[list[Mapping[str, Any]], str | None]:
    if not budget_path.is_file():
        return [], None
    try:
        payload = json.loads(budget_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("session budget ledger must be a JSON object")
        raw_attempts = payload["attempts"] if "attempts" in payload else []
        if not isinstance(raw_attempts, list):
            raise ValueError("session budget ledger attempts must be a JSON array")
        if any(not isinstance(item, Mapping) for item in raw_attempts):
            raise ValueError("session budget ledger attempts must contain only objects")
        attempts = list(raw_attempts)
        return attempts, None
    except Exception as exc:
        return [], f"{type(exc).__name__}:{str(exc)[:200]}"


def successor_session_live_limit_minutes(
    *, budget_path: Path, requested_max_live_minutes: int
) -> dict[str, Any]:
    """Reserve a full new resource TTL on top of attributable prior runtime."""

    attempts, budget_parse_error = _load_attempts(budget_path)
    blockers = ["session_budget_ledger_parse_failed"] if budget_parse_error else []
    prior_live_seconds = sum(attempt_runtime_seconds(attempt) for attempt in attempts)
    prior_live_minutes_ceiling = int(math.ceil(prior_live_seconds / 60.0))
    requested_minutes = max(0, int(requested_max_live_minutes))
    return {
        "schema_version": "vast_successor_session_live_limit.v1",
        "status": "blocked" if blockers else "passed",
        "budget_path": str(budget_path),
        "budget_ledger_present": budget_path.is_file(),
        "budget_parse_error": budget_parse_error,
        "attempt_count": len(attempts),
        "prior_live_runtime_seconds": round(prior_live_seconds, 6),
        "prior_live_runtime_minutes_ceiling": prior_live_minutes_ceiling,
        "requested_max_live_runtime_minutes": requested_minutes,
        "session_max_live_runtime_minutes": prior_live_minutes_ceiling + requested_minutes,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }


def build_vast_session_budget_guard(
    *,
    generated_at: str,
    budget_path: Path,
    session_max_live_minutes: int | None,
    requested_max_live_minutes: int,
    target_spend_usd: float,
    hard_cap_usd: float,
    max_hourly_rate: float,
) -> dict[str, Any]:
    """Evaluate cumulative runtime and spend without provider access or mutation."""

    attempts, budget_parse_error = _load_attempts(budget_path)
    blockers = ["session_budget_ledger_parse_failed"] if budget_parse_error else []
    warnings: list[str] = []
    prior_live_seconds = sum(attempt_runtime_seconds(attempt) for attempt in attempts)
    prior_estimated_cost = sum(attempt_estimated_cost(attempt) for attempt in attempts)
    requested_max_seconds = max(0, requested_max_live_minutes) * 60.0
    projected_max_cost = max(0.0, max_hourly_rate) * max(
        0, requested_max_live_minutes
    ) / 60.0
    session_max_seconds = (
        max(0, session_max_live_minutes) * 60.0
        if session_max_live_minutes is not None
        else None
    )
    if session_max_seconds is not None:
        if prior_live_seconds >= session_max_seconds:
            blockers.append("session_live_runtime_limit_exhausted")
        elif prior_live_seconds + requested_max_seconds > session_max_seconds:
            blockers.append("requested_live_runtime_would_exceed_session_limit")
    if prior_estimated_cost >= hard_cap_usd:
        blockers.append("session_estimated_spend_hard_cap_exhausted")
    elif prior_estimated_cost + projected_max_cost > hard_cap_usd:
        blockers.append("requested_max_spend_would_exceed_hard_cap")
    if prior_estimated_cost >= target_spend_usd:
        warnings.append("session_estimated_spend_target_already_exceeded")
    elif prior_estimated_cost + projected_max_cost > target_spend_usd:
        warnings.append("requested_max_spend_would_exceed_target")
    return {
        "schema_version": "vast_session_budget_guard.v1",
        "generated_at": generated_at,
        "status": "blocked" if blockers else "passed",
        "budget_path": str(budget_path),
        "budget_ledger_present": budget_path.is_file(),
        "budget_parse_error": budget_parse_error,
        "attempt_count": len(attempts),
        "prior_live_runtime_seconds": round(prior_live_seconds, 6),
        "prior_live_runtime_minutes": round(prior_live_seconds / 60.0, 6),
        "requested_max_live_runtime_minutes": requested_max_live_minutes,
        "session_max_live_runtime_minutes": session_max_live_minutes,
        "prior_estimated_cost_usd": round(prior_estimated_cost, 6),
        "projected_max_incremental_cost_usd": round(projected_max_cost, 6),
        "target_spend_usd": target_spend_usd,
        "hard_cap_usd": hard_cap_usd,
        "blockers": blockers,
        "warnings": warnings,
        "raw_secret_values_recorded": False,
    }
