"""Finite, worst-case spend admission for the paid Isaac G1 lane."""
from __future__ import annotations

import math
import os
from typing import Any, Mapping

ISAAC_G1_MAX_SPEND_USD_ENV = "BLUEPRINT_ISAAC_G1_MAX_SPEND_USD"
DEFAULT_MAX_HOURLY_RATE_USD = 5.0
TEARDOWN_RECONCILIATION_GRACE_SECONDS = 900


def capacity_preflight_hourly_rate(
    capacity: Mapping[str, Any] | None,
) -> float | None:
    """Return the highest rate among provider rows that can actually launch."""
    if not isinstance(capacity, Mapping) or capacity.get("status") != "available":
        return None
    rates: list[float] = []
    rows = list(capacity.get("viable_size_regions") or [])
    rows.extend(capacity.get("viable_gpu_types") or [])
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            rate = float(
                row.get("price_hourly")
                or row.get("on_demand_price_usd_per_hour")
            )
        except (TypeError, ValueError):
            continue
        if math.isfinite(rate) and rate > 0:
            rates.append(rate)
    return max(rates) if rates else None


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def isaac_g1_prelaunch_spend_guard(
    *,
    allow_paid: bool,
    provider_name: str,
    max_spend_usd: float | None,
    max_seconds: int,
    max_hourly_rate_usd: float | None,
    max_hourly_rate_source: str = "configured_provider_ceiling",
    contender_count: int = 1,
    marker_timeout_seconds: int = 0,
    startup_no_runtime_timeout_seconds: int = 0,
    max_attempts: int = 1,
) -> dict:
    env_budget = _float_or_none(os.getenv(ISAAC_G1_MAX_SPEND_USD_ENV))
    requested_budget = max_spend_usd if max_spend_usd is not None else env_budget
    hourly_rate_valid = bool(
        max_hourly_rate_usd is None
        or (
            math.isfinite(float(max_hourly_rate_usd))
            and float(max_hourly_rate_usd) > 0
        )
    )
    hourly_rate = (
        float(max_hourly_rate_usd)
        if max_hourly_rate_usd is not None and hourly_rate_valid
        else DEFAULT_MAX_HOURLY_RATE_USD
    )
    seconds = max(0, int(max_seconds or 0))
    contenders = max(1, int(contender_count or 1))
    # A pod with a runtime/public IP can remain in the marker loop for the
    # full timeout; the earlier no-runtime cutoff is not the worst case.
    # Supervised startup can consume one marker window and then one same-image
    # runtime-gate/canary window. Direct paths share this conservative ceiling.
    per_attempt_startup_seconds = 2 * max(0, int(marker_timeout_seconds or 0))
    attempts = max(1, int(max_attempts or 1))
    startup_budget_seconds = per_attempt_startup_seconds * attempts
    cleanup_budget_seconds = TEARDOWN_RECONCILIATION_GRACE_SECONDS * (attempts + 1)
    billable_budget_seconds = seconds + startup_budget_seconds + cleanup_budget_seconds
    estimated_max_spend_usd = round(
        (hourly_rate * (billable_budget_seconds / 3600.0)) * contenders, 4
    )
    blockers: list[str] = []
    if not allow_paid:
        blockers.append("paid_launch_not_requested")
    if not hourly_rate_valid:
        blockers.append("isaac_g1_max_hourly_rate_must_be_finite_positive")
    if requested_budget is None:
        blockers.append("isaac_g1_max_spend_usd_missing")
    elif not math.isfinite(float(requested_budget)):
        blockers.append("isaac_g1_max_spend_usd_must_be_finite")
    elif requested_budget <= 0:
        blockers.append("isaac_g1_max_spend_usd_must_be_positive")
    elif estimated_max_spend_usd > float(requested_budget):
        blockers.append("isaac_g1_estimated_spend_exceeds_budget")
    can_launch = bool(allow_paid and not blockers)
    return {
        "schema_version": "isaac_g1_kitchen_parity_prelaunch_spend_guard.v1",
        "status": "passed" if can_launch else "blocked",
        "provider": provider_name,
        "allow_paid": bool(allow_paid),
        "required_before_provider_launch": True,
        "can_launch": can_launch,
        "requested_budget_usd": requested_budget,
        "budget_source": (
            "argument"
            if max_spend_usd is not None
            else "env"
            if env_budget is not None
            else "missing"
        ),
        "estimated_max_spend_usd": estimated_max_spend_usd,
        "max_hourly_rate_usd": hourly_rate,
        "max_hourly_rate_source": str(max_hourly_rate_source),
        "max_seconds": seconds,
        "render_budget_seconds": seconds,
        "startup_budget_seconds": startup_budget_seconds,
        "startup_budget_per_attempt_seconds": per_attempt_startup_seconds,
        "teardown_reconciliation_grace_per_attempt_seconds": (
            TEARDOWN_RECONCILIATION_GRACE_SECONDS
        ),
        "cleanup_budget_seconds": cleanup_budget_seconds,
        "billable_budget_seconds": billable_budget_seconds,
        "marker_timeout_seconds": int(marker_timeout_seconds or 0),
        "startup_no_runtime_timeout_seconds": int(
            startup_no_runtime_timeout_seconds or 0
        ),
        "max_attempts": attempts,
        "contender_count": contenders,
        "blockers": blockers,
        "claim_boundary": {
            "spend_guard_only": True,
            "can_launch_is_not_provider_success": True,
            "can_launch_is_not_task_success": True,
            "no_provider_api_call_before_can_launch": True,
        },
    }
