"""Canonical parent-runtime and spend policy for scene configuration.

The six-stage provider chain is serialized inside one paid allocation.  Its
parent lease therefore has to cover the *sum* of the three unchanged GPU-stage
allowances, not merely the largest individual child timeout.  The additional
reserves below are explicit product policy: they bound bootstrap, deterministic
no-spend work, transfers, output sealing, and teardown without presenting those
values as measured production durations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


GPU_STAGE_TIMEOUT_SECONDS: Mapping[str, int] = {
    "artifixer3d_observed_object_removal": 7_800,
    "content_agents_rigid_replacement": 7_800,
    "simready_native_import_qualification": 1_800,
}
SERIAL_GPU_STAGE_TIMEOUT_SECONDS = sum(GPU_STAGE_TIMEOUT_SECONDS.values())

# Named conservative product-policy reserves.  These are authority limits, not
# empirical claims about how long a successful provider normally takes.
BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS = 6_000
OUTPUT_AND_CLOSURE_RESERVE_SECONDS = 1_800
REQUIRED_PARENT_TTL_SECONDS = (
    SERIAL_GPU_STAGE_TIMEOUT_SECONDS
    + BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS
    + OUTPUT_AND_CLOSURE_RESERVE_SECONDS
)

MAX_HOURLY_RATE_USD = 0.80
MAX_PROVIDER_COMPUTE_SPEND_USD = 6.0
MAX_EXTERNAL_SERVICE_SPEND_USD = 1.5
MAX_ATTEMPT_SPEND_USD = 10.0

PARENT_DEADLINE_EPOCH_ENV = "BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH"
OUTPUT_CLOSURE_RESERVE_SECONDS_ENV = (
    "BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_CLOSURE_RESERVE_SECONDS"
)


def ceil_live_minutes(ttl_seconds: int) -> int:
    """Convert an admitted second budget without shortening its hard lease."""

    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
        raise ValueError("scene_configuration_parent_runtime_budget_invalid")
    if ttl_seconds <= 0:
        raise ValueError("scene_configuration_parent_runtime_budget_invalid")
    return max(1, math.ceil(ttl_seconds / 60))


def required_remaining_stage_seconds(
    stages: list[Mapping[str, Any]], *, start_index: int
) -> int:
    """Return future GPU allowances plus the immutable closure reserve."""

    if not 0 <= start_index <= len(stages):
        raise ValueError("scene_configuration_parent_runtime_budget_invalid")
    remaining = OUTPUT_AND_CLOSURE_RESERVE_SECONDS
    for stage in stages[start_index:]:
        adapter = stage.get("adapter")
        adapter_id = (
            str(adapter.get("id") or "") if isinstance(adapter, Mapping) else ""
        )
        remaining += GPU_STAGE_TIMEOUT_SECONDS.get(adapter_id, 0)
    return remaining


def parent_runtime_budget_blockers(
    *,
    ttl_seconds: Any,
    maximum_hourly_rate_usd: Any,
    provider_compute_spend_cap_usd: Any,
) -> list[str]:
    """Validate that one authority can fund the canonical serialized lease."""

    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int)
        or isinstance(maximum_hourly_rate_usd, bool)
        or not isinstance(maximum_hourly_rate_usd, (int, float))
        or isinstance(provider_compute_spend_cap_usd, bool)
        or not isinstance(provider_compute_spend_cap_usd, (int, float))
    ):
        return ["scene_configuration_parent_runtime_budget_invalid"]
    rate = float(maximum_hourly_rate_usd)
    compute_cap = float(provider_compute_spend_cap_usd)
    if not math.isfinite(rate) or not math.isfinite(compute_cap):
        return ["scene_configuration_parent_runtime_budget_invalid"]
    blockers: list[str] = []
    if ttl_seconds < REQUIRED_PARENT_TTL_SECONDS:
        blockers.append(
            "scene_configuration_parent_runtime_budget_insufficient:"
            f"{REQUIRED_PARENT_TTL_SECONDS}:{ttl_seconds}"
        )
    required_compute = rate * ttl_seconds / 3600.0
    if compute_cap + 1e-9 < required_compute:
        blockers.append(
            "scene_configuration_provider_compute_budget_insufficient:"
            f"{required_compute:.6f}:{compute_cap:.6f}"
        )
    return blockers


__all__ = [
    "BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS",
    "GPU_STAGE_TIMEOUT_SECONDS",
    "MAX_ATTEMPT_SPEND_USD",
    "MAX_EXTERNAL_SERVICE_SPEND_USD",
    "MAX_HOURLY_RATE_USD",
    "MAX_PROVIDER_COMPUTE_SPEND_USD",
    "OUTPUT_AND_CLOSURE_RESERVE_SECONDS",
    "OUTPUT_CLOSURE_RESERVE_SECONDS_ENV",
    "PARENT_DEADLINE_EPOCH_ENV",
    "REQUIRED_PARENT_TTL_SECONDS",
    "SERIAL_GPU_STAGE_TIMEOUT_SECONDS",
    "ceil_live_minutes",
    "parent_runtime_budget_blockers",
    "required_remaining_stage_seconds",
]
