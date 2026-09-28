"""Rolling aggregate spend ceiling for one bounded native Arena attempt."""

from __future__ import annotations

import math
from typing import Any


def rolling_aggregate_spend_ceiling_usd(
    *, prior_spend_usd: Any, authorized_increment_usd: Any
) -> float:
    """Return the exact cumulative ceiling for one newly authorized scope."""

    values = (prior_spend_usd, authorized_increment_usd)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise ValueError("native_task_arena_rolling_spend_ceiling_invalid")
    prior = float(prior_spend_usd)
    increment = float(authorized_increment_usd)
    if (
        not math.isfinite(prior)
        or not math.isfinite(increment)
        or prior < 0
        or increment <= 0
    ):
        raise ValueError("native_task_arena_rolling_spend_ceiling_invalid")
    return round(prior + increment, 6)
