"""Exact official-charge period admission, independent of extraction layouts."""
from __future__ import annotations
from collections.abc import Mapping
from datetime import datetime
from typing import Any
import math

class VastOfficialBillingExtractionError(ValueError):
    """The retained billing evidence was incomplete, ambiguous, or altered."""


def validate_charge_period(row: Mapping[str, Any], source_receipt: Mapping[str, Any]) -> None:
    """Reject reversed/future/out-of-query periods without repricing any row.

    A source without a declared cohort must be refreshed before it can
    authorize financial closure; historical bytes are never rewritten.
    """
    start, end = row.get("start"), row.get("end")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (start, end)) or not 0 <= start <= end:
        raise VastOfficialBillingExtractionError("vast_official_charge_period_invalid")
    if not {"cohort_start_at", "cohort_end_at"}.issubset(source_receipt):
        raise VastOfficialBillingExtractionError("vast_official_charge_period_window_missing")
    try:
        lower = datetime.fromisoformat(str(source_receipt["cohort_start_at"]).replace("Z", "+00:00"))
        upper = datetime.fromisoformat(str(source_receipt["cohort_end_at"]).replace("Z", "+00:00"))
        if lower.tzinfo is None or upper.tzinfo is None:
            raise ValueError("unbound timezone")
        valid = lower.timestamp() <= start <= end <= upper.timestamp()
    except (KeyError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise VastOfficialBillingExtractionError("vast_official_charge_period_invalid")
