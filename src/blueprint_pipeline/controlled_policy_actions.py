"""Pure numeric action validation shared by the proxy and native worker."""
from __future__ import annotations
import math
from typing import Any

def validate_action_response(
    value: Any, *, action_schema: dict[str, Any]
) -> dict[str, list[list[float]]]:
    """Admit only the fixed numeric action tensor declared by the contract."""

    if not isinstance(value, dict) or set(value) != {"actions"}:
        raise ValueError("company_policy_proxy_response_shape_invalid")
    rows = value.get("actions")
    expected_rows = action_schema.get("chunk_rows")
    channels = action_schema.get("channels")
    if (
        not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows < 1
        or not isinstance(channels, list)
        or not channels
        or not isinstance(rows, list)
        or len(rows) != expected_rows
    ):
        raise ValueError("company_policy_proxy_response_shape_invalid")
    normalized: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(channels):
            raise ValueError("company_policy_proxy_response_shape_invalid")
        normalized_row: list[float] = []
        for value_item, channel in zip(row, channels, strict=True):
            if (
                isinstance(value_item, bool)
                or not isinstance(value_item, (int, float))
                or not math.isfinite(float(value_item))
                or not isinstance(channel, dict)
            ):
                raise ValueError("company_policy_proxy_response_value_invalid")
            bounds = channel.get("raw_accepted_bounds")
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or isinstance(bounds[0], bool)
                or isinstance(bounds[1], bool)
                or not isinstance(bounds[0], (int, float))
                or not isinstance(bounds[1], (int, float))
                or not float(bounds[0]) <= float(value_item) <= float(bounds[1])
            ):
                raise ValueError("company_policy_proxy_response_value_out_of_bounds")
            normalized_row.append(float(value_item))
        normalized.append(normalized_row)
    return {"actions": normalized}
