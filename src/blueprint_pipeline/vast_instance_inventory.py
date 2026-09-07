"""Pure shape admission for provider inventory responses."""
from collections.abc import Mapping, Callable
from typing import Any

def instance_inventory_valid(payload: Any, *, status_reader: Callable) -> bool:
    """A successful HTTP response must contain a complete, typed inventory."""
    if not isinstance(payload, Mapping) or payload.get("success") is False:
        return False
    value = next((payload[key] for key in ("instances", "results", "data", "response")
                  if key in payload), payload)
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping) and value:
        rows = ([value] if any(key in value for key in ("id", "instance_id", "contract_id"))
                else list(value.values()))
    else:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        identifier = row.get("id") or row.get("instance_id") or row.get("contract_id")
        if (isinstance(identifier, bool) or not str(identifier or "").isdigit()
                or int(identifier) <= 0 or not str(status_reader(row) or "").strip()):
            return False
    return True


def active_instance_rows(payload, *, status_reader, row_reader, sanitizer, terminal_statuses):
    if not instance_inventory_valid(payload, status_reader=status_reader):
        raise ValueError("vast_instance_inventory_invalid")
    rows = [sanitizer(row) for row in row_reader(payload)]
    return [row for row in rows if str(row.get("raw_status_normalized") or "").lower()
            and str(row["raw_status_normalized"]).lower() not in set(terminal_statuses)]
