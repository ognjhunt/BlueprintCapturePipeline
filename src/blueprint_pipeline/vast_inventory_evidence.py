"""Pure validation of retained Vast inventory evidence; no provider operations."""
from typing import Any, Mapping

def _inventory_is_confirmed_zero(
    value: Mapping[str, Any], *, name_prefix: str
) -> bool:
    """Accept only one internally consistent, API-confirmed Vast zero row."""

    return bool(
        value.get("status") == "observed"
        and value.get("provider") == "vast"
        and value.get("name_prefix") == name_prefix
        and value.get("api_confirmed") is True
        and value.get("live_resource_count") == 0
        and value.get("resources") == []
    )
