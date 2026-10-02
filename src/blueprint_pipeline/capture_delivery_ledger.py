"""Pure semantic delivery identities ended by a retained capture job ledger."""

import re
from collections.abc import Mapping
from typing import Any


def ended_producer_delivery_keys(ledger: Mapping[str, Any]) -> set[str]:
    """Select exact original deliveries; path or payload text grants no identity."""
    def key(value: Any) -> str:
        return value if isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) else ""

    keys = {key(ledger.get("terminal_producer_delivery_key"))}
    history = ledger.get("attempt_history")
    for row in history if isinstance(history, list) else ():
        if not isinstance(row, Mapping):
            continue
        if row.get("status") == "terminal_authority_ended":
            keys.add(key(row.get("producer_delivery_key")))
        elif row.get("status") == "reopened_after_terminal_authority":
            keys.add(key(row.get("terminal_producer_delivery_key")))
    if ledger.get("status") == "completed":
        keys.add(key(ledger.get("producer_delivery_key")))
    keys.discard("")
    return keys
