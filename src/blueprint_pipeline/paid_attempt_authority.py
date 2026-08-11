"""Shared fail-closed bindings for single-use paid-attempt authorities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def normalize_external_instance_allowlist(value: Any) -> tuple[int, ...] | None:
    """Normalize an explicit provider-owned external-instance allowlist.

    Paid launch authorities must bind the same exact IDs supplied to the
    provider prelaunch inventory guard.  An empty list is a valid assertion
    that no external billable instances may be present.
    """

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    normalized: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            return None
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        return None
    return tuple(sorted(normalized))


__all__ = ["normalize_external_instance_allowlist"]
