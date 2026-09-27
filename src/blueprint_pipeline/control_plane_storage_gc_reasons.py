"""Why the storage GC kept what it kept: typed retention reasons, with their bytes.

On 2026-09-27 the installed reclaim timer applied with evidence offload enabled
and reclaimed nothing, and its report could not say why: the evidence manifest
folded four different reasons into one ``active_or_unsealed`` counter. A GC
manifest now carries ``retained_by_reason``: every reason it kept an entry for,
as ``{reason: {"count": n, "bytes": b}}``. A reason is a typed string, never a
path, and ``bytes`` are logical bytes (a hardlinked file counts once per name).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def count_retained(
    by_reason: MutableMapping[str, dict[str, Any]], reason: str, size_bytes: int, *, kind: str | None = None
) -> None:
    """Count one retained entry of ``size_bytes`` under ``reason``, and under ``kind`` within it when given."""

    row = by_reason.setdefault(reason, {"count": 0, "bytes": 0})
    row["count"] += 1
    row["bytes"] += int(size_bytes)
    if kind is not None:
        detail = row.setdefault("by_kind", {}).setdefault(kind, {"count": 0, "bytes": 0})
        detail["count"] += 1
        detail["bytes"] += int(size_bytes)


__all__ = ["count_retained"]
