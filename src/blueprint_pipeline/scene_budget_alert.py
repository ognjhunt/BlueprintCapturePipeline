"""Warn the owner early: a scene's spend ceiling passed a whole multiple of its grant (2x, 3x, ...).

Owner decision 2026-11-10: a job's ceiling is its expected budget, and a
flagged job extends it automatically (``scene_budget_standing_extension``).
Instead of being asked late, the owner is paged once per job per whole
multiple of the grant's original figure. This gates nothing: it reads the
intent records and never changes them.

Run inside the stall alarm's tick (``control_plane_stall_alarm``), whose
pages the capacity controller delivers. A persisting page repeats hourly
there, so the row is emitted only for ``ANNOUNCE_SECONDS`` after the tick
that first saw the multiple: long enough for a failed delivery to be
retried on the next ticks, shorter than the hourly repeat.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

CODE = "scene_budget_doubled"
ANNOUNCE_SECONDS = 30 * 60
_STORES = ("execution-budget-extensions", "standing-budget-extensions")


def ceiling_multiple(directory: Path, intent: Mapping[str, Any]) -> tuple[int, float, float]:
    """``(whole multiple, effective ceiling, original figure)`` for one intent."""
    from .scene_budget_standing_extension import effective_scene_budget
    original = Decimal(str(intent["request"]["execution"]["max_total_spend_usd"]))
    ceiling = Decimal(str(effective_scene_budget(directory, intent)["max_total_spend_usd"]))
    return int(ceiling // original) if original > 0 else 0, float(ceiling), float(original)


def check_scene_budgets(root: Path | None, names: Sequence[str], *, now: float,
                        remembered: Mapping[str, Any] | None = None,
                        memory_out: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One page row naming each intent that newly passed a multiple; unreadable intents are counted."""
    from .task_evaluation_scene_intent_contracts import _read
    remembered = remembered if isinstance(remembered, Mapping) else {}
    if root is None:
        return {"status": "not_configured"}, []
    offenders, unreadable, extended = [], 0, 0
    for name in names:
        directory = root / name
        if not any((directory / store).is_dir() for store in _STORES):
            continue
        try:
            intent = _read(directory / "intent.json", "intent_digest")
            multiple, ceiling, original = ceiling_multiple(directory, intent)
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError):
            unreadable += 1
            continue
        extended += 1
        if multiple < 2:
            continue
        prior = remembered.get(name) if isinstance(remembered.get(name), Mapping) else {}
        known, since = prior.get("multiple"), prior.get("since")
        if not (type(known) is int and known >= multiple and isinstance(since, (int, float))):
            since = now  # first tick at this multiple
        multiple = max(multiple, known) if type(known) is int else multiple
        if memory_out is not None:
            memory_out[name] = {"multiple": multiple, "since": since}
        if now - since < ANNOUNCE_SECONDS:
            request = intent["request"]
            offenders.append({"target": f"{name}@{multiple}x", "label": str(request.get("submission_id"))[:120],
                              "multiple": multiple, "ceiling_usd": ceiling, "original_usd": original,
                              "age_seconds": round(now - since)})
    section = {"status": "observed", "extended": extended, "announced": len(offenders), "unreadable": unreadable}
    if not offenders:
        return section, []
    offenders.sort(key=lambda row: row["target"])
    first = offenders[0]
    return section, [{"code": CODE, "count": len(offenders), "targets": [row["target"] for row in offenders[:10]],
                      **first}]


def describe(alert: Mapping[str, Any]) -> str:
    more = int(alert.get("count") or 1) - 1
    tail = f" (+{more} more)" if more > 0 else ""
    return (f"scene spend ceiling now ${alert.get('ceiling_usd')}, {alert.get('multiple')}x its grant's "
            f"${alert.get('original_usd')} (nothing is stopped): {alert.get('label') or alert.get('target')}{tail}")
