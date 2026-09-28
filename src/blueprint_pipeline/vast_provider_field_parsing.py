"""Parse provider capability and version fields without coercing unknowns."""

from __future__ import annotations

import re
from typing import Any


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def normalized_binary_capability(value: Any) -> bool | None:
    """Normalize provider 0/1 capability fields without treating unknown as false."""
    if isinstance(value, bool):
        return value
    parsed = number(value)
    if parsed == 1:
        return True
    if parsed == 0:
        return False
    text = value.strip().lower() if isinstance(value, str) else ""
    if text in {"true", "yes"}:
        return True
    if text in {"false", "no"}:
        return False
    return None


def content_range_total_bytes(value: Any) -> int | None:
    text = value.strip() if isinstance(value, str) else ""
    if "/" not in text:
        return None
    total = text.rsplit("/", 1)[-1].strip()
    if not total or total == "*":
        return None
    try:
        parsed = int(total)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def version_tuple(value: Any) -> tuple[int, int, int] | None:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return None
    parts = re.findall(r"\d+", text)
    if not parts:
        return None
    numbers = [int(item) for item in parts[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]
