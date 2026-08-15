"""Small pure helpers shared by Vast offer-selection policy."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
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


def keyword_match_rank(value: Any, keywords: Sequence[str]) -> int:
    if not keywords:
        return 0
    haystack = _string(value).lower()
    return 0 if any(_string(keyword).lower() in haystack for keyword in keywords) else 1


def regex_match_rank(value: Any, pattern: str) -> int:
    text, regex = _string(value), _string(pattern)
    if not regex:
        return 0
    try:
        return 0 if re.search(regex, text, flags=re.IGNORECASE) else 1
    except re.error:
        return 1


def machine_id_set(values: Iterable[Any]) -> set[int]:
    result: set[int] = set()
    for value in values:
        number = _number(value)
        if number is not None:
            result.add(int(number))
    return result


def version_at_least(value: Any, minimum: str) -> bool:
    if not minimum:
        return True

    def version_tuple(candidate: Any) -> tuple[int, int, int] | None:
        parts = re.findall(r"\d+", _string(candidate))
        if not parts:
            return None
        numbers = [int(item) for item in parts[:3]]
        numbers.extend([0] * (3 - len(numbers)))
        return numbers[0], numbers[1], numbers[2]

    observed, required = version_tuple(value), version_tuple(minimum)
    if not observed or not required:
        return False
    width = max(len(observed), len(required))
    return observed + (0,) * (width - len(observed)) >= required + (0,) * (width - len(required))
