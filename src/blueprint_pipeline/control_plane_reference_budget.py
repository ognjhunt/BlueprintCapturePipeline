"""One public, pure resource budget for a bounded reference-collection invocation.

No filesystem or observation operations live here. Defaults of the existing
observers stay independent; only explicit composition shares these stricter caps.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Any

MAX_ROOTS, MAX_GROUPS = 16, 128
MAX_ROWS, MAX_ENTRIES = 10_000, 20_000
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_RAW_BYTES = MAX_OUTPUT_BYTES = 20 * 1024 * 1024
MAX_VALUES, MAX_DEPTH, MAX_FACTS = 100_000, 64, 20_000
MAX_BLOCKERS = 32


class ReferenceCollectionBudgetError(ValueError):
    """A fixed resource/API refusal; never carries input or exception text."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ReferenceCollectionBudget:
    """Single-use deadline/counters; closing or exhaustion cannot be reset."""

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic,
                 time_budget_seconds: float = 5.0):
        if (not callable(monotonic) or type(time_budget_seconds) not in (int, float)
                or not 0 < time_budget_seconds <= 5 or not math.isfinite(time_budget_seconds)):
            raise ReferenceCollectionBudgetError("reference_budget_parameters_invalid")
        self._monotonic, self._duration = monotonic, float(time_budget_seconds)
        self._deadline: float | None = None
        self._last: float | None = None
        self._closed = False
        self._failure: str | None = None
        self._limits = {"roots": MAX_ROOTS, "groups": MAX_GROUPS, "rows": MAX_ROWS,
                       "entries": MAX_ENTRIES, "raw_bytes": MAX_RAW_BYTES,
                       "values": MAX_VALUES, "facts": MAX_FACTS, "output_bytes": MAX_OUTPUT_BYTES}
        self._counts = dict.fromkeys(self.limits, 0)
        self.blockers: set[str] = set()

    @property
    def monotonic(self):
        return self._monotonic

    @property
    def duration(self):
        return self._duration

    @property
    def deadline(self):
        return self._deadline

    @property
    def last(self):
        return self._last

    @property
    def limits(self):
        return MappingProxyType(self._limits)

    @property
    def counts(self):
        return MappingProxyType(self._counts)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failure(self) -> str | None:
        return self._failure

    def fail(self, code: str) -> None:
        self._failure = self.failure or code
        self.block(code)
        raise ReferenceCollectionBudgetError(self.failure)

    def block(self, code: str) -> None:
        if code in self.blockers or len(self.blockers) < MAX_BLOCKERS:
            self.blockers.add(code)
        else:
            self.blockers.add("reference_blockers_truncated")

    def bind(self, *, monotonic: Callable[[], float], time_budget_seconds: float) -> None:
        if (monotonic is not time.monotonic and monotonic is not self.monotonic
                or time_budget_seconds != 5.0 and time_budget_seconds != self.duration):
            raise ReferenceCollectionBudgetError("reference_budget_parameters_invalid")

    def tick(self) -> None:
        if self.closed:
            self.fail("reference_budget_closed")
        if self.failure:
            raise ReferenceCollectionBudgetError(self.failure)
        try:
            current = self.monotonic()
            if type(current) not in (int, float) or not math.isfinite(current):
                raise ValueError
            current = float(current)
            if self.last is not None and current < self.last:
                raise ValueError
        except Exception:
            self.fail("reference_clock_invalid")
        self._last = current
        if self.deadline is None:
            self._deadline = current + self.duration
        if current >= self.deadline:
            self.fail("reference_deadline_exceeded")

    def available(self, kind: str, amount: int) -> None:
        self.tick()
        if kind not in self.limits or type(amount) is not int or amount < 0:
            raise ReferenceCollectionBudgetError("reference_budget_parameters_invalid")
        if amount > self.limits[kind] - self.counts[kind]:
            self.fail("reference_" + kind + "_limit")

    def charge(self, kind: str, amount: int = 1) -> None:
        self.available(kind, amount)
        self._counts[kind] += amount

    def measure(self, value: Any, *, cap: int | None = None) -> int:
        """Bound JSON representation before dataclass conversion/encoding."""
        limit = self.limits["output_bytes"] if cap is None else min(cap, self.limits["output_bytes"])
        total = 0
        def add(count: int) -> None:
            nonlocal total
            total += count
            if total > limit:
                self.fail("reference_output_bytes_limit")
        def visit(item: Any, depth: int) -> None:
            self.tick()
            if depth > MAX_DEPTH:
                self.fail("reference_depth_limit")
            self.charge("values")
            if isinstance(item, str):
                add(2)
                for offset, char in enumerate(item):
                    if offset % 1024 == 0:
                        self.tick()
                    ordinal = ord(char)
                    if 0xD800 <= ordinal <= 0xDFFF:
                        self.fail("reference_output_invalid")
                    add(2 if char in '\\"\b\f\n\r\t' else 6 if ordinal < 32 else 1 if ordinal < 128 else 2 if ordinal < 2048 else 3 if ordinal < 65536 else 4)
            elif is_dataclass(item) and not isinstance(item, type):
                members = fields(item)
                add(2)
                for index, field in enumerate(members):
                    add(2 if index else 0)
                    visit(field.name, depth + 1)
                    add(2)
                    visit(getattr(item, field.name), depth + 1)
            elif isinstance(item, dict):
                self.available("values", len(item))
                add(2)
                for index, (key, child) in enumerate(item.items()):
                    if not isinstance(key, str):
                        self.fail("reference_output_invalid")
                    add(2 if index else 0)
                    visit(key, depth + 1)
                    add(2)
                    visit(child, depth + 1)
            elif isinstance(item, (list, tuple)):
                self.available("values", len(item))
                add(2)
                for index, child in enumerate(item):
                    add(2 if index else 0)
                    visit(child, depth + 1)
            elif item is None:
                add(4)
            elif type(item) is bool:
                add(4 if item else 5)
            elif type(item) is int:
                if item.bit_length() > 4096:
                    self.fail("reference_output_bytes_limit")
                add(len(str(item)))
            elif type(item) is float and math.isfinite(item):
                add(len(str(item)))
            else:
                self.fail("reference_output_invalid")
        visit(value, 0)
        self.tick()
        return total

    def retain(self, value: Any) -> None:
        self.charge("output_bytes", self.measure(value, cap=self.limits["output_bytes"] - self.counts["output_bytes"]))

    def preflight(self, text: str) -> None:
        """Lexical-only JSON allocation proof; syntax remains the parser's job."""
        depth = 0
        quoted = escaped = atom = False
        for offset, char in enumerate(text):
            if offset % 1024 == 0:
                self.tick()
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted, atom = True, False
                self.charge("values")
            elif char in "{[":
                depth += 1
                if depth > MAX_DEPTH:
                    self.fail("reference_depth_limit")
                atom = False
                self.charge("values")
            elif char in "}]":
                depth -= 1
                atom = False
            elif char in " \t\r\n,:":
                atom = False
            elif not atom:
                atom = True
                self.charge("values")
        self.tick()

    def close(self) -> None:
        self._closed = True


def bind_budget(budget: ReferenceCollectionBudget | None, *, monotonic: Callable[[], float],
                time_budget_seconds: float, error: type[ValueError], code: str) -> None:
    if budget is not None:
        if not isinstance(budget, ReferenceCollectionBudget):
            raise error(code)
        try:
            budget.bind(monotonic=monotonic, time_budget_seconds=time_budget_seconds)
        except ReferenceCollectionBudgetError:
            raise error(code) from None
