"""Invocation-owned retained output allowance; no input/I/O/authority policy.

Transient child projections charge conservatively without refunds. Public legacy
joins do not use this sink unless their private composition path receives it.
"""
from __future__ import annotations

import json
import math


class RetainedEmissionBudgetError(ValueError):
    """Fixed private resource refusal, translated only by the outer caller."""


def _require(condition):
    if not condition:
        raise RetainedEmissionBudgetError('retained_lineage_emission_limit')


def _measure(value, limit):
    """Measure compact UTF8 without encoding whole strings/documents or fans."""
    size, pending = 0, [iter((value,))]
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if isinstance(item, dict):
            size += 2 + max(0, len(item) - 1) + len(item)
            pending.extend((iter(item.values()), iter(item.keys())))
        elif isinstance(item, list):
            size += 2 + max(0, len(item) - 1)
            pending.append(iter(item))
        elif isinstance(item, str):
            _require(len(item) <= limit - size)
            size += 2
            for char in item:
                code = ord(char)
                _require(not 0xD800 <= code <= 0xDFFF)
                size += (2 if char in '"\\\b\f\n\r\t' else 6 if code < 32 else
                         1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4)
                _require(size <= limit)
        elif item is None:
            size += 4
        elif type(item) is bool:
            size += 4 if item else 5
        elif type(item) in (int, float):
            _require(math.isfinite(item))
            size += len(json.dumps(item, allow_nan=False))
        else:
            _require(False)
        _require(size <= limit)
    return size


class RetainedEmissionBudget:
    def __init__(self, *, max_bytes, max_rows, max_references, _ancestors=()):
        _require(all(type(value) is int and value >= 0 for value in (max_bytes, max_rows, max_references)))
        self.caps = {'bytes': max_bytes, 'rows': max_rows, 'references': max_references}
        self.used = dict.fromkeys(self.caps, 0)
        self.ancestors = _ancestors

    @property
    def remaining_bytes(self):
        return min(s.caps['bytes'] - s.used['bytes'] for s in (*self.ancestors, self))

    def scope(self, *, max_bytes, max_rows, max_references):
        return RetainedEmissionBudget(max_bytes=max_bytes, max_rows=max_rows, max_references=max_references,
                                      _ancestors=(*self.ancestors, self))

    def reserve_row(self, row, *, reference=False):
        scopes = (*self.ancestors, self)
        for scope in scopes:
            _require(scope.used['rows'] < scope.caps['rows'])
            _require(not reference or scope.used['references'] < scope.caps['references'])
        size = _measure(row, self.remaining_bytes)
        for scope in scopes:
            scope.used['bytes'] += size
            scope.used['rows'] += 1
            scope.used['references'] += int(reference)

    def preflight_row(self, row, native_remaining, *, reference=False):
        # No counter change. Fail shared occurrence caps and stream the row
        # under min(native,parent) before the legacy stack-based measurement.
        for scope in (*self.ancestors, self):
            _require(scope.used['rows'] < scope.caps['rows'])
            _require(not reference or scope.used['references'] < scope.caps['references'])
        effective = min(native_remaining, self.remaining_bytes)
        _measure(row, effective)
        return effective

    def reserve_reference(self, row):
        self.reserve_row(row, reference=True)

    def reserve_provenance(self, values):
        return self.rows(values)

    def rows(self, values=(), *, reference=False):
        return _Rows(self, values, reference=reference)

    def check_document(self, value):
        # Framing check only. Each emission must already have been charged.
        _measure(value, min(scope.caps['bytes'] for scope in (*self.ancestors, self)))


class _Rows(list):
    def __init__(self, budget, values, *, reference):
        super().__init__()
        self.budget, self.reference = budget, reference
        self.extend(values)

    def append(self, row):
        self.budget.reserve_row(row, reference=self.reference)
        super().append(row)

    def extend(self, values):
        for row in values:
            self.append(row)

    def __iadd__(self, values):
        self.extend(values)
        return self
