"""Invocation-owned retained output allowance; no input/I/O/authority policy.

Transient child projections charge conservatively without refunds. Public legacy
joins do not use this sink unless their private composition path receives it.
"""
from __future__ import annotations

import json
import math
import sys

_BUDGET_MODULE_NAME = __package__ + '.control_plane_reference_budget'


class RetainedEmissionBudgetError(ValueError):
    """Fixed private resource refusal, translated only by the outer caller."""


def _require(condition):
    if not condition:
        raise RetainedEmissionBudgetError('retained_lineage_emission_limit')


def _work(budget):
    if budget is not None:
        # Read the live canonical module, never cache a class across reloads.
        module = sys.modules.get(_BUDGET_MODULE_NAME)
        if module is None:
            from .control_plane_reference_budget import ReferenceCollectionBudget
        else:
            ReferenceCollectionBudget = module.ReferenceCollectionBudget
        _require(type(budget) is ReferenceCollectionBudget)
        budget.tick()


def _work_kwargs(budget):
    return {} if budget is None else {'work_budget': budget}


def _work_items(values, budget):
    """Guard the next bounded work item before a comprehension expands it."""
    iterator = iter(values)
    while True:
        if isinstance(values, str):
            budget.tick()
        else:
            budget.available('values', 1)
        try:
            value = next(iterator)
        except StopIteration:
            return
        if not isinstance(values, str):
            budget.charge('values')
        yield value


def _work_rows(values, budget, native=None):
    iterator = iter(values)
    while True:
        budget.available('rows', 1)
        budget.available('facts', 1)
        if native is not None:
            native()
        try:
            row = next(iterator)
        except StopIteration:
            return
        yield row


def _work_order(budget, function, values, *args, **kwargs):
    budget.tick()
    retained = []
    for value in _work_items(values, budget):
        budget.charge('facts')
        retained.append(value)
    result = function(retained, *args, **kwargs)
    budget.tick()
    return result


def _work_collect(budget, function, values, *args, **kwargs):
    return _work_order(budget, function, values, *args, **kwargs)


def _work_sort(budget, values, *args, **kwargs):
    budget.tick()
    budget.available('values', len(values))
    for _ in _work_items(values, budget):
        pass
    values.sort(*args, **kwargs)
    budget.tick()


def _work_call(budget, function, *args, **kwargs):
    """Bound a pure imported encoder/validator at its caller, without changing it."""
    _work(budget)
    if args:
        budget.measure(args[0], cap=budget.limits['output_bytes'] - budget.counts['output_bytes'])
    result = function(*args, **kwargs)
    budget.tick()
    return result


def _work_parse(budget, function, text, **kwargs):
    from functools import partial
    for name in ('object_pairs_hook', 'parse_int', 'parse_float'):
        hook = kwargs.get(name)
        if hook is not None and getattr(hook, '__name__', '') in {'_pairs', '_numeric'}:
            kwargs[name] = partial(hook, work_budget=budget)
    budget.preflight(text)
    result = function(text, **kwargs)
    budget.tick()
    budget.measure(result)
    return result


def _work_hash(budget, function, raw):
    budget.tick()
    result = function(raw)
    budget.tick()
    return result


def _measure(value, limit, *, reserve_proofs=False, work_budget=None):
    """Measure compact UTF8 without encoding whole strings/documents or fans."""
    size, pending = 0, [iter((value,))]
    while pending:
        if work_budget is not None:
            work_budget.available('values', 1)
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if work_budget is not None:
            work_budget.charge('values')
        if isinstance(item, dict):
            size += 2 + max(0, len(item) - 1) + len(item)
            if reserve_proofs and {'role', 'path', 'sha256', 'size_bytes', 'seal_field', 'seal_digest'} <= item.keys():
                # Proof aliases can acquire canonical seal metadata after an
                # emission. Reserve its finite source-controlled label/digest
                # on EVERY occurrence, before storing that alias. The actual
                # document check below adds no such conservative headroom.
                field, digest = item['seal_field'], item['seal_digest']
                _require(field is None or (isinstance(field, str) and len(field) <= 64 and
                         all(char in 'abcdefghijklmnopqrstuvwxyz0123456789_' for char in field)))
                _require(digest is None or (isinstance(digest, str) and len(digest) == 71 and
                         digest.startswith('sha256:') and all(char in '0123456789abcdef' for char in digest[7:])))
                size += 66 - (4 if field is None else len(field) + 2)
                size += 73 - (4 if digest is None else len(digest) + 2)
            pending.extend((iter(item.values()), iter(item.keys())))
        elif isinstance(item, list):
            size += 2 + max(0, len(item) - 1)
            pending.append(iter(item))
        elif isinstance(item, str):
            _require(len(item) <= limit - size)
            size += 2
            # Check the same original deadline around bounded pure character
            # work. Every character still contributes its exact byte/cap check.
            for offset, char in enumerate(item):
                if work_budget is not None and offset % 1024 == 0:
                    work_budget.tick()
                code = ord(char)
                _require(not 0xD800 <= code <= 0xDFFF)
                size += (2 if char in '"\\\b\f\n\r\t' else 6 if code < 32 else
                         1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4)
                _require(size <= limit)
            if work_budget is not None:
                work_budget.tick()
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
    def __init__(self, *, max_bytes, max_rows, max_references, _ancestors=(), work_budget=None):
        _work(work_budget)
        self.work_budget = work_budget
        _require(all(type(value) is int and value >= 0 for value in (max_bytes, max_rows, max_references)))
        self.caps = {'bytes': max_bytes, 'rows': max_rows, 'references': max_references}
        self.used = dict.fromkeys(self.caps, 0)
        self.ancestors = _ancestors

    @property
    def remaining_bytes(self):
        remaining = min(s.caps['bytes'] - s.used['bytes'] for s in (*self.ancestors, self))
        if self.work_budget is not None:
            self.work_budget.tick()
            remaining = min(remaining, self.work_budget.limits['output_bytes'] - self.work_budget.counts['output_bytes'])
        return remaining

    def scope(self, *, max_bytes, max_rows, max_references):
        return RetainedEmissionBudget(max_bytes=max_bytes, max_rows=max_rows, max_references=max_references,
                                      _ancestors=(*self.ancestors, self), work_budget=self.work_budget)

    def reserve_row(self, row, *, reference=False):
        if self.work_budget is not None:
            self.work_budget.available('rows', 1)
            self.work_budget.available('facts', 1)
        scopes = (*self.ancestors, self)
        for scope in scopes:
            _require(scope.used['rows'] < scope.caps['rows'])
            _require(not reference or scope.used['references'] < scope.caps['references'])
        size = _measure(row, self.remaining_bytes, reserve_proofs=True, work_budget=self.work_budget)
        if self.work_budget is not None:
            self.work_budget.charge('rows')
            self.work_budget.charge('facts')
            self.work_budget.charge('output_bytes', size)
        for scope in scopes:
            scope.used['bytes'] += size
            scope.used['rows'] += 1
            scope.used['references'] += int(reference)

    def available_occurrence(self, *, reference=False):
        for scope in (*self.ancestors, self):
            _require(scope.used['rows'] < scope.caps['rows'])
            _require(not reference or scope.used['references'] < scope.caps['references'])

    def preflight_row(self, row, native_remaining, *, reference=False):
        if self.work_budget is not None:
            self.work_budget.available('rows', 1)
            self.work_budget.available('facts', 1)
        # No counter change. Fail shared occurrence caps and stream the row
        # under min(native,parent) before the legacy stack-based measurement.
        for scope in (*self.ancestors, self):
            _require(scope.used['rows'] < scope.caps['rows'])
            _require(not reference or scope.used['references'] < scope.caps['references'])
        effective = min(native_remaining, self.remaining_bytes)
        _measure(row, effective, reserve_proofs=True, work_budget=self.work_budget)
        return effective

    def reserve_reference(self, row):
        self.reserve_row(row, reference=True)

    def reserve_provenance(self, values):
        return self.rows(values)

    def rows(self, values=(), *, reference=False):
        return _Rows(self, values, reference=reference)

    def check_document(self, value):
        # Framing check only. Each emission must already have been charged.
        _measure(value, min(scope.caps['bytes'] for scope in (*self.ancestors, self)), work_budget=self.work_budget)


class _Rows(list):
    def __init__(self, budget, values, *, reference):
        super().__init__()
        self.budget, self.reference = budget, reference
        self.extend(values)

    def append(self, row):
        self.budget.reserve_row(row, reference=self.reference)
        super().append(row)

    def extend(self, values):
        if self.budget.work_budget is not None:
            values = _work_rows(values, self.budget.work_budget,
                                lambda: self.budget.available_occurrence(reference=self.reference))
        for row in values:
            self.append(row)

    def __iadd__(self, values):
        self.extend(values)
        return self
