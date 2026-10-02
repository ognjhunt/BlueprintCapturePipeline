# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_reference_budget.py
#   src/blueprint_pipeline/task_evaluation_scene_lineage_budget.py
"""ADP-009D: remove lookup overhead while retaining every resource check."""
import builtins
import subprocess
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_reference_budget as module
from blueprint_pipeline import task_evaluation_scene_lineage_budget as lineage

Budget = module.ReferenceCollectionBudget


def test_live_work_guard_avoids_repeated_import_and_keeps_every_tick(monkeypatch):
    imports, calls = [], []
    original = builtins.__import__
    def observed(name, *args, **kwargs):
        if name == 'control_plane_reference_budget':
            imports.append(name)
        return original(name, *args, **kwargs)
    budget = Budget(monotonic=lambda: calls.append(True) or 0)
    monkeypatch.setattr(builtins, '__import__', observed)
    for _ in range(4):
        lineage._work(budget)
    assert len(calls) == 4
    assert imports == []


def test_owned_tick_and_available_skip_readonly_wrapper_creation(monkeypatch):
    budget = Budget(monotonic=lambda: 0)
    reads = []
    for name in ('closed', 'failure', 'monotonic', 'last', 'deadline', 'duration', 'limits', 'counts'):
        original = getattr(Budget, name)
        def observed(self, *, name=name, original=original):
            reads.append(name)
            return original.fget(self)
        monkeypatch.setattr(Budget, name, property(observed))
    budget.available('rows', 1)
    assert reads == []
    first, second = budget.limits, budget.limits
    assert first is not second and dict(first) == dict(second)
    counts = budget.counts
    budget.charge('rows', 1)
    assert counts['rows'] == 1
    with pytest.raises(TypeError):
        counts['rows'] = 0


def test_subclass_property_and_clock_order_remains_original():
    events = []
    def prop(name):
        original = getattr(Budget, name)
        def observed(self):
            events.append(name)
            return original.fget(self)
        return property(observed)
    names = ('closed', 'failure', 'monotonic', 'last', 'deadline', 'duration', 'limits', 'counts')
    Subclass = type('ObservedBudget', (Budget,), {name: prop(name) for name in names})
    def clock():
        events.append('clock')
        return 0
    budget = Subclass(monotonic=clock)
    events.clear()
    budget.tick()
    assert events == ['closed', 'failure', 'monotonic', 'clock', 'last', 'deadline', 'duration', 'deadline']
    events.clear()
    budget.available('rows', 1)
    assert events == ['closed', 'failure', 'monotonic', 'clock', 'last', 'last', 'deadline', 'deadline',
                      'limits', 'limits', 'counts']


@pytest.mark.parametrize('value,reason', [(5, 'reference_deadline_exceeded'),
                                         (-1, 'reference_clock_invalid'),
                                         (True, 'reference_clock_invalid'),
                                         (float('nan'), 'reference_clock_invalid')])
def test_owned_tick_refusal_clock_trace_and_sticky_failure(value, reason):
    values = iter([0, value])
    calls = []
    def clock():
        calls.append(True)
        return next(values)
    budget = Budget(monotonic=clock)
    budget.tick()
    with pytest.raises(ValueError, match=reason):
        budget.tick()
    before = dict(budget.counts)
    with pytest.raises(ValueError, match=reason):
        budget.available('rows', 1)
    assert len(calls) == 2 and budget.failure == reason and dict(budget.counts) == before


def test_current_registered_canonical_type_refuses_stale_identity_before_clock(monkeypatch):
    calls = []
    budget = Budget(monotonic=lambda: calls.append(True) or 0)
    monkeypatch.setitem(sys.modules, module.__name__, SimpleNamespace(ReferenceCollectionBudget=type('Other', (), {})))
    with pytest.raises(ValueError, match='retained_lineage_emission_limit'):
        lineage._work(budget)
    assert calls == []


@pytest.mark.parametrize('absent', [True, False])
def test_missing_or_none_module_uses_original_relative_class_import(monkeypatch, absent):
    calls, imported = [], []
    budget = Budget(monotonic=lambda: calls.append(True) or 0)
    original = builtins.__import__
    def observed(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'control_plane_reference_budget' and level == 1:
            imported.append((name, tuple(fromlist), level))
            return SimpleNamespace(ReferenceCollectionBudget=Budget)
        return original(name, globals, locals, fromlist, level)
    if absent:
        monkeypatch.delitem(sys.modules, module.__name__)
    else:
        monkeypatch.setitem(sys.modules, module.__name__, None)
    monkeypatch.setattr(builtins, '__import__', observed)
    lineage._work(budget)
    assert imported == [('control_plane_reference_budget', ('ReferenceCollectionBudget',), 1)]
    assert calls == [True]


@pytest.mark.slow
def test_fresh_none_path_does_not_import_reference_budget():
    code = '''import sys
from blueprint_pipeline import task_evaluation_scene_lineage_budget as m
assert 'blueprint_pipeline.control_plane_reference_budget' not in sys.modules
m._work(None)
m.RetainedEmissionBudget(max_bytes=100, max_rows=1, max_references=1).rows().append({'a': 1})
assert 'blueprint_pipeline.control_plane_reference_budget' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_fresh_reload_checks_new_canonical_class_not_cached_identity():
    code = '''import importlib
from blueprint_pipeline import control_plane_reference_budget as b
from blueprint_pipeline import task_evaluation_scene_lineage_budget as m
old = b.ReferenceCollectionBudget(monotonic=lambda: 0)
m._work(old)
importlib.reload(b)
try:
    m._work(old)
except ValueError as error:
    assert str(error) == 'retained_lineage_emission_limit'
else:
    raise AssertionError('stale canonical class admitted')
m._work(b.ReferenceCollectionBudget(monotonic=lambda: 0))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
