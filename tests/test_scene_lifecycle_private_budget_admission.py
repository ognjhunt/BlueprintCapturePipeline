# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_plan.py
"""ADP-009D: reject unowned private budgets before their callbacks or cleanup."""
import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget as Budget
from blueprint_pipeline import task_evaluation_scene_lifecycle_plan as planner


def invoke(budget, context_anchor=None):
    return planner._build_scene_lifecycle_plan(intent_id='intent-1', context=None,
        observed_at_epoch=0, budget=budget, context_anchor=context_anchor)


@pytest.mark.parametrize('kind', ['lookalike', 'subclass', 'none', 'empty_exact', 'marker_exact', 'partial_exact'])
def test_invalid_private_budget_is_fixed_keep_without_candidate_or_context_callbacks(monkeypatch, kind):
    calls = []
    class Callbacks:
        def tick(self):
            calls.append('tick')
        @property
        def failure(self):
            calls.append('failure')
            return None
        def close(self):
            calls.append('close')
    class Subclass(Callbacks, Budget):
        pass
    budget = {'lookalike': lambda: Callbacks(), 'subclass': lambda: Subclass(monotonic=lambda: 0),
              'none': lambda: None, 'empty_exact': lambda: object.__new__(Budget),
              'marker_exact': lambda: object.__new__(Budget), 'partial_exact': lambda: object.__new__(Budget)}[kind]()
    if kind in ('marker_exact', 'partial_exact'):
        budget._initialization_started = True
    if kind == 'partial_exact':
        budget._monotonic, budget._duration = lambda: calls.append('clock'), 5.0
    monkeypatch.setattr(planner, '_context', lambda *a: pytest.fail('context touched before admission'))
    monkeypatch.setattr(Budget, '_for_scene_lifecycle_plan', lambda **k: pytest.fail('replacement budget allocated'))
    result = invoke(budget, context_anchor=Callbacks())
    assert calls == []
    assert result == planner.fallback('scene_lifecycle_budget_invalid')
    assert result['action'] == 'KEEP' and result['mutations'] == 0


def test_real_partial_initialization_fault_is_not_admitted_or_closed(monkeypatch):
    budget = object.__new__(Budget)
    original = Budget.__setattr__
    def interrupted(self, name, value):
        if name == '_counts':
            raise MemoryError
        original(self, name, value)
    with monkeypatch.context() as fault:
        fault.setattr(Budget, '__setattr__', interrupted)
        with pytest.raises(MemoryError):
            budget.__init__(monotonic=lambda: pytest.fail('partial object clocked'))
    assert budget._initialization_started and '_counts' not in vars(budget)
    before = dict(vars(budget))
    assert invoke(budget) == planner.fallback('scene_lifecycle_budget_invalid')
    assert vars(budget) == before


@pytest.mark.parametrize('state', ['closed', 'failed'])
def test_valid_owned_closed_or_failed_budget_keeps_sticky_state_and_counters(state):
    calls = []
    budget = Budget(monotonic=lambda: calls.append(True) or 0)
    budget.charge('rows', 1)
    if state == 'closed':
        budget.close()
        expected = 'reference_budget_closed'
    else:
        expected = 'reference_values_limit'
        with pytest.raises(ValueError, match=expected):
            budget.fail(expected)
    before = (dict(budget.counts), budget.deadline, budget.last, len(calls), budget._counts, budget._limits)
    result = invoke(budget)
    assert result == planner.fallback(expected)
    assert budget.failure == expected and budget.closed
    assert (dict(budget.counts), budget.deadline, budget.last, len(calls)) == before[:4]
    assert budget._counts is before[4] and budget._limits is before[5]
