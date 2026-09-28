# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_reference_budget.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_plan.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_cli.py
"""Finite scene policy starts once; native observations retain their defaults."""
import math

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget as Budget
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudgetError


def test_scene_factory_is_fresh_exact_type_with_fixed_initial_policy():
    def clock():
        return 0
    budget = Budget._for_scene_lifecycle_plan(monotonic=clock)
    assert type(budget) is Budget
    assert budget.monotonic is clock and budget.duration == 30
    assert budget.limits['values'] == 1_000_000
    native = Budget(monotonic=clock)
    assert native.duration == 5 and native.limits['values'] == 100_000
    assert {k: v for k, v in budget.limits.items() if k != 'values'} == {k: v for k, v in native.limits.items() if k != 'values'}
    assert all(value == 0 for value in budget.counts.values())
    assert budget.last is budget.deadline is budget.failure is None and not budget.closed
    assert Budget(values_limit=1).limits['values'] == 1
    assert Budget(values_limit=10_000).limits['values'] == 10_000


@pytest.mark.parametrize('duration', [False, 0, -1, math.nan, math.inf, 30.0001, '30'])
def test_scene_factory_refuses_invalid_duration_before_clock(duration):
    entered = []
    with pytest.raises(ReferenceCollectionBudgetError, match='^reference_budget_parameters_invalid$'):
        Budget._for_scene_lifecycle_plan(monotonic=lambda: entered.append(True), time_budget_seconds=duration)
    assert not entered


def test_native_constructor_still_refuses_more_than_five_seconds():
    with pytest.raises(ReferenceCollectionBudgetError, match='^reference_budget_parameters_invalid$'):
        Budget(time_budget_seconds=5.001)


@pytest.mark.parametrize('state', ['unused', 'consumed', 'failed', 'closed'])
@pytest.mark.parametrize('entry', ['constructor', 'initializer'])
def test_reentry_cannot_mutate_used_or_unused_budget(state, entry):
    calls = []
    def clock():
        calls.append(True)
        return 0
    budget = Budget(monotonic=clock)
    if state == 'consumed':
        budget.charge('roots')
    elif state == 'failed':
        with pytest.raises(ReferenceCollectionBudgetError):
            budget.charge('values', 100_001)
    elif state == 'closed':
        budget.close()
    objects = dict(budget.__dict__)
    before = (dict(budget.limits), dict(budget.counts), budget.duration, budget.last,
              budget.deadline, budget.failure, budget.closed, set(budget.blockers), len(calls))
    with pytest.raises(ReferenceCollectionBudgetError, match='^reference_budget_parameters_invalid$'):
        if entry == 'constructor':
            budget.__init__(monotonic=lambda: 50)
        else:
            budget._initialize_validated(monotonic=lambda: 50, duration=30, values_limit=1_000_000)
    after = (dict(budget.limits), dict(budget.counts), budget.duration, budget.last,
             budget.deadline, budget.failure, budget.closed, set(budget.blockers), len(calls))
    assert before == after
    assert all(budget.__dict__[key] is value for key, value in objects.items())


def test_partial_initialization_never_retries_or_returns_from_factory(monkeypatch):
    tokens = []
    def fail_after_marker(self, name, value):
        if name == '_counts':
            tokens.append(self)
            raise MemoryError('test-only fixed initialization fault')
        object.__setattr__(self, name, value)
    monkeypatch.setattr(Budget, '__setattr__', fail_after_marker)
    with pytest.raises(MemoryError):
        Budget._for_scene_lifecycle_plan()
    assert len(tokens) == 1
    partial = tokens[0]
    assert partial._initialization_started is True
    before = dict(partial.__dict__)
    with pytest.raises(ReferenceCollectionBudgetError, match='^reference_budget_parameters_invalid$'):
        partial.__init__()
    assert partial.__dict__ == before and len(tokens) == 1


def test_scene_factory_rejects_existing_budget_arbitrary_cap_and_subclass():
    class Subclass(Budget):
        pass
    with pytest.raises(ReferenceCollectionBudgetError, match='^reference_budget_parameters_invalid$'):
        Subclass._for_scene_lifecycle_plan()
    with pytest.raises(TypeError):
        Budget._for_scene_lifecycle_plan(budget=Budget())
    with pytest.raises(TypeError):
        Budget._for_scene_lifecycle_plan(values_limit=2_000_000)
