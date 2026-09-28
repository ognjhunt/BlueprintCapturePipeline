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


def test_scene_deadline_refuses_next_stat_and_cleans_owned_descriptors(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_acquisition as module
    current = [0]
    budget = Budget._for_scene_lifecycle_plan(monotonic=lambda: current[0])
    with module.Acquisition(budget, [str(tmp_path.resolve())]) as reader:
        current[0] = 30
        monkeypatch.setattr(module.os, 'stat', lambda *a, **k: pytest.fail('stat after scene deadline'))
        with pytest.raises(ReferenceCollectionBudgetError, match='^reference_deadline_exceeded$'):
            reader.stat(str(tmp_path / 'not-read.json'))
    assert not reader.handles and budget.failure == 'reference_deadline_exceeded'


@pytest.mark.parametrize('mode', ['native', 'reduced_scene'])
def test_private_builder_never_extends_supplied_clock_or_value_policy(tmp_path, mode):
    from tests.test_scene_lifecycle_plan import context_fixture
    from blueprint_pipeline.task_evaluation_scene_lifecycle_plan import _build_scene_lifecycle_plan
    context, intent = context_fixture(tmp_path, completed=True)
    current = [0]
    budget = (Budget(monotonic=lambda: current[0], values_limit=1) if mode == 'native'
              else Budget._for_scene_lifecycle_plan(monotonic=lambda: current[0], time_budget_seconds=0.25))
    if mode == 'reduced_scene':
        budget._limits['values'] = 1
    before = dict(budget.limits), budget.duration
    result = _build_scene_lifecycle_plan(intent_id=intent, context=context, observed_at_epoch=900000, budget=budget)
    assert result['blockers'] == ['reference_values_limit']
    assert (dict(budget.limits), budget.duration) == before and budget.closed


def test_deadline_before_final_publication_refuses_large_encoder(tmp_path, monkeypatch, capsys):
    import json
    from tests.test_scene_lifecycle_plan import context_fixture
    from tests.scene_lifecycle_fixture_support import stable_shared_ancestors
    from blueprint_pipeline import task_evaluation_scene_lifecycle_cli as cli
    current = [0]
    context, intent = context_fixture(tmp_path, completed=True)
    file = tmp_path / 'context.json'
    file.write_text(json.dumps(context))
    stable_shared_ancestors(monkeypatch, tmp_path)
    measured, encoded, entered = Budget.measure, json.dumps, []
    def expire_before_measure(self, value, **kwargs):
        if isinstance(value, dict) and value.get('schema_version') == 'task_evaluation_scene_lifecycle_plan.v1':
            entered.append(True)
            current[0] = self.deadline
        return measured(self, value, **kwargs)
    def refuse_large_encoder(value, *args, **kwargs):
        if isinstance(value, dict) and 'historical_lineage' in value:
            pytest.fail('accepted report encoded after deadline')
        return encoded(value, *args, **kwargs)
    monkeypatch.setattr(Budget, 'measure', expire_before_measure)
    monkeypatch.setattr(json, 'dumps', refuse_large_encoder)
    assert cli.main(['--intent-id', intent, '--context-file', str(file.resolve()), '--now', '900000'],
                    monotonic=lambda: current[0]) == 2
    report = json.loads(capsys.readouterr().out)
    assert entered and report['blockers'] == ['reference_deadline_exceeded']
    assert 'historical_lineage' not in report and report['action'] == 'KEEP'
