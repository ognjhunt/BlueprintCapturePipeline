"""Same work/clock semantics without per-character keyword dictionaries."""
import ast
import subprocess

import pytest

from blueprint_pipeline import task_evaluation_scene_downstream_contracts as contracts
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def original():
    raw = subprocess.check_output(['git', 'show',
        'b253116d6:src/blueprint_pipeline/task_evaluation_scene_downstream_contracts.py'], text=True)
    node = next(node for node in ast.parse(raw).body
                if isinstance(node, ast.FunctionDef) and node.name == 'bounded_size')
    namespace = dict(vars(contracts))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-bounded-size>', 'exec'), namespace)
    return namespace['bounded_size']


def test_original_measurement_oracle_does_not_require_checkout_history(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError('CI shallow checkout has no historical Git objects')
    monkeypatch.setattr(subprocess, 'check_output', unavailable)
    assert original()({'rows': [1, True, None]}, 4096) == 22


def observe(function, value, limit, *, fail_at=None):
    samples = []
    def clock():
        samples.append(len(samples))
        return float('nan') if len(samples) == fail_at else 0.0
    budget = ReferenceCollectionBudget(monotonic=clock)
    try:
        outcome = ('result', function(value, limit, work_budget=budget))
    except ValueError as error:
        outcome = (type(error).__name__, str(error))
    return outcome, dict(budget.counts), samples


def test_shared_measurement_does_not_allocate_keyword_dictionary_per_character(monkeypatch):
    allocations = []
    prior = contracts._work_kwargs
    def observed(budget):
        allocations.append(budget)
        return prior(budget)
    monkeypatch.setattr(contracts, '_work_kwargs', observed)
    budget = ReferenceCollectionBudget(monotonic=lambda:0)
    assert contracts.bounded_size('x' * 1024, 4096, work_budget=budget) == 1026
    assert len(allocations) == 0


@pytest.mark.parametrize('value,limit', [
    ('',2), ('\\"\n\t',100), ('é世😀',100), ('\ud800',100),
    ('x',0), ('x',2), ({'text':'x'*257,'rows':[1,True,None]},4096),
    ([float('inf')],100), ((1,2),100), ({1:'wrong-key'},100),
])
@pytest.mark.parametrize('fail_at', [None, 2, 20])
def test_shared_measurement_matches_original_refusal_counts_and_clock_trace(value,limit,fail_at):
    assert observe(contracts.bounded_size,value,limit,fail_at=fail_at) == observe(
        original(),value,limit,fail_at=fail_at)


def test_none_measurement_keeps_original_keyword_omission_and_values(monkeypatch):
    calls=[]
    prior=contracts.require
    def observed(condition,code,**kwargs):
        calls.append((condition,code,kwargs))
        return prior(condition,code,**kwargs)
    monkeypatch.setattr(contracts,'require',observed)
    expected=original()({'text':'é\\\n','rows':[1,False,None]},4096)
    old=list(calls)
    calls.clear()
    assert contracts.bounded_size({'text':'é\\\n','rows':[1,False,None]},4096) == expected
    assert calls == old
    assert all(not kwargs for _,_,kwargs in calls)
