# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owner_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_lineage_budget.py
"""ADP-009D: one optional work allowance reaches every retained child."""
import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from tests.test_scene_compilation_owner_preparations import fixture


def invoke(args, budget):
    from blueprint_pipeline import task_evaluation_scene_compilation_native_owner_inventory as m
    return m._join(args['intent_id'], args['seed_records'], args['downstream_records'],
                   args['source_records'], args['bridge_records'], args['roots'],
                   args['parent_routes'], args['retained_metadata_roots'], work_budget=budget)


def test_private_work_path_matches_native_and_uses_one_budget():
    from blueprint_pipeline import task_evaluation_scene_compilation_native_owner_inventory as m
    args = fixture()
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    expected = m.join_retained_scene_compilation_native_owner_inventory(**args)
    assert invoke(args, budget) == expected
    assert budget.counts['values'] > 0 and budget.counts['facts'] > 0
    assert budget.counts['raw_bytes'] == 0  # Already-acquired bytes are not new disk I/O.


def test_closed_budget_stops_before_any_parser(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as c
    args = fixture()
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.close()
    monkeypatch.setattr(c.json, 'loads', lambda *a, **k: pytest.fail('parser after close'))
    with pytest.raises(ValueError, match='reference_budget_closed'):
        invoke(args, budget)


def test_emission_scope_checks_shared_occurrences_before_measure(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_lineage_budget as m
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.charge('rows', budget.limits['rows'])
    sink = m.RetainedEmissionBudget(max_bytes=1000, max_rows=100, max_references=100,
                                   work_budget=budget)
    monkeypatch.setattr(m, '_measure', lambda *a, **k: pytest.fail('measure past shared row cap'))
    with pytest.raises(ValueError, match='reference_rows_limit'):
        sink.scope(max_bytes=900, max_rows=90, max_references=90).reserve_row({'a': 1})


def test_none_path_does_not_import_resource_budget(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_lineage_budget as m
    sink = m.RetainedEmissionBudget(max_bytes=1000, max_rows=100, max_references=100)
    monkeypatch.setattr(ReferenceCollectionBudget, 'tick', lambda *a: pytest.fail('None ticked'))
    sink.rows().append({'a': 1})
    assert sink.used['rows'] == 1

MODULES = ('preparation_lineage', 'source_attempt_lineage', 'inventory_seed',
           'downstream_contracts', 'downstream_inventory', 'downstream_execution', 'downstream_terminal',
           'source_family_contracts', 'source_family_inventory', 'source_family_website', 'source_family_sam',
           'source_family_adoption', 'compilation_owner_contracts', 'compilation_native_owner_inventory',
           'compilation_owner_preparations', 'compilation_owner_outputs', 'compilation_native_owners')


@pytest.mark.parametrize('suffix', MODULES)
def test_every_owned_private_helper_stops_at_closed_work_boundary(suffix):
    import importlib
    import inspect
    from types import SimpleNamespace
    module = importlib.import_module('blueprint_pipeline.task_evaluation_scene_' + suffix)
    candidates = [fn for fn in vars(module).values() if inspect.isfunction(fn)
                  and fn.__module__ == module.__name__ and not fn.__name__.startswith('join_retained_')]
    for cls in vars(module).values():
        if inspect.isclass(cls) and cls.__module__ == module.__name__:
            candidates.extend(fn.fget if isinstance(fn, property) else fn for fn in vars(cls).values()
                              if inspect.isfunction(fn) or isinstance(fn, property))
    assert candidates
    for fn in candidates:
        signature = inspect.signature(fn)
        assert 'work_budget' in signature.parameters, fn.__qualname__
        budget = ReferenceCollectionBudget(monotonic=lambda: 0)
        budget.close()
        arguments = {name: SimpleNamespace() if name == 'self' else None
                     for name, value in signature.parameters.items()
                     if value.default is inspect.Parameter.empty and
                     value.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
        with pytest.raises(ValueError, match='reference_budget_closed'):
            fn(**arguments, work_budget=budget)


def test_same_invocation_work_budget_reaches_every_lineage_module(monkeypatch):
    import importlib
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    observed = set()
    for suffix in MODULES:
        module = importlib.import_module('blueprint_pipeline.task_evaluation_scene_' + suffix)
        original = module._work
        def guard(actual, *, suffix=suffix, original=original):
            assert actual is budget
            observed.add(suffix)
            original(actual)
        monkeypatch.setattr(module, '_work', guard)
    invoke(fixture(), budget)
    assert observed == set(MODULES)


def test_shared_row_exhaustion_stops_before_next_generator_item():
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.charge('rows', budget.limits['rows'])
    sink = RetainedEmissionBudget(max_bytes=1000, max_rows=100, max_references=100, work_budget=budget)
    def values():
        pytest.fail('expanded generator after shared row exhaustion')
        yield {}
    with pytest.raises(ValueError, match='reference_rows_limit'):
        sink.rows(values())


def test_shared_sort_guard_stops_before_sort_allocation(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as c
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.charge('values', budget.limits['values'])
    monkeypatch.setattr(c, 'sorted', lambda *a, **k: pytest.fail('sorted allocated past work cap'), raising=False)
    with pytest.raises(ValueError, match='reference_values_limit'):
        c.unique([], 1000, work_budget=budget)


def test_sort_refuses_before_retaining_fanout():
    from blueprint_pipeline import task_evaluation_scene_lineage_budget as m
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.charge('values', budget.limits['values'] - 1)
    def source():
        yield 'first'
        pytest.fail('expanded second sorting value after quota')
        yield 'second'
    with pytest.raises(ValueError, match='reference_values_limit'):
        m._work_order(budget, sorted, source())
