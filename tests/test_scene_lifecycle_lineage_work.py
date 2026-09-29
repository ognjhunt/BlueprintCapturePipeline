# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owner_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_lineage_budget.py
"""ADP-009D: one optional work allowance reaches every retained child."""
import json

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
    actual = invoke(args, budget)
    # The private path records predecessor references once under their owning
    # child. The legacy public report repeats them in enclosing children, where
    # the same target can be unresolved because that child lacks its role.
    # Preserve all members and every target/source byte selector despite that
    # grouping difference; comparing only totals would miss a lost reference.
    def observable(value, selectors):
        if isinstance(value, dict):
            retained = {}
            for key, child in value.items():
                if key in {'raw_reference_obligations', 'remote_reference_obligations'}:
                    for row in child:
                        target = ({name: row[name] for name in ('path', 'sha256', 'size_bytes')}
                                  if key == 'raw_reference_obligations' else
                                  {name: row[name] for name in ('uri', 'digest', 'size_bytes')})
                        for source in row['source_provenance']:
                            raw_source = {name: source[name] for name in ('role', 'path', 'sha256', 'size_bytes')}
                            selectors.add(json.dumps([key, target, raw_source], sort_keys=True))
                else:
                    retained[key] = observable(child, selectors)
            return retained
        if isinstance(value, list):
            return [observable(child, selectors) for child in value]
        return value
    native_selectors, private_selectors = set(), set()
    assert observable(actual, private_selectors) == observable(expected, native_selectors)
    assert private_selectors == native_selectors
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
                  and fn.__module__ == module.__name__ and not fn.__name__.startswith(('join_retained_', 'join_scene_'))]
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


def test_final_sort_refuses_after_pair_construction_before_sorted(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as c
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.charge('values', budget.limits['values'] - 4)
    monkeypatch.setattr(c, 'bounded_size', lambda *a, **k: 0)
    monkeypatch.setattr(c, 'encoded', lambda row, **k: bytes([row['a']]))
    monkeypatch.setattr(c, 'sorted', lambda *a, **k: pytest.fail('unguarded final sorting allocation'), raising=False)
    with pytest.raises(ValueError, match='reference_values_limit'):
        c.unique([{'a': 1}, {'a': 2}, {'a': 3}], 1000, work_budget=budget)


@pytest.mark.parametrize('module,name', [('preparation_lineage', 'join_scene_preparation_lineage'),
                                         ('source_attempt_lineage', 'join_scene_source_attempt_lineage')])
def test_older_public_signatures_omit_private_work_allowance(module, name):
    import importlib
    import inspect
    function = getattr(importlib.import_module('blueprint_pipeline.task_evaluation_scene_' + module), name)
    assert 'work_budget' not in inspect.signature(function).parameters


def test_private_source_branch_refuses_live_work_without_emission_sink():
    from blueprint_pipeline import task_evaluation_scene_inventory_seed as seed
    from tests.test_scene_inventory_history import fixture as history_fixture
    args = history_fixture()
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    with pytest.raises(ValueError, match='scene_inventory_parameters_invalid'):
        seed._sources(args['intent_id'], args['records'], args['roots'], set(), [], work_budget=budget)
# A real dictionary requires three measurement nodes, three native traversal
# nodes, two guarded child consumptions and one scalar-encoding measurement.
# Duplicate wrappers must not invent
# two more consumptions and exhaust this exact allowance.
def test_bounded_size_uses_one_guard_for_each_actual_child_consumption():
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.task_evaluation_scene_downstream_contracts import bounded_size
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.charge('values', budget.limits['values'] - 9)
    assert bounded_size({'a': 1}, 100, work_budget=budget) == 7
    assert budget.counts['values'] == budget.limits['values']
    assert bounded_size({'a': 1}, 100) == 7
