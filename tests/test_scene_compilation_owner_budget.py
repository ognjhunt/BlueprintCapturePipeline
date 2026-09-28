# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_source_family_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owner_inventory.py
"""ADP-009D/day28: external private composition preserves native ceilings."""
import pytest

from tests.test_scene_source_family_website import fixture


def invoke(args, budget):
    from blueprint_pipeline import task_evaluation_scene_source_family_inventory as prior
    return prior._join(args['intent_id'], args['seed_records'], args['downstream_records'], args['source_records'],
        args['roots'], args['parent_routes'], args['retained_metadata_roots'], emission_budget=budget)


def test_exhausted_parent_sink_refuses_before_any_next_child(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_source_family_inventory as prior
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget, RetainedEmissionBudgetError
    budget = RetainedEmissionBudget(max_bytes=1024*1024, max_rows=0, max_references=1000)
    monkeypatch.setattr(prior, '_downstream_join', lambda **kw: pytest.fail('child called after exhausted parent'))
    with pytest.raises(RetainedEmissionBudgetError):
        invoke(fixture(), budget)
    assert budget.used['rows'] == 0


def test_larger_parent_cannot_weaken_child_native_rows_and_defaults_remain_public(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_source_family_inventory as prior
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    args = fixture()
    expected = prior.join_retained_scene_source_family_inventory(**args)
    budget = RetainedEmissionBudget(max_bytes=32*1024*1024, max_rows=20_000, max_references=20_000)
    assert invoke(args, budget) == expected and budget.used['rows'] > 0
    monkeypatch.setattr(prior, 'MAX_ROWS', 0)
    with pytest.raises(ValueError):
        invoke(args, RetainedEmissionBudget(max_bytes=32*1024*1024, max_rows=20_000, max_references=20_000))
