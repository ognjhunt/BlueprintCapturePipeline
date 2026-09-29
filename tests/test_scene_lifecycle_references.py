# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_references.py
"""One B, historical references only and explicit conservative input charges."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
from tests.test_scene_lifecycle_plan import context_fixture
from tests.test_preparation_activation_reference_records import record


def fixture(tmp_path):
    context, _ = context_fixture(tmp_path)
    original = record(state='pending')
    root = context['roots']['preparation_queue_root']
    actual = replace(original, queue_root=root, row_path=original.row_path.replace(original.queue_root, root))
    target = Path(actual.row_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(actual.raw_bytes)
    context['primary_queue_contracts'][0]['states'] = ['pending']
    return context, actual


def invoke(context, budget):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_references import observe
    sink = RetainedEmissionBudget(max_bytes=16*1024*1024, max_rows=10_000, max_references=10_000, work_budget=budget)
    return observe(context, 1000, budget, sink)


def test_actual_reader_bytes_and_supplied_input_work_are_separate(tmp_path):
    context, actual = fixture(tmp_path)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    result = invoke(context, budget)
    accounting = result['raw_accounting']
    assert accounting['reader_raw_bytes_by_child']['primary_queues'] == len(actual.raw_bytes)
    assert accounting['supplied_reference_input_work_bytes'] == len(actual.raw_bytes)
    assert accounting['conservative_accounting'] is True
    assert accounting['cumulative_raw_allowance_bytes'] == 2*len(actual.raw_bytes)
    assert result['historical_only'] and not result['references_clear']
    assert result['action'] == 'KEEP'


def test_reduced_shared_remainder_refuses_before_extra_input_charge(tmp_path):
    context, actual = fixture(tmp_path)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    budget.charge('raw_bytes', budget.limits['raw_bytes'] - len(actual.raw_bytes) - 1)
    result = invoke(context, budget)
    assert result['raw_accounting']['reader_raw_bytes_by_child']['primary_queues'] == len(actual.raw_bytes)
    assert result['raw_accounting']['supplied_reference_input_work_bytes'] == 0
    assert 'reference_raw_bytes_limit' in result['blockers']
    assert result['status'] == 'incomplete' and result['action'] == 'KEEP'


def test_refusal_after_input_charge_keeps_actual_charged_delta(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_preparation_activation_references as module
    context, actual = fixture(tmp_path)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    from blueprint_pipeline import task_evaluation_scene_lifecycle_references as consumer
    original = consumer.interpret_preparation_activation_references
    def fail_after_charge(self, raw):
        self.shared.fail('reference_values_limit')
    def interpret(*args, **kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(module._Scan, 'parse', fail_after_charge)
            return original(*args, **kwargs)
    monkeypatch.setattr(consumer, 'interpret_preparation_activation_references', interpret)
    result = invoke(context, budget)
    assert result['raw_accounting']['supplied_reference_input_work_bytes'] == len(actual.raw_bytes)
    assert result['status'] == 'incomplete' and result['references_clear'] is False


def test_pin_positive_path_is_kept_and_intersects_componentwise_measured_member(tmp_path):
    from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
    from blueprint_pipeline.task_evaluation_scene_lifecycle_references import intersect
    context, _ = context_fixture(tmp_path)
    member = context['roots']['preparation_input_root']+'/prep-1'
    write_storage_pin(pins_root=context['pins_root'], kind='preparation', owner_id='owner-1', paths=[member+'/payload.bin'],
                      ttl_seconds=1000, now=lambda: 1000)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    sink = RetainedEmissionBudget(max_bytes=16*1024*1024, max_rows=10_000, max_references=10_000, work_budget=budget)
    from blueprint_pipeline.task_evaluation_scene_lifecycle_references import observe
    observed = observe(context, 1000, budget, sink)
    keeps = intersect([{'path': member}, {'path': member+'-foreign'}], observed, budget, sink)
    assert len(keeps) == 1 and keeps[0]['member_path'] == member
    assert keeps[0]['reason'] == 'positive_historical_reference'
    assert keeps[0]['protected_path'] == member+'/payload.bin' and keeps[0]['action'] == 'KEEP'


def test_intersect_reuses_exact_measured_index_without_losing_keep_or_double_charging():
    from blueprint_pipeline.task_evaluation_scene_lifecycle_references import intersect
    members = [{'path': '/scene/owned'}, {'path': '/scene/other'}]
    observed = {'protections': [{'kind': 'positive_pin_path', 'path': '/scene/owned/file'}]}

    def run(index=None):
        budget = ReferenceCollectionBudget(monotonic=lambda: 0)
        sink = RetainedEmissionBudget(max_bytes=16*1024*1024, max_rows=10_000,
                                      max_references=10_000, work_budget=budget)
        result = intersect(members, observed, budget, sink, measured_by_path=index)
        return result, budget.counts['facts']

    original, original_facts = run()
    reused, reused_facts = run({row['path']: row for row in members})
    assert reused == original
    assert original_facts - reused_facts == len(members)
    with pytest.raises(ValueError):
        run({'/scene/owned': members[1], '/scene/other': members[0]})
