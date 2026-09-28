# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_plan.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_pool.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_acquisition.py
"""ADP-009D: actual bounded local acquisition produces KEEP-only plans."""
from __future__ import annotations

import json
from pathlib import Path

from tests.test_scene_inventory_history import fixture, event, project
from tests.test_scene_compilation_owner_preparations import fixture as owner_fixture


def context_fixture(tmp_path, *, completed=False):
    base = tmp_path.resolve()
    old = owner_fixture()
    roots = {key: value.replace('/retained', str(base)) for key, value in old['roots'].items()}
    for root in roots.values():
        Path(root).mkdir(parents=True, exist_ok=True)
    history = fixture()
    if completed:
        tail = event(history, updates={'status': 'completed', 'phase': 'terminal'})
        project(history, tail)
    for role, pairs in history['records'].items():
        pairs = ([] if pairs is None else [pairs]) if role in {'intent', 'projection'} else pairs
        for oldpath, raw in pairs:
            target = Path(oldpath.replace('/retained', str(base)))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    context = {'roots': roots, 'parent_routes': [{'queue_root': roots['preparation_queue_root'],
                'input_root': roots['preparation_input_root']}], 'retained_metadata_roots': [str(base / 'metadata')],
               'acquisition_anchors': [str(base)], 'retained_metadata_files': [],
               'pins_root': str(base / 'pins'),
               'primary_queue_contracts': [{'root_path': roots['preparation_queue_root'],
                   'states': ['pending', 'processing', 'awaiting_source_preparation', 'awaiting_capacity', 'materialized', 'completed', 'blocked']}],
               'auxiliary_queue_contracts': [{'family': 'preparation', 'root_path': roots['preparation_queue_root']},
                                            {'family': 'sam', 'root_path': roots['sam_queue_root']}],
               'reference_family_contracts': [{'family': 'preparation', 'queue_root': roots['preparation_queue_root']},
                                             {'family': 'activation', 'queue_root': roots['activation_queue_root']}],
               'progression_config': None}
    return context, history['intent_id']


def run(context, intent_id):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_plan import build_scene_lifecycle_plan
    return build_scene_lifecycle_plan(intent_id=intent_id, context=context, observed_at_epoch=900000,
                                      monotonic=lambda: 0)


def test_real_history_acquired_and_finished_observed_without_cleanup_authority(tmp_path):
    context, intent_id = context_fixture(tmp_path, completed=True)
    report = run(context, intent_id)
    assert report['schema_version'] == 'task_evaluation_scene_lifecycle_plan.v1'
    assert report['finished_observation']['status'] == 'completed'
    assert report['selected_intent_provenance']['path'] == context['roots']['intent_root'] + '/' + intent_id + '/intent.json'
    assert report['historical_lineage']['source_family_inventory']['downstream_inventory']['seed']['history']['chain_validated']
    assert report['mutations'] == 0 and report['action'] == 'KEEP'
    for flag in ('scene_inventory_complete', 'references_clear', 'process_fences_held', 'retirement_eligible',
                 'cleanup_authorized', 'restore_verified', 'fresh_remote_readback_verified'):
        assert report[flag] is False


def test_missing_history_is_unknown_and_family_absence_is_not_zero_bytes(tmp_path):
    context, intent_id = context_fixture(tmp_path)
    report = run(context, intent_id)
    assert report['finished_observation']['status'] == 'unknown'
    assert len(report['family_obligations']) == 10
    assert all(row['measured_allocated_bytes'] is None for row in report['family_obligations'])
    assert all(row['action'] == 'KEEP' for row in report['family_obligations'])


def test_raw_context_values_never_enter_typed_refusal(tmp_path):
    context, intent_id = context_fixture(tmp_path)
    context['unexpected-secret-key'] = 'secret value'
    report = run(context, intent_id)
    assert report['status'] == 'incomplete' and report['action'] == 'KEEP'
    assert 'secret' not in json.dumps(report)


def preparation_fixture(tmp_path):
    from tests.test_scene_inventory_preparations import fixture as prepared_fixture, seal
    context, intent_id = context_fixture(tmp_path)
    args = prepared_fixture()
    for role, rows in args['records'].items():
        if role in {'intent', 'projection'}:
            continue
        for oldpath, raw in rows:
            value = json.loads(raw)
            if role == 'preparation_results':
                for reference in value['references']:
                    reference['materialized_path'] = reference['materialized_path'].replace('/retained', str(tmp_path.resolve()))
                value = seal(value, 'result_digest')
            target = Path(oldpath.replace('/retained', str(tmp_path.resolve())))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value, sort_keys=True))
    member = Path(context['roots']['preparation_input_root']) / 'prep-1'
    member.mkdir()
    (member / 'payload.bin').write_bytes(b'tiny')
    return context, intent_id, member


def test_measures_exact_preparation_member_without_opening_payload(tmp_path, monkeypatch):
    import os
    context, intent_id, member = preparation_fixture(tmp_path)
    original = os.open
    def opened(name, *args, **kwargs):
        assert str(name) != 'payload.bin'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(os, 'open', opened)
    report = run(context, intent_id)
    row = next(row for row in report['measured_members'] if row['path'] == str(member))
    assert row['status'] == 'observed_scoped_metadata' and row['measured_allocated_bytes'] > 0
    assert report['unique_observed_allocated_bytes'] == row['measured_allocated_bytes']
    assert row['action'] == 'KEEP' and row['exclusive_ownership_proven'] is False


def test_hardlink_accounted_once_and_external_link_keeps_member(tmp_path):
    import os
    context, intent_id, member = preparation_fixture(tmp_path)
    os.link(member / 'payload.bin', member / 'copy.bin')
    os.link(member / 'payload.bin', tmp_path / 'outside.bin')
    report = run(context, intent_id)
    row = next(row for row in report['measured_members'] if row['path'] == str(member))
    assert row['observed_regular_names'] == 2
    assert row['observed_unique_regular_inodes'] == 1
    assert 'external_hardlink_or_unobserved_alias' in row['keeps']
    assert row['measured_allocated_bytes'] == member.stat().st_blocks * 512 + (member / 'payload.bin').stat().st_blocks * 512


def test_measurement_reserves_each_member_before_building_the_next(tmp_path, monkeypatch):
    import pytest
    from blueprint_pipeline import task_evaluation_scene_lifecycle_measurement as m
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import Acquisition
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    sink = RetainedEmissionBudget(max_bytes=100000, max_rows=100, max_references=100, work_budget=budget)
    first, second = tmp_path.resolve() / 'first', tmp_path.resolve() / 'second'
    first.write_bytes(b'1')
    second.write_bytes(b'2')
    monkeypatch.setattr(m, 'members', lambda *a: {str(path): [{'kind': 'preparation_projected_file', 'source_provenance': []}]
                                                for path in (first, second)})
    budget.charge('rows', budget.limits['rows'] - 1)
    with Acquisition(budget, [str(tmp_path.resolve())]) as reader:
        original = reader.stat
        def checked(path):
            assert path != str(second), 'next member built before shared row refusal'
            return original(path)
        monkeypatch.setattr(reader, 'stat', checked)
        with pytest.raises(ValueError, match='reference_rows_limit'):
            m.measure(reader, {}, sink, [])


def test_measurement_reserves_output_before_next_member(tmp_path, monkeypatch):
    import pytest
    from blueprint_pipeline import task_evaluation_scene_lifecycle_measurement as m
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import Acquisition
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    sink = RetainedEmissionBudget(max_bytes=1, max_rows=100, max_references=100, work_budget=budget)
    first, second = tmp_path.resolve() / 'first', tmp_path.resolve() / 'second'
    first.write_bytes(b'1')
    second.write_bytes(b'2')
    monkeypatch.setattr(m, 'members', lambda *a: {str(path): [{'kind': 'preparation_projected_file', 'source_provenance': []}]
                                                for path in (first, second)})
    with Acquisition(budget, [str(tmp_path.resolve())]) as reader:
        original = reader.stat
        def checked(path):
            assert path != str(second), 'next member built before shared output refusal'
            return original(path)
        monkeypatch.setattr(reader, 'stat', checked)
        with pytest.raises(ValueError, match='retained_lineage_emission_limit'):
            m.measure(reader, {}, sink, [])
