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
    assert row['observed_allocated_bytes'] > 0
    assert report['unique_observed_allocated_bytes'] == row['observed_allocated_bytes']
    if 'metadata_changed_after_observation' in report['blockers']:
        assert row['status'] == 'incomplete_scoped_metadata' and row['measured_allocated_bytes'] is None
    else:
        assert row['status'] == 'observed_scoped_metadata'
        assert row['measured_allocated_bytes'] == row['observed_allocated_bytes']
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
    from blueprint_pipeline.control_plane_disk_usage import allocated_bytes
    assert row['observed_allocated_bytes'] == allocated_bytes(member.stat()) + allocated_bytes((member / 'payload.bin').stat())
    if 'metadata_changed_after_observation' in report['blockers']:
        assert row['measured_allocated_bytes'] is None
    else:
        assert row['measured_allocated_bytes'] == row['observed_allocated_bytes']


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


def test_later_metadata_drift_keeps_accepted_measurement_with_null_current_total(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import Acquisition, AcquisitionError
    context, intent_id, member = preparation_fixture(tmp_path)
    def changed(self):
        raise AcquisitionError('scene_lifecycle_metadata_changed')
    monkeypatch.setattr(Acquisition, 'verify', changed)
    report = run(context, intent_id)
    row = next(row for row in report['measured_members'] if row['path'] == str(member))
    assert report['status'] == 'incomplete' and report['action'] == 'KEEP'
    assert row['observed_allocated_bytes'] > 0 and row['measured_allocated_bytes'] is None
    assert row['measured_logical_bytes'] is None and row['measured_apparent_bytes'] is None
    assert 'metadata_changed_after_observation' in row['keeps']
    assert report['historical_lineage']['mutations'] == 0


def test_deadline_finalization_uses_fixed_refusal_without_new_traversal(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import Acquisition
    context, intent_id, _ = preparation_fixture(tmp_path)
    def expired(self):
        self.budget.fail('reference_deadline_exceeded')
    monkeypatch.setattr(Acquisition, 'verify', expired)
    report = run(context, intent_id)
    assert 'measured_members' not in report
    assert report['blockers'] == ['reference_deadline_exceeded']
    assert report['action'] == 'KEEP' and report['cleanup_authorized'] is False


def test_sparse_or_inline_zero_block_metadata_uses_existing_conservative_method():
    from types import SimpleNamespace
    from blueprint_pipeline.task_evaluation_scene_lifecycle_measurement import allocated
    assert allocated(SimpleNamespace(st_blocks=0, st_size=4096)) == 4096
    assert allocated(SimpleNamespace(st_blocks=None, st_size=4096)) == 4096
    assert allocated(SimpleNamespace(st_blocks=8, st_size=1)) == 4096


def test_exact_shared_cache_object_is_measured_without_scanning_store(tmp_path):
    context, intent_id, member = preparation_fixture(tmp_path)
    import os
    from tests.test_scene_inventory_preparations import fixture as prepared
    receipt = json.loads(prepared()['records']['preparation_results'][0][1])['references'][0]
    cache = Path(context['roots']['content_store_root']) / receipt['digest'][7:]
    os.link(member/'payload.bin', cache)
    (cache.parent/'unrelated-payload.bin').write_bytes(b'unrelated')
    report = run(context, intent_id)
    row = next(row for row in report['measured_members'] if row['path'] == str(cache))
    assert 'prepared_cache_object' in row['kinds'] and 'shared_content_object_not_exclusive' in row['keeps']
    assert row['payload_bytes_verified'] is False and row['exclusive_ownership_proven'] is False
    assert all(row['path'] != str(cache.parent) and 'unrelated-payload' not in row['path']
               for row in report['measured_members'])


def test_empty_output_allowance_refuses_before_first_member_stat(tmp_path, monkeypatch):
    import pytest
    from blueprint_pipeline import task_evaluation_scene_lifecycle_measurement as m
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    context, _, member = preparation_fixture(tmp_path)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    sink = RetainedEmissionBudget(max_bytes=1, max_rows=10, max_references=10, work_budget=budget)
    monkeypatch.setattr(m, 'members', lambda *args: {str(member): [{'kind': 'preparation_workspace', 'source_provenance': []}]})
    class Reader:
        def stat(self, path):
            pytest.fail('measurement entered first stat with no framing allowance')
        def entries(self, path):
            pytest.fail('measurement entered child walk with no framing allowance')
    reader = Reader()
    reader.budget = budget
    with pytest.raises(ValueError):
        m.measure(reader, {}, sink, [])


def test_future_required_history_keeps_raw_and_blocks_cropped_pure_join(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_plan as planner
    context, intent_id = context_fixture(tmp_path, completed=True)
    event = next((Path(context['roots']['intent_root'])/intent_id/'progression-events').glob('*.json'))
    value = json.loads(event.read_bytes())
    value['schema_version'] = 'future_progression_event.v99'
    event.write_text(json.dumps(value))
    monkeypatch.setattr(planner.native, '_join', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('strict join was invoked with cropped history')))
    report = run(context, intent_id)
    assert 'historical_lineage' not in report
    assert 'strict_lineage_join_unavailable' in report['blockers']
    assert report['finished_observation']['status'] == 'unknown'
    assert report['reference_observation']['historical_only']
    assert any(row['path'] == str(event) and row['status'] == 'kept_unsupported_schema'
               for row in report['unselected_metadata_protections'])
    assert report['action'] == 'KEEP' and report['cleanup_authorized'] is False


def test_public_plan_secret_shaped_projection_refuses_without_returning_identity(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_plan as planner
    context, intent_id = context_fixture(tmp_path)
    original = planner.native._join
    def secret(*a, **k):
        result = original(*a, **k)
        result['remote_uri_observation'] = {'uri': 'https://example.invalid/data?X-Amz-Credential=unsafe-secret'}
        return result
    monkeypatch.setattr(planner.native, '_join', secret)
    report = run(context, intent_id)
    assert 'unsafe-secret' not in json.dumps(report)
    assert 'historical_lineage' not in report and report['action'] == 'KEEP'
