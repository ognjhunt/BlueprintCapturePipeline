"""Repair decisions must follow the exported geometry and reuse the admitted seams."""
from copy import deepcopy
import json

import numpy as np
import pytest

from blueprint_pipeline import atlas_scene_coverage as coverage
from blueprint_pipeline import atlas_scene_completion as completion
from blueprint_pipeline.atlas_depth_splat import posed_frame, write_depth_splat
from blueprint_pipeline.atlas_scene_gaps import project_exported_splat
from blueprint_pipeline.atlas_scene_recovery import _frame_binding
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData


def fixture(tmp_path):
    camera = {'intrinsics': {'width': 64, 'height': 48, 'fx': 60., 'fy': 60., 'cx': 32., 'cy': 24.},
              'extrinsics': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}
    source = posed_frame(frame_id='source', rgb=np.full((48, 64, 3), 90, np.uint8),
        depth=np.full((48, 64), 2., np.float32), confidence=np.ones((48, 64), np.float32), camera=camera)
    native = deepcopy(source)
    native['frame_id'] = 'reviewed-native'
    source['confidence'][12:36, 20:44] = 0
    plan = {'schema_version': 'atlas_task_surface_plan.v1',
        'native_bindings': {native['frame_id']: canonical_digest(_frame_binding(native))},
        'proposal': {'status': 'proposed', 'task_extent_summary': 'A work surface and nearby approach space',
            'remaining_required_surfaces': ['Missing work surface'], 'excluded_scene_regions': [],
            'views': [{'frame_id': native['frame_id'], 'reason': 'Useful whole surface', 'regions': [
                {'label': 'work surface', 'decision': 'include', 'purpose': 'work_surface',
                 'polygon_normalized': [[0, 0], [1, 0], [1, 1], [0, 1]], 'reason': 'Task support'}]}]}}
    plan['digest'] = canonical_digest(plan, digest_field='digest')
    reviewed = [{'frame': native, 'native_result_digest': 'native', 'native_frame_index': 0, 'review_digest': 'review'}]
    pose = {'digest': 'pose', 'binding': {'source_video_digest': 'video'},
            'operation': {'id': 'pose-operation'}, 'depth_unit': 'estimated', 'scale_status': 'estimated'}
    root = tmp_path / 'baseline'
    write_depth_splat(frames=[source], result=pose, output_root=root,
                      generated_result_digests=['original'], visual_review_digest='review')
    return dict(pose_root=tmp_path, pose=pose, frames=[source], geometry_frames=[source],
        reviewed_views=reviewed, region_plan=plan, splat_root=root, output_root=tmp_path,
        review={}, context={'context_digest': 'task-context'}, configuration={'remaining_cost_usd': 5},
        task_runner=object(), planner_runner=object(), pose_runner=object(), release_blockers=object(), max_polls=1)


def report(reason=None):
    value = {'status': 'passed', 'generation_result_digest': 'repair-native-result',
             'review': {'status': 'passed'}, 'generated_roots': [], 'preservation_paths': []}
    if reason:
        value['review'] = {'status': 'not_required', 'reason': reason}
    value['digest'] = canonical_digest(value, digest_field='digest')
    return value


def test_coverage_projects_the_export_and_ignores_invisible_points():
    data = SplatData(count=2, xyz=np.array([[0, 0, -2], [1, 0, -2]], np.float32),
        opacity=np.array([10, -10], np.float32), f_dc=np.zeros((2, 3), np.float32),
        scales=np.full((2, 3), np.log(0.002), np.float32), quats=np.array([[1, 0, 0, 0]] * 2, np.float32), properties=())
    camera = {'intrinsics': {'width': 64, 'height': 48, 'fx': 60., 'fy': 60., 'cx': 32., 'cy': 24.},
              'extrinsics': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}
    depth = project_exported_splat(data, camera)
    assert depth[24, 32] == 2 and np.count_nonzero(depth) == 9
    assert depth[24, 62] == 0


def test_repair_is_fused_and_then_reviewed_again_before_meshing(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    progress = []
    args['checkpoint'] = lambda stage, **fields: progress.append((stage, fields))
    calls = []
    def complete(**kw):
        calls.append(kw)
        assert kw['task_runner'] is args['task_runner'] and kw['planner_runner'] is args['planner_runner']
        assert kw['release_blockers'] is args['release_blockers']
        assert {f['frame_id'] for f in kw['reference_frames']} == (
            {'source', 'reviewed-native'} if len(calls) == 1
            else {'source', 'reviewed-native', 'targeted-repair'})
        assert kw['coverage_feedback']['small_distortions_acceptable'] is True
        if len(calls) == 1:
            new = deepcopy(args['reviewed_views'][0]['frame'])
            new['frame_id'] = 'targeted-repair'
            new['confidence'][:] = 0
            new['confidence'][12:36, 20:44] = 1
            return [new], report()
        assert kw['splat_root'] != args['splat_root'], 'Review must inspect the new export'
        assert 'targeted-repair' in {f['frame_id'] for f in kw['existing_geometry_frames']}
        return [], report('agent_found_no_material_task_gap')
    monkeypatch.setattr(completion, 'complete_scene', complete)
    (geometry, manifest, root), result = coverage.refine_scene_coverage(**args)
    assert len(calls) == 2 and len(geometry) == 2
    old = json.loads((args['splat_root'] / 'atlas_depth_splat_manifest.json').read_text())
    assert manifest['gaussian_count'] > old['gaussian_count']
    assert root != args['splat_root'] and result['task_coverage_sufficient'] is True
    assert result['whole_scene_complete'] is False and result['simulation_qualified'] is False
    fusion = [fields for stage, fields in progress if stage == 'automation_export_coverage_fuse']
    assert fusion == [{'reuse_cached_export': False, 'retained_frame_count': 2,
                       'added_generated_frame_ids': ['targeted-repair']}]


def test_pretty_image_with_no_usable_depth_does_not_trigger_repeated_purchases(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    empty = deepcopy(args['reviewed_views'][0]['frame'])
    empty['confidence'][:] = 0
    calls = []
    monkeypatch.setattr(completion, 'complete_scene', lambda **kw: (calls.append(kw) or ([empty], report())))
    (_, _, root), result = coverage.refine_scene_coverage(**args)
    assert len(calls) == 1 and root == args['splat_root']
    assert result['stop_reason'] == 'generated_views_added_no_exportable_geometry'
    assert result['task_coverage_sufficient'] is False


def test_an_empty_numerical_mask_alone_does_not_qualify_task_surface_coverage(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    monkeypatch.setattr(completion, 'complete_scene', lambda **_: ([], report('no_eligible_geometry_gaps')))
    _, result = coverage.refine_scene_coverage(**args)
    assert result['stop_reason'] == 'no_eligible_geometry_gaps'
    assert result['task_coverage_sufficient'] is False


def test_native_image_is_registered_and_remasked_against_export_before_fusion(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_recovery, atlas_scene_task_regions

    args = fixture(tmp_path)
    registered = deepcopy(args['reviewed_views'][0]['frame'])
    registered['frame_id'] = 'registered-view'
    raw = deepcopy(registered)
    raw['frame_id'] = 'raw-requested-view'
    raw['position'] += 100  # A requested metadata pose must not reach the export.
    calls = []
    def complete(**kw):
        calls.append('complete')
        if calls.count('complete') == 1:
            native_report = report()
            native_report['generated_roots'] = ['native-result-retained']
            native_report['digest'] = canonical_digest(native_report, digest_field='digest')
            return [raw], native_report
        assert 'registered-view' in {f['frame_id'] for f in kw['reference_frames']}
        return [], report('agent_found_no_material_task_gap')
    def register(**kw):
        calls.append('register')
        assert kw['pose_runner'] is args['pose_runner']
        assert [f['frame_id'] for f in kw['frames']] == ['source']
        return [{'frame': registered, 'native_result_digest': 'native', 'native_frame_index': 2,
                 'review_digest': 'review', 'registration_digest': 'registration'}], {
            'digest': 'registration', 'preservation_paths': [], 'depth_basis': 'joint_pose_registered'}
    def regions(**kw):
        calls.append('regions')
        value = deepcopy(args['region_plan'])
        value['native_bindings'] = {'registered-view': canonical_digest(_frame_binding(registered))}
        value['proposal']['views'][0]['frame_id'] = 'registered-view'
        value['plan_path'] = str(tmp_path/'registered-regions.json')
        value['digest'] = canonical_digest(value, digest_field='digest')
        (tmp_path/'registered-regions.json').write_text(json.dumps(value))
        return value
    monkeypatch.setattr(completion, 'complete_scene', complete)
    monkeypatch.setattr(atlas_scene_recovery, 'register_generated_completion', register)
    monkeypatch.setattr(atlas_scene_task_regions, 'plan_task_regions', regions)
    (frames, manifest, _), receipt = coverage.refine_scene_coverage(**args)
    assert calls == ['complete', 'register', 'regions', 'complete']
    assert {f['frame_id'] for f in frames} == {'source', 'registered-view'}
    assert 'registration' in manifest['generated_result_digests']
    assert receipt['completion_reports'][0]['registered_geometry_recovery']['exportable_samples'] > 0


def test_pending_generation_resumes_through_the_same_operation(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    state = {'status': 'processing', 'operation_id': 'already-submitted', 'stage': 'scene_completion'}
    monkeypatch.setattr(completion, 'complete_scene', lambda **_: (None, state))
    assert coverage.refine_scene_coverage(**args) == (None, state)


def test_reference_selection_refuses_changed_semantic_plan(tmp_path):
    args = fixture(tmp_path)
    args['region_plan']['proposal']['task_extent_summary'] = 'changed'
    with pytest.raises(ValueError, match='task_region_plan_changed'):
        coverage.refine_scene_coverage(**args)


def test_offline_export_refresh_preserves_old_bytes_and_reuses_current_cache(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_depth_splat

    args = fixture(tmp_path)
    saved = args['splat_root'] / 'atlas_depth_splat_manifest.json'
    old = json.loads(saved.read_text())
    old.pop('fusion_profile')
    old['digest'] = canonical_digest(old, digest_field='digest')
    saved.write_text(json.dumps(old))
    original_manifest = saved.read_bytes()
    original_spz = (args['splat_root'] / 'atlas_depth_splat.spz').read_bytes()
    monkeypatch.setattr(completion, 'complete_scene', lambda **_: pytest.fail('offline replay bought a repair'))
    args['completed_exports_only'] = True
    (_, manifest, root), result = coverage.refine_scene_coverage(**args)
    assert root != args['splat_root'] and manifest['fusion_profile'] == atlas_depth_splat.fusion_profile()
    assert manifest['rebuild_source_manifest_digest'] == old['digest']
    assert manifest['voxel_footprint_multiple'] == old['voxel_footprint_multiple']
    assert manifest['generated_result_digests'] == ['original']
    assert result['stop_reason'] == 'retained_exports_replayed_without_new_generation'
    assert not result['task_coverage_sufficient'] and not result['simulation_qualified']
    assert saved.read_bytes() == original_manifest
    assert (args['splat_root'] / 'atlas_depth_splat.spz').read_bytes() == original_spz
    assert (tmp_path / 'completion_coverage/replay-latest.json').is_file()
    assert not (tmp_path / 'completion_coverage/latest.json').exists()
    monkeypatch.setattr(atlas_depth_splat, 'write_depth_splat',
                        lambda **_: pytest.fail('unchanged derived export was rebuilt'))
    (_, reused, reused_root), _ = coverage.refine_scene_coverage(**args)
    assert reused['digest'] == manifest['digest'] and reused_root == root
    with (root / 'atlas_depth_splat.spz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='refreshed_export_changed'):
        coverage.refine_scene_coverage(**args)


def test_completed_paid_repair_is_adopted_before_fusion_upgrade(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_recovery, atlas_scene_task_regions, atlas_depth_splat

    args = fixture(tmp_path)
    saved = args['splat_root'] / 'atlas_depth_splat_manifest.json'
    old = json.loads(saved.read_text())
    old.pop('fusion_profile')
    old['digest'] = canonical_digest(old, digest_field='digest')
    saved.write_text(json.dumps(old))
    binding = {'revision': 'v3-complete-surface-review', 'export_digest': old['digest'],
               'task_region_plan_digest': args['region_plan']['digest'], 'task_context_digest': 'task-context'}
    identity = canonical_digest(binding)
    repair_root = tmp_path / 'completion_coverage' / identity[7:23]
    native_root = repair_root / 'generated/0'
    native_root.mkdir(parents=True)
    native = {'status': 'completed', 'operation': {'id': 'already-paid'}}
    native['digest'] = canonical_digest(native, digest_field='digest')
    (native_root / 'result.json').write_text(json.dumps(native))
    retained = report()
    retained['generated_roots'] = [str(native_root)]
    retained['digest'] = canonical_digest(retained, digest_field='digest')
    (repair_root / 'completion.json').write_text(json.dumps(retained))
    new = deepcopy(args['reviewed_views'][0]['frame'])
    new['frame_id'] = 'registered-repair'
    view = {'frame': new, 'native_result_digest': native['digest'], 'native_frame_index': 0,
            'review_digest': retained['digest'], 'registration_digest': 'registration'}
    regions = deepcopy(args['region_plan'])
    regions['native_bindings'] = {new['frame_id']: canonical_digest(_frame_binding(new))}
    regions['proposal']['views'][0]['frame_id'] = new['frame_id']
    regions['digest'] = canonical_digest(regions, digest_field='digest')
    old_repair_root = tmp_path / ('background_refined_splat-' + identity[7:23])
    previous = write_depth_splat(frames=[args['frames'][0], new], result=args['pose'],
        output_root=old_repair_root,
        generated_result_digests=['original', 'repair-native-result', 'registration', 'recovery'],
        visual_review_digest=canonical_digest({'previous': 'review', 'review': retained['review']}))
    previous.pop('fusion_profile')
    previous['digest'] = canonical_digest(previous, digest_field='digest')
    (old_repair_root / 'atlas_depth_splat_manifest.json').write_text(json.dumps(previous))
    calls = []
    def complete(**kw):
        calls.append(kw['root'])
        assert kw['root'] == repair_root
        return [new], retained
    monkeypatch.setattr(completion, 'complete_scene', complete)
    monkeypatch.setattr(atlas_scene_recovery, 'register_generated_completion', lambda **_: ([view], {
        'digest': 'registration', 'preservation_paths': [], 'depth_basis': 'registered_native'}))
    monkeypatch.setattr(atlas_scene_task_regions, 'plan_task_regions', lambda **_: regions)
    monkeypatch.setattr(atlas_scene_recovery, 'recover_reviewed_geometry', lambda **_: ([new], {
        'digest': 'recovery', 'preservation_paths': [], 'added_geometry_pixels': 100, 'exportable_samples': 100}))
    progress = []
    args.update(completed_exports_only=True, checkpoint=lambda stage, **fields: progress.append((stage, fields)))
    (geometry, manifest, _), result = coverage.refine_scene_coverage(**args)
    assert calls == [repair_root] and len(geometry) == 2 and len(result['completion_reports']) == 1
    assert manifest['fusion_profile'] == atlas_depth_splat.fusion_profile()
    assert manifest['generated_result_digests'] == previous['generated_result_digests']
    upgraded = [fields for stage, fields in progress if stage == 'automation_export_refresh_fusion']
    assert len(upgraded) == 1 and upgraded[0]['retained_frame_count'] == 2


def test_missing_completed_native_bytes_refuse_instead_of_repurchasing(tmp_path):
    retained = report()
    retained['generated_roots'] = [str(tmp_path / 'missing-native')]
    retained['digest'] = canonical_digest(retained, digest_field='digest')
    (tmp_path / 'completion.json').write_text(json.dumps(retained))
    with pytest.raises(ValueError, match='retained_native_result_missing'):
        coverage._retained_native_repair(tmp_path)


def test_offline_replay_verifies_current_export_bytes_too(tmp_path):
    args = fixture(tmp_path)
    (args['splat_root'] / 'atlas_depth_splat.spz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='refreshed_export_changed'):
        coverage.refine_scene_coverage(**args, completed_exports_only=True)


def test_recorded_source_action_is_recovered_fused_and_conditioned_before_generation(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args['pose']['digest'] = 'pose'
    calls, recoveries = [], []
    new = deepcopy(args['reviewed_views'][0]['frame'])
    new['frame_id'] = 'recorded-later'
    new['confidence'][:] = 0
    new['confidence'][12:36, 20:44] = 1
    receipt = {'status': 'completed', 'target_pose_digest': 'pose',
        'recovered_frame_ids': [new['frame_id']], 'frame_bindings': [_frame_binding(new)],
        'preservation_paths': [], 'rejected_views': []}
    receipt['digest'] = canonical_digest(receipt, digest_field='digest')
    def complete(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return None, {'status': 'source_recovery_required', 'digest': 'source-action'}
        assert [f['frame_id'] for f in kw['frames']] == ['source', 'recorded-later']
        assert 'recorded-later' in {f['frame_id'] for f in kw['reference_frames']}
        assert 'recorded-later' not in {f['frame_id'] for f in kw['existing_geometry_frames']}
        assert kw['coverage_feedback']['recorded_source_outcomes'][0]['recovered_frame_ids'] == ['recorded-later']
        assert kw['splat_root'] != args['splat_root']
        return [], report('agent_found_no_material_task_gap')
    def recover(action, known):
        recoveries.append(action)
        assert [f['frame_id'] for f in known] == ['source']
        return [new], receipt
    monkeypatch.setattr(completion, 'complete_scene', complete)
    args['source_recovery_runner'] = recover
    (geometry, manifest, _), result = coverage.refine_scene_coverage(**args)
    assert len(calls) == 2 and len(recoveries) == 1
    assert [f['frame_id'] for f in geometry] == ['source', 'recorded-later']
    assert manifest['pose_result_digest'] == 'pose'
    assert receipt['digest'] in manifest['generated_result_digests']
    assert result['recorded_source_recoveries'] == [receipt]


def test_shared_rgbd_refresh_replaces_old_geometry_and_all_generation_references(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    source = args['frames'][0]
    refreshed = deepcopy(source)
    refreshed['rgb'][:] = 190
    refreshed['depth'][:] = 2.1
    new = deepcopy(refreshed)
    new['frame_id'] = 'additional'
    receipt = {'status': 'completed', 'target_pose_digest': 'pose', 'recovered_frame_ids': ['additional'],
        'refreshed_frame_ids': ['source'], 'frame_bindings': [_frame_binding(f) for f in [refreshed, new]],
        'context_pose_root': str(tmp_path / 'fresh-pose'), 'pose_result_digest': 'fresh-pose-digest',
        'preservation_paths': [], 'rejected_views': []}
    receipt['digest'] = canonical_digest(receipt, digest_field='digest')
    calls = []
    def complete(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return None, {'status': 'source_recovery_required'}
        assert kw['context_pose_root'] == str(tmp_path / 'fresh-pose')
        assert kw['context_pose_digest'] == 'fresh-pose-digest'
        assert [f['frame_id'] for f in kw['frames']] == ['source', 'additional']
        assert np.all(kw['frames'][0]['rgb'] == 190)
        assert np.all(kw['frames'][0]['depth'] == 2.1)
        references = kw['reference_frames']
        assert len(references) == len({f['frame_id'] for f in references})
        assert np.all(next(f for f in references if f['frame_id'] == 'source')['rgb'] == 190)
        assert kw['existing_geometry_frames'] == []
        return [], report('agent_found_no_material_task_gap')
    monkeypatch.setattr(completion, 'complete_scene', complete)
    args['source_recovery_runner'] = lambda *_: ([refreshed, new], receipt)
    (geometry, _, _), result = coverage.refine_scene_coverage(**args)
    assert len(calls) == 2 and result['task_coverage_sufficient']
    assert np.all(next(f for f in geometry if f['frame_id'] == 'source')['rgb'] == 190)
    assert np.all(source['rgb'] == 90), 'Original retained data must stay unchanged'


def test_source_recovery_processing_does_not_start_generation(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    monkeypatch.setattr(completion, 'complete_scene', lambda **_: (None, {'status': 'source_recovery_required'}))
    pending = {'status': 'processing', 'stage': 'recorded_source_pose', 'operation_id': 'retained-pose'}
    args['source_recovery_runner'] = lambda *_: (None, pending)
    assert coverage.refine_scene_coverage(**args) == (None, pending)


def test_coverage_counts_a_slanted_surface_its_discs_bridge_but_keeps_real_gaps():
    camera = {'intrinsics': {'width': 64, 'height': 48, 'fx': 60., 'fy': 60., 'cx': 32., 'cy': 24.},
              'extrinsics': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}
    # Rows of centres 4 px apart, as an oblique view samples a surface, with a 12 px hole.
    columns = [c for c in range(8, 56) if not 26 <= c < 38]
    pixels = [(c, r) for r in range(8, 40, 4) for c in columns]
    xyz = np.array([[(c + .5 - 32) / 60 * 2, -(r + .5 - 24) / 60 * 2, -2] for c, r in pixels], np.float32)
    def export(sigma):
        return SplatData(count=len(xyz), xyz=xyz, opacity=np.full(len(xyz), 10, np.float32),
            f_dc=np.zeros((len(xyz), 3), np.float32), scales=np.full((len(xyz), 3), np.log(sigma), np.float32),
            quats=np.array([[1, 0, 0, 0]] * len(xyz), np.float32), properties=())
    rows = (slice(9, 37), slice(10, 22))
    assert (project_exported_splat(export(0.002), camera)[rows] == 0).any(), 'one-voxel discs leave row gaps'
    bridged = project_exported_splat(export(0.06), camera)  # 1.5 sigma ~ 2.7 px bridges 4 px rows
    assert (bridged[rows] > 0).all()
    assert (bridged[16:32, 30:34] == 0).all(), 'a real missing patch stays a gap'
