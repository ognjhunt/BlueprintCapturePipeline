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
        task_runner=object(), planner_runner=object(), release_blockers=object(), max_polls=1)


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
        scales=np.zeros((2, 3), np.float32), quats=np.array([[1, 0, 0, 0]] * 2, np.float32), properties=())
    camera = {'intrinsics': {'width': 64, 'height': 48, 'fx': 60., 'fy': 60., 'cx': 32., 'cy': 24.},
              'extrinsics': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}
    depth = project_exported_splat(data, camera)
    assert depth[24, 32] == 2 and np.count_nonzero(depth) == 9
    assert depth[24, 62] == 0


def test_repair_is_fused_and_then_reviewed_again_before_meshing(tmp_path, monkeypatch):
    args = fixture(tmp_path)
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
