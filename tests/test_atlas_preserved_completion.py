"""Offline rehearsal: pictures need registered, exportable depth before a preserved scene can use them."""
import numpy as np
import pytest

from blueprint_pipeline import atlas_preserved_completion as worker
from blueprint_pipeline import atlas_scene_completion, atlas_scene_recovery, atlas_depth_splat
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_depth_splat import camera, frame, plane_depth


def sealed(value):
    return {**value, 'digest': canonical_digest(value, digest_field='digest')}


def fixture(tmp_path, monkeypatch, outcome='geometry'):
    cam = camera()
    depth = plane_depth(cam)
    depth[20:44, 14:34] = 0
    source = frame(cam, depth, 'source')
    addition = frame(cam, plane_depth(cam), 'registered-addition')
    addition['confidence'][:] = 0
    addition['confidence'][23:41, 17:31] = 255
    pose = sealed({'operation': {'id': 'source-pose'}, 'binding': {'source_video_digest': 'capture'}})
    baseline_root = tmp_path / 'source-splat'
    baseline = atlas_depth_splat.write_depth_splat(frames=[source], result=pose, output_root=baseline_root)
    calls = []
    def complete(**options):
        calls.append('complete')
        assert options['context_basis'] == 'preserved_capture'
        assert options['source_survey_path'] == tmp_path / 'survey.json'
        if outcome == 'source':
            return None, {'status': 'source_recovery_required', 'reason': 'source_view_available',
                          'new_atlas_generation_calls': 0}
        generation = sealed({'status': 'completed', 'native_result_digests': ['native-result'],
            'source_geometry_modified': False, 'known_pixels_changed': 0,
            'generated_roots': [str(tmp_path / 'native')], 'preservation_paths': []})
        report = sealed({'status': 'passed', 'context_basis': 'preserved_capture',
            'source_pose_digest': pose['digest'], 'source_video_digest': 'capture',
            'task_context_digest': 'task', 'generated_geometry_ready': True,
            'generation': generation, 'generation_result_digest': generation['digest'],
            'generated_roots': generation['generated_roots'], 'preservation_paths': [],
            'review': {'status': 'passed'}})
        if outcome == 'no-gap':
            report = sealed({**report, 'generated_roots': [], 'review': {'status': 'not_required'}})
        write_json(options['root'] / 'completion.json', report)
        return ([] if outcome == 'no-gap' else [addition]), report
    def register(**options):
        calls.append('register')
        if outcome == 'registering':
            return None, {'status': 'processing', 'operation_id': 'same-retained-operation'}
        path = options['root'] / 'registration.json'
        report = sealed({'status': 'completed', 'source_frame_ids': ['source'],
            'source_geometry_modified': False, 'native_outputs_modified': False,
            'depth_basis': 'joint_pose_depth_registered_to_retained_source_world',
            'preservation_paths': [str(path)]})
        write_json(path, report)
        return [{'frame': addition}], report
    def recover(**options):
        calls.append('recover')
        path = options['root'] / 'recovery.json'
        report = sealed({'source_geometry_modified': False, 'known_pixels_changed': 0,
            'exportable_samples': 0 if outcome == 'picture-only' else 50,
            'preservation_paths': [str(path)]})
        write_json(path, report)
        return ([] if outcome == 'picture-only' else [addition]), report
    monkeypatch.setattr(atlas_scene_completion, 'complete_scene', complete)
    monkeypatch.setattr(atlas_scene_recovery, 'register_generated_completion', register)
    monkeypatch.setattr(atlas_scene_recovery, 'recover_reviewed_geometry', recover)
    args = dict(pose_root=tmp_path / 'pose', pose=pose, frames=[source],
        review={'subject_origin': 'no_manipulated_subject', 'automation': {'objects': [], 'removal_instruction': ''}},
        root=tmp_path / 'completion', job_root=tmp_path, context={'context_digest': 'task'}, configuration={},
        splat_root=baseline_root, source_survey_path=tmp_path / 'survey.json',
        task_runner=None, planner_runner=None, pose_runner=None, release_blockers=None, max_polls=1)
    return args, source, baseline, calls


@pytest.mark.parametrize('outcome', ['no-gap', 'source', 'registering', 'picture-only'])
def test_missing_or_unregistered_geometry_never_reaches_fusion(tmp_path, monkeypatch, outcome):
    args, source, baseline, calls = fixture(tmp_path, monkeypatch, outcome)
    before = source['depth'].copy()
    monkeypatch.setattr(atlas_depth_splat, 'write_depth_splat',
                        lambda **_: pytest.fail('No admitted additional geometry exists'))
    result = worker.run_preserved_completion(**args)
    assert np.array_equal(source['depth'], before)
    assert not result.get('generated_geometry_admitted')
    assert not list(tmp_path.glob('completion/splat-*'))
    if outcome == 'registering':
        assert result['operation_id'] == 'same-retained-operation'
    if outcome == 'picture-only':
        assert result['reason'] == 'atlas_preserved_completion_no_exportable_geometry'
    if outcome in {'source', 'no-gap'}:
        assert calls == ['complete']


def test_registered_additions_are_consumed_with_separate_provenance_and_reused(tmp_path, monkeypatch):
    from scipy.spatial import cKDTree
    from blueprint_pipeline.gaussian_splat_decode import read_standard_3dgs_ply
    args, source, baseline, calls = fixture(tmp_path, monkeypatch)
    before = source['depth'].copy()
    result = worker.run_preserved_completion(**args)
    report, splat = result['admission'], result['splat_manifest']
    assert result['generated_geometry_admitted'] is True
    assert splat['appearance_basis'] == worker.APPEARANCE_BASIS
    assert splat['rebuild_source_manifest_digest'] == baseline['digest']
    assert {row['frame_id'] for row in splat['frames']} == {'source', 'registered-addition'}
    assert np.array_equal(source['depth'], before)
    original = read_standard_3dgs_ply(args['splat_root'] / baseline['outputs'][0]['path'])
    completed = read_standard_3dgs_ply(result['splat_root'] + '/' + splat['outputs'][0]['path'])
    assert np.max(cKDTree(completed.xyz).query(original.xyz)[0]) < 0.0005
    assert report['simulation_qualified'] is False and report['metric_measurement_proven'] is False
    monkeypatch.setattr(atlas_depth_splat, 'write_depth_splat',
                        lambda **_: pytest.fail('Identical retained fusion was rebuilt'))
    resumed = worker.run_preserved_completion(**args)
    assert resumed['admission']['digest'] == report['digest']
    asset = tmp_path / report['assets'][0]['relative_path']
    asset.write_bytes(asset.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='evidence_changed'):
        worker.validate_preserved_completion(report=report, job_root=tmp_path,
            source_pose_digest=args['pose']['digest'], source_video_digest='capture',
            task_context_digest='task', splat=splat)


def test_changed_baseline_refuses_before_any_completion_call(tmp_path, monkeypatch):
    args, _, baseline, calls = fixture(tmp_path, monkeypatch)
    path = args['splat_root'] / baseline['outputs'][0]['path']
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='export_changed'):
        worker.run_preserved_completion(**args)
    assert calls == []


def test_preserved_completion_stays_inside_the_existing_allocator_release_gate(monkeypatch):
    from blueprint_pipeline import paid_resource_allocator as allocator, atlas_scene_automation
    monkeypatch.setattr(allocator, '_current_checkout_source_state', lambda: ('a' * 40, True, True))
    monkeypatch.setattr(allocator, '_source_checkout_blockers', lambda *args, **kw: (['source-hold'], 'a' * 40))
    monkeypatch.setattr(atlas_scene_automation, 'run_automation',
                        lambda **_: pytest.fail('Preserved context entered the object-editing controller'))
    def run(**options):
        assert callable(options['task_runner']) and callable(options['planner_runner'])
        assert callable(options['pose_runner'])
        commit, blockers = options['release_blockers']()
        return {'commit': commit, 'blockers': blockers}
    monkeypatch.setattr(worker, 'run_preserved_completion', run)
    assert allocator.run_sponsored_website_atlas_automation(preserved_capture=True) == {
        'commit': 'a' * 40, 'blockers': ['source-hold']}


@pytest.mark.parametrize('entrypoint', ['pose', 'task', 'plan', 'automation', 'preserved'])
@pytest.mark.parametrize('experimental', [False, True])
def test_all_atlas_callbacks_recheck_the_current_source_and_preserve_the_experimental_flag(
        tmp_path, monkeypatch, entrypoint, experimental):
    from blueprint_pipeline import paid_resource_allocator as allocator, website_atlas_pose
    from blueprint_pipeline import website_atlas_tasks, atlas_scene_planner, atlas_scene_automation
    commits = iter(['1' * 40, '2' * 40])
    seen = []
    monkeypatch.setattr(allocator, '_current_checkout_source_state', lambda: (next(commits), True, True))
    def blockers(commit, *, allow_pushed_branch_diagnostic):
        seen.append((commit, allow_pushed_branch_diagnostic))
        return ['existing-hold'], commit
    monkeypatch.setattr(allocator, '_source_checkout_blockers', blockers)
    def run(**options):
        return [options['release_blockers'](), options['release_blockers']()]
    targets = {'pose': (website_atlas_pose, 'run_admitted_atlas_pose'),
        'task': (website_atlas_tasks, 'run_admitted_atlas_task'),
        'plan': (atlas_scene_planner, 'review_survey'),
        'automation': (atlas_scene_automation, 'run_automation'),
        'preserved': (worker, 'run_preserved_completion')}
    monkeypatch.setattr(*targets[entrypoint], run)
    method = getattr(allocator, 'run_sponsored_website_atlas_' + ('automation' if entrypoint == 'preserved' else entrypoint))
    options = {'experimental': experimental}
    if entrypoint == 'pose':
        options.update(output_root=tmp_path, capture_id='capture', task_context={})
    if entrypoint == 'preserved':
        options['preserved_capture'] = True
    assert method(**options) == [('1' * 40, ['existing-hold']), ('2' * 40, ['existing-hold'])]
    assert seen == [('1' * 40, experimental), ('2' * 40, experimental)]
