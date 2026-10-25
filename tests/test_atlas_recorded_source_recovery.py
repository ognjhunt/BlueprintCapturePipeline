"""Replay recorded-view handoff without model calls, Atlas uploads or GPU work."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_recorded_source_recovery as recovery
from blueprint_pipeline import atlas_background_automation as background
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def bound(value):
    value['digest'] = canonical_digest(value, digest_field='digest')
    return value


def fixture(tmp_path):
    rows, edited, frames = [], [], []
    for i, key in enumerate(['anchor-a', 'anchor-b', 'later']):
        pixels = np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        original = tmp_path / f'{key}.png'
        Image.fromarray(pixels).save(original)
        rows.append({'frame_id': key, 'source_image_path': str(original),
                     'source_image_digest': _sha256_file(original), 'timestamp_seconds': float(i)})
        if key == 'later':
            continue
        clean = tmp_path / f'{key}-clean.png'
        Image.fromarray(pixels).save(clean)
        edited.append({'frame_id': key, 'image_path': str(clean), 'image_digest': _sha256_file(clean),
            'original_image_path': str(original), 'original_image_digest': _sha256_file(original),
            'remaining_pixel_count': 0, 'removed_task_object_ids': ['target']})
        camera = {'intrinsics': {'width': 64, 'height': 48, 'fx': 60., 'fy': 60., 'cx': 32., 'cy': 24.},
                  'extrinsics': {'position': [i * .1, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}
        frames.append(posed_frame(frame_id=key, rgb=pixels, depth=np.full((48, 64), 2., np.float32),
                                  confidence=np.ones((48, 64), np.float32), camera=camera))
    context = {'context_digest': 'task-context', 'description': 'Work on the target', 'capture_id': 'capture'}
    binding = {'source_video_digest': 'source-video', 'task_context_digest': canonical_digest(context)}
    prior = tmp_path / 'prepared.json'
    background._prepared_background(prior, binding, {'status': 'fallback_prepared', 'frames': edited,
        'review': review(edited), 'ready_for_reconstruction': True})
    survey = bound({'schema_version': 'atlas_video_survey.v1', 'source_video_digest': 'source-video',
        'display_rotation_degrees': 0, 'heldout_pixels_included': False, 'frames': rows})
    survey_path = tmp_path / 'source-survey.json'
    survey_path.write_text(json.dumps(survey))
    audit = bound({'gallery': {'survey_digest': survey['digest'], 'source_video_digest': 'source-video',
        'conditioned_frame_ids': ['anchor-a', 'anchor-b']}, 'additional_source_frame_ids': ['later'],
        'gap_evidence': [{'gap_id': 'worktop-gap', 'source_frame_ids': ['later']}], 'unresolved_gap_ids': []})
    action = bound({'status': 'source_recovery_required', 'source_evidence': audit})
    automation = {'objects': [{'target_id': 'target', 'label': 'task object', 'segmentation_prompt': 'task object',
        'disposition': 'remove', 'regions': [{'frame_id': 'anchor-a', 'box_xywh_normalized': [.2, .2, .5, .5]}]}],
        'removal_instruction': 'Remove all task object parts and contents; preserve the fixed scene.',
        'mask_margin_fraction': .01, 'scale_references': []}
    proposal = {'views': [{'frame_id': 'later', 'decision': 'use', 'fixed_environment_consistent': True,
        'objects': [{'target_id': 'target', 'visibility': 'visible', 'regions': [
            {'frame_id': 'later', 'box_xywh_normalized': [.2, .2, .5, .5]}]}],
        'reason': 'The moved removable object can be erased; the fixed worktop is recorded.'}],
        'transients': [], 'summary': 'Recover fixed support geometry, preserving source task state.'}
    pose = {'digest': 'original-pose', 'binding': {'source_video_digest': 'source-video'}}
    args = dict(action=action, source_survey_path=survey_path, background_path=prior,
        automation=automation, pose=pose, frames=frames, root=tmp_path / 'recovery', context=context,
        configuration={'policy': {'rgbd_view_cost_usd': .05}, 'remaining_cost_usd': 15},
        task_runner=lambda **_: pytest.fail('Raw view reached Atlas'), pose_runner=object(),
        planner_runner=lambda **_: {'proposal': deepcopy(proposal)}, release_blockers=lambda: ('clean-commit', []),
        max_polls=1)
    return args, proposal, rows


def review(rows):
    binding = {'frames': [{k: r[k] for k in ('frame_id', 'image_digest')} for r in rows]}
    return {'status': 'passed', 'binding': binding, 'request_digest': canonical_digest(binding),
        'review': {'consistent_background': True, 'task_objects_removed': True, 'unrelated_objects_preserved': True}}


@pytest.mark.parametrize('change', ['action', 'survey', 'pixels', 'unknown', 'already-conditioned'])
def test_source_identity_and_pixels_refuse_before_providers(tmp_path, change):
    args, _, rows = fixture(tmp_path)
    if change == 'action':
        args['action']['reason'] = 'tampered'
    elif change == 'survey':
        survey = json.loads(args['source_survey_path'].read_text())
        survey['heldout_pixels_included'] = True
        args['source_survey_path'].write_text(json.dumps(bound(survey)))
    elif change == 'pixels':
        Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(rows[-1]['source_image_path'])
    else:
        args['action']['source_evidence']['additional_source_frame_ids'] = [
            'not-recorded' if change == 'unknown' else 'anchor-a']
        bound(args['action']['source_evidence'])
        bound(args['action'])
    args['planner_runner'] = lambda **_: pytest.fail('Invalid recovery reached the model')
    with pytest.raises(ValueError, match='atlas_recorded_source_'):
        recovery.recover_recorded_sources(**args)


@pytest.mark.parametrize('change', ['uncertain', 'changed-environment', 'target', 'frame'])
def test_uncertain_environment_or_removal_never_reaches_cleaner(tmp_path, monkeypatch, change):
    args, proposal, _ = fixture(tmp_path)
    view = proposal['views'][0]
    if change == 'uncertain':
        view['objects'][0].update(visibility='uncertain', regions=[])
    elif change == 'changed-environment':
        view['fixed_environment_consistent'] = False
    elif change == 'target':
        view['objects'][0]['target_id'] = 'other-target'
    else:
        view['objects'][0]['regions'][0]['frame_id'] = 'anchor-a'
    monkeypatch.setattr(background, '_provider_preflight', lambda: None)
    monkeypatch.setattr(background, 'run_background', lambda **_: pytest.fail('Invalid removal reached cleanup'))
    with pytest.raises(ValueError, match='atlas_recorded_source_'):
        recovery.recover_recorded_sources(**args)


def test_transform_scales_depth_and_camera_together_without_changing_provider_data(tmp_path):
    args, _, _ = fixture(tmp_path)
    old = deepcopy(args['frames'][0])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    new = recovery.aligned_recorded_frames([old], {'scale': 2, 'rotation': rotation, 'translation': [1, 2, 3]})[0]
    assert np.array_equal(old['depth'], args['frames'][0]['depth'])
    assert np.allclose(new['depth'], 4) and np.allclose(new['position'], [1, 2, 3])
    from blueprint_pipeline.atlas_depth_splat import _rotation_matrix
    assert np.allclose(_rotation_matrix(new['extrinsics']['quaternion']), rotation)
    assert np.allclose(new['extrinsics']['position'], new['position'])


def test_offline_replay_reuses_accepted_edits_and_preserves_old_world(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation, atlas_privacy_screen
    from blueprint_pipeline import website_image_completion as editor, website_image_repair_agent

    args, _, _ = fixture(tmp_path)
    old_manifest = args['background_path'].read_bytes()
    old_frames = deepcopy(args['frames'])
    edits, pose_calls, reviews = [], [], []
    monkeypatch.setattr(background, '_provider_preflight', lambda: None)
    monkeypatch.setattr(atlas_privacy_screen, 'blur_faces', lambda image: (image, []))
    monkeypatch.setattr(atlas_privacy_screen, 'screen_identity', lambda: {'screen': 'offline-fixture'})
    monkeypatch.setattr(website_image_repair_agent, 'image_repair_enabled', lambda: False)
    monkeypatch.setattr(editor, '_validated_backend', lambda *_a, **_k:
                        ({}, {'pricing_binding': {'max_cost_per_request_usd': 1}}, 'backend'))
    def segment(**kw):
        assert [f['frame_id'] for f in kw['frames']] == ['later']
        obj = kw['objects'][0]
        return [{**obj, 'track': background.grounded_region_guide(target={**obj,
            'spatial_evidence': [{'box_xywh_normalized': [.2, .2, .5, .5]}]}, frames=kw['frames'])}]
    monkeypatch.setattr(background, 'segment_objects', segment)
    def edit(**kw):
        if kw.get('require_cached'):
            raise ValueError('website_image_completion_cached_batch_missing')
        edits.append([r['frame_id'] for r in kw['frames']])
        assert edits[-1] == ['later']
        rows = []
        for row in kw['frames']:
            path = tmp_path / (row['frame_id'] + '-accepted.png')
            Image.fromarray(np.full((48, 64, 3), 80, np.uint8)).save(path)
            rows.append({**row, 'image_path': str(path), 'image_digest': _sha256_file(path), 'remaining_pixel_count': 0})
        return rows
    def verify(**kw):
        reviews.append([r['frame_id'] for r in kw['frames']])
        assert reviews[-1] == ['anchor-a', 'anchor-b', 'later']
        return review(kw['frames'])
    monkeypatch.setattr(editor, 'complete_background_images', edit)
    monkeypatch.setattr(editor, 'verify_completed_background', verify)
    def pose(cleaned, root, *unused):
        pose_calls.append(cleaned)
        assert cleaned['review']['status'] == 'passed'
        assert cleaned['reused_prepared_frame_ids'] == ['anchor-a', 'anchor-b']
        new = deepcopy(args['frames'][0])
        new['frame_id'] = 'later'
        new['rgb'][:] = 80
        result = {'digest': 'new-pose'}
        shared = deepcopy(args['frames'])
        shared[0]['rgb'][:] = 140
        return (result, [*shared, new], root / 'fallback_pose-v3'), {'status': 'completed'}
    monkeypatch.setattr(atlas_scene_automation, '_fallback_pose', pose)
    from blueprint_pipeline import atlas_shared_view_alignment
    monkeypatch.setattr(atlas_shared_view_alignment, 'shared_view_alignment', lambda *_: {
        'scale': 1., 'rotation': np.eye(3), 'translation': np.zeros(3)})
    recovered, report = recovery.recover_recorded_sources(**args)
    assert edits == [['later']] and len(reviews) == 1
    assert [f['frame_id'] for f in recovered] == ['anchor-a', 'anchor-b', 'later']
    assert report['recovered_frame_ids'] == ['later']
    assert report['refreshed_frame_ids'] == ['anchor-a', 'anchor-b']
    assert np.all(recovered[0]['rgb'] == 140)
    assert report['context_pose_root'].endswith('fallback_pose-v3')
    assert (args['root'] / 'result-v2.json').is_file()
    assert report['target_pose_digest'] == 'original-pose' and report['pose_result_digest'] == 'new-pose'
    assert report['original_pose_unchanged'] and report['task_state_selection_unchanged']
    assert args['background_path'].read_bytes() == old_manifest
    assert all(np.array_equal(a['depth'], b['depth']) and np.array_equal(a['position'], b['position'])
               for a, b in zip(old_frames, args['frames'], strict=True))
    first_binding = report['frame_bindings']
    # An ordinary worker resume reuses prepared RGB and alignment receipts.
    second, again = recovery.recover_recorded_sources(**args)
    assert edits == [['later']] and len(reviews) == 1 and again['frame_bindings'] == first_binding
    assert np.array_equal(second[0]['depth'], recovered[0]['depth'])
    assert len(pose_calls) == 2  # The canonical pose adapter adopts its retained request.


def test_recovery_schema_is_accepted_by_the_production_sdk():
    from agents import AgentOutputSchema
    AgentOutputSchema(recovery.RecordedSourceProposal)


def retained_fixture(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_depth_splat
    from blueprint_pipeline.atlas_scene_recovery import _frame_binding

    args, _, rows = fixture(tmp_path)
    root = tmp_path/'retained/previous-camera-plan'
    root.mkdir(parents=True)
    survey = json.loads(args['source_survey_path'].read_text())
    (root/'survey.json').write_text(json.dumps(survey))
    known = args['frames']
    extra = deepcopy(known[0])
    extra['frame_id'] = 'later'
    new = [*deepcopy(known), extra]
    pose_root = root/'fallback_pose-v3'
    pose_root.mkdir()
    posed = bound({'binding': {'source_video_digest': 'source-video'}})
    monkeypatch.setattr(atlas_depth_splat, 'load_posed_rgbd', lambda path: (posed, deepcopy(new)))
    old = json.loads(args['background_path'].read_text())
    prepared = deepcopy(old['background']['frames'])
    prepared.append({**prepared[0], 'frame_id': 'later', 'original_image_path': rows[-1]['source_image_path'],
                     'original_image_digest': rows[-1]['source_image_digest']})
    background_path = root/'background/prepared_background.json'
    background._prepared_background(background_path, old['binding'], {
        'status': 'fallback_prepared', 'frames': prepared, 'review': review(prepared),
        'ready_for_reconstruction': True})
    alignment = bound({'binding': {'source_pose_digest': posed['digest'],
        'target_pose_digest': args['pose']['digest'], 'source_frames': [_frame_binding(f) for f in new],
        'target_frames': [_frame_binding(f) for f in known]},
        'fit': {'scale': 1., 'rotation': np.eye(3).tolist(), 'translation': [0, 0, 0]}})
    (root/'alignment.json').write_text(json.dumps(alignment))
    new = recovery.aligned_recorded_frames(new, alignment['fit'])
    result = bound({'schema_version': 'atlas_recorded_source_recovery.v2', 'status': 'completed',
        'binding': {'source_pose_digest': args['pose']['digest'], 'task_context_digest': args['context']['context_digest'],
            'automation_digest': canonical_digest(args['automation']),
            'target_frame_bindings': [_frame_binding(f) for f in known],
            'source_action_digest': 'an-older-camera-plan'},
        'target_pose_digest': args['pose']['digest'], 'context_pose_root': str(pose_root),
        'background_path': str(background_path), 'pose_result_digest': posed['digest'],
        'alignment_digest': alignment['digest'], 'frame_bindings': [_frame_binding(f) for f in new],
        'recovered_frame_ids': ['later'], 'refreshed_frame_ids': ['anchor-a', 'anchor-b'],
        'expanded_automation': args['automation']})
    (root/'result-v2.json').write_text(json.dumps(result))
    options = {k: args[k] for k in ('frames', 'pose', 'automation', 'context', 'source_survey_path')}
    return {**options, 'root': root.parent}, result, rows


def test_a_changed_camera_plan_and_expanded_gallery_reuse_completed_source_work(tmp_path, monkeypatch):
    args, report, _ = retained_fixture(tmp_path, monkeypatch)
    survey = json.loads(args['source_survey_path'].read_text())
    survey['frames'].append({**survey['frames'][0], 'frame_id': 'new-candidate', 'timestamp_seconds': 4.})
    args['source_survey_path'].write_text(json.dumps(bound(survey)))
    actual, receipts = recovery.adopt_recorded_recoveries(**args)
    assert [f['frame_id'] for f in actual] == ['anchor-a', 'anchor-b', 'later']
    assert receipts == [report]
    assert args['frames'][0]['frame_id'] == 'anchor-a' and len(args['frames']) == 2
    again, repeated = recovery.adopt_recorded_recoveries(**args)
    assert repeated == receipts and np.array_equal(again[-1]['depth'], actual[-1]['depth'])


@pytest.mark.parametrize('change', ['pixels', 'receipt', 'world'])
def test_completed_source_adoption_refuses_changed_evidence(tmp_path, monkeypatch, change):
    args, report, rows = retained_fixture(tmp_path, monkeypatch)
    path = args['root']/'previous-camera-plan/result-v2.json'
    if change == 'pixels':
        Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(rows[-1]['source_image_path'])
    elif change == 'receipt':
        report['recovered_frame_ids'] = ['invented']
        path.write_text(json.dumps(report))
    else:
        p = args['root']/'previous-camera-plan/alignment.json'
        alignment = json.loads(p.read_text())
        alignment['fit']['translation'] = [100, 0, 0]
        p.write_text(json.dumps(bound(alignment)))
    with pytest.raises(ValueError, match='atlas_retained_source_'):
        recovery.adopt_recorded_recoveries(**args)


def test_adopting_sources_keeps_a_completed_native_prefix_instead_of_replanning_it(tmp_path):
    initial = tmp_path/'scene-completion'
    initial.mkdir()
    report = bound({'status': 'passed', 'generated_roots': [str(initial/'native')]})
    (initial/'completion.json').write_text(json.dumps(report))
    sources = [{'digest': 'reviewed-source-recovery'}]
    assert recovery.source_completion_root(initial, tmp_path, sources) == initial
    report = bound({'status': 'source_recovery_required', 'generated_roots': []})
    (initial/'completion.json').write_text(json.dumps(report))
    expected = tmp_path/'source_conditioned_completion'/canonical_digest(
        {'recoveries': ['reviewed-source-recovery']})[7:23]
    assert recovery.source_completion_root(initial, tmp_path, sources) == expected
    report['status'] = 'passed'
    (initial/'completion.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='retained_initial_completion_changed'):
        recovery.source_completion_root(initial, tmp_path, sources)
