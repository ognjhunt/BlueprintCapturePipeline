"""Previously selected image masks must not hide other recoverable 3D gaps."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline import atlas_scene_recovery as recovery


def fixture():
    camera = {"intrinsics": {"width": 64, "height": 48, "fx": 60., "fy": 60., "cx": 32., "cy": 24.},
              "extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}}
    source = posed_frame(frame_id="source", rgb=np.full((48, 64, 3), 50, np.uint8),
                         depth=np.full((48, 64), 2., np.float32),
                         confidence=np.ones((48, 64), np.float32), camera=camera)
    native = deepcopy(source)
    native["frame_id"] = "recovered-native-0"
    native["rgb"][:] = 240
    source["confidence"][12:36, 20:44] = 0
    prior = deepcopy(native)
    prior["frame_id"] = "prior-small-fill"
    prior["rgb"][:] = 50
    prior["confidence"][:] = 0
    prior["confidence"][12:36, 20:24] = 1
    reviewed = [{"frame": native, "native_result_digest": "sha256:native",
                 "native_frame_index": 0, "review_digest": "sha256:review"}]
    return source, prior, reviewed


def test_recovery_fills_remaining_photo_hole_without_overwriting_the_scene_or_native_output(tmp_path):
    source, prior, reviewed = fixture()
    before = deepcopy([source, prior, reviewed[0]["frame"]])
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed, root=tmp_path)
    assert len(frames) == 1 and report["exportable_samples"] > 300
    actual = frames[0]
    assert actual["confidence"][14:34, 26:42].all()
    assert not actual["confidence"][:, :24].any(), "Existing photo and previous completion must stay fixed"
    assert np.all(actual["rgb"] == 50), "Generated replacement colours cannot overwrite photographed RGB"
    assert report["new_provider_calls"] == 0 and report["whole_scene_complete"] is False
    for current, original in zip([source, prior, reviewed[0]["frame"]], before, strict=True):
        for key in ("rgb", "depth", "confidence"):
            assert np.array_equal(current[key], original[key])


@pytest.mark.parametrize("depth", [0., 10.])
def test_recovery_preserves_scene_extent_instead_of_adding_an_entire_generated_room(tmp_path, depth):
    source, prior, reviewed = fixture()
    reviewed[0]["frame"]["depth"][:] = depth
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed, root=tmp_path)
    assert not frames and report["added_geometry_pixels"] == 0


def test_recovery_reuses_verified_masks_and_refuses_changed_output(tmp_path, monkeypatch):
    source, prior, reviewed = fixture()
    args = dict(frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed, root=tmp_path)
    frames, report = recovery.recover_reviewed_geometry(**args)
    from blueprint_pipeline import atlas_scene_chisel
    monkeypatch.setattr(atlas_scene_chisel, "project_source_depth",
                        lambda *_: pytest.fail("Cached recovery was recomputed"))
    cached, same = recovery.recover_reviewed_geometry(**args)
    assert same == report and np.array_equal(cached[0]["confidence"], frames[0]["confidence"])
    Image.fromarray(np.zeros((48, 64), np.uint8)).save(report["views"][0]["mask_path"])
    with pytest.raises(ValueError, match="output_changed"):
        recovery.recover_reviewed_geometry(**args)


def test_unreviewed_outputs_are_never_recovered(tmp_path):
    with pytest.raises(ValueError, match="review_not_bound"):
        recovery.reviewed_completion_views([{"status": "passed", "review": {"status": "needs_input"}}])
    source, prior, reviewed = fixture()
    reviewed[0].pop("review_digest")
    with pytest.raises(ValueError, match="review_not_bound"):
        recovery.recover_reviewed_geometry(frames=[source], geometry_frames=[source, prior],
                                           reviewed_views=reviewed, root=tmp_path)


def test_bound_review_cannot_admit_a_different_native_result(tmp_path):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    (tmp_path / "result.json").write_text(json.dumps({"digest": "another-result"}))
    report = {"status": "passed", "review": {"status": "passed"}, "generated_roots": [str(tmp_path)],
              "generation": {"native_result_digests": ["reviewed-result"]}}
    report["digest"] = canonical_digest(report, digest_field="digest")
    with pytest.raises(ValueError, match="not_reviewed"):
        recovery.reviewed_completion_views([report])


def test_no_gap_completion_has_no_native_outputs_to_recover():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    report = {"status": "passed", "review": {"status": "not_required"}, "generated_roots": []}
    report["digest"] = canonical_digest(report, digest_field="digest")
    assert recovery.reviewed_completion_views([report]) == []


def exported_fixture(tmp_path):
    from blueprint_pipeline.atlas_depth_splat import write_depth_splat
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    source, _, reviewed = fixture()
    exported = deepcopy(source)
    source['confidence'][:] = 1  # Its estimated depth existed, but was lost at export.
    pose = {'digest': 'pose', 'binding': {'source_video_digest': 'video'},
            'operation': {'id': 'pose'}, 'depth_unit': 'estimated', 'scale_status': 'estimated'}
    splat = tmp_path / 'export'
    write_depth_splat(frames=[exported], result=pose, output_root=splat,
                      generated_result_digests=[], visual_review_digest='review')
    native = reviewed[0]['frame']
    plan = {'schema_version': 'atlas_task_surface_plan.v1',
        'native_bindings': {native['frame_id']: canonical_digest(recovery._frame_binding(native))},
        'proposal': {'status': 'proposed', 'task_extent_summary': 'Complete the visible support',
            'remaining_required_surfaces': ['Support surface'], 'excluded_scene_regions': [],
            'views': [{'frame_id': native['frame_id'], 'reason': 'A reviewed support view', 'regions': [
                {'label': 'support', 'decision': 'include', 'purpose': 'work_surface',
                 'polygon_normalized': [[0, 0], [1, 0], [1, 1], [0, 1]], 'reason': 'Task support'}]}]},
        'plan_path': str(tmp_path / 'regions.json')}
    plan['digest'] = canonical_digest(plan, digest_field='digest')
    (tmp_path / 'regions.json').write_text(json.dumps(plan))
    return source, reviewed, splat, plan


def test_export_loss_can_recover_source_matching_depth_without_freezing_an_old_hypothesis(tmp_path):
    source, reviewed, splat, plan = exported_fixture(tmp_path)
    wrong = deepcopy(source)
    wrong['depth'][:] = 4  # A generated farther plane is not captured free-space authority.
    before = deepcopy([source, wrong, reviewed[0]['frame']])
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source, wrong], reviewed_views=reviewed, root=tmp_path / 'recovery',
        region_plan=plan, exported_splat_root=splat)
    assert len(frames) == 1 and report['exportable_samples'] > 300
    assert frames[0]['confidence'][15:33, 25:39].all()
    assert not frames[0]['confidence'][:, :18].any(), 'Existing exported geometry remains unchanged'
    assert np.all(frames[0]['rgb'] == 50), 'Actual photographed colour wins'
    assert report['revision'] == 'v2-export-depth' and report['new_provider_calls'] == 0
    for current, original in zip([source, wrong, reviewed[0]['frame']], before, strict=True):
        for key in ('rgb', 'depth', 'confidence'):
            assert np.array_equal(current[key], original[key])


def test_export_recovery_keeps_actual_source_free_space_and_requires_an_explicit_region_plan(tmp_path):
    source, reviewed, splat, plan = exported_fixture(tmp_path)
    reviewed[0]['frame']['depth'][:] = 1
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    plan['native_bindings'] = {reviewed[0]['frame']['frame_id']:
                              canonical_digest(recovery._frame_binding(reviewed[0]['frame']))}
    plan['digest'] = canonical_digest(plan, digest_field='digest')
    (tmp_path / 'regions.json').write_text(json.dumps(plan))
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source], reviewed_views=reviewed, root=tmp_path / 'recovery',
        region_plan=plan, exported_splat_root=splat)
    assert not frames and report['added_geometry_pixels'] == 0
    with pytest.raises(ValueError, match='region_plan_required'):
        recovery.recover_reviewed_geometry(frames=[source], geometry_frames=[source], reviewed_views=reviewed,
            root=tmp_path / 'refused', exported_splat_root=splat)


def test_farther_exported_plane_cannot_freeze_a_source_visible_foreground_hole(tmp_path):
    from blueprint_pipeline.atlas_depth_splat import write_depth_splat

    source, reviewed, _, plan = exported_fixture(tmp_path)
    farther = deepcopy(source)
    farther['frame_id'] = 'farther-generated-plane'
    farther['depth'][:] = 4
    pose = {'digest': 'pose', 'binding': {'source_video_digest': 'video'},
            'operation': {'id': 'pose'}, 'depth_unit': 'estimated', 'scale_status': 'estimated'}
    splat = tmp_path/'farther-export'
    write_depth_splat(frames=[farther], result=pose, output_root=splat,
                      generated_result_digests=[], visual_review_digest='review')
    frames, report = recovery.recover_reviewed_geometry(frames=[source], geometry_frames=[source, farther],
        reviewed_views=reviewed, root=tmp_path/'repair-foreground', region_plan=plan, exported_splat_root=splat)
    assert len(frames) == 1 and report['added_geometry_pixels'] > 1000
    assert frames[0]['confidence'][15:33, 25:39].all()
    assert np.all(frames[0]['rgb'] == 50), 'Source colours remain authoritative over the native fill'


def registration_fixture(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_completion, atlas_scene_generate
    from blueprint_pipeline.atlas_depth_splat import write_exr_depth
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    source, _, _ = fixture()
    source['confidence'][:] = 1
    other = deepcopy(source)
    other['frame_id'] = 'other-source'
    other['position'] += [.2, 0, 0]
    other['extrinsics']['position'] = other['position'].tolist()
    generated = deepcopy(source)
    generated['frame_id'] = 'novel'
    stage = tmp_path/'native/generated/0'
    stage.mkdir(parents=True)
    native = {'status': 'completed', 'operation': {'response': {'frames': [{}, {}, {}]}}}
    native['digest'] = canonical_digest(native, digest_field='digest')
    (stage/'result.json').write_text(json.dumps(native))
    plan = {'selected_camera_ids': ['requested-gap-camera'],
            'batches': [{'frame_ids': ['anchor-a', 'anchor-b', 'requested-gap-camera']}]}
    plan['digest'] = canonical_digest(plan, digest_field='digest')
    (stage.parent.parent/'rgbd-completion-plan.json').write_text(json.dumps(plan))
    report = {'status': 'passed', 'review': {'status': 'passed'}, 'generated_roots': [str(stage)],
              'selected_camera_ids': plan['selected_camera_ids'],
              'generation': {'native_result_digests': [native['digest']], 'mask_plan_digest': plan['digest']}}
    report['digest'] = canonical_digest(report, digest_field='digest')
    monkeypatch.setattr(atlas_scene_completion, 'load_generated_views', lambda root, ids:
        (native, [{**deepcopy(generated), 'frame_id': key} for key in ids], []))
    def context(*, frames, root):
        photos, uploads = [], {}
        for i, frame in enumerate(frames):
            root.mkdir(parents=True, exist_ok=True)
            image = root/f'{i}.png'
            Image.fromarray(frame['rgb']).save(image)
            depth = write_exr_depth(root/f'{i}.exr', frame['depth'])
            uploads[f'context-{i}-depth'] = {'path': str(depth), 'sha256': _sha256_file(depth)}
            photos.append({'frame_id': frame['frame_id'], 'source_frame_id': frame['frame_id'],
                'image_path': str(image), 'image_digest': _sha256_file(image),
                'camera': {'intrinsics': frame['intrinsics'], 'extrinsics': frame['extrinsics']}})
        return [], uploads, photos
    monkeypatch.setattr(atlas_scene_generate, 'prepare_rgbd_context', context)
    return [source, other], generated, report


def test_joint_registration_reuses_all_cleaned_sources_excludes_anchors_and_keeps_native_immutable(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    sources, generated, completion = registration_fixture(tmp_path, monkeypatch)
    before = deepcopy(sources)
    calls = []
    def fallback(background, root, source, context, policy, pose_runner, max_polls):
        assert {r['frame_id'] for r in background['frames'][:2]} == {f['frame_id'] for f in sources}
        assert len(background['frames']) == 3, 'Conditioning outputs must not be posed as new views'
        key = background['frames'][-1]['frame_id']
        camera = deepcopy(generated)
        camera['frame_id'] = key
        camera['position'] += [.3, 0, 0]
        camera['extrinsics']['position'] = camera['position'].tolist()
        joint = {'operation_receipt': {'input_usage': {'frames': []}}}
        joint['digest'] = canonical_digest(joint, digest_field='digest')
        calls.append(background)
        return (joint, [*deepcopy(sources), camera], root/'fallback_pose-v3'), {'status': 'completed'}
    monkeypatch.setattr(atlas_scene_automation, '_fallback_pose', fallback)
    registered, receipt = recovery.register_generated_completion(frames=sources, completion=completion,
        root=tmp_path/'registration', pose={'digest': 'pose', 'binding': {'source_video_digest': 'video'}},
        context={}, configuration={'policy': {'pose_input_cost_usd': .01}, 'remaining_cost_usd': 1},
        pose_runner=object(), max_polls=1)
    assert len(calls) == len(registered) == 1
    assert registered[0]['requested_camera_id'] == 'requested-gap-camera'
    assert registered[0]['frame']['position'][0] == pytest.approx(.3, abs=.001)
    assert receipt['source_geometry_modified'] is False and receipt['native_outputs_modified'] is False
    assert receipt['fit']['claim_ceiling'] == 'development_only'
    for original, after in zip(before, sources, strict=True):
        for key in ('rgb', 'depth', 'confidence', 'position', 'rotation'):
            assert np.array_equal(original[key], after[key])


def test_unfinished_joint_registration_returns_same_operation_without_qualifying_geometry(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation

    sources, _, completion = registration_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(atlas_scene_automation, '_fallback_pose',
        lambda *a, **kw: (None, {'status': 'processing', 'operation_id': 'already-admitted'}))
    value, report = recovery.register_generated_completion(frames=sources, completion=completion,
        root=tmp_path/'registration', pose={'digest': 'pose', 'binding': {'source_video_digest': 'video'}},
        context={}, configuration={'policy': {'pose_input_cost_usd': .01}, 'remaining_cost_usd': 1},
        pose_runner=object(), max_polls=1)
    assert value is None and report['operation_id'] == 'already-admitted'
    assert report['stage'] == 'generated_view_registration'
    with pytest.raises(ValueError, match='canonical_pose_runner_required'):
        recovery.register_generated_completion(frames=sources, completion=completion,
            root=tmp_path/'refused', pose={}, context={}, configuration={}, pose_runner=None, max_polls=1)
