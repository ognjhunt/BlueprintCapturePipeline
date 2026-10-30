"""Full photo coverage and paid Chisel boundaries, without external requests."""
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_chisel as chisel
from blueprint_pipeline.atlas_depth_splat import posed_frame


def source(key='source', x=0):
    y, u = np.indices((1280, 720))
    rgb = np.stack((u % 256, y % 256, (u + y) % 256), axis=-1).astype(np.uint8)
    return posed_frame(frame_id=key, rgb=rgb, depth=np.full((1280, 720), 2., np.float32),
        confidence=np.ones((1280, 720), np.float32), camera={
            'intrinsics': {'width': 720, 'height': 1280, 'fx': 600., 'fy': 600., 'cx': 360., 'cy': 640.},
            'extrinsics': {'position': [x, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}})


def candidates(frames):
    return [{'camera_id': f['frame_id'] + '-wide', 'source_frame_id': f['frame_id'], 'purpose': 'wide',
             'camera': {'intrinsics': {'width': 1280, 'height': 720, 'fx': 600., 'fy': 600., 'cx': 640., 'cy': 360.},
                        'extrinsics': deepcopy(f['extrinsics'])}} for f in frames]


def test_source_tiles_retain_entire_photo_with_calibrated_rays_and_centered_anchor(tmp_path):
    frame = source()
    original = deepcopy(frame)
    photos = chisel.prepare_source_photographs(frames=[frame], root=tmp_path)
    assert len(photos) == 1
    yy, xx = np.indices((720, 1280))
    covered = np.zeros((720, 1280), bool)
    for row in photos:
        left, top, right, bottom = row['source_rectangle']
        covered |= (xx + .5 >= left) & (xx + .5 < right) & (yy + .5 >= top) & (yy + .5 < bottom)
        assert Image.open(row['image_path']).size == (1280, 720)
        k = row['camera']['intrinsics']
        sx, sy = 100.5, top + 120.5
        tx, ty = (sx - left) * row['pixel_scale'], (sy - top) * row['pixel_scale']
        assert (tx - k['cx']) / k['fx'] == pytest.approx((sx - 640) / 600)
        assert (ty - k['cy']) / k['fy'] == pytest.approx((sy - 360) / 600)
        assert row['clockwise_quarter_turn_with_camera'] is True
        assert np.array_equal(np.asarray(Image.open(row['image_path'])), np.rot90(frame['rgb'], -1))
        from blueprint_pipeline.atlas_depth_splat import _rotation_matrix
        for u, v in ((.5, .5), (719.5, 1279.5), (360.5, 640.5)):
            original_ray = np.array([(u - 360) / 600, -(v - 640) / 600, -1.]) @ frame['rotation'].T
            ru, rv = 1280 - v, u
            rotated_ray = np.array([(ru - k['cx']) / k['fx'], -(rv - k['cy']) / k['fy'], -1.])
            rotated_ray = rotated_ray @ _rotation_matrix(row['camera']['extrinsics']['quaternion']).T
            assert np.allclose(original_ray, rotated_ray), 'Pixel rotation must not change its physical ray'
    assert covered.all(), 'Native cropping must not discard source-image context'
    assert any(row['camera']['intrinsics']['cx'] == pytest.approx(640)
               and row['camera']['intrinsics']['cy'] == pytest.approx(360) for row in photos)
    for key in ('rgb', 'depth', 'confidence'):
        assert np.array_equal(frame[key], original[key])


def test_native_rollout_limit_batches_targets_not_total_scene_coverage():
    pool = candidates([source('a'), source('b', .2)])
    selected = [{**pool[0], 'camera_id': f'novel-{i}', 'purpose': 'room'} for i in range(70)]
    groups = chisel.target_batches(selected=selected, candidates=pool, source_count=30)
    assert [len(group) for group in groups] == [17, 17, 17, 17, 12]
    assert [row for group in groups for row in group[2:]] == selected
    assert all(len(group) * 2 + 30 <= 64 and group[:2] == pool for group in groups)
    with pytest.raises(ValueError, match='rollout_capacity'):
        chisel.target_batches(selected=selected, candidates=pool, source_count=62)
    with pytest.raises(ValueError, match='observed_translation_missing'):
        chisel.target_batches(selected=selected, candidates=pool[:1], source_count=5)


def test_chisel_uploads_only_clean_photos_and_prices_rgbd_once_before_purchase(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_completion
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.common import write_json

    frames = [source('a'), source('b', .2)]
    pool = candidates(frames)
    calls = []

    def task(**kw):
        calls.append(kw)
        request = kw['request']
        assert kw['task'] == 'atlasChisel'
        assert len(request['contextFrames']) == 2 and len(request['sourceFrames']) == 2
        assert all(set(row) == {'camera', 'depth'} and 'logDepthAsset' in row['depth']
                   for row in request['contextFrames'])
        assert request['returnDepth'] is True and request['enhancePrompt'] is False
        assert kw['price_review']['maximum_cost_usd'] == .1
        assert all('maskAsset' not in f and 'depth' not in f for f in request['sourceFrames'])
        assert set(kw['uploads']) == {'photo-0', 'photo-1', 'depth-0', 'depth-1'}
        assert 'bare continuous architectural' in request['prompt'] and 'newly revealed' in request['prompt'].lower()
        for row in kw['uploads'].values():
            assert Image.open(row['path']).size == (1280, 720)
        return {'status': 'processing'}

    args = dict(frames=frames, selected=pool, candidates=pool, root=tmp_path / 'completion',
        pose={'digest': 'pose', 'binding': {'source_video_digest': 'source'}}, context={},
        configuration={'remaining_cost_usd': 1.59, 'policy': {'rgbd_view_cost_usd': .05, 'mesh_export_cost_usd': .5,
            'rate_card_reference': 'retained rate card', 'reviewed_by': 'test', 'reviewed_on': '2026-10-23'}},
        task_runner=task, max_polls=1)
    with pytest.raises(ValueError, match='remaining_budget_insufficient'):
        chisel.generate_chisel_completion(**args)
    assert not calls
    args['configuration']['remaining_cost_usd'] = 1.6
    generated, report = chisel.generate_chisel_completion(**args)
    assert generated is None and report['status'] == 'processing' and len(calls) == 1
    import json
    plan = json.loads((tmp_path / 'completion/chisel-completion-plan.json').read_text())
    assert plan['native_keep_masks_sent'] is False
    # Adopt a retained completed prefix without charging it in the new budget.
    stage = tmp_path / 'completion/generated/0'
    stage.mkdir(parents=True)
    retained = {'status': 'completed', 'binding': {'source_video_digest': 'source'}, 'outputs': []}
    retained['digest'] = canonical_digest(retained, digest_field='digest')
    write_json(stage / 'result.json', retained)
    (stage / 'atlas_pose').mkdir()
    write_json(stage / 'atlas_pose/submission.json', {'status': 'completed', 'result_digest': retained['digest']})
    # Load a hermetic native result through the real preservation/fusion boundary.
    returned = []
    for i, row in enumerate(pool):
        camera = row['camera']
        returned.append(posed_frame(frame_id='completion-' + row['camera_id'], camera=camera,
            rgb=np.full((720, 1280, 3), 200, np.uint8), depth=np.full((720, 1280), 2., np.float32),
            confidence=np.ones((720, 1280), np.float32)))
    monkeypatch.setattr(atlas_scene_completion, 'load_generated_views', lambda *a: (retained, returned, []))
    args['configuration']['remaining_cost_usd'] = 1.5
    args['task_runner'] = lambda **kw: {'status': 'completed'}
    (generated, images), report = chisel.generate_chisel_completion(**args)
    assert len(generated) == len(images) == 2
    assert report['generation_method'] == 'atlasChisel_posed_photographs_and_source_depth_no_keep_masks'
    assert report['source_geometry_modified'] is False and report['known_pixels_changed'] == 0


def test_depth_projection_keeps_optical_z_and_marks_unseen_rays_empty(tmp_path):
    from scipy.spatial.transform import Rotation
    from blueprint_pipeline.atlas_depth_splat import _rotation_matrix

    frame = source()
    camera = candidates([frame])[0]['camera']
    camera['extrinsics']['quaternion'] = Rotation.from_euler('Y', 15, degrees=True).as_quat().tolist()
    depth = chisel.project_source_depth([frame], camera)
    k = camera['intrinsics']
    yy, xx = np.indices(depth.shape)
    target_rays = np.stack(((xx + .5 - k['cx']) / k['fx'], (yy + .5 - k['cy']) / k['fy'], np.ones(depth.shape)), axis=-1)
    c = np.diag([1., -1., -1.])
    rotation = c @ _rotation_matrix(camera['extrinsics']['quaternion']).T @ frame['rotation'] @ c
    expected_source_z = (target_rays @ rotation)[..., 2]
    assert np.allclose((depth * expected_source_z)[depth > 0], 2., atol=1e-5)
    assert (depth > 0).any() and (depth == 0).any()
    frame['confidence'][:] = 0
    assert not chisel.project_source_depth([frame], camera).any()
    row = {'camera_id': 'empty', 'camera': camera}
    record = chisel.prepare_depth_context(frames=[frame], selected=[row], root=tmp_path)['empty']
    assert record['known_depth_pixel_count'] == 0 and Image.open(record['path']).mode == 'L'
    assert not np.asarray(Image.open(record['path'])).any()
