"""Novel-view completion must carry photographed RGB AND aligned depth."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_generate as generate
from blueprint_pipeline import atlas_scene_completion as completion
from blueprint_pipeline.atlas_depth_splat import posed_frame, read_exr_depth
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def source(key, x=0.):
    y, u = np.indices((1280, 720))
    depth = (1 + y / 1280).astype(np.float32)
    depth[400:850, 200:500] = 0
    return posed_frame(frame_id=key, rgb=np.stack((u % 256, y % 256, u * 0 + 100), axis=-1).astype(np.uint8),
        depth=depth, confidence=(depth > 0).astype(np.float32), camera={
            'intrinsics': {'width': 720, 'height': 1280, 'fx': 600., 'fy': 600., 'cx': 360., 'cy': 640.},
            'extrinsics': {'position': [x, 0., 0.], 'quaternion': [0., 0., 0., 1.], 'coordinateSystem': 'rub'}})


def test_context_rgb_and_depth_rotate_with_the_camera_without_hidden_crop(tmp_path):
    frame = source('a')
    before = deepcopy(frame)
    contexts, uploads, photos = generate.prepare_rgbd_context(frames=[frame], root=tmp_path)
    assert len(contexts) == 1 and len(uploads) == 2
    context = contexts[0]
    assert context['camera'] == photos[0]['camera']
    rgb = np.asarray(Image.open(uploads['context-0-rgb']['path']))
    assert np.array_equal(rgb, np.rot90(frame['rgb'], -1))
    values = read_exr_depth(Path(uploads['context-0-depth']['path']))
    valid = np.rot90(frame['confidence'] > 0, -1)
    assert np.array_equal(values > 0, valid)
    expected = np.rot90(frame['depth'], -1)
    assert 'depthAsset' in context['depth']
    assert np.allclose(values[valid], expected[valid])
    for name in ('rgb', 'depth', 'confidence'):
        assert np.array_equal(frame[name], before[name])


def test_generation_is_priced_validated_and_gap_limited_before_purchase(tmp_path, monkeypatch):
    frames = [source('a'), source('b', .2)]
    pool = [row for row in completion.camera_candidates(frames) if row['purpose'] == 'wide']
    for index, row in enumerate(pool):
        path = tmp_path / f'gap-{index}.png'
        Image.fromarray(np.full((720, 1280), 255, np.uint8)).save(path)
        row.update(completion_gap_mask_path=str(path), completion_gap_mask_digest=_sha256_file(path),
                   completion_gap_ids=['gap-' + str(index)], completion_gap_pixel_count=720 * 1280)
    calls = []
    def task(**kw):
        calls.append(kw)
        request = kw['request']
        assert kw['task'] == 'atlasGenerate'
        assert request['returnDepth'] and request['enhancePrompt'] is False
        assert len(request['contextFrames']) == 2 and len(request['targetCameras']) == 2
        assert all(set(row) == {'camera', 'depth', 'imageAsset'} for row in request['contextFrames'])
        assert len(kw['uploads']) == 4 and kw['price_review']['maximum_cost_usd'] == .1
        assert not any('maskAsset' in row for row in request['contextFrames'])
        return {'status': 'processing', 'operation_id': 'retained'}
    args = dict(frames=frames, selected=pool, candidates=pool, root=tmp_path / 'completion',
        pose={'digest': 'pose', 'binding': {'source_video_digest': 'source'}}, context={},
        configuration={'remaining_cost_usd': 1.59, 'remaining_requests': 40, 'policy': {'rgbd_view_cost_usd': .05,
            'mesh_export_cost_usd': .5, 'rate_card_reference': 'rates', 'reviewed_by': 'test', 'reviewed_on': '2026-10-23'}},
        task_runner=task, max_polls=1)
    with pytest.raises(ValueError, match='remaining_budget_insufficient'):
        generate.generate_rgbd_completion(**args)
    assert not calls
    args['configuration']['remaining_cost_usd'] = 2.
    args['configuration']['remaining_requests'] = 2
    with pytest.raises(ValueError, match='request_budget_insufficient:required=5,remaining=2'):
        generate.generate_rgbd_completion(**args)
    assert not calls
    import json
    forecast = json.loads((args['root'] / 'request-budget-preflight.json').read_text())
    assert forecast['minimum_remaining_requests'] == 5 and not forecast['reservation_created']
    args['configuration']['remaining_requests'] = 40
    value, result = generate.generate_rgbd_completion(**args)
    assert value is None and result['operation_id'] == 'retained'
    # Polling a retained operation must not depend on request headroom for new work.
    submission = args['root'] / 'generated/0/atlas_pose/submission.json'
    submission.parent.mkdir(parents=True)
    submission.write_text(json.dumps({'status': 'submitted', 'operation_id': 'retained'}))
    args['configuration']['remaining_requests'] = 0
    value, result = generate.generate_rgbd_completion(**args)
    assert value is None and result['operation_id'] == 'retained' and len(calls) == 2
    # Native completion must reach source-priority admission, not stop at downloads.
    returned = []
    for frame, target in zip(frames, pool, strict=True):
        pixels = np.rot90(frame['rgb'], -1).copy()
        pixels[:] = 255
        depth = np.rot90(np.where(frame['depth'] > 0, frame['depth'], 1.5), -1).copy()
        returned.append(posed_frame(frame_id='native-' + frame['frame_id'], rgb=pixels,
            depth=depth, confidence=np.ones(depth.shape), camera=target['camera']))
    native_images = [{'image_path': 'native-' + str(index), 'image_digest': 'digest-' + str(index)}
                     for index in range(len(returned))]
    monkeypatch.setattr(completion, 'load_generated_views', lambda *_: ({'digest': 'native'}, returned, native_images))
    args['task_runner'] = lambda **kw: {'status': 'completed'}
    (generated, images), result = generate.generate_rgbd_completion(**args)
    assert result['generation_method'] == 'atlasGenerate_posed_rgbd_photographs'
    assert result['known_pixels_changed'] == 0 and result['generated_geometry_pixel_count'] > 0
    assert len(generated) == len(images) == 2
    assert images[0]['native_image_path'] == 'native-0'
    for row, original in zip(generated, frames, strict=True):
        assert np.array_equal(row['rgb'], np.rot90(original['rgb'], -1))
        assert not np.any((row['confidence'] > 0) & np.rot90(original['confidence'] > 0, -1))


def test_generate_refuses_unbounded_whole_image_admission(tmp_path):
    frames = [source('a'), source('b', .2)]
    pool = [row for row in completion.camera_candidates(frames) if row['purpose'] == 'wide']
    with pytest.raises(ValueError, match='named_gap_masks_required'):
        generate.generate_rgbd_completion(frames=frames, selected=pool, candidates=pool,
            root=tmp_path, pose={}, context={}, configuration={}, task_runner=lambda **_: pytest.fail('Purchased'), max_polls=1)


def test_feedback_sends_all_reviewed_rgbd_and_does_not_freeze_a_photographed_export_gap(tmp_path):
    frames = [source('a'), source('b', .2)]
    reference = source('reviewed-reference', .1)
    reference['depth'][400:850, 200:500] = 1.5
    reference['confidence'][:] = 1
    pool = [row for row in completion.camera_candidates(frames) if row['purpose'] == 'wide']
    mask = np.zeros((720, 1280), np.uint8)
    mask[300:330, 500:530] = 255
    path = tmp_path / 'actual-export-gap.png'
    Image.fromarray(mask).save(path)
    target = {**pool[0], 'completion_gap_mask_path': str(path),
        'completion_gap_mask_digest': _sha256_file(path), 'completion_gap_ids': ['foreground-gap'],
        'completion_gap_pixel_count': int(np.count_nonzero(mask))}
    calls = []
    def task(**kw):
        calls.append(kw)
        assert len(kw['request']['contextFrames']) == 3
        assert len(kw['uploads']) == 6
        assert all('depth' in r for r in kw['request']['contextFrames'])
        assert not any('maskAsset' in r for r in kw['request']['contextFrames'])
        return {'status': 'processing', 'operation_id': 'targeted-view'}
    root = tmp_path / 'run'
    generate.generate_rgbd_completion(frames=frames, reference_frames=[*frames, reference],
        geometry_frames=frames, exported_gap_feedback=True, selected=[target], candidates=pool,
        root=root, pose={'digest': 'pose', 'binding': {'source_video_digest': 'source'}}, context={},
        configuration={'remaining_cost_usd': 5, 'remaining_requests': 40, 'policy': {'rgbd_view_cost_usd': .05,
            'mesh_export_cost_usd': .5, 'rate_card_reference': 'rates',
            'reviewed_by': 'test', 'reviewed_on': '2026-10-23'}}, task_runner=task, max_polls=1)
    import json
    plan = json.loads((root / 'rgbd-completion-plan.json').read_text())
    prepared = next(row for row in plan['prepared_views'] if row['frame_id'] == target['camera_id'])
    known = np.asarray(Image.open(prepared['source_depth_context_path']))
    assert not known[mask > 0].any(), 'A farther retained surface must not freeze the named foreground hole'
    assert len(calls) == 1


def test_target_facing_away_from_known_photos_refuses_before_purchase(tmp_path):
    frames = [source('a'), source('b', .2)]
    candidates = [row for row in completion.camera_candidates(frames) if row['purpose'] == 'wide']
    selected = [deepcopy(candidates[0])]
    selected[0].update(camera_id='unsupported-gap-view', purpose='gap_target')
    selected[0]['camera']['extrinsics']['position'] = [1000., 0., 0.]
    path = tmp_path / 'gap.png'
    mask = np.zeros((720, 1280), np.uint8)
    mask[300:350, 500:550] = 255
    Image.fromarray(mask).save(path)
    selected[0].update(completion_gap_mask_path=str(path), completion_gap_mask_digest=_sha256_file(path),
                       completion_gap_ids=['actual-gap'], completion_gap_pixel_count=2500)
    with pytest.raises(ValueError, match='target_context_insufficient'):
        generate.generate_rgbd_completion(frames=frames, selected=selected, candidates=candidates,
            root=tmp_path / 'run', pose={}, context={}, configuration={},
            task_runner=lambda **_: pytest.fail('Purchased an unsupported camera'), max_polls=1)
