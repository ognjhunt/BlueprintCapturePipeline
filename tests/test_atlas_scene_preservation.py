"""A generated appliance must not replace known countertop pixels or geometry."""
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline import atlas_scene_preservation as preservation


def camera(width=6, height=6, x=0):
    return {'intrinsics': {'width': width, 'height': height, 'fx': 6., 'fy': 6.,
                            'cx': width / 2, 'cy': height / 2},
            'extrinsics': {'position': [x, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}


def frame(key='source', depth=2., colour=(0, 150, 20), cam=None):
    cam = cam or camera()
    k = cam['intrinsics']
    return posed_frame(frame_id=key, rgb=np.full((k['height'], k['width'], 3), colour, np.uint8),
        depth=np.full((k['height'], k['width']), depth, np.float32),
        confidence=np.ones((k['height'], k['width']), np.float32), camera=cam)


def test_same_camera_keeps_every_rgb_pixel_even_when_depth_is_missing():
    source = frame()
    source['depth'][:] = np.nan
    source['confidence'][:] = 0
    source['rgb'][0, 0] = (255, 0, 0)
    rgb, keep = preservation.project_known_context([source], camera())
    assert keep.all() and np.array_equal(rgb, source['rgb'])


def test_wider_output_preserves_source_rays_without_stretching():
    source = frame()
    source['rgb'][:, 0] = (255, 0, 0)
    rgb, keep = preservation.project_known_context([source], camera(width=12))
    assert keep[:, 3:9].all() and not keep[:, :3].any() and not keep[:, 9:].any()
    assert np.array_equal(rgb[:, 3:9], source['rgb'])


def test_rotation_at_observed_center_keeps_photographed_rays_without_depth_or_rear_wraparound():
    from scipy.spatial.transform import Rotation

    cam = camera(width=64, height=64)
    cam['intrinsics'].update(fx=60., fy=60.)
    source = frame(cam=cam)
    source['rgb'][30:34, 30:34] = (255, 0, 0)
    source['depth'][:] = np.nan
    source['confidence'][:] = 0
    target = deepcopy(cam)
    target['extrinsics']['quaternion'] = Rotation.from_euler('Y', 15, degrees=True).as_quat().tolist()
    rgb, keep = preservation.project_known_context([source], target)
    landmark = (rgb[..., 0] > 180) & (rgb[..., 1] < 50) & keep
    yy, xx = np.nonzero(landmark)
    assert len(xx) and 46 < xx.mean() < 50 and 30 < yy.mean() < 33
    target['extrinsics']['quaternion'] = Rotation.from_euler('Y', 180, degrees=True).as_quat().tolist()
    _, rear = preservation.project_known_context([source], target)
    assert not rear.any(), 'A photograph behind the target camera must never wrap into its keep mask'


def test_every_masked_batch_keeps_all_source_photographs_without_limiting_total_views():
    from blueprint_pipeline.atlas_background_automation import batches

    anchors = [{'frame_id': f'source-{i}'} for i in range(6)]
    novel = [{'frame_id': f'novel-{i}'} for i in range(29)]
    groups = batches([*anchors, *novel], fixed_context=anchors)
    assert [len(group) for group in groups] == [16, 16, 15]
    assert all(group[:6] == anchors for group in groups)
    assert [row for group in groups for row in group[6:]] == novel
    # The vendor recommendation is not an input-frame ceiling.
    larger = [{'frame_id': f'source-{i}'} for i in range(18)]
    assert all(group[:18] == larger for group in batches([*larger, *novel[:2]], fixed_context=larger))


def test_full_source_context_overhead_is_priced_before_any_provider_call(tmp_path, monkeypatch):
    candidates = [{'camera_id': f'source-{i}', 'source_frame_id': str(i), 'purpose': 'wide',
                   'camera': camera(width=1280, height=720, x=.2 * i)} for i in range(6)]
    selected = [{'camera_id': f'novel-{i}', 'source_frame_id': '0', 'purpose': 'room',
                 'camera': candidates[0]['camera']} for i in range(21)]
    def prepared(**kw):
        return [{'frame_id': row['camera_id'], 'source_frame_id': row['source_frame_id'],
                 'camera': row['camera'], 'image_path': 'unused.png', 'image_digest': 'unused',
                 'keep_mask_path': 'unused-mask.png', 'keep_mask_digest': 'unused'} for row in kw['selected']]
    monkeypatch.setattr(preservation, 'prepare_completion_views', prepared)
    purchases = []
    def task(**kw):
        purchases.append(kw)
        return {'status': 'processing'}
    args = dict(frames=[], selected=selected, candidates=candidates, root=tmp_path,
        pose={'digest': 'pose', 'binding': {'source_video_digest': 'video'}}, context={},
        configuration={'remaining_cost_usd': 3., 'policy': {'rgbd_view_cost_usd': .05,
            'rate_card_reference': 'test', 'reviewed_by': 'test', 'reviewed_on': '2026-10-23'}},
        task_runner=task, max_polls=1)
    with pytest.raises(ValueError, match='remaining_budget_insufficient'):
        preservation.generate_masked_completion(**args)
    assert not purchases
    # Three native groups contain 16 + 16 + 9 paid views, including repeated
    # source conditioning. All $4.10, plus the review reserve, must fit first.
    args['configuration']['remaining_cost_usd'] = 5.101
    preservation.generate_masked_completion(**args)
    assert len(purchases) == 1
    assert len(purchases[0]['request']['contextFrames']) == 16
    assert purchases[0]['price_review']['maximum_cost_usd'] == 1.6


def test_projection_keeps_nearer_surface_in_front_of_farther_surface():
    near, far = frame(colour=(255, 0, 0)), frame(depth=4, colour=(0, 0, 255))
    rgb, keep = preservation.project_known_context([near, far], camera(x=.01))
    assert keep.all() and np.all(rgb == np.array([255, 0, 0]))


def test_novel_projection_requires_cross_view_depth_and_colour_support_without_changing_source():
    a, b = frame('a'), frame('b')
    a['rgb'][2, 2] = (0, 0, 0)
    before = deepcopy(a)
    projected = preservation.supported_projection_frames([a, b])
    assert projected[0]['confidence'][1, 1] > 0
    assert projected[0]['confidence'][2, 2] == 0
    assert projected[1]['confidence'][2, 2] == 0
    assert not projected[0]['confidence'][0].any()
    for key in ('rgb', 'depth', 'confidence'):
        assert np.array_equal(a[key], before[key])
    # Depth disagreement and a single unsupported observation cannot become
    # locked new-camera pixels, even with positive native confidence.
    assert not preservation.supported_projection_frames([a, frame(depth=4.)])[0]['confidence'].any()
    assert not preservation.supported_projection_frames([a])[0]['confidence'].any()


def test_novel_context_drops_thin_projection_noise_and_neutralizes_fillable_pixels(tmp_path):
    a, b = frame('a'), frame('b')
    a['confidence'][:] = 0
    b['confidence'][:] = 0
    a['confidence'][2:4, 2:4] = b['confidence'][2:4, 2:4] = 1
    selected = [{'camera_id': 'novel', 'source_frame_id': 'a', 'camera': camera(x=.01)},
                {'camera_id': 'original', 'source_frame_id': 'a', 'camera': camera()}]
    rows = preservation.prepare_completion_views(frames=[a, b], selected=selected, root=tmp_path)
    assert not np.asarray(Image.open(rows[0]['keep_mask_path'])).any()
    assert np.all(np.asarray(Image.open(rows[0]['image_path'])) == 127)
    assert rows[0]['projection_basis'] == 'cross_view_supported_rgbd_dense_patches'
    assert np.asarray(Image.open(rows[1]['keep_mask_path'])).all()
    assert np.array_equal(np.asarray(Image.open(rows[1]['image_path'])), a['rgb'])
    assert rows[1]['projection_basis'] == 'source_center_rgb_homography'


@pytest.mark.parametrize('depth,allowed', [(1., False), (2., False), (4., True)])
def test_generated_geometry_cannot_fill_observed_free_space_or_replace_known_surface(depth, allowed):
    reference = frame()
    generated = frame('generated', depth=depth)
    mask = preservation.unobserved_geometry_mask(generated, [reference])
    assert bool(mask.all()) is allowed and bool(mask.any()) is allowed


def test_generated_geometry_is_allowed_where_reference_depth_is_unknown_or_outside_capture():
    reference = frame()
    reference['depth'][:] = np.nan
    assert preservation.unobserved_geometry_mask(frame('generated'), [reference]).all()
    reference = frame()
    assert preservation.unobserved_geometry_mask(frame('generated', cam=camera(x=20)), [reference]).all()


def test_generated_space_cannot_expand_beyond_observed_scene_bounds():
    references = [frame('near'), frame('distant', depth=4., cam=camera(x=20))]
    bounds = preservation.observed_scene_bounds(references)
    assert bounds['additional_space_beyond_observed_bounds'] is False
    assert preservation.unobserved_geometry_mask(frame('inside', depth=3.), references, extent_bounds=bounds).any()
    assert not preservation.unobserved_geometry_mask(frame('outside', depth=10.), references, extent_bounds=bounds).any()


def test_hallucinated_countertop_is_copied_back_and_excluded_from_generated_geometry(tmp_path):
    source = frame()
    original = deepcopy(source)
    target = camera(width=12)
    rows = preservation.prepare_completion_views(frames=[source], selected=[{
        'camera_id': 'wide', 'source_frame_id': 'source', 'camera': target}], root=tmp_path / 'input')
    assert Image.open(rows[0]['keep_mask_path']).mode == 'L'
    hallucinated = frame('generated', colour=(255, 0, 255), cam=target)
    native_rgb = hallucinated['rgb'].copy()
    fixed, image, receipt, paths = preservation.preserve_generated_view(
        generated=hallucinated, prepared=rows[0], references=[source], output_root=tmp_path / 'derived',
        native_result_digest='native-operation', native_frame_index=0)
    assert np.array_equal(fixed['rgb'][:, 3:9], source['rgb'])
    assert np.array_equal(fixed['rgb'][:, :3], native_rgb[:, :3])
    assert not fixed['confidence'][:, 3:9].any() and fixed['confidence'][:, :3].all()
    assert receipt['known_pixels_changed'] == 0 and receipt['native_outputs_modified'] is False
    assert receipt['source_geometry_modified'] is False
    for key in ('rgb', 'depth', 'confidence'):
        assert np.array_equal(source[key], original[key])
    assert np.array_equal(hallucinated['rgb'], native_rgb)
    assert len(paths) == 4 and image['novel_view'] is True
    assert np.array_equal(np.asarray(Image.open(image['image_path'])), fixed['rgb'])


def test_changed_preservation_inputs_are_refused(tmp_path):
    source = frame()
    rows = preservation.prepare_completion_views(frames=[source], selected=[{
        'camera_id': 'wide', 'source_frame_id': 'source', 'camera': camera()}], root=tmp_path)
    Image.new('RGB', (6, 6), 'black').save(rows[0]['image_path'])
    with pytest.raises(ValueError, match='preservation_inputs_changed'):
        preservation.preserve_generated_view(generated=source, prepared=rows[0], references=[source],
            output_root=tmp_path / 'out', native_result_digest='native', native_frame_index=0)


def test_protected_photo_pixels_do_not_freeze_missing_source_depth(tmp_path):
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    source = frame()
    source['depth'][2:4, 2:4] = np.nan
    source['confidence'][2:4, 2:4] = 0
    row = preservation.prepare_completion_views(frames=[source], selected=[{
        'camera_id': 'hole', 'source_frame_id': 'source', 'camera': camera()}], root=tmp_path / 'input')[0]
    path = tmp_path / 'source-depth.png'
    Image.fromarray(np.where(np.isfinite(source['depth']), 255, 0).astype(np.uint8)).save(path)
    row.update(source_depth_context_path=str(path), source_depth_context_digest=_sha256_file(path))
    fixed, _, receipt, _ = preservation.preserve_generated_view(
        generated=frame('generated', colour=(255, 0, 255)), prepared=row, references=[source],
        output_root=tmp_path / 'out', native_result_digest='native', native_frame_index=0,
        extent_bounds=preservation.observed_scene_bounds([source]))
    assert np.array_equal(fixed['rgb'], source['rgb']), 'All photographed colour stays fixed'
    assert fixed['confidence'][2:4, 2:4].all(), 'Missing depth can receive new candidate geometry'
    assert not fixed['confidence'][:2].any(), 'Known geometry stays unchanged'
    assert receipt['generated_source_depth_hole_pixel_count'] == 4


def test_masked_completion_uses_paired_keep_masks_and_reuses_completed_operation(tmp_path, monkeypatch):
    import json
    from blueprint_pipeline import atlas_depth_splat
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from blueprint_pipeline.website_atlas_tasks import validate_task_request

    references = [frame('a'), frame('b', cam=camera(x=.2))]
    candidates = []
    for f in references:
        target = camera(width=1280, height=720, x=float(f['position'][0]))
        target['intrinsics'].update(fx=720., fy=720.)
        candidates.append({'camera_id': f['frame_id'] + '-wide', 'source_frame_id': f['frame_id'],
                           'purpose': 'wide', 'camera': target})
    monkeypatch.setattr(atlas_depth_splat, 'read_exr_depth', lambda _: np.full((720, 1280), 2., np.float32))
    purchases = []

    def task(**kw):
        validate_task_request(kw['task'], kw['request'])
        assert kw['task'] == 'atlasMasked' and kw['request']['enhancePrompt'] is False
        assert kw['request']['returnDepth'] is True
        assert kw['price_review']['maximum_cost_usd'] == .2
        for i, context in enumerate(kw['request']['contextFrames']):
            assert context['camera'] == kw['request']['targetCameras'][i]
            image = Image.open(kw['uploads'][f'image-{i}']['path'])
            mask = Image.open(kw['uploads'][f'mask-{i}']['path'])
            assert image.size == mask.size == (1280, 720) and mask.mode == 'L'
        root = kw['output_root']
        if (root / 'result.json').exists():
            return {'status': 'completed'}
        purchases.append(kw)
        root.mkdir(parents=True)
        outputs, returned = [], []
        for i, target in enumerate(kw['request']['targetCameras']):
            image, depth = root / f'{i}.png', root / f'{i}.exr'
            Image.new('RGB', (1280, 720), (255, 0, 255)).save(image)
            depth.write_bytes(b'fake-native-depth')
            outputs += [{'role': role, 'path': path.name, 'sha256': _sha256_file(path), 'bytes': path.stat().st_size}
                        for role, path in ((f'frames-{i}-imageAsset', image), (f'frames-{i}-depth-depthAsset', depth))]
            returned.append({'camera': target})
        result = {'status': 'completed', 'binding': {'source_video_digest': 'source'}, 'outputs': outputs,
                  'operation': {'id': 'fake-masked', 'done': True, 'response': {'frames': returned}}}
        result['digest'] = canonical_digest(result, digest_field='digest')
        (root / 'result.json').write_text(json.dumps(result))
        (root / 'atlas_pose').mkdir()
        (root / 'atlas_pose/submission.json').write_text(json.dumps({'status': 'completed', 'result_digest': result['digest']}))
        return {'status': 'completed'}

    args = dict(frames=references, selected=candidates, candidates=candidates, root=tmp_path / 'complete',
        pose={'digest': 'pose', 'binding': {'source_video_digest': 'source'}}, context={},
        configuration={'remaining_cost_usd': 5, 'policy': {'rgbd_view_cost_usd': .05,
            'rate_card_reference': 'fake rate', 'reviewed_by': 'test', 'reviewed_on': '2026-10-23'}},
        task_runner=task, max_polls=1, removed_objects=['dishwasher'])
    (generated, images), report = preservation.generate_masked_completion(**args)
    assert len(generated) == len(images) == 2 and len(purchases) == 1
    assert report['known_pixels_changed'] == 0 and report['source_geometry_modified'] is False
    assert len(report['generated_roots']) == len(report['native_result_digests']) == 1
    assert 'dishwasher' in purchases[0]['request']['prompt']
    for f in generated:
        assert np.all(f['rgb'][:, 300:980] == (0, 150, 20))
        assert not f['confidence'][:, 300:980].any()
    # An already paid prefix is not included in the remaining-spend preflight.
    args['configuration']['remaining_cost_usd'] = 1.01
    _, resumed = preservation.generate_masked_completion(**args)
    assert len(purchases) == 1 and resumed['digest'] == report['digest']
    args['configuration']['remaining_cost_usd'] = .5
    with pytest.raises(ValueError, match='remaining_budget_insufficient'):
        preservation.generate_masked_completion(**args)
    assert len(purchases) == 1


def test_novel_view_review_rejects_invented_objects_on_known_surfaces():
    from blueprint_pipeline.website_image_completion import _review_prompt

    prompt = _review_prompt([], frames=[{'novel_view': True}])
    assert 'invented, substituted or duplicated' in prompt
    assert 'consistent_background false' in prompt
    assert 'minor blur, seams, speckling' in prompt
    assert 'clear task obstruction' in prompt
    assert 'generic cabinet door' in prompt
