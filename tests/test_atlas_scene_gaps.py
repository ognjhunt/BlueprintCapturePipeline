"""Only named, coherent reconstruction gaps may receive new geometry."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_gaps as gaps
from blueprint_pipeline import atlas_scene_preservation as preservation
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def test_blank_exterior_and_raster_pinholes_are_not_scene_completion_targets():
    depth = np.zeros((80, 100), np.float32)
    depth[10:70, 10:90] = 2
    depth[25:40, 35:60] = 0  # A coherent bounded surface gap.
    depth[50, 50] = 0  # A sampling pinhole, not a new scene surface.
    labels, regions = gaps.gap_labels(depth, np.zeros(depth.shape, bool))
    assert len(regions) == 1 and labels[30, 40] == 1
    assert not labels[:10].any() and not labels[:, :10].any()
    assert labels[50, 50] == 0


def test_photographed_missing_depth_can_reach_image_edge_without_filling_unseen_canvas():
    depth = np.zeros((60, 80), np.float32)
    photo = np.zeros(depth.shape, bool)
    photo[:, :40] = True
    labels, regions = gaps.gap_labels(depth, photo)
    assert len(regions) == 1 and labels[:, :40].all() and not labels[:, 40:].any()


def test_agent_must_name_real_regions_and_mask_bytes_cannot_change(tmp_path):
    labels = np.zeros((30, 40), np.uint16)
    labels[5:12, 5:12], labels[18:25, 25:32] = 1, 2
    path = tmp_path / 'labels.png'
    Image.fromarray(labels).save(path)
    view = {'label_path': str(path), 'label_sha256': _sha256_file(path),
            'gaps': [{'gap_id': 'left', 'label': 1}, {'gap_id': 'right', 'label': 2}]}
    with pytest.raises(ValueError, match='named_gaps_required'):
        gaps.select_gap_mask(view, ['whole-room'], tmp_path / 'mask.png')
    selected = gaps.select_gap_mask(view, ['left'], tmp_path / 'mask.png')
    mask = np.asarray(Image.open(selected['completion_gap_mask_path'])) > 0
    assert np.array_equal(mask, labels == 1)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='gap_survey_changed'):
        gaps.select_gap_mask(view, ['left'], tmp_path / 'other.png')


def test_native_full_image_contributes_only_selected_gap_and_keeps_known_rgb(tmp_path):
    cam = {'intrinsics': {'width': 20, 'height': 16, 'fx': 20., 'fy': 20., 'cx': 10., 'cy': 8.},
           'extrinsics': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}
    source = posed_frame(frame_id='source', rgb=np.full((16, 20, 3), 50, np.uint8),
                         depth=np.zeros((16, 20), np.float32), confidence=np.zeros((16, 20)), camera=cam)
    prepared = preservation.prepare_completion_views(frames=[source],
        selected=[{'camera_id': 'source', 'source_frame_id': 'source', 'camera': cam}], root=tmp_path / 'inputs')[0]
    depth = tmp_path / 'depth.png'
    Image.fromarray(np.zeros((16, 20), np.uint8)).save(depth)
    mask = np.zeros((16, 20), np.uint8)
    mask[4:12, 5:15] = 255
    path = tmp_path / 'gap.png'
    Image.fromarray(mask).save(path)
    prepared.update(source_depth_context_path=str(depth), source_depth_context_digest=_sha256_file(depth),
                    completion_gap_mask_path=str(path), completion_gap_mask_digest=_sha256_file(path),
                    completion_gap_ids=['floor-gap'], completion_gap_pixel_count=80)
    generated = deepcopy(source)
    generated.update(frame_id='generated', rgb=np.full((16, 20, 3), 255, np.uint8),
                     depth=np.full((16, 20), 2., np.float32), confidence=np.ones((16, 20)))
    actual, _, receipt, _ = preservation.preserve_generated_view(generated=generated, prepared=prepared,
        references=[source], output_root=tmp_path / 'result', native_result_digest='native', native_frame_index=0)
    assert np.array_equal(actual['confidence'] > 0, mask > 0)
    assert np.array_equal(actual['rgb'], source['rgb'])
    assert receipt['generated_geometry_pixel_count'] == 80
    assert receipt['completion_gap_ids'] == ['floor-gap']


def test_no_gap_returns_existing_scene_without_planner_or_provider_purchase(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_completion, atlas_context_fov

    cam = {'intrinsics': {'width': 32, 'height': 18, 'fx': 30., 'fy': 30., 'cx': 16., 'cy': 9.},
           'extrinsics': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}
    source = posed_frame(frame_id='source', rgb=np.full((18, 32, 3), 50, np.uint8),
        depth=np.full((18, 32), 2., np.float32), confidence=np.ones((18, 32)), camera=cam)
    monkeypatch.setattr(atlas_context_fov, 'verify_pose_context_fov', lambda **_: {'digest': 'fov'})
    def forbidden(**_):
        pytest.fail('A scene without eligible gaps must not purchase generation or a planning review')
    actual, report = atlas_scene_completion.complete_scene(pose_root=tmp_path, pose={}, frames=[source],
        review={}, root=tmp_path / 'completion', context={}, configuration={}, task_runner=forbidden,
        planner_runner=forbidden, release_blockers=forbidden, max_polls=1)
    assert actual == [] and report['status'] == 'passed' and report['generated_roots'] == []


def test_gap_survey_reuses_verified_projection_and_refuses_changed_geometry(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_chisel

    cam = {'intrinsics': {'width': 32, 'height': 18, 'fx': 30., 'fy': 30., 'cx': 16., 'cy': 9.},
           'extrinsics': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}}
    source = posed_frame(frame_id='source', rgb=np.full((18, 32, 3), 50, np.uint8),
        depth=np.full((18, 32), 2., np.float32), confidence=np.ones((18, 32)), camera=cam)
    pool = [{'camera_id': 'source-wide', 'source_frame_id': 'source', 'purpose': 'wide', 'camera': cam}]
    report = gaps.survey_gaps(frames=[source], geometry_frames=[source], candidates=pool, root=tmp_path)
    monkeypatch.setattr(atlas_scene_chisel, 'project_source_depth', lambda *_: pytest.fail('Verified cache not adopted'))
    assert gaps.survey_gaps(frames=[source], geometry_frames=[source], candidates=pool, root=tmp_path) == report
    source['depth'][0, 0] = 0
    with pytest.raises(ValueError, match='gap_survey_changed'):
        gaps.survey_gaps(frames=[source], geometry_frames=[source], candidates=pool, root=tmp_path)


@pytest.mark.parametrize('plan_name', ['chisel-completion-plan.json', 'rgbd-completion-plan.json'])
def test_prior_completion_reuses_bound_composite_and_mask_without_regenerating(tmp_path, monkeypatch, plan_name):
    from blueprint_pipeline import atlas_scene_completion
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    preserved = tmp_path / 'preserved/0000-0000'
    preserved.mkdir(parents=True)
    image, mask = preserved / 'image.png', preserved / 'mask.png'
    Image.fromarray(np.full((8, 8, 3), 77, np.uint8)).save(image)
    pixels = np.zeros((8, 8), np.uint8)
    pixels[2:6, 2:6] = 255
    Image.fromarray(pixels).save(mask)
    receipt = {'frame_id': 'prior-completion-floor', 'native_result_digest': 'native', 'native_frame_index': 0,
               'assets': [{'role': role, 'path': str(path), 'sha256': _sha256_file(path)} for role, path in
                          [('composite_rgb', image), ('generated_geometry_mask', mask)]]}
    receipt['digest'] = canonical_digest(receipt, digest_field='digest')
    (preserved / 'preservation.json').write_text(json.dumps(receipt))
    report = {'status': 'passed', 'generated_roots': [str(tmp_path / 'native')],
              'generation': {'preservation_receipt_digests': [receipt['digest']]}}
    plan = {'source_pose_digest': 'pose', 'batches': [{'frame_ids': ['floor']}]} 
    for name, value in [('completion.json', report), (plan_name, plan)]:
        value['digest'] = canonical_digest(value, digest_field='digest')
        (tmp_path / name).write_text(json.dumps(value))
    native = {'frame_id': 'temporary', 'rgb': np.zeros((8, 8, 3), np.uint8), 'confidence': np.ones((8, 8))}
    monkeypatch.setattr(atlas_scene_completion, 'load_generated_views', lambda *_: ({'digest': 'native'}, [native], []))
    frames, actual = gaps.retained_completion(root=tmp_path, pose_digest='pose')
    assert actual == report and frames[0]['frame_id'] == 'prior-completion-floor'
    assert np.all(frames[0]['rgb'] == 77) and np.array_equal(frames[0]['confidence'] > 0, pixels > 0)
    with pytest.raises(ValueError, match='prior_incompatible'):
        gaps.retained_completion(root=tmp_path, pose_digest='another-coordinate-frame')
    mask.write_bytes(b'changed-mask')
    with pytest.raises(ValueError, match='prior_changed'):
        gaps.retained_completion(root=tmp_path, pose_digest='pose')
