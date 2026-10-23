"""The target ray hits the missing region; references and source pixels stay intact."""
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from blueprint_pipeline.atlas_completion_cameras import gap_target_candidates
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def scene(tmp_path):
    cam = {'intrinsics': {'width': 100, 'height': 80, 'fx': 80., 'fy': 80., 'cx': 50., 'cy': 40.},
           'extrinsics': {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.], 'coordinateSystem': 'rub'}}
    depth = np.full((80, 100), 2., np.float32)
    depth[20:40, 65:85] = 0
    first = posed_frame(frame_id='first', rgb=np.full((80, 100, 3), 90, np.uint8), depth=depth,
                        confidence=(depth > 0).astype(float), camera=cam)
    second_camera = deepcopy(cam)
    second_camera['extrinsics']['position'] = [-.25, 0., 0.]
    second = posed_frame(frame_id='second', rgb=first['rgb'].copy(), depth=depth.copy(),
                         confidence=first['confidence'].copy(), camera=second_camera)
    labels = np.zeros((80, 100), np.uint16)
    labels[20:40, 65:85] = 1
    path = tmp_path / 'gaps.png'
    Image.fromarray(labels).save(path)
    candidates = [{'camera_id': 'source', 'source_frame_id': 'first', 'purpose': 'wide', 'camera': cam}]
    survey = {'views': [{'camera_id': 'source', 'label_path': str(path), 'label_sha256': _sha256_file(path),
                         'gaps': [{'gap_id': 'source-gap-1', 'label': 1}]}]}
    return [first, second], candidates, survey


def test_camera_points_at_actual_gap_with_parallax_and_keeps_all_context(tmp_path):
    frames, pool, survey = scene(tmp_path)
    snapshots = [{k: frame[k].copy() for k in ['rgb', 'depth', 'confidence', 'position', 'rotation']} for frame in frames]
    before = deepcopy(pool)
    result = gap_target_candidates(frames, pool, survey)
    assert len(result) == 1
    target = result[0]
    assert target['source_gap_ids'] == ['source-gap-1']
    assert np.asarray(target['gap_target_polygon_pixels']).shape == (4, 2)
    # Its estimated centre lies on the known 2-unit plane, on the right of the image.
    focus = np.array(target['target_world_estimate'])
    assert focus[0] == pytest.approx(.625) and focus[1] == pytest.approx(.25)
    assert focus[2] == pytest.approx(-2.)
    e = target['camera']['extrinsics']
    assert e['position'] == [-.25, 0., 0.]
    forward = -Rotation.from_quat(e['quaternion']).as_matrix()[:, 2]
    direction = focus - e['position']
    assert np.allclose(forward, direction / np.linalg.norm(direction))
    assert pool == before
    for frame, snapshot in zip(frames, snapshots, strict=True):
        for key, value in snapshot.items():
            assert np.array_equal(frame[key], value)
    assert gap_target_candidates(frames, pool, survey) == result


def test_blank_exterior_and_absent_neighbour_depth_do_not_invent_a_target(tmp_path):
    frames, pool, survey = scene(tmp_path)
    assert gap_target_candidates(frames, pool, {'views': []}) == []
    for frame in frames:
        frame['confidence'][:] = 0
    assert gap_target_candidates(frames, pool, survey) == []


def test_changed_gap_mask_refuses_camera_planning(tmp_path):
    frames, pool, survey = scene(tmp_path)
    Image.fromarray(np.zeros((80, 100), np.uint16)).save(survey['views'][0]['label_path'])
    with pytest.raises(ValueError, match='gap_survey_changed'):
        gap_target_candidates(frames, pool, survey)
