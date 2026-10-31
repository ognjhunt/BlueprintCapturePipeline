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
    assert target['camera']['intrinsics']['fx'] == pool[0]['camera']['intrinsics']['fx']
    assert target['camera']['intrinsics']['fy'] == pool[0]['camera']['intrinsics']['fy']
    assert target['target_framing'] == 'source_fov_reaimed_at_gap'
    assert np.ptp(np.asarray(target['gap_target_polygon_pixels'])[:, 1]) <= 40.00001
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


def test_context_fallback_reaims_from_source_without_zooming_away_surroundings(tmp_path):
    frames, pool, survey = scene(tmp_path)
    target = gap_target_candidates(frames, pool, survey, source_centred=True)[0]
    assert target['camera']['extrinsics']['position'] == [0., 0., 0.]
    assert target['camera']['intrinsics']['fx'] == pool[0]['camera']['intrinsics']['fx']
    assert target['target_framing'] == 'source_fov_reaimed_at_gap'
    forward = -Rotation.from_quat(target['camera']['extrinsics']['quaternion']).as_matrix()[:, 2]
    direction = np.array(target['target_world_estimate']) - frames[0]['position']
    assert np.allclose(forward, direction / np.linalg.norm(direction))


def test_changed_gap_mask_refuses_camera_planning(tmp_path):
    frames, pool, survey = scene(tmp_path)
    Image.fromarray(np.zeros((80, 100), np.uint16)).save(survey['views'][0]['label_path'])
    with pytest.raises(ValueError, match='gap_survey_changed'):
        gap_target_candidates(frames, pool, survey)


def _with_hover_hypothesis(pool):
    """A screened free-space hypothesis 0.4 units from the plane: the largest view-angle change."""
    hover = deepcopy(pool[0])
    hover.update(camera_id='hover', purpose='disocclusion', camera_location_observed_free=True)
    hover['camera']['extrinsics']['position'] = [.9, .3, -1.6]
    return [*pool, hover]


def test_ranked_survey_places_the_camera_where_the_capture_observes_the_view(tmp_path):
    from blueprint_pipeline.atlas_scene_gaps import GAP_CAMERA_ORIGIN_RANKING

    frames, pool, survey = scene(tmp_path)
    pool = _with_hover_hypothesis(pool)
    # A survey saved without the ranking keeps the largest-angle choice: the
    # close-up over the plane (capture-447ae4f4's camera over the dining table).
    legacy = gap_target_candidates(frames, pool, survey)[0]
    assert legacy['camera']['extrinsics']['position'] == [.9, .3, -1.6]
    assert 'origin_ranking' not in legacy
    ranked = gap_target_candidates(frames, pool, {**survey, 'gap_camera_origin_ranking': GAP_CAMERA_ORIGIN_RANKING})[0]
    assert ranked['camera']['extrinsics']['position'] in ([0., 0., 0.], [-.25, 0., 0.])
    assert ranked['origin_ranking'] == GAP_CAMERA_ORIGIN_RANKING
    assert 0 < ranked['observed_context_fraction_estimate'] <= 1
    assert ranked['exposed_gap_pixel_estimate'] >= 16  # the gap's minimum dedicated-camera area here
    forward = -Rotation.from_quat(ranked['camera']['extrinsics']['quaternion']).as_matrix()[:, 2]
    direction = np.array(ranked['target_world_estimate']) - ranked['camera']['extrinsics']['position']
    assert np.allclose(forward, direction / np.linalg.norm(direction))
    # The source-centred fallback is unchanged by the ranking.
    fallback = gap_target_candidates(frames, pool, {**survey, 'gap_camera_origin_ranking': GAP_CAMERA_ORIGIN_RANKING},
                                     source_centred=True)[0]
    assert fallback == gap_target_candidates(frames, pool, survey, source_centred=True)[0]


def test_close_up_context_scores_low_and_a_gap_behind_a_known_surface_is_not_exposed():
    from blueprint_pipeline.atlas_completion_cameras import CONTEXT_SAMPLE_STRIDE, observed_context

    def camera(z):
        return {'intrinsics': {'width': 64, 'height': 48, 'fx': 48., 'fy': 48., 'cx': 32., 'cy': 24.},
                'extrinsics': {'position': [0., 0., z], 'quaternion': [0., 0., 0., 1.]}}

    # One sample per stride of a photograph one unit from the plane z = -1 (f = 48).
    spacing = CONTEXT_SAMPLE_STRIDE / 48
    x, y = np.meshgrid(np.arange(-1.5, 1.5 + 1e-9, spacing), np.arange(-1.2, 1.2 + 1e-9, spacing))
    plane = np.column_stack((x.ravel(), y.ravel(), -np.ones(x.size)))
    supported = np.ones(len(plane), bool)
    behind, front, weight = np.array([[0., 0., -2.]]), np.array([[0., 0., -.5]]), np.ones(1)
    far, hidden = observed_context(plane, supported, [], camera(1.), behind, weight)
    assert far == 1. and hidden == 0.
    _, seen = observed_context(plane, supported, [], camera(1.), front, weight)
    assert seen == pytest.approx(1 / 1.5 ** 2)
    # Twice as close as the photograph: one sample per four target pixels cannot be kept.
    assert observed_context(plane, supported, [], camera(-.5), behind, weight)[0] == 0.
    # Context no other view supports is not context the keep mask holds.
    assert observed_context(plane, ~supported, [], camera(1.), behind, weight)[0] == 0.


def test_no_camera_is_aimed_at_space_the_photographs_saw_without_depth(tmp_path):
    frames, pool, survey = scene(tmp_path)
    survey['views'][0]['gaps'][0]['photographed_without_depth'] = True
    assert gap_target_candidates(frames, pool, survey) == []
