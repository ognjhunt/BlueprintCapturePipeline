"""Reuse projection work without changing native geometry or camera coverage."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from blueprint_pipeline import atlas_scene_chisel as chisel
from blueprint_pipeline import atlas_scene_preservation as preservation
from blueprint_pipeline.atlas_depth_splat import posed_frame


def _camera(position=(0., 0., 0.), size=(64, 40)):
    width, height = size
    return {'intrinsics': {'width': width, 'height': height, 'fx': width * .8,
                           'fy': width * .8, 'cx': width / 2, 'cy': height / 2},
            'extrinsics': {'position': list(position), 'quaternion': [0., 0., 0., 1.],
                           'coordinateSystem': 'rub'}}


def _frames():
    rows = []
    for i in range(3):
        depth = np.full((40, 64), 2. + .2 * i, np.float32)
        depth[8:16, 20:30] = np.nan
        depth[25:30, 40:50] = 0
        confidence = np.ones_like(depth)
        confidence[:4] = 0
        rows.append(posed_frame(frame_id=str(i), rgb=np.full((40, 64, 3), 80, np.uint8),
            depth=depth, confidence=confidence, camera=_camera((i * .15, 0., 0.))))
    return rows


def test_batch_is_pixel_identical_to_single_camera_projection_for_mixed_cameras():
    frames = _frames()
    cameras = [_camera(), _camera((.15, 0., 0.)), _camera((.04, .02, .03)),
               _camera((.3, 0., 0.), (48, 30))]
    rotated = deepcopy(cameras[0])
    rotated['extrinsics']['quaternion'] = [0., np.sin(.1), 0., np.cos(.1)]
    cameras.append(rotated)
    expected = [chisel.project_source_depth(frames, camera) for camera in cameras]
    actual = chisel.project_source_depths(frames, cameras)
    # Frozen scalar outputs from the deployed f7b566662 projector, independent
    # of the single-camera wrapper now sharing the batched implementation.
    legacy_digests = [
        'a9760f007b68757bd093e29aa6526694a9a19cd051bdb6994ce59a3329584bff',
        'a045f105f99f16e1decaa52a7349547a510db6f7928af717c98c01a8fddba00b',
        '3d067772c4e42389d1b34c2845a35708d81dc32c5087c9fc8b1af7b397a32424',
        'a3271af05ee144f16838b7b37edd68105a843ca5f3c42841f8b6034dfb81260a',
        '148456877169a1a186587d88366f48659d652dff6a40a862115574c9c1f007d6',
    ]
    assert [hashlib.sha256(depth.astype('<f4').tobytes()).hexdigest()
            for depth in actual] == legacy_digests
    for one, many in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(one, many)
        assert many.dtype == np.float32 and np.isfinite(many).all()


def test_world_lifting_is_once_per_source_per_batch_and_never_cached_across_inputs(monkeypatch):
    frames = _frames()
    cameras = [_camera((.05 + .01 * i, 0., 0.)) for i in range(8)]
    calls = []
    world = preservation._world
    monkeypatch.setattr(preservation, '_world', lambda frame, valid: calls.append(frame['frame_id']) or world(frame, valid))
    chisel.project_source_depths(frames, cameras)
    assert calls == ['0', '1', '2']
    calls.clear()
    frames[0]['confidence'][:] = 0
    actual = chisel.project_source_depths(frames, cameras)
    assert calls == ['0', '1', '2']
    expected = [chisel.project_source_depth(frames, camera) for camera in cameras]
    for one, many in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(one, many)


def test_batch_bound_limits_working_memory_without_limiting_total_camera_coverage(tmp_path):
    frames = _frames()
    cameras = [_camera((.04 + .01 * i, 0., 0.)) for i in range(19)]
    with pytest.raises(ValueError, match='projection_batch_too_large'):
        chisel.project_source_depths(frames, cameras)
    selected = [{'camera_id': str(i), 'camera': camera} for i, camera in enumerate(cameras)]
    result = chisel.prepare_depth_context(frames=frames, selected=selected, root=tmp_path)
    assert list(result) == [str(i) for i in range(19)]
    assert chisel.project_source_depths(frames, []) == []


def test_unknown_geometry_remains_zero_and_does_not_leak_previous_batch():
    frames = _frames()
    for frame in frames:
        frame['confidence'][:] = 0
    cameras = [_camera(), _camera((.1, 0., .2))]
    assert all(not depth.any() for depth in chisel.project_source_depths(frames, cameras))


def test_gap_survey_batches_every_candidate_and_adopts_its_verified_report(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_gaps

    frames = _frames()
    candidates = [{'camera_id': str(i), 'source_frame_id': '0', 'purpose': 'wide',
                   'camera': _camera((i * .01, 0., .01))} for i in range(19)]
    calls = []
    project = chisel.project_source_depths

    def batch(sources, cameras):
        calls.append(len(cameras))
        return project(sources, cameras)

    monkeypatch.setattr(chisel, 'project_source_depths', batch)
    args = dict(frames=frames, geometry_frames=frames, candidates=candidates, root=tmp_path)
    report = atlas_scene_gaps.survey_gaps(**args)
    assert calls == [8, 8, 3]
    assert [row['camera_id'] for row in report['views']] == [str(i) for i in range(19)]
    monkeypatch.setattr(chisel, 'project_source_depths', lambda *_: pytest.fail('repeated a completed survey'))
    assert atlas_scene_gaps.survey_gaps(**args) == report
