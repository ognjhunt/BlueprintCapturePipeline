"""Translated generation hypotheses must stay inside photographed free space."""
from copy import deepcopy

import numpy as np

from blueprint_pipeline.atlas_completion_cameras import observed_free_position, disocclusion_candidates
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.atlas_scene_preservation import observed_scene_bounds


def source(x):
    return posed_frame(frame_id=str(x), rgb=np.zeros((80, 100, 3), np.uint8),
        depth=np.full((80, 100), 4., np.float32), confidence=np.ones((80, 100)), camera={
        'intrinsics': {'width': 100, 'height': 80, 'fx': 70., 'fy': 70., 'cx': 50., 'cy': 40.},
        'extrinsics': {'position': [x, 0, 0], 'quaternion': [0, 0, 0, 1], 'coordinateSystem': 'rub'}})


def test_generated_camera_candidates_are_free_bounded_and_scene_scale_relative():
    frames = [source(-.2), source(.2)]
    wide = [{'camera_id': f['frame_id'], 'source_frame_id': f['frame_id'], 'camera': {
        'intrinsics': deepcopy(f['intrinsics']), 'extrinsics': deepcopy(f['extrinsics'])}} for f in frames]
    bounds = observed_scene_bounds(frames)
    assert observed_free_position([0, 0, -1], frames, bounds)
    assert not observed_free_position([0, 0, -4], frames, bounds)  # Existing surface.
    assert not observed_free_position([0, 0, -5], frames, bounds)  # Outside scene.
    assert not observed_free_position([10, 0, -1], frames, bounds)  # Not photographed.
    rows = disocclusion_candidates(frames, wide)
    assert rows
    for row in rows:
        cam = row['camera']
        assert observed_free_position(cam['extrinsics']['position'], frames, bounds)
        assert row['collision_qualified'] is False
        assert np.isclose(np.linalg.norm(cam['extrinsics']['quaternion']), 1)
    enlarged, scaled_wide = deepcopy(frames), deepcopy(wide)
    for f, w in zip(enlarged, scaled_wide, strict=True):
        f['depth'] *= 10
        f['position'] *= 10
        f['extrinsics']['position'] = f['position'].tolist()
        w['camera']['extrinsics']['position'] = f['position'].tolist()
    scaled = disocclusion_candidates(enlarged, scaled_wide)
    assert len(scaled) == len(rows)
    for a, b in zip(rows, scaled, strict=True):
        assert np.allclose(np.array(a['camera']['extrinsics']['position']) * 10, b['camera']['extrinsics']['position'])
        assert np.allclose(a['camera']['extrinsics']['quaternion'], b['camera']['extrinsics']['quaternion'])
