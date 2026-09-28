from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import sealed_camera_render as renderer
from blueprint_pipeline.public_scene_inpainting_inputs import _camera_rows
from tests.test_public_scene_inpainting_inputs import _request
from tests.test_sealed_camera_render import DIGEST, _write_standard_3dgs_ply


def test_generated_camera_rows_seal_clipping_without_splat_bounds():
    rows = _camera_rows(_request(), np.array([1., 2., .28]))
    assert len(rows) == 6
    for row in rows:
        intrinsics = row['intrinsics']
        assert (intrinsics['near'], intrinsics['far']) == (.01, 100000.)
        assert intrinsics['width'] == intrinsics['height'] == 1024


@pytest.mark.parametrize('near,far', [(0, 100), (-1, 100), (1, 1), (2, 1), (float('nan'), 100), (.01, float('inf')), (True, 100)])
def test_clipping_rejects_nonphysical_or_nonfinite_planes(near, far):
    with pytest.raises(ValueError, match='calibrated_camera_clipping_invalid'):
        renderer.calibrated_clip_planes({'near':near, 'far':far})


def test_explicit_valid_planes_are_preserved():
    assert renderer.calibrated_clip_planes({'near':.05,'far':100}) == {'near':.05,'far':100.}


@pytest.mark.slow
def test_outlier_cannot_clip_close_targets_in_real_renderer(tmp_path: Path):
    splat = tmp_path/'scene.ply'
    bright = 1.77
    _write_standard_3dgs_ply(splat, [
        (0., 0., .6, bright, -1., -1.),
        (.12, 0., .6, -1., bright, -1.),
        (0., .09, .6, -1., -1., bright),
        (10000., 10000., 5., bright, bright, bright),
    ])
    before = splat.read_bytes()
    cameras = [{'camera_id':'close_target', 'T_world_camera_provider_frame':np.eye(4).tolist(),
                'intrinsics':{'fx':100.,'fy':100.,'cx':32.,'cy':24.,'width':64,'height':48}}]
    calibration = tmp_path/'cameras.json'
    calibration.write_text(json.dumps(cameras))
    manifest = renderer.render_splat_at_exact_cameras(
        splat_path=splat, cameras=cameras, output_dir=tmp_path/'render',
        provider_splat_import_receipt_digest=DIGEST, alignment_digest=DIGEST,
        camera_set_label='outlier-close-target-regression', calibrated_camera_file=calibration,
        purpose='renderer_projection_conformance', authorization_class='evaluation_authorized',
        background_rgb=0x102030,
    )
    assert splat.read_bytes() == before
    assert manifest['source_splat']['retained_gaussian_count'] == 4
    intrinsics = manifest['calibrated_cameras'][0]['spec']['intrinsics']
    assert (intrinsics['near'], intrinsics['far']) == (.01, 100000.)
    assert {key:intrinsics[key] for key in cameras[0]['intrinsics']} == cameras[0]['intrinsics']
    pixels = np.asarray(Image.open(tmp_path/'render'/manifest['renders'][0]['relative_path']).convert('RGB'))
    for x, y, channel in [(32,24,0),(52,24,1),(32,39,2)]:
        patch = pixels[y-3:y+4,x-3:x+4].reshape(-1,3).max(axis=0)
        assert int(np.argmax(patch)) == channel
        assert int(patch[channel]) > 100
    progress = [json.loads(row) for row in (tmp_path/'render/frames/renderer_progress.jsonl').read_text().splitlines()]
    events = [row['event'] for row in progress]
    assert events.index('settle_frame_started') < events.index('settle_frame_submitted')
    assert events.index('settle_frame_submitted') < events.index('image_readback_started')
    assert events.index('image_readback_started') < events.index('image_readback_completed')
