"""The declared output grid must not stand in for observed input coverage."""
import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_context_fov as fov


def texture(seed=14):
    import cv2

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (512, 288, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .5)
    cv2.putText(image, 'SINK LEFT', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    cv2.rectangle(image, (4, 300), (80, 480), (20, 20, 20), 5)
    return image


def test_portrait_grid_does_not_prove_portrait_field_of_view_was_kept():
    import cv2

    source = texture()
    # The measured retained run: central 31.6%, enlarged back to the source dimensions.
    returned = cv2.resize(source[112:400, 63:225], (288, 512))
    report = fov.image_coverage(source, returned)
    assert report['source_shape'] == report['returned_shape']
    assert report['retained_source_area_fraction'] == pytest.approx(.3164, abs=.005)
    assert report['full_source_field_of_view_preserved'] is False


def test_padding_preserves_source_content_through_native_square_and_output_resize():
    import cv2

    source = texture()
    padded, transform = fov.square_pad(source)
    x, y, width, height = transform['source_rect']
    assert np.array_equal(padded[y:y + height, x:x + width], source)
    # Replay the provider's square reconstruction then square target raster.
    native = cv2.resize(padded, (384, 384), interpolation=cv2.INTER_AREA)
    square = cv2.resize(native, (512, 512))
    # The requested portrait output now removes only the square padding.
    returned = square[:, x:x + width]
    report = fov.image_coverage(source, returned)
    assert report['retained_source_area_fraction'] > .99
    assert report['full_source_field_of_view_preserved'] is True


@pytest.mark.parametrize('landscape', [False, True])
def test_two_stage_resize_keeps_source_pixels_and_their_camera_rays(landscape):
    import cv2

    source = texture()
    if landscape:
        source = np.rot90(source).copy()
    padded, transform = fov.square_pad(source)
    x, y, width, height = transform['source_rect']
    side = padded.shape[0]
    native_size = 384
    native = cv2.resize(padded, (native_size, native_size), interpolation=cv2.INTER_AREA)
    scale = side / native_size
    native_k = np.array([[260., 0, native_size / 2], [0, 260., native_size / 2], [0, 0, 1]])
    native_to_target = np.array([[scale, 0, -x], [0, scale, -y], [0, 0, 1]])
    returned_k = native_to_target @ native_k
    square = cv2.resize(native, (side, side))
    returned = square[y:y + height, x:x + width]
    assert fov.image_coverage(source, returned)['full_source_field_of_view_preserved']
    # Pixel-centre coordinates use the same uniform resize and crop for RGB,
    # depth and confidence. A scene ray is invariant under that transformation.
    for u, v in ((.5, .5), (width - .5, height - .5), (width / 2, height / 2)):
        pixel = np.array([u, v, 1.])
        native_pixel = np.linalg.inv(native_to_target) @ pixel
        assert np.allclose(np.linalg.inv(returned_k) @ pixel, np.linalg.inv(native_k) @ native_pixel)


def test_padding_provenance_refuses_changed_source_or_changed_padding(tmp_path):
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    source = tmp_path / 'source.png'
    padded = tmp_path / 'padded.png'
    Image.fromarray(texture()).save(source)
    array, transform = fov.square_pad(texture())
    Image.fromarray(array).save(padded)
    inputs = {'frames': [{'frame_id': 'a', 'image_path': str(padded)}],
              'full_view_sources': [{'frame_id': 'a', 'image_path': str(source),
                                    'image_digest': _sha256_file(source), 'padding': transform}]}
    assert fov.full_view_sources(inputs, tmp_path) == inputs['full_view_sources']
    array[0, 0] = 0
    Image.fromarray(array).save(padded)
    with pytest.raises(ValueError, match='padding_changed'):
        fov.full_view_sources(inputs, tmp_path)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source_changed'):
        fov.full_view_sources(inputs, tmp_path)


def test_other_frame_or_mirrored_pixels_are_not_admitted_as_the_source():
    for changed in (texture(25), texture()[:, ::-1].copy()):
        with pytest.raises(ValueError, match='atlas_context_'):
            fov.image_coverage(texture(), changed)


def test_identity_retains_all_source_pixels_even_without_texture():
    source = np.full((128, 72, 3), 150, np.uint8)
    assert fov.image_coverage(source, source.copy())['retained_source_area_fraction'] == 1


def test_post_pose_refusal_is_retained_before_any_generation(tmp_path, monkeypatch):
    import cv2
    import json
    from blueprint_pipeline import website_atlas_pose

    source = texture()
    source_path = tmp_path / 'original.png'
    Image.fromarray(source).save(source_path)
    monkeypatch.setattr(website_atlas_pose, 'load_pending_atlas_pose_inputs', lambda _: {
        'digest': 'source-inputs', 'frames': [{'frame_id': '0043',
            'image_path': str(source_path), 'image_digest': 'original-digest'}]})
    cropped = cv2.resize(source[112:400, 63:225], (288, 512))
    with pytest.raises(ValueError, match='atlas_context_field_of_view_lost'):
        fov.verify_pose_context_fov(pose_root=tmp_path, frames=[{'frame_id': '0043', 'rgb': cropped}],
                                    output_root=tmp_path)
    report = json.loads((tmp_path / 'context-fov-audit.json').read_text())
    assert report['status'] == 'refused' and report['provider_calls'] == 0
    assert report['frames'][0]['retained_source_area_fraction'] < .32


def test_completion_rejects_cropped_context_before_planner_or_upload(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pose, atlas_scene_completion as completion
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    import cv2

    source = texture()
    image = tmp_path / 'approved.png'
    Image.fromarray(source).save(image)
    cropped = cv2.resize(source[112:400, 63:225], (288, 512))
    monkeypatch.setattr(website_atlas_pose, 'load_pending_atlas_pose_inputs', lambda _: {
        'digest': 'inputs', 'frames': [{'frame_id': 'a', 'image_path': str(image),
                                      'image_digest': _sha256_file(image)}]})
    def forbidden(**_):
        pytest.fail('Cropped input reached a model or upload')
    with pytest.raises(ValueError, match='field_of_view_lost'):
        completion.complete_scene(pose_root=tmp_path, pose={}, frames=[{'frame_id': 'a', 'rgb': cropped}],
            review={}, root=tmp_path / 'complete', context={}, configuration={},
            task_runner=forbidden, planner_runner=forbidden, release_blockers=forbidden, max_polls=1)
    assert not (tmp_path / 'complete/context-uploads').exists()


def test_a_verified_padding_is_decoded_once_per_job_and_a_change_is_still_refused(tmp_path, monkeypatch):
    # capture-447ae4f4 (2026-10-30): every tick decoded each retained source and
    # padded input three times to re-prove the same bytes.
    import time
    from pathlib import Path
    from blueprint_pipeline import file_digest_cache
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    monkeypatch.setattr(file_digest_cache, 'RACY_NANOSECONDS', -60 * 10**9)  # long-settled files
    source, padded = tmp_path / 'source.png', tmp_path / 'padded.png'
    Image.fromarray(texture()).save(source)
    array, transform = fov.square_pad(texture())
    Image.fromarray(array).save(padded)
    inputs = {'frames': [{'frame_id': 'a', 'image_path': str(padded)}],
              'full_view_sources': [{'frame_id': 'a', 'image_path': str(source),
                                    'image_digest': _sha256_file(source), 'padding': transform}]}
    decoded, opened = [], Image.open
    monkeypatch.setattr(Image, 'open', lambda path, *a, **k: (decoded.append(Path(path).name), opened(path, *a, **k))[1])
    sidecar = tmp_path / file_digest_cache.SIDECAR
    for _tick in range(3):
        with file_digest_cache.file_digest_cache(sidecar):
            assert fov.full_view_sources(inputs, tmp_path) == inputs['full_view_sources']
            assert fov.full_view_sources(inputs, tmp_path) == inputs['full_view_sources']
    assert decoded == ['source.png', 'padded.png']
    time.sleep(.05)
    array[0, 0] = 0
    Image.fromarray(array).save(padded)
    with file_digest_cache.file_digest_cache(sidecar), pytest.raises(ValueError, match='padding_changed'):
        fov.full_view_sources(inputs, tmp_path)
    changed = {**inputs['full_view_sources'][0], 'padding': {**transform, 'source_rect': [0, 0, 1, 1]}}
    with file_digest_cache.file_digest_cache(sidecar), pytest.raises(ValueError, match='padding_changed'):
        fov.full_view_sources({**inputs, 'full_view_sources': [changed]}, tmp_path)
