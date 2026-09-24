"""ADP-030: asset appearance uses original capture detail, geometry keeps its grid."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_object_observations import materialize_object_observations, validate_observation_handoff
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_website_task_preparation import _arguments, _assembly_inputs


def _with_original(args, tmp_path, *, rotation, size):
    geometry, masks = args['source_geometry'], args['task_masks']
    frame = geometry['frames'][0]
    original = tmp_path / 'original.png'
    image = Image.new('RGB', size, '#2244cc')
    image.putpixel((0, 0), (255, 0, 0))  # Stored top-left corner.
    image.save(original)
    frame.update(source_image_path=str(original), source_image_digest=_sha256_file(original),
                 display_rotation_degrees=rotation)
    geometry['digest'] = canonical_digest(geometry, digest_field='digest')
    masks['source_geometry_digest'] = geometry['digest']
    masks['digest'] = canonical_digest(masks, digest_field='digest')
    return original


def test_authoring_retains_original_pixels_and_rechecks_source_on_reuse(tmp_path):
    args = _arguments(tmp_path)
    original = _with_original(args, tmp_path, rotation=0, size=(1080, 1920))
    preparation = compile_website_scene_preparation(**args)
    kwargs = dict(preparation=preparation, source_geometry=args['source_geometry'], task_masks=args['task_masks'],
                  output_root=tmp_path / 'observations')
    result = materialize_object_observations(**kwargs)
    manifest, frames = validate_observation_handoff(Path(result['manifest']['path']), configuration=result['configuration'])
    assert frames[0].read_bytes() == original.read_bytes()
    assert manifest['frames'][0]['image_basis'] == 'original_capture'
    assert manifest['frames'][0]['triangle_count'] > 0
    assert materialize_object_observations(**kwargs) == result
    original.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source_changed'):
        materialize_object_observations(**kwargs)


def test_sideways_original_reaches_the_builder_upright(tmp_path):
    # A portrait phone clip decoded with -noautorotate is stored landscape; the
    # geometry and removal inputs rotate it by the display rotation, and so
    # must the builder's copy.
    args = _arguments(tmp_path)
    original = _with_original(args, tmp_path, rotation=-90, size=(1920, 1080))
    preparation = compile_website_scene_preparation(**args)
    result = materialize_object_observations(preparation=preparation, source_geometry=args['source_geometry'],
        task_masks=args['task_masks'], output_root=tmp_path / 'observations')
    manifest, frames = validate_observation_handoff(Path(result['manifest']['path']), configuration=result['configuration'])
    with Image.open(frames[0]) as sent, Image.open(original) as stored:
        assert sent.size == (1080, 1920)
        # Clockwise quarter turn: the stored top-left corner lands top-right.
        assert sent.getpixel((1079, 0)) == (255, 0, 0)
        assert np.array_equal(np.asarray(sent), np.asarray(stored.convert('RGB').rotate(-90, expand=True)))
    assert manifest['frames'][0]['display_rotation_applied_degrees'] == -90.0
    assert manifest['frames'][0]['source_image_digest'] == _sha256_file(original)


def test_original_without_a_display_rotation_is_refused(tmp_path):
    args = _arguments(tmp_path)
    _with_original(args, tmp_path, rotation=45, size=(1920, 1080))
    preparation = compile_website_scene_preparation(**args)
    with pytest.raises(ValueError, match='website_source_rotation_not_supported'):
        materialize_object_observations(preparation=preparation, source_geometry=args['source_geometry'],
            task_masks=args['task_masks'], output_root=tmp_path / 'observations')


def test_articulated_builder_receives_the_coverage_views(tmp_path):
    args = _arguments(tmp_path)
    args.update(_assembly_inputs(tmp_path))
    preparation = compile_website_scene_preparation(**args)
    assert preparation['status'] == 'intake_ready', preparation['blockers']
    kwargs = dict(preparation=preparation, source_geometry=args['source_geometry'], task_masks=args['task_masks'],
                  output_root=tmp_path / 'observations')
    result = materialize_object_observations(**kwargs)
    manifest, frames = validate_observation_handoff(Path(result['manifest']['path']), configuration=result['configuration'])
    selected = args['task_masks']['targets'][0]['authoring_coverage']['selected_frames']
    references = preparation['authoring_inputs']['configuration']['reference_frames']
    assert [row['frame_id'] for row in manifest['frames']] == [row['frame_id'] for row in selected]
    # The builder receives the provider-sized derivatives, each bound to its retained original.
    assert [_sha256_file(path) for path in frames] == [row['sha256'] for row in references]
    assert [row['retained_original_sha256'] for row in manifest['frames']] == [row['sha256'] for row in selected]
    assert {row['image_basis'] for row in manifest['frames']} == {'upright_coverage_frame'}
    assert all(row['reason'] and row['visible_parts'] for row in manifest['frames'])
    assert materialize_object_observations(**kwargs) == result
    Path(selected[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='source_changed'):
        materialize_object_observations(**kwargs)
