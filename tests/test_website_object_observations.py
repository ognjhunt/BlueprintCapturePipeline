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


def test_a_parametric_object_is_authored_from_its_shape_where_it_stood_and_its_frames_stay_observed(tmp_path):
    """The build plan sourced the job's object as a plain carton (owner direction 2026-11-23): compile authors it as a
    supplied shape at no cost, sized from its estimated bounds; its candidate is that shape, its frames still observed."""
    from pxr import Usd, UsdGeom
    from blueprint_pipeline.website_parametric_object import SOURCE
    args = _arguments(tmp_path)
    subject = next(row for row in args['removal_manifest']['entries'] if row.get('task_effect') == 'manipulated')
    args['removal_manifest'] = {**args['removal_manifest'],
                                'parametric_subject': {'target_id': subject['target_id'], 'shape': 'carton'}}
    preparation = compile_website_scene_preparation(**args)
    config = preparation['authoring_inputs']['configuration']
    assert preparation['authoring_inputs']['adapter'] == 'provided_mesh_rigid_authoring'
    assert 'authoring_backend' not in config and config['authoring_cost_usd'] == 0.0
    assert config['authoring_source'] == SOURCE and config['source_observation_kind'] == 'website_capture_frames'
    lower, upper = preparation['subject']['aabb_min_xyz'], preparation['subject']['aabb_max_xyz']
    assert config['parametric_design']['dimensions_m'] == pytest.approx([b - a for a, b in zip(lower, upper)], abs=1e-6)
    kwargs = dict(preparation=preparation, source_geometry=args['source_geometry'], task_masks=args['task_masks'],
                  output_root=tmp_path / 'observations')
    result = materialize_object_observations(**kwargs)
    manifest, frames = validate_observation_handoff(Path(result['manifest']['path']), configuration=result['configuration'])
    assert frames and manifest['complete_object_geometry'] is False and manifest['dimension_authority'] == 'estimated'
    assert manifest['candidate_geometry']['source'] == SOURCE and manifest['candidate_geometry']['authoring_cost_usd'] == 0
    stage = Usd.Stage.Open(str(Path(result['manifest']['path']).parent / manifest['candidate']['path']))
    shapes = [prim for prim in stage.GetPrimAtPath('/Root/SourceObjectCandidate').GetChildren()]
    assert [prim.GetName() for prim in shapes] == ['Parametric']
    xs, ys, zs = zip(*UsdGeom.Mesh(shapes[0]).GetPointsAttr().Get())
    assert [min(xs), min(ys), min(zs)] == pytest.approx(lower, abs=1e-4)
    assert [max(xs), max(ys), max(zs)] == pytest.approx(upper, abs=1e-4)
    assert materialize_object_observations(**kwargs) == result  # reused as it stands
    # The candidate must be the configured shape, and only a parametric configuration takes one.
    other = {**result['configuration'], 'parametric_design': {**config['parametric_design'], 'shape': 'tray'}}
    with pytest.raises(ValueError, match='website_parametric_candidate_binding_invalid'):
        validate_observation_handoff(Path(result['manifest']['path']), configuration=other)
    plain = {key: value for key, value in result['configuration'].items() if key != 'parametric_design'}
    with pytest.raises(ValueError, match='website_parametric_candidate_binding_invalid'):
        validate_observation_handoff(Path(result['manifest']['path']), configuration=plain)

