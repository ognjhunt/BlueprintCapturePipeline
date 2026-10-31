"""A gap is space no frame saw; a raster crack is none, and photographed-without-depth space is flagged."""
import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import atlas_observed_space as space
from blueprint_pipeline import atlas_scene_gaps as gaps
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _camera(position=(0., 0., 0.), size=(64, 48), focal=48.):
    width, height = size
    return {'intrinsics': {'width': width, 'height': height, 'fx': focal, 'fy': focal,
                           'cx': width / 2, 'cy': height / 2},
            'extrinsics': {'position': list(position), 'quaternion': [0., 0., 0., 1.], 'coordinateSystem': 'rub'}}


def _wall(depth=4., position=(0., 0., 0.), unranged=None, box=None):
    """A frame facing a wall ``depth`` ahead; ``unranged`` pixels lack confident depth, ``box`` is nearer."""
    values = np.full((48, 64), depth, np.float32)
    confidence = np.ones_like(values)
    if box is not None:
        values[box] = depth / 2
    if unranged is not None:
        confidence[unranged] = 0
    return posed_frame(frame_id=f'wall-{position[0]:+.2f}', rgb=np.full((48, 64, 3), 120, np.uint8),
                       depth=values, confidence=confidence, camera=_camera(position))


def _survey(tmp_path, frames, cameras, **fields):
    pool = [{'camera_id': f'view-{i}', 'source_frame_id': frames[0]['frame_id'], 'purpose': 'translated',
             'camera': camera, **fields} for i, camera in enumerate(cameras)]
    return gaps.survey_gaps(frames=frames, geometry_frames=frames, candidates=pool, root=tmp_path)


def _labels(row):
    return np.asarray(Image.open(row['label_path']))


def test_a_magnified_view_of_a_ranged_wall_has_cracks_but_no_gap(tmp_path):
    # Twice as close with twice the focal length: each photographed pixel spans
    # four here, so a one-pixel projection leaves a lattice of cracks inside the
    # projected scene outline a gap camera surveys.
    frames = [_wall()]
    close = _camera((0., 0., -2.), focal=96.)
    from blueprint_pipeline.atlas_scene_chisel import project_source_depth

    cracked = project_source_depth(frames, close)
    assert 0 < np.mean(cracked > 0) < .5
    assert gaps.gap_labels(cracked, np.zeros(cracked.shape, bool), scene_outline=True)[1], \
        'the cracks used to be named gaps'
    row = _survey(tmp_path, frames, [close], gap_domain='projected_scene_outline_hypothesis')['views'][0]
    assert row['gaps'] == [] and not _labels(row).any()
    assert row['uncovered_pixels_first_reaching']['ranged_surface'] > 0
    assert row['uncovered_pixels_first_reaching']['never_seen'] == 0


def test_space_every_photograph_saw_without_depth_is_its_own_flagged_region(tmp_path):
    patch = np.s_[14:34, 22:42]  # e.g. a chair frame with no confident depth in any photograph
    frames = [_wall(unranged=patch), _wall(position=(.1, 0., 0.), unranged=np.s_[14:34, 23:43])]
    target = _camera((.05, 0., 0.))
    report = _survey(tmp_path, frames, [target])
    row = report['views'][0]
    labels = _labels(row)
    # Still fillable (a recorded surface may only lack depth), but never an
    # unflagged missing surface that a plan would read as bare wall or floor.
    flagged = {g['label'] for g in row['gaps'] if g.get('photographed_without_depth')}
    assert labels[24, 32] in flagged
    assert set(np.unique(labels[18:30, 26:38])) <= flagged
    assert row['photographed_without_depth_gap_pixel_count'] == np.isin(labels, list(flagged)).sum() > 0
    assert row['uncovered_pixels_first_reaching']['photographed_without_depth'] > 0
    preview = np.asarray(Image.open(report['previews'][0]['path']))
    assert (preview[24, 32] == (0, 200, 255)).all()
    assert report['schema_version'] == 'atlas_geometry_gap_survey.v2'
    assert 'cyan' in report['legend'].lower() and report['observed_space']['voxel_size'] > 0


def test_space_no_frame_saw_stays_a_gap(tmp_path):
    # A nearer box hides part of the wall; moving right reveals wall behind its left edge.
    box = np.s_[16:32, 24:40]
    frames = [_wall(box=box)]
    target = _camera((.6, 0., 0.))
    row = _survey(tmp_path, frames, [target])['views'][0]
    assert row['gaps'] and row['uncovered_pixels_first_reaching']['never_seen'] >= row['gap_pixel_count']


def _slab_space(thickness):
    """Free air, a ``thickness`` slab seen only without depth, then a ranged wall at z = -4."""
    size = .05
    state = np.full((8, 8, 100), space.FREE, np.uint8)
    wall = round((4.4 - 4.) / size)  # z runs from -4.4 (index 0) to 0.6
    state[:, :, :wall + 1] = space.RANGED
    slab = round(thickness / size)
    state[:, :, wall + 1:wall + 1 + slab] = space.UNRANGED
    return {'schema_version': space.SCHEMA_VERSION, 'minimum': np.array([-.2, -.2, -4.4]),
            'maximum': np.array([.2, .2, .6]), 'voxel_size': size, 'state': state}


def test_depthless_space_just_at_a_surface_is_that_surface_and_space_in_front_is_not():
    camera = _camera(size=(4, 4), focal=40.)
    mask = np.ones((4, 4), bool)
    # A photograph's depthless ray continues behind what it saw: a thin layer at
    # the wall is the wall itself; a deep layer in front is the unranged object.
    assert (space.first_reached(_slab_space(.1), camera, mask) == space.RANGED).all()
    assert (space.first_reached(_slab_space(1.), camera, mask) == space.UNRANGED).all()
    assert (space.first_reached(_slab_space(1.), camera, np.zeros((4, 4), bool)) == space.NOT_WALKED).all()


def test_export_feedback_keeps_ranged_surfaces_the_export_lost_fillable(tmp_path, monkeypatch):
    frames = [_wall()]
    pool = [{'camera_id': 'view-0', 'source_frame_id': frames[0]['frame_id'], 'purpose': 'wide',
             'camera': _camera()}]
    # The photograph's own view: its ranged wall is no gap...
    assert gaps.survey_gaps(frames=frames, geometry_frames=frames, candidates=pool,
                            root=tmp_path / 'source')['views'][0]['gaps'] == []
    # ...but where the export lost it, that hole stays a fill target.
    monkeypatch.setattr(gaps, 'load_exported_splat', lambda root: ({'digest': 'export'}, object()))
    monkeypatch.setattr(gaps, 'project_exported_splat', lambda splat, camera: np.zeros((48, 64), np.float32))
    report = gaps.survey_gaps(frames=frames, geometry_frames=frames, candidates=pool, root=tmp_path / 'export',
                              splat_root=tmp_path)
    assert report['views'][0]['gaps']
    assert report['views'][0]['uncovered_pixels_first_reaching']['ranged_surface'] > 0


def test_a_survey_saved_before_the_observed_space_is_adopted_unchanged(tmp_path):
    frames = [_wall()]
    pool = [{'camera_id': 'view-0', 'source_frame_id': frames[0]['frame_id'], 'purpose': 'wide',
             'camera': _camera()}]
    report = gaps.survey_gaps(frames=frames, geometry_frames=frames, candidates=pool, root=tmp_path)
    legacy = deepcopy(report)
    for key in ('gap_basis', 'observed_space', 'gap_camera_origin_ranking'):
        legacy.pop(key)
    for row in legacy['views']:
        row.pop('uncovered_pixels_first_reaching')
        row.pop('photographed_without_depth_gap_pixel_count')
    legacy.update(schema_version='atlas_geometry_gap_survey.v1',
                  legend='Magenta identifies candidate missing geometry; black exterior is not a fill target.')
    legacy['digest'] = canonical_digest(legacy, digest_field='digest')
    (tmp_path / 'survey.json').write_text(json.dumps(legacy))
    assert gaps.survey_gaps(frames=frames, geometry_frames=frames, candidates=pool, root=tmp_path) == legacy


def test_voxel_counts_and_cache_follow_the_exact_frames():
    frames = [_wall(unranged=np.s_[20:28, 28:36])]
    carved = space.observed_space(photographs=frames, geometry_frames=frames)
    assert space.observed_space(photographs=frames, geometry_frames=frames) is carved
    counts = space.space_summary(carved)['voxel_counts']
    assert counts['free'] and counts['ranged_surface'] and counts['photographed_without_depth']
    assert sum(counts.values()) == carved['state'].size
    changed = [_wall()]
    assert space.space_summary(space.observed_space(photographs=changed, geometry_frames=changed))[
        'voxel_counts']['photographed_without_depth'] == 0
    with pytest.raises(ValueError):
        space.observed_space(photographs=[], geometry_frames=[])
