"""No provider calls: the preserved-scene completion reads one view's pixels at a time, with the same bytes.

The completion read every posed view several times, often inside per-chunk or
per-candidate loops, and a retained ``PosedFrame`` keeps what is read: about
10 MB per 1280x720 view, 1 GB for the 105 views the website sends since
PR #3467. ``tests.atlas_completion_legacy`` holds that code, frozen; it reads
plain dict frames here and the restructured code must agree byte for byte.
"""
import gc
import hashlib
import json
import shutil
import tracemalloc

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth_splat
from blueprint_pipeline import atlas_gap_composite as composite
from blueprint_pipeline import atlas_observed_space as observed
from blueprint_pipeline import atlas_scene_preservation as preservation
from tests import atlas_completion_legacy as legacy
from tests.atlas_posed_rgbd_synthetic import render_room, write_room_pose_root
from tests.atlas_preserved_completion_synthetic import HOLES


def keeps_no_pixels(frames):
    return all(frame._values.keys().isdisjoint(depth_splat.PIXEL_KEYS) for frame in frames)


@pytest.fixture
def retained(tmp_path, monkeypatch):
    """Six retained views of the room with unranged wall patches, read lazily and as eager dicts."""
    monkeypatch.setattr(preservation, '_SUPPORT_MASKS', {})
    monkeypatch.setattr(observed, '_SPACE', {})
    root = write_room_pose_root(tmp_path / 'pose', 6, width=160, height=96, holes=HOLES)
    _, frames = depth_splat.load_posed_rgbd(root)
    _, again = depth_splat.load_posed_rgbd(root)
    return frames, [legacy.eager(frame) for frame in again]


@pytest.mark.parametrize('budget', [0, 2 * 8 * 160 * 96, None])
def test_support_masks_compared_block_by_block_are_the_masks_of_all_views_at_once(retained, monkeypatch, budget):
    frames, eager = retained
    if budget is not None:
        monkeypatch.setattr(preservation, 'SUPPORT_REFERENCE_BYTES', budget)
    masks = [preservation._unpacked(mask) for mask in preservation._support_masks(frames)]
    reference = legacy.support_masks(eager)
    assert any(mask.any() for mask in reference)
    assert all(np.array_equal(mask, expected) for mask, expected in zip(masks, reference, strict=True))
    supported = preservation.supported_projection_frames(frames)
    for lazy, expected in zip(supported, legacy.supported_projection_frames(eager), strict=True):
        assert list(lazy) == list(expected)
        assert all(np.array_equal(lazy[key], expected[key]) for key in depth_splat.PIXEL_KEYS)
    assert keeps_no_pixels(frames)


def test_observed_space_carves_view_by_view_into_the_same_voxels(retained, tmp_path):
    frames, eager = retained
    _, separate = depth_splat.load_posed_rgbd(tmp_path / 'pose')
    # The preserved completion passes its photographs among the geometry frames; others pass their own.
    for photographs, geometry, expected in [(frames[:3], frames, (eager[:3], eager)),
                                            (separate[1:3], frames[2:], (eager[1:3], eager[2:]))]:
        observed._SPACE.clear()
        space = observed.observed_space(photographs=photographs, geometry_frames=geometry)
        low, high, size, state = legacy.observed_space_state(photographs=expected[0], geometry_frames=expected[1])
        assert space['voxel_size'] == size and np.array_equal(space['minimum'], low)
        assert np.array_equal(space['maximum'], high) and np.array_equal(space['state'], state)
        assert len(np.unique(state)) >= 3
    assert keeps_no_pixels(frames) and keeps_no_pixels(separate)


def test_camera_planning_reads_each_view_once_per_pass_and_places_the_same_cameras(retained, tmp_path):
    from blueprint_pipeline import atlas_completion_cameras as cameras
    from blueprint_pipeline.atlas_scene_completion import camera_candidates
    from blueprint_pipeline.atlas_scene_gaps import survey_gaps

    frames, eager = retained
    pool = camera_candidates(frames)
    wide = [row for row in pool if row['purpose'] == 'wide']
    expected = [row for row in pool if row['purpose'] == 'disocclusion']
    assert expected and expected == legacy.disocclusion_candidates(eager, wide)
    survey = survey_gaps(frames=frames, geometry_frames=frames, candidates=wide, root=tmp_path / 'gaps')
    assert keeps_no_pixels(frames)
    legacy_survey = {key: value for key, value in survey.items() if key != 'gap_camera_origin_ranking'}
    placed = []
    for plan, options in [(survey, {}), (legacy_survey, {}), (survey, {'source_centred': True})]:
        targets = cameras.gap_target_candidates(frames, pool, plan, **options)
        assert targets == legacy.gap_target_candidates(eager, pool, plan, **options)
        placed.append(targets)
    assert placed[0] and any('origin_ranking' in row for row in placed[0])
    assert keeps_no_pixels(frames)


def test_gap_composite_renders_one_reference_at_a_time(retained, monkeypatch):
    frames, eager = retained
    rgb, depth, _ = render_room(frames[1], 0.7)
    mask = np.zeros(depth.shape, bool)
    mask[20:60, 40:120] = True
    camera = {'intrinsics': frames[1]['intrinsics'], 'extrinsics': {**frames[1]['extrinsics'], 'position': [
        value + offset for value, offset in zip(frames[1]['extrinsics']['position'], (0.05, 0.0, 0.03))]}}
    moved = depth_splat.posed_frame(frame_id='gap', rgb=rgb, depth=depth.astype(np.float32),
                                    confidence=np.ones(depth.shape, np.float32), camera=camera)
    at_centre = depth_splat.posed_frame(frame_id='at-centre', rgb=rgb, depth=depth.astype(np.float32),
                                        confidence=np.ones(depth.shape, np.float32), camera=frames[1])
    views = [depth_splat.restrict_frame(view, mask) for view in (moved, at_centre)]
    references = legacy.observation_references(eager)
    expected = [composite.composite_view(view, references) for view in views]
    for budget in (None, 0):
        if budget is not None:
            monkeypatch.setattr(composite, 'COMPOSITE_RESIDENT_DEPTH_BYTES', budget)
        lazy = composite.observation_references(frames)
        for view, (want, receipt) in zip(views, expected, strict=True):
            got, got_receipt = composite.composite_view(view, lazy)
            assert json.dumps(got_receipt) == json.dumps(receipt)
            assert all(np.array_equal(got[key], want[key]) for key in ('rgb', 'depth', 'confidence'))
        assert keeps_no_pixels(frames)
    assert [receipt['observed_basis'] for _, receipt in expected] == ['recorded_depth_splats',
                                                                       'recorded_centre_photographs']


def _helpers_peak(pose_root):
    """Traced peak of the completion's per-view passes over one retained pose, budgets at zero."""
    from blueprint_pipeline.atlas_completion_cameras import context_samples
    from blueprint_pipeline.atlas_scene_chisel import project_source_depths
    from blueprint_pipeline.atlas_scene_completion import camera_candidates
    from blueprint_pipeline.atlas_scene_recovery import _frame_binding

    gc.collect()
    tracemalloc.start()
    try:
        _, frames = depth_splat.load_posed_rgbd(pose_root)
        preservation._SUPPORT_MASKS.clear()
        observed._SPACE.clear()
        [_frame_binding(frame) for frame in frames]
        pool = camera_candidates(frames)
        preservation.observed_scene_bounds(frames)
        observed.observed_space(photographs=frames, geometry_frames=frames)
        context_samples(frames)
        # Two cameras away from every recorded centre, on the views' own grid.
        moving = []
        for row in [row for row in pool if row['purpose'] in ('translated', 'disocclusion')][:2]:
            k = row['camera']['intrinsics']
            scale = frames[0]['intrinsics']['width'] / k['width']
            moving.append({**row['camera'], 'intrinsics': {
                **k, **{key: k[key] * scale for key in ('fx', 'fy', 'cx', 'cy')},
                'width': frames[0]['intrinsics']['width'], 'height': frames[0]['intrinsics']['height']}})
        assert len(moving) == 2
        preservation.project_known_contexts(preservation._reused_supported_frames(frames), moving)
        project_source_depths(frames, moving)
        references = composite.observation_references(frames)
        composite.render_observed(references, moving[0])
        assert keeps_no_pixels(frames)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_completion_passes_do_not_grow_with_the_number_of_views(tmp_path, monkeypatch):
    """Eight more views of the same room add far less than a fifth of a view's pixels each.

    The passes held every view read: about a view's pixels for each added view,
    and more where they kept derived copies. Now only planning samples (1/25
    and 1/64 of each view), packed support masks and remembered digests stay
    per view; the kept support references and smoothed depths are fixed
    budgets, set to nothing here. The observed-space grid spends one voxel per
    pixel up to ``OBSERVED_SPACE_VOXELS``, which 1280x720 views reach at three;
    these small views reach a smaller cap at four.
    """
    monkeypatch.setattr(observed, 'OBSERVED_SPACE_VOXELS', 2 ** 18)
    monkeypatch.setattr(preservation, 'SUPPORT_REFERENCE_BYTES', 0)
    monkeypatch.setattr(composite, 'COMPOSITE_RESIDENT_DEPTH_BYTES', 0)
    monkeypatch.setattr(preservation, '_SUPPORT_MASKS', {})
    monkeypatch.setattr(observed, '_SPACE', {})
    width, height = 256, 144
    roots = [write_room_pose_root(tmp_path / f'pose-{views}', 4, width=width, height=height, views=views,
                                  holes=HOLES) for views in (2, 4, 12)]
    _helpers_peak(roots[0])  # first-call allocations (imports, caches) are not the views'
    few, many = _helpers_peak(roots[1]), _helpers_peak(roots[2])
    view_bytes = width * height * (3 + 4 + 4)
    assert many - few < 8 * 0.2 * view_bytes, (few, many, view_bytes)


def test_lazy_views_read_their_pixels_when_used_and_keep_none(tmp_path):
    root = write_room_pose_root(tmp_path / 'pose', 3, width=64, height=48)
    _, frames = depth_splat.load_posed_rgbd(root)
    first = frames[0]
    rgb, depth, confidence = depth_splat.frame_arrays(first, 'rgb', 'depth', 'confidence')
    record = depth_splat.pixel_record(first, 'depth')
    assert record == (depth.dtype.str, depth.shape, hashlib.sha256(np.ascontiguousarray(depth).tobytes()).hexdigest())
    assert depth_splat.frame_shape(first, 'rgb') == rgb.shape and keeps_no_pixels(frames)
    view = depth_splat.decoded_frame(first, 'depth')
    assert list(view) == [key for key in first if key not in ('rgb', 'confidence')]
    assert np.array_equal(view['depth'], depth, equal_nan=True) and keeps_no_pixels(frames)
    derived = depth_splat.derived_frame(first, confidence=lambda value: value * 0)
    assert list(derived) == list(first) and not derived['confidence'].any() and derived == derived
    assert np.array_equal(derived['rgb'], rgb) and keeps_no_pixels(frames)
    # A replaced array is read, and digested, as the frame now holds it.
    first['depth'] = np.zeros_like(depth)
    assert depth_splat.pixel_digest(first, 'depth') == hashlib.sha256(np.zeros_like(depth).tobytes()).hexdigest()
    del first['depth']
    with pytest.raises(KeyError):
        depth_splat.frame_shape(first, 'depth')
    turn = depth_splat.read_in_turn(frames[1:])
    assert turn[0]['rgb'] is turn[0]['rgb'], 'kept while the same view is read'
    held = turn[0]['rgb']
    assert turn[1]['rgb'] is not held and turn[0]['rgb'] is not held, 'another view replaced it'
    assert keeps_no_pixels(frames[1:])


def _retained_view(root, frame_id, cam):
    """One digest-bound retained view of the room at ``cam``, as ``load_posed_rgbd`` returns it."""
    import cv2
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from tests.test_atlas_depth_splat import write_exr

    rgb, depth, _ = render_room(cam)
    folder = root / frame_id
    folder.mkdir(parents=True)
    write_exr(folder / 'depth.bin', depth.astype(np.float32))
    for role, pixels in (('image', np.ascontiguousarray(rgb[..., ::-1])),
                         ('confidence', np.full(depth.shape, 255, np.uint8))):
        assert cv2.imwrite(str(folder / f'{role}.png'), pixels)
        (folder / f'{role}.png').rename(folder / f'{role}.bin')
    return depth_splat.PosedFrame(frame_id=frame_id, camera=cam, outputs={
        role: (folder / f'{role}.bin', _sha256_file(folder / f'{role}.bin')) for role in ('image', 'depth', 'confidence')})


def test_a_splat_request_on_mixed_grids_reads_one_retained_view_at_a_time(tmp_path):
    from blueprint_pipeline.atlas_native_splat import prepare_splat_request
    from tests.atlas_posed_rgbd_synthetic import room_camera

    cameras = [room_camera(0, 6, width=48, height=80), room_camera(1, 6, width=128, height=72)]
    frames = [_retained_view(tmp_path / 'views', f'view-{index}', cam) for index, cam in enumerate(cameras)]
    _, eager_uploads, eager_provenance = prepare_splat_request(frames=[legacy.eager(frame) for frame in [
        _retained_view(tmp_path / 'again', f'view-{index}', cam) for index, cam in enumerate(cameras)]],
        root=tmp_path / 'eager')
    _, uploads, provenance = prepare_splat_request(frames=frames, root=tmp_path / 'lazy')
    assert keeps_no_pixels(frames)
    assert [row['sha256'] for row in uploads.values()] == [row['sha256'] for row in eager_uploads.values()]
    assert json.dumps(provenance) == json.dumps(eager_provenance) and len(uploads) > 2 * len(frames) + 1


@pytest.mark.slow
def test_a_whole_preserved_completion_keeps_no_view_and_writes_the_same_files(tmp_path, monkeypatch):
    """The rehearsal (``tests.atlas_preserved_completion_synthetic``) of a completion that admits
    generated geometry, at Atlas's 1280x720 grid: lazy views and eager dicts write every file alike,
    no retained view keeps its pixels, and eight more views of the same room barely move the peak.

    The completed splat's fusion (``fuse_posed_frames``) is left out of the peak: its voxel sums
    follow the scene's voxels, and each lap of this walk adds its own depth error to the same walls.
    Measured 2026-11-09 with every budget at nothing: 420 MB with 4 views, 436 MB with 12.
    """
    from tests.atlas_preserved_completion_synthetic import Rehearsal, output_tree

    monkeypatch.setattr(preservation, 'SUPPORT_REFERENCE_BYTES', 0)
    monkeypatch.setattr(composite, 'COMPOSITE_RESIDENT_DEPTH_BYTES', 0)
    monkeypatch.setattr(depth_splat, 'FUSION_RESIDENT_DEPTH_BYTES', 0)
    fuse, peaks = depth_splat.write_depth_splat, []

    def fused_apart(**options):
        if tracemalloc.is_tracing():
            peaks.append(tracemalloc.get_traced_memory()[1])
        try:
            return fuse(**options)
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.reset_peak()

    def rehearse(views, eager=False, trace=False):
        root = tmp_path / 'rehearsal'
        shutil.rmtree(root, ignore_errors=True)
        preservation._SUPPORT_MASKS.clear()
        observed._SPACE.clear()
        rehearsal = Rehearsal(root, 4, width=1280, height=720, views=views)
        pose, frames = depth_splat.load_posed_rgbd(rehearsal.pose_root)
        fuse(frames=frames, result=pose, output_root=root / 'splat')
        if eager:
            frames = [legacy.eager(frame) for frame in frames]
        peaks.clear()
        gc.collect()
        if trace:
            tracemalloc.start()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(depth_splat, 'write_depth_splat', fused_apart)
                result, composites = rehearsal.run(frames, pose)
            peak = max([*peaks, tracemalloc.get_traced_memory()[1]]) if trace else None
        finally:
            if trace:
                tracemalloc.stop()
        assert result['generated_geometry_admitted'] and composites
        if not eager:
            assert keeps_no_pixels(frames)
        return output_tree(root), composites, peak

    lazy_tree, lazy_composites, few = rehearse(4, trace=True)
    eager_tree, eager_composites, _ = rehearse(4, eager=True)
    assert lazy_tree == eager_tree and json.dumps(lazy_composites) == json.dumps(eager_composites)
    assert any('/admission-' in name for name in lazy_tree)
    _, _, many = rehearse(12, trace=True)
    view_bytes = 1280 * 720 * (3 + 4 + 4)
    assert many - few < 8 * 0.5 * view_bytes, (few, many)
