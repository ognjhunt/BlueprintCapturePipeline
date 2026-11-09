"""The completion path's view-reading code of origin/main before 2026-11-09, frozen as the reference.

These held every view's pixels: each read a frame's pixels as often as it
needed them, and a retained ``PosedFrame`` kept what was read. Run them on
plain dict frames (``eager``); the restructured code must give the same bytes.
Helpers that did not change (``_project``, ``_valid``, ``_world``,
``observed_context``, ``aimed_rotation``) are imported.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from blueprint_pipeline.atlas_completion_cameras import (
    CONTEXT_SAMPLE_STRIDE, GAP_EXPOSURE_SAMPLES, aimed_rotation, observed_context,
)
from blueprint_pipeline.atlas_depth_splat import EDGE_RELATIVE_JUMP, MAX_MEDIAN_CROSS_VIEW_RESIDUAL, smoothed_depth
from blueprint_pipeline.atlas_observed_space import (
    FREE, OBSERVED_SPACE_VOXELS, RANGED, UNRANGED, UNSEEN, _CHUNK_VOXELS,
)
from blueprint_pipeline.atlas_scene_preservation import _project, _valid, _world, observed_scene_bounds


def eager(frame):
    """The dict ``posed_frame`` returned for a retained view, every pixel decoded."""
    return {key: frame[key] for key in frame}


def support_masks(frames):
    masks = []
    validity = [_valid(frame) for frame in frames]
    for source, source_valid in zip(frames, validity, strict=True):
        valid = source_valid
        depth = source['depth']
        center = depth[1:-1, 1:-1]
        interior = valid[1:-1, 1:-1].copy()
        for neighbour in (depth[1:-1, 2:], depth[1:-1, :-2], depth[2:, 1:-1], depth[:-2, 1:-1]):
            interior &= np.isfinite(neighbour) & (np.abs(neighbour - center) <= EDGE_RELATIVE_JUMP * center)
        valid = np.zeros(valid.shape, bool)
        valid[1:-1, 1:-1] = interior
        points, y, x = _world(source, valid)
        supported = np.zeros(len(points), bool)
        colors = source['rgb'][y, x].astype(np.int16)
        for other, other_valid in zip(frames, validity, strict=True):
            if other is source:
                continue
            pending = np.flatnonzero(~supported)
            if not len(pending):
                break
            camera = {'intrinsics': other['intrinsics'], 'extrinsics': other['extrinsics']}
            z, u, v, inside = _project(points[pending], camera)
            ids = pending[inside]
            rows, cols = np.floor(v[inside]).astype(int), np.floor(u[inside]).astype(int)
            observed = other['depth'][rows, cols]
            agrees = (other_valid[rows, cols]
                      & (np.abs(z[inside] - observed) <= EDGE_RELATIVE_JUMP * observed)
                      & (np.max(np.abs(colors[ids] - other['rgb'][rows, cols].astype(np.int16)), axis=-1) <= 32))
            supported[ids[agrees]] = True
        keep = np.zeros(valid.shape, bool)
        keep[y[supported], x[supported]] = True
        masks.append(keep)
    return masks


def supported_projection_frames(frames):
    return [{**source, 'confidence': source['confidence'] * keep}
            for source, keep in zip(frames, support_masks(frames), strict=True)]


def observed_space_state(*, photographs, geometry_frames):
    """``atlas_observed_space.observed_space``'s grid parameters and voxel states."""
    bounds = observed_scene_bounds([*geometry_frames, *photographs])
    low, high = np.asarray(bounds["minimum"], float), np.asarray(bounds["maximum"], float)
    extent = np.maximum(high - low, 1e-6)
    pixels = sum(frame["depth"].size for frame in [*geometry_frames, *photographs])
    size = float(np.cbrt(np.prod(extent) / min(OBSERVED_SPACE_VOXELS, pixels)))
    shape = np.maximum(np.ceil(extent / size).astype(int), 1)
    half_diagonal = size * np.sqrt(3) / 2
    references = [(frame, _valid(frame)) for frame in geometry_frames]
    photographed = [(frame, _valid(frame)) for frame in photographs]
    state = np.full(int(np.prod(shape)), UNSEEN, np.uint8)
    for start in range(0, state.size, _CHUNK_VOXELS):
        index = np.arange(start, min(start + _CHUNK_VOXELS, state.size))
        centres = low + (np.column_stack(np.unravel_index(index, shape)) + .5) * size
        free = np.zeros(len(index), bool)
        ranged = np.zeros(len(index), bool)
        unranged = np.zeros(len(index), bool)
        for frame, valid in references:
            z, u, v, inside = _project(centres, frame)
            seen = np.flatnonzero(inside)
            rows, cols = np.floor(v[seen]).astype(int), np.floor(u[seen]).astype(int)
            known = valid[rows, cols]
            depth = frame["depth"][rows, cols]
            tolerance = np.maximum(MAX_MEDIAN_CROSS_VIEW_RESIDUAL * depth, half_diagonal)
            free[seen[known & (z[seen] < depth - tolerance)]] = True
            ranged[seen[known & (np.abs(z[seen] - depth) <= tolerance)]] = True
        for frame, valid in photographed:
            z, u, v, inside = _project(centres, frame)
            seen = np.flatnonzero(inside)
            unranged[seen[~valid[np.floor(v[seen]).astype(int), np.floor(u[seen]).astype(int)]]] = True
        state[index] = np.where(ranged, RANGED, np.where(free, FREE, np.where(unranged, UNRANGED, UNSEEN)))
    return low, high, size, state.reshape(shape)


def observed_free_position(position, frames, bounds):
    point = np.asarray(position)
    if not np.all((point >= bounds['minimum']) & (point <= bounds['maximum'])):
        return False
    for frame in frames:
        z, u, v, inside = _project(point[None], frame)
        if not inside[0]:
            continue
        y, x = int(v[0]), int(u[0])
        if _valid(frame)[y, x] and z[0] < .9 * frame['depth'][y, x]:
            return True
    return False


def disocclusion_candidates(frames, wide):
    from scipy.spatial.transform import Rotation

    if len(frames) < 2:
        return []
    bounds = observed_scene_bounds(frames)
    result = []
    for frame, original in zip(frames, wide, strict=True):
        valid = _valid(frame)
        yy, xx = np.nonzero(valid)
        sampled = np.zeros(valid.shape, bool)
        step = max(1, len(yy) // 1024)
        sampled[yy[::step], xx[::step]] = True
        points, _, _ = _world(frame, sampled)
        focus = np.median(points, axis=0)
        distance = float(np.median(frame['depth'][valid]))
        for name, axis in [('right', frame['rotation'][:, 0]), ('forward', -frame['rotation'][:, 2])]:
            direction = axis.copy()
            direction[1] = 0
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            direction /= length
            for sign in [-1, 1]:
                position = None
                for fraction in [.15, .3, .6]:
                    proposed = frame['position'] + sign * fraction * distance * direction
                    if observed_free_position(proposed, frames, bounds):
                        position = proposed
                if position is None:
                    continue
                surroundings = focus.copy()
                surroundings[1] = np.quantile(points[:, 1], .85)
                for aim, look_at in [('surface', focus), ('surroundings', surroundings)]:
                    forward = look_at - position
                    forward /= np.linalg.norm(forward)
                    right = np.cross(forward, [0., 1., 0.])
                    if np.linalg.norm(right) < 1e-6:
                        continue
                    right /= np.linalg.norm(right)
                    up = np.cross(right, forward)
                    target = deepcopy(original)
                    target.update(camera_id=original['camera_id'] + f'-disocclusion-{name}-{sign:+d}-{aim}',
                        purpose='disocclusion', gap_domain='projected_scene_outline_hypothesis',
                        translation_basis='source_depth_scaled_inside_observed_free_space',
                        camera_location_observed_free=True, collision_qualified=False)
                    k = target['camera']['intrinsics']
                    k.update(cx=k['width'] / 2, cy=k['height'] / 2)
                    target['camera']['extrinsics'].update(position=position.tolist(), quaternion=Rotation.from_matrix(
                        np.column_stack((right, up, -forward))).as_quat().tolist())
                    result.append(target)
    return result


def context_samples(frames):
    points, supported = [], []
    for frame, kept in zip(frames, supported_projection_frames(frames), strict=True):
        grid = np.zeros(frame['depth'].shape, bool)
        grid[::CONTEXT_SAMPLE_STRIDE, ::CONTEXT_SAMPLE_STRIDE] = True
        sample, y, x = _world(frame, grid & _valid(frame))
        points.append(sample)
        supported.append(_valid(kept)[y, x])
    return (np.concatenate(points) if points else np.empty((0, 3)),
            np.concatenate(supported) if supported else np.empty(0, bool))


def gap_target_candidates(frames, candidates, survey, *, source_centred=False, expected_frames=()):
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial.transform import Rotation
    from blueprint_pipeline.atlas_scene_gaps import GAP_CAMERA_ORIGIN_RANKING
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    cloud = []
    for frame in frames:
        stride = max(1, min(frame['depth'].shape) // 128)
        sample = np.zeros(frame['depth'].shape, bool)
        sample[::stride, ::stride] = True
        points, _, _ = _world(frame, sample & _valid(frame))
        cloud.append(points)
    cloud = np.concatenate(cloud) if cloud else np.empty((0, 3))
    if not len(cloud):
        return []
    by_id = {row['camera_id']: row for row in candidates}
    origins = [np.asarray(row['camera']['extrinsics']['position'], float) for row in candidates
               if row.get('camera_location_observed_free')]
    origins.extend(np.asarray(frame['position'], float) for frame in frames)
    origins = np.unique(np.asarray(origins), axis=0)
    source_validity = [(frame, _valid(frame)) for frame in frames]
    ranked = not source_centred and survey.get('gap_camera_origin_ranking') == GAP_CAMERA_ORIGIN_RANKING
    if ranked:
        samples, supported = context_samples(frames)
    result, targets = [], []
    for view in survey['views']:
        if not view['gaps']:
            continue
        original = by_id[view['camera_id']]
        path = Path(view['label_path'])
        if path.is_symlink() or _sha256_file(path) != view['label_sha256']:
            raise ValueError('atlas_completion_gap_survey_changed')
        labels = np.asarray(Image.open(path))
        camera = original['camera']
        k, e = camera['intrinsics'], camera['extrinsics']
        z, u, v, inside = _project(cloud, camera)
        indices = np.flatnonzero(inside)
        rows, cols = np.floor(v[inside]).astype(int), np.floor(u[inside]).astype(int)
        rotation = Rotation.from_quat(e['quaternion']).as_matrix()
        position = np.asarray(e['position'], float)
        expected_depth = None
        if expected_frames:
            from blueprint_pipeline.atlas_scene_gaps import expected_depth_for_candidate
            expected_depth = expected_depth_for_candidate(expected_frames, original)
        for gap in view['gaps']:
            if gap.get('photographed_without_depth'):
                continue
            mask = labels == gap['label']
            yy, xx = np.nonzero(mask)
            if not len(xx):
                continue
            uncertainty_pixels = MAX_MEDIAN_CROSS_VIEW_RESIDUAL ** 2 * k['fx'] * k['fy']
            if len(xx) < uncertainty_pixels:
                continue
            distance = distance_transform_edt(~mask)
            band = max(3., .1 * np.hypot(np.ptp(xx) + 1, np.ptp(yy) + 1))
            neighbours = indices[(distance[rows, cols] > 0) & (distance[rows, cols] <= band)]
            hypothesis = (expected_depth[mask] if expected_depth is not None else np.array([]))
            hypothesis = hypothesis[np.isfinite(hypothesis) & (hypothesis > 0)]
            if len(neighbours) < 3 and not len(hypothesis):
                continue
            depth = float(np.median(hypothesis) if len(hypothesis) else np.median(z[neighbours]))
            pixel = np.array([np.median(xx) + .5, np.median(yy) + .5])
            ray = np.array([(pixel[0] - k['cx']) / k['fx'],
                            -(pixel[1] - k['cy']) / k['fy'], -1.])
            focus = position + rotation @ (ray * depth)
            radius = depth * max(MAX_MEDIAN_CROSS_VIEW_RESIDUAL, max(3., band) / max(k['fx'], k['fy']))
            if any(np.linalg.norm(focus - known) <= max(radius, prior_radius)
                   for known, prior_radius in targets):
                continue
            parent_direction = focus - position
            parent_direction /= np.linalg.norm(parent_direction)
            choices = []
            for origin in origins:
                direction = focus - origin
                length = np.linalg.norm(direction)
                if length < radius:
                    continue
                obstructed = False
                for frame, valid in source_validity:
                    if not np.allclose(origin, frame['position'], atol=1e-8, rtol=0):
                        continue
                    zz, uu, vv, visible = _project(focus[None], frame)
                    if visible[0]:
                        y, x = int(vv[0]), int(uu[0])
                        obstructed = bool(valid[y, x] and zz[0] > 1.05 * frame['depth'][y, x])
                if not obstructed:
                    choices.append((float(1 - np.dot(direction / length, parent_direction)), origin))
            if not choices:
                continue
            origin = position if source_centred else max(choices, key=lambda item: item[0])[1]
            placement = {}
            if ranked:
                pick = np.linspace(0, len(xx) - 1, min(len(xx), GAP_EXPOSURE_SAMPLES)).astype(int)
                gap_rays = np.column_stack(((xx[pick] + .5 - k['cx']) / k['fx'],
                                            -(yy[pick] + .5 - k['cy']) / k['fy'], -np.ones(len(pick))))
                gap_points = position + (gap_rays * depth) @ rotation.T
                weights = np.full(len(pick), len(xx) / len(pick) * depth ** 2)
                scored = []
                for angle, candidate in choices:
                    aimed = aimed_rotation(candidate, focus)
                    if aimed is None:
                        continue
                    view_camera = {'intrinsics': {**k, 'cx': k['width'] / 2, 'cy': k['height'] / 2},
                                   'extrinsics': {**e, 'position': candidate.tolist(),
                                                  'quaternion': Rotation.from_matrix(aimed).as_quat().tolist()}}
                    fraction, exposed = observed_context(samples, supported, frames, view_camera,
                                                         gap_points, weights)
                    if exposed >= uncertainty_pixels:
                        scored.append(((fraction, angle, tuple(candidate)), candidate, fraction, exposed))
                if scored:
                    _, origin, fraction, exposed = max(scored, key=lambda item: item[0])
                    placement = {'origin_ranking': GAP_CAMERA_ORIGIN_RANKING,
                                 'observed_context_fraction_estimate': fraction,
                                 'exposed_gap_pixel_estimate': exposed}
            aimed = aimed_rotation(origin, focus)
            if aimed is None:
                continue
            target = deepcopy(original)
            identity_data = {'gap': gap['gap_id'], 'focus': focus.tolist(), 'origin': origin.tolist(),
                             'framing': 'source_fov_reaimed_at_gap_v2'}
            if source_centred:
                identity_data['framing'] = 'source_centred_full_fov_v2'
            identity = canonical_digest(identity_data)
            target.update(camera_id='gap-target-' + identity[7:23], purpose='gap_target',
                          source_gap_ids=[gap['gap_id']], target_world_estimate=focus.tolist(),
                          target_basis=('gap_centroid_ray_at_reviewed_depth_hypothesis' if len(hypothesis)
                                        else 'gap_centroid_ray_at_neighbouring_source_depth'),
                          camera_location_observed_free=True, collision_qualified=False, **placement)
            target['camera']['extrinsics'].update(position=origin.tolist(),
                                                  quaternion=Rotation.from_matrix(aimed).as_quat().tolist())
            target['camera']['intrinsics'].update(cx=k['width'] / 2, cy=k['height'] / 2)
            corners = np.array([[xx.min(), yy.min()], [xx.max() + 1, yy.min()],
                                [xx.max() + 1, yy.max() + 1], [xx.min(), yy.max() + 1]], float)
            rays = np.column_stack(((corners[:, 0] - k['cx']) / k['fx'],
                                    -(corners[:, 1] - k['cy']) / k['fy'], -np.ones(4)))
            boundary = position + (rays * depth) @ rotation.T
            zz, uu, vv, _ = _project(boundary, target['camera'])
            if not np.isfinite(uu).all() or not np.isfinite(vv).all() or np.any(zz <= 0):
                continue
            target_k = target['camera']['intrinsics']
            target.update(gap_target_polygon_pixels=np.column_stack((uu, vv)).tolist(),
                          gap_domain='projected_scene_outline_hypothesis',
                          target_extent_basis='source_gap_box_at_neighbouring_depth_estimate',
                          target_framing='source_fov_reaimed_at_gap',
                          target_vertical_fov_degrees=float(np.degrees(2 * np.arctan(
                              target_k['height'] / (2 * target_k['fy'])))))
            result.append(target)
            targets.append((focus, radius))
    return result


def observation_references(frames):
    return [{**frame, "depth": smoothed_depth(frame["depth"])} for frame in frames]
