"""Scene-relative disocclusion hypotheses inside source-observed free space.

These are estimated generation cameras, never collision-qualified trajectories.
The source image/depth/camera context remains unchanged.
"""
from copy import deepcopy

import numpy as np


def observed_free_position(position, frames, bounds):
    from .atlas_scene_preservation import _project, _valid

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
    """Move within photographed free space and look back at captured surfaces.

    Offsets scale with estimated scene depth. The farthest supported offset in
    each horizontal direction exposes surfaces hidden by the source trajectory.
    The planner still chooses which, if any, are useful generation requests.
    """
    from scipy.spatial.transform import Rotation
    from .atlas_scene_preservation import observed_scene_bounds, _world, _valid

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
                    # A downward capture must not restrict every new camera to
                    # another floor close-up; also survey its upper known scene.
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


# Observed context of a candidate gap camera is estimated on cells of this many
# pixels, from every frame's confident depth sampled at this stride.
CONTEXT_CELL_PIXELS = 16
CONTEXT_SAMPLE_STRIDE = 8
# At most this many of a gap's pixels are lifted to test whether a camera sees it.
GAP_EXPOSURE_SAMPLES = 2048


def aimed_rotation(origin, focus):
    """The RUB rotation looking from ``origin`` at ``focus`` with world-up kept up; None when vertical."""
    forward = np.asarray(focus, float) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    if np.linalg.norm(right) < 1e-6:
        return None
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), -forward))


def context_samples(frames):
    """Strided world samples of each frame's confident depth, and whether another view supports each.

    Support is the same cross-view agreement the completion keep mask requires.
    """
    from .atlas_scene_preservation import _reused_supported_frames, _valid, _world

    points, supported = [], []
    for frame, kept in zip(frames, _reused_supported_frames(frames), strict=True):
        grid = np.zeros(frame['depth'].shape, bool)
        grid[::CONTEXT_SAMPLE_STRIDE, ::CONTEXT_SAMPLE_STRIDE] = True
        sample, y, x = _world(frame, grid & _valid(frame))
        points.append(sample)
        supported.append(_valid(kept)[y, x])
    return (np.concatenate(points) if points else np.empty((0, 3)),
            np.concatenate(supported) if supported else np.empty(0, bool))


def observed_context(samples, supported, frames, camera, gap_points, gap_weights):
    """Estimate (observed context fraction, exposed gap pixels) for one candidate camera.

    A cell is observed when the cross-view-supported samples on its nearest
    surface reach one per target pixel, as the keep mask's one-pixel projection
    needs; at a recorded centre the photographs it holds are resampled exactly.
    A magnified close-up therefore scores low. Lifted gap points count as
    exposed where no nearer observed surface covers them. A point's weight is
    the source pixels it stands for times their squared depth there, so it
    covers weight / z**2 pixels at depth z here (same focal length).
    """
    from .atlas_depth_splat import MAX_MEDIAN_CROSS_VIEW_RESIDUAL, _rotation_matrix
    from .atlas_scene_preservation import _project, _same_center

    k = camera['intrinsics']
    cell = CONTEXT_CELL_PIXELS
    rows, cols = -(-int(k['height']) // cell), -(-int(k['width']) // cell)
    z, u, v, inside = _project(samples, camera)
    index = (v[inside] // cell).astype(int) * cols + (u[inside] // cell).astype(int)
    depth = z[inside]
    front = np.full(rows * cols, np.inf)
    np.minimum.at(front, index, depth)
    surface = depth <= front[index] * (1 + MAX_MEDIAN_CROSS_VIEW_RESIDUAL)
    density = (np.bincount(index[surface & supported[inside]], minlength=rows * cols)
               * CONTEXT_SAMPLE_STRIDE ** 2 / cell ** 2)
    observed = density >= 1
    rotation = _rotation_matrix(camera['extrinsics']['quaternion'])
    for frame in frames:
        if not _same_center(frame, camera):
            continue
        yy, xx = (np.indices((rows, cols)).reshape(2, -1) + .5) * cell
        rays = np.column_stack(((xx - k['cx']) / k['fx'], -(yy - k['cy']) / k['fy'], -np.ones(len(xx))))
        _, _, _, photographed = _project(frame['position'] + rays @ rotation.T, frame)
        observed |= photographed
    z, u, v, inside = _project(gap_points, camera)
    index = (v[inside] // cell).astype(int) * cols + (u[inside] // cell).astype(int)
    exposed = front[index] >= z[inside] * (1 - MAX_MEDIAN_CROSS_VIEW_RESIDUAL)
    return float(observed.mean()), float(np.sum(gap_weights[inside][exposed] / z[inside][exposed] ** 2))


def gap_target_candidates(frames, candidates, survey, *, source_centred=False, expected_frames=()):
    """Aim at actual gaps, estimating the aim point from neighbouring depth.

    Camera locations come from the captured trajectory or its already screened
    free-space hypotheses. The aim is a planning estimate, never new geometry.
    Nearby estimates of the same gap merge; no fixed camera count is imposed.

    A survey that names ``observed_context_v1`` ranking (v2) places each camera
    where the capture observes most of the view (``observed_context``) among
    the locations that still expose at least the gap's minimum dedicated-camera
    area; the largest change of view angle breaks ties, and stands alone when
    no location is estimated to expose the gap. In capture-447ae4f4
    (2026-10-30) the largest angle alone put both gap cameras 0.52 units above
    the dining table, 1.77 units past the nearest recorded camera, where the
    keep mask held 41.6% and 32.5% of the frame. A survey saved without it
    keeps that largest-angle choice, so a retained plan's cameras are unchanged.
    No camera is aimed at a gap flagged ``photographed_without_depth``.
    """
    from pathlib import Path
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial.transform import Rotation
    from .atlas_scene_preservation import _project, _valid, _world
    from .atlas_depth_splat import MAX_MEDIAN_CROSS_VIEW_RESIDUAL
    from .decision_evidence_contracts import canonical_digest
    from .local_reconstruction_adapters import _sha256_file

    cloud = []
    for frame in frames:
        # A planning sample only. Native context images and depth stay full size.
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
    from .atlas_scene_gaps import GAP_CAMERA_ORIGIN_RANKING
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
            from .atlas_scene_gaps import expected_depth_for_candidate
            expected_depth = expected_depth_for_candidate(expected_frames, original)
        for gap in view['gaps']:
            if gap.get('photographed_without_depth'):
                # The recorded views that photographed it are already candidates;
                # a new angle on it would only invite generation to redraw it.
                continue
            mask = labels == gap['label']
            yy, xx = np.nonzero(mask)
            if not len(xx):
                continue
            # Tiny holes already have the original candidate views. A dedicated
            # new angle must cover more than the accepted depth-uncertainty
            # footprint; otherwise a noisy surface produces hundreds of cameras.
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
            # Merge duplicate angular surveys of the same small scene region.
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
                # Do not point through a known obstruction from a recorded view.
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
            # If a translated hypothesis loses its photographed surroundings,
            # re-aim from the source view instead. Other context/anchor cameras
            # still supply the translation needed for joint output depth.
            origin = position if source_centred else max(choices, key=lambda item: item[0])[1]
            placement = {}
            if ranked:
                # The gap's pixels lifted to the depth hypothesis; each sample
                # stands for an equal share of them (``observed_context`` weights).
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
            # Carry the original gap's extent into the new camera. Re-aiming
            # must not turn the rest of the new image into a generation target.
            corners = np.array([[xx.min(), yy.min()], [xx.max() + 1, yy.min()],
                                [xx.max() + 1, yy.max() + 1], [xx.min(), yy.max() + 1]], float)
            rays = np.column_stack(((corners[:, 0] - k['cx']) / k['fx'],
                                    -(corners[:, 1] - k['cy']) / k['fy'], -np.ones(4)))
            boundary = position + (rays * depth) @ rotation.T
            zz, uu, vv, _ = _project(boundary, target['camera'])
            if not np.isfinite(uu).all() or not np.isfinite(vv).all() or np.any(zz <= 0):
                continue
            # The named mask already limits what may be added. Zooming the
            # output to a small patch can remove the connected fixture and
            # layout needed to interpret that patch. Keep this source camera's
            # field of view while re-aiming; adapt to its actual intrinsics,
            # rather than imposing one field of view on every capture.
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
