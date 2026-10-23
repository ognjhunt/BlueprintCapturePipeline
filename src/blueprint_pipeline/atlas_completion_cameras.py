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


def gap_target_candidates(frames, candidates, survey):
    """Aim at actual gaps, estimating the aim point from neighbouring depth.

    Camera locations come from the captured trajectory or its already screened
    free-space hypotheses. The aim is a planning estimate, never new geometry.
    Nearby estimates of the same gap merge; no fixed camera count is imposed.
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
        for gap in view['gaps']:
            mask = labels == gap['label']
            yy, xx = np.nonzero(mask)
            if not len(xx):
                continue
            distance = distance_transform_edt(~mask)
            band = max(3., .1 * np.hypot(np.ptp(xx) + 1, np.ptp(yy) + 1))
            neighbours = indices[(distance[rows, cols] > 0) & (distance[rows, cols] <= band)]
            if len(neighbours) < 3:
                continue
            depth = float(np.median(z[neighbours]))
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
            origin = max(choices, key=lambda item: item[0])[1]
            forward = focus - origin
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0., 1., 0.])
            if np.linalg.norm(right) < 1e-6:
                continue
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            target = deepcopy(original)
            identity = canonical_digest({'gap': gap['gap_id'], 'focus': focus.tolist(), 'origin': origin.tolist()})
            target.update(camera_id='gap-target-' + identity[7:23], purpose='gap_target',
                          source_gap_ids=[gap['gap_id']], target_world_estimate=focus.tolist(),
                          target_basis='gap_centroid_ray_at_neighbouring_source_depth',
                          camera_location_observed_free=True, collision_qualified=False)
            target['camera']['extrinsics'].update(position=origin.tolist(), quaternion=Rotation.from_matrix(
                np.column_stack((right, up, -forward))).as_quat().tolist())
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
            target.update(gap_target_polygon_pixels=np.column_stack((uu, vv)).tolist(),
                          gap_domain='projected_scene_outline_hypothesis',
                          target_extent_basis='source_gap_box_at_neighbouring_depth_estimate')
            result.append(target)
            targets.append((focus, radius))
    return result
