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
