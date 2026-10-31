"""What each uncovered pixel's ray reaches first in the space the frames observed.

The completion gap survey projects retained depth into each candidate camera and
marks uncovered pixels as missing surface. In capture-447ae4f4 (2026-10-30) that
over-counted at both selected gap cameras:

- At a magnified viewpoint the 1-px projection left raster cracks: 10,897 of
  one camera's 75,610 selected gap pixels were surfaces the frames ranged.
- The dining chairs' wooden frames had no confident depth in any of the 16
  frames. 87% and 40% of the selected gap pixels looked through them, and
  generation drew bare floor there.

``observed_space`` carves one voxel grid over the observed scene bounds from the
frames' confident depth. A voxel in front of a ranged pixel is free; one at its
depth is a ranged surface. A voxel that is neither, but that a photograph saw at
a pixel without confident depth, is unranged; anything else was never seen.

``first_reached`` walks each pixel's ray through free voxels and reports what it
meets first. A ray that first meets a ranged surface shows a raster crack; one
that first meets unranged space looks through something a photograph saw; one
that first meets never-seen space shows a missing surface. These are planning
estimates at the grid's resolution, never geometry.
"""
from __future__ import annotations

import hashlib

import numpy as np

SCHEMA_VERSION = "atlas_observed_space.v1"
# The grid spends this many voxels on the observed scene bounds, whatever their
# size: about 3.4 cm voxels for capture-447ae4f4's 7.6 x 2.8 x 7.5 room. It is
# never finer than the frames sample the scene: at most one voxel per pixel.
OBSERVED_SPACE_VOXELS = 2 ** 22
FREE, RANGED, UNRANGED, UNSEEN = 0, 1, 2, 3
# ``first_reached`` codes: the three non-free states, plus a ray that crosses
# only free space until it leaves the observed bounds. Pixels not asked about are 255.
LEFT_SCENE, NOT_WALKED = 4, 255
REACHED_NAMES = {RANGED: "ranged_surface", UNRANGED: "photographed_without_depth",
                 UNSEEN: "never_seen", LEFT_SCENE: "left_observed_bounds"}
_CHUNK_VOXELS = 1 << 20
_RAY_BATCH = 1 << 15
_RAY_STEPS = 32
# The last carved grid; the survey asks for the same frames' space several times.
_SPACE = {}


def _frames_key(photographs, geometry_frames):
    from .atlas_scene_preservation import _frames_digest

    return hashlib.sha256((_frames_digest(photographs) + _frames_digest(geometry_frames)).encode()).hexdigest()


def observed_space(*, photographs, geometry_frames):
    """Carve the observed scene bounds into free, ranged, unranged and never-seen voxels.

    ``geometry_frames`` carve free space and ranged surfaces from confident
    depth; only ``photographs`` (the real frames) mark space seen without depth.
    A ranged surface wins over another frame's free ray, so a grazing view
    cannot carve away a surface that a frame ranged.
    """
    from .atlas_depth_splat import MAX_MEDIAN_CROSS_VIEW_RESIDUAL
    from .atlas_scene_preservation import _project, _valid, observed_scene_bounds

    key = _frames_key(photographs, geometry_frames)
    if key in _SPACE:
        return _SPACE[key]
    bounds = observed_scene_bounds([*geometry_frames, *photographs])
    low, high = np.asarray(bounds["minimum"], float), np.asarray(bounds["maximum"], float)
    extent = np.maximum(high - low, 1e-6)
    pixels = sum(frame["depth"].size for frame in [*geometry_frames, *photographs])
    size = float(np.cbrt(np.prod(extent) / min(OBSERVED_SPACE_VOXELS, pixels)))
    shape = np.maximum(np.ceil(extent / size).astype(int), 1)
    # A voxel centre may sit half a diagonal from the surface it stands for.
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
    state = state.reshape(shape)
    space = {"schema_version": SCHEMA_VERSION, "minimum": low, "maximum": high, "voxel_size": size,
             "state": state}
    _SPACE.clear()
    _SPACE[key] = space
    return space


def space_summary(space):
    """The grid's parameters and voxel counts, for the survey report."""
    counts = np.bincount(space["state"].reshape(-1), minlength=4)
    return {"schema_version": space["schema_version"], "voxel_size": space["voxel_size"],
            "grid_shape": [int(n) for n in space["state"].shape],
            "voxel_counts": {"free": int(counts[FREE]), "ranged_surface": int(counts[RANGED]),
                             "photographed_without_depth": int(counts[UNRANGED]),
                             "never_seen": int(counts[UNSEEN])},
            "basis": "frame_confident_depth_carving_at_voxel_centres", "claim_ceiling": "development_only"}


def _march(space, origin, rays, start, far, stops):
    """Each ray's first sample in a ``stops`` state, every half voxel from ``start``: (state, distance).

    A ray that meets none before ``far`` reports ``LEFT_SCENE`` at ``far``.
    """
    size, state = space["voxel_size"], space["state"]
    shape = np.asarray(state.shape)
    flat = state.reshape(-1)
    stopping = np.zeros(256, bool)
    stopping[list(stops)] = True
    stopping = stopping[flat]
    # Voxel coordinates: the ray's start corner and its direction per unit distance.
    corner = (np.asarray(origin, float) - space["minimum"]) / size
    unit = rays / size
    strides = np.array([shape[1] * shape[2], shape[2], 1])
    reached = np.full(len(rays), LEFT_SCENE, np.uint8)
    distance = np.array(far, float)
    position = np.array(start, float)
    step = size / 2
    steps = np.arange(_RAY_STEPS) * step
    for first in range(0, len(rays), _RAY_BATCH):
        active = np.arange(first, min(first + _RAY_BATCH, len(rays)))
        active = active[position[active] < far[active]]
        while len(active):
            sampled = position[active, None] + steps
            within = sampled < far[active, None]
            index = np.zeros(sampled.shape, np.int64)
            for axis in range(3):
                cell = np.floor(corner[axis] + sampled * unit[active, axis, None]).astype(np.int64)
                np.clip(cell, 0, shape[axis] - 1, out=cell)
                index += cell * strides[axis]
            hit = within & stopping[index]
            met = hit.any(axis=1)
            column = np.argmax(hit[met], axis=1)
            reached[active[met]] = flat[index[met, column]]
            distance[active[met]] = sampled[met, column]
            position[active] += _RAY_STEPS * step
            active = active[~met & within[:, -1]]
    return reached, distance


def first_reached(space, camera, mask):
    """For each pixel in ``mask``, what its ray meets past free space (``NOT_WALKED`` elsewhere).

    The walk starts one voxel from the camera, so the camera's own voxel never
    stops it. A pixel is ``UNRANGED`` only when photographed-without-depth space
    stands clearly in front of the surface its ray reaches next (a ranged
    surface, never-seen space or the observed bounds): a photograph's depthless
    ray runs on behind the object it saw, so space just at that surface is the
    surface itself, and the pixel then counts as that surface's state
    (``LEFT_SCENE`` at the bounds). "Clearly" is the cross-view depth tolerance.
    """
    from .atlas_depth_splat import MAX_MEDIAN_CROSS_VIEW_RESIDUAL, _rotation_matrix

    k, e = camera["intrinsics"], camera["extrinsics"]
    result = np.full(mask.shape, NOT_WALKED, np.uint8)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return result
    low, high, size = space["minimum"], space["maximum"], space["voxel_size"]
    origin = np.asarray(e["position"], float)
    local = np.column_stack(((xs + .5 - k["cx"]) / k["fx"], -(ys + .5 - k["cy"]) / k["fy"], -np.ones(len(xs))))
    rays = local @ _rotation_matrix(e["quaternion"]).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        first, second = (low - origin) / rays, (high - origin) / rays
    near = np.nanmax(np.minimum(first, second), axis=1)
    far = np.nanmin(np.maximum(first, second), axis=1)
    # No frame sees the space around a camera centre: it lies beside every
    # frustum's apex. Unseen space before the ray first enters observed space
    # is the camera's own surroundings; a ray that never enters it is unseen.
    begin = np.maximum(near, size)
    entered, start = _march(space, origin, rays, begin, far, (FREE, RANGED, UNRANGED))
    reached = np.where(begin < far, UNSEEN, LEFT_SCENE).astype(np.uint8)
    distance = np.array(far, float)
    walked = np.flatnonzero(entered != LEFT_SCENE)
    reached[walked], distance[walked] = _march(space, origin, rays[walked], start[walked], far[walked],
                                               (RANGED, UNRANGED, UNSEEN))
    behind = np.flatnonzero(reached == UNRANGED)
    if len(behind):
        surface, at = _march(space, origin, rays[behind], distance[behind], far[behind], (RANGED, UNSEEN))
        tolerance = np.maximum(MAX_MEDIAN_CROSS_VIEW_RESIDUAL * at, size * np.sqrt(3) / 2)
        at_surface = at - distance[behind] <= tolerance
        reached[behind[at_surface]] = surface[at_surface]
    result[ys, xs] = reached
    return result
