"""Whole frames for generated gap views that keep what the capture observed.

World Labs' ``atlasGenerate`` takes posed photographs, their depth and target
cameras; it documents no keep or mask input, so every returned view is a whole
new image. Completion's preservation step pastes back only cross-view-supported
context, and everything else in the frame is the provider's. In capture-447ae4f4
(2026-10-30) both gap cameras sat over the dining table, 1.8 units past the
nearest recorded camera. Preservation kept 42% and 32% of their pixels. Where
the capture had observed the scene, the reviewed frames differed from it by a
mean of 30 and 24 (of 255) per channel, and the whole-view review refused both
for missing dining chairs, so their filled floor never reached the refined splat.

Before that review, each generated view an earlier review admitted only in part
is rebuilt (``composite_view``). Wherever a real frame observed the scene, its
pixel and depth are the real frames rendered into that camera. Generated pixels
remain only in the admitted region, blended into the observed pixels over
``FEATHER_PIXELS`` at its edge, and where no real frame observed anything. The provider outputs
and the completion receipts are unchanged. The render reuses the real frames
already posted whole in the same request, so nothing new is disclosed or bought.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .atlas_depth_splat import EDGE_RELATIVE_JUMP

SCHEMA_VERSION = "atlas_gap_composite.v1"
# Recorded with the whole-view review that judged these candidates. A review
# recorded without it judged the provider's whole frames and keeps them.
CANDIDATE_BASIS = "observed_render_outside_admitted_region.v1"
# The admitted fill blends into observed pixels over this many pixels at its
# edge; every other observed pixel is exact.
FEATHER_PIXELS = 4
# A reference pixel's square splat spans the distance to its neighbours in the
# target camera, up to this radius: at capture-447's gap cameras it covers 99.5%
# of footprints, so magnified or newly face-on surfaces render without cracks.
MAX_SPLAT_RADIUS = 8
# Another reference replaces a nearer camera's pixel only if clearly in front of it.
OCCLUSION_MARGIN = EDGE_RELATIVE_JUMP


def _digest(value) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _interior(frame) -> np.ndarray:
    """Valid depth away from discontinuities: edge pixels splat across the gap they bound."""
    from .atlas_scene_preservation import _valid

    valid = _valid(frame)
    depth = frame["depth"]
    center = depth[1:-1, 1:-1]
    inner = valid[1:-1, 1:-1].copy()
    for neighbour in (depth[1:-1, 2:], depth[1:-1, :-2], depth[2:, 1:-1], depth[:-2, 1:-1]):
        inner &= np.isfinite(neighbour) & (np.abs(neighbour - center) <= EDGE_RELATIVE_JUMP * center)
    result = np.zeros(valid.shape, bool)
    result[1:-1, 1:-1] = inner
    return result


def _splat_depth(frame, camera, height, width, minimum_region):
    """One reference's nearest depth in ``camera`` (inf unknown), each pixel splatted over its footprint.

    The footprint is how far its neighbouring pixels land in the target camera,
    so a surface seen obliquely and now seen face-on renders without cracks;
    remaining cracks fill from the same reference, never from another one.
    """
    from .atlas_scene_preservation import _project, _world

    zbuffer = np.full(height * width, np.inf)
    interior = _interior(frame)
    points, y, x = _world(frame, interior)
    if not len(points):
        return zbuffer
    z, u, v, inside = _project(points, camera)
    grid_u, grid_v = np.full(interior.shape, np.nan), np.full(interior.shape, np.nan)
    ahead = z > 0
    grid_u[y[ahead], x[ahead]], grid_v[y[ahead], x[ahead]] = u[ahead], v[ahead]
    footprint = np.zeros(interior.shape)
    for axis in (0, 1):
        step = np.fmax(np.abs(np.diff(grid_u, axis=axis)), np.abs(np.diff(grid_v, axis=axis)))
        for pad in ((0, 1), (1, 0)):  # Both pixels sharing a neighbour step.
            footprint = np.fmax(footprint, np.pad(step, [pad if a == axis else (0, 0) for a in (0, 1)],
                                                  constant_values=np.nan))
    keep = np.flatnonzero(inside)
    radius = np.clip(np.ceil((footprint[y[keep], x[keep]] - 1) / 2), 0, MAX_SPLAT_RADIUS).astype(int)
    z, rows, cols = z[keep], np.floor(v[keep]).astype(int), np.floor(u[keep]).astype(int)
    for size in np.unique(radius):
        chosen = np.flatnonzero(radius == size)
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                rr, cc = rows[chosen] + dy, cols[chosen] + dx
                ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                np.minimum.at(zbuffer, rr[ok] * width + cc[ok], z[chosen[ok]])
    return _fill_cracks(zbuffer.reshape(height, width), minimum_region).reshape(-1)


def _fill_cracks(depth, minimum_region):
    """Fill unobserved holes enclosed by observed pixels from their nearest observed pixel.

    Holes smaller than the smallest gap the survey names (``gap_labels``) are
    raster cracks between splats, not missing surfaces.
    """
    import cv2
    from scipy.ndimage import distance_transform_edt

    observed = np.isfinite(depth)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~observed).astype(np.uint8), connectivity=4)
    border = np.zeros(count, bool)
    border[np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))] = True
    cracks = (stats[:, cv2.CC_STAT_AREA] < minimum_region) & ~border
    cracks[0] = False  # Label 0 is the observed pixels themselves.
    fill = cracks[labels] & ~observed
    if fill.any():
        _, (iy, ix) = distance_transform_edt(~observed, return_indices=True)
        depth[fill] = depth[iy[fill], ix[fill]]
    return depth


def _drop_specks(observed, minimum_region):
    """Isolated observed fragments smaller than a named gap are noise, not context."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(observed.astype(np.uint8), connectivity=8)
    specks = stats[:, cv2.CC_STAT_AREA] < minimum_region
    specks[0] = False
    return observed & ~specks[labels]


def _sample(references, owner, depth, observed, camera):
    """Each observed pixel's colour, sampled bilinearly from the reference that rendered it."""
    from scipy.ndimage import map_coordinates
    from .atlas_depth_splat import _rotation_matrix
    from .atlas_scene_preservation import _project, _world

    target = {"intrinsics": camera["intrinsics"], "depth": depth,
              "rotation": _rotation_matrix(camera["extrinsics"]["quaternion"]),
              "position": np.asarray(camera["extrinsics"]["position"], float)}
    rgb = np.zeros((*depth.shape, 3), np.uint8)
    for index in np.unique(owner[observed]):
        frame = references[index]
        points, y, x = _world(target, observed & (owner == index))
        _, u, v, _ = _project(points, frame)
        # Pixel i spans [i, i + 1); its sample sits at its centre.
        coordinates = np.stack([v - .5, u - .5])
        rgb[y, x] = np.stack([map_coordinates(frame["rgb"][..., c], coordinates, order=1, mode="nearest")
                              for c in range(3)], axis=-1)
    return rgb


def observation_references(frames):
    """Real frames with the draft fusion's terrace-free depth (``smoothed_depth``), prepared once.

    Atlas depth arrives in plateaus of a few pixels; rendered close up from a new
    camera, raw plateaus become stripes across a surface the capture saw whole.
    """
    from .atlas_depth_splat import smoothed_depth

    return [{**frame, "depth": smoothed_depth(frame["depth"])} for frame in frames]


def render_observed(references, camera):
    """What the real frames observed, rendered into ``camera``: rgb, depth (0 unknown), observed mask, basis.

    ``references`` come from ``observation_references``. At a recorded camera
    centre every photographed ray is exact without depth, so the photographs are
    resampled there (depth where known). Elsewhere each reference's confident,
    non-edge depth is splatted, nearest camera first, a farther reference winning
    a pixel only when clearly in front of it; the rendered depth is smoothed as
    the draft's is, and each pixel's colour is sampled from its reference.
    """
    from .atlas_depth_splat import smoothed_depth
    from .atlas_scene_chisel import project_source_depths
    from .atlas_scene_preservation import _same_center, project_known_contexts

    k = camera["intrinsics"]
    height, width = int(k["height"]), int(k["width"])
    minimum_region = max(9, round(height * width / 4096))
    if any(_same_center(frame, camera) for frame in references):
        rgb, observed = project_known_contexts(references, [camera])[0]
        depth = project_source_depths(references, [camera])[0]
        return rgb, np.where(observed, depth, 0).astype(np.float32), observed, "recorded_centre_photographs"
    target = np.asarray(camera["extrinsics"]["position"], float)
    order = sorted(range(len(references)),
                   key=lambda i: (float(np.linalg.norm(np.asarray(references[i]["position"]) - target)), i))
    zbuffer = np.full(height * width, np.inf)
    owner = np.full(height * width, -1)
    for index in order:
        layer = _splat_depth(references[index], camera, height, width, minimum_region)
        take = np.isfinite(layer) & (~np.isfinite(zbuffer) | (layer < zbuffer * (1 - OCCLUSION_MARGIN)))
        zbuffer[take], owner[take] = layer[take], index
    owner = owner.reshape(height, width)
    observed = _drop_specks(np.isfinite(zbuffer).reshape(height, width), minimum_region)
    depth = smoothed_depth(np.where(observed, zbuffer.reshape(height, width), 0).astype(np.float32))
    return _sample(references, owner, depth, observed, camera), depth, observed, "recorded_depth_splats"


def composite_view(view, references):
    """The view with observed pixels and depth wherever a reference observed; the provider's elsewhere.

    ``view["keep_mask"]`` is the region an earlier review admitted; generated
    pixels always stay there. Unobserved pixels outside it keep the provider's
    pixels, which no observation contradicts. Admission, confidence and camera
    are unchanged. Returns the view and a receipt of what each pixel came from.
    """
    from scipy.ndimage import distance_transform_edt

    keep = np.asarray(view["keep_mask"], bool)
    camera = {"intrinsics": view["intrinsics"], "extrinsics": view["extrinsics"]}
    rgb, depth, observed, basis = render_observed(references, camera)
    if rgb.shape != view["rgb"].shape or depth.shape != view["depth"].shape:
        raise ValueError("atlas_gap_composite_grid_changed")
    generated = keep | ~observed
    # The fill blends into its observed surroundings; an unobserved fragment
    # beside an observed object must not wash that object out, so it does not.
    alpha = np.clip(1 - distance_transform_edt(~keep) / (FEATHER_PIXELS + 1), 0, 1)
    alpha = np.where(generated, 1.0, alpha)[..., None]
    pixels = np.rint(alpha * view["rgb"] + (1 - alpha) * rgb).astype(np.uint8)
    # A photographed pixel without measured depth keeps the provider's depth.
    composite_depth = np.where(generated | (depth <= 0), view["depth"], depth).astype(np.float32)
    replaced = observed & ~keep
    receipt = {
        "schema_version": SCHEMA_VERSION, "candidate_basis": CANDIDATE_BASIS, "observed_basis": basis,
        "reference_frame_ids": [frame["frame_id"] for frame in references],
        "total_pixel_count": int(keep.size),
        "observed_pixel_count": int(replaced.sum()),
        "admitted_generated_pixel_count": int(keep.sum()),
        "unobserved_generated_pixel_count": int((~observed & ~keep).sum()),
        "feathered_observed_pixel_count": int((replaced & (alpha[..., 0] > 0)).sum()),
        "feather_pixels": FEATHER_PIXELS,
        # How far the provider's pixels strayed from what was observed where they were replaced.
        "replaced_mean_absolute_difference": (
            float(np.abs(view["rgb"].astype(float) - rgb.astype(float))[replaced].mean()) if replaced.any() else None),
        "provider_rgb_sha256": _digest(view["rgb"]), "rgb_sha256": _digest(pixels),
        "depth_sha256": _digest(composite_depth),
        "generated_content": True, "claim_ceiling": "development_only"}
    return {**view, "rgb": pixels, "depth": composite_depth}, receipt
