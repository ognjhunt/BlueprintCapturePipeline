"""Generated gap views corrected only where they contradict a well-observed capture.

World Labs' ``atlasGenerate`` takes posed photographs, their depth and target
cameras; it documents no keep or mask input, so every returned view is a whole
new image. Completion's preservation step pastes sparse cross-view-supported
context into it (42% and 32% of capture-447ae4f4's two gap views, 2026-10-30),
and that speckled frame is what the whole-view review refused.

Before that review, each generated view an earlier review admitted only in part
is rebuilt (``composite_view``) from the provider's own pixels. The real frames
are rendered into its camera with per-pixel quality; a capture pixel and its
depth replace the provider's only where the capture observed the scene well and
the provider contradicts it beyond what content known to agree shows. The
admitted region, everything the capture saw poorly and everything it never saw
keep the provider's pixels. The provider outputs and completion receipts are
unchanged; the render reuses real frames already posted in the same request, so
nothing new is disclosed or bought.

Thresholds come from capture-447's retained frames (derivation in
``docs/atlas-native-splats.md``). Known to agree: the provider's re-render at the
two recorded centres it was given, against the capture rendered there from
frames at least 0.5 units away. Render quality: four real photographs against
their renders from frames at least 0.4 units away, in equal-count bins of 2,000
pixels; a quality level is good while the render's own q90 relative depth error
stays within the agreement tolerance.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .atlas_depth_splat import EDGE_RELATIVE_JUMP

SCHEMA_VERSION = "atlas_gap_composite.v2"
# Recorded (before any call) with the whole-view review that judges these
# candidates. A root reviewed, or with a review call started, without it keeps
# the provider's whole frames.
CANDIDATE_BASIS = "capture_where_provider_contradicts_well_observed.v1"
# Contradiction tolerances, from known-agree content. Relative depth: q99 of
# the provider-to-capture difference (0.0394 and 0.0305 at the two centres).
AGREE_RELATIVE_DEPTH = 0.0394
# Colour is compared after a Gaussian blur at the median provider-to-capture
# misregistration on that content (optical flow, 1.13 px), as the larger of its
# q99 max-channel difference (78.1) and the render's own q99 error inside the
# quality gate (80.9).
COMPARE_SIGMA_PIXELS = 1.13
AGREE_COLOUR = 80.9
# Quality gate. Stretch: target pixels one reference pixel spans (view-angle and
# distance change); the first bin over tolerance starts at 1.850. Incidence: the
# last bin over tolerance ends at 0.150. Depth confidence: no bin fails; 0.635 is
# the lowest calibrated. Agreeing views: q90 0.069 / 0.041 / 0.032 at 1 / 2 / 3.
MAX_STRETCH = 1.850
MIN_INCIDENCE = 0.150
MIN_CONFIDENCE = 0.635
MIN_VIEWS = 3
# A replaced region blends into the provider's agreeing neighbours over this many pixels.
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


def _incidence(frame) -> np.ndarray:
    """Per source pixel, the cosine between its ray and its surface normal (0 grazing, 1 face-on)."""
    from .atlas_scene_preservation import _world

    points = np.full((*frame["depth"].shape, 3), np.nan)
    valid = np.isfinite(frame["depth"]) & (frame["depth"] > 0)
    world, y, x = _world(frame, valid)
    points[y, x] = world
    across = np.zeros_like(points)
    down = np.zeros_like(points)
    across[:, :-1], down[:-1] = points[:, 1:] - points[:, :-1], points[1:] - points[:-1]
    normal = np.cross(across, down)
    ray = points - np.asarray(frame["position"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.abs((normal * ray).sum(-1)) / (np.linalg.norm(normal, axis=-1) * np.linalg.norm(ray, axis=-1))
    return np.nan_to_num(cosine, nan=0.0)


def _splat_depth(frame, camera, height, width, minimum_region):
    """One reference rendered into ``camera``: nearest depth (inf unknown) and that pixel's quality.

    Each pixel is splatted over its footprint: how far its neighbouring pixels
    land in the target camera, so a surface seen obliquely and now seen face-on
    renders without cracks; remaining cracks fill from the same reference. The
    quality planes are the winning pixel's footprint (its view-angle stretch),
    its depth confidence and its incidence cosine in the reference.
    """
    from .atlas_scene_preservation import _project, _world

    zbuffer = np.full(height * width, np.inf)
    quality = np.zeros((3, height * width))
    interior = _interior(frame)
    points, y, x = _world(frame, interior)
    if not len(points):
        return zbuffer, quality
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
    rows, cols = y[keep], x[keep]
    planes = np.stack([np.maximum(footprint[rows, cols], 1.0), frame["confidence"][rows, cols],
                       _incidence(frame)[rows, cols]])
    radius = np.clip(np.ceil((footprint[rows, cols] - 1) / 2), 0, MAX_SPLAT_RADIUS).astype(int)
    z, rows, cols = z[keep], np.floor(v[keep]).astype(int), np.floor(u[keep]).astype(int)
    passes = []
    for size in np.unique(radius):
        chosen = np.flatnonzero(radius == size)
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                rr, cc = rows[chosen] + dy, cols[chosen] + dx
                ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                passes.append((chosen[ok], rr[ok] * width + cc[ok]))
    for chosen, flat in passes:
        np.minimum.at(zbuffer, flat, z[chosen])
    for chosen, flat in passes:
        nearest = z[chosen] == zbuffer[flat]
        quality[:, flat[nearest]] = planes[:, chosen[nearest]]
    depth, quality = _fill_cracks(zbuffer.reshape(height, width), minimum_region,
                                  quality.reshape(3, height, width))
    return depth.reshape(-1), quality.reshape(3, -1)


def _fill_cracks(depth, minimum_region, extras=None):
    """Fill unobserved holes enclosed by observed pixels from their nearest observed pixel.

    Holes smaller than the smallest gap the survey names (``gap_labels``) are
    raster cracks between splats, not missing surfaces. ``extras`` (planes x
    rows x columns) are filled from the same pixels.
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
        if extras is not None:
            extras[:, fill] = extras[:, iy[fill], ix[fill]]
    return depth, extras


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
    """What the real frames observed, rendered into ``camera``.

    Returns ``rgb``, ``depth`` (0 unknown), ``observed``, ``basis`` and per-pixel
    quality: ``stretch`` (target pixels one reference pixel spans: the view-angle
    and distance change), ``confidence`` (its depth confidence), ``incidence``
    (cosine of its ray to the surface in the reference) and ``views`` (references
    whose depth agrees there). ``references`` come from ``observation_references``.
    At a recorded camera centre every photographed ray is exact without depth,
    so the photographs are resampled there (depth where known). Elsewhere each
    reference's confident, non-edge depth is splatted, nearest camera first, a
    farther reference winning a pixel only when clearly in front of it; the
    rendered depth is smoothed as the draft's is, and each pixel's colour is
    sampled from its reference.
    """
    from .atlas_depth_splat import smoothed_depth
    from .atlas_scene_chisel import project_source_depths
    from .atlas_scene_preservation import _same_center, project_known_contexts

    k = camera["intrinsics"]
    height, width = int(k["height"]), int(k["width"])
    minimum_region = max(9, round(height * width / 4096))
    if any(_same_center(frame, camera) for frame in references):
        rgb, observed = project_known_contexts(references, [camera])[0]
        depth = np.where(observed, project_source_depths(references, [camera])[0], 0).astype(np.float32)
        # A photographed ray at its own centre is exact; only its depth may be unknown.
        same = sum(_same_center(frame, camera) for frame in references)
        return {"rgb": rgb, "depth": depth, "observed": observed, "basis": "recorded_centre_photographs",
                "stretch": np.ones(observed.shape), "confidence": np.where(depth > 0, 1.0, 0.0),
                "incidence": np.ones(observed.shape), "views": np.where(observed, same, 0)}
    target = np.asarray(camera["extrinsics"]["position"], float)
    order = sorted(range(len(references)),
                   key=lambda i: (float(np.linalg.norm(np.asarray(references[i]["position"]) - target)), i))
    zbuffer = np.full(height * width, np.inf)
    owner = np.full(height * width, -1)
    quality = np.zeros((3, height * width))
    views = np.zeros(height * width, int)
    for index in order:
        layer, layer_quality = _splat_depth(references[index], camera, height, width, minimum_region)
        seen = np.isfinite(layer)
        agree = seen & np.isfinite(zbuffer)
        agree[agree] = np.abs(layer[agree] - zbuffer[agree]) <= OCCLUSION_MARGIN * zbuffer[agree]
        take = seen & (~np.isfinite(zbuffer) | (layer < zbuffer * (1 - OCCLUSION_MARGIN)))
        views[agree] += 1
        zbuffer[take], owner[take], quality[:, take], views[take] = layer[take], index, layer_quality[:, take], 1
    owner = owner.reshape(height, width)
    observed = _drop_specks(np.isfinite(zbuffer).reshape(height, width), minimum_region)
    depth = smoothed_depth(np.where(observed, zbuffer.reshape(height, width), 0).astype(np.float32))
    stretch, confidence, incidence = (np.where(observed, plane.reshape(height, width), 0) for plane in quality)
    return {"rgb": _sample(references, owner, depth, observed, camera), "depth": depth, "observed": observed,
            "basis": "recorded_depth_splats", "stretch": stretch, "confidence": confidence,
            "incidence": incidence, "views": np.where(observed, views.reshape(height, width), 0)}


def well_observed(rendered):
    """Pixels whose capture render is as faithful as content known to agree."""
    if rendered["basis"] == "recorded_centre_photographs":
        return rendered["observed"].copy()  # A photographed ray at its own centre is exact.
    return (rendered["observed"] & (rendered["depth"] > 0) & (rendered["stretch"] < MAX_STRETCH)
            & (rendered["incidence"] >= MIN_INCIDENCE) & (rendered["confidence"] >= MIN_CONFIDENCE)
            & (rendered["views"] >= MIN_VIEWS))


def composite_view(view, references):
    """The provider's view, with the capture's pixel and depth wherever it contradicts a well-observed capture.

    The provider's pixels are ``view["provider_rgb"]`` (its own image, before
    completion pasted context into it), else ``view["rgb"]``. ``view["keep_mask"]``
    is the region an earlier review admitted and always stays the provider's.
    Contradiction: relative depth beyond ``AGREE_RELATIVE_DEPTH`` where both have
    depth, or blurred colour beyond ``AGREE_COLOUR``. Isolated contradictions
    smaller than a named gap are noise. Admission, confidence and camera are
    unchanged. Returns the view and a receipt of what each pixel came from.
    """
    import cv2
    from scipy.ndimage import distance_transform_edt

    keep = np.asarray(view["keep_mask"], bool)
    provider = np.asarray(view.get("provider_rgb", view["rgb"]))
    camera = {"intrinsics": view["intrinsics"], "extrinsics": view["extrinsics"]}
    rendered = render_observed(references, camera)
    rgb, depth, observed = rendered["rgb"], rendered["depth"], rendered["observed"]
    if rgb.shape != provider.shape or rgb.shape != view["rgb"].shape or depth.shape != view["depth"].shape:
        raise ValueError("atlas_gap_composite_grid_changed")
    well = well_observed(rendered) & ~keep
    provider_depth = np.asarray(view["depth"], np.float32)
    measured = well & np.isfinite(provider_depth) & (provider_depth > 0) & (depth > 0)
    depth_contradiction = np.zeros(keep.shape, bool)
    depth_contradiction[measured] = (np.abs(provider_depth[measured] - depth[measured])
                                     > AGREE_RELATIVE_DEPTH * depth[measured])
    blurred = [cv2.GaussianBlur(np.asarray(image, np.float32), (0, 0), COMPARE_SIGMA_PIXELS) for image in (provider, rgb)]
    colour_contradiction = well & (np.abs(blurred[0] - blurred[1]).max(-1) > AGREE_COLOUR)
    height, width = keep.shape
    replace = _drop_specks(depth_contradiction | colour_contradiction, max(9, round(height * width / 4096)))
    alpha = np.zeros(keep.shape)
    if replace.any():
        alpha = np.clip(1 - distance_transform_edt(~replace) / (FEATHER_PIXELS + 1), 0, 1)
        alpha[~observed | keep] = 0  # Blend only into observed pixels, never into the admitted fill.
    pixels = np.rint(alpha[..., None] * rgb + (1 - alpha[..., None]) * provider).astype(np.uint8)
    # A photographed pixel without measured depth keeps the provider's depth.
    composite_depth = np.where(replace & (depth > 0), depth, provider_depth).astype(np.float32)
    receipt = {
        "schema_version": SCHEMA_VERSION, "candidate_basis": CANDIDATE_BASIS, "observed_basis": rendered["basis"],
        "reference_frame_ids": [frame["frame_id"] for frame in references],
        "thresholds": {"agree_relative_depth": AGREE_RELATIVE_DEPTH, "agree_colour": AGREE_COLOUR,
                       "compare_sigma_pixels": COMPARE_SIGMA_PIXELS, "max_stretch": MAX_STRETCH,
                       "min_incidence": MIN_INCIDENCE, "min_confidence": MIN_CONFIDENCE, "min_views": MIN_VIEWS,
                       "feather_pixels": FEATHER_PIXELS, "basis": "capture-447ae4f4 retained frames, 2026-10-31"},
        "total_pixel_count": int(keep.size),
        "observed_pixel_count": int((observed & ~keep).sum()),
        "well_observed_pixel_count": int(well.sum()),
        "depth_contradiction_pixel_count": int(depth_contradiction.sum()),
        "colour_contradiction_pixel_count": int(colour_contradiction.sum()),
        "replaced_pixel_count": int(replace.sum()),
        "feathered_pixel_count": int(((alpha > 0) & ~replace).sum()),
        "admitted_generated_pixel_count": int(keep.sum()),
        "provider_pixel_count": int((alpha == 0).sum()),
        # How far the provider's pixels strayed from the capture where they were replaced.
        "replaced_mean_absolute_difference": (
            float(np.abs(provider.astype(float) - rgb.astype(float))[replace].mean()) if replace.any() else None),
        "provider_rgb_sha256": _digest(provider), "reviewed_input_rgb_sha256": _digest(view["rgb"]),
        "rgb_sha256": _digest(pixels), "depth_sha256": _digest(composite_depth),
        "generated_content": True, "claim_ceiling": "development_only"}
    return {**view, "rgb": pixels, "depth": composite_depth}, receipt
