"""Where the filmed views will leave a refined splat weak, worked out before the splat is bought.

itest-c0cbbf20 (2026-11-18) came back blurry wherever the viewer left the filmed
line. Its views saw the room from about one direction: the table-top items had
85 depth-confirmed views from a 2.9-3.2 degree spread at 2-3 m, the shelf 17
views over 1.4 degrees. The table top had 6.4 degrees and looked clean, because
it is flat: a flat surface needs no parallax to place, a cluttered one does.

``coverage_map`` reads the posed views' confident depth on the observed-space
grid (``atlas_observed_space.observed_grid``: the same voxels) and records, for
every voxel a view ranged:

- how many posed views range it (seen, at its depth, within the cross-view tolerance);
- the angle spread of those views (the widest angle between two of their
  directions to the voxel, measured in eight directions of the plane across
  their mean direction);
- millimetres per pixel at its nearest view (its depth over the focal length,
  in metres when the scene measurement gives a scale);
- the best local sharpness among those views (Laplacian variance in a
  ``SHARPNESS_WINDOW`` px window at the posed image's resolution);
- its local flatness: the surface variation (smallest over summed eigenvalue)
  of each view's own depth around the voxel's pixel (``_planarity``; 0 is a
  plane), and whether at least half the views found it flat.

A voxel is weak when its surface is not flat and its views either spread less
than ``NARROW_SPREAD_DEGREES`` or resolve it no finer than ``COARSE_MM_PER_PIXEL``.
Weak voxels close to one another form a region; each region is recorded with
its centroid, extent, the views that saw it and an orbit of new viewpoints
(``ORBIT_DEGREES`` either side of where it was filmed from, about the world's
up axis) for generated views to fill. Regions the job's route views saw come
first, then the nearest (finest pixels). The thresholds are planning choices,
not measurements of what a viewer will see.

It buys nothing, never stops a job and runs once per pose: the record is
``weak_regions.json`` in the Atlas job root (``pipeline/atlas`` of a capture),
digest-sealed, with its per-voxel arrays beside it (``atlas_coverage_map.npz``,
bound by sha256). ``load_weak_regions`` is the reader for other stages.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "atlas_weak_regions.v1"
REVISION = "voxel-views-spread-resolution-flatness-v1"
WEAK_REGIONS_FILE = "weak_regions.json"
ARRAYS_FILE = "atlas_coverage_map.npz"
#: Depth pixels sampled for the surface, every this many rows and columns.
SURFACE_STRIDE = 2
#: A voxel is a ranged surface when at least this many sampled depth pixels land in it.
MIN_SURFACE_SAMPLES = 3
#: Each view's flatness is judged on its depth sampled every ``FLATNESS_STEP`` pixels, ``FLATNESS_WINDOW`` samples wide.
FLATNESS_STEP = 4
FLATNESS_WINDOW = 7
#: Surface variation up to this is flat; a plane scores about 0, an isotropic clutter 1/3.
FLAT_MAX_SURFACE_VARIATION = 0.02
NARROW_SPREAD_DEGREES = 10.0
COARSE_MM_PER_PIXEL = 3.0
SHARPNESS_WINDOW = 9
#: A view saw a region when it ranged at least this share of the region's voxels.
SEEN_FRACTION = 0.3
MIN_REGION_VOXELS = 6
MAX_REGIONS = 32
#: Regions wider than this (metres) are split on a grid of this cell size, so an orbit has one centre.
REGION_CELL_M = 0.6
ORBIT_DEGREES = (15, 30)
#: Orbits turn about the Atlas world's up axis (its splat manifests record ``atlas_world_rub_y_up``).
WORLD_UP = (0.0, 1.0, 0.0)
_DIRECTIONS = 8


def limits() -> dict[str, Any]:
    return {"surface_stride_pixels": SURFACE_STRIDE, "min_surface_samples": MIN_SURFACE_SAMPLES,
            "flatness_step_pixels": FLATNESS_STEP, "flatness_window_samples": FLATNESS_WINDOW,
            "flat_max_surface_variation": FLAT_MAX_SURFACE_VARIATION,
            "narrow_spread_degrees": NARROW_SPREAD_DEGREES, "coarse_mm_per_pixel": COARSE_MM_PER_PIXEL,
            "sharpness_window_pixels": SHARPNESS_WINDOW, "seen_fraction": SEEN_FRACTION,
            "min_region_voxels": MIN_REGION_VOXELS, "max_regions": MAX_REGIONS, "region_cell_m": REGION_CELL_M,
            "orbit_degrees": list(ORBIT_DEGREES)}


def _local_sharpness(rgb) -> np.ndarray:
    import cv2

    grey = np.asarray(rgb, np.float32).mean(axis=2)
    laplacian = cv2.Laplacian(grey, cv2.CV_32F)
    window = (SHARPNESS_WINDOW, SHARPNESS_WINDOW)
    mean = cv2.boxFilter(laplacian, -1, window)
    return np.maximum(cv2.boxFilter(laplacian * laplacian, -1, window) - mean * mean, 0.0)


def _surface(frames, low, size, shape):
    """Linear indices of ranged voxels (sorted) and the centroid of the depth samples in each."""
    from .atlas_depth_splat import decoded_frame
    from .atlas_scene_preservation import _valid, _world

    count = int(np.prod(shape))
    samples = np.zeros(count, np.float64)
    sums = np.zeros((3, count), np.float64)
    for frame in frames:
        frame = decoded_frame(frame, "depth", "confidence")
        valid = _valid(frame)
        stride = np.zeros_like(valid)
        stride[::SURFACE_STRIDE, ::SURFACE_STRIDE] = True
        points, _, _ = _world(frame, valid & stride)
        cells = np.floor((points - low) / size).astype(np.int64)
        inside = np.all((cells >= 0) & (cells < shape), axis=1)
        linear = np.ravel_multi_index(cells[inside].T, shape)
        samples += np.bincount(linear, minlength=count)
        for axis in range(3):
            sums[axis] += np.bincount(linear, weights=points[inside, axis], minlength=count)
    surface = np.flatnonzero(samples >= MIN_SURFACE_SAMPLES)
    return surface, (sums[:, surface] / samples[surface]).T


def _planarity(frame, valid) -> np.ndarray:
    """Surface variation of one view's own depth over ``FLATNESS_WINDOW`` cells of a 1/``FLATNESS_STEP`` grid.

    A view's depth is smooth along the surface it saw; different views disagree
    by a few per cent of the depth, so a neighbourhood taken across views would
    thicken every surface into a slab. NaN where under a quarter of the cells have depth.
    """
    import cv2

    k = frame["intrinsics"]
    step = FLATNESS_STEP
    depth = np.where(valid, frame["depth"], 0.0)[::step, ::step].astype(np.float64)
    known = (depth > 0).astype(np.float64)
    rows, cols = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    axes = ((cols * step + .5 - k["cx"]) / k["fx"] * depth, -(rows * step + .5 - k["cy"]) / k["fy"] * depth, -depth)
    window = (FLATNESS_WINDOW, FLATNESS_WINDOW)

    def box(value):
        return cv2.boxFilter(value * known, cv2.CV_64F, window, normalize=False, borderType=cv2.BORDER_CONSTANT)
    n = np.maximum(box(np.ones_like(depth)), 1.0)
    mean = [box(a) / n for a in axes]
    covariance = np.empty((*depth.shape, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            covariance[..., i, j] = covariance[..., j, i] = box(axes[i] * axes[j]) / n - mean[i] * mean[j]
    values = np.clip(np.linalg.eigvalsh(covariance.reshape(-1, 3, 3)), 0, None)
    variation = (values[:, 0] / np.maximum(values.sum(axis=1), 1e-18)).reshape(depth.shape)
    return np.where(n >= FLATNESS_WINDOW * FLATNESS_WINDOW / 4, variation, np.nan)


def _sightings(frames, centroids, size):
    """Per voxel, streamed over the views that range it: count, direction sum, finest pixel, sharpness, flatness.

    Also returns each view's ranged voxels (unique within a view) for the spread pass.
    """
    from .atlas_depth_splat import MAX_MEDIAN_CROSS_VIEW_RESIDUAL, decoded_frame
    from .atlas_scene_preservation import _project, _valid

    n = len(centroids)
    half_diagonal = size * math.sqrt(3) / 2
    stats = {"count": np.zeros(n, np.int32), "direction": np.zeros((n, 3)), "finest": np.full(n, np.inf),
             "sharpness": np.full(n, np.nan, np.float32), "flat_votes": np.zeros(n, np.int32),
             "flat_known": np.zeros(n, np.int32), "flatness_sum": np.zeros(n)}
    seen_by_view = []
    for frame in frames:
        frame = decoded_frame(frame, *(("depth", "confidence", "rgb") if "rgb" in frame else ("depth", "confidence")))
        z, u, v, inside = _project(centroids, frame)
        candidates = np.flatnonzero(inside)
        rows, cols = np.floor(v[candidates]).astype(int), np.floor(u[candidates]).astype(int)
        valid = _valid(frame)
        depth = frame["depth"][rows, cols]
        tolerance = np.maximum(MAX_MEDIAN_CROSS_VIEW_RESIDUAL * depth, half_diagonal)
        seen = valid[rows, cols] & (np.abs(z[candidates] - depth) <= tolerance)
        index, rows, cols = candidates[seen], rows[seen], cols[seen]
        seen_by_view.append(index.astype(np.int32))
        stats["count"][index] += 1
        toward = np.asarray(frame["position"], float) - centroids[index]
        stats["direction"][index] += toward / np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)
        stats["finest"][index] = np.minimum(stats["finest"][index], z[index] / float(frame["intrinsics"]["fx"]))
        if "rgb" in frame:
            stats["sharpness"][index] = np.fmax(stats["sharpness"][index], _local_sharpness(frame["rgb"])[rows, cols])
        grid = _planarity(frame, valid)
        flat = grid[np.minimum(rows // FLATNESS_STEP, grid.shape[0] - 1), np.minimum(cols // FLATNESS_STEP,
                                                                                      grid.shape[1] - 1)]
        known = np.isfinite(flat)
        stats["flat_known"][index[known]] += 1
        stats["flat_votes"][index[known & (flat <= FLAT_MAX_SURFACE_VARIATION)]] += 1
        stats["flatness_sum"][index[known]] += flat[known]
    return stats, seen_by_view


def _spread_degrees(seen_by_view, centroids, positions, mean):
    """The widest angle between two views of each voxel, from their extents across their mean direction."""
    n = len(centroids)
    mean = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    helper = np.where(np.abs(mean[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    first = np.cross(mean, helper)
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = np.cross(mean, first)
    angles = np.pi * np.arange(_DIRECTIONS) / _DIRECTIONS
    high = np.full((_DIRECTIONS, n), -np.inf, np.float32)
    low = np.full((_DIRECTIONS, n), np.inf, np.float32)
    for position, index in zip(positions, seen_by_view):
        toward = position - centroids[index]
        toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)
        a = np.einsum("ij,ij->i", toward, first[index])
        b = np.einsum("ij,ij->i", toward, second[index])
        p = (np.cos(angles)[:, None] * a + np.sin(angles)[:, None] * b).astype(np.float32)
        high[:, index] = np.maximum(high[:, index], p)
        low[:, index] = np.minimum(low[:, index], p)
    widest = np.where(np.isfinite(high), high - low, 0.0).max(axis=0)
    return np.degrees(2 * np.arcsin(np.clip(widest / 2, 0, 1)))


def _look_at(position, target, up):
    """(x, y, z, w) quaternion of a RUB camera at ``position`` looking at ``target`` with ``up`` up."""
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, [1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    m = np.column_stack([right, true_up, -forward])
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i], q[j], q[k] = s / 4, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return [float(value) for value in q]


def _rotate(vector, axis, degrees):
    """Rodrigues rotation of ``vector`` about unit ``axis``."""
    t = math.radians(degrees)
    return (vector * math.cos(t) + np.cross(axis, vector) * math.sin(t)
            + axis * float(np.dot(axis, vector)) * (1 - math.cos(t)))


def _orbit(region_id, centroid, positions, intrinsics, up):
    """Cameras ``ORBIT_DEGREES`` either side of the views that saw the region, at their median distance."""
    toward = positions - centroid
    distance = float(np.median(np.linalg.norm(toward, axis=1)))
    mean = toward.mean(axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-12)
    cameras = []
    for degrees in [-ORBIT_DEGREES[1], -ORBIT_DEGREES[0], ORBIT_DEGREES[0], ORBIT_DEGREES[1]]:
        position = centroid + _rotate(mean, up, degrees) * distance
        cameras.append({"camera_id": f"{region_id}-orbit{degrees:+d}", "orbit_degrees": degrees,
                        "intrinsics": dict(intrinsics),
                        "extrinsics": {"coordinateSystem": "rub", "position": [round(float(x), 6) for x in position],
                                       "quaternion": _look_at(position, centroid, up)},
                        "look_at": [round(float(x), 6) for x in centroid]})
    return {"centre": [round(float(x), 6) for x in centroid], "radius_units": round(distance, 6),
            "filmed_direction": [round(float(x), 6) for x in mean], "axis": [round(float(x), 6) for x in up],
            "cameras": cameras}


def _max_pair_angle(directions) -> float:
    if len(directions) < 2:
        return 0.0
    cosine = np.clip(directions @ directions.T, -1, 1)
    return float(np.degrees(np.arccos(cosine.min())))


def _regions(weak_index, surface, centroids, shape, size, metres):
    """Weak voxels grouped by touching (one voxel apart), split on ``REGION_CELL_M`` cells when wider."""
    from scipy import ndimage

    if not len(weak_index):
        return []
    grid = np.zeros(int(np.prod(shape)), bool)
    grid[surface[weak_index]] = True
    grid = grid.reshape(shape)
    labels, _ = ndimage.label(ndimage.binary_dilation(grid, structure=np.ones((3, 3, 3), bool)),
                              structure=np.ones((3, 3, 3), bool))
    label = labels.reshape(-1)[surface[weak_index]].astype(np.int64)
    cell = max(REGION_CELL_M / metres, size)
    groups = []
    for value in np.unique(label):
        members = weak_index[label == value]
        points = centroids[members]
        if float((points.max(axis=0) - points.min(axis=0)).max()) * metres > REGION_CELL_M:
            keys = np.floor((points - points.min(axis=0)) / cell).astype(np.int64)
            _, part = np.unique(keys, axis=0, return_inverse=True)
            groups += [members[part.reshape(-1) == p] for p in np.unique(part)]
        else:
            groups.append(members)
    return sorted((g for g in groups if len(g) >= MIN_REGION_VOXELS), key=lambda g: (-len(g), int(g[0])))


def coverage_map(frames: Sequence[Mapping[str, Any]], *, meters_per_unit: float | None = None,
                 priority_frame_ids: Sequence[str] = ()) -> dict[str, Any]:
    """The per-voxel coverage arrays, the weak regions and a summary; reads each view's depth three times."""
    from .atlas_observed_space import observed_grid

    metres = float(meters_per_unit) if meters_per_unit and meters_per_unit > 0 else 1.0
    low, _, size, shape = observed_grid(frames)
    surface, centroids = _surface(frames, low, size, shape)
    if not len(surface):
        raise ValueError("atlas_coverage_no_ranged_surface")
    positions = np.array([np.asarray(frame["position"], float) for frame in frames])
    stats, seen_by_view = _sightings(frames, centroids, size)
    count = stats["count"]
    spread = _spread_degrees(seen_by_view, centroids, positions, stats["direction"])
    mm_per_pixel = np.where(np.isfinite(stats["finest"]), stats["finest"] * metres * 1000.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        flatness = np.where(stats["flat_known"] > 0, stats["flatness_sum"] / stats["flat_known"], np.nan)
    # Flat when at least half the views that judged it found their own depth flat there (a median vote).
    flat = (stats["flat_known"] > 0) & (2 * stats["flat_votes"] >= stats["flat_known"])
    narrow = spread < NARROW_SPREAD_DEGREES
    coarse = ~(mm_per_pixel < COARSE_MM_PER_PIXEL)
    weak = ~flat & (narrow | coarse)
    groups = _regions(np.flatnonzero(weak), surface, centroids, shape, size, metres)
    group_of = np.full(len(surface), -1, np.int32)
    for number, members in enumerate(groups):
        group_of[members] = number
    # How many of each region's voxels every view ranged.
    seen_counts = np.zeros((len(groups), len(frames)), np.int64)
    for view, index in enumerate(seen_by_view):
        labels = group_of[index]
        seen_counts[:, view] = np.bincount(labels[labels >= 0], minlength=len(groups))
    shares = seen_counts / np.maximum([len(members) for members in groups], 1)[:, None] if groups else seen_counts
    route = np.array([str(frame["frame_id"]) in set(priority_frame_ids) for frame in frames], bool)
    # The job's own views first, then the nearest surfaces (finest pixels), then the largest.
    rank = sorted(range(len(groups)), key=lambda g: (not bool((shares[g][route] >= SEEN_FRACTION).any()),
                                                     float(np.nanmedian(mm_per_pixel[groups[g]])), -len(groups[g])))
    recorded = [groups[g] for g in rank[:MAX_REGIONS]]
    shares = shares[rank[:MAX_REGIONS]] if groups else shares
    region_of = np.full(len(surface), -1, np.int32)
    for number, members in enumerate(recorded):
        region_of[members] = number
    up = np.asarray(WORLD_UP, float)
    regions = []
    for number, members in enumerate(recorded):
        region_id = f"wr-{number:03d}"
        points = centroids[members]
        centroid = points.mean(axis=0)
        share = shares[number]
        seeing = [int(k) for k in np.argsort(-share, kind="stable") if share[k] >= SEEN_FRACTION]
        if not seeing:
            seeing = [int(np.argmax(share))]
        toward = positions[seeing] - centroid
        toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)
        nearest = min(seeing, key=lambda k: float(np.linalg.norm(positions[k] - centroid)))
        judged = np.isfinite(flatness[members])
        sharp = np.isfinite(stats["sharpness"][members])
        regions.append({
            "id": region_id, "centroid": [round(float(x), 6) for x in centroid],
            "extent": [round(float(x), 6) for x in points.max(axis=0) - points.min(axis=0) + size],
            "angle_spread_deg": round(_max_pair_angle(toward), 3),
            "mm_per_px": round(float(np.nanmedian(mm_per_pixel[members])), 3),
            "flatness": round(float(np.median(flatness[members][judged])), 4) if judged.any() else None,
            "view_count": len(seeing), "seen_by": [str(frames[k]["frame_id"]) for k in seeing],
            "orbit_deg": list(ORBIT_DEGREES), "voxel_count": int(len(members)),
            "median_voxel_angle_spread_deg": round(float(np.median(spread[members])), 3),
            "median_voxel_view_count": float(np.median(count[members])),
            "seen_by_job_route_view": bool((share[route] >= SEEN_FRACTION).any()),
            "best_local_sharpness": (round(float(np.median(stats["sharpness"][members][sharp])), 3)
                                     if sharp.any() else None),
            "reasons": sorted({*(["narrow_angle_spread"] if narrow[members].mean() >= .5 else []),
                               *(["coarse_mm_per_pixel"] if coarse[members].mean() >= .5 else [])})
            or ["narrow_angle_spread_or_coarse_mm_per_pixel"],
            "suggested_orbit": _orbit(region_id, centroid, positions[seeing], frames[nearest]["intrinsics"], up)})
    arrays = {"voxel_linear_index": surface.astype(np.int64), "centroid": centroids.astype(np.float32),
              "view_count": count, "angle_spread_deg": spread.astype(np.float32),
              "mm_per_px": mm_per_pixel.astype(np.float32), "best_local_sharpness": stats["sharpness"],
              "flatness": flatness.astype(np.float32), "flat": flat, "weak": weak, "region": region_of}
    summary = {"surface_voxel_count": int(len(surface)), "weak_voxel_count": int(weak.sum()),
               "flat_voxel_count": int(flat.sum()), "narrow_spread_voxel_count": int(narrow.sum()),
               "coarse_voxel_count": int(coarse.sum()), "weak_region_count": len(groups),
               "recorded_region_count": len(regions),
               "median_view_count": float(np.median(count)) if len(count) else None,
               "median_angle_spread_deg": round(float(np.median(spread)), 3) if len(spread) else None}
    return {"grid": {"minimum": [float(x) for x in low], "voxel_size_units": size,
                     "shape": [int(x) for x in shape]},
            "up_axis": [round(float(x), 6) for x in up], "regions": regions, "summary": summary, "arrays": arrays}


def _inputs_digest(frames, pose_digest, meters_per_unit, priority_frame_ids, scale_source=None):
    return canonical_digest({"revision": REVISION, "limits": limits(), "pose_result_digest": pose_digest,
                             "scale_source": dict(scale_source) if scale_source else None,
                             "meters_per_unit": meters_per_unit, "priority_frame_ids": sorted(priority_frame_ids),
                             "cameras": [{"frame_id": frame["frame_id"], "intrinsics": dict(frame["intrinsics"]),
                                          "extrinsics": dict(frame["extrinsics"])} for frame in frames]})


def read_weak_regions(root: Path) -> dict[str, Any] | None:
    """The sealed record in an Atlas job root, or None when there is none; a changed one raises."""
    from .local_reconstruction_adapters import _sha256_file

    path = Path(root) / WEAK_REGIONS_FILE
    if not path.is_file() or path.is_symlink():
        return None
    record = json.loads(path.read_text())
    if record.get("digest") != canonical_digest(record, digest_field="digest"):
        raise ValueError("atlas_weak_regions_changed")
    arrays = record.get("coverage_arrays")
    if arrays is not None:
        target = Path(root) / arrays["path"]
        if target.is_symlink() or not target.is_file() or _sha256_file(target) != arrays["sha256"]:
            raise ValueError("atlas_weak_regions_changed")
    return record


def load_weak_regions(capture_root: Path) -> list[dict[str, Any]] | None:
    """A capture's weak regions (``pipeline/atlas/weak_regions.json``), or None when it has no map yet."""
    record = read_weak_regions(Path(capture_root) / "pipeline" / "atlas")
    if record is None or record.get("status") != "mapped":
        return None
    return list(record["regions"])


def write_weak_regions(*, frames: Sequence[Mapping[str, Any]], output_root: Path, pose_digest: str | None,
                       meters_per_unit: float | None = None, priority_frame_ids: Sequence[str] = (),
                       scale_source: Mapping[str, Any] | None = None, scale_reason: str | None = None
                       ) -> dict[str, Any]:
    """Map the coverage once per pose and seal it in ``output_root``; a map that fails is recorded, not raised.

    ``priority_frame_ids`` (the job's route views) put the regions they saw first in the list.
    """
    import io
    import time

    from .local_reconstruction_adapters import _sha256_file

    output_root = Path(output_root)
    priority_frame_ids = sorted({str(key) for key in priority_frame_ids})
    inputs_digest = _inputs_digest(frames, pose_digest, meters_per_unit, priority_frame_ids, scale_source)
    try:
        saved = read_weak_regions(output_root)
    except (ValueError, KeyError, json.JSONDecodeError):
        saved = None
    if saved is not None and saved.get("inputs_digest") == inputs_digest:
        return saved
    started = time.monotonic()
    known_scale = bool(meters_per_unit and meters_per_unit > 0)
    record: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "revision": REVISION, "limits": limits(),
                              "pose_result_digest": pose_digest, "meters_per_unit": meters_per_unit,
                              "priority_frame_ids": priority_frame_ids,
                              "scale_basis": "scene_measurement_estimate" if known_scale
                              else "pose_units_taken_as_metres_unverified",
                              "scale_source": dict(scale_source) if scale_source else None,
                              **({"scale_unavailable_reason": scale_reason} if scale_reason else {})}
    try:
        mapped = coverage_map(frames, meters_per_unit=meters_per_unit, priority_frame_ids=priority_frame_ids)
    except Exception as exc:  # noqa: BLE001 - a map is planning evidence; it never stops the job
        record.update(status="not_mapped", reason=str(exc) if str(exc).startswith("atlas_") else type(exc).__name__,
                      regions=[], voxel_size_m=None)
    else:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **mapped["arrays"])
        output_root.mkdir(parents=True, exist_ok=True)
        arrays_path = output_root / ARRAYS_FILE
        arrays_path.write_bytes(buffer.getvalue())
        size = mapped["grid"]["voxel_size_units"]
        record.update(status="mapped", voxel_size_m=round(size * (meters_per_unit if known_scale else 1.0), 6),
                      grid=mapped["grid"], up_axis=mapped["up_axis"], regions=mapped["regions"],
                      summary=mapped["summary"],
                      coverage_arrays={"path": ARRAYS_FILE, "sha256": _sha256_file(arrays_path),
                                       "fields": sorted(mapped["arrays"])},
                      flags=sorted({*(["scale_unverified_mm_per_pixel_in_pose_units"] if not known_scale else []),
                                    *(["weak_regions_found"] if mapped["regions"] else []),
                                    *(["weak_regions_beyond_recorded_cap"]
                                      if mapped["summary"]["weak_region_count"] > MAX_REGIONS else [])}))
    record.update(inputs_digest=inputs_digest, seconds=round(time.monotonic() - started, 3),
                  basis="posed_view_confident_depth_on_the_observed_space_grid",
                  coordinates="posed_camera_world_frame_pose_units", claim_scope="development_only",
                  purchase_performed=False)
    record["digest"] = canonical_digest(record, digest_field="digest")
    write_json(output_root / WEAK_REGIONS_FILE, record)
    return record


def _scale(source_root, source_pose):
    """Metres per pose unit from the measurement the source pose adopted, its record, and why none if none.

    A scale-reviewed job names its reviewed measurement (``measurements_relative_path``,
    beside the pose); the first estimate it superseded is never read.
    """
    if source_root is None:
        return None, None, "source_root_unavailable"
    relative = (source_pose or {}).get("measurements_relative_path", "scene_measurements.json")
    root = Path(source_root)
    path = root / str(relative)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        return None, {"relative_path": str(relative)}, "scene_measurement_path_invalid"
    if not path.is_file():
        return None, {"relative_path": str(relative)}, "scene_measurement_missing"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, {"relative_path": str(relative)}, "scene_measurement_unreadable"
    if not isinstance(record, dict):
        return None, {"relative_path": str(relative)}, "scene_measurement_unreadable"
    value = record.get("meters_per_unit_estimate")
    source = {"relative_path": str(relative), "digest": record.get("digest"), "scale_status": record.get("scale_status")}
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value), source, None
    return None, source, "scene_measurement_without_scale"


def _record_failure(output_root, pose, reason):
    """A sealed ``not_mapped`` record naming why there is no map, best effort."""
    try:
        record = {"schema_version": SCHEMA_VERSION, "revision": REVISION, "status": "not_mapped", "reason": reason,
                  "pose_result_digest": pose.get("digest") if isinstance(pose, Mapping) else None, "regions": [],
                  "voxel_size_m": None, "claim_scope": "development_only", "purchase_performed": False}
        record["digest"] = canonical_digest(record, digest_field="digest")
        write_json(Path(output_root) / WEAK_REGIONS_FILE, record)
        return record
    except Exception:  # noqa: BLE001 - a record of a failure is itself best effort
        return None


def coverage_before_splat(*, frames: Sequence[Mapping[str, Any]], pose: Mapping[str, Any], output_root: Path,
                          source_root: Path | None = None, selection: Mapping[str, Any] | None = None,
                          source_pose: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The call before the refined splat is bought: the adopted scale, then ``write_weak_regions``.

    It never raises into the pipeline: anything it cannot use is recorded as the reason.
    A measurement it cannot read leaves mm per pixel in pose units, flagged with why.
    """
    try:
        metres, scale_source, scale_reason = _scale(source_root, source_pose)
        record = write_weak_regions(frames=frames, output_root=output_root, pose_digest=pose.get("digest"),
                                    meters_per_unit=metres,
                                    priority_frame_ids=(selection or {}).get("route_frame_ids") or (),
                                    scale_source=scale_source, scale_reason=scale_reason)
    except Exception as exc:  # noqa: BLE001 - nothing here may hold the splat purchase
        return _record_failure(output_root, pose, str(exc) if str(exc).startswith("atlas_") else type(exc).__name__)
    return record
