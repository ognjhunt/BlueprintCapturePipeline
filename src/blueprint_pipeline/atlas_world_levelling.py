"""Level an Atlas world on the surfaces it observed, before anything is placed in it.

An Atlas world's +Y is only roughly up. The dishwasher capture of 2026-10-23
came back with its floor 18 degrees off +Y: no collider face under the subject
was level, the support search found nothing, and a placed object would have
stood on a slope. Compile, its support search and the simulator frame all take
+Y as gravity, so the scene source levels the background first. The dominant
level surface of any built space (a floor, slab, deck, counter, bench or
table: they are parallel) gives the up direction. The lowest such layer, with
almost nothing below it and every camera standing above it, is the floor,
declared at y = 0. A scene with no level surface (open terrain) waits for input.

A single level plane holding a twentieth of the surface is the first test. A
scene whose level surface is split across many parallel planes (an aisle's
shelf boards on both sides, its floor strip partly trimmed) passes none of
them alone, though its up direction is plain: there the area-weighted normal
of every near-level face gives it, and the record says which basis was used.
The floor is declared by the same rules either way.

The retained Atlas outputs stay unchanged; the base scene points at levelled
copies in the provider's own units. Every value is an estimate.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

REVISION = "atlas_world_levelling.v8"
#: Surfaces tilted further than this from the provider's +Y are not taken for
#: gravity: compile's ground search bound, and walls stand at 90 degrees.
MAX_TILT_DEGREES = 45.0
PLANE_TOLERANCE_M = 0.02
SAMPLE_CAP = 200_000
RANSAC_PROBE = 20_000
RANSAC_ITERATIONS = 2000
#: The dominant level plane holds at least this share of the sampled surface
#: (the dishwasher floor held 8%; a level slice through walls holds about 2%).
MIN_LEVEL_SHARE = 0.04
#: Parallel level surfaces: faces within these windows of the current estimate,
#: narrowed in turn, pull it onto their shared normal (a mean shift on the
#: sphere). Along it, each level surface is a 2 cm layer holding at least this
#: share of the points beyond the walls and clutter at its height (the median
#: layer within half a metre); together the excesses must hold the share one
#: level plane would. An excess does not grow with the clutter, as a ratio to
#: it would: shelves full of books are dense walls.
PARALLEL_WINDOWS_DEGREES = (45.0, 30.0, 20.0, 12.0, 8.0, 5.0)
PARALLEL_SHIFTS = 3
PARALLEL_BACKGROUND_M = 0.5
PARALLEL_MIN_LAYER_SHARE = 0.005
PARALLEL_REFITS = 2
#: Coarse-to-fine: a direction a degree off spreads a 6 m floor over 10 cm, more than one 2 cm layer.
#: Layers this many times thicker are found and refitted first; only the 2 cm ones are counted.
PARALLEL_COARSE_LAYERS = (6.0, 3.0, 1.0)
#: A floor seen along a direction a couple of degrees off its own normal spreads over a band
#: (6 m at 2 degrees: 21 cm), and no 2 cm layer holds it. The lowest band holding a floor's
#: share is refitted at these thicknesses in turn, trimming points beyond 2.5 robust deviations,
#: until it is a 4 cm band about its own plane.
FLOOR_BANDS_M = (0.30, 0.15, 0.08, 0.04)
FLOOR_BAND_FIND_M = 0.10
FLOOR_TRIM_DEVIATIONS = 2.5
#: A floor too uneven to declare (no 2 cm plane holds a level plane's share) still gives the up
#: direction when its own plane and the mean of every surface's normal agree this closely: two
#: independent estimates. The scene is levelled on it, and no floor is declared.
DIRECTION_AGREEMENT_DEGREES = 0.5
#: Fallback floor, a development-only estimate, flagged (owner, 2026-11-07: "build the flagged fallback
#: floor"). When nothing observed levels the scene, but the floor band's own plane and every
#: surface's mean normal both lie within FALLBACK_MAX_TILT_DEGREES of the provider's +Y, +Y is taken
#: as up, and a flat floor at the band's median height stands in for an uneven reconstructed one. In
#: the collider copy, the near-level faces within FALLBACK_REPLACE_HALF_M of that height are replaced
#: by the flat floor; walls, shelf sides and the splat are kept as captured.
FALLBACK_MAX_TILT_DEGREES = 3.0
FALLBACK_BAND_M = 0.30
FALLBACK_REPLACE_HALF_M = 0.12
FALLBACK_LEVEL_FACE_DEGREES = 30.0
FALLBACK_FLOOR_MARGIN_M = 0.2
FALLBACK_FLOOR_CELL_M = 0.25
#: Floor under the filmed path, a development-only estimate, flagged. A camera never films the floor under
#: its own feet, so where the walk starts the reconstruction can have no floor at all. capture-5b067498
#: (2026-11-08): no collision face lay over the compiled start, and the G1 fell through the scene. The
#: person filming stood and walked there, so a floor is there: in the collider copy of a scene with a
#: declared floor, a flat floor at its height fills every cell within WALKED_FLOOR_MARGIN_M of a camera's
#: footprint (a G1 stance and its start offsets fit inside). Walls, shelves and the splat are as captured.
WALKED_FLOOR_MARGIN_M = 0.5
WALKED_FLOOR_CELL_M = 0.25
#: Kept in the record, so a refusal can be read without the mesh: the densest layers along the chosen
#: direction, and along +Y and every candidate direction the share of points in each 2 cm band of
#: height around the cameras (a floor shows as a peak a camera height below them, as thick as it is).
REPORTED_LAYERS = 12
PROFILE_BAND_M = 0.02
PROFILE_SPAN_M = (-3.0, 2.0)
#: Faces read at a time: a room's mesh has millions of faces (capture-447's: 6.5 M).
FACE_BLOCK = 1 << 20
#: The floor is the lowest 2 cm layer holding this share, with almost nothing below it.
MIN_FLOOR_SHARE = 0.01
MAX_BELOW_FLOOR_SHARE = 0.01
BELOW_FLOOR_M = 0.05
#: Counters, benches and most tables are shallower; a floor or slab seen across
#: a space is wider both ways. A narrower one levels the scene but is not declared.
MIN_FLOOR_EXTENT_M = 1.0
#: A first fit slightly off the floor only keeps a band of it; refit on the
#: band's plane until the whole floor is in (a degree is 3.5 cm at 2 m).
FLOOR_REFITS = 4
#: Every camera stands above the floor, and the typical camera this high above
#: it (estimated metres): handheld, pole or mezzanine views of any space, but
#: not close-ups over a table or bench.
CAMERA_MIN_HEIGHT_M = 0.2
CAMERA_MEDIAN_HEIGHT_BOUNDS_M = (0.5, 10.0)
SPLAT_NAME, MESH_NAME = "atlas_depth_splat_levelled.ply", "atlas_mesh_levelled.glb"


def _plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane: unit normal with a non-negative +Y part, and its offset along it."""
    centre = points.mean(axis=0)
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][-1]
    normal = normal if normal[1] >= 0 else -normal
    return normal, float(centre @ normal)


def _level_direction(points: np.ndarray, rng: np.random.Generator,
                     report: dict[str, Any] | None = None,
                     largest_plane: list[np.ndarray] | None = None) -> np.ndarray | None:
    """Unit normal of the dominant near-level plane, refined on all its points; None when there is none.

    ``report`` receives the largest share of the probe any near-level plane held, and ``largest_plane``
    that plane's normal, refined the same way, whether or not it passed.
    """
    limit = math.cos(math.radians(MAX_TILT_DEGREES))
    probe = points[rng.choice(len(points), RANSAC_PROBE, replace=False)] if len(points) > RANSAC_PROBE else points
    best, best_count, largest, widest = None, int(MIN_LEVEL_SHARE * len(probe)), 0, None
    for _ in range(RANSAC_ITERATIONS):
        a, b, c = probe[rng.choice(len(probe), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length <= 1e-12 or abs(normal[1]) < limit * length:
            continue
        normal = normal / length
        count = int(np.count_nonzero(np.abs((probe - a) @ normal) <= PLANE_TOLERANCE_M))
        if count > largest:
            largest, widest = count, (a, normal)
        if count > best_count:
            best, best_count = (a, normal), count
    if report is not None:
        report.update(probe_points=int(len(probe)), largest_level_plane_share=largest / max(len(probe), 1),
                      minimum_level_plane_share=MIN_LEVEL_SHARE)
    if largest_plane is not None and widest is not None:
        inliers = points[np.abs((points - widest[0]) @ widest[1]) <= PLANE_TOLERANCE_M]
        if len(inliers) >= 3:
            largest_plane.append(_plane(inliers)[0])
    if best is None:
        return None
    normal, _ = _plane(points[np.abs((points - best[0]) @ best[1]) <= PLANE_TOLERANCE_M])
    return normal if normal[1] >= limit else None


def _face_normals_and_areas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals turned to non-negative +Y, and areas, of every face with an area, ``FACE_BLOCK`` at a time."""
    normals, areas = [], []
    for start in range(0, len(faces), FACE_BLOCK):
        tri = vertices[faces[start:start + FACE_BLOCK]]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(cross, axis=1)
        keep = length > 1e-12
        unit = cross[keep] / length[keep, None]
        unit[unit[:, 1] < 0] *= -1.0
        normals.append(unit.astype(np.float32))
        areas.append((0.5 * length[keep]).astype(np.float32))
    if not normals:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    return np.concatenate(normals), np.concatenate(areas)


def _tilt(direction: np.ndarray) -> float:
    return math.degrees(math.acos(min(1.0, float(direction[1]))))


def _normal_mode(normals: np.ndarray, areas: np.ndarray) -> np.ndarray | None:
    """A mean shift over area-weighted face normals from +Y, narrowed window by window; None with none near."""
    estimate = np.array([0.0, 1.0, 0.0])
    for window in PARALLEL_WINDOWS_DEGREES:
        cosine = math.cos(math.radians(window))
        for _ in range(PARALLEL_SHIFTS):
            near = normals @ estimate.astype(np.float32) >= cosine
            if float(areas[near].sum(dtype=np.float64)) <= 0:
                return None
            mean = (normals[near].astype(np.float64) * areas[near, None]).sum(axis=0)
            estimate = mean / np.linalg.norm(mean)
    return estimate


def _dense_layers(points: np.ndarray, direction: np.ndarray,
                  thickness: float = 1.0) -> list[tuple[float, float, float]]:
    """(height, excess count, ratio to the background) of each level layer along ``direction``.

    A layer is 2 cm (times ``thickness`` for a coarse pass); its background is the median layer
    within ``PARALLEL_BACKGROUND_M``, and its excess what it holds beyond that.
    """
    from scipy.ndimage import median_filter

    step = thickness * PLANE_TOLERANCE_M / 2
    heights = points @ direction
    counts, edges = np.histogram(heights, bins=np.arange(heights.min(), heights.max() + 3 * step, step))
    pair = (counts[:-1] + counts[1:]).astype(np.float64)
    # Beyond the lowest and highest points there are none: padding with the edge layer itself would
    # make the floor, the lowest layer, its own background.
    background = median_filter(pair, size=2 * int(round(PARALLEL_BACKGROUND_M / step)) + 1,
                               mode="constant", cval=0.0)
    excess = pair - background
    dense = excess >= PARALLEL_MIN_LAYER_SHARE * len(points)
    # One layer per surface: the largest excess within two tolerances (4 cm at the finest), so a broad
    # bump in rough ground counts once, as a plane does.
    reach = 2 * int(round(2 * PLANE_TOLERANCE_M * thickness / step))
    return [(float(edges[i + 1]), float(excess[i]), float(pair[i] / max(background[i], 1.0)))
            for i in np.flatnonzero(dense)
            if excess[i] == excess[max(0, i - reach):i + reach + 1].max()
            and not (i > 0 and excess[i - 1] == excess[i])]


def _layered_direction(points: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, list, float]:
    """``direction`` refitted on its dense layers at once, each about its own height; its layers and their share."""
    estimate = direction
    for thickness in PARALLEL_COARSE_LAYERS:
        for _ in range(PARALLEL_REFITS):
            heights = points @ estimate
            groups = [points[np.abs(heights - height) <= thickness * PLANE_TOLERANCE_M] for height, _, _ in
                      _dense_layers(points, estimate, thickness)]
            groups = [group - group.mean(axis=0) for group in groups if len(group) >= 3]
            if not groups:
                break
            normal = np.linalg.svd(np.concatenate(groups), full_matrices=False)[2][-1]
            estimate = normal if normal[1] >= 0 else -normal
    layers = _dense_layers(points, estimate)
    return estimate, layers, sum(count for _, count, _ in layers) / len(points)


def _height_profile(points: np.ndarray, source: str, direction: np.ndarray,
                    camera_centres: np.ndarray | None) -> dict[str, Any]:
    """The share of ``points`` in each band of height along ``direction``, from the median camera."""
    reference = (float(np.median(np.asarray(camera_centres) @ direction))
                 if camera_centres is not None and len(camera_centres) else 0.0)
    heights = points @ direction - reference
    low, high = PROFILE_SPAN_M
    counts, _ = np.histogram(heights, bins=np.linspace(low, high, int(round((high - low) / PROFILE_BAND_M)) + 1))
    return {"source": source, "tilt_degrees": _tilt(direction),
            "reference": "median_camera_height" if camera_centres is not None else "origin",
            "band_m": PROFILE_BAND_M, "from_m": low, "to_m": high,
            "shares": [round(float(count) / len(points), 5) for count in counts],
            "below_share": float(np.mean(heights < low)), "above_share": float(np.mean(heights >= high))}


def _parallel_level_direction(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray,
                              report: dict[str, Any] | None = None, *,
                              directions: tuple[np.ndarray, ...] = (),
                              camera_centres: np.ndarray | None = None) -> np.ndarray | None:
    """The normal shared by parallel level surfaces; None when their layers hold too little of ``points``.

    ``vertices``, ``points`` and ``camera_centres`` are in metres. Each candidate direction is
    refitted on the dense 2 cm layers of ``points`` along it, and the one whose layers hold the most
    is kept. The candidates are the mean shift of the faces' area-weighted normals from +Y, and
    ``directions`` (the largest single level plane's normal: exact for that plane, so a long floor or
    board does not smear across layers). ``report`` receives every candidate, the kept one's densest
    layers, and along +Y and each candidate the share of points in each band of height around the
    median camera.
    """
    normals, areas = _face_normals_and_areas(vertices, faces)
    candidates = [("face_normal_mode", _normal_mode(normals, areas))] if len(areas) else []
    candidates += [("largest_level_plane", direction) for direction in directions]
    limit = math.cos(math.radians(MAX_TILT_DEGREES))
    tried = [(source, *_layered_direction(points, direction)) for source, direction in candidates
             if direction is not None and len(points) >= 3 and direction[1] >= limit]
    # A refit can carry a candidate past the search's tilt bound; such a direction never levels the scene.
    best = max((row for row in tried if row[1][1] >= limit), key=lambda row: row[3], default=None)
    if report is not None:
        report["parallel_candidates"] = [{"source": source, "tilt_degrees": _tilt(direction), "layers": len(layers),
                                          "share": share} for source, direction, layers, share in tried]
        report.update(parallel_level_layers=len(best[2]) if best else 0, parallel_level_share=best[3] if best else 0.0)
        direction = best[1] if best else np.array([0.0, 1.0, 0.0])
        reference = (float(np.median(np.asarray(camera_centres) @ direction))
                     if camera_centres is not None and len(camera_centres) else 0.0)
        if best:
            report.update(parallel_level_tilt_degrees=_tilt(best[1]), parallel_level_source=best[0],
                          densest_layers=[{"height_from_cameras_m": height - reference, "excess_share": count / len(points),
                                           "background_ratio": ratio} for height, count, ratio in
                                          sorted(best[2], key=lambda row: -row[1])[:REPORTED_LAYERS]])
        if len(points):
            profiles = [("plus_y", np.array([0.0, 1.0, 0.0]))] + [(source, found) for source, found, _, _ in tried]
            report["height_profiles"] = [_height_profile(points, source, found, camera_centres)
                                         for source, found in profiles]
    if best is None or best[3] < MIN_LEVEL_SHARE:
        return None
    return best[1]


def _floor(points: np.ndarray, normal: np.ndarray) -> dict[str, Any] | None:
    """The lowest well-populated 2 cm layer along ``normal``, refitted.

    None when points lie below it, or when it is no wider than a counter or table.
    """
    heights = points @ normal
    step = PLANE_TOLERANCE_M / 2
    counts, edges = np.histogram(heights, bins=np.arange(heights.min(), heights.max() + 3 * step, step))
    dense = np.flatnonzero(counts[:-1] + counts[1:] >= MIN_FLOOR_SHARE * len(points))
    if not len(dense):
        return None
    floor_normal, height = normal, float(edges[dense[0] + 1])
    for _ in range(FLOOR_REFITS):
        refit = _plane(points[np.abs(points @ floor_normal - height) <= PLANE_TOLERANCE_M])
        if refit[0] @ normal < math.cos(math.radians(MAX_TILT_DEGREES)):
            return None
        floor_normal, height = refit
    heights = points @ floor_normal
    below = int(np.count_nonzero(heights < height - BELOW_FLOOR_M))
    if below > MAX_BELOW_FLOOR_SHARE * len(points):
        return None
    inliers = points[np.abs(heights - height) <= PLANE_TOLERANCE_M]
    flat = inliers - np.outer(inliers @ floor_normal, floor_normal)
    flat -= flat.mean(axis=0)
    spans = sorted(float(np.subtract(*np.percentile(flat @ axis, [98, 2])))
                   for axis in np.linalg.svd(flat, full_matrices=False)[2][:2])
    if spans[0] < MIN_FLOOR_EXTENT_M:
        return None
    residual = inliers @ floor_normal - height
    return {"normal": floor_normal, "height_m": height, "inlier_share": len(inliers) / len(points),
            "below_share": below / len(points), "rms_m": float(np.sqrt(np.mean(residual ** 2))),
            "extent_m": spans[::-1]}


def _band_floor(points: np.ndarray, direction: np.ndarray,
                report: dict[str, Any] | None = None, fit: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """The floor found as the lowest band holding a floor's share, refitted coarse to fine; None if it fails.

    Along ``direction`` the lowest 10 cm holding ``MIN_FLOOR_SHARE`` of the points starts a 30 cm
    band; each band is fitted as a plane, points beyond ``FLOOR_TRIM_DEVIATIONS`` robust deviations
    are trimmed, and the band narrows about the new plane. The result must hold a level plane's
    share (``MIN_LEVEL_SHARE``) within 2 cm and meet the floor rules: almost nothing below it, and
    wider than a counter both ways. ``report`` receives what was found either way, and ``fit``
    the fitted floor whether or not it passed.
    """
    if len(points) < 3:
        return None
    heights = points @ direction
    step = PLANE_TOLERANCE_M
    counts, edges = np.histogram(heights, bins=np.arange(heights.min(), heights.max() + 2 * step, step))
    reach = int(round(FLOOR_BAND_FIND_M / step))
    windows = np.convolve(counts, np.ones(reach, dtype=np.int64), mode="valid")
    found = np.flatnonzero(windows >= MIN_FLOOR_SHARE * len(points))
    if not len(found):
        return None
    normal, height = direction, float(edges[found[0]]) + FLOOR_BAND_FIND_M / 2
    limit = math.cos(math.radians(MAX_TILT_DEGREES))
    for thickness in FLOOR_BANDS_M:
        band = points[np.abs(points @ normal - height) <= thickness / 2]
        if len(band) < 3:
            return None
        fitted, offset = _plane(band)
        residual = band @ fitted - offset
        spread = 1.4826 * float(np.median(np.abs(residual)))
        kept = band[np.abs(residual) <= max(FLOOR_TRIM_DEVIATIONS * spread, PLANE_TOLERANCE_M)]
        if len(kept) >= 3:
            fitted, offset = _plane(kept)
        if fitted[1] < limit:
            return None
        normal, height = fitted, offset
    heights = points @ normal
    inliers = points[np.abs(heights - height) <= PLANE_TOLERANCE_M]
    below = int(np.count_nonzero(heights < height - BELOW_FLOOR_M))
    spans = [0.0, 0.0]
    if len(inliers) >= 3:
        flat = inliers - np.outer(inliers @ normal, normal)
        flat -= flat.mean(axis=0)
        spans = sorted(float(np.subtract(*np.percentile(flat @ axis, [98, 2])))
                       for axis in np.linalg.svd(flat, full_matrices=False)[2][:2])
    residual = inliers @ normal - height if len(inliers) else np.zeros(1)
    value = {"normal": normal, "height_m": height, "inlier_share": len(inliers) / len(points),
             "below_share": below / len(points), "rms_m": float(np.sqrt(np.mean(residual ** 2))),
             "extent_m": spans[::-1]}
    if report is not None:
        report["floor_band"] = {"tilt_degrees": _tilt(normal), "inlier_share": value["inlier_share"],
                                "below_share": value["below_share"], "rms_m": value["rms_m"],
                                "extent_m": value["extent_m"]}
    if fit is not None:
        fit.append(value)
    if (value["inlier_share"] < MIN_LEVEL_SHARE or below > MAX_BELOW_FLOOR_SHARE * len(points)
            or spans[0] < MIN_FLOOR_EXTENT_M):
        return None
    return value


def _fallback_floor(points: np.ndarray) -> dict[str, Any] | None:
    """A flat floor along +Y at the median height of the lowest band holding a floor's share; None if none.

    The band is the lowest 10 cm holding ``MIN_FLOOR_SHARE`` of the points and the ``FALLBACK_BAND_M``
    above it. It must be wider than a counter both ways and have almost nothing below it.
    """
    if len(points) < 3:
        return None
    up = np.array([0.0, 1.0, 0.0])
    heights = points[:, 1]
    step = PLANE_TOLERANCE_M
    counts, edges = np.histogram(heights, bins=np.arange(heights.min(), heights.max() + 2 * step, step))
    reach = int(round(FLOOR_BAND_FIND_M / step))
    windows = np.convolve(counts, np.ones(reach, dtype=np.int64), mode="valid")
    found = np.flatnonzero(windows >= MIN_FLOOR_SHARE * len(points))
    if not len(found):
        return None
    base = float(edges[found[0]])
    band = points[(heights >= base - BELOW_FLOOR_M) & (heights <= base + FALLBACK_BAND_M)]
    if len(band) < 3:
        return None
    height = float(np.median(band[:, 1]))
    below = int(np.count_nonzero(heights < base - BELOW_FLOOR_M))
    flat = band[:, [0, 2]] - band[:, [0, 2]].mean(axis=0)
    spans = sorted(float(np.subtract(*np.percentile(flat @ axis, [98, 2])))
                   for axis in np.linalg.svd(flat, full_matrices=False)[2])
    if below > MAX_BELOW_FLOOR_SHARE * len(points) or spans[0] < MIN_FLOOR_EXTENT_M:
        return None
    low, high = np.percentile(band[:, 1], [5, 95])
    return {"normal": up, "height_m": height, "inlier_share": len(band) / len(points),
            "below_share": below / len(points), "rms_m": float(np.std(band[:, 1])), "extent_m": spans[::-1],
            "band_spread_m": [float(low - height), float(high - height)]}


def _replace_uneven_floor(mesh, *, meters_per_unit: float) -> dict[str, Any]:
    """In a levelled mesh (floor at y = 0), swap the near-level faces near the floor for a flat floor grid.

    Returns what was replaced. The flat floor covers the replaced faces' extent, along their own
    principal axes, plus ``FALLBACK_FLOOR_MARGIN_M``, in cells of ``FALLBACK_FLOOR_CELL_M``.
    """
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    metres = vertices * meters_per_unit
    tri = metres[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    level = np.abs(cross[:, 1]) >= math.cos(math.radians(FALLBACK_LEVEL_FACE_DEGREES)) * np.maximum(length, 1e-12)
    near = np.all(np.abs(tri[:, :, 1]) <= FALLBACK_REPLACE_HALF_M, axis=1)
    replaced = level & near & (length > 1e-12)
    corners = tri[replaced].reshape(-1, 3)[:, [0, 2]]
    if len(corners) < 3:
        # No face passes the stricter replacement test (a coarse or noisy band): the flat floor still
        # covers the band, read from its vertices, or the whole scene as a last resort.
        corners = metres[np.abs(metres[:, 1]) <= FALLBACK_REPLACE_HALF_M][:, [0, 2]]
    if len(corners) < 3:
        corners = metres[:, [0, 2]]
    centre = corners.mean(axis=0)
    axes = np.linalg.svd(corners - centre, full_matrices=False)[2]
    along = (corners - centre) @ axes.T
    low = np.percentile(along, 1, axis=0) - FALLBACK_FLOOR_MARGIN_M
    high = np.percentile(along, 99, axis=0) + FALLBACK_FLOOR_MARGIN_M
    counts = np.maximum(1, np.ceil((high - low) / FALLBACK_FLOOR_CELL_M).astype(int))
    u, v = (np.linspace(low[k], high[k], counts[k] + 1) for k in range(2))
    grid = np.array([[a, b] for a in u for b in v])
    xz = centre + grid @ axes
    floor_vertices = np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]]) / meters_per_unit
    n_v = len(v)
    cells = [(i * n_v + j, (i + 1) * n_v + j, i * n_v + j + 1, (i + 1) * n_v + j + 1)
             for i in range(len(u) - 1) for j in range(n_v - 1)]
    start = len(vertices)
    # Winding with +Y up for both triangles of each cell.
    floor_faces = np.array([tri_ for a, b, c, d in cells for tri_ in ((a, c, b), (b, c, d))], dtype=np.int64) + start
    colours = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and getattr(visual, "kind", None) == "vertex":
        existing = np.asarray(visual.vertex_colors)
        floor_colour = np.median(existing[np.unique(faces[replaced])], axis=0) if replaced.any() else existing[0]
        colours = np.vstack([existing, np.tile(floor_colour, (len(floor_vertices), 1))]).astype(np.uint8)
    rebuilt = trimesh.Trimesh(vertices=np.vstack([vertices, floor_vertices]),
                              faces=np.vstack([faces[~replaced], floor_faces]), vertex_colors=colours, process=False)
    rebuilt.remove_unreferenced_vertices()
    areas = 0.5 * length
    return {"mesh": rebuilt, "replaced_faces": int(replaced.sum()),
            "replaced_area_share": float(areas[replaced].sum() / max(areas.sum(), 1e-12)),
            "flat_floor_extent_m": [float(high[k] - low[k]) for k in range(2)],
            "flat_floor_faces": int(len(floor_faces))}


def _fill_walked_floor(mesh, *, footprints_m: np.ndarray, meters_per_unit: float) -> dict[str, Any] | None:
    """In a levelled mesh (floor at y = 0), add a flat floor under the cameras' footprints; None without any.

    ``footprints_m`` are the cameras' (x, z) in metres. Every cell of ``WALKED_FLOOR_CELL_M`` whose
    centre lies within ``WALKED_FLOOR_MARGIN_M`` of a footprint gets two faces at y = 0, so the walked
    path has a floor whatever the reconstruction kept there. Existing faces are kept.
    """
    import trimesh

    footprints = np.asarray(footprints_m, dtype=np.float64).reshape(-1, 2)
    if not len(footprints):
        return None
    cell = WALKED_FLOOR_CELL_M
    low = np.floor((footprints.min(axis=0) - WALKED_FLOOR_MARGIN_M) / cell) * cell
    # Each camera stamps the cells near it: work grows with the cameras, not with the walk's bounding box.
    reach = int(math.ceil(WALKED_FLOOR_MARGIN_M / cell))
    cells: set[tuple[int, int]] = set()
    for footprint in footprints:
        base = np.floor((footprint - low) / cell).astype(int)
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                i, j = int(base[0] + di), int(base[1] + dj)
                if i >= 0 and j >= 0 and float(np.linalg.norm(footprint - (low + (np.array([i, j]) + 0.5) * cell))) \
                        <= WALKED_FLOOR_MARGIN_M:
                    cells.add((i, j))
    rows = sorted(cells)
    if not rows:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    corners = [(low + np.array([i + di, j + dj]) * cell) for i, j in rows for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1))]
    xz = np.asarray(corners, dtype=np.float64)
    floor_vertices = np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]]) / meters_per_unit
    start = len(vertices)
    # Winding with +Y up for both triangles of each cell (a = (x, z), b = (x + 1, z), c = (x, z + 1), d).
    floor_faces = np.array([tri for k in range(len(rows)) for a, b, c, d in [tuple(start + 4 * k + n for n in range(4))]
                            for tri in ((a, c, b), (b, c, d))], dtype=np.int64)
    colours = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and getattr(visual, "kind", None) == "vertex":
        existing = np.asarray(visual.vertex_colors)
        colours = np.vstack([existing, np.tile(np.median(existing, axis=0), (len(floor_vertices), 1))]).astype(np.uint8)
    filled = trimesh.Trimesh(vertices=np.vstack([vertices, floor_vertices]), faces=np.vstack([faces, floor_faces]),
                             vertex_colors=colours, process=False)
    return {"mesh": filled, "claim": "development_only_estimate", "basis": "floor_under_the_filmed_path",
            "cells": len(rows), "faces_added": int(len(floor_faces)), "cell_m": cell,
            "margin_m": WALKED_FLOOR_MARGIN_M, "cameras": int(len(footprints))}


def _cameras_above(floor: dict[str, Any] | None, centres_m: np.ndarray) -> np.ndarray | None:
    """The cameras' heights over ``floor``, or None when they do not stand above it as people do."""
    if floor is None:
        return None
    heights = centres_m @ floor["normal"] - floor["height_m"]
    low, high = CAMERA_MEDIAN_HEIGHT_BOUNDS_M
    if not len(heights) or heights.min() < CAMERA_MIN_HEIGHT_M or not low <= float(np.median(heights)) <= high:
        return None
    return heights


def _rotation_to_up(normal: np.ndarray) -> np.ndarray:
    """The smallest rotation taking ``normal`` to +Y: no turn about the vertical."""
    axis, cosine = np.cross(normal, [0.0, 1.0, 0.0]), float(normal[1])
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + cosine)


def _rotate_quaternions(rotation: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Left-multiply wxyz Gaussian rotations by ``rotation`` (a turn under 90 degrees); norms are kept."""
    w0 = math.sqrt(1.0 + float(np.trace(rotation))) / 2.0
    x0, y0, z0 = (np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                            rotation[1, 0] - rotation[0, 1]]) / (4.0 * w0))
    w, x, y, z = quats.T
    return np.stack([w0 * w - x0 * x - y0 * y - z0 * z, w0 * x + x0 * w + y0 * z - z0 * y,
                     w0 * y - x0 * z + y0 * w + z0 * x, w0 * z + x0 * y - y0 * x + z0 * w], axis=1)


def level_atlas_world(*, splat_path: Path, splat_digest: str, mesh_path: Path, mesh_digest: str,
                      meters_per_unit: float, camera_centres: np.ndarray, output_root: Path) -> dict[str, Any]:
    """``levelled`` with copies of the splat and mesh, or ``needs_input`` with a reason.

    ``camera_centres`` are the background pose's cameras in provider units. The
    copies stay in provider units: rotated about the origin and, when the floor
    is observed, moved so that it lies at y = 0. Retries reuse the receipt.
    """
    centres = np.asarray(camera_centres, dtype=np.float64).reshape(-1, 3)
    binding = {"revision": REVISION, "splat_digest": splat_digest, "mesh_digest": mesh_digest,
               "meters_per_unit": float(meters_per_unit), "camera_centres": np.round(centres, 9).tolist()}
    root = output_root / canonical_digest(binding)[7:31]
    receipt_path = root / "levelling.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if (prior.get("binding") != binding or prior.get("digest") != canonical_digest(prior, digest_field="digest")
                or any(Path(row["path"]).parent != root or not Path(row["path"]).is_file()
                       or _sha256_file(Path(row["path"])) != row["sha256"] for row in prior.get("outputs", []))):
            raise ValueError("atlas_world_levelling_cache_changed")
        return prior
    import trimesh
    from .gaussian_splat_decode import read_standard_3dgs_ply, write_standard_3dgs_ply

    value: dict[str, Any] = {"schema_version": REVISION, "binding": binding}
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * meters_per_unit
    rng = np.random.default_rng(607)
    sample = vertices[rng.choice(len(vertices), SAMPLE_CAP, replace=False)] if len(vertices) > SAMPLE_CAP else vertices
    search: dict[str, Any] = {"sampled_vertices": int(len(sample))}
    fallback: dict[str, Any] | None = None
    largest: list[np.ndarray] = []
    normal = _level_direction(sample, rng, search, largest) if len(sample) >= 3 else None
    basis = "single_level_plane"
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if normal is None:
        normal = _parallel_level_direction(vertices, faces, sample, search,
                                           directions=tuple(largest), camera_centres=centres * meters_per_unit)
        basis = "parallel_level_surfaces"
    floor = _floor(sample, normal) if normal is not None else None
    camera_heights = _cameras_above(floor, centres * meters_per_unit)
    if camera_heights is None:
        # No 2 cm floor layer along the level direction: the floor may lie a couple of degrees off it
        # (a reconstructed world's shelves and floor need not be exactly parallel). Fit it on its own.
        mode = _normal_mode(*_face_normals_and_areas(vertices, faces)) if normal is None else None
        start = normal if normal is not None else mode
        fitted: list[dict[str, Any]] = []
        floor = _band_floor(sample, start if start is not None else np.array([0.0, 1.0, 0.0]), search, fitted)
        camera_heights = _cameras_above(floor, centres * meters_per_unit)
        if camera_heights is None:
            floor = None
        elif normal is None:
            normal, basis = floor["normal"], "observed_floor_band"
        if normal is None and mode is not None and fitted:
            # The floor is too uneven to declare, but its own plane and every surface's mean normal agree.
            band = fitted[0]
            agreement = math.degrees(math.acos(min(1.0, abs(float(band["normal"] @ mode)))))
            search["direction_agreement_degrees"] = agreement
            if (agreement <= DIRECTION_AGREEMENT_DEGREES and band["inlier_share"] >= MIN_FLOOR_SHARE
                    and band["extent_m"][1] >= MIN_FLOOR_EXTENT_M and band["normal"][1] >= math.cos(
                        math.radians(MAX_TILT_DEGREES))):
                normal, basis = band["normal"], "agreeing_normal_mode_and_floor_band"
        if normal is None and mode is not None and fitted:
            # The flagged fallback: both observed estimates lie near the provider's +Y; a flat floor
            # at the uneven band's median height stands in, as a development-only estimate.
            estimates = {"floor_band_tilt_degrees": _tilt(fitted[0]["normal"]),
                         "face_normal_mode_tilt_degrees": _tilt(mode),
                         "separation_degrees": search.get("direction_agreement_degrees")}
            search["fallback_estimates"] = estimates
            if max(estimates["floor_band_tilt_degrees"], estimates["face_normal_mode_tilt_degrees"]) \
                    <= FALLBACK_MAX_TILT_DEGREES:
                candidate = _fallback_floor(sample)
                heights = _cameras_above(candidate, centres * meters_per_unit)
                if heights is not None:
                    floor, camera_heights = candidate, heights
                    normal, basis = candidate["normal"], "provider_up_fallback"
                    fallback = {"basis": "flat_floor_at_uneven_band_median", "claim": "development_only_estimate",
                                "up_estimates": estimates, "band_spread_m": candidate["band_spread_m"],
                                "band_share": candidate["inlier_share"]}
    value["level_search"] = search
    splat = read_standard_3dgs_ply(splat_path) if normal is not None else None
    if normal is None or splat.sh_rest is not None:
        # Rotating view-dependent colour is not implemented; refuse rather than tilt it.
        value.update(status="needs_input", reason="website_atlas_scene_level_unobserved" if normal is None
                     else "website_atlas_scene_level_splat_view_dependent", outputs=[])
        value["digest"] = canonical_digest(value, digest_field="digest")
        write_json(receipt_path, value)
        return value
    up = floor["normal"] if floor is not None else normal
    rotation = _rotation_to_up(up)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    if floor is not None:
        transform[1, 3] = -floor["height_m"] / meters_per_unit
    mesh.apply_transform(transform)
    if fallback is not None:
        swapped = _replace_uneven_floor(mesh, meters_per_unit=meters_per_unit)
        mesh = swapped.pop("mesh")
        fallback.update(swapped)
    walked = None
    if floor is not None:
        # The cameras in the levelled copy, in metres: their (x, z) is where the person filming walked.
        levelled = (centres @ rotation.T + transform[:3, 3]) * meters_per_unit
        walked = _fill_walked_floor(mesh, footprints_m=levelled[:, [0, 2]], meters_per_unit=meters_per_unit)
        if walked is not None:
            mesh = walked.pop("mesh")
    root.mkdir(parents=True, exist_ok=True)
    mesh.export(str(root / MESH_NAME))
    xyz = splat.xyz.astype(np.float64) @ rotation.T + transform[:3, 3]
    quats = _rotate_quaternions(rotation, splat.quats.astype(np.float64))
    write_standard_3dgs_ply(dataclasses.replace(splat, xyz=xyz.astype(np.float32), quats=quats.astype(np.float32)),
                            root / SPLAT_NAME)
    value.update(
        status="levelled", provider_to_levelled=transform.tolist(),
        tilt_degrees=math.degrees(math.acos(min(1.0, float(up[1])))),
        up_basis=("fallback_fitted_floor" if fallback is not None else
                  "observed_floor" if floor is not None else "dominant_level_surface"),
        level_direction_basis=basis,
        **({"fallback_floor": fallback} if fallback is not None else {}),
        **({"walked_floor": walked} if walked is not None else {}),
        ground_plane_offset_m=0.0 if floor is not None else None,
        floor=None if floor is None else {
            "height_before_m": floor["height_m"], "inlier_share": floor["inlier_share"],
            "below_share": floor["below_share"], "rms_m": floor["rms_m"], "extent_m": floor["extent_m"],
            "camera_heights_m": [float(h) for h in camera_heights]},
        outputs=[{"role": role, "path": str(root / name), "sha256": _sha256_file(root / name)}
                 for role, name in (("splat_ply", SPLAT_NAME), ("supported_mesh_glb", MESH_NAME))],
        physical_measurement=False)
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(receipt_path, value)
    return value
