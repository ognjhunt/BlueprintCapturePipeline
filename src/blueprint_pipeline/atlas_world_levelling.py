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

REVISION = "atlas_world_levelling.v3"
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
#: Kept in the record, so a refusal can be read without the mesh: the densest layers along the chosen
#: direction, and the share of points in each band of height around the cameras.
REPORTED_LAYERS = 12
PROFILE_BAND_M = 0.1
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
    layers, and the share of points in each band of height around the median camera.
    """
    normals, areas = _face_normals_and_areas(vertices, faces)
    candidates = [("face_normal_mode", _normal_mode(normals, areas))] if len(areas) else []
    candidates += [("largest_level_plane", direction) for direction in directions]
    limit = math.cos(math.radians(MAX_TILT_DEGREES))
    tried = [(source, *_layered_direction(points, direction)) for source, direction in candidates
             if direction is not None and len(points) >= 3 and direction[1] >= limit]
    best = max(tried, key=lambda row: row[3], default=None)
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
            heights = points @ direction - reference
            low, high = PROFILE_SPAN_M
            counts, _ = np.histogram(heights, bins=np.linspace(low, high, int(round((high - low) / PROFILE_BAND_M)) + 1))
            report["height_profile"] = {
                "reference": "median_camera_height" if camera_centres is not None else "origin",
                "band_m": PROFILE_BAND_M, "from_m": low, "to_m": high,
                "shares": [round(float(count) / len(points), 5) for count in counts],
                "below_share": float(np.mean(heights < low)), "above_share": float(np.mean(heights >= high))}
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
    largest: list[np.ndarray] = []
    normal = _level_direction(sample, rng, search, largest) if len(sample) >= 3 else None
    basis = "single_level_plane"
    if normal is None:
        normal = _parallel_level_direction(vertices, np.asarray(mesh.faces, dtype=np.int64), sample, search,
                                           directions=tuple(largest), camera_centres=centres * meters_per_unit)
        basis = "parallel_level_surfaces"
    value["level_search"] = search
    splat = read_standard_3dgs_ply(splat_path) if normal is not None else None
    if normal is None or splat.sh_rest is not None:
        # Rotating view-dependent colour is not implemented; refuse rather than tilt it.
        value.update(status="needs_input", reason="website_atlas_scene_level_unobserved" if normal is None
                     else "website_atlas_scene_level_splat_view_dependent", outputs=[])
        value["digest"] = canonical_digest(value, digest_field="digest")
        write_json(receipt_path, value)
        return value
    floor = _floor(sample, normal)
    camera_heights = None
    if floor is not None:
        camera_heights = centres * meters_per_unit @ floor["normal"] - floor["height_m"]
        low, high = CAMERA_MEDIAN_HEIGHT_BOUNDS_M
        if (not len(camera_heights) or camera_heights.min() < CAMERA_MIN_HEIGHT_M
                or not low <= float(np.median(camera_heights)) <= high):
            floor = None
    up = floor["normal"] if floor is not None else normal
    rotation = _rotation_to_up(up)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    if floor is not None:
        transform[1, 3] = -floor["height_m"] / meters_per_unit
    mesh.apply_transform(transform)
    root.mkdir(parents=True, exist_ok=True)
    mesh.export(str(root / MESH_NAME))
    xyz = splat.xyz.astype(np.float64) @ rotation.T + transform[:3, 3]
    quats = _rotate_quaternions(rotation, splat.quats.astype(np.float64))
    write_standard_3dgs_ply(dataclasses.replace(splat, xyz=xyz.astype(np.float32), quats=quats.astype(np.float32)),
                            root / SPLAT_NAME)
    value.update(
        status="levelled", provider_to_levelled=transform.tolist(),
        tilt_degrees=math.degrees(math.acos(min(1.0, float(up[1])))),
        up_basis="observed_floor" if floor is not None else "dominant_level_surface",
        level_direction_basis=basis,
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
