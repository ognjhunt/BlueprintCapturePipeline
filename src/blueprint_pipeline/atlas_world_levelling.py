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

REVISION = "atlas_world_levelling.v1"
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


def _level_direction(points: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """Unit normal of the dominant near-level plane, refined on all its points; None when there is none."""
    limit = math.cos(math.radians(MAX_TILT_DEGREES))
    probe = points[rng.choice(len(points), RANSAC_PROBE, replace=False)] if len(points) > RANSAC_PROBE else points
    best, best_count = None, int(MIN_LEVEL_SHARE * len(probe))
    for _ in range(RANSAC_ITERATIONS):
        a, b, c = probe[rng.choice(len(probe), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length <= 1e-12 or abs(normal[1]) < limit * length:
            continue
        normal = normal / length
        count = int(np.count_nonzero(np.abs((probe - a) @ normal) <= PLANE_TOLERANCE_M))
        if count > best_count:
            best, best_count = (a, normal), count
    if best is None:
        return None
    normal, _ = _plane(points[np.abs((points - best[0]) @ best[1]) <= PLANE_TOLERANCE_M])
    return normal if normal[1] >= limit else None


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
    normal = _level_direction(sample, rng) if len(sample) >= 3 else None
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
