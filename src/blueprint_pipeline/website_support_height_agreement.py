"""Where is the table: the height the physics rests the task on against the height the cameras see (CPU).

At the end of preparation, under the object's start footprint and under the destination area, the
support's height is read three ways on a 2 cm grid:

- by physics: the highest collider triangle at each point (the room collider, simulator metres);
- in the picture: where a vertical ray from above has met half its opacity in the room's particle
  field (the same order-independent accumulation the render uses: ``1 - prod(1 - alpha)`` over
  each particle's peak response along the ray);
- in the site's own depth, where it exists: the median height of the support's masked, lifted
  points near each grid point (estimated depth through the registration, so it carries the
  registration's residual).

Recorded per region: the medians, and the gap collider - picture and collider - depth with their
10th and 90th percentiles. The task plane (where the object is seated: the highest collider point
under its start footprint) is compared with the picture's median there.

The flag: ``scene_estimate:support_height_differs_from_picture`` when the object's resting height
(the task plane) is more than ``FLAG_GAP_M`` (2 cm) above or below the picture's table under the
object or at the goal. It is carried to the publication like the other estimate flags
(``atlas_scene_estimates.preparation_flags``). Measurement only: nothing is corrected. Never a stop.
Whether the gap is one flat offset (the collider - picture spread, and the two regions' agreement)
is recorded beside it.

capture-447ae4f4 (2026-11-12, its retained collider, splat and depth): the raw collider sits a
median 4.3 mm (start) and 4.8 mm (goal) above the picture's table, spread 1.1 and 1.0 cm; the
picture's two regions differ by 4.9 mm; the site's depth puts the table 2.3 and 1.6 cm under the
collider. The task plane (0.8039 m, the highest collider point under the tray) is 1.2 cm above the
picture's table under the tray (0.7921 m) and 1.7 cm above it at the goal: no flag.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SCHEMA_VERSION = "website_support_height_agreement.v1"
GRID_STEP_M = 0.02
CONSISTENT_SPREAD_M = 0.015
REGIONS_AGREE_M = 0.01
MINIMUM_MEASURED_SHARE = 0.5
#: The resting height further than this from the picture's table, under the object or at the goal, is flagged.
FLAG_GAP_M = 0.02
FLAG = "scene_estimate:support_height_differs_from_picture"


def start_grid(minimum_xy: Sequence[float], maximum_xy: Sequence[float]) -> np.ndarray:
    xs = np.arange(minimum_xy[0], maximum_xy[0], GRID_STEP_M) + GRID_STEP_M / 2
    ys = np.arange(minimum_xy[1], maximum_xy[1], GRID_STEP_M) + GRID_STEP_M / 2
    return np.array([[x, y] for x in xs for y in ys], float).reshape(-1, 2)


def disc_grid(centre_xy: Sequence[float], radius_m: float, *, exclude: tuple | None = None) -> np.ndarray:
    steps = np.arange(-radius_m, radius_m + 1e-9, GRID_STEP_M)
    grid = np.array([[centre_xy[0] + x, centre_xy[1] + y] for x in steps for y in steps if x * x + y * y <= radius_m ** 2],
                    float).reshape(-1, 2)
    if exclude is not None:
        low, high = np.asarray(exclude[0], float), np.asarray(exclude[1], float)
        grid = grid[~np.all((grid >= low) & (grid <= high), axis=1)]
    return grid


def collider_heights(triangles: Any, xy: np.ndarray, *, band: Sequence[float]) -> np.ndarray:
    """The highest collider triangle at each point, among triangles reaching into ``band`` (metres)."""
    tris = np.asarray(triangles, float).reshape(-1, 3, 3)
    tris = tris[(tris[:, :, 2].max(axis=1) >= band[0]) & (tris[:, :, 2].min(axis=1) <= band[1])]
    out = np.full(len(xy), np.nan)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    e0, e1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    den = e0[:, 0] * e1[:, 1] - e1[:, 0] * e0[:, 1]
    ok = np.abs(den) > 1e-12
    for index, point in enumerate(xy):
        rel = point - a[:, :2]
        u = np.where(ok, (rel[:, 0] * e1[:, 1] - e1[:, 0] * rel[:, 1]) / np.where(ok, den, 1.0), -1.0)
        w = np.where(ok, (e0[:, 0] * rel[:, 1] - rel[:, 0] * e0[:, 1]) / np.where(ok, den, 1.0), -1.0)
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
        if hit.any():
            z = a[hit, 2] + u[hit] * (b[hit, 2] - a[hit, 2]) + w[hit] * (c[hit, 2] - a[hit, 2])
            out[index] = float(z.max())
    return out


def particle_heights(positions: Any, inverse_covariances: Any, opacities: Any, xy: np.ndarray, *,
                     top_z: float, reach_m: float = 0.08) -> np.ndarray:
    """Where a vertical ray from ``top_z`` down meets half its opacity, at each point (NaN when it never does)."""
    pos, inv, alpha0 = (np.asarray(positions, float), np.asarray(inverse_covariances, float),
                        np.asarray(opacities, float))
    out = np.full(len(xy), np.nan)
    below = pos[:, 2] < top_z
    for index, point in enumerate(xy):
        idx = np.nonzero(below & (np.abs(pos[:, 0] - point[0]) < reach_m) & (np.abs(pos[:, 1] - point[1]) < reach_m))[0]
        if not len(idx):
            continue
        m = pos[idx] - np.array([point[0], point[1], top_z])
        s = inv[idx]
        dsd = s[:, 2, 2]
        msd = -np.einsum("ni,ni->n", m, s[:, :, 2])
        msm = np.einsum("ni,nij,nj->n", m, s, m)
        t = msd / dsd
        alpha = np.minimum(0.99, alpha0[idx] * np.exp(-0.5 * (msm - msd * msd / dsd)))
        order = np.argsort(t)
        accumulated = 1.0 - np.cumprod(1.0 - alpha[order])
        hit = np.argmax(accumulated >= 0.5)
        if accumulated[hit] >= 0.5:
            out[index] = top_z - float(t[order][hit])
    return out


def depth_heights(points: Any, xy: np.ndarray, *, cell_m: float = 0.03, minimum_points: int = 5) -> np.ndarray:
    pts = np.asarray(points, float).reshape(-1, 3)
    out = np.full(len(xy), np.nan)
    for index, point in enumerate(xy):
        near = (np.abs(pts[:, 0] - point[0]) < cell_m) & (np.abs(pts[:, 1] - point[1]) < cell_m)
        if near.sum() >= minimum_points:
            out[index] = float(np.median(pts[near, 2]))
    return out


def _spread(values: np.ndarray) -> dict[str, Any] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return {"n": int(len(finite)), "p10_m": float(np.percentile(finite, 10)), "median_m": float(np.median(finite)),
            "p90_m": float(np.percentile(finite, 90))}


def region_record(collider: np.ndarray, picture: np.ndarray, depth: np.ndarray | None) -> dict[str, Any]:
    both = np.isfinite(collider) & np.isfinite(picture)
    record = {"points": int(len(collider)), "measured_share": float(both.mean()) if len(collider) else 0.0,
              "collider_median_z_m": float(np.nanmedian(collider)) if np.isfinite(collider).any() else None,
              "picture_median_z_m": float(np.nanmedian(picture)) if np.isfinite(picture).any() else None,
              "gap_collider_minus_picture": _spread(collider - picture)}
    if depth is not None:
        record["depth_median_z_m"] = float(np.nanmedian(depth)) if np.isfinite(depth).any() else None
        record["gap_collider_minus_depth"] = _spread(collider - depth)
    return record


def agreement(*, start: Mapping[str, Any], goal: Mapping[str, Any], task_plane_z_m: float | None) -> dict[str, Any]:
    """The record, its consistency and its flag, from the two regions' readings."""
    reasons = []
    for name, region in (("start", start), ("goal", goal)):
        gap = region.get("gap_collider_minus_picture")
        if region.get("measured_share", 0.0) < MINIMUM_MEASURED_SHARE or gap is None:
            reasons.append(f"{name}_not_measured")
        elif gap["p90_m"] - gap["p10_m"] > CONSISTENT_SPREAD_M:
            reasons.append(f"{name}_ragged")
    heights = [region.get("picture_median_z_m") for region in (start, goal)]
    if all(isinstance(value, float) for value in heights) and abs(heights[0] - heights[1]) > REGIONS_AGREE_M:
        reasons.append("regions_disagree")
    consistent = not reasons
    record: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "start": dict(start), "goal": dict(goal),
                              "one_flat_offset": consistent, "not_one_flat_offset_because": reasons, "flags": [],
                              "thresholds": {"flag_gap_m": FLAG_GAP_M, "consistent_spread_m": CONSISTENT_SPREAD_M,
                                             "regions_agree_m": REGIONS_AGREE_M,
                                             "minimum_measured_share": MINIMUM_MEASURED_SHARE},
                              "correction_applied": False, "stops_preparation": False}
    if task_plane_z_m is not None:
        record["task_plane_z_m"] = float(task_plane_z_m)
        for name, value in (("start", heights[0]), ("goal", heights[1])):
            if isinstance(value, float):
                record[f"{name}_task_plane_minus_picture_m"] = float(task_plane_z_m) - value
        if any(abs(record.get(f"{name}_task_plane_minus_picture_m", 0.0)) > FLAG_GAP_M for name in ("start", "goal")):
            record["flags"] = [FLAG]
    return record

def particle_field(splat_path: Any, runtime_to_simulator: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A standard 3DGS splat (runtime frame) as simulator positions, inverse covariances and opacities."""
    from .gaussian_splat_decode import read_standard_3dgs_ply

    splat = read_standard_3dgs_ply(splat_path)
    matrix = np.asarray(runtime_to_simulator, float)
    linear = matrix[:3, :3]
    positions = np.asarray(splat.xyz, float) @ linear.T + matrix[:3, 3]
    q = np.asarray(splat.quats, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotation = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                         np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                         np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    variance = np.exp(2.0 * np.asarray(splat.scales, float))
    covariance = np.einsum("ij,njk,nk,nlk,ml->nim", linear, rotation, variance, rotation, linear)
    inverse = np.linalg.inv(covariance + 1e-10 * np.eye(3))
    return positions, inverse, np.asarray(splat.opacity_sigmoid, float)


def measure(*, triangles: Any, positions: Any, inverse_covariances: Any, opacities: Any,
            start_minimum_xy: Sequence[float], start_maximum_xy: Sequence[float], goal_centre_xy: Sequence[float],
            goal_radius_m: float, task_plane_z_m: float, depth_points: Any = None) -> dict[str, Any]:
    """The record for one task: both regions read three ways, compared and flagged."""
    band = (task_plane_z_m - 0.15, task_plane_z_m + 0.15)
    regions = {"start": start_grid(start_minimum_xy, start_maximum_xy),
               "goal": disc_grid(goal_centre_xy, goal_radius_m, exclude=(start_minimum_xy, start_maximum_xy))}
    # Only particles near the two regions and the plane can meet these rays: filter once.
    pos = np.asarray(positions, float)
    every = np.vstack(list(regions.values()))
    keep = ((pos[:, 0] >= every[:, 0].min() - 0.1) & (pos[:, 0] <= every[:, 0].max() + 0.1)
            & (pos[:, 1] >= every[:, 1].min() - 0.1) & (pos[:, 1] <= every[:, 1].max() + 0.1)
            & (pos[:, 2] >= band[0] - 0.1) & (pos[:, 2] <= band[1]))
    positions, inverse_covariances = pos[keep], np.asarray(inverse_covariances, float)[keep]
    opacities = np.asarray(opacities, float)[keep]
    readings = {}
    for name, grid in regions.items():
        collider = collider_heights(triangles, grid, band=band)
        picture = particle_heights(positions, inverse_covariances, opacities, grid, top_z=band[1])
        depth = None if depth_points is None else depth_heights(depth_points, grid)
        readings[name] = region_record(collider, picture, depth)
    return agreement(start=readings["start"], goal=readings["goal"], task_plane_z_m=task_plane_z_m)


def measure_preparation(preparation: Mapping[str, Any], base_scene: Mapping[str, Any], *,
                        depth_points: Any = None, goal_radius_m: float | None = None) -> dict[str, Any]:
    """The record for a website preparation, from its own collider, splat and placement (flag-only)."""
    import trimesh

    matrix = np.asarray(preparation["coordinate_frame"]["runtime_to_simulator"], float)
    mesh = trimesh.load(base_scene["collision_mesh_path"], force="mesh", process=False)
    vertices = np.c_[np.asarray(mesh.vertices, float), np.ones(len(mesh.vertices))] @ matrix.T
    positions, inverse, opacities = particle_field(base_scene["splat_path"], matrix)
    subject, destination = preparation["subject"], preparation["destination"]
    record = measure(triangles=vertices[:, :3][np.asarray(mesh.faces)], positions=positions,
                     inverse_covariances=inverse, opacities=opacities,
                     start_minimum_xy=subject["aabb_min_xyz"][:2], start_maximum_xy=subject["aabb_max_xyz"][:2],
                     goal_centre_xy=destination["position_world_m"][:2],
                     goal_radius_m=float(goal_radius_m or destination.get("radius_m") or 0.15),
                     task_plane_z_m=float(subject["aabb_min_xyz"][2]), depth_points=depth_points)
    return record


def preparation_record(preparation: Mapping[str, Any], *, base_scene: Mapping[str, Any],
                             masks: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The support's height by physics against the picture and the site's depth (``website_support_height_agreement``).

    Flags only: a measurement that cannot run is recorded as not assessed, never a stop.
    """
    from .website_task_masks import masked_source_points

    try:
        depth = None
        rows = [row for row in masks.get("targets") or [] if row.get("target_role") == "destination" and row.get("track")]
        if rows:
            try:
                points = masked_source_points(rows[0]["track"], list(frames))
                matrix = (np.asarray(preparation["coordinate_frame"]["runtime_to_simulator"], float)
                          @ np.asarray(preparation["registration"]["source_to_runtime"], float))
                depth = (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :3]
            except Exception:  # noqa: BLE001 - the site's depth is optional evidence
                depth = None
        return measure_preparation(preparation, base_scene, depth_points=depth)
    except Exception as exc:  # noqa: BLE001 - a flag-only measurement must never stop a job
        return {"status": "not_assessed", "reason": type(exc).__name__ + ":" + str(exc)[:200], "flags": [],
                "stops_preparation": False}


__all__ = ["preparation_record", "measure", "measure_preparation", "particle_field", "CONSISTENT_SPREAD_M", "FLAG", "REGIONS_AGREE_M", "SCHEMA_VERSION", "FLAG_GAP_M", "agreement",
           "collider_heights", "depth_heights", "disc_grid", "particle_heights", "region_record", "start_grid"]
