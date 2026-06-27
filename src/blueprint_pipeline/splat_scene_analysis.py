"""Derive reusable scene geometry from a Gaussian-splat capture.

A captured splat scene has no metadata about which way is up, where the floor is, or
where a robot could stand. This module recovers that geometry from the splat centers so
the *same* analysis feeds both:

* camera framing — :func:`derive_eval_cameras` produces the eval's 6 named viewpoints
  framed to the real scene (interior-aware, up-axis-aware), and
* robot placement — :func:`suggest_robot_start` returns a free-floor standing pose for a
  task start point.

Everything is heuristic geometry on splat centers; it claims no physics, navigation, or
task-correctness. Up-axis and floor are estimated and can be overridden by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .gaussian_splat_decode import SplatData

DEFAULT_CAMERA_IDS = (
    "head_pov", "torso", "wrist", "third_person", "overhead", "task_focus",
)
DEFAULT_EYE_HEIGHT = 1.5      # metres-ish above floor for first-person cameras
DEFAULT_VISIBLE_OPACITY = 0.18


@dataclass
class SceneGeometry:
    up_axis: int                 # 0/1/2 world axis that is "up"
    up_sign: float               # +1 or -1 along up_axis
    horizontal_axes: tuple[int, int]
    floor: float                 # up-coordinate of the floor
    ceiling: float               # up-coordinate of the ceiling
    center: np.ndarray           # robust scene center (3,), up-coord at mid height
    aabb_min: np.ndarray         # robust AABB (3,)
    aabb_max: np.ndarray
    radius: float                # robust horizontal radius
    splat_count: int
    visible_count: int
    suggested_start: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def up_vector(self) -> np.ndarray:
        v = np.zeros(3, dtype=np.float64)
        v[self.up_axis] = self.up_sign
        return v

    def to_dict(self) -> dict:
        return {
            "up_axis": int(self.up_axis),
            "up_sign": float(self.up_sign),
            "horizontal_axes": list(self.horizontal_axes),
            "floor": float(self.floor),
            "ceiling": float(self.ceiling),
            "center": [float(x) for x in self.center],
            "aabb_min": [float(x) for x in self.aabb_min],
            "aabb_max": [float(x) for x in self.aabb_max],
            "radius": float(self.radius),
            "splat_count": int(self.splat_count),
            "visible_count": int(self.visible_count),
            "suggested_start": self.suggested_start,
            "diagnostics": self.diagnostics,
        }


def _robust_minmax(values: np.ndarray, pct: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(values, pct, axis=0)
    hi = np.percentile(values, 100.0 - pct, axis=0)
    return lo, hi


def analyze_scene(
    splat: SplatData,
    *,
    up_axis: int | None = None,
    visible_opacity: float = DEFAULT_VISIBLE_OPACITY,
    percentile: float = 1.0,
) -> SceneGeometry:
    """Estimate scene geometry from splat centers.

    up-axis heuristic: the world axis with the smallest *robust* extent is treated as
    vertical (room height < floor dimensions). Override with ``up_axis`` when known.
    """
    xyz = np.asarray(splat.xyz, dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    vis = splat.opacity_sigmoid >= visible_opacity
    mask = finite & vis
    if mask.sum() < 16:  # too few visible -> fall back to all finite splats
        mask = finite
    pts = xyz[mask]
    if pts.shape[0] == 0:
        raise ValueError("no finite splat centers to analyze")

    lo, hi = _robust_minmax(pts, percentile)
    extents = hi - lo

    if up_axis is None:
        up_axis = int(np.argmin(extents))
    horizontal = tuple(a for a in range(3) if a != up_axis)

    up_vals = pts[:, up_axis]
    up_lo = float(np.percentile(up_vals, percentile))
    up_hi = float(np.percentile(up_vals, 100.0 - percentile))
    # up_sign: the floor side is denser (floor + objects rest on it). Compare the density
    # of the bottom vs top 20% of the up range; "down" points toward the denser side.
    span = max(up_hi - up_lo, 1e-6)
    bottom = float((up_vals <= up_lo + 0.2 * span).mean())
    top = float((up_vals >= up_hi - 0.2 * span).mean())
    up_sign = 1.0 if bottom >= top else -1.0
    floor, ceiling = (up_lo, up_hi) if up_sign > 0 else (up_hi, up_lo)

    median = np.median(pts, axis=0)
    center = median.copy()
    center[up_axis] = 0.5 * (floor + ceiling)

    h0, h1 = horizontal
    foot_w = float(hi[h0] - lo[h0])
    foot_d = float(hi[h1] - lo[h1])
    radius = 0.5 * math.hypot(foot_w, foot_d)

    geom = SceneGeometry(
        up_axis=up_axis,
        up_sign=up_sign,
        horizontal_axes=horizontal,
        floor=floor,
        ceiling=ceiling,
        center=center,
        aabb_min=lo,
        aabb_max=hi,
        radius=radius,
        splat_count=int(splat.count),
        visible_count=int(mask.sum()),
        diagnostics={
            "extents": [float(x) for x in extents],
            "up_axis_extent": float(extents[up_axis]),
            "floor_density_bottom": bottom,
            "floor_density_top": top,
            "percentile": percentile,
            "visible_opacity": visible_opacity,
        },
    )
    geom.suggested_start = suggest_robot_start(splat, geom)
    return geom


def suggest_robot_start(
    splat: SplatData,
    geom: SceneGeometry,
    *,
    robot_height: float = 1.3,
    cell: float | None = None,
    clearance_radius: float = 0.45,
) -> dict:
    """Find a free standing spot on the floor: a coarse 2D occupancy grid over the
    footprint, counting splats in the standing volume [floor, floor+robot_height];
    pick the lowest-occupancy cell nearest the center as the robot start pose.

    Returns position (3,), facing yaw toward center, and the chosen up-axis — enough for
    the task layer to place the robot. Heuristic only; not a navigability guarantee.
    """
    h0, h1 = geom.horizontal_axes
    up = geom.up_axis
    xyz = np.asarray(splat.xyz, dtype=np.float64)
    vis = splat.opacity_sigmoid >= DEFAULT_VISIBLE_OPACITY
    finite = np.isfinite(xyz).all(axis=1)
    in_band = (
        (xyz[:, up] >= geom.floor + 0.05 * geom.up_sign * 0)  # band starts at floor
    )
    # standing band along up-axis between floor and floor+robot_height (sign-aware)
    band_lo = min(geom.floor, geom.floor + geom.up_sign * robot_height)
    band_hi = max(geom.floor, geom.floor + geom.up_sign * robot_height)
    band = (xyz[:, up] >= band_lo) & (xyz[:, up] <= band_hi)
    pts = xyz[finite & vis & band]

    lo0, hi0 = geom.aabb_min[h0], geom.aabb_max[h0]
    lo1, hi1 = geom.aabb_min[h1], geom.aabb_max[h1]
    extent = max(hi0 - lo0, hi1 - lo1, 1e-3)
    if cell is None:
        cell = max(extent / 24.0, 0.15)
    nx = max(1, int(math.ceil((hi0 - lo0) / cell)))
    ny = max(1, int(math.ceil((hi1 - lo1) / cell)))
    grid = np.zeros((nx, ny), dtype=np.int64)
    if pts.shape[0]:
        ix = np.clip(((pts[:, h0] - lo0) / cell).astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, h1] - lo1) / cell).astype(int), 0, ny - 1)
        np.add.at(grid, (ix, iy), 1)

    # candidate cells: occupancy below a low threshold, away from the footprint edge
    thresh = max(1, int(np.percentile(grid, 35)))
    cx = (geom.center[h0] - lo0) / cell
    cy = (geom.center[h1] - lo1) / cell
    best = None
    best_score = None
    for i in range(nx):
        for j in range(ny):
            if grid[i, j] > thresh:
                continue
            # distance to center (prefer central, open spots)
            d = math.hypot(i - cx, j - cy)
            score = grid[i, j] * 3.0 + d
            if best_score is None or score < best_score:
                best_score = score
                best = (i, j)
    if best is None:
        best = (int(round(cx)), int(round(cy)))
    bi, bj = best
    pos = np.zeros(3, dtype=np.float64)
    pos[h0] = lo0 + (bi + 0.5) * cell
    pos[h1] = lo1 + (bj + 0.5) * cell
    pos[up] = geom.floor
    # facing: yaw toward center in the horizontal plane
    facing = geom.center - pos
    yaw = math.degrees(math.atan2(facing[h1], facing[h0]))
    return {
        "position": [float(x) for x in pos],
        "facing_yaw_deg": float(yaw),
        "up_axis": int(up),
        "eye_position": [
            float(pos[a] + (DEFAULT_EYE_HEIGHT * geom.up_sign if a == up else 0.0))
            for a in range(3)
        ],
        "grid_cells": [int(nx), int(ny)],
        "grid_cell_size": float(cell),
        "occupancy_threshold": int(thresh),
        "claim_boundary": "free_floor_heuristic_only_not_navigability_proof",
    }


def _orbit_position(
    center: np.ndarray,
    up_axis: int,
    horizontal: tuple[int, int],
    azimuth_deg: float,
    elevation_deg: float,
    radius: float,
    up_sign: float,
) -> list[float]:
    h0, h1 = horizontal
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    pos = center.astype(np.float64).copy()
    pos[h0] += radius * math.cos(el) * math.cos(az)
    pos[h1] += radius * math.cos(el) * math.sin(az)
    pos[up_axis] += up_sign * radius * math.sin(el)
    return [float(x) for x in pos]


def derive_eval_cameras(
    geom: SceneGeometry,
    camera_ids: Sequence[str] = DEFAULT_CAMERA_IDS,
) -> list[dict]:
    """Produce render specs for the eval's named cameras, framed to the real scene.

    A blend of inside first-person (head_pov/torso/wrist) and elevated establishing
    (third_person/overhead/task_focus) views, all up-axis-aware. Each spec is
    ``{"id", "spec": {"pos", "target", "fov", "up"}}`` for the headless renderer.
    """
    up = geom.up_axis
    up_sign = geom.up_sign
    h0, h1 = geom.horizontal_axes
    center = geom.center.astype(np.float64)
    radius = max(geom.radius, 1e-3)
    up_vec = [float(geom.up_sign if a == up else 0.0) for a in range(3)]
    # an in-plane vector to use as image-up for a straight-down overhead
    plane_up = [1.0 if a == h1 else 0.0 for a in range(3)]
    height = abs(geom.ceiling - geom.floor)
    eye = geom.floor + up_sign * min(DEFAULT_EYE_HEIGHT, max(0.4 * height, 0.6))

    start = geom.suggested_start.get("position") or [float(x) for x in center]
    start = np.asarray(start, dtype=np.float64)

    def eye_pos(base: np.ndarray, e: float) -> list[float]:
        p = base.astype(np.float64).copy()
        p[up] = e
        return [float(x) for x in p]

    # head_pov: stand at the free start spot, eye height, look across the room at center
    head = eye_pos(start, eye)
    head_target = center.copy()
    head_target[up] = eye
    # torso: a bit behind/lower than head, wider fov
    torso_base = start + (center - start) * (-0.18)
    torso = eye_pos(torso_base, eye - up_sign * 0.25)
    torso_target = center.copy()
    torso_target[up] = eye - up_sign * 0.2
    # wrist: low, near the floor close to center (task/contact zone)
    wrist_base = start + (center - start) * 0.45
    wrist = eye_pos(wrist_base, geom.floor + up_sign * 0.55)
    wrist_target = center.copy()
    wrist_target[up] = geom.floor + up_sign * 0.25

    specs = {
        "head_pov": {"pos": head, "target": [float(x) for x in head_target], "fov": 70, "up": up_vec},
        "torso": {"pos": torso, "target": [float(x) for x in torso_target], "fov": 64, "up": up_vec},
        "wrist": {"pos": wrist, "target": [float(x) for x in wrist_target], "fov": 60, "up": up_vec},
        "third_person": {
            "pos": _orbit_position(center, up, (h0, h1), 225, 22, radius * 1.7, up_sign),
            "target": [float(x) for x in center], "fov": 52, "up": up_vec,
        },
        "overhead": {
            "pos": eye_pos(center, geom.ceiling + up_sign * max(0.9 * height, radius * 0.6)),
            "target": [float(x) for x in center], "fov": 62, "up": plane_up,
        },
        "task_focus": {
            "pos": _orbit_position(center, up, (h0, h1), 60, 28, radius * 1.35, up_sign),
            "target": [float(x) for x in (center + 0.0)], "fov": 48, "up": up_vec,
        },
    }
    out = []
    for cid in camera_ids:
        spec = specs.get(cid)
        if spec is None:
            spec = {
                "pos": _orbit_position(center, up, (h0, h1), 200, 20, radius * 1.6, up_sign),
                "target": [float(x) for x in center], "fov": 55, "up": up_vec,
            }
        out.append({"id": cid, "spec": spec})
    return out
