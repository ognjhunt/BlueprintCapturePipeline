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
    # Default to floor-at-min (up = +axis), the common gravity-aligned capture convention.
    # Only flip when the upper band is *clearly* denser than the lower (strong evidence the
    # dense floor plane sits at the high end); density alone is a weak signal otherwise.
    up_sign = -1.0 if top > 1.6 * bottom else 1.0
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
    task_target: Sequence[float] | None = None,
    standoff: float = 1.6,
) -> dict:
    """Find a free standing spot on the floor: a coarse 2D occupancy grid over the
    footprint, counting splats in the standing volume [floor, floor+robot_height];
    pick a low-occupancy cell as the robot start pose.

    When ``task_target`` (a 3D point, e.g. a task anchor/object) is given, the spot is
    biased toward a comfortable ``standoff`` distance from the target and the robot faces
    the target — i.e. a *task-specific* start. Without it, a central free spot is chosen.

    Returns position (3,), facing yaw, eye position, and the chosen up-axis — enough for
    the task layer to place the robot. Heuristic only; not a navigability guarantee.
    """
    h0, h1 = geom.horizontal_axes
    up = geom.up_axis
    xyz = np.asarray(splat.xyz, dtype=np.float64)
    vis = splat.opacity_sigmoid >= DEFAULT_VISIBLE_OPACITY
    finite = np.isfinite(xyz).all(axis=1)
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
    has_task = task_target is not None
    if has_task:
        tt = np.asarray(task_target, dtype=np.float64)
        tcx = (tt[h0] - lo0) / cell
        tcy = (tt[h1] - lo1) / cell
        ideal = standoff / cell
    best = None
    best_score = None
    for i in range(nx):
        for j in range(ny):
            if grid[i, j] > thresh:
                continue
            if has_task:
                # prefer free cells ~standoff from the task target, lightly central
                d_target = math.hypot(i - tcx, j - tcy)
                score = grid[i, j] * 3.0 + abs(d_target - ideal) + 0.1 * math.hypot(i - cx, j - cy)
            else:
                score = grid[i, j] * 3.0 + math.hypot(i - cx, j - cy)
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
    # facing: toward the task target when given, else toward the scene center
    look_at = np.asarray(task_target, dtype=np.float64) if has_task else geom.center
    facing = look_at - pos
    yaw = math.degrees(math.atan2(facing[h1], facing[h0]))
    return {
        "position": [float(x) for x in pos],
        "facing_yaw_deg": float(yaw),
        "task_target": [float(x) for x in tt] if has_task else None,
        "standoff_distance": float(standoff) if has_task else None,
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
    focus_point: Sequence[float] | None = None,
) -> list[dict]:
    """Produce render specs for the eval's named cameras, framed to the real scene.

    A blend of inside first-person (head_pov/torso/wrist) and elevated establishing
    (third_person/overhead/task_focus) views, all up-axis-aware. When ``focus_point``
    (a 3D task region, e.g. a task anchor/object) is given, the ``task_focus`` and
    ``wrist`` cameras aim at it for a *task-specific* view. Each spec is
    ``{"id", "spec": {"pos", "target", "fov", "up"}}`` for the headless renderer.
    """
    up = geom.up_axis
    up_sign = geom.up_sign
    h0, h1 = geom.horizontal_axes
    center = geom.center.astype(np.float64)
    focus = np.asarray(focus_point, dtype=np.float64) if focus_point is not None else center
    up_vec = [float(up_sign if a == up else 0.0) for a in range(3)]
    plane_up = [1.0 if a == h1 else 0.0 for a in range(3)]  # image-up for straight-down overhead
    fmin = geom.aabb_min.astype(np.float64)
    fmax = geom.aabb_max.astype(np.float64)
    w0 = max(float(fmax[h0] - fmin[h0]), 1e-3)
    w1 = max(float(fmax[h1] - fmin[h1]), 1e-3)
    lo0, hi0, lo1, hi1 = float(fmin[h0]), float(fmax[h0]), float(fmin[h1]), float(fmax[h1])
    c0, c1 = float(center[h0]), float(center[h1])
    f0, f1 = float(focus[h0]), float(focus[h1])
    height = max(abs(geom.ceiling - geom.floor), 1e-3)
    eye = geom.floor + up_sign * min(DEFAULT_EYE_HEIGHT, max(0.5 * height, 0.6))

    def P(a0: float, a1: float, u: float) -> np.ndarray:
        v = np.zeros(3, dtype=np.float64)
        v[h0], v[h1], v[up] = a0, a1, u
        return v

    # INTERIOR-FIRST framing: stations stand INSIDE the footprint looking across the room,
    # because interior captures read clearly from within, not from an outside orbit.
    # head_pov: near one wall at eye height, looking across to the far wall.
    head = P(lo0 + 0.22 * w0, c1, eye)
    head_t = P(hi0 - 0.04 * w0, c1, eye - up_sign * 0.06 * height)
    # torso: near the orthogonal wall, chest height, looking across the other axis.
    torso = P(c0, lo1 + 0.22 * w1, eye - up_sign * 0.12)
    torso_t = P(c0, hi1 - 0.04 * w1, eye - up_sign * 0.08 * height)
    # third_person: stand in an interior corner, eye-or-higher, look diagonally across the
    # whole room — a wide *interior* establishing shot (not an exterior dollhouse).
    tp = P(lo0 + 0.16 * w0, lo1 + 0.16 * w1, geom.floor + up_sign * min(0.8 * height, 2.0))
    tp_t = P(hi0 - 0.2 * w0, hi1 - 0.2 * w1, geom.floor + up_sign * 0.4 * height)
    # wrist: low, near the task/contact zone, looking at the focus point.
    wrist = P(f0 - 0.16 * w0, f1 - 0.16 * w1, geom.floor + up_sign * 0.7)
    wrist_t = P(f0, f1, geom.floor + up_sign * 0.25)
    # overhead: just under the ceiling at center, straight down — interior top-down layout.
    over = P(c0, c1, geom.ceiling - up_sign * 0.1 * height)
    over_t = P(c0, c1, geom.floor)
    # task_focus: stand a standoff back from the focus (toward room center), look at it.
    standoff = 0.32 * max(w0, w1)
    dx, dy = c0 - f0, c1 - f1
    norm = math.hypot(dx, dy)
    if norm < 1e-6:  # focus at center -> back toward an interior corner instead of degenerate
        dx, dy, norm = -1.0, -1.0, math.sqrt(2.0)
    tf = P(f0 + dx / norm * standoff, f1 + dy / norm * standoff, eye - up_sign * 0.1)

    def spec(pos: np.ndarray, target: np.ndarray, fov: float, up_v: list = up_vec) -> dict:
        return {
            "pos": [float(x) for x in pos],
            "target": [float(x) for x in target],
            "fov": fov,
            "up": up_v,
        }

    specs = {
        "head_pov": spec(head, head_t, 70),
        "torso": spec(torso, torso_t, 66),
        "wrist": spec(wrist, wrist_t, 60),
        "third_person": spec(tp, tp_t, 66),
        "overhead": spec(over, over_t, 72, plane_up),
        "task_focus": spec(tf, focus, 52),
    }
    out = []
    for cid in camera_ids:
        out.append({"id": cid, "spec": specs.get(cid) or spec(head, head_t, 64)})
    return out
