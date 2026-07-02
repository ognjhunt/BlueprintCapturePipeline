"""Carve coarse label AABBs into splat-backed occupancy columns (CPU only).

InteriorGS label boxes are honest about WHERE an object is but coarse about its
SHAPE: an L-shaped kitchen cabinetry run is labeled with one axis-aligned box
that swallows the aisle in front of it, so every stance in that kitchen reads as
"clipping the cupboard" even though the floor is open. The splat knows better —
gaussian centers pile up where the cabinetry actually is and are absent over the
open floor.

This module rasterizes the decoded splat centers into a 2D floor-occupancy grid
(body-height z band, opacity-weighted) and replaces each JUMBO label box with
the set of occupied grid columns inside it. The result is a drop-in obstacle
catalog for the placement probe and validator: small objects keep their exact
label boxes; only oversized boxes are refined, and only where the splat actually
has mass. When the splat says a jumbo box is dense everywhere, the refinement
honestly returns near-identical geometry.

Truth boundary: occupancy columns describe where reconstruction mass sits; they
are not collision meshes. They tighten placement geometry — they do not prove
physics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .gaussian_splat_decode import SplatData
from .scene_placement.types import SceneObject

# xy footprint area (m^2) above which a label box is "jumbo" — suspect of being a
# coarse hull around a non-convex fixture — and eligible for splat refinement.
DEFAULT_JUMBO_AREA_M2 = 2.5
DEFAULT_CELL_M = 0.15
# Occupancy z band above the floor: skip floor splats, stop below the ceiling so
# lamps/beams do not paint phantom floor obstacles.
DEFAULT_Z_BAND = (0.05, 1.30)
DEFAULT_MIN_OPACITY = 0.30
# Minimum opacity-weighted splat count for a cell to count as occupied. Splat
# density varies per capture; this is deliberately conservative (a couple of
# solid splats = occupied) so refinement never hallucinates free space inside
# real furniture.
DEFAULT_MIN_CELL_WEIGHT = 3.0


@dataclass
class FloorOccupancyGrid:
    """Opacity-weighted splat mass per floor cell within a body-height z band."""

    origin_xy: Tuple[float, float]
    cell_m: float
    weights: np.ndarray  # (nx, ny) float32
    z_band: Tuple[float, float]

    def cell_index(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor((x - self.origin_xy[0]) / self.cell_m)),
            int(math.floor((y - self.origin_xy[1]) / self.cell_m)),
        )

    def occupied_mask(self, min_weight: float = DEFAULT_MIN_CELL_WEIGHT) -> np.ndarray:
        return self.weights >= float(min_weight)


def build_floor_occupancy_grid(
    splat: SplatData,
    *,
    floor_z: float,
    cell_m: float = DEFAULT_CELL_M,
    z_band: Tuple[float, float] = DEFAULT_Z_BAND,
    min_opacity: float = DEFAULT_MIN_OPACITY,
) -> FloorOccupancyGrid:
    """Rasterize splat centers into an xy grid of opacity-weighted mass.

    Only centers inside ``floor_z + z_band`` (the band a standing robot's body
    sweeps) and with sigmoid opacity above ``min_opacity`` contribute — floor
    splats, ceiling fixtures, and transparent reconstruction fuzz do not create
    phantom obstacles.
    """
    xyz = splat.xyz
    opacity = splat.opacity_sigmoid
    z_lo = floor_z + float(z_band[0])
    z_hi = floor_z + float(z_band[1])
    keep = (xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi) & (opacity >= float(min_opacity))
    pts = xyz[keep]
    wts = opacity[keep]
    if pts.shape[0] == 0:
        return FloorOccupancyGrid(
            origin_xy=(0.0, 0.0),
            cell_m=float(cell_m),
            weights=np.zeros((1, 1), dtype=np.float32),
            z_band=(z_lo, z_hi),
        )
    origin = (float(pts[:, 0].min()), float(pts[:, 1].min()))
    nx = int(math.floor((float(pts[:, 0].max()) - origin[0]) / cell_m)) + 1
    ny = int(math.floor((float(pts[:, 1].max()) - origin[1]) / cell_m)) + 1
    ix = np.clip(((pts[:, 0] - origin[0]) / cell_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - origin[1]) / cell_m).astype(np.int64), 0, ny - 1)
    weights = np.zeros((nx, ny), dtype=np.float32)
    np.add.at(weights, (ix, iy), wts.astype(np.float32))
    return FloorOccupancyGrid(
        origin_xy=origin, cell_m=float(cell_m), weights=weights, z_band=(z_lo, z_hi)
    )


def _column_boxes_for_object(
    obj: SceneObject,
    grid: FloorOccupancyGrid,
    splat: SplatData,
    *,
    min_cell_weight: float,
) -> List[SceneObject]:
    """Occupied grid columns inside ``obj``'s xy box, as small SceneObjects.

    Column z extents come from the actual splat centers in that column (clipped
    to the object's own z span), so a low counter column stays low even when the
    label box also spans an upper cabinet.
    """
    cell = grid.cell_m
    ox, oy = grid.origin_xy
    x0 = max(int(math.floor((obj.bbox_min[0] - ox) / cell)), 0)
    y0 = max(int(math.floor((obj.bbox_min[1] - oy) / cell)), 0)
    x1 = min(int(math.floor((obj.bbox_max[0] - ox) / cell)), grid.weights.shape[0] - 1)
    y1 = min(int(math.floor((obj.bbox_max[1] - oy) / cell)), grid.weights.shape[1] - 1)
    if x1 < x0 or y1 < y0:
        return []
    sub = grid.weights[x0 : x1 + 1, y0 : y1 + 1]
    occupied = np.argwhere(sub >= float(min_cell_weight))
    if occupied.size == 0:
        return []

    # z extents per column from the splat centers falling in that column.
    xyz = splat.xyz
    opacity = splat.opacity_sigmoid
    in_band = (
        (xyz[:, 2] >= grid.z_band[0])
        & (xyz[:, 2] <= grid.z_band[1])
        & (opacity >= DEFAULT_MIN_OPACITY)
        & (xyz[:, 0] >= obj.bbox_min[0])
        & (xyz[:, 0] <= obj.bbox_max[0])
        & (xyz[:, 1] >= obj.bbox_min[1])
        & (xyz[:, 1] <= obj.bbox_max[1])
    )
    pts = xyz[in_band]
    col_ix = ((pts[:, 0] - ox) / cell).astype(np.int64) - x0
    col_iy = ((pts[:, 1] - oy) / cell).astype(np.int64) - y0

    boxes: List[SceneObject] = []
    for n, (ci, cj) in enumerate(occupied):
        cx0 = ox + (x0 + int(ci)) * cell
        cy0 = oy + (y0 + int(cj)) * cell
        bbox_min_x = max(cx0, obj.bbox_min[0])
        bbox_min_y = max(cy0, obj.bbox_min[1])
        bbox_max_x = min(cx0 + cell, obj.bbox_max[0])
        bbox_max_y = min(cy0 + cell, obj.bbox_max[1])
        mask = (col_ix == int(ci)) & (col_iy == int(cj))
        if mask.any():
            z_vals = pts[mask][:, 2]
            z_min = max(float(z_vals.min()), obj.bbox_min[2])
            z_max = min(float(z_vals.max()), obj.bbox_max[2])
        else:
            z_min, z_max = obj.bbox_min[2], obj.bbox_max[2]
        if z_max <= z_min:
            z_min, z_max = obj.bbox_min[2], obj.bbox_max[2]
        centroid = (
            0.5 * (bbox_min_x + bbox_max_x),
            0.5 * (bbox_min_y + bbox_max_y),
            0.5 * (z_min + z_max),
        )
        boxes.append(
            SceneObject(
                id=f"{obj.id}#{n}",
                label=obj.label,
                bbox_min=(bbox_min_x, bbox_min_y, z_min),
                bbox_max=(bbox_max_x, bbox_max_y, z_max),
                centroid=centroid,
                category=obj.category,
                source=f"{obj.source}+splat_occupancy",
                confidence=obj.confidence,
                extra={**obj.extra, "refined_from": obj.id},
            )
        )
    return boxes


def refine_coarse_obstacles(
    obstacles: Sequence[SceneObject],
    splat: SplatData,
    *,
    floor_z: float,
    jumbo_area_m2: float = DEFAULT_JUMBO_AREA_M2,
    cell_m: float = DEFAULT_CELL_M,
    z_band: Tuple[float, float] = DEFAULT_Z_BAND,
    min_opacity: float = DEFAULT_MIN_OPACITY,
    min_cell_weight: float = DEFAULT_MIN_CELL_WEIGHT,
    exclude_labels: Sequence[str] = ("wall",),
) -> tuple[List[SceneObject], dict]:
    """Replace jumbo label boxes with splat-backed occupancy columns.

    Returns ``(refined_obstacles, report)``. Small boxes and structural walls
    pass through untouched. A jumbo box with NO splat mass inside it also passes
    through untouched (no evidence to carve with beats hallucinating free
    space). The report lists what was refined for the preflight manifest.
    """
    grid = build_floor_occupancy_grid(
        splat, floor_z=floor_z, cell_m=cell_m, z_band=z_band, min_opacity=min_opacity
    )
    refined: List[SceneObject] = []
    report: dict = {"refined": [], "jumbo_area_m2": jumbo_area_m2, "cell_m": cell_m}
    excluded = {label.lower() for label in exclude_labels}
    for obj in obstacles:
        dx = obj.bbox_max[0] - obj.bbox_min[0]
        dy = obj.bbox_max[1] - obj.bbox_min[1]
        if (obj.label or "").lower() in excluded or dx * dy < float(jumbo_area_m2):
            refined.append(obj)
            continue
        columns = _column_boxes_for_object(
            obj, grid, splat, min_cell_weight=min_cell_weight
        )
        if not columns:
            refined.append(obj)  # no splat evidence inside: keep the honest coarse box
            report["refined"].append(
                {"id": obj.id, "label": obj.label, "columns": 0, "kept_coarse_box": True}
            )
            continue
        refined.extend(columns)
        report["refined"].append(
            {
                "id": obj.id,
                "label": obj.label,
                "columns": len(columns),
                "coarse_area_m2": round(dx * dy, 3),
                "occupied_area_m2": round(len(columns) * cell_m * cell_m, 3),
                "kept_coarse_box": False,
            }
        )
    return refined, report


__all__ = [
    "DEFAULT_CELL_M",
    "DEFAULT_JUMBO_AREA_M2",
    "DEFAULT_MIN_CELL_WEIGHT",
    "DEFAULT_MIN_OPACITY",
    "DEFAULT_Z_BAND",
    "FloorOccupancyGrid",
    "build_floor_occupancy_grid",
    "refine_coarse_obstacles",
]
