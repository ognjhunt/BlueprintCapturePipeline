"""An object a little larger than its destination surface, set down with its centre over it, and flagged.

Beta journey step: compile's destination check (``website_task_preparation``), only where no surface carries the
object's whole footprint at the destination (``support_under`` found nothing). A real object stands on a surface
a little smaller than itself with its centre over it: test job 4's lamp, 0.18 m across its short side, on a glass
shelf estimated 0.14 to 0.19 m deep. Only privacy, rights and money stop a job; this is a fact of the job, flagged.

The placement holds when, at the same destination centre:

- the footprint's centre and the scorer's disc about it (``website_target_radius``; the footprint itself where
  that is smaller) rest on one connected, near-level surface by ``support_under``'s own rules: the same lookup,
  window, level and connectivity, its nine probes the disc's centre, corners and edge midpoints;
- at least ``RESTING_OUTLINE_MIN_SHARE`` of the whole footprint lies over that surface's outline
  (``support_outline``), read on the ``RESTING_OUTLINE_GRID_STEPS`` grid, with the footprint as it is or turned to
  the surface's own long axis, whichever has more over the surface;
- the disc itself lies wholly on that surface's outline, as the sealer's admission reads it.

The record keeps the supported share, the largest overhang in metres and the turn. A footprint that does not
qualify returns None, and compile stops exactly as before. Every value is an estimate, never a measurement.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from .atlas_scene_estimates import DESTINATION_OVERHANG
from .website_support_geometry import (REGION_EDGE_ROUNDING_M, RESTING_OUTLINE_GRID_STEPS, RESTING_OUTLINE_MIN_SHARE,
                                       support_outline, support_under)

BASIS = "centre_and_disc_over_the_surface"


def _footprint(centre: np.ndarray, half: np.ndarray, angle: float) -> np.ndarray:
    """Plan points over the footprint (its edges included), turned by ``angle`` about its centre."""
    steps = np.linspace(-1.0, 1.0, RESTING_OUTLINE_GRID_STEPS)
    local = np.array([[x * half[0], y * half[1]] for x in steps for y in steps])
    turn = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return centre + local @ turn.T


def _long_axis(outline: Any) -> float:
    """The surface's long axis in plan, as an angle from the first plan axis."""
    points = outline.triangles.reshape(-1, 2)
    direction = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2][0]
    return float(math.atan2(direction[1], direction[0]))


def centre_supported(mesh: Any, lower: Sequence[float], upper: Sequence[float], *, up: int, meters_per_unit: float,
                     up_sign: int = 1, placement_residual_m: float | None = None, radius: float,
                     look: Callable[..., Any] = support_under) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The surface under a destination footprint's centre and the scorer's disc, and its overhang record; (None, None)
    when the footprint does not qualify. ``radius`` is the disc's, ``look`` compile's own ``support_under`` lookup."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    plan = [axis for axis in range(3) if axis != up]
    centre, half = (lower + upper)[plan] / 2, (upper - lower)[plan] / 2
    core_low, core_high = lower.copy(), upper.copy()
    core = np.minimum(half, radius)
    core_low[plan], core_high[plan] = centre - core, centre + core
    options = {"up": up, "meters_per_unit": meters_per_unit, "up_sign": up_sign,
               "placement_residual_m": placement_residual_m}
    surface = look(mesh, core_low, core_high, **options)
    if surface is None:
        return None, None
    outline = support_outline(mesh, surface, up=up, meters_per_unit=meters_per_unit,
                              placement_residual_m=placement_residual_m)
    if outline is None or not outline.holds(centre, radius, tolerance=REGION_EDGE_ROUNDING_M):
        return None, None
    rows, axis = [], _long_axis(outline)
    # Turned to the surface: the footprint's long side along the surface's long axis.
    own = 0.0 if half[0] >= half[1] else math.pi / 2
    turn = (axis - own + math.pi / 2) % math.pi - math.pi / 2
    reach = 2 * float(np.hypot(*half))
    for orientation, angle in (("as_is", 0.0), ("turned_to_the_surface", turn)):
        points = _footprint(centre, half, angle)
        covered = outline.covers(points)
        overhang = max((outline.distance(point, reach) for point in points[~covered]), default=0.0)
        rows.append({"orientation": orientation, "turn_degrees": math.degrees(angle),
                     "supported_share": float(covered.mean()),
                     "largest_overhang_m": float(min(overhang, reach) * meters_per_unit)})
    best = max(rows, key=lambda row: row["supported_share"])
    if best["supported_share"] < RESTING_OUTLINE_MIN_SHARE:
        return None, None
    record = {"basis": BASIS, **best, "required_share": RESTING_OUTLINE_MIN_SHARE,
              "footprint_m": sorted(float(2 * value * meters_per_unit) for value in half),
              "orientations": rows, "flag": DESTINATION_OVERHANG, "physical_measurement": False}
    return surface, record
