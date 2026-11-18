"""An object a little larger than the supported part of its destination surface, set down with its centre over it.

Beta journey step: compile's destination check (``website_task_preparation``), only where no surface carries the
object's whole footprint at the destination (``support_under`` found nothing). A real object stands on a surface a
little smaller than itself with its centre over it: test job 4's lamp, 0.18 m across its short side, on a glass
shelf estimated 0.14 to 0.19 m deep. Only privacy, rights and money stop a job; this is a fact of the job, flagged.

The object keeps the pose compile places it in (its box as it stands; the destination's orientation is the
identity), and is judged only in that pose. A spot holds it when:

- its centre and the scorer's disc about it (``website_target_radius``; the footprint where that is smaller) rest
  on one connected, near-level surface by ``support_under``'s own rules: compile's own lookup, window, level and
  connectivity, its nine probes the disc's centre, corners and edge midpoints;
- the disc lies wholly on that surface's outline (``support_outline``), as the sealer's admission reads it;
- at least ``RESTING_OUTLINE_MIN_SHARE`` of the footprint lies over that outline, read on the
  ``RESTING_OUTLINE_GRID_STEPS`` grid;
- no other collider geometry stands inside the object's box there, from the search window above the surface's top
  to the object's height: a wall a shelf is cut at, or clutter on a table.

The grounded centre is tried first; when its box meets other geometry the object slides along the surface to the
nearest spot that holds it, on the lattice of the footprint's own grid over the surface, nearest first (the
outline's cover is read once on that lattice and each spot's share summed from it). Wherever relocation, a named area or the site's choice puts it after,
the same test is read again there (``settled``): a spot that no longer holds it is the old stop, one that holds
all of it drops the record. The record keeps the share over the surface, the largest overhang in metres and how
far it slid; it is flagged only when something does hang over. A footprint with no spot that holds it returns
None, and compile stops exactly as before. Every value is an estimate, never a measurement.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .atlas_scene_estimates import DESTINATION_OVERHANG
from .website_support_geometry import (REGION_EDGE_ROUNDING_M, RESTING_OUTLINE_GRID_STEPS, RESTING_OUTLINE_MIN_SHARE,
                                       search_window_m, support_outline, support_under)

BASIS = "centre_and_disc_over_the_surface_box_clear"
#: The stop an object no spot holds keeps (``website_task_preparation``).
STOP = "task_destination_surface_contact_required"


def _plan(up: int) -> list[int]:
    return [axis for axis in range(3) if axis != up]


def _points(centre: np.ndarray, half: np.ndarray) -> np.ndarray:
    """The footprint's grid (its edges included) about ``centre``."""
    steps = np.linspace(-1.0, 1.0, RESTING_OUTLINE_GRID_STEPS)
    return centre + np.array([[x * half[0], y * half[1]] for x in steps for y in steps])


def _overhang(outline: Any, centre: np.ndarray, half: np.ndarray) -> float:
    """The footprint's largest distance past the outline (plan units): zero when all of it is over it."""
    points = _points(centre, half)
    reach = 2 * float(np.hypot(*half))
    return float(min(max((outline.distance(point, reach) for point in points[~outline.covers(points)]),
                         default=0.0), reach))


def _crossing(triangles: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Which triangles meet the box [low, high]: the separating axis test (box axes, each normal, edge crosses)."""
    centre, half = (low + high) / 2, (high - low) / 2
    points = triangles - centre
    meets = np.ones(len(points), dtype=bool)
    edges = [points[:, 1] - points[:, 0], points[:, 2] - points[:, 1], points[:, 0] - points[:, 2]]
    axes = [np.broadcast_to(np.eye(3)[axis], (len(points), 3)) for axis in range(3)]
    axes += [np.cross(edges[0], edges[1])]
    axes += [np.cross(np.eye(3)[axis], edge) for axis in range(3) for edge in edges]
    for axis in axes:
        projected = np.einsum("tij,tj->ti", points, axis)
        reach = np.abs(axis) @ half
        meets &= ~((projected.min(axis=1) > reach + 1e-12) | (projected.max(axis=1) < -reach - 1e-12))
    return meets


class _Obstacles:
    """The collider's triangles other than the surface's own, near the surface: read once, tested per spot."""

    def __init__(self, mesh: Any, surface: dict[str, Any], low: np.ndarray, high: np.ndarray):
        vertices, faces = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)
        own = np.zeros(len(faces), dtype=bool)
        own[np.asarray(surface.get("face_indices") or [], dtype=np.int64)] = True
        kept = []
        for start in range(0, len(faces), 1 << 18):
            block = vertices[faces[start:start + (1 << 18)]]
            near = ((block.max(axis=1) >= low).all(axis=1) & (block.min(axis=1) <= high).all(axis=1)
                    & ~own[start:start + len(block)])
            kept.append(block[near])
        self.triangles = np.concatenate(kept) if kept else np.zeros((0, 3, 3))

    def meet(self, low: np.ndarray, high: np.ndarray) -> bool:
        near = (self.triangles.max(axis=1) >= low).all(axis=1) & (self.triangles.min(axis=1) <= high).all(axis=1)
        return bool(_crossing(self.triangles[near], low, high).any()) if near.any() else False


class _Spot:
    """One surface, one object: whether a centre holds it, and its record there."""

    def __init__(self, mesh: Any, surface: dict[str, Any], lower: np.ndarray, upper: np.ndarray, *, up: int,
                 up_sign: int, meters_per_unit: float, placement_residual_m: float | None, radius: float):
        self.plan, self.up, self.up_sign, self.mpu, self.radius = _plan(up), up, up_sign, meters_per_unit, radius
        self.surface, self.half = surface, (upper - lower)[self.plan] / 2
        self.outline = support_outline(mesh, surface, up=up, meters_per_unit=meters_per_unit,
                                       placement_residual_m=placement_residual_m)
        top = float(surface["top_runtime_units"])
        clearance = search_window_m(placement_residual_m) / meters_per_unit
        ends = (top + up_sign * clearance, top + up_sign * float(upper[up] - lower[up]))
        self.layer = (min(ends), max(ends))
        self.obstacles = None
        if self.outline is not None:
            reach = 2 * float(np.hypot(*self.half))
            corners = self.outline.triangles.reshape(-1, 2)
            low, high = np.zeros(3), np.zeros(3)
            low[up], high[up] = self.layer
            low[self.plan], high[self.plan] = corners.min(axis=0) - reach, corners.max(axis=0) + reach
            self.obstacles = _Obstacles(mesh, surface, low, high)

    def box(self, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low, high = np.zeros(3), np.zeros(3)
        low[self.plan], high[self.plan] = centre - self.half, centre + self.half
        low[self.up], high[self.up] = self.layer
        return low, high

    def disc(self, centre: np.ndarray) -> bool:
        """Whether the scorer's disc about ``centre`` lies on the surface, as the sealer reads it."""
        return self.outline is not None and self.outline.holds(centre, self.radius,
                                                               tolerance=REGION_EDGE_ROUNDING_M / self.mpu)

    def share(self, centre: np.ndarray) -> float:
        return float(self.outline.covers(_points(centre, self.half)).mean())

    def holds(self, centre: np.ndarray, share: float) -> bool:
        """The disc on the surface, enough of the footprint over it and nothing else in its box there."""
        return (share >= RESTING_OUTLINE_MIN_SHARE and self.disc(centre)
                and not self.obstacles.meet(*self.box(centre)))

    def shares(self) -> tuple[np.ndarray, np.ndarray]:
        """Every centre on the footprint grid's lattice over the surface, and the share of the footprint over it there:
        the outline's cover read once on the lattice, each footprint's share a box sum of it."""
        step = 2 * self.half / (RESTING_OUTLINE_GRID_STEPS - 1)
        corners = self.outline.triangles.reshape(-1, 2)
        origin = corners.min(axis=0) - self.half
        counts = np.floor((corners.max(axis=0) + self.half - origin) / step).astype(int) + 1
        axes = [origin[index] + np.arange(counts[index]) * step[index] for index in range(2)]
        cover = self.outline.covers(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2))
        table = np.pad(cover.reshape(counts).astype(float).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        span, reach = RESTING_OUTLINE_GRID_STEPS, RESTING_OUTLINE_GRID_STEPS // 2
        if np.any(counts < span):
            return np.zeros((0, 2)), np.zeros(0)
        sums = table[span:, span:] - table[:-span, span:] - table[span:, :-span] + table[:-span, :-span]
        centres = np.stack(np.meshgrid(axes[0][reach:counts[0] - reach], axes[1][reach:counts[1] - reach],
                                       indexing="ij"), axis=-1).reshape(-1, 2)
        return centres, sums.reshape(-1) / span ** 2

    def record(self, centre: np.ndarray, share: float, start: np.ndarray, probes: int):
        moved = float(np.linalg.norm(centre - start)) * self.mpu
        overhang = _overhang(self.outline, centre, self.half)
        return {"basis": BASIS, "centre_runtime": centre.tolist(), "supported_share": share,
                "largest_overhang_m": overhang * self.mpu, "required_share": RESTING_OUTLINE_MIN_SHARE,
                "footprint_m": sorted(float(2 * value * self.mpu) for value in self.half), "orientation": "as_placed",
                "slid_m": moved, "geometry_probes": probes, "surface_top_runtime_units": self.surface["top_runtime_units"],
                "surface_face_indices": list(self.surface.get("face_indices") or []),
                **({"flag": DESTINATION_OVERHANG} if overhang > 0 else {}), "physical_measurement": False}


def centre_supported(mesh: Any, lower: Sequence[float], upper: Sequence[float], position: Sequence[float], *,
                     up: int, meters_per_unit: float, up_sign: int = 1, placement_residual_m: float | None = None,
                     radius: float, look: Callable[..., Any] = support_under
                     ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[float]]:
    """The surface, the record and the position to place on (``position`` with its plan centre slid where needed);
    (None, None, ``position``) when no spot holds the object. ``lower``/``upper`` are its footprint at ``position``,
    ``radius`` the scorer's disc, ``look`` compile's own ``support_under`` lookup."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    plan = _plan(up)
    start, half = (lower + upper)[plan] / 2, (upper - lower)[plan] / 2
    core_low, core_high = lower.copy(), upper.copy()
    core = np.minimum(half, radius)
    core_low[plan], core_high[plan] = start - core, start + core
    surface = look(mesh, core_low, core_high, up=up, meters_per_unit=meters_per_unit, up_sign=up_sign,
                   placement_residual_m=placement_residual_m)
    if surface is None:
        return None, None, list(position)
    spot = _Spot(mesh, surface, lower, upper, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit,
                 placement_residual_m=placement_residual_m, radius=radius)
    if spot.outline is None:
        return None, None, list(position)
    lattice, shares = spot.shares()
    held = shares >= RESTING_OUTLINE_MIN_SHARE
    candidates, shares = np.vstack([start, lattice[held]]), np.r_[spot.share(start), shares[held]]
    probes = 0
    for index in np.argsort(np.linalg.norm(candidates - start, axis=1), kind="stable"):
        centre, share = candidates[index], float(shares[index])
        probes += share >= RESTING_OUTLINE_MIN_SHARE
        if spot.holds(centre, share):
            placed = list(position)
            for offset, axis in enumerate(plan):
                placed[axis] = float(centre[offset])
            return surface, spot.record(centre, share, start, int(probes)), placed
    return None, None, list(position)


def settled(mesh: Any, record: dict[str, Any], lower: Sequence[float], upper: Sequence[float],
            final: Sequence[float], *, up: int, meters_per_unit: float, up_sign: int = 1,
            placement_residual_m: float | None = None, radius: float) -> tuple[dict[str, Any] | None, list[str]]:
    """The record read again where the object was finally placed (``final``, runtime units) after relocation, a
    named area or the site's choice: the record there; None with no code when that spot holds all of it; None and
    the old stop when it no longer holds it."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    surface = {"top_runtime_units": record["surface_top_runtime_units"], "face_indices": record["surface_face_indices"]}
    spot = _Spot(mesh, surface, lower, upper, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit,
                 placement_residual_m=placement_residual_m, radius=radius)
    centre = np.asarray(final, dtype=float)[_plan(up)]
    share = spot.share(centre) if spot.outline is not None else 0.0
    if spot.outline is None or not spot.holds(centre, share):
        return None, [STOP]
    placed = np.asarray(record["centre_runtime"], dtype=float)
    there = {**spot.record(centre, share, placed, record["geometry_probes"]), "slid_m": record["slid_m"],
             "moved_after_placement_m": float(np.linalg.norm(centre - placed)) * meters_per_unit}
    return (there if there["largest_overhang_m"] > 0 else None), []
