"""An object a little larger than the supported part of its destination surface, set down with its centre over it.

Beta journey step: compile's destination check (``website_task_preparation``), only where no surface carries the
object's whole footprint at the destination (``support_under`` found nothing). A real object stands on a surface a
little smaller than itself with its centre over it: test job 4's lamp, 0.18 m across its short side, on a glass
shelf estimated 0.14 to 0.19 m deep. Only privacy, rights and money stop a job; this is a fact of the job, flagged.

The object keeps the pose compile places it in (its box as it stands; the destination's orientation is the
identity), and is judged only in that pose. A spot holds it when:

- its centre and the scorer's disc about it (``website_target_radius``; the footprint where that is smaller) rest
  on one connected, near-level surface by ``support_under``'s own rules: compile's own lookup, window, level and
  connectivity, its nine probes the centre, corners and edge midpoints of the square about the disc. That square's
  corners lie 1.41 radii out, past the disc: where one falls off a narrow surface turned in plan while the disc
  lies wholly on it (job move-the-lamp-2: a 16 cm glass shelf at 59 degrees, the disc 6.4 cm from its nearest
  edge), the surface is looked up again with the nine probes inside the disc (its inscribed square), so the
  search below still runs;
- the disc the scorer will use there lies wholly on that surface's outline (``support_outline``), as the sealer's
  admission reads it;
- at least ``RESTING_OUTLINE_MIN_SHARE`` of the footprint lies over that outline, read on the
  ``RESTING_OUTLINE_GRID_STEPS`` grid;
- no other collider geometry stands inside the object's box there, from just above the surface (the sealer's
  ``SUPPORT_HEIGHT_TOLERANCE_M``) to the object's own height: a wall a shelf is cut at, or clutter on a table (its
  sides, however low). A near-level face within ``support_under``'s search window of the surface's top is that
  surface at the search's own resolution (test job 4: pieces of the glass shelf the scan did catch, 1 to 4 cm over
  the estimate), never an obstacle; one farther up is. Nor does a solid enclose the box, meeting none of its faces:
  the box wholly inside clutter taller and wider than the object (``_Obstacles.enclosed``);
- on the surface the object starts on, it stays clear of its start (``destination_clear_of_start``, with compile's
  own clearance).

The grounded centre is tried first and kept when it holds. Otherwise the object slides to the nearest spot that
holds within one footprint of it, on the footprint grid's own lattice: the outline's cover read once there
(``3 * (RESTING_OUTLINE_GRID_STEPS - 1) + 1`` points a side) and each spot's share summed from it, so the samples
never grow with the surface. Where nothing within one footprint holds it (clutter or a gap wider than the object:
the synthetic set's seed 43, a 32 x 21 cm box where a 21 cm object was to go), the same search runs outward from
spots one footprint apart over the surface, nearest first, ``CLUTTER_SEEDS`` at most (``_Spot.across``), on a
surface carrying the whole footprint and on one carrying only its centre alike.

A named area and the places offered to the site are only ever ones that hold it (``Placement.admits`` wraps the
sealer's admission). Wherever relocation, the named area or a site's stored choice finally puts it, it is read
again there (``Placement.settle``): a spot that holds all of it drops the record; one that no longer holds it moves
to the nearest spot near it that does, or back to the spot first placed, recorded. A job that had a spot that
holds the object never stops for it.

Where every spot tested with the disc on the surface and half the footprint over it refused only for other geometry
in its box (``IN_THE_WAY``), the object still goes to the nearest of them (``_through``): only privacy, rights and
money stop a job, and the estimated box takes in the whole object, not only the base that rests (job move-the-lamp-2,
25 Nov: the lamp's shade against the wall behind the glass shelf, the shelf's own estimate 8 cm uncertain, so the
lamp and its rule were left out of the package). From there the spot judges it without that rule, so a named area,
a site's choice and ``settle`` read it alike, and the placement carries the search's record, flagged
``scene_estimate:destination_box_meets_room`` (``Placement.fields``); every reader words it
(``website_destination_unheld.BOX_MEETS_ROOM_NOTE``). Any other footprint with no such spot returns None, and compile
stops exactly as before; given ``why``, it is filled with what the search found (``unheld``), which compile keeps as
``destination_unheld`` and every reader words (``website_destination_unheld``): no surface under its centre, or no
spot with the disc on the surface and half the footprint over it. Every value is an estimate, never a measurement.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .atlas_scene_estimates import DESTINATION_BOX_MEETS_ROOM, DESTINATION_OVERHANG
from .task_evaluation_surface_target import SUPPORT_HEIGHT_TOLERANCE_M
from .website_support_geometry import (FITTED_MAX_FACE_TILT_DEGREES, REGION_EDGE_ROUNDING_M, RESTING_OUTLINE_GRID_STEPS,
                                       RESTING_OUTLINE_MIN_SHARE, destination_clear_of_start, same_support_surface,
                                       search_window_m, support_outline, support_under)
from .website_destination_unheld import IN_THE_WAY, NO_SURFACE, TOO_SMALL
from .website_surface_target_admission import WEBSITE_TARGET_TOLERANCE_M

BASIS = "centre_and_disc_over_the_surface_box_clear"
#: Spots one footprint apart tried outward from, nearest first, when nothing within one footprint holds the object.
CLUTTER_SEEDS = 24
#: Why a spot does not hold the object: the first rule it fails, in ``_Spot.holds``' order.
NEAR_START, TOO_LITTLE_OVER, NEAR_AN_EDGE, IN_ITS_BOX = (
    "near_its_start", "too_little_of_it_over_the_surface", "disc_past_an_edge", "other_geometry_in_its_box")
#: The placement's record where other geometry stood in its box at every spot that otherwise held it (``_through``).
PLACED_THROUGH = "placed_with_other_geometry_in_its_box"


def _plan(up: int) -> list[int]:
    return [axis for axis in range(3) if axis != up]


def _points(centre: np.ndarray, half: np.ndarray) -> np.ndarray:
    """The footprint's grid (its edges included) about ``centre``."""
    steps = np.linspace(-1.0, 1.0, RESTING_OUTLINE_GRID_STEPS)
    return centre + np.array([[x * half[0], y * half[1]] for x in steps for y in steps])


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
    """The collider's triangles that stand in an object's way near the surface: read once, tested per spot.

    Not the surface's own faces, and not a near-level face within ``window`` of its ``top`` (along up): that is the
    surface at the support search's resolution. Steep faces, and near-level ones farther up, are obstacles."""

    def __init__(self, mesh: Any, surface: Mapping[str, Any], low: np.ndarray, high: np.ndarray, *, up: int,
                 up_sign: int, top: float, window: float,
                 itself: tuple[Sequence[float], Sequence[float]] | None = None):
        vertices, faces = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)
        own = np.zeros(len(faces), dtype=bool)
        own[np.asarray(surface.get("face_indices") or [], dtype=np.int64)] = True
        level = np.cos(np.deg2rad(FITTED_MAX_FACE_TILT_DEGREES))
        kept = []
        for start in range(0, len(faces), 1 << 18):
            block = vertices[faces[start:start + (1 << 18)]]
            near = ((block.max(axis=1) >= low).all(axis=1) & (block.min(axis=1) <= high).all(axis=1)
                    & ~own[start:start + len(block)])
            block = block[near]
            normals = np.cross(block[:, 1] - block[:, 0], block[:, 2] - block[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            gentle = np.abs(normals[:, up]) >= level * np.maximum(lengths, 1e-300)
            above = (up_sign * block[:, :, up]).min(axis=1) - up_sign * top > window
            block = block[~gentle | above]
            if itself is not None:
                # The scan holds the object itself where it starts: its own faces are not in its way.
                inside = ((block.min(axis=1) >= np.asarray(itself[0]) - window).all(axis=1)
                          & (block.max(axis=1) <= np.asarray(itself[1]) + window).all(axis=1))
                block = block[~inside]
            kept.append(block)
        self.triangles = np.concatenate(kept) if kept else np.zeros((0, 3, 3))
        self.up, self.up_sign = up, up_sign

    def meet(self, low: np.ndarray, high: np.ndarray) -> bool:
        """Whether geometry stands in the box: a face crosses it, or a solid encloses it (``enclosed``)."""
        near = (self.triangles.max(axis=1) >= low).all(axis=1) & (self.triangles.min(axis=1) <= high).all(axis=1)
        if near.any() and _crossing(self.triangles[near], low, high).any():
            return True
        return self.enclosed(low, high)

    def enclosed(self, low: np.ndarray, high: np.ndarray) -> bool:
        """Whether the box lies inside a solid standing on the surface, touching none of its faces.

        A box wholly inside clutter meets no face (the synthetic set's seed 28: a 25 cm box on the table, the
        object's box within its sides and under its top). Seen from the box's centre at its base, every way
        along the surface the nearest face is a face's back: it leaves the solid (its outward normal, by its
        winding, points the way looked). A wall, a leg or the inside of a ring is met on its front and leaves
        the box free; so does an open or inconsistently wound scan, which never reads as enclosed.
        """
        if not len(self.triangles):
            return False
        plan = [axis for axis in range(3) if axis != self.up]
        origin = (low + high) / 2
        origin[self.up] = low[self.up] if self.up_sign == 1 else high[self.up]
        a, b, c = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        e1, e2 = b - a, c - a
        normals = np.cross(e1, e2)
        for axis in plan:
            for sign in (1.0, -1.0):
                direction = np.zeros(3)
                direction[axis] = sign
                # Moller-Trumbore, one ray against every triangle.
                p = np.cross(direction, e2)
                det = (e1 * p).sum(axis=1)
                safe = np.abs(det) > 1e-18
                inverse = np.divide(1.0, det, out=np.zeros_like(det), where=safe)
                s = origin - a
                u = (s * p).sum(axis=1) * inverse
                q = np.cross(s, e1)
                v = (direction * q).sum(axis=1) * inverse
                t = (e2 * q).sum(axis=1) * inverse
                hit = safe & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
                if not hit.any():
                    return False
                nearest = np.flatnonzero(hit)[np.argmin(t[hit])]
                if normals[nearest] @ direction <= 0:
                    return False
        return True


class _Spot:
    """One surface, one object: whether a centre holds it, the nearest that does, and its record there."""

    def __init__(self, mesh: Any, surface: Mapping[str, Any], lower: np.ndarray, upper: np.ndarray, *, up: int,
                 up_sign: int, meters_per_unit: float, placement_residual_m: float | None,
                 start: tuple[Sequence[float], Sequence[float]] | None, clearance: float | None,
                 itself: tuple[Sequence[float], Sequence[float]] | None = None):
        self.plan, self.up, self.mpu, self.start, self.clearance = _plan(up), up, meters_per_unit, start, clearance
        self.surface, self.half = surface, (upper - lower)[self.plan] / 2
        self.outline = support_outline(mesh, surface, up=up, meters_per_unit=meters_per_unit,
                                       placement_residual_m=placement_residual_m)
        top, height = float(surface["top_runtime_units"]), float(upper[up] - lower[up])
        # From just above the surface (a face within the sealer's support tolerance is the surface) to its height.
        ends = (top + up_sign * min(SUPPORT_HEIGHT_TOLERANCE_M / meters_per_unit, height), top + up_sign * height)
        self.layer = (min(ends), max(ends))
        self.obstacles = None
        if self.outline is not None:
            reach = 2 * float(np.hypot(*self.half))
            corners = self.outline.triangles.reshape(-1, 2)
            low, high = np.zeros(3), np.zeros(3)
            low[up], high[up] = self.layer
            low[self.plan], high[self.plan] = corners.min(axis=0) - reach, corners.max(axis=0) + reach
            self.obstacles = _Obstacles(mesh, surface, low, high, up=up, up_sign=up_sign, top=top,
                                        window=search_window_m(placement_residual_m) / meters_per_unit, itself=itself)
        self.lattice_points = 0
        self.refused: dict[str, int] = {}  # each spot a search tested and did not hold, by the rule it failed
        self.through = False  # other geometry in its box no longer refuses a spot (``_through``)

    def box(self, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low, high = np.zeros(3), np.zeros(3)
        low[self.plan], high[self.plan] = centre - self.half, centre + self.half
        low[self.up], high[self.up] = self.layer
        return low, high

    def share(self, centre: np.ndarray) -> float:
        return float(self.outline.covers(_points(centre, self.half)).mean())

    def refusal(self, centre: np.ndarray, share: float, radius: float) -> str | None:
        """The first rule ``centre`` fails, in ``holds``' order; None when it holds the object."""
        if self.start is not None:
            point = np.zeros(3)
            point[self.plan] = centre
            if not destination_clear_of_start(*self.start, point, up=self.up, clearance=self.clearance):
                return NEAR_START
        if share < RESTING_OUTLINE_MIN_SHARE or self.outline is None:
            return TOO_LITTLE_OVER
        if not self.outline.holds(centre, radius, tolerance=REGION_EDGE_ROUNDING_M / self.mpu):
            return NEAR_AN_EDGE
        return IN_ITS_BOX if not self.through and self.obstacles.meet(*self.box(centre)) else None

    def holds(self, centre: np.ndarray, share: float, radius: float) -> bool:
        """The disc of ``radius`` on the surface, enough of the footprint over it, nothing else in its box there,
        and clear of its start on the surface it starts on."""
        return self.refusal(centre, share, radius) is None

    def nearest(self, around: np.ndarray, radius: float) -> tuple[np.ndarray, float, int] | None:
        """The nearest spot within one footprint of ``around`` that holds the object: its centre, share and the
        spots tested. On the footprint grid's lattice: the outline's cover read once there and each spot's share a
        box sum of it, so the samples are bounded whatever the surface's size."""
        intervals = RESTING_OUTLINE_GRID_STEPS - 1
        step = 2 * self.half / intervals  # the footprint grid's own spacing
        side = 3 * intervals + 1  # spots one footprint each way, and the footprint's own samples about each
        offsets = (np.arange(side) - side // 2)[:, None] * step[None, :]
        axes = [around[index] + offsets[:, index] for index in range(2)]
        cover = self.outline.covers(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2))
        self.lattice_points = side * side
        table = np.pad(cover.reshape(side, side).astype(float).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        span = RESTING_OUTLINE_GRID_STEPS  # the footprint's own samples
        sums = table[span:, span:] - table[:-span, span:] - table[span:, :-span] + table[:-span, :-span]
        reach = span // 2
        centres = np.stack(np.meshgrid(axes[0][reach:side - reach], axes[1][reach:side - reach], indexing="ij"),
                           axis=-1).reshape(-1, 2)
        shares = sums.reshape(-1) / span ** 2
        held = shares >= RESTING_OUTLINE_MIN_SHARE
        centres, shares = centres[held], shares[held]
        order = np.argsort(np.linalg.norm(centres - around, axis=1), kind="stable")
        for tested, index in enumerate(order, start=1):
            why = self.refusal(centres[index], float(shares[index]), radius)
            if why is None:
                return centres[index], float(shares[index]), tested
            self.refused[why] = self.refused.get(why, 0) + 1
        return None

    def across(self, around: np.ndarray, radius: float) -> tuple[np.ndarray, float, int] | None:
        """``nearest``, and where nothing within one footprint of ``around`` holds the object (clutter or a gap wider
        than it), the nearest from spots one footprint apart over the surface, nearest ``around`` first, each search
        bounded as ever, so the samples never grow with the surface (``CLUTTER_SEEDS`` at most)."""
        found = self.nearest(around, radius)
        if found is not None or self.outline is None:
            return found
        corners = self.outline.triangles.reshape(-1, 2)
        axes = [np.arange(corners[:, i].min(), corners[:, i].max() + 1e-12, 2 * self.half[i]) for i in range(2)]
        seeds = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
        seeds = seeds[self.outline.covers(seeds)] if len(seeds) else seeds
        for seed in seeds[np.argsort(np.linalg.norm(seeds - around, axis=1), kind="stable")][:CLUTTER_SEEDS]:
            found = self.nearest(seed, radius)
            if found is not None:
                return found
        return None

    def record(self, centre: np.ndarray, share: float, start: np.ndarray, tested: int) -> dict[str, Any] | None:
        """The record at ``centre``; None when nothing hangs over there."""
        points = _points(centre, self.half)
        reach = 2 * float(np.hypot(*self.half))
        overhang = float(min(max((self.outline.distance(point, reach) for point in points[~self.outline.covers(points)]),
                                 default=0.0), reach)) * self.mpu
        if overhang <= 0:
            return None
        return {"basis": BASIS, "centre_runtime": centre.tolist(), "supported_share": share,
                "largest_overhang_m": overhang, "required_share": RESTING_OUTLINE_MIN_SHARE,
                "footprint_m": sorted(float(2 * value * self.mpu) for value in self.half), "orientation": "as_placed",
                "slid_m": float(np.linalg.norm(centre - start)) * self.mpu, "spots_tested": tested,
                "lattice_points": self.lattice_points, "flag": DESTINATION_OVERHANG, "physical_measurement": False}


def unheld(spot: _Spot | None, half: np.ndarray, meters_per_unit: float) -> dict[str, Any]:
    """What a search that found no spot holding the object found (``spot`` None: no surface under its centre).

    ``cause`` only where the counts settle it: no surface; no spot tested with the disc on the surface and half the
    footprint over it (``TOO_SMALL``); or every spot tested that had both refused only for geometry in its box
    (``IN_THE_WAY``). Spots are counted each time a search tested one; a spot searched from two seeds counts twice."""
    refused = dict(sorted((spot.refused if spot is not None else {}).items()))
    cause = (NO_SURFACE if spot is None else None if refused.get(NEAR_START) else
             IN_THE_WAY if refused.get(IN_ITS_BOX) else TOO_SMALL)
    return {"status": "no_spot_holds_it", "basis": BASIS, **({"cause": cause} if cause else {}),
            "spots_tested": sum(refused.values()), "refused": refused,
            "footprint_m": sorted(float(2 * value * meters_per_unit) for value in half), "orientation": "as_placed",
            "physical_measurement": False}


def _through(spot: _Spot, around: np.ndarray, radius: float, half: np.ndarray, meters_per_unit: float
             ) -> tuple[tuple[np.ndarray, float, int] | None, dict[str, Any]]:
    """The nearest spot to ``around`` that holds the object but for other geometry in its box, and the record of the
    search that found none clear (``unheld``, flagged); (None, that record) unless other geometry was its only cause
    (``IN_THE_WAY``). From then on ``spot`` judges every spot without that rule."""
    found = unheld(spot, half, meters_per_unit)
    if found.get("cause") != IN_THE_WAY:
        return None, found
    spot.through = True
    near = spot.across(around, radius)
    if near is None:  # a spot failed only that rule, so one holds now; were none found, it stops as before
        spot.through = False
        return None, found
    return near, {**found, "status": PLACED_THROUGH, "flag": DESTINATION_BOX_MEETS_ROOM}


class Placement:
    """The overhang placement compile carries from its destination check to where the object finally lands; ``met``,
    the search's record where other geometry stood in its box at every spot that otherwise held it (``_through``)."""

    def __init__(self, spot: _Spot, centre: np.ndarray, grounded: np.ndarray, record: dict[str, Any] | None,
                 runtime_to_sim: Any, met: dict[str, Any] | None = None):
        self.spot, self.centre, self.grounded, self.record, self.met = spot, centre, grounded, record, met
        self.runtime_to_sim = np.asarray(runtime_to_sim, dtype=float)

    def fields(self) -> dict[str, Any]:
        """The preparation's records of where the object lands: what hangs over (``destination_overhang``) and the
        other geometry in its box there (``destination_box_meets_room``), each only when there is one."""
        return {**({"destination_overhang": self.record} if self.record else {}),
                **({"destination_box_meets_room": self.met} if self.met else {})}

    def _at(self, sealed: Mapping[str, Any]) -> tuple[np.ndarray, float]:
        """A sealed destination's centre (runtime plan) and the radius the scorer uses there (runtime units)."""
        runtime = np.linalg.solve(self.runtime_to_sim, [*sealed["position_world_m"], 1.0])[:3]
        return runtime[self.spot.plan], float(sealed.get("radius_m") or WEBSITE_TARGET_TOLERANCE_M) / self.spot.mpu

    def admits(self, admit: Callable[[Mapping[str, Any]], bool]) -> Callable[[Mapping[str, Any]], bool]:
        """The sealer's admission, and this object held at that place with that place's disc."""
        def admitted(sealed: Mapping[str, Any]) -> bool:
            centre, radius = self._at(sealed)
            return admit(sealed) and self.spot.holds(centre, self.spot.share(centre), radius)
        return admitted

    def settle(self, destination: dict[str, Any], choice: dict[str, Any] | None
               ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """The destination and its choice where the object finally holds, and the record there (``self.record``)."""
        centre, radius = self._at(destination)
        share = self.spot.share(centre)
        if self.spot.holds(centre, share, radius):
            self.record = self.spot.record(centre, share, self.grounded, 0)
            if self.record is not None:
                self.record["moved_after_placement_m"] = float(np.linalg.norm(centre - self.centre)) * self.spot.mpu
            return destination, choice
        # Relocation, the named area or the site's stored choice put it where it no longer holds: the nearest spot
        # near there that does, with the 5 cm disc, else the spot it was first placed on, which held.
        default = WEBSITE_TARGET_TOLERANCE_M / self.spot.mpu
        found = self.spot.nearest(centre, default)
        target, share = (found[0], found[1]) if found is not None else (self.centre, self.spot.share(self.centre))
        runtime = np.linalg.solve(self.runtime_to_sim, [*destination["position_world_m"], 1.0])[:3]
        runtime[self.spot.plan] = target
        position = (self.runtime_to_sim @ [*runtime, 1.0])[:3].tolist()
        moved = float(np.linalg.norm(target - centre)) * self.spot.mpu
        destination = {**{key: value for key, value in destination.items() if key != "radius_m"},
                       "position_world_m": position}
        if choice is not None:
            choice = {**choice, "position_world_m": position, "area_radius_m": WEBSITE_TARGET_TOLERANCE_M,
                      "moved_to_hold_the_object_m": moved}
        self.record = self.spot.record(target, share, self.grounded, found[2] if found is not None else 0)
        if self.record is not None:
            self.record.update(chosen_place_held=False, moved_to_hold_the_object_m=moved,
                               moved_to="nearest_spot_holding_it" if found is not None else "spot_first_placed")
        return destination, choice


def centre_supported(mesh: Any, lower: Sequence[float], upper: Sequence[float], position: Sequence[float], *,
                     up: int, meters_per_unit: float, up_sign: int = 1, placement_residual_m: float | None = None,
                     radius: float, runtime_to_sim: Any, start: tuple[Sequence[float], Sequence[float]],
                     start_support: Mapping[str, Any] | None, clearance: float,
                     look: Callable[..., Any] = support_under,
                     itself: tuple[Sequence[float], Sequence[float]] | None = None,
                     why: dict[str, Any] | None = None
                     ) -> tuple[dict[str, Any] | None, Placement | None, list]:
    """The surface, the placement and the position to place on (``position`` with its plan centre slid where
    needed), the spot nearest it that holds the object but for other geometry in its box where only that refused
    every spot (``_through``); else (None, None, ``position``), and then ``why``, when given, filled with ``unheld``'s
    record. ``lower``/``upper`` are its footprint at ``position``, ``start`` its start box on
    ``start_support`` with compile's ``clearance``, ``radius`` the scorer's disc, ``look`` compile's own
    ``support_under`` lookup."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    plan = _plan(up)
    start_centre, half = (lower + upper)[plan] / 2, (upper - lower)[plan] / 2
    core_low, core_high = lower.copy(), upper.copy()
    core, inside = np.minimum(half, radius), np.minimum(half, radius / np.sqrt(2))
    surface = None
    # The square about the disc first; where a corner of it is off the surface, the square inside the disc.
    for reach in (core, inside) if np.any(inside < core) else (core,):
        core_low[plan], core_high[plan] = start_centre - reach, start_centre + reach
        surface = look(mesh, core_low, core_high, up=up, meters_per_unit=meters_per_unit, up_sign=up_sign,
                       placement_residual_m=placement_residual_m)
        if surface is not None:
            break
    if surface is None:
        if why is not None:
            why.update(unheld(None, half, meters_per_unit))
        return None, None, list(position)
    same = start_support is not None and same_support_surface(surface, dict(start_support))
    spot = _Spot(mesh, surface, lower, upper, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit,
                 placement_residual_m=placement_residual_m, start=start if same else None, clearance=clearance,
                 itself=itself)
    if spot.outline is None:
        # No outline to test a spot on: no spot is proven, but the surface is reported (compile keeps a whole
        # footprint's own support then; an overhang has nothing to stand on).
        return surface, None, list(position)
    share, met = spot.share(start_centre), None
    if spot.holds(start_centre, share, radius):
        found = (start_centre, share, 1)
    else:
        found = spot.across(start_centre, radius)
        if found is None:
            found, met = _through(spot, start_centre, radius, half, meters_per_unit)
            if found is None:
                if why is not None:
                    why.update(met)
                return None, None, list(position)
    centre = found[0]
    placed = list(position)
    for offset, axis in enumerate(plan):
        placed[axis] = float(centre[offset])
    record = spot.record(centre, found[1], start_centre, found[2])
    return surface, Placement(spot, centre, start_centre, record, runtime_to_sim, met), placed


def destination_points(row: Mapping[str, Any], frames: Sequence[Mapping[str, Any]], matrix: Any) -> np.ndarray | None:
    """The destination's own observed points (its masks lifted through the frames), in the runtime frame.

    Compile's destination lookups take them as ``support_under``'s ``shown_by``: of the surfaces the search
    window reaches under the footprint, the destination is the one its own points lie on, not a box standing
    on it. None without a tracked mask, without geometry, or for a surface estimated from its visible edges
    (``website_unscanned_surface``): that surface is the destination's by construction, and the points of a
    see-through shelf lie on what is behind it.
    """
    from .website_task_masks import masked_view_points

    track = row.get("track")
    if row.get("estimated_surface_runtime") or not isinstance(track, Mapping) or not track.get("observations"):
        return None
    try:
        points = np.concatenate(masked_view_points(track, list(frames)))
    except (KeyError, ValueError, OSError):
        return None
    matrix = np.asarray(matrix, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def placed_clear(mesh: Any, lower: Sequence[float], upper: Sequence[float], position: Sequence[float],
                 whole: Mapping[str, Any] | None, *, start: tuple[Sequence[float], Sequence[float]],
                 start_support: Mapping[str, Any] | None, **options: Any) -> tuple[Any, Placement | None, list]:
    """Compile's destination on an "on" surface, given ``whole``: the support under the whole footprint, or None.

    With no support under the whole footprint, the object may still hold with its centre over one, an overhang
    flagged (``centre_supported``). With one, it can still hold something else there (clutter on a table): only
    when other geometry (never the object's own scan where it starts, ``start``) stands in the object's box at
    the grounded centre does it slide to the nearest spot that holds it; with none, the nearest that holds it but
    for that geometry (``_through``). Otherwise,
    and wherever no outline can be read to test it, it takes exactly the path it always took, and compile's own
    relocation keeps it off its start. ``why``, when given, is filled as ``centre_supported`` fills it wherever no
    spot holds the object."""
    if whole is None:
        return centre_supported(mesh, lower, upper, position, start=start, start_support=start_support, itself=start,
                                **options)
    why = options.pop("why", None)
    lower, upper, grounded = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), list(position)
    plan = _plan(options["up"])
    spot = _Spot(mesh, whole, lower, upper, up=options["up"], up_sign=options.get("up_sign", 1),
                 meters_per_unit=options["meters_per_unit"], placement_residual_m=options.get("placement_residual_m"),
                 start=None, clearance=None, itself=start)
    centre = (lower + upper)[plan] / 2
    if spot.outline is None or spot.obstacles is None or not spot.obstacles.meet(*spot.box(centre)):
        return whole, None, grounded
    found, met = spot.across(centre, options["radius"]), None
    if found is None:
        found, met = _through(spot, centre, options["radius"], (upper - lower)[plan] / 2, options["meters_per_unit"])
        if found is None:
            if why is not None:
                why.update(met)
            return None, None, grounded
    placed = list(grounded)
    for offset, axis in enumerate(plan):
        placed[axis] = float(found[0][offset])
    record = spot.record(found[0], found[1], centre, found[2])
    return whole, Placement(spot, found[0], centre, record, options["runtime_to_sim"], met), placed
