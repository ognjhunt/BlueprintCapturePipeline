"""Where a website pick-and-place task sets its object down, and the area that counts as there.

Owner direction 2026-11-11 (capture-447ae4f4): the job said "the other end of the table", yet the
sealed destination lay 8 cm from the tray's own centre, inside its start footprint, and the scored
area was a 5 cm disc around a point Blueprint chose. The test measured Blueprint, not the robot.

Now, in the simulator frame (metres, Z up):

- a place the job's sentence names (the analysis's ``destination_region``, registered into the scene)
  is where the object goes, and no later step moves it away;
- the scored area is that place, kept on the support under it, shrunk so the object's footprint
  stays supported, and kept off the start footprint: a disc whose centre must hold the object's
  centre, never smaller than the 5 cm disc. The disc is also the green mark the robots see, so the
  mark and the score are one area. Without place words the destination is the harness's choice and
  the area stays the 5 cm disc;
- each choice is recorded with its basis, its distance and direction from the start, and flags that
  reach the site's result. A flag never stops the job.

Pure: no file, collider or provider access. The surface check is passed in (``holds``).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .website_surface_target_admission import WEBSITE_TARGET_TOLERANCE_M

NAMED_REGION, FARTHEST_SPOT, GROUNDED_CENTRE = "named_region", "start_surface_farthest_spot", "grounded_centre"
DESTINATION_GUESSED = "scene_estimate:destination_guessed"
DESTINATION_NEAR_START = "scene_estimate:destination_near_start"
#: The largest area the website's contracts accept (``radius_m`` at most 0.5 m).
MAXIMUM_AREA_RADIUS_M = 0.5
SCHEMA = "website_destination_choice.v1"


def _plan(values: Sequence[float]) -> tuple[float, float]:
    return float(values[0]), float(values[1])


def _distance_to_box(point: Sequence[float], low: Sequence[float], high: Sequence[float]) -> float:
    dx = max(low[0] - point[0], 0.0, point[0] - high[0])
    dy = max(low[1] - point[1], 0.0, point[1] - high[1])
    return math.hypot(dx, dy)


def _inside_box(point: Sequence[float], low: Sequence[float], high: Sequence[float]) -> bool:
    return low[0] <= point[0] <= high[0] and low[1] <= point[1] <= high[1]


def area_for_region(*, region_min: Sequence[float], region_max: Sequence[float], support_min: Sequence[float],
                    support_max: Sequence[float], subject_min: Sequence[float], subject_max: Sequence[float],
                    holds: Callable[[Sequence[float], float], bool] | None = None,
                    minimum_radius_m: float = WEBSITE_TARGET_TOLERANCE_M,
                    admits: Callable[[Sequence[float], float], bool] | None = None) -> dict[str, Any] | None:
    """The disc that stands for a named place: its centre and radius, or None when the place cannot hold one.

    The region is clipped to the support it lies on and shrunk by half the object's longer plan side,
    so the object rests wholly on the support with its centre anywhere in the area. The centre is the
    clipped region's centre, moved within it as far from the start as it can go when the start is near.
    The radius is the largest that stays in the shrunk region, off the start footprint and on the
    surface (``holds``), at most 0.5 m; a place too small for the 5 cm disc keeps the 5 cm disc.
    ``admits`` is the sealer's own admission (review 2026-11-12): a disc it would refuse is shrunk, never
    below the 5 cm disc, and a centre where even that is refused is not a candidate.
    """
    half = max(subject_max[0] - subject_min[0], subject_max[1] - subject_min[1]) / 2.0
    low = [max(region_min[i], support_min[i]) + half for i in (0, 1)]
    high = [min(region_max[i], support_max[i]) - half for i in (0, 1)]
    clipped = [max(region_min[i], support_min[i]) for i in (0, 1)], [min(region_max[i], support_max[i]) for i in (0, 1)]
    if any(clipped[0][i] > clipped[1][i] for i in (0, 1)):
        return None  # The place does not lie on the support at all.
    for i in (0, 1):
        if low[i] > high[i]:
            # Narrower than the object: its centre line is the only spot.
            low[i] = high[i] = (clipped[0][i] + clipped[1][i]) / 2.0
    start_centre = [(subject_min[i] + subject_max[i]) / 2.0 for i in (0, 1)]
    candidates = [[(low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0]]
    # The corners and edge midpoints of the shrunk region, farthest from the start first.
    for x in (low[0], (low[0] + high[0]) / 2.0, high[0]):
        for y in (low[1], (low[1] + high[1]) / 2.0, high[1]):
            candidates.append([x, y])
    best = None
    for centre in candidates:
        room = min(centre[0] - low[0], high[0] - centre[0], centre[1] - low[1], high[1] - centre[1]) + half
        clear = _distance_to_box(centre, subject_min, subject_max)
        if clear <= minimum_radius_m:
            # Never inside or touching the start footprint: the task would start (nearly) complete.
            continue
        radius = min(room, clear, MAXIMUM_AREA_RADIUS_M)
        limit = ("named_place" if radius == room else "start_footprint" if radius == clear
                 else "website_maximum")
        while holds is not None and radius > minimum_radius_m and not holds(centre, radius):
            radius *= 0.8
            limit = "surface_edge"
        if holds is not None and not holds(centre, max(radius, minimum_radius_m)):
            continue
        while admits is not None and radius > minimum_radius_m and not admits(centre, radius):
            radius = max(minimum_radius_m, radius * 0.9)
            limit = "sealer_admission"
        if admits is not None and not admits(centre, max(radius, minimum_radius_m)):
            continue
        key = (radius >= minimum_radius_m, min(radius, MAXIMUM_AREA_RADIUS_M),
               math.dist(centre, start_centre))
        if best is None or key > best[0]:
            best = (key, centre, radius, limit)
    if best is None:
        return None
    _, centre, radius, limit = best
    if radius < minimum_radius_m:
        limit = "smallest_area"
    size = [float(clipped[1][i] - clipped[0][i]) for i in (0, 1)]
    return {"centre_xy": [float(centre[0]), float(centre[1])],
            "radius_m": float(max(minimum_radius_m, min(radius, MAXIMUM_AREA_RADIUS_M))),
            "region_clipped_min_xy": clipped[0], "region_clipped_max_xy": clipped[1],
            "object_half_side_m": float(half),
            # How the radius was found, so a reader sees it is the named place's size and not a cap.
            "radius_derivation": {
                "named_place_size_m": size, "object_longest_side_m": float(2.0 * half),
                "limited_by": limit, "smallest_radius_m": float(minimum_radius_m),
                "largest_radius_m": MAXIMUM_AREA_RADIUS_M,
                "rule": ("The object's centre must end inside the named place, kept on its support and in "
                         "from the support's edges by half the object's longest side. The disc reaches as "
                         "far as that allows from its centre, stays off the object's start footprint, "
                         "is at least the smallest radius and at most the largest the website accepts.")}}


def choose_destination(*, destination: Mapping[str, Any], subject_min: Sequence[float], subject_max: Sequence[float],
                       support_min: Sequence[float] | None, support_max: Sequence[float] | None,
                       region: Mapping[str, Any] | None, relocation: Mapping[str, Any] | None,
                       holds: Callable[[Sequence[float], float], bool] | None = None,
                       admit: Callable[[Mapping[str, Any]], bool] | None = None
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The sealed destination (with its ``radius_m``) and the record of how it was chosen.

    ``destination`` is the compiled destination, ``region`` the preparation's ``destination_region``
    record (its ``estimated_bounds_world_m`` when registered), ``relocation`` the harness's move if any.
    All simulator frame. A destination on another surface, or inside a container, keeps its position
    and the 5 cm disc; it is still recorded. ``admit`` runs the sealer's own admission on a sealed
    destination: the named place's area is only ever one it admits, and when nowhere in the named place
    is admitted the destination stays the harness's choice with the 5 cm disc, flagged and recorded.
    """
    sealed = dict(destination)
    basis = FARTHEST_SPOT if relocation is not None else GROUNDED_CENTRE
    area = None
    same_surface_on = destination.get("relation") == "on" and destination.get("support") is None
    bounds = (region or {}).get("estimated_bounds_world_m") if isinstance(region, Mapping) else None
    named_place_refused = False
    if (same_surface_on or destination.get("support") is not None) and isinstance(bounds, Mapping) \
            and support_min is not None and support_max is not None and destination.get("relation") == "on":
        surface_min, surface_max = ((destination["support"]["aabb_min_xyz"], destination["support"]["aabb_max_xyz"])
                                    if destination.get("support") is not None else (support_min, support_max))
        height = float(destination["position_world_m"][2])
        admits = None if admit is None else (lambda centre, radius: admit(
            {**destination, "position_world_m": [float(centre[0]), float(centre[1]), height], "radius_m": float(radius)}))
        area = area_for_region(region_min=bounds["minimum"], region_max=bounds["maximum"],
                               support_min=surface_min, support_max=surface_max,
                               subject_min=subject_min, subject_max=subject_max, holds=holds, admits=admits)
        named_place_refused = area is None
    if area is not None:
        basis = NAMED_REGION
        sealed["position_world_m"] = [*area["centre_xy"], float(destination["position_world_m"][2])]
        sealed["radius_m"] = area["radius_m"]
    # Without a named place the destination keeps the 5 cm disc by default: no field changes.
    start = [(subject_min[i] + subject_max[i]) / 2.0 for i in (0, 1)]
    goal = _plan(sealed["position_world_m"])
    distance = math.dist(start, goal)
    longest = max(subject_max[0] - subject_min[0], subject_max[1] - subject_min[1])
    inside = _inside_box(goal, subject_min, subject_max)
    flags = []
    if basis != NAMED_REGION:
        flags.append(DESTINATION_GUESSED)
    if inside or distance < longest:
        flags.append(DESTINATION_NEAR_START)
    record = {"schema_version": SCHEMA, "basis": basis,
              "position_world_m": list(sealed["position_world_m"]),
              "area_radius_m": float(sealed.get("radius_m", WEBSITE_TARGET_TOLERANCE_M)),
              "start_centre_world_m": [float(v) for v in start],
              "distance_from_start_m": float(distance),
              "direction_from_start_xy": ([float((goal[i] - start[i]) / distance) for i in (0, 1)]
                                          if distance > 0 else [0.0, 0.0]),
              "object_longest_side_m": float(longest), "inside_start_footprint": bool(inside),
              "flags": sorted(flags), "physical_measurement": False,
              **({"area": area} if area is not None else {}),
              # The sentence named a place, but no disc in it is one the sealer admits.
              **({"named_place_not_admitted": True} if named_place_refused else {}),
              **({"replaced_harness_relocation": dict(relocation)} if area is not None and relocation is not None else {})}
    return sealed, record
