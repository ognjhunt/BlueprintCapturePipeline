"""A flat support under a rigid task, at the height its sealed success contract places the object (CPU).

Beta journey step: L5 team policy evaluation. The scorer reads an episode only
when its first sample lies within 2 mm and 0.01 rad of the task's
``start_pose_world``, and counts a placement only inside the success region's
height window. Both are flat-table numbers; the physics runs on the scan's
triangles. capture-447ae4f4's first GPU probe (2026-11-03 22:54Z) showed what
that costs on a real scan: its tray, seated on the highest collider point
under its footprint (``rigid_support_seating``), settled 2.8 mm lower and
tilted 0.044 rad before anything moved, so the episode was unscorable; and
released at the goal it settled 1.2 mm under the sealed window, tilted again.
The table is flat; its scan is bumpy by several millimetres.

The model, ``support_region_planarized_to_success_plane``, makes the team half's
physics support the support the success contract describes:

- the plane is the support height the sealed success document places the
  object on: its surface target's height (``surface_target.surface_position_world_m``),
  else the contract target's rest (the target less the object's qualified
  collision bottom). The window must hold the rest on that plane; if it does
  not, no support can make the start, the goal and the window agree without a
  new success document (and the owner's confirmation of it), and the half says so;
- the region is the convex hull of the start footprint (plus ``PAD_MARGIN_M``)
  and the success document's target disc (which holds every placement the
  scorer accepts, and lies on the support's outline), inside the registered
  support's plan bounds; it covers the carry or push between;
- inside it, every collider vertex within the support's own scan band moves to
  the plane: up to the registered support's highest face (the registration's
  own top; capped at ``BAND_BOUNDS_M``), and as far below. Higher geometry is
  something else on the support and stays; so does everything outside.

Only point heights change, so the collider keeps its prims, topology and
contact paths. The planarized collider is a derived copy staged in the team
half's packet (``write_planarized_collider``); the sealed revision, its
collider, its success document and the owner's confirmation of it are
unchanged. The object's start is then seated on the plane like any other
support (``rigid_support_seating``).

``support_rest_check`` predicts on the CPU where a box with the object's
qualified collision bounds rests, from two readings of the triangles under its
footprint: the highest contact (the seat) and the three-point support (the
plane over the footprint's clipped triangles that is lowest under the centre of
mass, so the facet it would settle on). The start must rest within the
scorer's reset tolerance of the start pose and the goal inside the sealed
window, or the half is held by name before any probe spends. It is a
quasi-static estimate (no sliding, no dynamics); physics decides.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .rigid_support_seating import SEARCH_BELOW_SUPPORT_TOP_M, SEATING_AGREEMENT_M, footprint_points

SCHEMA = "blueprint.rigid_task_support_pad.v1"
MODEL = "support_region_planarized_to_success_plane"
#: Beyond the start footprint (and a goal region without a target disc), so the object never rests on the
#: region's edge.
PAD_MARGIN_M = 0.02
#: Sides of the polygon that holds a target disc.
DISC_SIDES = 24
#: The scan band the website preparation treats as one fitted surface (``website_support_geometry``
#: ``FITTED_BAND_BOUNDS_M``): below the plane at least the first, never past the second either way.
BAND_BOUNDS_M = (0.01, 0.03)
#: A triangle that touches the region moves whole: vertices are selected this far beyond it at most
#: (the longest edge there), so the region itself is exactly flat.
MAXIMUM_EDGE_REACH_M = 0.1
#: A vertex this close to the plane is on it already: no derived collider is written for a flat support.
ON_PLANE_M = SEATING_AGREEMENT_M
#: The steepest support plane read as a rest; steeper, the object tips off the footprint's points.
MAXIMUM_REST_SLOPE = 1.0
#: The success document's surface target and its contract target agree to this (the bind derives one from
#: the other).
PLANE_AGREEMENT_M = 1e-6
NEW_REVISION_REQUIRED = "new_success_document_revision_and_owner_reconfirmation"
#: A rigid team half compiled before its support was modelled (capture-447ae4f4's, 2026-11-03).
UNMODELLED_HALF = "rigid_task_support_unchecked_compiled_before_support_pad"


class RigidSupportPadError(ValueError):
    """The support cannot be modelled from these inputs; the message names why."""


def _vector(value: Any, size: int) -> list[float] | None:
    try:
        items = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return items if len(items) == size and all(math.isfinite(item) for item in items) else None


def _bounds(value: Any) -> tuple[list[float], list[float]] | None:
    if not isinstance(value, Mapping):
        return None
    lower, upper = _vector(value.get("minimum"), 3), _vector(value.get("maximum"), 3)
    if lower is None or upper is None or any(a >= b for a, b in zip(lower, upper, strict=True)):
        return None
    return lower, upper


def _yaw(quaternion: Sequence[float]) -> float | None:
    """The heading of an upright orientation (xyzw), or None when it tilts."""

    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not norm or abs(x) / norm > 1e-6 or abs(y) / norm > 1e-6:
        return None
    return 2.0 * math.atan2(z / norm, w / norm)


def _hull(points: np.ndarray) -> np.ndarray:
    """The convex hull of plan points, counter-clockwise (monotone chain)."""

    unique = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=float).reshape(-1, 2)})
    if len(unique) < 3:
        return np.asarray(unique, dtype=float).reshape(-1, 2)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _dilated(polygon: np.ndarray, distance: float) -> np.ndarray:
    """A convex polygon grown by at least ``distance`` everywhere (the hull of its corners' squares)."""

    offsets = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]) * float(distance)
    return _hull((polygon[:, None, :] + offsets[None, :, :]).reshape(-1, 2))


def _clipped(polygon: np.ndarray, lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    """A convex polygon clipped to an axis-aligned rectangle (Sutherland-Hodgman)."""

    points = [tuple(row) for row in np.asarray(polygon, dtype=float)]
    for axis, value, keep_above in ((0, lower[0], True), (0, upper[0], False),
                                    (1, lower[1], True), (1, upper[1], False)):
        if not points:
            break

        def inside(point: tuple[float, ...]) -> bool:
            return point[axis] >= value if keep_above else point[axis] <= value

        kept: list[tuple[float, ...]] = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            if inside(current) != inside(previous):
                t = (value - previous[axis]) / (current[axis] - previous[axis])
                kept.append(tuple(previous[i] + t * (current[i] - previous[i]) for i in range(2)))
            if inside(current):
                kept.append(current)
        points = kept
    return _hull(np.asarray(points, dtype=float).reshape(-1, 2)) if len(points) >= 3 else np.zeros((0, 2))


def inside_polygon(polygon: Sequence[Sequence[float]], xy: Any) -> np.ndarray:
    """Which plan points lie in (or on) a counter-clockwise convex polygon."""

    hull = np.asarray(polygon, dtype=float).reshape(-1, 2)
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    inside = np.ones(len(points), dtype=bool)
    if len(hull) < 3:
        return np.zeros(len(points), dtype=bool)
    for index in range(len(hull)):
        a, b = hull[index], hull[(index + 1) % len(hull)]
        inside &= ((b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0])) >= -1e-12
    return inside


def _plan_corners(bounds: tuple[list[float], list[float]], quaternion: Sequence[float]) -> np.ndarray:
    """The plan corners of a scoring-frame box under ``quaternion`` (xyzw), about its own origin."""

    from .adp_rigid_task_scoring import _rotate_xyzw

    corners = [[x, y, z] for x in (bounds[0][0], bounds[1][0]) for y in (bounds[0][1], bounds[1][1])
               for z in (bounds[0][2], bounds[1][2])]
    return np.asarray([_rotate_xyzw(corner, list(quaternion))[:2] for corner in corners], dtype=float)


def success_plane(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    """The support height the sealed success contract places the object on, and whether its window agrees.

    ``consistent`` is False, with a ``reason``, when no flat support can give a
    rest inside the window: only a new success document can then change it.
    """

    bounds = _bounds(task_spec.get("subject_collision_bounds_scoring_frame_m"))
    target = _vector(task_spec.get("target_position_world_m"), 3)
    if bounds is None or target is None:
        raise RigidSupportPadError("rigid_support_pad_contract_geometry_missing")
    bottom = bounds[0][2]
    implied = target[2] + bottom
    surface = task_spec.get("surface_target")
    surface_z = None
    if isinstance(surface, Mapping):
        position = _vector(surface.get("surface_position_world_m"), 3)
        surface_z = position[2] if position is not None else None
    plane = surface_z if surface_z is not None else implied
    windows = [task_spec.get("support_height_interval_m"),
               [((task_spec.get("destination_position_bounds_world_m") or {}).get(key) or [None] * 3)[2]
                for key in ("minimum", "maximum")]]
    try:
        low = max(float(window[0]) for window in windows if window is not None and window[0] is not None)
        high = min(float(window[1]) for window in windows if window is not None and window[1] is not None)
    except (TypeError, ValueError, IndexError):
        low = high = None
    rest = plane - bottom
    value: dict[str, Any] = {
        "plane_z_m": plane,
        "basis": "success_document_surface_target" if surface_z is not None else "success_contract_target_rest",
        "contract_target_rest_plane_z_m": implied, "rest_scoring_z_m": rest,
        "success_window_scoring_z_m": [low, high], "collision_bottom_scoring_z_m": bottom}
    if surface_z is not None and abs(surface_z - implied) > PLANE_AGREEMENT_M:
        value.update(consistent=False, reason="rigid_support_success_target_off_its_surface",
                     requires=NEW_REVISION_REQUIRED)
    elif low is None or high is None or not low <= rest <= high:
        value.update(consistent=False, reason="rigid_support_success_window_off_success_plane",
                     requires=NEW_REVISION_REQUIRED)
    else:
        value["consistent"] = True
    return value


def pad_region(task_spec: Mapping[str, Any], *, support_bounds: Mapping[str, Any] | None = None,
               margin_m: float = PAD_MARGIN_M) -> np.ndarray:
    """The plan region the support is flat over: start footprint, goal footprint anywhere in the region, between.

    The goal area is the success document's own target disc when it has one: it
    holds every placement the scorer accepts (whole-subject containment), and the
    website preparation placed it on the support's outline, not merely in its
    bounding box (capture-447ae4f4's oval table). Without one, the success region
    grown by the footprint's half-diagonal (any yaw). A counter-clockwise convex
    polygon in world xy, inside the registered support's plan bounds when given
    (``bounds_min_xyz_m``/``bounds_max_xyz_m``).
    """

    bounds = _bounds(task_spec.get("subject_collision_bounds_scoring_frame_m"))
    start = _vector(task_spec.get("start_pose_world"), 7)
    target = _vector(task_spec.get("target_position_world_m"), 3)
    if bounds is None or start is None or target is None:
        raise RigidSupportPadError("rigid_support_pad_contract_geometry_missing")
    footprint = _dilated(_hull(_plan_corners(bounds, start[3:]) + np.asarray(start[:2])), margin_m)
    surface = task_spec.get("surface_target")
    disc = _vector((surface or {}).get("surface_position_world_m"), 3) if isinstance(surface, Mapping) else None
    try:
        radius = float(surface["radius_m"]) if disc is not None else None  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        radius = None
    if disc is not None and radius is not None and math.isfinite(radius) and radius > 0:
        # A polygon holding the disc (its sides touch it).
        sides = DISC_SIDES
        corner = radius / math.cos(math.pi / sides)
        goal = np.array([[disc[0] + corner * math.cos(2 * math.pi * (k + 0.5) / sides),
                          disc[1] + corner * math.sin(2 * math.pi * (k + 0.5) / sides)] for k in range(sides)])
    else:
        region = _bounds(task_spec.get("destination_position_bounds_world_m"))
        goal_low = np.asarray(region[0][:2] if region is not None else target[:2])
        goal_high = np.asarray(region[1][:2] if region is not None else target[:2])
        # Any yaw at the goal: a contract that ignores orientation lets a team place it turned.
        reach = float(np.max(np.linalg.norm(_plan_corners(bounds, [0.0, 0.0, 0.0, 1.0]), axis=1))) + margin_m
        goal = np.array([[x, y] for x in (goal_low[0] - reach, goal_high[0] + reach)
                         for y in (goal_low[1] - reach, goal_high[1] + reach)])
    polygon = _hull(np.concatenate([footprint, goal]))
    if support_bounds is not None:
        low, high = _vector(support_bounds.get("bounds_min_xyz_m"), 3), _vector(support_bounds.get("bounds_max_xyz_m"), 3)
        if low is not None and high is not None:
            polygon = _clipped(polygon, low[:2], high[:2])
    if len(polygon) < 3:
        raise RigidSupportPadError("rigid_support_pad_region_outside_support")
    return polygon


def scan_band(plane_z: float, registered_top_z: float | None) -> list[float]:
    """The heights a support's own scanned faces move from: as far above the plane as its highest face, and as far below."""

    above = 0.0 if registered_top_z is None else min(max(float(registered_top_z) - plane_z, 0.0), BAND_BOUNDS_M[1])
    below = min(max(above, BAND_BOUNDS_M[0]), BAND_BOUNDS_M[1])
    return [plane_z - below, plane_z + above]


def _selection(triangles: np.ndarray, polygon: np.ndarray, band: Sequence[float]) -> tuple[np.ndarray, float, bool]:
    """The region grown by the longest edge of any in-band triangle touching it, that reach, and whether it was capped."""

    tris = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    if not len(tris):
        return polygon, 0.0, False
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    box_low, box_high = tris[:, :, :2].min(axis=1), tris[:, :, :2].max(axis=1)
    near = (np.all(box_high >= low, axis=1) & np.all(box_low <= high, axis=1)
            & np.any((tris[:, :, 2] >= band[0]) & (tris[:, :, 2] <= band[1]), axis=1))
    tris = tris[near]
    if not len(tris):
        return polygon, 0.0, False
    edges = np.linalg.norm(tris[:, [1, 2, 0], :2] - tris[:, :, :2], axis=2).max(axis=1)
    reach = float(edges.max())
    capped = reach > MAXIMUM_EDGE_REACH_M
    reach = min(reach, MAXIMUM_EDGE_REACH_M)
    return _dilated(polygon, reach), reach, capped


def planarized_points(points: Any, *, selection: Sequence[Sequence[float]], band: Sequence[float],
                      plane_z: float) -> tuple[np.ndarray, np.ndarray]:
    """World points with each one in ``selection`` and ``band`` moved to the plane, and which moved."""

    world = np.array(points, dtype=float).reshape(-1, 3)
    chosen = inside_polygon(selection, world[:, :2]) & (world[:, 2] >= band[0]) & (world[:, 2] <= band[1])
    world[chosen, 2] = plane_z
    return world, chosen


def _deviation(triangles: np.ndarray, lower: Sequence[float], upper: Sequence[float], plane_z: float,
               window: Sequence[float], band: Sequence[float], polygon: np.ndarray | None = None) -> dict[str, Any]:
    """The collider's deviation from the plane over a footprint (or the region).

    ``highest_above_plane_m`` is its highest point there (what a seat stands on, whatever it is);
    ``lowest_below_plane_m`` the lowest point of the support's own scan (inside ``band``); the points
    above the band (something on the support) and below it (another surface, or a hole's floor) are counted.
    """

    points = footprint_points(triangles, lower, upper, floor_z=window[0], ceiling_z=window[1])
    if polygon is not None:
        points = points[inside_polygon(polygon, points[:, :2])]
    scan = points[(points[:, 2] >= band[0]) & (points[:, 2] <= band[1])]
    return {"points": int(len(points)),
            "highest_above_plane_m": float(points[:, 2].max() - plane_z) if len(points) else None,
            "lowest_below_plane_m": float(scan[:, 2].min() - plane_z) if len(scan) else None,
            "points_above_scan_band": int(np.count_nonzero(points[:, 2] > band[1])),
            "points_below_scan_band": int(np.count_nonzero(points[:, 2] < band[0]))}


def plan_support_pad(task_spec: Mapping[str, Any], *, support_plane: Mapping[str, Any],
                     triangles: Any) -> dict[str, Any]:
    """The flat support this task's success contract describes over ``triangles``, and what it changes.

    ``triangles`` is the scene collider's own soup around the region
    (``rigid_support_seating.collider_triangles``). The result is the modelling
    record (JSON) and, under ``planarized_triangles``, the soup as the derived
    collider will hold it.
    """

    plane = success_plane(task_spec)
    plane_z = float(plane["plane_z_m"])
    try:
        registered_top = float(support_plane["top_z_m"])
    except (KeyError, TypeError, ValueError):
        registered_top = None
    polygon = pad_region(task_spec, support_bounds=support_plane)
    band = scan_band(plane_z, registered_top)
    tris = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    selection, reach, capped = _selection(tris, polygon, band)
    low, high = _vector(support_plane.get("bounds_min_xyz_m"), 3), _vector(support_plane.get("bounds_max_xyz_m"), 3)
    if low is not None and high is not None:
        # The support's own faces lie inside its bounds: nothing outside them is its scan.
        selection = _clipped(selection, low[:2], high[:2])
    moved_points, chosen = planarized_points(tris.reshape(-1, 3), selection=selection, band=band, plane_z=plane_z)
    planarized = moved_points.reshape(-1, 3, 3)
    # Each scanned point once (a soup repeats a point in every triangle it is a corner of).
    moved = np.unique(tris.reshape(-1, 3)[chosen], axis=0)
    shift = np.abs(moved[:, 2] - plane_z)
    bounds = _bounds(task_spec.get("subject_collision_bounds_scoring_frame_m"))
    assert bounds is not None  # success_plane read them
    start = [float(value) for value in task_spec["start_pose_world"][:2]]
    target = [float(value) for value in task_spec["target_position_world_m"][:2]]
    window = [plane_z - SEARCH_BELOW_SUPPORT_TOP_M, plane_z + BAND_BOUNDS_M[1]]
    footprints = {"start": ([start[i] + bounds[0][i] for i in (0, 1)], [start[i] + bounds[1][i] for i in (0, 1)]),
                  "goal": ([target[i] + bounds[0][i] for i in (0, 1)], [target[i] + bounds[1][i] for i in (0, 1)])}
    region_low, region_high = polygon.min(axis=0).tolist(), polygon.max(axis=0).tolist()
    record: dict[str, Any] = {
        "schema_version": SCHEMA, "model": MODEL,
        "why": ("the scan's triangles under the task are bumpy, the support they stand for is flat; the scorer's "
                "reset readback and the sealed success window are flat-support heights"),
        "plane": plane, "plane_z_m": plane_z,
        "registered_support_top_z_m": registered_top,
        "region_world_xy_m": polygon.tolist(), "selection_world_xy_m": selection.tolist(),
        "region_bounds_world_xy_m": {"minimum": region_low, "maximum": region_high},
        "region_margin_m": PAD_MARGIN_M,
        "scan_band_z_m": band, "selection_edge_reach_m": reach, "selection_edge_reach_capped": capped,
        "measured_scan_deviation": {
            "window_z_m": window,
            **{name: _deviation(tris, low, high, plane_z, window, band) for name, (low, high) in footprints.items()},
            "region": _deviation(tris, region_low, region_high, plane_z, window, band, polygon)},
        "read_points_moved": int(np.count_nonzero(shift > ON_PLANE_M)),
        "read_points_maximum_shift_m": float(shift.max()) if len(shift) else 0.0,
        "sealed_revision_changed": False, "success_document_changed": False,
        "estimate_boundary": ("collision geometry only: the appearance is the scan's; vertices outside the region "
                              "or the band, the rest of the room, keep their scanned heights")}
    return {"record": record, "planarized_triangles": planarized, "selection": selection.tolist()}


def box_rest(triangles: Any, *, centre_xy: Sequence[float], bounds: Any, yaw_rad: float = 0.0,
             floor_z: float, ceiling_z: float) -> dict[str, Any]:
    """Where a box with scoring-frame ``bounds`` set down at ``centre_xy`` rests on ``triangles`` (quasi-static).

    The highest contact seats its bottom flat on the highest point under its
    footprint. The three-point support is the plane over the footprint's points
    that is lowest under the centre of mass (the scoring origin): the facet of
    their upper convex hull it settles on, its tilt, and the scoring height
    (``plane + h / cos(tilt)``, ``h`` the origin's height above the bottom). The
    footprint does not slide; a plane steeper than ``MAXIMUM_REST_SLOPE`` tips.
    """

    from scipy.optimize import linprog

    box = _bounds(bounds)
    if box is None:
        raise RigidSupportPadError("rigid_support_pad_contract_geometry_missing")
    tris = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    centre = np.asarray(centre_xy, dtype=float)[:2]
    if yaw_rad:
        c, s = math.cos(-yaw_rad), math.sin(-yaw_rad)
        rotation = np.array([[c, -s], [s, c]])
        tris = tris.copy()
        tris[:, :, :2] = (tris[:, :, :2] - centre) @ rotation.T + centre
    lower = [centre[i] + box[0][i] for i in (0, 1)]
    upper = [centre[i] + box[1][i] for i in (0, 1)]
    # Each point once: a scan repeats a vertex in every triangle it is a corner of.
    points = np.unique(footprint_points(tris, lower, upper, floor_z=floor_z, ceiling_z=ceiling_z), axis=0)
    height = -box[0][2]
    if not len(points):
        return {"status": "no_support_under_footprint", "points": 0}
    top = float(points[:, 2].max())
    offset = points[:, :2] - centre
    # Lowest plane z = c + a dx + b dy at the centre (c) that no point stands above.
    solved = linprog(c=[0.0, 0.0, 1.0], A_ub=-np.column_stack([offset, np.ones(len(points))]), b_ub=-points[:, 2],
                     bounds=[(-MAXIMUM_REST_SLOPE, MAXIMUM_REST_SLOPE)] * 2 + [(None, None)], method="highs")
    reading: dict[str, Any] = {"points": int(len(points)), "highest_contact_z_m": top,
                               "highest_contact_rest_scoring_z_m": top + height}
    if not solved.success:
        return {**reading, "status": "three_point_support_unsolved", "reason": str(solved.message)[:120]}
    a, b, intercept = (float(value) for value in solved.x)
    slope = math.hypot(a, b)
    if max(abs(a), abs(b)) >= MAXIMUM_REST_SLOPE - 1e-9:
        return {**reading, "status": "tips_off_footprint_points", "slope": slope}
    contacts = points[np.abs(offset @ np.array([a, b]) + intercept - points[:, 2]) <= 1e-9]
    return {**reading, "status": "rests",
            "three_point_plane_z_at_centre_m": intercept, "three_point_tilt_rad": math.atan(slope),
            "three_point_rest_scoring_z_m": intercept + height * math.sqrt(1.0 + slope * slope),
            "contact_points": len(contacts)}


def support_rest_check(triangles: Any, task_spec: Mapping[str, Any], *,
                       seated_on_highest_contact: bool = False) -> dict[str, Any]:
    """Whether the object rests at its start within the reset tolerance and at its goal inside the window.

    Read on ``triangles`` (the collider the physics runs on, near the task);
    ``status`` is ``consistent``, or ``held`` with typed ``reasons``. With
    ``seated_on_highest_contact`` the start is first seated on the highest
    point under its footprint, as ``rigid_support_seating`` seats it on a scan.
    """

    bounds = _bounds(task_spec.get("subject_collision_bounds_scoring_frame_m"))
    start = _vector(task_spec.get("start_pose_world"), 7)
    target = _vector(task_spec.get("target_position_world_m"), 3)
    try:
        translation = float(task_spec["reset_translation_tolerance_m"])
        rotation = float(task_spec["reset_orientation_tolerance_rad"])
    except (KeyError, TypeError, ValueError):
        translation = rotation = None
    if bounds is None or start is None or target is None or translation is None or rotation is None:
        return {"status": "unchecked", "reason": "rigid_support_rest_check_inputs_missing"}
    yaw = _yaw(start[3:])
    if yaw is None:
        return {"status": "unchecked", "reason": "rigid_support_start_orientation_not_upright"}
    height = bounds[1][2] - bounds[0][2]
    reasons: list[str] = []
    bottom = start[2] + bounds[0][2]
    rest = box_rest(triangles, centre_xy=start[:2], bounds={"minimum": bounds[0], "maximum": bounds[1]}, yaw_rad=yaw,
                    floor_z=bottom - SEARCH_BELOW_SUPPORT_TOP_M, ceiling_z=bottom + height)
    if seated_on_highest_contact and "highest_contact_rest_scoring_z_m" in rest:
        start = [*start[:2], float(rest["highest_contact_rest_scoring_z_m"]), *start[3:]]
    start_reading: dict[str, Any] = {"start_scoring_z_m": start[2], "rest": rest,
                                     "reset_tolerance": {"translation_m": translation, "orientation_rad": rotation}}
    if rest["status"] != "rests":
        reasons.append(f"rigid_support_start_{rest['status']}")
    else:
        seat = rest["highest_contact_rest_scoring_z_m"] - start[2]
        settle = rest["three_point_rest_scoring_z_m"] - start[2]
        tilt = rest["three_point_tilt_rad"]
        start_reading.update(highest_contact_offset_m=seat, three_point_offset_m=settle, three_point_tilt_rad=tilt)
        if abs(seat) > translation or abs(settle) > translation or tilt > rotation:
            reasons.append(f"rigid_support_start_rest_off_start:{max(abs(seat), abs(settle)) * 1000:.1f}mm"
                           f"/{tilt:.3f}rad")
    plane = success_plane(task_spec)
    low, high = plane["success_window_scoring_z_m"]
    destination = _bounds(task_spec.get("destination_position_bounds_world_m"))
    goals = [target[:2]]
    if destination is not None:
        goals += [[x, y] for x in (destination[0][0], destination[1][0]) for y in (destination[0][1], destination[1][1])]
    goal_bottom = target[2] + bounds[0][2]
    goal_rows = []
    for xy in goals:
        row = box_rest(triangles, centre_xy=xy, bounds={"minimum": bounds[0], "maximum": bounds[1]},
                       floor_z=goal_bottom - SEARCH_BELOW_SUPPORT_TOP_M, ceiling_z=goal_bottom + height)
        within = (row["status"] == "rests" and low is not None and high is not None
                  and low <= row["highest_contact_rest_scoring_z_m"] <= high
                  and low <= row["three_point_rest_scoring_z_m"] <= high)
        goal_rows.append({"centre_xy_m": [float(v) for v in xy], "rest": row, "within_success_window": within})
    if not goal_rows[0]["within_success_window"]:
        row = goal_rows[0]["rest"]
        reasons.append(f"rigid_support_goal_{row['status']}" if row["status"] != "rests" else
                       f"rigid_support_goal_rest_outside_success_window:{row['three_point_rest_scoring_z_m']:.5f}"
                       f"/[{low:.5f},{high:.5f}]")
    elif not all(row["within_success_window"] for row in goal_rows):
        reasons.append("rigid_support_goal_region_rest_outside_success_window")
    return {"status": "held" if reasons else "consistent", "reasons": reasons, "start": start_reading,
            "goal": {"success_window_scoring_z_m": [low, high], "placements": goal_rows},
            "basis": "cpu_highest_contact_and_three_point_support", "physics_decides": True}


def hold_reason(record: Mapping[str, Any] | None) -> str | None:
    """The typed reason a half's support record holds its task before any probe spends, or None."""

    if not isinstance(record, Mapping) or record.get("status") != "held":
        return None
    reasons = [str(reason) for reason in record.get("reasons") or [] if reason]
    return reasons[0] if reasons else "rigid_support_held"


def half_support_hold(team_scene_inputs_path: Any) -> str | None:
    """Why a compiled team half's rigid task is held by its support record, or None (also when unreadable).

    A rigid half compiled before its support was modelled has no record: it is held by name, so a release
    that models it compiles it again (``team_scene_packet_worker.unscorable_half``).
    """

    import json

    try:
        if not team_scene_inputs_path:
            return None
        record = json.loads(Path(str(team_scene_inputs_path)).read_text(encoding="utf-8"))
        plan = json.loads((Path(record["base_native_packet"]["path"]).parent
                           / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(record, Mapping) or (plan.get("task_spec") or {}).get("task_kind") != "rigid_pick_place":
        return None
    if "task_support_pad" not in record:
        return UNMODELLED_HALF
    return hold_reason(record.get("task_support_pad"))


def write_planarized_collider(source: str | Path, destination: str | Path, *, selection: Sequence[Sequence[float]],
                              band: Sequence[float], plane_z: float) -> dict[str, Any]:
    """A derived copy of the scene collider with every collision mesh point in ``selection`` and ``band`` on the plane.

    Read from the compile's binary copy (``usd_binary_read_cache.editable``),
    edited in memory only, exported to ``destination`` (``.usdc``). Only point
    heights change (in the world frame, through each mesh's own transform).
    """

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

    from .scene_physics_preflight import _usd_matrix
    from .usd_binary_read_cache import editable

    opened = editable(source)
    layer = Sdf.Layer.OpenAsAnonymous(opened) if isinstance(opened, str) else opened
    stage = Usd.Stage.Open(layer)
    if (not stage or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
            or UsdGeom.GetStageUpAxis(stage) != "Z"):
        raise RigidSupportPadError("rigid_support_pad_collider_frame_invalid")
    cache = UsdGeom.XformCache()
    meshes: list[dict[str, Any]] = []
    moved_total, largest = 0, 0.0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        mesh = UsdGeom.Mesh(prim)
        attribute = mesh.GetPointsAttr()
        if attribute.GetNumTimeSamples():
            raise RigidSupportPadError("rigid_support_pad_collider_points_time_sampled")
        local = np.asarray(attribute.Get() or [], dtype=float).reshape(-1, 3)
        if not len(local):
            continue
        matrix = _usd_matrix(cache, prim)
        rotation, translation = matrix[:3, :3], matrix[:3, 3]
        world = local @ rotation.T + translation
        planar, chosen = planarized_points(world, selection=selection, band=band, plane_z=plane_z)
        shift = np.abs(world[chosen, 2] - plane_z)
        changed = chosen.copy()
        changed[chosen] = shift > 0.0
        if not changed.any():
            continue
        local[changed] = (planar[changed] - translation) @ np.linalg.inv(rotation).T
        attribute.Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
        extent = mesh.GetExtentAttr()
        if extent.HasAuthoredValue():
            extent.Set(Vt.Vec3fArray.FromNumpy(np.stack([local.min(axis=0), local.max(axis=0)]).astype(np.float32)))
        count = int(np.count_nonzero(shift > ON_PLANE_M))
        moved_total += count
        largest = max(largest, float(shift.max()) if len(shift) else 0.0)
        meshes.append({"prim_path": prim.GetPath().pathString, "points": int(len(local)),
                       "points_moved": count, "maximum_shift_m": float(shift.max()) if len(shift) else 0.0})
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not layer.Export(str(output)):
        raise RigidSupportPadError("rigid_support_pad_collider_export_failed")
    del stage
    return {"meshes": meshes, "points_moved": moved_total, "maximum_shift_m": largest}


__all__ = [
    "BAND_BOUNDS_M",
    "MODEL",
    "NEW_REVISION_REQUIRED",
    "PAD_MARGIN_M",
    "RigidSupportPadError",
    "SCHEMA",
    "box_rest",
    "UNMODELLED_HALF",
    "half_support_hold",
    "hold_reason",
    "inside_polygon",
    "pad_region",
    "plan_support_pad",
    "planarized_points",
    "scan_band",
    "success_plane",
    "support_rest_check",
    "write_planarized_collider",
]
