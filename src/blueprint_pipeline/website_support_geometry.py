"""Task-footprint contact against reconstructed triangles (ADP-030, day 28).

These are estimated collider contacts, never measurements. A surface's AABB
alone cannot show that it supports the task object: gaps and table edges matter.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def support_under(mesh: Any, lower: Sequence[float], upper: Sequence[float], *,
                  up: int, meters_per_unit: float, up_sign: int = 1,
                  placement_residual_m: float | None = None) -> dict[str, Any] | None:
    """Find a nearby, connected, nearly horizontal surface under the footprint.

    Probe the center, corners and edge midpoints; retain the actual face set for
    downstream contact checks. Refuse a remote floor or disconnected islands.
    A bumpy reconstructed surface with no exact level face under the footprint
    is fitted instead (``fitted_surface_under_footprint``), under the same refusals.

    The surface is searched within 3 cm of the footprint's estimated bottom, or
    within ``placement_residual_m`` (the registration's measured residual near
    the task) when that is larger: an estimate cannot place the bottom closer
    than its own fit (capture-447ae4f4: 7.2 cm, bottom 4 cm into the counter).
    """
    window_m = 0.03
    if (not isinstance(placement_residual_m, bool) and isinstance(placement_residual_m, (int, float))
            and math.isfinite(placement_residual_m)):
        window_m = max(window_m, float(placement_residual_m))
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(upper <= lower) or up not in {1, 2} or up_sign not in {-1, 1}
            or not np.isfinite(meters_per_unit) or meters_per_unit <= 0):
        raise ValueError("website_support_query_invalid")
    mesh = mesh.copy()
    if up_sign == -1:
        # Probe in an upright temporary frame; keep output in the source frame.
        transform = np.eye(4)
        transform[up, up] = -1
        mesh.apply_transform(transform)
        low, high = lower.copy(), upper.copy()
        low[up], high[up] = -upper[up], -lower[up]
        result = support_under(mesh, low, high, up=up, meters_per_unit=meters_per_unit,
                               placement_residual_m=placement_residual_m)
        if result is not None:
            result["top_runtime_units"] *= -1
            result["aabb_min"][up], result["aabb_max"][up] = -result["aabb_max"][up], -result["aabb_min"][up]
        return result
    mesh.merge_vertices()
    result = (_level_face_support(mesh, lower, upper, up=up, meters_per_unit=meters_per_unit, window_m=window_m)
              or _fitted_support(mesh, lower, upper, up=up, meters_per_unit=meters_per_unit, window_m=window_m))
    if result is not None:
        result.update(search_window_m=window_m, search_window_basis=(
            "registration_task_region_residual" if window_m > 0.03 else "default"))
    return result


def _level_face_support(mesh: Any, lower: np.ndarray, upper: np.ndarray, *, up: int,
                        meters_per_unit: float, window_m: float = 0.03) -> dict[str, Any] | None:
    """Exact level faces (within 5 degrees) under every probe, coplanar and connected."""
    horizontal = [axis for axis in range(3) if axis != up]
    triangles = np.asarray(mesh.triangles, dtype=float)
    normals = np.asarray(mesh.face_normals)
    eligible = np.flatnonzero(np.abs(normals[:, up]) >= np.cos(np.deg2rad(5)))
    if not len(eligible):
        return None
    projected = triangles[eligible][:, :, horizontal]
    a, b, c = projected[:, 0], projected[:, 1], projected[:, 2]
    e1, e2 = b - a, c - a
    determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    safe = np.abs(determinant) > 1e-12
    height_tolerance = window_m / meters_per_unit
    flat_tolerance = 0.005 / meters_per_unit
    probes = np.array([[x, y] for x in np.linspace(lower[horizontal[0]], upper[horizontal[0]], 3)
                       for y in np.linspace(lower[horizontal[1]], upper[horizontal[1]], 3)])
    hits = []
    for point in probes:
        offset = point - a
        u = np.divide(offset[:, 0] * e2[:, 1] - offset[:, 1] * e2[:, 0], determinant,
                      out=np.zeros_like(determinant), where=safe)
        v = np.divide(e1[:, 0] * offset[:, 1] - e1[:, 1] * offset[:, 0], determinant,
                      out=np.zeros_like(determinant), where=safe)
        heights = (triangles[eligible, 0, up] + u * (triangles[eligible, 1, up] - triangles[eligible, 0, up])
                   + v * (triangles[eligible, 2, up] - triangles[eligible, 0, up]))
        valid = (safe & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
                 & (np.abs(heights - lower[up]) <= height_tolerance))
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            return None
        chosen = candidates[np.argmax(heights[candidates])]
        hits.append((int(eligible[chosen]), float(heights[chosen])))
    top = max(height for _, height in hits)
    if top - min(height for _, height in hits) > flat_tolerance:
        return None
    coplanar = set(int(i) for i in eligible
                  if np.max(np.abs(triangles[i, :, up] - top)) <= flat_tolerance)
    neighbors: dict[int, list[int]] = {}
    for first, second in mesh.face_adjacency:
        if int(first) in coplanar and int(second) in coplanar:
            neighbors.setdefault(int(first), []).append(int(second))
            neighbors.setdefault(int(second), []).append(int(first))
    connected, pending = set(), [hits[0][0]]
    while pending:
        face = pending.pop()
        if face in connected or face not in coplanar:
            continue
        connected.add(face)
        pending.extend(neighbors.get(face, ()))
    if not all(face in connected for face, _ in hits):
        return None
    points = triangles[sorted(connected)].reshape(-1, 3)
    return {"top_runtime_units": top, "aabb_min": points.min(axis=0).tolist(),
            "aabb_max": points.max(axis=0).tolist(), "face_indices": sorted(connected),
            "basis": "reconstructed_triangle_footprint_contacts", "physical_measurement": False}


#: Splat-derived meshes (Atlas) are bumpy. On the 2026-10-23 countertop 1-3% of
#: faces lay within 5 degrees of level and its top spread over 2-4 cm, so no
#: exact level face carried a footprint. When none does, the surface under the
#: footprint is fitted, under the same refusals: level and smooth, one
#: connected surface, and a real triangle of it under every probe.
FITTED_MAX_TILT_DEGREES = 5.0
FITTED_MAX_RMS_M = 0.015
FITTED_BAND_BOUNDS_M = (0.01, 0.03)
#: Faces steeper than this are a surface's edges and walls, never the surface.
FITTED_MAX_FACE_TILT_DEGREES = 60.0
#: A 10 cm seed fits a bumpy counter to about 5 degrees; each refit over the grown
#: surface halves that (dishwasher capture 2026-10-23: 5.2, 3.9, 2.9, 2.2, 1.6 degrees).
FITTED_ITERATIONS = 8
#: Two fits of one bumpy surface converge on nearly, not always exactly, the same faces.
SAME_FITTED_SURFACE_MIN_OVERLAP = 0.9
#: The triangles themselves, not only their fitted plane, must meet the object:
#: a corrugated mesh fits a smooth plane while its ridges stand centimetres
#: proud of it. Near each probe the median actual height must sit near the
#: plane: the real 2026-10-23 counter undulated up to 1.6 cm, a 0.66/0.74 m
#: corrugation sits 4 cm off. The object rests on the highest of them.
FITTED_MAX_CONTACT_DEVIATION_M = 0.025
FITTED_CONTACT_RADIUS_M = 0.01


def _plane(points: np.ndarray, up: int) -> tuple[np.ndarray, np.ndarray, float]:
    middle = points.mean(axis=0)
    normal = np.linalg.svd(points - middle, full_matrices=False)[2][-1]
    normal = normal if normal[up] >= 0 else -normal
    return normal, middle, float(np.sqrt(np.mean(((points - middle) @ normal) ** 2)))


def _fitted_support(mesh: Any, lower: np.ndarray, upper: np.ndarray, *, up: int,
                    meters_per_unit: float, window_m: float = 0.03) -> dict[str, Any] | None:
    """Fit the bumpy surface under the footprint, grow it over connected faces, and test it."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    horizontal = [axis for axis in range(3) if axis != up]
    triangles = np.asarray(mesh.triangles, dtype=float)
    centres = triangles.mean(axis=1)
    gentle = np.abs(np.asarray(mesh.face_normals)[:, up]) >= np.cos(np.deg2rad(FITTED_MAX_FACE_TILT_DEGREES))
    within = np.ones(len(centres), dtype=bool)
    for axis in horizontal:
        within &= (centres[:, axis] >= lower[axis]) & (centres[:, axis] <= upper[axis])
    tolerance = window_m / meters_per_unit
    seed = np.flatnonzero(gentle & within & (np.abs(centres[:, up] - lower[up]) <= tolerance))
    if window_m > 0.03 and len(seed):
        # A window wider than 3 cm can also reach a second surface under the
        # footprint, such as a counter's edge below its top (capture-447ae4f4:
        # 0.62-0.69 under a 0.72 top). Seed on the 3 cm band of the highest
        # surface it reached: the object rests on the highest of them.
        top = np.percentile(centres[seed, up], 90)
        seed = seed[np.abs(centres[seed, up] - top) <= 0.03 / meters_per_unit]
    if len(seed) < 3:
        return None
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
    low_band, high_band = (bound / meters_per_unit for bound in FITTED_BAND_BOUNDS_M)
    faces = seed
    for _ in range(FITTED_ITERATIONS):
        normal, middle, rms = _plane(centres[faces], up)
        near = gentle & (np.abs((centres - middle) @ normal) <= min(max(2.5 * rms, low_band), high_band))
        linked = adjacency[near[adjacency[:, 0]] & near[adjacency[:, 1]]]
        graph = coo_matrix((np.ones(len(linked)), (linked[:, 0], linked[:, 1])), shape=(len(centres),) * 2)
        labels = connected_components(graph, directed=False)[1]
        seeded = labels[seed[near[seed]]]
        if not len(seeded):
            return None
        grown = np.flatnonzero(near & (labels == np.bincount(seeded).argmax()))
        if np.array_equal(grown, faces):
            break
        faces = grown
    normal, middle, rms = _plane(centres[faces], up)
    tilt = float(np.degrees(np.arccos(min(1.0, float(normal[up])))))
    if tilt > FITTED_MAX_TILT_DEGREES or rms * meters_per_unit > FITTED_MAX_RMS_M:
        return None
    # A real triangle of this surface lies under every probe (no gap, no edge),
    # and the triangles around it stand near the plane there; the object rests on the highest.
    widened = window_m > 0.03
    reach = window_m / meters_per_unit
    pad = (max(FITTED_CONTACT_RADIUS_M, window_m) if widened else FITTED_CONTACT_RADIUS_M) / meters_per_unit
    extent = triangles[faces][:, :, horizontal]
    local = faces[np.all((extent.max(axis=1) >= lower[horizontal] - pad)
                         & (extent.min(axis=1) <= upper[horizontal] + pad), axis=1)]
    a, b, c = (triangles[local][:, corner][:, horizontal] for corner in range(3))
    za, zb, zc = (triangles[local][:, corner, up] for corner in range(3))
    e1, e2 = b - a, c - a
    determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    safe = np.abs(determinant) > 1e-12
    spread = np.linspace(-1.0, 1.0, 5) * FITTED_CONTACT_RADIUS_M / meters_per_unit
    corners = triangles[local].reshape(-1, 3)

    def hole(x: float, y: float) -> tuple[float, np.ndarray] | None:
        # Within the registration's residual a probe may sit over a hole in the
        # surface (where the removed object itself sat in the mesh), never past
        # its edge: the surface must lie on every side of it within the residual.
        if not widened or not len(corners):
            return None
        offset = corners[:, horizontal] - np.array([x, y])
        distance = np.hypot(offset[:, 0], offset[:, 1])
        near = distance <= reach
        around = offset[near]
        if not (len(around) and all((around[:, axis] < 0).any() and (around[:, axis] > 0).any()
                                    for axis in range(2))):
            return None
        return float(distance.min()), corners[near][:, up]

    contacts, gaps = [], []
    for x in np.linspace(lower[horizontal[0]], upper[horizontal[0]], 3):
        for y in np.linspace(lower[horizontal[1]], upper[horizontal[1]], 3):
            heights, nearby = [], None
            for dx in spread:
                for dy in spread:
                    offset = np.array([x + dx, y + dy]) - a
                    u = np.divide(offset[:, 0] * e2[:, 1] - offset[:, 1] * e2[:, 0], determinant,
                                  out=np.full_like(determinant, -1.0), where=safe)
                    v = np.divide(e1[:, 0] * offset[:, 1] - e1[:, 1] * offset[:, 0], determinant,
                                  out=np.full_like(determinant, -1.0), where=safe)
                    inside = safe & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
                    if dx == 0 and dy == 0 and not inside.any():
                        found = hole(x, y)
                        if found is None:
                            return None
                        gap, nearby = found
                        gaps.append({"probe_runtime_units": [float(x), float(y)], "gap_m": gap * meters_per_unit})
                    heights.extend((za + u * (zb - za) + v * (zc - za))[inside])
            if not heights:
                heights = nearby.tolist()
            plane = middle[up] - (normal[horizontal[0]] * (x - middle[horizontal[0]])
                                  + normal[horizontal[1]] * (y - middle[horizontal[1]])) / normal[up]
            contacts.append(float(np.median(heights)))
            if abs(contacts[-1] - plane) * meters_per_unit > FITTED_MAX_CONTACT_DEVIATION_M:
                return None
    top = max(contacts)
    if abs(top - lower[up]) > tolerance:
        return None
    points = triangles[faces].reshape(-1, 3)
    return {"top_runtime_units": top, "aabb_min": points.min(axis=0).tolist(),
            "aabb_max": points.max(axis=0).tolist(), "face_indices": faces.tolist(),
            "basis": "fitted_surface_under_footprint", "fit_rms_m": rms * meters_per_unit,
            "fit_tilt_degrees": tilt, "physical_measurement": False,
            **({"probe_gaps_m": gaps, "probe_gap_basis": "registration_task_region_residual"} if gaps else {})}


def same_support_surface(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether two support results name one surface: the same exact faces, or mostly the same fitted ones.

    Level faces nearly all within a fitted surface are that surface: a conditioned
    collider's flattened patch (website_collider_conditioning) reads level under a
    footprint wholly on it, and fitted under one that spans a gap in it.
    """
    if first["basis"] != second["basis"]:
        bases = {first["basis"]: first, second["basis"]: second}
        if set(bases) != {"fitted_surface_under_footprint", "reconstructed_triangle_footprint_contacts"}:
            return False
        level = set(bases["reconstructed_triangle_footprint_contacts"]["face_indices"])
        fitted = set(bases["fitted_surface_under_footprint"]["face_indices"])
        return bool(level) and len(level & fitted) >= SAME_FITTED_SURFACE_MIN_OVERLAP * len(level)
    if first["basis"] != "fitted_surface_under_footprint":
        return first["face_indices"] == second["face_indices"]
    a, b = set(first["face_indices"]), set(second["face_indices"])
    return len(a & b) >= SAME_FITTED_SURFACE_MIN_OVERLAP * len(a | b)


def destination_clear_of_start(start_min: Sequence[float], start_max: Sequence[float],
                               centre: Sequence[float], *, up: int, clearance: float) -> bool:
    """Whether a target disc of radius ``clearance`` at ``centre`` excludes the start's centre.

    The same rule as ``surface_target_initial_overlap``
    (``task_evaluation_surface_target.region_excludes_start``) in this frame:
    otherwise the task could begin complete.
    """
    plan = [axis for axis in range(3) if axis != up]
    middle = [(float(start_min[axis]) + float(start_max[axis])) / 2 for axis in plan]
    return math.dist(middle, [float(centre[axis]) for axis in plan]) > clearance


#: The target disc's radius beyond the subject's half-diagonal (the website task's green region).
DESTINATION_MARGIN_M = 0.01
#: A relocated disc left tangent to its surface's bounds is moved this far inside them, so the
#: runtime-to-simulator conversion cannot round it past the edge the admission checks
#: (capture-447ae4f4 at the replacement's 0.25 size tolerance: 5.6e-17 m past, refused).
DESTINATION_EDGE_GUARD_M = 1e-9
#: Plan positions tried along each axis when a destination must move off the start.
DESTINATION_GRID_STEPS = 9
#: Footprints checked against the collider, farthest first, before none is placed. Only
#: footprints whose probe points all lie over the surface's own faces count against it:
#: a round table's bounding-box corners would otherwise use it up.
DESTINATION_MAX_PROBES = 24


def _plan_distance_to_triangles(point: Sequence[float], triangles: np.ndarray) -> float:
    """A plan point's distance to the nearest of ``triangles`` (shape (n, 3, 2)); zero over one."""
    p = np.asarray(point, dtype=float)
    corners = [triangles[:, i] for i in range(3)]
    crosses = []
    nearest = np.full(len(triangles), np.inf)
    for start, end in ((corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])):
        edge, offset = end - start, p - start
        crosses.append(edge[:, 0] * offset[:, 1] - edge[:, 1] * offset[:, 0])
        t = np.clip((offset * edge).sum(axis=1) / np.maximum((edge * edge).sum(axis=1), 1e-18), 0.0, 1.0)
        nearest = np.minimum(nearest, np.linalg.norm(offset - t[:, None] * edge, axis=1))
    a, b, c = crosses
    inside = ((a >= 0) & (b >= 0) & (c >= 0)) | ((a <= 0) & (b <= 0) & (c <= 0))
    return 0.0 if inside.any() else float(nearest.min()) if len(nearest) else math.inf


def farthest_supported_destination(mesh: Any, start_min: Sequence[float], start_max: Sequence[float],
                                   region_min: Sequence[float], region_max: Sequence[float],
                                   start_support: dict[str, Any], *, up: int, meters_per_unit: float,
                                   up_sign: int = 1, placement_residual_m: float | None = None,
                                   clearance: float) -> tuple[list[float], dict[str, Any]] | None:
    """The start footprint moved as far from the start as its own surface and ``region`` allow.

    Only for a destination grounded as the very surface the subject rests on
    (capture-447ae4f4: the whole dining table for "the other end of the same
    table", whose centre lay under the tray). Each candidate keeps the target
    disc inside ``region`` and the support's bounds and clear of the start, and
    must rest on the same surface by ``support_under`` against real triangles.
    Returns ``(centre, support)`` in runtime units, or None when none does.
    """
    start_min, start_max = np.asarray(start_min, dtype=float), np.asarray(start_max, dtype=float)
    plan = [axis for axis in range(3) if axis != up]
    half, origin = (start_max - start_min) / 2, (start_min + start_max) / 2
    steps = []
    for axis in plan:
        low = max(float(region_min[axis]) + half[axis], float(start_support["aabb_min"][axis]) + clearance)
        high = min(float(region_max[axis]) - half[axis], float(start_support["aabb_max"][axis]) - clearance)
        if low > high:
            return None
        steps.append(np.linspace(low, high, DESTINATION_GRID_STEPS))
    candidates = []
    for first, second in ((a, b) for a in steps[0] for b in steps[1]):
        centre = origin.copy()
        centre[plan[0]], centre[plan[1]] = first, second
        if destination_clear_of_start(start_min, start_max, centre, up=up, clearance=clearance):
            candidates.append(centre)
    faces = np.asarray(start_support.get("face_indices") or [], dtype=np.int64)
    if len(faces):
        # The same nine probe points support_under tests, each over or within its
        # search window of one of this surface's triangles (in plan, so any
        # tessellation); the exact test follows below.
        triangles = np.asarray(mesh.triangles, dtype=float)[faces][:, :, plan]
        window_m = 0.03
        if (not isinstance(placement_residual_m, bool) and isinstance(placement_residual_m, (int, float))
                and math.isfinite(placement_residual_m)):
            window_m = max(window_m, float(placement_residual_m))
        from scipy.spatial import cKDTree

        # Only triangles whose centre lies within their own reach plus the window can qualify.
        centres = triangles.mean(axis=1)
        reach = float(np.linalg.norm(triangles - centres[:, None], axis=2).max()) + window_m / meters_per_unit
        nearby = cKDTree(centres)
        signs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]

        def supported(point: list[float]) -> bool:
            rows = nearby.query_ball_point(point, reach)
            return bool(rows) and _plan_distance_to_triangles(point, triangles[rows]) <= window_m / meters_per_unit

        candidates = [centre for centre in candidates if all(
            supported([centre[plan[0]] + a * half[plan[0]], centre[plan[1]] + b * half[plan[1]]]) for a, b in signs)]
    candidates.sort(key=lambda centre: -math.dist(centre[plan], origin[plan]))
    for centre in candidates[:DESTINATION_MAX_PROBES]:
        offset = centre - origin
        found = support_under(mesh, start_min + offset, start_max + offset, up=up, meters_per_unit=meters_per_unit,
                              up_sign=up_sign, placement_residual_m=placement_residual_m)
        if found is not None and same_support_surface(found, start_support):
            centre = centre.copy()
            centre[up] = found["top_runtime_units"]
            # Only a centre within the guard of a bound moves; every other one is returned exactly.
            guard = DESTINATION_EDGE_GUARD_M / meters_per_unit
            for axis in plan:
                centre[axis] = min(max(centre[axis], float(start_support["aabb_min"][axis]) + clearance + guard),
                                   float(start_support["aabb_max"][axis]) - clearance - guard)
            return centre.tolist(), found
    return None


#: The unobserved lowest band of a built-in, floor-standing assembly (a toe
#: kick and the bottom of a front panel) is at most this tall.
FLOOR_GROUNDING_MAX_GAP_M = 0.35
#: How far around the footprint the floor it stands on is looked for.
FLOOR_SEARCH_MARGIN_M = 0.6
FLOOR_MIN_AREA_M2 = 0.1
#: A flat surface under at least this share of the footprint is somewhere
#: the object could rest instead; slivers of a generated bay are not.
INTERMEDIATE_SURFACE_MIN_FOOTPRINT_SHARE = 0.5


def ground_on_observed_floor(mesh: Any, lower: Sequence[float], upper: Sequence[float], *, up: int,
                             meters_per_unit: float, floor_height: float, up_sign: int = 1) -> dict[str, Any] | None:
    """The floor a rebuilt, built-in, floor-standing assembly stands on.

    The assembly's box is its observed body; the band nearest the floor is
    usually hidden (behind the door, a kick plate), and a generated world
    often leaves the empty bay under it with no flat face. ``floor_height`` is
    the floor the registration observed in the footage, along ``up`` in the
    source frame. The collider's own floor beside the footprint is used when
    it lies at that height; otherwise the observed floor plane is, and the
    record says the generated world had no floor there. Refused when the gap
    is larger than an unobserved kick band, or a real surface lies between.
    The caller extends the body down to the returned top; nothing is moved.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(upper <= lower) or up not in {1, 2} or up_sign not in {-1, 1}
            or not np.isfinite(meters_per_unit) or meters_per_unit <= 0 or not np.isfinite(floor_height)):
        raise ValueError("website_support_query_invalid")
    if up_sign == -1:
        mesh = mesh.copy()
        transform = np.eye(4)
        transform[up, up] = -1
        mesh.apply_transform(transform)
        low, high = lower.copy(), upper.copy()
        low[up], high[up] = -upper[up], -lower[up]
        result = ground_on_observed_floor(mesh, low, high, up=up, meters_per_unit=meters_per_unit,
                                          floor_height=-floor_height)
        if result is not None:
            result["top_runtime_units"] *= -1
            result["aabb_min"][up], result["aabb_max"][up] = -result["aabb_max"][up], -result["aabb_min"][up]
        return result
    gap = float(lower[up] - floor_height)
    if not 0.0 < gap * meters_per_unit <= FLOOR_GROUNDING_MAX_GAP_M:
        return None
    horizontal = [axis for axis in range(3) if axis != up]
    triangles = np.asarray(mesh.triangles, dtype=float)
    centers = triangles.mean(axis=1)
    areas = np.asarray(mesh.area_faces) * meters_per_unit ** 2
    level = np.abs(np.asarray(mesh.face_normals)[:, up]) >= np.cos(np.deg2rad(5))
    tolerance = 0.03 / meters_per_unit
    margin = FLOOR_SEARCH_MARGIN_M / meters_per_unit
    near = np.ones(len(triangles), dtype=bool)
    within = np.ones(len(triangles), dtype=bool)
    for axis in horizontal:
        near &= (centers[:, axis] >= lower[axis] - margin) & (centers[:, axis] <= upper[axis] + margin)
        within &= (centers[:, axis] >= lower[axis]) & (centers[:, axis] <= upper[axis])
    footprint_m2 = float(np.prod([(upper[a] - lower[a]) * meters_per_unit for a in horizontal]))
    between = level & within & (centers[:, up] > floor_height + tolerance) & (centers[:, up] < lower[up] + tolerance)
    if areas[between].sum() >= INTERMEDIATE_SURFACE_MIN_FOOTPRINT_SHARE * footprint_m2:
        return None
    floor = np.flatnonzero(level & near & (np.abs(centers[:, up] - floor_height) <= tolerance))
    footprint_min, footprint_max = lower.copy(), upper.copy()
    if areas[floor].sum() >= FLOOR_MIN_AREA_M2:
        top = float(np.average(centers[floor, up], weights=areas[floor]))
        points = triangles[floor].reshape(-1, 3)
        support_min, support_max = points.min(axis=0), points.max(axis=0)
        basis, faces = "collider_floor_beside_footprint", floor.tolist()
    else:
        top = float(floor_height)
        footprint_min[up] = footprint_max[up] = top
        support_min, support_max = footprint_min, footprint_max
        basis, faces = "registered_observed_floor_plane", []
    return {"top_runtime_units": top, "aabb_min": support_min.tolist(), "aabb_max": support_max.tolist(),
            "face_indices": faces, "extended_runtime_units": float(lower[up] - top),
            "collider_floor_area_m2": float(areas[floor].sum()), "basis": basis,
            "intermediate_surface_area_m2": float(areas[between].sum()), "physical_measurement": False}
