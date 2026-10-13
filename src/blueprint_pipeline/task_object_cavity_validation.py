"""Independent cavity checks shared by current and historical authoring adapters.

These probe final collision bytes; they do not author or plan geometry.
"""
from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from typing import Any

CAVITY_PROBE_STANDOFF_M = 0.01
CAVITY_MINIMUM_CLEAR_FRACTION = 0.6
CAVITY_PROBE_OFFSETS = (-0.3, 0.0, 0.3)
CAVITY_FIXTURE_MARGIN_M = 0.01
CAVITY_FIXTURE_MAXIMUM_VOLUME_FRACTION = 0.1

def _declared_fixture_boxes(cavity: Mapping[str, Any], *, maximum_volume_fraction=CAVITY_FIXTURE_MAXIMUM_VOLUME_FRACTION) -> list[tuple[list[float], list[float]]] | None:
    """A cavity's declared fixture boxes, or None when any is malformed, outside it or too large."""
    rows = cavity.get("declared_fixtures", [])
    if not isinstance(rows, list):
        return None
    cx, cy, cz = (float(v) for v in cavity["opening_center_link_m"])
    depth, half_w, half_h = (float(cavity[k]) for k in ("inner_depth_m", "opening_width_m", "opening_height_m"))
    half_w, half_h = half_w / 2, half_h / 2
    c_lo, c_hi = [cx - depth, cy - half_w, cz - half_h], [cx, cy + half_w, cz + half_h]
    boxes = []
    for row in rows:
        box = row.get("box_link_m") if isinstance(row, Mapping) else None
        try:
            lo, hi = [float(v) for v in box["minimum"]], [float(v) for v in box["maximum"]]
        except (KeyError, TypeError, ValueError):
            return None
        if (len(lo) != 3 or len(hi) != 3 or not all(map(math.isfinite, lo + hi))
                or any(not c_lo[i] - 1e-4 <= lo[i] < hi[i] <= c_hi[i] + 1e-4 for i in range(3))  # 5 dp rounding
                or math.prod(hi[i] - lo[i] for i in range(3))
                > maximum_volume_fraction * depth * 4 * half_w * half_h):
            return None
        boxes.append((lo, hi))
    return boxes

def interior_cavity_findings(vertices: Any, faces: Any, cavities: Sequence[Mapping[str, Any]], *, fixture_maximum_volume_fraction=CAVITY_FIXTURE_MAXIMUM_VOLUME_FRACTION) -> list[str]:
    """Rays from just outside each planned opening, along -X, must travel into a hollow and hit a closed back.

    Mesh coordinates are in the cavity's link frame. The first hit of every
    probe ray must lie at least ``CAVITY_MINIMUM_CLEAR_FRACTION`` of the planned
    inner depth past the opening; a solid block or a thin slab stops the ray
    early, a missing back lets it escape. Hits inside a box the cavity declares
    under ``declared_fixtures`` (planned tub hardware) are not obstructions;
    an undeclared, oversized or out-of-cavity box is.
    """
    import numpy as np

    triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int).reshape(-1, 3)]
    if not len(cavities):
        return ["body_interior_cavity_unplanned"]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    # Rays run along -X, so intersections reduce to 2-D point-in-triangle tests in YZ.
    e1, e2 = b[:, 1:] - a[:, 1:], c[:, 1:] - a[:, 1:]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    usable = np.abs(det) > 1e-14
    m = CAVITY_FIXTURE_MARGIN_M

    def clear_depth(origin_x: float, y: float, z: float, fixtures) -> float | None:
        rel = np.array([y, z]) - a[:, 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (rel[:, 0] * e2[:, 1] - rel[:, 1] * e2[:, 0]) / det
            v = (e1[:, 0] * rel[:, 1] - e1[:, 1] * rel[:, 0]) / det
            inside = usable & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        hit_x = a[inside, 0] + u[inside] * (b[inside, 0] - a[inside, 0]) + v[inside] * (c[inside, 0] - a[inside, 0])
        keep = hit_x < origin_x
        for lo, hi in fixtures:
            if lo[1] - m <= y <= hi[1] + m and lo[2] - m <= z <= hi[2] + m:
                keep &= (hit_x < lo[0] - m) | (hit_x > hi[0] + m)
        ahead = hit_x[keep]
        return float(origin_x - ahead.max()) - CAVITY_PROBE_STANDOFF_M if len(ahead) else None

    findings = []
    for cavity in cavities:
        cx, cy, cz = (float(v) for v in cavity["opening_center_link_m"])
        half_w, half_h = float(cavity["opening_width_m"]) / 2, float(cavity["opening_height_m"]) / 2
        fixtures = _declared_fixture_boxes(cavity, maximum_volume_fraction=fixture_maximum_volume_fraction)
        if fixtures is None:
            findings.append("body_interior_cavity_fixture_invalid:" + str(cavity["cavity_id"]))
            continue
        depths = [clear_depth(cx + CAVITY_PROBE_STANDOFF_M, cy + dy * half_w, cz + dz * half_h, fixtures)
                  for dy in CAVITY_PROBE_OFFSETS for dz in CAVITY_PROBE_OFFSETS]
        if any(depth is None for depth in depths):
            findings.append("body_interior_cavity_back_open:" + str(cavity["cavity_id"]))
        elif min(depths) < CAVITY_MINIMUM_CLEAR_FRACTION * float(cavity["inner_depth_m"]):
            findings.append("body_interior_cavity_closed:" + str(cavity["cavity_id"]))
    return findings

def collision_cavity_findings(pieces: Sequence[Mapping[str, Any]],
                              cavities: Sequence[Mapping[str, Any]]) -> list[str]:
    """Probe each planned cavity on the geometry PhysX will collide with, per declared approximation.

    Pieces are link-frame triangles. An ``analytic_box`` is exact, a
    ``convexHull`` mesh is probed as its hull, and any other approximation
    (``convexDecomposition``, a triangle mesh on a dynamic body) cannot be
    proven hollow before cooking, so it fails closed.
    """
    import numpy as np
    import trimesh

    vertices: list[Any] = []
    faces: list[Any] = []
    unproven = []
    for piece in pieces:
        approximation = str(piece.get("approximation") or "")
        points = np.asarray(piece.get("vertices") or [], dtype=float).reshape(-1, 3)
        triangles = np.asarray(piece.get("faces") or [], dtype=int).reshape(-1, 3)
        if approximation == "convexHull" and len(points) >= 4:
            hull = trimesh.Trimesh(points, triangles, process=False).convex_hull
            points, triangles = np.asarray(hull.vertices), np.asarray(hull.faces)
        elif approximation != "analytic_box":
            unproven.append("body_cavity_collision_approximation_unproven:" + (approximation or "unspecified"))
            continue
        faces.append(triangles + sum(len(v) for v in vertices))
        vertices.append(points)
    if unproven:
        return sorted(set(unproven))
    if not vertices:
        return ["body_cavity_collision_missing"]
    codes = interior_cavity_findings(np.concatenate(vertices), np.concatenate(faces), cavities)
    return [code.replace("body_interior_cavity_", "body_cavity_collision_", 1) for code in codes]

