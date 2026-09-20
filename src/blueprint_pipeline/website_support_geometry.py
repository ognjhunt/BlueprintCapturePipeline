"""Task-footprint contact against reconstructed triangles (ADP-030, day 28).

These are estimated collider contacts, never measurements. A surface's AABB
alone cannot show that it supports the task object: gaps and table edges matter.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def support_under(mesh: Any, lower: Sequence[float], upper: Sequence[float], *,
                  up: int, meters_per_unit: float, up_sign: int = 1) -> dict[str, Any] | None:
    """Find a nearby, connected, nearly horizontal surface under the footprint.

    Probe the center, corners and edge midpoints; retain the actual face set for
    downstream contact checks. Refuse a remote floor or disconnected islands.
    """
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
        result = support_under(mesh, low, high, up=up, meters_per_unit=meters_per_unit)
        if result is not None:
            result["top_runtime_units"] *= -1
            result["aabb_min"][up], result["aabb_max"][up] = -result["aabb_max"][up], -result["aabb_min"][up]
        return result
    mesh.merge_vertices()
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
    height_tolerance = 0.03 / meters_per_unit
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
