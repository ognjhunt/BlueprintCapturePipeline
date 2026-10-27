"""Find a conservative open container interior in capture-derived triangles.

Beta: job video to a measurable placement task. A destination mask only locates
the container. Its floor, four surrounding walls and unobstructed opening must
exist in the retained collider; no unseen interior is generated here.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import numpy as np

from .website_support_geometry import support_under


def _hit(triangles: np.ndarray, origin: np.ndarray, direction: np.ndarray):
    """Nearest positive two-sided ray/triangle hit, without an optional rtree."""
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = np.cross(np.broadcast_to(direction, second.shape), second)
    determinant = np.einsum("ij,ij->i", first, cross)
    safe = np.abs(determinant) > 1e-12
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=safe)
    offset = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", offset, cross) * inverse
    q = np.cross(offset, first)
    v = q @ direction * inverse
    distance = np.einsum("ij,ij->i", second, q) * inverse
    valid = safe & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (distance > 1e-8)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    index = int(indices[np.argmin(distance[indices])])
    return origin + distance[index] * direction, index


def _wall_covers_rectangle(triangles, *, axis, plane, other, lower, upper, tolerance):
    """A planar triangle union covers the side: corners covered, no internal boundary.

    Ray samples alone could mistake two rails for a wall. A hole in a planar
    triangle mesh necessarily has a boundary in this rectangle. Nonmanifold or
    overlapping triangulations conservatively fail rather than hiding that hole.
    """
    selected = np.flatnonzero(np.all(np.abs(triangles[:, :, axis] - plane) <= tolerance, axis=1))
    projected = triangles[selected][:, :, other]
    if not len(projected):
        return None
    points = [np.asarray(point) for point in product(*zip(lower, upper, strict=True))]
    for point in points:
        a, b, c = projected[:, 0], projected[:, 1], projected[:, 2]
        e1, e2, offset = b - a, c - a, point - a
        determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        safe = np.abs(determinant) > 1e-12
        u = np.divide(offset[:, 0] * e2[:, 1] - offset[:, 1] * e2[:, 0], determinant,
                      out=np.full_like(determinant, -1.), where=safe)
        v = np.divide(e1[:, 0] * offset[:, 1] - e1[:, 1] * offset[:, 0], determinant,
                      out=np.full_like(determinant, -1.), where=safe)
        if not np.any(safe & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)):
            return None
    edges = {}
    seen = set()
    for triangle in projected:
        vertices = [tuple(np.round(vertex, 9)) for vertex in triangle]
        key = tuple(sorted(vertices))
        if key in seen:
            continue
        seen.add(key)
        for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=True):
            edge = tuple(sorted((a, b)))
            edges[edge] = edges.get(edge, 0) + 1
    # Clip every mesh boundary segment against the open rectangle. A boundary
    # there describes missing or ambiguous wall coverage, even between samples.
    for (a, b), count in edges.items():
        if count == 2:
            continue
        first, last = 0., 1.
        for index in range(2):
            delta = b[index] - a[index]
            lo, hi = lower[index] + 1e-8, upper[index] - 1e-8
            if abs(delta) < 1e-12:
                if not lo < a[index] < hi:
                    first, last = 1., 0.
                    break
            else:
                near, far = sorted(((lo - a[index]) / delta, (hi - a[index]) / delta))
                first, last = max(first, near), min(last, far)
        if first < last:
            return None
    return selected.tolist()


def container_interior(mesh: Any, lower: Sequence[float], upper: Sequence[float], *,
                       subject_dimensions: Sequence[float], up: int, meters_per_unit: float,
                       up_sign: int = 1) -> dict[str, Any] | None:
    """Return a bounded empty cavity and its real floor, or no geometry.

    This deliberately supports open, roughly rectangular interiors. Cardinal
    wall hits are checked over the proposed object's footprint at two heights;
    a conservative triangle-envelope check rejects intrusions and lids. Bounds
    remain estimates, and the later robot-free physics check remains required.
    """
    low, high, dimensions = (np.asarray(value, dtype=float) for value in (lower, upper, subject_dimensions))
    if (any(value.shape != (3,) for value in (low, high, dimensions))
            or not np.isfinite([low, high, dimensions]).all() or np.any(high <= low)
            or np.any(dimensions <= 0) or up not in {1, 2} or up_sign not in {-1, 1}
            or not np.isfinite(meters_per_unit) or meters_per_unit <= 0):
        raise ValueError("website_container_query_invalid")
    if up_sign == -1:
        upright = mesh.copy()
        transform = np.eye(4)
        transform[up, up] = -1
        upright.apply_transform(transform)
        flipped_low, flipped_high = low.copy(), high.copy()
        flipped_low[up], flipped_high[up] = -high[up], -low[up]
        value = container_interior(upright, flipped_low, flipped_high, subject_dimensions=dimensions,
                                   up=up, meters_per_unit=meters_per_unit)
        if value is not None:
            for bounds in (value["interior_bounds"], value["floor"]):
                lo_key, hi_key = (("minimum", "maximum") if bounds is value["interior_bounds"]
                                  else ("aabb_min", "aabb_max"))
                bounds[lo_key][up], bounds[hi_key][up] = -bounds[hi_key][up], -bounds[lo_key][up]
            value["floor"]["top_runtime_units"] *= -1
        return value
    triangles = np.asarray(mesh.triangles, dtype=float)
    if not len(triangles) or not np.isfinite(triangles).all():
        return None
    horizontal = [axis for axis in range(3) if axis != up]
    margin = 0.002 / meters_per_unit
    center = (low + high) / 2
    origin = center.copy()
    origin[up] = high[up] + margin
    down = np.zeros(3)
    down[up] = -1
    bottom = _hit(triangles, origin, down)
    if bottom is None:
        return None
    floor_height = float(bottom[0][up])
    depth = high[up] - floor_height
    if floor_height < low[up] - margin or depth <= dimensions[up] + margin:
        return None
    floor_low, floor_high = center - dimensions / 2, center + dimensions / 2
    floor_low[up], floor_high[up] = floor_height, floor_height + dimensions[up]
    floor = support_under(mesh, floor_low, floor_high, up=up, meters_per_unit=meters_per_unit)
    if floor is None:
        return None
    floor_height = float(floor["top_runtime_units"])
    interior_low, interior_high = low.copy(), high.copy()
    interior_low[up] = floor_height
    wall_faces = set()
    # The walls must surround the full footprint, not just one central ray.
    for axis in horizontal:
        other = next(i for i in horizontal if i != axis)
        for sign in (-1, 1):
            edges = []
            for fraction, height in product((-0.5, 0.0, 0.5), (0.25, 0.75)):
                ray = center.copy()
                ray[other] += fraction * dimensions[other]
                ray[up] = floor_height + height * depth
                direction = np.zeros(3)
                direction[axis] = sign
                hit = _hit(triangles, ray, direction)
                if hit is None or not low[axis] - margin <= hit[0][axis] <= high[axis] + margin:
                    return None
                edges.append(float(hit[0][axis]))
                wall_faces.add(hit[1])
            if sign == -1:
                interior_low[axis] = max(edges) + margin
            else:
                interior_high[axis] = min(edges) - margin
    if np.any(interior_high - interior_low <= dimensions + margin):
        return None
    for axis in horizontal:
        other = [next(i for i in horizontal if i != axis), up]
        for sign in (-1, 1):
            plane = interior_low[axis] - margin if sign == -1 else interior_high[axis] + margin
            faces = _wall_covers_rectangle(triangles, axis=axis, plane=plane, other=other,
                lower=interior_low[other] + margin, upper=interior_high[other] - margin, tolerance=margin / 2)
            if faces is None:
                return None
            wall_faces.update(faces)
    # Whole-cavity emptiness is conservative: a triangle's enclosing box may
    # reject awkward geometry, but can never silently accept an obstruction.
    inner_low, inner_high = interior_low + margin, interior_high - margin
    overlaps = np.all((triangles.max(axis=1) > inner_low) & (triangles.min(axis=1) < inner_high), axis=1)
    if overlaps.any():
        return None
    # The measured walls can shift the cavity centre from the mask centre.
    # Bind support under that actual placement rather than reusing a remote patch.
    placement = (interior_low + interior_high) / 2
    floor_low, floor_high = placement - dimensions / 2, placement + dimensions / 2
    floor_low[up], floor_high[up] = floor_height, floor_height + dimensions[up]
    floor = support_under(mesh, floor_low, floor_high, up=up, meters_per_unit=meters_per_unit)
    if floor is None:
        return None
    return {"interior_bounds": {"minimum": interior_low.tolist(), "maximum": interior_high.tolist()},
            "floor": floor, "wall_face_indices": sorted(wall_faces),
            "basis": "reconstructed_open_cavity_floor_and_surrounding_walls", "physical_measurement": False}
