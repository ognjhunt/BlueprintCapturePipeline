"""Task-footprint contact against reconstructed triangles (ADP-030, day 28).

These are estimated collider contacts, never measurements. A surface's AABB
alone cannot show that it supports the task object: gaps and table edges matter.

A support lookup reads the collider as ``Trimesh.merge_vertices`` leaves it, in
bounded memory (``MergedCollider``): the merge is worked out once per collider,
triangles are read a block of faces at a time, and face adjacency is found only
among the faces a lookup could connect. A whole-mesh copy, merge and
``face_adjacency`` per lookup took a website preparation past 4.8 GB at
production density (6.5 M triangles) on an 8.3 GB host with no swap (#3339).
"""
from __future__ import annotations

import math
import weakref
from typing import Any, Iterator, Mapping, NamedTuple, Sequence

import numpy as np

#: Faces read at a time when a lookup scans the collider: one block's triangles are 19 MB of
#: float64, as #3330 bounds its ray casts (``RAY_PLANE_BLOCK``).
SUPPORT_FACE_BLOCK = 1 << 18
#: Vertices merged at a time: equal positions always fall in the same bucket.
MERGE_BUCKET_VERTICES = 1 << 19
_BUCKET_MIX = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                        0xD6E8FEB86659FD93, 0xFF51AFD7ED558CCD], dtype=np.uint64)


def _row_blocks(count: int) -> Iterator[slice]:
    """Slices of at most ``SUPPORT_FACE_BLOCK`` rows (one more at the end), never one row unless ``count`` is 1.

    numpy computes a lone (1, 3) row's ``@`` with another BLAS routine than a block's, which
    can differ in the last bit; in a block of two or more rows each row gets exactly the value
    it gets in the whole array.
    """
    size, start = max(2, SUPPORT_FACE_BLOCK), 0
    while start < count:
        stop = min(count, start + size)
        if count - stop == 1:
            stop = count
        yield slice(start, stop)
        start = stop


def _joined(parts: list[np.ndarray], dtype: Any = np.int64) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)


class _MergedRead:
    """What ``Trimesh.merge_vertices`` would make of one collider, worked out once.

    ``representative[v]`` is the vertex ``v`` merges into: the first referenced vertex
    with the same rounded position (and UV, on a textured mesh), the one
    ``merge_vertices`` keeps. Each upright frame (``up``, flipped or not) also keeps
    its faces within 60 degrees of level, their centres, and those within 5 degrees.
    """

    def __init__(self, mesh: Any):
        from trimesh import tol, util

        vertices, faces = _arrays(mesh)
        count = len(vertices)
        referenced = np.ones(count, dtype=bool)
        if len(faces):
            referenced[:] = False
            for block in _row_blocks(len(faces)):
                referenced[faces[block]] = True
        visual = getattr(mesh, "visual", None)
        uv = None
        if (visual is not None and visual.defined and visual.kind == "texture"
                and visual.uv is not None and len(visual.uv) == count):
            uv = np.asarray(visual.uv)
        digits = util.decimal_to_digits(tol.merge)

        def keys(index: Any) -> np.ndarray:
            stacked = [vertices[index] * (10 ** digits)]
            if uv is not None:
                stacked.append(uv[index] * (10 ** 4))
            return np.column_stack(stacked).round().astype(np.int64)

        self.representative = np.arange(count, dtype=np.int64)
        buckets = max(1, -(-int(np.count_nonzero(referenced)) // MERGE_BUCKET_VERTICES))
        if buckets > 1:
            bucket = np.zeros(count, dtype=np.int32)
            for block in _row_blocks(count):
                rows = keys(block).view(np.uint64)
                mixed = (rows * _BUCKET_MIX[:rows.shape[1]]).sum(axis=1, dtype=np.uint64)
                bucket[block] = (mixed % np.uint64(buckets)).astype(np.int32)
        for index in range(buckets):
            members = np.flatnonzero(referenced & (bucket == index)) if buckets > 1 else np.flatnonzero(referenced)
            if len(members):
                _, first, inverse = np.unique(keys(members), axis=0, return_index=True, return_inverse=True)
                self.representative[members] = members[first[inverse.reshape(-1)]]
        self.frames: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(mesh.vertices, dtype=np.float64).view(np.ndarray),
            np.asarray(mesh.faces, dtype=np.int64).view(np.ndarray).reshape(-1, 3))


def _reflection(axis: int) -> np.ndarray:
    matrix = np.eye(4)
    matrix[axis, axis] = -1
    return matrix


def _reflected(points: np.ndarray, reflection: np.ndarray) -> np.ndarray:
    """(n, 3, 3) corners moved as ``Trimesh.apply_transform`` moves vertices."""
    from trimesh.transformations import transform_points

    return transform_points(points.reshape(-1, 3), reflection).reshape(-1, 3, 3)


def triangle_blocks(mesh: Any, *, reflect: int | None = None) -> Iterator[tuple[slice, np.ndarray]]:
    """``mesh.triangles`` a block of faces at a time, as ``(faces, triangles)``, never cached on ``mesh``.

    With ``reflect`` (an axis), those of ``mesh.copy().apply_transform(...)`` reflecting that
    axis: the axis negated and every face's winding reversed, as trimesh moves them.
    """
    vertices, faces = _arrays(mesh)
    reflection = None if reflect is None else _reflection(reflect)
    for block in _row_blocks(len(faces)):
        if reflection is None:
            yield block, vertices[faces[block]]
        else:
            yield block, _reflected(vertices[faces[block][:, ::-1]], reflection)


#: One merged read per live collider, dropped with it (id -> (weak reference, data stamp, read)).
_MERGED_READS: dict[int, tuple[Any, Any, _MergedRead]] = {}


def _merged_read(mesh: Any) -> _MergedRead:
    """The collider's merged read: computed on its first lookup, reused until its data changes."""
    visual = getattr(mesh, "visual", None)
    try:
        stamp = (hash(mesh), hash(visual) if getattr(visual, "kind", None) == "texture" else None)
    except TypeError:
        return _MergedRead(mesh)
    key = id(mesh)
    entry = _MERGED_READS.get(key)
    if entry is not None and entry[0]() is mesh and entry[1] == stamp:
        return entry[2]
    read = _MergedRead(mesh)
    try:
        reference = weakref.ref(mesh, lambda _, key=key: _MERGED_READS.pop(key, None))
    except TypeError:
        return read
    _MERGED_READS[key] = (reference, stamp, read)
    return read


class MergedCollider:
    """A collider as support_under reads it, in an upright frame, without copying it.

    The same triangles, face normals and face adjacency as ``mesh.copy()`` with
    ``merge_vertices()`` (and, ``flipped``, the reflection of ``up`` applied first),
    read a block of faces at a time.
    """

    def __init__(self, mesh: Any, *, up: int, flipped: bool = False):
        self.vertices, self.faces = _arrays(mesh)
        self.up, self.flipped = up, flipped
        read = _merged_read(mesh)
        self.representative = read.representative
        self.reflection = _reflection(up) if flipped else None
        if (up, flipped) not in read.frames:
            read.frames[(up, flipped)] = self._frame()
        self.gentle, self.centres, self.level = read.frames[(up, flipped)]

    def corners(self, faces: Any) -> np.ndarray:
        """Merged vertex indices of ``faces`` in this frame's winding (a reflection reverses it)."""
        corners = self.representative[self.faces[faces]]
        return np.ascontiguousarray(corners[:, ::-1]) if self.flipped else corners

    def triangles(self, faces: Any) -> np.ndarray:
        """``mesh.triangles`` of the merged, upright copy, for ``faces`` only."""
        points = self.vertices[self.corners(faces)]
        return points if self.reflection is None else _reflected(points, self.reflection)

    def _frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from trimesh import triangles as triangle_math

        gentle, centres, level = [], [np.zeros((0, 3))], []
        for block in _row_blocks(len(self.faces)):
            triangles = self.triangles(block)
            # mesh.face_normals: zero for a face with no area.
            unit, valid = triangle_math.normals(triangles=triangles, crosses=triangle_math.cross(triangles))
            normals = np.zeros((len(triangles), 3))
            normals[valid] = unit
            tilt = np.abs(normals[:, self.up])
            keep = tilt >= np.cos(np.deg2rad(FITTED_MAX_FACE_TILT_DEGREES))
            gentle.append(np.flatnonzero(keep) + block.start)
            centres.append(triangles[keep].mean(axis=1))
            level.append(np.flatnonzero(tilt >= np.cos(np.deg2rad(5))) + block.start)
        return _joined(gentle), np.concatenate(centres), _joined(level)

    def bounds(self, faces: np.ndarray) -> tuple[list[float], list[float]]:
        """``triangles[faces].reshape(-1, 3)``'s minimum and maximum, a block of faces at a time.

        Equal to one reduction over all of them, unless both signed zeros meet at an
        extreme (which one a reduction keeps depends on its order): then read whole.
        """
        if len(faces) <= SUPPORT_FACE_BLOCK:
            points = self.triangles(faces).reshape(-1, 3)
            return points.min(axis=0).tolist(), points.max(axis=0).tolist()
        lows, highs, signed = [], [], np.zeros((2, 3), dtype=bool)
        for block in _row_blocks(len(faces)):
            points = self.triangles(faces[block]).reshape(-1, 3)
            lows.append(points.min(axis=0))
            highs.append(points.max(axis=0))
            zero, negative = points == 0, np.signbit(points)
            signed[0] |= (zero & negative).any(axis=0)
            signed[1] |= (zero & ~negative).any(axis=0)
        low, high = np.min(lows, axis=0), np.max(highs, axis=0)
        if (signed.all(axis=0) & ((low == 0) | (high == 0))).any():
            points = self.triangles(faces).reshape(-1, 3)
            return points.min(axis=0).tolist(), points.max(axis=0).tolist()
        return low.tolist(), high.tolist()

    def components(self, faces: np.ndarray) -> np.ndarray:
        """Connected-component labels of ``faces`` (ascending indices) over shared edges.

        Two faces are linked as ``Trimesh.face_adjacency`` links them: the merged
        collider uses their common edge exactly twice, once by each. Labels number
        the components in order of their first face.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        count = len(faces)
        if not count:
            return np.zeros(0, dtype=np.int32)
        size = np.int64(len(self.vertices))
        keys = _edge_keys(self.corners(faces), size)
        order = np.argsort(keys)
        ordered = keys[order]
        del keys
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        pairs = starts[np.diff(np.r_[starts, len(ordered)]) == 2]
        first, second, shared = order[pairs] // 3, order[pairs + 1] // 3, ordered[pairs]
        del order, ordered, starts
        distinct = first != second
        first, second, shared = first[distinct], second[distinct], shared[distinct]
        alone = ~self._used_elsewhere(shared, faces, size)
        graph = coo_matrix((np.ones(int(alone.sum())), (first[alone], second[alone])), shape=(count, count))
        return connected_components(graph, directed=False)[1]

    def _used_elsewhere(self, keys: np.ndarray, faces: np.ndarray, size: np.int64) -> np.ndarray:
        """Whether a face of the collider outside ``faces`` also uses each edge in ``keys``."""
        used = np.zeros(len(keys), dtype=bool)
        if not len(keys):
            return used
        order = np.argsort(keys)
        ordered = keys[order]
        marked = np.zeros(len(self.vertices), dtype=bool)
        marked[ordered // size] = True
        marked[ordered % size] = True
        member = np.zeros(len(self.faces), dtype=bool)
        member[faces] = True
        for block in _row_blocks(len(self.faces)):
            corners = self.representative[self.faces[block]]
            touching = (marked[corners].sum(axis=1) >= 2) & ~member[block]
            if not touching.any():
                continue
            edges = _edge_keys(corners[touching], size)
            position = np.minimum(np.searchsorted(ordered, edges), len(ordered) - 1)
            found = ordered[position] == edges
            used[order[position[found]]] = True
        return used


def _edge_keys(corners: np.ndarray, size: np.int64) -> np.ndarray:
    """Each face's three edges (0-1, 1-2, 2-0), as ``low * size + high`` of their merged vertices."""
    edges = corners[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    return np.minimum(edges[:, 0], edges[:, 1]) * size + np.maximum(edges[:, 0], edges[:, 1])


def search_window_m(placement_residual_m: float | None) -> float:
    """How far from a footprint's estimated bottom its surface is searched: 3 cm, or the residual when larger."""
    window_m = 0.03
    if (not isinstance(placement_residual_m, bool) and isinstance(placement_residual_m, (int, float))
            and math.isfinite(placement_residual_m)):
        window_m = max(window_m, float(placement_residual_m))
    return window_m


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

    The collider is read as its merged copy would be (``MergedCollider``), never
    copied: ``mesh`` is not changed.
    """
    window_m = search_window_m(placement_residual_m)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(upper <= lower) or up not in {1, 2} or up_sign not in {-1, 1}
            or not np.isfinite(meters_per_unit) or meters_per_unit <= 0):
        raise ValueError("website_support_query_invalid")
    if up_sign == -1:
        # Probe in an upright frame (the reflection of ``up``); keep output in the source frame.
        low, high = lower.copy(), upper.copy()
        low[up], high[up] = -upper[up], -lower[up]
        result = _support(MergedCollider(mesh, up=up, flipped=True), low, high, up=up,
                          meters_per_unit=meters_per_unit, window_m=window_m)
        if result is not None:
            result["top_runtime_units"] *= -1
            result["aabb_min"][up], result["aabb_max"][up] = -result["aabb_max"][up], -result["aabb_min"][up]
        return result
    return _support(MergedCollider(mesh, up=up), lower, upper, up=up, meters_per_unit=meters_per_unit,
                    window_m=window_m)


def _support(collider: MergedCollider, lower: np.ndarray, upper: np.ndarray, *, up: int,
             meters_per_unit: float, window_m: float) -> dict[str, Any] | None:
    result = (_level_face_support(collider, lower, upper, up=up, meters_per_unit=meters_per_unit, window_m=window_m)
              or _fitted_support(collider, lower, upper, up=up, meters_per_unit=meters_per_unit, window_m=window_m))
    if result is not None:
        result.update(search_window_m=window_m, search_window_basis=(
            "registration_task_region_residual" if window_m > 0.03 else "default"))
    return result


def _level_face_support(collider: MergedCollider, lower: np.ndarray, upper: np.ndarray, *, up: int,
                        meters_per_unit: float, window_m: float = 0.03) -> dict[str, Any] | None:
    """Exact level faces (within 5 degrees) under every probe, coplanar and connected."""
    horizontal = [axis for axis in range(3) if axis != up]
    eligible = collider.level
    if not len(eligible):
        return None
    height_tolerance = window_m / meters_per_unit
    flat_tolerance = 0.005 / meters_per_unit
    probes = np.array([[x, y] for x in np.linspace(lower[horizontal[0]], upper[horizontal[0]], 3)
                       for y in np.linspace(lower[horizontal[1]], upper[horizontal[1]], 3)])
    # The highest face under each probe, first by index among equals: a block of faces at a time.
    hits: list[tuple[int, float] | None] = [None] * len(probes)
    for block in _row_blocks(len(eligible)):
        faces = eligible[block]
        triangles = collider.triangles(faces)
        projected = triangles[:, :, horizontal]
        a, b, c = projected[:, 0], projected[:, 1], projected[:, 2]
        e1, e2 = b - a, c - a
        determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        safe = np.abs(determinant) > 1e-12
        for index, point in enumerate(probes):
            offset = point - a
            u = np.divide(offset[:, 0] * e2[:, 1] - offset[:, 1] * e2[:, 0], determinant,
                          out=np.zeros_like(determinant), where=safe)
            v = np.divide(e1[:, 0] * offset[:, 1] - e1[:, 1] * offset[:, 0], determinant,
                          out=np.zeros_like(determinant), where=safe)
            heights = (triangles[:, 0, up] + u * (triangles[:, 1, up] - triangles[:, 0, up])
                       + v * (triangles[:, 2, up] - triangles[:, 0, up]))
            valid = (safe & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
                     & (np.abs(heights - lower[up]) <= height_tolerance))
            candidates = np.flatnonzero(valid)
            if len(candidates):
                chosen = candidates[np.argmax(heights[candidates])]
                if hits[index] is None or heights[chosen] > hits[index][1]:
                    hits[index] = (int(faces[chosen]), float(heights[chosen]))
    if any(hit is None for hit in hits):
        return None
    top = max(height for _, height in hits)
    if top - min(height for _, height in hits) > flat_tolerance:
        return None
    coplanar = _joined([faces[np.abs(collider.triangles(faces)[:, :, up] - top).max(axis=1) <= flat_tolerance]
                        for faces in (eligible[block] for block in _row_blocks(len(eligible)))])
    # Every probe's face is coplanar and connected to the first probe's, over coplanar faces only.
    hit_faces = np.array([face for face, _ in hits])
    position = np.searchsorted(coplanar, hit_faces)
    if np.any(position >= len(coplanar)) or np.any(coplanar[np.minimum(position, len(coplanar) - 1)] != hit_faces):
        return None
    labels = collider.components(coplanar)
    if np.any(labels[position] != labels[position[0]]):
        return None
    connected = coplanar[labels == labels[position[0]]]
    low, high = collider.bounds(connected)
    return {"top_runtime_units": top, "aabb_min": low, "aabb_max": high, "face_indices": connected.tolist(),
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


def _fitted_support(collider: MergedCollider, lower: np.ndarray, upper: np.ndarray, *, up: int,
                    meters_per_unit: float, window_m: float = 0.03) -> dict[str, Any] | None:
    """Fit the bumpy surface under the footprint, grow it over connected faces, and test it.

    Faces steeper than ``FITTED_MAX_FACE_TILT_DEGREES`` never join, so the fit works on the
    collider's gentle faces (``collider.gentle``, positions into it until the surface is grown).
    """
    horizontal = [axis for axis in range(3) if axis != up]
    gentle, centres = collider.gentle, collider.centres
    within = np.ones(len(centres), dtype=bool)
    for axis in horizontal:
        within &= (centres[:, axis] >= lower[axis]) & (centres[:, axis] <= upper[axis])
    tolerance = window_m / meters_per_unit
    seed = np.flatnonzero(within & (np.abs(centres[:, up] - lower[up]) <= tolerance))
    if window_m > 0.03 and len(seed):
        # A window wider than 3 cm can also reach a second surface under the
        # footprint, such as a counter's edge below its top (capture-447ae4f4:
        # 0.62-0.69 under a 0.72 top). Seed on the 3 cm band of the highest
        # surface it reached: the object rests on the highest of them.
        top = np.percentile(centres[seed, up], 90)
        seed = seed[np.abs(centres[seed, up] - top) <= 0.03 / meters_per_unit]
    if len(seed) < 3:
        return None
    low_band, high_band = (bound / meters_per_unit for bound in FITTED_BAND_BOUNDS_M)
    faces = seed
    for _ in range(FITTED_ITERATIONS):
        normal, middle, rms = _plane(centres[faces], up)
        band = min(max(2.5 * rms, low_band), high_band)
        near = np.zeros(len(centres), dtype=bool)
        for block in _row_blocks(len(centres)):
            near[block] = np.abs((centres[block] - middle) @ normal) <= band
        candidates = np.flatnonzero(near)
        labels = collider.components(gentle[candidates])
        seeded = labels[np.searchsorted(candidates, seed[near[seed]])]
        if not len(seeded):
            return None
        grown = candidates[labels == np.bincount(seeded).argmax()]
        if np.array_equal(grown, faces):
            break
        faces = grown
    normal, middle, rms = _plane(centres[faces], up)
    tilt = float(np.degrees(np.arccos(min(1.0, float(normal[up])))))
    if tilt > FITTED_MAX_TILT_DEGREES or rms * meters_per_unit > FITTED_MAX_RMS_M:
        return None
    faces = gentle[faces]
    # A real triangle of this surface lies under every probe (no gap, no edge),
    # and the triangles around it stand near the plane there; the object rests on the highest.
    widened = window_m > 0.03
    reach = window_m / meters_per_unit
    pad = (max(FITTED_CONTACT_RADIUS_M, window_m) if widened else FITTED_CONTACT_RADIUS_M) / meters_per_unit
    local = []
    for block in _row_blocks(len(faces)):
        extent = collider.triangles(faces[block])[:, :, horizontal]
        local.append(faces[block][np.all((extent.max(axis=1) >= lower[horizontal] - pad)
                                         & (extent.min(axis=1) <= upper[horizontal] + pad), axis=1)])
    local = _joined(local)
    triangles = collider.triangles(local)
    a, b, c = (triangles[:, corner][:, horizontal] for corner in range(3))
    za, zb, zc = (triangles[:, corner, up] for corner in range(3))
    e1, e2 = b - a, c - a
    determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    safe = np.abs(determinant) > 1e-12
    spread = np.linspace(-1.0, 1.0, 5) * FITTED_CONTACT_RADIUS_M / meters_per_unit
    corners = triangles.reshape(-1, 3)

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
    low, high = collider.bounds(faces)
    return {"top_runtime_units": top, "aabb_min": low,
            "aabb_max": high, "face_indices": faces.tolist(),
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
#: How far past its surface's outline a placed region may reach, from rounding alone. The
#: admission allows twice this, so converting a placed region to the simulator frame cannot
#: refuse it.
REGION_EDGE_ROUNDING_M = DESTINATION_EDGE_GUARD_M / 2
#: Plan positions tried along each axis when a destination must move off the start.
DESTINATION_GRID_STEPS = 9
#: Plan positions tried along each axis for a region that must lie wholly on its surface. A
#: curved outline's farthest or nearest such spot falls between the coarse grid's points
#: (capture-447ae4f4: one of its 81 held the region on the oval top). 33 = 4 x 8 + 1, so every
#: coarse point is also one of these, bit for bit.
REGION_GRID_STEPS = 33
#: Footprints checked against the collider, nearest or farthest first, before none is placed;
#: once per rule. Only candidates that already pass the rule's plan test against the surface's
#: own faces count against it: a round table's bounding-box corners would otherwise use it up.
DESTINATION_MAX_PROBES = 24


class Destination(NamedTuple):
    """A moved destination in runtime units, the support under its footprint there, and
    whether its whole target region lies on that surface (``SupportOutline.holds``)."""

    centre: list[float]
    support: dict[str, Any]
    region_on_surface: bool


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


class SupportOutline:
    """A support surface in plan, in one frame and unit: its own triangles, the edges
    that bound it, and the gap its support search bridges (``window``)."""

    def __init__(self, triangles: np.ndarray, edges: np.ndarray, window: float, *, up: int):
        from scipy.spatial import cKDTree

        self.up, self.window = up, float(window)
        self.triangles_3d, self.edges_3d = np.asarray(triangles, dtype=float), np.asarray(edges, dtype=float)
        plan = [axis for axis in range(3) if axis != up]
        self.triangles, self.edges = self.triangles_3d[:, :, plan], self.edges_3d[:, :, plan]
        centres = self.triangles.mean(axis=1)
        self._reach = float(np.linalg.norm(self.triangles - centres[:, None], axis=2).max())
        self._nearby = cKDTree(centres)

    def in_frame(self, matrix: Sequence[Sequence[float]], *, up: int = 2) -> SupportOutline:
        """The same outline in another frame, ``matrix`` (4x4) a similarity such as ``runtime_to_simulator``."""
        matrix = np.asarray(matrix, dtype=float)
        scale = abs(float(np.linalg.det(matrix[:3, :3]))) ** (1 / 3)
        return SupportOutline(self.triangles_3d @ matrix[:3, :3].T + matrix[:3, 3],
                              self.edges_3d @ matrix[:3, :3].T + matrix[:3, 3], self.window * scale, up=up)

    def covers(self, points: np.ndarray) -> np.ndarray:
        """Whether each of the plan ``points`` (shape (n, 2)) lies over one of the surface's faces."""
        from itertools import chain

        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not len(points):
            return np.zeros(0, dtype=bool)
        rows = self._nearby.query_ball_point(points, self._reach)
        owner = np.repeat(np.arange(len(points)), [len(row) for row in rows])
        triangles = self.triangles[np.fromiter(chain.from_iterable(rows), dtype=np.int64, count=len(owner))]
        p, crosses = points[owner], []
        for start, end in ((0, 1), (1, 2), (2, 0)):
            edge, offset = triangles[:, end] - triangles[:, start], p - triangles[:, start]
            crosses.append(edge[:, 0] * offset[:, 1] - edge[:, 1] * offset[:, 0])
        a, b, c = crosses
        inside = ((a >= 0) & (b >= 0) & (c >= 0)) | ((a <= 0) & (b <= 0) & (c <= 0))
        covered = np.zeros(len(points), dtype=bool)
        covered[owner[inside]] = True
        return covered

    def distance(self, point: Sequence[float], limit: float) -> float:
        """A plan point's distance to the surface's faces, when within ``limit``; zero over one, else inf."""
        rows = self._nearby.query_ball_point(point, self._reach + limit)
        return _plan_distance_to_triangles(point, self.triangles[rows]) if rows else math.inf

    def holds(self, centre: Sequence[float], radius: float, *, tolerance: float) -> bool:
        """Whether a disc of ``radius`` at the plan point ``centre`` lies wholly on the surface.

        Its centre lies on the surface's faces, or in a gap of them within
        the window, and none of the surface's edges comes nearer than its
        radius, less ``tolerance`` (rounding only).
        """
        if self.distance(centre, self.window) > min(self.window, radius):
            return False
        if not len(self.edges):
            return True
        a, b = self.edges[:, 0], self.edges[:, 1]
        along = b - a
        p = np.asarray(centre, dtype=float)
        t = np.clip(((p - a) * along).sum(axis=1) / np.maximum((along * along).sum(axis=1), 1e-300), 0.0, 1.0)
        nearest = a + t[:, None] * along - p
        return bool(np.sqrt((nearest * nearest).sum(axis=1)).min() >= radius - tolerance)


#: Points along an edge's outward normal, as shares of twice the window, tested for the surface resuming.
GAP_PROBE_SHARES = (0.25, 0.5, 0.75, 1.0)
#: Where along each piece of an edge (no longer than the window) the surface must resume across it:
#: a tapered gap narrows from one end of a piece to the other.
GAP_PROBE_STATIONS = (0.0, 0.5, 1.0)


def support_outline(mesh: Any, support: Mapping[str, Any], *, up: int, meters_per_unit: float,
                    placement_residual_m: float | None = None) -> SupportOutline | None:
    """The surface a support's own faces cover, in the mesh's frame and units.

    A support's bounding box is not its surface: an oval top leaves the box's
    corners empty (capture-447ae4f4: a tenth of the target region lay past the
    dining table's edge). The surface's faces (``face_indices``) bound it at
    every edge exactly one of them uses. Across parts of those edges the
    surface resumes within twice the support search's window (a face the fit
    dropped; where the removed object sat): those bound a gap every point of
    which lies within the window of the surface, which the region may cross.
    That is established piece by piece, each no longer than the window, at
    both its ends and its middle: one probe cannot speak for a whole edge
    of a tapered gap (Codex review of #3170). Every other piece is the
    surface's edge. None when the support has no faces (a registered floor
    plane): its box is then all there is.
    """
    import trimesh

    faces = np.asarray(support.get("face_indices") or [], dtype=np.int64)
    if not len(faces):
        return None
    plan = [axis for axis in range(3) if axis != up]
    corners = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces)[faces]].reshape(-1, 3)
    # Merged as support_under merged them, so faces share the edges its adjacency used.
    surface = trimesh.Trimesh(vertices=corners, faces=np.arange(len(corners)).reshape(-1, 3), process=False)
    surface.merge_vertices()
    vertices, triangles = np.asarray(surface.vertices, dtype=float), np.asarray(surface.faces)
    triangles = triangles[(triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                          & (triangles[:, 2] != triangles[:, 0])]
    # Each face's three edges, with the corner opposite each: it lies on the surface's side.
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    opposite = triangles[:, [2, 0, 1]].reshape(-1)
    _, first, uses = np.unique(edges, axis=0, return_index=True, return_counts=True)
    single = np.sort(first[uses == 1])
    window = search_window_m(placement_residual_m) / meters_per_unit
    points = vertices[:, plan]
    a, b = points[edges[single, 0]], points[edges[single, 1]]
    normal = np.c_[b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]]
    length = np.hypot(normal[:, 0], normal[:, 1])
    single, a, b, normal, length = (value[length > 0] for value in (single, a, b, normal, length))
    normal /= length[:, None]
    # Outward: away from the corner opposite the edge in its one face.
    normal[((points[opposite[single]] - a) * normal).sum(axis=1) > 0] *= -1
    # Each edge in pieces no longer than the window; every piece is probed across at its stations.
    count = np.maximum(1, np.ceil(length / window)).astype(np.int64)
    owner = np.repeat(np.arange(len(single)), count)
    index = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    start, stop = index / count[owner], (index + 1) / count[owner]
    stations = np.asarray(GAP_PROBE_STATIONS)
    along = start[:, None] + (stop - start)[:, None] * stations
    at = a[owner][:, None, :] * (1 - along)[:, :, None] + b[owner][:, None, :] * along[:, :, None]
    probes = (at[:, :, None, :] + normal[owner][:, None, None, :]
              * (2 * window * np.asarray(GAP_PROBE_SHARES))[None, None, :, None])
    whole = SupportOutline(vertices[triangles], np.empty((0, 2, 3)), window, up=up)
    resumes = (whole.covers(probes.reshape(-1, 2)).reshape(len(owner), len(stations), len(GAP_PROBE_SHARES))
               .any(axis=2).all(axis=1))
    kept = ~resumes
    first_end, last_end = vertices[edges[single[owner[kept]], 0]], vertices[edges[single[owner[kept]], 1]]
    pieces = np.stack([first_end * (1 - start[kept])[:, None] + last_end * start[kept][:, None],
                       first_end * (1 - stop[kept])[:, None] + last_end * stop[kept][:, None]], axis=1)
    # An edge three or more faces use folds the surface: no side of it is known to be surface.
    folds = vertices[edges[np.sort(first[(uses % 2 == 1) & (uses > 1)])]].reshape(-1, 2, 3)
    return SupportOutline(vertices[triangles], np.concatenate([pieces, folds]), window, up=up)


def _footprint_over_surface(outline: SupportOutline | None, half: np.ndarray, *, plan: list[int]) -> Any:
    """Whether the nine probe points support_under tests lie over, or within its window of, the surface's faces.

    In plan, so any tessellation; the exact test is support_under itself.
    """
    if outline is None:
        return lambda centre: True
    signs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    return lambda centre: all(outline.distance([centre[plan[0]] + a * half[plan[0]], centre[plan[1]] + b * half[plan[1]]],
                                               outline.window) <= outline.window for a, b in signs)


def _first_supported(mesh: Any, lower: Sequence[float], upper: Sequence[float], region_min: Sequence[float],
                     region_max: Sequence[float], support: Mapping[str, Any], *, up: int, meters_per_unit: float,
                     up_sign: int, placement_residual_m: float | None, clearance: float, steps: int,
                     admissible: Any, farthest: bool) -> tuple[list[float], dict[str, Any]] | None:
    """The footprint ``lower``..``upper`` moved over a plan grid, farthest from or nearest to where it is.

    Each candidate keeps the footprint inside ``region`` and the target disc
    inside the support's bounds, passes ``admissible`` (a plan test), and
    must rest on the same surface by ``support_under`` against real
    triangles. Returns ``(centre, support)`` in runtime units, or None.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    plan = [axis for axis in range(3) if axis != up]
    half, origin = (upper - lower) / 2, (lower + upper) / 2
    grid = []
    for axis in plan:
        low = max(float(region_min[axis]) + half[axis], float(support["aabb_min"][axis]) + clearance)
        high = min(float(region_max[axis]) - half[axis], float(support["aabb_max"][axis]) - clearance)
        if low > high:
            return None
        grid.append(np.linspace(low, high, steps))
    candidates = []
    for first, second in ((a, b) for a in grid[0] for b in grid[1]):
        centre = origin.copy()
        centre[plan[0]], centre[plan[1]] = first, second
        if admissible(centre):
            candidates.append(centre)
    sign = -1 if farthest else 1
    candidates.sort(key=lambda centre: sign * math.dist(centre[plan], origin[plan]))
    for centre in candidates[:DESTINATION_MAX_PROBES]:
        offset = centre - origin
        found = support_under(mesh, lower + offset, upper + offset, up=up, meters_per_unit=meters_per_unit,
                              up_sign=up_sign, placement_residual_m=placement_residual_m)
        if found is not None and same_support_surface(found, support):
            centre = centre.copy()
            centre[up] = found["top_runtime_units"]
            # Only a centre within the guard of a bound moves; every other one is returned exactly.
            guard = DESTINATION_EDGE_GUARD_M / meters_per_unit
            for axis in plan:
                centre[axis] = min(max(centre[axis], float(support["aabb_min"][axis]) + clearance + guard),
                                   float(support["aabb_max"][axis]) - clearance - guard)
            return centre.tolist(), found
    return None


def farthest_supported_destination(mesh: Any, start_min: Sequence[float], start_max: Sequence[float],
                                   region_min: Sequence[float], region_max: Sequence[float],
                                   start_support: dict[str, Any], *, up: int, meters_per_unit: float,
                                   up_sign: int = 1, placement_residual_m: float | None = None,
                                   clearance: float, outline: SupportOutline | None = None) -> Destination | None:
    """The start footprint moved as far from the start as its own surface and ``region`` allow.

    Only for a destination grounded as the very surface the subject rests on
    (capture-447ae4f4: the whole dining table for "the other end of the same
    table", whose centre lay under the tray). Each candidate keeps the target
    disc of radius ``clearance`` clear of the start and the footprint inside
    ``region``, and must rest on the same surface by ``support_under``.

    The scorer accepts the subject wherever its whole footprint lies in the
    disc, and those footprints together cover the disc: every point of it is
    somewhere an accepted subject rests. So the farthest spot whose whole disc
    lies on the surface (``support_outline``, or the given ``outline`` of
    ``start_support``) is taken. Where the disc fits nowhere on it, the rule
    before this one places it (the footprint alone over the surface's faces),
    with ``region_on_surface`` False for the admission to flag. None when
    neither places one.
    """
    start_min, start_max = np.asarray(start_min, dtype=float), np.asarray(start_max, dtype=float)
    plan = [axis for axis in range(3) if axis != up]
    common = {"up": up, "meters_per_unit": meters_per_unit, "up_sign": up_sign,
              "placement_residual_m": placement_residual_m, "clearance": clearance, "farthest": True}

    def clear(centre: np.ndarray) -> bool:
        return destination_clear_of_start(start_min, start_max, centre, up=up, clearance=clearance)

    if outline is None:
        outline = support_outline(mesh, start_support, up=up, meters_per_unit=meters_per_unit,
                                  placement_residual_m=placement_residual_m)
    if outline is not None:
        rounding = REGION_EDGE_ROUNDING_M / meters_per_unit
        found = _first_supported(mesh, start_min, start_max, region_min, region_max, start_support,
                                 steps=REGION_GRID_STEPS, **common, admissible=lambda centre: clear(centre)
                                 and outline.holds(centre[plan], clearance, tolerance=rounding))
        if found is not None:
            return Destination(*found, True)
    over = _footprint_over_surface(outline, (start_max - start_min) / 2, plan=plan)
    found = _first_supported(mesh, start_min, start_max, region_min, region_max, start_support,
                             steps=DESTINATION_GRID_STEPS, **common,
                             admissible=lambda centre: clear(centre) and over(centre))
    return None if found is None else Destination(*found, outline is None)


def nearest_supported_destination(mesh: Any, lower: Sequence[float], upper: Sequence[float],
                                  region_min: Sequence[float], region_max: Sequence[float],
                                  support: dict[str, Any], *, up: int, meters_per_unit: float, up_sign: int = 1,
                                  placement_residual_m: float | None = None, clearance: float,
                                  outline: SupportOutline, start_min: Sequence[float] | None = None,
                                  start_max: Sequence[float] | None = None) -> Destination | None:
    """The grounded footprint ``lower``..``upper`` moved the least that puts its whole target disc on ``support``.

    For a destination whose disc of radius ``clearance`` reaches past its
    surface (``outline``, of ``support``) where it was grounded: within
    ``region`` and, with the start on the same surface, clear of it. None
    when the disc fits nowhere there: the grounded spot then stands, and the
    admission flags it.
    """
    plan = [axis for axis in range(3) if axis != up]
    rounding = REGION_EDGE_ROUNDING_M / meters_per_unit

    def admissible(centre: np.ndarray) -> bool:
        return ((start_min is None or destination_clear_of_start(start_min, start_max, centre, up=up,
                                                                 clearance=clearance))
                and outline.holds(centre[plan], clearance, tolerance=rounding))

    found = _first_supported(mesh, lower, upper, region_min, region_max, support, up=up,
                             meters_per_unit=meters_per_unit, up_sign=up_sign,
                             placement_residual_m=placement_residual_m, clearance=clearance,
                             steps=REGION_GRID_STEPS, admissible=admissible, farthest=False)
    return None if found is None else Destination(*found, True)


def footprint_at(start_min: Sequence[float], start_max: Sequence[float], centre: Sequence[float], *,
                 up: int, up_sign: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The start's footprint moved to ``centre``, its bottom (top when ``up_sign`` is -1) at ``centre[up]``."""
    lower, upper = np.asarray(start_min, dtype=float).copy(), np.asarray(start_max, dtype=float).copy()
    for axis in range(3):
        if axis != up:
            half_width = (start_max[axis] - start_min[axis]) / 2
            lower[axis], upper[axis] = centre[axis] - half_width, centre[axis] + half_width
    height = start_max[up] - start_min[up]
    lower[up] = centre[up] - (height if up_sign == -1 else 0)
    upper[up] = centre[up] + (height if up_sign == 1 else 0)
    return lower, upper


def supported_destination_near_place(mesh: Any, start_min: Sequence[float], start_max: Sequence[float],
                                     centre: Sequence[float], region_min: Sequence[float],
                                     region_max: Sequence[float], surface_min: Sequence[float],
                                     surface_max: Sequence[float], *, up: int, meters_per_unit: float,
                                     up_sign: int = 1, placement_residual_m: float | None = None,
                                     clearance: float) -> tuple[list[float], dict[str, Any]] | None:
    """The supported footprint nearest a named place whose own centre has no contact.

    A place at a surface's edge or corner puts the footprint's centre too near
    the boundary. Candidates on the coarse grid (``DESTINATION_GRID_STEPS``)
    overlap the place, keep the footprint inside the surface's bounds and clear
    of the start, and are probed nearest ``centre`` first, at most
    ``DESTINATION_MAX_PROBES`` times, by ``support_under`` against real triangles.
    Returns ``(centre, support)`` in runtime units, or None when none is supported.
    """
    plan = [axis for axis in range(3) if axis != up]
    half = (np.asarray(start_max, dtype=float) - np.asarray(start_min, dtype=float)) / 2
    target = np.asarray(centre, dtype=float)
    steps = []
    for axis in plan:
        low = max(float(region_min[axis]) - half[axis], float(surface_min[axis]) + half[axis])
        high = min(float(region_max[axis]) + half[axis], float(surface_max[axis]) - half[axis])
        if low > high:
            return None
        steps.append(np.linspace(low, high, DESTINATION_GRID_STEPS))
    candidates = []
    for first, second in ((a, b) for a in steps[0] for b in steps[1]):
        candidate = target.copy()
        candidate[plan[0]], candidate[plan[1]] = first, second
        if destination_clear_of_start(start_min, start_max, candidate, up=up, clearance=clearance):
            candidates.append(candidate)
    candidates.sort(key=lambda candidate: math.dist(candidate[plan], target[plan]))
    for candidate in candidates[:DESTINATION_MAX_PROBES]:
        found = support_under(mesh, *footprint_at(start_min, start_max, candidate, up=up, up_sign=up_sign), up=up,
                              meters_per_unit=meters_per_unit, up_sign=up_sign,
                              placement_residual_m=placement_residual_m)
        if found is not None:
            candidate[up] = found["top_runtime_units"]
            return candidate.tolist(), found
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
        # Ground in an upright frame (the reflection of ``up``); keep output in the source frame.
        low, high = lower.copy(), upper.copy()
        low[up], high[up] = -upper[up], -lower[up]
        result = _ground_upright(mesh, low, high, up=up, meters_per_unit=meters_per_unit,
                                 floor_height=-floor_height, reflected=True)
        if result is not None:
            result["top_runtime_units"] *= -1
            result["aabb_min"][up], result["aabb_max"][up] = -result["aabb_max"][up], -result["aabb_min"][up]
        return result
    return _ground_upright(mesh, lower, upper, up=up, meters_per_unit=meters_per_unit, floor_height=floor_height,
                           reflected=False)


def _ground_upright(mesh: Any, lower: np.ndarray, upper: np.ndarray, *, up: int, meters_per_unit: float,
                    floor_height: float, reflected: bool) -> dict[str, Any] | None:
    """``ground_on_observed_floor`` with ``up`` upright, reading the collider a block of faces at a time."""
    from trimesh import triangles as triangle_math

    gap = float(lower[up] - floor_height)
    if not 0.0 < gap * meters_per_unit <= FLOOR_GROUNDING_MAX_GAP_M:
        return None
    horizontal = [axis for axis in range(3) if axis != up]
    tolerance = 0.03 / meters_per_unit
    margin = FLOOR_SEARCH_MARGIN_M / meters_per_unit
    footprint_m2 = float(np.prod([(upper[a] - lower[a]) * meters_per_unit for a in horizontal]))
    # Face normals a collider already holds are its own (a mesh may be given them); a reflected copy holds none.
    cache = getattr(mesh, "_cache", None)
    given = None if reflected or cache is None or "face_normals" not in cache else np.asarray(mesh.face_normals)
    rows: dict[str, list[np.ndarray]] = {"between": [], "faces": [], "areas": [], "heights": [], "triangles": []}
    for block, triangles in triangle_blocks(mesh, reflect=up if reflected else None):
        centers = triangles.mean(axis=1)
        crosses = triangle_math.cross(triangles)
        areas = triangle_math.area(crosses=crosses) * meters_per_unit ** 2
        if given is None:
            unit, valid = triangle_math.normals(triangles=triangles, crosses=crosses)
            normals = np.zeros((len(triangles), 3))
            normals[valid] = unit
        else:
            normals = given[block]
        level = np.abs(normals[:, up]) >= np.cos(np.deg2rad(5))
        near = np.ones(len(triangles), dtype=bool)
        within = np.ones(len(triangles), dtype=bool)
        for axis in horizontal:
            near &= (centers[:, axis] >= lower[axis] - margin) & (centers[:, axis] <= upper[axis] + margin)
            within &= (centers[:, axis] >= lower[axis]) & (centers[:, axis] <= upper[axis])
        between = (level & within & (centers[:, up] > floor_height + tolerance)
                   & (centers[:, up] < lower[up] + tolerance))
        floor = level & near & (np.abs(centers[:, up] - floor_height) <= tolerance)
        rows["between"].append(areas[between])
        rows["faces"].append(np.flatnonzero(floor) + block.start)
        rows["areas"].append(areas[floor])
        rows["heights"].append(centers[floor, up])
        rows["triangles"].append(triangles[floor])
    between_area = float(_joined(rows["between"], np.float64).sum())
    if between_area >= INTERMEDIATE_SURFACE_MIN_FOOTPRINT_SHARE * footprint_m2:
        return None
    floor_areas = _joined(rows["areas"], np.float64)
    footprint_min, footprint_max = lower.copy(), upper.copy()
    if floor_areas.sum() >= FLOOR_MIN_AREA_M2:
        top = float(np.average(_joined(rows["heights"], np.float64), weights=floor_areas))
        points = np.concatenate(rows["triangles"]).reshape(-1, 3)
        support_min, support_max = points.min(axis=0), points.max(axis=0)
        basis, faces = "collider_floor_beside_footprint", _joined(rows["faces"]).tolist()
    else:
        top = float(floor_height)
        footprint_min[up] = footprint_max[up] = top
        support_min, support_max = footprint_min, footprint_max
        basis, faces = "registered_observed_floor_plane", []
    return {"top_runtime_units": top, "aabb_min": support_min.tolist(), "aabb_max": support_max.tolist(),
            "face_indices": faces, "extended_runtime_units": float(lower[up] - top),
            "collider_floor_area_m2": float(floor_areas.sum()), "basis": basis,
            "intermediate_surface_area_m2": between_area, "physical_measurement": False}
