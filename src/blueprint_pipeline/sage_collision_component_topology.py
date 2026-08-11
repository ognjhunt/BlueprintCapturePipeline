"""Connected-component and open-cavity evidence for SAGE collision meshes.

SAGE sometimes groups many disconnected scene objects into one USD mesh prim.
Prim-level bounds therefore cannot establish the collision identity of an
InteriorGS instance.  This module applies the authored USD transforms, splits
overlapping meshes by face connectivity, ranks each component against exact
publisher label bounds, and optionally probes a component with vertical rays.

The result is geometric public-dataset evidence only.  An open collision cavity
does not prove hidden appearance, material, physical equivalence, or native
simulator behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from .decision_evidence_contracts import canonical_digest
from .scene_placement.interiorgs_index import SceneObject, load_interiorgs_labels


SCHEMA_VERSION = "interiorgs_sage_collision_component_topology.v2"
COMPONENT_GEOMETRY_SCHEMA_VERSION = "sage_collision_component_geometry.v2"
MAX_OPENING_GRID_SIZE = 101
MINIMUM_SIDE_WALL_RAY_HIT_FRACTION = 1.0
MINIMUM_SIDE_PROJECTED_COVERAGE_FRACTION = 0.999999


class SageCollisionComponentTopologyError(ValueError):
    """Stable, sorted topology-inspection failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_openusd_stage_from_verified_snapshot(
    path: Path,
    *,
    usd_module: Any,
) -> tuple[Any, int, str]:
    """Open one immutable byte snapshot and return its exact identity."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise SageCollisionComponentTopologyError(["sage_collision_no_follow_unavailable"])
    descriptor: int | None = None
    snapshot_path: Path | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        before = os.fstat(descriptor)
        if before.st_size <= 0:
            raise SageCollisionComponentTopologyError(["sage_collision_usd_missing"])
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            prefix="sage-collision-snapshot-",
            suffix=path.suffix,
            delete=False,
        ) as snapshot:
            snapshot_path = Path(snapshot.name)
            copied = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                digest.update(chunk)
                snapshot.write(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or copied != before.st_size:
            raise SageCollisionComponentTopologyError(
                ["sage_collision_source_changed_while_reading"]
            )
        stage = usd_module.Stage.Open(str(snapshot_path), load=usd_module.Stage.LoadAll)
        if stage is None:
            raise SageCollisionComponentTopologyError(["sage_collision_usd_open_failed"])
        return stage, copied, "sha256:" + digest.hexdigest()
    except OSError as exc:
        raise SageCollisionComponentTopologyError(["sage_collision_usd_missing"]) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if snapshot_path is not None:
            try:
                snapshot_path.unlink()
            except FileNotFoundError:
                pass


def _load_interiorgs_labels_from_verified_snapshot(
    path: Path,
) -> tuple[list[SceneObject], int, str]:
    """Parse InteriorGS labels from the same immutable bytes we identify."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise SageCollisionComponentTopologyError(["interiorgs_labels_no_follow_unavailable"])
    descriptor: int | None = None
    snapshot_path: Path | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        before = os.fstat(descriptor)
        if before.st_size <= 0:
            raise SageCollisionComponentTopologyError(["interiorgs_labels_missing"])
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            prefix="interiorgs-labels-snapshot-",
            suffix=".json",
            delete=False,
        ) as snapshot:
            snapshot_path = Path(snapshot.name)
            copied = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                digest.update(chunk)
                snapshot.write(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or copied != before.st_size:
            raise SageCollisionComponentTopologyError(["interiorgs_labels_changed_while_reading"])
        try:
            rows = load_interiorgs_labels(snapshot_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise SageCollisionComponentTopologyError(["interiorgs_labels_invalid"]) from exc
        return rows, copied, "sha256:" + digest.hexdigest()
    except OSError as exc:
        raise SageCollisionComponentTopologyError(["interiorgs_labels_missing"]) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if snapshot_path is not None:
            try:
                snapshot_path.unlink()
            except FileNotFoundError:
                pass


def _resolve_regular_file(path: str | Path, *, missing_error: str) -> Path:
    """Resolve one ordinary file while preserving final-component symlink evidence."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise SageCollisionComponentTopologyError([missing_error])
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise SageCollisionComponentTopologyError([missing_error])
    return resolved


def _box_metrics(
    target_min: Sequence[float],
    target_max: Sequence[float],
    candidate_min: Sequence[float],
    candidate_max: Sequence[float],
) -> tuple[float, float, float]:
    intersection_extent = [
        max(
            0.0,
            min(float(target_max[index]), float(candidate_max[index]))
            - max(float(target_min[index]), float(candidate_min[index])),
        )
        for index in range(3)
    ]
    intersection = intersection_extent[0] * intersection_extent[1] * intersection_extent[2]
    target_volume = 1.0
    candidate_volume = 1.0
    for index in range(3):
        target_volume *= max(0.0, float(target_max[index]) - float(target_min[index]))
        candidate_volume *= max(0.0, float(candidate_max[index]) - float(candidate_min[index]))
    union = target_volume + candidate_volume - intersection
    return (
        intersection / union if union > 0.0 else 0.0,
        intersection / target_volume if target_volume > 0.0 else 0.0,
        intersection / candidate_volume if candidate_volume > 0.0 else 0.0,
    )


def _boxes_overlap(
    first_min: Sequence[float],
    first_max: Sequence[float],
    second_min: Sequence[float],
    second_max: Sequence[float],
) -> bool:
    return all(
        min(float(first_max[index]), float(second_max[index]))
        > max(float(first_min[index]), float(second_min[index]))
        for index in range(3)
    )


def _face_rows(counts: Sequence[int], indices: Sequence[int]) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    cursor = 0
    for raw_count in counts:
        count = int(raw_count)
        face = tuple(int(value) for value in indices[cursor : cursor + count])
        cursor += count
        if count >= 3:
            rows.append(face)
    if cursor != len(indices):
        raise SageCollisionComponentTopologyError(["sage_collision_face_index_count_mismatch"])
    return rows


def _connected_face_components(
    *, point_count: int, faces: Sequence[tuple[int, ...]]
) -> list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]]:
    parent = list(range(point_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    for face in faces:
        anchor = face[0]
        if anchor < 0 or anchor >= point_count:
            raise SageCollisionComponentTopologyError(["sage_collision_face_vertex_index_invalid"])
        for value in face[1:]:
            if value < 0 or value >= point_count:
                raise SageCollisionComponentTopologyError(
                    ["sage_collision_face_vertex_index_invalid"]
                )
            union(anchor, value)

    grouped_faces: dict[int, list[tuple[int, ...]]] = {}
    for face in faces:
        grouped_faces.setdefault(find(face[0]), []).append(face)
    result = []
    for root, component_faces in grouped_faces.items():
        vertices = tuple(sorted({value for face in component_faces for value in face}))
        result.append((vertices, tuple(component_faces)))
    result.sort(key=lambda row: (row[0][0], len(row[0]), len(row[1])))
    return result


def _triangles(faces: Iterable[Sequence[int]]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        for index in range(1, len(face) - 1):
            triangles.append((int(face[0]), int(face[index]), int(face[index + 1])))
    return triangles


def _vertical_first_hit(
    x: float,
    y: float,
    *,
    vertices: Sequence[Sequence[float]],
    triangles: Sequence[tuple[int, int, int]],
    start_z: float,
) -> float | None:
    hits: list[float] = []
    for first, second, third in triangles:
        a = vertices[first]
        b = vertices[second]
        c = vertices[third]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) <= 1e-14:
            continue
        weight_a = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        weight_b = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        weight_c = 1.0 - weight_a - weight_b
        if min(weight_a, weight_b, weight_c) < -1e-10:
            continue
        z = weight_a * a[2] + weight_b * b[2] + weight_c * c[2]
        if z <= start_z + 1e-10:
            hits.append(float(z))
    return max(hits) if hits else None


def _ray_triangle_first_hit_distance(
    origin: Sequence[float],
    direction: Sequence[float],
    *,
    vertices: Sequence[Sequence[float]],
    triangles: Sequence[tuple[int, int, int]],
    maximum_distance: float,
) -> float | None:
    """Return the nearest two-sided triangle hit for one bounded ray."""

    nearest: float | None = None
    epsilon = 1.0e-12
    for first, second, third in triangles:
        a = vertices[first]
        b = vertices[second]
        c = vertices[third]
        edge_ab = [b[index] - a[index] for index in range(3)]
        edge_ac = [c[index] - a[index] for index in range(3)]
        cross_direction_ac = [
            direction[1] * edge_ac[2] - direction[2] * edge_ac[1],
            direction[2] * edge_ac[0] - direction[0] * edge_ac[2],
            direction[0] * edge_ac[1] - direction[1] * edge_ac[0],
        ]
        determinant = sum(edge_ab[index] * cross_direction_ac[index] for index in range(3))
        if abs(determinant) <= epsilon:
            continue
        inverse = 1.0 / determinant
        offset = [origin[index] - a[index] for index in range(3)]
        u = inverse * sum(offset[index] * cross_direction_ac[index] for index in range(3))
        if u < -epsilon or u > 1.0 + epsilon:
            continue
        cross_offset_ab = [
            offset[1] * edge_ab[2] - offset[2] * edge_ab[1],
            offset[2] * edge_ab[0] - offset[0] * edge_ab[2],
            offset[0] * edge_ab[1] - offset[1] * edge_ab[0],
        ]
        v = inverse * sum(direction[index] * cross_offset_ab[index] for index in range(3))
        if v < -epsilon or u + v > 1.0 + epsilon:
            continue
        distance = inverse * sum(edge_ac[index] * cross_offset_ab[index] for index in range(3))
        if distance <= epsilon or distance > maximum_distance + epsilon:
            continue
        if nearest is None or distance < nearest:
            nearest = float(distance)
    return nearest


def _clip_projected_polygon(
    polygon: Sequence[tuple[float, float]],
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not polygon:
        return []

    def inside(point: tuple[float, float]) -> bool:
        value = point[axis]
        return value >= bound - 1.0e-12 if keep_greater else value <= bound + 1.0e-12

    result: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if abs(delta) > 1.0e-15:
                fraction = (bound - previous[axis]) / delta
                intersection = (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
                result.append(intersection)
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _projected_triangle_rectangle_intersection_area(
    triangle: Sequence[Sequence[float]],
    *,
    rectangle_min: Sequence[float],
    rectangle_max: Sequence[float],
) -> float:
    polygon = [(float(point[0]), float(point[1])) for point in triangle]
    for axis, bound, keep_greater in (
        (0, float(rectangle_min[0]), True),
        (0, float(rectangle_max[0]), False),
        (1, float(rectangle_min[1]), True),
        (1, float(rectangle_max[1]), False),
    ):
        polygon = _clip_projected_polygon(
            polygon,
            axis=axis,
            bound=bound,
            keep_greater=keep_greater,
        )
        if len(polygon) < 3:
            return 0.0
    doubled_area = abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    )
    return doubled_area / 2.0


def _clip_polygon_3d_axis(
    polygon: Sequence[tuple[float, float, float]],
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[tuple[float, float, float]]:
    if not polygon:
        return []

    def inside(point: tuple[float, float, float]) -> bool:
        value = point[axis]
        return value >= bound - 1.0e-12 if keep_greater else value <= bound + 1.0e-12

    result: list[tuple[float, float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if abs(delta) > 1.0e-15:
                fraction = (bound - previous[axis]) / delta
                result.append(
                    tuple(
                        previous[index] + fraction * (current[index] - previous[index])
                        for index in range(3)
                    )
                )
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _segment_intersection_x(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float | None:
    first_direction = (
        first_end[0] - first_start[0],
        first_end[1] - first_start[1],
    )
    second_direction = (
        second_end[0] - second_start[0],
        second_end[1] - second_start[1],
    )
    denominator = (
        first_direction[0] * second_direction[1] - first_direction[1] * second_direction[0]
    )
    if abs(denominator) <= 1.0e-15:
        return None
    offset = (
        second_start[0] - first_start[0],
        second_start[1] - first_start[1],
    )
    first_fraction = (
        offset[0] * second_direction[1] - offset[1] * second_direction[0]
    ) / denominator
    second_fraction = (
        offset[0] * first_direction[1] - offset[1] * first_direction[0]
    ) / denominator
    if not (
        -1.0e-12 <= first_fraction <= 1.0 + 1.0e-12 and -1.0e-12 <= second_fraction <= 1.0 + 1.0e-12
    ):
        return None
    return first_start[0] + first_fraction * first_direction[0]


def _projected_polygon_union_metrics(
    polygons: Sequence[Sequence[tuple[float, float]]],
    *,
    rectangle_min: Sequence[float],
    rectangle_max: Sequence[float],
) -> tuple[float, float]:
    """Exact vertical-decomposition union area for clipped convex polygons."""

    edges = [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for polygon in polygons
        for index in range(len(polygon))
    ]
    x_events = [float(rectangle_min[0]), float(rectangle_max[0])]
    x_events.extend(point[0] for polygon in polygons for point in polygon)
    for first_index, first in enumerate(edges):
        for second in edges[first_index + 1 :]:
            intersection_x = _segment_intersection_x(first[0], first[1], second[0], second[1])
            if (
                intersection_x is not None
                and float(rectangle_min[0]) - 1.0e-12
                <= intersection_x
                <= float(rectangle_max[0]) + 1.0e-12
            ):
                x_events.append(intersection_x)
    ordered: list[float] = []
    for value in sorted(x_events):
        value = min(float(rectangle_max[0]), max(float(rectangle_min[0]), value))
        if not ordered or abs(value - ordered[-1]) > 1.0e-12:
            ordered.append(value)

    area = 0.0
    maximum_gap = 0.0
    for left, right in zip(ordered, ordered[1:], strict=False):
        if right - left <= 1.0e-12:
            continue
        sample_x = (left + right) / 2.0
        intervals: list[tuple[float, float]] = []
        for polygon in polygons:
            intersections: list[float] = []
            for index, start in enumerate(polygon):
                end = polygon[(index + 1) % len(polygon)]
                low_x = min(start[0], end[0])
                high_x = max(start[0], end[0])
                if sample_x < low_x - 1.0e-12 or sample_x > high_x + 1.0e-12:
                    continue
                delta_x = end[0] - start[0]
                if abs(delta_x) <= 1.0e-15:
                    continue
                fraction = (sample_x - start[0]) / delta_x
                if -1.0e-12 <= fraction <= 1.0 + 1.0e-12:
                    intersections.append(start[1] + fraction * (end[1] - start[1]))
            if len(intersections) >= 2:
                intervals.append(
                    (
                        max(float(rectangle_min[1]), min(intersections)),
                        min(float(rectangle_max[1]), max(intersections)),
                    )
                )
        merged: list[list[float]] = []
        for lower, upper in sorted(intervals):
            if upper <= lower + 1.0e-12:
                continue
            if not merged or lower > merged[-1][1] + 1.0e-12:
                merged.append([lower, upper])
            else:
                merged[-1][1] = max(merged[-1][1], upper)
        covered_length = sum(upper - lower for lower, upper in merged)
        area += covered_length * (right - left)
        cursor = float(rectangle_min[1])
        for lower, upper in merged:
            maximum_gap = max(maximum_gap, max(0.0, lower - cursor))
            cursor = max(cursor, upper)
        maximum_gap = max(maximum_gap, max(0.0, float(rectangle_max[1]) - cursor))
    rectangle_area = (float(rectangle_max[0]) - float(rectangle_min[0])) * (
        float(rectangle_max[1]) - float(rectangle_min[1])
    )
    return min(rectangle_area, max(0.0, area)), maximum_gap


def _side_projected_coverage(
    *,
    vertices: Sequence[Sequence[float]],
    triangles: Sequence[tuple[int, int, int]],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    clear_min: Sequence[float],
    clear_max: Sequence[float],
    floor_z: float,
    cavity_depth: float,
) -> dict[str, Any]:
    junction_tolerance = max(1.0e-9, cavity_depth * 1.0e-8)
    z_min = floor_z + junction_tolerance
    z_max = floor_z + cavity_depth - junction_tolerance
    side_specs = {
        "x_min": (0, float(bounds_min[0]), float(clear_min[0]), 1),
        "x_max": (0, float(clear_max[0]), float(bounds_max[0]), 1),
        "y_min": (1, float(bounds_min[1]), float(clear_min[1]), 0),
        "y_max": (1, float(clear_max[1]), float(bounds_max[1]), 0),
    }
    sides: dict[str, Any] = {}
    for side, (normal_axis, band_min, band_max, transverse_axis) in side_specs.items():
        rectangle_min = [float(clear_min[transverse_axis]), z_min]
        rectangle_max = [float(clear_max[transverse_axis]), z_max]
        polygons: list[list[tuple[float, float]]] = []
        for triangle_indices in triangles:
            polygon_3d = [
                tuple(float(value) for value in vertices[index]) for index in triangle_indices
            ]
            polygon_3d = _clip_polygon_3d_axis(
                polygon_3d,
                axis=normal_axis,
                bound=band_min,
                keep_greater=True,
            )
            polygon_3d = _clip_polygon_3d_axis(
                polygon_3d,
                axis=normal_axis,
                bound=band_max,
                keep_greater=False,
            )
            if len(polygon_3d) < 3:
                continue
            polygon_2d = [(point[transverse_axis], point[2]) for point in polygon_3d]
            for axis, bound, keep_greater in (
                (0, rectangle_min[0], True),
                (0, rectangle_max[0], False),
                (1, rectangle_min[1], True),
                (1, rectangle_max[1], False),
            ):
                polygon_2d = _clip_projected_polygon(
                    polygon_2d,
                    axis=axis,
                    bound=bound,
                    keep_greater=keep_greater,
                )
            if len(polygon_2d) >= 3:
                polygons.append(polygon_2d)
        area, maximum_gap = _projected_polygon_union_metrics(
            polygons,
            rectangle_min=rectangle_min,
            rectangle_max=rectangle_max,
        )
        required_area = (rectangle_max[0] - rectangle_min[0]) * (
            rectangle_max[1] - rectangle_min[1]
        )
        coverage_fraction = area / required_area if required_area > 0.0 else 0.0
        sides[side] = {
            "projected_polygon_count": len(polygons),
            "required_projected_area_m2": round(required_area, 12),
            "covered_projected_area_m2": round(area, 12),
            "coverage_fraction": round(coverage_fraction, 9),
            "maximum_uncovered_aperture_m": round(maximum_gap, 9),
            "passed": (
                coverage_fraction >= MINIMUM_SIDE_PROJECTED_COVERAGE_FRACTION
                and maximum_gap <= 1.0e-9
            ),
        }
    return {
        "minimum_coverage_fraction": MINIMUM_SIDE_PROJECTED_COVERAGE_FRACTION,
        "maximum_uncovered_aperture_tolerance_m": 1.0e-9,
        "height_fraction_interval": [0.0, 1.0],
        "floor_and_rim_boundary_tolerance_m": round(junction_tolerance, 12),
        "sides": sides,
        "all_four_sides_passed": all(row["passed"] for row in sides.values()),
        "native_containment_qualified": False,
    }


def _polygon_surface_area_3d(
    polygon: Sequence[tuple[float, float, float]],
) -> float:
    """Return the surface area of one coplanar clipped polygon."""

    if len(polygon) < 3:
        return 0.0
    anchor = polygon[0]
    area = 0.0
    for index in range(1, len(polygon) - 1):
        first = tuple(polygon[index][axis] - anchor[axis] for axis in range(3))
        second = tuple(polygon[index + 1][axis] - anchor[axis] for axis in range(3))
        cross = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        area += 0.5 * sum(value * value for value in cross) ** 0.5
    return area


def _open_prism_obstruction_probe(
    *,
    vertices: Sequence[Sequence[float]],
    triangles: Sequence[tuple[int, int, int]],
    clear_min: Sequence[float],
    clear_max: Sequence[float],
    floor_z: float,
    cavity_depth: float,
) -> dict[str, Any]:
    """Analytically reject any surface inside the admitted open volume.

    A projected-area-only gate misses zero-thickness vertical sheets.  Clip
    every source triangle to the strict interior of the conservative opening
    and retain its true three-dimensional area instead.
    """

    lateral_tolerance = max(
        1.0e-9,
        min(
            float(clear_max[0]) - float(clear_min[0]),
            float(clear_max[1]) - float(clear_min[1]),
        )
        * 1.0e-8,
    )
    vertical_tolerance = max(1.0e-9, cavity_depth * 1.0e-8)
    prism_min = (
        float(clear_min[0]) + lateral_tolerance,
        float(clear_min[1]) + lateral_tolerance,
        floor_z + vertical_tolerance,
    )
    prism_max = (
        float(clear_max[0]) - lateral_tolerance,
        float(clear_max[1]) - lateral_tolerance,
        floor_z + cavity_depth - vertical_tolerance,
    )
    minimum_surface_area = max(
        1.0e-14,
        (float(clear_max[0]) - float(clear_min[0]))
        * (float(clear_max[1]) - float(clear_min[1]))
        * 1.0e-12,
    )
    obstructions: list[dict[str, Any]] = []
    total_surface_area = 0.0
    if all(prism_max[axis] > prism_min[axis] for axis in range(3)):
        for triangle_indices in triangles:
            polygon = [
                tuple(float(value) for value in vertices[index]) for index in triangle_indices
            ]
            for axis in range(3):
                polygon = _clip_polygon_3d_axis(
                    polygon,
                    axis=axis,
                    bound=prism_min[axis],
                    keep_greater=True,
                )
                polygon = _clip_polygon_3d_axis(
                    polygon,
                    axis=axis,
                    bound=prism_max[axis],
                    keep_greater=False,
                )
                if len(polygon) < 3:
                    break
            area = _polygon_surface_area_3d(polygon)
            if area <= minimum_surface_area:
                continue
            total_surface_area += area
            obstructions.append(
                {
                    "triangle_vertex_indices": list(triangle_indices),
                    "clipped_surface_area_m2": round(area, 12),
                }
            )
    return {
        "strict_prism_world_min_m": [round(value, 9) for value in prism_min],
        "strict_prism_world_max_m": [round(value, 9) for value in prism_max],
        "boundary_tolerance_m": {
            "lateral": round(lateral_tolerance, 12),
            "vertical": round(vertical_tolerance, 12),
        },
        "minimum_reported_surface_area_m2": minimum_surface_area,
        "obstruction_triangle_count": len(obstructions),
        "obstruction_surface_area_m2": round(total_surface_area, 12),
        "clear_of_non_floor_geometry": not obstructions,
        "obstructions": obstructions,
    }


def _side_wall_probe(
    *,
    vertices: Sequence[Sequence[float]],
    triangles: Sequence[tuple[int, int, int]],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    clear_min: Sequence[float],
    clear_max: Sequence[float],
    floor_z: float,
    cavity_depth: float,
) -> dict[str, Any]:
    sample_count_per_axis = 5
    height_fractions = (0.2, 0.5, 0.8)
    x_center = (float(clear_min[0]) + float(clear_max[0])) / 2.0
    y_center = (float(clear_min[1]) + float(clear_max[1])) / 2.0
    x_samples = [
        float(clear_min[0])
        + (float(clear_max[0]) - float(clear_min[0])) * (index + 0.5) / sample_count_per_axis
        for index in range(sample_count_per_axis)
    ]
    y_samples = [
        float(clear_min[1])
        + (float(clear_max[1]) - float(clear_min[1])) * (index + 0.5) / sample_count_per_axis
        for index in range(sample_count_per_axis)
    ]
    specifications = {
        "x_min": {
            "direction": (-1.0, 0.0, 0.0),
            "origins": [(x_center, y) for y in y_samples],
            "minimum_distance": x_center - float(clear_min[0]),
            "maximum_distance": x_center - float(bounds_min[0]),
        },
        "x_max": {
            "direction": (1.0, 0.0, 0.0),
            "origins": [(x_center, y) for y in y_samples],
            "minimum_distance": float(clear_max[0]) - x_center,
            "maximum_distance": float(bounds_max[0]) - x_center,
        },
        "y_min": {
            "direction": (0.0, -1.0, 0.0),
            "origins": [(x, y_center) for x in x_samples],
            "minimum_distance": y_center - float(clear_min[1]),
            "maximum_distance": y_center - float(bounds_min[1]),
        },
        "y_max": {
            "direction": (0.0, 1.0, 0.0),
            "origins": [(x, y_center) for x in x_samples],
            "minimum_distance": float(clear_max[1]) - y_center,
            "maximum_distance": float(bounds_max[1]) - y_center,
        },
    }
    sides: dict[str, Any] = {}
    for side, specification in specifications.items():
        hit_count = 0
        rays: list[dict[str, Any]] = []
        for x, y in specification["origins"]:
            for height_fraction in height_fractions:
                origin = (x, y, floor_z + cavity_depth * height_fraction)
                distance = _ray_triangle_first_hit_distance(
                    origin,
                    specification["direction"],
                    vertices=vertices,
                    triangles=triangles,
                    maximum_distance=float(specification["maximum_distance"]) + 1.0e-9,
                )
                qualified_distance = distance is not None and distance + 1.0e-9 >= float(
                    specification["minimum_distance"]
                )
                if qualified_distance:
                    hit_count += 1
                rays.append(
                    {
                        "origin_world_m": [round(value, 9) for value in origin],
                        "height_fraction": height_fraction,
                        "hit_distance_m": (None if distance is None else round(distance, 9)),
                        "minimum_qualified_distance_m": round(
                            float(specification["minimum_distance"]), 9
                        ),
                        "qualified_wall_hit": qualified_distance,
                    }
                )
        hit_fraction = hit_count / len(rays)
        sides[side] = {
            "ray_count": len(rays),
            "hit_count": hit_count,
            "hit_fraction": round(hit_fraction, 9),
            "passed": hit_fraction >= MINIMUM_SIDE_WALL_RAY_HIT_FRACTION,
            "rays": rays,
        }
    return {
        "minimum_hit_fraction": MINIMUM_SIDE_WALL_RAY_HIT_FRACTION,
        "sides": sides,
        "all_four_sides_passed": all(row["passed"] for row in sides.values()),
        "native_containment_qualified": False,
    }


def _opening_probe(
    *,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    grid_size: int,
    margin_fraction: float,
    floor_band_fraction: float,
    cap_band_fraction: float,
) -> dict[str, Any]:
    if grid_size < 3 or grid_size > MAX_OPENING_GRID_SIZE or grid_size % 2 == 0:
        raise SageCollisionComponentTopologyError(
            ["opening_probe_grid_size_must_be_odd_between_three_and_maximum"]
        )
    if not 0.0 <= margin_fraction < 0.5:
        raise SageCollisionComponentTopologyError(["opening_probe_margin_fraction_invalid"])
    extent = [float(bounds_max[i]) - float(bounds_min[i]) for i in range(3)]
    if min(extent) <= 0.0:
        raise SageCollisionComponentTopologyError(["opening_probe_bounds_degenerate"])
    x_min = float(bounds_min[0]) + extent[0] * margin_fraction
    x_max = float(bounds_max[0]) - extent[0] * margin_fraction
    y_min = float(bounds_min[1]) + extent[1] * margin_fraction
    y_max = float(bounds_max[1]) - extent[1] * margin_fraction
    x_values = [x_min + (x_max - x_min) * index / (grid_size - 1) for index in range(grid_size)]
    y_values = [y_min + (y_max - y_min) * index / (grid_size - 1) for index in range(grid_size)]
    triangles = _triangles(faces)
    start_z = float(bounds_max[2]) + max(0.01, extent[2] * 0.1)
    rows: list[dict[str, Any]] = []
    floor_cells: list[tuple[float, float, float]] = []
    cap_count = 0
    missing_count = 0
    for y in y_values:
        for x in x_values:
            hit = _vertical_first_hit(
                x,
                y,
                vertices=vertices,
                triangles=triangles,
                start_z=start_z,
            )
            if hit is None:
                band = "no_hit"
                normalized_height = None
                missing_count += 1
            else:
                normalized_height = (hit - float(bounds_min[2])) / extent[2]
                if normalized_height <= floor_band_fraction:
                    band = "floor"
                    floor_cells.append((x, y, hit))
                elif normalized_height >= cap_band_fraction:
                    band = "cap"
                    cap_count += 1
                else:
                    band = "mid"
            rows.append(
                {
                    "x_m": round(x, 9),
                    "y_m": round(y, 9),
                    "first_hit_z_m": None if hit is None else round(hit, 9),
                    "normalized_height": (
                        None if normalized_height is None else round(normalized_height, 9)
                    ),
                    "band": band,
                }
            )
    center_row = grid_size // 2
    center_column = grid_size // 2
    center = rows[center_row * grid_size + center_column]
    sample_count = len(rows)
    floor_fraction = len(floor_cells) / sample_count
    cap_fraction = cap_count / sample_count
    bands = [
        [rows[row * grid_size + column]["band"] for column in range(grid_size)]
        for row in range(grid_size)
    ]
    # A bounding box around every floor hit is unsafe: disconnected floor
    # patches can make empty or capped cells appear to be a clear opening.  Find
    # the largest axis-aligned, all-floor rectangle that contains the central
    # probe cell.  Every point inside the published opening is therefore backed
    # by a floor hit, while an asymmetric opening retains its measured offset.
    selected_rectangle: tuple[int, int, int, int] | None = None
    selected_cell_count = 0
    if bands[center_row][center_column] == "floor":
        for top in range(center_row + 1):
            for bottom in range(center_row, grid_size):
                valid_columns = [
                    all(bands[row_index][column] == "floor" for row_index in range(top, bottom + 1))
                    for column in range(grid_size)
                ]
                if not valid_columns[center_column]:
                    continue
                left = center_column
                while left > 0 and valid_columns[left - 1]:
                    left -= 1
                right = center_column
                while right + 1 < grid_size and valid_columns[right + 1]:
                    right += 1
                cell_count = (bottom - top + 1) * (right - left + 1)
                candidate = (top, bottom, left, right)
                if cell_count > selected_cell_count or (
                    cell_count == selected_cell_count
                    and selected_rectangle is not None
                    and candidate < selected_rectangle
                ):
                    selected_rectangle = candidate
                    selected_cell_count = cell_count

    clear_bounds = None
    selected_floor_cells: list[tuple[float, float, float]] = []
    if selected_rectangle is not None:
        top, bottom, left, right = selected_rectangle
        for row_index in range(top, bottom + 1):
            for column in range(left, right + 1):
                sample = rows[row_index * grid_size + column]
                hit = sample["first_hit_z_m"]
                if hit is None or sample["band"] != "floor":
                    raise SageCollisionComponentTopologyError(
                        ["opening_probe_internal_rectangle_not_all_floor"]
                    )
                selected_floor_cells.append(
                    (float(sample["x_m"]), float(sample["y_m"]), float(hit))
                )
        # Use the measured all-floor sample centres themselves as the clear
        # footprint.  Expanding by half a lattice cell would assert clearance
        # over unsampled space and can bridge narrow cap slats.
        clear_min_x = min(row[0] for row in selected_floor_cells)
        clear_max_x = max(row[0] for row in selected_floor_cells)
        clear_min_y = min(row[1] for row in selected_floor_cells)
        clear_max_y = max(row[1] for row in selected_floor_cells)
        if clear_max_x > clear_min_x and clear_max_y > clear_min_y:
            clear_bounds = {
                "world_xy_min_m": [round(clear_min_x, 9), round(clear_min_y, 9)],
                "world_xy_max_m": [round(clear_max_x, 9), round(clear_max_y, 9)],
                "size_xy_m": [
                    round(clear_max_x - clear_min_x, 9),
                    round(clear_max_y - clear_min_y, 9),
                ],
                "boundary_clearances_m": {
                    "x_min": round(clear_min_x - float(bounds_min[0]), 9),
                    "x_max": round(float(bounds_max[0]) - clear_max_x, 9),
                    "y_min": round(clear_min_y - float(bounds_min[1]), 9),
                    "y_max": round(float(bounds_max[1]) - clear_max_y, 9),
                },
            }
    median_floor_z = None
    if selected_floor_cells:
        ordered = sorted(row[2] for row in selected_floor_cells)
        median_floor_z = ordered[len(ordered) // 2]
    connected_rectangle_fraction = selected_cell_count / sample_count
    cavity_depth = None if median_floor_z is None else float(bounds_max[2]) - median_floor_z
    overhead_rows: list[dict[str, Any]] = []
    overhead_projected_intersection_area_m2 = 0.0
    wall_probe = None
    side_projected_coverage = None
    open_prism_probe = None
    if clear_bounds is not None and median_floor_z is not None and cavity_depth is not None:
        clear_min = clear_bounds["world_xy_min_m"]
        clear_max = clear_bounds["world_xy_max_m"]
        clear_area = (clear_max[0] - clear_min[0]) * (clear_max[1] - clear_min[1])
        minimum_obstruction_height = median_floor_z + max(
            1.0e-6, cavity_depth * floor_band_fraction
        )
        minimum_projected_area = max(1.0e-12, clear_area * 1.0e-10)
        for triangle_indices in triangles:
            triangle = [vertices[index] for index in triangle_indices]
            if max(float(point[2]) for point in triangle) <= minimum_obstruction_height:
                continue
            area = _projected_triangle_rectangle_intersection_area(
                triangle,
                rectangle_min=clear_min,
                rectangle_max=clear_max,
            )
            if area <= minimum_projected_area:
                continue
            overhead_projected_intersection_area_m2 += area
            overhead_rows.append(
                {
                    "triangle_vertex_indices": list(triangle_indices),
                    "projected_intersection_area_m2": round(area, 12),
                    "minimum_z_m": round(min(float(point[2]) for point in triangle), 9),
                    "maximum_z_m": round(max(float(point[2]) for point in triangle), 9),
                }
            )
        wall_probe = _side_wall_probe(
            vertices=vertices,
            triangles=triangles,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            clear_min=clear_min,
            clear_max=clear_max,
            floor_z=median_floor_z,
            cavity_depth=cavity_depth,
        )
        side_projected_coverage = _side_projected_coverage(
            vertices=vertices,
            triangles=triangles,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            clear_min=clear_min,
            clear_max=clear_max,
            floor_z=median_floor_z,
            cavity_depth=cavity_depth,
        )
        open_prism_probe = _open_prism_obstruction_probe(
            vertices=vertices,
            triangles=triangles,
            clear_min=clear_min,
            clear_max=clear_max,
            floor_z=median_floor_z,
            cavity_depth=cavity_depth,
        )
    overhead_clear = not overhead_rows
    open_cavity = (
        center["band"] == "floor"
        and connected_rectangle_fraction >= 0.2
        and clear_bounds is not None
        and overhead_clear
        and wall_probe is not None
        and wall_probe["all_four_sides_passed"]
        and side_projected_coverage is not None
        and side_projected_coverage["all_four_sides_passed"]
        and open_prism_probe is not None
        and open_prism_probe["clear_of_non_floor_geometry"]
    )
    return {
        "grid_size": grid_size,
        "sample_count": sample_count,
        "margin_fraction": margin_fraction,
        "floor_band_fraction": floor_band_fraction,
        "cap_band_fraction": cap_band_fraction,
        "center_first_hit_band": center["band"],
        "floor_hit_count": len(floor_cells),
        "floor_hit_fraction": round(floor_fraction, 9),
        "center_connected_floor_rectangle_cell_count": selected_cell_count,
        "center_connected_floor_rectangle_fraction": round(connected_rectangle_fraction, 9),
        "cap_hit_count": cap_count,
        "cap_hit_fraction": round(cap_fraction, 9),
        "no_hit_count": missing_count,
        "median_floor_z_m": (None if median_floor_z is None else round(median_floor_z, 9)),
        "cavity_depth_m": (None if cavity_depth is None else round(cavity_depth, 9)),
        "conservative_clear_opening": clear_bounds,
        "overhead_clearance_probe": {
            "projected_intersection_triangle_count": len(overhead_rows),
            "projected_intersection_area_m2": round(overhead_projected_intersection_area_m2, 12),
            "clear_of_above_floor_projected_geometry": overhead_clear,
            "intersections": overhead_rows,
        },
        "side_wall_probe": wall_probe,
        "side_projected_coverage": side_projected_coverage,
        "open_prism_obstruction_probe": open_prism_probe,
        "open_collision_cavity_passed": open_cavity,
        "samples": rows,
        "claim_boundary": {
            "hidden_appearance_observed": False,
            "material_properties_observed": False,
            "native_collision_qualified": False,
            "physical_equivalence_proven": False,
        },
    }


def _target_map(
    labels: Sequence[SceneObject], target_instance_ids: Sequence[str]
) -> dict[str, SceneObject]:
    requested = [str(value) for value in target_instance_ids]
    if not requested or len(set(requested)) != len(requested):
        raise SageCollisionComponentTopologyError(["target_instance_ids_missing_or_duplicate"])
    targets = {row.id: row for row in labels if row.id in set(requested)}
    missing = sorted(set(requested) - set(targets))
    if missing:
        raise SageCollisionComponentTopologyError(
            [f"interiorgs_target_instance_missing:{value}" for value in missing]
        )
    return targets


def read_sage_collision_component_geometry(
    *,
    sage_collision_usd_path: str | Path,
    expected_source_sha256: str,
    expected_source_size_bytes: int,
    prim_path: str,
    component_index: int,
    expected_geometry_digest: str,
) -> dict[str, Any]:
    """Read one exact transformed component and express it around a local pivot.

    ``component_index`` uses the same deterministic connectivity ordering as
    :func:`inspect_sage_collision_component_topology`.  The caller must bind
    both the source-file identity and geometry digest from that prior receipt;
    selecting a prim and index alone is not sufficient evidence.
    """

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise SageCollisionComponentTopologyError(
            ["sage_collision_openusd_runtime_missing"]
        ) from exc

    collision = _resolve_regular_file(
        sage_collision_usd_path,
        missing_error="sage_collision_usd_missing",
    )
    if not (
        isinstance(expected_source_sha256, str)
        and expected_source_sha256.startswith("sha256:")
        and len(expected_source_sha256) == 71
        and all(character in "0123456789abcdef" for character in expected_source_sha256[7:])
    ):
        raise SageCollisionComponentTopologyError(["sage_collision_expected_source_digest_invalid"])
    if (
        isinstance(expected_source_size_bytes, bool)
        or not isinstance(expected_source_size_bytes, int)
        or expected_source_size_bytes <= 0
    ):
        raise SageCollisionComponentTopologyError(["sage_collision_expected_source_size_invalid"])
    stage, source_size_bytes, source_sha256 = _load_openusd_stage_from_verified_snapshot(
        collision,
        usd_module=Usd,
    )
    if source_size_bytes != expected_source_size_bytes or source_sha256 != expected_source_sha256:
        raise SageCollisionComponentTopologyError(["sage_collision_source_identity_mismatch"])
    if not isinstance(prim_path, str) or not prim_path.startswith("/") or not prim_path.strip("/"):
        raise SageCollisionComponentTopologyError(["sage_collision_component_prim_path_invalid"])
    if isinstance(component_index, bool) or not isinstance(component_index, int):
        raise SageCollisionComponentTopologyError(["sage_collision_component_index_invalid"])
    if component_index < 0:
        raise SageCollisionComponentTopologyError(["sage_collision_component_index_invalid"])
    if not (
        isinstance(expected_geometry_digest, str)
        and expected_geometry_digest.startswith("sha256:")
        and len(expected_geometry_digest) == 71
        and all(character in "0123456789abcdef" for character in expected_geometry_digest[7:])
    ):
        raise SageCollisionComponentTopologyError(
            ["sage_collision_expected_geometry_digest_invalid"]
        )

    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise SageCollisionComponentTopologyError(["sage_collision_stage_not_z_up"])
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 1.0) > 1e-12:
        raise SageCollisionComponentTopologyError(["sage_collision_stage_not_meter_units"])

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
        raise SageCollisionComponentTopologyError(["sage_collision_component_mesh_prim_missing"])
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise SageCollisionComponentTopologyError(
            ["sage_collision_component_collision_api_missing"]
        )

    mesh = UsdGeom.Mesh(prim)
    raw_points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
    raw_counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
    raw_indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default()) or []
    if not raw_points or not raw_counts or not raw_indices:
        raise SageCollisionComponentTopologyError(
            ["sage_collision_component_mesh_topology_missing"]
        )
    world_matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    points_world: list[list[float]] = []
    for point in raw_points:
        world = world_matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        points_world.append([float(world[0]), float(world[1]), float(world[2])])
    faces = _face_rows(
        [int(value) for value in raw_counts],
        [int(value) for value in raw_indices],
    )
    components = _connected_face_components(point_count=len(points_world), faces=faces)
    if component_index >= len(components):
        raise SageCollisionComponentTopologyError(["sage_collision_component_index_out_of_range"])
    vertex_ids, component_faces = components[component_index]
    geometry_digest = canonical_digest(
        {
            "vertices_world_m": [
                [round(value, 9) for value in points_world[index]] for index in vertex_ids
            ],
            "faces_source_vertex_indices": [list(face) for face in component_faces],
        }
    )
    if geometry_digest != expected_geometry_digest:
        raise SageCollisionComponentTopologyError(
            ["sage_collision_component_geometry_digest_mismatch"]
        )

    world_vertices = [points_world[index] for index in vertex_ids]
    bounds_min = [min(row[axis] for row in world_vertices) for axis in range(3)]
    bounds_max = [max(row[axis] for row in world_vertices) for axis in range(3)]
    local_origin_world = [
        (bounds_min[0] + bounds_max[0]) / 2.0,
        (bounds_min[1] + bounds_max[1]) / 2.0,
        bounds_min[2],
    ]
    source_to_local = {source_index: index for index, source_index in enumerate(vertex_ids)}
    vertices_local = [
        [round(points_world[source_index][axis] - local_origin_world[axis], 9) for axis in range(3)]
        for source_index in vertex_ids
    ]
    faces_local = [
        [source_to_local[source_index] for source_index in face] for face in component_faces
    ]
    result: dict[str, Any] = {
        "schema_version": COMPONENT_GEOMETRY_SCHEMA_VERSION,
        "source": {
            "path": collision.name,
            "size_bytes": source_size_bytes,
            "sha256": source_sha256,
            "prim_path": prim_path,
            "component_index": component_index,
            "geometry_digest": geometry_digest,
            "collision_api_applied": True,
        },
        "coordinate_frame": {
            "up_axis": "Z",
            "meters_per_unit": meters_per_unit,
            "authored_prim_local_to_world_transform_applied": True,
            "local_origin_kind": "source_world_aabb_bottom_center",
            "local_origin_world_m": [round(value, 9) for value in local_origin_world],
        },
        "world_aabb_min_m": [round(value, 9) for value in bounds_min],
        "world_aabb_max_m": [round(value, 9) for value in bounds_max],
        "world_aabb_size_m": [round(bounds_max[axis] - bounds_min[axis], 9) for axis in range(3)],
        "vertices_local_m": vertices_local,
        "faces_local_vertex_indices": faces_local,
        "vertex_count": len(vertices_local),
        "face_count": len(faces_local),
        "local_geometry_digest": canonical_digest(
            {
                "vertices_local_m": vertices_local,
                "faces_local_vertex_indices": faces_local,
            }
        ),
        "claim_boundary": {
            "source_collision_component_exactly_replayed": False,
            "source_collision_component_deterministically_quantized": True,
            "geometry_quantization_resolution_m": 1.0e-9,
            "hidden_appearance_observed": False,
            "material_properties_observed": False,
            "native_collision_qualified": False,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def inspect_sage_collision_component_topology(
    *,
    labels_path: str | Path,
    target_instance_ids: Sequence[str],
    sage_collision_usd_path: str | Path,
    opening_probe_instance_ids: Sequence[str] = (),
    minimum_component_iou: float = 0.85,
    opening_grid_size: int = 9,
    opening_margin_fraction: float = 0.1,
) -> dict[str, Any]:
    """Match exact label boxes to transformed connected collision components."""

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise SageCollisionComponentTopologyError(
            ["sage_collision_openusd_runtime_missing"]
        ) from exc

    labels = _resolve_regular_file(
        labels_path,
        missing_error="interiorgs_labels_missing",
    )
    collision = _resolve_regular_file(
        sage_collision_usd_path,
        missing_error="sage_collision_usd_missing",
    )
    label_rows, labels_size_bytes, labels_sha256 = _load_interiorgs_labels_from_verified_snapshot(
        labels
    )
    targets = _target_map(label_rows, target_instance_ids)
    opening_targets = {str(value) for value in opening_probe_instance_ids}
    if not opening_targets.issubset(targets):
        raise SageCollisionComponentTopologyError(["opening_probe_target_not_in_requested_targets"])
    if not 0.0 < minimum_component_iou <= 1.0:
        raise SageCollisionComponentTopologyError(["minimum_component_iou_invalid"])

    stage, collision_size_bytes, collision_sha256 = _load_openusd_stage_from_verified_snapshot(
        collision, usd_module=Usd
    )
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise SageCollisionComponentTopologyError(["sage_collision_stage_not_z_up"])
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 1.0) > 1e-12:
        raise SageCollisionComponentTopologyError(["sage_collision_stage_not_meter_units"])

    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    candidate_rows: dict[str, list[dict[str, Any]]] = {target_id: [] for target_id in targets}
    geometry_by_key: dict[
        tuple[str, int], tuple[list[list[float]], tuple[tuple[int, ...], ...]]
    ] = {}
    inspected_mesh_count = 0
    inspected_component_count = 0

    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        aligned = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        prim_min = [float(aligned.GetMin()[index]) for index in range(3)]
        prim_max = [float(aligned.GetMax()[index]) for index in range(3)]
        overlapping_targets = {
            target_id: target
            for target_id, target in targets.items()
            if _boxes_overlap(target.bbox_min, target.bbox_max, prim_min, prim_max)
        }
        if not overlapping_targets:
            continue
        mesh = UsdGeom.Mesh(prim)
        raw_points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
        indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default()) or []
        if not raw_points or not counts or not indices:
            continue
        inspected_mesh_count += 1
        world_matrix = transform_cache.GetLocalToWorldTransform(prim)
        points: list[list[float]] = []
        for point in raw_points:
            world = world_matrix.Transform(
                Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
            )
            points.append([float(world[0]), float(world[1]), float(world[2])])
        faces = _face_rows([int(value) for value in counts], [int(value) for value in indices])
        components = _connected_face_components(point_count=len(points), faces=faces)
        inspected_component_count += len(components)
        for component_index, (vertex_ids, component_faces) in enumerate(components):
            component_points = [points[index] for index in vertex_ids]
            component_min = [min(row[index] for row in component_points) for index in range(3)]
            component_max = [max(row[index] for row in component_points) for index in range(3)]
            matched_targets = {
                target_id: target
                for target_id, target in overlapping_targets.items()
                if _boxes_overlap(target.bbox_min, target.bbox_max, component_min, component_max)
            }
            if not matched_targets:
                continue
            geometry_key = (str(prim.GetPath()), component_index)
            geometry_by_key[geometry_key] = (points, component_faces)
            geometry_digest = canonical_digest(
                {
                    "vertices_world_m": [
                        [round(value, 9) for value in points[index]] for index in vertex_ids
                    ],
                    "faces_source_vertex_indices": [list(face) for face in component_faces],
                }
            )
            for target_id, target in matched_targets.items():
                iou, target_coverage, component_coverage = _box_metrics(
                    target.bbox_min,
                    target.bbox_max,
                    component_min,
                    component_max,
                )
                candidate_rows[target_id].append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "component_index": component_index,
                        "world_aabb_min_m": [round(value, 9) for value in component_min],
                        "world_aabb_max_m": [round(value, 9) for value in component_max],
                        "world_aabb_size_m": [
                            round(component_max[i] - component_min[i], 9) for i in range(3)
                        ],
                        "vertex_count": len(vertex_ids),
                        "face_count": len(component_faces),
                        "collision_api_applied": prim.HasAPI(UsdPhysics.CollisionAPI),
                        "aabb_iou": round(iou, 9),
                        "target_coverage_fraction": round(target_coverage, 9),
                        "component_coverage_fraction": round(component_coverage, 9),
                        "geometry_digest": geometry_digest,
                    }
                )

    target_rows: list[dict[str, Any]] = []
    for target_id in [str(value) for value in target_instance_ids]:
        target = targets[target_id]
        ranked = sorted(
            candidate_rows[target_id],
            key=lambda row: (
                -float(row["aabb_iou"]),
                -float(row["target_coverage_fraction"]),
                str(row["prim_path"]),
                int(row["component_index"]),
            ),
        )
        qualified = [
            row
            for row in ranked
            if row["collision_api_applied"] and float(row["aabb_iou"]) >= minimum_component_iou
        ]
        best = dict(ranked[0]) if ranked else None
        opening_probe = None
        if best is not None and target_id in opening_targets:
            key = (str(best["prim_path"]), int(best["component_index"]))
            points, faces = geometry_by_key[key]
            opening_probe = _opening_probe(
                vertices=points,
                faces=faces,
                bounds_min=best["world_aabb_min_m"],
                bounds_max=best["world_aabb_max_m"],
                grid_size=opening_grid_size,
                margin_fraction=opening_margin_fraction,
                floor_band_fraction=0.25,
                cap_band_fraction=0.75,
            )
        target_rows.append(
            {
                "interiorgs_instance_id": target_id,
                "semantic_label": target.label,
                "label_world_aabb_min_m": [round(float(value), 9) for value in target.bbox_min],
                "label_world_aabb_max_m": [round(float(value), 9) for value in target.bbox_max],
                "candidate_component_count": len(ranked),
                "qualified_component_count": len(qualified),
                "component_collision_identity_passed": len(qualified) == 1,
                "best_component": best,
                "opening_probe": opening_probe,
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_files": {
            "interiorgs_labels": {
                "path": labels.name,
                "size_bytes": labels_size_bytes,
                "sha256": labels_sha256,
            },
            "sage_collision_usd": {
                "path": collision.name,
                "size_bytes": collision_size_bytes,
                "sha256": collision_sha256,
            },
        },
        "coordinate_frame": {
            "up_axis": "Z",
            "meters_per_unit": meters_per_unit,
            "authored_prim_local_to_world_transforms_applied": True,
            "interiorgs_to_sage_world_transform": "identity_after_authored_prim_transforms",
        },
        "thresholds": {
            "minimum_component_iou": minimum_component_iou,
            "opening_grid_size": opening_grid_size,
            "opening_margin_fraction": opening_margin_fraction,
        },
        "inspected_overlapping_mesh_count": inspected_mesh_count,
        "inspected_connected_component_count": inspected_component_count,
        "targets": target_rows,
        "all_component_collision_identities_passed": all(
            row["component_collision_identity_passed"] for row in target_rows
        ),
        "claim_boundary": {
            "publisher_frame_registration_consistency_only": True,
            "measurement_authoritative_surface_truth": False,
            "hidden_appearance_observed": False,
            "native_simulator_contacts_qualified": False,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def _resolved_under(path: str | Path, approved_roots: Sequence[str | Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in approved_roots]
    if not roots or not any(resolved == root or root in resolved.parents for root in roots):
        raise SageCollisionComponentTopologyError([f"path_outside_approved_roots:{resolved}"])
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--target-instance-id", action="append", required=True)
    parser.add_argument("--opening-probe-instance-id", action="append", default=[])
    parser.add_argument("--sage-collision-usd", required=True)
    parser.add_argument("--minimum-component-iou", type=float, default=0.85)
    parser.add_argument("--opening-grid-size", type=int, default=9)
    parser.add_argument("--opening-margin-fraction", type=float, default=0.1)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    labels = _resolved_under(args.labels, args.approved_root)
    collision = _resolved_under(args.sage_collision_usd, args.approved_root)
    output = _resolved_under(args.out, args.approved_root)
    result = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=args.target_instance_id,
        opening_probe_instance_ids=args.opening_probe_instance_id,
        sage_collision_usd_path=collision,
        minimum_component_iou=args.minimum_component_iou,
        opening_grid_size=args.opening_grid_size,
        opening_margin_fraction=args.opening_margin_fraction,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "receipt_digest": result["receipt_digest"],
                "all_component_collision_identities_passed": result[
                    "all_component_collision_identities_passed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "COMPONENT_GEOMETRY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SageCollisionComponentTopologyError",
    "inspect_sage_collision_component_topology",
    "main",
    "read_sage_collision_component_geometry",
]


if __name__ == "__main__":
    raise SystemExit(main())
