"""Collision-mesh floor support geometry for a robot stance in a reconstructed scene.

Used by the team embodiment stance and the website locomotion task to find the
dominant horizontal floor and to check a footprint's support against it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


class ExternalSceneRobotPlacementError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


class _SupportTriangleIndex:
    """Small deterministic XY grid over candidate floor triangles."""

    def __init__(self, triangles: np.ndarray, *, cell_size: float = 0.2) -> None:
        self.triangles = np.asarray(triangles, dtype=np.float64)
        self.cell_size = float(cell_size)
        self.minimum_xy = self.triangles[:, :, :2].min(axis=1)
        self.maximum_xy = self.triangles[:, :, :2].max(axis=1)
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.global_indices: list[int] = []
        for index, (minimum, maximum) in enumerate(
            zip(self.minimum_xy, self.maximum_xy, strict=True)
        ):
            first = np.floor(minimum / self.cell_size).astype(np.int64)
            last = np.floor(maximum / self.cell_size).astype(np.int64)
            cell_count = int(last[0] - first[0] + 1) * int(last[1] - first[1] + 1)
            if cell_count > 4096:
                self.global_indices.append(index)
                continue
            for x in range(int(first[0]), int(last[0]) + 1):
                for y in range(int(first[1]), int(last[1]) + 1):
                    self.cells.setdefault((x, y), []).append(index)

    def candidates(self, point: np.ndarray) -> np.ndarray:
        key = tuple(np.floor(np.asarray(point) / self.cell_size).astype(np.int64).tolist())
        indices = [*self.cells.get(key, []), *self.global_indices]
        if not indices:
            return self.triangles[:0]
        unique = np.unique(np.asarray(indices, dtype=np.int64))
        minimum = self.minimum_xy[unique]
        maximum = self.maximum_xy[unique]
        point_array = np.asarray(point, dtype=np.float64)
        covers_bounds = ((minimum <= point_array + 1e-9) & (maximum >= point_array - 1e-9)).all(
            axis=1
        )
        return self.triangles[unique[covers_bounds]]


#: Triangles measured at once: each block's float64 temporaries stay near 50 MB, however many the mesh
#: holds (a scanned room: 6.5 M), where the whole mesh at once took about 1.2 GB.
TRIANGLE_BLOCK = 1 << 18


def triangle_measures(stage_vertices: np.ndarray, faces: np.ndarray, *, block: int | None = None) -> dict[str, np.ndarray]:
    """Per triangle, worked out ``block`` triangles at a time: its doubled area, whether that is valid
    (finite and above 1e-12), its normal's absolute z (0 where invalid), its cross product's z, and its
    corners' mean, lowest and highest z.

    Each is the same arithmetic on the same corners as over the whole array at once, so the values are
    bit for bit those; only the block's temporaries are ever held, beside the per-triangle results.
    """
    vertices = np.asarray(stage_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    count, step = len(faces), int(block or TRIANGLE_BLOCK)
    values = {name: np.empty(count) for name in ("double_area", "normal_z", "cross_z", "centroid_z",
                                                 "lowest_z", "highest_z")}
    values["valid"] = np.empty(count, dtype=bool)
    for start in range(0, count, step):
        end = min(count, start + step)
        triangles = vertices[faces[start:end]]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
        valid = np.isfinite(double_area) & (double_area > 1e-12)
        normal_z = np.zeros_like(double_area)
        normal_z[valid] = np.abs(cross[valid, 2]) / double_area[valid]
        heights = triangles[:, :, 2]
        values["double_area"][start:end] = double_area
        values["valid"][start:end] = valid
        values["normal_z"][start:end] = normal_z
        values["cross_z"][start:end] = cross[:, 2]
        values["centroid_z"][start:end] = heights.mean(axis=1)
        values["lowest_z"][start:end] = heights.min(axis=1)
        values["highest_z"][start:end] = heights.max(axis=1)
    return values


def triangles_of(stage_vertices: np.ndarray, faces: np.ndarray, index: np.ndarray) -> np.ndarray:
    """The corners of the triangles ``index`` names (a mask or indices), as ``vertices[faces][index]`` gives them."""
    return np.asarray(stage_vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)[index]]


def _infer_horizontal_support_surface(
    *,
    stage_vertices: np.ndarray,
    faces: np.ndarray,
    measures: Mapping[str, np.ndarray] | None = None,
    return_support_triangles: bool = True,
) -> tuple[float, np.ndarray | None, dict[str, Any]]:
    """Infer the dominant lower horizontal surface without using the global minimum.

    A reconstruction can contain low outliers below the actual floor.  Treating the
    minimum vertex as ground makes those outliers authoritative and can also place a
    robot just outside the finite mesh.  This gate instead selects the lower
    horizontal height band with the greatest projected triangle area.  It remains a
    derived geometry candidate until Isaac observes live contact.

    The triangles are measured in blocks (``triangle_measures``, or ``measures`` already taken of
    these ones). A caller that needs only the floor's height passes ``return_support_triangles=False``
    and gets ``None`` in their place: a scanned room's floor is most of its triangles.
    """

    corners, corner_faces = np.asarray(stage_vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    measures = triangle_measures(corners, corner_faces) if measures is None else measures
    double_area, valid = measures["double_area"], measures["valid"]
    normal_z, centroids_z = measures["normal_z"], measures["centroid_z"]
    minimum_z = float(np.min(stage_vertices[:, 2]))
    maximum_z = float(np.max(stage_vertices[:, 2]))
    vertical_extent = max(0.0, maximum_z - minimum_z)
    all_horizontal = valid & (normal_z >= 0.9)
    horizontal_indices_all = np.flatnonzero(all_horizontal)
    if horizontal_indices_all.size == 0:
        raise ExternalSceneRobotPlacementError(
            ["external_placement_horizontal_support_surface_missing"]
        )
    horizontal_areas_all = 0.5 * double_area[horizontal_indices_all]
    order = np.argsort(centroids_z[horizontal_indices_all])
    ordered_indices = horizontal_indices_all[order]
    cumulative_area = np.cumsum(horizontal_areas_all[order])
    cumulative_area /= cumulative_area[-1]

    def weighted_height_quantile(quantile: float) -> float:
        index = min(int(np.searchsorted(cumulative_area, quantile, side="left")), len(order) - 1)
        return float(centroids_z[ordered_indices[index]])

    robust_lower_z = weighted_height_quantile(0.05)
    robust_upper_z = weighted_height_quantile(0.95)
    robust_vertical_extent = max(0.0, robust_upper_z - robust_lower_z)
    lower_ceiling = robust_lower_z + 0.45 * robust_vertical_extent
    horizontal = all_horizontal & (centroids_z <= lower_ceiling + 1e-9)
    if not bool(np.any(horizontal)):
        raise ExternalSceneRobotPlacementError(
            ["external_placement_horizontal_support_surface_missing"]
        )

    bin_width = max(0.05, vertical_extent / 100.0)
    horizontal_indices = np.flatnonzero(horizontal)
    horizontal_z = centroids_z[horizontal_indices]
    bin_ids = np.floor((horizontal_z - minimum_z) / bin_width).astype(np.int64)
    area_by_bin = np.bincount(bin_ids, weights=0.5 * double_area[horizontal_indices])
    selected_bin = int(np.argmax(area_by_bin))
    selected_indices = horizontal_indices[bin_ids == selected_bin]
    selected_areas = 0.5 * double_area[selected_indices]
    floor_z = float(np.average(centroids_z[selected_indices], weights=selected_areas))
    tolerance = 1.1 * bin_width
    support_mask = horizontal & (np.abs(centroids_z - floor_z) <= tolerance)
    support_index = np.flatnonzero(support_mask)
    if support_index.size == 0:
        raise ExternalSceneRobotPlacementError(
            ["external_placement_horizontal_support_surface_missing"]
        )
    # The support's plan bounds, a block of its triangles at a time: min and max are exact either way.
    support_low = np.full(2, np.inf)
    support_high = np.full(2, -np.inf)
    for start in range(0, support_index.size, TRIANGLE_BLOCK):
        plan = triangles_of(corners, corner_faces, support_index[start:start + TRIANGLE_BLOCK])[:, :, :2]
        support_low = np.minimum(support_low, plan.min(axis=(0, 1)))
        support_high = np.maximum(support_high, plan.max(axis=(0, 1)))
    support_triangles = (triangles_of(corners, corner_faces, support_mask) if return_support_triangles
                         else None)
    evidence = {
        "schema_version": "external_scene_support_surface_candidate.v1",
        "status": "derived_geometry_candidate_unverified_in_isaac",
        "selection_method": "dominant_lower_horizontal_triangle_area_band",
        "floor_height_collision_stage": round(floor_z, 9),
        "height_bin_width_stage_units": round(float(bin_width), 9),
        "height_tolerance_stage_units": round(float(tolerance), 9),
        "horizontal_normal_abs_z_minimum": 0.9,
        "lower_scene_fraction_limit": 0.45,
        "robust_horizontal_height_quantiles": {
            "q05": round(robust_lower_z, 9),
            "q95": round(robust_upper_z, 9),
        },
        "support_triangle_count": int(support_index.size),
        "support_bounds_xy_collision_stage": {
            "minimum": [round(float(value), 9) for value in support_low],
            "maximum": [round(float(value), 9) for value in support_high],
        },
        "selected_band_area_stage_units_squared": round(float(area_by_bin[selected_bin]), 9),
        "global_minimum_vertex_z_rejected_as_floor": round(minimum_z, 9),
        "live_isaac_contact_qualified": False,
    }
    return floor_z, support_triangles, evidence


def _support_heights_at_points(
    *,
    support_triangles: np.ndarray | _SupportTriangleIndex,
    points_xy: np.ndarray,
    floor_z: float,
    height_tolerance: float,
) -> list[float | None]:
    """Return an interpolated support height for each point, or ``None``."""

    index = (
        support_triangles
        if isinstance(support_triangles, _SupportTriangleIndex)
        else _SupportTriangleIndex(np.asarray(support_triangles, dtype=np.float64))
    )
    points = np.asarray(points_xy, dtype=np.float64)
    results: list[float | None] = []
    epsilon = 1e-9
    for point in points:
        local = index.candidates(point)
        if local.size == 0:
            results.append(None)
            continue
        a = local[:, 0, :2]
        b = local[:, 1, :2]
        c = local[:, 2, :2]
        v0 = b - a
        v1 = c - a
        v2 = point - a
        denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        nondegenerate = np.abs(denominator) > epsilon
        u = np.zeros_like(denominator)
        v = np.zeros_like(denominator)
        u[nondegenerate] = (
            v2[nondegenerate, 0] * v1[nondegenerate, 1]
            - v1[nondegenerate, 0] * v2[nondegenerate, 1]
        ) / denominator[nondegenerate]
        v[nondegenerate] = (
            v0[nondegenerate, 0] * v2[nondegenerate, 1]
            - v2[nondegenerate, 0] * v0[nondegenerate, 1]
        ) / denominator[nondegenerate]
        inside = nondegenerate & (u >= -epsilon) & (v >= -epsilon) & (u + v <= 1.0 + epsilon)
        if not bool(np.any(inside)):
            results.append(None)
            continue
        weights_a = 1.0 - u - v
        heights = weights_a * local[:, 0, 2] + u * local[:, 1, 2] + v * local[:, 2, 2]
        qualified = (
            inside
            & np.isfinite(heights)
            & (np.abs(heights - float(floor_z)) <= float(height_tolerance))
        )
        if not bool(np.any(qualified)):
            results.append(None)
            continue
        # Prefer the candidate closest to the selected floor band when triangles overlap.
        qualified_heights = heights[qualified]
        results.append(
            float(qualified_heights[np.argmin(np.abs(qualified_heights - float(floor_z)))])
        )
    return results


def _footprint_support_report(
    *,
    support_triangles: np.ndarray | _SupportTriangleIndex,
    position: Sequence[float],
    yaw: float,
    half_extent_xy: tuple[float, float],
    floor_z: float,
    height_tolerance: float,
) -> dict[str, Any]:
    """Require support beneath the base center and all four oriented corners."""

    half_x, half_y = (float(value) for value in half_extent_xy)
    local = np.asarray(
        [
            [0.0, 0.0],
            [half_x, half_y],
            [half_x, -half_y],
            [-half_x, half_y],
            [-half_x, -half_y],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    points = local @ rotation.T + np.asarray(position[:2], dtype=np.float64)
    heights = _support_heights_at_points(
        support_triangles=support_triangles,
        points_xy=points,
        floor_z=floor_z,
        height_tolerance=height_tolerance,
    )
    supported = [height is not None for height in heights]
    return {
        "schema_version": "external_scene_base_support_coverage.v1",
        "sample_method": "oriented_footprint_center_and_four_corners",
        "sample_points_xy_collision_stage": [
            [round(float(point[0]), 9), round(float(point[1]), 9)] for point in points
        ],
        "sample_support_heights_collision_stage": [
            None if height is None else round(float(height), 9) for height in heights
        ],
        "supported_sample_count": int(sum(supported)),
        "required_sample_count": len(supported),
        "full_sample_support_candidate": bool(all(supported)),
        "live_contact_qualified": False,
    }


__all__ = [
    "ExternalSceneRobotPlacementError",
]
