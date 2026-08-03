"""Propose a Franka runtime pose around a registered external-scene target.

The dynamic solver reuses Blueprint's general ring-scan placement engine with a
conservative collision-mesh triangle-overlap probe. The deterministic result remains a runtime
visualization candidate until metric scale, live contact, full footprint, and
reach checks are independently qualified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .external_scene_collision_candidate import _flatten_glb, _sha256
from .external_scene_inspection_outcome import build_franka_inspection_outcome_contract
from .provider_nurec_robot_placement import (
    build_default_franka_policy_trace_request,
    build_franka_inspection_controller_cohort_request,
)
from .scene_placement.placement import ring_scan_stand_pose
from .scene_placement.robot_profile import get_robot_profile
from .scene_placement.types import SceneObject


REQUEST_SCHEMA = "external_scene_robot_placement_request.v1"
RESULT_SCHEMA = "external_scene_robot_placement_candidate.v1"


class ExternalSceneRobotPlacementError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


class _TriangleXYIndex:
    """Deterministic uniform-grid broad phase for XY triangle queries."""

    def __init__(self, triangles_xy: np.ndarray, *, cell_size: float = 0.5) -> None:
        self.triangles = np.asarray(triangles_xy, dtype=np.float64)
        self.cell_size = float(cell_size)
        self.minimum = self.triangles.min(axis=1)
        self.maximum = self.triangles.max(axis=1)
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.global_indices: list[int] = []
        for index, (minimum, maximum) in enumerate(zip(self.minimum, self.maximum, strict=True)):
            first = np.floor(minimum / self.cell_size).astype(np.int64)
            last = np.floor(maximum / self.cell_size).astype(np.int64)
            cell_count = int(last[0] - first[0] + 1) * int(last[1] - first[1] + 1)
            if cell_count > 1024:
                self.global_indices.append(index)
                continue
            for x in range(int(first[0]), int(last[0]) + 1):
                for y in range(int(first[1]), int(last[1]) + 1):
                    self.cells.setdefault((x, y), []).append(index)

    def query(self, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
        first = np.floor(np.asarray(minimum) / self.cell_size).astype(np.int64)
        last = np.floor(np.asarray(maximum) / self.cell_size).astype(np.int64)
        indices: list[int] = list(self.global_indices)
        for x in range(int(first[0]), int(last[0]) + 1):
            for y in range(int(first[1]), int(last[1]) + 1):
                indices.extend(self.cells.get((x, y), []))
        if not indices:
            return self.triangles[:0]
        unique = np.unique(np.asarray(indices, dtype=np.int64))
        query_minimum = np.asarray(minimum, dtype=np.float64)
        query_maximum = np.asarray(maximum, dtype=np.float64)
        overlaps = (
            (self.maximum[unique] >= query_minimum) & (self.minimum[unique] <= query_maximum)
        ).all(axis=1)
        return self.triangles[unique[overlaps]]


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


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite3(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result).all() else None


def _footprint_overlap_counts(
    *,
    stage_vertices: np.ndarray,
    faces: np.ndarray,
    position: Sequence[float],
    yaw: float,
    floor_z: float,
    half_extent_xy: tuple[float, float],
    probe_clearance: float,
    obstacle_height: float,
) -> tuple[int, int]:
    """Count vertices and conservative triangle bounds crossing a robot footprint.

    Vertex-only probes miss large fixture triangles whose edges or interiors cross
    the footprint while every vertex remains outside it. Triangle AABB overlap in
    the robot-local frame is deliberately conservative: false positives lead to a
    different candidate or abstention, while false negatives can destabilize the
    articulation after spawn.
    """

    x, y = float(position[0]), float(position[1])
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    delta = stage_vertices[:, :2] - [x, y]
    local = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
            stage_vertices[:, 2],
        )
    )
    half_x = float(half_extent_xy[0]) + float(probe_clearance)
    half_y = float(half_extent_xy[1]) + float(probe_clearance)
    lower_z = float(floor_z) + 0.04
    upper_z = float(floor_z) + float(obstacle_height)
    vertex_hits = int(
        (
            (np.abs(local[:, 0]) <= half_x)
            & (np.abs(local[:, 1]) <= half_y)
            & (local[:, 2] >= lower_z)
            & (local[:, 2] <= upper_z)
        ).sum()
    )
    triangles = local[faces]
    minimum = triangles.min(axis=1)
    maximum = triangles.max(axis=1)
    vertical = (maximum[:, 2] >= lower_z) & (minimum[:, 2] <= upper_z)
    triangle_hits = _triangle_footprint_overlap_count(
        obstacle_triangles_xy=triangles[vertical, :, :2],
        position=(0.0, 0.0),
        yaw=0.0,
        half_extent_xy=(half_x, half_y),
    )
    return vertex_hits, triangle_hits


def _triangle_footprint_overlap_count(
    *,
    obstacle_triangles_xy: np.ndarray | _TriangleXYIndex,
    position: Sequence[float],
    yaw: float,
    half_extent_xy: tuple[float, float],
) -> int:
    """Conservatively count triangle bounds crossing an oriented footprint.

    The same robot-local test is used while searching and while reporting the
    selected pose. Keeping those two gates identical prevents a fast world-AABB
    prefilter from selecting a pose that the final oriented-footprint check then
    rejects.
    """

    half_x, half_y = (float(value) for value in half_extent_xy)
    x, y = float(position[0]), float(position[1])
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    if isinstance(obstacle_triangles_xy, _TriangleXYIndex):
        world_half_x = abs(cosine) * half_x + abs(sine) * half_y
        world_half_y = abs(sine) * half_x + abs(cosine) * half_y
        triangles = obstacle_triangles_xy.query(
            np.asarray([x - world_half_x, y - world_half_y]),
            np.asarray([x + world_half_x, y + world_half_y]),
        )
    else:
        triangles = np.asarray(obstacle_triangles_xy, dtype=np.float64)
    if triangles.size == 0:
        return 0
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError("external_placement_obstacle_triangles_xy_invalid")
    delta = triangles - [x, y]
    local_x = cosine * delta[:, :, 0] + sine * delta[:, :, 1]
    local_y = -sine * delta[:, :, 0] + cosine * delta[:, :, 1]
    return int(
        (
            (local_x.max(axis=1) >= -half_x)
            & (local_x.min(axis=1) <= half_x)
            & (local_y.max(axis=1) >= -half_y)
            & (local_y.min(axis=1) <= half_y)
        ).sum()
    )


def _infer_horizontal_support_surface(
    *,
    stage_vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Infer the dominant lower horizontal surface without using the global minimum.

    A reconstruction can contain low outliers below the actual floor.  Treating the
    minimum vertex as ground makes those outliers authoritative and can also place a
    robot just outside the finite mesh.  This gate instead selects the lower
    horizontal height band with the greatest projected triangle area.  It remains a
    derived geometry candidate until Isaac observes live contact.
    """

    triangles = np.asarray(stage_vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edges_a, edges_b)
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1e-12)
    normal_z = np.zeros_like(double_area)
    normal_z[valid] = np.abs(cross[valid, 2]) / double_area[valid]
    centroids_z = triangles[:, :, 2].mean(axis=1)
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
    support_triangles = triangles[support_mask]
    if support_triangles.size == 0:
        raise ExternalSceneRobotPlacementError(
            ["external_placement_horizontal_support_surface_missing"]
        )
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
        "support_triangle_count": int(support_triangles.shape[0]),
        "support_bounds_xy_collision_stage": {
            "minimum": [
                round(float(value), 9) for value in support_triangles[:, :, :2].min(axis=(0, 1))
            ],
            "maximum": [
                round(float(value), 9) for value in support_triangles[:, :, :2].max(axis=(0, 1))
            ],
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


def _select_supported_physics_probe(
    *,
    support_triangles: np.ndarray | _SupportTriangleIndex,
    obstacle_triangles_xy: np.ndarray | _TriangleXYIndex,
    position: Sequence[float],
    half_extent_xy: tuple[float, float],
    floor_z: float,
    height_tolerance: float,
) -> dict[str, Any] | None:
    """Pick a source-mesh-supported probe point outside the derived base mount."""

    radius_start = max(float(half_extent_xy[0]), float(half_extent_xy[1])) + 0.2
    candidates: list[list[float]] = []
    for radius in (radius_start, radius_start + 0.2, radius_start + 0.4):
        for index in range(16):
            angle = 2.0 * math.pi * index / 16.0
            candidates.append(
                [
                    float(position[0]) + radius * math.cos(angle),
                    float(position[1]) + radius * math.sin(angle),
                ]
            )
    points = np.asarray(candidates, dtype=np.float64)
    heights = _support_heights_at_points(
        support_triangles=support_triangles,
        points_xy=points,
        floor_z=floor_z,
        height_tolerance=height_tolerance,
    )
    best_candidate: dict[str, Any] | None = None
    for point, height in zip(points, heights, strict=True):
        if height is None:
            continue
        obstacle_hits = _triangle_footprint_overlap_count(
            obstacle_triangles_xy=obstacle_triangles_xy,
            position=point,
            yaw=0.0,
            half_extent_xy=(0.075, 0.075),
        )
        candidate = {
            "schema_version": "external_scene_physics_probe_candidate.v1",
            "selection_status": "derived_geometry_candidate_unverified_in_isaac",
            "selection_method": (
                "supported_floor_ring_outside_robot_mount"
                if obstacle_hits == 0
                else "supported_floor_ring_minimum_source_collision_complexity"
            ),
            "probe_xy_m": [round(float(point[0]), 9), round(float(point[1]), 9)],
            "ground_height_m": round(float(height), 9),
            "source_surface_support_observed": True,
            "obstacle_overlap_probe_hits": int(obstacle_hits),
            "probe_may_intersect_non_floor_source_geometry": bool(obstacle_hits > 0),
            "manufacture_ground_plane": False,
            "live_contact_qualified": False,
        }
        if obstacle_hits == 0:
            return candidate
        if best_candidate is None or obstacle_hits < int(
            best_candidate["obstacle_overlap_probe_hits"]
        ):
            best_candidate = candidate
    return best_candidate


def build_external_scene_robot_placement_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalSceneRobotPlacementError(["external_placement_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("external_placement_request_schema_invalid")
    if request.get("robot_id") != "franka_panda":
        errors.append("external_placement_robot_must_be_default_franka")
    for key in (
        "source_scene_digest",
        "target_analysis_digest",
        "target_binding_digest",
        "scene_frame_binding_digest",
        "collision_candidate_digest",
        "collision_source_digest",
    ):
        if not _digest(request.get(key)):
            errors.append(f"external_placement_{key}_invalid")
    if _finite3(request.get("target_position_collision_stage")) is None:
        errors.append("external_placement_target_position_invalid")
    uncertainty = request.get("target_spatial_uncertainty_stage_units")
    if (
        isinstance(uncertainty, bool)
        or not isinstance(uncertainty, (int, float))
        or not math.isfinite(float(uncertainty))
        or float(uncertainty) <= 0
    ):
        errors.append("external_placement_target_uncertainty_invalid")
    if request.get("metric_scale_status") not in {
        "validated",
        "provider_declared_not_independently_validated",
        "unverified",
    }:
        errors.append("external_placement_metric_status_invalid")
    if request.get("collision_status") not in {"candidate_compiled", "qualified"}:
        errors.append("external_placement_collision_status_invalid")
    if request.get("candidate_may_self_authorize") is not False:
        errors.append("external_placement_self_authorization_forbidden")
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        errors.append("external_placement_request_digest_mismatch")
    if errors:
        raise ExternalSceneRobotPlacementError(errors)
    request["request_digest"] = expected
    return request


def propose_external_scene_robot_placement(
    *,
    collision_glb_path: str | Path,
    request: Mapping[str, Any],
    target_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    admitted = build_external_scene_robot_placement_request(request)
    try:
        analysis = json.loads(json.dumps(dict(target_analysis), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalSceneRobotPlacementError(
            ["external_placement_target_analysis_not_json"]
        ) from exc
    supplied_analysis_digest = analysis.get("target_analysis_digest")
    expected_analysis_digest = canonical_digest(analysis, digest_field="target_analysis_digest")
    if supplied_analysis_digest != expected_analysis_digest:
        raise ExternalSceneRobotPlacementError(
            ["external_placement_target_analysis_digest_mismatch"]
        )
    if supplied_analysis_digest != admitted["target_analysis_digest"]:
        raise ExternalSceneRobotPlacementError(
            ["external_placement_request_target_analysis_mismatch"]
        )
    glb = Path(collision_glb_path).resolve(strict=True)
    if glb.suffix.lower() != ".glb" or _sha256(glb) != admitted["collision_source_digest"]:
        raise ExternalSceneRobotPlacementError(["external_placement_collision_source_mismatch"])
    vertices, faces, _ = _flatten_glb(glb)
    # Same exact GLB Y-up -> collision-stage Z-up transform used by the collider compiler.
    stage_vertices = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    floor_z, support_triangles, support_surface = _infer_horizontal_support_surface(
        stage_vertices=stage_vertices,
        faces=faces,
    )
    support_index = _SupportTriangleIndex(support_triangles)
    target = _finite3(admitted["target_position_collision_stage"])
    assert target is not None
    uncertainty = float(admitted["target_spatial_uncertainty_stage_units"])
    horizontal_half_extent = max(0.05, uncertainty)
    target_object = SceneObject(
        id="registered_scene_task_target",
        label=str(admitted.get("target_label") or "registered task target"),
        bbox_min=tuple(target - [horizontal_half_extent, horizontal_half_extent, 0.15]),
        bbox_max=tuple(target + [horizontal_half_extent, horizontal_half_extent, 0.15]),
        centroid=tuple(target),
        category="fixture",
        source="registered_external_reconstruction",
        confidence=float(admitted.get("visual_confidence") or 0.0),
    )
    profile = get_robot_profile("franka_panda")
    half_x, half_y, _ = profile.footprint_half_extent_xyz
    # Franka is a fixed-base arm. Derive a support-mount height that aligns the
    # nominal shoulder with the selected affordance instead of silently treating
    # the manipulator like a floor-standing robot. This is a simulation support
    # candidate only; physical baseplate strength and metric height remain
    # independent qualification gates.
    mount_height = max(
        0.05,
        float(target[2]) - floor_z - float(profile.shoulder_above_root_m),
    )
    mount_top_z = floor_z + mount_height
    triangle_stage = stage_vertices[faces]
    triangle_stage_minimum = triangle_stage.min(axis=1)
    triangle_stage_maximum = triangle_stage.max(axis=1)
    lower_obstacle_z = floor_z + 0.04
    support_obstacle_height = max(
        mount_height,
        2.0 * profile.footprint_half_extent_xyz[2],
    )
    upper_obstacle_z = floor_z + support_obstacle_height
    vertical_obstacle_faces = (triangle_stage_maximum[:, 2] >= lower_obstacle_z) & (
        triangle_stage_minimum[:, 2] <= upper_obstacle_z
    )
    triangle_edges_a = triangle_stage[:, 1] - triangle_stage[:, 0]
    triangle_edges_b = triangle_stage[:, 2] - triangle_stage[:, 0]
    triangle_cross = np.cross(triangle_edges_a, triangle_edges_b)
    triangle_double_area = np.linalg.norm(triangle_cross, axis=1)
    triangle_normal_abs_z = np.zeros_like(triangle_double_area)
    valid_triangle_area = triangle_double_area > 1e-12
    triangle_normal_abs_z[valid_triangle_area] = (
        np.abs(triangle_cross[valid_triangle_area, 2]) / triangle_double_area[valid_triangle_area]
    )
    triangle_centroid_z = triangle_stage[:, :, 2].mean(axis=1)
    floor_band_faces = (triangle_normal_abs_z >= 0.9) & (
        np.abs(triangle_centroid_z - floor_z)
        <= float(support_surface["height_tolerance_stage_units"])
    )
    # Floor reconstruction noise inside the selected support band is not an
    # obstacle. Leaving it in the obstacle index makes every supported stance
    # appear occupied even though the same triangles are its floor evidence.
    vertical_obstacle_faces &= ~floor_band_faces
    obstacle_triangles_xy = triangle_stage[vertical_obstacle_faces, :, :2]
    obstacle_index = _TriangleXYIndex(obstacle_triangles_xy)

    def probe(pose, yaw) -> int:
        # Use the exact same robot-local conservative triangle test here and in
        # the final report. A differently oriented world-AABB prefilter can
        # disagree with the final OBB-aligned bounds even when each test is
        # independently conservative.
        inflated_half_x = half_x + profile.probe_clearance_m
        inflated_half_y = half_y + profile.probe_clearance_m
        obstacle_hits = _triangle_footprint_overlap_count(
            obstacle_triangles_xy=obstacle_index,
            position=pose,
            yaw=yaw,
            half_extent_xy=(inflated_half_x, inflated_half_y),
        )
        support = _footprint_support_report(
            support_triangles=support_index,
            position=pose,
            yaw=yaw,
            half_extent_xy=(inflated_half_x, inflated_half_y),
            floor_z=floor_z,
            height_tolerance=float(support_surface["height_tolerance_stage_units"]),
        )
        missing_support_samples = int(support["required_sample_count"]) - int(
            support["supported_sample_count"]
        )
        return obstacle_hits + missing_support_samples

    stance = ring_scan_stand_pose(
        target_object,
        probe=probe,
        floor_z=floor_z,
        standing_distance=profile.standoff_range_m[0],
        max_standing_distance=profile.standoff_range_m[1],
        radial_step=profile.probe_step_m,
        n_azimuths=144,
        robot_profile=profile,
    )
    source_supported_stance_found = bool(stance.clear)
    bounded_floor_proxy_candidate = False
    source_collision_exclusion_required = False
    if not stance.clear:

        def bounded_proxy_probe(pose, yaw) -> int:
            obstacle_hits = _triangle_footprint_overlap_count(
                obstacle_triangles_xy=obstacle_index,
                position=pose,
                yaw=yaw,
                half_extent_xy=(
                    half_x + profile.probe_clearance_m,
                    half_y + profile.probe_clearance_m,
                ),
            )
            center_support = _footprint_support_report(
                support_triangles=support_index,
                position=pose,
                yaw=yaw,
                half_extent_xy=(0.0, 0.0),
                floor_z=floor_z,
                height_tolerance=float(support_surface["height_tolerance_stage_units"]),
            )
            return obstacle_hits + (0 if center_support["full_sample_support_candidate"] else 1)

        proxy_stance = ring_scan_stand_pose(
            target_object,
            probe=bounded_proxy_probe,
            floor_z=floor_z,
            standing_distance=profile.standoff_range_m[0],
            max_standing_distance=profile.standoff_range_m[1],
            radial_step=profile.probe_step_m,
            n_azimuths=144,
            robot_profile=profile,
        )
        if proxy_stance.clear:
            stance = proxy_stance
            bounded_floor_proxy_candidate = True
    if not stance.clear:

        def proxy_composed_support_probe(pose, yaw) -> int:
            support = _footprint_support_report(
                support_triangles=support_index,
                position=pose,
                yaw=yaw,
                half_extent_xy=(
                    half_x + profile.probe_clearance_m,
                    half_y + profile.probe_clearance_m,
                ),
                floor_z=floor_z,
                height_tolerance=float(support_surface["height_tolerance_stage_units"]),
            )
            return int(support["required_sample_count"]) - int(support["supported_sample_count"])

        proxy_composed_stance = ring_scan_stand_pose(
            target_object,
            probe=proxy_composed_support_probe,
            floor_z=floor_z,
            standing_distance=profile.standoff_range_m[0],
            max_standing_distance=profile.standoff_range_m[1],
            radial_step=profile.probe_step_m,
            n_azimuths=144,
            robot_profile=profile,
        )
        if proxy_composed_stance.clear:
            stance = proxy_composed_stance
            bounded_floor_proxy_candidate = True
            source_collision_exclusion_required = True
    reach_limit = float(profile.max_shoulder_to_affordance_m())

    def shoulder_distance(candidate) -> float:
        shoulder = np.asarray(candidate.position) + [
            0.0,
            0.0,
            mount_height + profile.shoulder_above_root_m,
        ]
        return float(np.linalg.norm(target - shoulder))

    placement_selection_strategy = (
        "proxy_composed_task_zone_candidate_required"
        if source_collision_exclusion_required
        else "bounded_floor_proxy_candidate_required"
        if bounded_floor_proxy_candidate
        else "nearest_nominal_standoff_collision_clear_candidate"
    )
    initial_shoulder_distance = shoulder_distance(stance)
    # A collision-clear pose at the nominal standoff can still put a high or deep
    # target just outside the arm envelope. Search the narrow gap down to the
    # profile's own base standoff, while retaining the footprint-clearance probe.
    # This is an analytic rescue candidate only; it cannot qualify metric reach.
    if stance.clear and initial_shoulder_distance > reach_limit:
        rescue_minimum_standoff = max(
            float(profile.standing_distance_m),
            float(profile.probe_clearance_m),
        )
        rescue = ring_scan_stand_pose(
            target_object,
            probe=probe,
            floor_z=floor_z,
            standing_distance=rescue_minimum_standoff,
            max_standing_distance=profile.standoff_range_m[0],
            radial_step=profile.probe_step_m,
            n_azimuths=144,
            robot_profile=profile,
        )
        if rescue.clear and shoulder_distance(rescue) <= reach_limit:
            stance = rescue
            placement_selection_strategy = "collision_clear_analytic_reach_rescue_candidate"

    vertex_hits, triangle_hits = _footprint_overlap_counts(
        stage_vertices=stage_vertices,
        faces=faces,
        position=stance.position,
        yaw=stance.yaw,
        floor_z=floor_z,
        half_extent_xy=(half_x, half_y),
        probe_clearance=profile.probe_clearance_m,
        obstacle_height=support_obstacle_height,
    )
    qualified_obstacle_hits = _triangle_footprint_overlap_count(
        obstacle_triangles_xy=obstacle_index,
        position=stance.position,
        yaw=stance.yaw,
        half_extent_xy=(
            half_x + profile.probe_clearance_m,
            half_y + profile.probe_clearance_m,
        ),
    )
    shoulder_distance_value = shoulder_distance(stance)
    analytic_reach_candidate = bool(shoulder_distance_value <= reach_limit)
    base_support = _footprint_support_report(
        support_triangles=support_index,
        position=stance.position,
        yaw=stance.yaw,
        half_extent_xy=(half_x + profile.probe_clearance_m, half_y + profile.probe_clearance_m),
        floor_z=floor_z,
        height_tolerance=float(support_surface["height_tolerance_stage_units"]),
    )
    footprint_supported = bool(base_support["full_sample_support_candidate"])
    footprint_clear = bool(
        source_supported_stance_found and qualified_obstacle_hits == 0 and footprint_supported
    )
    physics_probe_candidate = _select_supported_physics_probe(
        support_triangles=support_index,
        obstacle_triangles_xy=obstacle_index,
        position=stance.position,
        half_extent_xy=(half_x, half_y),
        floor_z=floor_z,
        height_tolerance=float(support_surface["height_tolerance_stage_units"]),
    )
    proxy_half_extents = (
        half_x + profile.probe_clearance_m + 0.05,
        half_y + profile.probe_clearance_m + 0.05,
    )
    floor_proxy = (
        {
            "schema_version": "bounded_floor_proxy_candidate.v1",
            "status": "simulator_support_candidate_only_requires_qualification",
            "prim_path": "/World/BlueprintDerivedSupport/BoundedFloorPatch",
            "center_xyz_collision_stage": [
                round(float(stance.position[0]), 9),
                round(float(stance.position[1]), 9),
                round(float(floor_z - 0.025), 9),
            ],
            "half_extents_xy_stage_units": [
                round(float(proxy_half_extents[0]), 9),
                round(float(proxy_half_extents[1]), 9),
            ],
            "thickness_stage_units": 0.05,
            "top_z_collision_stage": round(float(floor_z), 9),
            "bounded_to_robot_support_zone": True,
            "source_support_sample_count": int(base_support["supported_sample_count"]),
            "required_support_sample_count": int(base_support["required_sample_count"]),
            "source_collider_contact_qualification_effect": "none",
            "exclude_from_source_collider_physics_probe": True,
            "source_collision_exclusion_required_for_policy_lane": (
                source_collision_exclusion_required
            ),
            "independent_metric_scale_qualified": False,
            "physical_floor_continuity_qualified": False,
            "claim_boundary": (
                "Derived bounded Isaac support patch for simulation continuity only; "
                "not evidence that the captured floor is continuous, level, metric, or load-bearing."
            ),
        }
        if bounded_floor_proxy_candidate
        and bool(base_support["sample_support_heights_collision_stage"][0] is not None)
        and (qualified_obstacle_hits == 0 or source_collision_exclusion_required)
        and (not footprint_supported or source_collision_exclusion_required)
        else None
    )
    placement = {
        "schema_version": RESULT_SCHEMA,
        "status": "runtime_visualization_candidate_only" if footprint_clear else "abstained",
        "request_digest": admitted["request_digest"],
        "robot_id": "franka_panda",
        "official_isaac_asset": profile.simulator_asset_refs["isaac_asset"],
        "robot_prim_path": profile.usd_prim_path,
        "target_position_collision_stage": list(admitted["target_position_collision_stage"]),
        "target_spatial_uncertainty_stage_units": uncertainty,
        "robot_pose_xyzyaw_collision_stage": [
            round(float(stance.position[0]), 9),
            round(float(stance.position[1]), 9),
            round(float(mount_top_z), 9),
            round(float(stance.yaw), 12),
        ],
        "floor_height_collision_stage": round(floor_z, 9),
        "support_surface": support_surface,
        "base_support_coverage": base_support,
        "bounded_floor_proxy": floor_proxy,
        "proxy_composed_evaluation_plan": {
            "schema_version": "external_scene_proxy_composed_evaluation_plan.v1",
            "status": (
                "required_before_policy_evaluation"
                if source_collision_exclusion_required
                else "not_required"
            ),
            "appearance_source": "immutable_full_resolution_splat",
            "source_collision_qualification_preserved_separately": True,
            "source_collision_prim_path": (
                "/World/BlueprintReconstruction/Collision/ExternalSceneMesh"
            ),
            "source_collision_enabled_in_policy_lane": not source_collision_exclusion_required,
            "bounded_floor_proxy_required": bool(floor_proxy is not None),
            "task_zone_simready_asset_required": bool(
                "inspection" not in str(admitted.get("target_label") or "").lower()
                and "inspection"
                not in str(
                    (
                        target_analysis.get("selected_target")
                        if isinstance(target_analysis.get("selected_target"), Mapping)
                        else {}
                    ).get("task_family")
                    or ""
                ).lower()
            ),
            "policy_result_claim_ceiling": "exact_proxy_composed_simulation_only",
            "physical_site_claim_effect": "none",
        },
        "physics_probe_candidate": physics_probe_candidate,
        "fixed_base_support_mount": {
            "schema_version": "fixed_base_support_mount_candidate.v1",
            "status": "simulator_support_candidate_only",
            "prim_path": "/World/BlueprintDerivedSupport/FrankaMount",
            "center_xyz_collision_stage": [
                round(float(stance.position[0]), 9),
                round(float(stance.position[1]), 9),
                round(float(floor_z + 0.5 * mount_height), 9),
            ],
            "top_z_collision_stage": round(float(mount_top_z), 9),
            "height_stage_units": round(float(mount_height), 9),
            "half_extents_xy_stage_units": [round(float(half_x), 9), round(float(half_y), 9)],
            "collision_checked_height_stage_units": round(float(support_obstacle_height), 9),
            "static_collision_required": True,
            "physical_load_capacity_qualified": False,
            "independent_metric_height_qualified": False,
            "claim_boundary": (
                "Derived static simulator support only; not evidence of a physical baseplate, "
                "anchoring, load capacity, installation safety, or metric calibration."
            ),
        },
        "mesh_vertex_overlap_probe_hits": vertex_hits,
        "mesh_vertex_overlap_probe_clear": bool(vertex_hits == 0),
        "raw_mesh_triangle_aabb_overlap_probe_hits": triangle_hits,
        "mesh_triangle_aabb_overlap_probe_hits": qualified_obstacle_hits,
        "mesh_triangle_aabb_overlap_probe_clear": footprint_clear,
        "standoff_stage_units": round(float(stance.standoff_m), 9),
        "placement_selection_strategy": placement_selection_strategy,
        "analytic_shoulder_to_target_distance_stage_units": round(shoulder_distance_value, 9),
        "analytic_profile_reach_limit_stage_units": round(reach_limit, 9),
        "analytic_reach_candidate": analytic_reach_candidate,
        "metric_scale_status": admitted["metric_scale_status"],
        "metric_reach_qualified": bool(
            analytic_reach_candidate
            and admitted["metric_scale_status"] == "validated"
            and admitted["collision_status"] == "qualified"
        ),
        "collision_status": admitted["collision_status"],
        "scene_frame_binding_digest": admitted["scene_frame_binding_digest"],
        "collision_candidate_digest": admitted["collision_candidate_digest"],
        "target_analysis_digest": admitted["target_analysis_digest"],
        "target_binding_digest": admitted["target_binding_digest"],
        "formal_gaps": [
            *(
                []
                if admitted["metric_scale_status"] == "validated"
                else ["independent_metric_scale_missing"]
            ),
            *(
                []
                if admitted["collision_status"] == "qualified"
                else ["live_collision_contact_and_full_footprint_not_qualified"]
            ),
            *([] if footprint_supported else ["robot_base_support_surface_missing"]),
            *(
                ["bounded_floor_proxy_requires_independent_qualification"]
                if floor_proxy is not None
                else []
            ),
            *(
                ["source_collision_conflicts_with_supported_robot_task_zone"]
                if source_collision_exclusion_required
                else []
            ),
            *([] if physics_probe_candidate is not None else ["supported_physics_probe_missing"]),
            *(
                ["placement_below_nominal_standoff_range"]
                if stance.standoff_m < profile.standoff_range_m[0]
                else []
            ),
            "fixed_base_support_mount_not_physically_qualified",
            "access_reset_and_human_clearance_not_qualified",
        ],
        "candidate_may_self_authorize": False,
        "physical_execution_authorized": False,
        "proof_effect": "external_scene_runtime_robot_visualization_candidate",
        "claim_ceiling": "analytic_robot_placement_candidate",
    }
    placement["placement_proposal_digest"] = canonical_digest(
        placement, digest_field="placement_proposal_digest"
    )
    render_options = {
        "robot_id": "franka_panda",
        "robot_usd": str(profile.simulator_asset_refs["isaac_asset"]).lstrip("/"),
        "robot_prim_path": profile.usd_prim_path,
        "robot_pose": placement["robot_pose_xyzyaw_collision_stage"],
        "robot_ground_z": placement["fixed_base_support_mount"]["top_z_collision_stage"],
        "fixed_base_support_mount": placement["fixed_base_support_mount"],
        "bounded_floor_proxy": placement["bounded_floor_proxy"],
        "proxy_composed_evaluation_plan": placement["proxy_composed_evaluation_plan"],
        "robot_only_pass": True,
        "robot_placement_digest": placement["placement_proposal_digest"],
        "placement_proposal_digest": placement["placement_proposal_digest"],
        "physics_probe_candidate": placement["physics_probe_candidate"],
        "lights_path": "/World/Lights",
    }
    selected_target = analysis.get("selected_target")
    task_family = (
        str(selected_target.get("task_family") or "").lower()
        if isinstance(selected_target, Mapping)
        else ""
    )
    if "inspection" in task_family:
        inspection_outcome_contract = build_franka_inspection_outcome_contract(
            target_analysis=analysis,
            placement_proposal_digest=placement["placement_proposal_digest"],
            target_position_stage=placement["target_position_collision_stage"],
            scene_frame_binding_digest=placement["scene_frame_binding_digest"],
        )
        render_options["inspection_outcome_contract"] = inspection_outcome_contract
        render_options["articulated_policy_trace_request"] = (
            build_franka_inspection_controller_cohort_request(
                robot_prim_path=profile.usd_prim_path,
                target_position_stage=placement["target_position_collision_stage"],
            )
        )
    else:
        render_options["articulated_policy_trace_request"] = (
            build_default_franka_policy_trace_request(robot_prim_path=profile.usd_prim_path)
        )
    render_options["render_options_digest"] = canonical_digest(
        render_options, digest_field="render_options_digest"
    )
    return {"placement": placement, "render_options": render_options}


def write_external_scene_robot_placement_packet(
    *,
    collision_glb_path: str | Path,
    request: Mapping[str, Any],
    target_analysis: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the digest-bound placement, metric contract, and Isaac options."""

    packet = propose_external_scene_robot_placement(
        collision_glb_path=collision_glb_path,
        request=request,
        target_analysis=target_analysis,
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "request.v1.json", build_external_scene_robot_placement_request(request))
    write_json(root / "result.v1.json", packet["placement"])
    write_json(
        root / "inspection_outcome_contract.v1.json",
        packet["render_options"]["inspection_outcome_contract"],
    )
    write_json(root / "render_options.json", packet["render_options"])
    return packet


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-glb", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--target-analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    def load(path: str) -> Mapping[str, Any]:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ExternalSceneRobotPlacementError(["external_placement_cli_input_invalid"])
        return value

    target_analysis = load(args.target_analysis)
    if target_analysis.get("schema_version") == "rendered_scene_task_target_orchestration.v1":
        nested = target_analysis.get("target_analysis")
        if not isinstance(nested, Mapping):
            raise ExternalSceneRobotPlacementError(
                ["external_placement_target_analysis_wrapper_invalid"]
            )
        target_analysis = nested
    packet = write_external_scene_robot_placement_packet(
        collision_glb_path=args.collision_glb,
        request=load(args.request),
        target_analysis=target_analysis,
        output_dir=args.output_dir,
    )
    print(json.dumps(packet, sort_keys=True))
    return 0


__all__ = [
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "ExternalSceneRobotPlacementError",
    "build_external_scene_robot_placement_request",
    "propose_external_scene_robot_placement",
    "write_external_scene_robot_placement_packet",
]


if __name__ == "__main__":
    raise SystemExit(main())
