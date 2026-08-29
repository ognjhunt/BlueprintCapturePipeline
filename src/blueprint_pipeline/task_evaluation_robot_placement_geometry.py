"""Deterministic geometry authority for task-aware robot placement proposals."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom

from .decision_evidence_contracts import canonical_digest
from .franka_kinematics import solve_world_position_ik
from .scene_placement.robot_profile import get_robot_profile


GEOMETRY_GATE_SCHEMA_VERSION = "task_evaluation_robot_placement_geometry_gate.v1"
GEOMETRY_SUMMARY_SCHEMA_VERSION = "task_evaluation_robot_placement_geometry_summary.v1"
TRAJECTORY_IK_GATE_SCHEMA_VERSION = (
    "task_evaluation_robot_placement_trajectory_position_ik_gate.v1"
)


class RobotPlacementGeometryError(ValueError):
    """The exact scene or robot geometry cannot support a placement decision."""


@dataclass(frozen=True)
class SupportSurface:
    surface_id: str
    prim_path: str
    height_m: float
    area_m2: float
    minimum_xy_m: tuple[float, float]
    maximum_xy_m: tuple[float, float]
    triangle_indices: tuple[int, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "prim_path": self.prim_path,
            "height_m": self.height_m,
            "area_m2": self.area_m2,
            "minimum_xy_m": list(self.minimum_xy_m),
            "maximum_xy_m": list(self.maximum_xy_m),
            "triangle_count": len(self.triangle_indices),
        }


@dataclass(frozen=True)
class RobotPlacementGeometryIndex:
    scene_path: str
    scene_digest: str
    robot_asset_path: str
    robot_asset_digest: str
    robot_triangles: np.ndarray
    triangles: np.ndarray
    triangle_prim_paths: tuple[str, ...]
    triangle_minimum: np.ndarray
    triangle_maximum: np.ndarray
    triangle_normal_abs_z: np.ndarray
    watertight_prim_bounds: tuple[
        tuple[str, tuple[float, float, float], tuple[float, float, float]], ...
    ]
    support_surfaces: tuple[SupportSurface, ...]
    robot_local_bounds_minimum_m: tuple[float, float, float]
    robot_local_bounds_maximum_m: tuple[float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _triangulate(counts: Sequence[int], indices: Sequence[int]) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        polygon = [int(value) for value in indices[cursor : cursor + int(count)]]
        cursor += int(count)
        for index in range(1, len(polygon) - 1):
            faces.append((polygon[0], polygon[index], polygon[index + 1]))
    return np.asarray(faces, dtype=np.int64)


def _stage_triangles(stage: Usd.Stage) -> tuple[np.ndarray, tuple[str, ...]]:
    triangles: list[np.ndarray] = []
    prim_paths: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        if len(points) < 3 or not counts or not indices:
            continue
        faces = _triangulate(counts, indices)
        if not len(faces):
            continue
        matrix = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        )
        homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
        transformed = homogeneous @ matrix
        world_points = transformed[:, :3] / transformed[:, 3, None]
        local_triangles = world_points[faces]
        triangles.append(local_triangles)
        prim_paths.extend([str(prim.GetPath())] * len(local_triangles))
    if not triangles:
        raise RobotPlacementGeometryError("robot_placement_scene_triangles_missing")
    return np.concatenate(triangles, axis=0), tuple(prim_paths)


def _robot_bounds(stage: Usd.Stage) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        roots = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
        if len(roots) != 1:
            raise RobotPlacementGeometryError("robot_placement_robot_default_prim_missing")
        default_prim = roots[0]
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bounds = cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise RobotPlacementGeometryError("robot_placement_robot_bounds_missing")
    minimum = tuple(float(value) for value in bounds.GetMin())
    maximum = tuple(float(value) for value in bounds.GetMax())
    return minimum, maximum


def _support_surfaces(
    triangles: np.ndarray,
    prim_paths: Sequence[str],
    *,
    height_bin_m: float = 0.01,
    minimum_area_m2: float = 0.015,
) -> tuple[SupportSurface, ...]:
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_a, edge_b)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1.0e-12
    normal_abs_z = np.zeros(len(triangles), dtype=np.float64)
    normal_abs_z[valid] = np.abs(cross[valid, 2]) / double_area[valid]
    centroid_z = triangles[:, :, 2].mean(axis=1)
    groups: dict[tuple[str, int], list[int]] = {}
    for index in np.flatnonzero(valid & (normal_abs_z >= 0.95)):
        key = (str(prim_paths[int(index)]), int(round(centroid_z[index] / height_bin_m)))
        groups.setdefault(key, []).append(int(index))
    surfaces: list[SupportSurface] = []
    for (prim_path, _height_bin), indices in groups.items():
        index_array = np.asarray(indices, dtype=np.int64)
        area = float((double_area[index_array] * 0.5).sum())
        if area < minimum_area_m2:
            continue
        selected = triangles[index_array]
        height = float(np.average(centroid_z[index_array], weights=double_area[index_array]))
        surface_id = f"{prim_path}@z={height:.4f}"
        surfaces.append(
            SupportSurface(
                surface_id=surface_id,
                prim_path=prim_path,
                height_m=round(height, 6),
                area_m2=round(area, 6),
                minimum_xy_m=tuple(float(value) for value in selected[:, :, :2].min(axis=(0, 1))),
                maximum_xy_m=tuple(float(value) for value in selected[:, :, :2].max(axis=(0, 1))),
                triangle_indices=tuple(indices),
            )
        )
    surfaces.sort(key=lambda surface: (-surface.area_m2, surface.height_m, surface.surface_id))
    if not surfaces:
        raise RobotPlacementGeometryError("robot_placement_support_surfaces_missing")
    return tuple(surfaces)


def _watertight_prim_bounds(
    triangles: np.ndarray, prim_paths: Sequence[str]
) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Identify closed triangle shells for conservative inside-volume checks."""

    result = []
    for prim_path in sorted(set(prim_paths)):
        indices = np.asarray(
            [index for index, value in enumerate(prim_paths) if value == prim_path],
            dtype=np.int64,
        )
        selected = triangles[indices]
        if len(selected) < 4:
            continue
        edge_counts: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
        for triangle in selected:
            vertices = [tuple(np.round(vertex, 7)) for vertex in triangle]
            for first, second in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((vertices[first], vertices[second])))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        if edge_counts and all(count == 2 for count in edge_counts.values()):
            result.append(
                (
                    prim_path,
                    tuple(float(value) for value in selected.min(axis=(0, 1))),
                    tuple(float(value) for value in selected.max(axis=(0, 1))),
                )
            )
    return tuple(result)


def build_robot_placement_geometry_index(
    *, scene_collision_usd_path: str | Path, robot_asset_usd_path: str | Path
) -> RobotPlacementGeometryIndex:
    scene_path = Path(scene_collision_usd_path).expanduser().resolve(strict=True)
    robot_path = Path(robot_asset_usd_path).expanduser().resolve(strict=True)
    scene_stage = Usd.Stage.Open(str(scene_path))
    robot_stage = Usd.Stage.Open(str(robot_path))
    if scene_stage is None:
        raise RobotPlacementGeometryError("robot_placement_scene_usd_invalid")
    if robot_stage is None:
        raise RobotPlacementGeometryError("robot_placement_robot_usd_invalid")
    triangles, prim_paths = _stage_triangles(scene_stage)
    minimum = triangles.min(axis=1)
    maximum = triangles.max(axis=1)
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edges_a, edges_b)
    norm = np.linalg.norm(cross, axis=1)
    normal_abs_z = np.zeros(len(triangles), dtype=np.float64)
    valid = norm > 1.0e-12
    normal_abs_z[valid] = np.abs(cross[valid, 2]) / norm[valid]
    robot_minimum, robot_maximum = _robot_bounds(robot_stage)
    robot_triangles, _robot_prim_paths = _stage_triangles(robot_stage)
    robot_bounds_minimum = np.asarray(robot_minimum, dtype=np.float64) - 1.0e-4
    robot_bounds_maximum = np.asarray(robot_maximum, dtype=np.float64) + 1.0e-4
    inside_robot_bounds = np.all(
        (robot_triangles >= robot_bounds_minimum)
        & (robot_triangles <= robot_bounds_maximum),
        axis=(1, 2),
    )
    robot_triangles = robot_triangles[inside_robot_bounds]
    if not len(robot_triangles):
        raise RobotPlacementGeometryError("robot_placement_robot_preview_triangles_missing")
    return RobotPlacementGeometryIndex(
        scene_path=str(scene_path),
        scene_digest=_sha256(scene_path),
        robot_asset_path=str(robot_path),
        robot_asset_digest=_sha256(robot_path),
        robot_triangles=robot_triangles,
        triangles=triangles,
        triangle_prim_paths=prim_paths,
        triangle_minimum=minimum,
        triangle_maximum=maximum,
        triangle_normal_abs_z=normal_abs_z,
        watertight_prim_bounds=_watertight_prim_bounds(triangles, prim_paths),
        support_surfaces=_support_surfaces(triangles, prim_paths),
        robot_local_bounds_minimum_m=robot_minimum,
        robot_local_bounds_maximum_m=robot_maximum,
    )


def summarize_robot_placement_geometry(
    index: RobotPlacementGeometryIndex,
    *, target_position_world_m: Sequence[float],
    robot_id: str = "franka_panda",
) -> dict[str, Any]:
    if len(target_position_world_m) != 3:
        raise RobotPlacementGeometryError("robot_placement_target_invalid")
    target = [float(value) for value in target_position_world_m]
    if not all(math.isfinite(value) for value in target):
        raise RobotPlacementGeometryError("robot_placement_target_invalid")
    profile = get_robot_profile(robot_id)
    result: dict[str, Any] = {
        "schema_version": GEOMETRY_SUMMARY_SCHEMA_VERSION,
        "scene_collision_digest": index.scene_digest,
        "robot_asset_digest": index.robot_asset_digest,
        "robot_id": robot_id,
        "target_position_world_m": target,
        "scene_bounds_m": {
            "minimum": [float(value) for value in index.triangles.min(axis=(0, 1))],
            "maximum": [float(value) for value in index.triangles.max(axis=(0, 1))],
        },
        "robot_local_bounds_m": {
            "minimum": list(index.robot_local_bounds_minimum_m),
            "maximum": list(index.robot_local_bounds_maximum_m),
        },
        "robot_base_support_half_extents_xy_m": list(
            profile.footprint_half_extent_xyz[:2]
        ),
        "robot_shoulder_above_root_m": profile.shoulder_above_root_m,
        "maximum_shoulder_to_target_m": profile.max_shoulder_to_affordance_m(),
        "maximum_facing_error_degrees": profile.max_facing_error_deg,
        "support_surfaces": [surface.to_mapping() for surface in index.support_surfaces[:40]],
        "watertight_scene_prim_count": len(index.watertight_prim_bounds),
        "geometry_summary_digest": "",
    }
    result["geometry_summary_digest"] = canonical_digest(
        result, digest_field="geometry_summary_digest"
    )
    return result


def _point_in_triangle_xy(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = triangle[:, :2]
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) <= 1.0e-15:
        return False
    inverse = 1.0 / denominator
    u = (dot11 * dot02 - dot01 * dot12) * inverse
    v = (dot00 * dot12 - dot01 * dot02) * inverse
    return u >= -1.0e-8 and v >= -1.0e-8 and u + v <= 1.0 + 1.0e-8


def _yaw_from_quaternion(quaternion: Sequence[float]) -> tuple[float, bool]:
    x, y, z, w = (float(value) for value in quaternion)
    upright = abs(x) <= 1.0e-4 and abs(y) <= 1.0e-4
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw, upright


def _rotated_bounds(
    minimum: Sequence[float], maximum: Sequence[float], position: np.ndarray, yaw: float
) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    world = corners @ rotation.T + position
    return world.min(axis=0), world.max(axis=0)


def validate_robot_placement_geometry_candidate(
    *,
    index: RobotPlacementGeometryIndex,
    proposal: Mapping[str, Any],
    target_position_world_m: Sequence[float],
    robot_id: str = "franka_panda",
    support_height_tolerance_m: float = 0.02,
    trajectory_waypoints_world_m: Sequence[Sequence[float]] = (),
    trajectory_phase_ids: Sequence[str] = (),
    trajectory_orientations_world_xyzw: Sequence[Sequence[float]] = (),
    trajectory_gate_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = str(proposal.get("candidate_id") or "")
    pose = proposal.get("pose") if isinstance(proposal.get("pose"), Mapping) else {}
    try:
        position = np.asarray(pose.get("position_world_m"), dtype=np.float64)
        quaternion = [float(value) for value in pose.get("orientation_xyzw")]
        target = np.asarray(target_position_world_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RobotPlacementGeometryError("robot_placement_candidate_pose_invalid") from exc
    if position.shape != (3,) or target.shape != (3,) or len(quaternion) != 4:
        raise RobotPlacementGeometryError("robot_placement_candidate_pose_invalid")
    surface_id = str(proposal.get("support_surface_id") or "")
    surface = next(
        (surface for surface in index.support_surfaces if surface.surface_id == surface_id),
        None,
    )
    blockers: list[str] = []
    yaw, upright = _yaw_from_quaternion(quaternion)
    if not upright:
        blockers.append("robot_base_not_upright")
    profile = get_robot_profile(robot_id)
    half_x, half_y = profile.footprint_half_extent_xyz[:2]
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    local_samples = np.asarray(
        [[0.0, 0.0], [-half_x, -half_y], [-half_x, half_y], [half_x, -half_y], [half_x, half_y]],
        dtype=np.float64,
    )
    samples = local_samples @ rotation.T + position[:2]
    supported_count = 0
    support_height_error_m: float | None = None
    if surface is None:
        blockers.append("declared_support_surface_missing")
    else:
        selected = index.triangles[np.asarray(surface.triangle_indices, dtype=np.int64)]
        support_height_error_m = abs(float(position[2]) - surface.height_m)
        for sample in samples:
            if any(_point_in_triangle_xy(sample, triangle) for triangle in selected):
                supported_count += 1
        if support_height_error_m > support_height_tolerance_m:
            blockers.append("robot_root_not_on_declared_support_height")
        if supported_count != len(samples):
            blockers.append("robot_base_support_coverage_incomplete")

    robot_minimum, robot_maximum = _rotated_bounds(
        index.robot_local_bounds_minimum_m,
        index.robot_local_bounds_maximum_m,
        position,
        yaw,
    )
    overlap = (
        (index.triangle_maximum[:, 0] >= robot_minimum[0])
        & (index.triangle_minimum[:, 0] <= robot_maximum[0])
        & (index.triangle_maximum[:, 1] >= robot_minimum[1])
        & (index.triangle_minimum[:, 1] <= robot_maximum[1])
        & (index.triangle_maximum[:, 2] >= robot_minimum[2])
        & (index.triangle_minimum[:, 2] <= robot_maximum[2])
    )
    # Contact with the declared horizontal support at the root plane is allowed;
    # all scene geometry above that plane remains a collision blocker.
    allowed_support_contact = index.triangle_maximum[:, 2] <= (
        position[2] + support_height_tolerance_m
    )
    collision_indices = np.flatnonzero(overlap & ~allowed_support_contact)
    robot_center = 0.5 * (robot_minimum + robot_maximum)
    containing_prim_paths = []
    containment_tolerance_m = 1.0e-4
    for prim_path, minimum, maximum in index.watertight_prim_bounds:
        prim_minimum = np.asarray(minimum, dtype=np.float64)
        prim_maximum = np.asarray(maximum, dtype=np.float64)
        if np.all(robot_center > prim_minimum + containment_tolerance_m) and np.all(
            robot_center < prim_maximum - containment_tolerance_m
        ):
            containing_prim_paths.append(prim_path)
    if len(collision_indices) or containing_prim_paths:
        blockers.append("robot_reset_bounds_overlap_scene_geometry")

    shoulder = position + np.asarray([0.0, 0.0, profile.shoulder_above_root_m])
    shoulder_distance = float(np.linalg.norm(target - shoulder))
    reach_limit = float(profile.max_shoulder_to_affordance_m())
    if shoulder_distance > reach_limit:
        blockers.append("task_target_outside_analytic_reach")
    target_delta = target[:2] - position[:2]
    target_distance_xy = float(np.linalg.norm(target_delta))
    if target_distance_xy <= max(half_x, half_y) + profile.probe_clearance_m:
        blockers.append("task_target_inside_robot_base_keepout")
    target_yaw = math.atan2(float(target_delta[1]), float(target_delta[0]))
    facing_error = abs(math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)))
    facing_error_degrees = math.degrees(facing_error)
    if facing_error_degrees > profile.max_facing_error_deg:
        blockers.append("robot_not_facing_task_workspace")

    trajectory_gate = (
        dict(trajectory_gate_override)
        if trajectory_gate_override is not None
        else validate_robot_placement_trajectory_position_ik(
            proposal=proposal,
            trajectory_waypoints_world_m=trajectory_waypoints_world_m,
            trajectory_phase_ids=trajectory_phase_ids,
            trajectory_orientations_world_xyzw=(
                trajectory_orientations_world_xyzw
            ),
        )
    )
    if (
        trajectory_gate.get("schema_version")
        != TRAJECTORY_IK_GATE_SCHEMA_VERSION
        or trajectory_gate.get("trajectory_position_ik_gate_digest")
        != canonical_digest(
            trajectory_gate, digest_field="trajectory_position_ik_gate_digest"
        )
    ):
        raise RobotPlacementGeometryError(
            "robot_placement_trajectory_gate_override_invalid"
        )
    blockers.extend(trajectory_gate["blockers"])

    result: dict[str, Any] = {
        "schema_version": GEOMETRY_GATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "status": "passed" if not blockers else "rejected",
        "blockers": sorted(set(blockers)),
        "support_passed": bool(
            surface is not None
            and support_height_error_m is not None
            and support_height_error_m <= support_height_tolerance_m
            and supported_count == len(samples)
        ),
        "collision_passed": len(collision_indices) == 0 and not containing_prim_paths,
        "reachability_passed": bool(
            shoulder_distance <= reach_limit
            and trajectory_gate["all_waypoints_position_ik_solved"]
        ),
        "facing_passed": facing_error_degrees <= profile.max_facing_error_deg,
        "declared_support_surface_id": surface_id,
        "support_sample_count": len(samples),
        "supported_sample_count": supported_count,
        "support_height_error_m": support_height_error_m,
        "robot_reset_world_aabb_m": {
            "minimum": [float(value) for value in robot_minimum],
            "maximum": [float(value) for value in robot_maximum],
        },
        "scene_overlap_triangle_count": int(len(collision_indices)),
        "scene_overlap_prim_paths": sorted(
            {
                *(index.triangle_prim_paths[int(value)] for value in collision_indices),
                *containing_prim_paths,
            }
        )[:30],
        "robot_center_inside_watertight_prim_paths": sorted(containing_prim_paths)[:30],
        "shoulder_to_target_distance_m": shoulder_distance,
        "shoulder_to_target_limit_m": reach_limit,
        "facing_error_degrees": facing_error_degrees,
        "trajectory_position_ik_gate": trajectory_gate,
        "native_reset_contact_and_ik_still_required": True,
        "geometry_gate_digest": "",
    }
    result["geometry_gate_digest"] = canonical_digest(
        result, digest_field="geometry_gate_digest"
    )
    return result


def _trajectory_gate_worker(
    work: tuple[
        Mapping[str, Any],
        Sequence[Sequence[float]],
        Sequence[str],
        Sequence[Sequence[float]],
    ],
) -> dict[str, Any]:
    proposal, waypoints, phase_ids, orientations = work
    return validate_robot_placement_trajectory_position_ik(
        proposal=proposal,
        trajectory_waypoints_world_m=waypoints,
        trajectory_phase_ids=phase_ids,
        trajectory_orientations_world_xyzw=orientations,
    )


def _parallel_trajectory_gates(
    *,
    proposals: Sequence[Mapping[str, Any]],
    trajectory_waypoints_world_m: Sequence[Sequence[float]],
    trajectory_phase_ids: Sequence[str],
    trajectory_orientations_world_xyzw: Sequence[Sequence[float]],
    worker_count: int,
) -> list[dict[str, Any]]:
    """Evaluate independent base candidates concurrently, retaining order."""

    try:
        workers = int(worker_count)
    except (TypeError, ValueError) as exc:
        raise RobotPlacementGeometryError(
            "robot_placement_trajectory_worker_count_invalid"
        ) from exc
    if not 1 <= workers <= 32:
        raise RobotPlacementGeometryError(
            "robot_placement_trajectory_worker_count_invalid"
        )
    work = [
        (
            dict(proposal),
            [list(row) for row in trajectory_waypoints_world_m],
            list(trajectory_phase_ids),
            [list(row) for row in trajectory_orientations_world_xyzw],
        )
        for proposal in proposals
    ]
    if workers == 1 or len(work) <= 1:
        return [_trajectory_gate_worker(row) for row in work]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(work)), mp_context=context
    ) as executor:
        return list(executor.map(_trajectory_gate_worker, work, chunksize=4))


def validate_robot_placement_trajectory_position_ik(
    *,
    proposal: Mapping[str, Any],
    trajectory_waypoints_world_m: Sequence[Sequence[float]],
    trajectory_phase_ids: Sequence[str] = (),
    trajectory_orientations_world_xyzw: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    """Filter every trajectory point without claiming native pose qualification."""

    pose = proposal.get("pose") if isinstance(proposal.get("pose"), Mapping) else {}
    try:
        position = [float(value) for value in pose.get("position_world_m")]
        orientation = [float(value) for value in pose.get("orientation_xyzw")]
        waypoints = [
            [float(value) for value in waypoint]
            for waypoint in trajectory_waypoints_world_m
        ]
        phase_ids = [str(value) for value in trajectory_phase_ids]
        phase_orientations = [
            [float(value) for value in orientation]
            for orientation in trajectory_orientations_world_xyzw
        ]
    except (TypeError, ValueError) as exc:
        raise RobotPlacementGeometryError(
            "robot_placement_trajectory_waypoints_invalid"
        ) from exc
    rows_to_check = [position, orientation, *waypoints]
    if (
        len(position) != 3
        or len(orientation) != 4
        or any(len(waypoint) != 3 for waypoint in waypoints)
        or (phase_ids and len(phase_ids) != len(waypoints))
        or any(not value for value in phase_ids)
        or len(set(phase_ids)) != len(phase_ids)
        or (phase_orientations and len(phase_orientations) != len(waypoints))
        or any(len(orientation) != 4 for orientation in phase_orientations)
        or not all(
            math.isfinite(value)
            for row in [*rows_to_check, *phase_orientations]
            for value in row
        )
    ):
        raise RobotPlacementGeometryError(
            "robot_placement_trajectory_waypoints_invalid"
        )

    seed: list[float] | None = None
    rows: list[dict[str, Any]] = []
    for waypoint_index, waypoint in enumerate(waypoints):
        seed_used = list(seed) if seed is not None else None
        solved = solve_world_position_ik(
            target_position_world_m=waypoint,
            robot_base_position_world_m=position,
            robot_base_quaternion_world_xyzw=orientation,
            seed_joint_positions=seed,
            quaternion_world_xyzw=(
                phase_orientations[waypoint_index]
                if phase_orientations
                else None
            ),
        )
        if solved["solved"] is True:
            seed = list(solved["joint_positions"])
        rows.append(
            {
                "waypoint_index": waypoint_index,
                "phase_id": (
                    phase_ids[waypoint_index]
                    if phase_ids
                    else f"waypoint_{waypoint_index:02d}"
                ),
                "position_world_m": waypoint,
                "orientation_world_xyzw": (
                    phase_orientations[waypoint_index]
                    if phase_orientations
                    else None
                ),
                "seed_joint_positions": seed_used,
                "position_ik_solved": solved["solved"] is True,
                "position_error_m": float(solved["position_error_m"]),
                "manipulability": float(solved["manipulability"]),
                "solved_joint_positions": [
                    float(value) for value in solved["joint_positions"]
                ],
            }
        )
    all_solved = all(row["position_ik_solved"] for row in rows)
    blockers = [] if all_solved else ["robot_trajectory_position_ik_unreached"]
    result: dict[str, Any] = {
        "schema_version": TRAJECTORY_IK_GATE_SCHEMA_VERSION,
        "status": (
            "passed" if rows and all_solved else "not_requested" if not rows else "rejected"
        ),
        "blockers": blockers,
        "waypoint_count": len(rows),
        "all_waypoints_position_ik_solved": all_solved,
        "minimum_manipulability": (
            min(row["manipulability"] for row in rows) if rows else None
        ),
        "maximum_position_error_m": (
            max(row["position_error_m"] for row in rows) if rows else None
        ),
        "waypoints": rows,
        "orientation_ik_solved": False,
        "native_orientation_ik_required": True,
        "native_collision_contact_required": True,
        "trajectory_position_ik_gate_digest": "",
    }
    result["trajectory_position_ik_gate_digest"] = canonical_digest(
        result, digest_field="trajectory_position_ik_gate_digest"
    )
    return result


def enumerate_robot_placement_geometry_candidates(
    *,
    index: RobotPlacementGeometryIndex,
    target_position_world_m: Sequence[float],
    robot_id: str = "franka_panda",
    maximum_candidates: int = 48,
    trajectory_waypoints_world_m: Sequence[Sequence[float]] = (),
    trajectory_phase_ids: Sequence[str] = (),
    trajectory_orientations_world_xyzw: Sequence[Sequence[float]] = (),
    trajectory_worker_count: int = 1,
) -> list[dict[str, Any]]:
    """Return ranked, deterministic, geometry-passing candidates for agent context."""

    if not 1 <= int(maximum_candidates) <= 256:
        raise RobotPlacementGeometryError("robot_placement_candidate_cap_invalid")
    target = np.asarray(target_position_world_m, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise RobotPlacementGeometryError("robot_placement_target_invalid")
    profile = get_robot_profile(robot_id)
    surfaces = [
        surface
        for surface in index.support_surfaces
        if surface.height_m <= target[2] + 0.10
        and target[2] - surface.height_m <= profile.max_shoulder_to_affordance_m()
        and surface.minimum_xy_m[0] - profile.standoff_range_m[1] <= target[0]
        <= surface.maximum_xy_m[0] + profile.standoff_range_m[1]
        and surface.minimum_xy_m[1] - profile.standoff_range_m[1] <= target[1]
        <= surface.maximum_xy_m[1] + profile.standoff_range_m[1]
    ]
    surfaces.sort(
        key=lambda surface: (
            abs(surface.height_m - target[2]),
            -surface.area_m2,
            surface.surface_id,
        )
    )
    minimum_radius, maximum_radius = profile.standoff_range_m
    radii = sorted(
        np.arange(minimum_radius, maximum_radius + 1.0e-9, profile.probe_step_m),
        key=lambda value: (abs(float(value) - 0.45), float(value)),
    )
    prequalified: list[
        tuple[dict[str, Any], dict[str, Any], SupportSurface, float, int]
    ] = []
    for surface_index, surface in enumerate(surfaces[:12]):
        for radius_index, radius in enumerate(radii):
            for angle_index in range(72):
                angle = 2.0 * math.pi * angle_index / 72.0
                position = [
                    float(target[0] - radius * math.cos(angle)),
                    float(target[1] - radius * math.sin(angle)),
                    float(surface.height_m),
                ]
                proposal = {
                    "candidate_id": (
                        f"geometry_{surface_index:02d}_{radius_index:02d}_"
                        f"{surface.height_m:.3f}_"
                        f"{float(radius):.3f}_{angle_index:02d}"
                    ),
                    "support_surface_id": surface.surface_id,
                    "pose": {
                        "position_world_m": position,
                        "orientation_xyzw": [
                            0.0,
                            0.0,
                            math.sin(angle / 2.0),
                            math.cos(angle / 2.0),
                        ],
                    },
                }
                gate = validate_robot_placement_geometry_candidate(
                    index=index,
                    proposal=proposal,
                    target_position_world_m=target,
                    robot_id=robot_id,
                )
                if gate["status"] != "passed":
                    continue
                prequalified.append(
                    (proposal, gate, surface, float(radius), angle_index)
                )

    trajectory_gates = _parallel_trajectory_gates(
        proposals=[row[0] for row in prequalified],
        trajectory_waypoints_world_m=trajectory_waypoints_world_m,
        trajectory_phase_ids=trajectory_phase_ids,
        trajectory_orientations_world_xyzw=trajectory_orientations_world_xyzw,
        worker_count=trajectory_worker_count,
    )
    candidates: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    for (
        proposal,
        _cheap_gate,
        surface,
        radius,
        angle_index,
    ), trajectory_gate in zip(prequalified, trajectory_gates, strict=True):
        gate = validate_robot_placement_geometry_candidate(
            index=index,
            proposal=proposal,
            target_position_world_m=target,
            robot_id=robot_id,
            trajectory_gate_override=trajectory_gate,
        )
        if gate["status"] != "passed":
            continue
        score = (
            -float(
                gate["trajectory_position_ik_gate"]["minimum_manipulability"]
                or 0.0
            ),
            abs(surface.height_m + profile.shoulder_above_root_m - target[2]),
            abs(radius - 0.45),
            gate["shoulder_to_target_distance_m"],
            float(angle_index),
        )
        candidates.append((score, proposal, gate))
    candidates.sort(key=lambda item: item[0])
    return [
        {
            **proposal,
            "geometry_gate_digest": gate["geometry_gate_digest"],
            "shoulder_to_target_distance_m": gate["shoulder_to_target_distance_m"],
            "trajectory_position_ik_gate_digest": gate[
                "trajectory_position_ik_gate"
            ]["trajectory_position_ik_gate_digest"],
            "trajectory_minimum_manipulability": gate[
                "trajectory_position_ik_gate"
            ]["minimum_manipulability"],
            "trajectory_position_ik_gate": gate["trajectory_position_ik_gate"],
        }
        for _score, proposal, gate in candidates[: int(maximum_candidates)]
    ]


def render_robot_placement_geometry_previews(
    *,
    index: RobotPlacementGeometryIndex,
    proposal: Mapping[str, Any],
    target_position_world_m: Sequence[float],
    trajectory_waypoints_world_m: Sequence[Sequence[float]] = (),
    image_size: tuple[int, int] = (1000, 720),
) -> list[dict[str, Any]]:
    """Render digest-bound top and side geometry views without a paid GPU."""

    pose = proposal.get("pose") if isinstance(proposal.get("pose"), Mapping) else {}
    position = np.asarray(pose.get("position_world_m"), dtype=np.float64)
    quaternion = [float(value) for value in pose.get("orientation_xyzw")]
    target = np.asarray(target_position_world_m, dtype=np.float64)
    trajectory = np.asarray(trajectory_waypoints_world_m, dtype=np.float64)
    if trajectory.size == 0:
        trajectory = target.reshape(1, 3)
    if (
        position.shape != (3,)
        or target.shape != (3,)
        or trajectory.ndim != 2
        or trajectory.shape[1] != 3
        or not np.all(np.isfinite(trajectory))
        or len(quaternion) != 4
    ):
        raise RobotPlacementGeometryError("robot_placement_preview_pose_invalid")
    yaw, _ = _yaw_from_quaternion(quaternion)
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    robot_world_triangles = index.robot_triangles @ rotation.T + position
    robot_minimum, robot_maximum = _rotated_bounds(
        index.robot_local_bounds_minimum_m,
        index.robot_local_bounds_maximum_m,
        position,
        yaw,
    )
    def draw_projection(axes: tuple[int, int], label: str) -> dict[str, Any]:
        width, height = image_size
        margin = 55
        low = np.minimum(
            robot_world_triangles[:, :, list(axes)].min(axis=(0, 1)),
            trajectory[:, list(axes)].min(axis=0),
        )
        high = np.maximum(
            robot_world_triangles[:, :, list(axes)].max(axis=(0, 1)),
            trajectory[:, list(axes)].max(axis=0),
        )
        padding = np.maximum((high - low) * 0.18, 0.18)
        low -= padding
        high += padding
        span = np.maximum(high - low, 1.0e-6)

        def point(value: Sequence[float]) -> tuple[int, int]:
            normalized = (np.asarray(value, dtype=np.float64) - low) / span
            return (
                int(margin + normalized[0] * (width - 2 * margin)),
                int(height - margin - normalized[1] * (height - 2 * margin)),
            )

        image = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(image, "RGBA")
        selected_surface_id = str(proposal.get("support_surface_id") or "")
        selected_indices: set[int] = set()
        surface = next(
            (surface for surface in index.support_surfaces if surface.surface_id == selected_surface_id),
            None,
        )
        if surface is not None:
            selected_indices = set(surface.triangle_indices)
        projected_minimum = index.triangles[:, :, list(axes)].min(axis=1)
        projected_maximum = index.triangles[:, :, list(axes)].max(axis=1)
        nearby = np.flatnonzero(
            (projected_maximum[:, 0] >= low[0])
            & (projected_minimum[:, 0] <= high[0])
            & (projected_maximum[:, 1] >= low[1])
            & (projected_minimum[:, 1] <= high[1])
        )
        stride = max(1, len(nearby) // 8_000)
        for triangle_index in nearby[::stride]:
            triangle = index.triangles[triangle_index][:, list(axes)]
            colour = (
                (55, 130, 230, 95)
                if int(triangle_index) in selected_indices
                else (105, 105, 105, 32)
            )
            draw.polygon(
                [point(value) for value in triangle],
                fill=colour,
                outline=(95, 95, 95, 45),
            )
        robot_depth_axis = next(axis for axis in range(3) if axis not in axes)
        robot_order = np.argsort(
            robot_world_triangles[:, :, robot_depth_axis].mean(axis=1)
        )
        for triangle_index in robot_order:
            triangle = robot_world_triangles[int(triangle_index)][:, list(axes)]
            draw.polygon(
                [point(value) for value in triangle],
                fill=(220, 45, 45, 205),
                outline=(105, 0, 0, 210),
            )
        rectangle_min = point(robot_minimum[list(axes)])
        rectangle_max = point(robot_maximum[list(axes)])
        draw.rectangle(
            [
                min(rectangle_min[0], rectangle_max[0]),
                min(rectangle_min[1], rectangle_max[1]),
                max(rectangle_min[0], rectangle_max[0]),
                max(rectangle_min[1], rectangle_max[1]),
            ],
            fill=None,
            outline=(120, 0, 0, 220),
            width=2,
        )
        base_point = point(position[list(axes)])
        draw.ellipse(
            [base_point[0] - 6, base_point[1] - 6, base_point[0] + 6, base_point[1] + 6],
            fill=(130, 0, 0, 255),
        )
        facing_world = position + np.asarray(
            [0.30 * math.cos(yaw), 0.30 * math.sin(yaw), 0.0],
            dtype=np.float64,
        )
        facing_point = point(facing_world[list(axes)])
        draw.line([base_point, facing_point], fill=(255, 145, 0, 255), width=6)
        target_point = point(target[list(axes)])
        radius = 9
        draw.ellipse(
            [target_point[0] - radius, target_point[1] - radius, target_point[0] + radius, target_point[1] + radius],
            fill=(0, 170, 70, 255),
            outline=(0, 90, 40, 255),
            width=2,
        )
        trajectory_points = [point(row[list(axes)]) for row in trajectory]
        if len(trajectory_points) > 1:
            draw.line(trajectory_points, fill=(0, 120, 210, 255), width=5)
        for waypoint in trajectory_points:
            draw.ellipse(
                [
                    waypoint[0] - 4,
                    waypoint[1] - 4,
                    waypoint[0] + 4,
                    waypoint[1] + 4,
                ],
                fill=(0, 185, 235, 255),
                outline=(0, 70, 130, 255),
            )
        draw.text(
            (18, 15),
            (
                f"{label}: solid red=robot mesh, dark red=reset bounds, "
                "orange=facing, green=task target, cyan=tool trajectory, blue=support"
            ),
            fill=(0, 0, 0, 255),
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        payload = buffer.getvalue()
        return {
            "label": label,
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "image_url": "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
            "detail": "high",
        }

    return [draw_projection((0, 1), "top_down_xy"), draw_projection((0, 2), "side_xz")]


__all__ = [
    "RobotPlacementGeometryError",
    "RobotPlacementGeometryIndex",
    "build_robot_placement_geometry_index",
    "enumerate_robot_placement_geometry_candidates",
    "render_robot_placement_geometry_previews",
    "summarize_robot_placement_geometry",
    "validate_robot_placement_geometry_candidate",
]
