"""Deterministic geometry authority for task-aware robot placement proposals."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial import cKDTree

from .decision_evidence_contracts import canonical_digest
from .franka_kinematics import solve_world_position_ik
from .scene_physics_preflight import (
    ScenePhysicsPreflightError,
    robot_fixed_volume_extent,
    robot_fixed_volume_mask,
    task_motion_surface,
)
from .robot_arm_path_clearance import RobotArmPathClearanceError, arm_path_clearance
from .scene_placement.robot_profile import get_robot_profile


GEOMETRY_GATE_SCHEMA_VERSION = "task_evaluation_robot_placement_geometry_gate.v1"
GEOMETRY_SUMMARY_SCHEMA_VERSION = "task_evaluation_robot_placement_geometry_summary.v1"
TRAJECTORY_IK_GATE_SCHEMA_VERSION = (
    "task_evaluation_robot_placement_trajectory_position_ik_gate.v1"
)
# A placed robot's fixed volumes keep this far from every pose the task's moving
# part takes, which covers the 2 cm spacing of the swept samples and keeps an
# accepted pose clear of the paid-run sweep.
TASK_MOTION_CLEARANCE_M = 0.02


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
    background_collision_digest: str
    task_asset_bindings: tuple[Mapping[str, Any], ...]
    # Surface samples of every pose a task asset's moving part takes between
    # reset and its success threshold.
    task_motion_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The same part across its whole joint range: where it can go if a policy
    # opens it further or lets it fall.
    task_limit_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The whole range of each part that does not hold where it is released:
    # gravity carries it there whatever a policy does.
    task_fall_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


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


def _stage_triangles(stage: Usd.Stage, *, root_prim: Usd.Prim | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    triangles: list[np.ndarray] = []
    prim_paths: list[str] = []
    prims = Usd.PrimRange(
        root_prim if root_prim is not None else stage.GetPseudoRoot(),
        Usd.TraverseInstanceProxies(),
    )
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
            counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
            indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
            if len(points) < 3 or not counts or not indices:
                continue
            faces = _triangulate(counts, indices)
        elif any(prim.IsA(schema) for schema in (
            UsdGeom.Cube, UsdGeom.Cylinder, UsdGeom.Capsule,
            UsdGeom.Cone, UsdGeom.Sphere,
        )):
            # USD/PhysX analytic shapes are geometry too. Use their computed
            # local envelope, then the exact USD transform. This is exact for
            # cubes and conservative for curved shapes; native contact remains
            # required. Never silently omit an authored rack, panel or mount.
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
                UsdGeom.Boundable(prim), Usd.TimeCode.Default()
            )
            if extent is None or len(extent) != 2:
                raise RobotPlacementGeometryError("robot_placement_primitive_extent_missing")
            low, high = np.asarray(extent, dtype=np.float64)
            if not np.all(np.isfinite([low, high])) or np.any(high <= low):
                raise RobotPlacementGeometryError("robot_placement_primitive_extent_invalid")
            points = np.asarray([
                [low[0], low[1], low[2]], [high[0], low[1], low[2]],
                [high[0], high[1], low[2]], [low[0], high[1], low[2]],
                [low[0], low[1], high[2]], [high[0], low[1], high[2]],
                [high[0], high[1], high[2]], [low[0], high[1], high[2]],
            ])
            faces = _triangulate(
                [4] * 6, [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4,
                          1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7],
            )
        else:
            continue
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


def _robot_root(stage: Usd.Stage) -> Usd.Prim:
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        roots = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
        if len(roots) != 1:
            raise RobotPlacementGeometryError("robot_placement_robot_default_prim_missing")
        default_prim = roots[0]
    return default_prim


def _robot_bounds(stage: Usd.Stage) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    default_prim = _robot_root(stage)
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
    grouped_indices: dict[str, list[int]] = {}
    for index, prim_path in enumerate(prim_paths):
        grouped_indices.setdefault(prim_path, []).append(index)
    for prim_path, group in sorted(grouped_indices.items()):
        indices = np.asarray(group, dtype=np.int64)
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


def _placed_task_asset_triangles(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[np.ndarray], list[str], tuple[Mapping[str, Any], ...]]:
    triangles: list[np.ndarray] = []
    paths: list[str] = []
    bindings: list[Mapping[str, Any]] = []
    asset_ids: set[str] = set()
    for ordinal, row in enumerate(rows):
        asset_id = str(row.get("asset_id") or "")
        if not asset_id or asset_id in asset_ids:
            raise RobotPlacementGeometryError("robot_placement_task_asset_id_invalid")
        asset_ids.add(asset_id)
        asset_path = Path(str(row.get("asset_usd_path") or "")).expanduser().resolve(strict=True)
        asset_digest = _sha256(asset_path)
        if row.get("asset_digest") != asset_digest:
            raise RobotPlacementGeometryError("robot_placement_task_asset_digest_mismatch")
        pose = row.get("pose")
        if not isinstance(pose, Mapping):
            raise RobotPlacementGeometryError("robot_placement_task_asset_pose_invalid")
        position = np.asarray(pose.get("position_world_m"), dtype=np.float64)
        quaternion = np.asarray(pose.get("orientation_xyzw"), dtype=np.float64)
        if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion))
            or abs(float(np.linalg.norm(quaternion)) - 1.0) > 1.0e-5):
            raise RobotPlacementGeometryError("robot_placement_task_asset_pose_invalid")
        stage = Usd.Stage.CreateInMemory()
        actor = UsdGeom.Xform.Define(stage, f"/TaskActors/Actor_{ordinal:04d}")
        source = UsdGeom.Xform.Define(stage, f"{actor.GetPath()}/Asset")
        if not source.GetPrim().GetReferences().AddReference(str(asset_path)):
            raise RobotPlacementGeometryError("robot_placement_task_asset_reference_invalid")
        actor.AddTranslateOp().Set(Gf.Vec3d(*position))
        actor.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(quaternion[3]), Gf.Vec3d(*quaternion[:3]))
        )
        actor_triangles, actor_paths = _stage_triangles(stage)
        triangles.append(actor_triangles)
        paths.extend(actor_paths)
        # Hash each loaded external USD layer as well as package containers.
        # A root layer digest alone does not bind referenced collision meshes.
        layer_paths = {asset_path}
        for layer in stage.GetUsedLayers():
            if not layer.anonymous:
                layer_paths.add(Path(layer.realPath.split("[", 1)[0]).resolve(strict=True))
        binding = {
            "asset_id": asset_id, "asset_usd_path": str(asset_path),
            "asset_digest": asset_digest,
            "pose": {"position_world_m": position.tolist(), "orientation_xyzw": quaternion.tolist()},
            "collision_layers": [
                {"path": str(path), "digest": _sha256(path)} for path in sorted(layer_paths)
            ],
            "triangle_count": len(actor_triangles),
        }
        if row.get("task_motion") is not None:
            binding["task_motion"] = _task_motion(row["task_motion"])
        bindings.append(binding)
    return triangles, paths, tuple(bindings)


def _task_motion(value: Any) -> dict[str, Any]:
    """The joint a task drives, its span from reset to its success threshold, and
    whether its part holds where released and whether release is required.
    Absent fields retain the legacy assumptions: holds and requires release."""
    try:
        joint_id = str(value["joint_id"])
        task_range = [float(v) for v in value["task_range"]]
        reset = {str(k): float(v) for k, v in (value.get("reset_joint_positions") or {}).items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise RobotPlacementGeometryError("robot_placement_task_motion_invalid") from exc
    if (not joint_id or len(task_range) != 2 or not all(map(math.isfinite, [*task_range, *reset.values()]))
            or not isinstance(value.get("holds_released_position", True), bool)
            or not isinstance(value.get("release_required", True), bool)):
        raise RobotPlacementGeometryError("robot_placement_task_motion_invalid")
    motion = {"joint_id": joint_id, "task_range": task_range, "reset_joint_positions": reset}
    if "holds_released_position" in value:
        motion["holds_released_position"] = value["holds_released_position"]
    if "release_required" in value:
        motion["release_required"] = value["release_required"]
    return motion


def _task_motion_points(bindings: Sequence[Mapping[str, Any]], *, whole_range: bool = False,
                        falling_only: bool = False) -> np.ndarray:
    parts = [np.zeros((0, 3))]
    for binding in bindings:
        motion = binding.get("task_motion")
        if motion is None or (falling_only and (motion.get("holds_released_position", True)
                                               or not motion.get("release_required", True))):
            continue
        try:
            parts.append(task_motion_surface(
                asset_usd=binding["asset_usd_path"], asset_pose_world=binding["pose"],
                joint_id=motion["joint_id"], task_range=None if whole_range else motion["task_range"],
                reset_joint_positions=motion["reset_joint_positions"]))
        except ScenePhysicsPreflightError as exc:
            raise RobotPlacementGeometryError(f"robot_placement_task_motion_unmeasurable:{exc}") from exc
    return np.concatenate(parts)


def _task_motion_contact_count(
    index: RobotPlacementGeometryIndex, robot_id: str, position: np.ndarray, yaw: float,
    points: np.ndarray | None = None, margin_m: float = TASK_MOTION_CLEARANCE_M,
) -> int:
    """Swept task-part samples inside the robot's fixed volumes, grown by ``margin_m``, at this base pose."""
    try:
        reach = robot_fixed_volume_extent(robot_id) + margin_m
        points = index.task_motion_points if points is None else points
        near = points[(np.abs(points[:, 0] - position[0]) <= reach) & (np.abs(points[:, 1] - position[1]) <= reach)]
        relative = near - position
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        local = np.c_[cos_yaw * relative[:, 0] + sin_yaw * relative[:, 1],
                      cos_yaw * relative[:, 1] - sin_yaw * relative[:, 0], relative[:, 2]]
        return int(robot_fixed_volume_mask(robot_id, local, margin_m).sum())
    except ScenePhysicsPreflightError as exc:
        raise RobotPlacementGeometryError(f"robot_placement_task_motion_unmeasurable:{exc}") from exc


def build_robot_placement_geometry_index(
    *, scene_collision_usd_path: str | Path, robot_asset_usd_path: str | Path,
    task_assets: Sequence[Mapping[str, Any]] = (),
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
    # Task actors are obstacles at their actual scene poses. They do not become
    # mounting/support surfaces merely by having a horizontal panel or rack.
    support_surfaces = _support_surfaces(triangles, prim_paths)
    actor_triangles, actor_paths, actor_bindings = _placed_task_asset_triangles(task_assets)
    background_digest = _sha256(scene_path)
    scene_digest = background_digest
    if actor_bindings:
        triangles = np.concatenate([triangles, *actor_triangles], axis=0)
        prim_paths = (*prim_paths, *actor_paths)
        scene_digest = canonical_digest({
            "schema_version": "robot_placement_collision_context.v1",
            "background_collision_digest": background_digest,
            "task_assets": actor_bindings,
        })
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
    # Native asset references select the default robot prim. Publisher demo
    # objects outside it are not robot links; instance proxies inside it are.
    robot_triangles, _robot_prim_paths = _stage_triangles(robot_stage, root_prim=_robot_root(robot_stage))
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
        scene_digest=scene_digest,
        robot_asset_path=str(robot_path),
        robot_asset_digest=_sha256(robot_path),
        robot_triangles=robot_triangles,
        triangles=triangles,
        triangle_prim_paths=prim_paths,
        triangle_minimum=minimum,
        triangle_maximum=maximum,
        triangle_normal_abs_z=normal_abs_z,
        watertight_prim_bounds=_watertight_prim_bounds(triangles, prim_paths),
        support_surfaces=support_surfaces,
        robot_local_bounds_minimum_m=robot_minimum,
        robot_local_bounds_maximum_m=robot_maximum,
        background_collision_digest=background_digest,
        task_asset_bindings=actor_bindings,
        task_motion_points=_task_motion_points(actor_bindings),
        task_limit_points=_task_motion_points(actor_bindings, whole_range=True),
        task_fall_points=_task_motion_points(actor_bindings, whole_range=True, falling_only=True),
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
        "background_collision_digest": index.background_collision_digest,
        "task_assets": list(index.task_asset_bindings),
        "analytic_primitive_collision_method": "transformed_conservative_local_envelope",
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


def _supported_sample_count(samples: np.ndarray, triangles: np.ndarray) -> int:
    """Count footprint samples covered by any support triangle in XY."""

    triangle_xy = triangles[:, :, :2]
    a = triangle_xy[:, 0]
    v0 = triangle_xy[:, 2] - a
    v1 = triangle_xy[:, 1] - a
    dot00 = np.einsum("ij,ij->i", v0, v0)
    dot01 = np.einsum("ij,ij->i", v0, v1)
    dot11 = np.einsum("ij,ij->i", v1, v1)
    denominator = dot00 * dot11 - dot01 * dot01
    valid = np.abs(denominator) > 1.0e-15
    inverse = np.zeros_like(denominator)
    inverse[valid] = 1.0 / denominator[valid]
    supported = 0
    for sample in samples:
        v2 = sample - a
        dot02 = np.einsum("ij,ij->i", v0, v2)
        dot12 = np.einsum("ij,ij->i", v1, v2)
        u = (dot11 * dot02 - dot01 * dot12) * inverse
        v = (dot00 * dot12 - dot01 * dot02) * inverse
        inside = (
            valid
            & (u >= -1.0e-8)
            & (v >= -1.0e-8)
            & (u + v <= 1.0 + 1.0e-8)
        )
        supported += int(np.any(inside))
    return supported


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


def _support_triangles_within_height_tolerance(
    *, index: RobotPlacementGeometryIndex, surface: SupportSurface,
    root_height_m: float, tolerance_m: float,
) -> np.ndarray:
    """Include neighboring height bins within the declared support tolerance.

    Height bins discover candidate planes; they are not separate physical
    surfaces. Every included vertex must still lie inside the allowed height
    band, and support stays on the declared source mesh.
    """

    eligible = np.flatnonzero(
        (index.triangle_normal_abs_z >= 0.95)
        & (index.triangle_minimum[:, 2] >= root_height_m - tolerance_m)
        & (index.triangle_maximum[:, 2] <= root_height_m + tolerance_m)
    )
    same_mesh = np.fromiter(
        (index.triangle_prim_paths[int(i)] == surface.prim_path for i in eligible),
        dtype=bool, count=len(eligible),
    )
    return index.triangles[eligible[same_mesh]]


def _overlapping_scene_triangle_indices(
    *,
    scene_minimum: np.ndarray,
    scene_maximum: np.ndarray,
    scene_indices: np.ndarray,
    robot_minimum: np.ndarray,
    robot_maximum: np.ndarray,
) -> list[int]:
    """Apply the exact AABB predicate after a conservative spatial lookup.

    Overlapping boxes have center distance in each axis at most the sum of
    their half extents. The infinity-norm search therefore cannot discard an
    overlapping pair; the original inclusive comparisons decide every match.
    Querying one scene triangle at a time also bounds temporary storage.
    """

    if not len(scene_indices) or not len(robot_minimum):
        return []
    robot_centers = (robot_minimum + robot_maximum) * 0.5
    robot_radii = np.max((robot_maximum - robot_minimum) * 0.5, axis=1)
    # Separate triangle size scales so a few large faces cannot inflate every
    # lookup for the much more numerous small faces in a detailed robot asset.
    exponents = np.frexp(robot_radii)[1]
    groups = []
    for exponent in np.unique(exponents):
        indices = np.flatnonzero(exponents == exponent)
        groups.append(
            (indices, cKDTree(robot_centers[indices]), float(np.max(robot_radii[indices])))
        )
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(robot_minimum))),
        float(np.max(np.abs(robot_maximum))),
        float(np.max(np.abs(scene_minimum[scene_indices]))),
        float(np.max(np.abs(scene_maximum[scene_indices]))),
    )
    padding = 16.0 * np.finfo(np.float64).eps * coordinate_scale
    result: list[int] = []
    for scene_index in scene_indices:
        minimum = scene_minimum[int(scene_index)]
        maximum = scene_maximum[int(scene_index)]
        center = (minimum + maximum) * 0.5
        radius = float(np.max((maximum - minimum) * 0.5))
        for indices, tree, robot_radius in groups:
            nearby = tree.query_ball_point(center, radius + robot_radius + padding, p=np.inf)
            if nearby:
                selected = indices[nearby]
                if np.any(
                    np.all(
                        (robot_maximum[selected] >= minimum)
                        & (robot_minimum[selected] <= maximum),
                        axis=1,
                    )
                ):
                    result.append(int(scene_index))
                    break
    return result


def _candidate_support_and_reach_can_pass(
    *,
    index: RobotPlacementGeometryIndex,
    proposal: Mapping[str, Any],
    target: np.ndarray,
    robot_id: str,
    support_triangles: dict[tuple[str, float], np.ndarray],
) -> bool:
    """Reject necessary support/reach failures before detailed collision work."""

    pose = proposal["pose"]
    position = np.asarray(pose["position_world_m"], dtype=np.float64)
    yaw, upright = _yaw_from_quaternion(pose["orientation_xyzw"])
    profile = get_robot_profile(robot_id)
    shoulder = position + np.asarray([0.0, 0.0, profile.shoulder_above_root_m])
    half_x, half_y = profile.footprint_half_extent_xyz[:2]
    delta = target[:2] - position[:2]
    target_yaw = math.atan2(float(delta[1]), float(delta[0]))
    facing_error = abs(math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)))
    if (
        not upright
        or float(np.linalg.norm(target - shoulder)) > profile.max_shoulder_to_affordance_m()
        or float(np.linalg.norm(delta)) <= max(half_x, half_y) + profile.probe_clearance_m
        or math.degrees(facing_error) > profile.max_facing_error_deg
    ):
        return False
    surface_id = str(proposal.get("support_surface_id") or "")
    surface = next((s for s in index.support_surfaces if s.surface_id == surface_id), None)
    if surface is None or abs(float(position[2]) - surface.height_m) > 0.02:
        return False
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    samples = np.asarray(
        [[0.0, 0.0], [-half_x, -half_y], [-half_x, half_y], [half_x, -half_y], [half_x, half_y]]
    ) @ rotation.T + position[:2]
    key = (surface_id, float(position[2]))
    if key not in support_triangles:
        support_triangles[key] = _support_triangles_within_height_tolerance(
            index=index, surface=surface, root_height_m=float(position[2]), tolerance_m=0.02,
        )
    return _supported_sample_count(samples, support_triangles[key]) == len(samples)


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
        selected = _support_triangles_within_height_tolerance(
            index=index, surface=surface, root_height_m=float(position[2]),
            tolerance_m=support_height_tolerance_m,
        )
        support_height_error_m = abs(float(position[2]) - surface.height_m)
        supported_count = _supported_sample_count(samples, selected)
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
    broad_collision_indices = np.flatnonzero(overlap & ~allowed_support_contact)
    # A whole-arm AABB falsely rejects an arm beside a thin desktop: empty
    # space between its links intersects the desk even when no link does.
    # Triangle AABBs remain conservative while testing actual robot geometry.
    collision_indices: list[int] = []
    if len(broad_collision_indices):
        rotation_3d = np.asarray(
            [[math.cos(yaw), -math.sin(yaw), 0.0],
             [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        robot_triangles_world = index.robot_triangles @ rotation_3d.T + position
        robot_triangle_minimum = robot_triangles_world.min(axis=1)
        robot_triangle_maximum = robot_triangles_world.max(axis=1)
        collision_indices = _overlapping_scene_triangle_indices(
            scene_minimum=index.triangle_minimum,
            scene_maximum=index.triangle_maximum,
            scene_indices=broad_collision_indices,
            robot_minimum=robot_triangle_minimum,
            robot_maximum=robot_triangle_maximum,
        )
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
    # The base and shoulder cannot move out of a door's way, so a pose where the
    # task's moving part would strike them fails every episode it opens.
    motion_contacts = (
        _task_motion_contact_count(index, robot_id, position, yaw) if len(index.task_motion_points) else 0
    )
    if motion_contacts:
        blockers.append("robot_base_in_task_motion_path")
    # A part that falls once released strikes whatever stands anywhere on its way
    # down. Measured as actual overlap; the paid preflight confirms it finely.
    fall_contacts = (
        _task_motion_contact_count(index, robot_id, position, yaw, index.task_fall_points, margin_m=0.0)
        if len(index.task_fall_points) else 0
    )
    if fall_contacts:
        blockers.append("robot_base_in_task_part_fall_path")

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
    if len(index.task_motion_points):
        result["task_motion_contact_point_count"] = motion_contacts
        # Beyond the threshold a strike does not block: a policy may stop short.
        # Counted here, as actual overlap without the blocking margin, so the
        # search can prefer a base the part never reaches.
        result["task_limit_contact_point_count"] = _task_motion_contact_count(
            index, robot_id, position, yaw, index.task_limit_points, margin_m=0.0)
    if len(index.task_fall_points):
        result["task_fall_contact_point_count"] = fall_contacts
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


def _geometry_gate_chunk_worker(
    work: tuple[
        RobotPlacementGeometryIndex,
        Sequence[Mapping[str, Any]],
        Sequence[float],
        str,
        bool,
    ],
) -> list[dict[str, Any]]:
    index, proposals, target, robot_id, prefilter = work
    support_triangles: dict[tuple[str, float], np.ndarray] = {}
    target_array = np.asarray(target, dtype=np.float64)
    result = []
    for proposal in proposals:
        if prefilter and not _candidate_support_and_reach_can_pass(
            index=index, proposal=proposal, target=target_array, robot_id=robot_id,
            support_triangles=support_triangles,
        ):
            # Internal enumeration sentinel only. Every retained candidate gets
            # the complete gate below, including the unchanged collision rule.
            result.append({"status": "rejected"})
            continue
        result.append(validate_robot_placement_geometry_candidate(
            index=index,
            proposal=proposal,
            target_position_world_m=target,
            robot_id=robot_id,
        ))
    return result


def _validated_worker_count(worker_count: int, *, blocker: str) -> int:
    try:
        workers = int(worker_count)
    except (TypeError, ValueError) as exc:
        raise RobotPlacementGeometryError(blocker) from exc
    if not 1 <= workers <= 32:
        raise RobotPlacementGeometryError(blocker)
    return workers


def _parallel_geometry_gates(
    *,
    index: RobotPlacementGeometryIndex,
    proposals: Sequence[Mapping[str, Any]],
    target_position_world_m: Sequence[float],
    robot_id: str,
    worker_count: int,
    prefilter_nonpassing: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate complete geometry gates concurrently, retaining proposal order."""

    workers = _validated_worker_count(
        worker_count, blocker="robot_placement_geometry_worker_count_invalid"
    )
    if workers == 1 or len(proposals) <= 1:
        return _geometry_gate_chunk_worker(
            (index, proposals, target_position_world_m, robot_id, prefilter_nonpassing)
        )
    active_workers = min(workers, len(proposals))
    work = [
        (
            index,
            [dict(proposal) for proposal in proposals[offset::active_workers]],
            list(target_position_world_m),
            robot_id,
            prefilter_nonpassing,
        )
        for offset in range(active_workers)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=active_workers, mp_context=context
    ) as executor:
        chunks = list(executor.map(_geometry_gate_chunk_worker, work))
        return [chunks[i % active_workers][i // active_workers] for i in range(len(proposals))]


def _parallel_trajectory_gates(
    *,
    proposals: Sequence[Mapping[str, Any]],
    trajectory_waypoints_world_m: Sequence[Sequence[float]],
    trajectory_phase_ids: Sequence[str],
    trajectory_orientations_world_xyzw: Sequence[Sequence[float]],
    worker_count: int,
) -> list[dict[str, Any]]:
    """Evaluate independent base candidates concurrently, retaining order."""

    workers = _validated_worker_count(
        worker_count, blocker="robot_placement_trajectory_worker_count_invalid"
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


def _arm_path_inputs(
    index: RobotPlacementGeometryIndex,
    phase_ids: Sequence[str],
    waypoints: Sequence[Sequence[float]],
) -> dict[str, Any] | None:
    """What the arm-path gate needs: the one part the task moves and its planned handle contact.

    A task that moves no part, or plans no handle contact, leaves the gate nothing to measure.
    """
    moving = [binding for binding in index.task_asset_bindings if binding.get("task_motion")]
    phases = dict(zip(phase_ids, waypoints))
    if len(moving) != 1 or "handle_contact" not in phases or "approach" not in phases:
        return None
    motion = moving[0]["task_motion"]
    return {
        "asset_usd": moving[0]["asset_usd_path"], "asset_pose_world": moving[0]["pose"],
        "joint_id": motion["joint_id"], "task_range": motion["task_range"],
        "reset_joint_positions": motion["reset_joint_positions"],
        "contact_point_world": [float(v) for v in phases["handle_contact"]],
        "approach_point_world": [float(v) for v in phases["approach"]],
    }


def _arm_path_record(report: Mapping[str, Any]) -> dict[str, Any]:
    best = report.get("best_grip") or {}
    return {
        "status": report["status"], "margin_m": report["margin_m"], "grips_tried": report["grips_tried"],
        "grip_slide_m": best.get("slide_m"), "grip_roll_deg": best.get("roll_deg"),
        "minimum_clearance_m": best.get("minimum_clearance_m"), "report_digest": report["report_digest"],
    }


#: How many surfaces the height-proximity ranking contributes to the search.
_HEIGHT_RANKED_SURFACE_LIMIT = 12
#: How many of the largest supports are always searched regardless of height.
_LARGEST_SUPPORT_SURFACE_LIMIT = 3


def enumerate_robot_placement_geometry_candidates(
    *,
    index: RobotPlacementGeometryIndex,
    target_position_world_m: Sequence[float],
    robot_id: str = "franka_panda",
    maximum_candidates: int = 48,
    trajectory_waypoints_world_m: Sequence[Sequence[float]] = (),
    trajectory_phase_ids: Sequence[str] = (),
    trajectory_orientations_world_xyzw: Sequence[Sequence[float]] = (),
    geometry_worker_count: int = 1,
    trajectory_worker_count: int = 1,
    diagnostics: dict[str, Any] | None = None,
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
        # The arm reaches from its shoulder, which stands above the support.
        and target[2] - surface.height_m - profile.shoulder_above_root_m <= profile.max_shoulder_to_affordance_m()
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
    geometry_work: list[
        tuple[dict[str, Any], SupportSurface, float, int]
    ] = []
    # Surfaces are ranked by how close they sit to the target height, which is
    # the right instinct for mounting an arm at working height -- but the cap
    # then silently discarded the floor. Scene 840938: the floor ranked 18th of
    # 19 viable surfaces (90.8 m2) and was cut, so every evaluated candidate sat
    # on a shelf as small as 0.029 m2 and failed base-support coverage. A large
    # support is exactly where a fixed arm is usually mounted, so always keep
    # the biggest ones in the search even when their height ranks them last.
    considered = list(surfaces[:_HEIGHT_RANKED_SURFACE_LIMIT])
    considered_ids = {surface.surface_id for surface in considered}
    for surface in sorted(surfaces, key=lambda row: -row.area_m2)[
        :_LARGEST_SUPPORT_SURFACE_LIMIT
    ]:
        if surface.surface_id not in considered_ids:
            considered.append(surface)
            considered_ids.add(surface.surface_id)

    remaining = [surface for surface in surfaces if surface.surface_id not in considered_ids]
    all_geometry_gates: list[dict[str, Any]] = []
    all_trajectory_gates: list[dict[str, Any]] = []
    geometry_passing_count = 0
    searched_surface_count = 0
    search_pass_count = 0
    candidates: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    arm_inputs = _arm_path_inputs(index, trajectory_phase_ids, trajectory_waypoints_world_m)
    arm_path_not_applicable = (
        {"status": "not_applicable", "reason": "task_path_without_handle_contact"}
        if arm_inputs is None and any(b.get("task_motion") for b in index.task_asset_bindings) else None
    )
    arm_path_tested = 0
    arm_path_blockers: Counter[str] = Counter()
    for support_batch in (considered, remaining):
        if not support_batch:
            continue
        considered = support_batch
        geometry_work = []
        for surface_index, surface in enumerate(considered, start=searched_surface_count):
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
                    geometry_work.append(
                        (proposal, surface, float(radius), angle_index)
                    )

        geometry_gates = _parallel_geometry_gates(
            index=index,
            proposals=[row[0] for row in geometry_work],
            target_position_world_m=target,
            robot_id=robot_id,
            worker_count=geometry_worker_count,
            prefilter_nonpassing=True,
        )
        prequalified = [
            (proposal, gate, surface, radius, angle_index)
            for (proposal, surface, radius, angle_index), gate in zip(
                geometry_work, geometry_gates, strict=True
            )
            if gate["status"] == "passed"
        ]

        trajectory_gates = _parallel_trajectory_gates(
            proposals=[row[0] for row in prequalified],
            trajectory_waypoints_world_m=trajectory_waypoints_world_m,
            trajectory_phase_ids=trajectory_phase_ids,
            trajectory_orientations_world_xyzw=trajectory_orientations_world_xyzw,
            worker_count=trajectory_worker_count,
        )
        candidates = []
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
                # Preserve clearance differences even when conservative surface
                # sampling gives every pose some hits. This adds no rejection.
                int(gate.get("task_limit_contact_point_count") or 0),
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
        if arm_inputs is not None and candidates:
            # The part must not strike the arm: test bases in rank order until enough hold a clean grip.
            candidates.sort(key=lambda item: item[0])
            kept = []
            for score, proposal, gate in candidates:
                if len(kept) >= int(maximum_candidates):
                    break
                seed = next((row["solved_joint_positions"] for row in gate["trajectory_position_ik_gate"]["waypoints"]
                             if row["phase_id"] == "approach"), None)
                try:
                    report = arm_path_clearance(robot_id=robot_id, base_pose_world=proposal["pose"],
                                                seed_joint_positions=seed, **arm_inputs)
                except RobotArmPathClearanceError as exc:
                    raise RobotPlacementGeometryError(f"robot_placement_arm_path_unmeasurable:{exc}") from exc
                arm_path_tested += 1
                if report["status"] != "passed":
                    arm_path_blockers.update(blocker.split(":", 1)[0] for blocker in report["blockers"])
                    continue
                kept.append((score, {**proposal, "arm_path_clearance": _arm_path_record(report)}, gate))
            candidates = kept
        elif arm_path_not_applicable and candidates:
            candidates = [(score, {**proposal, "arm_path_clearance": arm_path_not_applicable}, gate)
                          for score, proposal, gate in candidates]
        all_geometry_gates.extend(geometry_gates)
        all_trajectory_gates.extend(trajectory_gates)
        geometry_passing_count += len(prequalified)
        searched_surface_count += len(considered)
        search_pass_count += 1
        if candidates:
            break
    candidates.sort(key=lambda item: item[0])
    if diagnostics is not None:
        diagnostics.update({
            "proposal_count": len(all_geometry_gates),
            "support_or_reach_prefilter_rejected_count": sum(len(gate) == 1 for gate in all_geometry_gates),
            "detailed_geometry_tested_count": sum(len(gate) > 1 for gate in all_geometry_gates),
            "detailed_geometry_blocker_counts": dict(Counter(
                blocker for gate in all_geometry_gates for blocker in gate.get("blockers", [])
            )),
            "geometry_passing_count": geometry_passing_count,
            "trajectory_waypoint_count": len(trajectory_waypoints_world_m),
            "trajectory_rejected_count": sum(gate["status"] == "rejected" for gate in all_trajectory_gates),
            "arm_path_tested_count": arm_path_tested,
            "arm_path_rejected_count": sum(arm_path_blockers.values()),
            "arm_path_blocker_counts": dict(arm_path_blockers),
            "candidate_count": min(len(candidates), int(maximum_candidates)),
            "support_search_pass_count": search_pass_count,
            "searched_support_surface_count": searched_surface_count,
            "available_support_surface_count": len(surfaces),
            "support_search_exhausted": searched_surface_count == len(surfaces),
        })
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
            **({"task_motion_contact_point_count": gate["task_motion_contact_point_count"],
                "task_limit_contact_point_count": gate["task_limit_contact_point_count"]}
               if "task_limit_contact_point_count" in gate else {}),
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
    """Render opaque, depth-tested local views without a paid GPU."""
    from .robot_placement_preview_rasterizer import render
    return render(index=index, proposal=proposal,
                  target_position_world_m=target_position_world_m,
                  trajectory_waypoints_world_m=trajectory_waypoints_world_m,
                  image_size=image_size)



__all__ = [
    "RobotPlacementGeometryError",
    "RobotPlacementGeometryIndex",
    "build_robot_placement_geometry_index",
    "enumerate_robot_placement_geometry_candidates",
    "render_robot_placement_geometry_previews",
    "summarize_robot_placement_geometry",
    "validate_robot_placement_geometry_candidate",
]
