"""Where a robot that stands or drives starts: a clear spot on the scene's floor, facing the task.

Beta journey step: L5 team policy evaluation. The scene's robot station is the
support mount Blueprint's fixed arm stands on, above the floor; a mobile
manipulator or a humanoid stands on the floor itself. It starts where a person
would stand to do the job: the clear floor spot closest to the task object,
facing it, with nothing that reaches the floor under its declared footprint
(``scene_placement.validation.validate_stand_pose``, the validator the G1
humanoid lane uses). The floor is the dominant low horizontal band of the
scene's own collision mesh (``external_scene_robot_placement``). Host-side and
deterministic (OpenUSD), so a robot with no clear stance is refused by name
before any GPU time is spent.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .team_embodiment import TeamEmbodimentError

SCHEMA = "blueprint.team_free_base_stance.v1"
SEARCH_RADIUS_M = 1.6
GRID_STEP_M = 0.05


def _scene_collision(plan: Mapping[str, Any], packet_dir: Path) -> tuple[Path, str, list[float]]:
    rows = [row for row in plan.get("objects") or [] if row.get("semantic_role") == "scene_collision"]
    if len(rows) != 1:
        raise TeamEmbodimentError(["team_embodiment_scene_collision_unavailable"])
    pose = rows[0].get("pose_world") if isinstance(rows[0].get("pose_world"), Mapping) else {}
    orientation = [float(v) for v in pose.get("orientation_xyzw") or []]
    if len(orientation) != 4 or not all(math.isclose(a, b, abs_tol=1e-9)
                                        for a, b in zip(orientation, [0.0, 0.0, 0.0, 1.0], strict=True)):
        raise TeamEmbodimentError(["team_embodiment_scene_collision_pose_unsupported"])
    path = Path(packet_dir).joinpath(*PurePosixPath(str(rows[0].get("usd_path") or "")).parts)
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if digest is None or digest != rows[0].get("sha256"):
        raise TeamEmbodimentError(["team_embodiment_scene_collision_unavailable"])
    return path, digest, [float(v) for v in pose["position_world_m"]]


def _world_triangles(path: Path, offset: list[float]) -> tuple[Any, Any]:
    """Every collision mesh triangle in the scene's world frame (Z up, meters)."""

    import numpy as np
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if (UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or (
            stage.HasAuthoredMetadata("metersPerUnit")
            and not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0))):
        raise TeamEmbodimentError(["team_embodiment_scene_collision_frame_unsupported"])
    cache = UsdGeom.XformCache()
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points, counts = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        matrix, first, cursor = cache.GetLocalToWorldTransform(prim), len(vertices), 0
        vertices.extend([float(v) + offset[axis] for axis, v in enumerate(matrix.Transform(point))]
                        for point in points)
        for count in counts:
            faces.extend((first + indices[cursor], first + indices[cursor + k], first + indices[cursor + k + 1])
                         for k in range(1, int(count) - 1))
            cursor += int(count)
    if not faces:
        raise TeamEmbodimentError(["team_embodiment_scene_collision_unavailable"])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def free_base_stance(manifest: Mapping[str, Any], *, scene_plan: Mapping[str, Any],
                     packet_dir: str | Path) -> dict[str, Any]:
    """The closest clear floor stance facing the task object, or raise by name (no GPU)."""

    from .external_scene_robot_placement import (
        ExternalSceneRobotPlacementError,
        _infer_horizontal_support_surface,
    )
    from .scene_placement.types import SceneObject
    from .scene_placement.usd_index import UsdSceneSpatialIndex
    from .scene_placement.validation import validate_stand_pose

    base = manifest["base"]
    task = scene_plan.get("task_spec") if isinstance(scene_plan.get("task_spec"), Mapping) else {}
    start, bounds = task.get("start_pose_world"), task.get("subject_collision_bounds_scoring_frame_m")
    if (not isinstance(start, list) or len(start) < 3 or not isinstance(bounds, Mapping)
            or len(bounds.get("minimum") or []) != 3 or len(bounds.get("maximum") or []) != 3):
        raise TeamEmbodimentError(["team_embodiment_task_bounds_unavailable"])
    collision, digest, offset = _scene_collision(scene_plan, Path(packet_dir))
    vertices, faces = _world_triangles(collision, offset)
    try:
        floor_z, _, _ = _infer_horizontal_support_surface(stage_vertices=vertices, faces=faces)
    except ExternalSceneRobotPlacementError as exc:
        raise TeamEmbodimentError(["team_embodiment_scene_floor_not_found"]) from exc
    def shifted(point: Any) -> tuple[float, ...]:
        return tuple(float(v) + offset[axis] for axis, v in enumerate(point))

    obstacles = [replace(box, bbox_min=shifted(box.bbox_min), bbox_max=shifted(box.bbox_max),
                         centroid=shifted(box.centroid))
                 for box in UsdSceneSpatialIndex(usd_path=str(collision)).obstacle_boxes()]
    subject = SceneObject(id="task_object", label="task object", centroid=tuple(float(v) for v in start[:3]),
                          bbox_min=tuple(float(start[i]) + float(bounds["minimum"][i]) for i in range(3)),
                          bbox_max=tuple(float(start[i]) + float(bounds["maximum"][i]) for i in range(3)))
    root = float(base["root_height_m"])
    footprint = (*(float(v) for v in base["footprint_half_extent_m"]), root)
    steps, best, clear = math.floor(SEARCH_RADIUS_M / GRID_STEP_M + 1e-9), None, 0
    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            x, y = float(start[0]) + ix * GRID_STEP_M, float(start[1]) + iy * GRID_STEP_M
            distance = math.hypot(x - float(start[0]), y - float(start[1]))
            if distance > SEARCH_RADIUS_M:
                continue
            yaw = math.atan2(float(start[1]) - y, float(start[0]) - x)
            if not validate_stand_pose((x, y, floor_z + root), yaw, subject, obstacles, floor_z,
                                       footprint_half_extent=footprint, pelvis_height=root).ok:
                continue
            clear += 1
            if best is None or (distance, ix, iy) < best[0]:
                best = ((distance, ix, iy), x, y, yaw)
    if best is None:
        raise TeamEmbodimentError(["team_embodiment_free_base_no_clear_stance"])
    _, x, y, yaw = best
    stance = {"schema_version": SCHEMA, "floor_z_m": float(floor_z),
              "position_world_m": [x, y, float(floor_z) + root],
              "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], "yaw_rad": yaw,
              "clear_stances": clear, "search_radius_m": SEARCH_RADIUS_M, "grid_step_m": GRID_STEP_M,
              "scene_collision_sha256": digest, "validator": "scene_placement.validation.validate_stand_pose",
              "stance_digest": ""}
    stance["stance_digest"] = canonical_digest(stance, digest_field="stance_digest")
    return stance


__all__ = ["SCHEMA", "free_base_stance"]
