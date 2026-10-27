"""Where a robot starts: a clear spot on the scene's floor, facing the task.

Beta journey step: L5 team policy evaluation. A mobile manipulator or a
humanoid stands on the floor itself; a fixed arm stands on a mount whose top is
level with the task's support (``team_scene_packet``). Either starts where a
person would stand to do the job: the clear floor spot closest to the task
object, facing it, with nothing that reaches the floor under its footprint
below its root (``scene_placement.validation.validate_stand_pose``, the
validator the G1 humanoid lane uses). The floor is the dominant low horizontal
band of the scene's own collision mesh (``external_scene_robot_placement``).
A task part that moves, such as a door or a drawer, also keeps the station out
of its way: no stance the part strikes on its way to success, or anywhere on a
fall gravity carries it through, and a stance it never reaches is preferred.
Host-side and deterministic (OpenUSD), so a robot with no clear stance is
refused by name before any GPU time is spent.
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
#: A task supported by the floor itself (an installed appliance, a floor
#: cabinet) puts a fixed arm's mount on the floor; this absorbs floor-fit noise.
FLOOR_MOUNT_TOLERANCE_M = 0.02


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


def _strikes(points: Any, *, base: Any, yaw: float, root: float, footprint: list[float],
             robot_id: str, margin_m: float) -> bool:
    """Whether moving-part samples enter the station's mount or the robot's fixed volumes on it."""

    import numpy as np

    from .scene_physics_preflight import robot_fixed_volume_extent, robot_fixed_volume_mask

    if points is None or not len(points):
        return False
    reach = max(robot_fixed_volume_extent(robot_id), math.hypot(*footprint)) + margin_m
    near = points[(np.abs(points[:, 0] - base[0]) <= reach) & (np.abs(points[:, 1] - base[1]) <= reach)]
    if not len(near):
        return False
    relative = near - base
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    local = np.c_[cos_yaw * relative[:, 0] + sin_yaw * relative[:, 1],
                  cos_yaw * relative[:, 1] - sin_yaw * relative[:, 0], relative[:, 2]]
    mount = ((np.abs(local[:, 0]) <= footprint[0] + margin_m) & (np.abs(local[:, 1]) <= footprint[1] + margin_m)
             & (local[:, 2] >= -root - margin_m) & (local[:, 2] <= margin_m))
    return bool(mount.any() or robot_fixed_volume_mask(robot_id, local, margin_m).any())


def ground_stance(*, collision_path: Path, collision_sha256: str, offset: list[float],
                  subject_center: list[float], subject_minimum: list[float], subject_maximum: list[float],
                  footprint_half_extent_m: list[float], root_height_m: float | None = None,
                  root_world_z_m: float | None = None, task_motion: Mapping[str, Any] | None = None,
                  ranked: int = 0) -> Any:
    """The closest clear floor stance facing the subject, or raise by name (no GPU).

    The root stands ``root_height_m`` above the floor, or at ``root_world_z_m``.
    ``task_motion`` holds world samples of a moving task part, for ``robot_id``'s
    fixed volumes on the station: ``path_points`` (reset to the success
    threshold) and ``fall_points`` (where gravity carries it) block a stance, and
    a stance ``limit_points`` (its whole range) reaches ranks after the rest.
    With ``ranked``, the ``ranked`` best clear stances are returned in order instead.
    """

    from .external_scene_robot_placement import (
        ExternalSceneRobotPlacementError,
        _infer_horizontal_support_surface,
    )
    from .scene_placement.types import SceneObject
    from .scene_placement.usd_index import UsdSceneSpatialIndex
    from .scene_placement.validation import validate_stand_pose

    vertices, faces = _world_triangles(collision_path, offset)
    try:
        floor_z, _, _ = _infer_horizontal_support_surface(stage_vertices=vertices, faces=faces)
    except ExternalSceneRobotPlacementError as exc:
        raise TeamEmbodimentError(["team_embodiment_scene_floor_not_found"]) from exc
    root = float(root_height_m) if root_height_m is not None else float(root_world_z_m) - float(floor_z)
    if root_height_m is None and -FLOOR_MOUNT_TOLERANCE_M <= root <= 0.0:
        root = 0.0
    if not math.isfinite(root) or root < 0.0 or (root_height_m is not None and root <= 0.0):
        raise TeamEmbodimentError(["team_embodiment_stance_height_invalid"])
    def shifted(point: Any) -> tuple[float, ...]:
        return tuple(float(v) + offset[axis] for axis, v in enumerate(point))

    obstacles = [replace(box, bbox_min=shifted(box.bbox_min), bbox_max=shifted(box.bbox_max),
                         centroid=shifted(box.centroid))
                 for box in UsdSceneSpatialIndex(usd_path=str(collision_path)).obstacle_boxes()]
    start = [float(v) for v in subject_center[:3]]
    subject = SceneObject(id="task_object", label="task object", centroid=tuple(start),
                          bbox_min=tuple(float(v) for v in subject_minimum),
                          bbox_max=tuple(float(v) for v in subject_maximum))
    footprint = (*(float(v) for v in footprint_half_extent_m), root)
    motion = dict(task_motion or {})
    if motion:
        import numpy as np

        from .task_evaluation_robot_placement_geometry import TASK_MOTION_CLEARANCE_M

        robot_id = str(motion.get("robot_id") or "")
        clearance = {key: np.asarray(motion.get(key, np.zeros((0, 3))), dtype=float).reshape(-1, 3)
                     for key in ("path_points", "fall_points", "limit_points")}
    steps, best, clear, struck = math.floor(SEARCH_RADIUS_M / GRID_STEP_M + 1e-9), None, 0, 0
    candidates: list[Any] = []
    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            x, y = start[0] + ix * GRID_STEP_M, start[1] + iy * GRID_STEP_M
            distance = math.hypot(x - start[0], y - start[1])
            if distance > SEARCH_RADIUS_M:
                continue
            yaw = math.atan2(start[1] - y, start[0] - x)
            if not validate_stand_pose((x, y, floor_z + root), yaw, subject, obstacles, floor_z,
                                       footprint_half_extent=footprint, pelvis_height=root).ok:
                continue
            reached = False
            if motion:
                # The path takes the sampling margin; a fall or the whole range counts actual overlap.
                place = {"base": np.asarray([x, y, floor_z + root]), "yaw": yaw, "root": root,
                         "footprint": list(footprint[:2]), "robot_id": robot_id}
                if (_strikes(clearance["path_points"], margin_m=TASK_MOTION_CLEARANCE_M, **place)
                        or _strikes(clearance["fall_points"], margin_m=0.0, **place)):
                    struck += 1
                    continue
                reached = _strikes(clearance["limit_points"], margin_m=0.0, **place)
            clear += 1
            candidates.append(((reached, distance, ix, iy), x, y, yaw))
            if best is None or (reached, distance, ix, iy) < best[0]:
                best = ((reached, distance, ix, iy), x, y, yaw)
    if best is None:
        raise TeamEmbodimentError(["team_embodiment_task_motion_no_clear_stance" if struck
                                   else "team_embodiment_free_base_no_clear_stance"])
    if ranked:
        return [_stance(row, floor_z=floor_z, root=root, footprint_half_extent_m=footprint_half_extent_m, clear=clear,
                        struck=struck, motion=motion, robot_id=robot_id if motion else None,
                        collision_sha256=collision_sha256) for row in sorted(candidates)[:ranked]]
    return _stance(best, floor_z=floor_z, root=root, footprint_half_extent_m=footprint_half_extent_m, clear=clear,
                   struck=struck, motion=motion, robot_id=robot_id if motion else None,
                   collision_sha256=collision_sha256)


def _stance(row: Any, *, floor_z: float, root: float, footprint_half_extent_m: list[float], clear: int, struck: int,
            motion: Mapping[str, Any], robot_id: str | None, collision_sha256: str) -> dict[str, Any]:
    (reached, *_), x, y, yaw = row
    stance = {"schema_version": SCHEMA, "floor_z_m": float(floor_z),
              "position_world_m": [x, y, float(floor_z) + root],
              "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], "yaw_rad": yaw,
              "footprint_half_extent_m": [float(v) for v in footprint_half_extent_m],
              "clear_stances": clear, "search_radius_m": SEARCH_RADIUS_M, "grid_step_m": GRID_STEP_M,
              **({"task_motion_clearance": {"robot_id": robot_id, "struck_stances": struck,
                                            "reached_beyond_task_range": reached}} if motion else {}),
              "scene_collision_sha256": collision_sha256,
              "validator": "scene_placement.validation.validate_stand_pose", "stance_digest": ""}
    stance["stance_digest"] = canonical_digest(stance, digest_field="stance_digest")
    return stance


def free_base_stance(manifest: Mapping[str, Any], *, scene_plan: Mapping[str, Any],
                     packet_dir: str | Path) -> dict[str, Any]:
    """The closest clear floor stance facing the task object, or raise by name (no GPU)."""

    base = manifest["base"]
    task = scene_plan.get("task_spec") if isinstance(scene_plan.get("task_spec"), Mapping) else {}
    start, bounds = task.get("start_pose_world"), task.get("subject_collision_bounds_scoring_frame_m")
    if (not isinstance(start, list) or len(start) < 3 or not isinstance(bounds, Mapping)
            or len(bounds.get("minimum") or []) != 3 or len(bounds.get("maximum") or []) != 3):
        raise TeamEmbodimentError(["team_embodiment_task_bounds_unavailable"])
    collision, digest, offset = _scene_collision(scene_plan, Path(packet_dir))
    return ground_stance(
        collision_path=collision, collision_sha256=digest, offset=offset, subject_center=list(start[:3]),
        subject_minimum=[float(start[i]) + float(bounds["minimum"][i]) for i in range(3)],
        subject_maximum=[float(start[i]) + float(bounds["maximum"][i]) for i in range(3)],
        footprint_half_extent_m=list(base["footprint_half_extent_m"]), root_height_m=float(base["root_height_m"]))


__all__ = ["SCHEMA", "free_base_stance", "ground_stance"]
