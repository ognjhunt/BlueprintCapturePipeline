"""Live Isaac stance, reach, facing, and collision validation contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def build_live_geometry_results(
    *,
    robot_xyz: Sequence[float],
    robot_quaternion_xyzw: Sequence[float],
    target_xyz: Sequence[float],
    overlapping_prim_paths: Sequence[str],
    robot_prim_path: str,
    max_reach_distance_m: float,
) -> dict[str, dict[str, Any]]:
    robot = [float(item) for item in robot_xyz]
    target = [float(item) for item in target_xyz]
    quat = [float(item) for item in robot_quaternion_xyzw]
    values = [*robot, *target, *quat, float(max_reach_distance_m)]
    if len(robot) != 3 or len(target) != 3 or len(quat) != 4 or not all(
        math.isfinite(item) for item in values
    ):
        raise ValueError("live_geometry_numeric_input_invalid")
    qx, qy, qz, qw = quat
    norm = math.sqrt(sum(item * item for item in quat))
    if norm <= 0:
        raise ValueError("live_geometry_robot_orientation_invalid")
    qx, qy, qz, qw = (item / norm for item in (qx, qy, qz, qw))
    forward = (
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qz * qw),
    )
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    horizontal_distance = math.hypot(dx, dy)
    direction = (dx / horizontal_distance, dy / horizontal_distance) if horizontal_distance else forward
    facing_dot = forward[0] * direction[0] + forward[1] * direction[1]
    distance_3d = math.dist(robot, target)
    ignored_terms = ("ground", "floor", "plane")
    collisions = sorted(
        {
            str(path)
            for path in overlapping_prim_paths
            if str(path)
            and not str(path).startswith(str(robot_prim_path).rstrip("/") + "/")
            and str(path) != str(robot_prim_path)
            and not any(term in str(path).lower() for term in ignored_terms)
        }
    )
    reach_valid = distance_3d <= float(max_reach_distance_m)
    facing_valid = facing_dot >= 0.5
    return {
        "stance": {
            "schema_version": "g1_kitchen_live_stance_validation.v1",
            "stance_valid": bool(reach_valid and facing_valid and not collisions),
            "reach_valid": reach_valid,
            "facing_valid": facing_valid,
            "robot_xyz": robot,
            "target_xyz": target,
            "distance_m": distance_3d,
            "max_reach_distance_m": float(max_reach_distance_m),
            "facing_dot": facing_dot,
            "measurement_source": "live_isaac_root_pose_target_world_bound",
        },
        "collision": {
            "schema_version": "g1_kitchen_live_collision_validation.v1",
            "collision_free": not collisions,
            "clearance_valid": not collisions,
            "overlapping_non_floor_non_robot_prim_paths": collisions,
            "measurement_source": "live_isaac_physx_overlap_box",
        },
    }
