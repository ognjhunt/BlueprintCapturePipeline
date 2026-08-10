"""Read one articulated scorer sample from the live Arena environment."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .native_articulated_task_state import compile_native_articulated_task_sample
from .native_pose_transforms import body_local_point_midpoint_geometry
from .native_task_arena_runtime import NativeTaskArenaEnvironment


class NativeTaskArenaReadbackError(ValueError):
    """Stable failures for missing or malformed native state."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _native_list(value: Any, *, error: str) -> Any:
    if value is None:
        raise NativeTaskArenaReadbackError([error])
    module = type(value).__module__
    if module == "warp" or module.startswith("warp."):
        import warp as wp

        value = wp.to_torch(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _first_environment(value: Any, *, error: str) -> list[Any]:
    rows = _native_list(value, error=error)
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], list):
        raise NativeTaskArenaReadbackError([error])
    return rows[0]


def _force_vectors(value: Any, *, sensor_id: str) -> list[list[float]]:
    nested = _native_list(
        value, error=f"native_task_arena_force_matrix_missing:{sensor_id}"
    )
    vectors: list[list[float]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list) and len(node) == 3 and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in node
        ):
            vector = [float(item) for item in node]
            if not all(math.isfinite(item) for item in vector):
                raise NativeTaskArenaReadbackError(
                    [f"native_task_arena_force_matrix_invalid:{sensor_id}"]
                )
            vectors.append(vector)
            return
        if isinstance(node, list):
            for child in node:
                visit(child)

    visit(nested)
    if not vectors:
        raise NativeTaskArenaReadbackError(
            [f"native_task_arena_force_matrix_invalid:{sensor_id}"]
        )
    return vectors


def _names(value: Any, *, error: str) -> list[str]:
    try:
        names = [str(item) for item in value]
    except TypeError as exc:
        raise NativeTaskArenaReadbackError([error]) from exc
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise NativeTaskArenaReadbackError([error])
    return names


def _quaternion_rotate_xyzw(
    quaternion: Sequence[float], vector: Sequence[float]
) -> list[float]:
    qx, qy, qz, qw = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 0.0:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_body_quaternion_invalid"]
        )
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _body_position(
    asset: Any, *, body_name: str, error: str
) -> tuple[list[float], list[float]]:
    data = getattr(asset, "data", None)
    names = _names(
        getattr(data, "body_names", None) or getattr(asset, "body_names", None),
        error=error,
    )
    if body_name not in names:
        raise NativeTaskArenaReadbackError([f"{error}:{body_name}"])
    poses = _first_environment(getattr(data, "body_pose_w", None), error=error)
    pose = poses[names.index(body_name)]
    if not isinstance(pose, list) or len(pose) < 7:
        raise NativeTaskArenaReadbackError([error])
    return [float(value) for value in pose[:3]], [float(value) for value in pose[3:7]]


class NativeArticulatedTaskArenaReadback:
    """Compile task samples from the exact scene handles returned by the builder."""

    def __init__(self, built: NativeTaskArenaEnvironment):
        self._built = built
        if built.plan.get("task_kind") != "articulated_open_close":
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_readback_task_kind_invalid"]
            )

    def read_task_sample(self) -> dict[str, Any]:
        env = getattr(self._built.env, "unwrapped", self._built.env)
        scene = getattr(env, "scene", None)
        if scene is None:
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_scene_readback_missing"]
            )
        try:
            task_object = scene[self._built.scene_asset_names["task_object"]]
            robot = scene["robot"]
        except (KeyError, TypeError) as exc:
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_scene_readback_missing"]
            ) from exc

        task_data = getattr(task_object, "data", None)
        joint_names = _names(
            getattr(task_object, "joint_names", None)
            or getattr(task_data, "joint_names", None),
            error="native_task_arena_joint_names_missing",
        )
        joint_positions = _first_environment(
            getattr(task_data, "joint_pos", None),
            error="native_task_arena_joint_positions_missing",
        )
        joint_velocities = _first_environment(
            getattr(task_data, "joint_vel", None),
            error="native_task_arena_joint_velocities_missing",
        )
        task_root_pose = _first_environment(
            getattr(task_data, "root_pose_w", None),
            error="native_task_arena_task_root_pose_missing",
        )[:7]

        sensor_forces: dict[str, list[list[float]]] = {}
        for logical_sensor_id, scene_names in self._built.contact_sensor_names.items():
            if isinstance(scene_names, str) or not scene_names:
                raise NativeTaskArenaReadbackError(
                    [
                        "native_task_arena_contact_sensor_instances_invalid:"
                        f"{logical_sensor_id}"
                    ]
                )
            aggregate: list[list[float]] = []
            for scene_name in scene_names:
                try:
                    sensor = scene[scene_name]
                except (KeyError, TypeError) as exc:
                    raise NativeTaskArenaReadbackError(
                        [
                            "native_task_arena_contact_sensor_missing:"
                            f"{logical_sensor_id}:{scene_name}"
                        ]
                    ) from exc
                aggregate.extend(
                    _force_vectors(
                        getattr(
                            getattr(sensor, "data", None),
                            "force_matrix_w",
                            None,
                        ),
                        sensor_id=f"{logical_sensor_id}:{scene_name}",
                    )
                )
            sensor_forces[logical_sensor_id] = aggregate

        grasp_frame = self._built.plan["robot"]["grasp_frame"]
        if grasp_frame.get("kind") != "body_local_point_midpoint":
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_grasp_frame_invalid"]
            )
        finger_poses: dict[str, list[float]] = {}
        for body_name in grasp_frame["body_names"]:
            position, quaternion = _body_position(
                robot,
                body_name=body_name,
                error="native_task_arena_grasp_body_missing",
            )
            finger_poses[body_name] = [*position, *quaternion]
        grasp_geometry = body_local_point_midpoint_geometry(
            grasp_frame=grasp_frame,
            body_poses_world=finger_poses,
        )
        grasp_position = grasp_geometry["midpoint_world_m"]

        articulation = self._built.plan["articulation"]
        link_position, link_quaternion = _body_position(
            task_object,
            body_name=articulation["moving_link_native_body_name"],
            error="native_task_arena_moving_link_missing",
        )
        rotated_handle = _quaternion_rotate_xyzw(
            link_quaternion, articulation["handle_grasp_point_link_m"]
        )
        handle_position = [
            link_position[axis] + rotated_handle[axis] for axis in range(3)
        ]
        reset_object = next(
            row
            for row in self._built.plan["objects"]
            if row["semantic_role"] == "task_object"
        )
        reset_pose = [
            *reset_object["pose_world"]["position_world_m"],
            *reset_object["pose_world"]["orientation_xyzw"],
        ]
        return compile_native_articulated_task_sample(
            task_spec=self._built.plan["task_spec"],
            task_sample_binding=self._built.plan["task_sample_binding"],
            task_state_binding=self._built.plan["task_state_binding"],
            native_joint_names=joint_names,
            native_joint_positions_rad=joint_positions,
            native_joint_velocities_rad_s=joint_velocities,
            task_robot_contact_forces_w_n=sensor_forces["task_robot_contact"],
            task_scene_contact_forces_w_n=sensor_forces["task_scene_contact"],
            robot_scene_contact_forces_w_n=sensor_forces["robot_scene_contact"],
            task_root_pose_world=task_root_pose,
            task_root_reset_pose_world=reset_pose,
            grasp_frame_position_world_m=grasp_position,
            handle_reference_position_world_m=handle_position,
        )


__all__ = [
    "NativeArticulatedTaskArenaReadback",
    "NativeTaskArenaReadbackError",
]
