"""Read one articulated scorer sample from the live Arena environment."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .native_articulated_task_state import compile_native_articulated_task_sample
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


def _quaternion_product_xyzw(
    a: Sequence[float], b: Sequence[float]
) -> list[float]:
    ax, ay, az, aw = (float(item) for item in a)
    bx, by, bz, bw = (float(item) for item in b)
    result = [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
    norm = math.sqrt(sum(item * item for item in result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_quaternion_invalid"]
        )
    return [item / norm for item in result]


def _compose_pose_xyzw(
    parent_pose: Sequence[float], child_pose: Sequence[float]
) -> list[float]:
    if len(parent_pose) != 7 or len(child_pose) != 7:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_scoring_frame_transform_invalid"]
        )
    offset = _quaternion_rotate_xyzw(parent_pose[3:], child_pose[:3])
    return [
        *[
            float(parent_pose[index]) + offset[index]
            for index in range(3)
        ],
        *_quaternion_product_xyzw(parent_pose[3:], child_pose[3:]),
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
    return [float(value) for value in pose[:3]], _native_wxyz_to_xyzw(pose[3:7])


def _native_wxyz_to_xyzw(value: Sequence[float]) -> list[float]:
    """Convert Isaac Lab's native WXYZ quaternion into contract XYZW order."""

    try:
        qw, qx, qy, qz = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_quaternion_invalid"]
        ) from exc
    quaternion = [qx, qy, qz, qw]
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_quaternion_invalid"]
        )
    return [item / norm for item in quaternion]


def _quaternion_angle_xyzw(a: Sequence[float], b: Sequence[float]) -> float:
    qa = [float(item) for item in a]
    qb = [float(item) for item in b]
    if len(qa) != 4 or len(qb) != 4:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_quaternion_invalid"]
        )
    norm_a = math.sqrt(sum(item * item for item in qa))
    norm_b = math.sqrt(sum(item * item for item in qb))
    if not math.isfinite(norm_a) or not math.isfinite(norm_b) or min(norm_a, norm_b) <= 0:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_quaternion_invalid"]
        )
    dot = abs(sum(x * y for x, y in zip(qa, qb, strict=True)) / (norm_a * norm_b))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _read_bound_locked_joint_state(
    *, asset: Any, sample_binding: Any, task_spec: Any
) -> dict[str, Any]:
    if not isinstance(sample_binding, dict):
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_rigid_joint_binding_invalid"]
        )
    joint_ids = list(sample_binding.get("joint_ids") or [])
    if not joint_ids:
        return {
            "joint_positions": {},
            "joint_velocities_per_s": {},
            "locked_joint_absolute_errors": {},
            "locked_joint_containment_violation": False,
        }
    native_by_id = sample_binding.get("native_joint_names")
    roles = sample_binding.get("joint_roles")
    graph = task_spec.get("articulation_graph") if isinstance(task_spec, dict) else None
    graph_joints = graph.get("joints") if isinstance(graph, dict) else None
    if (
        not isinstance(native_by_id, dict)
        or not isinstance(roles, dict)
        or set(joint_ids) != set(native_by_id)
        or set(joint_ids) != set(roles)
        or set(roles.values()) != {"locked"}
        or not isinstance(graph_joints, list)
    ):
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_rigid_joint_binding_invalid"]
        )
    graph_by_id = {
        str(row.get("joint_id") or ""): row
        for row in graph_joints
        if isinstance(row, dict)
    }
    if set(joint_ids) != set(graph_by_id):
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_rigid_joint_graph_mismatch"]
        )
    data = getattr(asset, "data", None)
    native_names = _names(
        getattr(asset, "joint_names", None) or getattr(data, "joint_names", None),
        error="native_task_arena_rigid_joint_names_missing",
    )
    native_positions = _first_environment(
        getattr(data, "joint_pos", None),
        error="native_task_arena_rigid_joint_positions_missing",
    )
    native_velocities = _first_environment(
        getattr(data, "joint_vel", None),
        error="native_task_arena_rigid_joint_velocities_missing",
    )
    if (
        len(native_positions) != len(native_names)
        or len(native_velocities) != len(native_names)
        or set(native_names) != set(native_by_id.values())
    ):
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_rigid_joint_state_invalid"]
        )
    position_by_native = dict(zip(native_names, native_positions, strict=True))
    velocity_by_native = dict(zip(native_names, native_velocities, strict=True))
    positions: dict[str, float] = {}
    velocities: dict[str, float] = {}
    errors: dict[str, float] = {}
    violation = False
    for joint_id in sorted(joint_ids):
        row = graph_by_id[joint_id]
        try:
            position = float(position_by_native[native_by_id[joint_id]])
            velocity = float(velocity_by_native[native_by_id[joint_id]])
            reset = float(row["reset_position"])
            tolerance = float(row["reset_tolerance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_rigid_joint_state_invalid"]
            ) from exc
        if not all(
            math.isfinite(value) for value in (position, velocity, reset, tolerance)
        ) or tolerance < 0.0:
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_rigid_joint_state_invalid"]
            )
        positions[joint_id] = position
        velocities[joint_id] = velocity
        errors[joint_id] = abs(position - reset)
        violation = violation or errors[joint_id] > tolerance
    return {
        "joint_positions": positions,
        "joint_velocities_per_s": velocities,
        "locked_joint_absolute_errors": errors,
        "locked_joint_containment_violation": violation,
    }


def read_native_task_arena_object_reset_state(
    built: NativeTaskArenaEnvironment,
    *,
    joint_tolerance_rad: float = 1.0e-4,
) -> dict[str, Any]:
    """Read and qualify every replacement root/joint reset from native state.

    The active task subject and every inactive replacement are intentionally
    treated alike.  This prevents an inactive asset from drifting across task
    resets while the active subject alone appears reproducible.
    """

    env = getattr(built.env, "unwrapped", built.env)
    scene = getattr(env, "scene", None)
    if scene is None:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_scene_readback_missing"]
        )
    state_binding = built.plan.get("task_state_binding") or {}
    task_spec = built.plan.get("task_spec") or {}
    try:
        translation_tolerance = float(
            state_binding.get("root_translation_tolerance_m")
            or task_spec["reset_translation_tolerance_m"]
        )
        orientation_tolerance = float(
            state_binding.get("root_orientation_tolerance_rad")
            or task_spec["reset_orientation_tolerance_rad"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_reset_tolerance_missing"]
        ) from exc

    rows: list[dict[str, Any]] = []
    for planned in built.plan["objects"]:
        if not (
            planned.get("task_subject") is True
            or planned.get("semantic_role") in {"task_object", "replacement"}
        ):
            continue
        runtime_name = str(planned.get("name") or "")
        try:
            asset = scene[built.scene_asset_names[runtime_name]]
        except (KeyError, TypeError) as exc:
            raise NativeTaskArenaReadbackError(
                [f"native_task_arena_reset_asset_missing:{runtime_name}"]
            ) from exc
        data = getattr(asset, "data", None)
        root_pose = _first_environment(
            getattr(data, "root_pose_w", None),
            error=f"native_task_arena_reset_root_pose_missing:{runtime_name}",
        )
        if len(root_pose) < 7:
            raise NativeTaskArenaReadbackError(
                [f"native_task_arena_reset_root_pose_invalid:{runtime_name}"]
            )
        observed_pose = {
            "position_world_m": [float(item) for item in root_pose[:3]],
            "orientation_xyzw": _native_wxyz_to_xyzw(root_pose[3:7]),
        }
        reset_state = planned.get("reset_state") or {}
        expected_pose = reset_state.get("root_pose_world") or planned["pose_world"]
        translation_error = math.dist(
            observed_pose["position_world_m"], expected_pose["position_world_m"]
        )
        orientation_error = _quaternion_angle_xyzw(
            observed_pose["orientation_xyzw"], expected_pose["orientation_xyzw"]
        )

        expected_joints = {
            str(name): float(position)
            for name, position in (reset_state.get("joint_positions") or {}).items()
        }
        joint_errors: dict[str, float] = {}
        missing_joint_names: list[str] = []
        unexpected_joint_names: list[str] = []
        if planned["object_type"] == "ARTICULATION":
            native_names = _names(
                getattr(asset, "joint_names", None)
                or getattr(data, "joint_names", None),
                error=f"native_task_arena_reset_joint_names_missing:{runtime_name}",
            )
            native_positions = _first_environment(
                getattr(data, "joint_pos", None),
                error=f"native_task_arena_reset_joint_positions_missing:{runtime_name}",
            )
            if len(native_positions) != len(native_names):
                raise NativeTaskArenaReadbackError(
                    [f"native_task_arena_reset_joint_state_invalid:{runtime_name}"]
                )
            observed_joints = dict(
                zip(native_names, (float(item) for item in native_positions), strict=True)
            )
            missing_joint_names = sorted(set(expected_joints) - set(observed_joints))
            unexpected_joint_names = sorted(set(observed_joints) - set(expected_joints))
            joint_errors = {
                name: abs(observed_joints[name] - expected)
                for name, expected in expected_joints.items()
                if name in observed_joints
            }

        passed = (
            translation_error <= translation_tolerance
            and orientation_error <= orientation_tolerance
            and not missing_joint_names
            and not unexpected_joint_names
            and max(joint_errors.values(), default=0.0) <= joint_tolerance_rad
        )
        rows.append(
            {
                "asset_id": planned.get("asset_id", runtime_name),
                "runtime_name": runtime_name,
                "task_subject": bool(planned.get("task_subject")),
                "object_type": planned["object_type"],
                "expected_root_pose_world": expected_pose,
                "observed_root_pose_world": observed_pose,
                "root_translation_error_m": translation_error,
                "root_orientation_error_rad": orientation_error,
                "joint_absolute_errors_rad": joint_errors,
                "missing_joint_names": missing_joint_names,
                "unexpected_joint_names": unexpected_joint_names,
                "passed": passed,
            }
        )
    if not rows:
        raise NativeTaskArenaReadbackError(
            ["native_task_arena_reset_assets_missing"]
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "root_translation_tolerance_m": translation_tolerance,
        "root_orientation_tolerance_rad": orientation_tolerance,
        "joint_tolerance_rad": joint_tolerance_rad,
        "objects": rows,
    }


def read_native_task_arena_scenario_parameters(
    built: NativeTaskArenaEnvironment,
) -> dict[str, Any]:
    """Compare every requested perturbation with native object/config state."""

    applications = list(
        (built.plan.get("scenario") or {}).get("parameter_applications") or []
    )
    env = getattr(built.env, "unwrapped", built.env)
    scene = getattr(env, "scene", None)
    configuration = built.native_configuration_readback or {}
    rows: list[dict[str, Any]] = []
    for application in applications:
        kind = application["readback_kind"]
        tolerance = float(application["application_tolerance"])
        expected = application["expected_native_value"]
        if kind.startswith("task_subject_root_"):
            if scene is None:
                raise NativeTaskArenaReadbackError(
                    ["native_task_arena_scene_readback_missing"]
                )
            runtime_name = application["runtime_name"]
            try:
                asset = scene[built.scene_asset_names[runtime_name]]
            except (KeyError, TypeError) as exc:
                raise NativeTaskArenaReadbackError(
                    [f"native_task_arena_scenario_asset_missing:{runtime_name}"]
                ) from exc
            pose = _first_environment(
                getattr(getattr(asset, "data", None), "root_pose_w", None),
                error=f"native_task_arena_scenario_root_pose_missing:{runtime_name}",
            )
            if kind == "task_subject_root_position_y_m":
                observed: Any = float(pose[1])
                error = abs(observed - float(expected))
            else:
                observed = _native_wxyz_to_xyzw(pose[3:7])
                error = _quaternion_angle_xyzw(observed, expected)
                tolerance = math.radians(tolerance)
        elif kind == "camera_offset_position_x_m":
            role = application["camera_role"]
            try:
                observed = float(configuration["cameras"][role]["offset_position_m"][0])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise NativeTaskArenaReadbackError(
                    [f"native_task_arena_scenario_camera_readback_missing:{role}"]
                ) from exc
            error = abs(observed - float(expected))
        else:
            raise NativeTaskArenaReadbackError(
                [f"native_task_arena_scenario_readback_kind_invalid:{kind}"]
            )
        rows.append(
            {
                "parameter_id": application["parameter_id"],
                "runtime_target": application["runtime_target"],
                "unit": application["unit"],
                "requested_resolved_value": application["resolved_value"],
                "expected_native_value": expected,
                "observed_native_value": observed,
                "absolute_error_native_unit": error,
                "application_tolerance_native_unit": tolerance,
                "passed": error <= tolerance,
            }
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "requested_parameter_count": len(applications),
        "parameters": rows,
        "native_readback_required": True,
    }


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
        native_task_root_pose = _first_environment(
            getattr(task_data, "root_pose_w", None),
            error="native_task_arena_task_root_pose_missing",
        )[:7]
        task_root_pose = [
            *[float(value) for value in native_task_root_pose[:3]],
            *_native_wxyz_to_xyzw(native_task_root_pose[3:7]),
        ]

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
        if grasp_frame.get("kind") != "body_midpoint":
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_grasp_frame_invalid"]
            )
        finger_positions = [
            _body_position(
                robot,
                body_name=body_name,
                error="native_task_arena_grasp_body_missing",
            )[0]
            for body_name in grasp_frame["body_names"]
        ]
        if len(finger_positions) != 2:
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_grasp_frame_invalid"]
            )
        grasp_position = [
            (finger_positions[0][axis] + finger_positions[1][axis]) / 2.0
            for axis in range(3)
        ]

        articulation = self._built.plan["articulation"]
        graph_articulation = articulation.get("graph_articulation") is True
        link_position, link_quaternion = _body_position(
            task_object,
            body_name=(
                articulation["interaction_link_native_body_name"]
                if graph_articulation
                else articulation["moving_link_native_body_name"]
            ),
            error=(
                "native_task_arena_interaction_link_missing"
                if graph_articulation
                else "native_task_arena_moving_link_missing"
            ),
        )
        rotated_handle = _quaternion_rotate_xyzw(
            link_quaternion,
            (
                articulation["contact_point_link_m"]
                if graph_articulation
                else articulation["handle_grasp_point_link_m"]
            ),
        )
        handle_position = [
            link_position[axis] + rotated_handle[axis] for axis in range(3)
        ]
        reset_object = next(
            row
            for row in self._built.plan["objects"]
            if row.get("task_subject") is True
            or row.get("semantic_role") == "task_object"
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
            robot_task_forbidden_contact_forces_w_n=(
                sensor_forces.get("robot_task_forbidden_collision")
            ),
            task_root_pose_world=task_root_pose,
            task_root_reset_pose_world=reset_pose,
            grasp_frame_position_world_m=grasp_position,
            handle_reference_position_world_m=handle_position,
        )


class NativeRigidTaskArenaReadback:
    """Read rigid root pose and exact contact channels from one Arena build."""

    def __init__(self, built: NativeTaskArenaEnvironment):
        self._built = built
        if built.plan.get("task_kind") != "rigid_pick_place":
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
        native_pose = _first_environment(
            getattr(getattr(task_object, "data", None), "root_pose_w", None),
            error="native_task_arena_task_root_pose_missing",
        )
        if len(native_pose) < 7:
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_task_root_pose_missing"]
            )
        contact_peaks: dict[str, float] = {}
        for logical_sensor_id in (
            "task_robot_contact",
            "task_support_contact",
            "task_scene_collision",
            "robot_task_forbidden_collision",
            "robot_scene_contact",
        ):
            scene_names = self._built.contact_sensor_names.get(logical_sensor_id)
            if (
                logical_sensor_id == "task_scene_collision"
                and not scene_names
                and not self._built.plan["articulation"].get(
                    "non_support_scene_contact_body_paths"
                )
            ):
                contact_peaks[logical_sensor_id] = 0.0
                continue
            if (
                logical_sensor_id == "robot_task_forbidden_collision"
                and not scene_names
                and not self._built.plan["articulation"].get(
                    "forbidden_robot_contact_body_paths"
                )
                and "collision_failure_minimum_force_n"
                not in (self._built.plan.get("task_spec") or {})
            ):
                contact_peaks[logical_sensor_id] = 0.0
                continue
            if isinstance(scene_names, str) or not scene_names:
                raise NativeTaskArenaReadbackError(
                    [f"native_task_arena_contact_sensor_missing:{logical_sensor_id}"]
                )
            vectors: list[list[float]] = []
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
                vectors.extend(
                    _force_vectors(
                        getattr(getattr(sensor, "data", None), "force_matrix_w", None),
                        sensor_id=f"{logical_sensor_id}:{scene_name}",
                    )
                )
            contact_peaks[logical_sensor_id] = max(
                math.sqrt(sum(component * component for component in vector))
                for vector in vectors
            )
        grasp_frame = self._built.plan["robot"]["grasp_frame"]
        finger_positions = [
            _body_position(
                robot,
                body_name=body_name,
                error="native_task_arena_grasp_body_missing",
            )[0]
            for body_name in grasp_frame.get("body_names") or []
        ]
        if len(finger_positions) != 2:
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_grasp_frame_invalid"]
            )
        asset_root_pose = [
                *[float(value) for value in native_pose[:3]],
                *_native_wxyz_to_xyzw(native_pose[3:7]),
            ]
        affordance = (self._built.plan.get("task_spec") or {}).get(
            "interaction_affordance"
        )
        transform = (
            affordance.get("asset_root_from_scoring_frame")
            if isinstance(affordance, dict)
            else None
        )
        if not isinstance(transform, dict):
            raise NativeTaskArenaReadbackError(
                ["native_task_arena_scoring_frame_transform_missing"]
            )
        scoring_pose = _compose_pose_xyzw(
            asset_root_pose,
            [
                *[float(value) for value in transform.get("position_m") or []],
                *[
                    float(value)
                    for value in transform.get("orientation_xyzw") or []
                ],
            ],
        )
        joint_state = _read_bound_locked_joint_state(
            asset=task_object,
            sample_binding=self._built.plan.get("task_sample_binding") or {},
            task_spec=self._built.plan.get("task_spec") or {},
        )
        return {
            "asset_root_pose_world": asset_root_pose,
            "task_scoring_pose_world": scoring_pose,
            # The shared rigid scorer consumes this compatibility key.  It is
            # populated only after the explicit asset-root -> scoring-frame
            # transform above succeeds.
            "task_object_pose_world": scoring_pose,
            "grasp_frame_position_world_m": [
                (finger_positions[0][axis] + finger_positions[1][axis]) / 2.0
                for axis in range(3)
            ],
            "finger_separation_m": math.dist(finger_positions[0], finger_positions[1]),
            "task_robot_contact_peak_force_n": contact_peaks["task_robot_contact"],
            "task_support_contact_peak_force_n": contact_peaks[
                "task_support_contact"
            ],
            "task_scene_collision_peak_force_n": contact_peaks[
                "task_scene_collision"
            ],
            "robot_scene_contact_peak_force_n": contact_peaks["robot_scene_contact"],
            "robot_task_forbidden_collision_peak_force_n": contact_peaks[
                "robot_task_forbidden_collision"
            ],
            **joint_state,
            "measurement_authority": "native_rigid_root_pose_and_filtered_contact_sensors",
        }


__all__ = [
    "NativeArticulatedTaskArenaReadback",
    "NativeRigidTaskArenaReadback",
    "NativeTaskArenaReadbackError",
    "read_native_task_arena_object_reset_state",
    "read_native_task_arena_scenario_parameters",
]
