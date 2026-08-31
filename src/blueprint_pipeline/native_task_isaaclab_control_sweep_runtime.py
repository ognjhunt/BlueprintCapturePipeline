"""Native tensor readback for lightweight vectorized Isaac Lab control search."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .rigid_frame_transforms import (
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    rotate_vector_xyzw,
)
from .task_evaluation_isaaclab_control_sweep import (
    build_isaaclab_control_search_outcome,
    compile_isaaclab_control_sweep_wave_commands,
)


class NativeIsaacLabControlSweepRuntimeError(ValueError):
    """The vector environment did not expose the required native tensors."""


def _array(value: Any, *, blocker: str) -> np.ndarray:
    candidate = value
    for method in ("detach", "cpu"):
        operation = getattr(candidate, method, None)
        if callable(operation):
            candidate = operation()
    operation = getattr(candidate, "numpy", None)
    if callable(operation):
        candidate = operation()
    try:
        result = np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NativeIsaacLabControlSweepRuntimeError(blocker) from exc
    if not np.isfinite(result).all():
        raise NativeIsaacLabControlSweepRuntimeError(blocker)
    return result


class NativeIsaacLabControlSweepTraceReader:
    """Read clone-local task, joint, and contact state without grading it."""

    def __init__(self, built: Any):
        self._built = built
        env = getattr(built.env, "unwrapped", built.env)
        scene = getattr(env, "scene", None)
        if scene is None:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_scene_missing"
            )
        self._scene = scene
        try:
            self._task_object = scene[built.scene_asset_names["task_object"]]
            self._robot = scene["robot"]
        except (KeyError, TypeError) as exc:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_scene_missing"
            ) from exc
        origins = _array(
            getattr(scene, "env_origins", None),
            blocker="control_search_native_env_origins_invalid",
        )
        if origins.ndim != 2 or origins.shape[1] != 3 or origins.shape[0] < 1:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_env_origins_invalid"
            )
        self._origins = origins

    @property
    def environment_count(self) -> int:
        return int(self._origins.shape[0])

    def scoring_positions_world_m(self) -> list[list[float]]:
        """Return registered-scene scoring positions, not clone-offset roots."""

        root_poses = _array(
            getattr(getattr(self._task_object, "data", None), "root_pose_w", None),
            blocker="control_search_native_task_pose_invalid",
        )
        if (
            root_poses.ndim != 2
            or root_poses.shape[0] != self.environment_count
            or root_poses.shape[1] < 7
        ):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_task_pose_invalid"
            )
        affordance = (self._built.plan.get("task_spec") or {}).get(
            "interaction_affordance"
        )
        transform = (
            affordance.get("asset_root_from_scoring_frame")
            if isinstance(affordance, Mapping)
            else None
        )
        if not isinstance(transform, Mapping):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_scoring_transform_invalid"
            )
        try:
            offset_position = [
                float(value) for value in transform["position_m"]
            ]
            offset_orientation = [
                float(value) for value in transform["orientation_xyzw"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_scoring_transform_invalid"
            ) from exc
        if len(offset_position) != 3 or len(offset_orientation) != 4:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_scoring_transform_invalid"
            )
        results = []
        for index, pose in enumerate(root_poses):
            asset_position = [
                float(pose[axis] - self._origins[index, axis])
                for axis in range(3)
            ]
            asset_orientation = [float(value) for value in pose[3:7]]
            scoring_orientation = quaternion_multiply_xyzw(
                asset_orientation,
                quaternion_conjugate_xyzw(offset_orientation),
            )
            rotated_offset = rotate_vector_xyzw(
                scoring_orientation, offset_position
            )
            results.append(
                [
                    asset_position[axis] - rotated_offset[axis]
                    for axis in range(3)
                ]
            )
        return results

    def arm_joint_positions_rad(
        self, *, arm_joint_names: Sequence[str]
    ) -> list[list[float]]:
        if (
            len(arm_joint_names) != 7
            or len(set(arm_joint_names)) != 7
            or any(not isinstance(name, str) or not name for name in arm_joint_names)
        ):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_arm_joints_invalid"
            )
        try:
            indices = [list(self._robot.joint_names).index(name) for name in arm_joint_names]
        except (AttributeError, ValueError) as exc:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_arm_joints_invalid"
            ) from exc
        positions = _array(
            getattr(getattr(self._robot, "data", None), "joint_pos", None),
            blocker="control_search_native_arm_joints_invalid",
        )
        if positions.ndim != 2 or positions.shape[0] != self.environment_count:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_arm_joints_invalid"
            )
        return positions[:, indices].tolist()

    def peak_contact_force_vectors_w_n(
        self, *, logical_sensor_ids: Sequence[str]
    ) -> list[list[float]]:
        """Return each clone's strongest exact sensor vector across channels."""

        if not logical_sensor_ids or any(
            not isinstance(value, str) or not value for value in logical_sensor_ids
        ):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_contact_channels_invalid"
            )
        strongest = np.zeros((self.environment_count, 3), dtype=np.float64)
        strongest_norm = np.zeros(self.environment_count, dtype=np.float64)
        for logical_sensor_id in logical_sensor_ids:
            scene_names = self._built.contact_sensor_names.get(logical_sensor_id)
            if isinstance(scene_names, str) or not scene_names:
                raise NativeIsaacLabControlSweepRuntimeError(
                    "control_search_native_contact_channels_invalid"
                )
            for scene_name in scene_names:
                try:
                    sensor = self._scene[scene_name]
                except (KeyError, TypeError) as exc:
                    raise NativeIsaacLabControlSweepRuntimeError(
                        "control_search_native_contact_channels_invalid"
                    ) from exc
                forces = _array(
                    getattr(getattr(sensor, "data", None), "force_matrix_w", None),
                    blocker="control_search_native_contact_tensor_invalid",
                )
                if (
                    forces.shape[0] != self.environment_count
                    or forces.shape[-1] != 3
                ):
                    raise NativeIsaacLabControlSweepRuntimeError(
                        "control_search_native_contact_tensor_invalid"
                    )
                flattened = forces.reshape(self.environment_count, -1, 3)
                norms = np.linalg.norm(flattened, axis=-1)
                selected = norms.argmax(axis=1)
                vectors = flattened[np.arange(self.environment_count), selected]
                selected_norms = norms[np.arange(self.environment_count), selected]
                replace = selected_norms > strongest_norm
                strongest[replace] = vectors[replace]
                strongest_norm[replace] = selected_norms[replace]
        if not np.isfinite(strongest).all() or any(
            value < 0.0 for value in strongest_norm
        ):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_contact_tensor_invalid"
            )
        return strongest.tolist()


class NativeIsaacLabControlSweepWaveRunner:
    """Apply one clone-indexed cuRobo wave and retain raw native measurements."""

    def __init__(
        self,
        *,
        plan: Mapping[str, Any],
        schedule: Mapping[str, Any],
        gripper_open_command: float,
        gripper_closed_command: float,
        steps_per_waypoint: int = 4,
        settle_steps: int = 8,
        torch_module: Any | None = None,
        peak_gpu_memory_probe: Callable[[], int] | None = None,
    ) -> None:
        try:
            opened = float(gripper_open_command)
            closed = float(gripper_closed_command)
        except (TypeError, ValueError) as exc:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_gripper_commands_invalid"
            ) from exc
        if (
            not np.isfinite([opened, closed]).all()
            or opened == closed
            or not isinstance(steps_per_waypoint, int)
            or isinstance(steps_per_waypoint, bool)
            or not 1 <= steps_per_waypoint <= 64
            or not isinstance(settle_steps, int)
            or isinstance(settle_steps, bool)
            or not 1 <= settle_steps <= 128
        ):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_gripper_commands_invalid"
            )
        self._plan = dict(plan)
        self._schedule = dict(schedule)
        self._open = opened
        self._closed = closed
        self._steps_per_waypoint = steps_per_waypoint
        self._settle_steps = settle_steps
        self._torch_module = torch_module
        self._peak_gpu_memory_probe = peak_gpu_memory_probe

    def _torch(self) -> Any:
        if self._torch_module is not None:
            return self._torch_module
        try:
            import torch
        except ImportError as exc:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_torch_unavailable"
            ) from exc
        return torch

    @staticmethod
    def _arm_joint_names(robot: Any) -> list[str]:
        names = [str(name) for name in getattr(robot, "joint_names", ())]
        arm = [f"panda_joint{index}" for index in range(1, 8)]
        if any(name not in names for name in arm):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_arm_joints_invalid"
            )
        return arm

    @staticmethod
    def _target_position(built: Any) -> list[float]:
        value = (built.plan.get("task_spec") or {}).get(
            "target_position_world_m"
        )
        try:
            target = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_target_invalid"
            ) from exc
        if len(target) != 3 or not np.isfinite(target).all():
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_target_invalid"
            )
        return target

    @staticmethod
    def _start_position(built: Any) -> list[float]:
        value = (built.plan.get("task_spec") or {}).get("start_pose_world")
        try:
            start = [float(item) for item in value[:3]]
        except (TypeError, ValueError) as exc:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_target_invalid"
            ) from exc
        if len(start) != 3 or not np.isfinite(start).all():
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_target_invalid"
            )
        return start

    def _peak_memory(self, torch: Any) -> int:
        if self._peak_gpu_memory_probe is not None:
            value = self._peak_gpu_memory_probe()
        else:
            cuda = getattr(torch, "cuda", None)
            probe = getattr(cuda, "max_memory_allocated", None)
            if not callable(probe):
                raise NativeIsaacLabControlSweepRuntimeError(
                    "control_search_native_gpu_memory_unavailable"
                )
            value = probe()
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_gpu_memory_unavailable"
            )
        return value

    def __call__(
        self,
        *,
        built: Any,
        wave: Mapping[str, Any],
        candidate_inventory: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        if plan.get("plan_digest") != self._plan.get("plan_digest"):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_plan_mismatch"
            )
        torch = self._torch()
        env = getattr(built.env, "unwrapped", built.env)
        scene = getattr(env, "scene", None)
        if scene is None:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_scene_missing"
            )
        try:
            robot = scene["robot"]
        except (KeyError, TypeError) as exc:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_scene_missing"
            ) from exc
        arm_joint_names = self._arm_joint_names(robot)
        commands = compile_isaaclab_control_sweep_wave_commands(
            plan=self._plan,
            schedule=self._schedule,
            candidate_inventory=candidate_inventory,
            wave_index=int(wave["wave_index"]),
            arm_joint_names=arm_joint_names,
        )
        reader = NativeIsaacLabControlSweepTraceReader(built)
        if reader.environment_count != commands["vector_env_count"]:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_environment_count_mismatch"
            )
        device = getattr(env, "device", None)
        if device is None:
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_device_missing"
            )
        reset_seed = min(
            int(row["resolved_seed"]) for row in commands["assignments"]
        )
        env.reset(seed=reset_seed)
        origins = torch.as_tensor(scene.env_origins, device=device, dtype=torch.float32)
        root_poses = torch.as_tensor(robot.data.root_pose_w, device=device).clone()
        joint_positions = torch.as_tensor(robot.data.joint_pos, device=device).clone()
        joint_velocities = torch.zeros_like(joint_positions)
        joint_names = [str(name) for name in robot.joint_names]
        arm_indices = [joint_names.index(name) for name in arm_joint_names]
        active_ids = [
            int(row["environment_index"]) for row in commands["assignments"]
        ]
        env_ids = torch.as_tensor(active_ids, device=device, dtype=torch.long)
        for row in commands["assignments"]:
            index = int(row["environment_index"])
            base = row["robot_base_pose_world"]
            root_poses[index, :3] = torch.as_tensor(
                base["position_world_m"], device=device, dtype=root_poses.dtype
            ) + origins[index]
            root_poses[index, 3:7] = torch.as_tensor(
                base["orientation_xyzw"], device=device, dtype=root_poses.dtype
            )
            joint_positions[index, arm_indices] = torch.as_tensor(
                row["robot_joint_reset_positions_rad"],
                device=device,
                dtype=joint_positions.dtype,
            )
        write_root = getattr(robot, "write_root_pose_to_sim", None)
        write_joints = getattr(robot, "write_joint_state_to_sim", None)
        if not callable(write_root) or not callable(write_joints):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_state_write_unavailable"
            )
        write_root(root_poses[env_ids], env_ids=env_ids)
        write_joints(
            joint_positions[env_ids],
            joint_velocities[env_ids],
            env_ids=env_ids,
        )
        observed_joints = reader.arm_joint_positions_rad(
            arm_joint_names=arm_joint_names
        )
        observed_positions = reader.scoring_positions_world_m()
        expected_start = self._start_position(built)
        reset_tolerance = float(
            (built.plan.get("task_spec") or {}).get(
                "reset_translation_tolerance_m", 0.002
            )
        )
        reset_passed: dict[int, bool] = {}
        for row in commands["assignments"]:
            index = int(row["environment_index"])
            joint_error = max(
                abs(observed - expected)
                for observed, expected in zip(
                    observed_joints[index],
                    row["robot_joint_reset_positions_rad"],
                    strict=True,
                )
            )
            position_error = math.dist(observed_positions[index], expected_start)
            reset_passed[index] = (
                joint_error <= 1.0e-3 and position_error <= reset_tolerance
            )
        traces: dict[int, dict[str, list[Any]]] = {
            index: {
                "positions": [observed_positions[index]],
                "forbidden": [[0.0, 0.0, 0.0]],
                "required": [[0.0, 0.0, 0.0]],
                "stages": ["reset"],
            }
            for index in active_ids
        }
        action = torch.as_tensor(robot.data.joint_pos, device=device)[:, arm_indices]
        action = torch.cat(
            (
                action,
                torch.full(
                    (reader.environment_count, 1),
                    self._open,
                    device=device,
                    dtype=action.dtype,
                ),
            ),
            dim=1,
        )
        forbidden_channels = tuple(
            name
            for name in (
                "robot_scene_contact",
                "robot_task_forbidden_collision",
                "task_scene_collision",
            )
            if built.contact_sensor_names.get(name)
        )
        if not forbidden_channels or not built.contact_sensor_names.get(
            "task_robot_contact"
        ):
            raise NativeIsaacLabControlSweepRuntimeError(
                "control_search_native_contact_channels_invalid"
            )

        def sample(stage_by_environment: Mapping[int, str]) -> None:
            positions = reader.scoring_positions_world_m()
            forbidden = reader.peak_contact_force_vectors_w_n(
                logical_sensor_ids=forbidden_channels
            )
            required = reader.peak_contact_force_vectors_w_n(
                logical_sensor_ids=("task_robot_contact",)
            )
            for environment_index, stage in stage_by_environment.items():
                traces[environment_index]["positions"].append(
                    positions[environment_index]
                )
                traces[environment_index]["forbidden"].append(
                    forbidden[environment_index]
                )
                traces[environment_index]["required"].append(
                    required[environment_index]
                )
                traces[environment_index]["stages"].append(stage)

        for waypoint_index in range(commands["maximum_waypoint_count"]):
            stages: dict[int, str] = {}
            for row in commands["assignments"]:
                environment_index = int(row["environment_index"])
                waypoint = row["waypoints"][
                    min(waypoint_index, row["waypoint_count"] - 1)
                ]
                action[environment_index, :7] = torch.as_tensor(
                    waypoint["arm_joint_positions_rad"],
                    device=device,
                    dtype=action.dtype,
                )
                action[environment_index, 7] = (
                    self._closed
                    if waypoint["gripper_state"] == "closed"
                    else self._open
                )
                stages[environment_index] = waypoint["stage_kind"]
            for _ in range(self._steps_per_waypoint):
                env.step(action)
                sample(stages)
        action[:, 7] = self._open
        settle_stages = {index: "settle" for index in active_ids}
        for _ in range(self._settle_steps):
            env.step(action)
            sample(settle_stages)
        target = self._target_position(built)
        contact_threshold = float(
            (built.plan.get("task_spec") or {}).get(
                "task_contact_minimum_force_n", 0.5
            )
        )
        outcomes = []
        assignments = {
            int(row["environment_index"]): row for row in commands["assignments"]
        }
        for environment_index in active_ids:
            trace = traces[environment_index]
            outcomes.append(
                build_isaaclab_control_search_outcome(
                    assignment=assignments[environment_index],
                    reset_readback_passed=reset_passed[environment_index],
                    task_position_trace_world_m=trace["positions"],
                    forbidden_contact_force_trace_w_n=trace["forbidden"],
                    required_contact_force_trace_w_n=trace["required"],
                    stage_kinds=trace["stages"],
                    target_position_world_m=target,
                    required_contact_minimum_force_n=contact_threshold,
                    settle_sample_count=self._settle_steps,
                )
            )
        return {
            "outcomes": outcomes,
            "peak_gpu_memory_bytes": self._peak_memory(torch),
        }


__all__ = [
    "NativeIsaacLabControlSweepRuntimeError",
    "NativeIsaacLabControlSweepTraceReader",
    "NativeIsaacLabControlSweepWaveRunner",
]
