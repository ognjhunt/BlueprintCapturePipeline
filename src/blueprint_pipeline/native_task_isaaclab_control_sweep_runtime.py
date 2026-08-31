"""Native tensor readback for lightweight vectorized Isaac Lab control search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .rigid_frame_transforms import (
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    rotate_vector_xyzw,
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


__all__ = [
    "NativeIsaacLabControlSweepRuntimeError",
    "NativeIsaacLabControlSweepTraceReader",
]
