"""Adapt the live Isaac environment to the episode loop's injected seam.

``adp009d_policy_episode`` deliberately never imports a simulator, so something
has to translate between it and Isaac.  That is this module, and it is kept
thin on purpose: every non-trivial decision -- observation format, action
semantics, scoring -- already lives in a tested module, and duplicating any of
it here would create a second place for those contracts to drift.

Two conversions are the substance of it, and both are measured facts rather
than conventions:

* **Camera frames.**  Isaac renders RGBA at 1280x720; the DROID views want
  uint8 RGB, which the observation adapter then resizes per candidate.  Alpha
  is dropped here rather than downstream because a constant alpha channel
  silently dominated a colour-signal check once already.
* **Gripper width.**  The scorer reads physical jaw opening in metres.  The
  Robotiq link origins are not jaw tips and can dynamically travel beyond the
  85 mm nameplate stroke, so their separation is affine-calibrated against the
  same run's measured closed/open convention probe.  The raw separation and a
  clamp flag remain in every object sample for audit.

Isaac imports happen inside methods, never at module import, so this file stays
importable -- and testable -- off-GPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:  # flat provider-bundle layout
    from adp009d_droid_observation import (
        DROID_EXTERIOR_VIEW_1,
        DROID_WRIST_VIEW,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_observation import (
        DROID_EXTERIOR_VIEW_1,
        DROID_WRIST_VIEW,
    )

ADAPTER_SCHEMA_VERSION = "adp009d_isaac_episode_adapter.v9"

# Isaac camera name -> the DROID view it serves.
CAMERA_VIEW_BINDING = {
    "external_camera": DROID_EXTERIOR_VIEW_1,
    "wrist_camera": DROID_WRIST_VIEW,
}
# Review-only cameras never enter ``read_policy_inputs``.  They are retained
# alongside the policy views so a human can understand the whole movement.
REVIEW_CAMERA_BINDING = {"external_camera_2": "overview"}
EVALUATION_CAMERA_BINDING = {
    "external_camera": "external",
    "wrist_camera": "wrist",
    **REVIEW_CAMERA_BINDING,
}
DEFAULT_CAMERA_SCENE_NAMES = {
    "external": "external_camera",
    "wrist": "wrist_camera",
    "overview": "external_camera_2",
}
# The two bodies whose separation is the gripper width, matching the
# convention probe so both read the same physical quantity.
FINGER_BODIES = ("left_inner_finger", "right_inner_finger")
# Ordered to match the already-measured approach controller.  ``base_link`` is
# the Robotiq tool body that carries the wrist camera in the live Arena asset.
END_EFFECTOR_BODY_CANDIDATES = ("panda_hand", "base_link", "panda_link7")
ARM_JOINT_COUNT = 7
# Frozen by the Robotiq 2F-85 task embodiment and pinned independently by the
# deterministic scorer.  A parity test keeps the flat-bundle duplicate honest.
GRIPPER_PHYSICAL_FULL_OPENING_M = 0.085
GRIPPER_WIDTH_SOURCE = "probe_calibrated_finger_body_separation"


class IsaacEpisodeAdapterError(RuntimeError):
    """Fail-closed adapter errors, raised before anything reaches the loop."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def bounded_absolute_joint_setpoint(
    *,
    measured_joint_positions_rad: Sequence[float],
    desired_joint_positions_rad: Sequence[float],
    previous_commanded_joint_positions_rad: Sequence[float],
    max_command_slew_per_step_rad: float,
    max_setpoint_lead_rad: float,
) -> list[float]:
    """Advance an absolute-position command without starving a slow actuator.

    ``max_command_slew_per_step_rad`` limits how far the command itself can move
    in one control step.  ``max_setpoint_lead_rad`` independently limits how far
    that command may get ahead of measured state.  Conflating those two limits
    kept the live v98 command permanently 0.03 rad from measured state: the
    actuator moved, but the target could never accumulate enough lead to reach
    the Cartesian goal within the frozen phase horizon.

    The returned value is the closest point to the desired IK solution in the
    intersection of both per-joint safety intervals.  An empty intersection is
    a reset/state discontinuity and fails closed rather than silently jumping.
    """

    try:
        measured = [float(value) for value in measured_joint_positions_rad]
        desired = [float(value) for value in desired_joint_positions_rad]
        previous = [float(value) for value in previous_commanded_joint_positions_rad]
        max_slew = float(max_command_slew_per_step_rad)
        max_lead = float(max_setpoint_lead_rad)
    except (TypeError, ValueError) as exc:
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_joint_setpoint_contract_invalid"]
        ) from exc
    if (
        not measured
        or len(measured) != len(desired)
        or len(measured) != len(previous)
        or not all(math.isfinite(value) for value in (*measured, *desired, *previous))
        or not math.isfinite(max_slew)
        or not math.isfinite(max_lead)
        or max_slew <= 0.0
        or max_lead < max_slew
    ):
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_joint_setpoint_contract_invalid"]
        )

    command: list[float] = []
    for measured_value, desired_value, previous_value in zip(
        measured, desired, previous, strict=True
    ):
        lower = max(previous_value - max_slew, measured_value - max_lead)
        upper = min(previous_value + max_slew, measured_value + max_lead)
        if lower > upper + 1.0e-12:
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_joint_setpoint_constraints_infeasible"]
            )
        command.append(min(max(desired_value, lower), upper))
    return command


def controlled_body_pose_for_grasp_frame_target(
    *,
    current_body_position_world_m: Sequence[float],
    current_body_quaternion_world_xyzw: Sequence[float],
    current_grasp_frame_position_world_m: Sequence[float],
    target_grasp_frame_position_world_m: Sequence[float],
    target_body_quaternion_world_xyzw: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Resolve the IK-body pose that puts the measured finger midpoint at target.

    The reset pose measures the complete body-to-finger-midpoint offset.  That
    offset is transformed into the controlled body's local frame, then applied
    at the task orientation.  This matters because the wrist-observability pose
    can point the tool offset away from the task: v89 made the pregrasp body
    target 0.93 m from the Franka base even though the finger target was within
    reach.  A scalar tool length or a world-space offset cannot represent this.
    """

    try:
        body = [float(value) for value in current_body_position_world_m]
        quaternion = [float(value) for value in current_body_quaternion_world_xyzw]
        grasp = [float(value) for value in current_grasp_frame_position_world_m]
        target = [float(value) for value in target_grasp_frame_position_world_m]
        target_quaternion = [
            float(value) for value in target_body_quaternion_world_xyzw
        ]
    except (TypeError, ValueError) as exc:
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_grasp_frame_transform_invalid"]
        ) from exc
    if (
        len(body) != 3
        or len(quaternion) != 4
        or len(grasp) != 3
        or len(target) != 3
        or len(target_quaternion) != 4
        or not all(
            math.isfinite(value)
            for value in (*body, *quaternion, *grasp, *target, *target_quaternion)
        )
        or abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0) > 1.0e-5
        or abs(
            math.sqrt(sum(value * value for value in target_quaternion)) - 1.0
        )
        > 1.0e-5
    ):
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_grasp_frame_transform_invalid"]
        )
    def _rotate(q: Sequence[float], vector: Sequence[float]) -> list[float]:
        x, y, z, w = q
        vx, vy, vz = vector
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ]

    body_to_grasp_world = [grasp[index] - body[index] for index in range(3)]
    body_to_grasp_local = _rotate(
        [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
        body_to_grasp_world,
    )
    target_body_to_grasp_world = _rotate(
        target_quaternion,
        body_to_grasp_local,
    )
    target_body = [
        target[index] - target_body_to_grasp_world[index] for index in range(3)
    ]
    return target_body, target_quaternion


def _as_array(value: Any) -> Any:
    """Whatever the simulator handed back, as a numpy array.

    Live Isaac returns a torch tensor; tests and replays hand back arrays.  This
    is the adapter boundary, so it accepts both rather than assuming one and
    failing obscurely on the other.
    """

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def rgb_from_camera_output(frame: Any) -> Any:
    """Isaac RGBA (or RGB) to contiguous uint8 RGB.

    Alpha is dropped here, at the boundary, because a constant alpha channel
    has already once dominated a statistic computed over all four channels and
    made a black render look like it carried signal.
    """

    import numpy as np

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise IsaacEpisodeAdapterError(
            [f"isaac_episode_camera_frame_shape_invalid:{tuple(array.shape)}"]
        )
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def rotation_row_major_from_quaternion_xyzw(quaternion: Sequence[float]) -> list[float]:
    """Convert exact pinned IsaacLab's normalized ``(x, y, z, w)`` quaternion."""

    values = [float(value) for value in quaternion]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise IsaacEpisodeAdapterError(["isaac_episode_end_effector_quaternion_invalid"])
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise IsaacEpisodeAdapterError(["isaac_episode_end_effector_quaternion_invalid"])
    x, y, z, w = (value / norm for value in values)
    return [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]


class IsaacEpisodeAdapter:
    """The live-simulator implementation of ``EpisodeEnvironment``."""

    def __init__(
        self,
        *,
        env: Any,
        robot: Any,
        rigid_task_object: Any = None,
        action_dim: int,
        reset_seed: int,
        to_torch: Any,
        gripper_closed_width_m: float,
        gripper_open_width_m: float,
        reset_callback: Callable[[], None] | None = None,
        scripted_pose_controller_reset_callback: Callable[[], None] | None = None,
        simulation_step_seconds: float | None = None,
        scripted_pose_action_callback: Callable[..., Sequence[float]] | None = None,
        camera_pose_callback: Callable[
            [str], tuple[Sequence[float], Sequence[float]] | None
        ]
        | None = None,
        task_sample_callback: Callable[[], Mapping[str, Any]] | None = None,
        camera_scene_names: Mapping[str, str] | None = None,
    ) -> None:
        self._env = env
        self._robot = robot
        self._rigid_object = rigid_task_object
        self._action_dim = int(action_dim)
        self._reset_seed = int(reset_seed)
        self._to_torch = to_torch
        self._gripper_closed_width_m = float(gripper_closed_width_m)
        self._gripper_open_width_m = float(gripper_open_width_m)
        self._reset_callback = reset_callback
        self._scripted_pose_controller_reset_callback = (
            scripted_pose_controller_reset_callback
        )
        self._simulation_step_seconds = (
            None
            if simulation_step_seconds is None
            else float(simulation_step_seconds)
        )
        self._scripted_pose_action_callback = scripted_pose_action_callback
        self._camera_pose_callback = camera_pose_callback
        self._task_sample_callback = task_sample_callback
        self._camera_scene_names = dict(
            DEFAULT_CAMERA_SCENE_NAMES
            if camera_scene_names is None
            else camera_scene_names
        )
        self._control_step_index = 0
        if self._rigid_object is None and self._task_sample_callback is None:
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_task_state_source_missing"]
            )
        if (
            not math.isfinite(self._gripper_closed_width_m)
            or not math.isfinite(self._gripper_open_width_m)
            or self._gripper_open_width_m - self._gripper_closed_width_m <= 1e-6
        ):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_gripper_width_calibration_invalid"]
            )
        required_camera_roles = set(DEFAULT_CAMERA_SCENE_NAMES)
        if (
            set(self._camera_scene_names) != required_camera_roles
            or any(
                not isinstance(scene_name, str) or not scene_name.strip()
                for scene_name in self._camera_scene_names.values()
            )
            or len(set(self._camera_scene_names.values()))
            != len(self._camera_scene_names)
        ):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_camera_scene_names_invalid"]
            )

        body_names = list(robot.data.body_names)
        missing = [name for name in FINGER_BODIES if name not in body_names]
        if missing:
            raise IsaacEpisodeAdapterError(
                [f"isaac_episode_finger_body_missing:{','.join(missing)}"]
            )
        self._finger_indices = [body_names.index(name) for name in FINGER_BODIES]
        end_effector_name = next(
            (name for name in END_EFFECTOR_BODY_CANDIDATES if name in body_names),
            None,
        )
        if end_effector_name is None:
            raise IsaacEpisodeAdapterError(["isaac_episode_end_effector_body_missing"])
        self._end_effector_name = end_effector_name
        self._end_effector_index = body_names.index(end_effector_name)

    # -- EpisodeEnvironment -------------------------------------------------

    def reset(self) -> None:
        if self._reset_callback is not None:
            self._reset_callback()
        else:
            self._env.reset(seed=self._reset_seed)
        if self._scripted_pose_controller_reset_callback is not None:
            self._scripted_pose_controller_reset_callback()
        self._control_step_index = 0

    def joint_limits(self) -> list[list[float]]:
        limits = self._to_torch(self._robot.data.joint_limits)[0, :ARM_JOINT_COUNT]
        return [[float(row[0]), float(row[1])] for row in limits]

    def read_policy_inputs(self) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for role, view in (
            ("external", DROID_EXTERIOR_VIEW_1),
            ("wrist", DROID_WRIST_VIEW),
        ):
            camera_name = self._camera_scene_names[role]
            camera = self._env.unwrapped.scene[camera_name]
            output = camera.data.output
            if "rgb" not in output:
                raise IsaacEpisodeAdapterError([f"isaac_episode_camera_rgb_missing:{camera_name}"])
            frame = _as_array(self._to_torch(output["rgb"]))[0]
            inputs[view] = rgb_from_camera_output(frame)
        inputs["joint_position"] = self.read_arm_joint_positions()
        inputs["gripper_position"] = self._droid_gripper_position()
        inputs["eef_9d"] = self._eef_9d()
        return inputs

    def read_evaluation_camera_inputs(self) -> dict[str, Any]:
        """Lossless policy views plus the review-only fixed overview stream."""

        images: dict[str, Any] = {}
        for camera_id, camera_name in self._camera_scene_names.items():
            try:
                camera = self._env.unwrapped.scene[camera_name]
            except (KeyError, TypeError) as exc:
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_evaluation_camera_missing:{camera_id}"]
                ) from exc
            output = camera.data.output
            if "rgb" not in output:
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_camera_rgb_missing:{camera_name}"]
                )
            frame = _as_array(self._to_torch(output["rgb"]))[0]
            images[camera_id] = rgb_from_camera_output(frame)
        return images

    def read_arm_joint_positions(self) -> list[float]:
        joints = self._to_torch(self._robot.data.joint_pos)[0, :ARM_JOINT_COUNT]
        return [float(value) for value in joints]

    def step(self, isaac_action: Sequence[float]) -> None:
        values = [float(v) for v in isaac_action]
        if len(values) != self._action_dim:
            raise IsaacEpisodeAdapterError(
                [f"isaac_episode_action_dim_mismatch:{len(values)}!={self._action_dim}"]
            )

        # Validate before importing the GPU runtime.  Besides producing the
        # precise contract error first, this keeps malformed-action checks
        # hermetic on hosts that deliberately do not install torch.
        import torch

        tensor = torch.tensor([values], device=self._env.unwrapped.device, dtype=torch.float32)
        self._env.step(tensor)
        self._control_step_index += 1

    def hold_action(self, *, gripper_command: float) -> list[float]:
        """Realize zero joint velocity in Arena's absolute-position action space."""

        command = float(gripper_command)
        if not math.isfinite(command):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_hold_gripper_command_invalid"]
            )
        return [*self.read_arm_joint_positions(), command]

    def scripted_action_for_pose(
        self,
        *,
        target_position_world_m: Sequence[float],
        target_quaternion_world_xyzw: Sequence[float] | None,
        gripper_command: float,
        max_joint_delta_rad: float,
        max_joint_setpoint_lead_rad: float,
    ) -> list[float]:
        """Resolve one deterministic pose-servo step through the injected native IK."""

        if self._scripted_pose_action_callback is None:
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_scripted_pose_controller_missing"]
            )
        values = self._scripted_pose_action_callback(
            target_position_world_m=[float(v) for v in target_position_world_m],
            target_quaternion_world_xyzw=(
                None
                if target_quaternion_world_xyzw is None
                else [float(v) for v in target_quaternion_world_xyzw]
            ),
            gripper_command=float(gripper_command),
            max_joint_delta_rad=float(max_joint_delta_rad),
            max_joint_setpoint_lead_rad=float(max_joint_setpoint_lead_rad),
        )
        action = [float(value) for value in values]
        if len(action) != self._action_dim or not all(
            math.isfinite(value) for value in action
        ):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_scripted_pose_action_invalid"]
            )
        return action

    def read_control_observation_metadata(self) -> dict[str, Any]:
        """Exact dual-camera calibration and deterministic episode timestamp."""

        if (
            self._simulation_step_seconds is None
            or not math.isfinite(self._simulation_step_seconds)
            or self._simulation_step_seconds <= 0.0
        ):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_simulation_step_seconds_missing"]
            )
        calibrations: dict[str, Any] = {}
        source_devices: dict[str, str] = {}
        synchronizations: dict[str, dict[str, Any]] = {}
        for camera_id, camera_name in self._camera_scene_names.items():
            camera = self._env.unwrapped.scene[camera_name]
            output = camera.data.output
            if "rgb" not in output:
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_camera_rgb_missing:{camera_name}"]
                )
            frame = _as_array(self._to_torch(output["rgb"]))[0]
            height, width = int(frame.shape[0]), int(frame.shape[1])
            intrinsic = _as_array(
                self._to_torch(camera.data.intrinsic_matrices)
            )[0]
            position = _as_array(self._to_torch(camera.data.pos_w))[0]
            quaternion = _as_array(self._to_torch(camera.data.quat_w_opengl))[0]
            world_pose_source = "isaac_sensor_buffer"
            if self._camera_pose_callback is not None:
                override = self._camera_pose_callback(camera_name)
                if override is not None:
                    position = _as_array(override[0])
                    quaternion = _as_array(override[1])
                    world_pose_source = "runtime_camera_pose_callback"
            if (
                position.shape != (3,)
                or quaternion.shape != (4,)
                or not all(math.isfinite(float(value)) for value in position)
                or not all(math.isfinite(float(value)) for value in quaternion)
                or abs(
                    math.sqrt(sum(float(value) ** 2 for value in quaternion))
                    - 1.0
                )
                > 1.0e-5
            ):
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_camera_world_pose_invalid:{camera_name}"]
                )
            rotation = rotation_row_major_from_quaternion_xyzw(quaternion)
            world_from_camera = [
                [rotation[row * 3 + column] for column in range(3)]
                + [float(position[row])]
                for row in range(3)
            ]
            world_from_camera.append([0.0, 0.0, 0.0, 1.0])
            clipping = getattr(getattr(camera, "cfg", None), "spawn", None)
            clipping_range = getattr(clipping, "clipping_range", None)
            if (
                not isinstance(clipping_range, Sequence)
                or len(clipping_range) != 2
            ):
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_camera_clipping_range_missing:{camera_name}"]
                )
            calibrations[camera_id] = {
                "camera_model": "pinhole",
                "intrinsic_matrix": [
                    [float(value) for value in row] for row in intrinsic
                ],
                "world_from_camera": world_from_camera,
                "resolution": [width, height],
                "near_m": float(clipping_range[0]),
                "far_m": float(clipping_range[1]),
                "world_pose_source": world_pose_source,
            }
            source_devices[camera_id] = str(
                getattr(output["rgb"], "device", self._env.unwrapped.device)
            )
            synchronizations[camera_id] = {
                "host_bytes_ready": True,
                "method": "environment_step_completed_before_read_only_host_copy",
            }
        simulation_time_s = self._control_step_index * self._simulation_step_seconds
        return {
            "timestamp_ns": int(round(simulation_time_s * 1_000_000_000)),
            "simulation_time_s": simulation_time_s,
            "calibrations": calibrations,
            "source_devices": source_devices,
            "synchronizations": synchronizations,
        }

    def read_object_sample(self) -> dict[str, Any]:
        if self._rigid_object is None:
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_rigid_task_object_missing"]
            )
        pose = self._to_torch(self._rigid_object.data.root_pose_w)[0]
        controlled_body_pose = self._to_torch(self._robot.data.body_pose_w)[
            0, self._end_effector_index, :7
        ]
        left, right = self._finger_positions()
        raw_separation = math.dist(left, right)
        width, unclamped_open_fraction, calibration_clamped = (
            self._calibrated_gripper_width(raw_separation)
        )
        sample: dict[str, Any] = {
            "can_pose_world": [float(v) for v in pose[:7]],
            "gripper_width_m": width,
            "gripper_body_separation_m": raw_separation,
            "gripper_width_open_fraction_unclamped": unclamped_open_fraction,
            "gripper_width_calibration_clamped": calibration_clamped,
            "controlled_body_name": self._end_effector_name,
            "controlled_body_pose_world": [
                float(value) for value in controlled_body_pose
            ],
        }
        sample["grasp_frame_position_world_m"] = [
            (left[axis] + right[axis]) / 2.0 for axis in range(3)
        ]
        return sample

    def read_task_sample(self) -> dict[str, Any]:
        """Read non-rigid task state through the runtime's native-state seam.

        The adapter does not know a generated asset's joint names or topology.
        The runtime resolves those from the sealed asset and supplies this
        callback; the task-neutral scorer independently validates every field.
        """

        if self._task_sample_callback is None:
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_task_sample_callback_missing"]
            )
        sample = self._task_sample_callback()
        if not isinstance(sample, Mapping):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_task_sample_callback_invalid"]
            )
        return dict(sample)

    # -- internals ----------------------------------------------------------

    def _finger_positions(self) -> tuple[list[float], list[float]]:
        poses = self._to_torch(self._robot.data.body_pose_w)[0]
        return (
            [float(poses[self._finger_indices[0]][axis]) for axis in range(3)],
            [float(poses[self._finger_indices[1]][axis]) for axis in range(3)],
        )

    def _raw_gripper_body_separation(self) -> float:
        # A Euclidean distance needs no tensor library, and keeping it in plain
        # arithmetic is what lets this adapter be tested without a GPU.
        left, right = self._finger_positions()
        return math.dist(left, right)

    def _calibrated_gripper_width(
        self, raw_separation_m: float
    ) -> tuple[float, float, bool]:
        """Map linkage separation onto the gripper's physical jaw stroke.

        The convention probe supplies the run-local endpoints.  Values outside
        them are linkage/dynamics overtravel rather than a jaw aperture wider
        than the 2F-85 can physically achieve, so the calibrated aperture is
        bounded while the unbounded fraction is retained for diagnosis.
        """

        span = self._gripper_open_width_m - self._gripper_closed_width_m
        unbounded = (float(raw_separation_m) - self._gripper_closed_width_m) / span
        bounded = min(1.0, max(0.0, unbounded))
        return (
            GRIPPER_PHYSICAL_FULL_OPENING_M * bounded,
            unbounded,
            not math.isclose(unbounded, bounded, rel_tol=0.0, abs_tol=1e-12),
        )

    def _droid_gripper_position(self) -> float:
        """DROID state convention: zero=open and one=closed."""

        _, open_fraction, _ = self._calibrated_gripper_width(
            self._raw_gripper_body_separation()
        )
        return min(1.0, max(0.0, 1.0 - open_fraction))

    def _eef_9d(self) -> Any:
        pose = self._to_torch(self._robot.data.body_pose_w)[
            0, self._end_effector_index
        ]
        values = [float(value) for value in pose[:7]]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise IsaacEpisodeAdapterError(["isaac_episode_end_effector_pose_invalid"])
        try:  # flat provider bundle
            from groot_n17_droid_policy_runtime import droid_eef_9d
        except ModuleNotFoundError:  # repository package
            from .groot_n17_droid_policy_runtime import droid_eef_9d
        return droid_eef_9d(
            position_m=values[:3],
            rotation_row_major=rotation_row_major_from_quaternion_xyzw(values[3:7]),
        )


def describe_adapter() -> dict[str, Any]:
    """Report the bindings this adapter applied, for the run receipt."""

    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "camera_view_binding": dict(CAMERA_VIEW_BINDING),
        "finger_bodies": list(FINGER_BODIES),
        "end_effector_body_candidates": list(END_EFFECTOR_BODY_CANDIDATES),
        "gripper_width_source": GRIPPER_WIDTH_SOURCE,
        "scripted_control_target_frame": "probe_calibrated_finger_midpoint",
        "scripted_control_body_pose_resolution": (
            "measured_body_local_to_finger_midpoint_applied_at_task_orientation"
        ),
        "scripted_control_physx_jacobian_frame": "world",
        "scripted_control_controller_error_frame": "robot_root",
        "scripted_control_jacobian_frame_transform": (
            "rotate_linear_and_angular_rows_world_to_robot_root"
        ),
        "gripper_physical_full_opening_m": GRIPPER_PHYSICAL_FULL_OPENING_M,
        "raw_gripper_body_separation_retained": True,
        "gripper_width_calibration_clamp_retained": True,
        "camera_alpha_dropped_at_boundary": True,
        "arm_joint_count": ARM_JOINT_COUNT,
        "isaaclab_pose_quaternion_order": "xyzw",
    }


def validate_adapter_bindings(bindings: Mapping[str, Any]) -> list[str]:
    """Check a retained adapter description still matches this contract."""

    errors: list[str] = []
    if bindings.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        errors.append("isaac_episode_adapter_schema_version_unexpected")
    if dict(bindings.get("camera_view_binding") or {}) != dict(CAMERA_VIEW_BINDING):
        errors.append("isaac_episode_adapter_camera_binding_drifted")
    if list(bindings.get("finger_bodies") or []) != list(FINGER_BODIES):
        errors.append("isaac_episode_adapter_finger_bodies_drifted")
    if list(bindings.get("end_effector_body_candidates") or []) != list(
        END_EFFECTOR_BODY_CANDIDATES
    ):
        errors.append("isaac_episode_adapter_end_effector_binding_drifted")
    if bindings.get("gripper_width_source") != GRIPPER_WIDTH_SOURCE:
        errors.append("isaac_episode_adapter_gripper_width_source_drifted")
    if bindings.get("scripted_control_target_frame") != (
        "probe_calibrated_finger_midpoint"
    ):
        errors.append("isaac_episode_adapter_scripted_control_target_frame_drifted")
    if bindings.get("scripted_control_body_pose_resolution") != (
        "measured_body_local_to_finger_midpoint_applied_at_task_orientation"
    ):
        errors.append("isaac_episode_adapter_scripted_control_body_pose_resolution_drifted")
    if bindings.get("scripted_control_physx_jacobian_frame") != "world":
        errors.append("isaac_episode_adapter_physx_jacobian_frame_drifted")
    if bindings.get("scripted_control_controller_error_frame") != "robot_root":
        errors.append("isaac_episode_adapter_controller_error_frame_drifted")
    if bindings.get("scripted_control_jacobian_frame_transform") != (
        "rotate_linear_and_angular_rows_world_to_robot_root"
    ):
        errors.append("isaac_episode_adapter_jacobian_frame_transform_drifted")
    if (
        bindings.get("gripper_physical_full_opening_m")
        != GRIPPER_PHYSICAL_FULL_OPENING_M
    ):
        errors.append("isaac_episode_adapter_gripper_stroke_drifted")
    if bindings.get("raw_gripper_body_separation_retained") is not True:
        errors.append("isaac_episode_adapter_raw_gripper_measurement_not_retained")
    if bindings.get("gripper_width_calibration_clamp_retained") is not True:
        errors.append("isaac_episode_adapter_gripper_clamp_not_retained")
    if bindings.get("camera_alpha_dropped_at_boundary") is not True:
        errors.append("isaac_episode_adapter_alpha_not_dropped")
    if bindings.get("isaaclab_pose_quaternion_order") != "xyzw":
        errors.append("isaac_episode_adapter_quaternion_order_drifted")
    return sorted(set(errors))


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "CAMERA_VIEW_BINDING",
    "DEFAULT_CAMERA_SCENE_NAMES",
    "END_EFFECTOR_BODY_CANDIDATES",
    "FINGER_BODIES",
    "GRIPPER_PHYSICAL_FULL_OPENING_M",
    "GRIPPER_WIDTH_SOURCE",
    "IsaacEpisodeAdapter",
    "IsaacEpisodeAdapterError",
    "bounded_absolute_joint_setpoint",
    "controlled_body_pose_for_grasp_frame_target",
    "describe_adapter",
    "rgb_from_camera_output",
    "rotation_row_major_from_quaternion_xyzw",
    "validate_adapter_bindings",
]
