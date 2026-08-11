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

ADAPTER_SCHEMA_VERSION = "adp009d_isaac_episode_adapter.v10"

DIRECT_GLOBAL_POSE_TARGET = "direct_global_pose_target"
ORIENTATION_FIRST_BOUNDED_LOCAL_INCREMENT = (
    "orientation_first_bounded_local_increment"
)

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
# The two bodies whose separation is the gripper width, matching the
# convention probe so both read the same physical quantity.
FINGER_BODIES = ("left_inner_finger", "right_inner_finger")
# The pinned Arena DROID embodiment does not define its tool frames at the
# inner-finger body origins.  Each semantic fingertip frame is translated by
# +46 mm along that finger body's local Z axis.  Applying this as a world-Z
# scalar would be wrong whenever a finger rotates, so the helper below composes
# the offset through each live body quaternion before averaging the two tools.
FINGER_TOOL_FRAME_LOCAL_OFFSET_M = (0.0, 0.0, 0.046)
FINGER_TOOL_FRAME_SOURCE = (
    "IsaacLab-Arena@8b4a3a47fc53de23e8205089d71109a2e2348acd:"
    "isaaclab_arena/embodiments/droid/droid.py:tool_leftfinger,tool_rightfinger"
)
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


def _rotate_vector_by_quaternion_xyzw(
    quaternion_xyzw: Sequence[float], vector: Sequence[float]
) -> list[float]:
    x, y, z, w = quaternion_xyzw
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def semantic_finger_tool_midpoint_world_m(
    *,
    left_finger_pose_world_xyzw: Sequence[float],
    right_finger_pose_world_xyzw: Sequence[float],
) -> list[float]:
    """Return the midpoint of Arena's two calibrated Robotiq tool frames.

    Each input is a seven-value world pose ``[x, y, z, qx, qy, qz, qw]`` for
    an inner-finger body.  The fixed local tool offset comes from the exact
    pinned Arena revision used by the immutable runtime bundle.
    """

    try:
        poses = [
            [float(value) for value in left_finger_pose_world_xyzw],
            [float(value) for value in right_finger_pose_world_xyzw],
        ]
    except (TypeError, ValueError) as exc:
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_finger_tool_frame_pose_invalid"]
        ) from exc
    if any(len(pose) != 7 for pose in poses) or not all(
        math.isfinite(value) for pose in poses for value in pose
    ):
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_finger_tool_frame_pose_invalid"]
        )

    tool_positions: list[list[float]] = []
    for pose in poses:
        quaternion = pose[3:7]
        if (
            abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0)
            > 1.0e-5
        ):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_finger_tool_frame_pose_invalid"]
            )
        offset_world = _rotate_vector_by_quaternion_xyzw(
            quaternion, FINGER_TOOL_FRAME_LOCAL_OFFSET_M
        )
        tool_positions.append(
            [pose[axis] + offset_world[axis] for axis in range(3)]
        )
    return [
        (tool_positions[0][axis] + tool_positions[1][axis]) / 2.0
        for axis in range(3)
    ]


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
    body_to_grasp_world = [grasp[index] - body[index] for index in range(3)]
    body_to_grasp_local = _rotate_vector_by_quaternion_xyzw(
        [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
        body_to_grasp_world,
    )
    target_body_to_grasp_world = _rotate_vector_by_quaternion_xyzw(
        target_quaternion,
        body_to_grasp_local,
    )
    target_body = [
        target[index] - target_body_to_grasp_world[index] for index in range(3)
    ]
    return target_body, target_quaternion


def bounded_grasp_frame_target_for_task_orientation(
    *,
    current_position_world_m: Sequence[float],
    current_quaternion_world_xyzw: Sequence[float],
    target_position_world_m: Sequence[float],
    target_quaternion_world_xyzw: Sequence[float],
    max_translation_step_m: float,
    orientation_tolerance_deg: float,
) -> dict[str, Any]:
    """Resolve one local Cartesian target without mixing large rotation/translation.

    The native DLS controller is a local solver.  Asking it to descend the full
    335 mm grasp distance in one step made it trade orientation for translation,
    sweep the open jaws laterally, and stall on the can.  Hold translation while
    task orientation is outside tolerance; otherwise move at most one bounded
    Cartesian increment toward the preregistered target.
    """

    try:
        current = [float(value) for value in current_position_world_m]
        target = [float(value) for value in target_position_world_m]
        current_quaternion = [
            float(value) for value in current_quaternion_world_xyzw
        ]
        target_quaternion = [
            float(value) for value in target_quaternion_world_xyzw
        ]
        max_step = float(max_translation_step_m)
        orientation_tolerance = float(orientation_tolerance_deg)
    except (TypeError, ValueError) as exc:
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_bounded_task_space_target_invalid"]
        ) from exc
    values = (*current, *target, *current_quaternion, *target_quaternion)
    if (
        len(current) != 3
        or len(target) != 3
        or len(current_quaternion) != 4
        or len(target_quaternion) != 4
        or not all(math.isfinite(value) for value in values)
        or not math.isfinite(max_step)
        or max_step <= 0.0
        or not math.isfinite(orientation_tolerance)
        or orientation_tolerance <= 0.0
        or abs(
            math.sqrt(sum(value * value for value in current_quaternion)) - 1.0
        )
        > 1.0e-5
        or abs(
            math.sqrt(sum(value * value for value in target_quaternion)) - 1.0
        )
        > 1.0e-5
    ):
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_bounded_task_space_target_invalid"]
        )
    quaternion_dot = abs(
        sum(
            current_value * target_value
            for current_value, target_value in zip(
                current_quaternion, target_quaternion, strict=True
            )
        )
    )
    orientation_error_deg = math.degrees(
        2.0 * math.acos(min(1.0, max(0.0, quaternion_dot)))
    )
    delta = [target[index] - current[index] for index in range(3)]
    requested_translation_m = math.sqrt(sum(value * value for value in delta))
    translation_held_for_orientation = orientation_error_deg > orientation_tolerance
    if translation_held_for_orientation or requested_translation_m == 0.0:
        resolved = list(current)
        translation_step_m = 0.0
    elif requested_translation_m <= max_step:
        resolved = list(target)
        translation_step_m = requested_translation_m
    else:
        scale = max_step / requested_translation_m
        resolved = [
            current[index] + delta[index] * scale for index in range(3)
        ]
        translation_step_m = max_step
    return {
        "position_world_m": resolved,
        "orientation_error_deg": orientation_error_deg,
        "orientation_tolerance_deg": orientation_tolerance,
        "translation_requested_m": requested_translation_m,
        "translation_step_m": translation_step_m,
        "max_translation_step_m": max_step,
        "translation_held_for_orientation": translation_held_for_orientation,
    }


def grasp_frame_target_for_task_space_strategy(
    *,
    current_position_world_m: Sequence[float],
    current_quaternion_world_xyzw: Sequence[float],
    target_position_world_m: Sequence[float],
    target_quaternion_world_xyzw: Sequence[float],
    max_translation_step_m: float,
    orientation_tolerance_deg: float,
    task_space_translation_strategy: str,
) -> dict[str, Any]:
    """Resolve an immutable-plan strategy into one native IK grasp target.

    Pregrasp starts from the wrist-camera evidence pose, which is roughly 152
    degrees from the top-down task orientation.  The already-proven v6
    controller converged by solving that obstacle-clear pose and translation
    together.  Holding translation against the *current* grasp point while
    rotating made the reference move with IK error and accumulated a 151 mm
    lateral miss.  Later phases remain locally bounded because they start in
    task orientation and operate around the sealed object.
    """

    strategy = str(task_space_translation_strategy).strip()
    if strategy not in {
        DIRECT_GLOBAL_POSE_TARGET,
        ORIENTATION_FIRST_BOUNDED_LOCAL_INCREMENT,
    }:
        raise IsaacEpisodeAdapterError(
            ["isaac_episode_task_space_translation_strategy_invalid"]
        )
    resolved = bounded_grasp_frame_target_for_task_orientation(
        current_position_world_m=current_position_world_m,
        current_quaternion_world_xyzw=current_quaternion_world_xyzw,
        target_position_world_m=target_position_world_m,
        target_quaternion_world_xyzw=target_quaternion_world_xyzw,
        max_translation_step_m=max_translation_step_m,
        orientation_tolerance_deg=orientation_tolerance_deg,
    )
    if strategy == DIRECT_GLOBAL_POSE_TARGET:
        resolved.update(
            {
                "position_world_m": [
                    float(value) for value in target_position_world_m
                ],
                "translation_step_m": resolved["translation_requested_m"],
                "translation_held_for_orientation": False,
            }
        )
    resolved["task_space_translation_strategy"] = strategy
    return resolved


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
        approved_can: Any,
        action_dim: int,
        reset_seed: int,
        to_torch: Any,
        gripper_closed_width_m: float,
        gripper_open_width_m: float,
        reset_callback: Callable[[], None] | None = None,
        simulation_step_seconds: float | None = None,
        scripted_pose_action_callback: Callable[..., Sequence[float]] | None = None,
        camera_pose_callback: Callable[
            [str], tuple[Sequence[float], Sequence[float]] | None
        ]
        | None = None,
        contact_sensor: Any | None = None,
    ) -> None:
        self._env = env
        self._robot = robot
        self._can = approved_can
        self._action_dim = int(action_dim)
        self._reset_seed = int(reset_seed)
        self._to_torch = to_torch
        self._gripper_closed_width_m = float(gripper_closed_width_m)
        self._gripper_open_width_m = float(gripper_open_width_m)
        self._reset_callback = reset_callback
        self._simulation_step_seconds = (
            None
            if simulation_step_seconds is None
            else float(simulation_step_seconds)
        )
        self._scripted_pose_action_callback = scripted_pose_action_callback
        self._camera_pose_callback = camera_pose_callback
        self._contact_sensor = contact_sensor
        self._control_step_index = 0
        if (
            not math.isfinite(self._gripper_closed_width_m)
            or not math.isfinite(self._gripper_open_width_m)
            or self._gripper_open_width_m - self._gripper_closed_width_m <= 1e-6
        ):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_gripper_width_calibration_invalid"]
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
        if contact_sensor is not None:
            contact_body_names = list(contact_sensor.body_names)
            missing_contact_bodies = [
                name for name in FINGER_BODIES if name not in contact_body_names
            ]
            if missing_contact_bodies:
                raise IsaacEpisodeAdapterError(
                    [
                        "isaac_episode_contact_sensor_body_missing:"
                        + ",".join(missing_contact_bodies)
                    ]
                )

    # -- EpisodeEnvironment -------------------------------------------------

    def reset(self) -> None:
        if self._reset_callback is not None:
            self._reset_callback()
        else:
            self._env.reset(seed=self._reset_seed)
        self._control_step_index = 0

    def joint_limits(self) -> list[list[float]]:
        limits = self._to_torch(self._robot.data.joint_limits)[0, :ARM_JOINT_COUNT]
        return [[float(row[0]), float(row[1])] for row in limits]

    def read_policy_inputs(self) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for camera_name, view in CAMERA_VIEW_BINDING.items():
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
        for camera_name, camera_id in EVALUATION_CAMERA_BINDING.items():
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

    def _arm_vector(self, attribute: str) -> list[float]:
        raw = getattr(self._robot.data, attribute, None)
        if raw is None:
            raise IsaacEpisodeAdapterError(
                [f"isaac_episode_arm_dynamics_missing:{attribute}"]
            )
        values = self._to_torch(raw)[0, :ARM_JOINT_COUNT]
        result = [float(value) for value in values]
        if len(result) != ARM_JOINT_COUNT or not all(
            math.isfinite(value) for value in result
        ):
            raise IsaacEpisodeAdapterError(
                [f"isaac_episode_arm_dynamics_invalid:{attribute}"]
            )
        return result

    def _body_contact_forces_world_n(self) -> dict[str, list[float]] | None:
        if self._contact_sensor is None:
            return None
        forces = self._to_torch(self._contact_sensor.data.net_forces_w)[0]
        body_names = list(self._contact_sensor.body_names)
        result: dict[str, list[float]] = {}
        for index, name in enumerate(body_names):
            vector = [float(value) for value in forces[index, :3]]
            if not all(math.isfinite(value) for value in vector):
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_contact_force_invalid:{name}"]
                )
            result[str(name)] = vector
        return result

    def _body_contact_partner_forces_n(self) -> dict[str, list[float]] | None:
        """Name the contact partner when the sensor was built with a filter.

        Supplementary to ``net_forces_w``: a stalled finger reporting a large net
        force but a zero partner force is being held by geometry outside the
        filter, which is the distinction the net force alone cannot make.  A
        sensor built without a filter degrades to ``None`` rather than failing a
        paid run, because the primary contact evidence is unaffected.
        """

        if self._contact_sensor is None:
            return None
        matrix = getattr(self._contact_sensor.data, "force_matrix_w", None)
        if matrix is None:
            return None
        values = self._to_torch(matrix)
        if values.ndim != 4 or values.shape[0] < 1 or values.shape[2] < 1:
            return None
        body_names = list(self._contact_sensor.body_names)
        result: dict[str, list[float]] = {}
        for index, name in enumerate(body_names):
            if index >= values.shape[1]:
                break
            # One filter partner is configured; sum any additional partners into
            # the same reading so a widened filter cannot silently drop force.
            vector = [
                float(values[0, index, :, axis].sum()) for axis in range(3)
            ]
            if not all(math.isfinite(value) for value in vector):
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_contact_partner_force_invalid:{name}"]
                )
            result[str(name)] = vector
        return result or None

    def _body_incoming_joint_wrenches(self) -> dict[str, list[float]]:
        raw = getattr(self._robot.data, "body_incoming_joint_wrench_b", None)
        if raw is None:
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_arm_dynamics_missing:body_incoming_joint_wrench_b"]
            )
        wrenches = self._to_torch(raw)[0]
        result: dict[str, list[float]] = {}
        for index, name in enumerate(self._robot.data.body_names):
            vector = [float(value) for value in wrenches[index, :6]]
            if not all(math.isfinite(value) for value in vector):
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_incoming_joint_wrench_invalid:{name}"]
                )
            result[str(name)] = vector
        return result

    def read_arm_dynamics_observation(self) -> dict[str, Any]:
        """Read actuator tracking and contact state through pinned Isaac APIs."""

        positions = self._arm_vector("joint_pos")
        velocities = self._arm_vector("joint_vel")
        targets = self._arm_vector("joint_pos_target")
        computed = self._arm_vector("computed_torque")
        applied = self._arm_vector("applied_torque")
        effort_limits = self._arm_vector("joint_effort_limits")
        utilization = [
            abs(torque) / limit if limit > 0.0 else 0.0
            for torque, limit in zip(applied, effort_limits, strict=True)
        ]
        return {
            "schema_version": "adp009d_arm_dynamics_observation.v1",
            "joint_position_rad": positions,
            "joint_velocity_rad_s": velocities,
            "joint_position_target_rad": targets,
            "computed_torque_nm": computed,
            "applied_torque_nm": applied,
            "joint_effort_limit_nm": effort_limits,
            "joint_effort_utilization": utilization,
            "torque_clip_residual_nm": [
                before - after
                for before, after in zip(computed, applied, strict=True)
            ],
            "body_contact_force_world_n": self._body_contact_forces_world_n(),
            "body_contact_partner_force_world_n": self._body_contact_partner_forces_n(),
            "body_incoming_joint_wrench_body": self._body_incoming_joint_wrenches(),
        }

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
        max_task_space_translation_step_m: float,
        orientation_tolerance_deg: float,
        task_space_translation_strategy: str,
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
            max_task_space_translation_step_m=float(
                max_task_space_translation_step_m
            ),
            orientation_tolerance_deg=float(orientation_tolerance_deg),
            task_space_translation_strategy=str(task_space_translation_strategy),
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
        for camera_name, camera_id in EVALUATION_CAMERA_BINDING.items():
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
        pose = self._to_torch(self._can.data.root_pose_w)[0]
        controlled_body_pose = self._to_torch(self._robot.data.body_pose_w)[
            0, self._end_effector_index, :7
        ]
        left_pose, right_pose = self._finger_poses()
        left, right = left_pose[:3], right_pose[:3]
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
        sample["gripper_body_midpoint_world_m"] = [
            (left[axis] + right[axis]) / 2.0 for axis in range(3)
        ]
        sample["grasp_frame_position_world_m"] = (
            semantic_finger_tool_midpoint_world_m(
                left_finger_pose_world_xyzw=left_pose,
                right_finger_pose_world_xyzw=right_pose,
            )
        )
        sample["grasp_frame_calibration"] = {
            "frame_id": "probe_calibrated_finger_midpoint",
            "finger_tool_frame_local_offset_m": list(
                FINGER_TOOL_FRAME_LOCAL_OFFSET_M
            ),
            "source": FINGER_TOOL_FRAME_SOURCE,
            "raw_body_midpoint_retained": True,
        }
        contact_forces = self._body_contact_forces_world_n()
        if contact_forces is not None:
            sample["finger_contact_forces_n"] = [
                math.sqrt(sum(component * component for component in contact_forces[name]))
                for name in FINGER_BODIES
            ]
        return sample

    # -- internals ----------------------------------------------------------

    def _finger_positions(self) -> tuple[list[float], list[float]]:
        left_pose, right_pose = self._finger_poses()
        return left_pose[:3], right_pose[:3]

    def _finger_poses(self) -> tuple[list[float], list[float]]:
        poses = self._to_torch(self._robot.data.body_pose_w)[0]
        return (
            [float(poses[self._finger_indices[0]][axis]) for axis in range(7)],
            [float(poses[self._finger_indices[1]][axis]) for axis in range(7)],
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
        "finger_tool_frame_local_offset_m": list(FINGER_TOOL_FRAME_LOCAL_OFFSET_M),
        "finger_tool_frame_source": FINGER_TOOL_FRAME_SOURCE,
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
        "arm_dynamics_observation_schema_version": (
            "adp009d_arm_dynamics_observation.v1"
        ),
        "contact_force_source": "IsaacLab ContactSensor.data.net_forces_w",
        "incoming_joint_wrench_source": (
            "IsaacLab ArticulationData.body_incoming_joint_wrench_b"
        ),
        "scripted_control_task_space_translation_strategies": [
            DIRECT_GLOBAL_POSE_TARGET,
            ORIENTATION_FIRST_BOUNDED_LOCAL_INCREMENT,
        ],
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
    if list(bindings.get("finger_tool_frame_local_offset_m") or []) != list(
        FINGER_TOOL_FRAME_LOCAL_OFFSET_M
    ):
        errors.append("isaac_episode_adapter_finger_tool_frame_offset_drifted")
    if bindings.get("finger_tool_frame_source") != FINGER_TOOL_FRAME_SOURCE:
        errors.append("isaac_episode_adapter_finger_tool_frame_source_drifted")
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
    if bindings.get("arm_dynamics_observation_schema_version") != (
        "adp009d_arm_dynamics_observation.v1"
    ):
        errors.append("isaac_episode_adapter_arm_dynamics_schema_drifted")
    if bindings.get("contact_force_source") != (
        "IsaacLab ContactSensor.data.net_forces_w"
    ):
        errors.append("isaac_episode_adapter_contact_force_source_drifted")
    if bindings.get("incoming_joint_wrench_source") != (
        "IsaacLab ArticulationData.body_incoming_joint_wrench_b"
    ):
        errors.append("isaac_episode_adapter_incoming_joint_wrench_source_drifted")
    if list(
        bindings.get("scripted_control_task_space_translation_strategies") or []
    ) != [
        DIRECT_GLOBAL_POSE_TARGET,
        ORIENTATION_FIRST_BOUNDED_LOCAL_INCREMENT,
    ]:
        errors.append("isaac_episode_adapter_task_space_strategy_drifted")
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
    "END_EFFECTOR_BODY_CANDIDATES",
    "FINGER_BODIES",
    "FINGER_TOOL_FRAME_LOCAL_OFFSET_M",
    "FINGER_TOOL_FRAME_SOURCE",
    "GRIPPER_PHYSICAL_FULL_OPENING_M",
    "GRIPPER_WIDTH_SOURCE",
    "IsaacEpisodeAdapter",
    "IsaacEpisodeAdapterError",
    "controlled_body_pose_for_grasp_frame_target",
    "describe_adapter",
    "rgb_from_camera_output",
    "rotation_row_major_from_quaternion_xyzw",
    "semantic_finger_tool_midpoint_world_m",
    "validate_adapter_bindings",
]
