"""Bind a task-neutral Arena build to the shared episode environment seam.

The Arena builder owns native scene names and the articulated readback owns
task state.  This module only joins those measured bindings to the existing
Isaac episode adapter; it does not select a scene, object class, joint name, or
task outcome.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_isaac_episode_adapter import IsaacEpisodeAdapter
from .native_franka_pose_servo import DEFAULT_VELOCITY_FEEDFORWARD_SCALE


SCHEMA_VERSION = "native_task_episode_environment.v2"


class NativeTaskEpisodeEnvironmentError(ValueError):
    """Stable failures while binding one native construction to episodes."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _gripper_endpoint(
    convention: Mapping[str, Any], *, command_field: str
) -> tuple[float, float]:
    try:
        command = float(convention[command_field])
        separation = float(convention["finger_separation_m"][str(command)])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_gripper_convention_invalid"]
        ) from exc
    if not math.isfinite(command) or not math.isfinite(separation):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_gripper_convention_invalid"]
        )
    return command, separation


def build_native_task_episode_environment(
    *,
    built: Any,
    gripper_convention: Mapping[str, Any],
    servo: Any,
    task_readback: Any | None,
    to_tensor: Any,
    scripted_pose_joint_targets: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[IsaacEpisodeAdapter, dict[str, Any]]:
    """Create the shared control/policy adapter from native Arena readbacks."""

    plan = getattr(built, "plan", None)
    env = getattr(built, "env", None)
    scene_asset_names = getattr(built, "scene_asset_names", None)
    camera_scene_names = getattr(built, "camera_scene_names", None)
    if (
        not isinstance(plan, Mapping)
        or env is None
        or not isinstance(scene_asset_names, Mapping)
        or not isinstance(camera_scene_names, Mapping)
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_arena_build_invalid"]
        )
    task_kind = str(plan.get("task_kind") or "")
    if task_kind not in {"rigid_pick_place", "articulated_open_close"}:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_task_kind_unsupported"]
        )
    try:
        seed = int(plan["scenario"]["seed"])
        control_frequency_hz = float(plan["cadence"]["control_frequency_hz"])
        action_dim = int(env.unwrapped.action_manager.total_action_dim)
        scene = env.unwrapped.scene
        robot = scene["robot"]
        task_object = scene[scene_asset_names["task_object"]]
        joint_wrench_sensor = scene["robot_joint_wrench"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_native_binding_missing"]
        ) from exc
    if (
        not math.isfinite(control_frequency_hz)
        or control_frequency_hz <= 0.0
        or action_dim != 8
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_action_or_cadence_invalid"]
        )
    closed_command, closed_separation = _gripper_endpoint(
        gripper_convention, command_field="closed_command"
    )
    open_command, open_separation = _gripper_endpoint(
        gripper_convention, command_field="open_command"
    )
    if (
        closed_command == open_command
        or open_separation - closed_separation <= 1.0e-6
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_gripper_convention_invalid"]
        )
    if task_kind == "articulated_open_close" and (
        task_readback is None
        or not callable(getattr(task_readback, "read_task_sample", None))
        or not callable(
            getattr(servo, "current_gripper_frame_axis_readback", None)
        )
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_task_readback_missing"]
        )
    if not callable(getattr(servo, "action_for_grasp_target", None)) or not callable(
        getattr(servo, "reset_command_state", None)
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_pose_servo_invalid"]
        )
    try:
        reset_orientation = [
            float(value) for value in servo.current_grasp_frame_pose_world()[3:7]
        ]
    except (AttributeError, TypeError, ValueError) as exc:
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_controlled_body_pose_missing"]
        ) from exc
    if len(reset_orientation) != 4 or not all(
        math.isfinite(value) for value in reset_orientation
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_controlled_body_pose_missing"]
        )

    joint_target_rows: list[dict[str, Any]] = []
    joint_targets_by_pose: dict[
        tuple[tuple[float, ...], tuple[float, ...]], list[float]
    ] = {}
    for index, raw in enumerate(scripted_pose_joint_targets or []):
        try:
            phase_id = str(raw["phase_id"])
            position = [float(value) for value in raw["target_position_world_m"]]
            quaternion = [
                float(value) for value in raw["target_quaternion_world_xyzw"]
            ]
            joints = [float(value) for value in raw["joint_positions_rad"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeTaskEpisodeEnvironmentError(
                [f"native_task_episode_scripted_joint_target_invalid:{index}"]
            ) from exc
        if (
            not phase_id
            or len(position) != 3
            or len(joints) != 7
            or len(quaternion) != 4
            or not all(
                math.isfinite(value)
                for value in [
                    *position,
                    *joints,
                    *quaternion,
                ]
            )
        ):
            raise NativeTaskEpisodeEnvironmentError(
                [f"native_task_episode_scripted_joint_target_invalid:{index}"]
            )
        key = (tuple(position), tuple(quaternion))
        if key in joint_targets_by_pose:
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_scripted_joint_target_duplicate"]
            )
        joint_targets_by_pose[key] = joints
        joint_target_rows.append(
            {
                "phase_id": phase_id,
                "target_position_world_m": position,
                "target_quaternion_world_xyzw": quaternion,
                "joint_positions_rad": joints,
            }
        )
    if joint_target_rows and not callable(
        getattr(servo, "action_for_joint_target", None)
    ):
        raise NativeTaskEpisodeEnvironmentError(
            ["native_task_episode_joint_target_servo_invalid"]
        )

    def reset() -> None:
        env.reset(seed=seed)

    def articulated_task_sample() -> dict[str, Any]:
        raw = task_readback.read_task_sample()
        frame = servo.current_gripper_frame_axis_readback()
        measured = frame.get("measured") if isinstance(frame, Mapping) else None
        if not isinstance(raw, Mapping) or not isinstance(measured, Mapping):
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_gripper_measurement_invalid"]
            )
        try:
            separation = float(measured["finger_separation_m"])
            body_position = [
                float(value)
                for value in measured["controlled_body_position_world_m"]
            ]
            body_quaternion = [
                float(value)
                for value in measured[
                    "controlled_body_quaternion_world_xyzw"
                ]
            ]
            midpoint = [
                float(value) for value in measured["finger_midpoint_world_m"]
            ]
            finger_positions = {
                str(name): [float(value) for value in position]
                for name, position in measured[
                    "finger_body_positions_world_m"
                ].items()
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_gripper_measurement_invalid"]
            ) from exc
        finger_values = [
            value
            for position in finger_positions.values()
            for value in position
        ]
        if (
            not math.isfinite(separation)
            or separation < 0.0
            or len(body_position) != 3
            or len(body_quaternion) != 4
            or len(midpoint) != 3
            or len(finger_positions) != 2
            or any(len(position) != 3 for position in finger_positions.values())
            or not all(
                math.isfinite(value)
                for value in [
                    *body_position,
                    *body_quaternion,
                    *midpoint,
                    *finger_values,
                ]
            )
        ):
            raise NativeTaskEpisodeEnvironmentError(
                ["native_task_episode_gripper_measurement_invalid"]
            )
        sample = dict(raw)
        sample.update(
            {
                "gripper_width_m": separation,
                "gripper_controlled_body_position_world_m": body_position,
                "gripper_controlled_body_quaternion_world_xyzw": body_quaternion,
                "gripper_finger_midpoint_world_m": midpoint,
                "gripper_finger_body_positions_world_m": finger_positions,
                "gripper_measurement_authority": (
                    "native_inner_finger_body_world_pose_readback"
                ),
            }
        )
        return sample

    def scripted_pose_action(**kwargs: Any) -> list[float]:
        quaternion = kwargs.get("target_quaternion_world_xyzw")
        resolved_quaternion = reset_orientation if quaternion is None else quaternion
        pose_key = (
            tuple(float(value) for value in kwargs["target_position_world_m"]),
            tuple(float(value) for value in resolved_quaternion),
        )
        joint_target = joint_targets_by_pose.get(pose_key)
        common = {
            "gripper_command": kwargs["gripper_command"],
            "max_joint_delta_rad": kwargs["max_joint_delta_rad"],
            "max_joint_setpoint_lead_rad": kwargs["max_joint_setpoint_lead_rad"],
            # Controls replay the dynamics construction qualified, so the
            # feedforward has to be the same on both sides.
            "velocity_feedforward_scale": kwargs.get(
                "velocity_feedforward_scale", DEFAULT_VELOCITY_FEEDFORWARD_SCALE
            ),
        }
        if joint_target is not None:
            action, _diagnostic = servo.action_for_joint_target(
                target_joint_positions_rad=joint_target,
                **common,
            )
        else:
            action, _diagnostic = servo.action_for_grasp_target(
                target_position_world_m=kwargs["target_position_world_m"],
                target_grasp_frame_quaternion_world_xyzw=resolved_quaternion,
                **common,
            )
        return [float(value) for value in action]

    adapter = IsaacEpisodeAdapter(
        env=env,
        robot=robot,
        rigid_task_object=(task_object if task_kind == "rigid_pick_place" else None),
        action_dim=action_dim,
        reset_seed=seed,
        to_torch=to_tensor,
        gripper_closed_width_m=closed_separation,
        gripper_open_width_m=open_separation,
        reset_callback=reset,
        scripted_pose_controller_reset_callback=servo.reset_command_state,
        simulation_step_seconds=1.0 / control_frequency_hz,
        scripted_pose_action_callback=scripted_pose_action,
        task_sample_callback=(
            articulated_task_sample
            if task_kind == "articulated_open_close"
            else None
        ),
        grasp_frame_pose_callback=servo.current_grasp_frame_pose_world,
        camera_scene_names=camera_scene_names,
        joint_wrench_sensor=joint_wrench_sensor,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "task_kind": task_kind,
        "action_dimension": action_dim,
        "reset_seed": seed,
        "control_frequency_hz": control_frequency_hz,
        "camera_scene_names": dict(camera_scene_names),
        "task_state_source": (
            "native_articulated_task_readback"
            if task_kind == "articulated_open_close"
            else "native_rigid_body_readback"
        ),
        "scripted_pose_source": (
            "construction_global_ik_joint_target_with_native_pose_fallback"
            if joint_target_rows
            else "native_franka_differential_ik_servo"
        ),
        "scripted_pose_joint_targets": joint_target_rows,
        "joint_wrench_source": "IsaacLab JointWrenchSensor force+torque",
        "joint_wrench_convention": "incoming_joint_frame",
        "controlled_body_orientation_source": "native_body_pose_readback",
        "grasp_frame_pose_source": (
            "native_franka_pose_servo.measured_controlled_body_to_grasp_frame"
        ),
        "gripper_state_source": (
            "native_inner_finger_body_world_pose_readback_each_sample"
            if task_kind == "articulated_open_close"
            else None
        ),
        "gripper_command_mapping": {
            "closed_command": closed_command,
            "open_command": open_command,
            "closed_finger_separation_m": closed_separation,
            "open_finger_separation_m": open_separation,
        },
    }
    return adapter, receipt


__all__ = [
    "NativeTaskEpisodeEnvironmentError",
    "SCHEMA_VERSION",
    "build_native_task_episode_environment",
]
