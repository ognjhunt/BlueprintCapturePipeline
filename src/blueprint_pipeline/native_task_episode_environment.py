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


SCHEMA_VERSION = "native_task_episode_environment.v1"


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
            float(value) for value in servo.current_body_pose_world()[3:7]
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

    def reset() -> None:
        env.reset(seed=seed)

    def scripted_pose_action(**kwargs: Any) -> list[float]:
        quaternion = kwargs.get("target_quaternion_world_xyzw")
        action, _diagnostic = servo.action_for_grasp_target(
            target_position_world_m=kwargs["target_position_world_m"],
            target_body_quaternion_world_xyzw=(
                reset_orientation if quaternion is None else quaternion
            ),
            gripper_command=kwargs["gripper_command"],
            max_joint_delta_rad=kwargs["max_joint_delta_rad"],
            max_joint_setpoint_lead_rad=kwargs["max_joint_setpoint_lead_rad"],
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
            task_readback.read_task_sample
            if task_kind == "articulated_open_close"
            else None
        ),
        camera_scene_names=camera_scene_names,
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
        "scripted_pose_source": "native_franka_differential_ik_servo",
        "controlled_body_orientation_source": "native_reset_readback",
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
