"""Embodiment-specific construction behind the shared native scene runner.

Scene objects, episode lifecycle, recording and scoring remain owned by the
existing task runner. This registry selects only the robot implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Mapping


@dataclass(frozen=True)
class NativeRobotAdapter:
    robot_id: str
    arena_module: str
    factory_module: str
    factory_name: str
    camera_roles: tuple[tuple[str, str, str], ...] = ()
    required_camera_roles: tuple[str, ...] = ()
    policy_camera_roles: tuple[str, ...] = ()
    validator_name: str | None = None


_ADAPTERS: dict[str, NativeRobotAdapter] = {}


def register_native_robot_adapter(adapter: NativeRobotAdapter) -> None:
    if not all(
        (adapter.robot_id, adapter.arena_module, adapter.factory_module, adapter.factory_name)
    ):
        raise ValueError("native_robot_adapter_identity_missing")
    if adapter.robot_id in _ADAPTERS:
        raise ValueError(f"native_robot_adapter_already_registered:{adapter.robot_id}")
    declared_roles = [role for role, _, _ in adapter.camera_roles]
    if (
        len(declared_roles) != len(set(declared_roles))
        or not set(adapter.required_camera_roles).issubset(declared_roles)
        or not set(adapter.policy_camera_roles).issubset(declared_roles)
        or not set(adapter.required_camera_roles) & set(adapter.policy_camera_roles)
    ):
        raise ValueError(f"native_robot_adapter_camera_contract_invalid:{adapter.robot_id}")
    _ADAPTERS[adapter.robot_id] = adapter


def native_robot_adapter(robot_id: str) -> NativeRobotAdapter:
    try:
        return _ADAPTERS[robot_id]
    except KeyError as exc:
        raise RuntimeError(f"native_task_arena_robot_embodiment_unadmitted:{robot_id}") from exc


def native_robot_modules() -> dict[str, str]:
    return {key: row.arena_module for key, row in _ADAPTERS.items()}


def validate_native_robot_plan(robot: Mapping[str, Any]) -> None:
    adapter = native_robot_adapter(str(robot.get("robot_id") or ""))
    if adapter.validator_name:
        getattr(importlib.import_module(adapter.factory_module), adapter.validator_name)(robot)


def build_native_robot_embodiment(
    robot: Mapping[str, Any],
    *,
    enable_cameras: bool,
    pose_class: Any,
) -> Any:
    adapter = native_robot_adapter(str(robot.get("robot_id") or ""))
    factory = getattr(importlib.import_module(adapter.factory_module), adapter.factory_name)
    return factory(robot, enable_cameras=enable_cameras, pose_class=pose_class)


def build_droid_embodiment(
    robot: Mapping[str, Any], *, enable_cameras: bool, pose_class: Any
) -> Any:
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment

    pose = robot["base_pose_world"]
    reset = dict(robot["joint_reset_positions_rad"])
    embodiment = DroidAbsoluteJointPositionEmbodiment(
        enable_cameras=enable_cameras,
        initial_pose=pose_class(
            position_xyz=tuple(pose["position_world_m"]),
            rotation_xyzw=tuple(pose["orientation_xyzw"]),
        ),
        initial_joint_pose=list(reset.values()),
    )
    embodiment.event_config.init_franka_arm_pose.params["default_pose"] = list(reset.values())
    embodiment.event_config.randomize_franka_joint_state.params["mean"] = 0.0
    embodiment.event_config.randomize_franka_joint_state.params["std"] = 0.0
    embodiment.get_scene_cfg()
    embodiment.scene_config.stand = None
    return embodiment


register_native_robot_adapter(
    NativeRobotAdapter(
        robot_id="franka_panda",
        arena_module="isaaclab_arena.embodiments.droid.droid",
        factory_module=__name__,
        factory_name="build_droid_embodiment",
        camera_roles=(
            ("external", "external_camera", "world"),
            ("wrist", "wrist_camera", "robot_body"),
            ("overview", "external_camera_2", "world"),
        ),
        required_camera_roles=("external", "wrist", "overview"),
        policy_camera_roles=("external", "wrist"),
    )
)
register_native_robot_adapter(
    NativeRobotAdapter(
        robot_id="unitree_g1",
        arena_module="isaaclab_arena.embodiments.g1.g1",
        factory_module="blueprint_pipeline.native_g1_embodiment",
        factory_name="build_g1_embodiment",
        validator_name="validate_g1_robot_plan",
        camera_roles=(
            ("head", "robot_head_cam", "robot_body"),
            ("left_wrist", "left_wrist_camera", "robot_body"),
            ("right_wrist", "right_wrist_camera", "robot_body"),
            ("overview", "external_camera_2", "world"),
        ),
        required_camera_roles=("head", "overview"),
        policy_camera_roles=("head", "left_wrist", "right_wrist"),
    )
)
