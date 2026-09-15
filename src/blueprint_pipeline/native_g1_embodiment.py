"""Spawn a sealed G1/Dex3 articulation in the existing captured-site arena.

The whole-body controller runs before this joint-target boundary. Raw VLA
latents or DROID arm actions must never be sent to this action manager.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER

ACTION_INTERFACE = "g1_dex3_named_joint_position.v1"
ACTUATOR_FIELDS = ("stiffness", "damping", "effort_limit", "velocity_limit")


def validate_g1_robot_plan(
    robot: Mapping[str, Any], *, verify_asset: bool = True
) -> dict[str, Any]:
    if (
        robot.get("robot_id") != "unitree_g1"
        or robot.get("hand_id") != "unitree_dex3_1"
        or robot.get("action_interface") != ACTION_INTERFACE
    ):
        raise ValueError("native_g1_embodiment_interface_mismatch")
    names = set(PROTOCOL_V4_FULL_JOINT_ORDER)
    reset = robot.get("joint_reset_positions_rad")
    actuators = robot.get("actuator_parameters")
    limits = robot.get("joint_position_limits_rad")
    if any(
        not isinstance(value, Mapping) or set(value) != names
        for value in (reset, actuators, limits)
    ):
        raise ValueError("native_g1_joint_inventory_mismatch")
    for name in PROTOCOL_V4_FULL_JOINT_ORDER:
        row = actuators[name]
        interval = limits[name]
        if (
            not isinstance(row, Mapping)
            or set(row) != set(ACTUATOR_FIELDS)
            or not isinstance(interval, (list, tuple))
            or len(interval) != 2
        ):
            raise ValueError("native_g1_joint_parameters_invalid")
        values = [reset[name], *interval, *row.values()]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("native_g1_joint_parameters_invalid")
        if (
            not interval[0] < interval[1]
            or not interval[0] <= reset[name] <= interval[1]
            or any(row[field] <= 0 for field in ACTUATOR_FIELDS)
        ):
            raise ValueError("native_g1_joint_parameters_invalid")
    asset = Path(str(robot.get("usd_path") or ""))
    digest = str(robot.get("usd_sha256") or "")
    if not asset.is_absolute() or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("native_g1_robot_asset_identity_missing")
    pose = robot.get("base_pose_world")
    if not isinstance(pose, Mapping):
        raise ValueError("native_g1_reset_pose_invalid")
    position, orientation = pose.get("position_world_m"), pose.get("orientation_xyzw")
    if (
        not isinstance(position, (list, tuple))
        or len(position) != 3
        or not isinstance(orientation, (list, tuple))
        or len(orientation) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            for value in [*position, *orientation]
        )
        or not math.isclose(sum(value * value for value in orientation), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("native_g1_reset_pose_invalid")
    if verify_asset:
        if asset.is_symlink() or not asset.is_file():
            raise ValueError("native_g1_robot_asset_unavailable")
        hasher = hashlib.sha256()
        with asset.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        if digest != "sha256:" + hasher.hexdigest():
            raise ValueError("native_g1_robot_asset_digest_mismatch")
    return deepcopy(dict(robot))


def build_g1_embodiment(robot: Mapping[str, Any], *, enable_cameras: bool, pose_class: Any) -> Any:
    robot = validate_g1_robot_plan(robot)
    # Resolve only after exact input validation and the scoped G1 import.
    import isaaclab.envs.mdp as mdp
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.managers import (
        EventTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        SceneEntityCfg,
    )
    from isaaclab.utils import configclass
    from isaaclab_arena.embodiments.g1.g1 import G1EmbodimentBase
    from isaaclab_arena.utils.cameras import ArenaCameraCfg

    pose = robot["base_pose_world"]
    embodiment = G1EmbodimentBase(
        enable_cameras=enable_cameras,
        initial_pose=pose_class(
            position_xyz=tuple(pose["position_world_m"]),
            rotation_xyzw=tuple(pose["orientation_xyzw"]),
        ),
    )
    cfg = embodiment.scene_config.robot
    cfg.spawn.usd_path = robot["usd_path"]
    parameters = robot["actuator_parameters"]
    cfg.actuators = {
        "whole_body_and_hands": IdealPDActuatorCfg(
            joint_names_expr=list(PROTOCOL_V4_FULL_JOINT_ORDER),
            **{
                field: {name: parameters[name][field] for name in PROTOCOL_V4_FULL_JOINT_ORDER}
                for field in ACTUATOR_FIELDS
            },
        )
    }

    @configclass
    class Actions:
        joint_positions = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(PROTOCOL_V4_FULL_JOINT_ORDER),
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )

    @configclass
    class PolicyObservations(ObservationGroupCfg):
        joint_positions = ObservationTermCfg(
            func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        joint_velocities = ObservationTermCfg(
            func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class Observations:
        policy = PolicyObservations()

    @configclass
    class Events:
        reset_robot_root = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {},
                "velocity_range": {},
            },
        )
        reset_robot_joints = EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )

    head = embodiment.camera_config.robot_head_cam

    @configclass
    class Cameras(ArenaCameraCfg):
        robot_head_cam = deepcopy(head)
        left_wrist_camera = deepcopy(head)
        right_wrist_camera = deepcopy(head)
        external_camera_2 = deepcopy(head)

    embodiment.action_config = Actions()
    embodiment.observation_config = Observations()
    embodiment.event_config = Events()
    embodiment.camera_config = Cameras()
    embodiment.get_scene_cfg()
    return embodiment
