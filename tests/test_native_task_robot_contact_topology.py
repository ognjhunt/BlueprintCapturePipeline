from __future__ import annotations

import hashlib

import pytest

from blueprint_pipeline import native_task_robot_contact_topology as topology_module
from blueprint_pipeline.native_task_robot_contact_topology import (
    NativeTaskRobotContactTopologyError,
    resolve_native_task_robot_contact_topology,
)


def test_franka_droid_profile_binds_every_contact_body_to_an_exact_path() -> None:
    topology = resolve_native_task_robot_contact_topology("franka_panda")

    assert topology["runtime_asset"]["sha256"] == (
        "sha256:c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2c"
    )
    assert topology["task_contact_body_paths"] == [
        (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/"
            "left_inner_finger"
        ),
        (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/"
            "right_inner_finger"
        ),
    ]
    assert len(topology["protected_collision_body_paths"]) == 18
    assert all(
        "*" not in path for path in topology["protected_collision_body_paths"]
    )


def test_unknown_robot_has_no_guessed_contact_topology() -> None:
    with pytest.raises(NativeTaskRobotContactTopologyError) as excinfo:
        resolve_native_task_robot_contact_topology("unknown_robot")

    assert excinfo.value.errors == (
        "native_task_robot_contact_topology_unavailable:unknown_robot",
    )


def test_g1_topology_uses_verified_asset_bodies_and_exact_contact_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_robot_registry

    verified = []
    monkeypatch.setattr(
        native_task_robot_registry,
        "validate_native_robot_plan",
        lambda robot: verified.append(robot["usd_sha256"]),
    )
    bodies = [
        "{ENV_REGEX_NS}/Robot/pelvis",
        "{ENV_REGEX_NS}/Robot/left_hand/finger_tip",
        "{ENV_REGEX_NS}/Robot/right_hand/finger_tip",
    ]
    monkeypatch.setattr(topology_module, "_asset_rigid_body_paths", lambda _path: bodies)
    robot = {
        "robot_id": "unitree_g1",
        "usd_path": "/verified/g1.usd",
        "usd_sha256": "sha256:" + "a" * 64,
        "task_contact_body_paths": bodies[1:],
    }

    result = resolve_native_task_robot_contact_topology("unitree_g1", robot)

    assert verified == [robot["usd_sha256"]]
    assert result["task_contact_body_paths"] == bodies[1:]
    assert result["protected_collision_body_paths"] == bodies
    assert result["runtime_asset"]["sha256"] == robot["usd_sha256"]


def test_g1_topology_refuses_unverified_contact_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_robot_registry

    monkeypatch.setattr(native_task_robot_registry, "validate_native_robot_plan", lambda _robot: None)
    monkeypatch.setattr(
        topology_module,
        "_asset_rigid_body_paths",
        lambda _path: ["{ENV_REGEX_NS}/Robot/pelvis"],
    )
    robot = {
        "robot_id": "unitree_g1",
        "usd_path": "/verified/g1.usd",
        "usd_sha256": "sha256:" + "a" * 64,
        "task_contact_body_paths": ["{ENV_REGEX_NS}/Robot/not_in_asset"],
    }

    with pytest.raises(NativeTaskRobotContactTopologyError) as excinfo:
        resolve_native_task_robot_contact_topology("unitree_g1", robot)

    assert "native_task_robot_contact_topology_task_bodies_unprotected" in excinfo.value.errors


def test_g1_topology_reads_collision_bodies_from_exact_usd(tmp_path) -> None:
    from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
    from blueprint_pipeline.native_g1_embodiment import ACTION_INTERFACE

    asset = tmp_path / "g1.usda"
    asset.write_text(
        '''#usda 1.0
(
    defaultPrim = "G1"
)
def Xform "G1"
{
    def Xform "pelvis" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "right_finger_tip" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
}
''',
        encoding="utf-8",
    )
    joints = {name: 0.0 for name in PROTOCOL_V4_FULL_JOINT_ORDER}
    robot = {
        "robot_id": "unitree_g1",
        "hand_id": "unitree_dex3_1",
        "action_interface": ACTION_INTERFACE,
        "usd_path": str(asset),
        "usd_sha256": "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest(),
        "base_pose_world": {
            "position_world_m": [0.0, 0.0, 0.8],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "joint_reset_positions_rad": joints,
        "joint_position_limits_rad": {name: [-1.0, 1.0] for name in joints},
        "actuator_parameters": {
            name: {"stiffness": 100, "damping": 2, "effort_limit": 25, "velocity_limit": 10}
            for name in joints
        },
        "task_contact_body_paths": ["{ENV_REGEX_NS}/Robot/right_finger_tip"],
    }

    result = resolve_native_task_robot_contact_topology("unitree_g1", robot)

    assert result["protected_collision_body_paths"] == [
        "{ENV_REGEX_NS}/Robot/pelvis",
        "{ENV_REGEX_NS}/Robot/right_finger_tip",
    ]

    robot["usd_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(NativeTaskRobotContactTopologyError):
        resolve_native_task_robot_contact_topology("unitree_g1", robot)
