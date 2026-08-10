from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from blueprint_pipeline.native_task_arena_scene_plan import (
    NativeTaskArenaScenePlanError,
    materialize_native_task_arena_scene_plan,
)
from blueprint_pipeline.native_task_runtime_contract import (
    DROID_FRANKA_RESET_JOINT_NAMES,
    materialize_native_task_runtime_contract,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
    return {
        "position_world_m": [x, y, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _camera(role: str) -> dict:
    wrist = role == "wrist"
    return {
        "role": role,
        "policy_input": role in {"external", "wrist"},
        "scoring_input": False,
        "pose_frame": "robot_body" if wrist else "world",
        "parent_prim_path": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
            if wrist
            else "{ENV_REGEX_NS}"
        ),
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "intrinsics": {
            "fx": 300.0,
            "fy": 300.0,
            "cx": 159.5,
            "cy": 119.5,
            "width": 320,
            "height": 240,
        },
    }


def _contract(tmp_path: Path, *, articulated: bool) -> tuple[dict, Path]:
    asset_directory = tmp_path / ("articulated" if articulated else "rigid")
    asset_directory.mkdir()
    assets = []
    for role, suffix in (
        ("scene_collision", ".usd"),
        ("scene_appearance", ".usdz"),
        ("task_object", ".usda"),
    ):
        path = asset_directory / f"{role}{suffix}"
        if articulated and role == "scene_collision":
            path.write_text(
                '''#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Root"
{
    def Mesh "floor" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
    def Mesh "counter" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
''',
                encoding="utf-8",
            )
        elif articulated and role == "task_object":
            path.write_text(
                '''#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset"
{
    def Xform "cabinet" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Mesh "body" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
            point3f[] points = [(-0.3, -0.2, 0), (0.3, 0.2, 1.6)]
        }
    }
    def Xform "upper_door" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Mesh "component_004" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
            point3f[] points = [(0, 0, 0), (0.24, 0.04, 0.04)]
        }
        def Mesh "handle_post_a" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
            point3f[] points = [(0, 0, 0), (0.03, 0.03, 0.04)]
        }
        def Mesh "handle_post_b" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
            point3f[] points = [(0, 0, 0), (0.03, 0.03, 0.04)]
        }
    }
    def Xform "lower_door" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def "joints"
    {
        def PhysicsRevoluteJoint "upper_door_hinge"
        {
            uniform token physics:axis = "Z"
            rel physics:body0 = </Asset/cabinet>
            rel physics:body1 = </Asset/upper_door>
            point3f physics:localPos0 = (-0.35696605, 0.35000005, 1.276497)
            point3f physics:localPos1 = (-0.35696605, 0.35000005, 1.276497)
            float physics:lowerLimit = 0
            float physics:upperLimit = 90
        }
        def PhysicsRevoluteJoint "lower_door_hinge"
        {
            uniform token physics:axis = "Z"
            rel physics:body0 = </Asset/cabinet>
            rel physics:body1 = </Asset/lower_door>
            point3f physics:localPos0 = (-0.35696605, 0.35000005, 0.474997)
            point3f physics:localPos1 = (-0.35696605, 0.35000005, 0.474997)
            float physics:lowerLimit = 0
            float physics:upperLimit = 90
        }
    }
}
''',
                encoding="utf-8",
            )
        else:
            path.write_bytes(f"fixture:{role}:{articulated}".encode())
        assets.append(
            {
                "semantic_role": role,
                "filename": path.name,
                "sha256": _digest(path),
                "pose_world": _pose(1.974, 1.479, 0.0)
                if role == "task_object"
                else _pose(),
            }
        )
    task_spec = (
        json.loads(
            (
                Path(__file__).parents[1]
                / "docs/arm_decision_proof_v1/manifests"
                / "second_scene_840796_scene_task_freeze.v1.json"
            ).read_text(encoding="utf-8")
        )["task_spec"]
        if articulated
        else {
            "task_kind": "rigid_pick_place",
            "prompt": "Pick and place the object.",
            "control_frequency_hz": 15,
            "maximum_action_steps": 450,
        }
    )
    joint_bindings = (
        [
            {
                "joint_id": "refrigerator_upper_door_hinge",
                "joint_prim_path": "/Asset/joints/upper_door_hinge",
                "native_joint_name": "upper_door_hinge",
                "role": "task_joint",
            },
            {
                "joint_id": "refrigerator_lower_door_hinge",
                "joint_prim_path": "/Asset/joints/lower_door_hinge",
                "native_joint_name": "lower_door_hinge",
                "role": "locked_joint",
            },
        ]
        if articulated
        else []
    )
    state_binding = (
        {
            "moving_link_prim_path": "/Asset/upper_door",
            "moving_link_native_body_name": "upper_door",
            "handle_prim_paths": [
                "/Asset/upper_door/component_004",
                "/Asset/upper_door/handle_post_a",
                "/Asset/upper_door/handle_post_b",
            ],
            "handle_grasp_point_link_m": [0.119962, 0.327634, 1.022997],
            "robot_gripper_contact_prim_pattern": (
                "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/.*"
            ),
            "robot_collision_prim_pattern": "{ENV_REGEX_NS}/Robot/.*",
            "task_contact_minimum_force_n": 0.5,
            "collision_failure_minimum_force_n": 1.0,
            "retreat_minimum_separation_m": 0.10,
            "root_translation_tolerance_m": 0.002,
            "root_orientation_tolerance_rad": 0.01,
        }
        if articulated
        else None
    )
    contract = materialize_native_task_runtime_contract(
        scene_id="840796" if articulated else "840313",
        task_id=(
            "840796_refrigerator_upper_door_open_v1"
            if articulated
            else "840313_canned_beverage_pick_place_v1"
        ),
        task_spec=task_spec,
        task_joint_bindings=joint_bindings,
        task_state_binding=state_binding,
        assets=assets,
        robot_base_pose_world=_pose(1.75, 1.99, 0.0),
        robot_joint_reset_positions_rad={
            name: float(index) / 100.0
            for index, name in enumerate(DROID_FRANKA_RESET_JOINT_NAMES)
        },
        cameras=[_camera(role) for role in ("external", "wrist", "overview")],
        scenario_cell_id="canonical_seed_17",
        scenario_instance_digest="sha256:" + "d" * 64,
        seed=17,
    )
    return contract, asset_directory


@pytest.mark.parametrize("articulated", [False, True])
def test_original_and_second_scene_compile_through_one_arena_plan(
    tmp_path: Path, articulated: bool
) -> None:
    contract, asset_directory = _contract(tmp_path, articulated=articulated)

    plan = materialize_native_task_arena_scene_plan(
        runtime_contract=contract,
        provider_asset_directory=asset_directory,
        physics_frequency_hz=120,
    )

    objects = {row["semantic_role"]: row for row in plan["objects"]}
    assert objects["task_object"]["object_type"] == (
        "ARTICULATION" if articulated else "RIGID"
    )
    assert objects["scene_collision"]["prim_path"] == (
        "{ENV_REGEX_NS}/scene_collision"
    )
    assert objects["scene_collision"]["visible"] is False
    assert plan["cadence"]["control_decimation"] == 8
    assert plan["cadence"]["physics_dt_seconds"] == pytest.approx(1.0 / 120.0)
    if articulated:
        articulation = plan["articulation"]
        assert articulation["task_joint_reset_positions_rad"] == {
            "lower_door_hinge": 0.0,
            "upper_door_hinge": 0.0,
        }
        channels = Counter(
            row["logical_sensor_id"] for row in articulation["contact_sensors"]
        )
        assert channels == {
            "task_robot_contact": 1,
            "task_scene_contact": 1,
            "robot_scene_contact": 18,
        }
        assert articulation["scene_contact_body_paths"] == [
            "{ENV_REGEX_NS}/scene_collision/counter",
            "{ENV_REGEX_NS}/scene_collision/floor",
        ]
        assert articulation["robot_contact_topology"]["robot_id"] == (
            "franka_panda"
        )
        for row in articulation["contact_sensors"]:
            assert "*" not in row["prim_path"]
            assert all("*" not in path for path in row["filter_prim_paths_expr"])
        assert articulation["task_joint_prim_paths"][
            "refrigerator_upper_door_hinge"
        ] == "{ENV_REGEX_NS}/task_object/joints/upper_door_hinge"
        motion = articulation["motion_geometry"]
        assert motion["target_joint_id"] == "refrigerator_upper_door_hinge"
        assert motion["hinge_point_world_m"] == pytest.approx(
            [1.61703395, 1.82900005, 1.276497]
        )
        assert motion["hinge_axis_world_unit"] == pytest.approx([0.0, 0.0, 1.0])
        assert motion["handle_grasp_point_closed_world_m"] == pytest.approx(
            [2.093962, 1.806634, 1.022997]
        )
        assert motion["scripted_sweep_angle_degrees"] == pytest.approx(50.0)
    else:
        assert plan["articulation"]["contact_sensors"] == []


def test_staged_asset_digest_mismatch_fails_before_isaac(tmp_path: Path) -> None:
    contract, asset_directory = _contract(tmp_path, articulated=True)
    (asset_directory / "task_object.usda").write_bytes(b"tampered")

    with pytest.raises(NativeTaskArenaScenePlanError) as excinfo:
        materialize_native_task_arena_scene_plan(
            runtime_contract=contract,
            provider_asset_directory=asset_directory,
            physics_frequency_hz=120,
        )

    assert excinfo.value.errors == (
        "native_task_arena_asset_digest_mismatch:task_object",
    )


def test_staged_asset_symlink_is_rejected(tmp_path: Path) -> None:
    contract, asset_directory = _contract(tmp_path, articulated=True)
    task_asset = asset_directory / "task_object.usda"
    retained = asset_directory / "retained.usda"
    task_asset.rename(retained)
    task_asset.symlink_to(retained)

    with pytest.raises(NativeTaskArenaScenePlanError) as excinfo:
        materialize_native_task_arena_scene_plan(
            runtime_contract=contract,
            provider_asset_directory=asset_directory,
            physics_frequency_hz=120,
        )

    assert excinfo.value.errors == ("native_task_arena_asset_missing:task_object",)


def test_non_integral_physics_control_ratio_is_rejected(tmp_path: Path) -> None:
    contract, asset_directory = _contract(tmp_path, articulated=True)

    with pytest.raises(NativeTaskArenaScenePlanError) as excinfo:
        materialize_native_task_arena_scene_plan(
            runtime_contract=contract,
            provider_asset_directory=asset_directory,
            physics_frequency_hz=100,
        )

    assert excinfo.value.errors == ("native_task_arena_control_cadence_invalid",)


def test_runtime_contract_tamper_is_rejected(tmp_path: Path) -> None:
    contract, asset_directory = _contract(tmp_path, articulated=True)
    contract["robot"]["base_pose_world"]["position_world_m"][0] += 0.2

    with pytest.raises(NativeTaskArenaScenePlanError) as excinfo:
        materialize_native_task_arena_scene_plan(
            runtime_contract=contract,
            provider_asset_directory=asset_directory,
            physics_frequency_hz=120,
        )

    assert excinfo.value.errors == (
        "native_task_arena_runtime_contract_digest_invalid",
    )
