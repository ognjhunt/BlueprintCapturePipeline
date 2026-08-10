from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from blueprint_pipeline.native_articulated_motion_geometry import (
    NativeArticulatedMotionGeometryError,
    derive_native_articulated_motion_geometry,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> None:
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
        double3 xformOp:translate = (0.1, 0.2, 0.0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Xform "door" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        double3 xformOp:translate = (0.1, 0.2, 0.0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def "joints"
    {
        def PhysicsRevoluteJoint "door_hinge"
        {
            uniform token physics:axis = "Z"
            rel physics:body0 = </Asset/cabinet>
            rel physics:body1 = </Asset/door>
            point3f physics:localPos0 = (0, 0, 1)
            point3f physics:localPos1 = (0, 0, 1)
            float physics:lowerLimit = 0
            float physics:upperLimit = 90
        }
    }
}
''',
        encoding="utf-8",
    )


def test_joint_and_task_transforms_own_world_motion_geometry(tmp_path: Path) -> None:
    asset = tmp_path / "door.usda"
    _asset(asset)
    half = math.sqrt(0.5)

    geometry = derive_native_articulated_motion_geometry(
        task_object_usd_path=asset,
        task_object_sha256=_sha(asset),
        target_joint_id="door",
        target_joint_prim_path="/Asset/joints/door_hinge",
        moving_link_prim_path="/Asset/door",
        handle_grasp_point_moving_link_m=[0.5, 0.0, 1.0],
        task_object_pose_world={
            "position_world_m": [2.0, 3.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, half, half],
        },
        reset_angle_rad=0.0,
        scripted_target_angle_rad=math.radians(45.0),
    )

    assert geometry["hinge_point_task_object_m"] == pytest.approx([0.1, 0.2, 1.0])
    assert geometry["hinge_point_world_m"] == pytest.approx([1.8, 3.1, 1.0])
    assert geometry["handle_grasp_point_closed_world_m"] == pytest.approx(
        [1.8, 3.6, 1.0]
    )
    assert geometry["hinge_axis_world_unit"] == pytest.approx([0.0, 0.0, 1.0])
    assert geometry["scripted_sweep_angle_degrees"] == pytest.approx(45.0)


def test_digest_mismatch_fails_before_geometry_is_used(tmp_path: Path) -> None:
    asset = tmp_path / "door.usda"
    _asset(asset)

    with pytest.raises(NativeArticulatedMotionGeometryError) as excinfo:
        derive_native_articulated_motion_geometry(
            task_object_usd_path=asset,
            task_object_sha256="sha256:" + "0" * 64,
            target_joint_id="door",
            target_joint_prim_path="/Asset/joints/door_hinge",
            moving_link_prim_path="/Asset/door",
            handle_grasp_point_moving_link_m=[0.5, 0.0, 1.0],
            task_object_pose_world={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            reset_angle_rad=0.0,
            scripted_target_angle_rad=math.radians(45.0),
        )

    assert excinfo.value.errors == (
        "native_articulated_motion_task_object_digest_mismatch",
    )
