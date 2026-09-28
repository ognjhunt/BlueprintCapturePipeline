from __future__ import annotations

import pytest

from blueprint_pipeline.native_franka_grasp_geometry import (
    measure_live_robotiq_grasp_geometry,
    validate_measured_grasp_geometry,
)


pxr = pytest.importorskip("pxr")


def test_live_pad_bounds_define_an_explicit_rigid_tcp() -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    for side, y in (("left", 0.04), ("right", -0.04)):
        finger = UsdGeom.Xform.Define(
            stage,
            f"/World/envs/env_0/Robot/Gripper/{side}_inner_finger",
        )
        finger.AddTranslateOp().Set(Gf.Vec3d(0.0, y, 0.12))
        proximal_collider = UsdGeom.Xform.Define(
            stage, f"{finger.GetPath()}/finger"
        )
        UsdPhysics.CollisionAPI.Apply(proximal_collider.GetPrim())
        proximal = UsdGeom.Cube.Define(
            stage, f"{proximal_collider.GetPath()}/geometry"
        )
        proximal.CreateSizeAttr(0.02)
        proximal.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.04))
        pad_collider = UsdGeom.Xform.Define(stage, f"{finger.GetPath()}/pad")
        UsdPhysics.CollisionAPI.Apply(pad_collider.GetPrim())
        pad = UsdGeom.Cube.Define(stage, f"{pad_collider.GetPath()}/geometry")
        pad.CreateSizeAttr(0.02)

    result = measure_live_robotiq_grasp_geometry(
        stage=stage,
        controlled_body_position_world_m=[0.0, 0.0, 0.0],
        controlled_body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    assert result["pad_separation_m"] == pytest.approx(0.08)
    assert result["controlled_body_to_grasp_frame"][
        "position_controlled_body_m"
    ] == pytest.approx([0.0, 0.0, 0.12])
    assert result["controlled_body_to_grasp_frame"][
        "orientation_xyzw"
    ] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert all(
        row["prim_path"].endswith("/pad")
        for row in result["selected_pad_colliders"].values()
    )
    assert all(
        len(result["matched_collision_candidates"][side]) == 2
        for side in ("left", "right")
    )
    assert validate_measured_grasp_geometry(result) == result


def test_live_pad_bounds_use_controlled_body_relative_frame() -> None:
    """A task-aware PhysX root pose need not be authored back into USD yet."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    UsdGeom.Xform.Define(
        stage, "/World/envs/env_0/Robot/Gripper"
    )
    controlled_body = UsdGeom.Xform.Define(
        stage, "/World/envs/env_0/Robot/Gripper/base_link"
    )
    for side, y in (("left", 0.04), ("right", -0.04)):
        finger = UsdGeom.Xform.Define(
            stage,
            f"/World/envs/env_0/Robot/Gripper/{side}_inner_finger",
        )
        finger.AddTranslateOp().Set(Gf.Vec3d(0.0, y, 0.12))
        pad_collider = UsdGeom.Xform.Define(stage, f"{finger.GetPath()}/pad")
        UsdPhysics.CollisionAPI.Apply(pad_collider.GetPrim())
        pad = UsdGeom.Cube.Define(stage, f"{pad_collider.GetPath()}/geometry")
        pad.CreateSizeAttr(0.02)

    # The live PhysX body has already been placed at the task-aware world pose,
    # while the authored USD transform is still at the origin. Geometry must be
    # measured relative to the controlled body rather than mixing those frames.
    result = measure_live_robotiq_grasp_geometry(
        stage=stage,
        controlled_body_position_world_m=[4.0, -6.0, 2.0],
        controlled_body_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    assert result["controlled_body_usd_prim_path"] == str(
        controlled_body.GetPath()
    )
    assert result["pad_separation_m"] == pytest.approx(0.08)
    assert result["body_to_grasp_frame_distance_m"] == pytest.approx(0.12)
    assert all(
        result["selected_pad_colliders"][side][
            "center_inner_finger_body_m"
        ]
        == pytest.approx([0.0, 0.0, 0.0])
        for side in ("left", "right")
    )
    assert result["grasp_frame_world"]["position_world_m"] == pytest.approx(
        [4.0, -6.0, 2.12]
    )
