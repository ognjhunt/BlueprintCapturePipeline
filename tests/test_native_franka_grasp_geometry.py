from __future__ import annotations

import pytest

from blueprint_pipeline.native_franka_grasp_geometry import (
    measure_live_robotiq_grasp_geometry,
    validate_measured_grasp_geometry,
)


pxr = pytest.importorskip("pxr")


def test_live_pad_bounds_define_an_explicit_rigid_tcp() -> None:
    from pxr import Gf, Usd, UsdGeom

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
        UsdGeom.Cube.Define(stage, f"{finger.GetPath()}/pad").CreateSizeAttr(0.02)

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
    assert validate_measured_grasp_geometry(result) == result
