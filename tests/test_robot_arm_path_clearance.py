from __future__ import annotations

import math

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.robot_arm_path_clearance import (
    ARM_LINK_BOUNDS,
    DIRECTIONS,
    RobotArmPathClearanceError,
    arm_path_clearance,
    measure_arm_link_bounds,
)
from blueprint_pipeline.scene_physics_preflight import ROBOT_FIXED_VOLUMES


def _cabinet(tmp_path):
    """A dishwasher-like cabinet behind x = 0: its 0.6 m wide, 0.8 m door drops forward about a hinge on y."""
    path = tmp_path / "cabinet.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Cabinet").GetPrim())
    for body, size, centre in (("base", (0.6, 0.6, 0.9), (-0.3, 0.0, 0.45)),
                               ("door", (0.04, 0.6, 0.8), (0.02, 0.0, 0.45))):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Cabinet/{body}").GetPrim())
        cube = UsdGeom.Cube.Define(stage, f"/Cabinet/{body}/col")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Cabinet/door_hinge")
    hinge.CreateBody0Rel().SetTargets(["/Cabinet/base"])
    hinge.CreateBody1Rel().SetTargets(["/Cabinet/door"])
    for create_pos, create_rot in ((hinge.CreateLocalPos0Attr, hinge.CreateLocalRot0Attr),
                                   (hinge.CreateLocalPos1Attr, hinge.CreateLocalRot1Attr)):
        create_pos().Set(Gf.Vec3f(0.02, 0.0, 0.05))
        create_rot().Set(Gf.Quatf(1.0))
    hinge.CreateAxisAttr().Set("Y")
    hinge.CreateLowerLimitAttr().Set(0.0)
    hinge.CreateUpperLimitAttr().Set(90.0)
    stage.GetRootLayer().Save()
    return path


def _measure(tmp_path, x, y):
    yaw = math.atan2(-y, 0.05 - x)
    return arm_path_clearance(
        robot_id="franka_panda",
        base_pose_world={"position_world_m": [x, y, 0.0], "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]},
        asset_usd=_cabinet(tmp_path),
        asset_pose_world={"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        joint_id="door_hinge", task_range=[0.0, math.radians(54)], reset_joint_positions={"door_hinge": 0.0},
        contact_point_world=[0.06, 0.0, 0.80], approach_point_world=[0.16, 0.0, 0.80])


def test_a_door_that_comes_down_on_the_elbow_is_held_nearer_the_robot(tmp_path):
    # 2026-10-26 dishwasher: from the one base 45 cm to the side, holding the middle of the handle put the
    # elbow 57 mm inside the door at 49 degrees; holding it 10-20 cm toward the robot cleared it.
    report = _measure(tmp_path, 0.60, 0.45)
    planned = report["grips"][0]
    assert (planned["slide_m"], planned["roll_deg"], planned["reached"]) == (0.0, 0.0, True)
    assert planned["link"] == "panda_link4" and planned["minimum_clearance_m"] < -0.03
    assert report["status"] == "passed" and report["blockers"] == []
    best = report["best_grip"]
    assert best["slide_m"] > 0.05 and best["clear"] is True  # toward the robot, which stands at +y
    assert report["handle_free_length_m"] == [0.28, 0.28]
    np.testing.assert_allclose(report["handle_axis_world"], [0.0, 1.0, 0.0], atol=1e-9)
    assert report["simulator_executed"] is False and report["report_digest"].startswith("sha256:")


def test_a_base_under_the_doors_arc_has_no_clean_grip(tmp_path):
    # Square in front, the door's top passes over the base above the shoulder volume the base check
    # measures, then comes down on the arm whatever the grip.
    report = _measure(tmp_path, 0.60, 0.0)
    assert report["status"] == "blocked" and report["best_grip"] is None
    assert report["grips_tried"] == 25 and not any(row["clear"] for row in report["grips"])
    (blocker,) = report["blockers"]
    assert blocker.startswith("robot_arm_in_task_part_path:door_hinge@") and blocker.split(":")[-1].startswith("panda_link")


def test_link_bounds_are_measured_from_the_robot_meshes_in_each_link_frame(tmp_path):
    part = Usd.Stage.CreateNew(str(tmp_path / "part.usda"))
    part.SetDefaultPrim(UsdGeom.Xform.Define(part, "/Part").GetPrim())
    mesh = UsdGeom.Mesh.Define(part, "/Part/Shell")
    mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for x in (-0.1, 0.1) for y in (-0.05, 0.05) for z in (0.0, 0.3)])
    part.GetRootLayer().Save()
    robot = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    robot.SetDefaultPrim(UsdGeom.Xform.Define(robot, "/panda").GetPrim())
    link = UsdGeom.Xform.Define(robot, "/panda/panda_link1")
    link.AddTranslateOp().Set(Gf.Vec3d(5.0, 0.0, 0.0))
    shell = UsdGeom.Xform.Define(robot, "/panda/panda_link1/geometry")
    shell.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.1))
    shell.GetPrim().GetReferences().AddReference(str(tmp_path / "part.usda"))
    shell.GetPrim().SetInstanceable(True)
    adapter = UsdGeom.Cylinder.Define(robot, "/panda/panda_link8/adapter")
    adapter.CreateRadiusAttr(0.04)
    adapter.CreateHeightAttr(0.02)
    robot.GetRootLayer().Save()
    bounds = measure_arm_link_bounds(tmp_path / "robot.usda", ["panda_link1", "panda_link8"])
    # Each bound holds the shape and is rounded up by at most 0.1 mm (the meshes are 32-bit floats).
    box = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.05, 0.05) for z in (0.1, 0.4)])
    extent = (box @ DIRECTIONS.T).max(axis=0)
    measured = np.asarray(bounds["panda_link1"])
    assert np.all(measured >= extent - 1e-7) and np.all(measured <= extent + 1.01e-4)
    np.testing.assert_allclose(measured[:6], [0.1, 0.05, 0.4, 0.1, 0.05, -0.1], atol=1.01e-4)
    np.testing.assert_allclose(bounds["panda_link8"][:3], [0.04, 0.04, 0.01], atol=1.01e-4)
    with pytest.raises(RobotArmPathClearanceError, match="robot_arm_path_link_missing:panda_link9"):
        measure_arm_link_bounds(tmp_path / "robot.usda", ["panda_link9"])


def test_the_link_table_is_bound_to_the_runtime_robot_the_base_volumes_were_measured_on():
    model = ARM_LINK_BOUNDS["franka_panda"]
    assert model["measured_from_sha256"] == ROBOT_FIXED_VOLUMES["franka_panda"]["measured_from_sha256"]
    assert [(name, frame) for name, (frame, _) in model["links"].items()] == [
        (f"panda_link{i}", i) for i in range(1, 9)]
    for _, offsets in model["links"].values():
        offsets = np.asarray(offsets)
        assert offsets.shape == (len(DIRECTIONS),) and np.all(offsets[:3] + offsets[3:6] > 0.0)
    with pytest.raises(RobotArmPathClearanceError, match="robot_arm_path_robot_unmeasured:ur5e"):
        arm_path_clearance(robot_id="ur5e", base_pose_world={}, asset_usd="x", asset_pose_world={}, joint_id="j",
                           task_range=[0.0, 1.0], reset_joint_positions=None, contact_point_world=[0.0, 0.0, 0.0],
                           approach_point_world=[0.1, 0.0, 0.0])
