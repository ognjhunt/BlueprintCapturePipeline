from __future__ import annotations

import hashlib
import json
import math
import shutil

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.robot_arm_path_clearance import (
    ARM_LINK_BOUNDS,
    DIRECTIONS,
    RobotArmPathAdmissionBlocked,
    RobotArmPathClearanceError,
    arm_path_clearance,
    measure_arm_link_bounds,
    packet_arm_path_clearance,
    require_robot_arm_path_admission,
    sample_scene_obstacles,
    scene_radius_m,
)
from blueprint_pipeline.scene_physics_preflight import ROBOT_FIXED_VOLUMES


def _cabinet(tmp_path, panel=None):
    """A dishwasher-like cabinet behind x = 0: its 0.6 m wide, 0.8 m door drops forward about a hinge on y.

    ``panel`` (size, centre) adds one more collider to the cabinet's fixed body.
    """
    path = tmp_path / ("cabinet.usda" if panel is None else "cabinet-panel.usda")
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
    if panel is not None:
        cube = UsdGeom.Cube.Define(stage, "/Cabinet/base/panel")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*panel[1]))
        cube.AddScaleOp().Set(Gf.Vec3f(*panel[0]))
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


def _measure(tmp_path, x, y, *, asset=None, scene=None):
    yaw = math.atan2(-y, 0.05 - x)
    return arm_path_clearance(
        robot_id="franka_panda",
        base_pose_world={"position_world_m": [x, y, 0.0], "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]},
        asset_usd=asset or _cabinet(tmp_path),
        asset_pose_world={"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        joint_id="door_hinge", task_range=[0.0, math.radians(54)], reset_joint_positions={"door_hinge": 0.0},
        contact_point_world=[0.06, 0.0, 0.80], approach_point_world=[0.16, 0.0, 0.80], scene_obstacles=scene)


def _box_triangles(low, high):
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
                       dtype=float)
    faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    return np.asarray([corners[[a, b, c]] for a, b, c, d in faces] + [corners[[a, c, d]] for a, b, c, d in faces])


def _room(*boxes):
    """A floor 100 m across at z = 0, and boxes (low, high) on it."""
    floor = np.array([[[-50, -50, 0], [50, -50, 0], [50, 50, 0]], [[-50, -50, 0], [50, 50, 0], [-50, 50, 0]]], dtype=float)
    return np.concatenate([floor, *(_box_triangles(*box) for box in boxes)])


COUNTER = ((-0.6, -1.0, 0.90), (0.04, 1.0, 0.94))  # over the cabinet, its edge flush with the door
WALL = ((0.30, -1.0, 0.0), (0.32, 1.5, 1.6))  # between the diagonal base and the handle


def test_a_door_that_comes_down_on_the_elbow_is_held_nearer_the_robot(tmp_path):
    # From a base off to the side, holding the middle of the handle leaves the door coming down on the
    # elbow from every arm pose tried; holding it toward the robot keeps the arm clear.
    report = _measure(tmp_path, 0.75, 0.45)
    planned = report["grips"][0]
    assert (planned["slide_m"], planned["roll_deg"], planned["reached"]) == (0.0, 0.0, True)
    assert planned["link"] == "panda_link4" and planned["minimum_clearance_m"] < -0.01
    assert report["status"] == "passed" and report["blockers"] == []
    best = report["best_grip"]
    assert best["slide_m"] > 0.05 and best["clear"] is True  # toward the robot, which stands at +y
    assert report["handle_free_length_m"] == [0.28, 0.28]
    np.testing.assert_allclose(report["handle_axis_world"], [0.0, 1.0, 0.0], atol=1e-9)
    assert report["simulator_executed"] is False and report["report_digest"].startswith("sha256:")


def test_the_planned_grip_counts_when_any_arm_pose_holds_it_clear(tmp_path):
    # 2026-10-26 dishwasher: the middle grip struck the door from the reach check's arm solution and
    # cleared it by 9 cm from the rest pose. The elbow's place is the arm's choice, not the grip's.
    from blueprint_pipeline import robot_arm_path_clearance as module

    report = _measure(tmp_path, 0.60, 0.45)
    assert report["status"] == "passed" and report["grips_tried"] == 1
    assert report["best_grip"]["slide_m"] == 0.0 and report["best_grip"]["minimum_clearance_m"] > 0.04
    assert module.REST_SEED_RAD["franka_panda"] == pytest.approx((0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0))


def test_a_base_under_the_doors_arc_has_no_clean_grip(tmp_path):
    # Square in front, the door's top passes over the base above the shoulder volume the base check
    # measures, then comes down on the arm whatever the grip.
    report = _measure(tmp_path, 0.60, 0.0)
    assert report["status"] == "blocked" and report["best_grip"] is None
    assert report["grips_tried"] == 25 and not any(row["clear"] for row in report["grips"])
    (blocker,) = report["blockers"]
    assert blocker.startswith("robot_arm_in_task_part_path:door_hinge@") and blocker.split(":")[-1].startswith("panda_link")


def test_the_room_around_the_task_can_block_an_arm_the_part_misses(tmp_path):
    # A counter over the cabinet leaves the diagonal base's planned grip clear. A wall between that base
    # and the handle leaves no grip whose arm does not cross it, and the blocker names the room.
    centre, radius = [0.60, 0.45, 0.0], scene_radius_m("franka_panda")
    kitchen = sample_scene_obstacles(_room(COUNTER), centre=centre, radius=radius)
    clear = _measure(tmp_path, 0.60, 0.45, scene=kitchen)
    assert clear["status"] == "passed" and clear["scene_checked"] is True
    assert clear["scene_obstacles_digest"] == kitchen.digest and clear["scene_sample_count"] > len(kitchen.points)
    assert clear["best_grip"]["slide_m"] == 0.0 and clear["best_grip"]["minimum_clearance_m"] >= clear["margin_m"]
    walled = _measure(tmp_path, 0.60, 0.45, scene=sample_scene_obstacles(_room(COUNTER, WALL), centre=centre, radius=radius))
    assert walled["status"] == "blocked" and walled["best_grip"] is None
    (blocker,) = walled["blockers"]
    assert blocker.startswith("robot_arm_in_scene_path:door_hinge@") and blocker.split(":")[-1].startswith("panda_link")
    # Without the scene the gate measures the part alone, as it did before.
    alone = _measure(tmp_path, 0.60, 0.45)
    assert alone["status"] == "passed" and alone["scene_checked"] is False and alone["scene_obstacles_digest"] is None


def test_the_task_assets_own_fixed_body_is_scene_the_arm_must_clear(tmp_path):
    # A tall side panel of the cabinet itself stands between the base and the handle. Only the gate knows
    # it does not move with the door, so it measures it whenever it is given the scene.
    asset = _cabinet(tmp_path, panel=((0.9, 0.04, 1.2), (0.45, 0.32, 0.6)))
    assert _measure(tmp_path, 0.60, 0.45, asset=asset)["status"] == "passed"
    empty = sample_scene_obstacles(np.zeros((0, 3, 3)), centre=[0.60, 0.45, 0.0], radius=scene_radius_m("franka_panda"))
    assert len(empty.points) == 0 and len(empty.near(np.zeros(3), 1.0)) == 0
    report = _measure(tmp_path, 0.60, 0.45, asset=asset, scene=empty)
    assert report["status"] == "blocked" and report["blockers"][0].startswith("robot_arm_in_scene_path:door_hinge@")


def test_the_scene_is_sampled_within_reach_and_a_big_floor_is_cut_first():
    triangles = _room(((10.0, 10.0, 0.0), (11.0, 11.0, 1.0)))
    obstacles = sample_scene_obstacles(triangles, centre=[0.0, 0.0, 0.5], radius=1.0)
    assert np.all(np.linalg.norm(obstacles.points - [0.0, 0.0, 0.5], axis=1) <= 1.0)
    assert np.all(obstacles.points[:, 2] == 0.0)  # the box 10 m away is not sampled
    # Every floor point in reach has a sample within the spacing.
    probe = np.array([[x, y, 0.0] for x in np.linspace(-0.7, 0.7, 15) for y in np.linspace(-0.7, 0.7, 15)
                      if x * x + y * y <= 0.49])
    assert float(obstacles.tree.query(probe)[0].max()) <= 0.01
    assert sample_scene_obstacles(triangles, centre=[0.0, 0.0, 0.5], radius=1.0).digest == obstacles.digest
    assert scene_radius_m("franka_panda") == pytest.approx(1.37)
    with pytest.raises(RobotArmPathClearanceError, match="robot_arm_path_robot_unmeasured:ur5e"):
        scene_radius_m("ur5e")


def test_a_grip_that_reaches_but_strikes_is_tried_again_turned_half_a_turn(tmp_path, monkeypatch):
    # The hand turned half a turn about the approach holds the same grip with the wrist elsewhere; a grip
    # whose first turn reaches but strikes must still try it before the grip is given up.
    from blueprint_pipeline import robot_arm_path_clearance as module

    yaw = math.atan2(-0.45, 0.05 - 0.60)
    world_from_base = module._pose_matrix({"position_world_m": [0.60, 0.45, 0.0],
                                           "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]})
    inside_door = (np.linalg.inv(world_from_base) @ np.array([0.02, 0.0, 0.45, 1.0]))[:3]
    first = {}

    def solve(*, rotation_world, **_):
        sign = 1.0 if rotation_world[1, 0] > 0 else -1.0
        first.setdefault("sign", sign)
        return np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, sign]), 0.0, 0.0

    def frames(joints):
        placed = np.repeat(np.eye(4)[None], 9, axis=0)
        placed[:, :3, 3] = inside_door if joints[6] == first["sign"] else (5.0, 5.0, 5.0)
        return placed

    monkeypatch.setattr(module, "solve_grasp_ik", solve)
    monkeypatch.setattr(module, "_frames", frames)
    report = _measure(tmp_path, 0.60, 0.45)
    assert report["status"] == "passed" and report["grips_tried"] == 1
    best = report["best_grip"]
    assert (best["slide_m"], best["roll_deg"], best["half_turn"], best["clear"]) == (0.0, 0.0, True, True)


def test_the_solvers_arithmetic_is_the_published_kinematics():
    # The array frames are franka_kinematics' chain, and the rotation vector inverts a rotation, a half
    # turn included.
    from scipy.spatial.transform import Rotation

    from blueprint_pipeline import robot_arm_path_clearance as module
    from blueprint_pipeline.franka_kinematics import FRANKA_JOINT_LIMITS_RAD, _chain

    rng = np.random.default_rng(3)
    limits = np.asarray(FRANKA_JOINT_LIMITS_RAD)
    for joints in rng.uniform(limits[:, 0], limits[:, 1], size=(50, 7)):
        np.testing.assert_allclose(module._frames(joints), np.asarray(_chain(list(joints))), atol=1e-12)
    for angle in (0.0, 1e-7, 0.4, 2.0, math.pi - 1e-4, math.pi):
        axis = rng.normal(size=3)
        rotation = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
        np.testing.assert_allclose(Rotation.from_rotvec(module._rotation_vector(rotation)).as_matrix(), rotation,
                                   atol=1e-9)


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


def _scene_usd(path, *boxes):
    """A static scene collision asset: the floor and boxes, as the packet compiler writes one."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Scene").GetPrim())
    for number, triangles in enumerate([_room()] + [_box_triangles(*box) for box in boxes]):
        mesh = UsdGeom.Mesh.Define(stage, f"/Scene/mesh_{number}")
        mesh.CreatePointsAttr([Gf.Vec3f(*(float(v) for v in point)) for point in triangles.reshape(-1, 3)])
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
        mesh.CreateFaceVertexIndicesAttr(list(range(3 * len(triangles))))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path, x, y, *boxes):
    """A compiled packet's scene plan around the cabinet, in a room of ``boxes``: the paid gate reads nothing else."""
    root = tmp_path / f"packet-{x}-{y}-{len(boxes)}"
    (root / "assets").mkdir(parents=True)
    asset = root / "assets" / "cabinet.usda"
    shutil.copy(_cabinet(tmp_path), asset)
    yaw = math.atan2(-y, 0.05 - x)
    identity = {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    plan = {
        "plan_digest": "sha256:" + "p" * 64,
        "objects": [{"name": "scene_collision", "object_type": "BASE", "semantic_role": "scene_collision",
                     "usd_path": "assets/collision.usda", "pose_world": identity,
                     "sha256": _scene_usd(root / "assets" / "collision.usda", *boxes)},
                    {"name": "task_object", "object_type": "ARTICULATION", "semantic_role": "task_object",
                     "usd_path": "assets/cabinet.usda", "sha256": "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest(),
                     "pose_world": identity, "reset_state": {"joint_positions": {"door_hinge": 0.0}}}],
        "articulation": {"task_joint_prim_paths": {"door": "{ENV_REGEX_NS}/task_object/door_hinge"}},
        "task_spec": {"task_kind": "articulated_open_close",
                      "executable_opening_threshold": {"target_joint_id": "door", "reset_position": 0.0,
                                                       "success_interval": [math.radians(54), math.radians(90)]},
                      "interaction_affordance": {"contact_body_prim_paths": ["/Cabinet/door"],
                                                 "contact_point_link_m": [0.06, 0.0, 0.80],
                                                 "approach_unit_asset_root": [1.0, 0.0, 0.0],
                                                 "precontact_clearance_m": 0.10}},
        "robot": {"robot_id": "franka_panda",
                  "base_pose_world": {"position_world_m": [x, y, 0.0],
                                      "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]},
                  "joint_reset_positions_rad": {f"panda_joint{i}": v for i, v in
                                                enumerate((0.0, -0.6283, 0.0, -2.5133, 0.0, 1.885, 0.0), start=1)}},
    }
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    return root


def test_the_paid_launch_rechecks_a_packet_from_its_own_plan(tmp_path):
    # A packet compiled before the placement gate existed reaches the paid launch with no record of it.
    diagonal = _packet(tmp_path, 0.60, 0.45, COUNTER)
    clear = packet_arm_path_clearance(diagonal)
    assert clear["status"] == "passed" and clear["arena_scene_plan_digest"] == "sha256:" + "p" * 64
    assert clear["joint_id"] == "door_hinge" and clear["joint_span"] == [0.0, pytest.approx(math.radians(54))]
    assert clear["scene_checked"] is True and clear["scene_sample_count"] > 0
    assert require_robot_arm_path_admission(diagonal)["report_digest"] == clear["report_digest"]
    with pytest.raises(RobotArmPathAdmissionBlocked) as struck:
        require_robot_arm_path_admission(_packet(tmp_path, 0.60, 0.0))
    (blocker,) = struck.value.blockers
    assert blocker.startswith("robot_arm_in_task_part_path:door_hinge@")
    # The room the simulator loads blocks the arm as the part does.
    with pytest.raises(RobotArmPathAdmissionBlocked) as walled:
        require_robot_arm_path_admission(_packet(tmp_path, 0.60, 0.45, COUNTER, WALL))
    (blocker,) = walled.value.blockers
    assert blocker.startswith("robot_arm_in_scene_path:door_hinge@")


def test_the_paid_launch_refuses_what_it_cannot_measure_and_skips_rigid_tasks(tmp_path):
    packet = _packet(tmp_path, 0.60, 0.45)
    (packet / "assets" / "cabinet.usda").write_text("#usda 1.0\n")
    with pytest.raises(RobotArmPathAdmissionBlocked) as unbound:
        require_robot_arm_path_admission(packet)
    assert unbound.value.blockers == ["robot_arm_path_refused:robot_arm_path_task_asset_unbound"]
    # Without the scene it would load, the arm's path is not measured, so the run is refused.
    roomless = _packet(tmp_path, 0.75, 0.45)
    (roomless / "assets" / "collision.usda").write_text("#usda 1.0\n")
    with pytest.raises(RobotArmPathAdmissionBlocked) as unmeasured:
        require_robot_arm_path_admission(roomless)
    assert unmeasured.value.blockers == ["robot_arm_path_refused:robot_arm_path_scene_collision_unbound"]
    rigid = _packet(tmp_path, 0.75, 0.30)
    plan = json.loads((rigid / "native_task_arena_scene_plan.v1.json").read_text())
    plan["task_spec"]["task_kind"] = "rigid_pick_place"
    (rigid / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    report = require_robot_arm_path_admission(rigid)
    assert (report["status"], report["reason"]) == ("not_applicable", "task_moves_no_part")
