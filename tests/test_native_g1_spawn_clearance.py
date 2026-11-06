"""Hermetic CPU spawn check of synthetic G1-shaped OpenUSD fixtures; no simulator, network or provider."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_spawn_clearance as spawn
from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline.native_g1_arena_robot_configuration import _RESET
from blueprint_pipeline.scene_physics_preflight import load_asset_model

PELVIS_Z = 0.79
UPRIGHT = [0.0, 0.0, 0.0, 1.0]
RESET = {name: _RESET.get(name, 0.0) for name in PROTOCOL_V4_FULL_JOINT_ORDER}
FOOT_HALF = (0.1, 0.04, 0.02)


def g1_like_asset(path: Path, *, sole_z: float = -PELVIS_Z, soles: str = "box", bare: tuple[str, ...] = (),
                  instanced_pelvis: bool = False, pelvis_shape: str = "box") -> Path:
    """A G1-shaped articulation: the 43 named revolute joints, each carrying one body with a solid collider.

    The pelvis (the root) is a 20 x 30 x 20 cm box at the origin, or (``pelvis_shape``) a capsule of radius
    10 cm or a cone; each foot's sole is ``sole_z`` below it, a 20 x 8 x 4 cm box or (``soles="spheres"``)
    four 5 mm spheres at its corners; every other body is a 2 cm sphere inside the trunk. Bodies named in
    ``bare`` get no collider.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/G1").GetPrim())
    UsdGeom.Scope.Define(stage, "/G1/joints")

    def box(prim_path: str, half, offset=(0.0, 0.0, 0.0)):
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.GetSizeAttr().Set(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*offset))
        cube.AddScaleOp().Set(Gf.Vec3f(*(2 * h for h in half)))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    def sphere(prim_path: str, radius: float, offset=(0.0, 0.0, 0.0)):
        ball = UsdGeom.Sphere.Define(stage, prim_path)
        ball.GetRadiusAttr().Set(radius)
        ball.AddTranslateOp().Set(Gf.Vec3d(*offset))
        UsdPhysics.CollisionAPI.Apply(ball.GetPrim())

    def body(name: str, position) -> str:
        xform = UsdGeom.Xform.Define(stage, "/G1/" + name)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        return "/G1/" + name

    body("pelvis", (0.0, 0.0, 0.0))
    if instanced_pelvis:
        # An instanceable collision scope, as robots imported with instanceable meshes keep it.
        prototype = stage.CreateClassPrim("/PelvisCollision")
        box("/PelvisCollision/box", (0.1, 0.15, 0.1))
        holder = stage.DefinePrim("/G1/pelvis/collisions", "Xform")
        holder.GetReferences().AddInternalReference(prototype.GetPath())
        holder.SetInstanceable(True)
    elif pelvis_shape in ("capsule", "cone"):
        shape = (UsdGeom.Capsule if pelvis_shape == "capsule" else UsdGeom.Cone).Define(stage, "/G1/pelvis/collision")
        shape.GetRadiusAttr().Set(0.1)
        shape.GetHeightAttr().Set(0.1)
        shape.GetAxisAttr().Set("Z")
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    elif "pelvis" not in bare:
        box("/G1/pelvis/collision", (0.1, 0.15, 0.1))
    for index, joint_name in enumerate(PROTOCOL_V4_FULL_JOINT_ORDER):
        link = joint_name.removesuffix("_joint") + "_link"
        if link.endswith("ankle_roll_link"):
            side = 1.0 if link.startswith("left") else -1.0
            lift = FOOT_HALF[2] if soles == "box" else 0.005
            position = (0.0, 0.1 * side, sole_z + lift)
        else:
            position = (0.0, 0.0, -0.5 + 0.02 * index)
        child = body(link, position)
        if link not in bare:
            if not link.endswith("ankle_roll_link"):
                sphere(child + "/collision", 0.02)
            elif soles == "box":
                box(child + "/collision", FOOT_HALF)
            else:
                for corner, (x, y) in enumerate(((0.08, 0.03), (0.08, -0.03), (-0.08, 0.03), (-0.08, -0.03))):
                    sphere(f"{child}/sole_{corner}", 0.005, (x, y, 0.0))
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/G1/joints/" + joint_name)
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/G1/pelvis")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(child)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateAxisAttr().Set("Y")
        joint.CreateLowerLimitAttr().Set(-180.0)
        joint.CreateUpperLimitAttr().Set(180.0)
    head = body("head_link", (0.0, 0.0, 0.4))
    if "head_link" not in bare:
        sphere(head + "/collision", 0.02)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/G1/joints/head_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/G1/pelvis")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(head)])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.4))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    stage.GetRootLayer().Save()
    return path


def room(path: Path, *, wall_x: float | None = None) -> Path:
    """A static room: a 6 m floor at z = 0 and, optionally, a floor-to-ceiling wall in the plane x = ``wall_x``."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    quads = {"floor": [(-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0)]}
    if wall_x is not None:
        quads["wall"] = [(wall_x, -2, 0), (wall_x, 2, 0), (wall_x, 2, 2), (wall_x, -2, 2)]
    for name, corners in quads.items():
        mesh = UsdGeom.Mesh.Define(stage, "/World/" + name)
        mesh.GetPointsAttr().Set([Gf.Vec3f(*corner) for corner in corners])
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return path


def stance(x: float = 0.0, y: float = 0.0) -> dict:
    return {"position_world_m": [x, y, PELVIS_Z], "orientation_xyzw": list(UPRIGHT)}


def measure(tmp_path: Path, *, wall_x: float | None = None, x: float = 0.0, **robot) -> dict:
    asset = g1_like_asset(tmp_path / "g1.usda", **robot)
    _, report = spawn.choose_spawn_stance(robot_usd=asset, joint_reset_positions=RESET, stances=[stance(x)],
        chosen=0, fallback_order=[], background_usd=room(tmp_path / "room.usda", wall_x=wall_x))
    return report


def test_a_body_box_inside_a_wall_is_overlap_with_its_depth_and_shortest_push(tmp_path):
    report = measure(tmp_path, wall_x=0.05)
    assert report["status"] == "overlap" and report["reason"] == spawn.OVERLAP_REASON
    # The pelvis box spans x in [-0.1, 0.1]: the wall at 0.05 is 5 cm inside it, and 5 cm back clears it.
    assert report["worst_bodies"][0]["body"] == "pelvis"
    assert report["worst_bodies"][0]["max_depth_m"] == pytest.approx(0.05, abs=0.004)
    assert report["max_depth_m"] == pytest.approx(0.05, abs=0.004)
    push = report["clearing_push"]
    assert push["distance_m"] == pytest.approx(0.05, abs=0.004) and push["direction_world"][0] < -0.98
    assert report["overlap_tolerance_m"] == 0.005 and report["foot_support_tolerance_m"] == spawn.FOOT_SUPPORT_TOLERANCE_M
    assert report["bodies_without_colliders"] == {"names": [], "count": 0} and report["unreadable_colliders"] == []
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_feet_standing_on_the_floor_within_tolerance_are_clear(tmp_path):
    report = measure(tmp_path, sole_z=-PELVIS_Z - 0.01)  # Soles 1 cm into the floor: settling, not overlap.
    assert report["status"] == "clear" and report["reason"] is None
    assert report["max_depth_m"] < 0.005 and report["clearing_push"]["distance_m"] == 0.0
    feet = report["foot_support"]
    assert feet["bodies_found"] == ["left_ankle_roll_link", "right_ankle_roll_link"] and feet["points"] > 0
    assert feet["max_sink_m"] == pytest.approx(0.01, abs=0.002)
    assert all(row["gap_m"] == pytest.approx(-0.01, abs=1e-6) for row in feet["soles"].values())
    assert report["stance_selection"]["fallback_used"] is False
    assert spawn.unsupported_feet(report) == [] and not any("foot_unsupported" in n for n in report["notices"])


@pytest.mark.parametrize("lift,floor_found", [(0.05, True), (0.15, False)])
def test_a_foot_with_no_floor_under_it_is_clear_but_flagged(tmp_path, lift, floor_found):
    # Soles above the floor: no overlap, but the robot drops (5 cm) or finds no floor within reach (15 cm).
    report = measure(tmp_path, sole_z=-PELVIS_Z + lift)
    assert report["status"] == "clear" and report["reason"] is None
    soles = report["foot_support"]["soles"]
    assert all((row["floor_z_m"] is not None) is floor_found for row in soles.values())
    assert spawn.unsupported_feet(report) == ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert [n for n in report["notices"] if "foot_unsupported" in n] == [
        "g1_spawn_foot_unsupported:left_ankle_roll_link", "g1_spawn_foot_unsupported:right_ankle_roll_link"]


@pytest.mark.parametrize("soles,flags", [
    ({"left_ankle_roll_link": {"floor_z_m": 0.0459, "gap_m": 0.0014},
      "right_ankle_roll_link": {"floor_z_m": 0.0459, "gap_m": 0.0014}}, []),
    # Capture-447's first real reading (2026-11-06): no upward surface within 10 cm under the right sole.
    ({"left_ankle_roll_link": {"floor_z_m": 0.045931, "gap_m": 0.001437},
      "right_ankle_roll_link": {"floor_z_m": None, "gap_m": None}}, ["g1_spawn_foot_unsupported"]),
    ({"left_ankle_roll_link": {"floor_z_m": 0.0, "gap_m": 0.021}}, ["g1_spawn_foot_unsupported"]),
    (None, []),  # A report with no sole readings says nothing about support.
])
def test_planning_flags_an_unsupported_foot_from_the_reports_soles(tmp_path, soles, flags):
    digest = "sha256:" + "a" * 64
    value = {"status": "clear", "scene_plan_digest": digest}
    if soles is not None:
        value["foot_support"] = {"soles": soles}
    (tmp_path / spawn.FILENAME).write_text(json.dumps(spawn.seal(value)))
    assert spawn.spawn_setup_flags(tmp_path, digest) == flags
    assert spawn.spawn_clearance_status(tmp_path, digest) == "clear"  # A flag, never a gate.
    assert spawn.spawn_setup_flags(tmp_path, "sha256:" + "b" * 64) == ["g1_spawn_clearance_unmeasured"]


def test_feet_sunk_past_the_tolerance_are_overlap_and_lift(tmp_path):
    report = measure(tmp_path, sole_z=-PELVIS_Z - 0.03)
    assert report["status"] == "overlap"
    assert report["worst_bodies"][0]["body"].endswith("ankle_roll_link")
    assert report["max_depth_m"] == pytest.approx(0.03, abs=0.002)
    assert report["clearing_push"] == {"distance_m": pytest.approx(0.03, abs=0.002), "direction_world": [0.0, 0.0, 1.0]}


def test_sole_colliders_wholly_under_the_floor_are_still_seen(tmp_path):
    # Four 5 mm sole spheres 3 cm down: no floor point lies inside any of them.
    report = measure(tmp_path, sole_z=-PELVIS_Z - 0.03, soles="spheres")
    assert report["overlapping_points"] == 0 and report["status"] == "overlap"
    assert report["max_depth_m"] == pytest.approx(0.03, abs=1e-6)
    assert report["clearing_push"]["direction_world"] == [0.0, 0.0, 1.0]


def test_a_body_without_collider_prims_is_listed_and_changes_nothing(tmp_path):
    # No collider prim, no contact in the engine either: it cannot start inside the room.
    report = measure(tmp_path, bare=("head_link",))
    assert report["status"] == "clear" and report["reason"] is None
    assert report["bodies_without_colliders"] == {"names": ["head_link"], "count": 1}
    assert report["unreadable_colliders"] == []
    assert measure(tmp_path / "wall", wall_x=0.05, bare=("head_link",))["status"] == "overlap"


def test_a_capsule_is_read_as_its_bounding_cylinder(tmp_path):
    from blueprint_pipeline.scene_physics_preflight import ScenePhysicsPreflightError

    report = measure(tmp_path, wall_x=0.05, pelvis_shape="capsule")
    # Radius 10 cm about the pelvis axis: the wall at x = 5 cm is 5 cm inside it.
    assert report["status"] == "overlap" and report["unreadable_colliders"] == []
    assert report["worst_bodies"][0]["body"] == "pelvis"
    assert report["worst_bodies"][0]["max_depth_m"] == pytest.approx(0.05, abs=0.004)
    [capsule] = [c for c in load_asset_model(tmp_path / "g1.usda", lenient_shapes=True).colliders if c.body == "/G1/pelvis"]
    assert capsule.kind == "cylinder" and capsule.half_height == pytest.approx(0.15)  # 5 cm half height + 10 cm cap
    with pytest.raises(ScenePhysicsPreflightError, match="collider_type_unsupported:Capsule"):
        load_asset_model(tmp_path / "g1.usda")  # The arm check is unchanged.


def test_a_collider_of_a_shape_with_no_solid_is_unmeasured_and_an_overlap_elsewhere_still_counts(tmp_path):
    report = measure(tmp_path, pelvis_shape="cone")
    assert report["status"] == "unmeasured" and report["reason"] == "g1_spawn_colliders_unreadable"
    assert report["unreadable_colliders"] == ["/G1/pelvis/collision"]
    assert "pelvis" not in report["bodies_without_colliders"]["names"]
    # The feet still cross a wall at x = 5 cm: a measured overlap stands.
    assert measure(tmp_path / "wall", wall_x=0.05, pelvis_shape="cone")["status"] == "overlap"


def test_an_unreadable_robot_usd_is_unmeasured_by_name(tmp_path):
    from tests.test_native_g1_arena_robot_configuration import _asset

    asset = tmp_path / "joints-without-bodies.usda"
    _asset(asset)  # The arena configuration fixture: joints with no bodies.
    used, report = spawn.choose_spawn_stance(robot_usd=asset, joint_reset_positions=RESET, stances=[stance()],
        chosen=0, fallback_order=[], background_usd=room(tmp_path / "room.usda"))
    assert used == 0 and report["status"] == "unmeasured"
    assert report["reason"] in {"scene_physics_asset_frame_invalid", "scene_physics_joint_bodies_invalid"}


def test_reset_joints_the_usd_lacks_are_unmeasured(tmp_path):
    asset = g1_like_asset(tmp_path / "g1.usda")
    _, report = spawn.choose_spawn_stance(robot_usd=asset, joint_reset_positions={**RESET, "tail_joint": 0.1},
        stances=[stance()], chosen=0, fallback_order=[], background_usd=room(tmp_path / "room.usda"))
    assert report["status"] == "unmeasured" and report["reason"] == "g1_spawn_reset_joints_unmatched:tail_joint"


def test_an_overlapping_stance_falls_back_to_the_first_clearing_candidate(tmp_path):
    asset = g1_like_asset(tmp_path / "g1.usda")
    background = room(tmp_path / "room.usda", wall_x=0.05)
    stances = [stance(0.0), stance(0.02), stance(-1.0), stance(-1.5)]
    used, report = spawn.choose_spawn_stance(robot_usd=asset, joint_reset_positions=RESET, stances=stances,
        chosen=0, fallback_order=[1, 2, 3], background_usd=background)
    assert used == 2 and report["status"] == "clear"
    selection = report["stance_selection"]
    assert selection["fallback_used"] is True and selection["search_rank_used"] == 2
    assert selection["fallback_stances_measured"] == 3
    assert selection["chosen_overlap"]["worst_bodies"][0]["body"] == "pelvis"
    # With no candidate clear, the chosen stance stays, with its overlap.
    used, report = spawn.choose_spawn_stance(robot_usd=asset, joint_reset_positions=RESET, stances=stances[:2],
        chosen=0, fallback_order=[1], background_usd=background)
    assert used == 0 and report["status"] == "overlap" and report["stance_selection"]["fallback_used"] is False


def test_instanced_colliders_are_read(tmp_path):
    asset = g1_like_asset(tmp_path / "g1.usda", instanced_pelvis=True)
    pelvis = "/G1/pelvis"
    assert not any(c.body == pelvis for c in load_asset_model(asset).colliders)
    assert any(c.body == pelvis for c in load_asset_model(asset, instance_proxies=True).colliders)
    _, report = spawn.choose_spawn_stance(robot_usd=asset, joint_reset_positions=RESET, stances=[stance()],
        chosen=0, fallback_order=[], background_usd=room(tmp_path / "room.usda", wall_x=0.05))
    assert report["worst_bodies"][0]["body"] == "pelvis"


def sealed_report(plan_digest: str, status: str = "clear") -> dict:
    return spawn.seal({"status": status, "scene_plan_digest": plan_digest})


@pytest.mark.parametrize("case,reading", [
    ("clear", "clear"), ("overlap", "overlap"), ("unmeasured", "unmeasured"), ("missing", "unmeasured"),
    ("other_plan", "unmeasured"), ("tampered", "unmeasured"),
])
def test_planning_reads_overlap_only_from_a_sealed_report_of_this_plan(tmp_path, case, reading):
    digest = "sha256:" + "a" * 64
    if case != "missing":
        value = sealed_report("sha256:" + "b" * 64 if case == "other_plan" else digest,
                              case if case in ("overlap", "unmeasured") else "clear")
        if case == "tampered":
            value["max_depth_m"] = 0.0  # Any change after sealing.
        (tmp_path / spawn.FILENAME).write_text(json.dumps(value))
    assert spawn.spawn_clearance_status(tmp_path, digest) == reading
