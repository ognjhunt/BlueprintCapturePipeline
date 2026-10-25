"""CPU physics preflight: the three dishwasher faults of 2026-10-25 on synthetic scenes.

A 0.6 x 0.6 x 0.9 m cabinet (solid base box) with a front door hinged at its
bottom edge, opening 0-90 degrees forward, stands on a floor. A back wall 3 cm
inside the cabinet stands for the bay walls; a floor bump where the open door
lands stands for the 90 degree floor contact.
"""
import json
import math

import pytest

from blueprint_pipeline import scene_physics_preflight as spp

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

POSE = {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
FAST = {"sample_spacing_m": 0.01, "revolute_step_rad": math.radians(5.0)}


def _stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _box(stage, path, *, size, centre):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _asset(tmp_path, *, mounted=False, extra_collider=None):
    path = tmp_path / "asset.usda"
    stage = _stage(path)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/Asset/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    _box(stage, "/Asset/base/col", size=(0.6, 0.6, 0.9), centre=(0.0, 0.0, 0.45))
    door = UsdGeom.Xform.Define(stage, "/Asset/door")
    door.AddTranslateOp().Set(Gf.Vec3d(0.32, 0.0, 0.05))
    UsdPhysics.RigidBodyAPI.Apply(door.GetPrim())
    _box(stage, "/Asset/door/col", size=(0.04, 0.6, 0.8), centre=(0.0, 0.0, 0.4))
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/door_hinge")
    hinge.CreateBody0Rel().SetTargets(["/Asset/base"])
    hinge.CreateBody1Rel().SetTargets(["/Asset/door"])
    hinge.CreateLocalPos0Attr().Set(Gf.Vec3f(0.32, 0.0, 0.05))
    hinge.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    hinge.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    hinge.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    hinge.CreateAxisAttr().Set("Y")
    hinge.CreateLowerLimitAttr().Set(0.0)
    hinge.CreateUpperLimitAttr().Set(90.0)
    if mounted:
        mount = UsdPhysics.FixedJoint.Define(stage, "/Asset/mount")
        mount.CreateBody1Rel().SetTargets(["/Asset/base"])
    if extra_collider is not None:
        extra_collider(stage)
    stage.GetRootLayer().Save()
    return path


def _quad(stage, path, corners):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*c) for c in corners])
    mesh.GetFaceVertexCountsAttr().Set([3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def _background(tmp_path, *, back_wall_x=None, floor_bump=False):
    path = tmp_path / "collision.usda"
    stage = _stage(path)
    UsdGeom.Xform.Define(stage, "/World")
    _quad(stage, "/World/floor", [(-0.6, -0.6, 0.0), (1.4, -0.6, 0.0), (1.4, 0.6, 0.0), (-0.6, 0.6, 0.0)])
    if back_wall_x is not None:
        x = back_wall_x
        _quad(stage, "/World/back_wall", [(x, -0.5, 0.0), (x, 0.5, 0.0), (x, 0.5, 1.1), (x, -0.5, 1.1)])
    if floor_bump:
        # Level with the open door's mid-thickness, 0.40-0.60 m in front of the hinge.
        _quad(stage, "/World/bump", [(0.72, -0.1, 0.05), (0.92, -0.1, 0.05), (0.92, 0.1, 0.05), (0.72, 0.1, 0.05)])
    stage.GetRootLayer().Save()
    return path


def test_clean_placement_passes(tmp_path):
    report = spp.scene_physics_preflight(background_usd=_background(tmp_path), asset_usd=_asset(tmp_path),
                                         asset_pose_world=POSE, **FAST)
    assert report["status"] == "passed" and report["blockers"] == []
    assert report["rest_overlap"]["penetrating_points"] == 0
    assert report["envelope_intrusion"]["intruding_points"] == 0
    assert report["motion_clearance"][0]["first_contact"] is None
    assert report["simulator_executed"] is False


def test_wall_inside_the_volume_is_found_by_face_depth_and_push(tmp_path):
    report = spp.scene_physics_preflight(background_usd=_background(tmp_path, back_wall_x=-0.27),
                                         asset_usd=_asset(tmp_path), asset_pose_world=POSE, **FAST)
    assert report["status"] == "blocked"
    back = report["envelope_intrusion"]["by_face"]["x-"]
    assert back["median_depth_m"] == pytest.approx(0.03, abs=0.004)
    assert set(report["envelope_intrusion"]["by_face"]) == {"x-"}
    rest = report["rest_overlap"]
    assert rest["max_depth_m"] == pytest.approx(0.03, abs=0.004)
    push = rest["clearing_translation"]
    # The shortest way out of the wall is straight forward, 3 cm.
    assert push["distance_m"] == pytest.approx(0.03, abs=0.005)
    assert push["direction_world"][0] > 0.9
    assert "scene_physics_envelope_intrusion:x-:3.0cm" in report["blockers"]


def test_door_meeting_the_floor_at_its_limit_is_found(tmp_path):
    asset = _asset(tmp_path)
    clear = spp.scene_physics_preflight(background_usd=_background(tmp_path), asset_usd=asset,
                                        asset_pose_world=POSE, **FAST)
    assert clear["motion_clearance"][0]["first_contact"] is None
    (tmp_path / "collision.usda").unlink()
    bumped = spp.scene_physics_preflight(background_usd=_background(tmp_path, floor_bump=True), asset_usd=asset,
                                         asset_pose_world=POSE, **FAST)
    motion = bumped["motion_clearance"][0]
    assert motion["joint_id"] == "door_hinge" and motion["moving_bodies"] == ["door"]
    assert math.degrees(motion["first_contact"]["value"]) >= 80
    assert bumped["blockers"] == ["scene_physics_motion_contact:door_hinge@90deg"]
    # The resting cabinet never touches the bump: only the motion is blocked.
    assert bumped["rest_overlap"]["penetrating_points"] == 0


def test_mount_to_world_is_read_as_mounted(tmp_path):
    model = spp.load_asset_model(_asset(tmp_path, mounted=True))
    assert model.mounted is True and model.root_body == "/Asset/base"
    free = tmp_path / "free"
    free.mkdir()
    assert spp.load_asset_model(_asset(free)).mounted is False


def test_door_kinematics_open_forward_about_the_hinge(tmp_path):
    model = spp.load_asset_model(_asset(tmp_path))
    open_door = spp.body_poses(model, {"door_hinge": math.pi / 2})["/Asset/door"]
    assert open_door[:3, 3] == pytest.approx([0.32, 0.0, 0.05], abs=1e-6)
    assert open_door[:3, 2] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_unsupported_collider_is_refused(tmp_path):
    def capsule(stage):
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Capsule.Define(stage, "/Asset/base/capsule").GetPrim())

    with pytest.raises(spp.ScenePhysicsPreflightError, match="collider_type_unsupported:Capsule"):
        spp.load_asset_model(_asset(tmp_path, extra_collider=capsule))


def _packet(tmp_path, background, asset):
    root = tmp_path / "packet"
    (root / "assets").mkdir(parents=True)
    for source, name in ((background, "collision.usda"), (asset, "replacement.usda")):
        (root / "assets" / name).write_bytes(source.read_bytes())
    plan = {"plan_digest": "sha256:" + "a" * 64, "objects": [
        {"name": "scene_collision", "semantic_role": "scene_collision", "object_type": "BASE",
         "usd_path": "assets/collision.usda", "sha256": spp._file_sha256(root / "assets" / "collision.usda"),
         "pose_world": POSE},
        {"name": "task_object", "semantic_role": "task_object", "object_type": "ARTICULATION",
         "usd_path": "assets/replacement.usda", "sha256": spp._file_sha256(root / "assets" / "replacement.usda"),
         "pose_world": POSE, "reset_state": {"root_pose_world": POSE, "joint_positions": {"door_hinge": 0.0}}}]}
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    return root, plan


def test_packet_entry_checks_every_articulation_and_binds_its_bytes(tmp_path, capsys):
    root, plan = _packet(tmp_path, _background(tmp_path, back_wall_x=-0.27), _asset(tmp_path))
    report = spp.preflight_native_packet(root, **FAST)
    assert report["status"] == "blocked" and report["arena_scene_plan_digest"] == plan["plan_digest"]
    assert report["blockers"][0].startswith("task_object:scene_physics_rest_overlap:")
    assert spp.main(["--packet", str(root)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    plan["objects"][1]["sha256"] = "sha256:" + "0" * 64
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    with pytest.raises(spp.ScenePhysicsPreflightError, match="asset_digest_mismatch:task_object"):
        spp.preflight_native_packet(root, **FAST)
