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


def _asset(tmp_path, *, mounted=False, extra_collider=None, door_scale=None, root_scale=None):
    path = tmp_path / "asset.usda"
    stage = _stage(path)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    if root_scale is not None:
        root.AddScaleOp().Set(Gf.Vec3f(root_scale, root_scale, root_scale))
    base = UsdGeom.Xform.Define(stage, "/Asset/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    _box(stage, "/Asset/base/col", size=(0.6, 0.6, 0.9), centre=(0.0, 0.0, 0.45))
    door = UsdGeom.Xform.Define(stage, "/Asset/door")
    door.AddTranslateOp().Set(Gf.Vec3d(0.32, 0.0, 0.05))
    if door_scale is not None:
        door.AddScaleOp().Set(Gf.Vec3f(door_scale, door_scale, door_scale))
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


def _background(tmp_path, *, back_wall_x=None, floor_bump=False, wall_disabled=False, floor_half=None):
    path = tmp_path / "collision.usda"
    stage = _stage(path)
    UsdGeom.Xform.Define(stage, "/World")
    if floor_half is None:
        _quad(stage, "/World/floor", [(-0.6, -0.6, 0.0), (1.4, -0.6, 0.0), (1.4, 0.6, 0.0), (-0.6, 0.6, 0.0)])
    else:
        h = floor_half
        _quad(stage, "/World/floor", [(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)])
    if back_wall_x is not None:
        x = back_wall_x
        _quad(stage, "/World/back_wall", [(x, -0.5, 0.0), (x, 0.5, 0.0), (x, 0.5, 1.1), (x, -0.5, 1.1)])
        if wall_disabled:
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/World/back_wall")).CreateCollisionEnabledAttr().Set(False)
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


def test_scale_above_a_body_is_refused(tmp_path):
    for name, options, path in (("door", {"door_scale": 2.0}, "/Asset/door"), ("root", {"root_scale": 2.0}, "/Asset")):
        folder = tmp_path / name
        folder.mkdir()
        with pytest.raises(spp.ScenePhysicsPreflightError, match="asset_scale_unsupported:" + path + "$"):
            spp.load_asset_model(_asset(folder, **options))


def test_disabled_background_collider_is_ignored(tmp_path):
    report = spp.scene_physics_preflight(background_usd=_background(tmp_path, back_wall_x=-0.27, wall_disabled=True),
                                         asset_usd=_asset(tmp_path), asset_pose_world=POSE, **FAST)
    assert report["status"] == "passed" and report["blockers"] == []


def test_huge_background_triangle_is_sampled_only_near_the_asset(tmp_path):
    # Two triangles 200 m across: sampled whole at 1 cm this would be about 4e8 points.
    report = spp.scene_physics_preflight(background_usd=_background(tmp_path, floor_half=100.0),
                                         asset_usd=_asset(tmp_path), asset_pose_world=POSE, **FAST)
    assert report["status"] == "passed"
    assert report["background_samples"] < 200_000


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


def test_contact_beyond_the_task_range_is_reported_without_blocking(tmp_path):
    asset, background = _asset(tmp_path), _background(tmp_path, floor_bump=True)

    def run(ranges):
        return spp.scene_physics_preflight(background_usd=background, asset_usd=asset, asset_pose_world=POSE,
                                           task_ranges=ranges, **FAST)

    beyond = run({"door_hinge": [0.0, math.radians(54)]})
    assert beyond["status"] == "passed" and beyond["blockers"] == []
    assert beyond["advisories"] == ["scene_physics_motion_contact_outside_task_range:door_hinge@90deg"]
    assert beyond["motion_clearance"][0]["task_range"] == [0.0, pytest.approx(math.radians(54))]
    assert run({"door_hinge": [0.0, math.radians(90)]})["blockers"] == ["scene_physics_motion_contact:door_hinge@90deg"]
    # A joint the task does not command only reports.
    passive = run({})
    assert passive["status"] == "passed" and passive["advisories"] == beyond["advisories"]
    # A goal between two sweep steps (85 and 90 degrees here) is sampled itself.
    between = run({"door_hinge": [0.0, math.radians(88)]})
    assert between["blockers"] == ["scene_physics_motion_contact:door_hinge@88deg"]
    assert between["advisories"] == ["scene_physics_motion_contact_outside_task_range:door_hinge@90deg"]


def test_task_range_never_excuses_rest_overlap_or_intrusion(tmp_path):
    report = spp.scene_physics_preflight(background_usd=_background(tmp_path, back_wall_x=-0.27),
                                         asset_usd=_asset(tmp_path), asset_pose_world=POSE, task_ranges={}, **FAST)
    assert report["status"] == "blocked"
    assert report["blockers"][0].startswith("scene_physics_rest_overlap:")
    assert any(b.startswith("scene_physics_envelope_intrusion:x-:") for b in report["blockers"])


def test_flat_or_empty_geometry_has_nothing_to_overlap(tmp_path):
    def flat(stage):
        mesh = UsdGeom.Mesh.Define(stage, "/Asset/door/flat")
        mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.1, 0.2, 0.3)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    model = spp.load_asset_model(_asset(tmp_path, extra_collider=flat))
    assert model.skipped_colliders == ("/Asset/door/flat",) and len(model.colliders) == 2
    empty = tmp_path / "empty.usda"
    stage = _stage(empty)
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Mesh.Define(stage, "/World/floor").GetPrim())
    stage.GetRootLayer().Save()
    report = spp.scene_physics_preflight(background_usd=empty, asset_usd=_asset(tmp_path), asset_pose_world=POSE, **FAST)
    assert report["status"] == "passed" and report["notices"] == ["scene_physics_background_empty_near_asset"]
    hollow = tmp_path / "hollow.usda"
    stage = _stage(hollow)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Asset").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Asset/base").GetPrim())
    flat_only = UsdGeom.Mesh.Define(stage, "/Asset/base/flat")
    flat_only.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.1, 0.2, 0.3)])
    UsdPhysics.CollisionAPI.Apply(flat_only.GetPrim())
    stage.GetRootLayer().Save()
    report = spp.scene_physics_preflight(background_usd=_background(tmp_path), asset_usd=hollow,
                                         asset_pose_world=POSE, **FAST)
    assert report["status"] == "passed" and report["notices"] == ["scene_physics_asset_solid_colliders_absent"]


def _with_task(root, plan, *, success_start_deg, robot_x=1.6):
    plan["task_spec"] = {"task_kind": "articulated_open_close", "executable_opening_threshold": {
        "target_joint_id": "door_hinge", "reset_position": 0.0,
        "success_interval": [math.radians(success_start_deg), math.radians(90)]}}
    # A Franka on the floor facing the cabinet; the open door reaches x = 1.14 m.
    plan["robot"] = {"robot_id": "franka_panda", "base_pose_world": {
        "position_world_m": [robot_x, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 1.0, 0.0]}}
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))


def test_admission_blocks_task_range_contact_and_reports_beyond_it(tmp_path):
    root, plan = _packet(tmp_path, _background(tmp_path, floor_bump=True), _asset(tmp_path))
    _with_task(root, plan, success_start_deg=54)
    report = spp.require_scene_physics_admission(root, **FAST)
    assert report["status"] == "passed" and report["blockers"] == []
    assert report["task_ranges"] == {"door_hinge": [0.0, pytest.approx(math.radians(54))]}
    assert report["advisories"] == ["task_object:scene_physics_motion_contact_outside_task_range:door_hinge@90deg"]
    _with_task(root, plan, success_start_deg=90)
    with pytest.raises(spp.ScenePhysicsAdmissionBlocked) as blocked:
        spp.require_scene_physics_admission(root, **FAST)
    assert blocked.value.blockers == ["task_object:scene_physics_motion_contact:door_hinge@90deg"]


def test_robot_in_the_doors_path_blocks_only_before_the_success_threshold(tmp_path):
    # A base the door strikes before reaching 54 degrees makes success impossible; one
    # it would strike only beyond (the 2026-10-26 shoulder at 60 degrees) is reported.
    root, plan = _packet(tmp_path, _background(tmp_path), _asset(tmp_path))
    _with_task(root, plan, success_start_deg=54, robot_x=0.75)
    with pytest.raises(spp.ScenePhysicsAdmissionBlocked) as blocked:
        spp.require_scene_physics_admission(root, **FAST)
    (blocker,) = blocked.value.blockers
    assert blocker.startswith("task_object:scene_physics_robot_in_motion_path:door_hinge@")
    assert int(blocker.rsplit("@", 1)[1].removesuffix("deg")) < 54
    _with_task(root, plan, success_start_deg=54, robot_x=1.0)
    beyond = spp.require_scene_physics_admission(root, **FAST)
    assert beyond["status"] == "passed" and beyond["robot_clearance"]["blockers"] == []
    (advisory,) = beyond["robot_clearance"]["advisories"]
    assert advisory.startswith("scene_physics_robot_in_motion_path_outside_task_range:door_hinge@")
    assert 54 < int(advisory.rsplit("@", 1)[1].removesuffix("deg")) < 90
    _with_task(root, plan, success_start_deg=54)
    report = spp.require_scene_physics_admission(root, **FAST)
    assert report["robot_clearance"]["status"] == "passed" and report["robot_clearance"]["advisories"] == []
    assert report["robot_clearance"]["rest_max_depth_m"] == 0.0
    assert report["robot_clearance"]["motion"]["task_range"] == [0.0, pytest.approx(math.radians(54))]
    _with_task(root, plan, success_start_deg=54, robot_x=0.1)
    with pytest.raises(spp.ScenePhysicsAdmissionBlocked, match="task_object:scene_physics_robot_rest_overlap:"):
        spp.require_scene_physics_admission(root, **FAST)
    for robot, refusal in (({"robot_id": "ur5e", "base_pose_world": POSE}, "robot_model_unmeasured:ur5e"),
                           ({"robot_id": "franka_panda"}, "robot_pose_missing")):
        plan["robot"] = robot
        (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
        with pytest.raises(spp.ScenePhysicsAdmissionBlocked, match="refused:scene_physics_" + refusal):
            spp.require_scene_physics_admission(root, **FAST)


def test_pre_v2_task_spec_commands_its_scripted_target_and_an_unreadable_span_is_refused():
    legacy = {"task_spec": {"task_kind": "articulated_open_close", "target_joint_id": "door",
                            "joint_reset_positions_rad": {"door": 0.1}, "scripted_positive_target_rad": 0.9},
              "articulation": {"task_joint_prim_paths": {"door": "{ENV_REGEX_NS}/task_object/joints/door_hinge"}}}
    assert spp.task_motion_ranges(legacy) == {"door_hinge": [0.1, 0.9]}
    del legacy["task_spec"]["scripted_positive_target_rad"]
    with pytest.raises(spp.ScenePhysicsPreflightError, match="scene_physics_task_range_unknown"):
        spp.task_motion_ranges(legacy)
    assert spp.task_motion_ranges({"task_spec": {"task_kind": "rigid_pick_place"}}) == {}


def test_admission_refuses_what_it_cannot_measure_and_skips_rigid_scenes(tmp_path):
    root, plan = _packet(tmp_path, _background(tmp_path), _asset(tmp_path))
    plan["objects"][1]["sha256"] = "sha256:" + "0" * 64
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    with pytest.raises(spp.ScenePhysicsAdmissionBlocked, match="refused:scene_physics_packet_asset_digest_mismatch"):
        spp.require_scene_physics_admission(root, **FAST)
    plan["objects"][1]["object_type"] = "BASE"
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    report = spp.require_scene_physics_admission(root)
    assert report["status"] == "not_applicable" and report["reason"] == "no_articulated_object"
