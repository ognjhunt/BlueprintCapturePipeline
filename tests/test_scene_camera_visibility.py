"""CPU camera visibility: the splat occlusion that stopped the 2026-10-25 canary, on a synthetic scene.

A 0.6 x 0.6 x 0.9 m cabinet with a front door stands at the origin. Its sealed contact
is on the door face at (0.34, 0, 0.7). An external policy camera 1.8 m in front looks
straight at it. A splat either stays behind the cabinet or puts a sheet of Gaussians
between the camera and the contact.
"""
import json

import pytest

from blueprint_pipeline import scene_camera_visibility as scv
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.scene_physics_preflight import _file_sha256

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

POSE = {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
# OpenCV: looking along -x, image right along +y, image down along -z.
FACING = [[0, 0, -1, 1.8], [1, 0, 0, 0], [0, -1, 0, 0.9], [0, 0, 0, 1]]
AWAY = [[0, 0, 1, 1.8], [-1, 0, 0, 0], [0, -1, 0, 0.9], [0, 0, 0, 1]]
INTRINSICS = {"fx": 300.0, "fy": 300.0, "cx": 159.5, "cy": 119.5, "width": 320, "height": 240}


def _stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _cabinet(path):
    stage = _stage(path)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Asset").GetPrim())
    for body, size, centre in (("base", (0.6, 0.6, 0.9), (0.0, 0.0, 0.45)), ("door", (0.04, 0.6, 0.8), (0.32, 0.0, 0.45))):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Asset/{body}").GetPrim())
        cube = UsdGeom.Cube.Define(stage, f"/Asset/{body}/col")
        cube.GetSizeAttr().Set(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/door_hinge")
    hinge.CreateBody0Rel().SetTargets(["/Asset/base"])
    hinge.CreateBody1Rel().SetTargets(["/Asset/door"])
    hinge.CreateLocalPos0Attr().Set(Gf.Vec3f(0.32, 0.0, 0.05))
    hinge.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    hinge.CreateAxisAttr().Set("Y")
    hinge.CreateLowerLimitAttr().Set(0.0)
    hinge.CreateUpperLimitAttr().Set(90.0)
    stage.GetRootLayer().Save()


def _splat(path, *, sheet_x=None):
    """A wall of Gaussians behind the cabinet, plus an optional sheet in front of the contact."""
    points = [(-0.5, y / 20, z / 20) for y in range(-12, 13) for z in range(0, 30)]
    if sheet_x is not None:
        points += [(sheet_x, y / 100, z / 100) for y in range(-20, 21) for z in range(55, 86)]
    stage = _stage(path)
    prim = stage.DefinePrim("/World/Gaussians")
    prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([Gf.Vec3f(*p) for p in points])
    prim.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set([Gf.Vec3f(0.01, 0.01, 0.01)] * len(points))
    prim.CreateAttribute("opacities", Sdf.ValueTypeNames.FloatArray).Set([0.9] * len(points))
    stage.GetRootLayer().Save()


def _packet(tmp_path, *, sheet_x=None, external=FACING, task_kind="articulated_open_close"):
    root = tmp_path / "packet"
    (root / "assets").mkdir(parents=True)
    _cabinet(root / "assets" / "replacement.usda")
    _splat(root / "assets" / "appearance.usda", sheet_x=sheet_x)
    affordance = {"joint_contact_path": [
        {"joint_positions": {"door_hinge": 0.0}, "contact_pose_asset_root": {"position_m": [0.34, 0.0, 0.7]}},
        {"joint_positions": {"door_hinge": 0.5}, "contact_pose_asset_root": {"position_m": [0.6, 0.0, 0.6]}}],
        "arrival_tolerance_m": 0.02, "sweep_clearance_m": 0.025, "affordance_digest": ""}
    affordance["affordance_digest"] = canonical_digest(affordance, digest_field="affordance_digest")
    plan = {"plan_digest": "sha256:" + "a" * 64, "task_kind": task_kind,
            "task_spec": {"interaction_affordance": affordance,
                          "executable_opening_threshold": {"target_joint_id": "door_hinge", "reset_position": 0.0}},
            "appearance_frame_alignment": {"representation": scv.SPLAT_REPRESENTATION,
                                           "appearance_prim_path": "/World/Gaussians"},
            "objects": [
                {"semantic_role": "scene_appearance", "object_type": "BASE", "usd_path": "assets/appearance.usda",
                 "sha256": _file_sha256(root / "assets" / "appearance.usda"), "pose_world": POSE},
                {"semantic_role": "task_object", "object_type": "ARTICULATION", "task_subject": True,
                 "usd_path": "assets/replacement.usda", "sha256": _file_sha256(root / "assets" / "replacement.usda"),
                 "pose_world": POSE, "reset_state": {"root_pose_world": POSE, "joint_positions": {"door_hinge": 0.0}}}],
            "cameras": [
                {"role": "external", "pose_frame": "world", "policy_input": True, "review_only": False,
                 "optical_convention": "opencv", "frame_from_camera_matrix": sum(external, []), "intrinsics": INTRINSICS},
                {"role": "overview", "pose_frame": "world", "policy_input": False, "review_only": True,
                 "optical_convention": "opencv", "frame_from_camera_matrix": sum(FACING, []), "intrinsics": INTRINSICS},
                {"role": "wrist", "pose_frame": "robot_body", "policy_input": True, "review_only": False}]}
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    return root, plan


def test_clear_contact_passes_and_reports_each_camera(tmp_path):
    root, _ = _packet(tmp_path)
    report = scv.require_scene_camera_admission(root)
    assert report["status"] == "passed" and report["blockers"] == []
    external = next(row for row in report["cameras"] if row["role"] == "external")
    assert external["patch_in_view"] and external["patch_occluded_pixels"] == 0
    assert external["patch_asset_pixels"] >= 0.9 * external["patch_pixels"]
    assert report["notices"] == ["scene_camera_not_world_fixed:wrist"]


def test_scene_in_front_of_the_contact_blocks_and_review_camera_only_reports(tmp_path):
    root, _ = _packet(tmp_path, sheet_x=0.6)
    report = scv.scene_camera_visibility(root)
    assert report["status"] == "blocked"
    assert report["blockers"] == ["scene_camera_contact_occluded_by_scene:external:100%"]
    assert "scene_camera_contact_occluded_by_scene:overview:100%" in report["advisories"]
    assert report["summary"].startswith("Blocked: a policy camera cannot see the handle contact patch")
    with pytest.raises(scv.SceneCameraVisibilityBlocked):
        scv.require_scene_camera_admission(root)


def test_contact_outside_the_policy_camera_blocks(tmp_path):
    root, _ = _packet(tmp_path, external=AWAY)
    assert scv.scene_camera_visibility(root)["blockers"] == ["scene_camera_contact_outside_view:external"]


def test_unmeasurable_packet_is_refused_and_rigid_task_is_not_applicable(tmp_path):
    root, plan = _packet(tmp_path)
    plan["objects"][0]["sha256"] = "sha256:" + "0" * 64
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    with pytest.raises(scv.SceneCameraVisibilityBlocked, match="refused:scene_camera_packet_asset_unbound"):
        scv.require_scene_camera_admission(root)
    plan["task_kind"] = "rigid_pick_place"
    (root / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    report = scv.require_scene_camera_admission(root)
    assert report["status"] == "not_applicable" and report["reason"] == "no_articulated_task_subject"
