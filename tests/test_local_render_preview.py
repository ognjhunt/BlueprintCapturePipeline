"""Hermetic tests for the no-GPU local dry-render preview.

The dry-render reproduces the SAME stance + camera + arm-skeleton math the GPU runner uses, so
placement/camera/POV-framing bugs (wrong stance side, camera cropping the arm, aiming into an
appliance) are caught locally in <1s before any cloud render. Importing the runner must NOT pull
in isaacsim, and the preview itself must run with only pxr + PIL (no Isaac, no GPU).
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("parity_runner_preview", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # raises if it imported isaacsim at module load
    return mod


M = _load()


def _dist(a, b) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def test_nominal_g1_rest_offsets_arm_reaches_toward_target() -> None:
    # The nominal (Isaac-free) skeleton must expose arm links the reach solver recognizes, and posing
    # the right arm toward a target must move the hand toward it while the shoulder stays put.
    offsets = M.nominal_g1_rest_offsets()
    skel = M._rest_skeleton_world(offsets, (0.0, 0.0, 0.79), 0.0)
    by_name = dict(skel)
    assert "right_shoulder_link" in by_name
    assert "right_hand_link" in by_name

    target = (0.6, -0.18, 1.0)  # in front, slightly to the robot's right, counter height
    reached = dict(M.compute_arm_reach_skeleton(skel, target, 1.0, arm="right"))

    assert _dist(reached["right_hand_link"], target) < _dist(by_name["right_hand_link"], target)
    assert reached["right_shoulder_link"] == pytest.approx(by_name["right_shoulder_link"])


def test_pov_framing_discriminates_arm_in_frame_vs_cropped() -> None:
    # This is the bug class that cost repeated GPU rounds: a POV camera that crops the reaching arm.
    # The local projection must SHOW the arm in frame for the manipulation camera, and must report it
    # OUT of frame for a camera aimed away — otherwise the preview can't catch the cropping bug.
    root, yaw = (0.0, 0.0, 0.79), 0.0
    look_at = (0.55, -0.15, 1.0)
    skel = M._rest_skeleton_world(M.nominal_g1_rest_offsets(), root, yaw)
    skel = M.compute_arm_reach_skeleton(skel, look_at, 1.0, arm="right")

    eye, tgt = M.manipulation_cam_pose(root, yaw, look_at=look_at, reach_arm="right")
    lms = M._project_skeleton(skel, eye=eye, target=tgt, up=(0.0, 0.0, 1.0),
                              vfov_deg=68.0, width=1280, height=960)
    ids = {l["landmark_id"] for l in lms}
    assert any(("right_hand" in i) or ("right_wrist" in i) for i in ids), ids

    # Camera planted in front of the robot looking further forward: the arm is behind the eye.
    bad = M._project_skeleton(skel, eye=(2.0, 0.0, 1.2), target=(4.0, 0.0, 1.0),
                              up=(0.0, 0.0, 1.0), vfov_deg=68.0, width=1280, height=960)
    bad_ids = {l["landmark_id"] for l in bad}
    assert not any("right_hand" in i for i in bad_ids), bad_ids


def _fridge_resolution():
    # Fridge front opens toward -x (door at x = bbox_min[0]); robot should stand in front facing -x.
    return {
        "status": "resolved",
        "source": "stage_label_match",
        "selected": {
            "target_object_id": "refrigerator",
            "target_object_label": "refrigerator",
            "center_xyz": [-1.98, 0.66, 1.03],
            "size_xyz": [0.88, 1.73, 2.06],
            "bbox_min_xyz": [-2.42, -0.21, 0.0],
            "bbox_max_xyz": [-1.54, 1.52, 2.06],
        },
    }


def test_render_local_preview_writes_artifacts_and_frames_target(monkeypatch, tmp_path) -> None:
    pytest.importorskip("pxr")
    pil = pytest.importorskip("PIL.Image")
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(M, "_resolve_task_target_from_stage", lambda _s, _sc: _fridge_resolution())

    summary = M.render_local_preview(
        stage=stage,
        scenario={"scenario_id": "open_fridge", "description": "open the refrigerator",
                  "floor_z_hint": 0.0},
        out_dir=tmp_path,
        manipulation_reach_arm="right",
        camera_vfov_deg=50.0,
        width=640,
        height=480,
    )

    png = tmp_path / "dry_render_preview.png"
    js = tmp_path / "dry_render_summary.json"
    assert png.exists() and png.stat().st_size > 0
    assert js.exists()
    persisted = json.loads(js.read_text())
    assert persisted["stance"]["status"] == "accepted"
    assert persisted["render_source"] == M.DRY_RENDER_SOURCE_MARKER
    assert persisted["render_provenance"]["X-Blueprint-Render-Source"] == "dry_render_preview"
    assert "NOT a rendered frame" in persisted["render_provenance"]["X-Blueprint-Render-Note"]
    with pil.open(png) as img:
        assert img.info["X-Blueprint-Render-Source"] == "dry_render_preview"
        assert "NOT a rendered frame" in img.info["X-Blueprint-Render-Note"]

    assert summary["stance"]["status"] == "accepted"
    # Robot stands clear of the fridge footprint (not inside the appliance volume) and faces it.
    pose = summary["stance"]["accepted_pose"]
    bmin = _fridge_resolution()["selected"]["bbox_min_xyz"]
    bmax = _fridge_resolution()["selected"]["bbox_max_xyz"]
    inside = (bmin[0] <= pose[0] <= bmax[0]) and (bmin[1] <= pose[1] <= bmax[1])
    assert not inside
    assert summary["checks"]["facing_error_deg"] < 5.0
    assert summary["pov_framing"]["target_in_frame"] is True
    assert summary["pov_framing"]["arm_landmarks_in_frame"] >= 1


def _real_kitchen_usd():
    base = Path(__file__).resolve().parents[1]
    for rel in (
        "output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/"
        "captures/downloads-walkthrough2-20260611/pipeline/lightwheel_kitchen_isaac_scenarios/"
        "assets/Collected_KitchenRoom/KitchenRoom.usd",
        "output/isaac_g1_dynamic_standing_contact_floor_asset/Collected_KitchenRoom/KitchenRoom.usd",
    ):
        p = base / rel
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Synthetic kitchen fixture (NOT real captured geometry — a hand-authored test
# stage). It exists so the CPU dry-render iteration loop is exercised in EVERY
# clean checkout, including ones without the heavy ~177MB real KitchenRoom asset.
# The fridge sits in a 3-wall niche so only the +x (door) side is clear, which
# forces the placement validator to make the SAME room-side/door-side stance
# choice the real kitchen forces — the exact bug class (stance behind the
# fridge against the wall) that cost a GPU round trip. Authored via pxr/usd-core
# so no large binary fixture is committed.
# ---------------------------------------------------------------------------
_SYNTH_FRIDGE_CENTER = (-1.98, 0.66, 1.03)
_SYNTH_FRIDGE_SIZE = (0.88, 1.73, 2.06)


def _author_synthetic_kitchen_prims(stage) -> None:
    """Author a floor + a refrigerator-in-a-niche onto ``stage`` (a TEST fixture, not real capture).

    The prim is named ``Refrigerator`` so the swappable ``scene_placement`` resolver maps the task
    string "open the refrigerator" onto it with no monkeypatching — the full target-resolution chain
    runs. Three walls (-x back, +/-y sides) leave only the +x door side open so the validator must
    stand the robot in front of the door, matching the real kitchen's door-side choice.
    """
    from pxr import Gf, UsdGeom  # type: ignore

    UsdGeom.Xform.Define(stage, "/World")

    def _box(path, translate, scale):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
        xf.AddScaleOp().Set(Gf.Vec3d(*scale))

    _box("/World/Floor", (0.0, 0.0, -0.025), (8.0, 8.0, 0.05))
    fc = _SYNTH_FRIDGE_CENTER
    _box("/World/BackWall", (fc[0] - 0.9, fc[1], 1.2), (0.12, 3.0, 2.4))
    _box("/World/LeftWall", (fc[0], fc[1] + 1.4, 1.2), (2.2, 0.12, 2.4))
    _box("/World/RightWall", (fc[0], fc[1] - 1.4, 1.2), (2.2, 0.12, 2.4))
    UsdGeom.Xform.Define(stage, "/World/Refrigerator")
    _box("/World/Refrigerator/body", fc, _SYNTH_FRIDGE_SIZE)


def _synthetic_kitchen_stage():
    """In-memory synthetic kitchen stage for the dry-render door-side regression (no real asset needed)."""
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    _author_synthetic_kitchen_prims(stage)
    return stage


def _synthetic_kitchen_usd_file(tmp_path) -> Path:
    """Write the synthetic kitchen to a ``.usda`` FILE so the ``--dry-render`` CLI (which opens a USD
    path, not an in-memory stage) can run end-to-end without the heavy real asset. Clearly labeled
    SYNTHETIC so it can never be mistaken for real captured geometry."""
    from pxr import Usd  # type: ignore

    path = tmp_path / "SyntheticKitchenRoom.usda"
    stage = Usd.Stage.CreateNew(str(path))
    _author_synthetic_kitchen_prims(stage)
    stage.SetMetadata("comment", "SYNTHETIC TEST FIXTURE — not real captured geometry")
    stage.GetRootLayer().Save()
    return path


def _kitchen_stage_real_or_synthetic():
    """Prefer the real KitchenRoom asset; fall back to the synthetic in-memory stage. Returns
    ``(stage, is_real)`` so the door-side regression runs in every checkout."""
    usd = _real_kitchen_usd()
    if usd is not None:
        return M._open_stage_local(str(usd)), True
    return _synthetic_kitchen_stage(), False


def _kitchen_usd_path_real_or_synthetic(tmp_path):
    """Prefer the real KitchenRoom asset path; fall back to a written synthetic ``.usda`` FILE. Returns
    ``(path, is_real)`` for CLI tests that need a USD path argument."""
    usd = _real_kitchen_usd()
    if usd is not None:
        return usd, True
    return _synthetic_kitchen_usd_file(tmp_path), False


def _write_synthetic_g1_asset(path: Path, *, renderable: bool) -> Path:
    """Write a tiny G1-like USD asset for local dry-render visual diagnostics.

    ``renderable=False`` intentionally creates articulation/collision/link Xforms with zero Gprims,
    reproducing the physics-only G1 composition that must fail closed. ``renderable=True`` adds a
    simple torso cube so the same traversal sees renderable geometry. This is a test fixture only.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/G1")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, xyz in {
        "right_shoulder_link": (0.18, -0.16, 0.36),
        "right_elbow_link": (0.34, -0.18, 0.30),
        "right_wrist_link": (0.48, -0.19, 0.25),
        "right_hand_link": (0.58, -0.19, 0.25),
        "left_shoulder_link": (0.18, 0.16, 0.36),
        "left_elbow_link": (0.34, 0.18, 0.30),
        "left_wrist_link": (0.48, 0.19, 0.25),
        "left_hand_link": (0.58, 0.19, 0.25),
        "head_link": (0.06, 0.0, 0.48),
    }.items():
        link = UsdGeom.Xform.Define(stage, f"/G1/{name}")
        link.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        UsdPhysics.CollisionAPI.Apply(link.GetPrim())
    if renderable:
        torso = UsdGeom.Cube.Define(stage, "/G1/torso_mesh")
        torso.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(torso.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.02, 0.0, 0.18))
        xf.AddScaleOp().Set(Gf.Vec3d(0.24, 0.18, 0.36))
    stage.GetRootLayer().Save()
    return path


def test_dry_render_artifact_snapshot_on_real_kitchen(tmp_path) -> None:
    # The high-fidelity path: a bound proxy robot + the geometric placement validator must reproduce
    # the real GPU run's side choice for "open the refrigerator" — standing in front of the open door
    # (toward the room) and facing the fridge — NOT behind it against the wall. This is the bug class
    # that cost a full GPU round trip to discover. Runs against the real KitchenRoom asset when present,
    # else a synthetic 3-wall-niche fridge stage that forces the SAME +x door-side stance — so this
    # regression executes in every clean checkout instead of skipping.
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")

    stage, _is_real = _kitchen_stage_real_or_synthetic()
    M._bind_proxy_robot(stage, "/World/G1")
    summary = M.render_local_preview(
        stage=stage,
        scenario={"scenario_id": "open_fridge", "description": "open the refrigerator",
                  "floor_z_hint": 0.0},
        out_dir=tmp_path,
        manipulation_reach_arm="right",
        camera_vfov_deg=50.0,
        width=640,
        height=480,
        robot_prim_path="/World/G1",
    )

    assert summary["stance"]["status"] == "accepted"
    checks = M._dry_render_checks(summary)
    assert checks["faces_target"] is True
    assert checks["target_in_frame"] is True
    assert checks["arm_in_frame"] is True
    assert checks["no_blockers"] is True
    assert checks["robot_visual_mesh_present"] is True
    assert checks["camera_pitch_within_cap"] is True
    assert checks["pov_geometry_pass"] is True
    center = summary["target"]["center_xyz"]
    pose = summary["stance"]["accepted_pose"]
    # Stand on the room side of the fridge center (the open-door side), matching the real run.
    assert pose[0] > center[0]
    assert summary["checks"]["facing_error_deg"] < 8.0
    assert summary["pov_framing"]["target_in_frame"] is True
    png = tmp_path / "dry_render_preview.png"
    js = tmp_path / "dry_render_summary.json"
    assert png.exists() and png.stat().st_size > 0
    assert js.exists()
    persisted = json.loads(js.read_text())
    assert persisted["render_source"] == "dry_render_preview"
    assert persisted["render_provenance"]["X-Blueprint-Render-Source"] == "dry_render_preview"
    # Provenance must also live in the PNG metadata so a screenshotted CPU proxy cannot be filed as a
    # real Isaac render.
    from PIL import Image  # type: ignore

    info = Image.open(png).info
    assert info.get("X-Blueprint-Render-Source") == "dry_render_preview"
    assert "NOT a rendered frame" in info.get("X-Blueprint-Render-Note", "")


def test_dry_render_checks_treat_zero_facing_error_as_passing() -> None:
    # A perfect 0.0 deg facing must read as PASS — guards the `0.0 or 99` falsy trap in the checklist.
    checks = M._dry_render_checks({
        "checks": {"facing_error_deg": 0.0},
        "pov_framing": {"target_in_frame": True, "arm_landmarks_in_frame": 3},
        "stance": {"status": "accepted", "blockers": None},
    })
    assert checks["faces_target"] is True
    assert checks["target_in_frame"] is True
    assert checks["arm_in_frame"] is True
    assert checks["no_blockers"] is True
    assert checks["robot_visual_mesh_present"] is True
    assert checks["camera_pitch_within_cap"] is True
    assert checks["pov_geometry_pass"] is True

    bad = M._dry_render_checks({
        "checks": {"facing_error_deg": 35.0},
        "pov_framing": {"target_in_frame": False, "arm_landmarks_in_frame": 0},
        "stance": {"status": "blocked", "blockers": ["x"]},
        "robot_visual_geometry": {"renderable_robot_geometry_present": False},
        "manipulation_pov_geometry": {"status": "FAIL"},
    })
    assert bad["faces_target"] is False
    assert bad["arm_in_frame"] is False
    assert bad["no_blockers"] is False
    assert bad["robot_visual_mesh_present"] is False
    assert bad["pov_geometry_pass"] is False


def test_dry_render_optional_g1_meshless_asset_fails_visual_gate(monkeypatch, tmp_path) -> None:
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(M, "_resolve_task_target_from_stage", lambda _s, _sc: _fridge_resolution())
    g1_asset = _write_synthetic_g1_asset(tmp_path / "g1_meshless.usd", renderable=False)
    binding = M._bind_g1_with_visual_fallback(stage, str(g1_asset), prim_path="/World/G1")

    summary = M.render_local_preview(
        stage=stage,
        scenario={"scenario_id": "open_fridge", "description": "open the refrigerator",
                  "floor_z_hint": 0.0},
        out_dir=tmp_path / "meshless",
        manipulation_reach_arm="right",
        camera_vfov_deg=50.0,
        width=640,
        height=480,
        robot_visual_prim_path="/World/G1",
        robot_binding=binding,
    )

    assert binding["visual_binding_status"] == "blocked_missing_renderable_robot_geometry"
    assert summary["robot_visual_geometry"]["renderable_robot_geometry_present"] is False
    assert M.ROBOT_VISUAL_MESH_MISSING_BLOCKER in summary["robot_visual_geometry"]["blockers"]
    checks = M._dry_render_checks(summary)
    assert checks["robot_visual_mesh_present"] is False


def test_dry_render_optional_g1_renderable_asset_passes_visual_gate(monkeypatch, tmp_path) -> None:
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(M, "_resolve_task_target_from_stage", lambda _s, _sc: _fridge_resolution())
    g1_asset = _write_synthetic_g1_asset(tmp_path / "g1_visual.usd", renderable=True)
    binding = M._bind_g1_with_visual_fallback(stage, str(g1_asset), prim_path="/World/G1")

    summary = M.render_local_preview(
        stage=stage,
        scenario={"scenario_id": "open_fridge", "description": "open the refrigerator",
                  "floor_z_hint": 0.0},
        out_dir=tmp_path / "visual",
        manipulation_reach_arm="right",
        camera_vfov_deg=50.0,
        width=640,
        height=480,
        robot_visual_prim_path="/World/G1",
        robot_binding=binding,
    )

    assert binding["visual_binding_status"] == "renderable_robot_geometry_found"
    assert summary["robot_visual_geometry"]["renderable_robot_geometry_present"] is True
    checks = M._dry_render_checks(summary)
    assert checks["robot_visual_mesh_present"] is True
    assert checks["camera_pitch_within_cap"] is True


def test_dry_render_cli_flag_is_parsed() -> None:
    args = M.build_arg_parser().parse_args(
        ["--request", "/x.json", "--out-dir", "/y", "--dry-render"]
    )
    assert args.dry_render is True


def test_add_workspace_fill_light_is_idempotent_on_warm_stage() -> None:
    # In a long-lived --serve worker the stage is reused across jobs, so per-scenario setup must be
    # idempotent. Run 1 surfaced this: the 2nd call raised "xformOp:translate already exists".
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    M._add_workspace_fill_light(stage, (1.0, 2.0, 1.0), intensity=30000.0)
    M._add_workspace_fill_light(stage, (1.5, 2.5, 1.0), intensity=30000.0)  # warm reuse must not raise

    prim = stage.GetPrimAtPath("/World/WorkspaceFill")
    assert prim.IsValid()
    translate_ops = [o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()
                     if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    assert len(translate_ops) == 1  # reused, not duplicated
    val = translate_ops[0].Get()
    assert abs(float(val[0]) - 1.5) < 1e-6  # refreshed to the latest target


def test_pose_arm_kinematic_extends_hand_forward_not_just_upper_arm() -> None:
    # The seed pose must swing the whole arm forward so the gripper is visible in the POV. It should
    # align shoulder->HAND toward the workspace direction, not merely aim the upper arm. This is an
    # initial manipulation-ready pose, not a contact/task-completion claim.
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, Gf  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/G1")
    sh = UsdGeom.Xform.Define(stage, "/World/G1/right_shoulder_link")
    UsdGeom.Xformable(sh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, -0.18, 0.55))
    el = UsdGeom.Xform.Define(stage, "/World/G1/right_shoulder_link/right_elbow_link")
    UsdGeom.Xformable(el.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.25))
    ha = UsdGeom.Xform.Define(stage, "/World/G1/right_shoulder_link/right_elbow_link/right_hand_link")
    UsdGeom.Xformable(ha.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.25))  # arm hangs straight down

    def hand_world():
        t = UsdGeom.XformCache().GetLocalToWorldTransform(ha.GetPrim()).ExtractTranslation()
        return (float(t[0]), float(t[1]), float(t[2]))

    target = (0.45, -0.18, 0.55)  # forward at shoulder height
    before = _dist(hand_world(), target)
    posed = M._pose_arm_kinematic_usd(stage, "/World/G1", target, arm="right", reach_frac=1.0)
    assert posed == 1
    after = _dist(hand_world(), target)
    assert after < before * 0.4  # hand swung from hanging-down to forward workspace direction
    assert hand_world()[0] > 0.35


def test_pov_headlamp_is_idempotent_and_front_of_camera() -> None:
    # The manipulation POV camera sees the arm's shadow side (fill light is at the door, beyond the
    # arm). A camera-side headlamp front-lights the arm+gripper. Must be idempotent on the warm stage
    # and sit between the camera eye and the look-at (so it lights what the camera sees).
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    eye = (-1.10, 0.66, 1.29)
    look_at = (-1.44, 0.66, 1.03)
    M._add_pov_headlamp(stage, eye, look_at, intensity=20000.0)
    M._add_pov_headlamp(stage, (-1.00, 0.66, 1.29), look_at, intensity=20000.0)  # warm reuse: no raise

    prim = stage.GetPrimAtPath("/World/PovHeadlamp")
    assert prim.IsValid()
    translate_ops = [o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()
                     if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    assert len(translate_ops) == 1  # reused, not duplicated
    pos = translate_ops[0].Get()
    # lamp sits toward the look-at from the (latest) camera eye, i.e. between camera and workspace
    assert -1.44 <= float(pos[0]) <= -1.00
    # MUST be soft (large radius, capped intensity) so the close camera-side fill is not a firefly
    # source on the nearby arm — passing the bright 30000 workspace value must be clamped.
    from pxr import UsdLux  # type: ignore
    light = UsdLux.SphereLight(prim)
    assert float(light.GetRadiusAttr().Get()) >= 0.4
    assert float(light.GetIntensityAttr().Get()) <= 6000.0


def test_robot_review_material_binds_only_robot_geometry() -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdShade  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/G1")
    robot_mesh = UsdGeom.Mesh.Define(stage, "/World/G1/torso_mesh")
    scene_mesh = UsdGeom.Mesh.Define(stage, "/World/Refrigerator/door_mesh")
    root_img = UsdGeom.Imageable(root.GetPrim())
    root_img.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    robot_img = UsdGeom.Imageable(robot_mesh.GetPrim())
    robot_img.GetPurposeAttr().Set("proxy")

    bound = M._apply_robot_review_material(
        stage,
        "/World/G1",
        override_authored_materials=True,
    )
    diag = M._robot_render_visibility_diagnostics(stage, "/World/G1")

    assert bound >= 2
    root_binding = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath("/World/G1")
    ).ComputeBoundMaterial()[0]
    robot_binding = UsdShade.MaterialBindingAPI(robot_mesh.GetPrim()).ComputeBoundMaterial()[0]
    scene_binding = UsdShade.MaterialBindingAPI(scene_mesh.GetPrim()).ComputeBoundMaterial()[0]
    assert root_binding and str(root_binding.GetPath()) == "/World/Materials/RobotReviewVisible"
    assert robot_binding and str(robot_binding.GetPath()) == "/World/Materials/RobotReviewVisible"
    assert not scene_binding
    assert root_img.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert robot_img.GetPurposeAttr().Get() == "default"
    assert diag["status"] == "PASS"
    assert diag["gprim_count"] == 1
    assert diag["material_bound_gprim_count"] == 1


def test_robot_material_preservation_normalizes_visibility_without_override() -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdShade  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/G1")
    robot_mesh = UsdGeom.Mesh.Define(stage, "/World/G1/torso_mesh")
    root_img = UsdGeom.Imageable(root.GetPrim())
    root_img.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    robot_img = UsdGeom.Imageable(robot_mesh.GetPrim())
    robot_img.GetPurposeAttr().Set("proxy")

    bound = M._apply_robot_review_material(
        stage,
        "/World/G1",
        override_authored_materials=False,
    )

    root_binding = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath("/World/G1")
    ).ComputeBoundMaterial()[0]
    robot_binding = UsdShade.MaterialBindingAPI(robot_mesh.GetPrim()).ComputeBoundMaterial()[0]
    assert bound == 0
    assert not root_binding
    assert not robot_binding
    assert root_img.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert robot_img.GetPurposeAttr().Get() == "default"


def test_robot_render_visibility_fails_closed_without_gprims() -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/G1")
    UsdGeom.Xform.Define(stage, "/World/G1/right_wrist_link")

    diag = M._robot_render_visibility_diagnostics(stage, "/World/G1")

    assert diag["status"] == "FAIL"
    assert M.ROBOT_VISUAL_MESH_MISSING_BLOCKER in diag["blockers"]
    assert diag["gprim_count"] == 0
    assert diag["renderable_robot_geometry_present"] is False


def test_robot_review_visual_proxies_use_link_geometry_without_scene_coords(monkeypatch) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/G1")
    for name, xyz in {
        "right_shoulder_link": (0.0, -0.16, 1.14),
        "right_elbow_link": (-0.12, -0.18, 1.06),
        "right_wrist_link": (-0.24, -0.20, 1.02),
        "right_hand_link": (-0.34, -0.22, 1.02),
        "left_shoulder_link": (0.0, 0.16, 1.14),
        "left_elbow_link": (-0.12, 0.18, 1.06),
        "left_wrist_link": (-0.24, 0.20, 1.02),
        "left_hand_link": (-0.34, 0.22, 1.02),
    }.items():
        prim = UsdGeom.Xform.Define(stage, f"/World/G1/{name}")
        prim.AddTranslateOp().Set(xyz)

    bbox = {
        "bbox_min_xyz": [-0.25, -0.22, 0.05],
        "bbox_max_xyz": [0.18, 0.22, 1.38],
        "center_xyz": [-0.035, 0.0, 0.715],
        "size_xyz": [0.43, 0.44, 1.33],
    }
    monkeypatch.setattr(M, "_world_bbox_for_prim", lambda _stage, _path: bbox)

    diag = M._create_robot_review_visual_proxies(
        stage,
        "/World/G1",
        proxy_root_path="/World/RobotReviewVisualProxies/test",
        arm="both",
    )

    assert diag["status"] == "PASS"
    assert diag["created_gprim_count"] >= 10
    assert diag["source_robot_prim_path"] == "/World/G1"
    assert "render aids, not collision" in diag["claim_boundary"]
    assert stage.GetPrimAtPath("/World/RobotReviewVisualProxies/test/right_wrist_to_hand").IsValid()
    assert stage.GetPrimAtPath("/World/RobotReviewVisualProxies/test/left_wrist_to_hand").IsValid()
    assert not stage.GetPrimAtPath("/World/G1/right_wrist_to_hand")


def test_robot_neutral_descendant_xforms_restore_warm_stage_mutation() -> None:
    # Warm serve reuses one USD stage. A reach pose must not compound into the next job/frame.
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    root = UsdGeom.Xform.Define(stage, "/World/G1")
    root_xf = UsdGeom.Xformable(root.GetPrim())
    root_xf.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    link = UsdGeom.Xform.Define(stage, "/World/G1/right_shoulder_link")
    link_xf = UsdGeom.Xformable(link.GetPrim())
    link_xf.AddTranslateOp().Set(Gf.Vec3d(0.1, -0.2, 1.0))

    neutral = M._capture_robot_neutral_descendant_xforms(stage, "/World/G1")

    link_xf.ClearXformOpOrder()
    link_xf.AddTranslateOp().Set(Gf.Vec3d(3.0, 4.0, 5.0))
    restored = M._restore_robot_neutral_descendant_xforms(stage, neutral)

    assert restored == 1
    pos = UsdGeom.XformCache().GetLocalToWorldTransform(link.GetPrim()).ExtractTranslation()
    assert (float(pos[0]), float(pos[1]), float(pos[2])) == pytest.approx((10.1, -0.2, 1.0))


def test_serve_cli_flags_parsed() -> None:
    args = M.build_arg_parser().parse_args([
        "--request", "/x.json", "--out-dir", "/y", "--kitchen-usd", "/k.usd", "--g1-usd", "/g.usd",
        "--serve", "--serve-dir", "/jobs", "--serve-idle-timeout", "120", "--serve-max-jobs", "3",
    ])
    assert args.serve is True
    assert args.serve_dir == "/jobs"
    assert args.serve_idle_timeout == 120.0
    assert args.serve_max_jobs == 3


def test_parse_scenarios_keeps_task_only_job_for_warm_serve() -> None:
    # The warm serve loop feeds each job through parse_scenarios; a task-only job must survive so the
    # stance/target resolve dynamically (no hardcoded start/target). Guards the serve render contract.
    out = M.parse_scenarios({"scenarios": [
        {"scenario_id": "open_fridge", "description": "open the refrigerator"}
    ]})
    assert len(out) == 1
    assert out[0]["scenario_id"] == "open_fridge"


def test_serve_wiring_shares_render_fn_with_single_shot() -> None:
    # Single-shot and warm-serve must drive the SAME per-scenario render fn, so warm renders are
    # byte-for-byte the single-shot render — just without re-booting Isaac.
    src = _RUNNER.read_text()
    assert "def _render_scenario(sc):" in src
    assert "for sc in scenarios:\n                _render_scenario(sc)" in src
    i_def = src.index("def _render_scenario(sc):")
    assert src.index("serve_render_loop(", i_def) > i_def
    assert "render_one=_serve_render_one" in src


def test_dry_render_cli_runs_end_to_end_on_real_kitchen(tmp_path) -> None:
    # main(--dry-render ...) must run the whole local preview from a task-only request + kitchen USD,
    # write a preview PNG, and exit 0 WITHOUT needing --g1-usd or any GPU. Runs against the real
    # KitchenRoom asset when present, else a written synthetic .usda — so the CLI runnability proof
    # executes in every clean checkout instead of skipping.
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")
    from PIL import Image  # type: ignore

    usd, _is_real = _kitchen_usd_path_real_or_synthetic(tmp_path)
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"scenarios": [
        {"scenario_id": "open_fridge", "description": "open the refrigerator", "floor_z_hint": 0.0}
    ]}))
    out = tmp_path / "out"
    rc = M.main([
        "--request", str(req), "--kitchen-usd", str(usd), "--out-dir", str(out),
        "--dry-render", "--width", "640", "--height", "480", "--manipulation-reach-arm", "right",
    ])
    assert rc == 0
    pngs = list(out.rglob("dry_render_preview.png"))
    assert pngs
    index = out / "dry_render" / "dry_render_index.json"
    assert index.exists()

    # The full preview ran (stance accepted), not just the blocked early-return.
    summaries = json.loads(index.read_text())
    assert summaries and summaries[0]["stance"]["status"] == "accepted"

    # Provenance must survive the whole CLI path: the PNG carries the render-source text chunk and the
    # summary JSON carries the same marker, so a CPU proxy can never be filed as a real Isaac render.
    info = Image.open(pngs[0]).info
    assert info.get("X-Blueprint-Render-Source") == "dry_render_preview"
    assert "NOT a rendered frame" in info.get("X-Blueprint-Render-Note", "")
    summary = json.loads((pngs[0].parent / "dry_render_summary.json").read_text())
    assert summary["render_source"] == "dry_render_preview"
    assert summary["render_provenance"]["X-Blueprint-Render-Source"] == "dry_render_preview"
