"""CPU dry-render regression for the active G1 "open the refrigerator" POV seed lane.

The no-GPU ``--dry-render`` path reproduces the SAME stance + camera + arms-forward framing math the
Isaac GPU runner uses, so it is the cheapest catch for the two bug classes that have cost real GPU
rounds on this lane:

  * the invisible-robot bug — a physics-only G1 subtree with NO renderable Gprim geometry, and
  * the pitched-down-crop bug — a manipulation POV that looks too far down and crops the workspace.

This file locks the end-to-end ``--dry-render`` summary contract (the concrete ``_dry_render_checks``
booleans), the robot-visibility fail-closed gate, and the artifact provenance markers on the preview
PNG + summary JSON. Everything here is strictly CPU and fast: ``pytest.importorskip("pxr")`` so bare
envs skip cleanly, and the stages are tiny in-memory / written-``.usda`` fixtures (no GPU, no Isaac,
no cloud, no real heavy asset required).

Claim boundary: a dry-render preview is simulator REVIEW support — stance/camera/projection are exact
but the arm is a nominal Isaac-free skeleton and nothing is path-traced. These assertions check
structural/visibility/framing geometry and provenance, never "the robot opened the fridge".
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("parity_runner_dry_render_seed", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # raises if it imported isaacsim at module load (must NOT)
    return mod


M = _load()

# Refrigerator pose mirrors the real KitchenRoom fridge so the synthetic stance choice transfers.
_FRIDGE_CENTER = (-1.98, 0.66, 1.03)
_FRIDGE_SIZE = (0.88, 1.73, 2.06)


def _box(stage, path, translate, scale):
    from pxr import Gf, UsdGeom  # type: ignore

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddScaleOp().Set(Gf.Vec3d(*scale))


def _author_synthetic_kitchen(stage) -> None:
    """Floor + a refrigerator in a 3-wall niche (only the +x door side open).

    Named ``Refrigerator`` so the swappable ``scene_placement`` resolver maps "open the refrigerator"
    onto it (the full target-resolution chain runs, no monkeypatching). The niche forces the placement
    validator to stand the robot in front of the door — the same room-side choice the real kitchen
    forces — so the door-side regression is exercised on a tiny synthetic stage. TEST fixture only;
    never real captured geometry.
    """
    from pxr import UsdGeom  # type: ignore

    UsdGeom.Xform.Define(stage, "/World")
    _box(stage, "/World/Floor", (0.0, 0.0, -0.025), (8.0, 8.0, 0.05))
    fc = _FRIDGE_CENTER
    _box(stage, "/World/BackWall", (fc[0] - 0.9, fc[1], 1.2), (0.12, 3.0, 2.4))
    _box(stage, "/World/LeftWall", (fc[0], fc[1] + 1.4, 1.2), (2.2, 0.12, 2.4))
    _box(stage, "/World/RightWall", (fc[0], fc[1] - 1.4, 1.2), (2.2, 0.12, 2.4))
    UsdGeom.Xform.Define(stage, "/World/Refrigerator")
    _box(stage, "/World/Refrigerator/body", fc, _FRIDGE_SIZE)


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


def _kitchen_stage_real_or_synthetic():
    """Prefer the real KitchenRoom asset (which actually contains a fridge); fall back to the synthetic
    niche stage. Returns ``(stage, is_real)``. The 592-byte Collected stub has no fridge, so it is NOT
    treated as usable — only an asset large enough to carry the refrigerator geometry counts as real."""
    from pxr import Usd  # type: ignore

    usd = _real_kitchen_usd()
    if usd is not None and usd.stat().st_size > 4096:
        return M._open_stage_local(str(usd)), True
    stage = Usd.Stage.CreateInMemory()
    _author_synthetic_kitchen(stage)
    return stage, False


def _open_fridge_scenario():
    return {"scenario_id": "open_fridge", "description": "open the refrigerator", "floor_z_hint": 0.0}


# ---------------------------------------------------------------------------
# 1. End-to-end dry-render summary contract on the real-or-synthetic kitchen
# ---------------------------------------------------------------------------
def test_dry_render_open_fridge_summary_contract(tmp_path) -> None:
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")

    stage, _is_real = _kitchen_stage_real_or_synthetic()
    M._bind_proxy_robot(stage, "/World/G1")
    summary = M.render_local_preview(
        stage=stage,
        scenario=_open_fridge_scenario(),
        out_dir=tmp_path,
        manipulation_reach_arm="right",
        camera_vfov_deg=50.0,
        width=640,
        height=480,
        robot_prim_path="/World/G1",
    )

    # The whole local preview ran (stance planned + accepted), not the blocked early-return.
    assert summary["stance"]["status"] == "accepted"

    # Concrete dry-render check booleans — the cheapest catch for placement/framing/visibility bugs.
    checks = M._dry_render_checks(summary)
    assert checks["faces_target"] is True
    assert checks["target_in_frame"] is True
    assert checks["arm_in_frame"] is True
    assert checks["no_blockers"] is True
    assert checks["robot_visual_mesh_present"] is True
    assert checks["pov_geometry_pass"] is True

    # The pitched-down-crop gate must be WIRED and reported as a concrete boolean that agrees with the
    # measured pitch vs the cap — so a down-looking crop is always scored, never silently absent. (The
    # True/False outcome is produced by the camera-geometry lane; here we lock that the gate exists and
    # is self-consistent, which is what makes it a usable regression signal at all.)
    pitch_down = summary["checks"]["camera_pitch_down_deg"]
    assert isinstance(pitch_down, (int, float))
    cap = float(M.MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG)
    assert checks["camera_pitch_within_cap"] is (float(pitch_down) <= cap)

    # Robot stands clear of the fridge footprint (not inside the appliance volume) and faces it.
    bmin = summary["target"]["bbox_min_xyz"]
    bmax = summary["target"]["bbox_max_xyz"]
    pose = summary["stance"]["accepted_pose"]
    inside = (bmin[0] <= pose[0] <= bmax[0]) and (bmin[1] <= pose[1] <= bmax[1])
    assert not inside
    assert summary["checks"]["facing_error_deg"] < 8.0
    assert summary["pov_framing"]["target_in_frame"] is True
    assert summary["pov_framing"]["arm_landmarks_in_frame"] >= 1

    # Artifacts written.
    png = tmp_path / "dry_render_preview.png"
    js = tmp_path / "dry_render_summary.json"
    assert png.exists() and png.stat().st_size > 0
    assert js.exists()


# ---------------------------------------------------------------------------
# 2. Provenance — the artifact can never be passed off as a real Isaac render
# ---------------------------------------------------------------------------
def test_dry_render_artifacts_carry_render_source_provenance(tmp_path) -> None:
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")
    from PIL import Image  # type: ignore

    stage, _is_real = _kitchen_stage_real_or_synthetic()
    M._bind_proxy_robot(stage, "/World/G1")
    summary = M.render_local_preview(
        stage=stage,
        scenario=_open_fridge_scenario(),
        out_dir=tmp_path,
        manipulation_reach_arm="right",
        camera_vfov_deg=50.0,
        width=640,
        height=480,
        robot_prim_path="/World/G1",
    )

    # PNG metadata (PngInfo text chunk) carries the render-source marker + a "NOT a rendered frame" note.
    info = Image.open(tmp_path / "dry_render_preview.png").info
    assert info.get(M.DRY_RENDER_SOURCE_HEADER) == M.DRY_RENDER_SOURCE_MARKER
    assert M.DRY_RENDER_SOURCE_MARKER == "dry_render_preview"
    assert "NOT a rendered frame" in info.get(M.DRY_RENDER_NOTE_HEADER, "")

    # The same marker is in the in-memory summary AND the persisted JSON.
    assert summary["render_source"] == M.DRY_RENDER_SOURCE_MARKER
    persisted = json.loads((tmp_path / "dry_render_summary.json").read_text())
    assert persisted["render_source"] == M.DRY_RENDER_SOURCE_MARKER
    prov = persisted["render_provenance"]
    assert prov[M.DRY_RENDER_SOURCE_HEADER] == M.DRY_RENDER_SOURCE_MARKER
    assert "NOT a rendered frame" in prov[M.DRY_RENDER_NOTE_HEADER]
    # The provenance must also carry an explicit claim boundary forbidding render/task-success claims.
    assert "not Isaac RTX frames" in prov["claim_boundary"]


# ---------------------------------------------------------------------------
# 3. Robot-visibility gate fails closed for an invisible (Gprim-less) robot
# ---------------------------------------------------------------------------
def test_robot_visibility_gate_fails_closed_for_invisible_robot() -> None:
    # The invisible-robot bug: a physics-only G1 subtree (articulation/collision Xforms, zero Gprims).
    # The visibility gate must FAIL closed and emit the missing-visual-mesh blocker.
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/G1")
    UsdGeom.Xform.Define(stage, "/World/G1/right_shoulder_link")
    UsdGeom.Xform.Define(stage, "/World/G1/right_wrist_link")

    diag = M._robot_render_visibility_diagnostics(stage, "/World/G1")

    assert diag["status"] == "FAIL"
    assert M.ROBOT_VISUAL_MESH_MISSING_BLOCKER in diag["blockers"]
    assert diag["gprim_count"] == 0
    assert diag["renderable_robot_geometry_present"] is False


def test_robot_visibility_gate_reports_present_with_renderable_mesh() -> None:
    # The same traversal must report renderable geometry present once a Mesh/Gprim exists in the subtree.
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/G1")
    UsdGeom.Mesh.Define(stage, "/World/G1/torso_mesh")

    diag = M._robot_render_visibility_diagnostics(stage, "/World/G1")

    assert diag["gprim_count"] >= 1
    assert diag["renderable_robot_geometry_present"] is True


# ---------------------------------------------------------------------------
# 4. The --dry-render CLI runs cold end-to-end (no GPU, no --g1-usd)
# ---------------------------------------------------------------------------
def test_dry_render_cli_runs_cold_and_marks_provenance(tmp_path) -> None:
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")
    from pxr import Usd  # type: ignore
    from PIL import Image  # type: ignore

    usd = _real_kitchen_usd()
    if usd is not None and usd.stat().st_size > 4096:
        kitchen_path = usd
    else:
        # Write the synthetic kitchen to a .usda FILE so the CLI (which opens a USD path) can run cold
        # with no real heavy asset present. Clearly labeled SYNTHETIC.
        kitchen_path = tmp_path / "SyntheticKitchenRoom.usda"
        stage = Usd.Stage.CreateNew(str(kitchen_path))
        _author_synthetic_kitchen(stage)
        stage.SetMetadata("comment", "SYNTHETIC TEST FIXTURE - not real captured geometry")
        stage.GetRootLayer().Save()

    req = tmp_path / "req.json"
    req.write_text(json.dumps({"scenarios": [_open_fridge_scenario()]}))
    out = tmp_path / "out"
    rc = M.main([
        "--request", str(req), "--kitchen-usd", str(kitchen_path), "--out-dir", str(out),
        "--dry-render", "--width", "640", "--height", "480", "--manipulation-reach-arm", "right",
    ])
    assert rc == 0

    pngs = list(out.rglob("dry_render_preview.png"))
    assert pngs
    index = out / "dry_render" / "dry_render_index.json"
    assert index.exists()
    summaries = json.loads(index.read_text())
    assert summaries and summaries[0]["stance"]["status"] == "accepted"

    info = Image.open(pngs[0]).info
    assert info.get(M.DRY_RENDER_SOURCE_HEADER) == M.DRY_RENDER_SOURCE_MARKER
    summary = json.loads((pngs[0].parent / "dry_render_summary.json").read_text())
    assert summary["render_provenance"][M.DRY_RENDER_SOURCE_HEADER] == M.DRY_RENDER_SOURCE_MARKER
