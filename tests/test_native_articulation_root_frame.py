from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.native_articulation_root_frame import (
    NativeArticulationRootFrameError,
    apply_root_api_binding,
    convert_root_pose,
    derive_root_frame_binding,
    rebind_root_frame_after_overlay,
    root_bound_usd_spawner,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _asset(tmp_path: Path, *, explicit=False, fixed=False, offset=False) -> Path:
    path = tmp_path / "mechanism.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Fixture")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    for name, position in (("base", (0.2, -0.3, 0.4) if offset else (0, 0, 0)),
                           ("panel", (0.23, 0, 0.1)), ("handle", (0.4, 0, 0.7))):
        body = UsdGeom.Xform.Define(stage, "/Fixture/" + name)
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        if name == "base" and offset:
            body.AddOrientOp().Set(Gf.Quatf(math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    # The middle panel is central in the undirected graph; base is its authored
    # parent. Auto-rooting on the panel silently changes the placement frame.
    for name, left, right in (("hinge", "base", "panel"), ("handle_mount", "panel", "handle")):
        joint = UsdPhysics.FixedJoint.Define(stage, "/Fixture/" + name)
        joint.CreateBody0Rel().SetTargets(["/Fixture/" + left])
        joint.CreateBody1Rel().SetTargets(["/Fixture/" + right])
    if fixed:
        world = UsdPhysics.FixedJoint.Define(stage, "/Fixture/world_anchor")
        world.CreateBody1Rel().SetTargets(["/Fixture/base"])
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(
        "/Fixture/panel" if explicit else "/Fixture"))
    stage.GetRootLayer().Save()
    return path


def _pose():
    return {"position_world_m": [3.0, -2.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]}


@pytest.mark.parametrize(("explicit", "fixed", "selected", "mode"), [
    (False, False, "base", "directed_body_root"),
    (True, False, "panel", "authored_body_root"),
    (False, True, "base", "authored_world_joint"),
])
def test_root_selection_preserves_authored_semantics_and_source_bytes(tmp_path, explicit, fixed, selected, mode):
    path = _asset(tmp_path, explicit=explicit, fixed=fixed)
    before = path.read_bytes()
    binding = derive_root_frame_binding(path)
    assert binding["native_root_body_prim_path"] == "/Fixture/" + selected
    assert binding["selection"] == mode
    assert binding["source_sha256"] == "sha256:" + hashlib.sha256(before).hexdigest()
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Imported", "Xform")
    prim.GetReferences().AddReference(str(path))
    apply_root_api_binding(stage, str(prim.GetPath()), binding)
    apis = [str(p.GetPath()) for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert apis == ["/World/Imported/" + ("world_anchor" if fixed else selected)]
    assert path.read_bytes() == before
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]) == (3 if fixed else 2)


def test_nonidentity_base_frame_round_trips_and_places_each_link(tmp_path):
    path = _asset(tmp_path, offset=True)
    binding = derive_root_frame_binding(path)
    native = convert_root_pose(_pose(), binding, to_native=True)
    assert native["position_world_m"] == pytest.approx([3.3, -1.8, 0.9])
    logical = convert_root_pose(native, binding, to_native=False)
    assert logical["position_world_m"] == pytest.approx(_pose()["position_world_m"])
    assert logical["orientation_xyzw"] == pytest.approx(_pose()["orientation_xyzw"])


def test_disconnected_and_ambiguous_roots_refuse_before_native_start(tmp_path):
    path = _asset(tmp_path)
    stage = Usd.Stage.Open(str(path))
    stage.RemovePrim("/Fixture/hinge")
    stage.GetRootLayer().Save()
    with pytest.raises(NativeArticulationRootFrameError, match="topology_ambiguous"):
        derive_root_frame_binding(path)


def test_binding_digest_and_source_identity_are_enforced(tmp_path):
    binding = derive_root_frame_binding(_asset(tmp_path))
    binding["native_root_body_prim_path"] = "/Fixture/panel"
    with pytest.raises(NativeArticulationRootFrameError, match="binding_invalid"):
        convert_root_pose(_pose(), binding, to_native=True)


def test_material_overlay_keeps_root_frame_but_geometry_frame_change_refuses(tmp_path):
    path = _asset(tmp_path)
    binding = derive_root_frame_binding(path)
    stage = Usd.Stage.Open(str(path))
    stage.GetDefaultPrim().CreateAttribute("candidate:materialNote", Sdf.ValueTypeNames.String).Set("friction variant")
    stage.GetRootLayer().Save()
    rebound = rebind_root_frame_after_overlay(binding, path)
    assert rebound["source_sha256"] != binding["source_sha256"]
    UsdGeom.Xformable(stage.GetPrimAtPath("/Fixture/base")).GetOrderedXformOps()[0].Set(Gf.Vec3d(1, 0, 0))
    stage.GetRootLayer().Save()
    with pytest.raises(NativeArticulationRootFrameError, match="overlay_frame_changed"):
        rebind_root_frame_after_overlay(binding, path)


def test_spawn_maps_native_initial_pose_back_to_namespace_before_root_override(tmp_path, monkeypatch):
    path = _asset(tmp_path, offset=True)
    binding = derive_root_frame_binding(path)
    calls = []
    stage = Usd.Stage.CreateInMemory()

    def spawn_from_usd(prim_path, cfg, *, translation, orientation, **kwargs):
        prim = UsdGeom.Xform.Define(stage, prim_path)
        prim.GetPrim().GetReferences().AddReference(str(cfg.usd_path))
        prim.AddTranslateOp().Set(Gf.Vec3d(*translation))
        prim.AddOrientOp().Set(Gf.Quatf(orientation[3], *orientation[:3]))
        calls.append((translation, orientation))
        return prim.GetPrim()

    for name in ("isaaclab.sim.spawners.from_files.from_files", "isaaclab.sim.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["isaaclab.sim.spawners.from_files.from_files"].spawn_from_usd = spawn_from_usd
    # The real @clone wraps this whole callback: root overrides occur before
    # cloning, rather than patching only the first already-cloned environment.
    sys.modules["isaaclab.sim.utils"].clone = lambda function: function
    spawn = root_bound_usd_spawner(binding)
    native = convert_root_pose(_pose(), binding, to_native=True)
    prim = spawn("/World/Imported", SimpleNamespace(usd_path=path),
                 translation=native["position_world_m"], orientation=native["orientation_xyzw"])
    assert calls[0][0] == pytest.approx(_pose()["position_world_m"])
    assert calls[0][1] == pytest.approx(_pose()["orientation_xyzw"])
    base = stage.GetPrimAtPath(str(prim.GetPath()) + "/base")
    assert base.HasAPI(UsdPhysics.ArticulationRootAPI)
    actual = UsdGeom.XformCache().GetLocalToWorldTransform(base).ExtractTranslation()
    assert list(actual) == pytest.approx(native["position_world_m"], abs=1e-6)
    binding["source_sha256"] = "sha256:" + "0" * 64
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    with pytest.raises(NativeArticulationRootFrameError, match="spawn_binding_mismatch"):
        root_bound_usd_spawner(binding)("/World/Other", SimpleNamespace(usd_path=path))


def test_runtime_initial_state_uses_bound_body_pose_and_keeps_reset_joint_contract(tmp_path, monkeypatch):
    from blueprint_pipeline.native_task_arena_runtime import build_native_task_arena_environment
    from tests.test_native_task_arena_runtime import (
        _ArenaBuilder, _install_fake_native_runtime, _sealed_scene_plan,
    )
    import blueprint_pipeline.native_task_arena_runtime as runtime

    _install_fake_native_runtime(monkeypatch)
    plan = _sealed_scene_plan()
    task = next(row for row in plan["objects"] if row["semantic_role"] == "task_object")
    binding = derive_root_frame_binding(_asset(tmp_path, offset=True))
    task["articulation_adaptation"] = {"native_root_frame_binding": binding}
    def sentinel(*args, **kwargs):
        return None
    monkeypatch.setattr(runtime, "root_bound_usd_spawner", lambda received: sentinel)
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    build_native_task_arena_environment(plan)
    obj = next(asset for asset in _ArenaBuilder.last.arena_env.scene.assets
               if asset.name == "task_object")
    expected = convert_root_pose(task["pose_world"], binding, to_native=True)
    assert obj.object_cfg.init_state.pos == pytest.approx(expected["position_world_m"])
    assert obj.object_cfg.init_state.rot == pytest.approx(expected["orientation_xyzw"])
    assert obj.spawn_cfg_addon["func"] is sentinel
    assert obj.object_cfg.init_state.joint_pos == plan["reset"]["task_joint_positions_rad"]


def test_reset_and_task_scoring_read_the_asset_frame_not_the_native_body_frame(tmp_path):
    from blueprint_pipeline.native_task_arena_readback import (
        NativeArticulatedTaskArenaReadback, read_native_task_arena_object_reset_state,
    )
    from tests.test_native_task_arena_readback import _built

    built = _built()
    planned = built.plan["objects"][0]
    binding = derive_root_frame_binding(_asset(tmp_path, offset=True))
    planned["articulation_adaptation"] = {"native_root_frame_binding": binding}
    native = convert_root_pose(planned["pose_world"], binding, to_native=True)
    data = built.env.unwrapped.scene["task_object"].data
    data.root_pose_w = [[*native["position_world_m"], *native["orientation_xyzw"]]]
    data.joint_pos = [[0.0, 0.0]]
    readback = read_native_task_arena_object_reset_state(built)
    assert readback["passed"] is True
    assert readback["objects"][0]["observed_root_pose_world"]["position_world_m"] == pytest.approx(
        planned["pose_world"]["position_world_m"])
    sample = NativeArticulatedTaskArenaReadback(built).read_task_sample()
    assert sample["native_readback"]["root_translation_delta_m"] == pytest.approx(0.0)
    assert sample["native_readback"]["root_orientation_delta_rad"] == pytest.approx(0.0)
    assert sample["containment_violation"] is False


def test_old_cached_packet_requires_root_binding_recompile_before_paid_use(tmp_path):
    from blueprint_pipeline.native_task_arena_runtime import (
        NativeTaskArenaRuntimeError, validate_native_task_arena_runtime_plan,
    )
    from tests.test_native_task_arena_runtime import _staged_packet

    root, plan = _staged_packet(tmp_path)
    task = next(row for row in plan["objects"] if row["semantic_role"] == "task_object")
    del task["articulation_adaptation"]["native_root_frame_binding"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    with pytest.raises(NativeTaskArenaRuntimeError, match="root_binding_missing"):
        validate_native_task_arena_runtime_plan(plan, bundle_root=root)
