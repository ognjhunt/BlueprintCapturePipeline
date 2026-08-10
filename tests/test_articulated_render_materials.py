from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from blueprint_pipeline.articulated_render_materials import (
    RENDER_MATERIAL_SCAFFOLD_SCHEMA_VERSION,
    ArticulatedRenderMaterialError,
    ensure_render_material_scaffold,
)


def _box(stage: Usd.Stage, path: str, z0: float, z1: float) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(x, y, z)
            for x in (-0.3, 0.3)
            for y in (-0.3, 0.3)
            for z in (z0, z1)
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
    )
    return mesh


def _twin(path: Path) -> Path:
    """An articulated candidate with physics bindings but no render material."""

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    physics_material = UsdShade.Material.Define(stage, "/Asset/Looks/PhysMat_a")
    UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    for link, z0, z1 in (("cabinet", 0.0, 1.6), ("upper_door", 0.94, 1.6)):
        xform = UsdGeom.Xform.Define(stage, f"/Asset/{link}")
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        for component in ("component_000", "component_001"):
            mesh = _box(stage, f"/Asset/{link}/{component}", z0, z1)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
                physics_material, materialPurpose="physics"
            )
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/upper_door_hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/cabinet")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/upper_door")])
    joint.CreateAxisAttr().Set("Z")
    joint.CreateLowerLimitAttr().Set(0.0)
    joint.CreateUpperLimitAttr().Set(90.0)
    stage.GetRootLayer().Save()
    return path


def _surfaces() -> list[dict]:
    return [
        {
            "material_id": "door_shell",
            "prim_paths": ["/Asset/upper_door/component_000"],
            "base_color": [0.659, 0.592, 0.544],
            "roughness": 0.45,
            "metallic": 0.0,
            "observed_albedo": True,
        },
        {
            "material_id": "cabinet_shell",
            "prim_paths": ["/Asset/cabinet/component_000"],
            "base_color": [0.62, 0.58, 0.55],
        },
    ]


def _scaffold(tmp_path: Path, **overrides):
    arguments = {
        "source_usd_path": _twin(tmp_path / "twin.usda"),
        "destination": tmp_path / "twin_render_ready.usda",
        "surfaces": _surfaces(),
    }
    arguments.update(overrides)
    return ensure_render_material_scaffold(**arguments)


def test_scaffold_creates_a_bound_render_material_the_texture_agent_can_find(
    tmp_path: Path,
) -> None:
    """The Texture Agent rejects a plan with no effectively bound material."""

    receipt = _scaffold(tmp_path)

    stage = Usd.Stage.Open(receipt["render_ready_usd_path"])
    target = stage.GetPrimAtPath("/Asset/upper_door/component_000")
    bound, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()
    assert bound and bound.GetPrim().IsValid()
    assert str(bound.GetPrim().GetPath()) == "/Asset/Looks/Render/door_shell"
    surface = bound.ComputeSurfaceSource()[0]
    assert surface and surface.GetPrim().IsValid()
    assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
    assert receipt["schema_version"] == RENDER_MATERIAL_SCAFFOLD_SCHEMA_VERSION
    assert receipt["status"] == "render_material_scaffold_authored"


def test_scaffold_seeds_the_observed_albedo_and_records_its_provenance(
    tmp_path: Path,
) -> None:
    receipt = _scaffold(tmp_path)

    stage = Usd.Stage.Open(receipt["render_ready_usd_path"])
    shader = UsdShade.Shader.Get(stage, "/Asset/Looks/Render/door_shell/Shader")
    colour = shader.GetInput("diffuseColor").Get()
    assert [round(float(v), 3) for v in colour] == [0.659, 0.592, 0.544]
    door = next(
        row for row in receipt["surfaces"] if row["material_id"] == "door_shell"
    )
    assert door["observed_albedo"] is True
    assert door["base_color"] == [0.659, 0.592, 0.544]
    assert receipt["claim_boundary"]["flat_colour_is_not_a_texture_pass"] is True
    assert receipt["claim_boundary"]["appearance_is_candidate_not_observed_truth"] is True


def test_scaffold_leaves_physics_and_articulation_untouched(tmp_path: Path) -> None:
    """Appearance authoring must never disturb what physics already owns."""

    receipt = _scaffold(tmp_path)

    stage = Usd.Stage.Open(receipt["render_ready_usd_path"])
    target = stage.GetPrimAtPath("/Asset/upper_door/component_000")
    physics, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial(
        materialPurpose="physics"
    )
    assert str(physics.GetPrim().GetPath()) == "/Asset/Looks/PhysMat_a"
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(joints) == 1 and len(roots) == 1 and len(bodies) == 2
    assert receipt["preserved"]["assembly_joint_count"] == 1
    assert receipt["preserved"]["articulation_root_count"] == 1
    assert receipt["preserved"]["physics_bindings_unchanged"] is True


def test_scaffold_never_writes_over_the_source(tmp_path: Path) -> None:
    source = _twin(tmp_path / "twin.usda")
    before = source.read_bytes()

    receipt = _scaffold(tmp_path, source_usd_path=source)

    assert source.read_bytes() == before
    assert Path(receipt["render_ready_usd_path"]) != source


def test_scaffold_rejects_a_target_prim_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedRenderMaterialError) as excinfo:
        _scaffold(
            tmp_path,
            surfaces=[
                {"material_id": "ghost", "prim_paths": ["/Asset/upper_door/missing"]}
            ],
        )

    assert any("target_prim_missing" in error for error in excinfo.value.errors)


def test_scaffold_rejects_a_target_that_is_not_geometry(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedRenderMaterialError) as excinfo:
        _scaffold(
            tmp_path,
            surfaces=[{"material_id": "bad", "prim_paths": ["/Asset/upper_door"]}],
        )

    assert any("target_prim_not_a_mesh" in error for error in excinfo.value.errors)


def test_scaffold_rejects_duplicate_material_ids(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedRenderMaterialError) as excinfo:
        _scaffold(
            tmp_path,
            surfaces=[
                {"material_id": "dup", "prim_paths": ["/Asset/cabinet/component_000"]},
                {"material_id": "dup", "prim_paths": ["/Asset/cabinet/component_001"]},
            ],
        )

    assert any("material_id_duplicated" in error for error in excinfo.value.errors)


def test_scaffold_is_deterministic(tmp_path: Path) -> None:
    first = _scaffold(tmp_path, destination=tmp_path / "a.usda")
    second = _scaffold(tmp_path, destination=tmp_path / "b.usda")

    assert first["render_ready_usd_sha256"] == second["render_ready_usd_sha256"]
    assert first["surfaces"] == second["surfaces"]


def test_rigid_fixture_scaffolds_with_a_single_surface(tmp_path: Path) -> None:
    """The original rigid shape must work with no articulation present."""

    path = tmp_path / "can.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    _box(stage, "/Asset/body", 0.0, 0.17)
    stage.GetRootLayer().Save()

    receipt = ensure_render_material_scaffold(
        source_usd_path=path,
        destination=tmp_path / "can_render_ready.usda",
        surfaces=[{"material_id": "can_body", "prim_paths": ["/Asset/body"]}],
    )

    assert receipt["preserved"]["assembly_joint_count"] == 0
    assert receipt["surfaces"][0]["bound_prim_count"] == 1


def test_receipt_file_round_trips(tmp_path: Path) -> None:
    receipt = _scaffold(tmp_path)

    stored = json.loads(
        Path(receipt["receipt_path"]).read_text(encoding="utf-8")
    )
    assert stored == receipt


def test_scaffold_follows_the_stage_default_prim_not_a_fixed_name(
    tmp_path: Path,
) -> None:
    """An asset whose root is not /Asset must still get bound materials."""

    path = tmp_path / "oven.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Oven")
    stage.SetDefaultPrim(root.GetPrim())
    _box(stage, "/Oven/door", 0.0, 0.9)
    stage.GetRootLayer().Save()

    receipt = ensure_render_material_scaffold(
        source_usd_path=path,
        destination=tmp_path / "oven_render_ready.usda",
        surfaces=[{"material_id": "door_shell", "prim_paths": ["/Oven/door"]}],
    )

    assert receipt["surfaces"][0]["material_path"] == "/Oven/Looks/Render/door_shell"
    opened = Usd.Stage.Open(receipt["render_ready_usd_path"])
    bound, _ = UsdShade.MaterialBindingAPI(
        opened.GetPrimAtPath("/Oven/door")
    ).ComputeBoundMaterial()
    assert bound and bound.GetPrim().IsValid()
