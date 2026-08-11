from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

import blueprint_pipeline.native_deformable_asset_stage_adapter as adapter_module
from blueprint_pipeline.native_deformable_asset_stage_adapter import (
    NativeDeformableAssetStageAdapterError,
    OpenUsdNativeDeformableStageAdapter,
)


def _source_asset(root: Path) -> tuple[Path, Path, list[float]]:
    texture = root / "textures" / "base.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"bounded-test-texture")
    path = root / "source.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    default = UsdGeom.Xform.Define(stage, "/Source").GetPrim()
    stage.SetDefaultPrim(default)
    mesh = UsdGeom.Mesh.Define(stage, "/Source/Visuals/Surface")
    points = Vt.Vec3fArray(
        [
            Gf.Vec3f(-1.0, -1.0, -1.0),
            Gf.Vec3f(1.0, -1.0, -1.0),
            Gf.Vec3f(0.0, 1.0, -1.0),
            Gf.Vec3f(0.0, 0.0, 1.0),
        ]
    )
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3, 3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    transform = Gf.Matrix4d(1.0)
    transform.SetScale(Gf.Vec3d(0.2, 0.3, 0.4))
    transform.SetTranslateOnly(Gf.Vec3d(1.0, 2.0, 3.0))
    UsdGeom.Xformable(mesh).AddTransformOp().Set(transform)
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0)] * 12))

    material = UsdShade.Material.Define(stage, "/Source/Looks/Material")
    preview = UsdShade.Shader.Define(stage, "/Source/Looks/Material/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
    texture_shader = UsdShade.Shader.Define(stage, "/Source/Looks/Material/Texture")
    texture_shader.CreateIdAttr("UsdUVTexture")
    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/base.png")
    )
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(mesh.GetPrim())
    world_points = [world.Transform(Gf.Vec3d(point)) for point in points]
    minimum = [min(float(point[axis]) for point in world_points) for axis in range(3)]
    maximum = [max(float(point[axis]) for point in world_points) for axis in range(3)]
    center = [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)]
    return path, texture, center


def _native_configuration() -> tuple[dict[str, object], dict[str, float]]:
    body_and_cooking: dict[str, object] = {
        "deformable_body_enabled": True,
        "kinematic_enabled": False,
        "self_collision": True,
        "solver_position_iteration_count": 28,
        "linear_damping": 0.18,
        "contact_offset": 0.003,
        "rest_offset": 0.001,
        "collision_simplification": True,
        "collision_simplification_remeshing": True,
        "collision_simplification_remeshing_resolution": 0,
        "simulation_hexahedral_resolution": 22,
    }
    material = {
        "density": 220.0,
        "static_friction": 2.2,
        "dynamic_friction": 2.2,
        "youngs_modulus": 180000.0,
        "poissons_ratio": 0.42,
        "elasticity_damping": 0.18,
    }
    return body_and_cooking, material


def _author_native_readback(
    stage: Usd.Stage,
    *,
    body_and_cooking: dict[str, object],
    material_properties: dict[str, float],
) -> None:
    body = stage.GetPrimAtPath("/Deformable")
    surface = stage.GetPrimAtPath("/Deformable/Visuals/Surface")
    for schema in ("OmniPhysicsDeformableBodyAPI", "PhysxBaseDeformableBodyAPI", "PhysxCollisionAPI"):
        body.AddAppliedSchema(schema)
    for field, value in body_and_cooking.items():
        if field in {"deformable_body_enabled", "kinematic_enabled", "mass"}:
            namespace = "omniphysics"
        elif field in {"contact_offset", "rest_offset"}:
            namespace = "physxCollision"
        elif field.startswith("collision_") or field == "simulation_hexahedral_resolution":
            continue
        else:
            namespace = "physxDeformableBody"
        name = field.split("_")[0] + "".join(
            part[:1].upper() + part[1:] for part in field.split("_")[1:]
        )
        if isinstance(value, bool):
            value_type = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            value_type = Sdf.ValueTypeNames.Int
        else:
            value_type = Sdf.ValueTypeNames.Double
        body.CreateAttribute(f"{namespace}:{name}", value_type).Set(value)
    tet_points = surface.GetAttribute("points").Get()
    surface.CreateAttribute(
        "physxDeformable:simulationPoints", Sdf.ValueTypeNames.Point3fArray
    ).Set(tet_points)
    surface.CreateAttribute("physxDeformable:simulationIndices", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray([0, 1, 2, 3])
    )
    surface.CreateAttribute("physxDeformable:collisionPoints", Sdf.ValueTypeNames.Point3fArray).Set(
        tet_points
    )
    surface.CreateAttribute("physxDeformable:collisionIndices", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray([0, 1, 2, 3])
    )

    physics = UsdShade.Material.Define(stage, "/Deformable/PhysicsMaterial")
    physics.GetPrim().AddAppliedSchema("PhysxDeformableBodyMaterialAPI")
    for field, value in material_properties.items():
        name = field.split("_")[0] + "".join(
            part[:1].upper() + part[1:] for part in field.split("_")[1:]
        )
        physics.GetPrim().CreateAttribute(
            f"physxDeformableBodyMaterial:{name}", Sdf.ValueTypeNames.Double
        ).Set(value)
    UsdShade.MaterialBindingAPI.Apply(surface).Bind(
        physics,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _install_hermetic_registered_physx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test-only registry seam; production always imports real ``PhysxSchema``."""

    def validate(prim, expected, *, error):
        del error
        authored = adapter_module._schema_names(prim)
        assert set(expected).issubset(authored)
        return sorted(expected.values())

    monkeypatch.setattr(adapter_module, "_registered_physx_schema_names", validate)


def _prepared_native_stage(
    root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch | None,
) -> tuple[OpenUsdNativeDeformableStageAdapter, Usd.Stage]:
    source, texture, center = _source_asset(root / "input")
    output = root / "result" / "deformable.usda"
    output.parent.mkdir(parents=True)
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    stage = adapter.create_clean_stage(
        output_path=output,
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    adapter.copy_surface_mesh_baking_points(
        stage=stage,
        source_usd_path=source,
        source_prim_path="/Source/Visuals/Surface",
        output_prim_path="/Deformable/Visuals/Surface",
        source_world_bounds_center_m=center,
        recenter_to_output_origin=True,
        bake_scale_xyz=[1.0, 1.0, 1.0],
        flatten_source_xform=True,
    )
    adapter.copy_bound_material_network(
        stage=stage,
        source_usd_path=source,
        material_prim_path_map={"/Source/Looks/Material": "/Deformable/Looks/Material_000"},
        output_looks_prim_path="/Deformable/Looks",
        output_visual_prim_path="/Deformable/Visuals/Surface",
        source_texture_paths={"base.png": texture},
        output_texture_asset_paths={"base.png": "textures/base.png"},
    )
    body_and_cooking, material_properties = _native_configuration()
    adapter.record_native_configuration(
        stage=stage,
        body_and_cooking_properties=body_and_cooking,
        material_properties=material_properties,
    )
    _author_native_readback(
        stage,
        body_and_cooking=body_and_cooking,
        material_properties=material_properties,
    )
    if monkeypatch is not None:
        _install_hermetic_registered_physx(monkeypatch)
    return adapter, stage


def _readback(adapter: OpenUsdNativeDeformableStageAdapter, stage: Usd.Stage) -> dict[str, object]:
    return dict(
        adapter.readback_prepared_stage(
            stage=stage,
            output_authoring_root_prim_path="/Deformable",
            output_deformable_schema_prim_path="/Deformable",
            output_visual_prim_path="/Deformable/Visuals/Surface",
        )
    )


def test_clean_stage_rebuild_bakes_metric_geometry_and_replays_native_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, texture, center = _source_asset(tmp_path / "input")
    output = tmp_path / "result" / "deformable.usda"
    output.parent.mkdir()
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    stage = adapter.create_clean_stage(
        output_path=output,
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    adapter.copy_surface_mesh_baking_points(
        stage=stage,
        source_usd_path=source,
        source_prim_path="/Source/Visuals/Surface",
        output_prim_path="/Deformable/Visuals/Surface",
        source_world_bounds_center_m=center,
        recenter_to_output_origin=True,
        bake_scale_xyz=[2.0, 3.0, 4.0],
        flatten_source_xform=True,
    )
    adapter.copy_bound_material_network(
        stage=stage,
        source_usd_path=source,
        material_prim_path_map={"/Source/Looks/Material": "/Deformable/Looks/Material_000"},
        output_looks_prim_path="/Deformable/Looks",
        output_visual_prim_path="/Deformable/Visuals/Surface",
        source_texture_paths={"base.png": texture},
        output_texture_asset_paths={"base.png": "textures/base.png"},
    )
    body_and_cooking, material_properties = _native_configuration()
    adapter.record_native_configuration(
        stage=stage,
        body_and_cooking_properties=body_and_cooking,
        material_properties=material_properties,
    )
    _author_native_readback(
        stage,
        body_and_cooking=body_and_cooking,
        material_properties=material_properties,
    )
    _install_hermetic_registered_physx(monkeypatch)
    adapter.save_stage(stage=stage)
    readback = adapter.readback_prepared_stage(
        stage=stage,
        output_authoring_root_prim_path="/Deformable",
        output_deformable_schema_prim_path="/Deformable",
        output_visual_prim_path="/Deformable/Visuals/Surface",
    )

    assert output.is_file()
    assert readback["stage_metadata"] == {
        "default_prim_path": "/Deformable",
        "meters_per_unit": 1.0,
        "up_axis": "Z",
    }
    visual = readback["visual_mesh"]
    assert visual["point_count"] == 4
    assert visual["triangle_count"] == 4
    assert visual["source_face_topology_sha256"] == visual["output_face_topology_sha256"]
    assert visual["aabb_center_m"] == [0.0, 0.0, 0.0]
    assert visual["dimensions_m"] == pytest.approx([0.8, 1.8, 3.2], abs=1.0e-6)
    assert visual["closed_volume_m3"] == pytest.approx(0.768, rel=1.0e-6)
    assert readback["simulation_topology"]["node_count"] == 4
    assert readback["simulation_topology"]["element_count"] == 1
    assert readback["collision_topology"]["node_count"] == 4
    assert readback["physics_material"]["properties"] == material_properties
    assert readback["physics_configuration"]["cooking_properties"] == {
        key: value
        for key, value in body_and_cooking.items()
        if key.startswith("collision_") or key == "simulation_hexahedral_resolution"
    }
    assert readback["material_binding"]["texture_asset_paths"] == ["textures/base.png"]
    assert math.isclose(readback["mass_properties"]["derived_mass_kg"], 168.96, rel_tol=1.0e-6)
    assert not stage.GetPrimAtPath("/Source").IsValid()
    copied_file = (
        stage.GetPrimAtPath("/Deformable/Looks/Material_000/Texture")
        .GetAttribute("inputs:file")
        .Get()
    )
    assert copied_file.path == "textures/base.png"


def test_material_asset_outside_allowlist_fails_closed(tmp_path: Path) -> None:
    source, texture, center = _source_asset(tmp_path / "input")
    stage = Usd.Stage.Open(str(source))
    stage.GetPrimAtPath("/Source/Looks/Material/Texture").GetAttribute("inputs:file").Set(
        Sdf.AssetPath("textures/not-declared.png")
    )
    stage.GetRootLayer().Save()
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    adapter.copy_surface_mesh_baking_points(
        stage=output_stage,
        source_usd_path=source,
        source_prim_path="/Source/Visuals/Surface",
        output_prim_path="/Deformable/Visuals/Surface",
        source_world_bounds_center_m=center,
        recenter_to_output_origin=True,
        bake_scale_xyz=[1.0, 1.0, 1.0],
        flatten_source_xform=True,
    )
    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_bound_material_network(
            stage=output_stage,
            source_usd_path=source,
            material_prim_path_map={"/Source/Looks/Material": "/Deformable/Looks/Material_000"},
            output_looks_prim_path="/Deformable/Looks",
            output_visual_prim_path="/Deformable/Visuals/Surface",
            source_texture_paths={"base.png": texture},
            output_texture_asset_paths={"base.png": "textures/base.png"},
        )
    assert "native_deformable_stage_material_asset_outside_allowlist" in exc.value.errors


def test_non_z_source_is_rejected(tmp_path: Path) -> None:
    source, _texture, center = _source_asset(tmp_path / "input")
    stage = Usd.Stage.Open(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.GetRootLayer().Save()
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_surface_mesh_baking_points(
            stage=output_stage,
            source_usd_path=source,
            source_prim_path="/Source/Visuals/Surface",
            output_prim_path="/Deformable/Visuals/Surface",
            source_world_bounds_center_m=center,
            recenter_to_output_origin=True,
            bake_scale_xyz=[1.0, 1.0, 1.0],
            flatten_source_xform=True,
        )
    assert "native_deformable_stage_source_axes_or_units_unsupported" in exc.value.errors


def test_live_visual_mesh_mutation_cannot_replay_cached_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, stage = _prepared_native_stage(tmp_path, monkeypatch=monkeypatch)
    mesh = stage.GetPrimAtPath("/Deformable/Visuals/Surface")
    points = list(mesh.GetAttribute("points").Get())
    points[0] = Gf.Vec3f(points[0][0] + 0.01, points[0][1], points[0][2])
    mesh.GetAttribute("points").Set(Vt.Vec3fArray(points))

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert "native_deformable_stage_visual_mesh_readback_mismatch" in exc.value.errors


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("repeated_simulation_indices", "native_deformable_stage_simulation_topology_invalid"),
        ("out_of_range_collision_indices", "native_deformable_stage_collision_topology_invalid"),
        ("nonfinite_simulation_points", "native_deformable_stage_simulation_points_invalid"),
    ],
)
def test_invalid_cooked_tetrahedra_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_error: str,
) -> None:
    adapter, stage = _prepared_native_stage(tmp_path, monkeypatch=monkeypatch)
    mesh = stage.GetPrimAtPath("/Deformable/Visuals/Surface")
    if case == "repeated_simulation_indices":
        mesh.GetAttribute("physxDeformable:simulationIndices").Set(Vt.IntArray([0, 0, 0, 0]))
    elif case == "out_of_range_collision_indices":
        mesh.GetAttribute("physxDeformable:collisionIndices").Set(Vt.IntArray([99, 99, 99, 99]))
    else:
        points = list(mesh.GetAttribute("physxDeformable:simulationPoints").Get())
        points[0] = Gf.Vec3f(float("nan"), 0.0, 0.0)
        mesh.GetAttribute("physxDeformable:simulationPoints").Set(Vt.Vec3fArray(points))

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert expected_error in exc.value.errors


@pytest.mark.parametrize(
    ("prim_path", "attribute_name", "expected_error"),
    [
        (
            "/Deformable",
            "physxDeformableBody:linearDamping",
            "native_deformable_stage_body_readback_mismatch:linear_damping",
        ),
        (
            "/Deformable/PhysicsMaterial",
            "physxDeformableBodyMaterial:density",
            "native_deformable_stage_material_readback_mismatch:density",
        ),
    ],
)
def test_nonfinite_native_configuration_readback_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prim_path: str,
    attribute_name: str,
    expected_error: str,
) -> None:
    adapter, stage = _prepared_native_stage(tmp_path, monkeypatch=monkeypatch)
    stage.GetPrimAtPath(prim_path).GetAttribute(attribute_name).Set(float("nan"))

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert expected_error in exc.value.errors


def test_raw_schema_tokens_do_not_prove_registered_physx_apis(tmp_path: Path) -> None:
    adapter, stage = _prepared_native_stage(tmp_path, monkeypatch=None)

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert {
        "native_deformable_stage_physx_schema_runtime_unavailable",
        "native_deformable_stage_native_schema_readback_invalid",
    }.intersection(exc.value.errors)


def test_source_composition_is_rejected_before_stage_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external, _texture, center = _source_asset(tmp_path / "outside")
    source = tmp_path / "package" / "source.usda"
    source.parent.mkdir()
    source_stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(source_stage, 1.0)
    UsdGeom.SetStageUpAxis(source_stage, UsdGeom.Tokens.z)
    holder = source_stage.DefinePrim("/Holder", "Xform")
    holder.GetReferences().AddReference(str(external), "/Source")
    source_stage.SetDefaultPrim(holder)
    source_stage.GetRootLayer().Save()
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    real_pxr = adapter_module._pxr()
    real_usd = real_pxr[2]
    stage_open_called = False

    class StageProxy:
        LoadAll = real_usd.Stage.LoadAll

        @staticmethod
        def Open(*args, **kwargs):
            nonlocal stage_open_called
            stage_open_called = True
            return real_usd.Stage.Open(*args, **kwargs)

    class UsdProxy:
        Stage = StageProxy
        TimeCode = real_usd.TimeCode
        PrimRange = real_usd.PrimRange

    monkeypatch.setattr(
        adapter_module,
        "_pxr",
        lambda: (*real_pxr[:2], UsdProxy, *real_pxr[3:]),
    )

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_surface_mesh_baking_points(
            stage=output_stage,
            source_usd_path=source,
            source_prim_path="/Holder/Visuals/Surface",
            output_prim_path="/Deformable/Visuals/Surface",
            source_world_bounds_center_m=center,
            recenter_to_output_origin=True,
            bake_scale_xyz=[1.0, 1.0, 1.0],
            flatten_source_xform=True,
        )
    assert "native_deformable_stage_source_composition_forbidden" in exc.value.errors
    assert stage_open_called is False


def test_source_parent_symlink_is_rejected(tmp_path: Path) -> None:
    source, _texture, center = _source_asset(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(source.parent, target_is_directory=True)
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_surface_mesh_baking_points(
            stage=output_stage,
            source_usd_path=alias / source.name,
            source_prim_path="/Source/Visuals/Surface",
            output_prim_path="/Deformable/Visuals/Surface",
            source_world_bounds_center_m=center,
            recenter_to_output_origin=True,
            bake_scale_xyz=[1.0, 1.0, 1.0],
            flatten_source_xform=True,
        )
    assert "native_deformable_stage_source_usd_invalid" in exc.value.errors


@pytest.mark.parametrize(
    "output_asset",
    ["../../outside.png", "/absolute/outside.png", "https://example.test/a.png", "pkg.usdz[a.png]"],
)
def test_output_texture_traversal_is_rejected(tmp_path: Path, output_asset: str) -> None:
    source, texture, center = _source_asset(tmp_path / "input")
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    adapter.copy_surface_mesh_baking_points(
        stage=output_stage,
        source_usd_path=source,
        source_prim_path="/Source/Visuals/Surface",
        output_prim_path="/Deformable/Visuals/Surface",
        source_world_bounds_center_m=center,
        recenter_to_output_origin=True,
        bake_scale_xyz=[1.0, 1.0, 1.0],
        flatten_source_xform=True,
    )

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_bound_material_network(
            stage=output_stage,
            source_usd_path=source,
            material_prim_path_map={"/Source/Looks/Material": "/Deformable/Looks/Material_000"},
            output_looks_prim_path="/Deformable/Looks",
            output_visual_prim_path="/Deformable/Visuals/Surface",
            source_texture_paths={"base.png": texture},
            output_texture_asset_paths={"base.png": output_asset},
        )
    assert "native_deformable_stage_output_texture_path_invalid" in exc.value.errors


def test_source_material_connection_to_output_namespace_is_rejected(tmp_path: Path) -> None:
    source, texture, center = _source_asset(tmp_path / "input")
    source_stage = Usd.Stage.Open(str(source))
    rogue = source_stage.GetPrimAtPath("/Source/Looks/Material/Preview").CreateAttribute(
        "inputs:rogue", Sdf.ValueTypeNames.Float
    )
    rogue.SetConnections([Sdf.Path("/Deformable/Looks/Material_000/Preview.outputs:surface")])
    source_stage.GetRootLayer().Save()
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    adapter.copy_surface_mesh_baking_points(
        stage=output_stage,
        source_usd_path=source,
        source_prim_path="/Source/Visuals/Surface",
        output_prim_path="/Deformable/Visuals/Surface",
        source_world_bounds_center_m=center,
        recenter_to_output_origin=True,
        bake_scale_xyz=[1.0, 1.0, 1.0],
        flatten_source_xform=True,
    )

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_bound_material_network(
            stage=output_stage,
            source_usd_path=source,
            material_prim_path_map={"/Source/Looks/Material": "/Deformable/Looks/Material_000"},
            output_looks_prim_path="/Deformable/Looks",
            output_visual_prim_path="/Deformable/Visuals/Surface",
            source_texture_paths={"base.png": texture},
            output_texture_asset_paths={"base.png": "textures/base.png"},
        )
    assert "native_deformable_stage_material_connection_outside_allowlist" in exc.value.errors


@pytest.mark.parametrize(
    "rogue_kind",
    [
        "provider_tetmesh",
        "outside_root",
        "inactive_outside_root",
        "schema_decoy",
        "asset_decoy",
        "connection_decoy",
    ],
)
def test_positive_stage_inventory_rejects_rogue_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rogue_kind: str
) -> None:
    adapter, stage = _prepared_native_stage(tmp_path, monkeypatch=monkeypatch)
    if rogue_kind in {"outside_root", "inactive_outside_root"}:
        stage.DefinePrim("/Rogue", "Xform")
        if rogue_kind == "inactive_outside_root":
            stage.GetPrimAtPath("/Rogue").SetActive(False)
    elif rogue_kind == "schema_decoy":
        stage.GetPrimAtPath("/Deformable/PhysicsMaterial").AddAppliedSchema(
            "PhysxDeformableBodyAPI"
        )
    elif rogue_kind == "asset_decoy":
        stage.GetPrimAtPath("/Deformable/Visuals/Surface").CreateAttribute(
            "custom:externalAsset", Sdf.ValueTypeNames.Asset
        ).Set(Sdf.AssetPath("../../outside.usd"))
    elif rogue_kind == "connection_decoy":
        stage.GetPrimAtPath("/Deformable/Visuals/Surface").CreateAttribute(
            "custom:decoyConnection", Sdf.ValueTypeNames.Token
        ).AddConnection("/Missing.value")
    else:
        rogue = stage.DefinePrim("/Deformable/Rogue", "TetMesh")
        rogue.GetAttribute("points").Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 1.0, 0.0),
                    Gf.Vec3f(0.0, 0.0, 1.0),
                ]
            )
        )
        rogue.GetAttribute("tetVertexIndices").Set(Vt.Vec4iArray([Gf.Vec4i(0, 1, 2, 3)]))
        rogue.CreateAttribute("Newton:solver", Sdf.ValueTypeNames.String).Set("provider")

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert (
        "native_deformable_stage_unexpected_prim_inventory" in exc.value.errors
        or "native_deformable_stage_forbidden_source_content_present" in exc.value.errors
    )


def test_material_network_mutation_fails_live_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, stage = _prepared_native_stage(tmp_path, monkeypatch=monkeypatch)
    texture_attribute = stage.GetPrimAtPath("/Deformable/Looks/Material_000/Texture").GetAttribute(
        "inputs:file"
    )
    texture_attribute.Set(Sdf.AssetPath("textures/substituted.png"))

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert "native_deformable_stage_material_network_readback_mismatch" in exc.value.errors


def test_physics_binding_strength_must_be_explicitly_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, stage = _prepared_native_stage(tmp_path, monkeypatch=monkeypatch)
    mesh = stage.GetPrimAtPath("/Deformable/Visuals/Surface")
    relationship = UsdShade.MaterialBindingAPI(mesh).GetDirectBindingRel("physics")
    assert relationship.ClearMetadata("bindMaterialAs")

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert "native_deformable_stage_material_binding_readback_invalid" in exc.value.errors


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("nonfinite_point", "native_deformable_stage_surface_points_invalid"),
        ("reflected_transform", "native_deformable_stage_source_transform_invalid"),
        ("time_varying_points", "native_deformable_stage_source_rest_state_time_varying"),
        ("open_surface", "native_deformable_stage_surface_not_closed_or_oriented"),
        ("hole", "native_deformable_stage_source_holes_forbidden"),
        ("asset_primvar", "native_deformable_stage_source_primvar_asset_forbidden"),
    ],
)
def test_source_geometry_and_rest_state_fail_closed(
    tmp_path: Path, case: str, expected_error: str
) -> None:
    source, _texture, center = _source_asset(tmp_path / "input")
    source_stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(source_stage.GetPrimAtPath("/Source/Visuals/Surface"))
    if case == "nonfinite_point":
        points = list(mesh.GetPointsAttr().Get())
        points[0] = Gf.Vec3f(float("nan"), points[0][1], points[0][2])
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    elif case == "reflected_transform":
        transform = Gf.Matrix4d(1.0)
        transform.SetScale(Gf.Vec3d(-0.2, 0.3, 0.4))
        transform.SetTranslateOnly(Gf.Vec3d(1.0, 2.0, 3.0))
        UsdGeom.Xformable(mesh).GetOrderedXformOps()[0].Set(transform)
    elif case == "time_varying_points":
        mesh.GetPointsAttr().Set(mesh.GetPointsAttr().Get(), Usd.TimeCode(1.0))
    elif case == "open_surface":
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3, 3, 3]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 2, 1, 0, 1, 3, 1, 2, 3]))
    elif case == "hole":
        mesh.GetHoleIndicesAttr().Set(Vt.IntArray([0]))
    else:
        UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "externalAsset", Sdf.ValueTypeNames.Asset, UsdGeom.Tokens.constant
        ).Set(Sdf.AssetPath("/tmp/outside.bin"))
    source_stage.GetRootLayer().Save()
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_surface_mesh_baking_points(
            stage=output_stage,
            source_usd_path=source,
            source_prim_path="/Source/Visuals/Surface",
            output_prim_path="/Deformable/Visuals/Surface",
            source_world_bounds_center_m=center,
            recenter_to_output_origin=True,
            bake_scale_xyz=[1.0, 1.0, 1.0],
            flatten_source_xform=True,
        )
    assert expected_error in exc.value.errors


def test_anisotropic_normal_transform_matches_baked_geometry(tmp_path: Path) -> None:
    source, _texture, center = _source_asset(tmp_path / "input")
    source_stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(source_stage.GetPrimAtPath("/Source/Visuals/Surface"))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 0.0)]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.constant)
    source_stage.GetRootLayer().Save()
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    adapter.copy_surface_mesh_baking_points(
        stage=output_stage,
        source_usd_path=source,
        source_prim_path="/Source/Visuals/Surface",
        output_prim_path="/Deformable/Visuals/Surface",
        source_world_bounds_center_m=center,
        recenter_to_output_origin=True,
        bake_scale_xyz=[2.0, 3.0, 4.0],
        flatten_source_xform=True,
    )

    observed = (
        UsdGeom.Mesh(output_stage.GetPrimAtPath("/Deformable/Visuals/Surface"))
        .GetNormalsAttr()
        .Get()[0]
    )
    expected = Gf.Vec3d(1.0 / 0.4, 1.0 / 0.9, 0.0).GetNormalized()
    assert list(observed) == pytest.approx(list(expected), abs=1.0e-6)


def test_frozen_point_resource_limit_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _texture, center = _source_asset(tmp_path / "input")
    monkeypatch.setattr(adapter_module, "_MAX_POINTS", 3)
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_surface_mesh_baking_points(
            stage=output_stage,
            source_usd_path=source,
            source_prim_path="/Source/Visuals/Surface",
            output_prim_path="/Deformable/Visuals/Surface",
            source_world_bounds_center_m=center,
            recenter_to_output_origin=True,
            bake_scale_xyz=[1.0, 1.0, 1.0],
            flatten_source_xform=True,
        )
    assert "native_deformable_stage_surface_points_invalid" in exc.value.errors


def test_cooked_and_stage_resource_limits_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, stage = _prepared_native_stage(tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(adapter_module, "_MAX_TET_POINTS", 3)
    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert "native_deformable_stage_simulation_points_invalid" in exc.value.errors

    adapter, stage = _prepared_native_stage(tmp_path / "stage", monkeypatch=monkeypatch)
    monkeypatch.setattr(adapter_module, "_MAX_TET_POINTS", 2_000_000)
    monkeypatch.setattr(adapter_module, "_MAX_STAGE_PRIMS", 1)
    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        _readback(adapter, stage)
    assert "native_deformable_stage_prim_limit_exceeded" in exc.value.errors


def test_malformed_public_input_is_typed_at_adapter_boundary(tmp_path: Path) -> None:
    source, _texture, _center = _source_asset(tmp_path / "input")
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    output_stage = adapter.create_clean_stage(
        output_path=tmp_path / "out.usda",
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )
    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.copy_surface_mesh_baking_points(
            stage=output_stage,
            source_usd_path=source,
            source_prim_path="/Source/Visuals/Surface",
            output_prim_path="/Deformable/Visuals/Surface",
            source_world_bounds_center_m=None,  # type: ignore[arg-type]
            recenter_to_output_origin=True,
            bake_scale_xyz=[1.0, 1.0, 1.0],
            flatten_source_xform=True,
        )
    assert "native_deformable_stage_surface_copy_failed" in exc.value.errors


def test_save_failure_releases_owned_current_stage_context(tmp_path: Path) -> None:
    class Context:
        exits = 0

        def __exit__(self, *_args) -> None:
            self.exits += 1

    class RootLayer:
        def Export(self, _path: str) -> bool:
            raise RuntimeError("bounded export failure")

    class Stage:
        def GetRootLayer(self) -> RootLayer:
            return RootLayer()

    stage = Stage()
    context = Context()
    adapter = OpenUsdNativeDeformableStageAdapter()
    adapter._state[stage] = {
        "surface": {},
        "material": {},
        "output_path": tmp_path / "out.usda",
        "current_stage_context": context,
    }

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.save_stage(stage=stage)
    assert "native_deformable_stage_save_failed" in exc.value.errors
    assert context.exits == 1
    adapter.release_current_stage(stage=stage)
    assert context.exits == 1


def test_current_stage_verification_exception_releases_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out.usda"
    adapter = OpenUsdNativeDeformableStageAdapter(stage_factory=Usd.Stage.CreateInMemory)
    stage = adapter.create_clean_stage(
        output_path=output,
        default_prim_path="/Deformable",
        meters_per_unit=1.0,
        up_axis="Z",
    )

    class Context:
        exits = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self.exits += 1

    context = Context()
    probes = 0

    def get_current_stage():
        nonlocal probes
        probes += 1
        if probes == 1:
            return None
        raise RuntimeError("bounded verification failure")

    stage_module = ModuleType("isaaclab.sim.utils.stage")
    stage_module.get_current_stage = get_current_stage
    stage_module.use_stage = lambda _stage: context
    isaaclab_module = ModuleType("isaaclab")
    sim_module = ModuleType("isaaclab.sim")
    utils_module = ModuleType("isaaclab.sim.utils")
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", utils_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.stage", stage_module)

    with pytest.raises(NativeDeformableAssetStageAdapterError) as exc:
        adapter.activate_and_verify_current_stage(stage=stage)
    assert "native_deformable_stage_current_stage_activation_failed" in exc.value.errors
    assert context.exits == 1
    assert adapter._entry(stage)["current_stage_context"] is None
