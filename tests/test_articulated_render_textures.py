from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

from blueprint_pipeline.articulated_render_textures import (
    RENDER_TEXTURE_SCHEMA_VERSION,
    ArticulatedRenderTextureError,
    bind_render_textures,
)


def _png(path: Path) -> Path:
    # A one-pixel PNG is enough: what matters is that the file resolves.
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
            "00000049454e44ae426082"
        )
    )
    return path


def _scaffolded(tmp_path: Path) -> Path:
    """An asset with a bound UsdPreviewSurface but no textures yet."""

    path = tmp_path / "twin.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/door/panel")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    material = UsdShade.Material.Define(stage, "/Asset/Looks/Render/door_shell")
    shader = UsdShade.Shader.Define(stage, "/Asset/Looks/Render/door_shell/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.6, 0.6))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return path


def _bind(tmp_path: Path, **overrides):
    arguments = {
        "source_usd_path": _scaffolded(tmp_path),
        "destination": tmp_path / "textured.usda",
        "bindings": [
            {
                "material_path": "/Asset/Looks/Render/door_shell",
                "albedo_path": str(_png(tmp_path / "albedo.png")),
                "roughness_path": str(_png(tmp_path / "rough.png")),
            }
        ],
    }
    arguments.update(overrides)
    return bind_render_textures(**arguments)


def test_the_albedo_reaches_the_surface_shader(tmp_path: Path) -> None:
    receipt = _bind(tmp_path)

    stage = Usd.Stage.Open(receipt["textured_usd_path"])
    shader = UsdShade.Shader.Get(stage, "/Asset/Looks/Render/door_shell/Shader")
    source = shader.GetInput("diffuseColor").GetConnectedSource()
    assert source is not None
    assert receipt["schema_version"] == RENDER_TEXTURE_SCHEMA_VERSION


def test_colour_maps_are_srgb_and_data_maps_are_raw(tmp_path: Path) -> None:
    """Reading a roughness map as sRGB silently gamma-shifts it.

    Nothing errors and the render still looks like a render - the surface is
    just wrong, in a direction nobody can eyeball. It is the single easiest
    texture mistake to make and the hardest to notice.
    """

    receipt = _bind(tmp_path)

    stage = Usd.Stage.Open(receipt["textured_usd_path"])
    albedo = UsdShade.Shader.Get(
        stage, "/Asset/Looks/Render/door_shell/albedo_texture"
    )
    rough = UsdShade.Shader.Get(
        stage, "/Asset/Looks/Render/door_shell/roughness_texture"
    )
    assert albedo.GetInput("sourceColorSpace").Get() == "sRGB"
    assert rough.GetInput("sourceColorSpace").Get() == "raw"


def test_a_texture_reader_without_uvs_fails_closed(tmp_path: Path) -> None:
    """A texture with no st input samples one texel and renders flat."""

    receipt = _bind(tmp_path)

    stage = Usd.Stage.Open(receipt["textured_usd_path"])
    albedo = UsdShade.Shader.Get(
        stage, "/Asset/Looks/Render/door_shell/albedo_texture"
    )
    assert albedo.GetInput("st").GetConnectedSource() is not None
    assert receipt["uv_reader_authored"] is True


def test_a_missing_texture_file_fails_closed(tmp_path: Path) -> None:
    """A dangling asset path renders as flat magenta, not as an error."""

    with pytest.raises(ArticulatedRenderTextureError) as excinfo:
        _bind(
            tmp_path,
            bindings=[
                {
                    "material_path": "/Asset/Looks/Render/door_shell",
                    "albedo_path": str(tmp_path / "absent.png"),
                }
            ],
        )

    assert any("texture_file_missing" in e for e in excinfo.value.errors)


def test_a_binding_onto_a_material_that_does_not_exist_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArticulatedRenderTextureError) as excinfo:
        _bind(
            tmp_path,
            bindings=[
                {
                    "material_path": "/Asset/Looks/Render/ghost",
                    "albedo_path": str(_png(tmp_path / "a.png")),
                }
            ],
        )

    assert any("material_missing" in e for e in excinfo.value.errors)


def test_the_source_is_never_written_over(tmp_path: Path) -> None:
    source = _scaffolded(tmp_path)
    before = source.read_bytes()

    _bind(tmp_path, source_usd_path=source)

    assert source.read_bytes() == before


def test_binding_is_deterministic(tmp_path: Path) -> None:
    first = _bind(tmp_path, destination=tmp_path / "a.usda")
    second = _bind(tmp_path, destination=tmp_path / "b.usda")

    assert first["textured_usd_sha256"] == second["textured_usd_sha256"]
