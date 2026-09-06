"""Completed formats retain geometry, appearance bindings, and declared frames."""
from pathlib import Path
import shutil

import numpy as np
import pytest

from blueprint_pipeline.task_evaluation_completed_scene_geometry import normalize_completed_mesh
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import _extract_source_candidate_subtree
from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import _normalize_candidate
from blueprint_pipeline.task_evaluation_completed_scene_splat import normalize_completed_splat
from blueprint_pipeline.gaussian_splat_decode import find_splat_transform_cli, read_standard_3dgs_ply
from tests.test_task_evaluation_completed_scene_source import MESH
from tests.test_provided_scene_splat import splat_bytes


def test_y_up_centimeter_mesh_retains_world_geometry_and_material_through_extraction(tmp_path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    source = tmp_path / "source.usda"
    source.write_bytes(MESH.replace(b'metersPerUnit = 1; upAxis = "Z"', b'metersPerUnit = 0.01; upAxis = "Y"'))
    stage = Usd.Stage.Open(str(source))
    material = UsdShade.Material.Define(stage, "/Looks/Paint")
    shader = UsdShade.Shader.Define(stage, "/Looks/Paint/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.4, 0.6))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/Book")).Bind(material)
    stage.GetRootLayer().Save()
    original_bytes = source.read_bytes()
    root = tmp_path / "normalized"
    normalized = normalize_completed_mesh(source=source, original_filename=source.name,
        coordinate_frame={"meters_per_unit": 0.01, "up_axis": "Y"}, output_root=root)
    derived = root / normalized["output"]["relative_path"]
    candidate = tmp_path / "candidate.usda"
    _extract_source_candidate_subtree(source_stage_path=derived, prim_path=normalized["object_mapping"]["/Book"], output_path=candidate)
    opened = Usd.Stage.Open(str(candidate))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(opened.GetDefaultPrim()).ComputeAlignedRange()
    assert list(box.GetMin()) == pytest.approx([0, -0.0077, 0], abs=1e-7)
    assert list(box.GetMax()) == pytest.approx([0.0015, -0.0075, 0.002], abs=1e-7)
    authored = tmp_path / "authored.usda"
    _normalize_candidate(candidate, authored)
    opened = Usd.Stage.Open(str(authored))
    visual = opened.GetPrimAtPath("/Asset/Geometry/Visual")
    bound, _ = UsdShade.MaterialBindingAPI(visual).ComputeBoundMaterial()
    assert bound
    shader_source = bound.ComputeSurfaceSource()[0]
    assert list(shader_source.GetInput("diffuseColor").Get()) == pytest.approx([0.2, 0.4, 0.6])
    assert source.read_bytes() == original_bytes
    assert normalize_completed_mesh(source=source, original_filename=source.name,
        coordinate_frame={"meters_per_unit": 0.01, "up_axis": "Y"}, output_root=root) == normalized


@pytest.mark.slow
def test_real_pinned_splat_converter_normalizes_y_up_without_training_or_repeating(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    if find_splat_transform_cli(repo) is None or shutil.which("node") is None:
        pytest.skip("Pinned optional Node converter is exercised with the renderer runtime installed")
    source = tmp_path / "source.ply"
    source.write_bytes(splat_bytes())
    def runtime(**_):
        return {"node": shutil.which("node"), "renderer_root": str(repo),
                "identity": {"runtime_digest": "sha256:" + "d" * 64}}
    result = normalize_completed_splat(source=source, coordinate_frame={"meters_per_unit": 0.01, "up_axis": "Y"},
        output_root=tmp_path / "out", runtime_resolver=runtime)
    original = read_standard_3dgs_ply(source)
    normalized = read_standard_3dgs_ply(tmp_path / "out/normalized.ply")
    np.testing.assert_allclose(normalized.xyz, original.xyz[:, [0, 2, 1]] * [0.01, -0.01, 0.01], atol=1e-6)
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_completed_scene_splat.subprocess.run",
                        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("completed conversion repeated")))
    assert normalize_completed_splat(source=source, coordinate_frame={"meters_per_unit": 0.01, "up_axis": "Y"},
        output_root=tmp_path / "out", runtime_resolver=runtime) == result
    assert result["reconstruction_performed"] is False and result["renderer_qualified"] is False
