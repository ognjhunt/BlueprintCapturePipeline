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


def test_runtime_execution_graph_is_refused_before_normalized_output(tmp_path):
    source = tmp_path / "graph.usda"
    source.write_bytes(MESH + b'\ndef OmniGraph "Execution" {}\n')
    with pytest.raises(ValueError, match="completed_mesh_prim_type_not_supported"):
        normalize_completed_mesh(source=source, original_filename=source.name,
            coordinate_frame={"meters_per_unit": 1.0, "up_axis": "Z"}, output_root=tmp_path / "out")
    assert not (tmp_path / "out/normalized_scene.usda").exists()


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


def test_the_pipelines_own_layer_past_the_source_cap_is_read_back(tmp_path, monkeypatch):
    # 2026-10-30 capture-447ae4f4: a 70.6 MB Atlas GLB wrote a 292 MB normalized_scene.usda, and
    # the read-back of the pipeline's own layer refused it under the 256 MiB source cap. A
    # normalization written before is still text and still read back past the cap.
    from blueprint_pipeline import provided_scene_mesh
    from blueprint_pipeline.provided_scene_mesh import inspect_mesh
    from blueprint_pipeline.reconstruction_capability import ReconstructionContractError

    source = tmp_path / "atlas_mesh_levelled.glb"
    source.write_bytes(_icosphere_glb())
    frame = {"meters_per_unit": 1.1, "up_axis": "Y"}
    root = tmp_path / "normalized"
    normalized = normalize_completed_mesh(source=source, original_filename=source.name,
                                          coordinate_frame=frame, output_root=root)
    output = as_text_normalization(root) / normalized["output"]["relative_path"].replace(".usdc", ".usda")
    monkeypatch.setattr(provided_scene_mesh, "MAX_INSPECTION_BYTES", output.stat().st_size - 1)
    assert inspect_mesh(output, original_filename=output.name, normalized_output=True,
                        coordinate_frame_declaration={"meters_per_unit": 1.0, "up_axis": "Z"}
                        )["objects"] == normalized["normalized_inspection"]["objects"]
    # Only the pipeline's own layer is read back past the cap; a supplied file of that size is refused.
    with pytest.raises(ReconstructionContractError, match="inspection_size_limit"):
        inspect_mesh(output, original_filename=output.name,
                     coordinate_frame_declaration={"meters_per_unit": 1.0, "up_axis": "Z"})
    monkeypatch.setattr(provided_scene_mesh, "MAX_INSPECTION_BYTES", source.stat().st_size - 1)
    with pytest.raises(ReconstructionContractError, match="inspection_size_limit"):
        normalize_completed_mesh(source=source, original_filename=source.name,
                                 coordinate_frame=frame, output_root=tmp_path / "refused")


def _icosphere_glb(*, colors: bool = False) -> bytes:
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=4)
    if colors:
        mesh.visual.vertex_colors = np.random.default_rng(3).integers(0, 256, (len(mesh.vertices), 4), dtype=np.uint8)
    return mesh.export(file_type="glb")


def as_text_normalization(root: Path) -> Path:
    """``root``'s normalization as code before 2026-11-03 wrote it: a USDA text layer, the receipt naming it."""
    import json

    from pxr import Sdf

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import sha

    receipt_path = root / "mesh_normalization.v1.json"
    receipt = json.loads(receipt_path.read_text())
    crate = root / receipt["output"]["relative_path"]
    text = crate.with_suffix(".usda")
    assert Sdf.Layer.OpenAsAnonymous(str(crate)).Export(str(text))
    crate.unlink()
    receipt["output"] = {"relative_path": text.name, "sha256": sha(text), "size_bytes": text.stat().st_size}
    receipt["normalization_digest"] = canonical_digest(receipt, digest_field="normalization_digest")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    return root


@pytest.mark.parametrize("frame", [{"meters_per_unit": 1.1, "up_axis": "Y"}, {"meters_per_unit": 0.01, "up_axis": "-Y"},
                                   {"meters_per_unit": 1.0, "up_axis": "Z"}])
def test_a_normalized_room_is_binary_usd_with_the_values_the_text_layer_had(tmp_path, frame):
    # capture-447's room as USDA text was 292 MB, and each open that found no stage of it alive parsed it
    # again at about 3.6 GB. The crate layer opens without a parse; its values are the text layer's.
    import trimesh
    from pxr import Usd, UsdGeom, Vt

    from blueprint_pipeline.provided_scene_mesh import inspect_mesh

    source = tmp_path / "atlas_mesh_levelled.glb"
    source.write_bytes(_icosphere_glb(colors=True))
    root = tmp_path / "normalized"
    normalized = normalize_completed_mesh(source=source, original_filename=source.name,
                                          coordinate_frame=frame, output_root=root)
    output = root / normalized["output"]["relative_path"]
    assert output.name == "normalized_scene.usdc" and output.read_bytes()[:8] == b"PXR-USDC"
    # As the Python lists written before converted them: each double to the nearest float.
    original = trimesh.load(source, file_type="glb", force="scene", process=False)
    [node] = original.graph.nodes_geometry
    transform, name = original.graph[node]
    points = trimesh.transform_points(original.geometry[name].vertices, transform) * frame["meters_per_unit"]
    if frame["up_axis"] in {"Y", "-Y"}:
        points = points[:, [0, 2, 1]] * ([1, -1, 1] if frame["up_axis"] == "Y" else [1, 1, -1])
    colors = np.asarray(original.geometry[name].visual.vertex_colors, dtype=float) / 255.0
    stage = Usd.Stage.Open(str(output))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(normalized["object_mapping"][str(node)]))
    assert mesh.GetPointsAttr().Get() == Vt.Vec3fArray(points.tolist())
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == original.geometry[name].faces.reshape(-1).tolist()
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3] * len(original.geometry[name].faces)
    assert mesh.GetDisplayColorPrimvar().Get() == Vt.Vec3fArray(colors[:, :3].tolist())
    assert mesh.GetDisplayOpacityPrimvar().Get() == Vt.FloatArray(colors[:, 3].tolist())
    # Its text form reads back the geometry the receipt sealed (the inspection also names the file's digest).
    text = tmp_path / "text.usda"
    assert stage.GetRootLayer().Export(str(text))
    assert inspect_mesh(text, original_filename=text.name, normalized_output=True,
                        coordinate_frame_declaration={"meters_per_unit": 1.0, "up_axis": "Z"}
                        )["objects"] == normalized["normalized_inspection"]["objects"]


def test_a_text_normalization_written_before_is_kept_as_it_is(tmp_path):
    # Already-published scenes keep their USDA collider: its receipt is reused, and nothing is rewritten.
    source = tmp_path / "atlas_mesh_levelled.glb"
    source.write_bytes(_icosphere_glb())
    frame = {"meters_per_unit": 1.0, "up_axis": "Z"}
    root = tmp_path / "normalized"
    normalize_completed_mesh(source=source, original_filename=source.name, coordinate_frame=frame,
                             output_root=root)
    before = {path.name: path.read_bytes() for path in as_text_normalization(root).iterdir()}
    again = normalize_completed_mesh(source=source, original_filename=source.name, coordinate_frame=frame,
                                     output_root=root)
    assert again["output"]["relative_path"] == "normalized_scene.usda"
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
