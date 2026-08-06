from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from blueprint_pipeline.adp009d_sage_franka_placement import (
    SageFrankaPlacementError,
    materialize_sage_collision_analysis_glb,
)


def _write_stage(path: Path, *, meters_per_unit: float = 1.0) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/Root")
    root.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Collider")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateExtentAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 0)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.SetDefaultPrim(root.GetPrim())
    assert stage.GetRootLayer().Save()
    assert stage.GetRootLayer().realPath == str(path)
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/Root")


def test_sage_conversion_preserves_world_coordinates_and_source_bytes(tmp_path) -> None:
    source = tmp_path / "scene.usda"
    _write_stage(source)
    before = source.read_bytes()

    receipt = materialize_sage_collision_analysis_glb(
        sage_usd_path=source,
        output_dir=tmp_path / "conversion",
    )

    assert source.read_bytes() == before
    assert receipt["source_usd_mutated"] is False
    assert receipt["simulation_asset_replacement"] is False
    assert receipt["mesh_count"] == 1
    assert receipt["vertex_count"] == 3
    assert receipt["triangle_count"] == 1
    glb = trimesh.load(receipt["analysis_glb"]["path"], force="mesh", process=False)
    stage_points = np.column_stack(
        (glb.vertices[:, 0], -glb.vertices[:, 2], glb.vertices[:, 1])
    )
    assert np.allclose(stage_points.min(axis=0), [1.0, 2.0, 3.0])
    assert np.allclose(stage_points.max(axis=0), [2.0, 3.0, 3.0])


def test_sage_conversion_rejects_nonmetric_stage(tmp_path) -> None:
    source = tmp_path / "centimeter_scene.usda"
    _write_stage(source, meters_per_unit=0.01)

    with pytest.raises(
        SageFrankaPlacementError, match="sage_collision_stage_not_meter_units"
    ):
        materialize_sage_collision_analysis_glb(
            sage_usd_path=source,
            output_dir=tmp_path / "conversion",
        )
