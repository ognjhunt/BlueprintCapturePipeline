from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.articulated_source_asset import (
    ArticulatedSourceAssetError,
    materialize_articulated_source_asset,
)


def _box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> list[dict]:
    return [
        {"x": x, "y": y, "z": z}
        for z in (z0, z1)
        for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))
    ]


def _fixture(tmp_path: Path, *, disconnected: bool) -> tuple[Path, Path]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            [
                {
                    "ins_id": "160" if not disconnected else "121",
                    "label": "canned beverage" if not disconnected else "refrigerator",
                    "bounding_box": _box(4.0, 5.0, 0.0, 5.0, 6.0, 2.0),
                }
            ]
        ),
        encoding="utf-8",
    )
    collision = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(collision))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    root.AddTranslateOp().Set(Gf.Vec3d(4.0, 5.0, 0.0))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Target")
    points = [
        (0, 0, 0), (0.45, 0, 0), (0.45, 1, 0), (0, 1, 0),
        (0, 0, 2), (0.45, 0, 2), (0.45, 1, 2), (0, 1, 2),
    ]
    counts = [4, 4, 4, 4, 4, 4]
    indices = [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 4, 0, 3, 7]
    if disconnected:
        points += [
            (0.55, 0, 0), (1, 0, 0), (1, 1, 0), (0.55, 1, 0),
            (0.55, 0, 2), (1, 0, 2), (1, 1, 2), (0.55, 1, 2),
        ]
        counts += [4, 4, 4, 4, 4, 4]
        indices += [8, 9, 10, 11, 12, 15, 14, 13, 8, 12, 13, 9, 9, 13, 14, 10, 10, 14, 15, 11, 12, 8, 11, 15]
    else:
        points[1] = (1, 0, 0)
        points[2] = (1, 1, 0)
        points[5] = (1, 0, 2)
        points[6] = (1, 1, 2)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return labels, collision


def test_original_rigid_fixture_extracts_one_component(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path, disconnected=False)

    receipt = materialize_articulated_source_asset(
        labels_path=labels,
        target_instance_id="160",
        sage_collision_usd_path=collision,
        output_dir=tmp_path / "out",
    )

    assert receipt["connected_component_count"] == 1
    assert receipt["joint_agent_0_5_2_input"]["single_source_mesh_requires_split_meshes"] is False
    assert receipt["asset_frame"]["origin_world_m"] == [4.5, 5.5, 0.0]
    assert (tmp_path / "out/articulated_source_mesh.usda").is_file()


def test_articulated_fixture_reports_split_candidates_without_naming_links(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path, disconnected=True)

    receipt = materialize_articulated_source_asset(
        labels_path=labels,
        target_instance_id="121",
        sage_collision_usd_path=collision,
        output_dir=tmp_path / "out",
    )

    assert receipt["connected_component_count"] == 2
    assert receipt["joint_agent_0_5_2_input"]["predicted_split_prim_count"] == 2
    assert receipt["joint_agent_0_5_2_input"]["connected_component_geom_subsets_authored"]
    assert receipt["claim_boundary"]["connected_components_are_not_rigid_links"]
    assert receipt["claim_boundary"]["joint_topology_inferred"] is False
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(tmp_path / "out/articulated_source_mesh.usda"))
    subsets = [
        subset
        for subset in UsdGeom.Subset.GetAllGeomSubsets(
            UsdGeom.Imageable(stage.GetPrimAtPath("/Asset/source_mesh"))
        )
        if subset.GetElementTypeAttr().Get() == UsdGeom.Tokens.face
        and subset.GetFamilyNameAttr().Get() == "blueprint_connected_components"
    ]
    assert len(subsets) == 2
    assert sorted(len(subset.GetIndicesAttr().Get()) for subset in subsets) == [6, 6]


def test_extractor_rejects_nonempty_output(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path, disconnected=False)
    output = tmp_path / "out"
    output.mkdir()
    (output / "owned.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(ArticulatedSourceAssetError) as caught:
        materialize_articulated_source_asset(
            labels_path=labels,
            target_instance_id="160",
            sage_collision_usd_path=collision,
            output_dir=output,
        )

    assert caught.value.errors == ("articulated_source_output_not_empty",)
    assert (output / "owned.txt").read_text(encoding="utf-8") == "preserve"
