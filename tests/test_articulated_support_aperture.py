from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_support_aperture import (
    SUPPORT_APERTURE_SCHEMA_VERSION,
    ArticulatedSupportApertureError,
    cut_support_link_aperture,
)
from blueprint_pipeline.articulated_interior_exposure import evaluate_interior_exposure


def _sealed(path: Path) -> Path:
    """A carcass whose front face walls off the interior behind it."""

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    for link in ("cabinet", "upper_door"):
        UsdPhysics.RigidBodyAPI.Apply(
            UsdGeom.Xform.Define(stage, f"/Asset/{link}").GetPrim()
        )

    def box(prim_path, xmin, xmax, ymin, ymax, zmin, zmax):
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        corners = [
            Gf.Vec3f(x, y, z)
            for x in (xmin, xmax)
            for y in (ymin, ymax)
            for z in (zmin, zmax)
        ]
        mesh.CreatePointsAttr(corners)
        quads = [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        ]
        counts, indices = [], []
        for quad in quads:
            counts.extend([3, 3])
            indices.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
            "convexHull"
        )
        return mesh

    box("/Asset/cabinet/shell", -0.35, 0.35, -0.35, 0.18, 0.0, 1.63)
    box("/Asset/cabinet/generated_interior", -0.31, 0.31, -0.31, 0.14, 0.04, 1.59)
    box("/Asset/upper_door/panel", -0.35, 0.35, 0.30, 0.35, 0.94, 1.63)
    stage.GetRootLayer().Save()
    return path


def _cut(tmp_path: Path, source: Path, **overrides):
    arguments = {
        "source_usd_path": source,
        "destination": tmp_path / "opened.usda",
        "support_link_path": "/Asset/cabinet",
        "aperture_x_interval_m": [-0.30, 0.30],
        "aperture_z_interval_m": [0.95, 1.58],
        "outward_axis": [0.0, 1.0, 0.0],
        "protected_prim_paths": ["/Asset/cabinet/generated_interior"],
    }
    arguments.update(overrides)
    return cut_support_link_aperture(**arguments)


def test_cutting_the_aperture_exposes_the_interior(tmp_path: Path) -> None:
    """The whole point: after the cut, rays through the door reach the cavity."""

    source = _sealed(tmp_path / "sealed.usda")
    before = evaluate_interior_exposure(
        replacement_usd_path=source,
        support_link_path="/Asset/cabinet",
        task_door_link_path="/Asset/upper_door",
        interior_prim_paths=["/Asset/cabinet/generated_interior"],
        aperture_plane_y_m=0.30,
        aperture_x_interval_m=[-0.28, 0.28],
        aperture_z_interval_m=[0.97, 1.56],
        samples_per_axis=9,
    )
    assert before["interior_exposed"] is False

    receipt = _cut(tmp_path, source)
    after = evaluate_interior_exposure(
        replacement_usd_path=receipt["opened_usd_path"],
        support_link_path="/Asset/cabinet",
        task_door_link_path="/Asset/upper_door",
        interior_prim_paths=["/Asset/cabinet/generated_interior"],
        aperture_plane_y_m=0.30,
        aperture_x_interval_m=[-0.28, 0.28],
        aperture_z_interval_m=[0.97, 1.56],
        samples_per_axis=9,
    )

    assert receipt["schema_version"] == SUPPORT_APERTURE_SCHEMA_VERSION
    assert receipt["status"] == "support_aperture_cut"
    assert receipt["faces_removed"] > 0
    assert after["interior_exposed"] is True
    assert after["exposed_fraction"] == 1.0


def test_the_cut_cannot_change_a_convex_hull_collider(tmp_path: Path) -> None:
    """Points are only added on the existing face plane, never moved or dropped."""

    source = _sealed(tmp_path / "sealed.usda")
    source_stage = Usd.Stage.Open(str(source))
    original = UsdGeom.Mesh(source_stage.GetPrimAtPath("/Asset/cabinet/shell"))
    original_points = np.array(
        [[float(v) for v in p] for p in original.GetPointsAttr().Get()]
    )

    receipt = _cut(tmp_path, source)

    opened_stage = Usd.Stage.Open(receipt["opened_usd_path"])
    opened = UsdGeom.Mesh(opened_stage.GetPrimAtPath("/Asset/cabinet/shell"))
    points = np.array([[float(v) for v in p] for p in opened.GetPointsAttr().Get()])
    assert points.shape[0] >= original_points.shape[0]
    assert np.array_equal(points[: original_points.shape[0]], original_points)
    assert points.min(axis=0).tolist() == original_points.min(axis=0).tolist()
    assert points.max(axis=0).tolist() == original_points.max(axis=0).tolist()
    assert receipt["collision"]["approximation_unchanged"] is True
    assert receipt["collision"]["convex_hull_point_set_preserved"] is True


def test_only_outward_facing_faces_on_the_aperture_plane_are_cut(
    tmp_path: Path,
) -> None:
    source = _sealed(tmp_path / "sealed.usda")

    receipt = _cut(tmp_path, source)

    cut = next(row for row in receipt["meshes"] if row["prim_path"].endswith("shell"))
    assert cut["faces_removed"] > 0
    assert cut["faces_added"] >= 0
    # the back, sides, top and bottom of the carcass survive
    opened = Usd.Stage.Open(receipt["opened_usd_path"])
    mesh = UsdGeom.Mesh(opened.GetPrimAtPath("/Asset/cabinet/shell"))
    counts = [int(v) for v in mesh.GetFaceVertexCountsAttr().Get()]
    assert len(counts) >= 8


def test_protected_prims_are_never_cut(tmp_path: Path) -> None:
    source = _sealed(tmp_path / "sealed.usda")

    receipt = _cut(tmp_path, source)

    touched = {row["prim_path"] for row in receipt["meshes"] if row["faces_removed"]}
    assert "/Asset/cabinet/generated_interior" not in touched


def test_a_cut_that_removes_nothing_fails_closed(tmp_path: Path) -> None:
    source = _sealed(tmp_path / "sealed.usda")

    with pytest.raises(ArticulatedSupportApertureError) as excinfo:
        _cut(
            tmp_path,
            source,
            aperture_z_interval_m=[2.4, 2.6],  # above the carcass entirely
        )

    assert any("aperture_removed_nothing" in error for error in excinfo.value.errors)


def test_an_over_wide_cut_fails_closed(tmp_path: Path) -> None:
    source = _sealed(tmp_path / "sealed.usda")

    with pytest.raises(ArticulatedSupportApertureError) as excinfo:
        _cut(
            tmp_path,
            source,
            aperture_x_interval_m=[-5.0, 5.0],
            aperture_z_interval_m=[-5.0, 5.0],
            maximum_removed_area_fraction=0.05,
        )

    assert any(
        "aperture_removed_area_above_ceiling" in error for error in excinfo.value.errors
    )


def test_the_source_asset_is_never_modified(tmp_path: Path) -> None:
    source = _sealed(tmp_path / "sealed.usda")
    before = source.read_bytes()

    receipt = _cut(tmp_path, source)

    assert source.read_bytes() == before
    assert receipt["source_usd_sha256"].startswith("sha256:")
    assert receipt["opened_usd_sha256"] != receipt["source_usd_sha256"]


def test_articulation_survives_the_cut(tmp_path: Path) -> None:
    source = _sealed(tmp_path / "sealed.usda")

    receipt = _cut(tmp_path, source)

    stage = Usd.Stage.Open(receipt["opened_usd_path"])
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(roots) == 1 and len(bodies) == 2
    assert receipt["preserved"]["articulation_root_count"] == 1
    assert receipt["preserved"]["rigid_body_count"] == 2


def test_cut_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    source = _sealed(tmp_path / "sealed.usda")

    first = _cut(tmp_path, source, destination=tmp_path / "a.usda")
    second = _cut(tmp_path, source, destination=tmp_path / "b.usda")

    assert first["opened_usd_sha256"] == second["opened_usd_sha256"]
    stored = json.loads(Path(first["receipt_path"]).read_text(encoding="utf-8"))
    assert stored == first
