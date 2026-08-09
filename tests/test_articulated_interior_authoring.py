from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_interior_authoring import (
    INTERIOR_AUTHORING_SCHEMA_VERSION,
    ArticulatedInteriorAuthoringError,
    author_articulated_interior,
)
from blueprint_pipeline.articulated_interior_exposure import evaluate_interior_exposure


PROV = "blueprint:articulatedReplacement:provenance"


def _shell(path: Path) -> Path:
    """A carcass with its aperture already cut and a solid placeholder inside."""

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

    def box(prim_path, xmin, xmax, ymin, ymax, zmin, zmax, tag=None):
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(x, y, z)
                for x in (xmin, xmax)
                for y in (ymin, ymax)
                for z in (zmin, zmax)
            ]
        )
        quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        counts, indices = [], []
        for quad in quads:
            counts.extend([3, 3])
            indices.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        if tag:
            mesh.GetPrim().CreateAttribute(PROV, Sdf.ValueTypeNames.String).Set(tag)
        return mesh

    box("/Asset/cabinet/shell_left", -0.35, -0.30, -0.35, 0.18, 0.0, 1.63)
    box("/Asset/cabinet/shell_right", 0.30, 0.35, -0.35, 0.18, 0.0, 1.63)
    box("/Asset/cabinet/shell_back", -0.35, 0.35, -0.35, -0.30, 0.0, 1.63)
    box(
        "/Asset/cabinet/generated_interior",
        -0.31, 0.31, -0.31, 0.14, 0.04, 1.59,
        tag="generated_candidate_geometry",
    )
    box("/Asset/upper_door/panel", -0.35, 0.35, 0.30, 0.35, 0.94, 1.63)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/upper_door_hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/cabinet")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/upper_door")])
    joint.CreateAxisAttr().Set("Z")
    stage.GetRootLayer().Save()
    return path


def _author(tmp_path: Path, **overrides):
    arguments = {
        "source_usd_path": _shell(tmp_path / "shell.usda"),
        "destination": tmp_path / "interior.usda",
        "support_link_path": "/Asset/cabinet",
        "replace_prim_paths": ["/Asset/cabinet/generated_interior"],
        "cavity_x_interval_m": [-0.30, 0.30],
        "cavity_y_interval_m": [-0.30, 0.15],
        "cavity_z_interval_m": [0.96, 1.58],
        "wall_thickness_m": 0.012,
        "shelf_count": 3,
        "shelf_thickness_m": 0.010,
        "door_bin_count": 0,
    }
    arguments.update(overrides)
    return author_articulated_interior(**arguments)


def test_the_placeholder_block_is_replaced_by_a_hollow_liner(tmp_path: Path) -> None:
    """A solid inset box is not an interior; nothing can go inside it."""

    receipt = _author(tmp_path)

    stage = Usd.Stage.Open(receipt["interior_usd_path"])
    # the placeholder block is gone; the path is now a scope holding real parts
    scope = stage.GetPrimAtPath("/Asset/cabinet/generated_interior")
    assert not scope.IsA(UsdGeom.Mesh)
    assert len(list(scope.GetChildren())) >= 5
    assert receipt["schema_version"] == INTERIOR_AUTHORING_SCHEMA_VERSION
    assert receipt["status"] == "articulated_interior_authored"
    assert receipt["removed_prim_paths"] == ["/Asset/cabinet/generated_interior"]
    parts = {row["role"] for row in receipt["parts"]}
    assert {"liner_back", "liner_left", "liner_right", "liner_floor", "liner_ceiling"} <= parts
    assert receipt["free_volume_m3"] > 0.05


def test_shelves_are_authored_inside_the_cavity(tmp_path: Path) -> None:
    receipt = _author(tmp_path, shelf_count=3)

    shelves = [row for row in receipt["parts"] if row["role"] == "shelf"]
    assert len(shelves) == 3
    zs = sorted(row["world_aabb_min_m"][2] for row in shelves)
    assert all(0.96 < z < 1.58 for z in zs)
    assert len(set(round(z, 4) for z in zs)) == 3  # distinct heights
    stage = Usd.Stage.Open(receipt["interior_usd_path"])
    for row in shelves:
        prim = stage.GetPrimAtPath(row["prim_path"])
        assert prim.IsValid() and prim.IsA(UsdGeom.Mesh)


def test_every_authored_part_is_labelled_generated_candidate_geometry(
    tmp_path: Path,
) -> None:
    """The real interior was never observed; nothing here may claim otherwise."""

    receipt = _author(tmp_path)

    stage = Usd.Stage.Open(receipt["interior_usd_path"])
    for row in receipt["parts"]:
        prim = stage.GetPrimAtPath(row["prim_path"])
        assert prim.GetAttribute(PROV).Get() == "generated_candidate_geometry"
    assert receipt["claim_boundary"]["interior_never_observed"] is True
    assert receipt["claim_boundary"]["matches_real_appliance_interior"] is False


def test_the_authored_interior_is_reachable_through_the_open_door(
    tmp_path: Path,
) -> None:
    receipt = _author(tmp_path)

    exposure = evaluate_interior_exposure(
        replacement_usd_path=receipt["interior_usd_path"],
        support_link_path="/Asset/cabinet",
        task_door_link_path="/Asset/upper_door",
        interior_prim_paths=[row["prim_path"] for row in receipt["parts"]],
        aperture_plane_y_m=0.30,
        aperture_x_interval_m=[-0.25, 0.25],
        aperture_z_interval_m=[0.99, 1.55],
        samples_per_axis=9,
    )

    assert exposure["interior_exposed"] is True


def test_articulation_and_physics_survive(tmp_path: Path) -> None:
    receipt = _author(tmp_path)

    stage = Usd.Stage.Open(receipt["interior_usd_path"])
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]) == 1
    assert len(
        [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    ) == 1
    assert receipt["preserved"]["rigid_body_count"] == 2


def test_shelves_that_would_not_fit_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedInteriorAuthoringError) as excinfo:
        _author(tmp_path, shelf_count=200)

    assert any("shelf_spacing_too_small" in error for error in excinfo.value.errors)


def test_a_cavity_smaller_than_its_walls_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedInteriorAuthoringError) as excinfo:
        _author(tmp_path, wall_thickness_m=0.4)

    assert any("cavity_too_small" in error for error in excinfo.value.errors)


def test_authoring_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    first = _author(tmp_path, destination=tmp_path / "a.usda")
    second = _author(tmp_path, destination=tmp_path / "b.usda")

    assert first["interior_usd_sha256"] == second["interior_usd_sha256"]
    stored = json.loads(Path(first["receipt_path"]).read_text(encoding="utf-8"))
    assert stored == first


def test_door_bins_are_authored_on_the_task_door_when_requested(
    tmp_path: Path,
) -> None:
    receipt = _author(
        tmp_path,
        door_bin_count=2,
        door_link_path="/Asset/upper_door",
        door_bin_y_interval_m=[0.24, 0.30],
    )

    bins = [row for row in receipt["parts"] if row["role"] == "door_bin"]
    assert len(bins) == 2
    assert all(row["prim_path"].startswith("/Asset/upper_door/") for row in bins)
