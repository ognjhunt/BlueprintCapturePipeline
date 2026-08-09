from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_interior_exposure import (
    INTERIOR_EXPOSURE_SCHEMA_VERSION,
    ArticulatedInteriorExposureError,
    evaluate_interior_exposure,
)


def _box(stage, path, xmin, xmax, ymin, ymax, zmin, zmax):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(x, y, z)
            for x in (xmin, xmax)
            for y in (ymin, ymax)
            for z in (zmin, zmax)
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
    )
    return mesh


def _asset(path: Path, *, sealed_shell: bool) -> Path:
    """A cabinet with an inset interior, optionally walled off at the front."""

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    for link in ("cabinet", "upper_door"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Asset/{link}").GetPrim())
    if sealed_shell:
        # a solid carcass: front face at y = 0.18 spans the whole aperture
        _box(stage, "/Asset/cabinet/shell", -0.35, 0.35, -0.35, 0.18, 0.0, 1.63)
    else:
        # the same carcass with the upper aperture cut away: side walls only
        _box(stage, "/Asset/cabinet/shell_left", -0.35, -0.30, -0.35, 0.18, 0.0, 1.63)
        _box(stage, "/Asset/cabinet/shell_right", 0.30, 0.35, -0.35, 0.18, 0.0, 1.63)
        _box(stage, "/Asset/cabinet/shell_back", -0.35, 0.35, -0.35, -0.30, 0.0, 1.63)
        _box(stage, "/Asset/cabinet/shell_lower", -0.35, 0.35, -0.35, 0.18, 0.0, 0.93)
    _box(stage, "/Asset/cabinet/generated_interior", -0.31, 0.31, -0.31, 0.14, 0.04, 1.59)
    _box(stage, "/Asset/upper_door/panel", -0.35, 0.35, 0.30, 0.35, 0.94, 1.63)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/upper_door_hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/cabinet")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/upper_door")])
    joint.CreateAxisAttr().Set("Z")
    stage.GetRootLayer().Save()
    return path


def _evaluate(path: Path, **overrides):
    arguments = {
        "replacement_usd_path": path,
        "support_link_path": "/Asset/cabinet",
        "task_door_link_path": "/Asset/upper_door",
        "interior_prim_paths": ["/Asset/cabinet/generated_interior"],
        "aperture_plane_y_m": 0.30,
        "aperture_x_interval_m": [-0.30, 0.30],
        "aperture_z_interval_m": [0.95, 1.58],
        "samples_per_axis": 9,
        "minimum_exposed_fraction": 0.75,
    }
    arguments.update(overrides)
    return evaluate_interior_exposure(**arguments)


def test_a_sealed_carcass_is_reported_as_hiding_its_own_interior(
    tmp_path: Path,
) -> None:
    """Opening the door must reveal the cavity, not a flat wall behind it."""

    asset = _asset(tmp_path / "sealed.usda", sealed_shell=True)

    receipt = _evaluate(asset)

    assert receipt["schema_version"] == INTERIOR_EXPOSURE_SCHEMA_VERSION
    assert receipt["status"] == "interior_not_exposed"
    assert receipt["interior_exposed"] is False
    assert receipt["exposed_fraction"] == 0.0
    assert "/Asset/cabinet/shell" in receipt["occluding_prim_paths"]
    assert receipt["blockers"] == ["articulated_interior_occluded_by_support_link"]


def test_an_open_carcass_exposes_the_interior(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "open.usda", sealed_shell=False)

    receipt = _evaluate(asset)

    assert receipt["status"] == "interior_exposed"
    assert receipt["interior_exposed"] is True
    assert receipt["exposed_fraction"] >= 0.75
    assert receipt["occluding_prim_paths"] == []
    assert receipt["blockers"] == []


def test_receipt_records_the_sampling_it_actually_did(tmp_path: Path) -> None:
    receipt = _evaluate(_asset(tmp_path / "open.usda", sealed_shell=False))

    assert receipt["samples"]["total"] == 81
    assert receipt["samples"]["hit_interior"] + receipt["samples"]["hit_support"] + (
        receipt["samples"]["hit_nothing"]
    ) == 81
    assert receipt["aperture"]["plane_y_m"] == 0.30
    assert receipt["receipt_digest"].startswith("sha256:")


def test_missing_interior_prim_fails_closed(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "open.usda", sealed_shell=False)

    with pytest.raises(ArticulatedInteriorExposureError) as excinfo:
        _evaluate(asset, interior_prim_paths=["/Asset/cabinet/nope"])

    assert any("interior_prim_missing" in error for error in excinfo.value.errors)


def test_missing_support_link_fails_closed(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "open.usda", sealed_shell=False)

    with pytest.raises(ArticulatedInteriorExposureError) as excinfo:
        _evaluate(asset, support_link_path="/Asset/nope")

    assert any("support_link_missing" in error for error in excinfo.value.errors)


def test_invalid_aperture_fails_closed(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "open.usda", sealed_shell=False)

    with pytest.raises(ArticulatedInteriorExposureError) as excinfo:
        _evaluate(asset, aperture_z_interval_m=[1.6, 0.9])

    assert any("aperture_interval_invalid" in error for error in excinfo.value.errors)


def test_evaluation_is_deterministic(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "open.usda", sealed_shell=False)

    first = _evaluate(asset)
    second = _evaluate(asset)

    assert first["receipt_digest"] == second["receipt_digest"]
    assert first["samples"] == second["samples"]


def test_receipt_round_trips(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "open.usda", sealed_shell=False)

    receipt = _evaluate(asset, destination=tmp_path / "exposure.json")

    stored = json.loads((tmp_path / "exposure.json").read_text(encoding="utf-8"))
    assert stored == receipt
