from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from blueprint_pipeline.agent_enrichment_acceptance import (
    AGENT_ENRICHMENT_ACCEPTANCE_SCHEMA_VERSION,
    AgentEnrichmentAcceptanceError,
    accept_agent_enriched_asset,
)


UPPER = (0.94, 1.60)
MASSES = {"/Asset/cabinet": 62.0, "/Asset/upper_door": 11.0, "/Asset/lower_door": 11.0}


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
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    counts, indices = [], []
    for quad in quads:
        counts.extend([3, 3])
        indices.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def _asset(path: Path, *, masses=None, render_material=True, seal=False) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    for link in ("cabinet", "upper_door", "lower_door"):
        prim = UsdGeom.Xform.Define(stage, f"/Asset/{link}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(
            (masses or MASSES)[f"/Asset/{link}"]
        )
    if seal:
        _box(stage, "/Asset/cabinet/shell", -0.35, 0.35, -0.35, 0.18, 0.0, 1.63)
    else:
        _box(stage, "/Asset/cabinet/shell_left", -0.35, -0.30, -0.35, 0.18, 0.0, 1.63)
        _box(stage, "/Asset/cabinet/shell_back", -0.35, 0.35, -0.35, -0.30, 0.0, 1.63)
    _box(stage, "/Asset/cabinet/generated_interior", -0.31, 0.31, -0.31, 0.14, 0.04, 1.59)
    _box(stage, "/Asset/upper_door/panel", -0.35, 0.35, 0.30, 0.35, UPPER[0], UPPER[1])
    _box(stage, "/Asset/lower_door/panel", -0.35, 0.35, 0.30, 0.35, 0.0, UPPER[0])
    for name, body in (("upper_door_hinge", "upper_door"), ("lower_door_hinge", "lower_door")):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/joints/{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/cabinet")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Asset/{body}")])
        joint.CreateAxisAttr().Set("Z")
        joint.CreateLowerLimitAttr().Set(0.0)
        joint.CreateUpperLimitAttr().Set(90.0)
    if render_material:
        material = UsdShade.Material.Define(stage, "/Asset/Looks/Render/door_shell")
        shader = UsdShade.Shader.Define(stage, "/Asset/Looks/Render/door_shell/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.81, 0.78, 0.76)
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(
            stage.GetPrimAtPath("/Asset/upper_door/panel")
        ).Bind(material)
    stage.GetRootLayer().Save()
    return path


def _accept(tmp_path: Path, enriched: Path, **overrides):
    arguments = {
        "baseline_usd_path": _asset(tmp_path / "baseline.usda"),
        "enriched_usd_path": enriched,
        "support_link_path": "/Asset/cabinet",
        "task_door_link_path": "/Asset/upper_door",
        "interior_prim_paths": ["/Asset/cabinet/generated_interior"],
        "aperture_plane_y_m": 0.30,
        "aperture_x_interval_m": [-0.28, 0.28],
        "aperture_z_interval_m": [0.97, 1.56],
        "required_render_material_prim_paths": ["/Asset/upper_door/panel"],
        "samples_per_axis": 7,
    }
    arguments.update(overrides)
    return accept_agent_enriched_asset(**arguments)


def test_an_enrichment_that_changes_nothing_blueprint_owns_is_accepted(
    tmp_path: Path,
) -> None:
    receipt = _accept(tmp_path, _asset(tmp_path / "enriched.usda"))

    assert receipt["schema_version"] == AGENT_ENRICHMENT_ACCEPTANCE_SCHEMA_VERSION
    assert receipt["status"] == "agent_enrichment_accepted"
    assert receipt["accepted"] is True
    assert receipt["blockers"] == []
    assert receipt["checks"]["articulation_preserved"] is True
    assert receipt["checks"]["authored_link_masses_unchanged"] is True
    assert receipt["checks"]["interior_still_exposed"] is True
    assert receipt["checks"]["render_materials_still_bound"] is True
    assert receipt["receipt_digest"].startswith("sha256:")


def test_an_agent_that_moved_an_authored_link_mass_is_rejected(
    tmp_path: Path,
) -> None:
    """Agents may add priors; they may not overwrite what Blueprint authored."""

    tampered = dict(MASSES)
    tampered["/Asset/upper_door"] = 4.0

    receipt = _accept(tmp_path, _asset(tmp_path / "enriched.usda", masses=tampered))

    assert receipt["accepted"] is False
    assert "agent_enrichment_authored_link_mass_changed" in receipt["blockers"]
    assert receipt["checks"]["authored_link_masses_unchanged"] is False
    assert receipt["link_masses"]["/Asset/upper_door"] == {
        "baseline_kg": 11.0,
        "enriched_kg": 4.0,
    }


def test_an_agent_that_dropped_a_joint_is_rejected(tmp_path: Path) -> None:
    enriched = _asset(tmp_path / "enriched.usda")
    stage = Usd.Stage.Open(str(enriched))
    stage.RemovePrim("/Asset/joints/lower_door_hinge")
    stage.GetRootLayer().Save()

    receipt = _accept(tmp_path, enriched)

    assert receipt["accepted"] is False
    assert "agent_enrichment_articulation_changed" in receipt["blockers"]


def test_an_agent_that_sealed_the_interior_is_rejected(tmp_path: Path) -> None:
    receipt = _accept(tmp_path, _asset(tmp_path / "enriched.usda", seal=True))

    assert receipt["accepted"] is False
    assert "agent_enrichment_interior_no_longer_exposed" in receipt["blockers"]


def test_an_agent_that_unbound_the_render_material_is_rejected(
    tmp_path: Path,
) -> None:
    receipt = _accept(
        tmp_path, _asset(tmp_path / "enriched.usda", render_material=False)
    )

    assert receipt["accepted"] is False
    assert "agent_enrichment_render_material_unbound" in receipt["blockers"]


def test_a_missing_enriched_asset_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(AgentEnrichmentAcceptanceError) as excinfo:
        _accept(tmp_path, tmp_path / "nope.usda")

    assert any("enriched_asset_missing" in error for error in excinfo.value.errors)


def test_acceptance_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    enriched = _asset(tmp_path / "enriched.usda")

    first = _accept(tmp_path, enriched, destination=tmp_path / "a.json")
    second = _accept(tmp_path, enriched, destination=tmp_path / "b.json")

    assert first["receipt_digest"] == second["receipt_digest"]
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == first
