from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from blueprint_pipeline.cad_agent_mesh_projection import (
    PACKET_SCHEMA_VERSION,
    CadAgentMeshProjectionError,
    extract_step_mesh_packet,
    materialize_mesh_usd_projection,
    validate_step_mesh_packet,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _packet(tmp_path: Path) -> tuple[Path, dict]:
    step = tmp_path / "agent-authored.step"
    step.write_bytes(b"exact-agent-authored-step")
    value = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "geometry_authority": "exact_agent_authored_step",
        "deterministic_geometry_generator_used": False,
        "conversion_only": True,
        "step": _record(step),
        "linear_tolerance_mm": 0.2,
        "angular_tolerance_rad": 0.1,
        "mesh_count": 2,
        "meshes": [
            {
                "prim_path": "/Asset/links/body/geometry/shell",
                "link_id": "body",
                "solid_id": "shell",
                "assembly_transform_applied": True,
                "points_mm": [
                    [0.0, 0.0, 0.0],
                    [100.0, 0.0, 0.0],
                    [0.0, 100.0, 0.0],
                ],
                "triangles": [[0, 1, 2]],
            },
            {
                "prim_path": "/Asset/links/door/geometry/rim",
                "link_id": "door",
                "solid_id": "rim",
                "assembly_transform_applied": True,
                "points_mm": [
                    [0.0, 0.0, 10.0],
                    [50.0, 0.0, 10.0],
                    [0.0, 50.0, 10.0],
                ],
                "triangles": [[0, 1, 2]],
            },
        ],
        "claim_boundary": {
            "cad_authored_by_projection": False,
            "appearance_working_copy_only": True,
            "collision_authority": False,
            "physics_authority": False,
            "simready_qualified": False,
        },
    }
    value["packet_digest"] = canonical_digest(value, digest_field="packet_digest")
    packet = tmp_path / "mesh-packet.json"
    packet.write_text(json.dumps(value), encoding="utf-8")
    return packet, value


def test_mesh_projection_preserves_agent_step_authority_without_physics(
    tmp_path: Path,
) -> None:
    packet, value = _packet(tmp_path)
    output = tmp_path / "agent-input.usda"
    receipt = materialize_mesh_usd_projection(
        packet_path=packet,
        output_usd_path=output,
    )

    assert receipt["step"] == value["step"]
    assert receipt["mesh_count"] == 2
    assert receipt["canonical_simulator_asset"] is False
    stage = Usd.Stage.Open(str(output))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetDefaultPrim().GetPath().pathString == "/Asset"
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/links/body/geometry/shell"))
    assert mesh
    assert list(mesh.GetPointsAttr().Get())[1][0] == pytest.approx(0.1)
    assert UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()[0]
    assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert (
        mesh.GetPrim().GetCustomDataByKey("blueprint:geometryAuthority")
        == "exact_agent_authored_step"
    )


def test_packet_rejects_deterministic_geometry_generator_claim(tmp_path: Path) -> None:
    _, value = _packet(tmp_path)
    value["deterministic_geometry_generator_used"] = True
    value["packet_digest"] = canonical_digest(value, digest_field="packet_digest")
    with pytest.raises(
        CadAgentMeshProjectionError,
        match="cad_agent_mesh_packet_generator_claim_invalid",
    ):
        validate_step_mesh_packet(value)


def test_packet_rejects_triangle_index_outside_exact_vertices(tmp_path: Path) -> None:
    _, value = _packet(tmp_path)
    value["meshes"][0]["triangles"] = [[0, 1, 3]]
    value["packet_digest"] = canonical_digest(value, digest_field="packet_digest")
    with pytest.raises(
        CadAgentMeshProjectionError,
        match="cad_agent_mesh_packet_row_invalid",
    ):
        validate_step_mesh_packet(value)


def test_step_extraction_applies_nonidentity_parent_assembly_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Location:
        def __init__(self, x: float):
            self.x = x

        def __mul__(self, other: "Location") -> "Location":
            return Location(self.x + other.x)

    class Vector:
        def __init__(self, x: float, y: float, z: float):
            self.X, self.Y, self.Z = x, y, z

    class Solid:
        label = "panel"
        children: list = []
        location = Location(2.0)

        def __init__(self, offset: float = 2.0):
            self.offset = offset

        def solids(self):
            return [self]

        def moved(self, location: Location):
            return Solid(self.offset + location.x)

        def tessellate(self, _linear: float, _angular: float):
            return (
                [
                    Vector(self.offset, 0.0, 0.0),
                    Vector(self.offset + 1.0, 0.0, 0.0),
                    Vector(self.offset, 1.0, 0.0),
                ],
                [(0, 1, 2)],
            )

    leaf = Solid()
    link = types.SimpleNamespace(
        label="display",
        location=Location(10.0),
        children=[leaf],
    )
    assembly = types.SimpleNamespace(
        location=Location(0.0),
        children=[link],
    )
    fake_build123d = types.ModuleType("build123d")
    fake_build123d.import_step = lambda _path: assembly
    monkeypatch.setitem(sys.modules, "build123d", fake_build123d)
    step = tmp_path / "assembly.step"
    step.write_bytes(b"assembly")

    packet = extract_step_mesh_packet(
        step_path=step,
        output_path=tmp_path / "packet.json",
    )
    assert packet["meshes"][0]["points_mm"][0] == [12.0, 0.0, 0.0]
    assert packet["meshes"][0]["assembly_transform_applied"] is True


def _packet_with_authored_colors(tmp_path: Path) -> Path:
    packet_path, value = _packet(tmp_path)
    for index, row in enumerate(value["meshes"]):
        row["agent_authored_display_color_rgba"] = [0.2 + 0.1 * index, 0.4, 0.6, 1.0]
    value.pop("packet_digest", None)
    value["packet_digest"] = canonical_digest(value, digest_field="packet_digest")
    packet_path.write_text(json.dumps(value), encoding="utf-8")
    return packet_path


def test_default_material_resolves_on_the_saved_stage_when_every_mesh_has_a_color(
    tmp_path: Path,
) -> None:
    """The receipt must name a material the shipped USD actually contains.

    Production regression: `default_material_path` was evaluated after
    `Save()`, and `material_for` defines its material lazily -- so whenever
    every mesh carried an authored colour (the intended case, the one the
    receipt reports as `agent_authored_display_colors_preserved`) the receipt
    named `/Asset/materials/neutral_fallback`, a prim invented after the file
    was written and bound to no mesh. The texture agent resolves
    `material_textures` against a material's alias paths and then its name, so
    it matched neither, planned zero jobs, and the run was rejected after the
    GPU was already rented.
    """

    packet = _packet_with_authored_colors(tmp_path)
    output = tmp_path / "agent-input.usda"

    receipt = materialize_mesh_usd_projection(packet_path=packet, output_usd_path=output)

    assert receipt["agent_authored_display_colors_preserved"] is True
    assert receipt["neutral_fallback_mesh_count"] == 0

    stage = Usd.Stage.Open(str(output))
    declared = receipt["default_material_path"]
    assert stage.GetPrimAtPath(declared).IsValid(), declared

    bound = {
        str(
            UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path))
            .ComputeBoundMaterial()[0]
            .GetPath()
        )
        for prim_path in receipt["mesh_prim_paths"]
    }
    assert declared in bound, (declared, sorted(bound))
