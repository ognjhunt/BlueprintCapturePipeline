from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_passive_destination_cad_agent import (
    materialize_passive_destination_cad_request,
)
from blueprint_pipeline.task_evaluation_passive_destination_simready import (
    INTENDED_SUPPORT_PRIM,
    materialize_passive_destination_simready,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_exact_step_visual_gets_five_colliders_and_static_qualification(
    tmp_path: Path,
) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    request_path = tmp_path / "request.json"
    request = materialize_passive_destination_cad_request(
        run_id="fixture-passive-destination",
        expected_production_commit=commit,
        destination_identity={"id": "document-tray", "version": "v1"},
        visible_label="blue document tray",
        dimensions_m={
            "outer_x": 0.33,
            "outer_y": 0.48,
            "base_thickness": 0.005,
            "wall_thickness": 0.005,
            "wall_height_above_base": 0.02,
            "minimum_interior_x": 0.32,
            "minimum_interior_y": 0.47,
            "minimum_interior_z": 0.018,
        },
        output_path=request_path,
    )
    step = _write(tmp_path / "tray.step", "fixture-step")
    inspection = _write(tmp_path / "inspection.json", "{}")
    generator = _write(tmp_path / "generator.py", "def gen_step(): pass\n")
    invocation = _write(tmp_path / "invocation.json", "{}")
    cad = {
        "schema_version": "task_evaluation_passive_destination_cad_result.v1",
        "status": "candidate_authored_pending_visual_static_native_qualification",
        "request": _record(request_path),
        "request_digest": request["request_digest"],
        "artifacts": {
            "step": _record(step),
            "inspection": _record(inspection),
            "generator_source": _record(generator),
            "agent_invocation": _record(invocation),
        },
        "result_digest": "",
    }
    cad["result_digest"] = canonical_digest(cad, digest_field="result_digest")
    cad_path = tmp_path / "cad-result.json"
    cad_path.write_text(json.dumps(cad) + "\n")
    visual = tmp_path / "visual.usda"
    stage = Usd.Stage.CreateNew(str(visual))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/links/tray/geometry/tray")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(-0.1, -0.1, 0), Gf.Vec3f(0.1, -0.1, 0), Gf.Vec3f(0, 0.1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()
    projection = {
        "schema_version": "cad_agent_mesh_usd_projection.v1",
        "status": "mesh_working_copy_authored",
        "step": _record(step),
        "output_usd": _record(visual),
        "mesh_prim_paths": ["/Asset/links/tray/geometry/tray"],
        "receipt_digest": "",
    }
    projection["receipt_digest"] = canonical_digest(
        projection, digest_field="receipt_digest"
    )
    projection_path = tmp_path / "projection.json"
    projection_path.write_text(json.dumps(projection) + "\n")

    result = materialize_passive_destination_simready(
        cad_result_path=cad_path,
        projection_receipt_path=projection_path,
        output_root=tmp_path / "output",
    )

    assert result["status"] == "static_qualified_pending_native_import_and_placement"
    assert result["intended_support_prim_paths"] == ["/Asset"]
    assert result["intended_support_collision_prim_paths"] == [INTENDED_SUPPORT_PRIM]
    reopened = Usd.Stage.Open(result["asset"]["path"])
    assert reopened.GetPrimAtPath("/Asset").HasAPI(UsdPhysics.RigidBodyAPI)
    colliders = [p for p in reopened.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 5
    for collider in colliders:
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(
            materialPurpose="physics"
        )
        assert str(material.GetPath()) == "/Asset/Materials/Physics"
    assert result["interior_bounds_body_frame_m"]["minimum"] == [-0.16, -0.235, 0.005]
    assert result["static_qualification"]["sha256"].startswith("sha256:")
