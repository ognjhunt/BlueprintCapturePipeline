from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.registered_replacement_asset import (
    materialize_registered_replacement_asset,
)


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_registered_asset_applies_heading_and_preserves_translation(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(3.0, 4.0, 5.0))
    stage.GetRootLayer().Save()
    binding = tmp_path / "binding.json"
    binding.write_text("{}", encoding="utf-8")
    composition = {
        "schema_version": "simready_agent_cad_visual_composition.v2",
        "status": "agent_cad_visuals_composed",
        "scene_id": "scene",
        "task_id": "task",
        "asset_id": "asset",
        "task_freeze_digest": "sha256:" + "f" * 64,
        "binding": {**_record(binding), "binding_digest": "sha256:" + "b" * 64},
        "output_usd": _record(source),
        "visual_meshes": [{"visual_mesh_path": "/Asset/mesh"}],
        "visual_mesh_count": 1,
        "agent_authored_display_color_mesh_count": 1,
        "neutral_fallback_mesh_count": 0,
        "generated_texture_map_count": 0,
        "collision_visual_isolation_verified": True,
        "claim_boundary": {},
        "receipt_digest": "sha256:" + "c" * 64,
    }
    registration = {
        "scene_id": "scene",
        "task_id": "task",
        "asset_id": "asset",
        "registration_digest": "sha256:" + "d" * 64,
        "T_observed_world_axes_from_asset_local_axes": [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    composition_path = tmp_path / "composition.json"
    registration_path = tmp_path / "registration.json"
    composition_path.write_text(json.dumps(composition), encoding="utf-8")
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.registered_replacement_asset.validate_agent_cad_visual_composition",
        lambda value: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.registered_replacement_asset.validate_replacement_asset_frame_registration",
        lambda value: value,
    )
    result = materialize_registered_replacement_asset(
        visual_composition_receipt_path=composition_path,
        frame_registration_path=registration_path,
        output_usd_path=tmp_path / "registered.usda",
        output_receipt_path=tmp_path / "registered.json",
    )
    reopened = Usd.Stage.Open(result["output_usd"]["path"])
    matrix = UsdGeom.Xformable(reopened.GetDefaultPrim()).GetLocalTransformation()
    assert matrix.ExtractTranslation() == Gf.Vec3d(3.0, 4.0, 5.0)
    assert matrix[0][0] == -1.0
    assert matrix[1][1] == -1.0
    assert result["geometry_generated_or_modified"] is False
    assert result["receipt_digest"] == canonical_digest(result, digest_field="receipt_digest")
