from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import trimesh

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_simready_control import (
    PublicSceneSimReadyControlError,
    materialize_parametric_simready_control,
)


DIAMETER_M = 0.062
HEIGHT_M = 0.169


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    evidence_root = tmp_path / "evidence"
    manifest_path = repo_root / "scene.json"
    component: dict[str, object] = {
        "role": "interiorgs_appearance_scene",
        "scene_mapping": {"publisher_scene_id": "840313"},
        "target_binding": {
            "semantic_label": "canned_beverage",
            "interiorgs_instance_id": "160",
            "obb_aabb_min_m": [-DIAMETER_M / 2.0, -DIAMETER_M / 2.0, 0.0],
            "obb_aabb_max_m": [DIAMETER_M / 2.0, DIAMETER_M / 2.0, HEIGHT_M],
        },
        "manifest_digest": "",
    }
    component["manifest_digest"] = canonical_digest(component, digest_field="manifest_digest")
    _write_json(manifest_path, component)

    generator_path = repo_root / "assets" / "generator.py"
    generator_path.parent.mkdir(parents=True, exist_ok=True)
    generator_path.write_text("# deterministic CAD generator\n", encoding="utf-8")
    step_path = repo_root / "assets" / "control.step"
    step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")

    cad_root = evidence_root / "cad"
    cad_root.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.cylinder(radius=DIAMETER_M * 500.0, height=HEIGHT_M * 1000.0)
    mesh.apply_translation([0.0, 0.0, HEIGHT_M * 500.0])
    mesh.export(cad_root / "control.stl")
    _write_json(
        cad_root / "inspection.json",
        {
            "ok": True,
            "errors": [],
            "tokens": [{"summary": {"kind": "part", "shapeCount": 1}}],
        },
    )
    Image.new("RGB", (32, 32), (0, 255, 0)).save(cad_root / "snapshot.png")
    Image.new("RGB", (32, 32), (0, 128, 0)).save(cad_root / "contact.png")

    request_path = repo_root / "request.json"
    _write_json(
        request_path,
        {
            "schema_version": "adp009a_parametric_simready_request.v1",
            "control_id": "test-control",
            "scene_component_manifest_path": "scene.json",
            "cad_evidence": {
                "generator_path": "assets/generator.py",
                "step_path": "assets/control.step",
                "stl_relative_path": "cad/control.stl",
                "inspection_relative_path": "cad/inspection.json",
                "snapshot_relative_path": "cad/snapshot.png",
                "scene_target_contact_sheet_relative_path": "cad/contact.png",
                "length_unit": "millimeter",
                "mesh_linear_deflection_mm": 0.05,
                "mesh_angular_deflection_rad": 0.05,
                "source_skill": {
                    "repository": "https://github.com/example/cad-skill",
                    "commit": "a" * 40,
                    "tree": "b" * 40,
                    "license": "MIT",
                },
            },
            "grasp_selection": {
                "points_local_m": [
                    [-DIAMETER_M / 2.0, 0.0, HEIGHT_M / 2.0],
                    [DIAMETER_M / 2.0, 0.0, HEIGHT_M / 2.0],
                ],
                "rationale": "Rigid central body.",
                "coordinate_note": "Base-centered local frame in meters.",
            },
            "visual_material": {
                "strategy": "observed_scene_control",
                "diffuse_color": [0.1, 0.7, 0.3],
                "roughness": 0.4,
                "metallic": 0.1,
                "authority": "visual evidence only",
            },
            "mass_kg": 0.355,
            "static_friction": 0.5,
            "dynamic_friction": 0.4,
            "restitution": 0.1,
            "usd_date_generated": "2026-08-04",
        },
    )
    return repo_root, evidence_root, request_path


def test_materializer_derives_mesh_usd_and_prepared_receipt(tmp_path: Path) -> None:
    repo_root, evidence_root, request_path = _fixture(tmp_path)
    output_usda = repo_root / "output" / "control.usda"
    output_receipt = repo_root / "output" / "receipt.json"

    receipt = materialize_parametric_simready_control(
        request_path=request_path,
        repo_root=repo_root,
        evidence_root=evidence_root,
        output_usda=output_usda,
        output_receipt=output_receipt,
    )

    assert receipt["status"] == "prepared_for_independent_validation"
    assert receipt["cad_evidence"]["mesh"]["watertight"] is True
    assert receipt["checks"]["simready_foundation_profile_passed"] is False
    assert receipt["geometry"]["obb_center_m"] == pytest.approx([0.0, 0.0, HEIGHT_M / 2.0])
    assert receipt["geometry"]["nominal_base_placement_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert receipt["geometry"]["world_placement_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert receipt["geometry"]["world_placement_datum"] == "center_of_base_datum"
    assert receipt["usd"]["sha256"].startswith("sha256:")
    authored = output_usda.read_text(encoding="utf-8")
    assert 'def Mesh "body"' in authored
    assert 'physics:approximation = "sdf"' in authored
    assert 'def BasisCurves "grasp_identifier_01"' in authored


def test_materializer_rejects_caller_asserted_qualification(tmp_path: Path) -> None:
    repo_root, evidence_root, request_path = _fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["status"] = "admitted"
    _write_json(request_path, request)

    with pytest.raises(
        PublicSceneSimReadyControlError, match="caller_asserted_qualification_forbidden"
    ):
        materialize_parametric_simready_control(
            request_path=request_path,
            repo_root=repo_root,
            evidence_root=evidence_root,
            output_usda=repo_root / "output.usda",
            output_receipt=repo_root / "receipt.json",
        )


def test_materializer_rejects_mesh_dimension_mismatch(tmp_path: Path) -> None:
    repo_root, evidence_root, request_path = _fixture(tmp_path)
    wrong_mesh = trimesh.creation.cylinder(radius=50.0, height=100.0)
    wrong_mesh.apply_translation([0.0, 0.0, 50.0])
    wrong_mesh.export(evidence_root / "cad" / "control.stl")

    with pytest.raises(PublicSceneSimReadyControlError, match="cad_mesh_dimension_mismatch"):
        materialize_parametric_simready_control(
            request_path=request_path,
            repo_root=repo_root,
            evidence_root=evidence_root,
            output_usda=repo_root / "output.usda",
            output_receipt=repo_root / "receipt.json",
        )


def test_materializer_rejects_cad_evidence_outside_root(tmp_path: Path) -> None:
    repo_root, evidence_root, request_path = _fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["cad_evidence"]["stl_relative_path"] = "../outside.stl"
    _write_json(request_path, request)

    with pytest.raises(PublicSceneSimReadyControlError, match="path_outside_approved_root"):
        materialize_parametric_simready_control(
            request_path=request_path,
            repo_root=repo_root,
            evidence_root=evidence_root,
            output_usda=repo_root / "output.usda",
            output_receipt=repo_root / "receipt.json",
        )
