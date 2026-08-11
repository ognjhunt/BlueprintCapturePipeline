from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from blueprint_pipeline.cad_agent_review_media import (
    CONTACT_SHEET_FILENAME,
    HTML_FILENAME,
    RECEIPT_FILENAME,
    CadAgentReviewMediaError,
    materialize_cad_agent_visual_comparison,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.simready_cad_agent_contract import (
    INSPECTION_SCHEMA_VERSION,
    file_record,
    seal_cad_agent_execution_receipt,
    seal_cad_agent_matrix,
    seal_cad_agent_output,
    seal_cad_agent_reference_manifest,
    seal_cad_agent_request,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_A_FREEZE = (
    ROOT
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_a_freeze.v1.json"
)


def _write_png(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 90), color).save(path)
    return path


def _write(path: Path, payload: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _task_freeze_fixture(root: Path, *, task_id: str, asset_id: str) -> Path:
    freeze = json.loads(TASK_A_FREEZE.read_text(encoding="utf-8"))
    freeze["task_id"] = task_id
    freeze["removal_plan"]["replacement_asset_id"] = asset_id
    freeze["task_freeze_digest"] = canonical_digest(
        freeze, digest_field="task_freeze_digest"
    )
    path = root / f"{task_id}_freeze.v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _backend(root: Path, backend_id: str) -> dict[str, object]:
    commit = "1" * 40
    archive = root / f"{backend_id}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as source:
        source.comment = commit.encode("ascii")
        source.writestr("LICENSE", "MIT\n")
    return {
        "backend_id": backend_id,
        "execution_mode": (
            "codex_skill_step_first"
            if backend_id == "earthtojake_text_to_cad"
            else "codex_agent_direct_repo_route"
        ),
        "agent_authored_geometry": True,
        "deterministic_geometry_generator_used": False,
        "graph_geometry_used_for_cad_authoring": False,
        "deterministic_format_conversion_only": True,
        "repository_url": (
            "https://github.com/earthtojake/text-to-cad"
            if backend_id == "earthtojake_text_to_cad"
            else "https://github.com/Pan-Chera/Multi-Agent-CAD"
        ),
        "commit": commit,
        "tree": "2" * 40,
        "source_archive": file_record(archive),
        "license": "MIT",
        "model_id": "fixture_model",
    }


def _candidate(
    root: Path,
    *,
    backend_id: str,
    replacement_slot: int = 1,
    task_id: str = "task_a_washer_door_open",
    asset_id: str = "840920_simready_washer_candidate",
    task_freeze_path: Path = TASK_A_FREEZE,
    reference_color: tuple[int, int, int] = (120, 10, 10),
) -> dict[str, object]:
    candidate_root = root / task_id / f"slot_{replacement_slot}" / backend_id
    brief = _write(candidate_root / "brief.md", "make fixture CAD\n")
    reference = _write_png(candidate_root / "reference.png", reference_color)
    reference_manifest = seal_cad_agent_reference_manifest(
        scene_id="fixture_scene",
        objects=[
            {
                "replacement_slot": replacement_slot,
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze_path": task_freeze_path,
                "reference_image_paths": [reference],
            }
        ],
    )
    reference_manifest_path = candidate_root / "reference_manifest.v1.json"
    reference_manifest_path.write_text(
        json.dumps(reference_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    request = seal_cad_agent_request(
        request_id=f"fixture-{backend_id}",
        scene_id="fixture_scene",
        task_id=task_id,
        asset_id=asset_id,
        replacement_slot=replacement_slot,
        backend=_backend(root / "sources", backend_id),
        task_freeze_path=task_freeze_path,
        cad_brief_path=brief,
        metric_envelope_mm=[600.112, 604.104004, 847.564026],
        reference_manifest_path=reference_manifest_path,
    )
    generator = _write(candidate_root / "candidate.py", "# agent source\n")
    step = _write(candidate_root / "candidate.step", b"ISO-10303-21;")
    snapshot = _write_png(
        candidate_root / f"{backend_id}_iso.png",
        (20, 110, 180) if backend_id == "earthtojake_text_to_cad" else (20, 160, 90),
    )
    inspection = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "passed",
        "step": file_record(step),
        "measured_envelope_mm": [600.112, 604.104004, 847.564026],
        "measured_center_mm": [0.0, 0.0, 423.782013],
        "topology": {"face_count": 6, "edge_count": 12},
        "inspector": {
            "backend": "build123d_import_step",
            "build123d_version": "fixture",
            "ocp_version": "fixture",
            "python_version": "fixture",
            "module_source": file_record(
                ROOT / "src/blueprint_pipeline/simready_cad_agent_contract.py"
            ),
        },
        "claim_boundary": {
            "step_bytes_reopened": True,
            "metric_envelope_measured": True,
            "visual_quality_qualified": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    inspection["receipt_digest"] = canonical_digest(
        inspection, digest_field="receipt_digest"
    )
    inspection_path = candidate_root / "inspection.v1.json"
    inspection_path.write_text(json.dumps(inspection), encoding="utf-8")
    execution = seal_cad_agent_execution_receipt(
        request=request,
        generator_source_path=generator,
        cad_brief_path=brief,
        output_step_path=step,
        event_rows=[{"event": "agent_authored", "status": "passed"}],
    )
    execution_path = candidate_root / "execution.v1.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    return seal_cad_agent_output(
        request=request,
        generator_source_path=generator,
        step_path=step,
        inspection_receipt_path=inspection_path,
        snapshot_paths=[snapshot],
        execution_receipt_path=execution_path,
        measured_envelope_mm=[600.112, 604.104004, 847.564026],
        actual_cost_usd=0.0,
    )


def _matrix(root: Path, *, mismatch_reference: bool = False) -> Path:
    first = _candidate(root, backend_id="earthtojake_text_to_cad")
    second = _candidate(
        root,
        backend_id="pan_chera_multi_agent_cad",
        reference_color=(10, 120, 10) if mismatch_reference else (120, 10, 10),
    )
    matrix = seal_cad_agent_matrix(
        objects=[
            {
                "replacement_slot": 1,
                "task_id": "task_a_washer_door_open",
                "asset_id": "840920_simready_washer_candidate",
                "candidates": [first, second],
            }
        ]
    )
    path = root / "cad_matrix.v1.json"
    path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    return path


def _five_object_matrix(root: Path) -> Path:
    objects = []
    for slot in range(1, 6):
        task_id = f"task_{slot}_replacement"
        asset_id = f"asset_{slot}"
        task_freeze_path = _task_freeze_fixture(
            root / "freezes", task_id=task_id, asset_id=asset_id
        )
        candidates = [
            _candidate(
                root,
                backend_id="earthtojake_text_to_cad",
                replacement_slot=slot,
                task_id=task_id,
                asset_id=asset_id,
                task_freeze_path=task_freeze_path,
                reference_color=(100 + slot, 10, 10),
            ),
            _candidate(
                root,
                backend_id="pan_chera_multi_agent_cad",
                replacement_slot=slot,
                task_id=task_id,
                asset_id=asset_id,
                task_freeze_path=task_freeze_path,
                reference_color=(100 + slot, 10, 10),
            ),
        ]
        objects.append(
            {
                "replacement_slot": slot,
                "task_id": task_id,
                "asset_id": asset_id,
                "candidates": candidates,
            }
        )
    matrix = seal_cad_agent_matrix(objects=objects)
    path = root / "cad_matrix_five_objects.v1.json"
    path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    return path


def test_visual_comparison_materializes_manifest_bound_contact_sheet(tmp_path: Path) -> None:
    matrix_path = _matrix(tmp_path)
    receipt = materialize_cad_agent_visual_comparison(
        matrix_path=matrix_path,
        output_dir=tmp_path / "review",
        title="Fixture CAD comparison",
    )

    assert receipt["schema_version"] == "scene_replacement_cad_agent_visual_comparison.v1"
    assert receipt["object_count"] == 1
    assert receipt["maximum_replacement_objects"] == 5
    assert receipt["replacement_object_capacity"] == {
        "minimum": 1,
        "maximum": 5,
        "sealed_slots": 1,
    }
    assert receipt["backend_ids"] == [
        "earthtojake_text_to_cad",
        "pan_chera_multi_agent_cad",
    ]
    assert receipt["rows"][0]["reference_signature"].startswith("sha256:")
    assert receipt["rows"][0]["candidates"][0]["backend_id"] == "earthtojake_text_to_cad"
    assert receipt["claim_boundary"]["simready_qualified"] is False
    assert (tmp_path / "review" / CONTACT_SHEET_FILENAME).is_file()
    assert (tmp_path / "review" / HTML_FILENAME).is_file()
    assert (tmp_path / "review" / RECEIPT_FILENAME).is_file()


def test_visual_comparison_materializes_five_object_contact_sheet(
    tmp_path: Path,
) -> None:
    matrix_path = _five_object_matrix(tmp_path)
    receipt = materialize_cad_agent_visual_comparison(
        matrix_path=matrix_path,
        output_dir=tmp_path / "review",
        title="Five object CAD comparison",
    )

    assert receipt["object_count"] == 5
    assert receipt["replacement_object_capacity"] == {
        "minimum": 1,
        "maximum": 5,
        "sealed_slots": 5,
    }
    assert [row["replacement_slot"] for row in receipt["rows"]] == [1, 2, 3, 4, 5]
    assert {row["task_id"] for row in receipt["rows"]} == {
        f"task_{slot}_replacement" for slot in range(1, 6)
    }
    assert all(len(row["candidates"]) == 2 for row in receipt["rows"])
    assert (tmp_path / "review" / CONTACT_SHEET_FILENAME).is_file()
    assert (tmp_path / "review" / HTML_FILENAME).is_file()
    assert (tmp_path / "review" / RECEIPT_FILENAME).is_file()


def test_visual_comparison_rejects_backend_reference_mismatch(tmp_path: Path) -> None:
    matrix_path = _matrix(tmp_path, mismatch_reference=True)

    with pytest.raises(
        CadAgentReviewMediaError, match="cad_review_candidate_reference_mismatch"
    ):
        materialize_cad_agent_visual_comparison(
            matrix_path=matrix_path,
            output_dir=tmp_path / "review",
        )


def test_results_doc_points_to_manifest_driven_visual_comparison() -> None:
    results = (
        ROOT
        / "docs/arm_decision_proof_v1/THIRD_SCENE_DUAL_TASK_E2E_RESULTS.md"
    ).read_text(encoding="utf-8")

    assert "cad_agent_visual_comparison.v1.json" in results
    assert "third_scene_840920_dual_task_cad_agent_visual_comparison_binding.v1.json" in results
    assert "sha256:c0b6784847d12259bc603e4ba1b47004505f7ebb04acc6f8997d4b649f56c8ad" in results
    assert "sha256:02cb4abbf136c7156105e198e4a8ca9c027b1b72967aa698f8c4f744932fc7b3" in results
    assert (
        "Dockerless static Content Agents bundle/config/input-USD preflight with "
        "local Docker dry-run retained as an explicit blocker when unavailable"
    ) in results
    assert (
        "all four bundles passed local/static preflight"
    ) not in results
    assert "third_scene_840920_agent_cad_visual_comparison_receipt.v1.json" not in results
    assert "sha256:05febab49762d604f047df6a7f8975b25f50b6ac432493566feecb90e90f020c" not in results
