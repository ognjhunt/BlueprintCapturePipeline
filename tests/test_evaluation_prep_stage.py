from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.evaluation_prep_stage import run_evaluation_prep_stage


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_capture(tmp_path: Path) -> Path:
    capture_root = tmp_path / "bucket" / "scenes" / "scene_eval" / "captures" / "cap_eval"
    pipeline_root = capture_root / "pipeline"
    raw_root = capture_root / "raw"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene_eval",
            "capture_id": "cap_eval",
            "metadata": {"task_statement": "Open and close the fridge door"},
        },
    )
    _write_json(
        raw_root / "object_index.json",
        {
            "objects": [
                {
                    "id": "1",
                    "label": "refrigerator",
                    "boundingBox": {
                        "center": [1.0, 0.0, 1.0],
                        "extents": [0.8, 0.8, 2.0],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "orientationQuaternion": [1, 0, 0, 0],
                    },
                }
            ]
        },
    )
    _write_json(
        pipeline_root / "opportunity_handoff.json",
        {
            "schema_version": "v1",
            "site_submission_id": "site-sub-1",
            "opportunity_id": "opp-1",
            "qualification_state": "ready",
            "downstream_evaluation_eligibility": True,
            "operator_approved_summary": "Qualified fridge-door opportunity.",
            "scoped_task_definition": {
                "task_id": "task-1",
                "scoped_task_statement": "Open and close the fridge door",
                "success_criteria": ["door opens", "door closes"],
                "in_scope_zone": "kitchen_fridge_zone",
            },
            "site_constraints": {
                "operating_constraints": ["daytime only"],
                "privacy_security_constraints": ["no PII"],
                "known_blockers": ["none"],
            },
        },
    )
    _write_json(
        pipeline_root / "qualification_record.json",
        {"readiness_state": "ready", "confidence": 0.92},
    )
    _write_json(
        pipeline_root / "task_scope_record.json",
        {
            "task_statement": "Open and close the fridge door",
            "target_object_ids": ["1"],
            "articulation_required_ids": ["1"],
            "task_zone": {"center": [1.0, 0.0, 1.0]},
            "success_criteria": ["door opens", "door closes"],
        },
    )
    advanced_dir = pipeline_root / "advanced_geometry"
    advanced_dir.mkdir(parents=True, exist_ok=True)
    for name in ("3dgs_compressed.ply", "labels.json", "structure.json", "task_targets.synthetic.json"):
        (advanced_dir / name).write_text("{}" if name.endswith(".json") else "ply\n", encoding="utf-8")
    _write_json(advanced_dir / "advanced_geometry_bundle.json", {"schema_version": "v1"})
    return capture_root


def test_evaluation_prep_stage_writes_required_contract(tmp_path: Path) -> None:
    capture_root = _build_capture(tmp_path)

    result = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rich_handoff = json.loads((capture_root / "pipeline" / "evaluation_prep" / "qualified_opportunity_handoff.json").read_text(encoding="utf-8"))
    anchors = json.loads((capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((capture_root / "pipeline" / "evaluation_prep" / "evaluation_prep_summary.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "ready_for_validation"
    assert manifest["artifacts"]["qualified_opportunity_handoff"] == "qualified_opportunity_handoff.json"
    assert rich_handoff["qualification_state"] == "ready"
    assert rich_handoff["downstream_evaluation_eligibility"] is True
    assert anchors["tasks"][0]["target_object_ids"] == ["1"]
    assert summary["task_count"] == 1
    assert summary["object_count"] == 1
