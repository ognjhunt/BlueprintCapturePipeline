from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_enhancement_audit import (
    enhancement_method_audit,
    enhancement_method_audits,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_enhancement_candidates_are_explicitly_rejected_and_proof_bounded() -> None:
    audits = enhancement_method_audits()
    assert [audit["method_id"] for audit in audits] == [
        "artifixer",
        "difix3d",
        "harmonizer",
    ]
    for audit in audits:
        assert audit["status"].startswith("rejected_")
        assert audit["blockers"]
        assert audit["baseline_reconstruction_required"] is True
        assert audit["frozen_real_heldout_views_required"] is True
        assert audit["hidden_heldout_available_to_candidate"] is False
        assert audit["unenhanced_baseline_preserved"] is True
        assert audit["generated_pixels_are_captured_evidence"] is False
        assert audit["metric_or_collision_proof_effect"] is False
        assert audit["enhancement_method_audit_digest"] == canonical_digest(
            audit, digest_field="enhancement_method_audit_digest"
        )
        schema = json.loads(
            (
                ROOT
                / "docs/schemas/reconstruction_enhancement_method_audit.v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        jsonschema.validate(audit, schema)


def test_difix_and_artifixer_record_separate_code_and_model_license_status() -> None:
    difix = enhancement_method_audit("difix3d")
    assert "noncommercial" in difix["source_license"]
    artifixer = enhancement_method_audit("artifixer")
    assert artifixer["source_license"] == "Apache-2.0"
    assert "research_and_development_only" in artifixer["model_license"]
