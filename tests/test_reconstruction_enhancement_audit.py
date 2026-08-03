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
        "fixer",
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
        assert audit["evaluation_evidence_use_permitted"] is False
        assert audit["policy_input_use_permitted"] is False
        assert audit["offline_reconstruction_modification_permitted"] is False
        assert audit["presentation_enhancement_after_inputs_sealed_only"] is True
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


def test_fixer_and_harmonizer_record_commercial_weights_but_stay_rejected() -> None:
    fixer = enhancement_method_audit("fixer")
    assert fixer["source_license"] == "Apache-2.0"
    assert "NVIDIA Open Model License" in fixer["model_license"]
    assert "commercial_use_permitted" in fixer["model_license"]
    assert fixer["status"].startswith("rejected_")
    assert "checkpoint_digest_not_pinned_in_worker" in fixer["blockers"]
    assert "real_heldout_baseline_comparison_not_executed" in fixer["blockers"]

    harmonizer = enhancement_method_audit("harmonizer")
    assert "NVIDIA Open Model License" in harmonizer["model_license"]
    assert "commercial_use_permitted" in harmonizer["model_license"]
    assert harmonizer["status"] == "rejected_pending_checkpoint_runtime_qualification"
    assert "checkpoint_digest_not_pinned_in_worker" in harmonizer["blockers"]
    assert "source_and_dependency_license_receipt_missing" in harmonizer["blockers"]
