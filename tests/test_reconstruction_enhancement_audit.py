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
    assert artifixer["source_tree"] == "f9283bfe5e3a6cc160fd418f4e66412746a19a07"
    assert artifixer["model_revision"] == "f96352ad72c84a628d5844b6543e94ae8c4479b3"
    lighter = artifixer["release_checkpoints"]["artifixer_1_3b_v1"]
    assert lighter["size_bytes"] == 6_715_346_651
    assert lighter["sha256"] == (
        "sha256:23e909fb4232c6a74a1c59eaf0ebfd419dd188e601aa0ab0145b9aaea821e059"
    )
    assert lighter["base_model_revision"] == (
        "0fad780a534b6463e45facd96134c9f345acfa5b"
    )
    assert artifixer["official_cuda12_base_image"]["linux_amd64_digest"] == (
        "sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb"
    )
    assert artifixer["modes"][-1] == (
        "artifixer3d_plus_postprocess_over_distilled_renders"
    )


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
