from __future__ import annotations

import logging
from pathlib import Path

import pytest

from blueprint_pipeline import first_gpu_candidate_audit as candidate_audit
from blueprint_pipeline import launch_bundle, proof_contracts
from blueprint_pipeline.core import logging_utils


def test_launch_bundle_edge_branches_use_descriptor_and_task_fallbacks() -> None:
    assert launch_bundle._string_list("one") == ["one"]
    assert launch_bundle._string_list(123) == []

    trust_score = launch_bundle.build_buyer_trust_score(
        descriptor={"quality": {"pose_match_rate": 0.9}},
        qualification_record={"confidence": 0.9},
        scorecard={"completeness_status": "sufficient"},
        metadata={
            "capture_rights": {
                "consent_status": "documented",
                "permission_document_uri": "gs://bucket/permission.pdf",
            }
        },
        provider_status="not_requested",
        fidelity_review={},
    )
    assert "multimodal capture review is incomplete" in trust_score["reasons"]
    low_review_score = launch_bundle.build_buyer_trust_score(
        descriptor={"quality": {"pose_match_rate": 0.9}},
        qualification_record={"confidence": 0.9},
        scorecard={"completeness_status": "sufficient"},
        metadata={"consent_status": "documented", "permission_document_uri": "gs://bucket/ok"},
        provider_status="complete",
        fidelity_review={
            "status": "succeeded",
            "scores": {"coverage": 0.1, "world_model_fitness": 0.1},
        },
    )
    assert "Gemini review found coverage gaps in the capture" in low_review_score["reasons"]
    assert "Gemini review found limited world-model fitness" in low_review_score["reasons"]

    bundle = launch_bundle.build_launch_qualification_bundle(
        descriptor={
            "metadata": {
                "capture_rights": {
                    "consent_status": "policy_only",
                    "permission_document_uri": "gs://bucket/permission.pdf",
                }
            }
        },
        qualification_record={"readiness_state": "ready", "confidence": 0.9, "risks": []},
        scorecard={"missing_evidence": []},
        readiness_decision={"missing_evidence": ""},
        site_intake={
            "task_context": {
                "task_statement": "Inspect pallet slots",
                "facility_template": "warehouse",
            }
        },
        buyer_trust_score={"score": 85, "band": "high"},
        provider_run={"status": "not_requested"},
    )
    assert bundle["rights_and_compliance_summary"]["consent_status"] == "policy_only"
    assert bundle["qualification_summary"]["task_statement"] == "Inspect pallet slots"


def test_first_gpu_candidate_audit_empty_discovery_and_ready_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert candidate_audit._string(None) == ""
    assert candidate_audit._capture_root_from_completion_marker(tmp_path / "wrong.json") is None
    assert candidate_audit._discover_capture_roots([tmp_path / "missing"]) == []
    assert candidate_audit._raw_videos(tmp_path / "capture") == []

    blocked = candidate_audit.build_first_gpu_candidate_audit(
        output_path=tmp_path / "blocked.json"
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["no_capture_roots_found"]

    monkeypatch.setattr(
        candidate_audit,
        "build_first_gpu_candidate_audit",
        lambda **_kwargs: {
            "status": "ready_candidate_found",
            "candidate_count": 1,
            "ready_candidate_count": 1,
            "blockers": [],
            "output_path": str(tmp_path / "ready.json"),
        },
    )
    assert candidate_audit.main(["--output", str(tmp_path / "ready.json")]) == 0
    assert "ready_candidates=1" in capsys.readouterr().out


def test_logging_utils_sanitizes_sensitive_and_structured_fields(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    assert logging_utils._redacted_marker("") == "<redacted:sensitive>"
    assert logging_utils._sanitize_value("api_key", "secret") == "<redacted:api-key>"
    assert logging_utils._sanitize_value("path", tmp_path / "artifact.json") == str(
        tmp_path / "artifact.json"
    )
    assert (
        logging_utils._sanitize_value("url", "https://x.test/file?X-Goog-Signature=abc")
        == "<redacted:signed-url>"
    )
    assert logging_utils._sanitize_value("nested", {"token": "secret"}) == {
        "token": "<redacted:token>"
    }
    assert logging_utils._sanitize_value("items", [{"password": "secret"}]) == [
        {"password": "<redacted:password>"}
    ]
    assert logging_utils._sanitize_value("count", 3) == 3
    assert logging_utils._format_fields({}) == ""

    logger = logging.getLogger("blueprint_pipeline.test_logging_utils")
    with caplog.at_level(logging.INFO, logger=logger.name):
        logging_utils.log_event(
            logger,
            logging.INFO,
            "coverage.event\r\nforged-event",
            message="coverage message\nforged-message",
            api_key="secret",
            artifact_path=tmp_path / "artifact.json",
            detail="safe detail\nforged-field\r",
            **{"not-an-identifier": "kept"},
        )

    record = caplog.records[-1]
    assert record.blueprint_event == "coverage.event\\r\\nforged-event"
    assert "\n" not in record.getMessage()
    assert "coverage message\\nforged-message" in record.getMessage()
    assert record.blueprint_fields["api_key"] == "<redacted:api-key>"
    assert record.blueprint_fields["detail"] == "safe detail\\nforged-field\\r"
    assert "detail='safe detail\\\\nforged-field\\\\r'" in record.getMessage()
    assert record.artifact_path == str(tmp_path / "artifact.json")
    assert not hasattr(record, "not-an-identifier")


def test_proof_contracts_review_states_and_missing_hosted_inputs() -> None:
    assert proof_contracts._string_list("alpha") == ["alpha"]
    assert proof_contracts._string_list(123) == []

    review = proof_contracts.build_rights_provenance_review(
        rights_summary={
            "consent_status": "unknown",
            "derived_scene_generation_allowed": True,
        },
        privacy_processing={"status": "no_people_detected"},
        provenance_summary={"status": "missing", "record": {"canonical_truth": False}},
        site_identity={},
        adjacent_systems=["wms", ""],
    )
    assert review["status"] == "needs_review"
    assert "rights_or_consent_requires_review" in review["blockers"]
    assert "provenance_not_grounded" in review["blockers"]
    assert review["site_labeling"]["adjacent_context_included"] is True
    blocked_review = proof_contracts.build_rights_provenance_review(
        rights_summary={"derived_scene_generation_allowed": False},
        privacy_processing={"status": "failed_closed"},
        provenance_summary={
            "status": "grounded",
            "record": {"canonical_truth": True, "grounding_level": "capture_backed"},
        },
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
    )
    assert blocked_review["status"] == "blocked"
    assert "rights_not_sufficient_for_derived_generation" in blocked_review["blockers"]
    assert "privacy_processing_failed_closed" in blocked_review["blockers"]
    privacy_review = proof_contracts.build_rights_provenance_review(
        rights_summary={
            "consent_status": "documented",
            "derived_scene_generation_allowed": True,
        },
        privacy_processing={"status": "not_run"},
        provenance_summary={
            "status": "grounded",
            "record": {"canonical_truth": True, "grounding_level": "capture_backed"},
        },
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
    )
    assert "privacy_processing_incomplete" in privacy_review["blockers"]

    package = proof_contracts.build_site_package_manifest(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id=None,
        opportunity_id=None,
        evaluation_prep_manifest={"canonical_package_status": "registration_blocked"},
        site_world_spec={},
        site_world_registration={},
        site_world_health={"launchable": False},
        launchable_export_bundle={"status": "missing"},
        site_identity={},
        adjacent_systems=[],
        rights_review={"status": "blocked"},
    )
    assert package["status"] == "blocked"
    assert "site_world_spec_missing" in package["blockers"]

    hosted = proof_contracts.build_hosted_review_readiness(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id=None,
        opportunity_id=None,
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
        preview_manifest_uri=None,
        worldlabs_launch_url=None,
        runtime_demo_manifest_uri=None,
        demo_readiness_state="blocked",
        demo_blockers=["demo_missing"],
        site_world_health={"launchable": False},
        launchable_export_bundle={"status": "missing"},
    )
    assert hosted["status"] == "blocked"
    assert "preview_manifest_missing" in hosted["blockers"]
    assert "runtime_demo_manifest_missing" in hosted["blockers"]

    ready_pack = proof_contracts.build_proof_pack_manifest(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id="site-submission-1",
        opportunity_id="opp-1",
        site_package_manifest={"status": "ready", "site_labeling": {"site_scope": "exact_site"}},
        rights_review={"status": "cleared"},
        hosted_review_readiness={"status": "ready"},
    )
    assert ready_pack["status"] == "ready"
    assert ready_pack["proof_pack_ready"] is True
    blocked_pack = proof_contracts.build_proof_pack_manifest(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id=None,
        opportunity_id=None,
        site_package_manifest={"status": "blocked"},
        rights_review={"status": "blocked"},
        hosted_review_readiness={"status": "blocked"},
    )
    assert blocked_pack["status"] == "blocked"
    assert blocked_pack["blockers"] == ["site_package_not_ready", "rights_review:blocked"]
