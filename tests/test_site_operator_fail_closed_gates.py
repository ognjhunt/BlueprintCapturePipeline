"""Site-operator fail-closed gate regressions.

Beta-launch audit: no missing site/operator/privacy evidence may ever read as a
green launch or buyer-ready state. These tests pin the fail-closed behavior for
missing consent, raw World Labs bypass, placeholder WebApp ids, stale sync
results, and incomplete rights packets.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.alpha_readiness import (
    build_launch_gate_summary,
    write_pipeline_sync_result,
)
from blueprint_pipeline.canonical_site_package import _world_labs_readiness
from blueprint_pipeline.evaluation_prep_stage import _privacy_processing_cleared
from blueprint_pipeline.launch_bundle import build_buyer_trust_score
from blueprint_pipeline.proof_contracts import (
    build_proof_pack_manifest,
    build_rights_provenance_review,
)
from blueprint_pipeline.site_package_orchestrator import _worldlabs_derived_rights_allowed


def _launch_gate_capture(tmp_path: Path, *, descriptor_overrides: dict | None = None) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    (capture_root / "pipeline").mkdir(parents=True, exist_ok=True)
    descriptor = {
        "schema_version": "v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/raw",
        "frames_index_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/frames/index.json",
        "capture_source": "iphone",
        "capture_modality": "iphone_arkit_lidar",
        "site_submission_id": "site-submission-real",
        "buyer_request_id": "buyer-request-real",
        "capture_job_id": "capture-job-real",
        "quoted_payout_cents": 6500,
    }
    descriptor.update(descriptor_overrides or {})
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    return capture_root


def _stage_checks(summary: dict) -> dict:
    return {check["name"]: check for check in summary["stage_checks"]}


def _operator_check(summary: dict, check_id: str) -> dict:
    return {check["id"]: check for check in summary["operator_required_checks"]}[check_id]


def test_operator_launch_evidence_template_is_parseable_and_non_verified() -> None:
    template = json.loads(
        (Path(__file__).resolve().parents[1] / "docs" / "operator_launch_evidence.template.json").read_text(
            encoding="utf-8"
        )
    )

    assert template["schema_version"] == "operator_launch_evidence.v1"
    checks = template["checks"]
    for check_id in (
        "buyer_payment_settlement",
        "capturer_payout_settlement",
        "stripe_connected_account_live_readiness",
        "buyer_artifact_access",
        "human_finance_review_owner",
    ):
        assert checks[check_id]["status"] == "manual_live_evidence_required"
    assert checks["buyer_artifact_access"]["buyer_session_ref"] == ""
    assert checks["operator_dpa_data_processing_terms"]["subprocessors"] == []
    assert checks["operator_dpa_data_processing_terms"]["access_audit_terms_uri"] == ""
    assert checks["cross_border_data_residency_posture"]["allowed_tester_countries"] == ["US"]
    assert checks["cross_border_data_residency_posture"]["non_us_participants_blocked"] is False
    assert checks["stripe_connected_account_live_readiness"]["provider_state_checked"] is False
    assert checks["buyer_payment_settlement"]["stripe_mode"] == ""
    assert checks["capturer_payout_settlement"]["stripe_mode"] == ""


# ---------------------------------------------------------------------------
# Missing consent / incomplete rights packets
# ---------------------------------------------------------------------------


def test_missing_rights_summary_never_clears_rights_review() -> None:
    review = build_rights_provenance_review(
        rights_summary=None,
        privacy_processing={"status": "person_removed"},
        provenance_summary={"status": "grounded", "record": {"canonical_truth": True}},
        site_identity={"site_id": "site-1"},
        adjacent_systems=None,
    )
    assert review["status"] == "blocked"
    assert review["rights"]["status"] == "blocked"
    assert "rights_not_sufficient_for_derived_generation" in review["blockers"]


def test_unknown_consent_never_clears_rights_review() -> None:
    review = build_rights_provenance_review(
        rights_summary={"derived_scene_generation_allowed": True},
        privacy_processing={"status": "person_removed"},
        provenance_summary={"status": "grounded", "record": {"canonical_truth": True}},
        site_identity={"site_id": "site-1"},
        adjacent_systems=None,
    )
    assert review["status"] != "cleared"
    assert review["rights"]["status"] == "needs_review"
    assert "rights_or_consent_requires_review" in review["blockers"]


def test_documented_consent_without_permission_document_is_incomplete_packet() -> None:
    review = build_rights_provenance_review(
        rights_summary={
            "derived_scene_generation_allowed": True,
            "consent_status": "documented",
        },
        privacy_processing={"status": "person_removed"},
        provenance_summary={"status": "grounded", "record": {"canonical_truth": True}},
        site_identity={"site_id": "site-1"},
        adjacent_systems=None,
    )
    assert review["status"] != "cleared"
    assert review["rights"]["status"] == "needs_review"
    assert "consent_documented_without_permission_document" in review["blockers"]


def test_documented_consent_with_permission_document_clears() -> None:
    review = build_rights_provenance_review(
        rights_summary={
            "derived_scene_generation_allowed": True,
            "consent_status": "documented",
            "permission_document_uri": "gs://bucket/rights/consent-packet.pdf",
        },
        privacy_processing={"status": "person_removed"},
        provenance_summary={"status": "grounded", "record": {"canonical_truth": True}},
        site_identity={"site_id": "site-1"},
        adjacent_systems=None,
    )
    assert review["status"] == "cleared"
    assert review["rights"]["status"] == "cleared"


def test_string_false_rights_fields_do_not_clear_or_revoke() -> None:
    review = build_rights_provenance_review(
        rights_summary={
            "derived_scene_generation_allowed": "false",
            "data_licensing_allowed": "false",
            "consent_revoked": "false",
            "consent_status": "documented",
            "permission_document_uri": "gs://bucket/rights/consent-packet.pdf",
        },
        privacy_processing={
            "status": "person_removed",
            "fail_closed": "false",
            "raw_retained": "false",
        },
        provenance_summary={"status": "grounded", "record": {"canonical_truth": True}},
        site_identity={"site_id": "site-1"},
        adjacent_systems=None,
    )

    assert review["status"] == "blocked"
    assert review["rights"]["status"] == "blocked"
    assert review["rights"]["consent_revoked"] is False
    assert review["rights"]["derived_scene_generation_allowed"] is False
    assert review["rights"]["data_licensing_allowed"] is False
    assert review["privacy"]["fail_closed"] is False
    assert review["privacy"]["raw_retained"] is False
    assert "rights_not_sufficient_for_derived_generation" in review["blockers"]
    assert "consent_revoked_takedown_required" not in review["blockers"]


def test_string_true_rights_fields_are_parsed_explicitly() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(
            derived_scene_generation_allowed="true",
            data_licensing_allowed="true",
            consent_revoked="false",
        )
    )

    assert review["status"] == "cleared"
    assert review["rights"]["derived_scene_generation_allowed"] is True
    assert review["rights"]["data_licensing_allowed"] is True
    assert review["rights"]["consent_revoked"] is False


def test_string_false_capture_rights_do_not_allow_worldlabs_generation() -> None:
    assert (
        _worldlabs_derived_rights_allowed(
            metadata={
                "capture_rights": {
                    "derived_scene_generation_allowed": "false",
                }
            }
        )
        is False
    )
    assert (
        _worldlabs_derived_rights_allowed(
            metadata={
                "capture_rights": {
                    "derived_scene_generation_allowed": "true",
                }
            }
        )
        is True
    )


def _cleared_rights_summary(**overrides: object) -> dict:
    summary: dict = {
        "derived_scene_generation_allowed": True,
        "consent_status": "documented",
        "permission_document_uri": "gs://bucket/rights/consent-packet.pdf",
    }
    summary.update(overrides)
    return summary


def _review_with_rights(rights_summary: dict, **kwargs: object) -> dict:
    privacy_processing = kwargs.pop("privacy_processing", {"status": "person_removed"})
    return build_rights_provenance_review(
        rights_summary=rights_summary,
        privacy_processing=privacy_processing,
        provenance_summary={"status": "grounded", "record": {"canonical_truth": True}},
        site_identity={"site_id": "site-1"},
        adjacent_systems=None,
        **kwargs,
    )


def test_revoked_consent_blocks_even_with_complete_packet() -> None:
    for revocation in (
        {"consent_status": "revoked"},
        {"consent_status": "withdrawn"},
        {"consent_revoked": True},
        {"consent_revoked_at": "2026-07-01T00:00:00Z"},
    ):
        review = _review_with_rights(_cleared_rights_summary(**revocation))
        assert review["status"] == "blocked", revocation
        assert review["rights"]["status"] == "blocked", revocation
        assert review["rights"]["consent_revoked"] is True, revocation
        assert "consent_revoked_takedown_required" in review["blockers"], revocation


def test_explicit_consent_scope_excluding_use_class_blocks() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(consent_scope=["robot_evaluation"]),
        required_use_classes=["model_training"],
    )
    assert review["status"] == "blocked"
    assert review["rights"]["status"] == "blocked"
    assert "consent_scope_excludes_use_class:model_training" in review["blockers"]
    assert review["rights"]["scope_excluded_use_classes"] == ["model_training"]


def test_unspecified_consent_scope_cannot_silently_grant_use_class() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(),
        required_use_classes=["model_training"],
    )
    assert review["status"] == "needs_review"
    assert review["rights"]["status"] == "needs_review"
    assert any(
        blocker.startswith("consent_scope_unspecified_for_required_use_classes")
        for blocker in review["blockers"]
    )


def test_location_only_consent_scope_is_not_product_use_grant() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(consent_scope=["warehouse-a"]),
        required_use_classes=["model_training"],
    )
    assert review["status"] == "needs_review"
    assert review["rights"]["status"] == "needs_review"
    assert review["rights"]["consent_scope"] == ["warehouse-a"]
    assert review["rights"]["consent_use_classes"] == []
    assert any(
        blocker.startswith("consent_scope_unspecified_for_required_use_classes")
        for blocker in review["blockers"]
    )


def test_consent_scope_covering_required_use_classes_clears() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(
            consent_scope=["robot_evaluation", "model_training"]
        ),
        required_use_classes=["model_training"],
    )
    assert review["status"] == "cleared"
    assert review["rights"]["status"] == "cleared"


def test_policy_only_without_operator_permission_blocks_private_industrial_sites() -> None:
    review = _review_with_rights(
        {
            "derived_scene_generation_allowed": True,
            "consent_status": "policy_only",
            "site_type": "warehouse",
        },
        privacy_processing={"status": "person_removed"},
    )

    assert review["status"] == "blocked"
    assert review["rights"]["status"] == "blocked"
    assert review["rights"]["operator_permission_required"] is True
    assert (
        "policy_only_requires_operator_permission_for_private_or_industrial_site"
        in review["blockers"]
    )


def test_policy_only_can_clear_public_site_only_when_site_scope_allows_it() -> None:
    review = _review_with_rights(
        {
            "derived_scene_generation_allowed": True,
            "consent_status": "policy_only",
            "site_type": "public_space",
        }
    )

    assert review["status"] == "cleared"
    assert review["rights"]["operator_permission_required"] is False
    assert review["rights"]["policy_only_evidence_complete"] is True


def test_industrial_privacy_clearance_requires_non_person_pii_redaction_scope() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(site_type="warehouse"),
        privacy_processing={"status": "person_removed", "redaction_target_classes": ["person"]},
    )

    assert review["status"] == "blocked"
    assert review["privacy"]["status"] == "blocked"
    assert "badge_id" in review["privacy"]["missing_required_redaction_classes"]
    assert any(
        blocker.startswith("privacy_industrial_redaction_scope_incomplete:")
        for blocker in review["blockers"]
    )


def test_industrial_privacy_clearance_accepts_explicit_pii_redaction_scope() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(site_type="warehouse"),
        privacy_processing={
            "status": "person_removed",
            "redaction_target_classes": [
                "person",
                "face",
                "badge_id",
                "screen",
                "whiteboard",
                "signage",
                "license_plate",
                "shipping_label",
            ],
        },
    )

    assert review["status"] == "cleared"
    assert review["privacy"]["missing_required_redaction_classes"] == []


def test_rights_review_carries_operator_revenue_terms_without_payout_claim() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(
            consent_scope=["robot_evaluation"],
            commercialization_terms={
                "license_model": "request_scoped",
                "revenue_share": {
                    "terms_uri": "gs://bucket/rights/revenue-share.pdf",
                    "operator_revenue_share_bps": 1500,
                    "payee_entity_id": "operator-1",
                },
                "exclusivity": {"exclusive": False},
            },
        ),
        required_use_classes=["robot_evaluation"],
    )
    assert review["status"] == "cleared"
    assert review["rights"]["commercialization_terms"]["license_model"] == (
        "request_scoped"
    )
    assert review["rights"]["operator_revenue_terms"]["operator_revenue_share_bps"] == 1500
    assert review["rights"]["exclusivity_terms"]["exclusive"] is False
    assert review["rights"]["revenue_share_commitment_made"] is False
    assert review["rights"]["payout_commitment_allowed"] is False


def test_fallback_redaction_is_flagged_for_manual_review() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(),
    )
    assert review["privacy"]["fallback_redaction_used"] is False
    fallback = build_rights_provenance_review(
        rights_summary=_cleared_rights_summary(),
        privacy_processing={"status": "face_anonymized_fallback"},
        provenance_summary={"status": "grounded", "record": {"canonical_truth": True}},
        site_identity={"site_id": "site-1"},
        adjacent_systems=None,
    )
    assert fallback["status"] == "needs_review"
    assert "privacy_fallback_redaction_requires_manual_review" in fallback["blockers"]
    assert fallback["privacy"]["status"] == "needs_review"
    assert fallback["privacy"]["fallback_redaction_used"] is True
    assert fallback["privacy"]["manual_review_recommended"] is True
    assert fallback["privacy"]["external_delivery_allowed"] is False


def test_evaluation_prep_does_not_treat_privacy_fallback_as_cleared() -> None:
    assert (
        _privacy_processing_cleared(
            rights_review=None,
            privacy_processing={"status": "face_anonymized_fallback"},
        )
        is False
    )
    assert (
        _privacy_processing_cleared(
            rights_review=None,
            privacy_processing={"status": "full_frame_redacted_local_proof"},
        )
        is False
    )


def test_proof_pack_manifest_fails_closed_on_missing_rights_review() -> None:
    manifest = build_proof_pack_manifest(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id="sub-1",
        opportunity_id=None,
        site_package_manifest={"status": "ready", "site_labeling": {}},
        rights_review=None,
        hosted_review_readiness={"status": "ready"},
    )
    assert manifest["status"] == "blocked"
    assert manifest["proof_pack_ready"] is False
    assert "rights_review:unavailable" in manifest["blockers"]


def test_launch_gate_blocks_when_rights_review_artifact_missing(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert summary["overall_status"] == "blocked"
    assert checks["rights_provenance_review_cleared"]["passed"] is False
    assert "missing" in checks["rights_provenance_review_cleared"]["detail"]


# ---------------------------------------------------------------------------
# Raw World Labs bypass
# ---------------------------------------------------------------------------


def test_world_labs_readiness_blocks_raw_bypass_in_production(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    readiness = _world_labs_readiness(
        worldlabs_input={
            "output_video_uri": "gs://bucket/worldlabs_input.mp4",
            "status": "ready",
            "input_labeling": {"raw_video_bypass_used": True, "privacy_safe_input": False},
        },
        privacy_processing={"status": "failed_closed"},
        rights_review={"status": "blocked"},
        provenance_summary={"status": "grounded"},
    )
    assert readiness["status"] == "blocked"
    assert "raw_video_bypass_used_in_production" in readiness["blockers"]
    # Production never downgrades privacy/rights failures to warnings.
    assert "privacy_processing_failed_closed" in readiness["blockers"]
    assert "rights_provenance_review_blocked" in readiness["blockers"]
    assert "privacy_safe_world_model_input_not_verified" in readiness["blockers"]


def test_world_labs_readiness_raw_bypass_never_reads_ready_outside_production(monkeypatch) -> None:
    monkeypatch.delenv("BLUEPRINT_LAUNCH_PROOF_MODE", raising=False)
    readiness = _world_labs_readiness(
        worldlabs_input={
            "output_video_uri": "gs://bucket/worldlabs_input.mp4",
            "status": "ready",
            "input_labeling": {"raw_video_bypass_used": True, "privacy_safe_input": False},
        },
        privacy_processing={"status": "not_run"},
        rights_review={"status": "cleared"},
        provenance_summary={"status": "grounded"},
    )
    # Labeled non-production preview stays visible, but it never downgrades
    # privacy failures to warnings.
    assert readiness["status"] == "blocked"
    assert "privacy_safe_world_model_input_not_verified" in readiness["blockers"]
    assert "raw_video_bypass_input_non_production" in readiness["warnings"]


def test_launch_gate_blocks_raw_worldlabs_bypass(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "worldlabs_input_audit.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "privacy_safe_input": False,
                "input_labeling": {"raw_video_bypass_used": True},
            }
        ),
        encoding="utf-8",
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert summary["overall_status"] == "blocked"
    assert checks["raw_worldlabs_bypass_not_used"]["passed"] is False
    assert "never buyer-ready" in checks["raw_worldlabs_bypass_not_used"]["detail"]


# ---------------------------------------------------------------------------
# Placeholder WebApp ids
# ---------------------------------------------------------------------------


def test_launch_gate_rejects_placeholder_webapp_ids(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(
        tmp_path,
        descriptor_overrides={
            "site_submission_id": "example-site-submission",
            "buyer_request_id": "placeholder-buyer-request",
            "capture_job_id": "your-capture-job",
        },
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert summary["overall_status"] == "blocked"
    assert checks["inbound_request_linked"]["passed"] is False
    assert checks["buyer_request_linked"]["passed"] is False
    assert checks["approved_marketplace_capture_job_linked"]["passed"] is False
    assert "not a real WebApp record" in checks["inbound_request_linked"]["detail"]


def test_launch_gate_rejects_capture_derived_webapp_ids(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(
        tmp_path,
        descriptor_overrides={
            "site_submission_id": "scene-1:capture-1",
            "buyer_request_id": "capture-1",
        },
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert summary["overall_status"] == "blocked"
    assert checks["inbound_request_linked"]["passed"] is False
    assert checks["buyer_request_linked"]["passed"] is False


# ---------------------------------------------------------------------------
# Stale / unverified sync results
# ---------------------------------------------------------------------------


def _sync_result(*, synced_at: str | None, verified: bool = True) -> dict:
    payload = {
        "status": "succeeded",
        "attempts": 1,
        "attachment_payload": {
            "upstream_links_verified": verified,
            "placeholder_fallback_allowed": False,
        },
    }
    if synced_at is not None:
        payload["synced_at"] = synced_at
    return {
        "status": "succeeded",
        "latest_stage": "evaluation_prep",
        "syncs": {"evaluation_prep": payload},
    }


def test_launch_gate_blocks_stale_webapp_sync_result(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    stale = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    (capture_root / "pipeline" / "webapp_sync_result.json").write_text(
        json.dumps(_sync_result(synced_at=stale)), encoding="utf-8"
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert summary["overall_status"] == "blocked"
    assert checks["webapp_sync_completed"]["passed"] is False
    assert "stale_sync_result" in checks["webapp_sync_completed"]["detail"]


def test_launch_gate_blocks_sync_result_without_timestamp(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "webapp_sync_result.json").write_text(
        json.dumps(_sync_result(synced_at=None)), encoding="utf-8"
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert checks["webapp_sync_completed"]["passed"] is False
    assert "synced_at_missing" in checks["webapp_sync_completed"]["detail"]


def test_launch_gate_blocks_sync_result_with_unverified_upstream_links(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    fresh = datetime.now(timezone.utc).isoformat()
    (capture_root / "pipeline" / "webapp_sync_result.json").write_text(
        json.dumps(_sync_result(synced_at=fresh, verified=False)), encoding="utf-8"
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert checks["webapp_sync_completed"]["passed"] is False
    assert "upstream_links_not_verified" in checks["webapp_sync_completed"]["detail"]


def test_launch_gate_accepts_fresh_verified_sync_result(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    fresh = datetime.now(timezone.utc).isoformat()
    (capture_root / "pipeline" / "webapp_sync_result.json").write_text(
        json.dumps(_sync_result(synced_at=fresh)), encoding="utf-8"
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert checks["webapp_sync_completed"]["passed"] is True


def test_write_pipeline_sync_result_stamps_synced_at(tmp_path: Path) -> None:
    pipeline_root = tmp_path / "pipeline"
    pipeline_root.mkdir(parents=True)
    payload = write_pipeline_sync_result(
        pipeline_root=pipeline_root,
        stage="qualification",
        result={"status": "succeeded"},
    )
    assert payload["syncs"]["qualification"]["synced_at"]
    assert payload["latest_synced_at"] == payload["syncs"]["qualification"]["synced_at"]


# ---------------------------------------------------------------------------
# Missing operator/launch evidence never reads green
# ---------------------------------------------------------------------------


def test_launch_gate_blocks_when_recapture_state_is_unknown(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert checks["recapture_not_required"]["passed"] is False
    assert "unknown" in checks["recapture_not_required"]["detail"]


def test_launch_gate_blocks_when_recapture_is_required(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "recapture_requirements.json").write_text(
        json.dumps(
            {
                "required": True,
                "missing_evidence": ["north aisle coverage"],
                "recommendations": ["recapture aisle at walking pace"],
            }
        ),
        encoding="utf-8",
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert checks["recapture_not_required"]["passed"] is False
    assert "north aisle coverage" in checks["recapture_not_required"]["detail"]


def test_launch_gate_blocks_payout_without_explicit_eligibility(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    # Legacy artifact: recommended amount present but no explicit decision.
    (capture_root / "pipeline" / "capturer_payout_recommendation.json").write_text(
        json.dumps({"status": "baseline", "recommended_payout_cents": 6500}),
        encoding="utf-8",
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert checks["capturer_payout_transition_ready"]["passed"] is False


def test_operator_launch_evidence_rejects_wrong_schema_version(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "operator_launch_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "operator_launch_evidence.v0",
                "checks": {
                    "legal_consent_posture_signoff": {
                        "status": "verified",
                        "signed_record_uri": "gs://operator/legal-signoff.json",
                        "verified_at": "2026-07-07T00:00:00+00:00",
                        "verified_by": "legal-owner",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_launch_gate_summary(capture_root=capture_root, env={})

    assert "operator_launch_evidence_schema_version_invalid" in summary["operator_evidence_status"]["schema_errors"]
    legal_check = {
        check["id"]: check for check in summary["operator_required_checks"]
    }["legal_consent_posture_signoff"]
    assert legal_check["passed"] is False
    assert "operator_launch_evidence_schema_version_invalid" in legal_check["evidence_validation_errors"]


def test_buyer_artifact_access_operator_evidence_requires_executed_authenticated_fetch(
    tmp_path: Path,
) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "operator_launch_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "operator_launch_evidence.v1",
                "checks": {
                    "buyer_artifact_access": {
                        "status": "verified",
                        "evidence_uri": "gs://operator/buyer-access-attestation.json",
                        "verified_at": "2026-07-07T00:00:00+00:00",
                        "verified_by": "ops-owner",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    buyer_access = _operator_check(summary, "buyer_artifact_access")

    assert buyer_access["passed"] is False
    assert buyer_access["evidence_validation_errors"] == [
        "missing_authenticated_buyer_session_ref",
        "missing_artifact_access_log",
        "missing_executed_artifact_access_fetch",
    ]


@pytest.mark.parametrize(
    ("check_id", "expected_errors"),
    [
        ("legal_consent_posture_signoff", ["missing_signed_legal_or_dpa_record"]),
        (
            "operator_dpa_data_processing_terms",
            [
                "missing_signed_legal_or_dpa_record",
                "missing_retention_policy_terms",
                "missing_subprocessor_list",
                "missing_access_audit_terms",
            ],
        ),
        (
            "paperclip_ops_relay_secret_rotation",
            ["missing_secret_version_ref", "missing_redeploy_evidence"],
        ),
        (
            "cross_border_data_residency_posture",
            [
                "missing_data_residency_or_transfer_record",
                "missing_us_only_scope_or_signed_transfer_terms",
            ],
        ),
        (
            "iphone_real_device_claim_flow",
            ["missing_real_device_recording", "missing_capture_job_id_continuity"],
        ),
        (
            "buyer_artifact_access",
            [
                "missing_authenticated_buyer_session_ref",
                "missing_artifact_access_log",
                "missing_executed_artifact_access_fetch",
            ],
        ),
    ],
)
def test_operator_launch_evidence_rejects_generic_evidence_uri_for_specific_live_checks(
    tmp_path: Path,
    check_id: str,
    expected_errors: list[str],
) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "operator_launch_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "operator_launch_evidence.v1",
                "checks": {
                    check_id: {
                        "status": "verified",
                        "evidence_uri": f"gs://operator/{check_id}.json",
                        "verified_at": "2026-07-07T00:00:00+00:00",
                        "verified_by": "ops-owner",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    check = _operator_check(build_launch_gate_summary(capture_root=capture_root, env={}), check_id)

    assert check["passed"] is False
    assert check["evidence_validation_errors"] == expected_errors


@pytest.mark.parametrize(
    ("check_id", "fields"),
    [
        (
            "buyer_payment_settlement",
            {"payment_intent_id": "pi_live_123", "stripe_event_id": "evt_live_payment_123"},
        ),
        (
            "capturer_payout_settlement",
            {
                "payout_id": "po_live_123",
                "transfer_id": "tr_live_123",
                "webhook_reconciliation_uri": "gs://operator/payout-reconciliation.json",
            },
        ),
    ],
)
def test_operator_payment_and_payout_evidence_require_live_mode(
    tmp_path: Path,
    check_id: str,
    fields: dict[str, str],
) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "operator_launch_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "operator_launch_evidence.v1",
                "checks": {
                    check_id: {
                        "status": "verified",
                        "evidence_uri": f"gs://operator/{check_id}.json",
                        "verified_at": "2026-07-07T00:00:00+00:00",
                        "verified_by": "ops-owner",
                        **fields,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    check = _operator_check(build_launch_gate_summary(capture_root=capture_root, env={}), check_id)

    assert check["passed"] is False
    assert check["evidence_validation_errors"] == ["stripe_mode_not_live"]


def test_buyer_artifact_access_operator_evidence_requires_successful_fetch_status(
    tmp_path: Path,
) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "operator_launch_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "operator_launch_evidence.v1",
                "checks": {
                    "buyer_artifact_access": {
                        "status": "verified",
                        "buyer_session_ref": "buyer-session-live-123",
                        "artifact_access_log_uri": "gs://operator/buyer-access-log.json",
                        "authenticated_fetch_status": "failed",
                        "verified_at": "2026-07-07T00:00:00+00:00",
                        "verified_by": "ops-owner",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    buyer_access = _operator_check(
        build_launch_gate_summary(capture_root=capture_root, env={}),
        "buyer_artifact_access",
    )

    assert buyer_access["passed"] is False
    assert buyer_access["evidence_validation_errors"] == [
        "missing_executed_artifact_access_fetch"
    ]


def test_launch_gate_blocks_ungrounded_provenance(tmp_path: Path) -> None:
    capture_root = _launch_gate_capture(tmp_path)
    (capture_root / "pipeline" / "provenance_summary.json").write_text(
        json.dumps({"status": "missing"}), encoding="utf-8"
    )
    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = _stage_checks(summary)
    assert checks["provenance_summary_grounded"]["passed"] is False


def test_buyer_trust_score_penalizes_missing_completeness_evidence() -> None:
    score = build_buyer_trust_score(
        descriptor={"quality": {"pose_match_rate": 0.9}},
        qualification_record={"confidence": 0.9},
        scorecard={},
        metadata={
            "capture_rights": {
                "consent_status": "documented",
                "permission_document_uri": "gs://bucket/rights/consent.pdf",
            }
        },
        provider_status="ready",
        fidelity_review={"status": "succeeded", "scores": {"coverage": 0.9, "world_model_fitness": 0.9}},
    )
    assert "capture completeness evidence is missing" in score["reasons"]
