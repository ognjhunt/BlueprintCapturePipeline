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

from blueprint_pipeline.alpha_readiness import (
    build_launch_gate_summary,
    write_pipeline_sync_result,
)
from blueprint_pipeline.canonical_site_package import _world_labs_readiness
from blueprint_pipeline.launch_bundle import build_buyer_trust_score
from blueprint_pipeline.proof_contracts import (
    build_proof_pack_manifest,
    build_rights_provenance_review,
)


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


def _cleared_rights_summary(**overrides: object) -> dict:
    summary: dict = {
        "derived_scene_generation_allowed": True,
        "consent_status": "documented",
        "permission_document_uri": "gs://bucket/rights/consent-packet.pdf",
    }
    summary.update(overrides)
    return summary


def _review_with_rights(rights_summary: dict, **kwargs: object) -> dict:
    return build_rights_provenance_review(
        rights_summary=rights_summary,
        privacy_processing={"status": "person_removed"},
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


def test_consent_scope_covering_required_use_classes_clears() -> None:
    review = _review_with_rights(
        _cleared_rights_summary(
            consent_scope=["robot_evaluation", "model_training"]
        ),
        required_use_classes=["model_training"],
    )
    assert review["status"] == "cleared"
    assert review["rights"]["status"] == "cleared"


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
    assert fallback["privacy"]["fallback_redaction_used"] is True
    assert fallback["privacy"]["manual_review_recommended"] is True


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
    # Labeled non-production preview stays possible but is never "ready".
    assert readiness["status"] == "review_required"
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
