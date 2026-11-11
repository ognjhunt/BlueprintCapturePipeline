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
from blueprint_pipeline.evaluation_prep_stage import _privacy_processing_cleared
from blueprint_pipeline.proof_contracts import build_proof_pack_manifest
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


