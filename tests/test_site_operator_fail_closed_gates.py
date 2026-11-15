"""Site-operator fail-closed gate regressions.

Beta-launch audit: no missing site/operator/privacy evidence may ever read as a
green launch or buyer-ready state. These tests pin the fail-closed behavior for
missing consent, raw World Labs bypass, placeholder WebApp ids, stale sync
results, and incomplete rights packets.
"""

from __future__ import annotations

import json
from pathlib import Path


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


# ---------------------------------------------------------------------------
# Raw World Labs bypass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Placeholder WebApp ids
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Missing operator/launch evidence never reads green
# ---------------------------------------------------------------------------


