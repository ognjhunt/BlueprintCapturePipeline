from __future__ import annotations

from blueprint_pipeline.proof_contracts import build_proof_path_status
from blueprint_pipeline.proof_contracts import build_hosted_review_readiness
from blueprint_pipeline.proof_contracts import build_site_package_manifest


def test_build_proof_path_status_exposes_verified_event_sequence() -> None:
    status = build_proof_path_status(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id="site-1",
        opportunity_id="op-1",
        rights_review={"status": "cleared", "site_labeling": {"region": "sacramento"}},
        site_package_manifest={"status": "ready", "site_labeling": {"region": "sacramento"}},
        proof_pack_manifest={"status": "ready"},
        hosted_review_readiness={"status": "ready"},
    )

    assert status["proof_pack_ready"] is True
    assert status["hosted_review_ready"] is True
    assert status["rights_cleared"] is True
    assert [item["event_name"] for item in status["event_statuses"]] == [
        "proof_pack_delivered",
        "hosted_review_started",
        "hosted_review_follow_up_sent",
        "human_commercial_handoff_started",
    ]
    assert all(item["status"] == "verified" for item in status["event_statuses"])


def test_build_proof_path_status_marks_unready_events_pending() -> None:
    status = build_proof_path_status(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id="site-1",
        opportunity_id="op-1",
        rights_review={"status": "blocked"},
        site_package_manifest={"status": "blocked"},
        proof_pack_manifest={"status": "blocked"},
        hosted_review_readiness={"status": "blocked"},
    )

    assert status["next_truthful_step"] == "clear_rights_or_privacy_review"
    assert all(item["status"] == "pending" for item in status["event_statuses"])


def test_build_proof_path_status_requires_rights_for_delivery_event() -> None:
    status = build_proof_path_status(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id="site-1",
        opportunity_id="op-1",
        rights_review={"status": "blocked"},
        site_package_manifest={"status": "ready"},
        proof_pack_manifest={"status": "ready"},
        hosted_review_readiness={"status": "ready"},
    )

    assert status["event_statuses"][0]["event_name"] == "proof_pack_delivered"
    assert status["event_statuses"][0]["status"] == "pending"


def test_build_site_package_manifest_marks_blocked_when_rights_review_is_blocked() -> None:
    manifest = build_site_package_manifest(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id="site-1",
        opportunity_id="op-1",
        evaluation_prep_manifest={"canonical_package_status": "ready"},
        site_world_spec={"status": "ready"},
        site_world_registration={"runtime_registration_status": "ready"},
        site_world_health={"status": "healthy", "launchable": True},
        launchable_export_bundle={"status": "ready"},
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
        rights_review={"status": "blocked"},
    )

    assert manifest["status"] == "blocked"
    assert "rights_review:blocked" in manifest["blockers"]


def test_build_hosted_review_readiness_marks_blocked_when_demo_blockers_exist() -> None:
    readiness = build_hosted_review_readiness(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id="site-1",
        opportunity_id="op-1",
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
        preview_manifest_uri="gs://bucket/preview.json",
        worldlabs_launch_url="https://example.test/worldlabs",
        runtime_demo_manifest_uri="gs://bucket/runtime-demo.json",
        demo_readiness_state="ready",
        demo_blockers=["operator_review_required"],
        site_world_health={"launchable": True},
        launchable_export_bundle={"status": "ready"},
    )

    assert readiness["status"] == "blocked"
    assert "operator_review_required" in readiness["blockers"]
