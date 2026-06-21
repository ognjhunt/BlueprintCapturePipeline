from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import same_capture_lineage as lineage
from blueprint_pipeline.same_capture_lineage import (
    build_same_capture_lineage_packet,
    validate_same_capture_lineage_packet,
    write_same_capture_lineage_packet,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_capture_chain(
    tmp_path: Path,
    *,
    capture_id: str = "capture-001",
    scene_id: str = "scene-001",
    capture_profile_id: str = "iphone_arkit_lidar",
    capture_modality: str = "iphone_arkit_lidar",
    include_webapp_ids: bool = True,
    geometry_source: str = "video_to_world",
    fallback_used: bool = False,
) -> Path:
    capture_root = tmp_path / "bucket" / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    pipeline_root = capture_root / "pipeline"
    raw_root.mkdir(parents=True)

    upstream_handoff = {
        "site_submission_id": "site-submission-001" if include_webapp_ids else None,
        "request_id": "request-001" if include_webapp_ids else None,
        "buyer_request_id": "buyer-request-001" if include_webapp_ids else None,
        "capture_job_id": "capture-job-001" if include_webapp_ids else None,
        "hosted_review_truth_state": "verified" if include_webapp_ids else "blocked_missing_upstream_ids",
        "blockers": [] if include_webapp_ids else [
            "missing_site_submission_id",
            "missing_request_id",
            "missing_buyer_request_id",
            "missing_capture_job_id",
        ],
    }
    manifest = {
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_profile_id": capture_profile_id,
        "capture_modality": capture_modality,
        "capture_capabilities": {
            "camera_pose": capture_profile_id != "android_xr_glasses",
            "camera_intrinsics": capture_profile_id != "android_xr_glasses",
            "depth": capture_profile_id != "android_xr_glasses",
            "geospatial": False,
        },
        "upstream_handoff": upstream_handoff,
    }
    _write_json(raw_root / "manifest.json", manifest)
    _write_json(
        raw_root / "capture_context.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "capture_profile_id": capture_profile_id,
            "capture_modality": capture_modality,
            "upstream_handoff": upstream_handoff,
        },
    )
    _write_json(
        raw_root / "capture_upload_complete.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "complete",
            "completed_at": "2026-05-30T12:00:00Z",
        },
    )
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "schema_version": "v1",
            "scene_id": scene_id,
            "capture_id": capture_id,
            "capture_profile_id": capture_profile_id,
            "capture_modality": capture_modality,
            "site_submission_id": upstream_handoff["site_submission_id"],
            "request_id": upstream_handoff["request_id"],
            "buyer_request_id": upstream_handoff["buyer_request_id"],
            "capture_job_id": upstream_handoff["capture_job_id"],
            "metadata": {
                "upstream_handoff": upstream_handoff,
                "hosted_review_blockers": upstream_handoff["blockers"],
            },
        },
    )
    _write_json(capture_root / "qa_report.json", {"scene_id": scene_id, "capture_id": capture_id, "status": "passed"})
    _write_json(
        capture_root / "pipeline_handoff.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "raw_prefix_uri": f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/raw",
            "upstream_handoff": upstream_handoff,
        },
    )
    _write_json(
        pipeline_root / "opportunity_handoff.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "site_submission_id": upstream_handoff["site_submission_id"],
            "request_id": upstream_handoff["request_id"],
            "buyer_request_id": upstream_handoff["buyer_request_id"],
            "capture_job_id": upstream_handoff["capture_job_id"],
            "upstream_link_truth_state": "verified" if include_webapp_ids else "blocked_missing_upstream_ids",
            "upstream_link_blockers": upstream_handoff["blockers"],
        },
    )
    _write_json(
        pipeline_root / "qualification_summary.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "ready",
            "readiness_state": "ready",
        },
    )
    _write_json(
        pipeline_root / "geometry" / "geometry_summary.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "completed",
            "geometry_source": geometry_source,
            "fallback_used": fallback_used,
            "provider_native_result": geometry_source == "video_to_world" and not fallback_used,
            "site_frame_available": geometry_source == "video_to_world" and not fallback_used,
            "scale_resolved": geometry_source == "video_to_world" and not fallback_used,
            "ready_for_world_model": True,
            "geometry_live_ready": geometry_source == "video_to_world" and not fallback_used,
            "launch_blockers": [] if not fallback_used else ["fallback_geometry_not_live_video_to_world"],
        },
    )
    _write_json(
        pipeline_root / "webapp_sync_result.json",
        {
            "status": "skipped" if include_webapp_ids else "failed",
            "latest_stage": "qualification",
            "syncs": {
                "qualification": {
                    "status": "skipped" if include_webapp_ids else "failed",
                    "reason": "sync_not_configured" if include_webapp_ids else "missing_upstream_pipeline_records",
                    "attachment_payload": {
                        "scene_id": scene_id,
                        "capture_id": capture_id,
                        "site_submission_id": upstream_handoff["site_submission_id"] or "",
                        "request_id": upstream_handoff["request_id"] or "",
                        "buyer_request_id": upstream_handoff["buyer_request_id"] or "",
                        "capture_job_id": upstream_handoff["capture_job_id"] or "",
                        "upstream_links_verified": include_webapp_ids,
                        "missing_upstream_links": [] if include_webapp_ids else [
                            "site_submission_id",
                            "request_id",
                            "buyer_request_id",
                            "capture_job_id",
                        ],
                    },
                    "buyer_access_check": {"buyer_access_checked": False, "buyer_accessible": False},
                }
            },
        },
    )
    (pipeline_root / ".qualification_pipeline_complete").write_text("complete\n", encoding="utf-8")
    return capture_root


def test_same_capture_lineage_packet_proves_one_local_chain(tmp_path: Path) -> None:
    capture_root = _build_capture_chain(tmp_path)

    packet = build_same_capture_lineage_packet(
        capture_root=capture_root,
        paperclip_issue_id="PC-123",
    )

    assert packet["schema_version"] == "same_capture_lineage_packet.v1"
    assert packet["status"] == "repo_proven"
    assert packet["capture_id"] == "capture-001"
    assert packet["raw_bundle"]["upload_completion"]["exists"] is True
    assert packet["bridge_handoff"]["pipeline_handoff_exists"] is True
    assert packet["pipeline_result"]["same_capture"] is True
    assert packet["webapp_upstream_ids"]["upstream_links_verified"] is True
    assert packet["paperclip_issue"]["issue_id"] == "PC-123"
    assert packet["claims"]["hosted_review_claim_allowed"] is True
    assert packet["claims"]["world_model_ready_claim_allowed"] is True
    assert packet["claims"]["launch_claim_allowed"] is False
    assert packet["repo_blockers"] == []
    assert "live_provider_runtime_payment_proof_not_in_repo_packet" in packet["remaining_runtime_gaps"]


def test_missing_webapp_ids_block_hosted_review_and_launch_claims(tmp_path: Path) -> None:
    capture_root = _build_capture_chain(tmp_path, include_webapp_ids=False)

    packet = build_same_capture_lineage_packet(
        capture_root=capture_root,
        paperclip_issue_id="PC-123",
    )

    assert packet["status"] == "blocked"
    assert packet["claims"]["hosted_review_claim_allowed"] is False
    assert packet["claims"]["launch_claim_allowed"] is False
    assert "missing_webapp_site_submission_id" in packet["repo_blockers"]
    assert "missing_webapp_request_id" in packet["repo_blockers"]
    assert "missing_webapp_buyer_request_id" in packet["repo_blockers"]
    assert "missing_webapp_capture_job_id" in packet["repo_blockers"]


def test_fallback_geometry_cannot_be_world_model_ready_proof(tmp_path: Path) -> None:
    capture_root = _build_capture_chain(
        tmp_path,
        geometry_source="fallback_geometry",
        fallback_used=True,
    )

    packet = build_same_capture_lineage_packet(
        capture_root=capture_root,
        paperclip_issue_id="PC-123",
    )

    assert packet["status"] == "blocked"
    assert packet["pipeline_result"]["geometry"]["ready_for_world_model"] is True
    assert packet["claims"]["world_model_ready_claim_allowed"] is False
    assert packet["claims"]["launch_claim_allowed"] is False
    assert "fallback_geometry_not_world_model_ready" in packet["repo_blockers"]


def test_android_xr_video_only_keeps_public_readiness_blocked(tmp_path: Path) -> None:
    capture_root = _build_capture_chain(
        tmp_path,
        capture_id="capture-xr-001",
        scene_id="scene-xr-001",
        capture_profile_id="android_xr_glasses",
        capture_modality="android_xr_video_only",
    )

    packet = build_same_capture_lineage_packet(
        capture_root=capture_root,
        paperclip_issue_id="PC-XR-123",
    )

    assert packet["status"] == "blocked"
    assert packet["claims"]["android_xr_public_readiness_claim_allowed"] is False
    assert packet["claims"]["public_readiness_claim_allowed"] is False
    assert "android_xr_video_only_requires_physical_hardware_proof" in packet["remaining_hardware_gaps"]
    assert "android_xr_video_only_requires_explicit_geometry_contract" in packet["repo_blockers"]


def test_same_capture_lineage_writer_persists_packet(tmp_path: Path) -> None:
    capture_root = _build_capture_chain(tmp_path)

    path = write_same_capture_lineage_packet(capture_root=capture_root, paperclip_issue_id="PC-123")

    packet = json.loads(path.read_text(encoding="utf-8"))
    assert path == capture_root / "pipeline" / "same_capture_lineage_packet.json"
    assert packet["status"] == "repo_proven"
    assert packet["capture_id"] == "capture-001"


def test_same_capture_lineage_validation_rejects_bad_claims() -> None:
    summary = validate_same_capture_lineage_packet(
        {
            "schema_version": "wrong",
            "status": "unexpected",
            "capture_id": "",
            "repo_blockers": "not-a-list",
            "claims": {
                "launch_claim_allowed": True,
                "world_model_ready_claim_allowed": True,
            },
            "pipeline_result": {"geometry": {"exists": False}},
        }
    )

    assert summary["status"] == "blocked"
    assert "same_capture_lineage_schema_version_invalid" in summary["blockers"]
    assert "same_capture_lineage_status_invalid" in summary["blockers"]
    assert "same_capture_lineage_capture_id_missing" in summary["blockers"]
    assert "same_capture_lineage_repo_blockers_missing" in summary["blockers"]
    assert "same_capture_lineage_launch_claim_must_remain_false" in summary["blockers"]
    assert "same_capture_lineage_world_model_claim_not_backed_by_live_geometry" in summary["blockers"]


def test_same_capture_lineage_low_level_missing_and_mismatch_packets(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert lineage._read_json(tmp_path / "missing.json") == {}
    assert lineage._read_json(invalid_json) == {}

    identity_missing = lineage._identity_blockers(
        identity_sources=[],
        primary_scene_id="",
        primary_capture_id="",
    )
    assert "same_capture_scene_id_missing" in identity_missing
    assert "same_capture_capture_id_missing" in identity_missing
    identity_mismatch = lineage._identity_blockers(
        identity_sources=[{"name": "source", "scene_id": "other-scene", "capture_id": "other-capture"}],
        primary_scene_id="scene-1",
        primary_capture_id="capture-1",
    )
    assert "same_capture_scene_id_mismatch:source" in identity_mismatch
    assert "same_capture_capture_id_mismatch:source" in identity_mismatch

    raw_missing = lineage._raw_bundle_packet(
        raw_root=tmp_path / "raw",
        manifest={},
        capture_context={},
        upload_completion={},
        primary_scene_id="scene-1",
        primary_capture_id="capture-1",
    )
    assert raw_missing["blockers"] == [
        "raw_manifest_missing",
        "raw_capture_context_missing",
        "raw_capture_upload_completion_missing",
    ]
    raw_mismatch = lineage._raw_bundle_packet(
        raw_root=tmp_path / "raw",
        manifest={"scene_id": "scene-1", "capture_id": "capture-1"},
        capture_context={"scene_id": "scene-1", "capture_id": "capture-1"},
        upload_completion={"scene_id": "other-scene", "capture_id": "other-capture"},
        primary_scene_id="scene-1",
        primary_capture_id="capture-1",
    )
    assert "raw_capture_upload_completion_capture_id_mismatch" in raw_mismatch["blockers"]
    assert "raw_capture_upload_completion_scene_id_mismatch" in raw_mismatch["blockers"]

    bridge_missing = lineage._bridge_handoff_packet(
        descriptor={},
        qa_report={},
        pipeline_handoff={},
        primary_scene_id="scene-1",
        primary_capture_id="capture-1",
    )
    assert bridge_missing["blockers"] == [
        "bridge_capture_descriptor_missing",
        "bridge_qa_report_missing",
        "bridge_pipeline_handoff_missing",
    ]
    bridge_mismatch = lineage._bridge_handoff_packet(
        descriptor={"scene_id": "other-scene", "capture_id": "other-capture"},
        qa_report={"scene_id": "scene-1", "capture_id": "capture-1"},
        pipeline_handoff={"scene_id": "scene-1", "capture_id": "capture-1"},
        primary_scene_id="scene-1",
        primary_capture_id="capture-1",
    )
    assert "bridge_capture_descriptor_capture_id_mismatch" in bridge_mismatch["blockers"]
    assert "bridge_capture_descriptor_scene_id_mismatch" in bridge_mismatch["blockers"]

    pipeline_missing = lineage._pipeline_result_packet(
        pipeline_root=tmp_path / "pipeline",
        opportunity_handoff={},
        qualification_summary={},
        geometry_summary={},
        primary_scene_id="scene-1",
        primary_capture_id="capture-1",
    )
    assert "pipeline_opportunity_handoff_missing" in pipeline_missing["blockers"]
    assert "pipeline_qualification_summary_missing" in pipeline_missing["blockers"]
    assert "pipeline_completion_marker_missing" in pipeline_missing["blockers"]
    pipeline_mismatch = lineage._pipeline_result_packet(
        pipeline_root=tmp_path / "pipeline",
        opportunity_handoff={"scene_id": "other-scene", "capture_id": "other-capture"},
        qualification_summary={"scene_id": "scene-1", "capture_id": "capture-1"},
        geometry_summary={"scene_id": "scene-1", "capture_id": "capture-1"},
        primary_scene_id="scene-1",
        primary_capture_id="capture-1",
    )
    assert "pipeline_opportunity_handoff_capture_id_mismatch" in pipeline_mismatch["blockers"]
    assert "pipeline_opportunity_handoff_scene_id_mismatch" in pipeline_mismatch["blockers"]


def test_same_capture_lineage_webapp_payload_edges() -> None:
    packet = lineage._webapp_upstream_ids_packet(
        webapp_sync={
            "status": "failed",
            "latest_stage": "qualification",
            "syncs": {
                "qualification": {
                    "attachment_payload": {
                        "scene_id": "other-scene",
                        "capture_id": "other-capture",
                        "site_submission_id": "example-submission",
                        "request_id": "scene-1/capture-1",
                        "buyer_request_id": "buyer-request-1",
                        "capture_job_id": "capture-1",
                        "upstream_links_verified": False,
                        "missing_upstream_links": ("buyer_request_id",),
                    }
                }
            },
        },
        opportunity_handoff={},
        descriptor={},
        manifest={},
        primary_scene_id="scene-1",
        primary_capture_id="capture-1",
    )

    assert "placeholder_webapp_site_submission_id" in packet["blockers"]
    assert "generated_capture_id_used_for_webapp_request_id" in packet["blockers"]
    assert "generated_capture_id_used_for_webapp_capture_job_id" in packet["blockers"]
    assert "webapp_sync_capture_id_mismatch" in packet["blockers"]
    assert "webapp_sync_scene_id_mismatch" in packet["blockers"]
    assert "webapp_sync_missing_buyer_request_id" in packet["blockers"]

    latest_payload, latest_source = lineage._latest_webapp_sync_payload(
        {
            "syncs": {
                "first": {"attachment_payload": {"request_id": "first"}},
                "second": {"attachment_payload": {"request_id": "second"}},
            }
        }
    )
    assert latest_payload["request_id"] == "second"
    assert latest_source == "pipeline.webapp_sync_result:latest"
    raw_payload, raw_source = lineage._latest_webapp_sync_payload({"request_id": "raw"})
    assert raw_payload["request_id"] == "raw"
    assert raw_source == "pipeline.webapp_sync_result"


def test_same_capture_lineage_hardware_gap_and_string_list_edges() -> None:
    assert (
        lineage._remaining_hardware_gaps(
            profile={},
            manifest={"physical_device_proof": {"physical_device_smoke_passed": True, "evidence_uri": "gs://proof"}},
            descriptor={},
        )
        == []
    )
    assert lineage._remaining_hardware_gaps(
        profile={"capture_profile_id": "meta_glasses", "capture_source": "meta"},
        manifest={},
        descriptor={},
    ) == ["glasses_physical_device_proof_not_in_repo_packet"]
    assert lineage._string_list(("one", "", "two")) == ["one", "two"]
    assert lineage._string_list("one") == []
