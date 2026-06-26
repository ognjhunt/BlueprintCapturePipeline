from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from blueprint_pipeline.provider_preview_qa import (
    _latest_webapp_sync_stage,
    _string_list,
    main,
    validate_provider_preview_packet,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(root / "capture_descriptor.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    _write_json(root / "raw" / "manifest.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    return root


def _uri(path: str) -> str:
    return f"gs://local-blueprint/scenes/scene-1/captures/capture-1/{path}"


def _write_privacy_safe_packet(root: Path) -> None:
    (root / "privacy").mkdir(parents=True, exist_ok=True)
    (root / "privacy" / "final_walkthrough.mov").write_bytes(b"privacy-safe-video")
    (root / "pipeline" / "worldlabs_input").mkdir(parents=True, exist_ok=True)
    (root / "pipeline" / "worldlabs_input" / "worldlabs_input.mp4").write_bytes(
        b"prepared-worldlabs-input"
    )
    output_checksum = sha256(b"prepared-worldlabs-input").hexdigest()
    source_checksum = sha256(b"privacy-safe-video").hexdigest()
    selected_uri = _uri("pipeline/worldlabs_input/worldlabs_input.mp4")

    _write_json(
        root / "pipeline" / "privacy_processing_manifest.json",
        {
            "schema_version": "v1",
            "status": "person_removed",
            "fail_closed": True,
            "people_detected": 1,
            "people_removed": 1,
            "depth_source": "depth_anything",
            "privacy_processed_video_uri": _uri("privacy/final_walkthrough.mov"),
            "world_model_video_uri": _uri("privacy/final_walkthrough.mov"),
        },
    )
    _write_json(
        root / "pipeline" / "privacy_verification_report.json",
        {"schema_version": "v1", "status": "passed"},
    )
    _write_json(
        root / "pipeline" / "worldlabs_input" / "worldlabs_input_manifest.json",
        {
            "schema_version": "v1",
            "status": "ready",
            "selected_video_source_id": "privacy_processed_video_uri",
            "selected_video_uri": _uri("privacy/final_walkthrough.mov"),
            "output_video_uri": selected_uri,
            "input_labeling": {
                "privacy_safe_input": True,
                "raw_video_bypass_used": False,
                "review_state": "standard_privacy_safe_preview",
            },
        },
    )
    _write_json(
        root / "pipeline" / "worldlabs_input_audit.json",
        {
            "schema_version": "v1",
            "status": "ready",
            "selected_video_source_id": "privacy_processed_video_uri",
            "selected_video_uri": _uri("privacy/final_walkthrough.mov"),
            "source_manifest_uri": _uri("pipeline/privacy_processing_manifest.json"),
            "source_checksum_sha256": source_checksum,
            "source_is_final_walkthrough": True,
            "derivative_of_final_walkthrough": True,
            "privacy_safe_input": True,
            "raw_video_bypass_used": False,
            "output_video_uri": selected_uri,
            "output_checksum_sha256": output_checksum,
        },
    )
    _write_json(
        root / "pipeline" / "site_package" / "canonical_site_package.json",
        {
            "package_type": "BlueprintCanonicalSitePackage",
            "conditioning": {
                "rgb_video": {
                    "privacy_safe_world_model_input": {
                        "uri": selected_uri,
                        "checksum_sha256": output_checksum,
                    }
                }
            },
        },
    )
    _write_json(
        root
        / "pipeline"
        / "site_package"
        / "provider_adapter_inputs"
        / "world_labs_marble.json",
        {
            "provider": "world_labs",
            "adapter": "marble",
            "status": "ready",
            "conditioning_inputs": {
                "rgb_video": {
                    "uri": selected_uri,
                    "privacy_safe": True,
                    "input_audit_uri": _uri("pipeline/worldlabs_input_audit.json"),
                    "source_manifest_uri": _uri("pipeline/privacy_processing_manifest.json"),
                    "checksum_sha256": output_checksum,
                    "source_checksum_sha256": source_checksum,
                }
            },
        },
    )
    _write_json(
        root / "pipeline" / "worldlabs_request_manifest.json",
        {
            "schema_version": "v1",
            "provider_name": "world_labs",
            "provider_model": "marble-1.1",
            "status": "ready_for_generation",
            "selected_video_source_id": "privacy_safe_world_model_input",
            "selected_video_uri": selected_uri,
            "selected_input_checksum_sha256": output_checksum,
            "source_input_checksum_sha256": source_checksum,
            "source_manifest_uri": _uri("pipeline/privacy_processing_manifest.json"),
            "worldlabs_input_audit_uri": _uri("pipeline/worldlabs_input_audit.json"),
            "privacy_safe_input": True,
            "input_labeling": {
                "privacy_safe_input": True,
                "raw_video_bypass_used": False,
            },
            "input_audit": {
                "privacy_safe_input": True,
                "raw_video_bypass_used": False,
                "source_manifest_uri": _uri("pipeline/privacy_processing_manifest.json"),
                "output_video_uri": selected_uri,
                "output_checksum_sha256": output_checksum,
                "source_checksum_sha256": source_checksum,
            },
        },
    )
    _write_json(
        root / "pipeline" / "provider_preview_status.json",
        {
            "schema_version": "v1",
            "status": "ready_for_generation",
            "provider_name": "world_labs",
            "site_submission_id": "site-submission-1",
            "request_id": "request-1",
            "buyer_request_id": "buyer-request-1",
            "capture_job_id": "capture-job-1",
        },
    )
    _write_json(
        root / "pipeline" / "provider_run_manifest.json",
        {
            "schema_version": "v1",
            "status": "ready_for_generation",
            "provider_name": "world_labs",
            "site_submission_id": "site-submission-1",
            "request_id": "request-1",
            "buyer_request_id": "buyer-request-1",
            "capture_job_id": "capture-job-1",
        },
    )
    _write_json(
        root / "pipeline" / "webapp_sync_result.json",
        {
            "status": "skipped",
            "latest_stage": "qualification",
            "syncs": {
                "qualification": {
                    "status": "skipped",
                    "reason": "sync_not_configured",
                    "attachment_payload": {
                        "scene_id": "scene-1",
                        "capture_id": "capture-1",
                        "site_submission_id": "site-submission-1",
                        "request_id": "request-1",
                        "buyer_request_id": "buyer-request-1",
                        "capture_job_id": "capture-job-1",
                        "upstream_links_verified": True,
                        "missing_upstream_links": [],
                    },
                    "buyer_access_check": {
                        "buyer_access_checked": False,
                        "buyer_accessible": False,
                    },
                }
            },
        },
    )


def test_provider_preview_qa_passes_privacy_safe_worldlabs_input_packet(tmp_path: Path) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)

    result = validate_provider_preview_packet(capture_root=root, mode="production")
    manifest = _read_json(root / "pipeline" / "provider_preview_qa_manifest.json")

    assert result["status"] == "passed"
    assert result["claim_ceiling"] == "provider_proof_pending"
    assert result["raw_path_policy"]["privacy_safe_input"] is True
    assert result["raw_path_policy"]["raw_video_bypass_used"] is False
    assert result["provider_operation_proof"]["status"] == "pending"
    assert "owner_gpu_simulator_execution_proof" in result["next_required_live_gates"]
    assert manifest["status"] == "passed"


def test_provider_preview_qa_accepts_local_full_frame_redaction_status(tmp_path: Path) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    privacy_manifest = _read_json(root / "pipeline" / "privacy_processing_manifest.json")
    privacy_manifest["status"] = "full_frame_redacted_local_proof"
    privacy_manifest["mode"] = "full_frame_redaction"
    privacy_manifest["local_repo_proof_only"] = True
    privacy_manifest["production_review_required"] = True
    _write_json(root / "pipeline" / "privacy_processing_manifest.json", privacy_manifest)
    verification = _read_json(root / "pipeline" / "privacy_verification_report.json")
    verification["status"] = "full_frame_redacted_local_proof"
    _write_json(root / "pipeline" / "privacy_verification_report.json", verification)

    result = validate_provider_preview_packet(capture_root=root, mode="production")

    assert result["status"] == "passed"
    assert result["redaction_proof"]["privacy_completed"] is True


def test_provider_preview_qa_blocks_raw_bypass_in_production(tmp_path: Path, monkeypatch) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS", "true")
    raw_uri = _uri("raw/walkthrough.mov")
    _write_json(
        root / "pipeline" / "worldlabs_input" / "worldlabs_input_manifest.json",
        {
            "schema_version": "v1",
            "status": "ready",
            "selected_video_source_id": "raw_video_uri",
            "selected_video_uri": raw_uri,
            "output_video_uri": _uri("pipeline/worldlabs_input/worldlabs_input.mp4"),
            "input_labeling": {"raw_video_bypass_used": True},
        },
    )
    _write_json(
        root / "pipeline" / "worldlabs_input_audit.json",
        {
            "schema_version": "v1",
            "status": "review_required",
            "selected_video_source_id": "raw_video_uri",
            "selected_video_uri": raw_uri,
            "privacy_safe_input": False,
            "raw_video_bypass_used": True,
            "output_video_uri": _uri("pipeline/worldlabs_input/worldlabs_input.mp4"),
            "output_checksum_sha256": "raw-sha",
        },
    )
    request = _read_json(root / "pipeline" / "worldlabs_request_manifest.json")
    request["privacy_safe_input"] = False
    request["selected_video_source_id"] = "raw_video_uri"
    request["input_labeling"] = {"raw_video_bypass_used": True}
    request["input_audit"] = {
        "privacy_safe_input": False,
        "raw_video_bypass_used": True,
        "output_video_uri": _uri("pipeline/worldlabs_input/worldlabs_input.mp4"),
    }
    _write_json(root / "pipeline" / "worldlabs_request_manifest.json", request)

    result = validate_provider_preview_packet(capture_root=root, mode="production")

    assert result["status"] == "blocked"
    assert "raw_worldlabs_bypass_env_enabled_in_production" in result["blockers"]
    assert "raw_video_bypass_used" in result["blockers"]
    assert "worldlabs_request_not_privacy_safe" in result["blockers"]
    assert result["raw_path_policy"]["production_mode_blocks_raw_bypass"] is True


def test_provider_preview_qa_requires_final_walkthrough_derivative_audit(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    audit = _read_json(root / "pipeline" / "worldlabs_input_audit.json")
    audit["source_is_final_walkthrough"] = False
    audit["derivative_of_final_walkthrough"] = False
    _write_json(root / "pipeline" / "worldlabs_input_audit.json", audit)

    result = validate_provider_preview_packet(capture_root=root, mode="production")

    assert result["status"] == "blocked"
    assert "worldlabs_input_not_final_walkthrough_derivative" in result["blockers"]


def test_provider_preview_qa_propagates_blocked_worldlabs_request(tmp_path: Path) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    request = _read_json(root / "pipeline" / "worldlabs_request_manifest.json")
    request["status"] = "blocked"
    request["blockers"] = ["rights_provenance_review_blocked"]
    _write_json(root / "pipeline" / "worldlabs_request_manifest.json", request)

    result = validate_provider_preview_packet(capture_root=root, mode="production")

    assert result["status"] == "blocked"
    assert "worldlabs_request_blocked" in result["blockers"]
    assert "rights_provenance_review_blocked" in result["blockers"]


def test_provider_preview_qa_requires_real_webapp_upstream_ids(tmp_path: Path) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    webapp_sync = _read_json(root / "pipeline" / "webapp_sync_result.json")
    qualification = webapp_sync["syncs"]["qualification"]  # type: ignore[index]
    attachment = qualification["attachment_payload"]  # type: ignore[index]
    attachment["buyer_request_id"] = ""  # type: ignore[index]
    attachment["missing_upstream_links"] = ["buyer_request_id"]  # type: ignore[index]
    attachment["upstream_links_verified"] = False  # type: ignore[index]
    _write_json(root / "pipeline" / "webapp_sync_result.json", webapp_sync)

    result = validate_provider_preview_packet(
        capture_root=root,
        mode="production",
        require_webapp_sync=True,
    )

    assert result["status"] == "blocked"
    assert "webapp_sync_missing_real_upstream_ids" in result["blockers"]
    assert "missing_webapp_buyer_request_id" in result["blockers"]
    assert "webapp_sync_upstream_links_not_verified" in result["blockers"]
    assert result["webapp_sync_projection"]["missing_upstream_ids"] == ["buyer_request_id"]


def test_provider_preview_qa_blocks_skipped_webapp_sync_when_required(tmp_path: Path) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)

    result = validate_provider_preview_packet(
        capture_root=root,
        mode="production",
        require_webapp_sync=True,
    )

    assert result["status"] == "blocked"
    assert "webapp_sync_not_succeeded" in result["blockers"]
    assert "webapp_sync_skipped_not_live" in result["blockers"]
    assert result["webapp_sync_projection"]["sync_succeeded"] is False
    assert result["webapp_sync_projection"]["upstream_links_verified"] is True


def test_provider_preview_qa_passes_required_webapp_sync_only_when_succeeded(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    webapp_sync = _read_json(root / "pipeline" / "webapp_sync_result.json")
    webapp_sync["status"] = "succeeded"
    qualification = webapp_sync["syncs"]["qualification"]  # type: ignore[index]
    qualification["status"] = "succeeded"  # type: ignore[index]
    qualification["response"] = {"ok": True, "requestId": "request-1"}  # type: ignore[index]
    _write_json(root / "pipeline" / "webapp_sync_result.json", webapp_sync)

    result = validate_provider_preview_packet(
        capture_root=root,
        mode="production",
        require_webapp_sync=True,
    )

    assert result["status"] == "passed"
    assert result["webapp_sync_projection"]["sync_succeeded"] is True
    assert result["webapp_sync_projection"]["upstream_links_verified"] is True


def test_provider_preview_qa_uses_descriptor_fallback_for_failed_sync_payload(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    _write_json(
        root / "pipeline" / "provider_preview_status.json",
        {"schema_version": "v1", "status": "ready_for_generation", "provider_name": "world_labs"},
    )
    _write_json(
        root / "pipeline" / "provider_run_manifest.json",
        {"schema_version": "v1", "status": "ready_for_generation", "provider_name": "world_labs"},
    )
    _write_json(
        root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_submission_id": "site-submission-1",
            "capture_job_id": "capture-job-1",
        },
    )
    _write_json(
        root / "pipeline" / "opportunity_handoff.json",
        {
            "site_submission_id": "site-submission-1",
            "buyer_request_id": "",
            "capture_job_id": "capture-job-1",
            "upstream_link_truth_state": "blocked_missing_upstream_ids",
            "upstream_link_blockers": ["missing_buyer_request_id"],
        },
    )
    _write_json(
        root / "pipeline" / "webapp_sync_result.json",
        {
            "status": "failed",
            "reason": "missing_upstream_pipeline_records: buyer_request_id",
            "blocker": "webapp_sync_requires_upstream_request_job_bootstrap",
        },
    )

    result = validate_provider_preview_packet(
        capture_root=root,
        mode="production",
        require_webapp_sync=True,
    )

    assert result["status"] == "blocked"
    assert result["webapp_sync_projection"]["upstream_ids"] == {
        "site_submission_id": "site-submission-1",
        "request_id": "site-submission-1",
        "buyer_request_id": "",
        "capture_job_id": "capture-job-1",
    }
    assert result["webapp_sync_projection"]["missing_upstream_ids"] == [
        "buyer_request_id"
    ]
    assert "missing_webapp_buyer_request_id" in result["blockers"]
    assert "missing_webapp_site_submission_id" not in result["blockers"]


def test_provider_preview_qa_uses_webapp_route_proof_ids_without_live_claim(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    _write_json(
        root / "pipeline" / "provider_preview_status.json",
        {
            "schema_version": "v1",
            "status": "ready_for_generation",
            "provider_name": "world_labs",
        },
    )
    _write_json(
        root / "pipeline" / "provider_run_manifest.json",
        {
            "schema_version": "v1",
            "status": "ready_for_generation",
            "provider_name": "world_labs",
        },
    )
    _write_json(
        root / "pipeline" / "webapp_sync_result.json",
        {
            "status": "failed",
            "reason": "site_submission_id or request_id is required",
            "blocker": "webapp_sync_requires_upstream_request_job_bootstrap",
        },
    )
    _write_json(
        root / "pipeline" / "webapp_route_forwarding_proof" / "webapp_route_forwarding_proof.json",
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "status": "forwarded_to_pipeline_intake",
            "job_request": {
                "job_id": "robot-eval-route-proof-1",
                "buyer_request_id": "buyer-request-route-proof-1",
                "site_package": {
                    "site_submission_id": "site-submission-route-proof-1",
                    "capture_job_id": "capture-job-route-proof-1",
                },
            },
            "proof_boundary": {
                "local_webapp_route_forwarding_proven": True,
                "pipeline_intake_staged_request_proven": True,
                "production_live_webapp_forwarding_proven": False,
                "simulator_execution_proven": False,
                "rank_fidelity_result_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )

    result = validate_provider_preview_packet(
        capture_root=root,
        mode="production",
        require_webapp_sync=True,
    )

    assert result["status"] == "blocked"
    assert result["webapp_sync_projection"]["sync_succeeded"] is False
    assert result["webapp_sync_projection"]["upstream_links_verified"] is True
    assert (
        result["webapp_sync_projection"]["upstream_links_verification_source"]
        == "local_webapp_route_forwarding_proof"
    )
    assert result["webapp_sync_projection"]["upstream_ids"] == {
        "site_submission_id": "site-submission-route-proof-1",
        "request_id": "robot-eval-route-proof-1",
        "buyer_request_id": "buyer-request-route-proof-1",
        "capture_job_id": "capture-job-route-proof-1",
    }
    assert result["webapp_sync_projection"]["missing_upstream_ids"] == []
    assert result["webapp_route_forwarding_projection"][
        "local_webapp_route_forwarding_proven"
    ] is True
    assert result["webapp_route_forwarding_projection"][
        "production_live_webapp_forwarding_proven"
    ] is False
    assert "webapp_sync_upstream_links_not_verified" not in result["blockers"]
    assert "production_live_webapp_forwarding_not_proven" in result["blockers"]
    assert "missing_webapp_site_submission_id" not in result["blockers"]
    assert "missing_webapp_request_id" not in result["blockers"]
    assert "missing_webapp_buyer_request_id" not in result["blockers"]
    assert "missing_webapp_capture_job_id" not in result["blockers"]
    assert "webapp_sync_failed" not in result["blockers"]
    assert "webapp_sync_not_succeeded" not in result["blockers"]
    assert "production_live_webapp_forwarding_not_proven" in result["blockers"]


def test_provider_preview_helper_branches_normalize_strings_and_latest_stage_fallback() -> None:
    assert _string_list("single") == ["single"]
    assert _string_list(7) == ["7"]
    assert _latest_webapp_sync_stage(
        {
            "latest_stage": "missing",
            "syncs": {
                "qualification": {"status": "failed"},
                "provider_preview": {"status": "succeeded"},
            },
        }
    ) == {"status": "succeeded"}


def test_provider_preview_qa_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode must be production or advisory"):
        validate_provider_preview_packet(capture_root=tmp_path / "capture", mode="review")


def test_provider_preview_qa_reports_missing_privacy_and_input_lineage(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)

    result = validate_provider_preview_packet(capture_root=root, mode="production")

    assert result["status"] == "blocked"
    assert "missing_privacy_final_walkthrough" in result["blockers"]
    assert "privacy_manifest_or_verification_not_complete" in result["blockers"]
    assert "privacy_output_not_final_walkthrough" in result["blockers"]
    assert "worldlabs_audit_output_uri_mismatch" in result["blockers"]
    assert "missing_worldlabs_input_audit_uri" in result["blockers"]
    assert "missing_worldlabs_source_manifest_uri" in result["blockers"]
    assert "missing_worldlabs_input_checksum" in result["blockers"]
    assert "canonical_package_missing_privacy_safe_rgb_video" in result["blockers"]
    assert "provider_adapter_missing_rgb_video" in result["blockers"]


def test_provider_preview_qa_flags_mismatched_artifacts_and_geometry_labels(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path / "mismatch")
    _write_privacy_safe_packet(root)
    selected_uri = _uri("pipeline/worldlabs_input/worldlabs_input.mp4")
    other_uri = _uri("pipeline/worldlabs_input/other.mp4")

    input_manifest = _read_json(root / "pipeline" / "worldlabs_input" / "worldlabs_input_manifest.json")
    input_manifest["output_video_uri"] = other_uri
    _write_json(root / "pipeline" / "worldlabs_input" / "worldlabs_input_manifest.json", input_manifest)

    input_audit = _read_json(root / "pipeline" / "worldlabs_input_audit.json")
    input_audit["output_video_uri"] = _uri("pipeline/worldlabs_input/audit-other.mp4")
    _write_json(root / "pipeline" / "worldlabs_input_audit.json", input_audit)

    canonical = _read_json(root / "pipeline" / "site_package" / "canonical_site_package.json")
    canonical["conditioning"]["rgb_video"]["privacy_safe_world_model_input"]["uri"] = other_uri  # type: ignore[index]
    _write_json(root / "pipeline" / "site_package" / "canonical_site_package.json", canonical)

    adapter = _read_json(
        root / "pipeline" / "site_package" / "provider_adapter_inputs" / "world_labs_marble.json"
    )
    adapter["conditioning_inputs"]["rgb_video"]["uri"] = other_uri  # type: ignore[index]
    _write_json(
        root / "pipeline" / "site_package" / "provider_adapter_inputs" / "world_labs_marble.json",
        adapter,
    )
    _write_json(
        root / "pipeline" / "geometry" / "geometry_summary.json",
        {"geometry_live_ready": True, "geometry_source": "local_sfm"},
    )

    result = validate_provider_preview_packet(capture_root=root, mode="production")

    assert result["status"] == "blocked"
    assert selected_uri != other_uri
    assert "worldlabs_audit_output_uri_mismatch" in result["blockers"]
    assert "worldlabs_input_manifest_output_uri_mismatch" in result["blockers"]
    assert "canonical_package_rgb_video_mismatch" in result["blockers"]
    assert "provider_adapter_rgb_video_mismatch" in result["blockers"]
    assert "fallback_geometry_marked_live_ready" in result["blockers"]

    warning_root = _capture_root(tmp_path / "geometry-warning")
    _write_privacy_safe_packet(warning_root)
    _write_json(
        warning_root / "pipeline" / "geometry" / "geometry_summary.json",
        {"geometry_live_ready": False, "geometry_source": "video_to_world"},
    )

    warning_result = validate_provider_preview_packet(
        capture_root=warning_root,
        mode="production",
    )

    assert warning_result["status"] == "passed"
    assert "geometry_present_but_not_live_ready" in warning_result["warnings"]


def test_provider_preview_qa_reports_missing_sync_and_placeholder_ids(
    tmp_path: Path,
) -> None:
    missing_sync_root = _capture_root(tmp_path / "missing-sync")
    _write_privacy_safe_packet(missing_sync_root)
    (missing_sync_root / "pipeline" / "webapp_sync_result.json").unlink()

    required_result = validate_provider_preview_packet(
        capture_root=missing_sync_root,
        mode="production",
        require_webapp_sync=True,
    )
    advisory_result = validate_provider_preview_packet(
        capture_root=missing_sync_root,
        mode="production",
        require_webapp_sync=False,
    )

    assert "missing_webapp_sync_result" in required_result["blockers"]
    assert "webapp_sync_not_required_or_not_present" in advisory_result["warnings"]

    placeholder_root = _capture_root(tmp_path / "placeholder-sync")
    _write_privacy_safe_packet(placeholder_root)
    webapp_sync = _read_json(placeholder_root / "pipeline" / "webapp_sync_result.json")
    webapp_sync["status"] = "succeeded"
    qualification = webapp_sync["syncs"]["qualification"]  # type: ignore[index]
    qualification["status"] = "succeeded"  # type: ignore[index]
    qualification["attachment_payload"] = {  # type: ignore[index]
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "site_submission_id": "example-site-submission",
        "request_id": "placeholder-request",
        "buyer_request_id": "test-buyer-request",
        "capture_job_id": "mock-capture-job",
        "upstream_links_verified": True,
        "missing_upstream_links": [],
    }
    _write_json(placeholder_root / "pipeline" / "webapp_sync_result.json", webapp_sync)

    placeholder_result = validate_provider_preview_packet(
        capture_root=placeholder_root,
        mode="production",
        require_webapp_sync=True,
    )

    assert "webapp_sync_placeholder_upstream_ids" in placeholder_result["blockers"]
    assert "placeholder_webapp_site_submission_id" in placeholder_result["blockers"]
    assert "placeholder_webapp_request_id" in placeholder_result["blockers"]
    assert "placeholder_webapp_buyer_request_id" in placeholder_result["blockers"]
    assert "placeholder_webapp_capture_job_id" in placeholder_result["blockers"]
    assert "webapp_sync_upstream_links_not_verified" in placeholder_result["blockers"]
    assert placeholder_result["webapp_sync_projection"]["placeholder_upstream_ids"] == [
        "site_submission_id",
        "request_id",
        "buyer_request_id",
        "capture_job_id",
    ]


def test_provider_preview_qa_sets_local_repo_claim_when_operation_world_manifests_are_current(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_safe_packet(root)
    _write_json(
        root / "pipeline" / "worldlabs_operation_manifest.json",
        {"schema_version": "v1", "status": "completed", "operation_id": "operation-1"},
    )
    _write_json(
        root / "pipeline" / "worldlabs_world_manifest.json",
        {"schema_version": "v1", "status": "ready", "world_id": "world-1"},
    )

    result = validate_provider_preview_packet(capture_root=root, mode="production")

    assert result["status"] == "passed"
    assert result["claim_ceiling"] == "local_repo_proof"
    assert result["provider_operation_proof"]["status"] == "proven"


def test_provider_preview_qa_main_reports_error_blocked_and_passed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--capture-root", str(tmp_path / "not-a-capture")]) == 2
    assert "status=error" in capsys.readouterr().out

    blocked_root = _capture_root(tmp_path / "blocked")
    assert main(["--capture-root", str(blocked_root)]) == 1
    blocked_output = capsys.readouterr().out
    assert "status=blocked" in blocked_output
    assert "blockers=missing_privacy_final_walkthrough" in blocked_output

    passed_root = _capture_root(tmp_path / "passed")
    _write_privacy_safe_packet(passed_root)
    assert main(["--capture-root", str(passed_root), "--mode", "advisory"]) == 0
    passed_output = capsys.readouterr().out
    assert "provider_preview_qa_manifest.json" in passed_output
    assert "status=passed" in passed_output
