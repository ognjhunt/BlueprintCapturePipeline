from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from blueprint_pipeline.provider_preview_qa import validate_provider_preview_packet


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
