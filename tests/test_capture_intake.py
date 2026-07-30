from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

import blueprint_pipeline.materialization as materialization
from blueprint_pipeline.capture_intake import (
    CaptureIntakeError,
    build_capture_admission,
    materialize_capture_intake,
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _envelope(
    payload: bytes,
    *,
    profile: str = "monocular_video",
    streams: list[str] | None = None,
) -> dict:
    streams = streams or ["retained_video"]
    return {
        "schema_version": "capture_intake_envelope.v1",
        "intake_id": "intake-1",
        "idempotency_key": "org-1-upload-1",
        "capture_authority_profile": profile,
        "source_type": profile,
        "original_files": [
            {
                "original_filename": "capture.mp4",
                "relative_path": "capture.mp4",
                "sha256": _digest(payload),
                "size_bytes": len(payload),
                "media_type": "video/mp4",
            }
        ],
        "scene_id": "scene-1",
        "customer_id": "customer-1",
        "organization_id": "org-1",
        "capture_device": {
            "manufacturer": "fixture",
            "model": "fixture-camera",
            "firmware_version": "1",
            "app_version": "1",
        },
        "timing_declaration": {"clock": "media_pts", "monotonic_time_available": False},
        "coordinate_frame_declaration": {"status": "not_available_from_video"},
        "available_sensor_streams": [
            {
                "stream_type": stream,
                "status": "available",
                "source_relative_path": "capture.mp4",
            }
            for stream in streams
        ],
        "governance": {
            "rights": "accepted",
            "consent": "accepted",
            "privacy": "cleared",
            "retention": {"max_days": 30},
            "revocation": {"supported": True, "historical_tombstone_retained": True},
            "provider_constraints": {"external_processing_allowed": False},
            "allowed_uses": ["evaluation"],
        },
        "requested_task_evaluation_run_audience": "design_partner",
        "known_task_specification": None,
        "calibration_board_dimensions": None,
        "operator_notes": [],
        "permitted_reconstruction_providers": ["local_only"],
        "permitted_evidence_uses": ["captured_observation", "task_discovery"],
        "upload_validation": {"status": "passed"},
        "malware_content_validation": {"status": "passed", "scanner": "fixture-scanner"},
    }


def test_monocular_materialization_is_content_addressed_idempotent_and_reduced_authority(
    tmp_path: Path,
) -> None:
    payload = b"rights-cleared-real-upload-bytes"
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "capture.mp4").write_bytes(payload)
    store = tmp_path / "store"
    envelope = _envelope(payload)

    first = materialize_capture_intake(envelope, upload_root=upload, store_root=store)
    second = materialize_capture_intake(envelope, upload_root=upload, store_root=store)

    assert first.artifact_root == second.artifact_root
    assert first.admission["status"] == "accepted"
    assert first.admission["state"] == "capture_accepted"
    assert first.admission["claim_ceiling"]["task_candidate_discovery"] is True
    assert first.admission["claim_ceiling"]["metric_geometry"] is False
    assert first.admission["claim_ceiling"]["collision_geometry"] is False
    assert first.admission["claim_ceiling"]["physical_task_success"] is False
    assert (
        first.admission["claim_ceiling"]["comparative_policy_ranking_verdict"]
        == "thesis_not_supported"
    )
    object_path = store / first.content_objects[0]["object_path"]
    assert object_path.read_bytes() == payload
    assert (object_path.stat().st_mode & 0o222) == 0
    assert json.loads(
        (first.artifact_root / "capture_intake_object_manifest.json").read_text()
    )["raw_inputs_mutated"] is False


def test_intake_identifier_cannot_escape_the_content_addressed_store(tmp_path: Path) -> None:
    payload = b"video"
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "capture.mp4").write_bytes(payload)
    envelope = _envelope(payload)
    envelope["intake_id"] = "../escape"

    with pytest.raises(CaptureIntakeError, match="intake_id:invalid_path_identifier"):
        materialize_capture_intake(
            envelope,
            upload_root=upload,
            store_root=tmp_path / "store",
        )
    assert not (tmp_path / "escape").exists()


def test_capture_intake_schema_accepts_the_runtime_contract() -> None:
    schema_path = Path(__file__).parents[1] / "docs/schemas/capture_intake_envelope.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(_envelope(b"video"))


def test_iphone_lidar_missing_decoded_pts_and_retention_map_requests_exact_recapture() -> None:
    streams = [
        "retained_video",
        "camera_poses",
        "camera_intrinsics",
        "depth",
        "depth_confidence",
        "tracking_state",
        "coordinate_frame_semantics",
    ]
    admission = build_capture_admission(
        _envelope(b"video", profile="iphone_arkit_lidar", streams=streams)
    )

    assert admission["status"] == "recapture_required"
    assert admission["missing_required_streams"] == [
        "decoded_video_pts",
        "frame_retention_mapping",
    ]
    instructions = " ".join(row["instruction"] for row in admission["recapture_plan"])
    assert "presentation timestamps" in instructions
    assert "encoder omissions" in instructions
    assert admission["claim_ceiling"]["metric_geometry"] is False


def test_complete_iphone_lidar_profile_admits_metric_but_not_physics_or_physical_claims() -> None:
    streams = [
        "retained_video",
        "decoded_video_pts",
        "frame_retention_mapping",
        "camera_poses",
        "camera_intrinsics",
        "depth",
        "depth_confidence",
        "tracking_state",
        "coordinate_frame_semantics",
        "motion",
    ]
    admission = build_capture_admission(
        _envelope(b"video", profile="iphone_arkit_lidar", streams=streams)
    )

    assert admission["status"] == "accepted"
    assert admission["claim_ceiling"]["metric_geometry"] is True
    assert admission["claim_ceiling"]["collision_geometry"] is False
    assert admission["claim_ceiling"]["contact_or_articulation"] is False
    assert admission["claim_ceiling"]["deployment_readiness"] is False


def test_rights_and_malware_admission_fail_closed() -> None:
    envelope = _envelope(b"video")
    envelope["governance"]["rights"] = "unknown"
    envelope["malware_content_validation"]["status"] = "pending"

    admission = build_capture_admission(envelope)

    assert admission["status"] == "rejected"
    assert admission["state"] == "failed"
    assert admission["governance_blockers"] == [
        "malware_content_validation_not_passed",
        "rights_not_accepted",
    ]
    assert admission["claim_ceiling"]["capture_admitted"] is False


def test_materialization_rejects_digest_mismatch_and_path_traversal(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "capture.mp4").write_bytes(b"actual")
    envelope = _envelope(b"faked!")
    with pytest.raises(CaptureIntakeError, match="digest_mismatch"):
        materialize_capture_intake(envelope, upload_root=upload, store_root=tmp_path / "store")

    unsafe = _envelope(b"actual")
    unsafe["original_files"][0]["relative_path"] = "../capture.mp4"
    with pytest.raises(CaptureIntakeError, match="relative_path:unsafe"):
        materialize_capture_intake(unsafe, upload_root=upload, store_root=tmp_path / "store")


def test_idempotency_key_cannot_be_rebound_to_different_input(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    store = tmp_path / "store"
    first_payload = b"first"
    (upload / "capture.mp4").write_bytes(first_payload)
    materialize_capture_intake(_envelope(first_payload), upload_root=upload, store_root=store)

    second_payload = b"second"
    (upload / "capture.mp4").write_bytes(second_payload)
    rebound = _envelope(second_payload)
    rebound["scene_id"] = "scene-2"
    with pytest.raises(CaptureIntakeError, match="idempotency_key_reuse"):
        materialize_capture_intake(rebound, upload_root=upload, store_root=store)


def test_external_reconstruction_requires_source_capture_digest_and_stays_derived() -> None:
    envelope = _envelope(
        b"mesh", profile="precomputed_external_reconstruction", streams=["external_reconstruction"]
    )
    envelope["original_files"][0]["original_filename"] = "scene.glb"
    envelope["original_files"][0]["relative_path"] = "scene.glb"
    envelope["available_sensor_streams"][0]["source_relative_path"] = "scene.glb"
    with pytest.raises(CaptureIntakeError, match="source_capture_digest"):
        build_capture_admission(envelope)

    bound = copy.deepcopy(envelope)
    bound["source_capture_binding"] = {"source_capture_digest": "sha256:" + "a" * 64}
    admission = build_capture_admission(bound)
    assert admission["status"] == "accepted"
    assert admission["claim_ceiling"]["external_reconstruction_is_raw_capture_authority"] is False
    assert "derived_reconstruction" in " ".join(admission["reduced_authority_reasons"])


def test_admitted_monocular_intake_flows_into_existing_capture_materialization(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1" / "raw"
    raw.mkdir(parents=True)
    payload = b"customer-video"
    (raw / "customer-video.mp4").write_bytes(payload)
    envelope = _envelope(payload)
    envelope["original_files"][0]["original_filename"] = "customer-video.mp4"
    envelope["original_files"][0]["relative_path"] = "customer-video.mp4"
    envelope["available_sensor_streams"][0]["source_relative_path"] = "customer-video.mp4"
    admission = build_capture_admission(envelope)
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": "scene-1",
                "width": 1920,
                "height": 1080,
                "capture_profile_id": "monocular_video",
            }
        ),
        encoding="utf-8",
    )
    (raw / "capture_intake_envelope.json").write_text(
        json.dumps(envelope), encoding="utf-8"
    )
    (raw / "capture_intake_admission.json").write_text(
        json.dumps(admission), encoding="utf-8"
    )

    result = materialization.build_capture_bundle_records(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        gcs_root=tmp_path,
        write_frames_index=False,
    )

    assert result["descriptor"]["capture_modality"] == "monocular_video"
    assert result["descriptor"]["raw_video_uri"].endswith("customer-video.mp4")
    assert result["descriptor"]["metadata"]["capture_intake"]["status"] == "accepted"
    assert result["descriptor"]["metadata"]["capture_intake_byte_verification"] == {
        "status": "verified",
        "verified_original_object_count": 1,
    }
    assert result["qa_report"]["capture_intake"]["claim_ceiling"]["metric_geometry"] is False
