from __future__ import annotations

import io
import json
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.capture_upload_intake import (
    CaptureUploadTransferError,
    process_capture_upload_submission,
)


def _payload() -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + b"rights-cleared-capture-bytes"


def _submission(payload: bytes | None = None) -> dict:
    payload = payload or _payload()
    return {
        "schema_version": "capture_upload_transfer_submission.v1",
        "capture_session_id": "capture-upload-session-1",
        "customer_id": "firebase-buyer-1",
        "organization_id": "org-1",
        "request": {
            "schema_version": "capture_upload_session_request.v1",
            "intake_id": "intake-upload-1",
            "idempotency_key": "org-1-upload-1",
            "capture_authority_profile": "camera_360_equirectangular",
            "source_type": "camera_360_equirectangular",
            "scene_id": "scene-1",
            "original_file": {
                "original_filename": "capture.mp4",
                "size_bytes": len(payload),
                "media_type": "video/mp4",
            },
            "capture_device": {"manufacturer": "fixture", "model": "360"},
            "timing_declaration": {"clock": "media_pts"},
            "coordinate_frame_declaration": {"status": "not_available_from_video"},
            "available_sensor_streams": [
                {"stream_type": "retained_video", "status": "available"},
                {"stream_type": "camera_metadata", "status": "available"},
            ],
            "governance": {
                "rights": "accepted",
                "consent": "accepted",
                "privacy": "cleared",
                "retention": {"max_days": 30},
                "revocation": {
                    "supported": True,
                    "historical_tombstone_retained": True,
                },
                "provider_constraints": {"external_processing_allowed": False},
                "allowed_uses": ["evaluation"],
            },
            "requested_task_evaluation_run_audience": "design_partner",
            "known_task_specification": None,
            "calibration_board_dimensions": None,
            "operator_notes": [],
            "permitted_reconstruction_providers": ["local_only"],
            "permitted_evidence_uses": ["captured_observation", "task_discovery"],
        },
        "transfer": {
            "provider": "backblaze",
            "url": "https://download.example.test/file/private/capture.mp4",
            "authorization": "short-lived-download-grant",
            "expires_at_iso": (
                datetime.now(timezone.utc) + timedelta(minutes=10)
            ).isoformat(),
        },
    }


def _opener(payload: bytes, seen: dict[str, object]):
    def open_transfer(**kwargs):
        seen.update(kwargs)
        return closing(io.BytesIO(payload))

    return open_transfer


def _scanner(path: Path) -> dict:
    assert path.read_bytes() == _payload()
    return {"status": "passed", "scanner": "fixture-clam", "definition_version": "1"}


def test_streams_scans_hashes_and_materializes_without_persisting_transfer_grant(
    tmp_path: Path,
) -> None:
    payload = _payload()
    seen: dict[str, object] = {}
    first = process_capture_upload_submission(
        _submission(payload),
        store_root=tmp_path,
        allowed_hosts=["download.example.test"],
        transfer_opener=_opener(payload, seen),
        malware_scanner=_scanner,
    )

    assert first["admission_status"] == "accepted"
    assert first["state"] == "capture_accepted"
    assert first["capture_digest"].startswith("sha256:")
    assert first["proof_boundary"] == {
        "server_sha256_verified": True,
        "raw_input_content_addressed": True,
        "capture_qa_completed": False,
        "task_success_established": False,
        "physical_task_success_established": False,
        "deployment_or_safety_approved": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    assert seen["authorization"] == "short-lived-download-grant"
    stored_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.json")
    )
    assert "short-lived-download-grant" not in stored_text
    assert "download.example.test" not in stored_text

    replay_seen: dict[str, object] = {}
    replay = process_capture_upload_submission(
        _submission(payload),
        store_root=tmp_path,
        allowed_hosts=["download.example.test"],
        transfer_opener=_opener(payload, replay_seen),
        malware_scanner=_scanner,
    )
    assert replay["already_exists"] is True
    assert replay["capture_digest"] == first["capture_digest"]
    assert replay_seen == {}


def test_transfer_fails_closed_on_host_query_size_media_and_malware(
    tmp_path: Path,
) -> None:
    with pytest.raises(CaptureUploadTransferError, match="capture_transfer_url_not_allowed"):
        process_capture_upload_submission(
            {
                **_submission(),
                "transfer": {
                    **_submission()["transfer"],
                    "url": "https://download.example.test/file/capture.mp4?token=leak",
                },
            },
            store_root=tmp_path / "query",
            allowed_hosts=["download.example.test"],
            transfer_opener=_opener(_payload(), {}),
            malware_scanner=_scanner,
        )

    short = _payload()[:-1]
    with pytest.raises(CaptureUploadTransferError, match="capture_transfer_size_mismatch"):
        process_capture_upload_submission(
            _submission(),
            store_root=tmp_path / "size",
            allowed_hosts=["download.example.test"],
            transfer_opener=_opener(short, {}),
            malware_scanner=_scanner,
        )

    invalid_media = b"not-an-mp4" + b"x" * (len(_payload()) - 10)
    with pytest.raises(CaptureUploadTransferError, match="capture_media_shape_invalid"):
        process_capture_upload_submission(
            _submission(invalid_media),
            store_root=tmp_path / "media",
            allowed_hosts=["download.example.test"],
            transfer_opener=_opener(invalid_media, {}),
            malware_scanner=lambda _path: {"status": "passed", "scanner": "fixture"},
        )

    with pytest.raises(CaptureUploadTransferError, match="malware_detected"):
        process_capture_upload_submission(
            _submission(),
            store_root=tmp_path / "malware",
            allowed_hosts=["download.example.test"],
            transfer_opener=_opener(_payload(), {}),
            malware_scanner=lambda _path: (_ for _ in ()).throw(
                CaptureUploadTransferError(["malware_detected"])
            ),
        )


def test_receipt_idempotency_rejects_changed_request_binding(tmp_path: Path) -> None:
    payload = _payload()
    process_capture_upload_submission(
        _submission(payload),
        store_root=tmp_path,
        allowed_hosts=["download.example.test"],
        transfer_opener=_opener(payload, {}),
        malware_scanner=_scanner,
    )
    changed = _submission(payload)
    changed["request"]["scene_id"] = "scene-2"
    with pytest.raises(CaptureUploadTransferError, match="capture_upload_idempotency_conflict"):
        process_capture_upload_submission(
            changed,
            store_root=tmp_path,
            allowed_hosts=["download.example.test"],
            transfer_opener=_opener(payload, {}),
            malware_scanner=_scanner,
        )


def test_configured_scanner_is_required_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_CAPTURE_MALWARE_SCANNER_ARGV_JSON", raising=False)
    with pytest.raises(CaptureUploadTransferError, match="malware_scanner_not_configured"):
        process_capture_upload_submission(
            _submission(),
            store_root=tmp_path,
            allowed_hosts=["download.example.test"],
            transfer_opener=_opener(_payload(), {}),
        )


def test_receipt_is_valid_json_and_contains_no_ephemeral_fields(tmp_path: Path) -> None:
    payload = _payload()
    receipt = process_capture_upload_submission(
        _submission(payload),
        store_root=tmp_path,
        allowed_hosts=["download.example.test"],
        transfer_opener=_opener(payload, {}),
        malware_scanner=_scanner,
    )
    persisted = json.loads(next((tmp_path / "transfer_receipts").glob("*.json")).read_text())
    assert persisted == receipt
    assert "transfer" not in persisted
    assert "authorization" not in json.dumps(persisted)
