from __future__ import annotations

import io
import json
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.capture_lifecycle import (
    CaptureLifecycleError,
    apply_capture_lifecycle_action,
    inspect_capture_lifecycle,
    record_external_revocation_evidence,
    record_provider_deletion_evidence,
)
from blueprint_pipeline.capture_upload_intake import (
    CaptureUploadTransferError,
    process_capture_upload_submission,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV


def _payload() -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + b"rights-cleared-lifecycle-capture"


def _submission(*, session_id: str, intake_id: str, idempotency_key: str) -> dict:
    payload = _payload()
    return {
        "schema_version": "capture_upload_transfer_submission.v1",
        "capture_session_id": session_id,
        "customer_id": "customer-1",
        "organization_id": "org-1",
        "request": {
            "schema_version": "capture_upload_session_request.v1",
            "intake_id": intake_id,
            "idempotency_key": idempotency_key,
            "capture_authority_profile": "monocular_video",
            "source_type": "monocular_video",
            "scene_id": "scene-1",
            "original_file": {
                "original_filename": "capture.mp4",
                "size_bytes": len(payload),
                "media_type": "video/mp4",
            },
            "capture_device": {"manufacturer": "fixture", "model": "camera"},
            "timing_declaration": {"clock": "media_pts"},
            "coordinate_frame_declaration": {"status": "not_available_from_video"},
            "available_sensor_streams": [
                {"stream_type": "retained_video", "status": "available"}
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
            "permitted_evidence_uses": ["captured_observation"],
        },
        "transfer": {
            "provider": "backblaze",
            "url": "https://download.example.test/capture.mp4",
            "authorization": "ephemeral-grant",
            "expires_at_iso": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        },
    }


def _upload(store: Path, *, session_id: str, intake_id: str, idempotency_key: str) -> dict:
    payload = _payload()

    def opener(**_: object):
        return closing(io.BytesIO(payload))

    def scanner(path: Path) -> dict:
        assert path.read_bytes() == payload
        return {"status": "passed", "scanner": "fixture", "definition_version": "1"}

    def qa(envelope: dict, *, upload_root: Path) -> dict:
        assert upload_root.joinpath("capture.mp4").is_file()
        report = {
            "schema_version": "capture_qa_report.v1",
            "intake_id": envelope["intake_id"],
            "envelope_digest": envelope["envelope_digest"],
            "capture_authority_profile": "monocular_video",
            "status": "accepted",
            "state": "capture_accepted",
            "checks": [],
            "recapture_plan": [],
            "missing_evidence": ["metric_scale"],
            "required_analysis": [],
            "next_cheapest_experiment": None,
            "quality_observations_digest": None,
            "quality_analysis_errors": [],
            "claim_ceiling": {
                "physical_task_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
            "prohibited_claims": ["physical_task_success"],
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        }
        report["qa_report_digest"] = canonical_digest(report, digest_field="qa_report_digest")
        return report

    return process_capture_upload_submission(
        _submission(
            session_id=session_id,
            intake_id=intake_id,
            idempotency_key=idempotency_key,
        ),
        store_root=store,
        allowed_hosts=["download.example.test"],
        transfer_opener=opener,
        malware_scanner=scanner,
        qa_builder=qa,
    )


def _apply(
    store: Path,
    work: Path,
    receipt: dict,
    *,
    action: str = "consent_revoked",
    now: datetime | None = None,
) -> dict:
    return apply_capture_lifecycle_action(
        store_root=store,
        work_root=work,
        capture_session_id=receipt["capture_session_id"],
        intake_id=receipt["intake_id"],
        capture_digest=receipt["capture_digest"],
        envelope_digest=receipt["envelope_digest"],
        action=action,
        actor={"role": "operator", "identity": "operator-1"},
        idempotency_key=f"lifecycle-{receipt['intake_id']}",
        now=now,
    )


def test_revocation_deletes_exact_payload_and_blocks_reupload(tmp_path: Path) -> None:
    store = tmp_path / "store"
    work = tmp_path / "work"
    receipt = _upload(
        store,
        session_id="capture-session-1",
        intake_id="intake-1",
        idempotency_key="upload-1",
    )
    artifact_root = store / receipt["artifact_reference"]["uri"]
    object_path = next(path for path in (store / "objects").rglob("*") if path.is_file())
    task_session = (
        work
        / "task_candidate_control_plane"
        / "sessions"
        / receipt["capture_session_id"]
    )
    task_session.mkdir(parents=True)
    (task_session / "candidate.json").write_text("{}", encoding="utf-8")
    run_root = work / "task_evaluation_runs" / "runs" / "run-1"
    (run_root / "artifacts").mkdir(parents=True)
    (run_root / "artifacts" / "run_context.json").write_text(
        json.dumps(
            {
                "capture_session_id": receipt["capture_session_id"],
                "intake_id": receipt["intake_id"],
            }
        ),
        encoding="utf-8",
    )
    testbed_root = work / "maintained_site_task_testbeds" / "testbed-1" / "1"
    testbed_root.mkdir(parents=True)
    (testbed_root / "testbed.json").write_text(
        json.dumps({"capture_digest": receipt["capture_digest"]}), encoding="utf-8"
    )

    tombstone = _apply(store, work, receipt)

    assert tombstone["raw_object_deleted"] is True
    assert tombstone["historical_digest_binding_preserved"] is True
    assert tombstone["provider_deletion_status"] == "not_required"
    assert tombstone["proof_boundary"]["comparative_policy_ranking_verdict"] == (
        "thesis_not_supported"
    )
    assert not artifact_root.exists()
    assert not object_path.exists()
    assert not task_session.exists()
    assert not run_root.exists()
    assert not testbed_root.exists()
    assert not list((store / "transfer_receipts").glob("*.json"))
    assert tombstone["deleted_bound_work_product_file_count"] == 3
    assert len(tombstone["deleted_bound_work_product_content_digests"]) == 3
    assert receipt["capture_session_id"] not in json.dumps(tombstone)
    assert receipt["intake_id"] not in json.dumps(tombstone)
    assert tombstone["external_revocation_complete"] is False
    assert _apply(store, work, receipt)["already_exists"] is True
    inspection = inspect_capture_lifecycle(
        store_root=store,
        capture_session_id=receipt["capture_session_id"],
        intake_id=receipt["intake_id"],
    )
    assert inspection["state"] == "tombstoned"
    assert inspection["provider_deletion_complete"] is True
    assert inspection["lifecycle_complete"] is False
    for index, (action, target, method) in enumerate(
        (
            (
                "sync_webapp_revocation_verdict",
                "Blueprint-WebApp",
                "signed_webapp_receipt",
            ),
            (
                "disable_signed_download_access",
                "capture-object-store",
                "storage_access_revocation_receipt",
            ),
        ),
        start=1,
    ):
        record_external_revocation_evidence(
            store_root=store,
            capture_session_id=receipt["capture_session_id"],
            intake_id=receipt["intake_id"],
            action=action,
            target_system=target,
            receipt_digest="sha256:" + str(index) * 64,
            completed_at="2026-07-30T12:00:00+00:00",
            verification_method=method,
            idempotency_key=f"external-revocation-{index}",
        )
    completed_lifecycle = inspect_capture_lifecycle(
        store_root=store,
        capture_session_id=receipt["capture_session_id"],
        intake_id=receipt["intake_id"],
    )
    assert completed_lifecycle["external_revocation_complete"] is True
    assert completed_lifecycle["lifecycle_complete"] is True

    with pytest.raises(CaptureUploadTransferError, match="capture_upload_revoked_or_expired"):
        _upload(
            store,
            session_id="capture-session-1",
            intake_id="intake-1",
            idempotency_key="upload-1",
        )


def test_shared_content_addressed_object_survives_until_last_reference_is_removed(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    work = tmp_path / "work"
    first = _upload(
        store,
        session_id="capture-session-1",
        intake_id="intake-1",
        idempotency_key="upload-1",
    )
    second = _upload(
        store,
        session_id="capture-session-2",
        intake_id="intake-2",
        idempotency_key="upload-2",
    )
    object_path = next(path for path in (store / "objects").rglob("*") if path.is_file())

    first_tombstone = _apply(store, work, first)
    assert first_tombstone["raw_object_deleted"] is False
    assert first_tombstone["raw_object_preserved_for_shared_active_references"] is True
    assert object_path.is_file()

    second_tombstone = _apply(store, work, second)
    assert second_tombstone["raw_object_deleted"] is True
    assert not object_path.exists()


def test_retention_fails_before_exact_deadline_then_tombstones(tmp_path: Path) -> None:
    store = tmp_path / "store"
    receipt = _upload(
        store,
        session_id="capture-session-1",
        intake_id="intake-1",
        idempotency_key="upload-1",
    )
    received = datetime.fromisoformat(receipt["capture_upload_received_at"])
    with pytest.raises(CaptureLifecycleError, match="capture_retention_not_expired"):
        _apply(
            store,
            tmp_path / "work",
            receipt,
            action="retention_expired",
            now=received + timedelta(days=29),
        )
    tombstone = _apply(
        store,
        tmp_path / "work",
        receipt,
        action="retention_expired",
        now=received + timedelta(days=30),
    )
    assert tombstone["action"] == "retention_expired"
    assert tombstone["retention_deadline"] == (received + timedelta(days=30)).isoformat()


def test_external_provider_obligation_remains_pending_until_receipt_is_recorded(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    work = tmp_path / "work"
    receipt = _upload(
        store,
        session_id="capture-session-1",
        intake_id="intake-1",
        idempotency_key="upload-1",
    )
    execution_path = (
        work
        / "reconstruction_control_plane"
        / "plans"
        / "plan-1"
        / "artifacts"
        / "execution_result.json"
    )
    execution_path.parent.mkdir(parents=True)
    result = {
        "intake_id": receipt["intake_id"],
        "capture_digest": receipt["capture_digest"],
        "provider_identity": "provider-1",
        "reconstruction_result_digest": "sha256:" + "a" * 64,
        "provider_receipt": {"receipt_id": "opaque-provider-receipt"},
        "deletion_evidence": None,
    }
    execution_path.write_text(json.dumps({"results": [result]}), encoding="utf-8")

    tombstone = _apply(store, work, receipt)
    assert tombstone["provider_deletion_status"] == "required"
    obligation = tombstone["provider_deletion_obligations"][0]
    inspection = inspect_capture_lifecycle(
        store_root=store,
        capture_session_id=receipt["capture_session_id"],
        intake_id=receipt["intake_id"],
    )
    assert inspection["provider_deletion_complete"] is False

    evidence = record_provider_deletion_evidence(
        store_root=store,
        capture_session_id=receipt["capture_session_id"],
        intake_id=receipt["intake_id"],
        obligation_digest=obligation["obligation_digest"],
        deletion_receipt_digest="sha256:" + "b" * 64,
        provider_identity="provider-1",
        deleted_at="2026-07-30T12:00:00+00:00",
        verification_method="provider_api_receipt",
        idempotency_key="provider-delete-1",
    )
    assert evidence["proof_boundary"][
        "receipt_metadata_is_independent_provider_verification"
    ] is False
    completed = inspect_capture_lifecycle(
        store_root=store,
        capture_session_id=receipt["capture_session_id"],
        intake_id=receipt["intake_id"],
    )
    assert completed["provider_deletion_complete"] is True


def test_authenticated_service_applies_and_inspects_exact_capture_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    receipt = _upload(
        store,
        session_id="capture-session-1",
        intake_id="intake-1",
        idempotency_key="upload-1",
    )
    manifest = tmp_path / "control" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "work"))
    monkeypatch.setenv(service.CAPTURE_UPLOAD_STORE_ROOT_ENV, str(store))
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "true")
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    client = TestClient(service.create_app())
    headers = {"authorization": "Bearer test-intake-token"}
    route = (
        "/api/live-pipeline/capture-upload-intakes/"
        f"{receipt['capture_session_id']}/{receipt['intake_id']}/lifecycle"
    )

    response = client.post(
        route,
        headers=headers,
        json={
            "schema_version": "capture_lifecycle_submission.v1",
            "capture_digest": receipt["capture_digest"],
            "envelope_digest": receipt["envelope_digest"],
            "action": "consent_revoked",
            "idempotency_key": "service-lifecycle-1",
        },
    )
    assert response.status_code == 200
    assert response.json()["serve_allowed"] is False
    inspection = client.get(route, headers=headers)
    assert inspection.status_code == 200
    assert inspection.json()["state"] == "tombstoned"
