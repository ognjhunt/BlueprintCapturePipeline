from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.capture_intake import validate_capture_intake_envelope
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV
from blueprint_pipeline.local_reconstruction_adapters import (
    LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
    LOCAL_DECODED_OBSERVATION_ADAPTER,
)
from blueprint_pipeline.reconstruction_control_plane import (
    ReconstructionControlPlaneError,
    authorize_reconstruction_plan,
    execute_reconstruction_plan,
    inspect_reconstruction_plan,
    load_reconstruction_compilation_inputs,
    prepare_reconstruction_plan,
)


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _seed_capture_store(root: Path) -> tuple[str, str, str]:
    session_id = "capture-session-1"
    intake_id = "intake-1"
    media = b"retained-video"
    capture_digest = _digest_bytes(media)
    object_relative = f"objects/sha256/{capture_digest[7:9]}/{capture_digest[7:]}"
    object_path = root / object_relative
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(media)
    envelope = validate_capture_intake_envelope(
        {
            "schema_version": "capture_intake_envelope.v1",
            "intake_id": intake_id,
            "idempotency_key": "org-1-intake-1",
            "capture_authority_profile": "monocular_video",
            "source_type": "monocular_video",
            "original_files": [
                {
                    "original_filename": "capture.mp4",
                    "relative_path": "capture.mp4",
                    "sha256": capture_digest,
                    "size_bytes": len(media),
                    "media_type": "video/mp4",
                }
            ],
            "scene_id": "scene-1",
            "customer_id": "customer-1",
            "organization_id": "org-1",
            "capture_device": {"manufacturer": "fixture", "model": "camera"},
            "timing_declaration": {"clock": "media_pts"},
            "coordinate_frame_declaration": {"status": "not_available_from_video"},
            "available_sensor_streams": [
                {
                    "stream_type": "retained_video",
                    "status": "available",
                    "source_relative_path": "capture.mp4",
                }
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
            "upload_validation": {"status": "passed"},
            "malware_content_validation": {"status": "passed"},
        }
    )
    envelope_digest = envelope["envelope_digest"]
    artifact_relative = f"intakes/{intake_id}/{envelope_digest[7:]}"
    artifact_root = root / artifact_relative
    object_manifest = {
        "schema_version": "capture_intake_object_manifest.v1",
        "envelope_digest": envelope_digest,
        "objects": [
            {
                "sha256": capture_digest,
                "size_bytes": len(media),
                "object_path": object_relative,
                "original_filename": "capture.mp4",
                "source_relative_path": "capture.mp4",
            }
        ],
        "raw_inputs_content_addressed": True,
        "raw_inputs_mutated": False,
    }
    object_manifest["manifest_digest"] = canonical_digest(
        object_manifest, digest_field="manifest_digest"
    )
    qa = {
        "schema_version": "capture_qa_report.v1",
        "status": "accepted",
        "state": "capture_accepted",
        "intake_id": intake_id,
        "envelope_digest": envelope_digest,
    }
    qa["qa_report_digest"] = canonical_digest(qa, digest_field="qa_report_digest")
    _write(artifact_root / "capture_intake_envelope.json", envelope)
    _write(artifact_root / "capture_intake_object_manifest.json", object_manifest)
    _write(artifact_root / "capture_qa_report.json", qa)
    receipt_key = hashlib.sha256(f"{session_id}\0{intake_id}".encode()).hexdigest()
    _write(
        root / "transfer_receipts" / f"{receipt_key}.json",
        {
            "schema_version": "capture_upload_receipt.v1",
            "capture_session_id": session_id,
            "intake_id": intake_id,
            "envelope_digest": envelope_digest,
            "capture_digest": capture_digest,
            "admission_status": "accepted",
            "state": "capture_accepted",
            "claim_ceiling": {"metric_geometry": False},
            "artifact_reference": {
                "uri": artifact_relative,
                "envelope_digest": envelope_digest,
            },
            "capture_qa_report": qa,
        },
    )
    return session_id, intake_id, capture_digest


def _stub_media_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.local_reconstruction_adapters.shutil.which",
        lambda name: f"/fake/{name}",
    )

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "-version":
            return subprocess.CompletedProcess(command, 0, f"{Path(command[0]).name} version test\n", "")
        if "-show_frames" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "streams": [
                            {
                                "index": 0,
                                "codec_name": "h264",
                                "width": 64,
                                "height": 48,
                                "avg_frame_rate": "10/1",
                                "time_base": "1/1000",
                            }
                        ],
                        "frames": [
                            {"best_effort_timestamp": "0", "best_effort_timestamp_time": "0.0"},
                            {"best_effort_timestamp": "100", "best_effort_timestamp_time": "0.1"},
                        ],
                    }
                ),
                "",
            )
        Image.new("RGB", (8, 8), color=(40, 20, 10)).save(Path(command[-1]), format="PNG")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "blueprint_pipeline.local_reconstruction_adapters.subprocess.run", fake_run
    )


def test_plan_authorize_execute_and_inspect_local_decoded_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "capture-store"
    state = tmp_path / "state"
    session_id, intake_id, capture_digest = _seed_capture_store(store)
    planned = prepare_reconstruction_plan(
        state_root=state,
        capture_store_root=store,
        capture_session_id=session_id,
        intake_id=intake_id,
        requested_claim_types=["task_discovery", "perception_visibility"],
        idempotency_key="plan-1",
    )

    assert planned["state"] == "authorization_required"
    assert planned["reconstruction_plan"]["source_capture"]["capture_digest"] == capture_digest
    assert planned["authorization_candidates"] == [
        {
            "method_id": "local_decoded_observation_index",
            "method_profile_digest": planned["reconstruction_plan"]["selected_methods"][0][
                "method_profile_digest"
            ],
            "adapter_reference": LOCAL_DECODED_OBSERVATION_ADAPTER,
            "execution_authorized": False,
        }
    ]
    replay = prepare_reconstruction_plan(
        state_root=state,
        capture_store_root=store,
        capture_session_id=session_id,
        intake_id=intake_id,
        requested_claim_types=["perception_visibility", "task_discovery"],
        idempotency_key="plan-1",
    )
    assert replay["plan_id"] == planned["plan_id"]

    authorization = authorize_reconstruction_plan(
        state_root=state,
        plan_id=planned["plan_id"],
        reconstruction_plan_digest=planned["reconstruction_plan"]["reconstruction_plan_digest"],
        authorized_adapter_references=[LOCAL_DECODED_OBSERVATION_ADAPTER],
        actor={"role": "operator", "identity": "operator-1"},
        idempotency_key="authorize-1",
    )
    assert authorization["paid_compute_authorized"] is False

    _stub_media_tools(monkeypatch)
    executed = execute_reconstruction_plan(
        state_root=state,
        capture_store_root=store,
        plan_id=planned["plan_id"],
    )
    assert executed["state"] == "completed"
    assert executed["cost_usd"] == 0.0
    assert executed["results"][0]["outputs"] == ["decoded_observation_frames"]
    assert executed["results"][0]["claim_ceiling"]["metric_geometry"] is False
    assert executed["proof_boundary"]["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    assert execute_reconstruction_plan(
        state_root=state,
        capture_store_root=store,
        plan_id=planned["plan_id"],
    )["already_exists"] is True

    inspection = inspect_reconstruction_plan(state_root=state, plan_id=planned["plan_id"])
    assert inspection["state"] == "completed"
    assert inspection["source_binding"]["capture_digest"] == capture_digest
    compilation_inputs = load_reconstruction_compilation_inputs(
        state_root=state,
        capture_store_root=store,
        plan_id=planned["plan_id"],
        execution_result_digest=executed["execution_result_digest"],
    )
    assert compilation_inputs["reconstruction_plan"] == planned["reconstruction_plan"]
    assert compilation_inputs["reconstruction_results"] == executed["results"]


def test_fails_closed_for_unplanned_adapter_and_mutated_capture(
    tmp_path: Path,
) -> None:
    store = tmp_path / "capture-store"
    state = tmp_path / "state"
    session_id, intake_id, _ = _seed_capture_store(store)
    planned = prepare_reconstruction_plan(
        state_root=state,
        capture_store_root=store,
        capture_session_id=session_id,
        intake_id=intake_id,
        requested_claim_types=["task_discovery"],
        idempotency_key="plan-1",
    )
    with pytest.raises(ReconstructionControlPlaneError, match="authorization_adapter_not_planned"):
        authorize_reconstruction_plan(
            state_root=state,
            plan_id=planned["plan_id"],
            reconstruction_plan_digest=planned["reconstruction_plan"]["reconstruction_plan_digest"],
            authorized_adapter_references=[LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER],
            actor={"role": "operator"},
            idempotency_key="authorize-1",
        )

    object_path = next((store / "objects").rglob("*"))
    while object_path.is_dir():
        object_path = next(object_path.rglob("*"))
    object_path.write_bytes(b"mutated")
    with pytest.raises(
        ReconstructionControlPlaneError, match="capture_object_digest_or_size_mismatch"
    ):
        prepare_reconstruction_plan(
            state_root=tmp_path / "other-state",
            capture_store_root=store,
            capture_session_id=session_id,
            intake_id=intake_id,
            requested_claim_types=["task_discovery"],
            idempotency_key="plan-2",
        )


def test_authenticated_service_plan_authorize_execute_and_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "capture-store"
    session_id, intake_id, _ = _seed_capture_store(store)
    manifest = tmp_path / "control" / "manifest.json"
    _write(manifest, {})
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "work"))
    monkeypatch.setenv(service.CAPTURE_UPLOAD_STORE_ROOT_ENV, str(store))
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "true")
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    client = TestClient(service.create_app())
    headers = {"authorization": "Bearer test-intake-token"}

    plan_response = client.post(
        "/api/live-pipeline/reconstructions/plan",
        headers=headers,
        json={
            "schema_version": "reconstruction_plan_submission.v1",
            "capture_session_id": session_id,
            "intake_id": intake_id,
            "requested_claim_types": ["task_discovery"],
            "idempotency_key": "service-plan-1",
        },
    )
    assert plan_response.status_code == 200
    planned = plan_response.json()
    plan_id = planned["plan_id"]

    authorization_response = client.post(
        f"/api/live-pipeline/reconstructions/{plan_id}/authorize",
        headers=headers,
        json={
            "schema_version": "reconstruction_authorization_submission.v1",
            "reconstruction_plan_digest": planned["reconstruction_plan"][
                "reconstruction_plan_digest"
            ],
            "authorized_adapter_references": [LOCAL_DECODED_OBSERVATION_ADAPTER],
            "actor": {"role": "operator", "identity": "operator-1"},
            "idempotency_key": "service-authorize-1",
        },
    )
    assert authorization_response.status_code == 200

    _stub_media_tools(monkeypatch)
    execute_response = client.post(
        f"/api/live-pipeline/reconstructions/{plan_id}/execute", headers=headers
    )
    assert execute_response.status_code == 200
    assert execute_response.json()["state"] == "completed"
    inspection_response = client.get(
        f"/api/live-pipeline/reconstructions/{plan_id}", headers=headers
    )
    assert inspection_response.status_code == 200
    assert inspection_response.json()["state"] == "completed"
