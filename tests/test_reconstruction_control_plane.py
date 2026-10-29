from __future__ import annotations

import hashlib
import io
import json
import subprocess
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import jsonschema
from fastapi.testclient import TestClient
from PIL import Image

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.capture_intake import validate_capture_intake_envelope
from blueprint_pipeline.capture_upload_intake import process_capture_upload_submission
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV
from blueprint_pipeline.local_reconstruction_adapters import (
    LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
    LOCAL_DECODED_OBSERVATION_ADAPTER,
    LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
)
from blueprint_pipeline.reconstruction_control_plane import (
    ReconstructionControlPlaneError,
    authorize_reconstruction_plan,
    execute_reconstruction_plan,
    inspect_reconstruction_plan,
    load_reconstruction_compilation_inputs,
    prepare_reconstruction_plan,
    prepare_owned_capture_atlas_plan,
)
from blueprint_pipeline.task_evaluation_supervisor import (
    validate_reconstruction_readiness_pointer,
)


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _seed_capture_store(root: Path, *, session_id: str = "capture-session-1",
                        intake_id: str = "intake-1") -> tuple[str, str, str]:
    media = b"retained-video"
    capture_digest = _digest_bytes(media)
    object_relative = f"objects/sha256/{capture_digest[7:9]}/{capture_digest[7:]}"
    object_path = root / object_relative
    object_path.parent.mkdir(parents=True, exist_ok=True)
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


def _seed_external_reconstruction_store(root: Path) -> tuple[str, str, str]:
    session_id = "capture-session-external-1"
    intake_id = "intake-external-1"
    asset = (
        b"ply\n"
        b"format ascii 1.0\n"
        b"element vertex 2\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"end_header\n"
        b"0 0 0 255 0 0\n1 1 1 0 255 0\n"
    )
    submission = {
        "schema_version": "capture_upload_transfer_submission.v1",
        "capture_session_id": session_id,
        "customer_id": "customer-1",
        "organization_id": "org-1",
        "request": {
            "schema_version": "capture_upload_session_request.v1",
            "intake_id": intake_id,
            "idempotency_key": "org-1-intake-external-1",
            "capture_authority_profile": "precomputed_external_reconstruction",
            "source_type": "precomputed_external_reconstruction",
            "scene_id": "scene-1",
            "original_file": {
                "original_filename": "polycam_pointcloud.ply",
                "size_bytes": len(asset),
                "media_type": "application/octet-stream",
            },
            "capture_device": {"manufacturer": "fixture", "model": "provider"},
            "timing_declaration": {"status": "not_included_in_import"},
            "coordinate_frame_declaration": {"status": "provider_declared_unverified"},
            "available_sensor_streams": [
                {"stream_type": "external_reconstruction", "status": "available"}
            ],
            "source_capture_binding": {
                "source_capture_digest": "sha256:" + "b" * 64,
                "provider_identity": "fixture-provider",
            },
            "governance": {
                "rights": "accepted",
                "consent": "accepted",
                "privacy": "restricted_local_only",
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
            "permitted_evidence_uses": ["appearance_review"],
        },
        "transfer": {
            "provider": "backblaze",
            "url": "https://download.example.test/private/polycam_pointcloud.ply",
            "authorization": "short-lived-test-grant",
            "expires_at_iso": (
                datetime.now(timezone.utc) + timedelta(minutes=10)
            ).isoformat(),
        },
    }
    receipt = process_capture_upload_submission(
        submission,
        store_root=root,
        allowed_hosts=["download.example.test"],
        transfer_opener=lambda **_kwargs: closing(io.BytesIO(asset)),
        malware_scanner=lambda _path: {
            "status": "passed",
            "scanner": "fixture-clam",
        },
    )
    return session_id, intake_id, str(receipt["capture_digest"])


def _stub_media_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.local_reconstruction_adapters.shutil.which",
        lambda name: f"/fake/{name}",
    )

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "-version":
            return subprocess.CompletedProcess(
                command, 0, f"{Path(command[0]).name} version test\n", ""
            )
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

    monkeypatch.setattr("blueprint_pipeline.local_reconstruction_adapters.subprocess.run", fake_run)


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
    assert (
        executed["proof_boundary"]["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    )
    assert (
        execute_reconstruction_plan(
            state_root=state,
            capture_store_root=store,
            plan_id=planned["plan_id"],
        )["already_exists"]
        is True
    )

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


def test_external_reconstruction_import_executes_as_partial_appearance_only(
    tmp_path: Path,
) -> None:
    store = tmp_path / "capture-store"
    state = tmp_path / "state"
    session_id, intake_id, asset_digest = _seed_external_reconstruction_store(store)
    planned = prepare_reconstruction_plan(
        state_root=state,
        capture_store_root=store,
        capture_session_id=session_id,
        intake_id=intake_id,
        requested_claim_types=["appearance_review"],
        idempotency_key="plan-external-1",
    )

    assert planned["state"] == "authorization_required"
    assert planned["authorization_candidates"] == [
        {
            "method_id": "local_external_reconstruction_import",
            "method_profile_digest": planned["reconstruction_plan"]["selected_methods"][0][
                "method_profile_digest"
            ],
            "adapter_reference": LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
            "execution_authorized": False,
        }
    ]
    assert (
        planned["reconstruction_plan"]["missing_representations"][0]["representation"]
        == "decoded_observation_frames"
    )
    authorization = authorize_reconstruction_plan(
        state_root=state,
        plan_id=planned["plan_id"],
        reconstruction_plan_digest=planned["reconstruction_plan"]["reconstruction_plan_digest"],
        authorized_adapter_references=[LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER],
        actor={"role": "operator", "identity": "operator-1"},
        idempotency_key="authorize-external-1",
    )
    assert authorization["paid_compute_authorized"] is False

    executed = execute_reconstruction_plan(
        state_root=state,
        capture_store_root=store,
        plan_id=planned["plan_id"],
    )
    assert executed["state"] == "partial"
    assert executed["missing_representations"] == [
        "decoded_observation_frames",
        "qualified_appearance_render",
    ]
    assert executed["results"][0]["capture_digest"] == asset_digest
    assert executed["results"][0]["outputs"] == ["appearance_layer"]
    assert executed["results"][0]["claim_ceiling"]["metric_geometry"] is False
    assert executed["results"][0]["claim_ceiling"]["collision_geometry"] is False
    assert executed["results"][0]["claim_ceiling"]["physical_task_success"] is False
    assert (
        executed["proof_boundary"]["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    )
    compilation_inputs = load_reconstruction_compilation_inputs(
        state_root=state,
        capture_store_root=store,
        plan_id=planned["plan_id"],
        execution_result_digest=executed["execution_result_digest"],
    )
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
    planned_readiness = planned["task_evaluation_supervisor_readiness"]
    assert planned_readiness["already_exists"] is False
    planned_snapshot = json.loads(
        Path(planned_readiness["snapshot_path"]).read_text(encoding="utf-8")
    )
    planned_stages = {row["stage_id"]: row for row in planned_snapshot["stages"]}
    assert (
        planned_stages["compile_frozen_frame_dataset"]["readiness_status"]
        == "awaiting_control_plane_authority"
    )

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
    authorized_readiness = authorization_response.json()[
        "task_evaluation_supervisor_readiness"
    ]
    assert authorized_readiness["previous_readiness_digest"] == planned_readiness[
        "readiness_digest"
    ]
    authorized_snapshot = json.loads(
        Path(authorized_readiness["snapshot_path"]).read_text(encoding="utf-8")
    )
    authorized_stages = {row["stage_id"]: row for row in authorized_snapshot["stages"]}
    assert (
        authorized_stages["compile_frozen_frame_dataset"]["readiness_status"]
        == "ready_for_bounded_tool_call"
    )

    _stub_media_tools(monkeypatch)
    execute_response = client.post(
        f"/api/live-pipeline/reconstructions/{plan_id}/execute", headers=headers
    )
    assert execute_response.status_code == 200, execute_response.text
    executed = execute_response.json()
    assert executed["state"] == "completed"
    executed_readiness = executed["task_evaluation_supervisor_readiness"]
    assert executed_readiness["previous_readiness_digest"] == authorized_readiness[
        "readiness_digest"
    ]
    executed_snapshot = json.loads(
        Path(executed_readiness["snapshot_path"]).read_text(encoding="utf-8")
    )
    executed_stages = {row["stage_id"]: row for row in executed_snapshot["stages"]}
    assert (
        executed_stages["compile_frozen_frame_dataset"]["readiness_status"]
        == "recorded_support_completed"
    )
    latest_pointer_path = Path(executed_readiness["latest_pointer_path"])
    latest_pointer = json.loads(latest_pointer_path.read_text(encoding="utf-8"))
    jsonschema.validate(
        latest_pointer,
        json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "docs/schemas/task_evaluation_reconstruction_readiness_pointer.v1.schema.json"
            ).read_text(encoding="utf-8")
        ),
    )
    assert latest_pointer["readiness_digest"] == executed_readiness["readiness_digest"]
    tampered_pointer = dict(latest_pointer)
    tampered_pointer["previous_readiness_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="reconstruction_readiness_pointer_invalid"):
        validate_reconstruction_readiness_pointer(tampered_pointer)
    history = sorted(
        latest_pointer_path.parent.glob(
            "reconstruction_execution_readiness_history/*.json"
        )
    )
    assert len(history) == 3

    replay_response = client.post(
        f"/api/live-pipeline/reconstructions/{plan_id}/execute", headers=headers
    )
    assert replay_response.status_code == 200
    replayed = replay_response.json()
    assert replayed["already_exists"] is True
    assert replayed["task_evaluation_supervisor_readiness"]["already_exists"] is True
    assert (
        replayed["task_evaluation_supervisor_readiness"]["readiness_digest"]
        == executed_readiness["readiness_digest"]
    )
    assert len(history) == len(
        list(
            latest_pointer_path.parent.glob(
                "reconstruction_execution_readiness_history/*.json"
            )
        )
    )
    inspection_response = client.get(
        f"/api/live-pipeline/reconstructions/{plan_id}", headers=headers
    )
    assert inspection_response.status_code == 200
    assert inspection_response.json()["state"] == "completed"


def _seed_owned_atlas_source(root: Path, *, session_id: str = "capture-session-1",
                             intake_id: str = "intake-1") -> tuple[str, str, str]:
    session, intake, digest = _seed_capture_store(root, session_id=session_id, intake_id=intake_id)
    receipt_path = root / 'transfer_receipts' / (hashlib.sha256(f'{session}\0{intake}'.encode()).hexdigest() + '.json')
    receipt = json.loads(receipt_path.read_text())
    previous = root / receipt['artifact_reference']['uri']
    envelope = json.loads((previous / 'capture_intake_envelope.json').read_text())
    envelope.pop('envelope_digest')
    envelope['known_task_specification'] = {
        'description': 'Navigate to the end of the aisle, avoid both shelves, and stop before the end bookshelf.',
        'task_family': 'customer_navigation_and_stopping',
        'success_parameters_status': 'proposed_unconfirmed',
        'operations': ['navigate', 'avoid_side_shelves', 'stop_before_end_shelving'],
    }
    envelope['governance']['consent'] = 'not_required'
    envelope = validate_capture_intake_envelope(envelope)
    artifact = root / 'intakes' / intake / envelope['envelope_digest'][7:]
    qa = json.loads((previous / 'capture_qa_report.json').read_text())
    qa['envelope_digest'] = envelope['envelope_digest']
    qa['qa_report_digest'] = canonical_digest(qa, digest_field='qa_report_digest')
    manifest = json.loads((previous / 'capture_intake_object_manifest.json').read_text())
    manifest['envelope_digest'] = envelope['envelope_digest']
    manifest['manifest_digest'] = canonical_digest(manifest, digest_field='manifest_digest')
    _write(artifact / 'capture_intake_envelope.json', envelope)
    _write(artifact / 'capture_qa_report.json', qa)
    _write(artifact / 'capture_intake_object_manifest.json', manifest)
    receipt.update(envelope_digest=envelope['envelope_digest'], capture_qa_report=qa,
                   artifact_reference={'uri': str(artifact.relative_to(root)),
                                       'envelope_digest': envelope['envelope_digest']})
    _write(receipt_path, receipt)
    return session, intake, digest


def test_owned_atlas_plan_retains_real_task_source_and_reuses_identical_reconstruction(tmp_path):
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, digest = _seed_owned_atlas_source(store)
    options = dict(state_root=state, capture_store_root=store, capture_session_id=session, intake_id=intake,
                   requested_claim_types=['navigation_clearance', 'task_discovery'])
    plan = prepare_owned_capture_atlas_plan(**options, idempotency_key='owned-plan-1')
    assert plan['source_video_digest'] == digest
    assert plan['plan_digest'] == cross_runtime_canonical_digest(plan, digest_field='plan_digest')
    context = plan['source_context']
    assert context['request_id'] == plan['plan_id']
    assert context['capture_id'] == session
    assert context['task_specification']['operations'] == ['navigate', 'avoid_side_shelves', 'stop_before_end_shelving']
    assert context['task_confirmation_status'] == 'proposed_unconfirmed'
    assert context['governance']['consent'] == 'not_required'
    assert context['governance']['provider_constraints']['external_processing_allowed'] is False
    assert plan['proof_boundary']['provider_disclosure_authorized'] is False
    assert plan['proof_boundary']['paid_compute_authorized'] is False
    assert plan['state'] == 'authorization_required'
    artifact = state / 'plans' / plan['plan_id'] / 'artifacts' / 'owned_atlas_preparation_plan.json'
    before = artifact.stat().st_mtime_ns
    assert prepare_owned_capture_atlas_plan(**options, idempotency_key='owned-plan-1') == plan
    assert prepare_owned_capture_atlas_plan(**options, idempotency_key='another-request-key') == plan
    assert artifact.stat().st_mtime_ns == before
    assert len(list((state / 'plans').iterdir())) == 1


def test_owned_atlas_request_key_cannot_change_claims_or_silently_replace_task(tmp_path):
    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, _ = _seed_owned_atlas_source(store)
    options = dict(state_root=state, capture_store_root=store, capture_session_id=session, intake_id=intake,
                   idempotency_key='owned-plan-1')
    first = prepare_owned_capture_atlas_plan(**options, requested_claim_types=['task_discovery'])
    with pytest.raises(ReconstructionControlPlaneError, match='immutable_artifact_conflict') as failure:
        prepare_owned_capture_atlas_plan(**options, requested_claim_types=['collision_contact'])
    assert failure.value.status_code == 409
    assert len(list((state / 'plans').iterdir())) == 1
    assert json.loads((state / 'plans' / first['plan_id'] / 'artifacts' / 'owned_atlas_preparation_plan.json').read_text()) == first


@pytest.mark.parametrize('change', ['bytes', 'revocation'])
def test_owned_atlas_plan_reopens_source_even_on_an_idempotent_retry(tmp_path, change):
    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, digest = _seed_owned_atlas_source(store)
    options = dict(state_root=state, capture_store_root=store, capture_session_id=session, intake_id=intake,
                   idempotency_key='owned-plan-1', requested_claim_types=['task_discovery'])
    prepare_owned_capture_atlas_plan(**options)
    if change == 'bytes':
        (store / 'objects' / 'sha256' / digest[7:9] / digest[7:]).write_bytes(b'changed source')
        expected = 'capture_object_digest_or_size_mismatch'
    else:
        marker = hashlib.sha256(f'{session}\0{intake}'.encode()).hexdigest() + '.json'
        _write(store / 'lifecycle_tombstones' / marker, {'revoked': True})
        expected = 'capture_revoked_or_expired'
    with pytest.raises(ReconstructionControlPlaneError, match=expected):
        prepare_owned_capture_atlas_plan(**options)


def test_owned_provider_plan_never_satisfies_local_v1_execution_authorization(tmp_path):
    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, _ = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
        idempotency_key='owned-plan-1')
    with pytest.raises(ReconstructionControlPlaneError, match='reconstruction_plan_not_found'):
        authorize_reconstruction_plan(state_root=state, plan_id=plan['plan_id'],
            reconstruction_plan_digest=plan['plan_digest'],
            authorized_adapter_references=[LOCAL_DECODED_OBSERVATION_ADAPTER], actor={'identity': 'owner'},
            idempotency_key='owned-authorization-1')


def test_owned_provider_plan_reports_missing_complete_task_as_input_gap(tmp_path):
    store = tmp_path / 'store'
    session, intake, _ = _seed_capture_store(store)
    with pytest.raises(ReconstructionControlPlaneError, match='owned_capture_atlas_task_description_required'):
        prepare_owned_capture_atlas_plan(state_root=tmp_path / 'state', capture_store_root=store,
            capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
            idempotency_key='owned-plan-1')


@pytest.mark.parametrize('change', ['bytes', 'revocation'])
def test_owned_atlas_transport_reopens_source_for_each_disclosure_but_can_reconcile_and_cleanup(tmp_path, change):
    from blueprint_pipeline.atlas_preparation_context import preparation_source_checkpoint
    from blueprint_pipeline.owned_capture_atlas_runtime import bind_owned_atlas_source
    from blueprint_pipeline.website_atlas_pose import AtlasClient

    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, digest = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
        idempotency_key='owned-plan-1')
    sent = []

    def transport(method, url, headers, data):
        sent.append((method, url))
        return 200, b'{}'

    with bind_owned_atlas_source(state_root=state, capture_store_root=store, plan_id=plan['plan_id']) as context:
        client = AtlasClient(base_url='https://api.example.test/api/v2', api_key='hermetic-test',
                             transport=transport, source_checkpoint=preparation_source_checkpoint(context))
        client.call('POST', '/assets', {'asset': {'displayName': 'source-view'}})
        assert len(sent) == 1
        if change == 'bytes':
            (store / 'objects' / 'sha256' / digest[7:9] / digest[7:]).write_bytes(b'changed source')
            expected = 'capture_object_digest_or_size_mismatch'
        else:
            marker = hashlib.sha256(f'{session}\0{intake}'.encode()).hexdigest() + '.json'
            _write(store / 'lifecycle_tombstones' / marker, {'revoked': True})
            expected = 'capture_revoked_or_expired'
        with pytest.raises(ReconstructionControlPlaneError, match=expected):
            client.call('POST', '/tasks:images2PosedRGBD', {'frames': []})
        with pytest.raises(ReconstructionControlPlaneError, match=expected):
            client.send_bytes('https://uploads.example.test/frame', method='PUT', headers={}, data=b'frame')
        assert len(sent) == 1
        # No new source disclosure: finish accounting/output retention and teardown.
        client.call('GET', '/operations/existing-operation')
        client.call('POST', '/assets/existing-output:createReadUrl')
        client.call('DELETE', '/assets/existing-source')
        assert len(sent) == 4
    # A resumed worker can still reconcile an already-bought operation.
    with bind_owned_atlas_source(state_root=state, capture_store_root=store, plan_id=plan['plan_id']) as context:
        client = AtlasClient(base_url='https://api.example.test/api/v2', api_key='hermetic-test',
                             transport=transport, source_checkpoint=preparation_source_checkpoint(context))
        client.call('GET', '/operations/existing-operation')
        with pytest.raises(ReconstructionControlPlaneError, match=expected):
            client.call('POST', '/tasks:atlasGenerate', {})


def test_owned_source_context_never_supplies_its_own_runtime_fence(tmp_path):
    from blueprint_pipeline.atlas_preparation_context import preparation_source_checkpoint
    from blueprint_pipeline.owned_capture_atlas_runtime import bind_owned_atlas_source

    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, _ = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
        idempotency_key='owned-plan-1')
    with pytest.raises(ValueError, match='owned_capture_atlas_source_fence_required'):
        preparation_source_checkpoint(plan['source_context'])
    with bind_owned_atlas_source(state_root=state, capture_store_root=store, plan_id=plan['plan_id']) as context:
        # The explicit callback carries the source fence across SDK worker threads.
        from concurrent.futures import ThreadPoolExecutor
        checkpoint = preparation_source_checkpoint(context)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(checkpoint, 'test-threaded-disclosure').result()
    with pytest.raises(ValueError, match='owned_capture_atlas_source_fence_required'):
        preparation_source_checkpoint(plan['source_context'])
    assert preparation_source_checkpoint({'schema_version': 'website_task_context.v1'}) is None


def test_owned_runtime_refuses_a_modified_retained_plan(tmp_path):
    from blueprint_pipeline.owned_capture_atlas_runtime import bind_owned_atlas_source

    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, _ = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
        idempotency_key='owned-plan-1')
    plan['source_video_digest'] = 'sha256:' + 'f' * 64
    _write(state / 'plans' / plan['plan_id'] / 'artifacts' / 'owned_atlas_preparation_plan.json', plan)
    with pytest.raises(ValueError, match='owned_capture_atlas_plan_binding_invalid'):
        with bind_owned_atlas_source(state_root=state, capture_store_root=store, plan_id=plan['plan_id']):
            pytest.fail('modified plan admitted')


def test_mutating_yielded_context_cannot_retarget_a_captured_source_fence(tmp_path):
    from copy import deepcopy
    from blueprint_pipeline.atlas_preparation_context import preparation_source_checkpoint
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    from blueprint_pipeline.owned_capture_atlas_runtime import bind_owned_atlas_source
    from blueprint_pipeline.website_atlas_pose import AtlasClient

    store, state = tmp_path / 'store', tmp_path / 'state'
    first = _seed_owned_atlas_source(store)
    other = _seed_owned_atlas_source(store, session_id='capture-session-2', intake_id='intake-2')
    plans = [prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
             capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
             idempotency_key='owned-plan-1') for session, intake, _ in [first, other]]
    sent = []

    def transport(*args):
        sent.append(args)
        return 200, b'{}'

    with bind_owned_atlas_source(state_root=state, capture_store_root=store, plan_id=plans[0]['plan_id']) as context:
        client = AtlasClient(base_url='https://api.example.test/api/v2', api_key='hermetic-test',
                             transport=transport, source_checkpoint=preparation_source_checkpoint(context))
        replacement = deepcopy(plans[1]['source_context'])
        replacement['request_id'] = plans[0]['plan_id']
        replacement['context_digest'] = cross_runtime_canonical_digest(replacement, digest_field='context_digest')
        context.clear()
        context.update(replacement)
        marker = hashlib.sha256(f'{first[0]}\0{first[1]}'.encode()).hexdigest() + '.json'
        _write(store / 'lifecycle_tombstones' / marker, {'revoked': True})
        with pytest.raises(ReconstructionControlPlaneError, match='capture_revoked_or_expired'):
            client.send_bytes('https://uploads.example.test/frame', method='PUT', headers={}, data=b'original-frame')
        assert sent == []


def test_owned_atlas_configuration_uses_existing_signed_transport_without_website_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline import website_task_context as transport
    from blueprint_pipeline.website_atlas_pipeline import load_configuration

    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, _ = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
        idempotency_key='owned-plan-1')
    context = plan['source_context']
    monkeypatch.setenv('PIPELINE_SYNC_WEBAPP_URL', 'https://tryblueprint.io/api/internal/pipeline/sync')
    monkeypatch.setattr(transport, 'load_pipeline_sync_token', lambda: 'hermetic-sync-secret')
    requests = []

    def request(url, **options):
        requests.append((url, options))
        return SimpleNamespace(body=json.dumps({'schema_version': 'owned_capture_atlas_configuration.v1',
            'source_video_digest': plan['source_video_digest'], 'task_context_digest': context['context_digest'],
            'dispatch_authorized': False, 'grant_creation_allowed': False}).encode())

    monkeypatch.setattr(transport, 'safe_request', request)
    result = load_configuration(context, plan['source_video_digest'])
    assert result['dispatch_authorized'] is False
    url, options = requests[0]
    assert url == f'https://tryblueprint.io/api/internal/pipeline/owned-captures/{session}/atlas-configuration'
    assert options['method'] == 'POST' and options['headers']
    assert json.loads(options['data']) == {'plan_id': plan['plan_id'], 'task_context_digest': context['context_digest'],
        'source_video_digest': plan['source_video_digest'], 'create': False}
    assert 'site-' not in url and 'walkthrough-' not in url
    with pytest.raises(ValueError, match='source_video_binding_mismatch'):
        load_configuration(context, 'sha256:' + 'f' * 64)
    assert len(requests) == 1


def test_owned_settlement_routes_without_reopening_revoked_processing_authority(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context as transport
    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, _ = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
        idempotency_key='owned-plan-1')
    marker = hashlib.sha256(f'{session}\0{intake}'.encode()).hexdigest() + '.json'
    _write(store / 'lifecycle_tombstones' / marker, {'revoked': True})
    calls = []
    monkeypatch.setattr(transport, 'website_webapp_request', lambda **kw: calls.append(kw) or {'status': 'settled'})
    result = transport.preparation_webapp_request(task_context=plan['source_context'], operation='preparation-settlement',
        payload={'request_id': plan['plan_id'], 'scene_id': plan['source_context']['scene_id'],
                 'settlement': {'allocation_binding_digest': 'sha256:' + 'd' * 64}})
    assert result == {'status': 'settled'}
    assert calls == [{'capture_id': session, 'operation': 'preparation-settlement',
        'payload': {'settlement': {'allocation_binding_digest': 'sha256:' + 'd' * 64}}, 'source_kind': 'owned_capture'}]


def test_owned_paid_reservation_requires_current_runtime_source_before_budget_or_rpc(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context as transport
    from blueprint_pipeline import paid_resource_admission as admission
    from blueprint_pipeline.owned_capture_atlas_runtime import bind_owned_atlas_source
    store, state = tmp_path / 'store', tmp_path / 'state'
    session, intake, _ = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=['task_discovery'],
        idempotency_key='owned-plan-1')
    monkeypatch.setattr(admission, 'require_paid_resource_budget', lambda **kw: pytest.fail('reached budget before source'))
    monkeypatch.setattr(transport, 'website_webapp_request', lambda **kw: pytest.fail('reached RPC before source'))
    options = dict(task_context=plan['source_context'], binding_digest='sha256:' + 'b' * 64,
                   maximum_cost_usd=1, request_count=1, resource_class='provider_reconstruction_api',
                   provider='world_labs_atlas', source_video_digest=plan['source_video_digest'])
    with pytest.raises(ValueError, match='source_fence_required'):
        transport.reserve_website_preparation_spend(**options)
    with bind_owned_atlas_source(state_root=state, capture_store_root=store, plan_id=plan['plan_id']):
        marker = hashlib.sha256(f'{session}\0{intake}'.encode()).hexdigest() + '.json'
        _write(store / 'lifecycle_tombstones' / marker, {'revoked': True})
        with pytest.raises(ReconstructionControlPlaneError, match='capture_revoked_or_expired'):
            transport.reserve_website_preparation_spend(**options)


def test_owned_atlas_http_plan_requires_authentication_and_never_accepts_authority_fields(tmp_path, monkeypatch):
    store = tmp_path / 'store'
    session, intake, _ = _seed_owned_atlas_source(store)
    manifest = tmp_path / 'control' / 'manifest.json'
    _write(manifest, {})
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / 'work'))
    monkeypatch.setenv(service.CAPTURE_UPLOAD_STORE_ROOT_ENV, str(store))
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, 'test-intake-token')
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, 'true')
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / 'nonces'))
    client = TestClient(service.create_app())
    command = {'schema_version': 'owned_capture_atlas_plan_submission.v1', 'capture_session_id': session,
               'intake_id': intake, 'requested_claim_types': ['navigation_clearance'], 'idempotency_key': 'owned-plan-1'}
    assert client.post('/api/live-pipeline/reconstructions/atlas-plan', json=command).status_code in {401, 403}
    headers = {'authorization': 'Bearer test-intake-token'}
    refused = client.post('/api/live-pipeline/reconstructions/atlas-plan', headers=headers,
                          json={**command, 'paid_compute_authorized': True})
    assert refused.status_code == 422
    response = client.post('/api/live-pipeline/reconstructions/atlas-plan', headers=headers, json=command)
    assert response.status_code == 200
    assert response.json()['source_context']['description'].startswith('Navigate')
    assert response.json()['proof_boundary']['provider_disclosure_authorized'] is False
