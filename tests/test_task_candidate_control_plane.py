from __future__ import annotations

import copy
import hmac
import json
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV
from blueprint_pipeline.task_candidate_control_plane import (
    TaskCandidateControlPlaneError,
    load_task_candidate_control_plane_state,
    process_task_candidate_decision_submission,
    publish_and_sync_task_candidate_discovery,
    publish_task_candidate_discovery,
    sync_task_candidate_discovery_to_webapp,
)
from blueprint_pipeline.task_candidate_control_plane_cli import main as control_plane_main
from blueprint_pipeline.task_candidate_discovery import build_task_candidate_discovery


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _discovery(discovery_id: str = "discovery-1") -> dict:
    return build_task_candidate_discovery(
        discovery_id=discovery_id,
        source_capture={
            "intake_id": "intake-1",
            "capture_digest": SHA_A,
            "capture_authority_profile": "camera_360_equirectangular",
        },
        capture_qa_report_digest=SHA_B,
        scene_analysis={
            "observed_site_facts": [
                {
                    "fact_id": "fact-tote",
                    "description": "A blue tote is directly visible on the table.",
                    "confidence": 0.98,
                    "supporting_frames": ["frame-10"],
                    "supporting_3d_regions": ["region-table"],
                }
            ],
            "inferred_objects_and_affordances": [],
            "unsupported_or_occluded_regions": [],
            "hazards": [],
            "privacy_sensitive_areas": [],
        },
        candidate_proposals=[
            {
                "description": "Move the blue tote into the marked box.",
                "observed_objects": [
                    {
                        "object_id": "tote-1",
                        "label": "blue tote",
                        "observation_fact_ids": ["fact-tote"],
                    }
                ],
                "target_regions": [{"region_id": "box-1", "label": "marked box"}],
                "required_robot_capabilities": ["rigid-object grasp"],
                "likely_task_family": "rigid_object_pick_place",
                "proposed_measurable_success_condition": {
                    "metric": "object_center_distance",
                    "operator": "<=",
                    "threshold": 0.05,
                    "units": "m",
                },
                "required_site_reset": "Return the tote to the table marker.",
                "supporting_frames": ["frame-10"],
                "supporting_3d_regions": ["region-table", "box-1"],
                "confidence": 0.94,
                "coverage": {"task_object": 0.8},
                "assumptions": ["The tote is movable."],
                "missing_evidence": ["Rear grasp surface is occluded."],
                "prohibited_claims": ["physical_task_success"],
                "estimated_evaluation_cost_usd": 2.5,
                "expected_customer_value": None,
            }
        ],
        proposal_method={
            "method_id": "local-task-proposer",
            "version": "1",
            "implementation_digest": SHA_C,
            "proposer_identity": "provider:model-a",
            "origin": "model_provider",
        },
    )


def _submission(discovery: dict, *, command_id: str = "task-command-1") -> dict:
    candidate = discovery["task_candidates"][0]
    return {
        "schema_version": "task_candidate_decision_submission.v1",
        "capture_session_id": "capture-review-1",
        "intake_id": "intake-1",
        "command": {
            "schema_version": "task_candidate_decision_command_record.v1",
            "command_request_id": command_id,
            "requester_user_id": "buyer-123",
            "actor": {"role": "customer", "identity": "firebase:buyer-123"},
            "capture_session_id": "capture-review-1",
            "intake_id": "intake-1",
            "discovery_digest": discovery["discovery_digest"],
            "task_candidate_id": candidate["task_candidate_id"],
            "candidate_digest": candidate["candidate_digest"],
            "action": "approve",
            "rationale": "This is the exact task we want evaluated.",
            "edited_task": None,
            "request_fingerprint_sha256": SHA_A,
            "idempotency_key": "approve-task-candidate-1",
            "pipeline_approval_status": "pending_pipeline_validation",
            "created_at_iso": "2026-07-29T21:00:00Z",
        },
    }


def test_discovery_publication_is_immutable_idempotent_and_keeps_successors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-control"
    discovery = _discovery()
    first = publish_task_candidate_discovery(
        state_root=root,
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=discovery,
    )
    replay = publish_task_candidate_discovery(
        state_root=root,
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=discovery,
    )
    successor = _discovery("discovery-2")
    publish_task_candidate_discovery(
        state_root=root,
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=successor,
    )

    assert first["already_exists"] is False
    assert replay["already_exists"] is True
    state = load_task_candidate_control_plane_state(
        state_root=root, capture_session_id="capture-review-1"
    )
    assert state["current_discovery_digest"] == successor["discovery_digest"]
    assert state["discovery_history"] == [
        discovery["discovery_digest"],
        successor["discovery_digest"],
    ]
    assert len(list((root / "sessions" / "capture-review-1" / "discoveries").glob("*.json"))) == 2


def test_control_plane_cli_publishes_and_inspects_without_a_sync_secret(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "task-control"
    discovery_path = tmp_path / "discovery.json"
    discovery_path.write_text(json.dumps(_discovery()), encoding="utf-8")

    assert control_plane_main(
        [
            "publish-discovery",
            "--state-root",
            str(root),
            "--capture-session-id",
            "capture-review-1",
            "--intake-id",
            "intake-1",
            "--discovery-json",
            str(discovery_path),
            "--sync-mode",
            "off",
        ]
    ) == 0
    published = json.loads(capsys.readouterr().out)
    assert published["status"] == "published"
    assert published["proof_boundary"]["candidate_is_customer_intent"] is False

    assert control_plane_main(
        [
            "inspect-state",
            "--state-root",
            str(root),
            "--capture-session-id",
            "capture-review-1",
        ]
    ) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["current_discovery_digest"] == _discovery()["discovery_digest"]


def test_pipeline_records_approval_once_without_compiling_a_request(tmp_path: Path) -> None:
    root = tmp_path / "task-control"
    discovery = _discovery()
    publish_task_candidate_discovery(
        state_root=root,
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=discovery,
    )
    submission = _submission(discovery)
    result = process_task_candidate_decision_submission(
        state_root=root, submission=submission
    )
    replay = process_task_candidate_decision_submission(
        state_root=root, submission=submission
    )

    assert result["pipeline_approval_status"] == "approved"
    assert result["approved_task_definition"]["approval_status"] == "approved"
    assert result["decision_evidence_request"] is None
    assert result["proof_boundary"] == {
        "webapp_command_is_pipeline_approval": False,
        "pipeline_decision_recorded": True,
        "approved_task_exists": True,
        "decision_evidence_request_compiled": False,
        "testbed_required_before_request_compilation": True,
        "task_success_established": False,
        "physical_success_established": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    assert replay["already_exists"] is True
    changed = copy.deepcopy(submission)
    changed["command"]["rationale"] = "Different payload with the same command id."
    with pytest.raises(
        TaskCandidateControlPlaneError,
        match="command_request_id_idempotency_conflict",
    ):
        process_task_candidate_decision_submission(state_root=root, submission=changed)


def test_stale_candidate_actor_spoof_and_second_approval_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "task-control"
    discovery = _discovery()
    publish_task_candidate_discovery(
        state_root=root,
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=discovery,
    )
    spoofed = _submission(discovery, command_id="task-command-spoofed")
    spoofed["command"]["actor"]["identity"] = "firebase:different-user"
    with pytest.raises(TaskCandidateControlPlaneError, match="actor_requester_mismatch"):
        process_task_candidate_decision_submission(state_root=root, submission=spoofed)

    accepted = _submission(discovery)
    process_task_candidate_decision_submission(state_root=root, submission=accepted)
    second = _submission(discovery, command_id="task-command-2")
    second["command"]["idempotency_key"] = "approve-task-candidate-1-again"
    with pytest.raises(TaskCandidateControlPlaneError, match="already_approved"):
        process_task_candidate_decision_submission(state_root=root, submission=second)

    successor = _discovery("discovery-2")
    publish_task_candidate_discovery(
        state_root=root,
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=successor,
    )
    successor_result = process_task_candidate_decision_submission(
        state_root=root,
        submission=_submission(successor, command_id="task-command-successor"),
    )
    assert successor_result["pipeline_approval_status"] == "approved"


def test_signed_live_service_consumes_the_published_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "control" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    work_dir = tmp_path / "intake-work"
    token = "test-task-control-secret"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(work_dir))
    monkeypatch.setenv(
        service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"blueprint-webapp": token}),
    )
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.delenv(service.INTAKE_TOKEN_ENV, raising=False)
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    discovery = _discovery()
    publish_task_candidate_discovery(
        state_root=work_dir / "task_candidate_control_plane",
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=discovery,
    )
    submission = _submission(discovery)
    body = json.dumps(submission, separators=(",", ":"))
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = "task-decision-nonce-1"
    signature = hmac.new(
        token.encode("utf-8"),
        f"{timestamp}.blueprint-webapp.{nonce}.{body}".encode("utf-8"),
        "sha256",
    ).hexdigest()

    response = TestClient(service.create_app()).post(
        "/api/live-pipeline/task-decisions",
        content=body,
        headers={
            "content-type": "application/json",
            "x-blueprint-pipeline-timestamp": timestamp,
            "x-blueprint-pipeline-client-id": "blueprint-webapp",
            "x-blueprint-pipeline-nonce": nonce,
            "x-blueprint-pipeline-signature": f"sha256={signature}",
        },
    )

    assert response.status_code == 200
    assert response.json()["pipeline_approval_status"] == "approved"
    assert response.json()["decision_evidence_request"] is None


def test_discovery_sync_uses_signed_projection_and_preserves_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            receipt = {
                "schema_version": "capture_task_discovery_publication_receipt.v1",
                "status": "published",
                "already_exists": False,
                "capture_session_id": "capture-review-1",
                "intake_id": "intake-1",
                "discovery_digest": _discovery()["discovery_digest"],
                "proof_boundary": {
                    "candidate_is_customer_intent": False,
                    "decision_evidence_request_compiled": False,
                    "task_success_established": False,
                },
            }
            return BytesIO(json.dumps(receipt).encode("utf-8")).read()

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "blueprint_pipeline.task_candidate_control_plane.urllib_request.urlopen",
        fake_urlopen,
    )
    discovery = _discovery()
    result = publish_and_sync_task_candidate_discovery(
        state_root=tmp_path / "task-control",
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=discovery,
        endpoint_url="https://webapp.example/api/internal/pipeline/capture-task-discoveries",
        token="sync-secret",
        sync_required=True,
    )

    assert result["webapp_sync"]["status"] == "succeeded"
    request = captured["request"]
    assert getattr(request, "full_url") == (
        "https://webapp.example/api/internal/pipeline/capture-task-discoveries"
    )
    assert getattr(request, "headers")["X-blueprint-pipeline-signature"].startswith(
        "sha256="
    )
    assert "sync-secret" not in json.dumps(result)
    sync_files = list(
        (tmp_path / "task-control" / "sessions" / "capture-review-1" / "webapp_sync").glob(
            "*.json"
        )
    )
    assert len(sync_files) == 1

    class WrongResponse(Response):
        def read(self) -> bytes:
            return BytesIO(
                b'{"schema_version":"capture_task_discovery_publication_receipt.v1",'
                b'"status":"published","already_exists":false,'
                b'"capture_session_id":"different-session"}'
            ).read()

    monkeypatch.setattr(
        "blueprint_pipeline.task_candidate_control_plane.urllib_request.urlopen",
        lambda *_args, **_kwargs: WrongResponse(),
    )
    mismatched = sync_task_candidate_discovery_to_webapp(
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=discovery,
        endpoint_url="https://webapp.example/api/internal/pipeline/capture-task-discoveries",
        token="sync-secret",
        max_attempts=1,
    )
    assert mismatched["status"] == "failed"
    assert mismatched["reason"] == "response_binding_mismatch"

    monkeypatch.setattr(
        "blueprint_pipeline.task_candidate_control_plane.urllib_request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )
    failed = sync_task_candidate_discovery_to_webapp(
        capture_session_id="capture-review-1",
        intake_id="intake-1",
        discovery=discovery,
        endpoint_url="https://webapp.example/api/internal/pipeline/capture-task-discoveries",
        token="sync-secret",
        max_attempts=2,
    )
    assert failed["status"] == "failed"
    assert failed["attempts"] == 2
