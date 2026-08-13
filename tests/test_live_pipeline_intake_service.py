from __future__ import annotations

import hmac
import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.live_pipeline_control_plane import (
    CONTROL_PLANE_OUTPUT_PATH_ENV,
    run_live_pipeline_control_plane,
)
from blueprint_pipeline.live_pipeline_intake_service import (
    INTAKE_TOKEN_ENV,
    create_app,
)
from blueprint_pipeline.task_evaluation_supervisor import replay_supervisor_run
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    canonical_digest as launch_digest,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture(autouse=True)
def _allow_legacy_bearer_for_existing_intake_tests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "BLUEPRINT_CAPTURE_SUPERVISOR_AGENT_MODEL",
        "BLUEPRINT_CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK",
        "BLUEPRINT_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD",
        "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "true")
    monkeypatch.setenv(
        service.INTAKE_NONCE_STORE_DIR_ENV,
        str(tmp_path / "shared-nonce-store"),
    )
    service._INTAKE_NONCE_CACHE.clear()


def _signed_intake_headers(
    token: str,
    body: str,
    *,
    timestamp: str | None = None,
    nonce: str = "nonce-valid-1",
    client_id: str = "test-client",
) -> dict[str, str]:
    resolved_timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    signature = hmac.new(
        token.encode("utf-8"),
        f"{resolved_timestamp}.{client_id}.{nonce}.{body}".encode("utf-8"),
        "sha256",
    ).hexdigest()
    return {
        "content-type": "application/json",
        "x-blueprint-pipeline-timestamp": resolved_timestamp,
        "x-blueprint-pipeline-nonce": nonce,
        "x-blueprint-pipeline-client-id": client_id,
        "x-blueprint-pipeline-signature": f"sha256={signature}",
    }


def _legacy_webapp_headers(token: str, *, nonce: str, body: str = "") -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).isoformat()
    signature = hmac.new(
        token.encode("utf-8"),
        f"{timestamp}.{nonce}.{body}".encode("utf-8"),
        "sha256",
    ).hexdigest()
    return {
        "x-blueprint-pipeline-timestamp": timestamp,
        "x-blueprint-pipeline-nonce": nonce,
        "x-blueprint-pipeline-signature": f"sha256={signature}",
    }


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1"})
    return capture_root


def _control_manifest(tmp_path: Path, capture_root: Path) -> Path:
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=output_path,
    )
    return output_path


def _seed_robot_eval_dataset_cards(capture_root: Path) -> None:
    dataset_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        dataset_dir / "task_cards.json",
        {
            "schema_version": "robot_eval_task_cards.v1",
            "cards": [
                {
                    "task_id": "scene_anchor_geometry_0",
                    "task_card_id": "task-card-1",
                }
            ],
        },
    )
    _write_json(
        dataset_dir / "scenario_cards.json",
        {
            "schema_version": "robot_eval_scenario_cards.v1",
            "cards": [
                {
                    "scenario_id": "scenario_scene_anchor_geometry_0_unitree_g1",
                    "task_id": "scene_anchor_geometry_0",
                    "robot_profile_id": "unitree_g1",
                }
            ],
        },
    )


def _task_evaluation_launch_request() -> dict[str, object]:
    source_bundle = {
        "bundle_id": "interiorgs-sage-new-scene-001",
        "source_kind": "interiorgs_sage",
        "uri": "gs://blueprint-runs/interiorgs-sage-new-scene-001.zip",
        "digest": "sha256:" + "a" * 64,
    }
    request: dict[str, object] = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": "launch-interiorgs-sage-001",
        "run_id": "run-interiorgs-sage-001",
        "launch_profile_id": "interiorgs-sage-franka-001",
        "launch_profile_digest": "sha256:" + "b" * 64,
        "source_bundle": source_bundle,
        "evaluation_run_spec": {
            "uri": "gs://blueprint-runs/evaluation-run-spec-001.json",
            "digest": "sha256:" + "c" * 64,
        },
        "authorization": {
            "actor": {"id": "founder-001", "role": "admin"},
            "authorized_at": datetime.now(timezone.utc).isoformat(),
            "rights": {
                "approved": True,
                "scope": "interiorgs_sage_simulator_evaluation",
                "evidence": {
                    "uri": "firestore://taskEvaluationLaunchAuthorities/rights-001",
                    "digest": "sha256:" + "d" * 64,
                },
            },
            "spend": {
                "approved": True,
                "currency": "USD",
                "max_spend_usd": 2.0,
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
            "execution": {"approved": True},
        },
        "required_controls": {
            "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "secret_profile_id": "canonical-vast-adp",
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "claim_ceiling": "development_only",
        "idempotency_key": "launch-interiorgs-sage-001",
    }
    request["request_digest"] = launch_digest(request, digest_field="request_digest")
    return request


def _public_task_evaluation_launch_profile() -> dict[str, object]:
    request = _task_evaluation_launch_request()
    return {
        "profile_id": request["launch_profile_id"],
        "profile_digest": request["launch_profile_digest"],
        "source_bundle": request["source_bundle"],
        "evaluation_run_spec": request["evaluation_run_spec"],
        "required_controls": request["required_controls"],
        "execution_admission": {
            "live_enabled": False,
            "readiness_receipt": {
                "uri": "https://github.com/ognjhunt/BlueprintCapturePipeline/blob/commit/readiness.json",
                "digest": "sha256:" + "e" * 64,
            },
            "blockers": ["scripted_positive_control_not_passed"],
        },
        "claim_ceiling": request["claim_ceiling"],
        "required_authorization": {
            "max_spend_usd": 6.0,
            "hard_ttl_seconds": 5400,
        },
    }


def _webapp_request(capture_root: Path, *, job_id: str = "webapp-job-1") -> dict[str, object]:
    buyer_request_id = "buyer-request-1"
    return {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "job_request": {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "buyer_request_id": buyer_request_id,
            "site_package": {
                "capture_root": str(capture_root),
                "site_submission_id": "site-submission-1",
                "capture_job_id": "capture-job-1",
                "buyer_request_id": buyer_request_id,
                "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
            },
            "owner_system": {
                "name": "Blueprint-WebApp",
                "request_id": job_id,
                "buyer_request_id": buyer_request_id,
                "site_submission_id": "site-submission-1",
                "capture_job_id": "capture-job-1",
            },
            "source": {
                "system": "Blueprint-WebApp",
                "selection_state": {
                    "buyer_request_id": buyer_request_id,
                    "site_submission_id": "site-submission-1",
                    "capture_job_id": "capture-job-1",
                },
            },
        },
    }


def _decision_evidence_envelope() -> dict[str, object]:
    return {
        "queue_contract": "blueprint.decision_evidence_request_inbox.v1",
        "request_id": "decision-request-1",
        "decision_id": "buyer-decision-1",
        "decision_request": {
            "schema_version": "blueprint.decision_evidence_request.v1",
            "request_id": "decision-request-1",
            "decision_id": "buyer-decision-1",
            "testbed": {"manifest_uri": "gs://local-blueprint/testbed.json"},
            "site_task": {
                "site_id": "site-1",
                "site_name": "site-one",
                "task_id": "task-a",
            },
            "candidates": [{"candidate_id": "policy-a", "kind": "policy"}],
            "routing_authority": {
                "system": "BlueprintCapturePipeline",
                "webapp_backend_selection_allowed": False,
            },
            "authorization": {"access_state": "provisioned"},
        },
    }


def _capture_handoff() -> dict[str, object]:
    return {
        "schema_version": "blueprint_capture_pipeline_handoff.v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "site_submission_id": "site-submission-1",
        "buyer_request_id": "buyer-request-1",
        "capture_job_id": "capture-job-1",
        "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
        "requested_lanes": ["evaluation_prep", "robot_eval_dataset", "task_evaluation_run"],
        "robot_eval_dataset_requested": True,
        "capture_descriptor_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        "pipeline_handoff_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline_handoff.json",
    }


def _live_closure_evidence(job_id: str = "webapp-job-1") -> dict[str, object]:
    return {
        "schema_version": "live_robot_eval_closure_evidence.v1",
        "job_id": job_id,
        "review_acceptance": {"accepted": True, "reviewer": "owner-reviewer"},
        "delivery": {
            "storage_upload_performed": True,
            "signed_urls": ["https://delivery.example/signed/package-1"],
            "entitlement_verified": True,
            "buyer_access_check": {
                "buyer_access_checked": True,
                "buyer_accessible": True,
                "status": "ok",
            },
        },
        "safety_contact_physics": {
            "physics_contact_validated": True,
            "non_ranking_operational_claim_validated": True,
            "rank_fidelity_result_proven": True,
            "methodology_uri_or_path": "owner://methodology",
            "contact_validation_uri_or_path": "owner://contact",
            "non_ranking_operational_claim_uri_or_path": "owner://safety",
            "operator_attestation": {
                "attested_by": "safety-owner",
                "attestation": "Owner accepted contact, physics, and safety evidence.",
            },
        },
    }


def test_live_pipeline_intake_service_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "control" / "manifest.json"
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "custom-work"))
    assert service._work_dir(manifest_path) == tmp_path / "custom-work"
    assert (
        service._request_from_payload({"schema_version": "robot_eval_job_request.v1"})[
            "schema_version"
        ]
        == "robot_eval_job_request.v1"
    )
    assert service._request_from_payload({"schema_version": "other"}) == {}
    assert service._first_string("", None) == ""
    assert service._list_from_payload(("a", "b")) == ["a", "b"]
    assert service._list_from_payload("bad") == []
    cards_path = tmp_path / "cards.json"
    _write_json(cards_path, [{"task_id": "task-1"}, "bad"])
    assert service._cards_from_file(cards_path) == [{"task_id": "task-1"}]

    missing_root = tmp_path / "missing-cards"
    assert service._select_dataset_task(missing_root) == (
        None,
        ["robot_eval_task_cards_missing", "robot_eval_scenario_cards_missing"],
    )
    empty_root = tmp_path / "empty-cards"
    _write_json(empty_root / "pipeline" / "robot_eval_dataset" / "task_cards.json", {"cards": []})
    _write_json(
        empty_root / "pipeline" / "robot_eval_dataset" / "scenario_cards.json", {"cards": []}
    )
    assert service._select_dataset_task(empty_root) == (
        None,
        ["robot_eval_task_cards_empty", "robot_eval_scenario_cards_empty"],
    )
    unmatched_root = tmp_path / "unmatched-cards"
    _write_json(
        unmatched_root / "pipeline" / "robot_eval_dataset" / "task_cards.json",
        {"cards": [{}, {"task_id": "task-1"}]},
    )
    _write_json(
        unmatched_root / "pipeline" / "robot_eval_dataset" / "scenario_cards.json",
        {"cards": [{"task_id": "other", "scenario_id": "scenario-1"}]},
    )
    assert service._select_dataset_task(unmatched_root) == (
        None,
        ["robot_eval_no_task_scenario_pair"],
    )


def test_capture_handoff_blocker_edges(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    payload = {
        "scene_id": "other-scene",
        "capture_id": "other-capture",
        "requested_outputs": ["task_evaluation_run"],
    }
    envelope, audit = service._capture_handoff_to_webapp_request(
        payload=payload, capture_root=capture_root
    )

    assert envelope is None
    assert "capture_handoff_scene_id_mismatch" in audit["blockers"]
    assert "capture_handoff_capture_id_mismatch" in audit["blockers"]
    assert "capture_handoff_missing_site_submission_id" in audit["blockers"]


def test_capture_handoff_rejects_dataset_older_than_upload_complete(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_robot_eval_dataset_cards(capture_root)
    _write_json(
        capture_root / "raw" / "capture_upload_complete.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "upload_completed_at": "2999-01-01T00:00:00Z",
            "upload_run_id": "upload-run-newer-than-dataset",
        },
    )

    envelope, audit = service._capture_handoff_to_webapp_request(
        payload=_capture_handoff(),
        capture_root=capture_root,
    )

    assert envelope is None
    assert "robot_eval_dataset_stale_for_capture_upload_complete" in audit["blockers"]


def test_capture_handoff_job_id_changes_for_distinct_upload_marker(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_json(
        capture_root / "raw" / "capture_upload_complete.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "upload_completed_at": "2026-01-01T00:00:00Z",
            "upload_run_id": "upload-run-1",
        },
    )
    _seed_robot_eval_dataset_cards(capture_root)

    first, first_audit = service._capture_handoff_to_webapp_request(
        payload=_capture_handoff(),
        capture_root=capture_root,
    )

    _write_json(
        capture_root / "raw" / "capture_upload_complete.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "upload_completed_at": "2026-01-01T00:00:00Z",
            "upload_run_id": "upload-run-2",
        },
    )
    second, second_audit = service._capture_handoff_to_webapp_request(
        payload=_capture_handoff(),
        capture_root=capture_root,
    )

    assert first is not None
    assert second is not None
    assert first_audit["job_id"] != second_audit["job_id"]
    assert first["job_request"]["job_id"] != second["job_request"]["job_id"]


def test_trigger_control_plane_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    # No subprocess may be spawned while the trigger is gated/unconfigured.
    def _no_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be invoked when trigger is gated")

    monkeypatch.setattr(service.subprocess, "run", _no_subprocess)

    # unit set but allow-env unset -> blocked, no spawn, exact missing-env blocker.
    monkeypatch.setenv(
        service.INTAKE_TRIGGER_SYSTEMD_UNIT_ENV,
        "blueprint-pipeline-control-plane.service",
    )
    monkeypatch.delenv(service.INTAKE_ALLOW_TRIGGER_ENV, raising=False)
    blocked = service._trigger_control_plane()
    assert blocked["status"] == "blocked"
    assert blocked["performed"] is False
    assert blocked["systemd_unit_configured"] is True
    assert blocked["blockers"] == [f"missing_env_{service.INTAKE_ALLOW_TRIGGER_ENV}"]
    assert service.INTAKE_ALLOW_TRIGGER_ENV == "BLUEPRINT_ALLOW_LIVE_PIPELINE_INTAKE_TRIGGER"

    # no unit configured -> not_configured, no spawn.
    monkeypatch.delenv(service.INTAKE_TRIGGER_SYSTEMD_UNIT_ENV, raising=False)
    monkeypatch.setenv(service.INTAKE_ALLOW_TRIGGER_ENV, "true")
    not_configured = service._trigger_control_plane()
    assert not_configured["status"] == "not_configured"
    assert not_configured["performed"] is False
    assert not_configured["systemd_unit_configured"] is False

    class Completed:
        returncode = 0
        stdout = "x" * 2100
        stderr = "err"

    monkeypatch.setenv(
        service.INTAKE_TRIGGER_SYSTEMD_UNIT_ENV,
        "blueprint-pipeline-control-plane.service",
    )
    monkeypatch.setenv(service.INTAKE_ALLOW_TRIGGER_ENV, "true")
    seen: dict[str, object] = {}

    def _safe_run(command: object, **kwargs: object) -> Completed:
        seen["command"] = command
        seen["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(service.subprocess, "run", _safe_run)
    triggered = service._trigger_control_plane()
    assert triggered["status"] == "triggered"
    assert triggered["performed"] is True
    assert len(triggered["stdout_tail"]) == 2000
    assert seen["command"] == [
        "systemctl",
        "start",
        "--no-block",
        "blueprint-pipeline-control-plane.service",
    ]
    assert seen["kwargs"]["shell"] is False  # type: ignore[index]

    monkeypatch.setenv(service.INTAKE_TRIGGER_SYSTEMD_UNIT_ENV, "bad;unit.service")
    invalid = service._trigger_control_plane()
    assert invalid["status"] == "blocked"
    assert invalid["blockers"] == ["intake_trigger_systemd_unit_invalid"]


def test_deployment_identity_fails_closed_without_exact_source_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app())

    # "No identity" now has to be arranged rather than assumed: the endpoint has
    # two sources, and a CI checkout is detached at a real commit, which the
    # service is right to report.
    monkeypatch.setattr(service, "running_source_commit", lambda *_a, **_k: "")
    monkeypatch.delenv(service.PIPELINE_SOURCE_COMMIT_ENV, raising=False)
    missing = client.get("/api/live-pipeline/version")
    assert missing.status_code == 503
    assert missing.json() == {
        "schema_version": service.DEPLOYMENT_IDENTITY_SCHEMA_VERSION,
        "service_schema_version": service.INTAKE_SCHEMA_VERSION,
        "commit_proven": False,
        "source_commit": None,
        # The payload now says where the answer came from, and a development
        # checkout on a branch is not a release worktree to read a commit from.
        "source_commit_source": "none",
        "blockers": ["deployment_identity_source_commit_unavailable"],
        "claim_ceiling": "deployed_service_identity_only",
        "default_object_removal": {
            "policy": "candidate_schema_resolved.v1",
            "paired_target_pipeline_mode": "dual_target_artifixer3d_only",
            "maximum_replacement_objects": 5,
            "generated_appearance_is_physical_evidence": False,
        },
    }

    monkeypatch.setenv(service.PIPELINE_SOURCE_COMMIT_ENV, "not-a-commit")
    assert client.get("/api/live-pipeline/version").status_code == 503

    source_commit = "a" * 40
    monkeypatch.setenv(service.PIPELINE_SOURCE_COMMIT_ENV, source_commit)
    proven = client.get("/api/live-pipeline/version")
    assert proven.status_code == 200
    assert proven.json()["commit_proven"] is True
    assert proven.json()["source_commit"] == source_commit
    assert proven.json()["default_object_removal"] == {
        "policy": "candidate_schema_resolved.v1",
        "paired_target_pipeline_mode": "dual_target_artifixer3d_only",
        "maximum_replacement_objects": 5,
        "generated_appearance_is_physical_evidence": False,
    }


def test_live_pipeline_intake_service_error_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(create_app())
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "token")
    missing_manifest = tmp_path / "missing-manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(missing_manifest))
    headers = {"x-blueprint-intake-token": "token"}

    health = client.get("/health").json()
    assert health["control_plane_ready"] is False
    assert health["authentication_configured"] is True
    assert health["token_configured"] is True
    assert health["task_evaluation_supervisor"]["agent_harness"] == "openai_agents_sdk"
    assert health["task_evaluation_supervisor"]["configuration_status"] == "valid"
    assert health["task_evaluation_supervisor"]["zero_spend_lifecycle_ready"] is True
    assert health["task_evaluation_supervisor"]["live_inference_configured"] is False
    assert health["task_evaluation_supervisor"]["live_inference_ready"] is False
    assert health["task_evaluation_supervisor"]["execution_profile_digest"].startswith(
        "sha256:"
    )
    assert (
        health["task_evaluation_supervisor"]["proof_or_recovery_authority_granted"] is False
    )
    assert "manifest_path" not in health
    assert "endpoints" not in health
    assert (
        client.post(
            "/api/live-pipeline/job-requests", json={}, headers={"x-blueprint-intake-token": "bad"}
        ).status_code
        == 401
    )

    endpoints = [
        "/api/live-pipeline/job-requests",
        "/api/live-pipeline/capture-upload-intakes",
        "/api/live-pipeline/capture-handoffs",
        "/api/live-pipeline/policy-packages",
        "/api/live-pipeline/real-robot-pov",
        "/api/live-pipeline/deployment-outcomes",
        "/api/live-pipeline/live-closure-evidence",
    ]
    for endpoint in endpoints:
        assert (
            client.post(
                endpoint, data="{", headers={**headers, "content-type": "application/json"}
            ).status_code
            == 400
        )
        assert client.post(endpoint, json=[], headers=headers).status_code == 400
        assert client.post(endpoint, json={}, headers=headers).status_code == 503

    manifest_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    _write_json(manifest_path, ["bad"])
    assert (
        client.post("/api/live-pipeline/capture-handoffs", json={}, headers=headers).status_code
        == 503
    )
    _write_json(manifest_path, {})
    assert (
        client.post("/api/live-pipeline/capture-handoffs", json={}, headers=headers).status_code
        == 503
    )

    assert client.get("/api/live-pipeline/intake-audit", headers=headers).status_code == 404
    _write_json(manifest_path.parent / "live_pipeline_input_intake_audit.json", ["bad"])
    assert client.get("/api/live-pipeline/intake-audit", headers=headers).status_code == 500


def test_live_pipeline_capture_upload_intake_is_authenticated_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "token")
    monkeypatch.setenv(
        CONTROL_PLANE_OUTPUT_PATH_ENV,
        str(tmp_path / "control" / "manifest.json"),
    )
    monkeypatch.setenv(
        service.CAPTURE_UPLOAD_STORE_ROOT_ENV,
        str(tmp_path / "capture-intakes"),
    )
    monkeypatch.setenv("BLUEPRINT_CAPTURE_SUPERVISOR_AGENT_MODEL", "gpt-5-mini")
    monkeypatch.setenv("BLUEPRINT_CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK", "true")
    monkeypatch.setenv("BLUEPRINT_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD", "0.5")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "true")
    seen: dict[str, object] = {}

    def process(payload, *, store_root):
        seen["payload"] = payload
        seen["store_root"] = store_root
        return {
            "schema_version": "capture_upload_intake_receipt.v1",
            "capture_session_id": "capture-upload-session-1",
            "intake_id": "intake-1",
            "request_digest": f"sha256:{'1' * 64}",
            "envelope_digest": f"sha256:{'2' * 64}",
            "capture_digest": f"sha256:{'3' * 64}",
            "size_bytes": 12,
            "admission_status": "accepted",
            "state": "capture_accepted",
            "claim_ceiling": {"physical_task_success": False},
            "artifact_reference": {
                "uri": "intakes/intake-1/fixture",
                "envelope_digest": f"sha256:{'2' * 64}",
            },
            "malware_content_validation": {
                "status": "passed",
                "scanner": "fixture",
            },
            "capture_qa_report": {"qa_report_digest": f"sha256:{'4' * 64}"},
            "already_exists": False,
            "proof_boundary": {
                "capture_qa_completed": False,
                "physical_task_success_established": False,
                "comparative_policy_ranking_verdict": "thesis_not_supported",
            },
        }

    monkeypatch.setattr(service, "process_capture_upload_submission", process)
    monkeypatch.setattr(
        service,
        "build_capture_qa_webapp_publication",
        lambda *, capture_session_id, report: {
            "schema_version": "capture_qa_publication.v1",
            "capture_session_id": capture_session_id,
            "qa_report_digest": report["qa_report_digest"],
        },
    )
    def supervise(
        *,
        capture_root,
        agent_model,
        allow_live_agents_sdk,
        agent_inference_budget_usd,
    ):
        seen["supervisor_options"] = {
            "agent_model": agent_model,
            "allow_live_agents_sdk": allow_live_agents_sdk,
            "agent_inference_budget_usd": agent_inference_budget_usd,
        }
        return {
            "schema_version": "task_evaluation_capture_supervisor_lifecycle.v3",
            "status": "blocked",
            "run_id": "capture-supervisor-fixture",
            "capture_build_alone_can_start_run": True,
            "proof_state_mutated_by_agent": False,
            "capture_root": str(capture_root),
        }

    monkeypatch.setattr(service, "run_capture_build_supervisor", supervise)
    submission = {
        "schema_version": "capture_upload_transfer_submission.v1",
        "capture_session_id": "capture-upload-session-1",
        "transfer": {
            "url": "https://download.example.test/file/capture.mp4",
            "authorization": "ephemeral-secret",
        },
    }
    client = TestClient(create_app())
    supervisor_health = client.get("/health").json()["task_evaluation_supervisor"]
    assert supervisor_health["configuration_status"] == "valid"
    assert supervisor_health["live_inference_configured"] is True
    assert supervisor_health["live_operator_gate_enabled"] is True
    assert supervisor_health["live_inference_ready"] is True
    assert supervisor_health["execution_profile_digest"].startswith("sha256:")
    response = client.post(
        "/api/live-pipeline/capture-upload-intakes",
        json=submission,
        headers={"x-blueprint-intake-token": "token"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["schema_version"] == "capture_upload_processing_result.v1"
    assert result["receipt"]["capture_digest"] == f"sha256:{'3' * 64}"
    assert result["capture_qa_publication"]["qa_report_digest"] == f"sha256:{'4' * 64}"
    assert result["task_evaluation_supervisor"]["status"] == "blocked"
    assert result["task_evaluation_supervisor"]["capture_build_alone_can_start_run"] is True
    assert result["task_evaluation_supervisor"]["capture_root"].endswith(
        "intakes/intake-1/fixture"
    )
    assert "ephemeral-secret" not in response.text
    assert "download.example.test" not in response.text
    assert seen["payload"] == submission
    assert seen["store_root"] == (tmp_path / "capture-intakes").resolve()
    assert seen["supervisor_options"] == {
        "agent_model": "gpt-5-mini",
        "allow_live_agents_sdk": True,
        "agent_inference_budget_usd": 0.5,
    }


def test_live_pipeline_capture_upload_refuses_invalid_supervisor_execution_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "token")
    monkeypatch.setenv(
        CONTROL_PLANE_OUTPUT_PATH_ENV,
        str(tmp_path / "control" / "manifest.json"),
    )
    monkeypatch.setenv(
        service.CAPTURE_UPLOAD_STORE_ROOT_ENV,
        str(tmp_path / "capture-intakes"),
    )
    monkeypatch.setenv("BLUEPRINT_CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK", "true")
    monkeypatch.setenv("BLUEPRINT_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD", "0")
    processing_started = False

    def process(*_args, **_kwargs):
        nonlocal processing_started
        processing_started = True
        raise AssertionError("capture processing must not start with invalid supervisor authority")

    monkeypatch.setattr(service, "process_capture_upload_submission", process)
    client = TestClient(create_app())
    supervisor_health = client.get("/health").json()["task_evaluation_supervisor"]
    assert supervisor_health == {
        "agent_harness": "openai_agents_sdk",
        "configuration_status": "invalid",
        "zero_spend_lifecycle_ready": False,
        "live_inference_configured": False,
        "live_operator_gate_enabled": False,
        "live_inference_ready": False,
        "execution_profile_digest": None,
        "proof_or_recovery_authority_granted": False,
    }
    response = client.post(
        "/api/live-pipeline/capture-upload-intakes",
        json={
            "schema_version": "capture_upload_transfer_submission.v1",
            "capture_session_id": "invalid-supervisor-config",
        },
        headers={"x-blueprint-intake-token": "token"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "capture supervisor execution configuration is invalid"
    assert processing_started is False


def test_live_pipeline_capture_upload_starts_real_replayable_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "token")
    monkeypatch.setenv(
        CONTROL_PLANE_OUTPUT_PATH_ENV,
        str(tmp_path / "control" / "manifest.json"),
    )
    store_root = (tmp_path / "capture-intakes").resolve()
    monkeypatch.setenv(service.CAPTURE_UPLOAD_STORE_ROOT_ENV, str(store_root))

    def process(payload, *, store_root):
        artifact_root = store_root / "intakes" / "intake-real" / "capture"
        _write_json(
            artifact_root / "capture_intake_envelope.json",
            {
                "schema_version": "capture_intake_envelope.v1",
                "capture_session_id": payload["capture_session_id"],
                "intake_id": "intake-real",
                "admission_status": "accepted",
                "state": "capture_accepted",
                "capture_digest": f"sha256:{'3' * 64}",
                "claim_ceiling": {"physical_task_success": False},
                "proof_boundary": {
                    "capture_qa_completed": False,
                    "physical_task_success_established": False,
                    "comparative_policy_ranking_verdict": "thesis_not_supported",
                },
            },
        )
        return {
            "schema_version": "capture_upload_intake_receipt.v1",
            "capture_session_id": payload["capture_session_id"],
            "intake_id": "intake-real",
            "request_digest": f"sha256:{'1' * 64}",
            "envelope_digest": f"sha256:{'2' * 64}",
            "capture_digest": f"sha256:{'3' * 64}",
            "size_bytes": 12,
            "admission_status": "accepted",
            "state": "capture_accepted",
            "claim_ceiling": {"physical_task_success": False},
            "artifact_reference": {
                "uri": "intakes/intake-real/capture",
                "envelope_digest": f"sha256:{'2' * 64}",
            },
            "malware_content_validation": {
                "status": "passed",
                "scanner": "fixture",
            },
            "capture_qa_report": {"qa_report_digest": f"sha256:{'4' * 64}"},
            "already_exists": False,
            "proof_boundary": {
                "capture_qa_completed": False,
                "physical_task_success_established": False,
                "comparative_policy_ranking_verdict": "thesis_not_supported",
            },
        }

    monkeypatch.setattr(service, "process_capture_upload_submission", process)
    monkeypatch.setattr(
        service,
        "build_capture_qa_webapp_publication",
        lambda *, capture_session_id, report: {
            "schema_version": "capture_qa_publication.v1",
            "capture_session_id": capture_session_id,
            "qa_report_digest": report["qa_report_digest"],
        },
    )
    submission = {
        "schema_version": "capture_upload_transfer_submission.v1",
        "capture_session_id": "capture-upload-real-supervisor",
        "transfer": {"url": "https://download.example.test/file/capture.mp4"},
    }
    response = TestClient(create_app()).post(
        "/api/live-pipeline/capture-upload-intakes",
        json=submission,
        headers={"x-blueprint-intake-token": "token"},
    )

    assert response.status_code == 200
    lifecycle = response.json()["task_evaluation_supervisor"]
    assert lifecycle["status"] == "blocked"
    assert lifecycle["agent_harness"] == "openai_agents_sdk"
    assert lifecycle["autonomy_mode"] == "execute_non_spend"
    assert lifecycle["capture_build_alone_can_start_run"] is True
    assert lifecycle["all_six_capabilities_registered"] is True
    assert lifecycle["proof_state_mutated_by_agent"] is False
    output_dir = Path(lifecycle["output_dir"])
    assert output_dir.is_relative_to(store_root / "intakes" / "intake-real" / "capture")
    assert Path(lifecycle["event_ledger_path"]).is_file()
    assert Path(lifecycle["terminal_report_path"]).is_file()
    assert replay_supervisor_run(output_dir)["status"] == "replay_verified"


def test_live_pipeline_capture_upload_intake_returns_typed_fail_closed_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "token")
    monkeypatch.setenv(
        CONTROL_PLANE_OUTPUT_PATH_ENV,
        str(tmp_path / "control" / "manifest.json"),
    )
    monkeypatch.setenv(service.CAPTURE_UPLOAD_STORE_ROOT_ENV, str(tmp_path / "store"))

    def reject(*_args, **_kwargs):
        raise service.CaptureUploadTransferError(["malware_detected"])

    monkeypatch.setattr(service, "process_capture_upload_submission", reject)
    response = TestClient(create_app()).post(
        "/api/live-pipeline/capture-upload-intakes",
        json={"schema_version": "capture_upload_transfer_submission.v1"},
        headers={"x-blueprint-intake-token": "token"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "schema_version": "capture_upload_intake_rejection.v1",
        "status": "rejected",
        "blockers": ["malware_detected"],
        "proof_boundary": {
            "capture_qa_completed": False,
            "task_success_established": False,
            "physical_task_success_established": False,
            "deployment_or_safety_approved": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }


def test_capture_handoff_blocked_after_conversion_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_robot_eval_dataset_cards(capture_root)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "token")
    monkeypatch.setattr(
        service,
        "build_live_pipeline_input_intake",
        lambda **_kwargs: {"status": "blocked", "input_blockers": ["blocked_after_conversion"]},
    )

    response = TestClient(create_app()).post(
        "/api/live-pipeline/capture-handoffs",
        json=_capture_handoff(),
        headers={"x-blueprint-intake-token": "token"},
    )

    assert response.status_code == 202
    assert response.json()["capture_handoff"]["converted_to_job_request"] is True

    calls: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(
            run=lambda app, host, port: calls.update({"app": app, "host": host, "port": port})
        ),
    )
    assert service.main(["--host", "0.0.0.0", "--port", "9999"]) == 0
    assert calls == {
        "app": "blueprint_pipeline.live_pipeline_intake_service:app",
        "host": "0.0.0.0",
        "port": 9999,
    }


def _deployment_outcomes(
    job_id: str = "webapp-job-1",
    *,
    include_evidence: bool = True,
) -> dict[str, object]:
    record: dict[str, object] = {
        "outcome_id": "pilot-outcome-1",
        "task_id": "place_return_in_bin",
        "scenario_id": "scenario_place_return_in_bin_mobile",
        "actual_success": False,
        "failure_mode_ids": ["missed_blocked_path"],
    }
    if include_evidence:
        record["evidence_refs"] = {"pilot_log": "owner://pilot/pilot-outcome-1"}
    return {
        "schema_version": "deployment_outcome_manifest.v1",
        "job_id": job_id,
        "records": [record],
    }


def _policy_package(job_id: str = "webapp-job-1") -> dict[str, object]:
    return {
        "schema_version": "robot_team_policy_package.v1",
        "job_id": job_id,
        "policy_package": {
            "policy_api_endpoint": {
                "endpoint_url": "https://robot-team.example/policy",
                "observation_schema_ref": "schemas/obs-v1.json",
                "action_schema_ref": "schemas/action-v1.json",
            }
        },
    }


def _real_robot_pov_manifest(job_id: str = "webapp-job-1") -> dict[str, object]:
    return {
        "schema_version": "real_robot_pov_manifest.v1",
        "job_id": job_id,
        "owner_system": "robot-team-owner-system",
        "records": [
            {
                "evidence_id": "real-pov-1",
                "task_id": "place_return_in_bin",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "scenario_eval_run_id": "scenario-run-1",
                "scenario_variation_instance_id": "scenario-variation-1",
                "robot_camera_video_uri": "owner://pov/scenario-run-1.mp4",
                "action_log_uri": "owner://actions/scenario-run-1.jsonl",
                "timestamp_alignment": "aligned_to_scenario_eval_run",
                "owner_evidence_refs": {
                    "camera": "owner://pov/scenario-run-1.mp4",
                    "action_log": "owner://actions/scenario-run-1.jsonl",
                },
                "operator_attestation": {
                    "attested_by": "robot-team-ops",
                    "attestation": "Robot POV and action logs are aligned to this eval run.",
                },
            }
        ],
    }


def test_live_pipeline_intake_service_requires_token(tmp_path: Path, monkeypatch) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.delenv(INTAKE_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.post("/api/live-pipeline/job-requests", json=_webapp_request(capture_root))

    assert response.status_code == 503


def test_live_pipeline_intake_service_rejects_legacy_bearer_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "missing-manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/job-requests",
        json={},
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 401
    assert "HMAC signature" in response.text


def test_live_pipeline_intake_service_accepts_signed_request_and_rejects_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "missing-manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    client = TestClient(create_app())
    body = "{}"
    headers = _signed_intake_headers("test-intake-token", body, nonce="nonce-valid-2")

    first = client.post("/api/live-pipeline/job-requests", data=body, headers=headers)
    replay = client.post("/api/live-pipeline/job-requests", data=body, headers=headers)

    assert first.status_code == 503
    assert replay.status_code == 401
    assert "replayed intake signature nonce" in replay.text


def test_legacy_webapp_hmac_compatibility_is_explicit_scoped_and_replay_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_WEBAPP_HMAC_ENV, "true")
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "incoming"))
    client = TestClient(create_app())
    headers = _legacy_webapp_headers("test-intake-token", nonce="legacy-audit-nonce-1")
    first = client.get("/api/live-pipeline/intake-audit", headers=headers)
    replay = client.get("/api/live-pipeline/intake-audit", headers=headers)
    # Authentication succeeded and the endpoint executed; no audit has been
    # staged in this isolated app, so its expected application response is 404.
    assert first.status_code == 404
    assert replay.status_code == 401
    assert "replayed intake signature nonce" in replay.text

    body = json.dumps({}, separators=(",", ":"))
    post = client.post(
        "/api/live-pipeline/job-requests",
        content=body,
        headers={
            **_legacy_webapp_headers("test-intake-token", nonce="legacy-post-nonce-1", body=body),
            "content-type": "application/json",
        },
    )
    assert post.status_code != 401

    # The opt-in does not authorize unrelated intake endpoints.
    unrelated = client.post(
        "/api/live-pipeline/capture-handoffs",
        content=body,
        headers={
            **_legacy_webapp_headers("test-intake-token", nonce="legacy-post-nonce-2", body=body),
            "content-type": "application/json",
        },
    )
    assert unrelated.status_code == 401

    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_WEBAPP_HMAC_ENV)
    rejected = client.get(
        "/api/live-pipeline/intake-audit",
        headers=_legacy_webapp_headers("test-intake-token", nonce="legacy-audit-nonce-2"),
    )
    assert rejected.status_code == 401


def test_signed_nonce_replay_is_rejected_across_app_instances_and_cache_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "missing-manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    body = "{}"
    headers = _signed_intake_headers("test-intake-token", body, nonce="nonce-cross-process-1")

    first = TestClient(create_app()).post(
        "/api/live-pipeline/job-requests", data=body, headers=headers
    )
    service._INTAKE_NONCE_CACHE.clear()
    replay_after_restart = TestClient(create_app()).post(
        "/api/live-pipeline/job-requests", data=body, headers=headers
    )

    assert first.status_code == 503
    assert replay_after_restart.status_code == 401
    assert "replayed intake signature nonce" in replay_after_restart.text
    nonce_claims = list(Path(os.environ[service.INTAKE_NONCE_STORE_DIR_ENV]).glob("*.json"))
    assert len(nonce_claims) == 1


def test_task_evaluation_launch_is_immutably_queued_before_async_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = tmp_path / "launch-queue"
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    catalog_path = tmp_path / "published-launch-profiles.json"
    _write_json(catalog_path, [_public_task_evaluation_launch_profile()])
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(tmp_path / "control.json"))
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": "token"}))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_QUEUE_ROOT_ENV, str(queue_root))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV, str(profile_dir))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH_ENV, str(catalog_path))
    monkeypatch.setenv(
        service.TASK_EVALUATION_LAUNCH_TRIGGER_SYSTEMD_UNIT_ENV,
        "blueprint-task-evaluation-launch-dispatcher.service",
    )
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_ALLOW_TRIGGER_ENV, "true")
    monkeypatch.delenv(service.TASK_EVALUATION_LAUNCH_EXECUTE_ENV, raising=False)
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    systemctl_calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        systemctl_calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(service.subprocess, "run", fake_run)
    payload = _task_evaluation_launch_request()
    body = json.dumps(payload, separators=(",", ":"))
    response = TestClient(create_app()).post(
        "/api/live-pipeline/task-evaluation-launches",
        data=body,
        headers=_signed_intake_headers(
            "token",
            body,
            nonce="task-launch-nonce-1",
            client_id="blueprint-webapp",
        ),
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert response.json()["provider_mutation_performed_inside_http_request"] is False
    assert response.json()["canonical_allocator_required"] is True
    queued = list((queue_root / "pending").glob("*.json"))
    assert len(queued) == 1
    assert json.loads(queued[0].read_text(encoding="utf-8"))["request_digest"] == payload[
        "request_digest"
    ]
    assert systemctl_calls == [
        [
            "systemctl",
            "start",
            "--no-block",
            "blueprint-task-evaluation-launch-dispatcher.service",
        ]
    ]


def test_task_evaluation_launch_rejects_tampering_before_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(tmp_path / "control.json"))
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": "token"}))
    catalog_path = tmp_path / "published-launch-profiles.json"
    _write_json(catalog_path, [_public_task_evaluation_launch_profile()])
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH_ENV, str(catalog_path))
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda argv, **_kwargs: calls.append(list(argv)) or SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    payload = _task_evaluation_launch_request()
    payload["launch_profile_digest"] = "sha256:" + "f" * 64
    body = json.dumps(payload, separators=(",", ":"))
    response = TestClient(create_app()).post(
        "/api/live-pipeline/task-evaluation-launches",
        data=body,
        headers=_signed_intake_headers(
            "token",
            body,
            nonce="task-launch-nonce-2",
            client_id="blueprint-webapp",
        ),
    )
    assert response.status_code == 422
    assert "launch_request_digest_mismatch" in response.text
    assert calls == []


def test_task_evaluation_launch_rejects_an_unpublished_signed_profile_before_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = tmp_path / "launch-queue"
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    catalog_path = tmp_path / "published-launch-profiles.json"
    _write_json(catalog_path, [_public_task_evaluation_launch_profile()])
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(tmp_path / "control.json"))
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": "token"}))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_QUEUE_ROOT_ENV, str(queue_root))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV, str(profile_dir))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH_ENV, str(catalog_path))
    monkeypatch.setenv(
        service.TASK_EVALUATION_LAUNCH_TRIGGER_SYSTEMD_UNIT_ENV,
        "blueprint-task-evaluation-launch-dispatcher.service",
    )
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_ALLOW_TRIGGER_ENV, "true")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda argv, **_kwargs: calls.append(list(argv)) or SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    payload = _task_evaluation_launch_request()
    payload["launch_profile_id"] = "interiorgs-sage-franka-unpublished"
    payload["launch_profile_digest"] = "sha256:" + "f" * 64
    payload["request_digest"] = launch_digest(payload, digest_field="request_digest")
    body = json.dumps(payload, separators=(",", ":"))

    response = TestClient(create_app()).post(
        "/api/live-pipeline/task-evaluation-launches",
        data=body,
        headers=_signed_intake_headers(
            "token",
            body,
            nonce="task-launch-unpublished-profile-nonce",
            client_id="blueprint-webapp",
        ),
    )

    assert response.status_code == 422
    assert "launch_profile_not_published" in response.text
    assert not list((queue_root / "pending").glob("*.json"))
    assert calls == []


def test_task_evaluation_launch_requires_a_published_catalog_before_queueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = tmp_path / "launch-queue"
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(tmp_path / "control.json"))
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": "token"}))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_QUEUE_ROOT_ENV, str(queue_root))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV, str(profile_dir))
    monkeypatch.delenv(service.TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH_ENV, raising=False)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda argv, **_kwargs: calls.append(list(argv)) or SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    payload = _task_evaluation_launch_request()
    body = json.dumps(payload, separators=(",", ":"))

    response = TestClient(create_app()).post(
        "/api/live-pipeline/task-evaluation-launches",
        data=body,
        headers=_signed_intake_headers(
            "token",
            body,
            nonce="task-launch-missing-catalog-nonce",
            client_id="blueprint-webapp",
        ),
    )

    assert response.status_code == 503
    assert response.json()["blockers"] == [
        "task_evaluation_launch_public_catalog_not_configured"
    ]
    assert not list((queue_root / "pending").glob("*.json"))
    assert calls == []


def test_public_task_evaluation_profile_catalog_is_validated_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_json(catalog_path, [_public_task_evaluation_launch_profile()])
    monkeypatch.setenv(
        service.TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH_ENV,
        str(catalog_path),
    )

    client = TestClient(create_app())
    response = client.get("/api/live-pipeline/task-evaluation-launch-profiles")
    health = client.get("/health").json()["task_evaluation_launch_queue"]

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "task_evaluation_launch_profile_catalog.v1",
        "status": "published",
        "profiles": [_public_task_evaluation_launch_profile()],
        "allocator_arguments_exposed": False,
        "secret_values_exposed": False,
    }
    assert str(catalog_path) not in response.text
    assert health["public_catalog_configured"] is True
    assert health["public_catalog_ready"] is True


def test_public_task_evaluation_profile_catalog_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    unsafe = _public_task_evaluation_launch_profile()
    unsafe["allocator"] = {"argv": ["--execute"]}
    _write_json(catalog_path, [unsafe])
    monkeypatch.setenv(
        service.TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH_ENV,
        str(catalog_path),
    )

    response = TestClient(create_app()).get(
        "/api/live-pipeline/task-evaluation-launch-profiles"
    )

    assert response.status_code == 503
    assert response.json()["profiles"] == []
    assert response.json()["blockers"] == [
        "task_evaluation_launch_public_catalog_invalid"
    ]
    assert str(catalog_path) not in response.text


def test_task_evaluation_launch_systemd_path_mode_never_shells_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV, str(profile_dir))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_ALLOW_TRIGGER_ENV, "true")
    monkeypatch.delenv(service.TASK_EVALUATION_LAUNCH_EXECUTE_ENV, raising=False)
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_TRIGGER_MODE_ENV, "systemd_path")
    monkeypatch.delenv(service.TASK_EVALUATION_LAUNCH_TRIGGER_SYSTEMD_UNIT_ENV, raising=False)
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("systemd path mode must not call systemctl"),
    )

    result = service._trigger_task_evaluation_launch_dispatcher()

    assert result == {
        "status": "armed_by_systemd_path",
        "performed": True,
        "allowed": True,
        "trigger_mode": "systemd_path",
        "provider_mutation_performed": False,
    }


def test_signed_client_scope_cannot_select_another_tenant_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_a = _capture_root(tmp_path / "tenant-a")
    root_b = _capture_root(tmp_path / "tenant-b")
    manifest_path = _control_manifest(tmp_path, root_a)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(
        service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"client-a": "secret-a", "client-b": "secret-b"}),
    )
    monkeypatch.setenv(
        service.INTAKE_CLIENT_ROOTS_ENV,
        json.dumps(
            {
                "client-a": {"site-a": str(root_a)},
                "client-b": {"site-b": str(root_b)},
            }
        ),
    )
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    payload = _webapp_request(root_b)
    payload["job_request"]["site_package"]["site_slug"] = "site-b"  # type: ignore[index]
    body = json.dumps(payload, separators=(",", ":"))

    response = TestClient(create_app()).post(
        "/api/live-pipeline/job-requests",
        data=body,
        headers=_signed_intake_headers(
            "secret-a",
            body,
            nonce="tenant-scope-nonce-1",
            client_id="client-a",
        ),
    )

    assert response.status_code == 403
    assert "outside authenticated client scope" in response.text
    assert not (root_b / "pipeline" / "robot_eval_job_requests").exists()


def test_intake_admission_enforces_body_rate_concurrency_and_storage_quotas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "missing-manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    headers = {"authorization": "Bearer test-intake-token"}
    client = TestClient(create_app())

    monkeypatch.setenv(service.INTAKE_MAX_BODY_BYTES_ENV, "8")
    oversized = client.post(
        "/api/live-pipeline/job-requests",
        data='{"payload":"too-large"}',
        headers={**headers, "content-type": "application/json"},
    )
    assert oversized.status_code == 413

    monkeypatch.setenv(service.INTAKE_MAX_BODY_BYTES_ENV, "4096")
    monkeypatch.setenv(service.INTAKE_RATE_LIMIT_PER_MINUTE_ENV, "1")
    first = client.post("/api/live-pipeline/job-requests", json={}, headers=headers)
    limited = client.post("/api/live-pipeline/job-requests", json={}, headers=headers)
    assert first.status_code == 503
    assert limited.status_code == 429

    state_path, _lock_path = service._admission_state_paths()
    state_path.unlink(missing_ok=True)
    monkeypatch.setenv(service.INTAKE_RATE_LIMIT_PER_MINUTE_ENV, "100")
    monkeypatch.setenv(service.INTAKE_MAX_CONCURRENT_ENV, "1")
    lease_id = service._claim_intake_admission("manual-client")
    try:
        concurrent = client.post("/api/live-pipeline/job-requests", json={}, headers=headers)
    finally:
        service._release_intake_admission(lease_id)
    assert concurrent.status_code == 503
    assert "concurrency quota" in concurrent.text

    work_dir = service._work_dir(manifest_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "quota.bin").write_bytes(b"quota")
    monkeypatch.setenv(service.INTAKE_MAX_STORAGE_BYTES_ENV, "1")
    storage = client.post("/api/live-pipeline/job-requests", json={}, headers=headers)
    assert storage.status_code == 503
    assert "storage quota" in storage.text


def test_live_pipeline_intake_service_rejects_stale_signed_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "missing-manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    client = TestClient(create_app())
    body = "{}"

    response = client.post(
        "/api/live-pipeline/job-requests",
        data=body,
        headers=_signed_intake_headers(
            "test-intake-token",
            body,
            timestamp="2000-01-01T00:00:00+00:00",
            nonce="nonce-valid-3",
        ),
    )

    assert response.status_code == 401
    assert "replay window" in response.text


def test_live_pipeline_intake_service_rejects_tampered_signed_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "missing-manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    client = TestClient(create_app())
    signed_body = "{}"

    response = client.post(
        "/api/live-pipeline/job-requests",
        data='{"tampered":true}',
        headers=_signed_intake_headers(
            "test-intake-token",
            signed_body,
            nonce="nonce-valid-4",
        ),
    )

    assert response.status_code == 401
    assert "invalid intake signature" in response.text


def test_live_pipeline_intake_service_stages_webapp_request(tmp_path: Path, monkeypatch) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/job-requests",
        json=_webapp_request(capture_root),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["webapp_job_request"]["missing_fields"] == []
    assert payload["webapp_staging"]["performed"] is True
    assert Path(payload["webapp_staging"]["target_path"]).is_file()
    assert payload["trigger"]["status"] == "not_configured"
    assert payload["proof_boundary"]["intake_sets_proof_booleans"] is False


def test_live_pipeline_intake_service_translates_and_idempotently_retries_decision_request(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_OVERWRITE", "true")
    client = TestClient(create_app())

    responses = [
        client.post(
            "/api/live-pipeline/job-requests",
            json=_decision_evidence_envelope(),
            headers={"authorization": "Bearer test-intake-token"},
        )
        for _ in range(2)
    ]

    assert [response.status_code for response in responses] == [200, 200]
    payload = responses[-1].json()
    assert payload["status"] == "staged_for_control_plane"
    assert (
        responses[0].json()["webapp_staging"]["target_path"]
        == payload["webapp_staging"]["target_path"]
    )
    staged_path = Path(payload["webapp_staging"]["target_path"])
    staged = json.loads(staged_path.read_text(encoding="utf-8"))
    request = staged["job_request"]
    assert request["job_id"] == "decision-request-1"
    assert request["site_package"]["capture_root"] == str(capture_root)
    assert request["decision_evidence_request"]["decision_id"] == "buyer-decision-1"
    assert request["proof_boundary"]["translation_proves_decision"] is False


def test_live_pipeline_intake_service_converts_capture_handoff_to_webapp_request(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_robot_eval_dataset_cards(capture_root)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_OVERWRITE", "true")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/capture-handoffs",
        json=_capture_handoff(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["capture_handoff"]["converted_to_job_request"] is True
    assert payload["capture_handoff"]["dataset_selection"]["task_id"] == "scene_anchor_geometry_0"
    target_path = Path(payload["webapp_staging"]["target_path"])
    assert target_path.is_file()
    envelope = json.loads(target_path.read_text(encoding="utf-8"))
    job_request = envelope["job_request"]
    assert envelope["source_kind"] == "capture_pipeline_handoff"
    assert job_request["source_kind"] == "capture_pipeline_handoff"
    assert job_request["robot_profile"] == {"robot_profile_id": "unitree_g1"}
    assert job_request["requested_tasks"] == [
        {
            "task_id": "scene_anchor_geometry_0",
            "scenario_ids": ["scenario_scene_anchor_geometry_0_unitree_g1"],
        }
    ]
    assert job_request["source"]["pipeline_handoff_uri"].endswith("pipeline_handoff.json")
    assert job_request["policy_package"]["high_level_skill_trace"]["ordered_skill_sequence"] == [
        "walk_to_target"
    ]


def test_capture_handoff_dataset_selection_defaults_unspecified_robot_to_franka(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    dataset_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(dataset_dir / "task_cards.json", {"cards": [{"task_id": "pick_cup"}]})
    _write_json(
        dataset_dir / "scenario_cards.json",
        {"cards": [{"task_id": "pick_cup", "scenario_id": "pick_cup_counter"}]},
    )

    selection, blockers = service._select_dataset_task(capture_root)

    assert blockers == []
    assert selection is not None
    assert selection["robot_profile_id"] == "franka_panda"


def test_capture_handoff_can_stage_per_request_capture_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_capture_root = _capture_root(tmp_path)
    request_capture_root = (
        tmp_path / "storage" / "bucket" / "scenes" / "scene-2" / "captures" / "capture-2"
    )
    _write_json(
        request_capture_root / "capture_descriptor.json",
        {"scene_id": "scene-2", "capture_id": "capture-2"},
    )
    _write_json(request_capture_root / "raw" / "manifest.json", {"scene_id": "scene-2"})
    _seed_robot_eval_dataset_cards(request_capture_root)
    manifest_path = _control_manifest(tmp_path, manifest_capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_OVERWRITE", "true")
    monkeypatch.setenv(
        service.INTAKE_CLIENT_ROOTS_ENV,
        json.dumps({"legacy-bearer": {"capture-2": str(request_capture_root)}}),
    )
    client = TestClient(create_app())
    handoff = {
        **_capture_handoff(),
        "scene_id": "scene-2",
        "capture_id": "capture-2",
        "capture_root": str(request_capture_root),
    }

    response = client.post(
        "/api/live-pipeline/capture-handoffs",
        json=handoff,
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    target_path = Path(payload["webapp_staging"]["target_path"])
    staged = json.loads(target_path.read_text(encoding="utf-8"))
    assert staged["job_request"]["site_package"]["capture_root"] == str(request_capture_root)
    assert staged["job_request"]["source"]["selection_state"]["scene_id"] == "scene-2"


def test_live_pipeline_intake_service_blocks_capture_handoff_without_robot_eval_request(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_robot_eval_dataset_cards(capture_root)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())
    handoff = {
        **_capture_handoff(),
        "requested_outputs": ["qualification"],
        "requested_lanes": ["qualification"],
        "robot_eval_dataset_requested": False,
    }

    response = client.post(
        "/api/live-pipeline/capture-handoffs",
        json=handoff,
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "capture_handoff:capture_handoff_robot_eval_not_requested" in payload["input_blockers"]


def test_live_pipeline_intake_service_ignores_caller_root_and_uses_server_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    other_capture_root = (
        tmp_path / "storage" / "bucket" / "scenes" / "scene-2" / "captures" / "capture-2"
    )
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/job-requests",
        json=_webapp_request(other_capture_root),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["webapp_job_request"]["capture_root_matches_control_plane"] is True
    assert payload["webapp_staging"]["performed"] is True
    candidate = json.loads(Path(payload["candidate"]["path"]).read_text(encoding="utf-8"))
    site_package = candidate["job_request"]["site_package"]
    assert site_package["capture_root"] == str(capture_root)
    assert site_package["capture_root_source"] == "authenticated_server_mapping"
    assert (
        candidate["job_request"]["authenticated_client_scope"]["caller_capture_root_authoritative"]
        is False
    )


def test_live_pipeline_intake_service_exposes_latest_audit(tmp_path: Path, monkeypatch) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())
    client.post(
        "/api/live-pipeline/job-requests",
        json=_webapp_request(capture_root),
        headers={"x-blueprint-intake-token": "test-intake-token"},
    )

    response = client.get(
        "/api/live-pipeline/intake-audit",
        headers={"x-blueprint-intake-token": "test-intake-token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["webapp_staging"]["performed"] is True


def test_live_pipeline_intake_service_stages_deployment_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/deployment-outcomes",
        json=_deployment_outcomes(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    target_path = Path(payload["deployment_outcomes_staging"]["target_path"])
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["deployment_outcomes"]["status"] == "ready_for_real_world_validation"
    assert payload["deployment_outcomes"]["record_count"] == 1
    assert payload["deployment_outcomes"]["owner_evidence_ready"] is True
    assert payload["deployment_outcomes"]["owner_evidence_record_count"] == 1
    assert payload["deployment_outcomes"]["missing_owner_evidence_record_ids"] == []
    assert payload["deployment_outcomes_staging"]["performed"] is True
    assert target_path == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "deployment_outcomes"
        / "inbox"
        / "pilot-outcome-1.json"
    )
    assert target_path.is_file()
    assert payload["proof_boundary"]["real_world_outcome_proven"] is False


def test_live_pipeline_intake_service_accepts_outcome_records_without_owner_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/deployment-outcomes",
        json=_deployment_outcomes(include_evidence=False),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["deployment_outcomes"]["status"] == "ready_for_real_world_validation"
    assert payload["deployment_outcomes"]["record_count"] == 1
    assert payload["deployment_outcomes"]["owner_evidence_ready"] is False
    assert payload["deployment_outcomes"]["owner_evidence_record_count"] == 0
    assert payload["deployment_outcomes"]["missing_owner_evidence_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert payload["deployment_outcomes_staging"]["performed"] is True


def test_live_pipeline_intake_service_stages_policy_package(tmp_path: Path, monkeypatch) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/policy-packages",
        json=_policy_package(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    target_path = Path(payload["policy_package_staging"]["target_path"])
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["policy_package"]["status"] == "ready_for_robot_eval_job"
    assert payload["policy_package"]["selected_modalities"] == ["policy_api_endpoint"]
    assert payload["policy_package_staging"]["performed"] is True
    assert target_path == (
        capture_root / "pipeline" / "robot_eval_inputs" / "webapp-job-1" / "policy_package.json"
    )
    assert target_path.is_file()
    assert payload["proof_boundary"]["robot_policy_execution_proven"] is False


def test_live_pipeline_intake_service_records_blocked_policy_package(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/policy-packages",
        json={
            "schema_version": "robot_team_policy_package.v1",
            "job_id": "../escape",
            "policy_package": {"docker_container": {"image_ref": "registry.example/policy:latest"}},
        },
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "policy_package:policy_package_job_id_unsafe" in payload["input_blockers"]
    assert "policy_package:policy_package.docker_container.digest" in payload["input_blockers"]
    assert payload["policy_package_staging"]["performed"] is False


def test_live_pipeline_intake_service_stages_real_robot_pov_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/real-robot-pov",
        json=_real_robot_pov_manifest(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    target_path = Path(payload["real_robot_pov_staging"]["target_path"])
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["real_robot_pov"]["status"] == "ready_for_robot_eval_job"
    assert payload["real_robot_pov"]["record_count"] == 1
    assert payload["real_robot_pov"]["missing_exact_key_record_ids"] == []
    assert payload["real_robot_pov"]["missing_evidence_record_ids"] == []
    assert payload["real_robot_pov_staging"]["performed"] is True
    assert target_path == (
        capture_root / "pipeline" / "robot_eval_inputs" / "real_robot_pov_manifest.json"
    )
    assert target_path.is_file()
    assert payload["proof_boundary"]["robot_pov_evidence_proven"] is False


def test_live_pipeline_intake_service_records_blocked_real_robot_pov_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/real-robot-pov",
        json={
            "schema_version": "real_robot_pov_manifest.v1",
            "job_id": "../escape",
            "records": [
                {
                    "evidence_id": "real-pov-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "robot_camera_video_uri": "owner://pov/scenario-run-1.mp4",
                }
            ],
        },
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "real_robot_pov:real_robot_pov_job_id_unsafe" in payload["input_blockers"]
    assert "real_robot_pov:real_robot_pov_missing_exact_keys" in payload["input_blockers"]
    assert "real_robot_pov:real_robot_pov_missing_action_logs" in payload["input_blockers"]
    assert payload["real_robot_pov_staging"]["performed"] is False


def test_live_pipeline_intake_service_records_blocked_deployment_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/deployment-outcomes",
        json={
            "schema_version": "deployment_outcome_manifest.v1",
            "job_id": "../escape",
            "records": [{"task_id": "task-only"}],
        },
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "deployment_outcomes:deployment_outcomes_job_id_unsafe" in payload["input_blockers"]
    assert payload["deployment_outcomes_staging"]["performed"] is False


def test_live_pipeline_intake_service_stages_live_closure_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/live-closure-evidence",
        json=_live_closure_evidence(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    target_path = Path(payload["live_closure_evidence_staging"]["target_path"])
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["live_closure_evidence"]["status"] == "ready_for_closure_audit"
    assert payload["live_closure_evidence_staging"]["performed"] is True
    assert target_path == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "live_eval_closure_evidence.json"
    )
    assert target_path.is_file()
    assert payload["proof_boundary"]["intake_sets_proof_booleans"] is False


def test_live_pipeline_intake_service_records_blocked_closure_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/live-closure-evidence",
        json={
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "job_id": "../escape",
        },
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "live_closure_evidence:live_closure_evidence_job_id_unsafe" in payload["input_blockers"]
    assert payload["live_closure_evidence_staging"]["performed"] is False


def _release_worktree(tmp_path, commit: str):
    """A release worktree exactly as the activation script leaves it: a `.git`
    file pointing at a gitdir whose HEAD holds the detached commit."""

    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    (gitdir / "HEAD").write_text(commit + "\n", encoding="utf-8")
    checkout = tmp_path / "release"
    (checkout / "src" / "blueprint_pipeline").mkdir(parents=True)
    (checkout / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    return checkout / "src" / "blueprint_pipeline" / "live_pipeline_intake_service.py"


def test_deployment_identity_reads_the_commit_the_code_came_from(tmp_path, monkeypatch):
    """Asking the running checkout what it is, instead of asking a variable
    what someone last declared it to be."""

    from blueprint_pipeline import live_pipeline_intake_service as intake

    monkeypatch.delenv(intake.PIPELINE_SOURCE_COMMIT_ENV, raising=False)
    module = _release_worktree(tmp_path, "a" * 40)

    payload = intake.deployment_identity_payload(module)

    assert payload["commit_proven"] is True
    assert payload["source_commit"] == "a" * 40
    assert payload["source_commit_source"] == "running_checkout"
    assert payload["blockers"] == []


def test_a_declared_commit_that_contradicts_the_running_code_proves_nothing(
    tmp_path, monkeypatch
):
    """Activate release B without updating the variable and the endpoint used
    to report A while the service ran B."""

    from blueprint_pipeline import live_pipeline_intake_service as intake

    monkeypatch.setenv(intake.PIPELINE_SOURCE_COMMIT_ENV, "b" * 40)
    module = _release_worktree(tmp_path, "a" * 40)

    payload = intake.deployment_identity_payload(module)

    assert payload["commit_proven"] is False
    assert payload["source_commit"] is None
    assert payload["source_commit_source"] == "conflicting"
    assert payload["blockers"] == [
        "deployment_identity_declared_commit_conflicts_with_running_checkout"
    ]


def test_the_environment_still_answers_where_there_is_no_checkout(tmp_path, monkeypatch):
    from blueprint_pipeline import live_pipeline_intake_service as intake

    monkeypatch.setenv(intake.PIPELINE_SOURCE_COMMIT_ENV, "c" * 40)
    detached = tmp_path / "no-checkout" / "module.py"
    detached.parent.mkdir(parents=True)

    payload = intake.deployment_identity_payload(detached)

    assert payload["commit_proven"] is True
    assert payload["source_commit"] == "c" * 40
    assert payload["source_commit_source"] == "environment"


def test_neither_source_available_fails_closed(tmp_path, monkeypatch):
    from blueprint_pipeline import live_pipeline_intake_service as intake

    monkeypatch.delenv(intake.PIPELINE_SOURCE_COMMIT_ENV, raising=False)
    detached = tmp_path / "no-checkout" / "module.py"
    detached.parent.mkdir(parents=True)

    payload = intake.deployment_identity_payload(detached)

    assert payload["commit_proven"] is False
    assert payload["blockers"] == ["deployment_identity_source_commit_unavailable"]


def test_a_branch_checkout_is_not_answered_for(tmp_path, monkeypatch):
    """A HEAD holding a symbolic ref is a development checkout that can move
    under the running process; a release worktree is detached at its commit."""

    from blueprint_pipeline import live_pipeline_intake_service as intake

    monkeypatch.delenv(intake.PIPELINE_SOURCE_COMMIT_ENV, raising=False)
    gitdir = tmp_path / "repo" / ".git"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    module = tmp_path / "repo" / "src" / "module.py"
    module.parent.mkdir(parents=True)

    assert intake.running_source_commit(module) == ""
    assert intake.deployment_identity_payload(module)["commit_proven"] is False


def test_the_endpoint_reports_a_detached_checkout_without_any_variable(
    tmp_path, monkeypatch
):
    """A release worktree knows its own commit, so no deploy-time env edit is
    needed for the endpoint to answer."""

    from blueprint_pipeline import live_pipeline_intake_service as intake

    monkeypatch.delenv(intake.PIPELINE_SOURCE_COMMIT_ENV, raising=False)
    monkeypatch.setattr(intake, "running_source_commit", lambda *_a, **_k: "d" * 40)
    client = TestClient(create_app())

    response = client.get("/api/live-pipeline/version")

    assert response.status_code == 200
    assert response.json()["source_commit"] == "d" * 40
    assert response.json()["source_commit_source"] == "running_checkout"
