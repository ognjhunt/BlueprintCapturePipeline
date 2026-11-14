from __future__ import annotations

import hmac
import json
import os
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


def test_nonce_store_reuses_peer_owned_exact_directory_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "shared-nonce-store"
    root.mkdir(mode=0o700)
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(root))
    chmod_calls: list[Path] = []
    original_chmod = Path.chmod

    def refuse_root_chmod(path: Path, mode: int, *args, **kwargs) -> None:
        if path.resolve() == root.resolve():
            chmod_calls.append(path)
            raise PermissionError("peer-owned exact directory must not be mutated")
        original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", refuse_root_chmod)

    assert service._nonce_store_dir() == root.resolve()
    assert chmod_calls == []


def test_nonce_store_refuses_when_secure_mode_cannot_be_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "shared-nonce-store"
    root.mkdir(mode=0o755)
    root.chmod(0o755)  # A restrictive runner umask must not skip this refusal path.
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(root))
    monkeypatch.setattr(
        Path,
        "chmod",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(RuntimeError, match="^intake_nonce_store_permission_install_failed$"):
        service._nonce_store_dir()


@pytest.mark.parametrize("route", ["/website-capture-withdrawals", "/api/live-pipeline/website-capture-withdrawals"])
def test_signed_website_withdrawal_uses_server_scope_and_never_deletes(tmp_path, monkeypatch, route):
    from blueprint_pipeline.consent_takedown import read_consent_state
    root = tmp_path / "scenes/site-req1/captures/walkthrough-req1"
    _write_json(root / "raw/manifest.json", {"site_submission_id": "req1", "scene_id": "site-req1",
        "capture_id": "walkthrough-req1", "consent_status": "granted"})
    (root / "raw/video.mp4").write_bytes(b"synthetic-media")
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "synthetic-token")
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "work"))
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_CLIENT_ROOTS_JSON", json.dumps({"test-client": {"req1": str(root)}}))
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "false")
    client = TestClient(create_app())
    vector = json.loads((Path(__file__).parent / "fixtures/website-withdrawal-vector.json").read_text())
    payload = vector["command"]
    assert vector["expected_path"] == "/api/live-pipeline/website-capture-withdrawals"
    def post(value, nonce, actor="test-client"):
        body = json.dumps(value)
        return client.post(route, content=body,
            headers=_signed_intake_headers("synthetic-token", body, nonce=nonce, client_id=actor))
    assert client.post(route, json=payload).status_code == 401
    assert post(payload, "foreign-actor", "other-client").status_code == 403
    assert post({**payload, "request_id": "other"}, "foreign-site").status_code == 403
    assert post({**payload, "capture_root": str(tmp_path)}, "caller-root").status_code == 409
    accepted = post(payload, "ack-first")
    assert accepted.status_code == 200
    assert accepted.json()["command_digest"] == vector["command_digest"]
    assert accepted.json()["deletion_confirmed"] is False
    assert post(payload, "ack-replay").json() == accepted.json()
    assert read_consent_state(root)["state"] == "revoked"
    assert (root / "raw/video.mp4").read_bytes() == b"synthetic-media"


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


def test_live_pipeline_intake_service_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "control" / "manifest.json"
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "custom-work"))
    assert service._work_dir(manifest_path) == tmp_path / "custom-work"
    assert service._first_string("", None) == ""
    assert service._list_from_payload(("a", "b")) == ["a", "b"]
    assert service._list_from_payload("bad") == []


def test_capture_handoff_blocker_edges(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    audit = service._capture_handoff_preparation_audit(
        payload={"scene_id": "other-scene", "capture_id": "other-capture"},
        capture_root=capture_root,
    )
    assert "capture_handoff_scene_id_mismatch" in audit["blockers"]
    assert "capture_handoff_capture_id_mismatch" in audit["blockers"]
    assert "capture_handoff_missing_site_submission_id" in audit["blockers"]
    assert audit["converted_to_job_request"] is False


def test_deployment_identity_fails_closed_without_exact_source_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app())
    monkeypatch.setattr(service, "media_tool_blockers", lambda: [])

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
        "disk_headroom": {
            "schema_version": "control_plane_disk_headroom.v1",
            "status": "unconfigured",
            "refused_roles": [],
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


def test_configured_disk_headroom_passes_role_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "default"
    scratch = tmp_path / "scratch"
    monkeypatch.setenv(service.CONTROL_PLANE_DISK_TARGET_ROOT_ENV, str(target))
    monkeypatch.setenv(service.CONTROL_PLANE_DISK_RESERVATION_ROOT_ENV, str(tmp_path / "ledger"))
    monkeypatch.setenv(
        "BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS",
        f"handoff_staging={scratch},launch_dispatch={scratch}",
    )
    seen: dict[str, object] = {}

    def fake_headroom(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "ok", "refused_roles": []}

    monkeypatch.setattr(service, "disk_headroom", fake_headroom)
    assert service._configured_disk_headroom()["status"] == "ok"
    assert seen["role_targets"] == {
        "handoff_staging": scratch,
        "launch_dispatch": scratch,
    }
    assert seen["lock_nonblocking"] is True


def test_deployment_identity_reports_source_while_busy_headroom_refuses_all_roles(monkeypatch, tmp_path):
    import fcntl
    from blueprint_pipeline import control_plane_disk_budget as disk_budget
    ledger = disk_budget._prepare_ledger_root(tmp_path / "ledger")
    descriptor = disk_budget.open_ledger_lock(ledger, require_mode=True)
    monkeypatch.setenv(service.CONTROL_PLANE_DISK_TARGET_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(service.CONTROL_PLANE_DISK_RESERVATION_ROOT_ENV, str(ledger))
    monkeypatch.setenv(service.PIPELINE_SOURCE_COMMIT_ENV, "a" * 40)
    monkeypatch.setattr(service, "running_source_commit", lambda *_args: "a" * 40)
    monkeypatch.delenv("BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS", raising=False)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        response = TestClient(create_app()).get("/api/live-pipeline/version")
        assert response.status_code == 200
        payload = response.json()
        assert payload["source_commit"] == "a" * 40 and payload["commit_proven"] is True
        assert payload["disk_headroom"]["status"] == "unknown_fail_closed"
        assert payload["disk_headroom"]["error_code"] == "control_plane_disk_budget_lock_busy"
        assert payload["disk_headroom"]["refused_roles"] == list(service.CHAIN_ROLES)
        for role in service.CHAIN_ROLES:
            assert service._disk_role_refused(payload, role) is True
    finally:
        os.close(descriptor)


def test_configured_disk_headroom_fails_closed_for_invalid_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(service.CONTROL_PLANE_DISK_TARGET_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(service.CONTROL_PLANE_DISK_RESERVATION_ROOT_ENV, str(tmp_path / "ledger"))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS", "launch_dispatch=relative")
    result = service._configured_disk_headroom()
    assert result["status"] == "unknown_fail_closed"
    assert "policy_canary_dispatch" in result["refused_roles"]


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
            "/api/live-pipeline/capture-upload-intakes", json={}, headers={"x-blueprint-intake-token": "bad"}
        ).status_code
        == 401
    )

    endpoints = ["/api/live-pipeline/capture-upload-intakes"]
    for endpoint in endpoints:
        assert (
            client.post(
                endpoint, data="{", headers={**headers, "content-type": "application/json"}
            ).status_code
            == 400
        )
        assert client.post(endpoint, json=[], headers=headers).status_code == 400
        assert client.post(endpoint, json={}, headers=headers).status_code == 503


@pytest.mark.parametrize('constraints, expected_live', [
    ({}, True), ({'automatic_model_inference_allowed': False}, False),
    ({'external_processing_allowed': False}, False),
])
def test_live_pipeline_capture_upload_intake_is_authenticated_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constraints, expected_live
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
        "request": {"governance": {"provider_constraints": constraints}},
        "transfer": {
            "url": "https://download.example.test/file/capture.mp4",
            "authorization": "ephemeral-secret",
        },
    }
    client = TestClient(create_app())
    supervisor_health = client.get("/health").json()["task_evaluation_supervisor"]
    assert client.get('/api/live-pipeline/health').json()['task_evaluation_supervisor'] == supervisor_health
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
        "allow_live_agents_sdk": expected_live,
        "agent_inference_budget_usd": 0.5 if expected_live else 0.0,
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


def test_live_pipeline_intake_service_requires_token(tmp_path: Path, monkeypatch) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.delenv(INTAKE_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.post("/api/live-pipeline/capture-upload-intakes", json={})

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
        "/api/live-pipeline/capture-upload-intakes",
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

    first = client.post("/api/live-pipeline/capture-upload-intakes", data=body, headers=headers)
    replay = client.post("/api/live-pipeline/capture-upload-intakes", data=body, headers=headers)

    assert first.status_code == 503
    assert replay.status_code == 401
    assert "replayed intake signature nonce" in replay.text


def test_capture_upload_health_accepts_host_authenticated_storage_without_b2_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(service.CAPTURE_UPLOAD_STORE_ROOT_ENV, str(tmp_path / "capture-store"))
    monkeypatch.setenv(service.CAPTURE_MALWARE_SCANNER_ARGV_ENV, '["/usr/bin/clamdscan"]')
    monkeypatch.delenv(service.CAPTURE_UPLOAD_ALLOWED_HOSTS_ENV, raising=False)
    response = TestClient(create_app()).get("/api/live-pipeline/health")
    assert response.status_code == 200
    assert response.json()["capture_upload_transfer_configured"] is True
    assert response.json()["capture_upload_backblaze_hosts_configured"] is False


def test_configured_scene_thumbnail_readback_returns_exact_private_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "incoming"))
    expected = b"exact-digest-bound-thumbnail"
    reference = {
        "uri": "s3://blueprint-production-inputs/blueprint/arm-decision-proof-v1/configured-scenes/team/thumbnail/sha256/"
        + "a" * 64
        + "/task-thumbnail.png",
        "digest": "sha256:" + "a" * 64,
        "size_bytes": len(expected),
    }
    observed: dict[str, object] = {}

    def readback(*, reference: dict[str, object], maximum_size_bytes: int) -> bytes:
        observed.update(reference)
        assert maximum_size_bytes == 16 * 1024 * 1024
        return expected

    monkeypatch.setattr(service, "read_configured_scene_object", readback)
    body = json.dumps(reference, separators=(",", ":"))
    response = TestClient(create_app()).post(
        "/api/live-pipeline/task-evaluation-configured-scene-artifact-readback",
        content=body,
        headers=_signed_intake_headers(
            "test-intake-token",
            body,
            nonce="configured-scene-thumbnail-readback",
            client_id="blueprint-webapp",
        ),
    )

    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-blueprint-artifact-sha256"] == reference["digest"]
    assert observed == reference


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
        "/api/live-pipeline/capture-upload-intakes", data=body, headers=headers
    )
    service._INTAKE_NONCE_CACHE.clear()
    replay_after_restart = TestClient(create_app()).post(
        "/api/live-pipeline/capture-upload-intakes", data=body, headers=headers
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
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    systemctl_calls: list[list[str]] = []
    binding_checks: list[dict[str, object]] = []

    def fake_webapp_record_binding(*, progress):
        binding_checks.append(dict(progress))
        return {
            "status": "succeeded",
            "launch_id": progress["launch_id"],
            "run_id": progress["run_id"],
            "request_digest": progress["request_digest"],
            "response": {
                "schema_version": (
                    "task_evaluation_launch_progress_web_sync_receipt.v1"
                ),
                "status": "recorded",
                "launch_id": progress["launch_id"],
                "run_id": progress["run_id"],
                "request_digest": progress["request_digest"],
                "phase": progress["phase"],
            },
        }

    def fake_run(argv, **_kwargs):
        systemctl_calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(
        service,
        "sync_launch_progress_to_webapp",
        fake_webapp_record_binding,
    )
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
    assert response.json()["webapp_record_bound"] is True
    assert response.json()["website_trigger_proven_at_intake"] is True
    assert [item["phase"] for item in binding_checks] == [
        "intake_webapp_record_binding"
    ]
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


def test_task_evaluation_launch_refuses_missing_webapp_record_before_queue_or_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_root = tmp_path / "launch-queue"
    catalog_path = tmp_path / "published-launch-profiles.json"
    _write_json(catalog_path, [_public_task_evaluation_launch_profile()])
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(tmp_path / "control.json"))
    monkeypatch.setenv(
        service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"blueprint-webapp": "token"}),
    )
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_QUEUE_ROOT_ENV, str(queue_root))
    monkeypatch.setenv(
        service.TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH_ENV,
        str(catalog_path),
    )
    systemctl_calls: list[list[str]] = []
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda argv, **_kwargs: systemctl_calls.append(list(argv)),
    )
    monkeypatch.setattr(
        service,
        "sync_launch_progress_to_webapp",
        lambda **_kwargs: {"status": "failed", "reason": "http_error:404"},
    )
    payload = _task_evaluation_launch_request()
    body = json.dumps(payload, separators=(",", ":"))

    response = TestClient(create_app()).post(
        "/api/live-pipeline/task-evaluation-launches",
        data=body,
        headers=_signed_intake_headers(
            "token",
            body,
            nonce="task-launch-missing-webapp-record",
            client_id="blueprint-webapp",
        ),
    )

    assert response.status_code == 409
    assert response.json()["blockers"] == ["webapp_launch_record_missing"]
    assert response.json()["accepted"] is False
    assert response.json()["webapp_record_bound"] is False
    assert response.json()["website_trigger_proven"] is False
    assert not list(queue_root.rglob("*.json"))
    assert systemctl_calls == []


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


def test_task_evaluation_launch_systemd_path_mode_checks_read_only_active_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV, str(profile_dir))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_ALLOW_TRIGGER_ENV, "true")
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_TRIGGER_MODE_ENV, "systemd_path")
    monkeypatch.delenv(service.TASK_EVALUATION_LAUNCH_TRIGGER_SYSTEMD_UNIT_ENV, raising=False)
    calls: list[list[str]] = []

    def completed(argv, **_kwargs):
        calls.append(list(argv))
        stdout = "enabled\n" if argv[1] == "is-enabled" else "active\n"
        return service.subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(service.subprocess, "run", completed)

    result = service._trigger_task_evaluation_launch_dispatcher()

    assert result == {
        "status": "armed_by_systemd_path",
        "performed": True,
        "allowed": True,
        "trigger_mode": "systemd_path",
        "systemd_path_unit": "blueprint-task-evaluation-launch-dispatcher.path",
        "systemd_path_enabled_state": "enabled",
        "systemd_path_active_state": "active",
        "provider_mutation_performed": False,
    }
    assert calls == [
        [
            "systemctl",
            "is-enabled",
            "blueprint-task-evaluation-launch-dispatcher.path",
        ],
        [
            "systemctl",
            "is-active",
            "blueprint-task-evaluation-launch-dispatcher.path",
        ],
    ]


def test_task_evaluation_launch_systemd_path_mode_refuses_inactive_watcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV, str(profile_dir))
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_ALLOW_TRIGGER_ENV, "true")
    monkeypatch.setenv(service.TASK_EVALUATION_LAUNCH_TRIGGER_MODE_ENV, "systemd_path")

    def completed(argv, **_kwargs):
        stdout = "enabled\n" if argv[1] == "is-enabled" else "inactive\n"
        return service.subprocess.CompletedProcess(argv, 3, stdout=stdout, stderr="")

    monkeypatch.setattr(service.subprocess, "run", completed)

    result = service._trigger_task_evaluation_launch_dispatcher()

    assert result == {
        "status": "blocked",
        "performed": False,
        "allowed": True,
        "trigger_mode": "systemd_path",
        "systemd_path_unit": "blueprint-task-evaluation-launch-dispatcher.path",
        "systemd_path_enabled_state": "enabled",
        "systemd_path_active_state": "inactive",
        "blockers": ["task_evaluation_launch_systemd_path_inactive"],
        "provider_mutation_performed": False,
    }


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
        "/api/live-pipeline/capture-upload-intakes",
        data='{"payload":"too-large"}',
        headers={**headers, "content-type": "application/json"},
    )
    assert oversized.status_code == 413

    monkeypatch.setenv(service.INTAKE_MAX_BODY_BYTES_ENV, "4096")
    monkeypatch.setenv(service.INTAKE_RATE_LIMIT_PER_MINUTE_ENV, "1")
    first = client.post("/api/live-pipeline/capture-upload-intakes", json={}, headers=headers)
    limited = client.post("/api/live-pipeline/capture-upload-intakes", json={}, headers=headers)
    assert first.status_code == 503
    assert limited.status_code == 429

    state_path, _lock_path = service._admission_state_paths()
    state_path.unlink(missing_ok=True)
    monkeypatch.setenv(service.INTAKE_RATE_LIMIT_PER_MINUTE_ENV, "100")
    monkeypatch.setenv(service.INTAKE_MAX_CONCURRENT_ENV, "1")
    lease_id = service._claim_intake_admission("manual-client")
    try:
        concurrent = client.post("/api/live-pipeline/capture-upload-intakes", json={}, headers=headers)
    finally:
        service._release_intake_admission(lease_id)
    assert concurrent.status_code == 503
    assert "concurrency quota" in concurrent.text

    work_dir = service._work_dir(manifest_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "quota.bin").write_bytes(b"quota")
    monkeypatch.setenv(service.INTAKE_MAX_STORAGE_BYTES_ENV, "1")
    storage = client.post("/api/live-pipeline/capture-upload-intakes", json={}, headers=headers)
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
        "/api/live-pipeline/capture-upload-intakes",
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
        "/api/live-pipeline/capture-upload-intakes",
        data='{"tampered":true}',
        headers=_signed_intake_headers(
            "test-intake-token",
            signed_body,
            nonce="nonce-valid-4",
        ),
    )

    assert response.status_code == 401
    assert "invalid intake signature" in response.text


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
    monkeypatch.setattr(intake, "media_tool_blockers", lambda: [])
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
    monkeypatch.setattr(intake, "media_tool_blockers", lambda: [])
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
    monkeypatch.setattr(intake, "media_tool_blockers", lambda: [])
    detached = tmp_path / "no-checkout" / "module.py"
    detached.parent.mkdir(parents=True)

    payload = intake.deployment_identity_payload(detached)

    assert payload["commit_proven"] is False
    assert payload["blockers"] == ["deployment_identity_source_commit_unavailable"]


def test_a_host_without_ffmpeg_names_it_and_ffprobe_alone_is_not_a_blocker(tmp_path, monkeypatch):
    """Sound removal fails closed per capture without ffmpeg; the deployed identity
    names the gap without refusing the commit proof. Without ffprobe removal still
    succeeds on the exact command, so that is not a blocker."""

    from blueprint_pipeline import capture_audio_minimization as audio
    from blueprint_pipeline import live_pipeline_intake_service as intake

    monkeypatch.delenv(intake.PIPELINE_SOURCE_COMMIT_ENV, raising=False)
    module = _release_worktree(tmp_path, "a" * 40)

    monkeypatch.setattr(audio.shutil, "which", lambda tool: "/usr/bin/ffprobe" if tool == "ffprobe" else None)
    payload = intake.deployment_identity_payload(module)
    assert payload["commit_proven"] is True
    assert payload["blockers"] == ["capture_audio_removal_tool_missing:ffmpeg"]

    monkeypatch.setattr(audio.shutil, "which", lambda tool: "/usr/bin/ffmpeg" if tool == "ffmpeg" else None)
    assert intake.deployment_identity_payload(module)["blockers"] == []


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
