from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.first_gpu_e2e_readiness import (
    FIRST_GPU_E2E_READINESS_SCHEMA_VERSION,
    LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
    build_first_gpu_e2e_readiness,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path, *, with_requested_outputs: bool = True) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    requested_outputs = (
        ["qualification", "robot_eval_dataset", "task_evaluation_run"]
        if with_requested_outputs
        else ["qualification"]
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
            "capture_capabilities": {"camera_pose": True},
            "requested_outputs": requested_outputs,
            "site_submission_id": "site-submission-1",
            "request_id": "request-1",
            "buyer_request_id": "buyer-request-1",
            "capture_job_id": "capture-job-1",
        },
    )
    _write_json(
        capture_root / "raw" / "capture_context.json",
        {"workflowName": "GPU proof smoke", "zone": "Zone A"},
    )
    _write_json(
        capture_root / "raw" / "intake_packet.json",
        {
            "workflowName": "GPU proof smoke",
            "taskSteps": ["load scene", "spawn robot", "run action trace"],
            "zone": "Zone A",
        },
    )
    _write_json(
        capture_root / "raw" / "capture_upload_complete.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix": "scenes/scene-1/captures/capture-1/raw",
        },
    )
    (capture_root / "raw" / "walkthrough.mov").write_bytes(b"fake-video")
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "requested_outputs": requested_outputs,
            "site_submission_id": "site-submission-1",
            "request_id": "request-1",
            "buyer_request_id": "buyer-request-1",
            "capture_job_id": "capture-job-1",
        },
    )
    return capture_root


def _write_gpu_handoff_artifacts(capture_root: Path) -> None:
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    _write_json(
        automation_dir / "gpu_handoff_packet.json",
        {
            "schema_version": "gpu_handoff_packet.v1",
            "status": "ready_for_owner_gpu_preflight_handoff",
            "ready_for_owner_gpu_preflight": True,
            "owner_gpu_simulator_execution_proven": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["owner_gpu_simulator_execution_not_run"],
        },
    )
    _write_json(
        automation_dir / "gpu_owner_system_proof_schema.json",
        {"schema_version": "gpu_owner_system_proof_schema.v1"},
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_blocked_manifest.v1",
            "status": "blocked",
            "blocker_id": "owner_gpu_simulator_execution_not_run",
        },
    )
    _write_json(
        automation_dir / "simulator_engine_plugin_registry.json",
        {"schema_version": "simulator_engine_plugin_registry.v1", "status": "ready"},
    )
    (automation_dir / "gpu_run_checklist.md").write_text("# GPU checklist\n", encoding="utf-8")


def _write_staged_webapp_request(capture_root: Path, *, local_rehearsal: bool = False) -> Path:
    request_path = capture_root / "pipeline" / "robot_eval_job_requests" / "webapp-job-1.json"
    request = {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": "webapp-job-1",
        "site_package": {
            "capture_root": str(capture_root.resolve()),
            "site_submission_id": "site-submission-1",
            "capture_job_id": "capture-job-1",
            "buyer_request_id": "buyer-request-1",
        },
        "owner_system": {
            "request_id": "request-1",
            "buyer_request_id": "buyer-request-1",
            "site_submission_id": "site-submission-1",
            "capture_job_id": "capture-job-1",
        },
    }
    if local_rehearsal:
        request["source_kind"] = LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    envelope = {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": "webapp-job-1",
        "job_request": request,
    }
    if local_rehearsal:
        envelope.update(
            {
                "source_kind": LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
                "local_rehearsal_only": True,
            },
        )
    _write_json(request_path, envelope)
    staged_path = capture_root / "pipeline" / "live_pipeline_staged_inputs.json"
    staged_payload = {
        "schema_version": "blueprint_live_pipeline_staged_inputs.v1",
        "configured_capture_root": str(capture_root.resolve()),
        "webapp_request": {
            "ready": True,
            "staged": True,
            "job_id": "webapp-job-1",
            "path": str(request_path),
            "target_path": str(request_path),
        },
    }
    if local_rehearsal:
        staged_payload["source_kind"] = LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
        staged_payload["local_rehearsal_only"] = True
        staged_payload["webapp_request"]["source_kind"] = LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    _write_json(staged_path, staged_payload)
    return staged_path


def _write_webapp_forwarding_preflight_report(
    path: Path,
    *,
    site_slug: str = "site-1",
    status: str = "ready_for_required_forwarding_with_probe",
    site_slugs: list[str] | None = None,
) -> Path:
    if site_slugs is None:
        site_slugs = [site_slug]
    _write_json(
        path,
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": status,
            "forwarding_required": True,
            "endpoint_configured": True,
            "configured_env": {
                "forward_url": {
                    "configured": True,
                    "valid": True,
                    "protocol": "https",
                    "origin": "https://pipeline.example",
                    "pathname": "/api/live-pipeline/job-requests",
                    "query_present": False,
                    "credentials_present": False,
                },
                "forward_token": {
                    "configured": True,
                    "redacted": True,
                },
                "forward_timeout_ms": {
                    "configured": True,
                    "value": 10000,
                    "valid": True,
                },
                "capture_root_by_site_json": {
                    "configured": True,
                    "valid": True,
                    "site_count": len(site_slugs),
                    "site_slugs": site_slugs,
                },
                "single_capture_root_override": {
                    "configured": False,
                },
            },
            "probe": {
                "requested": True,
                "attempted": True,
                "status": "reachable",
                "http_status": 200,
                "audit_status": "staged_for_control_plane",
            },
            "blockers": [],
            "warnings": [],
            "proof_boundary": {
                "command_is_read_only": True,
                "no_job_queued": True,
                "no_pipeline_mutation_requested": True,
                "no_gpu_allocated": True,
                "no_simulator_execution_proven": True,
                "no_robot_readiness_proven": True,
                "no_public_claim_upgrade_allowed": True,
            },
        },
    )
    return path


def test_first_gpu_readiness_blocks_missing_runtime_and_webapp_setup(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, with_requested_outputs=False)

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command=None,
    )

    assert result["schema_version"] == FIRST_GPU_E2E_READINESS_SCHEMA_VERSION
    assert result["status"] == "blocked"
    assert result["ready_for_first_gpu_attempt"] is False
    assert "requested_outputs:missing_requested_output:robot_eval_dataset" in result["blockers"]
    assert "requested_outputs:missing_requested_output:task_evaluation_run" in result["blockers"]
    assert "webapp_forwarding:missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL" in result["blockers"]
    assert "webapp_staged_request:missing_webapp_staged_inputs" in result["blockers"]
    assert "pipeline_gpu_handoff:missing_artifact:gpu_handoff_packet" in result["blockers"]
    assert "simulator_runtime:missing_simulator_command" in result["blockers"]
    assert result["stages"]["owner_gpu_proof"]["missing_is_expected_before_first_gpu_run"] is True
    assert result["claim_boundary"]["simulator_execution_performed"] is False


def test_first_gpu_readiness_accepts_ready_attempt_with_missing_owner_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
    command = tmp_path / "run_isaac_gpu_proof.sh"
    command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command=f"{command} --capture-root {capture_root}",
    )

    assert result["status"] == "ready_for_owner_gpu_attempt"
    assert result["ready_for_first_gpu_attempt"] is True
    assert result["owner_gpu_proof_ready"] is False
    assert result["blockers"] == []
    assert result["stages"]["pipeline_gpu_handoff"]["ready"] is True
    assert result["stages"]["webapp_forwarding"]["ready"] is True
    assert result["stages"]["webapp_staged_request"]["ready"] is True
    assert result["stages"]["webapp_staged_request"]["job_id"] == "webapp-job-1"
    assert result["stages"]["simulator_runtime"]["ready"] is True
    assert result["stages"]["simulator_runtime"]["command"]["command_location"] == "local"
    assert result["stages"]["simulator_runtime"]["command"]["executable_check_performed"] is True
    assert result["stages"]["owner_gpu_proof"]["missing_is_expected_before_first_gpu_run"] is True
    assert "simulator_runtime:runpod_allocation_is_external_or_request_manifest_only" in result["warnings"]


def test_first_gpu_readiness_uses_staged_webapp_request_for_upstream_truth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
            "site_submission_id": "capture-1",
            "request_id": "capture-1",
            "buyer_request_id": "capture-1",
            "capture_job_id": "capture-1",
        },
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
            "capture_capabilities": {"camera_pose": True},
            "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
            "site_submission_id": "capture-1",
            "request_id": "capture-1",
            "buyer_request_id": "capture-1",
            "capture_job_id": "capture-1",
        },
    )
    _write_gpu_handoff_artifacts(capture_root)
    staged_path = _write_staged_webapp_request(capture_root)
    command = tmp_path / "run_isaac_gpu_proof.sh"
    command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        webapp_staged_inputs_path=staged_path,
        simulator_command=f"{command} --capture-root {capture_root}",
    )

    upstream = result["stages"]["webapp_upstream_truth"]
    assert upstream["ready"] is True
    assert upstream["staged_webapp_request_used"] is True
    assert upstream["fields"] == {
        "site_submission_id": True,
        "request_id": True,
        "buyer_request_id": True,
        "capture_job_id": True,
    }
    assert all(
        source == "pipeline/live_pipeline_staged_inputs.json robot_eval_job_request.v1"
        for source in upstream["source_artifacts"].values()
    )
    assert not [
        blocker
        for blocker in result["blockers"]
        if blocker.startswith("webapp_upstream_truth:")
    ]


def test_first_gpu_readiness_accepts_remote_owner_command_without_local_executable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
    )

    command = result["stages"]["simulator_runtime"]["command"]
    assert result["status"] == "ready_for_owner_gpu_attempt"
    assert result["ready_for_first_gpu_attempt"] is True
    assert command["configured"] is True
    assert command["command_location"] == "remote"
    assert command["executable"] == "/opt/blueprint/run_isaac_gpu_proof.sh"
    assert command["executable_found"] is None
    assert command["executable_check_performed"] is False
    assert "simulator_runtime:simulator_command_executable_not_checked_remote_vm" in result[
        "warnings"
    ]


def test_first_gpu_readiness_accepts_webapp_forwarding_preflight_report_without_shell_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
    command = tmp_path / "run_isaac_gpu_proof.sh"
    command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    preflight_path = _write_webapp_forwarding_preflight_report(
        tmp_path / "forwarding_preflight.json",
    )
    monkeypatch.delenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", raising=False)
    monkeypatch.delenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", raising=False)
    monkeypatch.delenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        raising=False,
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        webapp_forwarding_preflight_path=preflight_path,
        simulator_command=f"{command} --capture-root {capture_root}",
    )

    stage = result["stages"]["webapp_forwarding"]
    assert result["status"] == "ready_for_owner_gpu_attempt"
    assert result["ready_for_first_gpu_attempt"] is True
    assert stage["forward_url_configured"] is False
    assert stage["forward_token_configured"] is False
    assert stage["forward_url_evidence_present"] is True
    assert stage["forward_token_evidence_present"] is True
    assert stage["capture_root_override_source"] == "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT"
    assert stage["forwarding_preflight"]["ready"] is True
    assert stage["forwarding_preflight"]["site_slug_covered"] is True
    assert stage["forwarding_preflight"]["probe_status"] == "reachable"
    assert "secret-token" not in json.dumps(result)


def test_first_gpu_readiness_blocks_bad_webapp_forwarding_preflight_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
    command = tmp_path / "run_isaac_gpu_proof.sh"
    command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    preflight_path = _write_webapp_forwarding_preflight_report(
        tmp_path / "forwarding_preflight.json",
        site_slugs=["other-site"],
    )
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        webapp_forwarding_preflight_path=preflight_path,
        simulator_command=f"{command} --capture-root {capture_root}",
    )

    assert result["status"] == "blocked"
    assert "webapp_forwarding:webapp_forwarding_preflight_missing_site_slug" in result[
        "blockers"
    ]
    assert result["stages"]["webapp_forwarding"]["forwarding_preflight"]["ready"] is False


def test_first_gpu_readiness_blocks_local_rehearsal_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root, local_rehearsal=True)
    command = tmp_path / "run_isaac_gpu_proof.sh"
    command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command=f"{command} --capture-root {capture_root}",
    )

    stage = result["stages"]["webapp_staged_request"]
    assert result["status"] == "blocked"
    assert "webapp_staged_request:webapp_staged_inputs_local_rehearsal_only" in result[
        "blockers"
    ]
    assert stage["local_rehearsal_only"] is True
    assert stage["local_rehearsal_allowed"] is False
    assert stage["source_kind"] == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    assert "webapp_staged_request:local_webapp_rehearsal_not_live_forwarding_proof" in result[
        "warnings"
    ]


def test_first_gpu_readiness_allows_local_rehearsal_when_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root, local_rehearsal=True)
    command = tmp_path / "run_isaac_gpu_proof.sh"
    command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command=f"{command} --capture-root {capture_root}",
        allow_local_webapp_rehearsal=True,
    )

    stage = result["stages"]["webapp_staged_request"]
    assert result["status"] == "ready_for_owner_gpu_attempt"
    assert result["ready_for_first_gpu_attempt"] is True
    assert "webapp_staged_request:webapp_staged_inputs_local_rehearsal_only" not in result[
        "blockers"
    ]
    assert stage["local_rehearsal_only"] is True
    assert stage["local_rehearsal_allowed"] is True
    assert "webapp_staged_request:local_webapp_rehearsal_not_live_forwarding_proof" in result[
        "warnings"
    ]


def test_first_gpu_readiness_blocks_invalid_existing_owner_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
    command = tmp_path / "run_isaac_gpu_proof.sh"
    command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    _write_json(
        capture_root / "pipeline" / "simulation_automation" / "gpu_owner_system_proof.json",
        {"scene_id": "wrong-scene", "exit_code": 1},
    )
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command=f"{command} --capture-root {capture_root}",
    )

    assert result["status"] == "blocked"
    assert "owner_gpu_proof:owner_gpu_proof_present_but_blocked" in result["blockers"]
    assert "owner_gpu_proof_scene_id_mismatch" in result["stages"]["owner_gpu_proof"]["proof_blockers"]


def test_first_gpu_readiness_cli_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    capture_root = _capture_root(tmp_path)
    output = tmp_path / "readiness.json"
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")

    exit_code = main(
        [
            "--capture-root",
            str(capture_root),
            "--no-require-webapp-forwarding",
            "--no-require-webapp-staged-request",
            "--simulator-command",
            "/missing/owner-command",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == FIRST_GPU_E2E_READINESS_SCHEMA_VERSION
    assert "simulator_runtime:simulator_command_executable_missing" in payload["blockers"]
