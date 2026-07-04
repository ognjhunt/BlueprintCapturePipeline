from __future__ import annotations

import json
from pathlib import Path

import blueprint_pipeline.first_gpu_e2e_readiness as readiness
from blueprint_pipeline.first_gpu_e2e_readiness import (
    FIRST_GPU_E2E_READINESS_SCHEMA_VERSION,
    LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
    _default_simulator_command,
    _default_staged_inputs_path,
    _first_executable,
    _nested_webapp_source,
    _parse_by_site_override,
    _path_matches,
    _pipeline_handoff_stage,
    _request_from_webapp_payload,
    _string_list,
    _webapp_forwarding_preflight_stage,
    _webapp_forwarding_stage,
    _webapp_staged_request_stage,
    _webapp_upstream_truth_stage,
    build_first_gpu_e2e_readiness,
    main,
)

import pytest

pytestmark = pytest.mark.slow


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
            "rank_fidelity_result_proven": False,
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
                "no_rank_fidelity_result_proven": True,
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


def _write_placeholder_upstream_ids(capture_root: Path) -> None:
    payload = {
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
        "capture_capabilities": {"camera_pose": True},
        "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
        "site_submission_id": "capture-1",
        "request_id": "capture-1",
        "buyer_request_id": "capture-1",
        "capture_job_id": "capture-1",
    }
    _write_json(capture_root / "raw" / "manifest.json", payload)
    _write_json(capture_root / "capture_descriptor.json", payload)


def test_first_gpu_readiness_small_helper_edges(tmp_path: Path, monkeypatch) -> None:
    assert _string_list("one") == ["one"]
    assert _string_list(42) == ["42"]
    assert _first_executable('"unterminated') == ""
    assert _first_executable("FOO=bar /bin/echo hello") == "/bin/echo"
    assert readiness._placeholder_like("", scene_id="scene-1", capture_id="capture-1")
    assert _request_from_webapp_payload(
        {"queue_contract": "robot_eval_job_request_inbox.v1", "job_request": {"job_id": "job-1"}}
    ) == {"job_id": "job-1"}
    assert _request_from_webapp_payload(
        {"schema_version": "robot_eval_job_request.v1", "job_id": "job-2"}
    )["job_id"] == "job-2"
    assert _request_from_webapp_payload({}) == {}
    assert _nested_webapp_source({}, "site_submission_id") == ""
    assert _path_matches("", tmp_path) is False

    loop = tmp_path / "loop"
    try:
        loop.symlink_to(loop)
        assert _path_matches(str(loop), tmp_path) is False
    except OSError:
        pass

    staged_inputs = tmp_path / "configured-staged.json"
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH", str(staged_inputs))
    assert _default_staged_inputs_path(tmp_path, None) == staged_inputs.resolve()
    monkeypatch.setenv("BLUEPRINT_ISAAC_LAB_ARENA_COMMAND", "/opt/arena/run.sh")
    assert _default_simulator_command("isaac_lab_arena", None) == "/opt/arena/run.sh"

    env_preflight = tmp_path / "env-preflight.json"
    _write_webapp_forwarding_preflight_report(env_preflight)
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT", str(env_preflight))
    assert _webapp_forwarding_preflight_stage(
        webapp_site_slug="site-1",
        require_webapp_forwarding=True,
        preflight_report_path=None,
    )["path"] == str(env_preflight.resolve())


def test_first_gpu_readiness_upstream_truth_handles_staged_request_edges(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_placeholder_upstream_ids(capture_root)
    no_request_id_payload = {
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
        "capture_capabilities": {"camera_pose": True},
        "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
        "site_submission_id": "capture-1",
        "buyer_request_id": "capture-1",
        "capture_job_id": "capture-1",
    }
    _write_json(capture_root / "raw" / "manifest.json", no_request_id_payload)
    _write_json(capture_root / "capture_descriptor.json", no_request_id_payload)
    _write_json(
        capture_root / "pipeline_handoff.json",
        {"owner_system": {"request_id": "request-from-owner-system"}},
    )
    stage = _webapp_upstream_truth_stage(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        staged_inputs_path=None,
    )
    assert stage["source_artifacts"]["request_id"] == "pipeline_handoff.json owner_system"
    _write_placeholder_upstream_ids(capture_root)

    invalid_staged = tmp_path / "invalid-staged.json"
    invalid_staged.write_text("{bad", encoding="utf-8")
    stage = _webapp_upstream_truth_stage(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        staged_inputs_path=invalid_staged,
    )
    assert "staged_webapp_request_read_failed:JSONDecodeError" in stage["warnings"]

    _write_json(
        tmp_path / "local-rehearsal-staged.json",
        {"local_rehearsal_only": True, "webapp_request": {}},
    )
    stage = _webapp_upstream_truth_stage(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        staged_inputs_path=tmp_path / "local-rehearsal-staged.json",
    )
    assert "staged_webapp_request_local_rehearsal_only" in stage["warnings"]

    invalid_request = tmp_path / "invalid-request.json"
    invalid_request.write_text("{bad", encoding="utf-8")
    _write_json(
        tmp_path / "invalid-request-staged.json",
        {"webapp_request": {"path": str(invalid_request)}},
    )
    stage = _webapp_upstream_truth_stage(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        staged_inputs_path=tmp_path / "invalid-request-staged.json",
    )
    assert "staged_webapp_request_payload_read_failed:JSONDecodeError" in stage["warnings"]

    _write_json(
        tmp_path / "path-missing-staged.json",
        {"webapp_request": {}},
    )
    stage = _webapp_upstream_truth_stage(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        staged_inputs_path=tmp_path / "path-missing-staged.json",
    )
    assert "staged_webapp_request_path_missing" in stage["warnings"]

    _write_json(
        tmp_path / "missing-file-staged.json",
        {"webapp_request": {"path": str(tmp_path / "missing-request.json")}},
    )
    stage = _webapp_upstream_truth_stage(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        staged_inputs_path=tmp_path / "missing-file-staged.json",
    )
    assert "staged_webapp_request_file_missing" in stage["warnings"]

    mismatch_request = tmp_path / "mismatch-request.json"
    _write_json(
        mismatch_request,
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "job-1",
            "site_package": {"capture_root": str(tmp_path / "other-capture")},
        },
    )
    _write_json(
        tmp_path / "mismatch-staged.json",
        {"webapp_request": {"path": str(mismatch_request)}},
    )
    stage = _webapp_upstream_truth_stage(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        staged_inputs_path=tmp_path / "mismatch-staged.json",
    )
    assert "staged_webapp_request_capture_root_mismatch" in stage["warnings"]

    list_staged = tmp_path / "list-staged.json"
    list_staged.write_text("[]", encoding="utf-8")
    stage = _webapp_upstream_truth_stage(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        staged_inputs_path=list_staged,
    )
    assert "staged_webapp_inputs_not_json_object" in stage["warnings"]


def test_first_gpu_readiness_forwarding_preflight_edges(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON", "{bad")
    assert _parse_by_site_override()["blockers"] == [
        "invalid_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"
    ]
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON", "[]")
    assert _parse_by_site_override()["blockers"] == [
        "invalid_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"
    ]
    monkeypatch.delenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON", raising=False)

    missing = _webapp_forwarding_preflight_stage(
        webapp_site_slug="site-1",
        require_webapp_forwarding=True,
        preflight_report_path=tmp_path / "missing.json",
    )
    assert missing["blockers"] == ["webapp_forwarding_preflight_report_missing"]

    invalid_json = tmp_path / "invalid-preflight.json"
    invalid_json.write_text("{bad", encoding="utf-8")
    invalid = _webapp_forwarding_preflight_stage(
        webapp_site_slug="site-1",
        require_webapp_forwarding=True,
        preflight_report_path=invalid_json,
    )
    assert invalid["blockers"] == [
        "webapp_forwarding_preflight_report_read_failed:JSONDecodeError"
    ]

    non_mapping = tmp_path / "non-mapping-preflight.json"
    non_mapping.write_text("[]", encoding="utf-8")
    assert _webapp_forwarding_preflight_stage(
        webapp_site_slug="site-1",
        require_webapp_forwarding=True,
        preflight_report_path=non_mapping,
    )["blockers"] == ["webapp_forwarding_preflight_report_not_json_object"]

    optional = tmp_path / "optional-preflight.json"
    _write_webapp_forwarding_preflight_report(
        optional,
        status="ready_for_optional_forwarding",
    )
    optional_stage = _webapp_forwarding_preflight_stage(
        webapp_site_slug="site-1",
        require_webapp_forwarding=False,
        preflight_report_path=optional,
    )
    assert optional_stage["preflight_status"] == "ready_for_optional_forwarding"

    bad = tmp_path / "bad-preflight.json"
    _write_json(
        bad,
        {
            "schema_version": "wrong",
            "status": "blocked",
            "forwarding_required": False,
            "endpoint_configured": False,
            "configured_env": {
                "forward_url": {"valid": False},
                "forward_token": {"configured": False, "redacted": False},
                "forward_timeout_ms": {"valid": False},
                "capture_root_by_site_json": {
                    "configured": True,
                    "valid": False,
                    "site_slugs": [],
                },
                "single_capture_root_override": {"configured": False},
            },
            "blockers": ["preflight_report_blocked"],
            "proof_boundary": {},
            "probe": {"requested": True, "status": "unreachable"},
        },
    )
    bad_stage = _webapp_forwarding_preflight_stage(
        webapp_site_slug="",
        require_webapp_forwarding=True,
        preflight_report_path=bad,
    )
    assert set(bad_stage["blockers"]) >= {
        "webapp_forwarding_preflight_schema_mismatch",
        "webapp_forwarding_preflight_status:blocked",
        "webapp_forwarding_preflight_not_required_mode",
        "webapp_forwarding_preflight_endpoint_not_configured",
        "webapp_forwarding_preflight_forward_url_invalid",
        "webapp_forwarding_preflight_token_not_configured",
        "webapp_forwarding_preflight_token_not_redacted",
        "webapp_forwarding_preflight_timeout_invalid",
        "webapp_forwarding_preflight_capture_root_map_invalid",
        "webapp_forwarding_preflight_report_has_blockers",
        "webapp_forwarding_preflight_probe_not_reachable",
    }
    assert any(
        blocker.startswith("webapp_forwarding_preflight_boundary_missing:")
        for blocker in bad_stage["blockers"]
    )
    assert "webapp_forwarding_preflight_site_slug_not_checked" in bad_stage["warnings"]

    not_probed = tmp_path / "not-probed-preflight.json"
    _write_webapp_forwarding_preflight_report(not_probed)
    payload = json.loads(not_probed.read_text(encoding="utf-8"))
    payload["probe"] = {"requested": False}
    _write_json(not_probed, payload)
    not_probed_stage = _webapp_forwarding_preflight_stage(
        webapp_site_slug="site-1",
        require_webapp_forwarding=True,
        preflight_report_path=not_probed,
    )
    assert "webapp_forwarding_preflight_not_network_probed" in not_probed_stage["warnings"]


def test_first_gpu_readiness_staged_request_stage_edges(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    assert _webapp_staged_request_stage(
        capture_root,
        staged_inputs_path=None,
        require_webapp_staged_request=False,
        allow_local_webapp_rehearsal=False,
    )["status"] == "not_required"

    invalid = tmp_path / "invalid-staged.json"
    invalid.write_text("{bad", encoding="utf-8")
    assert _webapp_staged_request_stage(
        capture_root,
        staged_inputs_path=invalid,
        require_webapp_staged_request=True,
        allow_local_webapp_rehearsal=False,
    )["blockers"] == ["webapp_staged_inputs_read_failed:JSONDecodeError"]

    non_mapping = tmp_path / "non-mapping-staged.json"
    non_mapping.write_text("[]", encoding="utf-8")
    assert _webapp_staged_request_stage(
        capture_root,
        staged_inputs_path=non_mapping,
        require_webapp_staged_request=True,
        allow_local_webapp_rehearsal=False,
    )["blockers"] == ["webapp_staged_inputs_not_json_object"]

    malformed = tmp_path / "malformed-staged.json"
    _write_json(
        malformed,
        {
            "schema_version": "wrong",
            "configured_capture_root": str(tmp_path / "other"),
            "webapp_request": {"staged": False, "ready": False},
        },
    )
    malformed_stage = _webapp_staged_request_stage(
        capture_root,
        staged_inputs_path=malformed,
        require_webapp_staged_request=True,
        allow_local_webapp_rehearsal=False,
    )
    assert set(malformed_stage["blockers"]) >= {
        "webapp_staged_inputs_schema_mismatch",
        "webapp_staged_inputs_capture_root_mismatch",
        "webapp_request_not_staged",
        "webapp_request_not_ready",
        "webapp_request_path_missing",
        "webapp_request_job_id_missing",
    }

    missing_configured = tmp_path / "missing-configured-staged.json"
    _write_json(
        missing_configured,
        {
            "schema_version": "blueprint_live_pipeline_staged_inputs.v1",
            "webapp_request": {"staged": True, "ready": True, "job_id": "job-1"},
        },
    )
    assert "webapp_staged_inputs_missing_configured_capture_root" in _webapp_staged_request_stage(
        capture_root,
        staged_inputs_path=missing_configured,
        require_webapp_staged_request=True,
        allow_local_webapp_rehearsal=False,
    )["blockers"]

    invalid_request = tmp_path / "invalid-request.json"
    invalid_request.write_text("{bad", encoding="utf-8")
    request_cases = {
        "missing-file": tmp_path / "missing-request.json",
        "invalid-json": invalid_request,
    }
    for name, request_path in request_cases.items():
        staged = tmp_path / f"{name}-staged.json"
        _write_json(
            staged,
            {
                "schema_version": "blueprint_live_pipeline_staged_inputs.v1",
                "configured_capture_root": str(capture_root.resolve()),
                "webapp_request": {
                    "staged": True,
                    "ready": True,
                    "job_id": "job-1",
                    "path": str(request_path),
                },
            },
        )
        blockers = _webapp_staged_request_stage(
            capture_root,
            staged_inputs_path=staged,
            require_webapp_staged_request=True,
            allow_local_webapp_rehearsal=False,
        )["blockers"]
        expected = (
            "webapp_request_file_missing"
            if name == "missing-file"
            else "webapp_request_read_failed:JSONDecodeError"
        )
        assert expected in blockers

    list_request = tmp_path / "list-request.json"
    list_request.write_text("[]", encoding="utf-8")
    mismatch_request = tmp_path / "mismatch-request.json"
    _write_json(
        mismatch_request,
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "job-1",
            "site_package": {"capture_root": str(tmp_path / "other")},
        },
    )
    missing_upstream_request = tmp_path / "missing-upstream-request.json"
    _write_json(
        missing_upstream_request,
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "job-1",
            "site_package": {"capture_root": str(capture_root.resolve())},
        },
    )
    for request_path, expected in (
        (list_request, "webapp_request_not_robot_eval_job_request_v1"),
        (mismatch_request, "webapp_request_capture_root_mismatch"),
        (missing_upstream_request, "webapp_request_missing_required_upstream_ids"),
    ):
        staged = tmp_path / f"{request_path.stem}-staged.json"
        _write_json(
            staged,
            {
                "schema_version": "blueprint_live_pipeline_staged_inputs.v1",
                "configured_capture_root": str(capture_root.resolve()),
                "webapp_request": {
                    "staged": True,
                    "ready": True,
                    "job_id": "job-1",
                    "path": str(request_path),
                },
            },
        )
        assert expected in _webapp_staged_request_stage(
            capture_root,
            staged_inputs_path=staged,
            require_webapp_staged_request=True,
            allow_local_webapp_rehearsal=False,
        )["blockers"]


def test_first_gpu_readiness_forwarding_stage_override_edges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")

    missing_slug = _webapp_forwarding_stage(
        capture_root,
        webapp_site_slug="",
        require_webapp_forwarding=True,
        webapp_forwarding_preflight_path=None,
    )
    assert "missing_webapp_site_slug_for_capture_root_override" in missing_slug["blockers"]

    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT",
        str(tmp_path / "wrong-capture-root"),
    )
    mismatch = _webapp_forwarding_stage(
        capture_root,
        webapp_site_slug="site-1",
        require_webapp_forwarding=True,
        webapp_forwarding_preflight_path=None,
    )
    assert mismatch["capture_root_override_source"] == "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT"
    assert "pipeline_capture_root_override_does_not_match_capture_root" in mismatch[
        "blockers"
    ]


def test_first_gpu_readiness_pipeline_handoff_rejects_illegal_claim_upgrade(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    handoff_path = (
        capture_root / "pipeline" / "simulation_automation" / "gpu_handoff_packet.json"
    )
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    payload["status"] = "blocked"
    payload["rank_fidelity_result_proven"] = True
    payload["public_claim_upgrade_allowed"] = True
    payload["blockers"] = [
        "owner_gpu_simulator_execution_not_run",
        "operator_gpu_driver_missing",
    ]
    _write_json(handoff_path, payload)

    stage = _pipeline_handoff_stage(capture_root)

    assert "gpu_handoff_packet_not_ready" in stage["blockers"]
    assert "gpu_handoff_illegally_marks_rank_fidelity" in stage["blockers"]
    assert "gpu_handoff_illegally_allows_public_claim_upgrade" in stage["blockers"]
    assert "operator_gpu_driver_missing" in stage["blockers"]


def test_first_gpu_readiness_build_normalizes_invalid_location_and_owner_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    seen: dict[str, str] = {}

    def ready_stage(*_args, **_kwargs):
        return {"status": "ready", "ready": True, "blockers": [], "warnings": []}

    def simulator_stage(**kwargs):
        seen["location"] = kwargs["simulator_command_location"]
        return {"status": "ready", "ready": True, "blockers": [], "warnings": []}

    monkeypatch.setattr(readiness, "_capture_preflight_stage", ready_stage)
    monkeypatch.setattr(readiness, "_requested_outputs_stage", ready_stage)
    monkeypatch.setattr(readiness, "_webapp_upstream_truth_stage", ready_stage)
    monkeypatch.setattr(readiness, "_webapp_forwarding_stage", ready_stage)
    monkeypatch.setattr(readiness, "_webapp_staged_request_stage", ready_stage)
    monkeypatch.setattr(readiness, "_pipeline_handoff_stage", ready_stage)
    monkeypatch.setattr(readiness, "_simulator_runtime_stage", simulator_stage)
    monkeypatch.setattr(
        readiness,
        "_owner_gpu_proof_stage",
        lambda _capture_root: {
            "status": "proven",
            "ready": True,
            "blockers": [],
            "warnings": [],
        },
    )

    result = build_first_gpu_e2e_readiness(
        capture_root=capture_root,
        simulator_command="/remote/command",
        simulator_command_location="invalid",
    )

    assert seen["location"] == "local"
    assert result["status"] == "owner_gpu_proof_present_audit_closure_next"


def test_first_gpu_readiness_cli_returns_zero_for_ready_attempt(
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
    output = tmp_path / "ready-readiness.json"

    exit_code = main(
        [
            "--capture-root",
            str(capture_root),
            "--webapp-site-slug",
            "site-1",
            "--simulator-command",
            f"{command} --capture-root {capture_root}",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ready_for_owner_gpu_attempt"
