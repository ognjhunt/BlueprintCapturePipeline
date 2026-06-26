from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.cross_repo_first_gpu_readiness import (
    CROSS_REPO_FIRST_GPU_READINESS_SCHEMA_VERSION,
    _build_first_gpu_external_input_packet,
    _build_gpu_spend_decision,
    _build_remediation_plan,
    _file_contains_check,
    _first_gpu_external_input_packet_markdown,
    _guarded_commands_by_category,
    _read_text,
    _remediation_for_blocker,
    _runtime_has_local_webapp_rehearsal,
    _runtime_preflight_result_summary,
    _run_packet_phase,
    build_cross_repo_first_gpu_readiness,
    main,
)
from blueprint_pipeline.first_gpu_run_packet import build_first_gpu_run_packet


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload, indent=2))


def _write_gpu_vm_runtime_preflight_result(capture_root: Path) -> None:
    _write_json(
        capture_root
        / "pipeline"
        / "first_gpu_e2e_run_packet"
        / "gpu_vm_runtime_preflight_result.json",
        {
            "schema_version": "first_gpu_vm_runtime_preflight_result.v1",
            "status": "ready_for_owner_command_attempt",
            "blockers": [],
            "warnings": [],
            "nvidia_smi": {"status": "ready"},
            "owner_command": {"status": "ready"},
            "sync_manifest": {"status": "ready"},
            "claim_boundary": {
                "simulator_execution_performed": False,
                "rank_fidelity_result_proven": False,
            },
        },
    )


def _pipeline_repo(tmp_path: Path) -> Path:
    root = tmp_path / "BlueprintCapturePipeline"
    _write(
        root / "pyproject.toml",
        "\n".join(
            [
                "blueprint-audit-first-gpu-e2e-readiness",
                "blueprint-stage-first-gpu-sample-video",
                "blueprint-run-owner-gpu-proof",
                "blueprint-write-owner-gpu-default-smoke-artifacts",
                "blueprint-build-first-gpu-run-packet",
            ]
        ),
    )
    _write(
        root / "src/blueprint_pipeline/live_pipeline_input_intake.py",
        "\n".join(
            [
                "LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION",
                "stage_webapp_request",
                "_audit_webapp_request",
                "WEBAPP_UPSTREAM_REQUIRED_FIELDS",
            ]
        ),
    )
    _write(
        root / "src/blueprint_pipeline/first_gpu_e2e_readiness.py",
        "\n".join(
            [
                "missing_webapp_staged_inputs",
                "webapp_staged_inputs_local_rehearsal_only",
                "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT",
                "owner_gpu_simulator_execution_not_run",
                "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
            ]
        ),
    )
    _write(
        root / "src/blueprint_pipeline/first_gpu_run_packet.py",
        "\n".join(
            [
                "gpu_provider_bootstrap.md",
                "gpu_provider_bootstrap.json",
                "webapp_handoff_verification_commands",
                "nvidia_nim_boundary",
                "avoid_for_isaac_sim",
                "blueprint-write-owner-gpu-default-smoke-artifacts",
                "owner_default_smoke_command_binding.sh",
                "live_policy_execution_contract.md",
                "default_test_robot_eval_job_request.template.json",
                "real_robot_pov_manifest.template.json",
                "stage_first_gpu_live_inputs.sh",
            ]
        ),
    )
    _write(
        root / "src/blueprint_pipeline/owner_gpu_proof_runner.py",
        "\n".join(
            [
                "BLUEPRINT_SCENE_LOAD_TRACE",
                "BLUEPRINT_SPAWN_TRACE",
                "BLUEPRINT_ACTION_OR_POLICY_TRACE",
                "BLUEPRINT_DEFAULT_SMOKE_POLICY",
                "BLUEPRINT_POLICY_EXECUTION_TRACE",
                "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE",
                "owner_gpu_simulator_execution_proof_manifest.json",
            ]
        ),
    )
    _write(
        root / "docs/FIRST_GPU_E2E_RUNBOOK.md",
        "\n".join(
            [
                "--provisioner runpod",
                "--allow-local-webapp-rehearsal",
                "blueprint-run-owner-gpu-proof",
                "Phase 3: GPU VM Bring-Up",
            ]
        ),
    )
    return root


def _capture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "BlueprintCapture"
    _write(
        root / "BlueprintCapture/Services/CaptureUploadService.swift",
        "\n".join(
            [
                'completionMarkerFilename = "capture_upload_complete.json"',
                '"robot_eval_dataset"',
                '"task_evaluation_run"',
                '"requested_outputs"',
                '"site_submission_id"',
                '"buyer_request_id"',
                '"capture_job_id"',
            ]
        ),
    )
    _write(
        root / "BlueprintCapture/Services/CaptureRawContractV3Validator.swift",
        '"capture_upload_complete.json"\n"manifest.json"',
    )
    _write(
        root / "cloud/extract-frames/src/index.ts",
        "\n".join(
            [
                "capture.raw_upload_complete.v1",
                "capture_descriptor.json",
                "pipeline_handoff.json",
                "robot_eval_dataset",
                "task_evaluation_run",
                "site_submission_id",
                "buyer_request_id",
                "capture_job_id",
            ]
        ),
    )
    _write(
        root / "cloud/extract-frames/src/index.test.ts",
        "\n".join(
            [
                "invalid_site_submission_id_placeholder",
                "invalid_capture_job_id_matches_capture_id",
                "pipeline_handoff_uri",
                "robot_eval_dataset_requested",
            ]
        ),
    )
    return root


def _webapp_repo(tmp_path: Path) -> Path:
    root = tmp_path / "Blueprint-WebApp"
    _write(
        root / "server/utils/robotEvalJobRequests.ts",
        "\n".join(
            [
                "robot_eval_job_request.v1",
                "robot_eval_job_request_inbox.v1",
                "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT",
                "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
                "capture_root_override_source",
                "public_claim_upgrade_allowed: false",
                "ready_for_owner_gpu_preflight",
            ]
        ),
    )
    _write(
        root / "server/routes/robot-eval-job-requests.ts",
        "\n".join(
            [
                "Invalid robot_eval_job_request.v1",
                "ROBOT_EVAL_JOB_REQUEST_INBOX_DIR",
                "forwardRobotEvalJobRequestToPipeline",
            ]
        ),
    )
    _write(
        root / "client/src/lib/robotEvalJobRequest.ts",
        "\n".join(
            [
                "robot_eval_job_request.v1",
                "cpu_pre_gpu_preflight",
                "ready_for_owner_gpu_preflight",
            ]
        ),
    )
    _write(
        root / "server/utils/pipelineStateMachine.ts",
        "robot_eval_job_request_uri\nready_for_owner_gpu_preflight",
    )
    _write(
        root / "server/tests/robot-eval-job-requests.test.ts",
        "\n".join(
            [
                "creates a durable Pipeline robot_eval_job_request.v1",
                "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
                "ready_for_owner_gpu_preflight: false",
                "missing_pipeline_capture_root_override_for_webapp_synced_artifact",
            ]
        ),
    )
    _write(
        root / "scripts/pipeline/export-first-gpu-webapp-rehearsal-request.ts",
        "\n".join(
            [
                "buildRobotEvalJobRequest",
                "validateRobotEvalJobRequest",
                "local_first_gpu_rehearsal_request",
                "live_webapp_forwarding_proven: false",
            ]
        ),
    )
    _write(
        root / "scripts/pipeline/audit-robot-eval-forwarding-readiness.ts",
        "\n".join(
            [
                "blueprint.webapp.robot_eval_forwarding_readiness.v1",
                "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN",
                "probe-intake-audit",
                "redacted: true",
                "no_job_queued",
                "no_gpu_allocated",
                "no_simulator_execution_proven",
            ]
        ),
    )
    return root


def _runtime_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    requested_outputs = ["qualification", "robot_eval_dataset", "task_evaluation_run"]
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
        request["source_kind"] = "local_first_gpu_rehearsal_request"
    envelope = {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": "webapp-job-1",
        "job_request": request,
    }
    if local_rehearsal:
        envelope["source_kind"] = "local_first_gpu_rehearsal_request"
        envelope["local_rehearsal_only"] = True
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
        staged_payload["source_kind"] = "local_first_gpu_rehearsal_request"
        staged_payload["local_rehearsal_only"] = True
        staged_payload["webapp_request"]["source_kind"] = "local_first_gpu_rehearsal_request"
    _write_json(staged_path, staged_payload)
    return staged_path


def _write_webapp_forwarding_preflight_report(path: Path, *, site_slug: str = "site-1") -> Path:
    _write_json(
        path,
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": "ready_for_required_forwarding_with_probe",
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
                    "site_count": 1,
                    "site_slugs": [site_slug],
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


def _write_scene_asset_artifacts(capture_root: Path) -> None:
    pipeline_dir = capture_root / "pipeline"
    automation_dir = pipeline_dir / "simulation_automation"
    asset_path = pipeline_dir / "worldlabs_assets" / "scene.glb"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"glb-scene")
    _write_json(
        pipeline_dir / "source_video_preflight_manifest.json",
        {
            "schema_version": "first_gpu_sample_video_preflight.v1",
            "status": "ready",
            "ready_for_worldlabs_first_clip_count": 1,
            "candidates": [],
        },
    )
    _write_json(
        pipeline_dir / "worldlabs_request_manifest.json",
        {"schema_version": "worldlabs_request_manifest.v1", "status": "submitted"},
    )
    _write_json(
        pipeline_dir / "worldlabs_world_manifest.json",
        {"schema_version": "worldlabs_world_manifest.v1", "status": "completed"},
    )
    _write_json(
        pipeline_dir / "worldlabs_export_manifest.json",
        {"output_collider_mesh_path": str(asset_path)},
    )
    _write_json(
        pipeline_dir / "worldlabs_assets" / "materialized_assets_manifest.json",
        {
            "schema_version": "worldlabs_materialized_assets.v1",
            "status": "ready",
            "downloads": [{"kind": "worldlabs_materialized_scene", "local_path": str(asset_path)}],
        },
    )
    _write_json(
        automation_dir / "scene_asset_preflight.json",
        {
            "schema_version": "scene_asset_preflight.v1",
            "status": "ready",
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "scene_frame_estimate.json",
        {
            "schema_version": "scene_frame_estimate.v1",
            "status": "ready",
            "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
            "floor_z_estimate": 0,
        },
    )
    _write_json(
        automation_dir / "spawn_pose_validation_manifest.json",
        {
            "schema_version": "spawn_pose_validation_manifest.v1",
            "status": "ready",
            "valid_spawn_candidate_count": 1,
        },
    )


def _make_runtime_ready(capture_root: Path, monkeypatch) -> None:
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


def test_cross_repo_first_gpu_readiness_blocks_without_runtime_capture(
    tmp_path: Path,
) -> None:
    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
    )

    assert result["schema_version"] == CROSS_REPO_FIRST_GPU_READINESS_SCHEMA_VERSION
    assert result["status"] == "blocked"
    assert result["phases"]["capture_to_pipeline"]["ready"] is True
    assert result["phases"]["webapp_to_pipeline"]["ready"] is True
    assert result["phases"]["pipeline_return"]["ready"] is True
    assert result["phases"]["runtime_capture"]["ready"] is False
    assert result["phases"]["run_packet"]["status"] == "not_checked"
    assert result["phases"]["run_packet"]["blockers"] == []
    assert (
        "runtime_capture:missing_capture_root_for_runtime_first_gpu_readiness"
        in result["blockers"]
    )
    remediation = result["remediation_plan"]
    assert remediation["status"] == "blocked"
    assert remediation["categories"]["sample_capture"]["safe_commands"] == [
        "blueprint-stage-first-gpu-sample-video --source-video <video> "
        "--storage-root output/first-gpu-sample-storage --scene-id <scene> "
        "--capture-id <capture> --run-simulation-automation"
    ]
    spend_decision = result["gpu_spend_decision"]
    assert spend_decision["status"] == "do_not_rent_gpu_yet"
    assert spend_decision["gpu_rental_recommended_now"] is False
    assert spend_decision["recommended_first_gpu_environment"] == "interactive_gpu_vm_or_pod"
    assert "sample_capture" in spend_decision["pre_spend_blocker_categories"]
    assert "do_not_allocate_runpod_or_equivalent_gpu_vm" in (
        spend_decision["must_not_do_until_ready"]
    )
    assert "not the primary Isaac/physics simulator runtime" in spend_decision["nvidia_nim_role"]
    assert spend_decision["claim_boundary"]["gpu_provisioning_performed"] is False
    external_inputs = result["first_gpu_external_input_packet"]
    assert external_inputs["schema_version"] == "first_gpu_external_input_packet.v1"
    assert external_inputs["status"] == "blocked"
    assert external_inputs["next_missing_category_id"] == "sample_capture"
    assert external_inputs["gpu_rental_recommended_now"] is False
    assert external_inputs["claim_boundary"]["external_inputs_collected"] is False
    assert result["claim_boundary"]["gpu_provisioning_performed"] is False


def test_cross_repo_first_gpu_readiness_surfaces_contract_gaps(
    tmp_path: Path,
) -> None:
    webapp_repo = _webapp_repo(tmp_path)
    (webapp_repo / "server/utils/robotEvalJobRequests.ts").write_text(
        "robot_eval_job_request.v1",
        encoding="utf-8",
    )

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=webapp_repo,
    )

    assert result["phases"]["webapp_to_pipeline"]["ready"] is False
    assert any(
        "webapp_to_pipeline:request_builder:missing_contract_text:"
        "server/utils/robotEvalJobRequests.ts:queue_contract" in blocker
        for blocker in result["blockers"]
    )
    assert result["remediation_plan"]["categories"]["repo_contract_or_unknown"]["blocker_count"] >= 1


def test_cross_repo_first_gpu_remediation_groups_runtime_blockers(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write(
        capture_root / "raw" / "manifest.json",
        json.dumps(
            {
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "video_uri": "walkthrough.mov",
                "requested_outputs": [
                    "qualification",
                    "robot_eval_dataset",
                    "task_evaluation_run",
                ],
            }
        ),
    )
    _write(capture_root / "raw" / "capture_context.json", json.dumps({"workflowName": "Test"}))
    _write(
        capture_root / "raw" / "capture_upload_complete.json",
        json.dumps({"scene_id": "scene-1", "capture_id": "capture-1"}),
    )
    (capture_root / "raw" / "walkthrough.mov").write_bytes(b"fake-video")
    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/missing/run_isaac_gpu_proof.sh",
    )

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/missing/run_isaac_gpu_proof.sh",
    )

    categories = result["remediation_plan"]["categories"]
    assert "webapp_upstream_truth" in categories
    assert "webapp_forwarding_env" in categories
    assert "webapp_staged_request" in categories
    assert "owner_gpu_command" in categories
    assert "owner_gpu_gate" in categories
    assert any(
        command.startswith("export ROBOT_EVAL_JOB_REQUEST_FORWARD_URL=")
        for command in categories["webapp_forwarding_env"]["safe_commands"]
    )
    assert categories["owner_gpu_gate"]["safe_commands"] == [
        "export BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true"
    ]
    spend_decision = result["gpu_spend_decision"]
    assert spend_decision["status"] == "do_not_rent_gpu_yet"
    assert spend_decision["gpu_rental_recommended_now"] is False
    assert "webapp_upstream_truth" in spend_decision["pre_spend_blocker_categories"]
    assert "pipeline_gpu_handoff" in spend_decision["pre_spend_blocker_categories"]
    assert "owner_gpu_command" in spend_decision["pre_spend_blocker_categories"]
    assert "do_not_run_gpu_vm_commands" in spend_decision["must_not_do_until_ready"]
    external_inputs = result["first_gpu_external_input_packet"]
    missing_by_category = {
        item["category_id"]: item for item in external_inputs["missing_inputs"]
    }
    assert external_inputs["next_missing_category_id"] == "webapp_upstream_truth"
    assert {
        item["name"] for item in missing_by_category["webapp_forwarding_env"]["required_inputs"]
    } >= {
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL",
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN",
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
    }
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN" in external_inputs[
        "secret_handling"
    ]["secret_input_names"]
    assert {
        item["name"] for item in missing_by_category["owner_gpu_gate"]["required_inputs"]
    } == {"BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"}
    owner_command_inputs = missing_by_category["owner_gpu_command"]
    assert {
        item["name"] for item in owner_command_inputs["required_inputs"]
    } >= {
        "OWNER_SIMULATOR_COMMAND",
        "owner_default_smoke_command_binding.sh",
        "OWNER_SCENE_LOAD_COMMAND",
        "OWNER_ROBOT_SPAWN_COMMAND",
        "OWNER_WALK_TO_TARGET_COMMAND",
        "SIM_ROBOT_POV_FRAME_PATH or SIM_ROBOT_POV_VIDEO_PATH",
    }
    owner_command_guarded = owner_command_inputs["guarded_commands"]
    assert any(
        item["name"] == "owner_command_binding_template_syntax_check"
        and item["safe_to_run_now"] is True
        and item["runs_owner_simulator_command"] is False
        for item in owner_command_guarded
    )
    proof_scope = external_inputs["first_gpu_proof_scope"]
    assert proof_scope["default_simulator_smoke"]["policy"] == "walk_to_target"
    assert proof_scope["default_simulator_smoke"][
        "owner_binding_template_exists"
    ] is True
    assert proof_scope["contract_artifacts"][
        "live_policy_execution_contract_exists"
    ] is True
    assert proof_scope["live_input_templates"][
        "default_test_robot_eval_job_request_template_exists"
    ] is True
    assert proof_scope["live_input_templates"][
        "real_robot_pov_manifest_template_exists"
    ] is True
    assert proof_scope["live_input_templates"]["live_input_staging_script_exists"] is True
    assert proof_scope["live_input_templates"]["staging_gate"] == (
        "BLUEPRINT_ALLOW_STAGING_FIRST_GPU_LIVE_INPUTS=true"
    )
    not_proven_claims = {
        item["claim"] for item in proof_scope["not_proven_by_first_gpu_smoke"]
    }
    assert not_proven_claims == {
        "live_robot_team_policy_execution",
        "real_robot_pov_evidence",
    }
    assert external_inputs["claim_boundary"]["default_sim_policy_execution_proven"] is False
    assert external_inputs["claim_boundary"]["sim_robot_pov_evidence_proven"] is False
    assert external_inputs["claim_boundary"]["robot_policy_execution_proven"] is False
    assert external_inputs["claim_boundary"]["real_robot_pov_evidence_proven"] is False
    upstream_guarded = missing_by_category["webapp_upstream_truth"]["guarded_commands"]
    assert upstream_guarded[0]["name"] == "webapp_upstream_truth_verification_commands"
    assert upstream_guarded[0]["path"].endswith(
        "webapp_upstream_truth_verification_commands.sh"
    )
    assert upstream_guarded[0]["safe_to_run_now"] is True
    assert upstream_guarded[0]["runs_live_webapp_call"] is False
    assert upstream_guarded[0]["runs_owner_simulator_command"] is False


def test_cross_repo_first_gpu_spend_blocks_when_ready_runtime_has_no_run_packet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _make_runtime_ready(capture_root, monkeypatch)

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
        output_path=tmp_path / "cross-repo-with-provider-blockers.json",
    )

    assert result["phases"]["runtime_capture"]["ready"] is True
    assert result["phases"]["run_packet"]["ready"] is False
    assert result["status"] == "blocked"
    assert "run_packet:missing_first_gpu_run_packet" in result["blockers"]
    assert "run_packet:missing_first_gpu_blocker_resolution" in result["blockers"]
    assert "run_packet:missing_first_gpu_webapp_handoff" in result["blockers"]
    assert "run_packet:missing_first_gpu_scene_asset_acquisition" in result["blockers"]
    assert "run_packet:missing_first_gpu_launch_order" in result["blockers"]
    assert "run_packet:missing_gpu_vm_runtime_preflight_plan" in result["blockers"]
    assert "run_packet:missing_gpu_vm_sync_manifest" in result["blockers"]
    spend_decision = result["gpu_spend_decision"]
    assert spend_decision["status"] == "do_not_rent_gpu_yet"
    assert spend_decision["gpu_rental_recommended_now"] is False
    assert "first_gpu_run_packet" in spend_decision["pre_spend_blocker_categories"]
    assert "gpu_vm_sync" in spend_decision["pre_spend_blocker_categories"]
    assert "gpu_vm_runtime_preflight" in spend_decision["pre_spend_blocker_categories"]


def test_cross_repo_first_gpu_uses_webapp_forwarding_preflight_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
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

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        webapp_forwarding_preflight_path=preflight_path,
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
        output_path=tmp_path / "cross-repo-with-preflight-report.json",
    )

    runtime = result["phases"]["runtime_capture"]
    forwarding = runtime["readiness"]["stages"]["webapp_forwarding"]
    assert runtime["ready"] is True
    assert forwarding["forward_url_configured"] is False
    assert forwarding["forward_url_evidence_present"] is True
    assert forwarding["forward_token_configured"] is False
    assert forwarding["forward_token_evidence_present"] is True
    assert forwarding["capture_root_override_source"] == (
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT"
    )
    assert forwarding["forwarding_preflight"]["ready"] is True
    assert forwarding["forwarding_preflight"]["probe_status"] == "reachable"
    assert not [
        blocker
        for blocker in result["blockers"]
        if "missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD" in blocker
    ]
    assert result["status"] == "blocked"
    assert "run_packet:missing_first_gpu_run_packet" in result["blockers"]


def test_cross_repo_first_gpu_surfaces_run_packet_operator_actions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv("WORLDLABS_API_KEY", "secret-worldlabs-key")
    monkeypatch.setenv("BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION", "true")
    _write_json(
        capture_root / "pipeline" / "source_video_preflight_manifest.json",
        {
            "schema_version": "first_gpu_sample_video_preflight.v1",
            "status": "ready",
            "ready_for_worldlabs_first_clip_count": 1,
            "candidates": [],
        },
    )

    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
        output_path=tmp_path / "cross-repo-with-provider-blockers.json",
    )

    action_categories = {
        str(item["category_id"]) for item in result["first_gpu_operator_actions"]
    }
    assert result["status"] == "blocked"
    assert result["first_gpu_operator_action_count"] == len(
        result["first_gpu_operator_actions"]
    )
    assert result["blocked_first_gpu_operator_action_count"] >= 1
    assert "webapp_staged_request" in action_categories
    assert "pipeline_gpu_handoff" in action_categories
    assert (
        "run_packet:scene_asset_acquisition_blocker:worldlabs_request_manifest_missing"
        in result["blockers"]
    )
    assert (
        "run_packet:scene_asset_acquisition_blocker:materialized_scene_asset_missing"
        in result["blockers"]
    )
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "status"
    ] == "blocked"
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "worldlabs_request_manifest_exists"
    ] is False
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_input_status"
    ] == "ready_for_worldlabs_request_inputs"
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "ready_for_worldlabs_request_inputs"
    ] is True
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_status"
    ] == "ready_to_submit_worldlabs_request"
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "ready_to_submit_worldlabs_request"
    ] is True
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "safe_to_submit_before_gpu_spend"
    ] is True
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_requires_env"
    ] == ["WORLDLABS_API_KEY", "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION"]
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_missing_env"
    ] == []
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_required_env_status"
    ] == {
        "WORLDLABS_API_KEY": {"configured": True, "value_redacted": True},
        "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION": {
            "configured": True,
            "required_value": "true",
        },
    }
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_requires_gpu"
    ] is False
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_script_path"
    ].endswith("worldlabs_provider_submission_commands.sh")
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_script_safe_to_run_now"
    ] is True
    assert result["phases"]["run_packet"]["checks"]["scene_asset_acquisition"][
        "provider_submission_script_requires_allow_env"
    ] == "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION"
    webapp_check = result["phases"]["run_packet"]["checks"]["webapp_handoff"]
    assert webapp_check["verification_script_path"].endswith(
        "webapp_handoff_verification_commands.sh"
    )
    assert webapp_check["verification_script_safe_to_run_now"] is True
    assert webapp_check["verification_runs_live_webapp_call"] is False
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL" in webapp_check["verification_missing_env"]
    assert "secret-worldlabs-key" not in json.dumps(result)
    assert "secret-token" not in json.dumps(result)
    assert "scene_asset_acquisition" in result["remediation_plan"]["categories"]
    assert (
        "BLUEPRINT_PREVIEW_PROVIDER=world_labs"
        in result["remediation_plan"]["categories"]["scene_asset_acquisition"][
            "safe_commands"
        ][0]
    )
    assert result["phases"]["run_packet"]["checks"]["blocker_resolution"][
        "action_count"
    ] == result["first_gpu_operator_action_count"]
    assert result["gpu_spend_decision"]["status"] == "do_not_rent_gpu_yet"
    external_inputs = result["first_gpu_external_input_packet"]
    assert "WORLDLABS_API_KEY" in external_inputs["secret_handling"]["secret_input_names"]
    scene_inputs = next(
        item
        for item in external_inputs["missing_inputs"]
        if item["category_id"] == "scene_asset_acquisition"
    )
    assert {item["name"] for item in scene_inputs["required_inputs"]} >= {
        "WORLDLABS_API_KEY",
        "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION",
        "pipeline/worldlabs_world_manifest.json",
    }
    scene_guarded = scene_inputs["guarded_commands"]
    assert scene_guarded[0]["name"] == "worldlabs_provider_submission_commands"
    assert scene_guarded[0]["path"].endswith("worldlabs_provider_submission_commands.sh")
    assert scene_guarded[0]["command"].endswith(
        "worldlabs_provider_submission_commands.sh"
    )
    assert scene_guarded[0]["safe_to_run_now"] is True
    assert scene_guarded[0]["runs_live_provider_call"] is True
    assert (
        scene_guarded[0]["requires_explicit_allow_env"]
        == "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION"
    )
    webapp_env_inputs = next(
        item
        for item in external_inputs["missing_inputs"]
        if item["category_id"] == "webapp_forwarding_env"
    )
    webapp_guarded = webapp_env_inputs["guarded_commands"]
    assert webapp_guarded[0]["name"] == "webapp_handoff_verification_commands"
    assert webapp_guarded[0]["path"].endswith(
        "webapp_handoff_verification_commands.sh"
    )
    assert webapp_guarded[0]["safe_to_run_now"] is True
    assert webapp_guarded[0]["runs_live_webapp_call"] is False
    runtime_inputs = next(
        item
        for item in external_inputs["missing_inputs"]
        if item["category_id"] == "gpu_vm_runtime_preflight"
    )
    runtime_guarded = runtime_inputs["guarded_commands"]
    assert runtime_guarded[0]["name"] == "gpu_vm_runtime_preflight"
    assert runtime_guarded[0]["path"].endswith("gpu_vm_runtime_preflight.sh")
    assert runtime_guarded[0]["runs_owner_simulator_command"] is False
    owner_gate_inputs = next(
        item
        for item in external_inputs["missing_inputs"]
        if item["category_id"] == "owner_gpu_gate"
    )
    owner_guarded = owner_gate_inputs["guarded_commands"]
    assert owner_guarded[0]["name"] == "gpu_vm_commands"
    assert owner_guarded[0]["path"].endswith("gpu_vm_commands.sh")
    assert owner_guarded[0]["safe_to_run_now"] is False
    assert owner_guarded[0]["runs_owner_simulator_command"] is True
    markdown = Path(external_inputs["markdown_path"]).read_text(encoding="utf-8")
    assert "First GPU External Input Packet" in markdown
    assert "WORLDLABS_API_KEY" in markdown
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN" in markdown
    assert "worldlabs_provider_submission_commands.sh" in markdown
    assert "webapp_handoff_verification_commands.sh" in markdown
    assert "gpu_vm_runtime_preflight.sh" in markdown
    assert "gpu_vm_commands.sh" in markdown
    assert "Default smoke policy: `walk_to_target`" in markdown
    assert "`live_robot_team_policy_execution`" in markdown
    assert "`real_robot_pov_evidence`" in markdown
    assert "live_policy_execution_contract.md" in markdown
    assert "default_test_robot_eval_job_request.template.json" in markdown
    assert "real_robot_pov_manifest.template.json" in markdown
    assert "stage_first_gpu_live_inputs.sh" in markdown
    assert "BLUEPRINT_ALLOW_STAGING_FIRST_GPU_LIVE_INPUTS=true" in markdown
    assert "secret-token" not in markdown
    assert "secret-worldlabs-key" not in markdown


def test_cross_repo_first_gpu_spend_blocks_when_run_packet_sync_is_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _make_runtime_ready(capture_root, monkeypatch)
    _write_scene_asset_artifacts(capture_root)
    (capture_root / "pipeline" / "simulation_automation" / "scene_frame_estimate.json").unlink()

    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
    )

    assert result["phases"]["runtime_capture"]["ready"] is True
    assert result["phases"]["run_packet"]["ready"] is False
    assert result["status"] == "blocked"
    assert "run_packet:gpu_vm_sync_manifest_not_ready" in result["blockers"]
    assert (
        "run_packet:gpu_vm_sync_manifest_blocker:missing_required_sync_file:scene_frame_estimate"
        in result["blockers"]
    )
    assert (
        "run_packet:gpu_vm_runtime_preflight_plan_blocks_vm_preflight"
        in result["blockers"]
    )
    assert result["gpu_spend_decision"]["status"] == "do_not_rent_gpu_yet"


def test_cross_repo_first_gpu_spend_blocks_when_live_policy_contract_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _make_runtime_ready(capture_root, monkeypatch)
    _write_scene_asset_artifacts(capture_root)

    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )
    (
        capture_root
        / "pipeline"
        / "first_gpu_e2e_run_packet"
        / "live_policy_execution_contract.md"
    ).unlink()

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
    )

    assert result["status"] == "blocked"
    assert "run_packet:live_policy_execution_contract_missing" in result["blockers"]
    packet_check = result["phases"]["run_packet"]["checks"]["first_gpu_run_packet"]
    assert packet_check["live_policy_execution_contract_exists"] is False
    assert result["first_gpu_external_input_packet"]["first_gpu_proof_scope"][
        "contract_artifacts"
    ]["live_policy_execution_contract_exists"] is False
    assert result["gpu_spend_decision"]["status"] == "do_not_rent_gpu_yet"


def test_cross_repo_first_gpu_spend_blocks_when_live_input_staging_script_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _make_runtime_ready(capture_root, monkeypatch)
    _write_scene_asset_artifacts(capture_root)

    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )
    (
        capture_root
        / "pipeline"
        / "first_gpu_e2e_run_packet"
        / "stage_first_gpu_live_inputs.sh"
    ).unlink()

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
    )

    assert result["status"] == "blocked"
    assert "run_packet:live_input_staging_commands_missing" in result["blockers"]
    packet_check = result["phases"]["run_packet"]["checks"]["first_gpu_run_packet"]
    assert packet_check["live_input_staging_commands_exists"] is False
    assert result["first_gpu_external_input_packet"]["first_gpu_proof_scope"][
        "live_input_templates"
    ]["live_input_staging_script_exists"] is False
    assert result["gpu_spend_decision"]["status"] == "do_not_rent_gpu_yet"


def test_cross_repo_first_gpu_spend_blocks_when_gpu_vm_preflight_result_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _make_runtime_ready(capture_root, monkeypatch)
    _write_scene_asset_artifacts(capture_root)

    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
    )

    assert result["status"] == "blocked"
    assert "run_packet:gpu_vm_runtime_preflight_result_not_ready" in result["blockers"]
    assert (
        "run_packet:gpu_vm_runtime_preflight_result:"
        "gpu_vm_runtime_preflight_result_missing"
    ) in result["blockers"]
    runtime_check = result["phases"]["run_packet"]["checks"][
        "gpu_vm_runtime_preflight_plan"
    ]
    assert runtime_check["safe_to_run_on_gpu_vm"] is True
    assert runtime_check["result_ready_for_owner_command_attempt"] is False
    assert runtime_check["result"]["exists"] is False
    external_inputs = result["first_gpu_external_input_packet"]
    missing_categories = {item["category_id"] for item in external_inputs["missing_inputs"]}
    assert "gpu_vm_runtime_preflight" in missing_categories
    assert result["gpu_spend_decision"]["status"] == "do_not_rent_gpu_yet"


def test_cross_repo_first_gpu_spend_allows_ready_runtime_and_ready_run_packet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _make_runtime_ready(capture_root, monkeypatch)
    _write_scene_asset_artifacts(capture_root)

    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )
    _write_gpu_vm_runtime_preflight_result(capture_root)
    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
    )

    assert result["status"] == "ready_for_owner_gpu_attempt"
    assert result["ready_for_owner_gpu_attempt"] is True
    assert result["local_webapp_rehearsal_only_observed"] is False
    assert result["full_e2e_webapp_live_forwarding_required_evidence_present"] is True
    assert result["blockers"] == []
    assert result["phases"]["runtime_capture"]["ready"] is True
    assert result["phases"]["run_packet"]["ready"] is True
    assert result["phases"]["run_packet"]["checks"]["launch_order"][
        "gpu_execution_allowed"
    ] is True
    assert result["phases"]["run_packet"]["checks"]["gpu_vm_runtime_preflight_plan"][
        "result_ready_for_owner_command_attempt"
    ] is True
    assert result["first_gpu_operator_action_count"] == 0
    assert result["first_gpu_operator_actions"] == []
    spend_decision = result["gpu_spend_decision"]
    assert spend_decision["status"] == "ready_to_rent_gpu_vm_for_owner_attempt"
    assert spend_decision["gpu_rental_recommended_now"] is True
    assert spend_decision["local_webapp_rehearsal_only_observed"] is False
    assert spend_decision[
        "full_e2e_webapp_live_forwarding_required_evidence_present"
    ] is True
    assert spend_decision["must_not_do_until_ready"] == []
    external_inputs = result["first_gpu_external_input_packet"]
    assert external_inputs["status"] == "ready"
    assert external_inputs["missing_input_category_count"] == 0
    assert external_inputs["missing_input_count"] == 0
    assert external_inputs["gpu_rental_recommended_now"] is True


def test_cross_repo_first_gpu_spend_blocks_local_webapp_rehearsal_even_when_allowed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root, local_rehearsal=True)
    _write_scene_asset_artifacts(capture_root)
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
        allow_local_webapp_rehearsal=True,
    )
    _write_gpu_vm_runtime_preflight_result(capture_root)
    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
        allow_local_webapp_rehearsal=True,
    )

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=_pipeline_repo(tmp_path),
        capture_repo=_capture_repo(tmp_path),
        webapp_repo=_webapp_repo(tmp_path),
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        simulator_command_location="remote",
        allow_local_webapp_rehearsal=True,
    )

    assert result["phases"]["runtime_capture"]["ready"] is True
    assert result["phases"]["run_packet"]["ready"] is True
    assert result["local_webapp_rehearsal_only_observed"] is True
    assert result["full_e2e_webapp_live_forwarding_required_evidence_present"] is False
    assert result["status"] == "blocked"
    assert result["ready_for_owner_gpu_attempt"] is False
    assert result["blockers"] == [
        (
            "full_e2e_webapp_live_forwarding:"
            "local_webapp_rehearsal_not_live_forwarding_proof"
        )
    ]
    assert "webapp_live_forwarding_proof" in result["remediation_plan"]["categories"]
    spend_decision = result["gpu_spend_decision"]
    assert spend_decision["status"] == "do_not_rent_gpu_yet"
    assert spend_decision["gpu_rental_recommended_now"] is False
    assert spend_decision["local_webapp_rehearsal_only_observed"] is True
    assert spend_decision[
        "full_e2e_webapp_live_forwarding_required_evidence_present"
    ] is False
    assert "webapp_live_forwarding_proof" in spend_decision["pre_spend_blocker_categories"]
    external_inputs = result["first_gpu_external_input_packet"]
    assert external_inputs["status"] == "blocked"
    assert external_inputs["next_missing_category_id"] == "webapp_live_forwarding_proof"
    assert external_inputs["gpu_rental_recommended_now"] is False
    assert external_inputs["missing_inputs"][0]["required_inputs"][0]["name"] == (
        "non_rehearsal_webapp_staged_request"
    )


def test_cross_repo_first_gpu_readiness_cli_writes_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "cross-repo.json"

    exit_code = main(
        [
            "--pipeline-repo",
            str(_pipeline_repo(tmp_path)),
            "--capture-repo",
            str(_capture_repo(tmp_path)),
            "--webapp-repo",
            str(_webapp_repo(tmp_path)),
            "--output",
            str(output),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 1
    assert "gpu_spend_decision=do_not_rent_gpu_yet" in stdout
    assert "gpu_rental_recommended_now=False" in stdout
    assert "external_input_packet_status=blocked" in stdout
    assert "next_missing_category=sample_capture" in stdout
    assert "external_input_packet_markdown=" in stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CROSS_REPO_FIRST_GPU_READINESS_SCHEMA_VERSION
    assert payload["provider_guidance"]["selected_provisioner"] == "runpod"
    assert payload["gpu_spend_decision"]["status"] == "do_not_rent_gpu_yet"
    assert payload["first_gpu_external_input_packet"]["status"] == "blocked"
    markdown_path = Path(payload["first_gpu_external_input_packet"]["markdown_path"])
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown_path.is_file()
    assert "First GPU External Input Packet" in markdown
    assert "Missing Inputs" in markdown
    assert "Secret input names" in markdown
    assert "secret-token" not in markdown


def _write_run_packet_required_jsons(
    packet_dir: Path,
    *,
    packet: dict[str, object] | None = None,
    blocker_resolution: dict[str, object] | None = None,
) -> None:
    _write_json(
        packet_dir / "first_gpu_run_packet.json",
        packet if packet is not None else {"ready_for_first_gpu_attempt": True},
    )
    _write_json(
        packet_dir / "first_gpu_blocker_resolution.json",
        blocker_resolution if blocker_resolution is not None else {},
    )
    for name in (
        "first_gpu_webapp_handoff.json",
        "first_gpu_scene_asset_acquisition.json",
        "first_gpu_launch_order.json",
        "gpu_vm_runtime_preflight_plan.json",
        "gpu_vm_sync_manifest.json",
    ):
        _write_json(packet_dir / name, {})


def test_cross_repo_helpers_cover_invalid_runtime_inputs(tmp_path: Path) -> None:
    latin_path = tmp_path / "latin.txt"
    latin_path.write_bytes(b"hello \xff world")
    assert _read_text(latin_path) == "hello  world"

    assert _runtime_preflight_result_summary(None)["blockers"] == [
        "gpu_vm_runtime_preflight_result_path_missing"
    ]
    assert _runtime_preflight_result_summary(tmp_path / "missing.json")["blockers"] == [
        "gpu_vm_runtime_preflight_result_missing"
    ]

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not-json", encoding="utf-8")
    assert _runtime_preflight_result_summary(invalid_json)["blockers"] == [
        "invalid_json:invalid.json:JSONDecodeError"
    ]

    invalid_payload = tmp_path / "invalid-payload.json"
    invalid_payload.write_text("[]", encoding="utf-8")
    assert _runtime_preflight_result_summary(invalid_payload)["blockers"] == [
        "invalid_json_payload:invalid-payload.json:list"
    ]

    blocked_result = tmp_path / "blocked-result.json"
    _write_json(
        blocked_result,
        {
            "status": "blocked",
            "blockers": ["nvidia_smi_missing", "owner_command_missing"],
        },
    )
    blocked_summary = _runtime_preflight_result_summary(blocked_result)
    assert blocked_summary["ready_for_owner_command_attempt"] is False
    assert blocked_summary["blockers"] == [
        "gpu_vm_runtime_preflight_result_blocker:nvidia_smi_missing",
        "gpu_vm_runtime_preflight_result_blocker:owner_command_missing",
        "gpu_vm_runtime_preflight_result_status:blocked",
    ]

    missing_file = _file_contains_check(
        tmp_path,
        "contracts/missing.txt",
        required=(("needle", "required text"),),
    )
    assert missing_file == {
        "path": str(tmp_path / "contracts/missing.txt"),
        "exists": False,
        "ready": False,
        "matched": {"needle": False},
        "blockers": ["missing_file:contracts/missing.txt"],
    }


def test_cross_repo_runtime_rehearsal_helper_handles_non_mapping_stages() -> None:
    assert _runtime_has_local_webapp_rehearsal({"readiness": {"stages": []}}) is False


def test_cross_repo_run_packet_phase_reports_fallback_generated_file_gaps(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    packet_dir = capture_root / "pipeline" / "first_gpu_e2e_run_packet"
    _write_run_packet_required_jsons(
        packet_dir,
        packet={"ready_for_first_gpu_attempt": True, "generated_files": {}},
    )

    result = _run_packet_phase(capture_root=capture_root)

    packet_check = result["checks"]["first_gpu_run_packet"]
    assert packet_check["owner_command_binding_template_path"].endswith(
        "owner_default_smoke_command_binding.sh"
    )
    assert packet_check["live_policy_execution_contract_path"].endswith(
        "live_policy_execution_contract.md"
    )
    assert packet_check["default_test_robot_eval_job_request_template_path"].endswith(
        "default_test_robot_eval_job_request.template.json"
    )
    assert set(result["blockers"]) >= {
        "owner_command_binding_template_missing",
        "live_policy_execution_contract_missing",
        "default_test_robot_eval_job_request_template_missing",
        "real_robot_pov_manifest_template_missing",
        "live_input_staging_commands_missing",
    }


def test_cross_repo_run_packet_phase_reports_blocker_resolution_mismatches(
    tmp_path: Path,
) -> None:
    mismatch_capture = tmp_path / "mismatch-capture"
    _write_run_packet_required_jsons(
        mismatch_capture / "pipeline" / "first_gpu_e2e_run_packet",
        packet={"ready_for_first_gpu_attempt": True, "generated_files": {}},
        blocker_resolution={
            "action_count": 2,
            "blocked_action_count": 0,
            "actions": [{"category_id": "first_gpu_run_packet"}],
        },
    )

    mismatch = _run_packet_phase(capture_root=mismatch_capture)

    assert "blocker_resolution_action_count_mismatch" in mismatch["blockers"]

    no_actions_capture = tmp_path / "no-actions-capture"
    _write_run_packet_required_jsons(
        no_actions_capture / "pipeline" / "first_gpu_e2e_run_packet",
        packet={"ready_for_first_gpu_attempt": False, "generated_files": {}},
        blocker_resolution={"action_count": 0, "blocked_action_count": 0, "actions": []},
    )

    no_actions = _run_packet_phase(capture_root=no_actions_capture)

    assert "blocker_resolution_missing_actions_for_blocked_packet" in no_actions[
        "blockers"
    ]


def test_cross_repo_remediation_maps_remaining_blocker_categories() -> None:
    expectations = {
        "webapp_forwarding_preflight:blocked": "webapp_forwarding_env",
        "webapp_staged_inputs_local_rehearsal_only": "webapp_staged_request",
        "webapp_handoff_blocker:missing_upstream_truth": "webapp_handoff_packet",
        "spawn_validation_blocked": "scene_spawn_preflight",
        "owner_command_binding_template_missing": "owner_gpu_command",
        "missing_simulator_command": "owner_gpu_command",
        "missing_local_scene_asset": "scene_spawn_preflight",
        "missing_scene_frame_estimate": "scene_spawn_preflight",
    }

    for blocker, category in expectations.items():
        assert _remediation_for_blocker(blocker)["category"] == category


def test_cross_repo_remediation_plan_preserves_unknown_custom_categories(
    monkeypatch,
) -> None:
    def custom_remediation(blocker: str) -> dict[str, object]:
        return {
            "blocker": blocker,
            "category": "custom_operator_step",
            "next_action": "Do the custom step.",
            "evidence_required": "The custom evidence exists.",
            "safe_command": "custom-command",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.cross_repo_first_gpu_readiness._remediation_for_blocker",
        custom_remediation,
    )

    plan = _build_remediation_plan(["custom:blocker"])

    assert list(plan["categories"]) == ["custom_operator_step"]
    assert plan["categories"]["custom_operator_step"]["safe_commands"] == [
        "custom-command"
    ]


def test_cross_repo_gpu_spend_decision_adds_live_forwarding_category_for_rehearsal() -> None:
    decision = _build_gpu_spend_decision(
        blockers=[],
        remediation_plan={"categories": {}},
        runtime_phase={
            "ready": True,
            "readiness": {
                "stages": {
                    "webapp_staged_request": {"local_rehearsal_only": True},
                },
            },
        },
        simulator="isaac_sim",
        provisioner="runpod",
    )

    assert decision["status"] == "do_not_rent_gpu_yet"
    assert decision["gpu_rental_recommended_now"] is False
    assert decision["pre_spend_blocker_categories"] == ["webapp_live_forwarding_proof"]


def test_cross_repo_guarded_commands_use_fallback_packet_paths(tmp_path: Path) -> None:
    assert _guarded_commands_by_category(None) == {}

    packet_dir = tmp_path / "packet"
    upstream = packet_dir / "webapp_upstream_truth_verification_commands.sh"
    _write(upstream, "#!/usr/bin/env bash\n")

    guarded = _guarded_commands_by_category({"packet_dir": str(packet_dir), "checks": {}})

    upstream_command = guarded["webapp_upstream_truth"][0]
    assert upstream_command["path"] == str(upstream)
    assert upstream_command["safe_to_run_now"] is True


def test_cross_repo_external_input_packet_skips_non_mapping_categories() -> None:
    packet = _build_first_gpu_external_input_packet(
        capture_root=None,
        webapp_site_slug="site-1",
        simulator="isaac_sim",
        provisioner="runpod",
        remediation_plan={
            "categories": {
                "not-a-category": "skip-me",
                "owner_gpu_gate": {
                    "blocker_count": 1,
                    "blockers": ["missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"],
                    "next_actions": ["set the gate"],
                    "evidence_required": ["gate is true"],
                    "safe_commands": ["export BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true"],
                },
            },
        },
        gpu_spend_decision={
            "gpu_rental_recommended_now": False,
            "must_not_do_until_ready": ["do_not_allocate_runpod_or_equivalent_gpu_vm"],
        },
        run_packet_phase=None,
    )

    assert packet["missing_input_category_count"] == 1
    assert packet["missing_inputs"][0]["category_id"] == "owner_gpu_gate"


def test_cross_repo_external_input_markdown_reports_no_missing_inputs() -> None:
    markdown = _first_gpu_external_input_packet_markdown(
        {
            "schema_version": "first_gpu_external_input_packet.v1",
            "status": "ready",
            "generated_at": "2026-06-21T00:00:00Z",
            "gpu_rental_recommended_now": True,
            "selected_simulator": "isaac_sim",
            "selected_provisioner": "runpod",
            "missing_input_category_count": 0,
            "missing_input_count": 0,
            "secret_handling": {
                "secrets_are_named_but_values_are_not_serialized": True,
                "secret_input_names": [],
            },
            "forbidden_actions_until_ready": [],
            "first_gpu_proof_scope": {},
            "missing_inputs": [],
        }
    )

    assert "## Missing Inputs" in markdown
    assert "- None." in markdown


def test_cross_repo_first_gpu_readiness_cli_returns_zero_for_ready_packet(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    capture_root = _runtime_capture_root(tmp_path)
    _make_runtime_ready(capture_root, monkeypatch)
    _write_scene_asset_artifacts(capture_root)
    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )
    _write_gpu_vm_runtime_preflight_result(capture_root)
    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        owner_command_location="remote",
    )
    output = tmp_path / "ready-cross-repo.json"

    exit_code = main(
        [
            "--pipeline-repo",
            str(_pipeline_repo(tmp_path)),
            "--capture-repo",
            str(_capture_repo(tmp_path)),
            "--webapp-repo",
            str(_webapp_repo(tmp_path)),
            "--capture-root",
            str(capture_root),
            "--webapp-site-slug",
            "site-1",
            "--simulator-command",
            "/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
            "--simulator-command-location",
            "remote",
            "--output",
            str(output),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "status=ready_for_owner_gpu_attempt" in stdout
    assert "gpu_rental_recommended_now=True" in stdout
    assert "next_missing_category=" not in stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["blockers"] == []
