from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.cross_repo_first_gpu_readiness import (
    CROSS_REPO_FIRST_GPU_READINESS_SCHEMA_VERSION,
    build_cross_repo_first_gpu_readiness,
    main,
)
from blueprint_pipeline.first_gpu_run_packet import build_first_gpu_run_packet


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload, indent=2))


def _pipeline_repo(tmp_path: Path) -> Path:
    root = tmp_path / "BlueprintCapturePipeline"
    _write(
        root / "pyproject.toml",
        "\n".join(
            [
                "blueprint-audit-first-gpu-e2e-readiness",
                "blueprint-stage-first-gpu-sample-video",
                "blueprint-run-owner-gpu-proof",
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
