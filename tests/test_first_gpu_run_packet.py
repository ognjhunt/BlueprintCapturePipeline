from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.first_gpu_run_packet import (
    FIRST_GPU_BLOCKER_RESOLUTION_SCHEMA_VERSION,
    FIRST_GPU_LAUNCH_ORDER_SCHEMA_VERSION,
    FIRST_GPU_RUN_PACKET_SCHEMA_VERSION,
    FIRST_GPU_SCENE_ASSET_ACQUISITION_SCHEMA_VERSION,
    FIRST_GPU_SIMULATOR_PATH_MATRIX_SCHEMA_VERSION,
    FIRST_GPU_VM_RUNTIME_PREFLIGHT_PLAN_SCHEMA_VERSION,
    FIRST_GPU_VM_SYNC_SCHEMA_VERSION,
    FIRST_GPU_WEBAPP_HANDOFF_SCHEMA_VERSION,
    build_first_gpu_run_packet,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_gpu_vm_runtime_preflight_result(packet_dir: Path) -> None:
    _write_json(
        packet_dir / "gpu_vm_runtime_preflight_result.json",
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
                "robot_readiness_proven": False,
            },
        },
    )


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
            "capture_capabilities": {"camera_pose": True},
            "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
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
            "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
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


def _write_blocked_gpu_handoff_with_details(capture_root: Path) -> None:
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    details = [
        {
            "blocker_id": "missing_local_scene_asset",
            "source_artifact": "scene_asset_preflight.json",
            "severity": "hard_pre_gpu_blocker",
            "required_input": "Provide a local materialized scene asset.",
            "proof_boundary": "required input only; does not prove simulator execution or robot readiness",
            "safe_next_command": "blueprint-run-simulation-automation --capture-root <capture-root>",
        },
        {
            "blocker_id": "missing_scene_frame_estimate",
            "source_artifact": "scene_frame_estimate.json",
            "severity": "hard_pre_gpu_blocker",
            "required_input": "Generate finite scene bounds before GPU execution.",
            "proof_boundary": "required input only; does not prove simulator execution or robot readiness",
            "safe_next_command": "blueprint-run-simulation-automation --capture-root <capture-root>",
        },
        {
            "blocker_id": "portable_collider_glb_missing",
            "source_artifact": "scene_asset_preflight.json",
            "severity": "review_or_backend_selection_blocker",
            "required_input": "Review collision assets if contact confidence is required.",
            "proof_boundary": "required input only; does not prove simulator execution or robot readiness",
        },
    ]
    _write_json(
        automation_dir / "gpu_handoff_packet.json",
        {
            "schema_version": "gpu_handoff_packet.v1",
            "status": "blocked_for_owner_gpu_preflight_handoff",
            "ready_for_owner_gpu_preflight": False,
            "owner_gpu_simulator_execution_proven": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["owner_gpu_simulator_execution_not_run", "spawn_validation_blocked"],
            "hard_preflight_blockers": [
                "missing_local_scene_asset",
                "missing_scene_frame_estimate",
            ],
            "pre_gpu_blocker_details": details,
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
            "pre_gpu_blocker_details": details,
        },
    )
    _write_json(
        automation_dir / "simulator_engine_plugin_registry.json",
        {"schema_version": "simulator_engine_plugin_registry.v1", "status": "ready"},
    )
    (automation_dir / "gpu_run_checklist.md").write_text("# GPU checklist\n", encoding="utf-8")


def _write_staged_webapp_request(capture_root: Path) -> Path:
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
    _write_json(
        request_path,
        {
            "queue_contract": "robot_eval_job_request_inbox.v1",
            "status": "queued_for_pipeline",
            "job_id": "webapp-job-1",
            "job_request": request,
        },
    )
    staged_path = capture_root / "pipeline" / "live_pipeline_staged_inputs.json"
    _write_json(
        staged_path,
        {
            "schema_version": "blueprint_live_pipeline_staged_inputs.v1",
            "configured_capture_root": str(capture_root.resolve()),
            "webapp_request": {
                "ready": True,
                "staged": True,
                "job_id": "webapp-job-1",
                "path": str(request_path),
                "target_path": str(request_path),
            },
        },
    )
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
                "no_robot_readiness_proven": True,
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


def test_first_gpu_run_packet_writes_command_and_env_files(tmp_path: Path, monkeypatch) -> None:
    capture_root = _capture_root(tmp_path)
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "real-secret-token")

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh",
        output_dir=tmp_path / "packet",
    )

    packet_path = Path(result["packet_path"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    env_example = Path(packet["generated_files"]["env_example"]).read_text(encoding="utf-8")
    local_commands = Path(packet["generated_files"]["local_preflight_commands"]).read_text(
        encoding="utf-8"
    )
    worldlabs_provider_submission_commands = Path(
        packet["generated_files"]["worldlabs_provider_submission_commands"]
    ).read_text(encoding="utf-8")
    webapp_upstream_truth_verification_commands = Path(
        packet["generated_files"]["webapp_upstream_truth_verification_commands"]
    ).read_text(encoding="utf-8")
    webapp_handoff_verification_commands = Path(
        packet["generated_files"]["webapp_handoff_verification_commands"]
    ).read_text(encoding="utf-8")
    gpu_commands = Path(packet["generated_files"]["gpu_vm_commands"]).read_text(encoding="utf-8")
    gpu_vm_runtime_preflight_script = Path(
        packet["generated_files"]["gpu_vm_runtime_preflight_script"]
    ).read_text(encoding="utf-8")
    gpu_vm_runtime_preflight_plan = json.loads(
        Path(packet["generated_files"]["gpu_vm_runtime_preflight_plan"]).read_text(
            encoding="utf-8"
        )
    )
    gpu_vm_runtime_preflight_markdown = Path(
        packet["generated_files"]["gpu_vm_runtime_preflight_markdown"]
    ).read_text(encoding="utf-8")
    simulator_path_matrix = json.loads(
        Path(packet["generated_files"]["simulator_path_matrix"]).read_text(encoding="utf-8")
    )
    simulator_path_matrix_markdown = Path(
        packet["generated_files"]["simulator_path_matrix_markdown"]
    ).read_text(encoding="utf-8")
    launch_order = json.loads(
        Path(packet["generated_files"]["launch_order"]).read_text(encoding="utf-8")
    )
    launch_order_markdown = Path(packet["generated_files"]["launch_order_markdown"]).read_text(
        encoding="utf-8"
    )
    owner_contract = Path(packet["generated_files"]["owner_command_contract"]).read_text(
        encoding="utf-8"
    )
    owner_command_binding_template = Path(
        packet["generated_files"]["owner_command_binding_template"]
    ).read_text(encoding="utf-8")
    isaac_smoke_script = Path(
        packet["generated_files"]["isaac_unitree_g1_smoke_script"]
    ).read_text(encoding="utf-8")
    isaac_smoke_launcher = Path(
        packet["generated_files"]["isaac_unitree_g1_smoke_launcher"]
    ).read_text(encoding="utf-8")
    live_policy_execution_contract = Path(
        packet["generated_files"]["live_policy_execution_contract"]
    ).read_text(encoding="utf-8")
    default_test_job_request_template = json.loads(
        Path(
            packet["generated_files"]["default_test_robot_eval_job_request_template"]
        ).read_text(encoding="utf-8")
    )
    real_robot_pov_manifest_template = json.loads(
        Path(packet["generated_files"]["real_robot_pov_manifest_template"]).read_text(
            encoding="utf-8"
        )
    )
    live_input_staging_commands = Path(
        packet["generated_files"]["live_input_staging_commands"]
    ).read_text(encoding="utf-8")
    provider_bootstrap = Path(packet["generated_files"]["gpu_provider_bootstrap"]).read_text(
        encoding="utf-8"
    )
    blocker_resolution_markdown = Path(
        packet["generated_files"]["blocker_resolution_markdown"]
    ).read_text(encoding="utf-8")
    provider_bootstrap_manifest = json.loads(
        Path(packet["generated_files"]["gpu_provider_bootstrap_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    blocker_resolution = json.loads(
        Path(packet["generated_files"]["blocker_resolution"]).read_text(encoding="utf-8")
    )
    scene_asset_acquisition = json.loads(
        Path(packet["generated_files"]["scene_asset_acquisition"]).read_text(encoding="utf-8")
    )
    scene_asset_acquisition_markdown = Path(
        packet["generated_files"]["scene_asset_acquisition_markdown"]
    ).read_text(encoding="utf-8")
    webapp_handoff = json.loads(
        Path(packet["generated_files"]["webapp_handoff"]).read_text(encoding="utf-8")
    )
    webapp_handoff_markdown = Path(packet["generated_files"]["webapp_handoff_markdown"]).read_text(
        encoding="utf-8"
    )
    vm_sync_manifest = json.loads(
        Path(packet["generated_files"]["gpu_vm_sync_manifest"]).read_text(encoding="utf-8")
    )
    vm_sync_markdown = Path(packet["generated_files"]["gpu_vm_sync_markdown"]).read_text(
        encoding="utf-8"
    )

    assert packet["schema_version"] == FIRST_GPU_RUN_PACKET_SCHEMA_VERSION
    assert packet["readiness_status"] == "blocked"
    assert packet["owner_command_supplied"] is True
    assert packet["owner_command_location"] == "remote"
    assert "simulator_runtime:simulator_command_executable_missing" not in packet["blockers"]
    assert "real-secret-token" not in env_example
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN='<set-in-shell-not-in-file>'" in env_example
    assert "WORLDLABS_API_KEY='<set-in-shell-not-in-file>'" in env_example
    assert "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION=false" in env_example
    assert "export PACKET_DIR=" in env_example
    assert "export OWNER_RAW_SIMULATOR_COMMAND=/opt/blueprint/run_isaac_gpu_proof.sh" in (
        env_example
    )
    assert 'export OWNER_DEFAULT_SMOKE_COMMAND_BINDING="$PACKET_DIR/owner_default_smoke_command_binding.sh"' in (
        env_example
    )
    assert "export ISAAC_SMOKE_SCRIPT=\"$PACKET_DIR/isaac_unitree_g1_smoke.py\"" in (
        env_example
    )
    assert (
        "export ISAAC_UNITREE_G1_SMOKE_COMMAND=\"bash $PACKET_DIR/run_isaac_unitree_g1_smoke.sh\""
        in env_example
    )
    assert "export BLUEPRINT_USE_DEFAULT_SMOKE_BINDING=false" in env_example
    assert 'export OWNER_SIMULATOR_COMMAND="$ISAAC_UNITREE_G1_SMOKE_COMMAND"' in (
        env_example
    )
    assert "OWNER_SCENE_LOAD_COMMAND='<command-that-loads-scene-and-writes-BLUEPRINT_SCENE_LOAD_TRACE>'" in (
        env_example
    )
    assert "OWNER_ROBOT_SPAWN_COMMAND='<command-that-spawns-robot-and-writes-BLUEPRINT_SPAWN_TRACE>'" in (
        env_example
    )
    assert "OWNER_WALK_TO_TARGET_COMMAND='<command-that-runs-default-walk-to-target-policy>'" in (
        env_example
    )
    assert "SIM_ROBOT_POV_FRAME_PATH='<simulator-pov-frame-path>'" in env_example
    assert "blueprint-audit-first-gpu-e2e-readiness" in local_commands
    assert "\\\n\n  --simulator-command" not in local_commands
    assert "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION=true" in (
        worldlabs_provider_submission_commands
    )
    assert "WORLDLABS_API_KEY must be set" in worldlabs_provider_submission_commands
    assert "blueprint-run-e2e" in worldlabs_provider_submission_commands
    assert "source_video_preflight_not_ready" in worldlabs_provider_submission_commands
    assert "first_gpu_webapp_upstream_truth_verification_result.v1" in (
        webapp_upstream_truth_verification_commands
    )
    assert "missing_or_placeholder_webapp_" in webapp_upstream_truth_verification_commands
    assert '"artifacts_mutated": False' in webapp_upstream_truth_verification_commands
    assert "robot_eval_job_request.v1 owner_system" in (
        webapp_upstream_truth_verification_commands
    )
    assert "first_gpu_webapp_handoff_verification_result.v1" in (
        webapp_handoff_verification_commands
    )
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN must be set" in (
        webapp_handoff_verification_commands
    )
    assert "webapp_request_missing_required_upstream_ids" in (
        webapp_handoff_verification_commands
    )
    assert "real-secret-token" not in webapp_handoff_verification_commands
    assert "blueprint-run-owner-gpu-proof" in gpu_commands
    assert "--simulator-backend isaac_sim" in gpu_commands
    assert "OWNER_DEFAULT_SMOKE_COMMAND_BINDING" in gpu_commands
    assert "BLUEPRINT_USE_DEFAULT_SMOKE_BINDING" in gpu_commands
    assert "ISAAC_UNITREE_G1_SMOKE_COMMAND" in gpu_commands
    assert 'OWNER_SIMULATOR_COMMAND="$ISAAC_UNITREE_G1_SMOKE_COMMAND"' in (
        gpu_commands
    )
    assert "nvidia-smi" in gpu_vm_runtime_preflight_script
    assert "isaac_driver_below_minimum" in gpu_vm_runtime_preflight_script
    assert "vulkaninfo" in gpu_vm_runtime_preflight_script
    assert "BLUEPRINT_ISAAC_MIN_DRIVER_VERSION" in gpu_vm_runtime_preflight_script
    assert "gpu_vm_runtime_preflight_result.json" in gpu_vm_runtime_preflight_script
    assert "ISAAC_UNITREE_G1_SMOKE_COMMAND" in gpu_vm_runtime_preflight_script
    assert 'OWNER_SIMULATOR_COMMAND="$ISAAC_UNITREE_G1_SMOKE_COMMAND"' in (
        gpu_vm_runtime_preflight_script
    )
    assert "sha256_mismatch:" in gpu_vm_runtime_preflight_script
    assert "sync_manifest_blocker:" in gpu_vm_runtime_preflight_script
    assert any(
        "driver >= 580.65.06" in item and "Vulkan" in item
        for item in gpu_vm_runtime_preflight_plan["inputs_checked_when_script_runs"]
    )
    assert gpu_vm_runtime_preflight_plan["schema_version"] == (
        FIRST_GPU_VM_RUNTIME_PREFLIGHT_PLAN_SCHEMA_VERSION
    )
    assert gpu_vm_runtime_preflight_plan["status"] == "blocked_for_owner_gpu_attempt"
    assert "pipeline_gpu_handoff:missing_artifact:gpu_handoff_packet" in (
        gpu_vm_runtime_preflight_plan["hard_stop_blockers"]
    )
    assert gpu_vm_runtime_preflight_plan["script"]["runs_owner_simulator_command"] is False
    assert (
        gpu_vm_runtime_preflight_plan["claim_boundary"]["owner_simulator_command_executed"]
        is False
    )
    assert gpu_vm_runtime_preflight_plan["claim_boundary"]["robot_readiness_proven"] is False
    assert "GPU VM Runtime Preflight" in gpu_vm_runtime_preflight_markdown
    assert simulator_path_matrix["schema_version"] == (
        FIRST_GPU_SIMULATOR_PATH_MATRIX_SCHEMA_VERSION
    )
    assert simulator_path_matrix["selected_simulator"] == "isaac_sim"
    assert simulator_path_matrix["status"] == "blocked_for_selected_simulator_attempt"
    assert simulator_path_matrix["nvidia_nim_boundary"]["primary_simulator_runtime"] is False
    assert simulator_path_matrix["first_gpu_recommendation"]["recommended_first_path"] == (
        "isaac_sim"
    )
    paths = {item["framework"]: item for item in simulator_path_matrix["paths"]}
    assert paths["isaac_sim"]["selected_for_this_packet"] is True
    assert paths["isaac_sim"]["recommended_first_gpu_smoke"] is True
    assert paths["mujoco"]["can_run_without_gpu_preflight"] is True
    assert paths["pybullet"]["can_run_without_gpu_preflight"] is True
    assert paths["isaac_lab_arena"]["recommended_first_gpu_smoke"] is False
    assert simulator_path_matrix["claim_boundary"]["nvidia_nim_used_as_simulator"] is False
    assert simulator_path_matrix["claim_boundary"]["robot_readiness_proven"] is False
    assert "Simulator Path Matrix" in simulator_path_matrix_markdown
    assert "NVIDIA NIM Boundary" in simulator_path_matrix_markdown
    assert launch_order["schema_version"] == FIRST_GPU_LAUNCH_ORDER_SCHEMA_VERSION
    assert launch_order["status"] == "blocked"
    assert launch_order["gpu_execution_allowed"] is False
    assert "webapp_live_handoff" in launch_order["blocked_step_ids"]
    assert "scene_asset_acquisition" in launch_order["blocked_step_ids"]
    assert "owner_gpu_simulator_proof" in launch_order["blocked_step_ids"]
    assert "do_not_run_gpu_vm_commands" in launch_order["forbidden_actions_until_ready"]
    launch_steps = {item["step_id"]: item for item in launch_order["steps"]}
    assert launch_steps["sample_video_preflight"]["status"] in {"ready", "warning"}
    assert launch_steps["gpu_vm_sync"]["may_run_now"] is False
    assert launch_steps["owner_gpu_simulator_proof"]["may_run_now"] is False
    for step in launch_steps.values():
        assert len(step["blockers"]) == len(set(step["blockers"]))
    assert launch_order["claim_boundary"]["simulator_execution_performed"] is False
    assert launch_order["claim_boundary"]["robot_readiness_proven"] is False
    assert "First GPU Launch Order" in launch_order_markdown
    assert "BLUEPRINT_SCENE_LOAD_TRACE" in owner_contract
    assert "BLUEPRINT_DEFAULT_SMOKE_POLICY" in owner_contract
    assert "BLUEPRINT_POLICY_EXECUTION_TRACE" in owner_contract
    assert "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE" in owner_contract
    assert "walk_to_target" in owner_contract
    assert "blueprint-write-owner-gpu-default-smoke-artifacts" in owner_contract
    assert "owner_default_smoke_command_binding.sh" in owner_contract
    assert packet["generated_files"]["isaac_unitree_g1_smoke_script"].endswith(
        "isaac_unitree_g1_smoke.py"
    )
    assert "OWNER_SCENE_LOAD_COMMAND" in owner_command_binding_template
    assert "OWNER_ROBOT_SPAWN_COMMAND" in owner_command_binding_template
    assert "OWNER_WALK_TO_TARGET_COMMAND" in owner_command_binding_template
    assert "blueprint_pipeline.owner_gpu_default_smoke_artifacts" in (
        owner_command_binding_template
    )
    assert "SIM_ROBOT_POV_FRAME_PATH" in owner_command_binding_template
    assert "SimulationApp" in isaac_smoke_script
    assert "omni.kit.asset_converter" in isaac_smoke_script
    assert "Robots/Unitree/G1/g1.usd" in isaac_smoke_script
    assert "BLUEPRINT_SCENE_LOAD_TRACE" in isaac_smoke_script
    assert "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE" in isaac_smoke_script
    assert "policy_downloaded_from_online" in isaac_smoke_script
    assert "python.sh" in isaac_smoke_launcher
    assert "ISAAC_PYTHON" in isaac_smoke_launcher
    assert "Live Policy Execution Contract" in live_policy_execution_contract
    assert "BLUEPRINT_ALLOW_POLICY_EXECUTION=true" in live_policy_execution_contract
    assert "robot_policy_execution_proven" in live_policy_execution_contract
    assert "default_test_policy_execution_proven" in live_policy_execution_contract
    assert "robot_team_policy_execution_proven" in live_policy_execution_contract
    assert "reference_replayed" in live_policy_execution_contract
    assert "scenario_eval_run_coverage_complete" in live_policy_execution_contract
    assert "default simulator policy execution only" in live_policy_execution_contract
    assert '"default_test_policy"' in live_policy_execution_contract
    assert default_test_job_request_template["schema_version"] == "robot_eval_job_request.v1"
    assert default_test_job_request_template["site_package"]["capture_root"] == str(capture_root)
    assert default_test_job_request_template["site_package"]["site_slug"] == "site-1"
    assert default_test_job_request_template["default_test_policy"] == {
        "policy_kind": "walk_to_target",
        "target": "walk_to_target_pose",
    }
    assert default_test_job_request_template["claim_boundary"][
        "default_test_policy_execution_requested"
    ] is True
    assert default_test_job_request_template["claim_boundary"][
        "robot_team_policy_execution_requested"
    ] is False
    assert real_robot_pov_manifest_template["schema_version"] == "real_robot_pov_manifest.v1"
    assert real_robot_pov_manifest_template["records"][0]["robot_camera_video_uri"] == (
        "<owner-system-robot-camera-video-uri>"
    )
    assert real_robot_pov_manifest_template["records"][0]["action_log_uri"] == (
        "<owner-system-action-log-uri>"
    )
    assert real_robot_pov_manifest_template["claim_boundary"][
        "generated_or_simulator_pov_not_accepted"
    ] is True
    assert "BLUEPRINT_ALLOW_STAGING_FIRST_GPU_LIVE_INPUTS" in live_input_staging_commands
    assert "placeholder values remain" in live_input_staging_commands
    assert "--stage-webapp-request" in live_input_staging_commands
    assert "--stage-real-robot-pov" in live_input_staging_commands
    assert "blueprint-intake-live-pipeline-inputs" in live_input_staging_commands
    assert "DEFAULT_POLICY_TARGET" in gpu_commands
    assert '--default-policy-target "$DEFAULT_POLICY_TARGET"' in gpu_commands
    assert "RunPod Pod or equivalent interactive GPU VM" in provider_bootstrap
    assert "NVIDIA NIM are not the primary simulator runtime" in provider_bootstrap
    assert "Avoid for Isaac Sim first smoke: A100, H100." in provider_bootstrap
    assert "First GPU Blocker Resolution" in blocker_resolution_markdown
    assert "source_video_preflight_manifest_missing" in blocker_resolution_markdown
    assert provider_bootstrap_manifest["gpu_guidance"]["recommended_gpu_class"] == (
        "RTX-class GPU with RT cores"
    )
    assert provider_bootstrap_manifest["nvidia_nim_boundary"]["primary_for_first_smoke"] is False
    assert provider_bootstrap_manifest["owner_command_location"] == "remote"
    assert provider_bootstrap_manifest["claim_boundary"]["gpu_provisioning_performed"] is False
    assert blocker_resolution["schema_version"] == FIRST_GPU_BLOCKER_RESOLUTION_SCHEMA_VERSION
    assert blocker_resolution["claim_boundary"]["robot_readiness_proven"] is False
    assert blocker_resolution["action_count"] == len(blocker_resolution["actions"])
    assert blocker_resolution["action_count"] >= 1
    assert blocker_resolution["blocked_action_count"] >= 1
    action_category_ids = {item["category_id"] for item in blocker_resolution["actions"]}
    assert "webapp_staged_request" in action_category_ids
    assert "pipeline_gpu_handoff" in action_category_ids
    assert all(
        item["proof_boundary"].startswith("Clearing this action only satisfies")
        for item in blocker_resolution["actions"]
    )
    assert "real-secret-token" not in json.dumps(blocker_resolution)
    category_ids = {item["category_id"] for item in blocker_resolution["categories"]}
    assert "source_video_preflight" in category_ids
    assert "webapp_staged_request" in category_ids
    assert "owner_gpu_gate" in category_ids
    assert scene_asset_acquisition["schema_version"] == (
        FIRST_GPU_SCENE_ASSET_ACQUISITION_SCHEMA_VERSION
    )
    assert scene_asset_acquisition["status"] == "blocked"
    assert "worldlabs_world_manifest_missing" in scene_asset_acquisition["blockers"]
    assert "materialized_scene_asset_missing" in scene_asset_acquisition["blockers"]
    assert scene_asset_acquisition["claim_boundary"]["live_provider_calls_performed"] is False
    assert scene_asset_acquisition["claim_boundary"]["remote_asset_downloads_performed"] is False
    assert "Scene Asset Acquisition" in scene_asset_acquisition_markdown
    assert webapp_handoff["schema_version"] == FIRST_GPU_WEBAPP_HANDOFF_SCHEMA_VERSION
    assert webapp_handoff["status"] == "blocked"
    assert "webapp_forwarding:missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL" in webapp_handoff[
        "blockers"
    ]
    assert "webapp_staged_request:missing_webapp_staged_inputs" in webapp_handoff["blockers"]
    assert webapp_handoff["forwarding"]["forward_token_configured"] is True
    assert webapp_handoff["forwarding"]["forward_token_value_redacted"] is True
    assert webapp_handoff["verification"]["required_env_status"][
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN"
    ] == {
        "configured": True,
        "value_redacted": True,
    }
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL" in webapp_handoff["verification"]["missing_env"]
    assert webapp_handoff["verification"]["script"]["path"] == (
        packet["generated_files"]["webapp_handoff_verification_commands"]
    )
    assert webapp_handoff["verification"]["script"]["safe_to_run_now"] is True
    assert webapp_handoff["verification"]["script"]["runs_live_webapp_call"] is False
    assert webapp_handoff["verification"]["script"]["stages_request"] is False
    assert webapp_handoff["claim_boundary"]["webapp_request_submitted_by_this_packet"] is False
    assert webapp_handoff["claim_boundary"]["live_forwarding_performed_by_this_packet"] is False
    assert webapp_handoff["claim_boundary"]["robot_readiness_proven"] is False
    assert "real-secret-token" not in json.dumps(webapp_handoff)
    assert "WebApp Handoff Packet" in webapp_handoff_markdown
    assert "real-secret-token" not in webapp_handoff_markdown
    assert packet["provider_guidance"]["nvidia_nim_boundary"]["primary_for_first_smoke"] is False
    assert packet["claim_boundary"]["robot_readiness_proven"] is False
    assert vm_sync_manifest["schema_version"] == FIRST_GPU_VM_SYNC_SCHEMA_VERSION
    assert vm_sync_manifest["claim_boundary"]["files_copied"] is False
    assert vm_sync_manifest["claim_boundary"]["simulator_execution_performed"] is False
    assert vm_sync_manifest["status"] == "blocked"
    sync_roles = {item["role"]: item for item in vm_sync_manifest["files"]}
    assert sync_roles["raw_manifest"]["exists"] is True
    assert sync_roles["raw_walkthrough_video"]["sha256"]
    assert sync_roles["run_packet_run_packet"]["exists"] is True
    assert sync_roles["run_packet_worldlabs_provider_submission_commands"]["exists"] is True
    assert sync_roles["run_packet_webapp_upstream_truth_verification_commands"][
        "exists"
    ] is True
    assert sync_roles["run_packet_webapp_handoff_verification_commands"]["exists"] is True
    assert sync_roles["run_packet_webapp_handoff"]["exists"] is True
    assert sync_roles["run_packet_gpu_vm_runtime_preflight_script"]["exists"] is True
    assert sync_roles["run_packet_owner_command_binding_template"]["exists"] is True
    assert sync_roles["run_packet_live_policy_execution_contract"]["exists"] is True
    assert sync_roles[
        "run_packet_default_test_robot_eval_job_request_template"
    ]["exists"] is True
    assert sync_roles["run_packet_real_robot_pov_manifest_template"]["exists"] is True
    assert sync_roles["run_packet_live_input_staging_commands"]["exists"] is True
    assert sync_roles["run_packet_simulator_path_matrix"]["exists"] is True
    assert sync_roles["run_packet_launch_order"]["exists"] is True
    assert "GPU VM Sync Manifest" in vm_sync_markdown
    for generated_path in (
        packet["generated_files"]["env_example"],
        packet["generated_files"]["local_preflight_commands"],
        packet["generated_files"]["worldlabs_provider_submission_commands"],
        packet["generated_files"]["webapp_upstream_truth_verification_commands"],
        packet["generated_files"]["webapp_handoff_verification_commands"],
        packet["generated_files"]["gpu_vm_commands"],
        packet["generated_files"]["gpu_vm_runtime_preflight_script"],
        packet["generated_files"]["owner_command_binding_template"],
        packet["generated_files"]["live_input_staging_commands"],
    ):
        parsed = subprocess.run(
            ["bash", "-n", generated_path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr


def test_owner_command_binding_template_runs_default_policy_helper(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    packet = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh",
        output_dir=tmp_path / "packet",
    )
    binding_path = Path(packet["generated_files"]["owner_command_binding_template"])
    proof_dir = tmp_path / "owner-proof"
    scene_script = tmp_path / "write_scene_trace.py"
    spawn_script = tmp_path / "write_spawn_trace.py"
    policy_script = tmp_path / "run_default_policy.py"
    scene_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "path = Path(os.environ['BLUEPRINT_SCENE_LOAD_TRACE'])",
                "path.parent.mkdir(parents=True, exist_ok=True)",
                "asset = {'name': os.environ['BLUEPRINT_ROBOT_ASSET_NAME'], 'uri_or_path': os.environ['BLUEPRINT_ROBOT_ASSET_URI_OR_PATH'], 'source': os.environ['BLUEPRINT_ROBOT_ASSET_SOURCE'], 'asset_class': os.environ['BLUEPRINT_ROBOT_ASSET_CLASS']}",
                "path.write_text(json.dumps({'status': 'loaded', 'scene_loaded': True, 'robot_asset': asset}))",
            ]
        ),
        encoding="utf-8",
    )
    spawn_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "path = Path(os.environ['BLUEPRINT_SPAWN_TRACE'])",
                "path.parent.mkdir(parents=True, exist_ok=True)",
                "asset = {'name': os.environ['BLUEPRINT_ROBOT_ASSET_NAME'], 'uri_or_path': os.environ['BLUEPRINT_ROBOT_ASSET_URI_OR_PATH'], 'source': os.environ['BLUEPRINT_ROBOT_ASSET_SOURCE'], 'asset_class': os.environ['BLUEPRINT_ROBOT_ASSET_CLASS']}",
                "path.write_text(json.dumps({'status': 'validated', 'spawn_pose_loaded': True, 'robot_asset': asset}))",
            ]
        ),
        encoding="utf-8",
    )
    policy_script.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "frame = Path(os.environ['SIM_ROBOT_POV_FRAME_PATH'])",
                "frame.parent.mkdir(parents=True, exist_ok=True)",
                "frame.write_bytes(b'fake simulator frame evidence')",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(Path.cwd() / "src"),
                    env.get("PYTHONPATH", ""),
                ]
            ),
            "BLUEPRINT_CAPTURE_ROOT": str(capture_root),
            "BLUEPRINT_SCENE_LOAD_TRACE": str(proof_dir / "owner_scene_load_trace.json"),
            "BLUEPRINT_SPAWN_TRACE": str(proof_dir / "owner_spawn_pose_trace.json"),
            "BLUEPRINT_POLICY_EXECUTION_TRACE": str(
                proof_dir / "owner_action_or_policy_trace.json"
            ),
            "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE": str(
                proof_dir / "owner_sim_robot_pov_evidence_manifest.json"
            ),
            "BLUEPRINT_ARTIFACT_MANIFEST": str(proof_dir / "owner_artifact_manifest.json"),
            "BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET": "dock_pose_a",
            "BLUEPRINT_ROBOT_ASSET_NAME": "Unitree G1",
            "BLUEPRINT_ROBOT_ASSET_URI_OR_PATH": "Robots/Unitree/G1/g1.usd",
            "BLUEPRINT_ROBOT_ASSET_SOURCE": "isaac_sim_robot_assets",
            "BLUEPRINT_ROBOT_ASSET_CLASS": "humanoid",
            "SIM_ROBOT_POV_FRAME_PATH": str(proof_dir / "frames" / "front-rgbd-0001.png"),
            "OWNER_SCENE_LOAD_COMMAND": f"{sys.executable} {scene_script}",
            "OWNER_ROBOT_SPAWN_COMMAND": f"{sys.executable} {spawn_script}",
            "OWNER_WALK_TO_TARGET_COMMAND": f"{sys.executable} {policy_script}",
        }
    )

    completed = subprocess.run(
        ["bash", str(binding_path)],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    policy_trace = json.loads(
        (proof_dir / "owner_action_or_policy_trace.json").read_text(encoding="utf-8")
    )
    sim_pov = json.loads(
        (proof_dir / "owner_sim_robot_pov_evidence_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    artifact_manifest = json.loads(
        (proof_dir / "owner_artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert policy_trace["default_policy_executed"] is True
    assert policy_trace["actions"][0]["target"] == "dock_pose_a"
    assert sim_pov["sim_robot_pov_captured"] is True
    assert sim_pov["frames"][0]["path"].endswith("front-rgbd-0001.png")
    assert any(item["kind"] == "sim_robot_pov_frame" for item in artifact_manifest["artifacts"])


def test_first_gpu_run_packet_mujoco_selects_menagerie_g1_and_cpu_first_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_scene_asset_artifacts(capture_root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        simulator="mujoco",
        output_dir=tmp_path / "packet",
        require_webapp_forwarding=False,
        require_webapp_staged_request=False,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    env_example = Path(packet["generated_files"]["env_example"]).read_text(encoding="utf-8")
    gpu_commands = Path(packet["generated_files"]["gpu_vm_commands"]).read_text(encoding="utf-8")
    gpu_vm_runtime_preflight_script = Path(
        packet["generated_files"]["gpu_vm_runtime_preflight_script"]
    ).read_text(encoding="utf-8")
    gpu_vm_runtime_preflight_plan = json.loads(
        Path(packet["generated_files"]["gpu_vm_runtime_preflight_plan"]).read_text(
            encoding="utf-8"
        )
    )
    owner_contract = Path(packet["generated_files"]["owner_command_contract"]).read_text(
        encoding="utf-8"
    )
    simulator_path_matrix = json.loads(
        Path(packet["generated_files"]["simulator_path_matrix"]).read_text(encoding="utf-8")
    )
    provider_bootstrap_manifest = json.loads(
        Path(packet["generated_files"]["gpu_provider_bootstrap_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    default_test_job_request_template = json.loads(
        Path(
            packet["generated_files"]["default_test_robot_eval_job_request_template"]
        ).read_text(encoding="utf-8")
    )
    vm_sync_manifest = json.loads(
        Path(packet["generated_files"]["gpu_vm_sync_manifest"]).read_text(encoding="utf-8")
    )
    mujoco_smoke_script = Path(
        packet["generated_files"]["mujoco_unitree_g1_smoke_script"]
    ).read_text(encoding="utf-8")

    assert packet["simulator"] == "mujoco"
    assert packet["owner_command_supplied"] is False
    assert packet["owner_command_generated_by_packet"] is True
    assert packet["owner_command_available_for_selected_path"] is True
    assert packet["owner_command_placeholder"].endswith("run_mujoco_unitree_g1_smoke.sh")
    assert "output/external_assets/mujoco_menagerie/unitree_g1/g1.xml" in env_example
    assert "google_deepmind_mujoco_menagerie" in env_example
    assert "export MUJOCO_UNITREE_G1_SMOKE_COMMAND=" in env_example
    assert 'export OWNER_SIMULATOR_COMMAND="$MUJOCO_UNITREE_G1_SMOKE_COMMAND"' in env_example
    assert "Robots/Unitree/G1/g1.usd" not in owner_contract
    assert "mujoco_unitree_g1_smoke.py" in owner_contract
    assert "MuJoCo Menagerie Unitree G1 MJCF" in owner_contract
    assert "MUJOCO_UNITREE_G1_SMOKE_COMMAND" in gpu_commands
    assert '--simulator-backend mujoco' in gpu_commands
    assert "mujoco_runtime_probe" in gpu_vm_runtime_preflight_script
    assert "optional_gpu_probe:" in gpu_vm_runtime_preflight_script
    assert "mujoco_menagerie_unitree_g1_xml_missing" in gpu_vm_runtime_preflight_script
    assert any(
        "mujoco imports" in item
        for item in gpu_vm_runtime_preflight_plan["inputs_checked_when_script_runs"]
    )
    assert provider_bootstrap_manifest["first_smoke_path"]["cheapest_serious_path"] is True
    assert provider_bootstrap_manifest["first_smoke_path"][
        "requires_paid_gpu_for_owner_runtime"
    ] is False
    assert provider_bootstrap_manifest["gpu_guidance"]["minimum_vram_gb"] == 0
    assert default_test_job_request_template["simulator_preference"] == "mujoco"
    simulator_robot_asset = default_test_job_request_template["robot_profile"][
        "simulator_robot_asset"
    ]
    assert simulator_robot_asset["name"] == "Unitree G1"
    assert (
        simulator_robot_asset["uri_or_path"]
        == "output/external_assets/mujoco_menagerie/unitree_g1/g1.xml"
    )
    assert simulator_robot_asset["source"] == "google_deepmind_mujoco_menagerie"
    assert simulator_robot_asset["asset_class"] == "humanoid_mjcf"
    assert simulator_robot_asset["fail_closed_if_missing"] is True
    assert simulator_path_matrix["selected_simulator"] == "mujoco"
    assert simulator_path_matrix["first_gpu_recommendation"]["recommended_first_path"] == "mujoco"
    paths = {item["framework"]: item for item in simulator_path_matrix["paths"]}
    assert paths["mujoco"]["recommended_first_gpu_smoke"] is True
    assert paths["isaac_sim"]["recommended_first_gpu_smoke"] is False
    assert "google_deepmind_mujoco_menagerie" in mujoco_smoke_script
    sync_roles = {item["role"]: item for item in vm_sync_manifest["files"]}
    assert sync_roles["run_packet_mujoco_unitree_g1_smoke_script"]["exists"] is True
    assert sync_roles["run_packet_mujoco_unitree_g1_smoke_launcher"]["exists"] is True
    assert any(
        role.startswith("mujoco_menagerie_unitree_g1_g1_xml_")
        for role in sync_roles
    )
    for generated_path in (
        packet["generated_files"]["gpu_vm_commands"],
        packet["generated_files"]["gpu_vm_runtime_preflight_script"],
        packet["generated_files"]["mujoco_unitree_g1_smoke_launcher"],
    ):
        parsed = subprocess.run(
            ["bash", "-n", generated_path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr


def test_first_gpu_blocker_resolution_includes_webapp_upstream_field_details(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    for path in (capture_root / "raw" / "manifest.json", capture_root / "capture_descriptor.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "site_submission_id",
            "request_id",
            "buyer_request_id",
            "capture_job_id",
        ):
            payload.pop(field, None)
        _write_json(path, payload)

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh",
        output_dir=tmp_path / "packet",
        require_webapp_forwarding=False,
        require_webapp_staged_request=False,
        require_gpu_gates=False,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    blocker_resolution = json.loads(
        Path(packet["generated_files"]["blocker_resolution"]).read_text(encoding="utf-8")
    )
    blocker_resolution_markdown = Path(
        packet["generated_files"]["blocker_resolution_markdown"]
    ).read_text(encoding="utf-8")
    verify_run = subprocess.run(
        ["bash", packet["generated_files"]["webapp_upstream_truth_verification_commands"]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify_run.returncode == 3
    verification_result = json.loads(
        (tmp_path / "packet" / "webapp_upstream_truth_verification_result.json").read_text(
            encoding="utf-8"
        )
    )
    actions = {item["category_id"]: item for item in blocker_resolution["actions"]}
    upstream_details = actions["webapp_upstream_truth"]["blocker_details"]

    assert verification_result["schema_version"] == (
        "first_gpu_webapp_upstream_truth_verification_result.v1"
    )
    assert verification_result["status"] == "blocked"
    assert verification_result["claim_boundary"]["artifacts_mutated"] is False
    assert verification_result["claim_boundary"]["webapp_requests_submitted"] is False
    assert verification_result["blockers"] == [
        "missing_or_placeholder_webapp_site_submission_id",
        "missing_or_placeholder_webapp_request_id",
        "missing_or_placeholder_webapp_buyer_request_id",
        "missing_or_placeholder_webapp_capture_job_id",
    ]
    assert {item["field"] for item in upstream_details} == {
        "site_submission_id",
        "request_id",
        "buyer_request_id",
        "capture_job_id",
    }
    assert all(
        "robot_eval_job_request.v1 owner_system" in item["accepted_evidence_sources"]
        for item in upstream_details
    )
    assert all(item["severity"] == "hard_pre_gpu_blocker" for item in upstream_details)
    assert "Accepted evidence sources:" in blocker_resolution_markdown
    assert "robot_eval_job_request.v1 site_package" in blocker_resolution_markdown


def test_first_gpu_scene_asset_acquisition_marks_provider_submission_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
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

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh",
        output_dir=tmp_path / "packet",
        require_webapp_forwarding=False,
        require_webapp_staged_request=False,
        require_gpu_gates=False,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    scene_asset_acquisition = json.loads(
        Path(packet["generated_files"]["scene_asset_acquisition"]).read_text(
            encoding="utf-8"
        )
    )
    scene_asset_acquisition_markdown = Path(
        packet["generated_files"]["scene_asset_acquisition_markdown"]
    ).read_text(encoding="utf-8")
    provider_submission = scene_asset_acquisition["provider_submission"]

    assert scene_asset_acquisition["status"] == "blocked"
    assert provider_submission["input_status"] == "ready_for_worldlabs_request_inputs"
    assert provider_submission["ready_for_worldlabs_request_inputs"] is True
    assert provider_submission["status"] == "ready_to_submit_worldlabs_request"
    assert provider_submission["ready_to_submit_worldlabs_request"] is True
    assert provider_submission["safe_to_submit_before_gpu_spend"] is True
    assert provider_submission["requires_env"] == [
        "WORLDLABS_API_KEY",
        "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION",
    ]
    assert provider_submission["missing_env"] == []
    assert provider_submission["required_env_status"]["WORLDLABS_API_KEY"] == {
        "configured": True,
        "value_redacted": True,
    }
    assert provider_submission["required_env_status"][
        "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION"
    ] == {
        "configured": True,
        "required_value": "true",
    }
    assert provider_submission["script"]["safe_to_run_now"] is True
    assert provider_submission["script"]["requires_explicit_allow_env"] == (
        "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION"
    )
    assert provider_submission["requires_gpu"] is False
    assert provider_submission["requires_live_provider_call"] is True
    assert provider_submission["input_video_preflight_ready"] is True
    assert scene_asset_acquisition["claim_boundary"]["gpu_provisioning_performed"] is False
    assert "secret-worldlabs-key" not in json.dumps(scene_asset_acquisition)
    assert "Provider Submission" in scene_asset_acquisition_markdown
    assert "ready_to_submit_worldlabs_request" in scene_asset_acquisition_markdown
    assert "WORLDLABS_API_KEY" in scene_asset_acquisition_markdown


def test_first_gpu_scene_asset_acquisition_blocks_provider_submission_without_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    monkeypatch.delenv("WORLDLABS_API_KEY", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION", raising=False)
    _write_json(
        capture_root / "pipeline" / "source_video_preflight_manifest.json",
        {
            "schema_version": "first_gpu_sample_video_preflight.v1",
            "status": "ready",
            "ready_for_worldlabs_first_clip_count": 1,
            "candidates": [],
        },
    )

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh",
        output_dir=tmp_path / "packet",
        require_webapp_forwarding=False,
        require_webapp_staged_request=False,
        require_gpu_gates=False,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    scene_asset_acquisition = json.loads(
        Path(packet["generated_files"]["scene_asset_acquisition"]).read_text(
            encoding="utf-8"
        )
    )
    scene_asset_acquisition_markdown = Path(
        packet["generated_files"]["scene_asset_acquisition_markdown"]
    ).read_text(encoding="utf-8")
    provider_submission = scene_asset_acquisition["provider_submission"]

    assert provider_submission["input_status"] == "ready_for_worldlabs_request_inputs"
    assert provider_submission["ready_for_worldlabs_request_inputs"] is True
    assert provider_submission["status"] == "blocked_missing_worldlabs_api_key"
    assert provider_submission["ready_to_submit_worldlabs_request"] is False
    assert provider_submission["safe_to_submit_before_gpu_spend"] is True
    assert provider_submission["missing_env"] == [
        "WORLDLABS_API_KEY",
        "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION",
    ]
    assert provider_submission["required_env_status"]["WORLDLABS_API_KEY"] == {
        "configured": False,
        "value_redacted": True,
    }
    assert provider_submission["required_env_status"][
        "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION"
    ] == {
        "configured": False,
        "required_value": "true",
    }
    assert provider_submission["script"]["safe_to_run_now"] is False
    assert "Missing env:" in scene_asset_acquisition_markdown
    assert "blocked_missing_worldlabs_api_key" in scene_asset_acquisition_markdown


def test_first_gpu_scene_asset_acquisition_blocks_provider_submission_without_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    monkeypatch.setenv("WORLDLABS_API_KEY", "secret-worldlabs-key")
    monkeypatch.delenv("BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION", raising=False)
    _write_json(
        capture_root / "pipeline" / "source_video_preflight_manifest.json",
        {
            "schema_version": "first_gpu_sample_video_preflight.v1",
            "status": "ready",
            "ready_for_worldlabs_first_clip_count": 1,
            "candidates": [],
        },
    )

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh",
        output_dir=tmp_path / "packet",
        require_webapp_forwarding=False,
        require_webapp_staged_request=False,
        require_gpu_gates=False,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    scene_asset_acquisition = json.loads(
        Path(packet["generated_files"]["scene_asset_acquisition"]).read_text(
            encoding="utf-8"
        )
    )
    provider_submission = scene_asset_acquisition["provider_submission"]

    assert provider_submission["input_status"] == "ready_for_worldlabs_request_inputs"
    assert provider_submission["status"] == "blocked_missing_worldlabs_submission_gate"
    assert provider_submission["ready_to_submit_worldlabs_request"] is False
    assert provider_submission["missing_env"] == [
        "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION"
    ]
    assert provider_submission["required_env_status"]["WORLDLABS_API_KEY"] == {
        "configured": True,
        "value_redacted": True,
    }
    assert provider_submission["script"]["safe_to_run_now"] is False
    assert "secret-worldlabs-key" not in json.dumps(scene_asset_acquisition)


def test_first_gpu_run_packet_cli_writes_packet(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    output_dir = tmp_path / "cli-packet"

    exit_code = main(
        [
            "--capture-root",
            str(capture_root),
            "--webapp-site-slug",
            "site-1",
            "--owner-command",
            "/opt/blueprint/run_isaac_gpu_proof.sh",
            "--output-dir",
            str(output_dir),
            "--no-require-webapp-forwarding",
            "--no-require-gpu-gates",
        ]
    )

    assert exit_code == 0
    packet = json.loads((output_dir / "first_gpu_run_packet.json").read_text(encoding="utf-8"))
    assert packet["schema_version"] == FIRST_GPU_RUN_PACKET_SCHEMA_VERSION
    assert packet["owner_command_supplied"] is True
    assert packet["owner_command_location"] == "remote"
    assert "simulator_runtime:missing_simulator_command" not in packet["blockers"]
    assert (output_dir / "first_gpu_e2e_readiness_manifest.json").is_file()
    assert (output_dir / "gpu_vm_commands.sh").is_file()
    assert (output_dir / "gpu_vm_runtime_preflight.sh").is_file()
    assert (output_dir / "gpu_vm_runtime_preflight_plan.json").is_file()
    assert (output_dir / "gpu_vm_runtime_preflight_plan.md").is_file()
    assert (output_dir / "isaac_unitree_g1_smoke.py").is_file()
    assert (output_dir / "run_isaac_unitree_g1_smoke.sh").is_file()
    assert (output_dir / "first_gpu_simulator_path_matrix.json").is_file()
    assert (output_dir / "first_gpu_simulator_path_matrix.md").is_file()
    assert (output_dir / "first_gpu_launch_order.json").is_file()
    assert (output_dir / "first_gpu_launch_order.md").is_file()
    assert (output_dir / "gpu_provider_bootstrap.md").is_file()
    assert (output_dir / "gpu_provider_bootstrap.json").is_file()
    assert (output_dir / "first_gpu_blocker_resolution.json").is_file()
    assert (output_dir / "first_gpu_blocker_resolution.md").is_file()
    assert (output_dir / "first_gpu_scene_asset_acquisition.json").is_file()
    assert (output_dir / "first_gpu_scene_asset_acquisition.md").is_file()
    assert (output_dir / "first_gpu_webapp_handoff.json").is_file()
    assert (output_dir / "first_gpu_webapp_handoff.md").is_file()
    assert (output_dir / "gpu_vm_sync_manifest.json").is_file()
    assert (output_dir / "gpu_vm_sync_manifest.md").is_file()


def test_first_gpu_blocker_resolution_includes_pre_gpu_blocker_details(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_blocked_gpu_handoff_with_details(capture_root)

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        output_dir=tmp_path / "packet",
        require_webapp_forwarding=False,
        require_webapp_staged_request=False,
        require_gpu_gates=False,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    blocker_resolution = json.loads(
        Path(packet["generated_files"]["blocker_resolution"]).read_text(encoding="utf-8")
    )
    blocker_resolution_markdown = Path(
        packet["generated_files"]["blocker_resolution_markdown"]
    ).read_text(encoding="utf-8")
    actions = {item["category_id"]: item for item in blocker_resolution["actions"]}

    scene_details = actions["scene_spawn_preflight"]["blocker_details"]
    handoff_details = actions["pipeline_gpu_handoff"]["blocker_details"]
    assert {item["blocker_id"] for item in scene_details} >= {
        "missing_local_scene_asset",
        "missing_scene_frame_estimate",
    }
    assert {item["blocker_id"] for item in handoff_details} >= {
        "missing_local_scene_asset",
        "missing_scene_frame_estimate",
        "portable_collider_glb_missing",
    }
    assert all(
        item["proof_boundary"].endswith("does not prove simulator execution or robot readiness")
        for item in scene_details + handoff_details
    )
    assert "Blocker details:" in blocker_resolution_markdown
    assert "missing_local_scene_asset" in blocker_resolution_markdown
    assert "portable_collider_glb_missing" in blocker_resolution_markdown


def test_first_gpu_launch_order_allows_owner_gpu_command_before_closure_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
    _write_scene_asset_artifacts(capture_root)
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    output_dir = tmp_path / "packet"
    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        output_dir=output_dir,
    )
    _write_gpu_vm_runtime_preflight_result(output_dir)
    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        output_dir=output_dir,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    launch_order = json.loads(
        Path(packet["generated_files"]["launch_order"]).read_text(encoding="utf-8")
    )
    blocker_resolution = json.loads(
        Path(packet["generated_files"]["blocker_resolution"]).read_text(encoding="utf-8")
    )
    webapp_handoff = json.loads(
        Path(packet["generated_files"]["webapp_handoff"]).read_text(encoding="utf-8")
    )
    launch_steps = {item["step_id"]: item for item in launch_order["steps"]}

    assert packet["readiness_status"] == "ready_for_owner_gpu_attempt"
    assert packet["ready_for_first_gpu_attempt"] is True
    assert packet["owner_gpu_proof_ready"] is False
    assert packet["blockers"] == []
    runtime_preflight_plan = json.loads(
        Path(packet["generated_files"]["gpu_vm_runtime_preflight_plan"]).read_text(
            encoding="utf-8"
        )
    )
    assert runtime_preflight_plan["result"]["ready_for_owner_command_attempt"] is True
    assert launch_order["status"] == "ready_for_owner_gpu_launch"
    assert launch_order["gpu_execution_allowed"] is True
    assert launch_steps["owner_gpu_simulator_proof"]["may_run_now"] is True
    assert launch_steps["post_gpu_readiness_audit"]["may_run_now"] is False
    assert launch_steps["post_gpu_readiness_audit"]["status"] == "pending_after_owner_gpu_run"
    assert "post_gpu_readiness_audit" not in launch_order["blocked_step_ids"]
    assert launch_order["next_action_step_ids"] == ["owner_gpu_simulator_proof"]
    assert launch_order["forbidden_actions_until_ready"] == [
        "do_not_claim_owner_gpu_or_robot_readiness",
    ]
    assert launch_order["claim_boundary"]["simulator_execution_performed"] is False
    assert launch_order["claim_boundary"]["robot_readiness_proven"] is False
    assert blocker_resolution["action_count"] == 0
    assert blocker_resolution["blocked_action_count"] == 0
    assert blocker_resolution["actions"] == []

    upstream_verify_run = subprocess.run(
        ["bash", packet["generated_files"]["webapp_upstream_truth_verification_commands"]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert upstream_verify_run.returncode == 0, upstream_verify_run.stderr
    upstream_result_path = Path(
        packet["generated_files"]["webapp_upstream_truth_verification_commands"]
    ).with_name("webapp_upstream_truth_verification_result.json")
    upstream_verification_result = json.loads(
        upstream_result_path.read_text(encoding="utf-8")
    )
    assert upstream_verification_result["status"] == "ready"
    assert upstream_verification_result["values_redacted"] == {
        "site_submission_id": True,
        "request_id": True,
        "buyer_request_id": True,
        "capture_job_id": True,
    }
    assert (
        upstream_verification_result["claim_boundary"]["webapp_requests_submitted"]
        is False
    )

    verify_run = subprocess.run(
        ["bash", packet["generated_files"]["webapp_handoff_verification_commands"]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify_run.returncode == 0, verify_run.stderr
    assert "secret-token" not in verify_run.stdout
    verification_result = json.loads(
        Path(webapp_handoff["verification"]["script"]["default_result_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert verification_result["status"] == "ready"
    assert verification_result["forwarding"]["forward_token_value_redacted"] is True
    assert verification_result["claim_boundary"]["live_forwarding_performed"] is False


def test_first_gpu_run_packet_accepts_webapp_forwarding_preflight_report_without_token_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
    _write_scene_asset_artifacts(capture_root)
    preflight_path = _write_webapp_forwarding_preflight_report(
        capture_root / "pipeline" / "webapp_forwarding_preflight.json",
    )
    monkeypatch.delenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", raising=False)
    monkeypatch.delenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", raising=False)
    monkeypatch.delenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        raising=False,
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    output_dir = tmp_path / "packet"
    build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        webapp_forwarding_preflight_path=preflight_path,
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        output_dir=output_dir,
    )
    _write_gpu_vm_runtime_preflight_result(output_dir)
    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        webapp_forwarding_preflight_path=preflight_path,
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        output_dir=output_dir,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    env_example = Path(packet["generated_files"]["env_example"]).read_text(encoding="utf-8")
    local_commands = Path(packet["generated_files"]["local_preflight_commands"]).read_text(
        encoding="utf-8"
    )
    webapp_handoff = json.loads(
        Path(packet["generated_files"]["webapp_handoff"]).read_text(encoding="utf-8")
    )
    launch_order = json.loads(
        Path(packet["generated_files"]["launch_order"]).read_text(encoding="utf-8")
    )

    assert packet["readiness_status"] == "ready_for_owner_gpu_attempt"
    assert packet["ready_for_first_gpu_attempt"] is True
    assert packet["webapp_forwarding_preflight_path"] == str(preflight_path)
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT" in env_example
    assert "--webapp-forwarding-preflight" in local_commands
    assert webapp_handoff["status"] == "ready_for_webapp_handoff_verification"
    assert webapp_handoff["forwarding"]["forward_url_configured"] is False
    assert webapp_handoff["forwarding"]["forward_token_configured"] is False
    assert webapp_handoff["forwarding"]["forward_url_evidence_present"] is True
    assert webapp_handoff["forwarding"]["forward_token_evidence_present"] is True
    assert webapp_handoff["forwarding"]["capture_root_override_source"] == (
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT"
    )
    assert webapp_handoff["forwarding"]["forwarding_preflight"]["ready"] is True
    assert webapp_handoff["verification"]["script"]["requires_forwarding_token_in_shell"] is False
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN" not in webapp_handoff[
        "verification"
    ]["missing_env"]
    assert launch_order["status"] == "ready_for_owner_gpu_launch"

    verify_run = subprocess.run(
        ["bash", packet["generated_files"]["webapp_handoff_verification_commands"]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify_run.returncode == 0, verify_run.stderr
    verification_result = json.loads(
        Path(webapp_handoff["verification"]["script"]["default_result_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert verification_result["status"] == "ready"
    assert verification_result["forwarding"]["forward_token_configured"] is False
    assert verification_result["forwarding"]["forward_token_evidence_present"] is True
    assert verification_result["forwarding"]["forwarding_preflight"]["ready"] is True
    assert verification_result["claim_boundary"]["live_forwarding_performed"] is False
    assert "secret-token" not in json.dumps(packet)
    assert "secret-token" not in verify_run.stdout


def test_first_gpu_launch_order_blocks_owner_gpu_command_when_sync_manifest_is_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_gpu_handoff_artifacts(capture_root)
    _write_staged_webapp_request(capture_root)
    _write_scene_asset_artifacts(capture_root)
    (capture_root / "pipeline" / "simulation_automation" / "scene_frame_estimate.json").unlink()
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL", "https://pipeline.example/intake")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "secret-token")
    monkeypatch.setenv(
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
        json.dumps({"site-1": str(capture_root.resolve())}),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh --capture-root /mnt/capture",
        output_dir=tmp_path / "packet",
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    launch_order = json.loads(
        Path(packet["generated_files"]["launch_order"]).read_text(encoding="utf-8")
    )
    vm_sync_manifest = json.loads(
        Path(packet["generated_files"]["gpu_vm_sync_manifest"]).read_text(encoding="utf-8")
    )
    runtime_preflight_plan = json.loads(
        Path(packet["generated_files"]["gpu_vm_runtime_preflight_plan"]).read_text(
            encoding="utf-8"
        )
    )
    launch_steps = {item["step_id"]: item for item in launch_order["steps"]}

    assert packet["readiness_status"] == "ready_for_owner_gpu_attempt"
    assert packet["ready_for_first_gpu_attempt"] is True
    assert vm_sync_manifest["status"] == "blocked"
    assert "missing_required_sync_file:scene_frame_estimate" in vm_sync_manifest["blockers"]
    assert runtime_preflight_plan["status"] == "blocked_for_owner_gpu_attempt"
    assert runtime_preflight_plan["gpu_vm_sync_status"] == "blocked"
    assert "gpu_vm_sync_manifest:missing_required_sync_file:scene_frame_estimate" in (
        runtime_preflight_plan["hard_stop_blockers"]
    )
    assert runtime_preflight_plan["script"]["safe_to_run_on_gpu_vm"] is False
    assert launch_order["status"] == "blocked"
    assert launch_order["gpu_execution_allowed"] is False
    assert "gpu_vm_sync" in launch_order["blocked_step_ids"]
    assert "gpu_vm_runtime_preflight" in launch_order["blocked_step_ids"]
    assert "owner_gpu_simulator_proof" not in launch_order["next_action_step_ids"]
    assert launch_order["next_action_step_ids"] == ["gpu_vm_sync", "gpu_vm_runtime_preflight"]
    assert launch_steps["gpu_vm_sync"]["may_run_now"] is False
    assert launch_steps["gpu_vm_runtime_preflight"]["may_run_now"] is False
    assert launch_steps["owner_gpu_simulator_proof"]["may_run_now"] is False
    assert "gpu_vm_runtime_preflight_not_ready" in launch_steps["owner_gpu_simulator_proof"][
        "blockers"
    ]
    assert "do_not_run_gpu_vm_commands" in launch_order["forbidden_actions_until_ready"]


def test_first_gpu_run_packet_blocks_when_owner_command_is_not_supplied(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        output_dir=tmp_path / "packet",
        require_webapp_forwarding=False,
        require_gpu_gates=False,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    blocker_resolution = json.loads(
        Path(packet["generated_files"]["blocker_resolution"]).read_text(encoding="utf-8")
    )
    blocker_resolution_markdown = Path(
        packet["generated_files"]["blocker_resolution_markdown"]
    ).read_text(encoding="utf-8")
    actions = {item["category_id"]: item for item in blocker_resolution["actions"]}
    owner_action = actions["owner_gpu_command"]
    owner_details = owner_action["blocker_details"]

    assert packet["owner_command_supplied"] is False
    assert packet["owner_command_location"] == "remote"
    assert "simulator_runtime:missing_simulator_command" in packet["blockers"]
    assert owner_details
    assert owner_details[0]["wrapper_command"] == "blueprint-run-owner-gpu-proof"
    assert "BLUEPRINT_SCENE_LOAD_TRACE" in owner_details[0]["trace_environment_variables"]
    assert "BLUEPRINT_ACTION_OR_POLICY_TRACE" in owner_details[0][
        "trace_environment_variables"
    ]
    assert "BLUEPRINT_POLICY_EXECUTION_TRACE" in owner_details[0][
        "trace_environment_variables"
    ]
    assert "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE" in owner_details[0][
        "trace_environment_variables"
    ]
    assert any(
        path.endswith("pipeline/simulation_automation/gpu_owner_system_proof.json")
        for path in owner_details[0]["expected_outputs"]
    )
    assert any(
        path.endswith(
            "pipeline/simulation_automation/owner_gpu_proof/"
            "owner_sim_robot_pov_evidence_manifest.json"
        )
        for path in owner_details[0]["expected_outputs"]
    )
    assert "Wrapper command: `blueprint-run-owner-gpu-proof`" in blocker_resolution_markdown
    assert "Trace env vars:" in blocker_resolution_markdown
    assert "gpu_owner_system_proof.json" in blocker_resolution_markdown


def test_first_gpu_run_packet_threads_local_rehearsal_flag(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)

    result = build_first_gpu_run_packet(
        capture_root=capture_root,
        webapp_site_slug="site-1",
        owner_command="/opt/blueprint/run_isaac_gpu_proof.sh",
        output_dir=tmp_path / "packet",
        allow_local_webapp_rehearsal=True,
    )

    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    local_commands = Path(packet["generated_files"]["local_preflight_commands"]).read_text(
        encoding="utf-8"
    )

    assert packet["allow_local_webapp_rehearsal"] is True
    assert "--allow-local-webapp-rehearsal" in local_commands
