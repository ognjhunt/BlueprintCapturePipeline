from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.g1_controlled_run_evidence import assemble_g1_controlled_run_evidence
from blueprint_pipeline.g1_controlled_proof_setup import build_g1_controlled_proof_setup
from blueprint_pipeline import realistic_readiness_rehearsal as rehearsal
from blueprint_pipeline.realistic_readiness_rehearsal import (
    REALISTIC_READINESS_REHEARSAL_SCHEMA_VERSION,
    build_realistic_readiness_rehearsal,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_complete_mujoco_rehearsal(capture_root: Path) -> Path:
    run_root = (
        capture_root
        / "pipeline"
        / "realistic_readiness_rehearsal"
        / "mujoco_g1_walk_to_target_run"
    )
    frame_path = run_root / "frames" / "sim_robot_follow_pov_0000.png"
    scene_trace = run_root / "scene_load_trace.json"
    spawn_trace = run_root / "spawn_trace.json"
    policy_trace = run_root / "policy_execution_trace.json"
    sim_pov = run_root / "sim_robot_pov_evidence_manifest.json"
    for artifact in (frame_path, scene_trace, spawn_trace, policy_trace, sim_pov):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
    manifest_path = run_root / "mujoco_g1_local_smoke_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "local_mujoco_g1_walk_to_target_smoke_manifest.v1",
            "status": "complete",
            "simulator_backend": "mujoco",
            "robot_asset": {
                "name": "Unitree G1",
                "source": "google_deepmind_mujoco_menagerie",
                "mujoco_g1_asset_execution_proven": True,
            },
            "policy_id": "blueprint_default_walk_to_target_smoke_policy",
            "policy_semantics": (
                "kinematic_root_pose_smoke_not_balanced_humanoid_locomotion_controller"
            ),
            "default_sim_policy_execution_proven": True,
            "sim_robot_pov_evidence_proven": True,
            "artifacts": {
                "scene_trace": str(scene_trace),
                "spawn_trace": str(spawn_trace),
                "policy_trace": str(policy_trace),
                "sim_robot_pov_evidence": str(sim_pov),
                "frames": [str(frame_path)],
            },
            "claim_boundary": {
                "local_cpu_mujoco_execution_proven": True,
                "mujoco_g1_asset_execution_proven": True,
                "real_robot_pov_evidence_proven": False,
                "generated_world_rank_fidelity_result_proven": False,
                "non_ranking_operational_claim_validated": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )
    return manifest_path


def _seed_job_request(capture_root: Path, job_id: str) -> Path:
    path = capture_root / "pipeline" / "robot_eval_jobs" / job_id / "job_request.json"
    _write_json(
        path,
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "buyer_request_id": "buyer-123",
            "requested_tasks": [
                {
                    "task_id": "walk_to_target",
                    "scenario_ids": ["site-a_walk_to_target_pose"],
                }
            ],
            "site_package": {
                "site_slug": "site-a",
                "site_submission_id": "site-submission-123",
                "request_id": "webapp-request-123",
                "buyer_request_id": "buyer-123",
                "capture_job_id": "capture-job-123",
                "capture_id": "capture-123",
                "capture_root": str(capture_root),
            },
        },
    )
    return path


def _seed_ready_physical_g1_evidence_drop(evidence_dir: Path, job_id: str) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "robot_camera_video.mp4").write_bytes(b"physical-g1-video")
    _write_json(
        evidence_dir / "timestamp_alignment.json",
        {
            "schema_version": "g1_timestamp_alignment.v1",
            "max_alignment_error_ms": 40,
        },
    )
    (evidence_dir / "action_log.jsonl").write_text(
        json.dumps({"kind": "action", "action_id": "walk_to_target"}) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "robot_state_log.jsonl").write_text(
        json.dumps({"kind": "state", "base_position": [0, 0, 0]}) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "command_log.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"kind": "policy_command_started", "command": "run-policy"}),
                json.dumps({"kind": "policy_command_completed", "exit_code": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        evidence_dir / "contact_collision_log.json",
        {
            "schema_version": "g1_contact_collision_log.v1",
            "status": "accepted",
            "events": [],
            "max_contact_force_n": 0,
        },
    )
    _write_json(
        evidence_dir / "hardware_validation.json",
        {
            "schema_version": "g1_hardware_validation.v1",
            "status": "accepted",
            "hardware_ready": True,
            "estop_verified": True,
        },
    )
    (evidence_dir / "policy_execution_trace.jsonl").write_text(
        json.dumps(
            {
                "policy_id": "unitree_rl_gym_g1_mujoco_policy_candidate",
                "kind": "policy_command_completed",
                "exit_code": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        evidence_dir / "policy_metrics.json",
        {
            "schema_version": "g1_policy_metrics.v1",
            "status": "accepted",
            "episode_count": 1,
            "success_rate": 1.0,
            "intervention_count": 0,
        },
    )
    _write_json(
        evidence_dir / "robot_team_review.json",
        {
            "schema_version": "g1_robot_team_review.v1",
            "review_decision": "accepted",
            "accepted": True,
            "reviewer_id": "robot-team-reviewer-a",
        },
    )
    _write_json(
        evidence_dir / "g1_controlled_run_inputs.json",
        {
            "schema_version": "g1_controlled_run_inputs.v1",
            "job_id": job_id,
            "run_id": "unitree-g1-controlled-run-001",
            "robot_serial_or_fleet_id": "unitree-g1-lab-001",
            "site_or_lab_location_id": "lab-a",
            "operator_id": "operator-a",
            "hardware_owner_id": "hardware-owner-a",
            "safety_reviewer_id": "safety-reviewer-a",
            "robot_team_reviewer_id": "robot-team-reviewer-a",
            "start_time_utc": "2026-06-12T14:00:00Z",
            "end_time_utc": "2026-06-12T14:02:00Z",
            "actual_status": "passed",
            "actual_success": True,
            "cycle_time_seconds": 42.0,
            "intervention_count": 0,
            "accepted_safety_thresholds": {
                "max_speed_mps": 0.4,
                "min_human_clearance_m": 2.0,
                "max_contact_force_n": 0,
                "emergency_stop_required": True,
            },
            "review_decision": "accepted",
            "storage_upload_performed": True,
            "entitlement_verified": True,
            "signed_customer_delivery_url": "https://signed.example.test/g1-run",
            "rights_privacy_status": "accepted",
            "external_use_allowed": True,
            "production_webapp_request_id": "webapp-request-123",
            "pipeline_intake_request_id": "pipeline-intake-123",
            "production_forward_url": "https://pipeline.example.test/api/live-pipeline",
            "webapp_response_status_code": "202",
            "sync_status": "succeeded",
            "operator_statement": "Operator signed the physical G1 evidence package.",
            "hardware_owner_statement": "Hardware owner signed the G1 identity and run.",
            "safety_reviewer_statement": "Safety reviewer accepted this controlled G1 run.",
            "robot_team_review_statement": (
                "Robot team accepted the non-default G1 policy package."
            ),
        },
    )


def _seed_provider_blockers(capture_root: Path, job_id: str) -> None:
    _write_json(
        capture_root
        / "pipeline"
        / "robot_eval_provider_inputs"
        / job_id
        / "provider_input_setup_manifest.json",
        {
            "status": "prepared_with_external_blockers",
            "blockers": [
                "worker_image_ref_is_candidate_until_built_and_pushed",
                "upload_failed:Forbidden",
            ],
            "proof_boundary": {
                "provider_inputs_uploaded": False,
                "image_ref_published_proven": False,
            },
        },
    )
    _write_json(
        capture_root / "pipeline" / "robot_eval_jobs" / job_id / "runpod_provider_adapter_result.json",
        {
            "status": "blocked",
            "api_call_performed": False,
            "runpod_side_effects_may_have_occurred": False,
            "blockers": ["provider_launch_request_not_ready"],
            "active_pod_count_before": None,
            "active_pod_count_after": None,
        },
    )
    _write_json(
        capture_root / "pipeline" / "g1_controlled_proof_setup" / "runpod_live_execution_proof.json",
        {
            "status": "blocked",
            "blockers": ["missing_env_RUNPOD_API_KEY"],
            "api_call_performed": False,
            "runpod_side_effects_may_have_occurred": False,
            "active_pod_count_before": None,
            "active_pod_count_after": None,
            "shutdown_or_termination_proof": False,
        },
    )
    _write_json(
        capture_root / "pipeline" / "robot_eval_jobs" / job_id / "live_eval_closure_manifest.json",
        {
            "status": "blocked",
            "blockers": ["real_robot_pov_evidence_missing", "non_ranking_operational_claim_missing"],
        },
    )
    _write_json(
        capture_root / "pipeline" / "production_handoff_readiness_manifest.json",
        {"status": "blocked_after_owner_gpu_handoff", "blockers": ["production_live_webapp_forwarding_not_proven"]},
    )
    _write_json(
        capture_root / "pipeline" / "provider_preview_qa_manifest.json",
        {"status": "blocked", "blockers": ["production_live_webapp_forwarding_not_proven"]},
    )


def _seed_same_entrypoint_worker_rehearsal(capture_root: Path, job_id: str) -> None:
    output_root = capture_root / "pipeline" / "realistic_readiness_rehearsal"
    worker_root = output_root / "same_entrypoint_worker_rehearsal"
    job_root = capture_root / "pipeline" / "robot_eval_jobs" / f"{job_id}-local-worker-rehearsal"
    _write_json(
        worker_root / "worker_runtime_manifest.json",
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "status": "blocked",
            "job_status": "blocked",
            "job_dir": str(job_root),
            "simulator": "mujoco",
            "provisioner": "fixture_local",
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "artifact_upload": {"status": "completed"},
            "blockers": [],
        },
    )
    _write_json(
        worker_root / "worker_runtime_preflight_detail.json",
        {
            "schema_version": "mujoco_worker_runtime_preflight.v1",
            "status": "passed",
            "blockers": [],
            "proof_boundary": {"runtime_preflight_executed": True},
        },
    )
    _write_json(
        job_root / "blocked_manifest.json",
        {"status": "blocked", "blockers": ["blocked_rights_privacy"]},
    )
    _write_json(
        job_root / "simulator_service_result.json",
        {
            "status": "blocked",
            "reason": "job_validation_blocked",
            "blockers": ["blocked_rights_privacy"],
            "simulator_execution_proven": False,
        },
    )


def _seed_container_worker_image_rehearsal(capture_root: Path) -> None:
    output_root = capture_root / "pipeline" / "realistic_readiness_rehearsal"
    worker_root = output_root / "container_worker_image_rehearsal"
    artifact_root = output_root / "container_worker_image_rehearsal_artifact_output"
    _write_json(
        worker_root / "container_image_manifest.json",
        {
            "schema_version": "container_worker_image_rehearsal_image.v1",
            "image_ref": "blueprint/mujoco-eval-worker:test",
            "image_id": "sha256:test",
            "architecture": "arm64",
            "os": "linux",
            "entrypoint": ["blueprint-run-robot-eval-worker"],
        },
    )
    _write_json(
        worker_root / "worker_runtime_manifest.json",
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "status": "blocked",
            "job_id": "robot-eval-test",
            "provisioner": "runpod",
            "simulator": "mujoco",
            "runtime_preflight_status": "blocked",
            "runtime_preflight_blockers": ["runtime_preflight_command_failed"],
            "artifact_upload": {"status": "completed"},
            "live_provider_calls_performed": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["worker_runtime_preflight_blocked"],
        },
    )
    _write_json(
        worker_root / "worker_runtime_preflight.json",
        {
            "schema_version": "robot_eval_worker_runtime_preflight.v1",
            "status": "blocked",
            "execution_performed": True,
            "exit_code": 2,
            "detail_status": "blocked",
            "detail_blockers": ["nvidia_smi_unavailable"],
            "blockers": ["runtime_preflight_command_failed"],
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )
    _write_json(
        worker_root / "worker_runtime_preflight_detail.json",
        {
            "schema_version": "mujoco_worker_runtime_preflight.v1",
            "status": "blocked",
            "blockers": ["nvidia_smi_unavailable"],
            "requirements": {"require_nvidia_smi": True, "require_egl_render": True},
            "proof_boundary": {"runtime_preflight_executed": True},
        },
    )
    for filename in (
        "worker_runtime_manifest.json",
        "worker_runtime_preflight.json",
        "worker_runtime_preflight_detail.json",
    ):
        source = worker_root / filename
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _seed_blocked_webapp_route_proof(capture_root: Path) -> None:
    _write_json(
        capture_root
        / "pipeline"
        / "webapp_route_forwarding_proof"
        / "webapp_route_forwarding_proof.json",
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "status": "blocked",
            "webapp_route": {
                "full_production_webapp_deployment_proven": True,
                "http_status": 202,
            },
            "pipeline_forward": {
                "status": "forwarded",
                "performed": True,
                "accepted": False,
                "pipeline_status": "blocked",
            },
            "pipeline_intake": {
                "accepted": False,
                "status": "blocked",
                "input_blockers": [
                    "webapp:request_capture_root_does_not_match_control_plane",
                    "staging:webapp_request_not_ready_for_staging",
                ],
            },
            "proof_boundary": {
                "production_live_webapp_forwarding_proven": False,
                "pipeline_intake_staged_request_proven": False,
                "full_webapp_db_persistence_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )


def _seed_successful_webapp_route_proof(capture_root: Path) -> Path:
    proof_path = (
        capture_root
        / "pipeline"
        / "webapp_route_forwarding_proof"
        / "webapp_route_forwarding_proof.production-path-g1.json"
    )
    _write_json(
        proof_path,
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "status": "forwarded_to_pipeline_intake",
            "webapp_route": {
                "full_production_webapp_deployment_proven": True,
                "http_status": 202,
            },
            "durable_store": {
                "status": "stored",
                "performed": True,
                "firestore": {
                    "status": "stored",
                    "performed": True,
                },
            },
            "pipeline_forward": {
                "status": "forwarded",
                "performed": True,
                "accepted": True,
                "pipeline_status": "staged_for_control_plane",
                "timeout_ms": 60000,
            },
            "pipeline_intake": {
                "accepted": True,
                "status": "staged_for_control_plane",
                "input_blockers": [],
            },
            "proof_boundary": {
                "production_live_webapp_forwarding_proven": True,
                "pipeline_intake_staged_request_proven": True,
                "full_webapp_db_persistence_proven": True,
                "public_claim_upgrade_allowed": False,
            },
        },
    )
    return proof_path


def _seed_successful_official_policy_execution(capture_root: Path) -> Path:
    execution_path = (
        capture_root
        / "pipeline"
        / "g1_controlled_proof_setup"
        / "official_unitree_g1_policy_execution"
        / "official_unitree_g1_policy_execution_manifest.json"
    )
    _write_json(
        execution_path,
        {
            "schema_version": "official_unitree_g1_policy_execution.v1",
            "status": "completed",
            "job_id": "robot-eval-test",
            "policy_id": "unitree_rl_gym_g1_pretrain_motion",
            "metrics": {
                "status": "completed",
                "steps": 2000,
                "control_updates": 200,
            },
            "execution": {
                "trace_path": str(execution_path.with_name("policy_execution_trace.jsonl")),
                "metrics_path": str(execution_path.with_name("policy_metrics.json")),
            },
            "proof_boundary": {
                "non_default_policy_execution_trace_proven": True,
                "policy_metrics_tied_to_scenario_variation": True,
                "robot_team_owner_acceptance_or_review_proven": False,
                "robot_team_policy_performance_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )
    execution_path.with_name("policy_execution_trace.jsonl").write_text(
        '{"step": 0}\n', encoding="utf-8"
    )
    _write_json(execution_path.with_name("policy_metrics.json"), {"status": "completed"})
    return execution_path


def test_realistic_rehearsal_records_simulator_proof_but_blocks_physical_claims(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_complete_mujoco_rehearsal(capture_root)
    _seed_provider_blockers(capture_root, job_id)
    _seed_same_entrypoint_worker_rehearsal(capture_root, job_id)
    _seed_container_worker_image_rehearsal(capture_root)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    output_path = Path(manifest["artifacts"]["manifest"])  # type: ignore[index]
    persisted = _read_json(output_path)
    matrix = persisted["requested_proof_matrix"]  # type: ignore[index]
    proof_boundary = persisted["proof_boundary"]  # type: ignore[index]
    current_state = persisted["current_proof_state"]  # type: ignore[index]
    assert persisted["schema_version"] == REALISTIC_READINESS_REHEARSAL_SCHEMA_VERSION
    assert persisted["status"] == "simulator_rehearsal_completed_external_evidence_blocked"
    assert persisted["default_robot"]["make_model"] == "Unitree G1"  # type: ignore[index]
    assert persisted["default_robot"]["robot_profile_id"] == "unitree_g1_humanoid"  # type: ignore[index]
    assert current_state["proven_gate_count"] == 0  # type: ignore[index]
    assert current_state["remaining_gate_count"] == 6  # type: ignore[index]
    assert current_state["all_live_product_gates_proven"] is False  # type: ignore[index]
    assert current_state["gates"]["generated_world_rank_fidelity"]["label"] == "Physical G1 readiness"  # type: ignore[index]
    assert current_state["gates"]["production_runpod_worker_execution"]["proven"] is False  # type: ignore[index]
    assert matrix["mujoco_unitree_g1_simulator_rehearsal"]["proven"] is True  # type: ignore[index]
    assert matrix["generated_world_rank_fidelity"]["proven"] is False  # type: ignore[index]
    assert matrix["real_robot_pov"]["proven"] is False  # type: ignore[index]
    assert matrix["robot_team_policy_performance"]["status"] == "not_proven_default_smoke_policy_only"  # type: ignore[index]
    assert matrix["production_runpod_worker_execution"]["status"] == "not_run_provider_gates_blocked"  # type: ignore[index]
    assert matrix["customer_through_website_testing_ready"]["status"] == "not_ready_missing_production_webapp_route_proof"  # type: ignore[index]
    assert "missing_BLUEPRINT_WEBAPP_PRODUCTION_URL" in matrix["customer_through_website_testing_ready"]["blockers"]  # type: ignore[index]
    assert "upload_failed:Forbidden" in matrix["production_runpod_worker_execution"]["blockers"]  # type: ignore[index]
    assert proof_boundary["mujoco_unitree_g1_simulator_rehearsal_proven"] is True  # type: ignore[index]
    assert proof_boundary["generated_world_rank_fidelity_result_proven"] is False  # type: ignore[index]
    assert proof_boundary["non_ranking_operational_claim_validated"] is False  # type: ignore[index]
    assert proof_boundary["runpod_api_call_performed"] is False  # type: ignore[index]
    assert proof_boundary["runpod_live_execution_api_call_performed"] is False  # type: ignore[index]
    assert proof_boundary["runpod_shutdown_or_termination_proof"] is False  # type: ignore[index]
    assert proof_boundary["raw_secrets_persisted"] is False  # type: ignore[index]
    assert Path(persisted["artifacts"]["report"]).is_file()  # type: ignore[index]
    assert Path(persisted["artifacts"]["external_input_packet"]).is_file()  # type: ignore[index]
    external_packet = _read_json(Path(persisted["artifacts"]["external_input_packet"]))  # type: ignore[index]
    missing_ids = {item["input_id"] for item in external_packet["missing_inputs"]}  # type: ignore[index]
    assert "physical_robot_run_package" in missing_ids
    assert "real_robot_pov_manifest" in missing_ids
    assert "robot_team_policy_package" in missing_ids
    assert "production_runpod_worker_execution_package" in missing_ids
    assert "production_webapp_forwarding_sync_package" in missing_ids
    worker_rehearsal = persisted["same_entrypoint_worker_rehearsal"]  # type: ignore[index]
    assert worker_rehearsal["performed"] is True  # type: ignore[index]
    assert worker_rehearsal["runtime_preflight_detail_status"] == "passed"  # type: ignore[index]
    assert worker_rehearsal["job_status"] == "blocked"  # type: ignore[index]
    assert "worker_job_status:blocked" in worker_rehearsal["blockers"]  # type: ignore[index]
    assert "blocked_rights_privacy" in worker_rehearsal["blockers"]  # type: ignore[index]
    assert proof_boundary["same_entrypoint_worker_rehearsal_performed"] is True  # type: ignore[index]
    assert proof_boundary["same_entrypoint_worker_rehearsal_completed"] is False  # type: ignore[index]
    assert proof_boundary["container_worker_image_rehearsal_performed"] is True  # type: ignore[index]
    assert proof_boundary["container_worker_image_runtime_preflight_executed"] is True  # type: ignore[index]
    assert proof_boundary["container_worker_image_runtime_preflight_passed"] is False  # type: ignore[index]
    assert proof_boundary["published_worker_image_ref_proven"] is False  # type: ignore[index]
    runpod_spend_boundary = persisted["runpod_spend_boundary"]  # type: ignore[index]
    assert runpod_spend_boundary["live_execution_proof"]["status"] == "blocked"  # type: ignore[index]
    assert (
        "missing_env_RUNPOD_API_KEY"
        in runpod_spend_boundary["live_execution_proof"]["blockers"]  # type: ignore[index]
    )
    assert Path(persisted["artifacts"]["runpod_live_execution_proof"]).is_file()  # type: ignore[index]
    container_rehearsal = persisted["container_worker_image_rehearsal"]  # type: ignore[index]
    assert container_rehearsal["performed"] is True  # type: ignore[index]
    assert "nvidia_smi_unavailable" in container_rehearsal["blockers"]  # type: ignore[index]
    gap_audit_path = Path(persisted["artifacts"]["evidence_gap_audit"])  # type: ignore[index]
    assert gap_audit_path.is_file()
    gap_audit = _read_json(gap_audit_path)
    assert gap_audit["conclusion"]["remaining_blockers_are_non_mujoco_external_blockers"] is True  # type: ignore[index]
    assert gap_audit["conclusion"]["customer_through_website_testing_ready"] is False  # type: ignore[index]
    assert gap_audit["requirements"]["production_runpod_worker_execution"]["proven"] is False  # type: ignore[index]
    assert gap_audit["requirements"]["local_container_worker_image_rehearsal"]["external_blocker"] is True  # type: ignore[index]


def test_realistic_rehearsal_records_official_g1_policy_candidate_without_performance_claim(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_complete_mujoco_rehearsal(capture_root)
    build_g1_controlled_proof_setup(capture_root=capture_root, job_id=job_id)

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    persisted = _read_json(Path(manifest["artifacts"]["manifest"]))  # type: ignore[index]
    matrix = persisted["requested_proof_matrix"]  # type: ignore[index]
    policy = matrix["robot_team_policy_performance"]  # type: ignore[index]
    input_artifacts = {item["name"]: item for item in persisted["input_artifacts"]}  # type: ignore[index]

    assert (
        policy["status"]
        == "not_proven_official_unitree_g1_candidate_selected_but_not_executed"
    )
    assert policy["proven"] is False
    assert "official_unitree_g1_policy_candidate_not_executed" in policy["blockers"]
    assert "missing_non_default_policy_execution_trace" in policy["blockers"]
    assert "official_unitree_g1_policy_candidate.json" in json.dumps(policy["evidence"])
    assert input_artifacts["official_unitree_g1_policy_candidate"]["exists"] is True
    assert (
        input_artifacts["official_unitree_g1_policy_candidate"]["status"]
        == "candidate_selected_execution_required"
    )
    assert persisted["proof_boundary"]["robot_team_policy_performance_proven"] is False  # type: ignore[index]


def test_realistic_rehearsal_records_official_g1_policy_execution_without_owner_acceptance(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_complete_mujoco_rehearsal(capture_root)
    build_g1_controlled_proof_setup(capture_root=capture_root, job_id=job_id)
    execution_path = _seed_successful_official_policy_execution(capture_root)

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    persisted = _read_json(Path(manifest["artifacts"]["manifest"]))  # type: ignore[index]
    policy = persisted["requested_proof_matrix"]["robot_team_policy_performance"]  # type: ignore[index]
    input_artifacts = {item["name"]: item for item in persisted["input_artifacts"]}  # type: ignore[index]

    assert (
        policy["status"]
        == "not_proven_official_unitree_g1_policy_executed_owner_acceptance_required"
    )
    assert policy["proven"] is False
    assert policy["blockers"] == ["missing_robot_team_owner_acceptance_or_review"]
    assert str(execution_path) in policy["evidence"]
    assert input_artifacts["official_unitree_g1_policy_execution"]["exists"] is True
    assert (
        persisted["artifacts"]["official_unitree_g1_policy_execution"]
        == str(execution_path)
    )
    assert persisted["proof_boundary"]["robot_team_policy_performance_proven"] is False  # type: ignore[index]


def test_realistic_rehearsal_promotes_ready_physical_g1_evidence_to_proof_matrix(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    evidence_dir = tmp_path / "physical-g1-evidence"
    _seed_job_request(capture_root, job_id)
    _seed_complete_mujoco_rehearsal(capture_root)
    _seed_provider_blockers(capture_root, job_id)
    _seed_ready_physical_g1_evidence_drop(evidence_dir, job_id)
    assembly = assemble_g1_controlled_run_evidence(
        capture_root=capture_root,
        evidence_dir=evidence_dir,
        job_id=job_id,
    )

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    assert assembly["status"] == "ready_for_live_input_staging"
    matrix = manifest["requested_proof_matrix"]
    assert matrix["generated_world_rank_fidelity"]["proven"] is True  # type: ignore[index]
    assert matrix["generated_world_rank_fidelity"]["status"] == "proven"  # type: ignore[index]
    assert matrix["non_ranking_operational_claim"]["proven"] is True  # type: ignore[index]
    assert matrix["non_ranking_operational_claim"]["status"] == "proven"  # type: ignore[index]
    assert matrix["real_robot_pov"]["proven"] is True  # type: ignore[index]
    assert matrix["real_robot_pov"]["status"] == "proven"  # type: ignore[index]
    assert matrix["robot_team_policy_performance"]["proven"] is True  # type: ignore[index]
    assert matrix["robot_team_policy_performance"]["status"] == "proven"  # type: ignore[index]
    assert manifest["proof_boundary"]["generated_world_rank_fidelity_result_proven"] is True  # type: ignore[index]
    assert manifest["proof_boundary"]["non_ranking_operational_claim_validated"] is True  # type: ignore[index]
    assert manifest["proof_boundary"]["real_robot_pov_evidence_proven"] is True  # type: ignore[index]
    assert manifest["proof_boundary"]["robot_team_policy_performance_proven"] is True  # type: ignore[index]
    assert "missing_physical_robot_run_manifest" not in manifest["non_mujoco_external_blockers"]  # type: ignore[index]
    assert "missing_robot_team_owner_acceptance_or_review" not in manifest["non_mujoco_external_blockers"]  # type: ignore[index]
    gap_audit = _read_json(Path(manifest["artifacts"]["evidence_gap_audit"]))  # type: ignore[index]
    assert gap_audit["requirements"]["generated_world_rank_fidelity"]["proven"] is True  # type: ignore[index]
    assert gap_audit["requirements"]["real_robot_pov"]["proven"] is True  # type: ignore[index]


def test_realistic_rehearsal_prefers_signed_proven_runpod_execution(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_complete_mujoco_rehearsal(capture_root)
    _seed_provider_blockers(capture_root, job_id)
    signed_proof = (
        capture_root
        / "pipeline"
        / "g1_controlled_proof_setup"
        / "signed_runpod_io"
        / "runpod_live_execution_proof.auditfix-g1-amd64.stop.with-runtime.json"
    )
    _write_json(
        signed_proof,
        {
            "schema_version": "runpod_live_execution_proof.v1",
            "status": "runpod_live_proof_collected",
            "blockers": [],
            "api_call_performed": True,
            "runpod_side_effects_may_have_occurred": True,
            "active_pod_count_before": 1,
            "active_pod_count_after": 0,
            "shutdown_or_termination_proof": True,
            "runtime_manifest_worker_completed": True,
            "production_runpod_worker_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    production = manifest["requested_proof_matrix"]["production_runpod_worker_execution"]
    proof_boundary = manifest["proof_boundary"]
    live_execution = manifest["runpod_spend_boundary"]["live_execution_proof"]
    assert production["proven"] is True  # type: ignore[index]
    assert production["status"] == "proven"  # type: ignore[index]
    assert production["blockers"] == []  # type: ignore[index]
    assert str(signed_proof) in production["evidence"]  # type: ignore[index]
    assert proof_boundary["production_runpod_worker_execution_proven"] is True  # type: ignore[index]
    assert proof_boundary["published_worker_image_ref_proven"] is True  # type: ignore[index]
    assert proof_boundary["generated_world_rank_fidelity_result_proven"] is False  # type: ignore[index]
    assert manifest["current_proof_state"]["gates"]["production_runpod_worker_execution"]["proven"] is True  # type: ignore[index]
    assert live_execution["path"] == str(signed_proof)  # type: ignore[index]
    assert live_execution["production_runpod_worker_execution_proven"] is True  # type: ignore[index]
    assert live_execution["simulator_execution_proven"] is True  # type: ignore[index]
    assert manifest["runpod_spend_boundary"]["reason_live_run_not_attempted"] is None  # type: ignore[index]
    gap_audit = _read_json(Path(manifest["artifacts"]["evidence_gap_audit"]))  # type: ignore[index]
    assert gap_audit["requirements"]["production_runpod_worker_execution"]["proven"] is True  # type: ignore[index]
    assert gap_audit["requirements"]["production_runpod_worker_execution"]["missing_inputs"] == []  # type: ignore[index]


def test_realistic_rehearsal_surfaces_live_webapp_route_blockers(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_complete_mujoco_rehearsal(capture_root)
    _seed_provider_blockers(capture_root, job_id)
    _seed_blocked_webapp_route_proof(capture_root)

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    website = manifest["requested_proof_matrix"]["customer_through_website_testing_ready"]
    assert website["status"] == "not_ready_production_webapp_route_blocked"  # type: ignore[index]
    assert "webapp:request_capture_root_does_not_match_control_plane" in website["blockers"]  # type: ignore[index]
    assert "production_webapp_pipeline_forward_not_accepted" in website["blockers"]  # type: ignore[index]
    assert website["evidence"]  # type: ignore[index]


def test_realistic_rehearsal_prefers_latest_webapp_route_attempt(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    proof_dir = capture_root / "pipeline" / "webapp_route_forwarding_proof"
    _seed_complete_mujoco_rehearsal(capture_root)
    _seed_provider_blockers(capture_root, job_id)
    _seed_blocked_webapp_route_proof(capture_root)
    latest_proof = proof_dir / "webapp_route_forwarding_proof.production-path-g1.json"
    _write_json(
        latest_proof,
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "generated_at": "2026-06-12T23:57:15.188Z",
            "status": "blocked",
            "webapp_route": {
                "full_production_webapp_deployment_proven": True,
                "http_status": 202,
            },
            "pipeline_forward": {
                "status": "forwarded",
                "performed": True,
                "accepted": False,
                "pipeline_status": "blocked",
            },
            "pipeline_intake": {
                "accepted": False,
                "status": "blocked",
                "input_blockers": [
                    "webapp:g1_capture_root_does_not_match_active_control_plane",
                ],
            },
            "proof_boundary": {
                "production_live_webapp_forwarding_proven": False,
                "pipeline_intake_staged_request_proven": False,
                "full_webapp_db_persistence_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    website = manifest["requested_proof_matrix"]["customer_through_website_testing_ready"]
    assert str(latest_proof) in website["evidence"]  # type: ignore[index]
    assert "webapp:g1_capture_root_does_not_match_active_control_plane" in website["blockers"]  # type: ignore[index]


def test_realistic_rehearsal_marks_customer_website_ready_when_production_route_stages(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_complete_mujoco_rehearsal(capture_root)
    _seed_provider_blockers(capture_root, job_id)
    proof_path = _seed_successful_webapp_route_proof(capture_root)
    _write_json(
        capture_root / "pipeline" / "production_handoff_readiness_manifest.json",
        {
            "schema_version": "production_handoff_readiness_manifest.v1",
            "status": "blocked",
            "blockers": [
                "production_live_webapp_forwarding_not_proven",
                "unrelated_external_owner_blocker",
            ],
        },
    )

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    website = manifest["requested_proof_matrix"]["customer_through_website_testing_ready"]
    assert website["proven"] is True  # type: ignore[index]
    assert website["blockers"] == []  # type: ignore[index]
    assert str(proof_path) in website["evidence"]  # type: ignore[index]
    assert manifest["proof_boundary"]["customer_through_website_testing_ready"] is True  # type: ignore[index]
    assert (
        "production_live_webapp_forwarding_not_proven"
        not in manifest["non_mujoco_external_blockers"]  # type: ignore[index]
    )
    assert "unrelated_external_owner_blocker" in manifest["non_mujoco_external_blockers"]  # type: ignore[index]
    gap_audit = _read_json(Path(manifest["artifacts"]["evidence_gap_audit"]))  # type: ignore[index]
    assert gap_audit["conclusion"]["customer_through_website_testing_ready"] is True  # type: ignore[index]


def test_realistic_rehearsal_summarizes_two_live_gates_proven_with_physical_gates_open(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_complete_mujoco_rehearsal(capture_root)
    _seed_provider_blockers(capture_root, job_id)
    _seed_successful_webapp_route_proof(capture_root)
    signed_proof = (
        capture_root
        / "pipeline"
        / "g1_controlled_proof_setup"
        / "signed_runpod_io"
        / "runpod_live_execution_proof.auditfix-g1-amd64.stop.with-runtime.json"
    )
    _write_json(
        signed_proof,
        {
            "schema_version": "runpod_live_execution_proof.v1",
            "status": "runpod_live_proof_collected",
            "blockers": [],
            "api_call_performed": True,
            "runpod_side_effects_may_have_occurred": True,
            "active_pod_count_before": 0,
            "active_pod_count_after": 0,
            "shutdown_or_termination_proof": True,
            "runtime_manifest_worker_completed": True,
            "production_runpod_worker_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    state = manifest["current_proof_state"]
    assert state["proven_gate_count"] == 2  # type: ignore[index]
    assert state["remaining_gate_count"] == 4  # type: ignore[index]
    assert set(state["proven"]) == {  # type: ignore[index]
        "production_runpod_worker_execution",
        "customer_through_website_testing_ready",
    }
    assert set(state["not_proven"]) == {  # type: ignore[index]
        "generated_world_rank_fidelity",
        "non_ranking_operational_claim",
        "real_robot_pov",
        "robot_team_policy_performance",
    }
    next_inputs = " ".join(state["next_external_inputs"])  # type: ignore[index]
    assert "RunPod" not in next_inputs
    assert "WebApp" not in next_inputs
    assert "real Unitree G1 run package" in next_inputs
    assert "reviewed safety/contact/threshold evidence" in next_inputs
    report = Path(manifest["artifacts"]["report"]).read_text(encoding="utf-8")  # type: ignore[index]
    assert "Live-product gates proven: `2/6`" in report
    assert "Production RunPod worker execution: proven" in report
    assert "Customer-through-website readiness: proven" in report


def test_realistic_rehearsal_fails_closed_when_mujoco_manifest_missing(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root)

    matrix = manifest["requested_proof_matrix"]
    assert manifest["status"] == "blocked_simulator_rehearsal_incomplete"
    assert matrix["mujoco_unitree_g1_simulator_rehearsal"]["proven"] is False  # type: ignore[index]
    assert manifest["default_robot"]["make_model"] == "Unitree G1"  # type: ignore[index]
    assert (
        "local_mujoco_rehearsal_missing_or_incomplete"
        in matrix["mujoco_unitree_g1_simulator_rehearsal"]["blockers"]  # type: ignore[index]
    )
    assert manifest["proof_boundary"]["public_claim_upgrade_allowed"] is False  # type: ignore[index]
    assert Path(manifest["artifacts"]["external_input_packet"]).is_file()  # type: ignore[index]
    assert Path(manifest["artifacts"]["evidence_gap_audit"]).is_file()  # type: ignore[index]


def test_realistic_rehearsal_surfaces_worker_preflight_blockers(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    worker_root = capture_root / "pipeline" / "realistic_readiness_rehearsal" / "same_entrypoint_worker_rehearsal"
    _seed_complete_mujoco_rehearsal(capture_root)
    _seed_provider_blockers(capture_root, job_id)
    _write_json(
        worker_root / "worker_runtime_manifest.json",
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "status": "blocked",
            "blockers": ["worker_runtime_preflight_blocked"],
            "runtime_preflight_status": "blocked",
            "runtime_preflight_blockers": ["runtime_preflight_command_failed"],
            "artifact_upload": {"status": "completed"},
        },
    )
    _write_json(
        worker_root / "worker_runtime_preflight_detail.json",
        {
            "schema_version": "mujoco_worker_runtime_preflight.v1",
            "status": "blocked",
            "blockers": ["nvidia_smi_unavailable"],
            "proof_boundary": {"runtime_preflight_executed": True},
        },
    )

    manifest = build_realistic_readiness_rehearsal(capture_root=capture_root, job_id=job_id)

    worker_rehearsal = manifest["same_entrypoint_worker_rehearsal"]
    assert worker_rehearsal["runtime_preflight_status"] == "blocked"  # type: ignore[index]
    assert "worker_runtime_preflight_blocked" in worker_rehearsal["blockers"]  # type: ignore[index]
    assert "runtime_preflight_command_failed" in worker_rehearsal["blockers"]  # type: ignore[index]
    assert "nvidia_smi_unavailable" in worker_rehearsal["blockers"]  # type: ignore[index]


def test_realistic_rehearsal_cli_writes_manifest(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    _seed_complete_mujoco_rehearsal(capture_root)

    exit_code = main(["--capture-root", str(capture_root)])

    assert exit_code == 0
    assert (
        capture_root
        / "pipeline"
        / "realistic_readiness_rehearsal"
        / "realistic_readiness_rehearsal_manifest.json"
    ).is_file()


def test_realistic_rehearsal_g1_assembly_helpers_collect_artifact_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ready_artifact = tmp_path / "ready.json"
    invalid_artifact = tmp_path / "invalid.json"
    blocked_artifact = tmp_path / "blocked.json"
    _write_json(ready_artifact, {"status": rehearsal.G1_READY_FOR_LIVE_STAGING_STATUS})
    invalid_artifact.write_text("{}", encoding="utf-8")
    _write_json(blocked_artifact, {"status": "blocked", "blockers": ["nested_blocker"]})
    original_optional_read_json = rehearsal.optional_read_json

    def read_none_for_invalid(path: Path):
        if path == invalid_artifact:
            return None
        return original_optional_read_json(path)

    monkeypatch.setattr(rehearsal, "optional_read_json", read_none_for_invalid)

    ready, evidence, blockers = rehearsal._ready_g1_assembly_artifacts(
        assembly_manifest={
            "status": "blocked",
            "blockers": ["assembly_blocker"],
            "file_blockers": ["file_blocker"],
            "config_blockers": ["config_blocker"],
            "content_blockers": ["content_blocker"],
            "artifacts": {
                "ready": str(ready_artifact),
                "missing_file": str(tmp_path / "missing.json"),
                "invalid": str(invalid_artifact),
                "blocked": str(blocked_artifact),
            },
        },
        assembly_path=tmp_path / "assembly.json",
        artifact_keys=["ready", "missing_ref", "missing_file", "invalid", "blocked"],
    )

    assert ready is False
    assert str(ready_artifact) in evidence
    assert "g1_controlled_run_evidence_assembly_not_ready" in blockers
    assert "missing_g1_assembly_artifact_ref:missing_ref" in blockers
    assert "missing_g1_assembly_artifact:missing_file" in blockers
    assert "invalid_g1_assembly_artifact_json:invalid" in blockers
    assert "g1_assembly_artifact_not_ready:blocked:blocked" in blockers
    assert "nested_blocker" in blockers


def test_realistic_rehearsal_container_and_webapp_route_helpers_block_unproven_inputs(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    jobs_root = capture_root / "pipeline" / "robot_eval_jobs"

    assert rehearsal._find_primary_job_id(capture_root, None) is None
    (jobs_root / "job-c").mkdir(parents=True)
    assert rehearsal._find_primary_job_id(capture_root, None) == "job-c"
    _write_json(jobs_root / "job-b" / "job_request.json", {"schema_version": "robot_eval_job_request.v1"})
    assert rehearsal._find_primary_job_id(capture_root, None) == "job-b"
    _write_json(jobs_root / "job-a" / "runpod_provider_adapter_result.json", {"status": "completed"})
    assert rehearsal._find_primary_job_id(capture_root, None) == "job-a"

    blockers = rehearsal._container_rehearsal_blockers(
        {"job_status": "failed"},
        None,
    )
    assert "worker_job_status:failed" in blockers

    proof_path = tmp_path / "webapp_route_forwarding_proof.json"
    _write_json(
        proof_path,
        {
            "proof_boundary": {
                "pipeline_intake_staged_request_proven": True,
                "full_webapp_db_persistence_proven": True,
                "production_live_webapp_forwarding_proven": True,
            },
            "webapp_route": {},
            "pipeline_forward": {"accepted": True},
            "pipeline_intake": {},
        },
    )

    item = rehearsal._webapp_route_proof_item(
        proof_path=proof_path,
        proof=_read_json(proof_path),
    )

    assert item["proven"] is True
    assert "production_webapp_route_not_proven" in item["blockers"]
    assert str(proof_path) in item["evidence"]
