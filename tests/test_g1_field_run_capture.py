from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.g1_field_run_capture import (
    G1_FIELD_RUN_CAPTURE_KIT_SCHEMA_VERSION,
    build_g1_field_run_capture_kit,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_job_request(capture_root: Path, job_id: str) -> None:
    _write_json(
        capture_root / "pipeline" / "robot_eval_jobs" / job_id / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "requested_tasks": [
                {
                    "task_id": "walk_to_target",
                    "scenario_ids": ["site-a_walk_to_target_pose"],
                }
            ],
            "site_package": {
                "site_slug": "site-a",
                "site_submission_id": "site-submission-123",
                "buyer_request_id": "buyer-123",
                "capture_job_id": "capture-job-123",
            },
        },
    )


def _seed_successful_webapp_route_proof(capture_root: Path) -> None:
    _write_json(
        capture_root
        / "pipeline"
        / "webapp_route_forwarding_proof"
        / "webapp_route_forwarding_proof.production-path-g1.json",
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "status": "forwarded_to_pipeline_intake",
            "webapp_route": {
                "route_url": "https://www.tryblueprint.io/api/robot-eval/job-requests",
                "http_status": 202,
                "full_production_webapp_deployment_proven": True,
            },
            "pipeline_forward": {
                "performed": True,
                "accepted": True,
                "pipeline_status": "staged_for_control_plane",
            },
            "pipeline_intake": {
                "accepted": True,
                "status": "staged_for_control_plane",
                "input_blockers": [],
            },
            "job_request": {
                "job_id": "robot-eval-production-route-123",
                "buyer_request_id": "buyer-live-123",
                "requested_tasks": [
                    {
                        "task_id": "walk_to_target",
                        "scenario_ids": ["site-live_walk_to_target_pose"],
                    }
                ],
                "site_package": {
                    "site_slug": "site-live",
                    "site_submission_id": "site-live-123",
                    "capture_job_id": "capture-live-123",
                    "capture_id": "capture-id-live-123",
                },
            },
            "durable_store": {
                "status": "stored",
                "firestore": {
                    "doc_id": "robot-eval-production-route-123",
                },
            },
            "proof_boundary": {
                "production_live_webapp_forwarding_proven": True,
                "pipeline_intake_staged_request_proven": True,
                "full_webapp_db_persistence_proven": True,
            },
        },
    )


def test_g1_field_run_capture_kit_writes_operator_packet_without_secrets(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_job_request(capture_root, job_id)

    manifest = build_g1_field_run_capture_kit(capture_root=capture_root, job_id=job_id)

    assert manifest["schema_version"] == G1_FIELD_RUN_CAPTURE_KIT_SCHEMA_VERSION
    assert manifest["status"] == "field_run_capture_ready_operator_inputs_required"
    assert manifest["default_robot"]["make_model"] == "Unitree G1"  # type: ignore[index]
    artifacts = manifest["artifacts"]  # type: ignore[assignment]
    config = _read_json(Path(artifacts["config"]))  # type: ignore[index]
    assert config["robot_ip"] == "192.168.123.164"
    assert config["job_context"]["job_id"] == job_id  # type: ignore[index]
    assert config["job_context"]["job_request_source"] == "robot_eval_jobs"  # type: ignore[index]
    assert config["job_context"]["task_id"] == "walk_to_target"  # type: ignore[index]
    assert config["job_context"]["scenario_id"] == "site-a_walk_to_target_pose"  # type: ignore[index]
    assert config["job_context"]["buyer_request_id"] == "buyer-123"  # type: ignore[index]
    assert config["job_context"]["site_submission_id"] == "site-submission-123"  # type: ignore[index]
    assert config["job_context"]["capture_job_id"] == "capture-job-123"  # type: ignore[index]
    assert config["job_context"]["robot_profile_id"] == "unitree_g1_humanoid"  # type: ignore[index]
    assert config["policy_id"] == "unitree_rl_gym_g1_mujoco_policy_candidate"
    assert config["task_id"] == "walk_to_target"
    assert config["scenario_eval_run_id"] == "robot-eval-test-site-a_walk_to_target_pose"
    assert config["scenario_variation_instance_id"] == "site-a_walk_to_target_pose"
    assert config["allowed_task_set"][0]["policy_id"] == (  # type: ignore[index]
        "unitree_rl_gym_g1_mujoco_policy_candidate"
    )
    assert (
        config["exclusion_and_abort_criteria"]["loose_or_inferred_anchor_matches_allowed"]  # type: ignore[index]
        is False
    )
    assert config["policy"]["source_repo"] == "https://github.com/unitreerobotics/unitree_rl_gym"  # type: ignore[index]
    assert config["policy"]["fallback_source_repo"] == "https://github.com/unitreerobotics/unitree_rl_lab"  # type: ignore[index]
    assert config["policy"]["sim_bridge_repo"] == "https://github.com/unitreerobotics/unitree_mujoco"  # type: ignore[index]
    assert config["timestamp_alignment"]["max_alignment_error_ms"] == 100  # type: ignore[index]
    assert config["timestamp_alignment"]["camera_timebase"] == "ffmpeg_capture_wall_clock"  # type: ignore[index]
    assert config["timestamp_alignment"]["robot_action_log_source"] == "action_log.jsonl"  # type: ignore[index]
    assert config["real_robot_pov_contract"]["physical_source_required"] is True  # type: ignore[index]
    assert config["real_robot_pov_contract"]["simulator_frames_count_as_real_pov"] is False  # type: ignore[index]
    assert config["real_robot_pov_contract"]["test_fixture_policy"]["synthetic_media_can_upgrade_readiness"] is False  # type: ignore[index]
    assert config["required_owner_evidence_before_physical_claim"] == [
        "robot_camera_video",
        "action_log",
        "timestamp_alignment",
        "hardware_validation",
        "contact_collision_log",
        "policy_metrics",
        "robot_team_review",
    ]
    assert config["attestation_requirements"]["operator_attestation_signed"] is False  # type: ignore[index]
    assert config["attestation_requirements"]["hardware_owner_attestation_signed"] is False  # type: ignore[index]
    assert config["attestation_requirements"]["safety_reviewer_attestation_signed"] is False  # type: ignore[index]
    assert config["attestation_requirements"]["robot_team_review_attestation_signed"] is False  # type: ignore[index]
    assert config["commands"]["required_env"] == [  # type: ignore[index]
        "BLUEPRINT_G1_CAMERA_SOURCE",
        "BLUEPRINT_G1_POLICY_COMMAND",
        "BLUEPRINT_G1_ACTION_LOG_COMMAND",
        "BLUEPRINT_G1_STATE_COMMAND",
        "BLUEPRINT_G1_CONTACT_COLLISION_COMMAND",
    ]
    templates = config["commands"]["official_unitree_real_policy_templates"]  # type: ignore[index]
    assert (
        templates["python_deploy_real"]
        == "python deploy/deploy_real/deploy_real.py ${BLUEPRINT_G1_NET_INTERFACE:-enp3s0} g1.yaml"
    )
    assert templates["cpp_g1_deploy_run"] == "./g1_deploy_run ${BLUEPRINT_G1_NET_INTERFACE:-enp3s0}"
    assert templates["source_repo"] == "https://github.com/unitreerobotics/unitree_rl_gym"
    logger_templates = config["commands"]["blueprint_dds_logger_templates"]  # type: ignore[index]
    assert "record_g1_dds_logs.py --mode action" in logger_templates["action_log_jsonl"]
    assert "record_g1_dds_logs.py --mode state" in logger_templates["robot_state_jsonl"]
    assert "record_g1_dds_logs.py --mode contact" in logger_templates["contact_collision_json"]
    review_templates = config["commands"]["review_command_templates"]  # type: ignore[index]
    assert "blueprint-review-g1-field-run-evidence" in review_templates["blocked_dry_review"]
    assert "--accept-safety --accept-policy --require-ready" in review_templates[
        "accepted_review_after_human_signoff"
    ]
    expected = _read_json(Path(artifacts["evidence_manifest"]))  # type: ignore[index]
    assert "robot_camera_video.mp4" in expected["required_files"]  # type: ignore[index]
    assert expected["required_exact_join_keys"] == [
        "scenario_eval_run_id",
        "policy_id",
        "task_id",
        "scenario_variation_instance_id",
    ]
    assert expected["expected_anchor_join_key"] == {
        "scenario_eval_run_id": "robot-eval-test-site-a_walk_to_target_pose",
        "policy_id": "unitree_rl_gym_g1_mujoco_policy_candidate",
        "task_id": "walk_to_target",
        "scenario_variation_instance_id": "site-a_walk_to_target_pose",
    }
    assert expected["required_owner_evidence_before_physical_claim"] == [
        "robot_camera_video",
        "action_log",
        "timestamp_alignment",
        "hardware_validation",
        "contact_collision_log",
        "policy_metrics",
        "robot_team_review",
    ]
    assert expected["required_signed_attestations_before_physical_claim"] == [
        "operator_attestation_signed",
        "hardware_owner_attestation_signed",
        "safety_reviewer_attestation_signed",
        "robot_team_review_attestation_signed",
    ]
    assert expected["physical_claim_gate"] == {  # type: ignore[index]
        "templates_are_not_evidence": True,
        "exact_join_keys_required": True,
        "owner_evidence_required": True,
        "unsigned_attestations_fail_closed": True,
        "sim_only_policy_comparison_blocked_by_missing_physical_evidence": False,
    }
    assert expected["contracts"]["controlled_field_anchor_request_packet"] == artifacts["controlled_field_anchor_request_packet"]  # type: ignore[index]
    assert expected["contracts"]["real_robot_pov_capture_contract"] == artifacts["real_robot_pov_capture_contract"]  # type: ignore[index]
    assert expected["contracts"]["safety_review_checklist"] == artifacts["safety_review_checklist"]  # type: ignore[index]
    assert expected["required_live_commands"] == [  # type: ignore[index]
        "BLUEPRINT_G1_CAMERA_SOURCE",
        "BLUEPRINT_G1_POLICY_COMMAND",
        "BLUEPRINT_G1_ACTION_LOG_COMMAND",
        "BLUEPRINT_G1_STATE_COMMAND",
        "BLUEPRINT_G1_CONTACT_COLLISION_COMMAND",
    ]
    script = Path(artifacts["capture_script"]).read_text(encoding="utf-8")  # type: ignore[index]
    assert "BLUEPRINT_ALLOW_G1_PHYSICAL_RUN" in script
    assert "BLUEPRINT_G1_CAMERA_SOURCE" in script
    assert "BLUEPRINT_G1_POLICY_COMMAND" in script
    assert "Missing BLUEPRINT_G1_ACTION_LOG_COMMAND" in script
    assert "Missing BLUEPRINT_G1_STATE_COMMAND" in script
    assert "Missing BLUEPRINT_G1_CONTACT_COLLISION_COMMAND" in script
    assert "ACTION_LOG_PID=$!" in script
    assert "STATE_PID=$!" in script
    assert "CONTACT_PID=$!" in script
    assert script.index("ACTION_LOG_PID=$!") < script.index('python - "$POLICY_COMMAND"')
    assert script.index("STATE_PID=$!") < script.index('python - "$POLICY_COMMAND"')
    assert script.index("CONTACT_PID=$!") < script.index('python - "$POLICY_COMMAND"')
    assert "g1_controlled_run_inputs.json" in script
    readme = Path(artifacts["readme"]).read_text(encoding="utf-8")  # type: ignore[index]
    assert "controlled_field_anchor_request_packet.json" in readme
    assert "real_robot_pov_capture_contract.json" in readme
    assert "safety_review_checklist.json" in readme
    assert "Job request source: `robot_eval_jobs`" in readme
    assert "Scenario: `site-a_walk_to_target_pose`" in readme
    assert "deploy/deploy_real/deploy_real.py" in readme
    assert "g1_deploy_run" in readme
    assert "blueprint-review-g1-field-run-evidence" in readme
    assert "operator_review_required" in script
    assert artifacts["review_manifest"].endswith("g1_field_run_review_manifest.json")  # type: ignore[index]
    logger_script = Path(artifacts["dds_logger_script"])  # type: ignore[index]
    assert logger_script.is_file()
    logger_source = logger_script.read_text(encoding="utf-8")
    assert "ChannelFactoryInitialize" in logger_source
    assert "rt/lowcmd" in logger_source
    assert "rt/lowstate" in logger_source
    assert "rpa_" not in json.dumps(manifest)
    assert "rpa_" not in script
    assert manifest["proof_boundary"]["requires_real_g1_hardware"] is True  # type: ignore[index]
    anchor_request = _read_json(Path(artifacts["controlled_field_anchor_request_packet"]))  # type: ignore[index]
    assert anchor_request["status"] == "not_requested_for_sim_only"
    assert anchor_request["blockers"] == []
    assert anchor_request["required_exact_join_keys"] == [
        "scenario_eval_run_id",
        "policy_id",
        "task_id",
        "scenario_variation_instance_id",
    ]
    assert anchor_request["anchor_join_key"] == {
        "scenario_eval_run_id": "robot-eval-test-site-a_walk_to_target_pose",
        "policy_id": "unitree_rl_gym_g1_mujoco_policy_candidate",
        "task_id": "walk_to_target",
        "scenario_variation_instance_id": "site-a_walk_to_target_pose",
    }
    assert anchor_request["loose_or_inferred_matches_allowed_for_calibration"] is False
    pov_contract = _read_json(Path(artifacts["real_robot_pov_capture_contract"]))  # type: ignore[index]
    assert pov_contract["job_context"]["job_id"] == job_id  # type: ignore[index]
    assert pov_contract["job_context"]["site_submission_id"] == "site-submission-123"  # type: ignore[index]
    assert pov_contract["job_context"]["scenario_id"] == "site-a_walk_to_target_pose"  # type: ignore[index]
    assert pov_contract["proof_boundary"]["requires_physical_robot_camera_or_sensor_evidence"] is True  # type: ignore[index]
    assert pov_contract["test_fixture_policy"]["synthetic_media_can_prove_real_robot_pov"] is False  # type: ignore[index]
    safety_checklist = _read_json(Path(artifacts["safety_review_checklist"]))  # type: ignore[index]
    assert safety_checklist["job_context"]["job_id"] == job_id  # type: ignore[index]
    assert safety_checklist["job_context"]["buyer_request_id"] == "buyer-123"  # type: ignore[index]
    assert safety_checklist["job_context"]["task_id"] == "walk_to_target"  # type: ignore[index]
    assert "contact_collision_log.json" in safety_checklist["required_files"]  # type: ignore[index]
    assert "explicit --accept-safety reviewer action" in safety_checklist["required_review_decisions"]  # type: ignore[index]


def test_g1_field_run_capture_kit_prefills_proven_webapp_route_fields(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_job_request(capture_root, job_id)
    _seed_successful_webapp_route_proof(capture_root)

    manifest = build_g1_field_run_capture_kit(capture_root=capture_root, job_id=job_id)

    artifacts = manifest["artifacts"]  # type: ignore[assignment]
    config = _read_json(Path(artifacts["config"]))  # type: ignore[index]
    assert config["production_webapp_request_id"] == "robot-eval-production-route-123"
    assert config["pipeline_intake_request_id"] == "robot-eval-production-route-123"
    assert (
        config["production_forward_url"]
        == "https://www.tryblueprint.io/api/robot-eval/job-requests"
    )
    assert config["webapp_response_status_code"] == "202"
    assert config["sync_status"] == "succeeded"


def test_g1_field_run_capture_kit_uses_production_webapp_job_when_local_job_missing(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    _seed_successful_webapp_route_proof(capture_root)

    manifest = build_g1_field_run_capture_kit(
        capture_root=capture_root,
        job_id="robot-eval-production-route-123",
    )

    context = manifest["job_context"]  # type: ignore[assignment]
    assert context["job_id"] == "robot-eval-production-route-123"
    assert context["job_request_found"] is True
    assert context["job_request_source"] == "webapp_route_forwarding_proof"
    assert context["site_slug"] == "site-live"
    assert context["site_submission_id"] == "site-live-123"
    assert context["buyer_request_id"] == "buyer-live-123"
    assert context["capture_job_id"] == "capture-live-123"
    assert context["capture_id"] == "capture-id-live-123"
    assert context["task_id"] == "walk_to_target"
    assert context["scenario_id"] == "site-live_walk_to_target_pose"
    config = _read_json(Path(manifest["artifacts"]["config"]))  # type: ignore[index]
    assert config["job_id"] == "robot-eval-production-route-123"
    assert config["job_context"]["job_request_source"] == "webapp_route_forwarding_proof"  # type: ignore[index]
    assert config["job_context"]["site_slug"] == "site-live"  # type: ignore[index]
    assert config["job_context"]["buyer_request_id"] == "buyer-live-123"  # type: ignore[index]
    pov_contract = _read_json(Path(manifest["artifacts"]["real_robot_pov_capture_contract"]))  # type: ignore[index]
    assert pov_contract["job_context"]["job_request_source"] == "webapp_route_forwarding_proof"  # type: ignore[index]
    assert pov_contract["job_context"]["site_slug"] == "site-live"  # type: ignore[index]
    safety_checklist = _read_json(Path(manifest["artifacts"]["safety_review_checklist"]))  # type: ignore[index]
    assert safety_checklist["job_context"]["buyer_request_id"] == "buyer-live-123"  # type: ignore[index]
    readme = Path(manifest["artifacts"]["readme"]).read_text(encoding="utf-8")  # type: ignore[index]
    assert "Job request source: `webapp_route_forwarding_proof`" in readme
    assert "Scenario: `site-live_walk_to_target_pose`" in readme


def test_g1_field_run_capture_kit_refreshes_stale_placeholder_webapp_inputs(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    evidence_dir = capture_root / "pipeline" / "g1_controlled_proof_setup" / "physical_g1_evidence_drop"
    _seed_job_request(capture_root, job_id)
    _seed_successful_webapp_route_proof(capture_root)
    _write_json(
        evidence_dir / "g1_controlled_run_inputs.json",
        {
            "schema_version": "g1_controlled_run_inputs.v1",
            "job_id": job_id,
            "operator_id": "operator-filled-value",
            "production_webapp_request_id": "<production-webapp-request-id>",
            "pipeline_intake_request_id": "<pipeline-intake-request-id>",
            "production_forward_url": "<production-forward-url>",
            "webapp_response_status_code": "<202>",
        },
    )

    build_g1_field_run_capture_kit(capture_root=capture_root, job_id=job_id)

    inputs = _read_json(evidence_dir / "g1_controlled_run_inputs.json")
    assert inputs["operator_id"] == "operator-filled-value"
    assert inputs["production_webapp_request_id"] == "robot-eval-production-route-123"
    assert inputs["pipeline_intake_request_id"] == "robot-eval-production-route-123"
    assert (
        inputs["production_forward_url"]
        == "https://www.tryblueprint.io/api/robot-eval/job-requests"
    )
    assert inputs["webapp_response_status_code"] == "202"
    assert inputs["sync_status"] == "succeeded"


def test_g1_field_run_capture_kit_refreshes_placeholder_input_job_id_to_webapp_job(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    evidence_dir = capture_root / "pipeline" / "g1_controlled_proof_setup" / "physical_g1_evidence_drop"
    _seed_successful_webapp_route_proof(capture_root)
    _write_json(
        evidence_dir / "g1_controlled_run_inputs.json",
        {
            "schema_version": "g1_controlled_run_inputs.v1",
            "job_id": "old-local-job",
            "robot_serial_or_fleet_id": "<unitree-g1-serial-or-fleet-id>",
            "operator_id": "<operator-id>",
            "production_webapp_request_id": "<production-webapp-request-id>",
        },
    )

    build_g1_field_run_capture_kit(
        capture_root=capture_root,
        job_id="robot-eval-production-route-123",
    )

    inputs = _read_json(evidence_dir / "g1_controlled_run_inputs.json")
    assert inputs["job_id"] == "robot-eval-production-route-123"
    assert inputs["production_webapp_request_id"] == "robot-eval-production-route-123"


def test_g1_field_run_capture_kit_cli_writes_manifest(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"

    exit_code = main(["--capture-root", str(capture_root)])

    assert exit_code == 0
    assert (
        capture_root
        / "pipeline"
        / "g1_controlled_proof_setup"
        / "field_run_capture_kit"
        / "g1_field_run_capture_kit_manifest.json"
    ).is_file()
