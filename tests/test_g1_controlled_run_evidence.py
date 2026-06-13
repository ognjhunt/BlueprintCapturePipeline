from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.g1_controlled_run_evidence import (
    G1_CONTROLLED_RUN_EVIDENCE_SCHEMA_VERSION,
    assemble_g1_controlled_run_evidence,
    main,
    write_g1_controlled_run_input_template,
)
from blueprint_pipeline.live_pipeline_control_plane import (
    JOB_REQUEST_INBOX_ENV,
    LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
)
from blueprint_pipeline.live_pipeline_input_intake import build_live_pipeline_input_intake


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            "source": {
                "selection_state": {
                    "site_submission_id": "site-submission-123",
                    "request_id": "webapp-request-123",
                    "buyer_request_id": "buyer-123",
                    "capture_job_id": "capture-job-123",
                    "task_id": "walk_to_target",
                    "scenario_id": "site-a_walk_to_target_pose",
                    "source_kind": "owner_agent_codex_request",
                }
            },
        },
    )
    return path


def _seed_evidence_drop(evidence_dir: Path) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "robot_camera_video.mp4").write_bytes(b"fake-mp4-bytes")
    _write_json(
        evidence_dir / "timestamp_alignment.json",
        {
            "schema_version": "g1_timestamp_alignment.v1",
            "max_alignment_error_ms": 50,
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


def _ready_config(job_id: str) -> dict[str, object]:
    return {
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
        "robot_team_review_statement": "Robot team accepted the non-default G1 policy package.",
    }


def test_g1_evidence_assembly_blocks_missing_files(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_job_request(capture_root, job_id)
    evidence_dir = tmp_path / "evidence"
    _write_json(evidence_dir / "g1_controlled_run_inputs.json", _ready_config(job_id))

    manifest = assemble_g1_controlled_run_evidence(
        capture_root=capture_root,
        evidence_dir=evidence_dir,
        job_id=job_id,
    )

    assert manifest["schema_version"] == G1_CONTROLLED_RUN_EVIDENCE_SCHEMA_VERSION
    assert manifest["status"] == "blocked_missing_evidence"
    assert "missing_evidence_file:robot_camera_video" in manifest["blockers"]
    assert Path(manifest["artifacts"]["real_robot_pov_manifest"]).is_file()  # type: ignore[index]
    assert manifest["proof_boundary"]["physical_robot_readiness_proven"] is False  # type: ignore[index]


def test_g1_evidence_assembly_blocks_review_required_content(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_job_request(capture_root, job_id)
    evidence_dir = tmp_path / "evidence"
    _seed_evidence_drop(evidence_dir)
    _write_json(
        evidence_dir / "hardware_validation.json",
        {
            "schema_version": "g1_hardware_validation.v1",
            "status": "operator_review_required",
            "hardware_ready": False,
            "estop_verified": False,
        },
    )
    _write_json(
        evidence_dir / "robot_team_review.json",
        {
            "schema_version": "g1_robot_team_review.v1",
            "review_decision": "not_reviewed",
            "accepted": False,
            "reviewer_id": "robot-team-reviewer-a",
        },
    )
    _write_json(evidence_dir / "g1_controlled_run_inputs.json", _ready_config(job_id))

    manifest = assemble_g1_controlled_run_evidence(
        capture_root=capture_root,
        evidence_dir=evidence_dir,
        job_id=job_id,
    )

    assert manifest["status"] == "blocked_missing_evidence"
    assert "hardware_validation_not_ready" in manifest["content_blockers"]
    assert "hardware_validation_estop_not_verified" in manifest["content_blockers"]
    assert "robot_team_review_not_accepted" in manifest["content_blockers"]


def test_g1_evidence_assembly_blocks_action_log_without_robot_action_record(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    _seed_job_request(capture_root, job_id)
    evidence_dir = tmp_path / "evidence"
    _seed_evidence_drop(evidence_dir)
    (evidence_dir / "action_log.jsonl").write_text(
        json.dumps(
            {
                "kind": "policy_command_action_trace_ref",
                "source": "policy_execution_trace.jsonl",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(evidence_dir / "g1_controlled_run_inputs.json", _ready_config(job_id))

    manifest = assemble_g1_controlled_run_evidence(
        capture_root=capture_root,
        evidence_dir=evidence_dir,
        job_id=job_id,
    )

    assert manifest["status"] == "blocked_missing_evidence"
    assert "action_log_missing_robot_action_record" in manifest["content_blockers"]


def test_g1_evidence_assembly_outputs_live_intake_ready_manifests(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    job_request_path = _seed_job_request(capture_root, job_id)
    evidence_dir = tmp_path / "evidence"
    _seed_evidence_drop(evidence_dir)
    _write_json(evidence_dir / "g1_controlled_run_inputs.json", _ready_config(job_id))

    manifest = assemble_g1_controlled_run_evidence(
        capture_root=capture_root,
        evidence_dir=evidence_dir,
        job_id=job_id,
    )

    assert manifest["status"] == "ready_for_live_input_staging"
    output_dir = Path(manifest["output_dir"])
    pov = _read_json(output_dir / "real_robot_pov_manifest.json")
    assert pov["records"][0]["robot_camera_video_uri"].endswith("robot_camera_video.mp4")  # type: ignore[index]
    assert pov["records"][0]["scenario_eval_run_id"]  # type: ignore[index]
    assert pov["proof_boundary"]["real_robot_pov_evidence_proven"] is False  # type: ignore[index]
    policy = _read_json(output_dir / "robot_team_policy_package.json")
    assert policy["policy_package"]["recorded_action_trace"]["trace_manifest_uri"].endswith(  # type: ignore[index]
        "policy_execution_trace.jsonl"
    )

    control_plane_manifest = capture_root / "pipeline" / "live_pipeline_control_plane" / "live_pipeline_control_plane_manifest.json"
    _write_json(
        control_plane_manifest,
        {
            "schema_version": LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
            "capture_root": str(capture_root),
            "job_request_inbox": str(capture_root / "pipeline" / "live_job_request_inbox"),
            "env": {JOB_REQUEST_INBOX_ENV: str(capture_root / "pipeline" / "live_job_request_inbox")},
        },
    )

    intake = build_live_pipeline_input_intake(
        manifest_path=control_plane_manifest,
        webapp_job_request=job_request_path,
        policy_package=output_dir / "robot_team_policy_package.json",
        real_robot_pov=output_dir / "real_robot_pov_manifest.json",
        deployment_outcomes=output_dir / "deployment_outcome_manifest.json",
        live_closure_evidence=output_dir / "live_eval_closure_evidence.json",
        output_path=output_dir / "live_pipeline_input_intake_audit.json",
    )

    assert intake["status"] == "ready_for_control_plane"
    assert intake["proof_boundary"]["policy_package_ready_for_robot_eval_job"] is True
    assert intake["proof_boundary"]["real_robot_pov_ready_for_job_ingest"] is True
    assert intake["proof_boundary"]["deployment_outcomes_ready_for_real_world_validation"] is True
    assert intake["proof_boundary"]["live_closure_evidence_ready_for_closure_audit"] is True


def test_g1_evidence_template_cli_writes_inputs(tmp_path: Path, capsys) -> None:
    capture_root = tmp_path / "capture"
    output_path = tmp_path / "evidence" / "g1_controlled_run_inputs.json"

    exit_code = main(
        [
            "write-template",
            "--capture-root",
            str(capture_root),
            "--output-path",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "template_written" in captured.out
    assert output_path.is_file()
    template = _read_json(output_path)
    assert template["schema_version"] == "g1_controlled_run_inputs.v1"
