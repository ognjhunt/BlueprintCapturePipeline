from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.g1_field_run_review import (
    G1_FIELD_RUN_REVIEW_SCHEMA_VERSION,
    review_g1_field_run_evidence,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_evidence_dir(evidence_dir: Path) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "robot_camera_video.mp4").write_bytes(b"physical-g1-video")
    _write_json(
        evidence_dir / "timestamp_alignment.json",
        {"schema_version": "g1_timestamp_alignment.v1", "max_alignment_error_ms": 60},
    )
    (evidence_dir / "action_log.jsonl").write_text(
        json.dumps({"kind": "action", "action_id": "lowcmd-1", "motor_targets": []}) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "robot_state_log.jsonl").write_text(
        json.dumps({"kind": "state", "motor_state": []}) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "command_log.jsonl").write_text(
        json.dumps({"kind": "policy_command_started", "command": "run-policy"}) + "\n"
        + json.dumps({"kind": "policy_command_completed", "exit_code": 0}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        evidence_dir / "contact_collision_log.json",
        {
            "schema_version": "g1_contact_collision_log.v1",
            "status": "operator_review_required",
            "events": [],
            "max_contact_force_n": 0,
        },
    )
    (evidence_dir / "policy_execution_trace.jsonl").write_text(
        json.dumps({"policy_id": "unitree_rl_gym_g1_mujoco_policy_candidate"})
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        evidence_dir / "g1_controlled_run_inputs.json",
        {
            "schema_version": "g1_controlled_run_inputs.v1",
            "job_id": "robot-eval-production-route-123",
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
            "cycle_time_seconds": 42,
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
            "production_webapp_request_id": "robot-eval-production-route-123",
            "pipeline_intake_request_id": "robot-eval-production-route-123",
            "production_forward_url": "https://www.tryblueprint.io/api/robot-eval/job-requests",
            "webapp_response_status_code": "202",
            "sync_status": "succeeded",
            "operator_statement": "Operator signed the physical G1 evidence package.",
            "hardware_owner_statement": "Hardware owner signed the G1 identity and run.",
            "safety_reviewer_statement": "Safety reviewer accepted this controlled G1 run.",
            "robot_team_review_statement": "Robot team accepted the non-default G1 policy package.",
        },
    )


def test_g1_field_run_review_blocks_without_explicit_acceptance(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    _seed_evidence_dir(evidence_dir)

    manifest = review_g1_field_run_evidence(evidence_dir=evidence_dir)

    assert manifest["schema_version"] == G1_FIELD_RUN_REVIEW_SCHEMA_VERSION
    assert manifest["status"] == "blocked_review_acceptance_required"
    assert "missing_explicit_safety_acceptance" in manifest["blockers"]
    assert "missing_explicit_policy_acceptance" in manifest["blockers"]
    assert _read_json(evidence_dir / "hardware_validation.json")["status"] == "operator_review_required"
    assert _read_json(evidence_dir / "robot_team_review.json")["accepted"] is False


def test_g1_field_run_review_accepts_evidence_when_reviewers_sign(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    _seed_evidence_dir(evidence_dir)

    manifest = review_g1_field_run_evidence(
        evidence_dir=evidence_dir,
        accept_safety=True,
        accept_policy=True,
    )

    assert manifest["status"] == "reviewed_evidence_ready_for_assembly"
    assert manifest["blockers"] == []
    hardware = _read_json(evidence_dir / "hardware_validation.json")
    assert hardware["status"] == "accepted"
    assert hardware["hardware_ready"] is True
    assert hardware["estop_verified"] is True
    metrics = _read_json(evidence_dir / "policy_metrics.json")
    assert metrics["status"] == "accepted"
    assert metrics["episode_count"] == 1
    assert metrics["success_rate"] == 1.0
    assert metrics["intervention_count"] == 0
    review = _read_json(evidence_dir / "robot_team_review.json")
    assert review["accepted"] is True
    assert review["review_decision"] == "accepted"
