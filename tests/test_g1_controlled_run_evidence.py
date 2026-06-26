from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import g1_controlled_run_evidence as g1
from blueprint_pipeline.g1_controlled_run_evidence import (
    G1_CONTROLLED_RUN_EVIDENCE_SCHEMA_VERSION,
    assemble_g1_controlled_run_evidence,
    main,
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
    assert manifest["proof_boundary"]["generated_world_rank_fidelity_result_proven"] is False  # type: ignore[index]


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


def test_g1_evidence_low_level_helper_edges(tmp_path: Path) -> None:
    assert g1._bool("passed") is True
    assert g1._number(True, default=9.0) == 9.0
    assert g1._number("bad", default=3.0) == 3.0
    assert g1._number(object(), default=4.0) == 4.0
    assert g1._safe_id("", "fallback") == "fallback"
    assert g1._file_ref(None) is None

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert g1._read_json_file(invalid_json, "bad") == (None, ["invalid_json_evidence:bad"])
    assert g1._read_json_records(invalid_json, "bad-records") == (
        [],
        ["invalid_json_evidence:bad-records"],
    )

    list_json = tmp_path / "records.json"
    list_json.write_text("[{\"a\": 1}]", encoding="utf-8")
    assert g1._read_json_records(list_json, "records") == ([{"a": 1}], [])
    mapping_json = tmp_path / "mapping.json"
    mapping_json.write_text("{\"a\": 1}", encoding="utf-8")
    assert g1._read_json_records(mapping_json, "mapping") == ([{"a": 1}], [])
    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("1", encoding="utf-8")
    assert g1._read_json_records(scalar_json, "scalar") == (
        [],
        ["invalid_json_evidence:scalar"],
    )

    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text("\nnot-json\n", encoding="utf-8")
    assert g1._read_json_records(jsonl, "jsonl")[1] == [
        "invalid_jsonl_evidence:jsonl:line_2"
    ]
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("\n", encoding="utf-8")
    assert g1._read_json_records(empty_jsonl, "empty") == (
        [],
        ["empty_evidence_records:empty"],
    )


def test_g1_evidence_config_validation_edges() -> None:
    blockers = g1._required_config_blockers(
        {
            "run_id": "<placeholder>",
            "robot_serial_or_fleet_id": "",
            "site_or_lab_location_id": "",
            "operator_id": "",
            "hardware_owner_id": "",
            "safety_reviewer_id": "",
            "robot_team_reviewer_id": "",
            "start_time_utc": "",
            "end_time_utc": "",
            "cycle_time_seconds": "not-a-number",
            "production_webapp_request_id": "",
            "pipeline_intake_request_id": "",
            "production_forward_url": "",
            "webapp_response_status_code": "",
            "operator_statement": "",
            "hardware_owner_statement": "<placeholder>",
            "safety_reviewer_statement": "",
            "robot_team_review_statement": "",
            "accepted_safety_thresholds": {
                "max_speed_mps": "<speed>",
                "min_human_clearance_m": "not-a-number",
                "max_contact_force_n": "",
            },
            "actual_status": "failed",
            "actual_success": False,
            "review_decision": "rejected",
            "sync_status": "failed",
            "signed_customer_delivery_url": "<url>",
            "storage_upload_performed": False,
            "entitlement_verified": False,
            "external_use_allowed": False,
        }
    )

    assert "missing_or_placeholder_config:run_id" in blockers
    assert "missing_or_placeholder_config:hardware_owner_statement" in blockers
    assert "missing_or_placeholder_safety_threshold:max_speed_mps" in blockers
    assert "non_numeric_safety_threshold:min_human_clearance_m" in blockers
    assert "non_numeric_config:cycle_time_seconds" in blockers
    assert "physical_run_status_not_passed" in blockers
    assert "physical_run_actual_success_not_true" in blockers
    assert "safety_review_decision_not_accepted" in blockers
    assert "webapp_sync_status_not_succeeded" in blockers
    assert "missing_or_placeholder_config:signed_customer_delivery_url" in blockers
    assert "storage_upload_not_performed" in blockers
    assert "entitlement_not_verified" in blockers
    assert "rights_privacy_external_use_not_allowed" in blockers


def test_g1_evidence_content_blocker_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unreadable = tmp_path / "unreadable.jsonl"
    unreadable.write_text("{}", encoding="utf-8")
    original_read_text = Path.read_text

    def raise_for_unreadable(self: Path, *args: object, **kwargs: object) -> str:
        if self == unreadable:
            raise OSError("unreadable")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raise_for_unreadable)
    assert g1._read_json_records(unreadable, "unreadable") == (
        [],
        ["unreadable_evidence_file:unreadable"],
    )

    video = tmp_path / "robot_camera_video.mp4"
    video.write_bytes(b"")
    alignment = tmp_path / "timestamp_alignment.json"
    _write_json(alignment, {})
    command = tmp_path / "command_log.jsonl"
    command.write_text(json.dumps({"kind": "policy_command_started"}) + "\n", encoding="utf-8")
    trace = tmp_path / "policy_execution_trace.jsonl"
    trace.write_text(json.dumps({"kind": "trace"}) + "\n", encoding="utf-8")
    contact = tmp_path / "contact_collision_log.json"
    _write_json(contact, {"status": "operator_review_required"})
    metrics = tmp_path / "policy_metrics.json"
    _write_json(metrics, {"status": "draft"})
    review = tmp_path / "robot_team_review.json"
    _write_json(review, {"accepted": True, "review_decision": "accepted", "reviewer_id": "<reviewer>"})

    blockers = g1._evidence_content_blockers(
        {
            "robot_camera_video": video,
            "timestamp_alignment": alignment,
            "command_log": command,
            "policy_execution_trace": trace,
            "contact_collision_log": contact,
            "policy_metrics": metrics,
            "robot_team_review": review,
        },
        {"accepted_safety_thresholds": {"max_contact_force_n": 5}},
    )

    assert "empty_evidence_file:robot_camera_video" in blockers
    assert "timestamp_alignment_missing_max_alignment_error_ms" in blockers
    assert "command_log_missing_policy_command_completed" in blockers
    assert "policy_execution_trace_missing_policy_id" in blockers
    assert "contact_collision_log_still_operator_review_required" in blockers
    assert "contact_collision_log_missing_max_contact_force_n" in blockers
    assert "policy_metrics_missing_episode_count" in blockers
    assert "policy_metrics_missing_success_rate" in blockers
    assert "policy_metrics_missing_intervention_count" in blockers
    assert "policy_metrics_status_not_accepted" in blockers
    assert "robot_team_review_missing_reviewer_id" in blockers

    _write_json(alignment, {"max_alignment_error_ms": 251})
    command.write_text(
        json.dumps({"kind": "policy_command_completed", "exit_code": 7}) + "\n",
        encoding="utf-8",
    )
    _write_json(contact, {"status": "accepted", "max_contact_force_n": 10})
    blockers = g1._evidence_content_blockers(
        {
            "timestamp_alignment": alignment,
            "command_log": command,
            "contact_collision_log": contact,
        },
        {"accepted_safety_thresholds": {"max_contact_force_n": 5}},
    )
    assert "timestamp_alignment_error_exceeds_250_ms" in blockers
    assert "command_log_policy_command_exit_nonzero" in blockers
    assert "contact_collision_log_exceeds_accepted_threshold" in blockers


def test_g1_evidence_assemble_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root = tmp_path / "capture"
    evidence_dir = tmp_path / "evidence"
    output_dir = tmp_path / "assembled"
    monkeypatch.setattr(
        g1,
        "assemble_g1_controlled_run_evidence",
        lambda **_: {"status": "blocked_missing_evidence", "output_dir": str(output_dir)},
    )

    assert g1.main(
        [
            "assemble",
            "--capture-root",
            str(capture_root),
            "--evidence-dir",
            str(evidence_dir),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    assert "blocked_missing_evidence" in capsys.readouterr().out
    assert g1.main(
        [
            "assemble",
            "--capture-root",
            str(capture_root),
            "--evidence-dir",
            str(evidence_dir),
            "--output-dir",
            str(output_dir),
            "--require-ready",
        ]
    ) == 1

    with pytest.raises(SystemExit):
        g1.main([])
