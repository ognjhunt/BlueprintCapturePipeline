from __future__ import annotations

import json
from pathlib import Path

from scripts.run_sim_only_beta_local_gate import _validate_sim_only_outputs


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_completed_job(capture_root: Path, job_id: str) -> None:
    job_root = capture_root / "pipeline" / "robot_eval_jobs" / job_id
    _write_json(
        job_root / "job_run_manifest.json",
        {"status": "simulator_command_completed", "simulator_execution_proven": True},
    )
    _write_json(
        job_root / "scenario_eval_matrix.json",
        {
            "status": "completed",
            "scenario_eval_run_count": 11,
            "semantic_spawn_target_coverage_complete": True,
            "deterministic_fallback_spawn_target_run_count": 0,
            "fallback_spawn_target_run_ids": [],
        },
    )
    _write_json(
        job_root / "simulator_service_result.json",
        {"status": "completed", "simulator_execution_proven": True},
    )
    _write_json(
        job_root / "simulator_command_batch_closure_manifest.json",
        {
            "status": "completed",
            "batch_execution_status": "completed",
            "attempt_count": 11,
            "scenario_eval_run_coverage_complete": True,
            "scenario_eval_run_id_coverage_exact": True,
            "metric_coverage_complete": True,
            "machine_trace_package_complete": True,
            "failure_label_coverage_complete": True,
            "visual_review_coverage_complete": True,
            "visual_coverage": {
                "all_required_runs_have_visual_recording": True,
                "all_video_files_complete": True,
            },
        },
    )
    _write_json(
        job_root / "robot_team_grade_eval_closure_manifest.json",
        {"status": "blocked_robot_team_grade_requirements", "sim_only_beta_core_complete": True},
    )


def test_local_gate_validates_route_proof_job_when_inbox_has_stale_rows(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    proof_path = capture_root / "pipeline" / "live_pipeline_control_plane" / "route.json"
    _write_json(
        proof_path,
        {
            "status": "forwarded_to_pipeline_intake",
            "job_request": {"job_id": "fresh-job"},
            "pipeline_intake": {
                "accepted": True,
                "status": "staged_for_control_plane",
                "input_blockers": [],
            },
            "proof_boundary": {
                "local_webapp_route_forwarding_proven": True,
                "pipeline_intake_staged_request_proven": True,
                "simulator_execution_proven": False,
            },
        },
    )
    _write_json(
        capture_root / "pipeline" / "robot_eval_job_requests" / "inbox_run_manifest.json",
        {
            "status": "completed",
            "processed_count": 2,
            "jobs": [
                {"job_id": "stale-job", "status": "simulator_command_completed"},
                {"job_id": "fresh-job", "status": "simulator_command_completed"},
            ],
        },
    )
    _write_completed_job(capture_root, "stale-job")
    _write_completed_job(capture_root, "fresh-job")

    report = _validate_sim_only_outputs(capture_root=capture_root, proof_path=proof_path)

    assert report["status"] == "passed"
    assert report["job_id"] == "fresh-job"
    assert report["route_proof_job_id"] == "fresh-job"
    assert report["blockers"] == []
