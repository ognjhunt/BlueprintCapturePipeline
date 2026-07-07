from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.robot_eval_dataset import build_real_site_robot_eval_dataset
from scripts.run_sim_only_beta_local_gate import _validate_sim_only_outputs
from scripts.run_sim_only_beta_local_gate import (
    _committed_fixture_capture_root,
    _materialize_fixture_capture_root,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_completed_job(
    capture_root: Path,
    job_id: str,
    *,
    sim_only_beta_core_complete: bool = True,
    sim_only_beta_blocked_requirement_ids: list[str] | None = None,
    requirements: list[dict[str, object]] | None = None,
    write_wam_handoff: bool = True,
) -> None:
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
        {
            "status": "blocked_robot_team_grade_requirements",
            "sim_only_beta_core_complete": sim_only_beta_core_complete,
            "sim_only_beta_blocked_requirement_ids": list(
                sim_only_beta_blocked_requirement_ids or []
            ),
            "requirements": list(requirements or []),
        },
    )
    if write_wam_handoff:
        _write_json(
            job_root / "policy_ranking_scorecard.json",
            {
                "status": "completed_visual_review_required",
                "policy_count": 2,
                "top_policy_id": None,
                "evaluator_top_policy_id": "site_finetune_policy",
                "single_best_policy_claimed": False,
                "comparison_blockers": [],
                "ranking_confidence": {
                    "ranking_ambiguous": False,
                    "confidence_level": "medium_evaluator_only",
                },
                "claim_boundary": {
                    "policy_ranking_is_evaluator_bounded": True,
                    "policy_ranking_is_not_evaluation_readiness": True,
                    "rank_fidelity_result_proven": False,
                    "public_claim_upgrade_allowed": False,
                },
            },
        )
        _write_json(
            job_root / "candidate_selection_report.json",
            {
                "status": "visual_review_required_candidate_shortlist",
                "top_policy_id": None,
                "evaluator_top_policy_id": "site_finetune_policy",
                "tie_or_ambiguity_status": "visual_review_required",
                "candidate_shortlist": [{"policy_id": "site_finetune_policy"}],
                "claim_boundary": {
                    "do_not_use_as_rank_fidelity_result": True,
                    "rank_fidelity_result_claimed": False,
                    "accepted_anchor_success_claimed": False,
                },
            },
        )
        _write_json(
            job_root / "wam_eval_claim_boundary.json",
            {
                "primary_proof_target": "policy_comparison_within_configured_evaluator",
                "policy_ranking_is_evaluator_bounded": True,
                "policy_ranking_is_not_evaluation_readiness": True,
                "simulator_execution_proven": False,
                "robot_policy_execution_proven": False,
                "real_world_outcome_proven": False,
                "rank_fidelity_result_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        )


def test_committed_fixture_generates_validated_semantic_spawn_target(
    tmp_path: Path,
) -> None:
    capture_root = _materialize_fixture_capture_root(
        source_capture_root=_committed_fixture_capture_root(),
        work_root=tmp_path / "fixture-work",
    )

    assert (capture_root / "pipeline" / "worldlabs_assets" / "scene.glb").is_file()
    assert (
        capture_root
        / "pipeline"
        / "external_assets"
        / "mujoco_menagerie"
        / "unitree_g1"
        / "BLUEPRINT_FIXTURE_ASSET.txt"
    ).is_file()

    build_real_site_robot_eval_dataset(capture_root=capture_root)

    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    task_cards = json.loads((robot_eval_root / "task_cards.json").read_text())
    scenario_cards = json.loads((robot_eval_root / "scenario_cards.json").read_text())
    object_index = json.loads((robot_eval_root / "site_card.json").read_text())[
        "geometry"
    ]["object_index"]

    assert object_index["physics_coverage_complete"] is True
    assert task_cards["cards"][0]["task_id"] == "fixture_counter_navigation"
    assert task_cards["cards"][0]["target_object_ids"] == ["fixture_service_counter"]
    assert task_cards["cards"][0]["semantic_grounding"][
        "validated_spawn_target_pair"
    ] is True
    assert scenario_cards["cards"][0]["scenario_id"] == (
        "scenario_fixture_counter_navigation_unitree_g1"
    )
    assert scenario_cards["cards"][0]["semantic_spawn_target"][
        "validated_spawn_target_pair"
    ] is True
    assert scenario_cards["cards"][0]["semantic_spawn_target"][
        "fallback_allowed_for_beta_release"
    ] is False


def test_local_gate_defaults_to_committed_fixture_and_writes_blocked_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "sim_only_beta_local_gate_report.json"
    webapp_repo = tmp_path / "Blueprint-WebApp"
    webapp_repo.mkdir()

    exit_code = main(
        [
            "--fixture-work-root",
            str(tmp_path / "fixture-work"),
            "--webapp-repo",
            str(webapp_repo),
            "--mujoco-g1-root",
            str(tmp_path / "missing-g1"),
            "--output-path",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text())
    assert exit_code == 1
    assert report["status"] == "blocked"
    assert report["failed_stage"] == "preflight"
    assert report["blockers"] == [f"mujoco_g1_root_missing:{tmp_path / 'missing-g1'}"]
    assert report["capture_root"].endswith(
        "local-blueprint-fixtures/scenes/sim-only-beta-fixture-site/captures/capture-001"
    )
    assert report["simulator_execution_proven"] is False
    assert report["public_claim_upgrade_allowed"] is False


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
    assert report["simulator_execution_proven"] is True
    assert report["wam_handoff_artifacts_satisfied"] is True
    assert report["wam_handoff_artifacts"]["policy_ranking"]["top_policy_id"] is None
    assert report["public_claim_upgrade_allowed"] is False
    assert report["proof_boundary"]["local_mujoco_simulator_execution_proven"] is True
    assert report["proof_boundary"]["simulator_execution_proven"] is True
    readiness_key = "physical_robot_" "readiness_proven"
    assert readiness_key not in report
    assert readiness_key not in report["proof_boundary"]


def test_local_gate_reports_precise_sim_only_requirement_when_core_blocks(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    proof_path = capture_root / "pipeline" / "live_pipeline_control_plane" / "route.json"
    _write_json(
        proof_path,
        {
            "status": "forwarded_to_pipeline_intake",
            "job_request": {"job_id": "job-1"},
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
            "processed_count": 1,
            "jobs": [{"job_id": "job-1", "status": "simulator_command_completed"}],
        },
    )
    _write_completed_job(
        capture_root,
        "job-1",
        sim_only_beta_core_complete=False,
        sim_only_beta_blocked_requirement_ids=["full_trace_package"],
        requirements=[
            {
                "requirement_id": "full_trace_package",
                "sim_only_beta_required": True,
                "passed": False,
                "blockers": ["missing_trace_artifact_control_stream"],
            },
        ],
    )

    report = _validate_sim_only_outputs(capture_root=capture_root, proof_path=proof_path)

    assert report["status"] == "blocked"
    assert report["blockers"] == ["sim_only_beta_requirement_full_trace_package_not_complete"]
    assert report["simulator_execution_proven"] is True
    assert report["sim_only_beta_requirements_satisfied"] is False
    assert report["wam_handoff_artifacts_satisfied"] is True
    assert report["sim_only_beta_blocked_requirement_ids"] == ["full_trace_package"]
    assert report["robot_team_grade_closure"]["sim_only_beta_requirement_blockers"] == {
        "full_trace_package": ["missing_trace_artifact_control_stream"],
    }
    assert report["public_claim_upgrade_allowed"] is False
    assert report["proof_boundary"]["local_mujoco_simulator_execution_proven"] is True
    assert report["proof_boundary"]["simulator_execution_proven"] is True
    readiness_key = "physical_robot_" "readiness_proven"
    assert readiness_key not in report
    assert readiness_key not in report["proof_boundary"]


def test_local_gate_passes_when_stale_core_false_has_no_sim_only_requirement_blocks(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    proof_path = capture_root / "pipeline" / "live_pipeline_control_plane" / "route.json"
    _write_json(
        proof_path,
        {
            "status": "forwarded_to_pipeline_intake",
            "job_request": {"job_id": "job-1"},
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
            "processed_count": 1,
            "jobs": [{"job_id": "job-1", "status": "simulator_command_completed"}],
        },
    )
    _write_completed_job(
        capture_root,
        "job-1",
        sim_only_beta_core_complete=False,
        requirements=[
            {
                "requirement_id": "full_trace_package",
                "sim_only_beta_required": True,
                "passed": True,
                "blockers": [],
            },
            {
                "requirement_id": "digital_twin_fidelity_qa",
                "sim_only_beta_required": False,
                "passed": False,
                "blockers": ["digital_twin_fidelity_qa_not_passed"],
            },
        ],
    )

    report = _validate_sim_only_outputs(capture_root=capture_root, proof_path=proof_path)

    assert report["status"] == "passed"
    assert report["blockers"] == []
    assert report["sim_only_beta_requirements_satisfied"] is True
    assert report["wam_handoff_artifacts_satisfied"] is True
    assert report["robot_team_grade_closure"]["sim_only_beta_core_complete"] is False
    assert report["robot_team_grade_closure"]["sim_only_beta_blocked_requirement_ids"] == []


def test_local_gate_blocks_precisely_when_wam_handoff_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    proof_path = capture_root / "pipeline" / "live_pipeline_control_plane" / "route.json"
    _write_json(
        proof_path,
        {
            "status": "forwarded_to_pipeline_intake",
            "job_request": {"job_id": "job-1"},
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
            "processed_count": 1,
            "jobs": [{"job_id": "job-1", "status": "simulator_command_completed"}],
        },
    )
    _write_completed_job(capture_root, "job-1", write_wam_handoff=False)

    report = _validate_sim_only_outputs(capture_root=capture_root, proof_path=proof_path)

    assert report["status"] == "blocked"
    assert report["simulator_execution_proven"] is True
    assert report["wam_handoff_artifacts_satisfied"] is False
    assert report["wam_handoff_blockers"] == [
        "wam_handoff_artifact_policy_ranking_scorecard_missing",
        "wam_handoff_artifact_candidate_selection_report_missing",
        "wam_handoff_artifact_wam_eval_claim_boundary_missing",
    ]
    assert report["proof_boundary"]["generated_world_rank_fidelity_result_proven"] is False


def test_local_gate_reports_missing_closure_artifacts_as_blockers(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    proof_path = capture_root / "pipeline" / "live_pipeline_control_plane" / "route.json"
    _write_json(
        proof_path,
        {
            "status": "forwarded_to_pipeline_intake",
            "job_request": {"job_id": "job-1"},
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
            "processed_count": 1,
            "jobs": [{"job_id": "job-1", "status": "blocked"}],
        },
    )
    job_root = capture_root / "pipeline" / "robot_eval_jobs" / "job-1"
    _write_json(
        job_root / "job_run_manifest.json",
        {"status": "blocked", "simulator_execution_proven": False},
    )
    _write_json(
        job_root / "scenario_eval_matrix.json",
        {
            "status": "blocked_invalid_requested_scope",
            "blockers": ["scenario_eval_matrix_semantic_spawn_target_missing"],
            "semantic_spawn_target_coverage_complete": False,
        },
    )
    _write_json(
        job_root / "simulator_service_result.json",
        {"status": "failed", "simulator_execution_proven": False},
    )
    _write_json(
        job_root / "robot_team_grade_eval_closure_manifest.json",
        {
            "status": "blocked_robot_team_grade_requirements",
            "sim_only_beta_blocked_requirement_ids": ["task_success_metrics"],
            "requirements": [
                {
                    "requirement_id": "task_success_metrics",
                    "sim_only_beta_required": True,
                    "passed": False,
                    "blockers": ["scenario_eval_matrix_blocked"],
                }
            ],
        },
    )

    report = _validate_sim_only_outputs(capture_root=capture_root, proof_path=proof_path)

    assert report["status"] == "blocked"
    assert "job_status_not_simulator_command_completed" in report["blockers"]
    assert "simulator_command_batch_closure_manifest_missing" in report["blockers"]
    assert "semantic_spawn_target_coverage_incomplete" in report["blockers"]
    assert report["scenario_eval_matrix"]["status"] == "blocked_invalid_requested_scope"
