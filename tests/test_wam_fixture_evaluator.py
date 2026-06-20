from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.wam_fixture_evaluator import main, run_wam_eval_job


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_job(tmp_path: Path) -> tuple[Path, Path]:
    capture_root = tmp_path / "capture-root"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-wam-fixture"
    job_dir.mkdir(parents=True)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "scenario_variation_instance_id": "variation-clearance",
                    "task_id": "tote_transfer",
                    "scenario_id": "blocked_path",
                    "variation_name": "blocked_path",
                    "split": "train",
                },
                {
                    "scenario_eval_run_id": "run_heldout_grasp_glare",
                    "scenario_variation_instance_id": "variation-grasp-glare",
                    "task_id": "tote_transfer",
                    "scenario_id": "glare_grasp",
                    "variation_name": "glare_grasp",
                    "split": "heldout",
                },
            ],
        },
    )
    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "job-wam-fixture",
            "evaluation_substrate": "fixture_wam",
            "policy_candidates": [
                {
                    "policy_id": "baseline_policy",
                    "capabilities": ["clearance_aware_navigation"],
                },
                {
                    "policy_id": "site_finetune_policy",
                    "capabilities": [
                        "clearance_aware_navigation",
                        "visual_recheck",
                        "grasp_alignment_correction",
                    ],
                },
            ],
        },
    )
    _write_json(
        job_dir / "policy_package_manifest.json",
        {
            "schema_version": "robot_eval_policy_package_manifest.v1",
            "status": "configured",
        },
    )
    return capture_root, job_dir


def test_fixture_wam_eval_job_writes_rollouts_labels_scorecard_and_boundaries(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    expected = {
        "evaluation_substrate_registry.json",
        "wam_evaluation_request.json",
        "wam_rollout_manifest.json",
        "wam_rollout_results.json",
        "vision_success_labels.json",
        "normalized_attempt_trace.json",
        "failure_labels.json",
        "policy_ranking_scorecard.json",
        "wam_eval_claim_boundary.json",
        "real_world_validation_followup_request.json",
        "srcc_validation_plan.json",
        "customer_handoff_report.json",
        "customer_handoff_report.md",
    }
    assert expected.issubset({path.name for path in job_dir.iterdir()})

    rollouts = _read_json(job_dir / "wam_rollout_results.json")
    labels = _read_json(job_dir / "vision_success_labels.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    scorecard = _read_json(job_dir / "policy_ranking_scorecard.json")
    claim_boundary = _read_json(job_dir / "wam_eval_claim_boundary.json")
    srcc_plan = _read_json(job_dir / "srcc_validation_plan.json")

    assert rollouts["rollout_count"] == 4
    assert labels["label_count"] == 4
    assert trace["attempt_count"] == 4
    assert scorecard["top_policy_id"] == "site_finetune_policy"
    assert scorecard["policy_count"] == 2
    assert claim_boundary["generated_rollouts_are_model_derived_support_artifacts"] is True
    assert claim_boundary["live_provider_calls_performed"] is False
    assert claim_boundary["customer_specific_srcc_claimed"] is False
    assert claim_boundary["passing_wam_heldout_eval_is_not_deployment_approval"] is True
    assert srcc_plan["status"] == "requires_real_world_rollout_anchors"
    assert srcc_plan["customer_specific_srcc_claimed"] is False


def test_fixture_wam_cli_and_live_provider_blocked_manifest(tmp_path: Path) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--job-dir",
                str(job_dir),
                "--evaluation-substrate",
                "fixture_wam",
            ]
        )
        == 0
    )

    blocked = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert blocked["status"] == "blocked"
    assert "cosmos3_wam_provider_adapter_not_configured_for_local_run" in blocked["blockers"]
    blocked_request = _read_json(job_dir / "wam_evaluation_request.json")
    blocked_boundary = _read_json(job_dir / "wam_eval_claim_boundary.json")
    assert blocked_request["status"] == "blocked"
    assert blocked_request["evaluation_substrate"] == "cosmos3_wam"
    assert blocked_boundary["live_provider_calls_performed"] is False
