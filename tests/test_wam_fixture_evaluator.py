from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline import wam_fixture_evaluator as wam_fixture_module
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


def _passed_short_visual_sanity_manifest(tmp_path: Path) -> dict:
    return {
        "schema_version": "persistent_wam_short_visual_sanity.v1",
        "generated_at": "2026-06-20T00:00:00+00:00",
        "status": "passed_short_visual_sanity",
        "short_visual_sanity_passed": True,
        "short_visual_sanity_manifest_path": str(
            tmp_path / "persistent_wam_short_visual_sanity_manifest.json"
        ),
        "visual_profile": "review_quality",
        "visually_useful_rollout": True,
        "source_policy_observation_visual_qa_status": "passed_visual_quality_gate",
        "wam_rollout_contact_sheet_path": str(tmp_path / "wam_rollout_contact_sheet.jpg"),
        "wam_rollout_visual_quality_report_path": str(
            tmp_path / "wam_rollout_visual_quality_report.json"
        ),
        "video_review_status_path": str(tmp_path / "video_review_status.json"),
        "review_video_path": str(tmp_path / "review.mp4"),
        "blockers": [],
        "claim_boundary": {
            "generated_observation_review_support_only": True,
            "short_visual_sanity_is_not_task_success_proof": True,
            "capture_truth": False,
        },
    }


def _reviewed_success_label(
    *,
    label_id: str,
    policy_id: str,
    run_id: str,
    task_success: bool,
    uncertainty_score: float = 0.12,
) -> dict:
    return {
        "label_id": label_id,
        "attempt_id": f"{label_id}_attempt",
        "rollout_id": f"{label_id}_rollout",
        "policy_id": policy_id,
        "scenario_eval_run_id": run_id,
        "task_id": "tote_transfer",
        "scenario_id": "review_quality_case",
        "task_success": task_success,
        "confidence": round(1.0 - uncertainty_score, 6),
        "uncertainty_score": uncertainty_score,
        "failure_mode_ids": [] if task_success else ["reviewed_generated_failure"],
        "visual_smoke_status": "passed_visual_quality_smoke",
        "visual_rollout_useful_for_task_success_review": True,
        "visual_review_blockers": [],
        "fixture_evaluator_only": False,
        "review_grade_visual_evidence_available": True,
        "review_grade_success_label": True,
        "review_status": "accepted_reviewed_success_label",
        "review_label_refs": [f"review_labels/{label_id}.json"],
        "frame_or_clip_refs": [f"review_media/{label_id}.mp4"],
    }


def test_fixture_wam_eval_job_writes_rollouts_labels_scorecard_and_boundaries(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    _write_json(
        job_dir / "deployment_outcome_ledger.json",
        {
            "records": [
                {
                    "scenarioEvalRunId": "run_heldout_grasp_glare",
                    "policyId": "site_finetune_policy",
                    "taskId": "tote_transfer",
                    "scenarioVariationInstanceId": "variation-grasp-glare",
                    "hardwareId": "unitree-g1",
                    "success": True,
                    "operatorAttestation": "operator-reviewed-rollout",
                },
                {
                    "id": "missing-owner-evidence",
                    "scenario_eval_run_id": "run_train_clearance",
                    "policy_id": "baseline_policy",
                    "hardware_id": "unitree-g1",
                },
            ]
        },
    )

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
        "candidate_selection_report.json",
        "candidate_selection_report.md",
        "visual_review_blocker_summary.json",
        "customer_handoff_report.json",
        "customer_handoff_report.md",
    }
    assert expected.issubset({path.name for path in job_dir.iterdir()})

    rollouts = _read_json(job_dir / "wam_rollout_results.json")
    labels = _read_json(job_dir / "vision_success_labels.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    failure_labels = _read_json(job_dir / "failure_labels.json")
    scorecard = _read_json(job_dir / "policy_ranking_scorecard.json")
    candidate_report = _read_json(job_dir / "candidate_selection_report.json")
    visual_blockers = _read_json(job_dir / "visual_review_blocker_summary.json")
    claim_boundary = _read_json(job_dir / "wam_eval_claim_boundary.json")
    srcc_plan = _read_json(job_dir / "srcc_validation_plan.json")
    anchor_manifest = _read_json(job_dir / "wam_real_world_validation_anchor_manifest.json")
    handoff = _read_json(job_dir / "customer_handoff_report.json")

    assert rollouts["rollout_count"] == 4
    assert labels["label_count"] == 4
    assert labels["fixture_evaluator_only"] is True
    assert labels["visual_rollout_useful_for_task_success_review"] is False
    assert labels["review_grade_success_labels"] is False
    assert labels["labels"][0]["visual_smoke_status"] == (
        "fixture_evaluator_only_no_visual_smoke"
    )
    assert labels["labels"][0]["fixture_evaluator_only"] is True
    assert trace["attempt_count"] == 4
    assert scorecard["status"] == "completed_visual_review_required"
    assert scorecard["top_policy_id"] is None
    assert scorecard["evaluator_top_policy_id"] == "site_finetune_policy"
    assert scorecard["policy_count"] == 2
    assert scorecard["fixture_evaluator_only"] is True
    assert scorecard["review_grade_policy_ranking"] is False
    assert scorecard["visual_rollout_useful_for_task_success_review"] is False
    assert "fixture_evaluator_only_no_review_grade_visual_evidence" in scorecard[
        "visual_review_blockers"
    ]
    assert scorecard["required_scenario_eval_run_ids"] == [
        "run_train_clearance",
        "run_heldout_grasp_glare",
    ]
    assert scorecard["coverage_complete"] is True
    assert scorecard["missing_by_policy"] == {
        "baseline_policy": [],
        "site_finetune_policy": [],
    }
    assert scorecard["extra_by_policy"] == {
        "baseline_policy": [],
        "site_finetune_policy": [],
    }
    assert scorecard["attempt_count_by_policy"] == {
        "baseline_policy": 2,
        "site_finetune_policy": 2,
    }
    assert scorecard["comparison_blockers"] == []
    assert scorecard["ranking_confidence"]["top_policy_margin"] > 0
    assert scorecard["ranking_confidence"]["ranking_ambiguous"] is False
    assert scorecard["single_best_policy_claimed"] is False
    assert candidate_report["status"] == "visual_review_required_candidate_shortlist"
    assert candidate_report["top_policy_id"] is None
    assert candidate_report["evaluator_top_policy_id"] == "site_finetune_policy"
    assert candidate_report["runner_up_policy_id"] == "baseline_policy"
    assert candidate_report["margin"]["predicted_success_rate"] == 0.5
    assert candidate_report["tie_or_ambiguity_status"] == "visual_review_required"
    assert [row["policy_id"] for row in candidate_report["candidate_shortlist"]] == [
        "site_finetune_policy",
        "baseline_policy",
    ]
    assert candidate_report["recommendation"]["recommended_policy_id"] is None
    assert candidate_report["recommendation"]["evaluator_top_policy_id"] == (
        "site_finetune_policy"
    )
    assert any(
        row["reason"] == "visual_review_blockers_present"
        for row in candidate_report["recommended_reruns"]
    )
    assert candidate_report["scenario_matrix_coverage"]["coverage_complete"] is True
    assert candidate_report["scenario_matrix_coverage"]["expected_candidate_attempt_count"] == 4
    assert candidate_report["scenario_matrix_coverage"]["observed_candidate_attempt_count"] == 4
    assert "real_world_validation_followup_request" not in candidate_report
    assert "real_world_validation_requests" not in candidate_report
    assert all(
        row["reason"] != "paired_real_world_anchor_missing"
        for row in candidate_report["recommended_reruns"]
    )
    assert scorecard["comparison_contract"]["comparison_scope"] == "configured_evaluator_only"
    assert scorecard["comparison_contract"]["same_observation_protocol"] is True
    assert scorecard["comparison_contract"]["same_action_protocol"] is True
    assert scorecard["comparison_contract"]["evaluation_readiness_claimed"] is False
    assert scorecard["comparison_contract"][
        "forward_inverse_consistency_metrics_are_support_signals_only"
    ] is True
    assert scorecard["comparison_contract"][
        "forward_inverse_consistency_does_not_upgrade_policy_ranking"
    ] is True
    assert scorecard["forward_inverse_consistency_signal_summary"]["status"] == "not_provided"
    assert scorecard["forward_inverse_consistency_signal_summary"]["support_signal_only"] is True
    assert failure_labels["fixture_evaluator_only"] is True
    assert failure_labels["review_grade_failure_diagnosis"] is False
    assert "fixture_evaluator_only_no_review_grade_visual_evidence" in failure_labels[
        "failure_diagnosis_blockers"
    ]
    assert claim_boundary["generated_rollouts_are_model_derived_support_artifacts"] is True
    assert claim_boundary["fixture_evaluator_only"] is True
    assert claim_boundary["visual_smoke_required_for_review_grade_policy_ranking"] is True
    assert claim_boundary["primary_proof_target"] == (
        "policy_comparison_within_configured_evaluator"
    )
    assert claim_boundary["policy_ranking_is_evaluator_bounded"] is True
    assert claim_boundary["policy_ranking_is_not_evaluation_readiness"] is True
    assert claim_boundary["live_provider_calls_performed"] is False
    assert claim_boundary["customer_specific_srcc_claimed"] is False
    assert claim_boundary["passing_wam_heldout_eval_is_not_rank_fidelity_result"] is True
    assert claim_boundary[
        "forward_inverse_consistency_is_reliability_review_signal_only"
    ] is True
    assert claim_boundary[
        "forward_inverse_consistency_does_not_upgrade_evaluator_bounded_policy_ranking"
    ] is True
    assert claim_boundary["policy_success_claimed_from_consistency"] is False
    assert claim_boundary["task_success_claimed_from_consistency"] is False
    assert claim_boundary["rank_fidelity_claimed_from_consistency"] is False
    assert claim_boundary["deployment_readiness_claimed_from_consistency"] is False
    assert claim_boundary["sensor_truth_claimed_from_consistency"] is False
    assert claim_boundary["external_validation_claimed_from_consistency"] is False
    assert srcc_plan["status"] == "requires_real_world_rollout_anchors"
    assert srcc_plan["customer_specific_srcc_claimed"] is False
    assert anchor_manifest["deployment_outcome_ledger_path"] == "deployment_outcome_ledger.json"
    assert anchor_manifest["usable_anchor_count"] == 1
    assert anchor_manifest["missing_or_incomplete_anchor_count"] == 1
    assert anchor_manifest["anchors"][0]["actual_success"] is True
    assert anchor_manifest["missing_anchor_requirements"][0]["record_id"] == "missing-owner-evidence"
    assert handoff["visual_reviewability_gate"]["status"] == "blocked_visual_review_required"
    assert handoff["forward_inverse_consistency_signal_summary"]["support_signal_only"] is True
    assert "fixture_evaluator_only_no_review_grade_visual_evidence" in handoff[
        "visual_reviewability_gate"
    ]["blockers"]
    assert visual_blockers["status"] == "blocked_visual_review_required"
    assert visual_blockers["recommended_policy_id"] is None
    assert visual_blockers["evaluator_top_policy_id"] == "site_finetune_policy"
    assert "fixture_evaluator_only_no_review_grade_visual_evidence" in visual_blockers[
        "blockers"
    ]


def test_forward_inverse_consistency_overclaims_do_not_upgrade_fixture_policy_ranking() -> None:
    labels = {
        "schema_version": "vision_success_labels.v1",
        "status": "completed",
        "visual_rollout_useful_for_task_success_review": True,
        "review_grade_success_labels": True,
        "fixture_evaluator_only": False,
        "forward_inverse_consistency_proven": True,
        "public_claim_upgrade_allowed": True,
        "labels": [
            {
                "policy_id": policy_id,
                "scenario_eval_run_id": run_id,
                "task_success": False,
                "confidence": 0.9,
                "uncertainty_score": 0.1,
                "forward_inverse_consistency_proven": True,
                "policy_success_claimed_from_consistency": True,
                "task_success_claimed_from_consistency": True,
                "rank_fidelity_claimed_from_consistency": True,
                "deployment_readiness_claimed_from_consistency": True,
                "sensor_truth_claimed_from_consistency": True,
                "external_validation_claimed_from_consistency": True,
                "public_claim_upgrade_allowed": True,
                "visual_rollout_useful_for_task_success_review": True,
                "fixture_evaluator_only": False,
            }
            for policy_id in ("policy_alpha", "policy_beta")
            for run_id in ("run_one", "run_two")
        ],
    }

    scorecard = wam_fixture_module._policy_scorecard(
        substrate="oscar_wam",
        labels=labels,
        generated_at="2026-06-20T00:00:00+00:00",
        required_scenario_eval_run_ids=["run_one", "run_two"],
        policy_ids=["policy_alpha", "policy_beta"],
    )

    summary = scorecard["forward_inverse_consistency_signal_summary"]
    assert summary["status"] == "support_signal_present"
    assert summary["support_signal_only"] is True
    assert summary["label_count_with_consistency_signal"] == 4
    assert "public_claim_upgrade_allowed" in summary["ignored_upgrade_fields_present"]
    assert summary["evaluator_bounded_policy_ranking_upgraded_by_consistency"] is False
    assert summary["policy_success_claimed_from_consistency"] is False
    assert summary["task_success_claimed_from_consistency"] is False
    assert summary["rank_fidelity_claimed_from_consistency"] is False
    assert summary["deployment_readiness_claimed_from_consistency"] is False
    assert summary["sensor_truth_claimed_from_consistency"] is False
    assert summary["external_validation_claimed_from_consistency"] is False
    assert all(row["predicted_success_count"] == 0 for row in scorecard["policy_rankings"])
    assert all(row["predicted_success_rate"] == 0.0 for row in scorecard["policy_rankings"])
    assert scorecard["top_policy_id"] is None
    assert scorecard["single_best_policy_claimed"] is False
    assert scorecard["review_grade_policy_ranking"] is False
    assert scorecard["ranking_confidence"]["ranking_ambiguous"] is True
    assert scorecard["comparison_contract"]["evaluation_readiness_claimed"] is False
    assert scorecard["claim_boundary"][
        "forward_inverse_consistency_does_not_upgrade_evaluator_bounded_policy_ranking"
    ] is True
    assert scorecard["claim_boundary"]["task_success_claimed_from_consistency"] is False
    assert scorecard["claim_boundary"]["external_validation_claimed_from_consistency"] is False


def test_fixture_wam_failure_labels_stay_review_required_and_breakage_aggregates(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)

    run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    labels = _read_json(job_dir / "failure_labels.json")
    breakage = _read_json(job_dir / "breakage_library.json")

    assert labels["status"] == "review_required"
    assert labels["failure_diagnosis_coverage_complete"] is True
    assert labels["failure_diagnosis_complete"] is False
    assert "failure_labels_nonreviewable_failure_hypotheses" in labels[
        "failure_diagnosis_blockers"
    ]
    first = labels["labels"][0]
    assert first["status"] == "review_required"
    assert first["reviewer_acceptance_required"] is True
    assert first["proof_effect"] == (
        "none_until_review_accepted_or_real_world_validation_supplied"
    )
    assert first["evidence_refs"]
    assert "normalized_attempt_trace.json" in first["source_trace_refs"]
    assert first["root_cause_category"]
    assert first["remediation_candidate"]
    assert first["unknown_when_evidence_weak"] is True
    assert first["authoritative_failure_diagnosis"] is False

    assert breakage["aggregation_keys"] == [
        "policy_id",
        "task_id",
        "scenario_id",
        "failure_mode_id",
        "root_cause_category",
    ]
    assert breakage["aggregation_count"] >= 1
    dominant = breakage["dominant_failure_modes"][0]
    assert dominant["failed_attempt_count"] >= 1
    assert dominant["exemplar_failed_attempts"]
    assert dominant["evidence_refs"]


def test_candidate_selection_report_shortlists_when_visual_review_is_blocked(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)

    run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    report = _read_json(job_dir / "candidate_selection_report.json")
    markdown = (job_dir / "candidate_selection_report.md").read_text(encoding="utf-8")
    handoff = _read_json(job_dir / "customer_handoff_report.json")

    assert report["primary_eval_question"] == (
        "which policy performed best in this evaluator, and what broke"
    )
    assert report["status"] == "visual_review_required_candidate_shortlist"
    assert report["top_policy_id"] is None
    assert report["evaluator_top_policy_id"] == "site_finetune_policy"
    assert report["runner_up_policy_id"] == "baseline_policy"
    assert [
        row["policy_id"] for row in report["candidate_shortlist"]
    ] == ["site_finetune_policy", "baseline_policy"]
    assert report["recommendation"]["status"] == "no_winner_claim_use_shortlist"
    assert report["decisive_scenarios"][0]["scenario_eval_run_id"] == (
        "run_heldout_grasp_glare"
    )
    assert report["decisive_scenarios"][0]["successful_policy_ids"] == [
        "site_finetune_policy"
    ]
    assert report["decisive_scenarios"][0]["failed_policy_ids"] == ["baseline_policy"]
    cluster_ids = {cluster["failure_mode_id"] for cluster in report["failure_clusters"]}
    assert "perception_ambiguity_failure" in cluster_ids
    assert "manipulation_alignment_failure" in cluster_ids
    assert report["dominant_failure_modes"][0]["evidence_strength"] == (
        "label_only_needs_review"
    )
    first_cluster = report["failure_clusters"][0]
    hooks = first_cluster["post_training_data_package_hooks"]
    assert hooks["data_to_collect"]
    assert hooks["scenario_variants_to_add"]
    assert hooks["policy_adapter_or_checkpoint_to_retry"][0]["policy_id"] == (
        "baseline_policy"
    )
    assert "real_world_validation_followup_request" not in report
    assert "real_world_validation_requests" not in report
    assert all(
        row["reason"] != "paired_real_world_anchor_missing"
        for row in report["recommended_reruns"]
    )
    assert report["claim_boundary"]["boundary_statement"] == (
        "sim-ranking handoff only; IRL validation is out of scope"
    )
    assert report["claim_boundary"]["do_not_use_as_rank_fidelity_result"] is True
    assert report["claim_boundary"]["rank_fidelity_result_claimed"] is False
    markdown_lower = markdown.lower()
    assert "sim-ranking handoff only" in markdown_lower
    assert "real-world validation" not in markdown_lower
    assert "ready for deployment" not in markdown_lower
    assert "policy-ranking ready" not in markdown_lower
    assert "approved for deployment" not in markdown_lower
    assert "will work irl" not in markdown_lower
    assert handoff["candidate_selection_report_path"] == "candidate_selection_report.json"
    assert handoff["candidate_selection_summary"]["status"] == (
        "visual_review_required_candidate_shortlist"
    )


def test_candidate_ranking_ambiguous_report_uses_shortlist_instead_of_best_policy(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    request = _read_json(job_dir / "job_request.json")
    request["policy_candidates"] = [
        {"policy_id": "policy_alpha", "capabilities": ["all"]},
        {"policy_id": "policy_beta", "capabilities": ["all"]},
    ]
    _write_json(job_dir / "job_request.json", request)

    run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    report = _read_json(job_dir / "candidate_selection_report.json")
    handoff = _read_json(job_dir / "customer_handoff_report.json")

    assert report["status"] == "visual_review_required_candidate_shortlist"
    assert report["top_policy_id"] is None
    assert report["selection"]["ranking_ambiguous"] is True
    assert "visual_review_blockers_or_fixture_only_labels_prevent_winner_claim" in report[
        "selection"
    ]["ambiguity_reasons"]
    assert report["tie_or_ambiguity_status"] == "visual_review_required"
    assert [row["policy_id"] for row in report["candidate_shortlist"]] == [
        "policy_alpha",
        "policy_beta",
    ]
    assert report["margin"]["predicted_success_rate"] == 0.0
    assert handoff["top_policy_id"] is None
    assert [
        row["policy_id"]
        for row in handoff["candidate_selection_summary"]["candidate_shortlist"]
    ] == [
        "policy_alpha",
        "policy_beta",
    ]


def test_candidate_selection_blocks_completed_visual_review_required_scorecard() -> None:
    selection = wam_fixture_module._candidate_selection_summary(
        {
            "status": "completed_visual_review_required",
            "visual_rollout_useful_for_task_success_review": True,
            "review_grade_success_labels": False,
            "review_grade_policy_ranking": False,
            "fixture_evaluator_only": False,
            "visual_review_blockers": [],
            "comparison_blockers": [],
            "ranking_confidence": {
                "ranking_ambiguous": False,
                "uncertainty_penalty_applied": False,
                "ood_blockers": [],
            },
            "policy_rankings": [
                {
                    "rank": 1,
                    "policy_id": "policy-a",
                    "predicted_success_rate": 1.0,
                    "mean_uncertainty": 0.1,
                },
                {
                    "rank": 2,
                    "policy_id": "policy-b",
                    "predicted_success_rate": 0.0,
                    "mean_uncertainty": 0.1,
                },
            ],
        }
    )

    assert selection["status"] == "visual_review_required_candidate_shortlist"
    assert selection["top_policy_id"] is None
    assert selection["evaluator_top_policy_id"] == "policy-a"
    assert "visual_review_blockers_or_fixture_only_labels_prevent_winner_claim" in selection[
        "ambiguity_reasons"
    ]
    assert [row["policy_id"] for row in selection["candidate_shortlist"]] == [
        "policy-a",
        "policy-b",
    ]


def test_review_grade_ranking_requires_short_sanity_manifest_and_review_refs(
    tmp_path: Path,
) -> None:
    labels = {
        "visual_rollout_useful_for_task_success_review": True,
        "review_grade_success_labels": True,
        "fixture_evaluator_only": False,
        "labels": [
            _reviewed_success_label(
                label_id="a-run-1",
                policy_id="policy-a",
                run_id="run-1",
                task_success=True,
            ),
            _reviewed_success_label(
                label_id="a-run-2",
                policy_id="policy-a",
                run_id="run-2",
                task_success=True,
            ),
            _reviewed_success_label(
                label_id="b-run-1",
                policy_id="policy-b",
                run_id="run-1",
                task_success=True,
            ),
            _reviewed_success_label(
                label_id="b-run-2",
                policy_id="policy-b",
                run_id="run-2",
                task_success=False,
            ),
        ],
    }

    scorecard = wam_fixture_module._policy_scorecard(
        substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
        required_scenario_eval_run_ids=["run-1", "run-2"],
        policy_ids=["policy-a", "policy-b"],
        labels=labels,
    )

    assert scorecard["status"] == "completed_visual_review_required"
    assert scorecard["top_policy_id"] is None
    assert scorecard["evaluator_top_policy_id"] == "policy-a"
    assert scorecard["review_grade_policy_ranking"] is False
    assert "short_visual_sanity_manifest_missing_for_review_grade_ranking" in scorecard[
        "visual_review_blockers"
    ]
    assert scorecard["short_visual_sanity_gate"]["passed"] is False
    selection = wam_fixture_module._candidate_selection_summary(scorecard)
    assert selection["status"] == "visual_review_required_candidate_shortlist"
    assert selection["top_policy_id"] is None


def test_review_grade_ranking_can_claim_winner_with_passed_short_sanity_gate(
    tmp_path: Path,
) -> None:
    labels = {
        "visual_rollout_useful_for_task_success_review": True,
        "review_grade_success_labels": True,
        "review_grade_visual_evidence_available": True,
        "fixture_evaluator_only": False,
        "short_visual_sanity_manifest": _passed_short_visual_sanity_manifest(tmp_path),
        "labels": [
            _reviewed_success_label(
                label_id="a-run-1",
                policy_id="policy-a",
                run_id="run-1",
                task_success=True,
            ),
            _reviewed_success_label(
                label_id="a-run-2",
                policy_id="policy-a",
                run_id="run-2",
                task_success=True,
            ),
            _reviewed_success_label(
                label_id="b-run-1",
                policy_id="policy-b",
                run_id="run-1",
                task_success=True,
            ),
            _reviewed_success_label(
                label_id="b-run-2",
                policy_id="policy-b",
                run_id="run-2",
                task_success=False,
            ),
        ],
    }

    scorecard = wam_fixture_module._policy_scorecard(
        substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
        required_scenario_eval_run_ids=["run-1", "run-2"],
        policy_ids=["policy-a", "policy-b"],
        labels=labels,
    )

    assert scorecard["status"] == "completed"
    assert scorecard["top_policy_id"] == "policy-a"
    assert scorecard["single_best_policy_claimed"] is True
    assert scorecard["review_grade_policy_ranking"] is True
    assert scorecard["short_visual_sanity_gate"]["passed"] is True
    assert scorecard["short_visual_sanity_gate"]["contact_sheet_refs"]
    assert scorecard["short_visual_sanity_gate"]["provenance_refs"]
    selection = wam_fixture_module._candidate_selection_summary(scorecard)
    assert selection["status"] == "clear_winner"
    assert selection["top_policy_id"] == "policy-a"


def test_review_grade_ranking_blocks_visually_weak_short_sanity_manifest(
    tmp_path: Path,
) -> None:
    weak_manifest = {
        **_passed_short_visual_sanity_manifest(tmp_path),
        "status": "blocked",
        "short_visual_sanity_passed": False,
        "visually_useful_rollout": False,
        "blockers": ["short_visual_sanity_wam_visual_quality_failed"],
    }
    labels = {
        "visual_rollout_useful_for_task_success_review": True,
        "review_grade_success_labels": True,
        "review_grade_visual_evidence_available": True,
        "fixture_evaluator_only": False,
        "short_visual_sanity_manifest": weak_manifest,
        "labels": [
            _reviewed_success_label(
                label_id="a-run-1",
                policy_id="policy-a",
                run_id="run-1",
                task_success=True,
            ),
            _reviewed_success_label(
                label_id="b-run-1",
                policy_id="policy-b",
                run_id="run-1",
                task_success=False,
            ),
        ],
    }

    scorecard = wam_fixture_module._policy_scorecard(
        substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
        required_scenario_eval_run_ids=["run-1"],
        policy_ids=["policy-a", "policy-b"],
        labels=labels,
    )

    assert scorecard["status"] == "completed_visual_review_required"
    assert scorecard["top_policy_id"] is None
    assert scorecard["short_visual_sanity_gate"]["passed"] is False
    assert "short_visual_sanity_manifest_not_passed" in scorecard[
        "visual_review_blockers"
    ]
    assert "short_visual_sanity_manifest_not_visually_useful" in scorecard[
        "visual_review_blockers"
    ]
    assert "short_visual_sanity_wam_visual_quality_failed" in scorecard[
        "visual_review_blockers"
    ]


def test_failure_diagnosis_blocks_when_review_label_refs_are_missing(
    tmp_path: Path,
) -> None:
    label = _reviewed_success_label(
        label_id="weak-failure",
        policy_id="policy-a",
        run_id="run-1",
        task_success=False,
    )
    label.pop("review_label_refs")
    label["review_status"] = "review_required"
    labels = {
        "visual_rollout_useful_for_task_success_review": True,
        "review_grade_success_labels": True,
        "fixture_evaluator_only": False,
        "labels": [label],
    }

    trace = wam_fixture_module._normalized_attempt_trace(
        substrate="fixture_wam",
        labels=labels,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    failure_labels = wam_fixture_module._failure_labels(
        substrate="fixture_wam",
        trace=trace,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert failure_labels["failure_diagnosis_complete"] is False
    assert "review_grade_failure_label_refs_missing" in failure_labels[
        "failure_diagnosis_blockers"
    ]
    assert "short_visual_sanity_manifest_missing_for_review_grade_ranking" in failure_labels[
        "failure_diagnosis_blockers"
    ]
    first = failure_labels["labels"][0]
    assert first["non_reviewable_failure_hypothesis"] is False
    assert first["label_id"] in failure_labels["nonreviewable_failure_hypothesis_label_ids"]


def test_candidate_failure_handoff_uses_unknown_when_failure_evidence_is_weak() -> None:
    report = wam_fixture_module._candidate_selection_report(
        job_id="job-weak-failure",
        substrate="fixture_wam",
        matrix={
            "runs": [
                {
                    "scenario_eval_run_id": "run-weak",
                    "task_id": "tote_transfer",
                    "scenario_id": "weak_failure",
                    "variation_name": "weak_failure",
                }
            ]
        },
        policies=[
            {"policy_id": "policy_alpha", "checkpoint": "ckpt-alpha"},
            {"policy_id": "policy_beta", "checkpoint": "ckpt-beta"},
        ],
        labels={
            "labels": [
                {
                    "label_id": "label-weak",
                    "attempt_id": "attempt-weak",
                    "rollout_id": "rollout-weak",
                    "scenario_eval_run_id": "run-weak",
                    "policy_id": "policy_alpha",
                    "task_success": False,
                    "uncertainty_score": 0.8,
                    "ood_flags": [],
                    "failure_mode_ids": [],
                }
            ]
        },
        failure_labels={
            "labels": [
                {
                    "label_id": "failure-label-weak",
                    "attempt_id": "attempt-weak",
                    "rollout_id": "rollout-weak",
                    "scenario_eval_run_id": "run-weak",
                    "policy_id": "policy_alpha",
                    "failure_mode_ids": [],
                }
            ]
        },
        scorecard={
            "status": "completed",
            "policy_count": 2,
            "scenario_attempt_count": 2,
            "policy_rankings": [
                {
                    "rank": 1,
                    "policy_id": "policy_beta",
                    "predicted_success_rate": 1.0,
                    "mean_uncertainty": 0.1,
                },
                {
                    "rank": 2,
                    "policy_id": "policy_alpha",
                    "predicted_success_rate": 0.0,
                    "mean_uncertainty": 0.8,
                },
            ],
        },
        followup={"status": "requested_real_world_validation_anchors"},
        anchor_manifest={"usable_anchor_count": 0},
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert report["failure_evidence_status"] == "unknown_needs_review"
    assert report["dominant_failure_modes"] == [
        {
            "failure_mode_id": "unknown_needs_review",
            "count": 1,
            "diagnosis": "unknown_needs_review",
            "evidence_strength": "weak",
        }
    ]
    assert report["failure_clusters"][0]["failure_mode_id"] is None
    assert report["failure_clusters"][0]["diagnosis"] == "unknown_needs_review"
    assert report["failure_clusters"][0]["post_training_data_package_hooks"][
        "policy_adapter_or_checkpoint_to_retry"
    ][0]["checkpoint_id"] == "ckpt-alpha"


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

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--job-dir",
                str(job_dir),
                "--evaluation-substrate",
                "mujoco",
            ]
        )
        == 1
    )


def test_fixture_wam_blocks_missing_scenario_matrix(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture-root"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-wam-missing"
    job_dir.mkdir(parents=True)

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert "scenario_eval_matrix_missing_or_empty" in result["blockers"]
    request = _read_json(job_dir / "wam_evaluation_request.json")
    assert request["status"] == "blocked"


def test_fixture_wam_uses_wam_request_policy_fields_and_redacts_references(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "job-wam-fixture",
            "wamEvaluation": {
                "policies": [
                    {
                        "name": "universal_policy",
                        "checkpoint": "ckpt-001",
                        "capabilities": ["all"],
                        "api_key": "must-not-persist",
                    }
                ]
            },
            "policyPackage": {"actionInterface": "joint_position_targets"},
            "robotProfile": {"robotModel": "unitree-g1"},
            "taskRequest": {"taskFamily": "warehouse_tote_transfer"},
            "classicalSimCrossChecks": ["isaac_sim", "mujoco", "unknown"],
        },
    )

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    policy_binding = _read_json(job_dir / "wam_policy_interface_binding.json")
    scorecard = _read_json(job_dir / "policy_ranking_scorecard.json")
    validation = _read_json(job_dir / "wam_customer_validation_envelope.json")
    cross_check = _read_json(job_dir / "wam_classical_sim_cross_check_plan.json")
    assert policy_binding["policies"][0]["policy_id"] == "policy_candidate_01"
    assert policy_binding["policies"][0]["checkpoint_id"] == "ckpt-001"
    assert policy_binding["policies"][0]["reference"]["api_key"] == "<redacted>"
    assert policy_binding["action_interface"] == "joint_position_targets"
    assert scorecard["status"] == "blocked_inconclusive_ranking"
    assert scorecard["evaluator_top_policy_id"] == "policy_candidate_01"
    assert scorecard["top_policy_id"] is None
    assert scorecard["single_best_policy_claimed"] is False
    selection = wam_fixture_module._candidate_selection_summary(scorecard)
    assert selection["status"] == "single_candidate_no_comparative_ranking"
    assert selection["top_policy_id"] is None
    assert selection["evaluator_top_policy_id"] == "policy_candidate_01"
    assert [row["policy_id"] for row in selection["candidate_shortlist"]] == [
        "policy_candidate_01",
    ]
    assert selection["ambiguity_reasons"] == ["only_one_policy_candidate_was_evaluated"]
    assert validation["hardware_id"] == "unitree-g1"
    assert validation["task_family"] == "warehouse_tote_transfer"
    assert cross_check["recommended_cross_checks"] == [
        "classical_sim_isaac",
        "classical_sim_mujoco",
    ]


def test_policy_ranking_blocks_missing_scenario_ids_without_candidate_winner() -> None:
    scorecard = wam_fixture_module._policy_scorecard(
        substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
        required_scenario_eval_run_ids=["run-1"],
        policy_ids=["policy-a", "policy-b"],
        labels={
            "labels": [
                {
                    "policy_id": "policy-a",
                    "task_success": True,
                    "uncertainty_score": 0.12,
                    "confidence": 0.88,
                },
                {
                    "policy_id": "policy-b",
                    "task_success": False,
                    "uncertainty_score": 0.12,
                    "confidence": 0.88,
                },
            ]
        },
    )

    assert scorecard["status"] == "blocked_inconclusive_ranking"
    assert scorecard["missing_by_policy"] == {"policy-a": ["run-1"], "policy-b": ["run-1"]}
    assert "policy_coverage_missing_required_scenario_eval_run_ids" in scorecard[
        "comparison_blockers"
    ]
    assert scorecard["top_policy_id"] is None
    assert scorecard["single_best_policy_claimed"] is False
    selection = wam_fixture_module._candidate_selection_summary(scorecard)
    assert selection["status"] == "blocked_inconclusive_candidate_selection"
    assert selection["top_policy_id"] is None
    assert selection["evaluator_top_policy_id"] == "policy-a"
    assert [row["policy_id"] for row in selection["candidate_shortlist"]] == [
        "policy-a",
        "policy-b",
    ]


def test_live_wam_provider_command_normalizes_rollouts_and_upload_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    script = tmp_path / "provider_fixture.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "payload = {",
                "    'wam_rollout_results': {",
                "        'rollouts': [",
                "            None,",
                "            {",
                "                'policyId': 'provider-policy',",
                "                'success': True,",
                "                'uncertainty_score': '0.21',",
                "                'failure_mode_ids': 'fixture_failure',",
                "                'ood_flags': 'glare',",
                "                'metrics': {'latency_ms': 12},",
                "                'claim_boundary': {'provider_reported': True},",
                "            },",
                "        ]",
                "    },",
                "    'artifact_upload_evidence': {",
                "        'upload_complete': True,",
                "        'api_key': 'secret-value',",
                "    },",
                "}",
                "with open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8') as fh:",
                "    json.dump(payload, fh)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", "true")
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_API_KEY", "test-auth")
    monkeypatch.setenv(
        "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND",
        f"{sys.executable} {script}",
    )

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        allow_live_provider=True,
        artifact_output_uri="gs://bucket/wam-output",
        budget_usd=12.5,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    assert result["claim_boundary"]["live_provider_calls_performed"] is True
    provider_execution = _read_json(job_dir / "wam_provider_execution_manifest.json")
    provider_upload = _read_json(job_dir / "wam_provider_artifact_upload_proof.json")
    rollouts = _read_json(job_dir / "wam_rollout_results.json")
    assert provider_execution["status"] == "completed"
    assert provider_execution["provider_command_used"] is True
    assert provider_execution["attempt_count"] == 1
    assert provider_execution["detail"]["normalized_rollout_count"] == 1
    assert provider_upload["status"] == "upload_proven"
    assert provider_upload["evidence"]["api_key"] == "<redacted>"
    assert rollouts["rollouts"][0]["scenario_eval_run_id"] == "scenario_eval_run_0002"
    assert rollouts["rollouts"][0]["policy_id"] == "provider-policy"
    assert rollouts["rollouts"][0]["failure_mode_ids"] == ["fixture_failure"]
    assert rollouts["rollouts"][0]["ood_flags"] == ["glare"]


def test_live_wam_provider_requires_explicit_and_env_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    monkeypatch.delenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", raising=False)
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_API_KEY", "test-auth")
    command = f"{sys.executable} -c 'print(1)'"

    missing_env = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        allow_live_provider=True,
        provider_command=command,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert missing_env["status"] == "blocked"
    assert "BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER_not_enabled" in missing_env["blockers"]

    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", "true")
    missing_explicit = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        provider_command=command,
        generated_at="2026-06-20T00:00:01+00:00",
    )
    assert missing_explicit["status"] == "blocked"
    assert "allow_live_wam_provider_not_enabled" in missing_explicit["blockers"]


def test_live_wam_provider_command_blocks_when_output_has_no_rollouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    script = tmp_path / "provider_empty.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "with open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8') as fh:",
                "    json.dump({'rollouts': []}, fh)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", "true")
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_API_KEY", "test-auth")

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        allow_live_provider=True,
        provider_command=f"{sys.executable} {script}",
        max_retries=1,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert "wam_provider_output_missing_rollouts" in result["blockers"]
    provider_execution = _read_json(job_dir / "wam_provider_execution_manifest.json")
    assert provider_execution["status"] == "blocked"
    assert provider_execution["attempt_count"] == 2


def test_fixture_wam_helper_edges_cover_forced_failure_and_generated_run_ids(
    tmp_path: Path,
) -> None:
    assert wam_fixture_module._string_list("") == []
    assert wam_fixture_module._number(True, default=None) is None

    runs = wam_fixture_module._matrix_runs(
        {
            "runs": [
                "not-a-mapping",
                {
                    "task_id": "tote_transfer",
                    "scenario_id": "human_crossing",
                    "variation_name": "human_crossing",
                },
            ]
        }
    )
    assert runs[0]["scenario_eval_run_id"] == "scenario_eval_run_0002"

    rollout = wam_fixture_module._rollout_for_run(
        job_dir=tmp_path,
        substrate="fixture_wam",
        policy={
            "policy_id": "forced_failure_policy",
            "capabilities": ["all"],
            "fixture_success_profile": {
                "fail_scenario_eval_run_ids": ["scenario_eval_run_0002"],
            },
        },
        run=runs[0],
        index=1,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert rollout["predicted_success"] is False
    assert "fixture_forced_failure" in rollout["ood_flags"]
    assert "fixture_policy_failure" in rollout["failure_mode_ids"]
    assert "dynamic_agent_safety_failure" in rollout["failure_mode_ids"]


def test_fixture_wam_blocks_when_policy_candidate_resolution_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    monkeypatch.setattr(wam_fixture_module, "_policy_candidates", lambda **_: [])

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert "policy_candidates_missing" in result["blockers"]


def test_policy_ranking_blocks_asymmetric_policy_scenario_coverage() -> None:
    scorecard = wam_fixture_module._policy_scorecard(
        substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
        required_scenario_eval_run_ids=["run-1", "run-2"],
        policy_ids=["policy-a", "policy-b"],
        labels={
            "visual_rollout_useful_for_task_success_review": True,
            "review_grade_success_labels": True,
            "labels": [
                {
                    "policy_id": "policy-a",
                    "scenario_eval_run_id": "run-1",
                    "task_success": True,
                    "uncertainty_score": 0.12,
                    "confidence": 0.88,
                },
                {
                    "policy_id": "policy-a",
                    "scenario_eval_run_id": "run-2",
                    "task_success": True,
                    "uncertainty_score": 0.12,
                    "confidence": 0.88,
                },
                {
                    "policy_id": "policy-b",
                    "scenario_eval_run_id": "run-1",
                    "task_success": True,
                    "uncertainty_score": 0.12,
                    "confidence": 0.88,
                },
            ]
        },
    )

    assert scorecard["status"] == "blocked_inconclusive_ranking"
    assert scorecard["coverage_complete"] is False
    assert scorecard["missing_by_policy"]["policy-b"] == ["run-2"]
    assert scorecard["attempt_count_by_policy"] == {"policy-a": 2, "policy-b": 1}
    assert "policy_coverage_missing_required_scenario_eval_run_ids" in scorecard[
        "comparison_blockers"
    ]
    assert scorecard["top_policy_id"] is None
    assert scorecard["single_best_policy_claimed"] is False


def test_policy_ranking_tie_band_is_ambiguous_without_best_policy_claim(
    tmp_path: Path,
) -> None:
    scorecard = wam_fixture_module._policy_scorecard(
        substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
        required_scenario_eval_run_ids=["run-1", "run-2"],
        policy_ids=["policy-a", "policy-b"],
        labels={
            "visual_rollout_useful_for_task_success_review": True,
            "review_grade_success_labels": True,
            "short_visual_sanity_manifest": _passed_short_visual_sanity_manifest(tmp_path),
            "labels": [
                _reviewed_success_label(
                    label_id="policy-a-run-1",
                    policy_id="policy-a",
                    run_id="run-1",
                    task_success=True,
                    uncertainty_score=0.1,
                ),
                _reviewed_success_label(
                    label_id="policy-a-run-2",
                    policy_id="policy-a",
                    run_id="run-2",
                    task_success=False,
                    uncertainty_score=0.1,
                ),
                _reviewed_success_label(
                    label_id="policy-b-run-1",
                    policy_id="policy-b",
                    run_id="run-1",
                    task_success=True,
                    uncertainty_score=0.2,
                ),
                _reviewed_success_label(
                    label_id="policy-b-run-2",
                    policy_id="policy-b",
                    run_id="run-2",
                    task_success=False,
                    uncertainty_score=0.2,
                ),
            ]
        },
    )

    assert scorecard["status"] == "completed_ambiguous_ranking"
    assert scorecard["coverage_complete"] is True
    assert scorecard["comparison_blockers"] == []
    assert scorecard["ranking_confidence"]["top_policy_margin"] == 0.0
    assert scorecard["ranking_confidence"]["ranking_ambiguous"] is True
    assert scorecard["ranking_confidence"]["confidence_level"] == "ambiguous"
    assert scorecard["evaluator_top_policy_id"] == "policy-a"
    assert scorecard["top_policy_id"] is None
    assert scorecard["single_best_policy_claimed"] is False


def test_policy_ranking_high_ood_or_uncertainty_downgrades_confidence(
    tmp_path: Path,
) -> None:
    scorecard = wam_fixture_module._policy_scorecard(
        substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
        required_scenario_eval_run_ids=["run-1", "run-2"],
        policy_ids=["policy-a", "policy-b"],
        labels={
            "visual_rollout_useful_for_task_success_review": True,
            "review_grade_success_labels": True,
            "short_visual_sanity_manifest": _passed_short_visual_sanity_manifest(tmp_path),
            "labels": [
                {
                    **_reviewed_success_label(
                        label_id="policy-a-run-1",
                        policy_id="policy-a",
                        run_id="run-1",
                        task_success=True,
                        uncertainty_score=0.72,
                    ),
                    "uncertainty_score": 0.72,
                    "confidence": 0.28,
                    "ood_flags": ["vision_distribution_shift"],
                },
                {
                    **_reviewed_success_label(
                        label_id="policy-a-run-2",
                        policy_id="policy-a",
                        run_id="run-2",
                        task_success=True,
                        uncertainty_score=0.68,
                    ),
                    "uncertainty_score": 0.68,
                    "confidence": 0.32,
                },
                _reviewed_success_label(
                    label_id="policy-b-run-1",
                    policy_id="policy-b",
                    run_id="run-1",
                    task_success=True,
                    uncertainty_score=0.1,
                ),
                _reviewed_success_label(
                    label_id="policy-b-run-2",
                    policy_id="policy-b",
                    run_id="run-2",
                    task_success=False,
                    uncertainty_score=0.1,
                ),
            ]
        },
    )

    assert scorecard["status"] == "completed_low_confidence_ranking"
    assert scorecard["top_policy_id"] is None
    assert scorecard["evaluator_top_policy_id"] == "policy-a"
    assert scorecard["single_best_policy_claimed"] is False
    assert scorecard["ranking_confidence"]["top_policy_margin"] == 0.5
    assert scorecard["ranking_confidence"]["uncertainty_penalty_applied"] is True
    assert scorecard["ranking_confidence"]["ood_blockers"] == [
        "policy:policy-a:ood_rate_high"
    ]
    assert scorecard["ranking_confidence"]["confidence_level"] == "low"
    assert scorecard["ranking_confidence"]["real_world_calibration_metrics"] == {
        "spearman_rank_correlation": "not_measured",
        "pearson_success_rate_correlation": "not_measured",
        "mean_maximum_rank_violation": "not_measured",
    }
    selection = wam_fixture_module._candidate_selection_summary(scorecard)
    assert selection["status"] == "low_confidence_candidate_shortlist"
    assert selection["top_policy_id"] is None
    assert selection["evaluator_top_policy_id"] == "policy-a"
    assert [row["policy_id"] for row in selection["candidate_shortlist"]] == [
        "policy-a",
        "policy-b",
    ]
