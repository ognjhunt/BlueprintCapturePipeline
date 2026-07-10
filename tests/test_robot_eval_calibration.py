from __future__ import annotations

from blueprint_pipeline import robot_eval_calibration as calibration
from blueprint_pipeline import robot_eval_execution as execution


def test_execution_module_preserves_calibration_compatibility_surface() -> None:
    assert execution._policy_anchor_summaries is calibration.policy_anchor_summaries
    assert (
        execution._calibration_metrics_from_policy_summaries
        is calibration.calibration_metrics_from_policy_summaries
    )
    assert (
        execution._rank_fidelity_claim_eligibility
        is calibration.evaluate_rank_fidelity_claim_eligibility
    )
    assert execution._accepted_anchor_calibration is calibration.build_accepted_anchor_calibration


def test_calibration_metrics_are_computed_by_shared_module() -> None:
    summaries = [
        {
            "policy_id": "policy-a",
            "checkpoint_id": "checkpoint-a",
            "predicted_success_rate": 0.9,
            "actual_success_rate": 0.8,
        },
        {
            "policy_id": "policy-b",
            "checkpoint_id": "checkpoint-b",
            "predicted_success_rate": 0.2,
            "actual_success_rate": 0.3,
        },
    ]

    result = calibration.calibration_metrics_from_policy_summaries(summaries)

    assert result["pearson_success_rate_correlation"] == 1.0
    assert result["spearman_rank_correlation"] == 1.0
    assert result["mmrv"] == 0.0
    assert result["mmrv_definition"] == ("simpler_pairwise_real_success_rate_margin.v1")


def test_rank_fidelity_decision_fails_closed_without_preregistered_study() -> None:
    result = calibration.evaluate_rank_fidelity_claim_eligibility(
        accepted_anchors=[],
        summaries=[],
        metrics={},
        confidence_intervals={},
        registered_split_estimands={},
        study_design=None,
    )

    assert result["status"] == "ineligible"
    assert result["public_rank_fidelity_claim_eligible"] is False
    assert result["deployment_accuracy_claim_supported"] is False
    assert result["real_world_success_rate_prediction_claim_supported"] is False
    assert "study_design_not_preregistered_and_locked" in result["blockers"]


def test_anchor_calibration_exposes_typed_not_measured_boundary() -> None:
    result = calibration.build_accepted_anchor_calibration(
        rows=[],
        prediction_rows=[],
        prediction_anchor_index={},
        prediction_conflict_ids=[],
        prediction_incomplete_rows=[],
    )

    assert result["status"] == "not_measured"
    assert result["accepted_anchor_count"] == 0
    assert "insufficient_anchor_count" in result["blockers"]
    assert result["rank_fidelity_claim_eligibility"]["public_rank_fidelity_claim_eligible"] is False
