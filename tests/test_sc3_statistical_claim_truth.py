from __future__ import annotations

import pytest

from blueprint_pipeline.robot_eval_execution import (
    _bootstrap_confidence_intervals,
    _calibration_metrics_from_policy_summaries,
    _macro_calibration_estimand,
    _policy_anchor_summaries,
    _rank_fidelity_claim_eligibility,
    _simpler_pairwise_margin_rank_violations,
)


def _summary(
    policy: str,
    predicted: float,
    actual: float,
    *,
    criterion: str = "criterion-a",
    split: str = "ind",
) -> dict:
    return {
        "policy_id": policy,
        "checkpoint_id": f"{policy}-checkpoint",
        "criterion_id": criterion,
        "registered_split": split,
        "task_family": "manipulation",
        "predicted_success_rate": predicted,
        "actual_success_rate": actual,
        "absolute_success_rate_error": abs(predicted - actual),
        "unit_of_analysis_key_explicit": True,
    }


def test_simpler_mmrv_matches_hand_calculated_margin_vectors_and_ties() -> None:
    correct = _calibration_metrics_from_policy_summaries(
        [
            _summary("a", 0.7, 0.9),
            _summary("b", 0.5, 0.6),
            _summary("c", 0.1, 0.2),
        ]
    )
    assert correct["mmrv"] == 0.0

    large_inversion = _calibration_metrics_from_policy_summaries(
        [
            _summary("a", 0.1, 0.9),
            _summary("b", 0.6, 0.6),
            _summary("c", 0.9, 0.2),
        ]
    )
    assert large_inversion["mmrv"] == 0.6
    assert large_inversion["maximum_pairwise_real_margin_rank_violation"] == 0.7
    assert large_inversion["mmrv_definition"] == ("simpler_pairwise_real_success_rate_margin.v1")

    near_tie = _simpler_pairwise_margin_rank_violations(
        [0.49, 0.50, 0.10],
        [0.51, 0.50, 0.10],
    )
    assert near_tie == pytest.approx([0.01, 0.01, 0.0])

    real_tie_has_zero_margin = _simpler_pairwise_margin_rank_violations(
        [0.4, 0.6, 0.1],
        [0.5, 0.5, 0.1],
    )
    assert real_tie_has_zero_margin == [0.0, 0.0, 0.0]

    simulated_tie_uses_reference_strict_comparison = _simpler_pairwise_margin_rank_violations(
        [0.5, 0.5, 0.2],
        [0.9, 0.6, 0.2],
    )
    assert simulated_tie_uses_reference_strict_comparison == pytest.approx([0.3, 0.0, 0.0])


def test_unit_level_micro_and_registered_cell_macro_are_reported_separately() -> None:
    summaries = [
        _summary("p1", 0.5, 0.9, criterion="a"),
        _summary("p2", 0.4, 0.8, criterion="a"),
        _summary("p1", 0.9, 0.2, criterion="b"),
        _summary("p2", 0.8, 0.1, criterion="b"),
    ]
    micro = _calibration_metrics_from_policy_summaries(summaries)
    macro = _macro_calibration_estimand(summaries)

    assert micro["pearson_success_rate_correlation"] < 0.0
    assert macro["metrics"]["pearson_success_rate_correlation"] == 1.0
    assert macro["cell_count"] == 2


def _bootstrap_anchors() -> list[dict]:
    rows: list[dict] = []
    for policy_index in range(3):
        for cluster_index in range(5):
            actual = cluster_index < (policy_index + 2)
            predicted = cluster_index < (policy_index + 1)
            rows.append(
                {
                    "policy_id": f"policy-{policy_index}",
                    "checkpoint_id": f"checkpoint-{policy_index}",
                    "criterion_id": "lifting",
                    "registered_split": "ind",
                    "task_family": "manipulation",
                    "matched_initial_condition_id": f"initial-{cluster_index}",
                    "predicted_success": predicted,
                    "actual_success": actual,
                }
            )
    return rows


def test_seeded_hierarchical_bootstrap_is_permutation_invariant() -> None:
    rows = _bootstrap_anchors()
    forward = _bootstrap_confidence_intervals(rows, seed=77, replicate_count=512)
    reversed_rows = _bootstrap_confidence_intervals(
        list(reversed(rows)), seed=77, replicate_count=512
    )

    assert forward == reversed_rows
    assert forward["_bootstrap"]["method"] == ("seeded_hierarchical_cluster_percentile.v1")
    assert forward["_bootstrap"]["seed"] == 77
    assert forward["_bootstrap"]["requested_replicate_count"] == 512
    assert forward["_bootstrap"]["matched_initial_condition_clusters_preserved"] is True


def _locked_design() -> dict:
    return {
        "study_id": "study-locked-001",
        "status": "preregistered_locked",
        "locked_test_data": True,
        "independent_policy_checkpoints": True,
        "primary_estimand": "unit_level_micro_checkpoint_criterion_points",
        "claim_scope": "all_registered_splits",
        "registered_splits": ["ind", "ood"],
        "minimum_matched_trials_per_cell": 20,
        "bootstrap": {"seed": 1729, "replicate_count": 10_000},
        "claim_thresholds": {
            "pearson_ci_lower_min": 0.8,
            "mmrv_ci_upper_max": 0.12,
        },
    }


def _passing_metric_inputs() -> tuple[dict, dict]:
    metrics = {
        "pearson_success_rate_correlation": 0.95,
        "mmrv": 0.05,
    }
    intervals = {
        "pearson_success_rate_correlation": {
            "lower": 0.9,
            "upper": 0.98,
            "sample_count": 10_000,
        },
        "mmrv": {"lower": 0.01, "upper": 0.08, "sample_count": 10_000},
        "_bootstrap": {
            "seed": 1729,
            "requested_replicate_count": 10_000,
        },
    }
    return metrics, intervals


def _passing_split_estimands() -> dict:
    metrics, intervals = _passing_metric_inputs()
    return {
        "estimand": "reported_separately_for_each_registered_split",
        "splits": {
            split: {
                "registered_split": split,
                "metrics": dict(metrics),
                "confidence_intervals": dict(intervals),
            }
            for split in ("ind", "ood")
        },
    }


def test_two_policy_perfect_diagnostic_can_never_unlock_public_claim() -> None:
    anchors = [
        {
            "policy_id": policy,
            "checkpoint_id": f"{policy}-checkpoint",
            "criterion_id": "criterion-a",
            "registered_split": "ind",
            "task_family": "manipulation",
            "matched_initial_condition_id": f"initial-{index}",
            "predicted_success": bool(index),
            "actual_success": bool(index),
        }
        for index, policy in enumerate(("policy-a", "policy-b"))
    ]
    summaries = _policy_anchor_summaries(anchors)
    metrics, intervals = _passing_metric_inputs()
    eligibility = _rank_fidelity_claim_eligibility(
        accepted_anchors=anchors,
        summaries=summaries,
        metrics=metrics,
        confidence_intervals=intervals,
        registered_split_estimands=_passing_split_estimands(),
        study_design=_locked_design(),
    )

    assert eligibility["status"] == "ineligible"
    assert eligibility["public_rank_fidelity_claim_eligible"] is False
    assert "independent_policy_checkpoint_count_lt_7" in eligibility["blockers"]
    assert eligibility["deployment_accuracy_claim_supported"] is False


def test_seven_checkpoint_locked_matched_design_can_reach_metric_eligibility() -> None:
    anchors: list[dict] = []
    for policy_index in range(7):
        for criterion in ("language", "lifting", "placing"):
            for split in ("ind", "ood"):
                for trial_index in range(20):
                    anchors.append(
                        {
                            "policy_id": f"policy-{policy_index}",
                            "checkpoint_id": f"checkpoint-{policy_index}",
                            "criterion_id": criterion,
                            "registered_split": split,
                            "task_family": "manipulation",
                            "matched_initial_condition_id": f"initial-{trial_index}",
                            "predicted_success": trial_index <= policy_index + 10,
                            "actual_success": trial_index <= policy_index + 10,
                        }
                    )
    metrics, intervals = _passing_metric_inputs()
    eligibility = _rank_fidelity_claim_eligibility(
        accepted_anchors=anchors,
        summaries=_policy_anchor_summaries(anchors),
        metrics=metrics,
        confidence_intervals=intervals,
        registered_split_estimands=_passing_split_estimands(),
        study_design=_locked_design(),
    )

    assert eligibility["status"] == "eligible"
    assert eligibility["public_rank_fidelity_claim_eligible"] is True
    assert eligibility["metrics"]["joint_rank_fidelity"]["eligible"] is True
    assert eligibility["blockers"] == []
