from __future__ import annotations

import pytest

from blueprint_pipeline.wam_score_claim_gate import (
    WAM_SCORE_CLAIM_GRADES,
    WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER,
    apply_wam_score_claim_gate,
    evaluate_wam_calibration_anchors,
    score_wam_consistency,
    score_wam_rollout_set_consistency,
)


def _trajectory(points: list[list[float]], *, start: float = 0.0, dt: float = 0.1) -> dict:
    return {
        "trajectory": [
            {"timestamp": start + index * dt, "position": point}
            for index, point in enumerate(points)
        ]
    }


def _passing_anchor_validation() -> dict:
    return {
        "schema_version": "policy_ranking_ladder_validation.v1",
        "status": "recovered",
        "ranker_ordering_recovered": True,
        "expected_ranking": [
            "policy_clean",
            "policy_clean_noise_0p1",
            "policy_clean_noise_0p3",
        ],
        "spearman_rank_correlation_vs_expected": 1.0,
        "blockers": [],
    }


# --- score_wam_consistency -------------------------------------------------


def test_matching_trajectories_score_high_and_pass() -> None:
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    result = score_wam_consistency(_trajectory(points), _trajectory(points))
    assert result["status"] == "scored"
    assert result["passed"] is True
    assert result["consistency_score"] == pytest.approx(1.0)
    assert result["temporal_consistency"] == pytest.approx(1.0)
    assert result["geometric_consistency"] == pytest.approx(1.0)
    assert result["compared_step_count"] == 4


def test_diverging_trajectory_scores_low_and_fails() -> None:
    reference = _trajectory(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    rollout = _trajectory(
        [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [4.0, 4.0, 0.0], [6.0, 6.0, 0.0]]
    )
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "scored"
    assert result["passed"] is False
    assert result["consistency_score"] is not None
    assert result["consistency_score"] < 0.5


def test_non_monotonic_timestamps_degrade_temporal_consistency() -> None:
    rollout = {
        "trajectory": [
            {"timestamp": 0.0, "position": [0.0, 0.0, 0.0]},
            {"timestamp": 0.2, "position": [0.1, 0.0, 0.0]},
            {"timestamp": 0.1, "position": [0.2, 0.0, 0.0]},
            {"timestamp": 0.3, "position": [0.3, 0.0, 0.0]},
        ]
    }
    reference = _trajectory(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "scored"
    assert result["temporal_consistency"] < 1.0
    assert result["passed"] is False


def test_missing_trajectory_fails_closed() -> None:
    result = score_wam_consistency({}, _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    assert result["status"] == "blocked"
    assert result["consistency_score"] is None
    assert result["passed"] is False
    assert any("rollout_trajectory" in blocker for blocker in result["blockers"])


def test_missing_reference_fails_closed() -> None:
    result = score_wam_consistency(_trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), {})
    assert result["status"] == "blocked"
    assert result["passed"] is False
    assert any("reference_trajectory" in blocker for blocker in result["blockers"])


def test_non_finite_values_fail_closed() -> None:
    rollout = _trajectory([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "blocked"
    assert result["passed"] is False
    assert "non_finite_trajectory_values" in result["blockers"]


def test_dimension_mismatch_fails_closed() -> None:
    rollout = {"trajectory": [{"timestamp": 0.0, "position": [0.0, 0.0]}, {"timestamp": 0.1, "position": [0.1, 0.0]}]}
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "blocked"
    assert "trajectory_dimension_mismatch" in result["blockers"]


def test_reference_accepts_bare_step_sequence_with_waypoints() -> None:
    reference = [
        {"timestamp": 0.0, "waypoint": [0.0, 0.0, 0.0]},
        {"timestamp": 0.1, "waypoint": [0.1, 0.0, 0.0]},
    ]
    rollout = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "scored"
    assert result["passed"] is True


def test_consistency_claim_boundary_never_upgrades_success() -> None:
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    result = score_wam_consistency(_trajectory(points), _trajectory(points))
    boundary = result["claim_boundary"]
    assert boundary["consistency_score_is_support_signal_not_task_success"] is True
    assert boundary["consistency_score_does_not_prove_rank_fidelity"] is True


# --- score_wam_rollout_set_consistency --------------------------------------


def test_rollout_set_consistency_aggregates_conservatively() -> None:
    good = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    bad = _trajectory([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]])
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    result = score_wam_rollout_set_consistency(
        rollouts=[
            {"rollout_id": "r1", **good},
            {"rollout_id": "r2", **bad},
        ],
        reference=reference,
    )
    assert result["status"] == "scored"
    assert result["scored_rollout_count"] == 2
    assert result["consistency_score"] == min(
        row["consistency_score"] for row in result["rollout_scores"]
    )
    assert result["passed"] is False


def test_rollout_set_consistency_blocks_when_no_rollout_scoreable() -> None:
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = score_wam_rollout_set_consistency(
        rollouts=[{"rollout_id": "r1", "generated_video_path": "x.mp4"}],
        reference=reference,
    )
    assert result["status"] == "blocked"
    assert result["consistency_score"] is None
    assert result["passed"] is False


# --- evaluate_wam_calibration_anchors ---------------------------------------


def test_passing_ladder_validation_yields_present_and_passed_anchors() -> None:
    check = evaluate_wam_calibration_anchors(_passing_anchor_validation())
    assert check["anchors_present"] is True
    assert check["anchors_passed"] is True
    assert check["anchor_set"] == [
        "policy_clean",
        "policy_clean_noise_0p1",
        "policy_clean_noise_0p3",
    ]
    assert check["blockers"] == []


def test_not_recovered_ladder_validation_fails_anchor_check() -> None:
    validation = _passing_anchor_validation()
    validation["status"] = "not_recovered"
    validation["ranker_ordering_recovered"] = False
    check = evaluate_wam_calibration_anchors(validation)
    assert check["anchors_present"] is True
    assert check["anchors_passed"] is False


def test_missing_anchor_validation_fails_closed() -> None:
    check = evaluate_wam_calibration_anchors(None)
    assert check["anchors_present"] is False
    assert check["anchors_passed"] is False
    assert check["anchor_set"] == []
    assert "calibration_anchor_validation_missing" in check["blockers"]


def test_unrecognized_anchor_schema_fails_closed() -> None:
    validation = _passing_anchor_validation()
    validation["schema_version"] = "something_else.v9"
    check = evaluate_wam_calibration_anchors(validation)
    assert check["anchors_present"] is False
    assert check["anchors_passed"] is False
    assert "calibration_anchor_validation_schema_unrecognized" in check["blockers"]


def test_single_anchor_set_is_too_small() -> None:
    validation = _passing_anchor_validation()
    validation["expected_ranking"] = ["policy_clean"]
    check = evaluate_wam_calibration_anchors(validation)
    assert check["anchors_passed"] is False
    assert "calibration_anchor_set_too_small" in check["blockers"]


# --- apply_wam_score_claim_gate ----------------------------------------------


def _passing_consistency() -> dict:
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    return score_wam_consistency(_trajectory(points), _trajectory(points))


def test_grade_ladder_orders_fixture_below_review_below_calibrated() -> None:
    assert WAM_SCORE_CLAIM_GRADES == (
        "fixture_evaluator_only",
        "review_grade",
        "calibrated_evaluator_grade",
    )


def test_above_review_claim_without_evidence_fails_closed_to_fixture() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=None,
        calibration_anchors=None,
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert gate["status"] == "failed_closed"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER in gate["blockers"]


def test_above_review_claim_with_only_consistency_fails_closed() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=_passing_consistency(),
        calibration_anchors=None,
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER in gate["blockers"]


def test_above_review_claim_with_only_anchors_fails_closed() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=None,
        calibration_anchors=evaluate_wam_calibration_anchors(_passing_anchor_validation()),
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER in gate["blockers"]


def test_review_grade_claim_without_evidence_is_capped_not_demoted() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="review_grade",
        consistency=None,
        calibration_anchors=None,
    )
    assert gate["granted_grade"] == "review_grade"
    assert gate["max_allowed_grade"] == "review_grade"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER not in gate["blockers"]
    assert gate["upgrade_requirements"]


def test_calibrated_grade_allowed_with_passing_consistency_and_anchors() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=_passing_consistency(),
        calibration_anchors=evaluate_wam_calibration_anchors(_passing_anchor_validation()),
    )
    assert gate["granted_grade"] == "calibrated_evaluator_grade"
    assert gate["max_allowed_grade"] == "calibrated_evaluator_grade"
    assert gate["status"] == "granted"
    assert gate["blockers"] == []
    assert gate["consistency"]["consistency_score"] == pytest.approx(1.0)
    assert gate["calibration_anchors"]["anchor_set"]


def test_failing_consistency_score_blocks_calibrated_grade() -> None:
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    rollout = _trajectory([[0.0, 0.0, 0.0], [3.0, 3.0, 0.0], [6.0, 6.0, 0.0]])
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=score_wam_consistency(rollout, reference),
        calibration_anchors=evaluate_wam_calibration_anchors(_passing_anchor_validation()),
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER in gate["blockers"]


def test_fixture_evaluator_only_run_never_exceeds_fixture_grade() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="review_grade",
        consistency=_passing_consistency(),
        calibration_anchors=evaluate_wam_calibration_anchors(_passing_anchor_validation()),
        fixture_evaluator_only=True,
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert gate["max_allowed_grade"] == "fixture_evaluator_only"


def test_unrecognized_requested_grade_fails_closed() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="deployment_grade",
        consistency=None,
        calibration_anchors=None,
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert "wam_score_claim_grade_unrecognized" in gate["blockers"]


def test_gate_payload_always_carries_anchor_set_and_consistency_number() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="review_grade",
        consistency=None,
        calibration_anchors=None,
    )
    assert "consistency_score" in gate["consistency"]
    assert "anchor_set" in gate["calibration_anchors"]
    boundary = gate["claim_boundary"]
    assert boundary["score_above_review_grade_requires_consistency_and_calibration_anchors"] is True
    assert boundary["bare_wam_score_reporting_forbidden"] is True
    assert boundary["rank_fidelity_result_proven"] is False
    assert boundary["public_claim_upgrade_allowed"] is False
