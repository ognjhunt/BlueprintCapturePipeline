"""Tests for the world-model-free control ranker."""

from __future__ import annotations

import hashlib

import pytest

from blueprint_pipeline import control_ranker


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _episode(*, jerk: float, toggles: int, terminated: bool, steps: int = 24):
    """Synthesise an action stream with a controlled amount of jerk."""

    actions = []
    for index in range(steps):
        # A smooth ramp plus an alternating component whose amplitude sets jerk.
        wobble = jerk * (1 if index % 2 == 0 else -1)
        gripper = 1.0 if (index // max(1, steps // max(1, toggles + 1))) % 2 else 0.0
        actions.append([index * 0.01 + wobble, index * 0.02, gripper])
    return {
        "action_sequence": actions,
        "terminated": terminated,
        "first_frame_statistic": 0.5,
    }


def _request(*, evaluator_scores=None, jerks=None):
    policies = ["alpha", "bravo", "charlie", "delta", "echo"]
    reference = {"alpha": 0.9, "bravo": 0.7, "charlie": 0.5, "delta": 0.3, "echo": 0.1}
    jerks = jerks or {"alpha": 0.0, "bravo": 0.02, "charlie": 0.04, "delta": 0.06, "echo": 0.08}
    evaluator_scores = evaluator_scores or reference
    return {
        "schema_version": control_ranker.REQUEST_SCHEMA_VERSION,
        "evaluator_id": "oscar_wam",
        "reference_results": [
            {
                "policy_id": policy,
                "checkpoint_sha256": _digest(policy),
                "score": reference[policy],
            }
            for policy in policies
        ],
        "evaluator_predictions": [
            {
                "policy_id": policy,
                "checkpoint_sha256": _digest(policy),
                "predicted_score": evaluator_scores[policy],
            }
            for policy in policies
        ],
        "policy_traces": [
            {
                "policy_id": policy,
                "checkpoint_sha256": _digest(policy),
                "gripper_dimension": 2,
                "episodes": [
                    _episode(
                        jerk=jerks[policy],
                        toggles=index,
                        terminated=reference[policy] > 0.2,
                    )
                    for index in range(1, 3)
                ],
            }
            for policy in policies
        ],
    }


def test_report_measures_and_ranks_every_baseline() -> None:
    report = control_ranker.build_control_ranker_report(
        _request(), bootstrap_replicates=200
    )

    assert report["status"] == "measured", report["blockers"]
    assert report["cohort_size"] == 5
    ids = {row["baseline_id"] for row in report["baselines"]}
    assert ids == set(control_ranker.BASELINE_IDS)
    assert all(row["uses_world_model"] is False for row in report["baselines"])


def test_a_cheap_proxy_that_tracks_quality_is_credited() -> None:
    """Jerk is constructed to track the reference here, so it must score well."""

    report = control_ranker.build_control_ranker_report(
        _request(), bootstrap_replicates=200
    )
    jerk = next(
        row for row in report["baselines"] if row["baseline_id"] == "action_chunk_jerk"
    )

    assert jerk["available"] is True
    assert jerk["metrics"]["pairwise_ordering_accuracy"]["estimate"] == pytest.approx(1.0)


def test_perfect_evaluator_shows_no_marginal_gain_over_a_perfect_proxy() -> None:
    """The point of the control arm: a headline number is not a contribution."""

    report = control_ranker.build_control_ranker_report(
        _request(), bootstrap_replicates=200
    )
    attribution = report["attribution"]

    assert attribution["evaluator_value"] == pytest.approx(1.0)
    assert attribution["best_world_model_free_baseline_value"] == pytest.approx(1.0)
    assert attribution["marginal_contribution"] == pytest.approx(0.0)
    assert attribution["evaluator_exceeds_best_baseline"] is False
    assert attribution["evaluator_advantage_separated_from_zero"] is False


def test_evaluator_that_beats_the_proxies_is_credited() -> None:
    scrambled_jerks = {
        "alpha": 0.08,
        "bravo": 0.0,
        "charlie": 0.06,
        "delta": 0.02,
        "echo": 0.04,
    }
    report = control_ranker.build_control_ranker_report(
        _request(jerks=scrambled_jerks), bootstrap_replicates=400
    )
    attribution = report["attribution"]

    assert attribution["marginal_contribution"] > 0.0
    assert attribution["evaluator_exceeds_best_baseline"] is True


def test_null_controls_are_reported_separately_and_never_win() -> None:
    report = control_ranker.build_control_ranker_report(
        _request(), bootstrap_replicates=200
    )

    assert report["attribution"]["best_world_model_free_baseline_id"] not in (
        control_ranker.NULL_BASELINE_IDS
    )
    null_values = report["attribution"]["null_control_values"]
    assert "constant" in null_values
    assert "seeded_pseudo_random" in null_values


def test_cohort_mismatch_fails_closed() -> None:
    request = _request()
    request["evaluator_predictions"] = request["evaluator_predictions"][:3]

    report = control_ranker.build_control_ranker_report(request, bootstrap_replicates=10)

    assert report["status"] == "blocked"
    assert "control_ranker_cohort_mismatch_across_arms" in report["blockers"]
    assert report["attribution"] == {}


def test_checkpoint_digest_mismatch_fails_closed() -> None:
    request = _request()
    request["evaluator_predictions"][0]["checkpoint_sha256"] = _digest("tampered")

    report = control_ranker.build_control_ranker_report(request, bootstrap_replicates=10)

    assert report["status"] == "blocked"
    assert any(
        item.startswith("control_ranker_checkpoint_digest_mismatch")
        for item in report["blockers"]
    )


def test_small_cohort_marks_the_interval_unreliable() -> None:
    request = _request()
    keep = {"alpha", "bravo", "charlie"}
    for key in ("reference_results", "evaluator_predictions", "policy_traces"):
        request[key] = [row for row in request[key] if row["policy_id"] in keep]

    report = control_ranker.build_control_ranker_report(request, bootstrap_replicates=200)

    assert report["status"] == "measured", report["blockers"]
    reliability = report["attribution"]["bootstrap_interval_reliability"]
    assert reliability["reliable"] is False
    assert any("sample_count_lt" in reason for reason in reliability["unreliable_reasons"])


def test_action_chunk_jerk_needs_four_steps() -> None:
    assert control_ranker.action_chunk_jerk([[0.0], [1.0], [2.0]]) is None
    assert control_ranker.action_chunk_jerk([[0.0], [1.0], [2.0], [3.0]]) == pytest.approx(0.0)


def test_gripper_toggle_rate_counts_transitions() -> None:
    actions = [[0.0], [1.0], [0.0], [1.0]]
    assert control_ranker.gripper_toggle_rate(
        actions, gripper_dimension=0
    ) == pytest.approx(1.0)
    steady = [[0.0], [0.0], [0.0]]
    assert control_ranker.gripper_toggle_rate(steady, gripper_dimension=0) == 0.0


def test_claim_boundary_refuses_to_upgrade() -> None:
    report = control_ranker.build_control_ranker_report(
        _request(), bootstrap_replicates=10
    )
    boundary = report["claim_boundary"]

    assert boundary["a_winning_baseline_is_not_an_evaluator"] is True
    assert boundary["attribution_is_not_real_world_rank_fidelity"] is True
    assert boundary["public_claim_upgrade_allowed"] is False
