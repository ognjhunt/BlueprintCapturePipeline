"""Tests for the small-sample rank-fidelity statistics."""

from __future__ import annotations

import math

import pytest

from blueprint_pipeline import rank_fidelity_statistics as stats


def test_normal_ppf_matches_known_quantiles() -> None:
    assert stats.normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert stats.normal_ppf(0.025) == pytest.approx(-1.959964, abs=1e-6)
    assert stats.normal_ppf(0.8) == pytest.approx(0.841621, abs=1e-6)
    assert stats.normal_ppf(0.99) == pytest.approx(2.326348, abs=1e-6)
    assert stats.normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert stats.normal_ppf(0.0) is None
    assert stats.normal_ppf(1.0) is None


def test_normal_ppf_round_trips_through_the_cdf() -> None:
    for probability in (0.01, 0.1, 0.35, 0.5, 0.77, 0.9, 0.999):
        assert stats.normal_cdf(stats.normal_ppf(probability)) == pytest.approx(
            probability, abs=1e-9
        )


def test_fisher_z_interval_reproduces_published_cohort_widths() -> None:
    """The published headlines rest on very few policies; the width shows it."""

    # RoboWorld: r = 0.989 over 8 policies.
    roboworld = stats.fisher_z_interval(0.989, 8)
    assert roboworld["defined"] is True
    assert roboworld["lower"] == pytest.approx(0.938, abs=0.002)

    # SC3-Eval: r = 0.929 over 7 policy checkpoints.
    sc3 = stats.fisher_z_interval(0.929, 7)
    assert sc3["defined"] is True
    assert sc3["lower"] == pytest.approx(0.586, abs=0.002)


def test_fisher_z_interval_is_undefined_where_it_must_be() -> None:
    assert stats.fisher_z_interval(0.9, 3)["undefined_reason"] == "sample_count_lt_4"
    assert (
        stats.fisher_z_interval(1.0, 20)["undefined_reason"]
        == "correlation_at_unit_boundary"
    )
    assert stats.fisher_z_interval(float("nan"), 20)["defined"] is False
    assert stats.fisher_z_interval(0.9, 7.5)["undefined_reason"] == (
        "sample_count_not_an_integer"
    )


def test_required_sample_count_for_lower_bound() -> None:
    # Certifying r >= 0.90 from a 0.95 point estimate needs ~33 policies.
    assert stats.required_sample_count_for_lower_bound(0.95, 0.90) == 33
    assert stats.required_sample_count_for_lower_bound(0.95, 0.80) == 11
    # A target at or above the estimate is unreachable at any sample size.
    assert stats.required_sample_count_for_lower_bound(0.90, 0.90) is None
    assert stats.required_sample_count_for_lower_bound(0.80, 0.90) is None


def test_wilson_interval_stays_inside_the_unit_range() -> None:
    perfect = stats.wilson_interval(28, 28)
    assert perfect["estimate"] == 1.0
    assert perfect["upper"] == 1.0
    assert 0.0 < perfect["lower"] < 1.0

    empty = stats.wilson_interval(0, 10)
    assert empty["lower"] == 0.0
    assert 0.0 < empty["upper"] < 1.0

    assert stats.wilson_interval(5, 0)["defined"] is False
    assert stats.wilson_interval(11, 10)["defined"] is False


def test_minimum_detectable_difference_shrinks_with_trials() -> None:
    values = [stats.minimum_detectable_difference(n) for n in (10, 50, 200, 1000)]
    assert all(values[index] > values[index + 1] for index in range(len(values) - 1))
    # Resolving a five-point success-rate gap takes many hundreds of trials.
    assert stats.minimum_detectable_difference(100) > 0.05
    assert stats.minimum_detectable_difference(1000) < 0.09
    assert stats.minimum_detectable_difference(0) is None


def test_minimum_detectable_difference_curve_shape() -> None:
    curve = stats.minimum_detectable_difference_curve([10, 100, 1000])
    assert [point["trials_per_arm"] for point in curve["points"]] == [10, 100, 1000]
    assert curve["baseline_success_rate"] == 0.5
    assert "design guide" in curve["approximation_note"]


def test_fisher_exact_greater_on_the_three_seed_ladder_case() -> None:
    """Three Bernoulli trials cannot separate adjacent rungs."""

    # 3/3 versus 2/3 -- a single success apart.
    assert stats.fisher_exact_greater(3, 3, 2, 3) == pytest.approx(0.5)
    # Even the most extreme three-trial table only just reaches 0.05.
    assert stats.fisher_exact_greater(3, 3, 0, 3) == pytest.approx(0.05)
    # With real replicate counts the same rate gap becomes decisive.
    assert stats.fisher_exact_greater(60, 60, 40, 60) < 1e-6
    assert stats.fisher_exact_greater(0, 0, 1, 1) is None


def test_fisher_exact_greater_is_symmetric_under_no_effect() -> None:
    assert stats.fisher_exact_greater(5, 10, 5, 10) > 0.5


def test_bootstrap_interval_reliability_flags_silent_drops() -> None:
    healthy = stats.bootstrap_interval_reliability(
        sample_count=12, replicates_attempted=10_000, replicates_defined=10_000
    )
    assert healthy["reliable"] is True
    assert healthy["undefined_replicate_fraction"] == 0.0

    dropped = stats.bootstrap_interval_reliability(
        sample_count=12, replicates_attempted=10_000, replicates_defined=8_000
    )
    assert dropped["reliable"] is False
    assert "undefined_replicate_fraction_above_threshold" in dropped["unreliable_reasons"]
    assert dropped["undefined_replicate_fraction"] == pytest.approx(0.2)

    tiny = stats.bootstrap_interval_reliability(
        sample_count=3, replicates_attempted=10_000, replicates_defined=10_000
    )
    assert tiny["reliable"] is False
    assert any("sample_count_lt" in reason for reason in tiny["unreliable_reasons"])


def test_paired_difference_interval() -> None:
    zero = stats.paired_difference_interval([0.0] * 10)
    assert zero["estimate"] == 0.0
    assert zero["lower"] == 0.0 and zero["upper"] == 0.0

    positive = stats.paired_difference_interval([0.2, 0.25, 0.18, 0.22, 0.21])
    assert positive["defined"] is True
    assert positive["lower"] > 0.0

    assert stats.paired_difference_interval([0.1])["defined"] is False


def test_non_finite_inputs_are_rejected_everywhere() -> None:
    assert stats.fisher_z_interval(math.inf, 10)["defined"] is False
    assert stats.minimum_detectable_difference(math.nan) is None
    assert stats.wilson_interval(math.nan, 10)["defined"] is False
    assert stats.required_sample_count_for_lower_bound(math.nan, 0.5) is None
