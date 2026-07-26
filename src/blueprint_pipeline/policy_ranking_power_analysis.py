"""Reproducible sample-size resolution for the frozen RoboArena holdout.

The released OSCAR subset supplies 63 complete seven-policy sessions.  After
the preregistered 7-session pilot and 7-session calibration partitions, every
remaining session is held out.  Session is the conservative independent unit;
the 21 within-session policy pairs are correlated and are not counted as 21
independent observations.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .common import write_json
from .policy_ranking_thesis import canonical_sha256


def _binomial_upper_tail(n: int, probability: float, threshold: int) -> float:
    return sum(
        math.comb(n, successes)
        * probability**successes
        * (1.0 - probability) ** (n - successes)
        for successes in range(threshold, n + 1)
    )


def build_power_analysis(
    *,
    heldout_sessions: int = 49,
    null_accuracy: float = 0.5,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> dict[str, Any]:
    """Return conservative clustered and exact-binomial resolution estimates."""

    if heldout_sessions <= 0:
        raise ValueError("heldout_sessions_must_be_positive")
    if not 0.0 < alpha < 0.5 or not 0.5 < target_power < 1.0:
        raise ValueError("invalid_alpha_or_power")
    normal = NormalDist()
    worst_case_standard_error = 0.5 / math.sqrt(heldout_sessions)
    clustered_mde = (
        normal.inv_cdf(1.0 - alpha) + normal.inv_cdf(target_power)
    ) * worst_case_standard_error
    critical_successes = next(
        successes
        for successes in range(heldout_sessions + 1)
        if _binomial_upper_tail(heldout_sessions, null_accuracy, successes) <= alpha
    )
    candidate = null_accuracy
    while candidate < 1.0:
        if (
            _binomial_upper_tail(heldout_sessions, candidate, critical_successes)
            >= target_power
        ):
            break
        candidate += 0.0001
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_power_analysis.v1",
        "analysis_unit": "heldout_session_cluster",
        "heldout_session_count": heldout_sessions,
        "within_session_policy_pair_count": 21,
        "within_session_pairs_treated_as_independent": False,
        "test": {
            "alternative": "pairwise_ranking_accuracy_gt_0.5",
            "one_sided_alpha": alpha,
            "target_power": target_power,
            "null_accuracy": null_accuracy,
        },
        "bounded_cluster_mean_approximation": {
            "worst_case_standard_deviation": 0.5,
            "worst_case_standard_error": worst_case_standard_error,
            "minimum_detectable_accuracy_gain": clustered_mde,
            "minimum_detectable_accuracy": null_accuracy + clustered_mde,
        },
        "exact_binomial_reference": {
            "assumption": "one independent binary correctness outcome per heldout session",
            "critical_successes": critical_successes,
            "critical_observed_accuracy": critical_successes / heldout_sessions,
            "null_tail_probability": _binomial_upper_tail(
                heldout_sessions, null_accuracy, critical_successes
            ),
            "minimum_accuracy_for_target_power": min(candidate, 1.0),
            "power_at_minimum_accuracy": _binomial_upper_tail(
                heldout_sessions, min(candidate, 1.0), critical_successes
            ),
        },
        "sample_size_basis": (
            "All 49 complete released sessions remaining after the frozen 7-session pilot "
            "and 7-session calibration partitions; no larger complete released holdout exists."
        ),
        "interpretation": {
            "small_effects_below_resolution_are_expected_to_be_inconclusive": True,
            "pair_specific_power_may_be_lower_due_to_ties_or_missing_comparisons": True,
            "adjacent_policy_pairs_are_not_exempted_from_the_frozen_decision_rule": True,
            "wide_registered_confidence_intervals_produce_inconclusive_not_success": True,
        },
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_json(Path(args.output), build_power_analysis())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
