"""Reproducible sample-size resolution for the frozen RoboArena holdout.

The released OSCAR subset supplies 63 complete seven-policy sessions.  After
the preregistered 7-session pilot and 7-session calibration partitions, every
remaining session is held out.  Session is the conservative independent unit;
the 21 within-session policy pairs are correlated and are not counted as 21
independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .common import write_json
from .policy_ranking_thesis import canonical_sha256
from .policy_ranking_thesis import _benchmark_session_labels


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


def build_label_basis_power_sensitivity(
    *,
    protocol: dict[str, Any],
    roboarena_root: str | Path,
    heldout_sessions: int = 49,
) -> dict[str, Any]:
    """Project held-out resolution from pilot label informativeness.

    This is an exploratory sensitivity analysis over already-open pilot labels.
    It does not change the registered primary label basis, thresholds, or
    decision rule.  A session remains the independent unit even when it has
    several informative policy pairs.
    """

    policies = list(protocol["policies"])
    pilot_sessions = list(protocol["partitions"]["pilot"])
    root = Path(roboarena_root).resolve()
    counts = {
        basis: {"sessions_with_any_informative_pair": 0, "informative_pair_count": 0}
        for basis in (
            "binary_success",
            "binary_then_partial",
            "preference_winner_vs_rest",
        )
    }
    for session_id in pilot_sessions:
        labels, preferred = _benchmark_session_labels(
            root / "evaluation_sessions" / session_id / "metadata.yaml"
        )
        per_session = {basis: 0 for basis in counts}
        for left_policy, right_policy in combinations(policies, 2):
            left = labels[left_policy]
            right = labels[right_policy]
            binary_delta = left["binary_success"] - right["binary_success"]
            if binary_delta != 0:
                per_session["binary_success"] += 1
                per_session["binary_then_partial"] += 1
            elif left["partial_success"] != right["partial_success"]:
                per_session["binary_then_partial"] += 1
            if preferred in {left_policy, right_policy}:
                per_session["preference_winner_vs_rest"] += 1
        for basis, pair_count in per_session.items():
            counts[basis]["informative_pair_count"] += pair_count
            counts[basis]["sessions_with_any_informative_pair"] += int(pair_count > 0)

    basis_results: dict[str, Any] = {}
    for basis, observed in counts.items():
        fraction = observed["sessions_with_any_informative_pair"] / len(pilot_sessions)
        projected = max(1, round(heldout_sessions * fraction))
        resolution = build_power_analysis(heldout_sessions=projected)
        basis_results[basis] = {
            **observed,
            "pilot_session_count": len(pilot_sessions),
            "pilot_informative_session_fraction": fraction,
            "projected_heldout_informative_session_count": projected,
            "projection_rule": "round(heldout_sessions * pilot_informative_session_fraction)",
            "projected_cluster_mde_accuracy": resolution[
                "bounded_cluster_mean_approximation"
            ]["minimum_detectable_accuracy"],
            "projected_exact_binomial_minimum_accuracy_for_80pct_power": resolution[
                "exact_binomial_reference"
            ]["minimum_accuracy_for_target_power"],
        }

    result: dict[str, Any] = {
        "schema_version": "policy_ranking_label_basis_power_sensitivity.v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_partition": "pilot",
        "source_labels_opened": True,
        "calibration_labels_opened": True,
        "heldout_labels_opened": False,
        "heldout_session_count": heldout_sessions,
        "within_session_pairs_treated_as_independent": False,
        "basis_results": basis_results,
        "interpretation": {
            "exploratory_sensitivity_only": True,
            "uses_pilot_labels_only": True,
            "registered_primary_basis_unchanged": "binary_then_partial",
            "registered_decision_rule_unchanged": True,
            "pilot_fraction_may_not_repeat_in_heldout": True,
            "low_binary_informativeness_can_make_binary_only_results_inconclusive": True,
        },
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--roboarena-root")
    parser.add_argument("--label-basis-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    if args.label_basis_sensitivity:
        if not args.protocol or not args.roboarena_root:
            parser.error(
                "--label-basis-sensitivity requires --protocol and --roboarena-root"
            )
        protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
        result = build_label_basis_power_sensitivity(
            protocol=protocol,
            roboarena_root=args.roboarena_root,
        )
    else:
        result = build_power_analysis()
    write_json(Path(args.output), result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
