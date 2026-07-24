"""Shared small-sample statistics for evaluator rank-fidelity reporting.

Blueprint gates public rank-fidelity claims on a confidence-interval *lower
bound* rather than a point estimate.  That is the correct posture, but it makes
the number of independent policies -- not the number of rollouts -- the binding
constraint on every claim the platform can make.  A correlation computed over
seven policies carries roughly four degrees of freedom no matter how many
rollouts were generated per policy, so a headline ``r`` near 0.95 can still be
consistent with a true correlation below 0.7.

This module collects the small-sample tools that make that visible:

* :func:`fisher_z_interval` -- a parametric interval for Pearson/Spearman that,
  unlike a resample, is defined at every sample size above three;
* :func:`required_sample_count_for_lower_bound` -- how many policies a cohort
  needs before a given lower bound is even reachable;
* :func:`wilson_interval` -- a proportion interval that stays inside ``[0, 1]``
  at the small trial counts pairwise-ordering accuracy actually runs at;
* :func:`minimum_detectable_difference` / :func:`minimum_detectable_difference_curve`
  -- the resolving power of a design, which is the quantity a buyer comparing
  two of their own checkpoints is actually asking about; and
* :func:`fisher_exact_greater` -- an exact one-sided test for ordering claims
  made from Bernoulli outcomes at single-digit replicate counts.

Everything here is pure arithmetic over caller-supplied numbers.  Nothing in
this module measures a robot, runs a model, or upgrades a claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


STATISTICS_METHOD_VERSION = "rank_fidelity_statistics.v1"

# Below this many paired observations a bootstrap percentile interval is not a
# meaningful summary: the resample space is small enough that the interval is
# dominated by which points were duplicated rather than by sampling variation.
MIN_RELIABLE_BOOTSTRAP_SAMPLE_COUNT = 8

# A resample that is constant in either coordinate yields an undefined Pearson
# value.  Dropping those silently narrows the reported interval, so callers must
# treat a high undefined fraction as an unreliable interval rather than a tight
# one.
MAX_UNDEFINED_REPLICATE_FRACTION = 0.05

DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def normal_cdf(value: float) -> float:
    """Standard normal CDF."""

    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_ppf(probability: float) -> float | None:
    """Inverse standard normal CDF.

    Implemented by bisection against :func:`normal_cdf`, which is exact up to
    the accuracy of ``math.erf``.  A rational approximation would be faster, but
    this is called a handful of times per report and a transcription error in a
    coefficient table would silently corrupt every interval and power
    calculation in the platform.  Monotone bisection cannot do that.
    """

    value = _number(probability)
    if value is None or not 0.0 < value < 1.0:
        return None
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if normal_cdf(middle) < value:
            low = middle
        else:
            high = middle
        if high - low < 1e-15:
            break
    return (low + high) / 2.0


def _two_sided_z(confidence: float) -> float | None:
    if not 0.0 < confidence < 1.0:
        return None
    return normal_ppf(1.0 - (1.0 - confidence) / 2.0)


def fisher_z_interval(
    correlation: Any,
    sample_count: Any,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Fisher z-transformed interval for a correlation coefficient.

    Returns a structured result at every input, including the reasons an
    interval could not be formed, so a caller can record *why* a number is
    missing rather than emitting a bare ``None``.
    """

    result: dict[str, Any] = {
        "method": "fisher_z.v1",
        "confidence": confidence,
        "estimate": None,
        "lower": None,
        "upper": None,
        "sample_count": None,
        "degrees_of_freedom": None,
        "defined": False,
        "undefined_reason": None,
    }
    value = _number(correlation)
    count = _number(sample_count)
    if value is None:
        result["undefined_reason"] = "correlation_not_finite"
        return result
    result["estimate"] = round(value, 6)
    if count is None or count != int(count):
        result["undefined_reason"] = "sample_count_not_an_integer"
        return result
    count = int(count)
    result["sample_count"] = count
    result["degrees_of_freedom"] = max(0, count - 3)
    if count < 4:
        # The Fisher standard error is 1/sqrt(n-3); at n<=3 it is undefined or
        # infinite, so no interval exists at any confidence level.
        result["undefined_reason"] = "sample_count_lt_4"
        return result
    if abs(value) >= 1.0:
        result["undefined_reason"] = "correlation_at_unit_boundary"
        return result
    critical = _two_sided_z(confidence)
    if critical is None:
        result["undefined_reason"] = "confidence_out_of_range"
        return result
    transformed = math.atanh(value)
    standard_error = 1.0 / math.sqrt(count - 3)
    result["lower"] = round(math.tanh(transformed - critical * standard_error), 6)
    result["upper"] = round(math.tanh(transformed + critical * standard_error), 6)
    result["standard_error"] = round(standard_error, 6)
    result["defined"] = True
    return result


def required_sample_count_for_lower_bound(
    correlation: Any,
    target_lower_bound: Any,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    maximum: int = 100_000,
) -> int | None:
    """Independent observations needed before ``target_lower_bound`` is reachable.

    Answers the planning question directly: assuming the point estimate holds,
    how many distinct policies must a cohort contain before the gate on the
    interval's lower bound can pass at all?
    """

    value = _number(correlation)
    target = _number(target_lower_bound)
    critical = _two_sided_z(confidence)
    if value is None or target is None or critical is None:
        return None
    if abs(value) >= 1.0 or abs(target) >= 1.0:
        return None
    if value <= target:
        return None
    separation = math.atanh(value) - math.atanh(target)
    if separation <= 0.0:
        return None
    required = 3.0 + (critical / separation) ** 2
    count = int(math.ceil(required))
    return count if count <= maximum else None


def wilson_interval(
    successes: Any,
    trials: Any,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because pairwise-ordering accuracy
    routinely sits near 1.0 over a few dozen pairs, where a Wald interval runs
    past the unit boundary and understates uncertainty.
    """

    result: dict[str, Any] = {
        "method": "wilson_score.v1",
        "confidence": confidence,
        "estimate": None,
        "lower": None,
        "upper": None,
        "trials": None,
        "defined": False,
        "undefined_reason": None,
    }
    success_count = _number(successes)
    trial_count = _number(trials)
    critical = _two_sided_z(confidence)
    if trial_count is None or trial_count < 1:
        result["undefined_reason"] = "trials_missing_or_non_positive"
        return result
    if success_count is None or success_count < 0 or success_count > trial_count:
        result["undefined_reason"] = "successes_out_of_range"
        return result
    if critical is None:
        result["undefined_reason"] = "confidence_out_of_range"
        return result
    result["trials"] = trial_count
    proportion = success_count / trial_count
    result["estimate"] = round(proportion, 6)
    denominator = 1.0 + critical**2 / trial_count
    center = (proportion + critical**2 / (2.0 * trial_count)) / denominator
    margin = (
        critical
        * math.sqrt(
            proportion * (1.0 - proportion) / trial_count
            + critical**2 / (4.0 * trial_count**2)
        )
        / denominator
    )
    result["lower"] = round(max(0.0, center - margin), 6)
    result["upper"] = round(min(1.0, center + margin), 6)
    result["defined"] = True
    return result


def minimum_detectable_difference(
    trials_per_arm: Any,
    *,
    baseline_rate: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float | None:
    """Smallest success-rate gap a two-arm design can resolve.

    Uses the standard pooled two-proportion approximation
    ``delta = (z_{1-alpha/2} + z_{power}) * sqrt(2 * p * (1 - p) / n)``.  It is
    an approximation -- the pooled variance is evaluated at the baseline rate
    rather than at the (unknown) alternative -- and is reported as such.

    This is the quantity behind "is my 80k checkpoint better than my 60k
    checkpoint": if the returned value exceeds the difference a buyer cares
    about, the honest answer is *indistinguishable at this trial count*, and no
    amount of ranking machinery changes that.
    """

    count = _number(trials_per_arm)
    rate = _number(baseline_rate)
    if count is None or count < 1:
        return None
    if rate is None or not 0.0 < rate < 1.0:
        return None
    alpha_z = _two_sided_z(1.0 - alpha)
    power_z = normal_ppf(power)
    if alpha_z is None or power_z is None:
        return None
    delta = (alpha_z + power_z) * math.sqrt(2.0 * rate * (1.0 - rate) / count)
    return round(min(1.0, delta), 6)


def minimum_detectable_difference_curve(
    trial_counts: Sequence[Any],
    *,
    baseline_rate: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> dict[str, Any]:
    """Resolving power across a range of per-arm trial counts."""

    points: list[dict[str, Any]] = []
    for candidate in trial_counts:
        count = _number(candidate)
        if count is None or count < 1 or count != int(count):
            continue
        points.append(
            {
                "trials_per_arm": int(count),
                "minimum_detectable_success_rate_difference": minimum_detectable_difference(
                    int(count), baseline_rate=baseline_rate, alpha=alpha, power=power
                ),
            }
        )
    return {
        "method": "pooled_two_proportion_normal_approximation.v1",
        "baseline_success_rate": round(float(baseline_rate), 6),
        "alpha": alpha,
        "power": power,
        "approximation_note": (
            "pooled variance evaluated at the baseline rate; treat as a design "
            "guide, not an exact operating characteristic"
        ),
        "points": points,
    }


def _log_factorial(value: int) -> float:
    return math.lgamma(value + 1)


def fisher_exact_greater(
    better_successes: Any,
    better_trials: Any,
    worse_successes: Any,
    worse_trials: Any,
) -> float | None:
    """One-sided Fisher exact p-value for ``better`` outranking ``worse``.

    Returns the probability, under the null of equal success rates, of seeing at
    least the observed advantage.  This is the right test at the replicate
    counts a seeded policy ladder actually runs at: with three Bernoulli trials
    per rung the only attainable rates are 0, 1/3, 2/3 and 1, so a strict
    ordering of point estimates arises by chance often enough that it cannot by
    itself support an acceptance decision.
    """

    a = _number(better_successes)
    n1 = _number(better_trials)
    c = _number(worse_successes)
    n2 = _number(worse_trials)
    if None in (a, n1, c, n2):
        return None
    if any(value != int(value) for value in (a, n1, c, n2)):
        return None
    a, n1, c, n2 = int(a), int(n1), int(c), int(n2)
    if n1 < 1 or n2 < 1 or not 0 <= a <= n1 or not 0 <= c <= n2:
        return None
    total = n1 + n2
    successes = a + c

    def _table_probability(count: int) -> float:
        other = successes - count
        if other < 0 or other > n2 or count > n1:
            return 0.0
        return math.exp(
            _log_factorial(n1)
            + _log_factorial(n2)
            + _log_factorial(successes)
            + _log_factorial(total - successes)
            - _log_factorial(count)
            - _log_factorial(n1 - count)
            - _log_factorial(other)
            - _log_factorial(n2 - other)
            - _log_factorial(total)
        )

    upper = min(n1, successes)
    tail = sum(_table_probability(count) for count in range(a, upper + 1))
    return round(min(1.0, max(0.0, tail)), 9)


def bootstrap_interval_reliability(
    *,
    sample_count: Any,
    replicates_attempted: Any,
    replicates_defined: Any,
) -> dict[str, Any]:
    """Judge whether a percentile bootstrap interval may be reported as-is.

    A percentile interval computed from resamples that silently discarded their
    undefined replicates is narrower than the data support.  This records the
    discard explicitly and marks the interval unreliable when either the cohort
    is too small for resampling to explore meaningfully or too many replicates
    were dropped.
    """

    count = _number(sample_count)
    attempted = _number(replicates_attempted)
    defined = _number(replicates_defined)
    reasons: list[str] = []
    undefined_fraction: float | None = None
    if attempted is not None and attempted > 0 and defined is not None:
        undefined_fraction = round(max(0.0, (attempted - defined) / attempted), 6)
    if count is None or count < MIN_RELIABLE_BOOTSTRAP_SAMPLE_COUNT:
        reasons.append(
            f"sample_count_lt_{MIN_RELIABLE_BOOTSTRAP_SAMPLE_COUNT}_for_percentile_bootstrap"
        )
    if attempted is None or attempted <= 0:
        reasons.append("bootstrap_replicates_not_recorded")
    if defined is None:
        reasons.append("defined_replicate_count_not_recorded")
    elif undefined_fraction is not None and undefined_fraction > MAX_UNDEFINED_REPLICATE_FRACTION:
        reasons.append("undefined_replicate_fraction_above_threshold")
    return {
        "method": STATISTICS_METHOD_VERSION,
        "replicates_attempted": int(attempted) if attempted is not None else None,
        "replicates_defined": int(defined) if defined is not None else None,
        "undefined_replicate_fraction": undefined_fraction,
        "maximum_undefined_replicate_fraction": MAX_UNDEFINED_REPLICATE_FRACTION,
        "minimum_reliable_sample_count": MIN_RELIABLE_BOOTSTRAP_SAMPLE_COUNT,
        "reliable": not reasons,
        "unreliable_reasons": sorted(set(reasons)),
    }


def paired_difference_interval(
    differences: Sequence[Any],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Normal-approximation interval for a mean paired difference.

    Used to report whether an evaluator's rank agreement genuinely exceeds a
    world-model-free control ranker's, rather than comparing two point
    estimates that were each computed with wide uncertainty.
    """

    values = [value for value in (_number(item) for item in differences) if value is not None]
    result: dict[str, Any] = {
        "method": "paired_mean_difference_normal.v1",
        "confidence": confidence,
        "estimate": None,
        "lower": None,
        "upper": None,
        "sample_count": len(values),
        "defined": False,
        "undefined_reason": None,
    }
    if len(values) < 2:
        result["undefined_reason"] = "fewer_than_two_paired_observations"
        return result
    critical = _two_sided_z(confidence)
    if critical is None:
        result["undefined_reason"] = "confidence_out_of_range"
        return result
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    result["estimate"] = round(mean, 6)
    result["lower"] = round(mean - critical * standard_error, 6)
    result["upper"] = round(mean + critical * standard_error, 6)
    result["standard_error"] = round(standard_error, 6)
    result["defined"] = True
    return result
