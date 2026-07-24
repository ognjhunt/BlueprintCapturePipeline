"""World-model-free control rankers for attributing evaluator rank fidelity.

Every published world-model evaluator result Blueprint tracks reports its
ablations *internally* -- a configuration of the method against other
configurations of the same method.  None of them report what a ranker with no
world model at all achieves on the same policy cohort.  That omission matters
commercially, not just scientifically: a cohort of independently-authored
policies spans a wide quality range, and wide-range ordering is often
recoverable from cheap signals that never generate a frame.  If a trivial proxy
recovers most of the ordering, then the evaluator's marginal contribution is the
gap above that proxy, not the whole headline number, and the compute economics
change completely.

This module supplies the missing control arm.  Each baseline ranks the same
policies over the same reference, using only evidence that exists *before* any
world model runs:

``action_chunk_jerk``
    Smoothness of the commanded action stream.  Reads only what the policy
    emitted.
``gripper_toggle_rate``
    How often the gripper command changes state.  Thrashing grippers correlate
    with poor manipulation without any scene understanding.
``episode_timeout_rate``
    Fraction of episodes that ran to the step limit without terminating.
``first_frame_prior``
    A statistic of the real initial observation only.  Under matched initial
    conditions this carries no policy information at all, which makes it a
    deliberate null control rather than a competitive baseline.
``constant``
    Assigns every policy the same score: a pure null with no ordering.
``seeded_pseudo_random``
    Deterministic noise: a null control with variance, so a reader can see what
    "no signal" looks like under the same interval machinery.

The report answers one question -- how much rank agreement does the evaluator add
over the best world-model-free baseline -- and answers it with a paired bootstrap
over policies rather than by subtracting two point estimates.

A baseline that ranks well is not an evaluator.  These proxies read commanded
actions, not consequences; they cannot detect whether the box actually moved, and
they are not proposed as a product.  They exist to price the evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .benchmark_protocol import external_rank_metrics
from .common import read_json_any, utc_now_iso, write_json
from .rank_fidelity_statistics import (
    bootstrap_interval_reliability,
    fisher_z_interval,
    paired_difference_interval,
)


REQUEST_SCHEMA_VERSION = "control_ranker_request.v1"
REPORT_SCHEMA_VERSION = "control_ranker_report.v1"

DEFAULT_BOOTSTRAP_SEED = 20260724
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
# Comparison metric for the attribution headline.  Pairwise ordering is the
# quantity a buyer's question reduces to and it degrades gracefully at the
# cohort sizes evaluator qualification actually runs at.
ATTRIBUTION_METRIC = "pairwise_ordering_accuracy"

BASELINE_IDS = (
    "action_chunk_jerk",
    "gripper_toggle_rate",
    "episode_timeout_rate",
    "first_frame_prior",
    "constant",
    "seeded_pseudo_random",
)
# Baselines that carry no policy information by construction.  They are reported
# separately so a reader can distinguish "a cheap signal ranks well" from "the
# metric flatters everything at this cohort size".
NULL_BASELINE_IDS = ("constant", "seeded_pseudo_random", "first_frame_prior")


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    out: list[float] = []
    for item in value:
        number = _number(item)
        if number is None:
            return None
        out.append(number)
    return out


def _digest(value: Any) -> str:
    text = _string(value).lower()
    return text if len(text) == 64 and all(c in "0123456789abcdef" for c in text) else ""


def action_chunk_jerk(action_sequence: Sequence[Any]) -> float | None:
    """Mean magnitude of the third finite difference of an action stream.

    Uses the third difference because it is the discrete analogue of jerk when
    actions are positional; the absolute scale is not comparable across action
    schemas, which is why this is only ever used to *rank* policies that share
    one schema.
    """

    vectors = [vector for vector in (_vector(step) for step in action_sequence) if vector]
    if len(vectors) < 4:
        return None
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        return None
    magnitudes: list[float] = []
    for index in range(len(vectors) - 3):
        third = [
            vectors[index + 3][dim]
            - 3.0 * vectors[index + 2][dim]
            + 3.0 * vectors[index + 1][dim]
            - vectors[index][dim]
            for dim in range(width)
        ]
        magnitudes.append(math.sqrt(sum(component**2 for component in third)))
    if not magnitudes:
        return None
    return sum(magnitudes) / len(magnitudes)


def gripper_toggle_rate(
    action_sequence: Sequence[Any],
    *,
    gripper_dimension: int,
    threshold: float = 0.5,
) -> float | None:
    """Gripper state changes per step, from the commanded action stream."""

    vectors = [vector for vector in (_vector(step) for step in action_sequence) if vector]
    if len(vectors) < 2:
        return None
    if any(gripper_dimension >= len(vector) for vector in vectors):
        return None
    states = [vector[gripper_dimension] >= threshold for vector in vectors]
    toggles = sum(1 for index in range(1, len(states)) if states[index] != states[index - 1])
    return toggles / (len(states) - 1)


def _policy_baseline_scores(
    policy_rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, dict[str, float | None]]:
    """Compute every baseline's per-policy score.

    Signs are normalised so that a higher score always means "predicted better",
    matching the convention of the reference scores they are compared against.
    """

    rng = random.Random(seed)
    scores: dict[str, dict[str, float | None]] = {
        baseline: {} for baseline in BASELINE_IDS
    }
    for row in policy_rows:
        policy_id = _string(row.get("policy_id"))
        episodes = _rows(row.get("episodes"))
        jerks: list[float] = []
        toggles: list[float] = []
        timeouts: list[float] = []
        priors: list[float] = []
        gripper_dimension = row.get("gripper_dimension")
        for episode in episodes:
            actions = episode.get("action_sequence") or []
            jerk = action_chunk_jerk(actions)
            if jerk is not None:
                jerks.append(jerk)
            if isinstance(gripper_dimension, int) and not isinstance(gripper_dimension, bool):
                toggle = gripper_toggle_rate(actions, gripper_dimension=gripper_dimension)
                if toggle is not None:
                    toggles.append(toggle)
            terminated = episode.get("terminated")
            if isinstance(terminated, bool):
                timeouts.append(0.0 if terminated else 1.0)
            prior = _number(episode.get("first_frame_statistic"))
            if prior is not None:
                priors.append(prior)

        def _mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        mean_jerk = _mean(jerks)
        mean_toggle = _mean(toggles)
        mean_timeout = _mean(timeouts)
        mean_prior = _mean(priors)
        # Negated so lower jerk / fewer toggles / fewer timeouts rank higher.
        scores["action_chunk_jerk"][policy_id] = (
            -mean_jerk if mean_jerk is not None else None
        )
        scores["gripper_toggle_rate"][policy_id] = (
            -mean_toggle if mean_toggle is not None else None
        )
        scores["episode_timeout_rate"][policy_id] = (
            -mean_timeout if mean_timeout is not None else None
        )
        scores["first_frame_prior"][policy_id] = mean_prior
        scores["constant"][policy_id] = 0.0
        scores["seeded_pseudo_random"][policy_id] = rng.random()
    return scores


def _metrics_with_intervals(
    predicted: Sequence[float], reference: Sequence[float]
) -> dict[str, Any]:
    metrics = external_rank_metrics(predicted, reference)
    enriched: dict[str, Any] = {}
    for name, value in metrics.items():
        entry: dict[str, Any] = {"estimate": round(value, 6) if value is not None else None}
        if name in {"pearson", "spearman"}:
            entry["fisher_z_interval_95"] = fisher_z_interval(value, len(predicted))
        enriched[name] = entry
    return enriched


def build_control_ranker_report(
    request: Mapping[str, Any],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Rank the cohort with world-model-free baselines and attribute the delta."""

    blockers: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("control_ranker_request_schema_missing_or_unsupported")

    reference_rows = _rows(request.get("reference_results"))
    evaluator_rows = _rows(request.get("evaluator_predictions"))
    policy_rows = _rows(request.get("policy_traces"))

    reference_by_policy = {
        _string(row.get("policy_id")): row for row in reference_rows if _string(row.get("policy_id"))
    }
    evaluator_by_policy = {
        _string(row.get("policy_id")): row for row in evaluator_rows if _string(row.get("policy_id"))
    }
    traces_by_policy = {
        _string(row.get("policy_id")): row for row in policy_rows if _string(row.get("policy_id"))
    }

    if not reference_by_policy:
        blockers.append("control_ranker_reference_results_missing")
    if not evaluator_by_policy:
        blockers.append("control_ranker_evaluator_predictions_missing")
    if not traces_by_policy:
        blockers.append("control_ranker_policy_traces_missing")

    # The attribution is only meaningful if every arm ranked exactly the same
    # cohort; a baseline evaluated on a different or smaller set of policies
    # would make the comparison a cohort artifact rather than a measurement.
    cohort = sorted(
        set(reference_by_policy) & set(evaluator_by_policy) & set(traces_by_policy)
    )
    if cohort != sorted(reference_by_policy) or cohort != sorted(evaluator_by_policy):
        blockers.append("control_ranker_cohort_mismatch_across_arms")
    for policy_id in cohort:
        expected = _digest(reference_by_policy[policy_id].get("checkpoint_sha256"))
        observed = _digest(evaluator_by_policy[policy_id].get("checkpoint_sha256"))
        trace_digest = _digest(traces_by_policy[policy_id].get("checkpoint_sha256"))
        if not expected or expected != observed or expected != trace_digest:
            blockers.append(f"control_ranker_checkpoint_digest_mismatch:{policy_id}")
    if len(cohort) < 3:
        blockers.append("control_ranker_requires_three_matched_policies")

    baseline_scores = _policy_baseline_scores(
        [traces_by_policy[policy_id] for policy_id in cohort], seed=bootstrap_seed
    )

    reference_values = [
        _number(reference_by_policy[policy_id].get("score")) for policy_id in cohort
    ]
    evaluator_values = [
        _number(evaluator_by_policy[policy_id].get("predicted_score")) for policy_id in cohort
    ]
    if any(value is None for value in reference_values):
        blockers.append("control_ranker_reference_score_invalid")
    if any(value is None for value in evaluator_values):
        blockers.append("control_ranker_evaluator_score_invalid")

    baselines: list[dict[str, Any]] = []
    evaluator_metrics: dict[str, Any] = {}
    attribution: dict[str, Any] = {}

    if not blockers:
        reference_clean = [float(value) for value in reference_values]
        evaluator_clean = [float(value) for value in evaluator_values]
        evaluator_metrics = _metrics_with_intervals(evaluator_clean, reference_clean)

        usable: dict[str, list[float]] = {}
        for baseline_id in BASELINE_IDS:
            per_policy = baseline_scores.get(baseline_id, {})
            values = [per_policy.get(policy_id) for policy_id in cohort]
            available = all(value is not None for value in values)
            row: dict[str, Any] = {
                "baseline_id": baseline_id,
                "is_null_control": baseline_id in NULL_BASELINE_IDS,
                "available": available,
                "uses_world_model": False,
                "per_policy_scores": {
                    policy_id: (round(value, 6) if value is not None else None)
                    for policy_id, value in zip(cohort, values)
                },
            }
            if available:
                clean = [float(value) for value in values]
                usable[baseline_id] = clean
                row["metrics"] = _metrics_with_intervals(clean, reference_clean)
            else:
                row["unavailable_reason"] = "insufficient_trace_evidence_for_baseline"
            baselines.append(row)

        def _metric_value(container: Mapping[str, Any]) -> float | None:
            return _mapping(container.get(ATTRIBUTION_METRIC)).get("estimate")

        evaluator_point = _metric_value(evaluator_metrics)
        scored_baselines = {
            baseline_id: _metric_value(
                next(row for row in baselines if row["baseline_id"] == baseline_id).get(
                    "metrics", {}
                )
            )
            for baseline_id in usable
        }
        informative = {
            baseline_id: value
            for baseline_id, value in scored_baselines.items()
            if value is not None and baseline_id not in NULL_BASELINE_IDS
        }
        best_baseline_id = max(informative, key=informative.get) if informative else None
        best_baseline_value = informative.get(best_baseline_id) if best_baseline_id else None

        # Paired bootstrap over policies: resample the cohort once per replicate
        # and recompute BOTH arms on the same resample, so the interval is on the
        # difference rather than on two independently-wide point estimates.
        differences: list[float] = []
        attempted = 0
        if best_baseline_id is not None and evaluator_point is not None:
            rng = random.Random(bootstrap_seed)
            baseline_clean = usable[best_baseline_id]
            indices = list(range(len(cohort)))
            for _ in range(bootstrap_replicates):
                attempted += 1
                sample = [rng.choice(indices) for _ in indices]
                sample_reference = [reference_clean[index] for index in sample]
                evaluator_metric = external_rank_metrics(
                    [evaluator_clean[index] for index in sample], sample_reference
                ).get(ATTRIBUTION_METRIC)
                baseline_metric = external_rank_metrics(
                    [baseline_clean[index] for index in sample], sample_reference
                ).get(ATTRIBUTION_METRIC)
                if (
                    evaluator_metric is not None
                    and baseline_metric is not None
                    and math.isfinite(evaluator_metric)
                    and math.isfinite(baseline_metric)
                ):
                    differences.append(evaluator_metric - baseline_metric)

        ordered = sorted(differences)
        def _percentile(fraction: float) -> float | None:
            if not ordered:
                return None
            position = min(
                len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1))))
            )
            return round(ordered[position], 6)

        marginal = (
            round(evaluator_point - best_baseline_value, 6)
            if evaluator_point is not None and best_baseline_value is not None
            else None
        )
        reliability = bootstrap_interval_reliability(
            sample_count=len(cohort),
            replicates_attempted=attempted,
            replicates_defined=len(differences),
        )
        attribution = {
            "metric": ATTRIBUTION_METRIC,
            "evaluator_value": evaluator_point,
            "best_world_model_free_baseline_id": best_baseline_id,
            "best_world_model_free_baseline_value": best_baseline_value,
            "marginal_contribution": marginal,
            "marginal_contribution_interval_95": [
                _percentile(0.025),
                _percentile(0.975),
            ],
            "marginal_contribution_paired_summary": paired_difference_interval(differences),
            "bootstrap_replicates_attempted": attempted,
            "bootstrap_replicates_defined": len(differences),
            "bootstrap_interval_reliability": reliability,
            "evaluator_exceeds_best_baseline": (
                marginal is not None and marginal > 0.0
            ),
            "evaluator_advantage_separated_from_zero": bool(
                (lower := _percentile(0.025)) is not None and lower > 0.0
            ),
            "null_control_values": {
                baseline_id: scored_baselines.get(baseline_id)
                for baseline_id in NULL_BASELINE_IDS
                if baseline_id in scored_baselines
            },
        }

    blockers = sorted(set(blockers))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "measured" if not blockers else "blocked",
        "cohort_policy_ids": cohort,
        "cohort_size": len(cohort),
        "evaluator_id": _string(request.get("evaluator_id")) or None,
        "evaluator_metrics": evaluator_metrics,
        "baselines": baselines,
        "attribution": attribution,
        "blockers": blockers,
        "claim_boundary": {
            "baselines_do_not_use_a_world_model": True,
            "baselines_read_commanded_actions_not_consequences": True,
            "a_winning_baseline_is_not_an_evaluator": True,
            "attribution_is_not_real_world_rank_fidelity": True,
            "marginal_contribution_is_scoped_to_this_cohort": True,
            "public_claim_upgrade_allowed": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank a policy cohort with world-model-free control baselines"
    )
    parser.add_argument("--input", required=True, help="control_ranker_request.v1 JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    args = parser.parse_args(argv)

    request = _mapping(read_json_any(Path(args.input)))
    report = build_control_ranker_report(
        request,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    write_json(Path(args.output), report)
    print(json.dumps({"path": args.output, "status": report["status"]}, sort_keys=True))
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
