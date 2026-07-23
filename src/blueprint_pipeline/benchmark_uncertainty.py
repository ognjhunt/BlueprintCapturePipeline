"""Hierarchical uncertainty, convergence, and sensitivity for policy ranking.

The report resamples policy, site, task-family/task, and initial-condition
clusters rather than treating every rollout as independent.  It complements
the frozen benchmark report with rollout-count convergence and leave-one-out
diagnostics; it does not upgrade a simulator result into real-world proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .benchmark_protocol import external_rank_metrics
from .common import read_json_any, write_json


REQUEST_SCHEMA_VERSION = "benchmark_uncertainty_request.v1"
REPORT_SCHEMA_VERSION = "benchmark_uncertainty_report.v1"
BOOTSTRAP_METHOD = "policy_site_task_initial_condition_hierarchical_percentile.v1"
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
METRIC_NAMES = (
    "pearson",
    "spearman",
    "kendall_tau_b",
    "pairwise_ordering_accuracy",
    "mmrv",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strict_rows(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(row, Mapping) for row in value):
        return [], False
    return [dict(row) for row in value], True


def _string(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _digest(value: Any) -> str:
    text = _string(value).lower().removeprefix("sha256:")
    return text if len(text) == 64 and all(char in "0123456789abcdef" for char in text) else ""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    attempt_ids: list[str] = []
    for index, row in enumerate(rows):
        attempt_id = _string(row.get("attempt_id"))
        attempt_ids.append(attempt_id)
        for field in (
            "attempt_id",
            "policy_id",
            "site_id",
            "task_id",
            "task_family_id",
            "initial_condition_id",
        ):
            if not _string(row.get(field)):
                blockers.append(f"uncertainty_row_identity_missing:{index}:{field}")
        predicted = _number(row.get("predicted_score"))
        reference = _number(row.get("reference_score"))
        if predicted is None:
            blockers.append(f"uncertainty_predicted_score_invalid:{index}")
        if reference is None:
            blockers.append(f"uncertainty_reference_score_invalid:{index}")
        if row.get("reference_independently_accepted") is not True:
            blockers.append(f"uncertainty_reference_not_independently_accepted:{index}")
        for field in (
            "policy_checkpoint_sha256",
            "initial_condition_sha256",
            "evaluator_output_sha256",
            "reference_output_sha256",
        ):
            if not _digest(row.get(field)):
                blockers.append(f"uncertainty_row_digest_missing:{index}:{field}")
    if any(not value for value in attempt_ids):
        blockers.append("uncertainty_attempt_ids_missing")
    if len(attempt_ids) != len(set(attempt_ids)):
        blockers.append("uncertainty_attempt_ids_duplicate")
    if len({_string(row.get("policy_id")) for row in rows}) < 3:
        blockers.append("uncertainty_report_requires_at_least_three_policies")
    return sorted(set(blockers))


def _policy_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_string(row.get("_bootstrap_policy_instance")) or _string(row.get("policy_id"))].append(row)
    return [
        {
            "policy_instance_id": policy_instance_id,
            "source_policy_id": _string(group[0].get("policy_id")),
            "predicted_score": sum(float(row["predicted_score"]) for row in group) / len(group),
            "reference_score": sum(float(row["reference_score"]) for row in group) / len(group),
            "rollout_count": len(group),
        }
        for policy_instance_id, group in sorted(grouped.items())
        if group
    ]


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    aggregates = _policy_aggregates(rows)
    return external_rank_metrics(
        [float(row["predicted_score"]) for row in aggregates],
        [float(row["reference_score"]) for row in aggregates],
    )


def _sample_hierarchy(
    rows: Sequence[Mapping[str, Any]], *, rng: random.Random
) -> list[dict[str, Any]]:
    """Sample policies, then nested sites, task families/tasks, and conditions."""

    policy_ids = sorted({_string(row.get("policy_id")) for row in rows})
    sampled_policy_ids = [rng.choice(policy_ids) for _ in policy_ids]
    sampled: list[dict[str, Any]] = []
    for policy_instance, policy_id in enumerate(sampled_policy_ids):
        policy_rows = [row for row in rows if _string(row.get("policy_id")) == policy_id]
        site_ids = sorted({_string(row.get("site_id")) for row in policy_rows})
        for site_instance, site_id in enumerate(rng.choice(site_ids) for _ in site_ids):
            site_rows = [row for row in policy_rows if _string(row.get("site_id")) == site_id]
            family_ids = sorted({_string(row.get("task_family_id")) for row in site_rows})
            for family_instance, family_id in enumerate(
                rng.choice(family_ids) for _ in family_ids
            ):
                family_rows = [
                    row
                    for row in site_rows
                    if _string(row.get("task_family_id")) == family_id
                ]
                task_ids = sorted({_string(row.get("task_id")) for row in family_rows})
                for task_instance, task_id in enumerate(rng.choice(task_ids) for _ in task_ids):
                    task_rows = [
                        row for row in family_rows if _string(row.get("task_id")) == task_id
                    ]
                    condition_ids = sorted(
                        {_string(row.get("initial_condition_id")) for row in task_rows}
                    )
                    for condition_instance, condition_id in enumerate(
                        rng.choice(condition_ids) for _ in condition_ids
                    ):
                        condition_rows = [
                            row
                            for row in task_rows
                            if _string(row.get("initial_condition_id")) == condition_id
                        ]
                        sampled_trials = [
                            rng.choice(condition_rows) for _ in condition_rows
                        ]
                        for trial_instance, source in enumerate(sampled_trials):
                            copied = dict(source)
                            copied["_bootstrap_policy_instance"] = (
                                f"p{policy_instance}:{policy_id}"
                            )
                            copied["_bootstrap_site_instance"] = f"s{site_instance}:{site_id}"
                            copied["_bootstrap_family_instance"] = (
                                f"f{family_instance}:{family_id}"
                            )
                            copied["_bootstrap_task_instance"] = f"t{task_instance}:{task_id}"
                            copied["_bootstrap_condition_instance"] = (
                                f"c{condition_instance}:{condition_id}"
                            )
                            copied["_bootstrap_trial_instance"] = trial_instance
                            sampled.append(copied)
    return sampled


def _bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]], *, seed: int, replicate_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    top_policy_counts: dict[str, int] = defaultdict(int)
    successful = 0
    for _ in range(replicate_count):
        sampled = _sample_hierarchy(rows, rng=rng)
        metrics = _metrics(sampled)
        replicate_succeeded = any(
            value is not None and math.isfinite(value) for value in metrics.values()
        )
        if replicate_succeeded:
            successful += 1
        for metric in METRIC_NAMES:
            value = metrics.get(metric)
            if value is not None and math.isfinite(value):
                samples[metric].append(float(value))
        aggregates = _policy_aggregates(sampled)
        if replicate_succeeded and aggregates:
            winner = max(
                aggregates,
                key=lambda row: (float(row["predicted_score"]), row["source_policy_id"]),
            )["source_policy_id"]
            top_policy_counts[str(winner)] += 1
    intervals = {
        metric: {
            "confidence": 0.95,
            "lower": round(lower, 6)
            if (lower := _percentile(values, 0.025)) is not None
            else None,
            "upper": round(upper, 6)
            if (upper := _percentile(values, 0.975)) is not None
            else None,
            "sample_count": len(values),
        }
        for metric, values in samples.items()
    }
    bootstrap = {
        "method": BOOTSTRAP_METHOD,
        "seed": seed,
        "requested_replicate_count": replicate_count,
        "successful_replicate_count": successful,
        "resampled_levels": [
            "policy",
            "site",
            "task_family",
            "task",
            "initial_condition",
            "trial",
        ],
        "top_policy_selection_frequency": {
            policy_id: round(count / max(1, successful), 6)
            for policy_id, count in sorted(top_policy_counts.items())
        },
    }
    return intervals, bootstrap


def _deterministic_prefix_order(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: hashlib.sha256(
            f"{seed}:{_string(row.get('attempt_id'))}".encode("utf-8")
        ).hexdigest(),
    )


def _convergence_counts(total: int, requested: Any) -> list[int]:
    if isinstance(requested, list):
        counts = sorted(
            {
                value
                for value in requested
                if isinstance(value, int) and not isinstance(value, bool) and 3 <= value <= total
            }
        )
        if counts:
            return counts if counts[-1] == total else [*counts, total]
    candidates = {3, max(3, total // 10), max(3, total // 4), max(3, total // 2), total}
    return sorted(value for value in candidates if value <= total)


def _convergence_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    trial_counts: Any,
    subsample_replicates: int,
) -> list[dict[str, Any]]:
    ordered = _deterministic_prefix_order(rows, seed=seed)
    result: list[dict[str, Any]] = []
    for count in _convergence_counts(len(ordered), trial_counts):
        metric_samples: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
        coverage_samples: list[dict[str, int]] = []
        for replicate in range(subsample_replicates):
            replicate_order = _deterministic_prefix_order(rows, seed=seed + replicate * 104729)
            sample = replicate_order[:count]
            metrics = _metrics(sample)
            for metric, value in metrics.items():
                if metric in metric_samples and value is not None and math.isfinite(value):
                    metric_samples[metric].append(float(value))
            coverage_samples.append(
                {
                    "policies": len({_string(row.get("policy_id")) for row in sample}),
                    "sites": len({_string(row.get("site_id")) for row in sample}),
                    "task_families": len(
                        {_string(row.get("task_family_id")) for row in sample}
                    ),
                    "tasks": len({_string(row.get("task_id")) for row in sample}),
                    "initial_conditions": len(
                        {_string(row.get("initial_condition_id")) for row in sample}
                    ),
                }
            )
        result.append(
            {
                "trial_count": count,
                "subsample_replicates": subsample_replicates,
                "coverage": {
                    key: {
                        "minimum": min(item[key] for item in coverage_samples),
                        "maximum": max(item[key] for item in coverage_samples),
                    }
                    for key in coverage_samples[0]
                },
                "metrics": {
                    metric: {
                        "median": round(median, 6)
                        if (median := _percentile(values, 0.5)) is not None
                        else None,
                        "confidence_interval_95": [
                            round(lower, 6)
                            if (lower := _percentile(values, 0.025)) is not None
                            else None,
                            round(upper, 6)
                            if (upper := _percentile(values, 0.975)) is not None
                            else None,
                        ],
                        "sample_count": len(values),
                    }
                    for metric, values in metric_samples.items()
                },
            }
        )
    return result


def _leave_one_out(
    rows: Sequence[Mapping[str, Any]], *, axis: str
) -> list[dict[str, Any]]:
    values = sorted({_string(row.get(axis)) for row in rows})
    return [
        {
            "omitted_value": value,
            "remaining_rollout_count": len(remaining),
            "remaining_policy_count": len(
                {_string(row.get("policy_id")) for row in remaining}
            ),
            "metrics": {
                metric: round(metric_value, 6) if metric_value is not None else None
                for metric, metric_value in _metrics(remaining).items()
            },
        }
        for value in values
        if (remaining := [row for row in rows if _string(row.get(axis)) != value])
    ]


def build_benchmark_uncertainty_report(request: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("uncertainty_request_schema_missing_or_unsupported")
    if request.get("frozen") is not True:
        blockers.append("uncertainty_request_must_be_frozen")
    for field in ("study_id", "study_version"):
        if not _string(request.get(field)):
            blockers.append(f"uncertainty_study_identity_missing:{field}")
    for field in ("benchmark_spec_sha256", "attempt_ledger_sha256", "reference_manifest_sha256"):
        if not _digest(request.get(field)):
            blockers.append(f"uncertainty_request_digest_missing:{field}")
    rows, payload_valid = _strict_rows(request.get("rows"))
    if not payload_valid or not rows:
        blockers.append("uncertainty_rows_missing_or_invalid")
    row_blockers = _validate_rows(rows)
    blockers.extend(row_blockers)
    rows_usable = bool(payload_valid and rows and not row_blockers)
    rows = sorted(rows, key=lambda row: _string(row.get("attempt_id")))
    bootstrap = _mapping(request.get("bootstrap"))
    seed = _integer(bootstrap.get("seed"), minimum=0)
    replicate_count = _integer(bootstrap.get("replicate_count"), minimum=1)
    if seed is None:
        blockers.append("uncertainty_bootstrap_seed_missing_or_invalid")
        seed = 1729
    if replicate_count is None:
        blockers.append("uncertainty_bootstrap_replicate_count_missing_or_invalid")
        replicate_count = DEFAULT_BOOTSTRAP_REPLICATES
    claim_eligible_replicates = replicate_count >= DEFAULT_BOOTSTRAP_REPLICATES
    subsample_replicates = _integer(request.get("convergence_subsample_replicates"), minimum=1)
    if subsample_replicates is None:
        blockers.append("uncertainty_convergence_subsample_replicates_missing_or_invalid")
        subsample_replicates = 200
    point_metrics = (
        _metrics(rows)
        if rows_usable and len(rows) >= 3
        else {metric: None for metric in METRIC_NAMES}
    )
    intervals: dict[str, Any] = {}
    bootstrap_detail: dict[str, Any] = {
        "method": BOOTSTRAP_METHOD,
        "seed": seed,
        "requested_replicate_count": replicate_count,
        "successful_replicate_count": 0,
    }
    if not blockers:
        intervals, bootstrap_detail = _bootstrap_intervals(
            rows, seed=seed, replicate_count=replicate_count
        )
    convergence = _convergence_report(
        rows,
        seed=seed,
        trial_counts=request.get("convergence_trial_counts"),
        subsample_replicates=subsample_replicates,
    ) if rows_usable else []
    blockers = sorted(set(blockers))
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study_id": _string(request.get("study_id")) or None,
        "study_version": _string(request.get("study_version")) or None,
        "status": "measured" if not blockers else "blocked",
        "frozen": request.get("frozen") is True,
        "benchmark_spec_sha256": _digest(request.get("benchmark_spec_sha256")) or None,
        "attempt_ledger_sha256": _digest(request.get("attempt_ledger_sha256")) or None,
        "reference_manifest_sha256": _digest(request.get("reference_manifest_sha256")) or None,
        "coverage": {
            "rollouts": len(rows),
            "policies": len({_string(row.get("policy_id")) for row in rows}),
            "sites": len({_string(row.get("site_id")) for row in rows}),
            "task_families": len({_string(row.get("task_family_id")) for row in rows}),
            "tasks": len({_string(row.get("task_id")) for row in rows}),
            "initial_conditions": len(
                {_string(row.get("initial_condition_id")) for row in rows}
            ),
        },
        "point_metrics": {
            metric: round(value, 6) if value is not None else None
            for metric, value in point_metrics.items()
        },
        "confidence_intervals": intervals,
        "bootstrap": bootstrap_detail,
        "convergence": convergence,
        "policy_rank_stability_vs_coverage": [
            {
                "trial_count": row["trial_count"],
                "spearman": row["metrics"]["spearman"],
                "pairwise_ordering_accuracy": row["metrics"][
                    "pairwise_ordering_accuracy"
                ],
                "coverage": row["coverage"],
            }
            for row in convergence
        ],
        "leave_one_policy_out": _leave_one_out(rows, axis="policy_id") if rows_usable else [],
        "leave_one_task_family_out": _leave_one_out(rows, axis="task_family_id")
        if rows_usable
        else [],
        "claim_eligibility": {
            "minimum_bootstrap_replicates": DEFAULT_BOOTSTRAP_REPLICATES,
            "bootstrap_replicate_count_sufficient": claim_eligible_replicates,
            "report_status_measured": not blockers,
            "public_rank_fidelity_claim_eligible": False,
            "reason": "separate_preregistered_external_anchor_claim_gate_required",
        },
        "blockers": blockers,
        "claim_boundary": {
            "uncertainty_report_is_not_physical_robot_success": True,
            "external_reference_acceptance_required": True,
            "leave_one_out_is_sensitivity_not_generalization_proof": True,
            "public_claim_upgrade_allowed": False,
        },
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def _load_mapping(path: str | Path) -> dict[str, Any]:
    value = read_json_any(Path(path))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_benchmark_uncertainty_report(_load_mapping(args.input))
    write_json(Path(args.output), report)
    return 0 if report["status"] == "measured" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
