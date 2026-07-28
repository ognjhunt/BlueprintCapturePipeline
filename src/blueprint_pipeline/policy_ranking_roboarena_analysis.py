"""Freeze label-blind Phase-A predictions, then unseal RoboArena outcomes.

The two entry points in this module deliberately form a one-way seam.  A
complete provider run is first reduced to an immutable, label-free prediction
artifact.  Benchmark metadata is read only by :func:`unseal_and_analyze`, which
requires the caller to present that artifact's digest.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .common import write_json
from .policy_ranking_roboarena_calibration import canonical_sha256


ANALYSIS_SCHEMA = "policy_ranking_roboarena_phase_a_analysis.v2"
FREEZE_SCHEMA = "policy_ranking_roboarena_frozen_predictions.v2"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260728
RISK_COVERAGE_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
RISK_INCREASE_TOLERANCE = 0.02
SUPERSEDED_ANALYSIS_CONTRACT_V2_SHA256 = (
    "b470d3fa01166a111b5b7a5f9618ca68bcd56fd6140021c6c0f92a2a8fc207ac"
)
SUPERSEDED_ANALYSIS_CONTRACT_V3_SHA256 = (
    "4a81ee0abde5f2a5a8fb064484338c42af369bab6b3a90906d30b0065228db2c"
)

# These episode-selectivity thresholds retain the corresponding strict values
# from the previous frozen Experiment-2 preregistration.  They were mapped to
# the v2 evaluator fields before the first provider response or outcome unseal.
SELECTIVITY = {
    "evaluator_uncertainty_max": 0.20,
    "action_following_confidence_min": 0.70,
    "temporal_consistency_min": 0.70,
    "pair_score_margin_min": 0.20,
}

GATES = {
    "policy_count_minimum": 7,
    "spearman_rho_minimum": 0.70,
    "kendall_tau_b_minimum": 0.50,
    "pairwise_accuracy_minimum": 0.70,
    "pairwise_accuracy_clustered_ci95_lower_minimum": 0.50,
    "true_top_policy_in_predicted_top_two": True,
    "selective_coverage_minimum": 0.50,
    "selective_pairwise_accuracy_minimum": 0.75,
    "all_gates_required": True,
}


def analysis_contract_v3(*, protocol_sha256: str, evaluator_digest: str) -> dict[str, Any]:
    """Return the immutable pre-canary analysis contract."""

    contract: dict[str, Any] = {
        "schema_version": "policy_ranking_roboarena_phase_a_analysis_lock.v3",
        "protocol_sha256": protocol_sha256,
        "evaluator_digest": evaluator_digest,
        "independent_unit": "roboarena_session",
        "primary_prediction_estimand": (
            "mean evaluator progress_score_0_to_5 divided by 5, with zero assigned "
            "to any Blueprint safety-abstained episode"
        ),
        "diagnostic_prediction_estimands": [
            "raw_mean_progress_score_0_to_5_divided_by_5",
            "mean_success_probability",
        ],
        "physical_ordering": (
            "mean binary success rate, with mean partial success as a deterministic tie-break"
        ),
        "primary_pairwise_label_basis": "per_session_binary_then_partial",
        "selectivity": dict(SELECTIVITY),
        "risk_coverage": {
            "session_uncertainty": (
                "maximum evaluator uncertainty across seven policy rows; forced to 1.0 "
                "when any row has Blueprint safety abstention"
            ),
            "full_empirical_curve": "all ordered session prefixes",
            "registered_gate_grid": list(RISK_COVERAGE_GRID),
            "isotonic_direction": "risk_non_decreasing_as_coverage_increases",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "adjacent_more_abstention_risk_increase_tolerance": RISK_INCREASE_TOLERANCE,
            "failure_rule": (
                "fail only when risk at the lower-coverage adjacent grid point minus risk "
                "at the higher-coverage point exceeds tolerance and its session-clustered "
                "bootstrap 95% lower confidence bound exceeds zero"
            ),
        },
        "uncertainty": {
            "policy_level_exact_permutations": "all_7_factorial_label_assignments",
            "session_clustered_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "session_clustered_bootstrap_seed": BOOTSTRAP_SEED,
        },
        "gates": dict(GATES),
        "supersedes_analysis_contract_sha256": SUPERSEDED_ANALYSIS_CONTRACT_V2_SHA256,
        "supersession_scope": "transport binding only; analysis rules unchanged",
        "provider_called": False,
        "outcome_labels_accessed": False,
    }
    contract["analysis_contract_sha256"] = canonical_sha256(contract)
    return contract


def analysis_contract(*, protocol_sha256: str, evaluator_digest: str) -> dict[str, Any]:
    """Return the v4 prospective contract after the zero-row schema fix."""

    contract = analysis_contract_v3(
        protocol_sha256=protocol_sha256, evaluator_digest=evaluator_digest
    )
    contract.pop("analysis_contract_sha256")
    contract.update(
        {
            "schema_version": "policy_ranking_roboarena_phase_a_analysis_lock.v4",
            "supersedes_analysis_contract_sha256": SUPERSEDED_ANALYSIS_CONTRACT_V3_SHA256,
            "supersession_scope": (
                "provider JSON-schema compatibility only; analysis rules unchanged"
            ),
        }
    )
    contract["analysis_contract_sha256"] = canonical_sha256(contract)
    return contract


def _without_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def freeze_predictions(
    evaluator_inventory: Mapping[str, Any], evaluator_run: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and freeze a complete label-blind provider matrix."""

    blockers: list[str] = []
    requests = list(evaluator_inventory.get("requests") or [])
    results = list(evaluator_run.get("results") or [])
    if evaluator_inventory.get("status") != "ready":
        blockers.append("evaluator_inventory_not_ready")
    if evaluator_run.get("status") != "completed":
        blockers.append("evaluator_run_not_completed")
    if evaluator_run.get("inventory_sha256") != evaluator_inventory.get("inventory_sha256"):
        blockers.append("inventory_digest_mismatch")
    if evaluator_run.get("outcome_labels_accessed") is not False:
        blockers.append("outcome_labels_must_remain_sealed")
    if evaluator_run.get("provider_called") is not True:
        blockers.append("provider_call_not_proven")
    if evaluator_run.get("failures"):
        blockers.append("provider_failures_present")

    request_by_id = {str(row.get("request_id")): row for row in requests}
    result_by_id = {str(row.get("request_id")): row for row in results}
    if len(request_by_id) != len(requests):
        blockers.append("duplicate_inventory_request_id")
    if len(result_by_id) != len(results):
        blockers.append("duplicate_result_request_id")
    if set(result_by_id) != set(request_by_id):
        blockers.append("result_request_set_not_exact")

    rows: list[dict[str, Any]] = []
    for request_id in sorted(set(request_by_id) & set(result_by_id)):
        request = request_by_id[request_id]
        result = result_by_id[request_id]
        if canonical_sha256(_without_digest(result, "result_sha256")) != result.get(
            "result_sha256"
        ):
            blockers.append(f"result_digest_invalid:{request_id}")
            continue
        for field in (
            "source_request_id",
            "session_id",
            "policy_id_internal_only",
            "evaluator_digest",
        ):
            if result.get(field) != request.get(field):
                blockers.append(f"result_identity_mismatch:{request_id}:{field}")
        if result.get("model") != evaluator_inventory.get("evaluator", {}).get("model"):
            blockers.append(f"model_snapshot_mismatch:{request_id}")
        if (
            result.get("policy_identity_sent_to_provider") is not False
            or result.get("benchmark_outcomes_sent_to_provider") is not False
            or result.get("physical_ground_truth_pixels_sent_to_provider") is not False
        ):
            blockers.append(f"provider_redaction_invariant_failed:{request_id}")
        payload = result.get("structured_response") or {}
        try:
            raw_progress = float(payload["progress_score_0_to_5"]) / 5.0
            success_probability = float(payload["success_probability"])
            uncertainty = float(payload["uncertainty"])
            temporal = float(payload["temporal_consistency"])
            action_following = float(payload["action_following_confidence"])
        except (KeyError, TypeError, ValueError):
            blockers.append(f"structured_response_numeric_fields_invalid:{request_id}")
            continue
        safety_abstain = bool(result.get("blueprint_safety_abstain"))
        rows.append(
            {
                "request_id": request_id,
                "source_request_id": request["source_request_id"],
                "session_id": request["session_id"],
                "policy_id": request["policy_id_internal_only"],
                "result_sha256": result["result_sha256"],
                "provider_response_id": result.get("response_id"),
                "model": result.get("model"),
                "raw_progress_score": raw_progress,
                "primary_safety_adjusted_progress_score": (0.0 if safety_abstain else raw_progress),
                "success_probability": success_probability,
                "stable_success_confirmed": bool(payload.get("stable_success_confirmed")),
                "temporal_consistency": temporal,
                "action_following_confidence": action_following,
                "evaluator_uncertainty": uncertainty,
                "effective_uncertainty": 1.0 if safety_abstain else uncertainty,
                "evaluator_abstain": bool(result.get("evaluator_abstain")),
                "blueprint_safety_abstain": safety_abstain,
                "abstention_sources": list(result.get("abstention_sources") or []),
                "deterministic_collapse_flags": list(
                    result.get("deterministic_collapse_flags") or []
                ),
                "artifact_flags": list(payload.get("artifact_flags") or []),
                "latency_seconds": float(result.get("latency_seconds") or 0.0),
                "usage": dict(result.get("usage") or {}),
            }
        )

    contract = analysis_contract(
        protocol_sha256=str(evaluator_inventory.get("protocol_sha256") or ""),
        evaluator_digest=str(
            evaluator_inventory.get("evaluator", {}).get("evaluator_digest") or ""
        ),
    )
    frozen: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "status": "frozen" if not blockers else "blocked",
        "protocol_sha256": evaluator_inventory.get("protocol_sha256"),
        "inventory_sha256": evaluator_inventory.get("inventory_sha256"),
        "evaluator_run_sha256": evaluator_run.get("run_sha256"),
        "analysis_contract": contract,
        "analysis_contract_sha256": contract["analysis_contract_sha256"],
        "prediction_row_count": len(rows),
        "session_count": len({row["session_id"] for row in rows}),
        "policy_count": len({row["policy_id"] for row in rows}),
        "rows": rows,
        "provider_called": bool(evaluator_run.get("provider_called")),
        "data_uploaded": bool(evaluator_run.get("data_uploaded")),
        "estimated_evaluator_cost_usd": float(evaluator_run.get("estimated_cost_usd") or 0.0),
        "outcome_labels_accessed": False,
        "blockers": sorted(set(blockers)),
    }
    frozen["frozen_predictions_sha256"] = canonical_sha256(frozen)
    return frozen


def _rank(values: Mapping[str, float], *, descending: bool = True) -> dict[str, float]:
    ordered = sorted(values, key=lambda item: values[item], reverse=descending)
    ranks: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for item in ordered[cursor:end]:
            ranks[item] = average
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _spearman(predicted: Mapping[str, float], actual: Mapping[str, float]) -> float | None:
    policies = sorted(set(predicted) & set(actual))
    predicted_ranks = _rank(predicted)
    actual_ranks = _rank(actual)
    return _pearson(
        [predicted_ranks[policy] for policy in policies],
        [actual_ranks[policy] for policy in policies],
    )


def _kendall_tau_b(predicted: Mapping[str, float], actual: Mapping[str, float]) -> float | None:
    policies = sorted(set(predicted) & set(actual))
    concordant = discordant = predicted_ties = actual_ties = 0
    for index, left in enumerate(policies):
        for right in policies[index + 1 :]:
            predicted_delta = predicted[left] - predicted[right]
            actual_delta = actual[left] - actual[right]
            if predicted_delta == 0 and actual_delta == 0:
                continue
            if predicted_delta == 0:
                predicted_ties += 1
            elif actual_delta == 0:
                actual_ties += 1
            elif predicted_delta * actual_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + predicted_ties) * (concordant + discordant + actual_ties)
    )
    return (concordant - discordant) / denominator if denominator else None


def _pairwise_accuracy(predicted: Mapping[str, float], actual: Mapping[str, float]) -> float | None:
    policies = sorted(set(predicted) & set(actual))
    scores: list[float] = []
    for index, left in enumerate(policies):
        for right in policies[index + 1 :]:
            actual_delta = actual[left] - actual[right]
            if actual_delta == 0:
                continue
            predicted_delta = predicted[left] - predicted[right]
            scores.append(
                0.5 if predicted_delta == 0 else float(predicted_delta * actual_delta > 0)
            )
    return sum(scores) / len(scores) if scores else None


def _mmrv(predicted: Mapping[str, float], actual_binary: Mapping[str, float]) -> float | None:
    policies = sorted(set(predicted) & set(actual_binary))
    if not policies:
        return None
    maxima: list[float] = []
    for policy in policies:
        maximum = 0.0
        for other in policies:
            if policy == other:
                continue
            disagreement = (predicted[policy] > predicted[other]) != (
                actual_binary[policy] > actual_binary[other]
            )
            if disagreement:
                maximum = max(maximum, abs(actual_binary[policy] - actual_binary[other]))
        maxima.append(maximum)
    return sum(maxima) / len(maxima)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _load_labels(
    roboarena_root: str | Path, session_ids: Sequence[str], policies: Sequence[str]
) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
    root = Path(roboarena_root).resolve()
    labels: dict[str, dict[str, dict[str, float]]] = {}
    blockers: list[str] = []
    registered = set(policies)
    for session_id in session_ids:
        path = root / "evaluation_sessions" / session_id / "metadata.yaml"
        if not path.is_file():
            blockers.append(f"label_metadata_missing:{session_id}")
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        outcomes: dict[str, dict[str, float]] = {}
        raw_policies = payload.get("policies") or []
        policy_rows = raw_policies.values() if isinstance(raw_policies, Mapping) else raw_policies
        for raw in policy_rows:
            if not isinstance(raw, Mapping):
                continue
            policy = str(raw.get("policy_name") or "")
            if policy in registered:
                outcomes[policy] = {
                    "binary_success": float(bool(raw.get("binary_success"))),
                    "partial_success": float(raw.get("partial_success") or 0.0),
                }
        if set(outcomes) != registered:
            blockers.append(f"label_policy_set_mismatch:{session_id}")
        labels[session_id] = outcomes
    return labels, blockers


def _policy_vectors(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Mapping[str, float]]],
    session_sample: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    predicted: dict[str, list[float]] = defaultdict(list)
    raw: dict[str, list[float]] = defaultdict(list)
    binary: dict[str, list[float]] = defaultdict(list)
    partial: dict[str, list[float]] = defaultdict(list)
    # Preserve bootstrap multiplicity by looping over the sampled sessions.
    lookup = {(str(row["session_id"]), str(row["policy_id"])): row for row in rows}
    policies = sorted({str(row["policy_id"]) for row in rows})
    for session_id in session_sample:
        for policy in policies:
            row = lookup.get((session_id, policy))
            label = labels.get(session_id, {}).get(policy)
            if row is None or label is None:
                continue
            predicted[policy].append(float(row["primary_safety_adjusted_progress_score"]))
            raw[policy].append(float(row["raw_progress_score"]))
            binary[policy].append(float(label["binary_success"]))
            partial[policy].append(float(label["partial_success"]))

    def means(value: Mapping[str, Sequence[float]]) -> dict[str, float]:
        return {key: sum(items) / len(items) for key, items in value.items() if items}

    return means(predicted), means(raw), means(binary), means(partial)


def _session_pairwise(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Mapping[str, float]]],
    session_sample: Sequence[str],
    *,
    selective: bool,
) -> tuple[float | None, int, int]:
    lookup = {(str(row["session_id"]), str(row["policy_id"])): row for row in rows}
    policies = sorted({str(row["policy_id"]) for row in rows})
    correct = 0.0
    evaluated = informative = 0
    for session_id in session_sample:
        for index, left in enumerate(policies):
            for right in policies[index + 1 :]:
                left_label = labels.get(session_id, {}).get(left)
                right_label = labels.get(session_id, {}).get(right)
                if left_label is None or right_label is None:
                    continue
                label_delta = left_label["binary_success"] - right_label["binary_success"]
                if label_delta == 0:
                    label_delta = left_label["partial_success"] - right_label["partial_success"]
                if label_delta == 0:
                    continue
                informative += 1
                left_row = lookup.get((session_id, left))
                right_row = lookup.get((session_id, right))
                if left_row is None or right_row is None:
                    continue
                delta = float(left_row["primary_safety_adjusted_progress_score"]) - float(
                    right_row["primary_safety_adjusted_progress_score"]
                )
                if selective:
                    episode_rows = (left_row, right_row)
                    if any(bool(row["blueprint_safety_abstain"]) for row in episode_rows):
                        continue
                    if any(
                        float(row["evaluator_uncertainty"])
                        > SELECTIVITY["evaluator_uncertainty_max"]
                        or float(row["action_following_confidence"])
                        < SELECTIVITY["action_following_confidence_min"]
                        or float(row["temporal_consistency"])
                        < SELECTIVITY["temporal_consistency_min"]
                        for row in episode_rows
                    ):
                        continue
                    if abs(delta) + 1e-12 < SELECTIVITY["pair_score_margin_min"]:
                        continue
                evaluated += 1
                correct += 0.5 if delta == 0 else float(delta * label_delta > 0)
    return (correct / evaluated if evaluated else None, evaluated, informative)


def _isotonic_non_decreasing(values: Sequence[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / weight, weight])
    fitted: list[float] = []
    for value, weight in blocks:
        fitted.extend([value] * int(weight))
    return fitted


def _risk_for_sessions(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Mapping[str, float]]],
    sessions: Sequence[str],
) -> float | None:
    accuracy, _, _ = _session_pairwise(rows, labels, sessions, selective=False)
    return None if accuracy is None else 1.0 - accuracy


def _ece(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    bins: int = 10,
) -> tuple[float | None, list[dict[str, Any]], float | None]:
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    brier: list[float] = []
    for row in rows:
        label = labels.get(str(row["session_id"]), {}).get(str(row["policy_id"]))
        if label is None:
            continue
        probability = float(row["success_probability"])
        actual = float(label["binary_success"])
        index = min(bins - 1, int(probability * bins))
        buckets[index].append((probability, actual))
        brier.append((probability - actual) ** 2)
    total = sum(len(bucket) for bucket in buckets)
    details: list[dict[str, Any]] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        confidence = sum(row[0] for row in bucket) / len(bucket) if bucket else None
        accuracy = sum(row[1] for row in bucket) / len(bucket) if bucket else None
        if bucket:
            ece += len(bucket) / total * abs(float(confidence) - float(accuracy))
        details.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(bucket),
                "mean_probability": confidence,
                "binary_success_rate": accuracy,
            }
        )
    return (ece if total else None), details, (sum(brier) / len(brier) if brier else None)


def unseal_and_analyze(
    frozen: Mapping[str, Any],
    *,
    expected_frozen_predictions_sha256: str,
    roboarena_root: str | Path,
    dataset_revision: str,
    unsealed_at: str | None = None,
) -> dict[str, Any]:
    """Load labels only after validating the complete frozen prediction digest."""

    blockers: list[str] = []
    observed_digest = canonical_sha256(_without_digest(frozen, "frozen_predictions_sha256"))
    if frozen.get("status") != "frozen":
        blockers.append("predictions_not_frozen")
    if observed_digest != frozen.get("frozen_predictions_sha256"):
        blockers.append("frozen_predictions_digest_invalid")
    if expected_frozen_predictions_sha256 != frozen.get("frozen_predictions_sha256"):
        blockers.append("expected_frozen_predictions_digest_mismatch")
    if blockers:
        return {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "blocked_before_label_access",
            "outcome_labels_accessed": False,
            "blockers": blockers,
        }

    rows = list(frozen.get("rows") or [])
    sessions = sorted({str(row["session_id"]) for row in rows})
    policies = sorted({str(row["policy_id"]) for row in rows})
    labels, label_blockers = _load_labels(roboarena_root, sessions, policies)
    blockers.extend(label_blockers)
    expected_rows = len(sessions) * len(policies)
    if len(rows) != expected_rows:
        blockers.append(f"prediction_matrix_incomplete:{len(rows)}_of_{expected_rows}")

    predicted, raw_predicted, binary, partial = _policy_vectors(rows, labels, sessions)
    physical_order = {
        policy: binary[policy] * (len(sessions) + 1) + partial.get(policy, 0.0) for policy in binary
    }
    spearman = _spearman(predicted, physical_order)
    kendall = _kendall_tau_b(predicted, physical_order)
    pairwise = _pairwise_accuracy(predicted, physical_order)
    mmrv = _mmrv(predicted, binary)
    session_pairwise, session_pair_count, informative_count = _session_pairwise(
        rows, labels, sessions, selective=False
    )
    selective_accuracy, selective_count, _ = _session_pairwise(
        rows, labels, sessions, selective=True
    )
    selective_coverage = selective_count / informative_count if informative_count else 0.0

    bootstrap_pairwise: list[float] = []
    rng = random.Random(BOOTSTRAP_SEED)
    session_uncertainty: dict[str, float] = {}
    by_session_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session_rows[str(row["session_id"])].append(row)
    for session_id, session_rows in by_session_rows.items():
        session_uncertainty[session_id] = max(
            float(row["effective_uncertainty"]) for row in session_rows
        )
    ordered_sessions = sorted(sessions, key=lambda item: (session_uncertainty[item], item))

    full_curve: list[dict[str, Any]] = []
    for count in range(1, len(ordered_sessions) + 1):
        retained = ordered_sessions[:count]
        risk = _risk_for_sessions(rows, labels, retained)
        full_curve.append(
            {
                "retained_session_count": count,
                "coverage": count / len(ordered_sessions),
                "uncertainty_threshold": session_uncertainty[retained[-1]],
                "risk": risk,
            }
        )
    finite_risks = [float(row["risk"]) for row in full_curve if row["risk"] is not None]
    fitted = _isotonic_non_decreasing(finite_risks)
    fit_cursor = 0
    for row in full_curve:
        if row["risk"] is not None:
            row["isotonic_smoothed_risk"] = fitted[fit_cursor]
            fit_cursor += 1
        else:
            row["isotonic_smoothed_risk"] = None

    grid_counts = {
        coverage: max(1, min(len(sessions), math.ceil(coverage * len(sessions))))
        for coverage in RISK_COVERAGE_GRID
    }
    grid: list[dict[str, Any]] = []
    for coverage in RISK_COVERAGE_GRID:
        count = grid_counts[coverage]
        grid.append(
            {
                "registered_coverage": coverage,
                "retained_session_count": count,
                "empirical_coverage": count / len(sessions),
                "risk": _risk_for_sessions(rows, labels, ordered_sessions[:count]),
            }
        )

    adjacent_bootstrap: list[list[float]] = [[] for _ in range(len(grid) - 1)]
    full_curve_bootstrap: list[list[float]] = [[] for _ in full_curve]
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [rng.choice(sessions) for _ in sessions]
        sample_predicted, _, sample_binary, sample_partial = _policy_vectors(rows, labels, sample)
        sample_order = {
            policy: sample_binary[policy] * (len(sample) + 1) + sample_partial.get(policy, 0.0)
            for policy in sample_binary
        }
        value = _pairwise_accuracy(sample_predicted, sample_order)
        if value is not None:
            bootstrap_pairwise.append(value)

        # Resample sessions as clusters, then order sampled clusters by the same
        # frozen uncertainty statistic. Duplicate cluster ids preserve bootstrap
        # multiplicity in the risk calculation.
        ordered_sample = sorted(sample, key=lambda item: (session_uncertainty[item], item))
        for count in range(1, len(ordered_sample) + 1):
            sampled_risk = _risk_for_sessions(rows, labels, ordered_sample[:count])
            if sampled_risk is not None:
                full_curve_bootstrap[count - 1].append(sampled_risk)
        sample_risks: list[float | None] = []
        for coverage in RISK_COVERAGE_GRID:
            count = max(1, min(len(sample), math.ceil(coverage * len(sample))))
            sample_risks.append(_risk_for_sessions(rows, labels, ordered_sample[:count]))
        for index in range(len(sample_risks) - 1):
            left, right = sample_risks[index], sample_risks[index + 1]
            if left is not None and right is not None:
                adjacent_bootstrap[index].append(left - right)

    for index, row in enumerate(full_curve):
        row["session_clustered_bootstrap_ci95"] = [
            _percentile(full_curve_bootstrap[index], 0.025),
            _percentile(full_curve_bootstrap[index], 0.975),
        ]
        row["bootstrap_valid_replicates"] = len(full_curve_bootstrap[index])

    adjacent_checks: list[dict[str, Any]] = []
    material_increase = False
    for index, (lower_coverage, higher_coverage) in enumerate(zip(grid, grid[1:])):
        if lower_coverage["risk"] is None or higher_coverage["risk"] is None:
            delta = None
        else:
            delta = float(lower_coverage["risk"]) - float(higher_coverage["risk"])
        ci = [
            _percentile(adjacent_bootstrap[index], 0.025),
            _percentile(adjacent_bootstrap[index], 0.975),
        ]
        supported = bool(
            delta is not None
            and delta > RISK_INCREASE_TOLERANCE
            and ci[0] is not None
            and ci[0] > 0.0
        )
        material_increase = material_increase or supported
        adjacent_checks.append(
            {
                "lower_coverage": lower_coverage["registered_coverage"],
                "higher_coverage": higher_coverage["registered_coverage"],
                "risk_increase_when_abstaining_more": delta,
                "bootstrap_ci95": ci,
                "material_statistically_supported_increase": supported,
            }
        )

    permutation_rows: list[tuple[float | None, float | None, float | None]] = []
    physical_values = [physical_order[policy] for policy in policies]
    for permutation in itertools.permutations(physical_values):
        permuted = dict(zip(policies, permutation, strict=True))
        permutation_rows.append(
            (
                _spearman(predicted, permuted),
                _kendall_tau_b(predicted, permuted),
                _pairwise_accuracy(predicted, permuted),
            )
        )
    exact_uncertainty = {
        "permutation_count": len(permutation_rows),
        "spearman_two_sided_p": (
            sum(
                abs(float(value[0])) >= abs(float(spearman))
                for value in permutation_rows
                if value[0] is not None
            )
            / sum(value[0] is not None for value in permutation_rows)
            if spearman is not None
            else None
        ),
        "kendall_tau_b_two_sided_p": (
            sum(
                abs(float(value[1])) >= abs(float(kendall))
                for value in permutation_rows
                if value[1] is not None
            )
            / sum(value[1] is not None for value in permutation_rows)
            if kendall is not None
            else None
        ),
        "pairwise_accuracy_one_sided_p": (
            sum(
                float(value[2]) >= float(pairwise)
                for value in permutation_rows
                if value[2] is not None
            )
            / sum(value[2] is not None for value in permutation_rows)
            if pairwise is not None
            else None
        ),
    }

    ece, calibration_bins, brier = _ece(rows, labels)
    predicted_rank = _rank(predicted)
    predicted_order = sorted(policies, key=lambda policy: (predicted_rank[policy], policy))
    maximum_physical_score = max(physical_order.values()) if physical_order else None
    physical_top = sorted(
        policy for policy in policies if physical_order[policy] == maximum_physical_score
    )
    top_two = set(predicted_order[:2])
    gate_results = {
        "policy_count": len(policies) >= GATES["policy_count_minimum"],
        "spearman_rho": spearman is not None and spearman >= GATES["spearman_rho_minimum"],
        "kendall_tau_b": kendall is not None and kendall >= GATES["kendall_tau_b_minimum"],
        "pairwise_accuracy": pairwise is not None
        and pairwise >= GATES["pairwise_accuracy_minimum"],
        "pairwise_accuracy_clustered_ci95_lower": bool(
            bootstrap_pairwise
            and float(_percentile(bootstrap_pairwise, 0.025) or -1.0)
            >= GATES["pairwise_accuracy_clustered_ci95_lower_minimum"]
        ),
        "true_top_policy_in_predicted_top_two": bool(set(physical_top) & top_two),
        "selective_coverage": selective_coverage >= GATES["selective_coverage_minimum"],
        "selective_pairwise_accuracy": bool(
            selective_accuracy is not None
            and selective_accuracy >= GATES["selective_pairwise_accuracy_minimum"]
        ),
        "uncertainty_aware_abstention_risk_rule": not material_increase,
    }
    report: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "completed" if not blockers else "completed_with_blockers",
        "known_answer_reproduction": True,
        "independent_confirmation": False,
        "frozen_predictions_sha256": frozen["frozen_predictions_sha256"],
        "analysis_contract_sha256": frozen["analysis_contract_sha256"],
        "label_unseal": {
            "unsealed_at": unsealed_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset_revision": dataset_revision,
            "join_key": ["session_id", "policy_id"],
            "prediction_digest_verified_before_access": True,
        },
        "outcome_labels_accessed": True,
        "session_count": len(sessions),
        "policy_count": len(policies),
        "prediction_row_count": len(rows),
        "policy_vectors": {
            "primary_safety_adjusted_progress": predicted,
            "raw_progress": raw_predicted,
            "physical_binary_success": binary,
            "physical_partial_success": partial,
            "physical_lexicographic_ordering_score": physical_order,
        },
        "rank_metrics": {
            "spearman_rho": spearman,
            "kendall_tau_b": kendall,
            "policy_pairwise_ordering_accuracy": pairwise,
            "session_pairwise_ordering_accuracy": session_pairwise,
            "session_pairwise_evaluated_count": session_pair_count,
            "session_pairwise_informative_count": informative_count,
            "session_clustered_pairwise_accuracy_ci95": [
                _percentile(bootstrap_pairwise, 0.025),
                _percentile(bootstrap_pairwise, 0.975),
            ],
            "mmrv_simpler_pairwise_real_binary_margin": mmrv,
            "predicted_order": predicted_order,
            "predicted_top_policy": predicted_order[0] if predicted_order else None,
            "physical_top_policies": physical_top,
            "physical_top_predicted_ranks": {
                policy: predicted_rank[policy] for policy in physical_top
            },
            "exact_permutation_uncertainty": exact_uncertainty,
        },
        "calibration": {
            "expected_calibration_error_10_equal_width_bins": ece,
            "brier_score": brier,
            "bins": calibration_bins,
        },
        "abstention": {
            "episode_abstention_count": sum(bool(row["blueprint_safety_abstain"]) for row in rows),
            "episode_abstention_rate": sum(bool(row["blueprint_safety_abstain"]) for row in rows)
            / len(rows),
            "episode_nonabstained_coverage": 1.0
            - sum(bool(row["blueprint_safety_abstain"]) for row in rows) / len(rows),
            "selective_pairwise_accuracy": selective_accuracy,
            "selective_pairwise_count": selective_count,
            "selective_pairwise_coverage": selective_coverage,
            "selectivity": dict(SELECTIVITY),
            "full_empirical_risk_coverage_curve": full_curve,
            "registered_grid": grid,
            "adjacent_risk_checks": adjacent_checks,
            "material_statistically_supported_risk_increase_when_abstaining_more": material_increase,
            "risk_rule_passed": not material_increase,
        },
        "gates": gate_results,
        "all_registered_gates_passed": not blockers and all(gate_results.values()),
        "blockers": sorted(set(blockers)),
        "claim_ceiling": (
            "non_independent_known_answer_reproduction_only; does not establish disjoint "
            "generalization, live policy-WAM behavior, Cosmos qualification, captured-site "
            "transfer, or physical deployment"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--inventory", required=True, type=Path)
    freeze_parser.add_argument("--run", required=True, type=Path)
    freeze_parser.add_argument("--output", required=True, type=Path)
    unseal_parser = commands.add_parser("unseal")
    unseal_parser.add_argument("--frozen", required=True, type=Path)
    unseal_parser.add_argument("--expected-sha256", required=True)
    unseal_parser.add_argument("--roboarena-root", required=True, type=Path)
    unseal_parser.add_argument("--dataset-revision", required=True)
    unseal_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        artifact = freeze_predictions(
            json.loads(args.inventory.read_text(encoding="utf-8")),
            json.loads(args.run.read_text(encoding="utf-8")),
        )
    else:
        artifact = unseal_and_analyze(
            json.loads(args.frozen.read_text(encoding="utf-8")),
            expected_frozen_predictions_sha256=args.expected_sha256,
            roboarena_root=args.roboarena_root,
            dataset_revision=args.dataset_revision,
        )
    write_json(args.output, artifact)
    print(
        json.dumps({key: value for key, value in artifact.items() if key not in {"rows"}}, indent=2)
    )
    return 0 if artifact.get("status") in {"frozen", "completed"} else 2


__all__ = [
    "analysis_contract",
    "analysis_contract_v3",
    "freeze_predictions",
    "unseal_and_analyze",
]


if __name__ == "__main__":
    raise SystemExit(main())
