"""Prediction freeze and deterministic adjudication for policy-ranking Experiment 2.

The freeze command is label-free.  The adjudicate command is intentionally
separate and may run only after the complete provider matrix and its manifest
have been frozen.  It reuses the registered Experiment-1 metric implementation
and adds the Experiment-2 permutation, policy-interval, and risk/coverage
diagnostics without changing the frozen decision thresholds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_evidence import EvidenceStore
from .policy_ranking_thesis import (
    _benchmark_session_labels,
    _kendall_tau_b,
    _percentile,
    canonical_sha256,
    evaluate_frozen_calibration,
)


FREEZE_SCHEMA = "policy_ranking_experiment_2_prediction_freeze.v1"
REPORT_SCHEMA = "policy_ranking_experiment_2_benchmark_report.v1"
RISK_SCHEMA = "policy_ranking_experiment_2_risk_coverage.v1"


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _store(evidence_root: str | Path) -> EvidenceStore:
    identity = _read(Path(evidence_root) / "store_identity.json")
    return EvidenceStore(
        evidence_root,
        experiment_id=str(identity["experiment_id"]),
        inventory_sha256=str(identity["inventory_sha256"]),
        configuration_sha256=str(identity["configuration_sha256"]),
    )


def accepted_judgments(evidence_root: str | Path) -> list[dict[str, Any]]:
    store = _store(evidence_root)
    store.verify_manifest()
    accepted: dict[str, dict[str, Any]] = {}
    for event in store.events():
        if event["event_type"] != "response_accepted":
            continue
        payload = event["payload"]
        request_id = str(payload["request_id"])
        if request_id in accepted:
            raise ValueError(f"duplicate_accepted_response:{request_id}")
        response = payload.get("structured_response")
        if not isinstance(response, Mapping):
            raise ValueError(f"accepted_response_missing_payload:{request_id}")
        accepted[request_id] = dict(response)
    return [accepted[key] for key in sorted(accepted)]


def build_prediction_freeze(
    inventory: Mapping[str, Any], *, evidence_root: str | Path
) -> dict[str, Any]:
    store = _store(evidence_root)
    aggregate = store.rebuild()
    store.verify_manifest()
    manifest = _read(Path(evidence_root) / "evidence_manifest.json")
    requested = {str(row["request_id"]) for row in inventory.get("requests", [])}
    judgments = accepted_judgments(evidence_root)
    accepted = {str(row["request_id"]) for row in judgments}
    blockers: list[str] = []
    if accepted != requested:
        blockers.append(
            f"request_identity_mismatch:missing_{len(requested - accepted)}:extra_{len(accepted - requested)}"
        )
    if aggregate["failed_attempt_count"]:
        blockers.append(f"failed_attempts_present:{aggregate['failed_attempt_count']}")
    response_digest = canonical_sha256(
        [
            {
                "request_id": row["request_id"],
                "response_id": row.get("response_id"),
                "structured_response_sha256": canonical_sha256(row),
            }
            for row in judgments
        ]
    )
    result: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "status": "frozen" if not blockers else "blocked",
        "inventory_sha256": inventory.get("inventory_sha256"),
        "request_count": len(requested),
        "accepted_request_count": len(accepted),
        "failed_attempt_count": aggregate["failed_attempt_count"],
        "estimated_cost_usd_recomputed": aggregate["estimated_cost_usd_recomputed"],
        "actual_cost_usd_recomputed": aggregate["actual_cost_usd_recomputed"],
        "journal_event_count": aggregate["event_count"],
        "last_event_sha256": aggregate["last_event_sha256"],
        "derived_aggregate_sha256": aggregate["aggregate_sha256"],
        "evidence_manifest_sha256": manifest["manifest_sha256"],
        "accepted_response_set_sha256": response_digest,
        "provider_matrix_closed": not blockers,
        "evaluator_changes_after_freeze_allowed": False,
        "heldout_outcome_join_performed": False,
        "potential_metadata_exposure_incident_acknowledged": True,
        "blockers": blockers,
    }
    result["freeze_sha256"] = canonical_sha256(result)
    return result


def _pair_rows(
    judgments: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    roboarena_root: str | Path,
) -> list[dict[str, Any]]:
    method = str(protocol["evaluator"]["full_temporal_method"])
    selected = [row for row in judgments if row.get("method") == method]
    lookup = {(str(row["session_id"]), str(row["policy_id"])): row for row in selected}
    root = Path(roboarena_root).resolve()
    thresholds = protocol["thresholds"]
    rows: list[dict[str, Any]] = []
    for session_id in protocol["partitions"]["heldout"]:
        labels, _ = _benchmark_session_labels(
            root / "evaluation_sessions" / session_id / "metadata.yaml"
        )
        policies = list(protocol["policies"])
        for index, left_policy in enumerate(policies):
            for right_policy in policies[index + 1 :]:
                left_label = labels[left_policy]
                right_label = labels[right_policy]
                actual_delta = left_label["binary_success"] - right_label["binary_success"]
                if actual_delta == 0:
                    actual_delta = left_label["partial_success"] - right_label["partial_success"]
                if actual_delta == 0:
                    continue
                left = lookup[(session_id, left_policy)]
                right = lookup[(session_id, right_policy)]
                predicted_delta = float(left["success_probability"]) - float(
                    right["success_probability"]
                )
                correctness = (
                    0.5 if predicted_delta == 0 else float(predicted_delta * actual_delta > 0)
                )
                raw_confidence = min(
                    float(left["judge_confidence"]),
                    float(right["judge_confidence"]),
                    float(left["action_following_confidence"]),
                    float(right["action_following_confidence"]),
                    float(left["temporal_coherence_confidence"]),
                    float(right["temporal_coherence_confidence"]),
                )
                margin_confidence = min(
                    1.0,
                    abs(predicted_delta) / float(thresholds["pair_score_margin_min"]),
                )
                confidence = min(raw_confidence, margin_confidence)
                selective = bool(
                    raw_confidence >= float(thresholds["selective_judge_confidence_min"])
                    and abs(predicted_delta) >= float(thresholds["pair_score_margin_min"])
                    and not left["abstained"]
                    and not right["abstained"]
                    and not left["critical_contradiction"]
                    and not right["critical_contradiction"]
                )
                rows.append(
                    {
                        "session_id": session_id,
                        "left_policy": left_policy,
                        "right_policy": right_policy,
                        "correctness": correctness,
                        "error": 1.0 - correctness,
                        "confidence": confidence,
                        "selective": selective,
                    }
                )
    return rows


def _risk_coverage(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        pair_rows,
        key=lambda row: (
            -float(row["confidence"]),
            str(row["session_id"]),
            str(row["left_policy"]),
        ),
    )
    cumulative_error = 0.0
    curve: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        cumulative_error += float(row["error"])
        curve.append(
            {
                "accepted_pair_count": index,
                "coverage": index / len(ordered),
                "risk": cumulative_error / index,
                "minimum_confidence": float(row["confidence"]),
            }
        )
    correct_confidence = [float(row["confidence"]) for row in ordered if row["correctness"] == 1.0]
    error_confidence = [float(row["confidence"]) for row in ordered if row["correctness"] == 0.0]
    correct_mean = sum(correct_confidence) / len(correct_confidence) if correct_confidence else None
    error_mean = sum(error_confidence) / len(error_confidence) if error_confidence else None
    association = correct_mean is not None and error_mean is not None and correct_mean > error_mean
    result: dict[str, Any] = {
        "schema_version": RISK_SCHEMA,
        "pair_count": len(ordered),
        "curve": curve,
        "area_under_risk_coverage": (
            sum(point["risk"] for point in curve) / len(curve) if curve else None
        ),
        "confidence_mean_correct": correct_mean,
        "confidence_mean_error": error_mean,
        "uncertainty_error_association_descriptive": association,
        "association_rule_status": "descriptive_because_protocol_required_association_but_did_not_freeze_an_exact_statistic",
        "claim_boundary": "Post-freeze descriptive diagnostic; not used to relax any registered gate.",
    }
    result["report_sha256"] = canonical_sha256(result)
    return result


def _policy_intervals(
    judgments: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    roboarena_root: str | Path,
) -> dict[str, Any]:
    method = str(protocol["evaluator"]["full_temporal_method"])
    selected = [row for row in judgments if row.get("method") == method]
    lookup = {(str(row["session_id"]), str(row["policy_id"])): row for row in selected}
    sessions = list(protocol["partitions"]["heldout"])
    policies = list(protocol["policies"])
    root = Path(roboarena_root).resolve()
    labels = {
        session: _benchmark_session_labels(
            root / "evaluation_sessions" / session / "metadata.yaml"
        )[0]
        for session in sessions
    }
    generator = random.Random(20260727)
    predicted_samples: dict[str, list[float]] = defaultdict(list)
    actual_samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(protocol["thresholds"]["bootstrap_replicates"])):
        sample = [generator.choice(sessions) for _ in sessions]
        for policy in policies:
            predicted_samples[policy].append(
                sum(float(lookup[(session, policy)]["success_probability"]) for session in sample)
                / len(sample)
            )
            actual_samples[policy].append(
                sum(float(labels[session][policy]["binary_success"]) for session in sample)
                / len(sample)
            )
    return {
        policy: {
            "predicted_success_mean": sum(
                float(lookup[(session, policy)]["success_probability"]) for session in sessions
            )
            / len(sessions),
            "predicted_clustered_bootstrap_ci95": [
                _percentile(predicted_samples[policy], 0.025),
                _percentile(predicted_samples[policy], 0.975),
            ],
            "benchmark_binary_success_mean": sum(
                float(labels[session][policy]["binary_success"]) for session in sessions
            )
            / len(sessions),
            "benchmark_clustered_bootstrap_ci95": [
                _percentile(actual_samples[policy], 0.025),
                _percentile(actual_samples[policy], 0.975),
            ],
        }
        for policy in policies
    }


def _kendall_permutation(
    predicted: Mapping[str, float], actual: Mapping[str, float]
) -> dict[str, Any]:
    policies = sorted(set(predicted) & set(actual))
    observed = _kendall_tau_b(predicted, actual)
    null: list[float] = []
    actual_values = [actual[policy] for policy in policies]
    for permutation in itertools.permutations(actual_values):
        value = _kendall_tau_b(predicted, dict(zip(policies, permutation)))
        if value is not None:
            null.append(value)
    p_value = (
        sum(value >= float(observed) for value in null) / len(null)
        if observed is not None and null
        else None
    )
    return {
        "observed_tau_b": observed,
        "exact_permutation_count": len(null),
        "one_sided_p_value": p_value,
    }


def build_benchmark_report(
    *,
    protocol: Mapping[str, Any],
    inventory: Mapping[str, Any],
    evidence_root: str | Path,
    prediction_freeze: Mapping[str, Any],
    causal_report: Mapping[str, Any],
    roboarena_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if prediction_freeze.get("status") != "frozen":
        raise ValueError("prediction_matrix_not_frozen")
    judgments = accepted_judgments(evidence_root)
    if len(judgments) != int(inventory["request_count"]):
        raise ValueError("prediction_matrix_incomplete")
    base = evaluate_frozen_calibration(
        judgments,
        protocol=protocol,
        roboarena_root=roboarena_root,
        partition="heldout",
        expected_evaluator_digest=str(inventory["evaluator_digest"]),
    )
    temporal_name = str(protocol["evaluator"]["full_temporal_method"])
    endpoint_name = str(protocol["evaluator"]["cheap_baseline_method"])
    temporal = base["methods"][temporal_name]
    endpoint = base["methods"][endpoint_name]
    pair_rows = _pair_rows(judgments, protocol=protocol, roboarena_root=roboarena_root)
    risk = _risk_coverage(pair_rows)
    intervals = _policy_intervals(judgments, protocol=protocol, roboarena_root=roboarena_root)
    predicted = temporal["predicted_policy_success"]
    actual = {
        policy: temporal["benchmark_policy_success"][policy]
        * (len(protocol["partitions"]["heldout"]) + 1)
        + temporal["benchmark_policy_partial_success"][policy]
        for policy in temporal["benchmark_policy_success"]
    }
    permutation = _kendall_permutation(predicted, actual)
    gates = {
        "pairwise_lower95_gt_chance": bool(
            temporal["session_pairwise_accuracy_bootstrap_ci95"][0] > 0.5
        ),
        "pairwise_accuracy_reaches_power_target": bool(
            temporal["session_pairwise_accuracy"]
            >= protocol["ranking_gates"]["minimum_detectable_pairwise_accuracy_at_80pct_power"]
        ),
        "kendall_positive_and_significant": bool(
            (permutation["observed_tau_b"] or 0.0) > 0.0
            and (permutation["one_sided_p_value"] or 1.0)
            < protocol["ranking_gates"]["kendall_permutation_one_sided_alpha"]
        ),
        "top_policy_correct": temporal["top_policy"] == temporal["benchmark_top_policy"],
        "selective_coverage": bool(
            temporal["selective_session_pairwise_coverage"]
            >= protocol["ranking_gates"]["selective_pairwise_coverage_min"]
        ),
        "selective_accuracy_lower95": bool(
            temporal["selective_session_pairwise_accuracy_bootstrap_ci95"][0] is not None
            and temporal["selective_session_pairwise_accuracy_bootstrap_ci95"][0]
            >= protocol["ranking_gates"]["selective_pairwise_accuracy_lower95_min"]
        ),
        "temporal_beats_endpoint_by_margin": bool(
            temporal["session_pairwise_accuracy"] - endpoint["session_pairwise_accuracy"]
            >= protocol["ranking_gates"]["temporal_minus_endpoint_pairwise_accuracy_min"]
        ),
        "causal_residual_alignment": bool(
            causal_report["gates"]["residual_excess_lower95_above_margin"]
        ),
        "causal_validity_pass_rate": bool(
            causal_report["gates"]["residual_validity_pass_rate_lower95"]
        ),
        "conditioning_annotation_not_sole_signal": bool(
            causal_report["gates"]["conditioning_annotation_not_sole_signal"]
        ),
        "uncertainty_error_association": bool(risk["uncertainty_error_association_descriptive"]),
        "complete_risk_coverage_curve": len(risk["curve"]) == len(pair_rows),
        "evidence_storage_complete": True,
        "pristine_metadata_sealing": False,
    }
    scientific_gates = {
        key: value for key, value in gates.items() if key != "pristine_metadata_sealing"
    }
    result: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": base["status"],
        "component": "frozen_benchmark_calibration",
        "component_verdict": (
            "not_supported"
            if not causal_report["all_action_following_validity_gates_passed"]
            or (base["status"] == "completed" and not all(scientific_gates.values()))
            else "inconclusive"
            if base["status"] != "completed"
            else "supported"
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "prediction_freeze_sha256": prediction_freeze["freeze_sha256"],
        "provider_request_count": inventory["request_count"],
        "provider_estimated_cost_usd": prediction_freeze["estimated_cost_usd_recomputed"],
        "base_registered_metrics": base,
        "kendall_exact_permutation": permutation,
        "policy_rankings_with_clustered_intervals": intervals,
        "risk_coverage_report_sha256": risk["report_sha256"],
        "gates": gates,
        "all_scientific_gates_passed": all(scientific_gates.values()),
        "procedural_deviation": "Potential held-out free-text context was displayed after the immutable provider matrix began but before it completed. Exact outcome fields were not displayed; no inputs, predictions, metrics, or thresholds changed.",
        "claim_boundary": "Benchmark ranking fidelity against third-party real-policy outcomes only; not Blueprint physical execution, captured-site accuracy, or WAM counterfactual causality.",
    }
    result["report_sha256"] = canonical_sha256(result)
    return result, risk


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--inventory", required=True)
    freeze.add_argument("--evidence-root", required=True)
    freeze.add_argument("--output", required=True)
    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--protocol", required=True)
    adjudicate.add_argument("--inventory", required=True)
    adjudicate.add_argument("--evidence-root", required=True)
    adjudicate.add_argument("--prediction-freeze", required=True)
    adjudicate.add_argument("--causal-report", required=True)
    adjudicate.add_argument("--roboarena-root", required=True)
    adjudicate.add_argument("--output", required=True)
    adjudicate.add_argument("--risk-output", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = build_prediction_freeze(_read(args.inventory), evidence_root=args.evidence_root)
        write_json(Path(args.output), result)
        return 0 if result["status"] == "frozen" else 2
    report, risk = build_benchmark_report(
        protocol=_read(args.protocol),
        inventory=_read(args.inventory),
        evidence_root=args.evidence_root,
        prediction_freeze=_read(args.prediction_freeze),
        causal_report=_read(args.causal_report),
        roboarena_root=args.roboarena_root,
    )
    write_json(Path(args.output), report)
    write_json(Path(args.risk_output), risk)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
