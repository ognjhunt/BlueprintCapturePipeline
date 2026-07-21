"""Blueprint-owned qualification for the distinct Cosmos 3 Edge experiment lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .common import read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import canonical_sha256


SCHEMA_VERSION = "cosmos3_edge_qualification.v1"
SCORECARD_SCHEMA_VERSION = "cosmos3_edge_blueprint_scorecard.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank(values: list[float]) -> list[float]:
    """Return average ranks for ties, with the lowest value ranked first."""

    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            result[original_index] = average
        index = end
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = _rank(left)
    y = _rank(right)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    denominator_left = sum((a - mean_x) ** 2 for a in x)
    denominator_right = sum((b - mean_y) ** 2 for b in y)
    denominator = (denominator_left * denominator_right) ** 0.5
    return numerator / denominator if denominator else None


def build_cosmos3_edge_qualification(
    *,
    attempt_manifest_path: str | Path,
    evaluator_runtime_receipt_path: str | Path,
    scorecard_path: str | Path,
    output_path: str | Path,
    expected_evaluator_id: str,
    minimum_grounding_score: float = 0.8,
    minimum_abstention_accuracy: float = 0.8,
    minimum_spearman_rank_correlation: float = 0.8,
    minimum_failure_recall: float = 1.0,
) -> dict[str, Any]:
    manifest_path = Path(attempt_manifest_path).resolve()
    receipt_path = Path(evaluator_runtime_receipt_path).resolve()
    card_path = Path(scorecard_path).resolve()
    manifest = _mapping(read_json_any(manifest_path))
    receipt = _mapping(read_json_any(receipt_path))
    scorecard = _mapping(read_json_any(card_path))
    blockers: list[str] = []
    if manifest.get("schema_version") != "cosmos3_edge_experiment_attempt_manifest.v1":
        blockers.append("edge_attempt_manifest_schema_invalid")
    if manifest.get("status") != "completed_advisory" or manifest.get("blockers"):
        blockers.append("edge_attempt_manifest_not_completed")
    boundary = _mapping(manifest.get("claim_boundary"))
    if boundary.get("cosmos3_nano_sc3_qualification_inherited") is not False:
        blockers.append("edge_attempt_manifest_improperly_inherits_nano_qualification")
    attempts = [
        _mapping(value) for value in manifest.get("attempts", []) if isinstance(value, Mapping)
    ]
    attempt_ids = {str(item.get("attempt_id") or "") for item in attempts}
    if not attempts or "" in attempt_ids or len(attempt_ids) != len(attempts):
        blockers.append("edge_attempt_identity_set_invalid")
    if any(item.get("status") != "completed" for item in attempts):
        blockers.append("edge_attempt_set_contains_incomplete_attempt")
    modes = {str(item.get("mode") or "") for item in attempts}
    if modes != {"forward_dynamics", "inverse_dynamics", "reasoning"}:
        blockers.append("edge_qualification_requires_all_three_modes")
    stability = [
        _mapping(value)
        for value in manifest.get("output_stability", [])
        if isinstance(value, Mapping)
    ]
    if not stability or any(int(row.get("repeat_count") or 0) < 2 for row in stability):
        blockers.append("edge_output_stability_not_measured_with_repeats")
    if any(row.get("exact_output_digest_stable") is not True for row in stability):
        blockers.append("edge_output_digest_not_stable_across_repeats")

    if receipt.get("status") != "validated":
        blockers.append("edge_evaluator_runtime_receipt_not_validated")
    if str(receipt.get("model_family") or "").lower() != "cosmos3edge":
        blockers.append("edge_evaluator_runtime_receipt_model_family_mismatch")
    if scorecard.get("schema_version") != SCORECARD_SCHEMA_VERSION:
        blockers.append("edge_scorecard_schema_invalid")
    if scorecard.get("frozen_before_scoring") is not True:
        blockers.append("edge_scorecard_not_frozen_before_scoring")
    evaluator_id = str(scorecard.get("configured_evaluator_id") or "")
    if not expected_evaluator_id or evaluator_id != expected_evaluator_id:
        blockers.append("edge_configured_evaluator_identity_mismatch")
    if scorecard.get("evaluator_runtime_receipt_sha256") != sha256_file(receipt_path):
        blockers.append("edge_scorecard_runtime_receipt_digest_mismatch")
    rows = [
        _mapping(value)
        for value in scorecard.get("attempt_scores", [])
        if isinstance(value, Mapping)
    ]
    scored_ids = {str(row.get("attempt_id") or "") for row in rows}
    if scored_ids != attempt_ids or len(rows) != len(attempts):
        blockers.append("edge_scorecard_attempt_coverage_mismatch")
    accepted_anchor_ids = {
        str(row.get("accepted_anchor_id") or "") for row in rows if row.get("accepted_anchor_id")
    }
    if len(accepted_anchor_ids) < 2:
        blockers.append("edge_qualification_requires_at_least_two_accepted_anchors")
    if any(row.get("anchor_review_status") != "accepted" for row in rows):
        blockers.append("edge_scorecard_contains_unaccepted_anchor")

    grounding = [_number(row.get("grounding_score")) for row in rows]
    grounding_values = [value for value in grounding if value is not None]
    grounding_mean = (
        sum(grounding_values) / len(grounding_values)
        if len(grounding_values) == len(rows) and rows
        else None
    )
    abstention_values = [row.get("abstention_correct") for row in rows]
    abstention_accuracy = (
        sum(value is True for value in abstention_values) / len(abstention_values)
        if rows and all(isinstance(value, bool) for value in abstention_values)
        else None
    )
    expected_ranks = [_number(row.get("expected_rank")) for row in rows]
    observed_scores = [_number(row.get("observed_score")) for row in rows]
    rank_correlation = (
        _spearman(
            [value for value in expected_ranks if value is not None],
            [-value for value in observed_scores if value is not None],
        )
        if all(value is not None for value in expected_ranks + observed_scores)
        else None
    )
    expected_failures = [row for row in rows if row.get("failure_expected") is True]
    failure_recall = (
        sum(row.get("failure_detected") is True for row in expected_failures)
        / len(expected_failures)
        if expected_failures
        else None
    )
    metrics = {
        "mean_grounding_score": grounding_mean,
        "abstention_accuracy": abstention_accuracy,
        "spearman_rank_correlation": rank_correlation,
        "failure_recall": failure_recall,
        "accepted_anchor_count": len(accepted_anchor_ids),
        "attempt_count": len(rows),
        "all_output_groups_exact_digest_stable": bool(stability)
        and all(row.get("exact_output_digest_stable") is True for row in stability),
    }
    thresholds = {
        "minimum_grounding_score": minimum_grounding_score,
        "minimum_abstention_accuracy": minimum_abstention_accuracy,
        "minimum_spearman_rank_correlation": minimum_spearman_rank_correlation,
        "minimum_failure_recall": minimum_failure_recall,
    }
    for metric, threshold in (
        ("mean_grounding_score", minimum_grounding_score),
        ("abstention_accuracy", minimum_abstention_accuracy),
        ("spearman_rank_correlation", minimum_spearman_rank_correlation),
        ("failure_recall", minimum_failure_recall),
    ):
        value = metrics[metric]
        if value is None or value < threshold:
            blockers.append(f"edge_qualification_threshold_not_met:{metric}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "qualified_advisory" if not blockers else "not_qualified",
        "attempt_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "evaluator_runtime_receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        },
        "scorecard": {"path": str(card_path), "sha256": sha256_file(card_path)},
        "configured_evaluator_id": evaluator_id or None,
        "metrics": metrics,
        "thresholds": thresholds,
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "edge_blueprint_data_qualification_measured": not blockers,
            "cosmos3_nano_qualification_inherited": False,
            "structured_physics_truth_proven": False,
            "real_world_correlation_proven": False,
            "safety_certification_proven": False,
            "default_model_change_allowed": False,
            "default_change_requires_separate_owner_decision": True,
        },
    }
    payload["qualification_fingerprint"] = canonical_sha256(payload)
    write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qualify Cosmos 3 Edge on Blueprint evidence")
    parser.add_argument("--attempt-manifest", required=True)
    parser.add_argument("--evaluator-runtime-receipt", required=True)
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--expected-evaluator-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_cosmos3_edge_qualification(
        attempt_manifest_path=args.attempt_manifest,
        evaluator_runtime_receipt_path=args.evaluator_runtime_receipt,
        scorecard_path=args.scorecard,
        expected_evaluator_id=args.expected_evaluator_id,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "qualified_advisory" else 2


if __name__ == "__main__":
    raise SystemExit(main())
