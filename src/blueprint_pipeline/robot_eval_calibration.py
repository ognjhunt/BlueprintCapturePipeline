"""Deployment calibration and rank-fidelity decision engine.

This module is the typed, shared seam for joining simulated predictions to
accepted real-world anchors, computing calibration metrics, bootstrapping
confidence intervals, and deciding whether a bounded rank-fidelity claim is
eligible.  It intentionally does not load artifacts or perform deployments.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Mapping, Sequence, TypedDict


ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION = "accepted_real_world_anchor.v1"
ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS = (
    "scenario_eval_run_id",
    "policy_id",
    "task_id",
    "scenario_variation_instance_id",
)
MIN_ACCEPTED_ANCHOR_COUNT_FOR_CALIBRATION = 4
MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION = 2
MIN_POLICY_CHECKPOINT_COUNT_FOR_PUBLIC_RANK_FIDELITY = 7
MIN_CRITERION_COUNT_FOR_PUBLIC_RANK_FIDELITY = 3
MIN_REGISTERED_SPLIT_COUNT_FOR_PUBLIC_RANK_FIDELITY = 2
MIN_MATCHED_TRIALS_PER_CELL_FOR_PUBLIC_RANK_FIDELITY = 20
DEFAULT_CALIBRATION_BOOTSTRAP_SEED = 1729
DEFAULT_CALIBRATION_BOOTSTRAP_REPLICATES = 10_000
RANK_FIDELITY_CLAIM_ELIGIBILITY_SCHEMA_VERSION = "rank_fidelity_claim_eligibility.v1"
UNIT_OF_ANALYSIS_FIELDS = (
    "policy_id",
    "checkpoint_id",
    "criterion_id",
    "registered_split",
    "task_family",
)


class RankFidelityClaimEligibility(TypedDict, total=False):
    schema_version: str
    status: str
    scope: str
    public_rank_fidelity_claim_eligible: bool
    deployment_accuracy_claim_supported: bool
    real_world_success_rate_prediction_claim_supported: bool
    design: Dict[str, Any]
    metrics: Dict[str, Any]
    blockers: List[str]
    claim_boundary: str


class AcceptedAnchorCalibration(TypedDict, total=False):
    status: str
    blockers: List[str]
    accepted_anchor_schema: Dict[str, Any]
    accepted_anchor_count: int
    accepted_anchors: List[Dict[str, Any]]
    rejected_anchors: List[Dict[str, Any]]
    policy_success_rate_rows: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Any]
    estimands: Dict[str, Any]
    rank_fidelity_claim_eligibility: RankFidelityClaimEligibility


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "passed",
        "success",
        "succeeded",
    }


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [_string(item) for item in value if _string(item)]
    return []


def _prediction_index(
    prediction_ledger: Mapping[str, Any],
    attempt_trace: Mapping[str, Any],
) -> Dict[tuple[str, str, str, str], Dict[str, Any]]:
    index: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}

    def add_prediction(record: Mapping[str, Any], prediction: Dict[str, Any]) -> None:
        task_id = _string(record.get("task_id"))
        scenario_id = _string(record.get("scenario_id"))
        run_id = _string(record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId"))
        variation_id = _string(
            record.get("scenario_variation_instance_id")
            or record.get("scenarioVariationInstanceId")
        )
        keys = []
        if run_id and variation_id:
            keys.append((task_id, scenario_id, run_id, variation_id))
        if run_id:
            keys.append((task_id, scenario_id, run_id, ""))
        if variation_id:
            keys.append((task_id, scenario_id, "", variation_id))
        keys.append((task_id, scenario_id, "", ""))
        for key in keys:
            if key[:2] != ("", "") and key not in index:
                index[key] = dict(prediction)

    for record in prediction_ledger.get("records", []) or []:
        if not isinstance(record, Mapping):
            continue
        add_prediction(record, dict(record))
    for attempt in attempt_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        prediction = {
            "task_id": _string(attempt.get("task_id")),
            "scenario_id": _string(attempt.get("scenario_id")),
            "scenario_eval_run_id": _string(
                attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId")
            )
            or None,
            "scenario_variation_instance_id": _string(
                attempt.get("scenario_variation_instance_id")
                or attempt.get("scenarioVariationInstanceId")
            )
            or None,
            "variation_name": attempt.get("variation_name"),
            "predicted_success": attempt.get("predicted_success"),
            "predicted_cycle_time_seconds": attempt.get("predicted_cycle_time_seconds"),
            "failure_mode_ids": attempt.get("failure_mode_ids") or [],
            "source": "normalized_attempt_trace",
        }
        add_prediction(attempt, prediction)
    return index


def _prediction_for_actual(
    predictions: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    task_id: str,
    scenario_id: str,
    scenario_eval_run_id: str,
    scenario_variation_instance_id: str,
) -> tuple[Dict[str, Any], str]:
    keys: List[tuple[tuple[str, str, str, str], str]] = []
    if scenario_eval_run_id and scenario_variation_instance_id:
        keys.append(
            (
                (task_id, scenario_id, scenario_eval_run_id, scenario_variation_instance_id),
                "scenario_eval_run_and_variation",
            )
        )
    if scenario_eval_run_id:
        keys.append(((task_id, scenario_id, scenario_eval_run_id, ""), "scenario_eval_run"))
    if scenario_variation_instance_id:
        keys.append(
            (
                (task_id, scenario_id, "", scenario_variation_instance_id),
                "scenario_variation_instance",
            )
        )
    if not scenario_eval_run_id and not scenario_variation_instance_id:
        keys.append(((task_id, scenario_id, "", ""), "task_scenario_fallback"))
    for key, match_level in keys:
        prediction = predictions.get(key)
        if prediction:
            return dict(prediction), match_level
    return {}, "unmatched"


def _predicted_success(record: Mapping[str, Any]) -> bool | None:
    if "predicted_success" in record:
        value = record.get("predicted_success")
        return _boolish(value) if value is not None else None
    for key in ("success", "task_success", "predicted_task_success"):
        if key in record and record.get(key) is not None:
            return _boolish(record.get(key))
    status = _string(record.get("predicted_status") or record.get("prediction_status")).lower()
    if status in {"pass", "passed", "success", "succeeded", "completed"}:
        return True
    if status in {"fail", "failed", "failure", "predicted_failure"}:
        return False
    failures = _string_list(record.get("failure_mode_ids"))
    if failures:
        return False
    return None


def _actual_success(record: Mapping[str, Any]) -> bool | None:
    for key in ("actual_success", "actualSuccess", "success", "passed"):
        if key in record and record.get(key) is not None:
            return _boolish(record.get(key))
    status = _string(record.get("actual_status") or record.get("status")).lower()
    if status in {"pass", "passed", "success", "succeeded", "completed"}:
        return True
    if status in {"fail", "failed", "failure", "timeout", "collision"}:
        return False
    return None


def _failure_ids(record: Mapping[str, Any], *keys: str) -> List[str]:
    for key in keys:
        values = _string_list(record.get(key))
        if values:
            return values
    return []


def _actual_signal_present(record: Mapping[str, Any]) -> bool:
    for key in (
        "actual_success",
        "actualSuccess",
        "success",
        "passed",
        "actual_status",
        "actualStatus",
        "status",
    ):
        if key in record and _string(record.get(key)):
            return True
    return bool(
        _failure_ids(record, "failure_mode_ids", "actual_failures", "actualFailures", "failures")
    )


def _anchor_variation_instance_id(record: Mapping[str, Any]) -> str:
    return _string(
        record.get("scenario_variation_instance_id")
        or record.get("scenarioVariationInstanceId")
        or record.get("variation_instance_id")
        or record.get("variationInstanceId")
        or record.get("variation_id")
        or record.get("variationId")
    )


def _anchor_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _string(record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId")),
        _string(record.get("policy_id") or record.get("policyId")),
        _string(record.get("task_id") or record.get("taskId")),
        _anchor_variation_instance_id(record),
    )


def _anchor_key_dict(key: tuple[str, str, str, str]) -> Dict[str, str]:
    return {
        "scenario_eval_run_id": key[0],
        "policy_id": key[1],
        "task_id": key[2],
        "scenario_variation_instance_id": key[3],
    }


def _missing_anchor_key_fields(key: tuple[str, str, str, str]) -> List[str]:
    return [field for field, value in zip(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS, key) if not value]


def _anchor_record_status(record: Mapping[str, Any]) -> str:
    return _string(
        record.get("anchor_status")
        or record.get("anchorStatus")
        or record.get("validation_status")
        or record.get("validationStatus")
        or record.get("review_status")
        or record.get("reviewStatus")
    ).lower()


def _anchor_record_is_stale(record: Mapping[str, Any]) -> bool:
    if _boolish(record.get("stale") or record.get("is_stale") or record.get("isStale")):
        return True
    return _anchor_record_status(record) in {"stale", "expired", "superseded"}


def _accepted_review_value(value: Any) -> bool:
    return _string(value).lower() in {
        "accepted",
        "approved",
        "passed",
        "succeeded",
        "complete",
        "completed",
    }


def _anchor_review_accepted(record: Mapping[str, Any]) -> bool:
    reviewer_decision = _mapping(record.get("reviewer_decision") or record.get("reviewerDecision"))
    if _boolish(
        record.get("accepted_for_calibration")
        or record.get("acceptedForCalibration")
        or reviewer_decision.get("accepted_for_calibration")
        or reviewer_decision.get("acceptedForCalibration")
    ):
        return True
    for key in (
        "calibration_review_decision",
        "calibrationReviewDecision",
        "policy_review_decision",
        "policyReviewDecision",
        "review_decision",
        "reviewDecision",
        "status",
    ):
        if _accepted_review_value(reviewer_decision.get(key)):
            return True
    return _accepted_review_value(_anchor_record_status(record))


def _attestation_signed(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    actor_id = _string(
        value.get("attested_by")
        or value.get("attestedBy")
        or value.get("operator_id")
        or value.get("operatorId")
        or value.get("owner_id")
        or value.get("ownerId")
    )
    statement = _string(
        value.get("statement")
        or value.get("attestation")
        or value.get("accepted_claim_boundary")
        or value.get("acceptedClaimBoundary")
    )
    status = _string(
        value.get("status") or value.get("signature_status") or value.get("signatureStatus")
    )
    signature_ref = _string(
        value.get("signature")
        or value.get("signature_ref")
        or value.get("signatureRef")
        or value.get("signed_at_utc")
        or value.get("signedAtUtc")
    )
    return bool(actor_id and statement and (status == "signed" or signature_ref))


def _physical_evidence_requested(record: Mapping[str, Any]) -> bool:
    reviewer_decision = _mapping(record.get("reviewer_decision") or record.get("reviewerDecision"))
    return bool(
        _boolish(record.get("physical_evidence_required"))
        or _boolish(record.get("physicalEvidenceRequired"))
        or _boolish(record.get("field_evidence_required"))
        or _boolish(record.get("fieldEvidenceRequired"))
        or _boolish(reviewer_decision.get("physical_evidence_required"))
        or _boolish(reviewer_decision.get("physicalEvidenceRequired"))
    )


def _physical_evidence_present(record: Mapping[str, Any]) -> bool:
    refs = _mapping(record.get("physical_evidence_refs") or record.get("physicalEvidenceRefs"))
    owner_refs = _mapping(record.get("owner_evidence_refs") or record.get("ownerEvidenceRefs"))
    evidence_refs = _mapping(record.get("evidence_refs") or record.get("evidenceRefs"))
    if refs:
        return True
    physical_keys = {
        "physical_robot_run_manifest",
        "robot_camera_video",
        "robot_pov_video",
        "video_review",
        "action_log",
        "robot_state_log",
        "timestamp_alignment",
        "operator_log",
    }
    if physical_keys.intersection(owner_refs) or physical_keys.intersection(evidence_refs):
        return True
    return bool(
        _string(record.get("robot_camera_video_uri") or record.get("robotCameraVideoUri"))
        or _string(record.get("robot_pov_video_uri") or record.get("robotPovVideoUri"))
        or _string(
            record.get("physical_robot_run_manifest_uri")
            or record.get("physicalRobotRunManifestUri")
        )
    )


def _prediction_anchor_rows(
    prediction_ledger: Mapping[str, Any],
    attempt_trace: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in prediction_ledger.get("records", []) or []:
        if not isinstance(record, Mapping):
            continue
        rows.append(
            {
                **dict(record),
                "prediction_source": record.get("source") or "prediction_outcome_ledger",
            }
        )
    for attempt in attempt_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        rows.append(
            {
                "task_id": _string(attempt.get("task_id") or attempt.get("taskId")),
                "scenario_id": _string(attempt.get("scenario_id") or attempt.get("scenarioId")),
                "scenario_eval_run_id": _string(
                    attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId")
                )
                or None,
                "scenario_variation_instance_id": _anchor_variation_instance_id(attempt) or None,
                "variation_name": attempt.get("variation_name") or attempt.get("variationName"),
                "policy_id": _string(attempt.get("policy_id") or attempt.get("policyId")),
                "predicted_success": _predicted_success(attempt),
                "predicted_cycle_time_seconds": _number(
                    attempt.get("predicted_cycle_time_seconds")
                    or attempt.get("cycle_time_seconds")
                    or _mapping(attempt.get("metrics")).get("cycle_time_seconds")
                ),
                "failure_mode_ids": attempt.get("failure_mode_ids") or [],
                "prediction_source": "normalized_attempt_trace",
            }
        )
    return rows


def _prediction_anchor_index(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    Dict[tuple[str, str, str, str], Dict[str, Any]],
    List[str],
    List[Dict[str, Any]],
]:
    index: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    conflicts: Dict[tuple[str, str, str, str], set[bool]] = {}
    incomplete: List[Dict[str, Any]] = []
    for row_index, row in enumerate(prediction_rows, start=1):
        key = _anchor_key(row)
        missing_fields = _missing_anchor_key_fields(key)
        record_id = (
            _string(row.get("record_id") or row.get("attempt_id") or row.get("id"))
            or f"prediction_row_{row_index:04d}"
        )
        predicted = _predicted_success(row)
        if missing_fields or predicted is None:
            incomplete.append(
                {
                    "record_id": record_id,
                    "missing_fields": missing_fields
                    + (["predicted_success"] if predicted is None else []),
                    "join_key": _anchor_key_dict(key),
                }
            )
            continue
        conflicts.setdefault(key, set()).add(bool(predicted))
        index.setdefault(
            key,
            {
                **dict(row),
                "record_id": record_id,
                "predicted_success": bool(predicted),
                "anchor_join_key": _anchor_key_dict(key),
            },
        )
    conflict_ids = [
        _string(index.get(key, {}).get("record_id")) or "|".join(key)
        for key, values in conflicts.items()
        if len(values) > 1
    ]
    return index, sorted(conflict_ids), incomplete


def _average_ranks(values: Sequence[float], *, descending: bool = False) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=descending)
    ranks = [0.0 for _ in values]
    position = 1
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (position + position + (end - cursor) - 1) / 2.0
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = average_rank
        position += end - cursor
        cursor = end
    return ranks


def _pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(sum(value * value for value in centered_a)) * math.sqrt(
        sum(value * value for value in centered_b)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_a, centered_b)) / denominator


def _policy_anchor_summaries(
    accepted_anchors: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate only at the declared checkpoint x criterion x split/task unit.

    Legacy anchors may omit the newer study-design fields. They remain usable for
    bounded diagnostics, but the fallback values are explicit in every summary
    and make the public-claim design gate ineligible.
    """

    grouped: Dict[tuple[str, str, str, str, str], List[Mapping[str, Any]]] = {}
    for row in accepted_anchors:
        policy_id = _string(row.get("policy_id")) or "policy_unspecified"
        checkpoint_id = (
            _string(
                row.get("checkpoint_id")
                or row.get("checkpointId")
                or row.get("policy_checkpoint_id")
                or row.get("policyCheckpointId")
            )
            or policy_id
        )
        criterion_id = _string(
            row.get("criterion_id")
            or row.get("criterionId")
            or row.get("success_criterion_id")
            or row.get("successCriterionId")
        ) or (_string(row.get("task_id")) or "criterion_unspecified")
        registered_split = (
            _string(
                row.get("registered_split")
                or row.get("registeredSplit")
                or row.get("evaluation_split")
                or row.get("evaluationSplit")
                or row.get("split")
            )
            or "unregistered"
        )
        task_family = _string(
            row.get("task_family")
            or row.get("taskFamily")
            or row.get("registered_task_family")
            or row.get("registeredTaskFamily")
        ) or (_string(row.get("task_id")) or "task_family_unspecified")
        bootstrap_policy = _string(row.get("_bootstrap_policy_instance"))
        bootstrap_cell = _string(row.get("_bootstrap_cell_instance"))
        if bootstrap_policy:
            checkpoint_id = f"{checkpoint_id}::bootstrap_policy_{bootstrap_policy}"
        if bootstrap_cell:
            task_family = f"{task_family}::bootstrap_cell_{bootstrap_cell}"
        grouped.setdefault(
            (policy_id, checkpoint_id, criterion_id, registered_split, task_family),
            [],
        ).append(row)
    summaries: List[Dict[str, Any]] = []
    for unit_key, rows in sorted(grouped.items()):
        policy_id, checkpoint_id, criterion_id, registered_split, task_family = unit_key
        predicted_successes = [bool(row.get("predicted_success")) for row in rows]
        actual_successes = [bool(row.get("actual_success")) for row in rows]
        predicted_success_rate = sum(predicted_successes) / len(predicted_successes)
        actual_success_rate = sum(actual_successes) / len(actual_successes)
        explicit_checkpoint = all(
            bool(
                _string(
                    row.get("checkpoint_id")
                    or row.get("checkpointId")
                    or row.get("policy_checkpoint_id")
                    or row.get("policyCheckpointId")
                )
            )
            for row in rows
        )
        explicit_criterion = all(
            bool(
                _string(
                    row.get("criterion_id")
                    or row.get("criterionId")
                    or row.get("success_criterion_id")
                    or row.get("successCriterionId")
                )
            )
            for row in rows
        )
        explicit_split = all(
            bool(
                _string(
                    row.get("registered_split")
                    or row.get("registeredSplit")
                    or row.get("evaluation_split")
                    or row.get("evaluationSplit")
                    or row.get("split")
                )
            )
            for row in rows
        )
        explicit_task_family = all(
            bool(
                _string(
                    row.get("task_family")
                    or row.get("taskFamily")
                    or row.get("registered_task_family")
                    or row.get("registeredTaskFamily")
                )
            )
            for row in rows
        )
        summaries.append(
            {
                "policy_id": policy_id,
                "checkpoint_id": checkpoint_id,
                "criterion_id": criterion_id,
                "registered_split": registered_split,
                "task_family": task_family,
                "unit_of_analysis_key": {
                    field: value for field, value in zip(UNIT_OF_ANALYSIS_FIELDS, unit_key)
                },
                "unit_of_analysis_key_explicit": bool(
                    explicit_checkpoint
                    and explicit_criterion
                    and explicit_split
                    and explicit_task_family
                ),
                "unit_of_analysis_fallbacks": sorted(
                    name
                    for name, explicit in (
                        ("checkpoint_id_from_policy_id", explicit_checkpoint),
                        ("criterion_id_from_task_id", explicit_criterion),
                        ("registered_split_unregistered", explicit_split),
                        ("task_family_from_task_id", explicit_task_family),
                    )
                    if not explicit
                ),
                "accepted_anchor_count": len(rows),
                "predicted_success_rate": round(predicted_success_rate, 6),
                "actual_success_rate": round(actual_success_rate, 6),
                "success_rate_error": round(predicted_success_rate - actual_success_rate, 6),
                "absolute_success_rate_error": round(
                    abs(predicted_success_rate - actual_success_rate),
                    6,
                ),
            }
        )
    return _summaries_with_rank_position_diagnostics(summaries)


def _summaries_with_rank_position_diagnostics(
    summaries: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ranked = [dict(row) for row in summaries]
    predicted_ranks = _average_ranks(
        [float(row.get("predicted_success_rate") or 0.0) for row in ranked],
        descending=True,
    )
    actual_ranks = _average_ranks(
        [float(row.get("actual_success_rate") or 0.0) for row in ranked],
        descending=True,
    )
    for row, predicted_rank, actual_rank in zip(ranked, predicted_ranks, actual_ranks):
        row["predicted_rank"] = predicted_rank
        row["actual_rank"] = actual_rank
        row["rank_position_error"] = abs(predicted_rank - actual_rank)
        row["normalized_rank_position_error"] = row["rank_position_error"] / max(1, len(ranked) - 1)
    return ranked


def _simpler_pairwise_margin_rank_violations(
    predicted_rates: Sequence[float],
    actual_rates: Sequence[float],
) -> List[float]:
    """Return each unit's maximum SIMPLER pairwise real-rate violation.

    This ports the reference implementation's strict ``>`` comparison. A tie
    in both domains is not a violation; a tie in one domain is treated exactly
    as the published strict-comparison XOR. A real-rate tie always has zero
    margin and therefore contributes zero even when the simulated tie differs.
    """

    if len(predicted_rates) != len(actual_rates):
        raise ValueError("predicted_and_actual_rate_lengths_must_match")
    maximum_violations: List[float] = []
    for index, (predicted, actual) in enumerate(zip(predicted_rates, actual_rates)):
        maximum = 0.0
        for other_index, (other_predicted, other_actual) in enumerate(
            zip(predicted_rates, actual_rates)
        ):
            if index == other_index:
                continue
            ordering_disagrees = (predicted > other_predicted) != (actual > other_actual)
            if ordering_disagrees:
                maximum = max(maximum, abs(actual - other_actual))
        maximum_violations.append(maximum)
    return maximum_violations


def _calibration_metrics_from_policy_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if len(summaries) < MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION:
        return {
            "spearman_rank_correlation": None,
            "pearson_success_rate_correlation": None,
            "mean_maximum_rank_violation": None,
            "mmrv": None,
            "maximum_pairwise_real_margin_rank_violation": None,
            "mean_normalized_rank_position_error": None,
            "maximum_normalized_rank_position_error": None,
            "mean_absolute_success_rate_error": None,
            "sim_vs_real_calibration_score": None,
            "mmrv_definition": "simpler_pairwise_real_success_rate_margin.v1",
        }
    ranked = _summaries_with_rank_position_diagnostics(summaries)
    predicted_rates = [float(row.get("predicted_success_rate") or 0.0) for row in ranked]
    actual_rates = [float(row.get("actual_success_rate") or 0.0) for row in ranked]
    predicted_ranks = [float(row.get("predicted_rank") or 0.0) for row in ranked]
    actual_ranks = [float(row.get("actual_rank") or 0.0) for row in ranked]
    rank_position_errors = [
        float(row.get("normalized_rank_position_error") or 0.0) for row in ranked
    ]
    absolute_errors = [float(row.get("absolute_success_rate_error") or 0.0) for row in ranked]
    pearson = _pearson(predicted_rates, actual_rates)
    spearman = _pearson(predicted_ranks, actual_ranks)
    mae = sum(absolute_errors) / len(absolute_errors)
    pairwise_margin_violations = _simpler_pairwise_margin_rank_violations(
        predicted_rates,
        actual_rates,
    )
    mmrv = sum(pairwise_margin_violations) / len(pairwise_margin_violations)
    return {
        "spearman_rank_correlation": round(spearman, 6) if spearman is not None else None,
        "pearson_success_rate_correlation": round(pearson, 6) if pearson is not None else None,
        "mean_maximum_rank_violation": round(mmrv, 6),
        "mmrv": round(mmrv, 6),
        "maximum_pairwise_real_margin_rank_violation": (
            round(max(pairwise_margin_violations), 6) if pairwise_margin_violations else None
        ),
        "mean_normalized_rank_position_error": round(
            sum(rank_position_errors) / len(rank_position_errors),
            6,
        ),
        "maximum_normalized_rank_position_error": (
            round(max(rank_position_errors), 6) if rank_position_errors else None
        ),
        "mean_absolute_success_rate_error": round(mae, 6),
        "sim_vs_real_calibration_score": round(max(0.0, 1.0 - mae), 6),
        "mmrv_definition": "simpler_pairwise_real_success_rate_margin.v1",
    }


def _macro_calibration_estimand(
    summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    cells: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in summaries:
        cells.setdefault(
            (
                _string(row.get("criterion_id")) or "criterion_unspecified",
                _string(row.get("registered_split")) or "unregistered",
                _string(row.get("task_family")) or "task_family_unspecified",
            ),
            [],
        ).append(row)
    cell_metrics: List[Dict[str, Any]] = []
    for cell_key, cell_rows in sorted(cells.items()):
        metrics = _calibration_metrics_from_policy_summaries(cell_rows)
        cell_metrics.append(
            {
                "criterion_id": cell_key[0],
                "registered_split": cell_key[1],
                "task_family": cell_key[2],
                "unit_count": len(cell_rows),
                **metrics,
            }
        )
    metric_names = (
        "pearson_success_rate_correlation",
        "spearman_rank_correlation",
        "mmrv",
        "mean_absolute_success_rate_error",
    )
    macro: Dict[str, float | None] = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in cell_metrics
            if isinstance(row.get(metric_name), (int, float))
            and not isinstance(row.get(metric_name), bool)
        ]
        macro[metric_name] = round(sum(values) / len(values), 6) if values else None
    return {
        "estimand": "equal_weight_registered_criterion_split_task_family_cells",
        "cell_count": len(cell_metrics),
        "metrics": macro,
        "cells": cell_metrics,
    }


def _registered_split_estimands(
    accepted_anchors: Sequence[Mapping[str, Any]],
    *,
    include_confidence_intervals: bool,
) -> Dict[str, Any]:
    by_split: Dict[str, List[Mapping[str, Any]]] = {}
    for row in accepted_anchors:
        split = _string(row.get("registered_split")) or "unregistered"
        by_split.setdefault(split, []).append(row)
    results: Dict[str, Any] = {}
    for split, split_rows in sorted(by_split.items()):
        summaries = _policy_anchor_summaries(split_rows)
        results[split] = {
            "registered_split": split,
            "unit_count": len(summaries),
            "accepted_anchor_count": len(split_rows),
            "metrics": _calibration_metrics_from_policy_summaries(summaries),
            "confidence_intervals": (
                _bootstrap_confidence_intervals(split_rows)
                if include_confidence_intervals
                and len(summaries) >= MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION
                else {}
            ),
        }
    return {
        "estimand": "reported_separately_for_each_registered_split",
        "splits": results,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[int(position)], 6)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 6)


def _bootstrap_confidence_intervals(
    accepted_anchors: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_CALIBRATION_BOOTSTRAP_SEED,
    replicate_count: int = DEFAULT_CALIBRATION_BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    """Seeded hierarchical cluster bootstrap over study cells and trials.

    Rows are canonically sorted before sampling, making the result invariant to
    input order. Each matched-initial-condition cluster is resampled as a whole;
    policies/checkpoints and criterion/split/task-family cells are resampled at
    their own levels. Duplicate draws receive private bootstrap instance IDs so
    they retain their sampling weight instead of being silently re-aggregated.
    """

    canonical_rows = sorted(
        (dict(row) for row in accepted_anchors),
        key=lambda row: json.dumps(row, sort_keys=True, default=str),
    )
    original_summaries = _policy_anchor_summaries(canonical_rows)
    if len(original_summaries) < MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION:
        return {}
    replicate_count = max(1, int(replicate_count))
    metric_samples: Dict[str, List[float]] = {
        "spearman_rank_correlation": [],
        "pearson_success_rate_correlation": [],
        "mean_maximum_rank_violation": [],
        "mmrv": [],
        "mean_normalized_rank_position_error": [],
        "mean_absolute_success_rate_error": [],
        "sim_vs_real_calibration_score": [],
    }
    policy_keys = sorted(
        {
            (
                _string(row.get("policy_id")) or "policy_unspecified",
                _string(row.get("checkpoint_id"))
                or _string(row.get("policy_checkpoint_id"))
                or _string(row.get("policy_id"))
                or "checkpoint_unspecified",
            )
            for row in canonical_rows
        }
    )
    cell_keys = sorted(
        {
            (
                _string(row.get("criterion_id"))
                or _string(row.get("task_id"))
                or "criterion_unspecified",
                _string(row.get("registered_split"))
                or _string(row.get("evaluation_split"))
                or "unregistered",
                _string(row.get("task_family"))
                or _string(row.get("task_id"))
                or "task_family_unspecified",
            )
            for row in canonical_rows
        }
    )
    rng = random.Random(int(seed))
    successful_replicates = 0
    for _replicate_index in range(replicate_count):
        sampled_policies = [rng.choice(policy_keys) for _ in policy_keys]
        sampled_cells = [rng.choice(cell_keys) for _ in cell_keys]
        sampled_rows: List[Dict[str, Any]] = []
        for cell_instance, cell_key in enumerate(sampled_cells):
            cell_rows = [
                row
                for row in canonical_rows
                if (
                    _string(row.get("criterion_id"))
                    or _string(row.get("task_id"))
                    or "criterion_unspecified",
                    _string(row.get("registered_split"))
                    or _string(row.get("evaluation_split"))
                    or "unregistered",
                    _string(row.get("task_family"))
                    or _string(row.get("task_id"))
                    or "task_family_unspecified",
                )
                == cell_key
            ]
            cluster_ids = sorted(
                {
                    _string(row.get("matched_initial_condition_id"))
                    or _string(row.get("initial_condition_id"))
                    or _anchor_variation_instance_id(row)
                    or _string(row.get("scenario_eval_run_id"))
                    for row in cell_rows
                }
            )
            if not cluster_ids:
                continue
            sampled_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
            for policy_instance, policy_key in enumerate(sampled_policies):
                for cluster_id in sampled_clusters:
                    for row in cell_rows:
                        row_policy_key = (
                            _string(row.get("policy_id")) or "policy_unspecified",
                            _string(row.get("checkpoint_id"))
                            or _string(row.get("policy_checkpoint_id"))
                            or _string(row.get("policy_id"))
                            or "checkpoint_unspecified",
                        )
                        row_cluster_id = (
                            _string(row.get("matched_initial_condition_id"))
                            or _string(row.get("initial_condition_id"))
                            or _anchor_variation_instance_id(row)
                            or _string(row.get("scenario_eval_run_id"))
                        )
                        if row_policy_key == policy_key and row_cluster_id == cluster_id:
                            sampled_rows.append(
                                {
                                    **dict(row),
                                    "_bootstrap_policy_instance": str(policy_instance),
                                    "_bootstrap_cell_instance": str(cell_instance),
                                }
                            )
        sample_summaries = _policy_anchor_summaries(sampled_rows)
        metrics = _calibration_metrics_from_policy_summaries(sample_summaries)
        if any(
            isinstance(metrics.get(key), (int, float)) and not isinstance(metrics.get(key), bool)
            for key in metric_samples
        ):
            successful_replicates += 1
        for key in metric_samples:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_samples[key].append(float(value))
    intervals: Dict[str, Any] = {}
    for key, values in metric_samples.items():
        intervals[key] = {
            "confidence": 0.95,
            "lower": _percentile(values, 0.025),
            "upper": _percentile(values, 0.975),
            "sample_count": len(values),
        }
    intervals["_bootstrap"] = {
        "method": "seeded_hierarchical_cluster_percentile.v1",
        "confidence": 0.95,
        "seed": int(seed),
        "requested_replicate_count": replicate_count,
        "successful_replicate_count": successful_replicates,
        "policy_checkpoint_cluster_count": len(policy_keys),
        "criterion_split_task_family_cluster_count": len(cell_keys),
        "matched_initial_condition_clusters_preserved": True,
        "input_order_canonicalized": True,
    }
    return intervals


def _rank_fidelity_claim_eligibility(
    *,
    accepted_anchors: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    confidence_intervals: Mapping[str, Any],
    registered_split_estimands: Mapping[str, Any] | None,
    study_design: Mapping[str, Any] | None,
) -> RankFidelityClaimEligibility:
    """Return a metric-specific, public-claim gate; never a deployment gate."""

    design = _mapping(study_design)
    design_blockers: List[str] = []
    if _string(design.get("status")) != "preregistered_locked":
        design_blockers.append("study_design_not_preregistered_and_locked")
    if not _string(
        design.get("study_id")
        or design.get("registration_id")
        or design.get("registration_uri")
        or design.get("registration_hash")
    ):
        design_blockers.append("study_registration_identity_missing")
    if design.get("locked_test_data") is not True:
        design_blockers.append("locked_test_data_not_proven")
    if design.get("independent_policy_checkpoints") is not True:
        design_blockers.append("independent_policy_checkpoint_design_not_proven")
    primary_estimand = _string(design.get("primary_estimand"))
    if primary_estimand not in {
        "unit_level_micro_checkpoint_criterion_points",
        "equal_weight_registered_criterion_split_task_family_cells",
    }:
        design_blockers.append("preregistered_primary_estimand_missing_or_unsupported")
    if _string(design.get("claim_scope")) != "all_registered_splits":
        design_blockers.append("claim_scope_must_cover_all_registered_splits")

    explicit_summaries = [
        row for row in summaries if row.get("unit_of_analysis_key_explicit") is True
    ]
    if len(explicit_summaries) != len(summaries):
        design_blockers.append("unit_of_analysis_keys_not_explicit_for_all_rows")
    policy_checkpoint_keys = {
        (_string(row.get("policy_id")), _string(row.get("checkpoint_id")))
        for row in explicit_summaries
    }
    criterion_ids = {
        _string(row.get("criterion_id"))
        for row in explicit_summaries
        if _string(row.get("criterion_id"))
    }
    registered_splits = {
        _string(row.get("registered_split"))
        for row in explicit_summaries
        if _string(row.get("registered_split"))
        and _string(row.get("registered_split")) != "unregistered"
    }
    task_families = {
        _string(row.get("task_family"))
        for row in explicit_summaries
        if _string(row.get("task_family"))
    }
    if len(policy_checkpoint_keys) < MIN_POLICY_CHECKPOINT_COUNT_FOR_PUBLIC_RANK_FIDELITY:
        design_blockers.append("independent_policy_checkpoint_count_lt_7")
    if len(criterion_ids) < MIN_CRITERION_COUNT_FOR_PUBLIC_RANK_FIDELITY:
        design_blockers.append("registered_criterion_count_lt_3")
    if len(registered_splits) < MIN_REGISTERED_SPLIT_COUNT_FOR_PUBLIC_RANK_FIDELITY:
        design_blockers.append("registered_split_count_lt_2")
    if not task_families:
        design_blockers.append("registered_task_family_missing")

    declared_splits = {
        _string(value) for value in design.get("registered_splits", []) or [] if _string(value)
    }
    if not declared_splits or not registered_splits.issubset(declared_splits):
        design_blockers.append("observed_split_not_covered_by_preregistration")

    declared_min_trials = int(_number(design.get("minimum_matched_trials_per_cell"), 0.0) or 0)
    if declared_min_trials < MIN_MATCHED_TRIALS_PER_CELL_FOR_PUBLIC_RANK_FIDELITY:
        design_blockers.append("preregistered_matched_trials_per_cell_lt_20")
    cell_policy_clusters: Dict[tuple[str, str, str], Dict[tuple[str, str], set[str]]] = {}
    for row in accepted_anchors:
        cell_key = (
            _string(row.get("criterion_id")),
            _string(row.get("registered_split")),
            _string(row.get("task_family")),
        )
        policy_key = (
            _string(row.get("policy_id")),
            _string(row.get("checkpoint_id")),
        )
        cluster_id = _string(row.get("matched_initial_condition_id")) or _string(
            row.get("initial_condition_id")
        )
        if all(cell_key) and all(policy_key) and cluster_id:
            cell_policy_clusters.setdefault(cell_key, {}).setdefault(policy_key, set()).add(
                cluster_id
            )
    matched_trials_by_cell: Dict[str, int] = {}
    for cell_key, by_policy in sorted(cell_policy_clusters.items()):
        shared = set.intersection(*by_policy.values()) if by_policy else set()
        matched_trials_by_cell["|".join(cell_key)] = len(shared)
    minimum_matched_trials = min(matched_trials_by_cell.values(), default=0)
    if minimum_matched_trials < max(
        declared_min_trials,
        MIN_MATCHED_TRIALS_PER_CELL_FOR_PUBLIC_RANK_FIDELITY,
    ):
        design_blockers.append("matched_initial_condition_trials_per_cell_insufficient")

    bootstrap = _mapping(confidence_intervals.get("_bootstrap"))
    declared_bootstrap = _mapping(design.get("bootstrap"))
    declared_seed_value = _number(declared_bootstrap.get("seed"))
    declared_seed = int(declared_seed_value) if declared_seed_value is not None else -1
    declared_replicates = int(_number(declared_bootstrap.get("replicate_count"), 0.0) or 0)
    if declared_seed != int(bootstrap.get("seed") or -1):
        design_blockers.append("bootstrap_seed_not_preregistered_or_mismatched")
    if declared_replicates < DEFAULT_CALIBRATION_BOOTSTRAP_REPLICATES:
        design_blockers.append("bootstrap_replicate_count_lt_10000")
    if declared_replicates != int(bootstrap.get("requested_replicate_count") or 0):
        design_blockers.append("bootstrap_replicate_count_mismatched")

    thresholds = _mapping(design.get("claim_thresholds"))
    pearson_lower_min = _number(thresholds.get("pearson_ci_lower_min"))
    mmrv_upper_max = _number(thresholds.get("mmrv_ci_upper_max"))
    if pearson_lower_min is None or not (-1.0 <= pearson_lower_min <= 1.0):
        design_blockers.append("frozen_pearson_ci_lower_threshold_missing_or_invalid")
    if mmrv_upper_max is None or not (0.0 <= mmrv_upper_max <= 1.0):
        design_blockers.append("frozen_mmrv_ci_upper_threshold_missing_or_invalid")

    split_estimands = _mapping(_mapping(registered_split_estimands).get("splits"))
    if set(split_estimands) != registered_splits:
        design_blockers.append("registered_split_estimands_missing_or_mismatched")
    for split in sorted(registered_splits):
        split_row = _mapping(split_estimands.get(split))
        split_metrics = _mapping(split_row.get("metrics"))
        split_intervals = _mapping(split_row.get("confidence_intervals"))
        split_pearson = _number(split_metrics.get("pearson_success_rate_correlation"))
        split_pearson_lower = _number(
            _mapping(split_intervals.get("pearson_success_rate_correlation")).get("lower")
        )
        split_mmrv = _number(split_metrics.get("mmrv"))
        split_mmrv_upper = _number(_mapping(split_intervals.get("mmrv")).get("upper"))
        if split_pearson is None:
            design_blockers.append(f"registered_split_pearson_missing:{split}")
        if (
            split_pearson_lower is None
            or pearson_lower_min is None
            or split_pearson_lower < pearson_lower_min
        ):
            design_blockers.append(f"registered_split_pearson_lower_bound_failed:{split}")
        if split_mmrv is None:
            design_blockers.append(f"registered_split_mmrv_missing:{split}")
        if split_mmrv_upper is None or mmrv_upper_max is None or split_mmrv_upper > mmrv_upper_max:
            design_blockers.append(f"registered_split_mmrv_upper_bound_failed:{split}")

    pearson_blockers = list(design_blockers)
    pearson = _number(metrics.get("pearson_success_rate_correlation"))
    pearson_interval = _mapping(confidence_intervals.get("pearson_success_rate_correlation"))
    pearson_lower = _number(pearson_interval.get("lower"))
    if pearson is None or not (-1.0 <= pearson <= 1.0):
        pearson_blockers.append("pearson_point_estimate_missing_or_out_of_range")
    if pearson_lower is None:
        pearson_blockers.append("pearson_confidence_lower_bound_missing")
    elif pearson_lower_min is not None and pearson_lower < pearson_lower_min:
        pearson_blockers.append("pearson_confidence_lower_bound_below_frozen_threshold")
    if int(pearson_interval.get("sample_count") or 0) < int(0.9 * max(1, declared_replicates)):
        pearson_blockers.append("pearson_bootstrap_effective_replicates_insufficient")

    mmrv_blockers = list(design_blockers)
    mmrv = _number(metrics.get("mmrv"))
    mmrv_interval = _mapping(confidence_intervals.get("mmrv"))
    mmrv_upper = _number(mmrv_interval.get("upper"))
    if mmrv is None or not (0.0 <= mmrv <= 1.0):
        mmrv_blockers.append("mmrv_point_estimate_missing_or_out_of_range")
    if mmrv_upper is None:
        mmrv_blockers.append("mmrv_confidence_upper_bound_missing")
    elif mmrv_upper_max is not None and mmrv_upper > mmrv_upper_max:
        mmrv_blockers.append("mmrv_confidence_upper_bound_above_frozen_threshold")
    if int(mmrv_interval.get("sample_count") or 0) < int(0.9 * max(1, declared_replicates)):
        mmrv_blockers.append("mmrv_bootstrap_effective_replicates_insufficient")

    pearson_blockers = sorted(set(pearson_blockers))
    mmrv_blockers = sorted(set(mmrv_blockers))
    joint_blockers = sorted(set(pearson_blockers + mmrv_blockers))
    return {
        "schema_version": RANK_FIDELITY_CLAIM_ELIGIBILITY_SCHEMA_VERSION,
        "status": "eligible" if not joint_blockers else "ineligible",
        "scope": "external_rank_fidelity_only",
        "deployment_accuracy_claim_supported": False,
        "real_world_success_rate_prediction_claim_supported": False,
        "public_rank_fidelity_claim_eligible": not joint_blockers,
        "design": {
            "study_id": design.get("study_id") or design.get("registration_id"),
            "status": design.get("status"),
            "primary_estimand": primary_estimand or None,
            "observed_policy_checkpoint_count": len(policy_checkpoint_keys),
            "minimum_policy_checkpoint_count": (
                MIN_POLICY_CHECKPOINT_COUNT_FOR_PUBLIC_RANK_FIDELITY
            ),
            "observed_criterion_count": len(criterion_ids),
            "minimum_criterion_count": MIN_CRITERION_COUNT_FOR_PUBLIC_RANK_FIDELITY,
            "observed_registered_splits": sorted(registered_splits),
            "minimum_registered_split_count": (MIN_REGISTERED_SPLIT_COUNT_FOR_PUBLIC_RANK_FIDELITY),
            "observed_task_families": sorted(task_families),
            "minimum_matched_trials_per_cell": minimum_matched_trials,
            "required_matched_trials_per_cell": max(
                declared_min_trials,
                MIN_MATCHED_TRIALS_PER_CELL_FOR_PUBLIC_RANK_FIDELITY,
            ),
            "matched_trials_by_cell": matched_trials_by_cell,
            "blockers": sorted(set(design_blockers)),
        },
        "metrics": {
            "pearson_success_rate_correlation": {
                "eligible": not pearson_blockers,
                "point_estimate": pearson,
                "confidence_lower_bound": pearson_lower,
                "required_confidence_lower_bound": pearson_lower_min,
                "blockers": pearson_blockers,
            },
            "mean_maximum_rank_violation": {
                "eligible": not mmrv_blockers,
                "point_estimate": mmrv,
                "confidence_upper_bound": mmrv_upper,
                "required_confidence_upper_bound": mmrv_upper_max,
                "definition": "simpler_pairwise_real_success_rate_margin.v1",
                "blockers": mmrv_blockers,
            },
            "joint_rank_fidelity": {
                "eligible": not joint_blockers,
                "blockers": joint_blockers,
            },
        },
        "blockers": joint_blockers,
        "claim_boundary": (
            "Eligibility supports only the preregistered external rank-fidelity "
            "estimand; it is not task-success prediction, deployment accuracy, "
            "physical readiness, or safety proof."
        ),
    }


def _accepted_anchor_calibration(
    *,
    rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    prediction_anchor_index: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    prediction_conflict_ids: Sequence[str],
    prediction_incomplete_rows: Sequence[Mapping[str, Any]],
    study_design: Mapping[str, Any] | None = None,
) -> AcceptedAnchorCalibration:
    if not rows:
        complete_prediction_keys = sorted(
            {
                _anchor_key(row)
                for row in prediction_rows
                if not _missing_anchor_key_fields(_anchor_key(row))
                and _predicted_success(row) is not None
            }
        )
        unmatched_prediction_rows = [_anchor_key_dict(key) for key in complete_prediction_keys]
        blockers = ["insufficient_anchor_count", "insufficient_policy_group_count"]
        if unmatched_prediction_rows:
            blockers.append("unmatched_prediction_rows")
        empty_metrics = {
            "spearman_rank_correlation": None,
            "pearson_success_rate_correlation": None,
            "mean_maximum_rank_violation": None,
            "mmrv": None,
            "maximum_pairwise_real_margin_rank_violation": None,
            "mean_normalized_rank_position_error": None,
            "maximum_normalized_rank_position_error": None,
            "mean_absolute_success_rate_error": None,
            "sim_vs_real_calibration_score": None,
            "mmrv_definition": "simpler_pairwise_real_success_rate_margin.v1",
        }
        claim_eligibility = _rank_fidelity_claim_eligibility(
            accepted_anchors=[],
            summaries=[],
            metrics=empty_metrics,
            confidence_intervals={},
            registered_split_estimands={},
            study_design=study_design,
        )
        return {
            "status": "not_measured",
            "blockers": sorted(set(blockers)),
            "accepted_anchor_schema": {
                "schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
                "join_keys": list(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS),
                "unit_of_analysis_fields": list(UNIT_OF_ANALYSIS_FIELDS),
                "public_rank_fidelity_requires_explicit_unit_of_analysis_fields": True,
                "required_prediction_fields": [
                    *ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS,
                    "predicted_success",
                ],
                "required_actual_fields": [
                    *ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS,
                    "actual_success",
                    "owner_evidence_or_operator_attestation",
                ],
                "accepted_anchor_status": "accepted",
                "claim_boundary": (
                    "Accepted anchors are paired prediction/actual records. They are "
                    "inputs for external accuracy calibration, not generated-world "
                    "rank-fidelity result."
                ),
            },
            "accepted_anchor_count": 0,
            "minimum_accepted_anchor_count": MIN_ACCEPTED_ANCHOR_COUNT_FOR_CALIBRATION,
            "policy_group_count": 0,
            "minimum_policy_group_count": MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION,
            "minimum_policy_checkpoint_count_for_public_rank_fidelity": (
                MIN_POLICY_CHECKPOINT_COUNT_FOR_PUBLIC_RANK_FIDELITY
            ),
            "accepted_anchors": [],
            "rejected_anchors": [],
            "policy_success_rate_rows": [],
            "unmatched_prediction_row_count": len(unmatched_prediction_rows),
            "unmatched_prediction_rows": unmatched_prediction_rows,
            "unmatched_actual_row_count": 0,
            "unmatched_actual_row_ids": [],
            "stale_anchor_row_count": 0,
            "stale_anchor_row_ids": [],
            "conflicting_anchor_row_count": 0,
            "conflicting_anchor_row_ids": [],
            "prediction_incomplete_row_count": len(prediction_incomplete_rows),
            "prediction_incomplete_rows": list(prediction_incomplete_rows),
            "confidence_intervals": {},
            "estimands": {
                "unit_level_micro": {
                    "estimand": "checkpoint_x_criterion_x_registered_split_task_family_points",
                    "metrics": empty_metrics,
                },
                "cell_macro": _macro_calibration_estimand([]),
                "registered_splits": _registered_split_estimands(
                    [], include_confidence_intervals=False
                ),
            },
            "rank_fidelity_claim_eligibility": claim_eligibility,
            **empty_metrics,
        }
    actual_keys: Dict[tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = _anchor_key(row)
        if not _missing_anchor_key_fields(key):
            actual_keys.setdefault(key, []).append(row)
    actual_conflict_keys = {
        key
        for key, keyed_rows in actual_keys.items()
        if len(
            {
                bool(item.get("actual_success"))
                for item in keyed_rows
                if item.get("actual_success") is not None
            }
        )
        > 1
    }
    stale_anchor_row_ids: List[str] = []
    unmatched_actual_row_ids: List[str] = []
    accepted_anchors: List[Dict[str, Any]] = []
    rejected_anchors: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        key = _anchor_key(row)
        record_id = _string(row.get("record_id")) or f"deployment_outcome_{row_index:04d}"
        missing_fields = _missing_anchor_key_fields(key)
        anchor_blockers: List[str] = []
        if missing_fields:
            anchor_blockers.append("missing_anchor_join_key_fields")
        if row.get("predicted_success") is None:
            anchor_blockers.append("missing_predicted_success")
        if row.get("actual_success") is None:
            anchor_blockers.append("missing_actual_success")
        if not row.get("owner_evidence_present"):
            anchor_blockers.append("missing_owner_evidence_or_operator_attestation")
        if not row.get("signed_operator_attestation_present"):
            anchor_blockers.append("owner_or_operator_attestation_not_signed")
        if _physical_evidence_requested(row) and not row.get("physical_evidence_present"):
            anchor_blockers.append("missing_required_physical_evidence")
        if not row.get("actual_result_signal_present"):
            anchor_blockers.append("missing_actual_result_signal")
        if not _anchor_review_accepted(row):
            anchor_blockers.append("anchor_review_not_accepted")
        if _anchor_record_is_stale(row):
            anchor_blockers.append("stale_anchor_row")
            stale_anchor_row_ids.append(record_id)
        if key in actual_conflict_keys:
            anchor_blockers.append("conflicting_actual_anchor_row")
        if key not in prediction_anchor_index:
            anchor_blockers.append("unmatched_actual_row")
            unmatched_actual_row_ids.append(record_id)
        if row.get("matched_prediction") and not row.get("strict_anchor_prediction_match"):
            anchor_blockers.append("loose_or_inferred_anchor_match_rejected")
        if anchor_blockers:
            rejected_anchors.append(
                {
                    "record_id": record_id,
                    "anchor_acceptance_status": "blocked",
                    "anchor_blockers": sorted(set(anchor_blockers)),
                    "anchor_join_key": _anchor_key_dict(key),
                }
            )
            continue
        accepted_anchors.append(
            {
                **dict(row),
                "record_id": record_id,
                "anchor_schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
                "anchor_acceptance_status": "accepted",
                "anchor_join_key": _anchor_key_dict(key),
            }
        )
    accepted_keys = {_anchor_key(row) for row in accepted_anchors}
    complete_prediction_keys = {
        _anchor_key(row)
        for row in prediction_rows
        if not _missing_anchor_key_fields(_anchor_key(row)) and _predicted_success(row) is not None
    }
    unmatched_prediction_keys = sorted(complete_prediction_keys - accepted_keys)
    unmatched_prediction_rows = [_anchor_key_dict(key) for key in unmatched_prediction_keys]
    conflicting_anchor_rows = sorted(
        {
            *[
                _string(row.get("record_id")) or "|".join(_anchor_key(row))
                for key in actual_conflict_keys
                for row in actual_keys.get(key, [])
            ],
            *prediction_conflict_ids,
        }
    )
    blockers: List[str] = []
    if len(accepted_anchors) < MIN_ACCEPTED_ANCHOR_COUNT_FOR_CALIBRATION:
        blockers.append("insufficient_anchor_count")
    if unmatched_prediction_rows:
        blockers.append("unmatched_prediction_rows")
    if unmatched_actual_row_ids:
        blockers.append("unmatched_actual_rows")
    if stale_anchor_row_ids:
        blockers.append("stale_anchor_rows")
    if conflicting_anchor_rows:
        blockers.append("conflicting_anchor_rows")
    policy_summaries = _policy_anchor_summaries(accepted_anchors)
    policy_checkpoint_groups = {
        (_string(row.get("policy_id")), _string(row.get("checkpoint_id")))
        for row in policy_summaries
    }
    if len(policy_checkpoint_groups) < MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION:
        blockers.append("insufficient_policy_group_count")
    metrics: Dict[str, Any] = {
        "spearman_rank_correlation": None,
        "pearson_success_rate_correlation": None,
        "mean_maximum_rank_violation": None,
        "mmrv": None,
        "maximum_pairwise_real_margin_rank_violation": None,
        "mean_normalized_rank_position_error": None,
        "maximum_normalized_rank_position_error": None,
        "mean_absolute_success_rate_error": None,
        "sim_vs_real_calibration_score": None,
        "mmrv_definition": "simpler_pairwise_real_success_rate_margin.v1",
    }
    confidence_intervals: Dict[str, Any] = {}
    macro_estimand = _macro_calibration_estimand(policy_summaries)
    registered_split_estimands = _registered_split_estimands(
        accepted_anchors,
        include_confidence_intervals=not blockers,
    )
    if not blockers:
        metrics = _calibration_metrics_from_policy_summaries(policy_summaries)
        confidence_intervals = _bootstrap_confidence_intervals(accepted_anchors)
    claim_eligibility = _rank_fidelity_claim_eligibility(
        accepted_anchors=accepted_anchors,
        summaries=policy_summaries,
        metrics=metrics,
        confidence_intervals=confidence_intervals,
        registered_split_estimands=registered_split_estimands,
        study_design=study_design,
    )
    status = (
        "not_measured" if not rows else "completed" if not blockers else "blocked_anchor_quality"
    )
    if "insufficient_anchor_count" in blockers and set(blockers).issubset(
        {"insufficient_anchor_count", "insufficient_policy_group_count"}
    ):
        status = "blocked_insufficient_anchor_count" if rows else "not_measured"
    return {
        "status": status,
        "blockers": sorted(set(blockers)),
        "accepted_anchor_schema": {
            "schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
            "join_keys": list(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS),
            "unit_of_analysis_fields": list(UNIT_OF_ANALYSIS_FIELDS),
            "public_rank_fidelity_requires_explicit_unit_of_analysis_fields": True,
            "required_prediction_fields": [
                *ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS,
                "predicted_success",
            ],
            "required_actual_fields": [
                *ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS,
                "actual_success",
                "owner_evidence_or_operator_attestation",
            ],
            "accepted_anchor_status": "accepted",
            "claim_boundary": (
                "Accepted anchors are paired prediction/actual records. They are "
                "inputs for external accuracy calibration, not generated-world rank-fidelity result."
            ),
        },
        "accepted_anchor_count": len(accepted_anchors),
        "minimum_accepted_anchor_count": MIN_ACCEPTED_ANCHOR_COUNT_FOR_CALIBRATION,
        "policy_group_count": len(policy_checkpoint_groups),
        "policy_checkpoint_group_count": len(policy_checkpoint_groups),
        "minimum_policy_group_count": MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION,
        "minimum_policy_checkpoint_count_for_public_rank_fidelity": (
            MIN_POLICY_CHECKPOINT_COUNT_FOR_PUBLIC_RANK_FIDELITY
        ),
        "accepted_anchors": accepted_anchors,
        "rejected_anchors": rejected_anchors,
        "policy_success_rate_rows": policy_summaries,
        "unmatched_prediction_row_count": len(unmatched_prediction_rows),
        "unmatched_prediction_rows": unmatched_prediction_rows,
        "unmatched_actual_row_count": len(set(unmatched_actual_row_ids)),
        "unmatched_actual_row_ids": sorted(set(unmatched_actual_row_ids)),
        "stale_anchor_row_count": len(set(stale_anchor_row_ids)),
        "stale_anchor_row_ids": sorted(set(stale_anchor_row_ids)),
        "conflicting_anchor_row_count": len(conflicting_anchor_rows),
        "conflicting_anchor_row_ids": conflicting_anchor_rows,
        "prediction_incomplete_row_count": len(prediction_incomplete_rows),
        "prediction_incomplete_rows": list(prediction_incomplete_rows),
        "confidence_intervals": confidence_intervals,
        "unit_of_analysis_fields": list(UNIT_OF_ANALYSIS_FIELDS),
        "estimands": {
            "unit_level_micro": {
                "estimand": "checkpoint_x_criterion_x_registered_split_task_family_points",
                "metrics": dict(metrics),
            },
            "cell_macro": macro_estimand,
            "registered_splits": registered_split_estimands,
        },
        "rank_fidelity_claim_eligibility": claim_eligibility,
        **metrics,
    }


# Public typed seam for new consumers.  The underscored names remain during the
# characterization-backed migration of legacy imports.
policy_anchor_summaries = _policy_anchor_summaries
calibration_metrics_from_policy_summaries = _calibration_metrics_from_policy_summaries
evaluate_rank_fidelity_claim_eligibility = _rank_fidelity_claim_eligibility
build_accepted_anchor_calibration = _accepted_anchor_calibration

__all__ = [
    "ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS",
    "ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION",
    "AcceptedAnchorCalibration",
    "RankFidelityClaimEligibility",
    "build_accepted_anchor_calibration",
    "calibration_metrics_from_policy_summaries",
    "evaluate_rank_fidelity_claim_eligibility",
    "policy_anchor_summaries",
]
