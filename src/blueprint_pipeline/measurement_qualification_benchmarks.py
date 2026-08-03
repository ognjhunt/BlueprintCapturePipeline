"""Runnable qualification-benchmark contracts for measurement adapters.

This module turns the three research blueprints into executable, split-safe
benchmark interfaces:

* Capture-to-Geometry-and-Contact;
* Capture-to-Observation; and
* Capture-to-Deformation (cloth, cable, granular, and tactile lanes).

Candidate adapters receive public case manifests, never sealed labels.  An
independent evaluator joins predictions to physical measurements by exact
digests and computes deterministic error, mismatch, harmful-false-negative,
and coverage summaries.  A passing report is only an R5 evidence candidate;
it cannot make the R6 decision or perform R7 catalog admission.
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import fmean, stdev
from typing import Any, Mapping, Protocol, Sequence

from .measurement_adapter_runtime import (
    validate_measurement_adapter_descriptor,
)
from .measurement_method_research_catalog import (
    qualification_benchmark_blueprints,
    research_intake_catalog,
)


BENCHMARK_SPEC_SCHEMA_VERSION = "measurement_qualification_benchmark_spec.v1"
BENCHMARK_CASE_SCHEMA_VERSION = "measurement_qualification_benchmark_case.v1"
BENCHMARK_PREDICTION_SCHEMA_VERSION = "measurement_benchmark_prediction.v1"
BENCHMARK_LABEL_SCHEMA_VERSION = "sealed_physical_benchmark_label.v1"
BENCHMARK_REPORT_SCHEMA_VERSION = "measurement_qualification_benchmark_report.v1"

BENCHMARK_IDS = frozenset(
    {
        "capture-to-geometry-and-contact",
        "capture-to-observation",
        "capture-to-deformation",
        "world-model-action-fidelity",
    }
)
SPLITS = frozenset({"development", "qualification"})
DEFORMATION_LANES = frozenset({"cloth", "cable", "granular", "tactile"})


class MeasurementBenchmarkError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


class MeasurementBenchmarkAdapter(Protocol):
    adapter_reference: str

    def predict(
        self,
        *,
        benchmark_spec: Mapping[str, Any],
        case_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementBenchmarkError("measurement_benchmark_artifact_not_json") from exc
    return result


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({_string(item) for item in value if _string(item)})


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _blueprint(benchmark_id: str) -> dict[str, Any]:
    result = next(
        (
            row
            for row in qualification_benchmark_blueprints()
            if row["benchmark_id"] == benchmark_id
        ),
        None,
    )
    if result is None:
        raise MeasurementBenchmarkError(f"measurement_benchmark_id_unknown:{benchmark_id}")
    return result


def build_qualification_benchmark_spec(
    *,
    benchmark_id: str,
    benchmark_version: str,
    method_ids: Sequence[str],
    development_split_digest: str,
    qualification_split_digest: str,
    capture_bundle_digests: Sequence[str],
    robot_controller_digests: Sequence[str],
    acceptance_thresholds: Mapping[str, float],
    compute_budget: Mapping[str, Any],
    minimum_repeated_trials: int = 2,
    lane: str | None = None,
) -> dict[str, Any]:
    blueprint = _blueprint(benchmark_id)
    catalog_ids = {row["candidate_id"] for row in research_intake_catalog()}
    selected = sorted({_string(item) for item in method_ids if _string(item)})
    errors: list[str] = []
    if not _string(benchmark_version):
        errors.append("measurement_benchmark_version_missing")
    allowed = set(blueprint["methods_compared"])
    if not selected or not set(selected) <= allowed or not set(selected) <= catalog_ids:
        errors.append("measurement_benchmark_method_scope_invalid")
    if benchmark_id == "capture-to-deformation":
        if lane not in DEFORMATION_LANES:
            errors.append("measurement_benchmark_deformation_lane_invalid")
        elif not set(selected) <= set(blueprint["lanes"][lane]):
            errors.append("measurement_benchmark_deformation_method_lane_mismatch")
    elif lane is not None:
        errors.append("measurement_benchmark_lane_not_applicable")
    if development_split_digest == qualification_split_digest:
        errors.append("measurement_benchmark_split_leakage")
    for name, value in (
        ("development", development_split_digest),
        ("qualification", qualification_split_digest),
    ):
        if not _string(value).startswith("sha256:"):
            errors.append(f"measurement_benchmark_{name}_split_digest_invalid")
    captures = sorted({_string(item) for item in capture_bundle_digests if _string(item)})
    robots = sorted({_string(item) for item in robot_controller_digests if _string(item)})
    if not captures or any(not item.startswith("sha256:") for item in captures):
        errors.append("measurement_benchmark_capture_digests_invalid")
    if not robots or any(not item.startswith("sha256:") for item in robots):
        errors.append("measurement_benchmark_robot_controller_digests_invalid")
    thresholds = dict(acceptance_thresholds)
    required_thresholds = {
        "maximum_mean_absolute_error",
        "maximum_mismatch_rate",
        "maximum_harmful_false_negative_rate",
        "minimum_coverage",
    }
    if set(thresholds) != required_thresholds or any(
        _number(value) is None or float(value) < 0 for value in thresholds.values()
    ):
        errors.append("measurement_benchmark_acceptance_thresholds_invalid")
    elif float(thresholds["minimum_coverage"]) > 1:
        errors.append("measurement_benchmark_minimum_coverage_invalid")
    if not isinstance(compute_budget, Mapping) or _number(compute_budget.get("usd")) is None:
        errors.append("measurement_benchmark_compute_budget_invalid")
    if (
        isinstance(minimum_repeated_trials, bool)
        or not isinstance(minimum_repeated_trials, int)
        or minimum_repeated_trials < 2
    ):
        errors.append("measurement_benchmark_minimum_repeated_trials_invalid")
    if errors:
        raise MeasurementBenchmarkError(*errors)
    spec = {
        "schema_version": BENCHMARK_SPEC_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "lane": lane,
        "purpose": blueprint["purpose"],
        "protocols": blueprint["protocols"],
        "physical_setup_requirements": blueprint["physical_setup"],
        "method_ids": selected,
        "metric_ids": blueprint["metrics"],
        "development_split_digest": development_split_digest,
        "qualification_split_digest": qualification_split_digest,
        "capture_bundle_digests": captures,
        "robot_controller_digests": robots,
        "acceptance_thresholds": thresholds,
        "compute_budget": dict(compute_budget),
        "minimum_repeated_trials": minimum_repeated_trials,
        "candidate_may_access_qualification_labels": False,
        "vendor_may_grade_qualification": False,
        "agent_may_approve": False,
        "r6_human_decision_required": True,
        "r7_catalog_admission_required": True,
    }
    spec["benchmark_spec_digest"] = _digest(spec, "benchmark_spec_digest")
    return validate_qualification_benchmark_spec(spec)


def validate_qualification_benchmark_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = _clone(value)
    errors: list[str] = []
    if spec.get("schema_version") != BENCHMARK_SPEC_SCHEMA_VERSION:
        errors.append("measurement_benchmark_spec_schema_invalid")
    if spec.get("benchmark_id") not in BENCHMARK_IDS:
        errors.append("measurement_benchmark_spec_id_invalid")
    for key in (
        "method_ids",
        "protocols",
        "physical_setup_requirements",
        "metric_ids",
        "capture_bundle_digests",
        "robot_controller_digests",
    ):
        if not isinstance(spec.get(key), list) or not spec.get(key):
            errors.append(f"measurement_benchmark_spec_{key}_invalid")
    if spec.get("benchmark_id") == "capture-to-deformation":
        if spec.get("lane") not in DEFORMATION_LANES:
            errors.append("measurement_benchmark_spec_lane_invalid")
    elif spec.get("lane") is not None:
        errors.append("measurement_benchmark_spec_lane_not_applicable")
    if spec.get("development_split_digest") == spec.get("qualification_split_digest"):
        errors.append("measurement_benchmark_split_leakage")
    for key in (
        "candidate_may_access_qualification_labels",
        "vendor_may_grade_qualification",
        "agent_may_approve",
    ):
        if spec.get(key) is not False:
            errors.append(f"measurement_benchmark_spec_{key}_must_be_false")
    if spec.get("r6_human_decision_required") is not True:
        errors.append("measurement_benchmark_spec_r6_human_decision_required")
    if spec.get("r7_catalog_admission_required") is not True:
        errors.append("measurement_benchmark_spec_r7_catalog_admission_required")
    if (
        isinstance(spec.get("minimum_repeated_trials"), bool)
        or not isinstance(spec.get("minimum_repeated_trials"), int)
        or spec["minimum_repeated_trials"] < 2
    ):
        errors.append("measurement_benchmark_minimum_repeated_trials_invalid")
    expected = _digest(spec, "benchmark_spec_digest")
    supplied = spec.get("benchmark_spec_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_benchmark_spec_digest_mismatch")
    if errors:
        raise MeasurementBenchmarkError(*errors)
    spec["benchmark_spec_digest"] = expected
    return spec


def build_benchmark_case_manifest(
    spec_value: Mapping[str, Any],
    *,
    case_id: str,
    split: str,
    input_artifact_digests: Sequence[str],
    task_class: str,
    material_regime: str,
    operating_point: Mapping[str, Any],
) -> dict[str, Any]:
    spec = validate_qualification_benchmark_spec(spec_value)
    if split not in SPLITS:
        raise MeasurementBenchmarkError("measurement_benchmark_case_split_invalid")
    inputs = sorted({_string(item) for item in input_artifact_digests if _string(item)})
    if not _string(case_id) or not inputs or any(not item.startswith("sha256:") for item in inputs):
        raise MeasurementBenchmarkError("measurement_benchmark_case_identity_invalid")
    case = {
        "schema_version": BENCHMARK_CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "benchmark_id": spec["benchmark_id"],
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "split": split,
        "split_digest": spec[f"{split}_split_digest"],
        "lane": spec.get("lane"),
        "task_class": task_class,
        "material_regime": material_regime,
        "input_artifact_digests": inputs,
        "operating_point": dict(operating_point),
        "requested_metric_ids": spec["metric_ids"],
        "sealed_labels_included": False,
        "physical_measurement_values_included": False,
    }
    case["case_manifest_digest"] = _digest(case, "case_manifest_digest")
    return validate_benchmark_case_manifest(case)


def validate_benchmark_case_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    case = _clone(value)
    errors: list[str] = []
    if case.get("schema_version") != BENCHMARK_CASE_SCHEMA_VERSION:
        errors.append("measurement_benchmark_case_schema_invalid")
    if case.get("benchmark_id") not in BENCHMARK_IDS or case.get("split") not in SPLITS:
        errors.append("measurement_benchmark_case_scope_invalid")
    for key in ("case_id", "benchmark_spec_digest", "split_digest", "task_class"):
        if not _string(case.get(key)):
            errors.append(f"measurement_benchmark_case_{key}_missing")
    if not isinstance(case.get("requested_metric_ids"), list):
        errors.append("measurement_benchmark_case_metrics_invalid")
    if case.get("sealed_labels_included") is not False:
        errors.append("measurement_benchmark_case_sealed_label_leakage")
    if case.get("physical_measurement_values_included") is not False:
        errors.append("measurement_benchmark_case_physical_value_leakage")
    expected = _digest(case, "case_manifest_digest")
    supplied = case.get("case_manifest_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_benchmark_case_digest_mismatch")
    if errors:
        raise MeasurementBenchmarkError(*errors)
    case["case_manifest_digest"] = expected
    return case


def build_benchmark_prediction(
    descriptor_value: Mapping[str, Any],
    case_value: Mapping[str, Any],
    *,
    observed_metrics: Mapping[str, Any],
    unsafe_condition_predicted: bool | None,
    execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    # Local import avoids a module cycle: the execution runner uses this
    # builder after it has produced and validated the receipt.
    from .measurement_adapter_execution import (
        validate_measurement_adapter_execution_receipt,
    )

    descriptor = validate_measurement_adapter_descriptor(descriptor_value)
    case = validate_benchmark_case_manifest(case_value)
    receipt = validate_measurement_adapter_execution_receipt(execution_receipt)
    metrics = dict(observed_metrics)
    binding_errors: list[str] = []
    for actual, expected, name in (
        (receipt["candidate_id"], descriptor["candidate_id"], "candidate"),
        (
            receipt["adapter_descriptor_digest"],
            descriptor["adapter_descriptor_digest"],
            "adapter_descriptor",
        ),
        (
            receipt["benchmark_spec_digest"],
            case["benchmark_spec_digest"],
            "benchmark_spec",
        ),
        (
            receipt["case_manifest_digest"],
            case["case_manifest_digest"],
            "case_manifest",
        ),
        (receipt["split"], case["split"], "split"),
    ):
        if actual != expected:
            binding_errors.append(f"measurement_benchmark_execution_binding_mismatch:{name}")
    if receipt["status"] != "completed":
        binding_errors.append("measurement_benchmark_execution_not_completed")
    if binding_errors:
        raise MeasurementBenchmarkError(*binding_errors)
    if not set(metrics) <= set(case["requested_metric_ids"]):
        raise MeasurementBenchmarkError("measurement_benchmark_prediction_metric_unknown")
    prediction = {
        "schema_version": BENCHMARK_PREDICTION_SCHEMA_VERSION,
        "prediction_id": f"prediction-{descriptor['candidate_id']}-{case['case_id']}",
        "benchmark_id": case["benchmark_id"],
        "benchmark_spec_digest": case["benchmark_spec_digest"],
        "case_id": case["case_id"],
        "case_manifest_digest": case["case_manifest_digest"],
        "split": case["split"],
        "candidate_id": descriptor["candidate_id"],
        "adapter_descriptor_digest": descriptor["adapter_descriptor_digest"],
        "execution_receipt_digest": receipt["execution_receipt_digest"],
        "execution_evidence_class": receipt["evidence_class"],
        "executor_id": receipt["executor_id"],
        "executor_independent_of_candidate": receipt["executor_independent_of_candidate"],
        "clean_environment_verified": receipt["clean_environment_verified"],
        "immutable_runtime_identity_verified": receipt["immutable_runtime_identity_verified"],
        "observed_metrics": metrics,
        "unsafe_condition_predicted": unsafe_condition_predicted,
        "qualification_labels_accessed": False,
        "vendor_graded": False,
        "agent_graded": False,
        "physical_success_established": False,
    }
    prediction["prediction_digest"] = _digest(prediction, "prediction_digest")
    return validate_benchmark_prediction(prediction)


def validate_benchmark_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    prediction = _clone(value)
    errors: list[str] = []
    if prediction.get("schema_version") != BENCHMARK_PREDICTION_SCHEMA_VERSION:
        errors.append("measurement_benchmark_prediction_schema_invalid")
    if not isinstance(prediction.get("observed_metrics"), Mapping):
        errors.append("measurement_benchmark_prediction_metrics_invalid")
    if prediction.get("execution_evidence_class") not in {
        "development_execution",
        "independent_qualification_execution",
    }:
        errors.append("measurement_benchmark_prediction_execution_class_invalid")
    if not _string(prediction.get("executor_id")):
        errors.append("measurement_benchmark_prediction_executor_missing")
    for key in (
        "executor_independent_of_candidate",
        "clean_environment_verified",
        "immutable_runtime_identity_verified",
    ):
        if prediction.get(key) not in {True, False}:
            errors.append(f"measurement_benchmark_prediction_{key}_invalid")
    if prediction.get("unsafe_condition_predicted") not in {True, False, None}:
        errors.append("measurement_benchmark_prediction_unsafe_value_invalid")
    for key in (
        "qualification_labels_accessed",
        "vendor_graded",
        "agent_graded",
        "physical_success_established",
    ):
        if prediction.get(key) is not False:
            errors.append(f"measurement_benchmark_prediction_{key}_must_be_false")
    expected = _digest(prediction, "prediction_digest")
    supplied = prediction.get("prediction_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_benchmark_prediction_digest_mismatch")
    if errors:
        raise MeasurementBenchmarkError(*errors)
    prediction["prediction_digest"] = expected
    return prediction


def build_sealed_physical_label(
    case_value: Mapping[str, Any],
    *,
    expected_metrics: Mapping[str, Any],
    unsafe_condition_observed: bool,
    physical_measurement_ids: Sequence[str],
    independent_evaluator_id: str,
) -> dict[str, Any]:
    case = validate_benchmark_case_manifest(case_value)
    measurements = sorted({_string(item) for item in physical_measurement_ids if _string(item)})
    if not measurements or not _string(independent_evaluator_id):
        raise MeasurementBenchmarkError("sealed_benchmark_label_authority_missing")
    metrics = dict(expected_metrics)
    if not set(metrics) <= set(case["requested_metric_ids"]):
        raise MeasurementBenchmarkError("sealed_benchmark_label_metric_unknown")
    label = {
        "schema_version": BENCHMARK_LABEL_SCHEMA_VERSION,
        "label_id": f"label-{case['case_id']}",
        "benchmark_id": case["benchmark_id"],
        "benchmark_spec_digest": case["benchmark_spec_digest"],
        "case_id": case["case_id"],
        "case_manifest_digest": case["case_manifest_digest"],
        "split": case["split"],
        "expected_metrics": metrics,
        "unsafe_condition_observed": unsafe_condition_observed,
        "physical_measurement_ids": measurements,
        "independent_evaluator_id": independent_evaluator_id,
        "label_visibility": "independent_evaluator_only",
        "candidate_accessed": False,
        "vendor_graded": False,
        "agent_graded": False,
    }
    label["sealed_label_digest"] = _digest(label, "sealed_label_digest")
    return validate_sealed_physical_label(label)


def validate_sealed_physical_label(value: Mapping[str, Any]) -> dict[str, Any]:
    label = _clone(value)
    errors: list[str] = []
    if label.get("schema_version") != BENCHMARK_LABEL_SCHEMA_VERSION:
        errors.append("sealed_benchmark_label_schema_invalid")
    if not isinstance(label.get("expected_metrics"), Mapping):
        errors.append("sealed_benchmark_label_metrics_invalid")
    if label.get("unsafe_condition_observed") not in {True, False}:
        errors.append("sealed_benchmark_label_unsafe_value_invalid")
    if not _strings(label.get("physical_measurement_ids")):
        errors.append("sealed_benchmark_label_physical_measurements_missing")
    if label.get("label_visibility") != "independent_evaluator_only":
        errors.append("sealed_benchmark_label_visibility_invalid")
    for key in ("candidate_accessed", "vendor_graded", "agent_graded"):
        if label.get(key) is not False:
            errors.append(f"sealed_benchmark_label_{key}_must_be_false")
    expected = _digest(label, "sealed_label_digest")
    supplied = label.get("sealed_label_digest")
    if supplied is not None and supplied != expected:
        errors.append("sealed_benchmark_label_digest_mismatch")
    if errors:
        raise MeasurementBenchmarkError(*errors)
    label["sealed_label_digest"] = expected
    return label


def _metric_error(observed: Any, expected: Any) -> tuple[str, float] | None:
    observed_number = _number(observed)
    expected_number = _number(expected)
    if observed_number is not None and expected_number is not None:
        return "absolute_error", abs(observed_number - expected_number)
    if isinstance(observed, (str, bool)) and isinstance(expected, type(observed)):
        return "mismatch", 0.0 if observed == expected else 1.0
    return None


def _wilson_interval(successes: float, total: int) -> dict[str, float | int]:
    """Deterministic 95% Wilson interval for a bounded rate."""

    if total <= 0:
        return {"lower": 0.0, "upper": 1.0, "level": 0.95, "sample_size": 0}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + (z * z / total)
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z * math.sqrt((rate * (1.0 - rate) / total) + (z * z / (4.0 * total * total))) / denominator
    )
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "level": 0.95,
        "sample_size": total,
    }


def _mean_interval(values: Sequence[float]) -> dict[str, float | int]:
    """Deterministic normal-approximation 95% interval for a mean."""

    if not values:
        return {"lower": 0.0, "upper": 0.0, "level": 0.95, "sample_size": 0}
    mean = fmean(values)
    margin = 0.0
    if len(values) > 1:
        margin = 1.959963984540054 * stdev(values) / math.sqrt(len(values))
    return {
        "lower": max(0.0, mean - margin),
        "upper": mean + margin,
        "level": 0.95,
        "sample_size": len(values),
    }


def evaluate_qualification_benchmark(
    spec_value: Mapping[str, Any],
    prediction_values: Sequence[Mapping[str, Any]],
    sealed_label_values: Sequence[Mapping[str, Any]],
    *,
    evaluator_id: str,
    independent_execution: bool,
) -> dict[str, Any]:
    spec = validate_qualification_benchmark_spec(spec_value)
    predictions = [validate_benchmark_prediction(row) for row in prediction_values]
    labels = [validate_sealed_physical_label(row) for row in sealed_label_values]
    if not _string(evaluator_id):
        raise MeasurementBenchmarkError("measurement_benchmark_evaluator_missing")
    prediction_by_case = {row["case_id"]: row for row in predictions}
    label_by_case = {row["case_id"]: row for row in labels}
    if len(prediction_by_case) != len(predictions) or len(label_by_case) != len(labels):
        raise MeasurementBenchmarkError("measurement_benchmark_duplicate_case")
    if set(prediction_by_case) != set(label_by_case):
        raise MeasurementBenchmarkError("measurement_benchmark_case_join_incomplete")
    split_values = {row["split"] for row in predictions + labels}
    if len(split_values) != 1:
        raise MeasurementBenchmarkError("measurement_benchmark_split_mismatch")
    split = split_values.pop()
    if any(row["independent_evaluator_id"] != evaluator_id for row in labels):
        raise MeasurementBenchmarkError("measurement_benchmark_evaluator_binding_mismatch")
    candidate_ids = {row["candidate_id"] for row in predictions}
    if evaluator_id in candidate_ids:
        raise MeasurementBenchmarkError("measurement_benchmark_candidate_self_grading")
    independent_receipts = all(
        row["execution_evidence_class"] == "independent_qualification_execution"
        and row["executor_independent_of_candidate"] is True
        and row["clean_environment_verified"] is True
        and row["immutable_runtime_identity_verified"] is True
        and row["executor_id"] not in candidate_ids
        for row in predictions
    )
    if independent_execution is True and not independent_receipts:
        raise MeasurementBenchmarkError("measurement_benchmark_independent_execution_proof_missing")
    expected_split_digest = spec[f"{split}_split_digest"]
    errors_by_metric: dict[str, list[float]] = {}
    mismatch_by_metric: dict[str, list[float]] = {}
    harmful_false_negatives = 0
    unsafe_cases = 0
    evaluated_metric_values = 0
    possible_metric_values = len(predictions) * len(spec["metric_ids"])
    joined_rows: list[dict[str, Any]] = []
    for case_id in sorted(prediction_by_case):
        prediction = prediction_by_case[case_id]
        label = label_by_case[case_id]
        if any(
            (
                prediction["benchmark_spec_digest"] != spec["benchmark_spec_digest"],
                label["benchmark_spec_digest"] != spec["benchmark_spec_digest"],
                prediction["case_manifest_digest"] != label["case_manifest_digest"],
            )
        ):
            raise MeasurementBenchmarkError(
                f"measurement_benchmark_case_binding_mismatch:{case_id}"
            )
        observed = dict(prediction["observed_metrics"])
        expected = dict(label["expected_metrics"])
        case_errors: dict[str, dict[str, Any]] = {}
        for metric_id in sorted(set(observed) & set(expected)):
            result = _metric_error(observed[metric_id], expected[metric_id])
            if result is None:
                continue
            kind, error = result
            target = errors_by_metric if kind == "absolute_error" else mismatch_by_metric
            target.setdefault(metric_id, []).append(error)
            case_errors[metric_id] = {"error_kind": kind, "error": error}
            evaluated_metric_values += 1
        if label["unsafe_condition_observed"] is True:
            unsafe_cases += 1
            if prediction["unsafe_condition_predicted"] is not True:
                harmful_false_negatives += 1
        joined_rows.append(
            {
                "case_id": case_id,
                "prediction_digest": prediction["prediction_digest"],
                "sealed_label_digest": label["sealed_label_digest"],
                "physical_measurement_ids": label["physical_measurement_ids"],
                "metric_errors": case_errors,
            }
        )
    absolute_errors = [value for values in errors_by_metric.values() for value in values]
    mismatches = [value for values in mismatch_by_metric.values() for value in values]
    mean_absolute_error = fmean(absolute_errors) if absolute_errors else 0.0
    mismatch_rate = fmean(mismatches) if mismatches else 0.0
    harmful_rate = harmful_false_negatives / unsafe_cases if unsafe_cases else 0.0
    coverage = evaluated_metric_values / possible_metric_values if possible_metric_values else 0.0
    thresholds = dict(spec["acceptance_thresholds"])
    threshold_checks = {
        "mean_absolute_error": mean_absolute_error
        <= float(thresholds["maximum_mean_absolute_error"]),
        "mismatch_rate": mismatch_rate <= float(thresholds["maximum_mismatch_rate"]),
        "harmful_false_negative_rate": harmful_rate
        <= float(thresholds["maximum_harmful_false_negative_rate"]),
        "coverage": coverage >= float(thresholds["minimum_coverage"]),
        "repeated_trials": len(joined_rows) >= spec["minimum_repeated_trials"],
    }
    passed = all(threshold_checks.values())
    qualification_split = split == "qualification"
    r5_candidate = (
        qualification_split
        and independent_execution is True
        and independent_receipts
        and len(joined_rows) >= spec["minimum_repeated_trials"]
    )
    confidence_intervals = {
        "mean_absolute_error": _mean_interval(absolute_errors),
        "mismatch_rate": _wilson_interval(sum(mismatches), len(mismatches)),
        "harmful_false_negative_rate": _wilson_interval(harmful_false_negatives, unsafe_cases),
        "coverage": _wilson_interval(evaluated_metric_values, possible_metric_values),
    }
    report = {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "benchmark_id": spec["benchmark_id"],
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "split": split,
        "split_digest": expected_split_digest,
        "evaluator_id": evaluator_id,
        "independent_execution": independent_execution is True,
        "independent_execution_receipts_verified": independent_receipts,
        "case_count": len(joined_rows),
        "minimum_repeated_trials": spec["minimum_repeated_trials"],
        "joined_cases": joined_rows,
        "metrics": {
            "mean_absolute_error": mean_absolute_error,
            "mismatch_rate": mismatch_rate,
            "harmful_false_negative_rate": harmful_rate,
            "coverage": coverage,
            "per_metric_mean_absolute_error": {
                key: fmean(values) for key, values in sorted(errors_by_metric.items())
            },
            "per_metric_mismatch_rate": {
                key: fmean(values) for key, values in sorted(mismatch_by_metric.items())
            },
        },
        "threshold_checks": threshold_checks,
        "confidence_intervals": confidence_intervals,
        "thresholds_passed": passed,
        "evidence_status": (
            "r5_evidence_candidate" if r5_candidate else "development_only_not_qualification"
        ),
        "r6_qualification_decision": None,
        "r7_catalog_admission": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "deployment_readiness_established": False,
        "safety_certification_established": False,
    }
    report["benchmark_report_digest"] = _digest(report, "benchmark_report_digest")
    return validate_qualification_benchmark_report(report)


def validate_qualification_benchmark_report(value: Mapping[str, Any]) -> dict[str, Any]:
    report = _clone(value)
    errors: list[str] = []
    if report.get("schema_version") != BENCHMARK_REPORT_SCHEMA_VERSION:
        errors.append("measurement_benchmark_report_schema_invalid")
    if report.get("evidence_status") not in {
        "development_only_not_qualification",
        "r5_evidence_candidate",
    }:
        errors.append("measurement_benchmark_report_evidence_status_invalid")
    if not isinstance(report.get("confidence_intervals"), Mapping):
        errors.append("measurement_benchmark_report_confidence_intervals_invalid")
    if report.get("independent_execution_receipts_verified") not in {True, False}:
        errors.append("measurement_benchmark_report_execution_receipts_invalid")
    if report.get("evidence_status") == "r5_evidence_candidate" and (
        report.get("split") != "qualification"
        or report.get("independent_execution") is not True
        or report.get("independent_execution_receipts_verified") is not True
        or report.get("case_count", 0) < report.get("minimum_repeated_trials", 2)
    ):
        errors.append("measurement_benchmark_report_r5_boundary_invalid")
    if report.get("r6_qualification_decision") is not None:
        errors.append("measurement_benchmark_report_r6_decision_forbidden")
    for key in (
        "r7_catalog_admission",
        "production_route_eligible",
        "physical_success_established",
        "deployment_readiness_established",
        "safety_certification_established",
    ):
        if report.get(key) is not False:
            errors.append(f"measurement_benchmark_report_{key}_must_be_false")
    expected = _digest(report, "benchmark_report_digest")
    supplied = report.get("benchmark_report_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_benchmark_report_digest_mismatch")
    if errors:
        raise MeasurementBenchmarkError(*errors)
    report["benchmark_report_digest"] = expected
    return report


def build_r4_preregistration_stage_data(
    spec_value: Mapping[str, Any],
) -> dict[str, Any]:
    spec = validate_qualification_benchmark_spec(spec_value)
    return {
        "frozen_benchmark_preregistration": {
            "task_site_classes": [spec["benchmark_id"]],
            "development_split_hash": spec["development_split_digest"],
            "qualification_split_hash": spec["qualification_split_digest"],
            "robot_controller_digests": spec["robot_controller_digests"],
            "capture_bundle_hashes": spec["capture_bundle_digests"],
            "metrics": spec["metric_ids"],
            "acceptance_thresholds": spec["acceptance_thresholds"],
            "minimum_repeated_trials": spec["minimum_repeated_trials"],
            "comparison_methods": spec["method_ids"],
            "compute_budget": spec["compute_budget"],
            "failure_criteria": [
                "threshold_exceeded",
                "case_join_incomplete",
                "hidden_label_leakage",
                "vendor_self_grading",
            ],
            "statistical_method": "deterministic_case_join_with_heldout_summary",
            "claim_ceiling": "C4",
            "benchmark_spec_digest": spec["benchmark_spec_digest"],
        },
        "heldout_labels_exposed": False,
    }


def build_r5_stage_data(report_value: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_qualification_benchmark_report(report_value)
    if report["evidence_status"] != "r5_evidence_candidate":
        raise MeasurementBenchmarkError("measurement_benchmark_report_not_r5_candidate")
    physical_ids = sorted(
        {item for row in report["joined_cases"] for item in row["physical_measurement_ids"]}
    )
    return {
        "heldout_evaluation": {
            "independent_execution": True,
            "hidden_case_hashes": [row["sealed_label_digest"] for row in report["joined_cases"]],
            "physical_measurement_ids": physical_ids,
            "repeated_trial_count": report["case_count"],
            "confidence_intervals": report["confidence_intervals"],
            "harmful_false_negative_analysis": {
                "rate": report["metrics"]["harmful_false_negative_rate"],
            },
            "retained_failure_ids": [
                row["case_id"]
                for row in report["joined_cases"]
                if any(metric["error"] > 0 for metric in row["metric_errors"].values())
            ],
            "clean_environment_rerun_id": report["benchmark_report_digest"],
            "qualification_split_hash": report["split_digest"],
            "benchmark_report_digest": report["benchmark_report_digest"],
        },
        "vendor_graded_qualification": False,
    }


__all__ = [
    "BENCHMARK_CASE_SCHEMA_VERSION",
    "BENCHMARK_IDS",
    "BENCHMARK_LABEL_SCHEMA_VERSION",
    "BENCHMARK_PREDICTION_SCHEMA_VERSION",
    "BENCHMARK_REPORT_SCHEMA_VERSION",
    "BENCHMARK_SPEC_SCHEMA_VERSION",
    "DEFORMATION_LANES",
    "MeasurementBenchmarkAdapter",
    "MeasurementBenchmarkError",
    "build_benchmark_case_manifest",
    "build_benchmark_prediction",
    "build_qualification_benchmark_spec",
    "build_r4_preregistration_stage_data",
    "build_r5_stage_data",
    "build_sealed_physical_label",
    "evaluate_qualification_benchmark",
    "validate_benchmark_case_manifest",
    "validate_benchmark_prediction",
    "validate_qualification_benchmark_report",
    "validate_qualification_benchmark_spec",
    "validate_sealed_physical_label",
]
