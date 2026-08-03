"""Task/site-specific observation challenge qualification contracts.

The generic measurement benchmark computes errors against sealed physical
labels.  This module adds the Capture-to-Observation coverage contract that the
generic benchmark intentionally cannot infer: exact task/site/sensor binding,
transparent and reflective materials, difficult visibility conditions,
controlled/natural/adverse lighting, paired physical/synthetic sensor outputs,
and repeated real/synthetic policy outcomes for an exact checkpoint cohort.

Completing this matrix can support an R5 evidence candidate only when joined to
an independently executed held-out observation benchmark report.  It never
creates a Q-SENSOR decision, R6 approval, R7 admission, or physical-success
claim.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .measurement_qualification_benchmarks import (
    validate_qualification_benchmark_report,
    validate_qualification_benchmark_spec,
)
from .measurement_sensor_stream_pairing import (
    SUPPORTED_MODALITIES,
    validate_sensor_stream_pairing_record,
)


SCOPE_SCHEMA_VERSION = "measurement_observation_qualification_scope.v1"
CASE_SCHEMA_VERSION = "measurement_observation_challenge_case.v1"
REPORT_SCHEMA_VERSION = "measurement_observation_challenge_report.v1"

MATERIAL_CHALLENGES = frozenset(
    {
        "opaque_control",
        "transparent",
        "reflective",
        "dark",
        "small",
        "thin",
        "occluded",
    }
)
LIGHTING_CHALLENGES = frozenset({"controlled", "natural", "adverse"})
PHYSICAL_REFERENCE_METHOD_ID = "direct-captured-observations"


class MeasurementObservationQualificationError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementObservationQualificationError(
            "observation_qualification_artifact_not_json"
        ) from exc


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _valid_digest(value: Any) -> bool:
    text = _string(value)
    return bool(
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(_string(item) for item in value if _string(item)))


def _digest_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {_string(key): _string(item) for key, item in value.items() if _string(key)}


def _governance_errors(value: Mapping[str, Any], *, prefix: str) -> list[str]:
    errors: list[str] = []
    for key in (
        "visual_similarity_is_primary",
        "q_sensor_qualification_created",
        "r6_decision_created",
        "r7_admission_created",
        "production_route_eligible",
        "physical_success_established",
        "agent_may_approve",
    ):
        if value.get(key) is not False:
            errors.append(f"{prefix}_{key}_must_be_false")
    return errors


def build_observation_qualification_scope(
    benchmark_spec_value: Mapping[str, Any],
    sensor_pairing_value: Mapping[str, Any],
    *,
    task_id: str,
    site_id: str,
    task_request_digest: str,
    site_evidence_profile_digest: str,
    policy_checkpoints: Mapping[str, str],
    required_material_challenges: Sequence[str] = tuple(sorted(MATERIAL_CHALLENGES)),
    required_lighting_challenges: Sequence[str] = tuple(sorted(LIGHTING_CHALLENGES)),
    minimum_repeated_trials_per_policy: int = 2,
) -> dict[str, Any]:
    spec = validate_qualification_benchmark_spec(benchmark_spec_value)
    pairing = validate_sensor_stream_pairing_record(sensor_pairing_value)
    errors: list[str] = []
    if spec["benchmark_id"] != "capture-to-observation":
        errors.append("observation_qualification_benchmark_scope_invalid")
    if pairing["decision"] != "accepted":
        errors.append("observation_qualification_sensor_pairing_not_accepted")
    if pairing["source_capture_digest"] not in spec["capture_bundle_digests"]:
        errors.append("observation_qualification_capture_pairing_mismatch")
    if not _string(task_id) or not _string(site_id):
        errors.append("observation_qualification_task_site_identity_missing")
    for name, digest in (
        ("task_request", task_request_digest),
        ("site_evidence_profile", site_evidence_profile_digest),
    ):
        if not _valid_digest(digest):
            errors.append(f"observation_qualification_{name}_digest_invalid")
    policies = _digest_map(policy_checkpoints)
    if len(policies) < 3 or any(not _valid_digest(item) for item in policies.values()):
        errors.append("observation_qualification_policy_cohort_invalid")
    materials = _strings(list(required_material_challenges))
    lighting = _strings(list(required_lighting_challenges))
    if len(materials) != len(required_material_challenges) or set(materials) != MATERIAL_CHALLENGES:
        errors.append("observation_qualification_material_challenges_invalid")
    if len(lighting) != len(required_lighting_challenges) or set(lighting) != LIGHTING_CHALLENGES:
        errors.append("observation_qualification_lighting_challenges_invalid")
    if (
        isinstance(minimum_repeated_trials_per_policy, bool)
        or not isinstance(minimum_repeated_trials_per_policy, int)
        or minimum_repeated_trials_per_policy < 2
    ):
        errors.append("observation_qualification_repeated_trials_invalid")
    synthetic_methods = sorted(set(spec["method_ids"]) - {PHYSICAL_REFERENCE_METHOD_ID})
    if not synthetic_methods:
        errors.append("observation_qualification_synthetic_method_missing")
    if errors:
        raise MeasurementObservationQualificationError(*errors)
    scope = {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "scope_id": f"observation:{site_id}:{task_id}:{spec['benchmark_version']}",
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "task_id": task_id,
        "site_id": site_id,
        "task_request_digest": task_request_digest,
        "site_evidence_profile_digest": site_evidence_profile_digest,
        "source_capture_digest": pairing["source_capture_digest"],
        "sensor_pairing_digest": pairing["pairing_digest"],
        "required_modalities": list(pairing["required_modalities"]),
        "synthetic_method_ids": synthetic_methods,
        "policy_checkpoints": dict(sorted(policies.items())),
        "required_material_challenges": sorted(materials),
        "required_lighting_challenges": sorted(lighting),
        "required_metric_ids": list(spec["metric_ids"]),
        "minimum_repeated_trials_per_policy": minimum_repeated_trials_per_policy,
        "paired_physical_synthetic_required": True,
        "visual_similarity_is_primary": False,
        "downstream_task_validity_is_primary": True,
        "q_sensor_qualification_created": False,
        "r6_decision_created": False,
        "r7_admission_created": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "agent_may_approve": False,
    }
    scope["scope_digest"] = canonical_digest(scope, digest_field="scope_digest")
    return validate_observation_qualification_scope(scope)


def validate_observation_qualification_scope(value: Mapping[str, Any]) -> dict[str, Any]:
    scope = _clone(value)
    errors: list[str] = []
    if scope.get("schema_version") != SCOPE_SCHEMA_VERSION:
        errors.append("observation_qualification_scope_schema_invalid")
    for key in ("scope_id", "task_id", "site_id"):
        if not _string(scope.get(key)):
            errors.append(f"observation_qualification_scope_{key}_missing")
    for key in (
        "benchmark_spec_digest",
        "task_request_digest",
        "site_evidence_profile_digest",
        "source_capture_digest",
        "sensor_pairing_digest",
    ):
        if not _valid_digest(scope.get(key)):
            errors.append(f"observation_qualification_scope_{key}_invalid")
    modalities = _strings(scope.get("required_modalities"))
    if (
        not modalities
        or len(modalities) != len(scope.get("required_modalities") or [])
        or not set(modalities) <= SUPPORTED_MODALITIES
    ):
        errors.append("observation_qualification_scope_modalities_invalid")
    materials = _strings(scope.get("required_material_challenges"))
    lighting = _strings(scope.get("required_lighting_challenges"))
    if (
        len(materials) != len(scope.get("required_material_challenges") or [])
        or set(materials) != MATERIAL_CHALLENGES
    ):
        errors.append("observation_qualification_scope_materials_invalid")
    if (
        len(lighting) != len(scope.get("required_lighting_challenges") or [])
        or set(lighting) != LIGHTING_CHALLENGES
    ):
        errors.append("observation_qualification_scope_lighting_invalid")
    methods = _strings(scope.get("synthetic_method_ids"))
    policies = _digest_map(scope.get("policy_checkpoints"))
    if not methods or len(methods) != len(scope.get("synthetic_method_ids") or []):
        errors.append("observation_qualification_scope_methods_invalid")
    if len(policies) < 3 or any(not _valid_digest(item) for item in policies.values()):
        errors.append("observation_qualification_scope_policies_invalid")
    metrics = _strings(scope.get("required_metric_ids"))
    if not metrics or len(metrics) != len(scope.get("required_metric_ids") or []):
        errors.append("observation_qualification_scope_metrics_invalid")
    minimum = scope.get("minimum_repeated_trials_per_policy")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        errors.append("observation_qualification_scope_repeated_trials_invalid")
    if scope.get("paired_physical_synthetic_required") is not True:
        errors.append("observation_qualification_scope_pairing_required")
    if scope.get("downstream_task_validity_is_primary") is not True:
        errors.append("observation_qualification_scope_task_validity_not_primary")
    errors.extend(_governance_errors(scope, prefix="observation_qualification_scope"))
    if scope.get("scope_digest") != canonical_digest(scope, digest_field="scope_digest"):
        errors.append("observation_qualification_scope_digest_mismatch")
    if errors:
        raise MeasurementObservationQualificationError(*errors)
    return scope


def build_observation_challenge_case(
    scope_value: Mapping[str, Any],
    *,
    case_id: str,
    split: str,
    material_challenges: Sequence[str],
    lighting_challenges: Sequence[str],
    physical_observation_artifacts: Mapping[str, str],
    synthetic_observation_artifacts: Mapping[str, Mapping[str, str]],
    policy_trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scope = validate_observation_qualification_scope(scope_value)
    case = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "scope_digest": scope["scope_digest"],
        "benchmark_spec_digest": scope["benchmark_spec_digest"],
        "task_id": scope["task_id"],
        "site_id": scope["site_id"],
        "split": split,
        "material_challenges": list(material_challenges),
        "lighting_challenges": list(lighting_challenges),
        "physical_observation_artifacts": dict(physical_observation_artifacts),
        "synthetic_observation_artifacts": {
            _string(method): dict(artifacts)
            for method, artifacts in synthetic_observation_artifacts.items()
        },
        "policy_trials": [dict(row) for row in policy_trials],
        "sealed_physical_labels_included": False,
        "visual_similarity_is_primary": False,
        "q_sensor_qualification_created": False,
        "r6_decision_created": False,
        "r7_admission_created": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "agent_may_approve": False,
    }
    case["case_digest"] = canonical_digest(case, digest_field="case_digest")
    return validate_observation_challenge_case(case, scope)


def validate_observation_challenge_case(
    value: Mapping[str, Any], scope_value: Mapping[str, Any]
) -> dict[str, Any]:
    case = _clone(value)
    scope = validate_observation_qualification_scope(scope_value)
    errors: list[str] = []
    if case.get("schema_version") != CASE_SCHEMA_VERSION:
        errors.append("observation_challenge_case_schema_invalid")
    if not _string(case.get("case_id")):
        errors.append("observation_challenge_case_id_missing")
    for key in ("scope_digest", "benchmark_spec_digest", "task_id", "site_id"):
        expected = scope[key]
        if case.get(key) != expected:
            errors.append(f"observation_challenge_case_{key}_mismatch")
    if case.get("split") not in {"development", "qualification"}:
        errors.append("observation_challenge_case_split_invalid")
    materials = _strings(case.get("material_challenges"))
    lighting = _strings(case.get("lighting_challenges"))
    if (
        not materials
        or len(materials) != len(case.get("material_challenges") or [])
        or not set(materials) <= set(scope["required_material_challenges"])
    ):
        errors.append("observation_challenge_case_materials_invalid")
    if (
        len(lighting) != 1
        or len(lighting) != len(case.get("lighting_challenges") or [])
        or not set(lighting) <= set(scope["required_lighting_challenges"])
    ):
        errors.append("observation_challenge_case_lighting_invalid")
    required_modalities = set(scope["required_modalities"])
    physical = _digest_map(case.get("physical_observation_artifacts"))
    if set(physical) != required_modalities or any(
        not _valid_digest(item) for item in physical.values()
    ):
        errors.append("observation_challenge_case_physical_artifacts_invalid")
    synthetic_raw = case.get("synthetic_observation_artifacts")
    synthetic = synthetic_raw if isinstance(synthetic_raw, Mapping) else {}
    if set(synthetic) != set(scope["synthetic_method_ids"]):
        errors.append("observation_challenge_case_synthetic_methods_incomplete")
    for method_id, artifacts in synthetic.items():
        normalized = _digest_map(artifacts)
        if set(normalized) != required_modalities or any(
            not _valid_digest(item) for item in normalized.values()
        ):
            errors.append(f"observation_challenge_case_synthetic_artifacts_invalid:{method_id}")
    trials = case.get("policy_trials")
    if (
        not isinstance(trials, list)
        or not trials
        or not all(isinstance(row, Mapping) for row in trials)
    ):
        errors.append("observation_challenge_case_policy_trials_invalid")
        trials = []
    seen_replicates: set[tuple[str, str]] = set()
    policies = scope["policy_checkpoints"]
    for index, row in enumerate(trials):
        policy_id = _string(row.get("policy_id"))
        replicate_id = _string(row.get("replicate_id"))
        if policy_id not in policies or row.get("policy_digest") != policies.get(policy_id):
            errors.append(f"observation_challenge_case_policy_binding_invalid:{index}")
        key = (policy_id, replicate_id)
        if not replicate_id or key in seen_replicates:
            errors.append(f"observation_challenge_case_replicate_invalid:{index}")
        seen_replicates.add(key)
        if not _valid_digest(row.get("physical_outcome_digest")):
            errors.append(f"observation_challenge_case_physical_outcome_invalid:{index}")
        synthetic_outcomes = _digest_map(row.get("synthetic_outcome_digests"))
        if set(synthetic_outcomes) != set(scope["synthetic_method_ids"]) or any(
            not _valid_digest(item) for item in synthetic_outcomes.values()
        ):
            errors.append(f"observation_challenge_case_synthetic_outcomes_invalid:{index}")
    if case.get("sealed_physical_labels_included") is not False:
        errors.append("observation_challenge_case_sealed_label_leakage")
    errors.extend(_governance_errors(case, prefix="observation_challenge_case"))
    if case.get("case_digest") != canonical_digest(case, digest_field="case_digest"):
        errors.append("observation_challenge_case_digest_mismatch")
    if errors:
        raise MeasurementObservationQualificationError(*errors)
    return case


def evaluate_observation_challenge_matrix(
    scope_value: Mapping[str, Any],
    case_values: Sequence[Mapping[str, Any]],
    *,
    evaluator_id: str,
    evaluator_independent_of_candidates: bool,
) -> dict[str, Any]:
    scope = validate_observation_qualification_scope(scope_value)
    cases = [validate_observation_challenge_case(row, scope) for row in case_values]
    blockers: list[str] = []
    if not cases:
        blockers.append("observation_challenge_cases_missing")
    if not _string(evaluator_id):
        blockers.append("observation_challenge_evaluator_missing")
    if evaluator_independent_of_candidates is not True:
        blockers.append("observation_challenge_evaluator_not_independent")
    case_ids = [row["case_id"] for row in cases]
    if len(case_ids) != len(set(case_ids)):
        blockers.append("observation_challenge_duplicate_case_id")
    splits = {row["split"] for row in cases}
    if len(splits) != 1:
        blockers.append("observation_challenge_split_mismatch")
    split = next(iter(splits), "development")
    material_counts = Counter(
        challenge for row in cases for challenge in row["material_challenges"]
    )
    lighting_counts = Counter(
        challenge for row in cases for challenge in row["lighting_challenges"]
    )
    condition_pair_counts = Counter(
        (material, lighting)
        for row in cases
        for material in row["material_challenges"]
        for lighting in row["lighting_challenges"]
    )
    for challenge in scope["required_material_challenges"]:
        if material_counts[challenge] == 0:
            blockers.append(f"observation_material_challenge_missing:{challenge}")
    for challenge in scope["required_lighting_challenges"]:
        if lighting_counts[challenge] == 0:
            blockers.append(f"observation_lighting_challenge_missing:{challenge}")
    for material in scope["required_material_challenges"]:
        for lighting in scope["required_lighting_challenges"]:
            if condition_pair_counts[(material, lighting)] == 0:
                blockers.append(f"observation_condition_pair_missing:{material}:{lighting}")
    trial_counts: Counter[tuple[str, str]] = Counter()
    for row in cases:
        for trial in row["policy_trials"]:
            trial_counts[(row["case_id"], trial["policy_id"])] += 1
    for row in cases:
        for policy_id in scope["policy_checkpoints"]:
            count = trial_counts[(row["case_id"], policy_id)]
            if count < scope["minimum_repeated_trials_per_policy"]:
                blockers.append(
                    f"observation_policy_repeats_insufficient:{row['case_id']}:{policy_id}"
                )
    matrix_complete = not blockers
    qualification_ready = bool(matrix_complete and split == "qualification")
    status = (
        "qualification_matrix_ready"
        if qualification_ready
        else "development_matrix_complete"
        if matrix_complete
        else "blocked"
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope_digest": scope["scope_digest"],
        "benchmark_spec_digest": scope["benchmark_spec_digest"],
        "task_id": scope["task_id"],
        "site_id": scope["site_id"],
        "split": split,
        "evaluator_id": evaluator_id,
        "evaluator_independent_of_candidates": evaluator_independent_of_candidates is True,
        "case_count": len(cases),
        "case_digests": [
            row["case_digest"] for row in sorted(cases, key=lambda row: row["case_id"])
        ],
        "coverage": {
            "material_challenges": dict(sorted(material_counts.items())),
            "lighting_challenges": dict(sorted(lighting_counts.items())),
            "material_lighting_pairs": {
                f"{material}:{lighting}": count
                for (material, lighting), count in sorted(condition_pair_counts.items())
            },
            "required_modalities": list(scope["required_modalities"]),
            "synthetic_method_ids": list(scope["synthetic_method_ids"]),
            "policy_ids": sorted(scope["policy_checkpoints"]),
            "minimum_repeated_trials_per_policy": scope["minimum_repeated_trials_per_policy"],
        },
        "blockers": sorted(set(blockers)),
        "matrix_complete": matrix_complete,
        "qualification_matrix_ready": qualification_ready,
        "status": status,
        "visual_similarity_is_primary": False,
        "q_sensor_qualification_created": False,
        "r6_decision_created": False,
        "r7_admission_created": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "agent_may_approve": False,
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return validate_observation_challenge_report(report)


def validate_observation_challenge_report(value: Mapping[str, Any]) -> dict[str, Any]:
    report = _clone(value)
    errors: list[str] = []
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append("observation_challenge_report_schema_invalid")
    for key in ("scope_digest", "benchmark_spec_digest"):
        if not _valid_digest(report.get(key)):
            errors.append(f"observation_challenge_report_{key}_invalid")
    if report.get("status") not in {
        "blocked",
        "development_matrix_complete",
        "qualification_matrix_ready",
    }:
        errors.append("observation_challenge_report_status_invalid")
    if not isinstance(report.get("blockers"), list) or not isinstance(
        report.get("coverage"), Mapping
    ):
        errors.append("observation_challenge_report_content_invalid")
    if not _string(report.get("task_id")) or not _string(report.get("site_id")):
        errors.append("observation_challenge_report_task_site_missing")
    if not _string(report.get("evaluator_id")):
        errors.append("observation_challenge_report_evaluator_missing")
    case_digests = report.get("case_digests")
    case_count = report.get("case_count")
    if (
        not isinstance(case_digests, list)
        or any(not _valid_digest(item) for item in case_digests)
        or len(case_digests) != len(set(case_digests))
        or isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count != len(case_digests)
    ):
        errors.append("observation_challenge_report_case_index_invalid")
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    matrix_complete = report.get("matrix_complete") is True
    if matrix_complete != (not blockers):
        errors.append("observation_challenge_report_completion_boundary_invalid")
    ready = report.get("qualification_matrix_ready") is True
    expected_ready = bool(
        matrix_complete
        and report.get("split") == "qualification"
        and report.get("evaluator_independent_of_candidates") is True
        and not blockers
    )
    if ready != expected_ready:
        errors.append("observation_challenge_report_ready_boundary_invalid")
    expected_status = (
        "qualification_matrix_ready"
        if expected_ready
        else "development_matrix_complete"
        if matrix_complete
        else "blocked"
    )
    if report.get("status") != expected_status:
        errors.append("observation_challenge_report_status_not_deterministic")
    errors.extend(_governance_errors(report, prefix="observation_challenge_report"))
    if report.get("report_digest") != canonical_digest(report, digest_field="report_digest"):
        errors.append("observation_challenge_report_digest_mismatch")
    if errors:
        raise MeasurementObservationQualificationError(*errors)
    return report


def build_observation_r5_candidate_stage_data(
    scope_value: Mapping[str, Any],
    challenge_report_value: Mapping[str, Any],
    benchmark_report_value: Mapping[str, Any],
) -> dict[str, Any]:
    scope = validate_observation_qualification_scope(scope_value)
    challenge = validate_observation_challenge_report(challenge_report_value)
    benchmark = validate_qualification_benchmark_report(benchmark_report_value)
    errors: list[str] = []
    if challenge["scope_digest"] != scope["scope_digest"]:
        errors.append("observation_r5_scope_binding_mismatch")
    if any(
        item != scope["benchmark_spec_digest"]
        for item in (
            challenge["benchmark_spec_digest"],
            benchmark["benchmark_spec_digest"],
        )
    ):
        errors.append("observation_r5_benchmark_binding_mismatch")
    if challenge["qualification_matrix_ready"] is not True:
        errors.append("observation_r5_challenge_matrix_not_ready")
    if benchmark["benchmark_id"] != "capture-to-observation":
        errors.append("observation_r5_benchmark_type_invalid")
    if benchmark["evidence_status"] != "r5_evidence_candidate":
        errors.append("observation_r5_benchmark_report_not_candidate")
    coverage = challenge.get("coverage")
    coverage = dict(coverage) if isinstance(coverage, Mapping) else {}
    material_coverage = coverage.get("material_challenges")
    lighting_coverage = coverage.get("lighting_challenges")
    pair_coverage = coverage.get("material_lighting_pairs")
    expected_pairs = {
        f"{material}:{lighting}"
        for material in scope["required_material_challenges"]
        for lighting in scope["required_lighting_challenges"]
    }
    if not (
        isinstance(material_coverage, Mapping)
        and set(material_coverage) == set(scope["required_material_challenges"])
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in material_coverage.values()
        )
    ):
        errors.append("observation_r5_material_coverage_invalid")
    if not (
        isinstance(lighting_coverage, Mapping)
        and set(lighting_coverage) == set(scope["required_lighting_challenges"])
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in lighting_coverage.values()
        )
    ):
        errors.append("observation_r5_lighting_coverage_invalid")
    if not (
        isinstance(pair_coverage, Mapping)
        and set(pair_coverage) == expected_pairs
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in pair_coverage.values()
        )
    ):
        errors.append("observation_r5_condition_pair_coverage_invalid")
    if coverage.get("required_modalities") != scope["required_modalities"]:
        errors.append("observation_r5_modality_coverage_invalid")
    if coverage.get("synthetic_method_ids") != scope["synthetic_method_ids"]:
        errors.append("observation_r5_method_coverage_invalid")
    if coverage.get("policy_ids") != sorted(scope["policy_checkpoints"]):
        errors.append("observation_r5_policy_coverage_invalid")
    if (
        coverage.get("minimum_repeated_trials_per_policy")
        != scope["minimum_repeated_trials_per_policy"]
    ):
        errors.append("observation_r5_policy_repeat_coverage_invalid")
    measured_metrics = {
        metric_id for row in benchmark["joined_cases"] for metric_id in row["metric_errors"]
    }
    missing_metrics = sorted(set(scope["required_metric_ids"]) - measured_metrics)
    errors.extend(f"observation_r5_required_metric_missing:{item}" for item in missing_metrics)
    if errors:
        raise MeasurementObservationQualificationError(*errors)
    physical_ids = sorted(
        {item for row in benchmark["joined_cases"] for item in row["physical_measurement_ids"]}
    )
    return {
        "heldout_evaluation": {
            "independent_execution": True,
            "task_id": scope["task_id"],
            "site_id": scope["site_id"],
            "task_request_digest": scope["task_request_digest"],
            "site_evidence_profile_digest": scope["site_evidence_profile_digest"],
            "sensor_pairing_digest": scope["sensor_pairing_digest"],
            "observation_scope_digest": scope["scope_digest"],
            "challenge_report_digest": challenge["report_digest"],
            "challenge_evaluator_id": challenge["evaluator_id"],
            "benchmark_report_digest": benchmark["benchmark_report_digest"],
            "benchmark_evaluator_id": benchmark["evaluator_id"],
            "qualification_split_hash": benchmark["split_digest"],
            "physical_measurement_ids": physical_ids,
            "measured_metric_ids": sorted(measured_metrics),
            "confidence_intervals": benchmark["confidence_intervals"],
            "challenge_coverage": challenge["coverage"],
        },
        "evidence_status": "r5_evidence_candidate",
        "q_sensor_qualification_created": False,
        "r6_human_decision_required": True,
        "r6_decision_created": False,
        "r7_catalog_admission": False,
        "production_route_eligible": False,
        "policy_rank_fidelity_public_claim_eligible": False,
        "physical_success_established": False,
        "agent_may_approve": False,
    }


__all__ = [
    "CASE_SCHEMA_VERSION",
    "LIGHTING_CHALLENGES",
    "MATERIAL_CHALLENGES",
    "MeasurementObservationQualificationError",
    "REPORT_SCHEMA_VERSION",
    "SCOPE_SCHEMA_VERSION",
    "build_observation_challenge_case",
    "build_observation_qualification_scope",
    "build_observation_r5_candidate_stage_data",
    "evaluate_observation_challenge_matrix",
    "validate_observation_challenge_case",
    "validate_observation_challenge_report",
    "validate_observation_qualification_scope",
]
