"""Human-controlled R0-R8 admission state machine for measurement methods.

Implements the research-admission protocol from the measurement-routing
research: R0 intake, R1 source verification, R2 rights/access/privacy, R3
adapter feasibility, R4 frozen benchmark preregistration, R5 held-out
evaluation, R6 independent qualification decision, R7 versioned catalog
admission, and R8 monitoring/requalification.

Fail-closed properties enforced here:

- stages advance one at a time and every completed stage's evidence is
  retained and re-checked, so a record cannot reach or claim R7 without a
  verifiable R0..R6 chain;
- a failed, research-only, or suspended R6 decision blocks catalog admission
  and production eligibility;
- development and qualification splits must differ (benchmark-leakage guard)
  and R5 must bind to the exact preregistered qualification split;
- vendor-submitted results can never substitute for independent execution;
- only humans approve, and any requalification trigger (engine/solver/API or
  model update, driver or numerical-backend change, adapter modification,
  regression, new failure mode, license/privacy change, capture-pipeline
  change, monitoring drift) suspends production eligibility until a new
  human decision is made.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "measurement_research_admission.v1"
STAGES = tuple(f"R{index}" for index in range(9))
DECISIONS = {
    "failed",
    "research_only",
    "approved_narrow_scope",
    "approved_conservative",
    "suspended",
}
APPROVED_DECISIONS = {"approved_narrow_scope", "approved_conservative"}
REQUALIFICATION_TRIGGERS = frozenset(
    {
        "engine_solver_api_or_model_update",
        "gpu_driver_or_numerical_backend_change",
        "adapter_modification",
        "regression_detected",
        "new_failure_mode_discovered",
        "license_or_privacy_change",
        "capture_pipeline_change",
        "monitoring_performance_drift",
        "qualification_scope_change_requested",
    }
)

_STAGE_APPROVER_ROLES: dict[str, frozenset[str]] = {
    "R0": frozenset({"research_analyst"}),
    "R1": frozenset({"research_lead"}),
    "R2": frozenset({"legal_owner", "privacy_owner"}),
    "R3": frozenset({"platform_owner"}),
    "R4": frozenset({"benchmark_owner", "independent_reviewer"}),
    "R5": frozenset({"benchmark_owner"}),
    "R6": frozenset({"research_lead", "domain_owner", "independent_reviewer"}),
    "R7": frozenset({"catalog_owner", "independent_reviewer"}),
    "R8": frozenset({"catalog_owner"}),
}


class MeasurementAdmissionError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementAdmissionError("admission_record_not_json") from exc
    if not isinstance(result, dict):
        raise MeasurementAdmissionError("admission_record_not_object")
    return result


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _digest(value: Mapping[str, Any], field: str = "admission_record_digest") -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _approval_roles(approvals: Any) -> set[str]:
    return {
        _string(row.get("role"))
        for row in approvals or []
        if isinstance(row, Mapping)
        and row.get("actor_type") == "human"
        and row.get("approved") is True
        and _string(row.get("signature_id"))
    }


def _r6_outcome(record: Mapping[str, Any]) -> str:
    stage = _string(record.get("stage"))
    if stage == "R6":
        data = record.get("stage_data")
    else:
        completed = record.get("completed_stage_data")
        data = completed.get("R6") if isinstance(completed, Mapping) else None
    data = dict(data) if isinstance(data, Mapping) else {}
    decision = data.get("qualification_decision")
    decision = dict(decision) if isinstance(decision, Mapping) else {}
    return _string(decision.get("outcome"))


def _stage_data_errors(record: Mapping[str, Any], stage: str, data: Mapping[str, Any]) -> list[str]:
    """Evidence-completeness checks for one stage, independent of approvals."""

    errors: list[str] = []
    if stage == "R0":
        for key in ("primary_sources", "method_identity", "claimed_scope", "access_status"):
            if not (
                _nonempty_list(data.get(key))
                if key == "primary_sources"
                else _nonempty_mapping(data.get(key))
            ):
                errors.append(f"R0_{key}_missing")
    elif stage == "R1":
        for key in (
            "source_verification",
            "code_access",
            "license_records",
            "vendor_claim_separation",
        ):
            if not _nonempty_mapping(data.get(key)):
                errors.append(f"R1_{key}_missing")
    elif stage == "R2":
        required = (
            "commercial_use",
            "site_data_ownership",
            "retention",
            "training_use",
            "subprocessors",
            "offline_option",
            "export_rights",
            "output_portability",
            "termination_deletion",
        )
        if any(key not in data for key in required):
            errors.append("R2_rights_privacy_review_incomplete")
    elif stage == "R3":
        required = (
            "scene_robot_formats",
            "coordinate_units",
            "collider_material_path",
            "controller_action_adapter",
            "sensor_adapter",
            "headless_execution",
            "deterministic_replay",
            "logs_state_access",
            "engineering_burden",
        )
        if any(key not in data for key in required):
            errors.append("R3_adapter_feasibility_incomplete")
    elif stage == "R4":
        prereg = data.get("frozen_benchmark_preregistration")
        required = (
            "task_site_classes",
            "development_split_hash",
            "qualification_split_hash",
            "robot_controller_digests",
            "capture_bundle_hashes",
            "metrics",
            "acceptance_thresholds",
            "comparison_methods",
            "compute_budget",
            "failure_criteria",
            "statistical_method",
            "claim_ceiling",
        )
        if not isinstance(prereg, Mapping) or any(key not in prereg for key in required):
            errors.append("R4_preregistration_incomplete")
        else:
            development = _string(prereg.get("development_split_hash"))
            qualification = _string(prereg.get("qualification_split_hash"))
            if not development or not qualification or development == qualification:
                errors.append("R4_development_and_qualification_splits_must_differ")
        if data.get("heldout_labels_exposed") is not False:
            errors.append("R4_heldout_labels_must_remain_hidden")
    elif stage == "R5":
        results = data.get("heldout_evaluation")
        required = (
            "independent_execution",
            "hidden_case_hashes",
            "physical_measurement_ids",
            "repeated_trial_count",
            "confidence_intervals",
            "harmful_false_negative_analysis",
            "retained_failure_ids",
            "clean_environment_rerun_id",
            "qualification_split_hash",
        )
        if not isinstance(results, Mapping) or any(key not in results for key in required):
            errors.append("R5_heldout_evaluation_incomplete")
        else:
            if results.get("independent_execution") is not True:
                errors.append("R5_independent_execution_required")
            completed = record.get("completed_stage_data")
            completed = dict(completed) if isinstance(completed, Mapping) else {}
            r4 = completed.get("R4")
            r4 = dict(r4) if isinstance(r4, Mapping) else {}
            prereg = r4.get("frozen_benchmark_preregistration")
            prereg = dict(prereg) if isinstance(prereg, Mapping) else {}
            preregistered_split = _string(prereg.get("qualification_split_hash"))
            if _string(results.get("qualification_split_hash")) != preregistered_split:
                errors.append("R5_heldout_split_binding_mismatch")
        if data.get("vendor_graded_qualification") is not False:
            errors.append("R5_vendor_self_grading_forbidden")
        if data.get("vendor_submitted_results_only") is True:
            errors.append("R5_vendor_submitted_results_cannot_qualify")
    elif stage == "R6":
        decision = data.get("qualification_decision")
        if not isinstance(decision, Mapping) or decision.get("outcome") not in DECISIONS:
            errors.append("R6_qualification_decision_invalid")
        if data.get("agent_approved") is not False:
            errors.append("R6_agent_approval_forbidden")
    elif stage == "R7":
        entry = data.get("catalog_admission")
        required = (
            "method_version",
            "capability_profile_digest",
            "scope_envelope",
            "qualification_ids",
            "expiration_date",
            "claim_ceiling",
            "known_failure_modes",
            "required_site_evidence",
            "prohibited_extrapolations",
        )
        if not isinstance(entry, Mapping) or any(key not in entry for key in required):
            errors.append("R7_catalog_admission_incomplete")
        if _r6_outcome(record) not in APPROVED_DECISIONS:
            errors.append("R7_requires_approved_r6_outcome")
    elif stage == "R8":
        monitoring = data.get("monitoring")
        required = ("monitoring_case_hashes", "requalification_triggers", "current_status")
        if not isinstance(monitoring, Mapping) or any(key not in monitoring for key in required):
            errors.append("R8_monitoring_contract_incomplete")
        else:
            for trigger in monitoring.get("requalification_triggers") or []:
                if _string(trigger) not in REQUALIFICATION_TRIGGERS:
                    errors.append(f"R8_requalification_trigger_unknown:{_string(trigger)}")
        if _r6_outcome(record) not in APPROVED_DECISIONS:
            errors.append("R8_requires_approved_r6_outcome")
    return errors


def _stage_errors(record: Mapping[str, Any], stage: str) -> list[str]:
    data = record.get("stage_data")
    data = dict(data) if isinstance(data, Mapping) else {}
    errors = _stage_data_errors(record, stage, data)
    roles = _approval_roles(record.get("approvals"))
    required_roles = _STAGE_APPROVER_ROLES.get(stage, frozenset())
    missing_roles = sorted(required_roles - roles)
    if missing_roles:
        errors.append(f"{stage}_required_approvals_missing:" + ",".join(missing_roles))
    return errors


def _chain_errors(record: Mapping[str, Any], stage: str) -> list[str]:
    errors: list[str] = []
    stage_index = STAGES.index(stage)
    history = record.get("transition_history")
    history = history if isinstance(history, list) else []
    if len(history) != stage_index:
        errors.append("admission_transition_history_incomplete")
    else:
        for index, transition in enumerate(history):
            if (
                not isinstance(transition, Mapping)
                or transition.get("from_stage") != STAGES[index]
                or transition.get("to_stage") != STAGES[index + 1]
                or not _string(transition.get("predecessor_digest")).startswith("sha256:")
                or not _nonempty_list(transition.get("approval_signature_ids"))
            ):
                errors.append(f"admission_transition_history_invalid:{index}")
    completed = record.get("completed_stage_data")
    completed = dict(completed) if isinstance(completed, Mapping) else {}
    expected_completed = set(STAGES[:stage_index])
    if set(completed) != expected_completed:
        errors.append("admission_completed_stage_data_incomplete")
    else:
        for prior_stage in STAGES[:stage_index]:
            prior_data = completed.get(prior_stage)
            prior_data = dict(prior_data) if isinstance(prior_data, Mapping) else {}
            errors.extend(_stage_data_errors(record, prior_stage, prior_data))
    return errors


def _expected_production_eligibility(record: Mapping[str, Any]) -> bool:
    stage = _string(record.get("stage"))
    return (
        stage in {"R7", "R8"}
        and record.get("suspended") is not True
        and _r6_outcome(record) in APPROVED_DECISIONS
    )


def validate_research_admission_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = _clone(value)
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("admission_schema_version_invalid")
    if not _string(record.get("candidate_id")) or not _string(record.get("method_id")):
        errors.append("admission_identity_missing")
    stage = _string(record.get("stage"))
    if stage not in STAGES:
        errors.append("admission_stage_invalid")
    if not isinstance(record.get("stage_data"), Mapping):
        errors.append("admission_stage_data_invalid")
    record.setdefault("completed_stage_data", {})
    if not isinstance(record.get("completed_stage_data"), Mapping):
        errors.append("admission_completed_stage_data_invalid")
    record.setdefault("suspended", False)
    if not isinstance(record.get("suspended"), bool):
        errors.append("admission_suspended_flag_invalid")
    record.setdefault("requalification_events", [])
    requalification_events = record.get("requalification_events")
    if not isinstance(requalification_events, list):
        errors.append("admission_requalification_events_invalid")
    else:
        for index, event in enumerate(requalification_events):
            if (
                not isinstance(event, Mapping)
                or _string(event.get("trigger")) not in REQUALIFICATION_TRIGGERS
                or not _string(event.get("signature_id"))
            ):
                errors.append(f"admission_requalification_event_invalid:{index}")
    approvals = record.get("approvals")
    if (
        not isinstance(approvals, list)
        or not approvals
        or any(not isinstance(row, Mapping) for row in approvals)
    ):
        errors.append("admission_approvals_invalid")
    elif any(row.get("actor_type") != "human" for row in approvals):
        errors.append("admission_agent_approval_forbidden")
    if not isinstance(record.get("transition_history"), list):
        errors.append("admission_transition_history_invalid")
    if not errors:
        errors.extend(_chain_errors(record, stage))
        errors.extend(_stage_errors(record, stage))
        if record.get("production_eligible") is not _expected_production_eligibility(record):
            errors.append("admission_production_eligibility_invalid")
    expected = _digest(record)
    supplied = record.get("admission_record_digest")
    if supplied is not None and supplied != expected:
        errors.append("admission_record_digest_mismatch")
    if errors:
        raise MeasurementAdmissionError(*errors)
    record["admission_record_digest"] = expected
    return record


def create_research_candidate(
    *, candidate_id: str, method_id: str, stage_data: Mapping[str, Any], approval: Mapping[str, Any]
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "method_id": method_id,
        "stage": "R0",
        "stage_data": dict(stage_data),
        "completed_stage_data": {},
        "approvals": [dict(approval)],
        "production_eligible": False,
        "suspended": False,
        "requalification_events": [],
        "transition_history": [],
    }
    return validate_research_admission_record(value)


def advance_research_admission(
    value: Mapping[str, Any],
    *,
    target_stage: str,
    stage_data: Mapping[str, Any],
    approvals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current = validate_research_admission_record(value)
    if current.get("suspended") is True:
        raise MeasurementAdmissionError("admission_suspended_requires_new_qualification_decision")
    current_index = STAGES.index(current["stage"])
    if target_stage not in STAGES or STAGES.index(target_stage) != current_index + 1:
        raise MeasurementAdmissionError("admission_transition_must_be_sequential")
    if any(row.get("actor_type") != "human" for row in approvals):
        raise MeasurementAdmissionError("admission_agent_approval_forbidden")
    if current["stage"] == "R6" and _r6_outcome(current) not in APPROVED_DECISIONS:
        raise MeasurementAdmissionError("admission_r6_outcome_blocks_catalog_admission")
    next_value = {
        **current,
        "stage": target_stage,
        "stage_data": dict(stage_data),
        "completed_stage_data": {
            **dict(current.get("completed_stage_data") or {}),
            current["stage"]: dict(current.get("stage_data") or {}),
        },
        "approvals": [dict(row) for row in approvals],
        "transition_history": [
            *current["transition_history"],
            {
                "from_stage": current["stage"],
                "to_stage": target_stage,
                "predecessor_digest": current["admission_record_digest"],
                "approval_signature_ids": sorted(
                    _string(row.get("signature_id")) for row in approvals
                ),
            },
        ],
    }
    next_value.pop("admission_record_digest", None)
    next_value["production_eligible"] = _expected_production_eligibility(next_value)
    return validate_research_admission_record(next_value)


def apply_requalification_trigger(
    value: Mapping[str, Any], *, trigger: str, detail: str, approval: Mapping[str, Any]
) -> dict[str, Any]:
    """Suspend production eligibility at R7/R8 on any requalification trigger.

    A newer engine release, driver or numerical-backend change, adapter edit,
    regression, new failure mode, license/privacy change, capture-pipeline
    change, or monitoring drift is a new method version, never a transparent
    substitution.
    """

    current = validate_research_admission_record(value)
    if current["stage"] not in {"R7", "R8"}:
        raise MeasurementAdmissionError("admission_requalification_only_after_catalog_admission")
    if _string(trigger) not in REQUALIFICATION_TRIGGERS:
        raise MeasurementAdmissionError(
            f"admission_requalification_trigger_unknown:{_string(trigger)}"
        )
    if not isinstance(approval, Mapping) or approval.get("actor_type") != "human":
        raise MeasurementAdmissionError("admission_agent_approval_forbidden")
    next_value = {
        **current,
        "suspended": True,
        "production_eligible": False,
        "requalification_events": [
            *current["requalification_events"],
            {
                "trigger": _string(trigger),
                "detail": _string(detail),
                "signature_id": _string(approval.get("signature_id")),
                "predecessor_digest": current["admission_record_digest"],
            },
        ],
    }
    next_value.pop("admission_record_digest", None)
    return validate_research_admission_record(next_value)


def admission_supports_production_route(value: Mapping[str, Any]) -> bool:
    """True only for a validated, unsuspended R7/R8 record with approved R6."""

    try:
        record = validate_research_admission_record(value)
    except MeasurementAdmissionError:
        return False
    return record["production_eligible"] is True


__all__ = [
    "APPROVED_DECISIONS",
    "DECISIONS",
    "MeasurementAdmissionError",
    "REQUALIFICATION_TRIGGERS",
    "SCHEMA_VERSION",
    "STAGES",
    "admission_supports_production_route",
    "advance_research_admission",
    "apply_requalification_trigger",
    "create_research_candidate",
    "validate_research_admission_record",
]
