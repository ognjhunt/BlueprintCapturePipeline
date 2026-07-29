"""Versioned contracts for the claim-level Decision/Evidence control plane.

These contracts sit above ``evaluation_run_contract.EvaluationRunSpec``.  They
describe why evidence is needed, which methods are qualified, what was planned,
and what decision the evidence supports.  They never replace raw capture truth
or treat provider availability as qualification.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence


TESTBED_SCHEMA_VERSION = "maintained_site_task_testbed.v1"
DECISION_REQUEST_SCHEMA_VERSION = "decision_evidence_request.v1"
METHOD_PROFILE_SCHEMA_VERSION = "evidence_method_profile.v1"
QUALIFICATION_SCHEMA_VERSION = "evidence_method_qualification.v1"
EVIDENCE_PLAN_SCHEMA_VERSION = "evidence_plan.v1"
EVIDENCE_RESULT_SCHEMA_VERSION = "normalized_evidence_result.v1"
DECISION_ENVELOPE_SCHEMA_VERSION = "decision_envelope.v1"
PHYSICAL_OUTCOME_SCHEMA_VERSION = "physical_outcome_join.v1"

METHOD_FAMILIES = {
    "analytic_geometry_kinematics",
    "captured_real_observation",
    "traditional_simulation",
    "learned_world_model",
    "external_provider_tool",
    "physical_evidence",
    "owner_attested_operational_input",
}
LIFECYCLE_STATES = {"draft", "active", "invalidated", "superseded", "retired"}
QUALIFICATION_STATUSES = {"qualified", "not_qualified", "debug_only", "expired"}
PARTITIONS = {"calibration", "heldout"}
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SECRET_KEYS = {
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


class DecisionEvidenceContractError(ValueError):
    """Fail-closed validation error with stable, sorted error identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: Mapping[str, Any], *, digest_field: str | None = None) -> str:
    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    return f"sha256:{hashlib.sha256(canonical_json(normalized).encode('utf-8')).hexdigest()}"


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise DecisionEvidenceContractError(["artifact:not_json_serializable"]) from exc
    if not isinstance(cloned, dict):
        raise DecisionEvidenceContractError(["artifact:not_mapping"])
    return cloned


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_rows(value: Any, *, nonempty: bool = False) -> bool:
    valid = isinstance(value, list) and all(isinstance(row, Mapping) for row in value)
    return valid and (bool(value) or not nonempty)


def _is_strings(value: Any, *, nonempty: bool = False) -> bool:
    valid = isinstance(value, list) and all(bool(_string(row)) for row in value)
    return valid and (bool(value) or not nonempty)


def _valid_number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if minimum is not None and number < minimum:
        return False
    if maximum is not None and number > maximum:
        return False
    return True


def _secret_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            lowered = key.lower()
            path = f"{prefix}.{key}" if prefix else key
            secret_key = lowered in _SECRET_KEYS or any(
                lowered.endswith(f"_{suffix}") for suffix in _SECRET_KEYS
            )
            if secret_key and nested not in (None, "", [], {}):
                paths.append(path)
            paths.extend(_secret_paths(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_secret_paths(nested, prefix=f"{prefix}[{index}]"))
    return paths


def _required_string(errors: list[str], value: Mapping[str, Any], key: str) -> None:
    if not _string(value.get(key)):
        errors.append(f"{key}:missing")


def _required_identifier(errors: list[str], value: Mapping[str, Any], key: str) -> None:
    text = _string(value.get(key))
    if not text:
        errors.append(f"{key}:missing")
    elif not _IDENTIFIER.fullmatch(text):
        errors.append(f"{key}:invalid")


def _required_digest(errors: list[str], value: Mapping[str, Any], key: str) -> None:
    text = _string(value.get(key))
    if not text:
        errors.append(f"{key}:missing")
    elif not _SHA256.fullmatch(text):
        errors.append(f"{key}:invalid_sha256")


def _required_mapping(errors: list[str], value: Mapping[str, Any], key: str) -> None:
    if not isinstance(value.get(key), Mapping) or not value.get(key):
        errors.append(f"{key}:missing_or_empty")


@dataclass(frozen=True)
class _ValidatedArtifact:
    """Immutable JSON-backed artifact wrapper shared by all router contracts."""

    _canonical: str
    SCHEMA_VERSION: ClassVar[str]
    DIGEST_FIELD: ClassVar[str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_ValidatedArtifact":
        if not isinstance(value, Mapping):
            raise DecisionEvidenceContractError(["artifact:not_mapping"])
        normalized = _clone(value)
        errors: list[str] = []
        if _string(normalized.get("schema_version")) != cls.SCHEMA_VERSION:
            errors.append(f"schema_version:must_be:{cls.SCHEMA_VERSION}")
        errors.extend(cls._validation_errors(normalized))
        errors.extend(f"secret_value_forbidden:{path}" for path in _secret_paths(normalized))
        expected_digest = canonical_digest(normalized, digest_field=cls.DIGEST_FIELD)
        supplied_digest = _string(normalized.get(cls.DIGEST_FIELD))
        if supplied_digest and supplied_digest != expected_digest:
            errors.append(f"{cls.DIGEST_FIELD}:mismatch")
        if errors:
            raise DecisionEvidenceContractError(errors)
        normalized[cls.DIGEST_FIELD] = expected_digest
        return cls(canonical_json(normalized))

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        raise NotImplementedError

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    @property
    def digest(self) -> str:
        return _string(self.to_mapping().get(self.DIGEST_FIELD))


@dataclass(frozen=True)
class MaintainedSiteTaskTestbed(_ValidatedArtifact):
    SCHEMA_VERSION: ClassVar[str] = TESTBED_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "testbed_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        _required_identifier(errors, value, "testbed_id")
        _required_identifier(errors, value, "version")
        if "predecessor_testbed_digest" not in value:
            errors.append("predecessor_testbed_digest:missing")
        elif value.get("predecessor_testbed_digest") is not None and not _SHA256.fullmatch(
            _string(value.get("predecessor_testbed_digest"))
        ):
            errors.append("predecessor_testbed_digest:invalid_sha256")
        if not _is_strings(value.get("supersedes"), nonempty=False):
            errors.append("supersedes:must_be_string_list")
        bundles = value.get("source_capture_bundles")
        if not _is_rows(bundles, nonempty=True):
            errors.append("source_capture_bundles:missing_or_invalid")
        else:
            for index, bundle in enumerate(bundles):
                for key in ("bundle_id", "version"):
                    if not _string(bundle.get(key)):
                        errors.append(f"source_capture_bundles[{index}].{key}:missing")
                if not _SHA256.fullmatch(_string(bundle.get("digest"))):
                    errors.append(f"source_capture_bundles[{index}].digest:invalid_sha256")
        references = value.get("artifact_references")
        required_refs = {
            "site_card",
            "task_cards",
            "scenario_cards",
            "eval_cards",
            "evaluator",
            "reset",
        }
        if not isinstance(references, Mapping):
            errors.append("artifact_references:missing")
        else:
            errors.extend(
                f"artifact_references.{key}:missing"
                for key in sorted(required_refs)
                if key not in references
            )
        for key in (
            "task_distribution",
            "supported_condition_ranges",
            "robot_sensor_controller_bindings",
            "governance",
            "validation_envelope",
        ):
            _required_mapping(errors, value, key)
        governance = value.get("governance")
        if isinstance(governance, Mapping):
            for key in ("rights", "consent", "privacy", "revocation", "allowed_uses"):
                if key not in governance:
                    errors.append(f"governance.{key}:missing")
        for key in (
            "evidence_inventory",
            "known_unsupported_conditions",
            "invalidation_triggers",
            "physical_outcome_history_refs",
        ):
            if not isinstance(value.get(key), list):
                errors.append(f"{key}:must_be_list")
        if _string(value.get("lifecycle_state")) not in LIFECYCLE_STATES:
            errors.append("lifecycle_state:unsupported")
        return errors


@dataclass(frozen=True)
class DecisionEvidenceRequest(_ValidatedArtifact):
    SCHEMA_VERSION: ClassVar[str] = DECISION_REQUEST_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "request_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("request_id", "decision_id", "testbed_id", "testbed_version"):
            _required_identifier(errors, value, key)
        _required_digest(errors, value, "testbed_digest")
        _required_string(errors, value, "decision_question")
        if not isinstance(value.get("candidates"), list):
            errors.append("candidates:must_be_list")
        claims = value.get("claims")
        if not _is_rows(claims, nonempty=True):
            errors.append("claims:missing_or_invalid")
        else:
            claim_ids: list[str] = []
            for index, claim in enumerate(claims):
                for key in ("claim_id", "claim_type", "subject"):
                    if not _string(claim.get(key)):
                        errors.append(f"claims[{index}].{key}:missing")
                claim_ids.append(_string(claim.get("claim_id")))
                if "measurable_threshold" not in claim:
                    errors.append(f"claims[{index}].measurable_threshold:missing")
                if not _string(claim.get("false_safe_consequence")):
                    errors.append(f"claims[{index}].false_safe_consequence:missing")
                if not _valid_number(
                    claim.get("acceptable_false_safe_risk"), minimum=0.0, maximum=1.0
                ):
                    errors.append(f"claims[{index}].acceptable_false_safe_risk:invalid")
                desired = claim.get("desired_confidence_or_coverage")
                if not isinstance(desired, Mapping) or not desired:
                    errors.append(
                        f"claims[{index}].desired_confidence_or_coverage:missing"
                    )
                if not isinstance(claim.get("permitted_abstention_behavior"), Mapping):
                    errors.append(
                        f"claims[{index}].permitted_abstention_behavior:missing"
                    )
            if len(set(claim_ids)) != len(claim_ids):
                errors.append("claims:duplicate_claim_id")
        budget = value.get("budget")
        if not isinstance(budget, Mapping):
            errors.append("budget:missing")
        elif not _valid_number(budget.get("max_cost_usd"), minimum=0.0):
            errors.append("budget.max_cost_usd:invalid")
        _required_string(errors, value, "deadline")
        if not isinstance(value.get("available_physical_evidence"), list):
            errors.append("available_physical_evidence:must_be_list")
        if not _is_strings(value.get("permitted_evidence_methods"), nonempty=True):
            errors.append("permitted_evidence_methods:missing_or_invalid")
        _required_mapping(errors, value, "restrictions")
        _required_string(errors, value, "requested_result_audience")
        _required_mapping(errors, value, "provenance")
        if isinstance(value.get("provenance"), Mapping) and not _string(
            value["provenance"].get("caller_identity")
        ):
            errors.append("provenance.caller_identity:missing")
        _required_string(errors, value, "idempotency_key")
        for forbidden in (
            "selected_method",
            "selected_provider",
            "selected_simulator",
            "runtime_provider_profile",
        ):
            if forbidden in value:
                errors.append(f"request_method_selection_forbidden:{forbidden}")
        return errors


@dataclass(frozen=True)
class EvidenceMethodProfile(_ValidatedArtifact):
    SCHEMA_VERSION: ClassVar[str] = METHOD_PROFILE_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "method_profile_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("method_id", "version"):
            _required_identifier(errors, value, key)
        _required_digest(errors, value, "implementation_digest")
        _required_string(errors, value, "adapter_reference")
        if _string(value.get("method_family")) not in METHOD_FAMILIES:
            errors.append("method_family:unsupported")
        if not _is_strings(value.get("supported_claim_types"), nonempty=True):
            errors.append("supported_claim_types:missing_or_invalid")
        if not _is_strings(value.get("required_inputs"), nonempty=False):
            errors.append("required_inputs:must_be_string_list")
        _required_mapping(errors, value, "applicability_envelope")
        if not isinstance(value.get("calibration_evidence_references"), list):
            errors.append("calibration_evidence_references:must_be_list")
        if not _valid_number(value.get("authority_tier"), minimum=0, maximum=4):
            errors.append("authority_tier:invalid")
        _required_string(errors, value, "proof_tier")
        _required_string(errors, value, "correlation_group")
        if not _is_strings(value.get("shared_dependencies"), nonempty=False):
            errors.append("shared_dependencies:must_be_string_list")
        for key in ("expected_cost_usd", "expected_latency_seconds"):
            if not _valid_number(value.get(key), minimum=0.0):
                errors.append(f"{key}:invalid")
        _required_string(errors, value, "reproducibility_level")
        for key in ("constraints", "provider_availability"):
            _required_mapping(errors, value, key)
        if not _is_strings(value.get("failure_modes"), nonempty=True):
            errors.append("failure_modes:missing_or_invalid")
        if not _is_strings(value.get("abstention_modes"), nonempty=True):
            errors.append("abstention_modes:missing_or_invalid")
        if not _is_strings(value.get("disqualifying_conditions"), nonempty=False):
            errors.append("disqualifying_conditions:must_be_string_list")
        if value.get("self_qualified") is True:
            errors.append("method_self_qualification_forbidden")
        return errors


@dataclass(frozen=True)
class QualificationRecord(_ValidatedArtifact):
    SCHEMA_VERSION: ClassVar[str] = QUALIFICATION_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "qualification_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        _required_identifier(errors, value, "qualification_id")
        for key in ("method_id", "method_version", "claim_type", "task_family"):
            _required_string(errors, value, key)
        for key in ("method_profile_digest", "implementation_digest", "evaluator_digest"):
            _required_digest(errors, value, key)
        for key in (
            "site_domain_conditions",
            "embodiment",
            "sensors",
            "controller_action_representation",
            "evaluator",
            "confidence_intervals",
            "provenance",
        ):
            _required_mapping(errors, value, key)
        if not _is_rows(value.get("predictions"), nonempty=True):
            errors.append("predictions:missing_or_invalid")
        if not _is_rows(value.get("accepted_real_outcomes"), nonempty=True):
            errors.append("accepted_real_outcomes:missing_or_invalid")
        if _string(value.get("calibration_partition")) not in PARTITIONS:
            errors.append("calibration_partition:unsupported")
        for key in ("coverage", "abstention_rate", "false_safe_rate", "false_reject_rate"):
            if not _valid_number(value.get(key), minimum=0.0, maximum=1.0):
                errors.append(f"{key}:invalid")
        if not isinstance(value.get("owner_evidence"), list) or not value.get("owner_evidence"):
            errors.append("owner_evidence:missing_or_invalid")
        if _string(value.get("status")) not in QUALIFICATION_STATUSES:
            errors.append("status:unsupported")
        if value.get("self_grading") is not False:
            errors.append("self_grading:must_be_false")
        subject_provider = _string(value.get("subject_provider_id"))
        evaluator_provider = _string(value.get("evaluator_provider_id"))
        if subject_provider and evaluator_provider and subject_provider == evaluator_provider:
            errors.append("provider_self_grading_forbidden")
        if _string(value.get("claim_type")) == "comparative_policy_ranking":
            _required_mapping(errors, value, "policy_checkpoint_identity")
            _required_mapping(errors, value, "perturbation_sensitivity_metrics")
            if not _valid_number(value.get("simulated_rollout_count"), minimum=1):
                errors.append("simulated_rollout_count:invalid")
            if not _valid_number(value.get("physical_rollout_count"), minimum=1):
                errors.append("physical_rollout_count:invalid")
        return errors


@dataclass(frozen=True)
class EvidencePlan(_ValidatedArtifact):
    SCHEMA_VERSION: ClassVar[str] = EVIDENCE_PLAN_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "plan_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("plan_id", "request_id", "decision_id", "testbed_id", "testbed_version"):
            _required_identifier(errors, value, key)
        for key in ("request_digest", "testbed_digest"):
            _required_digest(errors, value, key)
        if not _is_rows(value.get("claim_plans"), nonempty=True):
            errors.append("claim_plans:missing_or_invalid")
        for key in (
            "execution_order",
            "stop_conditions",
            "escalation_conditions",
            "physical_evidence_requests",
            "compiled_evaluation_run_specs",
            "non_evaluation_run_steps",
            "prohibited_claims",
            "shared_dependency_warnings",
        ):
            if not isinstance(value.get(key), list):
                errors.append(f"{key}:must_be_list")
        _required_mapping(errors, value, "budget_status")
        return errors


@dataclass(frozen=True)
class NormalizedEvidenceResult(_ValidatedArtifact):
    SCHEMA_VERSION: ClassVar[str] = EVIDENCE_RESULT_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "result_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("result_id", "request_id", "plan_id", "claim_id", "testbed_id"):
            _required_identifier(errors, value, key)
        for key in (
            "request_digest",
            "plan_digest",
            "testbed_digest",
            "method_profile_digest",
        ):
            _required_digest(errors, value, key)
        _required_mapping(errors, value, "method_profile_snapshot")
        if _string(value.get("status")) not in {
            "valid",
            "invalid",
            "uncertain",
            "contradictory",
            "unavailable",
            "evidence_requested",
        }:
            errors.append("status:unsupported")
        if not isinstance(value.get("validity"), bool):
            errors.append("validity:must_be_boolean")
        for key in ("uncertainty", "coverage"):
            if not _valid_number(value.get(key), minimum=0.0, maximum=1.0):
                errors.append(f"{key}:invalid")
        _required_mapping(errors, value, "applicability_envelope")
        if not _is_rows(value.get("raw_artifact_references"), nonempty=False):
            errors.append("raw_artifact_references:must_be_rows")
        _required_mapping(errors, value, "provenance")
        for key in ("cost_usd", "duration_seconds"):
            if not _valid_number(value.get(key), minimum=0.0):
                errors.append(f"{key}:invalid")
        for key in ("blockers", "invalid_rollout_reasons"):
            if not _is_strings(value.get(key), nonempty=False):
                errors.append(f"{key}:must_be_string_list")
        _required_mapping(errors, value, "claim_ceiling")
        return errors


@dataclass(frozen=True)
class DecisionEnvelope(_ValidatedArtifact):
    SCHEMA_VERSION: ClassVar[str] = DECISION_ENVELOPE_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "decision_envelope_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("decision_id", "request_id"):
            _required_identifier(errors, value, key)
        for key in ("request_digest", "plan_digest", "testbed_digest"):
            _required_digest(errors, value, key)
        _required_string(errors, value, "decision_question")
        if _string(value.get("overall_outcome")) not in {
            "decision",
            "partial_decision",
            "abstention",
        }:
            errors.append("overall_outcome:unsupported")
        if not _is_rows(value.get("per_claim_verdicts"), nonempty=True):
            errors.append("per_claim_verdicts:missing_or_invalid")
        for key in (
            "evidence_accepted",
            "evidence_rejected",
            "unsupported_conditions",
            "cross_method_disagreements",
            "shared_dependency_warnings",
            "physical_evidence_still_required",
            "input_run_result_testbed_digests",
        ):
            if not isinstance(value.get(key), list):
                errors.append(f"{key}:must_be_list")
        for key in ("validation_envelope", "uncertainty", "claim_ceiling"):
            _required_mapping(errors, value, key)
        for key in (
            "severity_weighted_false_safe_risk",
            "evidence_coverage",
            "abstention_rate",
        ):
            if not _valid_number(value.get(key), minimum=0.0, maximum=1.0):
                errors.append(f"{key}:invalid")
        if value.get("false_reject_estimate") is not None and not _valid_number(
            value.get("false_reject_estimate"), minimum=0.0, maximum=1.0
        ):
            errors.append("false_reject_estimate:invalid")
        for key in ("decision_rationale", "next_cheapest_experiment"):
            _required_string(errors, value, key)
        if value.get("deployment_approval") is not False:
            errors.append("deployment_approval:must_be_false")
        if value.get("safety_certification") is not False:
            errors.append("safety_certification:must_be_false")
        return errors


@dataclass(frozen=True)
class PhysicalOutcomeJoin(_ValidatedArtifact):
    SCHEMA_VERSION: ClassVar[str] = PHYSICAL_OUTCOME_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "physical_outcome_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in (
            "outcome_id",
            "testbed_id",
            "testbed_version",
            "site_id",
            "task_id",
            "scenario_id",
        ):
            _required_identifier(errors, value, key)
        for key in ("testbed_digest", "prediction_digest"):
            _required_digest(errors, value, key)
        for key in (
            "condition",
            "robot_embodiment",
            "sensors",
            "controller",
            "policy_checkpoint",
            "evaluator",
            "runtime_provider",
            "prediction",
            "observed_outcome",
            "timestamps",
            "mismatch_taxonomy",
            "provenance",
        ):
            _required_mapping(errors, value, key)
        if not isinstance(value.get("owner_evidence"), list) or not value.get("owner_evidence"):
            errors.append("owner_evidence:missing_or_invalid")
        if _string(value.get("partition")) not in PARTITIONS:
            errors.append("partition:unsupported")
        _required_digest(errors, value, "runtime_digest")
        _required_digest(errors, value, "evaluator_digest")
        return errors


__all__ = [
    "DECISION_ENVELOPE_SCHEMA_VERSION",
    "DECISION_REQUEST_SCHEMA_VERSION",
    "EVIDENCE_PLAN_SCHEMA_VERSION",
    "EVIDENCE_RESULT_SCHEMA_VERSION",
    "METHOD_FAMILIES",
    "METHOD_PROFILE_SCHEMA_VERSION",
    "PHYSICAL_OUTCOME_SCHEMA_VERSION",
    "QUALIFICATION_SCHEMA_VERSION",
    "TESTBED_SCHEMA_VERSION",
    "DecisionEnvelope",
    "DecisionEvidenceContractError",
    "DecisionEvidenceRequest",
    "EvidenceMethodProfile",
    "EvidencePlan",
    "MaintainedSiteTaskTestbed",
    "NormalizedEvidenceResult",
    "PhysicalOutcomeJoin",
    "QualificationRecord",
    "canonical_digest",
    "canonical_json",
]
