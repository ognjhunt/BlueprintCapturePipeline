"""Versioned, proof-safe contracts for the Task Evaluation Supervisor.

The supervisor is a control-plane client of Blueprint's deterministic
Decision/Evidence contracts.  Its artifacts may propose work, but they never
set proof booleans, qualify evidence, grant rights, or change a Decision
Envelope.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping, Sequence, TypeVar

from ..decision_evidence_contracts import canonical_digest, canonical_json


AUTHORITY_ENVELOPE_SCHEMA_VERSION = "task_evaluation_supervisor_authority.v1"
TOOL_DESCRIPTOR_SCHEMA_VERSION = "task_evaluation_supervisor_tool.v1"
ACTION_PROPOSAL_SCHEMA_VERSION = "task_evaluation_supervisor_action_proposal.v1"
CAPABILITY_RESULT_SCHEMA_VERSION = "task_evaluation_supervisor_capability_result.v1"
EVENT_SCHEMA_VERSION = "task_evaluation_supervisor_event.v1"
INVOCATION_SCHEMA_VERSION = "task_evaluation_supervisor_invocation.v1"
RUN_SCHEMA_VERSION = "task_evaluation_supervisor_run.v1"
STATE_SCHEMA_VERSION = "task_evaluation_supervisor_state.v1"
TERMINAL_REPORT_SCHEMA_VERSION = "task_evaluation_supervisor_report.v1"
PROOF_BOUNDARY_SCHEMA_VERSION = "task_evaluation_supervisor_proof_boundary.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_KEYS = {
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


class SupervisorContractError(ValueError):
    """Fail-closed validation error with stable, sorted error identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


class AutonomyMode(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ADVISE = "advise"
    EXECUTE_NON_SPEND = "execute_non_spend"
    EXECUTE_PREAUTHORIZED = "execute_preauthorized"
    CANDIDATE_POLICY = "candidate_policy"


class CapabilityKind(str, Enum):
    CLAIM_TASK_INTERPRETER = "claim_task_interpreter"
    CAPTURE_TESTBED_SUPERVISOR = "capture_testbed_supervisor"
    EVALUATION_METHOD_ROUTER = "evaluation_method_router"
    RUNTIME_FAILURE_RECOVERY = "runtime_failure_recovery"
    SCENARIO_ADVERSARIAL_PROPOSER = "scenario_adversarial_proposer"
    POST_RUN_DIAGNOSTICIAN = "post_run_diagnostician"


class SupervisorPhase(str, Enum):
    RECEIVED = "received"
    INTERPRETING = "interpreting"
    VALIDATING = "validating"
    INSPECTING = "inspecting"
    PLANNING = "planning"
    OBSERVING = "observing"
    DIAGNOSING = "diagnosing"
    AWAITING_AUTHORIZATION = "awaiting_authorization"
    TERMINAL = "terminal"


class ProposalDisposition(str, Enum):
    SHADOW_ONLY = "shadow_only"
    REQUIRES_OPERATOR_APPROVAL = "requires_operator_approval"
    ELIGIBLE = "eligible"
    REFUSED = "refused"


_ArtifactT = TypeVar("_ArtifactT", bound="ValidatedSupervisorArtifact")


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise SupervisorContractError(["artifact:not_json_serializable"]) from exc
    if not isinstance(cloned, dict):
        raise SupervisorContractError(["artifact:not_mapping"])
    return cloned


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _secret_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            lowered = key.lower()
            path = f"{prefix}.{key}" if prefix else key
            is_secret = lowered in _SECRET_KEYS or any(
                lowered.endswith(f"_{suffix}") for suffix in _SECRET_KEYS
            )
            if is_secret and nested not in (None, "", [], {}):
                paths.append(path)
            paths.extend(_secret_paths(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_secret_paths(nested, prefix=f"{prefix}[{index}]"))
    return paths


def _identifier(errors: list[str], value: Mapping[str, Any], key: str) -> None:
    text = _string(value.get(key))
    if not text:
        errors.append(f"{key}:missing")
    elif not _IDENTIFIER.fullmatch(text):
        errors.append(f"{key}:invalid")


def _digest(errors: list[str], value: Mapping[str, Any], key: str) -> None:
    text = _string(value.get(key))
    if not text:
        errors.append(f"{key}:missing")
    elif not _SHA256.fullmatch(text):
        errors.append(f"{key}:invalid_sha256")


def _strings(value: Any, *, nonempty: bool = False) -> bool:
    valid = isinstance(value, list) and all(bool(_string(item)) for item in value)
    return valid and (bool(value) or not nonempty)


def _rows(value: Any, *, nonempty: bool = False) -> bool:
    valid = isinstance(value, list) and all(isinstance(item, Mapping) for item in value)
    return valid and (bool(value) or not nonempty)


def _number(value: Any, *, minimum: float = 0.0) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
        return math.isfinite(number) and number >= minimum
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ValidatedSupervisorArtifact:
    """Immutable JSON-backed artifact with canonical digest validation."""

    _canonical: str
    SCHEMA_VERSION: ClassVar[str]
    DIGEST_FIELD: ClassVar[str]

    @classmethod
    def from_mapping(cls: type[_ArtifactT], value: Mapping[str, Any]) -> _ArtifactT:
        if not isinstance(value, Mapping):
            raise SupervisorContractError(["artifact:not_mapping"])
        normalized = _clone(value)
        errors: list[str] = []
        if _string(normalized.get("schema_version")) != cls.SCHEMA_VERSION:
            errors.append(f"schema_version:must_be:{cls.SCHEMA_VERSION}")
        errors.extend(cls._validation_errors(normalized))
        errors.extend(f"secret_value_forbidden:{path}" for path in _secret_paths(normalized))
        expected = canonical_digest(normalized, digest_field=cls.DIGEST_FIELD)
        supplied = _string(normalized.get(cls.DIGEST_FIELD))
        if supplied and supplied != expected:
            errors.append(f"{cls.DIGEST_FIELD}:mismatch")
        if errors:
            raise SupervisorContractError(errors)
        normalized[cls.DIGEST_FIELD] = expected
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
class AuthorityEnvelope(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = AUTHORITY_ENVELOPE_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "authority_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        _identifier(errors, value, "authority_id")
        try:
            mode = AutonomyMode(_string(value.get("mode")))
        except ValueError:
            errors.append("mode:unsupported")
            mode = None
        if not _strings(value.get("allowed_tool_ids")):
            errors.append("allowed_tool_ids:must_be_string_list")
        for key in ("max_cost_usd", "max_duration_seconds", "max_retries"):
            if not _number(value.get(key)):
                errors.append(f"{key}:invalid")
        if "agent_inference_budget_usd" in value and not _number(
            value.get("agent_inference_budget_usd")
        ):
            errors.append("agent_inference_budget_usd:invalid")
        for key in (
            "agent_inference_allowed",
            "action_spend_allowed",
            "external_processing_allowed",
        ):
            if key in value and not isinstance(value.get(key), bool):
                errors.append(f"{key}:must_be_boolean")
        if (
            value.get("agent_inference_allowed") is True
            and float(value.get("agent_inference_budget_usd") or 0) <= 0
        ):
            errors.append("agent_inference_requires_positive_budget")
        if (
            value.get("agent_inference_allowed") is True
            and value.get("external_processing_allowed") is not True
        ):
            errors.append("agent_inference_requires_external_processing_authority")
        if value.get("action_spend_allowed") is True:
            if mode is not AutonomyMode.EXECUTE_PREAUTHORIZED:
                errors.append("action_spend_allowed:wrong_mode")
            if float(value.get("max_cost_usd") or 0) <= 0:
                errors.append("action_spend_allowed:positive_cost_required")
            _digest(errors, value, "preauthorization_receipt_digest")
            if not _string(value.get("expires_at")):
                errors.append("preauthorization_expiry_missing")
        if mode is AutonomyMode.DISABLED and float(value.get("max_cost_usd") or 0) != 0:
            errors.append("disabled_mode_cost_must_be_zero")
        for key in (
            "proof_mutation_allowed",
            "rights_mutation_allowed",
            "budget_mutation_allowed",
            "hidden_labels_accessible",
            "physical_action_allowed",
        ):
            if value.get(key) is not False:
                errors.append(f"{key}:must_be_false")
        if not isinstance(value.get("immutable_input_digests"), list):
            errors.append("immutable_input_digests:must_be_list")
        return errors


@dataclass(frozen=True)
class ToolDescriptor(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = TOOL_DESCRIPTOR_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "tool_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("tool_id", "version"):
            _identifier(errors, value, key)
        for key in ("category", "mutability", "idempotency", "safety_level"):
            if not _string(value.get(key)):
                errors.append(f"{key}:missing")
        if _string(value.get("mutability")) not in {
            "read_only",
            "reversible_mutation",
            "external_side_effect",
        }:
            errors.append("mutability:unsupported")
        for key in ("input_schema", "output_schema", "required_authority"):
            if not isinstance(value.get(key), Mapping) or not value.get(key):
                errors.append(f"{key}:missing_or_empty")
        for key in ("max_cost_usd", "timeout_seconds", "max_retries"):
            if not _number(value.get(key)):
                errors.append(f"{key}:invalid")
        if not _strings(value.get("allowed_modes"), nonempty=True):
            errors.append("allowed_modes:missing_or_invalid")
        else:
            for mode in value.get("allowed_modes") or []:
                try:
                    AutonomyMode(_string(mode))
                except ValueError:
                    errors.append(f"allowed_modes:unsupported:{mode}")
        if _string(value.get("proof_effect")) not in {"none", "supporting_artifact_only"}:
            errors.append("proof_effect:unsupported")
        if not isinstance(value.get("evidence_requirements"), list):
            errors.append("evidence_requirements:must_be_list")
        if not isinstance(value.get("expected_artifacts"), list):
            errors.append("expected_artifacts:must_be_list")
        return errors


@dataclass(frozen=True)
class ActionProposal(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = ACTION_PROPOSAL_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "proposal_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("proposal_id", "run_id"):
            _identifier(errors, value, key)
        try:
            CapabilityKind(_string(value.get("capability")))
        except ValueError:
            errors.append("capability:unsupported")
        if not _string(value.get("action_type")):
            errors.append("action_type:missing")
        if value.get("tool_id") is not None and not _string(value.get("tool_id")):
            errors.append("tool_id:invalid")
        if not isinstance(value.get("parameters"), Mapping):
            errors.append("parameters:must_be_mapping")
        if not _strings(value.get("reasons"), nonempty=True):
            errors.append("reasons:missing_or_invalid")
        if not isinstance(value.get("evidence_refs"), list):
            errors.append("evidence_refs:must_be_list")
        if not _number(value.get("estimated_cost_usd")):
            errors.append("estimated_cost_usd:invalid")
        if _string(value.get("requested_proof_effect")) != "none":
            errors.append("requested_proof_effect:must_be_none")
        try:
            ProposalDisposition(_string(value.get("disposition")))
        except ValueError:
            errors.append("disposition:unsupported")
        return errors


@dataclass(frozen=True)
class CapabilityResult(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = CAPABILITY_RESULT_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "capability_result_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("result_id", "run_id"):
            _identifier(errors, value, key)
        try:
            CapabilityKind(_string(value.get("capability")))
        except ValueError:
            errors.append("capability:unsupported")
        if _string(value.get("status")) not in {"proposed", "abstained", "blocked", "disabled"}:
            errors.append("status:unsupported")
        if not isinstance(value.get("artifact"), Mapping):
            errors.append("artifact:must_be_mapping")
        if not _rows(value.get("proposals")):
            errors.append("proposals:must_be_rows")
        if not _strings(value.get("blockers")):
            errors.append("blockers:must_be_string_list")
        if not isinstance(value.get("evidence_refs"), list):
            errors.append("evidence_refs:must_be_list")
        if value.get("authoritative") is not False:
            errors.append("authoritative:must_be_false")
        if value.get("proof_booleans_mutable") is not False:
            errors.append("proof_booleans_mutable:must_be_false")
        if _string(value.get("proof_effect")) != "none":
            errors.append("proof_effect:must_be_none")
        return errors


@dataclass(frozen=True)
class SupervisorEvent(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = EVENT_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "event_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("event_id", "run_id"):
            _identifier(errors, value, key)
        if not isinstance(value.get("sequence"), int) or value.get("sequence", -1) < 0:
            errors.append("sequence:invalid")
        if not _string(value.get("event_type")):
            errors.append("event_type:missing")
        if value.get("previous_event_digest") is not None and not _SHA256.fullmatch(
            _string(value.get("previous_event_digest"))
        ):
            errors.append("previous_event_digest:invalid")
        if value.get("payload_digest") is not None and not _SHA256.fullmatch(
            _string(value.get("payload_digest"))
        ):
            errors.append("payload_digest:invalid")
        try:
            SupervisorPhase(_string(value.get("phase")))
        except ValueError:
            errors.append("phase:unsupported")
        if _string(value.get("proof_effect")) != "none":
            errors.append("proof_effect:must_be_none")
        if not _string(value.get("generated_at")):
            errors.append("generated_at:missing")
        return errors


@dataclass(frozen=True)
class AgentInvocationManifest(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = INVOCATION_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "invocation_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("invocation_id", "run_id"):
            _identifier(errors, value, key)
        try:
            CapabilityKind(_string(value.get("capability")))
        except ValueError:
            errors.append("capability:unsupported")
        for key in (
            "provider",
            "agent_harness",
            "adapter_id",
            "adapter_version",
            "generated_at",
            "cost_status",
            "uncertainty",
        ):
            if not _string(value.get(key)):
                errors.append(f"{key}:missing")
        for key in ("instruction_digest", "tool_registry_digest", "structured_output_digest"):
            _digest(errors, value, key)
        _digest(errors, value, "authority_digest")
        if not isinstance(value.get("input_artifact_digests"), list):
            errors.append("input_artifact_digests:must_be_list")
        if not isinstance(value.get("budget_state"), Mapping):
            errors.append("budget_state:must_be_mapping")
        else:
            for key in (
                "max_cost_usd",
                "reported_cost_usd",
                "cumulative_reserved_cost_usd",
                "remaining_unreserved_usd",
            ):
                if not _number((value.get("budget_state") or {}).get(key)):
                    errors.append(f"budget_state.{key}:invalid")
        if _string(value.get("validation_status")) not in {"accepted_as_proposal", "refused"}:
            errors.append("validation_status:unsupported")
        if not _number(value.get("cost_usd")) or not _number(value.get("latency_seconds")):
            errors.append("cost_or_latency:invalid")
        if _string(value.get("action_taken")) not in {
            "none_shadow_mode",
            "registered_read_only_tool_calls",
            "registered_non_spend_actions_executed",
            "registered_preauthorized_action_attempted",
        }:
            errors.append("action_taken:unsupported")
        if not isinstance(value.get("refusal"), bool):
            errors.append("refusal:must_be_boolean")
        if not isinstance(value.get("usage"), Mapping):
            errors.append("usage:must_be_mapping")
        if not isinstance(value.get("tool_observation_references"), list):
            errors.append("tool_observation_references:must_be_list")
        _digest(errors, value, "parent_event_digest")
        if _string(value.get("proof_effect")) != "none":
            errors.append("proof_effect:must_be_none")
        return errors


@dataclass(frozen=True)
class SupervisorRun(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = RUN_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "supervisor_run_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        _identifier(errors, value, "run_id")
        if not _string(value.get("customer_question")):
            errors.append("customer_question:missing")
        try:
            AutonomyMode(_string(value.get("mode")))
        except ValueError:
            errors.append("mode:unsupported")
        for key in ("authority_digest", "tool_registry_digest", "proof_boundary_digest"):
            _digest(errors, value, key)
        if not isinstance(value.get("input_artifact_digests"), list):
            errors.append("input_artifact_digests:must_be_list")
        if not _strings(value.get("capabilities"), nonempty=True):
            errors.append("capabilities:missing_or_invalid")
        if _string(value.get("status")) not in {"initialized", "disabled"}:
            errors.append("status:unsupported")
        if not _string(value.get("generated_at")):
            errors.append("generated_at:missing")
        return errors


@dataclass(frozen=True)
class SupervisorState(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = STATE_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "supervisor_state_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        _identifier(errors, value, "run_id")
        try:
            AutonomyMode(_string(value.get("mode")))
        except ValueError:
            errors.append("mode:unsupported")
        try:
            SupervisorPhase(_string(value.get("phase")))
        except ValueError:
            errors.append("phase:unsupported")
        if not isinstance(value.get("next_sequence"), int) or value.get("next_sequence", -1) < 0:
            errors.append("next_sequence:invalid")
        if value.get("last_event_digest") is not None and not _SHA256.fullmatch(
            _string(value.get("last_event_digest"))
        ):
            errors.append("last_event_digest:invalid")
        if not _strings(value.get("completed_capabilities")):
            errors.append("completed_capabilities:must_be_string_list")
        if not isinstance(value.get("terminal"), bool):
            errors.append("terminal:must_be_boolean")
        for key in ("spent_cost_usd", "remaining_cost_usd"):
            if not _number(value.get(key)):
                errors.append(f"{key}:invalid")
        if value.get("proof_state_mutated_by_agent") is not False:
            errors.append("proof_state_mutated_by_agent:must_be_false")
        return errors


@dataclass(frozen=True)
class TerminalSupervisorReport(ValidatedSupervisorArtifact):
    SCHEMA_VERSION: ClassVar[str] = TERMINAL_REPORT_SCHEMA_VERSION
    DIGEST_FIELD: ClassVar[str] = "terminal_report_digest"

    @classmethod
    def _validation_errors(cls, value: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        _identifier(errors, value, "run_id")
        if _string(value.get("status")) not in {
            "shadow_complete",
            "advise_complete",
            "non_spend_complete",
            "preauthorized_complete",
            "preauthorized_complete_with_failures",
            "disabled",
            "blocked",
        }:
            errors.append("status:unsupported")
        if not _rows(value.get("capability_results")):
            errors.append("capability_results:must_be_rows")
        if not _rows(value.get("invocation_manifests")):
            errors.append("invocation_manifests:must_be_rows")
        if not isinstance(value.get("event_count"), int) or value.get("event_count", -1) < 0:
            errors.append("event_count:invalid")
        for key in ("last_event_digest", "proof_boundary_digest", "tool_registry_digest"):
            _digest(errors, value, key)
        if value.get("authoritative_decision_produced_by_agent") is not False:
            errors.append("authoritative_decision_produced_by_agent:must_be_false")
        if value.get("proof_state_mutated_by_agent") is not False:
            errors.append("proof_state_mutated_by_agent:must_be_false")
        if not isinstance(value.get("actions_executed"), bool):
            errors.append("actions_executed:must_be_boolean")
        elif value.get("actions_executed") is True and _string(value.get("mode")) not in {
            AutonomyMode.EXECUTE_NON_SPEND.value,
            AutonomyMode.EXECUTE_PREAUTHORIZED.value,
        }:
            errors.append("actions_executed:not_allowed_in_mode")
        for key in (
            "registered_tool_reads_executed",
            "registered_non_spend_actions_executed",
            "registered_preauthorized_actions_executed",
        ):
            if not isinstance(value.get(key), int) or value.get(key, -1) < 0:
                errors.append(f"{key}:invalid")
        if not _string(value.get("customer_report_path")):
            errors.append("customer_report_path:missing")
        _digest(errors, value, "customer_report_digest")
        inference_spend = value.get("inference_spend")
        if not isinstance(inference_spend, Mapping):
            errors.append("inference_spend:must_be_mapping")
        else:
            for key in (
                "budget_usd",
                "reserved_max_cost_usd",
                "reported_cost_usd",
                "remaining_unreserved_usd",
            ):
                if not _number(inference_spend.get(key)):
                    errors.append(f"inference_spend.{key}:invalid")
            if not isinstance(inference_spend.get("live_invocation_count"), int):
                errors.append("inference_spend.live_invocation_count:invalid")
        action_spend = value.get("action_spend")
        if not isinstance(action_spend, Mapping):
            errors.append("action_spend:must_be_mapping")
        else:
            for key in (
                "authorized_max_cost_usd",
                "reported_actual_cost_usd",
                "reported_duration_seconds",
            ):
                if not _number(action_spend.get(key)):
                    errors.append(f"action_spend.{key}:invalid")
            if float(action_spend.get("reported_actual_cost_usd") or 0.0) > float(
                action_spend.get("authorized_max_cost_usd") or 0.0
            ):
                errors.append("action_spend:exceeds_authority")
        if not _string(value.get("generated_at")):
            errors.append("generated_at:missing")
        return errors


def proof_boundary() -> dict[str, Any]:
    """Return the invariant proof boundary shared by every supervisor mode."""

    value = {
        "schema_version": PROOF_BOUNDARY_SCHEMA_VERSION,
        "artifact_purpose": "agentic_orchestration_and_advisory_control_plane",
        "agent_controls": ["search", "sequencing", "explanation", "recovery_proposals"],
        "deterministic_kernel_controls": [
            "schema_validation",
            "artifact_hashes",
            "raw_capture_integrity",
            "rights_and_privacy",
            "authority_grants",
            "budgets_and_spend",
            "provider_admission",
            "frozen_splits",
            "hidden_labels",
            "success_predicates",
            "evaluator_thresholds",
            "claim_ceilings",
            "evidence_qualification",
            "proof_state_transitions",
            "final_decision",
        ],
        "proof_booleans_mutable_by_agent": False,
        "rights_mutable_by_agent": False,
        "budget_mutable_by_agent": False,
        "hidden_labels_accessible_by_agent": False,
        "raw_capture_mutable_by_agent": False,
        "deployment_approval_allowed": False,
        "safety_certification_allowed": False,
        "physical_success_claim_allowed_without_accepted_physical_evidence": False,
        "agent_output_is_accepted_evidence": False,
        "agent_may_grade_own_candidate": False,
        "policy_ranking_thesis_verdict": "thesis_not_supported",
    }
    value["proof_boundary_digest"] = canonical_digest(value, digest_field="proof_boundary_digest")
    return value


__all__ = [
    "ACTION_PROPOSAL_SCHEMA_VERSION",
    "AUTHORITY_ENVELOPE_SCHEMA_VERSION",
    "CAPABILITY_RESULT_SCHEMA_VERSION",
    "EVENT_SCHEMA_VERSION",
    "INVOCATION_SCHEMA_VERSION",
    "PROOF_BOUNDARY_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "TERMINAL_REPORT_SCHEMA_VERSION",
    "TOOL_DESCRIPTOR_SCHEMA_VERSION",
    "ActionProposal",
    "AgentInvocationManifest",
    "AuthorityEnvelope",
    "AutonomyMode",
    "CapabilityKind",
    "CapabilityResult",
    "ProposalDisposition",
    "SupervisorContractError",
    "SupervisorEvent",
    "SupervisorPhase",
    "SupervisorRun",
    "SupervisorState",
    "TerminalSupervisorReport",
    "ToolDescriptor",
    "ValidatedSupervisorArtifact",
    "proof_boundary",
]
