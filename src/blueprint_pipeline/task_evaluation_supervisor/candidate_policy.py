"""Phase 4 frozen candidate PolicyAdapter and hidden-evaluation separation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from ..common import read_json, write_json
from ..decision_evidence_contracts import canonical_digest
from ..evaluation_run_contract import validate_evaluation_run_spec
from ..paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .phase2_artifacts import AUTHORIZATION_RECEIPT_SCHEMA_VERSION


CANDIDATE_POLICY_MANIFEST_SCHEMA_VERSION = "task_evaluation_candidate_policy_manifest.v1"
CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION = "task_evaluation_candidate_policy_suite.v1"
CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION = "task_evaluation_candidate_policy_execution.v1"
CANDIDATE_COST_RESERVATION_SCHEMA_VERSION = "candidate_policy_cost_reservation.v1"
CANDIDATE_COST_SETTLEMENT_SCHEMA_VERSION = "candidate_policy_cost_settlement.v1"
CANDIDATE_COST_RECONCILIATION_SCHEMA_VERSION = (
    "task_evaluation_candidate_policy_cost_reconciliation.v1"
)
_STACK_TYPES = {"direct_policy", "decomposed_planner_policy", "verify_recover_supervisor"}
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CandidatePolicyError(ValueError):
    """Raised when a candidate could access or influence held-out evaluation."""


class CandidatePolicyRuntime(Protocol):
    candidate_id: str
    candidate_policy_manifest_digest: str
    runtime_configuration_digest: str
    provider_id: str
    provider_execution_planned: bool
    cost_accounting_authoritative: bool
    cost_authority_binding_digest: str | None
    paid_resource_class: str | None
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None

    def execute(
        self,
        *,
        evaluation_run_spec: Mapping[str, Any],
        output_dir: Path,
    ) -> Mapping[str, Any]: ...


class CandidateCostAuthority(Protocol):
    """Blueprint-owned metering boundary, separate from candidate code."""

    authority_id: str
    provider_id: str
    paid_resource_class: str
    cost_authority_binding_digest: str

    def reserve(
        self,
        *,
        candidate_id: str,
        candidate_evaluation_suite_digest: str,
        authorization_receipt_digest: str,
        max_cost_usd: float,
    ) -> Mapping[str, Any]: ...

    def settle(
        self,
        *,
        reservation: Mapping[str, Any],
        runtime_result: Mapping[str, Any] | None,
        runtime_exception_type: str | None,
    ) -> Mapping[str, Any]: ...


class IndependentCandidateEvaluator(Protocol):
    provider_id: str
    evaluator_digest: str

    def evaluate(
        self,
        *,
        candidate_id: str,
        trace: Mapping[str, Any],
        hidden_evaluation_manifest: Mapping[str, Any],
        success_predicate_digest: str,
    ) -> Mapping[str, Any]: ...


def _digest(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _SHA256_DIGEST.fullmatch(text):
        raise CandidatePolicyError(f"{field}:invalid_digest")
    return text


def _frozen_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidatePolicyError("candidate_frozen_at_invalid") from exc
    if parsed.tzinfo is None:
        raise CandidatePolicyError("candidate_frozen_at_timezone_required")
    return parsed.astimezone(timezone.utc).isoformat()


def _authorization_time(value: Any, *, field: str) -> datetime:
    text = str(value or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CandidatePolicyError(f"{field}:invalid") from exc
    if parsed.tzinfo is None:
        raise CandidatePolicyError(f"{field}:timezone_required")
    return parsed.astimezone(timezone.utc)


def _validate_cost_reservation(
    value: Mapping[str, Any],
    *,
    authority: CandidateCostAuthority,
    candidate_id: str,
    suite_digest: str,
    authorization_digest: str,
    max_cost_usd: float,
) -> dict[str, Any]:
    reservation = dict(value)
    expected = canonical_digest(reservation, digest_field="cost_reservation_digest")
    reserved_cost_value = reservation.get("reserved_max_cost_usd")
    if reserved_cost_value is None:
        raise CandidatePolicyError("candidate_cost_reservation_invalid")
    try:
        reserved_cost = float(reserved_cost_value)
    except (TypeError, ValueError) as exc:
        raise CandidatePolicyError("candidate_cost_reservation_invalid") from exc
    if (
        reservation.get("schema_version") != CANDIDATE_COST_RESERVATION_SCHEMA_VERSION
        or reservation.get("cost_reservation_digest") != expected
        or reservation.get("status") != "reserved"
        or reservation.get("authority_id") != authority.authority_id
        or reservation.get("provider_id") != authority.provider_id
        or reservation.get("paid_resource_class") != authority.paid_resource_class
        or reservation.get("cost_authority_binding_digest")
        != authority.cost_authority_binding_digest
        or reservation.get("candidate_id") != candidate_id
        or reservation.get("candidate_evaluation_suite_digest") != suite_digest
        or reservation.get("authorization_receipt_digest") != authorization_digest
        or not math.isfinite(reserved_cost)
        or reserved_cost != max_cost_usd
        or reservation.get("candidate_reported_usage_is_authoritative") is not False
        or reservation.get("proof_effect") != "none"
    ):
        raise CandidatePolicyError("candidate_cost_reservation_invalid")
    return reservation


def _validate_cost_settlement(
    value: Mapping[str, Any],
    *,
    authority: CandidateCostAuthority,
    reservation: Mapping[str, Any],
) -> dict[str, Any]:
    settlement = dict(value)
    expected = canonical_digest(settlement, digest_field="cost_settlement_digest")
    status = settlement.get("status")
    cost_is_final = settlement.get("cost_is_final")
    actual_cost_value = settlement.get("actual_cost_usd")
    actual_cost: float | None = None
    if actual_cost_value is not None:
        try:
            actual_cost = float(actual_cost_value)
        except (TypeError, ValueError) as exc:
            raise CandidatePolicyError("candidate_cost_settlement_invalid") from exc
    if (
        settlement.get("schema_version") != CANDIDATE_COST_SETTLEMENT_SCHEMA_VERSION
        or settlement.get("cost_settlement_digest") != expected
        or status not in {"reconciled", "reconciliation_required"}
        or settlement.get("authority_id") != authority.authority_id
        or settlement.get("provider_id") != authority.provider_id
        or settlement.get("paid_resource_class") != authority.paid_resource_class
        or settlement.get("cost_authority_binding_digest")
        != authority.cost_authority_binding_digest
        or settlement.get("candidate_id") != reservation.get("candidate_id")
        or settlement.get("cost_reservation_digest") != reservation.get("cost_reservation_digest")
        or not isinstance(cost_is_final, bool)
        or (status == "reconciled") is not cost_is_final
        or (cost_is_final and actual_cost is None)
        or (
            actual_cost is not None
            and (
                not math.isfinite(actual_cost)
                or actual_cost < 0
                or actual_cost > float(reservation["reserved_max_cost_usd"])
            )
        )
        or settlement.get("candidate_reported_cost_accepted") is not False
        or settlement.get("proof_effect") != "none"
    ):
        raise CandidatePolicyError("candidate_cost_settlement_invalid")
    return settlement


def _validate_execution_authorization(
    receipt: Mapping[str, Any] | None,
    *,
    suite_digest: str,
    hidden_digest: str,
    specs: Sequence[Mapping[str, Any]],
    runtimes: Mapping[str, CandidatePolicyRuntime],
    executed_at: str | None,
) -> dict[str, Any]:
    if receipt is None:
        raise CandidatePolicyError("candidate_execution_authorization_missing")
    value = dict(receipt)
    expected = canonical_digest(value, digest_field="authorization_receipt_digest")
    if (
        value.get("schema_version") != AUTHORIZATION_RECEIPT_SCHEMA_VERSION
        or value.get("authorization_receipt_digest") != expected
    ):
        raise CandidatePolicyError("candidate_execution_authorization_digest_mismatch")
    if (
        value.get("approved") is not True
        or value.get("issued_by_agent") is not False
        or value.get("granted_tool_id") != "execute_candidate_policy_suite"
        or value.get("proof_effect") != "none"
        or not str(value.get("operator_id") or "").strip()
    ):
        raise CandidatePolicyError("candidate_execution_not_operator_authorized")
    _digest(
        value.get("authorization_request_digest"),
        field="candidate_authorization_request_digest",
    )
    if tuple(sorted(str(row) for row in value.get("immutable_input_digests") or [])) != tuple(
        sorted((suite_digest, hidden_digest))
    ):
        raise CandidatePolicyError("candidate_execution_inputs_not_authorized")
    now = _authorization_time(
        executed_at or datetime.now(timezone.utc).isoformat(),
        field="candidate_execution_at",
    )
    issued = _authorization_time(value.get("issued_at"), field="candidate_authority_issued_at")
    expires = _authorization_time(value.get("expires_at"), field="candidate_authority_expires_at")
    granted_ttl_value = value.get("granted_ttl_seconds")
    if isinstance(granted_ttl_value, bool) or granted_ttl_value is None:
        raise CandidatePolicyError("candidate_execution_envelope_invalid")
    try:
        granted_ttl = float(granted_ttl_value)
    except (TypeError, ValueError) as exc:
        raise CandidatePolicyError("candidate_execution_envelope_invalid") from exc
    if (
        not math.isfinite(granted_ttl)
        or granted_ttl <= 0
        or expires <= issued
        or (expires - issued).total_seconds() > granted_ttl
    ):
        raise CandidatePolicyError("candidate_execution_authority_ttl_invalid")
    if now < issued or now >= expires:
        raise CandidatePolicyError("candidate_execution_authority_inactive")

    candidate_ids = tuple(sorted(runtimes))
    granted_actions = tuple(sorted(str(row) for row in value.get("granted_action_ids") or []))
    if granted_actions != candidate_ids:
        raise CandidatePolicyError("candidate_execution_actions_not_authorized")
    paid_providers = tuple(
        sorted(
            {
                runtime.provider_id
                for runtime in runtimes.values()
                if runtime.provider_execution_planned
            }
        )
    )
    granted_providers = tuple(sorted(str(row) for row in value.get("granted_provider_ids") or []))
    if granted_providers != paid_providers:
        raise CandidatePolicyError("candidate_execution_providers_not_authorized")
    if isinstance(value.get("granted_max_cost_usd"), bool) or isinstance(
        value.get("granted_retry_count"), bool
    ):
        raise CandidatePolicyError("candidate_execution_envelope_invalid")
    try:
        authorized_cost = float(value.get("granted_max_cost_usd") or 0.0)
        authorized_retries = int(value.get("granted_retry_count") or 0)
    except (TypeError, ValueError) as exc:
        raise CandidatePolicyError("candidate_execution_envelope_invalid") from exc
    required_cost = sum(
        float((dict(spec.get("policy_adapter") or {})).get("max_cost_usd") or 0.0) for spec in specs
    )
    if (
        not math.isfinite(authorized_cost)
        or authorized_cost < required_cost
        or authorized_retries
        < max(
            int((dict(spec.get("policy_adapter") or {})).get("retry_limit") or 0) for spec in specs
        )
    ):
        raise CandidatePolicyError("candidate_execution_envelope_insufficient")
    return value


def freeze_candidate_policy_manifest(
    *,
    candidate_id: str,
    stack_type: str,
    code_digest: str,
    model_provider: str,
    model_id: str,
    model_version: str,
    prompt_digest: str,
    tool_registry_digest: str,
    memory_skill_snapshot_digest: str,
    runtime_configuration_digest: str,
    max_cost_usd: float,
    retry_limit: int,
    observation_schema_ref: str,
    action_schema_ref: str,
    frozen_at: str,
) -> dict[str, Any]:
    if stack_type not in _STACK_TYPES:
        raise CandidatePolicyError("candidate_stack_type_invalid")
    if (
        not _CANDIDATE_ID.fullmatch(candidate_id)
        or not model_provider.strip()
        or not model_id.strip()
        or not model_version.strip()
        or not observation_schema_ref.strip()
        or not action_schema_ref.strip()
        or not frozen_at.strip()
    ):
        raise CandidatePolicyError("candidate_manifest_missing_fields")
    if (
        isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, (int, float))
        or not math.isfinite(float(max_cost_usd))
        or float(max_cost_usd) < 0
        or isinstance(retry_limit, bool)
        or not isinstance(retry_limit, int)
        or retry_limit < 0
    ):
        raise CandidatePolicyError("candidate_budget_or_retry_invalid")
    value: dict[str, Any] = {
        "schema_version": CANDIDATE_POLICY_MANIFEST_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "stack_type": stack_type,
        "code_digest": _digest(code_digest, field="code_digest"),
        "model_provider": model_provider,
        "model_id": model_id,
        "model_version": model_version,
        "prompt_digest": _digest(prompt_digest, field="prompt_digest"),
        "tool_registry_digest": _digest(tool_registry_digest, field="tool_registry_digest"),
        "memory_skill_snapshot_digest": _digest(
            memory_skill_snapshot_digest,
            field="memory_skill_snapshot_digest",
        ),
        "runtime_configuration_digest": _digest(
            runtime_configuration_digest,
            field="runtime_configuration_digest",
        ),
        "max_cost_usd": float(max_cost_usd),
        "retry_limit": retry_limit,
        "observation_schema_ref": observation_schema_ref,
        "action_schema_ref": action_schema_ref,
        "frozen_at": _frozen_time(frozen_at),
        "frozen_before_hidden_evaluation": True,
        "hidden_labels_included": False,
        "evaluator_configuration_included": False,
        "success_predicate_mutable_by_candidate": False,
        "candidate_may_grade_itself": False,
        "proof_effect": "none",
    }
    value["candidate_policy_manifest_digest"] = canonical_digest(
        value,
        digest_field="candidate_policy_manifest_digest",
    )
    return value


@dataclass(frozen=True)
class FrozenAgenticPolicyAdapter:
    """Replaceable adapter payload supplied to candidate execution only."""

    manifest: Mapping[str, Any]
    adapter_id: str = "blueprint_agentic_candidate_policy"
    adapter_version: str = "1"

    def __post_init__(self) -> None:
        if self.manifest.get("schema_version") != CANDIDATE_POLICY_MANIFEST_SCHEMA_VERSION:
            raise CandidatePolicyError("candidate_policy_manifest_schema_invalid")
        expected = canonical_digest(
            self.manifest,
            digest_field="candidate_policy_manifest_digest",
        )
        if self.manifest.get("candidate_policy_manifest_digest") != expected:
            raise CandidatePolicyError("candidate_policy_manifest_digest_mismatch")
        _digest(
            self.manifest.get("runtime_configuration_digest"),
            field="runtime_configuration_digest",
        )
        if self.manifest.get("frozen_before_hidden_evaluation") is not True:
            raise CandidatePolicyError("candidate_policy_not_frozen")
        if self.manifest.get("hidden_labels_included") is not False:
            raise CandidatePolicyError("candidate_policy_contains_hidden_labels")
        if (
            self.manifest.get("evaluator_configuration_included") is not False
            or self.manifest.get("success_predicate_mutable_by_candidate") is not False
            or self.manifest.get("candidate_may_grade_itself") is not False
            or self.manifest.get("proof_effect") != "none"
        ):
            raise CandidatePolicyError("candidate_policy_authority_boundary_invalid")

    def to_policy_adapter_mapping(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "policy_id": self.manifest["candidate_id"],
            "candidate_policy_manifest_digest": self.manifest["candidate_policy_manifest_digest"],
            "runtime_configuration_digest": self.manifest["runtime_configuration_digest"],
            "stack_type": self.manifest["stack_type"],
            "observation_schema_ref": self.manifest["observation_schema_ref"],
            "action_schema_ref": self.manifest["action_schema_ref"],
            "max_cost_usd": self.manifest["max_cost_usd"],
            "retry_limit": self.manifest["retry_limit"],
            "hidden_labels_included": False,
            "evaluator_authority": False,
            "proof_authority": False,
        }


def compile_neutral_candidate_policy_suite(
    *,
    base_evaluation_run_spec: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    frozen_scenario_manifest: Mapping[str, Any],
    evaluator_provider_id: str,
) -> dict[str, Any]:
    if not evaluator_provider_id.strip():
        raise CandidatePolicyError("evaluator_provider_id_missing")
    scenario_digest = canonical_digest(
        frozen_scenario_manifest,
        digest_field="frozen_scenario_manifest_digest",
    )
    if frozen_scenario_manifest.get("frozen_scenario_manifest_digest") != scenario_digest:
        raise CandidatePolicyError("frozen_scenario_manifest_digest_mismatch")
    if frozen_scenario_manifest.get("frozen") is not True:
        raise CandidatePolicyError("scenario_manifest_not_frozen")
    if frozen_scenario_manifest.get("hidden_labels_included") is not False:
        raise CandidatePolicyError("scenario_manifest_exposes_hidden_labels")
    scenario_ids = [str(row) for row in frozen_scenario_manifest.get("scenario_ids") or []]
    if (
        not scenario_ids
        or len(scenario_ids) != len(set(scenario_ids))
        or any(not row for row in scenario_ids)
    ):
        raise CandidatePolicyError("frozen_scenario_ids_invalid")
    required_types = _STACK_TYPES
    candidate_types = {str(row.get("stack_type") or "") for row in candidates}
    if candidate_types != required_types or len(candidates) != len(required_types):
        raise CandidatePolicyError("neutral_suite_requires_three_stack_types")
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CandidatePolicyError("neutral_suite_candidate_id_duplicate")
    observation_schema_refs = {str(row.get("observation_schema_ref") or "") for row in candidates}
    action_schema_refs = {str(row.get("action_schema_ref") or "") for row in candidates}
    if (
        len(observation_schema_refs) != 1
        or "" in observation_schema_refs
        or len(action_schema_refs) != 1
        or "" in action_schema_refs
    ):
        raise CandidatePolicyError("neutral_suite_interface_mismatch")

    evaluator_digest = _digest(
        frozen_scenario_manifest.get("evaluator_digest"),
        field="evaluator_digest",
    )
    success_digest = _digest(
        frozen_scenario_manifest.get("success_predicate_digest"),
        field="success_predicate_digest",
    )
    compiled: list[dict[str, Any]] = []
    for manifest in sorted(candidates, key=lambda row: str(row.get("candidate_id") or "")):
        adapter = FrozenAgenticPolicyAdapter(manifest)
        if str(manifest.get("model_provider")) == evaluator_provider_id:
            raise CandidatePolicyError("candidate_provider_self_grading_forbidden")
        spec = copy.deepcopy(dict(base_evaluation_run_spec))
        spec["run_id"] = f"{base_evaluation_run_spec['run_id']}-{manifest['candidate_id']}"
        spec["policy_adapter"] = adapter.to_policy_adapter_mapping()
        task_pack = dict(spec.get("task_scenario_pack") or {})
        task_pack["frozen_scenario_manifest_digest"] = scenario_digest
        task_pack["scenario_ids"] = list(frozen_scenario_manifest.get("scenario_ids") or [])
        task_pack["hidden_labels_included"] = False
        spec["task_scenario_pack"] = task_pack
        proof_contract = dict(spec.get("proof_contract") or {})
        proof_contract["evaluator_digest"] = evaluator_digest
        proof_contract["success_predicate_digest"] = success_digest
        proof_contract["candidate_may_modify"] = False
        spec["proof_contract"] = proof_contract
        metadata = dict(spec.get("metadata") or {})
        metadata.update(
            {
                "candidate_policy_manifest_digest": manifest["candidate_policy_manifest_digest"],
                "frozen_scenario_manifest_digest": scenario_digest,
                "evaluator_provider_id": evaluator_provider_id,
                "candidate_results_visible_to_scenario_generator": False,
                "candidate_self_grading": False,
                "simulation_only_unless_physical_evidence_joined": True,
            }
        )
        spec["metadata"] = metadata
        validation = validate_evaluation_run_spec(spec)
        if validation.get("status") != "passed":
            raise CandidatePolicyError(
                "candidate_evaluation_run_invalid:"
                + ",".join(str(row) for row in validation.get("errors") or [])
            )
        compiled.append(spec)

    value: dict[str, Any] = {
        "schema_version": CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION,
        "frozen_scenario_manifest_digest": scenario_digest,
        "evaluator_digest": evaluator_digest,
        "success_predicate_digest": success_digest,
        "hidden_label_manifest_digest": _digest(
            frozen_scenario_manifest.get("hidden_label_manifest_digest"),
            field="hidden_label_manifest_digest",
        ),
        "evaluator_provider_id": evaluator_provider_id,
        "candidate_evaluation_run_specs": compiled,
        "candidate_count": len(compiled),
        "same_scenarios_for_every_candidate": True,
        "same_evaluator_for_every_candidate": True,
        "same_success_predicates_for_every_candidate": True,
        "same_observation_schema_for_every_candidate": True,
        "same_action_schema_for_every_candidate": True,
        "observation_schema_ref": next(iter(observation_schema_refs)),
        "action_schema_ref": next(iter(action_schema_refs)),
        "hidden_labels_sent_to_candidates": False,
        "candidate_agents_control_evaluator": False,
        "candidate_agents_grade_themselves": False,
        "development_repair_during_hidden_evaluation": False,
        "claim_ceiling": "simulation_only_unless_qualified_physical_evidence_is_joined",
        "provider_execution_started": False,
        "proof_effect": "none",
    }
    value["candidate_evaluation_suite_digest"] = canonical_digest(
        value,
        digest_field="candidate_evaluation_suite_digest",
    )
    return value


def execute_neutral_candidate_policy_suite(
    suite: Mapping[str, Any],
    *,
    candidate_runtimes: Sequence[CandidatePolicyRuntime],
    candidate_cost_authorities: Sequence[CandidateCostAuthority] = (),
    evaluator: IndependentCandidateEvaluator,
    hidden_evaluation_manifest: Mapping[str, Any],
    output_dir: str | Path,
    allow_execution: bool = False,
    execution_authorization: Mapping[str, Any] | None = None,
    executed_at: str | None = None,
) -> dict[str, Any]:
    """Execute frozen candidates and grade traces at an independent hidden boundary."""

    expected_suite_digest = canonical_digest(
        suite,
        digest_field="candidate_evaluation_suite_digest",
    )
    if suite.get("schema_version") != CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION:
        raise CandidatePolicyError("candidate_evaluation_suite_schema_invalid")
    if suite.get("candidate_evaluation_suite_digest") != expected_suite_digest:
        raise CandidatePolicyError("candidate_evaluation_suite_digest_mismatch")
    if (
        suite.get("hidden_labels_sent_to_candidates") is not False
        or suite.get("candidate_agents_control_evaluator") is not False
        or suite.get("candidate_agents_grade_themselves") is not False
        or suite.get("development_repair_during_hidden_evaluation") is not False
    ):
        raise CandidatePolicyError("candidate_evaluation_suite_boundary_invalid")
    hidden_digest = canonical_digest(hidden_evaluation_manifest)
    if hidden_digest != suite.get("hidden_label_manifest_digest"):
        raise CandidatePolicyError("hidden_evaluation_manifest_digest_mismatch")
    if evaluator.provider_id != suite.get("evaluator_provider_id"):
        raise CandidatePolicyError("independent_evaluator_provider_mismatch")
    if evaluator.evaluator_digest != suite.get("evaluator_digest"):
        raise CandidatePolicyError("independent_evaluator_digest_mismatch")

    specs = [
        dict(row)
        for row in suite.get("candidate_evaluation_run_specs") or []
        if isinstance(row, Mapping)
    ]
    expected_candidates = {
        str((row.get("policy_adapter") or {}).get("policy_id") or ""): {
            "manifest_digest": str(
                (row.get("metadata") or {}).get("candidate_policy_manifest_digest") or ""
            ),
            "runtime_configuration_digest": str(
                (row.get("policy_adapter") or {}).get("runtime_configuration_digest") or ""
            ),
        }
        for row in specs
    }
    runtimes = {runtime.candidate_id: runtime for runtime in candidate_runtimes}
    if (
        len(specs) != int(suite.get("candidate_count") or 0)
        or not expected_candidates
        or "" in expected_candidates
        or set(runtimes) != set(expected_candidates)
        or len(runtimes) != len(candidate_runtimes)
    ):
        raise CandidatePolicyError("candidate_runtime_set_mismatch")
    cost_authorities: dict[tuple[str, str], CandidateCostAuthority] = {}
    for authority in candidate_cost_authorities:
        try:
            key = (authority.provider_id, authority.paid_resource_class)
            authority_id = authority.authority_id
            authority_binding_digest = authority.cost_authority_binding_digest
        except AttributeError as exc:
            raise CandidatePolicyError("candidate_cost_authority_invalid") from exc
        if (
            not all(str(item).strip() for item in (*key, authority_id))
            or not _SHA256_DIGEST.fullmatch(str(authority_binding_digest or ""))
            or key in cost_authorities
        ):
            raise CandidatePolicyError("candidate_cost_authority_invalid")
        cost_authorities[key] = authority
    required_cost_authorities: set[tuple[str, str]] = set()
    for candidate_id, runtime in runtimes.items():
        try:
            manifest_digest = runtime.candidate_policy_manifest_digest
            runtime_configuration_digest = runtime.runtime_configuration_digest
            provider_id = runtime.provider_id
            provider_execution_planned = runtime.provider_execution_planned
            cost_accounting_authoritative = runtime.cost_accounting_authoritative
            cost_authority_binding_digest = runtime.cost_authority_binding_digest
            paid_resource_class = runtime.paid_resource_class
        except AttributeError as exc:
            raise CandidatePolicyError("candidate_runtime_execution_profile_invalid") from exc
        if manifest_digest != expected_candidates[candidate_id]["manifest_digest"]:
            raise CandidatePolicyError("candidate_runtime_manifest_digest_mismatch")
        if (
            runtime_configuration_digest
            != expected_candidates[candidate_id]["runtime_configuration_digest"]
        ):
            raise CandidatePolicyError("candidate_runtime_configuration_digest_mismatch")
        if (
            not provider_id.strip()
            or not isinstance(provider_execution_planned, bool)
            or not isinstance(cost_accounting_authoritative, bool)
            or (provider_execution_planned and not str(paid_resource_class or "").strip())
            or (
                provider_execution_planned
                and not _SHA256_DIGEST.fullmatch(str(cost_authority_binding_digest or ""))
            )
        ):
            raise CandidatePolicyError("candidate_runtime_execution_profile_invalid")
        if provider_execution_planned:
            if cost_accounting_authoritative:
                raise CandidatePolicyError(
                    "candidate_runtime_self_declared_cost_authority_forbidden"
                )
            required_cost_authorities.add((provider_id, str(paid_resource_class)))

    root = Path(output_dir).expanduser().resolve()
    if not allow_execution:
        root.mkdir(parents=True, exist_ok=True)
        value: dict[str, Any] = {
            "schema_version": CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION,
            "status": "prepared",
            "candidate_evaluation_suite_digest": expected_suite_digest,
            "execution_started": False,
            "candidate_results": [],
            "hidden_evaluation_manifest_digest": hidden_digest,
            "hidden_labels_sent_to_candidates": False,
            "candidate_agents_grade_themselves": False,
            "proof_effect": "none",
        }
        value["candidate_evaluation_execution_digest"] = canonical_digest(
            value,
            digest_field="candidate_evaluation_execution_digest",
        )
        write_json(root / "candidate_evaluation_execution.json", value)
        return value

    authorization = _validate_execution_authorization(
        execution_authorization,
        suite_digest=expected_suite_digest,
        hidden_digest=hidden_digest,
        specs=specs,
        runtimes=runtimes,
        executed_at=executed_at,
    )
    authorization_digest = str(authorization["authorization_receipt_digest"])
    authorized_cost = float(authorization.get("granted_max_cost_usd") or 0.0)
    reported_cost = 0.0
    paid_admission_validated: list[str] = []
    cost_authority_validated: list[str] = []
    for candidate_id, runtime in sorted(runtimes.items()):
        if not runtime.provider_execution_planned:
            continue
        try:
            require_paid_resource_admission_grant(
                runtime.paid_resource_admission_grant,
                resource_class=str(runtime.paid_resource_class),
            )
        except (AttributeError, PaidResourceAdmissionBlocked) as exc:
            raise CandidatePolicyError(
                "candidate_paid_resource_admission_missing_or_invalid"
            ) from exc
        paid_admission_validated.append(candidate_id)
    if set(cost_authorities) != required_cost_authorities:
        raise CandidatePolicyError("candidate_cost_authority_missing_or_unexpected")
    for candidate_id, runtime in runtimes.items():
        if not runtime.provider_execution_planned:
            continue
        authority = cost_authorities[(runtime.provider_id, str(runtime.paid_resource_class))]
        if runtime.cost_authority_binding_digest != authority.cost_authority_binding_digest:
            raise CandidatePolicyError("candidate_cost_authority_binding_mismatch")

    # No execution artifact or candidate directory exists until both the
    # operator receipt and every planned paid-resource grant validate.
    root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    cost_reconciliation_required_candidate_ids: list[str] = []
    allowed_runtime_keys = {
        "schema_version",
        "status",
        "trace_artifact_path",
        "trace_artifact_digest",
        "blockers",
        "cost_usd",
        "duration_seconds",
        "provider_execution_started",
        "attempt_count",
    }
    hidden_markers: set[str] = set()

    def collect_hidden_markers(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                collect_hidden_markers(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_hidden_markers(nested)
        elif isinstance(value, str) and len(value) >= 8:
            hidden_markers.add(value)

    collect_hidden_markers(hidden_evaluation_manifest)
    for spec in sorted(specs, key=lambda row: str(row.get("run_id") or "")):
        policy = dict(spec.get("policy_adapter") or {})
        candidate_id = str(policy.get("policy_id") or "")
        if policy.get("hidden_labels_included") is not False:
            raise CandidatePolicyError("candidate_spec_exposes_hidden_labels")
        candidate_root = (root / "candidates" / candidate_id).resolve()
        if root not in candidate_root.parents:
            raise CandidatePolicyError("candidate_output_path_escape")
        candidate_root.mkdir(parents=True, exist_ok=True)
        runtime = runtimes[candidate_id]
        cost_authority: CandidateCostAuthority | None = None
        cost_reservation: dict[str, Any] | None = None
        cost_dir = candidate_root / "cost_authority"
        if runtime.provider_execution_planned:
            if cost_dir.exists():
                raise CandidatePolicyError("candidate_cost_reconciliation_required")
            cost_authority = cost_authorities[
                (runtime.provider_id, str(runtime.paid_resource_class))
            ]
            try:
                reservation_value = cost_authority.reserve(
                    candidate_id=candidate_id,
                    candidate_evaluation_suite_digest=expected_suite_digest,
                    authorization_receipt_digest=authorization_digest,
                    max_cost_usd=float(policy.get("max_cost_usd") or 0.0),
                )
            except Exception as exc:  # noqa: BLE001 - typed metering refusal
                raise CandidatePolicyError("candidate_cost_reservation_failed") from exc
            cost_reservation = _validate_cost_reservation(
                reservation_value,
                authority=cost_authority,
                candidate_id=candidate_id,
                suite_digest=expected_suite_digest,
                authorization_digest=authorization_digest,
                max_cost_usd=float(policy.get("max_cost_usd") or 0.0),
            )
            cost_dir.mkdir(parents=True, exist_ok=False)
            write_json(cost_dir / "reservation.json", cost_reservation)
            cost_authority_validated.append(candidate_id)
        try:
            runtime_result = dict(
                runtime.execute(
                    evaluation_run_spec=copy.deepcopy(spec),
                    output_dir=candidate_root,
                )
            )
        except Exception as exc:  # noqa: BLE001 - typed failure, no raw message
            settlement: dict[str, Any] | None = None
            if cost_authority is not None and cost_reservation is not None:
                try:
                    settlement_value = cost_authority.settle(
                        reservation=copy.deepcopy(cost_reservation),
                        runtime_result=None,
                        runtime_exception_type=type(exc).__name__,
                    )
                except Exception as meter_exc:  # noqa: BLE001 - typed metering refusal
                    raise CandidatePolicyError("candidate_cost_settlement_failed") from meter_exc
                settlement = _validate_cost_settlement(
                    settlement_value,
                    authority=cost_authority,
                    reservation=cost_reservation,
                )
                write_json(cost_dir / "settlement.json", settlement)
                if settlement["cost_is_final"]:
                    reported_cost += float(settlement["actual_cost_usd"])
                    if reported_cost > authorized_cost:
                        raise CandidatePolicyError("candidate_execution_authorized_cost_exceeded")
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "failure_type": "candidate_runtime_exception",
                    "exception_type": type(exc).__name__,
                    "evaluated": False,
                    "cost_reconciliation_required": bool(
                        runtime.provider_execution_planned
                        and (settlement is None or settlement["cost_is_final"] is not True)
                    ),
                    "cost_reservation_digest": (
                        cost_reservation.get("cost_reservation_digest")
                        if cost_reservation is not None
                        else None
                    ),
                    "cost_settlement_digest": (
                        settlement.get("cost_settlement_digest") if settlement is not None else None
                    ),
                    "proof_effect": "none",
                }
            )
            if runtime.provider_execution_planned and (
                settlement is None or settlement["cost_is_final"] is not True
            ):
                cost_reconciliation_required_candidate_ids.append(candidate_id)
                break
            continue
        if set(runtime_result) - allowed_runtime_keys:
            raise CandidatePolicyError("candidate_runtime_result_contains_unregistered_fields")
        try:
            runtime_cost = float(runtime_result.get("cost_usd") or 0.0)
            runtime_duration = float(runtime_result.get("duration_seconds") or 0.0)
            attempt_count = int(runtime_result.get("attempt_count") or 1)
        except (TypeError, ValueError) as exc:
            raise CandidatePolicyError("candidate_runtime_accounting_invalid") from exc
        if (
            not math.isfinite(runtime_cost)
            or runtime_cost < 0
            or runtime_cost > float(policy.get("max_cost_usd") or 0.0)
            or not math.isfinite(runtime_duration)
            or runtime_duration < 0
            or attempt_count < 1
            or attempt_count > int(policy.get("retry_limit") or 0) + 1
            or not isinstance(runtime_result.get("provider_execution_started"), bool)
            or runtime_result.get("provider_execution_started")
            is not runtime.provider_execution_planned
        ):
            raise CandidatePolicyError("candidate_runtime_accounting_invalid")
        settlement = None
        authoritative_runtime_cost = runtime_cost
        if cost_authority is not None and cost_reservation is not None:
            try:
                settlement_value = cost_authority.settle(
                    reservation=copy.deepcopy(cost_reservation),
                    runtime_result=copy.deepcopy(runtime_result),
                    runtime_exception_type=None,
                )
            except Exception as exc:  # noqa: BLE001 - typed metering refusal
                raise CandidatePolicyError("candidate_cost_settlement_failed") from exc
            settlement = _validate_cost_settlement(
                settlement_value,
                authority=cost_authority,
                reservation=cost_reservation,
            )
            write_json(cost_dir / "settlement.json", settlement)
            if settlement["cost_is_final"]:
                authoritative_runtime_cost = float(settlement["actual_cost_usd"])
            else:
                authoritative_runtime_cost = 0.0
                cost_reconciliation_required_candidate_ids.append(candidate_id)
        reported_cost += authoritative_runtime_cost
        if reported_cost > authorized_cost:
            raise CandidatePolicyError("candidate_execution_authorized_cost_exceeded")
        if runtime_result.get("status") != "completed":
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": str(runtime_result.get("status") or "failed"),
                    "blockers": list(runtime_result.get("blockers") or []),
                    "evaluated": False,
                    "cost_reconciliation_required": bool(
                        settlement is not None and settlement["cost_is_final"] is not True
                    ),
                    "cost_reservation_digest": (
                        cost_reservation.get("cost_reservation_digest")
                        if cost_reservation is not None
                        else None
                    ),
                    "cost_settlement_digest": (
                        settlement.get("cost_settlement_digest") if settlement is not None else None
                    ),
                    "proof_effect": "none",
                }
            )
            if settlement is not None and settlement["cost_is_final"] is not True:
                break
            continue
        trace_path = (
            candidate_root / str(runtime_result.get("trace_artifact_path") or "")
        ).resolve()
        if candidate_root not in trace_path.parents or not trace_path.is_file():
            raise CandidatePolicyError("candidate_trace_path_invalid")
        trace = read_json(trace_path)
        trace_digest = canonical_digest(trace)
        if runtime_result.get("trace_artifact_digest") != trace_digest:
            raise CandidatePolicyError("candidate_trace_digest_mismatch")
        serialized_trace = json.dumps(trace, sort_keys=True)
        if any(marker in serialized_trace for marker in hidden_markers):
            raise CandidatePolicyError("candidate_trace_hidden_label_leakage")
        evaluation = dict(
            evaluator.evaluate(
                candidate_id=candidate_id,
                trace=trace,
                hidden_evaluation_manifest=dict(hidden_evaluation_manifest),
                success_predicate_digest=str(suite["success_predicate_digest"]),
            )
        )
        allowed_evaluation_keys = {
            "schema_version",
            "candidate_id",
            "status",
            "outcome",
            "metrics",
            "decisive_evidence",
            "uncertainty",
            "blockers",
            "evaluator_digest",
            "success_predicate_digest",
            "candidate_self_graded",
            "physical_validation_proven",
            "claim_ceiling",
        }
        if set(evaluation) - allowed_evaluation_keys:
            raise CandidatePolicyError("independent_candidate_evaluation_unregistered_fields")
        if (
            evaluation.get("schema_version") != "candidate_policy_independent_evaluation.v1"
            or evaluation.get("candidate_id") != candidate_id
            or evaluation.get("evaluator_digest") != suite.get("evaluator_digest")
            or evaluation.get("success_predicate_digest") != suite.get("success_predicate_digest")
            or evaluation.get("candidate_self_graded") is not False
            or evaluation.get("physical_validation_proven") is not False
            or evaluation.get("status") not in {"completed", "blocked"}
            or evaluation.get("outcome") not in {"passed", "failed", "inconclusive", "abstention"}
        ):
            raise CandidatePolicyError("independent_candidate_evaluation_invalid")
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, Mapping) or any(
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            for metric in metrics.values()
        ):
            raise CandidatePolicyError("independent_candidate_evaluation_metrics_invalid")
        serialized_evaluation = json.dumps(evaluation, sort_keys=True)
        if any(marker in serialized_evaluation for marker in hidden_markers):
            raise CandidatePolicyError("independent_candidate_evaluation_hidden_label_leakage")
        evaluation["independent_evaluation_digest"] = canonical_digest(
            evaluation,
            digest_field="independent_evaluation_digest",
        )
        evaluation_path = candidate_root / "independent_evaluation.json"
        write_json(evaluation_path, evaluation)
        results.append(
            {
                "candidate_id": candidate_id,
                "status": "evaluated",
                "trace_artifact_digest": trace_digest,
                "independent_evaluation_digest": evaluation["independent_evaluation_digest"],
                "outcome": evaluation.get("outcome"),
                "claim_ceiling": evaluation.get("claim_ceiling"),
                "candidate_self_graded": False,
                "hidden_labels_sent_to_candidate": False,
                "candidate_reported_cost_usd": runtime_cost,
                "candidate_reported_cost_accepted": cost_authority is None,
                "cost_reservation_digest": (
                    cost_reservation.get("cost_reservation_digest")
                    if cost_reservation is not None
                    else None
                ),
                "cost_settlement_digest": (
                    settlement.get("cost_settlement_digest") if settlement is not None else None
                ),
                "proof_effect": "none",
            }
        )

    value = {
        "schema_version": CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION,
        "status": (
            "completed"
            if results
            and all(row.get("status") == "evaluated" for row in results)
            and not cost_reconciliation_required_candidate_ids
            else "partial"
        ),
        "candidate_evaluation_suite_digest": expected_suite_digest,
        "execution_started": True,
        "candidate_results": results,
        "hidden_evaluation_manifest_digest": hidden_digest,
        "hidden_labels_sent_to_candidates": False,
        "candidate_agents_grade_themselves": False,
        "independent_evaluator_provider_id": evaluator.provider_id,
        "independent_evaluator_digest": evaluator.evaluator_digest,
        "authorization_receipt_digest": authorization_digest,
        "authorized_max_cost_usd": authorized_cost,
        "reported_cost_usd": round(reported_cost, 6),
        "reported_cost_is_final": not cost_reconciliation_required_candidate_ids,
        "cost_reconciliation_required_candidate_ids": sorted(
            cost_reconciliation_required_candidate_ids
        ),
        "paid_resource_admission_validated_candidate_ids": sorted(paid_admission_validated),
        "cost_authority_validated_candidate_ids": sorted(cost_authority_validated),
        "claim_ceiling": suite.get("claim_ceiling"),
        "physical_validation_proven": False,
        "deployment_approval_proven": False,
        "proof_effect": "none",
    }
    value["candidate_evaluation_execution_digest"] = canonical_digest(
        value,
        digest_field="candidate_evaluation_execution_digest",
    )
    write_json(root / "candidate_evaluation_execution.json", value)
    return value


def reconcile_neutral_candidate_policy_costs(
    execution_dir: str | Path,
    *,
    candidate_cost_authorities: Sequence[CandidateCostAuthority],
) -> dict[str, Any]:
    """Reconcile delayed provider costs without rerunning or regrading candidates."""

    root = Path(execution_dir).expanduser().resolve()
    execution_path = root / "candidate_evaluation_execution.json"
    if not execution_path.is_file():
        raise CandidatePolicyError("candidate_execution_artifact_missing")
    execution = read_json(execution_path)
    execution_digest = canonical_digest(
        execution,
        digest_field="candidate_evaluation_execution_digest",
    )
    if (
        execution.get("schema_version") != CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION
        or execution.get("candidate_evaluation_execution_digest") != execution_digest
        or execution.get("execution_started") is not True
    ):
        raise CandidatePolicyError("candidate_execution_artifact_invalid")
    pending_ids = sorted(
        str(item)
        for item in execution.get("cost_reconciliation_required_candidate_ids") or []
        if str(item).strip()
    )
    if not pending_ids or len(pending_ids) != len(set(pending_ids)):
        raise CandidatePolicyError("candidate_cost_reconciliation_not_required")
    authorities: dict[tuple[str, str], CandidateCostAuthority] = {}
    for authority in candidate_cost_authorities:
        try:
            key = (authority.provider_id, authority.paid_resource_class)
            binding_digest = authority.cost_authority_binding_digest
            authority_id = authority.authority_id
        except AttributeError as exc:
            raise CandidatePolicyError("candidate_cost_authority_invalid") from exc
        if (
            not all(str(item).strip() for item in (*key, authority_id))
            or not _SHA256_DIGEST.fullmatch(str(binding_digest or ""))
            or key in authorities
        ):
            raise CandidatePolicyError("candidate_cost_authority_invalid")
        authorities[key] = authority

    execution_results = {
        str(row.get("candidate_id") or ""): dict(row)
        for row in execution.get("candidate_results") or []
        if isinstance(row, Mapping)
    }
    if any(candidate_id not in execution_results for candidate_id in pending_ids):
        raise CandidatePolicyError("candidate_cost_reconciliation_result_missing")

    reconciliations: list[dict[str, Any]] = []
    total_cost = float(execution.get("reported_cost_usd") or 0.0)
    all_final = True
    for candidate_id in pending_ids:
        cost_dir = (root / "candidates" / candidate_id / "cost_authority").resolve()
        if root not in cost_dir.parents:
            raise CandidatePolicyError("candidate_cost_reconciliation_path_escape")
        reservation_path = cost_dir / "reservation.json"
        if not reservation_path.is_file():
            raise CandidatePolicyError("candidate_cost_reservation_missing")
        reservation = read_json(reservation_path)
        provider_id = str(reservation.get("provider_id") or "")
        resource_class = str(reservation.get("paid_resource_class") or "")
        matched_authority = authorities.get((provider_id, resource_class))
        if matched_authority is None:
            raise CandidatePolicyError("candidate_cost_authority_missing_or_unexpected")
        validated_reservation = _validate_cost_reservation(
            reservation,
            authority=matched_authority,
            candidate_id=candidate_id,
            suite_digest=str(execution.get("candidate_evaluation_suite_digest") or ""),
            authorization_digest=str(execution.get("authorization_receipt_digest") or ""),
            max_cost_usd=float(reservation.get("reserved_max_cost_usd") or 0.0),
        )
        if (
            validated_reservation.get("cost_authority_binding_digest")
            != matched_authority.cost_authority_binding_digest
        ):
            raise CandidatePolicyError("candidate_cost_authority_binding_mismatch")
        reconciliation_dir = cost_dir / "reconciliations"
        existing_final: dict[str, Any] | None = None
        if reconciliation_dir.is_dir():
            for existing_path in sorted(reconciliation_dir.glob("*.json")):
                existing = read_json(existing_path)
                try:
                    checked = _validate_cost_settlement(
                        existing,
                        authority=matched_authority,
                        reservation=validated_reservation,
                    )
                except CandidatePolicyError:
                    continue
                if checked["cost_is_final"] is True:
                    existing_final = checked
                    break
        if existing_final is not None:
            settlement = existing_final
        else:
            prior_result = execution_results[candidate_id]
            try:
                settlement_value = matched_authority.settle(
                    reservation=copy.deepcopy(validated_reservation),
                    runtime_result=None,
                    runtime_exception_type=(str(prior_result.get("exception_type") or "") or None),
                )
            except Exception as exc:  # noqa: BLE001 - typed metering refusal
                raise CandidatePolicyError("candidate_cost_settlement_failed") from exc
            settlement = _validate_cost_settlement(
                settlement_value,
                authority=matched_authority,
                reservation=validated_reservation,
            )
            reconciliation_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                reconciliation_dir
                / f"{str(settlement['cost_settlement_digest']).removeprefix('sha256:')}.json",
                settlement,
            )
        final = settlement["cost_is_final"] is True
        if final:
            total_cost += float(settlement["actual_cost_usd"])
        else:
            all_final = False
        reconciliations.append(
            {
                "candidate_id": candidate_id,
                "status": settlement["status"],
                "cost_is_final": final,
                "actual_cost_usd": settlement.get("actual_cost_usd"),
                "cost_reservation_digest": settlement["cost_reservation_digest"],
                "cost_settlement_digest": settlement["cost_settlement_digest"],
                "candidate_reported_cost_accepted": False,
                "proof_effect": "none",
            }
        )

    authorized_cost = float(execution.get("authorized_max_cost_usd") or 0.0)
    if total_cost > authorized_cost:
        raise CandidatePolicyError("candidate_execution_authorized_cost_exceeded")
    report: dict[str, Any] = {
        "schema_version": CANDIDATE_COST_RECONCILIATION_SCHEMA_VERSION,
        "status": "reconciled" if all_final else "reconciliation_required",
        "candidate_evaluation_execution_digest": execution_digest,
        "candidate_evaluation_suite_digest": execution.get("candidate_evaluation_suite_digest"),
        "authorization_receipt_digest": execution.get("authorization_receipt_digest"),
        "candidate_reconciliations": reconciliations,
        "authorized_max_cost_usd": authorized_cost,
        "reported_cost_usd": round(total_cost, 8) if all_final else None,
        "reported_cost_is_final": all_final,
        "candidate_reported_cost_accepted": False,
        "candidate_execution_repeated": False,
        "candidate_evaluation_repeated": False,
        "proof_effect": "none",
    }
    report["candidate_cost_reconciliation_digest"] = canonical_digest(
        report,
        digest_field="candidate_cost_reconciliation_digest",
    )
    report_dir = root / "cost_reconciliations"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        str(report["candidate_cost_reconciliation_digest"]).removeprefix("sha256:") + ".json"
    )
    if report_path.exists() and read_json(report_path) != report:
        raise CandidatePolicyError("candidate_cost_reconciliation_append_conflict")
    write_json(report_path, report)
    return report


__all__ = [
    "CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION",
    "CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION",
    "CANDIDATE_COST_RESERVATION_SCHEMA_VERSION",
    "CANDIDATE_COST_SETTLEMENT_SCHEMA_VERSION",
    "CANDIDATE_COST_RECONCILIATION_SCHEMA_VERSION",
    "CANDIDATE_POLICY_MANIFEST_SCHEMA_VERSION",
    "CandidateCostAuthority",
    "CandidatePolicyError",
    "FrozenAgenticPolicyAdapter",
    "IndependentCandidateEvaluator",
    "CandidatePolicyRuntime",
    "compile_neutral_candidate_policy_suite",
    "execute_neutral_candidate_policy_suite",
    "reconcile_neutral_candidate_policy_costs",
    "freeze_candidate_policy_manifest",
]
