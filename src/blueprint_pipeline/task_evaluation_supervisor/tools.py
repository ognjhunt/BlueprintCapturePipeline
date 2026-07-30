"""Capability-gated tool registry for the Task Evaluation Supervisor.

Phase 0/1 descriptors expose proposal and inspection surfaces only.  No tool
in this registry owns proof transitions, paid allocation, or physical actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from ..common import write_json
from ..decision_evidence_contracts import canonical_digest
from ..decision_evidence_router import route_decision_evidence
from ..evaluation_run_contract import validate_evaluation_run_spec
from .contracts import (
    TOOL_OBSERVATION_SCHEMA_VERSION,
    ActionProposal,
    AuthorityEnvelope,
    AutonomyMode,
    ToolDescriptor,
    ToolObservation,
)
from .phase2_artifacts import (
    authorization_request,
    clarification_request,
    scenario_proposal_set,
    write_phase2_artifact,
)


TOOL_REGISTRY_SCHEMA_VERSION = "task_evaluation_supervisor_tool_registry.v1"


def _descriptor(
    tool_id: str,
    category: str,
    *,
    expected_artifacts: Sequence[str],
    input_properties: Mapping[str, Mapping[str, Any]],
    required_inputs: Sequence[str],
    mutability: str = "read_only",
    allowed_modes: Sequence[str] = (
        "shadow",
        "advise",
        "execute_non_spend",
        "execute_preauthorized",
    ),
    minimum_mode: str = "shadow",
    max_cost_usd: float = 0.0,
    max_retries: int = 0,
    timeout_seconds: float = 30.0,
    idempotency: str = "deterministic_for_bound_inputs",
) -> ToolDescriptor:
    safety_level = {
        "read_only": "proof_safe_read_only",
        "reversible_mutation": "proof_safe_reversible_non_spend",
        "external_side_effect": "preauthorized_external_side_effect",
    }[mutability]
    rollback_reason = {
        "read_only": "read_only",
        "reversible_mutation": "delete_supervisor_scoped_generated_artifacts",
        "external_side_effect": "mandatory_provider_teardown_and_provider_zero_proof",
    }[mutability]
    return ToolDescriptor.from_mapping(
        {
            "schema_version": "task_evaluation_supervisor_tool.v1",
            "tool_id": tool_id,
            "version": "1",
            "category": category,
            "mutability": mutability,
            "idempotency": idempotency,
            "input_schema": {
                "type": "object",
                "required": list(required_inputs),
                "properties": {key: dict(schema) for key, schema in input_properties.items()},
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "required": ["schema_version", "status"],
                "properties": {
                    "schema_version": {"type": "string"},
                    "status": {"type": "string"},
                    "artifact_references": {"type": "array"},
                    "proof_effect": {"const": "none"},
                },
                "additionalProperties": True,
            },
            "expected_artifacts": list(expected_artifacts),
            "max_cost_usd": max_cost_usd,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "safety_level": safety_level,
            "required_authority": {"minimum_mode": minimum_mode},
            "allowed_modes": list(allowed_modes),
            "proof_effect": "none",
            "evidence_requirements": [],
            "rollback": {
                "required": mutability != "read_only",
                "reason": rollback_reason,
            },
        }
    )


def default_tool_descriptors() -> tuple[ToolDescriptor, ...]:
    """Return the stable Phase 0/1 tool surface.

    These are descriptors, not generic execution handles.  Later phases may
    bind implementations only after their mutation and authority contracts are
    independently tested.
    """

    return (
        _descriptor(
            "inspect_site_task_testbed",
            "capture_testbed_inspection",
            expected_artifacts=["capture_testbed_inspection.v1"],
            input_properties={"testbed_digest": {"type": "string"}},
            required_inputs=["testbed_digest"],
        ),
        _descriptor(
            "validate_proposed_claim_graph",
            "claim_contract_validation",
            expected_artifacts=["proposed_claim_graph.v1"],
            input_properties={"request_digest": {"type": "string"}},
            required_inputs=["request_digest"],
        ),
        _descriptor(
            "materialize_clarification_request",
            "claim_contract_validation",
            expected_artifacts=["task_evaluation_clarification_request.v1"],
            input_properties={
                "source_digest": {"type": "string"},
                "questions": {"type": "array"},
                "blocking_fields": {"type": "array"},
            },
            required_inputs=["source_digest", "questions", "blocking_fields"],
            mutability="reversible_mutation",
        ),
        _descriptor(
            "compile_deterministic_evidence_plan",
            "evidence_method_routing",
            expected_artifacts=["evidence_plan.v1"],
            input_properties={"plan_digest": {"type": "string"}},
            required_inputs=["plan_digest"],
        ),
        _descriptor(
            "materialize_compiled_leaf_runs",
            "local_leaf_run_compilation",
            expected_artifacts=["evidence_plan.v1", "evaluation_run_spec.v1"],
            input_properties={
                "request_digest": {"type": "string"},
                "testbed_digest": {"type": "string"},
            },
            required_inputs=["request_digest", "testbed_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
        ),
        _descriptor(
            "inspect_normalized_evidence_results",
            "runtime_failure_diagnosis",
            expected_artifacts=["typed_failure_diagnosis.v1"],
            input_properties={
                "result_digest": {"type": "string"},
                "execution_requested": {"type": "boolean"},
            },
            required_inputs=["result_digest", "execution_requested"],
        ),
        _descriptor(
            "propose_targeted_recapture",
            "capture_testbed_inspection",
            expected_artifacts=["targeted_recapture_request.v1"],
            input_properties={
                "source_digest": {"type": "string"},
                "missing_evidence": {"type": "array"},
                "full_site_recapture_requested": {"type": "boolean"},
            },
            required_inputs=[
                "source_digest",
                "missing_evidence",
                "full_site_recapture_requested",
            ],
            mutability="reversible_mutation",
        ),
        _descriptor(
            "propose_adversarial_scenarios",
            "scenario_generation",
            expected_artifacts=["scenario_proposal_set.v1"],
            input_properties={
                "request_digest": {"type": "string"},
                "scenarios": {"type": "array"},
                "candidate_results_observed": {"type": "boolean"},
            },
            required_inputs=["request_digest", "scenarios", "candidate_results_observed"],
            mutability="reversible_mutation",
        ),
        _descriptor(
            "materialize_authorization_request",
            "runtime_failure_diagnosis",
            expected_artifacts=["task_evaluation_authorization_request.v1"],
            input_properties={
                "tool_id": {"type": "string"},
                "reason": {"type": "string"},
                "requested_max_cost_usd": {"type": "number"},
                "requested_ttl_seconds": {"type": "integer"},
                "requested_retry_count": {"type": "integer"},
                "requested_provider_ids": {"type": "array"},
                "requested_action_ids": {"type": "array"},
            },
            required_inputs=[
                "tool_id",
                "reason",
                "requested_max_cost_usd",
                "requested_ttl_seconds",
                "requested_retry_count",
                "requested_provider_ids",
                "requested_action_ids",
            ],
            mutability="reversible_mutation",
        ),
        _descriptor(
            "execute_preauthorized_recovery",
            "runtime_failure_recovery",
            expected_artifacts=["task_evaluation_recovery_result.v1"],
            input_properties={
                "action_id": {"type": "string"},
                "provider_id": {"type": "string"},
                "immutable_commit_sha": {"type": "string"},
                "input_digests": {"type": "array"},
                "projected_cost_usd": {"type": "number"},
                "failure_type": {"type": "string"},
            },
            required_inputs=[
                "action_id",
                "provider_id",
                "immutable_commit_sha",
                "input_digests",
                "projected_cost_usd",
                "failure_type",
            ],
            mutability="external_side_effect",
            allowed_modes=["execute_preauthorized"],
            minimum_mode="execute_preauthorized",
            max_cost_usd=100.0,
            max_retries=3,
            timeout_seconds=3_600.0,
            idempotency="receipt_bound_attempt;provider_action_not_assumed_idempotent",
        ),
        _descriptor(
            "explain_deterministic_decision",
            "post_run_diagnosis",
            expected_artifacts=["post_run_diagnosis.v1"],
            input_properties={"decision_envelope_digest": {"type": "string"}},
            required_inputs=["decision_envelope_digest"],
        ),
    )


@dataclass(frozen=True)
class ToolRegistry:
    _tools: Mapping[str, ToolDescriptor]

    @classmethod
    def from_descriptors(cls, values: Sequence[ToolDescriptor]) -> "ToolRegistry":
        tools: dict[str, ToolDescriptor] = {}
        for descriptor in values:
            mapping = descriptor.to_mapping()
            tool_id = str(mapping["tool_id"])
            if tool_id in tools:
                raise ValueError(f"duplicate_supervisor_tool:{tool_id}")
            tools[tool_id] = descriptor
        return cls(tools)

    @classmethod
    def default(cls) -> "ToolRegistry":
        return cls.from_descriptors(default_tool_descriptors())

    def resolve(self, tool_id: str) -> ToolDescriptor | None:
        return self._tools.get(str(tool_id or "").strip())

    def allowed_tool_ids_for_capability(self, capability: str) -> tuple[str, ...]:
        return tuple(
            tool_id
            for tool_id in _CAPABILITY_TOOL_IDS.get(str(capability or ""), ())
            if tool_id in self._tools
        )

    def manifest(self) -> dict[str, Any]:
        descriptors = [self._tools[tool_id].to_mapping() for tool_id in sorted(self._tools)]
        value = {
            "schema_version": TOOL_REGISTRY_SCHEMA_VERSION,
            "tools": descriptors,
            "unrestricted_shell_available": False,
            "unrestricted_filesystem_available": False,
            "unrestricted_network_available": False,
            "unrestricted_provider_access_available": False,
            "proof_mutation_tools_registered": False,
            "paid_tools_registered": any(
                float(row.get("max_cost_usd") or 0) > 0 for row in descriptors
            ),
            "physical_action_tools_registered": False,
        }
        value["tool_registry_digest"] = canonical_digest(value, digest_field="tool_registry_digest")
        return value

    @property
    def digest(self) -> str:
        return str(self.manifest()["tool_registry_digest"])

    def disposition(
        self,
        proposal_value: Mapping[str, Any],
        authority_value: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...]]:
        """Deterministically classify a proposal without executing it."""

        proposal = ActionProposal.from_mapping(proposal_value).to_mapping()
        authority = AuthorityEnvelope.from_mapping(authority_value).to_mapping()
        mode = AutonomyMode(str(authority["mode"]))
        blockers: list[str] = []
        tool_id = str(proposal.get("tool_id") or "")
        tool = self.resolve(tool_id) if tool_id else None
        if tool_id and tool is None:
            blockers.append("unregistered_tool")
        tool_value = tool.to_mapping() if tool is not None else {}
        if (
            tool is not None
            and mode is not AutonomyMode.ADVISE
            and mode.value not in set(tool_value.get("allowed_modes") or [])
        ):
            blockers.append("tool_not_allowed_in_mode")
        if tool_id and tool_id not in set(authority.get("allowed_tool_ids") or []):
            blockers.append("tool_not_in_authority_envelope")
        proposal_cost = float(proposal.get("estimated_cost_usd") or 0)
        if tool is not None and proposal_cost > float(tool_value.get("max_cost_usd") or 0):
            blockers.append("proposal_exceeds_tool_cost_limit")
        if mode is not AutonomyMode.ADVISE and proposal_cost > float(
            authority.get("max_cost_usd") or 0
        ):
            blockers.append("proposal_exceeds_cost_authority")
        if str(proposal.get("requested_proof_effect") or "") != "none":
            blockers.append("proof_mutation_requested")
        if tool is not None:
            blockers.extend(
                self._input_schema_errors(
                    proposal.get("parameters"), tool_value.get("input_schema")
                )
            )
        if blockers:
            return "refused", tuple(sorted(set(blockers)))
        if mode is AutonomyMode.DISABLED:
            return "refused", ("supervisor_disabled",)
        if mode is AutonomyMode.SHADOW:
            return "shadow_only", ()
        if mode is AutonomyMode.ADVISE:
            return "requires_operator_approval", ()
        return "eligible", ()

    @staticmethod
    def _input_schema_errors(value: Any, schema_value: Any) -> list[str]:
        if not isinstance(value, Mapping):
            return ["tool_input_not_mapping"]
        schema = dict(schema_value) if isinstance(schema_value, Mapping) else {}
        properties = dict(schema.get("properties") or {})
        required = {str(item) for item in schema.get("required") or []}
        errors = [f"tool_input_missing:{key}" for key in sorted(required - set(value))]
        if schema.get("additionalProperties") is False:
            errors.extend(
                f"tool_input_unknown:{key}" for key in sorted(set(value) - set(properties))
            )
        type_checks = {
            "string": lambda item: isinstance(item, str),
            "boolean": lambda item: isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "array": lambda item: isinstance(item, list),
            "object": lambda item: isinstance(item, Mapping),
        }
        for key in sorted(set(value) & set(properties)):
            expected = str(dict(properties[key]).get("type") or "")
            check = type_checks.get(expected)
            if check is not None and not check(value[key]):
                errors.append(f"tool_input_type:{key}:{expected}")
        return errors


def validate_tool_observation_binding(
    observation_value: Mapping[str, Any],
    *,
    run_id: str,
    capability: str,
    registry: ToolRegistry,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a tool result against the exact registered execution scope."""

    observation = ToolObservation.from_mapping(observation_value).to_mapping()
    validated_authority = AuthorityEnvelope.from_mapping(authority).to_mapping()
    if observation["run_id"] != run_id:
        raise ValueError("tool_observation_run_mismatch")
    if observation["capability"] != capability:
        raise ValueError("tool_observation_capability_mismatch")
    if observation["authority_digest"] != validated_authority["authority_digest"]:
        raise ValueError("tool_observation_authority_mismatch")
    tool_id = str(observation["tool_id"])
    descriptor = registry.resolve(tool_id)
    if descriptor is None:
        raise ValueError("tool_observation_unregistered_tool")
    descriptor_value = descriptor.to_mapping()
    if tool_id not in registry.allowed_tool_ids_for_capability(capability):
        raise ValueError("tool_observation_capability_tool_mismatch")
    if tool_id not in set(validated_authority.get("allowed_tool_ids") or []):
        raise ValueError("tool_observation_tool_not_authorized")
    if observation["tool_version"] != descriptor_value["version"]:
        raise ValueError("tool_observation_version_mismatch")
    if observation["mutability"] != descriptor_value["mutability"]:
        raise ValueError("tool_observation_mutability_mismatch")
    expected_runtime = (
        "blueprint_preauthorized_recovery_controller"
        if observation["mutability"] == "external_side_effect"
        else "blueprint_local_deterministic_non_spend"
    )
    if observation["runtime_identity"] != expected_runtime:
        raise ValueError("tool_observation_runtime_identity_mismatch")
    if observation["output_digest"] != canonical_digest(observation["typed_result"]):
        raise ValueError("tool_observation_output_digest_mismatch")
    cost = float(observation["cost_usd"])
    duration = float(observation["duration_seconds"])
    retries = int(observation["retries"])
    if cost > float(descriptor_value["max_cost_usd"]):
        raise ValueError("tool_observation_tool_cost_exceeded")
    if cost > float(validated_authority["max_cost_usd"]):
        raise ValueError("tool_observation_authority_cost_exceeded")
    if duration > float(descriptor_value["timeout_seconds"]):
        raise ValueError("tool_observation_tool_duration_exceeded")
    if duration > float(validated_authority["max_duration_seconds"]):
        raise ValueError("tool_observation_authority_duration_exceeded")
    if retries > int(descriptor_value["max_retries"]):
        raise ValueError("tool_observation_tool_retries_exceeded")
    if retries > int(validated_authority["max_retries"]):
        raise ValueError("tool_observation_authority_retries_exceeded")
    if observation["mutability"] != "external_side_effect" and cost != 0:
        raise ValueError("tool_observation_non_spend_cost_nonzero")
    if observation["mutability"] == "external_side_effect" and (
        validated_authority["mode"] != AutonomyMode.EXECUTE_PREAUTHORIZED.value
    ):
        raise ValueError("tool_observation_external_side_effect_wrong_mode")
    return observation


@dataclass(frozen=True)
class RegisteredToolBinding:
    """One SDK-callable binding to a deterministic, read-only Blueprint tool."""

    tool_id: str
    description: str
    input_schema: Mapping[str, Any]
    timeout_seconds: float
    invoke: Callable[[Mapping[str, Any]], Mapping[str, Any]]


_CAPABILITY_TOOL_IDS: dict[str, tuple[str, ...]] = {
    "claim_task_interpreter": (
        "validate_proposed_claim_graph",
        "materialize_clarification_request",
    ),
    "capture_testbed_supervisor": (
        "inspect_site_task_testbed",
        "propose_targeted_recapture",
    ),
    "evaluation_method_router": (
        "compile_deterministic_evidence_plan",
        "materialize_compiled_leaf_runs",
    ),
    "runtime_failure_recovery": (
        "inspect_normalized_evidence_results",
        "materialize_authorization_request",
        "execute_preauthorized_recovery",
    ),
    "scenario_adversarial_proposer": ("propose_adversarial_scenarios",),
    "post_run_diagnostician": ("explain_deterministic_decision",),
}


def _safe_artifact_name(value: Any) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-.")
    return rendered[:192] or "leaf-run"


def _materialize_leaf_runs(
    *,
    context: Any,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request = context.decision_request
    testbed = context.testbed
    request_digest = request.get("request_digest") if isinstance(request, Mapping) else None
    testbed_digest = testbed.get("testbed_digest") if isinstance(testbed, Mapping) else None
    if (
        not isinstance(request, Mapping)
        or not isinstance(testbed, Mapping)
        or not request_digest
        or not testbed_digest
        or arguments.get("request_digest") != request_digest
        or arguments.get("testbed_digest") != testbed_digest
    ):
        raise ValueError("registered_tool_bound_artifact_mismatch:materialize_compiled_leaf_runs")
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:materialize_compiled_leaf_runs")
    generated_root = Path(root_value) / "generated"
    output_root = generated_root / "compiled_leaf_runs"
    plan = route_decision_evidence(
        request,
        testbed,
        context.method_profiles,
        context.qualifications,
    ).to_mapping()
    if isinstance(context.evidence_plan, Mapping):
        supplied_digest = context.evidence_plan.get("plan_digest")
        if supplied_digest != plan.get("plan_digest"):
            raise ValueError("deterministic_evidence_plan_drift")
    plan_path = generated_root / "evidence_plan.json"
    write_json(plan_path, plan)
    plan_reference = {
        "artifact_path": str(plan_path.relative_to(Path(root_value))),
        "artifact_digest": plan["plan_digest"],
        "artifact_type": "evidence_plan.v1",
        "plan_id": plan["plan_id"],
    }
    rows = plan.get("compiled_evaluation_run_specs")
    if not isinstance(rows, list):
        raise ValueError("compiled_leaf_run_specs_not_list")
    references: list[dict[str, Any]] = [plan_reference]
    seen_run_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("compiled_leaf_run_spec_not_mapping")
        spec = dict(row)
        validation = validate_evaluation_run_spec(spec)
        if validation.get("status") != "passed":
            raise ValueError("compiled_leaf_run_spec_invalid")
        run_id = str(spec.get("run_id") or "")
        if not run_id or run_id in seen_run_ids:
            raise ValueError("compiled_leaf_run_id_missing_or_duplicate")
        seen_run_ids.add(run_id)
        artifact_path = output_root / f"{_safe_artifact_name(run_id)}.json"
        write_json(artifact_path, spec)
        references.append(
            {
                "artifact_path": str(artifact_path.relative_to(Path(root_value))),
                "artifact_digest": canonical_digest(spec),
                "artifact_type": "evaluation_run_spec.v1",
                "run_id": run_id,
            }
        )
    return (
        {
            "contract_present": True,
            "digest_matches": True,
            "plan_digest": plan["plan_digest"],
            "compiled_leaf_run_count": len(references) - 1,
            "compiled_leaf_run_references": references[1:],
            "provider_execution_started": False,
            "proof_state_changed": False,
        },
        references,
    )


def _materialize_targeted_recapture_request(
    *,
    context: Any,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:propose_targeted_recapture")
    testbed_digest = (
        context.testbed.get("testbed_digest") if isinstance(context.testbed, Mapping) else None
    )
    capture_digest = (
        context.capture_build.get("capture_build_digest")
        if isinstance(context.capture_build, Mapping)
        else None
    )
    source_digest = arguments.get("source_digest")
    if source_digest not in {testbed_digest, capture_digest}:
        raise ValueError("registered_tool_bound_artifact_mismatch:propose_targeted_recapture")
    missing = arguments.get("missing_evidence")
    if not isinstance(missing, list) or not missing:
        raise ValueError("targeted_recapture_missing_evidence_required")
    normalized_missing = sorted(
        {str(item).strip() for item in missing if isinstance(item, str) and str(item).strip()}
    )
    if not normalized_missing:
        raise ValueError("targeted_recapture_missing_evidence_required")
    if len(normalized_missing) > 50 or any(len(item) > 500 for item in normalized_missing):
        raise ValueError("targeted_recapture_scope_out_of_range")
    if arguments.get("full_site_recapture_requested") is True:
        raise ValueError("full_site_recapture_requires_separate_operator_authorization")
    request: dict[str, Any] = {
        "schema_version": "targeted_recapture_request.v1",
        "request_id": f"{context.run_id}-targeted-recapture",
        "run_id": context.run_id,
        "source_digest": source_digest,
        "source_type": "site_task_testbed" if source_digest == testbed_digest else "capture_build",
        "missing_evidence": normalized_missing,
        "requested_scope": "targeted_only",
        "full_site_recapture_requested": False,
        "status": "proposed_for_review",
        "capture_started": False,
        "rights_clearance_inferred": False,
        "raw_capture_mutated": False,
        "authoritative": False,
        "proof_effect": "none",
    }
    request["targeted_recapture_request_digest"] = canonical_digest(
        request,
        digest_field="targeted_recapture_request_digest",
    )
    artifact_path = (
        Path(root_value)
        / "generated"
        / "targeted_recapture_requests"
        / f"{_safe_artifact_name(context.run_id)}.json"
    )
    write_json(artifact_path, request)
    reference = {
        "artifact_path": str(artifact_path.relative_to(Path(root_value))),
        "artifact_digest": request["targeted_recapture_request_digest"],
        "artifact_type": "targeted_recapture_request.v1",
        "request_id": request["request_id"],
    }
    return (
        {
            "contract_present": True,
            "digest_matches": True,
            "request_id": request["request_id"],
            "targeted_recapture_request_digest": request["targeted_recapture_request_digest"],
            "capture_started": False,
            "proof_state_changed": False,
        },
        [reference],
    )


def _source_digest(context: Any, value: Any) -> str:
    candidates = {
        (context.capture_build or {}).get("capture_build_digest"),
        (context.decision_request or {}).get("request_digest"),
        (context.testbed or {}).get("testbed_digest"),
    }
    if value not in candidates:
        raise ValueError("registered_tool_source_digest_mismatch")
    return str(value)


def _materialize_clarification_request(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:materialize_clarification_request"
        )
    artifact = clarification_request(
        run_id=context.run_id,
        source_digest=_source_digest(context, arguments.get("source_digest")),
        questions=arguments.get("questions") or [],
        blocking_fields=arguments.get("blocking_fields") or [],
    )
    path = write_phase2_artifact(
        root_value,
        "generated/clarification_requests/request.json",
        artifact,
    )
    reference = {
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": artifact["clarification_request_digest"],
        "artifact_type": "task_evaluation_clarification_request.v1",
        "request_id": artifact["request_id"],
    }
    return {
        "contract_present": True,
        "digest_matches": True,
        "request_id": artifact["request_id"],
        "awaiting_customer_response": True,
        "proof_state_changed": False,
    }, [reference]


def _materialize_authorization_request(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:materialize_authorization_request"
        )
    authority = context.authority_envelope or {}
    artifact = authorization_request(
        run_id=context.run_id,
        tool_id=str(arguments.get("tool_id") or ""),
        reason=str(arguments.get("reason") or ""),
        requested_max_cost_usd=float(arguments.get("requested_max_cost_usd") or 0.0),
        requested_ttl_seconds=int(arguments.get("requested_ttl_seconds") or 0),
        immutable_input_digests=authority.get("immutable_input_digests") or [],
        requested_retry_count=int(arguments.get("requested_retry_count") or 0),
        requested_provider_ids=arguments.get("requested_provider_ids") or [],
        requested_action_ids=arguments.get("requested_action_ids") or [],
    )
    path = write_phase2_artifact(
        root_value,
        "generated/authorization_requests/request.json",
        artifact,
    )
    reference = {
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": artifact["authorization_request_digest"],
        "artifact_type": "task_evaluation_authorization_request.v1",
        "request_id": artifact["request_id"],
    }
    return {
        "contract_present": True,
        "digest_matches": True,
        "request_id": artifact["request_id"],
        "authorization_granted": False,
        "proof_state_changed": False,
    }, [reference]


def _materialize_scenario_proposal_set(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:propose_adversarial_scenarios")
    request_digest = _source_digest(context, arguments.get("request_digest"))
    scenarios = arguments.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("scenario_proposals_must_be_list")
    artifact = scenario_proposal_set(
        run_id=context.run_id,
        request_digest=request_digest,
        scenarios=[row for row in scenarios if isinstance(row, Mapping)],
        candidate_results_observed=arguments.get("candidate_results_observed") is True,
    )
    path = write_phase2_artifact(
        root_value,
        "generated/scenario_proposals/proposal_set.json",
        artifact,
    )
    reference = {
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": artifact["scenario_proposal_set_digest"],
        "artifact_type": "task_evaluation_scenario_proposal_set.v1",
        "proposal_set_id": artifact["proposal_set_id"],
    }
    return {
        "contract_present": True,
        "digest_matches": True,
        "scenario_count": len(artifact["scenarios"]),
        "frozen": False,
        "hidden_labels_included": False,
        "proof_state_changed": False,
    }, [reference]


def _execute_preauthorized_recovery(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    controller = getattr(context, "recovery_controller", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:execute_preauthorized_recovery")
    if controller is None:
        raise ValueError("preauthorized_recovery_controller_missing")
    result = controller.execute(arguments)
    path = write_phase2_artifact(
        root_value,
        (f"generated/recovery_attempts/{_safe_artifact_name(result['attempt_id'])}.json"),
        result,
    )
    reference = {
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": result["recovery_result_digest"],
        "artifact_type": "task_evaluation_recovery_result.v1",
        "attempt_id": result["attempt_id"],
    }
    return result, [reference]


def _bound_artifact(
    context: Any,
    *,
    tool_id: str,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    produced_artifact_references: list[dict[str, Any]] = []
    if tool_id == "validate_proposed_claim_graph":
        value = context.decision_request
        digest_key = "request_digest"
        expected = arguments.get(digest_key)
        actual = value.get(digest_key) if isinstance(value, Mapping) else None
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "claim_ids": [
                str(row.get("claim_id"))
                for row in (value or {}).get("claims", [])
                if isinstance(row, Mapping) and row.get("claim_id")
            ]
            if isinstance(value, Mapping)
            else [],
        }
    elif tool_id == "materialize_clarification_request":
        return _materialize_clarification_request(context=context, arguments=arguments)
    elif tool_id == "inspect_site_task_testbed":
        value = context.testbed
        expected = arguments.get("testbed_digest")
        actual = value.get("testbed_digest") if isinstance(value, Mapping) else None
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "evidence_inventory_count": len((value or {}).get("evidence_inventory", []))
            if isinstance(value, Mapping)
            else 0,
            "governance": dict((value or {}).get("governance") or {})
            if isinstance(value, Mapping)
            else {},
        }
    elif tool_id == "compile_deterministic_evidence_plan":
        value = context.evidence_plan
        expected = arguments.get("plan_digest")
        actual = value.get("plan_digest") if isinstance(value, Mapping) else None
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "step_count": len((value or {}).get("steps", [])) if isinstance(value, Mapping) else 0,
            "compiled_by_agent": False,
        }
    elif tool_id == "materialize_compiled_leaf_runs":
        return _materialize_leaf_runs(context=context, arguments=arguments)
    elif tool_id == "propose_targeted_recapture":
        return _materialize_targeted_recapture_request(
            context=context,
            arguments=arguments,
        )
    elif tool_id == "materialize_authorization_request":
        return _materialize_authorization_request(context=context, arguments=arguments)
    elif tool_id == "execute_preauthorized_recovery":
        return _execute_preauthorized_recovery(context=context, arguments=arguments)
    elif tool_id == "propose_adversarial_scenarios":
        return _materialize_scenario_proposal_set(context=context, arguments=arguments)
    elif tool_id == "inspect_normalized_evidence_results":
        expected = arguments.get("result_digest")
        selected = next(
            (
                row
                for row in context.evidence_results
                if isinstance(row, Mapping) and row.get("result_digest") == expected
            ),
            None,
        )
        typed_result = {
            "contract_present": selected is not None,
            "digest_matches": selected is not None,
            "status": selected.get("status") if selected is not None else None,
            "failure_type": selected.get("failure_type") if selected is not None else None,
            "execution_requested": arguments.get("execution_requested") is True,
        }
    elif tool_id == "explain_deterministic_decision":
        value = context.decision_envelope
        expected = arguments.get("decision_envelope_digest")
        actual = value.get("decision_envelope_digest") if isinstance(value, Mapping) else None
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "overall_outcome": value.get("overall_outcome") if isinstance(value, Mapping) else None,
            "claim_ceiling": value.get("claim_ceiling") if isinstance(value, Mapping) else None,
            "verdict_changed_by_tool": False,
        }
    else:  # pragma: no cover - construction prevents this branch
        raise ValueError(f"registered_non_spend_tool_not_implemented:{tool_id}")
    if not typed_result.get("contract_present") or not typed_result.get("digest_matches"):
        raise ValueError(f"registered_tool_bound_artifact_mismatch:{tool_id}")
    return typed_result, produced_artifact_references


def non_spend_tool_bindings(
    *,
    capability: str,
    context: Any,
    registry: ToolRegistry,
    authority: Mapping[str, Any],
) -> tuple[RegisteredToolBinding, ...]:
    """Bind only capability-scoped read tools in execute_non_spend mode."""

    validated_authority = AuthorityEnvelope.from_mapping(authority).to_mapping()
    if validated_authority.get("mode") not in {
        AutonomyMode.EXECUTE_NON_SPEND.value,
        AutonomyMode.EXECUTE_PREAUTHORIZED.value,
    }:
        return ()
    bindings: list[RegisteredToolBinding] = []
    for tool_id in _CAPABILITY_TOOL_IDS.get(capability, ()):
        descriptor = registry.resolve(tool_id)
        if descriptor is None:
            raise ValueError(f"registered_non_spend_tool_missing:{tool_id}")
        descriptor_value = descriptor.to_mapping()
        if validated_authority["mode"] not in set(descriptor_value.get("allowed_modes") or []):
            continue

        def invoke(
            arguments: Mapping[str, Any],
            *,
            selected_tool_id: str = tool_id,
            selected_descriptor_value: Mapping[str, Any] = descriptor_value,
        ) -> Mapping[str, Any]:
            try:
                proposal = ActionProposal.from_mapping(
                    {
                        "schema_version": "task_evaluation_supervisor_action_proposal.v1",
                        "proposal_id": f"{context.run_id}-{selected_tool_id}-sdk-call",
                        "run_id": context.run_id,
                        "capability": capability,
                        "action_type": (
                            "registered_read_only_tool_call"
                            if selected_descriptor_value["mutability"] == "read_only"
                            else "registered_scoped_tool_call"
                        ),
                        "tool_id": selected_tool_id,
                        "parameters": dict(arguments),
                        "reasons": ["agents_sdk_requested_registered_observation"],
                        "evidence_refs": [],
                        "estimated_cost_usd": 0.0,
                        "requested_proof_effect": "none",
                        "disposition": "shadow_only",
                    }
                )
                disposition, blockers = registry.disposition(
                    proposal.to_mapping(), validated_authority
                )
                if disposition != "eligible" or blockers:
                    raise ValueError(
                        f"registered_tool_call_refused:{selected_tool_id}:{','.join(blockers)}"
                    )
                typed_result, produced_artifact_references = _bound_artifact(
                    context,
                    tool_id=selected_tool_id,
                    arguments=arguments,
                )
                status = (
                    "completed"
                    if selected_tool_id != "execute_preauthorized_recovery"
                    or typed_result.get("status") == "completed"
                    else "failed"
                )
                typed_failure = typed_result.get("typed_failure") if status == "failed" else None
            except ValueError as exc:
                typed_result = {}
                produced_artifact_references = []
                status = "refused"
                typed_failure = {
                    "failure_type": "deterministic_tool_refusal",
                    "reason": str(exc),
                    "retryable": False,
                }
            observation: dict[str, Any] = {
                "schema_version": TOOL_OBSERVATION_SCHEMA_VERSION,
                "run_id": context.run_id,
                "capability": capability,
                "tool_id": selected_tool_id,
                "tool_version": "1",
                "status": status,
                "typed_result": typed_result,
                "typed_failure": typed_failure,
                "produced_artifact_references": produced_artifact_references,
                "input_digest": canonical_digest(dict(arguments)),
                "output_digest": canonical_digest(typed_result),
                "runtime_identity": (
                    "blueprint_preauthorized_recovery_controller"
                    if selected_tool_id == "execute_preauthorized_recovery"
                    else "blueprint_local_deterministic_non_spend"
                ),
                "mutability": selected_descriptor_value["mutability"],
                "cost_usd": float(typed_result.get("actual_cost_usd") or 0.0),
                "duration_seconds": float(typed_result.get("duration_seconds") or 0.0),
                "retries": max(0, int(typed_result.get("attempt_number") or 1) - 1),
                "authority_digest": validated_authority["authority_digest"],
                "proof_effect": "none",
                "warnings": ["tool_observation_is_not_accepted_evidence"],
                "suggested_next_legal_actions": list(
                    typed_result.get("suggested_next_legal_actions") or []
                ),
            }
            observation["observation_digest"] = canonical_digest(
                observation, digest_field="observation_digest"
            )
            return validate_tool_observation_binding(
                observation,
                run_id=context.run_id,
                capability=capability,
                registry=registry,
                authority=validated_authority,
            )

        bindings.append(
            RegisteredToolBinding(
                tool_id=tool_id,
                description=(
                    f"Blueprint registered read-only tool {tool_id}. Returns a typed "
                    "non-authoritative observation with proof_effect=none."
                ),
                input_schema=dict(descriptor_value["input_schema"]),
                timeout_seconds=float(descriptor_value["timeout_seconds"]),
                invoke=invoke,
            )
        )
    return tuple(bindings)


__all__ = [
    "RegisteredToolBinding",
    "TOOL_OBSERVATION_SCHEMA_VERSION",
    "TOOL_REGISTRY_SCHEMA_VERSION",
    "ToolRegistry",
    "default_tool_descriptors",
    "non_spend_tool_bindings",
    "validate_tool_observation_binding",
]
