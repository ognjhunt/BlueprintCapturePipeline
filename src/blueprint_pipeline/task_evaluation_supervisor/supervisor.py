"""Durable Task Evaluation Supervisor state machine.

Phase 1 runs all six capabilities in shadow mode.  It records proposals and
tool eligibility but executes no action and cannot mutate the proof kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..common import read_json, utc_now_iso, write_json
from ..decision_evidence_contracts import (
    DecisionEnvelope,
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    EvidencePlan,
    MaintainedSiteTaskTestbed,
    NormalizedEvidenceResult,
    QualificationRecord,
    canonical_digest,
)
from .agents_sdk import (
    AGENTS_SDK_HARNESS_ID,
    DEFAULT_SUPERVISOR_AGENT_MODEL,
    AgentsSDKInvoker,
    agents_sdk_capabilities,
)
from .capabilities import (
    SupervisorCapability,
    SupervisorContext,
    capability_instruction_digest,
)
from .contracts import (
    AgentInvocationManifest,
    AuthorityEnvelope,
    AutonomyMode,
    CapabilityKind,
    CapabilityResult,
    SupervisorEvent,
    SupervisorPhase,
    SupervisorRun,
    SupervisorState,
    TerminalSupervisorReport,
    proof_boundary,
)
from .ledger import AppendOnlyEventLedger
from .phase2_artifacts import deterministic_customer_report, write_phase2_artifact
from .recovery import PreauthorizedRecoveryController
from .tools import ToolRegistry


@dataclass(frozen=True)
class SupervisorExecution:
    run: SupervisorRun
    state: SupervisorState
    report: TerminalSupervisorReport
    capability_results: tuple[CapabilityResult, ...]
    invocation_manifests: tuple[AgentInvocationManifest, ...]
    output_dir: Path


def _validated_context(context: SupervisorContext) -> SupervisorContext:
    capture_build = dict(context.capture_build) if context.capture_build is not None else None
    if capture_build is not None:
        expected = capture_build.get("capture_build_digest")
        actual = canonical_digest(capture_build, digest_field="capture_build_digest")
        if expected != actual:
            raise ValueError("capture_build_digest_mismatch")
    return replace(
        context,
        capture_build=capture_build,
        decision_request=(
            DecisionEvidenceRequest.from_mapping(context.decision_request).to_mapping()
            if context.decision_request is not None
            else None
        ),
        testbed=(
            MaintainedSiteTaskTestbed.from_mapping(context.testbed).to_mapping()
            if context.testbed is not None
            else None
        ),
        method_profiles=tuple(
            EvidenceMethodProfile.from_mapping(value).to_mapping()
            for value in context.method_profiles
        ),
        qualifications=tuple(
            QualificationRecord.from_mapping(value).to_mapping() for value in context.qualifications
        ),
        evidence_plan=(
            EvidencePlan.from_mapping(context.evidence_plan).to_mapping()
            if context.evidence_plan is not None
            else None
        ),
        evidence_results=tuple(
            NormalizedEvidenceResult.from_mapping(value).to_mapping()
            for value in context.evidence_results
        ),
        decision_envelope=(
            DecisionEnvelope.from_mapping(context.decision_envelope).to_mapping()
            if context.decision_envelope is not None
            else None
        ),
    )


def _write_kernel_inputs(root: Path, context: SupervisorContext) -> dict[str, Any]:
    inputs_root = root / "kernel_inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    def record(name: str, value: Mapping[str, Any]) -> None:
        path = inputs_root / f"{name}.json"
        write_json(path, value)
        artifacts.append(
            {
                "name": name,
                "artifact_path": f"kernel_inputs/{name}.json",
                "digest": canonical_digest(value),
            }
        )

    for name, value in (
        ("capture_build", context.capture_build),
        ("decision_request", context.decision_request),
        ("site_task_testbed", context.testbed),
        ("evidence_plan", context.evidence_plan),
        ("decision_envelope", context.decision_envelope),
    ):
        if isinstance(value, Mapping):
            record(name, value)
    for prefix, values in (
        ("method_profile", context.method_profiles),
        ("qualification", context.qualifications),
        ("evidence_result", context.evidence_results),
    ):
        for index, value in enumerate(values):
            record(f"{prefix}_{index}", value)
    manifest: dict[str, Any] = {
        "schema_version": "task_evaluation_supervisor_kernel_inputs.v1",
        "run_id": context.run_id,
        "artifacts": artifacts,
        "agent_output_included": False,
        "hidden_labels_included": False,
    }
    manifest["kernel_inputs_manifest_digest"] = canonical_digest(
        manifest, digest_field="kernel_inputs_manifest_digest"
    )
    write_json(root / "kernel_inputs_manifest.json", manifest)
    return manifest


def _input_digests(context: SupervisorContext) -> list[str]:
    candidates: list[Any] = []
    for value, key in (
        (context.capture_build, "capture_build_digest"),
        (context.decision_request, "request_digest"),
        (context.testbed, "testbed_digest"),
        (context.evidence_plan, "plan_digest"),
        (context.decision_envelope, "decision_envelope_digest"),
    ):
        if isinstance(value, Mapping):
            candidates.append(value.get(key))
    for value in context.method_profiles:
        candidates.append(value.get("method_profile_digest"))
    for value in context.qualifications:
        candidates.append(value.get("qualification_digest"))
    for value in context.evidence_results:
        candidates.append(value.get("result_digest"))
    return sorted({str(value) for value in candidates if str(value or "").startswith("sha256:")})


def default_authority_envelope(
    *,
    run_id: str,
    mode: AutonomyMode,
    tool_registry: ToolRegistry,
    immutable_input_digests: Sequence[str],
    agent_inference_budget_usd: float = 0.0,
    allow_agent_inference: bool = False,
    action_max_cost_usd: float = 0.0,
    action_max_retries: int = 0,
    action_ttl_seconds: int = 300,
    preauthorization_receipt_digest: str | None = None,
    preauthorization_expires_at: str | None = None,
) -> AuthorityEnvelope:
    manifest = tool_registry.manifest()
    allowed_tool_ids = [row["tool_id"] for row in manifest["tools"]]
    return AuthorityEnvelope.from_mapping(
        {
            "schema_version": "task_evaluation_supervisor_authority.v1",
            "authority_id": f"{run_id}-authority",
            "mode": mode.value,
            "allowed_tool_ids": [] if mode is AutonomyMode.DISABLED else allowed_tool_ids,
            "max_cost_usd": action_max_cost_usd,
            "agent_inference_budget_usd": agent_inference_budget_usd,
            "agent_inference_allowed": allow_agent_inference,
            "action_spend_allowed": (
                mode is AutonomyMode.EXECUTE_PREAUTHORIZED and action_max_cost_usd > 0
            ),
            "max_duration_seconds": action_ttl_seconds,
            "max_retries": action_max_retries,
            "expires_at": preauthorization_expires_at,
            "preauthorization_receipt_digest": preauthorization_receipt_digest,
            "immutable_input_digests": sorted(set(immutable_input_digests)),
            "proof_mutation_allowed": False,
            "rights_mutation_allowed": False,
            "budget_mutation_allowed": False,
            "hidden_labels_accessible": False,
            "physical_action_allowed": False,
            "external_processing_allowed": allow_agent_inference,
        }
    )


_CAPABILITY_PHASE = {
    CapabilityKind.CLAIM_TASK_INTERPRETER: SupervisorPhase.INTERPRETING,
    CapabilityKind.CAPTURE_TESTBED_SUPERVISOR: SupervisorPhase.INSPECTING,
    CapabilityKind.EVALUATION_METHOD_ROUTER: SupervisorPhase.PLANNING,
    CapabilityKind.RUNTIME_FAILURE_RECOVERY: SupervisorPhase.DIAGNOSING,
    CapabilityKind.SCENARIO_ADVERSARIAL_PROPOSER: SupervisorPhase.PLANNING,
    CapabilityKind.POST_RUN_DIAGNOSTICIAN: SupervisorPhase.DIAGNOSING,
}


class TaskEvaluationSupervisor:
    """One persistent supervisor for one Task Evaluation Run."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        capabilities: Sequence[SupervisorCapability] | None = None,
        agents_sdk_invoker: AgentsSDKInvoker | None = None,
        agent_model: str = DEFAULT_SUPERVISOR_AGENT_MODEL,
        allow_live_agents_sdk: bool = False,
        agent_inference_budget_usd: float = 0.0,
        recovery_controller: PreauthorizedRecoveryController | None = None,
    ) -> None:
        self.tool_registry = tool_registry or ToolRegistry.default()
        self.agent_inference_budget_usd = agent_inference_budget_usd
        self.allow_live_agents_sdk = allow_live_agents_sdk
        self.recovery_controller = recovery_controller
        self.capabilities = tuple(
            capabilities
            or agents_sdk_capabilities(
                tool_registry_manifest=self.tool_registry.manifest(),
                tool_registry=self.tool_registry,
                invoker=agents_sdk_invoker,
                model=agent_model,
                allow_live=allow_live_agents_sdk,
                max_inference_cost_usd=agent_inference_budget_usd,
            )
        )
        kinds = [capability.kind for capability in self.capabilities]
        if len(kinds) != len(set(kinds)):
            raise ValueError("duplicate_supervisor_capability")

    def _event(
        self,
        *,
        ledger: AppendOnlyEventLedger,
        run_id: str,
        phase: SupervisorPhase,
        event_type: str,
        generated_at: str,
        payload_digest: str | None,
    ) -> SupervisorEvent:
        existing = ledger.read()
        sequence = len(existing)
        previous = existing[-1].digest if existing else None
        event = SupervisorEvent.from_mapping(
            {
                "schema_version": "task_evaluation_supervisor_event.v1",
                "event_id": f"{run_id}-event-{sequence}",
                "run_id": run_id,
                "sequence": sequence,
                "event_type": event_type,
                "phase": phase.value,
                "previous_event_digest": previous,
                "payload_digest": payload_digest,
                "proof_effect": "none",
                "generated_at": generated_at,
            }
        )
        return ledger.append(event.to_mapping())

    def run(
        self,
        context: SupervisorContext,
        *,
        output_dir: str | Path,
        mode: AutonomyMode | str = AutonomyMode.SHADOW,
        generated_at: str | None = None,
        resume: bool = True,
    ) -> SupervisorExecution:
        context = _validated_context(context)
        # Canonicalize once so every artifact writer and relative-path binding
        # uses the same root. On macOS, for example, /tmp aliases /private/tmp;
        # mixing those spellings makes valid generated artifacts appear to
        # escape the supervisor run directory.
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        ledger_path = root / "supervisor_events.jsonl"

        try:
            selected_mode = mode if isinstance(mode, AutonomyMode) else AutonomyMode(str(mode))
        except ValueError as exc:
            raise ValueError(f"unsupported_supervisor_autonomy_mode:{mode}") from exc
        persisted_run_value: dict[str, Any] | None = None
        run_path = root / "task_evaluation_supervisor_run.json"
        if ledger_path.exists() and ledger_path.stat().st_size:
            if not resume:
                raise ValueError("supervisor_output_event_ledger_already_exists")
            if not run_path.is_file():
                raise ValueError("supervisor_resume_run_artifact_missing")
            persisted_run_value = dict(read_json(run_path))
        timestamp = str(
            (persisted_run_value or {}).get("generated_at") or generated_at or utc_now_iso()
        )
        boundary = proof_boundary()
        tool_manifest = self.tool_registry.manifest()
        input_digests = _input_digests(context)
        authority = default_authority_envelope(
            run_id=context.run_id,
            mode=selected_mode,
            tool_registry=self.tool_registry,
            immutable_input_digests=input_digests,
            agent_inference_budget_usd=self.agent_inference_budget_usd,
            allow_agent_inference=(
                self.allow_live_agents_sdk
                and selected_mode
                in {
                    AutonomyMode.SHADOW,
                    AutonomyMode.EXECUTE_NON_SPEND,
                    AutonomyMode.EXECUTE_PREAUTHORIZED,
                }
            ),
            action_max_cost_usd=(
                self.recovery_controller.max_cost_usd
                if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
                and self.recovery_controller is not None
                else 0.0
            ),
            action_max_retries=(
                int(
                    self.recovery_controller.policy.receipt.get("granted_retry_count") or 0
                )
                if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
                and self.recovery_controller is not None
                else 0
            ),
            action_ttl_seconds=(
                int(self.recovery_controller.policy.receipt.get("granted_ttl_seconds") or 300)
                if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
                and self.recovery_controller is not None
                else 300
            ),
            preauthorization_receipt_digest=(
                str(
                    self.recovery_controller.policy.receipt.get(
                        "authorization_receipt_digest"
                    )
                )
                if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
                and self.recovery_controller is not None
                else None
            ),
            preauthorization_expires_at=(
                str(self.recovery_controller.policy.receipt.get("expires_at"))
                if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
                and self.recovery_controller is not None
                else None
            ),
        )
        capability_context = replace(
            context,
            autonomy_mode=selected_mode.value,
            authority_envelope=authority.to_mapping(),
            supervisor_output_dir=str(root),
            recovery_controller=self.recovery_controller,
        )
        capability_names = [capability.kind.value for capability in self.capabilities]
        run = SupervisorRun.from_mapping(
            {
                "schema_version": "task_evaluation_supervisor_run.v1",
                "run_id": context.run_id,
                "customer_question": context.customer_question,
                "mode": selected_mode.value,
                "authority_digest": authority.digest,
                "tool_registry_digest": self.tool_registry.digest,
                "proof_boundary_digest": boundary["proof_boundary_digest"],
                "input_artifact_digests": input_digests,
                "capabilities": capability_names,
                "agent_harness": AGENTS_SDK_HARNESS_ID,
                "status": "disabled" if selected_mode is AutonomyMode.DISABLED else "initialized",
                "generated_at": timestamp,
            }
        )
        if persisted_run_value is not None:
            persisted_run = SupervisorRun.from_mapping(persisted_run_value)
            if persisted_run.digest != run.digest:
                raise ValueError("supervisor_resume_run_contract_mismatch")
            for artifact_name, expected_digest, artifact_type in (
                ("authority_envelope.json", authority.digest, AuthorityEnvelope),
            ):
                artifact_path = root / artifact_name
                if not artifact_path.is_file():
                    raise ValueError(f"supervisor_resume_artifact_missing:{artifact_name}")
                persisted = artifact_type.from_mapping(read_json(artifact_path))
                if persisted.digest != expected_digest:
                    raise ValueError(f"supervisor_resume_artifact_mismatch:{artifact_name}")
            persisted_tool_manifest = read_json(root / "tool_registry_manifest.json")
            if persisted_tool_manifest.get("tool_registry_digest") != self.tool_registry.digest:
                raise ValueError("supervisor_resume_tool_registry_mismatch")
            persisted_boundary = read_json(root / "proof_boundary.json")
            if persisted_boundary.get("proof_boundary_digest") != boundary["proof_boundary_digest"]:
                raise ValueError("supervisor_resume_proof_boundary_mismatch")
        write_json(root / "task_evaluation_supervisor_run.json", run.to_mapping())
        write_json(root / "authority_envelope.json", authority.to_mapping())
        write_json(root / "tool_registry_manifest.json", tool_manifest)
        write_json(root / "proof_boundary.json", boundary)
        _write_kernel_inputs(root, context)

        ledger = AppendOnlyEventLedger(ledger_path)
        existing_events = ledger.read()
        if existing_events:
            first_event = existing_events[0].to_mapping()
            if (
                first_event.get("event_type") != "supervisor_run_received"
                or first_event.get("payload_digest") != run.digest
            ):
                raise ValueError("supervisor_resume_initial_event_mismatch")
            last_event = existing_events[-1]
        else:
            last_event = self._event(
                ledger=ledger,
                run_id=context.run_id,
                phase=SupervisorPhase.RECEIVED,
                event_type="supervisor_run_received",
                generated_at=timestamp,
                payload_digest=run.digest,
            )

        results: list[CapabilityResult] = []
        invocations: list[AgentInvocationManifest] = []
        completed_capabilities: set[CapabilityKind] = set()
        for event in ledger.read()[1:]:
            event_value = event.to_mapping()
            prefix = "capability_proposal_recorded:"
            event_type = str(event_value.get("event_type") or "")
            if not event_type.startswith(prefix):
                continue
            try:
                completed_kind = CapabilityKind(event_type.removeprefix(prefix))
            except ValueError as exc:
                raise ValueError("supervisor_resume_unknown_capability_event") from exc
            if completed_kind in completed_capabilities:
                raise ValueError("supervisor_resume_duplicate_capability_event")
            result_path = root / "capabilities" / f"{completed_kind.value}.json"
            invocation_path = root / "invocations" / f"{completed_kind.value}.json"
            if not result_path.is_file() or not invocation_path.is_file():
                raise ValueError("supervisor_resume_committed_artifact_missing")
            result = CapabilityResult.from_mapping(read_json(result_path))
            invocation = AgentInvocationManifest.from_mapping(read_json(invocation_path))
            if (
                result.to_mapping().get("run_id") != context.run_id
                or result.to_mapping().get("capability") != completed_kind.value
                or result.digest != event_value.get("payload_digest")
                or invocation.to_mapping().get("run_id") != context.run_id
                or invocation.to_mapping().get("capability") != completed_kind.value
                or invocation.to_mapping().get("structured_output_digest") != result.digest
            ):
                raise ValueError("supervisor_resume_committed_artifact_mismatch")
            completed_capabilities.add(completed_kind)
            results.append(result)
            invocations.append(invocation)
        run_blockers: list[str] = []
        for result in results:
            if result.to_mapping().get("status") == "blocked":
                run_blockers.append(f"capability_blocked:{result.to_mapping()['capability']}")
        reported_cost_usd = sum(
            float(row.to_mapping().get("cost_usd") or 0.0) for row in invocations
        )
        reserved_cost_usd = max(
            (
                float(
                    (row.to_mapping().get("budget_state") or {}).get(
                        "cumulative_reserved_cost_usd", 0.0
                    )
                    or 0.0
                )
                for row in invocations
            ),
            default=0.0,
        )
        live_invocation_count = sum(
            1
            for row in invocations
            if row.to_mapping().get("provider") == "openai"
            and row.to_mapping().get("validation_status") == "accepted_as_proposal"
        )
        can_invoke = selected_mode in {
            AutonomyMode.SHADOW,
            AutonomyMode.EXECUTE_NON_SPEND,
            AutonomyMode.EXECUTE_PREAUTHORIZED,
        }
        if selected_mode not in {
            AutonomyMode.DISABLED,
            AutonomyMode.SHADOW,
            AutonomyMode.EXECUTE_NON_SPEND,
            AutonomyMode.EXECUTE_PREAUTHORIZED,
        }:
            run_blockers.append(f"autonomy_mode_not_enabled_in_phase1:{selected_mode.value}")
        if (
            selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
            and self.recovery_controller is None
        ):
            can_invoke = False
            run_blockers.append("preauthorized_recovery_controller_missing")

        capabilities_dir = root / "capabilities"
        invocations_dir = root / "invocations"
        if can_invoke:
            capabilities_dir.mkdir(parents=True, exist_ok=True)
            invocations_dir.mkdir(parents=True, exist_ok=True)
            for capability in self.capabilities:
                if capability.kind in completed_capabilities:
                    continue
                try:
                    proposed = capability.propose(capability_context)
                    result = CapabilityResult.from_mapping(proposed.to_mapping())
                    dispositions: list[dict[str, Any]] = []
                    for proposal in result.to_mapping()["proposals"]:
                        disposition, blockers = self.tool_registry.disposition(
                            proposal, authority.to_mapping()
                        )
                        dispositions.append(
                            {
                                "proposal_id": proposal["proposal_id"],
                                "disposition": disposition,
                                "blockers": list(blockers),
                                "executed": False,
                            }
                        )
                    result_value = result.to_mapping()
                    result_value["proposal_dispositions"] = dispositions
                    result_value.pop("capability_result_digest", None)
                    result = CapabilityResult.from_mapping(result_value)
                    validation_status = "accepted_as_proposal"
                except Exception as exc:  # noqa: BLE001 - persist bounded refusal evidence
                    result = CapabilityResult.from_mapping(
                        {
                            "schema_version": "task_evaluation_supervisor_capability_result.v1",
                            "result_id": f"{context.run_id}-{capability.kind.value}",
                            "run_id": context.run_id,
                            "capability": capability.kind.value,
                            "status": "blocked",
                            "artifact": {
                                "schema_version": "supervisor_capability_refusal.v1",
                                "error_type": type(exc).__name__,
                                "raw_error_message_recorded": False,
                            },
                            "proposals": [],
                            "proposal_dispositions": [],
                            "blockers": ["capability_output_invalid_or_execution_failed"],
                            "evidence_refs": [],
                            "authoritative": False,
                            "proof_booleans_mutable": False,
                            "proof_effect": "none",
                        }
                    )
                    validation_status = "refused"
                    run_blockers.append(f"capability_blocked:{capability.kind.value}")

                result_path = capabilities_dir / f"{capability.kind.value}.json"
                write_json(result_path, result.to_mapping())
                input_digest = canonical_digest(
                    {
                        "customer_question": context.customer_question,
                        "input_artifact_digests": input_digests,
                        "capability": capability.kind.value,
                    }
                )
                metadata_getter = getattr(capability, "invocation_metadata", None)
                invocation_metadata = dict(metadata_getter()) if callable(metadata_getter) else {}
                tool_observations = [
                    dict(row)
                    for row in invocation_metadata.get("tool_observations") or []
                    if isinstance(row, Mapping)
                ]
                observation_refs: list[dict[str, Any]] = []
                if tool_observations:
                    observations_dir = root / "observations"
                    observations_dir.mkdir(parents=True, exist_ok=True)
                    for ordinal, observation in enumerate(tool_observations):
                        observation_path = (
                            observations_dir / f"{capability.kind.value}-{ordinal:03d}.json"
                        )
                        write_json(observation_path, observation)
                        observation_refs.append(
                            {
                                "artifact_path": str(observation_path.relative_to(root)),
                                "digest": observation.get("observation_digest"),
                            }
                        )
                usage = dict(invocation_metadata.get("usage") or {})
                reported_cost = invocation_metadata.get("cost_usd")
                cost_usd = float(reported_cost) if isinstance(reported_cost, (int, float)) else 0.0
                reported_cost_usd += cost_usd
                cumulative_reserved = usage.get("cumulative_reserved_cost_usd")
                if isinstance(cumulative_reserved, (int, float)):
                    reserved_cost_usd = max(reserved_cost_usd, float(cumulative_reserved))
                remaining_unreserved_usd = max(
                    0.0, self.agent_inference_budget_usd - reserved_cost_usd
                )
                is_live_invocation = (
                    invocation_metadata.get("provider") == "openai"
                    and validation_status == "accepted_as_proposal"
                )
                if is_live_invocation:
                    live_invocation_count += 1
                observation_mutabilities = {
                    str(row.get("mutability") or "") for row in tool_observations
                }
                action_taken = (
                    "registered_preauthorized_action_attempted"
                    if "external_side_effect" in observation_mutabilities
                    else "registered_non_spend_actions_executed"
                    if "reversible_mutation" in observation_mutabilities
                    else "registered_read_only_tool_calls"
                    if tool_observations
                    else "none_shadow_mode"
                )
                invocation = AgentInvocationManifest.from_mapping(
                    {
                        "schema_version": "task_evaluation_supervisor_invocation.v1",
                        "invocation_id": f"{context.run_id}-{capability.kind.value}-invocation",
                        "run_id": context.run_id,
                        "capability": capability.kind.value,
                        "provider": invocation_metadata.get("provider")
                        or "deterministic_evaluation_baseline",
                        "model": invocation_metadata.get("model"),
                        "agent_harness": AGENTS_SDK_HARNESS_ID,
                        "agents_sdk_version": invocation_metadata.get("sdk_version"),
                        "adapter_id": capability.adapter_id,
                        "adapter_version": capability.adapter_version,
                        "instruction_digest": capability_instruction_digest(capability),
                        "tool_registry_digest": self.tool_registry.digest,
                        "authority_digest": authority.digest,
                        "input_artifact_digests": [input_digest, *input_digests],
                        "budget_state": {
                            "max_cost_usd": self.agent_inference_budget_usd,
                            "reported_cost_usd": reported_cost_usd,
                            "cumulative_reserved_cost_usd": reserved_cost_usd,
                            "remaining_unreserved_usd": remaining_unreserved_usd,
                        },
                        "structured_output_digest": result.digest,
                        "validation_status": validation_status,
                        "action_taken": action_taken,
                        "refusal": validation_status == "refused",
                        "usage": usage,
                        "trace_id": invocation_metadata.get("trace_id"),
                        "cost_usd": cost_usd,
                        "cost_status": invocation_metadata.get("cost_status") or "not_applicable",
                        "latency_seconds": float(invocation_metadata.get("latency_seconds") or 0.0),
                        "proof_effect": "none",
                        "uncertainty": "not_a_proof_signal",
                        "tool_observation_references": observation_refs,
                        "parent_event_digest": last_event.digest,
                        "generated_at": timestamp,
                    }
                )
                write_json(
                    invocations_dir / f"{capability.kind.value}.json", invocation.to_mapping()
                )
                results.append(result)
                invocations.append(invocation)
                last_event = self._event(
                    ledger=ledger,
                    run_id=context.run_id,
                    phase=_CAPABILITY_PHASE[capability.kind],
                    event_type=f"capability_proposal_recorded:{capability.kind.value}",
                    generated_at=timestamp,
                    payload_digest=result.digest,
                )
                completed_capabilities.add(capability.kind)
                in_progress_state = SupervisorState.from_mapping(
                    {
                        "schema_version": "task_evaluation_supervisor_state.v1",
                        "run_id": context.run_id,
                        "mode": selected_mode.value,
                        "phase": _CAPABILITY_PHASE[capability.kind].value,
                        "next_sequence": len(ledger.read()),
                        "last_event_digest": last_event.digest,
                        "completed_capabilities": [
                            item.kind.value
                            for item in self.capabilities
                            if item.kind in completed_capabilities
                        ],
                        "terminal": False,
                        "spent_cost_usd": reported_cost_usd,
                        "remaining_cost_usd": remaining_unreserved_usd,
                        "proof_state_mutated_by_agent": False,
                        "terminal_report_digest": None,
                    }
                )
                write_json(root / "supervisor_state.json", in_progress_state.to_mapping())

        executed_tool_observations: list[dict[str, Any]] = []
        for invocation in invocations:
            for reference in invocation.to_mapping().get("tool_observation_references") or []:
                if not isinstance(reference, Mapping):
                    continue
                observation_path = root / str(reference.get("artifact_path") or "")
                if observation_path.is_file():
                    observation = read_json(observation_path)
                    if observation.get("status") in {"completed", "failed"}:
                        executed_tool_observations.append(observation)
        registered_tool_reads_executed = sum(
            1
            for observation in executed_tool_observations
            if observation.get("status") == "completed"
            and observation.get("mutability", "read_only") == "read_only"
        )
        registered_non_spend_actions_executed = sum(
            1
            for observation in executed_tool_observations
            if observation.get("status") == "completed"
            and observation.get("mutability") == "reversible_mutation"
        )
        registered_preauthorized_actions_executed = sum(
            1
            for observation in executed_tool_observations
            if observation.get("mutability") == "external_side_effect"
        )
        generated_artifact_references = [
            dict(reference)
            for observation in executed_tool_observations
            for reference in observation.get("produced_artifact_references") or []
            if isinstance(reference, Mapping)
        ]
        preauthorized_failures = any(
            observation.get("mutability") == "external_side_effect"
            and observation.get("status") != "completed"
            for observation in executed_tool_observations
        )
        terminal_status = (
            "disabled"
            if selected_mode is AutonomyMode.DISABLED
            else "blocked"
            if run_blockers
            else "non_spend_complete"
            if selected_mode is AutonomyMode.EXECUTE_NON_SPEND
            else "preauthorized_complete_with_failures"
            if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
            and preauthorized_failures
            else "preauthorized_complete"
            if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
            else "shadow_complete"
        )
        terminal_payload_digest = canonical_digest(
            {
                "run_id": context.run_id,
                "status": terminal_status,
                "capability_result_digests": [result.digest for result in results],
                "blockers": sorted(set(run_blockers)),
            }
        )
        customer_report = deterministic_customer_report(
            context=context,
            capability_results=[result.to_mapping() for result in results],
            invocation_manifests=[invocation.to_mapping() for invocation in invocations],
            generated_artifact_references=generated_artifact_references,
            tool_observations=executed_tool_observations,
        )
        customer_report_path = write_phase2_artifact(
            root,
            "customer_decision_report.json",
            customer_report,
        )
        terminal_events = [
            event
            for event in ledger.read()
            if str(event.to_mapping().get("event_type") or "").startswith(
                "supervisor_run_terminal:"
            )
        ]
        if terminal_events:
            if len(terminal_events) != 1:
                raise ValueError("supervisor_resume_duplicate_terminal_event")
            last_event = terminal_events[0]
            terminal_event = last_event.to_mapping()
            if (
                terminal_event.get("event_type") != f"supervisor_run_terminal:{terminal_status}"
                or terminal_event.get("payload_digest") != terminal_payload_digest
            ):
                raise ValueError("supervisor_resume_terminal_event_mismatch")
        else:
            last_event = self._event(
                ledger=ledger,
                run_id=context.run_id,
                phase=SupervisorPhase.TERMINAL,
                event_type=f"supervisor_run_terminal:{terminal_status}",
                generated_at=timestamp,
                payload_digest=terminal_payload_digest,
            )
        event_count = len(ledger.read())
        report = TerminalSupervisorReport.from_mapping(
            {
                "schema_version": "task_evaluation_supervisor_report.v1",
                "run_id": context.run_id,
                "status": terminal_status,
                "mode": selected_mode.value,
                "customer_question": context.customer_question,
                "capability_results": [
                    {
                        "capability": result.to_mapping()["capability"],
                        "status": result.to_mapping()["status"],
                        "digest": result.digest,
                        "artifact_path": f"capabilities/{result.to_mapping()['capability']}.json",
                    }
                    for result in results
                ],
                "invocation_manifests": [
                    {
                        "capability": invocation.to_mapping()["capability"],
                        "digest": invocation.digest,
                        "artifact_path": f"invocations/{invocation.to_mapping()['capability']}.json",
                    }
                    for invocation in invocations
                ],
                "event_count": event_count,
                "last_event_digest": last_event.digest,
                "proof_boundary_digest": boundary["proof_boundary_digest"],
                "tool_registry_digest": self.tool_registry.digest,
                "blockers": sorted(set(run_blockers)),
                "authoritative_decision_source": "deterministic_decision_envelope_only",
                "authoritative_decision_produced_by_agent": False,
                "proof_state_mutated_by_agent": False,
                "actions_executed": (
                    registered_non_spend_actions_executed > 0
                    or registered_preauthorized_actions_executed > 0
                ),
                "registered_tool_reads_executed": registered_tool_reads_executed,
                "registered_non_spend_actions_executed": (
                    registered_non_spend_actions_executed
                ),
                "registered_preauthorized_actions_executed": (
                    registered_preauthorized_actions_executed
                ),
                "customer_report_path": str(customer_report_path.relative_to(root)),
                "customer_report_digest": customer_report["customer_report_digest"],
                "action_spend": {
                    "authorized_max_cost_usd": float(
                        authority.to_mapping().get("max_cost_usd") or 0.0
                    ),
                    "reported_actual_cost_usd": sum(
                        float(row.get("cost_usd") or 0.0)
                        for row in executed_tool_observations
                    ),
                    "reported_duration_seconds": sum(
                        float(row.get("duration_seconds") or 0.0)
                        for row in executed_tool_observations
                    ),
                    "preauthorization_receipt_digest": authority.to_mapping().get(
                        "preauthorization_receipt_digest"
                    ),
                },
                "inference_spend": {
                    "budget_usd": self.agent_inference_budget_usd,
                    "reserved_max_cost_usd": reserved_cost_usd,
                    "reported_cost_usd": reported_cost_usd,
                    "remaining_unreserved_usd": max(
                        0.0, self.agent_inference_budget_usd - reserved_cost_usd
                    ),
                    "live_invocation_count": live_invocation_count,
                    "reported_cost_is_final": live_invocation_count == 0,
                },
                "generated_at": timestamp,
            }
        )
        write_json(root / "terminal_supervisor_report.json", report.to_mapping())
        state = SupervisorState.from_mapping(
            {
                "schema_version": "task_evaluation_supervisor_state.v1",
                "run_id": context.run_id,
                "mode": selected_mode.value,
                "phase": SupervisorPhase.TERMINAL.value,
                "next_sequence": event_count,
                "last_event_digest": last_event.digest,
                "completed_capabilities": [result.to_mapping()["capability"] for result in results],
                "terminal": True,
                "spent_cost_usd": reported_cost_usd,
                "remaining_cost_usd": max(0.0, self.agent_inference_budget_usd - reserved_cost_usd),
                "proof_state_mutated_by_agent": False,
                "terminal_report_digest": report.digest,
            }
        )
        write_json(root / "supervisor_state.json", state.to_mapping())
        return SupervisorExecution(
            run=run,
            state=state,
            report=report,
            capability_results=tuple(results),
            invocation_manifests=tuple(invocations),
            output_dir=root,
        )


__all__ = [
    "SupervisorExecution",
    "TaskEvaluationSupervisor",
    "default_authority_envelope",
]
