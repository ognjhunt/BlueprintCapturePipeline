"""Durable, manager-led Task Evaluation Supervisor state machine.

The OpenAI Agents SDK manager selects eligible specialist capabilities and
replans from each validated result. All proposals remain outside the proof
kernel until deterministic contracts accept them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
    agents_sdk_capabilities,
)
from .capabilities import (
    SupervisorCapability,
    SupervisorContext,
    capability_instruction_digest,
)
from .capture_ingress import validate_capture_build_ingress
from .contracts import (
    AgentInvocationManifest,
    AuthorityEnvelope,
    AutonomyMode,
    CapabilityKind,
    CapabilityResult,
    SupervisorContractError,
    SupervisorEvent,
    SupervisorPhase,
    SupervisorRun,
    SupervisorState,
    TerminalSupervisorReport,
    proof_boundary,
)
from .ledger import AppendOnlyEventLedger
from .inference_reservations import InferenceReservationAudit
from .manager import (
    OpenAIAgentsSDKSupervisorManager,
    validate_manager_decision,
    validate_manager_invocation,
    validate_manager_refusal,
)
from .phase2_artifacts import (
    deterministic_customer_report,
    recapture_reinspection as build_recapture_reinspection,
    validate_clarification_receipt,
    validate_clarification_request,
    validate_authorization_receipt,
    validate_authorization_request,
    validate_targeted_recapture_receipt,
    validate_targeted_recapture_request,
    write_phase2_artifact,
)
from .recovery import PreauthorizedRecoveryController
from .tools import ToolRegistry, validate_tool_observation_binding


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
        capture_build = validate_capture_build_ingress(capture_build)
    testbed = (
        MaintainedSiteTaskTestbed.from_mapping(context.testbed).to_mapping()
        if context.testbed is not None
        else None
    )
    decision_request = (
        DecisionEvidenceRequest.from_mapping(context.decision_request).to_mapping()
        if context.decision_request is not None
        else None
    )
    clarification_request = (
        validate_clarification_request(context.clarification_request)
        if context.clarification_request is not None
        else None
    )
    clarification_receipt = (
        validate_clarification_receipt(
            context.clarification_receipt,
            request=clarification_request,
        )
        if context.clarification_receipt is not None
        else None
    )
    if clarification_receipt is not None and clarification_request is None:
        raise ValueError("clarification_receipt_requires_request")
    authorization_request = (
        validate_authorization_request(context.authorization_request)
        if context.authorization_request is not None
        else None
    )
    authorization_receipt = (
        validate_authorization_receipt(
            context.authorization_receipt,
            request=authorization_request,
        )
        if context.authorization_receipt is not None
        else None
    )
    if authorization_receipt is not None and authorization_request is None:
        raise ValueError("authorization_receipt_requires_request")
    recapture_request = (
        validate_targeted_recapture_request(context.targeted_recapture_request)
        if context.targeted_recapture_request is not None
        else None
    )
    recapture_receipt = (
        validate_targeted_recapture_receipt(
            context.targeted_recapture_receipt,
            request=recapture_request,
            capture_build=capture_build,
        )
        if context.targeted_recapture_receipt is not None
        else None
    )
    if recapture_receipt is not None and (recapture_request is None or capture_build is None):
        raise ValueError("targeted_recapture_receipt_requires_request_and_capture")
    if context.recapture_reinspection is not None:
        raise ValueError("recapture_reinspection_is_kernel_derived")
    derived_recapture_reinspection = (
        build_recapture_reinspection(
            run_id=context.run_id,
            request=recapture_request,
            receipt=recapture_receipt,
            capture_build=capture_build,
            testbed=testbed,
        )
        if recapture_request is not None
        and recapture_receipt is not None
        and capture_build is not None
        and testbed is not None
        else None
    )
    return replace(
        context,
        capture_build=capture_build,
        decision_request=decision_request,
        testbed=testbed,
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
        clarification_request=clarification_request,
        clarification_receipt=clarification_receipt,
        authorization_request=authorization_request,
        authorization_receipt=authorization_receipt,
        targeted_recapture_request=recapture_request,
        targeted_recapture_receipt=recapture_receipt,
        recapture_reinspection=derived_recapture_reinspection,
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
        ("clarification_request", context.clarification_request),
        ("clarification_receipt", context.clarification_receipt),
        ("authorization_request", context.authorization_request),
        ("authorization_receipt", context.authorization_receipt),
        ("targeted_recapture_request", context.targeted_recapture_request),
        ("targeted_recapture_receipt", context.targeted_recapture_receipt),
        ("recapture_reinspection", context.recapture_reinspection),
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
        (context.clarification_request, "clarification_request_digest"),
        (context.clarification_receipt, "clarification_receipt_digest"),
        (context.authorization_request, "authorization_request_digest"),
        (context.authorization_receipt, "authorization_receipt_digest"),
        (context.targeted_recapture_request, "targeted_recapture_request_digest"),
        (context.targeted_recapture_receipt, "targeted_recapture_receipt_digest"),
        (context.recapture_reinspection, "recapture_reinspection_digest"),
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
        supervisor_manager: OpenAIAgentsSDKSupervisorManager | None = None,
    ) -> None:
        self.tool_registry = tool_registry or ToolRegistry.default()
        self.agent_inference_budget_usd = agent_inference_budget_usd
        self.allow_live_agents_sdk = allow_live_agents_sdk
        self.recovery_controller = recovery_controller
        sdk_config = OpenAIAgentsSDKConfig(
            model=agent_model,
            allow_live_invocation=allow_live_agents_sdk,
            max_inference_cost_usd=agent_inference_budget_usd,
        )
        shared_sdk_invoker = agents_sdk_invoker or OpenAIAgentsSDKInvoker(sdk_config)
        self.capabilities = tuple(
            capabilities
            or agents_sdk_capabilities(
                tool_registry_manifest=self.tool_registry.manifest(),
                tool_registry=self.tool_registry,
                invoker=shared_sdk_invoker,
                model=agent_model,
                allow_live=allow_live_agents_sdk,
                max_inference_cost_usd=agent_inference_budget_usd,
            )
        )
        self.manager = supervisor_manager or OpenAIAgentsSDKSupervisorManager(
            invoker=shared_sdk_invoker,
            config=sdk_config,
            tool_registry_digest=self.tool_registry.digest,
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
        if self.recovery_controller is not None and context.authorization_receipt is not None:
            controller_receipt_digest = self.recovery_controller.policy.receipt.get(
                "authorization_receipt_digest"
            )
            if (
                context.authorization_receipt.get("authorization_receipt_digest")
                != controller_receipt_digest
            ):
                raise ValueError("recovery_controller_authorization_receipt_mismatch")
        # Canonicalize once so every artifact writer and relative-path binding
        # uses the same root. On macOS, for example, /tmp aliases /private/tmp;
        # mixing those spellings makes valid generated artifacts appear to
        # escape the supervisor run directory.
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        ledger_path = root / "supervisor_events.jsonl"
        inference_reservations = InferenceReservationAudit(
            run_root=root,
            run_id=context.run_id,
        )
        existing_inference_reservation_manifest = inference_reservations.manifest()
        configured_invokers: set[int] = set()
        for invoker_candidate in (
            getattr(self.manager, "invoker", None),
            *(getattr(capability, "invoker", None) for capability in self.capabilities),
        ):
            configure_audit = getattr(
                invoker_candidate,
                "configure_reservation_audit",
                None,
            )
            if not callable(configure_audit) or id(invoker_candidate) in configured_invokers:
                continue
            configure_audit(
                record_reservation=inference_reservations.record_reservation,
                record_completion=inference_reservations.record_completion,
                restored_reserved_cost_usd=float(
                    existing_inference_reservation_manifest["reserved_max_cost_usd"]
                ),
            )
            configured_invokers.add(id(invoker_candidate))

        try:
            selected_mode = mode if isinstance(mode, AutonomyMode) else AutonomyMode(str(mode))
        except ValueError as exc:
            raise ValueError(f"unsupported_supervisor_autonomy_mode:{mode}") from exc
        if (
            selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
            and self.recovery_controller is not None
            and (context.authorization_request is None or context.authorization_receipt is None)
        ):
            raise ValueError("preauthorized_recovery_requires_recorded_authorization_pair")
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
                    AutonomyMode.ADVISE,
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
                int(self.recovery_controller.policy.receipt.get("granted_retry_count") or 0)
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
                str(self.recovery_controller.policy.receipt.get("authorization_receipt_digest"))
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
                "manager_agent_required": True,
                "manager_adapter_id": self.manager.adapter_id,
                "manager_adapter_version": self.manager.adapter_version,
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
        manager_decisions: list[dict[str, Any]] = []
        manager_invocations: list[dict[str, Any]] = []
        manager_refusals: list[dict[str, Any]] = []
        completed_capabilities: set[CapabilityKind] = set()
        pending_capability: CapabilityKind | None = None
        manager_terminal_reason: str | None = None
        for event in ledger.read()[1:]:
            event_value = event.to_mapping()
            manager_prefix = "supervisor_manager_decision_recorded:"
            manager_refusal_prefix = "supervisor_manager_refused:"
            prefix = "capability_proposal_recorded:"
            event_type = str(event_value.get("event_type") or "")
            if event_type.startswith(manager_refusal_prefix):
                if manager_terminal_reason is not None or pending_capability is not None:
                    raise ValueError("supervisor_resume_manager_refusal_sequence_invalid")
                try:
                    manager_step = int(event_type.removeprefix(manager_refusal_prefix))
                except ValueError as exc:
                    raise ValueError("supervisor_resume_manager_refusal_step_invalid") from exc
                refusal_path = root / "manager" / "refusals" / f"step-{manager_step:03d}.json"
                if not refusal_path.is_file():
                    raise ValueError("supervisor_resume_manager_refusal_missing")
                refusal = dict(read_json(refusal_path))
                refusal = validate_manager_refusal(
                    refusal,
                    run_id=context.run_id,
                    completed_results=results,
                    step_index=manager_step,
                )
                refusal_digest = canonical_digest(
                    refusal,
                    digest_field="supervisor_manager_refusal_digest",
                )
                if (
                    refusal.get("schema_version") != "task_evaluation_supervisor_manager_refusal.v1"
                    or refusal.get("supervisor_manager_refusal_digest") != refusal_digest
                    or event_value.get("payload_digest") != refusal_digest
                    or refusal.get("step_index") != manager_step
                    or refusal.get("proof_effect") != "none"
                ):
                    raise ValueError("supervisor_resume_manager_refusal_mismatch")
                manager_refusals.append(refusal)
                manager_terminal_reason = "blocked"
                continue
            if event_type.startswith(manager_prefix):
                if pending_capability is not None or manager_terminal_reason is not None:
                    raise ValueError("supervisor_resume_manager_sequence_invalid")
                try:
                    manager_step = int(event_type.removeprefix(manager_prefix))
                except ValueError as exc:
                    raise ValueError("supervisor_resume_manager_step_invalid") from exc
                if manager_step != len(manager_decisions):
                    raise ValueError("supervisor_resume_manager_step_invalid")
                decision_path = root / "manager" / "decisions" / f"step-{manager_step:03d}.json"
                manager_invocation_path = (
                    root / "manager" / "invocations" / f"step-{manager_step:03d}.json"
                )
                if not decision_path.is_file() or not manager_invocation_path.is_file():
                    raise ValueError("supervisor_resume_manager_artifact_missing")
                decision = dict(read_json(decision_path))
                decision = validate_manager_decision(
                    decision,
                    context=capability_context,
                    completed_results=results,
                    step_index=manager_step,
                )
                decision_digest = canonical_digest(
                    decision,
                    digest_field="supervisor_manager_decision_digest",
                )
                if (
                    decision.get("schema_version")
                    != "task_evaluation_supervisor_manager_decision.v1"
                    or decision.get("supervisor_manager_decision_digest") != decision_digest
                    or event_value.get("payload_digest") != decision_digest
                    or decision.get("step_index") != manager_step
                    or decision.get("observed_capability_result_digests")
                    != sorted(result.digest for result in results)
                ):
                    raise ValueError("supervisor_resume_manager_decision_mismatch")
                manager_invocation = dict(read_json(manager_invocation_path))
                manager_invocation = validate_manager_invocation(
                    manager_invocation,
                    run_id=context.run_id,
                    step_index=manager_step,
                    structured_output_digest=decision_digest,
                    tool_registry_digest=self.tool_registry.digest,
                    authority_digest=authority.digest,
                    input_artifact_digests=[
                        *input_digests,
                        *decision["observed_capability_result_digests"],
                    ],
                    manager_adapter_id=self.manager.adapter_id,
                    manager_adapter_version=self.manager.adapter_version,
                    max_cost_usd=self.agent_inference_budget_usd,
                    parent_event_digest=event_value.get("previous_event_digest"),
                )
                invocation_digest = canonical_digest(
                    manager_invocation,
                    digest_field="manager_invocation_digest",
                )
                if (
                    manager_invocation.get("schema_version")
                    != "task_evaluation_supervisor_manager_invocation.v1"
                    or manager_invocation.get("manager_invocation_digest") != invocation_digest
                    or manager_invocation.get("structured_output_digest") != decision_digest
                ):
                    raise ValueError("supervisor_resume_manager_invocation_mismatch")
                if decision.get("status") == "continue":
                    try:
                        pending_capability = CapabilityKind(
                            str(decision.get("next_capability") or "")
                        )
                    except ValueError as exc:
                        raise ValueError("supervisor_resume_manager_capability_invalid") from exc
                elif decision.get("status") == "terminal":
                    manager_terminal_reason = str(decision.get("terminal_reason") or "")
                    if not manager_terminal_reason:
                        raise ValueError("supervisor_resume_manager_terminal_invalid")
                else:
                    raise ValueError("supervisor_resume_manager_status_invalid")
                manager_decisions.append(decision)
                manager_invocations.append(manager_invocation)
                continue
            if not event_type.startswith(prefix):
                continue
            try:
                completed_kind = CapabilityKind(event_type.removeprefix(prefix))
            except ValueError as exc:
                raise ValueError("supervisor_resume_unknown_capability_event") from exc
            if completed_kind in completed_capabilities:
                raise ValueError("supervisor_resume_duplicate_capability_event")
            if pending_capability is not completed_kind:
                raise ValueError("supervisor_resume_manager_capability_sequence_mismatch")
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
            pending_capability = None
        run_blockers: list[str] = (
            ["supervisor_manager_output_invalid_or_failed"] if manager_refusals else []
        )
        for result in results:
            if result.to_mapping().get("status") == "blocked":
                run_blockers.append(f"capability_blocked:{result.to_mapping()['capability']}")
        reported_cost_usd = sum(
            float(row.to_mapping().get("cost_usd") or 0.0) for row in invocations
        ) + sum(float(row.get("cost_usd") or 0.0) for row in manager_invocations)
        recorded_reservations = [
            float(existing_inference_reservation_manifest["reserved_max_cost_usd"]),
            *[
                float(
                    (row.to_mapping().get("budget_state") or {}).get(
                        "cumulative_reserved_cost_usd", 0.0
                    )
                    or 0.0
                )
                for row in invocations
            ],
            *[
                float(
                    (row.get("budget_state") or {}).get("cumulative_reserved_cost_usd", 0.0) or 0.0
                )
                for row in manager_invocations
            ],
        ]
        reserved_cost_usd = max(recorded_reservations)
        live_invocation_count = sum(
            1
            for row in invocations
            if row.to_mapping().get("provider") == "openai"
            and row.to_mapping().get("validation_status") == "accepted_as_proposal"
        ) + sum(
            1
            for row in manager_invocations
            if row.get("provider") == "openai"
            and row.get("validation_status") == "accepted_as_control_decision"
        )
        can_invoke = selected_mode in {
            AutonomyMode.SHADOW,
            AutonomyMode.ADVISE,
            AutonomyMode.EXECUTE_NON_SPEND,
            AutonomyMode.EXECUTE_PREAUTHORIZED,
        }
        if selected_mode not in {
            AutonomyMode.DISABLED,
            AutonomyMode.SHADOW,
            AutonomyMode.ADVISE,
            AutonomyMode.EXECUTE_NON_SPEND,
            AutonomyMode.EXECUTE_PREAUTHORIZED,
        }:
            run_blockers.append(f"autonomy_mode_not_enabled_in_phase1:{selected_mode.value}")
        if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED and self.recovery_controller is None:
            can_invoke = False
            run_blockers.append("preauthorized_recovery_controller_missing")

        capabilities_dir = root / "capabilities"
        invocations_dir = root / "invocations"
        if can_invoke:
            capabilities_dir.mkdir(parents=True, exist_ok=True)
            invocations_dir.mkdir(parents=True, exist_ok=True)
            manager_decisions_dir = root / "manager" / "decisions"
            manager_invocations_dir = root / "manager" / "invocations"
            manager_refusals_dir = root / "manager" / "refusals"
            manager_decisions_dir.mkdir(parents=True, exist_ok=True)
            manager_invocations_dir.mkdir(parents=True, exist_ok=True)
            manager_refusals_dir.mkdir(parents=True, exist_ok=True)
            capability_by_kind = {capability.kind: capability for capability in self.capabilities}

            def managed_capability_order() -> Iterator[SupervisorCapability]:
                nonlocal last_event
                nonlocal live_invocation_count
                nonlocal manager_terminal_reason
                nonlocal pending_capability
                nonlocal reported_cost_usd
                nonlocal reserved_cost_usd

                while manager_terminal_reason is None:
                    if pending_capability is None:
                        manager_step = len(manager_decisions)
                        try:
                            manager_decision = self.manager.choose_next(
                                context=capability_context,
                                completed_results=results,
                                step_index=manager_step,
                            )
                        except Exception as exc:  # noqa: BLE001 - bounded manager refusal
                            refusal_value: dict[str, Any] = {
                                "schema_version": ("task_evaluation_supervisor_manager_refusal.v1"),
                                "run_id": context.run_id,
                                "step_index": manager_step,
                                "status": "refused",
                                "error_type": type(exc).__name__,
                                "raw_error_message_recorded": False,
                                "observed_capability_result_digests": sorted(
                                    result.digest for result in results
                                ),
                                "agent_harness": AGENTS_SDK_HARNESS_ID,
                                "proof_effect": "none",
                            }
                            refusal_value["supervisor_manager_refusal_digest"] = canonical_digest(
                                refusal_value,
                                digest_field="supervisor_manager_refusal_digest",
                            )
                            refusal_value = validate_manager_refusal(
                                refusal_value,
                                run_id=context.run_id,
                                completed_results=results,
                                step_index=manager_step,
                            )
                            write_json(
                                manager_refusals_dir / f"step-{manager_step:03d}.json",
                                refusal_value,
                            )
                            manager_refusals.append(refusal_value)
                            last_event = self._event(
                                ledger=ledger,
                                run_id=context.run_id,
                                phase=SupervisorPhase.PLANNING,
                                event_type=f"supervisor_manager_refused:{manager_step}",
                                generated_at=timestamp,
                                payload_digest=refusal_value["supervisor_manager_refusal_digest"],
                            )
                            run_blockers.append("supervisor_manager_output_invalid_or_failed")
                            manager_terminal_reason = "blocked"
                            break
                        decision_value = dict(manager_decision.value)
                        decision_value = validate_manager_decision(
                            decision_value,
                            context=capability_context,
                            completed_results=results,
                            step_index=manager_step,
                        )
                        manager_cost = (
                            float(manager_decision.invocation.cost_usd)
                            if isinstance(
                                manager_decision.invocation.cost_usd,
                                (int, float),
                            )
                            else 0.0
                        )
                        reported_cost_usd += manager_cost
                        manager_usage = dict(manager_decision.invocation.usage)
                        cumulative_reserved = manager_usage.get("cumulative_reserved_cost_usd")
                        if isinstance(cumulative_reserved, (int, float)):
                            reserved_cost_usd = max(
                                reserved_cost_usd,
                                float(cumulative_reserved),
                            )
                        if manager_decision.invocation.provider == "openai":
                            live_invocation_count += 1
                        remaining_unreserved_usd = max(
                            0.0,
                            self.agent_inference_budget_usd - reserved_cost_usd,
                        )
                        manager_invocation_value: dict[str, Any] = {
                            "schema_version": ("task_evaluation_supervisor_manager_invocation.v1"),
                            "invocation_id": (
                                f"{context.run_id}-manager-{manager_step}-invocation"
                            ),
                            "run_id": context.run_id,
                            "step_index": manager_step,
                            "provider": manager_decision.invocation.provider,
                            "model": manager_decision.invocation.model,
                            "agent_harness": AGENTS_SDK_HARNESS_ID,
                            "agents_sdk_version": manager_decision.invocation.sdk_version,
                            "adapter_id": self.manager.adapter_id,
                            "adapter_version": self.manager.adapter_version,
                            "instruction_digest": canonical_digest(
                                {"instruction": self.manager.instruction}
                            ),
                            "tool_registry_digest": self.tool_registry.digest,
                            "authority_digest": authority.digest,
                            "input_artifact_digests": [
                                *input_digests,
                                *decision_value["observed_capability_result_digests"],
                            ],
                            "budget_state": {
                                "max_cost_usd": self.agent_inference_budget_usd,
                                "reported_cost_usd": reported_cost_usd,
                                "cumulative_reserved_cost_usd": reserved_cost_usd,
                                "remaining_unreserved_usd": remaining_unreserved_usd,
                            },
                            "structured_output_digest": manager_decision.digest,
                            "validation_status": "accepted_as_control_decision",
                            "action_taken": "specialist_sequence_selected",
                            "refusal": False,
                            "usage": manager_usage,
                            "trace_id": manager_decision.invocation.trace_id,
                            "cost_usd": manager_cost,
                            "cost_status": manager_decision.invocation.cost_status,
                            "latency_seconds": (manager_decision.invocation.latency_seconds),
                            "proof_effect": "none",
                            "uncertainty": "not_a_proof_signal",
                            "parent_event_digest": last_event.digest,
                            "generated_at": timestamp,
                        }
                        manager_invocation_value["manager_invocation_digest"] = canonical_digest(
                            manager_invocation_value,
                            digest_field="manager_invocation_digest",
                        )
                        manager_invocation_value = validate_manager_invocation(
                            manager_invocation_value,
                            run_id=context.run_id,
                            step_index=manager_step,
                            structured_output_digest=manager_decision.digest,
                            tool_registry_digest=self.tool_registry.digest,
                            authority_digest=authority.digest,
                            input_artifact_digests=[
                                *input_digests,
                                *decision_value["observed_capability_result_digests"],
                            ],
                            manager_adapter_id=self.manager.adapter_id,
                            manager_adapter_version=self.manager.adapter_version,
                            max_cost_usd=self.agent_inference_budget_usd,
                            parent_event_digest=last_event.digest,
                        )
                        write_json(
                            manager_decisions_dir / f"step-{manager_step:03d}.json",
                            decision_value,
                        )
                        write_json(
                            manager_invocations_dir / f"step-{manager_step:03d}.json",
                            manager_invocation_value,
                        )
                        manager_decisions.append(decision_value)
                        manager_invocations.append(manager_invocation_value)
                        last_event = self._event(
                            ledger=ledger,
                            run_id=context.run_id,
                            phase=SupervisorPhase.PLANNING,
                            event_type=(f"supervisor_manager_decision_recorded:{manager_step}"),
                            generated_at=timestamp,
                            payload_digest=manager_decision.digest,
                        )
                        if decision_value["status"] == "terminal":
                            manager_terminal_reason = str(decision_value["terminal_reason"])
                            break
                        pending_capability = CapabilityKind(str(decision_value["next_capability"]))
                    capability = capability_by_kind.get(pending_capability)
                    if capability is None:
                        run_blockers.append("supervisor_manager_capability_unavailable")
                        manager_terminal_reason = "blocked"
                        break
                    yield capability
                    pending_capability = None

            for capability in managed_capability_order():
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
                    if result.to_mapping().get("status") == "blocked":
                        run_blockers.append(f"capability_blocked:{capability.kind.value}")
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

                metadata_getter = getattr(capability, "invocation_metadata", None)
                invocation_metadata = dict(metadata_getter()) if callable(metadata_getter) else {}
                raw_tool_observations = invocation_metadata.get("tool_observations") or []
                try:
                    if not isinstance(raw_tool_observations, list) or any(
                        not isinstance(row, Mapping) for row in raw_tool_observations
                    ):
                        raise SupervisorContractError(
                            ["tool_observations:must_be_list_of_mappings"]
                        )
                    tool_observations = [
                        validate_tool_observation_binding(
                            row,
                            run_id=context.run_id,
                            capability=capability.kind.value,
                            registry=self.tool_registry,
                            authority=authority.to_mapping(),
                        )
                        for row in raw_tool_observations
                    ]
                except (SupervisorContractError, ValueError):
                    tool_observations = []
                    result = CapabilityResult.from_mapping(
                        {
                            "schema_version": "task_evaluation_supervisor_capability_result.v1",
                            "result_id": f"{context.run_id}-{capability.kind.value}",
                            "run_id": context.run_id,
                            "capability": capability.kind.value,
                            "status": "blocked",
                            "artifact": {
                                "schema_version": "supervisor_tool_observation_refusal.v1",
                                "error_type": "SupervisorContractError",
                                "raw_tool_result_recorded": False,
                            },
                            "proposals": [],
                            "proposal_dispositions": [],
                            "blockers": ["tool_observation_contract_invalid"],
                            "evidence_refs": [],
                            "authoritative": False,
                            "proof_booleans_mutable": False,
                            "proof_effect": "none",
                        }
                    )
                    validation_status = "refused"
                    run_blockers.append(f"capability_blocked:{capability.kind.value}")
                tool_observation_integrity_status = invocation_metadata.get(
                    "tool_observation_integrity_status"
                )
                if tool_observation_integrity_status not in {
                    None,
                    "matched",
                    "invoker_failed_before_tool_execution",
                }:
                    result = CapabilityResult.from_mapping(
                        {
                            "schema_version": "task_evaluation_supervisor_capability_result.v1",
                            "result_id": f"{context.run_id}-{capability.kind.value}",
                            "run_id": context.run_id,
                            "capability": capability.kind.value,
                            "status": "blocked",
                            "artifact": {
                                "schema_version": (
                                    "supervisor_tool_observation_transport_refusal.v1"
                                ),
                                "integrity_status": tool_observation_integrity_status,
                                "raw_adapter_tool_result_recorded": False,
                                "trusted_tool_observations_preserved": bool(tool_observations),
                            },
                            "proposals": [],
                            "proposal_dispositions": [],
                            "blockers": ["tool_observation_transport_mismatch"],
                            "evidence_refs": [],
                            "authoritative": False,
                            "proof_booleans_mutable": False,
                            "proof_effect": "none",
                        }
                    )
                    validation_status = "refused"
                    run_blockers.append(f"capability_blocked:{capability.kind.value}")
                result_value = result.to_mapping()
                result_value["structured_observations"] = [
                    {
                        "tool_id": observation.get("tool_id"),
                        "status": observation.get("status"),
                        "typed_result": observation.get("typed_result"),
                        "typed_failure": observation.get("typed_failure"),
                        "produced_artifact_references": observation.get(
                            "produced_artifact_references"
                        )
                        or [],
                        "observation_digest": observation.get("observation_digest"),
                        "proof_effect": observation.get("proof_effect"),
                        "suggested_next_legal_actions": observation.get(
                            "suggested_next_legal_actions"
                        )
                        or [],
                    }
                    for observation in tool_observations
                ]
                result_value["observations_are_non_authoritative"] = True
                result_value.pop("capability_result_digest", None)
                result = CapabilityResult.from_mapping(result_value)
                result_path = capabilities_dir / f"{capability.kind.value}.json"
                write_json(result_path, result.to_mapping())
                input_digest = canonical_digest(
                    {
                        "customer_question": context.customer_question,
                        "input_artifact_digests": input_digests,
                        "capability": capability.kind.value,
                    }
                )
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
        inference_reservation_manifest = inference_reservations.write_manifest()
        reserved_cost_usd = max(
            reserved_cost_usd,
            float(inference_reservation_manifest["reserved_max_cost_usd"]),
        )
        terminal_status = (
            "disabled"
            if selected_mode is AutonomyMode.DISABLED
            else "blocked"
            if run_blockers
            else "non_spend_complete"
            if selected_mode is AutonomyMode.EXECUTE_NON_SPEND
            else "advise_complete"
            if selected_mode is AutonomyMode.ADVISE
            else "preauthorized_complete_with_failures"
            if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED and preauthorized_failures
            else "preauthorized_complete"
            if selected_mode is AutonomyMode.EXECUTE_PREAUTHORIZED
            else "shadow_complete"
        )
        terminal_payload_digest = canonical_digest(
            {
                "run_id": context.run_id,
                "status": terminal_status,
                "capability_result_digests": [result.digest for result in results],
                "supervisor_manager_decision_digests": [
                    str(row["supervisor_manager_decision_digest"]) for row in manager_decisions
                ],
                "supervisor_manager_terminal_reason": manager_terminal_reason,
                "supervisor_manager_refusal_digests": [
                    str(row["supervisor_manager_refusal_digest"]) for row in manager_refusals
                ],
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
                "manager_decisions": [
                    {
                        "step_index": row["step_index"],
                        "status": row["status"],
                        "next_capability": row.get("next_capability"),
                        "terminal_reason": row.get("terminal_reason"),
                        "digest": row["supervisor_manager_decision_digest"],
                        "artifact_path": (
                            f"manager/decisions/step-{int(row['step_index']):03d}.json"
                        ),
                    }
                    for row in manager_decisions
                ],
                "manager_invocations": [
                    {
                        "step_index": row["step_index"],
                        "digest": row["manager_invocation_digest"],
                        "artifact_path": (
                            f"manager/invocations/step-{int(row['step_index']):03d}.json"
                        ),
                    }
                    for row in manager_invocations
                ],
                "manager_terminal_reason": manager_terminal_reason,
                "manager_refusals": [
                    {
                        "step_index": row["step_index"],
                        "digest": row["supervisor_manager_refusal_digest"],
                        "artifact_path": (
                            f"manager/refusals/step-{int(row['step_index']):03d}.json"
                        ),
                    }
                    for row in manager_refusals
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
                "registered_non_spend_actions_executed": (registered_non_spend_actions_executed),
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
                        float(row.get("cost_usd") or 0.0) for row in executed_tool_observations
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
                    "manager_invocation_count": len(manager_invocations),
                    "reservation_count": inference_reservation_manifest["reservation_count"],
                    "in_flight_unknown_count": inference_reservation_manifest[
                        "in_flight_unknown_count"
                    ],
                    "reservation_manifest_digest": inference_reservation_manifest[
                        "inference_reservation_manifest_digest"
                    ],
                    "reservation_manifest_path": "inference_reservations/manifest.json",
                    "reported_cost_is_final": (
                        live_invocation_count == 0
                        and inference_reservation_manifest["in_flight_unknown_count"] == 0
                    ),
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
