"""OpenAI Agents SDK Task Evaluation Supervisor with a deterministic proof boundary."""

from .agents_sdk import (
    AGENTS_SDK_HARNESS_ID,
    DEFAULT_SUPERVISOR_AGENT_MODEL,
    AgentsSDKAgentSpec,
    AgentsSDKCapabilityOutput,
    AgentsSDKHarnessError,
    AgentsSDKInvocationBlocked,
    AgentsSDKInvocationResult,
    AgentsSDKInvoker,
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
    agents_sdk_capabilities,
)
from .capabilities import SupervisorContext, deterministic_baseline_capabilities
from .capture_ingress import CaptureBuildIngressError, load_capture_build_ingress
from .candidate_policy import (
    CandidatePolicyRuntime,
    CandidatePolicyError,
    FrozenAgenticPolicyAdapter,
    IndependentCandidateEvaluator,
    compile_neutral_candidate_policy_suite,
    execute_neutral_candidate_policy_suite,
    freeze_candidate_policy_manifest,
)
from .contracts import (
    ActionProposal,
    AgentInvocationManifest,
    AuthorityEnvelope,
    AutonomyMode,
    CapabilityKind,
    CapabilityResult,
    SupervisorContractError,
    SupervisorEvent,
    SupervisorRun,
    SupervisorState,
    TerminalSupervisorReport,
    ToolDescriptor,
    proof_boundary,
)
from .ledger import AppendOnlyEventLedger, SupervisorLedgerError
from .inference_reservations import (
    InferenceReservationAudit,
    InferenceReservationError,
)
from .lifecycle import (
    CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION,
    run_capture_build_supervisor,
)
from .manager import (
    AgentsSDKSupervisorManagerOutput,
    OpenAIAgentsSDKSupervisorManager,
    SupervisorManagerDecision,
    SupervisorManagerError,
)
from .phase2_artifacts import (
    Phase2ArtifactError,
    authorization_receipt,
    authorization_request,
    clarification_receipt,
    clarification_request,
    deterministic_customer_report,
    freeze_scenario_manifest,
    scenario_proposal_set,
)
from .pigey_candidate_runtime import PigeyScenarioBinding, PigeySimCandidateRuntime
from .replay import SupervisorReplayError, replay_supervisor_run
from .recovery import (
    PreauthorizedRecoveryController,
    PreauthorizedRecoveryPolicy,
    RecoveryControlError,
)
from .evaluation import (
    SupervisorEvaluationCase,
    SupervisorEvaluationError,
    compare_supervisor_to_baseline,
    evaluate_supervisor_execution,
    load_supervisor_evaluation_corpus,
)
from .supervisor import SupervisorExecution, TaskEvaluationSupervisor
from .tools import RegisteredToolBinding, ToolRegistry
from .vast_recovery_adapter import VastRecoveryAdapterError, VastWAMRecoveryAdapter

__all__ = [
    "AGENTS_SDK_HARNESS_ID",
    "DEFAULT_SUPERVISOR_AGENT_MODEL",
    "ActionProposal",
    "AgentsSDKAgentSpec",
    "AgentsSDKCapabilityOutput",
    "AgentsSDKHarnessError",
    "AgentsSDKInvocationBlocked",
    "AgentsSDKInvocationResult",
    "AgentsSDKInvoker",
    "AgentInvocationManifest",
    "AppendOnlyEventLedger",
    "AuthorityEnvelope",
    "AutonomyMode",
    "CapabilityKind",
    "CapabilityResult",
    "CandidatePolicyError",
    "CandidatePolicyRuntime",
    "CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION",
    "CaptureBuildIngressError",
    "FrozenAgenticPolicyAdapter",
    "IndependentCandidateEvaluator",
    "InferenceReservationAudit",
    "InferenceReservationError",
    "OpenAIAgentsSDKConfig",
    "OpenAIAgentsSDKInvoker",
    "OpenAIAgentsSDKSupervisorManager",
    "Phase2ArtifactError",
    "PigeyScenarioBinding",
    "PigeySimCandidateRuntime",
    "PreauthorizedRecoveryController",
    "PreauthorizedRecoveryPolicy",
    "RegisteredToolBinding",
    "RecoveryControlError",
    "SupervisorContext",
    "SupervisorContractError",
    "SupervisorEvent",
    "SupervisorExecution",
    "SupervisorEvaluationCase",
    "SupervisorEvaluationError",
    "SupervisorLedgerError",
    "SupervisorManagerDecision",
    "SupervisorManagerError",
    "AgentsSDKSupervisorManagerOutput",
    "SupervisorReplayError",
    "SupervisorRun",
    "SupervisorState",
    "TaskEvaluationSupervisor",
    "TerminalSupervisorReport",
    "ToolDescriptor",
    "ToolRegistry",
    "VastRecoveryAdapterError",
    "VastWAMRecoveryAdapter",
    "agents_sdk_capabilities",
    "deterministic_baseline_capabilities",
    "compare_supervisor_to_baseline",
    "compile_neutral_candidate_policy_suite",
    "execute_neutral_candidate_policy_suite",
    "evaluate_supervisor_execution",
    "load_capture_build_ingress",
    "load_supervisor_evaluation_corpus",
    "proof_boundary",
    "replay_supervisor_run",
    "run_capture_build_supervisor",
    "authorization_receipt",
    "authorization_request",
    "clarification_receipt",
    "clarification_request",
    "deterministic_customer_report",
    "freeze_scenario_manifest",
    "freeze_candidate_policy_manifest",
    "scenario_proposal_set",
]
