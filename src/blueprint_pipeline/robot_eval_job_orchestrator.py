"""Headless robot-eval job orchestration lane.

This module coordinates a repo-local robot-team evaluation or training request
through deterministic validation, provisioning, simulator, training, and
evaluation manifests. Provider, GPU, simulator, training, and agent execution
paths fail closed unless explicit environment and CLI gates are present.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .robot_eval_dataset import build_real_site_robot_eval_dataset
from .site_eval_director import build_site_eval_director


JOB_REQUEST_SCHEMA_VERSION = "robot_eval_job_request.v1"
JOB_VALIDATION_SCHEMA_VERSION = "robot_eval_job_validation.v1"
JOB_PLAN_SCHEMA_VERSION = "robot_eval_job_plan.v1"
AGENT_ORCHESTRATION_PLAN_SCHEMA_VERSION = "robot_eval_agent_orchestration_plan.v1"
GPU_PROVISIONING_REQUEST_SCHEMA_VERSION = "robot_eval_gpu_provisioning_request.v1"
GPU_PROVISIONING_RESULT_SCHEMA_VERSION = "robot_eval_gpu_provisioning_result.v1"
SIMULATOR_SERVICE_REQUEST_SCHEMA_VERSION = "robot_eval_simulator_service_request.v1"
SIMULATOR_SERVICE_RESULT_SCHEMA_VERSION = "robot_eval_simulator_service_result.v1"
POLICY_PACKAGE_MANIFEST_SCHEMA_VERSION = "robot_eval_policy_package_manifest.v1"
TRAINING_REQUEST_SCHEMA_VERSION = "robot_eval_training_request.v1"
TRAINING_RESULT_SCHEMA_VERSION = "robot_eval_training_result.v1"
EVALUATION_REQUEST_SCHEMA_VERSION = "robot_eval_evaluation_request.v1"
EVALUATION_RESULT_SCHEMA_VERSION = "robot_eval_evaluation_result.v1"
NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION = "robot_eval_job_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "robot_eval_job_failure_labels.v1"
PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION = "robot_eval_job_prediction_outcome_ledger.v1"
CALIBRATION_REPORT_SCHEMA_VERSION = "robot_eval_job_calibration_report.v1"
BREAKAGE_LIBRARY_SCHEMA_VERSION = "robot_eval_job_breakage_library.v1"
PROOF_BOUNDARY_SCHEMA_VERSION = "robot_eval_job_proof_boundary.v1"
JOB_RUN_MANIFEST_SCHEMA_VERSION = "robot_eval_job_run_manifest.v1"
BLOCKED_MANIFEST_SCHEMA_VERSION = "robot_eval_job_blocked_manifest.v1"

PROVISIONERS = (
    "fixture_local",
    "local_process",
    "docker_local",
    "vast",
    "runpod",
    "gcp",
)
SIMULATORS = ("fixture", "mujoco", "pybullet", "newton", "isaac_sim")
OPERATIONS = ("evaluate_only", "train_only", "train_then_evaluate")

REQUIRED_ROBOT_EVAL_INPUTS = {
    "robot_eval_site_card": "robot_eval_dataset/site_card.json",
    "robot_eval_task_cards": "robot_eval_dataset/task_cards.json",
    "robot_eval_scenario_cards": "robot_eval_dataset/scenario_cards.json",
    "robot_eval_cards": "robot_eval_dataset/eval_cards.json",
    "robot_eval_proof_boundaries": "robot_eval_dataset/proof_boundaries.json",
}

POLICY_MODALITY_ORDER = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)

POLICY_MODALITY_STATUSES = {
    "policy_api_endpoint": "needs_policy_api_endpoint_ref",
    "docker_container": "needs_docker_container_ref",
    "recorded_action_trace": "needs_recorded_action_trace_ref",
    "high_level_skill_trace": "needs_high_level_skill_trace_ref",
    "teleop_demo": "needs_teleop_demo_ref",
    "sim_controller_plugin": "needs_sim_controller_plugin_ref",
}

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "robot_eval_job_orchestration_only",
    "repo_local_only_by_default": True,
    "agents_advisory_only": True,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "gpu_provisioning_performed": False,
    "simulators_run": False,
    "gpu_training_run": False,
    "messages_sent": False,
    "payments_touched": False,
    "deployments_performed": False,
    "simulator_execution_proven": False,
    "robot_readiness_proven": False,
    "robot_policy_execution_proven": False,
    "physics_contact_validated": False,
    "safety_validated": False,
    "training_completed": False,
    "public_claim_upgrade_allowed": False,
    "disallowed_claims": [
        "robot_ready",
        "deployment_ready",
        "simulator_execution_completed",
        "physics_contact_validated",
        "robot_policy_execution_passed",
        "training_completed",
        "safety_validated",
        "public_deployment_ready",
    ],
    "proof_upgrade_requires": [
        "rights/privacy clearance for the exact use",
        "owner-system simulator load and action traces",
        "owner-system robot policy, teleoperation, or action logs",
        "training run manifest and checkpoint evidence",
        "physics/contact validation logs",
        "safety methodology and validation evidence",
        "actual outcome records",
    ],
}


class RobotEvalJobAgentAdapter(Protocol):
    def build_plan(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class FakeRobotEvalJobAgentAdapter:
    """Network-free advisory agent adapter for local state-machine tests."""

    adapter_name: str = "fake"

    def build_plan(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "schema_version": AGENT_ORCHESTRATION_PLAN_SCHEMA_VERSION,
            "adapter": self.adapter_name,
            "status": "completed",
            "agent_authority": "advisory_only",
            "network_required": False,
            "live_provider_calls_performed": False,
            "decisions": [
                {
                    "next_command": "validate_job_request",
                    "owned_by": "deterministic_orchestrator",
                    "reason": "Validate rights, policy references, and proof requirements first.",
                },
                {
                    "next_command": "run_allowed_provisioner",
                    "owned_by": "deterministic_orchestrator",
                    "reason": "Use the selected provisioner only after deterministic gates pass.",
                },
                {
                    "next_command": "run_allowed_simulator_or_fixture",
                    "owned_by": "deterministic_orchestrator",
                    "reason": "Collect result manifests without upgrading proof claims.",
                },
            ],
            "diagnostics": [
                {
                    "status": "review_only",
                    "summary": (
                        "Agents can inspect manifests, suggest retries, and summarize blockers; "
                        "deterministic manifests own proof state."
                    ),
                }
            ],
            "plan_context_fingerprint": _sha_payload(plan_context),
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }


@dataclass(frozen=True)
class AgentsSdkRobotEvalJobAdapter:
    """Optional OpenAI Agents SDK request-manifest adapter.

    The adapter deliberately fails closed unless the SDK, API key, and explicit
    environment gate are present. It does not execute a live agent here.
    """

    agents_sdk_available: bool | None = None
    openai_api_key: str | None = None
    env_gate_allowed: bool | None = None
    model: str = "gpt-4.1"

    def build_plan(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        agents_available = (
            self.agents_sdk_available
            if self.agents_sdk_available is not None
            else _module_available(("agents", "openai_agents"))
        )
        api_key_present = bool(
            _string(self.openai_api_key)
            if self.openai_api_key is not None
            else _string(os.getenv("OPENAI_API_KEY"))
        )
        env_allowed = (
            bool(self.env_gate_allowed)
            if self.env_gate_allowed is not None
            else _env_truthy("BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION")
        )
        blockers: List[str] = []
        if not agents_available:
            blockers.append("missing_openai_agents_sdk")
        if not api_key_present:
            blockers.append("missing_openai_api_key")
        if not env_allowed:
            blockers.append("missing_env_BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION")
        return {
            "schema_version": AGENT_ORCHESTRATION_PLAN_SCHEMA_VERSION,
            "adapter": "openai_agents_sdk_robot_eval_job",
            "status": "blocked" if blockers else "request_manifest_ready",
            "blockers": blockers,
            "missing_inputs": list(blockers),
            "agent_authority": "advisory_only",
            "execution_performed": False,
            "network_required_if_executed": True,
            "request": {
                "purpose": "headless_robot_eval_job_orchestration_advisory_only",
                "model": self.model,
                "job_id": _string(plan_context.get("job_id")),
                "capture_root": _string(plan_context.get("capture_root")),
                "allowed_actions": [
                    "choose_next_deterministic_command",
                    "inspect_manifests_and_logs",
                    "retry_safe_failures",
                    "request_gpu_or_simulator_provisioning",
                    "summarize_blockers",
                    "route_for_human_review",
                ],
                "prohibited_actions": [
                    "override_rights_privacy_blockers",
                    "mark_robot_readiness_proven",
                    "mark_simulator_proof_complete_without_result_manifest",
                    "mark_training_proof_complete_without_checkpoint_manifest",
                    "spend_money_without_explicit_gates",
                    "call_live_providers_without_explicit_gates",
                ],
            },
            "evidence": {
                "openai_agents_sdk_available": bool(agents_available),
                "openai_api_key_present": api_key_present,
                "env_BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION": env_allowed,
            },
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }


def _module_available(candidates: Sequence[str]) -> bool:
    return any(importlib.util.find_spec(candidate) is not None for candidate in candidates)


def _env_truthy(name: str) -> bool:
    return _string(os.getenv(name)).lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "on", "allowed", "cleared"}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _relative_if_file(base_dir: Path, target: Path) -> str | None:
    return _relative_to(base_dir, target) if target.is_file() else None


def _sha_payload(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_job_request(job_request: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(job_request, Mapping):
        return dict(job_request)
    payload = read_json_any(Path(job_request))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected job request JSON object at {job_request}")
    return dict(payload)


def _write_job_json(job_dir: Path, name: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    write_json(job_dir / name, payload)
    return dict(payload)


def _field(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return None


def _modality_payload(policy_package: Mapping[str, Any], modality: str) -> Dict[str, Any]:
    camel = {
        "policy_api_endpoint": "policyApiEndpoint",
        "docker_container": "dockerContainer",
        "recorded_action_trace": "recordedActionTrace",
        "high_level_skill_trace": "highLevelSkillTrace",
        "teleop_demo": "teleopDemo",
        "sim_controller_plugin": "simControllerPlugin",
    }[modality]
    return _mapping(policy_package.get(modality) or policy_package.get(camel))


def _validate_policy_modality(
    *,
    modality: str,
    payload: Mapping[str, Any],
) -> tuple[str, List[str]]:
    missing: List[str] = []
    status = "reference_present_requires_owner_system_review"
    if not payload:
        return "blocked", [f"policy_package.{modality}"]
    if modality == "policy_api_endpoint":
        endpoint = _string(_field(payload, "endpoint_url", "endpointUrl", "url"))
        if not (endpoint.startswith("https://") or endpoint.startswith("http://")):
            missing.append("policy_package.policy_api_endpoint.endpoint_url")
    elif modality == "docker_container":
        if not _string(_field(payload, "image_ref", "imageRef")):
            missing.append("policy_package.docker_container.image_ref")
        digest = _string(_field(payload, "digest", "digestChecksum"))
        if not digest.startswith("sha256:"):
            missing.append("policy_package.docker_container.digest")
    elif modality == "recorded_action_trace":
        if not _string(_field(payload, "trace_manifest_uri", "traceManifestUri")):
            missing.append("policy_package.recorded_action_trace.trace_manifest_uri")
        if not _string(_field(payload, "timestamp_alignment", "timestampAlignment")):
            missing.append("policy_package.recorded_action_trace.timestamp_alignment")
    elif modality == "high_level_skill_trace":
        sequence = payload.get("ordered_skill_sequence") or payload.get("orderedSkillSequence")
        if not (isinstance(sequence, list) and sequence):
            missing.append("policy_package.high_level_skill_trace.ordered_skill_sequence")
    elif modality == "teleop_demo":
        if not _string(_field(payload, "demo_artifact_uri", "demoArtifactUri")):
            missing.append("policy_package.teleop_demo.demo_artifact_uri")
        if not _string(
            _field(payload, "rights_privacy_attestation", "rightsPrivacyAttestation")
        ):
            missing.append("policy_package.teleop_demo.rights_privacy_attestation")
    elif modality == "sim_controller_plugin":
        if not _string(_field(payload, "simulator_framework", "simulatorFramework")):
            missing.append("policy_package.sim_controller_plugin.simulator_framework")
        if not _string(_field(payload, "plugin_uri", "pluginUri")):
            missing.append("policy_package.sim_controller_plugin.plugin_uri")
    if missing:
        status = "blocked"
    return status, missing


def _policy_package_manifest(
    *,
    request: Mapping[str, Any],
    generated_at: str,
) -> tuple[Dict[str, Any], List[str], List[str]]:
    policy_package = _mapping(request.get("policy_package") or request.get("policyPackage"))
    modalities: Dict[str, Dict[str, Any]] = {}
    missing_inputs: List[str] = []
    missing_statuses: List[str] = []
    for modality in POLICY_MODALITY_ORDER:
        payload = _modality_payload(policy_package, modality)
        status, missing = _validate_policy_modality(modality=modality, payload=payload)
        if missing:
            missing_inputs.extend(missing)
            missing_statuses.append(POLICY_MODALITY_STATUSES[modality])
        modalities[modality] = {
            "status": status,
            "missing_inputs": missing,
            "missing_evidence_status": POLICY_MODALITY_STATUSES[modality],
            "reference": dict(payload),
            "download_performed": False,
            "owner_system_review_required": status != "blocked",
            "claim_boundary": (
                "reference_present_only_not_policy_execution_or_robot_readiness_proof"
            ),
        }
    manifest = {
        "schema_version": POLICY_PACKAGE_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked" if missing_inputs else "review_required",
        "modalities": modalities,
        "missing_inputs": missing_inputs,
        "missing_evidence_statuses": missing_statuses,
        "downloads_performed": False,
        "policy_execution_proven": False,
        "robot_readiness_proven": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return manifest, missing_inputs, missing_statuses


def _request_rights_blocked(request: Mapping[str, Any]) -> bool:
    scope = _mapping(request.get("rights_privacy_scope") or request.get("rightsPrivacyScope"))
    status = _string(scope.get("status")).lower()
    blocked_statuses = {
        "blocked",
        "denied",
        "failed",
        "unsafe",
        "not_allowed",
        "missing",
    }
    explicit_allowed = (
        scope.get("external_use_allowed")
        if "external_use_allowed" in scope
        else scope.get("externalUseAllowed")
    )
    if status in blocked_statuses:
        return True
    if explicit_allowed is not None and not _boolish(explicit_allowed):
        return True
    return False


def _site_card_rights_blocked(pipeline_dir: Path) -> bool:
    site_card = _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "site_card.json")
    rights = _mapping(_mapping(site_card.get("provenance_rights_review_status")).get("rights_privacy"))
    return bool(rights.get("blocked"))


def _missing_robot_eval_inputs(pipeline_dir: Path) -> List[str]:
    return [
        key
        for key, relative_path in REQUIRED_ROBOT_EVAL_INPUTS.items()
        if not (pipeline_dir / relative_path).is_file()
    ]


def _ensure_robot_eval_cards(*, capture_root: Path, pipeline_dir: Path) -> List[str]:
    missing = _missing_robot_eval_inputs(pipeline_dir)
    if missing:
        build_real_site_robot_eval_dataset(capture_root=capture_root)
    return _missing_robot_eval_inputs(pipeline_dir)


def _job_validation(
    *,
    request: Mapping[str, Any],
    policy_missing_inputs: Sequence[str],
    policy_missing_statuses: Sequence[str],
    missing_robot_eval_inputs: Sequence[str],
    generated_at: str,
    pipeline_dir: Path,
) -> Dict[str, Any]:
    missing_inputs: List[str] = []
    blockers: List[str] = []
    missing_evidence_statuses = list(policy_missing_statuses)
    operation = _string(request.get("operation") or "evaluate_only")
    if operation not in OPERATIONS:
        blockers.append("invalid_requested_operation")
        missing_inputs.append("operation")
    if not _mapping(request.get("customer")):
        blockers.append("missing_customer")
        missing_inputs.append("customer")
    if not _mapping(request.get("robot_profile") or request.get("robotProfile")):
        blockers.append("missing_robot_profile")
        missing_inputs.append("robot_profile")
    if not _string_list(request.get("requested_tasks") or request.get("requestedTasks")):
        blockers.append("missing_requested_tasks")
        missing_inputs.append("requested_tasks")
    if policy_missing_inputs:
        blockers.append("missing_policy_evidence")
        missing_inputs.extend(policy_missing_inputs)
    if missing_robot_eval_inputs:
        blockers.append("missing_robot_eval_dataset_cards")
        missing_inputs.extend(missing_robot_eval_inputs)
    if _request_rights_blocked(request) or _site_card_rights_blocked(pipeline_dir):
        blockers.append("blocked_rights_privacy")
        missing_inputs = ["rights_privacy_clearance"]
    return {
        "schema_version": JOB_VALIDATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked" if blockers else "passed",
        "blockers": _dedupe(blockers),
        "missing_inputs": _dedupe(missing_inputs),
        "missing_evidence_statuses": _dedupe(missing_evidence_statuses),
        "rights_privacy_blocked": "blocked_rights_privacy" in blockers,
        "policy_evidence_complete": not policy_missing_inputs,
        "robot_eval_dataset_cards_present": not missing_robot_eval_inputs,
        "evidence_requirements": {
            "policy_package_modalities": list(POLICY_MODALITY_ORDER),
            "robot_eval_dataset_inputs": dict(REQUIRED_ROBOT_EVAL_INPUTS),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _agent_plan(
    *,
    adapter: RobotEvalJobAgentAdapter | None,
    plan_context: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    if adapter is None:
        return {
            "schema_version": AGENT_ORCHESTRATION_PLAN_SCHEMA_VERSION,
            "generated_at": generated_at,
            "adapter": "none",
            "status": "not_requested",
            "agent_authority": "advisory_only",
            "decisions": [],
            "diagnostics": [],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    plan = adapter.build_plan(plan_context=plan_context)
    plan.setdefault("generated_at", generated_at)
    plan.setdefault("claim_boundary", dict(CLAIM_BOUNDARY))
    return dict(plan)


def _gpu_provisioning_request(
    *,
    request: Mapping[str, Any],
    job_id: str,
    provisioner: str,
    budget_usd: float | None,
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    budget = _mapping(request.get("budget"))
    return {
        "schema_version": GPU_PROVISIONING_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "provider": provisioner,
        "status": "planned",
        "requested_budget_usd": budget_usd
        if budget_usd is not None
        else _number(budget.get("budget_usd") or budget.get("budgetUsd")),
        "timeout_seconds": timeout_seconds,
        "execution_allowed_by_default": False,
        "live_provider_calls_allowed_by_default": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _gpu_provisioning_result(
    *,
    request_manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    allow_gpu_provisioning: bool,
    generated_at: str,
) -> Dict[str, Any]:
    provider = _string(request_manifest.get("provider")) or "fixture_local"
    if validation.get("status") == "blocked":
        return {
            "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "provider": provider,
            "status": "blocked",
            "reason": "job_validation_blocked",
            "blockers": _string_list(validation.get("blockers")),
            "execution_performed": False,
            "live_provider_calls_performed": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    if provider == "fixture_local":
        return {
            "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "provider": "fixture_local",
            "status": "allocated",
            "allocation_id": "fixture-gpu-local-0",
            "gpu_class": "fixture",
            "execution_performed": True,
            "live_provider_calls_performed": False,
            "cost_usd": 0.0,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_GPU_PROVISIONING")
    blockers: List[str] = []
    if not env_allowed:
        blockers.append("missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING")
    if not allow_gpu_provisioning:
        blockers.append("missing_cli_allow_gpu_provisioning")
    return {
        "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "provider": provider,
        "status": "blocked" if blockers else "request_manifest_ready",
        "reason": "approval_required" if blockers else "explicitly_gated_request_ready",
        "blockers": blockers,
        "execution_performed": False,
        "live_provider_calls_performed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _simulator_request(
    *,
    job_id: str,
    simulator: str,
    request: Mapping[str, Any],
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SIMULATOR_SERVICE_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "framework": simulator,
        "status": "planned",
        "requested_tasks": request.get("requested_tasks") or request.get("requestedTasks") or [],
        "robot_profile": _mapping(request.get("robot_profile") or request.get("robotProfile")),
        "timeout_seconds": timeout_seconds,
        "execution_allowed_by_default": False,
        "fixture_backend_proves_local_loop_only": simulator == "fixture",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _blocked_simulator_result(
    *,
    simulator: str,
    blockers: Sequence[str],
    generated_at: str,
    reason: str = "approval_required",
    execution_performed: bool = False,
) -> Dict[str, Any]:
    return {
        "schema_version": SIMULATOR_SERVICE_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "framework": simulator,
        "status": "blocked",
        "reason": reason,
        "blockers": list(blockers),
        "execution_performed": execution_performed,
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "artifact_paths": {},
        "simulators_run": False,
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _run_command_simulator(
    *,
    simulator: str,
    command_text: str,
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    command = shlex.split(command_text)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return _blocked_simulator_result(
            simulator=simulator,
            blockers=["missing_simulator_command_dependency"],
            generated_at=generated_at,
            reason="missing_dependency",
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": SIMULATOR_SERVICE_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "framework": simulator,
            "status": "failed",
            "reason": "timeout",
            "blockers": ["simulator_command_timeout"],
            "command": command,
            "raw_command": command_text,
            "execution_performed": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "exit_code": None,
            "artifact_paths": {},
            "simulators_run": True,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    status = "completed" if completed.returncode == 0 else "failed"
    return {
        "schema_version": SIMULATOR_SERVICE_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "framework": simulator,
        "status": status,
        "reason": None if status == "completed" else f"simulator_exit_code:{completed.returncode}",
        "blockers": [] if status == "completed" else [f"simulator_exit_code:{completed.returncode}"],
        "command": command,
        "raw_command": command_text,
        "execution_performed": True,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
        "artifact_paths": {},
        "simulators_run": True,
        "simulator_execution_proven": status == "completed",
        "robot_policy_execution_proven": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulator_execution_proven": status == "completed",
        },
    }


def _copy_site_eval_artifacts(*, pipeline_dir: Path, job_dir: Path, generated_at: str) -> Dict[str, Dict[str, Any]]:
    automation_dir = pipeline_dir / "simulation_automation"
    sources = {
        "normalized_attempt_trace": automation_dir / "normalized_simulator_attempt_trace.json",
        "failure_labels": automation_dir / "failure_labels.json",
        "prediction_outcome_ledger": automation_dir / "site_eval_prediction_outcome_ledger.json",
        "calibration_report": automation_dir / "site_eval_calibration_report.json",
        "breakage_library": automation_dir / "learned_facility_breakage_library.json",
    }
    placeholders = {
        "normalized_attempt_trace": {
            "schema_version": NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "attempt_count": 0,
            "attempts": [],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
        "failure_labels": {
            "schema_version": FAILURE_LABELS_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_available",
            "label_count": 0,
            "labels": [],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
        "prediction_outcome_ledger": {
            "schema_version": PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_available",
            "record_count": 0,
            "records": [],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
        "calibration_report": {
            "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_available",
            "record_count": 0,
            "records": [],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
        "breakage_library": {
            "schema_version": BREAKAGE_LIBRARY_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_available",
            "record_count": 0,
            "records": [],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
    }
    copied: Dict[str, Dict[str, Any]] = {}
    for key, source in sources.items():
        payload = _read_optional_mapping(source) or dict(placeholders[key])
        if key == "normalized_attempt_trace":
            payload.setdefault("schema_version", NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION)
        write_json(job_dir / f"{key}.json", payload)
        copied[key] = dict(payload)
    return copied


def _run_fixture_simulator(
    *,
    capture_root: Path,
    pipeline_dir: Path,
    job_dir: Path,
    generated_at: str,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[str]]:
    result = build_site_eval_director(capture_root=capture_root)
    copied = _copy_site_eval_artifacts(
        pipeline_dir=pipeline_dir,
        job_dir=job_dir,
        generated_at=generated_at,
    )
    trace = copied["normalized_attempt_trace"]
    blockers = _string_list(trace.get("blockers"))
    if not blockers:
        blocked_manifest = _read_optional_mapping(
            pipeline_dir / "simulation_automation" / "site_eval_fixture_runner_blocked_manifest.json"
        )
        blockers = _string_list(blocked_manifest.get("blockers"))
    if _string(trace.get("status")) == "completed":
        simulator_result = {
            "schema_version": SIMULATOR_SERVICE_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "framework": "fixture",
            "status": "completed",
            "reason": None,
            "blockers": [],
            "execution_performed": True,
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "artifact_paths": {
                "site_eval_director_run_manifest": _relative_to(
                    job_dir,
                    pipeline_dir / "simulation_automation" / "site_eval_director_run_manifest.json",
                ),
                "normalized_attempt_trace": "normalized_attempt_trace.json",
            },
            "site_eval_director_status": result.get("status"),
            "simulators_run": False,
            "fixture_runner_executed": True,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        return simulator_result, copied, []
    simulator_result = _blocked_simulator_result(
        simulator="fixture",
        blockers=blockers or [_string(result.get("status")) or "fixture_runner_blocked"],
        generated_at=generated_at,
        reason="fixture_runner_blocked",
    )
    simulator_result["site_eval_director_status"] = result.get("status")
    return simulator_result, copied, _string_list(simulator_result.get("blockers"))


def _run_simulator(
    *,
    simulator: str,
    validation: Mapping[str, Any],
    provisioning_result: Mapping[str, Any],
    capture_root: Path,
    pipeline_dir: Path,
    job_dir: Path,
    allow_simulator_execution: bool,
    allowed_simulators: Sequence[str],
    simulator_commands: Mapping[str, str],
    timeout_seconds: int,
    generated_at: str,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[str]]:
    if validation.get("status") == "blocked":
        result = _blocked_simulator_result(
            simulator=simulator,
            blockers=_string_list(validation.get("blockers")),
            generated_at=generated_at,
            reason="job_validation_blocked",
        )
        copied = _copy_site_eval_artifacts(
            pipeline_dir=pipeline_dir,
            job_dir=job_dir,
            generated_at=generated_at,
        )
        return result, copied, []
    if provisioning_result.get("status") == "blocked":
        result = _blocked_simulator_result(
            simulator=simulator,
            blockers=["gpu_provisioning_blocked"],
            generated_at=generated_at,
            reason="gpu_provisioning_blocked",
        )
        copied = _copy_site_eval_artifacts(
            pipeline_dir=pipeline_dir,
            job_dir=job_dir,
            generated_at=generated_at,
        )
        return result, copied, ["gpu_provisioning_blocked"]
    if simulator == "fixture":
        return _run_fixture_simulator(
            capture_root=capture_root,
            pipeline_dir=pipeline_dir,
            job_dir=job_dir,
            generated_at=generated_at,
        )
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
    allowed = set(allowed_simulators)
    command_text = _string(simulator_commands.get(simulator))
    blockers: List[str] = []
    if not env_allowed:
        blockers.append("missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
    if not allow_simulator_execution:
        blockers.append("missing_cli_allow_simulator_execution")
    if simulator not in allowed:
        blockers.append(f"missing_cli_allow_simulator_{simulator}")
    if not command_text:
        blockers.append(f"missing_simulator_command_{simulator}")
    if blockers:
        result = _blocked_simulator_result(
            simulator=simulator,
            blockers=blockers,
            generated_at=generated_at,
        )
    else:
        result = _run_command_simulator(
            simulator=simulator,
            command_text=command_text,
            timeout_seconds=timeout_seconds,
            generated_at=generated_at,
        )
    copied = _copy_site_eval_artifacts(
        pipeline_dir=pipeline_dir,
        job_dir=job_dir,
        generated_at=generated_at,
    )
    return result, copied, ["simulator_service_blocked"] if result.get("status") == "blocked" else []


def _training_request(
    *,
    request: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    operation = _string(request.get("operation") or "evaluate_only")
    preference = _mapping(
        request.get("cosmos_training_preference")
        or request.get("cosmosTrainingPreference")
    )
    if operation == "evaluate_only":
        status = "not_requested"
    else:
        status = "export_manifest_only"
    return {
        "schema_version": TRAINING_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "operation": operation,
        "preference": preference,
        "export_only_by_default": True,
        "execution_allowed_by_default": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _training_result(
    *,
    request_manifest: Mapping[str, Any],
    allow_training: bool,
    training_command: str | None,
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    if request_manifest.get("status") == "not_requested":
        return {
            "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_requested",
            "execution_performed": False,
            "gpu_training_run": False,
            "training_completed": False,
            "checkpoint_path": None,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_COSMOS_TRAINING")
    blockers: List[str] = []
    if not env_allowed:
        blockers.append("missing_env_BLUEPRINT_ALLOW_COSMOS_TRAINING")
    if not allow_training:
        blockers.append("missing_cli_allow_training")
    if not _string(training_command):
        blockers.append("missing_training_command")
    if blockers:
        return {
            "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "reason": "approval_required",
            "blockers": blockers,
            "execution_performed": False,
            "gpu_training_run": False,
            "training_completed": False,
            "checkpoint_path": None,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    command = shlex.split(_string(training_command))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    checkpoint_path = _string(
        _mapping(request_manifest.get("preference")).get("checkpoint_path")
        or os.getenv("BLUEPRINT_COSMOS_CHECKPOINT_PATH")
    )
    status = "completed" if completed.returncode == 0 and checkpoint_path else "blocked"
    blockers = [] if status == "completed" else ["missing_training_checkpoint_manifest"]
    return {
        "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "reason": None if status == "completed" else "missing_checkpoint_manifest",
        "blockers": blockers,
        "command": command,
        "execution_performed": True,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
        "gpu_training_run": True,
        "training_completed": status == "completed",
        "checkpoint_path": checkpoint_path or None,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "gpu_training_run": True,
            "training_completed": status == "completed",
        },
    }


def _evaluation_request(
    *,
    request: Mapping[str, Any],
    simulator_result: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    operation = _string(request.get("operation") or "evaluate_only")
    status = "not_requested" if operation == "train_only" else "planned"
    return {
        "schema_version": EVALUATION_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "operation": operation,
        "simulator_result_status": simulator_result.get("status"),
        "requested_tasks": request.get("requested_tasks") or request.get("requestedTasks") or [],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _evaluation_result(
    *,
    evaluation_request: Mapping[str, Any],
    simulator_result: Mapping[str, Any],
    copied_artifacts: Mapping[str, Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    if evaluation_request.get("status") == "not_requested":
        status = "not_requested"
        blockers: List[str] = []
    elif simulator_result.get("status") == "blocked":
        status = "blocked"
        blockers = _string_list(simulator_result.get("blockers"))
    else:
        trace = _mapping(copied_artifacts.get("normalized_attempt_trace"))
        attempts = [
            item for item in trace.get("attempts", []) or [] if isinstance(item, Mapping)
        ]
        if attempts and any(not bool(item.get("success")) for item in attempts):
            status = "completed_with_failures"
        else:
            status = "completed"
        blockers = []
    return {
        "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "blockers": blockers,
        "simulator_result_status": simulator_result.get("status"),
        "normalized_attempt_trace_path": "normalized_attempt_trace.json",
        "failure_labels_path": "failure_labels.json",
        "prediction_outcome_ledger_path": "prediction_outcome_ledger.json",
        "calibration_report_path": "calibration_report.json",
        "breakage_library_path": "breakage_library.json",
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _proof_boundary(
    *,
    simulator: str,
    simulator_result: Mapping[str, Any],
    training_result: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    training_completed = bool(training_result.get("training_completed"))
    simulator_proven = bool(simulator_result.get("simulator_execution_proven")) and simulator != "fixture"
    return {
        "schema_version": PROOF_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_only",
        "fixture_only_proof": simulator == "fixture",
        "gpu_provisioning_performed": False,
        "simulators_run": bool(simulator_result.get("simulators_run")),
        "gpu_training_run": bool(training_result.get("gpu_training_run")),
        "simulator_execution_proven": simulator_proven,
        "robot_readiness_proven": False,
        "robot_policy_execution_proven": False,
        "physics_contact_validated": False,
        "safety_validated": False,
        "training_completed": training_completed,
        "public_claim_upgrade_allowed": False,
        "remaining_required_evidence": list(CLAIM_BOUNDARY["proof_upgrade_requires"]),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulators_run": bool(simulator_result.get("simulators_run")),
            "gpu_training_run": bool(training_result.get("gpu_training_run")),
            "simulator_execution_proven": simulator_proven,
            "training_completed": training_completed,
        },
    }


def _job_plan(
    *,
    job_id: str,
    request: Mapping[str, Any],
    validation: Mapping[str, Any],
    agent_plan: Mapping[str, Any],
    provisioner: str,
    simulator: str,
    job_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": JOB_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "blocked" if validation.get("status") == "blocked" else "planned",
        "operation": _string(request.get("operation") or "evaluate_only"),
        "provisioner": provisioner,
        "simulator": simulator,
        "job_dir": str(job_dir),
        "state_machine": [
            "request_loaded",
            "validation",
            "agent_orchestration_plan",
            "gpu_provisioning",
            "simulator_service",
            "training",
            "evaluation",
            "proof_boundary",
            "job_run_manifest",
        ],
        "agent_plan_status": agent_plan.get("status"),
        "deterministic_manifests_are_proof_source": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _blocked_manifest(
    *,
    job_id: str,
    blockers: Sequence[str],
    missing_inputs: Sequence[str],
    generated_at: str,
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": BLOCKED_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "blocked",
        "blockers": _dedupe(blockers),
        "missing_inputs": _dedupe(missing_inputs),
        "attempted_commands": ["build_robot_eval_job"],
        "evidence": dict(evidence),
        "blocker_category": "local_code_or_gated_infra",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _job_status(
    *,
    blockers: Sequence[str],
    simulator: str,
    simulator_result: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
) -> str:
    if blockers:
        return "blocked"
    if simulator == "fixture" and evaluation_result.get("status") == "completed":
        return "fixture_evaluation_completed"
    if simulator == "fixture" and evaluation_result.get("status") == "completed_with_failures":
        return "fixture_evaluation_completed_with_failures"
    if simulator_result.get("status") == "completed":
        return "simulator_command_completed"
    return _string(evaluation_result.get("status")) or "completed"


def _artifact_paths(job_dir: Path) -> Dict[str, str]:
    names = [
        "job_request.json",
        "job_validation.json",
        "job_plan.json",
        "agent_orchestration_plan.json",
        "gpu_provisioning_request.json",
        "gpu_provisioning_result.json",
        "simulator_service_request.json",
        "simulator_service_result.json",
        "policy_package_manifest.json",
        "training_request.json",
        "training_result.json",
        "evaluation_request.json",
        "evaluation_result.json",
        "normalized_attempt_trace.json",
        "failure_labels.json",
        "prediction_outcome_ledger.json",
        "calibration_report.json",
        "breakage_library.json",
        "proof_boundary.json",
        "job_run_manifest.json",
        "blocked_manifest.json",
    ]
    return {Path(name).stem: name for name in names if (job_dir / name).is_file()}


def build_robot_eval_job(
    *,
    capture_root: str | Path,
    job_request: str | Path | Mapping[str, Any],
    job_id: str,
    agent_adapter: RobotEvalJobAgentAdapter | None = None,
    provisioner: str = "fixture_local",
    simulator: str = "fixture",
    allow_gpu_provisioning: bool = False,
    allow_simulator_execution: bool = False,
    allowed_simulators: Sequence[str] = (),
    simulator_commands: Mapping[str, str] | None = None,
    allow_training: bool = False,
    training_command: str | None = None,
    timeout_seconds: int = 120,
    budget_usd: float | None = None,
) -> Dict[str, Any]:
    if provisioner not in PROVISIONERS:
        raise ValueError(f"Unsupported provisioner: {provisioner}")
    if simulator not in SIMULATORS:
        raise ValueError(f"Unsupported simulator: {simulator}")
    context = resolve_local_capture_context(capture_root)
    generated_at = utc_now_iso()
    pipeline_dir = context.pipeline_root
    job_dir = pipeline_dir / "robot_eval_jobs" / job_id
    ensure_dir(job_dir)

    request = _read_job_request(job_request)
    request.setdefault("schema_version", JOB_REQUEST_SCHEMA_VERSION)
    request.setdefault("job_id", job_id)
    request.setdefault("capture_root", str(context.capture_root))
    _write_job_json(job_dir, "job_request.json", request)

    missing_robot_eval_inputs = _ensure_robot_eval_cards(
        capture_root=context.capture_root,
        pipeline_dir=pipeline_dir,
    )
    policy_manifest, policy_missing_inputs, policy_missing_statuses = _policy_package_manifest(
        request=request,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "policy_package_manifest.json", policy_manifest)
    validation = _job_validation(
        request=request,
        policy_missing_inputs=policy_missing_inputs,
        policy_missing_statuses=policy_missing_statuses,
        missing_robot_eval_inputs=missing_robot_eval_inputs,
        generated_at=generated_at,
        pipeline_dir=pipeline_dir,
    )
    _write_job_json(job_dir, "job_validation.json", validation)

    plan_context = {
        "repo_root": str(Path(__file__).resolve().parents[2]),
        "capture_root": str(context.capture_root),
        "job_id": job_id,
        "request": request,
        "validation": validation,
        "provisioner": provisioner,
        "simulator": simulator,
    }
    agent_plan = _agent_plan(
        adapter=agent_adapter,
        plan_context=plan_context,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "agent_orchestration_plan.json", agent_plan)
    job_plan = _job_plan(
        job_id=job_id,
        request=request,
        validation=validation,
        agent_plan=agent_plan,
        provisioner=provisioner,
        simulator=simulator,
        job_dir=job_dir,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "job_plan.json", job_plan)

    gpu_request = _gpu_provisioning_request(
        request=request,
        job_id=job_id,
        provisioner=provisioner,
        budget_usd=budget_usd,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "gpu_provisioning_request.json", gpu_request)
    gpu_result = _gpu_provisioning_result(
        request_manifest=gpu_request,
        validation=validation,
        allow_gpu_provisioning=allow_gpu_provisioning,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "gpu_provisioning_result.json", gpu_result)

    sim_request = _simulator_request(
        job_id=job_id,
        simulator=simulator,
        request=request,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "simulator_service_request.json", sim_request)
    sim_result, copied_artifacts, simulator_blockers = _run_simulator(
        simulator=simulator,
        validation=validation,
        provisioning_result=gpu_result,
        capture_root=context.capture_root,
        pipeline_dir=pipeline_dir,
        job_dir=job_dir,
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=allowed_simulators,
        simulator_commands=simulator_commands or {},
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "simulator_service_result.json", sim_result)

    training_req = _training_request(request=request, generated_at=generated_at)
    _write_job_json(job_dir, "training_request.json", training_req)
    training_res = _training_result(
        request_manifest=training_req,
        allow_training=allow_training,
        training_command=training_command,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "training_result.json", training_res)

    eval_request = _evaluation_request(
        request=request,
        simulator_result=sim_result,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "evaluation_request.json", eval_request)
    eval_result = _evaluation_result(
        evaluation_request=eval_request,
        simulator_result=sim_result,
        copied_artifacts=copied_artifacts,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "evaluation_result.json", eval_result)

    proof_boundary = _proof_boundary(
        simulator=simulator,
        simulator_result=sim_result,
        training_result=training_res,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "proof_boundary.json", proof_boundary)

    blockers: List[str] = []
    missing_inputs: List[str] = []
    evidence: Dict[str, Any] = {
        "job_validation_status": validation.get("status"),
        "gpu_provisioning_status": gpu_result.get("status"),
        "simulator_service_status": sim_result.get("status"),
        "training_status": training_res.get("status"),
        "evaluation_status": eval_result.get("status"),
    }
    if validation.get("status") == "blocked":
        blockers.extend(_string_list(validation.get("blockers")))
        missing_inputs.extend(_string_list(validation.get("missing_inputs")))
    elif gpu_result.get("status") == "blocked":
        blockers.append("gpu_provisioning_blocked")
        missing_inputs.extend(_string_list(gpu_result.get("blockers")))
    if simulator_blockers and validation.get("status") != "blocked":
        blockers.extend(simulator_blockers)
        evidence["simulator_blockers"] = _string_list(sim_result.get("blockers"))
    if training_res.get("status") == "blocked":
        blockers.append("training_blocked")
        missing_inputs.extend(_string_list(training_res.get("blockers")))

    status = _job_status(
        blockers=blockers,
        simulator=simulator,
        simulator_result=sim_result,
        evaluation_result=eval_result,
    )
    if blockers:
        blocked = _blocked_manifest(
            job_id=job_id,
            blockers=blockers,
            missing_inputs=missing_inputs,
            generated_at=generated_at,
            evidence=evidence,
        )
        _write_job_json(job_dir, "blocked_manifest.json", blocked)

    run_manifest = {
        "schema_version": JOB_RUN_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": status,
        "state": "blocked" if blockers else "completed",
        "capture_root": str(context.capture_root),
        "job_dir": str(job_dir),
        "operation": _string(request.get("operation") or "evaluate_only"),
        "provisioner": provisioner,
        "simulator": simulator,
        "agent_orchestration_status": agent_plan.get("status"),
        "validation_status": validation.get("status"),
        "gpu_provisioning_status": gpu_result.get("status"),
        "simulator_service_status": sim_result.get("status"),
        "training_status": training_res.get("status"),
        "evaluation_status": eval_result.get("status"),
        "blockers": _dedupe(blockers),
        "missing_inputs": _dedupe(missing_inputs),
        "artifacts": {},
        "live_provider_calls_performed": False,
        "remote_asset_downloads_performed": False,
        "simulators_run": bool(sim_result.get("simulators_run")),
        "gpu_training_run": bool(training_res.get("gpu_training_run")),
        "messages_sent": False,
        "payments_touched": False,
        "deployments_performed": False,
        "simulator_execution_proven": bool(proof_boundary.get("simulator_execution_proven")),
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    run_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "job_id": job_id,
            "validation": validation,
            "gpu_result": gpu_result,
            "sim_result": sim_result,
            "training_result": training_res,
            "evaluation_result": eval_result,
        }
    )
    _write_job_json(job_dir, "job_run_manifest.json", run_manifest)
    run_manifest["artifacts"] = _artifact_paths(job_dir)
    _write_job_json(job_dir, "job_run_manifest.json", run_manifest)

    return {
        "schema_version": "robot_eval_job_result.v1",
        "job_id": job_id,
        "capture_root": str(context.capture_root),
        "job_dir": str(job_dir),
        "manifest_path": str((job_dir / "job_run_manifest.json").resolve()),
        "status": status,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _parse_simulator_commands(values: Sequence[str] | None) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for value in values or []:
        framework, sep, command = value.partition("=")
        if not sep or framework not in SIMULATORS or framework == "fixture" or not command.strip():
            raise ValueError(
                "--simulator-command must be formatted as "
                "<mujoco|pybullet|newton|isaac_sim>=<command>"
            )
        commands[framework] = command.strip()
    return commands


def _agent_adapter_from_mode(mode: str) -> RobotEvalJobAgentAdapter | None:
    if mode == "fake":
        return FakeRobotEvalJobAgentAdapter()
    if mode == "agents-sdk":
        return AgentsSdkRobotEvalJobAdapter()
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fail-closed headless robot-eval job from a local request manifest"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--job-request", required=True, help="Robot eval job request JSON")
    parser.add_argument("--job-id", required=True, help="Deterministic job id")
    parser.add_argument(
        "--agent-mode",
        choices=("none", "fake", "agents-sdk"),
        default="none",
        help="Optional advisory agent adapter; deterministic manifests remain authoritative",
    )
    parser.add_argument("--provisioner", choices=PROVISIONERS, default="fixture_local")
    parser.add_argument("--simulator", choices=SIMULATORS, default="fixture")
    parser.add_argument(
        "--allow-gpu-provisioning",
        action="store_true",
        help="Permit gated non-fixture provisioning only with matching environment approval",
    )
    parser.add_argument(
        "--allow-simulator-execution",
        action="store_true",
        help="Permit non-fixture simulator command execution only with matching env approval",
    )
    parser.add_argument(
        "--allow-simulator",
        action="append",
        choices=tuple(item for item in SIMULATORS if item != "fixture"),
        default=[],
        help="Allow one non-fixture simulator framework; repeat for multiple frameworks",
    )
    parser.add_argument(
        "--simulator-command",
        action="append",
        default=[],
        help="Explicit simulator command as <framework>=<command>",
    )
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Permit command-based Cosmos training only with matching environment approval",
    )
    parser.add_argument("--training-command", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--budget-usd", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        result = build_robot_eval_job(
            capture_root=args.capture_root,
            job_request=args.job_request,
            job_id=args.job_id,
            agent_adapter=_agent_adapter_from_mode(args.agent_mode),
            provisioner=args.provisioner,
            simulator=args.simulator,
            allow_gpu_provisioning=args.allow_gpu_provisioning,
            allow_simulator_execution=args.allow_simulator_execution,
            allowed_simulators=args.allow_simulator,
            simulator_commands=_parse_simulator_commands(args.simulator_command),
            allow_training=args.allow_training,
            training_command=args.training_command,
            timeout_seconds=args.timeout_seconds,
            budget_usd=args.budget_usd,
        )
    except (OSError, ValueError) as exc:
        print(f"[robot-eval-job] FAILED: {exc}")
        return 1
    print(f"[robot-eval-job] manifest={result['manifest_path']}")
    print(f"[robot-eval-job] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
