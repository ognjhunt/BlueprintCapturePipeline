"""Headless robot-eval job orchestration lane with gated live SDK operators.

This module coordinates a repo-local robot-team evaluation or training request
through deterministic validation, provisioning, simulator, training, and
evaluation manifests. Provider, GPU, simulator, training, and agent execution
paths run only when explicit environment and CLI gates are present. Agents may
coordinate messy failures, retries, review routing, summaries, and next-command
selection, while deterministic artifacts own validation, packaging, rerun policy,
checksums, and proof booleans.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlparse

from .agent_operator_runtime import (
    LIVE_AGENTS_SDK_ENV,
    OperatorExecutor,
    OperatorRunConfig,
    blocked_operator_ledger,
    completed_operator_ledger,
    env_truthy,
    external_action_gates,
    proof_effect,
    run_agents_sdk_operator,
)
from .arena_result_ingest import build_arena_result_ingest
from .action_normalization import build_action_normalization_from_trace
from .benchmark_protocol import execute_benchmark_protocol_request
from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .cpu_simulator_preflight import CPU_BACKENDS, build_cpu_simulator_preflight
from .episode_spec import build_episode_specs
from .evaluator_qualification_workflow import build_evaluator_qualification_workflow
from .evaluation_run import compile_evaluation_run
from .failure_diagnosis_contract import (
    FAILURE_LABEL_PROOF_EFFECT,
    dedupe as _dedupe_refs,
    evidence_refs as _failure_evidence_refs,
    failure_root_cause_category as _failure_root_cause_category,
    frame_or_clip_refs as _failure_frame_or_clip_refs,
    remediation_candidate as _failure_remediation_candidate,
    review_status_for_failure_label as _failure_review_status,
)
from .local_capture import resolve_local_capture_context
from .live_robot_eval_closure import build_live_robot_eval_closure_manifest
from .pipeline_settings import PipelineSettings
from .post_training_data_package import build_post_training_data_package_export
from .canonical_training_quality_pipeline import (
    run_canonical_training_quality_from_request,
)
from .robot_eval_gpu_startup_pipeline import build_gpu_startup_pipeline_plan
from . import robot_eval_closure_decisions as _closure_decisions
from .robot_eval_claim_contracts import robot_eval_job_claim_boundary
from .robot_eval_execution import (
    build_scenario_eval_matrix,
    build_deployment_validation_bundle,
    build_policy_execution_bundle,
    build_robot_pov_observation_bundle,
    build_simulator_command_artifacts,
    default_test_policy_package_from_request,
    fingerprint_execution_artifacts,
)
from .sc3_eval_protocol import SC3_EVAL_PROTOCOL_ARTIFACT, build_sc3_eval_protocol_artifact
from .robot_eval_dataset import build_real_site_robot_eval_dataset
from .robot_eval_job_request_contract import (
    ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT,
    ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION,
)
from .robot_eval_evaluation_run_adapter import (
    build_robot_eval_evaluation_run_spec,
    execute_robot_eval_request_as_evaluation_run,
    execute_robot_eval_cli_evaluation_run,
)
from .scene_asset_preflight import build_scene_asset_preflight
from .security_controls import contained_path, strict_identifier
from .simulation_automation import build_simulation_automation
from .site_eval_director import build_site_eval_director
from .success_claim_contracts import (
    build_artifact_freshness_evidence,
    build_contact_state_change_proof,
    build_media_validity,
    build_physical_readiness,
    build_policy_action_execution,
    build_review_task_success,
    build_simulator_execution,
    build_task_success_contract_result,
    coerce_strict_success,
    derive_task_proof_requirements,
)
from .task_eval_run_report import build_task_eval_run_report
from .wam_eval_substrate import (
    WAM_EVALUATION_SUBSTRATES,
    requested_evaluation_substrate,
)
from .wam_fixture_evaluator import run_wam_eval_job
from .wam_provider_runtime import parse_wam_provider_commands
from .wam_score_claim_gate import WAM_SCORE_CLAIM_GATE_SCHEMA_VERSION


JOB_REQUEST_SCHEMA_VERSION = ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION
JOB_VALIDATION_SCHEMA_VERSION = "robot_eval_job_validation.v1"
JOB_REQUEST_ENRICHMENT_SCHEMA_VERSION = "robot_eval_job_request_enrichment.v1"
JOB_PLAN_SCHEMA_VERSION = "robot_eval_job_plan.v1"
AGENT_ORCHESTRATION_PLAN_SCHEMA_VERSION = "robot_eval_agent_orchestration_plan.v1"
GPU_PROVISIONING_REQUEST_SCHEMA_VERSION = "robot_eval_gpu_provisioning_request.v1"
GPU_PROVISIONING_RESULT_SCHEMA_VERSION = "robot_eval_gpu_provisioning_result.v1"
SCHEDULER_DECISION_SCHEMA_VERSION = "robot_eval_execution_scheduler_decision.v1"
WORKER_LAUNCH_PLAN_SCHEMA_VERSION = "robot_eval_worker_launch_plan.v1"
GPU_PROVIDER_LAUNCH_REQUEST_SCHEMA_VERSION = "robot_eval_gpu_provider_launch_request.v1"
GPU_PROVIDER_RACE_HANDOFF_SCHEMA_VERSION = "robot_eval_gpu_provider_race_handoff.v1"
PROVIDER_RACE_RUNTIME_LAUNCHER_BLOCKER = (
    "provider_race_runtime_launcher_not_implemented"
)
PROVIDER_RACE_LAUNCHER_COMMAND = "blueprint-run-robot-eval-provider-race"
PROVIDER_RACE_LAUNCHER_RESULT_NAME = "gpu_provider_race_launcher_result.json"
PROVIDER_PRELAUNCH_SPEND_GUARD_SCHEMA_VERSION = "robot_eval_provider_prelaunch_spend_guard.v1"
GPU_COST_CONTROL_LEDGER_SCHEMA_VERSION = "robot_eval_gpu_cost_control_ledger.v1"
REMOTE_CLOUD_EXECUTION_CLOSURE_SCHEMA_VERSION = "robot_eval_remote_cloud_execution_closure.v1"
SIMULATOR_SERVICE_REQUEST_SCHEMA_VERSION = "robot_eval_simulator_service_request.v1"
SIMULATOR_SERVICE_RESULT_SCHEMA_VERSION = "robot_eval_simulator_service_result.v1"
SIMULATOR_PROVIDER_ADAPTER_SCHEMA_VERSION = "robot_eval_simulator_provider_adapter_manifest.v1"
SIMULATOR_SELECTION_POLICY_SCHEMA_VERSION = "robot_eval_simulator_selection_policy.v1"
POLICY_PACKAGE_MANIFEST_SCHEMA_VERSION = "robot_eval_policy_package_manifest.v1"
TRAINING_REQUEST_SCHEMA_VERSION = "robot_eval_training_request.v1"
TRAINING_RESULT_SCHEMA_VERSION = "robot_eval_training_result.v1"
EVALUATION_REQUEST_SCHEMA_VERSION = "robot_eval_evaluation_request.v1"
EVALUATION_RESULT_SCHEMA_VERSION = "robot_eval_evaluation_result.v1"
ROBOT_EVAL_REPORT_SCHEMA_VERSION = "robot_eval_job_report.v1"
ROBOT_TEAM_GRADE_EVAL_CLOSURE_SCHEMA_VERSION = (
    _closure_decisions.ROBOT_TEAM_GRADE_EVAL_CLOSURE_SCHEMA_VERSION
)
WEBAPP_ROBOT_EVAL_STATUS_PROJECTION_SCHEMA_VERSION = (
    _closure_decisions.WEBAPP_ROBOT_EVAL_STATUS_PROJECTION_SCHEMA_VERSION
)
NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION = "robot_eval_job_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "robot_eval_job_failure_labels.v1"
PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION = "robot_eval_job_prediction_outcome_ledger.v1"
CALIBRATION_REPORT_SCHEMA_VERSION = "robot_eval_job_calibration_report.v1"
BREAKAGE_LIBRARY_SCHEMA_VERSION = "robot_eval_job_breakage_library.v1"
PROOF_BOUNDARY_SCHEMA_VERSION = "robot_eval_job_proof_boundary.v1"
JOB_RUN_MANIFEST_SCHEMA_VERSION = "robot_eval_job_run_manifest.v1"
BLOCKED_MANIFEST_SCHEMA_VERSION = "robot_eval_job_blocked_manifest.v1"
JOB_REQUEST_INBOX_RUN_SCHEMA_VERSION = "robot_eval_job_request_inbox_run.v1"
ROBOT_EVAL_QUEUE_TERMINAL_SUCCESS = "terminal_success"
ROBOT_EVAL_QUEUE_PERMANENT_INVALID = "permanent_invalid"
ROBOT_EVAL_QUEUE_RETRYABLE_BLOCKED = "retryable_blocked"
ROBOT_EVAL_QUEUE_FATAL_INFRASTRUCTURE = "fatal_infrastructure"
ROBOT_EVAL_RETRYABLE_EXIT_CODE = 75
ROBOT_EVAL_PERMANENT_INVALID_EXIT_CODE = 65
ROBOT_EVAL_FATAL_INFRASTRUCTURE_EXIT_CODE = 70
WEBAPP_FORWARD_CAPTURE_ROOT_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT"
WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"
REAL_WORLD_VALIDATION_FOLLOWUP_REQUEST_QUEUE_SCHEMA_VERSION = (
    "real_world_validation_followup_request_queue.v1"
)


class InboxRequestChangedError(RuntimeError):
    """The producer replaced an inbox request while the consumer was snapshotting it."""

PROVISIONERS = (
    "fixture_local",
    "local_process",
    "docker_local",
    "vast",
    "runpod",
    "gcp",
)
SIMULATORS = ("fixture", "mujoco", "pybullet", "newton", "isaac_sim", "isaac_lab_arena")
ISAAC_SIMULATORS = {"isaac_sim", "isaac_lab_arena"}
DEFAULT_SIMULATOR_SELECTION_POLICY_MODE = "mujoco_first_unless_proof_requires_isaac"
DEFAULT_MUJOCO_FIRST_REASONS = [
    "cheapest_first_real_simulator_pass",
    "fast_cpu_or_low_cost_owner_runtime",
    "compatible_mjcf_robot_asset_or_default_unitree_g1_smoke",
    "early_policy_and_spawn_smoke_before_gpu_spend",
]
DEFAULT_ISAAC_ESCALATION_REASONS = [
    "rich_usd_or_openusd_scene_load_required",
    "isaac_robot_asset_proof_required",
    "rtx_sensor_or_camera_rendering_required",
    "contact_or_physics_validation_requires_isaac_stack",
]
DEFAULT_ARENA_ESCALATION_REASONS = [
    "isaac_lab_arena_batch_rollouts_required",
    "large_scenario_matrix_or_sharded_eval_required",
    "owner_arena_result_ingest_required",
]
DEFAULT_STARTUP_EXPECTED_OUTPUTS = [
    "scheduler_decision",
    "worker_launch_plan",
    "gpu_startup_pipeline_plan",
    "worker_manifest",
    "gpu_provider_launch_request",
    "gpu_provider_launcher_result",
    "runpod_provider_adapter_result",
    "gpu_cost_control_ledger",
    "startup_architecture_audit",
    "worker_runtime_manifest",
    "worker_runtime_preflight",
    "job_run_manifest",
    "proof_boundary",
    "metrics",
    "trace",
    "simulator_pov",
    "stdout_log",
    "stderr_log",
]
OPERATIONS = ("evaluate_only", "train_only", "train_then_evaluate")
LIVE_GPU_PROVISIONERS = {"vast", "runpod", "lambda_cloud", "gcp"}
WORKER_MANIFEST_SCHEMA_VERSION = "robot_eval_worker_manifest.v1"
WORKER_MANIFEST_URI_ENV = "BLUEPRINT_EVAL_MANIFEST_URI"
WORKER_ARTIFACT_OUTPUT_URI_ENV = "BLUEPRINT_ARTIFACT_OUTPUT_URI"
WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV = "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI"
REMOTE_WORKER_MANIFEST_URI_SCHEMES = {"https", "gs", "s3", "r2"}
REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES = {"gs", "s3", "r2"}
ARTIFACT_OUTPUT_WRITE_SECRET_ENV_VARS_BY_SCHEME = {
    "gs": ["GOOGLE_APPLICATION_CREDENTIALS"],
    "s3": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "r2": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
}
ARTIFACT_OUTPUT_WRITE_PLAINTEXT_ENV_VARS_BY_SCHEME = {
    "r2": ["BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL", "AWS_ENDPOINT_URL"],
}
GENERIC_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"
GENERIC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV = (
    "BLUEPRINT_WORKER_IMAGE_MANIFEST_DIAGNOSTIC"
)
WORKER_IMAGE_REF_ENV_BY_SIMULATOR = {
    "isaac_sim": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
    "isaac_lab_arena": "BLUEPRINT_ISAAC_ARENA_EVAL_WORKER_IMAGE_REF",
    "mujoco": "BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF",
    "pybullet": "BLUEPRINT_PYBULLET_EVAL_WORKER_IMAGE_REF",
    "newton": "BLUEPRINT_NEWTON_EVAL_WORKER_IMAGE_REF",
}
WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV_BY_SIMULATOR = {
    "isaac_sim": "BLUEPRINT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC",
    "isaac_lab_arena": "BLUEPRINT_ISAAC_ARENA_WORKER_IMAGE_MANIFEST_DIAGNOSTIC",
}

SIMULATOR_PROVIDER_PROFILES: Dict[str, Dict[str, Any]] = {
    "fixture": {
        "provider_family": "repo_local_fixture",
        "execution_surface": "site_eval_director_artifacts",
        "command_required": False,
        "optional_dependencies": [],
        "default_output_contract": "site_eval_director_normalized_artifacts",
    },
    "mujoco": {
        "provider_family": "cpu_physics_engine",
        "execution_surface": "gated_owner_command",
        "command_required": True,
        "optional_dependencies": ["mujoco"],
        "default_output_contract": "robot_eval_simulator_command_output.v1",
    },
    "pybullet": {
        "provider_family": "cpu_physics_engine",
        "execution_surface": "gated_owner_command",
        "command_required": True,
        "optional_dependencies": ["pybullet"],
        "default_output_contract": "robot_eval_simulator_command_output.v1",
    },
    "newton": {
        "provider_family": "gpu_physics_engine",
        "execution_surface": "gated_owner_command",
        "command_required": True,
        "optional_dependencies": ["newton"],
        "default_output_contract": "robot_eval_simulator_command_output.v1",
    },
    "isaac_sim": {
        "provider_family": "gpu_simulator",
        "execution_surface": "gated_owner_command",
        "command_required": True,
        "optional_dependencies": ["isaac-sim"],
        "default_output_contract": "robot_eval_simulator_command_output.v1",
    },
    "isaac_lab_arena": {
        "provider_family": "isaac_lab_arena_batch_harness",
        "execution_surface": "gated_owner_command_or_owner_results_ingest",
        "command_required": True,
        "optional_dependencies": ["isaac-sim", "isaac-lab"],
        "default_output_contract": "robot_eval_simulator_command_output.v1",
    },
}

REQUIRED_ROBOT_EVAL_INPUTS = {
    "robot_eval_site_card": "robot_eval_dataset/site_card.json",
    "robot_eval_task_cards": "robot_eval_dataset/task_cards.json",
    "robot_eval_scenario_cards": "robot_eval_dataset/scenario_cards.json",
    "robot_eval_cards": "robot_eval_dataset/eval_cards.json",
    "robot_eval_proof_boundaries": "robot_eval_dataset/proof_boundaries.json",
}

POLICY_MODALITY_ORDER = _closure_decisions.POLICY_MODALITY_ORDER

POLICY_MODALITY_STATUSES = {
    "policy_api_endpoint": "needs_policy_api_endpoint_ref",
    "docker_container": "needs_docker_container_ref",
    "recorded_action_trace": "needs_recorded_action_trace_ref",
    "high_level_skill_trace": "needs_high_level_skill_trace_ref",
    "teleop_demo": "needs_teleop_demo_ref",
    "sim_controller_plugin": "needs_sim_controller_plugin_ref",
}
POLICY_OBSERVATION_SCHEMA_ID = "blueprint.robot_eval.observation.v1"
POLICY_ACTION_SCHEMA_ID = "blueprint.robot_eval.action_trace.v1"
POLICY_OBSERVATION_SCHEMA_REF = "blueprint://schemas/robot_eval_observation.v1"
POLICY_ACTION_SCHEMA_REF = "blueprint://schemas/robot_eval_action_trace.v1"
POLICY_REQUIRED_OBSERVATION_FIELDS = (
    "observation_id",
    "scenario_eval_run_id",
    "scenario_variation_instance_id",
    "task_id",
    "scenario_id",
    "camera",
    "robot_profile_id",
    "render_frame_paths",
)
POLICY_REQUIRED_ACTION_OUTPUT_FIELDS = (
    "scenario_eval_run_id",
    "scenario_variation_instance_id",
    "task_id",
    "scenario_id",
    "status",
    "success",
    "actions",
    "metrics",
    "failure_mode_ids",
)

CLAIM_BOUNDARY: Dict[str, Any] = robot_eval_job_claim_boundary()

# Compatibility aliases during the characterization-backed module split.
_scenario_eval_matrix_runs = _closure_decisions._scenario_eval_matrix_runs
_artifact_paths = _closure_decisions._artifact_paths
_explicitly_blocked_scenario_eval_run_records = (
    _closure_decisions._explicitly_blocked_scenario_eval_run_records
)
_valid_explicitly_blocked_scenario_eval_run_ids = (
    _closure_decisions._valid_explicitly_blocked_scenario_eval_run_ids
)
_capture_root_from_job_dir = _closure_decisions._capture_root_from_job_dir
_webapp_robot_eval_status_projection = (
    _closure_decisions.build_webapp_robot_eval_status_projection
)
_robot_team_grade_eval_closure_manifest = (
    _closure_decisions.build_robot_team_grade_eval_closure
)


class RobotEvalJobAgentAdapter(Protocol):
    def build_plan(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class FakeRobotEvalJobAgentAdapter:
    """Network-free deterministic agent adapter for local state-machine tests."""

    adapter_name: str = "fake"

    def build_plan(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "schema_version": AGENT_ORCHESTRATION_PLAN_SCHEMA_VERSION,
            "adapter": self.adapter_name,
            "status": "completed",
            "operator_mode": "deterministic_test_operator",
            "agent_authority": "deterministic_test_operator",
            "proof_booleans_mutable_by_agent": False,
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
            "operator_ledger": completed_operator_ledger(
                adapter=self.adapter_name,
                output={
                    "final_output": "Local fake adapter selected the deterministic validation-first path.",
                    "commands_chosen": ["validate_job_request"],
                },
                default_command="validate_job_request",
                proof_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"],
            ),
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
    """Optional OpenAI Agents SDK live robot-eval operator."""

    agents_sdk_available: bool | None = None
    openai_api_key: str | None = None
    live_env_allowed: bool | None = None
    allow_live_operator: bool = False
    model: str = "gpt-4.1"
    executor: OperatorExecutor | None = None

    def build_plan(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        agents_available = (
            self.agents_sdk_available
            if self.agents_sdk_available is not None
            else bool(self.executor is not None) or _module_available(("agents", "openai_agents"))
        )
        api_key_present = bool(
            _string(self.openai_api_key)
            if self.openai_api_key is not None
            else _string(os.getenv("OPENAI_API_KEY"))
        )
        env_allowed = (
            bool(self.live_env_allowed)
            if self.live_env_allowed is not None
            else env_truthy(LIVE_AGENTS_SDK_ENV)
            or _env_truthy("BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION")
        )
        blockers: List[str] = []
        if not agents_available:
            blockers.append("missing_openai_agents_sdk")
        if not api_key_present:
            blockers.append("missing_openai_api_key")
        if not self.allow_live_operator:
            blockers.append("missing_cli_allow_live_agent_operator")
        if not env_allowed:
            blockers.append(f"missing_env_{LIVE_AGENTS_SDK_ENV}")
        command = "choose_next_deterministic_robot_eval_command"
        live_output: Dict[str, Any] | None = None
        execution_performed = False
        execution_failed = False
        if not blockers:
            try:
                live_output = run_agents_sdk_operator(
                    OperatorRunConfig(
                        adapter="openai_agents_sdk_robot_eval_job",
                        model=self.model,
                        prompt=_agents_sdk_robot_eval_job_prompt(plan_context),
                        plan_context=plan_context,
                        executor=self.executor,
                    )
                )
                execution_performed = True
            except RuntimeError as exc:
                blockers.append(str(exc))
                execution_failed = True
            except Exception as exc:
                blockers.append(f"agents_sdk_operator_execution_failed:{type(exc).__name__}")
                execution_failed = True
        status = (
            "operator_completed"
            if execution_performed and not blockers
            else "operator_failed"
            if execution_failed
            else "blocked"
        )
        operator_ledger = (
            completed_operator_ledger(
                adapter="openai_agents_sdk_robot_eval_job",
                output=live_output or {},
                default_command=command,
                proof_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"],
            )
            if execution_performed and not blockers
            else blocked_operator_ledger(
                adapter="openai_agents_sdk_robot_eval_job",
                blockers=blockers,
                command_chosen=command if not blockers else None,
                proof_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"],
            )
        )
        return {
            "schema_version": AGENT_ORCHESTRATION_PLAN_SCHEMA_VERSION,
            "adapter": "openai_agents_sdk_robot_eval_job",
            "status": status,
            "blockers": blockers,
            "missing_inputs": list(blockers),
            "operator_mode": "live_operator" if execution_performed else "live_operator_blocked",
            "agent_authority": "live_operator_when_gated",
            "proof_booleans_mutable_by_agent": False,
            "execution_performed": execution_performed,
            "network_required_if_executed": True,
            "operator_ledger": operator_ledger,
            "request": {
                "purpose": "headless_robot_eval_job_orchestration_live_operator",
                "model": self.model,
                "job_id": _string(plan_context.get("job_id")),
                "capture_root": _string(plan_context.get("capture_root")),
                "allowed_actions": [
                    "choose_next_deterministic_command",
                    "inspect_manifests_and_logs",
                    "trigger_allowed_deterministic_reruns",
                    "request_gpu_or_simulator_provisioning",
                    "summarize_blockers",
                    "route_for_human_review",
                    "maintain_progress_ledger",
                ],
                "prohibited_actions": [
                    "override_rights_privacy_blockers",
                    "mark_rank_fidelity_result_proven",
                    "mark_simulator_proof_complete_without_result_manifest",
                    "mark_training_proof_complete_without_checkpoint_manifest",
                    "spend_money_without_explicit_gates",
                    "call_live_providers_without_explicit_gates",
                ],
            },
            "evidence": {
                "openai_agents_sdk_available": bool(agents_available),
                "openai_api_key_present": api_key_present,
                "cli_allow_live_operator": self.allow_live_operator,
                LIVE_AGENTS_SDK_ENV: env_allowed,
                "legacy_env_BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION": _env_truthy(
                    "BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION"
                ),
                **external_action_gates(),
            },
            "proof_effect": proof_effect(
                deterministic_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"]
            ),
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }


def _module_available(candidates: Sequence[str]) -> bool:
    return any(importlib.util.find_spec(candidate) is not None for candidate in candidates)


def _agents_sdk_robot_eval_job_prompt(plan_context: Mapping[str, Any]) -> str:
    return (
        "Act as the Blueprint Agents SDK robot-eval pipeline operator. Inspect the "
        "request, validation, preflight, simulation, and job context. Choose the next "
        "safe deterministic command or allowed rerun, summarize blockers, route human "
        "review when needed, and maintain a progress ledger. Do not set proof booleans "
        "directly; proof can only come from deterministic accepted artifacts.\n\n"
        f"{json.dumps(plan_context, sort_keys=True, default=str)[:12000]}"
    )


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


def _number_field(payload: Mapping[str, Any], *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return _number(payload.get(key), default)
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


def _first_allowed_backend(candidates: Sequence[str], allowed_backends: Sequence[str]) -> str:
    allowed = set(allowed_backends or SIMULATORS)
    for candidate in candidates:
        if candidate in SIMULATORS and candidate in allowed:
            return candidate
    for fallback in ("mujoco", "isaac_sim", "isaac_lab_arena", "pybullet", "fixture"):
        if fallback in allowed:
            return fallback
    return "fixture"


def _routing_backend(value: Any) -> str:
    backend = _string(value)
    return backend if backend in SIMULATORS else ""


def _simulator_role(simulator: str, recommended_backend: str, routing: Mapping[str, Any]) -> str:
    if simulator == recommended_backend:
        return "recommended_policy_backend"
    proxy_backends = set(_string_list(routing.get("proxy_backends") or routing.get("proxyBackends")))
    escalation_backends = set(
        _string_list(routing.get("escalation_backends") or routing.get("escalationBackends"))
    )
    if simulator == "fixture":
        return "fixture_local_orchestration_only"
    if simulator in proxy_backends or simulator in {"mujoco", "pybullet"}:
        return "proxy_or_owner_accepted_backend"
    if simulator in escalation_backends or simulator in ISAAC_SIMULATORS:
        return "isaac_escalation_backend"
    return "operator_selected_backend"


def _required_proof_classes(
    *,
    request: Mapping[str, Any],
    routing: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> List[str]:
    values: List[str] = []
    for source in (
        routing.get("required_proof_classes"),
        routing.get("requiredProofClasses"),
        policy.get("required_proof_classes"),
        policy.get("requiredProofClasses"),
        request.get("required_proof_classes"),
        request.get("requiredProofClasses"),
    ):
        values.extend(_string_list(source))
    return _dedupe(values)


def resolve_simulator_selection_policy(
    request: Mapping[str, Any],
    *,
    selected_simulator: str = "fixture",
) -> Dict[str, Any]:
    """Resolve the request's simulator routing intent without running a simulator."""

    execution_request = _mapping(request.get("execution_request"))
    routing = _mapping(execution_request.get("simulator_routing"))
    policy = _mapping(routing.get("selection_policy") or routing.get("selectionPolicy"))
    policy_mode = _string(policy.get("mode")) or DEFAULT_SIMULATOR_SELECTION_POLICY_MODE
    allowed_backends = _string_list(routing.get("allowed_backends") or routing.get("allowedBackends"))
    if not allowed_backends:
        allowed_backends = list(SIMULATORS)
    requested_backend = _routing_backend(
        routing.get("requested_backend") or routing.get("requestedBackend")
    )
    request_simulator_preference = _string(
        request.get("simulator_preference") or request.get("simulatorPreference")
    )
    explicit_simulator_preference = _routing_backend(request_simulator_preference)
    default_first_pass_backend = _routing_backend(
        policy.get("first_pass_backend")
        or policy.get("firstPassBackend")
        or routing.get("default_first_pass_backend")
        or routing.get("defaultFirstPassBackend")
        or routing.get("default_first_gpu_backend")
        or routing.get("defaultFirstGpuBackend")
    )
    if not default_first_pass_backend:
        default_first_pass_backend = "mujoco"

    required_classes = _required_proof_classes(
        request=request,
        routing=routing,
        policy=policy,
    )
    normalized_classes = {item.lower() for item in required_classes}
    arena_reasons = _string_list(
        policy.get("use_isaac_lab_arena_when") or policy.get("useIsaacLabArenaWhen")
    ) or list(DEFAULT_ARENA_ESCALATION_REASONS)
    isaac_reasons = _string_list(
        policy.get("escalate_to_isaac_when") or policy.get("escalateToIsaacWhen")
    ) or list(DEFAULT_ISAAC_ESCALATION_REASONS)
    mujoco_reasons = _string_list(
        policy.get("use_mujoco_when") or policy.get("useMujocoWhen")
    ) or list(DEFAULT_MUJOCO_FIRST_REASONS)
    arena_reason_set = {item.lower() for item in arena_reasons}
    isaac_reason_set = {item.lower() for item in isaac_reasons}

    escalation_required = False
    recommendation_reasons: List[str] = []
    if normalized_classes.intersection(arena_reason_set):
        recommended_backend = _first_allowed_backend(
            ["isaac_lab_arena", "isaac_sim", default_first_pass_backend],
            allowed_backends,
        )
        escalation_required = recommended_backend in ISAAC_SIMULATORS
        recommendation_reasons = sorted(normalized_classes.intersection(arena_reason_set))
    elif normalized_classes.intersection(isaac_reason_set):
        recommended_backend = _first_allowed_backend(
            ["isaac_sim", "isaac_lab_arena", default_first_pass_backend],
            allowed_backends,
        )
        escalation_required = recommended_backend in ISAAC_SIMULATORS
        recommendation_reasons = sorted(normalized_classes.intersection(isaac_reason_set))
    elif requested_backend and requested_backend != "fixture":
        recommended_backend = _first_allowed_backend([requested_backend], allowed_backends)
        recommendation_reasons = ["explicit_requested_backend"]
    elif explicit_simulator_preference and explicit_simulator_preference != "fixture":
        recommended_backend = _first_allowed_backend([explicit_simulator_preference], allowed_backends)
        recommendation_reasons = ["explicit_simulator_preference"]
    else:
        recommended_backend = _first_allowed_backend([default_first_pass_backend], allowed_backends)
        recommendation_reasons = list(mujoco_reasons if recommended_backend == "mujoco" else [])

    selected_allowed = selected_simulator in allowed_backends
    warnings: List[str] = []
    if selected_simulator != recommended_backend:
        warnings.append("selected_simulator_differs_from_request_policy_recommendation")
    if selected_simulator == "fixture" and recommended_backend != "fixture":
        warnings.append("fixture_local_loop_does_not_satisfy_customer_eval_backend_policy")
    if not selected_allowed:
        warnings.append("selected_simulator_not_allowed_by_request_policy")

    return {
        "schema_version": SIMULATOR_SELECTION_POLICY_SCHEMA_VERSION,
        "mode": policy_mode,
        "requested_backend": requested_backend or _string(routing.get("requested_backend")) or None,
        "request_simulator_preference": request_simulator_preference or None,
        "allowed_backends": list(allowed_backends),
        "selected_backend": selected_simulator,
        "recommended_backend": recommended_backend,
        "selected_backend_allowed_by_request": selected_allowed,
        "selected_backend_matches_recommendation": selected_simulator == recommended_backend,
        "selected_backend_role": _simulator_role(selected_simulator, recommended_backend, routing),
        "default_first_pass_backend": default_first_pass_backend,
        "mujoco_first_applies": recommended_backend == "mujoco" and not escalation_required,
        "escalation_required": escalation_required,
        "required_proof_classes": required_classes,
        "recommendation_reasons": recommendation_reasons,
        "use_mujoco_when": mujoco_reasons,
        "escalate_to_isaac_when": isaac_reasons,
        "use_isaac_lab_arena_when": arena_reasons,
        "non_blocking_warnings": _dedupe(warnings),
        "proof_boundary": {
            "webapp_request_selects_policy_not_execution": True,
            "mujoco_proof_does_not_clear_isaac_sim_gate": True,
            "isaac_sim_proof_does_not_clear_real_robot_or_safety_gate": True,
            "simulator_execution_proven_by_this_policy": False,
            "rank_fidelity_result_proven_by_this_policy": False,
        },
    }


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "on", "allowed", "cleared"}


def _strict_bool(value: Any) -> bool:
    return value is True


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _relative_if_file(base_dir: Path, target: Path) -> str | None:
    return _relative_to(base_dir, target) if target.is_file() else None


def _sha_payload(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _claim_robot_eval_job_execution(
    *,
    job_dir: Path,
    job_id: str,
    request_fingerprint: str,
    generated_at: str,
) -> Dict[str, Any]:
    """Atomically claim one immutable job namespace for one server attempt."""

    ensure_dir(job_dir)
    commit_path = job_dir / "job_commit.json"
    if commit_path.exists():
        raise ValueError("robot_eval_job_id_already_committed")
    claim_path = job_dir / "job_claim.json"
    server_attempt_id = f"attempt-{uuid.uuid4().hex}"
    attempt_dir = contained_path(
        job_dir / "attempts",
        server_attempt_id,
        field="robot_eval_attempt_dir",
    )
    claim = {
        "schema_version": "robot_eval_job_claim.v1",
        "generated_at": generated_at,
        "status": "claimed",
        "job_id": job_id,
        "server_attempt_id": server_attempt_id,
        "request_fingerprint": request_fingerprint,
        "attempt_dir": str(attempt_dir),
        "claim_is_immutable": True,
        "final_commit_required": True,
    }
    encoded = (json.dumps(claim, sort_keys=True, indent=2) + "\n").encode("utf-8")
    # Publish the claim atomically: write the full payload to a private temp file
    # first, then hard-link it to the claim path. os.link fails with EEXIST for
    # every claimant but one, and any claimant that loses the race reads a claim
    # file that already carries its complete content — creating the final path
    # with O_EXCL and writing afterwards let a concurrent loser read an empty
    # claim and misreport a same-fingerprint claim as a fingerprint mismatch.
    temp_path = job_dir / f".job_claim.{server_attempt_id}.tmp"
    descriptor = os.open(
        temp_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temp_path, claim_path)
    except FileExistsError as exc:
        os.unlink(temp_path)
        existing = _read_optional_mapping(claim_path)
        if existing.get("request_fingerprint") != request_fingerprint:
            raise ValueError("robot_eval_job_id_request_fingerprint_mismatch") from exc
        raise ValueError("robot_eval_job_id_already_claimed") from exc
    os.unlink(temp_path)
    directory_descriptor = os.open(job_dir, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        attempt_dir / "attempt_claim.json",
        {
            **claim,
            "job_claim_path": str(claim_path),
        },
    )
    return {**claim, "job_claim_path": str(claim_path)}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_job_request(job_request: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(job_request, Mapping):
        payload = job_request
    else:
        payload = read_json_any(Path(job_request))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected job request JSON object at {job_request}")
    if payload.get("queue_contract") == ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT and isinstance(
        payload.get("job_request"), Mapping
    ):
        return dict(payload["job_request"])
    return dict(payload)


ACTUAL_OUTCOME_REQUEST_KEYS = (
    "actual_outcomes",
    "actualOutcomes",
    "real_world_outcomes",
    "realWorldOutcomes",
    "deployment_outcomes",
    "deploymentOutcomes",
    "actual_outcome_manifest_uri",
    "actualOutcomeManifestUri",
    "deployment_outcome_manifest_uri",
    "deploymentOutcomeManifestUri",
)


def _request_without_actual_outcomes(request: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in dict(request).items() if key not in ACTUAL_OUTCOME_REQUEST_KEYS
    }


def _build_real_world_validation_followup_request_queue(
    *,
    capture_root: Path,
    pipeline_dir: Path,
    job_dir: Path,
    parent_job_id: str,
    request: Mapping[str, Any],
    followup_plan: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    queue_dir = pipeline_dir / "robot_eval_job_requests" / "followup_drafts" / parent_job_id
    ensure_dir(queue_dir)
    followup_actions = [
        dict(action)
        for action in followup_plan.get("follow_up_actions", []) or []
        if isinstance(action, Mapping) and action.get("action_type") == "rerun_scenario_eval"
    ]
    queued_requests: List[Dict[str, Any]] = []
    queued_request_paths: List[str] = []
    base_request = _request_without_actual_outcomes(request)
    for index, action in enumerate(followup_actions, start=1):
        task_id = _string(action.get("task_id"))
        scenario_id = _string(action.get("scenario_id"))
        run_id = _string(action.get("scenario_eval_run_id"))
        variation_instance_id = _string(action.get("scenario_variation_instance_id"))
        variation_name = _string(action.get("variation_name"))
        action_id = _string(action.get("action_id")) or f"followup_action_{index:04d}"
        followup_job_id = f"{parent_job_id}-followup-{index:04d}"
        requested_eval_run = {
            "scenario_eval_run_id": run_id,
            "scenario_variation_instance_id": variation_instance_id,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "variation_name": variation_name,
            "source_followup_action_id": action_id,
        }
        queued_request = {
            **base_request,
            "schema_version": JOB_REQUEST_SCHEMA_VERSION,
            "job_id": followup_job_id,
            "parent_job_id": parent_job_id,
            "capture_root": str(capture_root),
            "operation": _string(base_request.get("operation") or "evaluate_only"),
            "requested_tasks": [
                {
                    "task_id": task_id,
                    "scenario_ids": [scenario_id] if scenario_id else [],
                }
            ],
            "requested_scenario_eval_runs": [requested_eval_run],
            "source_followup_action_id": action_id,
            "source_followup_plan_path": str(
                (job_dir / "real_world_validation_followup_plan.json").resolve()
            ),
            "followup_depth": int(_number(base_request.get("followup_depth"), 0.0) or 0) + 1,
            "actual_outcome_inputs_required_after_rerun": True,
            "source": {
                **_mapping(base_request.get("source")),
                "real_world_validation_followup": {
                    "parent_job_id": parent_job_id,
                    "source_followup_action_id": action_id,
                    "source_plan_path": "real_world_validation_followup_plan.json",
                    "claim_boundary": "followup_request_is_rerun_input_not_rank_fidelity_proof",
                },
            },
            "external_input_sources": {
                **_mapping(base_request.get("external_input_sources")),
                "real_world_validation_followup_plan": str(
                    (job_dir / "real_world_validation_followup_plan.json").resolve()
                ),
            },
            "claim_boundary": {
                **_mapping(base_request.get("claim_boundary")),
                "rank_fidelity_result_proven": False,
                "public_claim_upgrade_allowed": False,
                "followup_request_is_not_deployment_outcome_proof": True,
            },
        }
        request_path = queue_dir / f"{followup_job_id}.json"
        write_json(request_path, queued_request)
        queued_requests.append(queued_request)
        queued_request_paths.append(str(request_path.resolve()))

    status = (
        "ready_for_inbox_processing"
        if queued_requests
        else "no_followup_requests_queued"
        if followup_plan
        else "blocked_missing_followup_plan"
    )
    queue = {
        "schema_version": REAL_WORLD_VALIDATION_FOLLOWUP_REQUEST_QUEUE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "parent_job_id": parent_job_id,
        "capture_root": str(capture_root),
        "inbox_dir": str(queue_dir.resolve()),
        "queued_request_count": len(queued_requests),
        "queued_request_paths": queued_request_paths,
        "queued_requests": queued_requests,
        "source_artifacts": {
            "real_world_validation_followup_plan": "real_world_validation_followup_plan.json",
        },
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "artifact_purpose": "real_world_validation_followup_job_request_queue",
            "queue_is_draft_input_for_next_control_plane_pass": True,
            "queued_requests_do_not_prove_rerun_execution": True,
        },
    }
    _write_job_json(job_dir, "real_world_validation_followup_request_queue.json", queue)
    return queue


def _policy_package_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    policy_package = _mapping(payload.get("policy_package") or payload.get("policyPackage"))
    merged = dict(policy_package)
    direct = {
        modality: _modality_payload(payload, modality)
        for modality in POLICY_MODALITY_ORDER
        if _modality_payload(payload, modality)
    }
    for modality, value in direct.items():
        merged.setdefault(modality, value)
    for modality, value in default_test_policy_package_from_request(payload).items():
        merged.setdefault(modality, value)
    return merged


def _load_staged_policy_package(*, capture_root: Path, job_id: str) -> Dict[str, Any]:
    path = capture_root / "pipeline" / "robot_eval_inputs" / job_id / "policy_package.json"
    payload = _read_optional_mapping(path)
    if not payload:
        return {}
    staged_job_id = _string(payload.get("job_id") or payload.get("jobId"))
    if staged_job_id and staged_job_id != job_id:
        return {}
    package = _policy_package_from_payload(payload)
    if not package:
        return {}
    return {
        "policy_package": package,
        "source_path": str(path),
        "schema_version": payload.get("schema_version"),
    }


def _apply_staged_policy_package(
    *,
    request: Mapping[str, Any],
    capture_root: Path,
    job_id: str,
) -> Dict[str, Any]:
    updated = dict(request)
    staged = _load_staged_policy_package(capture_root=capture_root, job_id=job_id)
    if not staged:
        return updated
    existing = _mapping(updated.get("policy_package") or updated.get("policyPackage"))
    merged = dict(existing)
    for modality in POLICY_MODALITY_ORDER:
        if _modality_payload(merged, modality):
            continue
        payload = _modality_payload(staged["policy_package"], modality)
        if payload:
            merged[modality] = payload
    updated["policy_package"] = merged
    updated.setdefault("external_input_sources", {})
    if isinstance(updated["external_input_sources"], Mapping):
        updated["external_input_sources"] = {
            **dict(updated["external_input_sources"]),
            "staged_policy_package": staged["source_path"],
        }
    return updated


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
        return "not_selected", []
    if modality == "policy_api_endpoint":
        endpoint = _string(_field(payload, "endpoint_url", "endpointUrl", "url"))
        parsed_endpoint = urlparse(endpoint)
        if (
            parsed_endpoint.scheme != "https"
            or not parsed_endpoint.hostname
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or parsed_endpoint.fragment
        ):
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
        if not _string(_field(payload, "rights_privacy_attestation", "rightsPrivacyAttestation")):
            missing.append("policy_package.teleop_demo.rights_privacy_attestation")
    elif modality == "sim_controller_plugin":
        if not _string(_field(payload, "simulator_framework", "simulatorFramework")):
            missing.append("policy_package.sim_controller_plugin.simulator_framework")
        if not _string(_field(payload, "plugin_uri", "pluginUri")):
            missing.append("policy_package.sim_controller_plugin.plugin_uri")
    if missing:
        status = "blocked"
    return status, missing


def _policy_adapter_smoke_contract(modality: str) -> Dict[str, Any]:
    smoke_runners = {
        "policy_api_endpoint": "http_policy_api_observation_probe",
        "docker_container": "docker_run_observation_manifest_probe",
        "recorded_action_trace": "recorded_action_trace_replay_probe",
        "high_level_skill_trace": "high_level_skill_trace_replay_probe",
        "teleop_demo": "teleop_demo_replay_probe",
        "sim_controller_plugin": "sim_controller_plugin_probe",
    }
    return {
        "schema_version": "policy_adapter_smoke_contract.v1",
        "modality": modality,
        "smoke_runner": smoke_runners.get(modality, f"{modality}_probe"),
        "observation_manifest_input": "robot_pov_observation_manifest.json",
        "scenario_eval_matrix_input": "scenario_eval_matrix.json",
        "observation_schema_id": POLICY_OBSERVATION_SCHEMA_ID,
        "action_schema_id": POLICY_ACTION_SCHEMA_ID,
        "observation_schema_ref": POLICY_OBSERVATION_SCHEMA_REF,
        "action_schema_ref": POLICY_ACTION_SCHEMA_REF,
        "required_observation_fields": list(POLICY_REQUIRED_OBSERVATION_FIELDS),
        "required_action_output_fields": list(POLICY_REQUIRED_ACTION_OUTPUT_FIELDS),
        "required_attempt_fields": [
            "scenario_eval_run_id",
            "scenario_variation_instance_id",
            "task_id",
            "scenario_id",
            "success",
            "metrics",
            "failure_mode_ids",
        ],
        "required_artifact_outputs": [
            "normalized_attempt_trace.json",
            "policy_execution_trace.json",
            "prediction_ledger.json",
        ],
        "smoke_output_acceptance": {
            "normalized_attempt_trace_required": True,
            "failure_labels_required": True,
            "scenario_variation_exact_keys_required": True,
        },
        "reproducible_replay_contract": {
            "scenario_eval_run_id_exact_coverage_required": True,
            "scenario_variation_instance_id_exact_coverage_required": True,
            "runtime_spawn_goal_variation_mutation_allowed": False,
            "policy_outputs_must_be_replayable_from_manifest_inputs": True,
        },
        "proof_boundary": dict(CLAIM_BOUNDARY),
    }


def _policy_interface_contract(
    *,
    modality: str,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    observation_schema_ref = (
        _string(_field(payload, "observation_schema_ref", "observationSchemaRef"))
        or POLICY_OBSERVATION_SCHEMA_REF
    )
    action_schema_ref = (
        _string(_field(payload, "action_schema_ref", "actionSchemaRef"))
        or POLICY_ACTION_SCHEMA_REF
    )
    container_runtime = None
    if modality == "docker_container":
        image_ref = _string(_field(payload, "image_ref", "imageRef"))
        digest = _string(_field(payload, "digest", "digestChecksum"))
        tag = image_ref.rsplit(":", 1)[1] if ":" in image_ref.rsplit("/", 1)[-1] else ""
        container_runtime = {
            "image_ref": image_ref or None,
            "digest": digest or None,
            "digest_required": True,
            "image_ref_tag": tag or None,
            "image_ref_uses_latest_tag": tag == "latest",
            "image_ref_has_explicit_tag": bool(tag),
            "runtime_image_pinned_by_digest": digest.startswith("sha256:"),
            "versioned_runtime_image_required": True,
            "versioned_runtime_image_proven": bool(
                digest.startswith("sha256:") and image_ref and tag != "latest"
            ),
        }
    return {
        "schema_version": "robot_team_policy_interface_contract.v1",
        "modality": modality,
        "observation_schema": {
            "schema_id": POLICY_OBSERVATION_SCHEMA_ID,
            "schema_ref": observation_schema_ref,
            "source": "owner_supplied"
            if observation_schema_ref != POLICY_OBSERVATION_SCHEMA_REF
            else "blueprint_default",
            "required_fields": list(POLICY_REQUIRED_OBSERVATION_FIELDS),
        },
        "action_schema": {
            "schema_id": POLICY_ACTION_SCHEMA_ID,
            "schema_ref": action_schema_ref,
            "source": "owner_supplied"
            if action_schema_ref != POLICY_ACTION_SCHEMA_REF
            else "blueprint_default",
            "required_fields": list(POLICY_REQUIRED_ACTION_OUTPUT_FIELDS),
        },
        "runtime_inputs": {
            "observation_manifest": "robot_pov_observation_manifest.json",
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "policy_package_manifest": "policy_package_manifest.json",
        },
        "policy_adapter_pack": {
            "schema_version": "robot_team_policy_adapter_pack_contract.v1",
            "adapter_pack_mode": modality,
            "customer_supplied_policy_supported": True,
            "launch_review_without_execution_supported": True,
            "execution_claim_requires_policy_execution_manifest": True,
            "same_observation_action_contract_for_all_modes": True,
            "supported_modes": list(POLICY_MODALITY_ORDER),
        },
        "reproducible_replay": {
            "exact_scenario_eval_run_id_coverage_required": True,
            "exact_scenario_variation_instance_id_coverage_required": True,
            "normalized_attempt_trace_output_required": True,
            "failure_labels_output_required": True,
            "checksums_or_digest_required_for_external_artifacts": True,
            "runtime_spawn_goal_variation_mutation_allowed": False,
        },
        "container_runtime": container_runtime,
        "claim_boundary": (
            "Policy interface contract defines adapter IO and replay requirements; "
            "it is not proof that a robot-team policy executed or is safe."
        ),
    }


def _policy_package_manifest(
    *,
    request: Mapping[str, Any],
    generated_at: str,
) -> tuple[Dict[str, Any], List[str], List[str]]:
    policy_package = _policy_package_from_payload(request)
    modalities: Dict[str, Dict[str, Any]] = {}
    missing_inputs: List[str] = []
    missing_statuses: List[str] = []
    selected_modalities: List[str] = []
    for modality in POLICY_MODALITY_ORDER:
        payload = _modality_payload(policy_package, modality)
        status, missing = _validate_policy_modality(modality=modality, payload=payload)
        selected = bool(payload)
        if selected:
            selected_modalities.append(modality)
        if missing:
            missing_inputs.extend(missing)
            missing_statuses.append(POLICY_MODALITY_STATUSES[modality])
        modalities[modality] = {
            "status": status,
            "selected": selected,
            "missing_inputs": missing,
            "missing_evidence_status": (POLICY_MODALITY_STATUSES[modality] if missing else None),
            "reference": dict(payload),
            "download_performed": False,
            "owner_system_review_required": status not in {"blocked", "not_selected"},
            "interface_contract": _policy_interface_contract(
                modality=modality,
                payload=payload,
            ),
            "adapter_smoke_contract": _policy_adapter_smoke_contract(modality),
            "claim_boundary": (
                "reference_present_only_not_policy_execution_or_rank_fidelity_proof"
            ),
        }
    if not selected_modalities:
        missing_inputs.append("policy_package.one_supported_modality")
        missing_statuses.append("needs_robot_team_test_modality")
    manifest = {
        "schema_version": POLICY_PACKAGE_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked" if missing_inputs else "review_required",
        "selected_modalities": selected_modalities,
        "modalities": modalities,
        "missing_inputs": missing_inputs,
        "missing_evidence_statuses": missing_statuses,
        "interface_contract": {
            "schema_version": "robot_team_policy_interface_contract_summary.v1",
            "observation_schema_id": POLICY_OBSERVATION_SCHEMA_ID,
            "action_schema_id": POLICY_ACTION_SCHEMA_ID,
            "observation_schema_ref": POLICY_OBSERVATION_SCHEMA_REF,
            "action_schema_ref": POLICY_ACTION_SCHEMA_REF,
            "required_observation_fields": list(POLICY_REQUIRED_OBSERVATION_FIELDS),
            "required_action_output_fields": list(POLICY_REQUIRED_ACTION_OUTPUT_FIELDS),
            "reproducible_replay_required": True,
            "runtime_spawn_goal_variation_mutation_allowed": False,
        },
        "policy_adapter_pack_contract": {
            "schema_version": "robot_team_policy_adapter_pack_contract.v1",
            "same_observation_action_contract_for_all_modes": True,
            "supported_modalities": list(POLICY_MODALITY_ORDER),
            "selected_modalities": selected_modalities,
            "reviewable_without_execution": bool(selected_modalities and not missing_inputs),
            "execution_claim_requires_policy_execution_manifest": True,
            "provider_worker_http_workers_supported": True,
            "customer_owned_endpoint_or_container_launch_reviewable": True,
            "claim_boundary": (
                "Adapter pack review validates references and IO contracts only; "
                "policy execution proof must come from policy_execution_manifest.json."
            ),
        },
        "downloads_performed": False,
        "policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
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
    rights = _mapping(
        _mapping(site_card.get("provenance_rights_review_status")).get("rights_privacy")
    )
    return bool(rights.get("blocked"))


def _missing_robot_eval_inputs(pipeline_dir: Path) -> List[str]:
    return [
        key
        for key, relative_path in REQUIRED_ROBOT_EVAL_INPUTS.items()
        if not (pipeline_dir / relative_path).is_file()
    ]


def _empty_robot_eval_card_inputs(pipeline_dir: Path) -> List[str]:
    empty: List[str] = []
    card_inputs = {
        "robot_eval_task_cards_empty": "robot_eval_dataset/task_cards.json",
        "robot_eval_scenario_cards_empty": "robot_eval_dataset/scenario_cards.json",
    }
    for key, relative_path in card_inputs.items():
        path = pipeline_dir / relative_path
        if not path.is_file():
            continue
        payload = _read_optional_mapping(path)
        cards = payload.get("cards")
        count = _number(payload.get("task_card_count") or payload.get("scenario_card_count"))
        if count == 0 or (isinstance(cards, list) and not cards):
            empty.append(key)
    return empty


def _ensure_robot_eval_cards(*, capture_root: Path, pipeline_dir: Path) -> List[str]:
    missing = _missing_robot_eval_inputs(pipeline_dir)
    empty = _empty_robot_eval_card_inputs(pipeline_dir)
    if missing or empty:
        build_real_site_robot_eval_dataset(capture_root=capture_root)
    return [*_missing_robot_eval_inputs(pipeline_dir), *_empty_robot_eval_card_inputs(pipeline_dir)]


def _card_rows(path: Path) -> List[Dict[str, Any]]:
    payload = _read_optional_mapping(path)
    cards = payload.get("cards")
    if not isinstance(cards, Sequence) or isinstance(cards, (str, bytes, bytearray)):
        return []
    return [dict(card) for card in cards if isinstance(card, Mapping)]


def _default_requested_tasks_from_cards(pipeline_dir: Path) -> List[Dict[str, Any]]:
    task_cards = _card_rows(pipeline_dir / "robot_eval_dataset" / "task_cards.json")
    scenario_cards = _card_rows(pipeline_dir / "robot_eval_dataset" / "scenario_cards.json")
    scenarios_by_task: Dict[str, List[str]] = {}
    for scenario in scenario_cards:
        task_id = _string(scenario.get("task_id") or scenario.get("taskId"))
        scenario_id = _string(scenario.get("scenario_id") or scenario.get("scenarioId"))
        if not task_id or not scenario_id:
            continue
        scenarios_by_task.setdefault(task_id, [])
        if scenario_id not in scenarios_by_task[task_id]:
            scenarios_by_task[task_id].append(scenario_id)

    requested: List[Dict[str, Any]] = []
    for task in task_cards:
        task_id = _string(task.get("task_id") or task.get("taskId"))
        if not task_id:
            continue
        requested.append(
            {
                "task_id": task_id,
                "scenario_ids": scenarios_by_task.get(task_id, []),
                "source": "robot_eval_dataset/task_cards.json",
            }
        )
    return requested


def _default_robot_profile_from_cards(pipeline_dir: Path) -> Dict[str, Any]:
    scenario_cards = _card_rows(pipeline_dir / "robot_eval_dataset" / "scenario_cards.json")
    profile_id = ""
    for scenario in scenario_cards:
        profile_id = _string(
            scenario.get("robot_profile_id")
            or scenario.get("robotProfileId")
            or scenario.get("robot_profile")
        )
        if profile_id:
            break
    if not profile_id:
        profile_id = "unitree_g1"
    return {
        "robot_profile_id": profile_id,
        "embodiment": "humanoid" if profile_id == "unitree_g1" else "mobile_robot",
        "sensors": ["rgb", "depth", "proprioception"],
        "source": "robot_eval_dataset/scenario_cards.json",
    }


def _default_customer_from_request(request: Mapping[str, Any], *, job_id: str) -> Dict[str, Any]:
    site_package = _mapping(request.get("site_package") or request.get("sitePackage"))
    buyer_request_id = _string(
        request.get("buyer_request_id")
        or request.get("buyerRequestId")
        or site_package.get("buyer_request_id")
        or site_package.get("buyerRequestId")
    )
    raw_id = buyer_request_id or job_id
    customer_id = "robot-team-beta-" + sha256(raw_id.encode("utf-8")).hexdigest()[:12]
    return {
        "id": customer_id,
        "name": "Robot Team Beta Reference",
        "source": "pipeline_beta_request_enrichment",
        "buyer_request_id": buyer_request_id or None,
    }


def _default_reference_policy_package() -> Dict[str, Any]:
    return {
        "high_level_skill_trace": {
            "policy_id": "blueprint-beta-reference-policy",
            "policy_kind": "walk_to_target",
            "skill_taxonomy_version": "blueprint-navigation-reference-v1",
            "ordered_skill_sequence": [
                "localize_in_capture_frame",
                "plan_route_to_target",
                "walk_to_target",
                "stop_at_goal",
            ],
            "source": "pipeline_beta_request_enrichment",
            "reference_only": True,
            "robot_team_policy_execution_proven": False,
        }
    }


def _default_policy_comparison_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "policy_id": "blueprint_default_walk_to_target_smoke_policy",
            "display_name": "Blueprint default walk-to-target smoke policy",
            "candidate_role": "baseline_reference",
            "source": "pipeline_beta_reference_policy_comparison",
            "reference_only": True,
            "candidate_behavior_distinctness_proven": False,
            "robot_team_policy_execution_proven": False,
        },
        {
            "policy_id": "blueprint_conservative_clearance_walk_to_target_policy",
            "display_name": "Blueprint conservative clearance walk-to-target policy",
            "candidate_role": "conservative_reference",
            "source": "pipeline_beta_reference_policy_comparison",
            "reference_only": True,
            "candidate_behavior_distinctness_proven": False,
            "robot_team_policy_execution_proven": False,
        },
    ]


def _request_policy_candidates(request: Mapping[str, Any]) -> List[Dict[str, Any]]:
    execution_request = _mapping(
        request.get("execution_request") or request.get("executionRequest")
    )
    wam_request = _mapping(
        request.get("wam_evaluation")
        or request.get("wamEvaluation")
        or execution_request.get("wam_evaluation")
        or execution_request.get("wamEvaluation")
    )
    raw = (
        request.get("policy_candidates")
        or request.get("policyCandidates")
        or request.get("policies")
        or request.get("checkpoints")
        or execution_request.get("policy_candidates")
        or execution_request.get("policyCandidates")
        or wam_request.get("policy_candidates")
        or wam_request.get("policyCandidates")
        or wam_request.get("policies")
        or wam_request.get("checkpoints")
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        payload = _mapping(item)
        policy_id = (
            _string(
                payload.get("policy_id")
                or payload.get("policyId")
                or payload.get("candidate_id")
                or payload.get("candidateId")
                or payload.get("id")
            )
            or f"policy_candidate_{index:02d}"
        )
        if policy_id in seen:
            continue
        seen.add(policy_id)
        candidates.append(
            {
                **payload,
                "policy_id": policy_id,
                "display_name": _string(payload.get("display_name") or payload.get("name"))
                or policy_id,
            }
        )
    return candidates


def _enrich_incomplete_beta_job_request(
    *,
    request: Mapping[str, Any],
    pipeline_dir: Path,
    job_id: str,
    generated_at: str,
    simulator: str = "fixture",
    evaluation_substrate: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    enriched = dict(request)
    added_fields: List[str] = []
    sources: Dict[str, str] = {}

    if not _mapping(enriched.get("customer") or enriched.get("customerProfile")):
        enriched["customer"] = _default_customer_from_request(enriched, job_id=job_id)
        added_fields.append("customer")
        sources["customer"] = "buyer_request_id_or_job_id"

    if not _mapping(enriched.get("robot_profile") or enriched.get("robotProfile")):
        enriched["robot_profile"] = _default_robot_profile_from_cards(pipeline_dir)
        added_fields.append("robot_profile")
        sources["robot_profile"] = "robot_eval_dataset/scenario_cards.json"

    if not _string_list(enriched.get("requested_tasks") or enriched.get("requestedTasks")):
        requested_tasks = _default_requested_tasks_from_cards(pipeline_dir)
        if requested_tasks:
            enriched["requested_tasks"] = requested_tasks
            added_fields.append("requested_tasks")
            sources["requested_tasks"] = "robot_eval_dataset/task_cards.json"

    if not _mapping(
        enriched.get("policy_package") or enriched.get("policyPackage")
    ) and not default_test_policy_package_from_request(enriched):
        enriched["policy_package"] = _default_reference_policy_package()
        added_fields.append("policy_package")
        sources["policy_package"] = "pipeline_beta_reference_policy"

    if (
        simulator != "fixture" or _string(evaluation_substrate) in WAM_EVALUATION_SUBSTRATES
    ) and not _request_policy_candidates(enriched):
        enriched["policy_candidates"] = _default_policy_comparison_candidates()
        enriched["policy_comparison_mode"] = True
        added_fields.append("policy_candidates")
        added_fields.append("policy_comparison_mode")
        sources["policy_candidates"] = "pipeline_beta_reference_policy_comparison"
        sources["policy_comparison_mode"] = "pipeline_beta_reference_policy_comparison"

    manifest = {
        "schema_version": JOB_REQUEST_ENRICHMENT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "enriched" if added_fields else "not_required",
        "job_id": job_id,
        "added_fields": added_fields,
        "sources": sources,
        "source_request_preserved_path": "job_request_source.json",
        "enriched_request_path": "job_request.json",
        "claim_boundary": (
            "Fills missing beta orchestration inputs from capture-grounded dataset cards "
            "and a reference policy package. This is not production WebApp proof, "
            "robot-team policy execution proof, physical robot proof, or deployment proof."
        ),
    }
    return enriched, manifest


def _job_validation(
    *,
    request: Mapping[str, Any],
    policy_missing_inputs: Sequence[str],
    policy_missing_statuses: Sequence[str],
    missing_robot_eval_inputs: Sequence[str],
    generated_at: str,
    pipeline_dir: Path,
    benchmark_protocol_status: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    benchmark_status = benchmark_protocol_status or {}
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
    if benchmark_status.get("status") == "blocked":
        blockers.append("benchmark_protocol_blocked")
        missing_inputs.extend(_string_list(benchmark_status.get("blockers")))
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
        "benchmark_protocol_status": benchmark_status.get("status"),
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
            "operator_mode": "not_requested",
            "agent_authority": "not_requested",
            "proof_booleans_mutable_by_agent": False,
            "decisions": [],
            "diagnostics": [],
            "operator_ledger": {
                "generated_at": generated_at,
                "operator_mode": "not_requested",
                "decisions": [],
                "tool_call_summaries": [],
                "commands_chosen": [],
                "refusals": [],
                "blockers": [],
                "proof_effect": proof_effect(
                    deterministic_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"]
                ),
            },
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    plan = adapter.build_plan(plan_context=plan_context)
    plan.setdefault("generated_at", generated_at)
    plan.setdefault("claim_boundary", dict(CLAIM_BOUNDARY))
    return dict(plan)


def _execution_worker_profile(simulator: str) -> Dict[str, Any]:
    if simulator in {"isaac_sim", "isaac_lab_arena"}:
        return {
            "worker_image_family": (
                "isaac-arena-eval-worker" if simulator == "isaac_lab_arena" else "isaac-eval-worker"
            ),
            "dockerfile_path": "deploy/docker/robot_eval_worker/isaac/Dockerfile",
            "entrypoint": "blueprint-run-robot-eval-worker",
            "preferred_gpu_class": "rtx_rt_core_24gb_or_larger",
            "disallowed_gpu_classes": ["a100", "h100"],
            "persistent_cache_targets": [
                "docker_layers",
                "isaac_kit_cache",
                "isaac_extension_cache",
                "robot_usd_assets",
                "converted_scene_assets",
                "policy_bundles",
            ],
            "cold_start_sensitive": True,
        }
    if simulator == "mujoco":
        return {
            "worker_image_family": "mujoco-eval-worker",
            "dockerfile_path": "deploy/docker/robot_eval_worker/mujoco/Dockerfile",
            "entrypoint": "blueprint-run-robot-eval-worker",
            "preferred_gpu_class": "cpu_or_low_cost_gpu_when_rendering",
            "disallowed_gpu_classes": [
                "a100",
                "h100",
                "rtx_a6000_unless_render_or_latency_policy_requires_it",
                "rtx_6000_ada_unless_render_or_latency_policy_requires_it",
            ],
            "persistent_cache_targets": [
                "docker_layers",
                "mjcf_assets",
                "policy_bundles",
                "converted_scenes",
                "worker_deps",
            ],
            "cold_start_sensitive": False,
        }
    if simulator == "pybullet":
        return {
            "worker_image_family": "pybullet-eval-worker",
            "dockerfile_path": None,
            "entrypoint": "blueprint-run-robot-eval-worker",
            "preferred_gpu_class": "cpu_or_low_cost_gpu_when_rendering",
            "persistent_cache_targets": ["docker_layers", "urdf_assets", "policy_bundles"],
            "cold_start_sensitive": False,
        }
    if simulator == "newton":
        return {
            "worker_image_family": "newton-eval-worker",
            "dockerfile_path": None,
            "entrypoint": "blueprint-run-robot-eval-worker",
            "preferred_gpu_class": "gpu_physics_worker",
            "persistent_cache_targets": ["docker_layers", "scene_assets", "policy_bundles"],
            "cold_start_sensitive": True,
        }
    return {
        "worker_image_family": "repo-local-fixture",
        "dockerfile_path": None,
        "entrypoint": "blueprint-run-robot-eval-worker",
        "preferred_gpu_class": "none",
        "persistent_cache_targets": [],
        "cold_start_sensitive": False,
    }


def _required_scheduler_artifact_paths(
    automation_dir: Path,
    required_artifacts: Sequence[str],
) -> Dict[str, Path]:
    known = {
        "scene_asset_inventory": automation_dir / "scene_asset_inventory.json",
        "scene_asset_dependency_audit": automation_dir / "scene_asset_dependency_audit.json",
        "cpu_preflight_scorecard": automation_dir / "cpu_preflight_scorecard.json",
        "episode_spec_manifest": automation_dir / "episode_spec_manifest.json",
        "gpu_handoff_packet": automation_dir / "gpu_handoff_packet.json",
    }
    return {artifact: known[artifact] for artifact in required_artifacts if artifact in known}


def _selected_simulator_cpu_preflight_gate(
    *,
    simulator: str,
    pipeline_dir: Path,
    owner_gpu_cpu_preflight: Mapping[str, Any],
) -> Dict[str, Any]:
    owner_ready = bool(owner_gpu_cpu_preflight.get("ready_for_owner_gpu_preflight"))
    owner_hard_blockers = _string_list(owner_gpu_cpu_preflight.get("hard_preflight_blockers"))
    cpu_smoke_path = pipeline_dir / "simulation_automation" / "cpu_simulator_preflight_manifest.json"
    cpu_smoke = _read_optional_mapping(cpu_smoke_path)
    completed_backends = _string_list(cpu_smoke.get("local_cpu_smoke_completed_backends"))
    blocked_backends = _string_list(cpu_smoke.get("blocked_backends"))
    failed_backends = _string_list(cpu_smoke.get("failed_backends"))
    backend_results = _mapping(cpu_smoke.get("backend_results"))
    selected_result = _mapping(backend_results.get(simulator))
    selected_backend_cpu_smoke_complete = bool(
        simulator in CPU_BACKENDS
        and simulator in completed_backends
        and simulator not in blocked_backends
        and simulator not in failed_backends
        and (
            not selected_result
            or selected_result.get("status") == "completed_local_cpu_smoke"
        )
    )
    selected_backend_cpu_smoke_blockers = _string_list(selected_result.get("blockers"))
    ready_for_selected_provider_preflight = bool(
        owner_ready or selected_backend_cpu_smoke_complete
    )
    source_artifact = (
        "../simulation_automation/cpu_simulator_preflight_manifest.json"
        if selected_backend_cpu_smoke_complete and not owner_ready
        else "../simulation_automation/cpu_preflight_manifest.json"
    )
    notes: List[str] = []
    if selected_backend_cpu_smoke_complete and not owner_ready:
        notes.extend(
            [
                "selected_cpu_simulator_smoke_passed_for_provider_package",
                "broad_owner_gpu_preflight_blockers_remain_for_isaac_or_owner_gpu_claims",
            ]
        )
    return {
        "selected_simulator": simulator,
        "selected_simulator_is_cpu_smoke_backend": simulator in CPU_BACKENDS,
        "ready_for_owner_gpu_preflight": owner_ready,
        "ready_for_selected_simulator_provider_preflight": ready_for_selected_provider_preflight,
        "source_artifact": source_artifact,
        "owner_gpu_cpu_preflight_status": owner_gpu_cpu_preflight.get("status"),
        "owner_gpu_hard_preflight_blockers": owner_hard_blockers,
        "local_cpu_smoke_manifest_present": bool(cpu_smoke),
        "local_cpu_smoke_status": cpu_smoke.get("status"),
        "local_cpu_smoke_completed_backends": completed_backends,
        "local_cpu_smoke_blocked_backends": blocked_backends,
        "local_cpu_smoke_failed_backends": failed_backends,
        "selected_backend_cpu_smoke_complete": selected_backend_cpu_smoke_complete,
        "selected_backend_cpu_smoke_blockers": selected_backend_cpu_smoke_blockers,
        "claim_boundary": {
            "selected_cpu_smoke_does_not_clear_owner_gpu_preflight": True,
            "selected_cpu_smoke_does_not_clear_isaac_or_digital_twin_gate": True,
            "selected_cpu_smoke_does_not_prove_remote_provider_execution": True,
            "selected_cpu_smoke_does_not_approve_provider_spend": True,
        },
        "notes": notes,
    }


def _build_scheduler_decision(
    *,
    request: Mapping[str, Any],
    job_id: str,
    provisioner: str,
    simulator: str,
    pipeline_dir: Path,
    cpu_preflight: Mapping[str, Any],
    budget_usd: float | None,
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    execution_request = _mapping(request.get("execution_request"))
    queueing = _mapping(execution_request.get("queueing"))
    preflight = _mapping(execution_request.get("preflight"))
    gpu_allocation = _mapping(execution_request.get("gpu_allocation"))
    artifact_contract = _mapping(execution_request.get("artifact_contract"))
    request_budget = _mapping(request.get("budget"))
    required_artifacts = _string_list(
        preflight.get("required_artifacts")
        or [
            "scene_asset_inventory",
            "scene_asset_dependency_audit",
            "cpu_preflight_scorecard",
            "episode_spec_manifest",
            "gpu_handoff_packet",
        ]
    )
    automation_dir = pipeline_dir / "simulation_automation"
    required_paths = _required_scheduler_artifact_paths(automation_dir, required_artifacts)
    artifact_status = {
        artifact: {
            "path": _relative_to(pipeline_dir, path),
            "present": path.is_file(),
        }
        for artifact, path in required_paths.items()
    }
    execution_request_present = bool(execution_request)
    requested_cpu_preflight_required = bool(
        preflight.get("cpu_preflight_required_before_gpu") is not False
    )
    gpu_or_external_simulator = simulator not in {"fixture", "mujoco", "pybullet"}
    external_provisioner = provisioner != "fixture_local"
    gpu_allocation_requested = gpu_or_external_simulator or external_provisioner
    cpu_preflight_required = requested_cpu_preflight_required or gpu_allocation_requested
    ready_for_owner_gpu_preflight = bool(cpu_preflight.get("ready_for_owner_gpu_preflight"))
    selected_simulator_cpu_preflight_gate = _selected_simulator_cpu_preflight_gate(
        simulator=simulator,
        pipeline_dir=pipeline_dir,
        owner_gpu_cpu_preflight=cpu_preflight,
    )
    ready_for_selected_simulator_provider_preflight = bool(
        selected_simulator_cpu_preflight_gate.get(
            "ready_for_selected_simulator_provider_preflight"
        )
    )
    simulator_selection_policy = resolve_simulator_selection_policy(
        request,
        selected_simulator=simulator,
    )
    allowed_backends = _string_list(simulator_selection_policy.get("allowed_backends"))
    blockers: List[str] = []
    if execution_request_present:
        if execution_request.get("webapp_role") != "queue_and_forward_only":
            blockers.append("execution_request_webapp_role_not_queue_only")
        if execution_request.get("scheduler_owner") != "BlueprintCapturePipeline":
            blockers.append("execution_request_scheduler_owner_not_pipeline")
        if allowed_backends and simulator not in allowed_backends:
            blockers.append("scheduler_selected_simulator_not_allowed_by_execution_request")
        if gpu_allocation_requested and not requested_cpu_preflight_required:
            blockers.append("execution_request_cpu_preflight_gate_disabled_for_gpu")
        if (
            cpu_preflight_required
            and gpu_allocation_requested
            and not ready_for_selected_simulator_provider_preflight
        ):
            blockers.append("scheduler_cpu_preflight_not_ready_for_gpu")
        missing_required = [
            artifact
            for artifact, status in artifact_status.items()
            if not bool(status.get("present"))
        ]
        if cpu_preflight_required and gpu_allocation_requested and missing_required:
            blockers.append("scheduler_required_preflight_artifacts_missing")
        if gpu_allocation.get("allocation_allowed_by_webapp") is not False:
            blockers.append("execution_request_webapp_gpu_allocation_boundary_missing")
        if gpu_allocation.get("gpu_spend_approved") is not False:
            blockers.append("execution_request_must_not_approve_gpu_spend")
        if artifact_contract.get("public_claim_upgrade_allowed") is not False:
            blockers.append("execution_request_public_claim_upgrade_boundary_missing")
    requested_budget = (
        budget_usd
        if budget_usd is not None
        else _number_field(request_budget, "budget_usd", "budgetUsd")
    )
    status = (
        "blocked"
        if blockers
        else "local_fixture_only"
        if simulator == "fixture" and provisioner == "fixture_local"
        else "awaiting_explicit_gpu_and_simulator_gates"
        if gpu_allocation_requested
        else "ready_for_cpu_proxy_execution_gate"
    )
    recommended_action = (
        "do_not_allocate_gpu"
        if blockers or not gpu_allocation_requested
        else "wait_for_explicit_provider_gate"
        if external_provisioner
        else "wait_for_explicit_owner_gpu_gate"
    )
    return {
        "schema_version": SCHEDULER_DECISION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": status,
        "source_execution_request_present": execution_request_present,
        "webapp_role": execution_request.get("webapp_role") or "queue_and_forward_only",
        "scheduler_owner": execution_request.get("scheduler_owner") or "BlueprintCapturePipeline",
        "queueing": {
            "mode": queueing.get("mode") or "async_job",
            "customer_response": queueing.get("customer_response") or "job_id_and_status_only",
            "web_request_must_not_wait_for_simulator": bool(
                queueing.get("web_request_must_not_wait_for_simulator") is not False
            ),
        },
        "selection": {
            "provisioner": provisioner,
            "simulator": simulator,
            "request_simulator_preference": request.get("simulator_preference")
            or request.get("simulatorPreference"),
            "recommended_simulator": simulator_selection_policy.get("recommended_backend"),
            "simulator_selection_policy_mode": simulator_selection_policy.get("mode"),
            "selected_simulator_matches_request_policy": bool(
                simulator_selection_policy.get("selected_backend_matches_recommendation")
            ),
            "request_operation": request.get("operation") or "evaluate_only",
            "worker_profile": _execution_worker_profile(simulator),
        },
        "simulator_selection_policy": simulator_selection_policy,
        "cpu_preflight_gate": {
            "required_before_gpu": cpu_preflight_required,
            "blocks_gpu_when_missing": bool(preflight.get("blocks_gpu_when_missing") is not False),
            "ready_for_owner_gpu_preflight": ready_for_owner_gpu_preflight,
            "ready_for_selected_simulator_provider_preflight": (
                ready_for_selected_simulator_provider_preflight
            ),
            "selected_simulator_cpu_preflight_gate": selected_simulator_cpu_preflight_gate,
            "status": cpu_preflight.get("status"),
            "hard_preflight_blockers": _string_list(cpu_preflight.get("hard_preflight_blockers")),
            "required_artifact_status": artifact_status,
        },
        "gpu_allocation": {
            "mode": gpu_allocation.get("mode") or "on_demand_with_optional_warm_pool",
            "allocation_owner": (
                gpu_allocation.get("allocation_owner")
                or "BlueprintCapturePipeline_or_owner_gpu_worker"
            ),
            "allocation_allowed_by_webapp": bool(
                gpu_allocation.get("allocation_allowed_by_webapp") is True
            ),
            "gpu_spend_approved_by_webapp": bool(gpu_allocation.get("gpu_spend_approved") is True),
            "requested_budget_usd": requested_budget,
            "hard_timeout_seconds": int(
                _number(gpu_allocation.get("hard_timeout_seconds"), timeout_seconds)
                or timeout_seconds
            ),
            "idle_shutdown_required": bool(
                gpu_allocation.get("idle_shutdown_required") is not False
            ),
            "persistent_cache_recommended": bool(
                gpu_allocation.get("persistent_cache_recommended") is not False
            ),
            "provider_gpu_priority_fallback_list": _provider_gpu_priority_for_simulator(
                simulator,
                gpu_allocation,
            ),
            "warm_pool_policy": _warm_pool_policy(gpu_allocation=gpu_allocation),
            "live_provider_calls_allowed_by_default": False,
            "recommended_action": recommended_action,
        },
        "artifact_contract": {
            "expected_outputs": (
                _dedupe(
                    [
                        *_string_list(artifact_contract.get("expected_outputs")),
                        *DEFAULT_STARTUP_EXPECTED_OUTPUTS,
                    ]
                )
            ),
            "simulator_execution_proven_by_webapp": bool(
                artifact_contract.get("simulator_execution_proven_by_webapp") is True
            ),
            "public_claim_upgrade_allowed": bool(
                artifact_contract.get("public_claim_upgrade_allowed") is True
            ),
        },
        "blockers": _dedupe(blockers),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _provider_credential_env_vars(provisioner: str) -> List[str]:
    if provisioner == "runpod":
        return ["RUNPOD_API_KEY"]
    if provisioner == "lambda_cloud":
        return ["LAMBDA_API_KEY"]
    if provisioner == "vast":
        return ["VAST_API_KEY"]
    if provisioner == "gcp":
        return ["GOOGLE_APPLICATION_CREDENTIALS"]
    return []


def _provider_launch_operation(provisioner: str) -> str:
    if provisioner == "runpod":
        return "enqueue_runpod_serverless_or_on_demand_worker"
    if provisioner == "lambda_cloud":
        return "launch_lambda_cloud_instance_and_run_worker"
    if provisioner == "vast":
        return "create_vast_instance_and_run_worker"
    if provisioner == "gcp":
        return "create_gcp_gpu_worker_and_run_job"
    if provisioner == "docker_local":
        return "start_local_docker_worker"
    if provisioner == "local_process":
        return "start_local_process_worker"
    return "no_provider_launch_required"


def _provider_adapter_command(provisioner: str) -> str | None:
    return {
        "runpod": "blueprint-run-runpod-provider-adapter",
        "vast": "blueprint-run-vast-provider-adapter",
        "lambda_cloud": "blueprint-run-lambda-provider-adapter",
    }.get(provisioner)


def _provider_adapter_id(provisioner: str) -> str | None:
    return {
        "runpod": "runpod_provider_adapter.v1",
        "vast": "vast_provider_adapter.v1",
        "lambda_cloud": "lambda_provider_adapter.v1",
    }.get(provisioner)


def _provider_race_runtime_readiness(
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Describe whether the customer path can execute provider failover safely.

    Robot-eval adapters expose command launchers, not in-process provider objects
    with loser teardown hooks. The customer path therefore wires serial adapter
    failover: try runnable providers in priority order, stop on the first
    successful adapter result, and leave parallel racing to render lanes that
    have provider objects with teardown-owned loser cleanup.
    """

    race_candidates = [
        _mapping(candidate)
        for candidate in candidates
        if candidate.get("race_candidate") is True
        and _string(candidate.get("provider"))
    ]
    runtime_candidates: List[Dict[str, Any]] = []
    runtime_blockers: List[str] = []
    for candidate in race_candidates:
        provider = _string(candidate.get("provider"))
        adapter_command = _string(candidate.get("adapter_command"))
        adapter_id = _string(candidate.get("adapter_id"))
        blockers: List[str] = []
        if not adapter_command or adapter_command != _provider_adapter_command(provider):
            blockers.append(f"{provider}_provider_race_adapter_command_missing")
        if not adapter_id or adapter_id != _provider_adapter_id(provider):
            blockers.append(f"{provider}_provider_race_adapter_id_missing")
        if blockers:
            runtime_blockers.extend(blockers)
        runtime_candidates.append(
            {
                "provider": provider,
                "adapter_command_present": bool(adapter_command),
                "adapter_id": adapter_id or None,
                "teardown_owned_allocation_contract_present": not blockers,
                "customer_path_race_runtime_eligible": False,
                "customer_path_serial_failover_runtime_eligible": not blockers,
                "blockers": blockers,
            }
        )
    eligible_count = sum(
        1
        for candidate in runtime_candidates
        if candidate.get("customer_path_serial_failover_runtime_eligible") is True
    )
    ready = len(race_candidates) > 1 and eligible_count > 1
    return {
        "schema_version": "robot_eval_provider_race_runtime_readiness.v1",
        "status": "serial_failover_runtime_ready"
        if ready
        else "blocked_pending_runnable_provider_adapters",
        "customer_path_provider_failover_runtime_wired": ready,
        "runtime_candidate_count": len(race_candidates),
        "runtime_eligible_candidate_count": eligible_count,
        "runtime_candidates": runtime_candidates,
        "blockers": []
        if ready
        else (
            _dedupe(runtime_blockers)
            or ["provider_race_requires_two_runtime_eligible_candidates"]
        ),
        "diagnostic_blockers": _dedupe(runtime_blockers),
        "claim_boundary": {
            "provider_race_handoff_is_not_live_runtime_failover": not ready,
            "customer_provider_failover_runtime_mode": "serial_adapter_failover"
            if ready
            else None,
            "parallel_provider_race_runtime_claimed": False,
            "serial_provider_launch_must_remain_blocked_until_runtime_wired": not ready,
            "teardown_owned_loser_cleanup_required_before_customer_runtime": True,
            "failed_adapter_must_emit_teardown_or_open_billing_risk": True,
            "fresh_job_bound_terminal_artifact_required_before_winner": True,
        },
    }


def _provider_race_contract(
    *,
    selected_provider: str,
    gpu_selection: Mapping[str, Any],
    startup_pipeline: Mapping[str, Any],
) -> Dict[str, Any]:
    managed_provider_policy = _mapping(startup_pipeline.get("managed_provider_policy"))
    marketplace_policy = _mapping(startup_pipeline.get("marketplace_policy"))
    priority = _string_list(
        startup_pipeline.get("provider_api_priority")
        or managed_provider_policy.get("provider_api_priority")
        or startup_pipeline.get("provider_priority")
        or marketplace_policy.get("provider_api_priority")
    )
    gpu_priority = _string_list(
        gpu_selection.get("provider_gpu_priority_fallback_list")
        or gpu_selection.get("provider_gpu_priority")
    )
    if selected_provider and selected_provider not in priority:
        priority.insert(0, selected_provider)
    candidates = [
        {
            "provider": provider,
            "operation": _provider_launch_operation(provider),
            "race_candidate": provider in LIVE_GPU_PROVISIONERS,
            "adapter_id": _provider_adapter_id(provider),
            "adapter_command": _provider_adapter_command(provider),
            "selected": provider == selected_provider,
        }
        for provider in priority
    ]
    race_candidates = [
        candidate for candidate in candidates if candidate.get("race_candidate") is True
    ]
    race_required = len(race_candidates) > 1
    runtime_readiness = _provider_race_runtime_readiness(race_candidates)
    selected_tier = _string(startup_pipeline.get("selected_provider_tier"))
    return {
        "schema_version": "robot_eval_provider_race_contract.v1",
        "status": "configured" if candidates else "single_provider_only",
        "provider_race_contract_ready": race_required,
        "race_required_for_customer_path": race_required,
        "customer_path_provider_failover_wired": bool(
            runtime_readiness.get("customer_path_provider_failover_runtime_wired")
        ),
        "customer_path_provider_failover_handoff_wired": race_required,
        "customer_path_provider_failover_runtime_wired": bool(
            runtime_readiness.get("customer_path_provider_failover_runtime_wired")
        ),
        "customer_path_provider_failover_runtime_status": runtime_readiness.get("status"),
        "customer_path_provider_failover_runtime_blockers": _string_list(
            runtime_readiness.get("blockers")
        ),
        "provider_race_handoff_path": "gpu_provider_race_handoff.json"
        if race_required
        else None,
        "customer_path_serial_launch_blocked_unless_override": race_required,
        "selected_provider": selected_provider or None,
        "selected_provider_tier": selected_tier or None,
        "provider_selection_owner": startup_pipeline.get("provider_selection_owner"),
        "candidate_count": len(candidates),
        "race_candidate_count": len(race_candidates),
        "gpu_model_priority_count": len(gpu_priority),
        "candidates": candidates,
        "runtime_readiness": runtime_readiness,
        "race_module": "blueprint_pipeline.provider_race",
        "launcher_contract": {
            "provider_race_launcher_available": race_required,
            "provider_race_launcher_command": PROVIDER_RACE_LAUNCHER_COMMAND
            if race_required
            else None,
            "provider_race_launcher_result_path": PROVIDER_RACE_LAUNCHER_RESULT_NAME
            if race_required
            else None,
            "launch_every_candidate_requires_prelaunch_can_launch_true": True,
            "runtime_mode": "serial_adapter_failover" if runtime_readiness.get(
                "customer_path_provider_failover_runtime_wired"
            ) is True else "blocked",
            "terminate_losers_required": True,
            "circuit_breaker_state_required": True,
            "boot_marker_required_before_winner": True,
            "teardown_owned_loser_cleanup_required": True,
            "fresh_job_bound_terminal_artifact_required_before_winner": True,
            "failed_adapter_must_emit_teardown_or_open_billing_risk": True,
            "handoff_packet_path": "gpu_provider_race_handoff.json"
            if race_required
            else None,
            "serial_provider_launch_default_allowed": not race_required,
            "serial_provider_launch_override_requires": [
                "BLUEPRINT_ALLOW_SERIAL_GPU_PROVIDER_LAUNCH=true",
                "--allow-serial-provider-launch",
            ]
            if race_required
            else [],
        },
    }


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _fleet_budget_guard_from_spend_snapshot() -> Dict[str, Any]:
    path_text = _string(
        os.getenv("BLUEPRINT_GPU_SPEND_GUARD_SNAPSHOT_PATH")
        or os.getenv("BLUEPRINT_GPU_SPEND_GUARD_REPORT")
    )
    max_age_seconds = int(
        _number(os.getenv("BLUEPRINT_GPU_SPEND_GUARD_MAX_AGE_SECONDS"), 15 * 60)
        or 15 * 60
    )
    blockers: List[str] = []
    snapshot: Dict[str, Any] = {}
    path: Path | None = Path(path_text).expanduser() if path_text else None
    if path is None:
        blockers.append("fleet_budget_spend_guard_snapshot_path_missing")
    elif not path.is_file():
        blockers.append("fleet_budget_spend_guard_snapshot_missing")
    else:
        snapshot = _read_optional_mapping(path)
        if not snapshot:
            blockers.append("fleet_budget_spend_guard_snapshot_parse_failed")

    generated_at = _parse_iso_datetime(snapshot.get("generated_at")) if snapshot else None
    age_seconds: float | None = None
    fleet_budget = _mapping(snapshot.get("fleet_budget"))
    if snapshot:
        if snapshot.get("schema_version") != "gpu_spend_guard.v1":
            blockers.append("fleet_budget_spend_guard_snapshot_schema_invalid")
        if generated_at is None:
            blockers.append("fleet_budget_spend_guard_snapshot_generated_at_invalid")
        else:
            age_seconds = (datetime.now(timezone.utc) - generated_at).total_seconds()
            if age_seconds < 0 or age_seconds > max_age_seconds:
                blockers.append("fleet_budget_spend_guard_snapshot_stale")
        if snapshot.get("reap_candidate_ids"):
            blockers.append("fleet_budget_spend_guard_snapshot_has_reap_candidates")
        if fleet_budget.get("status") != "passed":
            blockers.append("fleet_budget_not_passed")
            blockers.extend(
                f"fleet_budget:{blocker}"
                for blocker in _string_list(fleet_budget.get("blockers"))
            )

    return {
        "schema_version": "robot_eval_provider_fleet_budget_guard.v1",
        "status": "passed" if not blockers else "blocked",
        "snapshot_path": str(path) if path else None,
        "snapshot_generated_at": snapshot.get("generated_at"),
        "snapshot_age_seconds": age_seconds,
        "max_age_seconds": max_age_seconds,
        "fleet_budget": fleet_budget or None,
        "live_instance_count": snapshot.get("live_instance_count"),
        "total_burn_per_hour_usd": snapshot.get("total_burn_per_hour_usd"),
        "blockers": _dedupe(blockers),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "fleet_budget_is_cost_gate_only": True,
            "fleet_budget_is_not_provider_runtime_proof": True,
        },
    }


def _provider_prelaunch_spend_guard(
    *,
    provider: str,
    request_manifest: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    worker_launch_plan: Mapping[str, Any],
    startup_pipeline: Mapping[str, Any],
    local_sim_only_prerequisite: Mapping[str, Any],
    approval_blockers: Sequence[str],
    request_blockers: Sequence[str],
    env_allowed: bool,
    allow_gpu_provisioning: bool,
) -> Dict[str, Any]:
    external_provider = provider != "fixture_local"
    gpu_allocation = _mapping(scheduler_decision.get("gpu_allocation"))
    launch_mode = _mapping(worker_launch_plan.get("launch_mode"))
    cost_controls = _mapping(worker_launch_plan.get("cost_controls"))
    gpu_selection = _mapping(worker_launch_plan.get("gpu_selection"))
    artifact_upload = _mapping(worker_launch_plan.get("artifact_upload_contract"))
    requested_budget = (
        _number(request_manifest.get("requested_budget_usd"))
        if request_manifest.get("requested_budget_usd") is not None
        else _number(cost_controls.get("requested_budget_usd"))
        if cost_controls.get("requested_budget_usd") is not None
        else _number(gpu_allocation.get("requested_budget_usd"))
    )
    hard_timeout_seconds = int(
        _number(
            launch_mode.get("hard_timeout_seconds")
            or request_manifest.get("timeout_seconds")
            or gpu_allocation.get("hard_timeout_seconds"),
            0,
        )
        or 0
    )
    max_active_workers = int(_number(launch_mode.get("max_active_workers"), 0) or 0)
    idle_shutdown_required = bool(launch_mode.get("idle_shutdown_required"))
    watchdog_required = bool(launch_mode.get("external_watchdog_ttl_required"))
    watchdog_ttl_seconds = int(
        _number(launch_mode.get("external_watchdog_ttl_seconds"), 0) or 0
    )
    upload_before_shutdown_required = bool(
        artifact_upload.get("upload_before_shutdown_required")
        or cost_controls.get("finalizer_must_upload_artifacts_before_shutdown")
    )
    local_sim_required = bool(
        local_sim_only_prerequisite.get("required_before_provider_spend")
    )
    local_sim_ready = bool(
        not external_provider
        or not local_sim_required
        or (
            local_sim_only_prerequisite.get("status") == "passed"
            and local_sim_only_prerequisite.get("local_sim_only_evidence_clean") is True
        )
    )
    fleet_budget_guard = (
        _fleet_budget_guard_from_spend_snapshot() if external_provider else {}
    )
    blockers: List[str] = []
    if external_provider:
        if requested_budget is None:
            blockers.append("prelaunch_missing_requested_budget_usd")
        elif requested_budget <= 0:
            blockers.append("prelaunch_requested_budget_must_be_positive")
        if hard_timeout_seconds <= 0:
            blockers.append("prelaunch_missing_hard_timeout_seconds")
        if max_active_workers != 1:
            blockers.append("prelaunch_max_active_workers_must_equal_one")
        if not idle_shutdown_required:
            blockers.append("prelaunch_idle_shutdown_required")
        if not watchdog_required or watchdog_ttl_seconds <= hard_timeout_seconds:
            blockers.append("prelaunch_external_watchdog_ttl_must_exceed_hard_timeout")
        if not upload_before_shutdown_required:
            blockers.append("prelaunch_artifact_upload_before_shutdown_required")
        if not local_sim_ready:
            blockers.append("prelaunch_local_sim_only_prerequisite_not_passed")
            blockers.extend(
                f"local_sim_only_prerequisite:{blocker}"
                for blocker in _string_list(local_sim_only_prerequisite.get("blockers"))
            )
        if fleet_budget_guard.get("status") != "passed":
            blockers.append("prelaunch_fleet_budget_guard_not_passed")
            blockers.extend(
                f"fleet_budget_guard:{blocker}"
                for blocker in _string_list(fleet_budget_guard.get("blockers"))
            )
        if approval_blockers:
            blockers.extend(f"approval:{blocker}" for blocker in approval_blockers)
        if request_blockers:
            blockers.extend(f"launch_request:{blocker}" for blocker in request_blockers)
    can_launch = bool(external_provider and not blockers)
    return {
        "schema_version": PROVIDER_PRELAUNCH_SPEND_GUARD_SCHEMA_VERSION,
        "status": "passed" if can_launch else ("not_required" if not external_provider else "blocked"),
        "required_before_provider_launch": bool(external_provider),
        "can_launch": can_launch,
        "provider": provider,
        "requested_budget_usd": requested_budget,
        "max_billable_gpu_seconds": hard_timeout_seconds,
        "max_active_workers": max_active_workers,
        "checks": {
            "env_BLUEPRINT_ALLOW_GPU_PROVISIONING_present": env_allowed,
            "cli_allow_gpu_provisioning_present": bool(allow_gpu_provisioning),
            "requested_budget_declared": requested_budget is not None,
            "requested_budget_positive": requested_budget is not None and requested_budget > 0,
            "hard_timeout_declared": hard_timeout_seconds > 0,
            "max_active_workers_one": max_active_workers == 1,
            "idle_shutdown_required": idle_shutdown_required,
            "external_watchdog_ttl_exceeds_hard_timeout": bool(
                watchdog_required and watchdog_ttl_seconds > hard_timeout_seconds
            ),
            "artifact_upload_before_shutdown_required": upload_before_shutdown_required,
            "local_sim_only_prerequisite_ready": local_sim_ready,
            "fleet_budget_guard_passed": (
                fleet_budget_guard.get("status") == "passed"
                if external_provider
                else None
            ),
        },
        "blockers": _dedupe(blockers),
        "fleet_budget_guard": fleet_budget_guard or None,
        "provider_race": _provider_race_contract(
            selected_provider=provider,
            gpu_selection=gpu_selection,
            startup_pipeline=startup_pipeline,
        ),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "can_launch_is_spend_gate_only": True,
            "can_launch_is_not_runtime_success": True,
            "can_launch_is_not_task_success": True,
        },
    }


def _worker_image_ref_env_var(simulator: str) -> str:
    return WORKER_IMAGE_REF_ENV_BY_SIMULATOR.get(
        simulator,
        GENERIC_WORKER_IMAGE_REF_ENV,
    )


def _configured_worker_image_ref(simulator: str) -> tuple[str, str]:
    env_var = _worker_image_ref_env_var(simulator)
    image_ref = _string(os.getenv(env_var))
    if image_ref:
        return image_ref, env_var
    generic_image_ref = _string(os.getenv(GENERIC_WORKER_IMAGE_REF_ENV))
    if generic_image_ref:
        return generic_image_ref, GENERIC_WORKER_IMAGE_REF_ENV
    return "", env_var


def _worker_image_manifest_diagnostic_env_var(simulator: str) -> str:
    return WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV_BY_SIMULATOR.get(
        simulator,
        GENERIC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV,
    )


def _configured_worker_image_size_diagnostic(
    simulator: str,
    image_ref: str,
) -> dict[str, Any]:
    env_var = _worker_image_manifest_diagnostic_env_var(simulator)
    path_text = _string(os.getenv(env_var))
    source_env_var = env_var
    if not path_text:
        path_text = _string(os.getenv(GENERIC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV))
        source_env_var = GENERIC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV
    if not path_text:
        return {
            "image_size_diagnostic_env_var": env_var,
            "image_size_diagnostic_present": False,
        }
    diagnostic_path = Path(path_text).expanduser()
    base = {
        "image_size_diagnostic_env_var": source_env_var,
        "image_size_diagnostic_path": str(diagnostic_path),
        "image_size_diagnostic_present": False,
    }
    if not diagnostic_path.is_file():
        return base
    try:
        diagnostic = _mapping(read_json_any(diagnostic_path))
    except Exception as exc:
        return {
            **base,
            "image_size_diagnostic_read_error": type(exc).__name__,
        }
    diagnostic_image_ref = _string(diagnostic.get("image_ref"))
    image_ref_matches = bool(
        image_ref and diagnostic_image_ref and diagnostic_image_ref == image_ref
    )
    if not image_ref_matches:
        return {
            **base,
            "image_size_diagnostic_image_ref": diagnostic_image_ref,
            "image_size_diagnostic_image_ref_matches": False,
        }
    return {
        **base,
        "image_size_diagnostic_present": True,
        "image_size_diagnostic_image_ref": diagnostic_image_ref or None,
        "image_size_diagnostic_image_ref_matches": True,
        "image_size_diagnostic": diagnostic,
    }


def _worker_image_ref_is_versioned(image_ref: str) -> bool:
    if not image_ref:
        return False
    if "@sha256:" in image_ref:
        return True
    name = image_ref.rsplit("/", 1)[-1]
    if ":" not in name:
        return False
    tag = name.rsplit(":", 1)[-1].strip().lower()
    return bool(tag and tag not in {"latest", "local", "dev", "test"})


def _worker_image_ref_is_provider_fetchable(image_ref: str, *, versioned: bool) -> bool:
    if not image_ref or not versioned:
        return False
    lowered = image_ref.lower()
    if any(marker in lowered for marker in ("placeholder", "<", ">", "candidate")):
        return False
    return True


def _configured_worker_artifact_output_uri() -> str:
    return _string(os.getenv(WORKER_ARTIFACT_OUTPUT_URI_ENV))


def _configured_worker_manifest_uri() -> str:
    return _string(os.getenv(WORKER_MANIFEST_URI_ENV))


def _configured_capture_root_bundle_uri() -> str:
    return _string(os.getenv(WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV))


def _uri_scheme(uri: str) -> str:
    parsed = urlparse(uri)
    return parsed.scheme or "local"


def _worker_manifest_uri_is_fetchable_by_provider(
    uri: str,
    *,
    live_gpu_provider: bool,
) -> bool:
    if not uri:
        return False
    scheme = _uri_scheme(uri)
    if live_gpu_provider:
        return scheme in REMOTE_WORKER_MANIFEST_URI_SCHEMES
    return scheme in {"local", *REMOTE_WORKER_MANIFEST_URI_SCHEMES}


def _provider_uri_is_fetchable(uri: str, *, live_gpu_provider: bool) -> bool:
    if not uri:
        return False
    scheme = _uri_scheme(uri)
    if live_gpu_provider:
        return scheme in REMOTE_WORKER_MANIFEST_URI_SCHEMES
    return scheme in {"local", *REMOTE_WORKER_MANIFEST_URI_SCHEMES}


def _provider_artifact_output_uri_is_writable(
    uri: str, *, live_gpu_provider: bool
) -> bool:
    if not uri:
        return False
    scheme = _uri_scheme(uri)
    if live_gpu_provider:
        return scheme in REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES
    return scheme in {"local", "file", *REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES}


def _artifact_output_write_auth_contract(
    uri: str,
    *,
    external_provider: bool,
    provider_writable: bool,
) -> Dict[str, Any]:
    scheme = _uri_scheme(uri) if uri else ""
    required_secret_env_vars = _string_list(
        ARTIFACT_OUTPUT_WRITE_SECRET_ENV_VARS_BY_SCHEME.get(scheme, [])
    )
    required_plaintext_env_vars = _string_list(
        ARTIFACT_OUTPUT_WRITE_PLAINTEXT_ENV_VARS_BY_SCHEME.get(scheme, [])
    )
    remote_object_storage_output = scheme in REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES
    write_auth_required = bool(external_provider)
    write_auth_contract_ready = bool(
        not write_auth_required or (provider_writable and required_secret_env_vars)
    )
    if remote_object_storage_output:
        authorization_mode = "worker_storage_credentials"
    elif scheme in {"local", "file"}:
        authorization_mode = "local_filesystem"
    elif scheme:
        authorization_mode = "unsupported_uri_scheme"
    else:
        authorization_mode = "missing_output_uri"
    return {
        "schema_version": "robot_eval_artifact_output_write_auth_contract.v1",
        "artifact_output_uri_scheme": scheme or None,
        "authorization_mode": authorization_mode,
        "write_auth_required_for_provider": write_auth_required,
        "write_auth_contract_ready": write_auth_contract_ready,
        "required_secret_env_vars": required_secret_env_vars,
        "required_plaintext_env_vars": required_plaintext_env_vars,
        "secret_values_in_artifact": False,
        "signed_uri_or_storage_credentials_required": write_auth_required,
        "presigned_put_uri_provided": False,
        "storage_credentials_declared": bool(required_secret_env_vars),
        "claim_boundary": (
            "Records the object-storage write authorization contract only; it does "
            "not store credentials or prove a live upload happened."
        ),
    }


def _persistent_cache_paths(simulator: str) -> Dict[str, str]:
    root = _string(os.getenv("BLUEPRINT_PROVIDER_CACHE_ROOT")) or "/opt/blueprint/cache"
    if simulator == "mujoco":
        return {
            "mujoco_assets": _string(os.getenv("BLUEPRINT_MUJOCO_ASSET_CACHE"))
            or f"{root}/mujoco_assets",
            "policy_files": _string(os.getenv("BLUEPRINT_POLICY_CACHE"))
            or f"{root}/policy_files",
            "converted_scenes": _string(os.getenv("BLUEPRINT_CONVERTED_SCENE_CACHE"))
            or f"{root}/converted_scenes",
            "worker_deps": _string(os.getenv("BLUEPRINT_WORKER_DEPS_CACHE"))
            or f"{root}/worker_deps",
        }
    return {
        "scene_assets": f"{root}/scene_assets",
        "policy_files": f"{root}/policy_files",
        "worker_deps": f"{root}/worker_deps",
    }


def _provider_gpu_priority_for_simulator(
    simulator: str,
    gpu_allocation: Mapping[str, Any],
) -> List[str]:
    explicit = _string_list(
        gpu_allocation.get("provider_gpu_priority")
        or gpu_allocation.get("gpu_priority_fallback_list")
        or gpu_allocation.get("runpod_gpu_priority")
    )
    if explicit:
        return explicit
    if simulator == "mujoco":
        return [
            "NVIDIA L4",
            "NVIDIA RTX 4000 Ada Generation",
            "NVIDIA RTX A4000",
            "NVIDIA RTX 3090",
            "NVIDIA RTX A5000",
        ]
    if simulator in ISAAC_SIMULATORS:
        return [
            "NVIDIA RTX 4090",
            "NVIDIA RTX A6000",
            "NVIDIA RTX 6000 Ada Generation",
        ]
    return []


def _warm_pool_policy(
    *,
    gpu_allocation: Mapping[str, Any],
    launch_limits: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    launch_limits = launch_limits or {}
    warm_config = _mapping(
        gpu_allocation.get("warm_pool_policy") or gpu_allocation.get("warm_pool")
    )
    latency_slo_seconds = _number(
        warm_config.get("latency_slo_seconds")
        or gpu_allocation.get("latency_slo_seconds")
    )
    estimated_idle_cost = _number(warm_config.get("estimated_idle_cost_usd_per_hour"), 0.0) or 0.0
    max_idle_cost = _number(
        warm_config.get("max_idle_cost_usd_per_hour")
        or gpu_allocation.get("max_idle_cost_usd_per_hour"),
        0.0,
    ) or 0.0
    warm_requested = bool(
        warm_config.get("enabled") is True
        or gpu_allocation.get("prefer_warm_gpu") is True
        or _string(gpu_allocation.get("mode")) in {"warm_pool", "active_worker"}
    )
    latency_justifies_idle = bool(
        warm_config.get("latency_justifies_idle_cost") is True
        or (latency_slo_seconds is not None and latency_slo_seconds <= 60)
    )
    idle_cost_allowed = estimated_idle_cost <= max_idle_cost
    warm_recommended = bool(warm_requested and latency_justifies_idle and idle_cost_allowed)
    max_active_workers = int(
        _number(
            warm_config.get("max_active_workers")
            or gpu_allocation.get("max_active_workers")
            or launch_limits.get("max_active_workers"),
            1,
        )
        or 1
    )
    reasons: List[str] = []
    if not warm_requested:
        reasons.append("warm_pool_not_requested")
    if warm_requested and not latency_justifies_idle:
        reasons.append("latency_policy_does_not_justify_idle_cost")
    if warm_requested and not idle_cost_allowed:
        reasons.append("warm_idle_cost_exceeds_policy")
    if warm_recommended:
        reasons.append("latency_policy_justifies_idle_cost")
    return {
        "decision": "warm_active_worker" if warm_recommended else "scale_to_zero_on_demand",
        "warm_worker_recommended": warm_recommended,
        "active_worker_target": 1 if warm_recommended else 0,
        "max_active_workers": max(1, max_active_workers),
        "scale_to_zero_default": not warm_recommended,
        "latency_slo_seconds": latency_slo_seconds,
        "estimated_idle_cost_usd_per_hour": estimated_idle_cost,
        "max_idle_cost_usd_per_hour": max_idle_cost,
        "decision_reasons": reasons,
    }


def _runtime_preflight_contract(
    *,
    simulator: str,
    provisioner: str,
    worker_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    runtime_required = simulator != "fixture"
    live_gpu_provider = provisioner in LIVE_GPU_PROVISIONERS and simulator != "fixture"
    requires_gpu_inventory = bool(
        live_gpu_provider or simulator in {"isaac_sim", "isaac_lab_arena", "newton"}
    )
    if simulator in ISAAC_SIMULATORS:
        required_checks = [
            "nvidia_smi_gpu_inventory",
            "driver_version",
            "vulkan_device",
            "rtx_renderer_available",
            "isaac_headless_launch",
            "blank_scene_load",
            "test_frame_render",
            "shader_cache_writable",
        ]
        renderer_context = "vulkan_rtx"
    elif simulator == "mujoco":
        required_checks = [
            "python_import_mujoco",
            "headless_context_selection",
            "egl_context_when_rendering",
            "blank_model_or_scene_load",
            "short_rollout_smoke",
        ]
        renderer_context = "egl_when_rendering"
    elif simulator == "pybullet":
        required_checks = [
            "python_import_pybullet",
            "headless_connection",
            "tiny_renderer_or_egl_smoke",
            "blank_scene_load",
            "short_rollout_smoke",
        ]
        renderer_context = "tiny_renderer_or_egl_when_rendering"
    elif simulator == "newton":
        required_checks = [
            "nvidia_smi_gpu_inventory",
            "driver_version",
            "gpu_physics_runtime_import",
            "blank_scene_load",
            "short_rollout_smoke",
        ]
        renderer_context = "gpu_physics_runtime"
    else:
        required_checks = []
        renderer_context = "not_required"
    runtime_command = _runtime_preflight_command_for_simulator(simulator)
    return {
        "required_before_scene_load": runtime_required,
        "required_for_provider": runtime_required and provisioner != "fixture_local",
        "worker_blocks_scene_load_on_failed_preflight": runtime_required,
        "executed_by": "blueprint-run-robot-eval-worker",
        "result_artifact": "worker_runtime_preflight.json",
        "command": runtime_command or None,
        "run_before": "scene_load_and_policy_execution",
        "simulator": simulator,
        "worker_image_family": worker_profile.get("worker_image_family"),
        "requires_gpu_inventory": requires_gpu_inventory,
        "renderer_context": renderer_context,
        "required_checks": required_checks,
        "nvidia_smi_required": requires_gpu_inventory,
        "vulkan_required": simulator in ISAAC_SIMULATORS,
        "egl_required_when_rendering": simulator in {"mujoco", "pybullet"},
        "blank_scene_or_model_load_required": runtime_required,
        "test_frame_render_required": simulator in ISAAC_SIMULATORS,
        "runtime_preflight_is_not_simulator_proof": True,
        "public_claim_upgrade_allowed": False,
    }


def _runtime_preflight_command_for_simulator(simulator: str) -> str:
    if simulator == "mujoco":
        return "python -m blueprint_pipeline.mujoco_worker_runtime_preflight --smoke-steps 2"
    return ""


def _build_worker_launch_plan(
    *,
    request: Mapping[str, Any],
    job_id: str,
    provisioner: str,
    simulator: str,
    scheduler_decision: Mapping[str, Any],
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    selection = _mapping(scheduler_decision.get("selection"))
    worker_profile = _mapping(selection.get("worker_profile"))
    gpu_allocation = _mapping(scheduler_decision.get("gpu_allocation"))
    artifact_contract = _mapping(scheduler_decision.get("artifact_contract"))
    scheduler_blockers = _string_list(scheduler_decision.get("blockers"))
    external_provider = provisioner != "fixture_local"
    live_gpu_provider = provisioner in LIVE_GPU_PROVISIONERS and simulator != "fixture"
    image_ref, image_ref_env_var = _configured_worker_image_ref(simulator)
    image_ref_versioned = _worker_image_ref_is_versioned(image_ref)
    image_ref_fetchable = _worker_image_ref_is_provider_fetchable(
        image_ref,
        versioned=image_ref_versioned,
    )
    worker_manifest_uri = _configured_worker_manifest_uri()
    worker_manifest_uri_required = external_provider
    worker_manifest_uri_fetchable = _worker_manifest_uri_is_fetchable_by_provider(
        worker_manifest_uri,
        live_gpu_provider=live_gpu_provider,
    )
    capture_root_bundle_uri = _configured_capture_root_bundle_uri()
    capture_root_bundle_required = live_gpu_provider
    capture_root_bundle_fetchable = _provider_uri_is_fetchable(
        capture_root_bundle_uri,
        live_gpu_provider=live_gpu_provider,
    )
    artifact_output_uri = _configured_worker_artifact_output_uri()
    artifact_output_required = external_provider
    artifact_output_uri_scheme = _uri_scheme(artifact_output_uri) if artifact_output_uri else None
    artifact_output_uri_provider_writable = _provider_artifact_output_uri_is_writable(
        artifact_output_uri,
        live_gpu_provider=external_provider,
    )
    artifact_output_write_auth = _artifact_output_write_auth_contract(
        artifact_output_uri,
        external_provider=external_provider,
        provider_writable=artifact_output_uri_provider_writable,
    )
    hard_timeout_seconds = int(
        _number(gpu_allocation.get("hard_timeout_seconds"), timeout_seconds)
        or timeout_seconds
    )
    shutdown_grace_seconds = int(
        _number(gpu_allocation.get("shutdown_grace_seconds"), 60) or 60
    )
    idle_timeout_seconds = int(
        _number(gpu_allocation.get("idle_timeout_seconds"), 60) or 60
    )
    external_watchdog_ttl_seconds = int(
        _number(
            gpu_allocation.get("external_watchdog_ttl_seconds"),
            hard_timeout_seconds + shutdown_grace_seconds,
        )
        or hard_timeout_seconds + shutdown_grace_seconds
    )
    launch_limits = {
        "max_active_workers": int(_number(gpu_allocation.get("max_active_workers"), 1) or 1),
    }
    warm_policy = _warm_pool_policy(
        gpu_allocation=gpu_allocation,
        launch_limits=launch_limits,
    )
    image_size_diagnostic = _configured_worker_image_size_diagnostic(
        simulator,
        image_ref,
    )
    runtime_preflight_contract = _runtime_preflight_contract(
        simulator=simulator,
        provisioner=provisioner,
        worker_profile=worker_profile,
    )
    image_blockers: List[str] = []
    if live_gpu_provider and not image_ref:
        image_blockers.append("missing_prebuilt_worker_image_ref")
    elif live_gpu_provider and not image_ref_versioned:
        image_blockers.append("prebuilt_worker_image_ref_not_versioned")
    elif live_gpu_provider and not image_ref_fetchable:
        image_blockers.append("prebuilt_worker_image_ref_not_provider_fetchable")
    artifact_blockers: List[str] = []
    if artifact_output_required and not artifact_output_uri:
        artifact_blockers.append("missing_worker_artifact_output_uri")
    elif artifact_output_required and not artifact_output_uri_provider_writable:
        artifact_blockers.append("worker_artifact_output_uri_not_provider_writable")
    elif artifact_output_required and not bool(
        artifact_output_write_auth.get("write_auth_contract_ready")
    ):
        artifact_blockers.append("worker_artifact_output_write_auth_contract_missing")
    manifest_uri_blockers: List[str] = []
    if worker_manifest_uri_required and not worker_manifest_uri:
        manifest_uri_blockers.append("missing_worker_manifest_uri")
    elif worker_manifest_uri_required and not worker_manifest_uri_fetchable:
        manifest_uri_blockers.append("worker_manifest_uri_not_fetchable_by_provider")
    input_bundle_blockers: List[str] = []
    if capture_root_bundle_required and not capture_root_bundle_uri:
        input_bundle_blockers.append("missing_capture_root_bundle_uri")
    elif capture_root_bundle_required and not capture_root_bundle_fetchable:
        input_bundle_blockers.append("capture_root_bundle_uri_not_fetchable_by_provider")
    blockers = _dedupe(
        [
            *scheduler_blockers,
            *image_blockers,
            *manifest_uri_blockers,
            *input_bundle_blockers,
            *artifact_blockers,
        ]
    )
    status = (
        "blocked_by_scheduler"
        if scheduler_blockers
        else "blocked_missing_prebuilt_worker_image_ref"
        if image_blockers
        and "prebuilt_worker_image_ref_not_provider_fetchable" not in image_blockers
        else "blocked_unfetchable_prebuilt_worker_image_ref"
        if "prebuilt_worker_image_ref_not_provider_fetchable" in image_blockers
        else "blocked_missing_worker_manifest_uri"
        if "missing_worker_manifest_uri" in manifest_uri_blockers
        else "blocked_invalid_worker_manifest_uri"
        if manifest_uri_blockers
        else "blocked_missing_capture_root_bundle_uri"
        if "missing_capture_root_bundle_uri" in input_bundle_blockers
        else "blocked_invalid_capture_root_bundle_uri"
        if input_bundle_blockers
        else "blocked_missing_worker_artifact_output_uri"
        if "missing_worker_artifact_output_uri" in artifact_blockers
        else "blocked_invalid_worker_artifact_output_uri"
        if artifact_blockers
        else "not_required_for_fixture_local"
        if not external_provider and simulator == "fixture"
        else "awaiting_explicit_provider_gate"
        if external_provider
        else "planned_for_local_execution"
    )
    credential_env_vars = _provider_credential_env_vars(provisioner)
    image_family = _string(worker_profile.get("worker_image_family")) or "repo-local-fixture"
    expected_outputs = (
        _dedupe(
            [
                *_string_list(artifact_contract.get("expected_outputs")),
                *DEFAULT_STARTUP_EXPECTED_OUTPUTS,
            ]
        )
    )
    return {
        "schema_version": WORKER_LAUNCH_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": status,
        "provider": provisioner,
        "simulator": simulator,
        "live_provider_calls_allowed_by_default": False,
        "live_provider_calls_performed": False,
        "scheduler_decision_path": "scheduler_decision.json",
        "scheduler_decision_status": scheduler_decision.get("status"),
        "scheduler_blockers": scheduler_blockers,
        "worker_image": {
            "image_family": image_family,
            "dockerfile_path": worker_profile.get("dockerfile_path"),
            "entrypoint": worker_profile.get("entrypoint") or "blueprint-run-robot-eval-worker",
            "version_pin_required": True,
            "prebuilt_image_required": external_provider or simulator != "fixture",
            "published_image_ref_required": live_gpu_provider,
            "image_ref_env_var": image_ref_env_var,
            "configured_image_ref": image_ref or None,
            "configured_image_ref_present": bool(image_ref),
            "configured_image_ref_is_versioned": image_ref_versioned,
            "configured_image_ref_fetchable_by_provider": image_ref_fetchable,
            **image_size_diagnostic,
            "runtime_dependency_install_disallowed": True,
            "runtime_asset_guessing_disallowed": True,
        },
        "gpu_selection": {
            "preferred_gpu_class": worker_profile.get("preferred_gpu_class"),
            "disallowed_gpu_classes": _string_list(worker_profile.get("disallowed_gpu_classes")),
            "cheap_cpu_or_gpu_allowed": simulator in {"mujoco", "pybullet", "fixture"},
            "provider_gpu_priority_fallback_list": _provider_gpu_priority_for_simulator(
                simulator,
                gpu_allocation,
            ),
            "requires_isaac_class_gpu": simulator in ISAAC_SIMULATORS,
        },
        "launch_mode": {
            "mode": gpu_allocation.get("mode") or "on_demand_with_optional_warm_pool",
            "scale_to_zero_default": True,
            "warm_pool_allowed_when_explicitly_requested": True,
            "warm_pool_policy": warm_policy,
            "active_worker_target": warm_policy.get("active_worker_target"),
            "max_active_workers": int(_number(gpu_allocation.get("max_active_workers"), 1) or 1),
            "idle_shutdown_required": bool(
                gpu_allocation.get("idle_shutdown_required") is not False
            ),
            "idle_timeout_seconds": idle_timeout_seconds,
            "hard_timeout_seconds": hard_timeout_seconds,
            "shutdown_grace_seconds": shutdown_grace_seconds,
            "external_watchdog_ttl_required": external_provider,
            "external_watchdog_ttl_seconds": external_watchdog_ttl_seconds,
            "external_watchdog_owner": "provider_launcher_or_owner_control_plane",
        },
        "cache_plan": {
            "persistent_cache_recommended": bool(
                gpu_allocation.get("persistent_cache_recommended") is not False
            ),
            "targets": _string_list(worker_profile.get("persistent_cache_targets")),
            "paths": _persistent_cache_paths(simulator),
            "install_simulator_during_customer_job": False,
            "install_python_dependencies_during_customer_job": False,
        },
        "runtime_preflight_contract": runtime_preflight_contract,
        "worker_entrypoint_contract": {
            "job_manifest_env": WORKER_MANIFEST_URI_ENV,
            "expected_command_shape": (
                "blueprint-run-robot-eval-worker --manifest ${BLUEPRINT_EVAL_MANIFEST_URI}"
            ),
            "package_console_script": "blueprint-run-robot-eval-worker",
            "delegates_to_console_script": "blueprint-run-robot-eval-job",
            "web_request_waits_for_worker": False,
        },
        "input_bundle": {
            "job_request_path": "job_request.json",
            "scenario_eval_matrix_path": "scenario_eval_matrix.json",
            "policy_package_manifest_path": "policy_package_manifest.json",
            "scene_asset_preflight_path": "../simulation_automation/scene_asset_preflight.json",
            "gpu_handoff_packet_path": "../simulation_automation/gpu_handoff_packet.json",
            "capture_root_bundle_uri_env_var": WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV,
            "capture_root_bundle_uri": capture_root_bundle_uri or None,
            "capture_root_bundle_uri_required_for_provider": capture_root_bundle_required,
            "capture_root_bundle_uri_fetchable_by_provider": capture_root_bundle_fetchable,
            "capture_root_bundle_uri_scheme": _uri_scheme(capture_root_bundle_uri)
            if capture_root_bundle_uri
            else None,
            "capture_root_local_path_disallowed_for_live_provider": live_gpu_provider,
            "capture_root_bundle_expected_format": "zip_or_tar_with_capture_descriptor_json",
            "original_customer_request_id": request.get("request_id")
            or request.get("requestId")
            or None,
        },
        "worker_manifest_input_contract": {
            "worker_manifest_path": "worker_manifest.json",
            "worker_manifest_schema": WORKER_MANIFEST_SCHEMA_VERSION,
            "worker_manifest_uri_env_var": WORKER_MANIFEST_URI_ENV,
            "worker_manifest_uri_required_for_provider": worker_manifest_uri_required,
            "configured_worker_manifest_uri": worker_manifest_uri or None,
            "configured_worker_manifest_uri_present": bool(worker_manifest_uri),
            "worker_manifest_uri_scheme": _uri_scheme(worker_manifest_uri)
            if worker_manifest_uri
            else None,
            "worker_manifest_uri_fetchable_by_provider": worker_manifest_uri_fetchable,
            "live_provider_remote_uri_required": live_gpu_provider,
            "allowed_uri_schemes": [
                "local",
                "https",
                "gs",
                "s3",
                "r2",
            ],
            "remote_provider_uri_schemes": sorted(REMOTE_WORKER_MANIFEST_URI_SCHEMES),
            "local_path_only_disallowed_for_live_provider": live_gpu_provider,
        },
        "artifact_upload_contract": {
            "destination_required": True,
            "destination_ref": "job-scoped-object-storage-prefix",
            "configured_artifact_output_uri": artifact_output_uri or None,
            "configured_artifact_output_uri_present": bool(artifact_output_uri),
            "artifact_output_uri_scheme": artifact_output_uri_scheme,
            "artifact_output_uri_provider_writable": artifact_output_uri_provider_writable,
            "artifact_output_write_auth": artifact_output_write_auth,
            "artifact_output_write_auth_contract_ready": bool(
                artifact_output_write_auth.get("write_auth_contract_ready")
            ),
            "artifact_output_uri_env_var": WORKER_ARTIFACT_OUTPUT_URI_ENV,
            "artifact_output_uri_required_for_provider": artifact_output_required,
            "manifest_input_uri_schemes": ["local", "https", "gs", "s3", "r2"],
            "artifact_output_uri_schemes": ["file", "gs", "s3", "r2"],
            "remote_provider_writable_artifact_output_uri_schemes": sorted(
                REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES
            ),
            "s3_compatible_storage_supported": True,
            "r2_requires_endpoint_env": True,
            "expected_outputs": expected_outputs,
            "upload_before_shutdown_required": True,
        },
        "approval_gates": {
            "env_BLUEPRINT_ALLOW_GPU_PROVISIONING_required": external_provider,
            "cli_allow_gpu_provisioning_required": external_provider,
            "env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION_required": simulator != "fixture",
            "cli_allow_simulator_execution_required": simulator != "fixture",
            "allowed_simulator_flag_required": simulator != "fixture",
        },
        "secret_policy": {
            "provider_credential_env_vars": credential_env_vars,
            "store_provider_credentials_in_artifacts": False,
            "redact_provider_tokens_from_logs": True,
            "customer_visible_provider_secrets_allowed": False,
        },
        "cost_controls": {
            "requested_budget_usd": gpu_allocation.get("requested_budget_usd"),
            "budget_required_before_live_allocation": external_provider,
            "customer_concurrency_limit_required": True,
            "record_actual_gpu_time_required": True,
            "finalizer_must_upload_artifacts_before_shutdown": True,
        },
        "blockers": blockers,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _build_worker_manifest(
    *,
    request: Mapping[str, Any],
    job_id: str,
    capture_root: Path,
    provisioner: str,
    simulator: str,
    evaluation_substrate: str | None,
    worker_launch_plan: Mapping[str, Any],
    allowed_simulators: Sequence[str],
    simulator_commands: Mapping[str, str],
    allow_wam_provider: bool,
    wam_provider_commands: Mapping[str, str],
    wam_artifact_output_uri: str | None,
    wam_provider_max_retries: int,
    wam_provider_timeout_seconds: int | None,
    timeout_seconds: int,
    budget_usd: float | None,
    generated_at: str,
) -> Dict[str, Any]:
    artifact_contract = _mapping(worker_launch_plan.get("artifact_upload_contract"))
    input_bundle = _mapping(worker_launch_plan.get("input_bundle"))
    manifest_input_contract = _mapping(
        worker_launch_plan.get("worker_manifest_input_contract")
    )
    runtime_preflight_contract = _mapping(
        worker_launch_plan.get("runtime_preflight_contract")
    )
    runtime_preflight_command = _string(runtime_preflight_contract.get("command"))
    worker_manifest_uri = _string(
        manifest_input_contract.get("configured_worker_manifest_uri")
    )
    worker_manifest_uri_required = bool(
        manifest_input_contract.get("worker_manifest_uri_required_for_provider")
    )
    worker_manifest_uri_fetchable = bool(
        manifest_input_contract.get("worker_manifest_uri_fetchable_by_provider")
    )
    artifact_output_uri = _string(artifact_contract.get("configured_artifact_output_uri"))
    artifact_output_required = bool(
        artifact_contract.get("artifact_output_uri_required_for_provider")
    )
    artifact_output_provider_writable = bool(
        artifact_contract.get("artifact_output_uri_provider_writable")
    )
    artifact_output_write_auth = _mapping(artifact_contract.get("artifact_output_write_auth"))
    artifact_output_write_auth_ready = bool(
        artifact_contract.get("artifact_output_write_auth_contract_ready")
        or artifact_output_write_auth.get("write_auth_contract_ready")
    )
    capture_root_bundle_uri = _string(input_bundle.get("capture_root_bundle_uri"))
    blockers: List[str] = []
    if worker_manifest_uri_required and not worker_manifest_uri:
        blockers.append("missing_worker_manifest_uri")
    elif worker_manifest_uri_required and not worker_manifest_uri_fetchable:
        blockers.append("worker_manifest_uri_not_fetchable_by_provider")
    if artifact_output_required and not artifact_output_uri:
        blockers.append("missing_worker_artifact_output_uri")
    if artifact_output_required and artifact_output_uri and not artifact_output_provider_writable:
        blockers.append("worker_artifact_output_uri_not_provider_writable")
    if (
        artifact_output_required
        and artifact_output_uri
        and artifact_output_provider_writable
        and not artifact_output_write_auth_ready
    ):
        blockers.append("worker_artifact_output_write_auth_contract_missing")
    return {
        "schema_version": WORKER_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "blocked" if blockers else "ready_for_worker_upload",
        "capture_root": str(capture_root),
        "provisioner": provisioner,
        "simulator": simulator,
        "evaluation_substrate": evaluation_substrate or None,
        "timeout_seconds": timeout_seconds,
        "budget_usd": budget_usd,
        "allowed_simulators": list(allowed_simulators),
        "simulator_commands": dict(simulator_commands),
        "allow_wam_provider": bool(allow_wam_provider),
        "wam_provider_commands": dict(wam_provider_commands),
        "wam_artifact_output_uri": wam_artifact_output_uri or artifact_output_uri or None,
        "wam_provider_max_retries": wam_provider_max_retries,
        "wam_provider_timeout_seconds": wam_provider_timeout_seconds or timeout_seconds,
        "input_bundle": dict(input_bundle),
        "capture_root_bundle_uri": capture_root_bundle_uri or None,
        "capture_root_bundle_uri_env_var": WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV,
        "worker_manifest_uri": worker_manifest_uri or None,
        "worker_manifest_uri_required": worker_manifest_uri_required,
        "worker_manifest_uri_env_var": WORKER_MANIFEST_URI_ENV,
        "worker_manifest_uri_fetchable_by_provider": worker_manifest_uri_fetchable,
        "worker_manifest_uri_scheme": manifest_input_contract.get(
            "worker_manifest_uri_scheme"
        ),
        "runtime_preflight_contract": runtime_preflight_contract,
        "runtime_preflight_command": runtime_preflight_command or None,
        "artifact_output_uri": artifact_output_uri or None,
        "artifact_output_uri_required": artifact_output_required,
        "artifact_output_uri_scheme": artifact_contract.get("artifact_output_uri_scheme"),
        "artifact_output_uri_provider_writable": artifact_output_provider_writable,
        "artifact_output_write_auth": artifact_output_write_auth,
        "artifact_output_write_auth_contract_ready": artifact_output_write_auth_ready,
        "artifact_output_uri_env_var": WORKER_ARTIFACT_OUTPUT_URI_ENV,
        "worker_launch_plan_path": "worker_launch_plan.json",
        "job_request": dict(request),
        "blockers": blockers,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _build_worker_provider_command(
    *,
    allow_gpu_provisioning: bool,
    allow_simulator_execution: bool,
    allowed_simulators: Sequence[str],
    simulator_commands: Mapping[str, str],
    evaluation_substrate: str | None,
    allow_wam_provider: bool,
    wam_provider_commands: Mapping[str, str],
    wam_artifact_output_uri: str | None,
    wam_provider_max_retries: int,
    wam_provider_timeout_seconds: int | None,
) -> str:
    command = [
        "blueprint-run-robot-eval-worker",
        "--manifest",
        "${BLUEPRINT_EVAL_MANIFEST_URI}",
    ]
    if allow_gpu_provisioning:
        command.append("--allow-gpu-provisioning")
    if allow_simulator_execution:
        command.append("--allow-simulator-execution")
    if evaluation_substrate:
        command.extend(["--evaluation-substrate", evaluation_substrate])
    for simulator in _dedupe(_string_list(allowed_simulators)):
        if simulator and simulator != "fixture":
            command.extend(["--allowed-simulator", simulator])
    allowed_simulator_set = set(_dedupe(_string_list(allowed_simulators)))
    if allow_simulator_execution or allowed_simulator_set:
        for simulator in sorted(simulator_commands):
            simulator_command = _string(simulator_commands.get(simulator))
            simulator_allowed = allow_simulator_execution or simulator in allowed_simulator_set
            if simulator and simulator_allowed and simulator_command:
                command.extend(["--simulator-command", f"{simulator}={simulator_command}"])
    if allow_wam_provider:
        command.append("--allow-wam-provider")
    for substrate in sorted(wam_provider_commands):
        wam_provider_command = _string(wam_provider_commands.get(substrate))
        if substrate and wam_provider_command:
            command.extend(["--wam-provider-command", f"{substrate}={wam_provider_command}"])
    if wam_artifact_output_uri:
        command.extend(["--wam-artifact-output-uri", wam_artifact_output_uri])
    if wam_provider_max_retries:
        command.extend(["--wam-provider-max-retries", str(wam_provider_max_retries)])
    if wam_provider_timeout_seconds:
        command.extend(["--wam-provider-timeout-seconds", str(wam_provider_timeout_seconds)])
    command_prefix = "blueprint-run-robot-eval-worker --manifest ${BLUEPRINT_EVAL_MANIFEST_URI}"
    command_tail = shlex.join(command[3:])
    return f"{command_prefix} {command_tail}" if command_tail else command_prefix


def _build_gpu_provider_launch_request(
    *,
    request_manifest: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    worker_launch_plan: Mapping[str, Any],
    worker_manifest: Mapping[str, Any],
    allow_gpu_provisioning: bool,
    allow_simulator_execution: bool,
    allowed_simulators: Sequence[str],
    simulator_commands: Mapping[str, str],
    evaluation_substrate: str | None,
    allow_wam_provider: bool,
    wam_provider_commands: Mapping[str, str],
    wam_artifact_output_uri: str | None,
    wam_provider_max_retries: int,
    wam_provider_timeout_seconds: int | None,
    generated_at: str,
    gpu_startup_pipeline_plan: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    provider = _string(request_manifest.get("provider")) or "fixture_local"
    job_id = _string(request_manifest.get("job_id"))
    worker_image = _mapping(worker_launch_plan.get("worker_image"))
    gpu_selection = _mapping(worker_launch_plan.get("gpu_selection"))
    launch_mode = _mapping(worker_launch_plan.get("launch_mode"))
    cache_plan = _mapping(worker_launch_plan.get("cache_plan"))
    entrypoint_contract = _mapping(worker_launch_plan.get("worker_entrypoint_contract"))
    manifest_input_contract = _mapping(
        worker_launch_plan.get("worker_manifest_input_contract")
    )
    runtime_preflight_contract = _mapping(
        worker_launch_plan.get("runtime_preflight_contract")
    )
    artifact_upload_contract = _mapping(worker_launch_plan.get("artifact_upload_contract"))
    cost_controls = _mapping(worker_launch_plan.get("cost_controls"))
    scheduler_blockers = _string_list(scheduler_decision.get("blockers"))
    worker_blockers = _string_list(worker_launch_plan.get("blockers"))
    worker_manifest_blockers = _string_list(worker_manifest.get("blockers"))
    startup_pipeline = _mapping(gpu_startup_pipeline_plan)
    startup_blockers = _string_list(startup_pipeline.get("blockers"))
    external_provider = provider != "fixture_local"
    local_sim_only_prerequisite = {
        "schema_version": "robot_eval_provider_local_sim_only_prerequisite.v1",
        "required_before_provider_spend": bool(external_provider),
        "status": "not_evaluated_yet" if external_provider else "not_required",
        "source_artifact": "robot_team_grade_eval_closure_manifest.json",
        "local_sim_only_evidence_clean": False if external_provider else None,
        "sim_only_beta_core_complete": None,
        "sim_only_beta_blocked_requirement_ids": [],
        "blockers": (
            ["local_sim_only_closure_not_evaluated_yet"]
            if external_provider
            else []
        ),
        "claim_boundary": {
            "provider_spend_requires_local_sim_only_evidence_clean": True,
            "local_sim_only_clean_does_not_prove_remote_provider_execution": True,
            "local_sim_only_clean_does_not_prove_launch_approval": True,
        },
    }
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_GPU_PROVISIONING")
    provider_worker_command = _build_worker_provider_command(
        allow_gpu_provisioning=allow_gpu_provisioning,
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=allowed_simulators,
        simulator_commands=simulator_commands,
        evaluation_substrate=evaluation_substrate,
        allow_wam_provider=allow_wam_provider,
        wam_provider_commands=wam_provider_commands,
        wam_artifact_output_uri=wam_artifact_output_uri,
        wam_provider_max_retries=wam_provider_max_retries,
        wam_provider_timeout_seconds=wam_provider_timeout_seconds,
    )
    approval_blockers: List[str] = []
    if external_provider:
        if not env_allowed:
            approval_blockers.append("missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING")
        if not allow_gpu_provisioning:
            approval_blockers.append("missing_cli_allow_gpu_provisioning")
    blockers = _dedupe(
        [
            *scheduler_blockers,
            *worker_blockers,
            *worker_manifest_blockers,
            *startup_blockers,
            *approval_blockers,
        ]
    )
    status = (
        "blocked_by_scheduler"
        if scheduler_blockers
        else "not_required_for_fixture_local"
        if provider == "fixture_local"
        else "blocked_by_worker_plan"
        if worker_blockers
        else "blocked_by_worker_manifest"
        if worker_manifest_blockers
        else "blocked_by_startup_pipeline"
        if startup_blockers
        else "blocked_by_explicit_provider_gate"
        if approval_blockers
        else "request_manifest_ready"
    )
    storage_secret_env_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ]
    plaintext_env_vars = [
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "BLUEPRINT_WORKER_DIR",
        "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI",
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "BLUEPRINT_ALLOW_GPU_PROVISIONING",
        "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
        "BLUEPRINT_ALLOWED_SIMULATORS",
        "BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL",
        "AWS_ENDPOINT_URL",
    ]
    prelaunch_spend_guard = _provider_prelaunch_spend_guard(
        provider=provider,
        request_manifest=request_manifest,
        scheduler_decision=scheduler_decision,
        worker_launch_plan=worker_launch_plan,
        startup_pipeline=startup_pipeline,
        local_sim_only_prerequisite=local_sim_only_prerequisite,
        approval_blockers=approval_blockers,
        request_blockers=blockers,
        env_allowed=env_allowed,
        allow_gpu_provisioning=allow_gpu_provisioning,
    )
    return {
        "schema_version": GPU_PROVIDER_LAUNCH_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "provider": provider,
        "status": status,
        "reason": (
            "scheduler_decision_blocked"
            if scheduler_blockers
            else "fixture_local_does_not_require_gpu_provider"
            if provider == "fixture_local"
            else "worker_launch_plan_blocked"
            if worker_blockers
            else "worker_manifest_blocked"
            if worker_manifest_blockers
            else "startup_pipeline_blocked"
            if startup_blockers
            else "explicit_provider_gate_required"
            if approval_blockers
            else "provider_launch_request_ready_for_explicit_launcher"
        ),
        "operation": _provider_launch_operation(provider),
        "live_provider_calls_allowed_by_default": False,
        "live_provider_calls_performed": False,
        "prelaunch_spend_guard": prelaunch_spend_guard,
        "provider_race_handoff_path": (
            "gpu_provider_race_handoff.json"
            if _mapping(prelaunch_spend_guard.get("provider_race")).get(
                "race_required_for_customer_path"
            )
            is True
            else None
        ),
        "scheduler_decision_path": "scheduler_decision.json",
        "scheduler_decision_status": scheduler_decision.get("status"),
        "worker_launch_plan_path": "worker_launch_plan.json",
        "worker_launch_plan_status": worker_launch_plan.get("status"),
        "worker_manifest_path": "worker_manifest.json",
        "worker_manifest_status": worker_manifest.get("status"),
        "gpu_startup_pipeline_plan_path": "gpu_startup_pipeline_plan.json",
        "gpu_startup_pipeline_plan_status": startup_pipeline.get("status"),
        "gpu_provisioning_request_path": "gpu_provisioning_request.json",
        "provider_request_shape": {
            "provider_api": provider,
            "api_payload_is_provider_adapter_template": True,
            "api_payload_values_are_redacted": True,
            "operation": _provider_launch_operation(provider),
            "image": {
                "image_family": worker_image.get("image_family"),
                "owner_published_image_ref_required": bool(
                    worker_image.get("published_image_ref_required")
                ),
                "configured_image_ref": worker_image.get("configured_image_ref"),
                "image_ref_env_var": worker_image.get("image_ref_env_var"),
                "configured_image_ref_present": bool(
                    worker_image.get("configured_image_ref_present")
                ),
                "configured_image_ref_is_versioned": bool(
                    worker_image.get("configured_image_ref_is_versioned")
                ),
                "configured_image_ref_fetchable_by_provider": bool(
                    worker_image.get("configured_image_ref_fetchable_by_provider")
                ),
                "dockerfile_path": worker_image.get("dockerfile_path"),
                "entrypoint": worker_image.get("entrypoint"),
                "runtime_dependency_install_disallowed": bool(
                    worker_image.get("runtime_dependency_install_disallowed")
                ),
            },
            "command": provider_worker_command,
            "environment": {
                "plaintext_env_var_names": plaintext_env_vars,
                "secret_env_var_names": _dedupe(
                    [
                        *_provider_credential_env_vars(provider),
                        *storage_secret_env_vars,
                    ]
                ),
                "secret_values_in_artifact": False,
                "customer_visible_secret_values_allowed": False,
            },
            "inputs": {
                "manifest_env_var": entrypoint_contract.get("job_manifest_env"),
                "manifest_uri_required": True,
                "manifest_uri_required_for_provider": bool(
                    manifest_input_contract.get(
                        "worker_manifest_uri_required_for_provider"
                    )
                ),
                "manifest_uri": worker_manifest.get("worker_manifest_uri"),
                "manifest_uri_env_var": WORKER_MANIFEST_URI_ENV,
                "manifest_uri_configured": bool(worker_manifest.get("worker_manifest_uri")),
                "manifest_uri_fetchable_by_provider": bool(
                    worker_manifest.get("worker_manifest_uri_fetchable_by_provider")
                ),
                "manifest_uri_scheme": worker_manifest.get("worker_manifest_uri_scheme"),
                "worker_manifest_path": "worker_manifest.json",
                "worker_manifest_schema": worker_manifest.get("schema_version"),
                "worker_manifest_local_path_ready": worker_manifest.get("status")
                == "ready_for_worker_upload",
                "capture_root_bundle_uri": worker_manifest.get("capture_root_bundle_uri"),
                "capture_root_bundle_uri_env_var": WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV,
                "capture_root_bundle_uri_configured": bool(
                    worker_manifest.get("capture_root_bundle_uri")
                ),
                "capture_root_bundle_uri_fetchable_by_provider": bool(
                    _mapping(worker_manifest.get("input_bundle")).get(
                        "capture_root_bundle_uri_fetchable_by_provider"
                    )
                ),
                "artifact_output_uri_required": bool(
                    artifact_upload_contract.get("destination_required")
                ),
                "artifact_output_uri": worker_manifest.get("artifact_output_uri"),
                "artifact_output_uri_scheme": worker_manifest.get("artifact_output_uri_scheme"),
                "artifact_output_uri_provider_writable": bool(
                    worker_manifest.get("artifact_output_uri_provider_writable")
                ),
                "artifact_output_write_auth": _mapping(
                    worker_manifest.get("artifact_output_write_auth")
                ),
                "artifact_output_write_auth_contract_ready": bool(
                    worker_manifest.get("artifact_output_write_auth_contract_ready")
                ),
                "provider_writable_artifact_output_uri_schemes": _string_list(
                    artifact_upload_contract.get(
                        "remote_provider_writable_artifact_output_uri_schemes"
                    )
                ),
                "manifest_input_uri_schemes": _string_list(
                    artifact_upload_contract.get("manifest_input_uri_schemes")
                ),
                "artifact_output_uri_schemes": _string_list(
                    artifact_upload_contract.get("artifact_output_uri_schemes")
                ),
            },
            "runtime_preflight": {
                "required_before_scene_load": bool(
                    runtime_preflight_contract.get("required_before_scene_load")
                ),
                "required_for_provider": bool(
                    runtime_preflight_contract.get("required_for_provider")
                ),
                "worker_blocks_scene_load_on_failed_preflight": bool(
                    runtime_preflight_contract.get(
                        "worker_blocks_scene_load_on_failed_preflight"
                    )
                ),
                "executed_by": runtime_preflight_contract.get("executed_by"),
                "result_artifact": runtime_preflight_contract.get("result_artifact"),
                "run_before": runtime_preflight_contract.get("run_before"),
                "simulator": runtime_preflight_contract.get("simulator"),
                "renderer_context": runtime_preflight_contract.get("renderer_context"),
                "required_checks": _string_list(
                    runtime_preflight_contract.get("required_checks")
                ),
                "nvidia_smi_required": bool(
                    runtime_preflight_contract.get("nvidia_smi_required")
                ),
                "vulkan_required": bool(
                    runtime_preflight_contract.get("vulkan_required")
                ),
                "egl_required_when_rendering": bool(
                    runtime_preflight_contract.get("egl_required_when_rendering")
                ),
                "blank_scene_or_model_load_required": bool(
                    runtime_preflight_contract.get(
                        "blank_scene_or_model_load_required"
                    )
                ),
                "test_frame_render_required": bool(
                    runtime_preflight_contract.get("test_frame_render_required")
                ),
                "runtime_preflight_is_not_simulator_proof": bool(
                    runtime_preflight_contract.get(
                        "runtime_preflight_is_not_simulator_proof"
                    )
                ),
            },
            "startup_pipeline": {
                "plan_path": "gpu_startup_pipeline_plan.json",
                "status": startup_pipeline.get("status"),
                "strategy": startup_pipeline.get("strategy"),
                "provider_selection_owner": startup_pipeline.get(
                    "provider_selection_owner"
                ),
                "selected_provider": startup_pipeline.get("selected_provider"),
                "selected_provider_tier": startup_pipeline.get(
                    "selected_provider_tier"
                ),
                "selected_provider_is_marketplace": bool(
                    startup_pipeline.get("selected_provider_is_marketplace")
                ),
                "managed_provider_policy": _mapping(
                    startup_pipeline.get("managed_provider_policy")
                ),
                "marketplace_policy": _mapping(
                    startup_pipeline.get("marketplace_policy")
                ),
                "preflight_canary_policy": _mapping(
                    startup_pipeline.get("preflight_canary_policy")
                ),
                "same_sku_burst_policy": _mapping(
                    startup_pipeline.get("same_sku_burst_policy")
                ),
                "webapp_boundary": _mapping(startup_pipeline.get("webapp_boundary")),
                "launcher_must_fail_closed_on_startup_blockers": bool(
                    _mapping(startup_pipeline.get("launcher_contract")).get(
                        "launcher_must_fail_closed_on_startup_blockers"
                    )
                ),
                "blockers": startup_blockers,
            },
            "local_sim_only_prerequisite": local_sim_only_prerequisite,
            "gpu": {
                "preferred_gpu_class": gpu_selection.get("preferred_gpu_class"),
                "disallowed_gpu_classes": _string_list(
                    gpu_selection.get("disallowed_gpu_classes")
                ),
                "cheap_cpu_or_gpu_allowed": bool(gpu_selection.get("cheap_cpu_or_gpu_allowed")),
                "provider_gpu_priority": _string_list(
                    gpu_selection.get("provider_gpu_priority_fallback_list")
                ),
                "priority_fallback_list": _string_list(
                    gpu_selection.get("provider_gpu_priority_fallback_list")
                ),
                "requires_isaac_class_gpu": bool(
                    gpu_selection.get("requires_isaac_class_gpu")
                ),
            },
            "cache": {
                "persistent_cache_recommended": bool(
                    cache_plan.get("persistent_cache_recommended")
                ),
                "targets": _string_list(cache_plan.get("targets")),
                "paths": _mapping(cache_plan.get("paths")),
                "install_simulator_during_customer_job": bool(
                    cache_plan.get("install_simulator_during_customer_job")
                ),
                "install_python_dependencies_during_customer_job": bool(
                    cache_plan.get("install_python_dependencies_during_customer_job")
                ),
            },
            "limits": {
                "max_active_workers": launch_mode.get("max_active_workers"),
                "active_worker_target": launch_mode.get("active_worker_target"),
                "hard_timeout_seconds": launch_mode.get("hard_timeout_seconds")
                or request_manifest.get("timeout_seconds"),
                "idle_timeout_seconds": launch_mode.get("idle_timeout_seconds"),
                "idle_shutdown_required": bool(launch_mode.get("idle_shutdown_required")),
                "scale_to_zero_default": bool(launch_mode.get("scale_to_zero_default")),
                "warm_pool_policy": _mapping(launch_mode.get("warm_pool_policy")),
                "shutdown_grace_seconds": launch_mode.get("shutdown_grace_seconds"),
                "external_watchdog_ttl_required": bool(
                    launch_mode.get("external_watchdog_ttl_required")
                ),
                "external_watchdog_ttl_seconds": launch_mode.get(
                    "external_watchdog_ttl_seconds"
                ),
                "external_watchdog_owner": launch_mode.get("external_watchdog_owner"),
                "requested_budget_usd": request_manifest.get("requested_budget_usd")
                if request_manifest.get("requested_budget_usd") is not None
                else cost_controls.get("requested_budget_usd"),
            },
            "artifact_finalizer": {
                "upload_before_shutdown_required": bool(
                    artifact_upload_contract.get("upload_before_shutdown_required")
                ),
                "record_actual_gpu_time_required": bool(
                    cost_controls.get("record_actual_gpu_time_required")
                ),
            },
        },
        "gate_requirements": {
            "env_BLUEPRINT_ALLOW_GPU_PROVISIONING_required": external_provider,
            "env_BLUEPRINT_ALLOW_GPU_PROVISIONING_present": env_allowed,
            "cli_allow_gpu_provisioning_required": external_provider,
            "cli_allow_gpu_provisioning_present": bool(allow_gpu_provisioning),
            "provider_credential_env_vars": _provider_credential_env_vars(provider),
            "provider_secret_values_in_artifact": False,
        },
        "blockers": blockers,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _build_gpu_provider_race_handoff(
    *,
    provider_launch_request: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    prelaunch_guard = _mapping(provider_launch_request.get("prelaunch_spend_guard"))
    provider_race = _mapping(
        prelaunch_guard.get("provider_race")
        or provider_launch_request.get("provider_race")
    )
    candidates = [
        _mapping(candidate)
        for candidate in provider_race.get("candidates") or []
        if isinstance(candidate, Mapping)
    ]
    race_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("race_candidate") is True
        and _string(candidate.get("provider"))
    ]
    runnable_candidates = [
        {
            "provider": candidate.get("provider"),
            "operation": candidate.get("operation"),
            "adapter_id": candidate.get("adapter_id"),
            "adapter_command": candidate.get("adapter_command"),
            "selected": bool(candidate.get("selected")),
            "launch_request_path": "gpu_provider_launch_request.json",
            "adapter_result_path": (
                f"{_string(candidate.get('provider'))}_provider_adapter_result.json"
            ),
        }
        for candidate in race_candidates
        if _string(candidate.get("adapter_command"))
    ]
    race_required = bool(provider_race.get("race_required_for_customer_path"))
    runtime_readiness = _mapping(
        provider_race.get("runtime_readiness")
    ) or _provider_race_runtime_readiness(race_candidates)
    blockers: List[str] = []
    if not race_required:
        blockers.append("provider_race_not_required_for_customer_path")
    if prelaunch_guard.get("required_before_provider_launch") is True and (
        prelaunch_guard.get("can_launch") is not True
    ):
        blockers.append("prelaunch_spend_guard_not_passed")
        blockers.extend(_string_list(prelaunch_guard.get("blockers")))
    if race_required and len(runnable_candidates) < 2:
        blockers.append("provider_race_requires_two_adapter_commands")
    runtime_wired = bool(
        runtime_readiness.get("customer_path_provider_failover_runtime_wired")
    )
    if race_required and not runtime_wired:
        blockers.append("customer_path_provider_failover_runtime_not_wired")
        blockers.extend(_string_list(runtime_readiness.get("blockers")))
    runtime_launcher_available = bool(race_required and len(runnable_candidates) >= 2)
    runtime_launcher_blockers = (
        [] if runtime_launcher_available or not race_required else [
            PROVIDER_RACE_RUNTIME_LAUNCHER_BLOCKER
        ]
    )
    blockers.extend(runtime_launcher_blockers)
    ready = (
        race_required
        and runtime_wired
        and runtime_launcher_available
        and not blockers
    )
    return {
        "schema_version": GPU_PROVIDER_RACE_HANDOFF_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": provider_launch_request.get("job_id"),
        "status": "ready_for_customer_provider_race_runtime"
        if ready
        else "blocked_before_provider_race_launcher",
        "reason": "provider_race_handoff_ready"
        if ready
        else "provider_race_handoff_blocked",
        "blockers": _dedupe(blockers),
        "provider_launch_request_path": "gpu_provider_launch_request.json",
        "prelaunch_spend_guard": prelaunch_guard or None,
        "provider_race": provider_race or None,
        "race_module": provider_race.get("race_module")
        or "blueprint_pipeline.provider_race",
        "provider_race_required_for_customer_path": race_required,
        "customer_path_provider_failover_handoff_wired": race_required,
        "customer_path_provider_failover_runtime_wired": runtime_wired,
        "customer_path_provider_failover_runtime_status": runtime_readiness.get("status"),
        "customer_path_provider_failover_runtime_blockers": _string_list(
            runtime_readiness.get("blockers")
        ),
        "provider_race_runtime_readiness": runtime_readiness,
        "provider_race_runtime_launcher_available": runtime_launcher_available,
        "provider_race_runtime_launcher_blockers": runtime_launcher_blockers,
        "provider_race_launcher_result_path": PROVIDER_RACE_LAUNCHER_RESULT_NAME
        if runtime_launcher_available
        else None,
        "customer_path_provider_failover_wired": runtime_wired,
        "live_provider_calls_performed": False,
        "serial_provider_launch_default_allowed": not race_required,
        "candidate_count": len(candidates),
        "race_candidate_count": len(race_candidates),
        "runnable_candidate_count": len(runnable_candidates),
        "runnable_candidates": runnable_candidates,
        "launcher_command": (
            f"{PROVIDER_RACE_LAUNCHER_COMMAND} "
            "--provider-launch-request gpu_provider_launch_request.json "
            "--handoff gpu_provider_race_handoff.json "
            "--allow-live-provider-race"
            if runtime_launcher_available
            else None
        ),
        "execution_contract": {
            "launch_every_candidate_requires_prelaunch_can_launch_true": True,
            "runtime_mode": "serial_adapter_failover" if runtime_wired else "blocked",
            "terminate_losers_required": True,
            "boot_marker_required_before_winner": True,
            "circuit_breaker_state_required": True,
            "teardown_owned_loser_cleanup_required": True,
            "fresh_job_bound_terminal_artifact_required_before_winner": True,
            "failed_adapter_must_emit_teardown_or_open_billing_risk": True,
            "write_provider_launcher_result_required": True,
            "provider_race_runtime_launcher_required": race_required,
            "provider_race_runtime_launcher_available": runtime_launcher_available,
            "provider_race_launcher_result_path": PROVIDER_RACE_LAUNCHER_RESULT_NAME
            if runtime_launcher_available
            else None,
            "live_failover_execution_proven": False,
        },
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "race_handoff_is_not_provider_execution": True,
            "provider_race_module_exists_but_live_race_not_executed": True,
            "provider_race_handoff_is_not_customer_runtime_failover": not runtime_wired,
            "provider_race_handoff_blocked_by_prelaunch_guard": (
                prelaunch_guard.get("required_before_provider_launch") is True
                and prelaunch_guard.get("can_launch") is not True
            ),
            "provider_race_runtime_launcher_not_implemented": not runtime_launcher_available,
            "provider_race_launcher_command_available": runtime_launcher_available,
            "parallel_provider_race_runtime_claimed": False,
            "teardown_owned_loser_cleanup_not_proven": True,
            "live_provider_calls_performed": False,
            "remote_cloud_execution_proven": False,
        },
    }


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
        else _number_field(budget, "budget_usd", "budgetUsd"),
        "timeout_seconds": timeout_seconds,
        "execution_allowed_by_default": False,
        "live_provider_calls_allowed_by_default": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _gpu_provisioning_result(
    *,
    request_manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    worker_launch_plan: Mapping[str, Any],
    provider_launch_request: Mapping[str, Any],
    allow_gpu_provisioning: bool,
    generated_at: str,
) -> Dict[str, Any]:
    provider = _string(request_manifest.get("provider")) or "fixture_local"
    scheduler_blockers = _string_list(scheduler_decision.get("blockers"))
    worker_blockers = _string_list(worker_launch_plan.get("blockers"))
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_GPU_PROVISIONING")
    approval_blockers: List[str] = []
    if provider != "fixture_local":
        if not env_allowed:
            approval_blockers.append("missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING")
        if not allow_gpu_provisioning:
            approval_blockers.append("missing_cli_allow_gpu_provisioning")
    if validation.get("status") == "blocked":
        return {
            "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "provider": provider,
            "status": "blocked",
            "reason": "job_validation_blocked",
            "blockers": _string_list(validation.get("blockers")),
            "worker_launch_plan_path": "worker_launch_plan.json",
            "worker_launch_plan_status": worker_launch_plan.get("status"),
            "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
            "gpu_provider_launch_request_status": provider_launch_request.get("status"),
            "execution_performed": False,
            "live_provider_calls_performed": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    if scheduler_blockers:
        blockers = _dedupe([*scheduler_blockers, *approval_blockers])
        return {
            "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "provider": provider,
            "status": "blocked",
            "reason": "scheduler_decision_blocked",
            "blockers": blockers,
            "scheduler_decision_path": "scheduler_decision.json",
            "worker_launch_plan_path": "worker_launch_plan.json",
            "worker_launch_plan_status": worker_launch_plan.get("status"),
            "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
            "gpu_provider_launch_request_status": provider_launch_request.get("status"),
            "execution_performed": False,
            "live_provider_calls_performed": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    if worker_blockers:
        blockers = _dedupe([*worker_blockers, *approval_blockers])
        return {
            "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "provider": provider,
            "status": "blocked",
            "reason": "worker_launch_plan_blocked",
            "blockers": blockers,
            "scheduler_decision_path": "scheduler_decision.json",
            "worker_launch_plan_path": "worker_launch_plan.json",
            "worker_launch_plan_status": worker_launch_plan.get("status"),
            "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
            "gpu_provider_launch_request_status": provider_launch_request.get("status"),
            "execution_performed": False,
            "live_provider_calls_performed": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    if provider_launch_request.get("status") == "blocked_by_startup_pipeline":
        return {
            "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "provider": provider,
            "status": "blocked",
            "reason": "provider_launch_request_blocked",
            "blockers": _string_list(provider_launch_request.get("blockers")),
            "scheduler_decision_path": "scheduler_decision.json",
            "worker_launch_plan_path": "worker_launch_plan.json",
            "worker_launch_plan_status": worker_launch_plan.get("status"),
            "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
            "gpu_provider_launch_request_status": provider_launch_request.get("status"),
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
            "worker_launch_plan_path": "worker_launch_plan.json",
            "worker_launch_plan_status": worker_launch_plan.get("status"),
            "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
            "gpu_provider_launch_request_status": provider_launch_request.get("status"),
            "cost_usd": 0.0,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    prelaunch_spend_guard = _mapping(provider_launch_request.get("prelaunch_spend_guard"))
    if prelaunch_spend_guard and prelaunch_spend_guard.get("can_launch") is not True:
        blockers = _dedupe(
            [
                "prelaunch_spend_guard_not_passed",
                *_string_list(prelaunch_spend_guard.get("blockers")),
                *approval_blockers,
            ]
        )
        return {
            "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "provider": provider,
            "status": "blocked",
            "reason": "prelaunch_spend_guard_blocked",
            "blockers": blockers,
            "prelaunch_spend_guard": prelaunch_spend_guard,
            "scheduler_decision_path": "scheduler_decision.json",
            "worker_launch_plan_path": "worker_launch_plan.json",
            "worker_launch_plan_status": worker_launch_plan.get("status"),
            "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
            "gpu_provider_launch_request_status": provider_launch_request.get("status"),
            "execution_performed": False,
            "live_provider_calls_performed": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    blockers = approval_blockers
    return {
        "schema_version": GPU_PROVISIONING_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "provider": provider,
        "status": "blocked" if blockers else "request_manifest_ready",
        "reason": "approval_required" if blockers else "explicitly_gated_request_ready",
        "blockers": blockers,
        "scheduler_decision_path": "scheduler_decision.json",
        "worker_launch_plan_path": "worker_launch_plan.json",
        "worker_launch_plan_status": worker_launch_plan.get("status"),
        "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
        "gpu_provider_launch_request_status": provider_launch_request.get("status"),
        "execution_performed": False,
        "live_provider_calls_performed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _gpu_cost_control_ledger(
    *,
    request: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    worker_launch_plan: Mapping[str, Any],
    provider_launch_request: Mapping[str, Any],
    gpu_result: Mapping[str, Any],
    sim_result: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    provider = _string(gpu_result.get("provider")) or _string(
        provider_launch_request.get("provider")
    ) or "fixture_local"
    job_id = _string(provider_launch_request.get("job_id")) or _string(gpu_result.get("job_id"))
    gpu_allocation = _mapping(scheduler_decision.get("gpu_allocation"))
    launch_mode = _mapping(worker_launch_plan.get("launch_mode"))
    cost_controls = _mapping(worker_launch_plan.get("cost_controls"))
    provider_shape = _mapping(provider_launch_request.get("provider_request_shape"))
    prelaunch_spend_guard = _mapping(provider_launch_request.get("prelaunch_spend_guard"))
    provider_limits = _mapping(provider_shape.get("limits"))
    artifact_finalizer = _mapping(provider_shape.get("artifact_finalizer"))
    gate_requirements = _mapping(provider_launch_request.get("gate_requirements"))
    budget = _mapping(request.get("budget"))
    hard_timeout_seconds = int(
        _number(
            provider_limits.get("hard_timeout_seconds")
            or launch_mode.get("hard_timeout_seconds")
            or gpu_allocation.get("hard_timeout_seconds")
            or budget.get("timeout_seconds")
            or budget.get("timeoutSeconds"),
            0,
        )
        or 0
    )
    requested_budget_usd = (
        _number(provider_limits.get("requested_budget_usd"))
        if provider_limits.get("requested_budget_usd") is not None
        else _number(gpu_allocation.get("requested_budget_usd"))
        if gpu_allocation.get("requested_budget_usd") is not None
        else _number_field(budget, "budget_usd", "budgetUsd")
    )
    external_watchdog_ttl_seconds = int(
        _number(
            provider_limits.get("external_watchdog_ttl_seconds")
            or launch_mode.get("external_watchdog_ttl_seconds"),
            hard_timeout_seconds + int(_number(launch_mode.get("shutdown_grace_seconds"), 60) or 60),
        )
        or 0
    )
    provider_blockers = _string_list(provider_launch_request.get("blockers"))
    prelaunch_blockers: List[str] = []
    external_provider = provider != "fixture_local"
    if prelaunch_spend_guard and prelaunch_spend_guard.get("can_launch") is not True:
        prelaunch_blockers.append("prelaunch_spend_guard_not_passed")
        prelaunch_blockers.extend(_string_list(prelaunch_spend_guard.get("blockers")))
    provisioning_blockers = _string_list(gpu_result.get("blockers"))
    scheduler_blockers = _string_list(scheduler_decision.get("blockers"))
    live_provider_calls = bool(
        provider_launch_request.get("live_provider_calls_performed")
        or gpu_result.get("live_provider_calls_performed")
        or sim_result.get("live_provider_calls_performed")
    )
    provider_shutdown_evidence = _mapping(
        gpu_result.get("provider_shutdown")
        or gpu_result.get("provider_shutdown_proof")
        or gpu_result.get("shutdown_proof")
    )
    provider_shutdown_proven = bool(
        gpu_result.get("provider_shutdown_proven")
        or gpu_result.get("clean_shutdown_proven")
        or gpu_result.get("zero_active_workers_after_run")
        or provider_shutdown_evidence.get("provider_shutdown_proven")
        or provider_shutdown_evidence.get("clean_shutdown_proven")
        or provider_shutdown_evidence.get("zero_active_workers_after_run")
        or provider_shutdown_evidence.get("pod_terminated")
        or provider_shutdown_evidence.get("worker_terminated")
    )
    actual_gpu_seconds: float | None = None
    actual_gpu_time_source = "not_observed"
    if provider == "fixture_local":
        actual_gpu_seconds = 0.0
        actual_gpu_time_source = "fixture_local_no_gpu"
    elif live_provider_calls and gpu_result.get("actual_gpu_seconds") is not None:
        actual_gpu_seconds = _number(gpu_result.get("actual_gpu_seconds"), 0.0)
        actual_gpu_time_source = "gpu_provisioning_result"
    blockers = _dedupe([
        *scheduler_blockers,
        *provider_blockers,
        *prelaunch_blockers,
        *provisioning_blockers,
    ])
    status = (
        "blocked_before_allocation"
        if blockers
        else "fixture_local_no_gpu"
        if provider == "fixture_local"
        else "ready_for_explicit_provider_launcher"
        if gpu_result.get("status") == "request_manifest_ready"
        else _string(gpu_result.get("status")) or "planned"
    )
    lifecycle_state = (
        "blocked-before-allocation"
        if status == "blocked_before_allocation"
        else "completed"
        if status in {"provider_runtime_observed", "fixture_local_no_gpu"}
        or sim_result.get("status") in {"completed", "simulator_command_completed"}
        else "failed"
        if _string(sim_result.get("status")).startswith("failed")
        or _string(gpu_result.get("status")).startswith("failed")
        else "running"
        if live_provider_calls and actual_gpu_seconds is None
        else "stopped"
        if _string(status) == "stopped"
        else "planned"
    )
    estimated_gpu_seconds = (
        0
        if status in {"blocked_before_allocation", "fixture_local_no_gpu"}
        else hard_timeout_seconds
    )
    return {
        "schema_version": GPU_COST_CONTROL_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "provider": provider,
        "status": status,
        "lifecycle_state": lifecycle_state,
        "supported_lifecycle_states": [
            "blocked-before-allocation",
            "running",
            "completed",
            "failed",
            "stopped",
        ],
        "scheduler_decision_path": "scheduler_decision.json",
        "worker_launch_plan_path": "worker_launch_plan.json",
        "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
        "gpu_provisioning_result_path": "gpu_provisioning_result.json",
        "simulator_service_result_path": "simulator_service_result.json",
        "live_provider_calls_performed": live_provider_calls,
        "execution_performed": bool(gpu_result.get("execution_performed")),
        "prelaunch_spend_guard": prelaunch_spend_guard or None,
        "budget": {
            "requested_budget_usd": requested_budget_usd,
            "budget_required_before_live_allocation": bool(
                cost_controls.get("budget_required_before_live_allocation")
            ),
            "gpu_spend_approved_by_webapp": bool(
                gpu_allocation.get("gpu_spend_approved_by_webapp")
            ),
            "allocation_allowed_by_webapp": bool(
                gpu_allocation.get("allocation_allowed_by_webapp")
            ),
        },
        "worker_limits": {
            "max_active_workers": int(
                _number(provider_limits.get("max_active_workers") or launch_mode.get("max_active_workers"), 1)
                or 1
            ),
            "customer_concurrency_limit_required": bool(
                cost_controls.get("customer_concurrency_limit_required")
            ),
            "hard_timeout_seconds": hard_timeout_seconds,
            "max_billable_gpu_seconds": hard_timeout_seconds,
            "idle_timeout_seconds": int(
                _number(
                    provider_limits.get("idle_timeout_seconds")
                    or launch_mode.get("idle_timeout_seconds"),
                    0,
                )
                or 0
            ),
            "idle_shutdown_required": bool(
                provider_limits.get("idle_shutdown_required")
                or launch_mode.get("idle_shutdown_required")
            ),
            "scale_to_zero_default": bool(provider_limits.get("scale_to_zero_default")),
            "external_watchdog_ttl_required": bool(
                provider_limits.get("external_watchdog_ttl_required")
                or launch_mode.get("external_watchdog_ttl_required")
            ),
            "external_watchdog_ttl_seconds": external_watchdog_ttl_seconds,
            "external_watchdog_owner": provider_limits.get("external_watchdog_owner")
            or launch_mode.get("external_watchdog_owner"),
        },
        "gpu_time": {
            "estimated_gpu_seconds": estimated_gpu_seconds,
            "actual_gpu_seconds": actual_gpu_seconds,
            "actual_gpu_time_source": actual_gpu_time_source,
            "actual_gpu_time_record_required": bool(
                artifact_finalizer.get("record_actual_gpu_time_required")
                or cost_controls.get("record_actual_gpu_time_required")
            ),
            "actual_gpu_time_record_present": actual_gpu_seconds is not None,
        },
        "artifact_finalizer": {
            "upload_before_shutdown_required": bool(
                artifact_finalizer.get("upload_before_shutdown_required")
                or cost_controls.get("finalizer_must_upload_artifacts_before_shutdown")
            ),
            "artifact_upload_contract_path": "worker_launch_plan.json",
            "shutdown_after_artifacts_required": external_provider,
            "worker_artifacts_finalized_before_shutdown": False,
            "provider_shutdown_proven": provider_shutdown_proven,
            "provider_shutdown_evidence": provider_shutdown_evidence,
        },
        "gate_requirements": {
            "env_BLUEPRINT_ALLOW_GPU_PROVISIONING_required": bool(
                gate_requirements.get("env_BLUEPRINT_ALLOW_GPU_PROVISIONING_required")
            ),
            "env_BLUEPRINT_ALLOW_GPU_PROVISIONING_present": bool(
                gate_requirements.get("env_BLUEPRINT_ALLOW_GPU_PROVISIONING_present")
            ),
            "cli_allow_gpu_provisioning_required": bool(
                gate_requirements.get("cli_allow_gpu_provisioning_required")
            ),
            "cli_allow_gpu_provisioning_present": bool(
                gate_requirements.get("cli_allow_gpu_provisioning_present")
            ),
            "provider_secret_values_in_artifact": False,
        },
        "blockers": blockers,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _remote_cloud_execution_closure_manifest(
    *,
    job_id: str,
    provisioner: str,
    simulator: str,
    worker_launch_plan: Mapping[str, Any],
    worker_manifest: Mapping[str, Any],
    provider_launch_request: Mapping[str, Any],
    gpu_result: Mapping[str, Any],
    gpu_cost_ledger: Mapping[str, Any],
    sim_result: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    external_provider = provisioner != "fixture_local"
    live_gpu_provider = provisioner in LIVE_GPU_PROVISIONERS and simulator != "fixture"
    worker_image = _mapping(worker_launch_plan.get("worker_image"))
    input_bundle = _mapping(worker_launch_plan.get("input_bundle"))
    manifest_input = _mapping(worker_launch_plan.get("worker_manifest_input_contract"))
    artifact_upload = _mapping(worker_launch_plan.get("artifact_upload_contract"))
    artifact_output_write_auth = _mapping(artifact_upload.get("artifact_output_write_auth"))
    launch_mode = _mapping(worker_launch_plan.get("launch_mode"))
    provider_input_setup = _mapping(provider_launch_request.get("provider_input_setup"))
    provider_input_setup_blockers = _string_list(provider_input_setup.get("blockers"))
    provider_package_validation = _mapping(
        provider_input_setup.get("provider_package_validation")
    )
    provider_inputs_uploaded = provider_input_setup.get("provider_inputs_uploaded")
    provider_input_setup_upload_ready = bool(
        not provider_input_setup or provider_inputs_uploaded is True
    )
    provider_shape = _mapping(provider_launch_request.get("provider_request_shape"))
    local_sim_only_prerequisite = _mapping(
        provider_shape.get("local_sim_only_prerequisite")
        or provider_launch_request.get("local_sim_only_prerequisite")
    )
    local_sim_only_prereq_blockers = _string_list(
        local_sim_only_prerequisite.get("blockers")
    )
    local_sim_only_prereq_required = bool(
        local_sim_only_prerequisite.get("required_before_provider_spend")
    )
    local_sim_only_prereq_ready = bool(
        not external_provider
        or not local_sim_only_prereq_required
        or (
            local_sim_only_prerequisite.get("status") == "passed"
            and local_sim_only_prerequisite.get("local_sim_only_evidence_clean")
            is True
        )
    )
    provider_inputs = _mapping(provider_shape.get("inputs"))
    provider_limits = _mapping(provider_shape.get("limits"))
    cost_budget = _mapping(gpu_cost_ledger.get("budget"))
    prelaunch_spend_guard = _mapping(
        gpu_cost_ledger.get("prelaunch_spend_guard")
        or provider_launch_request.get("prelaunch_spend_guard")
    )
    worker_limits = _mapping(gpu_cost_ledger.get("worker_limits"))
    gpu_time = _mapping(gpu_cost_ledger.get("gpu_time"))
    artifact_finalizer = _mapping(gpu_cost_ledger.get("artifact_finalizer"))

    def upload_evidence_complete(evidence: Mapping[str, Any]) -> bool:
        if evidence.get("status") != "completed":
            return False
        count = _number(
            evidence.get("uploaded_file_count")
            or evidence.get("copied_file_count")
            or evidence.get("artifact_count"),
            0,
        )
        return bool(
            (count is not None and count > 0)
            or _string_list(evidence.get("object_keys"))
            or _string_list(evidence.get("relative_paths"))
            or _string(evidence.get("object_key"))
            or _string(evidence.get("destination_path"))
        )

    image_ready = (
        not live_gpu_provider
        or (
            bool(worker_image.get("configured_image_ref_present"))
            and bool(worker_image.get("configured_image_ref_is_versioned"))
            and bool(worker_image.get("configured_image_ref_fetchable_by_provider"))
        )
    )
    worker_manifest_ready = (
        not external_provider
        or (
            bool(manifest_input.get("configured_worker_manifest_uri_present"))
            and bool(manifest_input.get("worker_manifest_uri_fetchable_by_provider"))
            and _string(worker_manifest.get("status")) == "ready_for_worker_upload"
        )
    )
    capture_bundle_ready = (
        not live_gpu_provider
        or bool(input_bundle.get("capture_root_bundle_uri_fetchable_by_provider"))
    )
    artifact_output_ready = (
        not external_provider
        or bool(artifact_upload.get("configured_artifact_output_uri_present"))
    )
    artifact_output_provider_writable = (
        not external_provider
        or bool(artifact_upload.get("artifact_output_uri_provider_writable"))
    )
    artifact_output_write_auth_ready = (
        not external_provider
        or bool(
            artifact_upload.get("artifact_output_write_auth_contract_ready")
            or artifact_output_write_auth.get("write_auth_contract_ready")
        )
    )
    requested_budget = cost_budget.get("requested_budget_usd")
    budget_ready = not external_provider or requested_budget is not None
    hard_timeout_seconds = int(
        _number(
            provider_limits.get("hard_timeout_seconds")
            or worker_limits.get("hard_timeout_seconds")
            or launch_mode.get("hard_timeout_seconds"),
            0,
        )
        or 0
    )
    timeout_ready = hard_timeout_seconds > 0
    shutdown_required = bool(
        worker_limits.get("idle_shutdown_required") or provider_limits.get("idle_shutdown_required")
    )
    upload_before_shutdown_required = bool(
        artifact_finalizer.get("upload_before_shutdown_required")
        or artifact_upload.get("upload_before_shutdown_required")
    )
    provider_request_ready = (
        not external_provider
        or (
            _string(provider_launch_request.get("status")) == "request_manifest_ready"
            and not provider_input_setup_blockers
            and provider_input_setup_upload_ready
            and local_sim_only_prereq_ready
        )
    )
    live_provider_calls = (
        _strict_bool(gpu_cost_ledger.get("live_provider_calls_performed"))
        or _strict_bool(gpu_result.get("live_provider_calls_performed"))
    )
    actual_gpu_time_present = _strict_bool(gpu_time.get("actual_gpu_time_record_present"))
    runtime_observed = _string(gpu_cost_ledger.get("status")) == "provider_runtime_observed"
    simulator_execution_proven = _strict_bool(sim_result.get("simulator_execution_proven"))
    provider_shutdown_evidence = _mapping(artifact_finalizer.get("provider_shutdown_evidence"))
    artifact_upload_evidence = _mapping(artifact_finalizer.get("artifact_upload_evidence"))
    finalizer_refresh_upload_evidence = _mapping(
        artifact_finalizer.get("finalizer_refresh_upload_evidence")
    )
    runtime_manifest_upload_evidence = _mapping(
        artifact_finalizer.get("runtime_manifest_upload_evidence")
    )
    provider_package_validation_blockers = _string_list(
        provider_package_validation.get("blockers")
    )
    provider_package_validation_status = _string(provider_package_validation.get("status"))
    package_validated_without_spend = bool(
        provider_package_validation_status.startswith("validated")
    )
    worker_finalizer_proven = (
        _strict_bool(artifact_finalizer.get("worker_artifacts_finalized_before_shutdown"))
        or _strict_bool(artifact_finalizer.get("worker_finalizer_completed_before_shutdown"))
    )
    artifact_upload_evidence_complete = bool(
        not (external_provider and live_provider_calls)
        or (
            upload_evidence_complete(artifact_upload_evidence)
            and upload_evidence_complete(finalizer_refresh_upload_evidence)
            and upload_evidence_complete(runtime_manifest_upload_evidence)
        )
    )
    provider_shutdown_proven = (
        _strict_bool(artifact_finalizer.get("provider_shutdown_proven"))
        or _strict_bool(provider_shutdown_evidence.get("provider_shutdown_proven"))
        or _strict_bool(provider_shutdown_evidence.get("clean_shutdown_proven"))
        or _strict_bool(provider_shutdown_evidence.get("zero_active_workers_after_run"))
        or _strict_bool(provider_shutdown_evidence.get("pod_terminated"))
        or _strict_bool(provider_shutdown_evidence.get("worker_terminated"))
    )
    clean_shutdown_proven = bool(
        external_provider
        and live_provider_calls
        and runtime_observed
        and actual_gpu_time_present
        and upload_before_shutdown_required
        and worker_finalizer_proven
        and artifact_upload_evidence_complete
        and provider_shutdown_proven
    )
    remote_cloud_execution_proven = bool(
        external_provider
        and live_provider_calls
        and simulator_execution_proven
        and actual_gpu_time_present
    )
    contract_blockers: List[str] = []
    if not image_ready:
        contract_blockers.append("remote_worker_image_not_pinned_or_fetchable")
    if not worker_manifest_ready:
        contract_blockers.append("remote_worker_manifest_uri_not_fetchable")
    if not capture_bundle_ready:
        contract_blockers.append("remote_capture_root_bundle_uri_not_fetchable")
    if not artifact_output_ready:
        contract_blockers.append("remote_artifact_output_uri_missing")
    if artifact_output_ready and not artifact_output_provider_writable:
        contract_blockers.append("remote_artifact_output_uri_not_provider_writable")
    if (
        artifact_output_ready
        and artifact_output_provider_writable
        and not artifact_output_write_auth_ready
    ):
        contract_blockers.append("remote_artifact_output_write_auth_contract_missing")
    if not budget_ready:
        contract_blockers.append("remote_budget_not_declared")
    if not timeout_ready:
        contract_blockers.append("remote_timeout_not_declared")
    if external_provider and not shutdown_required:
        contract_blockers.append("remote_idle_shutdown_not_required")
    if external_provider and not upload_before_shutdown_required:
        contract_blockers.append("remote_artifact_upload_before_shutdown_not_required")
    if not provider_request_ready:
        contract_blockers.append("remote_provider_launch_request_not_ready")
    if prelaunch_spend_guard and prelaunch_spend_guard.get("can_launch") is not True:
        contract_blockers.append("remote_prelaunch_spend_guard_not_passed")
        contract_blockers.extend(
            f"prelaunch_spend_guard:{blocker}"
            for blocker in _string_list(prelaunch_spend_guard.get("blockers"))
        )
    if not local_sim_only_prereq_ready:
        contract_blockers.append("remote_local_sim_only_prerequisite_not_passed")
    contract_blockers.extend(
        f"local_sim_only_prerequisite:{blocker}"
        for blocker in local_sim_only_prereq_blockers
    )
    contract_blockers.extend(
        f"provider_package_validation:{blocker}"
        for blocker in provider_package_validation_blockers
    )
    contract_blockers.extend(
        f"provider_input_setup:{blocker}" for blocker in provider_input_setup_blockers
    )
    if provider_input_setup and provider_inputs_uploaded is not True:
        blocker = "provider_input_setup:provider_inputs_upload_not_proven"
        if blocker not in contract_blockers:
            contract_blockers.append(blocker)
    runtime_blockers: List[str] = []
    if external_provider and not live_provider_calls:
        runtime_blockers.append("remote_provider_runtime_not_executed")
    if external_provider and live_provider_calls and not actual_gpu_time_present:
        runtime_blockers.append("remote_actual_gpu_time_not_recorded")
    if external_provider and live_provider_calls and not clean_shutdown_proven:
        runtime_blockers.append("remote_clean_shutdown_not_proven")
    if external_provider and live_provider_calls and not worker_finalizer_proven:
        runtime_blockers.append("remote_worker_finalizer_not_proven")
    if external_provider and live_provider_calls and not artifact_upload_evidence_complete:
        runtime_blockers.append("remote_artifact_upload_evidence_incomplete")
    if external_provider and live_provider_calls and not provider_shutdown_proven:
        runtime_blockers.append("remote_provider_shutdown_not_proven")
    if not external_provider:
        status = "not_required_for_local_execution"
    elif contract_blockers:
        status = "blocked_before_remote_execution"
    elif remote_cloud_execution_proven and clean_shutdown_proven:
        status = "remote_execution_completed_with_shutdown_proof"
    elif remote_cloud_execution_proven:
        status = "remote_execution_completed_missing_shutdown_proof"
    else:
        status = "ready_for_explicit_provider_runtime"
    runtime_phase = (
        _string(gpu_result.get("phase"))
        or _string(gpu_result.get("status"))
        or _string(gpu_cost_ledger.get("phase"))
        or _string(gpu_cost_ledger.get("status"))
        or _string(provider_launch_request.get("status"))
        or status
    )
    pod_id = (
        _string(gpu_result.get("pod_id"))
        or _string(gpu_result.get("provider_pod_id"))
        or _string(provider_shutdown_evidence.get("pod_id"))
    )
    provider_job_id = (
        _string(gpu_result.get("provider_job_id"))
        or _string(provider_launch_request.get("provider_job_id"))
        or job_id
    )
    provider_runtime_started_at = (
        _string(gpu_result.get("provider_runtime_started_at"))
        or _string(gpu_result.get("started_at"))
        or _string(gpu_result.get("start_time"))
        or _string(gpu_cost_ledger.get("provider_runtime_started_at"))
        or _string(gpu_cost_ledger.get("started_at"))
        or _string(gpu_cost_ledger.get("start_time"))
    )
    max_wait_seconds_value = _number(
        gpu_result.get("max_wait_seconds")
        or gpu_cost_ledger.get("max_wait_seconds")
        or worker_limits.get("max_wait_seconds")
        or provider_limits.get("max_wait_seconds")
        or launch_mode.get("max_wait_seconds")
        or worker_limits.get("external_watchdog_ttl_seconds")
        or provider_limits.get("external_watchdog_ttl_seconds")
        or launch_mode.get("external_watchdog_ttl_seconds"),
        None,
    )
    max_wait_seconds = (
        int(max_wait_seconds_value) if max_wait_seconds_value is not None else None
    )

    def first_size_bytes(*payloads: Mapping[str, Any]) -> int | None:
        for payload in payloads:
            for key in (
                "uploaded_size_bytes",
                "copied_size_bytes",
                "object_size_bytes",
                "size_bytes",
                "total_size_bytes",
            ):
                value = _number(payload.get(key), None)
                if value is not None and value >= 0:
                    return int(value)
        return None

    output_object_size_bytes = first_size_bytes(
        artifact_upload_evidence,
        finalizer_refresh_upload_evidence,
        runtime_manifest_upload_evidence,
    )
    teardown_status = (
        "not_required_for_local_execution"
        if not external_provider
        else "clean_shutdown_proven"
        if clean_shutdown_proven
        else "teardown_not_proven"
    )
    continuing_spend_from_this_run = bool(
        external_provider and live_provider_calls and not provider_shutdown_proven
    )
    watchdog_boundary = {
        "hard_timeout_seconds": hard_timeout_seconds,
        "idle_timeout_seconds": worker_limits.get("idle_timeout_seconds"),
        "external_watchdog_ttl_seconds": worker_limits.get(
            "external_watchdog_ttl_seconds"
        )
        or provider_limits.get("external_watchdog_ttl_seconds")
        or launch_mode.get("external_watchdog_ttl_seconds"),
        "max_billable_gpu_seconds": worker_limits.get("max_billable_gpu_seconds"),
        "watchdog_owner": worker_limits.get("external_watchdog_owner")
        or provider_limits.get("external_watchdog_owner")
        or launch_mode.get("external_watchdog_owner"),
    }
    artifact_manifest = {
        "artifact_output_uri": artifact_upload.get("configured_artifact_output_uri"),
        "artifact_upload_evidence_complete": artifact_upload_evidence_complete,
        "artifact_upload_evidence": artifact_upload_evidence,
        "finalizer_refresh_upload_evidence": finalizer_refresh_upload_evidence,
        "runtime_manifest_upload_evidence": runtime_manifest_upload_evidence,
    }
    return {
        "schema_version": REMOTE_CLOUD_EXECUTION_CLOSURE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "provider": provisioner,
        "simulator": simulator,
        "status": status,
        "provider_input_setup": {
            "status": provider_input_setup.get("status"),
            "manifest_path": provider_input_setup.get("manifest_path"),
            "provider_inputs_uploaded": provider_input_setup.get(
                "provider_inputs_uploaded"
            ),
            "image_ref_published_proven": provider_input_setup.get(
                "image_ref_published_proven"
            ),
            "blockers": provider_input_setup_blockers,
            "artifact_output_uri": provider_input_setup.get("artifact_output_uri"),
            "capture_root_bundle_uri": provider_input_setup.get("capture_root_bundle_uri"),
            "worker_manifest_uri": provider_input_setup.get("worker_manifest_uri"),
        }
        if provider_input_setup
        else {},
        "contract_ready_for_remote_runtime": bool(external_provider and not contract_blockers),
        "remote_cloud_execution_proven": remote_cloud_execution_proven,
        "clean_shutdown_proven": clean_shutdown_proven,
        "live_provider_calls_performed": live_provider_calls,
        "phase": runtime_phase,
        "provider_job_id": provider_job_id,
        "pod_id": pod_id or None,
        "provider_runtime_started_at": provider_runtime_started_at or None,
        "max_wait_seconds": max_wait_seconds,
        "watchdog_boundary": watchdog_boundary,
        "max_spend_usd": requested_budget,
        "output_uri": artifact_upload.get("configured_artifact_output_uri"),
        "output_object_size_bytes": output_object_size_bytes,
        "output_zip_or_object_size_bytes": output_object_size_bytes,
        "artifact_manifest": artifact_manifest,
        "teardown_status": teardown_status,
        "continuing_spend_from_this_run": continuing_spend_from_this_run,
        "provider_package": {
            "job_id": job_id,
            "provider_launch_request_job_id": provider_launch_request.get("job_id"),
            "capture_root": _mapping(provider_package_validation.get("exact_ids")).get(
                "capture_root"
            )
            or worker_manifest.get("capture_root"),
            "capture_root_bundle_uri": input_bundle.get("capture_root_bundle_uri"),
            "worker_manifest_uri": manifest_input.get("configured_worker_manifest_uri"),
            "artifact_output_uri": artifact_upload.get("configured_artifact_output_uri"),
            "provider_package_validation_status": provider_package_validation_status
            or None,
            "provider_package_validated_without_spend": package_validated_without_spend,
            "local_sim_only_prerequisite_status": local_sim_only_prerequisite.get(
                "status"
            ),
            "local_sim_only_evidence_clean": local_sim_only_prerequisite.get(
                "local_sim_only_evidence_clean"
            ),
        },
        "local_sim_only_prerequisite": local_sim_only_prerequisite,
        "prelaunch_spend_guard": prelaunch_spend_guard or None,
        "provider_package_validation": provider_package_validation,
        "checks": {
            "versioned_worker_image_ref_pinned": image_ready,
            "worker_manifest_uri_fetchable": worker_manifest_ready,
            "capture_root_bundle_uri_fetchable": capture_bundle_ready,
            "artifact_output_uri_configured": artifact_output_ready,
            "artifact_output_uri_provider_writable": artifact_output_provider_writable,
            "artifact_output_write_auth_contract_ready": artifact_output_write_auth_ready,
            "budget_declared": budget_ready,
            "hard_timeout_declared": timeout_ready,
            "idle_shutdown_required": shutdown_required,
            "artifact_upload_before_shutdown_required": upload_before_shutdown_required,
            "provider_launch_request_ready": provider_request_ready,
            "prelaunch_spend_guard_can_launch": bool(
                prelaunch_spend_guard.get("can_launch")
            ) if prelaunch_spend_guard else None,
            "local_sim_only_prerequisite_ready": local_sim_only_prereq_ready,
            "provider_package_validated_without_spend": package_validated_without_spend,
            "actual_gpu_time_record_present": actual_gpu_time_present,
            "simulator_execution_proven": simulator_execution_proven,
            "worker_finalizer_proven": worker_finalizer_proven,
            "artifact_upload_evidence_complete": artifact_upload_evidence_complete,
            "provider_shutdown_proven": provider_shutdown_proven,
        },
        "inputs": {
            "worker_manifest_uri": manifest_input.get("configured_worker_manifest_uri"),
            "worker_manifest_uri_scheme": manifest_input.get("worker_manifest_uri_scheme"),
            "capture_root_bundle_uri": input_bundle.get("capture_root_bundle_uri"),
            "capture_root_bundle_uri_scheme": input_bundle.get(
                "capture_root_bundle_uri_scheme"
            ),
            "provider_manifest_uri": provider_inputs.get("manifest_uri"),
            "provider_capture_root_bundle_uri": provider_inputs.get("capture_root_bundle_uri"),
        },
        "outputs": {
            "artifact_output_uri": artifact_upload.get("configured_artifact_output_uri"),
            "artifact_output_uri_scheme": artifact_upload.get("artifact_output_uri_scheme"),
            "artifact_output_uri_provider_writable": artifact_output_provider_writable,
            "artifact_output_write_auth": artifact_output_write_auth,
            "artifact_output_write_auth_contract_ready": artifact_output_write_auth_ready,
            "artifact_output_uri_env_var": artifact_upload.get("artifact_output_uri_env_var"),
            "upload_before_shutdown_required": upload_before_shutdown_required,
            "worker_artifacts_finalized_before_shutdown": worker_finalizer_proven,
            "artifact_upload_evidence_complete": artifact_upload_evidence_complete,
            "artifact_upload_evidence": artifact_upload_evidence,
            "finalizer_refresh_upload_evidence": finalizer_refresh_upload_evidence,
            "runtime_manifest_upload_evidence": runtime_manifest_upload_evidence,
            "provider_shutdown_proven": provider_shutdown_proven,
            "provider_shutdown_evidence": provider_shutdown_evidence,
        },
        "runtime_tracking": {
            "phase": runtime_phase,
            "provider_job_id": provider_job_id,
            "pod_id": pod_id or None,
            "provider_runtime_started_at": provider_runtime_started_at or None,
            "max_wait_seconds": max_wait_seconds,
            "watchdog_boundary": watchdog_boundary,
            "output_object_size_bytes": output_object_size_bytes,
            "output_zip_or_object_size_bytes": output_object_size_bytes,
            "artifact_manifest": artifact_manifest,
            "teardown_status": teardown_status,
            "continuing_spend_from_this_run": continuing_spend_from_this_run,
        },
        "cost_and_timeout_controls": {
            "requested_budget_usd": requested_budget,
            "hard_timeout_seconds": hard_timeout_seconds,
            "idle_timeout_seconds": worker_limits.get("idle_timeout_seconds"),
            "max_billable_gpu_seconds": worker_limits.get("max_billable_gpu_seconds"),
            "actual_gpu_seconds": gpu_time.get("actual_gpu_seconds"),
            "actual_gpu_time_record_present": actual_gpu_time_present,
        },
        "contract_blockers": contract_blockers,
        "runtime_blockers": runtime_blockers,
        "blockers": _dedupe([*contract_blockers, *runtime_blockers]),
        "artifact_paths": {
            "worker_launch_plan": "worker_launch_plan.json",
            "worker_manifest": "worker_manifest.json",
            "gpu_provider_launch_request": "gpu_provider_launch_request.json",
            "gpu_provisioning_result": "gpu_provisioning_result.json",
            "gpu_cost_control_ledger": "gpu_cost_control_ledger.json",
            "simulator_service_result": "simulator_service_result.json",
        },
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "remote_cloud_execution_proven": remote_cloud_execution_proven,
            "clean_provider_shutdown_proven": clean_shutdown_proven,
            "provider_details_exposed_to_webapp": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _simulator_request(
    *,
    job_id: str,
    simulator: str,
    request: Mapping[str, Any],
    scenario_eval_matrix: Mapping[str, Any],
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
        "scenario_eval_matrix_path": "scenario_eval_matrix.json",
        "scenario_eval_run_count": int(scenario_eval_matrix.get("scenario_eval_run_count") or 0),
        "required_variation_names": list(
            scenario_eval_matrix.get("required_variation_names") or []
        ),
        "variation_names_covered": list(scenario_eval_matrix.get("variation_names_covered") or []),
        "scenario_variation_instances_path": (
            "../simulation_automation/scenario_variation_instances.json"
            if scenario_eval_matrix.get("variation_instance_count")
            else None
        ),
        "robot_profile": _mapping(request.get("robot_profile") or request.get("robotProfile")),
        "timeout_seconds": timeout_seconds,
        "execution_allowed_by_default": False,
        "fixture_backend_proves_local_loop_only": simulator == "fixture",
        "arena_environment_packet_path": (
            "../simulation_automation/arena_environment_packet.json"
            if simulator == "isaac_lab_arena"
            else None
        ),
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


def _command_executable(command_text: str) -> str | None:
    if not command_text:
        return None
    try:
        command = shlex.split(command_text)
    except ValueError:
        return None
    return command[0] if command else None


def _write_simulator_provider_adapter_manifest(
    *,
    job_dir: Path,
    simulator: str,
    status: str,
    blockers: Sequence[str],
    allow_simulator_execution: bool,
    env_allows_simulator_execution: bool,
    allowed_simulators: Sequence[str],
    command_text: str,
    artifact_paths: Mapping[str, Any],
    generated_at: str,
    simulator_output_ingested: bool = False,
    reason: str | None = None,
) -> Dict[str, Any]:
    profile = dict(SIMULATOR_PROVIDER_PROFILES.get(simulator, {}))
    command_configured = bool(command_text)
    manifest = {
        "schema_version": SIMULATOR_PROVIDER_ADAPTER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "simulator": simulator,
        "status": status,
        "reason": reason,
        "provider_profile": profile,
        "plugin_contract": {
            "env": {
                "BLUEPRINT_SIMULATOR_OUTPUT": "path where provider command writes JSON output",
                "BLUEPRINT_CAPTURE_ROOT": "capture root used by the simulator adapter",
                "BLUEPRINT_SIMULATOR_FRAMEWORK": "selected simulator framework id",
            },
            "accepted_output_shapes": [
                "attempts[]",
                "records[]",
                "outcomes[]",
                "single attempt object",
            ],
            "normalized_outputs": [
                "normalized_attempt_trace.json",
                "failure_labels.json",
                "prediction_outcome_ledger.json",
                "calibration_report.json",
                "breakage_library.json",
            ],
        },
        "gates": {
            "env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": env_allows_simulator_execution,
            "allow_simulator_execution_flag": bool(allow_simulator_execution),
            "simulator_allowlisted": simulator == "fixture" or simulator in set(allowed_simulators),
            "command_configured": command_configured,
            "blockers": list(blockers),
        },
        "command_ref": {
            "configured": command_configured,
            "sha256": sha256(command_text.encode("utf-8")).hexdigest()
            if command_configured
            else None,
            "executable": _command_executable(command_text),
        },
        "normalization": {
            "simulator_output_ingested": bool(simulator_output_ingested),
            "artifact_paths": dict(artifact_paths),
            "deterministic_manifests_are_proof_source": True,
        },
        "owner_system_review_required": simulator != "fixture",
        "simulator_execution_proven": status == "completed" and simulator != "fixture",
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(job_dir / "simulator_provider_adapter_manifest.json", manifest)
    return manifest


def _attach_simulator_provider_manifest(result: Mapping[str, Any]) -> Dict[str, Any]:
    artifact_paths = {
        **_mapping(result.get("artifact_paths")),
        "simulator_provider_adapter_manifest": "simulator_provider_adapter_manifest.json",
    }
    return {
        **dict(result),
        "provider_adapter_manifest_path": "simulator_provider_adapter_manifest.json",
        "artifact_paths": artifact_paths,
    }


def _run_command_simulator(
    *,
    simulator: str,
    command_text: str,
    timeout_seconds: int,
    generated_at: str,
    output_path: Path | None = None,
    capture_root: Path | None = None,
    scenario_eval_matrix_path: Path | None = None,
) -> Dict[str, Any]:
    command = shlex.split(command_text)
    if output_path is not None:
        ensure_dir(output_path.parent)
    env = os.environ.copy()
    if output_path is not None:
        env["BLUEPRINT_SIMULATOR_OUTPUT"] = str(output_path)
    if capture_root is not None:
        env["BLUEPRINT_CAPTURE_ROOT"] = str(capture_root)
    if scenario_eval_matrix_path is not None:
        env["BLUEPRINT_SCENARIO_EVAL_MATRIX"] = str(scenario_eval_matrix_path)
    env["BLUEPRINT_SIMULATOR_FRAMEWORK"] = simulator
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
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
            "simulator_output_path": str(output_path) if output_path else None,
            "simulators_run": True,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    status = "completed" if completed.returncode == 0 else "failed"
    simulator_output_payload = None
    if output_path is not None and output_path.is_file():
        try:
            simulator_output_payload = read_json_any(output_path)
        except (OSError, json.JSONDecodeError, ValueError):
            simulator_output_payload = None
    if not simulator_output_payload and completed.stdout.strip().startswith(("{", "[")):
        try:
            simulator_output_payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            simulator_output_payload = None
    output_mapping = _mapping(simulator_output_payload)
    output_declares_execution = output_mapping.get("simulator_execution_proven")
    invalid_execution_proof_claim = (
        "simulator_execution_proven" in output_mapping
        and output_declares_execution is not True
    )
    if status == "completed" and invalid_execution_proof_claim:
        status = "blocked"
    isaac_sim_execution_proven = _strict_bool(output_mapping.get("isaac_sim_execution_proven"))
    isaac_robot_asset_execution_proven = _strict_bool(
        output_mapping.get("isaac_robot_asset_execution_proven")
    )
    unitree_g1_asset_spawned = (
        _strict_bool(output_mapping.get("unitree_g1_asset_spawned"))
        or _strict_bool(output_mapping.get("unitree_g1_robot_asset_spawned"))
    )
    return {
        "schema_version": SIMULATOR_SERVICE_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "framework": simulator,
        "status": status,
        "reason": None
        if status == "completed"
        else "simulator_output_declared_execution_not_proven"
        if invalid_execution_proof_claim
        else f"simulator_exit_code:{completed.returncode}",
        "blockers": []
        if status == "completed"
        else ["simulator_output_declared_execution_not_proven"]
        if invalid_execution_proof_claim
        else [f"simulator_exit_code:{completed.returncode}"],
        "command": command,
        "raw_command": command_text,
        "execution_performed": True,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
        "artifact_paths": {
            "simulator_output": str(output_path) if output_path and output_path.is_file() else None
        },
        "simulator_output_path": str(output_path) if output_path else None,
        "simulator_output_payload": simulator_output_payload,
        "simulators_run": True,
        "simulator_execution_proven": status == "completed",
        "isaac_sim_execution_proven": status == "completed" and isaac_sim_execution_proven,
        "isaac_robot_asset_execution_proven": (
            status == "completed" and isaac_robot_asset_execution_proven
        ),
        "unitree_g1_asset_spawned": status == "completed" and unitree_g1_asset_spawned,
        "robot_policy_execution_proven": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulator_execution_proven": status == "completed",
            "isaac_sim_execution_proven": status == "completed" and isaac_sim_execution_proven,
            "isaac_robot_asset_execution_proven": (
                status == "completed" and isaac_robot_asset_execution_proven
            ),
        },
    }


def _copy_site_eval_artifacts(
    *, pipeline_dir: Path, job_dir: Path, generated_at: str
) -> Dict[str, Dict[str, Any]]:
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




def _attempt_for_matrix_run(
    *,
    attempts: Sequence[Mapping[str, Any]],
    matrix_run: Mapping[str, Any],
    fallback_index: int,
) -> Mapping[str, Any]:
    task_id = _string(matrix_run.get("task_id") or matrix_run.get("taskId"))
    scenario_id = _string(matrix_run.get("scenario_id") or matrix_run.get("scenarioId"))
    for attempt in attempts:
        if _string(attempt.get("scenario_id") or attempt.get("scenarioId")) == scenario_id and (
            not task_id or _string(attempt.get("task_id") or attempt.get("taskId")) == task_id
        ):
            return attempt
    return attempts[fallback_index % len(attempts)]


def _expand_fixture_artifacts_to_scenario_eval_runs(
    *,
    copied_artifacts: Mapping[str, Mapping[str, Any]],
    scenario_eval_matrix: Mapping[str, Any],
    job_dir: Path,
    generated_at: str,
) -> Dict[str, Dict[str, Any]]:
    matrix_runs = _scenario_eval_matrix_runs(scenario_eval_matrix)
    copied = {key: dict(value) for key, value in copied_artifacts.items()}
    trace = _mapping(copied.get("normalized_attempt_trace"))
    attempts = [item for item in trace.get("attempts", []) or [] if isinstance(item, Mapping)]
    if _string(trace.get("runner")) != "fixture" or not matrix_runs or not attempts:
        return copied

    required_run_ids = sorted(_string(run.get("scenario_eval_run_id")) for run in matrix_runs)
    covered_run_ids = sorted(
        {
            _string(attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId"))
            for attempt in attempts
            if _string(attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId"))
        }
    )
    if set(required_run_ids).issubset(set(covered_run_ids)):
        return copied

    expanded_attempts: List[Dict[str, Any]] = []
    for index, matrix_run in enumerate(matrix_runs):
        source = _attempt_for_matrix_run(
            attempts=attempts,
            matrix_run=matrix_run,
            fallback_index=index,
        )
        run_id = _string(matrix_run.get("scenario_eval_run_id"))
        source_attempt_id = (
            _string(source.get("attempt_id") or source.get("attemptId")) or "fixture_attempt"
        )
        expanded = {
            **dict(source),
            "attempt_id": f"{source_attempt_id}__{run_id}",
            "scenario_eval_run_id": run_id,
            "scenario_variation_instance_id": _string(
                matrix_run.get("scenario_variation_instance_id")
                or matrix_run.get("scenarioVariationInstanceId")
            )
            or None,
            "variation_name": _string(
                matrix_run.get("variation_name") or matrix_run.get("variationName")
            )
            or None,
            "task_id": _string(matrix_run.get("task_id") or matrix_run.get("taskId")),
            "scenario_id": _string(matrix_run.get("scenario_id") or matrix_run.get("scenarioId")),
            "matrix_run_source": {
                "scenario_eval_run_id": run_id,
                "baseline_capture_layout": bool(matrix_run.get("baseline_capture_layout")),
            },
            "claim_boundary": (
                "fixture_attempt_expanded_to_scenario_eval_run_contract_not_real_simulator_rollout"
            ),
        }
        expanded_attempts.append(expanded)

    failures = [
        attempt
        for attempt in expanded_attempts
        if coerce_strict_success(attempt.get("success")) is not True
    ]
    expanded_trace = {
        **dict(trace),
        "schema_version": trace.get("schema_version") or NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION,
        "generated_at": trace.get("generated_at") or generated_at,
        "status": "completed",
        "attempt_count": len(expanded_attempts),
        "scenario_eval_run_count": len(matrix_runs),
        "covered_scenario_eval_run_ids": required_run_ids,
        "missing_scenario_eval_run_ids": [],
        "scenario_eval_run_coverage_complete": True,
        "expanded_from_site_eval_fixture_attempts": True,
        "attempts": expanded_attempts,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    expanded_failure_label_rows: List[Dict[str, Any]] = []
    expanded_nonreviewable_label_ids: List[str] = []
    for index, attempt in enumerate(failures, start=1):
        label_id = f"fixture_label_{index:04d}"
        failure_mode_ids = _string_list(attempt.get("failure_mode_ids"))
        frame_refs = _failure_frame_or_clip_refs(attempt)
        source_trace_refs = _dedupe_refs(
            [
                "normalized_attempt_trace.json",
                *_string_list(attempt.get("source_trace_refs")),
            ]
        )
        evidence_refs = _failure_evidence_refs(
            attempt,
            extra_refs=tuple(source_trace_refs),
        )
        review_status = _failure_review_status(
            supplied_review_status=attempt.get("review_status"),
            supplied_status=attempt.get("status"),
            generated_rollout=True,
            frame_or_clip_ref_count=len(frame_refs),
        )
        root_cause_category = _failure_root_cause_category(
            failure_mode_ids,
            failure_reason=_string(attempt.get("failure_reason")) or None,
        )
        if review_status == "non_reviewable_failure_hypothesis":
            expanded_nonreviewable_label_ids.append(label_id)
        expanded_failure_label_rows.append(
            {
                "label_id": label_id,
                "attempt_id": attempt.get("attempt_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
                "variation_name": attempt.get("variation_name"),
                "task_id": attempt.get("task_id"),
                "scenario_id": attempt.get("scenario_id"),
                "policy_id": attempt.get("policy_id"),
                "failure_mode_ids": failure_mode_ids,
                "failure_reason": _string(attempt.get("failure_reason")) or None,
                "source": "site_eval_fixture_expanded_to_scenario_eval_matrix",
                "evidence_refs": evidence_refs,
                "source_trace_refs": source_trace_refs,
                "frame_or_clip_refs": frame_refs,
                "visual_smoke_ref": attempt.get("visual_smoke_ref")
                or attempt.get("visualSmokeRef"),
                "confidence": attempt.get("confidence"),
                "status": "review_required",
                "review_status": review_status,
                "reviewer_acceptance_required": True,
                "root_cause_category": root_cause_category,
                "remediation_candidate": _failure_remediation_candidate(
                    root_cause_category,
                    failure_mode_ids,
                ),
                "unknown_when_evidence_weak": bool(
                    not frame_refs
                    or not evidence_refs
                    or review_status == "non_reviewable_failure_hypothesis"
                ),
                "non_reviewable_failure_hypothesis": (
                    review_status == "non_reviewable_failure_hypothesis"
                ),
                "generated_wam_rollout": True,
                "model_derived_support_artifact": True,
                "proof_effect": FAILURE_LABEL_PROOF_EFFECT,
            }
        )
    expanded_failure_diagnosis_coverage_complete = all(
        label.get("failure_mode_ids")
        and label.get("evidence_refs")
        and label.get("review_status")
        for label in expanded_failure_label_rows
    )
    failure_labels = {
        **_mapping(copied.get("failure_labels")),
        "schema_version": FAILURE_LABELS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_failures_labeled",
        "label_count": len(failures),
        "failed_attempt_count": len(failures),
        "covered_failed_attempt_ids": sorted(
            _string(attempt.get("attempt_id")) for attempt in failures
        ),
        "missing_failed_attempt_ids": [],
        "covered_failed_scenario_eval_run_ids": sorted(
            _string(attempt.get("scenario_eval_run_id")) for attempt in failures
        ),
        "missing_failed_scenario_eval_run_ids": [],
        "failed_run_label_coverage_complete": True,
        "failure_diagnosis_coverage_complete": expanded_failure_diagnosis_coverage_complete,
        "failure_diagnosis_review_complete": not expanded_nonreviewable_label_ids,
        "failure_diagnosis_complete": (
            expanded_failure_diagnosis_coverage_complete
            and not expanded_nonreviewable_label_ids
        ),
        "failure_diagnosis_blockers": (
            ["failure_labels_nonreviewable_failure_hypotheses"]
            if expanded_nonreviewable_label_ids
            else []
        ),
        "nonreviewable_failure_hypothesis_label_ids": expanded_nonreviewable_label_ids,
        "labels": expanded_failure_label_rows,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    prediction_records = [
        {
            "scenario_id": attempt.get("scenario_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "policy_id": attempt.get("policy_id"),
            "predicted_status": "passed"
            if coerce_strict_success(attempt.get("success")) is True
            else "failed",
            "predicted_success": coerce_strict_success(attempt.get("success")) is True,
            "predicted_cycle_time_seconds": _number(
                _mapping(attempt.get("metrics")).get("cycle_time_seconds")
                or attempt.get("predicted_cycle_time_seconds")
            ),
            "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
            "source": "site_eval_fixture_expanded_to_scenario_eval_matrix",
            "actual_status": "needs_actual_outcome",
        }
        for attempt in expanded_attempts
    ]
    prediction_ledger = {
        **_mapping(copied.get("prediction_outcome_ledger")),
        "schema_version": PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "record_count": len(prediction_records),
        "records": prediction_records,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    calibration_report = {
        **_mapping(copied.get("calibration_report")),
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "needs_real_world_outcomes",
        "record_count": len(prediction_records),
        "records": prediction_records,
        "sim_vs_real_calibration_score": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    label_by_attempt_id = {
        _string(label.get("attempt_id")): label
        for label in expanded_failure_label_rows
        if _string(label.get("attempt_id"))
    }
    aggregation_map: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    dominant_map: Dict[str, Dict[str, Any]] = {}
    breakage_records: List[Dict[str, Any]] = []
    for attempt in failures:
        label = label_by_attempt_id.get(_string(attempt.get("attempt_id"))) or {}
        failure_mode_ids = _string_list(label.get("failure_mode_ids")) or _string_list(
            attempt.get("failure_mode_ids")
        ) or ["unknown_failure_mode"]
        root_cause_category = _string(label.get("root_cause_category")) or _failure_root_cause_category(
            failure_mode_ids,
            failure_reason=_string(attempt.get("failure_reason")) or None,
        )
        media_refs = _failure_frame_or_clip_refs(label or attempt)
        refs = _failure_evidence_refs(label or attempt)
        record = {
            "scenario_id": attempt.get("scenario_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "policy_id": attempt.get("policy_id"),
            "failure_mode_ids": failure_mode_ids,
            "failure_reason": _string(attempt.get("failure_reason")) or None,
            "root_cause_category": root_cause_category,
            "review_required": True,
            "review_status": label.get("review_status"),
            "evidence_refs": refs,
            "frame_or_clip_refs": media_refs,
        }
        breakage_records.append(record)
        exemplar = {
            "attempt_id": attempt.get("attempt_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "policy_id": attempt.get("policy_id"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "failure_mode_ids": failure_mode_ids,
            "root_cause_category": root_cause_category,
            "evidence_refs": refs,
            "frame_or_clip_refs": media_refs,
            "visual_smoke_ref": label.get("visual_smoke_ref"),
            "review_status": label.get("review_status"),
        }
        for failure_mode_id in failure_mode_ids:
            key = (
                _string(attempt.get("policy_id")) or "unknown_policy",
                _string(attempt.get("task_id")) or "unknown_task",
                _string(attempt.get("scenario_id")) or "unknown_scenario",
                failure_mode_id,
                root_cause_category,
            )
            bucket = aggregation_map.setdefault(
                key,
                {
                    "policy_id": key[0],
                    "task_id": key[1],
                    "scenario_id": key[2],
                    "failure_mode_id": key[3],
                    "root_cause_category": key[4],
                    "failed_attempt_count": 0,
                    "scenario_eval_run_ids": [],
                    "exemplar_failed_attempts": [],
                    "media_refs": [],
                    "evidence_refs": [],
                },
            )
            bucket["failed_attempt_count"] += 1
            bucket["scenario_eval_run_ids"] = _dedupe_refs(
                [*bucket["scenario_eval_run_ids"], _string(attempt.get("scenario_eval_run_id"))]
            )
            if len(bucket["exemplar_failed_attempts"]) < 3:
                bucket["exemplar_failed_attempts"].append(exemplar)
            bucket["media_refs"] = _dedupe_refs([*bucket["media_refs"], *media_refs])
            bucket["evidence_refs"] = _dedupe_refs([*bucket["evidence_refs"], *refs])
            dominant = dominant_map.setdefault(
                failure_mode_id,
                {
                    "failure_mode_id": failure_mode_id,
                    "failed_attempt_count": 0,
                    "root_cause_categories": [],
                    "exemplar_failed_attempts": [],
                    "media_refs": [],
                    "evidence_refs": [],
                },
            )
            dominant["failed_attempt_count"] += 1
            dominant["root_cause_categories"] = _dedupe_refs(
                [*dominant["root_cause_categories"], root_cause_category]
            )
            if len(dominant["exemplar_failed_attempts"]) < 3:
                dominant["exemplar_failed_attempts"].append(exemplar)
            dominant["media_refs"] = _dedupe_refs([*dominant["media_refs"], *media_refs])
            dominant["evidence_refs"] = _dedupe_refs([*dominant["evidence_refs"], *refs])
    aggregations = sorted(
        aggregation_map.values(),
        key=lambda row: (
            -int(row["failed_attempt_count"]),
            _string(row["policy_id"]),
            _string(row["task_id"]),
            _string(row["scenario_id"]),
            _string(row["failure_mode_id"]),
            _string(row["root_cause_category"]),
        ),
    )
    dominant_failure_modes = sorted(
        dominant_map.values(),
        key=lambda row: (-int(row["failed_attempt_count"]), _string(row["failure_mode_id"])),
    )
    breakage_library = {
        **_mapping(copied.get("breakage_library")),
        "schema_version": BREAKAGE_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_breakages_recorded",
        "record_count": len(failures),
        "records": breakage_records,
        "aggregation_keys": [
            "policy_id",
            "task_id",
            "scenario_id",
            "failure_mode_id",
            "root_cause_category",
        ],
        "aggregation_count": len(aggregations),
        "aggregations": aggregations,
        "dominant_failure_modes": dominant_failure_modes,
        "dominant_failure_mode_id": dominant_failure_modes[0]["failure_mode_id"]
        if dominant_failure_modes
        else None,
        "source_failure_labels": "failure_labels.json",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }

    copied.update(
        {
            "normalized_attempt_trace": expanded_trace,
            "failure_labels": failure_labels,
            "prediction_outcome_ledger": prediction_ledger,
            "calibration_report": calibration_report,
            "breakage_library": breakage_library,
        }
    )
    for artifact_name, payload in copied.items():
        if artifact_name in {
            "normalized_attempt_trace",
            "failure_labels",
            "prediction_outcome_ledger",
            "calibration_report",
            "breakage_library",
        }:
            write_json(job_dir / f"{artifact_name}.json", payload)
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
            pipeline_dir
            / "simulation_automation"
            / "site_eval_fixture_runner_blocked_manifest.json"
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
        _write_simulator_provider_adapter_manifest(
            job_dir=job_dir,
            simulator=simulator,
            status=_string(result.get("status")),
            blockers=_string_list(result.get("blockers")),
            allow_simulator_execution=allow_simulator_execution,
            env_allows_simulator_execution=_env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"),
            allowed_simulators=allowed_simulators,
            command_text=_string(simulator_commands.get(simulator)),
            artifact_paths=_mapping(result.get("artifact_paths")),
            generated_at=generated_at,
            reason=_string(result.get("reason")) or None,
        )
        copied = _copy_site_eval_artifacts(
            pipeline_dir=pipeline_dir,
            job_dir=job_dir,
            generated_at=generated_at,
        )
        return _attach_simulator_provider_manifest(result), copied, []
    if provisioning_result.get("status") == "blocked":
        result = _blocked_simulator_result(
            simulator=simulator,
            blockers=["gpu_provisioning_blocked"],
            generated_at=generated_at,
            reason="gpu_provisioning_blocked",
        )
        _write_simulator_provider_adapter_manifest(
            job_dir=job_dir,
            simulator=simulator,
            status=_string(result.get("status")),
            blockers=_string_list(result.get("blockers")),
            allow_simulator_execution=allow_simulator_execution,
            env_allows_simulator_execution=_env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"),
            allowed_simulators=allowed_simulators,
            command_text=_string(simulator_commands.get(simulator)),
            artifact_paths=_mapping(result.get("artifact_paths")),
            generated_at=generated_at,
            reason=_string(result.get("reason")) or None,
        )
        copied = _copy_site_eval_artifacts(
            pipeline_dir=pipeline_dir,
            job_dir=job_dir,
            generated_at=generated_at,
        )
        return _attach_simulator_provider_manifest(result), copied, ["gpu_provisioning_blocked"]
    if simulator == "fixture":
        result, copied, blockers = _run_fixture_simulator(
            capture_root=capture_root,
            pipeline_dir=pipeline_dir,
            job_dir=job_dir,
            generated_at=generated_at,
        )
        _write_simulator_provider_adapter_manifest(
            job_dir=job_dir,
            simulator=simulator,
            status=_string(result.get("status")),
            blockers=_string_list(result.get("blockers")),
            allow_simulator_execution=allow_simulator_execution,
            env_allows_simulator_execution=False,
            allowed_simulators=allowed_simulators,
            command_text="",
            artifact_paths=_mapping(result.get("artifact_paths")),
            generated_at=generated_at,
            simulator_output_ingested=bool(copied.get("normalized_attempt_trace")),
            reason=_string(result.get("reason")) or None,
        )
        return _attach_simulator_provider_manifest(result), copied, blockers
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
        simulator_output_path = job_dir / f"{simulator}_simulator_output.json"
        result = _run_command_simulator(
            simulator=simulator,
            command_text=command_text,
            timeout_seconds=timeout_seconds,
            generated_at=generated_at,
            output_path=simulator_output_path,
            capture_root=capture_root,
            scenario_eval_matrix_path=job_dir / "scenario_eval_matrix.json",
        )
    simulator_output_payload = result.pop("simulator_output_payload", None)
    if result.get("status") == "completed" and simulator_output_payload is not None:
        copied = build_simulator_command_artifacts(
            job_dir=job_dir,
            simulator=simulator,
            simulator_output=simulator_output_payload,
            generated_at=generated_at,
        )
        copied = {
            key: value
            for key, value in copied.items()
            if key
            in {
                "normalized_attempt_trace",
                "failure_labels",
                "prediction_outcome_ledger",
                "calibration_report",
                "breakage_library",
            }
        }
        normalized_trace = _mapping(copied.get("normalized_attempt_trace"))
        result = {
            **dict(result),
            "attempt_count": int(_number(normalized_trace.get("attempt_count")) or 0),
            "covered_scenario_eval_run_ids": _string_list(
                normalized_trace.get("covered_scenario_eval_run_ids")
            ),
            "missing_scenario_eval_run_ids": _string_list(
                normalized_trace.get("missing_scenario_eval_run_ids")
            ),
            "scenario_eval_run_coverage_complete": bool(
                normalized_trace.get("scenario_eval_run_coverage_complete")
            ),
            "task_success_summary": _mapping(normalized_trace.get("task_success_summary")),
            "successful_task_attempt_count": int(
                _number(normalized_trace.get("successful_task_attempt_count")) or 0
            ),
            "failed_task_attempt_count": int(
                _number(normalized_trace.get("failed_task_attempt_count")) or 0
            ),
            "artifact_paths": {
                **_mapping(result.get("artifact_paths")),
                "normalized_attempt_trace": "normalized_attempt_trace.json",
                "failure_labels": "failure_labels.json",
                "prediction_outcome_ledger": "prediction_outcome_ledger.json",
                "calibration_report": "calibration_report.json",
                "breakage_library": "breakage_library.json",
            },
        }
    else:
        copied = _copy_site_eval_artifacts(
            pipeline_dir=pipeline_dir,
            job_dir=job_dir,
            generated_at=generated_at,
        )
    _write_simulator_provider_adapter_manifest(
        job_dir=job_dir,
        simulator=simulator,
        status=_string(result.get("status")),
        blockers=_string_list(result.get("blockers")),
        allow_simulator_execution=allow_simulator_execution,
        env_allows_simulator_execution=env_allowed,
        allowed_simulators=allowed_simulators,
        command_text=command_text,
        artifact_paths=_mapping(result.get("artifact_paths")),
        generated_at=generated_at,
        simulator_output_ingested=(
            result.get("status") == "completed" and simulator_output_payload is not None
        ),
        reason=_string(result.get("reason")) or None,
    )
    result = _attach_simulator_provider_manifest(result)
    return (
        result,
        copied,
        ["simulator_service_blocked"] if result.get("status") == "blocked" else [],
    )


def _training_request(
    *,
    request: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    operation = _string(request.get("operation") or "evaluate_only")
    preference = _mapping(
        request.get("cosmos_training_preference") or request.get("cosmosTrainingPreference")
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


def _metric_number(attempt: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    metrics = _mapping(attempt.get("metrics"))
    for key in keys:
        value = attempt.get(key)
        if value is None:
            value = metrics.get(key)
        number = _number(value)
        if number is not None:
            return number
    return default


def _standard_policy_scorecard(attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    attempt_count = len(attempts)
    if not attempt_count:
        return {
            "success_rate": 0.0,
            "cycle_time": {"mean_seconds": None, "sample_count": 0},
            "intervention_rate": 0.0,
            "unsafe_proximity": {"event_count": 0},
            "collision_risk": {"event_count": 0},
            "object_drop": {"event_count": 0},
            "wrong_object": {"event_count": 0},
            "timeout": {"event_count": 0},
            "recovery_success": {"success_rate": None, "success_count": 0, "attempt_count": 0},
            "world_model_uncertainty": {
                "status": "not_available",
                "mean_score": None,
                "sample_count": 0,
            },
            "sim_vs_real_calibration_score": None,
        }
    successes = sum(
        1 for attempt in attempts if coerce_strict_success(attempt.get("success")) is True
    )
    cycle_times = [
        value
        for attempt in attempts
        if (value := _metric_number(attempt, "cycle_time_seconds", default=-1.0)) >= 0.0
    ]
    intervention_count = sum(
        _metric_number(attempt, "intervention_count", "interventions", default=0.0)
        for attempt in attempts
    )
    unsafe_count = sum(
        _metric_number(
            attempt,
            "unsafe_proximity_event_count",
            "unsafe_proximity_count",
            default=0.0,
        )
        for attempt in attempts
    )
    collision_count = sum(
        _metric_number(
            attempt,
            "collision_risk_event_count",
            "collision_count",
            "contact_event_count",
            default=0.0,
        )
        for attempt in attempts
    )
    object_drop_count = sum(
        _metric_number(attempt, "object_drop_count", "drop_count", default=0.0)
        for attempt in attempts
    )
    wrong_object_count = sum(
        _metric_number(attempt, "wrong_object_count", default=0.0) for attempt in attempts
    )
    timeout_count = sum(
        _metric_number(attempt, "timeout_count", default=0.0) for attempt in attempts
    )
    recovery_attempt_count = sum(
        _metric_number(attempt, "recovery_attempt_count", default=0.0) for attempt in attempts
    )
    recovery_success_count = sum(
        _metric_number(attempt, "recovery_success_count", default=0.0) for attempt in attempts
    )
    uncertainty_values = [
        value
        for attempt in attempts
        if (
            value := _metric_number(
                attempt,
                "world_model_uncertainty",
                "uncertainty",
                default=-1.0,
            )
        )
        >= 0.0
    ]
    return {
        "success_rate": round(successes / float(attempt_count), 6),
        "cycle_time": {
            "mean_seconds": round(sum(cycle_times) / len(cycle_times), 6) if cycle_times else None,
            "sample_count": len(cycle_times),
        },
        "intervention_rate": round(intervention_count / float(attempt_count), 6),
        "unsafe_proximity": {"event_count": int(unsafe_count)},
        "collision_risk": {"event_count": int(collision_count)},
        "object_drop": {"event_count": int(object_drop_count)},
        "wrong_object": {"event_count": int(wrong_object_count)},
        "timeout": {"event_count": int(timeout_count)},
        "recovery_success": {
            "success_rate": round(recovery_success_count / recovery_attempt_count, 6)
            if recovery_attempt_count
            else None,
            "success_count": int(recovery_success_count),
            "attempt_count": int(recovery_attempt_count),
        },
        "world_model_uncertainty": {
            "status": "scored" if uncertainty_values else "not_available",
            "mean_score": round(sum(uncertainty_values) / len(uncertainty_values), 6)
            if uncertainty_values
            else None,
            "sample_count": len(uncertainty_values),
        },
        "sim_vs_real_calibration_score": None,
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
        attempts = [item for item in trace.get("attempts", []) or [] if isinstance(item, Mapping)]
        trace_status = _string(trace.get("status"))
        if trace_status.startswith("blocked") or trace_status in {"not_available", "missing"}:
            status = "blocked"
            blockers = _string_list(trace.get("blockers")) or ["normalized_attempt_trace_missing"]
        elif attempts and any(
            coerce_strict_success(item.get("success")) is not True for item in attempts
        ):
            status = "completed_with_failures"
            blockers = []
        else:
            status = "completed"
            blockers = []
    trace_for_scorecard = _mapping(copied_artifacts.get("normalized_attempt_trace"))
    attempts_for_scorecard = [
        item for item in trace_for_scorecard.get("attempts", []) or [] if isinstance(item, Mapping)
    ]
    return {
        "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "blockers": blockers,
        "simulator_result_status": simulator_result.get("status"),
        "arena_result_ingest_status": _mapping(
            copied_artifacts.get("normalized_attempt_trace")
        ).get("status"),
        "arena_metrics_path": (
            "arena_eval_metrics.json"
            if _mapping(copied_artifacts.get("arena_eval_metrics"))
            else None
        ),
        "normalized_attempt_trace_path": "normalized_attempt_trace.json",
        "failure_labels_path": "failure_labels.json",
        "prediction_outcome_ledger_path": "prediction_outcome_ledger.json",
        "calibration_report_path": "calibration_report.json",
        "breakage_library_path": "breakage_library.json",
        "standard_policy_scorecard": _standard_policy_scorecard(attempts_for_scorecard),
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _ordered_unique_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _string(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _attempt_task_success(attempt: Mapping[str, Any]) -> bool:
    if "task_success" in attempt:
        return coerce_strict_success(attempt.get("task_success")) is True
    if "success" in attempt:
        return coerce_strict_success(attempt.get("success")) is True
    return coerce_strict_success(_mapping(attempt.get("task_outcome")).get("task_success")) is True


def _matrix_policy_run_index(
    scenario_eval_matrix: Mapping[str, Any],
) -> tuple[Dict[str, str], Dict[str, str], List[str]]:
    run_to_base: Dict[str, str] = {}
    run_to_policy: Dict[str, str] = {}
    base_run_ids: List[str] = []
    for run in scenario_eval_matrix.get("runs", []) or []:
        if not isinstance(run, Mapping):
            continue
        run_id = _string(run.get("scenario_eval_run_id") or run.get("scenarioEvalRunId"))
        if not run_id:
            continue
        base_run_id = (
            _string(run.get("base_scenario_eval_run_id") or run.get("baseScenarioEvalRunId"))
            or run_id
        )
        policy_id = _string(run.get("policy_id") or run.get("policyId"))
        run_to_base[run_id] = base_run_id
        if policy_id:
            run_to_policy[run_id] = policy_id
        if base_run_id not in base_run_ids:
            base_run_ids.append(base_run_id)
    declared_base_ids = _string_list(
        scenario_eval_matrix.get("policy_comparison_base_scenario_eval_run_ids")
    )
    if declared_base_ids:
        base_run_ids = declared_base_ids
    return run_to_base, run_to_policy, base_run_ids


def _score_range_blockers(attempts: Sequence[Mapping[str, Any]]) -> List[str]:
    blockers: List[str] = []
    for attempt in attempts:
        for key in (
            "confidence",
            "confidence_score",
            "uncertainty",
            "uncertainty_score",
            "predicted_success_probability",
        ):
            if key not in attempt:
                continue
            value = _number(attempt.get(key))
            if value is None or value < 0.0 or value > 1.0:
                blockers.append(f"attempt_{key}_out_of_range")
        metrics = _mapping(attempt.get("metrics"))
        for key in ("confidence", "uncertainty", "predicted_success_probability"):
            if key not in metrics:
                continue
            value = _number(metrics.get(key))
            if value is None or value < 0.0 or value > 1.0:
                blockers.append(f"attempt_metric_{key}_out_of_range")
    return _dedupe(blockers)


def _simulator_policy_visual_gate(job_dir: Path) -> tuple[bool, List[str], Dict[str, Any]]:
    visual_media_coverage = _read_optional_mapping(
        job_dir / "simulator_command_batch_visual_media_coverage.json"
    )
    visual_review_ledger = _read_optional_mapping(job_dir / "visual_review_ledger.json")
    blockers: List[str] = []
    if visual_media_coverage:
        if visual_media_coverage.get("all_required_runs_have_visual_recording") is not True:
            blockers.append("visual_media_coverage_not_complete_for_all_runs")
        if visual_media_coverage.get("all_required_runs_have_robot_pov_video") is not True:
            blockers.append("robot_pov_video_coverage_not_complete")
        if visual_media_coverage.get("all_required_runs_have_third_person_video") is not True:
            blockers.append("third_person_video_coverage_not_complete")
    else:
        blockers.append("visual_media_coverage_manifest_missing")
    if visual_review_ledger:
        if visual_review_ledger.get("visual_review_coverage_complete") is not True:
            blockers.append("visual_review_coverage_not_complete_for_all_runs")
    else:
        blockers.append("visual_review_ledger_missing")
    return not blockers, blockers, {
        "visual_media_coverage": visual_media_coverage,
        "visual_review_ledger": visual_review_ledger,
    }


def _write_candidate_selection_report(
    *,
    job_dir: Path,
    scorecard: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    single_best = bool(scorecard.get("single_best_policy_claimed"))
    ambiguous = _string(scorecard.get("status")) == "completed_ambiguous_ranking"
    report = {
        "schema_version": "policy_candidate_selection_report.v1",
        "generated_at": generated_at,
        "status": "clear_winner"
        if single_best
        else "ambiguous_candidate_shortlist"
        if ambiguous
        else "visual_review_required_candidate_shortlist",
        "top_policy_id": scorecard.get("top_policy_id"),
        "evaluator_top_policy_id": scorecard.get("evaluator_top_policy_id"),
        "policy_rankings": list(scorecard.get("policy_rankings") or []),
        "tie_or_ambiguity_status": "ambiguous_or_tied"
        if ambiguous
        else "single_evaluator_top_policy"
        if scorecard.get("evaluator_top_policy_id")
        else "review_required",
        "policy_ranking_scorecard_path": "policy_ranking_scorecard.json",
        "claim_boundary": {
            "configured_evaluator_only": True,
            "do_not_use_as_rank_fidelity_result": True,
            "real_world_outcome_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    _write_job_json(job_dir, "candidate_selection_report.json", report)
    lines = [
        "# Candidate Selection Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Evaluator top policy: `{report.get('evaluator_top_policy_id')}`",
        f"- Single best policy claimed: `{single_best}`",
        "",
        "This report is bounded to the configured evaluator and is not a rank-fidelity, safety, deployment, or real-world outcome claim.",
    ]
    write_text(job_dir / "candidate_selection_report.md", "\n".join(lines))
    return report


def _write_simulator_policy_ranking_scorecard(
    *,
    job_dir: Path,
    request: Mapping[str, Any],
    scenario_eval_matrix: Mapping[str, Any],
    simulator_result: Mapping[str, Any],
    copied_artifacts: Mapping[str, Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    if (job_dir / "policy_ranking_scorecard.json").is_file():
        return _read_optional_mapping(job_dir / "policy_ranking_scorecard.json")
    candidates = _request_policy_candidates(request)
    trace = _mapping(copied_artifacts.get("normalized_attempt_trace")) or _read_optional_mapping(
        job_dir / "normalized_attempt_trace.json"
    )
    attempts = [
        dict(item)
        for item in trace.get("attempts", []) or []
        if isinstance(item, Mapping)
    ]
    policy_comparison_requested = bool(
        scenario_eval_matrix.get("policy_comparison_mode")
        or scenario_eval_matrix.get("policy_comparison_requested")
        or len(candidates) >= 2
    )
    if not policy_comparison_requested and not candidates:
        return {}

    run_to_base, run_to_policy, base_run_ids = _matrix_policy_run_index(
        scenario_eval_matrix
    )
    required_run_ids = _ordered_unique_strings(
        base_run_ids
        or [
            attempt.get("scenario_eval_run_id")
            for attempt in attempts
            if _string(attempt.get("scenario_eval_run_id"))
        ]
    )
    declared_policy_ids = _ordered_unique_strings(
        [
            *[candidate.get("policy_id") for candidate in candidates],
            *[
                run.get("policy_id")
                for run in scenario_eval_matrix.get("runs", []) or []
                if isinstance(run, Mapping)
            ],
            *[attempt.get("policy_id") for attempt in attempts],
        ]
    )
    by_policy: Dict[str, List[Dict[str, Any]]] = {
        policy_id: [] for policy_id in declared_policy_ids
    }
    for attempt in attempts:
        actual_run_id = _string(
            attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId")
        )
        policy_id = (
            _string(attempt.get("policy_id") or attempt.get("policyId"))
            or run_to_policy.get(actual_run_id)
            or "policy"
        )
        base_run_id = run_to_base.get(actual_run_id) or _string(
            attempt.get("base_scenario_eval_run_id")
            or attempt.get("baseScenarioEvalRunId")
            or actual_run_id
        )
        by_policy.setdefault(policy_id, []).append(
            {
                **attempt,
                "policy_id": policy_id,
                "base_scenario_eval_run_id": base_run_id,
                "actual_scenario_eval_run_id": actual_run_id,
            }
        )
    if not declared_policy_ids:
        declared_policy_ids = list(by_policy)

    required_run_set = set(required_run_ids)
    per_policy_coverage: List[Dict[str, Any]] = []
    missing_by_policy: Dict[str, List[str]] = {}
    extra_by_policy: Dict[str, List[str]] = {}
    attempt_count_by_policy: Dict[str, int] = {}
    duplicate_required_attempts_by_policy: Dict[str, List[str]] = {}
    rankings: List[Dict[str, Any]] = []
    for policy_id in declared_policy_ids:
        policy_attempts = by_policy.get(policy_id, [])
        observed_base_ids = _ordered_unique_strings(
            attempt.get("base_scenario_eval_run_id") for attempt in policy_attempts
        )
        attempt_counts = {
            run_id: sum(
                1
                for attempt in policy_attempts
                if _string(attempt.get("base_scenario_eval_run_id")) == run_id
            )
            for run_id in observed_base_ids
        }
        missing = [run_id for run_id in required_run_ids if run_id not in set(observed_base_ids)]
        extra = sorted(set(observed_base_ids) - required_run_set) if required_run_ids else []
        duplicates = [
            run_id
            for run_id, count in attempt_counts.items()
            if run_id in required_run_set and count > 1
        ]
        attempt_count = len(policy_attempts)
        expected_attempt_count = len(required_run_ids)
        coverage_complete = bool(
            required_run_ids
            and not missing
            and not extra
            and not duplicates
            and attempt_count == expected_attempt_count
        )
        successes = sum(1 for attempt in policy_attempts if _attempt_task_success(attempt))
        success_rate = round(successes / float(attempt_count), 6) if attempt_count else 0.0
        per_policy_coverage.append(
            {
                "policy_id": policy_id,
                "required_scenario_eval_run_ids": list(required_run_ids),
                "covered_scenario_eval_run_ids": [
                    run_id for run_id in required_run_ids if run_id in set(observed_base_ids)
                ],
                "missing_scenario_eval_run_ids": missing,
                "extra_scenario_eval_run_ids": extra,
                "duplicate_required_scenario_eval_run_ids": duplicates,
                "attempt_count": attempt_count,
                "expected_attempt_count": expected_attempt_count,
                "coverage_complete": coverage_complete,
            }
        )
        missing_by_policy[policy_id] = missing
        extra_by_policy[policy_id] = extra
        attempt_count_by_policy[policy_id] = attempt_count
        duplicate_required_attempts_by_policy[policy_id] = duplicates
        rankings.append(
            {
                "policy_id": policy_id,
                "attempt_count": attempt_count,
                "task_success_count": successes,
                "task_success_rate": success_rate,
                "score": success_rate,
                "score_basis": "task_success_rate",
                "coverage_complete": coverage_complete,
            }
        )
    rankings.sort(key=lambda row: (-float(row["score"]), _string(row["policy_id"])))
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank

    comparison_blockers: List[str] = []
    if len(declared_policy_ids) < 2:
        comparison_blockers.append("policy_comparison_requires_at_least_two_candidates")
    if not required_run_ids:
        comparison_blockers.append("policy_comparison_required_scenario_eval_run_ids_missing")
    coverage_complete = bool(
        required_run_ids
        and per_policy_coverage
        and all(row.get("coverage_complete") for row in per_policy_coverage)
    )
    if not coverage_complete:
        comparison_blockers.append("policy_comparison_policy_coverage_not_symmetric")
    score_range_blockers = _score_range_blockers(attempts)
    score_ranges_valid = not score_range_blockers
    comparison_blockers.extend(score_range_blockers)

    top_score = float(rankings[0]["score"]) if rankings else None
    tied_top_policy_ids = [
        _string(row.get("policy_id"))
        for row in rankings
        if top_score is not None and abs(float(row.get("score") or 0.0) - top_score) <= 1e-9
    ]
    reference_only_policy_ids = {
        _string(candidate.get("policy_id"))
        for candidate in candidates
        if candidate.get("reference_only") or candidate.get("referenceOnly")
    }
    distinct_behavior_proven = any(
        bool(candidate.get("candidate_behavior_distinctness_proven"))
        or bool(candidate.get("candidateBehaviorDistinctnessProven"))
        for candidate in candidates
    )
    visual_gate_passed, visual_review_blockers, visual_evidence = _simulator_policy_visual_gate(
        job_dir
    )
    ambiguous = len(tied_top_policy_ids) != 1
    evaluator_top_policy_id = tied_top_policy_ids[0] if len(tied_top_policy_ids) == 1 else None
    all_reference_only = bool(
        declared_policy_ids
        and reference_only_policy_ids
        and set(declared_policy_ids).issubset(reference_only_policy_ids)
    )
    single_best_policy_claimed = bool(
        evaluator_top_policy_id
        and visual_gate_passed
        and not all_reference_only
        and not comparison_blockers
    )
    if comparison_blockers:
        scorecard_status = "blocked_inconclusive_ranking"
    elif ambiguous:
        scorecard_status = "completed_ambiguous_ranking"
    elif not visual_gate_passed or all_reference_only:
        scorecard_status = "completed_low_confidence_ranking"
    else:
        scorecard_status = "completed"

    scorecard = {
        "schema_version": "policy_ranking_scorecard.v1",
        "generated_at": generated_at,
        "status": scorecard_status,
        "evaluation_substrate": _string(request.get("evaluation_substrate"))
        or f"classical_sim_{_string(simulator_result.get('framework')) or 'simulator'}",
        "ranking_basis": "simulator_task_success_over_policy_candidate_scenario_matrix",
        "policy_count": len(declared_policy_ids),
        "policy_ids": declared_policy_ids,
        "required_scenario_eval_run_ids": required_run_ids,
        "required_scenario_eval_run_id_basis": "base_scenario_eval_run_id",
        "policy_comparison_actual_scenario_eval_run_ids": _string_list(
            scenario_eval_matrix.get("policy_comparison_expanded_scenario_eval_run_ids")
        ),
        "per_policy_coverage": per_policy_coverage,
        "coverage_complete": coverage_complete,
        "missing_by_policy": missing_by_policy,
        "extra_by_policy": extra_by_policy,
        "attempt_count_by_policy": attempt_count_by_policy,
        "duplicate_required_attempts_by_policy": duplicate_required_attempts_by_policy,
        "score_ranges_valid": score_ranges_valid,
        "comparison_blockers": _dedupe(comparison_blockers),
        "policy_rankings": rankings,
        "top_policy_id": evaluator_top_policy_id if single_best_policy_claimed else None,
        "evaluator_top_policy_id": evaluator_top_policy_id,
        "tied_top_policy_ids": tied_top_policy_ids,
        "single_best_policy_claimed": single_best_policy_claimed,
        "ranking_confidence": {
            "status": "ambiguous"
            if ambiguous
            else "reference_only_low_confidence"
            if all_reference_only
            else "review_grade"
            if visual_gate_passed and not comparison_blockers
            else "visual_review_required",
            "visual_gate_passed": visual_gate_passed,
            "policy_behavior_distinctness_proven": distinct_behavior_proven,
            "score_delta_to_runner_up": round(
                float(rankings[0]["score"]) - float(rankings[1]["score"]), 6
            )
            if len(rankings) >= 2
            else None,
        },
        "visual_smoke_status": "passed" if visual_gate_passed else "review_required",
        "visual_rollout_useful_for_task_success_review": visual_gate_passed,
        "visual_review_blockers": visual_review_blockers,
        "review_grade_policy_ranking": bool(visual_gate_passed and not comparison_blockers),
        "fixture_evaluator_only": False,
        "simulator_evaluator_only": True,
        "evaluator_only": True,
        "simulator_reference_candidate_only": all_reference_only,
        "candidate_behavior_distinctness_proven": distinct_behavior_proven,
        "comparison_contract": {
            "comparison_scope": "configured_evaluator_only",
            "same_observation_protocol": True,
            "same_observation_protocol_id": scenario_eval_matrix.get(
                "policy_comparison_observation_protocol_id"
            ),
            "same_action_protocol": True,
            "same_action_protocol_id": scenario_eval_matrix.get(
                "policy_comparison_action_protocol_id"
            ),
            "tie_handling": "single_best_policy_claimed_false_when_tied_or_reference_only",
            "evaluation_readiness_claimed": False,
            "external_deployment_grade_claimed": False,
        },
        "visual_gate_evidence": visual_evidence,
        "artifact_paths": {
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "normalized_attempt_trace": "normalized_attempt_trace.json",
            "visual_media_coverage": "simulator_command_batch_visual_media_coverage.json"
            if (job_dir / "simulator_command_batch_visual_media_coverage.json").is_file()
            else None,
            "visual_review_ledger": "visual_review_ledger.json"
            if (job_dir / "visual_review_ledger.json").is_file()
            else None,
        },
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "primary_proof_target": "policy_comparison_within_configured_evaluator",
            "policy_ranking_is_evaluator_bounded": True,
            "policy_ranking_is_not_evaluation_readiness": True,
            "simulator_evaluator_only": True,
            "fixture_evaluator_only": False,
            "rank_fidelity_result_proven": False,
            "real_world_outcome_proven": False,
            "public_claim_upgrade_allowed": False,
            "single_best_policy_claimed": single_best_policy_claimed,
            "candidate_behavior_distinctness_proven": distinct_behavior_proven,
        },
    }
    _write_job_json(job_dir, "policy_ranking_scorecard.json", scorecard)
    _write_candidate_selection_report(
        job_dir=job_dir,
        scorecard=scorecard,
        generated_at=generated_at,
    )
    _write_job_json(
        job_dir,
        "wam_eval_claim_boundary.json",
        {
            "schema_version": "wam_eval_claim_boundary.v1",
            "generated_at": generated_at,
            "evaluation_substrate": scorecard["evaluation_substrate"],
            "primary_proof_target": "policy_comparison_within_configured_evaluator",
            "policy_ranking_is_evaluator_bounded": True,
            "policy_ranking_is_not_evaluation_readiness": True,
            "simulator_evaluator_only": True,
            "generated_rollouts_are_model_derived_support_artifacts": False,
            "passing_wam_heldout_eval_is_not_rank_fidelity_result": True,
            "customer_specific_srcc_claimed": False,
            "rank_fidelity_result_proven": False,
            "real_world_outcome_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )
    return scorecard


def _robot_eval_report_markdown(report: Mapping[str, Any]) -> str:
    scorecard = _mapping(report.get("evaluator_scores"))
    cycle_time = _mapping(scorecard.get("cycle_time"))
    closure = _mapping(report.get("live_eval_closure"))
    proof = _mapping(report.get("proof_boundary"))
    scenario = _mapping(report.get("scenario_eval"))
    policy = _mapping(report.get("policy_interface"))
    lines = [
        "# Robot Eval Report",
        "",
        f"- Job: `{report.get('job_id')}`",
        f"- Status: `{report.get('job_status')}`",
        f"- Scene: `{report.get('scene_id')}`",
        f"- Capture: `{report.get('capture_id')}`",
        f"- Scenario eval runs: `{scenario.get('scenario_eval_run_count')}`",
        f"- Variations covered: `{len(_string_list(scenario.get('variation_names_covered')))}`",
        f"- Executed policy modalities: `{', '.join(_string_list(policy.get('executed_modalities')))}`",
        f"- Evaluation status: `{report.get('evaluation_status')}`",
        f"- Success rate: `{scorecard.get('success_rate')}`",
        f"- Mean cycle time seconds: `{cycle_time.get('mean_seconds')}`",
        f"- Intervention rate: `{scorecard.get('intervention_rate')}`",
        f"- Live closure status: `{closure.get('status')}`",
        f"- Live end-to-end verified: `{closure.get('live_end_to_end_verified')}`",
        f"- Robot policy execution proven: `{proof.get('robot_policy_execution_proven')}`",
        f"- Real-world outcome proven: `{proof.get('real_world_outcome_proven')}`",
        f"- Public claim upgrade allowed: `{proof.get('public_claim_upgrade_allowed')}`",
        "",
        "## Core Artifacts",
        "",
    ]
    for label, path in _mapping(report.get("artifact_paths")).items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Proof Boundary",
            "",
            "This report summarizes the eval harness output. It does not upgrade generated-world rank fidelity, safety, simulator execution, policy execution, or deployment claims beyond the referenced proof-boundary and live-closure artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_robot_eval_report(
    *,
    job_dir: Path,
    job_id: str,
    scene_id: str,
    capture_id: str,
    job_status: str,
    blockers: Sequence[str],
    request: Mapping[str, Any],
    scenario_eval_matrix: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
    policy_execution_manifest: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
    deployment_validation: Mapping[str, Any],
    live_closure: Mapping[str, Any],
    proof_boundary: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    deployment_ledger = _mapping(deployment_validation.get("ledger"))
    calibration = _mapping(deployment_validation.get("calibration_report"))
    followup_plan = _mapping(deployment_validation.get("followup_plan"))
    followup_request_queue = _mapping(deployment_validation.get("followup_request_queue"))
    modality_results = _mapping(policy_execution_manifest.get("modality_results"))
    executed_modalities = [
        modality
        for modality, result in modality_results.items()
        if isinstance(result, Mapping) and int(_number(result.get("attempt_count")) or 0) > 0
    ]
    report = {
        "schema_version": ROBOT_EVAL_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "job_status": job_status,
        "operation": _string(request.get("operation") or "evaluate_only"),
        "status": "generated",
        "blockers": _string_list(blockers),
        "neutral_eval_harness_flow": [
            "site_task_scenario",
            "robot_policy_interface",
            "sim_or_world_model_rollout",
            "evaluator_scores",
            "proof_boundary",
            "report_generated",
        ],
        "scenario_eval": {
            "status": scenario_eval_matrix.get("status"),
            "scenario_eval_run_count": scenario_eval_matrix.get("scenario_eval_run_count"),
            "required_variation_names": _string_list(
                scenario_eval_matrix.get("required_variation_names")
            ),
            "variation_names_covered": _string_list(
                scenario_eval_matrix.get("variation_names_covered")
            ),
            "missing_required_variation_names": _string_list(
                scenario_eval_matrix.get("missing_required_variation_names")
            ),
        },
        "policy_interface": {
            "status": policy_manifest.get("status"),
            "configured_modalities": _string_list(policy_manifest.get("selected_modalities")),
            "selected_modalities": _string_list(
                policy_execution_manifest.get("selected_modalities")
            ),
            "executed_modalities": executed_modalities,
            "supported_modalities": _string_list(policy_manifest.get("supported_modalities"))
            or list(POLICY_MODALITY_ORDER),
            "policy_execution_status": policy_execution_manifest.get("status"),
            "robot_policy_execution_proven": _strict_bool(
                policy_execution_manifest.get("robot_policy_execution_proven")
            ),
        },
        "evaluation_status": evaluation_result.get("status"),
        "evaluator_scores": _mapping(evaluation_result.get("standard_policy_scorecard")),
        "real_world_validation": {
            "deployment_outcome_status": deployment_ledger.get("status"),
            "real_world_outcome_records_present": _strict_bool(
                deployment_ledger.get("real_world_outcome_records_present")
            ),
            "real_world_outcome_proven": _strict_bool(
                deployment_ledger.get("real_world_outcome_proven")
            ),
            "owner_evidence_record_count": int(
                deployment_ledger.get("owner_evidence_record_count") or 0
            ),
            "missing_owner_evidence_record_ids": _string_list(
                deployment_ledger.get("missing_owner_evidence_record_ids")
            ),
            "followup_plan_status": followup_plan.get("status"),
            "followup_action_count": int(
                _mapping(followup_plan.get("summary")).get("action_count") or 0
            ),
            "real_world_validation_followup_plan_path": (
                "real_world_validation_followup_plan.json"
            ),
            "followup_request_queue_status": followup_request_queue.get("status"),
            "followup_request_queue_count": int(
                followup_request_queue.get("queued_request_count") or 0
            ),
            "real_world_validation_followup_request_queue_path": (
                "real_world_validation_followup_request_queue.json"
            ),
        },
        "predicted_vs_actual": {
            "sim_vs_real_calibration_status": calibration.get("status"),
            "sim_vs_real_calibration_score": calibration.get("sim_vs_real_calibration_score"),
            "prediction_vs_actual_deployment_summary_path": (
                "prediction_vs_actual_deployment_summary.json"
            ),
        },
        "live_eval_closure": {
            "status": live_closure.get("status"),
            "repo_local_artifacts_ready": _strict_bool(
                live_closure.get("repo_local_artifacts_ready")
            ),
            "live_external_ready": _strict_bool(live_closure.get("live_external_ready")),
            "live_end_to_end_verified": _strict_bool(
                live_closure.get("live_end_to_end_verified")
            ),
            "blockers": _string_list(live_closure.get("blockers")),
        },
        "requirement_coverage": {
            "schema_version": _mapping(live_closure.get("requirement_coverage")).get(
                "schema_version"
            ),
            "requirement_count": _mapping(live_closure.get("requirement_coverage")).get(
                "requirement_count"
            ),
            "passed_count": _mapping(live_closure.get("requirement_coverage")).get("passed_count"),
            "blocked_count": _mapping(live_closure.get("requirement_coverage")).get(
                "blocked_count"
            ),
            "blocked_requirement_ids": _string_list(
                _mapping(live_closure.get("requirement_coverage")).get("blocked_requirement_ids")
            ),
        },
        "proof_boundary": {
            "review_acceptance_proven": _strict_bool(
                proof_boundary.get("review_acceptance_proven")
            ),
            "rights_privacy_scope_proven": _strict_bool(
                proof_boundary.get("rights_privacy_scope_proven")
            ),
            "signed_delivery_access_proven": _strict_bool(
                proof_boundary.get("signed_delivery_access_proven")
            ),
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": _strict_bool(
                proof_boundary.get("safety_validation_proven")
            ),
            "simulator_execution_proven": _strict_bool(
                proof_boundary.get("simulator_execution_proven")
            ),
            "robot_policy_execution_proven": _strict_bool(
                proof_boundary.get("robot_policy_execution_proven")
            ),
            "real_world_outcome_proven": _strict_bool(
                proof_boundary.get("real_world_outcome_proven")
            ),
            "physics_contact_validated": _strict_bool(
                proof_boundary.get("physics_contact_validated")
            ),
            "non_ranking_operational_claim_validated": _strict_bool(
                proof_boundary.get("non_ranking_operational_claim_validated")
            ),
            "rank_fidelity_result_proven": _strict_bool(
                proof_boundary.get("rank_fidelity_result_proven")
            ),
            "public_claim_upgrade_allowed": _strict_bool(
                proof_boundary.get("public_claim_upgrade_allowed")
            ),
        },
        "artifact_paths": {
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "evaluation_result": "evaluation_result.json",
            "task_eval_run_report": "task_eval_run_report.json",
            "policy_execution_manifest": "policy_execution_manifest.json",
            "policy_execution_trace": "policy_execution_trace.json",
            "deployment_outcome_ledger": "deployment_outcome_ledger.json",
            "prediction_vs_actual_deployment_summary": (
                "prediction_vs_actual_deployment_summary.json"
            ),
            "real_world_validation_followup_plan": ("real_world_validation_followup_plan.json"),
            "real_world_validation_followup_request_queue": (
                "real_world_validation_followup_request_queue.json"
            ),
            **(
                {
                    "evaluation_substrate_registry": "evaluation_substrate_registry.json",
                    "wam_evaluation_request": "wam_evaluation_request.json",
                    "wam_rollout_manifest": "wam_rollout_manifest.json",
                    "wam_rollout_results": "wam_rollout_results.json",
                    "vision_success_labels": "vision_success_labels.json",
                    "policy_ranking_scorecard": "policy_ranking_scorecard.json",
                    "wam_eval_claim_boundary": "wam_eval_claim_boundary.json",
                    "real_world_validation_followup_request": (
                        "real_world_validation_followup_request.json"
                    ),
                    "srcc_validation_plan": "srcc_validation_plan.json",
                }
                if (job_dir / "wam_rollout_manifest.json").is_file()
                else {}
            ),
            **(
                {
                    "policy_ranking_scorecard": "policy_ranking_scorecard.json",
                    "candidate_selection_report": "candidate_selection_report.json",
                    "wam_eval_claim_boundary": "wam_eval_claim_boundary.json",
                }
                if (job_dir / "policy_ranking_scorecard.json").is_file()
                else {}
            ),
            "live_eval_closure_manifest": "live_eval_closure_manifest.json",
            "proof_boundary": "proof_boundary.json",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    _write_job_json(job_dir, "robot_eval_report.json", report)
    write_text(job_dir / "robot_eval_report.md", _robot_eval_report_markdown(report))
    return report


def _epoch_from_iso(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _freshness_for_artifact(
    *,
    path: Path,
    generated_at: str,
) -> Dict[str, Any]:
    run_started_epoch = _epoch_from_iso(generated_at)
    artifact_mtime_epoch = path.stat().st_mtime if path.is_file() else None
    return build_artifact_freshness_evidence(
        artifact_mtime_epoch=artifact_mtime_epoch,
        run_started_epoch=run_started_epoch,
    )


def _attempts_for_task_eval_report(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    for index, attempt in enumerate(trace.get("attempts") or []):
        if not isinstance(attempt, Mapping):
            continue
        row = dict(attempt)
        strict_success = coerce_strict_success(
            row.get("success")
            if "success" in row
            else row.get("task_success")
            if "task_success" in row
            else _mapping(row.get("task_outcome")).get("task_success")
        )
        if strict_success is not None:
            row["success"] = strict_success
        row.setdefault("attempt_id", f"attempt_{index + 1:04d}")
        attempts.append(row)
    return attempts


def _trace_task_success_for_ledger(attempts: Sequence[Mapping[str, Any]]) -> bool | None:
    if not attempts:
        return None
    verdicts: List[bool] = []
    for attempt in attempts:
        strict_success = coerce_strict_success(attempt.get("success"))
        if strict_success is None:
            return None
        verdicts.append(strict_success)
    return all(verdicts)


def _task_eval_task_metadata(
    *,
    request: Mapping[str, Any],
    scenario_eval_matrix: Mapping[str, Any],
) -> Dict[str, Any]:
    for task in request.get("requested_tasks") or request.get("requestedTasks") or []:
        if isinstance(task, Mapping):
            return dict(task)
    for run in _scenario_eval_matrix_runs(scenario_eval_matrix):
        row = _mapping(run)
        if row:
            return {
                key: value
                for key, value in row.items()
                if key
                in {
                    "task_id",
                    "task_name",
                    "scenario_id",
                    "task_success_contract",
                    "success_contract",
                    "affordance_object_ids",
                    "target_object_ids",
                    "success_state_change",
                }
            }
    return {}


def _first_policy_id(
    *,
    request: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> str | None:
    for source in (
        attempts,
        request.get("policy_candidates") or request.get("policyCandidates") or [],
        policy_manifest.get("policy_candidates") or policy_manifest.get("policies") or [],
    ):
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes, bytearray)):
            continue
        for row in source:
            if isinstance(row, Mapping):
                policy_id = _string(row.get("policy_id") or row.get("policyId"))
                if policy_id:
                    return policy_id
    return _string(policy_manifest.get("policy_id") or policy_manifest.get("policyId")) or None


def _task_eval_review_verdicts(job_dir: Path) -> List[Dict[str, Any]]:
    verdicts: List[Dict[str, Any]] = []
    for source_name in (
        "vision_success_labels",
        "rollout_vision_labels",
        "accepted_failure_labels",
        "review_resolution_ledger",
    ):
        payload = _read_optional_mapping(job_dir / f"{source_name}.json")
        if not payload:
            continue
        rows = payload.get("labels") or payload.get("accepted_labels") or payload.get("reviews")
        if isinstance(rows, Mapping):
            rows = rows.values()
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            rows = [payload]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            success = coerce_strict_success(
                row.get("task_success")
                if "task_success" in row
                else row.get("success")
                if "success" in row
                else row.get("review_task_success")
            )
            if success is None:
                continue
            verdicts.append(
                {
                    "success": success,
                    "reviewer": _string(
                        row.get("reviewer")
                        or row.get("reviewer_id")
                        or row.get("source")
                        or source_name
                    )
                    or source_name,
                    "source_artifact": f"{source_name}.json",
                }
            )
    return verdicts


def _wam_score_claim_payload_for_buyer_report(
    *,
    wam_eval: Mapping[str, Any],
    wam_scorecard: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any] | None:
    if not wam_eval:
        return None
    explicit = _mapping(
        wam_scorecard.get("wam_score_claim")
        or wam_scorecard.get("wam_score_claim_gate")
        or wam_eval.get("wam_score_claim")
        or wam_eval.get("wam_score_claim_gate")
    )
    if explicit:
        return explicit

    review_grade = wam_scorecard.get("review_grade_policy_ranking") is True
    grade = "review_grade" if review_grade else "fixture_evaluator_only"
    consistency_summary = _mapping(
        wam_scorecard.get("forward_inverse_consistency_signal_summary")
    )
    return {
        "schema_version": WAM_SCORE_CLAIM_GATE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "granted",
        "requested_grade": grade,
        "granted_grade": grade,
        "max_allowed_grade": grade,
        "fixture_evaluator_only": not review_grade,
        "consistency_measured_and_passed": False,
        "calibration_anchors_present_and_passed": False,
        "consistency": {
            "status": _string(consistency_summary.get("status")) or "missing",
            "consistency_score": consistency_summary.get("consistency_score"),
            "passed": False,
            "blockers": _string_list(consistency_summary.get("blockers")),
        },
        "calibration_anchors": {
            "anchors_present": False,
            "anchors_passed": False,
            "anchor_set": [],
            "anchor_validation_status": "not_measured",
        },
        "blockers": _string_list(wam_scorecard.get("comparison_blockers"))
        + _string_list(wam_scorecard.get("visual_review_blockers")),
        "claim_boundary": {
            "derived_from_policy_ranking_scorecard": True,
            "grade_is_evaluator_bounded_not_rank_fidelity": True,
            "rank_fidelity_result_proven": False,
        },
    }


def _wam_success_label_provenance_for_buyer_report(
    *,
    wam_eval: Mapping[str, Any],
    wam_scorecard: Mapping[str, Any],
) -> Dict[str, Any]:
    score_source = _string(
        wam_scorecard.get("score_source")
        or wam_scorecard.get("ranking_basis")
        or wam_eval.get("score_source")
    )
    generated_video_judge = bool(
        wam_scorecard.get("score_source_is_generated_video_judge")
        or _mapping(wam_scorecard.get("claim_boundary")).get(
            "score_source_is_generated_video_judge"
        )
        or "generated_video" in score_source
        or "vlm" in score_source
    )
    disclosure = (
        "WAM success labels and score rates are judgments over model-derived generated "
        "rollout video; they are not measured physical robot success, simulator "
        "contact-state proof, or rank-fidelity proof."
        if generated_video_judge
        else "WAM score provenance is evaluator-bounded and must be displayed with its source."
    )
    raw_rows = (
        wam_scorecard.get("scorecard_rows")
        or wam_scorecard.get("policy_scores")
        or wam_scorecard.get("ranked_policies")
        or wam_scorecard.get("per_policy_coverage")
        or []
    )
    rows: List[Dict[str, Any]] = []
    if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes, bytearray)):
        for index, row in enumerate(raw_rows, start=1):
            if not isinstance(row, Mapping):
                continue
            rows.append(
                {
                    "row_id": _string(row.get("row_id") or row.get("policy_id"))
                    or f"policy_score_row_{index:04d}",
                    "policy_id": _string(row.get("policy_id") or row.get("policyId")) or None,
                    "success_rate": row.get("success_rate")
                    if row.get("success_rate") is not None
                    else row.get("score"),
                    "score_source": score_source or None,
                    "success_label_provenance_type": "generated_video_vlm_judge"
                    if generated_video_judge
                    else "evaluator_bounded",
                    "buyer_disclosure": disclosure,
                    "generated_video_vlm_judge": generated_video_judge,
                    "real_world_task_success_proven": False,
                    "rank_fidelity_result_proven": False,
                }
            )
    if not rows:
        rows.append(
            {
                "row_id": "wam_scorecard_summary",
                "policy_id": _string(wam_scorecard.get("top_policy_id")) or None,
                "success_rate": wam_scorecard.get("success_rate"),
                "score_source": score_source or None,
                "success_label_provenance_type": "generated_video_vlm_judge"
                if generated_video_judge
                else "evaluator_bounded",
                "buyer_disclosure": disclosure,
                "generated_video_vlm_judge": generated_video_judge,
                "real_world_task_success_proven": False,
                "rank_fidelity_result_proven": False,
            }
        )
    return {
        "schema_version": "wam_success_label_provenance_disclosure.v1",
        "status": "disclosed" if rows else "not_available",
        "score_source": score_source or None,
        "generated_video_vlm_judge": generated_video_judge,
        "success_rate_requires_provenance_disclosure": True,
        "success_rate_provenance_disclosed": bool(rows),
        "success_rate_buyer_display_allowed": bool(rows),
        "buyer_disclosure": disclosure,
        "rows": rows,
        "claim_boundary": {
            "generated_video_success_labels_are_support_evidence": generated_video_judge,
            "real_world_task_success_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _write_task_eval_run_buyer_report(
    *,
    job_dir: Path,
    job_id: str,
    scene_id: str,
    capture_id: str,
    request: Mapping[str, Any],
    scenario_eval_matrix: Mapping[str, Any],
    simulator_result: Mapping[str, Any],
    copied_artifacts: Mapping[str, Mapping[str, Any]],
    robot_pov_manifest: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
    policy_execution_manifest: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
    proof_boundary: Mapping[str, Any],
    live_closure: Mapping[str, Any],
    gpu_result: Mapping[str, Any],
    gpu_cost_ledger: Mapping[str, Any],
    remote_cloud_closure: Mapping[str, Any],
    wam_eval_result: Mapping[str, Any] | None = None,
    generated_at: str,
) -> Dict[str, Any]:
    trace = _mapping(copied_artifacts.get("normalized_attempt_trace")) or _read_optional_mapping(
        job_dir / "normalized_attempt_trace.json"
    )
    attempts = _attempts_for_task_eval_report(trace)
    task_metadata = _task_eval_task_metadata(
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
    )
    visual_coverage = _read_optional_mapping(
        job_dir / "simulator_command_batch_visual_media_coverage.json"
    )
    visual_media_present = bool(
        visual_coverage.get("all_required_runs_have_visual_recording")
        or visual_coverage.get("all_required_runs_have_robot_pov_video")
        or visual_coverage.get("all_required_runs_have_third_person_video")
    )
    media_decodable = coerce_strict_success(
        visual_coverage.get("all_required_videos_decodable")
        if "all_required_videos_decodable" in visual_coverage
        else visual_coverage.get("all_videos_decodable")
    )
    media_validity = build_media_validity(
        media_present=visual_media_present,
        decodable=media_decodable,
        visual_stats={
            "blockers": _string_list(visual_coverage.get("blockers")),
        },
        freshness=_freshness_for_artifact(
            path=job_dir / "simulator_command_batch_visual_media_coverage.json",
            generated_at=generated_at,
        )
        if visual_media_present
        else None,
    )
    review_task_success = build_review_task_success(
        media_validity=media_validity,
        reviewer_verdicts=_task_eval_review_verdicts(job_dir),
        camera_evidence={
            "robot_pov_camera_mode": _string(
                robot_pov_manifest.get("robot_pov_camera_mode")
                or robot_pov_manifest.get("camera_mode")
            ),
            "visible_embodied_robot_action_evidence": coerce_strict_success(
                robot_pov_manifest.get("visible_embodied_robot_action_evidence")
                if "visible_embodied_robot_action_evidence" in robot_pov_manifest
                else robot_pov_manifest.get("robot_pov_evidence_proven")
            ),
        },
    )
    task_success_contract = build_task_success_contract_result(
        task_metadata=task_metadata,
        trace_task_success=_trace_task_success_for_ledger(attempts),
        contract_evidence=None,
        reach_evidence=None,
    )
    normalized_trace_path = job_dir / "normalized_attempt_trace.json"
    simulator_execution = build_simulator_execution(
        provider_runtime_status=_string(simulator_result.get("status"))
        or _string(evaluation_result.get("status")),
        output_artifacts_present=bool(attempts),
        artifact_freshness=_freshness_for_artifact(
            path=normalized_trace_path,
            generated_at=generated_at,
        )
        if normalized_trace_path.is_file()
        else None,
        frames_rendered=int(_number(visual_coverage.get("frame_count")) or 0)
        if "frame_count" in visual_coverage
        else None,
        execution_log_present=None,
    )
    policy_id = _first_policy_id(
        request=request,
        policy_manifest=policy_manifest,
        attempts=attempts,
    )
    policy_action_execution = build_policy_action_execution(
        action_source="learned_policy"
        if policy_execution_manifest.get("robot_policy_execution_proven") is True
        else _string(policy_execution_manifest.get("action_source")),
        policy_id=policy_id,
        action_trace_present=(job_dir / "policy_execution_trace.json").is_file()
        or (job_dir / "policy_execution_trace.jsonl").is_file(),
        actions_executed_in_simulator=policy_execution_manifest.get(
            "robot_policy_execution_proven"
        ),
    )
    contact_state_change = build_contact_state_change_proof(
        proof_requirements=derive_task_proof_requirements(task_metadata),
        contact_reports=[],
        state_change_measurement=None,
    )
    physical_readiness = build_physical_readiness(
        real_robot_execution_evidence={
            "physical_robot_executed": proof_boundary.get(
                "physical_robot_readiness_proven"
            )
            or proof_boundary.get("real_world_outcome_proven"),
            "run_manifest_uri": proof_boundary.get("physical_run_manifest_uri"),
        },
        deployment_approval={
            "approved": proof_boundary.get("deployment_approval_proven"),
            "approver": proof_boundary.get("deployment_approver"),
        },
    )
    rights_privacy_gate = {
        "status": "cleared"
        if proof_boundary.get("rights_privacy_scope_proven") is True
        else "not_cleared",
        "cleared": proof_boundary.get("rights_privacy_scope_proven") is True,
        "signed_delivery_access_proven": proof_boundary.get(
            "signed_delivery_access_proven"
        )
        is True,
        "live_closure_status": live_closure.get("status"),
    }
    wam_eval = _mapping(wam_eval_result)
    wam_provider_execution = _mapping(wam_eval.get("wam_provider_execution_manifest"))
    wam_policy_binding = _mapping(wam_eval.get("wam_policy_interface_binding"))
    wam_scorecard = _mapping(wam_eval.get("policy_ranking_scorecard"))
    wam_claim_boundary = _mapping(wam_eval.get("wam_eval_claim_boundary"))
    wam_task_eval_report = _mapping(wam_eval.get("task_eval_run_report"))
    wam_score_claim_payload = _wam_score_claim_payload_for_buyer_report(
        wam_eval=wam_eval,
        wam_scorecard=wam_scorecard,
        generated_at=generated_at,
    )
    wam_success_label_provenance = _wam_success_label_provenance_for_buyer_report(
        wam_eval=wam_eval,
        wam_scorecard=wam_scorecard,
    ) if wam_eval else {}
    provider_execution: Dict[str, Any] = {
        "gpu_provisioning_status": gpu_result.get("status"),
        "gpu_cost_control_ledger_status": gpu_cost_ledger.get("status"),
        "remote_cloud_execution_status": remote_cloud_closure.get("status"),
        "simulator_service_status": simulator_result.get("status"),
        "simulator_framework": simulator_result.get("framework"),
        "evaluation_substrate": _string(
            request.get("evaluation_substrate") or request.get("evaluationSubstrate")
        )
        or None,
        "evaluation_status": evaluation_result.get("status"),
        "live_eval_closure_status": live_closure.get("status"),
    }
    if wam_eval:
        provider_execution.update(
            {
                "evaluation_substrate": wam_eval.get("evaluation_substrate"),
                "wam_evaluation_status": wam_eval.get("status"),
                "wam_provider_execution_status": wam_provider_execution.get("status"),
                "wam_provider_command_used": bool(
                    wam_provider_execution.get("provider_command_used")
                ),
                "wam_policy_ranking_scorecard_status": wam_scorecard.get("status"),
                "wam_task_eval_run_report_status": wam_task_eval_report.get("status"),
                "wam_task_eval_run_report_evidence_level": wam_task_eval_report.get(
                    "evidence_level"
                ),
                "generated_wam_rollouts_are_model_derived_support_artifacts": True,
                "wam_evaluator_bounded_policy_ranking_only": True,
            }
        )
    policy_binding_payload: Dict[str, Any] = {
        "policy_package_status": policy_manifest.get("status"),
        "policy_execution_status": policy_execution_manifest.get("status"),
        "policy_id": policy_id,
    }
    if wam_policy_binding:
        policy_binding_payload["wam_policy_interface_binding_status"] = (
            wam_policy_binding.get("status")
        )
        policy_binding_payload["wam_policy_interface_binding_path"] = (
            "wam_policy_interface_binding.json"
        )
    report = build_task_eval_run_report(
        job_id=job_id,
        scene_id=scene_id,
        capture_id=capture_id,
        attempt_trace={**trace, "attempts": attempts},
        task_metadata=task_metadata,
        success_claim_layers={
            "media_validity": media_validity,
            "review_task_success": review_task_success,
            "task_success_contract": task_success_contract,
            "simulator_execution": simulator_execution,
            "policy_action_execution": policy_action_execution,
            "contact_state_change": contact_state_change,
            "physical_readiness": physical_readiness,
        },
        provider_execution=provider_execution,
        policy_binding=policy_binding_payload,
        rights_privacy_gate=rights_privacy_gate,
        wam_evaluation=wam_score_claim_payload,
        buyer_claim_proof_boundary={
            "live_simulator_execution_proven": _mapping(
                live_closure.get("proof_boundary")
            ).get("simulator_execution_proven")
            is True,
            "live_policy_execution_proven": _mapping(
                live_closure.get("proof_boundary")
            ).get("robot_policy_execution_proven")
            is True,
        },
        live_closure=live_closure,
        buyer_claim_copy={
            "request": {
                "buyer_facing_copy": request.get("buyer_facing_copy"),
                "marketing_copy": request.get("marketing_copy"),
                "report_copy": request.get("report_copy"),
                "public_claims": request.get("public_claims"),
            },
        },
        # Live consent re-read at buyer-report emit closes the revoke-after-manifest window.
        capture_root=_capture_root_from_job_dir(job_dir),
        generated_at=generated_at,
    )
    if wam_eval:
        composer_wam_section = _mapping(report.get("wam_evaluation"))
        report = {
            **dict(report),
            "wam_evaluation": {
                **composer_wam_section,
                "status": wam_eval.get("status"),
                "evaluation_substrate": wam_eval.get("evaluation_substrate"),
                "provider_execution_status": wam_provider_execution.get("status"),
                "provider_command_used": bool(
                    wam_provider_execution.get("provider_command_used")
                ),
                "policy_ranking_scorecard_status": wam_scorecard.get("status"),
                "success_label_provenance": wam_success_label_provenance,
                "claim_boundary_path": "wam_eval_claim_boundary.json"
                if wam_claim_boundary
                else None,
                "task_eval_run_report_source_status": wam_task_eval_report.get(
                    "status"
                ),
                "task_eval_run_report_source_evidence_level": wam_task_eval_report.get(
                    "evidence_level"
                ),
                "artifacts_are_support_outputs_not_deployment_claims": True,
            },
            "claim_boundary": {
                **_mapping(report.get("claim_boundary")),
                "generated_wam_rollouts_are_model_derived_support_artifacts": True,
                "wam_evaluator_bounded_policy_ranking_is_not_task_success": True,
                "wam_success_rate_requires_label_provenance_disclosure": True,
                "wam_success_label_provenance_disclosed": bool(
                    wam_success_label_provenance.get("success_rate_provenance_disclosed")
                ),
                "wam_policy_ranking_is_not_evaluation_readiness": True,
                "wam_outputs_do_not_prove_physical_readiness": True,
            },
        }
    _write_job_json(job_dir, "task_eval_run_report.json", report)
    return report


def _proof_boundary(
    *,
    simulator: str,
    simulator_result: Mapping[str, Any],
    training_result: Mapping[str, Any],
    policy_execution_manifest: Mapping[str, Any] | None = None,
    deployment_outcome_ledger: Mapping[str, Any] | None = None,
    generated_at: str,
) -> Dict[str, Any]:
    training_completed = _strict_bool(training_result.get("training_completed"))
    simulator_proven = (
        _strict_bool(simulator_result.get("simulator_execution_proven"))
        and simulator != "fixture"
    )
    policy_execution_proven = _strict_bool(
        _mapping(policy_execution_manifest).get("robot_policy_execution_proven")
    )
    real_world_outcome_proven = _strict_bool(
        _mapping(deployment_outcome_ledger).get("real_world_outcome_proven")
    )
    real_world_outcome_records_present = _strict_bool(
        _mapping(deployment_outcome_ledger).get("real_world_outcome_records_present")
    )
    owner_evidence_record_count = int(
        _mapping(deployment_outcome_ledger).get("owner_evidence_record_count") or 0
    )
    missing_owner_evidence_record_ids = _string_list(
        _mapping(deployment_outcome_ledger).get("missing_owner_evidence_record_ids")
    )
    remaining = list(CLAIM_BOUNDARY["proof_upgrade_requires"])
    if policy_execution_proven:
        remaining = [
            item
            for item in remaining
            if item != "owner-system robot policy, teleoperation, or action logs"
        ]
    if real_world_outcome_proven:
        remaining = [item for item in remaining if item != "actual outcome records"]
    return {
        "schema_version": PROOF_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_only",
        "fixture_only_proof": simulator == "fixture",
        "gpu_provisioning_performed": False,
        "simulators_run": _strict_bool(simulator_result.get("simulators_run")),
        "gpu_training_run": _strict_bool(training_result.get("gpu_training_run")),
        "review_acceptance_proven": False,
        "rights_privacy_scope_proven": False,
        "signed_delivery_access_proven": False,
        "customer_handoff_ready": False,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "safety_validation_proven": False,
        "simulator_execution_proven": simulator_proven,
        "rank_fidelity_result_proven": False,
        "robot_policy_execution_proven": policy_execution_proven,
        "real_world_outcome_records_present": real_world_outcome_records_present,
        "owner_evidence_record_count": owner_evidence_record_count,
        "missing_owner_evidence_record_ids": missing_owner_evidence_record_ids,
        "real_world_outcome_proven": real_world_outcome_proven,
        "physics_contact_validated": False,
        "non_ranking_operational_claim_validated": False,
        "training_completed": training_completed,
        "public_claim_upgrade_allowed": False,
        "remaining_required_evidence": remaining,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulators_run": _strict_bool(simulator_result.get("simulators_run")),
            "gpu_training_run": _strict_bool(training_result.get("gpu_training_run")),
            "review_acceptance_proven": False,
            "rights_privacy_scope_proven": False,
            "signed_delivery_access_proven": False,
            "customer_handoff_ready": False,
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": False,
            "simulator_execution_proven": simulator_proven,
            "robot_policy_execution_proven": policy_execution_proven,
            "real_world_outcome_records_present": real_world_outcome_records_present,
            "training_completed": training_completed,
        },
    }


def _apply_live_closure_to_proof_boundary(
    *,
    proof_boundary: Mapping[str, Any],
    live_closure: Mapping[str, Any],
) -> Dict[str, Any]:
    closure_boundary = _mapping(live_closure.get("proof_boundary"))
    live_end_to_end_verified = _strict_bool(live_closure.get("live_end_to_end_verified"))
    closure_live_end_to_end_verified = _strict_bool(
        closure_boundary.get("live_end_to_end_verified")
    )
    updated = {
        **dict(proof_boundary),
        "live_eval_closure_status": live_closure.get("status"),
        "live_eval_closure_manifest_path": "live_eval_closure_manifest.json",
        "live_end_to_end_verified": live_end_to_end_verified,
        "live_eval_closure_blockers": _string_list(live_closure.get("blockers")),
        "review_acceptance_proven": _strict_bool(
            closure_boundary.get("review_acceptance_proven")
        ),
        "rights_privacy_scope_proven": _strict_bool(
            closure_boundary.get("rights_privacy_scope_proven")
        ),
        "signed_delivery_access_proven": _strict_bool(
            closure_boundary.get("signed_delivery_access_proven")
        ),
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "safety_validation_proven": _strict_bool(
            closure_boundary.get("safety_validation_proven")
        ),
    }
    if closure_live_end_to_end_verified:
        for field in (
            "simulator_execution_proven",
            "robot_policy_execution_proven",
            "real_world_outcome_proven",
            "physics_contact_validated",
            "non_ranking_operational_claim_validated",
            "rank_fidelity_result_proven",
            "public_claim_upgrade_allowed",
        ):
            updated[field] = _strict_bool(closure_boundary.get(field))
        updated["status"] = "live_end_to_end_verified"
        updated["remaining_required_evidence"] = []
    updated["claim_boundary"] = {
        **_mapping(updated.get("claim_boundary")),
        "live_eval_closure_manifest_path": "live_eval_closure_manifest.json",
        "live_end_to_end_verified": live_end_to_end_verified,
        "review_acceptance_proven": _strict_bool(updated.get("review_acceptance_proven")),
        "rights_privacy_scope_proven": _strict_bool(
            updated.get("rights_privacy_scope_proven")
        ),
        "signed_delivery_access_proven": _strict_bool(
            updated.get("signed_delivery_access_proven")
        ),
        "customer_handoff_ready": _strict_bool(updated.get("customer_handoff_ready")),
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "safety_validation_proven": _strict_bool(updated.get("safety_validation_proven")),
        "simulator_execution_proven": _strict_bool(
            updated.get("simulator_execution_proven")
        ),
        "robot_policy_execution_proven": _strict_bool(
            updated.get("robot_policy_execution_proven")
        ),
        "real_world_outcome_proven": _strict_bool(updated.get("real_world_outcome_proven")),
        "physics_contact_validated": _strict_bool(updated.get("physics_contact_validated")),
        "non_ranking_operational_claim_validated": _strict_bool(
            updated.get("non_ranking_operational_claim_validated")
        ),
        "rank_fidelity_result_proven": _strict_bool(
            updated.get("rank_fidelity_result_proven")
        ),
        "public_claim_upgrade_allowed": _strict_bool(
            updated.get("public_claim_upgrade_allowed")
        ),
    }
    return updated


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
    evaluation_run_plan: Mapping[str, Any],
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
        "evaluation_run": {
            "status": evaluation_run_plan.get("status"),
            "schema_version": evaluation_run_plan.get("schema_version"),
            "spec_digest": evaluation_run_plan.get("spec_digest"),
            "plan_path": "evaluation_run_plan.json",
            "spec_path": "evaluation_run_spec.json",
        },
        "state_machine": [
            "request_loaded",
            "validation",
            "evaluation_run_contract",
            "agent_orchestration_plan",
            "scheduler_decision",
            "worker_launch_plan",
        "gpu_provider_launch_request",
        "worker_manifest",
        "gpu_cost_control_ledger",
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




def build_robot_eval_job(
    *,
    capture_root: str | Path,
    job_request: str | Path | Mapping[str, Any],
    job_id: str,
    agent_adapter: RobotEvalJobAgentAdapter | None = None,
    provisioner: str = "fixture_local",
    simulator: str = "fixture",
    evaluation_substrate: str | None = None,
    allow_wam_provider: bool = False,
    wam_provider_command: str | None = None,
    wam_provider_commands: Mapping[str, str] | None = None,
    wam_artifact_output_uri: str | None = None,
    wam_provider_max_retries: int = 0,
    wam_provider_timeout_seconds: int | None = None,
    allow_gpu_provisioning: bool = False,
    allow_simulator_execution: bool = False,
    allowed_simulators: Sequence[str] = (),
    simulator_commands: Mapping[str, str] | None = None,
    allow_cpu_simulator_preflight: bool = False,
    cpu_preflight_backends: Sequence[str] = CPU_BACKENDS,
    cpu_preflight_smoke_steps: int = 10,
    allow_cpu_preflight_render: bool = False,
    allow_training: bool = False,
    training_command: str | None = None,
    allow_policy_execution: bool = False,
    policy_execution_commands: Mapping[str, str] | None = None,
    timeout_seconds: int = 120,
    budget_usd: float | None = None,
    arena_results_dir: str | Path | None = None,
    arena_scenario_count: int = 500,
    arena_shard_size: int = 50,
    arena_num_envs: int = 16,
    arena_retry_budget: int = 2,
    allow_rollout_vision_labeling: bool = False,
    vision_labeling_command: str | None = None,
    allow_delivery_upload: bool = False,
    delivery_command: str | None = None,
    arena_operator_mode: str = "none",
    allow_live_agents_sdk: bool = False,
    allow_live_codex_sdk: bool = False,
) -> Dict[str, Any]:
    if provisioner not in PROVISIONERS:
        raise ValueError(f"Unsupported provisioner: {provisioner}")
    if simulator not in SIMULATORS:
        raise ValueError(f"Unsupported simulator: {simulator}")
    canonical_job_id = strict_identifier(job_id, field="job_id")
    request = _read_job_request(job_request)
    declared_job_ids = {
        _string(request.get(key))
        for key in ("job_id", "jobId")
        if _string(request.get(key))
    }
    if len(declared_job_ids) > 1 or (
        declared_job_ids and canonical_job_id not in declared_job_ids
    ):
        raise ValueError("job_id_argument_request_mismatch")
    context = resolve_local_capture_context(capture_root)
    generated_at = utc_now_iso()
    pipeline_dir = context.pipeline_root
    job_root = pipeline_dir / "robot_eval_jobs"
    ensure_dir(job_root)
    job_dir = contained_path(
        job_root,
        canonical_job_id,
        field="robot_eval_job_dir",
    )
    request.setdefault("schema_version", JOB_REQUEST_SCHEMA_VERSION)
    request["job_id"] = canonical_job_id
    request.pop("jobId", None)
    request.setdefault("capture_root", str(context.capture_root))
    request = _apply_staged_policy_package(
        request=request,
        capture_root=context.capture_root,
        job_id=canonical_job_id,
    )
    request_fingerprint = _sha_payload(
        {
            "schema_version": "robot_eval_job_request_fingerprint.v1",
            "job_id": canonical_job_id,
            "capture_root": str(context.capture_root),
            "request": request,
            "execution_contract": {
                "provisioner": provisioner,
                "simulator": simulator,
                "evaluation_substrate": evaluation_substrate,
                "allow_wam_provider": allow_wam_provider,
                "wam_provider_commands": dict(wam_provider_commands or {}),
                "allow_gpu_provisioning": allow_gpu_provisioning,
                "allow_simulator_execution": allow_simulator_execution,
                "allowed_simulators": list(allowed_simulators),
                "simulator_commands": dict(simulator_commands or {}),
                "allow_training": allow_training,
                "training_command": training_command,
                "allow_policy_execution": allow_policy_execution,
                "policy_execution_commands": dict(policy_execution_commands or {}),
                "timeout_seconds": timeout_seconds,
                "budget_usd": budget_usd,
            },
        }
    )
    execution_claim = _claim_robot_eval_job_execution(
        job_dir=job_dir,
        job_id=canonical_job_id,
        request_fingerprint=request_fingerprint,
        generated_at=generated_at,
    )
    job_id = canonical_job_id
    source_request = dict(request)
    selected_evaluation_substrate = requested_evaluation_substrate(
        request,
        explicit=evaluation_substrate,
    )
    if selected_evaluation_substrate and not _string(
        request.get("evaluation_substrate") or request.get("evaluationSubstrate")
    ):
        request["evaluation_substrate"] = selected_evaluation_substrate

    missing_robot_eval_inputs = _ensure_robot_eval_cards(
        capture_root=context.capture_root,
        pipeline_dir=pipeline_dir,
    )
    request, request_enrichment = _enrich_incomplete_beta_job_request(
        request=request,
        pipeline_dir=pipeline_dir,
        job_id=job_id,
        generated_at=generated_at,
        simulator=simulator,
        evaluation_substrate=selected_evaluation_substrate,
    )
    _write_job_json(job_dir, "job_request_source.json", source_request)
    _write_job_json(job_dir, "job_request_enrichment_manifest.json", request_enrichment)
    _write_job_json(job_dir, "job_request.json", request)
    benchmark_protocol_status = execute_benchmark_protocol_request(
        request,
        output_dir=job_dir / "benchmark_protocol",
        allowed_root=context.capture_root,
    )
    scene_preflight = build_scene_asset_preflight(capture_root=context.capture_root)
    episode_specs = build_episode_specs(capture_root=context.capture_root)
    cpu_preflight = build_cpu_simulator_preflight(
        capture_root=context.capture_root,
        allow_cpu_simulator_preflight=allow_cpu_simulator_preflight,
        backends=cpu_preflight_backends,
        smoke_steps=cpu_preflight_smoke_steps,
        allow_render=allow_cpu_preflight_render,
    )
    simulation_automation = build_simulation_automation(
        capture_root=context.capture_root,
        allow_cpu_simulator_preflight=allow_cpu_simulator_preflight,
        cpu_preflight_backends=cpu_preflight_backends,
        cpu_preflight_smoke_steps=cpu_preflight_smoke_steps,
        allow_cpu_preflight_render=allow_cpu_preflight_render,
    )
    owner_gpu_cpu_preflight = (
        _read_optional_mapping(pipeline_dir / "simulation_automation" / "cpu_preflight_manifest.json")
        or cpu_preflight
    )
    scenario_eval_matrix = build_scenario_eval_matrix(
        capture_root=context.capture_root,
        job_dir=job_dir,
        job_request=request,
        generated_at=generated_at,
    )
    effective_simulator_commands = dict(simulator_commands or {})
    selected_wam_provider_commands = dict(wam_provider_commands or {})
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
        benchmark_protocol_status=benchmark_protocol_status,
    )
    _write_job_json(job_dir, "job_validation.json", validation)
    evaluation_run_spec = build_robot_eval_evaluation_run_spec(
        job_id=job_id,
        request=request,
        capture_root=context.capture_root,
        scene_preflight=scene_preflight,
        scenario_eval_matrix=scenario_eval_matrix,
        policy_manifest=policy_manifest,
        provisioner=provisioner,
        simulator=simulator,
        budget_usd=budget_usd,
        timeout_seconds=timeout_seconds,
    )
    evaluation_run_plan = compile_evaluation_run(
        evaluation_run_spec,
        output_dir=job_dir,
        generated_at=generated_at,
    )
    validation["evaluation_run_contract"] = {
        "status": evaluation_run_plan["status"],
        "schema_version": evaluation_run_plan["schema_version"],
        "spec_digest": evaluation_run_plan["spec_digest"],
        "component_bindings": evaluation_run_plan["component_bindings"],
        "warnings": evaluation_run_plan["validation"]["warnings"],
        "plan_path": "evaluation_run_plan.json",
        "spec_path": "evaluation_run_spec.json",
    }
    if evaluation_run_plan["status"] != "prepared":
        validation["status"] = "blocked"
        validation["blockers"] = _dedupe(
            [
                *_string_list(validation.get("blockers")),
                "evaluation_run_contract_blocked",
                *[
                    f"evaluation_run:{error}"
                    for error in evaluation_run_plan["validation"]["errors"]
                ],
            ]
        )
        validation["evaluation_run_contract"]["errors"] = evaluation_run_plan[
            "validation"
        ]["errors"]
    _write_job_json(job_dir, "job_validation.json", validation)
    robot_pov_manifest = build_robot_pov_observation_bundle(
        capture_root=context.capture_root,
        job_dir=job_dir,
        job_request=request,
        generated_at=generated_at,
        scenario_eval_matrix=scenario_eval_matrix,
    )
    policy_execution = build_policy_execution_bundle(
        capture_root=context.capture_root,
        job_dir=job_dir,
        job_request=request,
        observation_manifest=robot_pov_manifest,
        allow_policy_execution=allow_policy_execution and validation.get("status") != "blocked",
        allow_reference_replay=validation.get("status") != "blocked",
        policy_execution_commands=policy_execution_commands or {},
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    policy_trace = _mapping(policy_execution.get("trace"))
    policy_package_request = _mapping(
        request.get("policy_package") or request.get("policyPackage")
    )
    action_space = _mapping(
        request.get("action_space")
        or request.get("actionSpace")
        or policy_package_request.get("action_space")
        or policy_package_request.get("actionSpace")
    )
    if not action_space:
        for modality in POLICY_MODALITY_ORDER:
            modality_payload = _modality_payload(policy_package_request, modality)
            action_space = _mapping(
                modality_payload.get("action_space")
                or modality_payload.get("actionSpace")
            )
            if action_space:
                break
    build_action_normalization_from_trace(
        output_dir=job_dir,
        trace=policy_trace,
        source_trace_path=job_dir / "policy_execution_trace.json",
        consumed_by="robot_eval_policy_execution_and_sc3_protocol",
        action_space=action_space,
    )

    plan_context = {
        "repo_root": str(Path(__file__).resolve().parents[2]),
        "capture_root": str(context.capture_root),
        "job_id": job_id,
        "request": request,
        "validation": validation,
        "provisioner": provisioner,
        "simulator": simulator,
        "scene_preflight": scene_preflight,
        "episode_specs": episode_specs,
        "cpu_preflight": cpu_preflight,
        "owner_gpu_cpu_preflight": owner_gpu_cpu_preflight,
        "simulation_automation": simulation_automation,
        "scenario_eval_matrix": scenario_eval_matrix,
        "evaluation_run_plan": evaluation_run_plan,
        "robot_pov_observation_manifest": robot_pov_manifest,
        "policy_execution_manifest": _mapping(policy_execution.get("manifest")),
        "policy_execution_trace": _mapping(policy_execution.get("trace")),
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
        evaluation_run_plan=evaluation_run_plan,
    )
    _write_job_json(job_dir, "job_plan.json", job_plan)

    scheduler_decision = _build_scheduler_decision(
        request=request,
        job_id=job_id,
        provisioner=provisioner,
        simulator=simulator,
        pipeline_dir=pipeline_dir,
        cpu_preflight=owner_gpu_cpu_preflight,
        budget_usd=budget_usd,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "scheduler_decision.json", scheduler_decision)

    gpu_request = _gpu_provisioning_request(
        request=request,
        job_id=job_id,
        provisioner=provisioner,
        budget_usd=budget_usd,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "gpu_provisioning_request.json", gpu_request)
    worker_launch_plan = _build_worker_launch_plan(
        request=request,
        job_id=job_id,
        provisioner=provisioner,
        simulator=simulator,
        scheduler_decision=scheduler_decision,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "worker_launch_plan.json", worker_launch_plan)
    gpu_startup_pipeline_plan = build_gpu_startup_pipeline_plan(
        request=request,
        job_id=job_id,
        provisioner=provisioner,
        simulator=simulator,
        scheduler_decision=scheduler_decision,
        worker_launch_plan=worker_launch_plan,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "gpu_startup_pipeline_plan.json", gpu_startup_pipeline_plan)
    worker_manifest = _build_worker_manifest(
        request=request,
        job_id=job_id,
        capture_root=context.capture_root,
        provisioner=provisioner,
        simulator=simulator,
        evaluation_substrate=selected_evaluation_substrate or None,
        worker_launch_plan=worker_launch_plan,
        allowed_simulators=allowed_simulators,
        simulator_commands=effective_simulator_commands,
        allow_wam_provider=allow_wam_provider,
        wam_provider_commands=selected_wam_provider_commands,
        wam_artifact_output_uri=wam_artifact_output_uri,
        wam_provider_max_retries=wam_provider_max_retries,
        wam_provider_timeout_seconds=wam_provider_timeout_seconds,
        timeout_seconds=timeout_seconds,
        budget_usd=budget_usd,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "worker_manifest.json", worker_manifest)
    provider_launch_request = _build_gpu_provider_launch_request(
        request_manifest=gpu_request,
        scheduler_decision=scheduler_decision,
        worker_launch_plan=worker_launch_plan,
        worker_manifest=worker_manifest,
        allow_gpu_provisioning=allow_gpu_provisioning,
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=allowed_simulators,
        simulator_commands=effective_simulator_commands,
        evaluation_substrate=selected_evaluation_substrate or None,
        allow_wam_provider=allow_wam_provider,
        wam_provider_commands=selected_wam_provider_commands,
        wam_artifact_output_uri=wam_artifact_output_uri,
        wam_provider_max_retries=wam_provider_max_retries,
        wam_provider_timeout_seconds=wam_provider_timeout_seconds,
        generated_at=generated_at,
        gpu_startup_pipeline_plan=gpu_startup_pipeline_plan,
    )
    _write_job_json(job_dir, "gpu_provider_launch_request.json", provider_launch_request)
    provider_race_handoff = _build_gpu_provider_race_handoff(
        provider_launch_request=provider_launch_request,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "gpu_provider_race_handoff.json", provider_race_handoff)
    gpu_result = _gpu_provisioning_result(
        request_manifest=gpu_request,
        validation=validation,
        scheduler_decision=scheduler_decision,
        worker_launch_plan=worker_launch_plan,
        provider_launch_request=provider_launch_request,
        allow_gpu_provisioning=allow_gpu_provisioning,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "gpu_provisioning_result.json", gpu_result)

    sim_request = _simulator_request(
        job_id=job_id,
        simulator=simulator,
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
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
        simulator_commands=effective_simulator_commands,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "simulator_service_result.json", sim_result)
    if simulator == "fixture":
        copied_artifacts = _expand_fixture_artifacts_to_scenario_eval_runs(
            copied_artifacts=copied_artifacts,
            scenario_eval_matrix=scenario_eval_matrix,
            job_dir=job_dir,
            generated_at=generated_at,
        )

    arena_ingest: Dict[str, Any] = {}
    if simulator == "isaac_lab_arena" or arena_results_dir:
        arena_ingest = build_arena_result_ingest(
            capture_root=context.capture_root,
            job_dir=job_dir,
            arena_results_dir=arena_results_dir
            or request.get("arena_results_dir")
            or request.get("arenaResultsDir"),
            output_dir=job_dir,
            job_request=request,
            scenario_count=arena_scenario_count,
            shard_size=arena_shard_size,
            num_envs=arena_num_envs,
            timeout_seconds=timeout_seconds,
            retry_budget=arena_retry_budget,
            cost_budget_usd=budget_usd,
            allow_rollout_vision_labeling=allow_rollout_vision_labeling,
            vision_labeling_command=vision_labeling_command,
            allow_delivery_upload=allow_delivery_upload,
            delivery_command=delivery_command,
            operator_mode=arena_operator_mode,
            allow_live_agents_sdk=allow_live_agents_sdk,
            allow_live_codex_sdk=allow_live_codex_sdk,
        )
        copied_artifacts["normalized_attempt_trace"] = _mapping(arena_ingest.get("attempt_trace"))
        copied_artifacts["failure_labels"] = _mapping(arena_ingest.get("failure_labels"))
        copied_artifacts["arena_eval_metrics"] = _mapping(arena_ingest.get("metrics"))
        if (
            _mapping(arena_ingest.get("run_manifest")).get("status") == "completed"
            and sim_result.get("status") == "blocked"
        ):
            sim_result = {
                **dict(sim_result),
                "status": "completed_from_supplied_arena_results",
                "reason": "arena_results_dir_ingested_without_running_simulator_command",
                "blockers": [],
                "artifact_paths": {
                    **_mapping(sim_result.get("artifact_paths")),
                    "arena_result_ingest_run_manifest": "arena_result_ingest_run_manifest.json",
                    "normalized_attempt_trace": "normalized_attempt_trace.json",
                    "failure_labels": "failure_labels.json",
                },
                "execution_performed": False,
                "simulators_run": False,
                "simulator_execution_proven": False,
                "robot_policy_execution_proven": False,
            }
            simulator_blockers = []
            _write_job_json(job_dir, "simulator_service_result.json", sim_result)

    wam_eval_result: Dict[str, Any] = {}
    wam_eval_blockers: List[str] = []
    if selected_evaluation_substrate in WAM_EVALUATION_SUBSTRATES:
        wam_eval_result = run_wam_eval_job(
            capture_root=context.capture_root,
            job_dir=job_dir,
            evaluation_substrate=selected_evaluation_substrate,
            allow_live_provider=allow_wam_provider,
            provider_command=(
                _string(selected_wam_provider_commands.get(selected_evaluation_substrate))
                or _string(wam_provider_command)
            ),
            artifact_output_uri=wam_artifact_output_uri,
            budget_usd=budget_usd,
            max_retries=wam_provider_max_retries,
            timeout_seconds=wam_provider_timeout_seconds or timeout_seconds,
            generated_at=generated_at,
        )
        for artifact_key in (
            "normalized_attempt_trace",
            "failure_labels",
            "prediction_outcome_ledger",
            "calibration_report",
            "breakage_library",
        ):
            artifact_payload = _mapping(wam_eval_result.get(artifact_key))
            if artifact_payload:
                copied_artifacts[artifact_key] = artifact_payload
        if wam_eval_result.get("status") != "completed":
            wam_eval_blockers = _string_list(wam_eval_result.get("blockers")) or [
                "wam_evaluation_blocked"
            ]

    simulator_policy_scorecard = _write_simulator_policy_ranking_scorecard(
        job_dir=job_dir,
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result=sim_result,
        copied_artifacts=copied_artifacts,
        generated_at=generated_at,
    )

    gpu_cost_ledger = _gpu_cost_control_ledger(
        request=request,
        scheduler_decision=scheduler_decision,
        worker_launch_plan=worker_launch_plan,
        provider_launch_request=provider_launch_request,
        gpu_result=gpu_result,
        sim_result=sim_result,
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "gpu_cost_control_ledger.json", gpu_cost_ledger)
    gpu_result = {
        **dict(gpu_result),
        "gpu_cost_control_ledger_path": "gpu_cost_control_ledger.json",
        "gpu_cost_control_ledger_status": gpu_cost_ledger.get("status"),
    }
    _write_job_json(job_dir, "gpu_provisioning_result.json", gpu_result)
    remote_cloud_closure = _remote_cloud_execution_closure_manifest(
        job_id=job_id,
        provisioner=provisioner,
        simulator=simulator,
        worker_launch_plan=worker_launch_plan,
        worker_manifest=worker_manifest,
        provider_launch_request=provider_launch_request,
        gpu_result=gpu_result,
        gpu_cost_ledger=gpu_cost_ledger,
        sim_result=sim_result,
        generated_at=generated_at,
    )
    _write_job_json(
        job_dir,
        "remote_cloud_execution_closure_manifest.json",
        remote_cloud_closure,
    )

    robot_pov_manifest = build_robot_pov_observation_bundle(
        capture_root=context.capture_root,
        job_dir=job_dir,
        job_request=request,
        generated_at=generated_at,
        attempt_trace=_mapping(copied_artifacts.get("normalized_attempt_trace")),
    )

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
    if wam_eval_result:
        eval_result = {
            **dict(eval_result),
            "evaluation_substrate": selected_evaluation_substrate,
            "wam_evaluation_status": wam_eval_result.get("status"),
            "wam_rollout_manifest_path": "wam_rollout_manifest.json",
            "wam_rollout_results_path": "wam_rollout_results.json",
            "vision_success_labels_path": "vision_success_labels.json",
            "policy_ranking_scorecard_path": "policy_ranking_scorecard.json",
            "wam_eval_claim_boundary_path": "wam_eval_claim_boundary.json",
            "real_world_validation_followup_request_path": (
                "real_world_validation_followup_request.json"
            ),
            "srcc_validation_plan_path": "srcc_validation_plan.json",
            "claim_boundary": {
                **_mapping(eval_result.get("claim_boundary")),
                "wam_eval_claim_boundary_path": "wam_eval_claim_boundary.json",
                "primary_proof_target": "policy_comparison_within_configured_evaluator",
                "policy_ranking_is_evaluator_bounded": True,
                "policy_ranking_is_not_evaluation_readiness": True,
                "traditional_sim_is_optional_cross_check_for_wam_eval": True,
                "generated_wam_rollouts_are_model_derived_support_artifacts": True,
                "customer_specific_srcc_claimed": False,
                "passing_wam_eval_is_not_rank_fidelity_result": True,
            },
        }
    elif simulator_policy_scorecard:
        eval_result = {
            **dict(eval_result),
            "evaluation_substrate": simulator_policy_scorecard.get("evaluation_substrate"),
            "policy_ranking_scorecard_path": "policy_ranking_scorecard.json",
            "candidate_selection_report_path": "candidate_selection_report.json",
            "wam_eval_claim_boundary_path": "wam_eval_claim_boundary.json",
            "claim_boundary": {
                **_mapping(eval_result.get("claim_boundary")),
                "wam_eval_claim_boundary_path": "wam_eval_claim_boundary.json",
                "primary_proof_target": "policy_comparison_within_configured_evaluator",
                "policy_ranking_is_evaluator_bounded": True,
                "policy_ranking_is_not_evaluation_readiness": True,
                "simulator_evaluator_only": True,
                "customer_specific_srcc_claimed": False,
                "passing_simulator_eval_is_not_rank_fidelity_result": True,
            },
        }
    _write_job_json(job_dir, "evaluation_result.json", eval_result)
    prediction_ledger = _mapping(copied_artifacts.get("prediction_outcome_ledger"))
    if not prediction_ledger:
        prediction_ledger = _read_optional_mapping(job_dir / "prediction_outcome_ledger.json")
    deployment_validation = build_deployment_validation_bundle(
        capture_root=context.capture_root,
        job_dir=job_dir,
        job_request=request,
        prediction_ledger=prediction_ledger,
        attempt_trace=_mapping(copied_artifacts.get("normalized_attempt_trace")),
        generated_at=generated_at,
    )
    followup_request_queue = _build_real_world_validation_followup_request_queue(
        capture_root=context.capture_root,
        pipeline_dir=pipeline_dir,
        job_dir=job_dir,
        parent_job_id=job_id,
        request=request,
        followup_plan=_mapping(deployment_validation.get("followup_plan")),
        generated_at=generated_at,
    )
    deployment_validation = {
        **dict(deployment_validation),
        "followup_request_queue": followup_request_queue,
    }
    deployment_calibration = _mapping(deployment_validation.get("calibration_report"))
    calibration_score = deployment_calibration.get("sim_vs_real_calibration_score")
    if calibration_score is not None:
        scorecard = dict(_mapping(eval_result.get("standard_policy_scorecard")))
        scorecard["sim_vs_real_calibration_score"] = calibration_score
        eval_result = {
            **dict(eval_result),
            "standard_policy_scorecard": scorecard,
            "deployment_outcome_intake_manifest_path": "deployment_outcome_intake_manifest.json",
            "deployment_outcome_ledger_path": "deployment_outcome_ledger.json",
            "sim_vs_real_calibration_report_path": "sim_vs_real_calibration_report.json",
        }
        _write_job_json(job_dir, "evaluation_result.json", eval_result)

    sc3_eval_protocol = build_sc3_eval_protocol_artifact(
        generated_at=generated_at,
        job_request=request,
        policy_package_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        robot_pov_observation_manifest=robot_pov_manifest,
        policy_ranking_scorecard=_read_optional_mapping(job_dir / "policy_ranking_scorecard.json"),
        prediction_outcome_correlation_ledger=_read_optional_mapping(
            job_dir / "wam_prediction_outcome_correlation_ledger.json"
        ),
        sim_vs_real_calibration_report=_mapping(deployment_validation.get("calibration_report")),
        wam_eval_claim_boundary=_read_optional_mapping(job_dir / "wam_eval_claim_boundary.json"),
        action_normalization_manifest=_read_optional_mapping(
            job_dir / "action_validation_manifest.json"
        ),
    )
    _write_job_json(job_dir, SC3_EVAL_PROTOCOL_ARTIFACT, sc3_eval_protocol)
    eval_result = {
        **dict(eval_result),
        "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
        "sc3_eval_protocol_status": sc3_eval_protocol.get("status"),
        "sc3_correlation_claim_status": sc3_eval_protocol.get("correlation_claim_status"),
        "claim_boundary": {
            **_mapping(eval_result.get("claim_boundary")),
            "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
            "sc3_self_consistency_is_reliability_support_only": True,
            "sc3_protocol_does_not_claim_blueprint_90_percent_accuracy": True,
        },
    }
    _write_job_json(job_dir, "evaluation_result.json", eval_result)

    proof_boundary = _proof_boundary(
        simulator=simulator,
        simulator_result=sim_result,
        training_result=training_res,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        deployment_outcome_ledger=_mapping(deployment_validation.get("ledger")),
        generated_at=generated_at,
    )
    if wam_eval_result:
        proof_boundary = {
            **dict(proof_boundary),
            "evaluation_substrate": selected_evaluation_substrate,
            "wam_eval_claim_boundary_path": "wam_eval_claim_boundary.json",
            "primary_proof_target": "policy_comparison_within_configured_evaluator",
            "policy_ranking_is_evaluator_bounded": True,
            "policy_ranking_is_not_evaluation_readiness": True,
            "traditional_sim_is_optional_cross_check_for_wam_eval": True,
            "generated_wam_rollouts_are_model_derived_support_artifacts": True,
            "customer_specific_srcc_claimed": False,
            "passing_wam_eval_is_not_rank_fidelity_result": True,
            "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
            "sc3_self_consistency_is_reliability_support_only": True,
            "claim_boundary": {
                **_mapping(proof_boundary.get("claim_boundary")),
                "wam_eval_claim_boundary_path": "wam_eval_claim_boundary.json",
                "primary_proof_target": "policy_comparison_within_configured_evaluator",
                "policy_ranking_is_evaluator_bounded": True,
                "policy_ranking_is_not_evaluation_readiness": True,
                "traditional_sim_is_optional_cross_check_for_wam_eval": True,
                "generated_wam_rollouts_are_model_derived_support_artifacts": True,
                "customer_specific_srcc_claimed": False,
                "passing_wam_eval_is_not_rank_fidelity_result": True,
                "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
                "sc3_self_consistency_is_reliability_support_only": True,
            },
        }
    elif simulator_policy_scorecard:
        proof_boundary = {
            **dict(proof_boundary),
            "evaluation_substrate": simulator_policy_scorecard.get("evaluation_substrate"),
            "wam_eval_claim_boundary_path": "wam_eval_claim_boundary.json",
            "primary_proof_target": "policy_comparison_within_configured_evaluator",
            "policy_ranking_is_evaluator_bounded": True,
            "policy_ranking_is_not_evaluation_readiness": True,
            "simulator_evaluator_only": True,
            "customer_specific_srcc_claimed": False,
            "passing_simulator_eval_is_not_rank_fidelity_result": True,
            "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
            "sc3_self_consistency_is_reliability_support_only": True,
            "claim_boundary": {
                **_mapping(proof_boundary.get("claim_boundary")),
                "wam_eval_claim_boundary_path": "wam_eval_claim_boundary.json",
                "primary_proof_target": "policy_comparison_within_configured_evaluator",
                "policy_ranking_is_evaluator_bounded": True,
                "policy_ranking_is_not_evaluation_readiness": True,
                "simulator_evaluator_only": True,
                "customer_specific_srcc_claimed": False,
                "passing_simulator_eval_is_not_rank_fidelity_result": True,
                "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
                "sc3_self_consistency_is_reliability_support_only": True,
            },
        }
    else:
        proof_boundary = {
            **dict(proof_boundary),
            "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
            "sc3_self_consistency_is_reliability_support_only": True,
            "claim_boundary": {
                **_mapping(proof_boundary.get("claim_boundary")),
                "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
                "sc3_self_consistency_is_reliability_support_only": True,
                "sc3_protocol_does_not_claim_blueprint_90_percent_accuracy": True,
            },
        }
    _write_job_json(job_dir, "proof_boundary.json", proof_boundary)
    blockers: List[str] = []
    missing_inputs: List[str] = []
    evidence: Dict[str, Any] = {
        "job_validation_status": validation.get("status"),
        "scheduler_decision_status": scheduler_decision.get("status"),
        "scheduler_decision_blockers": _string_list(scheduler_decision.get("blockers")),
        "worker_launch_plan_status": worker_launch_plan.get("status"),
        "worker_launch_plan_blockers": _string_list(worker_launch_plan.get("blockers")),
        "gpu_provider_launch_request_status": provider_launch_request.get("status"),
        "gpu_provider_launch_request_blockers": _string_list(
            provider_launch_request.get("blockers")
        ),
        "gpu_cost_control_ledger_status": gpu_cost_ledger.get("status"),
        "gpu_cost_control_ledger_blockers": _string_list(gpu_cost_ledger.get("blockers")),
        "gpu_provisioning_status": gpu_result.get("status"),
        "simulator_service_status": sim_result.get("status"),
        "evaluation_substrate": selected_evaluation_substrate or None,
        "wam_evaluation_status": wam_eval_result.get("status") if wam_eval_result else None,
        "robot_pov_status": robot_pov_manifest.get("status"),
        "robot_pov_evidence_proven": _strict_bool(
            robot_pov_manifest.get("robot_pov_evidence_proven")
        ),
        "policy_execution_status": _mapping(policy_execution.get("manifest")).get("status"),
        "sc3_eval_protocol_status": sc3_eval_protocol.get("status"),
        "sc3_correlation_claim_status": sc3_eval_protocol.get("correlation_claim_status"),
        "deployment_outcome_status": _mapping(deployment_validation.get("ledger")).get("status"),
        "real_world_validation_followup_plan_status": _mapping(
            deployment_validation.get("followup_plan")
        ).get("status"),
        "real_world_outcome_records_present": _strict_bool(
            _mapping(deployment_validation.get("ledger")).get("real_world_outcome_records_present")
        ),
        "owner_evidence_record_count": int(
            _mapping(deployment_validation.get("ledger")).get("owner_evidence_record_count") or 0
        ),
        "missing_owner_evidence_record_ids": _string_list(
            _mapping(deployment_validation.get("ledger")).get("missing_owner_evidence_record_ids")
        ),
        "sim_vs_real_calibration_status": _mapping(
            deployment_validation.get("calibration_report")
        ).get("status"),
        "live_eval_closure_status": "pending_live_eval_closure",
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
    if wam_eval_blockers and validation.get("status") != "blocked":
        blockers.append("wam_evaluation_blocked")
        missing_inputs.extend(wam_eval_blockers)
        evidence["wam_evaluation_blockers"] = wam_eval_blockers
    if scenario_eval_matrix.get("status") != "completed":
        blockers.append("scenario_eval_matrix_blocked")
        missing_inputs.extend(_string_list(scenario_eval_matrix.get("blockers")))
        evidence["scenario_eval_matrix_blockers"] = _string_list(
            scenario_eval_matrix.get("blockers")
        )
    if training_res.get("status") == "blocked":
        blockers.append("training_blocked")
        missing_inputs.extend(_string_list(training_res.get("blockers")))
    if eval_result.get("status") == "blocked" and validation.get("status") != "blocked":
        blockers.append("evaluation_blocked")
        missing_inputs.extend(_string_list(eval_result.get("blockers")))
        evidence["evaluation_blockers"] = _string_list(eval_result.get("blockers"))

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

    _write_robot_eval_report(
        job_dir=job_dir,
        job_id=job_id,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        job_status=status,
        blockers=blockers,
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
        policy_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        evaluation_result=eval_result,
        deployment_validation=deployment_validation,
        live_closure={"status": "pending_live_eval_closure", "blockers": []},
        proof_boundary=proof_boundary,
        generated_at=generated_at,
    )
    live_closure = build_live_robot_eval_closure_manifest(
        capture_root=context.capture_root,
        job_dir=job_dir,
        job_request=request,
        generated_at=generated_at,
    )
    proof_boundary = _apply_live_closure_to_proof_boundary(
        proof_boundary=proof_boundary,
        live_closure=live_closure,
    )
    _write_job_json(job_dir, "proof_boundary.json", proof_boundary)
    task_eval_run_report = _write_task_eval_run_buyer_report(
        job_dir=job_dir,
        job_id=job_id,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result=sim_result,
        copied_artifacts=copied_artifacts,
        robot_pov_manifest=robot_pov_manifest,
        policy_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        evaluation_result=eval_result,
        proof_boundary=proof_boundary,
        live_closure=live_closure,
        gpu_result=gpu_result,
        gpu_cost_ledger=gpu_cost_ledger,
        remote_cloud_closure=remote_cloud_closure,
        wam_eval_result=wam_eval_result,
        generated_at=generated_at,
    )
    evidence["live_eval_closure_status"] = live_closure.get("status")
    evidence["task_eval_run_report_status"] = task_eval_run_report.get("status")
    evidence["task_eval_run_report_evidence_level"] = task_eval_run_report.get(
        "evidence_level"
    )
    robot_eval_report = _write_robot_eval_report(
        job_dir=job_dir,
        job_id=job_id,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        job_status=status,
        blockers=blockers,
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
        policy_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        evaluation_result=eval_result,
        deployment_validation=deployment_validation,
        live_closure=live_closure,
        proof_boundary=proof_boundary,
        generated_at=generated_at,
    )
    webapp_status_projection = _webapp_robot_eval_status_projection(
        job_dir=job_dir,
        job_id=job_id,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        status=status,
        blockers=blockers,
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result=sim_result,
        copied_artifacts=copied_artifacts,
        robot_pov_manifest=robot_pov_manifest,
        policy_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        evaluation_result=eval_result,
        proof_boundary=proof_boundary,
        live_closure=live_closure,
        data_package_export={},
        generated_at=generated_at,
    )
    _write_job_json(
        job_dir,
        "webapp_robot_eval_status_projection.json",
        webapp_status_projection,
    )
    robot_team_grade_closure = _robot_team_grade_eval_closure_manifest(
        job_dir=job_dir,
        job_id=job_id,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        status=status,
        blockers=blockers,
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result=sim_result,
        copied_artifacts=copied_artifacts,
        robot_pov_manifest=robot_pov_manifest,
        policy_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        evaluation_result=eval_result,
        proof_boundary=proof_boundary,
        live_closure=live_closure,
        remote_cloud_closure=remote_cloud_closure,
        webapp_status_projection=webapp_status_projection,
        data_package_export={},
        generated_at=generated_at,
    )
    _write_job_json(
        job_dir,
        "robot_team_grade_eval_closure_manifest.json",
        robot_team_grade_closure,
    )
    run_canonical_training_quality_from_request(
        job_dir=job_dir,
        request=request,
    )
    data_package_export = build_post_training_data_package_export(
        capture_root=context.capture_root,
        job_dir=job_dir,
    )
    webapp_status_projection = _webapp_robot_eval_status_projection(
        job_dir=job_dir,
        job_id=job_id,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        status=status,
        blockers=blockers,
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result=sim_result,
        copied_artifacts=copied_artifacts,
        robot_pov_manifest=robot_pov_manifest,
        policy_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        evaluation_result=eval_result,
        proof_boundary=proof_boundary,
        live_closure=live_closure,
        data_package_export=data_package_export,
        generated_at=generated_at,
    )
    _write_job_json(
        job_dir,
        "webapp_robot_eval_status_projection.json",
        webapp_status_projection,
    )
    robot_team_grade_closure = _robot_team_grade_eval_closure_manifest(
        job_dir=job_dir,
        job_id=job_id,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        status=status,
        blockers=blockers,
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result=sim_result,
        copied_artifacts=copied_artifacts,
        robot_pov_manifest=robot_pov_manifest,
        policy_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        evaluation_result=eval_result,
        proof_boundary=proof_boundary,
        live_closure=live_closure,
        remote_cloud_closure=remote_cloud_closure,
        webapp_status_projection=webapp_status_projection,
        data_package_export=data_package_export,
        generated_at=generated_at,
    )
    _write_job_json(
        job_dir,
        "robot_team_grade_eval_closure_manifest.json",
        robot_team_grade_closure,
    )
    webapp_status_projection = _webapp_robot_eval_status_projection(
        job_dir=job_dir,
        job_id=job_id,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        status=status,
        blockers=blockers,
        request=request,
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result=sim_result,
        copied_artifacts=copied_artifacts,
        robot_pov_manifest=robot_pov_manifest,
        policy_manifest=policy_manifest,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        evaluation_result=eval_result,
        proof_boundary=proof_boundary,
        live_closure=live_closure,
        data_package_export=data_package_export,
        generated_at=generated_at,
    )
    _write_job_json(
        job_dir,
        "webapp_robot_eval_status_projection.json",
        webapp_status_projection,
    )

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
        "evaluation_substrate": selected_evaluation_substrate or None,
        "wam_evaluation_status": wam_eval_result.get("status") if wam_eval_result else None,
        "policy_ranking_scorecard_path": (
            "policy_ranking_scorecard.json"
            if (job_dir / "policy_ranking_scorecard.json").is_file()
            else None
        ),
        "agent_orchestration_status": agent_plan.get("status"),
        "agent_operator_mode": agent_plan.get("operator_mode"),
        "agent_operator_ledger": "agent_orchestration_plan.json",
        "scene_asset_preflight_status": scene_preflight.get("status"),
        "episode_spec_status": episode_specs.get("status"),
        "episode_count": episode_specs.get("episode_count"),
        "cpu_simulator_preflight_status": cpu_preflight.get("status"),
        "simulation_automation_status": simulation_automation.get("status"),
        "validation_status": validation.get("status"),
        "benchmark_protocol_status": benchmark_protocol_status.get("status"),
        "benchmark_protocol_status_path": (
            "benchmark_protocol/benchmark_protocol_status.json"
        ),
        "evaluation_run_status": evaluation_run_plan.get("status"),
        "evaluation_run_spec_digest": evaluation_run_plan.get("spec_digest"),
        "evaluation_run_spec_path": "evaluation_run_spec.json",
        "evaluation_run_plan_path": "evaluation_run_plan.json",
        "scheduler_decision_status": scheduler_decision.get("status"),
        "scheduler_decision_path": "scheduler_decision.json",
        "worker_launch_plan_status": worker_launch_plan.get("status"),
        "worker_launch_plan_path": "worker_launch_plan.json",
        "gpu_startup_pipeline_plan_status": gpu_startup_pipeline_plan.get("status"),
        "gpu_startup_pipeline_plan_path": "gpu_startup_pipeline_plan.json",
        "gpu_provider_launch_request_status": provider_launch_request.get("status"),
        "gpu_provider_launch_request_path": "gpu_provider_launch_request.json",
        "gpu_cost_control_ledger_status": gpu_cost_ledger.get("status"),
        "gpu_cost_control_ledger_path": "gpu_cost_control_ledger.json",
        "remote_cloud_execution_closure_status": remote_cloud_closure.get("status"),
        "remote_cloud_execution_closure_path": (
            "remote_cloud_execution_closure_manifest.json"
        ),
        "remote_cloud_execution_proven": _strict_bool(
            remote_cloud_closure.get("remote_cloud_execution_proven")
        ),
        "remote_cloud_clean_shutdown_proven": _strict_bool(
            remote_cloud_closure.get("clean_shutdown_proven")
        ),
        "robot_team_grade_eval_closure_status": robot_team_grade_closure.get("status"),
        "robot_team_grade_eval_closure_path": (
            "robot_team_grade_eval_closure_manifest.json"
        ),
        "robot_team_grade_evaluation_complete": _strict_bool(
            robot_team_grade_closure.get("robot_team_grade_evaluation_complete")
        ),
        "sim_only_beta_core_complete": _strict_bool(
            robot_team_grade_closure.get("sim_only_beta_core_complete")
        ),
        "sim_only_customer_handoff_complete": _strict_bool(
            robot_team_grade_closure.get("sim_only_customer_handoff_complete")
        ),
        "evaluation_readiness_complete": _strict_bool(
            robot_team_grade_closure.get("evaluation_readiness_complete")
        ),
        "gpu_provisioning_status": gpu_result.get("status"),
        "simulator_service_status": sim_result.get("status"),
        "scenario_eval_matrix_status": scenario_eval_matrix.get("status"),
        "scenario_eval_run_count": scenario_eval_matrix.get("scenario_eval_run_count"),
        "scenario_variation_names_covered": scenario_eval_matrix.get("variation_names_covered"),
        "robot_pov_observation_status": robot_pov_manifest.get("status"),
        "robot_pov_evidence_proven": _strict_bool(
            robot_pov_manifest.get("robot_pov_evidence_proven")
        ),
        "policy_execution_status": _mapping(policy_execution.get("manifest")).get("status"),
        "sc3_eval_protocol_status": sc3_eval_protocol.get("status"),
        "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
        "sc3_correlation_claim_status": sc3_eval_protocol.get("correlation_claim_status"),
        "arena_result_ingest_status": _mapping(arena_ingest.get("run_manifest")).get("status")
        if arena_ingest
        else None,
        "deployment_outcome_status": _mapping(deployment_validation.get("ledger")).get("status"),
        "real_world_validation_followup_plan_status": _mapping(
            deployment_validation.get("followup_plan")
        ).get("status"),
        "real_world_validation_followup_action_count": int(
            _mapping(_mapping(deployment_validation.get("followup_plan")).get("summary")).get(
                "action_count"
            )
            or 0
        ),
        "real_world_validation_followup_request_queue_status": _mapping(
            deployment_validation.get("followup_request_queue")
        ).get("status"),
        "real_world_validation_followup_request_count": int(
            _mapping(deployment_validation.get("followup_request_queue")).get(
                "queued_request_count"
            )
            or 0
        ),
        "sim_vs_real_calibration_status": _mapping(
            deployment_validation.get("calibration_report")
        ).get("status"),
        "live_eval_closure_status": live_closure.get("status"),
        "live_eval_closure_manifest_path": "live_eval_closure_manifest.json",
        "live_end_to_end_verified": _strict_bool(
            live_closure.get("live_end_to_end_verified")
        ),
        "training_status": training_res.get("status"),
        "evaluation_status": eval_result.get("status"),
        "task_eval_run_report_status": task_eval_run_report.get("status"),
        "task_eval_run_report_evidence_level": task_eval_run_report.get("evidence_level"),
        "task_eval_run_report_path": "task_eval_run_report.json",
        "robot_eval_report_status": robot_eval_report.get("status"),
        "robot_eval_report_path": "robot_eval_report.json",
        "post_training_data_package_export_status": data_package_export.get("status"),
        "webapp_robot_eval_status_projection_status": webapp_status_projection.get("status"),
        "webapp_robot_eval_buyer_display_state": webapp_status_projection.get(
            "buyer_display_state"
        ),
        "blockers": _dedupe(blockers),
        "missing_inputs": _dedupe(missing_inputs),
        "artifacts": {},
        "cpu_preflight_artifacts": {
            "scene_asset_inventory": "../simulation_automation/scene_asset_inventory.json",
            "scene_asset_dependency_audit": (
                "../simulation_automation/scene_asset_dependency_audit.json"
            ),
            "scene_asset_preflight": "../simulation_automation/scene_asset_preflight.json",
            "scene_asset_inspection": "../simulation_automation/scene_asset_inspection.json",
            "scene_frame_estimate": "../simulation_automation/scene_frame_estimate.json",
            "collider_proxy_plan": "../simulation_automation/collider_proxy_plan.json",
            "cpu_scene_proxy_manifest": "../simulation_automation/cpu_scene_proxy_manifest.json",
            "cpu_preflight_scorecard": "../simulation_automation/cpu_preflight_scorecard.json",
            "task_anchor_proposal_manifest": (
                "../simulation_automation/task_anchor_proposal_manifest.json"
            ),
            "episode_spec_manifest": "../simulation_automation/episode_spec_manifest.json",
            "episode_spec": "../simulation_automation/episode_spec.v1.json",
            "episode_specs": "../simulation_automation/episode_specs.json",
            "episode_setup_manifest": "../simulation_automation/episode_setup_manifest.json",
            "spawn_pose_validation_manifest": (
                "../simulation_automation/spawn_pose_validation_manifest.json"
            ),
            "cpu_simulator_preflight_manifest": (
                "../simulation_automation/cpu_simulator_preflight_manifest.json"
            ),
            "cpu_preflight_manifest": "../simulation_automation/cpu_preflight_manifest.json",
            "pre_gpu_readiness_summary": (
                "../simulation_automation/pre_gpu_readiness_summary.json"
            ),
            "arena_environment_packet": "../simulation_automation/arena_environment_packet.json",
            "gpu_handoff_packet": "../simulation_automation/gpu_handoff_packet.json",
            "gpu_owner_system_proof_schema": (
                "../simulation_automation/gpu_owner_system_proof_schema.json"
            ),
            "gpu_run_checklist": "../simulation_automation/gpu_run_checklist.md",
            "owner_gpu_simulator_execution_blocked_manifest": (
                "../simulation_automation/owner_gpu_simulator_execution_blocked_manifest.json"
            ),
        "post_training_data_package_export_manifest": (
            "post_training_data_package_export_manifest.json"
        ),
        "webapp_robot_eval_status_projection": "webapp_robot_eval_status_projection.json",
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "robot_pov_observation_manifest": "robot_pov_observation_manifest.json",
            "robot_camera_profile_registry": "robot_camera_profile_registry.json",
            "robot_camera_profile_launch_readiness": (
                "robot_camera_profile_launch_readiness.json"
            ),
            "owner_robot_camera_calibration_request": (
                "owner_robot_camera_calibration_request.json"
            ),
            "robot_pov_observation_candidate_set": "robot_pov_observation_candidate_set.json",
            "selected_initial_policy_observation": "selected_initial_policy_observation.json",
            "robot_pov_observations": "robot_pov_observations.jsonl",
            "robot_pov_frame_sequence_manifest": "robot_pov_frame_sequence_manifest.json",
            "robot_pov_render_storyboard": "robot_pov_render_storyboard.json",
            "policy_execution_manifest": "policy_execution_manifest.json",
            "policy_execution_trace": "policy_execution_trace.json",
            "policy_execution_trace_jsonl": "policy_execution_trace.jsonl",
            "action_validation_manifest": "action_validation_manifest.json",
            "sc3_eval_protocol": SC3_EVAL_PROTOCOL_ARTIFACT,
            "task_eval_run_report": "task_eval_run_report.json",
            "scheduler_decision": "scheduler_decision.json",
            "worker_launch_plan": "worker_launch_plan.json",
        "gpu_provider_launch_request": "gpu_provider_launch_request.json",
        "gpu_provider_race_handoff": "gpu_provider_race_handoff.json",
        "worker_manifest": "worker_manifest.json",
        "gpu_cost_control_ledger": "gpu_cost_control_ledger.json",
        "remote_cloud_execution_closure_manifest": (
            "remote_cloud_execution_closure_manifest.json"
        ),
        "robot_team_grade_eval_closure_manifest": (
            "robot_team_grade_eval_closure_manifest.json"
        ),
        "deployment_outcome_intake_manifest": "deployment_outcome_intake_manifest.json",
            "deployment_outcome_ledger": "deployment_outcome_ledger.json",
            "sim_vs_real_calibration_report": "sim_vs_real_calibration_report.json",
            "prediction_vs_actual_deployment_summary": (
                "prediction_vs_actual_deployment_summary.json"
            ),
            "real_world_validation_followup_plan": ("real_world_validation_followup_plan.json"),
            "real_world_validation_followup_request_queue": (
                "real_world_validation_followup_request_queue.json"
            ),
            "live_eval_closure_manifest": "live_eval_closure_manifest.json",
            "arena_eval_schedule": "arena_eval_schedule.json",
            "arena_result_ingest_ledger": "arena_result_ingest_ledger.json",
            "arena_eval_metrics": "arena_eval_metrics.json",
            "clips_manifest": "clips_manifest.json",
            "review_resolution_ledger": "review_resolution_ledger.json",
            "accepted_failure_labels": "accepted_failure_labels.json",
            "candidate_selection_report": "candidate_selection_report.json",
            "candidate_selection_report_markdown": "candidate_selection_report.md",
            "customer_handoff_report": "customer_handoff_report.json",
            "delivery_manifest": "delivery_manifest.json",
            "arena_rerun_plan": "arena_rerun_plan.json",
            "live_operator_ledger": "live_operator_ledger.json",
        },
        "live_provider_calls_performed": False,
        "remote_asset_downloads_performed": False,
        "local_cpu_preflight_smoke_ran": _strict_bool(
            _read_optional_mapping(
                pipeline_dir / "simulation_automation" / "cpu_simulator_preflight_manifest.json"
            ).get("local_cpu_smoke_ran")
        ),
        "simulators_run": _strict_bool(sim_result.get("simulators_run")),
        "gpu_training_run": _strict_bool(training_res.get("gpu_training_run")),
        "messages_sent": False,
        "payments_touched": False,
        "deployments_performed": False,
        "review_acceptance_proven": _strict_bool(
            proof_boundary.get("review_acceptance_proven")
        ),
        "rights_privacy_scope_proven": _strict_bool(
            proof_boundary.get("rights_privacy_scope_proven")
        ),
        "signed_delivery_access_proven": _strict_bool(
            proof_boundary.get("signed_delivery_access_proven")
        ),
        "customer_handoff_ready": _strict_bool(
            robot_team_grade_closure.get("sim_only_customer_handoff_complete")
        ),
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "safety_validation_proven": _strict_bool(
            proof_boundary.get("safety_validation_proven")
        ),
        "simulator_execution_proven": _strict_bool(
            proof_boundary.get("simulator_execution_proven")
        ),
        "robot_policy_execution_proven": _strict_bool(
            proof_boundary.get("robot_policy_execution_proven")
        ),
        "real_world_outcome_records_present": _strict_bool(
            proof_boundary.get("real_world_outcome_records_present")
        ),
        "owner_evidence_record_count": int(proof_boundary.get("owner_evidence_record_count") or 0),
        "missing_owner_evidence_record_ids": _string_list(
            proof_boundary.get("missing_owner_evidence_record_ids")
        ),
        "real_world_outcome_proven": _strict_bool(
            proof_boundary.get("real_world_outcome_proven")
        ),
        "physics_contact_validated": _strict_bool(
            proof_boundary.get("physics_contact_validated")
        ),
        "non_ranking_operational_claim_validated": _strict_bool(
            proof_boundary.get("non_ranking_operational_claim_validated")
        ),
        "rank_fidelity_result_proven": _strict_bool(
            proof_boundary.get("rank_fidelity_result_proven")
        ),
        "public_claim_upgrade_allowed": _strict_bool(
            proof_boundary.get("public_claim_upgrade_allowed")
        ),
        "live_eval_closure_blockers": _string_list(live_closure.get("blockers")),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "live_eval_closure_manifest_path": "live_eval_closure_manifest.json",
            "live_end_to_end_verified": _strict_bool(
                live_closure.get("live_end_to_end_verified")
            ),
            "review_acceptance_proven": _strict_bool(
                proof_boundary.get("review_acceptance_proven")
            ),
            "rights_privacy_scope_proven": _strict_bool(
                proof_boundary.get("rights_privacy_scope_proven")
            ),
            "signed_delivery_access_proven": _strict_bool(
                proof_boundary.get("signed_delivery_access_proven")
            ),
            "customer_handoff_ready": _strict_bool(
                robot_team_grade_closure.get("sim_only_customer_handoff_complete")
            ),
            "sc3_eval_protocol_path": SC3_EVAL_PROTOCOL_ARTIFACT,
            "sc3_self_consistency_is_reliability_support_only": True,
            "sc3_protocol_does_not_claim_blueprint_90_percent_accuracy": True,
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": _strict_bool(
                proof_boundary.get("safety_validation_proven")
            ),
            "simulator_execution_proven": _strict_bool(
                proof_boundary.get("simulator_execution_proven")
            ),
            "robot_policy_execution_proven": _strict_bool(
                proof_boundary.get("robot_policy_execution_proven")
            ),
            "real_world_outcome_records_present": _strict_bool(
                proof_boundary.get("real_world_outcome_records_present")
            ),
            "real_world_outcome_proven": _strict_bool(
                proof_boundary.get("real_world_outcome_proven")
            ),
            "physics_contact_validated": _strict_bool(
                proof_boundary.get("physics_contact_validated")
            ),
            "non_ranking_operational_claim_validated": _strict_bool(
                proof_boundary.get("non_ranking_operational_claim_validated")
            ),
            "rank_fidelity_result_proven": _strict_bool(
                proof_boundary.get("rank_fidelity_result_proven")
            ),
            "public_claim_upgrade_allowed": _strict_bool(
                proof_boundary.get("public_claim_upgrade_allowed")
            ),
        },
    }
    evaluator_qualification_request = request.get("evaluator_qualification_request")
    evaluator_qualification: Dict[str, Any] | None = None
    if evaluator_qualification_request is not None:
        evaluator_qualification = build_evaluator_qualification_workflow(
            _mapping(evaluator_qualification_request)
        )
        _write_job_json(
            job_dir,
            "evaluator_qualification_workflow.json",
            evaluator_qualification,
        )
    run_manifest["evaluator_qualification_status"] = (
        evaluator_qualification.get("status")
        if evaluator_qualification is not None
        else "not_requested"
    )
    run_manifest["evaluator_scientific_qualification_status"] = (
        evaluator_qualification.get("scientific_qualification_status")
        if evaluator_qualification is not None
        else "not_requested"
    )
    run_manifest["evaluator_qualification_path"] = (
        "evaluator_qualification_workflow.json"
        if evaluator_qualification is not None
        else None
    )
    run_manifest["evaluator_qualification_claim_boundary"] = {
        "optional_support_layer": True,
        "ordinary_task_eval_completion_requires_qualification": False,
        "public_launch_claim_requires_qualified_result": True,
    }
    run_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "job_id": job_id,
            "validation": validation,
            "scheduler_decision": scheduler_decision,
            "worker_launch_plan": worker_launch_plan,
            "provider_launch_request": provider_launch_request,
            "gpu_cost_control_ledger": gpu_cost_ledger,
            "gpu_result": gpu_result,
            "sim_result": sim_result,
            "robot_pov_manifest": robot_pov_manifest,
            "policy_execution": _mapping(policy_execution.get("manifest")),
            "deployment_validation": _mapping(deployment_validation.get("calibration_report")),
            "real_world_validation_followup_plan": _mapping(
                deployment_validation.get("followup_plan")
            ),
            "real_world_validation_followup_request_queue": _mapping(
                deployment_validation.get("followup_request_queue")
            ),
            "training_result": training_res,
            "evaluation_result": eval_result,
            "task_eval_run_report": task_eval_run_report,
            "benchmark_protocol_status": benchmark_protocol_status,
            "wam_eval_result": {
                "status": wam_eval_result.get("status"),
                "evaluation_substrate": wam_eval_result.get("evaluation_substrate"),
                "policy_ranking_scorecard": _mapping(
                    wam_eval_result.get("policy_ranking_scorecard")
                ),
            }
            if wam_eval_result
            else {},
            "evaluator_qualification": evaluator_qualification or {},
            "live_closure": live_closure,
            "execution_fingerprint": fingerprint_execution_artifacts(
                robot_pov_manifest,
                _mapping(policy_execution.get("manifest")),
                _mapping(deployment_validation.get("calibration_report")),
                _mapping(deployment_validation.get("followup_plan")),
                _mapping(deployment_validation.get("followup_request_queue")),
            ),
        }
    )
    run_manifest["request_fingerprint"] = request_fingerprint
    run_manifest["server_attempt_id"] = execution_claim["server_attempt_id"]
    run_manifest["job_claim_path"] = "job_claim.json"
    run_manifest["attempt_dir"] = os.path.relpath(
        str(execution_claim["attempt_dir"]),
        start=str(job_dir),
    ).replace("\\", "/")
    run_manifest["final_commit_required"] = True
    _write_job_json(job_dir, "job_run_manifest.json", run_manifest)
    from .robot_eval_startup_architecture_audit import (
        build_robot_eval_startup_architecture_audit,
    )

    startup_architecture_audit = build_robot_eval_startup_architecture_audit(
        job_dir=job_dir,
        output_path=job_dir / "startup_architecture_audit.json",
    )
    run_manifest["startup_architecture_audit_status"] = startup_architecture_audit.get("status")
    run_manifest["startup_architecture_audit_path"] = "startup_architecture_audit.json"
    run_manifest["startup_architecture_compliant"] = bool(
        startup_architecture_audit.get("architecture_compliant")
    )
    sim_only_provider_plan: Dict[str, Any] = {}
    if simulator == "mujoco" and provisioner in LIVE_GPU_PROVISIONERS:
        from .sim_only_provider_execution_planner import (
            build_sim_only_provider_execution_layer,
        )

        sim_only_provider_plan = build_sim_only_provider_execution_layer(
            capture_root=context.capture_root,
            job_id=job_id,
            job_dir=job_dir,
        )
        run_manifest["sim_only_provider_execution_plan_status"] = (
            sim_only_provider_plan.get("status")
        )
        run_manifest["sim_only_provider_execution_plan_path"] = (
            "sim_only_provider_execution_plan.json"
        )
        run_manifest["sim_only_provider_preflight_status"] = _mapping(
            sim_only_provider_plan.get("preflight")
        ).get("status")
        run_manifest["sim_only_provider_runtime_manifest_status"] = _mapping(
            sim_only_provider_plan.get("runtime_manifest")
        ).get("status")
        run_manifest["sim_only_provider_cost_ledger_status"] = _mapping(
            sim_only_provider_plan.get("cost_ledger")
        ).get("status")
        run_manifest["claim_boundary"] = {
            **_mapping(run_manifest.get("claim_boundary")),
            "simulator_beta_success_evaluated_by_sim_only_gate": True,
            "generated_world_rank_fidelity_claimed": False,
        }
    run_manifest["artifacts"] = _artifact_paths(job_dir)
    _write_job_json(job_dir, "job_run_manifest.json", run_manifest)

    commit = {
        "schema_version": "robot_eval_job_commit.v1",
        "generated_at": utc_now_iso(),
        "status": "committed",
        "job_id": job_id,
        "server_attempt_id": execution_claim["server_attempt_id"],
        "request_fingerprint": request_fingerprint,
        "job_run_manifest_sha256": _sha_file(job_dir / "job_run_manifest.json"),
        "job_claim_sha256": _sha_file(job_dir / "job_claim.json"),
        "artifacts": dict(run_manifest["artifacts"]),
        "claim_boundary": {
            "commit_marks_complete_artifact_set": True,
            "absence_of_commit_means_run_is_uncommitted": True,
            "job_namespace_reuse_allowed": False,
        },
    }
    _write_job_json(job_dir, "job_commit.json", commit)

    return {
        "schema_version": "robot_eval_job_result.v1",
        "job_id": job_id,
        "capture_root": str(context.capture_root),
        "job_dir": str(job_dir),
        "manifest_path": str((job_dir / "job_run_manifest.json").resolve()),
        "status": status,
        "request_fingerprint": request_fingerprint,
        "server_attempt_id": execution_claim["server_attempt_id"],
        "commit_path": str((job_dir / "job_commit.json").resolve()),
        "live_eval_closure_status": live_closure.get("status"),
        "benchmark_protocol_status": benchmark_protocol_status.get("status"),
        "webapp_benchmark_projection_path": (
            str((job_dir / "benchmark_protocol" / "webapp_benchmark_projection.json").resolve())
            if (job_dir / "benchmark_protocol" / "webapp_benchmark_projection.json").is_file()
            else None
        ),
        "live_end_to_end_verified": _strict_bool(
            live_closure.get("live_end_to_end_verified")
        ),
        "claim_boundary": dict(run_manifest["claim_boundary"]),
    }


def _job_id_from_request(path: Path, request: Mapping[str, Any]) -> str:
    raw = _string(
        request.get("job_id")
        or request.get("jobId")
        or _mapping(request.get("owner_system")).get("request_id")
        or path.stem
    )
    return strict_identifier(raw, field="job_id")


def _webapp_request_identity(request: Mapping[str, Any]) -> tuple[str, ...] | None:
    site_package = _mapping(request.get("site_package") or request.get("sitePackage"))
    source = _mapping(request.get("source"))
    selection = _mapping(source.get("selection_state") or source.get("selectionState"))
    source_kind = _string(
        request.get("source_kind") or source.get("source_kind") or selection.get("source_kind")
    )
    identity = (
        _string(request.get("buyer_request_id") or request.get("buyerRequestId")),
        _string(site_package.get("site_submission_id") or site_package.get("siteSubmissionId") or selection.get("site_submission_id")),
        _string(site_package.get("capture_job_id") or site_package.get("captureJobId") or selection.get("capture_job_id")),
        _string(site_package.get("capture_id") or site_package.get("captureId") or selection.get("capture_id")),
        _string(site_package.get("site_slug") or site_package.get("siteSlug") or selection.get("site_slug")),
        source_kind,
    )
    if not any(identity):
        return None
    if source_kind == "webapp_route_forwarding_proof":
        return (
            "webapp_route_forwarding_proof",
            _string(site_package.get("capture_root") or site_package.get("captureRoot")),
            identity[3],
            identity[4],
            identity[5],
        )
    return identity


def _site_slug_from_request(request: Mapping[str, Any]) -> str:
    site_package = _mapping(request.get("site_package") or request.get("sitePackage"))
    source = _mapping(request.get("source"))
    selection = _mapping(source.get("selection_state") or source.get("selectionState"))
    return _string(
        site_package.get("site_slug")
        or site_package.get("siteSlug")
        or site_package.get("site_id")
        or site_package.get("siteId")
        or selection.get("site_slug")
        or selection.get("siteSlug")
        or selection.get("site_id")
        or selection.get("siteId")
    )


def _is_webapp_synced_artifact_capture_root(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    return (
        normalized == "/synced-artifacts"
        or normalized.startswith("/synced-artifacts/")
        or normalized == "synced-artifacts"
        or normalized.startswith("synced-artifacts/")
    )


def _capture_root_by_site_overrides() -> Dict[str, str]:
    raw = _string(os.getenv(WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV))
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_env_{WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"invalid_env_{WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV}")
    return {
        _string(site_slug): _string(capture_root)
        for site_slug, capture_root in parsed.items()
        if _string(site_slug) and _string(capture_root)
    }


def _capture_root_override_for_request(request: Mapping[str, Any]) -> tuple[str, str] | None:
    site_slug = _site_slug_from_request(request)
    if site_slug:
        site_override = _capture_root_by_site_overrides().get(site_slug)
        if site_override:
            return site_override, WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV
    global_override = _string(os.getenv(WEBAPP_FORWARD_CAPTURE_ROOT_ENV))
    if global_override:
        return global_override, WEBAPP_FORWARD_CAPTURE_ROOT_ENV
    return None


def _request_capture_root(request: Dict[str, Any], default: Path) -> Path:
    site_package = dict(_mapping(request.get("site_package") or request.get("sitePackage")))
    value = _string(site_package.get("capture_root") or site_package.get("captureRoot"))
    override = _capture_root_override_for_request(request)
    if override is not None:
        override_value, override_env = override
        resolved = Path(override_value).expanduser().resolve()
    else:
        resolved = default.expanduser().resolve()
        override_env = "control_plane_default_capture_root"
        if value:
            if _is_webapp_synced_artifact_capture_root(value):
                raise ValueError(
                    "missing_pipeline_capture_root_override_for_webapp_synced_artifact"
                )
            caller_root = Path(value).expanduser().resolve()
            if caller_root != resolved:
                raise ValueError("request_capture_root_not_server_mapped")
    site_package["capture_root"] = str(resolved)
    if value:
        site_package["webapp_capture_root"] = value
    site_package["capture_root_override_source"] = (
        f"env:{override_env}"
        if override is not None
        else override_env
    )
    site_package["caller_capture_root_authoritative"] = False
    request["site_package"] = site_package
    owner_system = dict(_mapping(request.get("owner_system") or request.get("ownerSystem")))
    owner_system["pipeline_control_plane_capture_root"] = str(resolved)
    request["owner_system"] = owner_system
    return resolved


def _inbox_request_sort_key(path: Path) -> tuple[int, str]:
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return (mtime_ns, path.name)


def _robot_eval_queue_disposition(status: Any) -> str:
    normalized = _string(status).lower()
    if normalized in {
        "fixture_evaluation_completed",
        "fixture_evaluation_completed_with_failures",
        "simulator_command_completed",
        "completed",
        "completed_with_failures",
        "uploaded",
    }:
        return ROBOT_EVAL_QUEUE_TERMINAL_SUCCESS
    if normalized in {"permanent_invalid", "quarantined"}:
        return ROBOT_EVAL_QUEUE_PERMANENT_INVALID
    if normalized in {"fatal_infrastructure", "failed_infrastructure"}:
        return ROBOT_EVAL_QUEUE_FATAL_INFRASTRUCTURE
    return ROBOT_EVAL_QUEUE_RETRYABLE_BLOCKED


def _robot_eval_exit_code(result: Mapping[str, Any]) -> int:
    disposition = _string(result.get("queue_disposition")) or _robot_eval_queue_disposition(
        result.get("status")
    )
    if disposition == ROBOT_EVAL_QUEUE_TERMINAL_SUCCESS:
        return 0
    if disposition == ROBOT_EVAL_QUEUE_PERMANENT_INVALID:
        return ROBOT_EVAL_PERMANENT_INVALID_EXIT_CODE
    if disposition == ROBOT_EVAL_QUEUE_FATAL_INFRASTRUCTURE:
        return ROBOT_EVAL_FATAL_INFRASTRUCTURE_EXIT_CODE
    return ROBOT_EVAL_RETRYABLE_EXIT_CODE


def _exclusive_inbox_run(function: Any) -> Any:
    """Serialize inbox consumers across processes; a crash releases ``flock``."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        raw_inbox = kwargs.get("inbox_dir")
        if raw_inbox is None:
            raise ValueError("inbox_dir is required")
        inbox = Path(raw_inbox)
        ensure_dir(inbox)
        lock_path = inbox / ".robot_eval_inbox.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return wrapped


def _processed_request_marker_path(processed_dir: Path, request_path: Path, digest: str) -> Path:
    safe_name = _safe_inbox_request_name(request_path)
    return processed_dir / f"{safe_name}.{digest[:16]}.processed.json"


def _safe_inbox_request_name(request_path: Path) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-"
        for char in request_path.name
    ).strip(".-") or "request"


def _exception_record(exc: BaseException) -> Dict[str, str]:
    return {
        "error_class": type(exc).__name__,
        "error_message": str(exc),
    }


def _write_processed_request_marker(
    *,
    processed_dir: Path,
    request_path: Path,
    digest: str,
    status: str,
    job_id: str,
    generated_at: str,
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    ensure_dir(processed_dir)
    marker = {
        "schema_version": "robot_eval_job_request_processed_marker.v1",
        "generated_at": generated_at,
        "source_request_path": str(request_path),
        "source_request_sha256": digest,
        "status": status,
        "job_id": job_id,
        "reason": reason,
    }
    if extra:
        marker.update(dict(extra))
    marker_path = _processed_request_marker_path(processed_dir, request_path, digest)
    write_json(marker_path, marker)
    return {"path": str(marker_path), **marker}


def _write_retryable_request_attempt(
    *,
    attempts_dir: Path,
    request_path: Path,
    digest: str,
    generated_at: str,
    job_id: str,
    status: str,
    reason: str,
    blockers: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    ensure_dir(attempts_dir)
    safe_name = _safe_inbox_request_name(request_path)
    attempt_path = attempts_dir / f"{safe_name}.{digest[:16]}.attempt.json"
    previous = _read_optional_mapping(attempt_path)
    attempt_count = int(previous.get("attempt_count") or 0) + 1
    history = [
        dict(item)
        for item in previous.get("attempt_history") or []
        if isinstance(item, Mapping)
    ]
    attempt = {
        "attempt_number": attempt_count,
        "attempted_at": generated_at,
        "status": status,
        "reason": reason,
        "blockers": _dedupe(blockers),
    }
    if extra:
        attempt.update(dict(extra))
    ledger = {
        "schema_version": "robot_eval_job_request_attempt_ledger.v1",
        "revision": int(previous.get("revision") or 0) + 1,
        "generated_at": generated_at,
        "source_request_path": str(request_path),
        "source_request_sha256": digest,
        "job_id": job_id,
        "status": status,
        "queue_disposition": status,
        "reason": reason,
        "blockers": _dedupe(blockers),
        "attempt_count": attempt_count,
        "attempt_history": [*history, attempt],
        "processed_marker_written": False,
        "retry_expected": status
        in {
            ROBOT_EVAL_QUEUE_RETRYABLE_BLOCKED,
            ROBOT_EVAL_QUEUE_FATAL_INFRASTRUCTURE,
        },
    }
    write_json(attempt_path, ledger)
    return {"path": str(attempt_path), **ledger}


def _write_inbox_quarantine_record(
    *,
    request_path: Path,
    digest: str,
    generated_at: str,
    phase: str,
    reason: str,
    exc: BaseException,
    quarantine_dir: Path,
    dead_letter_dir: Path,
    processed_dir: Path,
    job_id: str | None = None,
    request_capture_root: Path | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    ensure_dir(quarantine_dir)
    ensure_dir(dead_letter_dir)
    safe_name = _safe_inbox_request_name(request_path)
    dead_letter_path = dead_letter_dir / f"{safe_name}.{digest[:16]}.json"
    copy_error: Dict[str, str] | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=dead_letter_path.parent,
            prefix=f".{dead_letter_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with request_path.open("rb") as source, os.fdopen(
                file_descriptor, "wb"
            ) as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_path, dead_letter_path)
            directory_fd = os.open(dead_letter_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    except Exception as copy_exc:  # noqa: BLE001 - dead-letter recording must not abort the batch
        copy_error = _exception_record(copy_exc)
    record = {
        "schema_version": "robot_eval_job_request_quarantine.v1",
        "generated_at": generated_at,
        "status": ROBOT_EVAL_QUEUE_PERMANENT_INVALID,
        "queue_disposition": ROBOT_EVAL_QUEUE_PERMANENT_INVALID,
        "phase": phase,
        "reason": reason,
        "source_request_path": str(request_path),
        "source_request_sha256": digest,
        "dead_letter_request_path": str(dead_letter_path) if copy_error is None else None,
        "dead_letter_copy_succeeded": copy_error is None,
        "job_id": job_id,
        "request_capture_root": str(request_capture_root) if request_capture_root else None,
        **_exception_record(exc),
        "copy_error": copy_error,
        "operator_action": (
            "Inspect the dead-letter request, fix or remove the source inbox file, "
            "and resubmit changed content before retrying."
        ),
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "quarantine_is_operational_failure_evidence_not_eval_result": True,
            "dead_letter_request_is_for_operator_triage_only": True,
        },
    }
    record_path = quarantine_dir / f"{safe_name}.{digest[:16]}.quarantine.json"
    write_json(record_path, record)
    marker = _write_processed_request_marker(
        processed_dir=processed_dir,
        request_path=request_path,
        digest=digest,
        status=ROBOT_EVAL_QUEUE_PERMANENT_INVALID,
        job_id=job_id or _safe_inbox_request_name(request_path),
        generated_at=generated_at,
        reason=reason,
        extra={
            "queue_disposition": ROBOT_EVAL_QUEUE_PERMANENT_INVALID,
            "phase": phase,
            "quarantine_record_path": str(record_path),
            "dead_letter_request_path": str(dead_letter_path) if copy_error is None else None,
            **_exception_record(exc),
        },
    )
    return {"path": str(record_path), **record}, marker


@_exclusive_inbox_run
def run_robot_eval_job_request_inbox(
    *,
    capture_root: str | Path,
    inbox_dir: str | Path,
    agent_adapter: RobotEvalJobAgentAdapter | None = None,
    provisioner: str = "fixture_local",
    simulator: str = "fixture",
    evaluation_substrate: str | None = None,
    allow_wam_provider: bool = False,
    wam_provider_command: str | None = None,
    wam_provider_commands: Mapping[str, str] | None = None,
    wam_artifact_output_uri: str | None = None,
    wam_provider_max_retries: int = 0,
    wam_provider_timeout_seconds: int | None = None,
    allow_gpu_provisioning: bool = False,
    allow_simulator_execution: bool = False,
    allowed_simulators: Sequence[str] = (),
    simulator_commands: Mapping[str, str] | None = None,
    allow_cpu_simulator_preflight: bool = False,
    cpu_preflight_backends: Sequence[str] = CPU_BACKENDS,
    cpu_preflight_smoke_steps: int = 10,
    allow_cpu_preflight_render: bool = False,
    allow_training: bool = False,
    training_command: str | None = None,
    allow_policy_execution: bool = False,
    policy_execution_commands: Mapping[str, str] | None = None,
    timeout_seconds: int = 120,
    budget_usd: float | None = None,
    arena_results_dir: str | Path | None = None,
    arena_scenario_count: int = 500,
    arena_shard_size: int = 50,
    arena_num_envs: int = 16,
    arena_retry_budget: int = 2,
    allow_rollout_vision_labeling: bool = False,
    vision_labeling_command: str | None = None,
    allow_delivery_upload: bool = False,
    delivery_command: str | None = None,
    arena_operator_mode: str = "none",
    allow_live_agents_sdk: bool = False,
    allow_live_codex_sdk: bool = False,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    inbox_path = Path(inbox_dir)
    queue_root = context.pipeline_root / "robot_eval_job_requests"
    ensure_dir(queue_root)
    generated_at = utc_now_iso()
    processed_dir = inbox_path / ".processed"
    quarantine_dir = inbox_path / ".quarantine"
    dead_letter_dir = inbox_path / ".dead_letter"
    attempts_dir = inbox_path / ".attempts"
    request_paths = sorted(
        path
        for path in inbox_path.glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )
    loaded_requests: List[Dict[str, Any]] = []
    skipped_processed_requests: List[Dict[str, Any]] = []
    quarantined_requests: List[Dict[str, Any]] = []
    quarantine_markers: List[Dict[str, Any]] = []
    retryable_requests: List[Dict[str, Any]] = []
    fatal_infrastructure_requests: List[Dict[str, Any]] = []
    for request_path in request_paths:
        try:
            request_digest = _sha_file(request_path)
        except Exception as exc:  # noqa: BLE001 - one unreadable inbox file must not abort the batch
            fallback_digest = _sha_payload(
                {
                    "source_request_path": str(request_path),
                    "phase": "hash",
                    "error_class": type(exc).__name__,
                }
            )
            attempt = _write_retryable_request_attempt(
                attempts_dir=attempts_dir,
                request_path=request_path,
                digest=fallback_digest,
                generated_at=generated_at,
                job_id=_safe_inbox_request_name(request_path),
                status=ROBOT_EVAL_QUEUE_FATAL_INFRASTRUCTURE,
                reason="request_file_unreadable",
                blockers=["request_file_unreadable"],
                extra={"phase": "hash", **_exception_record(exc)},
            )
            fatal_infrastructure_requests.append(attempt)
            continue
        marker_path = _processed_request_marker_path(
            processed_dir,
            request_path,
            request_digest,
        )
        if marker_path.is_file():
            skipped_processed_requests.append(
                {
                    "source_request_path": str(request_path),
                    "source_request_sha256": request_digest,
                    "processed_marker_path": str(marker_path),
                    "reason": "already_processed_same_content",
                }
            )
            continue
        try:
            request = _read_job_request(request_path)
            if _sha_file(request_path) != request_digest:
                raise InboxRequestChangedError("request_changed_during_snapshot")
            request.setdefault("schema_version", JOB_REQUEST_SCHEMA_VERSION)
            identity = _webapp_request_identity(request)
        except InboxRequestChangedError as exc:
            retryable_requests.append(
                _write_retryable_request_attempt(
                    attempts_dir=attempts_dir,
                    request_path=request_path,
                    digest=request_digest,
                    generated_at=generated_at,
                    job_id=_safe_inbox_request_name(request_path),
                    status=ROBOT_EVAL_QUEUE_RETRYABLE_BLOCKED,
                    reason="request_changed_during_snapshot",
                    blockers=["request_changed_during_snapshot"],
                    extra={"phase": "load", **_exception_record(exc)},
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001 - malformed request is quarantined per request
            quarantine, marker = _write_inbox_quarantine_record(
                request_path=request_path,
                digest=request_digest,
                generated_at=generated_at,
                phase="load",
                reason="request_json_or_contract_invalid",
                exc=exc,
                quarantine_dir=quarantine_dir,
                dead_letter_dir=dead_letter_dir,
                processed_dir=processed_dir,
            )
            quarantined_requests.append(quarantine)
            quarantine_markers.append(marker)
            continue
        loaded_requests.append(
            {
                "path": request_path,
                "request": request,
                "sha256": request_digest,
                "processed_marker_path": marker_path,
                "identity": identity,
                "sort_key": _inbox_request_sort_key(request_path),
            }
        )
    selected_requests: List[Dict[str, Any]] = []
    selected_by_identity: Dict[tuple[str, ...], Dict[str, Any]] = {}
    superseded_requests: List[Dict[str, Any]] = []
    for item in loaded_requests:
        identity = item.get("identity")
        if identity is None:
            selected_requests.append(item)
            continue
        previous = selected_by_identity.get(identity)
        if previous is None:
            selected_by_identity[identity] = item
            continue
        if item["sort_key"] >= previous["sort_key"]:
            superseded_requests.append(previous)
            selected_by_identity[identity] = item
        else:
            superseded_requests.append(item)
    selected_requests.extend(selected_by_identity.values())
    selected_requests = sorted(selected_requests, key=lambda item: str(item["path"]))
    jobs: List[Dict[str, Any]] = []
    processed_markers: List[Dict[str, Any]] = []
    for item in selected_requests:
        request_path = item["path"]
        request = dict(item["request"])
        job_id = _safe_inbox_request_name(request_path)
        request_capture_root = context.capture_root
        try:
            job_id = _job_id_from_request(request_path, request)
            request_capture_root = _request_capture_root(request, context.capture_root)
            request_context = (
                resolve_local_capture_context(request_capture_root)
                if request_capture_root != context.capture_root
                else context
            )
            request["job_id"] = job_id
            request["capture_root"] = str(request_context.capture_root)
            job_queue_root = request_context.pipeline_root / "robot_eval_job_requests"
            ensure_dir(job_queue_root)
            queued_dir = job_queue_root / job_id
            ensure_dir(queued_dir)
            write_json(queued_dir / "job_request.json", request)
            result = execute_robot_eval_request_as_evaluation_run(
                capture_root=request_context.capture_root,
                job_request=request,
                job_id=job_id,
                agent_adapter=agent_adapter,
                provisioner=provisioner,
                simulator=simulator,
                evaluation_substrate=evaluation_substrate,
                allow_wam_provider=allow_wam_provider,
                wam_provider_command=wam_provider_command,
                wam_provider_commands=wam_provider_commands or {},
                wam_artifact_output_uri=wam_artifact_output_uri,
                wam_provider_max_retries=wam_provider_max_retries,
                wam_provider_timeout_seconds=wam_provider_timeout_seconds,
                allow_gpu_provisioning=allow_gpu_provisioning,
                allow_simulator_execution=allow_simulator_execution,
                allowed_simulators=allowed_simulators,
                simulator_commands=simulator_commands or {},
                allow_cpu_simulator_preflight=allow_cpu_simulator_preflight,
                cpu_preflight_backends=cpu_preflight_backends,
                cpu_preflight_smoke_steps=cpu_preflight_smoke_steps,
                allow_cpu_preflight_render=allow_cpu_preflight_render,
                allow_training=allow_training,
                training_command=training_command,
                allow_policy_execution=allow_policy_execution,
                policy_execution_commands=policy_execution_commands or {},
                timeout_seconds=timeout_seconds,
                budget_usd=budget_usd,
                arena_results_dir=arena_results_dir,
                arena_scenario_count=arena_scenario_count,
                arena_shard_size=arena_shard_size,
                arena_num_envs=arena_num_envs,
                arena_retry_budget=arena_retry_budget,
                allow_rollout_vision_labeling=allow_rollout_vision_labeling,
                vision_labeling_command=vision_labeling_command,
                allow_delivery_upload=allow_delivery_upload,
                delivery_command=delivery_command,
                arena_operator_mode=arena_operator_mode,
                allow_live_agents_sdk=allow_live_agents_sdk,
                allow_live_codex_sdk=allow_live_codex_sdk,
            )
        except Exception as exc:  # noqa: BLE001 - persist retry evidence and continue the batch
            attempt = _write_retryable_request_attempt(
                attempts_dir=attempts_dir,
                request_path=request_path,
                digest=str(item["sha256"]),
                generated_at=generated_at,
                job_id=job_id,
                status=ROBOT_EVAL_QUEUE_FATAL_INFRASTRUCTURE,
                reason="robot_eval_job_request_processing_failed",
                blockers=["robot_eval_job_request_processing_failed"],
                extra={
                    "phase": "process",
                    "request_capture_root": str(request_capture_root),
                    **_exception_record(exc),
                },
            )
            fatal_infrastructure_requests.append(attempt)
            continue
        disposition = _robot_eval_queue_disposition(result.get("status"))
        run_manifest = _read_optional_mapping(Path(str(result["manifest_path"])))
        retry_blockers = _string_list(run_manifest.get("blockers"))
        if disposition != ROBOT_EVAL_QUEUE_TERMINAL_SUCCESS and not retry_blockers:
            retry_blockers = [f"robot_eval_job_status:{_string(result.get('status')) or 'unknown'}"]
        jobs.append(
            {
                "job_id": job_id,
                "status": result["status"],
                "queue_disposition": disposition,
                "retry_blockers": retry_blockers,
                "source_request_path": str(request_path),
                "queued_request_path": str((queued_dir / "job_request.json").resolve()),
                "request_capture_root": str(request_context.capture_root),
                "job_dir": result["job_dir"],
                "job_run_manifest_uri": result["manifest_path"],
                "public_claim_upgrade_allowed": False,
            }
        )
        if disposition == ROBOT_EVAL_QUEUE_TERMINAL_SUCCESS:
            processed_markers.append(
                _write_processed_request_marker(
                    processed_dir=processed_dir,
                    request_path=request_path,
                    digest=str(item["sha256"]),
                    status="processed",
                    job_id=job_id,
                    generated_at=generated_at,
                    reason="terminal_robot_eval_job_result",
                    extra={"queue_disposition": disposition},
                )
            )
        else:
            retryable_requests.append(
                _write_retryable_request_attempt(
                    attempts_dir=attempts_dir,
                    request_path=request_path,
                    digest=str(item["sha256"]),
                    generated_at=generated_at,
                    job_id=job_id,
                    status=ROBOT_EVAL_QUEUE_RETRYABLE_BLOCKED,
                    reason="robot_eval_job_returned_retryable_blocked",
                    blockers=retry_blockers,
                    extra={
                        "job_dir": result.get("job_dir"),
                        "job_run_manifest_uri": result.get("manifest_path"),
                        "job_status": result.get("status"),
                    },
                )
            )
    for item in superseded_requests:
        processed_markers.append(
            _write_processed_request_marker(
                processed_dir=processed_dir,
                request_path=item["path"],
                digest=str(item["sha256"]),
                status="superseded",
                job_id=_job_id_from_request(item["path"], item["request"]),
                generated_at=generated_at,
                reason="superseded_by_newer_webapp_request_for_same_identity",
            )
        )
    processed_markers.extend(quarantine_markers)
    terminal_jobs = [
        job
        for job in jobs
        if job.get("queue_disposition") == ROBOT_EVAL_QUEUE_TERMINAL_SUCCESS
    ]
    if fatal_infrastructure_requests:
        status = ROBOT_EVAL_QUEUE_FATAL_INFRASTRUCTURE
        queue_disposition = ROBOT_EVAL_QUEUE_FATAL_INFRASTRUCTURE
    elif retryable_requests:
        status = ROBOT_EVAL_QUEUE_RETRYABLE_BLOCKED
        queue_disposition = ROBOT_EVAL_QUEUE_RETRYABLE_BLOCKED
    elif terminal_jobs and quarantined_requests:
        status = "completed_with_permanent_invalid_requests"
        queue_disposition = ROBOT_EVAL_QUEUE_PERMANENT_INVALID
    elif terminal_jobs:
        status = "completed"
        queue_disposition = ROBOT_EVAL_QUEUE_TERMINAL_SUCCESS
    elif quarantined_requests:
        status = ROBOT_EVAL_QUEUE_PERMANENT_INVALID
        queue_disposition = ROBOT_EVAL_QUEUE_PERMANENT_INVALID
    else:
        status = "empty"
        queue_disposition = ROBOT_EVAL_QUEUE_TERMINAL_SUCCESS
    existing_manifest = _read_optional_mapping(queue_root / "inbox_run_manifest.json")
    manifest = {
        "schema_version": JOB_REQUEST_INBOX_RUN_SCHEMA_VERSION,
        "revision": int(existing_manifest.get("revision") or 0) + 1,
        "generated_at": generated_at,
        "status": status,
        "queue_disposition": queue_disposition,
        "capture_root": str(context.capture_root),
        "inbox_dir": str(inbox_path),
        "queue_root": str(queue_root),
        "discovered_request_count": len(request_paths),
        "input_request_count": len(loaded_requests),
        "skipped_processed_request_count": len(skipped_processed_requests),
        "skipped_processed_requests": skipped_processed_requests,
        "job_attempt_count": len(jobs),
        "processed_count": len(terminal_jobs),
        "terminal_success_count": len(terminal_jobs),
        "retryable_request_count": len(retryable_requests),
        "retryable_requests": retryable_requests,
        "fatal_infrastructure_request_count": len(fatal_infrastructure_requests),
        "fatal_infrastructure_requests": fatal_infrastructure_requests,
        "attempt_ledger_dir": str(attempts_dir),
        "quarantined_request_count": len(quarantined_requests),
        "quarantined_requests": quarantined_requests,
        "quarantine_dir": str(quarantine_dir),
        "dead_letter_dir": str(dead_letter_dir),
        "superseded_request_count": len(superseded_requests),
        "superseded_requests": [
            {
                "source_request_path": str(item["path"]),
                "job_id": _job_id_from_request(item["path"], item["request"]),
                "identity": list(item["identity"]) if item.get("identity") else None,
                "reason": "superseded_by_newer_webapp_request_for_same_identity",
            }
            for item in sorted(superseded_requests, key=lambda entry: str(entry["path"]))
        ],
        "processed_marker_dir": str(processed_dir),
        "processed_markers": processed_markers,
        "jobs": jobs,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(queue_root / "inbox_run_manifest.json", manifest)
    return manifest


def _parse_simulator_commands(values: Sequence[str] | None) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for value in values or []:
        framework, sep, command = value.partition("=")
        if not sep or framework not in SIMULATORS or framework == "fixture" or not command.strip():
            raise ValueError(
                "--simulator-command must be formatted as "
                "<mujoco|pybullet|newton|isaac_sim|isaac_lab_arena>=<command>"
            )
        commands[framework] = command.strip()
    return commands


def _parse_policy_execution_commands(values: Sequence[str] | None) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for value in values or []:
        modality, sep, command = value.partition("=")
        if not sep or modality not in POLICY_MODALITY_ORDER or not command.strip():
            raise ValueError(
                "--policy-execution-command must be formatted as "
                "<policy_api_endpoint|docker_container|recorded_action_trace|"
                "high_level_skill_trace|teleop_demo|sim_controller_plugin>=<command>"
            )
        commands[modality] = command.strip()
    return commands


def _agent_adapter_from_mode(
    mode: str, *, allow_live_operator: bool
) -> RobotEvalJobAgentAdapter | None:
    if mode == "fake":
        return FakeRobotEvalJobAgentAdapter()
    if mode == "agents-sdk":
        return AgentsSdkRobotEvalJobAdapter(allow_live_operator=allow_live_operator)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fail-closed headless robot-eval job from a local request manifest"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--job-request", default=None, help="Robot eval job request JSON")
    parser.add_argument("--job-id", default=None, help="Deterministic job id")
    parser.add_argument(
        "--evaluation-run-spec",
        default=None,
        help="Authoritative evaluation_run.v1 JSON; mutually exclusive with legacy requests",
    )
    parser.add_argument("--evaluation-run-output-dir", default=None)
    parser.add_argument(
        "--job-request-inbox",
        default=None,
        help="Directory of robot_eval_job_request.v1 JSON files to run automatically",
    )
    parser.add_argument(
        "--agent-mode",
        choices=("none", "fake", "agents-sdk"),
        default="none",
        help="Optional agent operator adapter; deterministic manifests remain authoritative",
    )
    parser.add_argument(
        "--allow-live-agent-operator",
        action="store_true",
        help=f"Allow live Agents SDK execution when {LIVE_AGENTS_SDK_ENV}=true and credentials exist",
    )
    parser.add_argument("--provisioner", choices=PROVISIONERS, default="fixture_local")
    parser.add_argument("--simulator", choices=SIMULATORS, default="fixture")
    parser.add_argument(
        "--evaluation-substrate",
        default=None,
        help=(
            "Optional evaluation substrate such as fixture_wam, cosmos3_wam, "
            "oscar_wam, classical_sim_mujoco, classical_sim_isaac, or recorded_trace. "
            "Legacy simulator aliases are accepted."
        ),
    )
    parser.add_argument(
        "--allow-wam-provider",
        action="store_true",
        help=(
            "Permit WAM provider adapter execution only with matching env approval "
            "and provider auth envs."
        ),
    )
    parser.add_argument(
        "--wam-provider-command",
        action="append",
        default=[],
        help="Explicit WAM provider adapter command as <cosmos3_wam|oscar_wam>=<command>",
    )
    parser.add_argument(
        "--wam-artifact-output-uri",
        default=None,
        help="Optional provider-writable artifact destination passed to the WAM adapter.",
    )
    parser.add_argument("--wam-provider-max-retries", type=int, default=0)
    parser.add_argument("--wam-provider-timeout-seconds", type=int, default=None)
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
        "--allow-cpu-simulator-preflight",
        action="store_true",
        help="Permit optional CPU MuJoCo/PyBullet smoke only with matching env approval",
    )
    parser.add_argument(
        "--cpu-preflight-backend",
        action="append",
        choices=CPU_BACKENDS,
        default=[],
        help="CPU preflight backend to include; repeatable. Defaults to MuJoCo and PyBullet.",
    )
    parser.add_argument("--cpu-preflight-smoke-steps", type=int, default=10)
    parser.add_argument(
        "--allow-cpu-preflight-render",
        action="store_true",
        help="Allow optional TinyRenderer path for PyBullet CPU preflight.",
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
    parser.add_argument(
        "--allow-policy-execution",
        action="store_true",
        help="Permit gated policy API/container/command execution with matching env approval",
    )
    parser.add_argument(
        "--policy-execution-command",
        action="append",
        default=[],
        help="Explicit policy adapter command as <modality>=<command>",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument(
        "--arena-results-dir",
        default=None,
        help="Existing local Isaac Lab-Arena rollout result directory to ingest",
    )
    parser.add_argument("--arena-scenario-count", type=int, default=500)
    parser.add_argument("--arena-shard-size", type=int, default=50)
    parser.add_argument("--arena-num-envs", type=int, default=16)
    parser.add_argument("--arena-retry-budget", type=int, default=2)
    parser.add_argument(
        "--allow-rollout-vision-labeling",
        action="store_true",
        help="Allow gated rollout vision labeling command with matching env approval",
    )
    parser.add_argument("--vision-labeling-command", default=None)
    parser.add_argument(
        "--allow-delivery-upload",
        action="store_true",
        help="Allow gated package upload/signed-access command with matching env approval",
    )
    parser.add_argument("--delivery-command", default=None)
    parser.add_argument(
        "--arena-operator-mode",
        choices=("none", "fake", "agents-sdk"),
        default="none",
        help="Arena package operator mode. Fake is local-only and gated by env.",
    )
    parser.add_argument("--allow-live-agents-sdk", action="store_true")
    parser.add_argument("--allow-live-codex-sdk", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = PipelineSettings.from_env()
        settings.validate_cli_admission(
            allow_gpu_provisioning=bool(args.allow_gpu_provisioning),
            allow_simulator_execution=bool(args.allow_simulator_execution),
            allow_cosmos_training=bool(args.allow_training),
            allow_live_agents_sdk_operator=bool(args.allow_live_agent_operator),
        )
        simulator_commands = _parse_simulator_commands(args.simulator_command)
        policy_execution_commands = _parse_policy_execution_commands(args.policy_execution_command)
        wam_provider_commands = parse_wam_provider_commands(args.wam_provider_command)
        if args.evaluation_run_spec:
            if args.job_request or args.job_id or args.job_request_inbox:
                raise ValueError(
                    "--evaluation-run-spec is mutually exclusive with legacy request inputs"
                )
            execution = execute_robot_eval_cli_evaluation_run(
                args,
                agent_adapter=_agent_adapter_from_mode(
                    args.agent_mode,
                    allow_live_operator=args.allow_live_agent_operator,
                ),
                simulator_commands=simulator_commands,
                policy_execution_commands=policy_execution_commands,
                wam_provider_commands=wam_provider_commands,
            )
            result = dict(execution.adapter_result or execution.manifest)
            print(f"[robot-eval-job] evaluation_run={execution.manifest['spec_digest']}")
            print(f"[robot-eval-job] status={result['status']}")
            return _robot_eval_exit_code(result)
        if args.job_request_inbox:
            result = run_robot_eval_job_request_inbox(
                capture_root=args.capture_root,
                inbox_dir=args.job_request_inbox,
                agent_adapter=_agent_adapter_from_mode(
                    args.agent_mode,
                    allow_live_operator=args.allow_live_agent_operator,
                ),
                provisioner=args.provisioner,
                simulator=args.simulator,
                evaluation_substrate=args.evaluation_substrate,
                allow_wam_provider=args.allow_wam_provider,
                wam_provider_commands=wam_provider_commands,
                wam_artifact_output_uri=args.wam_artifact_output_uri,
                wam_provider_max_retries=args.wam_provider_max_retries,
                wam_provider_timeout_seconds=args.wam_provider_timeout_seconds,
                allow_gpu_provisioning=args.allow_gpu_provisioning,
                allow_simulator_execution=args.allow_simulator_execution,
                allowed_simulators=args.allow_simulator,
                simulator_commands=simulator_commands,
                allow_cpu_simulator_preflight=args.allow_cpu_simulator_preflight,
                cpu_preflight_backends=args.cpu_preflight_backend or CPU_BACKENDS,
                cpu_preflight_smoke_steps=args.cpu_preflight_smoke_steps,
                allow_cpu_preflight_render=args.allow_cpu_preflight_render,
                allow_training=args.allow_training,
                training_command=args.training_command,
                allow_policy_execution=args.allow_policy_execution,
                policy_execution_commands=policy_execution_commands,
                timeout_seconds=args.timeout_seconds,
                budget_usd=args.budget_usd,
                arena_results_dir=args.arena_results_dir,
                arena_scenario_count=args.arena_scenario_count,
                arena_shard_size=args.arena_shard_size,
                arena_num_envs=args.arena_num_envs,
                arena_retry_budget=args.arena_retry_budget,
                allow_rollout_vision_labeling=args.allow_rollout_vision_labeling,
                vision_labeling_command=args.vision_labeling_command,
                allow_delivery_upload=args.allow_delivery_upload,
                delivery_command=args.delivery_command,
                arena_operator_mode=args.arena_operator_mode,
                allow_live_agents_sdk=args.allow_live_agents_sdk,
                allow_live_codex_sdk=args.allow_live_codex_sdk,
            )
            print(
                "[robot-eval-job] inbox_manifest="
                f"{Path(args.capture_root) / 'pipeline' / 'robot_eval_job_requests' / 'inbox_run_manifest.json'}"
            )
            print(f"[robot-eval-job] status={result['status']}")
            print(f"[robot-eval-job] processed_count={result['processed_count']}")
            return _robot_eval_exit_code(result)
        if not args.job_request or not args.job_id:
            raise ValueError(
                "--job-request and --job-id are required unless --job-request-inbox is provided"
            )
        result = execute_robot_eval_request_as_evaluation_run(
            capture_root=args.capture_root,
            job_request=args.job_request,
            job_id=args.job_id,
            agent_adapter=_agent_adapter_from_mode(
                args.agent_mode,
                allow_live_operator=args.allow_live_agent_operator,
            ),
            provisioner=args.provisioner,
            simulator=args.simulator,
            evaluation_substrate=args.evaluation_substrate,
            allow_wam_provider=args.allow_wam_provider,
            wam_provider_commands=wam_provider_commands,
            wam_artifact_output_uri=args.wam_artifact_output_uri,
            wam_provider_max_retries=args.wam_provider_max_retries,
            wam_provider_timeout_seconds=args.wam_provider_timeout_seconds,
            allow_gpu_provisioning=args.allow_gpu_provisioning,
            allow_simulator_execution=args.allow_simulator_execution,
            allowed_simulators=args.allow_simulator,
            simulator_commands=simulator_commands,
            allow_cpu_simulator_preflight=args.allow_cpu_simulator_preflight,
            cpu_preflight_backends=args.cpu_preflight_backend or CPU_BACKENDS,
            cpu_preflight_smoke_steps=args.cpu_preflight_smoke_steps,
            allow_cpu_preflight_render=args.allow_cpu_preflight_render,
            allow_training=args.allow_training,
            training_command=args.training_command,
            allow_policy_execution=args.allow_policy_execution,
            policy_execution_commands=policy_execution_commands,
            timeout_seconds=args.timeout_seconds,
            budget_usd=args.budget_usd,
            arena_results_dir=args.arena_results_dir,
            arena_scenario_count=args.arena_scenario_count,
            arena_shard_size=args.arena_shard_size,
            arena_num_envs=args.arena_num_envs,
            arena_retry_budget=args.arena_retry_budget,
            allow_rollout_vision_labeling=args.allow_rollout_vision_labeling,
            vision_labeling_command=args.vision_labeling_command,
            allow_delivery_upload=args.allow_delivery_upload,
            delivery_command=args.delivery_command,
            arena_operator_mode=args.arena_operator_mode,
            allow_live_agents_sdk=args.allow_live_agents_sdk,
            allow_live_codex_sdk=args.allow_live_codex_sdk,
        )
    except OSError as exc:
        print(f"[robot-eval-job] FAILED: {exc}")
        return ROBOT_EVAL_FATAL_INFRASTRUCTURE_EXIT_CODE
    except ValueError as exc:
        print(f"[robot-eval-job] FAILED: {exc}")
        return ROBOT_EVAL_PERMANENT_INVALID_EXIT_CODE
    except Exception as exc:  # noqa: BLE001 - CLI must map infrastructure failure explicitly
        print(
            "[robot-eval-job] FATAL_INFRASTRUCTURE: "
            f"{type(exc).__name__}: {exc}"
        )
        return ROBOT_EVAL_FATAL_INFRASTRUCTURE_EXIT_CODE
    print(f"[robot-eval-job] manifest={result['manifest_path']}")
    print(f"[robot-eval-job] status={result['status']}")
    return _robot_eval_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
