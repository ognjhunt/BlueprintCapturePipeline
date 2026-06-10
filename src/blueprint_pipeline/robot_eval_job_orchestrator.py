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
import importlib.util
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

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
from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .cpu_simulator_preflight import CPU_BACKENDS, build_cpu_simulator_preflight
from .episode_spec import build_episode_specs
from .local_capture import resolve_local_capture_context
from .live_robot_eval_closure import build_live_robot_eval_closure_manifest
from .post_training_data_package import build_post_training_data_package_export
from .robot_eval_execution import (
    build_scenario_eval_matrix,
    build_deployment_validation_bundle,
    build_policy_execution_bundle,
    build_robot_pov_observation_bundle,
    build_simulator_command_artifacts,
    fingerprint_execution_artifacts,
)
from .robot_eval_dataset import build_real_site_robot_eval_dataset
from .scene_asset_preflight import build_scene_asset_preflight
from .simulation_automation import build_simulation_automation
from .site_eval_director import build_site_eval_director


JOB_REQUEST_SCHEMA_VERSION = "robot_eval_job_request.v1"
JOB_VALIDATION_SCHEMA_VERSION = "robot_eval_job_validation.v1"
JOB_PLAN_SCHEMA_VERSION = "robot_eval_job_plan.v1"
AGENT_ORCHESTRATION_PLAN_SCHEMA_VERSION = "robot_eval_agent_orchestration_plan.v1"
GPU_PROVISIONING_REQUEST_SCHEMA_VERSION = "robot_eval_gpu_provisioning_request.v1"
GPU_PROVISIONING_RESULT_SCHEMA_VERSION = "robot_eval_gpu_provisioning_result.v1"
SIMULATOR_SERVICE_REQUEST_SCHEMA_VERSION = "robot_eval_simulator_service_request.v1"
SIMULATOR_SERVICE_RESULT_SCHEMA_VERSION = "robot_eval_simulator_service_result.v1"
SIMULATOR_PROVIDER_ADAPTER_SCHEMA_VERSION = "robot_eval_simulator_provider_adapter_manifest.v1"
POLICY_PACKAGE_MANIFEST_SCHEMA_VERSION = "robot_eval_policy_package_manifest.v1"
TRAINING_REQUEST_SCHEMA_VERSION = "robot_eval_training_request.v1"
TRAINING_RESULT_SCHEMA_VERSION = "robot_eval_training_result.v1"
EVALUATION_REQUEST_SCHEMA_VERSION = "robot_eval_evaluation_request.v1"
EVALUATION_RESULT_SCHEMA_VERSION = "robot_eval_evaluation_result.v1"
ROBOT_EVAL_REPORT_SCHEMA_VERSION = "robot_eval_job_report.v1"
NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION = "robot_eval_job_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "robot_eval_job_failure_labels.v1"
PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION = "robot_eval_job_prediction_outcome_ledger.v1"
CALIBRATION_REPORT_SCHEMA_VERSION = "robot_eval_job_calibration_report.v1"
BREAKAGE_LIBRARY_SCHEMA_VERSION = "robot_eval_job_breakage_library.v1"
PROOF_BOUNDARY_SCHEMA_VERSION = "robot_eval_job_proof_boundary.v1"
JOB_RUN_MANIFEST_SCHEMA_VERSION = "robot_eval_job_run_manifest.v1"
BLOCKED_MANIFEST_SCHEMA_VERSION = "robot_eval_job_blocked_manifest.v1"
JOB_REQUEST_INBOX_RUN_SCHEMA_VERSION = "robot_eval_job_request_inbox_run.v1"
REAL_WORLD_VALIDATION_FOLLOWUP_REQUEST_QUEUE_SCHEMA_VERSION = (
    "real_world_validation_followup_request_queue.v1"
)

PROVISIONERS = (
    "fixture_local",
    "local_process",
    "docker_local",
    "vast",
    "runpod",
    "gcp",
)
SIMULATORS = ("fixture", "mujoco", "pybullet", "newton", "isaac_sim", "isaac_lab_arena")
OPERATIONS = ("evaluate_only", "train_only", "train_then_evaluate")

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
    "agent_operator_mode_allowed": True,
    "agents_may_mutate_proof_booleans": False,
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
        payload = job_request
    else:
        payload = read_json_any(Path(job_request))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected job request JSON object at {job_request}")
    if (
        payload.get("queue_contract") == "robot_eval_job_request_inbox.v1"
        and isinstance(payload.get("job_request"), Mapping)
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
        key: value
        for key, value in dict(request).items()
        if key not in ACTUAL_OUTCOME_REQUEST_KEYS
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
                    "claim_boundary": "followup_request_is_rerun_input_not_robot_readiness_proof",
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
                "robot_readiness_proven": False,
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
    if policy_package:
        return policy_package
    direct = {
        modality: _modality_payload(payload, modality)
        for modality in POLICY_MODALITY_ORDER
        if _modality_payload(payload, modality)
    }
    return direct


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
        "proof_boundary": dict(CLAIM_BOUNDARY),
    }


def _policy_package_manifest(
    *,
    request: Mapping[str, Any],
    generated_at: str,
) -> tuple[Dict[str, Any], List[str], List[str]]:
    policy_package = _mapping(request.get("policy_package") or request.get("policyPackage"))
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
            "missing_evidence_status": (
                POLICY_MODALITY_STATUSES[modality] if missing else None
            ),
            "reference": dict(payload),
            "download_performed": False,
            "owner_system_review_required": status not in {"blocked", "not_selected"},
            "adapter_smoke_contract": _policy_adapter_smoke_contract(modality),
            "claim_boundary": (
                "reference_present_only_not_policy_execution_or_robot_readiness_proof"
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
        "variation_names_covered": list(
            scenario_eval_matrix.get("variation_names_covered") or []
        ),
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
        "robot_readiness_proven": False,
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
        "artifact_paths": {
            "simulator_output": str(output_path) if output_path and output_path.is_file() else None
        },
        "simulator_output_path": str(output_path) if output_path else None,
        "simulator_output_payload": simulator_output_payload,
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


def _scenario_eval_matrix_runs(scenario_eval_matrix: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        dict(run)
        for run in scenario_eval_matrix.get("runs", []) or []
        if isinstance(run, Mapping) and _string(run.get("scenario_eval_run_id"))
    ]


def _attempt_for_matrix_run(
    *,
    attempts: Sequence[Mapping[str, Any]],
    matrix_run: Mapping[str, Any],
    fallback_index: int,
) -> Mapping[str, Any]:
    task_id = _string(matrix_run.get("task_id") or matrix_run.get("taskId"))
    scenario_id = _string(matrix_run.get("scenario_id") or matrix_run.get("scenarioId"))
    for attempt in attempts:
        if (
            _string(attempt.get("scenario_id") or attempt.get("scenarioId")) == scenario_id
            and (
                not task_id
                or _string(attempt.get("task_id") or attempt.get("taskId")) == task_id
            )
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
        source_attempt_id = _string(source.get("attempt_id") or source.get("attemptId")) or "fixture_attempt"
        expanded = {
            **dict(source),
            "attempt_id": f"{source_attempt_id}__{run_id}",
            "scenario_eval_run_id": run_id,
            "scenario_variation_instance_id": _string(
                matrix_run.get("scenario_variation_instance_id")
                or matrix_run.get("scenarioVariationInstanceId")
            )
            or None,
            "variation_name": _string(matrix_run.get("variation_name") or matrix_run.get("variationName"))
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

    failures = [attempt for attempt in expanded_attempts if not bool(attempt.get("success"))]
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
        "labels": [
            {
                "label_id": f"fixture_label_{index:04d}",
                "attempt_id": attempt.get("attempt_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
                "variation_name": attempt.get("variation_name"),
                "task_id": attempt.get("task_id"),
                "scenario_id": attempt.get("scenario_id"),
                "policy_id": attempt.get("policy_id"),
                "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
                "failure_reason": _string(attempt.get("failure_reason")) or None,
                "status": "review_required",
                "proof_effect": "none_until_review_accepted_or_owner_proof_supplied",
            }
            for index, attempt in enumerate(failures, start=1)
        ],
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
            "predicted_status": "passed" if attempt.get("success") else "failed",
            "predicted_success": bool(attempt.get("success")),
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
    breakage_library = {
        **_mapping(copied.get("breakage_library")),
        "schema_version": BREAKAGE_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_breakages_recorded",
        "record_count": len(failures),
        "records": [
            {
                "scenario_id": attempt.get("scenario_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
                "variation_name": attempt.get("variation_name"),
                "task_id": attempt.get("task_id"),
                "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
                "failure_reason": _string(attempt.get("failure_reason")) or None,
                "review_required": True,
            }
            for attempt in failures
        ],
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
        result = {
            **dict(result),
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
    successes = sum(1 for attempt in attempts if bool(attempt.get("success")))
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
            "mean_seconds": round(sum(cycle_times) / len(cycle_times), 6)
            if cycle_times
            else None,
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
        attempts = [
            item for item in trace.get("attempts", []) or [] if isinstance(item, Mapping)
        ]
        trace_status = _string(trace.get("status"))
        if trace_status.startswith("blocked") or trace_status in {"not_available", "missing"}:
            status = "blocked"
            blockers = _string_list(trace.get("blockers")) or ["normalized_attempt_trace_missing"]
        elif attempts and any(not bool(item.get("success")) for item in attempts):
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
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


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
            "This report summarizes the eval harness output. It does not upgrade robot readiness, safety, simulator execution, policy execution, or deployment claims beyond the referenced proof-boundary and live-closure artifacts.",
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
            "robot_policy_execution_proven": bool(
                policy_execution_manifest.get("robot_policy_execution_proven")
            ),
        },
        "evaluation_status": evaluation_result.get("status"),
        "evaluator_scores": _mapping(evaluation_result.get("standard_policy_scorecard")),
        "real_world_validation": {
            "deployment_outcome_status": deployment_ledger.get("status"),
            "real_world_outcome_records_present": bool(
                deployment_ledger.get("real_world_outcome_records_present")
            ),
            "real_world_outcome_proven": bool(
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
            "sim_vs_real_calibration_score": calibration.get(
                "sim_vs_real_calibration_score"
            ),
            "prediction_vs_actual_deployment_summary_path": (
                "prediction_vs_actual_deployment_summary.json"
            ),
        },
        "live_eval_closure": {
            "status": live_closure.get("status"),
            "repo_local_artifacts_ready": bool(live_closure.get("repo_local_artifacts_ready")),
            "live_external_ready": bool(live_closure.get("live_external_ready")),
            "live_end_to_end_verified": bool(live_closure.get("live_end_to_end_verified")),
            "blockers": _string_list(live_closure.get("blockers")),
        },
        "requirement_coverage": {
            "schema_version": _mapping(live_closure.get("requirement_coverage")).get(
                "schema_version"
            ),
            "requirement_count": _mapping(live_closure.get("requirement_coverage")).get(
                "requirement_count"
            ),
            "passed_count": _mapping(live_closure.get("requirement_coverage")).get(
                "passed_count"
            ),
            "blocked_count": _mapping(live_closure.get("requirement_coverage")).get(
                "blocked_count"
            ),
            "blocked_requirement_ids": _string_list(
                _mapping(live_closure.get("requirement_coverage")).get(
                    "blocked_requirement_ids"
                )
            ),
        },
        "proof_boundary": {
            "simulator_execution_proven": bool(
                proof_boundary.get("simulator_execution_proven")
            ),
            "robot_policy_execution_proven": bool(
                proof_boundary.get("robot_policy_execution_proven")
            ),
            "real_world_outcome_proven": bool(
                proof_boundary.get("real_world_outcome_proven")
            ),
            "physics_contact_validated": bool(
                proof_boundary.get("physics_contact_validated")
            ),
            "safety_validated": bool(proof_boundary.get("safety_validated")),
            "robot_readiness_proven": bool(proof_boundary.get("robot_readiness_proven")),
            "public_claim_upgrade_allowed": bool(
                proof_boundary.get("public_claim_upgrade_allowed")
            ),
        },
        "artifact_paths": {
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "evaluation_result": "evaluation_result.json",
            "policy_execution_manifest": "policy_execution_manifest.json",
            "policy_execution_trace": "policy_execution_trace.json",
            "deployment_outcome_ledger": "deployment_outcome_ledger.json",
            "prediction_vs_actual_deployment_summary": (
                "prediction_vs_actual_deployment_summary.json"
            ),
            "real_world_validation_followup_plan": (
                "real_world_validation_followup_plan.json"
            ),
            "real_world_validation_followup_request_queue": (
                "real_world_validation_followup_request_queue.json"
            ),
            "live_eval_closure_manifest": "live_eval_closure_manifest.json",
            "proof_boundary": "proof_boundary.json",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    _write_job_json(job_dir, "robot_eval_report.json", report)
    write_text(job_dir / "robot_eval_report.md", _robot_eval_report_markdown(report))
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
    training_completed = bool(training_result.get("training_completed"))
    simulator_proven = bool(simulator_result.get("simulator_execution_proven")) and simulator != "fixture"
    policy_execution_proven = bool(
        _mapping(policy_execution_manifest).get("robot_policy_execution_proven")
    )
    real_world_outcome_proven = bool(
        _mapping(deployment_outcome_ledger).get("real_world_outcome_proven")
    )
    real_world_outcome_records_present = bool(
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
        "simulators_run": bool(simulator_result.get("simulators_run")),
        "gpu_training_run": bool(training_result.get("gpu_training_run")),
        "simulator_execution_proven": simulator_proven,
        "robot_readiness_proven": False,
        "robot_policy_execution_proven": policy_execution_proven,
        "real_world_outcome_records_present": real_world_outcome_records_present,
        "owner_evidence_record_count": owner_evidence_record_count,
        "missing_owner_evidence_record_ids": missing_owner_evidence_record_ids,
        "real_world_outcome_proven": real_world_outcome_proven,
        "physics_contact_validated": False,
        "safety_validated": False,
        "training_completed": training_completed,
        "public_claim_upgrade_allowed": False,
        "remaining_required_evidence": remaining,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulators_run": bool(simulator_result.get("simulators_run")),
            "gpu_training_run": bool(training_result.get("gpu_training_run")),
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
    updated = {
        **dict(proof_boundary),
        "live_eval_closure_status": live_closure.get("status"),
        "live_eval_closure_manifest_path": "live_eval_closure_manifest.json",
        "live_end_to_end_verified": bool(live_closure.get("live_end_to_end_verified")),
        "live_eval_closure_blockers": _string_list(live_closure.get("blockers")),
    }
    if bool(closure_boundary.get("live_end_to_end_verified")):
        for field in (
            "simulator_execution_proven",
            "robot_policy_execution_proven",
            "real_world_outcome_proven",
            "physics_contact_validated",
            "safety_validated",
            "robot_readiness_proven",
            "public_claim_upgrade_allowed",
        ):
            updated[field] = bool(closure_boundary.get(field))
        updated["status"] = "live_end_to_end_verified"
        updated["remaining_required_evidence"] = []
    updated["claim_boundary"] = {
        **_mapping(updated.get("claim_boundary")),
        "live_eval_closure_manifest_path": "live_eval_closure_manifest.json",
        "live_end_to_end_verified": bool(live_closure.get("live_end_to_end_verified")),
        "simulator_execution_proven": bool(updated.get("simulator_execution_proven")),
        "robot_policy_execution_proven": bool(updated.get("robot_policy_execution_proven")),
        "real_world_outcome_proven": bool(updated.get("real_world_outcome_proven")),
        "physics_contact_validated": bool(updated.get("physics_contact_validated")),
        "safety_validated": bool(updated.get("safety_validated")),
        "robot_readiness_proven": bool(updated.get("robot_readiness_proven")),
        "public_claim_upgrade_allowed": bool(updated.get("public_claim_upgrade_allowed")),
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
        "simulator_provider_adapter_manifest.json",
        "simulator_command_artifacts_manifest.json",
        "scenario_eval_matrix.json",
        "policy_package_manifest.json",
        "robot_pov_observation_manifest.json",
        "robot_pov_observations.jsonl",
        "robot_pov_frame_sequence_manifest.json",
        "robot_pov_render_storyboard.json",
        "policy_execution_manifest.json",
        "policy_execution_trace.json",
        "policy_execution_trace.jsonl",
        "training_request.json",
        "training_result.json",
        "evaluation_request.json",
        "evaluation_result.json",
        "robot_eval_report.json",
        "arena_eval_schedule.json",
        "arena_eval_retry_queue.json",
        "arena_eval_cost_ledger.json",
        "arena_eval_resume_manifest.json",
        "policy_adapter_manifest.json",
        "arena_result_ingest_ledger.json",
        "arena_artifact_checksums.json",
        "arena_eval_metrics.json",
        "normalized_attempt_trace.json",
        "failure_labels.json",
        "clips_manifest.json",
        "rollout_vision_labels.json",
        "review_resolution_ledger.json",
        "accepted_failure_labels.json",
        "prediction_outcome_ledger.json",
        "calibration_report.json",
        "breakage_library.json",
        "deployment_outcome_ledger.json",
        "sim_vs_real_calibration_report.json",
        "prediction_vs_actual_deployment_summary.json",
        "real_world_validation_followup_plan.json",
        "real_world_validation_followup_request_queue.json",
        "live_eval_closure_manifest.json",
        "arena_rerun_plan.json",
        "arena_rerun_lineage.json",
        "customer_handoff_report.json",
        "customer_handoff_report.md",
        "delivery_manifest.json",
        "signed_access_manifest.json",
        "live_operator_ledger.json",
        "dataset_card.json",
        "license_manifest.json",
        "package_index.json",
        "checksums.json",
        "archive_manifest.json",
        "post_training_data_package_export_manifest.json",
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
    context = resolve_local_capture_context(capture_root)
    generated_at = utc_now_iso()
    pipeline_dir = context.pipeline_root
    job_dir = pipeline_dir / "robot_eval_jobs" / job_id
    ensure_dir(job_dir)

    request = _read_job_request(job_request)
    request.setdefault("schema_version", JOB_REQUEST_SCHEMA_VERSION)
    request.setdefault("job_id", job_id)
    request.setdefault("capture_root", str(context.capture_root))
    request = _apply_staged_policy_package(
        request=request,
        capture_root=context.capture_root,
        job_id=job_id,
    )
    _write_job_json(job_dir, "job_request.json", request)

    missing_robot_eval_inputs = _ensure_robot_eval_cards(
        capture_root=context.capture_root,
        pipeline_dir=pipeline_dir,
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
    scenario_eval_matrix = build_scenario_eval_matrix(
        capture_root=context.capture_root,
        job_dir=job_dir,
        job_request=request,
        generated_at=generated_at,
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
        "simulation_automation": simulation_automation,
        "scenario_eval_matrix": scenario_eval_matrix,
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
        simulator_commands=simulator_commands or {},
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
        copied_artifacts["normalized_attempt_trace"] = _mapping(
            arena_ingest.get("attempt_trace")
        )
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

    proof_boundary = _proof_boundary(
        simulator=simulator,
        simulator_result=sim_result,
        training_result=training_res,
        policy_execution_manifest=_mapping(policy_execution.get("manifest")),
        deployment_outcome_ledger=_mapping(deployment_validation.get("ledger")),
        generated_at=generated_at,
    )
    _write_job_json(job_dir, "proof_boundary.json", proof_boundary)
    blockers: List[str] = []
    missing_inputs: List[str] = []
    evidence: Dict[str, Any] = {
        "job_validation_status": validation.get("status"),
        "gpu_provisioning_status": gpu_result.get("status"),
        "simulator_service_status": sim_result.get("status"),
        "robot_pov_status": robot_pov_manifest.get("status"),
        "robot_pov_evidence_proven": bool(
            robot_pov_manifest.get("robot_pov_evidence_proven")
        ),
        "policy_execution_status": _mapping(policy_execution.get("manifest")).get("status"),
        "deployment_outcome_status": _mapping(deployment_validation.get("ledger")).get("status"),
        "real_world_validation_followup_plan_status": _mapping(
            deployment_validation.get("followup_plan")
        ).get("status"),
        "real_world_outcome_records_present": bool(
            _mapping(deployment_validation.get("ledger")).get(
                "real_world_outcome_records_present"
            )
        ),
        "owner_evidence_record_count": int(
            _mapping(deployment_validation.get("ledger")).get("owner_evidence_record_count") or 0
        ),
        "missing_owner_evidence_record_ids": _string_list(
            _mapping(deployment_validation.get("ledger")).get(
                "missing_owner_evidence_record_ids"
            )
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
    evidence["live_eval_closure_status"] = live_closure.get("status")
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
    data_package_export = build_post_training_data_package_export(
        capture_root=context.capture_root,
        job_dir=job_dir,
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
        "agent_orchestration_status": agent_plan.get("status"),
        "agent_operator_mode": agent_plan.get("operator_mode"),
        "agent_operator_ledger": "agent_orchestration_plan.json",
        "scene_asset_preflight_status": scene_preflight.get("status"),
        "episode_spec_status": episode_specs.get("status"),
        "episode_count": episode_specs.get("episode_count"),
        "cpu_simulator_preflight_status": cpu_preflight.get("status"),
        "simulation_automation_status": simulation_automation.get("status"),
        "validation_status": validation.get("status"),
        "gpu_provisioning_status": gpu_result.get("status"),
        "simulator_service_status": sim_result.get("status"),
        "scenario_eval_matrix_status": scenario_eval_matrix.get("status"),
        "scenario_eval_run_count": scenario_eval_matrix.get("scenario_eval_run_count"),
        "scenario_variation_names_covered": scenario_eval_matrix.get("variation_names_covered"),
        "robot_pov_observation_status": robot_pov_manifest.get("status"),
        "robot_pov_evidence_proven": bool(
            robot_pov_manifest.get("robot_pov_evidence_proven")
        ),
        "policy_execution_status": _mapping(policy_execution.get("manifest")).get("status"),
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
        "live_end_to_end_verified": bool(live_closure.get("live_end_to_end_verified")),
        "training_status": training_res.get("status"),
        "evaluation_status": eval_result.get("status"),
        "robot_eval_report_status": robot_eval_report.get("status"),
        "robot_eval_report_path": "robot_eval_report.json",
        "post_training_data_package_export_status": data_package_export.get("status"),
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
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "robot_pov_observation_manifest": "robot_pov_observation_manifest.json",
            "robot_pov_observations": "robot_pov_observations.jsonl",
            "robot_pov_frame_sequence_manifest": "robot_pov_frame_sequence_manifest.json",
            "robot_pov_render_storyboard": "robot_pov_render_storyboard.json",
            "policy_execution_manifest": "policy_execution_manifest.json",
            "policy_execution_trace": "policy_execution_trace.json",
            "policy_execution_trace_jsonl": "policy_execution_trace.jsonl",
            "deployment_outcome_intake_manifest": "deployment_outcome_intake_manifest.json",
            "deployment_outcome_ledger": "deployment_outcome_ledger.json",
            "sim_vs_real_calibration_report": "sim_vs_real_calibration_report.json",
            "prediction_vs_actual_deployment_summary": (
                "prediction_vs_actual_deployment_summary.json"
            ),
            "real_world_validation_followup_plan": (
                "real_world_validation_followup_plan.json"
            ),
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
            "customer_handoff_report": "customer_handoff_report.json",
            "delivery_manifest": "delivery_manifest.json",
            "arena_rerun_plan": "arena_rerun_plan.json",
            "live_operator_ledger": "live_operator_ledger.json",
        },
        "live_provider_calls_performed": False,
        "remote_asset_downloads_performed": False,
        "local_cpu_preflight_smoke_ran": bool(
            _read_optional_mapping(
                pipeline_dir / "simulation_automation" / "cpu_simulator_preflight_manifest.json"
            ).get("local_cpu_smoke_ran")
        ),
        "simulators_run": bool(sim_result.get("simulators_run")),
        "gpu_training_run": bool(training_res.get("gpu_training_run")),
        "messages_sent": False,
        "payments_touched": False,
        "deployments_performed": False,
        "simulator_execution_proven": bool(proof_boundary.get("simulator_execution_proven")),
        "robot_policy_execution_proven": bool(
            proof_boundary.get("robot_policy_execution_proven")
        ),
        "real_world_outcome_records_present": bool(
            proof_boundary.get("real_world_outcome_records_present")
        ),
        "owner_evidence_record_count": int(
            proof_boundary.get("owner_evidence_record_count") or 0
        ),
        "missing_owner_evidence_record_ids": _string_list(
            proof_boundary.get("missing_owner_evidence_record_ids")
        ),
        "real_world_outcome_proven": bool(proof_boundary.get("real_world_outcome_proven")),
        "physics_contact_validated": bool(proof_boundary.get("physics_contact_validated")),
        "safety_validated": bool(proof_boundary.get("safety_validated")),
        "robot_readiness_proven": bool(proof_boundary.get("robot_readiness_proven")),
        "public_claim_upgrade_allowed": bool(proof_boundary.get("public_claim_upgrade_allowed")),
        "live_eval_closure_blockers": _string_list(live_closure.get("blockers")),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "live_eval_closure_manifest_path": "live_eval_closure_manifest.json",
            "live_end_to_end_verified": bool(live_closure.get("live_end_to_end_verified")),
            "simulator_execution_proven": bool(proof_boundary.get("simulator_execution_proven")),
            "robot_policy_execution_proven": bool(
                proof_boundary.get("robot_policy_execution_proven")
            ),
            "real_world_outcome_records_present": bool(
                proof_boundary.get("real_world_outcome_records_present")
            ),
            "real_world_outcome_proven": bool(proof_boundary.get("real_world_outcome_proven")),
            "physics_contact_validated": bool(proof_boundary.get("physics_contact_validated")),
            "safety_validated": bool(proof_boundary.get("safety_validated")),
            "robot_readiness_proven": bool(proof_boundary.get("robot_readiness_proven")),
            "public_claim_upgrade_allowed": bool(
                proof_boundary.get("public_claim_upgrade_allowed")
            ),
        },
    }
    run_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "job_id": job_id,
            "validation": validation,
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
        "live_eval_closure_status": live_closure.get("status"),
        "live_end_to_end_verified": bool(live_closure.get("live_end_to_end_verified")),
        "claim_boundary": dict(run_manifest["claim_boundary"]),
    }


def _job_id_from_request(path: Path, request: Mapping[str, Any]) -> str:
    raw = _string(
        request.get("job_id")
        or request.get("jobId")
        or _mapping(request.get("owner_system")).get("request_id")
        or path.stem
    )
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in raw)
    return cleaned.strip("-_") or path.stem


def run_robot_eval_job_request_inbox(
    *,
    capture_root: str | Path,
    inbox_dir: str | Path,
    agent_adapter: RobotEvalJobAgentAdapter | None = None,
    provisioner: str = "fixture_local",
    simulator: str = "fixture",
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
    request_paths = sorted(
        path for path in inbox_path.glob("*.json") if path.is_file() and not path.name.startswith(".")
    )
    jobs: List[Dict[str, Any]] = []
    for request_path in request_paths:
        request = _read_job_request(request_path)
        request.setdefault("schema_version", JOB_REQUEST_SCHEMA_VERSION)
        job_id = _job_id_from_request(request_path, request)
        request["job_id"] = job_id
        request["capture_root"] = str(context.capture_root)
        queued_dir = queue_root / job_id
        ensure_dir(queued_dir)
        write_json(queued_dir / "job_request.json", request)
        result = build_robot_eval_job(
            capture_root=context.capture_root,
            job_request=request,
            job_id=job_id,
            agent_adapter=agent_adapter,
            provisioner=provisioner,
            simulator=simulator,
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
        jobs.append(
            {
                "job_id": job_id,
                "status": result["status"],
                "source_request_path": str(request_path),
                "queued_request_path": str((queued_dir / "job_request.json").resolve()),
                "job_dir": result["job_dir"],
                "job_run_manifest_uri": result["manifest_path"],
                "public_claim_upgrade_allowed": False,
            }
        )
    status = "completed" if jobs else "empty"
    manifest = {
        "schema_version": JOB_REQUEST_INBOX_RUN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "capture_root": str(context.capture_root),
        "inbox_dir": str(inbox_path),
        "queue_root": str(queue_root),
        "processed_count": len(jobs),
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


def _agent_adapter_from_mode(mode: str, *, allow_live_operator: bool) -> RobotEvalJobAgentAdapter | None:
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
        simulator_commands = _parse_simulator_commands(args.simulator_command)
        policy_execution_commands = _parse_policy_execution_commands(
            args.policy_execution_command
        )
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
            return 0
        if not args.job_request or not args.job_id:
            raise ValueError("--job-request and --job-id are required unless --job-request-inbox is provided")
        result = build_robot_eval_job(
            capture_root=args.capture_root,
            job_request=args.job_request,
            job_id=args.job_id,
            agent_adapter=_agent_adapter_from_mode(
                args.agent_mode,
                allow_live_operator=args.allow_live_agent_operator,
            ),
            provisioner=args.provisioner,
            simulator=args.simulator,
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
    except (OSError, ValueError) as exc:
        print(f"[robot-eval-job] FAILED: {exc}")
        return 1
    print(f"[robot-eval-job] manifest={result['manifest_path']}")
    print(f"[robot-eval-job] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
