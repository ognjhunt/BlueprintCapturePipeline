"""Fail-closed simulation automation orchestration lane with gated live SDK operators.

This module turns existing capture/package/World Labs/Marble artifacts into a
deterministic simulation automation package. It plans conversion, simulator
execution, training, evaluation, and proof collection without calling providers,
downloading assets, running simulators, or running GPU training unless explicit
run-time approvals are present. Agents SDK and Codex SDK adapters may execute as
live operators under explicit gates to coordinate failures, retries, summaries,
and code fixes, but deterministic artifacts own proof state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from .agent_operator_runtime import (
    CODEX_CLI_HOST_OAUTH_ENV,
    LIVE_AGENTS_SDK_ENV,
    LIVE_CODEX_SDK_ENV,
    OperatorExecutor,
    OperatorRunConfig,
    blocked_operator_ledger,
    codex_cli_path as resolve_codex_cli_path,
    completed_operator_ledger,
    env_truthy,
    external_action_gates,
    proof_effect,
    run_agents_sdk_operator,
    run_codex_cli_operator,
    run_codex_sdk_operator,
)
from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .cpu_simulator_preflight import CPU_BACKENDS, build_cpu_simulator_preflight
from .episode_spec import EpisodeSpecAgentAdapter, FakeEpisodeSpecAgentAdapter, build_episode_specs
from .local_capture import resolve_local_capture_context
from .scene_asset_preflight import build_scene_asset_preflight
from .scenario_variation_instantiator import build_scenario_variation_instances


SIMULATION_AUTOMATION_SCHEMA_VERSION = "simulation_automation_plan.v1"
SIMULATION_AUTOMATION_RUN_SCHEMA_VERSION = "simulation_automation_run_manifest.v1"
ASSET_CONVERSION_PLAN_SCHEMA_VERSION = "simulation_asset_conversion_plan.v1"
SIMULATOR_EXECUTION_SCHEMA_VERSION = "simulator_execution_manifest.v1"
SIMULATOR_ENGINE_PLUGIN_REGISTRY_SCHEMA_VERSION = "simulator_engine_plugin_registry.v1"
SIMULATOR_REQUEST_SCHEMA_VERSION = "simulator_request.v1"
SIMULATOR_RESULT_SCHEMA_VERSION = "simulator_result.v1"
TRAINING_ORCHESTRATION_SCHEMA_VERSION = "training_orchestration_manifest.v1"
PROOF_BOUNDARY_SCHEMA_VERSION = "simulation_automation_proof_boundary.v1"
AGENT_LEDGER_SCHEMA_VERSION = "simulation_automation_agent_decision_ledger.v1"
GPU_HANDOFF_PACKET_SCHEMA_VERSION = "gpu_handoff_packet.v1"
GPU_OWNER_SYSTEM_PROOF_SCHEMA_VERSION = "gpu_owner_system_proof_schema.v1"
OWNER_GPU_BLOCKED_MANIFEST_SCHEMA_VERSION = "owner_gpu_simulator_execution_blocked_manifest.v1"
OWNER_GPU_PROOF_MANIFEST_SCHEMA_VERSION = "owner_gpu_simulator_execution_proof_manifest.v1"
ARENA_ENVIRONMENT_PACKET_SCHEMA_VERSION = "arena_environment_packet.v1"

SIMULATOR_FRAMEWORKS = ("isaac_sim", "isaac_lab_arena", "mujoco", "pybullet", "newton")
ISAAC_SIMULATOR_FRAMEWORKS = {"isaac_sim", "isaac_lab_arena"}
DEFAULT_ISAAC_HUMANOID_ROBOT_ASSET = {
    "name": "Unitree G1",
    "uri_or_path": "Robots/Unitree/G1/g1.usd",
    "source": "isaac_sim_robot_assets",
    "asset_class": "humanoid",
    "catalog_reference": "Isaac Sim Robot Assets: Robots/Unitree/G1/g1.usd",
}
WORLD_MODEL_ENGINE_TARGETS = (
    "worldlabs_world_model",
    "marble_simready",
    "cosmos_predict",
    "native_site_reference",
)
TRAINING_RUNNER = "blueprint_pipeline.synthesis.cosmos_lora_training.run_cosmos_lora_training"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "simulation_automation_orchestration_only",
    "repo_local_only_by_default": True,
    "agent_operator_mode_allowed": True,
    "agents_may_mutate_proof_booleans": False,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "simulators_run": False,
    "gpu_training_run": False,
    "messages_sent": False,
    "payments_touched": False,
    "deployments_performed": False,
    "simulator_execution_proven": False,
    "isaac_sim_execution_proven": False,
    "isaac_robot_asset_execution_proven": False,
    "robot_readiness_proven": False,
    "training_proof_available": False,
    "robot_policy_execution_proven": False,
    "physics_contact_validated": False,
    "safety_contact_proof_available": False,
    "public_claim_upgrade_allowed": False,
    "disallowed_claims": [
        "robot_ready",
        "deployment_ready",
        "simulator_execution_completed",
        "physics_validated",
        "robot_policy_execution_passed",
        "training_completed",
        "safety_contact_validated",
    ],
    "proof_upgrade_requires": [
        "simulator load trace",
        "simulator stdout/stderr and exit code",
        "action or policy logs",
        "physics/contact validation logs",
        "training run logs and checkpoint manifest",
        "robot-team-owned robot assets",
        "accepted simulator or real robot trial evidence",
    ],
}


class SimulationAutomationAgentAdapter(Protocol):
    """Optional agent operator interface.

    The deterministic orchestration code owns manifest status and claim
    boundaries. Agent adapters may execute live work under gates, but cannot set
    proof booleans directly.
    """

    def build_ledger(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class FakeSimulationAutomationAgentAdapter:
    """Network-free adapter used by tests and local smoke runs."""

    adapter_name: str = "fake"

    def build_ledger(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        del plan_context
        return {
            "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
            "adapter": self.adapter_name,
            "status": "completed",
            "operator_mode": "deterministic_test_operator",
            "network_required": False,
            "live_provider_calls_performed": False,
            "proof_booleans_mutable_by_agent": False,
            "decisions": [
                {
                    "decision": "plan_next_actions",
                    "summary": (
                        "Use deterministic manifests; keep simulator and training execution blocked until explicit approvals and dependencies exist."
                    ),
                    "owned_by": "deterministic_orchestrator",
                }
            ],
            "diagnostics": [
                {
                    "status": "blocked",
                    "blockers": ["approval_required"],
                    "summary": "Simulator execution and training are intentionally blocked by default.",
                }
            ],
            "operator_ledger": completed_operator_ledger(
                adapter=self.adapter_name,
                output={
                    "final_output": "Local fake adapter selected deterministic manifest-first work.",
                    "commands_chosen": ["build_simulation_automation"],
                },
                default_command="build_simulation_automation",
                proof_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"],
            ),
            "proof_effect": proof_effect(
                deterministic_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"]
            ),
        }


@dataclass(frozen=True)
class CodexSdkSimulationAutomationAgentAdapter:
    """Optional Codex SDK live code-maintainer operator."""

    thread_id: str | None = None
    sandbox: str = "workspace-write"
    codex_sdk_available: bool | None = None
    openai_api_key: str | None = None
    codex_cli_path: str | None = None
    live_env_allowed: bool | None = None
    allow_live_operator: bool = False
    model: str = "gpt-4.1"
    executor: OperatorExecutor | None = None

    def build_ledger(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        packages = ["openai_codex", "openai_codex_sdk", "codex_sdk"]
        installed = next((name for name in packages if importlib.util.find_spec(name)), None)
        sdk_available = (
            self.codex_sdk_available
            if self.codex_sdk_available is not None
            else bool(self.executor is not None) or bool(installed)
        )
        api_key_present = bool(
            _string(self.openai_api_key)
            if self.openai_api_key is not None
            else _string(os.getenv("OPENAI_API_KEY"))
        )
        resolved_codex_cli = _string(self.codex_cli_path) or resolve_codex_cli_path()
        codex_cli_ready = bool(resolved_codex_cli)
        codex_cli_host_oauth_allowed = env_truthy(CODEX_CLI_HOST_OAUTH_ENV)
        transport_ready = bool(self.executor is not None) or (
            sdk_available and api_key_present
        ) or (codex_cli_ready and codex_cli_host_oauth_allowed)
        env_allowed = (
            bool(self.live_env_allowed)
            if self.live_env_allowed is not None
            else env_truthy(LIVE_CODEX_SDK_ENV)
        )
        request = {
            "action": "resume_thread" if self.thread_id else "start_thread",
            "thread_id": self.thread_id,
            "sandbox": self.sandbox if self.sandbox in {"read-only", "workspace-write"} else "read-only",
            "workspace": str(plan_context.get("repo_root") or ""),
            "prompt_purpose": "simulation_automation_live_code_maintenance",
            "allowed_actions": [
                "diagnose_pipeline_failure",
                "patch_code",
                "run_focused_tests",
                "produce_diff_summary",
                "summarize_remaining_blockers",
            ],
            "prohibited_actions": [
                "set_proof_booleans_true_without_deterministic_artifacts",
                "call_live_providers_without_explicit_gates",
                "spend_money_without_explicit_gates",
            ],
        }
        blockers: List[str] = []
        if not transport_ready and not sdk_available:
            blockers.append("missing_codex_sdk")
        if not transport_ready and not api_key_present:
            blockers.append("missing_openai_api_key")
        if not transport_ready and not codex_cli_ready:
            blockers.append("missing_codex_cli")
        if not transport_ready and codex_cli_ready and not codex_cli_host_oauth_allowed:
            blockers.append(f"missing_env_{CODEX_CLI_HOST_OAUTH_ENV}")
        if not self.allow_live_operator:
            blockers.append("missing_cli_allow_live_codex_sdk_operator")
        if not env_allowed:
            blockers.append(f"missing_env_{LIVE_CODEX_SDK_ENV}")
        live_output: Dict[str, Any] | None = None
        execution_performed = False
        execution_failed = False
        if not blockers:
            try:
                config = OperatorRunConfig(
                    adapter="codex_sdk_simulation_code_maintainer",
                    model=self.model,
                    prompt=_codex_simulation_operator_prompt(plan_context),
                    plan_context=plan_context,
                    executor=self.executor,
                    sandbox=_string(request["sandbox"]),
                    cwd=_string(plan_context.get("repo_root")) or None,
                    codex_bin=resolved_codex_cli or "codex",
                )
                live_output = (
                    run_codex_sdk_operator(config)
                    if self.executor is not None or (sdk_available and api_key_present)
                    else run_codex_cli_operator(
                        OperatorRunConfig(
                            **{
                                **config.__dict__,
                                "adapter": "codex_cli_simulation_code_maintainer",
                            }
                        )
                    )
                )
                execution_performed = True
            except RuntimeError as exc:
                blockers.append(str(exc))
                execution_failed = True
            except Exception as exc:
                blockers.append(f"codex_sdk_operator_execution_failed:{type(exc).__name__}")
                execution_failed = True
        if blockers:
            return {
                "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
                "adapter": "codex_sdk",
                "status": "operator_failed" if execution_failed else "blocked",
                "operator_mode": "live_operator_blocked",
                "network_required": True,
                "optional_dependency": installed or "openai-codex",
                "reason": "live_operator_gate_or_execution_blocked",
                "execution_performed": execution_performed,
                "request": request,
                "decisions": [],
                "operator_ledger": blocked_operator_ledger(
                    adapter="codex_sdk_simulation_code_maintainer",
                    blockers=blockers,
                    command_chosen=None,
                    proof_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"],
                ),
                "diagnostics": [
                    {
                        "status": "blocked",
                        "blockers": blockers,
                        "summary": "Codex SDK live code-maintainer execution was blocked or failed.",
                    }
                ],
                "evidence": {
                    "codex_sdk_available": bool(sdk_available),
                    "codex_cli_available": codex_cli_ready,
                    "codex_cli_host_oauth_allowed": codex_cli_host_oauth_allowed,
                    "openai_api_key_present": api_key_present,
                    "cli_allow_live_operator": self.allow_live_operator,
                    LIVE_CODEX_SDK_ENV: env_allowed,
                    CODEX_CLI_HOST_OAUTH_ENV: codex_cli_host_oauth_allowed,
                    **external_action_gates(),
                },
                "proof_effect": proof_effect(
                    deterministic_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"]
                ),
            }
        operator_ledger = completed_operator_ledger(
            adapter="codex_sdk_simulation_code_maintainer",
            output=live_output or {},
            default_command="diagnose_patch_and_test_simulation_automation",
            proof_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"],
        )
        return {
            "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
            "adapter": "codex_sdk",
            "status": "operator_completed",
            "operator_mode": "live_operator",
            "network_required": True,
            "execution_performed": True,
            "optional_dependency": installed,
            "request": request,
            "decisions": operator_ledger["decisions"],
            "operator_ledger": operator_ledger,
            "diagnostics": [],
            "evidence": {
                "codex_sdk_available": bool(sdk_available),
                "codex_cli_available": codex_cli_ready,
                "codex_cli_host_oauth_allowed": codex_cli_host_oauth_allowed,
                "openai_api_key_present": api_key_present,
                "cli_allow_live_operator": self.allow_live_operator,
                LIVE_CODEX_SDK_ENV: env_allowed,
                CODEX_CLI_HOST_OAUTH_ENV: codex_cli_host_oauth_allowed,
                **external_action_gates(),
            },
            "proof_effect": proof_effect(
                deterministic_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"]
            ),
        }


@dataclass(frozen=True)
class AgentsSdkCodexMCPAdapter:
    """Optional Agents SDK pipeline operator with Codex-compatible workspace scope."""

    sandbox: str = "workspace-write"
    agents_sdk_available: bool | None = None
    openai_api_key: str | None = None
    live_env_allowed: bool | None = None
    allow_live_operator: bool = False
    model: str = "gpt-4.1"
    executor: OperatorExecutor | None = None

    def build_ledger(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        agents_package = next(
            (name for name in ("agents", "openai_agents") if importlib.util.find_spec(name)),
            None,
        )
        agents_available = (
            self.agents_sdk_available
            if self.agents_sdk_available is not None
            else bool(self.executor is not None) or bool(agents_package)
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
        )
        request = {
            "agent_type": "openai_agents_sdk",
            "mcp_server": "codex",
            "workspace": str(plan_context.get("repo_root") or ""),
            "sandbox": self.sandbox if self.sandbox in {"read-only", "workspace-write"} else "read-only",
            "tool_scope": [
                "inspect_manifests_and_logs",
                "choose_next_deterministic_command",
                "trigger_allowed_reruns",
                "diagnose_failures",
                "summarize_traces",
                "route_review_items",
                "maintain_progress_ledger",
            ],
        }
        blockers: List[str] = []
        if not agents_available:
            blockers.append("missing_openai_agents_sdk")
        if not api_key_present:
            blockers.append("missing_openai_api_key")
        if not self.allow_live_operator:
            blockers.append("missing_cli_allow_live_agents_sdk_operator")
        if not env_allowed:
            blockers.append(f"missing_env_{LIVE_AGENTS_SDK_ENV}")
        live_output: Dict[str, Any] | None = None
        execution_performed = False
        execution_failed = False
        if not blockers:
            try:
                live_output = run_agents_sdk_operator(
                    OperatorRunConfig(
                        adapter="openai_agents_sdk_simulation_operator",
                        model=self.model,
                        prompt=_agents_sdk_simulation_operator_prompt(plan_context),
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
        if blockers:
            return {
                "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
                "adapter": "openai_agents_sdk_codex_mcp",
                "status": "operator_failed" if execution_failed else "blocked",
                "operator_mode": "live_operator_blocked",
                "network_required": True,
                "optional_dependency": agents_package or "openai-agents",
                "reason": "live_operator_gate_or_execution_blocked",
                "execution_performed": execution_performed,
                "request": request,
                "decisions": [],
                "operator_ledger": blocked_operator_ledger(
                    adapter="openai_agents_sdk_simulation_operator",
                    blockers=blockers,
                    command_chosen=None,
                    proof_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"],
                ),
                "diagnostics": [
                    {
                        "status": "blocked",
                        "blockers": blockers,
                        "summary": "Agents SDK live pipeline operator execution was blocked or failed.",
                    }
                ],
                "evidence": {
                    "openai_agents_sdk_available": bool(agents_available),
                    "openai_api_key_present": api_key_present,
                    "cli_allow_live_operator": self.allow_live_operator,
                    LIVE_AGENTS_SDK_ENV: env_allowed,
                    **external_action_gates(),
                },
                "proof_effect": proof_effect(
                    deterministic_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"]
                ),
            }
        operator_ledger = completed_operator_ledger(
            adapter="openai_agents_sdk_simulation_operator",
            output=live_output or {},
            default_command="choose_next_simulation_automation_command",
            proof_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"],
        )
        return {
            "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
            "adapter": "openai_agents_sdk_codex_mcp",
            "status": "operator_completed",
            "operator_mode": "live_operator",
            "network_required": True,
            "execution_performed": True,
            "optional_dependency": agents_package,
            "request": request,
            "decisions": operator_ledger["decisions"],
            "operator_ledger": operator_ledger,
            "diagnostics": [],
            "evidence": {
                "openai_agents_sdk_available": bool(agents_available),
                "openai_api_key_present": api_key_present,
                "cli_allow_live_operator": self.allow_live_operator,
                LIVE_AGENTS_SDK_ENV: env_allowed,
                **external_action_gates(),
            },
            "proof_effect": proof_effect(
                deterministic_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"]
            ),
        }


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _codex_simulation_operator_prompt(plan_context: Mapping[str, Any]) -> str:
    return (
        "Act as the Blueprint Codex SDK code maintainer for the simulation automation "
        "lane. Diagnose failures, patch code if needed, run focused tests, and summarize "
        "diffs. Do not set proof booleans directly; proof can only come from deterministic "
        "accepted artifacts.\n\n"
        f"{json.dumps(plan_context, sort_keys=True, default=str)[:12000]}"
    )


def _agents_sdk_simulation_operator_prompt(plan_context: Mapping[str, Any]) -> str:
    return (
        "Act as the Blueprint Agents SDK simulation pipeline operator. Inspect manifests "
        "and logs, choose safe deterministic commands or reruns, summarize blockers, route "
        "review, and maintain a progress ledger. Do not set proof booleans directly.\n\n"
        f"{json.dumps(plan_context, sort_keys=True, default=str)[:12000]}"
    )


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


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


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _relative_if_file(base_dir: Path, target: Path) -> str | None:
    return _relative_to(base_dir, target) if target.is_file() else None


def _sha_payload(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _timestamp(*payloads: Mapping[str, Any]) -> str:
    for payload in payloads:
        for key in ("updated_at", "generated_at", "completed_at", "created_at"):
            value = _string(payload.get(key))
            if value:
                return value
    return utc_now_iso()


def _source_artifacts(*, automation_dir: Path, pipeline_dir: Path) -> Dict[str, str]:
    candidates = {
        "capture_descriptor": pipeline_dir.parent / "capture_descriptor.json",
        "raw_manifest": pipeline_dir.parent / "raw" / "manifest.json",
        "worldlabs_request_manifest": pipeline_dir / "worldlabs_request_manifest.json",
        "worldlabs_operation_manifest": pipeline_dir / "worldlabs_operation_manifest.json",
        "worldlabs_world_manifest": pipeline_dir / "worldlabs_world_manifest.json",
        "marble_simready_bridge": pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json",
        "marble_asset_validation": pipeline_dir / "marble_sim_assets" / "marble_asset_validation.json",
        "simready_scene_manifest": pipeline_dir / "simready" / "simready_scene_manifest.json",
        "simready_validation": pipeline_dir / "simready" / "simready_validation.json",
        "robot_eval_dataset_manifest": (
            pipeline_dir / "robot_eval_dataset" / "robot_eval_dataset_manifest.json"
        ),
        "scene_asset_inspection": automation_dir / "scene_asset_inspection.json",
        "scene_asset_inventory": automation_dir / "scene_asset_inventory.json",
        "scene_asset_dependency_audit": automation_dir / "scene_asset_dependency_audit.json",
        "scene_asset_preflight": automation_dir / "scene_asset_preflight.json",
        "scene_frame_estimate": automation_dir / "scene_frame_estimate.json",
        "collider_proxy_plan": automation_dir / "collider_proxy_plan.json",
        "cpu_scene_proxy_manifest": automation_dir / "cpu_scene_proxy_manifest.json",
        "cpu_preflight_scorecard": automation_dir / "cpu_preflight_scorecard.json",
        "task_anchor_proposal_manifest": automation_dir / "task_anchor_proposal_manifest.json",
        "episode_spec_manifest": automation_dir / "episode_spec_manifest.json",
        "episode_spec": automation_dir / "episode_spec.v1.json",
        "episode_specs": automation_dir / "episode_specs.json",
        "episode_setup_manifest": automation_dir / "episode_setup_manifest.json",
        "spawn_pose_validation_manifest": automation_dir / "spawn_pose_validation_manifest.json",
        "scenario_variation_instances": automation_dir / "scenario_variation_instances.json",
        "cpu_simulator_preflight_manifest": automation_dir / "cpu_simulator_preflight_manifest.json",
        "cpu_preflight_manifest": automation_dir / "cpu_preflight_manifest.json",
        "pre_gpu_readiness_summary": automation_dir / "pre_gpu_readiness_summary.json",
        "arena_environment_packet": automation_dir / "arena_environment_packet.json",
        "simulator_engine_plugin_registry": (
            automation_dir / "simulator_engine_plugin_registry.json"
        ),
        "gpu_handoff_packet": automation_dir / "gpu_handoff_packet.json",
        "gpu_owner_system_proof_schema": automation_dir / "gpu_owner_system_proof_schema.json",
        "owner_gpu_simulator_execution_blocked_manifest": (
            automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json"
        ),
        "cosmos_training_export": pipeline_dir / "cosmos_training_export" / "manifest.json",
        "cosmos_lora_training": (
            pipeline_dir / "cosmos_training_export" / "training_run_manifest.json"
        ),
        "robot_eval_site_card": pipeline_dir / "robot_eval_dataset" / "site_card.json",
        "robot_eval_task_cards": pipeline_dir / "robot_eval_dataset" / "task_cards.json",
        "robot_eval_scenario_cards": pipeline_dir / "robot_eval_dataset" / "scenario_cards.json",
        "robot_eval_eval_cards": pipeline_dir / "robot_eval_dataset" / "eval_cards.json",
        "robot_eval_proof_boundaries": (
            pipeline_dir / "robot_eval_dataset" / "proof_boundaries.json"
        ),
    }
    return {
        key: rel
        for key, path in sorted(candidates.items())
        if (rel := _relative_if_file(automation_dir, path))
    }


OWNER_GPU_REQUIRED_FIELDS = (
    "owner_system_id",
    "simulator_backend",
    "simulator_version",
    "gpu_model",
    "command",
    "started_at",
    "completed_at",
    "exit_code",
    "stdout_uri_or_path",
    "stderr_uri_or_path",
    "scene_load_trace_uri_or_path",
    "action_or_policy_trace_uri_or_path",
    "default_smoke_policy_uri_or_path",
    "policy_execution_trace_uri_or_path",
    "sim_robot_pov_evidence_uri_or_path",
    "artifact_manifest_uri_or_path",
    "pass_fail_criteria",
    "operator_attestation",
)

OWNER_GPU_FORBIDDEN_TRUE_FIELDS = (
    "isaac_robot_asset_execution_proven",
    "isaac_sim_execution_proven",
    "robot_readiness_proven",
    "robot_policy_execution_proven",
    "physics_contact_validated",
    "safety_validated",
    "public_claim_upgrade_allowed",
)


def _resolve_owner_proof_artifact(value: Any, *, proof_dir: Path) -> Path | None:
    text = _string(value)
    if not text or text.startswith(("gs://", "http://", "https://")):
        return None
    path = Path(text)
    return path if path.is_absolute() else proof_dir / path


def _read_owner_proof_json_artifact(
    value: Any,
    *,
    proof_dir: Path,
) -> tuple[Dict[str, Any], str | None, bool]:
    path = _resolve_owner_proof_artifact(value, proof_dir=proof_dir)
    if path is None:
        return {}, "owner_proof_artifact_not_local", False
    if not path.is_file():
        return {}, "owner_proof_artifact_missing", False
    try:
        payload = read_json_any(path)
    except Exception:
        return {}, "owner_proof_artifact_invalid_json", True
    if not isinstance(payload, Mapping):
        return {}, "owner_proof_artifact_non_object_json", True
    return dict(payload), None, True


def _owner_proof_file_exists(value: Any, *, proof_dir: Path) -> tuple[bool, str | None]:
    path = _resolve_owner_proof_artifact(value, proof_dir=proof_dir)
    if path is None:
        return False, "owner_proof_artifact_not_local"
    if not path.is_file():
        return False, "owner_proof_artifact_missing"
    return True, None


def _trace_status_ok(payload: Mapping[str, Any], *, true_field: str) -> bool:
    status = _string(payload.get("status")).lower()
    return bool(payload.get(true_field)) or status in {
        "accepted",
        "complete",
        "completed",
        "loaded",
        "passed",
        "ready",
        "succeeded",
        "validated",
    }


def _attestation_ok(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    return bool(
        _string(value.get("attested_by") or value.get("operator_id") or value.get("owner"))
        and _string(
            value.get("attestation")
            or value.get("statement")
            or value.get("accepted_claim_boundary")
        )
    )


def _pass_fail_ok(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    if "passed" in value:
        return bool(value.get("passed"))
    status = _string(value.get("status")).lower()
    return status in {"passed", "accepted", "completed", "succeeded"}


def _default_smoke_policy_ok(payload: Mapping[str, Any]) -> bool:
    return _string(payload.get("policy_kind")) == "walk_to_target" and bool(
        _string(payload.get("target"))
    )


def _robot_asset_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = _string(value)
    return {"name": text} if text else {}


def _owner_robot_asset(*payloads: Mapping[str, Any]) -> Dict[str, Any]:
    for payload in payloads:
        for key in ("robot_asset", "owner_robot_asset", "spawned_robot_asset"):
            asset = _robot_asset_mapping(payload.get(key))
            if asset:
                return asset
    return {}


def _robot_asset_name(asset: Mapping[str, Any]) -> str:
    return _string(asset.get("name") or asset.get("asset_name") or asset.get("robot_name"))


def _robot_asset_path(asset: Mapping[str, Any]) -> str:
    return _string(
        asset.get("uri_or_path")
        or asset.get("usd_path")
        or asset.get("asset_path")
        or asset.get("path")
        or asset.get("uri")
    )


def _normalize_asset_text(value: str) -> str:
    return value.strip().lower().replace("\\", "/")


def _is_unitree_g1_isaac_asset(asset: Mapping[str, Any]) -> bool:
    name = _normalize_asset_text(_robot_asset_name(asset))
    path = _normalize_asset_text(_robot_asset_path(asset))
    source = _normalize_asset_text(_string(asset.get("source") or asset.get("catalog")))
    path_matches = path.endswith("robots/unitree/g1/g1.usd") or path.endswith("g1/g1.usd")
    name_matches = ("unitree" in name and "g1" in name) or name in {"g1", "unitree_g1"}
    source_matches = "isaac" in source or "robots/unitree" in path
    return path_matches and (name_matches or source_matches)


def _is_unitree_g1_mujoco_asset(asset: Mapping[str, Any]) -> bool:
    name = _normalize_asset_text(_robot_asset_name(asset))
    path = _normalize_asset_text(_robot_asset_path(asset))
    source = _normalize_asset_text(_string(asset.get("source") or asset.get("catalog")))
    path_matches = path.endswith("unitree_g1/g1.xml") or path.endswith("g1.xml")
    name_matches = ("unitree" in name and "g1" in name) or name in {"g1", "unitree_g1"}
    source_matches = "mujoco" in source or "menagerie" in source or "unitree_g1" in path
    return path_matches and (name_matches or source_matches)


def _robot_assets_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_path = _normalize_asset_text(_robot_asset_path(left))
    right_path = _normalize_asset_text(_robot_asset_path(right))
    if left_path and right_path:
        return left_path == right_path or left_path.endswith(right_path) or right_path.endswith(left_path)
    left_name = _normalize_asset_text(_robot_asset_name(left))
    right_name = _normalize_asset_text(_robot_asset_name(right))
    return bool(left_name and right_name and left_name == right_name)


def _sim_robot_pov_ok(payload: Mapping[str, Any]) -> bool:
    if _trace_status_ok(payload, true_field="sim_robot_pov_captured"):
        return True
    if _string(payload.get("robot_camera_video_uri") or payload.get("video_uri")):
        return True
    return bool(payload.get("frames") or payload.get("frame_paths") or payload.get("frame_sequence"))


def _owner_required_field_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return True


def validate_owner_gpu_system_proof(
    *,
    proof_path: str | Path,
    capture_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    proof_file = Path(proof_path)
    proof_dir = proof_file.parent
    generated_at = utc_now_iso()
    blockers: List[str] = []
    warnings: List[str] = []
    proof = _read_optional_mapping(proof_file)
    context = resolve_local_capture_context(capture_root) if capture_root is not None else None

    if not proof:
        manifest = {
            "schema_version": OWNER_GPU_PROOF_MANIFEST_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "missing",
            "proof_path": str(proof_file),
            "blockers": ["owner_gpu_simulator_execution_not_run"],
            "missing_inputs": ["gpu_owner_system_proof.json"],
            "owner_gpu_simulator_execution_proven": False,
            "scene_loaded_in_owner_simulator": False,
            "spawn_pose_loaded": False,
            "robot_readiness_proven": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        if output_path:
            write_json(Path(output_path), manifest)
        return manifest

    missing_fields = [
        field for field in OWNER_GPU_REQUIRED_FIELDS if not _owner_required_field_present(proof.get(field))
    ]
    if not _string(proof.get("spawn_pose_validation_uri_or_path") or proof.get("spawn_trace_uri_or_path")):
        missing_fields.append("spawn_pose_validation_uri_or_path")
    if missing_fields:
        blockers.append("owner_gpu_proof_missing_required_fields")

    for field in OWNER_GPU_FORBIDDEN_TRUE_FIELDS:
        if bool(proof.get(field)):
            blockers.append(f"owner_proof_attempted_forbidden_{field}")

    if context is not None:
        if _string(proof.get("scene_id")) and _string(proof.get("scene_id")) != context.scene_id:
            blockers.append("owner_gpu_proof_scene_id_mismatch")
        if _string(proof.get("capture_id")) and _string(proof.get("capture_id")) != context.capture_id:
            blockers.append("owner_gpu_proof_capture_id_mismatch")

    try:
        exit_code = int(proof.get("exit_code"))
    except (TypeError, ValueError):
        exit_code = -1
    if exit_code != 0:
        blockers.append("owner_gpu_simulator_exit_code_nonzero")

    stdout_exists, stdout_reason = _owner_proof_file_exists(
        proof.get("stdout_uri_or_path"),
        proof_dir=proof_dir,
    )
    stderr_exists, stderr_reason = _owner_proof_file_exists(
        proof.get("stderr_uri_or_path"),
        proof_dir=proof_dir,
    )
    if not stdout_exists:
        blockers.append(f"stdout_{stdout_reason}")
    if not stderr_exists:
        blockers.append(f"stderr_{stderr_reason}")

    scene_load, scene_reason, scene_present = _read_owner_proof_json_artifact(
        proof.get("scene_load_trace_uri_or_path"),
        proof_dir=proof_dir,
    )
    spawn_trace, spawn_reason, spawn_present = _read_owner_proof_json_artifact(
        proof.get("spawn_pose_validation_uri_or_path") or proof.get("spawn_trace_uri_or_path"),
        proof_dir=proof_dir,
    )
    action_trace, action_reason, action_present = _read_owner_proof_json_artifact(
        proof.get("action_or_policy_trace_uri_or_path"),
        proof_dir=proof_dir,
    )
    default_policy, default_policy_reason, default_policy_present = _read_owner_proof_json_artifact(
        proof.get("default_smoke_policy_uri_or_path"),
        proof_dir=proof_dir,
    )
    policy_trace, policy_reason, policy_present = _read_owner_proof_json_artifact(
        proof.get("policy_execution_trace_uri_or_path")
        or proof.get("action_or_policy_trace_uri_or_path"),
        proof_dir=proof_dir,
    )
    sim_robot_pov, sim_pov_reason, sim_pov_present = _read_owner_proof_json_artifact(
        proof.get("sim_robot_pov_evidence_uri_or_path"),
        proof_dir=proof_dir,
    )
    artifact_manifest, artifact_reason, artifact_present = _read_owner_proof_json_artifact(
        proof.get("artifact_manifest_uri_or_path"),
        proof_dir=proof_dir,
    )
    for label, reason in (
        ("scene_load_trace", scene_reason),
        ("spawn_trace", spawn_reason),
        ("action_or_policy_trace", action_reason),
        ("default_smoke_policy", default_policy_reason),
        ("policy_execution_trace", policy_reason),
        ("sim_robot_pov_evidence", sim_pov_reason),
        ("artifact_manifest", artifact_reason),
    ):
        if reason:
            blockers.append(f"{label}_{reason}")

    scene_loaded = scene_present and _trace_status_ok(scene_load, true_field="scene_loaded")
    spawn_loaded = spawn_present and _trace_status_ok(spawn_trace, true_field="spawn_pose_loaded")
    action_trace_ok = action_present and (
        _trace_status_ok(action_trace, true_field="policy_trace_loaded")
        or bool(action_trace.get("actions") or action_trace.get("attempts") or action_trace.get("records"))
    )
    default_policy_ok = default_policy_present and _default_smoke_policy_ok(default_policy)
    policy_execution_ok = policy_present and (
        _trace_status_ok(policy_trace, true_field="default_policy_executed")
        or _trace_status_ok(policy_trace, true_field="policy_execution_completed")
        or bool(
            policy_trace.get("actions")
            or policy_trace.get("attempts")
            or policy_trace.get("records")
        )
    )
    sim_robot_pov_ok = sim_pov_present and _sim_robot_pov_ok(sim_robot_pov)
    artifact_manifest_ok = artifact_present and (
        _trace_status_ok(artifact_manifest, true_field="artifact_manifest_complete")
        or bool(artifact_manifest.get("artifacts") or artifact_manifest.get("files"))
    )
    simulator_backend = _string(proof.get("simulator_backend"))
    isaac_robot_asset_required = simulator_backend in ISAAC_SIMULATOR_FRAMEWORKS
    proof_robot_asset = _owner_robot_asset(proof)
    scene_robot_asset = _owner_robot_asset(scene_load)
    spawn_robot_asset = _owner_robot_asset(spawn_trace)
    robot_asset = spawn_robot_asset or proof_robot_asset or scene_robot_asset
    robot_asset_trace_present = bool(spawn_robot_asset or scene_robot_asset)
    robot_asset_matches_proof = (
        not proof_robot_asset
        or not robot_asset
        or _robot_assets_match(proof_robot_asset, robot_asset)
    )
    unitree_g1_asset_spawned = bool(spawn_robot_asset) and _is_unitree_g1_isaac_asset(
        spawn_robot_asset
    )
    mujoco_g1_asset_spawned = bool(spawn_robot_asset) and _is_unitree_g1_mujoco_asset(
        spawn_robot_asset
    )
    isaac_robot_asset_valid = (
        isaac_robot_asset_required
        and bool(proof_robot_asset)
        and bool(spawn_robot_asset)
        and robot_asset_matches_proof
        and unitree_g1_asset_spawned
    )
    mujoco_g1_asset_valid = (
        simulator_backend == "mujoco"
        and bool(proof_robot_asset)
        and bool(spawn_robot_asset)
        and robot_asset_matches_proof
        and mujoco_g1_asset_spawned
    )
    if not scene_loaded:
        blockers.append("owner_gpu_scene_load_trace_not_proven")
    if not spawn_loaded:
        blockers.append("owner_gpu_spawn_trace_not_proven")
    if not action_trace_ok:
        blockers.append("owner_gpu_action_or_policy_trace_not_proven")
    if not default_policy_ok:
        blockers.append("owner_gpu_default_smoke_policy_not_proven")
    if not policy_execution_ok:
        blockers.append("owner_gpu_default_policy_execution_trace_not_proven")
    if not sim_robot_pov_ok:
        blockers.append("owner_gpu_sim_robot_pov_evidence_not_proven")
    if not artifact_manifest_ok:
        blockers.append("owner_gpu_artifact_manifest_not_proven")
    if not _attestation_ok(proof.get("operator_attestation")):
        blockers.append("owner_gpu_operator_attestation_missing_or_incomplete")
    if not _pass_fail_ok(proof.get("pass_fail_criteria")):
        blockers.append("owner_gpu_pass_fail_criteria_not_passed")
    if isaac_robot_asset_required:
        if not proof_robot_asset:
            blockers.append("owner_gpu_proof_missing_isaac_robot_asset")
        if not spawn_robot_asset:
            blockers.append("owner_gpu_spawn_trace_missing_isaac_robot_asset")
        if proof_robot_asset and spawn_robot_asset and not robot_asset_matches_proof:
            blockers.append("owner_gpu_robot_asset_mismatch")
        if spawn_robot_asset and not unitree_g1_asset_spawned:
            blockers.append("owner_gpu_unitree_g1_asset_not_spawned")

    unique_blockers: List[str] = []
    for blocker in blockers:
        if blocker and blocker not in unique_blockers:
            unique_blockers.append(blocker)
    accepted = not unique_blockers
    manifest = {
        "schema_version": OWNER_GPU_PROOF_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "accepted" if accepted else "blocked",
        "proof_path": str(proof_file),
        "owner_system_id": proof.get("owner_system_id"),
        "simulator_backend": proof.get("simulator_backend"),
        "simulator_version": proof.get("simulator_version"),
        "gpu_model": proof.get("gpu_model"),
        "robot_asset": robot_asset or None,
        "expected_isaac_robot_asset": dict(DEFAULT_ISAAC_HUMANOID_ROBOT_ASSET),
        "exit_code": exit_code,
        "blockers": unique_blockers,
        "warnings": warnings,
        "missing_inputs": missing_fields,
        "evidence": {
            "stdout_present": stdout_exists,
            "stderr_present": stderr_exists,
            "scene_load_trace_present": scene_present,
            "scene_loaded_in_owner_simulator": scene_loaded,
            "spawn_trace_present": spawn_present,
            "spawn_pose_loaded": spawn_loaded,
            "action_or_policy_trace_present": action_present,
            "action_or_policy_trace_valid": action_trace_ok,
            "default_smoke_policy_present": default_policy_present,
            "default_smoke_policy_valid": default_policy_ok,
            "policy_execution_trace_present": policy_present,
            "default_policy_execution_trace_valid": policy_execution_ok,
            "sim_robot_pov_evidence_present": sim_pov_present,
            "sim_robot_pov_evidence_valid": sim_robot_pov_ok,
            "artifact_manifest_present": artifact_present,
            "artifact_manifest_valid": artifact_manifest_ok,
            "robot_asset_trace_present": robot_asset_trace_present,
            "robot_asset_matches_proof": robot_asset_matches_proof,
            "isaac_robot_asset_required": isaac_robot_asset_required,
            "unitree_g1_asset_spawned": unitree_g1_asset_spawned,
            "mujoco_g1_asset_spawned": mujoco_g1_asset_spawned,
            "isaac_robot_asset_valid": isaac_robot_asset_valid,
            "mujoco_g1_asset_valid": mujoco_g1_asset_valid,
            "operator_attestation_present": _attestation_ok(proof.get("operator_attestation")),
            "pass_fail_criteria_passed": _pass_fail_ok(proof.get("pass_fail_criteria")),
        },
        "owner_gpu_simulator_execution_proven": accepted,
        "simulator_execution_proven": accepted,
        "isaac_sim_execution_proven": accepted and isaac_robot_asset_valid,
        "isaac_robot_asset_execution_proven": accepted and isaac_robot_asset_valid,
        "unitree_g1_asset_spawned": accepted and unitree_g1_asset_spawned,
        "mujoco_g1_asset_spawned": accepted and mujoco_g1_asset_spawned,
        "mujoco_g1_asset_execution_proven": accepted and mujoco_g1_asset_valid,
        "scene_loaded_in_owner_simulator": accepted and scene_loaded,
        "spawn_pose_loaded": accepted and spawn_loaded,
        "owner_gpu_default_policy_execution_proven": (
            accepted and default_policy_ok and policy_execution_ok
        ),
        "default_sim_policy_execution_proven": (
            accepted and default_policy_ok and policy_execution_ok
        ),
        "owner_gpu_sim_robot_pov_evidence_proven": accepted and sim_robot_pov_ok,
        "real_robot_pov_evidence_proven": False,
        "robot_policy_execution_proven": False,
        "physics_contact_validated": False,
        "safety_validated": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulator_execution_proven": accepted,
            "owner_gpu_default_policy_execution_proven": (
                accepted and default_policy_ok and policy_execution_ok
            ),
            "default_sim_policy_execution_proven": (
                accepted and default_policy_ok and policy_execution_ok
            ),
            "owner_gpu_sim_robot_pov_evidence_proven": accepted and sim_robot_pov_ok,
            "isaac_sim_execution_proven": accepted and isaac_robot_asset_valid,
            "isaac_robot_asset_execution_proven": accepted and isaac_robot_asset_valid,
            "mujoco_g1_asset_execution_proven": accepted and mujoco_g1_asset_valid,
            "real_robot_pov_evidence_proven": False,
        },
    }
    if output_path:
        write_json(Path(output_path), manifest)
    return manifest


def _worldlabs_summary(world_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    assets = _mapping(world_manifest.get("assets"))
    splats = _mapping(assets.get("splats"))
    mesh = _mapping(assets.get("mesh"))
    semantics = _mapping(
        splats.get("semantics_metadata")
        or assets.get("semantics_metadata")
        or world_manifest.get("semantics_metadata")
    )
    manifest_present = bool(world_manifest)
    return {
        "status": "present" if manifest_present else "optional_missing",
        "artifact_present": manifest_present,
        "world_id": _string(world_manifest.get("world_id") or world_manifest.get("id")) or None,
        "world_marble_url": _string(world_manifest.get("world_marble_url")) or None,
        "model": _string(world_manifest.get("model")) or None,
        "spz_available": bool(_mapping(splats.get("spz_urls")) or splats.get("spz_url")),
        "ply_available": bool(_mapping(splats.get("ply_urls")) or splats.get("ply_url")),
        "usd_available": bool(_mapping(splats.get("usd_urls")) or splats.get("usd_url")),
        "collider_mesh_glb_url": _string(
            mesh.get("collider_mesh_url")
            or mesh.get("collider_mesh_glb_url")
            or assets.get("collider_mesh_url")
        )
        or None,
        "metric_scale_factor": semantics.get("metric_scale_factor"),
        "ground_plane_offset": semantics.get("ground_plane_offset"),
    }


def _build_plan(
    *,
    context: Any,
    automation_dir: Path,
    pipeline_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    world_manifest = _read_optional_mapping(pipeline_dir / "worldlabs_world_manifest.json")
    marble_bridge = _read_optional_mapping(
        pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json"
    )
    marble_validation = _read_optional_mapping(
        pipeline_dir / "marble_sim_assets" / "marble_asset_validation.json"
    )
    simready_scene = _read_optional_mapping(pipeline_dir / "simready" / "simready_scene_manifest.json")
    simready_validation = _read_optional_mapping(pipeline_dir / "simready" / "simready_validation.json")
    cosmos_export = _read_optional_mapping(pipeline_dir / "cosmos_training_export" / "manifest.json")
    cpu_preflight_scorecard = _read_optional_mapping(
        pipeline_dir / "simulation_automation" / "cpu_preflight_scorecard.json"
    )
    worldlabs = _worldlabs_summary(world_manifest)
    plan = {
        "schema_version": SIMULATION_AUTOMATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "planned",
        "source_artifacts": _source_artifacts(
            automation_dir=automation_dir,
            pipeline_dir=pipeline_dir,
        ),
        "world_model_sources": {
            "worldlabs": worldlabs,
            "marble": {
                "status": "present" if marble_bridge else "optional_missing",
                "artifact_present": bool(marble_bridge),
                "bridge_status": _string(marble_bridge.get("status")) or None,
                "validation_status": _string(marble_validation.get("overall_status")) or None,
                "physics_collision_review_ready": bool(
                    marble_validation.get("physics_collision_review_ready")
                ),
                "robot_readiness_proven": bool(marble_validation.get("robot_readiness_proven")),
            },
            "simready": {
                "scene_status": _string(simready_scene.get("status")) or None,
                "validation_status": _string(simready_validation.get("overall_status")) or None,
                "simulator_execution_proven": bool(
                    _mapping(simready_validation.get("claim_boundary")).get(
                        "simulator_execution_proven"
                    )
                ),
                "robot_readiness_proven": bool(
                    _mapping(simready_validation.get("claim_boundary")).get("robot_readiness_proven")
                ),
            },
            "cpu_preflight": {
                "scorecard_status": _string(cpu_preflight_scorecard.get("status")) or None,
                "isaac_usd_import_candidate": bool(
                    cpu_preflight_scorecard.get("isaac_usd_import_candidate")
                ),
                "isaac_usd_collision_verified": bool(
                    cpu_preflight_scorecard.get("isaac_usd_collision_verified")
                ),
                "isaac_usd_collision_unverified": bool(
                    cpu_preflight_scorecard.get("isaac_usd_collision_unverified")
                ),
                "portable_collider_glb_missing": bool(
                    cpu_preflight_scorecard.get("portable_collider_glb_missing")
                ),
                "cpu_proxy_collision_estimated": bool(
                    cpu_preflight_scorecard.get("cpu_proxy_collision_estimated")
                ),
                "simulator_execution_not_run": True,
            },
        },
        "automation_scope": {
            "plans_asset_conversion": True,
            "plans_scene_asset_preflight": True,
            "plans_episode_spec": True,
            "plans_cpu_simulator_preflight": True,
            "plans_simulator_execution": True,
            "plans_training": True,
            "plans_eval_and_proof_collection": True,
            "live_provider_calls_allowed": False,
            "remote_asset_downloads_allowed_by_default": False,
            "cpu_simulator_preflight_allowed_by_default": False,
            "simulator_execution_allowed_by_default": False,
            "gpu_training_allowed_by_default": False,
        },
        "training_sources": {
            "cosmos_export_status": _string(cosmos_export.get("status")) or "missing",
            "cosmos_export_manifest": (
                _relative_to(automation_dir, pipeline_dir / "cosmos_training_export" / "manifest.json")
                if (pipeline_dir / "cosmos_training_export" / "manifest.json").is_file()
                else None
            ),
            "runner": TRAINING_RUNNER,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    plan["deterministic_fingerprint"] = _sha_payload(
        {
            "scene_id": plan["scene_id"],
            "capture_id": plan["capture_id"],
            "source_artifacts": plan["source_artifacts"],
            "world_model_sources": plan["world_model_sources"],
            "training_sources": plan["training_sources"],
        }
    )
    return plan


def _conversion_status(
    *,
    framework: str,
    worldlabs: Mapping[str, Any],
    cpu_preflight: Mapping[str, Any],
) -> Dict[str, Any]:
    collider = _string(worldlabs.get("collider_mesh_glb_url"))
    has_visual = bool(
        worldlabs.get("ply_available")
        or worldlabs.get("usd_available")
        or worldlabs.get("spz_available")
        or cpu_preflight.get("isaac_usd_import_candidate")
        or cpu_preflight.get("cpu_proxy_collision_estimated")
    )
    if framework == "isaac_sim":
        if not has_visual:
            status = "blocked"
            blockers = ["missing_visual_asset"]
        elif worldlabs.get("usd_available") or cpu_preflight.get("isaac_usd_import_candidate"):
            status = "isaac_usd_import_candidate"
            blockers = (
                []
                if cpu_preflight.get("isaac_usd_collision_verified")
                else ["isaac_usd_collision_unverified"]
            )
        elif worldlabs.get("ply_available"):
            status = "planned_asset_import_ready"
            blockers = ["cpu_proxy_collision_estimated"]
        else:
            status = "planned_requires_conversion"
            blockers = ["spz_to_ply_or_usd_required"]
        output_format = "OpenUSD_USD_or_USDA"
        recommended_steps = [
            "resolve explicit Marble export or local conversion manifest",
            "convert SPZ/PLY/GLB into USD review scene as needed",
            "import into Isaac Sim headless only after explicit approval",
        ]
    elif framework == "isaac_lab_arena":
        if not has_visual:
            status = "blocked"
            blockers = ["missing_visual_asset"]
        elif worldlabs.get("usd_available") or cpu_preflight.get("isaac_usd_import_candidate"):
            status = "arena_environment_packet_ready_for_owner_review"
            blockers = (
                []
                if cpu_preflight.get("isaac_usd_collision_verified")
                else ["isaac_usd_collision_unverified"]
            )
        elif worldlabs.get("ply_available") or collider:
            status = "planned_requires_owner_asset_mapping"
            blockers = ["arena_scene_asset_mapping_required"]
        else:
            status = "planned_requires_conversion"
            blockers = ["spz_to_usd_or_arena_scene_asset_required"]
        output_format = "Isaac_Lab_Arena_scene_embodiment_task_package"
        recommended_steps = [
            "review arena_environment_packet.json for scene, embodiment, task, metrics, and episode bindings",
            "map Blueprint scene assets into Isaac Lab-Arena Scene primitives on the owner system",
            "map robot-team assets into Isaac Lab-Arena Embodiment primitives without altering capture truth",
            "run Isaac Lab-Arena only after explicit approval and owner-system proof capture",
        ]
    elif framework == "mujoco":
        status = "planned_requires_conversion" if collider else "blocked"
        blockers = [] if collider else ["portable_collider_glb_missing"]
        if not collider and cpu_preflight.get("cpu_proxy_collision_estimated"):
            blockers.append("cpu_proxy_collision_estimated")
        output_format = "MJCF_XML"
        recommended_steps = [
            "convert collider GLB to MuJoCo mesh/MJCF",
            "attach robot-owned MJCF assets outside Blueprint capture truth",
            "compile/load with MuJoCo only after explicit approval",
        ]
    elif framework == "pybullet":
        status = "planned_requires_conversion" if collider else "blocked"
        blockers = [] if collider else ["portable_collider_glb_missing"]
        if not collider and cpu_preflight.get("cpu_proxy_collision_estimated"):
            blockers.append("cpu_proxy_collision_estimated")
        output_format = "URDF_or_SDF"
        recommended_steps = [
            "convert collider GLB to URDF/SDF collision assets",
            "attach robot-owned URDF assets outside Blueprint capture truth",
            "load through PyBullet DIRECT only after explicit approval",
        ]
    else:
        status = "planned_requires_conversion" if collider else "blocked"
        blockers = [] if collider else ["portable_collider_glb_missing"]
        output_format = "OpenUSD_or_Newton_scene_manifest"
        recommended_steps = [
            "keep Newton as a replaceable backend through Isaac Lab/Newton or native Newton manifests",
            "convert collider/scene assets to OpenUSD-compatible review assets",
            "run Newton/MuJoCo-Warp only after explicit approval and GPU/runtime proof",
        ]
    return {
        "framework": framework,
        "status": status,
        "blockers": blockers,
        "input_assets": {
            "collider_mesh_glb_url": collider or None,
            "portable_collider_glb_present": bool(collider),
            "spz_available": bool(worldlabs.get("spz_available")),
            "ply_available": bool(worldlabs.get("ply_available")),
            "usd_available": bool(worldlabs.get("usd_available")),
            "cpu_proxy_collision_estimated": bool(
                cpu_preflight.get("cpu_proxy_collision_estimated")
            ),
        },
        "target_format": output_format,
        "conversion_executed": False,
        "remote_asset_downloads_performed": False,
        "recommended_steps": recommended_steps,
    }


def _build_asset_conversion_plan(*, plan: Mapping[str, Any], generated_at: str) -> Dict[str, Any]:
    worldlabs = _mapping(_mapping(plan.get("world_model_sources")).get("worldlabs"))
    cpu_preflight = _mapping(_mapping(plan.get("world_model_sources")).get("cpu_preflight"))
    frameworks = {
        framework: _conversion_status(
            framework=framework,
            worldlabs=worldlabs,
            cpu_preflight=cpu_preflight,
        )
        for framework in SIMULATOR_FRAMEWORKS
    }
    blockers = [
        f"{framework}:{blocker}"
        for framework, payload in frameworks.items()
        for blocker in payload.get("blockers", [])
    ]
    return {
        "schema_version": ASSET_CONVERSION_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": plan.get("scene_id"),
        "capture_id": plan.get("capture_id"),
        "status": "blocked" if blockers else "planned",
        "blockers": blockers,
        "frameworks": frameworks,
        "download_policy": {
            "remote_asset_downloads_allowed": False,
            "remote_asset_downloads_performed": False,
            "asset_conversion_execution_allowed_by_default": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _cards_from_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cards = payload.get("cards")
    if isinstance(cards, list):
        return [dict(item) for item in cards if isinstance(item, Mapping)]
    if isinstance(payload, list):  # type: ignore[unreachable]
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _task_card_index(cards: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for index, card in enumerate(cards):
        task_id = _string(card.get("task_id") or card.get("id") or f"task_{index + 1}")
        if task_id:
            out[task_id] = dict(card)
    return out


def _scenario_card_index(cards: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for index, card in enumerate(cards):
        scenario_id = _string(card.get("scenario_id") or card.get("id") or f"scenario_{index + 1}")
        if scenario_id:
            out[scenario_id] = dict(card)
    return out


def _arena_scene_components(
    *,
    context: Any,
    site_card: Mapping[str, Any],
    inventory: Mapping[str, Any],
    dependency_audit: Mapping[str, Any],
    conversion_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    assets = [dict(item) for item in inventory.get("assets") or [] if isinstance(item, Mapping)]
    return {
        "scene_component_id": f"blueprint_scene_{context.scene_id}",
        "arena_primitive": "Scene",
        "source_site_card": "robot_eval_dataset/site_card.json",
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_id": _string(site_card.get("site_id")) or None,
        "site_type": _string(site_card.get("site_type")) or None,
        "layout_truth_source": "Blueprint raw capture and downstream Site Card",
        "asset_inputs": assets,
        "dependency_summary": {
            "missing_local_file_count": dependency_audit.get("missing_local_file_count", 0),
            "hard_missing_local_file_count": dependency_audit.get(
                "hard_missing_local_file_count",
                dependency_audit.get("missing_local_file_count") or 0,
            ),
            "remote_ref_count": dependency_audit.get("remote_ref_count", 0),
            "unresolved_ref_count": dependency_audit.get("unresolved_ref_count", 0),
        },
        "conversion_status": _mapping(
            _mapping(conversion_plan.get("frameworks")).get("isaac_lab_arena")
        ).get("status"),
        "capture_truth_authority": "raw_capture_not_arena_generated_scene",
    }


def _arena_task_components(
    *,
    task_cards: Sequence[Mapping[str, Any]],
    episode_spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    by_task = _task_card_index(task_cards)
    task_ids = list(by_task)
    for episode in episode_spec.get("episodes") or []:
        if not isinstance(episode, Mapping):
            continue
        task_id = _string(episode.get("task_id"))
        if task_id and task_id not in task_ids:
            task_ids.append(task_id)
    components: List[Dict[str, Any]] = []
    for task_id in task_ids:
        card = by_task.get(task_id, {})
        components.append(
            {
                "task_component_id": f"arena_task_{task_id}",
                "arena_primitive": "Task",
                "task_id": task_id,
                "objective": _string(
                    card.get("task_statement")
                    or card.get("task_text")
                    or card.get("name")
                    or task_id
                ),
                "task_category": _string(card.get("task_category") or "scene_review"),
                "required_metrics": _string_list(card.get("required_metrics")),
                "success_criteria": card.get("success_criteria")
                or card.get("thresholds")
                or "owner_eval_card_thresholds_required",
                "source": "robot_eval_dataset/task_cards.json" if card else "episode_spec.v1.json",
                "review_required": True,
                "claim_boundary": "arena_task_component_defines_eval_scope_not_execution_proof",
            }
        )
    return components


def _arena_scenario_components(
    *,
    scenario_cards: Sequence[Mapping[str, Any]],
    episode_spec: Mapping[str, Any],
    variation_instances: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    by_scenario = _scenario_card_index(scenario_cards)
    scenario_ids = list(by_scenario)
    for episode in episode_spec.get("episodes") or []:
        if not isinstance(episode, Mapping):
            continue
        scenario_id = _string(episode.get("scenario_id"))
        if scenario_id and scenario_id not in scenario_ids:
            scenario_ids.append(scenario_id)
    variation_ids_by_scenario = _variation_instance_ids_by_scenario(variation_instances)
    components: List[Dict[str, Any]] = []
    for scenario_id in scenario_ids:
        card = by_scenario.get(scenario_id, {})
        labels = _mapping(card.get("observed_vs_inferred_labels"))
        variation_ids = variation_ids_by_scenario.get(scenario_id, [])
        components.append(
            {
                "scenario_component_id": f"arena_scenario_{scenario_id}",
                "scenario_id": scenario_id,
                "task_id": _string(card.get("task_id")) or None,
                "robot_profile_id": _string(card.get("robot_profile_id")) or None,
                "normal_scenario": card.get("normal_scenario"),
                "variation": card.get("variation"),
                "edge_case": card.get("edge_case"),
                "observed_vs_inferred_labels": labels,
                "scenario_variation_instance_ids": variation_ids,
                "scenario_variation_count": len(variation_ids),
                "engine_mutation_plan_path": (
                    "scenario_variation_instances.json" if variation_ids else None
                ),
                "review_required": True,
                "source": (
                    "robot_eval_dataset/scenario_cards.json"
                    if card
                    else "episode_spec.v1.json"
                ),
                "claim_boundary": "arena_scenario_component_is_review_scope_not_simulator_result",
            }
        )
    return components


def _variation_instance_ids_by_scenario(
    variation_instances: Mapping[str, Any],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    instances = variation_instances.get("instances")
    if not isinstance(instances, list):
        return out
    for instance in instances:
        if not isinstance(instance, Mapping):
            continue
        scenario_id = _string(instance.get("scenario_id"))
        instance_id = _string(instance.get("instance_id"))
        if not scenario_id or not instance_id:
            continue
        out.setdefault(scenario_id, [])
        if instance_id not in out[scenario_id]:
            out[scenario_id].append(instance_id)
    return out


def _arena_eval_bindings(eval_cards: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    bindings: List[Dict[str, Any]] = []
    for index, card in enumerate(eval_cards):
        eval_id = _string(card.get("eval_card_id") or card.get("id") or f"eval_card_{index + 1}")
        bindings.append(
            {
                "eval_binding_id": f"arena_eval_binding_{eval_id}",
                "eval_card_id": eval_id,
                "scenario_id": _string(card.get("scenario_id")) or None,
                "task_id": _string(card.get("task_id")) or None,
                "prediction_source": _string(card.get("prediction_source")) or None,
                "metrics": card.get("metrics") or card.get("required_metrics") or [],
                "validation": card.get("validation") or {},
                "blocked_upgrades": _string_list(card.get("blocked_upgrades")),
                "proof_boundary": card.get("proof_boundary"),
                "source": "robot_eval_dataset/eval_cards.json",
            }
        )
    return bindings


def _arena_embodiment_components(episode_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for episode in episode_spec.get("episodes") or []:
        if not isinstance(episode, Mapping):
            continue
        profile = _mapping(episode.get("robot_profile"))
        profile_id = _string(episode.get("robot_profile_id") or profile.get("robot_profile_id"))
        if not profile_id or profile_id in seen:
            continue
        seen.add(profile_id)
        out.append(
            {
                "embodiment_component_id": f"arena_embodiment_{profile_id}",
                "arena_primitive": "Embodiment",
                "robot_profile_id": profile_id,
                "embodiment": _string(profile.get("embodiment")) or None,
                "base_type": _string(profile.get("base_type")) or None,
                "sensors": _string_list(profile.get("sensors")),
                "source": profile.get("source") or "episode_spec.v1.json",
                "owner_robot_asset_mapping_required": True,
                "claim_boundary": "embodiment_reference_only_not_robot_policy_or_asset_proof",
            }
        )
    return out


def _arena_episode_bindings(
    *,
    context: Any,
    episode_spec: Mapping[str, Any],
    task_components: Sequence[Mapping[str, Any]],
    scenario_components: Sequence[Mapping[str, Any]],
    variation_instances: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    task_ids = {_string(item.get("task_id")) for item in task_components}
    scenario_ids = {_string(item.get("scenario_id")) for item in scenario_components}
    variation_ids_by_scenario = _variation_instance_ids_by_scenario(variation_instances)
    bindings: List[Dict[str, Any]] = []
    for episode in episode_spec.get("episodes") or []:
        if not isinstance(episode, Mapping):
            continue
        episode_id = _string(episode.get("episode_id"))
        task_id = _string(episode.get("task_id"))
        scenario_id = _string(episode.get("scenario_id"))
        robot_profile_id = _string(episode.get("robot_profile_id"))
        missing = _string_list(episode.get("missing_proof_labels"))
        if task_id not in task_ids:
            missing.append("arena_task_component_missing")
        if scenario_id not in scenario_ids:
            missing.append("arena_scenario_component_missing")
        scenario_variation_ids = variation_ids_by_scenario.get(scenario_id, [])
        bindings.append(
            {
                "arena_environment_name": f"blueprint_{context.scene_id}_{episode_id}",
                "episode_id": episode_id,
                "scene_component_id": f"blueprint_scene_{context.scene_id}",
                "task_component_id": f"arena_task_{task_id}",
                "scenario_component_id": f"arena_scenario_{scenario_id}",
                "embodiment_component_id": f"arena_embodiment_{robot_profile_id}",
                "robot_spawn_pose": episode.get("robot_spawn_pose") or {},
                "camera_pose": episode.get("camera_pose") or {},
                "allowed_motion_region": episode.get("allowed_motion_region") or {},
                "target_region": episode.get("target_region") or {},
                "reset_conditions": episode.get("reset_conditions") or [],
                "scenario_variation_instance_ids": scenario_variation_ids,
                "scenario_variation_count": len(scenario_variation_ids),
                "engine_mutation_plan_path": (
                    "scenario_variation_instances.json" if scenario_variation_ids else None
                ),
                "arena_builder_target": "IsaacLabArenaEnvironment",
                "manager_based_rl_env_cfg_required": True,
                "review_required": True,
                "missing_proof_labels": _string_list(missing),
                "proof_booleans": {
                    "simulator_execution_proven": False,
                    "robot_readiness_proven": False,
                    "physics_contact_validated": False,
                    "safety_validated": False,
                },
            }
        )
    return bindings


def _build_arena_environment_packet(
    *,
    context: Any,
    automation_dir: Path,
    pipeline_dir: Path,
    conversion_plan: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    site_card = _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "site_card.json")
    task_cards = _cards_from_payload(
        _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "task_cards.json")
    )
    scenario_cards = _cards_from_payload(
        _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "scenario_cards.json")
    )
    eval_cards = _cards_from_payload(
        _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "eval_cards.json")
    )
    proof_boundaries = _read_optional_mapping(
        pipeline_dir / "robot_eval_dataset" / "proof_boundaries.json"
    )
    episode_spec = _read_optional_mapping(automation_dir / "episode_spec.v1.json")
    variation_instances = _read_optional_mapping(automation_dir / "scenario_variation_instances.json")
    inventory = _read_optional_mapping(automation_dir / "scene_asset_inventory.json")
    dependency_audit = _read_optional_mapping(automation_dir / "scene_asset_dependency_audit.json")

    task_components = _arena_task_components(task_cards=task_cards, episode_spec=episode_spec)
    scenario_components = _arena_scenario_components(
        scenario_cards=scenario_cards,
        episode_spec=episode_spec,
        variation_instances=variation_instances,
    )
    embodiment_components = _arena_embodiment_components(episode_spec)
    eval_bindings = _arena_eval_bindings(eval_cards)
    episode_bindings = _arena_episode_bindings(
        context=context,
        episode_spec=episode_spec,
        task_components=task_components,
        scenario_components=scenario_components,
        variation_instances=variation_instances,
    )
    source_blockers: List[str] = []
    if not site_card:
        source_blockers.append("robot_eval_site_card_missing")
    if not task_cards:
        source_blockers.append("robot_eval_task_cards_missing")
    if not scenario_cards:
        source_blockers.append("robot_eval_scenario_cards_missing")
    if not eval_cards:
        source_blockers.append("robot_eval_eval_cards_missing")
    if not episode_bindings:
        source_blockers.append("episode_spec_v1_missing_or_empty")
    if not variation_instances:
        source_blockers.append("scenario_variation_instances_missing")

    blocker_set = _string_list(source_blockers)
    for episode in episode_bindings:
        blocker_set.extend(_string_list(episode.get("missing_proof_labels")))
    blocker_set.extend(
        [
            "owner_gpu_simulator_execution_not_run",
            "isaac_lab_arena_owner_install_not_verified",
            "owner_robot_asset_mapping_required",
        ]
    )
    blockers = _string_list(blocker_set)
    packet = {
        "schema_version": ARENA_ENVIRONMENT_PACKET_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "ready_for_owner_arena_pack_review"
        if episode_bindings
        else "blocked_missing_episode_spec",
        "backend": "isaac_lab_arena",
        "compatibility_target": {
            "framework": "NVIDIA Isaac Lab-Arena",
            "environment_builder": "ArenaEnvBuilder",
            "runtime_config_target": "ManagerBasedRLEnvCfg",
            "package_role": "Blueprint Arena Pack review input",
        },
        "source_artifacts": {
            "site_card": "../robot_eval_dataset/site_card.json" if site_card else None,
            "task_cards": "../robot_eval_dataset/task_cards.json" if task_cards else None,
            "scenario_cards": "../robot_eval_dataset/scenario_cards.json"
            if scenario_cards
            else None,
            "eval_cards": "../robot_eval_dataset/eval_cards.json" if eval_cards else None,
            "proof_boundaries": "../robot_eval_dataset/proof_boundaries.json"
            if proof_boundaries
            else None,
            "episode_spec": "episode_spec.v1.json" if episode_spec else None,
            "scenario_variation_instances": "scenario_variation_instances.json"
            if variation_instances
            else None,
            "scene_asset_inventory": "scene_asset_inventory.json" if inventory else None,
            "scene_asset_dependency_audit": "scene_asset_dependency_audit.json"
            if dependency_audit
            else None,
            "asset_conversion_plan": "asset_conversion_plan.json",
        },
        "arena_components": {
            "scene": _arena_scene_components(
                context=context,
                site_card=site_card,
                inventory=inventory,
                dependency_audit=dependency_audit,
                conversion_plan=conversion_plan,
            ),
            "embodiments": embodiment_components,
            "tasks": task_components,
            "scenarios": scenario_components,
            "scenario_variation_instances": {
                "path": "scenario_variation_instances.json" if variation_instances else None,
                "status": variation_instances.get("status") if variation_instances else "missing",
                "instance_count": variation_instances.get("instance_count", 0)
                if variation_instances
                else 0,
                "engine_targets": variation_instances.get("engine_targets", [])
                if variation_instances
                else [],
                "engine_mutation_plan_available": bool(
                    _mapping(variation_instances.get("engine_mutation_plan"))
                )
                if variation_instances
                else False,
            },
            "eval_bindings": eval_bindings,
            "episode_bindings": episode_bindings,
        },
        "package_layout": {
            "suggested_root": "blueprint_arena_pack/",
            "suggested_files": [
                "scene.py",
                "embodiments.py",
                "tasks.py",
                "metrics.py",
                "episodes.yaml",
                "README.md",
                "proof_boundary.json",
            ],
            "export_executed": False,
            "remote_asset_downloads_performed": False,
        },
        "execution_policy": {
            "simulator_execution_allowed_by_default": False,
            "required_env_gate": "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true",
            "required_cli_gate": "--allow-simulator isaac_lab_arena",
            "required_command_gate": "--simulator-command isaac_lab_arena=<owner command>",
            "owner_gpu_proof_required": True,
        },
        "blockers": blockers,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "physics_contact_validated": False,
        "safety_validated": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    packet["deterministic_fingerprint"] = _sha_payload(
        {
            "scene_id": packet["scene_id"],
            "capture_id": packet["capture_id"],
            "arena_components": packet["arena_components"],
            "source_artifacts": packet["source_artifacts"],
        }
    )
    write_json(automation_dir / "arena_environment_packet.json", packet)
    return packet


def _plugin_runtime_kind(framework: str) -> str:
    if framework in {"isaac_sim", "isaac_lab_arena", "newton"}:
        return "gpu_world_sim_runtime"
    return "cpu_or_gpu_physics_runtime"


def _plugin_provider_family(framework: str) -> str:
    if framework == "isaac_lab_arena":
        return "isaac_lab_arena"
    if framework == "isaac_sim":
        return "isaac_sim"
    if framework == "newton":
        return "newton_or_warp_physics"
    return "cpu_physics_engine"


def _world_model_plugin_provider_family(engine: str) -> str:
    return {
        "worldlabs_world_model": "worldlabs",
        "marble_simready": "marble",
        "cosmos_predict": "nvidia_cosmos",
        "native_site_reference": "blueprint_native_site_reference",
    }.get(engine, "replaceable_world_model_engine")


def _world_model_plugin_inputs(engine: str) -> Dict[str, str | None]:
    common_inputs: Dict[str, str | None] = {
        "simulation_automation_plan": "simulation_automation_plan.json",
        "scenario_variation_instances": "scenario_variation_instances.json",
        "site_card": "../robot_eval_dataset/site_card.json",
        "task_cards": "../robot_eval_dataset/task_cards.json",
        "scenario_cards": "../robot_eval_dataset/scenario_cards.json",
    }
    if engine == "worldlabs_world_model":
        return {
            **common_inputs,
            "world_manifest": "../worldlabs_world_manifest.json",
        }
    if engine == "marble_simready":
        return {
            **common_inputs,
            "simready_bridge": "../marble_sim_assets/marble_simready_bridge.json",
        }
    if engine == "cosmos_predict":
        return {
            **common_inputs,
            "gpu_handoff_packet": "gpu_handoff_packet.json",
            "dense_world_model_export": "../world_model_export/dense_index.jsonl",
        }
    return {
        **common_inputs,
        "site_reference_projection": "../sites/site_reference_summary_projection.json",
    }


def _build_world_model_engine_plugins(
    *,
    plan: Mapping[str, Any],
    scenario_variation_instances: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    plugins: Dict[str, Dict[str, Any]] = {}
    world_sources = _mapping(plan.get("world_model_sources"))
    for engine in WORLD_MODEL_ENGINE_TARGETS:
        source_key = {
            "worldlabs_world_model": "worldlabs",
            "marble_simready": "marble",
            "cosmos_predict": "cosmos",
            "native_site_reference": "site_reference",
        }.get(engine, engine)
        source = _mapping(world_sources.get(source_key))
        plugins[engine] = {
            "plugin_id": f"blueprint_{engine}_engine_plugin",
            "engine": engine,
            "provider_family": _world_model_plugin_provider_family(engine),
            "runtime_kind": "world_model_support_engine",
            "adapter_contract_status": "ready",
            "managed_execution_supported": True,
            "inputs": _world_model_plugin_inputs(engine),
            "outputs_expected": {
                "scenario_support_assets": f"world_model_plugins/{engine}/scenario_support_assets.json",
                "uncertainty_summary": f"world_model_plugins/{engine}/world_model_uncertainty.json",
                "engine_adapter_manifest": f"world_model_plugins/{engine}/adapter_manifest.json",
            },
            "source_status": source.get("status") or ("present" if source else "optional_missing"),
            "source_artifact": source.get("path") or source.get("uri"),
            "scenario_variation_instance_count": scenario_variation_instances.get(
                "instance_count",
                0,
            ),
            "normalization_contract": {
                "input_formats": [
                    "capture_grounded_reference_media",
                    "scenario_variation_instances",
                    "engine_specific_support_assets",
                ],
                "output_format": "blueprint_world_model_support_assets.v1",
                "world_model_uncertainty_required": True,
            },
            "proof_boundary": {
                **dict(CLAIM_BOUNDARY),
                "world_model_support_assets_generated": False,
                "world_model_uncertainty_scored": False,
                "simulator_execution_proven": False,
                "robot_policy_execution_proven": False,
                "robot_readiness_proven": False,
            },
        }
    return plugins


def _simulator_adapter_smoke_contract(framework: str) -> Dict[str, Any]:
    return {
        "schema_version": "simulator_adapter_smoke_contract.v1",
        "framework": framework,
        "smoke_runner": f"{framework}_owner_command_probe",
        "required_env": {
            "BLUEPRINT_CAPTURE_ROOT": "absolute capture root under owner runtime",
            "BLUEPRINT_SIMULATOR_FRAMEWORK": framework,
            "BLUEPRINT_SCENARIO_EVAL_MATRIX": "scenario_eval_matrix.json",
            "BLUEPRINT_SIMULATOR_OUTPUT": "path to simulator attempts JSON output",
        },
        "required_output_fields": [
            "scenario_eval_run_id",
            "scenario_variation_instance_id",
            "task_id",
            "scenario_id",
            "success",
            "metrics",
        ],
        "accepted_output_shapes": [
            "attempts[]",
            "records[]",
            "outcomes[]",
            "single attempt object",
        ],
        "smoke_output_acceptance": {
            "normalized_attempt_trace_required": True,
            "failure_labels_required": True,
            "prediction_ledger_required": True,
        },
        "proof_boundary": dict(CLAIM_BOUNDARY),
    }


def _build_simulator_engine_plugin_registry(
    *,
    context: Any,
    automation_dir: Path,
    plan: Mapping[str, Any],
    conversion_plan: Mapping[str, Any],
    scenario_variation_instances: Mapping[str, Any],
    allow_simulator_execution: bool,
    allowed_simulators: Sequence[str],
    simulator_commands: Mapping[str, str],
    generated_at: str,
) -> Dict[str, Any]:
    env_allows = _env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
    allowed = {item for item in allowed_simulators if item in SIMULATOR_FRAMEWORKS}
    plugins: Dict[str, Dict[str, Any]] = {}
    for framework in SIMULATOR_FRAMEWORKS:
        command_text = _string(simulator_commands.get(framework))
        command = shlex.split(command_text) if command_text else []
        missing_gates: List[str] = []
        if not env_allows:
            missing_gates.append("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true")
        if not allow_simulator_execution:
            missing_gates.append("--allow-simulator-execution")
        if framework not in allowed:
            missing_gates.append(f"--allow-simulator {framework}")
        if not command:
            missing_gates.append(f"--simulator-command {framework}=<owner command>")
        conversion = _mapping(_mapping(conversion_plan.get("frameworks")).get(framework))
        plugins[framework] = {
            "plugin_id": f"blueprint_{framework}_sim_engine_plugin",
            "framework": framework,
            "provider_family": _plugin_provider_family(framework),
            "runtime_kind": _plugin_runtime_kind(framework),
            "adapter_contract_status": "ready",
            "managed_execution_supported": True,
            "inputs": {
                "simulation_automation_plan": "simulation_automation_plan.json",
                "asset_conversion_plan": "asset_conversion_plan.json",
                "arena_environment_packet": "arena_environment_packet.json"
                if framework == "isaac_lab_arena"
                else None,
                "scenario_variation_instances": "scenario_variation_instances.json",
                "episode_spec": "episode_spec.v1.json",
                "cpu_preflight_manifest": "cpu_simulator_preflight_manifest.json",
            },
            "outputs_expected": {
                "simulator_request": f"simulators/{framework}_request.json",
                "simulator_result": f"simulators/{framework}_result.json",
                "normalized_attempt_trace": "normalized_attempt_trace.json",
                "robot_pov_observation_manifest": "robot_pov_observation_manifest.json",
                "robot_pov_frame_sequence_manifest": "robot_pov_frame_sequence_manifest.json",
                "policy_execution_trace": "policy_execution_trace.json",
                "owner_gpu_proof": "gpu_owner_system_proof.json",
            },
            "conversion_status": conversion.get("status"),
            "scenario_variation_instance_count": scenario_variation_instances.get(
                "instance_count",
                0,
            ),
            "scenario_variation_engine_mutation_plan_status": _mapping(
                _mapping(scenario_variation_instances.get("engine_mutation_plan")).get(framework)
            ).get("status"),
            "execution_manager": {
                "status": "ready_to_run_command" if not missing_gates else "gated_waiting_for_owner_runtime",
                "env_gate_allows": env_allows,
                "allow_simulator_execution_flag": bool(allow_simulator_execution),
                "simulator_allowlisted": framework in allowed,
                "command_configured": bool(command),
                "command": command,
                "command_sha256": sha256(command_text.encode("utf-8")).hexdigest()
                if command_text
                else None,
                "missing_gates": missing_gates,
                "timeout_seconds_default": 120,
            },
            "normalization_contract": {
                "input_formats": [
                    "simulator_attempts_json",
                    "engine_stdout_stderr",
                    "owner_artifact_manifest",
                ],
                "output_format": "blueprint_normalized_attempt_trace.v1",
                "failure_label_mapping_required": True,
            },
            "adapter_smoke_contract": _simulator_adapter_smoke_contract(framework),
            "proof_boundary": dict(CLAIM_BOUNDARY),
        }

    world_model_plugins = _build_world_model_engine_plugins(
        plan=plan,
        scenario_variation_instances=scenario_variation_instances,
    )
    registry = {
        "schema_version": SIMULATOR_ENGINE_PLUGIN_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": plan.get("scene_id"),
        "capture_id": plan.get("capture_id"),
        "capture_root": str(context.capture_root),
        "status": "ready_for_gated_managed_execution",
        "engine_targets": list(SIMULATOR_FRAMEWORKS),
        "world_model_engine_targets": list(WORLD_MODEL_ENGINE_TARGETS),
        "plugin_count": len(plugins),
        "world_model_plugin_count": len(world_model_plugins),
        "plugins": plugins,
        "world_model_plugins": world_model_plugins,
        "scenario_variation_instances_path": "scenario_variation_instances.json",
        "simulator_execution_manifest_path": "simulator_execution_manifest.json",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(automation_dir / "simulator_engine_plugin_registry.json", registry)
    return registry


def _request_for_framework(
    *,
    framework: str,
    plan: Mapping[str, Any],
    conversion_plan: Mapping[str, Any],
    command: str | None,
    generated_at: str,
) -> Dict[str, Any]:
    conversion = _mapping(_mapping(conversion_plan.get("frameworks")).get(framework))
    arena_packet_path = (
        "arena_environment_packet.json" if framework == "isaac_lab_arena" else None
    )
    return {
        "schema_version": SIMULATOR_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "framework": framework,
        "scene_id": plan.get("scene_id"),
        "capture_id": plan.get("capture_id"),
        "status": "requested_if_approved",
        "command": shlex.split(command) if command else [],
        "conversion_status": conversion.get("status"),
        "arena_environment_packet_path": arena_packet_path,
        "input_assets": conversion.get("input_assets") or {},
        "expected_outputs": [
            "stdout",
            "stderr",
            "exit_code",
            "load_trace",
            "artifact_paths",
        ],
        "approval_gates": [
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true",
            f"--allow-simulator {framework}",
            "explicit simulator command or runner dependency",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _blocked_simulator_result(
    *,
    framework: str,
    result_path: Path,
    request_path: Path,
    reason: str,
    command: Sequence[str],
    blockers: Sequence[str],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SIMULATOR_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "framework": framework,
        "status": "blocked",
        "reason": reason,
        "blockers": list(blockers),
        "command": list(command),
        "request_manifest": str(request_path),
        "blocked_manifest": str(result_path),
        "stdout_path": None,
        "stderr_path": None,
        "exit_code": None,
        "artifact_paths": [],
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _run_simulator_command(
    *,
    framework: str,
    result_path: Path,
    request_path: Path,
    command: Sequence[str],
    capture_root: Path,
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    stdout_path = result_path.with_name(f"{framework}_stdout.log")
    stderr_path = result_path.with_name(f"{framework}_stderr.log")
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(capture_root.resolve()),
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result = _blocked_simulator_result(
            framework=framework,
            result_path=result_path,
            request_path=request_path,
            reason="execution_error",
            command=command,
            blockers=[str(exc) or exc.__class__.__name__],
            generated_at=generated_at,
        )
        write_json(result_path, result)
        return result
    write_text(stdout_path, completed.stdout)
    write_text(stderr_path, completed.stderr)
    status = "completed" if completed.returncode == 0 else "failed"
    return {
        "schema_version": SIMULATOR_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "framework": framework,
        "status": status,
        "reason": None if status == "completed" else f"simulator_exit_code:{completed.returncode}",
        "blockers": [] if status == "completed" else [f"simulator_exit_code:{completed.returncode}"],
        "command": list(command),
        "request_manifest": str(request_path),
        "blocked_manifest": None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "exit_code": completed.returncode,
        "artifact_paths": [str(stdout_path), str(stderr_path)],
        "simulator_execution_proven": status == "completed",
        "robot_readiness_proven": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulators_run": True,
            "simulator_execution_proven": status == "completed",
        },
    }


def _build_simulator_execution_manifest(
    *,
    context: Any,
    automation_dir: Path,
    plan: Mapping[str, Any],
    conversion_plan: Mapping[str, Any],
    allow_simulator_execution: bool,
    allowed_simulators: Sequence[str],
    simulator_commands: Mapping[str, str],
    generated_at: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    simulator_dir = automation_dir / "simulators"
    ensure_dir(simulator_dir)
    env_allows = _env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
    global_allowed = bool(allow_simulator_execution and env_allows)
    allowed = {item for item in allowed_simulators if item in SIMULATOR_FRAMEWORKS}
    results: List[Dict[str, Any]] = []
    requests: Dict[str, str] = {}
    result_paths: Dict[str, str] = {}
    for framework in SIMULATOR_FRAMEWORKS:
        command_text = _string(simulator_commands.get(framework))
        request_path = simulator_dir / f"{framework}_request.json"
        result_path = simulator_dir / f"{framework}_result.json"
        request = _request_for_framework(
            framework=framework,
            plan=plan,
            conversion_plan=conversion_plan,
            command=command_text or None,
            generated_at=generated_at,
        )
        write_json(request_path, request)
        requests[framework] = _relative_to(automation_dir, request_path)
        result_paths[framework] = _relative_to(automation_dir, result_path)

        command = shlex.split(command_text) if command_text else []
        if not global_allowed:
            result = _blocked_simulator_result(
                framework=framework,
                result_path=result_path,
                request_path=request_path,
                reason="approval_required",
                command=command,
                blockers=[
                    "Set BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true and pass --allow-simulator-execution.",
                ],
                generated_at=generated_at,
            )
        elif framework not in allowed:
            result = _blocked_simulator_result(
                framework=framework,
                result_path=result_path,
                request_path=request_path,
                reason="approval_required",
                command=command,
                blockers=[f"Pass --allow-simulator {framework} for this run."],
                generated_at=generated_at,
            )
        elif not command:
            result = _blocked_simulator_result(
                framework=framework,
                result_path=result_path,
                request_path=request_path,
                reason="missing_execution_command",
                command=[],
                blockers=[f"Provide --simulator-command {framework}=<command>."],
                generated_at=generated_at,
            )
        elif shutil.which(command[0]) is None:
            result = _blocked_simulator_result(
                framework=framework,
                result_path=result_path,
                request_path=request_path,
                reason="missing_dependency",
                command=command,
                blockers=[f"command_missing:{command[0]}"],
                generated_at=generated_at,
            )
        else:
            result = _run_simulator_command(
                framework=framework,
                result_path=result_path,
                request_path=request_path,
                command=command,
                capture_root=context.capture_root,
                timeout_seconds=timeout_seconds,
                generated_at=generated_at,
            )
        write_json(result_path, result)
        results.append(result)

    any_completed = any(item.get("status") == "completed" for item in results)
    any_failed = any(item.get("status") == "failed" for item in results)
    if any_failed:
        overall_status = "failed"
    elif any_completed:
        overall_status = "completed"
    else:
        overall_status = "blocked"
    manifest = {
        "schema_version": SIMULATOR_EXECUTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": plan.get("scene_id"),
        "capture_id": plan.get("capture_id"),
        "overall_status": overall_status,
        "execution_gate": {
            "allow_simulator_execution_flag": bool(allow_simulator_execution),
            "env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": env_allows,
            "allowed_simulators": sorted(allowed),
        },
        "simulator_requests": requests,
        "simulator_results": results,
        "simulator_result_paths": result_paths,
        "simulators_run": any(item.get("status") in {"completed", "failed"} for item in results),
        "simulator_execution_proven": any_completed,
        "robot_readiness_proven": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulators_run": any(item.get("status") in {"completed", "failed"} for item in results),
            "simulator_execution_proven": any_completed,
        },
    }
    write_json(automation_dir / "simulator_execution_manifest.json", manifest)
    return manifest


def _training_orchestration_manifest(
    *,
    context: Any,
    automation_dir: Path,
    plan: Mapping[str, Any],
    allow_training: bool,
    training_command: str | None,
    training_timeout_seconds: int | None,
    generated_at: str,
) -> Dict[str, Any]:
    env_allows = _env_truthy("BLUEPRINT_ALLOW_COSMOS_TRAINING")
    export_manifest_path = context.pipeline_root / "cosmos_training_export" / "manifest.json"
    if not (allow_training and env_allows):
        return {
            "schema_version": TRAINING_ORCHESTRATION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "scene_id": plan.get("scene_id"),
            "capture_id": plan.get("capture_id"),
            "status": "blocked",
            "reason": "approval_required",
            "runner": TRAINING_RUNNER,
            "export_manifest_path": _relative_to(automation_dir, export_manifest_path)
            if export_manifest_path.is_file()
            else None,
            "training_command_template": training_command,
            "approval_gates": [
                "BLUEPRINT_ALLOW_COSMOS_TRAINING=true",
                "--allow-training",
            ],
            "gpu_training_run": False,
            "checkpoint_path": None,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    from .synthesis.cosmos_lora_training import run_cosmos_lora_training

    result = run_cosmos_lora_training(
        capture_root=context.capture_root,
        training_command=training_command,
        timeout_seconds=training_timeout_seconds,
    )
    return {
        "schema_version": TRAINING_ORCHESTRATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": plan.get("scene_id"),
        "capture_id": plan.get("capture_id"),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "runner": TRAINING_RUNNER,
        "export_manifest_path": _relative_to(automation_dir, export_manifest_path)
        if export_manifest_path.is_file()
        else None,
        "training_command_template": training_command,
        "training_run_manifest_path": _relative_to(
            automation_dir,
            context.pipeline_root / "cosmos_training_export" / "training_run_manifest.json",
        ),
        "gpu_training_run": result.get("status") in {"completed", "failed"},
        "checkpoint_path": result.get("checkpoint_path"),
        "runner_result": result,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "gpu_training_run": result.get("status") in {"completed", "failed"},
            "training_proof_available": result.get("status") == "completed",
        },
    }


def _proof_aware_claim_boundary(
    *,
    simulators_run: bool = False,
    simulator_execution_proven: bool = False,
    owner_gpu_simulator_execution_proven: bool = False,
    isaac_sim_execution_proven: bool = False,
    isaac_robot_asset_execution_proven: bool = False,
    mujoco_g1_asset_execution_proven: bool = False,
    local_mujoco_g1_asset_execution_proven: bool = False,
    owner_gpu_default_policy_execution_proven: bool = False,
    owner_gpu_sim_robot_pov_evidence_proven: bool = False,
    gpu_training_run: bool = False,
    training_proof_available: bool = False,
) -> Dict[str, Any]:
    effective_simulator_proven = bool(
        simulator_execution_proven or owner_gpu_simulator_execution_proven
    )
    effective_simulators_run = bool(simulators_run or effective_simulator_proven)
    disallowed_claims = [
        claim
        for claim in CLAIM_BOUNDARY["disallowed_claims"]
        if not (claim == "simulator_execution_completed" and effective_simulator_proven)
    ]
    proof_upgrade_requires: List[str] = []
    for requirement in CLAIM_BOUNDARY["proof_upgrade_requires"]:
        if effective_simulator_proven and requirement in {
            "simulator load trace",
            "simulator stdout/stderr and exit code",
        }:
            continue
        if owner_gpu_default_policy_execution_proven and requirement == "action or policy logs":
            proof_upgrade_requires.append(
                "robot-team policy/action logs beyond the default smoke policy"
            )
            continue
        if effective_simulator_proven and requirement == "accepted simulator or real robot trial evidence":
            proof_upgrade_requires.append(
                "accepted real robot trial evidence for physical-readiness claims"
            )
            continue
        proof_upgrade_requires.append(requirement)
    return {
        **dict(CLAIM_BOUNDARY),
        "simulators_run": effective_simulators_run,
        "gpu_training_run": bool(gpu_training_run),
        "simulator_execution_proven": effective_simulator_proven,
        "isaac_sim_execution_proven": bool(isaac_sim_execution_proven),
        "isaac_robot_asset_execution_proven": bool(isaac_robot_asset_execution_proven),
        "mujoco_g1_asset_execution_proven": bool(mujoco_g1_asset_execution_proven),
        "local_mujoco_g1_asset_execution_proven": bool(
            local_mujoco_g1_asset_execution_proven
        ),
        "owner_gpu_simulator_execution_proven": bool(owner_gpu_simulator_execution_proven),
        "owner_gpu_default_policy_execution_proven": bool(
            owner_gpu_default_policy_execution_proven
        ),
        "default_sim_policy_execution_proven": bool(
            owner_gpu_default_policy_execution_proven
        ),
        "owner_gpu_sim_robot_pov_evidence_proven": bool(
            owner_gpu_sim_robot_pov_evidence_proven
        ),
        "real_robot_pov_evidence_proven": False,
        "robot_readiness_proven": False,
        "robot_policy_execution_proven": False,
        "physics_contact_validated": False,
        "safety_contact_proof_available": False,
        "training_proof_available": bool(training_proof_available),
        "public_claim_upgrade_allowed": False,
        "disallowed_claims": _string_list(disallowed_claims),
        "proof_upgrade_requires": _string_list(proof_upgrade_requires),
    }


def _proof_boundary(
    *,
    simulator_execution: Mapping[str, Any],
    training: Mapping[str, Any],
    owner_gpu_proof: Mapping[str, Any],
    local_mujoco_g1_smoke: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    simulator_proven = bool(simulator_execution.get("simulator_execution_proven"))
    owner_gpu_proven = bool(owner_gpu_proof.get("owner_gpu_simulator_execution_proven"))
    default_policy_proven = bool(owner_gpu_proof.get("owner_gpu_default_policy_execution_proven"))
    sim_robot_pov_proven = bool(owner_gpu_proof.get("owner_gpu_sim_robot_pov_evidence_proven"))
    isaac_sim_proven = bool(owner_gpu_proof.get("isaac_sim_execution_proven"))
    isaac_robot_asset_proven = bool(owner_gpu_proof.get("isaac_robot_asset_execution_proven"))
    owner_mujoco_g1_proven = bool(owner_gpu_proof.get("mujoco_g1_asset_execution_proven"))
    local_mujoco_g1_proven = bool(
        local_mujoco_g1_smoke.get("local_cpu_mujoco_execution_proven")
        and local_mujoco_g1_smoke.get("mujoco_g1_asset_execution_proven")
        and local_mujoco_g1_smoke.get("unitree_g1_asset_spawned")
    )
    if owner_mujoco_g1_proven:
        selected_simulator_asset_evidence = []
    elif local_mujoco_g1_proven:
        selected_simulator_asset_evidence = [
            "owner-runtime MuJoCo execution with the real Menagerie Unitree G1 MJCF asset",
        ]
    else:
        selected_simulator_asset_evidence = [
            "owner-runtime simulator execution with a selected real Unitree G1 asset",
        ]
    if not isaac_sim_proven and not owner_mujoco_g1_proven:
        selected_simulator_asset_evidence.append(
            "Isaac Sim execution with Unitree G1 USD robot asset only if the selected lane is Isaac"
        )
    training_completed = str(training.get("status") or "") == "completed"
    proof_claim_boundary = _proof_aware_claim_boundary(
        simulators_run=bool(simulator_execution.get("simulators_run")),
        simulator_execution_proven=simulator_proven,
        owner_gpu_simulator_execution_proven=owner_gpu_proven,
        isaac_sim_execution_proven=isaac_sim_proven,
        isaac_robot_asset_execution_proven=isaac_robot_asset_proven,
        mujoco_g1_asset_execution_proven=owner_mujoco_g1_proven,
        local_mujoco_g1_asset_execution_proven=local_mujoco_g1_proven,
        owner_gpu_default_policy_execution_proven=default_policy_proven,
        owner_gpu_sim_robot_pov_evidence_proven=sim_robot_pov_proven,
        gpu_training_run=bool(training.get("gpu_training_run")),
        training_proof_available=training_completed,
    )
    return {
        "schema_version": PROOF_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "owner_gpu_simulator_execution_proven": owner_gpu_proven,
        "simulator_execution_proven": simulator_proven or owner_gpu_proven,
        "isaac_sim_execution_proven": isaac_sim_proven,
        "isaac_robot_asset_execution_proven": isaac_robot_asset_proven,
        "unitree_g1_asset_spawned": bool(owner_gpu_proof.get("unitree_g1_asset_spawned")),
        "mujoco_g1_asset_execution_proven": owner_mujoco_g1_proven,
        "local_mujoco_g1_asset_execution_proven": local_mujoco_g1_proven,
        "local_mujoco_g1_smoke_manifest_path": (
            "mujoco_g1_local_smoke/mujoco_g1_local_smoke_manifest.json"
            if local_mujoco_g1_smoke
            else None
        ),
        "owner_gpu_default_policy_execution_proven": default_policy_proven,
        "default_sim_policy_execution_proven": default_policy_proven,
        "owner_gpu_sim_robot_pov_evidence_proven": sim_robot_pov_proven,
        "real_robot_pov_evidence_proven": False,
        "robot_readiness_proven": False,
        "robot_policy_execution_proven": False,
        "physics_contact_validated": False,
        "safety_contact_proof_available": False,
        "public_claim_upgrade_allowed": False,
        "training_proof": {
            "training_completed": training_completed,
            "checkpoint_path": training.get("checkpoint_path"),
            "training_run_manifest_path": training.get("training_run_manifest_path"),
        },
        "simulator_proof": {
            "simulators_run": bool(simulator_execution.get("simulators_run")),
            "result_paths": simulator_execution.get("simulator_result_paths") or {},
            "owner_gpu_proof_manifest": (
                "owner_gpu_simulator_execution_proof_manifest.json"
                if owner_gpu_proof
                else None
            ),
            "default_policy_execution_proven": default_policy_proven,
            "sim_robot_pov_evidence_proven": sim_robot_pov_proven,
            "isaac_sim_execution_proven": isaac_sim_proven,
            "isaac_robot_asset_execution_proven": isaac_robot_asset_proven,
            "mujoco_g1_asset_execution_proven": owner_mujoco_g1_proven,
            "local_mujoco_g1_asset_execution_proven": local_mujoco_g1_proven,
        },
        "remaining_required_evidence": [
            *selected_simulator_asset_evidence,
            "robot-team policy/action logs beyond the default smoke policy",
            "real robot POV video and aligned action logs",
            "physics/contact validation logs",
            "robot-team-owned robot assets",
            "accepted simulator or real robot trial evidence",
        ],
        "claim_boundary": proof_claim_boundary,
    }


def _gpu_backend_recommendations(
    *,
    inventory: Mapping[str, Any],
    collider_proxy_plan: Mapping[str, Any],
    conversion_plan: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    assets = [item for item in inventory.get("assets") or [] if isinstance(item, Mapping)]
    asset_types = {_string(item.get("asset_type")) for item in assets}
    real_collider = bool(collider_proxy_plan.get("real_collider_proven"))
    proxy_estimated = bool(collider_proxy_plan.get("proxy_estimated"))
    recommendations: List[Dict[str, Any]] = []
    if asset_types.intersection({"usd", "usda", "usdc"}) or "usd" in asset_types:
        recommendations.append(
            {
                "backend": "isaac_sim",
                "priority": 1,
                "recommendation": "preferred_for_rich_usd_scene_assets",
                "why": [
                    "USD-like asset detected",
                    "best fit for OpenUSD scene references, materials, and future Isaac Lab review",
                ],
                "requires_owner_gpu": True,
                "compatible_with_proxy_only": proxy_estimated,
                "conversion_status": _mapping(
                    _mapping(conversion_plan.get("frameworks")).get("isaac_sim")
                ).get("status"),
            }
        )
        recommendations.append(
            {
                "backend": "isaac_lab_arena",
                "priority": 2,
                "recommendation": "candidate_for_composable_policy_eval_package",
                "why": [
                    "USD-like asset detected",
                    "Arena packet can bind Blueprint scene, embodiment, task, scenario, and eval components",
                ],
                "requires_owner_gpu": True,
                "compatible_with_proxy_only": proxy_estimated,
                "conversion_status": _mapping(
                    _mapping(conversion_plan.get("frameworks")).get("isaac_lab_arena")
                ).get("status"),
                "artifact": "arena_environment_packet.json",
            }
        )
    if real_collider or asset_types.intersection({"urdf", "mjcf", "obj", "glb", "gltf"}):
        recommendations.append(
            {
                "backend": "mujoco",
                "priority": 3,
                "recommendation": "candidate_for_portable_collision_or_generated_proxy_fixture",
                "why": [
                    "Portable collision metadata or conversion target is available"
                    if real_collider
                    else "Only proxy/generated fixture is available; use for limited owner preflight only",
                ],
                "requires_owner_gpu": False,
                "compatible_with_proxy_only": proxy_estimated,
                "conversion_status": _mapping(
                    _mapping(conversion_plan.get("frameworks")).get("mujoco")
                ).get("status"),
            }
        )
        recommendations.append(
            {
                "backend": "pybullet",
                "priority": 4,
                "recommendation": "candidate_for_urdf_or_generated_proxy_fixture",
                "why": [
                    "URDF/collision assets or generated proxy fixture can support load sanity checks",
                ],
                "requires_owner_gpu": False,
                "compatible_with_proxy_only": proxy_estimated,
                "conversion_status": _mapping(
                    _mapping(conversion_plan.get("frameworks")).get("pybullet")
                ).get("status"),
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "backend": "isaac_sim",
                "priority": 1,
                "recommendation": "owner_review_required_before_backend_selection",
                "why": ["No compatible local scene asset or collider/proxy plan was proven locally"],
                "requires_owner_gpu": True,
                "compatible_with_proxy_only": False,
                "conversion_status": None,
            }
        )
        recommendations.append(
            {
                "backend": "isaac_lab_arena",
                "priority": 2,
                "recommendation": "owner_review_required_before_arena_package_execution",
                "why": ["Arena package can still be reviewed, but no compatible scene asset was proven locally"],
                "requires_owner_gpu": True,
                "compatible_with_proxy_only": False,
                "conversion_status": _mapping(
                    _mapping(conversion_plan.get("frameworks")).get("isaac_lab_arena")
                ).get("status"),
                "artifact": "arena_environment_packet.json",
            }
        )
    return sorted(recommendations, key=lambda item: int(item.get("priority") or 99))


def _gpu_owner_system_proof_schema(*, generated_at: str, scene_id: str, capture_id: str) -> Dict[str, Any]:
    return {
        "schema_version": GPU_OWNER_SYSTEM_PROOF_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "required_fields": [
            "owner_system_id",
            "simulator_backend",
            "simulator_version",
            "gpu_model",
            "robot_asset",
            "command",
            "started_at",
            "completed_at",
            "exit_code",
            "stdout_uri_or_path",
            "stderr_uri_or_path",
            "scene_load_trace_uri_or_path",
            "spawn_pose_validation_uri_or_path",
            "action_or_policy_trace_uri_or_path",
            "default_smoke_policy_uri_or_path",
            "policy_execution_trace_uri_or_path",
            "sim_robot_pov_evidence_uri_or_path",
            "artifact_manifest_uri_or_path",
            "pass_fail_criteria",
            "operator_attestation",
        ],
        "proof_booleans_allowed_true_only_with_owner_evidence": [
            "owner_system_simulator_execution_proven",
            "scene_loaded_in_owner_simulator",
            "spawn_pose_loaded",
            "owner_gpu_default_policy_execution_proven",
            "owner_gpu_sim_robot_pov_evidence_proven",
            "isaac_sim_execution_proven",
            "isaac_robot_asset_execution_proven",
        ],
        "default_isaac_robot_asset": dict(DEFAULT_ISAAC_HUMANOID_ROBOT_ASSET),
        "conditional_requirements": {
            "isaac_sim_or_isaac_lab_arena": [
                "proof.robot_asset identifies Unitree G1",
                "spawn trace robot_asset identifies Robots/Unitree/G1/g1.usd",
                "proof robot_asset and spawn trace robot_asset match",
            ],
        },
        "proof_booleans_still_require_separate_evidence": {
            "real_robot_pov_evidence_proven": [
                "physical robot camera video",
                "aligned physical robot action log",
                "owner attestation for the real robot run",
            ],
            "robot_readiness_proven": [
                "accepted robot policy/action logs",
                "buyer/operator-approved methodology",
                "actual evaluation result manifest",
            ],
            "physics_contact_validated": [
                "contact validation logs",
                "collider methodology review",
            ],
            "safety_validated": [
                "safety methodology",
                "operator safety signoff",
            ],
        },
        "disallowed_without_owner_system_evidence": list(CLAIM_BOUNDARY["disallowed_claims"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _append_blocker_detail(
    details: List[Dict[str, Any]],
    seen: set[str],
    *,
    blocker_id: str,
    source_artifact: str,
    severity: str,
    required_input: str,
    safe_next_command: str | None = None,
) -> None:
    if not blocker_id or blocker_id in seen:
        return
    seen.add(blocker_id)
    detail: Dict[str, Any] = {
        "blocker_id": blocker_id,
        "source_artifact": source_artifact,
        "severity": severity,
        "required_input": required_input,
        "proof_boundary": "required input only; does not prove simulator execution or robot readiness",
    }
    if safe_next_command:
        detail["safe_next_command"] = safe_next_command
    details.append(detail)


def _gpu_handoff_blocker_details(
    *,
    scene_preflight: Mapping[str, Any],
    spawn_validation: Mapping[str, Any],
    cpu_preflight: Mapping[str, Any],
    owner_gpu_proven: bool,
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    seen: set[str] = set()
    hard_blockers = _string_list(cpu_preflight.get("hard_preflight_blockers"))
    scene_blockers = _string_list(scene_preflight.get("blockers"))
    spawn_blockers = (
        _string_list(spawn_validation.get("blockers"))
        if spawn_validation.get("status") == "blocked"
        else []
    )
    all_blockers = _string_list([*hard_blockers, *scene_blockers, *spawn_blockers])

    if not owner_gpu_proven:
        _append_blocker_detail(
            details,
            seen,
            blocker_id="owner_gpu_simulator_execution_not_run",
            source_artifact="gpu_owner_system_proof.json",
            severity="expected_before_first_gpu_attempt",
            required_input=(
                "Run the selected simulator backend on an owner GPU system and provide "
                "gpu_owner_system_proof.json plus stdout/stderr/load/spawn/action traces."
            ),
            safe_next_command=(
                "blueprint-run-owner-gpu-proof --capture-root <capture-root> "
                "--proof-dir <proof-dir> --command <owner simulator command>"
            ),
        )

    for blocker in all_blockers:
        if blocker == "missing_local_scene_asset":
            _append_blocker_detail(
                details,
                seen,
                blocker_id=blocker,
                source_artifact="scene_asset_preflight.json",
                severity="hard_pre_gpu_blocker",
                required_input=(
                    "Provide a local materialized scene asset with geometry or bounds, such as "
                    "World Labs/SimReady/OpenUSD/glTF/PLY output referenced by the capture package."
                ),
                safe_next_command="blueprint-run-simulation-automation --capture-root <capture-root>",
            )
        elif blocker == "missing_scene_frame_estimate":
            _append_blocker_detail(
                details,
                seen,
                blocker_id=blocker,
                source_artifact="scene_frame_estimate.json",
                severity="hard_pre_gpu_blocker",
                required_input=(
                    "Generate scene_frame_estimate.json from a local scene asset that exposes "
                    "finite min/max bounds and floor_z_estimate for CPU spawn sanity checks."
                ),
                safe_next_command="blueprint-run-simulation-automation --capture-root <capture-root>",
            )
        elif blocker == "scene_bounds_missing_or_invalid":
            _append_blocker_detail(
                details,
                seen,
                blocker_id=blocker,
                source_artifact="spawn_pose_validation_manifest.json",
                severity="hard_pre_gpu_blocker",
                required_input=(
                    "Provide finite scene bounds before owner GPU execution; spawn validation "
                    "cannot distinguish an invalid spawn from missing scene geometry."
                ),
                safe_next_command="blueprint-run-simulation-automation --capture-root <capture-root>",
            )
        elif blocker == "scene_bounds_empty_or_inverted":
            _append_blocker_detail(
                details,
                seen,
                blocker_id=blocker,
                source_artifact="spawn_pose_validation_manifest.json",
                severity="hard_pre_gpu_blocker",
                required_input="Repair scene bounds so max xyz is greater than min xyz on every axis.",
                safe_next_command="blueprint-run-simulation-automation --capture-root <capture-root>",
            )
        elif blocker == "spawn_outside_scene_bounds":
            _append_blocker_detail(
                details,
                seen,
                blocker_id=blocker,
                source_artifact="spawn_pose_validation_manifest.json",
                severity="hard_pre_gpu_blocker",
                required_input=(
                    "Choose or review a robot spawn pose inside the scene bounds before owner GPU execution."
                ),
                safe_next_command="blueprint-run-simulation-automation --capture-root <capture-root>",
            )
        elif blocker == "spawn_inside_known_or_proxy_geometry":
            _append_blocker_detail(
                details,
                seen,
                blocker_id=blocker,
                source_artifact="spawn_pose_validation_manifest.json",
                severity="hard_pre_gpu_blocker",
                required_input=(
                    "Choose a robot spawn pose outside known/proxy geometry before owner GPU execution."
                ),
                safe_next_command="blueprint-run-simulation-automation --capture-root <capture-root>",
            )
        elif blocker in {"portable_collider_glb_missing", "isaac_usd_collision_unverified"}:
            _append_blocker_detail(
                details,
                seen,
                blocker_id=blocker,
                source_artifact="scene_asset_preflight.json",
                severity="review_or_backend_selection_blocker",
                required_input=(
                    "Provide or review collision assets if the selected simulator/backend requires "
                    "contact or collision confidence; this does not block visual scene-load smoke by itself."
                ),
            )
        elif blocker == "simulator_execution_not_run":
            continue
        else:
            _append_blocker_detail(
                details,
                seen,
                blocker_id=blocker,
                source_artifact="pre_gpu_readiness_summary.json",
                severity="pre_gpu_review_required",
                required_input="Inspect the named source artifact and supply the missing pre-GPU input.",
            )
    return details


def _spawn_validation_summary(spawn_validation: Mapping[str, Any]) -> Dict[str, Any]:
    validations = [
        dict(item)
        for item in spawn_validation.get("validations") or []
        if isinstance(item, Mapping)
    ]
    return {
        "status": spawn_validation.get("status"),
        "episode_count": spawn_validation.get("episode_count") or len(validations),
        "blockers": _string_list(spawn_validation.get("blockers")),
        "valid_candidate_count": sum(
            int(item.get("valid_candidate_count") or 0) for item in validations
        ),
        "candidate_count": sum(int(item.get("candidate_count") or 0) for item in validations),
    }


def _gpu_run_checklist_text(packet: Mapping[str, Any]) -> str:
    commands = packet.get("command_templates") or {}
    recommended = packet.get("recommended_backends") or []
    primary_backend = _string(_mapping(recommended[0] if recommended else {}).get("backend")) or "isaac_sim"
    primary_command = _string(_mapping(commands).get(primary_backend)) or _string(
        _mapping(commands).get("isaac_sim")
    )
    return "\n".join(
        [
            "# GPU Owner-System Run Checklist",
            "",
            "Status: owner-system GPU simulator execution is not proven by this packet.",
            "",
            "## CPU Already Checked",
            "- Local scene asset inventory and dependency audit were generated.",
            "- Collider/proxy planning was generated with review-required labels.",
            "- Episode specs and task-anchor proposals were generated as advisory setup inputs.",
            "- Spawn pose validation ran against local bounds and proxy metadata where available.",
            "",
            "## Owner GPU Must Prove",
            "- The selected scene loads in the owner simulator.",
            "- The selected spawn pose loads without immediate invalid state.",
            "- Simulator stdout, stderr, exit code, and load trace are captured.",
            "- The default walk-to-target smoke policy writes an execution trace.",
            "- Simulator robot POV evidence is captured as video or frame evidence.",
            "- Contact, safety, real robot POV, and robot-readiness claims remain false without their own logs.",
            "",
            "## Recommended First Command",
            "```bash",
            primary_command or "<provide owner-system simulator command>",
            "```",
            "",
            "## Pass Criteria",
            "- Command exits 0 before timeout.",
            "- Proof schema JSON is filled with owner-system logs and artifact paths.",
            "- Scene load trace names the backend, version, scene asset, and spawn pose.",
            "- Policy trace and simulator POV evidence manifest are present and valid.",
            "",
            "## Fail Criteria",
            "- Missing dependency, missing asset reference, nonzero exit, timeout, empty load trace, or no owner attestation.",
            "- Any attempt to mark robot readiness, safety, or contact proof from CPU-only artifacts.",
            "",
        ]
    )


def _build_gpu_handoff_artifacts(
    *,
    context: Any,
    automation_dir: Path,
    plan: Mapping[str, Any],
    conversion_plan: Mapping[str, Any],
    generated_at: str,
    simulator_timeout_seconds: int,
) -> Dict[str, Any]:
    inventory = _read_optional_mapping(automation_dir / "scene_asset_inventory.json")
    dependency_audit = _read_optional_mapping(automation_dir / "scene_asset_dependency_audit.json")
    collider_proxy_plan = _read_optional_mapping(automation_dir / "collider_proxy_plan.json")
    spawn_validation = _read_optional_mapping(automation_dir / "spawn_pose_validation_manifest.json")
    cpu_preflight = _read_optional_mapping(automation_dir / "cpu_preflight_manifest.json")
    pre_gpu_summary = _read_optional_mapping(automation_dir / "pre_gpu_readiness_summary.json")
    arena_packet = _read_optional_mapping(automation_dir / "arena_environment_packet.json")
    owner_gpu_proof = _read_optional_mapping(
        automation_dir / "owner_gpu_simulator_execution_proof_manifest.json"
    )
    owner_gpu_proven = bool(owner_gpu_proof.get("owner_gpu_simulator_execution_proven"))
    default_policy_proven = bool(owner_gpu_proof.get("owner_gpu_default_policy_execution_proven"))
    sim_robot_pov_proven = bool(owner_gpu_proof.get("owner_gpu_sim_robot_pov_evidence_proven"))
    isaac_sim_proven = bool(owner_gpu_proof.get("isaac_sim_execution_proven"))
    isaac_robot_asset_proven = bool(owner_gpu_proof.get("isaac_robot_asset_execution_proven"))
    mujoco_g1_asset_proven = bool(owner_gpu_proof.get("mujoco_g1_asset_execution_proven"))
    handoff_claim_boundary = _proof_aware_claim_boundary(
        simulators_run=owner_gpu_proven,
        simulator_execution_proven=owner_gpu_proven,
        owner_gpu_simulator_execution_proven=owner_gpu_proven,
        isaac_sim_execution_proven=isaac_sim_proven,
        isaac_robot_asset_execution_proven=isaac_robot_asset_proven,
        mujoco_g1_asset_execution_proven=mujoco_g1_asset_proven,
        owner_gpu_default_policy_execution_proven=default_policy_proven,
        owner_gpu_sim_robot_pov_evidence_proven=sim_robot_pov_proven,
    )
    recommended_backends = _gpu_backend_recommendations(
        inventory=inventory,
        collider_proxy_plan=collider_proxy_plan,
        conversion_plan=conversion_plan,
    )
    command_templates = {
        "isaac_sim": (
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true "
            "blueprint-run-simulation-automation --capture-root <capture-root> "
            "--allow-simulator-execution --allow-simulator isaac_sim "
            "--simulator-command isaac_sim='<owner isaac sim headless command "
            "--scene <scene-asset> --proof-out <proof-dir>>'"
        ),
        "isaac_lab_arena": (
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true "
            "blueprint-run-simulation-automation --capture-root <capture-root> "
            "--allow-simulator-execution --allow-simulator isaac_lab_arena "
            "--simulator-command isaac_lab_arena='<owner Isaac Lab-Arena command "
            "--arena-pack pipeline/simulation_automation/arena_environment_packet.json "
            "--proof-out <proof-dir>>'"
        ),
        "mujoco": (
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true "
            "blueprint-run-simulation-automation --capture-root <capture-root> "
            "--allow-simulator-execution --allow-simulator mujoco "
            "--simulator-command mujoco='<owner mujoco load command "
            "--mjcf pipeline/simulation_automation/mujoco_cpu_preflight/episode_scene.xml>'"
        ),
        "pybullet": (
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true "
            "blueprint-run-simulation-automation --capture-root <capture-root> "
            "--allow-simulator-execution --allow-simulator pybullet "
            "--simulator-command pybullet='<owner pybullet load command "
            "--urdf pipeline/simulation_automation/pybullet_cpu_preflight/episode_scene.urdf>'"
        ),
    }
    proof_schema = _gpu_owner_system_proof_schema(
        generated_at=generated_at,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
    )
    blockers = [] if owner_gpu_proven else ["owner_gpu_simulator_execution_not_run"]
    hard_missing_dependency_count = int(
        dependency_audit.get(
            "hard_missing_local_file_count",
            dependency_audit.get("missing_local_file_count") or 0,
        )
        or 0
    )
    if hard_missing_dependency_count > 0:
        blockers.append("missing_scene_asset_dependencies")
    if spawn_validation.get("status") == "blocked":
        blockers.append("spawn_validation_blocked")
    blocker_details = _gpu_handoff_blocker_details(
        scene_preflight=_read_optional_mapping(automation_dir / "scene_asset_preflight.json"),
        spawn_validation=spawn_validation,
        cpu_preflight=cpu_preflight,
        owner_gpu_proven=owner_gpu_proven,
    )
    spawn_summary = _spawn_validation_summary(spawn_validation)
    packet = {
        "schema_version": GPU_HANDOFF_PACKET_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "ready_for_owner_gpu_preflight_handoff"
        if bool(cpu_preflight.get("ready_for_owner_gpu_preflight"))
        else "blocked_for_owner_gpu_preflight_handoff",
        "ready_for_owner_gpu_preflight": bool(cpu_preflight.get("ready_for_owner_gpu_preflight")),
        "owner_gpu_simulator_execution_proven": owner_gpu_proven,
        "simulator_execution_proven": owner_gpu_proven,
        "owner_gpu_default_policy_execution_proven": default_policy_proven,
        "default_sim_policy_execution_proven": default_policy_proven,
        "owner_gpu_sim_robot_pov_evidence_proven": sim_robot_pov_proven,
        "real_robot_pov_evidence_proven": False,
        "robot_readiness_proven": False,
        "isaac_sim_execution_proven": isaac_sim_proven,
        "isaac_robot_asset_execution_proven": isaac_robot_asset_proven,
        "mujoco_g1_asset_execution_proven": mujoco_g1_asset_proven,
        "cpu_checked": pre_gpu_summary.get("cpu_checked") or [],
        "cpu_artifacts": {
            "scene_asset_inventory": "scene_asset_inventory.json",
            "scene_asset_dependency_audit": "scene_asset_dependency_audit.json",
            "collider_proxy_plan": "collider_proxy_plan.json",
            "cpu_scene_proxy_manifest": "cpu_scene_proxy_manifest.json",
            "task_anchor_proposal_manifest": "task_anchor_proposal_manifest.json",
            "episode_specs": "episode_specs.json",
            "arena_environment_packet": "arena_environment_packet.json",
            "spawn_pose_validation_manifest": "spawn_pose_validation_manifest.json",
            "cpu_preflight_manifest": "cpu_preflight_manifest.json",
            "pre_gpu_readiness_summary": "pre_gpu_readiness_summary.json",
        },
        "dependency_summary": {
            "missing_local_file_count": dependency_audit.get("missing_local_file_count", 0),
            "hard_missing_local_file_count": hard_missing_dependency_count,
            "owner_system_material_warning_count": dependency_audit.get(
                "owner_system_material_warning_count",
                0,
            ),
            "remote_ref_count": dependency_audit.get("remote_ref_count", 0),
            "unresolved_ref_count": dependency_audit.get("unresolved_ref_count", 0),
        },
        "spawn_validation_summary": spawn_summary,
        "hard_preflight_blockers": _string_list(cpu_preflight.get("hard_preflight_blockers")),
        "pre_gpu_blocker_details": blocker_details,
        "recommended_backends": recommended_backends,
        "target_backend_guidance": {
            "isaac_sim": "Prefer for rich USD/OpenUSD scenes, references, materials, and future Isaac Lab workflows.",
            "isaac_lab_arena": (
                "Use when the owner wants a composable Isaac Lab-Arena scene/embodiment/task/eval package; "
                "the packet is a review input until owner-system Arena execution proof exists."
            ),
            "mujoco": "Use for compatible MJCF or generated/proxy fixtures, not for rich USD scene proof.",
            "pybullet": "Use for URDF/proxy fixture load checks, not as rich-scene proof unless owner accepts it.",
        },
        "required_dependencies": {
            "isaac_sim": ["NVIDIA GPU", "Isaac Sim/Isaac Lab owner install", "OpenUSD dependencies"],
            "isaac_lab_arena": [
                "NVIDIA GPU",
                "Owner-pinned Isaac Lab-Arena install",
                "Isaac Lab/Isaac Sim compatible versions",
                "robot-team embodiment assets",
                "OpenUSD or mapped scene assets",
            ],
            "mujoco": ["mujoco Python package or owner MuJoCo install"],
            "pybullet": ["pybullet Python package or owner PyBullet install"],
        },
        "arena_package": {
            "path": "arena_environment_packet.json" if arena_packet else None,
            "status": arena_packet.get("status") if arena_packet else "missing",
            "backend": "isaac_lab_arena",
            "episode_count": len(
                _mapping(arena_packet.get("arena_components")).get("episode_bindings") or []
            )
            if arena_packet
            else 0,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
        },
        "command_templates": command_templates,
        "environment_gates": {
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": "true",
            "BLUEPRINT_ALLOW_GPU_PROVISIONING": "true only if a paid/non-fixture GPU provisioner is intentionally used",
        },
        "expected_logs": [
            "stdout",
            "stderr",
            "exit_code",
            "scene_load_trace",
            "spawn_pose_trace",
            "action_or_policy_trace",
            "default_smoke_policy",
            "policy_execution_trace",
            "sim_robot_pov_evidence",
            "artifact_manifest",
            "owner_attestation",
        ],
        "owner_gpu_proof_manifest_path": (
            "owner_gpu_simulator_execution_proof_manifest.json"
            if owner_gpu_proof
            else None
        ),
        "pass_fail_criteria": {
            "pass": [
                "command exits 0 before timeout",
                "owner proof schema is complete",
                "scene load trace identifies backend, scene asset, and spawn pose",
                "default walk-to-target policy trace records an attempted action",
                "simulator robot POV evidence manifest records camera/video/frame evidence",
            ],
            "fail": [
                "nonzero exit",
                "timeout",
                "missing referenced asset",
                "empty or missing load trace",
                "missing owner attestation",
            ],
        },
        "output_artifacts_expected": [
            "gpu_owner_system_proof.json",
            "owner_simulator_stdout.log",
            "owner_simulator_stderr.log",
            "owner_scene_load_trace.json",
            "owner_spawn_pose_trace.json",
            "owner_default_smoke_policy.json",
            "owner_action_or_policy_trace.json",
            "owner_sim_robot_pov_evidence_manifest.json",
            "owner_artifact_manifest.json",
        ],
        "timeout_seconds": simulator_timeout_seconds,
        "blockers": blockers,
        "claim_boundary": handoff_claim_boundary,
    }
    blocked_manifest = {
        "schema_version": OWNER_GPU_BLOCKED_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "resolved" if owner_gpu_proven else "blocked",
        "blocker_id": "owner_gpu_simulator_execution_not_run",
        "owner": "robot_team_or_owner_system_operator",
        "required_input": "Run the selected simulator backend on an owner GPU system and provide the proof schema outputs.",
        "safe_proof_command": command_templates.get("isaac_sim"),
        "retry_condition": "Retry after owner-system proof artifacts are written and synced.",
        "disallowed_workaround": "Do not mark simulator, robot readiness, contact, policy, or safety proof from CPU-only artifacts.",
        "pre_gpu_blocker_details": blocker_details,
        "next_artifacts": packet["output_artifacts_expected"],
        "claim_boundary": handoff_claim_boundary,
    }
    write_json(automation_dir / "gpu_owner_system_proof_schema.json", proof_schema)
    write_json(automation_dir / "gpu_handoff_packet.json", packet)
    write_json(
        automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json",
        blocked_manifest,
    )
    write_text(automation_dir / "gpu_run_checklist.md", _gpu_run_checklist_text(packet))
    return {
        "packet": packet,
        "proof_schema": proof_schema,
        "blocked_manifest": blocked_manifest,
    }


def _default_agent_ledger(*, generated_at: str) -> Dict[str, Any]:
    return {
        "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "adapter": "none",
        "status": "not_requested",
        "operator_mode": "not_requested",
        "proof_booleans_mutable_by_agent": False,
        "network_required": False,
        "live_provider_calls_performed": False,
        "decisions": [],
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
        "diagnostics": [],
        "proof_effect": proof_effect(
            deterministic_artifacts_required=CLAIM_BOUNDARY["proof_upgrade_requires"]
        ),
    }


def build_simulation_automation(
    *,
    capture_root: str | Path,
    scene_assets: Sequence[str | Path] | None = None,
    allow_simulator_execution: bool = False,
    allowed_simulators: Sequence[str] | None = None,
    simulator_commands: Mapping[str, str] | None = None,
    simulator_timeout_seconds: int = 120,
    allow_cpu_simulator_preflight: bool = False,
    cpu_preflight_backends: Sequence[str] | None = None,
    cpu_preflight_smoke_steps: int = 10,
    allow_cpu_preflight_render: bool = False,
    allow_training: bool = False,
    training_command: str | None = None,
    training_timeout_seconds: int | None = None,
    agent_adapter: SimulationAutomationAgentAdapter | None = None,
    episode_agent_adapter: EpisodeSpecAgentAdapter | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    automation_dir = pipeline_dir / "simulation_automation"
    ensure_dir(automation_dir)
    generated_at = _timestamp(
        _read_optional_mapping(pipeline_dir / "worldlabs_world_manifest.json"),
        _read_optional_mapping(pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json"),
        _read_optional_mapping(pipeline_dir / "simready" / "simready_scene_manifest.json"),
    )
    scene_preflight = build_scene_asset_preflight(
        capture_root=context.capture_root,
        scene_assets=scene_assets or (),
    )
    episode_specs = build_episode_specs(
        capture_root=context.capture_root,
        agent_adapter=episode_agent_adapter,
    )
    cpu_preflight = build_cpu_simulator_preflight(
        capture_root=context.capture_root,
        allow_cpu_simulator_preflight=allow_cpu_simulator_preflight,
        backends=cpu_preflight_backends or CPU_BACKENDS,
        smoke_steps=cpu_preflight_smoke_steps,
        allow_render=allow_cpu_preflight_render,
    )
    scenario_variation_instances = build_scenario_variation_instances(
        capture_root=context.capture_root,
        output_dir=automation_dir,
        generated_at=generated_at,
    )

    plan = _build_plan(
        context=context,
        automation_dir=automation_dir,
        pipeline_dir=pipeline_dir,
        generated_at=generated_at,
    )
    conversion_plan = _build_asset_conversion_plan(plan=plan, generated_at=generated_at)
    arena_environment_packet = _build_arena_environment_packet(
        context=context,
        automation_dir=automation_dir,
        pipeline_dir=pipeline_dir,
        conversion_plan=conversion_plan,
        generated_at=generated_at,
    )
    simulator_engine_plugin_registry = _build_simulator_engine_plugin_registry(
        context=context,
        automation_dir=automation_dir,
        plan=plan,
        conversion_plan=conversion_plan,
        scenario_variation_instances=scenario_variation_instances,
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=allowed_simulators or [],
        simulator_commands=simulator_commands or {},
        generated_at=generated_at,
    )
    simulator_execution = _build_simulator_execution_manifest(
        context=context,
        automation_dir=automation_dir,
        plan=plan,
        conversion_plan=conversion_plan,
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=allowed_simulators or [],
        simulator_commands=simulator_commands or {},
        generated_at=generated_at,
        timeout_seconds=simulator_timeout_seconds,
    )
    training = _training_orchestration_manifest(
        context=context,
        automation_dir=automation_dir,
        plan=plan,
        allow_training=allow_training,
        training_command=training_command,
        training_timeout_seconds=training_timeout_seconds,
        generated_at=generated_at,
    )
    owner_gpu_proof_path = automation_dir / "gpu_owner_system_proof.json"
    owner_gpu_proof = (
        validate_owner_gpu_system_proof(
            proof_path=owner_gpu_proof_path,
            capture_root=context.capture_root,
            output_path=automation_dir / "owner_gpu_simulator_execution_proof_manifest.json",
        )
        if owner_gpu_proof_path.is_file()
        else {}
    )
    local_mujoco_g1_smoke = _read_optional_mapping(
        automation_dir / "mujoco_g1_local_smoke" / "mujoco_g1_local_smoke_manifest.json"
    )
    proof_boundary = _proof_boundary(
        simulator_execution=simulator_execution,
        training=training,
        owner_gpu_proof=owner_gpu_proof,
        local_mujoco_g1_smoke=local_mujoco_g1_smoke,
        generated_at=generated_at,
    )
    gpu_handoff = _build_gpu_handoff_artifacts(
        context=context,
        automation_dir=automation_dir,
        plan=plan,
        conversion_plan=conversion_plan,
        generated_at=generated_at,
        simulator_timeout_seconds=simulator_timeout_seconds,
    )
    plan_context = {
        "repo_root": str(Path(__file__).resolve().parents[2]),
        "capture_root": str(context.capture_root),
        "plan": plan,
        "scene_preflight": scene_preflight,
        "episode_specs": episode_specs,
        "cpu_preflight": cpu_preflight,
        "scenario_variation_instances": scenario_variation_instances,
        "conversion_plan": conversion_plan,
        "arena_environment_packet": arena_environment_packet,
        "simulator_engine_plugin_registry": simulator_engine_plugin_registry,
        "simulator_execution": simulator_execution,
        "training": training,
        "proof_boundary": proof_boundary,
        "gpu_handoff": gpu_handoff.get("packet"),
    }
    agent_ledger = (
        agent_adapter.build_ledger(plan_context=plan_context)
        if agent_adapter is not None
        else _default_agent_ledger(generated_at=generated_at)
    )
    agent_ledger.setdefault("generated_at", generated_at)
    agent_ledger.setdefault("claim_boundary", dict(CLAIM_BOUNDARY))

    status_inputs = [
        str(conversion_plan.get("status") or ""),
        str(simulator_execution.get("overall_status") or ""),
        str(training.get("status") or ""),
    ]
    status = "completed" if all(item == "completed" for item in status_inputs) else "blocked"
    run_claim_boundary = _proof_aware_claim_boundary(
        simulators_run=bool(simulator_execution.get("simulators_run")),
        simulator_execution_proven=bool(proof_boundary.get("simulator_execution_proven")),
        owner_gpu_simulator_execution_proven=bool(
            owner_gpu_proof.get("owner_gpu_simulator_execution_proven")
        ),
        isaac_sim_execution_proven=bool(proof_boundary.get("isaac_sim_execution_proven")),
        isaac_robot_asset_execution_proven=bool(
            proof_boundary.get("isaac_robot_asset_execution_proven")
        ),
        mujoco_g1_asset_execution_proven=bool(
            proof_boundary.get("mujoco_g1_asset_execution_proven")
        ),
        local_mujoco_g1_asset_execution_proven=bool(
            proof_boundary.get("local_mujoco_g1_asset_execution_proven")
        ),
        owner_gpu_default_policy_execution_proven=bool(
            owner_gpu_proof.get("owner_gpu_default_policy_execution_proven")
        ),
        owner_gpu_sim_robot_pov_evidence_proven=bool(
            owner_gpu_proof.get("owner_gpu_sim_robot_pov_evidence_proven")
        ),
        gpu_training_run=bool(training.get("gpu_training_run")),
        training_proof_available=str(training.get("status") or "") == "completed",
    )
    run_manifest = {
        "schema_version": SIMULATION_AUTOMATION_RUN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": status,
        "plan_path": "simulation_automation_plan.json",
        "scene_asset_inventory_path": "scene_asset_inventory.json",
        "scene_asset_dependency_audit_path": "scene_asset_dependency_audit.json",
        "scene_asset_preflight_path": "scene_asset_preflight.json",
        "scene_asset_inspection_path": "scene_asset_inspection.json",
        "scene_frame_estimate_path": "scene_frame_estimate.json",
        "collider_proxy_plan_path": "collider_proxy_plan.json",
        "cpu_scene_proxy_manifest_path": "cpu_scene_proxy_manifest.json",
        "cpu_preflight_scorecard_path": "cpu_preflight_scorecard.json",
        "task_anchor_proposal_manifest_path": "task_anchor_proposal_manifest.json",
        "episode_spec_manifest_path": "episode_spec_manifest.json",
        "episode_spec_path": "episode_spec.v1.json",
        "episode_specs_path": "episode_specs.json",
        "episode_setup_manifest_path": "episode_setup_manifest.json",
        "spawn_pose_validation_manifest_path": "spawn_pose_validation_manifest.json",
        "cpu_simulator_preflight_manifest_path": "cpu_simulator_preflight_manifest.json",
        "cpu_preflight_manifest_path": "cpu_preflight_manifest.json",
        "pre_gpu_readiness_summary_path": "pre_gpu_readiness_summary.json",
        "scenario_variation_instances_path": "scenario_variation_instances.json",
        "arena_environment_packet_path": "arena_environment_packet.json",
        "simulator_engine_plugin_registry_path": "simulator_engine_plugin_registry.json",
        "gpu_handoff_packet_path": "gpu_handoff_packet.json",
        "gpu_owner_system_proof_schema_path": "gpu_owner_system_proof_schema.json",
        "owner_gpu_simulator_execution_proof_manifest_path": (
            "owner_gpu_simulator_execution_proof_manifest.json"
            if owner_gpu_proof
            else None
        ),
        "local_mujoco_g1_smoke_manifest_path": (
            "mujoco_g1_local_smoke/mujoco_g1_local_smoke_manifest.json"
            if local_mujoco_g1_smoke
            else None
        ),
        "gpu_run_checklist_path": "gpu_run_checklist.md",
        "owner_gpu_simulator_execution_blocked_manifest_path": (
            "owner_gpu_simulator_execution_blocked_manifest.json"
        ),
        "asset_conversion_plan_path": "asset_conversion_plan.json",
        "simulator_execution_manifest_path": "simulator_execution_manifest.json",
        "training_orchestration_manifest_path": "training_orchestration_manifest.json",
        "proof_boundary_path": "proof_boundary.json",
        "agent_decision_ledger_path": "agent_decision_ledger.json",
        "agent_operator_status": agent_ledger.get("status"),
        "agent_operator_mode": agent_ledger.get("operator_mode"),
        "scene_asset_preflight_status": scene_preflight.get("status"),
        "episode_spec_status": episode_specs.get("status"),
        "episode_count": episode_specs.get("episode_count"),
        "cpu_simulator_preflight_status": cpu_preflight.get("status"),
        "pre_gpu_readiness_status": _read_optional_mapping(
            automation_dir / "pre_gpu_readiness_summary.json"
        ).get("status"),
        "scenario_variation_instances_status": scenario_variation_instances.get("status"),
        "scenario_variation_instance_count": scenario_variation_instances.get("instance_count"),
        "arena_environment_packet_status": arena_environment_packet.get("status"),
        "simulator_engine_plugin_registry_status": simulator_engine_plugin_registry.get("status"),
        "gpu_handoff_status": _mapping(gpu_handoff.get("packet")).get("status"),
        "owner_gpu_simulator_execution_proven": bool(
            owner_gpu_proof.get("owner_gpu_simulator_execution_proven")
        ),
        "owner_gpu_default_policy_execution_proven": bool(
            owner_gpu_proof.get("owner_gpu_default_policy_execution_proven")
        ),
        "default_sim_policy_execution_proven": bool(
            owner_gpu_proof.get("default_sim_policy_execution_proven")
        ),
        "owner_gpu_sim_robot_pov_evidence_proven": bool(
            owner_gpu_proof.get("owner_gpu_sim_robot_pov_evidence_proven")
        ),
        "isaac_sim_execution_proven": bool(owner_gpu_proof.get("isaac_sim_execution_proven")),
        "isaac_robot_asset_execution_proven": bool(
            owner_gpu_proof.get("isaac_robot_asset_execution_proven")
        ),
        "unitree_g1_asset_spawned": bool(owner_gpu_proof.get("unitree_g1_asset_spawned")),
        "mujoco_g1_asset_execution_proven": bool(
            owner_gpu_proof.get("mujoco_g1_asset_execution_proven")
        ),
        "local_mujoco_g1_asset_execution_proven": bool(
            proof_boundary.get("local_mujoco_g1_asset_execution_proven")
        ),
        "real_robot_pov_evidence_proven": False,
        "live_provider_calls_performed": False,
        "remote_asset_downloads_performed": bool(local_mujoco_g1_smoke.get("asset_source_manifest")),
        "local_cpu_preflight_smoke_ran": bool(
            _read_optional_mapping(
                automation_dir / "cpu_simulator_preflight_manifest.json"
            ).get("local_cpu_smoke_ran")
        ),
        "simulators_run": bool(run_claim_boundary.get("simulators_run")),
        "gpu_training_run": bool(training.get("gpu_training_run")),
        "messages_sent": False,
        "payments_touched": False,
        "deployments_performed": False,
        "simulator_execution_proven": bool(proof_boundary.get("simulator_execution_proven")),
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": run_claim_boundary,
    }

    write_json(automation_dir / "simulation_automation_plan.json", plan)
    write_json(automation_dir / "asset_conversion_plan.json", conversion_plan)
    write_json(automation_dir / "training_orchestration_manifest.json", training)
    write_json(automation_dir / "proof_boundary.json", proof_boundary)
    write_json(automation_dir / "agent_decision_ledger.json", agent_ledger)
    write_json(automation_dir / "simulation_automation_run_manifest.json", run_manifest)
    return {
        "schema_version": "simulation_automation_result.v1",
        "capture_root": str(context.capture_root),
        "automation_dir": str(automation_dir),
        "manifest_path": str((automation_dir / "simulation_automation_run_manifest.json").resolve()),
        "plan_path": str((automation_dir / "simulation_automation_plan.json").resolve()),
        "status": status,
        "claim_boundary": dict(run_manifest["claim_boundary"]),
    }


def _parse_simulator_commands(values: Sequence[str] | None) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for value in values or []:
        framework, sep, command = value.partition("=")
        if not sep or framework not in SIMULATOR_FRAMEWORKS or not command.strip():
            raise ValueError(
                "--simulator-command must be formatted as "
                "<isaac_sim|isaac_lab_arena|mujoco|pybullet|newton>=<command>"
            )
        commands[framework] = command.strip()
    return commands


def _agent_adapter_from_args(args: argparse.Namespace) -> SimulationAutomationAgentAdapter | None:
    if args.agent_mode == "fake":
        return FakeSimulationAutomationAgentAdapter()
    if args.agent_mode == "codex-sdk":
        return CodexSdkSimulationAutomationAgentAdapter(
            thread_id=args.codex_thread_id,
            sandbox=args.codex_sandbox,
            allow_live_operator=args.allow_live_agent_operator,
        )
    if args.agent_mode == "agents-sdk":
        return AgentsSdkCodexMCPAdapter(
            sandbox=args.codex_sandbox,
            allow_live_operator=args.allow_live_agent_operator,
        )
    return None


def _episode_agent_adapter_from_args(args: argparse.Namespace) -> EpisodeSpecAgentAdapter | None:
    if args.agent_mode == "fake":
        return FakeEpisodeSpecAgentAdapter()
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build fail-closed simulation automation manifests for a local capture package"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument(
        "--scene-asset",
        action="append",
        default=[],
        help="Optional local PLY/USD scene asset for CPU scene preflight; repeatable",
    )
    parser.add_argument(
        "--allow-cpu-simulator-preflight",
        action="store_true",
        help="Permit optional CPU MuJoCo/PyBullet smoke only when BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true is also set",
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
        "--allow-simulator-execution",
        action="store_true",
        help="Permit simulator execution only when BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true is also set",
    )
    parser.add_argument(
        "--allow-simulator",
        action="append",
        choices=SIMULATOR_FRAMEWORKS,
        default=[],
        help="Per-run simulator allowlist entry; repeat for multiple frameworks",
    )
    parser.add_argument(
        "--simulator-command",
        action="append",
        default=[],
        help="Explicit simulator command as <framework>=<command>",
    )
    parser.add_argument("--simulator-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Permit Cosmos training only when BLUEPRINT_ALLOW_COSMOS_TRAINING=true is also set",
    )
    parser.add_argument("--training-command", default=None)
    parser.add_argument("--training-timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--agent-mode",
        choices=("none", "fake", "codex-sdk", "agents-sdk"),
        default="none",
        help="Optional agent operator adapter; deterministic manifests remain authoritative",
    )
    parser.add_argument(
        "--allow-live-agent-operator",
        action="store_true",
        help=(
            f"Allow live SDK operator execution when {LIVE_AGENTS_SDK_ENV} or "
            f"{LIVE_CODEX_SDK_ENV} is true and SDK credentials or explicit Codex CLI host-OAuth exist"
        ),
    )
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write"),
        default="workspace-write",
        help="Sandbox for optional Codex SDK or Agents SDK operators",
    )
    parser.add_argument("--codex-thread-id", default=None)
    args = parser.parse_args(argv)

    try:
        result = build_simulation_automation(
            capture_root=args.capture_root,
            scene_assets=args.scene_asset,
            allow_simulator_execution=args.allow_simulator_execution,
            allowed_simulators=args.allow_simulator,
            simulator_commands=_parse_simulator_commands(args.simulator_command),
            simulator_timeout_seconds=args.simulator_timeout_seconds,
            allow_cpu_simulator_preflight=args.allow_cpu_simulator_preflight,
            cpu_preflight_backends=args.cpu_preflight_backend or CPU_BACKENDS,
            cpu_preflight_smoke_steps=args.cpu_preflight_smoke_steps,
            allow_cpu_preflight_render=args.allow_cpu_preflight_render,
            allow_training=args.allow_training,
            training_command=args.training_command,
            training_timeout_seconds=args.training_timeout_seconds,
            agent_adapter=_agent_adapter_from_args(args),
            episode_agent_adapter=_episode_agent_adapter_from_args(args),
        )
    except (OSError, ValueError) as exc:
        print(f"[simulation-automation] FAILED: {exc}")
        return 1
    print(f"[simulation-automation] manifest={result['manifest_path']}")
    print(f"[simulation-automation] plan={result['plan_path']}")
    print(f"[simulation-automation] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
