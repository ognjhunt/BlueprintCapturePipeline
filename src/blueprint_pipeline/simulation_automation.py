"""Fail-closed simulation automation orchestration lane.

This module turns existing capture/package/World Labs/Marble artifacts into a
deterministic simulation automation package. It plans conversion, simulator
execution, training, evaluation, and proof collection without calling providers,
downloading assets, running simulators, or running GPU training unless explicit
run-time approvals are present.
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
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .cpu_simulator_preflight import CPU_BACKENDS, build_cpu_simulator_preflight
from .episode_spec import EpisodeSpecAgentAdapter, FakeEpisodeSpecAgentAdapter, build_episode_specs
from .local_capture import resolve_local_capture_context
from .scene_asset_preflight import build_scene_asset_preflight


SIMULATION_AUTOMATION_SCHEMA_VERSION = "simulation_automation_plan.v1"
SIMULATION_AUTOMATION_RUN_SCHEMA_VERSION = "simulation_automation_run_manifest.v1"
ASSET_CONVERSION_PLAN_SCHEMA_VERSION = "simulation_asset_conversion_plan.v1"
SIMULATOR_EXECUTION_SCHEMA_VERSION = "simulator_execution_manifest.v1"
SIMULATOR_REQUEST_SCHEMA_VERSION = "simulator_request.v1"
SIMULATOR_RESULT_SCHEMA_VERSION = "simulator_result.v1"
TRAINING_ORCHESTRATION_SCHEMA_VERSION = "training_orchestration_manifest.v1"
PROOF_BOUNDARY_SCHEMA_VERSION = "simulation_automation_proof_boundary.v1"
AGENT_LEDGER_SCHEMA_VERSION = "simulation_automation_agent_decision_ledger.v1"
GPU_HANDOFF_PACKET_SCHEMA_VERSION = "gpu_handoff_packet.v1"
GPU_OWNER_SYSTEM_PROOF_SCHEMA_VERSION = "gpu_owner_system_proof_schema.v1"
OWNER_GPU_BLOCKED_MANIFEST_SCHEMA_VERSION = "owner_gpu_simulator_execution_blocked_manifest.v1"
OWNER_GPU_PROOF_MANIFEST_SCHEMA_VERSION = "owner_gpu_simulator_execution_proof_manifest.v1"

SIMULATOR_FRAMEWORKS = ("isaac_sim", "mujoco", "pybullet", "newton")
TRAINING_RUNNER = "blueprint_pipeline.synthesis.cosmos_lora_training.run_cosmos_lora_training"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "simulation_automation_orchestration_only",
    "repo_local_only_by_default": True,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "simulators_run": False,
    "gpu_training_run": False,
    "messages_sent": False,
    "payments_touched": False,
    "deployments_performed": False,
    "simulator_execution_proven": False,
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
    """Optional agent helper interface.

    The deterministic orchestration code owns manifest status and claim
    boundaries. Agent adapters can only add advisory planning and diagnostics.
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
            "network_required": False,
            "live_provider_calls_performed": False,
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
        }


@dataclass(frozen=True)
class CodexSdkSimulationAutomationAgentAdapter:
    """Optional Codex SDK planning adapter.

    The package name and API have changed over time, so the adapter is
    deliberately best-effort and fail-closed. When an SDK is available, this
    class records the intended start/resume request metadata for a Codex thread;
    it never mutates deterministic status or proof booleans.
    """

    thread_id: str | None = None
    sandbox: str = "read-only"

    def build_ledger(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        packages = ["openai_codex", "openai_codex_sdk", "codex_sdk"]
        installed = next((name for name in packages if importlib.util.find_spec(name)), None)
        request = {
            "action": "resume_thread" if self.thread_id else "start_thread",
            "thread_id": self.thread_id,
            "sandbox": self.sandbox if self.sandbox in {"read-only", "workspace-write"} else "read-only",
            "workspace": str(plan_context.get("repo_root") or ""),
            "prompt_purpose": "simulation_automation_advisory_planning_only",
        }
        if not installed:
            return {
                "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
                "adapter": "codex_sdk",
                "status": "blocked",
                "network_required": True,
                "optional_dependency": "openai-codex-sdk/openai-codex",
                "reason": "missing_optional_dependency",
                "request": request,
                "decisions": [],
                "diagnostics": [
                    {
                        "status": "blocked",
                        "blockers": ["missing_optional_dependency"],
                        "summary": "Codex SDK assistance was requested but no supported Codex SDK import was available.",
                    }
                ],
            }
        return {
            "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
            "adapter": "codex_sdk",
            "status": "request_manifest_ready",
            "network_required": True,
            "optional_dependency": installed,
            "request": request,
            "decisions": [
                {
                    "decision": "codex_thread_request_prepared",
                    "summary": "A Codex SDK thread request can be started or resumed by caller-owned SDK code.",
                    "owned_by": "agent_adapter",
                }
            ],
            "diagnostics": [],
        }


@dataclass(frozen=True)
class AgentsSdkCodexMCPAdapter:
    """Optional Agents SDK + Codex MCP advisory adapter."""

    sandbox: str = "read-only"

    def build_ledger(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        agents_package = next(
            (name for name in ("agents", "openai_agents") if importlib.util.find_spec(name)),
            None,
        )
        request = {
            "agent_type": "openai_agents_sdk",
            "mcp_server": "codex",
            "workspace": str(plan_context.get("repo_root") or ""),
            "sandbox": self.sandbox if self.sandbox in {"read-only", "workspace-write"} else "read-only",
            "tool_scope": [
                "propose_commands",
                "diagnose_failures",
                "summarize_traces",
                "update_next_action_plan",
            ],
        }
        if not agents_package:
            return {
                "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
                "adapter": "openai_agents_sdk_codex_mcp",
                "status": "blocked",
                "network_required": True,
                "optional_dependency": "openai-agents",
                "reason": "missing_optional_dependency",
                "request": request,
                "decisions": [],
                "diagnostics": [
                    {
                        "status": "blocked",
                        "blockers": ["missing_optional_dependency"],
                        "summary": "Agents SDK assistance was requested but the Agents SDK import was unavailable.",
                    }
                ],
            }
        return {
            "schema_version": AGENT_LEDGER_SCHEMA_VERSION,
            "adapter": "openai_agents_sdk_codex_mcp",
            "status": "request_manifest_ready",
            "network_required": True,
            "optional_dependency": agents_package,
            "request": request,
            "decisions": [
                {
                    "decision": "agents_sdk_codex_mcp_request_prepared",
                    "summary": "Agents SDK + Codex MCP advisory orchestration can be launched by caller-owned code.",
                    "owned_by": "agent_adapter",
                }
            ],
            "diagnostics": [],
        }


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


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
        "cpu_simulator_preflight_manifest": automation_dir / "cpu_simulator_preflight_manifest.json",
        "cpu_preflight_manifest": automation_dir / "cpu_preflight_manifest.json",
        "pre_gpu_readiness_summary": automation_dir / "pre_gpu_readiness_summary.json",
        "gpu_handoff_packet": automation_dir / "gpu_handoff_packet.json",
        "gpu_owner_system_proof_schema": automation_dir / "gpu_owner_system_proof_schema.json",
        "owner_gpu_simulator_execution_blocked_manifest": (
            automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json"
        ),
        "cosmos_training_export": pipeline_dir / "cosmos_training_export" / "manifest.json",
        "cosmos_lora_training": (
            pipeline_dir / "cosmos_training_export" / "training_run_manifest.json"
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
    "artifact_manifest_uri_or_path",
    "pass_fail_criteria",
    "operator_attestation",
)

OWNER_GPU_FORBIDDEN_TRUE_FIELDS = (
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
    artifact_manifest, artifact_reason, artifact_present = _read_owner_proof_json_artifact(
        proof.get("artifact_manifest_uri_or_path"),
        proof_dir=proof_dir,
    )
    for label, reason in (
        ("scene_load_trace", scene_reason),
        ("spawn_trace", spawn_reason),
        ("action_or_policy_trace", action_reason),
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
    artifact_manifest_ok = artifact_present and (
        _trace_status_ok(artifact_manifest, true_field="artifact_manifest_complete")
        or bool(artifact_manifest.get("artifacts") or artifact_manifest.get("files"))
    )
    if not scene_loaded:
        blockers.append("owner_gpu_scene_load_trace_not_proven")
    if not spawn_loaded:
        blockers.append("owner_gpu_spawn_trace_not_proven")
    if not action_trace_ok:
        blockers.append("owner_gpu_action_or_policy_trace_not_proven")
    if not artifact_manifest_ok:
        blockers.append("owner_gpu_artifact_manifest_not_proven")
    if not _attestation_ok(proof.get("operator_attestation")):
        blockers.append("owner_gpu_operator_attestation_missing_or_incomplete")
    if not _pass_fail_ok(proof.get("pass_fail_criteria")):
        blockers.append("owner_gpu_pass_fail_criteria_not_passed")

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
            "artifact_manifest_present": artifact_present,
            "artifact_manifest_valid": artifact_manifest_ok,
            "operator_attestation_present": _attestation_ok(proof.get("operator_attestation")),
            "pass_fail_criteria_passed": _pass_fail_ok(proof.get("pass_fail_criteria")),
        },
        "owner_gpu_simulator_execution_proven": accepted,
        "simulator_execution_proven": accepted,
        "scene_loaded_in_owner_simulator": accepted and scene_loaded,
        "spawn_pose_loaded": accepted and spawn_loaded,
        "robot_policy_execution_proven": False,
        "physics_contact_validated": False,
        "safety_validated": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulator_execution_proven": accepted,
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
    return {
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


def _request_for_framework(
    *,
    framework: str,
    plan: Mapping[str, Any],
    conversion_plan: Mapping[str, Any],
    command: str | None,
    generated_at: str,
) -> Dict[str, Any]:
    conversion = _mapping(_mapping(conversion_plan.get("frameworks")).get(framework))
    return {
        "schema_version": SIMULATOR_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "framework": framework,
        "scene_id": plan.get("scene_id"),
        "capture_id": plan.get("capture_id"),
        "status": "requested_if_approved",
        "command": shlex.split(command) if command else [],
        "conversion_status": conversion.get("status"),
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


def _proof_boundary(
    *,
    simulator_execution: Mapping[str, Any],
    training: Mapping[str, Any],
    owner_gpu_proof: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    simulator_proven = bool(simulator_execution.get("simulator_execution_proven"))
    owner_gpu_proven = bool(owner_gpu_proof.get("owner_gpu_simulator_execution_proven"))
    training_completed = str(training.get("status") or "") == "completed"
    return {
        "schema_version": PROOF_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "owner_gpu_simulator_execution_proven": owner_gpu_proven,
        "simulator_execution_proven": simulator_proven or owner_gpu_proven,
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
        },
        "remaining_required_evidence": [
            "real simulator load traces",
            "action logs",
            "physics/contact validation logs",
            "robot-team-owned robot assets",
            "accepted simulator or real robot trial evidence",
        ],
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulators_run": bool(simulator_execution.get("simulators_run")),
            "simulator_execution_proven": simulator_proven or owner_gpu_proven,
            "training_proof_available": training_completed,
        },
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
    if real_collider or asset_types.intersection({"urdf", "mjcf", "obj", "glb", "gltf"}):
        recommendations.append(
            {
                "backend": "mujoco",
                "priority": 2,
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
                "priority": 3,
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
            "command",
            "started_at",
            "completed_at",
            "exit_code",
            "stdout_uri_or_path",
            "stderr_uri_or_path",
            "scene_load_trace_uri_or_path",
            "spawn_pose_validation_uri_or_path",
            "action_or_policy_trace_uri_or_path",
            "artifact_manifest_uri_or_path",
            "pass_fail_criteria",
            "operator_attestation",
        ],
        "proof_booleans_allowed_true_only_with_owner_evidence": [
            "owner_system_simulator_execution_proven",
            "scene_loaded_in_owner_simulator",
            "spawn_pose_loaded",
        ],
        "proof_booleans_still_require_separate_evidence": {
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
            "- Action or policy traces are captured for any robot-eval claim.",
            "- Contact, safety, and policy success claims remain false without their own logs.",
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
    owner_gpu_proof = _read_optional_mapping(
        automation_dir / "owner_gpu_simulator_execution_proof_manifest.json"
    )
    owner_gpu_proven = bool(owner_gpu_proof.get("owner_gpu_simulator_execution_proven"))
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
        "robot_readiness_proven": False,
        "cpu_checked": pre_gpu_summary.get("cpu_checked") or [],
        "cpu_artifacts": {
            "scene_asset_inventory": "scene_asset_inventory.json",
            "scene_asset_dependency_audit": "scene_asset_dependency_audit.json",
            "collider_proxy_plan": "collider_proxy_plan.json",
            "cpu_scene_proxy_manifest": "cpu_scene_proxy_manifest.json",
            "task_anchor_proposal_manifest": "task_anchor_proposal_manifest.json",
            "episode_specs": "episode_specs.json",
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
        "recommended_backends": recommended_backends,
        "target_backend_guidance": {
            "isaac_sim": "Prefer for rich USD/OpenUSD scenes, references, materials, and future Isaac Lab workflows.",
            "mujoco": "Use for compatible MJCF or generated/proxy fixtures, not for rich USD scene proof.",
            "pybullet": "Use for URDF/proxy fixture load checks, not as rich-scene proof unless owner accepts it.",
        },
        "required_dependencies": {
            "isaac_sim": ["NVIDIA GPU", "Isaac Sim/Isaac Lab owner install", "OpenUSD dependencies"],
            "mujoco": ["mujoco Python package or owner MuJoCo install"],
            "pybullet": ["pybullet Python package or owner PyBullet install"],
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
            "owner_artifact_manifest.json",
        ],
        "timeout_seconds": simulator_timeout_seconds,
        "blockers": blockers,
        "claim_boundary": dict(CLAIM_BOUNDARY),
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
        "next_artifacts": packet["output_artifacts_expected"],
        "claim_boundary": dict(CLAIM_BOUNDARY),
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
        "network_required": False,
        "live_provider_calls_performed": False,
        "decisions": [],
        "diagnostics": [],
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

    plan = _build_plan(
        context=context,
        automation_dir=automation_dir,
        pipeline_dir=pipeline_dir,
        generated_at=generated_at,
    )
    conversion_plan = _build_asset_conversion_plan(plan=plan, generated_at=generated_at)
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
    proof_boundary = _proof_boundary(
        simulator_execution=simulator_execution,
        training=training,
        owner_gpu_proof=owner_gpu_proof,
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
        "conversion_plan": conversion_plan,
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
        "gpu_handoff_packet_path": "gpu_handoff_packet.json",
        "gpu_owner_system_proof_schema_path": "gpu_owner_system_proof_schema.json",
        "owner_gpu_simulator_execution_proof_manifest_path": (
            "owner_gpu_simulator_execution_proof_manifest.json"
            if owner_gpu_proof
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
        "scene_asset_preflight_status": scene_preflight.get("status"),
        "episode_spec_status": episode_specs.get("status"),
        "episode_count": episode_specs.get("episode_count"),
        "cpu_simulator_preflight_status": cpu_preflight.get("status"),
        "pre_gpu_readiness_status": _read_optional_mapping(
            automation_dir / "pre_gpu_readiness_summary.json"
        ).get("status"),
        "gpu_handoff_status": _mapping(gpu_handoff.get("packet")).get("status"),
        "owner_gpu_simulator_execution_proven": bool(
            owner_gpu_proof.get("owner_gpu_simulator_execution_proven")
        ),
        "live_provider_calls_performed": False,
        "remote_asset_downloads_performed": False,
        "local_cpu_preflight_smoke_ran": bool(
            _read_optional_mapping(
                automation_dir / "cpu_simulator_preflight_manifest.json"
            ).get("local_cpu_smoke_ran")
        ),
        "simulators_run": bool(simulator_execution.get("simulators_run")),
        "gpu_training_run": bool(training.get("gpu_training_run")),
        "messages_sent": False,
        "payments_touched": False,
        "deployments_performed": False,
        "simulator_execution_proven": bool(proof_boundary.get("simulator_execution_proven")),
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "simulators_run": bool(simulator_execution.get("simulators_run")),
            "gpu_training_run": bool(training.get("gpu_training_run")),
            "simulator_execution_proven": bool(proof_boundary.get("simulator_execution_proven")),
            "owner_gpu_simulator_execution_proven": bool(
                owner_gpu_proof.get("owner_gpu_simulator_execution_proven")
            ),
        },
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
                "--simulator-command must be formatted as <isaac_sim|mujoco|pybullet|newton>=<command>"
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
        )
    if args.agent_mode == "agents-sdk":
        return AgentsSdkCodexMCPAdapter(sandbox=args.codex_sandbox)
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
        help="Optional advisory agent adapter; deterministic manifests remain authoritative",
    )
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write"),
        default="read-only",
        help="Sandbox request for optional Codex SDK or Agents SDK adapter manifests",
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
