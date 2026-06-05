"""Deterministic site-eval director lane.

This module assembles real-site robot eval cards and existing simulator-review
manifests into local scenario/task simulation request manifests. Optional
Agents SDK and Codex SDK adapters only write advisory request manifests; they do
not execute agents or upgrade proof booleans.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from .common import ensure_dir, read_json_any, write_json
from .local_capture import resolve_local_capture_context


SCENARIO_EXECUTION_PLAN_SCHEMA_VERSION = "site_eval_scenario_execution_plan.v1"
TASK_SIMULATION_REQUESTS_SCHEMA_VERSION = "site_eval_task_simulation_requests.v1"
SCENARIO_SIMULATOR_MATRIX_SCHEMA_VERSION = "site_eval_scenario_simulator_matrix.v1"
AGENT_REVIEW_QUEUE_SCHEMA_VERSION = "site_eval_agent_review_queue.v1"
SITE_EVAL_DIRECTOR_RUN_SCHEMA_VERSION = "site_eval_director_run_manifest.v1"
SITE_EVAL_DIRECTOR_PROOF_BOUNDARY_SCHEMA_VERSION = "site_eval_director_proof_boundary.v1"
SITE_EVAL_DIRECTOR_BLOCKED_SCHEMA_VERSION = "site_eval_director_blocked_manifest.v1"
AGENTS_SDK_REQUEST_SCHEMA_VERSION = "agents_sdk_site_eval_director_request.v1"
CODEX_SDK_REQUEST_SCHEMA_VERSION = "codex_sdk_code_maintainer_request.v1"
NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION = "site_eval_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "site_eval_failure_labels.v1"
SITE_EVAL_PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION = "site_eval_prediction_outcome_ledger.v1"
SITE_EVAL_CALIBRATION_REPORT_SCHEMA_VERSION = "site_eval_calibration_report.v1"
FACILITY_BREAKAGE_LIBRARY_SCHEMA_VERSION = "learned_facility_breakage_library.v1"
UPDATED_EVAL_CARDS_SCHEMA_VERSION = "site_eval_updated_eval_cards.v1"
COSMOS_ORCHESTRATION_EXPORTS_SCHEMA_VERSION = "site_eval_cosmos_orchestration_exports.v1"
REAL_EVIDENCE_BLOCKED_SCHEMA_VERSION = "site_eval_real_evidence_blocked_manifest.v1"

DETERMINISTIC_DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
SIMULATOR_FRAMEWORKS = ("isaac_sim", "mujoco", "pybullet", "newton")
RUNNER_FRAMEWORKS = ("fixture", *SIMULATOR_FRAMEWORKS)
REAL_EVIDENCE_INPUTS = {
    "robot_pov": "robot_eval_inputs/robot_pov_evidence_manifest.json",
    "human_demo": "robot_eval_inputs/human_demo_evidence_manifest.json",
    "action_logs": "robot_eval_inputs/action_log_manifest.json",
    "actual_outcomes": "robot_eval_inputs/actual_outcome_manifest.json",
}
BREAKAGE_CATEGORIES = (
    "glare",
    "clutter",
    "occlusion",
    "narrow_clearance",
    "blocked_path",
    "human_crossing",
    "reflective_surface",
    "signage_perception_issue",
    "carts_forklifts_doors",
    "localization_drift",
    "manipulation_miss",
    "safety_proximity",
)

REQUIRED_ROBOT_EVAL_INPUTS = {
    "robot_eval_site_card": "robot_eval_dataset/site_card.json",
    "robot_eval_task_cards": "robot_eval_dataset/task_cards.json",
    "robot_eval_scenario_cards": "robot_eval_dataset/scenario_cards.json",
    "robot_eval_cards": "robot_eval_dataset/eval_cards.json",
    "robot_eval_proof_boundaries": "robot_eval_dataset/proof_boundaries.json",
}

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "site_eval_director_orchestration_and_review_only",
    "repo_local_only": True,
    "agents_advisory_only": True,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
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
        "owner-system simulator load and action traces",
        "owner-system robot policy or teleoperation logs",
        "physics/contact validation logs",
        "safety methodology and validation evidence",
        "actual outcome records",
        "rights/privacy clearance for the exact use",
    ],
}


class SiteEvalDirectorAdapter(Protocol):
    def build_request_manifest(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]: ...


class CodeMaintainerAdapter(Protocol):
    def build_request_manifest(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class AgentsSdkSiteEvalDirectorAdapter:
    """Optional Agents SDK site-eval coordinator request writer."""

    agents_sdk_available: bool | None = None
    openai_api_key: str | None = None
    model: str = "gpt-4.1"

    def build_request_manifest(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
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
        blockers: List[str] = []
        if not agents_available:
            blockers.append("missing_openai_agents_sdk")
        if not api_key_present:
            blockers.append("missing_openai_api_key")
        status = "blocked" if blockers else "request_manifest_ready"
        return {
            "schema_version": AGENTS_SDK_REQUEST_SCHEMA_VERSION,
            "adapter": "openai_agents_sdk_site_eval_director",
            "status": status,
            "blockers": blockers,
            "missing_inputs": list(blockers),
            "execution_performed": False,
            "network_required_if_executed": True,
            "agent_authority": "advisory_only",
            "request": {
                "purpose": "site_eval_workflow_coordination_advisory_only",
                "model": self.model,
                "capture_root": str(plan_context.get("capture_root") or ""),
                "scenario_execution_plan": "scenario_execution_plan.json",
                "task_simulation_requests": "task_simulation_requests.json",
                "scenario_simulator_matrix": "scenario_simulator_matrix.json",
                "allowed_outputs": [
                    "review comments",
                    "missing-proof diagnostics",
                    "next-action plan",
                ],
                "prohibited_outputs": [
                    "robot readiness proof",
                    "simulator execution proof",
                    "safety validation proof",
                    "public claim upgrade",
                ],
            },
            "attempted_commands": [],
            "evidence": {
                "openai_agents_sdk_available": bool(agents_available),
                "openai_api_key_present": api_key_present,
            },
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }


@dataclass(frozen=True)
class CodexSdkCodeMaintainerAdapter:
    """Optional Codex coding specialist request writer."""

    codex_sdk_available: bool | None = None
    openai_api_key: str | None = None
    codex_mcp_server_available: bool | None = None
    codex_cli_path: str | None = None
    sandbox: str = "read-only"

    def build_request_manifest(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        sdk_available = (
            self.codex_sdk_available
            if self.codex_sdk_available is not None
            else _module_available(("openai_codex", "openai_codex_sdk", "codex_sdk"))
        )
        api_key_present = bool(
            _string(self.openai_api_key)
            if self.openai_api_key is not None
            else _string(os.getenv("OPENAI_API_KEY"))
        )
        codex_cli = _string(self.codex_cli_path) or shutil.which("codex")
        mcp_available = (
            self.codex_mcp_server_available
            if self.codex_mcp_server_available is not None
            else _codex_mcp_server_available(codex_cli)
        )
        blockers: List[str] = []
        if not sdk_available:
            blockers.append("missing_codex_sdk")
        if not api_key_present:
            blockers.append("missing_openai_api_key")
        if not mcp_available:
            blockers.append("missing_codex_mcp_server")
        status = "blocked" if blockers else "request_manifest_ready"
        sandbox = self.sandbox if self.sandbox in {"read-only", "workspace-write"} else "read-only"
        return {
            "schema_version": CODEX_SDK_REQUEST_SCHEMA_VERSION,
            "adapter": "codex_sdk_code_maintainer",
            "status": status,
            "blockers": blockers,
            "missing_inputs": list(blockers),
            "execution_performed": False,
            "network_required_if_executed": True,
            "agent_authority": "advisory_only",
            "request": {
                "purpose": "implementation_diagnosis_and_code_fix_advisory_only",
                "capture_root": str(plan_context.get("capture_root") or ""),
                "workspace": str(plan_context.get("repo_root") or ""),
                "sandbox": sandbox,
                "mcp_server_command": ["codex", "mcp-server"],
                "allowed_request_types": [
                    "implementation_diagnosis",
                    "code_fix_patch_plan",
                ],
                "prohibited_request_types": [
                    "site_eval_business_decision",
                    "simulator_execution",
                    "live_provider_call",
                    "proof_or_readiness_claim_upgrade",
                    "deployment_or_payment_action",
                ],
            },
            "attempted_commands": [],
            "evidence": {
                "codex_sdk_available": bool(sdk_available),
                "openai_api_key_present": api_key_present,
                "codex_cli_path": codex_cli,
                "codex_mcp_server_available": bool(mcp_available),
            },
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }


def _module_available(candidates: Sequence[str]) -> bool:
    return any(importlib.util.find_spec(candidate) is not None for candidate in candidates)


def _codex_mcp_server_available(codex_cli: str | None) -> bool:
    if not codex_cli:
        return False
    try:
        completed = subprocess.run(
            [codex_cli, "mcp-server", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    output = f"{completed.stdout}\n{completed.stderr}".lower()
    return completed.returncode == 0 or "mcp-server" in output


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
    seen: set[str] = set()
    out: List[str] = []
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def _number(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _json_file_present(pipeline_dir: Path, relative_path: str) -> bool:
    return (pipeline_dir / relative_path).is_file()


def _timestamp(*payloads: Mapping[str, Any]) -> str:
    for payload in payloads:
        for key in ("updated_at", "generated_at", "completed_at", "created_at"):
            text = _string(payload.get(key))
            if text:
                return text
    return DETERMINISTIC_DEFAULT_GENERATED_AT


@dataclass(frozen=True)
class NormalizedAttempt:
    attempt: Dict[str, Any]
    blocked: Dict[str, Any] | None = None


class SimulatorRunner(Protocol):
    framework: str

    def run(
        self,
        *,
        context: Any,
        pipeline_dir: Path,
        automation_dir: Path,
        scenario_plan: Mapping[str, Any],
        generated_at: str,
    ) -> NormalizedAttempt: ...


@dataclass(frozen=True)
class FixtureSimulatorRunner:
    """Local deterministic runner backed by fixture attempt records."""

    framework: str = "fixture"
    fixture_relative_path: str = "robot_eval_inputs/headless_fixture_attempts.json"

    def run(
        self,
        *,
        context: Any,
        pipeline_dir: Path,
        automation_dir: Path,
        scenario_plan: Mapping[str, Any],
        generated_at: str,
    ) -> NormalizedAttempt:
        del automation_dir
        fixture_path = pipeline_dir / self.fixture_relative_path
        scenarios = [
            dict(item)
            for item in scenario_plan.get("scenarios", []) or []
            if isinstance(item, Mapping)
        ]
        scenario_ids = {_string(item.get("scenario_id")) for item in scenarios}
        payload = _read_optional_mapping(fixture_path)
        raw_attempts = payload.get("attempts") if payload else []
        fixture_attempts = [item for item in raw_attempts or [] if isinstance(item, Mapping)]
        accepted_scenarios = {
            _string(scenario.get("scenario_id"))
            for scenario in scenarios
            if set(_string_list(scenario.get("agent_inferred_components"))).issubset(
                {"edge_case"}
            )
            or _string(scenario.get("review_status")) == "accepted"
        }
        blockers: List[str] = []
        if not fixture_path.is_file():
            blockers.append("missing_fixture_attempts")
        if not accepted_scenarios and scenarios:
            blockers.append("generated_or_inferred_scenarios_require_review")
        normalized: List[Dict[str, Any]] = []
        for index, item in enumerate(fixture_attempts):
            scenario_id = _string(item.get("scenario_id"))
            if scenario_id not in scenario_ids:
                continue
            if scenario_id not in accepted_scenarios:
                continue
            attempt_id = _string(item.get("attempt_id")) or f"fixture_attempt_{index + 1}"
            metrics = _mapping(item.get("metrics"))
            safety_events = item.get("safety_events")
            contact_trace = item.get("contact_trace")
            action_trace = item.get("action_trace")
            failure_mode_ids = _string_list(item.get("failure_mode_ids"))
            if not failure_mode_ids and bool(item.get("success")) is False:
                failure_mode_ids = _infer_failure_modes(
                    metrics=metrics,
                    safety_events=safety_events,
                    contact_trace=contact_trace,
                )
            normalized.append(
                {
                    "attempt_id": attempt_id,
                    "scenario_id": scenario_id,
                    "task_id": _string(item.get("task_id")),
                    "policy_id": _string(item.get("policy_id")) or "fixture_policy",
                    "engine": "fixture",
                    "runner": "fixture",
                    "status": "completed",
                    "success": bool(item.get("success")),
                    "outcome_source": "fixture_outcome",
                    "predicted_success": item.get("predicted_success"),
                    "predicted_cycle_time_seconds": item.get(
                        "predicted_cycle_time_seconds"
                    ),
                    "predicted_intervention_count": item.get(
                        "predicted_intervention_count"
                    ),
                    "predicted_safety_event_count": item.get(
                        "predicted_safety_event_count"
                    ),
                    "metrics": {
                        "cycle_time_seconds": _number(
                            metrics.get("cycle_time_seconds"),
                            default=_number(item.get("cycle_time_seconds")),
                        ),
                        "intervention_count": _int(
                            metrics.get("intervention_count"),
                            default=_int(item.get("intervention_count")),
                        ),
                        "contact_event_count": _int(
                            metrics.get("contact_event_count"),
                            default=_int(item.get("contact_event_count")),
                        ),
                        "safety_event_count": _int(
                            metrics.get("safety_event_count"),
                            default=_int(item.get("safety_event_count")),
                        ),
                    },
                    "action_trace": action_trace if isinstance(action_trace, list) else [],
                    "contact_trace": contact_trace if isinstance(contact_trace, list) else [],
                    "timing_metrics": _mapping(item.get("timing_metrics")),
                    "safety_events": safety_events if isinstance(safety_events, list) else [],
                    "artifact_paths": _mapping(item.get("artifact_paths")),
                    "failure_mode_ids": failure_mode_ids,
                    "breakage_categories": _string_list(item.get("breakage_categories")),
                    "label_review_status": _string(item.get("label_review_status"))
                    or "automatic",
                    "owner_system": _string(item.get("owner_system"))
                    or "BlueprintCapturePipeline.fixture",
                    "provenance": {
                        "fixture_manifest": self.fixture_relative_path,
                        "fixture_record_index": index,
                        "generated_at": generated_at,
                    },
                    "claim_boundary": "fixture_attempt_proves_local_loop_only_not_real_simulator_or_robot_execution",
                }
            )
        if fixture_path.is_file() and not normalized:
            blockers.append("fixture_attempts_did_not_match_review_accepted_scenarios")
        blocked_manifest = None
        status = "completed" if normalized and not blockers else "blocked"
        if blockers:
            blocked_manifest = {
                "schema_version": SITE_EVAL_DIRECTOR_BLOCKED_SCHEMA_VERSION,
                "generated_at": generated_at,
                "scene_id": context.scene_id,
                "capture_id": context.capture_id,
                "status": "blocked",
                "blockers": blockers,
                "missing_inputs": []
                if fixture_path.is_file()
                else ["robot_eval_inputs/headless_fixture_attempts"],
                "attempted_commands": ["fixture_simulator_runner"],
                "evidence": {
                    "fixture_path": str(fixture_path),
                    "scenario_ids": sorted(scenario_ids),
                    "accepted_scenario_ids": sorted(accepted_scenarios),
                },
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
        return NormalizedAttempt(
            attempt={
                "schema_version": NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "scene_id": context.scene_id,
                "capture_id": context.capture_id,
                "status": status,
                "runner": "fixture",
                "attempt_count": len(normalized),
                "attempts": sorted(
                    normalized,
                    key=lambda item: (
                        _string(item.get("scenario_id")),
                        _string(item.get("attempt_id")),
                    ),
                ),
                "simulators_run": False,
                "fixture_runner_executed": bool(normalized),
                "real_simulator_execution_proven": False,
                "claim_boundary": dict(CLAIM_BOUNDARY),
            },
            blocked=blocked_manifest,
        )


def _source_artifacts(*, automation_dir: Path, pipeline_dir: Path) -> Dict[str, str]:
    candidates = {
        "robot_eval_site_card": pipeline_dir / "robot_eval_dataset" / "site_card.json",
        "robot_eval_task_cards": pipeline_dir / "robot_eval_dataset" / "task_cards.json",
        "robot_eval_scenario_cards": pipeline_dir / "robot_eval_dataset" / "scenario_cards.json",
        "robot_eval_cards": pipeline_dir / "robot_eval_dataset" / "eval_cards.json",
        "robot_eval_proof_boundaries": (
            pipeline_dir / "robot_eval_dataset" / "proof_boundaries.json"
        ),
        "worldlabs_request_manifest": pipeline_dir / "worldlabs_request_manifest.json",
        "worldlabs_operation_manifest": pipeline_dir / "worldlabs_operation_manifest.json",
        "worldlabs_world_manifest": pipeline_dir / "worldlabs_world_manifest.json",
        "marble_simready_bridge": (
            pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json"
        ),
        "marble_asset_validation": (
            pipeline_dir / "marble_sim_assets" / "marble_asset_validation.json"
        ),
        "simready_scene_manifest": pipeline_dir / "simready" / "simready_scene_manifest.json",
        "simready_validation": pipeline_dir / "simready" / "simready_validation.json",
        "simulation_automation_plan": (
            pipeline_dir / "simulation_automation" / "simulation_automation_plan.json"
        ),
        "simulation_automation_run_manifest": (
            pipeline_dir / "simulation_automation" / "simulation_automation_run_manifest.json"
        ),
        "asset_conversion_plan": (
            pipeline_dir / "simulation_automation" / "asset_conversion_plan.json"
        ),
        "simulator_execution_manifest": (
            pipeline_dir / "simulation_automation" / "simulator_execution_manifest.json"
        ),
        "training_orchestration_manifest": (
            pipeline_dir / "simulation_automation" / "training_orchestration_manifest.json"
        ),
        "simulation_automation_proof_boundary": (
            pipeline_dir / "simulation_automation" / "proof_boundary.json"
        ),
    }
    return {
        key: rel
        for key, path in sorted(candidates.items())
        if (rel := _relative_if_file(automation_dir, path))
    }


def _missing_required_inputs(pipeline_dir: Path) -> List[str]:
    return [
        key
        for key, relative_path in REQUIRED_ROBOT_EVAL_INPUTS.items()
        if not (pipeline_dir / relative_path).is_file()
    ]


def _blocked_manifest(
    *,
    context: Any,
    source_artifacts: Mapping[str, str],
    missing_inputs: Sequence[str],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SITE_EVAL_DIRECTOR_BLOCKED_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "blocked",
        "blockers": ["missing_robot_eval_dataset_cards"],
        "missing_inputs": list(missing_inputs),
        "attempted_commands": ["build_site_eval_director"],
        "evidence": {
            "required_inputs": dict(REQUIRED_ROBOT_EVAL_INPUTS),
            "source_artifacts": dict(source_artifacts),
            "capture_root": str(context.capture_root),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _write_blocked_outputs(
    *,
    context: Any,
    automation_dir: Path,
    source_artifacts: Mapping[str, str],
    missing_inputs: Sequence[str],
    generated_at: str,
) -> Dict[str, Any]:
    blocked = _blocked_manifest(
        context=context,
        source_artifacts=source_artifacts,
        missing_inputs=missing_inputs,
        generated_at=generated_at,
    )
    common_blocked = {
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "blocked",
        "blockers": list(blocked["blockers"]),
        "missing_inputs": list(missing_inputs),
        "attempted_commands": ["build_site_eval_director"],
        "evidence": dict(blocked["evidence"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    scenario_plan = {
        "schema_version": SCENARIO_EXECUTION_PLAN_SCHEMA_VERSION,
        **common_blocked,
        "scenario_count": 0,
        "scenarios": [],
    }
    task_requests = {
        "schema_version": TASK_SIMULATION_REQUESTS_SCHEMA_VERSION,
        **common_blocked,
        "task_request_count": 0,
        "requests": [],
    }
    matrix = {
        "schema_version": SCENARIO_SIMULATOR_MATRIX_SCHEMA_VERSION,
        **common_blocked,
        "frameworks": list(SIMULATOR_FRAMEWORKS),
        "matrix": [],
    }
    review_queue = {
        "schema_version": AGENT_REVIEW_QUEUE_SCHEMA_VERSION,
        **common_blocked,
        "items": [
            {
                "review_id": "missing_robot_eval_dataset_cards",
                "status": "blocked",
                "reason": "missing_robot_eval_dataset_cards",
                "missing_inputs": list(missing_inputs),
            }
        ],
    }
    proof_boundary = {
        "schema_version": SITE_EVAL_DIRECTOR_PROOF_BOUNDARY_SCHEMA_VERSION,
        **common_blocked,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "robot_policy_execution_proven": False,
        "physics_contact_validated": False,
        "safety_validated": False,
        "training_completed": False,
        "public_claim_upgrade_allowed": False,
    }
    run_manifest = {
        "schema_version": SITE_EVAL_DIRECTOR_RUN_SCHEMA_VERSION,
        **common_blocked,
        "scenario_execution_plan_path": "scenario_execution_plan.json",
        "task_simulation_requests_path": "task_simulation_requests.json",
        "scenario_simulator_matrix_path": "scenario_simulator_matrix.json",
        "agent_review_queue_path": "agent_review_queue.json",
        "proof_boundary_path": "site_eval_director_proof_boundary.json",
        "blocked_manifest_path": "site_eval_director_blocked_manifest.json",
        "agent_request_manifests": {},
        "headless_loop_artifacts": {},
        "live_provider_calls_performed": False,
        "remote_asset_downloads_performed": False,
        "simulators_run": False,
        "gpu_training_run": False,
        "messages_sent": False,
        "payments_touched": False,
        "deployments_performed": False,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
    }
    write_json(automation_dir / "scenario_execution_plan.json", scenario_plan)
    write_json(automation_dir / "task_simulation_requests.json", task_requests)
    write_json(automation_dir / "scenario_simulator_matrix.json", matrix)
    write_json(automation_dir / "agent_review_queue.json", review_queue)
    write_json(automation_dir / "site_eval_director_proof_boundary.json", proof_boundary)
    write_json(automation_dir / "site_eval_director_run_manifest.json", run_manifest)
    write_json(automation_dir / "site_eval_director_blocked_manifest.json", blocked)
    return run_manifest


def _cards(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        return []
    return [dict(item) for item in raw_cards if isinstance(item, Mapping)]


def _agent_inferred_components(scenario_card: Mapping[str, Any]) -> List[str]:
    components: set[str] = set()
    labels = _mapping(scenario_card.get("observed_vs_inferred_labels"))
    for key, value in labels.items():
        if "agent_inferred" in _string(value):
            components.add(_string(key))
    for key in ("normal_scenario", "variation", "edge_case"):
        status = _string(_mapping(scenario_card.get(key)).get("ground_truth_status"))
        if "agent_inferred" in status:
            components.add(key)
    return sorted(component for component in components if component)


def _eval_cards_by_scenario(eval_cards: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for card in eval_cards:
        scenario_id = _string(card.get("scenario_id"))
        if not scenario_id:
            continue
        grouped.setdefault(scenario_id, []).append(dict(card))
    return grouped


def _framework_statuses(asset_conversion_plan: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    frameworks = _mapping(asset_conversion_plan.get("frameworks"))
    statuses: Dict[str, Dict[str, Any]] = {}
    for framework in SIMULATOR_FRAMEWORKS:
        payload = _mapping(frameworks.get(framework))
        statuses[framework] = {
            "status": _string(payload.get("status")) or "missing_simulation_automation_plan",
            "blockers": _string_list(payload.get("blockers")),
            "input_assets": _mapping(payload.get("input_assets")),
            "target_format": payload.get("target_format"),
        }
    return statuses


def _real_engine_blockers(
    *, allow_simulator_execution: bool, allowed_simulators: Sequence[str]
) -> List[Dict[str, Any]]:
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
    allowed = {_string(item) for item in allowed_simulators}
    blockers: List[Dict[str, Any]] = []
    for framework in SIMULATOR_FRAMEWORKS:
        framework_blockers: List[str] = []
        if not env_allowed:
            framework_blockers.append("missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
        if not allow_simulator_execution:
            framework_blockers.append("missing_cli_allow_simulator_execution")
        if framework not in allowed:
            framework_blockers.append(f"missing_cli_allow_simulator_{framework}")
        blockers.append(
            {
                "framework": framework,
                "status": "blocked",
                "blockers": framework_blockers,
                "execution_performed": False,
                "claim_boundary": "real_simulator_engine_fail_closed_until_env_and_cli_gates_are_present",
            }
        )
    return blockers


def _infer_failure_modes(
    *,
    metrics: Mapping[str, Any],
    safety_events: Any,
    contact_trace: Any,
) -> List[str]:
    modes: List[str] = []
    if _int(metrics.get("intervention_count")) > 0:
        modes.append("failure_intervention_required")
    if _int(metrics.get("safety_event_count")) > 0 or (
        isinstance(safety_events, list) and safety_events
    ):
        modes.append("failure_safety_threshold_violation")
    if _int(metrics.get("contact_event_count")) > 0 or (
        isinstance(contact_trace, list) and contact_trace
    ):
        modes.append("failure_contact_collision")
    if not modes:
        modes.append("failure_task_not_attempted")
    return modes


def _failure_labels(
    *,
    context: Any,
    normalized_trace: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    labels: List[Dict[str, Any]] = []
    for attempt in normalized_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        failure_ids = _string_list(attempt.get("failure_mode_ids"))
        if bool(attempt.get("success")) and not failure_ids:
            label_status = "success"
        elif _string(attempt.get("label_review_status")) == "human_reviewed":
            label_status = "human_reviewed"
        elif failure_ids:
            label_status = "automatic"
        else:
            label_status = "review_required"
            failure_ids = ["failure_evidence_missing"]
        labels.append(
            {
                "attempt_id": _string(attempt.get("attempt_id")),
                "scenario_id": _string(attempt.get("scenario_id")),
                "task_id": _string(attempt.get("task_id")),
                "label_status": label_status,
                "success": bool(attempt.get("success")),
                "failure_mode_ids": failure_ids,
                "label_source": "fixture_outcome"
                if label_status in {"success", "automatic"}
                else label_status,
                "review_required": label_status == "review_required",
                "claim_boundary": "failure_label_is_attempt_local_not_deployment_readiness",
            }
        )
    return {
        "schema_version": FAILURE_LABELS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "labeled" if labels else "blocked",
        "label_count": len(labels),
        "labels": sorted(labels, key=lambda item: item["attempt_id"]),
        "taxonomy_source": "../robot_eval_dataset/failure_taxonomy.json",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _calibration_rows(normalized_trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for attempt in normalized_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        metrics = _mapping(attempt.get("metrics"))
        predicted_success_raw = attempt.get("predicted_success")
        predicted_success = (
            bool(predicted_success_raw) if predicted_success_raw is not None else None
        )
        actual_success = bool(attempt.get("success"))
        predicted_cycle = attempt.get("predicted_cycle_time_seconds")
        predicted_interventions = attempt.get("predicted_intervention_count")
        predicted_safety = attempt.get("predicted_safety_event_count")
        cycle_time = _number(metrics.get("cycle_time_seconds"))
        intervention_count = _int(metrics.get("intervention_count"))
        contact_count = _int(metrics.get("contact_event_count"))
        safety_count = _int(metrics.get("safety_event_count"))
        rows.append(
            {
                "record_id": f"site_eval_{_string(attempt.get('attempt_id'))}",
                "attempt_id": _string(attempt.get("attempt_id")),
                "scenario_id": _string(attempt.get("scenario_id")),
                "task_id": _string(attempt.get("task_id")),
                "policy_id": _string(attempt.get("policy_id")),
                "engine": _string(attempt.get("engine")) or "fixture",
                "prediction_source": "fixture_prediction",
                "actual_source": _string(attempt.get("outcome_source")) or "fixture_outcome",
                "predicted_success": predicted_success,
                "actual_success": actual_success,
                "success_delta": None
                if predicted_success is None
                else int(actual_success) - int(predicted_success),
                "cycle_time_error_seconds": None
                if predicted_cycle is None
                else cycle_time - _number(predicted_cycle),
                "intervention_delta": None
                if predicted_interventions is None
                else intervention_count - _int(predicted_interventions),
                "safety_event_delta": None
                if predicted_safety is None
                else safety_count - _int(predicted_safety),
                "contact_event_delta": contact_count,
                "actual_metrics": {
                    "cycle_time_seconds": cycle_time,
                    "intervention_count": intervention_count,
                    "contact_event_count": contact_count,
                    "safety_event_count": safety_count,
                },
                "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
                "uncertainty": "low_fixture_deterministic"
                if predicted_success is not None
                else "prediction_missing",
                "owner_system": _string(attempt.get("owner_system")),
                "proof_artifact_paths": _mapping(attempt.get("artifact_paths")),
                "claim_boundary": "fixture_calibration_is_local_loop_evidence_only",
            }
        )
    return sorted(rows, key=lambda item: item["record_id"])


def _group_average(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _aggregate_calibration(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        group = tuple(_string(row.get(key)) for key in group_keys)
        grouped.setdefault(group, []).append(row)
    aggregates: List[Dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        aggregates.append(
            {
                **{key: value for key, value in zip(group_keys, group)},
                "record_count": len(group_rows),
                "actual_success_rate": sum(
                    1 for row in group_rows if bool(row.get("actual_success"))
                )
                / len(group_rows),
                "mean_success_delta": _group_average(group_rows, "success_delta"),
                "mean_cycle_time_error_seconds": _group_average(
                    group_rows, "cycle_time_error_seconds"
                ),
                "mean_intervention_delta": _group_average(group_rows, "intervention_delta"),
                "mean_safety_event_delta": _group_average(group_rows, "safety_event_delta"),
                "mean_contact_event_delta": _group_average(group_rows, "contact_event_delta"),
            }
        )
    return aggregates


def _prediction_outcome_and_calibration(
    *,
    context: Any,
    site_card: Mapping[str, Any],
    normalized_trace: Mapping[str, Any],
    generated_at: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    rows = _calibration_rows(normalized_trace)
    for record in rows:
        record["site_id"] = site_card.get("site_id")
    ledger = {
        "schema_version": SITE_EVAL_PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_id": site_card.get("site_id"),
        "status": "calibrated" if rows else "blocked",
        "record_count": len(rows),
        "records": rows,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    calibration = {
        "schema_version": SITE_EVAL_CALIBRATION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_id": site_card.get("site_id"),
        "status": "calibrated" if rows else "blocked",
        "record_count": len(rows),
        "aggregates": {
            "by_site": _aggregate_calibration(rows, ["site_id"])
            if site_card.get("site_id")
            else [],
            "by_task": _aggregate_calibration(rows, ["task_id"]),
            "by_scenario": _aggregate_calibration(rows, ["scenario_id"]),
            "by_policy": _aggregate_calibration(rows, ["policy_id"]),
            "by_engine": _aggregate_calibration(rows, ["engine"]),
        },
        "uncertainty_policy": "fixture_predictions_missing_or_local_only_do_not_support_public_threshold_claims",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return ledger, calibration


def _breakage_categories_from_attempt(attempt: Mapping[str, Any]) -> List[str]:
    categories = set(_string_list(attempt.get("breakage_categories")))
    modes = set(_string_list(attempt.get("failure_mode_ids")))
    if "failure_navigation_blocked" in modes:
        categories.add("blocked_path")
    if "failure_localization_or_pose_drift" in modes:
        categories.add("localization_drift")
    if "failure_manipulation_miss" in modes:
        categories.add("manipulation_miss")
    if "failure_perception_occlusion" in modes:
        categories.add("occlusion")
    if "failure_safety_threshold_violation" in modes:
        categories.add("safety_proximity")
    if "failure_contact_collision" in modes:
        categories.add("narrow_clearance")
    return sorted(category for category in categories if category in BREAKAGE_CATEGORIES)


def _facility_breakage_library(
    *,
    context: Any,
    site_card: Mapping[str, Any],
    normalized_trace: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    counts = {category: 0 for category in BREAKAGE_CATEGORIES}
    for attempt in normalized_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping) or bool(attempt.get("success")):
            continue
        categories = _breakage_categories_from_attempt(attempt)
        for category in categories:
            counts[category] += 1
        records.append(
            {
                "breakage_record_id": f"breakage_{_string(attempt.get('attempt_id'))}",
                "site_id": site_card.get("site_id"),
                "scenario_id": _string(attempt.get("scenario_id")),
                "task_id": _string(attempt.get("task_id")),
                "attempt_id": _string(attempt.get("attempt_id")),
                "categories": categories,
                "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
                "source": "fixture_attempt",
                "repeat_count": 1,
                "claim_boundary": "facility_breakage_record_is_learned_local_failure_pattern_not_readiness_claim",
            }
        )
    return {
        "schema_version": FACILITY_BREAKAGE_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_id": site_card.get("site_id"),
        "status": "updated" if records else "empty",
        "category_counts": counts,
        "record_count": len(records),
        "records": sorted(records, key=lambda item: item["breakage_record_id"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _updated_eval_cards(
    *,
    context: Any,
    eval_cards: Sequence[Mapping[str, Any]],
    normalized_trace: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    attempts_by_scenario: Dict[str, List[Mapping[str, Any]]] = {}
    for attempt in normalized_trace.get("attempts", []) or []:
        if isinstance(attempt, Mapping):
            attempts_by_scenario.setdefault(_string(attempt.get("scenario_id")), []).append(attempt)
    cards: List[Dict[str, Any]] = []
    for card in eval_cards:
        scenario_id = _string(card.get("scenario_id"))
        attempts = attempts_by_scenario.get(scenario_id, [])
        successes = [bool(attempt.get("success")) for attempt in attempts]
        updated = dict(card)
        updated["site_eval_director_update"] = {
            "status": "fixture_outcome_attached" if attempts else "needs_actual_outcome",
            "attempt_count": len(attempts),
            "fixture_success_count": sum(1 for success in successes if success),
            "fixture_failure_count": sum(1 for success in successes if not success),
            "validation_status": "fixture_validated_local_loop"
            if attempts
            else "needs_actual_outcome",
            "proof_boundary": "fixture_outcome_updates_eval_card_without_upgrading_robot_or_public_claims",
        }
        updated["blocked_upgrades"] = sorted(
            set(_string_list(updated.get("blocked_upgrades")))
            | {
                "real_simulator_execution_completed",
                "robot_policy_execution_proven",
                "public_claim_upgrade",
            }
        )
        cards.append(updated)
    return {
        "schema_version": UPDATED_EVAL_CARDS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "updated" if cards else "blocked",
        "eval_card_count": len(cards),
        "cards": sorted(cards, key=lambda item: _string(item.get("eval_card_id"))),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _cosmos_orchestration_exports(
    *,
    context: Any,
    scenario_plan: Mapping[str, Any],
    normalized_trace: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    breakage_library: Mapping[str, Any],
    generated_at: str,
    allow_training: bool,
) -> Dict[str, Any]:
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_COSMOS_TRAINING")
    training_allowed = env_allowed and allow_training
    return {
        "schema_version": COSMOS_ORCHESTRATION_EXPORTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "request_manifests_ready",
        "scenario_generation_request": {
            "status": "review_required",
            "scenario_ids": [
                _string(item.get("scenario_id"))
                for item in scenario_plan.get("scenarios", []) or []
                if isinstance(item, Mapping)
            ],
            "output_requires_human_acceptance": True,
        },
        "failure_mining_export": {
            "status": "export_manifest_ready",
            "source_attempt_trace": "normalized_simulator_attempt_trace.json",
            "source_failure_labels": "failure_labels.json",
            "breakage_record_count": int(breakage_library.get("record_count") or 0),
        },
        "post_training_dataset_manifest": {
            "status": "blocked" if not training_allowed else "request_manifest_ready",
            "blockers": []
            if training_allowed
            else [
                "missing_env_BLUEPRINT_ALLOW_COSMOS_TRAINING",
                "missing_cli_allow_training",
            ],
            "training_completed": False,
        },
        "policy_eval_data_export": {
            "status": "export_manifest_ready",
            "attempt_count": int(normalized_trace.get("attempt_count") or 0),
            "calibration_record_count": int(calibration_report.get("record_count") or 0),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _real_evidence_blocked_manifest(
    *,
    context: Any,
    pipeline_dir: Path,
    generated_at: str,
) -> Dict[str, Any] | None:
    missing = [
        key
        for key, relative_path in REAL_EVIDENCE_INPUTS.items()
        if not _json_file_present(pipeline_dir, relative_path)
    ]
    if not missing:
        return None
    return {
        "schema_version": REAL_EVIDENCE_BLOCKED_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "blocked",
        "blockers": ["missing_real_robot_or_owner_system_evidence"],
        "missing_inputs": missing,
        "required_inputs": dict(REAL_EVIDENCE_INPUTS),
        "attempted_commands": ["build_site_eval_director"],
        "evidence": {
            key: {
                "relative_path": relative_path,
                "present": _json_file_present(pipeline_dir, relative_path),
            }
            for key, relative_path in REAL_EVIDENCE_INPUTS.items()
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _rights_privacy_blocked(site_card: Mapping[str, Any]) -> bool:
    rights = _mapping(
        _mapping(site_card.get("provenance_rights_review_status")).get("rights_privacy")
    )
    return bool(rights.get("blocked")) or _string(rights.get("rights_status")) in {
        "missing",
        "blocked",
        "not_allowed",
        "permission_required",
        "failed",
    }


def _simulator_execution_status(
    *, framework: str, simulator_execution_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    for result in simulator_execution_manifest.get("simulator_results", []) or []:
        if isinstance(result, Mapping) and _string(result.get("framework")) == framework:
            return {
                "status": _string(result.get("status")) or "missing",
                "reason": result.get("reason"),
                "simulator_execution_proven": bool(result.get("simulator_execution_proven")),
            }
    return {
        "status": _string(simulator_execution_manifest.get("overall_status")) or "missing",
        "reason": None,
        "simulator_execution_proven": False,
    }


def _scenario_execution_plan(
    *,
    context: Any,
    source_artifacts: Mapping[str, str],
    site_card: Mapping[str, Any],
    task_cards: Sequence[Mapping[str, Any]],
    scenario_cards: Sequence[Mapping[str, Any]],
    eval_cards: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    tasks_by_id = {_string(card.get("task_id")): dict(card) for card in task_cards}
    eval_by_scenario = _eval_cards_by_scenario(eval_cards)
    scenarios: List[Dict[str, Any]] = []
    for scenario in scenario_cards:
        scenario_id = _string(scenario.get("scenario_id"))
        task_id = _string(scenario.get("task_id"))
        components = _agent_inferred_components(scenario)
        missing_annotations = _string_list(scenario.get("required_missing_annotations"))
        related_eval_cards = eval_by_scenario.get(scenario_id, [])
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_card_id": _string(scenario.get("scenario_card_id")),
                "task_id": task_id,
                "task_card_id": _string(tasks_by_id.get(task_id, {}).get("task_card_id")),
                "robot_profile_id": _string(scenario.get("robot_profile_id")),
                "site_id": site_card.get("site_id"),
                "execution_mode": "review_only",
                "status": "planned_for_review",
                "requires_human_review": bool(components or missing_annotations),
                "agent_inferred_components": components,
                "required_missing_annotations": missing_annotations,
                "eval_card_ids": [
                    _string(card.get("eval_card_id")) for card in related_eval_cards if card
                ],
                "prediction_sources": [
                    _string(card.get("prediction_source")) for card in related_eval_cards if card
                ],
                "simulator_execution_requested": False,
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
                "proof_owner": "deterministic_site_eval_director",
                "source_artifacts": dict(source_artifacts),
                "claim_boundary": "scenario_execution_plan_is_review_scope_not_simulator_or_robot_result",
            }
        )
    scenarios = sorted(scenarios, key=lambda item: item["scenario_id"])
    payload = {
        "schema_version": SCENARIO_EXECUTION_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "review_ready" if scenarios else "blocked",
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "source_artifacts": dict(source_artifacts),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    payload["deterministic_fingerprint"] = _sha_payload(
        {
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "scenario_ids": [scenario["scenario_id"] for scenario in scenarios],
            "task_ids": [scenario["task_id"] for scenario in scenarios],
            "source_artifacts": dict(source_artifacts),
        }
    )
    return payload


def _task_simulation_requests(
    *,
    context: Any,
    task_cards: Sequence[Mapping[str, Any]],
    scenario_plan: Mapping[str, Any],
    framework_statuses: Mapping[str, Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    scenarios = [
        scenario
        for scenario in scenario_plan.get("scenarios", [])
        if isinstance(scenario, Mapping)
    ]
    requests: List[Dict[str, Any]] = []
    for task in task_cards:
        task_id = _string(task.get("task_id"))
        scenario_ids = sorted(
            _string(scenario.get("scenario_id"))
            for scenario in scenarios
            if _string(scenario.get("task_id")) == task_id
        )
        blockers = [] if scenario_ids else ["missing_scenario_card_for_task"]
        requests.append(
            {
                "task_id": task_id,
                "task_card_id": _string(task.get("task_card_id")),
                "task_statement": _string(task.get("task_statement")),
                "status": "request_manifest_ready" if scenario_ids else "blocked",
                "scenario_ids": scenario_ids,
                "simulator_frameworks": {
                    framework: {
                        "conversion_status": _string(payload.get("status")),
                        "blockers": _string_list(payload.get("blockers")),
                    }
                    for framework, payload in framework_statuses.items()
                },
                "execution_requested": False,
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
                "blockers": blockers,
                "claim_boundary": "task_simulation_request_is_manifest_only_not_execution",
            }
        )
    requests = sorted(requests, key=lambda item: item["task_id"])
    return {
        "schema_version": TASK_SIMULATION_REQUESTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "review_ready" if requests and all(not item["blockers"] for item in requests) else "blocked",
        "task_request_count": len(requests),
        "requests": requests,
        "execution_policy": {
            "simulator_execution_allowed_by_default": False,
            "requires_simulation_automation_gate": True,
            "remote_asset_downloads_allowed_by_default": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _scenario_simulator_matrix(
    *,
    context: Any,
    scenario_plan: Mapping[str, Any],
    framework_statuses: Mapping[str, Mapping[str, Any]],
    simulator_execution_manifest: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for scenario in scenario_plan.get("scenarios", []) or []:
        if not isinstance(scenario, Mapping):
            continue
        for framework in SIMULATOR_FRAMEWORKS:
            conversion = _mapping(framework_statuses.get(framework))
            execution = _simulator_execution_status(
                framework=framework,
                simulator_execution_manifest=simulator_execution_manifest,
            )
            conversion_status = _string(conversion.get("status")) or "missing"
            rows.append(
                {
                    "scenario_id": _string(scenario.get("scenario_id")),
                    "task_id": _string(scenario.get("task_id")),
                    "robot_profile_id": _string(scenario.get("robot_profile_id")),
                    "framework": framework,
                    "conversion_status": conversion_status,
                    "conversion_blockers": _string_list(conversion.get("blockers")),
                    "request_status": "blocked"
                    if conversion_status.startswith("blocked")
                    or conversion_status == "missing_simulation_automation_plan"
                    else "request_manifest_only",
                    "simulator_execution_status": execution["status"],
                    "source_simulator_execution_proven": bool(
                        execution.get("simulator_execution_proven")
                    ),
                    "site_eval_director_simulator_execution_proven": False,
                    "robot_readiness_proven": False,
                    "claim_boundary": "matrix_is_planning_only_not_simulator_execution",
                }
            )
    rows = sorted(rows, key=lambda item: (item["scenario_id"], item["framework"]))
    return {
        "schema_version": SCENARIO_SIMULATOR_MATRIX_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "review_ready" if rows else "blocked",
        "frameworks": list(SIMULATOR_FRAMEWORKS),
        "matrix_count": len(rows),
        "matrix": rows,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _agent_review_queue(
    *,
    context: Any,
    scenario_plan: Mapping[str, Any],
    eval_cards: Sequence[Mapping[str, Any]],
    proof_boundaries: Mapping[str, Any],
    agent_request_manifests: Mapping[str, str],
    generated_at: str,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []

    def add(item: Dict[str, Any]) -> None:
        review_id = _string(item.get("review_id"))
        if review_id and review_id not in {_string(existing.get("review_id")) for existing in items}:
            items.append(item)

    for scenario in scenario_plan.get("scenarios", []) or []:
        if not isinstance(scenario, Mapping):
            continue
        if bool(scenario.get("agent_inferred_components")):
            scenario_id = _string(scenario.get("scenario_id"))
            add(
                {
                    "review_id": f"{scenario_id}_agent_inferred_review",
                    "status": "review_required",
                    "card_family": "scenario_cards",
                    "scenario_id": scenario_id,
                    "task_id": _string(scenario.get("task_id")),
                    "reason": "agent_inferred_scenario_requires_operator_review",
                    "agent_inferred_components": _string_list(
                        scenario.get("agent_inferred_components")
                    ),
                    "claim_boundary": "agent_inferred_variations_are_review_inputs_only",
                }
            )
        for annotation in _string_list(scenario.get("required_missing_annotations")):
            scenario_id = _string(scenario.get("scenario_id"))
            add(
                {
                    "review_id": f"{scenario_id}_{annotation}",
                    "status": "review_required",
                    "card_family": "scenario_cards",
                    "scenario_id": scenario_id,
                    "task_id": _string(scenario.get("task_id")),
                    "reason": annotation,
                    "claim_boundary": "missing_annotation_blocks_stronger_eval_claims",
                }
            )
    for card in eval_cards:
        validation = _mapping(card.get("validation"))
        if _string(validation.get("actual_status")) == "needs_actual_outcome":
            add(
                {
                    "review_id": f"{_string(card.get('eval_card_id'))}_actual_outcome",
                    "status": "review_required",
                    "card_family": "eval_cards",
                    "scenario_id": _string(card.get("scenario_id")),
                    "task_id": _string(card.get("task_id")),
                    "reason": "missing_actual_outcome_blocks_validation",
                    "claim_boundary": "prediction_without_actual_outcome_is_advisory_only",
                }
            )
    if not bool(proof_boundaries.get("simulator_execution_proven")):
        add(
            {
                "review_id": "simulator_execution_proof_missing",
                "status": "review_required",
                "card_family": "proof_boundaries",
                "reason": "simulator_execution_proof_missing",
                "claim_boundary": "missing_simulator_trace_blocks_execution_claim",
            }
        )
    items = sorted(items, key=lambda item: item["review_id"])
    return {
        "schema_version": AGENT_REVIEW_QUEUE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "review_required" if items else "empty",
        "item_count": len(items),
        "items": items,
        "agent_request_manifests": dict(agent_request_manifests),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _proof_boundary(
    *,
    context: Any,
    robot_eval_proof_boundaries: Mapping[str, Any],
    simulator_execution_manifest: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SITE_EVAL_DIRECTOR_PROOF_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "review_ready",
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "robot_policy_execution_proven": False,
        "physics_contact_validated": False,
        "safety_validated": False,
        "training_completed": False,
        "public_claim_upgrade_allowed": False,
        "source_proof_summary": {
            "robot_eval_dataset_simulator_execution_proven": bool(
                robot_eval_proof_boundaries.get("simulator_execution_proven")
            ),
            "robot_eval_dataset_robot_policy_execution_proven": bool(
                robot_eval_proof_boundaries.get("robot_policy_execution_proven")
            ),
            "simulation_automation_simulator_execution_proven": bool(
                simulator_execution_manifest.get("simulator_execution_proven")
            ),
            "simulation_automation_robot_readiness_proven": bool(
                simulator_execution_manifest.get("robot_readiness_proven")
            ),
        },
        "remaining_required_evidence": list(CLAIM_BOUNDARY["proof_upgrade_requires"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _write_agent_request_manifests(
    *,
    automation_dir: Path,
    plan_context: Mapping[str, Any],
    agents_adapter: SiteEvalDirectorAdapter | None,
    codex_adapter: CodeMaintainerAdapter | None,
) -> Dict[str, str]:
    request_paths: Dict[str, str] = {}
    if agents_adapter is not None:
        payload = agents_adapter.build_request_manifest(plan_context=plan_context)
        path = automation_dir / "agents_sdk_site_eval_director_request.json"
        write_json(path, payload)
        request_paths["agents_sdk_site_eval_director"] = path.name
    if codex_adapter is not None:
        payload = codex_adapter.build_request_manifest(plan_context=plan_context)
        path = automation_dir / "codex_sdk_code_maintainer_request.json"
        write_json(path, payload)
        request_paths["codex_sdk_code_maintainer"] = path.name
    return request_paths


def build_site_eval_director(
    *,
    capture_root: str | Path,
    agents_adapter: SiteEvalDirectorAdapter | None = None,
    codex_adapter: CodeMaintainerAdapter | None = None,
    runner: SimulatorRunner | None = None,
    allow_simulator_execution: bool = False,
    allowed_simulators: Sequence[str] = (),
    allow_training: bool = False,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    automation_dir = pipeline_dir / "simulation_automation"
    ensure_dir(automation_dir)
    robot_eval_dir = pipeline_dir / "robot_eval_dataset"
    site_card = _read_optional_mapping(robot_eval_dir / "site_card.json")
    task_cards_payload = _read_optional_mapping(robot_eval_dir / "task_cards.json")
    scenario_cards_payload = _read_optional_mapping(robot_eval_dir / "scenario_cards.json")
    eval_cards_payload = _read_optional_mapping(robot_eval_dir / "eval_cards.json")
    robot_eval_proof_boundaries = _read_optional_mapping(robot_eval_dir / "proof_boundaries.json")
    worldlabs_world_manifest = _read_optional_mapping(pipeline_dir / "worldlabs_world_manifest.json")
    marble_bridge = _read_optional_mapping(
        pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json"
    )
    simready_scene = _read_optional_mapping(pipeline_dir / "simready" / "simready_scene_manifest.json")
    asset_conversion_plan = _read_optional_mapping(
        automation_dir / "asset_conversion_plan.json"
    )
    simulator_execution_manifest = _read_optional_mapping(
        automation_dir / "simulator_execution_manifest.json"
    )
    generated_at = _timestamp(
        site_card,
        task_cards_payload,
        scenario_cards_payload,
        eval_cards_payload,
        worldlabs_world_manifest,
        marble_bridge,
        simready_scene,
    )
    source_artifacts = _source_artifacts(automation_dir=automation_dir, pipeline_dir=pipeline_dir)
    missing_inputs = _missing_required_inputs(pipeline_dir)
    if missing_inputs:
        run_manifest = _write_blocked_outputs(
            context=context,
            automation_dir=automation_dir,
            source_artifacts=source_artifacts,
            missing_inputs=missing_inputs,
            generated_at=generated_at,
        )
        return {
            "schema_version": "site_eval_director_result.v1",
            "capture_root": str(context.capture_root),
            "automation_dir": str(automation_dir),
            "manifest_path": str(
                (automation_dir / "site_eval_director_run_manifest.json").resolve()
            ),
            "status": run_manifest["status"],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }

    task_cards = _cards(task_cards_payload)
    scenario_cards = _cards(scenario_cards_payload)
    eval_cards = _cards(eval_cards_payload)
    framework_statuses = _framework_statuses(asset_conversion_plan)
    real_engine_blockers = _real_engine_blockers(
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=allowed_simulators,
    )
    scenario_plan = _scenario_execution_plan(
        context=context,
        source_artifacts=source_artifacts,
        site_card=site_card,
        task_cards=task_cards,
        scenario_cards=scenario_cards,
        eval_cards=eval_cards,
        generated_at=generated_at,
    )
    task_requests = _task_simulation_requests(
        context=context,
        task_cards=task_cards,
        scenario_plan=scenario_plan,
        framework_statuses=framework_statuses,
        generated_at=generated_at,
    )
    matrix = _scenario_simulator_matrix(
        context=context,
        scenario_plan=scenario_plan,
        framework_statuses=framework_statuses,
        simulator_execution_manifest=simulator_execution_manifest,
        generated_at=generated_at,
    )
    proof_boundary = _proof_boundary(
        context=context,
        robot_eval_proof_boundaries=robot_eval_proof_boundaries,
        simulator_execution_manifest=simulator_execution_manifest,
        generated_at=generated_at,
    )
    plan_context = {
        "repo_root": str(Path(__file__).resolve().parents[2]),
        "capture_root": str(context.capture_root),
        "scenario_execution_plan": scenario_plan,
        "task_simulation_requests": task_requests,
        "scenario_simulator_matrix": matrix,
        "proof_boundary": proof_boundary,
    }
    agent_request_manifests = _write_agent_request_manifests(
        automation_dir=automation_dir,
        plan_context=plan_context,
        agents_adapter=agents_adapter,
        codex_adapter=codex_adapter,
    )
    review_queue = _agent_review_queue(
        context=context,
        scenario_plan=scenario_plan,
        eval_cards=eval_cards,
        proof_boundaries=robot_eval_proof_boundaries,
        agent_request_manifests=agent_request_manifests,
        generated_at=generated_at,
    )
    headless_loop_artifacts: Dict[str, str] = {}
    fixture_result = (runner or FixtureSimulatorRunner()).run(
        context=context,
        pipeline_dir=pipeline_dir,
        automation_dir=automation_dir,
        scenario_plan=scenario_plan,
        generated_at=generated_at,
    )
    normalized_trace = fixture_result.attempt
    if _rights_privacy_blocked(site_card):
        normalized_trace = {
            "schema_version": NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "status": "blocked",
            "runner": "fixture",
            "blockers": ["blocked_rights_privacy"],
            "attempt_count": 0,
            "attempts": [],
            "simulators_run": False,
            "fixture_runner_executed": False,
            "real_simulator_execution_proven": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        fixture_result = NormalizedAttempt(
            attempt=normalized_trace,
            blocked={
                "schema_version": SITE_EVAL_DIRECTOR_BLOCKED_SCHEMA_VERSION,
                "generated_at": generated_at,
                "scene_id": context.scene_id,
                "capture_id": context.capture_id,
                "status": "blocked",
                "blockers": ["blocked_rights_privacy"],
                "missing_inputs": ["rights_privacy_clearance"],
                "attempted_commands": ["build_site_eval_director"],
                "evidence": {
                    "rights_privacy": _mapping(
                        _mapping(site_card.get("provenance_rights_review_status")).get(
                            "rights_privacy"
                        )
                    )
                },
                "claim_boundary": dict(CLAIM_BOUNDARY),
            },
        )
    failure_labels = _failure_labels(
        context=context,
        normalized_trace=normalized_trace,
        generated_at=generated_at,
    )
    site_eval_ledger, calibration_report = _prediction_outcome_and_calibration(
        context=context,
        site_card=site_card,
        normalized_trace=normalized_trace,
        generated_at=generated_at,
    )
    breakage_library = _facility_breakage_library(
        context=context,
        site_card=site_card,
        normalized_trace=normalized_trace,
        generated_at=generated_at,
    )
    updated_eval_cards = _updated_eval_cards(
        context=context,
        eval_cards=eval_cards,
        normalized_trace=normalized_trace,
        generated_at=generated_at,
    )
    cosmos_exports = _cosmos_orchestration_exports(
        context=context,
        scenario_plan=scenario_plan,
        normalized_trace=normalized_trace,
        calibration_report=calibration_report,
        breakage_library=breakage_library,
        generated_at=generated_at,
        allow_training=allow_training,
    )
    real_evidence_blocked = _real_evidence_blocked_manifest(
        context=context,
        pipeline_dir=pipeline_dir,
        generated_at=generated_at,
    )
    headless_payloads = {
        "normalized_simulator_attempt_trace": normalized_trace,
        "failure_labels": failure_labels,
        "site_eval_prediction_outcome_ledger": site_eval_ledger,
        "site_eval_calibration_report": calibration_report,
        "learned_facility_breakage_library": breakage_library,
        "updated_eval_cards": updated_eval_cards,
        "cosmos_orchestration_exports": cosmos_exports,
    }
    for key, payload in headless_payloads.items():
        path = automation_dir / f"{key}.json"
        write_json(path, payload)
        headless_loop_artifacts[key] = path.name
    if fixture_result.blocked is not None:
        write_json(
            automation_dir / "site_eval_fixture_runner_blocked_manifest.json",
            fixture_result.blocked,
        )
        headless_loop_artifacts[
            "site_eval_fixture_runner_blocked_manifest"
        ] = "site_eval_fixture_runner_blocked_manifest.json"
    if real_evidence_blocked is not None:
        write_json(
            automation_dir / "site_eval_real_evidence_blocked_manifest.json",
            real_evidence_blocked,
        )
        headless_loop_artifacts[
            "site_eval_real_evidence_blocked_manifest"
        ] = "site_eval_real_evidence_blocked_manifest.json"
    status = "review_ready" if scenario_cards and task_cards else "blocked"
    if normalized_trace.get("status") == "completed":
        status = "fixture_loop_completed"
    run_manifest = {
        "schema_version": SITE_EVAL_DIRECTOR_RUN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": status,
        "scenario_execution_plan_path": "scenario_execution_plan.json",
        "task_simulation_requests_path": "task_simulation_requests.json",
        "scenario_simulator_matrix_path": "scenario_simulator_matrix.json",
        "agent_review_queue_path": "agent_review_queue.json",
        "proof_boundary_path": "site_eval_director_proof_boundary.json",
        "agent_request_manifests": dict(agent_request_manifests),
        "headless_loop_artifacts": dict(headless_loop_artifacts),
        "source_artifacts": dict(source_artifacts),
        "scenario_count": int(scenario_plan.get("scenario_count") or 0),
        "task_request_count": int(task_requests.get("task_request_count") or 0),
        "matrix_count": int(matrix.get("matrix_count") or 0),
        "review_queue_item_count": int(review_queue.get("item_count") or 0),
        "normalized_attempt_count": int(normalized_trace.get("attempt_count") or 0),
        "failure_label_count": int(failure_labels.get("label_count") or 0),
        "calibration_record_count": int(calibration_report.get("record_count") or 0),
        "breakage_record_count": int(breakage_library.get("record_count") or 0),
        "real_engine_execution_requests": real_engine_blockers,
        "live_provider_calls_performed": False,
        "remote_asset_downloads_performed": False,
        "simulators_run": False,
        "gpu_training_run": False,
        "messages_sent": False,
        "payments_touched": False,
        "deployments_performed": False,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "fixture_runner_executed": bool(normalized_trace.get("fixture_runner_executed")),
        "real_robot_or_owner_system_evidence_blocked": real_evidence_blocked is not None,
        "cosmos_training_completed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    run_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "source_artifacts": source_artifacts,
            "scenario_plan": scenario_plan.get("deterministic_fingerprint"),
            "task_request_count": run_manifest["task_request_count"],
            "matrix_count": run_manifest["matrix_count"],
            "normalized_attempt_count": run_manifest["normalized_attempt_count"],
            "failure_label_count": run_manifest["failure_label_count"],
            "calibration_record_count": run_manifest["calibration_record_count"],
            "breakage_record_count": run_manifest["breakage_record_count"],
            "agent_request_manifests": agent_request_manifests,
            "headless_loop_artifacts": headless_loop_artifacts,
        }
    )

    write_json(automation_dir / "scenario_execution_plan.json", scenario_plan)
    write_json(automation_dir / "task_simulation_requests.json", task_requests)
    write_json(automation_dir / "scenario_simulator_matrix.json", matrix)
    write_json(automation_dir / "agent_review_queue.json", review_queue)
    write_json(automation_dir / "site_eval_director_proof_boundary.json", proof_boundary)
    write_json(automation_dir / "site_eval_director_run_manifest.json", run_manifest)
    return {
        "schema_version": "site_eval_director_result.v1",
        "capture_root": str(context.capture_root),
        "automation_dir": str(automation_dir),
        "manifest_path": str((automation_dir / "site_eval_director_run_manifest.json").resolve()),
        "status": status,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic site-eval director manifests for a local capture package"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument(
        "--agents-sdk-site-eval",
        action="store_true",
        help="Write an optional OpenAI Agents SDK advisory request or blocked manifest",
    )
    parser.add_argument(
        "--codex-sdk-code-maintainer",
        action="store_true",
        help="Write an optional Codex SDK code-maintainer request or blocked manifest",
    )
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write"),
        default="read-only",
        help="Sandbox request for the optional Codex SDK code-maintainer manifest",
    )
    parser.add_argument("--codex-cli-path", default=None)
    parser.add_argument(
        "--allow-simulator-execution",
        action="store_true",
        help="Allow real simulator execution only when paired with env and simulator allow-list gates",
    )
    parser.add_argument(
        "--allow-simulator",
        choices=SIMULATOR_FRAMEWORKS,
        action="append",
        default=[],
        help="Allow one real simulator framework if env and command gates are also present",
    )
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Allow Cosmos training request readiness only when paired with env gates",
    )
    args = parser.parse_args(argv)
    try:
        result = build_site_eval_director(
            capture_root=args.capture_root,
            agents_adapter=AgentsSdkSiteEvalDirectorAdapter()
            if args.agents_sdk_site_eval
            else None,
            codex_adapter=CodexSdkCodeMaintainerAdapter(
                codex_cli_path=args.codex_cli_path,
                sandbox=args.codex_sandbox,
            )
            if args.codex_sdk_code_maintainer
            else None,
            allow_simulator_execution=args.allow_simulator_execution,
            allowed_simulators=args.allow_simulator,
            allow_training=args.allow_training,
        )
    except (OSError, ValueError) as exc:
        print(f"[site-eval-director] FAILED: {exc}")
        return 1
    print(f"[site-eval-director] manifest={result['manifest_path']}")
    print(f"[site-eval-director] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
