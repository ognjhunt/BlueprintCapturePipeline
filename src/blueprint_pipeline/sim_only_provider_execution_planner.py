"""Plan fail-closed sim-only provider execution for Unitree G1 MuJoCo beta jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .agent_operator_runtime import (
    LIVE_AGENTS_SDK_ENV,
    OperatorExecutor,
    OperatorRunConfig,
    blocked_operator_ledger,
    completed_operator_ledger,
    env_truthy,
    external_action_gates,
    module_available,
    proof_effect,
    run_agents_sdk_operator,
)
from .common import ensure_dir, optional_read_json, utc_now_iso, write_json
from .simulator_beta_readiness import build_simulator_beta_readiness


SIM_ONLY_PROVIDER_EXECUTION_PLAN_SCHEMA_VERSION = "sim_only_provider_execution_plan.v1"
SIM_ONLY_PROVIDER_PREFLIGHT_SCHEMA_VERSION = "sim_only_provider_preflight.v1"
SIM_ONLY_PROVIDER_RUNTIME_SCHEMA_VERSION = "sim_only_provider_runtime_manifest.v1"
SIM_ONLY_PROVIDER_COST_LEDGER_SCHEMA_VERSION = "sim_only_provider_cost_ledger.v1"
SIM_ONLY_PROVIDER_ARTIFACTS_SCHEMA_VERSION = "sim_only_provider_artifacts.v1"

LIVE_AGENT_PLANNER_ENV = "BLUEPRINT_ALLOW_SIM_ONLY_PROVIDER_AGENT_PLANNER"
DEFAULT_AGENT_MODEL = "gpt-5.5"

REMOTE_INPUT_SCHEMES = {"http", "https", "gs", "s3", "r2"}
WRITABLE_OUTPUT_SCHEMES = {"gs", "s3", "r2", "file", "local"}
SIGNED_URL_SECRET_QUERY_KEYS = {
    "x-goog-signature",
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "signature",
    "sig",
    "token",
}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [_string(item) for item in value if _string(item)]
    return []


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    number = _number(value)
    return int(number) if number is not None else default


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _uri_scheme(uri: str) -> str:
    parsed = urlparse(uri)
    return parsed.scheme or "local"


def _redact_uri(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.query:
        return value
    redacted_query = []
    for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in SIGNED_URL_SECRET_QUERY_KEYS:
            redacted_query.append((key, f"<redacted:{key}>"))
        else:
            redacted_query.append((key, query_value))
    return urlunparse(parsed._replace(query=urlencode(redacted_query)))


def _redact_runtime_value(value: Any, key_hint: str = "") -> Any:
    lowered_key = key_hint.lower()
    if isinstance(value, Mapping):
        return {
            str(key): _redact_runtime_value(item, str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_runtime_value(item, key_hint) for item in value]
    if isinstance(value, str):
        if any(token in lowered_key for token in ("secret", "token", "api_key", "apikey")):
            return f"<redacted:{key_hint or 'secret'}>" if value else value
        if "://" in value:
            return _redact_uri(value)
    return value


def _job_dir(capture_root: Path, job_id: str | None, job_dir: str | Path | None) -> Path:
    if job_dir:
        return Path(job_dir).expanduser().resolve()
    if not job_id:
        raise ValueError("job_id or job_dir is required")
    return (capture_root / "pipeline" / "robot_eval_jobs" / job_id).resolve()


def _read_job_artifacts(job_path: Path) -> dict[str, dict[str, Any]]:
    names = {
        "job_request": "job_request.json",
        "scheduler_decision": "scheduler_decision.json",
        "worker_launch_plan": "worker_launch_plan.json",
        "worker_manifest": "worker_manifest.json",
        "gpu_provider_launch_request": "gpu_provider_launch_request.json",
        "gpu_cost_control_ledger": "gpu_cost_control_ledger.json",
        "gpu_provisioning_result": "gpu_provisioning_result.json",
        "runpod_provider_adapter_result": "runpod_provider_adapter_result.json",
        "runpod_live_execution_proof": "runpod_live_execution_proof.json",
        "worker_runtime_manifest": "worker_runtime_manifest.json",
        "startup_architecture_audit": "startup_architecture_audit.json",
        "simulator_service_result": "simulator_service_result.json",
    }
    return {
        key: optional_read_json(job_path / filename) or {}
        for key, filename in names.items()
    }


def _cache_paths(simulator: str) -> dict[str, str]:
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


def _provider_gpu_priority(simulator: str, provider_shape: Mapping[str, Any]) -> list[str]:
    gpu = _mapping(provider_shape.get("gpu"))
    explicit = _string_list(
        gpu.get("provider_gpu_priority")
        or gpu.get("priority_fallback_list")
        or gpu.get("gpu_type_priority")
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
    if simulator in {"isaac_sim", "isaac_lab_arena"}:
        return [
            "NVIDIA RTX 4090",
            "NVIDIA RTX A6000",
            "NVIDIA RTX 6000 Ada Generation",
        ]
    return []


def _warm_pool_policy(
    *,
    request: Mapping[str, Any],
    worker_launch_plan: Mapping[str, Any],
    provider_shape: Mapping[str, Any],
) -> dict[str, Any]:
    execution_request = _mapping(request.get("execution_request"))
    gpu_allocation = _mapping(execution_request.get("gpu_allocation"))
    warm_config = _mapping(
        gpu_allocation.get("warm_pool_policy") or gpu_allocation.get("warm_pool")
    )
    launch_mode = _mapping(worker_launch_plan.get("launch_mode"))
    limits = _mapping(provider_shape.get("limits"))
    max_active_workers = _int(
        warm_config.get("max_active_workers")
        or limits.get("max_active_workers")
        or launch_mode.get("max_active_workers"),
        1,
    )
    latency_slo_seconds = _number(
        warm_config.get("latency_slo_seconds")
        or gpu_allocation.get("latency_slo_seconds")
    )
    max_idle_cost = _number(
        warm_config.get("max_idle_cost_usd_per_hour")
        or gpu_allocation.get("max_idle_cost_usd_per_hour"),
        0.0,
    )
    estimated_idle_cost = _number(warm_config.get("estimated_idle_cost_usd_per_hour"), 0.0)
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
    decision = "warm_active_worker" if warm_recommended else "scale_to_zero_on_demand"
    reasons = []
    if not warm_requested:
        reasons.append("warm_pool_not_requested")
    if warm_requested and not latency_justifies_idle:
        reasons.append("latency_policy_does_not_justify_idle_cost")
    if warm_requested and not idle_cost_allowed:
        reasons.append("warm_idle_cost_exceeds_policy")
    if warm_recommended:
        reasons.append("latency_policy_justifies_idle_cost")
    return {
        "decision": decision,
        "warm_worker_recommended": warm_recommended,
        "active_worker_target": 1 if warm_recommended else 0,
        "max_active_workers": max(1, max_active_workers),
        "scale_to_zero_default": not warm_recommended,
        "latency_slo_seconds": latency_slo_seconds,
        "estimated_idle_cost_usd_per_hour": estimated_idle_cost,
        "max_idle_cost_usd_per_hour": max_idle_cost,
        "decision_reasons": reasons,
    }


def _artifact_output_writable(uri: str) -> bool:
    if not uri:
        return False
    return _uri_scheme(uri) in WRITABLE_OUTPUT_SCHEMES


def _preflight_result(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    provider_gpu_priority: Sequence[str],
) -> dict[str, Any]:
    scheduler = _mapping(artifacts.get("scheduler_decision"))
    worker_plan = _mapping(artifacts.get("worker_launch_plan"))
    provider_request = _mapping(artifacts.get("gpu_provider_launch_request"))
    provider_shape = _mapping(provider_request.get("provider_request_shape"))
    image = _mapping(provider_shape.get("image"))
    inputs = _mapping(provider_shape.get("inputs"))
    limits = _mapping(provider_shape.get("limits"))
    blockers = _dedupe(
        [
            *_string_list(scheduler.get("blockers")),
            *_string_list(worker_plan.get("blockers")),
            *_string_list(provider_request.get("blockers")),
        ]
    )
    if provider_request.get("status") != "request_manifest_ready":
        blockers.append("provider_launch_request_not_ready")
    if not _string(image.get("configured_image_ref")):
        blockers.append("missing_prebuilt_worker_image_ref")
    if image.get("configured_image_ref_is_versioned") is not True:
        blockers.append("prebuilt_worker_image_ref_not_versioned")
    if image.get("configured_image_ref_fetchable_by_provider") is False:
        blockers.append("prebuilt_worker_image_ref_not_provider_fetchable")
    if not _string(inputs.get("manifest_uri")):
        blockers.append("missing_worker_manifest_uri")
    elif inputs.get("manifest_uri_fetchable_by_provider") is not True:
        blockers.append("worker_manifest_uri_not_fetchable_by_provider")
    if not _string(inputs.get("capture_root_bundle_uri")):
        blockers.append("missing_capture_root_bundle_uri")
    elif inputs.get("capture_root_bundle_uri_fetchable_by_provider") is not True:
        blockers.append("capture_root_bundle_uri_not_fetchable_by_provider")
    artifact_output_uri = _string(inputs.get("artifact_output_uri"))
    if not artifact_output_uri:
        blockers.append("missing_provider_artifact_output_uri")
    elif not _artifact_output_writable(artifact_output_uri):
        blockers.append("provider_artifact_output_uri_not_writable")
    if not _number(limits.get("hard_timeout_seconds")):
        blockers.append("missing_hard_timeout_seconds")
    if not _number(limits.get("external_watchdog_ttl_seconds")):
        blockers.append("missing_external_watchdog_ttl_seconds")
    if not provider_gpu_priority:
        blockers.append("missing_provider_gpu_priority_fallback_list")
    blockers = _dedupe(blockers)
    return {
        "schema_version": SIM_ONLY_PROVIDER_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked_before_spend",
        "spend_blocked": bool(blockers),
        "cpu_local_preflight_required_before_gpu_spend": True,
        "provider_inputs_present": "missing_worker_manifest_uri" not in blockers
        and "missing_capture_root_bundle_uri" not in blockers,
        "image_ref_provider_fetchable": "missing_prebuilt_worker_image_ref" not in blockers
        and "prebuilt_worker_image_ref_not_versioned" not in blockers
        and (
            "prebuilt_worker_image_ref_not_provider_fetchable" not in blockers
        ),
        "output_uri_writable_by_policy": bool(
            artifact_output_uri and _artifact_output_writable(artifact_output_uri)
        ),
        "blockers": blockers,
    }


def _agent_planner(
    *,
    plan_context: Mapping[str, Any],
    allow_live_agent_planner: bool,
    executor: OperatorExecutor | None,
    model: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    if executor is None and not module_available(("agents", "openai_agents")):
        blockers.append("missing_openai_agents_sdk")
    if not _string(os.getenv("OPENAI_API_KEY")) and executor is None:
        blockers.append("missing_openai_api_key")
    if not allow_live_agent_planner:
        blockers.append("missing_cli_allow_sim_only_provider_agent_planner")
    if not env_truthy(LIVE_AGENTS_SDK_ENV):
        blockers.append(f"missing_env_{LIVE_AGENTS_SDK_ENV}")
    if not env_truthy(LIVE_AGENT_PLANNER_ENV):
        blockers.append(f"missing_env_{LIVE_AGENT_PLANNER_ENV}")

    proof_artifacts_required = [
        "sim_only_provider_preflight.json",
        "gpu_provider_launch_request.json",
        "gpu_cost_control_ledger.json",
        "runpod_live_execution_proof.json",
        "worker_runtime_manifest.json",
        "simulator_beta_readiness_manifest.json",
    ]
    if blockers:
        return {
            "status": "blocked",
            "adapter": "openai_agents_sdk",
            "model": model,
            "agent_authority": "advisory_plan_only_when_gated",
            "proof_booleans_mutable_by_agent": False,
            "operator_ledger": blocked_operator_ledger(
                adapter="sim_only_provider_execution_planner",
                blockers=blockers,
                command_chosen="propose_sim_only_provider_execution_plan",
                proof_artifacts_required=proof_artifacts_required,
            ),
            "blockers": blockers,
            "proof_effect": proof_effect(
                deterministic_artifacts_required=proof_artifacts_required
            ),
        }

    prompt = (
        "Given this redacted Blueprint sim-only provider context, propose the safest "
        "cheapest RunPod execution plan. Return blockers instead of proposing spend "
        "when inputs, image refs, output URI, budget, timeout, or shutdown proof are missing. "
        "Do not set proof booleans.\n\n"
        "redacted_plan_context:\n"
        f"{json.dumps(plan_context, sort_keys=True, default=str)[:12000]}"
    )
    output = run_agents_sdk_operator(
        OperatorRunConfig(
            adapter="sim_only_provider_execution_planner",
            model=model,
            prompt=prompt,
            plan_context=plan_context,
            executor=executor,
        )
    )
    ledger = completed_operator_ledger(
        adapter="sim_only_provider_execution_planner",
        output=output,
        default_command="propose_sim_only_provider_execution_plan",
        proof_artifacts_required=proof_artifacts_required,
    )
    return {
        "status": "operator_completed",
        "adapter": "openai_agents_sdk",
        "model": model,
        "agent_authority": "advisory_plan_only",
        "proof_booleans_mutable_by_agent": False,
        "operator_ledger": ledger,
        "agent_output": output,
        "proof_effect": proof_effect(
            deterministic_artifacts_required=proof_artifacts_required
        ),
    }


def _runtime_manifest(
    *,
    job_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    runpod_proof: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    worker_runtime = _mapping(artifacts.get("worker_runtime_manifest"))
    adapter_result = _mapping(artifacts.get("runpod_provider_adapter_result"))
    status = "not_started"
    if worker_runtime:
        status = _string(worker_runtime.get("status")) or "observed"
    elif adapter_result.get("api_call_performed") is True:
        status = "provider_submitted_runtime_pending"
    if runpod_proof.get("production_runpod_worker_execution_proven") is True:
        status = "completed"
    elif runpod_proof.get("status") == "blocked":
        status = "blocked"
    return {
        "schema_version": SIM_ONLY_PROVIDER_RUNTIME_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": status,
        "worker_runtime_manifest_observed": bool(worker_runtime),
        "worker_runtime_manifest_status": worker_runtime.get("status"),
        "runpod_live_execution_proof_status": runpod_proof.get("status"),
        "production_runpod_worker_execution_proven": bool(
            runpod_proof.get("production_runpod_worker_execution_proven") is True
        ),
        "simulator_execution_proven": bool(runpod_proof.get("simulator_execution_proven") is True),
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "shutdown_or_termination_proof": bool(
            runpod_proof.get("shutdown_or_termination_proof") is True
        ),
        "active_pod_count_before": runpod_proof.get("active_pod_count_before"),
        "active_pod_count_after": runpod_proof.get("active_pod_count_after"),
        "blockers": _string_list(worker_runtime.get("blockers"))
        or _string_list(runpod_proof.get("blockers")),
    }


def _cost_ledger(
    *,
    job_id: str,
    provider: str,
    preflight: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    runtime_manifest: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    provider_request = _mapping(artifacts.get("gpu_provider_launch_request"))
    provider_shape = _mapping(provider_request.get("provider_request_shape"))
    limits = _mapping(provider_shape.get("limits"))
    adapter_result = _mapping(artifacts.get("runpod_provider_adapter_result"))
    runpod_proof = _mapping(artifacts.get("runpod_live_execution_proof"))
    status = "blocked-before-allocation" if preflight.get("spend_blocked") else "planned"
    if adapter_result.get("api_call_performed") is True and runtime_manifest.get("status") not in {
        "completed",
        "failed",
        "blocked",
    }:
        status = "running"
    if runtime_manifest.get("status") == "completed":
        status = "completed"
    if runtime_manifest.get("status") == "failed":
        status = "failed"
    if runpod_proof.get("shutdown_or_termination_proof") is True:
        status = "stopped" if status in {"completed", "failed", "running"} else status
    return {
        "schema_version": SIM_ONLY_PROVIDER_COST_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "provider": provider,
        "status": status,
        "supported_states": [
            "blocked-before-allocation",
            "running",
            "completed",
            "failed",
            "stopped",
        ],
        "max_budget_per_job_usd": _number(limits.get("requested_budget_usd")),
        "hard_timeout_seconds": _int(limits.get("hard_timeout_seconds"), 0),
        "watchdog_timeout_seconds": _int(limits.get("external_watchdog_ttl_seconds"), 0),
        "idle_timeout_seconds": _int(limits.get("idle_timeout_seconds"), 0),
        "max_active_workers": _int(limits.get("max_active_workers"), 1),
        "estimated_billable_gpu_seconds": 0
        if preflight.get("spend_blocked")
        else _int(limits.get("hard_timeout_seconds"), 0),
        "actual_gpu_seconds": _mapping(artifacts.get("gpu_cost_control_ledger"))
        .get("gpu_time", {})
        .get("actual_gpu_seconds"),
        "blockers": _string_list(preflight.get("blockers")),
    }


def build_sim_only_provider_execution_layer(
    *,
    capture_root: str | Path,
    job_id: str | None = None,
    job_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    allow_live_agent_planner: bool = False,
    agent_executor: OperatorExecutor | None = None,
    agent_model: str | None = None,
    update_simulator_beta_readiness: bool = False,
) -> dict[str, Any]:
    root = Path(capture_root).expanduser().resolve()
    path = _job_dir(root, job_id, job_dir)
    ensure_dir(path)
    resolved_job_id = job_id or path.name
    generated_at = utc_now_iso()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else path
    ensure_dir(out_dir)
    artifacts = _read_job_artifacts(path)
    request = _mapping(artifacts.get("job_request"))
    worker_plan = _mapping(artifacts.get("worker_launch_plan"))
    provider_request = _mapping(artifacts.get("gpu_provider_launch_request"))
    provider_shape = _mapping(provider_request.get("provider_request_shape"))
    scheduler = _mapping(artifacts.get("scheduler_decision"))
    simulator = _string(worker_plan.get("simulator")) or _string(scheduler.get("simulator"))
    if not simulator:
        simulator = _string(provider_shape.get("runtime_preflight", {}).get("simulator")) or "mujoco"
    provider = _string(provider_request.get("provider")) or _string(worker_plan.get("provider")) or "runpod"
    provider_gpu_priority = _provider_gpu_priority(simulator, provider_shape)
    warm_policy = _warm_pool_policy(
        request=request,
        worker_launch_plan=worker_plan,
        provider_shape=provider_shape,
    )
    preflight = _preflight_result(
        artifacts=artifacts,
        provider_gpu_priority=provider_gpu_priority,
    )
    runpod_proof = _mapping(artifacts.get("runpod_live_execution_proof"))
    runtime_manifest = _runtime_manifest(
        job_id=resolved_job_id,
        artifacts=artifacts,
        runpod_proof=runpod_proof,
        generated_at=generated_at,
    )
    cost_ledger = _cost_ledger(
        job_id=resolved_job_id,
        provider=provider,
        preflight=preflight,
        artifacts=artifacts,
        runtime_manifest=runtime_manifest,
        generated_at=generated_at,
    )
    plan_context = _redact_runtime_value(
        {
            "job_id": resolved_job_id,
            "capture_root": str(root),
            "provider": provider,
            "simulator": simulator,
            "job_request": request,
            "scheduler_decision": scheduler,
            "worker_launch_plan": worker_plan,
            "gpu_provider_launch_request": provider_request,
            "preflight": preflight,
            "warm_pool_policy": warm_policy,
            "cost_ledger": cost_ledger,
        }
    )
    agent_planner = _agent_planner(
        plan_context=plan_context,
        allow_live_agent_planner=allow_live_agent_planner,
        executor=agent_executor,
        model=agent_model or os.getenv("BLUEPRINT_SIM_ONLY_PROVIDER_AGENT_MODEL") or DEFAULT_AGENT_MODEL,
    )
    readiness: dict[str, Any] = {
        "status": "not_refreshed",
        "update_requested": update_simulator_beta_readiness,
    }
    if update_simulator_beta_readiness:
        readiness_manifest = build_simulator_beta_readiness(capture_root=root)
        readiness = {
            "status": readiness_manifest.get("status"),
            "ready_for_simulator_beta": readiness_manifest.get("ready_for_simulator_beta"),
            "manifest_path": readiness_manifest.get("artifacts", {}).get("manifest"),
            "blocking_gate_ids": readiness_manifest.get("blocking_gate_ids"),
        }
    plan = {
        "schema_version": SIM_ONLY_PROVIDER_EXECUTION_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": resolved_job_id,
        "status": "ready_for_provider_launch" if preflight["status"] == "passed" else "blocked_before_spend",
        "provider": provider,
        "provider_priority": ["runpod", "vast", "gcp"],
        "selected_provider": provider,
        "simulator_scope": {
            "scope": "simulator_only_beta",
            "robot": "unitree_g1",
            "simulator_backend": simulator,
            "task": "walk_to_target",
            "physical_robot_readiness": "out_of_scope_for_simulator_beta",
            "physical_safety_validation": "out_of_scope_for_simulator_beta",
            "physical_policy_acceptance": "out_of_scope_for_simulator_beta",
        },
        "cheapest_sufficient_path": {
            "cpu_local_preflight_before_gpu_spend": True,
            "mujoco_non_render_or_light_render_avoids_isaac_class_gpus": simulator == "mujoco",
            "prebuilt_mujoco_worker_image_preferred": simulator == "mujoco",
            "scale_to_zero_default": bool(warm_policy.get("scale_to_zero_default")),
            "warm_gpu_only_when_latency_policy_justifies_idle_cost": True,
        },
        "provider_gpu_priority_fallback_list": provider_gpu_priority,
        "avoid_gpu_classes": [
            "A100",
            "H100",
            "RTX A6000 unless render_or_latency_policy_requires_it",
            "RTX 6000 Ada unless render_or_latency_policy_requires_it",
        ]
        if simulator == "mujoco"
        else [],
        "warm_pool_policy": warm_policy,
        "persistent_cache_paths": _cache_paths(simulator),
        "preflight_path": "sim_only_provider_preflight.json",
        "runtime_manifest_path": "sim_only_provider_runtime_manifest.json",
        "cost_ledger_path": "sim_only_provider_cost_ledger.json",
        "artifacts_manifest_path": "sim_only_provider_artifacts_manifest.json",
        "agent_planner": agent_planner,
        "preflight": preflight,
        "runtime_manifest": runtime_manifest,
        "cost_ledger": cost_ledger,
        "simulator_beta_readiness_update": readiness,
        "external_action_gates": external_action_gates(),
        "live_provider_calls_performed": False,
        "proof_booleans_mutable_by_agent": False,
        "secret_values_in_artifact": False,
        "claim_boundary": {
            "simulator_beta_success_evaluated_by_sim_only_gate": True,
            "physical_robot_readiness_claimed": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
        "blockers": _string_list(preflight.get("blockers")),
    }
    artifacts_manifest = {
        "schema_version": SIM_ONLY_PROVIDER_ARTIFACTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": resolved_job_id,
        "artifacts": {
            "sim_only_provider_execution_plan": "sim_only_provider_execution_plan.json",
            "sim_only_provider_preflight": "sim_only_provider_preflight.json",
            "sim_only_provider_runtime_manifest": "sim_only_provider_runtime_manifest.json",
            "sim_only_provider_cost_ledger": "sim_only_provider_cost_ledger.json",
            "gpu_provider_launch_request": "gpu_provider_launch_request.json",
            "gpu_cost_control_ledger": "gpu_cost_control_ledger.json",
            "worker_manifest": "worker_manifest.json",
            "runpod_provider_adapter_result": "runpod_provider_adapter_result.json",
            "runpod_live_execution_proof": "runpod_live_execution_proof.json",
        },
        "artifact_presence": {
            key: bool(payload)
            for key, payload in artifacts.items()
        },
        "secret_values_in_artifact": False,
    }
    write_json(out_dir / "sim_only_provider_preflight.json", preflight)
    write_json(out_dir / "sim_only_provider_runtime_manifest.json", runtime_manifest)
    write_json(out_dir / "sim_only_provider_cost_ledger.json", cost_ledger)
    write_json(out_dir / "sim_only_provider_artifacts_manifest.json", artifacts_manifest)
    write_json(out_dir / "sim_only_provider_execution_plan.json", plan)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan fail-closed sim-only provider execution for a robot-eval job."
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--job-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-live-agent-planner", action="store_true")
    parser.add_argument("--agent-model")
    parser.add_argument("--update-simulator-beta-readiness", action="store_true")
    args = parser.parse_args(argv)
    result = build_sim_only_provider_execution_layer(
        capture_root=args.capture_root,
        job_id=args.job_id,
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        allow_live_agent_planner=args.allow_live_agent_planner,
        agent_model=args.agent_model,
        update_simulator_beta_readiness=args.update_simulator_beta_readiness,
    )
    print(result["status"])
    print(str(Path(args.output_dir or args.job_dir or "").expanduser()) if args.output_dir else result["artifacts_manifest_path"])
    return 0 if result["status"] == "ready_for_provider_launch" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
