"""Managed GPU startup policy for website-origin robot-eval jobs.

The plan produced here is deterministic policy plumbing. It does not launch a
provider, allocate a GPU, or upgrade simulator artifacts into generated-world rank fidelity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .provider_worker_contract import (
    build_provider_worker_contract,
    classify_policy_worker_command,
)


GPU_STARTUP_PIPELINE_PLAN_SCHEMA_VERSION = "robot_eval_gpu_startup_pipeline_plan.v1"

LOCAL_PROVISIONERS = {"fixture_local", "local_process", "docker_local"}
LIVE_GPU_PROVISIONERS = {"runpod", "lambda_cloud", "vast", "gcp"}
PROVIDER_PRIORITY = ["runpod", "lambda_cloud", "gcp", "vast"]
MANAGED_PROVIDER_PRIORITY = [
    "runpod_secure_cloud",
    "lambda_cloud",
    "aws_g6",
    "coreweave",
]
LARGE_IMAGE_TOTAL_WARN_BYTES = 12_000_000_000
LARGE_IMAGE_LAYER_WARN_BYTES = 8_000_000_000
RUNPOD_IMAGE_STARTUP_CANARY_ARTIFACT = "runpod_image_startup_canary_output.zip"
LARGE_WORKER_IMAGE_COLD_START_BLOCKER = (
    "large_worker_image_requires_canary_or_warm_provider"
)
MARKETPLACE_PROVIDER_TIERS = {
    "marketplace",
    "marketplace_quarantined",
    "community",
    "community_cloud",
    "runpod_community",
    "vast_marketplace",
}
CLAIM_BOUNDARY = {
    "simulator_proof_only": True,
    "generated_world_rank_fidelity_result_proven": False,
    "generated_world_policy_evaluation_scope_proven": False,
    "non_ranking_operational_claim_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [_string(item) for item in value if _string(item)]
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _bytes_value(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0:
        return None
    return int(number)


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _bool_from_policy(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "allow", "allowed"}
    return default


def _provider_tier(provisioner: str, startup_policy: Mapping[str, Any]) -> str:
    explicit = _string(
        startup_policy.get("provider_tier")
        or startup_policy.get("selected_provider_tier")
        or startup_policy.get(f"{provisioner}_provider_tier")
    )
    if explicit:
        return explicit
    if provisioner == "runpod":
        return "managed_secure_cloud_preferred"
    if provisioner == "lambda_cloud":
        return "managed_lambda_cloud"
    if provisioner == "gcp":
        return "hyperscaler_managed"
    if provisioner == "vast":
        return "marketplace_quarantined"
    if provisioner in LOCAL_PROVISIONERS:
        return "local_development"
    return "unknown"


def _selected_provider_is_marketplace(provisioner: str, provider_tier: str) -> bool:
    tier = provider_tier.strip().lower()
    return provisioner == "vast" or tier in MARKETPLACE_PROVIDER_TIERS


def _startup_policy_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    execution_request = _mapping(request.get("execution_request"))
    gpu_allocation = _mapping(execution_request.get("gpu_allocation"))
    return _mapping(
        gpu_allocation.get("startup_pipeline")
        or gpu_allocation.get("provider_startup_pipeline")
        or execution_request.get("gpu_startup_pipeline")
        or request.get("gpu_startup_pipeline")
    )


def _image_size_metadata(image: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "image_size_diagnostic",
        "worker_image_manifest_diagnostic",
        "image_size_metadata",
        "image_manifest",
        "registry_manifest",
        "manifest_inspection",
    ):
        metadata = _mapping(image.get(key))
        if metadata:
            return metadata
    return {}


def _image_layer_sizes_bytes(image: Mapping[str, Any]) -> list[int]:
    metadata = _image_size_metadata(image)
    sizes: list[int] = []
    for source in (image, metadata):
        for key in (
            "compressed_layer_sizes_bytes",
            "layer_sizes_bytes",
            "layers_size_bytes",
            "layer_size_bytes",
        ):
            value = source.get(key)
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for item in value:
                    size = _bytes_value(item)
                    if size is not None:
                        sizes.append(size)
            else:
                size = _bytes_value(value)
                if size is not None:
                    sizes.append(size)
        layers = source.get("layers")
        if isinstance(layers, Sequence) and not isinstance(
            layers, (str, bytes, bytearray)
        ):
            for layer in layers:
                layer_mapping = _mapping(layer)
                for key in ("compressed_size_bytes", "size_bytes", "size"):
                    size = _bytes_value(layer_mapping.get(key))
                    if size is not None:
                        sizes.append(size)
                        break
    return sizes


def _worker_image_size_diagnostic(
    worker_launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    image = _mapping(worker_launch_plan.get("worker_image"))
    metadata = _image_size_metadata(image)
    layer_sizes = _image_layer_sizes_bytes(image)
    explicit_largest = next(
        (
            _bytes_value(source.get(key))
            for source in (image, metadata)
            for key in (
                "largest_compressed_layer_size_bytes",
                "largest_layer_size_bytes",
                "max_layer_size_bytes",
            )
            if _bytes_value(source.get(key)) is not None
        ),
        None,
    )
    largest_layer_size = (
        explicit_largest if explicit_largest is not None else max(layer_sizes, default=None)
    )
    explicit_total = next(
        (
            _bytes_value(source.get(key))
            for source in (image, metadata)
            for key in (
                "total_compressed_size_bytes",
                "compressed_size_bytes",
                "manifest_total_size_bytes",
                "total_layer_size_bytes",
            )
            if _bytes_value(source.get(key)) is not None
        ),
        None,
    )
    total_size = explicit_total if explicit_total is not None else (
        sum(layer_sizes) if layer_sizes else None
    )
    explicit_large = next(
        (
            source.get("large_image_pull_risk")
            for source in (image, metadata)
            if isinstance(source.get("large_image_pull_risk"), bool)
        ),
        None,
    )
    large_total = (
        total_size is not None and total_size >= LARGE_IMAGE_TOTAL_WARN_BYTES
    )
    large_layer = (
        largest_layer_size is not None
        and largest_layer_size >= LARGE_IMAGE_LAYER_WARN_BYTES
    )
    warnings: list[str] = []
    if large_total:
        warnings.append("large_worker_image_total_size_may_exceed_startup_watchdog")
    if large_layer:
        warnings.append("large_worker_image_layer_may_exceed_startup_watchdog")
    return {
        "metadata_present": bool(
            metadata
            or layer_sizes
            or total_size is not None
            or largest_layer_size is not None
            or explicit_large is not None
        ),
        "total_compressed_size_bytes": total_size,
        "largest_layer_size_bytes": largest_layer_size,
        "large_image_pull_risk": bool(
            explicit_large is True or large_total or large_layer
        ),
        "warnings": warnings,
    }


def _worker_image_policy(worker_launch_plan: Mapping[str, Any]) -> dict[str, Any]:
    image = _mapping(worker_launch_plan.get("worker_image"))
    image_size_diagnostic = _worker_image_size_diagnostic(worker_launch_plan)
    return {
        "configured_image_ref": image.get("configured_image_ref"),
        "image_ref_env_var": image.get("image_ref_env_var"),
        "image_ref_present": bool(image.get("configured_image_ref_present")),
        "version_pin_required": bool(image.get("version_pin_required") is not False),
        "configured_image_ref_is_versioned": bool(
            image.get("configured_image_ref_is_versioned")
        ),
        "configured_image_ref_fetchable_by_provider": bool(
            image.get("configured_image_ref_fetchable_by_provider")
        ),
        "runtime_dependency_install_disallowed": bool(
            image.get("runtime_dependency_install_disallowed")
        ),
        "image_size_metadata_available": bool(
            image_size_diagnostic.get("metadata_present")
        ),
        "total_compressed_size_bytes": image_size_diagnostic.get(
            "total_compressed_size_bytes"
        ),
        "largest_layer_size_bytes": image_size_diagnostic.get(
            "largest_layer_size_bytes"
        ),
        "large_image_pull_risk": bool(
            image_size_diagnostic.get("large_image_pull_risk")
        ),
        "image_size_warnings": _string_list(image_size_diagnostic.get("warnings")),
    }


def _active_worker_target(value: Any) -> int:
    number = _number(value)
    if number is None:
        return 0
    return max(0, int(number))


def _startup_policy_bool(
    startup_policy: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: bool = False,
) -> bool:
    for key in keys:
        if key in startup_policy:
            return _bool_from_policy(startup_policy.get(key), default=default)
    return default


def _same_image_canary_status(startup_policy: Mapping[str, Any]) -> str:
    return _string(
        startup_policy.get("same_image_startup_canary_status")
        or startup_policy.get("image_startup_canary_status")
        or startup_policy.get("runpod_image_startup_canary_status")
    ).lower()


def _same_image_canary_completed(startup_policy: Mapping[str, Any]) -> bool:
    if _startup_policy_bool(
        startup_policy,
        (
            "same_image_startup_canary_completed",
            "image_startup_canary_completed",
            "runpod_image_startup_canary_completed",
        ),
        default=False,
    ):
        return True
    return _same_image_canary_status(startup_policy) in {
        "passed",
        "success",
        "succeeded",
        "complete",
        "completed",
        "ready",
    }


def _image_startup_canary_launch(startup_policy: Mapping[str, Any]) -> bool:
    if _startup_policy_bool(
        startup_policy,
        (
            "image_startup_canary_only",
            "same_image_startup_canary_only",
            "runpod_image_startup_canary_only",
        ),
        default=False,
    ):
        return True
    mode = _string(
        startup_policy.get("mode")
        or startup_policy.get("launch_mode")
        or startup_policy.get("provider_mode")
    ).lower()
    return mode in {
        "image-startup-canary-pod",
        "image_startup_canary",
        "same_image_startup_canary",
        "runpod_image_startup_canary",
        "canary_only",
    }


def _simulator_is_isaac(simulator: str) -> bool:
    return simulator.strip().lower() in {"isaac", "isaac_sim", "isaac-sim"}


def _large_image_cold_start_policy(
    *,
    provisioner: str,
    simulator: str,
    live_provider_job: bool,
    startup_policy: Mapping[str, Any],
    warm_pool_policy: Mapping[str, Any],
    worker_image_policy: Mapping[str, Any],
) -> dict[str, Any]:
    active_worker_target = _active_worker_target(
        warm_pool_policy.get("active_worker_target")
    )
    warm_worker_available_or_requested = bool(
        warm_pool_policy.get("warm_worker_recommended")
    ) or active_worker_target > 0
    warm_worker_available_or_requested = (
        warm_worker_available_or_requested
        or _startup_policy_bool(
            startup_policy,
            (
                "warm_worker_available",
                "warm_worker_requested",
                "prewarmed_worker_available",
                "prewarmed_worker_requested",
            ),
            default=False,
        )
    )
    warm_decision = _string(warm_pool_policy.get("decision")).lower()
    if warm_decision in {
        "warm_pool",
        "existing_warm_worker",
        "existing_pod_start",
        "reuse_warm_provider",
    }:
        warm_worker_available_or_requested = True
    cold_scale_to_zero_start = not warm_worker_available_or_requested and (
        not warm_decision
        or warm_decision
        in {
            "scale_to_zero_on_demand",
            "scale_to_zero",
            "on_demand",
            "cold_on_demand",
        }
        or warm_pool_policy.get("scale_to_zero_default") is not False
    )
    same_image_canary_completed = _same_image_canary_completed(startup_policy)
    image_startup_canary_launch = _image_startup_canary_launch(startup_policy)
    explicit_debug_override = _startup_policy_bool(
        startup_policy,
        (
            "allow_large_runpod_image_fresh_start",
            "large_runpod_image_fresh_start_override",
        ),
        default=False,
    )
    large_runpod_isaac_image = bool(
        live_provider_job
        and provisioner == "runpod"
        and _simulator_is_isaac(simulator)
        and worker_image_policy.get("large_image_pull_risk") is True
    )
    same_image_canary_required = bool(
        large_runpod_isaac_image
        and cold_scale_to_zero_start
        and not same_image_canary_completed
        and not warm_worker_available_or_requested
    )
    blockers: list[str] = []
    if (
        same_image_canary_required
        and not image_startup_canary_launch
        and not explicit_debug_override
    ):
        blockers.append(LARGE_WORKER_IMAGE_COLD_START_BLOCKER)
    customer_eval_launch_allowed = bool(
        not same_image_canary_required or explicit_debug_override
    )
    return {
        "selected_provider": provisioner,
        "simulator": simulator,
        "large_runpod_isaac_image": large_runpod_isaac_image,
        "large_image_pull_risk": bool(worker_image_policy.get("large_image_pull_risk")),
        "image_size_metadata_available": bool(
            worker_image_policy.get("image_size_metadata_available")
        ),
        "total_compressed_size_bytes": worker_image_policy.get(
            "total_compressed_size_bytes"
        ),
        "largest_layer_size_bytes": worker_image_policy.get(
            "largest_layer_size_bytes"
        ),
        "cold_scale_to_zero_start": cold_scale_to_zero_start,
        "warm_worker_available_or_requested": warm_worker_available_or_requested,
        "same_image_startup_canary_status": _same_image_canary_status(startup_policy)
        or None,
        "same_image_startup_canary_completed": same_image_canary_completed,
        "image_startup_canary_launch": image_startup_canary_launch,
        "same_image_startup_canary_required_before_customer_eval": bool(
            large_runpod_isaac_image and not same_image_canary_completed
        ),
        "fresh_start_debug_override": explicit_debug_override,
        "customer_eval_launch_allowed": customer_eval_launch_allowed,
        "canary_launch_allowed": not blockers,
        "required_artifact": (
            RUNPOD_IMAGE_STARTUP_CANARY_ARTIFACT
            if large_runpod_isaac_image
            else None
        ),
        "blockers": blockers,
        "claim_boundary": {
            "canary_proves_container_user_command_and_artifact_upload_only": True,
            "canary_proves_isaac_scene_or_wam_quality": False,
            "canary_proves_robot_policy_readiness": False,
        },
    }


def _cache_policy(worker_launch_plan: Mapping[str, Any]) -> dict[str, Any]:
    cache = _mapping(worker_launch_plan.get("cache_plan"))
    return {
        "persistent_cache_recommended": bool(cache.get("persistent_cache_recommended")),
        "cache_targets": _string_list(cache.get("targets")),
        "cache_paths": _mapping(cache.get("paths")),
        "install_simulator_during_customer_job": bool(
            cache.get("install_simulator_during_customer_job")
        ),
        "install_python_dependencies_during_customer_job": bool(
            cache.get("install_python_dependencies_during_customer_job")
        ),
    }


def _preflight_checks(
    *,
    runtime_preflight_contract: Mapping[str, Any],
    live_provider_job: bool,
) -> list[str]:
    checks = [
        "worker_runtime_preflight",
        "container_image_digest_or_version_pin",
        "artifact_output_write_smoke",
    ]
    if live_provider_job:
        checks.append("nvidia_smi_gpu_inventory")
    checks.extend(_string_list(runtime_preflight_contract.get("required_checks")))
    checks.extend(
        [
            "scene_asset_load_smoke",
            "headless_render_smoke",
            "short_policy_wam_loop_canary",
        ]
    )
    return _dedupe(checks)


def _policy_worker_command_from_launch_plan(worker_launch_plan: Mapping[str, Any]) -> str:
    command = _string(
        worker_launch_plan.get("policy_command")
        or worker_launch_plan.get("policy_worker_command")
        or worker_launch_plan.get("worker_command")
        or worker_launch_plan.get("entrypoint_command")
    )
    if command:
        return command
    worker = _mapping(worker_launch_plan.get("worker"))
    command = _string(
        worker.get("policy_command")
        or worker.get("policy_worker_command")
        or worker.get("entrypoint_command")
    )
    if command:
        return command
    runtime_contract = _mapping(worker_launch_plan.get("runtime_contract"))
    return _string(
        runtime_contract.get("policy_command")
        or runtime_contract.get("policy_worker_command")
    )


def _provider_worker_session_policy(
    *,
    provisioner: str,
    worker_launch_plan: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    worker_role = _string(worker_launch_plan.get("worker_role")) or "robot_eval_worker"
    policy_command = _policy_worker_command_from_launch_plan(worker_launch_plan)
    command_classification = classify_policy_worker_command(policy_command)
    contract = build_provider_worker_contract(
        generated_at=generated_at,
        provider=provisioner or "provider_neutral",
        worker_role=worker_role,
        policy_command=policy_command,
    )
    blockers: list[str] = []
    if policy_command and not command_classification.get("repeated_policy_loop_allowed"):
        blockers.extend(_string_list(command_classification.get("blockers")))
    return {
        "schema_version": "provider_worker_session_policy.v1",
        "generated_at": generated_at,
        "status": "blocked" if blockers else "ready_for_session_orchestration",
        "provider": provisioner or "provider_neutral",
        "worker_role": worker_role,
        "session_scope": "one_ready_worker_per_evaluation_job_or_worker_role",
        "allocation_lifecycle": {
            "allocate_provider_worker_once_per_eval_job": True,
            "load_models_once_before_first_infer": True,
            "reuse_ready_worker_for_all_policy_steps": True,
            "provider_allocation_per_inference_allowed": False,
            "shutdown_after_eval_job_final_artifacts": True,
        },
        "http_contract": contract["http_contract"],
        "readiness_gate": {
            "readyz_required_before_first_infer": True,
            "healthz_is_not_model_ready": True,
            "infer_requires_readyz_success": True,
            "shutdown_response_requires_provider_teardown_artifact_for_cost_proof": True,
        },
        "provider_adapter_responsibilities": contract["worker_lifecycle"][
            "provider_adapter_responsibilities"
        ],
        "policy_loop_responsibilities": contract["worker_lifecycle"][
            "policy_loop_responsibilities"
        ],
        "policy_command_configured": bool(policy_command),
        "policy_command_classification": command_classification,
        "blockers": _dedupe(blockers),
        "claim_boundary": {
            "session_policy_is_not_provider_execution_proof": True,
            "worker_readyz_artifact_required_before_customer_eval": True,
            "remote_provider_shutdown_not_proven_by_plan": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "raw_secret_values_recorded": False,
        },
    }


def build_gpu_startup_pipeline_plan(
    *,
    request: Mapping[str, Any],
    job_id: str,
    provisioner: str,
    simulator: str,
    scheduler_decision: Mapping[str, Any],
    worker_launch_plan: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the fail-closed startup policy artifact for a robot-eval job."""

    generated_at = generated_at or utc_now_iso()
    execution_request = _mapping(request.get("execution_request"))
    gpu_allocation = _mapping(execution_request.get("gpu_allocation"))
    startup_policy = _startup_policy_from_request(request)
    live_provider_job = provisioner in LIVE_GPU_PROVISIONERS and simulator != "fixture"
    external_execution = provisioner not in LOCAL_PROVISIONERS or simulator != "fixture"
    website_queue_request = bool(execution_request) or bool(
        request.get("webapp_request_id") or request.get("webappRequestId")
    )
    customer_gpu_job = bool(website_queue_request and external_execution)
    provider_tier = _provider_tier(provisioner, startup_policy)
    provider_is_marketplace = _selected_provider_is_marketplace(provisioner, provider_tier)
    provider_priority = _string_list(
        startup_policy.get("provider_priority")
        or gpu_allocation.get("provider_priority")
        or gpu_allocation.get("provider_api_priority")
    ) or list(PROVIDER_PRIORITY)
    managed_provider_priority = _string_list(
        startup_policy.get("managed_provider_priority")
        or gpu_allocation.get("managed_provider_priority")
    ) or list(MANAGED_PROVIDER_PRIORITY)
    marketplace_allowed = _bool_from_policy(
        startup_policy.get("marketplace_allowed_for_customer_job"),
        default=False,
    )
    strict_canary_required = _bool_from_policy(
        startup_policy.get("strict_preflight_canary_required"),
        default=bool(external_execution),
    )
    runtime_preflight_contract = _mapping(
        worker_launch_plan.get("runtime_preflight_contract")
    )
    warm_pool_policy = _mapping(_mapping(worker_launch_plan.get("launch_mode")).get("warm_pool_policy"))
    worker_image_policy = _worker_image_policy(worker_launch_plan)
    large_image_cold_start_policy = _large_image_cold_start_policy(
        provisioner=provisioner,
        simulator=simulator,
        live_provider_job=live_provider_job,
        startup_policy=startup_policy,
        warm_pool_policy=warm_pool_policy,
        worker_image_policy=worker_image_policy,
    )
    worker_blockers = _string_list(worker_launch_plan.get("blockers"))
    scheduler_blockers = _string_list(scheduler_decision.get("blockers"))
    worker_session_policy = _provider_worker_session_policy(
        provisioner=provisioner,
        worker_launch_plan=worker_launch_plan,
        generated_at=generated_at,
    )
    worker_session_blockers = _string_list(worker_session_policy.get("blockers"))
    blockers: list[str] = []
    warnings: list[str] = []

    if customer_gpu_job and gpu_allocation.get("allocation_allowed_by_webapp") is not False:
        blockers.append("webapp_gpu_allocation_boundary_missing")
    if customer_gpu_job and gpu_allocation.get("gpu_spend_approved") is not False:
        blockers.append("webapp_must_not_approve_gpu_spend")
    if live_provider_job and provisioner not in provider_priority:
        blockers.append("selected_provider_missing_from_startup_provider_priority")
    if customer_gpu_job and provider_is_marketplace and not marketplace_allowed:
        blockers.append("marketplace_provider_requires_explicit_customer_job_override")
    if customer_gpu_job and provider_is_marketplace and not strict_canary_required:
        blockers.append("marketplace_provider_requires_strict_preflight_canary")
    if worker_blockers:
        warnings.append("worker_launch_plan_has_blockers")
    if scheduler_blockers:
        warnings.append("scheduler_decision_has_blockers")
    if worker_session_blockers:
        blockers.extend(worker_session_blockers)
    large_image_cold_start_blockers = _string_list(
        large_image_cold_start_policy.get("blockers")
    )
    if large_image_cold_start_blockers:
        blockers.extend(large_image_cold_start_blockers)
    if (
        large_image_cold_start_policy.get("large_runpod_isaac_image") is True
        and large_image_cold_start_policy.get("image_startup_canary_launch") is not True
        and large_image_cold_start_policy.get("same_image_startup_canary_completed")
        is not True
    ):
        warnings.append("large_runpod_isaac_image_requires_same_image_startup_canary")

    status = (
        "blocked_before_customer_gpu_allocation"
        if blockers
        else "startup_pipeline_ready"
        if external_execution
        else "local_fixture_or_dev_path"
    )
    required_checks = _preflight_checks(
        runtime_preflight_contract=runtime_preflight_contract,
        live_provider_job=live_provider_job,
    )
    required_artifacts = [
        "worker_runtime_preflight.json",
        "startup_worker_canary_result.json",
    ]
    if large_image_cold_start_policy.get("large_runpod_isaac_image") is True:
        required_artifacts.append(RUNPOD_IMAGE_STARTUP_CANARY_ARTIFACT)
    return {
        "schema_version": GPU_STARTUP_PIPELINE_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": status,
        "strategy": "managed_gpu_worker_pool_with_strict_preflight",
        "provider_selection_owner": "BlueprintCapturePipeline",
        "selected_provider": provisioner,
        "selected_provider_tier": provider_tier,
        "selected_provider_is_marketplace": provider_is_marketplace,
        "selected_provider_is_managed": not provider_is_marketplace,
        "simulator": simulator,
        "website_queue_request": website_queue_request,
        "customer_gpu_job": customer_gpu_job,
        "webapp_boundary": {
            "webapp_role": execution_request.get("webapp_role")
            or "queue_and_forward_only",
            "webapp_gpu_spend_approval_allowed": False,
            "webapp_gpu_allocation_allowed": False,
            "pipeline_owns_provider_selection": True,
            "web_request_waits_for_simulator": False,
        },
        "managed_provider_policy": {
            "managed_provider_priority": managed_provider_priority,
            "provider_api_priority": provider_priority,
            "near_term_default": "runpod_secure_cloud",
            "secondary_default": "lambda_cloud",
            "enterprise_fallback": "aws_g6",
            "scale_reserved_capacity_fallback": "coreweave",
            "selected_provider_tier": provider_tier,
            "random_marketplace_host_disallowed_for_customer_default": True,
        },
        "marketplace_policy": {
            "customer_job_marketplace_default": (
                "avoid_unless_explicit_strict_preflight_canary"
            ),
            "marketplace_providers": ["vast", "runpod_community"],
            "selected_provider_is_marketplace": provider_is_marketplace,
            "marketplace_quarantine_required": provider_is_marketplace,
            "explicit_marketplace_customer_job_override": marketplace_allowed,
            "strict_preflight_canary_required": strict_canary_required,
            "customer_scene_load_allowed_before_canary_passes": False,
        },
        "preflight_canary_policy": {
            "required_before_customer_eval": bool(external_execution),
            "customer_eval_waits_for_canary": bool(external_execution),
            "block_scene_load_until_preflight_passes": bool(external_execution),
            "required_artifacts": _dedupe(required_artifacts),
            "same_image_startup_canary_required": bool(
                large_image_cold_start_policy.get(
                    "same_image_startup_canary_required_before_customer_eval"
                )
            ),
            "same_image_startup_canary_artifact": (
                RUNPOD_IMAGE_STARTUP_CANARY_ARTIFACT
                if large_image_cold_start_policy.get("large_runpod_isaac_image")
                is True
                else None
            ),
            "primary_result_artifact": runtime_preflight_contract.get(
                "result_artifact"
            )
            or "worker_runtime_preflight.json",
            "required_checks": required_checks,
            "runtime_preflight_contract": {
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
                "renderer_context": runtime_preflight_contract.get("renderer_context"),
                "nvidia_smi_required": bool(
                    runtime_preflight_contract.get("nvidia_smi_required")
                ),
            },
        },
        "provider_worker_session_policy": worker_session_policy,
        "same_sku_burst_policy": {
            "enabled": bool(live_provider_job),
            "burst_workers_must_use_same_image_ref": True,
            "burst_workers_must_use_same_gpu_family": True,
            "max_gpu_sku_families_per_customer_job": 1,
            "provider_worker_selection_disallows_random_hosts": True,
            "worker_image_ref": worker_image_policy.get("configured_image_ref"),
            "worker_image_version_pin_required": True,
        },
        "worker_image_policy": worker_image_policy,
        "large_image_cold_start_policy": large_image_cold_start_policy,
        "warm_pool_policy": {
            "decision": warm_pool_policy.get("decision") or "scale_to_zero_on_demand",
            "warm_worker_recommended": bool(
                warm_pool_policy.get("warm_worker_recommended")
            ),
            "scale_to_zero_default": bool(
                warm_pool_policy.get("scale_to_zero_default") is not False
            ),
            "active_worker_target": warm_pool_policy.get("active_worker_target"),
            "max_active_workers": warm_pool_policy.get("max_active_workers"),
            "decision_reasons": _string_list(warm_pool_policy.get("decision_reasons")),
        },
        "cache_policy": _cache_policy(worker_launch_plan),
        "cost_policy": {
            "budget_required_before_live_allocation": bool(live_provider_job),
            "max_budget_usd": gpu_allocation.get("max_budget_usd")
            or gpu_allocation.get("requested_budget_usd"),
            "record_actual_gpu_time_required": True,
            "idle_shutdown_required": bool(
                gpu_allocation.get("idle_shutdown_required") is not False
            ),
            "max_active_workers": _mapping(worker_launch_plan.get("launch_mode")).get(
                "max_active_workers"
            ),
        },
        "launcher_contract": {
            "startup_pipeline_plan_path": "gpu_startup_pipeline_plan.json",
            "provider_launch_request_path": "gpu_provider_launch_request.json",
            "worker_launch_plan_path": "worker_launch_plan.json",
            "worker_manifest_path": "worker_manifest.json",
            "launcher_must_fail_closed_on_startup_blockers": True,
            "live_provider_calls_allowed_by_default": False,
        },
        "inherited_status": {
            "scheduler_decision_status": scheduler_decision.get("status"),
            "worker_launch_plan_status": worker_launch_plan.get("status"),
            "scheduler_blockers": scheduler_blockers,
            "worker_blockers": worker_blockers,
        },
        "live_provider_calls_performed": False,
        "secret_values_in_artifact": False,
        "blockers": _dedupe(blockers),
        "warnings": _dedupe(warnings),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def build_gpu_startup_pipeline_plan_for_job_dir(
    *,
    job_dir: str | Path,
    output_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    job_path = Path(job_dir).expanduser().resolve()
    request = _mapping(read_json_any(job_path / "job_request.json"))
    scheduler_decision = _mapping(read_json_any(job_path / "scheduler_decision.json"))
    worker_launch_plan = _mapping(read_json_any(job_path / "worker_launch_plan.json"))
    selection = _mapping(scheduler_decision.get("selection"))
    job_id = _string(request.get("job_id")) or job_path.name
    provisioner = (
        _string(worker_launch_plan.get("provider"))
        or _string(selection.get("provisioner"))
        or "fixture_local"
    )
    simulator = (
        _string(worker_launch_plan.get("simulator"))
        or _string(selection.get("simulator"))
        or "fixture"
    )
    plan = build_gpu_startup_pipeline_plan(
        request=request,
        job_id=job_id,
        provisioner=provisioner,
        simulator=simulator,
        scheduler_decision=scheduler_decision,
        worker_launch_plan=worker_launch_plan,
        generated_at=generated_at,
    )
    destination = Path(output_path).expanduser() if output_path else job_path / (
        "gpu_startup_pipeline_plan.json"
    )
    ensure_dir(destination.parent)
    write_json(destination, plan)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a managed GPU startup policy plan for a robot-eval job."
    )
    parser.add_argument("--job-dir", required=True, help="Robot eval job directory")
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to <job-dir>/gpu_startup_pipeline_plan.json",
    )
    args = parser.parse_args(argv)
    build_gpu_startup_pipeline_plan_for_job_dir(
        job_dir=args.job_dir,
        output_path=args.output,
    )
    print(str(Path(args.output).expanduser()) if args.output else str(Path(args.job_dir) / "gpu_startup_pipeline_plan.json"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
