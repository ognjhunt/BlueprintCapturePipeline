"""Managed GPU startup policy for website-origin robot-eval jobs.

The plan produced here is deterministic policy plumbing. It does not launch a
provider, allocate a GPU, or upgrade simulator artifacts into robot readiness.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json


GPU_STARTUP_PIPELINE_PLAN_SCHEMA_VERSION = "robot_eval_gpu_startup_pipeline_plan.v1"

LOCAL_PROVISIONERS = {"fixture_local", "local_process", "docker_local"}
LIVE_GPU_PROVISIONERS = {"runpod", "vast", "gcp"}
PROVIDER_PRIORITY = ["runpod", "gcp", "vast"]
MANAGED_PROVIDER_PRIORITY = [
    "runpod_secure_cloud",
    "lambda_cloud",
    "aws_g6",
    "coreweave",
]
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
    "physical_robot_readiness_proven": False,
    "deployment_readiness_proven": False,
    "safety_validation_proven": False,
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


def _worker_image_policy(worker_launch_plan: Mapping[str, Any]) -> dict[str, Any]:
    image = _mapping(worker_launch_plan.get("worker_image"))
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
    worker_blockers = _string_list(worker_launch_plan.get("blockers"))
    scheduler_blockers = _string_list(scheduler_decision.get("blockers"))
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
            "required_artifacts": [
                "worker_runtime_preflight.json",
                "startup_worker_canary_result.json",
            ],
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
