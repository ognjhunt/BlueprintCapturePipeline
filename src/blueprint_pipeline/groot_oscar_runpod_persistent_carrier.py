"""Canonical admission for the five-call persistent GR00T + learned-WAM lane."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .groot_oscar_infrastructure_admission import (
    build_runpod_serve_plane_admission,
)
from .groot_oscar_runpod_carrier_volume import verify_carrier_volume_admission


PERSISTENT_CARRIER_PROBE_KIND = "persistent-policy-wam-loop"
PERSISTENT_POLICY_CALL_COUNT = 5
PERSISTENT_LEARNED_WAM_GENERATION_COUNT = 4
PERSISTENT_CONTAINER_DISK_GIB = 240
PERSISTENT_NETWORK_VOLUME_GIB = 120
PERSISTENT_LOOP_MAX_WAIT_SECONDS = 18_000
PERSISTENT_WATCHDOG_MAX_TTL_SECONDS = 18_600
PERSISTENT_CARRIER_IMAGE_REF = (
    "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:"
    "b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def prepare_persistent_carrier_launch(
    *,
    request: Mapping[str, Any],
    release: Mapping[str, Any],
    model_cache: Mapping[str, Any],
    preflight: Mapping[str, Any],
    carrier_volume_admission: Mapping[str, Any],
    loop_step_count: int,
    max_wait_seconds: int,
) -> dict[str, Any]:
    """Bind one exact carrier, volume, runtime, model cache, GPU, and campaign."""

    volume = _mapping(preflight.get("volume"))
    runtime = _mapping(preflight.get("runtime"))
    spend = _mapping(preflight.get("spend"))
    serve = build_runpod_serve_plane_admission(
        release=release,
        model_cache=model_cache,
        volume=volume,
        runtime=runtime,
        spend=spend,
        maximum_ttl_seconds=PERSISTENT_WATCHDOG_MAX_TTL_SECONDS,
    )
    carrier = verify_carrier_volume_admission(carrier_volume_admission)
    blockers = [*serve.get("blockers", []), *carrier.get("blockers", [])]
    if preflight.get("status") != "verified":
        blockers.append("persistent_carrier_preflight_not_verified")
    if loop_step_count != PERSISTENT_POLICY_CALL_COUNT:
        blockers.append("persistent_carrier_requires_exactly_five_policy_calls")
    if max_wait_seconds != PERSISTENT_LOOP_MAX_WAIT_SECONDS:
        blockers.append("persistent_carrier_requires_18000_second_run_bound")
    if (
        type(spend.get("hard_ttl_seconds")) is not int
        or int(spend.get("hard_ttl_seconds") or 0)
        < PERSISTENT_WATCHDOG_MAX_TTL_SECONDS
    ):
        blockers.append("persistent_carrier_watchdog_below_18600_seconds")
    if carrier.get("source_release_image_ref") != serve.get("release_image_ref"):
        blockers.append("persistent_carrier_source_release_mismatch")
    if carrier.get("network_volume_id") != serve.get("network_volume_id"):
        blockers.append("persistent_carrier_network_volume_mismatch")
    if carrier.get("data_center_id") != serve.get("data_center_id"):
        blockers.append("persistent_carrier_data_center_mismatch")
    if carrier.get("model_cache_root") != serve.get("model_cache_path"):
        blockers.append("persistent_carrier_model_cache_path_mismatch")
    if carrier.get("size_gib") != PERSISTENT_NETWORK_VOLUME_GIB:
        blockers.append("persistent_carrier_requires_exactly_120_gib_volume")
    if carrier.get("carrier_image_ref") != PERSISTENT_CARRIER_IMAGE_REF:
        blockers.append("persistent_carrier_exact_image_digest_mismatch")
    gpu_type = _string(serve.get("gpu_type_id"))
    if not gpu_type or "H100" in gpu_type.upper():
        blockers.append("persistent_carrier_h100_disallowed")

    bound = deepcopy(dict(request))
    shape = _mapping(bound.get("provider_request_shape"))
    image = _mapping(shape.get("image"))
    image.update(
        {
            "configured_image_ref": carrier.get("carrier_image_ref"),
            "configured_image_ref_is_versioned": True,
            "configured_image_ref_fetchable_by_provider": True,
        }
    )
    gpu = _mapping(shape.get("gpu"))
    gpu.update(
        {
            "gpu_count": 1,
            "preferred_gpu_type_id": gpu_type,
            "provider_gpu_priority": [gpu_type] if gpu_type else [],
            "container_disk_in_gb": PERSISTENT_CONTAINER_DISK_GIB,
            "volume_in_gb": PERSISTENT_NETWORK_VOLUME_GIB,
        }
    )
    cache = _mapping(shape.get("cache"))
    cache_paths = _mapping(cache.get("paths"))
    cache_paths["groot_oscar_models"] = carrier.get("model_cache_root")
    cache["paths"] = cache_paths
    limits = _mapping(shape.get("limits"))
    limits.update(
        {
            "hard_timeout_seconds": PERSISTENT_LOOP_MAX_WAIT_SECONDS,
            "external_watchdog_ttl_seconds": spend.get("hard_ttl_seconds"),
            "max_active_workers": 1,
        }
    )
    shape.update(
        {
            "operation": "enqueue_runpod_persistent_policy_wam_loop",
            "image": image,
            "gpu": gpu,
            "cache": cache,
            "limits": limits,
            "network_volume_id": carrier.get("network_volume_id"),
            "data_center_id": carrier.get("data_center_id"),
            "allowed_cuda_versions": [serve.get("required_cuda_version")],
            "persistent_campaign": {
                "policy_call_count": PERSISTENT_POLICY_CALL_COUNT,
                "learned_wam_generation_count": (
                    PERSISTENT_LEARNED_WAM_GENERATION_COUNT
                ),
                "same_pod_required": True,
                "provider_output_replay_disallowed": True,
                "max_wait_seconds": PERSISTENT_LOOP_MAX_WAIT_SECONDS,
            },
            "claim_boundary": {
                "technical_persistent_loop_only": True,
                "does_not_prove_semantic_task_success": True,
                "does_not_prove_physical_robot_readiness": True,
            },
        }
    )
    bound["operation"] = "enqueue_runpod_persistent_policy_wam_loop"
    bound["provider_request_shape"] = shape
    unique_blockers = sorted(set(str(item) for item in blockers if str(item)))
    admission = {
        **serve,
        "status": "admitted" if not unique_blockers else "blocked",
        "blockers": unique_blockers,
        "probe_kind": PERSISTENT_CARRIER_PROBE_KIND,
        "carrier_volume": carrier,
        "campaign_contract": {
            "policy_call_count": PERSISTENT_POLICY_CALL_COUNT,
            "learned_wam_generation_count": PERSISTENT_LEARNED_WAM_GENERATION_COUNT,
            "container_disk_gib": PERSISTENT_CONTAINER_DISK_GIB,
            "network_volume_gib": PERSISTENT_NETWORK_VOLUME_GIB,
            "max_wait_seconds": PERSISTENT_LOOP_MAX_WAIT_SECONDS,
            "maximum_watchdog_ttl_seconds": PERSISTENT_WATCHDOG_MAX_TTL_SECONDS,
            "same_pod_required": True,
            "h100_allowed": False,
        },
        "claim_boundary": {
            "runtime_and_model_bytes_preverified": not unique_blockers,
            "provider_attachment_not_yet_proven": True,
            "policy_and_wam_execution_not_yet_proven": True,
            "semantic_task_success_not_proven": True,
        },
    }
    return {
        "status": "admitted" if not unique_blockers else "blocked",
        "blockers": unique_blockers,
        "admission": admission,
        "bound_request": bound,
        "provider_mutations_performed": 0,
    }
