"""Asynchronous RunPod warm-capacity reconciliation policy.

The customer bind path only creates a durable scale request.  This module is
the separate mutation boundary that may satisfy one claimed request.  It
enforces L40S-first and permits one A40 attempt only after RunPod explicitly
rejects the L40S create without allocating a pod.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence


AUTOSCALER_SCHEMA_VERSION = "production_gpu_runpod_autoscaler.v1"
PRIMARY_GPU = "NVIDIA L40S"
FALLBACK_GPU = "NVIDIA A40"
RUNPOD_GPU_POOL_CLASS = "runpod-secure-l40s-preferred-a40-fallback"


class ScalePool(Protocol):
    def claim_scale_request(
        self, *, autoscaler_id: str, lease_seconds: float = 120.0
    ) -> dict[str, Any] | None: ...

    def release_scale_request(
        self, *, scale_request_id: str, scale_token: str, retryable: bool
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RunPodAutoscalerAuthorization:
    allow_paid: bool = False
    total_spend_cap_usd: float = 20.0
    combined_gpu_wall_time_cap_seconds: int = 19_154
    spent_usd: float = 0.0
    used_gpu_wall_time_seconds: int = 0
    attempt_wall_time_limit_seconds: int = 300
    max_hourly_rate_usd: float = 1.0
    allow_a40_after_capacity_rejection: bool = True

    def blockers(self) -> list[str]:
        blockers: list[str] = []
        if not self.allow_paid:
            blockers.append("paid_runpod_autoscaling_not_authorized")
        if not 0 < float(self.total_spend_cap_usd) <= 20:
            blockers.append("total_spend_cap_exceeds_authorized_usd_20")
        if not 0 < int(self.combined_gpu_wall_time_cap_seconds) <= 19_154:
            blockers.append("gpu_wall_time_cap_exceeds_authorized_19154_seconds")
        if float(self.spent_usd) >= float(self.total_spend_cap_usd):
            blockers.append("total_spend_cap_exhausted")
        worst_case_attempt_spend = (
            float(self.max_hourly_rate_usd)
            * int(self.attempt_wall_time_limit_seconds)
            / 3600.0
        )
        if worst_case_attempt_spend > (
            float(self.total_spend_cap_usd) - float(self.spent_usd)
        ):
            blockers.append("insufficient_remaining_spend_for_attempt")
        if not 0 < float(self.max_hourly_rate_usd) <= 1.0:
            blockers.append("runpod_hourly_rate_exceeds_authorized_limit")
        remaining_wall = int(self.combined_gpu_wall_time_cap_seconds) - int(
            self.used_gpu_wall_time_seconds
        )
        if int(self.attempt_wall_time_limit_seconds) > remaining_wall:
            blockers.append("insufficient_remaining_gpu_wall_time_for_attempt")
        if not 30 <= int(self.attempt_wall_time_limit_seconds) <= 840:
            blockers.append("autoscaler_attempt_wall_time_limit_out_of_range")
        return blockers


def _capacity_rejected_without_allocation(result: Mapping[str, Any]) -> bool:
    return (
        result.get("capacity_outcome") is True
        and result.get("allocation_created") is False
        and not result.get("instance_id")
        and result.get("allocation_outcome_ambiguous") is not True
    )


def reconcile_one_scale_request(
    *,
    pool: ScalePool,
    autoscaler_id: str,
    exact_worker_image_ref: str,
    authorization: RunPodAutoscalerAuthorization,
    launcher: Callable[[Mapping[str, Any], Sequence[str], int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Claim and reconcile one deficit outside the customer request path.

    ``launcher`` owns provider mutation, pending-teardown persistence, and the
    eventual warm-serve probe.  Its input is deliberately restricted to the
    claimed release, one GPU type, and a bounded wall-time.
    """

    blockers = authorization.blockers()
    if blockers:
        return {
            "schema_version": AUTOSCALER_SCHEMA_VERSION,
            "status": "blocked_before_scale_claim",
            "blockers": blockers,
            "provider_calls_performed": 0,
        }
    return {
        "schema_version": AUTOSCALER_SCHEMA_VERSION,
        "status": "blocked_before_scale_claim",
        "blockers": [
            "legacy_runpod_autoscaler_disabled_use_paid_resource_allocator"
        ],
        "provider_calls_performed": 0,
    }
    claim = pool.claim_scale_request(
        autoscaler_id=autoscaler_id,
        lease_seconds=float(authorization.attempt_wall_time_limit_seconds + 60),
    )
    if claim is None:
        return {
            "schema_version": AUTOSCALER_SCHEMA_VERSION,
            "status": "idle_no_scale_request",
            "provider_calls_performed": 0,
        }
    request_id = str(claim["scale_request_id"])
    scale_token = str(claim["scale_token"])
    if str(claim.get("worker_image_ref") or "") != str(exact_worker_image_ref):
        pool.release_scale_request(
            scale_request_id=request_id, scale_token=scale_token, retryable=False
        )
        return {
            "schema_version": AUTOSCALER_SCHEMA_VERSION,
            "status": "blocked_release_mismatch",
            "blockers": ["claimed_worker_image_not_exact_release"],
            "provider_calls_performed": 0,
            "scale_request_id": request_id,
        }
    if str(claim.get("gpu_family") or "") != RUNPOD_GPU_POOL_CLASS:
        pool.release_scale_request(
            scale_request_id=request_id, scale_token=scale_token, retryable=False
        )
        return {
            "schema_version": AUTOSCALER_SCHEMA_VERSION,
            "status": "blocked_gpu_pool_class_mismatch",
            "blockers": ["claimed_gpu_family_not_runpod_l40s_a40_pool_class"],
            "provider_calls_performed": 0,
            "scale_request_id": request_id,
        }

    attempts: list[dict[str, Any]] = []
    primary = dict(
        launcher(
            claim,
            (PRIMARY_GPU,),
            int(authorization.attempt_wall_time_limit_seconds),
        )
    )
    attempts.append({"gpu_type": PRIMARY_GPU, "result": primary})
    selected = primary
    if (
        _capacity_rejected_without_allocation(primary)
        and authorization.allow_a40_after_capacity_rejection
    ):
        fallback = dict(
            launcher(
                claim,
                (FALLBACK_GPU,),
                int(authorization.attempt_wall_time_limit_seconds),
            )
        )
        attempts.append({"gpu_type": FALLBACK_GPU, "result": fallback})
        selected = fallback

    if selected.get("instance_id"):
        # Registration of the same exact release satisfies the claimed demand.
        # Keep the claim leased so another autoscaler cannot duplicate capacity.
        return {
            "schema_version": AUTOSCALER_SCHEMA_VERSION,
            "status": "capacity_launched_waiting_for_worker_registration",
            "scale_request_id": request_id,
            "instance_id": selected["instance_id"],
            "gpu_pool_class": RUNPOD_GPU_POOL_CLASS,
            "actual_gpu_model": attempts[-1]["gpu_type"],
            "attempts": attempts,
            "customer_request_provider_calls": 0,
            "provider_calls_performed": len(attempts),
        }

    if selected.get("allocation_outcome_ambiguous") is True:
        # Do not release the claim: that could create a duplicate paid worker.
        return {
            "schema_version": AUTOSCALER_SCHEMA_VERSION,
            "status": "manual_provider_reconciliation_required",
            "scale_request_id": request_id,
            "attempts": attempts,
            "provider_calls_performed": len(attempts),
            "blockers": ["runpod_allocation_outcome_ambiguous"],
        }

    released = pool.release_scale_request(
        scale_request_id=request_id,
        scale_token=scale_token,
        retryable=True,
    )
    return {
        "schema_version": AUTOSCALER_SCHEMA_VERSION,
        "status": "capacity_not_allocated_retryable",
        "scale_request_id": request_id,
        "attempts": attempts,
        "provider_calls_performed": len(attempts),
        "scale_request_status": released.get("status"),
    }
