from __future__ import annotations

from blueprint_pipeline.production_gpu_runpod_autoscaler import (
    RunPodAutoscalerAuthorization,
    reconcile_one_scale_request,
)


IMAGE = "docker.io/blueprint/worker@sha256:" + "a" * 64


class Pool:
    def __init__(self, image: str = IMAGE) -> None:
        self.image = image
        self.released: list[dict] = []

    def claim_scale_request(self, *, autoscaler_id: str, lease_seconds: float = 120) -> dict:
        return {
            "scale_request_id": "gps-1",
            "scale_token": "gsc-token",
            "worker_image_ref": self.image,
            "gpu_family": "runpod-secure-l40s-preferred-a40-fallback",
            "autoscaler_id": autoscaler_id,
            "lease_seconds": lease_seconds,
        }

    def release_scale_request(self, **kwargs) -> dict:
        self.released.append(kwargs)
        return {"status": "pending" if kwargs["retryable"] else "cancelled"}


def _auth(**changes) -> RunPodAutoscalerAuthorization:
    values = {
        "allow_paid": True,
        "total_spend_cap_usd": 20,
        "combined_gpu_wall_time_cap_seconds": 10_980,
        "spent_usd": 1,
        "used_gpu_wall_time_seconds": 7_745,
        "attempt_wall_time_limit_seconds": 300,
    }
    values.update(changes)
    return RunPodAutoscalerAuthorization(**values)


def test_l40s_is_the_only_first_attempt() -> None:
    calls: list[tuple[str, ...]] = []

    def launch(_claim, gpu_types, _limit):
        calls.append(tuple(gpu_types))
        return {"instance_id": "pod-1", "status": "launched"}

    result = reconcile_one_scale_request(
        pool=Pool(), autoscaler_id="autoscaler-1", exact_worker_image_ref=IMAGE,
        authorization=_auth(), launcher=launch,
    )

    assert calls == [("NVIDIA L40S",)]
    assert result["status"] == "capacity_launched_waiting_for_worker_registration"


def test_a40_fallback_requires_authoritative_no_allocation_capacity_rejection() -> None:
    outcomes = [
        {
            "capacity_outcome": True,
            "allocation_created": False,
            "allocation_outcome_ambiguous": False,
        },
        {"instance_id": "pod-a40", "status": "launched"},
    ]
    calls: list[tuple[str, ...]] = []

    def launch(_claim, gpu_types, _limit):
        calls.append(tuple(gpu_types))
        return outcomes.pop(0)

    result = reconcile_one_scale_request(
        pool=Pool(), autoscaler_id="autoscaler-1", exact_worker_image_ref=IMAGE,
        authorization=_auth(), launcher=launch,
    )

    assert calls == [("NVIDIA L40S",), ("NVIDIA A40",)]
    assert result["instance_id"] == "pod-a40"


def test_ambiguous_l40s_outcome_never_falls_back_or_releases_claim() -> None:
    pool = Pool()
    calls: list[tuple[str, ...]] = []

    def launch(_claim, gpu_types, _limit):
        calls.append(tuple(gpu_types))
        return {"allocation_outcome_ambiguous": True}

    result = reconcile_one_scale_request(
        pool=pool, autoscaler_id="autoscaler-1", exact_worker_image_ref=IMAGE,
        authorization=_auth(), launcher=launch,
    )

    assert calls == [("NVIDIA L40S",)]
    assert result["status"] == "manual_provider_reconciliation_required"
    assert pool.released == []


def test_no_paid_mutation_or_claim_without_explicit_authorization() -> None:
    calls = []
    pool = Pool()

    result = reconcile_one_scale_request(
        pool=pool, autoscaler_id="autoscaler-1", exact_worker_image_ref=IMAGE,
        authorization=_auth(allow_paid=False),
        launcher=lambda *_args: calls.append(True) or {},
    )

    assert result["status"] == "blocked_before_scale_claim"
    assert result["provider_calls_performed"] == 0
    assert calls == []


def test_combined_wall_time_cap_is_enforced_before_claim() -> None:
    result = reconcile_one_scale_request(
        pool=Pool(), autoscaler_id="autoscaler-1", exact_worker_image_ref=IMAGE,
        authorization=_auth(used_gpu_wall_time_seconds=10_800),
        launcher=lambda *_args: {},
    )

    assert "insufficient_remaining_gpu_wall_time_for_attempt" in result["blockers"]


def test_spend_headroom_is_enforced_before_claim() -> None:
    result = reconcile_one_scale_request(
        pool=Pool(), autoscaler_id="autoscaler-1", exact_worker_image_ref=IMAGE,
        authorization=_auth(spent_usd=19.99),
        launcher=lambda *_args: {},
    )

    assert "insufficient_remaining_spend_for_attempt" in result["blockers"]


def test_concrete_l40s_fingerprint_cannot_silently_accept_a40_fallback() -> None:
    pool = Pool()
    original_claim = pool.claim_scale_request

    def concrete_claim(**kwargs):
        claim = original_claim(**kwargs)
        claim["gpu_family"] = "NVIDIA L40S"
        return claim

    pool.claim_scale_request = concrete_claim  # type: ignore[method-assign]
    calls = []
    result = reconcile_one_scale_request(
        pool=pool, autoscaler_id="autoscaler-1", exact_worker_image_ref=IMAGE,
        authorization=_auth(), launcher=lambda *_args: calls.append(True) or {},
    )

    assert result["status"] == "blocked_gpu_pool_class_mismatch"
    assert calls == []
