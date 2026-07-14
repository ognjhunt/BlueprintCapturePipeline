import time

from blueprint_pipeline.production_gpu_launch_qualification import build_launch_qualification
from blueprint_pipeline.production_gpu_worker_pool import REQUIRED_READY_CHECKS, release_fingerprint


IMAGE = "docker.io/blueprint/worker@sha256:" + "a" * 64
HOST = "runpod-secure-l40s-active-worker-v1"
GPU = "runpod-secure-l40s-preferred-a40-fallback"


def _registration(worker_id: str) -> dict:
    return {
        "registration_payload": {
            "worker_id": worker_id,
            "host_image_id": HOST,
            "worker_image_ref": IMAGE,
            "gpu_family": GPU,
            "readiness": {name: True for name in REQUIRED_READY_CHECKS},
        }
    }


def _inputs() -> dict:
    fingerprint = release_fingerprint(
        host_image_id=HOST, worker_image_ref=IMAGE, gpu_family=GPU
    )
    return {
        "host_image_id": HOST,
        "worker_image_ref": IMAGE,
        "gpu_family": GPU,
        "min_ready_workers": 2,
        "registrations": [_registration("pod-1"), _registration("pod-2")],
        "pool_snapshot": {"worker_counts": {"ready": 2}},
        "bind_probe": {
            "schema_version": "production_gpu_warm_bind_probe.v1",
            "status": "passed",
            "sample_count": 20,
            "warm_bind_p95_seconds": 1.5,
            "provider_calls_performed": 0,
        },
        "replenishment_probe": {
            "schema_version": "production_gpu_replenishment_probe.v1",
            "status": "passed",
            "sample_count": 3,
            "release_fingerprint": fingerprint,
            "cold_replenishment_p95_seconds": 500,
            "customer_request_provider_calls": 0,
            "provisioning_occurred_asynchronously": True,
        },
        "rollback_drill": {
            "schema_version": "production_gpu_rollback_drill.v1",
            "status": "passed",
            "release_fingerprint": fingerprint,
            "candidate_quarantined_before_provider_teardown": True,
            "customer_bind_routed_to_quarantined_worker": False,
        },
        "current_provider_inventory": {
            "api_confirmed": True,
            "observed_at_epoch": time.time(),
            "live_resource_count": 2,
            "resources": [{"id": "pod-1"}, {"id": "pod-2"}],
        },
        "teardown_drill": {
            "status": "PASS",
            "api_confirmed_absent": True,
            "teardown_proof": {"status": "PASS"},
        },
    }


def test_complete_current_live_evidence_opens_customer_launch_gate() -> None:
    result = build_launch_qualification(**_inputs())

    assert result["status"] == "customer_launch_ready"
    assert result["blockers"] == []
    assert result["distinct_registered_worker_ids"] == ["pod-1", "pod-2"]


def test_historical_rehearsal_with_zero_current_inventory_stays_closed() -> None:
    values = _inputs()
    values["current_provider_inventory"] = {
        "api_confirmed": True,
        "observed_at_epoch": time.time(),
        "live_resource_count": 0,
        "resources": [],
    }
    result = build_launch_qualification(**values)

    assert result["status"] == "local_contract_ready_live_proof_required"
    assert "live_evidence_missing:current_warm_capacity" in result["blockers"]
    assert "live_evidence_missing:provider_inventory" in result["blockers"]


def test_duplicate_registration_cannot_satisfy_two_worker_minimum() -> None:
    values = _inputs()
    values["registrations"] = [_registration("pod-1"), _registration("pod-1")]
    result = build_launch_qualification(**values)

    assert "live_evidence_missing:ready_worker_count" in result["blockers"]
