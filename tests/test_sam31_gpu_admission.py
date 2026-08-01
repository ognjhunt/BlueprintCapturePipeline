import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sam31_gpu_admission import (
    CHECKPOINT_DIGEST,
    CHECKPOINT_REPOSITORY_REVISION,
    LICENSE_TERMS_DIGEST,
    OFFICIAL_CODE_REVISION,
    PREFLIGHT_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    build_sam31_gpu_canary_admission,
    collect_sam31_vast_preflight,
    prepare_sam31_gpu_canary,
)


SHA = "sha256:" + "a" * 64
COMMIT = "b" * 40
IMAGE = "registry.example/blueprint/sam31@sha256:" + "c" * 64


def _request() -> dict:
    value = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "source_track_canary",
        "source_profile": "monocular_video",
        "source_commit_sha": COMMIT,
        "worker_image_digest": IMAGE,
        "worker_stack_manifest_digest": SHA,
        "input_bundle_digest": SHA,
        "input_bundle_size_bytes": 1024,
        "source_track_run_request_digest": SHA,
        "capture_digest": SHA,
        "retained_video_digest": SHA,
        "camera_solution_digest": SHA,
        "frame_registry_digest": SHA,
        "frame_count": 16,
        "checkpoint_family": "facebook/sam3.1",
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "checkpoint_repository_revision": CHECKPOINT_REPOSITORY_REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "license_terms_digest": LICENSE_TERMS_DIGEST,
        "license_use_authorization_digest": SHA,
        "privacy_use_authorization_digest": SHA,
        "trade_controls_review_digest": SHA,
        "execution_authorization_digest": SHA,
        "checkpoint_access_authorized": True,
        "commercial_evidence_use_authorized": True,
        "rights_cleared_for_external_processing": True,
        "privacy_safe_for_external_processing": True,
        "trade_controls_reviewed": True,
        "model_self_grading_forbidden": True,
        "metric_claim_upgrade_forbidden": True,
        "physics_claim_upgrade_forbidden": True,
        "physical_claim_upgrade_forbidden": True,
        "network_access_during_inference_forbidden": True,
        "customer_data_training_allowed": False,
        "allowed_evidence_uses": ["semantic_analysis"],
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 600,
        "retry_cap": 0,
        "authority_id": "design-partner-beta-authorization",
        "proof_effect": "none",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


def _preflight(*, live_resources: int = 0) -> dict:
    value = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "verified" if live_resources == 0 else "blocked",
        "provider": "vast",
        "observed_at_epoch": 1_000.0,
        "provider_api_verified": True,
        "provider_inventory_verified_zero": live_resources == 0,
        "conflicting_owner_present": False,
        "watchdog": {"status": "armed", "independent_process": True},
        "single_gpu_available": True,
        "gpu_type_id": "L40S",
        "gpu_memory_bytes": 48 * 1024**3,
        "container_disk_bytes": 80 * 1024**3,
        "on_demand_price_usd_per_hour": 0.50,
        "selected_offer": {"gpu_name": "L40S"},
        "blockers": [] if live_resources == 0 else ["sam31_gpu_provider_inventory_not_zero"],
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "capacity_reserved": False,
        "proof_effect": "none",
        "claim_ceiling": "provider_capacity_and_zero_inventory_snapshot_only",
    }
    value["preflight_digest"] = canonical_digest(value, digest_field="preflight_digest")
    return value


def _build(*, request=None, preflight=None, execute=False, qualified=False):
    return build_sam31_gpu_canary_admission(
        request=request or _request(),
        preflight=preflight or _preflight(),
        provider="vast",
        expected_source_commit=COMMIT,
        checkout_source_commit=COMMIT,
        checkout_clean=True,
        max_spend_usd=1.0,
        hard_ttl_seconds=600,
        retry_cap=0,
        authority_id="design-partner-beta-authorization",
        execute=execute,
        execution_adapter_qualified=qualified,
        observed_now_epoch=1_001.0,
    )


def test_dry_run_binds_exact_model_input_and_claim_ceiling() -> None:
    admission, bound = _build()
    assert admission["status"] == "dry_run_ready"
    assert admission["blockers"] == []
    assert admission["checkpoint_digest"] == CHECKPOINT_DIGEST
    assert admission["scientific_qualification_inferred"] is False
    assert admission["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    assert bound["provider_mutation_authorized"] is False
    assert bound["bound_request_digest"] == canonical_digest(
        bound, digest_field="bound_request_digest"
    )


def test_execute_requires_qualified_adapter_and_then_authorizes_one_mutation() -> None:
    blocked, blocked_bound = _build(execute=True, qualified=False)
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["sam31_vast_execution_adapter_not_qualified"]
    assert blocked_bound["provider_mutation_authorized"] is False

    ready, ready_bound = _build(execute=True, qualified=True)
    assert ready["status"] == "execute_ready"
    assert ready["blockers"] == []
    assert ready_bound["provider_mutation_authorized"] is True


@pytest.mark.parametrize(
    ("field", "replacement", "blocker"),
    [
        ("checkpoint_digest", SHA, "sam31_gpu_checkpoint_digest_mismatch"),
        ("official_code_revision", "d" * 40, "sam31_gpu_official_code_revision_mismatch"),
        ("license_terms_digest", SHA, "sam31_gpu_license_terms_digest_mismatch"),
        ("model_self_grading_forbidden", False, "sam31_gpu_model_self_grading_forbidden_required"),
        (
            "metric_claim_upgrade_forbidden",
            False,
            "sam31_gpu_metric_claim_upgrade_forbidden_required",
        ),
        (
            "physics_claim_upgrade_forbidden",
            False,
            "sam31_gpu_physics_claim_upgrade_forbidden_required",
        ),
        (
            "physical_claim_upgrade_forbidden",
            False,
            "sam31_gpu_physical_claim_upgrade_forbidden_required",
        ),
        (
            "comparative_policy_ranking_verdict",
            "inconclusive",
            "sam31_gpu_comparative_policy_ranking_verdict_mismatch",
        ),
    ],
)
def test_exact_science_and_license_boundaries_fail_closed(
    field: str, replacement, blocker: str
) -> None:
    request = _request()
    request[field] = replacement
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    admission, bound = _build(request=request, execute=True, qualified=True)
    assert admission["status"] == "blocked"
    assert blocker in admission["blockers"]
    assert bound["provider_mutation_authorized"] is False


def test_provider_nonzero_and_stale_preflight_block_before_mutation() -> None:
    nonzero, bound = _build(preflight=_preflight(live_resources=1), execute=True, qualified=True)
    assert nonzero["status"] == "blocked"
    assert "sam31_gpu_provider_inventory_not_zero" in nonzero["blockers"]
    assert bound["provider_mutation_authorized"] is False

    stale = _preflight()
    stale["observed_at_epoch"] = 1.0
    stale["preflight_digest"] = canonical_digest(stale, digest_field="preflight_digest")
    admission, _ = _build(preflight=stale)
    assert "sam31_gpu_preflight_stale_or_future" in admission["blockers"]


def test_budget_ttl_and_retry_are_exactly_bound() -> None:
    request = _request()
    request["max_spend_usd"] = 0.01
    request["hard_ttl_seconds"] = 3_601
    request["retry_cap"] = 2
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    admission, _ = build_sam31_gpu_canary_admission(
        request=request,
        preflight=_preflight(),
        provider="vast",
        expected_source_commit=COMMIT,
        checkout_source_commit=COMMIT,
        checkout_clean=True,
        max_spend_usd=0.01,
        hard_ttl_seconds=3_601,
        retry_cap=2,
        authority_id="design-partner-beta-authorization",
        execute=True,
        execution_adapter_qualified=True,
        observed_now_epoch=1_001.0,
    )
    assert "sam31_gpu_explicit_ttl_invalid" in admission["blockers"]
    assert "sam31_gpu_explicit_retry_cap_invalid" in admission["blockers"]
    assert "sam31_gpu_budget_below_worst_case_cost" in admission["blockers"]


def test_processed_only_profile_and_oversized_bundle_fail_closed() -> None:
    request = _request()
    request["source_profile"] = "processed_public_dataset"
    request["input_bundle_size_bytes"] = 512 * 1024**2 + 1
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    admission, _ = _build(request=request, execute=True, qualified=True)
    assert "sam31_gpu_source_profile_unsupported" in admission["blockers"]
    assert "sam31_gpu_input_bundle_size_invalid" in admission["blockers"]


def test_preflight_collects_provider_zero_without_mutation() -> None:
    prefixes: list[str] = []

    def inventory(prefix: str) -> dict:
        prefixes.append(prefix)
        return {"api_confirmed": True, "live_resource_count": 0}

    result = collect_sam31_vast_preflight(
        name_prefix="blueprint-sam31-",
        container_disk_bytes=80 * 1024**3,
        watchdog={"status": "armed", "independent_process": True},
        conflicting_owner_present=False,
        capacity_probe=lambda _request: {
            "status": "available",
            "selected_offer": {
                "gpu_name": "L40S",
                "gpu_ram_mb": 48_000,
                "hourly_rate_usd": 0.50,
            },
        },
        inventory_probe=inventory,
        max_hourly_rate_usd=1.0,
        clock=lambda: 100.0,
    )
    assert result["status"] == "verified"
    assert result["provider_mutations_performed"] == 0
    assert prefixes == ["blueprint-sam31-", ""]


def test_prepare_writes_fail_closed_artifacts(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    preflight_path = tmp_path / "preflight.json"
    request_path.write_text(json.dumps(_request()), encoding="utf-8")
    preflight = _preflight()
    preflight["observed_at_epoch"] = time.time()
    preflight["preflight_digest"] = canonical_digest(preflight, digest_field="preflight_digest")
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    admission_path = tmp_path / "admission.json"
    bound_path = tmp_path / "bound.json"
    adapter_path = tmp_path / "adapter.json"
    result = prepare_sam31_gpu_canary(
        request_path=request_path,
        preflight_path=preflight_path,
        admission_out=admission_path,
        bound_request_out=bound_path,
        adapter_output=adapter_path,
        provider="vast",
        expected_source_commit=COMMIT,
        checkout_source_commit=COMMIT,
        checkout_clean=True,
        max_spend_usd=1.0,
        hard_ttl_seconds=600,
        retry_cap=0,
        authority_id="design-partner-beta-authorization",
        execute=False,
    )
    assert result["status"] == "dry_run_ready"
    assert json.loads(admission_path.read_text())["status"] == "dry_run_ready"
    assert json.loads(bound_path.read_text())["provider_mutation_authorized"] is False
    assert json.loads(adapter_path.read_text())["paid_execution_started"] is False
