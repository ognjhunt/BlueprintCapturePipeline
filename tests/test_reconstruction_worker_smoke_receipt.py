from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_worker_contracts import (
    PINNED_MODEL_ASSETS,
    build_worker_build_receipt,
    build_worker_stack_manifest,
)
from blueprint_pipeline.reconstruction_worker_image_healthcheck import SCHEMA_VERSION
from blueprint_pipeline.reconstruction_worker_smoke_receipt import (
    ReconstructionWorkerSmokeReceiptError,
    compile_reconstruction_worker_smoke_receipt,
)


D = ["sha256:" + str(index) * 64 for index in range(1, 7)]
SHA = "a" * 40
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64


def _stack() -> dict:
    return build_worker_stack_manifest(
        {
            "worker_family": "blueprint-reconstruction-worker",
            "runnable_platform": "linux/amd64",
            "headless_required": True,
            "display_required": False,
            "source_commit_sha": SHA,
            "qualification_status": "candidate_unbuilt",
            "minimum_vram_gb": 24,
            "supported_compute_capabilities": [75, 80, 86, 89],
            "tested_driver_range": {"status": "not_yet_tested"},
            "model_assets": list(PINNED_MODEL_ASSETS),
            "hidden_heldout_access": False,
            "trainer_self_grading": False,
        }
    )


def _build(stack: dict) -> dict:
    return build_worker_build_receipt(
        {
            "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
            "status": "built",
            "resolved_image_digest": IMAGE,
            "source_commit_sha": SHA,
            "build_context_digest": D[0],
            "duration_seconds": 120.0,
            "cost_usd": 0.2,
            "logs": [{"artifact_id": "build.log", "digest": D[1]}],
            "blockers": [],
            "scientific_qualification_inferred": False,
        }
    )


def _request(stack: dict) -> dict:
    value = {
        "schema_version": "reconstruction_gpu_canary_request.v1",
        "operation": "worker_smoke",
        "capture_profile": "trainer_smoke_fixture",
        "source_commit_sha": SHA,
        "worker_image_digest": IMAGE,
        "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
        "reconstruction_dataset_digest": D[0],
        "frozen_split_digest": D[1],
        "calibration_digest": D[2],
        "deterministic_configuration_digest": D[3],
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 600,
        "retry_cap": 0,
        "authority_id": "human-paid-authority-fixture",
        "proof_effect": "none",
        "request_digest": D[4],
        "bound_provider": "vast",
        "bound_preflight_digest": D[5],
        "bound_checkout_source_commit": SHA,
        "bound_checkout_clean": True,
        "provider_mutation_authorized": True,
    }
    value["bound_request_digest"] = canonical_digest(
        value, digest_field="bound_request_digest"
    )
    return value


def _runtime(request: dict) -> dict:
    health = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-08-01T12:00:00Z",
        "status": "passed",
        "mode": "gpu_runtime",
        "checks": [
            {"check_id": "worker_family", "status": "passed"},
            {"check_id": "nvidia_runtime", "status": "passed"},
        ],
        "blockers": [],
        "display_attached": False,
        "runtime_identity": {
            "worker_family": "blueprint-reconstruction-worker",
            "source_commit_sha": SHA,
            "container_image_digest": IMAGE,
        },
        "hidden_heldout_observations_accessed": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "worker_image_compatibility_only",
    }
    health["healthcheck_digest"] = canonical_digest(
        health, digest_field="healthcheck_digest"
    )
    value = {
        "schema_version": "reconstruction_vast_worker_smoke_result.v1",
        "status": "passed",
        "request_digest": request["request_digest"],
        "worker_image_digest": IMAGE,
        "healthcheck": health,
        "hidden_heldout_observations_accessed": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "worker_image_compatibility_only",
    }
    value["runtime_result_digest"] = canonical_digest(
        value, digest_field="runtime_result_digest"
    )
    return value


def _teardown(request: dict) -> dict:
    value = {
        "schema_version": "reconstruction_vast_teardown_receipt.v1",
        "status": "PASS",
        "provider": "vast",
        "request_digest": request["request_digest"],
        "bound_request_digest": request["bound_request_digest"],
        "worker_image_digest": IMAGE,
        "instance_id": "vast-42",
        "terminate_result": {"status": "stopped", "instance_id": "vast-42"},
        "provider_zero_verified": True,
        "timestamp": "2026-08-01T12:05:00Z",
    }
    value["teardown_receipt_digest"] = canonical_digest(
        value, digest_field="teardown_receipt_digest"
    )
    return value


def _provider_zero(request: dict) -> dict:
    value = {
        "schema_version": "reconstruction_vast_provider_zero_verification.v1",
        "status": "PASS",
        "provider": "vast",
        "request_digest": request["request_digest"],
        "bound_request_digest": request["bound_request_digest"],
        "scoped_live_resource_count": 0,
        "global_live_resource_count": 0,
        "api_confirmed": True,
        "timestamp": "2026-08-01T12:05:01Z",
    }
    value["provider_zero_digest"] = canonical_digest(
        value, digest_field="provider_zero_digest"
    )
    return value


def _execution(request: dict, runtime: dict, teardown: dict, zero: dict) -> dict:
    value = {
        "schema_version": "reconstruction_vast_worker_smoke_execution.v1",
        "status": "completed",
        "request_digest": request["request_digest"],
        "bound_request_digest": request["bound_request_digest"],
        "worker_image_digest": IMAGE,
        "source_commit_sha": SHA,
        "provider": "vast",
        "instance_id": "vast-42",
        "provider_runtime_result_digest": runtime["runtime_result_digest"],
        "duration_seconds": 120.0,
        "cost_usd": 0.02,
        "provider_mutations_performed": 2,
        "provider_mutation_outcome_ambiguous": False,
        "blockers": [],
        "teardown_receipt_digest": teardown["teardown_receipt_digest"],
        "provider_zero_digest": zero["provider_zero_digest"],
        "provider_zero_verified": True,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "worker_image_compatibility_only",
    }
    value["execution_result_digest"] = canonical_digest(
        value, digest_field="execution_result_digest"
    )
    return value


def _authority() -> dict:
    return {
        "authority_id": "human-paid-authority-fixture",
        "paid_compute_authorized": True,
        "provider_processing_authorized": True,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 600,
        "retry_cap": 0,
    }


def _inputs() -> dict:
    stack = _stack()
    request = _request(stack)
    runtime = _runtime(request)
    teardown = _teardown(request)
    zero = _provider_zero(request)
    return {
        "worker_stack_manifest": stack,
        "worker_build_receipt": _build(stack),
        "bound_request": request,
        "provider_runtime_result": runtime,
        "execution_result": _execution(request, runtime, teardown, zero),
        "teardown_receipt": teardown,
        "provider_zero_verification": zero,
        "execution_authority": _authority(),
    }


def test_vast_evidence_normalizes_to_canonical_non_scientific_smoke_receipt() -> None:
    receipt = compile_reconstruction_worker_smoke_receipt(**_inputs())

    assert receipt["schema_version"] == "reconstruction_worker_smoke_test_receipt.v1"
    assert receipt["status"] == "passed"
    assert receipt["resolved_image_digest"] == IMAGE
    assert receipt["provider_runtime_identity"]["provider"] == "vast"
    assert receipt["provider_zero_verified"] is True
    assert receipt["cost_usd"] == 0.02
    assert receipt["scientific_qualification_inferred"] is False
    assert receipt["proof_effect"] == "none"
    assert receipt["claim_ceiling"] == "worker_image_compatibility_only"
    assert receipt["smoke_test_receipt_digest"] == canonical_digest(
        receipt, digest_field="smoke_test_receipt_digest"
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_worker_smoke_test_receipt.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(receipt, schema)


def test_tampered_teardown_or_provider_zero_cannot_normalize() -> None:
    inputs = _inputs()
    inputs["provider_zero_verification"]["global_live_resource_count"] = 1
    inputs["provider_zero_verification"]["provider_zero_digest"] = canonical_digest(
        inputs["provider_zero_verification"], digest_field="provider_zero_digest"
    )
    with pytest.raises(
        ReconstructionWorkerSmokeReceiptError,
        match="worker_smoke_provider_zero_not_accepted",
    ):
        compile_reconstruction_worker_smoke_receipt(**inputs)

    inputs = _inputs()
    inputs["teardown_receipt"]["status"] = "FAIL"
    inputs["teardown_receipt"]["teardown_receipt_digest"] = canonical_digest(
        inputs["teardown_receipt"], digest_field="teardown_receipt_digest"
    )
    with pytest.raises(
        ReconstructionWorkerSmokeReceiptError,
        match="worker_smoke_teardown_not_accepted",
    ):
        compile_reconstruction_worker_smoke_receipt(**inputs)


def test_prompt_authority_cannot_replace_bound_paid_authority() -> None:
    inputs = _inputs()
    inputs["execution_authority"] = copy.deepcopy(inputs["execution_authority"])
    inputs["execution_authority"]["authority_id"] = "agent-derived-from-user-prompt"
    with pytest.raises(
        ReconstructionWorkerSmokeReceiptError,
        match="worker_smoke_execution_authority_invalid",
    ):
        compile_reconstruction_worker_smoke_receipt(**inputs)


def test_smoke_cost_or_duration_cannot_exceed_bound_envelope() -> None:
    for field, value in (("cost_usd", 1.01), ("duration_seconds", 601.0)):
        inputs = _inputs()
        inputs["execution_result"][field] = value
        inputs["execution_result"]["execution_result_digest"] = canonical_digest(
            inputs["execution_result"], digest_field="execution_result_digest"
        )
        with pytest.raises(
            ReconstructionWorkerSmokeReceiptError,
            match="worker_smoke_execution_bounds_invalid",
        ):
            compile_reconstruction_worker_smoke_receipt(**inputs)
