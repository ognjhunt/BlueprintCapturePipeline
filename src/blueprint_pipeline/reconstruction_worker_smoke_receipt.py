"""Normalize Vast worker-smoke evidence into the canonical trainer receipt."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .reconstruction_vast_worker_smoke import (
    ReconstructionVastSmokeError,
    validate_worker_smoke_result,
)
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_worker_build_receipt,
    build_worker_smoke_receipt,
    build_worker_stack_manifest,
)


class ReconstructionWorkerSmokeReceiptError(ValueError):
    def __init__(self, codes: list[str]) -> None:
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReconstructionWorkerSmokeReceiptError(
            ["worker_smoke_normalization_input_not_json"]
        ) from exc


def _finite(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _canonical_receipt(
    value: Mapping[str, Any], *, schema: str, digest_field: str
) -> dict[str, Any]:
    receipt = _clone(value)
    if receipt.get("schema_version") != schema:
        raise ReconstructionWorkerSmokeReceiptError(
            [f"worker_smoke_{digest_field}_schema_invalid"]
        )
    if receipt.get(digest_field) != canonical_digest(
        receipt, digest_field=digest_field
    ):
        raise ReconstructionWorkerSmokeReceiptError(
            [f"worker_smoke_{digest_field}_digest_mismatch"]
        )
    return receipt


def compile_reconstruction_worker_smoke_receipt(
    *,
    worker_stack_manifest: Mapping[str, Any],
    worker_build_receipt: Mapping[str, Any],
    bound_request: Mapping[str, Any],
    provider_runtime_result: Mapping[str, Any],
    execution_result: Mapping[str, Any],
    teardown_receipt: Mapping[str, Any],
    provider_zero_verification: Mapping[str, Any],
    execution_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile accepted runtime evidence without upgrading scientific claims."""

    try:
        stack = build_worker_stack_manifest(worker_stack_manifest)
        build = build_worker_build_receipt(worker_build_receipt)
    except ReconstructionWorkerContractError as exc:
        raise ReconstructionWorkerSmokeReceiptError(
            ["worker_smoke_stack_or_build_receipt_invalid"]
        ) from exc
    request = _clone(bound_request)
    execution = _canonical_receipt(
        execution_result,
        schema="reconstruction_vast_worker_smoke_execution.v1",
        digest_field="execution_result_digest",
    )
    teardown = _canonical_receipt(
        teardown_receipt,
        schema="reconstruction_vast_teardown_receipt.v1",
        digest_field="teardown_receipt_digest",
    )
    provider_zero = _canonical_receipt(
        provider_zero_verification,
        schema="reconstruction_vast_provider_zero_verification.v1",
        digest_field="provider_zero_digest",
    )
    authority = _clone(execution_authority)
    errors: list[str] = []

    if request.get("bound_request_digest") != canonical_digest(
        request, digest_field="bound_request_digest"
    ):
        errors.append("worker_smoke_bound_request_digest_mismatch")
    if (
        request.get("schema_version") != "reconstruction_gpu_canary_request.v1"
        or request.get("operation") != "worker_smoke"
        or request.get("expected_runtime_result_schema")
        != "reconstruction_vast_worker_smoke_result.v1"
        or request.get("bound_provider") != "vast"
        or request.get("bound_checkout_clean") is not True
        or request.get("provider_mutation_authorized") is not True
        or request.get("candidate_may_read_hidden_heldout") is not False
        or request.get("trainer_may_grade_heldout") is not False
        or request.get("proof_effect") != "none"
    ):
        errors.append("worker_smoke_bound_request_not_accepted")
    image = build.get("resolved_image_digest")
    if (
        build.get("status") != "built"
        or build.get("blockers") != []
        or build.get("worker_stack_manifest_digest")
        != stack.get("worker_stack_manifest_digest")
        or build.get("source_commit_sha") != stack.get("source_commit_sha")
        or request.get("worker_stack_manifest_digest")
        != stack.get("worker_stack_manifest_digest")
        or request.get("source_commit_sha") != stack.get("source_commit_sha")
        or request.get("bound_checkout_source_commit") != stack.get("source_commit_sha")
        or request.get("worker_image_digest") != image
    ):
        errors.append("worker_smoke_build_request_binding_invalid")
    try:
        runtime = validate_worker_smoke_result(
            provider_runtime_result,
            request_digest=str(request.get("request_digest") or ""),
            worker_image_digest=str(image or ""),
            source_commit_sha=str(stack.get("source_commit_sha") or ""),
        )
    except ReconstructionVastSmokeError as exc:
        raise ReconstructionWorkerSmokeReceiptError(
            [f"worker_smoke_runtime_result_invalid:{code}" for code in str(exc).split(";")]
        ) from exc

    request_digest = request.get("request_digest")
    bound_digest = request.get("bound_request_digest")
    if (
        execution.get("status") != "completed"
        or execution.get("blockers") != []
        or execution.get("provider") != "vast"
        or execution.get("request_digest") != request_digest
        or execution.get("bound_request_digest") != bound_digest
        or execution.get("worker_image_digest") != image
        or execution.get("source_commit_sha") != stack.get("source_commit_sha")
        or execution.get("provider_runtime_result_digest")
        != runtime.get("runtime_result_digest")
        or execution.get("teardown_receipt_digest")
        != teardown.get("teardown_receipt_digest")
        or execution.get("provider_zero_digest") != provider_zero.get("provider_zero_digest")
        or execution.get("provider_zero_verified") is not True
        or execution.get("provider_mutation_outcome_ambiguous") is not False
        or execution.get("scientific_qualification_inferred") is not False
        or execution.get("proof_effect") != "none"
    ):
        errors.append("worker_smoke_execution_not_accepted")
    if (
        teardown.get("status") != "PASS"
        or teardown.get("provider") != "vast"
        or teardown.get("request_digest") != request_digest
        or teardown.get("bound_request_digest") != bound_digest
        or teardown.get("worker_image_digest") != image
        or teardown.get("provider_zero_verified") is not True
    ):
        errors.append("worker_smoke_teardown_not_accepted")
    if (
        provider_zero.get("status") != "PASS"
        or provider_zero.get("provider") != "vast"
        or provider_zero.get("request_digest") != request_digest
        or provider_zero.get("bound_request_digest") != bound_digest
        or provider_zero.get("api_confirmed") is not True
        or provider_zero.get("scoped_live_resource_count") != 0
        or provider_zero.get("global_live_resource_count") != 0
    ):
        errors.append("worker_smoke_provider_zero_not_accepted")

    cost = execution.get("cost_usd")
    duration = execution.get("duration_seconds")
    if (
        not _finite(cost)
        or not _finite(duration)
        or not _finite(request.get("max_spend_usd"), minimum=0.01)
        or float(cost or 0) > float(request.get("max_spend_usd") or 0)
        or not isinstance(request.get("hard_ttl_seconds"), int)
        or float(duration or 0) > request.get("hard_ttl_seconds", 0)
    ):
        errors.append("worker_smoke_execution_bounds_invalid")
    if (
        authority.get("authority_id") != request.get("authority_id")
        or authority.get("paid_compute_authorized") is not True
        or authority.get("provider_processing_authorized") is not True
        or authority.get("max_spend_usd") != request.get("max_spend_usd")
        or authority.get("hard_ttl_seconds") != request.get("hard_ttl_seconds")
        or authority.get("retry_cap") != request.get("retry_cap")
    ):
        errors.append("worker_smoke_execution_authority_invalid")
    health = runtime.get("healthcheck")
    health = health if isinstance(health, Mapping) else {}
    health_checks = health.get("checks")
    health_checks = health_checks if isinstance(health_checks, list) else []
    check_ids = [
        row.get("check_id") for row in health_checks if isinstance(row, Mapping)
    ]
    if (
        not health_checks
        or len(check_ids) != len(health_checks)
        or any(not isinstance(check_id, str) or not check_id for check_id in check_ids)
        or len(check_ids) != len(set(check_ids))
        or any(
            not isinstance(row, Mapping) or row.get("status") != "passed"
            for row in health_checks
        )
    ):
        errors.append("worker_smoke_healthcheck_ledger_invalid")
    if errors:
        raise ReconstructionWorkerSmokeReceiptError(errors)

    checks = [
        {
            "check_id": str(row["check_id"]),
            "status": "passed",
            "output_digest": canonical_digest({"healthcheck_observation": dict(row)}),
        }
        for row in health_checks
    ]
    return build_worker_smoke_receipt(
        {
            "build_receipt_digest": build["build_receipt_digest"],
            "resolved_image_digest": image,
            "source_commit_sha": stack["source_commit_sha"],
            "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
            "status": "passed",
            "checks": checks,
            "display_attached": False,
            "scientific_qualification_inferred": False,
            "provider_runtime_identity": {
                "provider": "vast",
                "runtime": "gpu-canary",
                "instance_id": execution.get("instance_id"),
            },
            "runtime_result_digest": runtime["runtime_result_digest"],
            "healthcheck_digest": health["healthcheck_digest"],
            "execution_result_digest": execution["execution_result_digest"],
            "teardown_receipt_digest": teardown["teardown_receipt_digest"],
            "provider_zero_digest": provider_zero["provider_zero_digest"],
            "provider_zero_verified": True,
            "cost_usd": float(cost),
            "duration_seconds": float(duration),
            "authority_used": authority,
            "proof_effect": "none",
            "claim_ceiling": "worker_image_compatibility_only",
        }
    )


__all__ = [
    "ReconstructionWorkerSmokeReceiptError",
    "compile_reconstruction_worker_smoke_receipt",
]
