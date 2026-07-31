from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.reconstruction_provider_contracts import (
    ReconstructionProviderContractError,
    build_reconstruction_provider_admission,
    build_reconstruction_provider_deletion_receipt,
    build_reconstruction_provider_execution_receipt,
    build_reconstruction_provider_execution_request,
    require_reconstruction_provider_execution_authority,
)
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.phase2_artifacts import (
    authorization_receipt,
    authorization_request,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import ToolRegistry, non_spend_tool_bindings


D = ["sha256:" + character * 64 for character in "abcdef"]
ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ACTIONS = [
    "confidential_capture_upload",
    "provider_deletion",
    "provider_output_download",
    "provider_reconstruction_execution",
]


def _admission(**updates) -> dict:
    value = {
        "stable_run_identity": "provider-run-1",
        "provider_identity": "qualified-fixture-provider",
        "provider_product": "fixture-reconstruction",
        "product_tier": "commercial",
        "terms_version": "fixture-terms-1",
        "terms_digest": D[3],
        "legal_reviewer_identity": "fixture-legal-reviewer",
        "legal_reviewer_is_agent": False,
        "legal_review_receipt_digest": D[4],
        "provider_capability_review_digest": D[5],
        "source_capture_digest": D[0],
        "reviewed_at": "2026-07-30T13:00:00Z",
        "review_expires_at": "2026-08-30T13:00:00Z",
        "commercial_use_authorized": True,
        "confidential_capture_upload_authorized_by_terms": True,
        "retention_terms_acceptable": True,
        "deletion_process_verified": True,
        "model_training_terms_acceptable": True,
        "competitive_use_terms_acceptable": True,
        "resale_terms_acceptable": True,
        "benchmarking_terms_acceptable": True,
        "programmatic_upload_job_download_api_qualified": True,
        "canonical_paid_allocator_route_qualified": True,
        "trusted_legal_review_accepted": True,
        "provider_credentials_available": True,
        "provider_mutations_performed": False,
        "proof_effect": "none",
        "claim_ceiling": "none",
    }
    value.update(updates)
    return build_reconstruction_provider_admission(value)


def _execution_request(**updates) -> dict:
    inputs = sorted(D[:3])
    auth_request = authorization_request(
        run_id="provider-run-1",
        tool_id="invoke_authorized_reconstruction_provider",
        reason="Run an exact rights-cleared provider fixture",
        requested_max_cost_usd=10.0,
        requested_ttl_seconds=3600,
        requested_retry_count=1,
        immutable_input_digests=inputs,
        requested_provider_ids=["qualified-fixture-provider"],
        requested_action_ids=REQUIRED_ACTIONS,
    )
    auth_receipt = authorization_receipt(
        request=auth_request,
        operator_id="fixture-operator",
        approved=True,
        granted_max_cost_usd=8.0,
        granted_ttl_seconds=3600,
        granted_retry_count=1,
        issued_at="2026-07-30T13:00:00Z",
        expires_at="2026-07-30T14:00:00Z",
        granted_provider_ids=["qualified-fixture-provider"],
        granted_action_ids=REQUIRED_ACTIONS,
    )
    admission = _admission()
    value = {
        "stable_run_identity": "provider-run-1",
        "source_capture_identity": "capture-1",
        "source_capture_digest": D[0],
        "original_file_references": [{"artifact_id": "capture", "digest": D[0]}],
        "producing_method": "provider_execution_request_compiler",
        "implementation_version": "1",
        "source_commit_sha": "1" * 40,
        "deterministic_configuration_digest": D[1],
        "train_heldout_split_digest": D[2],
        "input_digests": [
            {"artifact_id": f"input-{index}", "digest": digest}
            for index, digest in enumerate(inputs)
        ],
        "output_digests": [],
        "camera_calibration_binding": {"status": "provider_input_bound"},
        "coordinate_frame_declaration": {"status": "provider_input_bound"},
        "units": "unverified",
        "metric_scale_status": "unverified",
        "container_image_digest": None,
        "provider_runtime_identity": {"provider_identity": "qualified-fixture-provider"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "provider_identity": "qualified-fixture-provider",
        "provider_admission": admission,
        "provider_admission_digest": admission["provider_admission_digest"],
        "authorization_receipt": auth_receipt,
        "authorization_receipt_digest": auth_receipt["authorization_receipt_digest"],
        "authority_used": {
            "authorization_receipt_digest": auth_receipt["authorization_receipt_digest"]
        },
        "immutable_input_digests": inputs,
        "authorized_actions": REQUIRED_ACTIONS,
        "max_cost_usd": 8.0,
        "ttl_seconds": 3600,
        "retry_cap": 1,
        "timestamp": "2026-07-30T13:05:00Z",
        "authority_expires_at": "2026-07-30T14:00:00Z",
        "warnings": [],
        "blockers": [],
        "parent_artifact_or_event": {"digest": admission["provider_admission_digest"]},
        "operator_upload_authorized": True,
        "confidential_capture_processing_authorized": True,
        "spending_authorized": True,
        "post_job_deletion_required": True,
        "authorization_issued_by_agent": False,
        "candidate_may_read_hidden_heldout": False,
        "proof_effect": "provider_execution_request_only",
        "claim_ceiling": "none",
    }
    value.update(updates)
    return build_reconstruction_provider_execution_request(value)


def test_current_scaniverse_remote_posture_fails_closed_without_mutation() -> None:
    admission = _admission(
        provider_identity="scaniverse",
        provider_product="scaniverse-web",
        commercial_use_authorized=False,
        confidential_capture_upload_authorized_by_terms=False,
        deletion_process_verified=False,
        model_training_terms_acceptable=False,
        competitive_use_terms_acceptable=False,
        benchmarking_terms_acceptable=False,
        programmatic_upload_job_download_api_qualified=False,
        canonical_paid_allocator_route_qualified=False,
        trusted_legal_review_accepted=False,
        provider_credentials_available=False,
    )
    assert admission["status"] == "blocked"
    assert admission["provider_mutations_performed"] is False
    assert "provider_programmatic_execution_api_unqualified" in admission["blockers"]
    assert "canonical_paid_allocator_route_unqualified" in admission["blockers"]


def test_provider_request_execution_and_deletion_receipts_preserve_claim_boundaries() -> None:
    request = _execution_request()
    execution = build_reconstruction_provider_execution_receipt(
        {
            "provider_execution_request_digest": request["provider_execution_request_digest"],
            "source_capture_digest": D[0],
            "train_heldout_split_digest": D[2],
            "provider_identity": "qualified-fixture-provider",
            "provider_job_identity": "fixture-job-1",
            "status": "succeeded_unqualified",
            "cost_usd": 4.0,
            "duration_seconds": 120.0,
            "attempt_count": 1,
            "provider_runtime_identity": {
                "provider_identity": "qualified-fixture-provider",
                "runtime_identity": "fixture-runtime-1",
            },
            "downloaded_outputs": [
                {"artifact_id": "asset", "digest": D[3], "download_complete": True, "hash_verified": True}
            ],
            "failure": None,
            "deletion_status": "pending",
            "warnings": [],
            "blockers": [],
            "timestamp": "2026-07-30T13:07:00Z",
            "provider_success_is_blueprint_qualification": False,
            "metric_scale_proven": False,
            "collision_geometry_validated": False,
            "isaac_compatibility_proven": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
            "proof_effect": "provider_output_derived_support_only",
            "claim_ceiling": "external_reconstruction_import",
        },
        request=request,
    )
    deletion = build_reconstruction_provider_deletion_receipt(
        {
            "provider_execution_receipt_digest": execution[
                "provider_execution_receipt_digest"
            ],
            "provider_identity": "qualified-fixture-provider",
            "status": "verified_deleted",
            "provider_evidence": {"deletion_job_identity": "delete-1", "status": "deleted"},
            "timestamp": "2026-07-30T13:10:00Z",
            "independently_verified": True,
            "provider_zero_proven": False,
            "proof_effect": "provider_deletion_evidence_only",
            "claim_ceiling": "none",
        },
        execution_receipt=execution,
    )
    assert execution["status"] == "succeeded_unqualified"
    assert execution["provider_success_is_blueprint_qualification"] is False
    assert deletion["provider_zero_proven"] is False
    for name, artifact in (
        ("reconstruction_provider_admission.v1.schema.json", request["provider_admission"]),
        ("reconstruction_provider_execution_request.v1.schema.json", request),
        ("reconstruction_provider_execution_receipt.v1.schema.json", execution),
        ("reconstruction_provider_deletion_receipt.v1.schema.json", deletion),
    ):
        schema = json.loads((ROOT / "docs/schemas" / name).read_text(encoding="utf-8"))
        jsonschema.validate(artifact, schema)


def test_provider_contract_rejects_agent_authority_stale_scope_and_claim_promotion() -> None:
    request = _execution_request()
    assert require_reconstruction_provider_execution_authority(
        request, at_time="2026-07-30T13:30:00Z"
    ) == request
    with pytest.raises(ReconstructionProviderContractError, match="authority_expired"):
        require_reconstruction_provider_execution_authority(
            request, at_time="2026-07-30T14:00:00Z"
        )
    altered = dict(request)
    altered.pop("provider_execution_request_digest")
    altered["authorization_issued_by_agent"] = True
    with pytest.raises(ReconstructionProviderContractError, match="agent_authority_forbidden"):
        build_reconstruction_provider_execution_request(altered)

    with pytest.raises(ReconstructionProviderContractError, match="budget_exceeded"):
        build_reconstruction_provider_execution_receipt(
            {
                "provider_execution_request_digest": request[
                    "provider_execution_request_digest"
                ],
                "source_capture_digest": D[0],
                "train_heldout_split_digest": D[2],
                "provider_identity": "qualified-fixture-provider",
                "provider_job_identity": "fixture-job-2",
                "status": "failed",
                "cost_usd": 9.0,
                "duration_seconds": 1.0,
                "attempt_count": 1,
                "provider_runtime_identity": {
                    "provider_identity": "qualified-fixture-provider",
                    "runtime_identity": "fixture-runtime-2",
                },
                "downloaded_outputs": [],
                "failure": {"code": "budget_exhaustion", "retryable": False},
                "deletion_status": "pending",
                "warnings": [],
                "blockers": ["budget_exhaustion"],
                "timestamp": "2026-07-30T13:07:00Z",
                "provider_success_is_blueprint_qualification": False,
                "metric_scale_proven": False,
                "collision_geometry_validated": False,
                "isaac_compatibility_proven": False,
                "physical_success_proven": False,
                "deployment_readiness_proven": False,
                "proof_effect": "provider_output_derived_support_only",
                "claim_ceiling": "external_reconstruction_import",
            },
            request=request,
        )


def test_remote_provider_tool_is_registered_but_not_callable_without_adapter(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry.default()
    descriptor = registry.resolve("invoke_authorized_reconstruction_provider")
    assert descriptor is not None
    assert descriptor.to_mapping()["mutability"] == "external_side_effect"
    context = SupervisorContext(
        run_id="provider-run-1",
        customer_question="Invoke provider",
        supervisor_output_dir=str(tmp_path),
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_PREAUTHORIZED,
        tool_registry=registry,
        action_max_cost_usd=10.0,
        action_ttl_seconds=3600,
        action_max_retries=1,
        immutable_input_digests=D[:3],
        preauthorization_receipt_digest=D[3],
        preauthorization_expires_at="2026-07-30T14:00:00Z",
    ).to_mapping()
    bindings = non_spend_tool_bindings(
        capability="capture_testbed_supervisor",
        context=context,
        registry=registry,
        authority=authority,
    )
    assert "invoke_authorized_reconstruction_provider" not in {
        binding.tool_id for binding in bindings
    }
