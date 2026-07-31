"""Fail-closed contracts for remote reconstruction-provider execution.

This module does not contact a provider. It records deterministic admission,
execution, and deletion evidence so a future adapter cannot treat a provider's
success response as Blueprint reconstruction qualification.
"""

from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.phase2_artifacts import (
    Phase2ArtifactError,
    validate_authorization_receipt,
)


PROVIDER_ADMISSION_SCHEMA = "reconstruction_provider_admission.v1"
PROVIDER_EXECUTION_REQUEST_SCHEMA = "reconstruction_provider_execution_request.v1"
PROVIDER_EXECUTION_RECEIPT_SCHEMA = "reconstruction_provider_execution_receipt.v1"
PROVIDER_DELETION_RECEIPT_SCHEMA = "reconstruction_provider_deletion_receipt.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_ACTIONS = {
    "confidential_capture_upload",
    "provider_reconstruction_execution",
    "provider_output_download",
    "provider_deletion",
}
_ADMISSION_GATES = {
    "commercial_use_authorized": "provider_commercial_use_not_authorized",
    "confidential_capture_upload_authorized_by_terms": "provider_confidential_upload_terms_unacceptable",
    "retention_terms_acceptable": "provider_retention_terms_unacceptable",
    "deletion_process_verified": "provider_deletion_process_unverified",
    "model_training_terms_acceptable": "provider_model_training_terms_unacceptable",
    "competitive_use_terms_acceptable": "provider_competitive_use_terms_unacceptable",
    "resale_terms_acceptable": "provider_resale_terms_unacceptable",
    "benchmarking_terms_acceptable": "provider_benchmarking_terms_unacceptable",
    "programmatic_upload_job_download_api_qualified": "provider_programmatic_execution_api_unqualified",
    "canonical_paid_allocator_route_qualified": "canonical_paid_allocator_route_unqualified",
    "trusted_legal_review_accepted": "trusted_legal_review_missing",
    "provider_credentials_available": "provider_credentials_unavailable",
}


class ReconstructionProviderContractError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _digest(value: Any, code: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        errors.append(code)


def _time(value: Any, code: str, errors: list[str]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        errors.append(code)
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(code)
        return None
    return parsed


def _number(value: Any, code: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(code)
        return -1.0
    number = float(value)
    if not math.isfinite(number) or number < 0:
        errors.append(code)
    return number


def _finalize(value: Mapping[str, Any], schema: str, digest_field: str) -> dict[str, Any]:
    result = dict(value)
    supplied = result.pop(digest_field, None)
    result["schema_version"] = schema
    expected = canonical_digest(result, digest_field=digest_field)
    if supplied is not None and supplied != expected:
        raise ReconstructionProviderContractError([f"{digest_field}_mismatch"])
    result[digest_field] = expected
    return result


def build_reconstruction_provider_admission(value: Mapping[str, Any]) -> dict[str, Any]:
    """Derive admission solely from trusted governance gate inputs."""

    admission = dict(value)
    errors: list[str] = []
    for key in (
        "stable_run_identity",
        "provider_identity",
        "provider_product",
        "product_tier",
        "terms_version",
        "legal_reviewer_identity",
    ):
        if not isinstance(admission.get(key), str) or not admission[key].strip():
            errors.append(f"{key}_missing")
    for key in (
        "source_capture_digest",
        "terms_digest",
        "legal_review_receipt_digest",
        "provider_capability_review_digest",
    ):
        _digest(admission.get(key), f"{key}_invalid", errors)
    reviewed_at = _time(admission.get("reviewed_at"), "provider_reviewed_at_invalid", errors)
    review_expires_at = _time(
        admission.get("review_expires_at"), "provider_review_expires_at_invalid", errors
    )
    if (
        reviewed_at is not None
        and review_expires_at is not None
        and review_expires_at <= reviewed_at
    ):
        errors.append("provider_review_window_invalid")
    if admission.get("legal_reviewer_is_agent") is not False:
        errors.append("provider_legal_review_cannot_be_agent_issued")
    blockers = sorted(
        blocker for field, blocker in _ADMISSION_GATES.items() if admission.get(field) is not True
    )
    expected_status = "admitted" if not blockers else "blocked"
    if admission.get("status") not in (None, expected_status):
        errors.append("provider_admission_status_not_derived")
    if admission.get("blockers") not in (None, blockers):
        errors.append("provider_admission_blockers_not_derived")
    if admission.get("provider_mutations_performed") is not False:
        errors.append("provider_admission_cannot_record_mutation")
    if admission.get("proof_effect") != "none" or admission.get("claim_ceiling") != "none":
        errors.append("provider_admission_claim_boundary_invalid")
    if errors:
        raise ReconstructionProviderContractError(errors)
    admission["status"] = expected_status
    admission["blockers"] = blockers
    return _finalize(admission, PROVIDER_ADMISSION_SCHEMA, "provider_admission_digest")


def build_reconstruction_provider_execution_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact provider request; blocked admission can never execute."""

    request = dict(value)
    errors: list[str] = []
    for key in (
        "stable_run_identity",
        "source_capture_identity",
        "producing_method",
        "implementation_version",
    ):
        if not isinstance(request.get(key), str) or not request[key].strip():
            errors.append(f"provider_execution_{key}_missing")
    for key in ("original_file_references", "input_digests", "warnings", "blockers"):
        if not isinstance(request.get(key), list):
            errors.append(f"provider_execution_{key}_invalid")
    if request.get("output_digests") != []:
        errors.append("provider_execution_request_cannot_predeclare_outputs")
    for key in (
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "provider_runtime_identity",
        "authority_used",
        "parent_artifact_or_event",
    ):
        if not isinstance(request.get(key), Mapping):
            errors.append(f"provider_execution_{key}_invalid")
    if request.get("units") not in {"meters", "unverified"}:
        errors.append("provider_execution_units_invalid")
    if request.get("metric_scale_status") not in {"verified", "unverified"}:
        errors.append("provider_execution_metric_scale_status_invalid")
    if request.get("container_image_digest") is not None:
        errors.append("provider_managed_runtime_cannot_claim_blueprint_container")
    if _number(request.get("cost_usd"), "provider_execution_request_cost_invalid", errors) != 0:
        errors.append("provider_execution_request_cost_must_be_zero")
    if (
        _number(
            request.get("duration_seconds"),
            "provider_execution_request_duration_invalid",
            errors,
        )
        != 0
    ):
        errors.append("provider_execution_request_duration_must_be_zero")
    admission_value = request.get("provider_admission")
    try:
        admission = build_reconstruction_provider_admission(
            admission_value if isinstance(admission_value, Mapping) else {}
        )
    except ReconstructionProviderContractError as exc:
        errors.extend(f"provider_admission:{code}" for code in exc.codes)
        admission = {}
    if admission.get("status") != "admitted":
        errors.append("provider_execution_not_admitted")
    if request.get("provider_identity") != admission.get("provider_identity"):
        errors.append("provider_execution_provider_mismatch")
    for key in (
        "source_capture_digest",
        "deterministic_configuration_digest",
        "train_heldout_split_digest",
        "authorization_receipt_digest",
        "provider_admission_digest",
    ):
        _digest(request.get(key), f"{key}_invalid", errors)
    if request.get("provider_admission_digest") != admission.get("provider_admission_digest"):
        errors.append("provider_execution_admission_digest_mismatch")
    if request.get("source_capture_digest") != admission.get("source_capture_digest"):
        errors.append("provider_execution_source_capture_mismatch")
    authorization_value = request.get("authorization_receipt")
    try:
        authorization = validate_authorization_receipt(
            authorization_value if isinstance(authorization_value, Mapping) else {}
        )
    except Phase2ArtifactError:
        errors.append("provider_execution_authorization_receipt_invalid")
        authorization = {}
    if authorization.get("approved") is not True:
        errors.append("provider_execution_not_operator_authorized")
    if authorization.get("granted_tool_id") != "invoke_authorized_reconstruction_provider":
        errors.append("provider_execution_wrong_authorized_tool")
    if request.get("authorization_receipt_digest") != authorization.get(
        "authorization_receipt_digest"
    ):
        errors.append("provider_execution_authorization_digest_mismatch")
    if request.get("stable_run_identity") != authorization.get("run_id"):
        errors.append("provider_execution_authorization_run_mismatch")
    if request.get("provider_identity") not in set(
        authorization.get("granted_provider_ids") or []
    ):
        errors.append("provider_execution_provider_not_authorized")
    if _COMMIT.fullmatch(str(request.get("source_commit_sha") or "")) is None:
        errors.append("provider_execution_source_commit_invalid")
    inputs = request.get("immutable_input_digests")
    if not isinstance(inputs, list) or not inputs:
        errors.append("provider_execution_inputs_missing")
        inputs = []
    if inputs != sorted(set(inputs)):
        errors.append("provider_execution_inputs_not_canonical")
    for item in inputs:
        _digest(item, "provider_execution_input_digest_invalid", errors)
    recorded_inputs = {
        item.get("digest")
        for item in request.get("input_digests") or []
        if isinstance(item, Mapping)
    }
    if recorded_inputs != set(inputs):
        errors.append("provider_execution_input_ledger_mismatch")
    required_inputs = {
        request.get("source_capture_digest"),
        request.get("deterministic_configuration_digest"),
        request.get("train_heldout_split_digest"),
    }
    if not required_inputs.issubset(set(inputs)):
        errors.append("provider_execution_required_input_binding_missing")
    if inputs != authorization.get("immutable_input_digests"):
        errors.append("provider_execution_authorized_inputs_mismatch")
    actions = request.get("authorized_actions")
    if not isinstance(actions, list) or set(actions) != _REQUIRED_ACTIONS:
        errors.append("provider_execution_authorized_actions_incomplete")
    if set(actions or []) != set(authorization.get("granted_action_ids") or []):
        errors.append("provider_execution_actions_not_authorized")
    max_cost = _number(request.get("max_cost_usd"), "provider_execution_max_cost_invalid", errors)
    if max_cost <= 0:
        errors.append("provider_execution_explicit_positive_budget_required")
    if max_cost > float(authorization.get("granted_max_cost_usd") or 0):
        errors.append("provider_execution_budget_exceeds_authority")
    ttl = request.get("ttl_seconds")
    retries = request.get("retry_cap")
    if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl < 1:
        errors.append("provider_execution_ttl_invalid")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        errors.append("provider_execution_retry_cap_invalid")
    if isinstance(ttl, int) and ttl > int(authorization.get("granted_ttl_seconds") or 0):
        errors.append("provider_execution_ttl_exceeds_authority")
    if isinstance(retries, int) and retries > int(
        authorization.get("granted_retry_count") or 0
    ):
        errors.append("provider_execution_retries_exceed_authority")
    compiled_at = _time(request.get("timestamp"), "provider_execution_timestamp_invalid", errors)
    expires_at = _time(request.get("authority_expires_at"), "provider_execution_expiry_invalid", errors)
    if compiled_at is not None and expires_at is not None and expires_at <= compiled_at:
        errors.append("provider_execution_authority_stale")
    review_expires_at = _time(
        admission.get("review_expires_at"), "provider_execution_review_expiry_invalid", errors
    )
    if (
        compiled_at is not None
        and review_expires_at is not None
        and review_expires_at <= compiled_at
    ):
        errors.append("provider_execution_legal_review_stale")
    if request.get("authority_expires_at") != authorization.get("expires_at"):
        errors.append("provider_execution_expiry_mismatch")
    for field in (
        "operator_upload_authorized",
        "confidential_capture_processing_authorized",
        "spending_authorized",
        "post_job_deletion_required",
    ):
        if request.get(field) is not True:
            errors.append(f"provider_execution_{field}_missing")
    if request.get("authorization_issued_by_agent") is not False:
        errors.append("provider_execution_agent_authority_forbidden")
    authority_used = request.get("authority_used")
    if not isinstance(authority_used, Mapping) or authority_used.get(
        "authorization_receipt_digest"
    ) != request.get("authorization_receipt_digest"):
        errors.append("provider_execution_authority_ledger_mismatch")
    parent = request.get("parent_artifact_or_event")
    if not isinstance(parent, Mapping) or parent.get("digest") != request.get(
        "provider_admission_digest"
    ):
        errors.append("provider_execution_parent_binding_mismatch")
    if request.get("candidate_may_read_hidden_heldout") is not False:
        errors.append("provider_execution_hidden_heldout_access_forbidden")
    if request.get("proof_effect") != "provider_execution_request_only" or request.get(
        "claim_ceiling"
    ) != "none":
        errors.append("provider_execution_request_claim_boundary_invalid")
    if errors:
        raise ReconstructionProviderContractError(errors)
    request["provider_admission"] = admission
    return _finalize(
        request, PROVIDER_EXECUTION_REQUEST_SCHEMA, "provider_execution_request_digest"
    )


def require_reconstruction_provider_execution_authority(
    value: Mapping[str, Any], *, at_time: str
) -> dict[str, Any]:
    """Recheck request authority immediately before a provider mutation."""

    request = build_reconstruction_provider_execution_request(value)
    errors: list[str] = []
    observed = _time(at_time, "provider_execution_observed_time_invalid", errors)
    issued = _time(
        request.get("authorization_receipt", {}).get("issued_at"),
        "provider_execution_authority_issued_at_invalid",
        errors,
    )
    expires = _time(
        request.get("authority_expires_at"), "provider_execution_expiry_invalid", errors
    )
    review_expires = _time(
        request.get("provider_admission", {}).get("review_expires_at"),
        "provider_execution_review_expiry_invalid",
        errors,
    )
    if observed is not None and issued is not None and observed < issued:
        errors.append("provider_execution_authority_not_yet_valid")
    if observed is not None and expires is not None and observed >= expires:
        errors.append("provider_execution_authority_expired")
    if observed is not None and review_expires is not None and observed >= review_expires:
        errors.append("provider_execution_legal_review_expired")
    if errors:
        raise ReconstructionProviderContractError(errors)
    return request


def build_reconstruction_provider_execution_receipt(
    value: Mapping[str, Any], *, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize untrusted provider output without accepting provider self-grading."""

    execution_request = build_reconstruction_provider_execution_request(request)
    receipt = dict(value)
    errors: list[str] = []
    inherited_fields = (
        "stable_run_identity",
        "source_capture_identity",
        "source_capture_digest",
        "original_file_references",
        "source_commit_sha",
        "deterministic_configuration_digest",
        "train_heldout_split_digest",
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "units",
        "metric_scale_status",
        "container_image_digest",
        "authority_used",
    )
    for field in inherited_fields:
        if field in receipt and receipt[field] != execution_request.get(field):
            errors.append(f"provider_receipt_{field}_mismatch")
        receipt[field] = execution_request.get(field)
    receipt["input_digests"] = list(execution_request["input_digests"])
    receipt["parent_artifact_or_event"] = {
        "digest": execution_request["provider_execution_request_digest"]
    }
    receipt["producing_method"] = "authorized_external_reconstruction_provider"
    receipt["implementation_version"] = "1"
    if receipt.get("provider_execution_request_digest") != execution_request.get(
        "provider_execution_request_digest"
    ):
        errors.append("provider_receipt_request_binding_mismatch")
    for key in ("source_capture_digest", "train_heldout_split_digest"):
        if receipt.get(key) != execution_request.get(key):
            errors.append(f"provider_receipt_{key}_mismatch")
    if receipt.get("provider_identity") != execution_request.get("provider_identity"):
        errors.append("provider_receipt_identity_mismatch")
    status = receipt.get("status")
    if status not in {"succeeded_unqualified", "failed", "interrupted"}:
        errors.append("provider_receipt_status_invalid")
    if not isinstance(receipt.get("provider_job_identity"), str) or not receipt.get(
        "provider_job_identity"
    ):
        errors.append("provider_receipt_job_identity_missing")
    cost = _number(receipt.get("cost_usd"), "provider_receipt_cost_invalid", errors)
    duration = _number(receipt.get("duration_seconds"), "provider_receipt_duration_invalid", errors)
    attempts = receipt.get("attempt_count")
    if cost > float(execution_request.get("max_cost_usd") or 0):
        errors.append("provider_receipt_budget_exceeded")
    if duration > int(execution_request.get("ttl_seconds") or 0):
        errors.append("provider_receipt_ttl_exceeded")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 1
        or attempts > int(execution_request.get("retry_cap") or 0) + 1
    ):
        errors.append("provider_receipt_attempt_count_invalid")
    outputs = receipt.get("downloaded_outputs")
    if not isinstance(outputs, list):
        errors.append("provider_receipt_outputs_invalid")
        outputs = []
    if status == "succeeded_unqualified" and not outputs:
        errors.append("provider_receipt_success_outputs_missing")
    for output in outputs:
        if not isinstance(output, Mapping):
            errors.append("provider_receipt_output_not_object")
            continue
        _digest(output.get("digest"), "provider_receipt_output_digest_invalid", errors)
        if output.get("download_complete") is not True or output.get("hash_verified") is not True:
            errors.append("provider_receipt_output_not_verified")
    receipt["output_digests"] = [
        {"artifact_id": output.get("artifact_id"), "digest": output.get("digest")}
        for output in outputs
        if isinstance(output, Mapping)
    ]
    if status != "succeeded_unqualified" and not isinstance(receipt.get("failure"), Mapping):
        errors.append("provider_receipt_typed_failure_missing")
    if receipt.get("deletion_status") not in {"pending", "verified_not_retained"}:
        errors.append("provider_receipt_deletion_status_invalid")
    if not isinstance(receipt.get("provider_runtime_identity"), Mapping) or receipt.get(
        "provider_runtime_identity", {}
    ).get("provider_identity") != receipt.get("provider_identity"):
        errors.append("provider_receipt_runtime_identity_invalid")
    for key in ("warnings", "blockers"):
        if not isinstance(receipt.get(key), list):
            errors.append(f"provider_receipt_{key}_invalid")
    _time(receipt.get("timestamp"), "provider_receipt_timestamp_invalid", errors)
    for field in (
        "provider_success_is_blueprint_qualification",
        "metric_scale_proven",
        "collision_geometry_validated",
        "isaac_compatibility_proven",
        "physical_success_proven",
        "deployment_readiness_proven",
    ):
        if receipt.get(field) is not False:
            errors.append(f"provider_receipt_forbidden_claim:{field}")
    if receipt.get("proof_effect") != "provider_output_derived_support_only" or receipt.get(
        "claim_ceiling"
    ) != "external_reconstruction_import":
        errors.append("provider_receipt_claim_boundary_invalid")
    if errors:
        raise ReconstructionProviderContractError(errors)
    return _finalize(
        receipt, PROVIDER_EXECUTION_RECEIPT_SCHEMA, "provider_execution_receipt_digest"
    )


def build_reconstruction_provider_deletion_receipt(
    value: Mapping[str, Any], *, execution_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Record provider deletion evidence without equating it to provider zero."""

    receipt = dict(value)
    errors: list[str] = []
    execution_digest = execution_receipt.get("provider_execution_receipt_digest")
    _digest(execution_digest, "provider_deletion_execution_digest_invalid", errors)
    if (
        execution_receipt.get("schema_version") != PROVIDER_EXECUTION_RECEIPT_SCHEMA
        or execution_digest
        != canonical_digest(
            execution_receipt, digest_field="provider_execution_receipt_digest"
        )
    ):
        errors.append("provider_deletion_execution_receipt_invalid")
    if receipt.get("provider_execution_receipt_digest") != execution_digest:
        errors.append("provider_deletion_execution_binding_mismatch")
    if receipt.get("provider_identity") != execution_receipt.get("provider_identity"):
        errors.append("provider_deletion_identity_mismatch")
    if receipt.get("status") not in {"verified_deleted", "verified_not_retained", "failed"}:
        errors.append("provider_deletion_status_invalid")
    if not isinstance(receipt.get("provider_evidence"), Mapping):
        errors.append("provider_deletion_evidence_missing")
    _time(receipt.get("timestamp"), "provider_deletion_timestamp_invalid", errors)
    if receipt.get("independently_verified") is not True:
        errors.append("provider_deletion_independent_verification_missing")
    if receipt.get("provider_zero_proven") is not False:
        errors.append("provider_deletion_cannot_claim_provider_zero")
    if receipt.get("proof_effect") != "provider_deletion_evidence_only" or receipt.get(
        "claim_ceiling"
    ) != "none":
        errors.append("provider_deletion_claim_boundary_invalid")
    if errors:
        raise ReconstructionProviderContractError(errors)
    return _finalize(
        receipt, PROVIDER_DELETION_RECEIPT_SCHEMA, "provider_deletion_receipt_digest"
    )


__all__ = [
    "PROVIDER_ADMISSION_SCHEMA",
    "PROVIDER_DELETION_RECEIPT_SCHEMA",
    "PROVIDER_EXECUTION_RECEIPT_SCHEMA",
    "PROVIDER_EXECUTION_REQUEST_SCHEMA",
    "ReconstructionProviderContractError",
    "build_reconstruction_provider_admission",
    "build_reconstruction_provider_deletion_receipt",
    "build_reconstruction_provider_execution_receipt",
    "build_reconstruction_provider_execution_request",
    "require_reconstruction_provider_execution_authority",
]
