"""Validate bounded authority for private processing of public-scene derivatives."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "public_scene_execution_authority.v1"
CONCURRENT_SCHEMA_VERSION = "public_scene_execution_authority.v2"
REQUIRED_PROVIDERS = {"vast", "nvidia_nim", "nvidia_remote_renderer", "object_store"}
REQUIRED_PURPOSES = {
    "released_code_inpainting",
    "articulation_topology_inference",
    "native_simulator_qualification",
    "two_candidate_policy_evaluation",
}
AURA_RETRY_WORKLOAD = "aurafusion360_interiorgs"


class PublicSceneExecutionAuthorityError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def validate_public_scene_execution_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate human authority without expanding publisher license claims."""

    try:
        authority = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PublicSceneExecutionAuthorityError(["execution_authority_not_json"]) from exc
    if not isinstance(authority, dict):
        raise PublicSceneExecutionAuthorityError(["execution_authority_invalid"])
    errors: list[str] = []
    schema_version = authority.get("schema_version")
    if schema_version not in {SCHEMA_VERSION, CONCURRENT_SCHEMA_VERSION}:
        errors.append("execution_authority_schema_invalid")
    if authority.get("authority_kind") != "explicit_user_direction_in_current_goal":
        errors.append("execution_authority_kind_invalid")
    for key in (
        "authority_reference",
        "authorized_by",
        "authorized_on",
        "publisher_scene_id",
        "target_instance_id",
    ):
        if not isinstance(authority.get(key), str) or not authority[key].strip():
            errors.append(f"execution_authority_{key}_invalid")
    for key in (
        "freeze_digest",
        "prior_rights_authority_digest",
        "interiorgs_terms_text_digest",
        "aura_adapter_receipt_digest",
        "joint_agent_source_receipt_digest",
        "joint_agent_source_asset_digest",
    ):
        if not _digest(authority.get(key)):
            errors.append(f"execution_authority_{key}_invalid")
    if not REQUIRED_PROVIDERS.issubset(set(authority.get("provider_scope") or [])):
        errors.append("execution_authority_provider_scope_incomplete")
    if not REQUIRED_PURPOSES.issubset(set(authority.get("purpose_scope") or [])):
        errors.append("execution_authority_purpose_scope_incomplete")
    spend = authority.get("hard_total_spend_cap_usd")
    if (
        isinstance(spend, bool)
        or not isinstance(spend, (int, float))
        or not math.isfinite(float(spend))
        or not 0 < float(spend) <= 100
    ):
        errors.append("execution_authority_spend_cap_invalid")
    ttl = authority.get("maximum_single_resource_ttl_seconds")
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, int)
        or not 60 <= ttl <= 14_400
    ):
        errors.append("execution_authority_ttl_invalid")
    required_true = (
        "remote_upload_authorized",
        "paid_compute_authorized",
        "derived_aura_adapter_upload_authorized",
        "sage_cc_by_nc_derived_asset_upload_authorized",
        "provider_zero_required_before_and_after",
        "teardown_required",
    )
    for key in required_true:
        if authority.get(key) is not True:
            errors.append(f"execution_authority_{key}_not_true")
    if schema_version == SCHEMA_VERSION:
        if authority.get("one_instance_at_a_time") is not True:
            errors.append("execution_authority_one_instance_at_a_time_not_true")
    else:
        if authority.get("one_instance_at_a_time") is not False:
            errors.append("execution_authority_one_instance_at_a_time_not_false")
        if authority.get("maximum_concurrent_paid_instances") != 2:
            errors.append("execution_authority_maximum_concurrent_paid_instances_invalid")
        if authority.get("concurrent_paid_compute_authorized") is not True:
            errors.append("execution_authority_concurrent_paid_compute_not_true")
        if not isinstance(authority.get("concurrent_authority_reference"), str) or not authority[
            "concurrent_authority_reference"
        ].strip():
            errors.append("execution_authority_concurrent_authority_reference_invalid")
        active_ids = authority.get("known_active_instance_ids_at_authorization")
        if (
            not isinstance(active_ids, list)
            or len(active_ids) != 1
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in active_ids)
        ):
            errors.append("execution_authority_known_active_instance_ids_invalid")
        if authority.get("prelaunch_provider_inventory_rule") != (
            "no_unadmitted_active_or_billable_resources;"
            "explicitly_allowed_active_instance_ids_must_be_read_back"
        ):
            errors.append("execution_authority_prelaunch_provider_inventory_rule_invalid")
    required_false = (
        "raw_interiorgs_downloaded_bytes_upload_authorized",
        "public_disclosure_authorized",
        "model_training_authorized",
        "commercial_use_authorized",
        "automatic_paid_retry_authorized",
    )
    for key in required_false:
        if authority.get(key) is not False:
            errors.append(f"execution_authority_{key}_not_false")
    attempt_authority = authority.get("attempt_authority")
    if attempt_authority is not None:
        if not isinstance(attempt_authority, dict):
            errors.append("execution_authority_attempt_authority_invalid")
        else:
            if attempt_authority.get("workload") != AURA_RETRY_WORKLOAD:
                errors.append("execution_authority_attempt_workload_invalid")
            attempt_ordinal = attempt_authority.get("attempt_ordinal")
            if (
                isinstance(attempt_ordinal, bool)
                or not isinstance(attempt_ordinal, int)
                or attempt_ordinal < 1
            ):
                errors.append("execution_authority_attempt_ordinal_invalid")
            if attempt_authority.get("maximum_paid_attempts") != 1:
                errors.append("execution_authority_maximum_paid_attempts_invalid")
            if attempt_authority.get("maximum_automatic_retries") != 0:
                errors.append("execution_authority_maximum_automatic_retries_invalid")
            if not _digest(attempt_authority.get("previous_execution_receipt_digest")):
                errors.append("execution_authority_previous_execution_receipt_digest_invalid")
            camera_id = attempt_authority.get("required_reference_camera_id")
            if not isinstance(camera_id, str) or not camera_id.strip():
                errors.append("execution_authority_required_reference_camera_id_invalid")
            runtime_index = attempt_authority.get("required_runtime_reference_index")
            if (
                isinstance(runtime_index, bool)
                or not isinstance(runtime_index, int)
                or runtime_index < 0
            ):
                errors.append("execution_authority_required_runtime_reference_index_invalid")
            if attempt_authority.get("unretained_trajectory_generation_allowed") is not False:
                errors.append(
                    "execution_authority_unretained_trajectory_generation_allowed_not_false"
                )
    if authority.get("retention_policy") != "bounded_to_goal_then_provider_zero":
        errors.append("execution_authority_retention_policy_invalid")
    if authority.get("dataset_claim") != (
        "internal_noncommercial_private_processing_authority_only;"
        "does_not_change_publisher_nonredistribution_terms"
    ):
        errors.append("execution_authority_dataset_claim_invalid")
    if authority.get("authorization_digest") != canonical_digest(
        authority, digest_field="authorization_digest"
    ):
        errors.append("execution_authority_digest_invalid")
    if errors:
        raise PublicSceneExecutionAuthorityError(errors)
    return authority


def validate_aura_retry_authority_binding(
    value: Mapping[str, Any],
    *,
    adapter_receipt_digest: str,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
) -> dict[str, Any]:
    """Bind a fresh Aura attempt to its exact prepared adapter and resource limits."""

    authority = validate_public_scene_execution_authority(value)
    errors: list[str] = []
    attempt_authority = authority.get("attempt_authority")
    if not isinstance(attempt_authority, dict):
        errors.append("execution_authority_aura_attempt_authority_missing")
    if authority.get("aura_adapter_receipt_digest") != adapter_receipt_digest:
        errors.append("execution_authority_aura_adapter_receipt_digest_mismatch")
    if float(authority.get("hard_total_spend_cap_usd", -1)) != float(hard_cap_usd):
        errors.append("execution_authority_aura_spend_cap_mismatch")
    if authority.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds:
        errors.append("execution_authority_aura_ttl_mismatch")
    if authority.get("one_instance_at_a_time") is not True:
        errors.append("execution_authority_aura_one_instance_at_a_time_required")
    if authority.get("automatic_paid_retry_authorized") is not False:
        errors.append("execution_authority_aura_automatic_retry_must_be_false")
    if errors:
        raise PublicSceneExecutionAuthorityError(errors)
    return authority


__all__ = [
    "PublicSceneExecutionAuthorityError",
    "CONCURRENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "validate_aura_retry_authority_binding",
    "validate_public_scene_execution_authority",
]
