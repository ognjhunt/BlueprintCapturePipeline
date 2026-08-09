from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_execution_authority import (
    PublicSceneExecutionAuthorityError,
    validate_public_scene_execution_authority,
)

def _authority() -> dict:
    value = {
        "schema_version": "public_scene_execution_authority.v1",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "goal-user-message-2026-08-08",
        "authorized_by": "fixture-owner",
        "authorized_on": "2026-08-08",
        "publisher_scene_id": "840796",
        "target_instance_id": "123",
        "freeze_digest": "sha256:" + "0" * 64,
        "prior_rights_authority_digest": "sha256:" + "1" * 64,
        "interiorgs_terms_text_digest": "sha256:" + "2" * 64,
        "aura_adapter_receipt_digest": "sha256:" + "3" * 64,
        "joint_agent_source_receipt_digest": "sha256:" + "4" * 64,
        "joint_agent_source_asset_digest": "sha256:" + "5" * 64,
        "provider_scope": [
            "vast",
            "nvidia_nim",
            "nvidia_remote_renderer",
            "object_store",
        ],
        "purpose_scope": [
            "released_code_inpainting",
            "articulation_topology_inference",
            "native_simulator_qualification",
            "two_candidate_policy_evaluation",
        ],
        "hard_total_spend_cap_usd": 12.0,
        "maximum_single_resource_ttl_seconds": 14400,
        "remote_upload_authorized": True,
        "paid_compute_authorized": True,
        "derived_aura_adapter_upload_authorized": True,
        "sage_cc_by_nc_derived_asset_upload_authorized": True,
        "one_instance_at_a_time": True,
        "provider_zero_required_before_and_after": True,
        "teardown_required": True,
        "raw_interiorgs_downloaded_bytes_upload_authorized": False,
        "public_disclosure_authorized": False,
        "model_training_authorized": False,
        "commercial_use_authorized": False,
        "automatic_paid_retry_authorized": False,
        "retention_policy": "bounded_to_goal_then_provider_zero",
        "dataset_claim": (
            "internal_noncommercial_private_processing_authority_only;"
            "does_not_change_publisher_nonredistribution_terms"
        ),
        "authorization_digest": "",
    }
    value["authorization_digest"] = canonical_digest(
        value, digest_field="authorization_digest"
    )
    return value


def test_execution_authority_binds_private_processing_and_twelve_dollar_cap() -> None:
    authority = _authority()

    validated = validate_public_scene_execution_authority(authority)

    assert validated["hard_total_spend_cap_usd"] == 12.0
    assert validated["raw_interiorgs_downloaded_bytes_upload_authorized"] is False
    assert validated["derived_aura_adapter_upload_authorized"] is True
    assert validated["automatic_paid_retry_authorized"] is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("hard_total_spend_cap_usd", 0, "spend_cap_invalid"),
        ("one_instance_at_a_time", False, "one_instance_at_a_time_not_true"),
        ("model_training_authorized", True, "model_training_authorized_not_false"),
        (
            "raw_interiorgs_downloaded_bytes_upload_authorized",
            True,
            "raw_interiorgs_downloaded_bytes_upload_authorized_not_false",
        ),
    ],
)
def test_execution_authority_fails_closed_on_scope_expansion(
    field: str, value: object, error: str
) -> None:
    authority = copy.deepcopy(_authority())
    authority[field] = value
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    with pytest.raises(PublicSceneExecutionAuthorityError, match=error):
        validate_public_scene_execution_authority(authority)
