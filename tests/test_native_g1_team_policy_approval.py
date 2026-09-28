"""Only an exact operator approval may promote a registered delivery to runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_team_policy_approval import (
    SCHEMA,
    validate_g1_team_policy_approval,
)
from tests.test_native_g1_team_scored_scene_episode import _inputs
from tests.test_team_policy_delivery_profile import OWNER, _profile


NOW = 1_800_000_000.0


def _approval(setup, profile, binding, *, objective="task_success"):
    value = {
        "schema_version": SCHEMA,
        "owner": OWNER,
        "profile_digest": profile["profile_digest"],
        "source_setup_digest": setup["setup_digest"],
        "source_packet_receipt_digest": setup["source_packet_receipt_digest"],
        "objective_ids": [objective],
        "runtime_binding": binding,
        "site_observation_exchange_authorized": True,
        "model_execution_rights_reviewed": True,
        "operator_reviewer": "Nijel Hunt",
        "expires_at_epoch": NOW + 3600,
        "claim_ceiling": "development_only",
        "public_redistribution_authorized": False,
    }
    return {**value, "approval_digest": canonical_digest(value)}


@pytest.mark.parametrize("delivery,binding", [
    (
        {
            "mode": "authenticated_endpoint",
            "endpoint_url": "https://policy.example.org/v1/action",
            "auth_secret_ref": "secretref:team/policy",
            "timeout_ms": 5000,
        },
        {
            "mode": "authenticated_endpoint",
            "approved_origin": "https://policy.example.org",
            "resolved_secret_ref": "secretref:team/policy",
        },
    ),
    (
        {
            "mode": "container",
            "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
            "protocol": "jsonl_observation_action_v1",
        },
        {
            "mode": "container",
            "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
            "gpu_device": 0,
        },
    ),
    (
        {
            "mode": "noncontainer_artifact",
            "artifact_uri": "https://files.example.org/policy.tar.gz",
            "artifact_sha256": "sha256:" + "b" * 64,
            "entrypoint": "run/policy",
            "protocol": "jsonl_observation_action_v1",
        },
        {
            "mode": "noncontainer_artifact",
            "artifact_sha256": "sha256:" + "b" * 64,
            "staged_artifact_path": "/workspace/team-artifacts/policy.tar.gz",
        },
    ),
])
def test_approval_binds_each_delivery_to_profile_and_packet(
    tmp_path: Path, monkeypatch, delivery, binding
) -> None:
    setup, _, _ = _inputs(tmp_path, monkeypatch)
    profile = _profile(setup, delivery)
    approval = _approval(setup, profile, {**binding, "profile_digest": profile["profile_digest"]})
    assert validate_g1_team_policy_approval(
        approval, profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
        objective_id="task_success", now_epoch=NOW,
    ) == approval


def test_approval_rejects_changed_rights_objective_and_runtime_binding(
    tmp_path: Path, monkeypatch
) -> None:
    setup, _, _ = _inputs(tmp_path, monkeypatch)
    profile = _profile(setup, {
        "mode": "authenticated_endpoint",
        "endpoint_url": "https://policy.example.org/v1/action",
        "auth_secret_ref": "secretref:team/policy", "timeout_ms": 5000,
    })
    binding = {
        "mode": "authenticated_endpoint", "profile_digest": profile["profile_digest"],
        "approved_origin": "https://policy.example.org",
        "resolved_secret_ref": "secretref:team/policy",
    }
    for change in (
        {"site_observation_exchange_authorized": False},
        {"model_execution_rights_reviewed": False},
        {"objective_ids": ["g1_navigation_goal"]},
        {"runtime_binding": {**binding, "approved_origin": "https://other.example.org"}},
    ):
        approval = _approval(setup, profile, binding)
        approval.update(change)
        approval["approval_digest"] = canonical_digest(approval, digest_field="approval_digest")
        with pytest.raises(ValueError, match="approval_binding_invalid"):
            validate_g1_team_policy_approval(
                approval, profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
                objective_id="task_success", now_epoch=NOW,
            )
    approval = _approval(setup, profile, binding)
    with pytest.raises(ValueError, match="approval_binding_invalid"):
        validate_g1_team_policy_approval(
            approval, profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
            objective_id="task_success", now_epoch=NOW + 3600,
        )
