"""A team can select a reviewed-interface G1 profile without starting spend."""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.native_g1_team_policy_run_request import (
    SCHEMA,
    validate_g1_team_policy_run_request,
)
from tests.test_native_g1_team_scored_scene_episode import _inputs
from tests.test_team_policy_delivery_profile import OWNER, _profile


NOW = 1_800_000_000.0


def _request(setup, profile):
    value = {
        "schema_version": SCHEMA,
        "run_id": "team-g1-book-1",
        "owner": OWNER,
        "scene_id": setup["scene_id"],
        "task_id": setup["task_id"],
        "source_packet_receipt_digest": setup["source_packet_receipt_digest"],
        "source_setup_digest": setup["setup_digest"],
        "robot_preset_id": profile["robot_preset_id"],
        "objective_id": "task_success",
        "policy_profile": profile,
        "authorization": {
            "maximum_cost_usd": 12,
            "hard_ttl_seconds": 14400,
            "expires_at_epoch": NOW + 3600,
            "retry_cap": 0,
        },
        "site_observation_exchange_authorized": True,
        "claim_ceiling": "development_only",
        "public_redistribution_authorized": False,
    }
    return {**value, "request_digest": cross_runtime_canonical_digest(value)}


def test_team_policy_request_binds_profile_to_owner_and_task(
    tmp_path: Path, monkeypatch
) -> None:
    setup, _, _ = _inputs(tmp_path, monkeypatch)
    profile = _profile(setup, {
        "mode": "container",
        "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
        "protocol": "jsonl_observation_action_v1",
    })
    request = _request(setup, profile)
    assert validate_g1_team_policy_run_request(
        request, trusted_setup=setup, authenticated_owner=OWNER, now_epoch=NOW
    ) == request


def test_team_policy_request_rejects_profile_authority_or_cost_drift(
    tmp_path: Path, monkeypatch
) -> None:
    setup, _, _ = _inputs(tmp_path, monkeypatch)
    profile = _profile(setup, {
        "mode": "noncontainer_artifact",
        "artifact_uri": "https://files.example.org/policy.tar.gz",
        "artifact_sha256": "sha256:" + "b" * 64,
        "entrypoint": "run/policy",
        "protocol": "jsonl_observation_action_v1",
    })
    original = _request(setup, profile)
    for change in (
        {"owner": {"user_id": "different", "organization_id": OWNER["organization_id"]}},
        {"source_setup_digest": "sha256:" + "f" * 64},
        {"site_observation_exchange_authorized": False},
        {"public_redistribution_authorized": True},
    ):
        request = {**original, **change}
        request["request_digest"] = cross_runtime_canonical_digest(
            request, digest_field="request_digest"
        )
        with pytest.raises(ValueError, match="request_binding_invalid"):
            validate_g1_team_policy_run_request(
                request, trusted_setup=setup, authenticated_owner=OWNER, now_epoch=NOW
            )
    for cap in (12.01, 1.001, -1):
        request = {**original, "authorization": {**original["authorization"], "maximum_cost_usd": cap}}
        request["request_digest"] = cross_runtime_canonical_digest(
            request, digest_field="request_digest"
        )
        with pytest.raises(ValueError, match="authorization_invalid"):
            validate_g1_team_policy_run_request(
                request, trusted_setup=setup, authenticated_owner=OWNER, now_epoch=NOW
            )
