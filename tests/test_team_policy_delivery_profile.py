"""The same owner/task/interface binding applies to every delivery mode."""

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.team_policy_delivery_profile import validate_team_policy_delivery_profile


OWNER = {"user_id": "owner-1", "organization_id": "team-1"}


def _setup() -> dict:
    setup = {
        "scene_id": "scene-841757",
        "task_id": "book-task",
        "robot_presets": [
            {
                "robot_preset_id": "unitree_g1_dex3_sonic_v1",
                "embodiment_id": "unitree_g1_dex3_v1",
                "observation_schema": {"schema_id": "humanoidarena_head_rgb_state64_v1"},
                "action_schema": {"schema_id": "humanoidarena_semantic_v3"},
            }
        ],
    }
    setup["setup_digest"] = cross_runtime_canonical_digest(setup)
    return setup


def _profile(setup: dict, delivery: dict) -> dict:
    profile = {
        "schema_version": "team_policy_delivery_profile.v1",
        "owner": OWNER,
        "label": "Team policy v1",
        "source_setup_digest": setup["setup_digest"],
        "source_scene_id": setup["scene_id"],
        "source_task_id": setup["task_id"],
        "robot_preset_id": "unitree_g1_dex3_sonic_v1",
        "embodiment_id": "unitree_g1_dex3_v1",
        "observation_schema_id": "humanoidarena_head_rgb_state64_v1",
        "action_schema_id": "humanoidarena_semantic_v3",
        "delivery": delivery,
        "status": "registered_for_runtime_review",
        "claim_ceiling": "planning_only",
        "provider_mutation_performed": False,
        "public_redistribution_authorized": False,
    }
    profile["profile_digest"] = cross_runtime_canonical_digest(profile)
    return profile


@pytest.mark.parametrize(
    "delivery",
    [
        {
            "mode": "authenticated_endpoint",
            "endpoint_url": "https://policy.example.org/v1/action",
            "auth_secret_ref": "secretref:team/policy",
            "timeout_ms": 5000,
        },
        {
            "mode": "container",
            "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
            "protocol": "jsonl_observation_action_v1",
        },
        {
            "mode": "noncontainer_artifact",
            "artifact_uri": "https://files.example.org/policy.tar.gz",
            "artifact_sha256": "sha256:" + "b" * 64,
            "entrypoint": "policy/run.py",
            "protocol": "jsonl_observation_action_v1",
        },
    ],
)
def test_three_delivery_modes_bind_same_owner_and_robot(delivery: dict) -> None:
    setup = _setup()
    profile = _profile(setup, delivery)
    assert (
        validate_team_policy_delivery_profile(
            profile,
            trusted_setup=setup,
            authenticated_owner=OWNER,
        )
        == profile
    )


def test_delivery_profile_rejects_owner_setup_and_interface_tampering() -> None:
    setup = _setup()
    profile = _profile(
        setup,
        {
            "mode": "container",
            "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
            "protocol": "jsonl_observation_action_v1",
        },
    )
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_team_policy_delivery_profile(
            profile,
            trusted_setup=setup,
            authenticated_owner={"user_id": "another-owner", "organization_id": "team-1"},
        )
    changed = {**profile, "action_schema_id": "another-action-schema"}
    changed["profile_digest"] = cross_runtime_canonical_digest(
        changed, digest_field="profile_digest"
    )
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_team_policy_delivery_profile(
            changed, trusted_setup=setup, authenticated_owner=OWNER
        )
    setup["scene_id"] = "another-scene"
    with pytest.raises(ValueError, match="setup_invalid"):
        validate_team_policy_delivery_profile(
            profile, trusted_setup=setup, authenticated_owner=OWNER
        )


@pytest.mark.parametrize(
    "delivery",
    [
        {
            "mode": "authenticated_endpoint",
            "endpoint_url": "https://127.0.0.1/v1/action",
            "auth_secret_ref": "secretref:team/policy",
            "timeout_ms": 5000,
        },
        {
            "mode": "container",
            "image_ref": "registry.example.org/team/g1:latest",
            "protocol": "jsonl_observation_action_v1",
        },
        {
            "mode": "noncontainer_artifact",
            "artifact_uri": "https://files.example.org/policy.tar.gz",
            "artifact_sha256": "sha256:" + "b" * 64,
            "entrypoint": "../run.py",
            "protocol": "jsonl_observation_action_v1",
        },
    ],
)
def test_delivery_profile_rejects_untrusted_delivery_identity(delivery: dict) -> None:
    setup = _setup()
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_team_policy_delivery_profile(
            _profile(setup, delivery),
            trusted_setup=setup,
            authenticated_owner=OWNER,
        )
