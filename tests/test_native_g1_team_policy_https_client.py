"""A reviewed team endpoint must return owner-bound G1 actions without redirects."""

import json

import numpy as np
import pytest

from blueprint_pipeline.core.security_controls import BoundedHttpResponse
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.native_g1_team_policy_https_client import NativeG1TeamPolicyHttpsClient


OWNER = {"user_id": "owner-1", "organization_id": "team-1"}
SETUP = "sha256:" + "a" * 64


def _profile() -> dict:
    profile = {
        "schema_version": "team_policy_delivery_profile.v1",
        "owner": OWNER,
        "label": "G1 team policy",
        "source_setup_digest": SETUP,
        "source_scene_id": "scene-841757",
        "source_task_id": "book-task",
        "robot_preset_id": "unitree_g1",
        "embodiment_id": "unitree_g1",
        "observation_schema_id": "g1.semantic_v3",
        "action_schema_id": "g1.semantic_v3",
        "delivery": {
            "mode": "authenticated_endpoint",
            "endpoint_url": "https://policy.example.org/v1/action",
            "auth_secret_ref": "secretref:team/policy",
            "timeout_ms": 5000,
        },
        "status": "registered_for_runtime_review",
        "claim_ceiling": "planning_only",
        "provider_mutation_performed": False,
        "public_redistribution_authorized": False,
    }
    profile["profile_digest"] = cross_runtime_canonical_digest(profile)
    return profile


def _client(profile: dict, fetcher) -> NativeG1TeamPolicyHttpsClient:
    return NativeG1TeamPolicyHttpsClient(
        profile=profile,
        expected_owner=OWNER,
        expected_setup_digest=SETUP,
        expected_interface={
            "robot_preset_id": "unitree_g1",
            "embodiment_id": "unitree_g1",
            "observation_schema_id": "g1.semantic_v3",
            "action_schema_id": "g1.semantic_v3",
        },
        approved_origin="https://policy.example.org",
        resolved_secret_ref="secretref:team/policy",
        credential="private-token",
        fetcher=fetcher,
    )


def test_endpoint_client_exchanges_bound_reset_and_g1_action() -> None:
    calls = []
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]

    def fetcher(url, **options):
        request = json.loads(options["data"])
        calls.append((url, options, request))
        response = {
            "protocol": request["protocol"],
            "profile_digest": request["profile_digest"],
            "request_id": request["request_id"],
        }
        response.update({"ok": True} if request["kind"] == "reset" else {"action_chunk": [action]})
        return BoundedHttpResponse(
            body=json.dumps(response).encode(),
            status=200,
            content_type="application/json",
            final_url=url,
        )

    client = _client(_profile(), fetcher)
    client.reset(seed=5)
    assert client.infer_chunk(
        front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        observation_state=[0.0] * 64,
        task="Place the book.",
    ) == [action]
    assert client.candidate_policy_queried
    assert [row[2]["request_id"] for row in calls] == [0, 1]
    assert calls[0][2]["seed"] == 5
    assert calls[1][2]["observation"]["state"] == [0.0] * 64
    assert all(row[1]["headers"]["Authorization"] == "Bearer private-token" for row in calls)
    assert all(row[1]["allowed_origins"] == ("https://policy.example.org",) for row in calls)
    assert all(row[1]["max_redirects"] == 0 for row in calls)


def test_endpoint_client_rejects_changed_profile_owner_and_origin() -> None:
    profile = _profile()
    profile["owner"] = {"user_id": "someone-else", "organization_id": "team-1"}
    with pytest.raises(ValueError, match="profile_invalid"):
        _client(profile, None)
    profile = _profile()
    profile["delivery"]["endpoint_url"] = "https://other.example.org/v1/action"
    profile["profile_digest"] = cross_runtime_canonical_digest(
        profile, digest_field="profile_digest"
    )
    with pytest.raises(ValueError, match="configuration_invalid"):
        _client(profile, None)
    profile = _profile()
    profile["delivery"]["auth_secret_ref"] = "secretref:another-team/policy"
    profile["profile_digest"] = cross_runtime_canonical_digest(
        profile, digest_field="profile_digest"
    )
    with pytest.raises(ValueError, match="configuration_invalid"):
        _client(profile, None)


def test_endpoint_client_rejects_response_from_wrong_profile() -> None:
    def fetcher(url, **options):
        request = json.loads(options["data"])
        return BoundedHttpResponse(
            body=json.dumps(
                {
                    "protocol": request["protocol"],
                    "profile_digest": "sha256:" + "f" * 64,
                    "request_id": request["request_id"],
                    "ok": True,
                }
            ).encode(),
            status=200,
            content_type="application/json",
            final_url=url,
        )

    with pytest.raises(ValueError, match="response_identity_invalid"):
        _client(_profile(), fetcher).reset(seed=5)


def test_endpoint_client_rejects_changed_transport_target() -> None:
    def fetcher(url, **_options):
        return BoundedHttpResponse(
            body=b"{}",
            status=200,
            content_type="application/json",
            final_url="https://other.example.org/v1/action",
        )

    with pytest.raises(ValueError, match="response_transport_invalid"):
        _client(_profile(), fetcher).reset(seed=5)
