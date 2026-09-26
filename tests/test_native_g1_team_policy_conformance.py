"""Synthetic G1 checks must not be confused with a scored site episode."""

import json

import numpy as np
import pytest

from blueprint_pipeline.core.security_controls import BoundedHttpResponse
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_team_policy_conformance import (
    TASK,
    run_g1_team_endpoint_synthetic_conformance,
    run_g1_team_policy_synthetic_conformance,
    run_g1_team_process_synthetic_conformance,
)
from tests.test_native_g1_team_policy_jsonl_client import _close, _process
from tests.test_team_policy_delivery_profile import OWNER, _profile, _setup


def _g1_profile():
    setup = _setup()
    return setup, _profile(
        setup,
        {
            "mode": "container",
            "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
            "protocol": "jsonl_observation_action_v1",
        },
    )


def test_synthetic_conformance_calls_real_client_with_no_site_data():
    setup, profile = _g1_profile()
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]

    class Client:
        profile_digest = profile["profile_digest"]
        reset_seed = None

        def reset(self, *, seed):
            self.reset_seed = seed

        def infer_chunk(self, *, front_rgb, observation_state, task):
            assert front_rgb.dtype == np.uint8
            assert front_rgb.shape == (480, 640, 3)
            assert not front_rgb.any()
            assert observation_state == [0.0] * 64
            assert task == TASK
            return [action]

    client = Client()
    receipt = run_g1_team_policy_synthetic_conformance(
        profile=profile,
        trusted_setup=setup,
        authenticated_owner=OWNER,
        policy_client=client,
    )
    assert client.reset_seed == 0
    assert receipt["status"] == "synthetic_wire_compatible"
    assert receipt["synthetic_policy_query_count"] == 1
    assert receipt["site_policy_query_count"] == 0
    assert receipt["runtime_identity_verified"] is False
    assert receipt["paid_launch_authorized"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


def test_synthetic_conformance_rejects_wrong_client_and_action():
    setup, profile = _g1_profile()

    class WrongClient:
        profile_digest = "sha256:" + "f" * 64

        def reset(self, *, seed):
            raise AssertionError("mismatched client received reset")

        def infer_chunk(self, **_kwargs):
            raise AssertionError("mismatched client received observation")

    with pytest.raises(ValueError, match="client_binding_invalid"):
        run_g1_team_policy_synthetic_conformance(
            profile=profile,
            trusted_setup=setup,
            authenticated_owner=OWNER,
            policy_client=WrongClient(),
        )

    class InvalidActionClient:
        def reset(self, *, seed):
            assert seed == 0

        def infer_chunk(self, **_kwargs):
            return [[0.0] * 39]

    with pytest.raises(ValueError, match="semantic_action"):
        run_g1_team_policy_synthetic_conformance(
            profile=profile,
            trusted_setup=setup,
            authenticated_owner=OWNER,
            policy_client=InvalidActionClient(),
        )


def test_registered_https_endpoint_can_be_probed_without_site_data_or_secret_receipt():
    setup = _setup()
    profile = _profile(
        setup,
        {
            "mode": "authenticated_endpoint",
            "endpoint_url": "https://policy.example.org/v1/action",
            "auth_secret_ref": "secretref:team/policy",
            "timeout_ms": 5000,
        },
    )
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    calls = []

    def fetcher(url, **options):
        request = json.loads(options["data"])
        calls.append(request)
        assert options["headers"]["Authorization"] == "Bearer private-token"
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

    receipt = run_g1_team_endpoint_synthetic_conformance(
        profile=profile,
        trusted_setup=setup,
        authenticated_owner=OWNER,
        approved_origin="https://policy.example.org",
        resolved_secret_ref="secretref:team/policy",
        credential="private-token",
        fetcher=fetcher,
    )
    assert [request["kind"] for request in calls] == ["reset", "infer"]
    assert receipt["delivery_mode"] == "authenticated_endpoint"
    assert receipt["site_observation_sent"] is False
    assert "private-token" not in str(receipt)


@pytest.mark.parametrize("mode", ["container", "noncontainer_artifact"])
def test_registered_process_wire_can_be_probed_in_a_child_process(mode):
    setup, profile = _g1_profile()
    if mode == "noncontainer_artifact":
        profile = _profile(
            setup,
            {
                "mode": mode,
                "artifact_uri": "https://files.example.org/policy.tar.gz",
                "artifact_sha256": "sha256:" + "b" * 64,
                "entrypoint": "policy/run.py",
                "protocol": "jsonl_observation_action_v1",
            },
        )
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    process = _process("{'ok': True, 'action_chunk': " + repr([action]) + "}")
    try:
        receipt = run_g1_team_process_synthetic_conformance(
            profile=profile,
            trusted_setup=setup,
            authenticated_owner=OWNER,
            process=process,
            timeout_seconds=2,
        )
    finally:
        _close(process)
    assert receipt["delivery_mode"] == mode
    assert receipt["synthetic_policy_query_count"] == 1
    assert receipt["runtime_identity_verified"] is False
    assert receipt["paid_launch_authorized"] is False
