"""A team's own DROID policy over authenticated HTTPS: actions in, nothing else trusted.

Beta journey step: L5 team policy evaluation.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.droid_https_policy_client import (
    METADATA_SCHEMA,
    DroidHttpsPolicySpec,
    DroidHttpsTeamPolicyClient,
)
from blueprint_pipeline.policy_interface_binding import observation_interface, resolve_policy_interface

ORIGIN = "https://policy.acme-robotics.example"
SPEC = DroidHttpsPolicySpec(policy_id="team-acme.pi05-ft-7", endpoint_url=ORIGIN + "/v1/act",
                            metadata_url=ORIGIN + "/v1/metadata", allowed_origin=ORIGIN,
                            model_identity_digest="sha256:" + "7" * 64, action_chunk_rows=10, open_loop_horizon=8)
OBSERVATION = {"observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
               "observation/wrist_image_left": np.full((224, 224, 3), 9, dtype=np.uint8),
               "observation/joint_position": np.zeros(7), "observation/gripper_position": np.zeros(1),
               "prompt": "open the door"}


def _service(*, metadata: dict[str, Any] | None = None, response: Any = None):
    calls: list[dict[str, Any]] = []

    def transport(*, url: str, method: str, body: bytes | None, headers: dict[str, str]) -> bytes:
        calls.append({"url": url, "method": method, "headers": headers})
        if url.endswith("/metadata"):
            value = metadata or {**SPEC.expected_metadata()}
        else:
            value = response if response is not None else {"actions": np.zeros((10, 8)).tolist()}
        return json.dumps(value).encode() if not isinstance(value, bytes) else value

    return calls, transport


def test_interface_registry_keeps_the_frozen_pair_on_its_pinned_servers() -> None:
    assert resolve_policy_interface({"candidate_id": "pi05_droid"}) == "openpi_droid.v1"
    assert resolve_policy_interface({"candidate_id": "groot_n17_droid"}) == "groot_n17_droid.v1"
    with pytest.raises(ValueError, match="frozen_candidate_mismatch"):
        resolve_policy_interface({"candidate_id": "pi05_droid", "policy_interface_id": "openpi_droid_https.v1"})
    team = {"candidate_id": "team-acme.pi05-ft-7", "policy_interface_id": "openpi_droid_https.v1"}
    assert observation_interface(team) == "pi05_droid"


def test_a_query_returns_only_actions_and_retains_evidence_without_the_credential() -> None:
    calls, transport = _service()
    client = DroidHttpsTeamPolicyClient(spec=SPEC, bearer_token="team-secret-token", transport=transport)
    readiness = client.preflight_readiness()
    assert readiness["identity_verified"] is True and readiness["candidate_policy_queried"] is False
    assert readiness["server_metadata"]["identity_basis"] == "team_declared_endpoint_metadata"
    sunk: list[Any] = []
    client.bind_request_evidence_sink(sunk.append)
    actions = client.infer(OBSERVATION)
    assert np.asarray(actions).shape == (10, 8) and client.candidate_policy_queried is True
    evidence = client.last_inference_evidence()
    assert evidence["action_payload_returned"] is True and evidence["actions_extracted"] is True
    assert len(sunk) == 1 and sunk[0]["transport"] == "droid_https_json_bearer"
    assert all(call["headers"]["Authorization"] == "Bearer team-secret-token" for call in calls)
    assert "team-secret-token" not in json.dumps([evidence, sunk[0], client.evidence_summary()])


@pytest.mark.parametrize("response, error", [
    ({"actions": [[0.0] * 8] * 10, "task_success": True}, "envelope_invalid"),
    ({"result": "done"}, "envelope_invalid"),
    (b'{"actions": [[NaN]]}', "nonfinite"),
])
def test_anything_but_an_actions_envelope_is_refused_after_the_query(response: Any, error: str) -> None:
    _calls, transport = _service(response=response)
    client = DroidHttpsTeamPolicyClient(spec=SPEC, bearer_token="token", transport=transport)
    with pytest.raises(ValueError, match=error):
        client.infer(OBSERVATION)


def test_declared_identity_drift_missing_credential_and_foreign_origins_fail_closed() -> None:
    _calls, transport = _service(metadata={**SPEC.expected_metadata(), "action_chunk_rows": 15})
    with pytest.raises(ValueError, match="identity_mismatch"):
        DroidHttpsTeamPolicyClient(spec=SPEC, bearer_token="token", transport=transport).preflight_readiness()
    with pytest.raises(ValueError, match="credential_unavailable"):
        DroidHttpsTeamPolicyClient(spec=SPEC, bearer_token=None, transport=transport)
    for change in ({"metadata_url": "https://other.example/v1/metadata"},
                   {"endpoint_url": "http://policy.acme-robotics.example/v1/act"},
                   {"action_chunk_rows": 50}, {"model_identity_digest": "unpinned"}):
        with pytest.raises(ValueError):
            DroidHttpsPolicySpec(**{**SPEC.__dict__, **change}).validate()
    assert SPEC.expected_metadata()["schema_version"] == METADATA_SCHEMA
