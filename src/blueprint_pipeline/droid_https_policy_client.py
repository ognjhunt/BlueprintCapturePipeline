"""A robot team's own DROID policy, reached over authenticated HTTPS.

Beta journey step: L5 team policy evaluation. The Quick-10 policy canary drives
policy clients through one duck-typed seam (``infer`` plus evidence methods).
The frozen pi0.5-DROID candidate is served by Blueprint's pinned OpenPI server;
this client serves a team's own policy on the same DROID Franka (Robotiq 2F-85)
contract (interface ``openpi_droid_https.v1``): the identical two 224x224
views, joint and gripper state and prompt go out, and a joint-position action
chunk comes back. The simulator executes and scores every action; nothing the
team's service returns besides actions is read as an outcome.

Transport reuses the proven bounded HTTPS boundary: one exact allowlisted
origin, no redirects, JSON only, bounded bytes and time, and a bearer token
read from the environment variable the sealed execution spec names (the
credential is never retained in evidence). Images travel as lossless PNG.
Identity is the team's declared metadata (policy id, interface, chunk shape,
model identity digest), checked on every readiness handshake; Blueprint cannot
see the team's weights, so the receipt says the identity is team-declared.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

# The canary bundle ships the ``core`` subpackage beside this module.
from .core.security_controls import fetch_bounded_https

INTERFACE_ID = "openpi_droid_https.v1"
REQUEST_SCHEMA = "blueprint.droid_https_policy_request.v1"
METADATA_SCHEMA = "blueprint.droid_https_policy_metadata.v1"
TRANSPORT = "droid_https_json_bearer"
RESPONSE_KEYS = frozenset({"actions", "policy_timing", "server_timing"})
SUPPORTED_ACTION_CHUNK_ROWS = frozenset({10, 15})
MAX_RESPONSE_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30
IMAGE_KEYS = ("observation/exterior_image_1_left", "observation/wrist_image_left")
STATE_KEYS = ("observation/joint_position", "observation/gripper_position")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

Transport = Callable[..., bytes]


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True)
class DroidHttpsPolicySpec:
    policy_id: str
    endpoint_url: str
    metadata_url: str
    allowed_origin: str
    model_identity_digest: str
    action_chunk_rows: int
    open_loop_horizon: int
    action_space: str = "joint_position"
    policy_interface_id: str = INTERFACE_ID

    def validate(self) -> None:
        for url in (self.endpoint_url, self.metadata_url):
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                    or parsed.fragment or _origin(url) != self.allowed_origin):
                raise ValueError("droid_https_policy_endpoint_not_approved")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.policy_id):
            raise ValueError("droid_https_policy_id_invalid")
        if self.policy_interface_id != INTERFACE_ID or self.action_space != "joint_position":
            raise ValueError("droid_https_policy_interface_mismatch")
        if not _DIGEST.fullmatch(self.model_identity_digest):
            raise ValueError("droid_https_policy_model_identity_invalid")
        if (self.action_chunk_rows not in SUPPORTED_ACTION_CHUNK_ROWS
                or not 1 <= self.open_loop_horizon <= self.action_chunk_rows):
            raise ValueError("droid_https_policy_chunk_contract_invalid")

    def expected_metadata(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": METADATA_SCHEMA, "policy_id": self.policy_id,
                "policy_interface_id": self.policy_interface_id, "action_space": self.action_space,
                "action_chunk_rows": self.action_chunk_rows, "model_identity_digest": self.model_identity_digest}


def encode_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """The exact DROID observation as JSON: lossless PNG views, finite state, prompt."""

    import numpy as np
    from PIL import Image

    if set(observation) != {*IMAGE_KEYS, *STATE_KEYS, "prompt"}:
        raise ValueError("droid_https_policy_observation_keys_invalid")
    wire: dict[str, Any] = {}
    for key in IMAGE_KEYS:
        image = np.asarray(observation[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("droid_https_policy_observation_image_invalid")
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        wire[key] = {"encoding": "png_base64", "height": int(image.shape[0]), "width": int(image.shape[1]),
                     "data": base64.b64encode(buffer.getvalue()).decode("ascii")}
    for key in STATE_KEYS:
        values = [float(value) for value in np.asarray(observation[key], dtype=float).reshape(-1)]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("droid_https_policy_observation_state_invalid")
        wire[key] = values
    if not isinstance(observation["prompt"], str) or not observation["prompt"].strip():
        raise ValueError("droid_https_policy_observation_prompt_invalid")
    wire["prompt"] = observation["prompt"]
    return wire


class DroidHttpsTeamPolicyClient:
    """Team-hosted DROID policy with per-operation declared-identity checks."""

    learned_policy = True
    # A subclass serving another wire contract over the same transport overrides these.
    interface_id = INTERFACE_ID
    request_schema = REQUEST_SCHEMA
    transport_name = TRANSPORT

    def __init__(self, *, spec: DroidHttpsPolicySpec, bearer_token: str | None,
                 transport: Transport | None = None) -> None:
        spec.validate()
        if not bearer_token or any(character in bearer_token for character in "\r\n"):
            raise ValueError("droid_https_policy_credential_unavailable")
        self._spec = spec
        self._token = bearer_token
        self._transport = transport or self._bounded_https
        self.policy_id = spec.policy_id
        self.action_space = spec.action_space
        self.action_chunk_rows = spec.action_chunk_rows
        self.open_loop_horizon = spec.open_loop_horizon
        self.candidate_policy_queried = False
        self._server_metadata: dict[str, Any] | None = None
        self._last_inference_evidence: dict[str, Any] | None = None
        self._request_evidence: dict[str, Any] | None = None
        self._request_evidence_sink: Callable[[Mapping[str, Any]], None] | None = None
        self._generation = 0

    def _bounded_https(self, *, url: str, method: str, body: bytes | None, headers: Mapping[str, str]) -> bytes:
        return fetch_bounded_https(url, method=method, data=body, headers=dict(headers),
                                   timeout_seconds=REQUEST_TIMEOUT_SECONDS, max_bytes=MAX_RESPONSE_BYTES,
                                   allowed_origins=(self._spec.allowed_origin,),
                                   allowed_content_types=("application/json",), max_redirects=0).body

    def _call(self, url: str, method: str, body: bytes | None) -> Any:
        headers = {"Accept": "application/json", "Authorization": "Bearer " + self._token}
        if body is not None:
            headers["Content-Type"] = "application/json"
        raw = self._transport(url=url, method=method, body=body, headers=headers)
        if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("droid_https_policy_response_size_invalid")

        def reject(_value: str) -> None:
            raise ValueError("droid_https_policy_response_nonfinite")

        return json.loads(raw, parse_constant=reject)

    def _verified_metadata(self) -> dict[str, Any]:
        observed = self._call(self._spec.metadata_url, "GET", None)
        expected = self._spec.expected_metadata()
        if not isinstance(observed, Mapping) or dict(observed) != expected:
            raise ValueError("droid_https_policy_identity_mismatch")
        self._generation += 1
        self._server_metadata = {**expected, "identity_sha256": _canonical(expected),
                                 "identity_basis": "team_declared_endpoint_metadata"}
        return self._server_metadata

    @property
    def server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata) if self._server_metadata is not None else self._verified_metadata()

    def bind_request_evidence_sink(self, sink: Callable[[Mapping[str, Any]], None] | None) -> None:
        self._request_evidence_sink = sink

    def last_request_evidence(self) -> dict[str, Any]:
        if self._request_evidence is None:
            raise ValueError("droid_https_policy_request_evidence_missing")
        return json.loads(json.dumps(self._request_evidence))

    def _encode(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return encode_observation(observation)

    def infer(self, observation: Mapping[str, Any]) -> Any:
        """Send one observation, retain truthful evidence, return only the action chunk."""

        try:
            from policy_request_evidence import capture_request
        except ModuleNotFoundError:
            from .policy_request_evidence import capture_request
        self._last_inference_evidence = None
        self._request_evidence = None
        metadata = self._verified_metadata()
        body = json.dumps({"schema_version": self.request_schema, "request_id": secrets.token_hex(24),
                           "policy_interface_id": self.interface_id, "observation": self._encode(observation)},
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self._request_evidence = capture_request(observation, transport=self.transport_name)
        if self._request_evidence_sink is not None:
            self._request_evidence_sink(self._request_evidence)
        response = self._call(self._spec.endpoint_url, "POST", body)
        retained = json.loads(json.dumps(response, allow_nan=False))
        keys = sorted(str(key) for key in response) if isinstance(response, Mapping) else []
        self._last_inference_evidence = {
            "server_response_received": True, "transport": self.transport_name,
            "transport_connection_generation": self._generation,
            "server_identity_sha256": metadata["identity_sha256"], "identity_basis": metadata["identity_basis"],
            "request_body_sha256": _sha(body), "wire_response_type": type(response).__name__,
            "wire_response_keys": keys, "raw_vendor_action_response": retained,
            "raw_vendor_action_response_digest": _canonical({"raw_vendor_action_response": retained}),
            "raw_vendor_action_response_role": "genuine_decoded_team_https_response_before_normalization",
            "action_payload_returned": isinstance(response, Mapping) and "actions" in response,
            "actions_extracted": False,
        }
        # A response from the team's service is a completed query even when the
        # envelope is then refused; only the simulator decides the outcome.
        self.candidate_policy_queried = True
        if not isinstance(response, Mapping) or not set(response) <= RESPONSE_KEYS or "actions" not in response:
            raise ValueError("droid_https_policy_response_envelope_invalid")
        actions = response["actions"]
        self._last_inference_evidence.update(actions_extracted=True)
        return actions

    def last_inference_evidence(self) -> dict[str, Any]:
        if self._last_inference_evidence is None:
            raise ValueError("droid_https_policy_inference_evidence_missing")
        return json.loads(json.dumps(self._last_inference_evidence, allow_nan=False))

    def preflight_readiness(self) -> dict[str, Any]:
        """Re-check the declared identity for the next episode without inference."""

        prior = bool(self.candidate_policy_queried or self._last_inference_evidence is not None)
        metadata = self._verified_metadata()
        self.candidate_policy_queried = False
        self._last_inference_evidence = None
        self._request_evidence = None
        return {"identity_verified": True, "transport": self.transport_name,
                "readiness_method": "declared_identity_handshake_without_inference",
                "connection_generation": self._generation, "candidate_policy_queried": False,
                "candidate_inference_performed": False, "policy_state_advanced": False,
                "last_inference_evidence": None, "prior_candidate_policy_query_observed": prior,
                "server_metadata": metadata}

    def close(self) -> None:
        return None

    def evidence_summary(self) -> dict[str, Any]:
        return {"transport": self.transport_name, "identity_verified": True, "server_metadata": self.server_metadata,
                "connection_generation": self._generation,
                "last_inference_evidence": (self.last_inference_evidence()
                                            if self._last_inference_evidence is not None else None)}


__all__ = ["INTERFACE_ID", "DroidHttpsPolicySpec", "DroidHttpsTeamPolicyClient", "encode_observation"]
