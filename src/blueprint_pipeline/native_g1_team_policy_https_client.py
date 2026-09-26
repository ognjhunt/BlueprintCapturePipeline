"""Owner-bound HTTPS transport for a reviewed G1 team policy endpoint.

The caller supplies a credential from a protected secret resolver and an
operator-approved origin. Neither is persisted in the profile or a receipt.
The bounded fetcher pins public DNS resolution and forbids redirects.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from .core.security_controls import (
    BoundedHttpResponse,
    fetch_bounded_https,
    json_shape_within_limits,
)
from .decision_evidence_contracts import cross_runtime_canonical_digest
from .native_g1_humanoidarena_policy_client import (
    build_semantic_v3_infer_request,
    validate_semantic_v3_infer_response,
)
from .native_g1_team_policy_jsonl_client import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, PROTOCOL


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_REF = re.compile(r"^secretref:[A-Za-z0-9][A-Za-z0-9._:/-]{0,190}$")


class NativeG1TeamPolicyHttpsClient:
    """Exchange one bound G1 observation/action at a time over HTTPS."""

    def __init__(
        self,
        *,
        profile: Mapping[str, Any],
        expected_owner: Mapping[str, str],
        expected_setup_digest: str,
        expected_interface: Mapping[str, str],
        approved_origin: str,
        resolved_secret_ref: str,
        credential: str,
        fetcher: Callable[..., BoundedHttpResponse] = fetch_bounded_https,
    ) -> None:
        delivery = profile.get("delivery")
        profile_digest = profile.get("profile_digest")
        if (
            profile.get("schema_version") != "team_policy_delivery_profile.v1"
            or profile.get("status") != "registered_for_runtime_review"
            or profile.get("claim_ceiling") != "planning_only"
            or profile.get("provider_mutation_performed") is not False
            or profile.get("public_redistribution_authorized") is not False
            or profile.get("owner") != dict(expected_owner)
            or profile.get("source_setup_digest") != expected_setup_digest
            or any(
                profile.get(key) != expected_interface.get(key)
                for key in (
                    "robot_preset_id",
                    "embodiment_id",
                    "observation_schema_id",
                    "action_schema_id",
                )
            )
            or not isinstance(profile_digest, str)
            or not _DIGEST.fullmatch(profile_digest)
            or profile_digest
            != cross_runtime_canonical_digest(profile, digest_field="profile_digest")
            or not isinstance(delivery, Mapping)
            or delivery.get("mode") != "authenticated_endpoint"
        ):
            raise ValueError("g1_team_policy_endpoint_profile_invalid")
        url = delivery.get("endpoint_url")
        timeout_ms = delivery.get("timeout_ms")
        try:
            parsed = urlsplit(url) if isinstance(url, str) else None
            approved = urlsplit(approved_origin)
            endpoint_port = parsed.port if parsed is not None else None
            approved_port = approved.port
        except ValueError as exc:
            raise ValueError("g1_team_policy_endpoint_configuration_invalid") from exc
        if (
            parsed is None
            or len(url) > 2048
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or endpoint_port not in {None, 443}
            or approved.scheme != "https"
            or approved.path not in {"", "/"}
            or approved.query
            or approved.fragment
            or approved.hostname != parsed.hostname
            or approved_port not in {None, 443}
            or not isinstance(resolved_secret_ref, str)
            or not _SECRET_REF.fullmatch(resolved_secret_ref)
            or delivery.get("auth_secret_ref") != resolved_secret_ref
            or type(timeout_ms) is not int
            or not 100 <= timeout_ms <= 30_000
            or not isinstance(credential, str)
            or not 1 <= len(credential) <= 4096
            or any(not 33 <= ord(char) <= 126 for char in credential)
        ):
            raise ValueError("g1_team_policy_endpoint_configuration_invalid")
        self.url = url
        self.approved_origin = approved_origin
        self.timeout_seconds = timeout_ms / 1000
        self._credential = credential
        self.profile_digest = profile_digest
        self._fetcher = fetcher
        self._request_index = 0
        self.candidate_policy_queried = False

    def _exchange(self, kind: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request_id = self._request_index
        self._request_index += 1
        request = {
            "protocol": PROTOCOL,
            "profile_digest": self.profile_digest,
            "request_id": request_id,
            "kind": kind,
            **payload,
        }
        data = json.dumps(request, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_REQUEST_BYTES:
            raise ValueError("g1_team_policy_endpoint_request_oversized")
        response = self._fetcher(
            self.url,
            method="POST",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + self._credential,
            },
            timeout_seconds=self.timeout_seconds,
            max_bytes=MAX_RESPONSE_BYTES,
            allowed_origins=(self.approved_origin,),
            allowed_content_types=("application/json",),
            max_redirects=0,
        )
        if (
            response.status != 200
            or response.final_url != self.url
            or response.content_type != "application/json"
        ):
            raise ValueError("g1_team_policy_endpoint_response_transport_invalid")
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("g1_team_policy_endpoint_response_not_json") from exc
        if (
            not isinstance(value, Mapping)
            or not json_shape_within_limits(value, max_depth=16, max_items=10000)
            or value.get("protocol") != PROTOCOL
            or value.get("profile_digest") != self.profile_digest
            or type(value.get("request_id")) is not int
            or value["request_id"] != request_id
        ):
            raise ValueError("g1_team_policy_endpoint_response_identity_invalid")
        return value

    def reset(self, *, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("g1_team_policy_endpoint_seed_invalid")
        response = self._exchange("reset", {"seed": seed})
        if response.get("ok") is not True:
            raise ValueError("g1_team_policy_endpoint_reset_ack_invalid")
        self.candidate_policy_queried = False

    def infer_chunk(
        self, *, front_rgb: Any, observation_state: Sequence[float], task: str
    ) -> list[list[float]]:
        payload = build_semantic_v3_infer_request(
            front_rgb=front_rgb,
            observation_state=observation_state,
            task=task,
        )
        response = self._exchange("infer", payload)
        self.candidate_policy_queried = True
        return validate_semantic_v3_infer_response(response)
