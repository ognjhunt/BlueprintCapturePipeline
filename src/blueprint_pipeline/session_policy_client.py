"""The episode's client for a policy served on the session's own machine.

Plan 2.1 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``): a
policy reaches the simulator through one contract, whoever serves it. A policy
Blueprint runs itself is served by a contract server on the rented GPU
(``openpi_droid_contract_server``) and reached here over loopback HTTP. The
request, the response, the identity handshake and every evidence field are the
unchanged ``DroidHttpsTeamPolicyClient`` of ``openpi_droid_https.v1``; only the
transport differs, because a loopback listener needs neither TLS nor a public
origin. The URLs must name ``http://127.0.0.1:<port>`` exactly; anything else
is refused before a request is made.

Failure attribution, as on the native path. A server that cannot be reached
(nothing listening: it never became ready, or it is already gone before this
request) is the session's infrastructure, never the policy:
``session_policy_server_unavailable``. A connection lost while an inference
request was in flight (reset or closed before the answer) means the server
died serving that request: on the native path the same crash is the policy's
failure for that try, so here it is ``droid_https_policy_response_connection_lost``.
The tries after it find nothing listening and are void as infrastructure. An
answer the server gave (an error status or a bad envelope) and an answer that
took longer than the native OpenPI inference bound stay the policy's too. A
lost identity read (the metadata request) stays infrastructure: no inference
was in flight.
"""
from __future__ import annotations

import http.client
import re
import urllib.error
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

# The canary bundle ships the ``core`` subpackage beside this module.
from .core.security_controls import SecurityValidationError, fetch_bounded_service_url
from .droid_https_policy_client import (
    INTERFACE_ID, MAX_RESPONSE_BYTES, SUPPORTED_ACTION_CHUNK_ROWS, DroidHttpsPolicySpec, DroidHttpsTeamPolicyClient,
)
from .openpi_droid_policy_runtime import OPENPI_INFERENCE_TIMEOUT_SECONDS

LOOPBACK_HOST = "127.0.0.1"
UNAVAILABLE = "session_policy_server_unavailable"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
# ``core.security_controls`` wraps a refused connect and a connection closed mid-response in these.
_REFUSED = "all validated remote IP connections failed"
_CLOSED = "remote connection closed before completion"
_UNREACHABLE = frozenset({_REFUSED, _CLOSED})
#: A connection the server dropped after accepting the request (it died while serving it).
_LOST = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, http.client.RemoteDisconnected,
         http.client.IncompleteRead)
LOST = "droid_https_policy_response_connection_lost"


def loopback_origin(port: int) -> str:
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("session_policy_port_invalid")
    return f"http://{LOOPBACK_HOST}:{port}"


@dataclass(frozen=True)
class LoopbackDroidPolicySpec(DroidHttpsPolicySpec):
    """The contract's policy spec, with its endpoint on the session's loopback."""

    def validate(self) -> None:
        origin = urlsplit(self.allowed_origin)
        if (origin.scheme != "http" or origin.hostname != LOOPBACK_HOST or origin.port is None
                or loopback_origin(origin.port) != self.allowed_origin):
            raise ValueError("session_policy_endpoint_not_loopback")
        for url in (self.endpoint_url, self.metadata_url):
            parsed = urlsplit(url)
            if (f"{parsed.scheme}://{parsed.netloc}" != self.allowed_origin or parsed.query or parsed.fragment
                    or not parsed.path.startswith("/") or parsed.username or parsed.password):
                raise ValueError("session_policy_endpoint_not_loopback")
        if self.endpoint_url == self.metadata_url:
            raise ValueError("session_policy_endpoint_not_loopback")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.policy_id):
            raise ValueError("droid_https_policy_id_invalid")
        if self.policy_interface_id != INTERFACE_ID or self.action_space != "joint_position":
            raise ValueError("droid_https_policy_interface_mismatch")
        if not _DIGEST.fullmatch(self.model_identity_digest):
            raise ValueError("droid_https_policy_model_identity_invalid")
        if (self.action_chunk_rows not in SUPPORTED_ACTION_CHUNK_ROWS
                or not 1 <= self.open_loop_horizon <= self.action_chunk_rows):
            raise ValueError("droid_https_policy_chunk_contract_invalid")


def loopback_transport(spec: LoopbackDroidPolicySpec, *, fetch: Any = fetch_bounded_service_url):
    """The client's transport over loopback HTTP, with the native inference time bound."""

    spec.validate()

    def send(*, url: str, method: str, body: bytes | None, headers: Mapping[str, str]) -> bytes:
        if f"{urlsplit(url).scheme}://{urlsplit(url).netloc}" != spec.allowed_origin:
            raise ValueError("session_policy_endpoint_not_loopback")
        inference = url == spec.endpoint_url and method == "POST"
        try:
            return fetch(url, method=method, data=body, headers=dict(headers),
                         timeout_seconds=int(OPENPI_INFERENCE_TIMEOUT_SECONDS), max_bytes=MAX_RESPONSE_BYTES,
                         allowed_origins=(spec.allowed_origin,), allowed_content_types=("application/json",),
                         max_redirects=0).body
        except urllib.error.HTTPError as exc:
            # The server answered: its refusal is the policy's, as on the native path.
            raise ValueError(f"droid_https_policy_response_status_invalid:{int(exc.code)}") from None
        except TimeoutError:
            raise ValueError("droid_https_policy_response_timeout") from None
        except SecurityValidationError as exc:
            message = str(exc)
            if "time limit" in message:
                raise ValueError("droid_https_policy_response_timeout") from None
            cause = exc.__cause__ or exc
            if inference and (message == _CLOSED or isinstance(cause, _LOST)):
                raise ValueError(f"{LOST}:{type(cause).__name__}") from None  # died serving this request
            if message in _UNREACHABLE:  # Nothing listening, or an identity read dropped.
                raise ValueError(f"{UNAVAILABLE}:{type(cause).__name__}") from None
            raise ValueError("droid_https_policy_response_transport_invalid") from None
        except (ConnectionError, http.client.HTTPException, OSError) as exc:
            if inference and isinstance(exc, _LOST):
                raise ValueError(f"{LOST}:{type(exc).__name__}") from None
            raise ValueError(f"{UNAVAILABLE}:{type(exc).__name__}") from None

    return send


def session_policy_client(spec: Mapping[str, Any], bearer_token: str | None, *,
                          fetch: Any = fetch_bounded_service_url) -> DroidHttpsTeamPolicyClient:
    """The unchanged contract client over the session's loopback transport."""

    policy = LoopbackDroidPolicySpec(**spec["policy_spec"])
    return DroidHttpsTeamPolicyClient(spec=policy, bearer_token=bearer_token,
                                      transport=loopback_transport(policy, fetch=fetch))


__all__ = ["LOOPBACK_HOST", "LOST", "LoopbackDroidPolicySpec", "UNAVAILABLE", "loopback_origin", "loopback_transport",
           "session_policy_client"]
