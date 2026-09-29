"""Use an operator-qualified policy sandbox from a separate simulator worker.

The GPU simulator cannot launch an untrusted container inside its Vast Docker
instance. A dedicated worker qualifies the image before publishing this
job-bound HTTPS bridge. The simulator only sends projected observations and
requires a terminal cleanup acknowledgement before accepting the episode.
"""
from __future__ import annotations

import json
import re
import time
import http.client
import ipaddress
import ssl
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from .controlled_policy_configuration import canonical_request_digest
from .core.security_controls import fetch_bounded_https

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IMAGE = re.compile(r"[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}\Z")
_PATH = "/v1/controlled-policy"


def validate_remote_sandbox_bridge(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate operator-owned bridge data before an observation can leave."""
    row = dict(value)
    if set(row) != {
        "schema_version", "job_id", "canonical_request_digest", "contract_digest",
        "image_ref", "endpoint_url", "bearer_token", "qualification_receipt_digest",
        "expires_at_iso", "model_artifact_sha256", "tls_certificate_pem",
    } or row["schema_version"] != "blueprint.qualified_policy_bridge.v1":
        raise ValueError("controlled_policy_bridge_schema_invalid")
    parsed = urlsplit(str(row["endpoint_url"]))
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
            or parsed.fragment or parsed.query or parsed.path.rstrip("/") != _PATH):
        raise ValueError("controlled_policy_bridge_endpoint_invalid")
    if (not isinstance(row["bearer_token"], str) or len(row["bearer_token"]) < 32
            or len(row["bearer_token"]) > 512 or any(c.isspace() for c in row["bearer_token"])):
        raise ValueError("controlled_policy_bridge_token_invalid")
    if (not isinstance(row["job_id"], str) or not row["job_id"]
            or not _IMAGE.fullmatch(str(row["image_ref"]))
            or any(not _DIGEST.fullmatch(str(row[field])) for field in (
                "canonical_request_digest", "contract_digest", "qualification_receipt_digest"
            ))):
        raise ValueError("controlled_policy_bridge_binding_invalid")
    model_digest = row["model_artifact_sha256"]
    if model_digest is not None and not _DIGEST.fullmatch(str(model_digest)):
        raise ValueError("controlled_policy_bridge_model_binding_invalid")
    certificate = row["tls_certificate_pem"]
    if certificate is not None:
        if (not isinstance(certificate, str) or len(certificate) > 16_384
                or not certificate.startswith("-----BEGIN CERTIFICATE-----\n")
                or "-----END CERTIFICATE-----" not in certificate):
            raise ValueError("controlled_policy_bridge_certificate_invalid")
        try:
            if not ipaddress.ip_address(parsed.hostname or "").is_global:
                raise ValueError("controlled_policy_bridge_address_not_public")
            ssl.create_default_context(cadata=certificate)
        except (ValueError, ssl.SSLError) as exc:
            raise ValueError("controlled_policy_bridge_certificate_invalid") from exc
    try:
        expiry = datetime.fromisoformat(str(row["expires_at_iso"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("controlled_policy_bridge_expiry_invalid") from exc
    if (expiry.tzinfo is None or expiry <= datetime.now(timezone.utc)
            or expiry > datetime.now(timezone.utc) + timedelta(hours=1)):
        raise ValueError("controlled_policy_bridge_expired")
    return row


class RemoteQualifiedSandboxFactory:
    """Trusted simulator-side half of one already-qualified sandbox session."""

    transport_name = "qualified_sandbox_https_bridge"

    def __init__(self, bridge: Mapping[str, Any], *,
                 request: Callable[..., bytes] | None = None) -> None:
        self.bridge = validate_remote_sandbox_bridge(bridge)
        self.request = request or self._request

    def _request(self, *, route: str, body: Mapping[str, Any], timeout: float,
                 max_bytes: int) -> bytes:
        endpoint = self.bridge["endpoint_url"] + route
        parsed = urlsplit(endpoint)
        data = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "Authorization": "Bearer " + self.bridge["bearer_token"]}
        certificate = self.bridge["tls_certificate_pem"]
        if certificate is None:
            return fetch_bounded_https(
                endpoint, method="POST", data=data, headers=headers,
                timeout_seconds=timeout, max_bytes=max_bytes,
                allowed_origins=(f"{parsed.scheme}://{parsed.netloc}",),
                allowed_content_types=("application/json",), max_redirects=0,
            ).body
        context = ssl.create_default_context(cadata=certificate)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443,
            timeout=timeout, context=context)
        try:
            connection.request("POST", parsed.path, body=data, headers=headers)
            response = connection.getresponse()
            if response.status != 200 or response.headers.get_content_type() != "application/json":
                raise ValueError("controlled_policy_bridge_http_response_invalid")
            if response.length is not None and response.length > max_bytes:
                raise ValueError("controlled_policy_bridge_response_size_invalid")
            result = response.read(max_bytes + 1)
            if len(result) > max_bytes:
                raise ValueError("controlled_policy_bridge_response_size_invalid")
            return result
        finally:
            connection.close()

    def _json(self, route: str, body: Mapping[str, Any], *, timeout: float = 30) -> dict[str, Any]:
        result = json.loads(self.request(route=route, body=body, timeout=timeout, max_bytes=65_536))
        if not isinstance(result, dict):
            raise ValueError("controlled_policy_bridge_response_invalid")
        return result

    def preflight(self, *, contract: Mapping[str, Any], job_request: Mapping[str, Any]) -> dict[str, str]:
        """Reject an unready or mismatched sandbox before paid GPU allocation."""
        bridge = self.bridge
        normalized = validate_company_policy_container_contract_v2(contract)
        package = job_request.get("policy_package") or {}
        selected = [name for name in ("docker_container", "sim_controller_plugin") if package.get(name)]
        if (bridge["job_id"] != job_request.get("job_id")
                or bridge["canonical_request_digest"] != canonical_request_digest(job_request)
                or bridge["contract_digest"] != normalized["contract_digest"]
                or bridge["image_ref"] != normalized["container"]["image"]
                or len(selected) != 1):
            raise ValueError("controlled_policy_bridge_frozen_binding_mismatch")
        artifact = (package[selected[0]] or {}).get("model_artifact")
        model_digest = artifact.get("sha256") if isinstance(artifact, Mapping) else None
        if bridge["model_artifact_sha256"] != model_digest:
            raise ValueError("controlled_policy_bridge_model_binding_mismatch")
        binding = {"job_id": bridge["job_id"], "canonical_request_digest": bridge["canonical_request_digest"],
                   "contract_digest": bridge["contract_digest"], "image_ref": bridge["image_ref"]}
        ready = self._json("/ready", binding)
        if (ready.get("status") != "qualified_before_first_observation"
                or ready.get("qualification_receipt_digest") != bridge["qualification_receipt_digest"]
                or any(ready.get(key) != value for key, value in binding.items())):
            raise ValueError("controlled_policy_bridge_not_qualified")
        return binding

    def __call__(self, *, contract: Mapping[str, Any], job_request: Mapping[str, Any],
                 qualified_session: Callable[..., Mapping[str, Any]]) -> dict[str, Any]:
        binding = self.preflight(contract=contract, job_request=job_request)
        calls = 0
        session: Mapping[str, Any] | None = None
        try:
            def transport(body: bytes, timeout: float) -> bytes:
                nonlocal calls
                if not isinstance(body, bytes) or len(body) > 8 * 1024 * 1024:
                    raise ValueError("controlled_policy_bridge_observation_size_invalid")
                wire = json.loads(body)
                result = self.request(route="/actions", body={**binding, "observation": wire},
                                      timeout=timeout, max_bytes=65_536)
                calls += 1
                return result
            session = qualified_session(transport)
        finally:
            # Cleanup is required even after a policy or simulator exception.
            finish = self._json("/finish", {**binding, "policy_calls": calls})
            if finish.get("status") != "cleanup_pending":
                raise ValueError("controlled_policy_bridge_finish_rejected")
            deadline = time.monotonic() + 120
            while True:
                terminal = self._json("/terminal", binding, timeout=10)
                if terminal.get("status") != "terminal_pending":
                    break
                if time.monotonic() >= deadline:
                    raise ValueError("controlled_policy_bridge_terminal_timeout")
                time.sleep(1)
        if (terminal.get("status") != "controlled_session_completed"
                or terminal.get("cleanup_complete") is not True
                or terminal.get("policy_calls") != calls
                or any(terminal.get(key) != value for key, value in binding.items())):
            raise ValueError("controlled_policy_bridge_cleanup_unverified")
        return {"status": "controlled_session_completed", "controlled_session": dict(session or {}),
                "terminal_receipt": terminal}
