"""Pinned HTTPS control-plane client for the dedicated policy sandbox worker."""
from __future__ import annotations

import json
import os
import re
import ssl
import stat
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .controlled_policy_configuration import canonical_request_digest
from .controlled_policy_remote_sandbox import validate_remote_sandbox_bridge


_JOB = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,191}$")
_MAX_RESPONSE = 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any):
        return None


class SandboxManagerClient:
    def __init__(self, *, endpoint_url: str, token_file: Path, certificate_file: Path) -> None:
        parts = urlsplit(endpoint_url)
        if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
                or parts.path not in {"", "/"} or parts.query or parts.fragment):
            raise ValueError("policy_sandbox_manager_endpoint_invalid")
        for path in (token_file, certificate_file):
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise ValueError("policy_sandbox_manager_identity_path_invalid")
        mode = stat.S_IMODE(token_file.stat().st_mode)
        if mode & 0o077 or token_file.stat().st_uid not in {0, os.geteuid()}:
            raise ValueError("policy_sandbox_manager_token_file_not_private")
        token = token_file.read_text().strip()
        if not 32 <= len(token) <= 512 or any(char.isspace() for char in token):
            raise ValueError("policy_sandbox_manager_token_invalid")
        certificate = certificate_file.read_text()
        if "-----BEGIN CERTIFICATE-----" not in certificate:
            raise ValueError("policy_sandbox_manager_certificate_invalid")
        self.endpoint = endpoint_url.rstrip("/")
        self.token = token
        context = ssl.create_default_context(cadata=certificate)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.opener = urllib.request.build_opener(_NoRedirect(),
            urllib.request.HTTPSHandler(context=context))

    def _post(self, path: str, value: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False,
            separators=(",", ":")).encode()
        request = urllib.request.Request(self.endpoint + path, method="POST", data=encoded,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError("policy_sandbox_manager_http_status_invalid")
                body = response.read(_MAX_RESPONSE + 1)
        except urllib.error.HTTPError as exc:
            raise ValueError("policy_sandbox_manager_request_rejected") from exc
        if len(body) > _MAX_RESPONSE:
            raise ValueError("policy_sandbox_manager_response_too_large")
        row = json.loads(body)
        if not isinstance(row, dict):
            raise ValueError("policy_sandbox_manager_response_invalid")
        return row

    def prepare(self, *, job_request: Mapping[str, Any], contract: Mapping[str, Any],
                admission_receipt: Mapping[str, Any]) -> dict[str, Any]:
        response = self._post("/v1/sessions", {"job_request": job_request,
            "contract": contract, "admission_receipt": admission_receipt}, timeout=240)
        if response.get("status") != "qualified" or not isinstance(response.get("manifest"), dict):
            raise ValueError("policy_sandbox_manager_not_qualified")
        bridge = validate_remote_sandbox_bridge(response["manifest"])
        if (bridge["job_id"] != job_request.get("job_id")
                or bridge["canonical_request_digest"] != canonical_request_digest(job_request)
                or bridge["contract_digest"] != contract.get("contract_digest")):
            raise ValueError("policy_sandbox_manager_binding_mismatch")
        return bridge

    def allow_network(self, *, job_id: str, instance_id: int, outbound_ipv4: str) -> dict[str, Any]:
        if not _JOB.fullmatch(job_id):
            raise ValueError("policy_sandbox_manager_job_id_invalid")
        return self._post(f"/v1/sessions/{job_id}/network", {
            "instance_id": instance_id, "outbound_ipv4": outbound_ipv4}, timeout=120)

    def close_network(self, *, job_id: str) -> dict[str, Any]:
        if not _JOB.fullmatch(job_id):
            raise ValueError("policy_sandbox_manager_job_id_invalid")
        return self._post(f"/v1/sessions/{job_id}/close", {}, timeout=120)
