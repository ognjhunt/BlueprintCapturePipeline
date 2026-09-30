"""Fetch job-bound checkpoint access through the signed production control plane.

Registry secrets go directly from the credential broker to the isolated worker.
Bearer tokens are bound to one HTTPS endpoint and kept in private runtime inputs.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from .controlled_policy_configuration import canonical_request_digest


class CheckpointPolicyCredentialClient:
    def __init__(self, *, request: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        from .agent_run_executor import AgentRunWebAppClient

        self.reference = payload.get("credential_ref")
        self.kind = payload.get("credential_kind")
        if (not isinstance(self.reference, str)
                or re.fullmatch(r"policy-credential-[0-9a-f-]{36}", self.reference) is None
                or self.kind not in {"registry", "bearer"}):
            raise ValueError("checkpoint_policy_credential_reference_invalid")
        token_path = Path(os.environ.get("BLUEPRINT_PIPELINE_SYNC_TOKEN_FILE", ""))
        if not token_path.is_absolute() or token_path.is_symlink() or not token_path.is_file():
            raise ValueError("checkpoint_policy_credential_worker_identity_invalid")
        metadata = token_path.stat()
        mode = stat.S_IMODE(metadata.st_mode)
        protected_group = (mode in {0o440, 0o640}
            and metadata.st_gid in {os.getgid(), *os.getgroups()})
        if (metadata.st_uid not in {0, os.getuid()}
                or not (mode in {0o400, 0o600} or protected_group)):
            raise ValueError("checkpoint_policy_credential_worker_identity_invalid")
        token = token_path.read_text().strip()
        endpoint = os.environ.get("BLUEPRINT_WEBAPP_URL", "")
        if not endpoint.startswith("https://") or len(token) < 32:
            raise ValueError("checkpoint_policy_credential_service_not_configured")
        self.client = AgentRunWebAppClient(base_url=endpoint, token=token, timeout_seconds=60)
        self.binding = {"job_id": request["job_id"],
            "canonical_request_digest": canonical_request_digest(request)}

    def _request(self, action: str, **values: Any) -> dict[str, Any]:
        row = self.client._json(
            "/api/internal/pipeline/checkpoint-policy-credentials/" + self.reference,
            method="POST", payload={**self.binding, "action": action, **values})
        if row.get("ok") is not True:
            raise ValueError("checkpoint_policy_credential_delivery_rejected")
        if action != "bind_admission" and any(row.get(k) != v for k, v in self.binding.items()):
            raise ValueError("checkpoint_policy_credential_job_binding_mismatch")
        return row

    def access(self) -> dict[str, Any]:
        row = self._request("access")
        if row.get("credential_ref") != self.reference or row.get("kind") != self.kind:
            raise ValueError("checkpoint_policy_credential_kind_mismatch")
        return row

    def registry_lease(self, *, contract: Mapping[str, Any], tenant_id: str) -> str:
        row = self._request("registry_lease", contract=dict(contract), tenant_id=tenant_id)
        lease = row.get("lease") or {}
        identity = lease.get("lease_id")
        if (not isinstance(identity, str)
                or re.fullmatch(r"policy-registry-lease-[0-9a-f]{47}", identity) is None
                or lease.get("run_id") != self.binding["job_id"]
                or lease.get("contract_digest") != contract["contract_digest"]
                or lease.get("image") != contract["container"]["image"]
                or lease.get("status") != "active" or lease.get("single_use") is not True):
            raise ValueError("checkpoint_policy_registry_lease_binding_mismatch")
        return identity

    def bind_admission(self, *, contract: Mapping[str, Any], tenant_id: str,
                       admission: Mapping[str, Any], lease_id: str) -> None:
        row = self._request("bind_admission", contract=dict(contract), tenant_id=tenant_id,
            admission_receipt=dict(admission))
        if row.get("lease_id") != lease_id or row.get("admission_id") != admission["admission_id"]:
            raise ValueError("checkpoint_policy_registry_admission_binding_mismatch")
