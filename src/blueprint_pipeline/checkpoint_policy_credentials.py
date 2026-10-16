"""Fetch job-bound checkpoint access through the signed production control plane.

Registry secrets go directly from the credential broker to the isolated worker.
Bearer tokens are bound to one HTTPS endpoint and kept in private runtime inputs.

A scored team evaluation (beta journey step L5) claims its bearer token as a
single-use lease (``release_bearer_lease``): the request names the lease id the
admission seals, ``bearer_lease_id(credential_ref, job_id, pipeline_run_id)``,
and the WebApp must release the token for that lease at most once, bound to the
claimed run, and say so in its reply. A reply that does not confirm the one
release is refused, so the admission's ``single_use`` is enforced by the
WebApp, not only asserted here.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from .controlled_policy_configuration import canonical_request_digest
from .decision_evidence_contracts import canonical_digest

BEARER_LEASE_ACTION = "bearer_lease_release"


def bearer_lease_id(*, credential_ref: str, job_id: str, pipeline_run_id: str) -> str:
    """The one bearer lease a claimed run may hold; the WebApp recomputes it from the same three values.

    ``"policy-access-"`` plus the first 32 hex digits of SHA-256 over the compact,
    key-sorted JSON of ``{"credential_ref", "job_id", "pipeline_run_id"}``.
    """

    return "policy-access-" + canonical_digest({"credential_ref": credential_ref, "job_id": job_id,
                                                "pipeline_run_id": pipeline_run_id})[7:39]


class CheckpointPolicyCredentialClient:
    def __init__(self, *, request: Mapping[str, Any], payload: Mapping[str, Any],
                 pipeline_run_id: str | None = None) -> None:
        from .agent_run_executor import AgentRunWebAppClient

        # This is the accepted dispatch owner, carried separately from the frozen
        # request. Never replace it with a job ID or create a new attempt here.
        if (not isinstance(pipeline_run_id, str) or not 1 <= len(pipeline_run_id) <= 200
                or pipeline_run_id != pipeline_run_id.strip()):
            raise ValueError("checkpoint_policy_credential_claim_invalid")
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
            "pipeline_run_id": pipeline_run_id,
            "canonical_request_digest": canonical_request_digest(request)}

    def _request(self, action: str, **values: Any) -> dict[str, Any]:
        try:
            row = self.client._json(
                "/api/internal/pipeline/checkpoint-policy-credentials/" + self.reference,
                method="POST", payload={**self.binding, "action": action, **values})
        except (OSError, ValueError):
            # The native failure receipt persists exception text. Keep transport
            # errors and response bodies out of that receipt and worker logs.
            raise ValueError("checkpoint_policy_credential_delivery_unavailable") from None
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

    def release_bearer_lease(self, *, lease_id: str) -> dict[str, Any]:
        """Claim the run's single-use bearer lease; the WebApp must confirm its one release.

        A WebApp without the lease contract refuses the action, and a second claim
        of the same lease must be refused by the WebApp (both surface as
        ``delivery_unavailable``); a reply that does not confirm exactly one
        release bound to this run is ``bearer_lease_unenforced``.
        """

        expected = bearer_lease_id(credential_ref=self.reference, job_id=self.binding["job_id"],
                                   pipeline_run_id=self.binding["pipeline_run_id"])
        if self.kind != "bearer" or lease_id != expected:
            raise ValueError("checkpoint_policy_bearer_lease_invalid")
        row = self._request(BEARER_LEASE_ACTION, lease_id=lease_id)
        lease = row.get("lease") if isinstance(row.get("lease"), Mapping) else {}
        if (row.get("credential_ref") != self.reference or row.get("kind") != "bearer"
                or lease.get("lease_id") != lease_id or lease.get("single_use") is not True
                or lease.get("status") != "released" or lease.get("release_count") != 1
                or lease.get("credential_ref") != self.reference
                or any(lease.get(key) != self.binding[key] for key in ("job_id", "pipeline_run_id"))):
            raise ValueError("checkpoint_policy_bearer_lease_unenforced")
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
