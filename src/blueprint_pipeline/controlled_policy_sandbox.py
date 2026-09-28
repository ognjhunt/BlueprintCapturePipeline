"""ADP-050/day 28: connect admitted policies to the qualified sandbox runtime.

All configuration and resolvers belong to trusted worker startup. Admission is
still no-spend; separate upstream rights/run/spend authority is required before
any local container mutation, and scene access is authorized after qualification.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from .company_policy_sandbox_executor import (
    CommandRunner,
    CredentialBroker,
    execute_company_policy_sandbox_preobservation,
)
from .company_policy_sandbox_v2 import build_company_policy_sandbox_plan


@dataclass(frozen=True)
class ControlledSandboxConfiguration:
    pipeline_release_sha: str
    worker_identity: str
    blueprint_proxy_image: str
    blueprint_proxy_contract_digest: str
    seccomp_profile_id: str
    seccomp_profile_path: str
    seccomp_profile_digest: str
    apparmor_profile_id: str
    apparmor_profile_source_path: str
    apparmor_profile_digest: str
    allowed_registry_hosts: tuple[str, ...]
    receipt_root: Path


class ControlledSandboxFactory:
    """Concrete ``ControlledPolicyExecutor.sandbox_factory`` implementation.

    ``admission_resolver`` retrieves the retained receipt for this exact owned
    job and contract; it must not trust a receipt embedded in customer JSON.
    ``worker_boot_receipt_resolver`` returns independently signed boot evidence
    bound to the generated attempt, worker and release. This class never creates
    boot evidence or upgrades admission into permission to execute.
    """

    def __init__(
        self, *, configuration: ControlledSandboxConfiguration,
        admission_resolver: Callable[..., Mapping[str, Any]],
        registry_address_resolver: Callable[..., tuple[str, ...]],
        worker_boot_receipt_resolver: Callable[..., Mapping[str, Any]],
        authorize_execution: Callable[..., bool],
        authorize_scene_access: Callable[..., bool],
        runner: CommandRunner, broker: CredentialBroker | None,
        attestation_key: bytes, attestation_key_id: str,
    ) -> None:
        if not isinstance(attestation_key, bytes) or len(attestation_key) < 32:
            raise ValueError("controlled_policy_attestation_key_invalid")
        self.configuration = configuration
        self.admission_resolver = admission_resolver
        self.registry_address_resolver = registry_address_resolver
        self.worker_boot_receipt_resolver = worker_boot_receipt_resolver
        self.authorize_execution = authorize_execution
        self.authorize_scene_access = authorize_scene_access
        self.runner = runner
        self.broker = broker
        self.attestation_key = attestation_key
        self.attestation_key_id = attestation_key_id

    def __call__(
        self, *, contract: Mapping[str, Any], job_request: Mapping[str, Any],
        qualified_session: Callable[..., Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        normalized = validate_company_policy_container_contract_v2(contract)
        admission = self.admission_resolver(contract=normalized, job_request=job_request)
        if self.authorize_execution(
            admission=admission, contract=normalized, job_request=job_request,
        ) is not True:
            raise ValueError("controlled_policy_upstream_execution_authority_required")
        configuration = self.configuration
        attempt_id = f"controlled-policy-{secrets.token_hex(16)}"
        plan = build_company_policy_sandbox_plan(
            admission_receipt=admission, contract=normalized,
            sandbox_attempt_id=attempt_id,
            pipeline_release_sha=configuration.pipeline_release_sha,
            worker_identity=configuration.worker_identity,
            runtime_class="runsc",
            blueprint_proxy_image=configuration.blueprint_proxy_image,
            blueprint_proxy_contract_digest=configuration.blueprint_proxy_contract_digest,
            seccomp_profile_id=configuration.seccomp_profile_id,
            seccomp_profile_path=configuration.seccomp_profile_path,
            seccomp_profile_digest=configuration.seccomp_profile_digest,
            apparmor_profile_id=configuration.apparmor_profile_id,
            apparmor_profile_source_path=configuration.apparmor_profile_source_path,
            apparmor_profile_digest=configuration.apparmor_profile_digest,
            registry_addresses=self.registry_address_resolver(contract=normalized),
            allowed_registry_hosts=configuration.allowed_registry_hosts,
        )
        boot_receipt = self.worker_boot_receipt_resolver(plan=plan)
        root = configuration.receipt_root.expanduser()
        if root.is_symlink():
            raise ValueError("controlled_policy_receipt_root_symlink_forbidden")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_mode & 0o077:
            raise ValueError("controlled_policy_receipt_root_must_be_private")
        attempt_root = root / attempt_id
        attempt_root.mkdir(mode=0o700, exist_ok=False)
        return execute_company_policy_sandbox_preobservation(
            plan=plan, contract=normalized, broker=self.broker, runner=self.runner,
            attestation_key=self.attestation_key, attestation_key_id=self.attestation_key_id,
            worker_boot_receipt=boot_receipt,
            output_path=attempt_root / "sandbox_execution.json",
            qualified_session=qualified_session,
            authorize_scene_access=lambda qualified_plan, qualification: self.authorize_scene_access(
                plan=qualified_plan, qualification=qualification,
                job_request=job_request,
            ) is True,
        )
