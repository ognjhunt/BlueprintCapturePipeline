"""Bind selected dispatch inputs without granting or consuming paid authority.

The canonical allocator must still admit spend, hold launch slots and arm an
independent watchdog. This object carries private transport inputs separately
from its safe receipt and reopens live authority at the mutation boundary.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_policy_authority import verify_g1_team_policy_authority
from .native_g1_team_policy_credentials import (
    G1TeamPolicyCredentialBinding,
    resolve_g1_team_policy_credential,
)
from .native_g1_team_policy_conformance import run_g1_team_endpoint_synthetic_conformance
from .native_g1_team_provider_bundle import load_verified_g1_team_provider_bundle
from .native_g1_team_provider_runtime import CREDENTIAL_FILE_ENV


@dataclass(frozen=True, repr=False)
class G1TeamDispatchInputs:
    bundle: Mapping[str, Any] = field(repr=False)
    _credential: G1TeamPolicyCredentialBinding = field(repr=False)
    _arguments: Mapping[str, Any] = field(repr=False)
    _receipt: Mapping[str, Any] = field(repr=False)

    def __repr__(self) -> str:
        return "G1TeamDispatchInputs(spend_admitted=False)"

    def safe_receipt(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._receipt))

    def runtime_secret_file_paths(self) -> dict[str, Path]:
        """Private adapter argument; never serialize as run receipt metadata."""
        return {CREDENTIAL_FILE_ENV: self._credential.credential_file}

    def recheck(self) -> dict[str, Any]:
        self._credential.recheck()
        current = verify_g1_team_dispatch_inputs(**self._arguments)
        if current.safe_receipt() != self.safe_receipt():
            raise ValueError("g1_team_dispatch_inputs_changed")
        return current.safe_receipt()

    def probe_synthetic_endpoint(self, *, fetcher: Any = None) -> dict[str, Any]:
        """Check the chosen wire before allocation without captured observations."""
        try:
            self.recheck()
            authority = verify_g1_team_policy_authority(**{
                **self._arguments["authority_arguments"], "now_epoch": time.time(),
            })
            profile = authority["intent"]["request"]["policy_profile"]
            binding = authority["operator_approval"]["runtime_binding"]
            conformance = run_g1_team_endpoint_synthetic_conformance(
                profile=profile, trusted_setup=authority["trusted_setup"],
                authenticated_owner=profile["owner"],
                approved_origin=binding["approved_origin"],
                resolved_secret_ref=binding["resolved_secret_ref"],
                credential=self._credential.read_for_endpoint_probe(), fetcher=fetcher,
            )
            self.recheck()
        except Exception:
            # A remote endpoint or fetcher may include credentials or response
            # bodies in its exception. Neither belongs in controller logs.
            raise ValueError("g1_team_endpoint_synthetic_preflight_failed") from None
        receipt = {
            "schema_version": "native_g1_team_preallocation_synthetic_check.v1",
            "status": "synthetic_wire_compatible_before_allocation",
            "selected_input_receipt_digest": self._receipt["receipt_digest"],
            "synthetic_conformance": conformance,
            "gpu_provider_mutation_performed": False,
            "claim_ceiling": "development_only", "public_redistribution_authorized": False,
        }
        receipt["receipt_digest"] = digest(receipt, digest_field="receipt_digest")
        return receipt


def verify_g1_team_dispatch_inputs(
    *, bundle_receipt_path: Path, authority_arguments: Mapping[str, Any],
    expected_implementation_commit: str, credential_registry_path: Path | None,
    max_hourly_rate_usd: float, hard_cap_usd: float, hard_ttl_seconds: int,
) -> G1TeamDispatchInputs:
    """Verify exact input bytes, owner rights, authorized budgets and mode support."""

    current_arguments = dict(authority_arguments)
    current_arguments["now_epoch"] = time.time()
    bundle = load_verified_g1_team_provider_bundle(
        bundle_receipt_path, expected_implementation_commit=expected_implementation_commit,
        authority_arguments=current_arguments,
    )
    authority = verify_g1_team_policy_authority(**current_arguments)
    intent = authority["intent"]
    authorization = intent["request"]["authorization"]
    rate, cap, ttl = max_hourly_rate_usd, hard_cap_usd, hard_ttl_seconds
    if (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 5
            or type(cap) not in (int, float) or not math.isfinite(cap)
            or not 0 < cap <= min(12, authorization["maximum_cost_usd"])
            or type(ttl) is not int or not 1800 <= ttl <= authorization["hard_ttl_seconds"]):
        raise ValueError("g1_team_dispatch_budget_invalid")
    if bundle["policy_runtime_required"]:
        # This is an observed missing capability, not an authorization prompt
        # or a permanent rejection of OCI/archive delivery. The paired-runtime
        # implementation must extend this boundary before those modes launch.
        raise ValueError("g1_team_dispatch_paired_policy_runtime_required")
    if credential_registry_path is None:
        raise ValueError("g1_team_dispatch_credential_registry_required")
    credential = resolve_g1_team_policy_credential(
        registry_path=credential_registry_path, authority_arguments=current_arguments,
        now_epoch=current_arguments["now_epoch"],
    )
    receipt = {
        "schema_version": "native_g1_team_dispatch_input_verification.v1",
        "status": "verified_inputs_not_spend_admitted",
        "implementation_commit": expected_implementation_commit,
        "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
        "owner": intent["request"]["owner"],
        "execution_packet_digest": bundle["execution_packet_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_manifest_digest": bundle["manifest_digest"],
        "profile_digest": bundle["policy_profile_digest"],
        "operator_approval_digest": authority["operator_approval"]["approval_digest"],
        "source_packet_receipt_digest": bundle["source_packet_receipt_digest"],
        "delivery_mode": bundle["delivery_mode"],
        "credential_binding": credential.safe_receipt(),
        "max_hourly_rate_usd": rate, "hard_cap_usd": cap,
        "hard_ttl_seconds": ttl, "retry_cap": 0,
        "canonical_allocator_required": True,
        "spend_admission_status": "not_checked", "provider_mutation_performed": False,
        "claim_ceiling": "development_only", "public_redistribution_authorized": False,
    }
    receipt["receipt_digest"] = digest(receipt, digest_field="receipt_digest")
    arguments = {
        "bundle_receipt_path": bundle_receipt_path, "authority_arguments": authority_arguments,
        "expected_implementation_commit": expected_implementation_commit,
        "credential_registry_path": credential_registry_path,
        "max_hourly_rate_usd": rate, "hard_cap_usd": cap, "hard_ttl_seconds": ttl,
    }
    return G1TeamDispatchInputs(bundle, credential, arguments, receipt)
