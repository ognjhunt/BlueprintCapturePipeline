#!/usr/bin/env python3
"""Serve one authorized policy container to a remote native simulator episode.

Run only on a dedicated Linux worker with Docker/runsc and loaded security
profiles. The operator must first seal the exact plan, task-bound contract,
boot receipt, and execution authority. This command never creates that authority.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from blueprint_pipeline.company_policy_sandbox_executor import (
    HttpCredentialBroker, SubprocessCommandRunner,
    execute_company_policy_sandbox_preobservation,
)
from blueprint_pipeline.controlled_policy_bridge_server import QualifiedPolicyBridge
from blueprint_pipeline.controlled_policy_configuration import canonical_request_digest
from blueprint_pipeline.policy_model_onnx import validate_model_task_binding


class BlueprintArtifactRegistryBroker:
    """One-use VM identity for a Blueprint-owned private policy image only."""

    def __init__(self, *, lease_id: str, registry_host: str, image_ref: str) -> None:
        if (not lease_id.startswith("blueprint-worker-token-")
                or registry_host != "us-central1-docker.pkg.dev"
                or not image_ref.startswith("us-central1-docker.pkg.dev/blueprint-8c1ca/pipeline-jobs/")
                or "@sha256:" not in image_ref):
            raise ValueError("controlled_policy_worker_registry_scope_invalid")
        self.lease_id = lease_id
        self.registry_host = registry_host
        self.delivery_id: str | None = None

    def claim(self, *, lease_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        if lease_id != self.lease_id or self.delivery_id is not None:
            raise ValueError("controlled_policy_worker_token_claim_invalid")
        request = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(request, timeout=5) as response:
            token = json.loads(response.read(4096))
        if (token.get("token_type") != "Bearer" or not isinstance(token.get("access_token"), str)
                or len(token["access_token"]) < 32 or int(token.get("expires_in", 0)) < 60):
            raise ValueError("controlled_policy_worker_token_invalid")
        self.delivery_id = "blueprint-worker-delivery-" + secrets.token_hex(16)
        return {"credential": {"registry_server": self.registry_host,
                               "username": "oauth2accesstoken", "secret": token["access_token"]},
                "delivery_receipt": {"delivery_id": self.delivery_id}}

    def acknowledge(self, *, lease_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        if (lease_id != self.lease_id or self.delivery_id is None
                or body.get("delivery_id") != self.delivery_id
                or not str(body.get("image_pull_receipt_digest", "")).startswith("sha256:")):
            raise ValueError("controlled_policy_worker_token_ack_invalid")
        delivery_id = self.delivery_id
        self.delivery_id = None
        return {"lease_receipt": {"status": "consumed", "ciphertext_deleted": True,
                                  "delivery_id": delivery_id,
                                  "source": "short_lived_vm_identity_no_persisted_ciphertext"}}


def _object(path: Path, *, private: bool = False) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("controlled_policy_bridge_input_path_invalid")
    if private and (stat.S_IMODE(path.stat().st_mode) & 0o077 or path.stat().st_uid != os.geteuid()):
        raise ValueError("controlled_policy_bridge_private_input_invalid")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("controlled_policy_bridge_input_object_required")
    return value


def _secret(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("controlled_policy_bridge_secret_path_invalid")
    if stat.S_IMODE(path.stat().st_mode) & 0o077 or path.stat().st_uid != os.geteuid():
        raise ValueError("controlled_policy_bridge_secret_mode_invalid")
    value = path.read_bytes().strip()
    if not 32 <= len(value) <= 512 or any(byte in b"\r\n \t" for byte in value):
        raise ValueError("controlled_policy_bridge_secret_invalid")
    return value


def _authority(value: Mapping[str, Any], *, plan: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    if set(value) != {"schema_version", "job_id", "canonical_request_digest", "plan_digest",
                      "expires_at_iso", "approved_by", "max_policy_calls"}:
        raise ValueError("controlled_policy_bridge_authority_schema_invalid")
    try:
        expiry = datetime.fromisoformat(str(value["expires_at_iso"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("controlled_policy_bridge_authority_expiry_invalid") from exc
    if (value["schema_version"] != "blueprint.controlled_policy_bridge_authority.v1"
            or value["job_id"] != request.get("job_id")
            or value["canonical_request_digest"] != canonical_request_digest(request)
            or value["plan_digest"] != plan.get("plan_digest")
            or expiry.tzinfo is None or expiry <= datetime.now(timezone.utc)
            or not isinstance(value["approved_by"], str) or not value["approved_by"]
            or type(value["max_policy_calls"]) is not int or not 1 <= value["max_policy_calls"] <= 128):
        raise ValueError("controlled_policy_bridge_authority_binding_invalid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "contract", "job-request", "authority", "worker-boot-receipt",
                 "attestation-key-file", "bearer-token-file", "tls-certificate", "tls-private-key",
                 "manifest-out", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--attestation-key-id", required=True)
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=1800)
    parser.add_argument("--broker-base-url")
    parser.add_argument("--broker-token-file", type=Path)
    parser.add_argument("--blueprint-owned-vm-identity", action="store_true")
    parser.add_argument("--broker-client-id", default="blueprint-policy-sandbox-worker")
    parser.add_argument("--ack", choices=["authorized-controlled-policy-session"], required=True)
    args = parser.parse_args()
    plan = _object(args.plan)
    contract = _object(args.contract)
    request = _object(args.job_request)
    authority = _object(args.authority, private=True)
    boot = _object(args.worker_boot_receipt, private=True)
    _authority(authority, plan=plan, request=request)
    modalities = [name for name in ("docker_container", "sim_controller_plugin")
                  if (request.get("policy_package") or {}).get(name)]
    if len(modalities) != 1 or plan.get("contract_digest") != contract.get("contract_digest"):
        raise ValueError("controlled_policy_bridge_policy_binding_invalid")
    payload = request["policy_package"][modalities[0]]
    if payload.get("execution_profile") != "controlled_observation_v1":
        raise ValueError("controlled_policy_bridge_execution_profile_invalid")
    artifact = payload.get("model_artifact")
    if artifact is not None:
        if payload.get("runner_profile") != "onnx_state_mlp_cpu_v1":
            raise ValueError("controlled_policy_bridge_runner_profile_invalid")
        validate_model_task_binding(artifact, contract)
    elif payload.get("image_ref") != contract["container"]["image"]:
        raise ValueError("controlled_policy_bridge_image_binding_invalid")
    if modalities[0] == "sim_controller_plugin":
        if (payload.get("execution_profile") != "controlled_observation_v1"
                or payload.get("transport") != "isolated_container_http_json_v1"):
            raise ValueError("controlled_policy_bridge_controller_transport_invalid")
        from blueprint_pipeline.controlled_native_isaac import validate_native_controller_interface
        validate_native_controller_interface(contract)
    if args.manifest_out.exists() or args.output.exists():
        raise ValueError("controlled_policy_bridge_output_exists")
    token = _secret(args.bearer_token_file).decode("ascii")
    key = _secret(args.attestation_key_file)
    if (not args.tls_certificate.is_file() or args.tls_certificate.is_symlink()
            or not args.tls_private_key.is_file() or args.tls_private_key.is_symlink()
            or stat.S_IMODE(args.tls_private_key.stat().st_mode) & 0o077):
        raise ValueError("controlled_policy_bridge_tls_identity_invalid")
    visibility = contract["container"]["visibility"]
    if visibility == "private":
        if args.blueprint_owned_vm_identity:
            if args.broker_base_url or args.broker_token_file:
                raise ValueError("controlled_policy_bridge_broker_modes_conflict")
            broker = BlueprintArtifactRegistryBroker(
                lease_id=str(plan["credential_broker_request_binding"]["registry_credential_lease_id"]),
                registry_host=str(plan["registry"]["host"]),
                image_ref=str(contract["container"]["image"]))
        else:
            if not args.broker_base_url or args.broker_token_file is None:
                raise ValueError("controlled_policy_bridge_private_registry_broker_required")
            broker = HttpCredentialBroker(base_url=args.broker_base_url,
                token_file=args.broker_token_file, client_id=args.broker_client_id)
    else:
        broker = None
    bridge = QualifiedPolicyBridge(contract=contract, job_request=request,
        endpoint_url=args.endpoint_url, bearer_token=token, manifest_path=args.manifest_out,
        bind_host=args.bind_host, bind_port=args.bind_port,
        tls_certificate=args.tls_certificate, tls_private_key=args.tls_private_key,
        tls_certificate_pem=args.tls_certificate.read_text(),
        maximum_seconds=args.maximum_seconds, max_policy_calls=authority["max_policy_calls"])
    result = bridge.run(lambda session: execute_company_policy_sandbox_preobservation(
        plan=plan, contract=contract, broker=broker, runner=SubprocessCommandRunner(),
        attestation_key=key, attestation_key_id=args.attestation_key_id,
        worker_boot_receipt=boot, output_path=args.output,
        qualified_session=session,
        authorize_scene_access=lambda qualified_plan, qualification: (
            qualified_plan["plan_digest"] == authority["plan_digest"]
            and qualification.get("status") == "qualified_before_first_observation"),
    ))
    print(json.dumps({"status": result["status"],
        "terminal_receipt_digest": result.get("terminal_receipt", {}).get("receipt_digest"),
        "cleanup_complete": result.get("terminal_receipt", {}).get("cleanup_complete") is True,
        "real_observation_sent": result.get("real_observation_sent") is True}, sort_keys=True))
    return 0 if result["status"] == "controlled_session_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
