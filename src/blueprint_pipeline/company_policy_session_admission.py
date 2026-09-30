"""Materialize a team-approved policy contract for one confirmed native job."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .company_policy_container_admission import stage_company_policy_container_admission
from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2


_IMAGE = re.compile(r"^(.+)@(sha256:[0-9a-f]{64})$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _policy_payload(request: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    package = request.get("policy_package") or {}
    matches = [(name, dict(package[name])) for name in ("docker_container", "sim_controller_plugin")
               if isinstance(package.get(name), Mapping)]
    if len(matches) != 1 or matches[0][1].get("execution_profile") != "controlled_observation_v1":
        raise ValueError("policy_session_modality_invalid")
    return matches[0]


def materialize_company_policy_contract(*, job_request: Mapping[str, Any],
                                        template: Mapping[str, Any],
                                        approved_model_runner_image: str) -> dict[str, Any]:
    """Bind an operator-approved team/task template to one immutable artifact."""
    request = json.loads(json.dumps(job_request, allow_nan=False))
    candidate = json.loads(json.dumps(template, allow_nan=False))
    modality, payload = _policy_payload(request)
    artifact = payload.get("model_artifact") if modality == "docker_container" else None
    if artifact is not None:
        if payload.get("runner_profile") != "onnx_state_mlp_cpu_v1" or not isinstance(artifact, dict):
            raise ValueError("policy_session_model_profile_invalid")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError("policy_session_model_digest_invalid")
        image = approved_model_runner_image
        repository = str(artifact.get("uri") or "")
        revision = str(artifact.get("storage_generation") or "")
        candidate["container"].update({
            "visibility": "private",
            "serve_command": ["python", "-m", "blueprint_pipeline.policy_model_server"],
            "port": 8600,
            "run_as_uid": 65532,
            "run_as_gid": 65532,
            "gpu_required": False,
        })
    else:
        image = payload.get("image_ref")
        match = _IMAGE.fullmatch(image) if isinstance(image, str) else None
        if match is None:
            raise ValueError("policy_session_image_not_digest_pinned")
        repository, digest = match.groups()
        revision = digest
        candidate["container"]["visibility"] = (
            "private" if payload.get("registry_credential_lease_id")
            or image.startswith("us-central1-docker.pkg.dev/blueprint-8c1ca/pipeline-jobs/")
            else "public"
        )
    if not isinstance(image, str) or _IMAGE.fullmatch(image) is None:
        raise ValueError("policy_session_runner_image_invalid")
    if not repository or (artifact is not None and urlsplit(repository).scheme != "gs"):
        raise ValueError("policy_session_artifact_identity_invalid")
    checkpoint_id = str(request.get("robot_profile", {}).get("robot_profile_id") or "")
    if not checkpoint_id:
        raise ValueError("policy_session_checkpoint_missing")
    candidate["policy_id"] = "policy_" + hashlib.sha256(checkpoint_id.encode()).hexdigest()[:32]
    candidate["display_name"] = "Policy " + checkpoint_id[-16:]
    candidate["checkpoint_identity"] = {
        "repository": repository,
        "revision": revision,
        "inventory_digest": digest,
    }
    candidate["container"]["image"] = image
    candidate.pop("contract_digest", None)
    return validate_company_policy_container_contract_v2(candidate)


def stage_company_policy_session_admission(*, job_request: Mapping[str, Any],
                                           contract: Mapping[str, Any],
                                           tenant_id: str, root: Path,
                                           allowed_registry_hosts: list[str],
                                           registry_credential_lease_id: str | None = None) -> dict[str, Any]:
    """Stage no-spend admission; private third-party images require a real lease."""
    request = json.loads(json.dumps(job_request, allow_nan=False))
    normalized = validate_company_policy_container_contract_v2(contract)
    visibility = normalized["container"]["visibility"]
    image = normalized["container"]["image"]
    if visibility == "private" and not registry_credential_lease_id:
        if image.startswith("us-central1-docker.pkg.dev/blueprint-8c1ca/pipeline-jobs/"):
            registry_credential_lease_id = ("policy-registry-lease-"
                + hashlib.sha256(request["job_id"].encode()).hexdigest()[:47])
        else:
            raise ValueError("policy_session_customer_registry_lease_required")
    if visibility == "public" and registry_credential_lease_id is not None:
        raise ValueError("policy_session_public_registry_lease_forbidden")
    payload = {
        "schema_version": "company_policy_container_admission_request.v1",
        "tenant_id": tenant_id,
        "run_id": request["job_id"],
        "submission_id": request["robot_profile"]["robot_profile_id"],
        "company_id": normalized["company_id"],
        "contract_digest": normalized["contract_digest"],
        "contract": normalized,
        "registry_credential_lease_id": registry_credential_lease_id,
        "claim_ceiling": "development_only",
        "launch_authority_granted": False,
        "provider_mutation_authorized": False,
    }
    receipt = stage_company_policy_container_admission(value=payload, root=root,
        allowed_registry_hosts=allowed_registry_hosts)
    if (receipt.get("run_id") != request["job_id"]
            or receipt.get("contract_digest") != normalized["contract_digest"]):
        raise ValueError("policy_session_admission_binding_invalid")
    return receipt
