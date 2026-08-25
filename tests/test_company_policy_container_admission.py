from __future__ import annotations

import copy
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.company_policy_container_admission import (
    ADMISSION_ROOT_ENV,
    CompanyPolicyContainerAdmissionError,
    stage_company_policy_container_admission,
    validate_company_policy_container_admission_request,
)
from blueprint_pipeline.company_policy_container_contract_v2 import (
    ACTION_ROUTE,
    CLAIM_CEILING,
    LIVE_HANDSHAKE_KIND,
    LIVE_PROTOCOL_VERSION,
    SCHEMA_VERSION,
    validate_company_policy_container_contract_v2,
)


def _contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": "acme_widget_grasp_v3",
        "company_id": "acme_robotics",
        "display_name": "ACME Widget Grasp v3",
        "checkpoint_identity": {
            "repository": "registry.acme.example/models/widget-grasp",
            "revision": "2026.08.1",
            "inventory_digest": "sha256:" + "a" * 64,
        },
        "claim_ceiling": CLAIM_CEILING,
        "rights": {
            "license": "ACME evaluation license 2026-08",
            "rights_provenance": "acme_msa_2026_07_appendix_b",
            "rights_evidence_uri": "blueprint-rights://acme/widget-grasp/2026-08",
            "rights_evidence_digest": "sha256:" + "b" * 64,
            "provider_use_status": "permitted_for_this_evaluation",
            "redistribution_status": "weights_remain_in_company_container",
            "rights_ready": True,
        },
        "container": {
            "image": "registry.acme.example/widget-grasp@sha256:" + "c" * 64,
            "visibility": "private",
            "serve_command": ["python", "-m", "acme_policy.serve", "--port", "8600"],
            "port": 8600,
            "handshake": {
                "kind": LIVE_HANDSHAKE_KIND,
                "protocol_version": LIVE_PROTOCOL_VERSION,
                "action_route": ACTION_ROUTE,
            },
            "run_as_uid": 65532,
            "run_as_gid": 65532,
            "gpu_required": True,
            "resources": {
                "cpus": 8.0,
                "memory_mib": 32768,
                "pids_limit": 512,
                "tmpfs_mib": 2048,
                "startup_timeout_seconds": 300,
                "request_timeout_ms": 2500,
            },
        },
        "robot": {
            "embodiment_id": "franka_panda_robotiq_2f85_v1",
            "definition_uri": "blueprint-robot://franka-panda-robotiq-2f85/v1",
            "definition_digest": "sha256:" + "d" * 64,
            "joint_names": ["panda_joint1"],
            "joint_limits": [
                {"name": "panda_joint1", "lower": -2.0, "upper": 2.0, "unit": "radian"}
            ],
            "gripper": {
                "name": "gripper",
                "command_interval": [0.0, 1.0],
                "unit": "normalized_fraction",
                "executed_semantics": "clip_then_map_to_parallel_jaw_width",
            },
        },
        "observation_schema": {
            "cameras": [
                {
                    "name": "external_rgb",
                    "width": 320,
                    "height": 180,
                    "color_space": "rgb",
                    "dtype": "uint8",
                    "layout": "hwc",
                    "encoding": "lossless_png",
                    "calibration_uri": "blueprint-calibration://scene/external/v1",
                    "calibration_digest": "sha256:" + "e" * 64,
                }
            ],
            "state_fields": [
                {"name": "joint_position", "shape": [1], "dtype": "float32", "unit": "radian"}
            ],
            "prompt": {"mode": "text", "required": True},
            "control_frequency_hz": 15.0,
        },
        "action_schema": {
            "adapter_id": "absolute_joint_position_gripper_v1",
            "chunk_rows": 15,
            "channels": [
                {
                    "name": "panda_joint1",
                    "kind": "bounded_continuous",
                    "command_interval": [-2.0, 2.0],
                    "raw_accepted_bounds": [-2.0, 2.0],
                    "unit": "radian",
                    "executed_semantics": "absolute_joint_position",
                },
                {
                    "name": "gripper",
                    "kind": "threshold_scalar",
                    "command_interval": [0.0, 1.0],
                    "raw_accepted_bounds": [-0.25, 1.25],
                    "unit": "normalized_fraction",
                    "executed_semantics": "clip_then_map_to_parallel_jaw_width",
                },
            ],
            "normalization": {
                "observation": "none",
                "action": "none",
                "gripper": "raw_envelope_then_clip_to_command_interval",
            },
        },
    }


def _request(*, private: bool = True) -> dict[str, Any]:
    contract = _contract()
    contract["container"]["visibility"] = "private" if private else "public"
    normalized = validate_company_policy_container_contract_v2(contract)
    return {
        "schema_version": "company_policy_container_admission_request.v1",
        "tenant_id": "tenant-01",
        "run_id": "run-01",
        "submission_id": "policy-candidate-" + "1" * 40,
        "company_id": "acme_robotics",
        "contract_digest": normalized["contract_digest"],
        "contract": normalized,
        "registry_credential_lease_id": (
            "policy-registry-lease-" + "2" * 47 if private else None
        ),
        "claim_ceiling": "development_only",
        "launch_authority_granted": False,
        "provider_mutation_authorized": False,
    }


def test_admission_is_digest_bound_secret_free_and_no_spend(tmp_path: Path) -> None:
    receipt = stage_company_policy_container_admission(value=_request(), root=tmp_path)

    assert receipt["accepted"] is True
    assert receipt["status"] == "admitted_no_spend"
    assert receipt["registry_credential_consumed"] is False
    assert receipt["profile_published"] is False
    assert receipt["launch_queued"] is False
    assert receipt["launch_authority_granted"] is False
    assert receipt["provider_mutation_performed"] is False
    admission_root = tmp_path / receipt["admission_id"]
    assert (admission_root / "admission_request.json").stat().st_mode & 0o777 == 0o600
    assert (admission_root / "admission_receipt.json").stat().st_mode & 0o777 == 0o600
    retained = json.loads((admission_root / "admission_request.json").read_text())
    assert retained["contract_digest"] == receipt["contract_digest"]
    assert "registry_secret" not in json.dumps(retained)


def test_admission_retry_is_idempotent_and_conflict_fails_closed(tmp_path: Path) -> None:
    request = _request()
    first = stage_company_policy_container_admission(value=request, root=tmp_path)
    second = stage_company_policy_container_admission(value=request, root=tmp_path)
    assert first["admission_id"] == second["admission_id"]
    assert second["already_exists"] is True

    conflict = copy.deepcopy(request)
    conflict["contract"]["display_name"] = "Tampered"
    with pytest.raises(CompanyPolicyContainerAdmissionError) as excinfo:
        stage_company_policy_container_admission(value=conflict, root=tmp_path)
    assert any("contract_digest_mismatch" in item for item in excinfo.value.blockers)


@pytest.mark.parametrize(
    "mutate,blocker",
    [
        (lambda value: value.update(registry_secret="nope"), "secret_carrier_detected"),
        (lambda value: value.update(launch_authority_granted=True), "launch_authority_forbidden"),
        (lambda value: value.update(provider_mutation_authorized=True), "provider_authority_forbidden"),
        (lambda value: value.update(company_id="other_company"), "company_id_mismatch"),
        (lambda value: value.update(registry_credential_lease_id=None), "private_lease_required"),
    ],
)
def test_invalid_or_authoritative_handoffs_fail_closed(mutate, blocker: str) -> None:
    request = _request()
    mutate(request)
    with pytest.raises(CompanyPolicyContainerAdmissionError) as excinfo:
        validate_company_policy_container_admission_request(request)
    assert any(blocker in item for item in excinfo.value.blockers)


def test_public_image_forbids_credential_lease() -> None:
    request = _request(private=False)
    request["registry_credential_lease_id"] = "policy-registry-lease-" + "2" * 47
    with pytest.raises(CompanyPolicyContainerAdmissionError) as excinfo:
        validate_company_policy_container_admission_request(request)
    assert "company_policy_container_admission_public_lease_forbidden" in excinfo.value.blockers


def test_http_intake_requires_signed_admission_and_never_queues(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "admissions"
    monkeypatch.setenv(ADMISSION_ROOT_ENV, str(root))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "intake-work"))
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv(
        service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"blueprint-webapp": "test-intake-token"}),
    )
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    client = TestClient(service.create_app())

    request_body = json.dumps(_request(), separators=(",", ":")).encode("utf-8")
    unauthorized = client.post(
        "/api/live-pipeline/company-policy-containers", content=request_body
    )
    assert unauthorized.status_code == 401
    timestamp = datetime.now(timezone.utc).isoformat()
    client_id = "blueprint-webapp"
    nonce = "company-policy-" + "3" * 48
    signature = hmac.new(
        b"test-intake-token",
        f"{timestamp}.{client_id}.{nonce}.".encode("utf-8") + request_body,
        "sha256",
    ).hexdigest()
    response = client.post(
        "/api/live-pipeline/company-policy-containers",
        content=request_body,
        headers={
            "content-type": "application/json",
            "x-blueprint-pipeline-timestamp": timestamp,
            "x-blueprint-pipeline-client-id": client_id,
            "x-blueprint-pipeline-nonce": nonce,
            "x-blueprint-pipeline-signature": f"sha256={signature}",
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["launch_queued"] is False
    assert not (tmp_path / "task_evaluation_launches").exists()
