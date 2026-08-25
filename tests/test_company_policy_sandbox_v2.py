from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.company_policy_container_contract_v2 import (
    SECURITY_PROFILE,
    validate_company_policy_container_contract_v2,
)
from blueprint_pipeline.company_policy_sandbox_v2 import (
    DENIAL_PROBES,
    REACHABILITY_PROBES,
    CompanyPolicySandboxV2Error,
    build_company_policy_sandbox_plan,
    evaluate_preobservation_sandbox_qualification,
)


def _contract() -> dict[str, Any]:
    return {
        "schema_version": "company_policy_container_contract.v2",
        "policy_id": "acme_widget_grasp_v3",
        "company_id": "acme_robotics",
        "display_name": "ACME Widget Grasp v3",
        "checkpoint_identity": {
            "repository": "registry.acme.example/models/widget-grasp",
            "revision": "2026.08.1",
            "inventory_digest": "sha256:" + "a" * 64,
        },
        "claim_ceiling": "development_only",
        "rights": {
            "license": "ACME evaluation license",
            "rights_provenance": "acme_msa_appendix_b",
            "rights_evidence_uri": "blueprint-rights://acme/widget-grasp",
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
                "kind": "http_json_v1",
                "protocol_version": "1.0",
                "action_route": "/v1/actions",
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
        "security_profile": SECURITY_PROFILE,
    }


def _admission() -> dict[str, Any]:
    contract = validate_company_policy_container_contract_v2(_contract())
    return {
        "accepted": True,
        "status": "admitted_no_spend",
        "admission_id": "company-policy-admission-" + "1" * 40,
        "admission_digest": "sha256:" + "2" * 64,
        "contract_digest": contract["contract_digest"],
        "tenant_id": "tenant-acme-12345678",
        "run_id": "run-12345678",
        "submission_id": "policy-candidate-12345678",
        "company_id": "acme_robotics",
        "registry_credential_lease_id": "policy-registry-lease-12345678",
        "launch_queued": False,
        "launch_authority_granted": False,
        "provider_mutation_authorized": False,
        "provider_mutation_performed": False,
        "registry_credential_consumed": False,
    }


def _plan(**overrides: Any) -> dict[str, Any]:
    arguments = {
        "admission_receipt": _admission(),
        "contract": _contract(),
        "sandbox_attempt_id": "sandbox-attempt-12345678",
        "pipeline_release_sha": "3" * 40,
        "worker_identity": "blueprint-policy-sandbox-worker-01",
        "runtime_class": "runsc",
        "blueprint_proxy_image": "ghcr.io/blueprint/policy-proxy@sha256:" + "4" * 64,
        "blueprint_proxy_contract_digest": "sha256:" + "5" * 64,
        "seccomp_profile_id": "blueprint-policy-seccomp-v2",
        "seccomp_profile_digest": "sha256:" + "6" * 64,
        "apparmor_profile_id": "blueprint-policy-apparmor-v2",
        "apparmor_profile_digest": "sha256:" + "7" * 64,
        "registry_addresses": ["93.184.216.34"],
        "allowed_registry_hosts": ["registry.acme.example"],
    }
    arguments.update(overrides)
    return build_company_policy_sandbox_plan(**arguments)


def _evidence(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_digest": plan["plan_digest"],
        "policy_image_digest_verified": True,
        "credential_deleted_before_policy_start": True,
        "docker_config_deleted_before_policy_start": True,
        "dedicated_worker_verified": True,
        "isolated_runtime_verified": True,
        "raw_container_logs_retained": False,
        "teardown_plan_armed": True,
        "network_probes": {
            **{
                name: {"status": "denied", "redirect_followed": False}
                for name in DENIAL_PROBES
            },
            **{
                name: {"status": "reachable", "redirect_followed": False}
                for name in REACHABILITY_PROBES
            },
        },
        "synthetic_conformance": {
            "status": "conformant",
            "contract_digest": plan["contract_digest"],
            "real_observation_used": False,
        },
        "completed_phase_order": plan["phase_order"][:-1],
    }


def test_plan_has_no_policy_mounts_egress_logs_or_host_network() -> None:
    plan = _plan()
    policy = plan["commands"]["policy_run"]
    proxy = plan["commands"]["proxy_run"]

    assert "--network" in policy
    assert any(str(item).startswith("container:blueprint-policy-proxy-") for item in policy)
    assert "host" not in policy
    assert "--read-only" in policy
    assert "--cap-drop=ALL" in policy
    assert "--log-driver=none" in policy
    assert "--mount" not in policy
    assert not any(str(item).startswith("--env") for item in policy)
    assert "--network=none" in proxy
    assert proxy.count("--mount") == 1
    assert plan["security"]["policy_mounts"] == []
    assert plan["blueprint_ipc"]["policy_socket_mounted"] is False
    assert plan["launch_authority_granted"] is False
    assert plan["provider_mutation_performed"] is False
    assert plan["security"]["customer_visible_output"] == (
        "aggregate_metrics_and_redacted_status_only"
    )
    assert plan["security"]["raw_actions_customer_visible"] is False
    assert plan["credential_broker_request_binding"]["schema_version"] == (
        "company_policy_registry_credential_claim.v1"
    )
    assert plan["credential_broker_request_binding"]["tenant_id"] == (
        "tenant-acme-12345678"
    )
    assert "--action-schema-b64" in proxy


@pytest.mark.parametrize(
    "runtime", ["runc", "docker", "", "nvidia", "kata-runtime", "firecracker-containerd"]
)
def test_shared_kernel_runtime_is_refused(runtime: str) -> None:
    with pytest.raises(CompanyPolicySandboxV2Error) as excinfo:
        _plan(runtime_class=runtime)
    assert "company_policy_sandbox_shared_kernel_runtime_forbidden" in excinfo.value.blockers


@pytest.mark.parametrize("addresses", [[], ["127.0.0.1"], ["10.0.0.8"], ["169.254.169.254"]])
def test_registry_must_resolve_only_to_global_addresses(addresses: list[str]) -> None:
    with pytest.raises(CompanyPolicySandboxV2Error):
        _plan(registry_addresses=addresses)


def test_exact_allowlisted_registry_is_required() -> None:
    with pytest.raises(CompanyPolicySandboxV2Error) as excinfo:
        _plan(allowed_registry_hosts=["ghcr.io"])
    assert "company_policy_sandbox_registry_origin_not_allowed" in excinfo.value.blockers


def test_measured_no_egress_and_synthetic_conformance_unlock_observation() -> None:
    plan = _plan()
    receipt = evaluate_preobservation_sandbox_qualification(
        plan=plan, evidence=_evidence(plan)
    )
    assert receipt["status"] == "qualified_before_first_observation"
    assert receipt["first_observation_permitted"] is True
    assert receipt["network_denial_probe_count"] == len(DENIAL_PROBES)
    assert receipt["provider_mutation_authorized"] is False


@pytest.mark.parametrize("probe", DENIAL_PROBES)
def test_every_egress_probe_fails_closed(probe: str) -> None:
    plan = _plan()
    evidence = _evidence(plan)
    evidence["network_probes"][probe]["status"] = "reachable"
    with pytest.raises(CompanyPolicySandboxV2Error) as excinfo:
        evaluate_preobservation_sandbox_qualification(plan=plan, evidence=evidence)
    assert any(probe in blocker for blocker in excinfo.value.blockers)


def test_phase_reordering_and_real_observation_in_conformance_refuse() -> None:
    plan = _plan()
    evidence = _evidence(plan)
    evidence["completed_phase_order"] = list(reversed(evidence["completed_phase_order"]))
    evidence["synthetic_conformance"]["real_observation_used"] = True
    with pytest.raises(CompanyPolicySandboxV2Error) as excinfo:
        evaluate_preobservation_sandbox_qualification(plan=plan, evidence=evidence)
    assert "company_policy_sandbox_phase_order_invalid" in excinfo.value.blockers
    assert "company_policy_sandbox_synthetic_conformance_invalid" in excinfo.value.blockers


def test_tampered_admission_and_plan_digests_refuse() -> None:
    admission = _admission()
    admission["contract_digest"] = "sha256:" + "0" * 64
    with pytest.raises(CompanyPolicySandboxV2Error):
        _plan(admission_receipt=admission)

    plan = _plan()
    tampered = copy.deepcopy(plan)
    tampered["security"]["root_filesystem_read_only"] = False
    with pytest.raises(CompanyPolicySandboxV2Error) as excinfo:
        evaluate_preobservation_sandbox_qualification(plan=tampered, evidence=_evidence(plan))
    assert "company_policy_sandbox_plan_digest_mismatch" in excinfo.value.blockers


def test_no_spend_cli_writes_exact_plan_without_launching(tmp_path: Path, capsys) -> None:
    script = Path(__file__).parents[1] / "scripts/build_company_policy_sandbox_plan.py"
    spec = importlib.util.spec_from_file_location("build_company_policy_sandbox_plan", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    admission_path = tmp_path / "admission.json"
    contract_path = tmp_path / "contract.json"
    output_path = tmp_path / "plan.json"
    admission_path.write_text(json.dumps(_admission()), encoding="utf-8")
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")

    assert module.main([
        "--admission-receipt", str(admission_path),
        "--contract", str(contract_path),
        "--sandbox-attempt-id", "sandbox-attempt-12345678",
        "--pipeline-release-sha", "3" * 40,
        "--worker-identity", "blueprint-policy-sandbox-worker-01",
        "--runtime-class", "runsc",
        "--proxy-image", "ghcr.io/blueprint/policy-proxy@sha256:" + "4" * 64,
        "--proxy-contract-digest", "sha256:" + "5" * 64,
        "--seccomp-profile-id", "blueprint-policy-seccomp-v2",
        "--seccomp-profile-digest", "sha256:" + "6" * 64,
        "--apparmor-profile-id", "blueprint-policy-apparmor-v2",
        "--apparmor-profile-digest", "sha256:" + "7" * 64,
        "--registry-address", "93.184.216.34",
        "--allowed-registry-host", "registry.acme.example",
        "--output", str(output_path),
        "--ack", "no-spend-plan-only",
    ]) == 0
    plan = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)
    assert summary["plan_digest"] == plan["plan_digest"]
    assert summary["provider_mutation_performed"] is False
    assert plan["status"] == "planned_no_spend"
