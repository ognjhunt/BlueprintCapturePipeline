from __future__ import annotations

import copy
import hashlib
import hmac
import importlib.util
import json
import subprocess
from collections.abc import Callable, Mapping
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
from blueprint_pipeline.company_policy_sandbox_executor import (
    CompanyPolicySandboxExecutorError,
    execute_company_policy_sandbox_preobservation,
    verify_executor_receipt,
)
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


_ATTESTATION_KEY = b"company-policy-test-attestation-key-32-bytes"


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
        "seccomp_profile_path": "/etc/blueprint/company-policy/seccomp-v2.json",
        "seccomp_profile_digest": "sha256:" + "6" * 64,
        "apparmor_profile_id": "blueprint-policy-apparmor-v2",
        "apparmor_profile_source_path": "/etc/blueprint/company-policy/apparmor-v2.profile",
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
        "worker_boot_receipt_digest": "sha256:" + "8" * 64,
        "scene_bytes_present_during_pull": False,
        "observation_bytes_present_during_pull": False,
        "registry_redirect_behavior": "not_measured_by_docker_engine",
        "image_pull_receipt_digest": "sha256:" + "9" * 64,
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


def _executor_receipt(plan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "schema_version": "company_policy_sandbox_executor_receipt.v1",
        "source": "trusted_company_policy_sandbox_executor",
        "status": "qualified_before_first_observation",
        "plan_digest": plan["plan_digest"],
        "evidence": evidence,
        "attestation_key_id": "company-policy-test-key",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = cross_runtime_canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt["attestation_hmac_sha256"] = hmac.new(
        _ATTESTATION_KEY,
        receipt["receipt_digest"].encode(),
        hashlib.sha256,
    ).hexdigest()
    return receipt


def _worker_boot_receipt(plan: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "schema_version": "company_policy_worker_boot_receipt.v1",
        "source": "trusted_company_policy_worker_bootstrap",
        "status": "dedicated_ephemeral_worker_ready",
        "worker_identity": plan["worker_identity"],
        "pipeline_release_sha": plan["pipeline_release_sha"],
        "sandbox_attempt_id": plan["sandbox_attempt_id"],
        "dedicated_ephemeral_worker": True,
        "scene_bytes_present": False,
        "observation_bytes_present": False,
        "mounted_customer_input_paths": [],
        "attestation_key_id": "company-policy-test-key",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = cross_runtime_canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt["attestation_hmac_sha256"] = hmac.new(
        _ATTESTATION_KEY,
        receipt["receipt_digest"].encode(),
        hashlib.sha256,
    ).hexdigest()
    return receipt


class _Broker:
    def __init__(self) -> None:
        self.claims: list[tuple[str, Mapping[str, Any]]] = []
        self.acknowledgements: list[tuple[str, Mapping[str, Any]]] = []

    def claim(self, *, lease_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.claims.append((lease_id, dict(body)))
        return {
            "ok": True,
            "credential": {
                "registry_server": "registry.acme.example",
                "username": "robot-team",
                "secret": "short-lived-secret",
            },
            "delivery_receipt": {"delivery_id": "policy-registry-delivery-12345678"},
        }

    def acknowledge(self, *, lease_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.acknowledgements.append((lease_id, dict(body)))
        return {
            "ok": True,
            "lease_receipt": {
                "status": "consumed",
                "ciphertext_deleted": True,
                "delivery_id": "policy-registry-delivery-12345678",
            },
        }


class _Runner:
    def __init__(
        self,
        image: str,
        *,
        reachable_probe: str | None = None,
        retain_image: bool = False,
        timeout_on_smoke: bool = False,
    ):
        self.image = image
        self.reachable_probe = reachable_probe
        self.retain_image = retain_image
        self.timeout_on_smoke = timeout_on_smoke
        self.commands: list[list[str]] = []
        self.probe_index = 0

    def run(self, argv, *, timeout_seconds: int):
        command = [str(item) for item in argv]
        self.commands.append(command)
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, '{"runsc":{}}', "")
        if command[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(command, 0, "27.3.1\n", "")
        if command[:4] == ["docker", "image", "inspect", "--format"]:
            return subprocess.CompletedProcess(command, 0, json.dumps([command[-1]]), "")
        if "python" in command and "-c" in command and any(
            "runsc-smoke-ok" in item for item in command
        ):
            if self.timeout_on_smoke:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            return subprocess.CompletedProcess(command, 0, "runsc-smoke-ok\n", "")
        if "probe" in command:
            probe = DENIAL_PROBES[self.probe_index]
            self.probe_index += 1
            status = "reachable" if probe == self.reachable_probe else "denied"
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"status": status, "redirect_followed": False}),
                "",
            )
        if command[:3] == ["docker", "container", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", "No such container")
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0 if self.retain_image else 1,
                "[]" if self.retain_image else "",
                "" if self.retain_image else "No such image",
            )
        return subprocess.CompletedProcess(command, 0, "container-id\n", "")


def _executor_plan(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    seccomp = tmp_path / "seccomp.json"
    apparmor = tmp_path / "apparmor.profile"
    seccomp.write_text('{"defaultAction":"SCMP_ACT_ERRNO"}\n', encoding="utf-8")
    apparmor.write_text("profile blueprint-policy-apparmor-v2 {}\n", encoding="utf-8")
    plan = _plan()
    old_ipc = plan["cleanup_targets"]["ipc_directory"]
    old_seccomp = plan["security"]["seccomp_profile_path"]
    plan["security"]["seccomp_profile_path"] = str(seccomp)
    plan["security"]["seccomp_profile_digest"] = (
        "sha256:" + hashlib.sha256(seccomp.read_bytes()).hexdigest()
    )
    plan["security"]["apparmor_profile_source_path"] = str(apparmor)
    plan["security"]["apparmor_profile_digest"] = (
        "sha256:" + hashlib.sha256(apparmor.read_bytes()).hexdigest()
    )
    plan["cleanup_targets"]["registry_config_directory"] = str(tmp_path / "docker-config")
    plan["cleanup_targets"]["ipc_directory"] = str(tmp_path / "ipc")
    plan["commands"]["image_pull"][2] = str(tmp_path / "docker-config")
    plan["commands"]["proxy_run"] = [
        item.replace(old_ipc, str(tmp_path / "ipc")).replace(old_seccomp, str(seccomp))
        for item in plan["commands"]["proxy_run"]
    ]
    for command_name in ("policy_run", "runtime_smoke"):
        plan["commands"][command_name] = [
            item.replace(old_seccomp, str(seccomp))
            for item in plan["commands"][command_name]
        ]
    plan["plan_digest"] = cross_runtime_canonical_digest(plan, digest_field="plan_digest")
    apparmor_loaded = tmp_path / "apparmor-loaded"
    apparmor_loaded.write_text(
        "blueprint-policy-apparmor-v2 (enforce)\n", encoding="utf-8"
    )
    return plan, apparmor_loaded, seccomp


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
        plan=plan,
        executor_receipt=_executor_receipt(plan, _evidence(plan)),
        attestation_key=_ATTESTATION_KEY,
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
        evaluate_preobservation_sandbox_qualification(
            plan=plan,
            executor_receipt=_executor_receipt(plan, evidence),
            attestation_key=_ATTESTATION_KEY,
        )
    assert any(probe in blocker for blocker in excinfo.value.blockers)


def test_phase_reordering_and_real_observation_in_conformance_refuse() -> None:
    plan = _plan()
    evidence = _evidence(plan)
    evidence["completed_phase_order"] = list(reversed(evidence["completed_phase_order"]))
    evidence["synthetic_conformance"]["real_observation_used"] = True
    with pytest.raises(CompanyPolicySandboxV2Error) as excinfo:
        evaluate_preobservation_sandbox_qualification(
            plan=plan,
            executor_receipt=_executor_receipt(plan, evidence),
            attestation_key=_ATTESTATION_KEY,
        )
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
        evaluate_preobservation_sandbox_qualification(
            plan=tampered,
            executor_receipt=_executor_receipt(plan, _evidence(plan)),
            attestation_key=_ATTESTATION_KEY,
        )
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
        "--seccomp-profile-path", "/etc/blueprint/company-policy/seccomp-v2.json",
        "--seccomp-profile-digest", "sha256:" + "6" * 64,
        "--apparmor-profile-id", "blueprint-policy-apparmor-v2",
        "--apparmor-profile-source-path", "/etc/blueprint/company-policy/apparmor-v2.profile",
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


def test_trusted_executor_claims_pulls_probes_attests_and_cleans(tmp_path: Path) -> None:
    plan, apparmor_loaded, _seccomp = _executor_plan(tmp_path)
    broker = _Broker()
    runner = _Runner(plan["images"]["policy"])
    result = execute_company_policy_sandbox_preobservation(
        plan=plan,
        contract=_contract(),
        broker=broker,
        runner=runner,
        attestation_key=_ATTESTATION_KEY,
        attestation_key_id="company-policy-test-key",
        worker_boot_receipt=_worker_boot_receipt(plan),
        output_path=tmp_path / "executor-result.json",
        proxy_request=lambda _path, _payload, _timeout: {
            "actions": [[0.0, 1.0] for _ in range(15)]
        },
        socket_ready=lambda _path, _timeout: True,
        apparmor_profiles_path=apparmor_loaded,
        allowed_runtime_root=tmp_path,
    )

    assert result["status"] == "qualified_dry_run_no_real_observation"
    assert result["qualification_receipt"]["first_observation_permitted"] is True
    assert result["terminal_receipt"]["cleanup_complete"] is True
    assert result["terminal_receipt"]["customer_policy_image_removed"] is True
    assert result["executor_receipt"]["evidence"]["registry_redirect_behavior"] == (
        "not_measured_by_docker_engine"
    )
    assert result["executor_receipt"]["evidence"]["scene_bytes_present_during_pull"] is False
    assert (
        result["executor_receipt"]["evidence"]["observation_bytes_present_during_pull"]
        is False
    )
    assert len(broker.claims) == 1
    assert len(broker.acknowledgements) == 1
    claim_body = broker.claims[0][1]
    assert claim_body["sandbox_plan_digest"] == plan["plan_digest"]
    assert claim_body["tenant_id"] == "tenant-acme-12345678"
    assert broker.acknowledgements[0][1]["pulled_image_digest"] == (
        "sha256:" + "c" * 64
    )
    verify_executor_receipt(
        result["executor_receipt"],
        key=_ATTESTATION_KEY,
        expected_plan_digest=plan["plan_digest"],
    )
    assert not Path(plan["cleanup_targets"]["registry_config_directory"]).exists()
    assert not Path(plan["cleanup_targets"]["ipc_directory"]).exists()


def test_trusted_executor_blocks_reachable_egress_and_still_cleans(tmp_path: Path) -> None:
    plan, apparmor_loaded, _seccomp = _executor_plan(tmp_path)
    runner = _Runner(plan["images"]["policy"], reachable_probe="public_ipv4")
    result = execute_company_policy_sandbox_preobservation(
        plan=plan,
        contract=_contract(),
        broker=_Broker(),
        runner=runner,
        attestation_key=_ATTESTATION_KEY,
        attestation_key_id="company-policy-test-key",
        worker_boot_receipt=_worker_boot_receipt(plan),
        output_path=tmp_path / "executor-result.json",
        proxy_request=lambda *_args: {"actions": [[0.0, 1.0] for _ in range(15)]},
        socket_ready=lambda *_args: True,
        apparmor_profiles_path=apparmor_loaded,
        allowed_runtime_root=tmp_path,
    )

    assert result["status"] == "blocked_before_first_observation"
    assert any("egress_reachable:public_ipv4" in item for item in result["blockers"])
    assert result["terminal_receipt"]["cleanup_complete"] is True
    assert result["real_observation_sent"] is False


def test_trusted_executor_surfaces_image_teardown_readback_failure(tmp_path: Path) -> None:
    plan, apparmor_loaded, _seccomp = _executor_plan(tmp_path)
    runner = _Runner(plan["images"]["policy"], retain_image=True)
    result = execute_company_policy_sandbox_preobservation(
        plan=plan,
        contract=_contract(),
        broker=_Broker(),
        runner=runner,
        attestation_key=_ATTESTATION_KEY,
        attestation_key_id="company-policy-test-key",
        worker_boot_receipt=_worker_boot_receipt(plan),
        output_path=tmp_path / "executor-result.json",
        proxy_request=lambda *_args: {"actions": [[0.0, 1.0] for _ in range(15)]},
        socket_ready=lambda *_args: True,
        apparmor_profiles_path=apparmor_loaded,
        allowed_runtime_root=tmp_path,
    )

    assert result["status"] == "blocked_teardown_incomplete"
    assert "company_policy_sandbox_customer_image_retained" in result["blockers"]
    assert result["terminal_receipt"]["cleanup_complete"] is False
    assert result["terminal_receipt"]["customer_policy_image_removed"] is False


def test_runtime_timeout_still_seals_terminal_cleanup_receipt(tmp_path: Path) -> None:
    plan, apparmor_loaded, _seccomp = _executor_plan(tmp_path)
    runner = _Runner(plan["images"]["policy"], timeout_on_smoke=True)
    result = execute_company_policy_sandbox_preobservation(
        plan=plan,
        contract=_contract(),
        broker=_Broker(),
        runner=runner,
        attestation_key=_ATTESTATION_KEY,
        attestation_key_id="company-policy-test-key",
        worker_boot_receipt=_worker_boot_receipt(plan),
        output_path=tmp_path / "executor-result.json",
        proxy_request=lambda *_args: {"actions": [[0.0, 1.0] for _ in range(15)]},
        socket_ready=lambda *_args: True,
        apparmor_profiles_path=apparmor_loaded,
        allowed_runtime_root=tmp_path,
    )

    assert result["status"] == "blocked_before_first_observation"
    assert result["terminal_receipt"]["cleanup_complete"] is True
    assert result["real_observation_sent"] is False
    assert (tmp_path / "executor-result.json").exists()


def test_signed_worker_boot_receipt_with_scene_bytes_refuses_before_mutation(
    tmp_path: Path,
) -> None:
    plan, apparmor_loaded, _seccomp = _executor_plan(tmp_path)
    receipt = _worker_boot_receipt(plan)
    receipt["scene_bytes_present"] = True
    receipt["receipt_digest"] = cross_runtime_canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt["attestation_hmac_sha256"] = hmac.new(
        _ATTESTATION_KEY,
        receipt["receipt_digest"].encode(),
        hashlib.sha256,
    ).hexdigest()
    runner = _Runner(plan["images"]["policy"])

    with pytest.raises(CompanyPolicySandboxExecutorError) as excinfo:
        execute_company_policy_sandbox_preobservation(
            plan=plan,
            contract=_contract(),
            broker=_Broker(),
            runner=runner,
            attestation_key=_ATTESTATION_KEY,
            attestation_key_id="company-policy-test-key",
            worker_boot_receipt=receipt,
            output_path=tmp_path / "executor-result.json",
            apparmor_profiles_path=apparmor_loaded,
            allowed_runtime_root=tmp_path,
        )

    assert "worker_boot_receipt_attestation_invalid" in str(excinfo.value)
    assert runner.commands == []


@pytest.mark.parametrize(
    ("command_name", "mutation"),
    [
        ("cleanup", lambda plan: plan["commands"]["cleanup"].append(
            ["docker", "system", "prune", "--force"]
        )),
        ("policy_command", lambda plan: plan["commands"]["policy_run"].insert(
            plan["commands"]["policy_run"].index(plan["images"]["policy"]),
            "--privileged",
        )),
        ("network_probe", lambda plan: plan["commands"]["network_probes"][
            "public_ipv4"
        ].__setitem__(-3, "127.0.0.1")),
    ],
)
def test_tampered_command_plan_refuses_before_mutation(
    tmp_path: Path, command_name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    plan, apparmor_loaded, _seccomp = _executor_plan(tmp_path)
    mutation(plan)
    plan["plan_digest"] = cross_runtime_canonical_digest(plan, digest_field="plan_digest")
    runner = _Runner(plan["images"]["policy"])

    with pytest.raises(CompanyPolicySandboxExecutorError) as excinfo:
        execute_company_policy_sandbox_preobservation(
            plan=plan,
            contract=_contract(),
            broker=_Broker(),
            runner=runner,
            attestation_key=_ATTESTATION_KEY,
            attestation_key_id="company-policy-test-key",
            worker_boot_receipt=_worker_boot_receipt(plan),
            output_path=tmp_path / "executor-result.json",
            apparmor_profiles_path=apparmor_loaded,
            allowed_runtime_root=tmp_path,
        )

    assert command_name in str(excinfo.value)
    assert runner.commands == []


def test_forged_executor_receipt_cannot_unlock_first_observation() -> None:
    plan = _plan()
    receipt = _executor_receipt(plan, _evidence(plan))
    receipt["evidence"]["network_probes"]["public_ipv4"]["status"] = "reachable"
    with pytest.raises(CompanyPolicySandboxV2Error) as excinfo:
        evaluate_preobservation_sandbox_qualification(
            plan=plan,
            executor_receipt=receipt,
            attestation_key=_ATTESTATION_KEY,
        )
    assert "company_policy_sandbox_executor_attestation_invalid" in excinfo.value.blockers
