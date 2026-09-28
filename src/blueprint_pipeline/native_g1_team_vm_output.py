"""Reopen actual host/child cleanup and guest score/media for one G1 VM run."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Mapping

from .native_g1_team_paid_output import _receipt as _paid_receipt, verify_g1_team_paid_output
from .native_g1_team_archive_gpu import (
    PROBE_FILENAME as ARCHIVE_GPU_PROBE_FILENAME, validate_archive_gpu_binding, validate_archive_gpu_probe,
)
from .native_g1_team_policy_relay import RelayBinding, validate_relay_conformance, validate_relay_close
from .native_g1_team_provider_bundle import RESULT_FILENAME, RESULT_SCHEMA
from .native_g1_team_runtime_session import CONFORMANCE_FILENAME, SESSION_SCHEMA
from .native_g1_team_vm_host import (
    HOST_FILENAME, HOST_SCHEMA, PREFLIGHT_FILENAME, PREFLIGHT_SCHEMA,
    SIMULATOR_FILENAME, SIMULATOR_SCHEMA, RELAY_FILENAME,
)
from .native_g1_team_worker_supervisor import (
    EXIT_FILENAME, EXIT_SCHEMA, RESULT_FILENAME as SUPERVISOR_FILENAME,
    RESULT_SCHEMA as SUPERVISOR_SCHEMA,
)
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE

_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _receipt(path: Path, *, digest_field: str) -> dict[str, Any]:
    if path.resolve() != path or (path.is_file() and path.stat().st_size > 8 * 1024 * 1024):
        raise ValueError("g1_vm_output_receipt_path_invalid")
    return _paid_receipt(path, digest_field=digest_field)


def verify_g1_team_vm_evidence(
    *, output_dir: Path, execution_packet: Mapping[str, Any],
    scene_plan_digest: str, scene_packet_receipt_digest: str,
) -> dict[str, Any]:
    """Compute evidence only after both close owners are terminal."""
    root = Path(output_dir)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("g1_vm_output_root_invalid")
    profile = execution_packet["request"]["policy_profile"]
    binding = RelayBinding(execution_packet["packet_digest"], profile["profile_digest"],
                           execution_packet["trusted_setup"]["setup_digest"], profile["delivery"]["mode"])
    guest = _receipt(root / RESULT_FILENAME, digest_field="result_digest")
    if (guest.get("schema_version") != RESULT_SCHEMA or guest.get("status") != "completed_development_only"
            or guest.get("execution_packet_digest") != binding.packet_digest
            or guest.get("worker_output_relative_path") != "selected-worker/worker"
            or guest.get("candidate_policy_queried") is not True
            or guest.get("provider_teardown_verified") is not False
            or guest.get("official_billing_reconciled") is not False
            or guest.get("public_redistribution_authorized") is not False):
        raise ValueError("g1_vm_guest_result_invalid")
    worker_root = root / "selected-worker/worker"
    if worker_root.resolve() != worker_root:
        raise ValueError("g1_vm_output_receipt_path_invalid")
    verified = verify_g1_team_paid_output(
        output_dir=worker_root, execution_packet=execution_packet,
        scene_plan_digest=scene_plan_digest, scene_packet_receipt_digest=scene_packet_receipt_digest,
    )
    if guest.get("verified_output") != verified:
        raise ValueError("g1_vm_guest_verification_changed")
    supervisor = _receipt(root / "selected-worker" / SUPERVISOR_FILENAME, digest_field="result_digest")
    worker_exit = _receipt(root / "selected-worker/private_diagnostics" / EXIT_FILENAME, digest_field="receipt_digest")
    if (supervisor.get("schema_version") != SUPERVISOR_SCHEMA
            or supervisor.get("status") != "completed_development_only"
            or supervisor.get("execution_packet_digest") != binding.packet_digest
            or supervisor.get("verified_output") != verified
            or supervisor.get("child_exit_receipt_digest") != worker_exit["receipt_digest"]
            or guest.get("supervised_result_digest") != supervisor["result_digest"]
            or worker_exit.get("schema_version") != EXIT_SCHEMA or worker_exit.get("status") != "exited"
            or type(worker_exit.get("returncode")) is not int or worker_exit["returncode"] != 0
            or type(worker_exit.get("child_pid")) is not int or worker_exit["child_pid"] <= 0
            or worker_exit.get("process_group_kill_requested") is not False
            or worker_exit.get("error_type") is not None):
        raise ValueError("g1_vm_guest_worker_exit_invalid")
    policy_root = root / "policy-host"
    conformance = _receipt(policy_root / "runtime" / CONFORMANCE_FILENAME, digest_field="receipt_digest")
    session = _receipt(policy_root / "runtime" / (SESSION_SCHEMA + ".json"), digest_field="receipt_digest")
    validate_relay_conformance(conformance, binding)
    validate_relay_close(session, binding, conformance_digest=conformance["receipt_digest"],
                         linked_episode_digest=verified["episode_result_digest"])
    guest_runtime = worker_root / "episode/runtime"
    if (conformance != _receipt(guest_runtime / CONFORMANCE_FILENAME, digest_field="receipt_digest")
            or session != _receipt(guest_runtime / (SESSION_SCHEMA + ".json"), digest_field="receipt_digest")):
        raise ValueError("g1_vm_host_guest_session_mismatch")
    relay = _receipt(policy_root / RELAY_FILENAME, digest_field="receipt_digest")
    if (relay.get("schema_version") != "native_g1_private_policy_relay_session.v1"
            or relay.get("status") != "policy_session_closed" or relay.get("binding") != asdict(binding)
            or relay.get("policy_session_closed") is not True or relay.get("failure_type") is not None
            or relay.get("synthetic_conformance_digest") != conformance["receipt_digest"]
            or relay.get("policy_session_close_digest") != session["receipt_digest"]
            or type(relay.get("inference_query_count")) is not int
            or relay["inference_query_count"] != verified["policy_query_count"]
            or relay.get("provider_teardown_verified") is not False):
        raise ValueError("g1_vm_policy_relay_evidence_invalid")
    child_schema = ("native_g1_team_container_teardown.v1" if binding.delivery_mode == "container"
                    else "native_g1_team_artifact_teardown.v1")
    child = _receipt(policy_root / "runtime" / (child_schema + ".json"), digest_field="receipt_digest")
    if (child.get("schema_version") != child_schema or child["receipt_digest"] != session["child_teardown_digest"]
            or child.get("profile_digest") != binding.profile_digest
            or type(child.get("process_exit_code")) is not int or child.get("process_close_error") is not None
            or child.get("provider_teardown_verified") is not False):
        raise ValueError("g1_vm_policy_child_evidence_invalid")
    preflight = _receipt(root / PREFLIGHT_FILENAME, digest_field="receipt_digest")
    if (preflight.get("schema_version") != PREFLIGHT_SCHEMA or preflight.get("status") != "host_capabilities_observed"
            or preflight.get("execution_packet_digest") != binding.packet_digest
            or preflight.get("delivery_mode") != binding.delivery_mode
            or preflight.get("simulator_image_ref") != NATIVE_TASK_ARENA_IMAGE
            or preflight.get("python_abi") != "cp312" or preflight.get("numpy_version") != "2.3.1"
            or preflight.get("rfc8785_version") != "0.1.4"
            or preflight.get("claim_ceiling") != "development_only"
            or preflight.get("provider_teardown_verified") is not False
            or preflight.get("guest_gpu_inference_verified") is not False
            or preflight.get("archive_gpu_device_exposure_verified") is not False
            or not isinstance(preflight.get("simulator_local_image_id"), str)
            or not _SHA.fullmatch(preflight["simulator_local_image_id"])):
        raise ValueError("g1_vm_host_preflight_evidence_invalid")
    if binding.delivery_mode == "container":
        if (child.get("status") != "container_removed" or child.get("container_absent_verified") is not True
                or child.get("image_ref") != profile["delivery"]["image_ref"]
                or not isinstance(child.get("local_image_id"), str)
                or not _SHA.fullmatch(child["local_image_id"])
                or child["local_image_id"] != preflight.get("policy_local_image_id")):
            raise ValueError("g1_vm_policy_container_evidence_invalid")
    else:
        if (child.get("status") != "process_exited"
                or child.get("artifact_sha256") != profile["delivery"]["artifact_sha256"]):
            raise ValueError("g1_vm_policy_archive_evidence_invalid")
        gpu = preflight.get("archive_gpu_device_binding")
        validate_archive_gpu_binding(gpu, profile_digest=binding.profile_digest, execution_packet_digest=binding.packet_digest)
        probe = _receipt(policy_root / "runtime" / ARCHIVE_GPU_PROBE_FILENAME, digest_field="receipt_digest")
        validate_archive_gpu_probe(probe, binding=gpu)
        if (child.get("gpu_binding_digest") != gpu["receipt_digest"]
                or child.get("gpu_namespace_probe_digest") != probe["receipt_digest"]):
            raise ValueError("g1_vm_archive_gpu_evidence_invalid")
    simulator_root = root / "vm-simulator"
    simulator = _receipt(simulator_root / SIMULATOR_FILENAME, digest_field="receipt_digest")
    exited = _receipt(simulator_root / "private_diagnostics" / EXIT_FILENAME, digest_field="receipt_digest")
    if (simulator.get("schema_version") != SIMULATOR_SCHEMA or simulator.get("status") != "container_removed"
            or simulator.get("container_absent_verified") is not True
            or simulator.get("image_ref") != NATIVE_TASK_ARENA_IMAGE
            or simulator.get("local_image_id") != preflight.get("simulator_local_image_id")
            or simulator.get("error_type") is not None
            or simulator.get("provider_teardown_verified") is not False
            or exited.get("schema_version") != EXIT_SCHEMA or exited.get("status") != "exited"
            or type(exited.get("returncode")) is not int or exited["returncode"] != 0
            or type(exited.get("child_pid")) is not int or exited["child_pid"] <= 0
            or exited.get("process_group_kill_requested") is not False or exited.get("error_type") is not None):
        raise ValueError("g1_vm_simulator_close_evidence_invalid")
    return {
        "schema_version": "native_g1_team_vm_output_verification.v1", "status": "verified_development_only",
        "execution_packet_digest": binding.packet_digest,
        "preflight_digest": preflight["receipt_digest"], "relay_digest": relay["receipt_digest"],
        "policy_session_close_digest": session["receipt_digest"], "policy_child_teardown_digest": child["receipt_digest"],
        "simulator_teardown_digest": simulator["receipt_digest"], "child_exit_receipt_digest": exited["receipt_digest"],
        "verified_output": verified, "provider_teardown_verified": False,
        "archive_gpu_namespace_probe_digest": probe["receipt_digest"] if binding.delivery_mode == "noncontainer_artifact" else None,
        "archive_gpu_device_memory_access_verified": binding.delivery_mode == "noncontainer_artifact",
        "guest_gpu_inference_verified": False,
        "official_billing_reconciled": False, "public_redistribution_authorized": False,
        "claim_ceiling": "development_only",
    }


def verify_g1_team_vm_host_output(**arguments: Any) -> dict[str, Any]:
    result = _receipt(arguments["output_dir"] / HOST_FILENAME, digest_field="result_digest")
    verified = verify_g1_team_vm_evidence(**arguments)
    if (result.get("schema_version") != HOST_SCHEMA or result.get("status") != "completed_development_only"
            or result.get("execution_packet_digest") != arguments["execution_packet"]["packet_digest"]
            or result.get("verified_output") != verified or result.get("blocker_type") is not None
            or result.get("policy_session_closed") is not True
            or any(result.get(field) != verified[field] for field in (
                "preflight_digest", "relay_digest", "simulator_teardown_digest", "child_exit_receipt_digest"))
            or result.get("provider_teardown_verified") is not False
            or result.get("official_billing_reconciled") is not False
            or result.get("public_redistribution_authorized") is not False):
        raise ValueError("g1_vm_host_result_invalid")
    return verified
