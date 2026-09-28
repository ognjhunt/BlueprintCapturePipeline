"""Own simulator and separately isolated team policy on one admitted VM.

This entry point does not allocate compute or grant image, model, site or spend
authority. The canonical controller must admit the VM/bootstrap and retain its
independent watchdog. Cloud teardown and posted billing remain external gates.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest
from .native_g1_team_container_runtime import _verified_local_image
from .native_g1_team_policy_relay import G1PolicyRelayServer, RelayBinding
from .native_g1_team_provider_bundle import ENTRYPOINT, RESULT_FILENAME
from .native_g1_team_provider_runtime import (
    POLICY_RELAY_FILE_ENV, verify_g1_team_sealed_inputs,
)
from .native_g1_team_relay_runtime_session import write_g1_team_relay_config
from .native_g1_team_runtime_session import open_g1_team_runtime_session
from .native_g1_team_worker_supervisor import (
    MAX_TIMEOUT_SECONDS, run_g1_team_worker_process,
)
from .native_task_isaaclab_launch import (
    NATIVE_TASK_ARENA_IMAGE, NATIVE_TASK_ARENA_MINIMUM_DRIVER_VERSION,
)

HOST_SCHEMA = "native_g1_team_vm_host_result.v1"
HOST_FILENAME = HOST_SCHEMA + ".json"
PREFLIGHT_SCHEMA = "native_g1_team_vm_host_preflight.v1"
PREFLIGHT_FILENAME = PREFLIGHT_SCHEMA + ".json"
SIMULATOR_SCHEMA = "native_g1_team_simulator_container_teardown.v1"
SIMULATOR_FILENAME = SIMULATOR_SCHEMA + ".json"
RELAY_FILENAME = "native_g1_private_policy_relay_session.v1.json"
_NAME = re.compile(r"blueprint-g1-simulator-[0-9a-f]{32}\Z")
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_JOIN_SECONDS = 95  # Existing JSONL exchange <=30s, lease close <=50s.


def _directory(path: Path) -> None:
    if (not isinstance(path, Path) or not path.is_absolute() or path.resolve() != path
            or not path.is_dir() or any(c in str(path) for c in ",\n\r\x00")):
        raise ValueError("g1_vm_host_directory_invalid")


def _write(path: Path, value: dict[str, Any], *, field: str = "receipt_digest") -> dict[str, Any]:
    value[field] = canonical_digest(value, digest_field=field)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def simulator_container_command(
    *, runtime_root: Path, output_dir: Path, relay_directory: Path, container_name: str,
) -> list[str]:
    for path in (runtime_root, output_dir, relay_directory):
        _directory(path)
    if (not isinstance(container_name, str) or not _NAME.fullmatch(container_name)
            or output_dir != runtime_root.parent / "runtime_output"):
        raise ValueError("g1_vm_simulator_configuration_invalid")
    provisioning = runtime_root / "provisioned_runtime_sources"
    command = ["docker", "run", "--pull", "never", "--name", container_name,
               "--init", "--user", "0:0", "--network", "bridge", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--gpus", "device=0",
               "--pids-limit", "1024", "--shm-size", "16g",
               "--env", "ACCEPT_EULA=Y", "--env", "NVIDIA_DRIVER_CAPABILITIES=all",
               "--env", "BLUEPRINT_ADP_ARENA_OUTPUT_DIR=" + str(output_dir),
               "--env", POLICY_RELAY_FILE_ENV + "=" + str(relay_directory / "private.json"),
               "--env", "PYTHONPATH=" + str(runtime_root)]
    for source, readonly in ((runtime_root.parent, True), (output_dir, False),
                             (provisioning, False), (relay_directory, True)):
        command += ["--mount", f"type=bind,source={source},target={source}" + (",readonly" if readonly else "")]
    command += ["--entrypoint", "/bin/bash", NATIVE_TASK_ARENA_IMAGE,
                str(runtime_root.parent / ENTRYPOINT)]
    return command


def close_simulator_container(
    *, container_name: str, image_id: str, output_dir: Path,
) -> dict[str, Any]:
    if (not isinstance(container_name, str) or not _NAME.fullmatch(container_name)
            or not isinstance(image_id, str) or not _SHA.fullmatch(image_id)):
        raise ValueError("g1_vm_simulator_identity_invalid")
    absent, removal_code, error = False, None, None
    try:
        removed = subprocess.run(["docker", "rm", "--force", container_name],
                                 capture_output=True, text=True, check=False, timeout=20)
        removal_code = removed.returncode
        inspected = subprocess.run(["docker", "container", "inspect", container_name],
                                   capture_output=True, text=True, check=False, timeout=20)
        absent = inspected.returncode != 0 and any(
            message in inspected.stderr for message in ("No such object:", "No such container:")
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        error = type(exc).__name__
    return _write(output_dir / SIMULATOR_FILENAME, {
        "schema_version": SIMULATOR_SCHEMA,
        "status": "container_removed" if absent else "teardown_blocked",
        "container_name": container_name, "image_ref": NATIVE_TASK_ARENA_IMAGE,
        "local_image_id": image_id, "docker_remove_exit_code": removal_code,
        "container_absent_verified": absent, "error_type": error,
        "provider_teardown_verified": False, "claim_ceiling": "development_only",
    })


def preflight_g1_vm_host(packet: dict[str, Any]) -> dict[str, Any]:
    """Observe the already provisioned host; never install or pull an image."""
    if (platform.system() != "Linux" or platform.machine() != "x86_64"
            or os.geteuid() != 0 or sys.version_info[:2] != (3, 12)
            or version("numpy") != "2.3.1" or version("rfc8785") != "0.1.4"):
        raise ValueError("g1_vm_host_abi_not_qualified")
    info = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                          capture_output=True, text=True, check=False, timeout=20)
    runtimes = json.loads(info.stdout)
    if info.returncode != 0 or not isinstance(runtimes, dict) or "nvidia" not in runtimes:
        raise ValueError("g1_vm_host_nvidia_runtime_unavailable")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False, timeout=20)
    floor = tuple(int(part) for part in NATIVE_TASK_ARENA_MINIMUM_DRIVER_VERSION.split("."))
    drivers = [row.strip().split(",") for row in gpu.stdout.splitlines()]
    if (gpu.returncode != 0 or not any(len(row) == 2 and row[0].strip() == "0"
            and re.fullmatch(r"\d+\.\d+\.\d+", row[1].strip())
            and tuple(int(part) for part in row[1].strip().split(".")) >= floor for row in drivers)):
        raise ValueError("g1_vm_host_gpu_driver_unavailable")
    profile = packet["request"]["policy_profile"]
    mode = profile["delivery"]["mode"]
    policy_id = None
    if mode == "container":
        policy_id = _verified_local_image(profile["delivery"]["image_ref"])
    elif mode == "noncontainer_artifact":
        result = subprocess.run(["bwrap", "--version"], capture_output=True, check=False, timeout=10)
        if result.returncode != 0:
            raise ValueError("g1_vm_host_archive_sandbox_unavailable")
    else:
        raise ValueError("g1_vm_host_delivery_mode_invalid")
    return {
        "schema_version": PREFLIGHT_SCHEMA, "status": "host_capabilities_observed",
        "execution_packet_digest": packet["packet_digest"], "delivery_mode": mode,
        "simulator_image_ref": NATIVE_TASK_ARENA_IMAGE,
        "simulator_local_image_id": _verified_local_image(NATIVE_TASK_ARENA_IMAGE),
        "policy_local_image_id": policy_id, "python_abi": "cp312",
        "numpy_version": "2.3.1", "rfc8785_version": "0.1.4",
        "guest_gpu_inference_verified": False, "archive_gpu_device_exposure_verified": False,
        "provider_teardown_verified": False, "claim_ceiling": "development_only",
    }


def run_g1_team_vm_host(
    *, runtime_root: Path, output_dir: Path, timeout_seconds: float = MAX_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run once; retain failure and independently verify both actual close paths."""
    _directory(runtime_root)
    if (not isinstance(output_dir, Path) or output_dir != runtime_root.parent / "runtime_output"
            or output_dir.resolve() != output_dir or output_dir.is_symlink()
            or (output_dir / HOST_FILENAME).exists() or (output_dir / RESULT_FILENAME).exists()
            or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS):
        raise ValueError("g1_vm_host_inputs_invalid")
    output_dir.mkdir(exist_ok=True, mode=0o700)
    policy_root, simulator_root = output_dir / "policy-host", output_dir / "vm-simulator"
    policy_root.mkdir(mode=0o700)
    simulator_root.mkdir(mode=0o700)
    state: dict[str, Any] = {
        "schema_version": HOST_SCHEMA, "status": "blocked", "stage_reached": "sealed_inputs",
        "execution_packet_digest": None, "preflight_digest": None,
        "relay_digest": None, "simulator_teardown_digest": None, "policy_session_closed": False,
        "child_exit_receipt_digest": None, "verified_output": None, "blocker_type": None,
        "provider_teardown_verified": False, "official_billing_reconciled": False,
        "public_redistribution_authorized": False, "claim_ceiling": "development_only",
    }
    session, server, thread, preflight, name = None, None, None, None, None
    joined = False
    relay_results = []
    try:
        inputs = verify_g1_team_sealed_inputs(runtime_root)
        packet = inputs["packet"]
        profile = packet["request"]["policy_profile"]
        binding = RelayBinding(packet["packet_digest"], profile["profile_digest"],
                               packet["trusted_setup"]["setup_digest"], profile["delivery"]["mode"])
        state["execution_packet_digest"] = packet["packet_digest"]
        state["stage_reached"] = "host_preflight"
        preflight = _write(output_dir / PREFLIGHT_FILENAME, preflight_g1_vm_host(packet))
        state["preflight_digest"] = preflight["receipt_digest"]
        # Runtime session creates its own directory, separate from simulator inputs.
        state["stage_reached"] = "policy_synthetic_conformance"
        session = open_g1_team_runtime_session(
            profile=profile, trusted_setup=packet["trusted_setup"], authenticated_owner=packet["request"]["owner"],
            approved_binding=packet["operator_approval"]["runtime_binding"], output_dir=policy_root / "runtime",
        )
        with tempfile.TemporaryDirectory(prefix="g1-vm-relay-", dir=str(Path("/tmp").resolve())) as private:
            private_root = Path(private)
            secret = secrets.token_hex(32)
            server = G1PolicyRelayServer(path=private_root / "wire", binding=binding, secret=secret)
            write_g1_team_relay_config(path=private_root / "private.json", binding=binding,
                                      socket_path=server.path, secret=secret)
            thread = threading.Thread(target=lambda: relay_results.append(server.serve_one(
                session_factory=lambda: session, timeout_seconds=600)), daemon=True)
            thread.start()
            provisioning = runtime_root / "provisioned_runtime_sources"
            provisioning.mkdir(mode=0o700)
            name = "blueprint-g1-simulator-" + secrets.token_hex(16)
            command = simulator_container_command(runtime_root=runtime_root, output_dir=output_dir,
                                                  relay_directory=private_root, container_name=name)
            state["stage_reached"] = "simulator_process"
            try:
                exited = run_g1_team_worker_process(command=command,
                    diagnostics_dir=simulator_root / "private_diagnostics", timeout_seconds=timeout_seconds)
                state["child_exit_receipt_digest"] = exited["receipt_digest"]
                if exited["status"] != "exited" or exited["returncode"] != 0:
                    raise ValueError("g1_vm_simulator_process_failed")
            finally:
                # Daemon containers can outlive their CLI; always verify removal.
                closed = close_simulator_container(container_name=name,
                    image_id=preflight["simulator_local_image_id"], output_dir=simulator_root)
                state["simulator_teardown_digest"] = closed["receipt_digest"]
                server.close()
                joined = True
                thread.join(timeout=_JOIN_SECONDS)
                if thread.is_alive():
                    raise ValueError("g1_vm_policy_close_owner_not_terminal")
            state["stage_reached"] = "host_output_verification"
    except BaseException as exc:  # noqa: BLE001 - SIGTERM/failure still needs owned cleanup
        state["blocker_type"] = type(exc).__name__
    finally:
        if server is not None:
            server.close()
        if thread is not None and thread.is_alive() and not joined:
            thread.join(timeout=_JOIN_SECONDS)
        if thread is not None and thread.is_alive():
            state["blocker_type"] = state["blocker_type"] or "PolicyCloseOwnerNotTerminal"
        if session is not None and (thread is None or not thread.is_alive()):
            try:
                state["policy_session_closed"] = session.close()["status"] == "closed"
            except Exception as exc:
                state["blocker_type"] = state["blocker_type"] or type(exc).__name__
        if relay_results:
            relay = _write(policy_root / RELAY_FILENAME, relay_results[0])
            state["relay_digest"] = relay["receipt_digest"]
        if state["blocker_type"] is None:
            try:
                from .native_g1_team_vm_output import verify_g1_team_vm_evidence
                state["verified_output"] = verify_g1_team_vm_evidence(
                    output_dir=output_dir, execution_packet=inputs["packet"],
                    scene_plan_digest=inputs["manifest"]["scene_plan_digest"],
                    scene_packet_receipt_digest=inputs["manifest"]["scene_packet_receipt_digest"],
                )
                state["status"] = "completed_development_only"
            except Exception as exc:
                state["blocker_type"] = type(exc).__name__
    return _write(output_dir / HOST_FILENAME, state, field="result_digest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    def stopped(*_: Any) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise RuntimeError("g1_vm_host_stop_requested")
    signal.signal(signal.SIGTERM, stopped)
    result = run_g1_team_vm_host(runtime_root=args.runtime_root, output_dir=args.output_dir)
    print(json.dumps({"status": result["status"], "result_digest": result["result_digest"]}))
    return 0 if result["status"] == "completed_development_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
