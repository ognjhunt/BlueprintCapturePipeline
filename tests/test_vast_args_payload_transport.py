from __future__ import annotations

import os
import shlex
import subprocess

import blueprint_pipeline.vast_provider_adapter as vpa
from blueprint_pipeline import vast_args_payload_transport as transport


MAX_SAFE_VAST_ARGS_STR_BYTES = 16_000


def test_scene_configuration_args_payload_compresses_below_vast_safe_limit() -> None:
    probe = vpa._probe_shell_script(
        "https://example.invalid/heartbeat",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
    )

    payload = vpa._create_payload(
        image=vpa.DEFAULT_ISAAC_IMAGE,
        label="blueprint-scene-configuration-size-regression",
        launch_mode="args",
        probe_script=probe,
        disk_gb=vpa.DEFAULT_ISAAC_DISK_GB,
    )

    args_str = payload["args_str"]
    assert len(args_str.encode("utf-8")) <= MAX_SAFE_VAST_ARGS_STR_BYTES
    assert transport.VAST_ARGS_GZIP_BASE64_MARKER in args_str
    command = shlex.split(args_str)
    syntax = subprocess.run(
        ["bash", "-n", "-c", command[2]],
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr.decode("utf-8", errors="replace")


def test_compressed_args_transport_preserves_probe_output_and_exit_code() -> None:
    probe = (
        "echo BLUEPRINT_COMPRESSED_ARGS_STARTED\n"
        + ("# deterministic-compressible-padding\n" * 1_000)
        + "exit 7\n"
    )
    payload = vpa._create_payload(
        image="image",
        label="compressed-args-probe",
        launch_mode="args",
        probe_script=probe,
        disk_gb=20,
    )
    args_str = payload["args_str"]
    command = shlex.split(args_str)

    assert transport.VAST_ARGS_GZIP_BASE64_MARKER in args_str
    assert len(args_str.encode("utf-8")) <= MAX_SAFE_VAST_ARGS_STR_BYTES
    executed = subprocess.run(
        command,
        env={**os.environ, "BLUEPRINT_VAST_ARGS_LOG_HOLD_SECONDS": "0"},
        capture_output=True,
        check=False,
        text=True,
    )
    assert executed.returncode == 7
    assert "BLUEPRINT_COMPRESSED_ARGS_STARTED" in executed.stdout
    assert "BLUEPRINT_VAST_ARGS_LOG_HOLD_STARTED" in executed.stdout
    assert "BLUEPRINT_VAST_ARGS_LOG_HOLD_DONE" in executed.stdout


def test_scene_configuration_onstart_payload_compresses_below_vast_safe_limit() -> None:
    """ssh_direct launches hit the same 16384-byte ceiling as args mode.

    Run ``...-2ad2f8f6-...-154844Z`` (offer 39678103) was refused with
    ``error 400/3471: Invalid args: len(image) > 1024, or len(args) > 16384,
    or len(label) > 256``: the args-mode fix left the onstart branch --
    which ssh_direct uses -- uncompressed.
    """

    probe = vpa._probe_shell_script(
        "https://example.invalid/heartbeat",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
    )
    assert len(probe.encode("utf-8")) > MAX_SAFE_VAST_ARGS_STR_BYTES

    payload = vpa._create_payload(
        image=vpa.DEFAULT_ISAAC_IMAGE,
        label="blueprint-scene-configuration-onstart-size",
        launch_mode="ssh_direct",
        probe_script=probe,
        disk_gb=vpa.DEFAULT_ISAAC_DISK_GB,
    )

    onstart = payload["onstart"]
    assert len(onstart.encode("utf-8")) <= MAX_SAFE_VAST_ARGS_STR_BYTES
    assert transport.VAST_ARGS_GZIP_BASE64_MARKER in onstart
    syntax = subprocess.run(
        ["bash", "-n", "-c", onstart],
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr.decode("utf-8", errors="replace")


def test_compressed_onstart_preserves_probe_output_and_exit_code() -> None:
    probe = (
        "echo BLUEPRINT_COMPRESSED_ONSTART_STARTED\n"
        + ("# deterministic-compressible-padding\n" * 1_000)
        + "exit 7\n"
    )
    onstart = transport.onstart_mode_script(probe)
    assert len(onstart.encode("utf-8")) <= MAX_SAFE_VAST_ARGS_STR_BYTES
    completed = subprocess.run(
        ["bash", "-c", onstart],
        capture_output=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 7
    assert b"BLUEPRINT_COMPRESSED_ONSTART_STARTED" in completed.stdout


def test_small_onstart_script_is_passed_through_unchanged() -> None:
    probe = "echo hello\n"
    assert transport.onstart_mode_script(probe) == probe
