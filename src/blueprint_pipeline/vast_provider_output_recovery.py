"""Bounded last-chance recovery of a completed Vast provider archive.

The primary transport is the immutable object-store PUT.  This module exists
only for the narrow case where the remote worker has sealed the output zip but
that PUT failed: recover the exact file over strict pinned SSH before teardown
can make the evidence permanently unavailable.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any, Mapping

from .gpu_render_providers import (
    _validated_vast_known_hosts_pin,
    enroll_vast_ssh_host_key,
)


VAST_SSH_IDENTITY_FILE_ENV = "BLUEPRINT_VAST_SSH_IDENTITY_FILE"
DEFAULT_VAST_SSH_IDENTITY_FILE = "~/.ssh/id_ed25519"
MAX_RECOVERY_SECONDS = 900.0

_REMOTE_ARCHIVE_BY_BUNDLE_KIND = {
    "adp_retained_scene_render": "/workspace/adp_retained_scene_render_provider_runtime_output.zip",
    "adp_arena": "/workspace/adp_arena_provider_runtime_output.zip",
    "adp009d_policy_runtime_smoke": "/workspace/adp_arena_provider_runtime_output.zip",
    "adp009d_isaac": "/workspace/adp_arena_provider_runtime_output.zip",
    "adp009d_articulated_native": "/workspace/adp_arena_provider_runtime_output.zip",
    "native_task_arena": "/workspace/adp_arena_provider_runtime_output.zip",
    "native_task_arena_policy_canary_session": "/workspace/adp_arena_provider_runtime_output.zip",
    "paired_target_native_import": "/workspace/adp_arena_provider_runtime_output.zip",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_file() -> Path | None:
    path = Path(
        os.getenv(VAST_SSH_IDENTITY_FILE_ENV, DEFAULT_VAST_SSH_IDENTITY_FILE)
    ).expanduser()
    try:
        mode = path.stat().st_mode & 0o777
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if path.is_symlink() or not path.is_file() or mode & 0o077:
        return None
    return resolved


def _ssh_command(
    *, host: str, port: int, identity: Path, known_hosts: Path, remote: str
) -> list[str]:
    return [
        "ssh",
        "-i",
        str(identity),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "--",
        f"root@{host}",
        remote,
    ]


def recover_provider_output_before_teardown(
    *,
    connection: Mapping[str, Any],
    provider_bundle_kind: str,
    output_path: str | Path,
    attempt_dir: str | Path,
    expected_size_bytes: int,
    minimum_free_bytes: int = 0,
    timeout_seconds: float = MAX_RECOVERY_SECONDS,
) -> dict[str, Any]:
    """Stream one sealed remote archive to local disk before provider teardown."""

    remote_path = _REMOTE_ARCHIVE_BY_BUNDLE_KIND.get(provider_bundle_kind)
    if remote_path is None:
        return {
            "status": "not_supported",
            "blockers": ["provider_output_ssh_recovery_bundle_kind_unsupported"],
            "raw_secret_values_recorded": False,
        }
    host = str(connection.get("ssh_host") or "").strip()
    try:
        port = int(connection.get("ssh_port") or 0)
    except (TypeError, ValueError):
        port = 0
    if not host or port < 1 or port > 65535:
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_endpoint_invalid"],
            "raw_secret_values_recorded": False,
        }
    if (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes <= 0
    ):
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_expected_size_invalid"],
            "raw_secret_values_recorded": False,
        }
    identity = _identity_file()
    if identity is None:
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_identity_invalid"],
            "raw_secret_values_recorded": False,
        }
    enrollment = enroll_vast_ssh_host_key(
        {"ssh_host": host, "ssh_port": port}, attempt_dir=attempt_dir
    )
    known_hosts_value = str(enrollment.get("known_hosts_file") or "")
    pin = (
        _validated_vast_known_hosts_pin(known_hosts_value, host=host, port=port)
        if enrollment.get("status") == "enrolled" and known_hosts_value
        else None
    )
    if pin is None:
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_host_key_pin_invalid"],
            "raw_secret_values_recorded": False,
        }
    known_hosts, known_hosts_sha256 = pin
    try:
        timeout = min(MAX_RECOVERY_SECONDS, max(1.0, float(timeout_seconds)))
    except (TypeError, ValueError):
        timeout = MAX_RECOVERY_SECONDS
    metadata_remote = shlex.join(
        [
            "sh",
            "-c",
            'bytes=$(wc -c < "$1" | tr -d "[:space:]") && sha256sum "$1" | awk -v b="$bytes" \'{print b " " $1}\'',
            "provider-output-metadata",
            remote_path,
        ]
    )
    try:
        metadata = subprocess.run(
            _ssh_command(
                host=host,
                port=port,
                identity=identity,
                known_hosts=known_hosts,
                remote=metadata_remote,
            ),
            check=False,
            capture_output=True,
            timeout=min(120.0, timeout),
            text=True,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_metadata_timeout"],
            "known_hosts_sha256": known_hosts_sha256,
            "raw_secret_values_recorded": False,
        }
    match = re.fullmatch(r"([0-9]+) ([0-9a-f]{64})\n?", metadata.stdout or "")
    if metadata.returncode != 0 or match is None:
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_metadata_invalid"],
            "returncode": metadata.returncode,
            "known_hosts_sha256": known_hosts_sha256,
            "raw_secret_values_recorded": False,
        }
    remote_size = int(match.group(1))
    remote_sha256 = match.group(2)
    if remote_size != expected_size_bytes:
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_remote_size_mismatch"],
            "expected_size_bytes": expected_size_bytes,
            "remote_size_bytes": remote_size,
            "known_hosts_sha256": known_hosts_sha256,
            "raw_secret_values_recorded": False,
        }
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < remote_size + max(0, int(minimum_free_bytes)):
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_disk_capacity_insufficient"],
            "required_bytes": remote_size + max(0, int(minimum_free_bytes)),
            "free_bytes": free_bytes,
            "known_hosts_sha256": known_hosts_sha256,
            "raw_secret_values_recorded": False,
        }
    partial = destination.with_name(destination.name + ".ssh-recovery.partial")
    partial.unlink(missing_ok=True)
    stream_remote = shlex.join(["cat", "--", remote_path])
    try:
        with partial.open("xb") as output:
            transfer = subprocess.run(
                _ssh_command(
                    host=host,
                    port=port,
                    identity=identity,
                    known_hosts=known_hosts,
                    remote=stream_remote,
                ),
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        partial.unlink(missing_ok=True)
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_transfer_timeout"],
            "known_hosts_sha256": known_hosts_sha256,
            "raw_secret_values_recorded": False,
        }
    if (
        transfer.returncode != 0
        or not partial.is_file()
        or partial.stat().st_size != remote_size
        or _sha256(partial) != remote_sha256
    ):
        partial.unlink(missing_ok=True)
        return {
            "status": "blocked",
            "blockers": ["provider_output_ssh_recovery_transfer_invalid"],
            "returncode": transfer.returncode,
            "known_hosts_sha256": known_hosts_sha256,
            "raw_secret_values_recorded": False,
        }
    os.replace(partial, destination)
    return {
        "status": "completed",
        "recovery_attempted": True,
        "recovered_size_bytes": remote_size,
        "recovered_sha256": f"sha256:{remote_sha256}",
        "known_hosts_sha256": known_hosts_sha256,
        "strict_host_key_checking": True,
        "streamed_to_disk": True,
        "raw_secret_values_recorded": False,
    }


__all__ = ["recover_provider_output_before_teardown"]
