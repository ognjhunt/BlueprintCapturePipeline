"""Bind one controlled-policy GPU allocation to its private sandbox network lease."""
from __future__ import annotations

import ipaddress
import json
import os
import shlex
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from .company_policy_sandbox_manager_client import SandboxManagerClient
from .gpu_render_providers import (
    _validated_vast_known_hosts_pin,
    _vast_ssh_endpoint,
    enroll_vast_ssh_host_key,
)
from .vast_provider_output_recovery import _identity_file, _ssh_command


CONTEXT_ENV = "BLUEPRINT_POLICY_SANDBOX_NETWORK_CONTEXT"
_REMOTE_IP_SCRIPT = (
    "set -eu; "
    "curl -fsS --connect-timeout 5 --max-time 10 https://api.ipify.org; "
    "printf '\\n'; "
    "curl -fsS --connect-timeout 5 --max-time 10 https://checkip.amazonaws.com"
)


def _context() -> dict[str, Any] | None:
    configured = os.environ.get(CONTEXT_ENV)
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("policy_network_context_path_invalid")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("policy_network_context_not_private")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {
        "job_id", "endpoint_url", "token_file", "certificate_file"
    }:
        raise ValueError("policy_network_context_invalid")
    return value


def _client(value: Mapping[str, Any]) -> SandboxManagerClient:
    return SandboxManagerClient(
        endpoint_url=str(value["endpoint_url"]),
        token_file=Path(str(value["token_file"])),
        certificate_file=Path(str(value["certificate_file"])),
    )


def _outbound_ipv4(connection: Mapping[str, Any], *, attempt_dir: Path) -> str:
    endpoint = _vast_ssh_endpoint(connection)
    identity = _identity_file()
    if endpoint is None or identity is None:
        raise ValueError("policy_network_gpu_ssh_unavailable")
    host, port = endpoint
    enrollment = enroll_vast_ssh_host_key(
        {"ssh_host": host, "ssh_port": port}, attempt_dir=attempt_dir, timeout_seconds=10,
    )
    known_hosts_value = str(enrollment.get("known_hosts_file") or "")
    pin = (_validated_vast_known_hosts_pin(known_hosts_value, host=host, port=port)
           if enrollment.get("status") == "enrolled" and known_hosts_value else None)
    if pin is None:
        raise ValueError("policy_network_gpu_host_key_unavailable")
    known_hosts, _ = pin
    remote = shlex.join(["sh", "-c", _REMOTE_IP_SCRIPT])
    command = _ssh_command(host=host, port=port, identity=identity,
                           known_hosts=known_hosts, remote=remote)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            completed = subprocess.run(command, capture_output=True, timeout=35, check=False)
            if completed.returncode == 0 and len(completed.stdout) <= 128:
                addresses = completed.stdout.decode("ascii").splitlines()
                if len(addresses) == 2 and addresses[0] == addresses[1]:
                    address = ipaddress.ip_address(addresses[0].strip())
                    if isinstance(address, ipaddress.IPv4Address) and address.is_global:
                        return str(address)
        except (OSError, UnicodeError, ValueError, subprocess.TimeoutExpired):
            pass
        time.sleep(5)
    raise TimeoutError("policy_network_gpu_outbound_address_unverified")


def allow_policy_network(*, instance_id: int, connection: Mapping[str, Any],
                         attempt_dir: Path) -> dict[str, Any] | None:
    value = _context()
    if value is None:
        return None
    outbound = _outbound_ipv4(connection, attempt_dir=attempt_dir / "policy_network_ssh")
    receipt = _client(value).allow_network(job_id=str(value["job_id"]),
        instance_id=instance_id, outbound_ipv4=outbound)
    if (receipt.get("status") != "network_allowed"
            or receipt.get("job_id") != value["job_id"]
            or receipt.get("instance_id") != instance_id
            or receipt.get("outbound_ipv4") != outbound):
        raise ValueError("policy_network_lease_receipt_invalid")
    return receipt


def close_policy_network() -> dict[str, Any] | None:
    value = _context()
    if value is None:
        return None
    receipt = _client(value).close_network(job_id=str(value["job_id"]))
    if receipt.get("status") != "network_closed" or receipt.get("job_id") != value["job_id"]:
        raise ValueError("policy_network_close_receipt_invalid")
    return receipt
