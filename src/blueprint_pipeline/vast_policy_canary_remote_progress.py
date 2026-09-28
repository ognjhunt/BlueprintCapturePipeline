"""Read fixed policy-cell milestones when Vast's container log stops advancing.

This is a bounded diagnostic channel, not an episode or success receipt. Remote
file contents are never retained; only allowlisted milestones are returned.
"""

from __future__ import annotations

from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping

from .gpu_render_providers import (
    _validated_vast_known_hosts_pin,
    _vast_ssh_endpoint,
    enroll_vast_ssh_host_key,
)
from .vast_provider_output_recovery import _identity_file, _ssh_command


_CELL_STAGES = frozenset({
    "static_preflight_passed", "isaac_launch_started", "isaac_launch_completed",
    "observation_gate_passed", "policy_loaded", "episode_completed",
})
_REMOTE_STAGES = frozenset({
    "runtime_source_receipt", "media_toolchain_receipt", "cell_directory",
})
_CELL_LINE = re.compile(r"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=([0-9]{1,2}):stage=([a-z_]+)")
_REMOTE_LINE = re.compile(r"REMOTE_STAGE:([a-z_]+)")
_MAX_OUTPUT_BYTES = 8192

_REMOTE_SCRIPT = """
{
for root in /workspace /tmp/blueprint_vast_work; do
  out="$root/adp_arena_provider_bundle/runtime_output"
  [ -f "$out/native_task_runtime_source_provisioning.v1.json" ] && printf 'REMOTE_STAGE:runtime_source_receipt\\n'
  [ -f "$out/adp009d_media_toolchain_status.json" ] && printf 'REMOTE_STAGE:media_toolchain_receipt\\n'
  [ -d "$out/cell_runs" ] && printf 'REMOTE_STAGE:cell_directory\\n'
  for file in "$out"/cell_runs/[0-9][0-9]/cell_progress.log; do
    [ -f "$file" ] && cat "$file"
  done
done
} | head -c 8193
"""


def probe_policy_canary_remote_progress(
    connection: Mapping[str, Any], *, attempt_dir: Path,
) -> dict[str, Any]:
    """Return at most 100 validated milestones over attempt-pinned SSH."""

    endpoint = _vast_ssh_endpoint(connection)
    identity = _identity_file()
    if endpoint is None or identity is None:
        return {"status": "unavailable", "milestones": [], "reason": "endpoint_or_identity_unavailable"}
    host, port = endpoint
    enrollment = enroll_vast_ssh_host_key(
        {"ssh_host": host, "ssh_port": port},
        attempt_dir=attempt_dir, timeout_seconds=10,
    )
    known_hosts_value = str(enrollment.get("known_hosts_file") or "")
    pin = (
        _validated_vast_known_hosts_pin(known_hosts_value, host=host, port=port)
        if enrollment.get("status") == "enrolled" and known_hosts_value
        else None
    )
    if pin is None:
        return {"status": "unavailable", "milestones": [], "reason": "host_key_pin_unavailable"}
    known_hosts, known_hosts_sha256 = pin
    remote = shlex.join(["sh", "-c", _REMOTE_SCRIPT])
    try:
        completed = subprocess.run(
            _ssh_command(
                host=host, port=port, identity=identity,
                known_hosts=known_hosts, remote=remote,
            ),
            check=False, capture_output=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "milestones": [], "reason": "ssh_probe_failed"}
    output = completed.stdout or b""
    if completed.returncode != 0 or len(output) > _MAX_OUTPUT_BYTES:
        return {"status": "unavailable", "milestones": [], "reason": "ssh_probe_invalid"}
    milestones: list[str] = []
    for raw_line in output.decode("ascii", errors="ignore").splitlines():
        remote_match = _REMOTE_LINE.fullmatch(raw_line)
        if remote_match and remote_match.group(1) in _REMOTE_STAGES:
            milestones.append(raw_line)
            continue
        cell_match = _CELL_LINE.fullmatch(raw_line)
        if cell_match and int(cell_match.group(1)) < 10 and cell_match.group(2) in _CELL_STAGES:
            milestones.append(raw_line)
    if len(milestones) > 100:
        return {"status": "unavailable", "milestones": [], "reason": "milestone_count_exceeded"}
    return {
        "status": "observed", "milestones": milestones,
        "known_hosts_sha256": known_hosts_sha256,
        "strict_host_key_checking": True,
        "raw_remote_output_recorded": False,
    }


__all__ = ["probe_policy_canary_remote_progress"]
