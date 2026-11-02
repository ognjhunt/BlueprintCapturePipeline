"""Fixed screening-control delivery over the existing controller-only pinned SSH.

No key/token is created or forwarded. The existing adapter must establish the
current own-instance identity and host pin before registering this callback.
This module is deliberately excluded from the GPU provider import closure.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping

from .beta_screening_runtime_control import MAX_BYTES, make_delivery

# The remote command and paths are code constants. Only bounded JSON travels on
# stdin; an input can never select a shell fragment, path, user or interpreter.
_REMOTE_SCRIPT = r'''
import json,sys
from pathlib import Path
raw=sys.stdin.buffer.read(32769)
if len(raw)>32768: raise ValueError("control_input_too_large")
delivery=json.loads(raw)
session=delivery["session"]
matches=[]
for root in (Path("/workspace/adp_arena_provider_bundle"),Path("/tmp/blueprint_vast_work/adp_arena_provider_bundle")):
    runtime=root/"provider_runtime"
    authority_path=runtime/"runtime_inputs"/"policy_canary_session_authority.json"
    if root.is_symlink() or runtime.is_symlink() or authority_path.is_symlink() or not authority_path.is_file(): continue
    authority=json.loads(authority_path.read_text())
    if all(authority.get(key)==value for key,value in session.items()): matches.append(runtime)
if len(matches)!=1: raise ValueError("control_owned_runtime_unavailable")
sys.path.insert(0,str(matches[0]))
from blueprint_pipeline.beta_screening_runtime_control import install_runtime_control
stored=install_runtime_control(runtime_root=matches[0],delivery=delivery)
print(json.dumps({"status":"delivered","binding_digest":stored["control"]["binding_digest"],
                  "control_digest":stored["control"]["control_digest"],"delivery_digest":stored["delivery_digest"]}))
'''
REMOTE_COMMAND = shlex.join(["python3", "-c", _REMOTE_SCRIPT])


def push_runtime_control(*, connection: Mapping[str, Any], attempt_dir: Path,
                         authority: Mapping[str, Any], binding: Mapping[str, Any],
                         control: Mapping[str, Any]) -> dict[str, Any]:
    """Push only to a verified current own-resource connection, using its pin."""
    from .gpu_render_providers import _validated_vast_known_hosts_pin, _vast_ssh_endpoint
    from .vast_provider_output_recovery import _identity_file, _ssh_command

    value = make_delivery(authority=authority, binding=binding, control=control)
    endpoint = _vast_ssh_endpoint(connection)
    if (connection.get("resource_name") != authority["resource_name"]
            or type(connection.get("instance_id")) is not int or connection["instance_id"] <= 0
            or endpoint is None):
        raise ValueError("beta_screening_control_delivery_owner_unproven")
    host, port = endpoint
    # Enrollment belongs to the existing provider's ready hook. This helper
    # cannot enroll a new endpoint or widen the SSH action surface.
    pin_path = Path(attempt_dir) / "vast_ssh_known_hosts"
    pin = _validated_vast_known_hosts_pin(pin_path, host=host, port=port)
    identity = _identity_file()
    if pin is None or identity is None:
        return {"status": "unavailable", "reason": "beta_screening_control_ssh_pin_or_identity_unavailable"}
    known_hosts, _fingerprint = pin
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(payload) > MAX_BYTES:
        raise ValueError("beta_screening_control_delivery_too_large")
    try:
        result = subprocess.run(_ssh_command(host=host, port=port, identity=identity,
            known_hosts=known_hosts, remote=REMOTE_COMMAND), input=payload, check=False, capture_output=True, timeout=20)
        if result.returncode or len(result.stdout or b"") > 4096:
            return {"status": "unavailable", "reason": "beta_screening_control_ssh_delivery_unavailable"}
        receipt = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return {"status": "unavailable", "reason": "beta_screening_control_ssh_delivery_unavailable"}
    if (not isinstance(receipt, dict) or set(receipt) != {"status", "binding_digest", "control_digest", "delivery_digest"}
            or receipt["status"] != "delivered" or receipt["binding_digest"] != control["binding_digest"]
            or receipt["control_digest"] != control["control_digest"]):
        return {"status": "unavailable", "reason": "beta_screening_control_ssh_receipt_unproven"}
    return receipt


def subscribe_own_runtime_control(*, connection: Mapping[str, Any], attempt_dir: Path,
                                  authority: Mapping[str, Any]):
    """Connect the existing allocator monitor to this verified provider only."""
    from .beta_policy_screening_control import subscribe_runtime_control

    def sink(binding, control):
        # An early provider startup can precede bundle extraction. A failed
        # delivery never advances the GPU's last verified observation; a later
        # real monitor response retries it within the same original deadline.
        return push_runtime_control(connection=connection, attempt_dir=attempt_dir,
                                    authority=authority, binding=binding, control=control)

    return subscribe_runtime_control(authority_digest=authority["authority_digest"],
                                     resource_name=authority["resource_name"], sink=sink)


def runtime_control_ready_for_authority(*, authority: Mapping[str, Any], execute: bool):
    """Prepare the bounded relay only for an existing Blueprint screening monitor."""
    if authority.get("run_kind") != "blueprint_beta_screening" or not execute:
        return None
    from .beta_policy_screening_control import require_runtime_control_monitor
    require_runtime_control_monitor(authority_digest=authority["authority_digest"], resource_name=authority["resource_name"])

    def ready(connection):
        return subscribe_own_runtime_control(connection=connection,
            attempt_dir=Path(connection["control_pin_dir"]), authority=authority)
    return ready
