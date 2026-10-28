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


def runtime_control_probe(*, probe, ready, job_dir: Path, instance_id: int, resource_name: str):
    """Subscribe only after the existing own-instance probe establishes its SSH pin."""
    from . import gpu_render_providers
    unsubscribe = None

    def wrapped(connection):
        nonlocal unsubscribe
        result = probe(connection)
        if unsubscribe is not None or result.get("status") != "observed":
            return result
        # The specific provider endpoint is bound to the ID created by this
        # adapter. Older provider responses may omit their optional label;
        # the receiver additionally joins the exact sealed session authority.
        if (connection.get("observed") is not True or connection.get("exited")
                or connection.get("instance_id") != instance_id
                or connection.get("resource_name") not in {None, resource_name}):
            return result
        endpoint = gpu_render_providers._vast_ssh_endpoint(connection)
        pin_dir = job_dir / "policy_remote_progress_ssh"
        if endpoint is None or gpu_render_providers._validated_vast_known_hosts_pin(
                pin_dir / "vast_ssh_known_hosts", host=endpoint[0], port=endpoint[1]) is None:
            return result
        close = ready({**connection, "instance_id": instance_id, "resource_name": resource_name,
                       "control_pin_dir": str(pin_dir)})
        if not callable(close):
            raise ValueError("beta_screening_runtime_control_subscription_invalid")
        unsubscribe = close
        return result

    def close():
        nonlocal unsubscribe
        subscribed, unsubscribe = unsubscribe, None
        if subscribed is not None:
            subscribed()
    return wrapped, close


class RuntimeControlRelay:
    """Keep optional provider subscription setup and exact teardown in one scope."""

    def __init__(self, *, ready, provider_bundle_kind: str, resource_name: str | None):
        if ready is not None and (not callable(ready)
                or provider_bundle_kind != "native_task_arena_policy_canary_session" or not resource_name):
            raise ValueError("beta_screening_runtime_control_configuration_invalid")
        self.ready, self.resource_name, self.unsubscribe = ready, resource_name, None

    def wrap(self, probe, *, job_dir: Path, instance_id: int):
        if self.ready is None:
            return probe
        wrapped, self.unsubscribe = runtime_control_probe(probe=probe, ready=self.ready, job_dir=job_dir,
            instance_id=instance_id, resource_name=self.resource_name)
        return wrapped

    def close(self) -> None:
        subscribed, self.unsubscribe = self.unsubscribe, None
        if subscribed is not None:
            try:
                subscribed()
            except BaseException:  # noqa: BLE001 - even an interrupted unsubscribe must allow exact owned teardown
                pass
