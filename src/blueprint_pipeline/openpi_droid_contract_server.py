"""Serve an OpenPI DROID policy on the session's own machine over ``openpi_droid_https.v1``.

Plan 2.1 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``): one
standard connection for every policy. This server loads a policy Blueprint runs
itself with the native loader (``openpi_droid_policy_runtime.
load_identity_bound_policy``, the same verified checkpoint, pinned config and
``create_trained_policy`` the websocket server uses) and answers exactly the
requests ``droid_https_policy_client.DroidHttpsTeamPolicyClient`` sends:

- ``GET <metadata path>`` returns the spec's ``expected_metadata()`` and nothing
  else (the client refuses any other bytes);
- ``POST <endpoint path>`` takes ``{schema_version, request_id,
  policy_interface_id, observation}`` with lossless PNG views, the joint and
  gripper state and the prompt, rebuilds the exact arrays the native path
  hands the policy (uint8 HxWx3 views, float64 ``(7,)`` and ``(1,)`` state),
  calls ``policy.infer`` once and returns the policy's own output (``actions``,
  ``policy_timing``) plus ``server_timing``, as OpenPI's websocket server does.

Readiness never infers: a metadata read does not advance the policy's random
state, so the first inference is the first episode's, as on the native path.

Privacy and rights. The server listens on 127.0.0.1 only, on the rented GPU
that already runs the simulator, and receives the same observations the native
path gives the policy in-process. It makes no outbound call: the checkpoint is
verified on disk before it starts, it reads nothing from the network, and its
only socket is the loopback listener. Nothing new leaves the session. A bearer
token generated on the session (never on the host, never in the bundle) keeps
other local processes from driving the policy.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import io
import json
import math
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# The contract's own constants, read from its client: the server invents nothing.
from .droid_https_policy_client import IMAGE_KEYS, INTERFACE_ID, METADATA_SCHEMA, REQUEST_SCHEMA, STATE_KEYS

SCHEMA_VERSION = "blueprint.openpi_droid_contract_server.v1"
LOOPBACK_HOST = "127.0.0.1"
# The native DROID observation's state shapes (``adp009d_droid_observation``): seven joints, one gripper.
STATE_SHAPES = dict(zip(STATE_KEYS, ((7,), (1,)), strict=True))
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 1024
REQUEST_READ_TIMEOUT_SECONDS = 30.0


class ContractRequestError(ValueError):
    """A request the contract does not allow; answered 400 and never sent to the policy."""


def decode_observation(wire: Any) -> dict[str, Any]:
    """Invert ``droid_https_policy_client.encode_observation`` exactly."""

    import numpy as np
    from PIL import Image

    if not isinstance(wire, Mapping) or set(wire) != {*IMAGE_KEYS, *STATE_SHAPES, "prompt"}:
        raise ContractRequestError("contract_observation_keys_invalid")
    observation: dict[str, Any] = {}
    for key in IMAGE_KEYS:
        view = wire[key]
        if (not isinstance(view, Mapping) or set(view) != {"encoding", "height", "width", "data"}
                or view["encoding"] != "png_base64" or not isinstance(view["data"], str)
                or type(view["height"]) is not int or type(view["width"]) is not int
                or not 0 < view["height"] <= MAX_IMAGE_SIDE or not 0 < view["width"] <= MAX_IMAGE_SIDE):
            raise ContractRequestError("contract_observation_image_invalid")
        try:
            raw = base64.b64decode(view["data"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (view["width"], view["height"]):
                    raise ContractRequestError("contract_observation_image_invalid")
                array = np.asarray(image, dtype=np.uint8).copy()
        except (binascii.Error, OSError, ValueError) as exc:
            if isinstance(exc, ContractRequestError):
                raise
            raise ContractRequestError("contract_observation_image_invalid") from None
        observation[key] = array
    for key, shape in STATE_SHAPES.items():
        values = wire[key]
        if (not isinstance(values, list) or len(values) != shape[0]
                or not all(type(value) in (int, float) and math.isfinite(value) for value in values)):
            raise ContractRequestError("contract_observation_state_invalid")
        observation[key] = np.asarray(values, dtype=float)
    if not isinstance(wire["prompt"], str) or not wire["prompt"].strip():
        raise ContractRequestError("contract_observation_prompt_invalid")
    observation["prompt"] = wire["prompt"]
    return observation


def _wire(value: Any) -> Any:
    """JSON form of the policy's own output; non-finite numbers stay visible to the client."""

    if isinstance(value, Mapping):
        return {str(key): _wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _wire(tolist())
    raise TypeError("contract_response_value_unserializable")


class ContractPolicyService:
    """The contract over one loaded policy; transport-free so tests drive it directly."""

    def __init__(self, *, policy: Any, policy_spec: Mapping[str, Any], bearer_token: str,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not bearer_token or any(character in bearer_token for character in "\r\n "):
            raise ValueError("contract_server_token_invalid")
        if policy_spec.get("policy_interface_id", INTERFACE_ID) != INTERFACE_ID:
            raise ValueError("contract_server_interface_mismatch")
        self._policy = policy
        self._token = bearer_token
        self._clock = clock
        self.metadata = {"schema_version": METADATA_SCHEMA, "policy_id": policy_spec["policy_id"],
                         "policy_interface_id": INTERFACE_ID,
                         "action_space": policy_spec.get("action_space", "joint_position"),
                         "action_chunk_rows": policy_spec["action_chunk_rows"],
                         "model_identity_digest": policy_spec["model_identity_digest"]}
        self.metadata_path = urlsplit(policy_spec["metadata_url"]).path
        self.endpoint_path = urlsplit(policy_spec["endpoint_url"]).path
        self.inference_count = 0

    def authorized(self, header: str | None) -> bool:
        expected = "Bearer " + self._token
        return isinstance(header, str) and hmac.compare_digest(header.encode(), expected.encode())

    def infer(self, body: bytes) -> dict[str, Any]:
        def reject(_value: str) -> None:
            raise ContractRequestError("contract_request_nonfinite")

        try:
            request = json.loads(body, parse_constant=reject)
        except json.JSONDecodeError:
            raise ContractRequestError("contract_request_not_json") from None
        if (not isinstance(request, Mapping)
                or set(request) != {"schema_version", "request_id", "policy_interface_id", "observation"}
                or request["schema_version"] != REQUEST_SCHEMA or request["policy_interface_id"] != INTERFACE_ID
                or not isinstance(request["request_id"], str) or not 1 <= len(request["request_id"]) <= 128):
            raise ContractRequestError("contract_request_envelope_invalid")
        observation = decode_observation(request["observation"])
        started = self._clock()
        output = self._policy.infer(observation)
        self.inference_count += 1
        if not isinstance(output, Mapping):
            # The client refuses a non-object envelope; never reshape the policy's answer.
            return {"policy_output_not_object": _wire(output)}
        response = {str(key): _wire(value) for key, value in output.items()}
        response["server_timing"] = {"infer_ms": round((self._clock() - started) * 1000.0, 3)}
        return response


def _handler(service: ContractPolicyService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "BlueprintContractPolicy/1"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            return None  # Request lines carry no secrets, but observations stay out of logs.

        def _send(self, status: int, value: Any) -> None:
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            if service.authorized(self.headers.get("Authorization")):
                return True
            self._send(401, {"error": "contract_request_unauthorized"})
            return False

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            if not self._authorized():
                return
            if self.path != service.metadata_path:
                self._send(404, {"error": "contract_path_unknown"})
                return
            self._send(200, service.metadata)

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            if not self._authorized():
                return
            if self.path != service.endpoint_path:
                self._send(404, {"error": "contract_path_unknown"})
                return
            length = self.headers.get("Content-Length")
            if (length is None or not length.isdigit() or not 0 < int(length) <= MAX_REQUEST_BYTES
                    or self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json"):
                self._send(400, {"error": "contract_request_size_or_type_invalid"})
                return
            try:
                response = service.infer(self.rfile.read(int(length)))
            except ContractRequestError as exc:
                self._send(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - the policy's own failure, typed, no detail
                self._send(500, {"error": "contract_policy_inference_failed", "type": type(exc).__name__})
                return
            self._send(200, response)

    return Handler


class ContractHTTPServer(HTTPServer):
    """One request at a time, as OpenPI's websocket server serves one client."""

    allow_reuse_address = True

    def get_request(self):  # type: ignore[no-untyped-def]
        connection, address = super().get_request()
        connection.settimeout(REQUEST_READ_TIMEOUT_SECONDS)
        return connection, address


def build_server(service: ContractPolicyService, *, port: int, host: str = LOOPBACK_HOST) -> ContractHTTPServer:
    if host != LOOPBACK_HOST:
        raise ValueError("contract_server_must_be_loopback_only")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("contract_server_port_invalid")
    return ContractHTTPServer((host, port), _handler(service))


def read_token(path: str | Path) -> str:
    token_path = Path(path)
    status = token_path.lstat()
    if not token_path.is_absolute() or token_path.is_symlink() or status.st_mode & 0o077 or not 0 < status.st_size <= 512:
        raise ValueError("contract_server_token_file_not_private")
    return token_path.read_text(encoding="utf-8").strip()


def serve(*, execution_spec_path: str | Path, checkpoint_dir: str | Path, checkpoint_inventory_path: str | Path,
          token_file: str | Path, port: int, identity_out: str | Path) -> None:
    """Load the native policy and serve the contract until terminated."""

    from .openpi_droid_policy_runtime import OpenPIDroidPolicySpec, load_identity_bound_policy

    execution = json.loads(Path(execution_spec_path).read_text(encoding="utf-8"))
    served = execution.get("session_policy_server")
    if not isinstance(served, Mapping) or execution.get("policy_interface_id") != INTERFACE_ID:
        raise ValueError("contract_server_execution_spec_invalid")
    native = OpenPIDroidPolicySpec(**served["native_policy_spec"])
    policy_spec = execution["policy_spec"]
    if urlsplit(policy_spec["endpoint_url"]).port != port:
        raise ValueError("contract_server_port_mismatch")
    service_token = read_token(token_file)
    policy, local_verification = load_identity_bound_policy(
        spec=native, checkpoint_dir=checkpoint_dir, checkpoint_inventory_path=checkpoint_inventory_path)
    service = ContractPolicyService(policy=policy, policy_spec=policy_spec, bearer_token=service_token)
    server = build_server(service, port=port)
    # Written after load and bind: the starter reads it into its receipt. No secret in it.
    Path(identity_out).write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "status": "serving", "host": LOOPBACK_HOST, "port": port,
        "native_server_metadata": native.server_metadata(), "local_checkpoint_verification": local_verification,
        "contract_metadata": service.metadata, "outbound_calls": "none"}, sort_keys=True), encoding="utf-8")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-inventory", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity-out", required=True)
    args = parser.parse_args(argv)
    serve(execution_spec_path=args.execution_spec, checkpoint_dir=args.checkpoint_dir,
          checkpoint_inventory_path=args.checkpoint_inventory, token_file=args.token_file, port=args.port,
          identity_out=args.identity_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ContractPolicyService", "ContractRequestError", "build_server", "decode_observation", "serve"]
