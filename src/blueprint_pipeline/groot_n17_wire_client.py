"""Protocol-pinned, wire-only client for frozen GR00T N1.7.

This module implements only the ZeroMQ/MessagePack surface used by NVIDIA's
``gr00t.policy.server_client.PolicyClient`` at revision
``b9955401d50c92a29258732e3ad6ccd579f1bdc0``.  Its upstream source digest is
``d29d2e68e97ceae1762243315f23f22e696538da60fecd87d9a1391d20f7f1b9``.

It deliberately imports no GR00T, transformers, torch, model, or server code.
That keeps Isaac's interpreter a thin client while the full frozen policy
environment remains in its separate server virtual environment.

The serializer safety boundary mirrors the pinned Apache-2.0 upstream client:
object-dtype arrays and pickle-bearing responses are rejected before
``msgpack_numpy`` can invoke pickle.
"""

from __future__ import annotations

import functools
import io
import json
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


GROOT_SOURCE_REVISION = "b9955401d50c92a29258732e3ad6ccd579f1bdc0"
UPSTREAM_SERVER_CLIENT_SHA256 = (
    "d29d2e68e97ceae1762243315f23f22e696538da60fecd87d9a1391d20f7f1b9"
)
WIRE_DEPENDENCY_VERSIONS = {
    "pyzmq": "27.0.1",
    "msgpack": "1.1.0",
    "msgpack-numpy": "0.4.8",
}
SELF_CHECK_MARKER = "GROOT_N17_WIRE_CLIENT_SELF_CHECK_OK"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _safe_encode(value: Any, *, chain: Callable[[Any], Any] | None = None) -> Any:
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        raise TypeError("groot_wire_object_dtype_encode_forbidden")
    return mnp.encode(value, chain=chain)


def _validated_modality_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("groot_wire_modality_json_invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("groot_wire_modality_payload_not_object")
    modality_keys = value.get("modality_keys")
    delta_indices = value.get("delta_indices")
    if (
        not isinstance(modality_keys, Sequence)
        or isinstance(modality_keys, (str, bytes))
        or not modality_keys
        or any(not str(item).strip() for item in modality_keys)
    ):
        raise ValueError("groot_wire_modality_keys_invalid")
    if (
        not isinstance(delta_indices, Sequence)
        or isinstance(delta_indices, (str, bytes))
    ):
        raise ValueError("groot_wire_modality_delta_indices_invalid")
    try:
        checked_indices = [int(item) for item in delta_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError("groot_wire_modality_delta_indices_invalid") from exc
    checked = dict(value)
    checked["modality_keys"] = [str(item) for item in modality_keys]
    checked["delta_indices"] = checked_indices
    return checked


def _decode_custom(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    marker_keys = (
        "__ModalityConfig__",
        b"__ModalityConfig__",
        "__ModalityConfig_class__",
        b"__ModalityConfig_class__",
    )
    if any(key in value for key in marker_keys):
        payload_key = next(
            (key for key in ("as_json", b"as_json") if key in value), None
        )
        if payload_key is None:
            raise ValueError("groot_wire_modality_payload_missing")
        return _validated_modality_mapping(value[payload_key])
    return value


def _safe_decode(value: Any, *, chain: Callable[[Any], Any] | None = None) -> Any:
    if isinstance(value, dict):
        marker = value.get("__ndarray_class__", value.get(b"__ndarray_class__"))
        if marker:
            payload = value.get("as_npy", value.get(b"as_npy"))
            if payload is None:
                raise ValueError("groot_wire_ndarray_payload_missing")
            return np.load(io.BytesIO(payload), allow_pickle=False)
        nd_value = value.get(b"nd", value.get("nd"))
        kind_value = value.get(b"kind", value.get("kind"))
        if nd_value and kind_value in (b"O", "O"):
            raise ValueError("groot_wire_object_dtype_decode_forbidden")
    return mnp.decode(value, chain=chain)


def encode_wire_message(value: Any) -> bytes:
    """Encode one request with the exact pinned serializer chain."""

    return msgpack.packb(
        value,
        default=functools.partial(_safe_encode, chain=lambda item: item),
    )


def decode_wire_message(payload: bytes) -> Any:
    """Decode one response with pickle disabled at every ndarray boundary."""

    return msgpack.unpackb(
        payload,
        object_hook=functools.partial(_safe_decode, chain=_decode_custom),
        raw=False,
    )


def run_wire_codec_self_check(*, require_distribution_versions: bool) -> None:
    """Exercise the wire codec without a network request or secret access."""

    if require_distribution_versions:
        observed = {
            package: metadata.version(package)
            for package in WIRE_DEPENDENCY_VERSIONS
        }
        if observed != WIRE_DEPENDENCY_VERSIONS:
            raise RuntimeError(
                "groot_wire_dependency_version_mismatch:"
                + json.dumps(observed, sort_keys=True)
            )
    numeric = np.arange(12, dtype=np.float32).reshape(3, 4)
    decoded_numeric = decode_wire_message(encode_wire_message({"array": numeric}))
    if not np.array_equal(decoded_numeric["array"], numeric):
        raise RuntimeError("groot_wire_numeric_round_trip_failed")
    modality = decode_wire_message(
        msgpack.packb(
            {
                "config": {
                    "__ModalityConfig__": True,
                    "as_json": {
                        "modality_keys": ["joint_position"],
                        "delta_indices": [0, 1],
                    },
                }
            }
        )
    )
    if modality["config"]["delta_indices"] != [0, 1]:
        raise RuntimeError("groot_wire_modality_round_trip_failed")
    try:
        encode_wire_message(np.asarray([object()], dtype=object))
    except TypeError as exc:
        if str(exc) != "groot_wire_object_dtype_encode_forbidden":
            raise
    else:
        raise RuntimeError("groot_wire_object_dtype_not_rejected")


class GrootN17WirePolicyClient:
    """One-request-at-a-time client matching the frozen GR00T wire protocol."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
        strict: bool = False,
        *,
        context_factory: Callable[[], Any] | None = None,
    ) -> None:
        if strict:
            raise ValueError("groot_wire_strict_mode_unsupported")
        if str(host) not in LOOPBACK_HOSTS or not 1 <= int(port) <= 65535:
            raise ValueError("groot_wire_endpoint_invalid")
        if int(timeout_ms) < 1:
            raise ValueError("groot_wire_timeout_invalid")
        self.host = str(host)
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.api_token = api_token
        self._context = (context_factory or zmq.Context)()
        self._closed = False
        self._socket = self._new_socket()

    def _new_socket(self) -> Any:
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        endpoint_host = f"[{self.host}]" if self.host == "::1" else self.host
        socket.connect(f"tcp://{endpoint_host}:{self.port}")
        return socket

    def _replace_invalid_socket(self) -> None:
        old_socket = self._socket
        old_socket.close(linger=0)
        self._socket = self._new_socket()

    def call_endpoint(
        self,
        endpoint: str,
        data: Mapping[str, Any] | None = None,
        requires_input: bool = True,
    ) -> Any:
        if self._closed:
            raise RuntimeError("groot_wire_client_closed")
        request: dict[str, Any] = {"endpoint": str(endpoint)}
        if requires_input:
            request["data"] = dict(data or {})
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self._socket.send(encode_wire_message(request))
            message = self._socket.recv()
        except zmq.error.Again:
            self._replace_invalid_socket()
            raise
        if message == b"ERROR":
            raise RuntimeError("groot_wire_server_error_frame")
        response = decode_wire_message(message)
        if isinstance(response, Mapping) and "error" in response:
            raise RuntimeError(f"groot_wire_server_error:{response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
        except zmq.error.ZMQError:
            return False
        return True

    def get_modality_config(self) -> Mapping[str, Any]:
        response = self.call_endpoint("get_modality_config", requires_input=False)
        if not isinstance(response, Mapping):
            raise ValueError("groot_wire_modality_response_not_object")
        return response

    def get_action(
        self,
        observation: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        response = self.call_endpoint(
            "get_action",
            {"observation": dict(observation), "options": options},
        )
        if (
            not isinstance(response, Sequence)
            or isinstance(response, (str, bytes))
            or len(response) != 2
        ):
            raise ValueError("groot_wire_action_response_invalid")
        return response[0], response[1]

    def reset(self, options: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        response = self.call_endpoint("reset", {"options": options})
        if not isinstance(response, Mapping):
            raise ValueError("groot_wire_reset_response_not_object")
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.close(linger=0)
        finally:
            self._context.term()

    def __enter__(self) -> GrootN17WirePolicyClient:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def main() -> int:
    run_wire_codec_self_check(require_distribution_versions=True)
    print(SELF_CHECK_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GROOT_SOURCE_REVISION",
    "GrootN17WirePolicyClient",
    "LOOPBACK_HOSTS",
    "SELF_CHECK_MARKER",
    "UPSTREAM_SERVER_CLIENT_SHA256",
    "WIRE_DEPENDENCY_VERSIONS",
    "decode_wire_message",
    "encode_wire_message",
    "run_wire_codec_self_check",
]
