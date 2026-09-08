"""Lossless typed policy request values and observed transport serialization.

Records contain scientific inputs only, never API credentials. They can be
decoded independently without a model or a vendor policy installation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

try:
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:
    from .decision_evidence_contracts import canonical_digest


def snapshot_request(value: Any) -> Any:
    import numpy as np
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("policy_request_array_invalid")
        raw = np.ascontiguousarray(array).tobytes()
        return {"kind": "array", "dtype": array.dtype.str, "shape": list(array.shape),
                "bytes_base64": base64.b64encode(raw).decode("ascii"),
                "raw_sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("policy_request_key_invalid")
        if any(key.lower() in {"api_key", "api_token", "authorization", "password"} for key in value):
            raise ValueError("policy_request_credential_disclosure_forbidden")
        return {"kind": "mapping", "items": [[key, snapshot_request(value[key])] for key in sorted(value)]}
    if isinstance(value, (list, tuple)):
        return {"kind": "list", "items": [snapshot_request(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int)) or isinstance(value, float) and math.isfinite(value):
        return {"kind": "scalar", "value": value}
    raise ValueError("policy_request_value_invalid")


def restore_request(node: Mapping[str, Any]) -> Any:
    import numpy as np
    kind = node.get("kind")
    if kind == "array":
        dtype = np.dtype(node["dtype"])
        shape = node["shape"]
        if dtype.kind not in "biuf" or not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("policy_request_array_invalid")
        raw = base64.b64decode(node["bytes_base64"], validate=True)
        if len(raw) != math.prod(shape) * dtype.itemsize or node["raw_sha256"] != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ValueError("policy_request_array_digest_invalid")
        array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        if not np.isfinite(array).all():
            raise ValueError("policy_request_array_nonfinite")
        return array
    if kind == "mapping":
        result = {}
        for key, value in node["items"]:
            if not isinstance(key, str) or key in result:
                raise ValueError("policy_request_duplicate_key")
            result[key] = restore_request(value)
        return result
    if kind == "list":
        return [restore_request(value) for value in node["items"]]
    if kind == "scalar":
        value = node["value"]
        if snapshot_request(value) != dict(node):
            raise ValueError("policy_request_scalar_invalid")
        return value
    raise ValueError("policy_request_node_invalid")


def capture_request(request: Mapping[str, Any], *, transport: str,
                    scientific_wire_bytes: bytes | None = None,
                    decoded_wire_request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    snapshot = snapshot_request(request)
    if scientific_wire_bytes is not None and decoded_wire_request is None:
        raise ValueError("policy_request_wire_decode_required")
    if decoded_wire_request is not None and snapshot_request(decoded_wire_request) != snapshot:
        raise ValueError("policy_request_wire_values_mismatch")
    value = {"schema_version": "policy_request_evidence.v1", "transport": transport,
             "request": snapshot, "request_digest": canonical_digest(snapshot),
             "serialization_verified": scientific_wire_bytes is not None and decoded_wire_request is not None,
             "scientific_wire_bytes_base64": base64.b64encode(scientific_wire_bytes).decode("ascii") if scientific_wire_bytes is not None else None,
             "scientific_wire_sha256": "sha256:" + hashlib.sha256(scientific_wire_bytes).hexdigest() if scientific_wire_bytes is not None else None,
             "credential_material_retained": False}
    value["evidence_digest"] = canonical_digest(value, digest_field="evidence_digest")
    return value


def validate_request_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value), allow_nan=False))
    if receipt.get("schema_version") != "policy_request_evidence.v1" or receipt.get("evidence_digest") != canonical_digest(receipt, digest_field="evidence_digest"):
        raise ValueError("policy_request_evidence_digest_invalid")
    decoded = restore_request(receipt["request"])
    if snapshot_request(decoded) != receipt["request"] or receipt["request_digest"] != canonical_digest(receipt["request"]):
        raise ValueError("policy_request_values_invalid")
    if receipt.get("credential_material_retained") is not False:
        raise ValueError("policy_request_credential_disclosure_forbidden")
    if receipt.get("scientific_wire_bytes_base64") is not None:
        raw = base64.b64decode(receipt["scientific_wire_bytes_base64"], validate=True)
        if receipt["scientific_wire_sha256"] != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ValueError("policy_request_wire_digest_invalid")
        if receipt.get("serialization_verified") is not True or snapshot_request(decode_scientific_wire(raw)) != receipt["request"]:
            raise ValueError("policy_request_wire_values_mismatch")
    elif receipt.get("serialization_verified") is not False:
        raise ValueError("policy_request_missing_verified_wire")
    return receipt


def decode_scientific_wire(raw: bytes) -> Any:
    """Decode the numeric-only MessagePack forms used by both pinned clients."""
    import msgpack
    import numpy as np

    def decode(value):
        if not isinstance(value, dict):
            return value
        keys = {key.decode() if isinstance(key, bytes) else key: item for key, item in value.items()}
        ndarray = keys.get("nd") is True or keys.get("__ndarray__") is True
        if ndarray:
            dtype = np.dtype(keys.get("type", keys.get("dtype")))
            shape = keys["shape"]
            data = keys["data"]
            if dtype.kind not in "biuf" or not isinstance(data, bytes) or any(type(size) is not int or size < 0 for size in shape):
                raise ValueError("policy_request_wire_array_invalid")
            if math.prod(shape) * dtype.itemsize != len(data):
                raise ValueError("policy_request_wire_array_size_invalid")
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        if keys.get("nd") is False or keys.get("__npgeneric__") is True:
            dtype = np.dtype(keys.get("type", keys.get("dtype")))
            if dtype.kind not in "biuf":
                raise ValueError("policy_request_wire_scalar_invalid")
            data = keys["data"]
            return np.frombuffer(data, dtype=dtype)[0] if isinstance(data, bytes) else dtype.type(data)
        return value

    return msgpack.unpackb(raw, raw=False, object_hook=decode)


def persist_request_evidence(value: Mapping[str, Any], *, root: Path, episode_id: str, query_index: int,
                             binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    receipt = validate_request_evidence(value)
    receipt["episode_binding"] = {**dict(binding or {}), "episode_id": episode_id, "query_index": query_index}
    receipt["evidence_digest"] = canonical_digest(receipt, digest_field="evidence_digest")
    directory = root / "media" / episode_id / "policy-requests"
    if root.resolve() not in directory.resolve().parents:
        raise ValueError("policy_request_path_outside_root")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{query_index:06d}.json"
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    if path.read_bytes() != raw:
        raise ValueError("policy_request_retention_readback_failed")
    return {"role": "exact_policy_request", "relative_path": path.relative_to(root).as_posix(),
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
            "query_index": query_index, "request_digest": receipt["request_digest"],
            "serialization_verified": receipt["serialization_verified"]}


class ObservedWebsocket:
    """Observe the pinned vendor's actual send bytes without replacing its codec."""
    def __init__(self, websocket: Any, *, request: Mapping[str, Any], decoder: Callable[[bytes], Any],
                 sink: Callable[[Mapping[str, Any]], None]):
        self._websocket, self._request, self._decoder, self._sink = websocket, request, decoder, sink

    def __getattr__(self, name):
        return getattr(self._websocket, name)

    def send(self, message, *args, **kwargs):
        if not isinstance(message, bytes):
            raise ValueError("openpi_policy_wire_message_not_bytes")
        evidence = capture_request(self._request, transport="openpi_websocket_msgpack_numpy",
            scientific_wire_bytes=message, decoded_wire_request=self._decoder(message))
        self._sink(evidence)
        return self._websocket.send(message, *args, **kwargs)
