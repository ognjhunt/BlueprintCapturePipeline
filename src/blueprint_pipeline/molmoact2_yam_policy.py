"""Opt-in local MolmoAct2 YAM adapter for named-arm beta observations.

The publisher owns q01/q99 normalization and returns absolute joint angles.
Only the two independent opening fractions become native closure fractions.
This module does not register, provision or claim to have executed a candidate.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import http.client
import json
import math
import numbers
import re
from typing import Any

import numpy as np

from .molmoact2_yam_assets import INTERFACE_ID, MODEL_ID, MODEL_REVISION, NORM_SHA256, NORM_TAG, SOURCE_REVISION
from .policy_request_evidence import capture_request, decode_scientific_wire

ARMS = ("left", "right")
JOINTS = tuple(f"{side}_joint{i}" for side in ARMS for i in range(1, 7))
GRIPPERS = ("left_gripper", "right_gripper")
WIRE_CAMERAS = ("top_cam", "left_cam", "right_cam")
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _array(value: Any, shape: tuple[int, ...], code: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if (raw.shape != shape or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real)
                                 for x in raw.flat)):
        raise ValueError(code)
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(code)
    return result


def encode_array(array: np.ndarray) -> dict[str, Any]:
    """The public json-numpy 2.1.1 wire format, without global json patching."""
    array = np.ascontiguousarray(array)
    return {"__numpy__": base64.b64encode(array.tobytes()).decode(),
            "dtype": array.dtype.str, "shape": list(array.shape)}


def decode_array(value: Any, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    if (not isinstance(value, dict) or set(value) != {"__numpy__", "dtype", "shape"}
            or value["shape"] != list(shape) or value["dtype"] != np.dtype(dtype).str
            or not isinstance(value["__numpy__"], str)):
        raise ValueError("molmoact2_wire_array_invalid")
    expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
    if len(value["__numpy__"]) != 4 * ((expected_bytes + 2) // 3):
        raise ValueError("molmoact2_wire_array_invalid")
    raw = base64.b64decode(value["__numpy__"], validate=True)
    if len(raw) != expected_bytes:
        raise ValueError("molmoact2_wire_array_invalid")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


@dataclass(frozen=True)
class MolmoAct2YAMBinding:
    camera_names: tuple[str, str, str]
    configuration_digest: str
    camera_binding_reference: str
    execution_horizon: int = 25

    def validate(self) -> None:
        if (len(self.camera_names) != 3 or len(set(self.camera_names)) != 3
                or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
                       for name in self.camera_names)
                or not isinstance(self.configuration_digest, str)
                or not re.fullmatch(r"sha256:[a-f0-9]{64}", self.configuration_digest)
                or not isinstance(self.camera_binding_reference, str)
                or not 1 <= len(self.camera_binding_reference) <= 512
                or not self.camera_binding_reference.isprintable()
                or type(self.execution_horizon) is not int or not 1 <= self.execution_horizon <= 30):
            raise ValueError("molmoact2_binding_invalid")

    def identity(self) -> dict[str, Any]:
        self.validate()
        return {"interface_id": INTERFACE_ID, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "source_revision": SOURCE_REVISION, "norm_stats_sha256": NORM_SHA256, "norm_tag": NORM_TAG,
                "embodiment_id": "i2rt_yam_bimanual", "tool_profile": "LINEAR_4310",
                "configuration_digest": self.configuration_digest,
                "camera_names": list(self.camera_names), "camera_binding_reference": self.camera_binding_reference,
                "image_shape": [360, 640, 3], "control_hz": 30,
                "prediction_horizon": 30, "execution_horizon": self.execution_horizon,
                "action_layout": "left_joint1..6,left_open,right_joint1..6,right_open",
                "action_units": "absolute_joint_radians_and_opening_fraction",
                "normalization": "checkpoint_q01_q99_once_in_model_grippers_unscaled",
                "gripper_projection": "saturate_finite_opening_to_[0,1]_then_one_minus_opening",
                "num_steps": 10, "inference_action_mode": "continuous", "enable_depth_reasoning": False,
                "normalize_language": True, "enable_cuda_graph": False, "dtype": "float32"}


def encode_observation(observation: Mapping[str, Any], binding: MolmoAct2YAMBinding) -> dict[str, Any]:
    binding.validate()
    if not isinstance(observation, Mapping) or set(observation) != {"images", "state", "prompt"}:
        raise ValueError("molmoact2_observation_keys_invalid")
    images, state, prompt = (observation[key] for key in ("images", "state", "prompt"))
    if (not isinstance(images, Mapping) or set(images) != set(binding.camera_names)
            or not isinstance(state, Mapping) or state.get("arm_names") != list(ARMS)
            or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384):
        raise ValueError("molmoact2_named_observation_invalid")
    names, grippers = state.get("joint_names"), state.get("gripper_names")
    if (not isinstance(names, list) or len(names) != 12 or set(names) != set(JOINTS)
            or not isinstance(grippers, list) or len(grippers) != 2 or set(grippers) != set(GRIPPERS)
            or state.get("arm_joint_names") != {side: list(JOINTS[i * 6:(i + 1) * 6]) for i, side in enumerate(ARMS)}
            or state.get("arm_gripper_names") != dict(zip(ARMS, GRIPPERS, strict=True))):
        raise ValueError("molmoact2_yam_joint_binding_invalid")
    positions = _array(state.get("joint_position"), (12,), "molmoact2_joint_state_invalid")
    closures = _array(state.get("gripper_position"), (2,), "molmoact2_gripper_state_invalid")
    if ((closures < 0) | (closures > 1)).any():
        raise ValueError("molmoact2_gripper_state_invalid")
    raw_state = []
    for index, side in enumerate(ARMS):
        raw_state.extend(float(positions[names.index(name)]) for name in JOINTS[index * 6:(index + 1) * 6])
        raw_state.append(1.0 - float(closures[grippers.index(f"{side}_gripper")]))
    wire = {"instruction": prompt, "state": encode_array(np.asarray(raw_state, dtype=np.float32)),
            "num_steps": 10, "enable_cuda_graph": False}
    for local, remote in zip(binding.camera_names, WIRE_CAMERAS, strict=True):
        image = np.asarray(images[local])
        if image.dtype != np.uint8 or image.shape != (360, 640, 3):
            raise ValueError("molmoact2_camera_profile_invalid")
        wire[remote] = encode_array(image)
    return wire


def decode_actions(actions: Any, *, joint_names: list[str], gripper_names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    raw = _array(actions, (30, 14), "molmoact2_action_shape_or_finite_invalid")
    if len(joint_names) != 12 or set(joint_names) != set(JOINTS) or set(gripper_names) != set(GRIPPERS) or len(gripper_names) != 2:
        raise ValueError("molmoact2_action_binding_invalid")
    columns = {name: raw[:, i + (1 if i >= 6 else 0)] for i, name in enumerate(JOINTS)}
    openings = raw[:, [6, 13]]
    applied = np.clip(openings, 0.0, 1.0)
    closures = {name: 1.0 - applied[:, i] for i, name in enumerate(GRIPPERS)}
    result = np.column_stack([*(columns[name] for name in joint_names), *(closures[name] for name in gripper_names)])
    return result, {"rule": "saturate_finite_opening_to_[0,1]_then_one_minus_opening",
                    "arm_names": list(ARMS), "raw_openings": openings.tolist(),
                    "applied_openings": applied.tolist(), "clipped_count": int((openings != applied).sum())}


class MolmoAct2YAMPolicyClient:
    learned_policy = True
    policy_id = MODEL_ID
    action_space = "joint_position"
    action_chunk_rows = 30
    action_hz = 30
    require_named_arm_state = True

    def __init__(self, *, binding: MolmoAct2YAMBinding, host: str = "127.0.0.1", port: int = 8202,
                 timeout_seconds: float = 180, connection_factory: Callable[..., Any] | None = None):
        binding.validate()
        if host not in {"127.0.0.1", "::1"} or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("molmoact2_local_endpoint_required")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, numbers.Real)
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600):
            raise ValueError("molmoact2_timeout_invalid")
        self.binding = MolmoAct2YAMBinding(tuple(binding.camera_names), binding.configuration_digest,
                                         binding.camera_binding_reference, binding.execution_horizon)
        self.required_policy_views = self.binding.camera_names
        self.open_loop_horizon = self.binding.execution_horizon
        self._host, self._port, self._timeout = host, port, float(timeout_seconds)
        self._factory = connection_factory or http.client.HTTPConnection
        self._session = None
        self._sequence = 0
        self._poisoned = False
        self._request_sink = None
        self._request_evidence = self._inference_evidence = None
        self.candidate_policy_queried = False

    def _exchange(self, method: str, path: str, wire: dict[str, Any] | None = None) -> dict[str, Any]:
        connection = self._factory(self._host, self._port, timeout=self._timeout)
        try:
            headers = {"Content-Type": "application/json"}
            if self._session:
                headers.update({"X-Blueprint-Session": self._session,
                                "X-Blueprint-Sequence": str(self._sequence + 1)})
            connection.request(method, path, body=_json(wire) if wire is not None else None, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_MESSAGE_BYTES + 1)
            if path == "/act" and method == "POST":
                self.candidate_policy_queried = True
                self._inference_evidence.update(server_response_received=True,
                                                wire_response_sha256=hashlib.sha256(raw).hexdigest())
            if response.status != 200 or len(raw) > MAX_MESSAGE_BYTES:
                raise ValueError("molmoact2_response_invalid")
            result = decode_scientific_wire(raw)
            if not isinstance(result, dict):
                raise ValueError("molmoact2_response_invalid")
            return result
        finally:
            connection.close()

    def reset(self) -> None:
        self._session = None
        self._poisoned = True
        response = self._exchange("POST", "/reset", {})
        if (response.get("identity") != self.binding.identity() or response.get("checkpoint_bytes_verified") is not True
                or not isinstance(response.get("session"), str) or not re.fullmatch(r"[a-f0-9]{32}", response["session"])):
            raise ValueError("molmoact2_server_identity_mismatch")
        self._session, self._sequence, self._poisoned = response["session"], 0, False
        self._request_evidence = self._inference_evidence = None
        self.candidate_policy_queried = False

    def bind_request_evidence_sink(self, sink: Callable[[Mapping[str, Any]], Any]) -> None:
        self._request_sink = sink

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._poisoned:
            raise ValueError("molmoact2_episode_reset_required")
        wire = encode_observation(observation, self.binding)
        if self._session is None:
            self.reset()
        self._request_evidence = capture_request(wire, transport="molmoact2_local_json_numpy_http",
                                                scientific_wire_bytes=_json(wire), decoded_wire_request=wire)
        if self._request_sink:
            self._request_sink(self.last_request_evidence())
        self._inference_evidence = {"session": self._session, "sequence": self._sequence + 1,
                                    "request_sha256": self._request_evidence["scientific_wire_sha256"],
                                    "transmission_attempted": True, "server_response_received": False,
                                    "action_payload_returned": False, "actions_extracted": False}
        try:
            response = self._exchange("POST", "/act", wire)
            self._inference_evidence.update(raw_vendor_action_response=response, action_payload_returned="actions" in response)
            if (response.get("session") != self._session or type(response.get("sequence")) is not int
                    or response["sequence"] != self._sequence + 1):
                raise ValueError("molmoact2_response_sequence_invalid")
            raw = decode_array(response.get("actions"), (30, 14), "<f4")
            result, projection = decode_actions(raw, joint_names=observation["state"]["joint_names"],
                                               gripper_names=observation["state"]["gripper_names"])
            self._sequence += 1
            self._inference_evidence.update(actions_extracted=True, gripper_projection=projection)
            return result
        except BaseException:
            self._poisoned = True
            self._session = None
            raise

    def last_request_evidence(self) -> dict[str, Any]:
        if self._request_evidence is None:
            raise ValueError("molmoact2_request_evidence_missing")
        return json.loads(_json(self._request_evidence))

    def last_inference_evidence(self) -> dict[str, Any]:
        if self._inference_evidence is None:
            raise ValueError("molmoact2_inference_evidence_missing")
        return json.loads(_json(self._inference_evidence))

    def evidence_summary(self) -> dict[str, Any]:
        return {"policy_id": MODEL_ID, "policy_interface_id": INTERFACE_ID,
                "candidate_policy_queried": self.candidate_policy_queried,
                "last_inference_evidence": self.last_inference_evidence() if self._inference_evidence else None,
                "native_execution_validated": False, "catalog_registered": False}

    def close(self) -> None:
        self._session = None
        self._poisoned = True
