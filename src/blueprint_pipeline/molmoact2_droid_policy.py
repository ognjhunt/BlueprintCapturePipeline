"""Opt-in MolmoAct2 DROID: real exterior/wrist RGB and absolute joint targets.

This profile follows the original DROID 15 Hz / normalized closure contract.
The publisher's ManiSkill FR3 example has different cadence/gripper mappings.
No candidate is registered, provisioned or claimed executed by this module.
"""
from __future__ import annotations

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

from .molmoact2_droid_assets import INTERFACE_ID, MODEL_ID, MODEL_REVISION, NORM_SHA256, NORM_TAG, SOURCE_REVISION
from .molmoact2_yam_policy import MAX_MESSAGE_BYTES, _json, _array, encode_array, decode_array
from .policy_request_evidence import capture_request, decode_scientific_wire
from .adp009d_droid_action_execution import (
    DroidActionExecutionError, BLOCKER_CHUNK_SHAPE, BLOCKER_CHUNK_NONFINITE,
)

CAMERAS = ("observation/exterior_image_1_left", "observation/wrist_image_left")
WIRE_CAMERAS = ("external_cam", "wrist_cam")
JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


@dataclass(frozen=True)
class MolmoAct2DROIDBinding:
    configuration_digest: str
    camera_binding_reference: str
    execution_horizon: int = 8

    @property
    def camera_names(self) -> tuple[str, str]:
        return CAMERAS

    def validate(self) -> None:
        if (not isinstance(self.configuration_digest, str)
                or not re.fullmatch(r"sha256:[a-f0-9]{64}", self.configuration_digest)
                or not isinstance(self.camera_binding_reference, str)
                or not 1 <= len(self.camera_binding_reference) <= 512
                or not self.camera_binding_reference.isprintable()
                or type(self.execution_horizon) is not int or not 1 <= self.execution_horizon <= 15):
            raise ValueError("molmoact2_binding_invalid")

    def identity(self) -> dict[str, Any]:
        self.validate()
        return {"interface_id": INTERFACE_ID, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "source_revision": SOURCE_REVISION, "norm_stats_sha256": NORM_SHA256, "norm_tag": NORM_TAG,
                "embodiment_id": "droid_franka_robotiq_2f85", "robot_profile_id": "droid_franka_panda_robotiq_2f85_v1",
                "configuration_digest": self.configuration_digest, "joint_names": list(JOINTS),
                "camera_names": list(CAMERAS), "camera_binding_reference": self.camera_binding_reference,
                "image_shape": [360, 640, 3], "control_hz": 15,
                "prediction_horizon": 15, "execution_horizon": self.execution_horizon,
                "execution_horizon_source": "explicit_harness_configuration_not_publisher_recommendation",
                "action_layout": "panda_joint1..7,gripper_closure",
                "action_units": "absolute_joint_radians_and_closure_fraction",
                "normalization": "checkpoint_q01_q99_once_in_model_gripper_unscaled",
                "gripper_projection": "identity_raw_closure_native_bounds_and_calibration_apply",
                "num_steps": 10, "inference_action_mode": "continuous", "enable_depth_reasoning": False,
                "normalize_language": True, "enable_cuda_graph": False, "dtype": "float32"}


def encode_observation(observation: Mapping[str, Any], binding: MolmoAct2DROIDBinding) -> dict[str, Any]:
    binding.validate()
    if not isinstance(observation, Mapping) or set(observation) != {*CAMERAS,
            "observation/joint_position", "observation/gripper_position", "prompt"}:
        raise ValueError("molmoact2_observation_keys_invalid")
    prompt = observation["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
        raise ValueError("molmoact2_prompt_invalid")
    joints = _array(observation["observation/joint_position"], (7,), "molmoact2_joint_state_invalid")
    closure = _array(observation["observation/gripper_position"], (1,), "molmoact2_gripper_state_invalid")
    if not 0 <= closure[0] <= 1:
        raise ValueError("molmoact2_gripper_state_invalid")
    wire = {"instruction": prompt, "state": encode_array(np.concatenate((joints, closure))),
            "num_steps": 10, "enable_cuda_graph": False}
    for local, remote in zip(CAMERAS, WIRE_CAMERAS, strict=True):
        image = np.asarray(observation[local])
        if image.dtype != np.uint8 or image.shape != (360, 640, 3):
            raise ValueError("molmoact2_camera_profile_invalid")
        wire[remote] = encode_array(image)
    return wire


def decode_actions(actions: Any) -> np.ndarray:
    # Preserve raw robot-scale absolute values. The native execution layer owns
    # joint limits and measured gripper projection, including any clipping record.
    try:
        values = _array(actions, (15, 8), "molmoact2_action_shape_or_finite_invalid")
    except (TypeError, ValueError, OverflowError) as error:
        try:
            numeric = np.asarray(actions, dtype=float)
            nonfinite = numeric.shape == (15, 8) and not np.isfinite(numeric).all()
        except (TypeError, ValueError, OverflowError):
            nonfinite = False
        raise DroidActionExecutionError([BLOCKER_CHUNK_NONFINITE if nonfinite else BLOCKER_CHUNK_SHAPE]) from error
    return values


class MolmoAct2DROIDPolicyClient:
    learned_policy = True
    policy_id = MODEL_ID
    action_space = "joint_position"
    action_chunk_rows = 15
    action_hz = 15

    def __init__(self, *, binding: MolmoAct2DROIDBinding, host: str = "127.0.0.1", port: int = 8203,
                 timeout_seconds: float = 180, connection_factory: Callable[..., Any] | None = None):
        binding.validate()
        if host not in {"127.0.0.1", "::1"} or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("molmoact2_local_endpoint_required")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, numbers.Real)
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600):
            raise ValueError("molmoact2_timeout_invalid")
        self.binding = MolmoAct2DROIDBinding(binding.configuration_digest, binding.camera_binding_reference, binding.execution_horizon)
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

    def preflight_readiness(self) -> dict[str, Any]:
        """Reconfirm exact server identity without querying or advancing the model."""
        self.reset()
        return {**self.evidence_summary(), "identity_verified": True, "identity": self.binding.identity(),
                "candidate_inference_performed": False, "policy_state_advanced": False}

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
            try:
                raw = decode_array(response.get("actions"), (15, 8), "<f4")
            except (TypeError, ValueError, OverflowError) as error:
                raise DroidActionExecutionError([BLOCKER_CHUNK_SHAPE]) from error
            result = decode_actions(raw)
            self._sequence += 1
            self._inference_evidence.update(actions_extracted=True, gripper_projection="raw_closure_native_execution_mapping_required")
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
                "native_execution_validated": False, "catalog_registration_authority": "sealed_beta_admission",
                "execution_binding_digest": getattr(self, "execution_binding_digest", None)}

    def close(self) -> None:
        self._session = None
        self._poisoned = True
