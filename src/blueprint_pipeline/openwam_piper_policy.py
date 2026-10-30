"""Local OpenWAM real-PiPER EEF20 transport for the beta named-arm runtime.

This transport does not itself register a candidate. The released checkpoint
uses recorded per-arm base/TCP poses; callers must bind those frames explicitly.
Neither a PiPER robot name nor the unrelated RoboDojo dual-X5 simulator fixes
that calibration. Physics, scene cameras and policy cadence remain in the
candidate's configuration. No task/scorer state crosses this boundary.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import math
import numbers
import re
from typing import Any

import numpy as np

from .policy_request_evidence import capture_request, decode_scientific_wire
from .team_arm_kinematics import POSE_CHUNK_SCHEMA, pose_matrix, quaternion_xyzw

INTERFACE_ID = "openwam_piper_eef20.v1"
MODEL_ID = "OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper"
MODEL_REVISION = "12bfca6fb289174ab5a0f4f7eb363dc77ed265f0"
SOURCE_REVISION = "3950b6d9ea848dc980a82a89b4554af2c082544a"
WEIGHT_SHA256 = "ba9b2696624b12676610ac7bdc18882d088f819d9993b37d05d5d13f6466f004"
ARMS = ("left", "right")
WIRE_CAMERAS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
MAX_MESSAGE_BYTES = 32 * 1024 * 1024
PROMPT_PREFIX = "A video recorded from a robot's point of view executing the following instruction: "


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _vector(value: Any, width: int, code: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if (raw.shape != (width,) or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real)
                                   for x in raw.flat)):
        raise ValueError(code)
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(code)
    return array


def _transform(value: Any) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != {"position_m", "orientation_xyzw"}:
        raise ValueError("openwam_frame_transform_invalid")
    return pose_matrix(_vector(value["position_m"], 3, "openwam_frame_transform_invalid"),
                       value["orientation_xyzw"])


def _validate_binding(self: Any, *, arms: tuple[str, ...], camera_count: int) -> None:
    """Shared strict scalar/frame validation for fixed model-specific contracts."""
    if (len(self.camera_names) != camera_count or len(set(self.camera_names)) != camera_count
            or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", x)
                   for x in self.camera_names)
            or isinstance(self.control_hz, bool) or not isinstance(self.control_hz, numbers.Real)
            or not math.isfinite(self.control_hz) or not 0 < self.control_hz <= 120
            or not isinstance(self.configuration_digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.configuration_digest)
            or not isinstance(self.frame_binding_reference, str)
            or not 1 <= len(self.frame_binding_reference) <= 512
            or not self.frame_binding_reference.isprintable()
            or type(self.execution_horizon) is not int or not 1 <= self.execution_horizon <= 32):
        raise ValueError("openwam_piper_binding_invalid")
    for group in (self.policy_base_from_native_base, self.native_tcp_from_policy_tcp):
        if not isinstance(group, Mapping) or set(group) != set(arms):
            raise ValueError("openwam_frame_binding_invalid")
        for value in group.values():
            _transform(value)



@dataclass(frozen=True)
class OpenWAMPiperBinding:
    """T_A_B maps coordinates in B into A; no implicit calibration defaults.

    ``policy_base_from_native_base`` is T_policy-base_native-base and
    ``native_tcp_from_policy_tcp`` is T_native-TCP_policy-TCP, per named arm.
    ``control_hz`` is an explicitly chosen simulation rate, not a claimed
    publisher rate. The retained reference identifies its calibration/assumptions.
    """

    camera_names: tuple[str, str, str]
    control_hz: float
    configuration_digest: str
    frame_binding_reference: str
    policy_base_from_native_base: Mapping[str, Any]
    native_tcp_from_policy_tcp: Mapping[str, Any]
    execution_horizon: int = 8

    def validate(self) -> None:
        _validate_binding(self, arms=ARMS, camera_count=3)

    def identity(self) -> dict[str, Any]:
        self.validate()
        return {"interface_id": INTERFACE_ID, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "source_revision": SOURCE_REVISION, "weight_sha256": WEIGHT_SHA256,
                "action_layout": "left.xyz_rot6d_columns_open,right.xyz_rot6d_columns_open",
                "frame": "per_arm_robot_base", "prediction_horizon": 32,
                "execution_horizon": self.execution_horizon, "response_rows": 1,
                "control_hz": float(self.control_hz), "configuration_digest": self.configuration_digest,
                "frame_binding_reference": self.frame_binding_reference,
                "camera_names": list(self.camera_names),
                "policy_base_from_native_base": dict(self.policy_base_from_native_base),
                "native_tcp_from_policy_tcp": dict(self.native_tcp_from_policy_tcp),
                "normalization": "checkpoint_minmax_and_unify_once_in_upstream_server",
                "gripper_projection": "saturate_finite_opening_to_[0,1]_at_native_interface",
                "prompt_prefix": PROMPT_PREFIX,
                "canvas_height": 384, "canvas_width": 320, "inference_mode": "sync"}


def encode_observation(observation: Mapping[str, Any], binding: OpenWAMPiperBinding) -> dict[str, Any]:
    """Lossless source RGB + raw EEF20; upstream owns its L-shaped composition."""
    from PIL import Image

    binding.validate()
    if not isinstance(observation, Mapping) or set(observation) != {"images", "state", "prompt"}:
        raise ValueError("openwam_observation_keys_invalid")
    images, state, prompt = (observation[key] for key in ("images", "state", "prompt"))
    if (not isinstance(images, Mapping) or set(images) != set(binding.camera_names)
            or not isinstance(state, Mapping) or state.get("arm_names") != list(ARMS)
            or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384):
        raise ValueError("openwam_named_observation_invalid")
    groups, tools, poses = (state.get(key) for key in ("arm_joint_names", "arm_gripper_names", "arm_eef_pose"))
    if any(not isinstance(x, Mapping) or set(x) != set(ARMS) for x in (groups, tools, poses)):
        raise ValueError("openwam_arm_binding_invalid")
    names, grips = state.get("joint_names"), state.get("gripper_names")
    if (not isinstance(names, list) or len(names) != 12 or len(set(names)) != 12
            or not all(isinstance(x, str) and x for x in names)
            or not isinstance(grips, list) or len(grips) != 2 or len(set(grips)) != 2
            or not all(isinstance(x, str) and x for x in grips)):
        raise ValueError("openwam_piper_joint_or_gripper_binding_invalid")
    _vector(state.get("joint_position"), 12, "openwam_joint_state_invalid")
    closures = _vector(state.get("gripper_position"), 2, "openwam_gripper_state_invalid")
    if (closures < 0).any() or (closures > 1).any():
        raise ValueError("openwam_gripper_state_invalid")
    used: list[str] = []
    raw_state: list[float] = []
    for arm in ARMS:
        joints = groups[arm]
        if (not isinstance(joints, list) or len(joints) != 6
                or any(not isinstance(x, str) or x not in names for x in joints)
                or not isinstance(tools[arm], str) or tools[arm] not in grips):
            raise ValueError("openwam_piper_joint_or_gripper_binding_invalid")
        used.extend(joints)
        native_pose = _transform(poses[arm])
        policy_pose = (_transform(binding.policy_base_from_native_base[arm]) @ native_pose
                       @ _transform(binding.native_tcp_from_policy_tcp[arm]))
        raw_state.extend(policy_pose[:3, 3].tolist())
        # Publisher poses.py concatenates the FIRST TWO COLUMNS, not rows.
        raw_state.extend(policy_pose[:3, :2].T.reshape(-1).tolist())
        raw_state.append(1.0 - float(closures[grips.index(tools[arm])]))
    if len(set(used)) != 12 or set(used) != set(names) or set(tools.values()) != set(grips):
        raise ValueError("openwam_piper_joint_or_gripper_binding_invalid")
    wire_images = {}
    for local_name, wire_name in zip(binding.camera_names, WIRE_CAMERAS, strict=True):
        rgb = np.asarray(images[local_name])
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or not 1 <= rgb.shape[0] <= 4096 or not 1 <= rgb.shape[1] <= 4096):
            raise ValueError("openwam_rgb_invalid")
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        wire_images[wire_name] = base64.b64encode(buffer.getvalue()).decode("ascii")
    wire = {"type": "obs", "images": wire_images, "state": raw_state, "prompt": PROMPT_PREFIX + prompt}
    if len(_json(wire).encode()) > MAX_MESSAGE_BYTES:
        raise ValueError("openwam_observation_too_large")
    return wire


def decode_action(action: Any, binding: OpenWAMPiperBinding) -> dict[str, Any]:
    """Physical EEF20 -> native IK target, saturating only gripper opening.

    The publisher's min-max inverse does not bound diffusion predictions.
    Finite gripper overshoot therefore saturates at the tool's physical limits;
    the client retains the raw response and reports this projection explicitly.
    """
    binding.validate()
    raw = _vector(action, 20, "openwam_action_shape_invalid")
    positions, orientations, closures = [], [], []
    for index, arm in enumerate(ARMS):
        part = raw[index * 10:(index + 1) * 10]
        first, second = part[3:6].copy(), part[6:9].copy()
        norm = float(np.linalg.norm(first))
        if norm < 1e-8:
            raise ValueError("openwam_action_rotation_degenerate")
        first /= norm
        second -= first * float(first @ second)
        norm = float(np.linalg.norm(second))
        if norm < 1e-8:
            raise ValueError("openwam_action_rotation_degenerate")
        second /= norm
        target = np.eye(4)
        target[:3, 3] = part[:3]
        target[:3, :3] = np.column_stack((first, second, np.cross(first, second)))
        native = (np.linalg.inv(_transform(binding.policy_base_from_native_base[arm])) @ target
                  @ np.linalg.inv(_transform(binding.native_tcp_from_policy_tcp[arm])))
        positions.append(native[:3, 3].tolist())
        orientations.append(quaternion_xyzw(native[:3, :3]))
        closures.append(1.0 - float(np.clip(part[9], 0.0, 1.0)))
    return {"schema_version": POSE_CHUNK_SCHEMA, "frame": "arm_base", "arm_names": list(ARMS),
            "positions_m": [positions], "orientations_xyzw": [orientations], "gripper_closures": [closures]}


class OpenWAMPiperPolicyClient:
    """One local server action per simulator query; no implicit inference retries."""

    learned_policy = True
    policy_id = MODEL_ID
    interface_id = INTERFACE_ID
    binding_type = OpenWAMPiperBinding
    encode_observation = staticmethod(encode_observation)
    decode_action = staticmethod(decode_action)
    action_space = "named_arm_pose"
    action_chunk_rows = 1
    open_loop_horizon = 1
    require_named_arm_state = True

    def __init__(self, *, binding: OpenWAMPiperBinding, host: str = "127.0.0.1", port: int = 8848,
                 socket_factory: Callable[..., Any] | None = None, timeout_seconds: float = 180):
        binding.validate()
        if host not in {"127.0.0.1", "::1"} or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("openwam_local_endpoint_required")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, numbers.Real)
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600):
            raise ValueError("openwam_timeout_invalid")
        # Copy caller-owned mappings so a later mutation cannot change frame identity.
        data = json.loads(_json(binding.__dict__))
        data["camera_names"] = tuple(data["camera_names"])
        self.binding = self.binding_type(**data)
        self._url = f"ws://{'[' + host + ']' if ':' in host else host}:{port}"
        self._factory, self._timeout = socket_factory, float(timeout_seconds)
        self._socket = None
        self._sequence = 0
        self._ready = False
        self._poisoned = False
        self._request_sink = None
        self._request_evidence = self._inference_evidence = None
        self.candidate_policy_queried = False
        self.execution_binding_digest = None
        self.required_policy_views = tuple(binding.camera_names)
        self.action_hz = float(binding.control_hz)

    def _exchange(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._socket is None:
            raise ValueError("openwam_connection_missing")
        self._socket.send(_json(payload))
        response = self._socket.recv(timeout=self._timeout)
        if payload.get("type") == "obs":
            # Completed transport is true even if the server returns invalid JSON.
            self.candidate_policy_queried = True
            self._inference_evidence.update(server_response_received=True, wire_response_type=type(response).__name__)
            if isinstance(response, (str, bytes)):
                raw = response.encode() if isinstance(response, str) else response
                self._inference_evidence["wire_response_sha256"] = hashlib.sha256(raw).hexdigest()
        if not isinstance(response, str) or len(response.encode()) > MAX_MESSAGE_BYTES:
            raise ValueError("openwam_response_invalid")
        value = decode_scientific_wire(response.encode())
        if not isinstance(value, dict):
            raise ValueError("openwam_response_invalid")
        return value

    def _open(self) -> None:
        if self._socket is not None:
            return
        factory = self._factory
        if factory is None:
            import websockets
            # The dedicated client requirements pin 16.0, matching upstream's
            # >=16,<17 dependency. Older project-wide clients lack proxy=None.
            if not re.fullmatch(r"16\.[0-9]+(?:\.[0-9]+)?", websockets.__version__):
                raise ValueError("openwam_client_requires_websockets_16")
            from websockets.sync.client import connect
            factory = connect
        self._socket = factory(self._url, open_timeout=self._timeout, max_size=MAX_MESSAGE_BYTES, proxy=None)
        try:
            pong = self._exchange({"type": "ping"})
            if (pong.get("type") != "pong" or pong.get("blueprint_identity") != self.binding.identity()
                    or pong.get("checkpoint_bytes_verified") is not True):
                raise ValueError("openwam_server_identity_mismatch")
        except BaseException:
            self.close()
            raise

    def bind_request_evidence_sink(self, sink: Callable[[Mapping[str, Any]], Any]) -> None:
        self._request_sink = sink

    def reset(self) -> None:
        self._ready = False
        self._open()
        try:
            if self._exchange({"type": "reset"}) != {"type": "reset_ack"}:
                raise ValueError("openwam_reset_failed")
        except BaseException:
            self._poisoned = True
            self.close()
            raise
        self._sequence = 0
        self._poisoned = False
        self._ready = True
        self._request_evidence = self._inference_evidence = None
        self.candidate_policy_queried = False

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self._poisoned:
            raise ValueError("openwam_episode_reset_required")
        wire = self.encode_observation(observation, self.binding)
        if not self._ready:
            self.reset()
        self._request_evidence = capture_request(wire, transport="openwam_local_websocket_json",
                                                scientific_wire_bytes=_json(wire).encode(), decoded_wire_request=wire)
        if self._request_sink is not None:
            self._request_sink(self.last_request_evidence())  # Persist exact bytes before transmission.
        self._inference_evidence = {"request_id": f"openwam-{self._sequence + 1}",
                                    "request_sha256": self._request_evidence["scientific_wire_sha256"],
                                    "transmission_attempted": True, "server_response_received": False,
                                    "action_payload_returned": False, "actions_extracted": False}
        try:
            response = self._exchange(wire)
            self.candidate_policy_queried = True
            self._inference_evidence.update(server_response_received=True, raw_vendor_action_response=response,
                                            raw_vendor_action_response_role="decoded_openwam_response_before_native_frame_conversion",
                                            action_payload_returned="action" in response)
            if (response.get("type") != "action" or type(response.get("step")) is not int
                    or response["step"] != self._sequence + 1):
                raise ValueError("openwam_response_sequence_invalid")
            result = self.decode_action(response.get("action"), self.binding)
            self._inference_evidence["gripper_projection"] = self.gripper_projection(response["action"])
            self._sequence += 1
            self._inference_evidence["actions_extracted"] = True
            return result
        except BaseException:
            # The server may have consumed a buffered action. Never retry implicitly.
            self._poisoned = True
            self.close()
            raise

    def gripper_projection(self, action: Any) -> dict[str, Any]:
        raw = [float(action[index]) for index in (9, 19)]
        applied = np.clip(raw, 0.0, 1.0).tolist()
        return {"rule": "saturate_finite_opening_to_[0,1]_at_native_interface",
                "arm_names": list(ARMS), "raw_openings": raw, "applied_openings": applied,
                "clipped": [before != after for before, after in zip(raw, applied, strict=True)]}

    def last_request_evidence(self) -> dict[str, Any]:
        if self._request_evidence is None:
            raise ValueError("openwam_request_evidence_missing")
        return json.loads(_json(self._request_evidence))

    def last_inference_evidence(self) -> dict[str, Any]:
        if self._inference_evidence is None:
            raise ValueError("openwam_inference_evidence_missing")
        return json.loads(_json(self._inference_evidence))

    def evidence_summary(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "policy_interface_id": self.interface_id,
                "candidate_policy_queried": self.candidate_policy_queried,
                "last_inference_evidence": self.last_inference_evidence() if self._inference_evidence else None,
                "native_execution_validated": False, "catalog_registered": None,
                "catalog_registration_evidence": "owned_by_screening_admission",
                "execution_binding_digest": self.execution_binding_digest}

    def close(self) -> None:
        sock, self._socket = self._socket, None
        self._ready = False
        if sock is not None:
            sock.close()


__all__ = ["OpenWAMPiperBinding", "OpenWAMPiperPolicyClient", "encode_observation", "decode_action"]
