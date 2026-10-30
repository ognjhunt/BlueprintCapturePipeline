"""OpenWAM single-FR3 EEF10 adapter; nominal simulation tooling is not publisher calibration."""
from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import io
import math
from typing import Any

import numpy as np

from .openwam_piper_policy import (
    MAX_MESSAGE_BYTES, SOURCE_REVISION, OpenWAMPiperPolicyClient, _json, _transform, _vector, _validate_binding,
)
from .team_arm_kinematics import POSE_CHUNK_SCHEMA, quaternion_xyzw

INTERFACE_ID = "openwam_franka_eef10.v1"
MODEL_ID = "OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka"
MODEL_REVISION = "e75522eb38f9b4148d5ecb985ab80c490ddb3854"
WEIGHT_SHA256 = "76e53e963525d9eebdb70e5285908c6497082270de66f9227874844e96bc43bb"
ARMS = ("arm",)
WIRE_CAMERAS = ("head_camera", "left_wrist_camera")


@dataclass(frozen=True)
class OpenWAMFrankaBinding:
    camera_names: tuple[str, str]
    control_hz: float
    configuration_digest: str
    frame_binding_reference: str
    policy_base_from_native_base: Mapping[str, Any]
    native_tcp_from_policy_tcp: Mapping[str, Any]
    execution_horizon: int = 8

    def validate(self) -> None:
        _validate_binding(self, arms=ARMS, camera_count=2)
        if not math.isclose(self.control_hz, 15, rel_tol=0, abs_tol=0):
            raise ValueError("openwam_franka_binding_invalid")

    def identity(self) -> dict[str, Any]:
        self.validate()
        return {"interface_id": INTERFACE_ID, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_REVISION, "weight_sha256": WEIGHT_SHA256,
            "action_layout": "arm.xyz_rot6d_columns_signed_opening", "frame": "robot_base",
            "prediction_horizon": 32, "execution_horizon": self.execution_horizon, "response_rows": 1,
            "control_hz": 15.0, "configuration_digest": self.configuration_digest,
            "frame_binding_reference": self.frame_binding_reference, "camera_names": list(self.camera_names),
            "policy_base_from_native_base": dict(self.policy_base_from_native_base),
            "native_tcp_from_policy_tcp": dict(self.native_tcp_from_policy_tcp),
            "normalization": "checkpoint_minmax_and_unify_once_in_upstream_server",
            "gripper_projection": "saturate_finite_signed_opening_to_[-1,1]_then_closure=(1-opening)/2",
            "prompt_prefix": "", "source_image_width": 640, "source_image_height": 480,
            "canvas_height": 384, "canvas_width": 320, "third_camera": "configured_null_structural_black_padding",
            "inference_mode": "sync"}


def encode_observation(observation: Mapping[str, Any], binding: OpenWAMFrankaBinding) -> dict[str, Any]:
    from PIL import Image

    binding.validate()
    if not isinstance(observation, Mapping) or set(observation) != {"images", "state", "prompt"}:
        raise ValueError("openwam_observation_keys_invalid")
    images, state, prompt = (observation[key] for key in ("images", "state", "prompt"))
    if (not isinstance(images, Mapping) or set(images) != set(binding.camera_names)
            or not isinstance(state, Mapping) or state.get("arm_names") != ["arm"]
            or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384):
        raise ValueError("openwam_franka_named_observation_invalid")
    groups, tools, poses = (state.get(key) for key in ("arm_joint_names", "arm_gripper_names", "arm_eef_pose"))
    if any(not isinstance(value, Mapping) or set(value) != {"arm"} for value in (groups, tools, poses)):
        raise ValueError("openwam_arm_binding_invalid")
    names, grips = state.get("joint_names"), state.get("gripper_names")
    if (not isinstance(names, list) or len(names) != 7 or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != 7 or groups["arm"] != names or not isinstance(grips, list) or len(grips) != 1
            or not isinstance(grips[0], str) or not grips[0] or tools["arm"] != grips[0]):
        raise ValueError("openwam_franka_joint_or_gripper_binding_invalid")
    _vector(state.get("joint_position"), 7, "openwam_joint_state_invalid")
    closure = _vector(state.get("gripper_position"), 1, "openwam_gripper_state_invalid")[0]
    if not 0 <= closure <= 1:
        raise ValueError("openwam_gripper_state_invalid")
    pose = (_transform(binding.policy_base_from_native_base["arm"]) @ _transform(poses["arm"])
            @ _transform(binding.native_tcp_from_policy_tcp["arm"]))
    raw = [*pose[:3, 3].tolist(), *pose[:3, :2].T.reshape(-1).tolist(), 1.0 - 2.0 * float(closure)]
    wire_images = {}
    for local, wire in zip(binding.camera_names, WIRE_CAMERAS, strict=True):
        image = np.asarray(images[local])
        if image.dtype != np.uint8 or image.shape != (480, 640, 3):
            raise ValueError("openwam_franka_rgb_invalid")
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        wire_images[wire] = base64.b64encode(buffer.getvalue()).decode("ascii")
    result = {"type": "obs", "images": wire_images, "state": raw, "prompt": prompt}
    if len(_json(result).encode()) > MAX_MESSAGE_BYTES:
        raise ValueError("openwam_observation_too_large")
    return result


def decode_action(action: Any, binding: OpenWAMFrankaBinding) -> dict[str, Any]:
    binding.validate()
    raw = _vector(action, 10, "openwam_action_shape_invalid")
    first, second = raw[3:6].copy(), raw[6:9].copy()
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
    target[:3, 3] = raw[:3]
    target[:3, :3] = np.column_stack((first, second, np.cross(first, second)))
    native = (np.linalg.inv(_transform(binding.policy_base_from_native_base["arm"])) @ target
              @ np.linalg.inv(_transform(binding.native_tcp_from_policy_tcp["arm"])))
    return {"schema_version": POSE_CHUNK_SCHEMA, "frame": "arm_base", "arm_names": ["arm"],
        "positions_m": [[native[:3, 3].tolist()]], "orientations_xyzw": [[quaternion_xyzw(native[:3, :3])]],
        "gripper_closures": [[(1.0 - float(np.clip(raw[9], -1, 1))) / 2.0]]}


class OpenWAMFrankaPolicyClient(OpenWAMPiperPolicyClient):
    """Same bounded local transport; distinct model handshake, frame and tool semantics."""

    policy_id = MODEL_ID
    interface_id = INTERFACE_ID
    binding_type = OpenWAMFrankaBinding
    encode_observation = staticmethod(encode_observation)
    decode_action = staticmethod(decode_action)

    def gripper_projection(self, action: Any) -> dict[str, Any]:
        raw = float(action[9])
        applied = float(np.clip(raw, -1, 1))
        return {"rule": "saturate_finite_signed_opening_to_[-1,1]_then_closure=(1-opening)/2",
            "arm_names": ["arm"], "raw_openings": [raw], "applied_openings": [applied],
            "clipped": [raw != applied], "native_closures": [(1-applied)/2]}
