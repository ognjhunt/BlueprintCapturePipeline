"""Dream's published bimanual EEF protocol through the beta L5 policy seam.

The wire contract was inspected in dreamlabs-client 1.0.3 (Apache-2.0),
``dream/client.py``, ``transport.py``, ``models.py`` and ``_window.py``. The
320x176 JPEG/85 preprocessing matches dreamlabs 1.0.5's MuJoCo example.
Neither package is imported: in particular no SDK recorder, credentials-file
discovery, model alias, remote IK, retry or telemetry channel is started.

Account model discovery binds *provider metadata*, never an unseen weight
digest. A successful local conformance test does not establish account access,
provider retention terms, native execution, or physical compatibility.
"""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import math
import re
import secrets
import statistics
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .policy_request_evidence import capture_request
from .safe_outbound_http import pinned_api_policy, request
from .team_arm_kinematics import POSE_CHUNK_SCHEMA, quaternion_xyzw, rotation_xyzw

INTERFACE_ID = "dreamlabs_dream1.v1"
TRANSPORT = "dreamlabs_https_json_bearer"
GATEWAY_ORIGIN = "https://api.dreamlabs.ai"
ENDPOINT_URL = GATEWAY_ORIGIN + "/v1/get_action"
METADATA_URL = "https://platform.dreamlabs.ai/v1/models"
SDK_VERSION = "1.0.3"
SDK_WHEEL_SHA256 = "sha256:2a2afaaead696a5f76de5fb3a671dbf7ac85fa69c3311deae40f698f621af5c2"
SDK_WHEEL_URL = ("https://files.pythonhosted.org/packages/1d/ed/"
                 "caf212b8d88b2079d257a78792c538c64d56358caea8588c1771739df736/"
                 "dreamlabs_client-1.0.3-py3-none-any.whl")
EXAMPLE_WHEEL_SHA256 = "sha256:f26fe4e78e209bce4435a46ae22fd9d17529840ffbc7ae36a0be49a8eed8669d"
CAMERA_ROLES = ("overhead", "wrist_l", "wrist_r")
ARM_NAMES = ("left", "right")
POLICY_CONFIG_DEFAULTS = {"action_space": "named_arm_pose", "action_hz": 20,
                          "observation_schema": "dream_modern_eef.v1", "robot": "other_embodiment",
                          "record": False, "sdk_version": SDK_VERSION, "sdk_wheel_sha256": SDK_WHEEL_SHA256}
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 8 * 1024 * 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
Transport = Callable[..., bytes]


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _json(value: Any, *, sort_keys: bool = False) -> bytes:
    # The SDK's JSON encoding uses these compact separators and ensure_ascii.
    return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), allow_nan=False).encode("utf-8")


def provider_metadata_digest(model: Mapping[str, Any]) -> str:
    """Digest the exact selected JSON model record, not the weights behind its id."""
    return _sha(_json(dict(model), sort_keys=True))


def validate_policy_config(value: Any, *, camera_names: list[str]) -> dict[str, Any]:
    """Normalize only reviewed, nonsensitive Dream configuration fields."""
    if (not isinstance(value, Mapping) or not {"model_id", "camera_roles"} <= set(value)
            or set(value) - {"model_id", "camera_roles", *POLICY_CONFIG_DEFAULTS}):
        raise ValueError("dreamlabs_policy_config_invalid")
    config = {**POLICY_CONFIG_DEFAULTS, **dict(value)}
    # Use the same runtime validator without introducing a second, looser
    # staging schema. These local placeholders are never returned or sealed.
    DreamLabsPolicySpec(policy_id="validation", endpoint_url=ENDPOINT_URL, metadata_url=METADATA_URL,
                        allowed_origin=GATEWAY_ORIGIN, model_identity_digest="sha256:" + "0" * 64,
                        embodiment_id="validation", action_chunk_rows=1, open_loop_horizon=1,
                        camera_names=camera_names, **config).validate()
    return json.loads(_json(config))


def validate_robot_contract(manifest: Mapping[str, Any], policy_config: Any) -> dict[str, Any]:
    """Require the actual native two-arm, two-tool and three-camera binding."""
    arms, grippers, cameras = (manifest.get(key) for key in ("arms", "grippers", "cameras"))
    if (manifest.get("schema_version") != "blueprint.robot_embodiment.v2" or manifest.get("form") != "bimanual_arm"
            or manifest.get("base") != {"mode": "fixed"} or manifest.get("control_hz") != 60
            or type(manifest.get("action_chunk_rows")) is not int or not 1 <= manifest["action_chunk_rows"] <= 48
            or not isinstance(arms, list) or len(arms) != 2
            or [arm.get("name") if isinstance(arm, Mapping) else None for arm in arms] != list(ARM_NAMES)
            or not isinstance(grippers, list) or len(grippers) != 2
            or not isinstance(cameras, list) or len(cameras) != 3):
        raise ValueError("dreamlabs_native_bimanual_contract_invalid")
    camera_by_name = {camera.get("name"): camera for camera in cameras if isinstance(camera, Mapping)}
    config = validate_policy_config(policy_config, camera_names=list(camera_by_name))
    gripper_by_name = {tool.get("name"): tool for tool in grippers if isinstance(tool, Mapping)}
    for arm, role in zip(arms, ("wrist_l", "wrist_r"), strict=True):
        tool = gripper_by_name.get(arm.get("gripper_name"))
        ordered = arm.get("joint_names")
        if (not isinstance(tool, Mapping) or tool.get("tool_link") != arm.get("tool_link")
                or not isinstance(ordered, list) or not 1 <= len(ordered) <= 12
                or camera_by_name[config["camera_roles"][role]].get("mount_link") != arm.get("tool_link")):
            raise ValueError("dreamlabs_native_arm_tool_camera_binding_invalid")
    if camera_by_name[config["camera_roles"]["overhead"]].get("mount_link") is not None:
        raise ValueError("dreamlabs_native_overhead_camera_binding_invalid")
    return config


def _decode(raw: bytes) -> Any:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_RESPONSE_BYTES:
        raise ValueError("dreamlabs_response_size_invalid")

    def constant(_value: str) -> None:
        raise ValueError("dreamlabs_response_nonfinite")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("dreamlabs_response_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, parse_constant=constant, object_pairs_hook=pairs)
        _json(value)  # Includes overflowed JSON exponents (1e999), not only NaN tokens.
        return value
    except (UnicodeError, TypeError, ValueError):
        raise ValueError("dreamlabs_response_json_invalid") from None


def _token(value: str | None) -> str:
    # The factory reads the canonical *_FILE lease; no env/file discovery here.
    if not isinstance(value, str) or not 0 < len(value) <= 65536 or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ValueError("dreamlabs_credential_unavailable")
    return value


def _bounded_https(*, url: str, method: str, body: bytes | None,
                   headers: Mapping[str, str], timeout_seconds: int = 15) -> bytes:
    # Separate fixed, credential-free URLs prevent token forwarding to an
    # arbitrary caller origin. The fixed-origin shared transport preserves the
    # cloud environment's HTTPS proxy without following redirects or retrying.
    approved = {ENDPOINT_URL, GATEWAY_ORIGIN + "/v1/end_session", METADATA_URL}
    if url not in approved:
        raise ValueError("dreamlabs_endpoint_not_approved")
    origin = GATEWAY_ORIGIN if url != METADATA_URL else "https://platform.dreamlabs.ai"
    response = request(url, method=method, data=body, headers=dict(headers),
                       timeout_seconds=timeout_seconds, max_response_bytes=MAX_RESPONSE_BYTES,
                       policy=pinned_api_policy(origin, max_response_bytes=MAX_RESPONSE_BYTES))
    if (response.content_type or "").split(";", 1)[0].strip().lower() != "application/json":
        raise ValueError("dreamlabs_response_content_type_invalid")
    return response.body


def _call(transport: Transport, *, url: str, method: str, body: bytes | None,
          bearer_token: str, user_agent: str, timeout_seconds: int = 15) -> Any:
    headers = {"Accept": "application/json", "Authorization": "Bearer " + bearer_token,
               "Dream-Client-Version": SDK_VERSION, "User-Agent": user_agent}
    if body is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
    try:
        raw = transport(url=url, method=method, body=body, headers=headers, timeout_seconds=timeout_seconds)
    except Exception:
        # Never carry an exception's URL, headers, credential path or provider
        # response text into logs/receipts. There is no automatic POST retry.
        raise ValueError("dreamlabs_transport_failed") from None
    return _decode(raw)


def validate_provider_model_record(model: Mapping[str, Any], *, model_id: str) -> dict[str, Any]:
    """Validate a retained SDK-compatible record without network or inference."""
    if not isinstance(model, Mapping) or model.get("id") != model_id:
        raise ValueError("dreamlabs_exact_account_model_unavailable")
    if (model.get("servable") is not True or model.get("embodiment") != "yam"
            or not isinstance(model.get("cameras"), list) or not all(isinstance(role, str) for role in model["cameras"])
            or sorted(model["cameras"]) != sorted(CAMERA_ROLES)
            or model.get("embodiment_config") is not None or model.get("parts_withheld", False) is not False
            or not all(isinstance(model.get(key), str) and model[key] for key in ("owner", "base_model"))):
        raise ValueError("dreamlabs_model_contract_unsupported")
    # SDK fields only: an unexpected extension must be reviewed, rather than
    # persisting an arbitrary provider string which could contain credentials.
    if set(model) - {"id", "owner", "base_model", "embodiment", "servable", "created_at", "cameras",
                     "embodiment_config", "parts_withheld"}:
        raise ValueError("dreamlabs_model_metadata_keys_unsupported")
    return {"model_id": model_id, "provider_model_record": model,
            "model_identity_digest": provider_metadata_digest(model), "identity_basis": "provider_metadata",
            "weights_identity_verified": False, "candidate_inference_performed": False}


def discover_model_metadata(*, model_id: str, bearer_token: str | None,
                            transport: Transport | None = None) -> dict[str, Any]:
    """Official account discovery, without inference or a Blueprint GET schema.

    This can run before a spec is sealed. Retain its selected record and digest
    for the spec's ``model_identity_digest``; never substitute another id.
    """
    if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
        raise ValueError("dreamlabs_model_id_invalid")
    answer = _call(transport or _bounded_https, url=METADATA_URL, method="GET", body=None,
                   bearer_token=_token(bearer_token), user_agent="dream-client/" + SDK_VERSION + " " + secrets.token_hex(4))
    if not isinstance(answer, list) or len(answer) > 4096 or any(not isinstance(row, dict) for row in answer):
        raise ValueError("dreamlabs_models_response_invalid")
    matches = [row for row in answer if row.get("id") == model_id]
    if len(matches) != 1:
        raise ValueError("dreamlabs_exact_account_model_unavailable")
    return validate_provider_model_record(matches[0], model_id=model_id)


@dataclass(frozen=True)
class DreamLabsPolicySpec:
    policy_id: str
    endpoint_url: str
    metadata_url: str
    allowed_origin: str
    model_identity_digest: str
    embodiment_id: str
    action_chunk_rows: int
    open_loop_horizon: int
    model_id: str
    camera_names: list[str] = field(default_factory=list)
    camera_roles: dict[str, str] = field(default_factory=dict)
    action_space: str = "named_arm_pose"
    action_hz: int = 20
    policy_interface_id: str = INTERFACE_ID
    observation_schema: str = "dream_modern_eef.v1"
    robot: str = "other_embodiment"
    record: bool = False
    sdk_version: str = SDK_VERSION
    sdk_wheel_sha256: str = SDK_WHEEL_SHA256

    def validate(self) -> None:
        if (self.endpoint_url != ENDPOINT_URL or self.metadata_url != METADATA_URL
                or self.allowed_origin != GATEWAY_ORIGIN):
            raise ValueError("dreamlabs_endpoint_not_approved")
        if any(not isinstance(value, str) or not _ID.fullmatch(value) for value in (self.policy_id, self.embodiment_id)):
            raise ValueError("dreamlabs_policy_identity_invalid")
        if not isinstance(self.model_id, str) or not _MODEL_ID.fullmatch(self.model_id):
            raise ValueError("dreamlabs_model_id_invalid")
        if not isinstance(self.model_identity_digest, str) or not _DIGEST.fullmatch(self.model_identity_digest):
            raise ValueError("dreamlabs_provider_metadata_digest_invalid")
        if (self.policy_interface_id != INTERFACE_ID or self.action_space != "named_arm_pose"
                or type(self.action_hz) is not int or self.action_hz != 20
                or self.observation_schema != "dream_modern_eef.v1" or self.robot != "other_embodiment"
                or self.record is not False or self.sdk_version != SDK_VERSION or self.sdk_wheel_sha256 != SDK_WHEEL_SHA256):
            raise ValueError("dreamlabs_protocol_contract_unsupported")
        if (type(self.action_chunk_rows) is not int or not 1 <= self.action_chunk_rows <= 48
                or type(self.open_loop_horizon) is not int or not 1 <= self.open_loop_horizon <= self.action_chunk_rows):
            raise ValueError("dreamlabs_chunk_contract_invalid")
        if (not isinstance(self.camera_names, list) or len(self.camera_names) != 3
                or any(not isinstance(name, str) or not _ID.fullmatch(name) for name in self.camera_names)
                or len(set(self.camera_names)) != 3 or not isinstance(self.camera_roles, dict)
                or set(self.camera_roles) != set(CAMERA_ROLES)
                or any(not isinstance(name, str) for name in self.camera_roles.values())
                or sorted(self.camera_roles.values()) != sorted(self.camera_names)):
            raise ValueError("dreamlabs_camera_role_binding_invalid")


def _vector(value: Any, size: int, code: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(code) from None
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(code)
    return result


def _encode_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    state = observation.get("state")
    prompt = observation.get("prompt")
    if (not isinstance(state, Mapping) or not isinstance(state.get("arm_names"), list)
            or state["arm_names"] != list(ARM_NAMES)
            or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384):
        raise ValueError("dreamlabs_named_arm_observation_invalid")
    names, joints = state.get("joint_names"), state.get("joint_position")
    gripper_names, closures = state.get("gripper_names"), state.get("gripper_position")
    if (not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or not isinstance(gripper_names, list)
            or len(gripper_names) != 2 or not all(isinstance(name, str) for name in gripper_names)
            or len(set(gripper_names)) != 2):
        raise ValueError("dreamlabs_named_joint_binding_invalid")
    _vector(joints, len(names), "dreamlabs_joint_state_invalid")
    closures = _vector(closures, 2, "dreamlabs_gripper_state_invalid")
    if (closures < 0).any() or (closures > 1).any():
        raise ValueError("dreamlabs_gripper_state_invalid")
    arm_joints, poses, tools = (state.get(key) for key in ("arm_joint_names", "arm_eef_pose", "arm_gripper_names"))
    if any(not isinstance(value, Mapping) or set(value) != set(ARM_NAMES) for value in (arm_joints, poses, tools)):
        raise ValueError("dreamlabs_named_arm_binding_invalid")
    wire: dict[str, Any] = {"prompt": prompt}
    seen: set[str] = set()
    for arm in ARM_NAMES:
        ordered, pose = arm_joints[arm], poses[arm]
        tool = tools[arm]
        if (not isinstance(ordered, list) or not 1 <= len(ordered) <= 12
                or any(not isinstance(name, str) or name not in names or name in seen for name in ordered)
                or len(set(ordered)) != len(ordered) or not isinstance(tool, str) or tool not in gripper_names
                or not isinstance(pose, Mapping) or set(pose) != {"position_m", "orientation_xyzw"}):
            raise ValueError("dreamlabs_named_arm_binding_invalid")
        seen.update(ordered)
        xyz = _vector(pose["position_m"], 3, "dreamlabs_tcp_position_invalid")
        rotation = rotation_xyzw(pose["orientation_xyzw"])
        opening = 1.0 - float(closures[gripper_names.index(tool)])
        # Dream's non-YAM mode masks joints. Its YAM-trained slots still have
        # width six; actual arm joints remain local to native IK, never padded
        # or truncated and presented as another robot's measured joint state.
        wire[f"state.{arm}_arm"] = [0.0] * 6
        wire[f"state.{arm}_gripper"] = [opening]
        wire[f"state.{arm}_eef_xyz"] = xyz.tolist()
        wire[f"state.{arm}_eef_rot6"] = rotation[:, :2].T.reshape(-1).tolist()
        wire[f"state.{arm}_grip"] = [opening]
    if set(tools.values()) != set(gripper_names) or seen != set(names):
        raise ValueError("dreamlabs_named_arm_binding_invalid")
    return wire


def _decode_actions(value: Any, maximum_rows: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("per_key"), Mapping):
        raise ValueError("dreamlabs_eef_actions_missing")
    per_key = value["per_key"]
    arrays: dict[tuple[str, str], np.ndarray] = {}
    rows = None
    for arm in ARM_NAMES:
        for part, width in (("eef_xyz", 3), ("eef_rot6", 6), ("grip", 1)):
            try:
                array = np.asarray(per_key.get(f"action.{arm}_{part}"), dtype=float)
            except (TypeError, ValueError):
                raise ValueError("dreamlabs_eef_action_shape_invalid") from None
            if (array.ndim != 2 or array.shape[1] != width or not 1 <= array.shape[0] <= maximum_rows
                    or not np.isfinite(array).all() or rows is not None and array.shape[0] != rows):
                raise ValueError("dreamlabs_eef_action_shape_invalid")
            rows = array.shape[0]
            arrays[(arm, part)] = array
        grip = arrays[(arm, "grip")]
        if (grip < 0).any() or (grip > 1).any():
            raise ValueError("dreamlabs_eef_gripper_out_of_range")
    orientations = []
    for row in range(rows or 0):
        pair = []
        for arm in ARM_NAMES:
            rot6 = arrays[(arm, "eef_rot6")][row]
            first, second = rot6[:3], rot6[3:]
            norm = float(np.linalg.norm(first))
            if norm < 1e-8:
                raise ValueError("dreamlabs_eef_rotation_degenerate")
            first = first / norm
            second = second - first * (first @ second)
            norm = float(np.linalg.norm(second))
            if norm < 1e-8:
                raise ValueError("dreamlabs_eef_rotation_degenerate")
            second /= norm
            pair.append(quaternion_xyzw(np.column_stack((first, second, np.cross(first, second)))))
        orientations.append(pair)
    return {"schema_version": POSE_CHUNK_SCHEMA, "frame": "arm_base", "arm_names": list(ARM_NAMES),
            "positions_m": [[arrays[(arm, "eef_xyz")][row].tolist() for arm in ARM_NAMES] for row in range(rows or 0)],
            "orientations_xyzw": orientations,
            "gripper_closures": [[1 - float(arrays[(arm, "grip")][row, 0]) for arm in ARM_NAMES]
                                 for row in range(rows or 0)]}


class DreamLabsPolicyClient:
    """One explicit account model, one session and a real 5 Hz camera history."""

    learned_policy = True
    interface_id = INTERFACE_ID
    transport_name = TRANSPORT
    action_hz = 20
    observation_hz = 5
    require_named_arm_state = True

    def __init__(self, *, spec: DreamLabsPolicySpec, bearer_token: str | None,
                 transport: Transport | None = None) -> None:
        spec.validate()
        self._spec, self._token, self._transport = spec, _token(bearer_token), transport or _bounded_https
        self._user_agent = "dream-client/" + SDK_VERSION + " " + secrets.token_hex(4)
        self.policy_id, self.action_space = spec.policy_id, spec.action_space
        self.action_chunk_rows, self.open_loop_horizon = spec.action_chunk_rows, spec.open_loop_horizon
        self.candidate_policy_queried = False
        self._session_id: str | None = None
        self._known_session_ids: set[str] = set()
        self._session_unknown = False
        self._closed = False
        self._faulted = False
        self._metadata: dict[str, Any] | None = None
        self._generation = 0
        self._request_evidence: dict[str, Any] | None = None
        self._inference_evidence: dict[str, Any] | None = None
        self._sink: Callable[[Mapping[str, Any]], None] | None = None
        self._frames: deque[dict[str, Any]] = deque(maxlen=9)
        self._recent: deque[float] = deque(maxlen=8)
        self._due: float | None = None
        self._last_frames: list[dict[str, Any]] = []
        self._release_status = "not_needed"

    def _active(self) -> None:
        if self._closed or self._faulted:
            raise ValueError("dreamlabs_episode_closed_or_faulted")

    def _call(self, url: str, method: str, body: bytes | None, *, timeout_seconds: int = 15) -> Any:
        return _call(self._transport, url=url, method=method, body=body, bearer_token=self._token,
                     user_agent=self._user_agent, timeout_seconds=timeout_seconds)

    def discover_model_metadata(self) -> dict[str, Any]:
        self._active()
        try:
            record = discover_model_metadata(model_id=self._spec.model_id, bearer_token=self._token, transport=self._transport)
        except ValueError:
            self._faulted = True
            raise
        if record["model_identity_digest"] != self._spec.model_identity_digest:
            self._faulted = True
            raise ValueError("dreamlabs_provider_metadata_drift")
        self._generation += 1
        self._metadata = {**record, "identity_sha256": record["model_identity_digest"],
                          "policy_interface_id": INTERFACE_ID, "sdk_version": SDK_VERSION,
                          "sdk_wheel_sha256": SDK_WHEEL_SHA256, "sdk_wheel_url": SDK_WHEEL_URL,
                          "preprocessing_example_wheel_sha256": EXAMPLE_WHEEL_SHA256,
                          "observation_schema": self._spec.observation_schema,
                          "sdk_recording_enabled": False, "provider_inference_retention": "not_established", "additional_telemetry_channels": 0}
        return json.loads(_json(self._metadata))

    @property
    def server_metadata(self) -> dict[str, Any]:
        return json.loads(_json(self._metadata)) if self._metadata is not None else self.discover_model_metadata()

    def preflight_readiness(self) -> dict[str, Any]:
        metadata = self.discover_model_metadata()
        return {"identity_verified": True, "transport": TRANSPORT, "readiness_method": "dream_account_models_without_inference",
                "connection_generation": self._generation, "candidate_policy_queried": False,
                "candidate_inference_performed": False, "policy_state_advanced": False,
                "prior_candidate_policy_query_observed": self.candidate_policy_queried,
                "last_inference_evidence": None, "server_metadata": metadata}

    def observe_frame(self, observation: Mapping[str, Any], simulation_time_s: float) -> bool:
        """Keep an actual camera tick when due; no query-time frame duplication."""
        from PIL import Image

        self._active()
        _encode_state(observation)
        if (isinstance(simulation_time_s, bool) or not isinstance(simulation_time_s, (int, float))
                or not math.isfinite(simulation_time_s) or simulation_time_s < 0
                or self._recent and simulation_time_s < self._recent[-1]):
            raise ValueError("dreamlabs_frame_timestamp_invalid")
        images = observation.get("images")
        if not isinstance(images, Mapping) or set(images) != set(self._spec.camera_names):
            raise ValueError("dreamlabs_camera_binding_invalid")
        checked = {}
        for role in CAMERA_ROLES:
            image = np.asarray(images[self._spec.camera_roles[role]])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not 0 < image.shape[0] <= 4096 or not 0 < image.shape[1] <= 4096:
                raise ValueError("dreamlabs_camera_image_invalid")
            checked[role] = image
        # Same timestamp is the same simulator tick, never a second observation.
        if self._recent and simulation_time_s == self._recent[-1]:
            return False
        recent = [*self._recent, float(simulation_time_s)][-8:]
        gaps = [later - earlier for earlier, later in itertools.pairwise(recent)]
        slack = min(statistics.median(gaps) / 2, 0.1) if gaps else 0.0
        if self._due is not None and simulation_time_s < self._due - slack:
            self._recent.append(float(simulation_time_s))
            return False
        encoded = {}
        for role, image in checked.items():
            output = io.BytesIO()
            Image.fromarray(image).resize((320, 176), Image.Resampling.BILINEAR).save(output, format="JPEG", quality=85)
            encoded[role] = output.getvalue()
        anchor = self._due if self._due is not None and simulation_time_s < self._due + 0.1 else simulation_time_s
        self._due = float(anchor) + 0.2
        self._recent.append(float(simulation_time_s))
        self._frames.append({"simulation_time_s": float(simulation_time_s), "jpeg": encoded,
                             "source": {role: {"source_rgb_shape": list(image.shape),
                                                "source_rgb_sha256": _sha(np.ascontiguousarray(image).tobytes())}
                                        for role, image in checked.items()}})
        return True

    def bind_request_evidence_sink(self, sink: Callable[[Mapping[str, Any]], None] | None) -> None:
        self._sink = sink

    def last_request_evidence(self) -> dict[str, Any]:
        if self._request_evidence is None:
            raise ValueError("dreamlabs_request_evidence_missing")
        return json.loads(_json(self._request_evidence))

    def last_policy_input_frames(self) -> list[dict[str, Any]]:
        """Actual JPEG-decoded RGB for lossless retention, distinct from render RGB."""
        return [{**frame, "rgb": frame["rgb"].copy()} for frame in self._last_frames]

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        from PIL import Image

        self._request_evidence, self._inference_evidence = None, None
        self._last_frames = []
        self._active()
        if not self._frames:
            raise ValueError("dreamlabs_real_camera_history_missing")
        wire_observation = _encode_state(observation)
        metadata = self.discover_model_metadata()
        wire_observation.update({"video." + role: [base64.b64encode(frame["jpeg"][role]).decode("ascii") for frame in self._frames]
                                 for role in CAMERA_ROLES})
        wire_observation["embodiment"] = self._spec.robot
        request = {"checkpoint": self._spec.model_id, "session_id": self._session_id,
                   "reset": False, "observation": wire_observation}
        body = _json(request)
        if len(body) > MAX_REQUEST_BYTES:
            raise ValueError("dreamlabs_request_size_invalid")
        self._request_evidence = capture_request(request, transport=TRANSPORT,
                                                 scientific_wire_bytes=body, decoded_wire_request=json.loads(body))
        prepared_frames = []
        for frame in self._frames:
            for role in CAMERA_ROLES:
                jpeg = frame["jpeg"][role]
                with Image.open(io.BytesIO(jpeg)) as image:
                    rgb = np.asarray(image.convert("RGB")).copy()
                prepared_frames.append({"camera_name": self._spec.camera_roles[role], "provider_role": role,
                                          "simulation_time_s": frame["simulation_time_s"], "rgb": rgb,
                                          "image_sha256": _sha(jpeg), "decoded_rgb_sha256": _sha(rgb.tobytes()),
                                          **frame["source"][role],
                                          "preprocessing": {"resize_width": 320, "resize_height": 176,
                                                            "resampling": "bilinear", "encoding": "jpeg", "jpeg_quality": 85}})
        if self._sink is not None:
            self._sink(self._request_evidence)  # Storage refusal prevents transport.
        # Only expose model-input media after retention succeeds and a send
        # will be attempted. A sink refusal must not become a transmitted frame.
        self._last_frames = [{**frame, "transmission_attempted": True, "server_response_received": False}
                             for frame in prepared_frames]
        self._inference_evidence = {"transport": TRANSPORT, "transport_connection_generation": self._generation,
                                    "server_identity_sha256": metadata["identity_sha256"], "identity_basis": "provider_metadata",
                                    "weights_identity_verified": False, "request_body_sha256": _sha(body),
                                    "transmission_attempted": True, "server_response_received": False,
                                    "action_payload_returned": False, "actions_extracted": False,
                                    "sdk_recording_enabled": False, "provider_inference_retention": "not_established", "additional_telemetry_channels": 0,
                                    "action_hz": 20, "observation_hz": 5,
                                    "camera_history_timestamps_s": [frame["simulation_time_s"] for frame in self._frames],
                                    "policy_input_frames": [{key: value for key, value in frame.items() if key != "rgb"}
                                                            for frame in self._last_frames]}
        try:
            response = self._call(ENDPOINT_URL, "POST", body)
        except ValueError:
            self._faulted, self._session_unknown = True, True
            raise
        self.candidate_policy_queried = True
        self._inference_evidence["server_response_received"] = True
        for frame in [*self._last_frames, *self._inference_evidence["policy_input_frames"]]:
            frame["server_response_received"] = True
        status = response.get("status") if isinstance(response, dict) else None
        returned = response.get("session_id") if isinstance(response, dict) else None
        valid_session = isinstance(returned, str) and 0 < len(returned) <= 256 and not any(c.isspace() for c in returned)
        if valid_session:
            # A replacement/error does not establish that the old hosted seat
            # ended. Keep both IDs until each release is independently confirmed.
            self._known_session_ids.add(returned)
        if not isinstance(status, dict) or not isinstance(status.get("code"), str):
            self._faulted, self._session_unknown = True, True
            raise ValueError("dreamlabs_response_envelope_invalid")
        code = status["code"]
        self._inference_evidence["provider_status_code"] = code if re.fullmatch(r"DA-[0-9]{4}", code) else "invalid"
        if code not in {"DA-0000", "DA-0001"}:
            self._faulted = True
            if valid_session:
                self._session_id = returned
            else:
                self._session_unknown = True
            raise ValueError("dreamlabs_provider_refused")
        if not valid_session:
            self._faulted, self._session_unknown = True, True
            raise ValueError("dreamlabs_session_id_missing")
        previous = self._session_id
        self._session_id = returned
        if previous is not None and (returned != previous or code == "DA-0001"):
            self._faulted = True
            raise ValueError("dreamlabs_session_replaced_mid_episode")
        self._inference_evidence["action_payload_returned"] = "actions" in response
        try:
            actions = _decode_actions(response.get("actions"), self.action_chunk_rows)
        except ValueError:
            self._faulted = True
            raise
        # Retain the genuine numeric vendor action fields before conversion.
        raw = {key: response["actions"]["per_key"][key] for key in
               (f"action.{arm}_{part}" for arm in ARM_NAMES for part in ("eef_xyz", "eef_rot6", "grip"))}
        self._inference_evidence.update(actions_extracted=True, raw_vendor_action_response={"per_key": raw},
                                        raw_vendor_action_response_digest=_sha(_json({"per_key": raw}, sort_keys=True)),
                                        raw_vendor_action_response_role="genuine_dream_eef_fields_before_normalization")
        return actions

    def last_inference_evidence(self) -> dict[str, Any]:
        if self._inference_evidence is None:
            raise ValueError("dreamlabs_inference_evidence_missing")
        return json.loads(_json(self._inference_evidence))

    def _release(self) -> None:
        had_sessions = bool(self._known_session_ids)
        # A first response supplies one ID; the first changed/error response
        # faults the episode before another infer, so there can be at most two.
        if len(self._known_session_ids) > 2:
            self._release_status = "outcome_unknown"
            raise ValueError("dreamlabs_session_outcome_unknown")
        self._release_status = "unconfirmed" if had_sessions else "not_needed"
        for session_id in sorted(self._known_session_ids):
            try:
                response = self._call(GATEWAY_ORIGIN + "/v1/end_session", "POST",
                                      _json({"checkpoint": self._spec.model_id, "session_id": session_id}), timeout_seconds=2)
            except ValueError:
                continue  # Still attempt the other known release, never infer.
            if (isinstance(response, dict) and isinstance(response.get("status"), dict)
                    and response["status"].get("code") == "DA-0000"):
                self._known_session_ids.remove(session_id)
                if self._session_id == session_id:
                    self._session_id = None
        if self._known_session_ids:
            raise ValueError("dreamlabs_session_release_unconfirmed")
        if self._session_unknown:
            # Releasing the last known id does not prove a lost response did
            # not mint a replacement session. Never mark that outcome closed.
            self._session_id, self._release_status = None, "outcome_unknown"
            raise ValueError("dreamlabs_session_outcome_unknown")
        self._session_id = None
        self._release_status = "confirmed" if had_sessions else "not_needed"

    def reset(self) -> None:
        if self._closed:
            raise ValueError("dreamlabs_episode_closed_or_faulted")
        self._faulted = True
        self._frames.clear()
        self._recent.clear()
        self._due, self._request_evidence, self._inference_evidence = None, None, None
        self._last_frames = []
        self._release()
        self._faulted = False
        self.candidate_policy_queried = False

    reset_episode = reset

    def close(self) -> None:
        if self._closed:
            if self._known_session_ids or self._session_unknown:
                raise ValueError("dreamlabs_session_release_unconfirmed")
            return
        try:
            self._release()
        finally:
            self._frames.clear()
            self._recent.clear()
            self._due = None
            self._closed, self._token = True, ""

    def evidence_summary(self) -> dict[str, Any]:
        return {"transport": TRANSPORT, "identity_verified": self._metadata is not None,
                "server_metadata": json.loads(_json(self._metadata)), "connection_generation": self._generation,
                "sdk_recording_enabled": False, "provider_inference_retention": "not_established", "additional_telemetry_channels": 0,
                "session_release_status": self._release_status,
                "known_sessions_pending_release": len(self._known_session_ids), "session_outcome_unknown": self._session_unknown,
                "last_inference_evidence": self.last_inference_evidence() if self._inference_evidence is not None else None}


__all__ = ["DreamLabsPolicyClient", "DreamLabsPolicySpec", "discover_model_metadata", "provider_metadata_digest",
           "validate_policy_config", "validate_robot_contract", "validate_provider_model_record"]
