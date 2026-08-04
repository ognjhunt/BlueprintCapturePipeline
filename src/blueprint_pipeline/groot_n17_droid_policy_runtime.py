"""Identity-bound GR00T N1.7 DROID client for the Franka simulator loop.

This is deliberately a thin translation around NVIDIA's public ``PolicyClient``.
It does not load a model, allocate a GPU, or add another simulator.  The adapter
converts Blueprint's existing DROID observation into GR00T's nested modality
format and returns the joint-position-plus-gripper chunk used by the existing
Franka runner.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .droid_policy_bridge import DROID_OPEN_LOOP_HORIZON
from .policy_ranking_thesis import canonical_sha256


MODEL_ID = "nvidia/GR00T-N1.7-DROID"
EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
GROOT_SOURCE_REVISION = "b9955401d50c92a29258732e3ad6ccd579f1bdc0"
CHECKPOINT_REVISION = "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"
LANGUAGE_KEY = "annotation.language.language_instruction"
VIDEO_KEYS = ("exterior_image_1_left", "wrist_image_left")
STATE_KEYS = ("eef_9d", "gripper_position", "joint_position")
ACTION_KEYS = ("gripper_position", "joint_position")


def _is_git_sha1(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _delta_indices(config: Any) -> tuple[int, ...]:
    value = getattr(config, "delta_indices", None)
    if value is None and isinstance(config, Mapping):
        value = config.get("delta_indices")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("groot_modality_delta_indices_missing")
    rows = tuple(int(item) for item in value)
    if not rows:
        raise ValueError("groot_modality_delta_indices_empty")
    return rows


def _modality_keys(config: Any) -> tuple[str, ...]:
    value = getattr(config, "modality_keys", None)
    if value is None and isinstance(config, Mapping):
        value = config.get("modality_keys")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("groot_modality_keys_missing")
    return tuple(str(item) for item in value)


def droid_eef_9d(*, position_m: Sequence[float], rotation_row_major: Sequence[float]) -> Any:
    """Mirror NVIDIA's DROID XYZ plus corrected rotation-6D convention."""

    import numpy as np

    position = np.asarray(position_m, dtype=np.float64)
    rotation = np.asarray(rotation_row_major, dtype=np.float64).reshape(3, 3)
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise ValueError("groot_droid_eef_pose_invalid")
    correction = np.asarray(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    corrected = rotation @ correction
    return np.concatenate((position, corrected[:2, :].reshape(6))).astype(np.float32)


@dataclass(frozen=True)
class GrootN17DroidPolicySpec:
    model_id: str = MODEL_ID
    embodiment_tag: str = EMBODIMENT_TAG
    groot_source_revision: str = GROOT_SOURCE_REVISION
    checkpoint_revision: str = CHECKPOINT_REVISION
    open_loop_horizon: int = DROID_OPEN_LOOP_HORIZON

    def validate(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError("groot_model_id_mismatch")
        if self.embodiment_tag != EMBODIMENT_TAG:
            raise ValueError("groot_embodiment_tag_mismatch")
        if not _is_git_sha1(self.groot_source_revision):
            raise ValueError("groot_source_revision_invalid")
        if not _is_git_sha1(self.checkpoint_revision):
            raise ValueError("groot_checkpoint_revision_invalid")
        if self.open_loop_horizon != DROID_OPEN_LOOP_HORIZON:
            raise ValueError("groot_open_loop_horizon_mismatch")

    def identity(self) -> dict[str, Any]:
        self.validate()
        result = {
            "model_id": self.model_id,
            "embodiment_tag": self.embodiment_tag,
            "groot_source_revision": self.groot_source_revision,
            "checkpoint_revision": self.checkpoint_revision,
            "open_loop_horizon": self.open_loop_horizon,
        }
        result["identity_sha256"] = canonical_sha256(result)
        return result


def validate_worker_identity_receipt(
    receipt: Mapping[str, Any], *, expected: GrootN17DroidPolicySpec
) -> dict[str, Any]:
    """Require materialized worker evidence; a server endpoint cannot self-identify."""

    expected_identity = expected.identity()
    blockers: list[str] = []
    if receipt.get("status") != "verified":
        blockers.append("worker_receipt_not_verified")
    for key in (
        "model_id",
        "embodiment_tag",
        "groot_source_revision",
        "checkpoint_revision",
    ):
        if receipt.get(key) != expected_identity[key]:
            blockers.append(f"worker_receipt_{key}_mismatch")
    checkpoint_digest = str(receipt.get("checkpoint_files_sha256") or "")
    environment_digest = str(receipt.get("environment_lock_sha256") or "")
    for label, value in (
        ("checkpoint_files", checkpoint_digest),
        ("environment_lock", environment_digest),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            blockers.append(f"worker_receipt_{label}_sha256_invalid")
    if blockers:
        raise ValueError(";".join(sorted(blockers)))
    return dict(receipt)


def _resize_with_pad(image: Any, *, height: int = 180, width: int = 320) -> Any:
    import numpy as np
    from PIL import Image

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("groot_droid_image_invalid")
    scale = min(width / source.shape[1], height / source.shape[0])
    resized_width = max(1, round(source.shape[1] * scale))
    resized_height = max(1, round(source.shape[0] * scale))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = np.asarray(
        Image.fromarray(source).resize((resized_width, resized_height), resampling),
        dtype=np.uint8,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


class GrootN17DroidPolicyClient:
    """Adapt NVIDIA's ZMQ policy client to Blueprint's existing DROID runner."""

    learned_policy = True
    policy_id = MODEL_ID
    action_space = "joint_position"

    def __init__(
        self,
        *,
        spec: GrootN17DroidPolicySpec,
        worker_identity_receipt: Mapping[str, Any],
        host: str,
        port: int = 5555,
        api_token: str | None = None,
        timeout_ms: int = 15000,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        spec.validate()
        if not str(host).strip() or not 1 <= int(port) <= 65535:
            raise ValueError("groot_policy_server_endpoint_invalid")
        if client_factory is None:
            try:
                from gr00t.policy.server_client import PolicyClient
            except ImportError as exc:  # pragma: no cover - GPU runtime only
                raise RuntimeError("groot_policy_client_not_installed") from exc
            client_factory = PolicyClient
        self._spec = spec
        self._worker_receipt = validate_worker_identity_receipt(
            worker_identity_receipt, expected=spec
        )
        self._client = client_factory(
            host=str(host),
            port=int(port),
            timeout_ms=int(timeout_ms),
            api_token=api_token,
            strict=False,
        )
        if self._client.ping() is not True:
            raise ValueError("groot_policy_server_unreachable")
        modality = self._client.get_modality_config()
        if not isinstance(modality, Mapping):
            raise ValueError("groot_modality_config_invalid")
        for name in ("video", "state", "action", "language"):
            if name not in modality:
                raise ValueError(f"groot_modality_missing:{name}")
        if _modality_keys(modality["video"]) != VIDEO_KEYS:
            raise ValueError("groot_droid_video_keys_mismatch")
        if _modality_keys(modality["state"]) != STATE_KEYS:
            raise ValueError("groot_droid_state_keys_mismatch")
        if set(_modality_keys(modality["action"])) != set(ACTION_KEYS + ("eef_9d",)):
            raise ValueError("groot_droid_action_keys_mismatch")
        language_keys = _modality_keys(modality["language"])
        if language_keys != (LANGUAGE_KEY,):
            raise ValueError("groot_droid_language_key_mismatch")
        self._video_delta_indices = _delta_indices(modality["video"])
        if self._video_delta_indices not in {(0,), (-15, 0)}:
            raise ValueError("groot_droid_video_history_unsupported")
        self.action_chunk_rows = len(_delta_indices(modality["action"]))
        if self.action_chunk_rows < DROID_OPEN_LOOP_HORIZON:
            raise ValueError("groot_droid_action_chunk_too_short")
        self._frames: deque[dict[str, Any]] = deque(
            maxlen=max(-min(self._video_delta_indices), 0) + 1
        )

    def infer(self, observation: Mapping[str, Any]) -> Any:
        import numpy as np

        exterior = _resize_with_pad(observation.get("observation/exterior_image_1_left"))
        wrist = _resize_with_pad(observation.get("observation/wrist_image_left"))
        joints = np.asarray(observation.get("observation/joint_position"), dtype=np.float32)
        gripper = np.asarray(observation.get("observation/gripper_position"), dtype=np.float32)
        eef = np.asarray(observation.get("observation/eef_9d"), dtype=np.float32)
        prompt = str(observation.get("prompt") or "").strip()
        if joints.shape != (7,) or gripper.shape != (1,) or eef.shape != (9,) or not prompt:
            raise ValueError("groot_droid_observation_state_invalid")
        self._frames.append({"exterior": exterior, "wrist": wrist})
        if self._video_delta_indices == (0,):
            exterior_video = exterior[None, None, ...]
            wrist_video = wrist[None, None, ...]
        else:
            historical = self._frames[0]
            exterior_video = np.stack((historical["exterior"], exterior))[None, ...]
            wrist_video = np.stack((historical["wrist"], wrist))[None, ...]
        request = {
            "video": {
                VIDEO_KEYS[0]: exterior_video,
                VIDEO_KEYS[1]: wrist_video,
            },
            "state": {
                "eef_9d": eef[None, None, ...],
                "gripper_position": gripper[None, None, ...],
                "joint_position": joints[None, None, ...],
            },
            "language": {LANGUAGE_KEY: [[prompt]]},
        }
        response = self._client.get_action(request)
        if not isinstance(response, Sequence) or len(response) != 2:
            raise ValueError("groot_policy_response_invalid")
        actions = response[0]
        if not isinstance(actions, Mapping):
            raise ValueError("groot_policy_actions_invalid")
        joint_chunk = np.asarray(actions.get("joint_position"), dtype=float)
        gripper_chunk = np.asarray(actions.get("gripper_position"), dtype=float)
        expected_joint_shape = (1, self.action_chunk_rows, 7)
        expected_gripper_shape = (1, self.action_chunk_rows, 1)
        if (
            joint_chunk.shape != expected_joint_shape
            or gripper_chunk.shape != expected_gripper_shape
        ):
            raise ValueError("groot_policy_action_shape_mismatch")
        chunk = np.concatenate((joint_chunk[0], gripper_chunk[0]), axis=1)
        if not np.isfinite(chunk).all():
            raise ValueError("groot_policy_action_nonfinite")
        chunk[:, 7] = (chunk[:, 7] > 0.5).astype(float)
        return chunk

    def evidence_summary(self) -> dict[str, Any]:
        return {
            "transport": "nvidia_groot_zmq_msgpack",
            "identity_verified": True,
            "policy_identity": self._spec.identity(),
            "worker_identity_receipt": self._worker_receipt,
            "video_delta_indices": list(self._video_delta_indices),
            "action_chunk_rows": self.action_chunk_rows,
        }


__all__ = [
    "CHECKPOINT_REVISION",
    "EMBODIMENT_TAG",
    "GROOT_SOURCE_REVISION",
    "GrootN17DroidPolicyClient",
    "GrootN17DroidPolicySpec",
    "MODEL_ID",
    "droid_eef_9d",
    "validate_worker_identity_receipt",
]
