"""Identity-bound GR00T N1.7 DROID client for the Franka simulator loop.

This is deliberately a thin translation around Blueprint's protocol-pinned
wire-only implementation of NVIDIA's public ``PolicyClient``.
It does not load a model, allocate a GPU, or add another simulator.  The adapter
converts Blueprint's existing DROID observation into GR00T's nested modality
format and returns the joint-position-plus-gripper chunk used by the existing
Franka runner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MODEL_ID = "nvidia/GR00T-N1.7-DROID"
EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
GROOT_SOURCE_REVISION = "b9955401d50c92a29258732e3ad6ccd579f1bdc0"
CHECKPOINT_REVISION = "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"
LANGUAGE_KEY = "annotation.language.language_instruction"
VIDEO_KEYS = ("exterior_image_1_left", "wrist_image_left")
STATE_KEYS = ("eef_9d", "gripper_position", "joint_position")
ACTION_KEYS = ("gripper_position", "joint_position")
# The released DROID bridge executes eight rows from every policy chunk.  Keep
# this module flat-bundle safe: it is copied beside the worker scripts rather
# than imported as part of ``blueprint_pipeline`` on the rented host.
DROID_OPEN_LOOP_HORIZON = 8
DROID_ACTION_CHUNK_ROWS = 40
# The exact checkpoint root ``processor_config.json`` is what
# ``AutoProcessor.from_pretrained(model_dir)`` loads.  At frozen revision
# 05e7cc97... that 2,833-byte Git blob (55b4d74b...) declares one current
# video frame.  The older ``experiment_cfg/config.yaml`` retained beside the
# checkpoint says ``[-15, 0]``, but the server never loads that training
# artifact.  Requiring its two-frame history makes the identity handshake
# deterministically refuse the genuine checkpoint before the first query.
FROZEN_VIDEO_DELTA_INDICES = (0,)
FROZEN_STATE_DELTA_INDICES = (0,)
FROZEN_ACTION_DELTA_INDICES = tuple(range(DROID_ACTION_CHUNK_ROWS))
FROZEN_LANGUAGE_DELTA_INDICES = (0,)
# The checkpoint's exact root statistics file is content-bound by the worker
# identity receipt.  Its DROID state statistics are base-frame Cartesian
# coordinates, not scene/world coordinates.  Enforce the observed translation
# support before a query so a future frame regression cannot spend an episode
# asking the frozen checkpoint to act from an impossible state.
CHECKPOINT_STATISTICS_SOURCE = (
    "nvidia/GR00T-N1.7-DROID@05e7cc97e40dbd33b0890c35cc0214fcb0547ab5:"
    "statistics.json:oxe_droid_relative_eef_relative_joint.state.eef_9d"
)
CHECKPOINT_STATISTICS_SHA256 = (
    "127832f7df25cda15da4ba6be81737f96b65673d0f892f9fc1bce1bc062fa858"
)
CHECKPOINT_STATISTICS_GIT_BLOB_SHA1 = "03e76c7666bafe2e31fcc2320ee5ffcdddc6d675"
DROID_EEF_POSITION_OBSERVED_MIN_M = (
    -0.1557805985212326,
    -0.8236568570137024,
    -0.24001094698905945,
)
DROID_EEF_POSITION_OBSERVED_MAX_M = (
    0.8575563430786133,
    0.8196876049041748,
    1.0066224336624146,
)
EEF_FRAME_PROVENANCE_SCHEMA_VERSION = "droid_eef_frame_provenance.v1"
EEF_FRAME_PROVENANCE_KEY = "observation/eef_9d_frame_provenance"
EEF_FRAME_BODY_NAME = "panda_link8"
EEF_FRAME_BODY_SOURCE = (
    "droid-dataset/droid@ba46d4af805bce44e6a40cff10ed094ee5090ab8:"
    "config/panda/franka_panda.yaml:ee_link_name"
)
EEF_FRAME_STATE_SOURCE = (
    "live_panda_link8_pose_world_transformed_by_live_robot_root_pose"
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe_vendor_response(value: Any) -> Any:
    """Retain the decoded vendor response without admitting nonfinite values."""

    import math

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_vendor_response(item)
            for key, item in value.items()
        }
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite_float": repr(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe_vendor_response(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe_vendor_response(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe_vendor_response(item())
    return {"unsupported_type": f"{type(value).__module__}.{type(value).__name__}"}


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


def _validated_eef_frame_provenance(
    value: Any, *, eef_position_m: Any
) -> dict[str, Any]:
    """Require explicit frame proof instead of guessing it from coordinate size."""

    import numpy as np

    if not isinstance(value, Mapping):
        raise ValueError("groot_droid_eef_frame_provenance_invalid")
    try:
        provenance = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("groot_droid_eef_frame_provenance_invalid") from exc
    position = np.asarray(provenance.get("position_robot_root_m"), dtype=np.float64)
    if (
        provenance.get("schema_version") != EEF_FRAME_PROVENANCE_SCHEMA_VERSION
        or provenance.get("state_frame") != "robot_root"
        or provenance.get("body_name") != EEF_FRAME_BODY_NAME
        or provenance.get("body_source") != EEF_FRAME_BODY_SOURCE
        or provenance.get("state_source") != EEF_FRAME_STATE_SOURCE
        or position.shape != (3,)
        or not np.isfinite(position).all()
        or not np.allclose(position, eef_position_m, rtol=0.0, atol=1.0e-6)
        or provenance.get("provenance_digest")
        != "sha256:"
        + _canonical_sha256(
            {
                key: item
                for key, item in provenance.items()
                if key != "provenance_digest"
            }
        )
    ):
        raise ValueError("groot_droid_eef_frame_provenance_invalid")
    return provenance


def _eef_position_support_evidence(position_m: Any) -> dict[str, Any]:
    """Describe checkpoint-range clipping without mislabeling it as incompatibility."""

    import numpy as np

    position = np.asarray(position_m, dtype=np.float64)
    minimum = np.asarray(DROID_EEF_POSITION_OBSERVED_MIN_M, dtype=np.float64)
    maximum = np.asarray(DROID_EEF_POSITION_OBSERVED_MAX_M, dtype=np.float64)
    below = np.maximum(minimum - position, 0.0)
    above = np.maximum(position - maximum, 0.0)
    return {
        "position_m": position.tolist(),
        "minimum_m": minimum.tolist(),
        "maximum_m": maximum.tolist(),
        "inside_checkpoint_observed_extrema": bool(
            np.all(below == 0.0) and np.all(above == 0.0)
        ),
        "below_minimum_by_m": below.tolist(),
        "above_maximum_by_m": above.tolist(),
        "maximum_excess_m": float(max(np.max(below), np.max(above))),
        "source": CHECKPOINT_STATISTICS_SOURCE,
        "source_sha256": CHECKPOINT_STATISTICS_SHA256,
        "source_git_blob_sha1": CHECKPOINT_STATISTICS_GIT_BLOB_SHA1,
        "frozen_processor_use_percentiles": True,
        "frozen_processor_clip_outliers": True,
        "query_blocking": False,
        "interpretation": (
            "inside_checkpoint_observed_extrema"
            if np.all(below == 0.0) and np.all(above == 0.0)
            else "outside_checkpoint_observed_extrema_clipped_by_frozen_processor"
        ),
    }


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
        result["identity_sha256"] = _canonical_sha256(result)
        return result


def validate_worker_identity_receipt(
    receipt: Mapping[str, Any], *, expected: GrootN17DroidPolicySpec
) -> dict[str, Any]:
    """Require materialized worker evidence; a server endpoint cannot self-identify."""

    try:  # flat provider-bundle layout
        from adp009d_groot_worker_identity import expected_checkpoint_content_binding
    except ModuleNotFoundError:  # repository package
        from .adp009d_groot_worker_identity import expected_checkpoint_content_binding

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
    if receipt.get("checkpoint_content_manifest_digest") != (
        expected_checkpoint_content_binding()["file_manifest_digest"]
    ):
        blockers.append("worker_receipt_checkpoint_content_manifest_mismatch")
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
                from groot_n17_wire_client import GrootN17WirePolicyClient
            except ModuleNotFoundError:  # repository package
                from .groot_n17_wire_client import GrootN17WirePolicyClient
            client_factory = GrootN17WirePolicyClient
        self._spec = spec
        self._worker_receipt = validate_worker_identity_receipt(
            worker_identity_receipt, expected=spec
        )
        client = client_factory(
            host=str(host),
            port=int(port),
            timeout_ms=int(timeout_ms),
            api_token=api_token,
            strict=False,
        )
        try:
            if client.ping() is not True:
                raise ValueError("groot_policy_server_unreachable")
            modality = client.get_modality_config()
            if not isinstance(modality, Mapping):
                raise ValueError("groot_modality_config_invalid")
            for name in ("video", "state", "action", "language"):
                if name not in modality:
                    raise ValueError(f"groot_modality_missing:{name}")
            if _modality_keys(modality["video"]) != VIDEO_KEYS:
                raise ValueError("groot_droid_video_keys_mismatch")
            if _modality_keys(modality["state"]) != STATE_KEYS:
                raise ValueError("groot_droid_state_keys_mismatch")
            if set(_modality_keys(modality["action"])) != set(
                ACTION_KEYS + ("eef_9d",)
            ):
                raise ValueError("groot_droid_action_keys_mismatch")
            language_keys = _modality_keys(modality["language"])
            if language_keys != (LANGUAGE_KEY,):
                raise ValueError("groot_droid_language_key_mismatch")
            self._video_delta_indices = _delta_indices(modality["video"])
            self._state_delta_indices = _delta_indices(modality["state"])
            self._action_delta_indices = _delta_indices(modality["action"])
            self._language_delta_indices = _delta_indices(modality["language"])
            for name, actual, expected in (
                (
                    "video",
                    self._video_delta_indices,
                    FROZEN_VIDEO_DELTA_INDICES,
                ),
                (
                    "state",
                    self._state_delta_indices,
                    FROZEN_STATE_DELTA_INDICES,
                ),
                (
                    "action",
                    self._action_delta_indices,
                    FROZEN_ACTION_DELTA_INDICES,
                ),
                (
                    "language",
                    self._language_delta_indices,
                    FROZEN_LANGUAGE_DELTA_INDICES,
                ),
            ):
                if actual != expected:
                    raise ValueError(f"groot_droid_{name}_delta_indices_mismatch")
            self.action_chunk_rows = DROID_ACTION_CHUNK_ROWS
        except BaseException:
            closer = getattr(client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
            raise
        self._client = client
        self._last_inference_evidence: dict[str, Any] | None = None
        self.candidate_policy_queried = False

    def infer(self, observation: Mapping[str, Any]) -> Any:
        import numpy as np

        # Never let a refused observation inherit a prior episode's successful
        # response receipt when the warm client is reused.
        self._last_inference_evidence = None
        exterior = _resize_with_pad(observation.get("observation/exterior_image_1_left"))
        wrist = _resize_with_pad(observation.get("observation/wrist_image_left"))
        joints = np.asarray(observation.get("observation/joint_position"), dtype=np.float32)
        gripper = np.asarray(observation.get("observation/gripper_position"), dtype=np.float32)
        eef = np.asarray(observation.get("observation/eef_9d"), dtype=np.float32)
        prompt = str(observation.get("prompt") or "").strip()
        if joints.shape != (7,) or gripper.shape != (1,) or eef.shape != (9,) or not prompt:
            raise ValueError("groot_droid_observation_state_invalid")
        if not (
            np.isfinite(joints).all()
            and np.isfinite(gripper).all()
            and np.isfinite(eef).all()
        ):
            raise ValueError("groot_droid_observation_state_nonfinite")
        frame_provenance = _validated_eef_frame_provenance(
            observation.get(EEF_FRAME_PROVENANCE_KEY), eef_position_m=eef[:3]
        )
        support_evidence = _eef_position_support_evidence(eef[:3])
        # The frozen processor uses percentile normalization and clips outliers.
        # Its statistics are empirical normalization data, not a declared API
        # support boundary.  Preserve an excursion as unqualified diagnostic
        # evidence, while the explicit frame provenance above remains the
        # fail-closed protection against accidentally sending scene/world XYZ.
        self._last_inference_evidence = {
            "server_response_received": False,
            "action_payload_returned": False,
            "actions_extracted": False,
            "eef_frame_provenance": frame_provenance,
            "eef_position_observed_support": support_evidence,
        }
        exterior_video = exterior[None, None, ...]
        wrist_video = wrist[None, None, ...]
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
        retained_response = _json_safe_vendor_response(response)
        self._last_inference_evidence.update({
            "server_response_received": True,
            "wire_response_type": type(response).__name__,
            "raw_vendor_action_response": retained_response,
            "raw_vendor_action_response_digest": (
                "sha256:"
                + _canonical_sha256(
                    {"raw_vendor_action_response": retained_response}
                )
            ),
            "raw_vendor_action_response_role": (
                "genuine_decoded_vendor_wire_response_before_candidate_validation"
            ),
            "action_payload_returned": True,
            "actions_extracted": False,
        })
        self.candidate_policy_queried = True
        if not isinstance(response, Sequence) or len(response) != 2:
            raise ValueError("groot_policy_response_invalid")
        actions = response[0]
        if not isinstance(actions, Mapping):
            raise ValueError("groot_policy_actions_invalid")
        joint_chunk = np.asarray(actions.get("joint_position"), dtype=float)
        gripper_chunk = np.asarray(actions.get("gripper_position"), dtype=float)
        eef_chunk = np.asarray(actions.get("eef_9d"), dtype=float)
        expected_joint_shape = (1, self.action_chunk_rows, 7)
        expected_gripper_shape = (1, self.action_chunk_rows, 1)
        expected_eef_shape = (1, self.action_chunk_rows, 9)
        if (
            joint_chunk.shape != expected_joint_shape
            or gripper_chunk.shape != expected_gripper_shape
            or eef_chunk.shape != expected_eef_shape
        ):
            raise ValueError("groot_policy_action_shape_mismatch")
        native_chunk = np.concatenate(
            (eef_chunk[0], gripper_chunk[0], joint_chunk[0]), axis=1
        )
        chunk = np.concatenate((joint_chunk[0], gripper_chunk[0]), axis=1)
        if not np.isfinite(native_chunk).all():
            raise ValueError("groot_policy_action_nonfinite")
        native_components = {
            "eef_9d": eef_chunk[0].tolist(),
            "gripper_position": gripper_chunk[0].tolist(),
            "joint_position": joint_chunk[0].tolist(),
        }
        self._last_inference_evidence.update(
            {
                "native_action_chunk_shape": [self.action_chunk_rows, 17],
                "native_action_component_order": [
                    "eef_9d",
                    "gripper_position",
                    "joint_position",
                ],
                "native_action_components": native_components,
                "native_action_chunk_sha256": _canonical_sha256(native_components),
                "execution_projection": (
                    "joint_position_plus_binarized_gripper_first_8_rows;"
                    "eef_9d_retained_not_executed"
                ),
                "joint_position_server_output": (
                    "absolute_after_checkpoint_relative_action_decode"
                ),
                "actions_extracted": True,
            }
        )
        chunk[:, 7] = (chunk[:, 7] > 0.5).astype(float)
        return chunk

    def last_inference_evidence(self) -> dict[str, Any]:
        if self._last_inference_evidence is None:
            raise ValueError("groot_policy_inference_evidence_missing")
        return json.loads(json.dumps(self._last_inference_evidence, allow_nan=False))

    def reset(self) -> None:
        """Reset the remote policy without carrying state across episodes."""

        response = self._client.reset()
        if not isinstance(response, Mapping):
            raise ValueError("groot_policy_reset_response_invalid")

    def preflight_readiness(self) -> dict[str, Any]:
        """Reconfirm live transport/identity for the next warm-session episode."""

        prior_query_observed = bool(
            self.candidate_policy_queried or self._last_inference_evidence is not None
        )
        if self._client.ping() is not True:
            raise ValueError("groot_policy_server_unreachable")
        self.reset()
        return {
            "identity_verified": True,
            "transport": "nvidia_groot_zmq_msgpack",
            "readiness_method": "live_ping_and_reset_without_inference",
            "candidate_policy_queried": False,
            "candidate_inference_performed": False,
            "policy_state_advanced": False,
            "last_inference_evidence": None,
            "prior_candidate_policy_query_observed": prior_query_observed,
            "policy_identity": self._spec.identity(),
            "worker_identity_receipt_digest": self._worker_receipt.get(
                "receipt_digest"
            ),
        }

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()

    def evidence_summary(self) -> dict[str, Any]:
        return {
            "transport": "nvidia_groot_zmq_msgpack",
            "identity_verified": True,
            "policy_identity": self._spec.identity(),
            "worker_identity_receipt": self._worker_receipt,
            "video_delta_indices": list(self._video_delta_indices),
            "state_delta_indices": list(self._state_delta_indices),
            "action_delta_indices": list(self._action_delta_indices),
            "language_delta_indices": list(self._language_delta_indices),
            "video_history_source": "current_policy_query_observation_only",
            "eef_position_observed_support": {
                "minimum_m": list(DROID_EEF_POSITION_OBSERVED_MIN_M),
                "maximum_m": list(DROID_EEF_POSITION_OBSERVED_MAX_M),
                "source": CHECKPOINT_STATISTICS_SOURCE,
                "source_sha256": CHECKPOINT_STATISTICS_SHA256,
                "source_git_blob_sha1": CHECKPOINT_STATISTICS_GIT_BLOB_SHA1,
                "enforced_before_policy_query": False,
                "frame_provenance_enforced_before_policy_query": True,
                "frozen_processor_use_percentiles": True,
                "frozen_processor_clip_outliers": True,
            },
            "action_chunk_rows": self.action_chunk_rows,
            "last_inference_evidence": (
                self.last_inference_evidence()
                if self._last_inference_evidence is not None
                else None
            ),
        }


__all__ = [
    "CHECKPOINT_REVISION",
    "EMBODIMENT_TAG",
    "GROOT_SOURCE_REVISION",
    "GrootN17DroidPolicyClient",
    "GrootN17DroidPolicySpec",
    "MODEL_ID",
    "droid_eef_9d",
    "CHECKPOINT_STATISTICS_SHA256",
    "CHECKPOINT_STATISTICS_GIT_BLOB_SHA1",
    "CHECKPOINT_STATISTICS_SOURCE",
    "DROID_EEF_POSITION_OBSERVED_MAX_M",
    "DROID_EEF_POSITION_OBSERVED_MIN_M",
    "EEF_FRAME_PROVENANCE_KEY",
    "EEF_FRAME_PROVENANCE_SCHEMA_VERSION",
    "validate_worker_identity_receipt",
]
