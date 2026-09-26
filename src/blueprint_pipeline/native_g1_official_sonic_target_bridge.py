"""Extract one measured SONIC target frame without letting SONIC step Isaac.

HumanoidArena's pinned SonicActionProvider combines inference and physics in
``get_action``. Blueprint owns the shared scene step, so this adapter invokes
the provider's reference/encoder/decoder path only. It rejects the upstream
default-pose fallback unless both ONNX sessions actually ran for this action.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .gear_sonic_joint_order_contract import (
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PROTOCOL_V4_LEFT_HAND_JOINT_NAMES,
    PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES,
)
from .native_g1_humanoidarena_interface import (
    CANONICAL_BODY_JOINT_NAMES_29,
    parse_semantic_v3_action,
)


PINNED_ACTION_PROVIDER_SHA256 = "701b75f0effec0c28c215a813a10d70db0a33dbf01000c7f1c142d6a8a60568c"
OFFICIAL_HAND_ORDER = (
    "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_pinned_sonic_source(path: str | Path) -> None:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("g1_sonic_source_unavailable")
    digest = _sha256_file(source)
    if digest != PINNED_ACTION_PROVIDER_SHA256:
        raise ValueError("g1_sonic_source_revision_mismatch")


def _require_artifact(path: str | Path, digest: str) -> None:
    artifact = Path(path)
    if (
        artifact.is_symlink()
        or not artifact.is_file()
        or not digest.startswith("sha256:")
        or len(digest) != 71
        or _sha256_file(artifact) != digest.removeprefix("sha256:")
    ):
        raise ValueError("g1_sonic_model_artifact_identity_mismatch")


class _CountingSession:
    def __init__(self, session: Any) -> None:
        if session is None or not callable(getattr(session, "run", None)):
            raise ValueError("g1_sonic_onnx_session_missing")
        self.session = session
        self.calls = 0

    def run(self, *args: Any, **kwargs: Any) -> Any:
        result = self.session.run(*args, **kwargs)
        self.calls += 1
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session, name)


class _CountingTimings(list[float]):
    def __init__(self, values: list[float]) -> None:
        super().__init__(values)
        self.appends = 0

    def append(self, value: float) -> None:
        self.appends += 1
        super().append(value)


class WxyzRootDataView:
    """Show SONIC WXYZ root state from native Beta2 XYZW readbacks."""

    def __init__(self, native_data: Any) -> None:
        self._native_data = native_data

    @property
    def root_state_w(self) -> Any:
        native = self._native_data.root_state_w
        if len(native.shape) == 2 and native.shape[1] >= 7:
            converted = native.clone()
            converted[:, 3] = native[:, 6]
            converted[:, 4:7] = native[:, 3:6]
            return converted
        # Some native startup readbacks expose an incomplete combined state.
        # Reconstruct the same world-frame 13-vector from its named tensors;
        # never synthesize a pose or silently replace missing velocity data.
        try:
            position = self._native_data.root_pos_w
            quaternion = self._native_data.root_quat_w
            linear = self._native_data.root_lin_vel_w
            angular = self._native_data.root_ang_vel_w
            batch = position.shape[0]
            if (
                batch < 1
                or position.shape != (batch, 3)
                or quaternion.shape != (batch, 4)
                or linear.shape != (batch, 3)
                or angular.shape != (batch, 3)
                or any(value.device != position.device or value.dtype != position.dtype
                       for value in (quaternion, linear, angular))
            ):
                raise ValueError("g1_sonic_native_root_state_invalid")
            return torch.cat((
                position, quaternion[:, [3, 0, 1, 2]], linear, angular,
            ), dim=1)
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("g1_sonic_native_root_state_invalid") from exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native_data, name)


class _RobotView:
    def __init__(self, robot: Any) -> None:
        self._robot = robot
        self.data = WxyzRootDataView(robot.data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)


class _SceneView:
    def __init__(self, scene: Any) -> None:
        self._scene = scene
        self._robot = _RobotView(scene["robot"])

    def __getitem__(self, key: str) -> Any:
        return self._robot if key == "robot" else self._scene[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scene, name)


class SonicWxyzEnvironmentView:
    def __init__(self, native_environment: Any) -> None:
        self._native_environment = native_environment
        self.scene = _SceneView(native_environment.scene)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native_environment, name)


class NativeG1OfficialSonicTargetBridge:
    def __init__(
        self,
        *,
        provider: Any,
        source_path: str | Path,
        encoder_sha256: str,
        decoder_sha256: str,
        joint_limits: Mapping[str, tuple[float, float] | list[float]],
    ) -> None:
        require_pinned_sonic_source(source_path)
        if (
            provider.__class__.__name__ != "SonicActionProvider"
            or inspect.getsourcefile(provider.__class__) is None
            or Path(inspect.getsourcefile(provider.__class__)).resolve() != Path(source_path).resolve()
            or not isinstance(getattr(provider, "env", None), SonicWxyzEnvironmentView)
            or not getattr(provider, "_sonic_joint29_mode", False)
            or getattr(provider, "_use_vla_latent64", True)
            or getattr(provider, "_dex3_dds", None) is not None
            or not callable(getattr(provider, "_apply_lerobot_semantic_action", None))
            or not callable(getattr(provider, "_run_gear_sonic", None))
        ):
            raise ValueError("g1_sonic_provider_contract_invalid")
        if set(joint_limits) != set(PROTOCOL_V4_FULL_JOINT_ORDER):
            raise ValueError("g1_sonic_joint_limits_invalid")
        _require_artifact(provider.encoder_path, encoder_sha256)
        _require_artifact(provider.decoder_path, decoder_sha256)
        self.provider = provider
        self.limits = joint_limits
        self.encoder = _CountingSession(provider._encoder)
        self.decoder = _CountingSession(provider._decoder)
        if not isinstance(provider._perf_encoder_ms, list) or not isinstance(provider._perf_decoder_ms, list):
            raise ValueError("g1_sonic_controller_timing_contract_invalid")
        self.encoder_timings = _CountingTimings(provider._perf_encoder_ms)
        self.decoder_timings = _CountingTimings(provider._perf_decoder_ms)
        provider._encoder = self.encoder
        provider._decoder = self.decoder
        provider._perf_encoder_ms = self.encoder_timings
        provider._perf_decoder_ms = self.decoder_timings

    def targets_for_action(self, action: list[float]) -> dict[str, float]:
        parse_semantic_v3_action(action)
        before = (self.encoder.calls, self.decoder.calls)
        before_timings = (self.encoder_timings.appends, self.decoder_timings.appends)
        self.provider._apply_lerobot_semantic_action(np.asarray(action, dtype=np.float32))
        if not (
            self.provider._smpl_data_valid
            and self.provider._latest_consumed_new_this_step
        ):
            raise RuntimeError("g1_sonic_reference_not_consumed")
        self.provider._latest_decoder_target = np.full(29, np.nan, dtype=np.float32)
        self.provider._latest_decoder_raw_action = np.full(29, np.nan, dtype=np.float32)
        body = np.asarray(self.provider._run_gear_sonic(), dtype=np.float64)
        measured_target = np.asarray(self.provider._latest_decoder_target, dtype=np.float64)
        measured_raw = np.asarray(self.provider._latest_decoder_raw_action, dtype=np.float64)
        if (
            body.shape != (29,)
            or not np.isfinite(body).all()
            or measured_target.shape != (29,)
            or measured_raw.shape != (29,)
            or not np.isfinite(measured_target).all()
            or not np.isfinite(measured_raw).all()
            or not np.allclose(body, measured_target, rtol=0.0, atol=1e-6)
            or (self.encoder.calls, self.decoder.calls)
            != (before[0] + 1, before[1] + 1)
            or (self.encoder_timings.appends, self.decoder_timings.appends)
            != (before_timings[0] + 1, before_timings[1] + 1)
        ):
            raise RuntimeError("g1_sonic_controller_inference_unmeasured")
        left = np.asarray(self.provider._left_hand_target, dtype=np.float64)
        right = np.asarray(self.provider._right_hand_target, dtype=np.float64)
        if left.shape != (7,) or right.shape != (7,):
            raise RuntimeError("g1_sonic_hand_targets_invalid")
        targets = dict(zip(CANONICAL_BODY_JOINT_NAMES_29, body.tolist(), strict=True))
        for side, values, expected in (
            ("left", left, PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
            ("right", right, PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        ):
            names = [f"{side}_hand_{suffix}_joint" for suffix in OFFICIAL_HAND_ORDER]
            if set(names) != set(expected):
                raise AssertionError("g1_sonic_hand_joint_order_drift")
            targets.update(zip(names, values.tolist(), strict=True))
        for name in PROTOCOL_V4_FULL_JOINT_ORDER:
            value = targets[name]
            lower, upper = self.limits[name]
            if not all(math.isfinite(float(item)) for item in (value, lower, upper)) or not lower <= value <= upper:
                raise ValueError("g1_sonic_controller_target_out_of_limits")
        return targets
